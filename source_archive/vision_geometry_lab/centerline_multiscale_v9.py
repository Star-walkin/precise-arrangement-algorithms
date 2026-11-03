"""Paired V9 global-context/local-detail experiment (offline CPU only).

Both arms use the same original full-image RGB+human-free-seed global input,
the same nine seed-defined windows, architecture, initialization, sampler and
supervision.  ``native_detail`` uses native local RGB; ``lowpass_control`` uses
the same RGB after a V6 512x1024 fit-content down/up round trip.  No target or
quality geometry may select, remove or move a window.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch import nn
import torch.nn.functional as F

from .centerline_model_v5 import (
    _canonical_json_sha256,
    _file_sha256,
    _local_points,
    _resample_polyline,
    _write_json,
    inspect_manifest,
    masked_two_head_loss,
)
from .centerline_multiscale_contract_v9 import (
    CONTRACT_VERSION,
    GLOBAL_HW,
    LOCAL_MODEL_SIZE,
    LOWPASS_HW,
    MODES,
    OUTPUT_SCALE_XY,
    PIXEL_CENTER_MAPPING,
    TILE_NATIVE_SIZE,
    array_sha256,
    blend_weight_map,
    canonical_json_sha256,
    rebuild_input_contract,
    tile_in_frame_mask,
)


EXPERIMENT_VERSION = 9
CHECKPOINT_FORMAT = "centerline_multiscale_v9_seed_conditioned_pair"
MODEL_VERSION = 1
RASTER_VERSION = 1
FULL_IMAGE_HW = (3000, 4096)
FULL_ROI_XYXY = [0, 0, 4096, 3000]
TRAIN_COUNT = 10
HOLDOUT_COUNT = 8
DEFAULT_OUTPUT = Path(__file__).parent / "outputs/centerline_v9"
OUTPUT_HEADS = ("body_logits", "center_logits")
INPUT_SEMANTICS = (
    "original full RGB plus reviewed human free-seed at global scale",
    "local RGB arm plus the identical original-seed local heatmap",
)


def _state_dict_sha256(state_dict):
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name].detach().cpu().contiguous().numpy()
        header = json.dumps({"name": name, "shape": list(value.shape), "dtype": str(value.dtype)},
                            sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(value.tobytes())
    return digest.hexdigest()


def _load_rgb_uint8(record, *, verify_hash=True):
    path = Path(record["image_path"])
    if verify_hash and _file_sha256(path) != record["image_sha256"]:
        raise ValueError(f"Image source hash mismatch: {record['sample_id']}")
    with Image.open(path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    expected = tuple(record["image_shape_hw"])
    if rgb.shape[:2] != expected:
        raise ValueError(f"Image shape mismatch: {record['sample_id']}")
    return np.ascontiguousarray(rgb)


def _target_polylines(record):
    parsed = dict(record["parsed_polylines"])
    parsed.update(record.get("parsed_supervision_polylines", {}))
    return parsed


def _polyline_image(size_wh, segments, width, blur=0.0, *, normalize=False):
    image = Image.new("L", tuple(map(int, size_wh)), 0)
    drawer = ImageDraw.Draw(image)
    for segment in segments:
        points = [tuple(map(float, point)) for point in segment]
        if len(points) >= 2:
            drawer.line(points, fill=255, width=max(1, int(width)), joint="curve")
    if blur > 0:
        image = image.filter(ImageFilter.GaussianBlur(float(blur)))
    result = np.asarray(image, dtype=np.float32)/255.0
    peak = float(result.max(initial=0.0))
    if normalize and peak > 0:
        result = result/peak
    return result


def rasterize_native_supervision(record):
    """Native-pixel targets; target curves are pre-clipped before blur/fill."""
    if record["label"]["status"] not in ("clear", "partial"):
        raise ValueError("V9 target rasterization requires a reviewed clear/partial record")
    x0, y0, x1, y1 = map(int, record["roi_xyxy"])
    width, height = x1-x0, y1-y0
    parsed = _target_polylines(record)
    local = {name: _local_points(points, record["roi_xyxy"])
             for name, points in parsed.items()}
    center_segments = [local["centerline"]] if len(local["centerline"]) >= 2 else []
    center = _polyline_image((width, height), center_segments, 1, 2.0, normalize=True)
    center_valid = _polyline_image((width, height), center_segments, 64, 0.0)
    body = np.zeros((height, width), dtype=np.float32)
    body_valid = np.zeros_like(body)
    if len(local["edge_flange"]) >= 2 and len(local["edge_surface"]) >= 2:
        first = _resample_polyline(local["edge_flange"])
        second = _resample_polyline(local["edge_surface"])
        polygon = [tuple(point) for point in np.vstack([first, second[::-1]])]
        body_image = Image.new("L", (width, height), 0)
        ImageDraw.Draw(body_image).polygon(polygon, fill=255)
        body = np.asarray(body_image, dtype=np.float32)/255.0
        # 33 native pixels on each side => odd 67-pixel max-filter kernel.
        body_valid = np.asarray(body_image.filter(ImageFilter.MaxFilter(67)),
                                dtype=np.float32)/255.0
    exclusion_image = Image.new("L", (width, height), 0)
    drawer = ImageDraw.Draw(exclusion_image)
    for polygon in (*record["parsed_ignore_polygons"],
                    *record.get("parsed_quality_exclude_polygons", [])):
        drawer.polygon(_local_points(polygon, record["roi_xyxy"]), fill=255)
    exclusion = np.asarray(exclusion_image, dtype=np.float32)/255.0
    return dict(body_target=body, center_target=center, body_valid=body_valid,
                center_valid=center_valid, exclusion=exclusion,
                target_source=("parsed_supervision_polylines"
                               if "parsed_supervision_polylines" in record
                               else "legacy parsed_polylines fallback"),
                center_gaussian_sigma_native_px=2.0,
                center_valid_corridor_width_native_px=64,
                body_valid_dilation_each_side_native_px=33)


def _extract_padded(array, xyxy):
    left, top, right, bottom = map(int, xyxy)
    output_shape = (bottom-top, right-left)+array.shape[2:]
    result = np.zeros(output_shape, dtype=array.dtype)
    sx0, sy0, sx1, sy1 = max(0, left), max(0, top), min(array.shape[1], right), min(array.shape[0], bottom)
    if sx1 > sx0 and sy1 > sy0:
        dx0, dy0 = sx0-left, sy0-top
        result[dy0:dy0+sy1-sy0, dx0:dx0+sx1-sx0] = array[sy0:sy1, sx0:sx1]
    return result


def _resize_plane(array, size, resample):
    image = Image.fromarray(np.rint(np.clip(array, 0, 1)*255).astype(np.uint8), "L")
    return np.asarray(image.resize((size, size), resample=resample), dtype=np.float32)/255.0


def _local_supervision(native, tile_layout, native_hw):
    result = []
    for tile in tile_layout["tiles"]:
        xyxy = tile["native_xyxy"]
        body = _resize_plane(_extract_padded(native["body_target"], xyxy),
                             LOCAL_MODEL_SIZE, Image.Resampling.BILINEAR)
        center = _resize_plane(_extract_padded(native["center_target"], xyxy),
                               LOCAL_MODEL_SIZE, Image.Resampling.BILINEAR)
        body_valid = _resize_plane(_extract_padded(native["body_valid"], xyxy),
                                   LOCAL_MODEL_SIZE, Image.Resampling.NEAREST)
        center_valid = _resize_plane(_extract_padded(native["center_valid"], xyxy),
                                     LOCAL_MODEL_SIZE, Image.Resampling.NEAREST)
        exclusion = _extract_padded(native["exclusion"], xyxy)
        excluded = F.max_pool2d(torch.from_numpy(exclusion)[None, None], 2, 2)
        excluded = F.max_pool2d(excluded, 3, 1, 1)[0, 0].numpy() > 0
        in_frame = tile_in_frame_mask(tile, native_hw) > 0
        keep = (~excluded) & in_frame
        body_valid = (body_valid > 0) & keep
        center_valid = (center_valid > 0) & keep
        result.append(dict(
            body_target=torch.from_numpy(body[None]).float(),
            center_target=torch.from_numpy(center[None]).float(),
            body_valid_mask=torch.from_numpy(body_valid[None]).float(),
            center_valid_mask=torch.from_numpy(center_valid[None]).float(),
            in_frame_mask=torch.from_numpy(in_frame[None]),
        ))
    return result


def prepare_record(record, mode, *, include_supervision=True, verify_image_hash=True):
    """Build fixed inputs first; optional targets cannot affect layout or inputs."""
    started = time.perf_counter()
    rgb = _load_rgb_uint8(record, verify_hash=verify_image_hash)
    seed = record["parsed_polylines"]["free_seed"]
    contract = rebuild_input_contract(rgb, seed, mode)
    prepared = dict(
        sample_id=record["sample_id"], group_id=record["group_id"], split=record["split"],
        global_input=torch.from_numpy(contract["global_input"]).float(),
        local_inputs=torch.from_numpy(contract["local_inputs"]).float(),
        input_contract=contract["input_contract"], tile_layout=contract["tile_layout"],
        global_geometry=contract["global_geometry"], lowpass_geometry=contract["lowpass_geometry"],
        preparation_seconds=None,
    )
    if include_supervision:
        native = rasterize_native_supervision(record)
        prepared["supervision"] = _local_supervision(native, contract["tile_layout"], rgb.shape[:2])
        prepared["supervision_metadata"] = {
            key: native[key] for key in (
                "target_source", "center_gaussian_sigma_native_px",
                "center_valid_corridor_width_native_px", "body_valid_dilation_each_side_native_px")}
    prepared["preparation_seconds"] = time.perf_counter()-started
    return prepared


class _ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, output_channels), nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, output_channels), nn.SiLU(inplace=True))

    def forward(self, value):
        return self.layers(value)


class GlobalDescriptorEncoder(nn.Module):
    """Eight-channel whole-frame descriptor; not evidence of global understanding."""
    def __init__(self, base_channels=8, descriptor_channels=8):
        super().__init__()
        base = int(base_channels)
        self.layers = nn.Sequential(
            _ConvBlock(4, base), nn.MaxPool2d(2),
            _ConvBlock(base, base*2), nn.MaxPool2d(2),
            nn.Conv2d(base*2, descriptor_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, descriptor_channels), nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1))

    def forward(self, value):
        if value.ndim != 4 or value.shape[1] != 4:
            raise ValueError("global input must be N x 4 x 256 x 384")
        return self.layers(value).flatten(1)


class LocalFusionUNet(nn.Module):
    def __init__(self, base_channels=8, descriptor_channels=8):
        super().__init__()
        base = int(base_channels)
        self.input_channels = 4+int(descriptor_channels)
        self.encoder1 = _ConvBlock(self.input_channels, base)
        self.encoder2 = _ConvBlock(base, base*2)
        self.bridge = _ConvBlock(base*2, base*4)
        self.decoder2 = _ConvBlock(base*4+base*2, base*2)
        self.decoder1 = _ConvBlock(base*2+base, base)
        self.body_head = nn.Conv2d(base, 1, 1)
        self.center_head = nn.Conv2d(base, 1, 1)

    def forward(self, local_input, descriptor):
        if local_input.ndim != 4 or local_input.shape[1] != 4:
            raise ValueError("local input must be N x 4 x 384 x 384")
        if descriptor.ndim != 2 or descriptor.shape[0] != local_input.shape[0]:
            raise ValueError("descriptor batch mismatch")
        expanded = descriptor[:, :, None, None].expand(
            -1, -1, local_input.shape[-2], local_input.shape[-1])
        value = torch.cat([local_input, expanded], dim=1)
        first = self.encoder1(value)
        second = self.encoder2(F.max_pool2d(first, 2, ceil_mode=True))
        bridge = self.bridge(F.max_pool2d(second, 2, ceil_mode=True))
        up2 = F.interpolate(bridge, size=second.shape[-2:], mode="bilinear", align_corners=False)
        decoded2 = self.decoder2(torch.cat([up2, second], dim=1))
        up1 = F.interpolate(decoded2, size=first.shape[-2:], mode="bilinear", align_corners=False)
        decoded1 = self.decoder1(torch.cat([up1, first], dim=1))
        return {"body_logits": self.body_head(decoded1),
                "center_logits": self.center_head(decoded1)}


class MultiScaleCenterlineModel(nn.Module):
    def __init__(self, base_channels=8, descriptor_channels=8):
        super().__init__()
        self.base_channels = int(base_channels)
        self.descriptor_channels = int(descriptor_channels)
        if self.base_channels < 2 or self.descriptor_channels != 8:
            raise ValueError("base_channels must be >=2 and V9 descriptor_channels must equal 8")
        self.global_encoder = GlobalDescriptorEncoder(self.base_channels, self.descriptor_channels)
        self.local_head = LocalFusionUNet(self.base_channels, self.descriptor_channels)

    def encode_global(self, global_input):
        return self.global_encoder(global_input)

    def decode_local(self, local_input, descriptor):
        return self.local_head(local_input, descriptor)

    def forward(self, global_input, local_input):
        return self.decode_local(local_input, self.encode_global(global_input))


def architecture_metadata(base_channels=8):
    return dict(
        name="global_descriptor_local_unet", version=MODEL_VERSION,
        base_channels=int(base_channels), global_input_channels=4, local_input_channels=4,
        global_descriptor_channels=8, local_fusion_channels=12, local_pool_depth=2,
        pretrained_weights=None,
        global_descriptor_claim=("adaptive-average-pooled whole-frame descriptor; "
                                 "not evidence of learned global understanding"),
        output_heads=list(OUTPUT_HEADS),
    )


def build_model(base_channels=8):
    return MultiScaleCenterlineModel(int(base_channels), 8)


def build_training_sequence(records, steps, batch_size, seed):
    """Mode-independent shuffled epochs over every record x every fixed tile."""
    if not records or int(steps) <= 0 or int(batch_size) <= 0:
        raise ValueError("records, steps and batch_size must be positive")
    base = [(index, tile) for index in range(len(records)) for tile in range(9)]
    generator = np.random.default_rng(int(seed))
    needed = int(steps)*int(batch_size)
    flattened = []
    while len(flattened) < needed:
        order = generator.permutation(len(base))
        flattened.extend(base[int(index)] for index in order)
    flattened = flattened[:needed]
    batches = []
    for start in range(0, needed, int(batch_size)):
        batches.append([dict(sample_id=records[index]["sample_id"], record_index=index,
                             tile_index=tile)
                        for index, tile in flattened[start:start+int(batch_size)]])
    return batches


class PreparedRecordCache:
    def __init__(self, records, mode, *, cache_samples=False):
        if mode not in MODES:
            raise ValueError(f"Unsupported mode {mode}")
        self.records = list(records)
        self.mode = mode
        self.cache_samples = bool(cache_samples)
        self._cache = {}

    def get(self, index):
        index = int(index)
        if index in self._cache:
            return self._cache[index]
        prepared = prepare_record(self.records[index], self.mode, include_supervision=True)
        if self.cache_samples:
            self._cache[index] = prepared
        if self.cache_samples:
            print(json.dumps({"event": "prepared_record", "mode": self.mode,
                              "sample_id": self.records[index]["sample_id"],
                              "seconds": prepared["preparation_seconds"]}), flush=True)
        return prepared

    def cache_bytes(self):
        total = 0
        for prepared in self._cache.values():
            total += prepared["global_input"].numel()*prepared["global_input"].element_size()
            total += prepared["local_inputs"].numel()*prepared["local_inputs"].element_size()
            for tile in prepared.get("supervision", []):
                for key in ("body_target", "center_target", "body_valid_mask", "center_valid_mask"):
                    total += tile[key].numel()*tile[key].element_size()
        return int(total)


def _batch_from_sequence(cache, items):
    global_inputs, local_inputs = [], []
    supervision = {key: [] for key in
                   ("body_target", "center_target", "body_valid_mask", "center_valid_mask")}
    zero_items = 0
    for item in items:
        prepared = cache.get(item["record_index"])
        tile_index = item["tile_index"]
        global_inputs.append(prepared["global_input"])
        local_inputs.append(prepared["local_inputs"][tile_index])
        tile = prepared["supervision"][tile_index]
        for key in supervision:
            supervision[key].append(tile[key])
        zero_items += int(float(tile["body_valid_mask"].sum()+tile["center_valid_mask"].sum()) == 0)
    batch = dict(global_input=torch.stack(global_inputs), local_input=torch.stack(local_inputs),
                 **{key: torch.stack(value) for key, value in supervision.items()})
    return batch, zero_items


def _loss(model, batch, center_weight=1.0):
    outputs = model(batch["global_input"], batch["local_input"])
    return outputs, masked_two_head_loss(
        outputs, batch["body_target"], batch["center_target"],
        body_valid_mask=batch["body_valid_mask"],
        center_valid_mask=batch["center_valid_mask"], center_weight=float(center_weight))


def _supervision_summary(cache):
    summary = dict(total_tiles=0, zero_supervision_tiles=0, body_valid_tiles=0,
                   center_valid_tiles=0, body_positive_tiles=0, center_positive_tiles=0)
    for index in range(len(cache.records)):
        prepared = cache.get(index)
        for tile in prepared["supervision"]:
            body_valid = tile["body_valid_mask"] > 0
            center_valid = tile["center_valid_mask"] > 0
            summary["total_tiles"] += 1
            summary["body_valid_tiles"] += int(body_valid.any())
            summary["center_valid_tiles"] += int(center_valid.any())
            summary["body_positive_tiles"] += int(((tile["body_target"] > 0)*body_valid).any())
            summary["center_positive_tiles"] += int(((tile["center_target"] > 0)*center_valid).any())
            summary["zero_supervision_tiles"] += int(not body_valid.any() and not center_valid.any())
    summary["body_head_supervised"] = summary["body_positive_tiles"] > 0
    summary["center_head_supervised"] = summary["center_positive_tiles"] > 0
    return summary


def _pixel_diagnostic(model, cache, center_weight, note):
    totals = dict(body_loss_sum=0.0, center_loss_sum=0.0, body_valid=0.0,
                  center_valid=0.0, zero_tiles=0)
    model.eval()
    with torch.inference_mode():
        for record_index in range(len(cache.records)):
            prepared = cache.get(record_index)
            descriptor = model.encode_global(prepared["global_input"].unsqueeze(0))
            for tile_index, tile in enumerate(prepared["supervision"]):
                outputs = model.decode_local(prepared["local_inputs"][tile_index:tile_index+1], descriptor)
                losses = masked_two_head_loss(
                    outputs, tile["body_target"][None], tile["center_target"][None],
                    body_valid_mask=tile["body_valid_mask"][None],
                    center_valid_mask=tile["center_valid_mask"][None], center_weight=float(center_weight))
                bw, cw = float(losses["body_valid_weight"]), float(losses["center_valid_weight"])
                totals["body_loss_sum"] += float(losses["body_loss"])*bw
                totals["center_loss_sum"] += float(losses["center_loss"])*cw
                totals["body_valid"] += bw
                totals["center_valid"] += cw
                totals["zero_tiles"] += int(bw+cw == 0)
    body_loss = totals["body_loss_sum"]/totals["body_valid"] if totals["body_valid"] else None
    center_loss = totals["center_loss_sum"]/totals["center_valid"] if totals["center_valid"] else None
    combined = (None if body_loss is None and center_loss is None else
                (0.0 if body_loss is None else body_loss)+float(center_weight)*(0.0 if center_loss is None else center_loss))
    return dict(masked_two_head_loss=combined, body_loss=body_loss, center_loss=center_loss,
                body_valid_pixels=int(totals["body_valid"]),
                center_valid_pixels=int(totals["center_valid"]),
                zero_supervision_tiles=int(totals["zero_tiles"]), note=note,
                path_identity_or_accuracy_established=False)


def _validate_fixed_manifest(inspected):
    train_records = inspected["usable"]["train"]
    holdout_records = inspected["usable"]["holdout"]
    if len(train_records) != TRAIN_COUNT or len(holdout_records) != HOLDOUT_COUNT:
        raise ValueError(f"Frozen V9 manifest requires {TRAIN_COUNT} train and {HOLDOUT_COUNT} holdout records")
    for record in train_records+holdout_records:
        if record["roi_xyxy"] != FULL_ROI_XYXY or tuple(record["image_shape_hw"]) != FULL_IMAGE_HW:
            raise ValueError("V9 fixed experiment requires full 4096x3000 ROI for every record")
        if record["sample_id"].startswith("C06__"):
            raise ValueError("C06 is outside the fixed V9 train/holdout manifest")
        if len(record["parsed_polylines"]["free_seed"]) < 2:
            raise ValueError("V9 fixed records require reviewed human free_seed")
    return train_records, holdout_records


def _ensure_fresh_output(path):
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"checkpoint.pt").exists() or (output/"report.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing V9 run: {output}")
    return output


def train_manifest(args):
    started = time.perf_counter()
    torch.set_num_threads(int(args.threads))
    if args.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    output = _ensure_fresh_output(args.output_dir)
    inspected = inspect_manifest(args.manifest)
    train_records, holdout_records = _validate_fixed_manifest(inspected)
    sequence = build_training_sequence(train_records, args.steps, args.batch_size, args.seed)
    sequence_sha = canonical_json_sha256(sequence)
    train_cache = PreparedRecordCache(
        train_records, args.mode, cache_samples=bool(args.cache_samples))
    holdout_cache = PreparedRecordCache(
        holdout_records, args.mode, cache_samples=bool(args.cache_samples))
    supervision = _supervision_summary(train_cache)
    if not supervision["body_head_supervised"] or not supervision["center_head_supervised"]:
        raise ValueError("Frozen V9 comparison requires positive body and center supervision")

    # Reset immediately before model construction so preprocessing cannot alter initialization.
    torch.manual_seed(int(args.seed))
    model = build_model(args.base_channels).cpu()
    initial_state_sha = _state_dict_sha256(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    history = []
    zero_items_total = 0
    pure_zero_batches_total = 0
    loop_started = time.perf_counter()
    model.train()
    for step, items in enumerate(sequence, 1):
        batch, zero_items = _batch_from_sequence(train_cache, items)
        zero_items_total += zero_items
        pure_zero_batch = zero_items == len(items)
        pure_zero_batches_total += int(pure_zero_batch)
        optimizer.zero_grad(set_to_none=True)
        forward_started = time.perf_counter()
        _, losses = _loss(model, batch, args.center_weight)
        forward_seconds = time.perf_counter()-forward_started
        losses["loss"].backward()
        optimizer.step()
        history.append(dict(step=step, loss=float(losses["loss"].detach()),
                            body_loss=float(losses["body_loss"].detach()),
                            center_loss=float(losses["center_loss"].detach()),
                            body_valid_pixels=int(losses["body_valid_weight"]),
                            center_valid_pixels=int(losses["center_valid_weight"]),
                            zero_supervision_items=zero_items,
                            pure_zero_supervision_batch=pure_zero_batch,
                            sample_tiles=[dict(sample_id=item["sample_id"],
                                               tile_index=item["tile_index"]) for item in items],
                            forward_seconds=forward_seconds))
        if step == 1 or step == len(sequence) or step % max(1, len(sequence)//10) == 0:
            print(json.dumps({"event": "training_progress", "mode": args.mode,
                              "step": step, "steps": len(sequence),
                              "loss": history[-1]["loss"]}), flush=True)
    loop_seconds = time.perf_counter()-loop_started
    train_diagnostic = _pixel_diagnostic(
        model, train_cache, args.center_weight,
        "post-training train tiles; fitting diagnostic, not independent accuracy")
    holdout_diagnostic = _pixel_diagnostic(
        model, holdout_cache, args.center_weight,
        "group-held-out tiles evaluated once after training; no early stopping or tuning")

    module_path = Path(__file__).resolve()
    contract_path = module_path.with_name("centerline_multiscale_contract_v9.py")
    architecture = architecture_metadata(args.base_channels)
    checkpoint = dict(
        format=CHECKPOINT_FORMAT, schema_version=1, experiment_version=EXPERIMENT_VERSION,
        model_version=MODEL_VERSION, raster_version=RASTER_VERSION, mode=args.mode,
        model_config=dict(base_channels=int(args.base_channels), descriptor_channels=8,
                          global_hw=list(GLOBAL_HW), local_hw=[LOCAL_MODEL_SIZE, LOCAL_MODEL_SIZE],
                          output_heads=list(OUTPUT_HEADS)),
        architecture=architecture, state_dict=model.state_dict(),
        initial_head_state_sha256=initial_state_sha,
        initial_model_state_sha256=initial_state_sha,
        final_state_sha256=_state_dict_sha256(model.state_dict()),
        input_semantics=list(INPUT_SEMANTICS),
        training_source="reviewed_manifest_annotations",
        training_manifest_sha256=inspected["manifest_sha256"],
        training_sample_ids=[record["sample_id"] for record in train_records],
        train_group_ids=sorted({record["group_id"] for record in train_records}),
        training_annotation_sha256_by_sample={record["sample_id"]: record["annotation_sha256"]
                                              for record in train_records},
        train_annotation_sha256_by_sample={record["sample_id"]: record["annotation_sha256"]
                                           for record in train_records},
        train_quality_sha256_by_sample={record["sample_id"]: record["quality_sha256"]
                                        for record in train_records},
        sampler_sequence_sha256=sequence_sha,
        body_head_supervised=supervision["body_head_supervised"],
        center_head_supervised=supervision["center_head_supervised"],
        source_sha256={module_path.name: _file_sha256(module_path),
                       contract_path.name: _file_sha256(contract_path)},
        field_validation_performed=False,
    )
    checkpoint_path = output/"checkpoint.pt"
    temporary = output/"checkpoint.pt.tmp"
    torch.save(checkpoint, temporary)
    os.replace(temporary, checkpoint_path)
    report = dict(
        schema_version=1, experiment_version=EXPERIMENT_VERSION,
        status="manifest_training_complete_unvalidated", mode=args.mode,
        manifest=str(Path(args.manifest).resolve()), manifest_sha256=inspected["manifest_sha256"],
        architecture=architecture,
        train_group_ids=checkpoint["train_group_ids"],
        train_annotation_sha256_by_sample=checkpoint["train_annotation_sha256_by_sample"],
        train_quality_sha256_by_sample=checkpoint["train_quality_sha256_by_sample"],
        paired_design=dict(
            changed_variable_only="local RGB native detail versus fixed lowpass roundtrip",
            global_input_identical_across_modes=True, local_seed_identical_across_modes=True,
            tile_layout_identical_across_modes=True, supervision_identical_across_modes=True,
            initialization_reset_immediately_before_model=True,
            sampler_independent_of_model_rng=True, augmentation=None,
            holdout_used_for_training=False, holdout_used_for_early_stopping=False,
        ),
        data=dict(
            train_sample_ids=[record["sample_id"] for record in train_records],
            holdout_sample_ids=[record["sample_id"] for record in holdout_records],
            train_group_ids=sorted({record["group_id"] for record in train_records}),
            holdout_group_ids=sorted({record["group_id"] for record in holdout_records}),
            training_annotation_sha256_by_sample=checkpoint["training_annotation_sha256_by_sample"],
            train_quality_sha256_by_sample=checkpoint["train_quality_sha256_by_sample"],
            holdout_annotation_sha256_by_sample={record["sample_id"]: record["annotation_sha256"]
                                                 for record in holdout_records},
            holdout_quality_sha256_by_sample={record["sample_id"]: record["quality_sha256"]
                                               for record in holdout_records},
            all_roi_xyxy=FULL_ROI_XYXY, all_native_hw=list(FULL_IMAGE_HW),
            c06_included=False, skipped=inspected["skipped"],
        ),
        supervision=supervision,
        training=dict(
            device="cpu", steps_completed=int(args.steps), batch_size=int(args.batch_size),
            base_channels=int(args.base_channels), learning_rate=float(args.learning_rate),
            center_weight=float(args.center_weight), seed=int(args.seed), threads=int(args.threads),
            cache_samples=bool(args.cache_samples),
            initial_head_state_sha256=initial_state_sha,
            sampler_sequence_sha256=sequence_sha, sampler_sequence=sequence,
            zero_supervision_items_sampled=int(zero_items_total),
            pure_zero_supervision_batches=int(pure_zero_batches_total),
            zero_supervision_policy=(
                "kept in fixed sampler and never resampled; a pure-zero batch has zero current "
                "masked gradient, while the scheduled Adam step may still apply carried optimizer momentum"),
            loop_seconds=loop_seconds, cache_bytes=train_cache.cache_bytes()+holdout_cache.cache_bytes(),
            loss_history=history,
        ),
        train_pixel_diagnostic=train_diagnostic, holdout_pixel_diagnostic=holdout_diagnostic,
        artifacts=dict(checkpoint=str(checkpoint_path), checkpoint_sha256=_file_sha256(checkpoint_path)),
        elapsed_seconds_before_report=time.perf_counter()-started,
        claims=dict(seed_assisted=True, fully_automatic=False, field_accuracy_established=False,
                    production_ready=False, hardware_access=False, force_command=None),
    )
    _write_json(output/"report.json", report)
    print(json.dumps({"report": str(output/"report.json"), "checkpoint": str(checkpoint_path),
                      "mode": args.mode, "status": report["status"]}), flush=True)
    return report


def load_model(checkpoint_path):
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT or checkpoint.get("experiment_version") != 9:
        raise ValueError("V9 requires its independent paired multiscale checkpoint format")
    if checkpoint.get("model_version") != MODEL_VERSION or checkpoint.get("raster_version") != RASTER_VERSION:
        raise ValueError("Unsupported V9 model/raster version")
    if checkpoint.get("mode") not in MODES:
        raise ValueError("V9 checkpoint mode is invalid")
    config = checkpoint.get("model_config", {})
    if (config.get("descriptor_channels") != 8 or config.get("global_hw") != list(GLOBAL_HW)
            or config.get("local_hw") != [LOCAL_MODEL_SIZE, LOCAL_MODEL_SIZE]
            or config.get("output_heads") != list(OUTPUT_HEADS)):
        raise ValueError("V9 checkpoint architecture/input contract mismatch")
    model = build_model(int(config.get("base_channels", 0))).cpu()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def _stitch_tiles(tile_probabilities, tile_layout, native_hw):
    height, width = map(int, native_hw)
    output_hw = (height//2, width//2)
    body_sum = np.zeros(output_hw, dtype=np.float64)
    center_sum = np.zeros(output_hw, dtype=np.float64)
    weight_sum = np.zeros(output_hw, dtype=np.float64)
    weight = blend_weight_map().astype(np.float64)
    if len(tile_probabilities) != 9:
        raise ValueError("V9 stitching requires exactly nine tile predictions")
    for tile, prediction in zip(tile_layout["tiles"], tile_probabilities):
        body, center = prediction
        if body.shape != (LOCAL_MODEL_SIZE, LOCAL_MODEL_SIZE) or center.shape != body.shape:
            raise ValueError("Unexpected local probability shape")
        valid = tile_in_frame_mask(tile, native_hw).astype(bool)
        left, top, _, _ = tile["native_xyxy"]
        gx0, gy0 = left//2, top//2
        gx1, gy1 = gx0+LOCAL_MODEL_SIZE, gy0+LOCAL_MODEL_SIZE
        out_x0, out_y0 = max(0, gx0), max(0, gy0)
        out_x1, out_y1 = min(output_hw[1], gx1), min(output_hw[0], gy1)
        if out_x1 <= out_x0 or out_y1 <= out_y0:
            continue
        sx0, sy0 = out_x0-gx0, out_y0-gy0
        sx1, sy1 = sx0+out_x1-out_x0, sy0+out_y1-out_y0
        local_weight = weight[sy0:sy1, sx0:sx1]*valid[sy0:sy1, sx0:sx1]
        body_sum[out_y0:out_y1, out_x0:out_x1] += body[sy0:sy1, sx0:sx1]*local_weight
        center_sum[out_y0:out_y1, out_x0:out_x1] += center[sy0:sy1, sx0:sx1]*local_weight
        weight_sum[out_y0:out_y1, out_x0:out_x1] += local_weight
    coverage = weight_sum > 0
    body = np.zeros(output_hw, dtype=np.float32)
    center = np.zeros(output_hw, dtype=np.float32)
    body[coverage] = (body_sum[coverage]/weight_sum[coverage]).astype(np.float32)
    center[coverage] = (center_sum[coverage]/weight_sum[coverage]).astype(np.float32)
    expected = rebuild_expected_coverage(tile_layout, native_hw)
    if not np.array_equal(coverage.astype(np.uint8), expected):
        raise AssertionError("Stitched coverage differs from the seed-only contract")
    if np.any(body[~coverage] != 0) or np.any(center[~coverage] != 0):
        raise AssertionError("Uncovered V9 probabilities must be exact zero")
    return body, center, coverage.astype(np.uint8)


def rebuild_expected_coverage(tile_layout, native_hw):
    # Local import keeps the explicit shared-contract dependency visible.
    from .centerline_multiscale_contract_v9 import expected_coverage_mask
    return expected_coverage_mask(tile_layout, native_hw)


def predict_record(model, checkpoint, record, *, tile_batch_size=1, verify_image_hash=True):
    total_started = time.perf_counter()
    read_started = time.perf_counter()
    rgb = _load_rgb_uint8(record, verify_hash=verify_image_hash)
    read_ms = (time.perf_counter()-read_started)*1000.0
    preprocess_started = time.perf_counter()
    contract = rebuild_input_contract(rgb, record["parsed_polylines"]["free_seed"], checkpoint["mode"])
    preprocess_ms = (time.perf_counter()-preprocess_started)*1000.0
    global_tensor = torch.from_numpy(contract["global_input"]).float().unsqueeze(0)
    local_tensor = torch.from_numpy(contract["local_inputs"]).float()
    global_started = time.perf_counter()
    with torch.inference_mode():
        descriptor = model.encode_global(global_tensor)
    global_ms = (time.perf_counter()-global_started)*1000.0
    tiles_started = time.perf_counter()
    predictions = []
    with torch.inference_mode():
        for start in range(0, 9, int(tile_batch_size)):
            local = local_tensor[start:start+int(tile_batch_size)]
            desc = descriptor.expand(len(local), -1)
            outputs = model.decode_local(local, desc)
            body = torch.sigmoid(outputs["body_logits"])[:, 0].cpu().numpy()
            center = torch.sigmoid(outputs["center_logits"])[:, 0].cpu().numpy()
            predictions.extend((body[index].astype(np.float32, copy=False),
                                center[index].astype(np.float32, copy=False))
                               for index in range(len(body)))
    tile_ms = (time.perf_counter()-tiles_started)*1000.0
    stitch_started = time.perf_counter()
    body, center, coverage = _stitch_tiles(predictions, contract["tile_layout"], rgb.shape[:2])
    stitch_ms = (time.perf_counter()-stitch_started)*1000.0
    timings = dict(image_read_and_hash_ms=read_ms, input_preprocess_ms=preprocess_ms,
                   global_encoder_ms=global_ms, nine_tile_head_ms=tile_ms,
                   weighted_stitch_ms=stitch_ms,
                   end_to_end_ms=(time.perf_counter()-total_started)*1000.0,
                   cache_used=False,
                   interpretation="CPU end-to-end for this export; not a real-time claim")
    return body, center, coverage, contract, timings


def _write_prediction(path, body, center, coverage):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path,
                        body_probability=body.astype(np.float32, copy=False),
                        center_probability=center.astype(np.float32, copy=False),
                        coverage_mask=coverage.astype(np.uint8, copy=False))
    return path


def infer_manifest(args):
    torch.set_num_threads(int(args.threads))
    model, checkpoint = load_model(args.checkpoint)
    inspected = inspect_manifest(args.manifest)
    records = [record for record in inspected["inference_records"]
               if args.split == "all" or record["split"] == args.split]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite V9 export: {output}")
    checkpoint_hash = _file_sha256(args.checkpoint)
    module_path = Path(__file__).resolve()
    contract_path = module_path.with_name("centerline_multiscale_contract_v9.py")
    training_report_path = Path(args.checkpoint).resolve().parent/"report.json"
    if not training_report_path.is_file():
        raise ValueError("V9 checkpoint must retain its training report beside it")
    training_report_hash = _file_sha256(training_report_path)
    exported = []
    for record in records:
        if record["roi_xyxy"] != FULL_ROI_XYXY or tuple(record["image_shape_hw"]) != FULL_IMAGE_HW:
            raise ValueError("V9 export requires full 4096x3000 ROI")
        body, center, coverage, contract, timings = predict_record(
            model, checkpoint, record, tile_batch_size=args.tile_batch_size)
        prediction_path = _write_prediction(
            output/"predictions"/f"{record['sample_id']}.npz", body, center, coverage)
        input_contract = contract["input_contract"]
        metadata = dict(
            schema_version=1, experiment_version=EXPERIMENT_VERSION,
            format=CHECKPOINT_FORMAT, status="unvalidated_model_inference",
            sample_id=record["sample_id"], group_id=record["group_id"], split=record["split"],
            image_path=str(Path(record["image_path"]).resolve()), image_sha256=record["image_sha256"],
            annotation_path=str(Path(record["annotation_path"]).resolve()),
            annotation_sha256=record["annotation_sha256"], quality_sha256=record["quality_sha256"],
            free_seed_sha256=input_contract["free_seed_sha256"],
            annotation_free_seed_sha256=record["free_seed_sha256"],
            full_image_hw=list(FULL_IMAGE_HW), roi_xyxy=list(FULL_ROI_XYXY),
            native_roi_hw=list(FULL_IMAGE_HW), probability_map_hw=[1500, 2048],
            scale_xy=list(OUTPUT_SCALE_XY), pixel_center_mapping=PIXEL_CENTER_MAPPING,
            output_npz=str(prediction_path), output_sha256=_file_sha256(prediction_path),
            probability_arrays=dict(
                body_probability="native/2 full-image float32; exact zero outside coverage",
                center_probability="native/2 full-image float32; exact zero outside coverage",
                coverage_mask="native/2 uint8 binary union of in-frame seed-defined tile support"),
            coverage_restricts_reference_denominator=False,
            experiment_arm=checkpoint["mode"], mode=checkpoint["mode"],
            architecture=checkpoint["architecture"], tile_layout=contract["tile_layout"],
            blend_policy="fixed positive 0.05+0.95 outer-Hann weighted overlap average",
            global_geometry=contract["global_geometry"], lowpass_geometry=contract["lowpass_geometry"],
            global_input_sha256=input_contract["global_input_sha256"],
            local_inputs_sha256=input_contract["local_inputs_sha256"],
            expected_coverage_mask_sha256=input_contract["expected_coverage_mask_sha256"],
            input_contract_sha256=input_contract["input_contract_sha256"],
            input_contract=input_contract,
            contract_source=str(contract_path), contract_source_sha256=_file_sha256(contract_path),
            implementation_source=str(module_path), implementation_source_sha256=_file_sha256(module_path),
            model_source=str(module_path), model_source_sha256=_file_sha256(module_path),
            checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=checkpoint_hash,
            training_report=str(training_report_path), training_report_sha256=training_report_hash,
            checkpoint_training_source=checkpoint.get("training_source"),
            manifest_sha256=inspected["manifest_sha256"],
            training_manifest_sha256=checkpoint.get("training_manifest_sha256"),
            training_sample_ids=checkpoint.get("training_sample_ids", []),
            train_group_ids=checkpoint.get("train_group_ids", []),
            train_annotation_sha256_by_sample=checkpoint.get("train_annotation_sha256_by_sample", {}),
            train_quality_sha256_by_sample=checkpoint.get("train_quality_sha256_by_sample", {}),
            initial_head_state_sha256=checkpoint.get("initial_head_state_sha256"),
            sampler_sequence_sha256=checkpoint.get("sampler_sequence_sha256"),
            timings_ms=timings, inference_ms=timings["end_to_end_ms"],
            human_free_seed_used=True, free_seed_source="human_polyline",
            ground_truth_center_or_edges_used_as_input=False,
            quality_used_for_input=False,
            tile_selection_inputs=contract["tile_layout"]["selection_inputs"],
            body_head_supervised=bool(checkpoint.get("body_head_supervised", False)),
            center_head_supervised=bool(checkpoint.get("center_head_supervised", False)),
            real_annotation_trained=checkpoint.get("training_source") == "reviewed_manifest_annotations",
            field_accuracy_established=False, hardware_access=False, force_command=None,
        )
        metadata_path = prediction_path.with_suffix(".json")
        _write_json(metadata_path, metadata)
        exported.append(dict(sample_id=record["sample_id"], split=record["split"],
                             prediction=str(prediction_path), metadata=str(metadata_path),
                             end_to_end_ms=timings["end_to_end_ms"]))
        print(json.dumps({"event": "exported", "sample_id": record["sample_id"],
                          "end_to_end_ms": timings["end_to_end_ms"]}), flush=True)
    selected = [sample for sample in inspected["manifest"]["samples"]
                if args.split == "all" or sample["split"] == args.split]
    exported_ids = {row["sample_id"] for row in exported}
    reason_by_id = {row["sample_id"]: row["reason"] for row in inspected["skipped"]}
    skipped = [dict(sample_id=sample["sample_id"], split=sample["split"],
                    reason=reason_by_id.get(sample["sample_id"], "absent_or_no_reviewed_free_seed"))
               for sample in selected if sample["sample_id"] not in exported_ids]
    summary = dict(
        schema_version=1, experiment_version=EXPERIMENT_VERSION,
        status="complete" if exported else "blocked_no_reviewed_free_seed",
        manifest=str(Path(args.manifest).resolve()), manifest_sha256=inspected["manifest_sha256"],
        checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=checkpoint_hash,
        mode=checkpoint["mode"], architecture=checkpoint["architecture"],
        exported=exported, skipped=skipped,
        absent_samples_have_no_output=True,
        ground_truth_center_or_edges_used_as_input=False,
        quality_used_for_input=False, hardware_access=False, force_command=None,
        field_accuracy_established=False,
    )
    _write_json(output/"summary.json", summary)
    print(json.dumps({"summary": str(output/"summary.json"), "exported": len(exported),
                      "skipped": len(skipped)}), flush=True)
    return summary


def _synthetic_record(output):
    height, width = 768, 1024
    yy, xx = np.mgrid[:height, :width]
    checker = ((xx//3+yy//3) % 2)*70
    rgb = np.stack([70+checker, 95+(xx % 31)*3, 80+(yy % 29)*3], axis=2)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    center = np.asarray([[120., 390.], [220., 390.], [400., 380.], [650., 330.], [880., 250.]])
    image_path = output/"synthetic_input.png"
    Image.fromarray(rgb, "RGB").save(image_path)
    image_hash = _file_sha256(image_path)
    raw = dict(centerline=center, edge_flange=center+np.array([0., -6.]),
               edge_surface=center+np.array([0., 6.]),
               free_seed=np.asarray([[60., 390.], [120., 390.], [220., 390.]]))
    trusted = {name: values[:4].copy() for name, values in raw.items() if name != "free_seed"}
    return dict(
        sample_id="synthetic_v9", group_id="synthetic_group", split="train",
        image_path=str(image_path.resolve()), image_sha256=image_hash,
        image_shape_hw=[height, width], roi_xyxy=[0, 0, width, height],
        annotation_path=str((output/"synthetic_label_not_written.json").resolve()),
        annotation_sha256="synthetic-not-a-field-label", quality_sha256="synthetic",
        free_seed_sha256=_canonical_json_sha256(raw["free_seed"].tolist()),
        label={"status": "partial"}, parsed_polylines=raw,
        parsed_supervision_polylines=trusted,
        parsed_ignore_polygons=[],
        parsed_quality_exclude_polygons=[np.asarray([[700., 0.], [1023., 0.],
                                                    [1023., 767.], [700., 767.]])],
    )


def synthetic_smoke(args):
    torch.set_num_threads(int(args.threads))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"report.json").exists():
        raise FileExistsError(f"Refusing to overwrite synthetic smoke: {output}")
    record = _synthetic_record(output)
    prepared = {mode: prepare_record(record, mode, include_supervision=True)
                for mode in MODES}
    assert torch.equal(prepared["native_detail"]["global_input"],
                       prepared["lowpass_control"]["global_input"])
    assert torch.equal(prepared["native_detail"]["local_inputs"][:, 3],
                       prepared["lowpass_control"]["local_inputs"][:, 3])
    if torch.equal(prepared["native_detail"]["local_inputs"][:, :3],
                   prepared["lowpass_control"]["local_inputs"][:, :3]):
        raise AssertionError("Synthetic high-frequency RGB did not distinguish experiment arms")
    for tile in range(9):
        for key in ("body_target", "center_target", "body_valid_mask", "center_valid_mask"):
            if not torch.equal(prepared["native_detail"]["supervision"][tile][key],
                               prepared["lowpass_control"]["supervision"][tile][key]):
                raise AssertionError("Paired V9 supervision differs by mode")
    sequence = build_training_sequence([record], args.steps, args.batch_size, args.seed)
    arm_reports = {}
    initial_hashes = []
    for mode in MODES:
        torch.manual_seed(int(args.seed))
        model = build_model(args.base_channels).cpu()
        initial_hash = _state_dict_sha256(model.state_dict())
        initial_hashes.append(initial_hash)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        cache = PreparedRecordCache([record], mode, cache_samples=True)
        cache._cache[0] = prepared[mode]
        losses = []
        for items in sequence:
            batch, zero_items = _batch_from_sequence(cache, items)
            optimizer.zero_grad(set_to_none=True)
            _, values = _loss(model, batch, 1.0)
            if not torch.isfinite(values["loss"]):
                raise AssertionError("Synthetic V9 loss is non-finite")
            values["loss"].backward()
            optimizer.step()
            losses.append(dict(loss=float(values["loss"].detach()), zero_items=zero_items))
        arm_reports[mode] = dict(initial_state_sha256=initial_hash, losses=losses,
                                 final_state_sha256=_state_dict_sha256(model.state_dict()),
                                 global_input_sha256=prepared[mode]["input_contract"]["global_input_sha256"],
                                 local_inputs_sha256=prepared[mode]["input_contract"]["local_inputs_sha256"])
    if len(set(initial_hashes)) != 1:
        raise AssertionError("Paired V9 arms did not start from identical model state")
    report = dict(
        schema_version=1, experiment_version=EXPERIMENT_VERSION,
        status="synthetic_mechanical_smoke_complete", synthetic_only=True,
        mode_sequence=list(MODES), steps=int(args.steps), batch_size=int(args.batch_size),
        base_channels=int(args.base_channels), seed=int(args.seed), threads=int(args.threads),
        sampler_sequence=sequence, sampler_sequence_sha256=canonical_json_sha256(sequence),
        arms=arm_reports, paired_checks=dict(
            identical_initial_state=True, identical_sampler=True, identical_global_input=True,
            identical_local_seed=True, identical_targets_and_valid_masks=True,
            local_rgb_differs=True),
        claims=dict(field_data_used=False, field_accuracy_established=False,
                    production_ready=False, hardware_access=False, checkpoint_saved=False),
    )
    _write_json(output/"report.json", report)
    print(json.dumps({"report": str(output/"report.json"), "status": report["status"]}), flush=True)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train-manifest", help="Run one frozen V9 field-training arm")
    train.add_argument("--manifest", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--mode", required=True, choices=MODES)
    train.add_argument("--steps", type=int, default=300)
    train.add_argument("--seed", type=int, default=20260927)
    train.add_argument("--threads", type=int, default=2)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--base-channels", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--center-weight", type=float, default=1.0)
    train.add_argument("--cache-samples", action="store_true",
                       help="Cache only the selected 10 train and 8 holdout preprocessed records")
    train.set_defaults(action=train_manifest)
    infer = commands.add_parser("infer-manifest", help="Export native/2 full-image paired-arm maps")
    infer.add_argument("--manifest", required=True)
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--output-dir", required=True)
    infer.add_argument("--split", choices=("train", "holdout", "all"), default="all")
    infer.add_argument("--threads", type=int, default=2)
    infer.add_argument("--tile-batch-size", type=int, default=1)
    infer.set_defaults(action=infer_manifest)
    smoke = commands.add_parser("synthetic-smoke", help="Two-arm CPU mechanical smoke; no field claim")
    smoke.add_argument("--output-dir", default=str(DEFAULT_OUTPUT/"synthetic_smoke"))
    smoke.add_argument("--steps", type=int, default=2)
    smoke.add_argument("--seed", type=int, default=20260927)
    smoke.add_argument("--threads", type=int, default=2)
    smoke.add_argument("--batch-size", type=int, default=2)
    smoke.add_argument("--base-channels", type=int, default=4)
    smoke.set_defaults(action=synthetic_smoke)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if (getattr(args, "steps", 1) <= 0 or getattr(args, "batch_size", 1) <= 0
            or getattr(args, "base_channels", 2) < 2 or args.threads <= 0):
        raise ValueError("steps, batch-size, threads must be positive and base-channels >=2")
    if hasattr(args, "tile_batch_size") and args.tile_batch_size <= 0:
        raise ValueError("tile-batch-size must be positive")
    return args.action(args)


if __name__ == "__main__":
    main()
