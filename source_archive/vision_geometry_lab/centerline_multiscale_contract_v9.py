"""Torch-free V9 input/tile contract shared by training, export and evaluation.

Only RGB pixels and the reviewed free-seed polyline are accepted here.  Target
curves, quality masks and scoring references are intentionally absent from the
API, so they cannot influence window placement or input hashes.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


CONTRACT_VERSION = 1
MODES = ("native_detail", "lowpass_control")
GLOBAL_HW = (256, 384)
LOWPASS_HW = (512, 1024)
TILE_NATIVE_SIZE = 768
LOCAL_MODEL_SIZE = 384
TILE_U = (0, 512, 1024)
TILE_V = (-384, 0, 384)
OUTPUT_SCALE_XY = (0.5, 0.5)
ROUNDING_POLICY = "nearest multiple of 2; exact ties toward positive infinity"
PIXEL_CENTER_MAPPING = "map_xy=(full_image_xy+0.5)*[0.5,0.5]-0.5"


def canonical_json_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_sha256(array):
    value = np.ascontiguousarray(array)
    header = json.dumps({"shape": list(value.shape), "dtype": str(value.dtype)},
                        sort_keys=True, separators=(",", ":")).encode("ascii")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "little"))
    digest.update(header)
    digest.update(value.tobytes())
    return digest.hexdigest()


def _validate_rgb(image_rgb_uint8):
    image = np.asarray(image_rgb_uint8)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image_rgb_uint8 must be an HxWx3 uint8 array")
    height, width = image.shape[:2]
    if height < 2 or width < 2 or height % 2 or width % 2:
        raise ValueError("V9 requires even full-image height and width >= 2")
    return np.ascontiguousarray(image), height, width


def _validate_seed(free_seed_xy, height, width):
    seed = np.asarray(free_seed_xy, dtype=np.float64)
    if (seed.ndim != 2 or seed.shape[1] != 2 or len(seed) < 2
            or not np.isfinite(seed).all()):
        raise ValueError("free_seed_xy must contain at least two finite [x,y] points")
    if np.any(seed[:, 0] < 0) or np.any(seed[:, 0] > width-1):
        raise ValueError("free_seed x coordinate outside full image")
    if np.any(seed[:, 1] < 0) or np.any(seed[:, 1] > height-1):
        raise ValueError("free_seed y coordinate outside full image")
    direction = seed[-1]-seed[0]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        raise ValueError("free_seed first and last points must differ")
    return seed, direction/norm


def _round_even(value):
    if not math.isfinite(float(value)):
        raise ValueError("tile coordinate must be finite")
    return int(2*math.floor(float(value)/2+0.5))


def build_tile_layout(free_seed_xy, native_hw):
    """Build nine target-independent native windows from free-seed geometry."""
    height, width = map(int, native_hw)
    if height < 2 or width < 2 or height % 2 or width % 2:
        raise ValueError("native_hw must contain even positive dimensions")
    seed, direction = _validate_seed(free_seed_xy, height, width)
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    endpoint = seed[-1]
    tiles = []
    for u in TILE_U:
        for v in TILE_V:
            center = endpoint+float(u)*direction+float(v)*normal
            left = _round_even(center[0]-TILE_NATIVE_SIZE/2)
            top = _round_even(center[1]-TILE_NATIVE_SIZE/2)
            right, bottom = left+TILE_NATIVE_SIZE, top+TILE_NATIVE_SIZE
            padding = [max(0, -left), max(0, -top), max(0, right-width), max(0, bottom-height)]
            tiles.append(dict(
                index=len(tiles), u_px=int(u), v_px=int(v), center_xy=center.tolist(),
                native_xyxy=[left, top, right, bottom], padding_native_ltrb=padding,
                local_model_hw=[LOCAL_MODEL_SIZE, LOCAL_MODEL_SIZE],
                model_to_full_image_pixel_center=(
                    "full_xy=(local_xy+0.5)*2-0.5+[tile_left,tile_top]"),
                output_map_offset_xy=[left//2, top//2],
            ))
    return dict(
        layout_version=CONTRACT_VERSION,
        selection_inputs="free_seed endpoint plus unit(first-to-last) direction only",
        target_or_quality_used=False,
        seed_endpoint_xy=endpoint.tolist(), unit_direction_xy=direction.tolist(),
        unit_normal_xy=normal.tolist(), u_values_px=list(TILE_U), v_values_px=list(TILE_V),
        enumeration_order="u outer, v inner",
        native_tile_hw=[TILE_NATIVE_SIZE, TILE_NATIVE_SIZE],
        local_model_hw=[LOCAL_MODEL_SIZE, LOCAL_MODEL_SIZE],
        native_to_local_scale_xy=[0.5, 0.5], top_left_rounding=ROUNDING_POLICY,
        tiles=tiles,
    )


def expected_coverage_mask(tile_layout, native_hw):
    """Binary union of in-image 2x2 native footprints on the native/2 grid."""
    height, width = map(int, native_hw)
    if height % 2 or width % 2:
        raise ValueError("native dimensions must be even")
    coverage = np.zeros((height//2, width//2), dtype=np.uint8)
    for tile in tile_layout["tiles"]:
        left, top, right, bottom = map(int, tile["native_xyxy"])
        out_left, out_top = left//2, top//2
        out_right, out_bottom = right//2, bottom//2
        x0, x1 = max(0, out_left), min(width//2, out_right)
        y0, y1 = max(0, out_top), min(height//2, out_bottom)
        if x1 > x0 and y1 > y0:
            coverage[y0:y1, x0:x1] = 1
    return coverage


def tile_in_frame_mask(tile, native_hw):
    """Return the local 384x384 binary mask whose native 2x2 footprint is in-frame."""
    height, width = map(int, native_hw)
    left, top, _, _ = map(int, tile["native_xyxy"])
    native_x0 = left+2*np.arange(LOCAL_MODEL_SIZE, dtype=np.int64)
    native_y0 = top+2*np.arange(LOCAL_MODEL_SIZE, dtype=np.int64)
    valid_x = (native_x0 >= 0) & (native_x0+1 < width)
    valid_y = (native_y0 >= 0) & (native_y0+1 < height)
    return np.ascontiguousarray(np.outer(valid_y, valid_x).astype(np.uint8))


def blend_weight_map():
    """Fixed positive raised-cosine weights; positivity preserves binary coverage."""
    axis = np.hanning(LOCAL_MODEL_SIZE).astype(np.float32)
    weight = 0.05+0.95*np.outer(axis, axis)
    return np.ascontiguousarray(weight, dtype=np.float32)


def _normalize_nonempty(array):
    value = np.asarray(array, dtype=np.float32)
    peak = float(value.max(initial=0.0))
    return value/peak if peak > 0 else np.zeros_like(value)


def render_free_seed(native_hw, free_seed_xy):
    """V5-compatible reviewed free-seed encoding, implemented without torch."""
    height, width = map(int, native_hw)
    seed, _ = _validate_seed(free_seed_xy, height, width)
    short = max(1, min(height, width))
    line_width = max(1, round(short*0.006))
    blur = max(1.0, short*0.008)
    image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(image).line([tuple(point) for point in seed], fill=255,
                               width=int(line_width), joint="curve")
    image = image.filter(ImageFilter.GaussianBlur(float(blur)))
    return _normalize_nonempty(np.asarray(image, dtype=np.float32)/255.0)


def _fitpad_geometry(native_hw, target_hw):
    native_height, native_width = map(int, native_hw)
    target_height, target_width = map(int, target_hw)
    scale = min(target_width/native_width, target_height/native_height)
    content_width = max(1, min(target_width, int(round(native_width*scale))))
    content_height = max(1, min(target_height, int(round(native_height*scale))))
    left = (target_width-content_width)//2
    top = (target_height-content_height)//2
    return dict(native_hw=[native_height, native_width], model_input_hw=[target_height, target_width],
                content_hw=[content_height, content_width],
                content_xyxy=[left, top, left+content_width, top+content_height],
                padding_ltrb=[left, top, target_width-content_width-left,
                              target_height-content_height-top],
                scale_xy=[content_width/native_width, content_height/native_height],
                mapping="model_xy=(native_xy+0.5)*scale_xy-0.5+pad_left_top")


def _resize_rgb(rgb_uint8, output_hw):
    height, width = map(int, output_hw)
    return np.asarray(Image.fromarray(rgb_uint8, "RGB").resize(
        (width, height), resample=Image.Resampling.BILINEAR), dtype=np.float32)/255.0


def _resize_plane(plane, output_hw, *, normalize=False):
    height, width = map(int, output_hw)
    image = Image.fromarray(np.rint(np.clip(plane, 0, 1)*255).astype(np.uint8), "L")
    result = np.asarray(image.resize((width, height), resample=Image.Resampling.BILINEAR),
                        dtype=np.float32)/255.0
    return _normalize_nonempty(result) if normalize else result


def _fitpad_input(rgb_uint8, seed, target_hw):
    geometry = _fitpad_geometry(rgb_uint8.shape[:2], target_hw)
    content_hw = geometry["content_hw"]
    rgb = _resize_rgb(rgb_uint8, content_hw)
    seed_resized = _resize_plane(seed, content_hw, normalize=True)
    height, width = geometry["model_input_hw"]
    left, top, right, bottom = geometry["content_xyxy"]
    canvas_rgb = np.zeros((height, width, 3), dtype=np.float32)
    canvas_seed = np.zeros((height, width), dtype=np.float32)
    canvas_rgb[top:bottom, left:right] = rgb
    canvas_seed[top:bottom, left:right] = seed_resized
    return np.ascontiguousarray(np.concatenate(
        [canvas_rgb.transpose(2, 0, 1), canvas_seed[None]], axis=0), dtype=np.float32), geometry


def lowpass_roundtrip_rgb(image_rgb_uint8):
    """V6 512x1024 fit content down/up, without injecting padded black borders."""
    image, height, width = _validate_rgb(image_rgb_uint8)
    geometry = _fitpad_geometry((height, width), LOWPASS_HW)
    down = Image.fromarray(image, "RGB").resize(
        (geometry["content_hw"][1], geometry["content_hw"][0]),
        resample=Image.Resampling.BILINEAR)
    restored = np.asarray(down.resize((width, height), resample=Image.Resampling.BILINEAR),
                          dtype=np.uint8)
    return np.ascontiguousarray(restored), geometry


def _extract_padded(array, xyxy):
    left, top, right, bottom = map(int, xyxy)
    tile_height, tile_width = bottom-top, right-left
    shape = (tile_height, tile_width)+array.shape[2:]
    result = np.zeros(shape, dtype=array.dtype)
    source_left, source_top = max(0, left), max(0, top)
    source_right, source_bottom = min(array.shape[1], right), min(array.shape[0], bottom)
    if source_right > source_left and source_bottom > source_top:
        destination_left, destination_top = source_left-left, source_top-top
        result[destination_top:destination_top+source_bottom-source_top,
               destination_left:destination_left+source_right-source_left] = (
                   array[source_top:source_bottom, source_left:source_right])
    return result


def _local_inputs(local_rgb_uint8, seed, tile_layout):
    inputs = []
    for tile in tile_layout["tiles"]:
        rgb_tile = _extract_padded(local_rgb_uint8, tile["native_xyxy"])
        seed_tile = _extract_padded(seed, tile["native_xyxy"])
        rgb_resized = _resize_rgb(rgb_tile, (LOCAL_MODEL_SIZE, LOCAL_MODEL_SIZE))
        seed_resized = _resize_plane(seed_tile, (LOCAL_MODEL_SIZE, LOCAL_MODEL_SIZE), normalize=True)
        inputs.append(np.concatenate([rgb_resized.transpose(2, 0, 1), seed_resized[None]], axis=0))
    return np.ascontiguousarray(np.stack(inputs), dtype=np.float32)


def rebuild_input_contract(image_rgb_uint8, free_seed_xy, mode):
    """Rebuild every V9 inference input/hash and expected binary coverage.

    This function intentionally has no label, target, quality or checkpoint
    argument.  Evaluators can therefore recompute the contract before trusting
    exported metadata.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    image, height, width = _validate_rgb(image_rgb_uint8)
    seed_points, _ = _validate_seed(free_seed_xy, height, width)
    seed = render_free_seed((height, width), seed_points)
    layout = build_tile_layout(seed_points, (height, width))
    global_input, global_geometry = _fitpad_input(image, seed, GLOBAL_HW)
    if mode == "native_detail":
        local_rgb = image
        lowpass_geometry = None
    else:
        local_rgb, lowpass_geometry = lowpass_roundtrip_rgb(image)
    local_inputs = _local_inputs(local_rgb, seed, layout)
    coverage = expected_coverage_mask(layout, (height, width))
    global_hash = array_sha256(global_input)
    local_hash = array_sha256(local_inputs)
    coverage_hash = array_sha256(coverage)
    seed_hash = canonical_json_sha256(seed_points.tolist())
    payload = dict(
        contract_version=CONTRACT_VERSION, mode=mode, native_hw=[height, width],
        free_seed_sha256=seed_hash, global_hw=list(GLOBAL_HW), lowpass_hw=list(LOWPASS_HW),
        native_tile_hw=[TILE_NATIVE_SIZE, TILE_NATIVE_SIZE],
        local_model_hw=[LOCAL_MODEL_SIZE, LOCAL_MODEL_SIZE],
        output_scale_xy=list(OUTPUT_SCALE_XY), pixel_center_mapping=PIXEL_CENTER_MAPPING,
        tile_layout_sha256=canonical_json_sha256(layout),
        global_input_sha256=global_hash, local_inputs_sha256=local_hash,
        expected_coverage_mask_sha256=coverage_hash,
        global_rgb_source="original full RGB in both modes",
        local_seed_source="original reviewed free-seed heatmap in both modes",
        local_rgb_source=("original full RGB" if mode == "native_detail"
                          else "original RGB downsampled through V6 512x1024 fit content then restored"),
    )
    return dict(
        input_contract=dict(payload, input_contract_sha256=canonical_json_sha256(payload)),
        tile_layout=layout, global_geometry=global_geometry,
        lowpass_geometry=lowpass_geometry,
        global_input=global_input, local_inputs=local_inputs,
        expected_coverage_mask=coverage,
    )
