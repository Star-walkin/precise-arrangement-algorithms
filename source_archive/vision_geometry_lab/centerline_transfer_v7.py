"""Frozen-SAM2 feature transfer candidate for seed-conditioned centreline maps.

This is an offline experiment.  The only task input is native ROI RGB plus the
reviewed human free-span seed.  Target centre/edge annotations are used solely
by the masked training loss; they are never passed to SAM2 or to inference.

V7 deliberately keeps two paired feature modes with one identical trainable
head: ``sam2_frozen`` uses RGB-only features from the factory SAM2.1 Hiera tiny
checkpoint, while ``zero_control`` replaces every such feature with exact
zeros.  SAM2 parameters are never stored in, or updated by, a V7 checkpoint.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import re
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .centerline_model_v5 import (
    INPUT_SEMANTICS, OUTPUT_HEADS, RASTER_ENCODING_VERSION,
    _canonical_json_sha256, _file_sha256, inspect_manifest,
    masked_two_head_loss,
)
from .centerline_context_v6 import (
    DEFAULT_TARGET_HW, _positive_hw, _supervision_summary,
    crop_letterbox_probabilities, letterbox_manifest_input,
    rasterize_letterbox_record,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / 'outputs/centerline_v7/manifest_model'
DEFAULT_SAM_CHECKPOINT = ROOT / 'vision_training/checkpoints/sam2.1_hiera_tiny.pt'
SAM_CONFIG_NAME = 'configs/sam2.1/sam2.1_hiera_t.yaml'
SAM_CONFIG_PATH = ROOT / 'vision_training/vendor/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml'
SAM_BUILD_PATH = ROOT / 'vision_training/vendor/sam2/sam2/build_sam.py'
SAM_BASE_PATH = ROOT / 'vision_training/vendor/sam2/sam2/modeling/sam2_base.py'
EXPECTED_SAM_CHECKPOINT_SHA256 = '7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69'
EXPECTED_SAM_CONFIG_SHA256 = '9eb1c5e9db577925464492ccbbf6810731bcbbde31325fb8d297d78c3eb5d9dc'
CHECKPOINT_FORMAT = 'centerline_transfer_v7_frozen_sam2_two_head'
FEATURE_CACHE_FORMAT = 'centerline_transfer_v7_rgb_only_sam2_features'
EXPERIMENT_VERSION = 7
PREPROCESS_VERSION = 1
SAM_SQUARE_SIZE = 1024
SAM_FEATURE_CHANNELS = (32, 64, 256)
SAM_FEATURE_STRIDES = (4, 8, 16)
FEATURE_MODES = ('sam2_frozen', 'zero_control')


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(payload, 'utf-8')
    os.replace(temporary, path)


def _module_sha256():
    return _file_sha256(Path(__file__))


def _vendor_code_sha256():
    return _canonical_json_sha256({
        'build_sam.py': _file_sha256(SAM_BUILD_PATH),
        'sam2_base.py': _file_sha256(SAM_BASE_PATH),
        'config': _file_sha256(SAM_CONFIG_PATH),
    })


def sam_embedding_geometry(canvas_hw=DEFAULT_TARGET_HW, square_size=SAM_SQUARE_SIZE):
    """Map a V6 letterbox canvas into a square without resizing it.

    The factory configuration produces feature grids at strides 4, 8 and 16.
    We require embedding boundaries aligned to all three grids so cropping is
    an exact integer slice, not a second interpolation or an ambiguous round.
    """
    height, width = _positive_hw(canvas_hw, 'canvas_hw')
    if isinstance(square_size, bool) or not isinstance(square_size, int) or square_size <= 0:
        raise ValueError('square_size must be a positive integer')
    if height > square_size or width > square_size:
        raise ValueError('V6 canvas must fit inside the SAM square without resizing')
    left = (square_size - width) // 2
    top = (square_size - height) // 2
    right, bottom = left + width, top + height
    crops = []
    full_shapes = []
    for channels, stride in zip(SAM_FEATURE_CHANNELS, SAM_FEATURE_STRIDES):
        if square_size % stride or any(value % stride for value in (left, top, right, bottom)):
            raise ValueError('SAM embedding boundaries must align to feature strides 4/8/16')
        full = square_size // stride
        crop = [left // stride, top // stride, right // stride, bottom // stride]
        full_shapes.append([channels, full, full])
        crops.append(crop)
    return {
        'canvas_hw': [height, width], 'sam_square_hw': [square_size, square_size],
        'canvas_xyxy_in_sam': [left, top, right, bottom],
        'feature_strides': list(SAM_FEATURE_STRIDES),
        'feature_full_chw': full_shapes, 'feature_crop_xyxy': crops,
        'mapping': 'SAM square pixels contain the V6 canvas unchanged at canvas_xyxy_in_sam',
    }


def sam_normalized_square(input_tensor, square_size=SAM_SQUARE_SIZE):
    """Embed RGB in a black SAM square and apply the official mean/std only.

    ``input_tensor`` may include the seed channel, but only its first three RGB
    planes are read.  No centre, edge, valid-mask, or seed value can enter the
    frozen backbone.
    """
    if input_tensor.ndim != 3 or input_tensor.shape[0] < 3:
        raise ValueError('Expected CxHxW with at least three RGB channels')
    rgb = input_tensor[:3].detach().to(dtype=torch.float32, device='cpu')
    if not torch.isfinite(rgb).all() or torch.any((rgb < 0) | (rgb > 1)):
        raise ValueError('RGB must be finite and normalized to [0,1]')
    geometry = sam_embedding_geometry(rgb.shape[-2:], int(square_size))
    left, top, right, bottom = geometry['canvas_xyxy_in_sam']
    square = torch.zeros((1, 3, square_size, square_size), dtype=torch.float32)
    square[:, :, top:bottom, left:right] = rgb.unsqueeze(0)
    mean = torch.tensor((.485, .456, .406), dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor((.229, .224, .225), dtype=torch.float32).view(1, 3, 1, 1)
    return (square - mean) / std, geometry


def crop_sam_feature_pyramid(backbone_fpn, geometry):
    if not isinstance(backbone_fpn, (tuple, list)) or len(backbone_fpn) != 3:
        raise ValueError('Expected the three SAM2 FPN levels')
    cropped = []
    for index, (feature, expected, crop) in enumerate(zip(
            backbone_fpn, geometry['feature_full_chw'], geometry['feature_crop_xyxy'])):
        if feature.ndim != 4 or feature.shape[0] != 1 or list(feature.shape[1:]) != expected:
            raise ValueError(f'Unexpected SAM2 feature level {index} shape: {list(feature.shape)}')
        left, top, right, bottom = crop
        value = feature[0, :, top:bottom, left:right].detach().float().cpu().contiguous()
        if not torch.isfinite(value).all():
            raise ValueError(f'Non-finite SAM2 feature level {index}')
        cropped.append(value)
    return tuple(cropped)


def expected_cropped_feature_shapes(canvas_hw=DEFAULT_TARGET_HW):
    geometry = sam_embedding_geometry(canvas_hw, SAM_SQUARE_SIZE)
    result = []
    for channels, crop in zip(SAM_FEATURE_CHANNELS, geometry['feature_crop_xyxy']):
        left, top, right, bottom = crop
        result.append((channels, bottom - top, right - left))
    return tuple(result)


class FrozenSam2Extractor:
    """Factory SAM2.1 tiny image encoder, held in eval/inference-only mode."""
    def __init__(self, checkpoint=DEFAULT_SAM_CHECKPOINT, threads=None):
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f'SAM2 factory checkpoint not found: {checkpoint}')
        checkpoint_sha = _file_sha256(checkpoint)
        if checkpoint_sha != EXPECTED_SAM_CHECKPOINT_SHA256:
            raise ValueError('Only the verified factory SAM2.1 Hiera tiny checkpoint is accepted')
        config_sha = _file_sha256(SAM_CONFIG_PATH)
        if config_sha != EXPECTED_SAM_CONFIG_SHA256:
            raise ValueError('SAM2.1 Hiera tiny config hash mismatch')
        if threads is not None:
            torch.set_num_threads(int(threads))
        from sam2.build_sam import build_sam2
        started = time.perf_counter()
        self.model = build_sam2(SAM_CONFIG_NAME, str(checkpoint), device='cpu', mode='eval',
                                apply_postprocessing=False)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.load_seconds = time.perf_counter() - started
        self.checkpoint = checkpoint
        self.checkpoint_sha256 = checkpoint_sha
        self.config_sha256 = config_sha
        self.extractor_code_sha256 = _module_sha256()
        self.vendor_code_sha256 = _vendor_code_sha256()
        self.parameter_count = int(sum(parameter.numel() for parameter in self.model.parameters()))
        self.trainable_parameter_count = int(sum(parameter.numel() for parameter in self.model.parameters()
                                                 if parameter.requires_grad))

    def extract(self, input_tensor):
        if self.model is None:
            raise RuntimeError('SAM2 extractor has been released')
        square, geometry = sam_normalized_square(input_tensor, SAM_SQUARE_SIZE)
        started = time.perf_counter()
        with torch.inference_mode():
            backbone = self.model.forward_image(square)
        elapsed = time.perf_counter() - started
        features = crop_sam_feature_pyramid(backbone['backbone_fpn'], geometry)
        return features, {
            'forward_seconds': elapsed,
            'feature_bytes_float32': int(sum(value.numel() * value.element_size()
                                                 for value in features)),
            'feature_shapes_chw': [list(value.shape) for value in features],
            'sam_embedding': geometry,
        }

    def release(self):
        self.model = None


def _safe_sample_id(value):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value))[:120]


class FeatureProvider:
    """Disk-backed RGB-only SAM cache or its exact-zero paired control."""
    def __init__(self, feature_mode, target_hw=DEFAULT_TARGET_HW, cache_dir=None,
                 sam_checkpoint=DEFAULT_SAM_CHECKPOINT, extractor=None, threads=None):
        if feature_mode not in FEATURE_MODES:
            raise ValueError(f'feature_mode must be one of {FEATURE_MODES}')
        self.feature_mode = feature_mode
        self.target_hw = _positive_hw(target_hw, 'target_hw')
        self.embedding = sam_embedding_geometry(self.target_hw, SAM_SQUARE_SIZE)
        self.cache_dir = Path(cache_dir).resolve() if cache_dir is not None else None
        self.sam_checkpoint = Path(sam_checkpoint).resolve() if sam_checkpoint is not None else None
        self.extractor = extractor
        self.threads = threads
        self.source_info = None
        if self.feature_mode == 'sam2_frozen':
            if self.sam_checkpoint is None or not self.sam_checkpoint.is_file():
                raise FileNotFoundError(f'SAM2 factory checkpoint not found: {self.sam_checkpoint}')
            checkpoint_sha = _file_sha256(self.sam_checkpoint)
            if checkpoint_sha != EXPECTED_SAM_CHECKPOINT_SHA256:
                raise ValueError('Only the verified factory SAM2.1 Hiera tiny checkpoint is accepted')
            if _file_sha256(SAM_CONFIG_PATH) != EXPECTED_SAM_CONFIG_SHA256:
                raise ValueError('SAM2.1 Hiera tiny config hash mismatch')
            self.source_info = {
                'checkpoint': str(self.sam_checkpoint),
                'checkpoint_sha256': checkpoint_sha,
                'config_sha256': EXPECTED_SAM_CONFIG_SHA256,
                'extractor_code_sha256': _module_sha256(),
                'vendor_code_sha256': _vendor_code_sha256(),
            }
            if self.extractor is not None:
                supplied = {
                    'checkpoint': str(Path(self.extractor.checkpoint).resolve()),
                    'checkpoint_sha256': self.extractor.checkpoint_sha256,
                    'config_sha256': self.extractor.config_sha256,
                    'extractor_code_sha256': self.extractor.extractor_code_sha256,
                    'vendor_code_sha256': self.extractor.vendor_code_sha256,
                }
                if supplied != self.source_info:
                    raise ValueError('Supplied SAM2 extractor provenance mismatch')
        self.cache_hits = 0
        self.cache_misses = 0
        self.extraction_seconds = 0.
        self.cache_paths = {}

    def _ensure_extractor(self):
        if self.feature_mode == 'zero_control':
            return None
        if self.extractor is None:
            self.extractor = FrozenSam2Extractor(self.sam_checkpoint, self.threads)
        return self.extractor

    def _rgb_sha256(self, input_tensor):
        rgb = input_tensor[:3].detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
        return hashlib.sha256(rgb.tobytes(order='C')).hexdigest()

    def _identity(self, record, input_tensor):
        return {
            'format': FEATURE_CACHE_FORMAT,
            'schema_version': 1,
            'sample_id': record['sample_id'],
            'image_sha256': record['image_sha256'],
            'roi_xyxy': list(record['roi_xyxy']),
            'model_input_hw': list(self.target_hw),
            'rgb_canvas_sha256': self._rgb_sha256(input_tensor),
            'source_checkpoint_sha256': self.source_info['checkpoint_sha256'],
            'sam_config_sha256': self.source_info['config_sha256'],
            'extractor_code_sha256': self.source_info['extractor_code_sha256'],
            'vendor_code_sha256': self.source_info['vendor_code_sha256'],
            'input_semantics': 'rgb_only_no_seed_no_target',
            'sam_embedding': self.embedding,
        }

    def _path(self, identity):
        if self.cache_dir is None:
            raise ValueError('sam2_frozen requires a feature cache directory')
        key = _canonical_json_sha256(identity)
        return self.cache_dir / f"{_safe_sample_id(identity['sample_id'])}.{key[:20]}.npz", key

    @staticmethod
    def _write_cache(path, identity, key, features, extraction):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(identity)
        metadata.update(cache_key=key, extraction=extraction,
                        ground_truth_used=False, free_seed_used=False)
        temporary = path.with_suffix(path.suffix + '.tmp')
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, metadata_json=np.asarray(
                json.dumps(metadata, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                           allow_nan=False)),
                feature_0=features[0].numpy().astype(np.float32, copy=False),
                feature_1=features[1].numpy().astype(np.float32, copy=False),
                feature_2=features[2].numpy().astype(np.float32, copy=False))
        os.replace(temporary, path)

    def _load_cache(self, path, identity, key):
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != {'metadata_json', 'feature_0', 'feature_1', 'feature_2'}:
                raise ValueError('Unexpected V7 feature cache arrays')
            metadata = json.loads(str(arrays['metadata_json'].item()))
            features = tuple(torch.from_numpy(arrays[f'feature_{index}'].copy()).float()
                             for index in range(3))
        if metadata.get('cache_key') != key:
            raise ValueError('V7 feature cache key mismatch')
        stored_identity = {name: metadata.get(name) for name in identity}
        if stored_identity != identity:
            raise ValueError('V7 feature cache provenance mismatch')
        expected = expected_cropped_feature_shapes(self.target_hw)
        if tuple(tuple(value.shape) for value in features) != expected:
            raise ValueError('V7 feature cache tensor shape mismatch')
        if any(not torch.isfinite(value).all() for value in features):
            raise ValueError('V7 feature cache contains non-finite values')
        return features, metadata

    def get(self, record, input_tensor):
        if list(input_tensor.shape[-2:]) != list(self.target_hw):
            raise ValueError('Input tensor does not match V7 model canvas')
        if self.feature_mode == 'zero_control':
            features = tuple(torch.zeros(shape, dtype=torch.float32)
                             for shape in expected_cropped_feature_shapes(self.target_hw))
            provenance = {
                'source_checkpoint': None, 'source_checkpoint_sha256': None,
                'extractor_code_sha256': _module_sha256(),
                'cache_npz': None, 'cache_sha256': None,
                'input_semantics': 'rgb_only_no_seed_no_target',
                'backbone_trainable': False, 'strict_zero_features': True,
            }
            return features, provenance
        identity = self._identity(record, input_tensor)
        path, key = self._path(identity)
        if path.is_file():
            features, metadata = self._load_cache(path, identity, key)
            self.cache_hits += 1
        else:
            extractor = self._ensure_extractor()
            features, extraction = extractor.extract(input_tensor)
            if tuple(tuple(value.shape) for value in features) != expected_cropped_feature_shapes(self.target_hw):
                raise ValueError('Extracted SAM2 feature crop does not match the V6 canvas')
            self._write_cache(path, identity, key, features, extraction)
            features, metadata = self._load_cache(path, identity, key)
            self.cache_misses += 1
            self.extraction_seconds += float(extraction['forward_seconds'])
        self.cache_paths[record['sample_id']] = str(path)
        provenance = {
            'source_checkpoint': self.source_info['checkpoint'],
            'source_checkpoint_sha256': identity['source_checkpoint_sha256'],
            'extractor_code_sha256': identity['extractor_code_sha256'],
            'cache_npz': str(path), 'cache_sha256': _file_sha256(path),
            'input_semantics': 'rgb_only_no_seed_no_target',
            'backbone_trainable': False, 'strict_zero_features': False,
            'cache_key': key,
            'original_extraction_seconds': metadata['extraction']['forward_seconds'],
        }
        return features, provenance

    def prepare(self, records):
        results = {}
        for record in records:
            input_tensor, _, _, _ = letterbox_manifest_input(record, self.target_hw)
            _, provenance = self.get(record, input_tensor)
            results[record['sample_id']] = provenance
        return results

    def release_extractor(self):
        if self.extractor is not None and hasattr(self.extractor, 'release'):
            self.extractor.release()
        self.extractor = None


class TransferManifestDataset(Dataset):
    def __init__(self, records, provider, cache_samples=False):
        self.records = list(records)
        self.provider = provider
        self.cache_samples = bool(cache_samples)
        self._cache = ([self._make(index) for index in range(len(self.records))]
                       if self.cache_samples else None)

    def _make(self, index):
        record = self.records[index]
        sample = rasterize_letterbox_record(record, self.provider.target_hw)
        features, _ = self.provider.get(record, sample['input'])
        for level, value in enumerate(features):
            sample[f'feature_{level}'] = value
        return sample

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self._cache[index] if self._cache is not None else self._make(index)

    def cache_bytes(self):
        if self._cache is None:
            return 0
        keys = ('input', 'body_target', 'center_target', 'body_valid_mask',
                'center_valid_mask', 'feature_0', 'feature_1', 'feature_2')
        return int(sum(sample[key].numel() * sample[key].element_size()
                       for sample in self._cache for key in keys))


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


class SeedConditionedFeatureHead(nn.Module):
    """One paired head for real frozen features and exact-zero controls."""
    def __init__(self, base_channels=8, projected_feature_channels=8):
        super().__init__()
        if base_channels < 2 or projected_feature_channels < 1:
            raise ValueError('base_channels >=2 and projected_feature_channels >=1 are required')
        self.base_channels = int(base_channels)
        self.projected_feature_channels = int(projected_feature_channels)
        self.feature_projections = nn.ModuleList([
            nn.Conv2d(channels, self.projected_feature_channels, 1)
            for channels in SAM_FEATURE_CHANNELS])
        self.input_stem = _ConvBlock(4, self.base_channels)
        fused_channels = self.base_channels + 3 * self.projected_feature_channels
        self.fusion = _ConvBlock(fused_channels, self.base_channels * 2)
        self.refine = _ConvBlock(self.base_channels * 2, self.base_channels)
        self.body_head = nn.Conv2d(self.base_channels, 1, 1)
        self.center_head = nn.Conv2d(self.base_channels, 1, 1)

    def forward(self, input_tensor, feature_pyramid):
        if input_tensor.ndim != 4 or input_tensor.shape[1] != 4:
            raise ValueError('Expected N x 4 x H x W input (RGB + human seed)')
        if not isinstance(feature_pyramid, (tuple, list)) or len(feature_pyramid) != 3:
            raise ValueError('Expected three frozen/zero feature levels')
        expected = expected_cropped_feature_shapes(input_tensor.shape[-2:])
        for index, (value, shape) in enumerate(zip(feature_pyramid, expected)):
            if value.ndim != 4 or value.shape[0] != input_tensor.shape[0] or tuple(value.shape[1:]) != shape:
                raise ValueError(f'Feature level {index} shape mismatch')
        high_hw = feature_pyramid[0].shape[-2:]
        seed_rgb = F.interpolate(input_tensor, size=high_hw, mode='bilinear', align_corners=False)
        pieces = [self.input_stem(seed_rgb)]
        for projection, feature in zip(self.feature_projections, feature_pyramid):
            projected = projection(feature)
            if projected.shape[-2:] != high_hw:
                projected = F.interpolate(projected, size=high_hw, mode='bilinear',
                                          align_corners=False)
            pieces.append(projected)
        hidden = self.refine(self.fusion(torch.cat(pieces, dim=1)))
        body = self.body_head(hidden)
        center = self.center_head(hidden)
        output_hw = input_tensor.shape[-2:]
        return {
            'body_logits': F.interpolate(body, size=output_hw, mode='bilinear', align_corners=False),
            'center_logits': F.interpolate(center, size=output_hw, mode='bilinear', align_corners=False),
        }


def build_head(base_channels=8, projected_feature_channels=8):
    return SeedConditionedFeatureHead(base_channels, projected_feature_channels)


def _state_dict_sha256(state_dict):
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(tensor.dtype).encode('ascii'))
        digest.update(json.dumps(list(tensor.shape), separators=(',', ':')).encode('ascii'))
        digest.update(tensor.numpy().tobytes(order='C'))
    return digest.hexdigest()


def initialize_head(seed, base_channels=8, projected_feature_channels=8):
    """Reset RNG after feature preparation so paired modes start identically."""
    torch.manual_seed(int(seed))
    model = build_head(base_channels, projected_feature_channels)
    return model, _state_dict_sha256(model.state_dict())


def training_order_generator(seed):
    """Dedicated sampler RNG; independent of SAM construction/global RNG."""
    return torch.Generator().manual_seed(int(seed) + 31)


def _batch_features(batch):
    return tuple(batch[f'feature_{index}'] for index in range(3))


def _batch_losses(model, batch, center_weight, forward_timer=None):
    started = time.perf_counter()
    outputs = model(batch['input'], _batch_features(batch))
    if forward_timer is not None:
        forward_timer['seconds'] += time.perf_counter() - started
    losses = masked_two_head_loss(
        outputs, batch['body_target'], batch['center_target'], None, center_weight,
        batch['body_valid_mask'], batch['center_valid_mask'])
    return outputs, losses


def _pixel_metrics(model, loader, center_weight, interpretation):
    totals = dict(body_loss_sum=0., center_loss_sum=0., body_intersection=0., body_sum=0.,
                  center_intersection=0., center_sum=0., body_valid_pixels=0.,
                  center_valid_pixels=0.)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            outputs, losses = _batch_losses(model, batch, center_weight)
            body_valid = batch['body_valid_mask'] > 0
            center_valid = batch['center_valid_mask'] > 0
            body_prediction = (torch.sigmoid(outputs['body_logits']) >= .5) & body_valid
            center_prediction = (torch.sigmoid(outputs['center_logits']) >= .5) & center_valid
            body_truth = (batch['body_target'] >= .5) & body_valid
            center_truth = (batch['center_target'] >= .5) & center_valid
            body_weight = float(losses['body_valid_weight'])
            center_pixels = float(losses['center_valid_weight'])
            totals['body_loss_sum'] += float(losses['body_loss']) * body_weight
            totals['center_loss_sum'] += float(losses['center_loss']) * center_pixels
            totals['body_intersection'] += float((body_prediction & body_truth).sum())
            totals['body_sum'] += float(body_prediction.sum() + body_truth.sum())
            totals['center_intersection'] += float((center_prediction & center_truth).sum())
            totals['center_sum'] += float(center_prediction.sum() + center_truth.sum())
            totals['body_valid_pixels'] += body_weight
            totals['center_valid_pixels'] += center_pixels
    def ratio(numerator, denominator):
        return None if denominator <= 0 else numerator / denominator
    body_loss = ratio(totals['body_loss_sum'], totals['body_valid_pixels'])
    center_loss = ratio(totals['center_loss_sum'], totals['center_valid_pixels'])
    combined = (None if body_loss is None and center_loss is None else
                (0. if body_loss is None else body_loss) +
                float(center_weight) * (0. if center_loss is None else center_loss))
    return {
        'masked_two_head_loss': combined, 'body_masked_loss': body_loss,
        'center_masked_loss': center_loss,
        'body_dice_at_0_5': ratio(2 * totals['body_intersection'], totals['body_sum']),
        'center_dice_at_0_5': ratio(2 * totals['center_intersection'], totals['center_sum']),
        'body_valid_pixels': int(totals['body_valid_pixels']),
        'center_valid_pixels': int(totals['center_valid_pixels']),
        'interpretation': interpretation,
    }


def _provider(args, target_hw, cache_dir=None, extractor=None):
    return FeatureProvider(args.feature_mode, target_hw,
                           cache_dir if cache_dir is not None else getattr(args, 'feature_cache_dir', None),
                           getattr(args, 'sam_checkpoint', DEFAULT_SAM_CHECKPOINT), extractor,
                           getattr(args, 'threads', None))


def _blocked_report(output, inspected, status):
    report = {
        'schema_version': 1, 'experiment_version': EXPERIMENT_VERSION, 'status': status,
        'manifest_sha256': inspected['manifest_sha256'],
        'skipped_samples': inspected['skipped'], 'artifacts': {'checkpoint': None},
        'claim_limits': {'field_accuracy_established': False, 'production_ready': False,
                         'fully_automatic': False, 'seed_assisted': True},
    }
    _write_json(output / 'report.json', report)
    return report


def train_manifest(args):
    command_started = time.perf_counter()
    torch.set_num_threads(int(args.threads))
    torch.manual_seed(int(args.seed))
    target_hw = (int(args.target_height), int(args.target_width))
    sam_embedding_geometry(target_hw, SAM_SQUARE_SIZE)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    inspected = inspect_manifest(args.manifest)
    train_records = inspected['usable']['train']
    holdout_records = inspected['usable']['holdout']
    if not train_records:
        return _blocked_report(output, inspected, 'blocked_no_reviewed_labels')
    cache_dir = (Path(args.feature_cache_dir).resolve() if args.feature_cache_dir else
                 output / 'rgb_only_feature_cache')
    provider = _provider(args, target_hw, cache_dir)
    feature_preparation_started = time.perf_counter()
    feature_provenance = provider.prepare([*train_records, *holdout_records])
    feature_preparation_seconds = time.perf_counter() - feature_preparation_started
    provider.release_extractor()
    dataset_started = time.perf_counter()
    training = TransferManifestDataset(train_records, provider, args.cache_samples)
    holdout = TransferManifestDataset(holdout_records, provider, args.cache_samples)
    dataset_cache_seconds = time.perf_counter() - dataset_started
    supervision = _supervision_summary(training)
    if not supervision['body_head_supervised'] and not supervision['center_head_supervised']:
        return _blocked_report(output, inspected, 'blocked_no_positive_target_supervision')
    loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        generator=training_order_generator(args.seed))
    train_metric_loader = DataLoader(training, batch_size=args.batch_size, shuffle=False,
                                     num_workers=0)
    holdout_loader = (DataLoader(holdout, batch_size=args.batch_size, shuffle=False,
                                 num_workers=0) if len(holdout) else None)
    # Loading/building SAM2 is allowed to consume torch's global RNG.  Reset it
    # here, after all feature preparation, so frozen-feature and strict-zero
    # runs with the same preregistered seed have byte-identical initial heads.
    model, initial_head_state_sha256 = initialize_head(
        args.seed, args.base_channels, args.projected_feature_channels)
    model = model.cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    iterator = iter(loader)
    model.train()
    loop_started = time.perf_counter()
    forward_timer = {'seconds': 0.}
    batch_fetch_seconds = 0.
    for step in range(int(args.steps)):
        fetch_started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch_fetch_seconds += time.perf_counter() - fetch_started
        optimizer.zero_grad(set_to_none=True)
        _, losses = _batch_losses(model, batch, args.center_weight, forward_timer)
        if float(losses['body_valid_weight'] + losses['center_valid_weight']) <= 0:
            raise ValueError('Eligible training batch unexpectedly has no supervised pixels')
        losses['loss'].backward()
        optimizer.step()
        history.append({
            'step': step + 1, 'loss': float(losses['loss'].detach()),
            'body_loss': float(losses['body_loss'].detach()),
            'center_loss': float(losses['center_loss'].detach()),
            'body_valid_pixels': int(losses['body_valid_weight']),
            'center_valid_pixels': int(losses['center_valid_weight']),
            'sample_ids': list(batch['sample_id']),
        })
    loop_seconds = time.perf_counter() - loop_started
    train_metrics = _pixel_metrics(
        model, train_metric_loader, args.center_weight,
        'post-training train split diagnostic; not an independent accuracy estimate')
    holdout_metrics = (_pixel_metrics(
        model, holdout_loader, args.center_weight,
        'fixed group-held-out diagnostic; not used for gradients, early stopping, or selection')
        if holdout_loader is not None else None)
    extractor_sha = _module_sha256()
    sam_source = None
    if args.feature_mode == 'sam2_frozen':
        sam_source = {
            'source_checkpoint': str(Path(args.sam_checkpoint).resolve()),
            'source_checkpoint_sha256': EXPECTED_SAM_CHECKPOINT_SHA256,
            'config_name': SAM_CONFIG_NAME,
            'config_sha256': EXPECTED_SAM_CONFIG_SHA256,
            'extractor_code_sha256': extractor_sha,
            'vendor_code_sha256': _vendor_code_sha256(),
            'backbone_trainable': False,
        }
    checkpoint_path = output / 'checkpoint.pt'
    temporary = output / 'checkpoint.pt.tmp'
    checkpoint = {
        'format': CHECKPOINT_FORMAT, 'schema_version': 1,
        'experiment_version': EXPERIMENT_VERSION,
        'preprocess_version': PREPROCESS_VERSION,
        'raster_encoding_version': RASTER_ENCODING_VERSION,
        'model_config': {
            'feature_mode': args.feature_mode, 'input_channels': 4,
            'base_channels': int(args.base_channels),
            'projected_feature_channels': int(args.projected_feature_channels),
            'sam_feature_channels': list(SAM_FEATURE_CHANNELS),
            'heads': list(OUTPUT_HEADS), 'input_size_hw': list(target_hw),
            'sam_square_size': SAM_SQUARE_SIZE,
        },
        'head_state_dict': model.state_dict(),
        'initial_head_state_sha256': initial_head_state_sha256,
        'head_parameter_count': int(sum(parameter.numel() for parameter in model.parameters())),
        'sam2_state_included': False, 'sam2': sam_source,
        'extractor_code_sha256': extractor_sha,
        'input_semantics': list(INPUT_SEMANTICS),
        'feature_input_semantics': 'rgb_only_no_seed_no_target',
        'output_semantics': {'body': 'target rope-body probability',
                             'center': 'target rope-centre heatmap'},
        'preprocess': {
            'name': 'v6_native_roi_fit_pad_then_embed_unchanged_in_sam_square',
            'padding_supervision': 'invalid', 'sam_embedding': provider.embedding,
            'probability_export': 'crop V6 model padding; content spans native ROI',
        },
        'training_source': 'reviewed_manifest_annotations',
        'training_manifest_sha256': inspected['manifest_sha256'],
        'training_annotation_sha256_by_sample': {
            record['sample_id']: record['annotation_sha256'] for record in train_records},
        'train_quality_sha256_by_sample': {
            record['sample_id']: record['quality_sha256'] for record in train_records},
        'train_group_ids': sorted({record['group_id'] for record in train_records}),
        'body_head_supervised': supervision['body_head_supervised'],
        'center_head_supervised': supervision['center_head_supervised'],
        'body_valid_sample_count': supervision['body_valid_sample_count'],
        'center_valid_sample_count': supervision['center_valid_sample_count'],
        'body_positive_sample_count': supervision['body_positive_sample_count'],
        'center_positive_sample_count': supervision['center_positive_sample_count'],
        'paired_control_contract': 'identical head shapes and initialization seed; only frozen features versus strict zeros',
        'initialization_seed': int(args.seed), 'field_validation_performed': False,
    }
    torch.save(checkpoint, temporary)
    os.replace(temporary, checkpoint_path)
    train_groups = sorted({record['group_id'] for record in train_records})
    holdout_groups = sorted({record['group_id'] for record in holdout_records})
    report = {
        'schema_version': 1, 'experiment_version': EXPERIMENT_VERSION,
        'status': ('manifest_training_complete_unvalidated'
                   if supervision['supervision_complete'] else 'partial_supervision'),
        'manifest': str(Path(args.manifest).resolve()),
        'manifest_sha256': inspected['manifest_sha256'],
        'feature_mode': args.feature_mode, 'preprocess_version': PREPROCESS_VERSION,
        'raster_encoding_version': RASTER_ENCODING_VERSION,
        'data': {
            'split_source': 'manifest group split; no random neighbouring-frame split',
            'train_sample_ids': [record['sample_id'] for record in train_records],
            'holdout_sample_ids': [record['sample_id'] for record in holdout_records],
            'train_groups': train_groups, 'holdout_groups': holdout_groups,
            'groups_disjoint': not bool(set(train_groups) & set(holdout_groups)),
            'holdout_used_for_training': False, 'holdout_used_for_early_stopping': False,
            'holdout_used_for_model_selection': False,
            'train_annotation_sha256_by_sample': checkpoint['training_annotation_sha256_by_sample'],
            'train_quality_sha256_by_sample': checkpoint['train_quality_sha256_by_sample'],
            'skipped_samples': inspected['skipped'], 'augmentation': None,
        },
        'head': {
            'name': 'seed_conditioned_multiscale_feature_head',
            'parameter_count': checkpoint['head_parameter_count'],
            'base_channels': int(args.base_channels),
            'projected_feature_channels': int(args.projected_feature_channels),
            'same_shape_for_both_feature_modes': True,
            'initial_head_state_sha256': initial_head_state_sha256,
            'initialization_seed_reset_after_feature_preparation': True,
        },
        'sam2': sam_source,
        'supervision': supervision,
        'training': {
            'device': 'cpu', 'steps_completed': int(args.steps),
            'loss': 'V6/V5 masked per-head binary cross entropy',
            'loop_seconds': loop_seconds, 'model_forward_seconds': forward_timer['seconds'],
            'batch_fetch_seconds': batch_fetch_seconds,
            'feature_preparation_seconds': feature_preparation_seconds,
            'sam2_uncached_forward_seconds': provider.extraction_seconds,
            'feature_cache_hits': provider.cache_hits, 'feature_cache_misses': provider.cache_misses,
            'cache_samples': bool(args.cache_samples),
            'dataset_cache_seconds': dataset_cache_seconds,
            'dataset_cache_bytes': training.cache_bytes() + holdout.cache_bytes(),
            'model_input_hw': list(target_hw), 'sam_square_hw': [SAM_SQUARE_SIZE] * 2,
            'batch_size': int(args.batch_size), 'learning_rate': float(args.learning_rate),
            'center_weight': float(args.center_weight), 'seed': int(args.seed),
            'data_order_seed': int(args.seed) + 31,
            'threads': int(args.threads), 'loss_history': history,
        },
        'feature_provenance_by_sample': feature_provenance,
        'train_pixel_diagnostic': train_metrics,
        'holdout_pixel_diagnostic': holdout_metrics,
        'artifacts': {'checkpoint': str(checkpoint_path),
                      'checkpoint_sha256': _file_sha256(checkpoint_path)},
        'command_elapsed_before_report_seconds': time.perf_counter() - command_started,
        'claim_limits': {
            'field_accuracy_established': False, 'production_ready': False,
            'fully_automatic': False, 'seed_assisted': True,
            'pretrained_features_are_task_capability_proof': False,
        },
    }
    _write_json(output / 'report.json', report)
    print(json.dumps({'report': str(output / 'report.json'), 'checkpoint': str(checkpoint_path),
                      'status': report['status'], 'loop_seconds': loop_seconds}))
    return report


def load_head_checkpoint(checkpoint_path):
    checkpoint = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=True)
    if checkpoint.get('format') != CHECKPOINT_FORMAT or checkpoint.get('experiment_version') != 7:
        raise ValueError('Incompatible checkpoint: V7 rejects V5/V6 and unrelated checkpoint formats')
    if checkpoint.get('preprocess_version') != PREPROCESS_VERSION:
        raise ValueError('Incompatible V7 preprocess_version')
    if checkpoint.get('raster_encoding_version') != RASTER_ENCODING_VERSION:
        raise ValueError('Incompatible line-map raster encoding')
    config = checkpoint.get('model_config', {})
    if (config.get('feature_mode') not in FEATURE_MODES or config.get('input_channels') != 4
            or config.get('heads') != list(OUTPUT_HEADS)
            or config.get('sam_feature_channels') != list(SAM_FEATURE_CHANNELS)):
        raise ValueError('V7 checkpoint input/feature/head semantics mismatch')
    target_hw = _positive_hw(config.get('input_size_hw'), 'checkpoint model_config.input_size_hw')
    if config.get('sam_square_size') != SAM_SQUARE_SIZE:
        raise ValueError('V7 checkpoint SAM square mismatch')
    if checkpoint.get('extractor_code_sha256') != _module_sha256():
        raise ValueError('V7 extractor code changed since training; retrain/re-export under a frozen protocol')
    model = build_head(int(config.get('base_channels', 0)),
                       int(config.get('projected_feature_channels', 0))).cpu()
    model.load_state_dict(checkpoint['head_state_dict'], strict=True)
    model.eval()
    return model, checkpoint, target_hw


def _predict(model, input_tensor, features):
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model(input_tensor.unsqueeze(0), tuple(value.unsqueeze(0) for value in features))
        body = torch.sigmoid(outputs['body_logits'])[0, 0].numpy().astype(np.float32, copy=False)
        center = torch.sigmoid(outputs['center_logits'])[0, 0].numpy().astype(np.float32, copy=False)
    return body, center, (time.perf_counter() - started) * 1000.


def _write_prediction(path, body, center):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, body_probability=body.astype(np.float32, copy=False),
                        center_probability=center.astype(np.float32, copy=False))
    return path


def infer_manifest(args):
    torch.set_num_threads(int(args.threads))
    model, checkpoint, target_hw = load_head_checkpoint(args.checkpoint)
    feature_mode = checkpoint['model_config']['feature_mode']
    if args.feature_mode is not None and args.feature_mode != feature_mode:
        raise ValueError('Requested feature_mode does not match the trained V7 checkpoint')
    args.feature_mode = feature_mode
    inspected = inspect_manifest(args.manifest)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = [record for record in inspected['inference_records']
               if args.split == 'all' or record['split'] == args.split]
    cache_dir = (Path(args.feature_cache_dir).resolve() if args.feature_cache_dir else
                 output / 'rgb_only_feature_cache')
    provider = _provider(args, target_hw, cache_dir)
    provider.prepare(records)
    provider.release_extractor()
    checkpoint_hash = _file_sha256(args.checkpoint)
    exported = []
    for record in records:
        started = time.perf_counter()
        input_tensor, geometry, rgb, seed = letterbox_manifest_input(record, target_hw)
        input_path = output / 'inputs' / f"{record['sample_id']}.npz"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(input_path, rgb=rgb.astype(np.float32, copy=False),
                            seed_heatmap=seed.astype(np.float32, copy=False))
        feature_started = time.perf_counter()
        features, feature_provenance = provider.get(record, input_tensor)
        feature_access_ms = (time.perf_counter() - feature_started) * 1000.
        body_padded, center_padded, head_ms = _predict(model, input_tensor, features)
        body = crop_letterbox_probabilities(body_padded, geometry)
        center = crop_letterbox_probabilities(center_padded, geometry)
        prediction_path = _write_prediction(
            output / 'predictions' / f"{record['sample_id']}.npz", body, center)
        metadata = {
            'schema_version': 1, 'experiment_version': EXPERIMENT_VERSION,
            'raster_encoding_version': RASTER_ENCODING_VERSION,
            'preprocess_version': PREPROCESS_VERSION,
            'status': 'unvalidated_model_inference', 'feature_mode': feature_mode,
            'feature_provenance': feature_provenance,
            'sample_id': record['sample_id'], 'group_id': record['group_id'],
            'manifest_split': record['split'], 'image_sha256': record['image_sha256'],
            'annotation_sha256': record['annotation_sha256'],
            'free_seed_sha256': record['free_seed_sha256'],
            'quality_sha256': record['quality_sha256'], 'roi_xyxy': record['roi_xyxy'],
            'native_roi_hw': [int(rgb.shape[0]), int(rgb.shape[1])],
            'model_input_hw': list(target_hw), 'probability_map_hw': list(body.shape),
            'scale_xy': geometry['scale_xy'], 'letterbox': geometry,
            'sam_embedding': provider.embedding,
            'probability_arrays': {
                'body_probability': 'unpadded V6 aspect-preserving content spanning native ROI',
                'center_probability': 'unpadded V6 aspect-preserving content spanning native ROI'},
            'input_npz': str(input_path), 'input_sha256': _file_sha256(input_path),
            'output_npz': str(prediction_path), 'output_sha256': _file_sha256(prediction_path),
            'checkpoint': str(Path(args.checkpoint).resolve()),
            'checkpoint_sha256': checkpoint_hash,
            'trained_checkpoint': str(Path(args.checkpoint).resolve()),
            'trained_checkpoint_sha256': checkpoint_hash,
            'inference_ms': feature_access_ms + head_ms,
            'feature_access_ms': feature_access_ms, 'head_forward_ms': head_ms,
            'total_elapsed_seconds': time.perf_counter() - started,
            'architecture': 'seed_conditioned_multiscale_feature_head',
            'train_group_ids': checkpoint.get('train_group_ids', []),
            'train_quality_sha256_by_sample': checkpoint.get('train_quality_sha256_by_sample', {}),
            'human_free_seed_used': True, 'free_seed_source': 'human_polyline',
            'free_seed_point_count': int(len(record['parsed_polylines']['free_seed'])),
            'ground_truth_center_or_edges_used_as_input': False,
            'checkpoint_training_source': checkpoint.get('training_source', 'unknown'),
            'real_annotation_trained': checkpoint.get('training_source') == 'reviewed_manifest_annotations',
            'body_head_supervised': bool(checkpoint.get('body_head_supervised', False)),
            'center_head_supervised': bool(checkpoint.get('center_head_supervised', False)),
            'body_valid_sample_count': int(checkpoint.get('body_valid_sample_count', 0)),
            'center_valid_sample_count': int(checkpoint.get('center_valid_sample_count', 0)),
            'supervision_complete': bool(checkpoint.get('body_head_supervised', False)
                                         and checkpoint.get('center_head_supervised', False)),
            'field_accuracy_established': False, 'hardware_access': False,
            'force_command': None,
        }
        metadata_path = prediction_path.with_suffix('.json')
        _write_json(metadata_path, metadata)
        exported.append({'sample_id': record['sample_id'], 'split': record['split'],
                         'prediction': str(prediction_path), 'metadata': str(metadata_path)})
    selected = [sample for sample in inspected['manifest']['samples']
                if args.split == 'all' or sample['split'] == args.split]
    exported_ids = {row['sample_id'] for row in exported}
    reason_by_id = {row['sample_id']: row['reason'] for row in inspected['skipped']}
    skipped = [{'sample_id': sample['sample_id'], 'split': sample['split'],
                'reason': reason_by_id.get(sample['sample_id'], 'no_reviewed_free_seed')}
               for sample in selected if sample['sample_id'] not in exported_ids]
    summary = {
        'schema_version': 1, 'experiment_version': EXPERIMENT_VERSION,
        'status': 'complete' if exported else 'blocked_no_reviewed_free_seed',
        'feature_mode': feature_mode,
        'raster_encoding_version': RASTER_ENCODING_VERSION,
        'preprocess_version': PREPROCESS_VERSION,
        'manifest': str(Path(args.manifest).resolve()),
        'manifest_sha256': inspected['manifest_sha256'], 'requested_split': args.split,
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'checkpoint_sha256': checkpoint_hash,
        'checkpoint_training_source': checkpoint.get('training_source', 'unknown'),
        'architecture': 'seed_conditioned_multiscale_feature_head',
        'model_input_hw': list(target_hw), 'exported': exported, 'skipped': skipped,
        'ground_truth_center_or_edges_used_as_input': False,
        'hardware_access': False, 'force_command': None,
    }
    _write_json(output / 'summary.json', summary)
    print(json.dumps({'summary': str(output / 'summary.json'), 'exported': len(exported),
                      'skipped': len(skipped), 'status': summary['status']}))
    return summary


def probe_sam2(args):
    torch.set_num_threads(int(args.threads))
    inspected = inspect_manifest(args.manifest)
    records = inspected['inference_records']
    if args.sample_id:
        records = [record for record in records if record['sample_id'] == args.sample_id]
    if not records:
        raise ValueError('No reviewed free-seed sample found for probe')
    record = records[0]
    input_tensor, _, _, _ = letterbox_manifest_input(record, DEFAULT_TARGET_HW)
    extractor = FrozenSam2Extractor(args.sam_checkpoint, args.threads)
    features, stats = extractor.extract(input_tensor)
    report = {
        'schema_version': 1, 'experiment_version': EXPERIMENT_VERSION,
        'sample_id': record['sample_id'], 'device': 'cpu',
        'checkpoint': str(extractor.checkpoint),
        'checkpoint_sha256': extractor.checkpoint_sha256,
        'config_sha256': extractor.config_sha256,
        'parameter_count': extractor.parameter_count,
        'trainable_parameter_count': extractor.trainable_parameter_count,
        'load_seconds': extractor.load_seconds, **stats,
        'features_require_grad': [bool(value.requires_grad) for value in features],
        'claim': 'single-image engineering timing only; not field accuracy',
    }
    if args.output:
        _write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train-manifest', help='Train one frozen-feature/control head')
    train.add_argument('--manifest', required=True)
    train.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    train.add_argument('--feature-mode', choices=FEATURE_MODES, required=True)
    train.add_argument('--sam-checkpoint', default=str(DEFAULT_SAM_CHECKPOINT))
    train.add_argument('--feature-cache-dir')
    train.add_argument('--steps', type=int, default=60)
    train.add_argument('--target-height', type=int, default=DEFAULT_TARGET_HW[0])
    train.add_argument('--target-width', type=int, default=DEFAULT_TARGET_HW[1])
    train.add_argument('--base-channels', type=int, default=8)
    train.add_argument('--projected-feature-channels', type=int, default=8)
    train.add_argument('--batch-size', type=int, default=2)
    train.add_argument('--learning-rate', type=float, default=1e-3)
    train.add_argument('--center-weight', type=float, default=1.)
    train.add_argument('--seed', type=int, default=20260926)
    train.add_argument('--threads', type=int, default=2)
    train.add_argument('--cache-samples', action='store_true')
    train.set_defaults(action=train_manifest)
    infer = commands.add_parser('infer-manifest', help='Export V7 probabilities')
    infer.add_argument('--manifest', required=True)
    infer.add_argument('--checkpoint', required=True)
    infer.add_argument('--output-dir', required=True)
    infer.add_argument('--feature-mode', choices=FEATURE_MODES)
    infer.add_argument('--sam-checkpoint', default=str(DEFAULT_SAM_CHECKPOINT))
    infer.add_argument('--feature-cache-dir')
    infer.add_argument('--split', choices=('all', 'train', 'holdout'), default='all')
    infer.add_argument('--threads', type=int, default=2)
    infer.set_defaults(action=infer_manifest)
    probe = commands.add_parser('probe-sam2', help='Read-only one-image feature shape/timing probe')
    probe.add_argument('--manifest', required=True)
    probe.add_argument('--sample-id')
    probe.add_argument('--sam-checkpoint', default=str(DEFAULT_SAM_CHECKPOINT))
    probe.add_argument('--threads', type=int, default=2)
    probe.add_argument('--output')
    probe.set_defaults(action=probe_sam2)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    for name in ('steps', 'target_height', 'target_width', 'base_channels',
                 'projected_feature_channels', 'batch_size', 'threads'):
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise ValueError(f'{name} must be positive')
    if hasattr(args, 'learning_rate') and (not math.isfinite(args.learning_rate)
                                           or args.learning_rate <= 0):
        raise ValueError('learning_rate must be finite and positive')
    if hasattr(args, 'center_weight') and (not math.isfinite(args.center_weight)
                                          or args.center_weight < 0):
        raise ValueError('center_weight must be finite and non-negative')
    return args.action(args)


if __name__ == '__main__':
    main()
