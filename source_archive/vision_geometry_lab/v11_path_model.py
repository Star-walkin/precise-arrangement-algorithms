"""V11 spatial strip model and masked path loss; no dataset/hardware access.

Inputs: RGB, normalized gray, seed heatmap, normalized u/v, image validity.
Unlike V9, the U-Net retains aligned two-dimensional context. A bottleneck
horizontal dilated-convolution stack propagates context across the strip.
No visibility head or global pooled descriptor is used.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


MODEL_VERSION = 1
INPUT_CHANNELS = 8
MAX_SLOPE = 3.0
SLOPE_LOSS_WEIGHT = .1


class _SpatialBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(math.gcd(4, output_channels), output_channels), nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(math.gcd(4, output_channels), output_channels), nn.SiLU())

    def forward(self, inputs):
        return self.layers(inputs)


class _HorizontalContext(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, (1, 3), padding=(0, dilation),
                          dilation=(1, dilation), groups=channels, bias=False),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.GroupNorm(math.gcd(4, channels), channels), nn.SiLU())
            for dilation in (1, 2, 4, 8)])

    def forward(self, inputs):
        for layer in self.layers:
            inputs = inputs + layer(inputs)
        return inputs


def _checked_binary_mask(mask, shape, name, *, device):
    if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != tuple(shape):
        raise ValueError(f'{name} must have shape {tuple(shape)}')
    if mask.device != device:
        raise ValueError(f'{name} must be on the logits device')
    if not torch.isfinite(mask).all() or torch.any((mask != 0) & (mask != 1)):
        raise ValueError(f'{name} must be a finite binary mask')
    return mask.bool()


def _masked_log_softmax(logits, row_valid, column_valid=None):
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or not logits.is_floating_point():
        raise ValueError('center_logits must be floating N x H x W')
    rows = _checked_binary_mask(row_valid, logits.shape, 'row_valid', device=logits.device)
    if column_valid is not None:
        columns = _checked_binary_mask(column_valid, (logits.shape[0], logits.shape[2]),
                                        'column_valid', device=logits.device)
        rows = rows & columns[:, None, :]
    else:
        columns = rows.any(dim=1)
    observed = logits[rows]
    if observed.numel() and not torch.isfinite(observed).all():
        raise ValueError('center_logits must be finite wherever valid')
    # Sanitize unknown logits BEFORE log-softmax; NaN/Inf in unknown rows must
    # not leak into loss or gradients. Finite minimum keeps all-empty columns safe.
    safe = torch.where(rows, logits, torch.zeros_like(logits))
    safe = safe.masked_fill(~rows, torch.finfo(logits.dtype).min)
    log_probabilities = F.log_softmax(safe, dim=1)
    log_probabilities = torch.where(rows, log_probabilities, torch.zeros_like(log_probabilities))
    columns = columns & rows.any(dim=1)
    return log_probabilities, rows, columns


def mask_normalized_probabilities(outputs, batch_or_row_valid, column_valid=None):
    """Return N x H x W column probabilities; no valid rows => exact zeros.

For real inference pass IMAGE validity rows, not a GT supervision batch.
Passing a batch uses row_valid/column_valid and is for supervised diagnostics.
Unknown logits have exactly zero gradient, even when they contain NaN/Inf.
"""
    logits = outputs['center_logits'] if isinstance(outputs, dict) else outputs
    if isinstance(batch_or_row_valid, dict):
        row_valid = batch_or_row_valid['row_valid']
        if column_valid is None:
            column_valid = batch_or_row_valid.get('column_valid')
    else:
        row_valid = batch_or_row_valid
    logp, rows, _ = _masked_log_softmax(logits, row_valid, column_valid)
    return torch.where(rows, torch.exp(logp), torch.zeros_like(logp))


class SpatialStripPathModel(nn.Module):
    def __init__(self, base_channels=8):
        super().__init__()
        self.base_channels = int(base_channels)
        if self.base_channels < 2:
            raise ValueError('base_channels must be at least 2')
        channels = [self.base_channels * (2 ** level) for level in range(5)]
        self.encoders = nn.ModuleList([
            _SpatialBlock(INPUT_CHANNELS if level == 0 else channels[level-1], channels[level])
            for level in range(5)])
        self.context = _HorizontalContext(channels[-1])
        self.decoders = nn.ModuleList([
            _SpatialBlock(channels[level+1] + channels[level], channels[level])
            for level in (3, 2, 1, 0)])
        self.center_head = nn.Conv2d(channels[0], 1, 1)
        self.dense_slope_head = nn.Conv2d(channels[0], 1, 1)

    def forward(self, inputs):
        if (not isinstance(inputs, torch.Tensor) or inputs.ndim != 4
                or inputs.shape[1] != INPUT_CHANNELS or min(inputs.shape[-2:]) < 16):
            raise ValueError('inputs must be N x 8 x H x W with H,W >=16')
        if not inputs.is_floating_point() or not torch.isfinite(inputs).all():
            raise ValueError('inputs must be finite floating tensors')
        # Image validity is an input measurement, never the annotation mask.
        image_rows = inputs[:, 7] > .5
        skips = []
        value = inputs
        for level, encoder in enumerate(self.encoders):
            if level:
                value = F.max_pool2d(value, 2, ceil_mode=True)
            value = encoder(value)
            skips.append(value)
        value = self.context(value)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            value = F.interpolate(value, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            value = decoder(torch.cat((value, skip), dim=1))
        center_logits = self.center_head(value)[:, 0]
        dense_slope = MAX_SLOPE * torch.tanh(self.dense_slope_head(value)[:, 0])
        # Detach routing weights: the slope loss must not send gradients into
        # center logits in rows whose center annotation is unknown.
        weights = mask_normalized_probabilities(center_logits.detach(), image_rows)
        slope = (dense_slope * weights).sum(dim=1)
        return dict(center_logits=center_logits, slope=slope)


def build_model(seed, base_channels=8):
    """Deterministic CPU initialization without consuming the caller RNG state."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return SpatialStripPathModel(base_channels)


def path_loss(outputs, batch):
    """Soft column CE + 0.1 masked SmoothL1 slope, with explicit unknown masks.

    Required batch keys: target_distribution (distribution alias accepted),
    column_valid, row_valid, slope_target, slope_valid. Targets are normalized
    over only valid rows per supervised column. Invalid columns/rows can carry
    arbitrary target values including NaN; they contribute no gradients.
    """
    logits, predicted_slope = outputs['center_logits'], outputs['slope']
    target = batch.get('target_distribution', batch.get('distribution'))
    if not isinstance(target, torch.Tensor) or target.shape != logits.shape or target.device != logits.device:
        raise ValueError('target_distribution must match center_logits')
    logp, rows, active_columns = _masked_log_softmax(logits, batch['row_valid'], batch['column_valid'])
    requested_columns = batch['column_valid'].bool()
    if torch.any(requested_columns & ~active_columns):
        raise ValueError('supervised column has no valid rows')
    safe_target = torch.where(rows, target, torch.zeros_like(target)).to(logits.dtype)
    observed_target = safe_target[rows]
    if observed_target.numel() and (not torch.isfinite(observed_target).all() or torch.any(observed_target < 0)):
        raise ValueError('target_distribution must be finite and nonnegative wherever valid')
    target_mass = safe_target.sum(dim=1)
    if torch.any(active_columns & (target_mass <= 0)):
        raise ValueError('supervised column has zero target mass')
    normalized_target = safe_target / target_mass.clamp_min(1e-12)[:, None, :]
    column_count = active_columns.sum()
    center_loss = -(normalized_target * logp).sum() / column_count.clamp_min(1)
    expected_slope_shape = (logits.shape[0], logits.shape[2])
    if not isinstance(predicted_slope, torch.Tensor) or tuple(predicted_slope.shape) != expected_slope_shape:
        raise ValueError('slope must be N x W')
    slope_valid = _checked_binary_mask(batch['slope_valid'], expected_slope_shape,
                                       'slope_valid', device=logits.device)
    slope_target = batch['slope_target']
    if not isinstance(slope_target, torch.Tensor) or slope_target.shape != predicted_slope.shape or slope_target.device != logits.device:
        raise ValueError('slope_target must match slope')
    if slope_valid.any() and (not torch.isfinite(slope_target[slope_valid]).all()
                            or not torch.isfinite(predicted_slope[slope_valid]).all()):
        raise ValueError('slope and target must be finite wherever slope_valid')
    safe_prediction = torch.where(slope_valid, predicted_slope, torch.zeros_like(predicted_slope))
    safe_slope_target = torch.where(slope_valid, slope_target, torch.zeros_like(slope_target)).to(predicted_slope.dtype)
    slope_count = slope_valid.sum()
    slope_loss = F.smooth_l1_loss(safe_prediction, safe_slope_target, reduction='sum', beta=1.) / slope_count.clamp_min(1)
    total = center_loss + SLOPE_LOSS_WEIGHT * slope_loss
    return dict(loss=total, center_loss=center_loss, column_ce_loss=center_loss, slope_loss=slope_loss,
                valid_columns=column_count.detach(), valid_slope_columns=slope_count.detach(),
                slope_weight=SLOPE_LOSS_WEIGHT)


def architecture_metadata(base_channels=8):
    return dict(name='spatial_strip_unet', version=MODEL_VERSION, input_channels=INPUT_CHANNELS,
                base_channels=int(base_channels), downsample_stages=4,
                horizontal_bottleneck_dilations=[1, 2, 4, 8], global_pooling=False,
                center_output='N x H x W logits, softmax over H per column',
                slope_output='N x W, center-weighted bounded dense slope, weights detached',
                max_abs_slope=MAX_SLOPE, slope_loss_weight=SLOPE_LOSS_WEIGHT,
                visibility_head=False, pretrained_weights=None)
