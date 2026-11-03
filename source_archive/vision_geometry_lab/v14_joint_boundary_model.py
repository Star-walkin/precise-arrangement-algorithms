"""V14 single-factor auxiliary two-boundary supervision on the V11 model.

No training/data/image/GT file access is implemented here. Center and slope
outputs and their loss are unchanged. Edge channels are independent annotation
side labels, not validated physical boundaries or rope-instance identities.
Unknown boundary support is masked out, never treated as a negative example.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .v11_path_model import (
    INPUT_CHANNELS, MAX_SLOPE, SpatialStripPathModel,
    _checked_binary_mask, _masked_log_softmax,
    mask_normalized_probabilities, path_loss,
    architecture_metadata as baseline_architecture_metadata,
)


EDGE_LOSS_WEIGHT = .5
PARAMETERS = dict(edge_loss_weight=EDGE_LOSS_WEIGHT,
    edge_loss='soft masked column cross-entropy; average over all valid side/columns',
    edge_channels=2, width_regression_loss=False, equal_width_constraint=False,
    center_midpoint_constraint=False, edge_order_constraint=False,
    center_decoder_changed=False, center_slope_baseline='V11 SpatialStripPathModel',
    synthetic_without_edge_labels='all edge masks false; no fabricated synthetic boundaries',
    inference_mask='image validity only, never annotation support',
    physical_boundary_confirmed=False, physical_center_confirmed=False,
    rope_identity_confirmed=False, visibility_head=False, hardware_control=False)


class JointBoundaryStripPathModel(SpatialStripPathModel):
    """Identical V11 initialization first; only then initialize an edge head."""

    def __init__(self, base_channels=8):
        super().__init__(base_channels)
        self.edge_head = nn.Conv2d(self.base_channels, 2, 1)

    def forward(self, inputs):
        # Same ordered floating-point operations as the frozen V11 forward.
        # No new head is evaluated until BOTH unchanged primary outputs exist.
        if (not isinstance(inputs, torch.Tensor) or inputs.ndim != 4
                or inputs.shape[1] != INPUT_CHANNELS or min(inputs.shape[-2:]) < 16):
            raise ValueError('inputs must be N x 8 x H x W with H,W >=16')
        if not inputs.is_floating_point() or not torch.isfinite(inputs).all():
            raise ValueError('inputs must be finite floating tensors')
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
        weights = mask_normalized_probabilities(center_logits.detach(), image_rows)
        slope = (dense_slope * weights).sum(dim=1)
        edge_logits = self.edge_head(value)
        return dict(center_logits=center_logits, slope=slope, edge_logits=edge_logits)


def build_model(seed, base_channels=8):
    """Same-seed shared tensors match V11 bit-for-bit, caller RNG unchanged."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return JointBoundaryStripPathModel(base_channels)


def load_shared_baseline(model, baseline_state_dict):
    """Load only an exact V11 shared state mapping; leave the new head intact."""
    if not isinstance(model, JointBoundaryStripPathModel):
        raise ValueError('model must be JointBoundaryStripPathModel')
    current = model.state_dict()
    expected = {key for key in current if not key.startswith('edge_head.')}
    supplied = set(baseline_state_dict)
    if supplied != expected:
        raise ValueError('baseline state keys must exactly equal the V11 shared keys')
    for key in expected:
        value = baseline_state_dict[key]
        if not isinstance(value, torch.Tensor) or value.shape != current[key].shape:
            raise ValueError('baseline state tensor shape mismatch: '+key)
    result = model.load_state_dict(baseline_state_dict, strict=False)
    if set(result.missing_keys) != {'edge_head.weight', 'edge_head.bias'} or result.unexpected_keys:
        raise RuntimeError('unexpected shared baseline loading result')
    return dict(shared_tensors_loaded=len(expected), edge_head_reinitialized=False,
                missing_new_head_keys=list(result.missing_keys))


def _checked_edge_logits(outputs):
    logits = outputs.get('edge_logits')
    center = outputs['center_logits']
    expected = (center.shape[0], 2, center.shape[1], center.shape[2])
    if (not isinstance(logits, torch.Tensor) or tuple(logits.shape) != expected
            or not logits.is_floating_point() or logits.device != center.device):
        raise ValueError('edge_logits must be floating N x 2 x H x W on the center device')
    return logits


def _edge_column_ce(logits, batch):
    target = batch.get('edge_target_distribution')
    if not isinstance(target, torch.Tensor) or target.shape != logits.shape or target.device != logits.device:
        raise ValueError('edge_target_distribution must match edge_logits')
    rows = _checked_binary_mask(batch['edge_row_valid'], logits.shape,
                                'edge_row_valid', device=logits.device)
    column_shape = (logits.shape[0], 2, logits.shape[3])
    columns = _checked_binary_mask(batch['edge_column_valid'], column_shape,
                                   'edge_column_valid', device=logits.device)
    n, _, height, width = logits.shape
    flat_logits = logits.reshape(n*2, height, width)
    logp, active_rows, active_columns = _masked_log_softmax(
        flat_logits, rows.reshape(n*2, height, width), columns.reshape(n*2, width))
    if torch.any(columns.reshape(n*2, width) & ~active_columns):
        raise ValueError('supervised edge column has no valid rows')
    safe_target = torch.where(active_rows, target.reshape(n*2, height, width),
                              torch.zeros_like(flat_logits)).to(logits.dtype)
    observed_target = safe_target[active_rows]
    if observed_target.numel() and (not torch.isfinite(observed_target).all() or torch.any(observed_target < 0)):
        raise ValueError('edge targets must be finite and nonnegative wherever supervised')
    target_mass = safe_target.sum(dim=1)
    if torch.any(active_columns & (target_mass <= 0)):
        raise ValueError('supervised edge column has zero target mass')
    normalized_target = safe_target/target_mass.clamp_min(1e-12)[:, None, :]
    count = active_columns.sum()
    loss = -(normalized_target*logp).sum()/count.clamp_min(1)
    by_side = active_columns.reshape(n, 2, width).sum(dim=(0, 2))
    return loss, count.detach(), by_side.detach()


def joint_path_loss(outputs, batch, *, edge_weight=EDGE_LOSS_WEIGHT):
    """Original V11 path loss plus 0.5 auxiliary independent-edge column CE.

    At weight zero the original loss tensor is returned unchanged and edge
    targets are not inspected. Otherwise all three edge target/mask fields
    must be supplied. Unknown logits/targets may be NaN: invalid support has
    exactly zero gradient, and an all-false edge batch adds exactly zero.
    No missing synthetic boundaries are invented from centerline labels.
    """
    weight = float(edge_weight)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('edge_weight must be finite and nonnegative')
    result = path_loss(outputs, batch)
    result['primary_path_loss'] = result['loss']
    if weight == 0:
        edge_loss = result['loss'].new_zeros(())
        count = torch.zeros((), dtype=torch.long, device=result['loss'].device)
        by_side = torch.zeros(2, dtype=torch.long, device=result['loss'].device)
        result.update(edge_ce_loss=edge_loss, valid_edge_columns=count,
                      valid_edge_columns_by_side=by_side, edge_weight=weight)
        return result
    logits = _checked_edge_logits(outputs)
    edge_loss, count, by_side = _edge_column_ce(logits, batch)
    primary_loss = result['loss']
    result.update(loss=primary_loss+weight*edge_loss, primary_path_loss=primary_loss,
        edge_ce_loss=edge_loss, valid_edge_columns=count,
        valid_edge_columns_by_side=by_side, edge_weight=weight)
    return result


def image_masked_edge_probabilities(outputs, image_row_valid):
    """Normalize each side over H using IMAGE support only, never a GT batch.

    Supply an explicit N x H x W image-validity tensor, normally inputs[:,7].
    This signature intentionally does not accept a batch or annotation mask
    dictionary. Both edge channels receive exactly the same image support.
    """
    logits = _checked_edge_logits(outputs)
    n, _, height, width = logits.shape
    image_rows = _checked_binary_mask(image_row_valid, (n, height, width),
                                      'image_row_valid', device=logits.device)
    rows = image_rows[:, None].expand(-1, 2, -1, -1).reshape(n*2, height, width)
    probability = mask_normalized_probabilities(logits.reshape(n*2, height, width), rows)
    return probability.reshape(n, 2, height, width)


def architecture_metadata(base_channels=8):
    result = baseline_architecture_metadata(base_channels)
    result.update(name='v14_v11_trunk_auxiliary_two_boundaries',
        baseline_architecture='spatial_strip_unet_v11', auxiliary_model_version=1,
        edge_output='N x 2 x H x W logits; independent H softmax per side/column',
        edge_channel_order='caller side labels, no implied physical validation',
        shared_initialization='bit-identical to V11 with the same seed/base_channels',
        width_output='not regressed; any width is a diagnostic derived from independent edges',
        parameters=PARAMETERS.copy(), physical_geometry_confirmed=False,
        visibility_head=False, no_physical_event_output=True, no_equipment_commands=True)
    return result
