"""V23 local observability mechanism. No identity, event, or hardware claims.

The frozen V11 conditional row distribution is never re-normalized after gating.
A zero query is a missing-query contract, not evidence of physical absence.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

FEATURE_CHANNELS = 30


def local_features(dense, probability, inputs):
    """GT-free frozen-feature summaries: [N,30,W]."""
    if dense.ndim != 4 or dense.shape[1] != 8:
        raise ValueError('Expected eight frozen decoder channels')
    if probability.shape != dense.shape[:1] + dense.shape[2:] or inputs.shape[1] != 8:
        raise ValueError('Shape mismatch')
    valid = inputs[:, 7] > .5
    weight = valid.to(dense.dtype)
    denom = weight.sum(dim=1).clamp_min(1)
    weighted = (dense * probability[:, None]).sum(dim=2)
    mean = (dense * weight[:, None]).sum(dim=2) / denom[:, None]
    maximum = dense.masked_fill(~valid[:, None], -1e4).amax(dim=2)
    maximum = torch.where(valid.any(dim=1)[:, None], maximum, torch.zeros_like(maximum))
    entropy = -(probability * probability.clamp_min(1e-9).log()).sum(dim=1)
    entropy /= np.log(dense.shape[2])
    gray = inputs[:, :3].mean(dim=1)
    gray_mean = (gray * weight).sum(dim=1) / denom
    gray_std = (((gray - gray_mean[:, None]) ** 2 * weight).sum(dim=1) / denom).sqrt()
    normalized_abs = (inputs[:, 3].abs() * weight).sum(dim=1) / denom
    query = inputs[:, 4].amax(dim=(1, 2))[:, None].expand(-1, inputs.shape[-1])
    extra = torch.stack((entropy, probability.amax(dim=1), gray_mean, gray_std,
                         normalized_abs, query), dim=1)
    result = torch.cat((weighted, mean, maximum, extra), dim=1).detach()
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite features')
    return result


class VisibilityHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv1d(FEATURE_CHANNELS, 24, 5, padding=2),
            nn.SiLU(), nn.Conv1d(24, 16, 5, padding=2), nn.SiLU(), nn.Conv1d(16, 1, 1))

    def forward(self, features):
        if features.ndim != 3 or features.shape[1] != FEATURE_CHANNELS:
            raise ValueError('Expected N x 30 x W')
        return self.layers(features)[:, 0]


def gate_probabilities(probability, observable, threshold=.5):
    if observable.shape != probability.shape[:1] + probability.shape[2:]:
        raise ValueError('Gate shape mismatch')
    if not torch.isfinite(probability).all() or not torch.isfinite(observable).all():
        raise ValueError('Nonfinite probabilities')
    if (probability < 0).any() or (probability > 1).any() or (observable < 0).any() or (observable > 1).any():
        raise ValueError('Probabilities outside [0,1]')
    # No distribution renormalization: low evidence must remain low mass.
    return probability * observable[:, None] * (observable >= threshold)[:, None]


def masked_loss(logits, target, supervised):
    if logits.shape != target.shape or logits.shape != supervised.shape:
        raise ValueError('Loss shape mismatch')
    mask = supervised.bool()
    if not mask.any():
        raise ValueError('No supervised columns')
    if not torch.isfinite(logits[mask]).all() or not torch.isfinite(target[mask]).all():
        raise ValueError('Nonfinite supervised loss')
    return nn.functional.binary_cross_entropy_with_logits(logits[mask], target[mask])


def intervention(inputs, kind, seed=0, *, heldout=False):
    """Change query OR synthetic RGB occlusion; coordinates/crop never change.

    Synthetic strips erase all rows at deterministic randomly chosen columns,
    independent of target. These are not realistic glove visibility labels.
    """
    source = np.asarray(inputs, np.float32)
    if source.shape != (8, 256, 512) or not np.isfinite(source).all():
        raise ValueError('Expected finite 8x256x512 input')
    changed = source.copy()
    erased = np.zeros(512, bool)
    guard = np.zeros(512, bool)
    blocks = []
    if kind == 'original':
        pass
    elif kind == 'zero_query':
        changed[4] = 0
    elif kind == 'synthetic_occlusion':
        rng = np.random.default_rng(seed)
        for slot in range(3):
            left = int(rng.integers(48 + slot * 145, 90 + slot * 145))
            width = int(rng.integers(28, 59))
            right = min(511, left + width)
            base = rng.uniform(.1, .9, (3, 1, 1))
            if heldout:
                # Different texture from training, still artificial full-strip occlusion.
                yy, xx = np.indices((256, right-left))
                texture = .04 * np.sin(.29 * xx + .17 * yy)
                patch = np.clip(base + texture[None], 0, 1)
            else:
                patch = np.broadcast_to(base, (3, 256, right-left))
            changed[:3, :, left:right] = patch
            erased[left:right] = True
            guard[max(0,left-4):left+4] = True
            guard[right-4:min(512,right+4)] = True
            blocks.append([left, right])
        # Recompute the same local-normalization channel from modified RGB.
        from .v11_path_contract import _box_sum, NORMALIZATION_RADIUS
        valid = changed[7] > .5
        changed[:3, ~valid] = 0
        gray = changed[:3].transpose(1,2,0) @ np.array([.299,.587,.114])
        count = _box_sum(valid, NORMALIZATION_RADIUS)
        mean = _box_sum(gray, NORMALIZATION_RADIUS) / np.maximum(count, 1)
        second = _box_sum(gray*gray, NORMALIZATION_RADIUS) / np.maximum(count, 1)
        std = np.sqrt(np.maximum(second-mean*mean, 0))
        changed[3] = np.clip((gray-mean)/np.maximum(std, 3/255.), -3, 3)/3
        changed[3, ~valid] = 0
    else:
        raise ValueError('Unknown intervention')
    return dict(input=changed, erased=erased, guard=guard, blocks=blocks,
                synthetic=kind=='synthetic_occlusion', kind=kind)


def supervision(trusted_columns, variant):
    trusted = np.asarray(trusted_columns, bool).copy()
    mask = trusted & ~variant['guard']
    target = np.ones(512, np.float32)
    if variant['kind'] == 'zero_query':
        target[:] = 0
    elif variant['kind'] == 'synthetic_occlusion':
        target[variant['erased']] = 0
    # Unannotated/uncertain tails remain unsupervised in every variant.
    return target, mask
