"""V18b's single derivative-unit correction, separate from frozen V18 code.

For a vertically constant image, OpenCV's unnormalised 3x3 x-Sobel equals
8 * numpy.gradient(profile) at interior pixels. This fixes the numerical scale
of the threshold, not the distinction between mean(abs(gradient)) and
abs(gradient(mean intensity)) in textured images. No accuracy/noise improvement
or material/new-layer identity is asserted. No threshold sweep is provided.
"""
from __future__ import annotations

import numpy as np

from .v18_front_observability import sample_normal_edges


GRADIENT_UNITS = 'gray_levels_per_reference_pixel'
SEMANTIC_GRADIENT_UNITS = 'gray_levels_per_current_pixel'
OLD_UNNORMALIZED_SOBEL_THRESHOLD = 8.
SOBEL_3X3_TO_CENTRAL_DIFFERENCE_SCALE = 8.
THRESHOLD = OLD_UNNORMALIZED_SOBEL_THRESHOLD / SOBEL_3X3_TO_CENTRAL_DIFFERENCE_SCALE
THRESHOLD_SOURCE = 'unnormalized_3x3_Sobel_threshold_8_divided_by_1D_kernel_gain_8'


def semantic_from_seeds(gray, seed_x, rows):
    """Repeat V18's current-image semantic-seed profile search at threshold 1.

    The archived seeds are inputs; no semantic network is rerun. Output fields
    exactly match the frozen driver's x/seed/gradient/contrast arrays. Search
    limits, seven-row mean, 28px Gaussian weighting and signed endpoint contrast
    are retained. A strong distant gradient cannot validate a weak winning peak.
    The seed is diagnostic input, not confirmation of a physical boundary.
    """
    gray = np.asarray(gray)
    seeds = np.asarray(seed_x, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.float64)
    if (gray.ndim != 2 or min(gray.shape) < 5
            or not np.issubdtype(gray.dtype, np.number)):
        raise ValueError('gray must be a two-dimensional numeric image')
    if rows.ndim != 1 or seeds.shape != rows.shape or not np.isfinite(rows).all():
        raise ValueError('seed_x and finite rows must be matching one-dimensional arrays')
    fields = {key: np.full(len(rows), np.nan)
              for key in ('x', 'seed', 'gradient', 'contrast')}
    fields['seed'][:] = seeds
    for i, (seed, y) in enumerate(zip(seeds, rows)):
        if not np.isfinite(seed) or not 0 <= y < gray.shape[0]:
            continue
        xa = max(2, int(round(seed-65)))
        xb = min(gray.shape[1]-2, int(round(seed+66)))
        if xb-xa < 8:
            continue
        ya, yb = max(0, int(y)-3), min(gray.shape[0], int(y)+4)
        strip = gray[ya:yb, xa-1:xb+1]
        if not strip.size or not np.isfinite(strip).all():
            continue
        profile = strip.mean(axis=0, dtype=np.float64)
        gradient = np.gradient(profile)[1:-1]
        if len(gradient) < 3 or np.max(np.abs(gradient)) < THRESHOLD:
            continue
        xs = np.arange(xa, xb)
        score = np.abs(gradient)*np.exp(-.5*((xs-seed)/28.)**2)
        k = int(np.argmax(score))
        if abs(gradient[k]) < THRESHOLD:
            continue
        fields['x'][i] = xa+k
        fields['gradient'][i] = gradient[k]
        # Preserve the frozen driver's signed endpoint diagnostic. This is not
        # the two five-sample means used by the normal-profile contrast below.
        fields['contrast'][i] = profile[min(len(profile)-1, k+6)]-profile[max(0, k-4)]
    return fields


def sample_unit_normal(gray, A, rows, baseline):
    """Use frozen direct profiles, discard old decisions, apply one unit scale.

    Output keys/coordinate conventions match sample_normal_edges. The old
    threshold-8 offsets are never fallback observations: all candidate fields
    are cleared before independent recomputation at 1 gray level/reference px.
    Bounds, affine sampling, finite support and saturation diagnostics remain
    exactly those of the frozen sampler. No extra profiles are interpolated.
    """
    result = sample_normal_edges(gray, A, rows, baseline)
    for key in ('offset', 'gradient_signed', 'peak_ratio', 'contrast'):
        result[key] = np.full_like(result[key], np.nan)
    result['supported'] = np.zeros_like(result['supported'], dtype=bool)
    offsets = result['offsets_px']
    weights = np.exp(-.5*(offsets/28.)**2)
    for j, profile in enumerate(result['profile']):
        if not np.isfinite(profile).all():
            continue
        gradient = np.gradient(profile)
        magnitude = np.abs(gradient)
        if np.max(magnitude) < THRESHOLD:
            result['reason'][j] = 'gradient_below_unit_threshold'
            continue
        score = magnitude*weights
        k = int(np.argmax(score))
        if magnitude[k] < THRESHOLD:
            result['reason'][j] = 'selected_gradient_below_unit_threshold'
            continue
        if k < 5 or k+5 >= len(offsets):
            result['reason'][j] = 'peak_contrast_support_incomplete'
            continue
        outside = np.abs(offsets-offsets[k]) > 8
        secondary = float(np.max(score[outside]))
        result['offset'][j] = offsets[k]
        result['gradient_signed'][j] = gradient[k]
        result['peak_ratio'][j] = float(score[k])/max(secondary, 1e-6)
        result['contrast'][j] = abs(float(profile[k+1:k+6].mean()-profile[k-5:k].mean()))
        result['supported'][j] = True
        result['reason'][j] = 'direct_unit_gradient_candidate_not_material_identity'
    return result
