"""Fixed-reference, direct-image contour observations for V18 research only.

No truth labels, temporal event decisions, device drivers or learned path inputs
are used here. A returned gradient peak is an image candidate, not a verified
material boundary, new layer, or mechanical revolution.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


MINIMUM_BASELINE_FRAMES = 20
NORMAL_WINDOW_ROWS = 5
SEARCH_RADIUS_PX = 65
GRADIENT_THRESHOLD = 8.0
WEIGHT_SIGMA_PX = 28.0
SECOND_PEAK_EXCLUSION_PX = 8
TANGENT_RADIUS_PX = 3


def _rows_array(rows):
    result = np.asarray(rows, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError('rows must be a finite one-dimensional array')
    if len(result) > 1 and not np.all(np.diff(result) > 0):
        raise ValueError('rows must be strictly increasing')
    return result


def _runs(indices):
    return [run for run in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
            if len(run)]


def build_fixed_baseline(times, rows, aligned_x, ready_s=5.):
    """Build once from finite timestamps <= ready_s; never update from later data.

    ``aligned_x[i, j]`` is the directly observed *left* silhouette x at reference
    image row ``rows[j]`` and time ``times[i]``. Coordinates must already share
    a reference image frame. ``baseline_mad_px`` is the unscaled median absolute
    deviation, not a calibrated uncertainty. Normals point outward to image left:
    (-1, dx/dy) / sqrt(1 + (dx/dy)**2).

    At least 20 eligible frames and 20 finite observations per row are required.
    Normals use at most five neighbouring rows within the same valid run; fewer
    than three local rows or numerically degenerate fitting leaves them unknown.
    A finite x_ref without a normal is retained for diagnostics only.
    """
    rows = _rows_array(rows)
    times = np.asarray(times, dtype=np.float64)
    aligned_x = np.asarray(aligned_x, dtype=np.float64)
    if times.ndim != 1 or aligned_x.shape != (len(times), len(rows)):
        raise ValueError('aligned_x must have shape (len(times), len(rows))')
    if not math.isfinite(float(ready_s)):
        raise ValueError('ready_s must be finite')
    eligible = np.isfinite(times) & (times <= float(ready_s))
    selected = aligned_x[eligible]
    count = np.isfinite(selected).sum(axis=0)
    x_ref = np.full(len(rows), np.nan)
    mad = np.full(len(rows), np.nan)
    nx = np.full(len(rows), np.nan)
    ny = np.full(len(rows), np.nan)
    location_supported = ((count >= MINIMUM_BASELINE_FRAMES)
                          & (int(eligible.sum()) >= MINIMUM_BASELINE_FRAMES))
    for j in np.flatnonzero(location_supported):
        values = selected[:, j]
        values = values[np.isfinite(values)]
        x_ref[j] = np.median(values)
        mad[j] = np.median(np.abs(values - x_ref[j]))
    radius = NORMAL_WINDOW_ROWS // 2
    for run in _runs(np.flatnonzero(location_supported)):
        for k, j in enumerate(run):
            neighbours = run[max(0, k-radius):min(len(run), k+radius+1)]
            if len(neighbours) < 3:
                continue
            yy = rows[neighbours] - np.mean(rows[neighbours])
            xx = x_ref[neighbours] - np.mean(x_ref[neighbours])
            denominator = float(yy @ yy)
            if not math.isfinite(denominator) or denominator <= 1e-12:
                continue
            slope = float(yy @ xx) / denominator
            length = math.hypot(1., slope)
            if not math.isfinite(slope) or not math.isfinite(length):
                continue
            nx[j], ny[j] = -1. / length, slope / length
    supported = location_supported & np.isfinite(nx) & np.isfinite(ny)
    return dict(x_ref=x_ref, nx=nx, ny=ny, baseline_mad_px=mad,
                supported=supported, location_supported=location_supported,
                support_count=count, rows=rows.copy(), ready_s=float(ready_s),
                baseline_frame_count=int(eligible.sum()),
                baseline_times_s=times[eligible].copy(),
                baseline_ready=bool(int(eligible.sum()) >= MINIMUM_BASELINE_FRAMES),
                boundary_identity_confirmed=False, new_layer_confirmed=False)


def sample_normal_edges(gray_current, ref_to_current, rows, baseline):
    """Sample direct normal profiles in the current grayscale (0..255) image.

    A finite, nonsingular 2x3 affine maps reference points into current pixels.
    The 131 offsets -65..65 and seven tangent offsets -3..3 are in *reference*
    pixels, transformed together by that affine. Positive offset is outward.
    Every bilinear footprint must be in bounds: a clipped search strip becomes
    unknown, never a replicated border edge. Unknown profiles/offsets remain NaN.

    ``gradient_signed`` differentiates intensity toward the outward normal;
    ``contrast`` is the absolute mean difference of the five samples on each
    side of the selected peak (excluding that peak). ``peak_ratio`` compares
    Gaussian-weighted primary and strongest secondary gradients outside +/-8px;
    its denominator has a 1e-6 numerical floor. Neither is an identity score.
    ``clipped_fraction`` counts sampled intensities <=0 or >=255, not out-of-bounds
    samples and not a test for physical occlusion. Saturation is diagnostic only.
    """
    rows = _rows_array(rows)
    gray = np.asarray(gray_current)
    if gray.ndim != 2 or min(gray.shape) < 2 or not np.issubdtype(gray.dtype, np.number):
        raise ValueError('gray_current must be a two-dimensional numeric image')
    affine = np.asarray(ref_to_current, dtype=np.float64)
    if (affine.shape != (2, 3) or not np.isfinite(affine).all()
            or abs(float(np.linalg.det(affine[:, :2]))) <= 1e-12):
        raise ValueError('ref_to_current must be a finite nonsingular 2x3 affine')
    values = {key: np.asarray(baseline[key])
              for key in ('x_ref', 'nx', 'ny', 'supported')}
    if any(value.shape != rows.shape for value in values.values()):
        raise ValueError('baseline arrays must match rows')
    if 'rows' in baseline and not np.array_equal(np.asarray(baseline['rows']), rows):
        raise ValueError('baseline row coordinates changed')
    nrows = len(rows)
    offsets = np.arange(-SEARCH_RADIUS_PX, SEARCH_RADIUS_PX + 1, dtype=np.float64)
    profile = np.full((nrows, len(offsets)), np.nan)
    result = dict(offset=np.full(nrows, np.nan), gradient_signed=np.full(nrows, np.nan),
                  peak_ratio=np.full(nrows, np.nan), contrast=np.full(nrows, np.nan),
                  clipped_fraction=np.full(nrows, np.nan), profile=profile,
                  offsets_px=offsets, supported=np.zeros(nrows, dtype=bool),
                  reason=np.full(nrows, 'baseline_unsupported', dtype=object),
                  boundary_identity_confirmed=False, new_layer_confirmed=False)
    finite = (values['supported'].astype(bool) & np.isfinite(values['x_ref'])
              & np.isfinite(values['nx']) & np.isfinite(values['ny']))
    norms = np.hypot(values['nx'], values['ny'])
    finite &= np.isfinite(norms) & (np.abs(norms-1.) < 1e-6)
    valid_indices = np.flatnonzero(finite)
    if not len(valid_indices):
        return result
    normal = np.column_stack([values['nx'][valid_indices], values['ny'][valid_indices]])
    tangent = np.column_stack([normal[:, 1], -normal[:, 0]])
    centres = np.column_stack([values['x_ref'][valid_indices], rows[valid_indices]])
    tangent_offsets = np.arange(-TANGENT_RADIUS_PX, TANGENT_RADIUS_PX + 1, dtype=np.float64)
    points = (centres[:, None, None, :]
              + offsets[None, None, :, None]*normal[:, None, None, :]
              + tangent_offsets[None, :, None, None]*tangent[:, None, None, :])
    current = points @ affine[:, :2].T + affine[:, 2]
    h, w = gray.shape
    # Strictly retain the complete bilinear footprint, including the extra pixel
    # required by interpolation; no border fill participates in a valid profile.
    in_bounds = (np.isfinite(current).all(axis=-1)
                 & (current[..., 0] >= 0.) & (current[..., 0] < w-1.)
                 & (current[..., 1] >= 0.) & (current[..., 1] < h-1.))
    whole_strip = in_bounds.all(axis=(1, 2))
    result['reason'][valid_indices[~whole_strip]] = 'sampling_out_of_bounds'
    chosen = np.flatnonzero(whole_strip)
    if not len(chosen):
        return result
    # Chunking prevents cv2.remap's signed-short image-height limit from being
    # reached if a caller requests an unusually dense source-row grid.
    weights = np.exp(-.5 * (offsets / WEIGHT_SIGMA_PX)**2)
    gray_float = gray.astype(np.float32, copy=False)
    for chunk_start in range(0, len(chosen), 1024):
        chunk = chosen[chunk_start:chunk_start+1024]
        points_chunk = current[chunk]
        shape = (-1, len(offsets))
        sampled = cv2.remap(gray_float,
                            points_chunk[..., 0].reshape(shape).astype(np.float32),
                            points_chunk[..., 1].reshape(shape).astype(np.float32),
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0).reshape(len(chunk), len(tangent_offsets), len(offsets))
        for local, cube in zip(chunk, sampled):
            j = valid_indices[local]
            if not np.isfinite(cube).all():
                result['reason'][j] = 'nonfinite_image_samples'
                continue
            p = cube.astype(np.float64).mean(axis=0)
            profile[j] = p
            result['clipped_fraction'][j] = float(np.mean((cube <= 0.) | (cube >= 255.)))
            gradient = np.gradient(p)
            magnitude = np.abs(gradient)
            if float(np.max(magnitude)) < GRADIENT_THRESHOLD:
                result['reason'][j] = 'gradient_below_threshold'
                continue
            weighted = magnitude * weights
            peak = int(np.argmax(weighted))
            # The winning candidate itself must carry evidence; a strong distant
            # edge must not validate an unrelated weak peak near the reference.
            if magnitude[peak] < GRADIENT_THRESHOLD:
                result['reason'][j] = 'selected_gradient_below_threshold'
                continue
            if peak < 5 or peak + 5 >= len(offsets):
                result['reason'][j] = 'peak_contrast_support_incomplete'
                continue
            competitor = np.abs(offsets-offsets[peak]) > SECOND_PEAK_EXCLUSION_PX
            secondary = float(np.max(weighted[competitor]))
            result['offset'][j] = offsets[peak]
            result['gradient_signed'][j] = gradient[peak]
            result['peak_ratio'][j] = float(weighted[peak]) / max(secondary, 1e-6)
            result['contrast'][j] = abs(float(p[peak+1:peak+6].mean()-p[peak-5:peak].mean()))
            result['supported'][j] = True
            result['reason'][j] = 'direct_gradient_candidate_not_material_identity'
    return result
