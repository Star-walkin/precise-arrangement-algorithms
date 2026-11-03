"""A3: remeasure a saved automatic free-rope seed, then test straight continuation.

All geometry is in the original, unmirrored image. A measured optical stripe is
not a physical diameter, contact point, or confirmed material identity. This
module never reads an old selected connection and never initializes equipment.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import cv2
import numpy as np


PARAMETERS = {
    'outside_minimum_profiles': 14,
    'outside_minimum_span_px': 100.,
    'outside_sample_step_px': 3.,
    'outside_maximum_samples': 280,
    'maximum_seed_angle_change_deg': 2.5,
    'maximum_seed_normal_shift_width': .38,
    'extension_step_width': .22,
    'extension_maximum_px': 1800.,
    'extension_maximum_image_fraction': .45,
    'maximum_unsupported_run_width': .85,
    'extension_center_shift_width': .25,
    'extension_width_minimum_ratio': .65,
    'extension_width_maximum_ratio': 1.45,
    'extension_minimum_contrast': 5.,
    'extension_noise_contrast_factor': 1.7,
    'extension_minimum_gradient': .7,
    'texture_minimum_coherence': .2,
    'texture_mismatch_deg': 34.,
    'texture_background_advantage_deg': 12.,
    'maximum_profile_gap_width_for_mask': .6,
    'wide_profile_radius_pitch': 1.8,
    'wide_profile_tangent_span_pitch': 2.2,
    'wide_profile_quantile': .70,
    'wide_minimum_consistent_profiles': 14,
    'narrow_seed_pitch_ratio': .45,
    'yellow_background_fraction': .55,
    'maximum_inferred_run_width': 3.,
    'single_edge_window_width': .16,
}

_A2_PATH = Path(__file__).resolve().parent.parent / 'camera_condition_incoming_rope_20261026_v2/free_refinement.py'
_SPEC = importlib.util.spec_from_file_location('_a3_original_free_refinement', _A2_PATH)
_A2 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_A2)


def dependencies():
    return [str(_A2_PATH)]


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-9)


def _angle_difference(a, b):
    return abs((float(a) - float(b) + 90.) % 180. - 90.)


def _texture(gray, center, direction, width, length=None):
    """Describe interior strand texture, deliberately excluding outer rope edges.

    The structure tensor measures image texture orientation. Weak/isotropic
    texture stays unknown; it must not be interpreted as a physical rope axis.
    """
    d = _unit(direction); n = np.array([-d[1], d[0]])
    length = float(length or np.clip(width * 5., 40., 180.))
    ts = np.arange(-length / 2, length / 2 + .5, 1., dtype=np.float32)
    ns = np.arange(-max(2., width * .28), max(2., width * .28) + .5, 1., dtype=np.float32)
    xy = np.asarray(center)[None, None, :] + ts[:, None, None] * d + ns[None, :, None] * n
    h, w = gray.shape
    valid = (xy[..., 0] >= 1) & (xy[..., 0] < w - 2) & (xy[..., 1] >= 1) & (xy[..., 1] < h - 2)
    if valid.mean() < .95:
        return {'status': 'unknown_border', 'angle_deg': None, 'coherence': 0., 'energy': 0.}
    patch = cv2.remap(gray, xy[..., 0].astype(np.float32), xy[..., 1].astype(np.float32), cv2.INTER_LINEAR)
    high = patch - cv2.GaussianBlur(patch, (0, 0), max(.8, width * .15))
    # Patch rows run along d and columns along n.
    gt = cv2.Sobel(high, cv2.CV_32F, 0, 1, ksize=3) / 8.
    gn = cv2.Sobel(high, cv2.CV_32F, 1, 0, ksize=3) / 8.
    gt = gt[2:-2, 1:-1]; gn = gn[2:-2, 1:-1]
    if not gt.size:
        return {'status': 'unknown_small', 'angle_deg': None, 'coherence': 0., 'energy': 0.}
    jtt, jnn, jtn = float(np.mean(gt * gt)), float(np.mean(gn * gn)), float(np.mean(gt * gn))
    energy = jtt + jnn
    coherence = float(math.hypot(jtt - jnn, 2 * jtn) / max(energy, 1e-6))
    axis = .5 * math.atan2(2 * jtn, jtt - jnn) + math.pi / 2
    image_vector = math.cos(axis) * d + math.sin(axis) * n
    angle = float(math.degrees(math.atan2(image_vector[1], image_vector[0])) % 180.)
    local_angles = (np.arctan2(gn, gt) + math.pi / 2) % math.pi
    hist, _ = np.histogram(local_angles, bins=12, range=(0, math.pi), weights=gt * gt + gn * gn)
    hist = hist / max(float(hist.sum()), 1e-6)
    return {'status': 'supported' if energy >= 1. and coherence >= .15 else 'weak_texture',
            'angle_deg': angle, 'coherence': coherence, 'energy': energy,
            'tensor_tn': [[jtt, jtn], [jtn, jnn]], 'orientation_histogram_tangent_frame': hist.tolist(),
            'patch_mean': float(np.mean(patch)), 'patch_std': float(np.std(patch)),
            'coordinate_meaning': 'image texture line orientation modulo 180 degrees'}


def _aggregate_texture(samples):
    useful = [s for s in samples if s['status'] == 'supported']
    if not useful:
        return {'status': 'unknown', 'angle_deg': None, 'coherence': 0., 'samples': samples}
    weights = np.asarray([s['coherence'] * min(s['energy'], 100.) for s in useful])
    angles = np.radians([s['angle_deg'] * 2 for s in useful])
    vector = np.array([np.sum(weights * np.cos(angles)), np.sum(weights * np.sin(angles))])
    return {'status': 'supported', 'angle_deg': float(math.degrees(math.atan2(vector[1], vector[0])) / 2 % 180),
            'coherence': float(np.linalg.norm(vector) / max(float(weights.sum()), 1e-6)),
            'median_local_coherence': float(np.median([s['coherence'] for s in useful])),
            'samples': samples, 'meaning': 'raw interior strand texture only; not material identity'}


def _paint_pairs(shape, left, right, maximum_gap, allowed=None):
    mask = np.zeros(shape, np.uint8)
    left = np.asarray(left, float).reshape(-1, 2); right = np.asarray(right, float).reshape(-1, 2)
    for i in range(len(left) - 1):
        if allowed is not None and not (allowed[i] and allowed[i + 1]):
            continue
        if max(np.linalg.norm(left[i + 1] - left[i]), np.linalg.norm(right[i + 1] - right[i])) > maximum_gap:
            continue
        polygon = np.rint([left[i], right[i], right[i + 1], left[i + 1]]).astype(np.int32)
        cv2.fillConvexPoly(mask, polygon, 255)
    return mask


def _seed(data):
    for key in ('a2_record', 'a1_record'):
        record = data.get(key) or {}
        free = record.get('free_segment', record)
        if free.get('status') == 'observed' and free.get('coordinate_system') == 'original_unmirrored_image_xy_pixels':
            try:
                if float(free.get('width_px') or 0.) > 1. and np.asarray(free['observed_segment_xy']).shape == (2, 2):
                    return free, key
            except (ValueError, TypeError, KeyError):
                continue
    return None, None


def _remeasure(image, surface, seed):
    segment = np.asarray(seed['observed_segment_xy'], float)
    if not np.isfinite(segment).all():
        return None, {'reason': 'invalid_seed_coordinates'}
    center = segment.mean(axis=0); d = _unit(segment[1] - segment[0])
    if d[0] < 0: d = -d
    n = np.array([-d[1], d[0]]); width = float(seed['width_px'])
    span = float(np.linalg.norm(segment[1] - segment[0]))
    count = min(PARAMETERS['outside_maximum_samples'], max(2, int(span / PARAMETERS['outside_sample_step_px']) + 1))
    t = np.linspace(-span / 2, span / 2, count)
    anchors = center + t[:, None] * d
    pairs, diag = _A2._measure_pairs(image, surface, seed, anchors, center, d, n, width)
    if len(pairs) < PARAMETERS['outside_minimum_profiles']:
        return None, {**diag, 'reason': 'outside_paired_profiles_insufficient'}
    coeff, keep, fitdiag = _A2._joint_fit(pairs, width)
    diag.update(fitdiag)
    if coeff is None:
        return None, {**diag, 'reason': 'outside_joint_fit_rejected'}
    pairs = pairs[keep]; t = pairs[:, 0]
    new_d = _unit(d + coeff[1] * n)
    measured_widths = pairs[:, 2] - pairs[:, 1]
    failures = []
    if np.ptp(t) < PARAMETERS['outside_minimum_span_px']: failures.append('outside_span_short')
    if math.degrees(math.acos(float(np.clip(d @ new_d, -1., 1.)))) > PARAMETERS['maximum_seed_angle_change_deg']: failures.append('seed_angle_change')
    if abs(coeff[0]) > max(3., width * PARAMETERS['maximum_seed_normal_shift_width']): failures.append('seed_normal_shift')
    if np.percentile(np.abs(measured_widths - np.median(measured_widths)), 90) > max(3., width * .30): failures.append('width_unstable')
    if fitdiag['fit_edge_residual_p90_abs_px'] > max(2.5, width * .22): failures.append('edge_fit_residual')
    if failures:
        return None, {**diag, 'reason': 'outside_revalidation_failed', 'failures': failures}
    bases = center + t[:, None] * d
    left = bases + pairs[:, 1, None] * n; right = bases + pairs[:, 2, None] * n
    actual = (left + right) / 2
    ends_t = np.array([t.min(), t.max()])
    ends = center + ends_t[:, None] * d + (coeff[0] + coeff[1] * ends_t)[:, None] * n
    toward, toward_status, candidates, hits = _A2._refined_surface_direction(ends, surface, new_d)
    normal = np.array([-new_d[1], new_d[0]])
    result = {'status': 'observed', 'coordinate_system': 'original_unmirrored_image_xy_pixels',
              'observed_segment_xy': ends.tolist(), 'observed_points_xy': actual.tolist(),
              'boundary_left_xy': left.tolist(), 'boundary_right_xy': right.tolist(),
              'center_xy': ends.mean(axis=0).tolist(), 'direction_xy': new_d.tolist(), 'normal_xy': normal.tolist(),
              'width_px': float(coeff[2]), 'width_interval_px': np.percentile(measured_widths, [10, 90]).tolist(),
              'width_kind': 'original_optical_two_edge_spacing_not_physical_diameter',
              'angle_deg': float(math.degrees(math.atan2(new_d[1], new_d[0])) % 180.),
              'toward_surface_direction_xy': toward, 'toward_surface_status': toward_status,
              'toward_surface_candidates_xy': candidates, 'surface_ray_hits': hits,
              'observed_points_kind': 'new_original_outside_surface_paired_edge_midpoints',
              'profile_sample_count': len(pairs), 'support_length_px': float(np.ptp(t)),
              'diagnostics': diag, 'contact_or_tangency_claim': False}
    return result, diag


def _local_channel(image, center, direction, width, brightness, achromatic):
    """Select channel by measured flank color, never by candidate success.

    Yellow background needs color contrast. Neutral steel needs raw brightness;
    using min/max there destroys the very rope/background contrast we need.
    """
    d = _unit(direction); n = np.array([-d[1], d[0]])
    ts = np.linspace(-width, width, 7)
    ns = np.r_[np.linspace(-1.6 * width, -1.1 * width, 5), np.linspace(1.1 * width, 1.6 * width, 5)]
    xy = np.asarray(center) + ts[:, None, None] * d + ns[None, :, None] * n
    colors = cv2.remap(image, xy[..., 0].astype(np.float32), xy[..., 1].astype(np.float32), cv2.INTER_LINEAR).astype(np.float32)
    b, g, r = colors[..., 0], colors[..., 1], colors[..., 2]
    yellow = (r > b * 1.35 + 8) & (g > b * 1.25 + 8) & (np.maximum(r, g) > 40)
    fraction = float(np.mean(yellow))
    if fraction >= PARAMETERS['yellow_background_fraction']:
        return achromatic, 'local_yellow_flank_achromatic', fraction
    return brightness, 'local_neutral_flank_original_min_brightness', fraction


def _wide_remeasure(image, surface, seed, narrow, pitch):
    """Measure the outside rope envelope without fixing width to the old seed.

    A short along-rope upper-quantile profile averages the alternating individual
    strand highlights. We then find its TWO outer transitions, check both are
    outside the supplied surface and require repeated, straight, stable pairs.
    Pitch bounds the search only; it is never substituted as a measured width.
    """
    seg = np.asarray(seed['observed_segment_xy'], float); center = seg.mean(axis=0)
    d = _unit(seg[1] - seg[0]); d = d if d[0] >= 0 else -d
    n = np.array([-d[1], d[0]]); span = float(np.linalg.norm(seg[1] - seg[0]))
    old_width = float(seed['width_px']); scale = max(float(pitch), old_width)
    radius = max(PARAMETERS['wide_profile_radius_pitch'] * float(pitch), 1.8 * old_width, 16.)
    ns = np.arange(-radius, radius + .25, .5, dtype=np.float32)
    half_length = min(80., PARAMETERS['wide_profile_tangent_span_pitch'] * scale / 2)
    ts = np.linspace(-half_length, half_length, 25, dtype=np.float32)
    tvalues = np.linspace(-span / 2, span / 2, min(80, max(16, int(span / 8))))
    bgr = image.astype(np.float32); brightness = bgr.min(axis=2)
    achromatic = brightness * 255. / np.maximum(bgr.max(axis=2), 1.)
    pairs = []; rejected = {}; h, w = surface.shape
    def reject(reason): rejected[reason] = rejected.get(reason, 0) + 1
    for t in tvalues:
        point = center + t * d
        cx, cy = np.rint(point).astype(int)
        if not (0 <= cx < w and 0 <= cy < h) or surface[cy, cx] != 0:
            reject('center_not_outside_surface'); continue
        channel, kind, _ = _local_channel(image, point, d, max(old_width, pitch * .65), brightness, achromatic)
        xy = point + ts[:, None, None] * d + ns[None, :, None] * n
        if np.any(xy[..., 0] < 1) or np.any(xy[..., 0] >= w - 2) or np.any(xy[..., 1] < 1) or np.any(xy[..., 1] >= h - 2):
            reject('wide_profile_at_border'); continue
        patch = cv2.remap(channel, xy[..., 0].astype(np.float32), xy[..., 1].astype(np.float32), cv2.INTER_LINEAR)
        profile = np.quantile(patch, PARAMETERS['wide_profile_quantile'], axis=0).astype(np.float32)
        profile = cv2.GaussianBlur(profile[None, :], (5, 1), .8)[0]
        central = np.flatnonzero(abs(ns) <= min(radius * .45, max(old_width * .5, pitch * .35)))
        peak_id = int(central[np.argmax(profile[central])]); peak = float(profile[peak_id])
        left_base = float(np.percentile(profile[ns < -radius * .72], 30))
        right_base = float(np.percentile(profile[ns > radius * .72], 30))
        if min(peak - left_base, peak - right_base) < 17.:
            reject('wide_outer_background_contrast_missing'); continue
        threshold = np.where(ns <= ns[peak_id], left_base + .35 * (peak - left_base), right_base + .35 * (peak - right_base))
        foreground = (profile >= threshold).astype(np.uint8)[None, :]
        kernel = max(3, int(round(.14 * scale / .5)) | 1)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((1, kernel), np.uint8))[0]
        lo = peak_id; hi = peak_id
        while lo > 0 and foreground[lo - 1]: lo -= 1
        while hi + 1 < len(ns) and foreground[hi + 1]: hi += 1
        if lo <= 1 or hi >= len(ns) - 2:
            reject('wide_component_touches_search_boundary'); continue
        gradient = np.gradient(profile, .5)
        allowance = max(1.5, .1 * scale)
        lw = np.flatnonzero(abs(ns - ns[lo]) <= allowance)
        rw = np.flatnonzero(abs(ns - ns[hi]) <= allowance)
        li = int(lw[np.argmax(gradient[lw])]); ri = int(rw[np.argmax(-gradient[rw])])
        if min(float(gradient[li]), float(-gradient[ri])) < 1.2:
            reject('wide_opposite_gradients_missing'); continue
        left = float(ns[0] + _A2._quadratic_extremum(gradient, li, .5))
        right = float(ns[0] + _A2._quadratic_extremum(-gradient, ri, .5))
        width = right - left
        if not max(3., .2 * pitch) <= width <= min(1.8 * scale, 1.7 * radius):
            reject('wide_component_width_out_of_bounds'); continue
        if abs((left + right) / 2) > max(old_width * .55, pitch * .6):
            reject('wide_component_off_seed_axis'); continue
        edges = point + np.asarray([left, right])[:, None] * n
        ex, ey = np.rint(edges[:, 0]).astype(int), np.rint(edges[:, 1]).astype(int)
        if np.any(surface[ey, ex] != 0):
            reject('wide_edges_inside_supplied_surface'); continue
        pairs.append([t, left, right, min(float(gradient[li]), float(-gradient[ri])), 0.])
    diag = {'tested_profiles': len(tvalues), 'paired_profiles': len(pairs), 'reject_counts': rejected,
            'search_radius_px': radius, 'seed_width_px': old_width, 'pitch_used_only_to_bound_search': True,
            'method': 'raw_local_channel_tangent_quantile_envelope_outer_gradients'}
    if len(pairs) < PARAMETERS['wide_minimum_consistent_profiles']:
        return narrow, {**diag, 'status': 'insufficient_whole_width_evidence'}
    pair = np.asarray(pairs); coeff, keep, fitdiag = _A2._joint_fit(pair, float(np.median(pair[:, 2] - pair[:, 1])))
    diag['fit'] = fitdiag
    if coeff is None: return narrow, {**diag, 'status': 'wide_joint_fit_rejected'}
    pair = pair[keep]; widths = pair[:, 2] - pair[:, 1]
    median = float(np.median(widths)); mad = float(1.4826 * np.median(abs(widths - median)))
    if np.ptp(pair[:, 0]) < 100 or mad > max(2.5, median * .2) or fitdiag['fit_edge_residual_p90_abs_px'] > max(2.5, median * .2):
        return narrow, {**diag, 'status': 'wide_edges_not_stable', 'median_width_px': median, 'width_mad_px': mad}
    diag.update({'status': 'whole_envelope_supported', 'median_width_px': median, 'width_mad_px': mad,
                 'fit_width_px': float(coeff[2]), 'width_ratio_vs_seed': float(coeff[2] / old_width)})
    if narrow is not None and coeff[2] < 1.30 * narrow['width_px']:
        return narrow, {**diag, 'geometry_action': 'original_two_edge_measurement_preserved'}
    new_d = _unit(d + coeff[1] * n)
    if math.degrees(math.acos(float(np.clip(d @ new_d, -1., 1.)))) > 2.5:
        return narrow, {**diag, 'status': 'wide_angle_changed', 'geometry_action': 'not_used'}
    t = pair[:, 0]; bases = center + t[:, None] * d
    left = bases + pair[:, 1, None] * n; right = bases + pair[:, 2, None] * n
    end_t = np.asarray([t.min(), t.max()]);ends = center + end_t[:, None] * d + (coeff[0] + coeff[1] * end_t)[:, None] * n
    toward, toward_status, directions, hits = _A2._refined_surface_direction(ends, surface, new_d)
    out = dict(narrow or {})
    out.update({'status': 'observed', 'coordinate_system': 'original_unmirrored_image_xy_pixels',
        'observed_segment_xy': ends.tolist(), 'observed_points_xy': ((left+right)/2).tolist(),
        'boundary_left_xy': left.tolist(), 'boundary_right_xy': right.tolist(),
        'center_xy': ends.mean(axis=0).tolist(), 'direction_xy': new_d.tolist(), 'normal_xy': [-float(new_d[1]), float(new_d[0])],
        'width_px': float(coeff[2]), 'width_interval_px': np.percentile(widths,[10,90]).tolist(),
        'width_kind': 'measured_tangent_aggregated_optical_outer_envelope_not_physical_diameter',
        'angle_deg': float(math.degrees(math.atan2(new_d[1],new_d[0]))%180.),
        'toward_surface_direction_xy': toward,'toward_surface_status': toward_status,'toward_surface_candidates_xy': directions,
        'surface_ray_hits': hits,'observed_points_kind':'new_outside_tangent_profile_outer_edge_midpoints',
        'profile_sample_count':len(pair),'support_length_px':float(np.ptp(t)), 'contact_or_tangency_claim':False})
    return out, {**diag, 'geometry_action': 'expanded_to_independently_supported_outer_edges'}


def _extension_profile(channel, center, direction, width):
    """Measure two local edges around the fixed free-line prediction.

    Bright and dark stripe polarities are both tested on the raw channel. This
    accommodates a rope crossing a brighter surface, but requires both flanks
    and correct signed gradients; one dark groove alone is insufficient.
    """
    d = _unit(direction); n = np.array([-d[1], d[0]])
    offsets = np.arange(-1.65 * width, 1.65 * width + .25, .5, dtype=np.float32)
    tangent_offsets = np.asarray([-width * .12, 0., width * .12], np.float32)
    xy = np.asarray(center) + offsets[None, :, None] * n + tangent_offsets[:, None, None] * d
    h, w = channel.shape
    if np.any(xy[..., 0] < 1) or np.any(xy[..., 0] >= w - 2) or np.any(xy[..., 1] < 1) or np.any(xy[..., 1] >= h - 2):
        return None, 'image_border'
    patch = cv2.remap(channel, xy[..., 0].astype(np.float32), xy[..., 1].astype(np.float32), cv2.INTER_LINEAR)
    profile = cv2.GaussianBlur(np.mean(patch, axis=0)[None, :], (5, 1), .7)[0]
    noise = max(1., 1.4826 * float(np.median(np.abs(np.diff(profile, n=2)))))
    gradient = np.gradient(profile, .5)
    options = []
    for polarity in (1., -1.):
        signed = polarity * profile; grad = polarity * gradient
        lwin = np.flatnonzero((offsets >= -.8 * width) & (offsets <= -.25 * width))
        rwin = np.flatnonzero((offsets >= .25 * width) & (offsets <= .8 * width))
        li = int(lwin[np.argmax(grad[lwin])]); ri = int(rwin[np.argmax(-grad[rwin])])
        left, right = float(offsets[li]), float(offsets[ri]); measured_width = right - left
        if not PARAMETERS['extension_width_minimum_ratio'] * width <= measured_width <= PARAMETERS['extension_width_maximum_ratio'] * width:
            continue
        if abs((left + right) / 2) > PARAMETERS['extension_center_shift_width'] * width:
            continue
        inner = signed[(offsets >= left + .22 * measured_width) & (offsets <= right - .22 * measured_width)]
        lbg = signed[(offsets >= left - .5 * width) & (offsets <= left - .15 * width)]
        rbg = signed[(offsets >= right + .15 * width) & (offsets <= right + .5 * width)]
        if min(len(inner), len(lbg), len(rbg)) < 2: continue
        contrast = min(float(np.median(inner) - np.median(lbg)), float(np.median(inner) - np.median(rbg)))
        gradient_strength = min(float(grad[li]), float(-grad[ri]))
        if contrast < max(PARAMETERS['extension_minimum_contrast'], noise * PARAMETERS['extension_noise_contrast_factor']): continue
        if gradient_strength < max(PARAMETERS['extension_minimum_gradient'], .35 * noise): continue
        # Subpixel peak interpolation remains local to the measured derivative.
        left = float(offsets[0] + _A2._quadratic_extremum(grad, li, .5))
        right = float(offsets[0] + _A2._quadratic_extremum(-grad, ri, .5))
        options.append({'left_xy': (center + left * n).tolist(), 'right_xy': (center + right * n).tolist(),
                        'center_xy': (center + .5 * (left + right) * n).tolist(), 'width_px': right - left,
                        'center_shift_px': .5 * (left + right), 'polarity': 'bright' if polarity > 0 else 'dark',
                        'contrast': contrast, 'gradient_strength': gradient_strength, 'noise': noise,
                        'score': float(min(1., contrast / max(15., 4. * noise)) * min(1., gradient_strength / 3.))})
    if not options: return None, 'bilateral_raw_profile_unsupported'
    options.sort(key=lambda a: a['score'], reverse=True)
    return options[0], 'paired_edges_measured'


def _single_edge_support(channel, center, direction, width):
    """One strong local edge supports a hypothesis, not a new width/center."""
    d = _unit(direction); n = np.array([-d[1], d[0]])
    ns = np.arange(-1.1 * width, 1.1 * width + .25, .5, dtype=np.float32)
    ts = np.asarray([-width*.18, 0., width*.18], np.float32)
    xy = center + ts[:,None,None]*d + ns[None,:,None]*n
    h,w=channel.shape
    if np.any(xy[...,0]<1) or np.any(xy[...,0]>=w-2) or np.any(xy[...,1]<1) or np.any(xy[...,1]>=h-2):return None
    patch=cv2.remap(channel,xy[...,0].astype(np.float32),xy[...,1].astype(np.float32),cv2.INTER_LINEAR)
    patch=cv2.GaussianBlur(patch,(5,1),.8);grad=np.gradient(patch,.5,axis=1)
    options=[]
    for side in (-1,1):
        expected=side*.5*width
        ids=np.flatnonzero(abs(ns-expected)<=max(1.,PARAMETERS['single_edge_window_width']*width))
        if len(ids)<2:continue
        strength=np.mean(abs(grad[:,ids]),axis=0);k=int(ids[np.argmax(strength)])
        noise=max(1.,float(np.median(abs(np.diff(patch,n=2,axis=1)))))
        signs=np.sign(grad[:,k]);coherent=max(np.mean(signs>0),np.mean(signs<0))
        value=float(np.mean(abs(grad[:,k])))
        if coherent<.99 or value<max(2.,2.*noise):continue
        options.append({'measured_side':int(side),'edge_xy':(center+float(ns[k])*n).tolist(),
                        'edge_strength':value,'noise':noise,'three_tangent_profile_sign_agreement':float(coherent)})
    if not options:return None
    return max(options,key=lambda a:a['edge_strength'])


def _extend(image, gray, channel, surface, free, direction, texture, outward=False):
    d = _unit(direction); n = np.array([-d[1], d[0]]); width = float(free['width_px'])
    ends = np.asarray(free['observed_segment_xy'], float)
    start = ends[int(np.argmax(ends @ d))]
    step = float(np.clip(width * PARAMETERS['extension_step_width'], 2., 5.))
    maximum = min(PARAMETERS['extension_maximum_px'], PARAMETERS['extension_maximum_image_fraction'] * max(gray.shape))
    brightness=np.min(image,axis=2).astype(np.float32)
    achromatic=brightness*255./np.maximum(np.max(image,axis=2).astype(np.float32),1.)
    samples = []; left = []; right = []; centers = []; states = []; unsupported = 0.; first_bad = None; inferred_run=0.
    last_good = start.copy(); stop_reason = 'bounded_search_limit'
    for distance in np.arange(0., maximum + step * .5, step):
        prediction = start + distance * d
        local_channel,channel_kind,yellow_fraction=_local_channel(image,prediction,d,width,brightness,achromatic)
        measured, reason = _extension_profile(local_channel, prediction, d, width)
        single=None
        # A dark stripe can be a cast shadow or a wound groove. It must not
        # relocate this rope or become a new measured diameter.
        if measured is not None and measured['polarity']=='dark':
            reason='dark_pair_fixed_geometry_hypothesis';measured=None
        if measured is None and reason!='image_border' and not outward:
            single=_single_edge_support(local_channel,prediction,d,width)
        local_texture = _texture(gray, prediction, d, width)
        bgs = [_texture(gray, prediction + side * 1.35 * width * n, d, width) for side in (-1, 1)]
        texture_error = None; background_error = None
        if (measured is not None or single is not None) and texture['status'] == 'supported' and local_texture['status'] == 'supported':
            texture_error = _angle_difference(texture['angle_deg'], local_texture['angle_deg'])
            known_bgs = [v for v in bgs if v['status'] == 'supported']
            if known_bgs: background_error = min(_angle_difference(v['angle_deg'], local_texture['angle_deg']) for v in known_bgs)
            if (local_texture['coherence'] >= PARAMETERS['texture_minimum_coherence'] and texture_error > PARAMETERS['texture_mismatch_deg'] and
                    background_error is not None and background_error + PARAMETERS['texture_background_advantage_deg'] < texture_error):
                measured = None; single=None; reason = 'texture_follows_background_not_free_reference'
        if reason == 'image_border': stop_reason = reason; break
        cx, cy = np.rint(prediction).astype(int)
        in_surface = bool(0 <= cy < surface.shape[0] and 0 <= cx < surface.shape[1] and surface[cy, cx] != 0)
        if outward and in_surface:stop_reason='outward_ray_reaches_supplied_surface';break
        if single is not None:
            inferred_run+=step
            if inferred_run>max(12.,width*PARAMETERS['maximum_inferred_run_width']):
                single=None;reason='single_edge_inference_budget_exceeded'
            else:reason='one_raw_edge_fixed_free_line_and_width_inferred'
        elif measured is not None:inferred_run=0.
        record = {'t_px': float(distance), 'predicted_xy': prediction.tolist(), 'in_surface': in_surface,
                  'state': 'observed' if measured is not None else ('inferred' if single is not None else 'unsupported'), 'reason': reason,
                  'measured': measured, 'texture': local_texture, 'background_texture': bgs,
                  'free_texture_angle_error_deg': texture_error, 'background_texture_angle_error_deg': background_error,
                  'single_edge_evidence':single,'channel':channel_kind,'yellow_flank_fraction':yellow_fraction}
        samples.append(record)
        if measured is None and single is None:
            unsupported += step
            if first_bad is None: first_bad = prediction.copy()
            centers.append(prediction.tolist()); left.append([None, None]); right.append([None, None]); states.append('unsupported')
            if unsupported >= max(8., width * PARAMETERS['maximum_unsupported_run_width']):
                stop_reason = reason; break
        elif measured is not None:
            unsupported = 0.; first_bad = None; last_good = np.asarray(measured['center_xy'])
            centers.append(measured['center_xy']); left.append(measured['left_xy']); right.append(measured['right_xy']); states.append('observed')
        else:
            unsupported=0.
            if first_bad is None:first_bad=prediction.copy()
            centers.append(prediction.tolist());left.append((prediction-width*.5*n).tolist());right.append((prediction+width*.5*n).tolist());states.append('inferred')
    # Unsupported probes are retained in records but never painted as a rope.
    mask_left = np.asarray([[0., 0.] if a[0] is None else a for a in left])
    mask_right = np.asarray([[0., 0.] if a[0] is None else a for a in right])
    mask = _paint_pairs(gray.shape, mask_left, mask_right, max(5., width * PARAMETERS['maximum_profile_gap_width_for_mask']), [s == 'observed' for s in states])
    inferred_mask=_paint_pairs(gray.shape,mask_left,mask_right,max(5.,width*PARAMETERS['maximum_profile_gap_width_for_mask']),[s in ('observed','inferred') for s in states])
    inferred_mask[mask!=0]=0
    return {'direction_xy': d.tolist(), 'origin_xy': start.tolist(), 'centerline_xy': centers,
            'boundary_left_xy': left, 'boundary_right_xy': right, 'point_states': states,
            'support_samples': samples, 'stop_reason': stop_reason, 'visible_end_xy': last_good.tolist(),
            'bend_seed_interval_xy': [last_good.tolist(), (first_bad if first_bad is not None else last_good).tolist()],
            'observed_mask': mask,'inferred_mask':inferred_mask, 'direction_is_straight': True, 'contact_claim': False,
            'role':'outward_free_extension' if outward else 'inward_straight_candidate',
            'observed_profile_count': states.count('observed'),'inferred_profile_count':states.count('inferred'), 'unsupported_profile_count': states.count('unsupported')}


def refine_and_extend(data):
    """Pure-array entry point; seeds are saved AUTO predictions, never old bends.

    Required: original uint8 BGR, surface uint8/bool. a1_record/a2_record may be
    whole saved records or just their free_segment dictionaries. Missing or
    rejected seeds return unknown with zero masks; no manual endpoints are used.
    """
    image = data['original']; surface = data['surface']
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or surface.shape != image.shape[:2]:
        raise ValueError('original must be uint8 HxWx3; surface must match HxW')
    empty = np.zeros(surface.shape, np.uint8)
    result = {'status': 'unknown', 'free_segment': {'status': 'unknown'}, 'geometry': {'status': 'unknown'},
              'free_mask': empty.copy(), 'straight_observed_mask': empty.copy(),'straight_inferred_mask':empty.copy(), 'straight_candidates': [],'outward_extension':None,
              'texture_descriptor': {'status': 'unknown'}, 'diagnostics': {'parameters': dict(PARAMETERS),
              'old_selected_connection_used': False, 'physical_material_identity_confirmed': False},
              'source_dependencies': dependencies()}
    seed, source = _seed(data)
    if seed is None:
        result['diagnostics']['reason'] = 'no_usable_automatic_free_seed'; return result
    free, diagnostics = _remeasure(image, surface, seed)
    result['diagnostics'].update({'seed_source': source, 'outside_remeasurement': diagnostics})
    free,wide_diagnostics=_wide_remeasure(image,surface,seed,free,float(data.get('pitch') or seed['width_px']))
    result['diagnostics']['wide_remeasurement']=wide_diagnostics
    result['diagnostics']['original_seed']={'observed_segment_xy':seed['observed_segment_xy'],'width_px':seed['width_px']}
    if free is None:
        result['diagnostics']['reason'] = 'automatic_seed_not_revalidated'; return result
    if free['width_px']<PARAMETERS['narrow_seed_pitch_ratio']*float(data.get('pitch') or free['width_px']) and wide_diagnostics['status']!='whole_envelope_supported':
        result['status']='free_width_ambiguous';result['free_segment']={**free,'status':'width_ambiguous'}
        result['diagnostics']['reason']='narrow_strand_seed_without_independent_whole_width_support';return result
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    d = np.asarray(free['direction_xy']); width = free['width_px']
    points = np.asarray(free['observed_points_xy'])
    ids = np.unique(np.rint(np.linspace(0, len(points) - 1, min(9, len(points)))).astype(int))
    texture = _aggregate_texture([_texture(gray, points[i], d, width) for i in ids])
    channel, channel_name = _A2._channel(image, seed)
    toward = free['toward_surface_direction_xy']
    outward=None
    if toward is not None:
        outward=_extend(image,gray,channel,surface,free,-np.asarray(toward),texture,outward=True)
        observed=[s['measured'] for s in outward['support_samples'] if s['state']=='observed' and not s['in_surface']]
        if observed:
            # Edge names follow each measurement's normal. Reverse the names
            # only when the outward normal points against the free-line normal;
            # an unconditional swap crossed the splice polygon in one view.
            outward_d=np.asarray(outward['direction_xy'])
            outward_n=np.array([-outward_d[1],outward_d[0]])
            free_n=np.asarray(free['normal_xy'])
            swap_sides=bool(outward_n@free_n<0.)
            left_key,right_key=('right_xy','left_xy') if swap_sides else ('left_xy','right_xy')
            lp=np.vstack((np.asarray(free['boundary_left_xy']),np.asarray([m[left_key] for m in observed])))
            rp=np.vstack((np.asarray(free['boundary_right_xy']),np.asarray([m[right_key] for m in observed])))
            cp=(lp+rp)/2;order=np.argsort(cp@d);lp,rp,cp=lp[order],rp[order],cp[order]
            free['boundary_left_xy']=lp.tolist();free['boundary_right_xy']=rp.tolist();free['observed_points_xy']=cp.tolist()
            end=np.asarray(outward['visible_end_xy']);ends=np.asarray(free['observed_segment_xy']);far=int(np.argmin(ends@np.asarray(toward)));ends[far]=end
            free['observed_segment_xy']=ends.tolist();free['support_length_px']=float(np.linalg.norm(ends[1]-ends[0]))
            free['profile_sample_count']=len(cp);free['outward_extended_from_raw_edges']=True
    free_mask = _paint_pairs(surface.shape, free['boundary_left_xy'], free['boundary_right_xy'], max(6., width * .9))
    free_mask[surface != 0] = 0
    directions = [toward] if toward is not None else free['toward_surface_candidates_xy']
    candidates = [_extend(image, gray, channel, surface, free, direction, texture) for direction in directions]
    combined = empty.copy();combined_inferred=empty.copy()
    for candidate in candidates: combined |= candidate['observed_mask'];combined_inferred|=candidate['inferred_mask']
    combined_inferred[combined!=0]=0
    ends = np.asarray(free['observed_segment_xy'])
    origin = ends[int(np.argmax(ends @ np.asarray(toward)))] if toward is not None else None
    geometry = {'status': 'observed' if toward is not None else 'direction_ambiguous',
                'origin_xy': origin.tolist() if origin is not None else None, 'direction_xy': toward,
                'width_px': width, 'free_end_xy': origin.tolist() if origin is not None else None,
                'visible_end_xy': candidates[0]['visible_end_xy'] if toward is not None and candidates else None,
                'direction_candidates_xy': directions, 'observed_segment_xy': free['observed_segment_xy'],
                'contact_claim': False}
    result.update({'status': 'observed', 'free_segment': free, 'geometry': geometry, 'free_mask': free_mask,
                   'straight_observed_mask': combined,'straight_inferred_mask':combined_inferred, 'straight_candidates': candidates,'outward_extension':outward, 'texture_descriptor': texture})
    result['diagnostics'].update({'reason': 'outside_original_pairs_revalidated', 'channel': channel_name,
                                  'free_pixels': int(np.count_nonzero(free_mask)),
                                  'straight_measured_pixels': int(np.count_nonzero(combined)),
                                  'interpretation': 'measured optical free stripe plus tested straight hypotheses; no bend/contact decision'})
    return result
