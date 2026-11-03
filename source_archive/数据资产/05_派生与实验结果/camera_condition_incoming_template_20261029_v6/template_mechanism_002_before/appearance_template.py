"""A6: learn this image's free-rope appearance, then measure local candidates.

No F3 lanes, selected old bends, labels, hardware or candidate-painted images are
used. A5 is imported read-only for raw ribbon validity/outer-edge measurements;
the new evidence is a measured free-rope profile + texture distribution. Matching
is local appearance compatibility, never proof of same material identity.
"""
from __future__ import annotations

from pathlib import Path
import importlib.util
import math
import cv2
import numpy as np


_A5_PATH = Path(__file__).resolve().parent / 'photometric_ribbon.py'
_SPEC = importlib.util.spec_from_file_location('_a6_readonly_a5_ribbon', _A5_PATH)
_A5 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_A5)
build_fields = _A5.build_fields

PARAMETERS = {
    'maximum_template_samples': 48,
    'minimum_template_samples': 4,
    'minimum_template_span_widths': 1.5,
    'tangent_half_widths': 1.25,
    'tangent_samples': 25,
    'normal_half_widths': 1.30,
    'normal_samples': 41,
    'template_surface_overlap_maximum': 0.,
    'minimum_texture_std_raw': 1.,
    'minimum_texture_energy': .10,
    'minimum_texture_coherence': .15,
    'minimum_template_profile_match': .45,
    'minimum_supported_score': .40,
    'neighbor_distance_pitch': 1.10,
    'minimum_raw_profile_contrast': 2.,
}


def dependencies():
    return [str(_A5_PATH)]


def _arrays(xy, tangents, widths):
    p = np.asarray(xy, np.float32).reshape(-1, 2)
    n = len(p)
    t = np.asarray(tangents, np.float32)
    if t.shape == (2,):
        t = np.repeat(t[None], n, axis=0)
    t = t.reshape(-1, 2)
    w = np.broadcast_to(np.asarray(widths, np.float32), (n,)).copy()
    if len(t) != n or not np.isfinite(np.r_[p.ravel(), t.ravel(), w]).all() or np.any(w <= 1):
        raise ValueError('Finite Nx2 centers/tangents and widths > 1 pixel required')
    norm = np.linalg.norm(t, axis=1)
    if np.any(norm < 1e-6):
        raise ValueError('Nonzero tangents required')
    t = t / norm[:, None]
    # Canonical image orientation makes descriptors invariant to path traversal
    # reversal; changing the material identity is not part of this operation.
    reverse = (t[:, 0] < 0) | ((abs(t[:, 0]) < 1e-7) & (t[:, 1] < 0))
    t[reverse] *= -1
    return p, t, w


def _describe(fields, xy, tangents, widths, surface=None):
    """Raw-pixel descriptors; no values are synthesized from proposed curves."""
    p, t, w = _arrays(xy, tangents, widths)
    n = np.c_[-t[:, 1], t[:, 0]]
    along = np.linspace(-1.25, 1.25, PARAMETERS['tangent_samples'], dtype=np.float32)
    across = np.linspace(-1.30, 1.30, PARAMETERS['normal_samples'], dtype=np.float32)
    q = p[:, None, None, :] + w[:, None, None, None] * (
        along[None, :, None, None] * t[:, None, None, :] + across[None, None, :, None] * n[:, None, None, :])
    raw = _A5._sample(fields['gray'], q)
    h, ww = fields['image_shape']
    valid = (q[..., 0] >= 1) & (q[..., 0] < ww-2) & (q[..., 1] >= 1) & (q[..., 1] < h-2)
    # Trim along-axis extrema, retaining a measured cross-sectional profile.
    profile = np.sort(raw, axis=1)[:, 4:-4, :].mean(axis=1)
    body = abs(across) <= .30
    left = (across >= -1.20) & (across <= -.85)
    right = (across >= .85) & (across <= 1.20)
    b = profile[:, body].mean(axis=1)
    l = profile[:, left].mean(axis=1)
    r = profile[:, right].mean(axis=1)
    weight = (across-float(across[left].mean())) / float(across[right].mean()-across[left].mean())
    background = l[:, None]*(1-weight)[None] + r[:, None]*weight[None]
    delta = b-(l+r)/2
    polarity = np.where(delta >= 0, 1., -1.)
    normalized = np.clip((profile-background)*polarity[:, None]/np.maximum(abs(delta), 2.)[:, None], -3., 3.)
    # Remove static cross-sectional shading before inspecting strand texture.
    # No location/phase matching: a rope can translate by a strand period.
    interior = raw[:, :, body]
    detail = interior-interior.mean(axis=1, keepdims=True)
    detail -= detail.mean(axis=2, keepdims=True)
    dt = float(along[1]-along[0]); dn = float(across[1]-across[0])
    gt = np.diff(detail, axis=1)[:, :, :-1]/(dt*w[:, None, None])
    gn = np.diff(detail, axis=2)[:, :-1, :]/(dn*w[:, None, None])
    tt = np.mean(gt*gt, axis=(1, 2)); nn = np.mean(gn*gn, axis=(1, 2)); tn = np.mean(gt*gn, axis=(1, 2))
    energy = tt+nn
    coherence = np.sqrt((tt-nn)**2+4*tn*tn)/np.maximum(energy, 1e-8)
    angle = .5*np.arctan2(2*tn, tt-nn)+np.pi/2
    std = detail.std(axis=(1, 2))
    window = np.hanning(detail.shape[1])[:, None]*np.hanning(detail.shape[2])[None, :]
    power = abs(np.fft.rfft2(detail*window[None], axes=(1, 2)))**2
    fa = np.fft.fftfreq(detail.shape[1], d=dt)[:, None]
    fn = np.fft.rfftfreq(detail.shape[2], d=dn)[None, :]
    frequency = np.hypot(fa, fn)
    bands = np.array([power[:, (frequency >= lo) & (frequency < hi)].sum(axis=1)
                      for lo, hi in [(0.30, 1.5), (1.5, 3.), (3., 6.), (6., 30.)]]).T
    spectrum = bands/np.maximum(bands.sum(axis=1, keepdims=True), 1e-8)
    texture_valid = (std >= PARAMETERS['minimum_texture_std_raw']) & (energy >= PARAMETERS['minimum_texture_energy'])
    texture_valid &= coherence >= PARAMETERS['minimum_texture_coherence']
    output = {
        'raw_profile': profile, 'normalized_profile': normalized,
        'body_gray_raw': b, 'side_gray_raw': np.c_[l, r], 'delta_gray_raw': delta,
        'polarity': polarity, 'texture_angle_relative_axis_rad': angle,
        'texture_coherence': coherence, 'texture_energy_raw': energy, 'texture_std_raw': std,
        'texture_spectrum': spectrum, 'texture_valid': texture_valid,
        'valid_fraction': valid.mean(axis=(1, 2)), 'normal_coordinates_widths': across,
    }
    if surface is not None:
        output['surface_overlap_fraction'] = (_A5._sample((surface > 0).astype(np.float32), q) > .01).mean(axis=(1, 2))
    return output


def _distribution(a):
    a = np.asarray(a, float)
    middle = np.median(a, axis=0)
    return {'median': middle.tolist(), 'mad': np.median(abs(a-middle), axis=0).tolist()}


def build_template(data, free):
    """Build a JSON-safe self template from measured, uncontaminated free pixels.

    data: original uint8 BGR, surface uint8, pitch.
    free: A4/A5-style free_segment with observed_points_xy and paired edges.
    No fallback to predicted in-surface points or endpoints is permitted.
    """
    out = {'status': 'unknown', 'reason': 'no_measured_free_points', 'parameters': dict(PARAMETERS),
           'measurement_source': 'original_outside_surface_measured_free_points',
           'same_material_identity_verified': False, 'F3_used': False}
    segment = free.get('free_segment', {})
    if segment.get('status') != 'observed':
        return out
    points = np.asarray(segment.get('observed_points_xy', []), float).reshape(-1, 2)
    left = np.asarray(segment.get('boundary_left_xy', []), float).reshape(-1, 2)
    right = np.asarray(segment.get('boundary_right_xy', []), float).reshape(-1, 2)
    if len(points) < PARAMETERS['minimum_template_samples'] or len(left) != len(points) or len(right) != len(points):
        out['reason'] = 'insufficient_measured_paired_free_points'; return out
    direction = segment.get('direction_xy')
    if direction is None:
        out['reason'] = 'no_measured_free_direction'; return out
    widths = np.linalg.norm(right-left, axis=1)
    good = np.isfinite(np.c_[points, widths]).all(axis=1) & (widths > 1.)
    ids = np.flatnonzero(good)
    if not len(ids):
        out['reason'] = 'invalid_measured_free_points'; return out
    ids = ids[np.unique(np.rint(np.linspace(0, len(ids)-1, min(len(ids), PARAMETERS['maximum_template_samples']))).astype(int))]
    points = points[ids]; widths = widths[ids]
    fields = build_fields(data['original'], float(np.median(widths)), float(data['pitch']))
    desc = _describe(fields, points, direction, widths, data['surface'])
    raw = _A5.sample_ribbon(fields, points, direction, widths)
    outside = desc['surface_overlap_fraction'] <= PARAMETERS['template_surface_overlap_maximum']
    outside &= desc['valid_fraction'] >= .97
    envelope = raw['supported'] | raw['saturated_outline_supported']
    use = outside & envelope
    out.update(candidate_sample_count=len(points), measured_point_indices=ids.tolist(),
               excluded_surface_overlap_count=int(np.sum(~outside)), accepted_sample_count=int(use.sum()))
    if np.count_nonzero(use) < PARAMETERS['minimum_template_samples']:
        out['reason'] = 'too_few_uncontaminated_outer_edge_samples'; return out
    projected = points[use] @ (np.asarray(direction)/np.linalg.norm(direction))
    width_ref = float(np.median(widths[use])); span = float(np.ptp(projected))
    out['measured_span_px'] = span
    if span < PARAMETERS['minimum_template_span_widths']*width_ref:
        out['reason'] = 'uncontaminated_span_too_short'; return out
    profile_use = use & raw['reliable'] & (abs(desc['delta_gray_raw']) >= PARAMETERS['minimum_raw_profile_contrast'])
    texture_use = profile_use & desc['texture_valid']
    full = bool(np.count_nonzero(profile_use) >= PARAMETERS['minimum_template_samples'])
    texture_ok = bool(np.count_nonzero(texture_use) >= PARAMETERS['minimum_template_samples'])
    out.update(status='ready' if full else 'outline_only', reason='outside_original_self_template',
               reference_width_px=width_ref, width_distribution=_distribution(widths[use]),
               measured_edge_width_distribution=_distribution(raw['measured_edge_width_px'][use]),
               paired_edge_raw_distribution=_distribution(raw['paired_edge_contrast_raw'][use]),
               source_points_xy=points[use].tolist(), source_widths_px=widths[use].tolist(),
               source_point_indices=ids[use].tolist(), accepted_surface_overlap_fraction=desc['surface_overlap_fraction'][use].tolist(),
               raw_body_distribution=_distribution(desc['body_gray_raw'][use]),
               raw_side_distribution=_distribution(desc['side_gray_raw'][use]),
               raw_profile_distribution=_distribution(desc['raw_profile'][use]),
               normalized_profile_distribution=_distribution(desc['normalized_profile'][profile_use]) if full else None,
               normal_coordinates_widths=desc['normal_coordinates_widths'].tolist(),
               shadow_width_distribution=_distribution(np.c_[raw['left_shadow_width_px'][use], raw['right_shadow_width_px'][use]]/widths[use, None]),
               profile_available=full, profile_sample_count=int(profile_use.sum()),
               texture_available=texture_ok, texture_sample_count=int(texture_use.sum()),
               texture_spectrum_distribution=_distribution(desc['texture_spectrum'][texture_use]) if texture_ok else None)
    if texture_ok:
        weight = desc['texture_coherence'][texture_use]
        angle = desc['texture_angle_relative_axis_rad'][texture_use]
        vector = np.sum(weight*np.exp(2j*angle))/max(float(weight.sum()), 1e-8)
        out.update(texture_angle_relative_axis_rad=float(np.angle(vector)/2),
                   texture_orientation_concentration=float(abs(vector)),
                   texture_coherence_distribution=_distribution(weight),
                   texture_std_raw_distribution=_distribution(desc['texture_std_raw'][texture_use]))
    return out


def _evaluate_core(fields, xy, tangents, widths, template):
    p, t, w = _arrays(xy, tangents, widths)
    raw = _A5.sample_ribbon(fields, p, t, w)
    desc = _describe(fields, p, t, w)
    n = len(p)
    ready = template.get('status') in ('ready', 'outline_only')
    profile_valid = raw['reliable'] & (abs(desc['delta_gray_raw']) >= PARAMETERS['minimum_raw_profile_contrast'])
    profile_valid &= bool(ready and template.get('profile_available'))
    profile_score = np.zeros(n, float)
    profile_error = np.zeros(n, float)  # availability flag, not NaN, carries unknown
    if template.get('profile_available'):
        distribution = template['normalized_profile_distribution']
        ref = np.asarray(distribution['median']); spread = np.asarray(distribution['mad'])
        current = desc['normalized_profile']
        # Both amplitude normalization and signed polarity are measured, not
        # selected from a bank of best-fitting bright/dark peaks.
        error = np.mean(abs(current-ref)/(.40+spread), axis=1)
        cosine = np.sum(current*ref, axis=1)/np.maximum(np.linalg.norm(current, axis=1)*np.linalg.norm(ref), 1e-8)
        profile_error = error
        profile_score = np.where(profile_valid, .55*np.clip(cosine, 0, 1)+.45*np.exp(-error), 0.)
    texture_valid = desc['texture_valid'] & raw['reliable'] & bool(ready and template.get('texture_available'))
    texture_score = np.zeros(n, float); angle_error = np.zeros(n, float); spectral_match = np.zeros(n, float)
    if template.get('texture_available'):
        ref_angle = float(template['texture_angle_relative_axis_rad'])
        diff = .5*np.angle(np.exp(2j*(desc['texture_angle_relative_axis_rad']-ref_angle)))
        angle_error = abs(diff)
        orient = np.clip((1+np.cos(2*diff))/2, 0, 1)
        spectrum = np.asarray(template['texture_spectrum_distribution']['median'])
        spectrum /= max(float(spectrum.sum()), 1e-8)
        spectral_match = np.sum(np.sqrt(np.maximum(0., desc['texture_spectrum']*spectrum)), axis=1)
        concentration = float(template.get('texture_orientation_concentration', 0.))
        texture_score = np.where(texture_valid, concentration*.6*orient+(1-.6*concentration)*spectral_match, 0.)
    reference_width = float(template.get('reference_width_px', fields['width_px']))
    observed_ratio = raw['measured_edge_width_px']/reference_width
    proposed_ratio = w/reference_width
    width_score = np.exp(-abs(np.log(np.maximum(observed_ratio, 1e-6)))/.55)
    width_valid = (proposed_ratio >= .5) & (proposed_ratio <= 1.8) & (observed_ratio >= .45) & (observed_ratio <= 2.)
    # Texture absence is not evidence for or against identity. Texture can aid
    # available profile evidence; identical periodic neighboring ropes remain
    # locally indistinguishable and must be kept ambiguous by the caller.
    template_match = np.where(texture_valid, .75*profile_score+.25*texture_score, profile_score)
    local_score = .50*template_match+.35*raw['score']+.15*width_score
    local_score = np.where(profile_valid, local_score, 0.)
    outline = raw['saturated_outline_supported'] & bool(ready)
    local_score = np.maximum(local_score, np.where(outline, .45*raw['saturated_outline_score']+.10*width_score, 0.))
    valid = bool(ready) & (profile_valid | outline)
    supported = valid & width_valid & (raw['supported'] | outline)
    supported &= (outline | (profile_score >= PARAMETERS['minimum_template_profile_match'])) & (local_score >= PARAMETERS['minimum_supported_score'])
    score = np.where(valid, np.clip(local_score, 0., 1.), 0.).astype(np.float32)
    reason = np.zeros(n, np.uint16)
    for bit, fail in [(1, not ready), (2, ~raw['reliable']), (4, ~profile_valid), (8, ~width_valid),
                      (16, ~(raw['supported'] | outline)), (32, profile_valid & (profile_score < PARAMETERS['minimum_template_profile_match']))]:
        reason |= np.where(fail, bit, 0).astype(np.uint16)
    return {'score': score, 'valid': valid, 'supported': supported,
            'profile_match': profile_score, 'profile_valid': profile_valid, 'profile_error': profile_error,
            'texture_match': texture_score, 'texture_valid': texture_valid,
            'texture_angle_error_rad': angle_error, 'texture_spectral_match': spectral_match,
            'width_match': width_score, 'width_valid': width_valid,
            'measured_width_px': raw['measured_edge_width_px'], 'proposed_width_ratio': proposed_ratio,
            'raw_pair': raw['paired_edge_contrast_raw'], 'raw_left_edge': raw['left_edge_contrast_raw'], 'raw_right_edge': raw['right_edge_contrast_raw'],
            'delta_gray': desc['delta_gray_raw'], 'body_gray_raw': desc['body_gray_raw'], 'side_gray_raw': desc['side_gray_raw'],
            'raw_profile': desc['raw_profile'], 'normalized_profile': desc['normalized_profile'],
            'texture_angle_relative_axis_rad': desc['texture_angle_relative_axis_rad'], 'texture_coherence': desc['texture_coherence'],
            'texture_energy_raw': desc['texture_energy_raw'], 'texture_spectrum': desc['texture_spectrum'],
            'left_shadow_width_px': raw['left_shadow_width_px'], 'right_shadow_width_px': raw['right_shadow_width_px'],
            'polarity': desc['polarity'], 'raw_reliable': raw['reliable'], 'raw_supported': raw['supported'],
            'body_saturation_fraction': raw['body_saturation_fraction'], 'saturated_outline_supported': outline,
            'along_consistency': raw['along_consistency'], 'body_coverage': raw['body_coverage'],
            'unknown_bits': reason,
            'unknown_legend': {1: 'template_unavailable', 2: 'raw_mean_unreliable', 4: 'profile_unavailable',
                               8: 'width_incompatible', 16: 'outer_envelope_unsupported', 32: 'free_profile_mismatch'},
            'measurement_source': 'original_pixels_compared_to_measured_free_template', 'identity_verified': False}


def evaluate_template(fields, xy, tangents, widths, template, compare_neighbors=True):
    """Return N arrays. No candidate generation, F3, I/O or identity decision.

    score is local self-template compatibility, not probability. valid indicates
    usable raw/template evidence; supported also requires the raw whole envelope.
    Neighbor comparison is reported separately and only valid for all three
    non-clipped profile measurements. It is not folded into score implicitly.
    """
    p, t, w = _arrays(xy, tangents, widths)
    if not len(p):
        raise ValueError('At least one point required')
    if not compare_neighbors:
        out = _evaluate_core(fields, p, t, w, template)
        out.update(neighbor_advantage=np.zeros(len(p)), neighbor_comparison_valid=np.zeros(len(p), bool))
        return out
    normal = np.c_[-t[:, 1], t[:, 0]]
    distance = float(fields['pitch_px'])*PARAMETERS['neighbor_distance_pitch']
    all_points = np.vstack([p, p-distance*normal, p+distance*normal])
    measured = _evaluate_core(fields, all_points, np.tile(t, (3, 1)), np.tile(w, 3), template)
    n = len(p)
    out = {k: (v[:n].copy() if isinstance(v, np.ndarray) else v) for k, v in measured.items()}
    available = measured['profile_valid'][:n] & measured['profile_valid'][n:2*n] & measured['profile_valid'][2*n:]
    left = measured['score'][n:2*n]; right = measured['score'][2*n:]
    out.update(neighbor_advantage=np.where(available, out['score']-np.maximum(left, right), 0.),
               neighbor_comparison_valid=available, neighbor_left_score=left, neighbor_right_score=right,
               neighbor_left_supported=measured['supported'][n:2*n], neighbor_right_supported=measured['supported'][2*n:],
               neighbor_distance_px=distance)
    return out
