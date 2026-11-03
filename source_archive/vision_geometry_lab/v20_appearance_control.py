"""Frozen flange-appearance controls, NOT rotational phase or layer events.

Only independent flange descriptors select past partners. Archived side-profile
evidence is compared afterwards, without influencing partner choice. No model
inference, truth labels, live hardware, turn counting or actuator commands.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import datetime
from pathlib import Path

import cv2
import numpy as np

from vision_geometry_lab import v19_front_trial as v19


LAB = Path(__file__).resolve().parent
BASE = LAB / 'outputs/v20_material_audit/appearance_control'
HISTORIES = ('local_2_to_12s', 'fixed_startup_5_to_10s')
PARTNERS = ('appearance_best', 'distinct_runner_up', 'temporal_midpoint_control')
RULES = dict(
    roi_count=2, roi_coordinates='canonical_reference_xyxy_half_open_integer',
    extraction='native_reference_patch_affine_remap_only_then_area_resize',
    descriptor_width=64, descriptor_height=32, highpass_sigma=3.,
    minimum_highpass_std_gray=2., maximum_clipped_fraction=.05,
    invalid_support='masked_normalized_Gaussian_then_common_supported_pixel_cosine',
    pose_required=True, minimum_delay_s=2., local_maximum_delay_s=12.,
    fixed_reference_start_s=5., fixed_reference_end_s=10., fixed_query_after_s=12.,
    ranking='minimum_of_two_independent_ROI_cosines', minimum_score=.85,
    runner_exclusion_radius_s=.3, minimum_runner_margin=.03,
    maximum_ROI_argmax_time_difference_s=.2,
    missing_distinct_runner='unknown_not_accept', tie_break='earliest_source_time',
    control='lower_median_candidate_time_after_excluding_best_and_runner; not guaranteed dissimilar',
    comparison='both_endpoints_V19_quality_and_finite_direct_normal_offset',
    partitions='four_contiguous_original_row_index_quarters_no_repacking',
    minimum_common_rows_per_partition=10, summary_bucket_seconds=5., summary_end_s=60.,
    final_bucket_end_inclusive=True,
    profile_similarity='raw_cosine_and_mean_removed_cosine_no_threshold_gate',
    color_diagnostics='64x32_uniform_reference_center_samples_BGR_mean_std_HSV_saturation_and_gray_clipping_not_gates',
    reference_update=False, threshold_search=False, truth_input=False,
    material_identity_confirmed=False, mechanical_phase_confirmed=False,
    physical_new_layer_confirmed=False, deployment_allowed=False,
    hardware_access=False)
STOP_RULE = dict(
    insufficient_or_ambiguous_match='Unknown; do not reduce thresholds or select a more favorable ROI after replay.',
    ROI_scope='Manually selected first-frame development ROIs, not a general detector or training ROI.',
    forbidden_inferences=['material_identity_confirmation', 'mechanical_phase_or_revolution_confirmation',
                          'physical_new_layer_start_t1_confirmation', 'first_revolution_end_t2_confirmation'],
    deployment='No deployment or device actuation from this nuisance-control experiment.',
    threshold_tuning=False)


def validate_rois(rois, width, height):
    a = np.asarray(rois)
    if a.shape != (2, 4) or a.dtype.kind not in 'iu':
        raise ValueError('exactly two integer canonical xyxy ROIs required')
    a = a.astype(np.int64)
    for x1, y1, x2, y2 in a:
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError('ROI outside reference image or empty')
        if x2-x1 < 64 or y2-y1 < 32:
            raise ValueError('ROI must support downsampling to 64 by 32')
    if max(a[:, 0]) < min(a[:, 2]) and max(a[:, 1]) < min(a[:, 3]):
        raise ValueError('ROIs must not overlap')
    return a.tolist()


def patch_descriptor(gray, ref_to_current, roi, pose_valid=True):
    """Sample just one reference patch; return explicit pixel-support evidence."""
    size = (RULES['descriptor_width'], RULES['descriptor_height'])
    empty = dict(valid=False, reason='pose_unavailable', descriptor=np.zeros(2048, np.float32),
                 support=np.zeros(2048, bool), patch=np.full((32, 64), np.nan, np.float32),
                 clipped_fraction=np.nan, highpass_std_gray=np.nan)
    if not pose_valid:
        return empty
    gray = np.asarray(gray)
    A = np.asarray(ref_to_current, float)
    if gray.ndim != 2 or gray.dtype.kind not in 'uif':
        raise ValueError('grayscale numeric image required')
    if A.shape != (2, 3) or not np.isfinite(A).all() or abs(np.linalg.det(A[:, :2])) < 1e-8:
        return {**empty, 'reason': 'invalid_pose'}
    r = np.asarray(roi)
    if r.shape != (4,) or r.dtype.kind not in 'iu' or r[2] <= r[0] or r[3] <= r[1]:
        raise ValueError('integer nonempty xyxy ROI required')
    x1, y1, x2, y2 = map(int, r)
    xx, yy = np.meshgrid(np.arange(x1, x2, dtype=np.float32), np.arange(y1, y2, dtype=np.float32))
    mx = (A[0, 0]*xx+A[0, 1]*yy+A[0, 2]).astype(np.float32)
    my = (A[1, 0]*xx+A[1, 1]*yy+A[1, 2]).astype(np.float32)
    h, w = gray.shape
    supported = (mx >= 0) & (mx <= w-1) & (my >= 0) & (my <= h-1)
    clipped = 1.-float(supported.mean())
    if clipped > RULES['maximum_clipped_fraction']+1e-12:
        return {**empty, 'reason': 'patch_clipped', 'clipped_fraction': clipped}
    sampled = cv2.remap(gray.astype(np.float32, copy=False), mx, my, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0.)
    supported &= np.isfinite(sampled)
    if not supported.any():
        return {**empty, 'reason': 'no_pixel_support', 'clipped_fraction': clipped}
    # Area support is transformed alongside gray values; invalid border samples
    # cannot become artificial dark texture. Partial output cells are excluded.
    weight = cv2.resize(supported.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    total = cv2.resize(np.where(supported, sampled, 0.), size, interpolation=cv2.INTER_AREA)
    small = np.divide(total, weight, out=np.zeros_like(total), where=weight > 0.)
    mask = weight >= 1.-1e-6
    sm = cv2.GaussianBlur(np.where(mask, small, 0.), (0, 0), RULES['highpass_sigma'])
    sw = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), RULES['highpass_sigma'])
    smooth = np.divide(sm, sw, out=np.zeros_like(sm), where=sw > 1e-9)
    high = small-smooth
    if not mask.any():
        return {**empty, 'reason': 'no_resized_support', 'clipped_fraction': clipped}
    high -= float(high[mask].mean())
    std = float(high[mask].std())
    result = {**empty, 'support': mask.ravel(), 'patch': np.where(mask, small, np.nan),
              'clipped_fraction': clipped, 'highpass_std_gray': std}
    if not np.isfinite(std) or std < RULES['minimum_highpass_std_gray']:
        return {**result, 'reason': 'weak_texture'}
    high = np.where(mask, high, 0.).ravel().astype(np.float32)
    norm = float(np.linalg.norm(high))
    if norm <= 0.:
        return {**result, 'reason': 'zero_descriptor_norm'}
    return {**result, 'valid': True, 'reason': 'descriptor_only_not_material_identity',
            'descriptor': high/norm}


def _times(times):
    t = np.asarray(times, float)
    if t.ndim != 1 or not len(t) or not np.isfinite(t).all() or np.any(t < 0.) or np.any(np.diff(t) <= 0.):
        raise ValueError('finite strictly increasing nonnegative times required')
    return t


def color_diagnostics(bgr, ref_to_current, roi, pose_valid=True):
    """Sparse patch color/exposure summaries; no occlusion/material classifier."""
    if not pose_valid:
        return dict(available=False, reason='pose_unavailable')
    a = np.asarray(ref_to_current, float)
    if a.shape != (2, 3) or not np.isfinite(a).all() or abs(np.linalg.det(a[:, :2])) < 1e-8:
        return dict(available=False, reason='invalid_pose')
    x1, y1, x2, y2 = roi
    xx, yy = np.meshgrid(x1+(np.arange(64)+.5)*(x2-x1)/64.-.5,
                         y1+(np.arange(32)+.5)*(y2-y1)/32.-.5)
    mx = (a[0, 0]*xx+a[0, 1]*yy+a[0, 2]).astype(np.float32)
    my = (a[1, 0]*xx+a[1, 1]*yy+a[1, 2]).astype(np.float32)
    h, w = bgr.shape[:2]
    mask = (mx >= 0) & (mx <= w-1) & (my >= 0) & (my <= h-1)
    sampled = cv2.remap(bgr, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if not mask.any():
        return dict(available=False, reason='no_color_support')
    colors = sampled[mask].astype(float)
    gray = cv2.cvtColor(sampled, cv2.COLOR_BGR2GRAY)[mask]
    hsv = cv2.cvtColor(sampled, cv2.COLOR_BGR2HSV)[mask]
    return dict(available=True, sample_count=int(mask.sum()), bgr_mean=colors.mean(axis=0).tolist(),
        bgr_std=colors.std(axis=0).tolist(), gray_mean=float(gray.mean()),
        gray_below_or_equal_5_fraction=float(np.mean(gray <= 5)),
        gray_above_or_equal_250_fraction=float(np.mean(gray >= 250)),
        mean_HSV_saturation_0_to_255=float(hsv[:, 1].mean()),
        material_identity_or_occlusion_classified=False)


def eligible_history(times, current, history):
    t = _times(times)
    if not 0 <= current < len(t):
        raise ValueError('current index outside timeline')
    delay = t[current]-t
    eligible = (np.arange(len(t)) < current) & (delay >= RULES['minimum_delay_s']-1e-9)
    if history == HISTORIES[0]:
        eligible &= delay <= RULES['local_maximum_delay_s']+1e-9
    elif history == HISTORIES[1]:
        eligible &= ((t >= RULES['fixed_reference_start_s']-1e-9)
                     & (t <= RULES['fixed_reference_end_s']+1e-9)
                     & (t[current] > RULES['fixed_query_after_s']))
    else:
        raise ValueError('unknown history')
    return np.flatnonzero(eligible)


def cosine_common(a, b, support_a, support_b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.asarray(support_a, bool) & np.asarray(support_b, bool) & np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return np.nan
    den = np.linalg.norm(a[mask])*np.linalg.norm(b[mask])
    return float(np.clip(np.dot(a[mask], b[mask])/den, -1., 1.)) if den > 1e-12 else np.nan


def select_match(times, descriptors, supports, valid, current, history):
    """Appearance-only, past-only selection; no profile/quality/truth arguments."""
    t = _times(times)
    d, s, v = np.asarray(descriptors), np.asarray(supports, bool), np.asarray(valid, bool)
    if d.ndim != 3 or d.shape[:2] != (len(t), 2) or s.shape != d.shape or v.shape != (len(t), 2):
        raise ValueError('descriptor/support/valid shape mismatch')
    pool = eligible_history(t, current, history)
    out = dict(history=history, current_index=int(current), time_s=float(t[current]),
               accepted=False, reason='current_descriptor_unavailable', best_index=None,
               runner_index=None, control_index=None, eligible_time_count=int(len(pool)),
               descriptor_candidate_count=0, best_scores=None, best_min_score=None,
               runner_scores=None, runner_min_score=None, runner_margin=None,
               roi_best_indices=None, roi_best_time_difference_s=None)
    if not v[current].all():
        return out
    pool = pool[v[pool].all(axis=1)]
    scores = np.array([[cosine_common(d[current, k], d[j, k], s[current, k], s[j, k])
                        for k in range(2)] for j in pool], float).reshape((-1, 2))
    good = np.isfinite(scores).all(axis=1)
    pool, scores = pool[good], scores[good]
    out['descriptor_candidate_count'] = int(len(pool))
    if not len(pool):
        return {**out, 'reason': 'no_past_descriptor_candidate'}
    ranks = scores.min(axis=1)
    # Pool is chronological; numpy's first argmax is deterministic earliest tie.
    best_pos = int(np.argmax(ranks)); best = int(pool[best_pos])
    roi_best = [int(pool[np.argmax(scores[:, k])]) for k in range(2)]
    disagreement = float(abs(t[roi_best[0]]-t[roi_best[1]]))
    out.update(best_index=best, best_scores=scores[best_pos].tolist(), best_min_score=float(ranks[best_pos]),
               roi_best_indices=roi_best, roi_best_time_difference_s=disagreement,
               best_time_s=float(t[best]), delay_s=float(t[current]-t[best]))
    other = np.flatnonzero(abs(t[pool]-t[best]) > RULES['runner_exclusion_radius_s']+1e-9)
    runner = None
    if len(other):
        r = int(other[np.argmax(ranks[other])]); runner = int(pool[r])
        out.update(runner_index=runner, runner_scores=scores[r].tolist(), runner_min_score=float(ranks[r]),
                   runner_margin=float(ranks[best_pos]-ranks[r]), runner_time_s=float(t[runner]))
    controls = [int(j) for j in pool if int(j) not in (best, runner)]
    if controls:
        control = controls[(len(controls)-1)//2]
        cpos = int(np.flatnonzero(pool == control)[0])
        out.update(control_index=control, control_time_s=float(t[control]),
                   control_scores=scores[cpos].tolist(), control_is_guaranteed_nonmatch=False)
    reasons = []
    if ranks[best_pos] < RULES['minimum_score']-1e-12:
        reasons.append('appearance_score_below_threshold')
    if runner is None:
        reasons.append('no_isolated_runner_margin')
    elif out['runner_margin'] < RULES['minimum_runner_margin']-1e-12:
        reasons.append('appearance_partner_ambiguous')
    if disagreement > RULES['maximum_ROI_argmax_time_difference_s']+1e-9:
        reasons.append('ROI_partner_disagreement')
    return {**out, 'accepted': not reasons,
            'reason': ';'.join(reasons) if reasons else 'similar_appearance_only_not_physical_phase'}


def compare_pair(offsets, profiles, quality, current, past):
    """Compare immutable same-row measurements AFTER partner selection."""
    off, pro, q = np.asarray(offsets, float), np.asarray(profiles, float), np.asarray(quality, bool)
    if off.ndim != 2 or q.shape != off.shape or pro.shape != (*off.shape, 131):
        raise ValueError('offset/profile/quality shape mismatch')
    if not 0 <= past < current < len(off):
        raise ValueError('pair must be strictly past-to-current')
    common = q[current] & q[past] & np.isfinite(off[current]) & np.isfinite(off[past])
    delta = np.where(common, off[current]-off[past], np.nan)
    raw, centered = np.full(off.shape[1], np.nan), np.full(off.shape[1], np.nan)
    for j in np.flatnonzero(common):
        a, b = pro[current, j], pro[past, j]
        if np.isfinite(a).all() and np.isfinite(b).all():
            mask = np.ones(131, bool)
            raw[j] = cosine_common(a, b, mask, mask)
            centered[j] = cosine_common(a-a.mean(), b-b.mean(), mask, mask)
    partitions = []
    for part, ids in enumerate(np.array_split(np.arange(off.shape[1]), 4)):
        n = int(common[ids].sum()); enough = n >= RULES['minimum_common_rows_per_partition']
        rc, cc = np.isfinite(raw[ids]), np.isfinite(centered[ids])
        partitions.append(dict(partition=part, original_row_indices=ids.tolist(), common_rows=n,
            valid=enough, median_offset_delta_px=float(np.nanmedian(delta[ids])) if enough else None,
            raw_profile_rows=int(rc.sum()), centered_profile_rows=int(cc.sum()),
            median_raw_profile_cosine=float(np.nanmedian(raw[ids])) if rc.sum() >= 10 else None,
            median_centered_profile_cosine=float(np.nanmedian(centered[ids])) if cc.sum() >= 10 else None))
    return dict(common=common, delta=delta, raw_profile_cosine=raw,
                centered_profile_cosine=centered, partitions=partitions)


def _read(path):
    return v19._read(path)


def _write(path, value):
    return v19._write(path, value)


def _source_paths():
    return [Path(__file__), LAB/'v19_front_trial.py']


def _dependency_paths(v19_base, upstream_base):
    return [Path(v19_base)/'protocol.json', Path(upstream_base)/'protocol.json',
            *[Path(v19_base)/side/'predictions.json' for side in ('left', 'right')],
            *[Path(upstream_base)/side/name for side in ('left', 'right') for name in ('index.json', 'measurement.npz')]]


def _load_source(v19_base, upstream_base, side):
    index, arrays = v19._load_source(upstream_base, side)
    with np.load(Path(upstream_base)/side/'measurement.npz', allow_pickle=False) as z:
        arrays['transforms'] = z['transforms'].copy()
    pred = _read(Path(v19_base)/side/'predictions.json')
    if (pred['status'] != 'complete' or pred['protocol_sha256'] != v19.sha(Path(v19_base)/'protocol.json')
            or pred['source_measurement_sha256'] != v19.sha(Path(upstream_base)/side/'measurement.npz')
            or pred['source_index_sha256'] != v19.sha(Path(upstream_base)/side/'index.json')
            or pred['video'] != index['video']):
        raise ValueError('V19 complete source binding invalid')
    if len(pred['records']) != len(index['records']):
        raise ValueError('V19 source record count mismatch')
    for a, b in zip(pred['records'], index['records']):
        if a['frame_index'] != b['frame_index'] or abs(a['time_s']-b['time_s']) > 1e-9:
            raise ValueError('V19 source timeline mismatch')
    q = np.asarray([r['quality_mask'] for r in pred['records']])
    if q.dtype.kind != 'b' or q.shape != arrays['normal_offset'].shape:
        raise ValueError('V19 quality schema invalid')
    if arrays['transforms'].shape != (len(q), 2, 3):
        raise ValueError('pose transform shape invalid')
    arrays['quality'] = q
    return index, arrays


def freeze(rois, base=BASE, v19_base=v19.BASE):
    base, v19_base = Path(base), Path(v19_base)
    if base.exists():
        raise FileExistsError('fresh appearance-control output directory required')
    p19 = v19.verify(v19_base); upstream = Path(p19['upstream_base'])
    if set(rois) != {'left', 'right'}:
        raise ValueError('left and right reference ROIs both required')
    videos = {}; checked = {}
    for side in ('left', 'right'):
        index, _ = _load_source(v19_base, upstream, side)
        videos[side] = index['video']
        checked[side] = validate_rois(rois[side], index['video']['width'], index['video']['height'])
    protocol = dict(schema='v20_appearance_control_protocol_1', rules=deepcopy(RULES),
        stop_rule=deepcopy(STOP_RULE),
        frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), rois=checked, videos=videos,
        v19_base=str(v19_base), upstream_base=str(upstream),
        source_hashes={str(p): v19.sha(p) for p in _source_paths()},
        dependencies={str(p): v19.sha(p) for p in _dependency_paths(v19_base, upstream)},
        scope='posthoc_development_nuisance_control_not_phase_or_event_validation',
        no_truth_files_read=True)
    _write(base/'protocol.json', protocol)
    return protocol


def verify(base=BASE):
    base = Path(base); p = _read(base/'protocol.json')
    if p.get('schema') != 'v20_appearance_control_protocol_1' or p['rules'] != RULES:
        raise ValueError('appearance-control frozen rules changed')
    if p.get('stop_rule') != STOP_RULE:
        raise ValueError('appearance-control stop rule changed')
    v19.verify(Path(p['v19_base']))
    if set(p['source_hashes']) != {str(x) for x in _source_paths()}:
        raise ValueError('source inventory mismatch')
    if set(p['dependencies']) != {str(x) for x in _dependency_paths(p['v19_base'], p['upstream_base'])}:
        raise ValueError('dependency inventory mismatch')
    for path, digest in {**p['source_hashes'], **p['dependencies']}.items():
        if v19.sha(path) != digest:
            raise ValueError('frozen source or dependency changed '+path)
    for side in ('left', 'right'):
        video = p['videos'][side]
        validate_rois(p['rois'][side], video['width'], video['height'])
    return p


def extract(base=BASE, side='left'):
    """Decode every archived source frame and hash-check it, without inference."""
    if side not in ('left', 'right'):
        raise ValueError('unknown session')
    base = Path(base); p = verify(base); dest = base/side/'extract'
    if dest.exists():
        raise FileExistsError('do not overwrite extraction')
    index, arrays = _load_source(p['v19_base'], p['upstream_base'], side)
    dest.mkdir(parents=True); _write(dest/'started.json', {'status': 'started_not_complete'})
    n = len(index['records'])
    descriptors = np.zeros((n, 2, 2048), np.float32); support = np.zeros_like(descriptors, bool)
    patches = np.full((n, 2, 32, 64), np.nan, np.float32); valid = np.zeros((n, 2), bool)
    std = np.full((n, 2), np.nan); clipped = std.copy(); records = []
    cap = cv2.VideoCapture(index['video']['path'])
    if not cap.isOpened():
        raise ValueError('source video unavailable')
    source_pos = -1
    try:
        for i, record in enumerate(index['records']):
            while source_pos < record['frame_index']:
                ok, raw = cap.read(); source_pos += 1
                if not ok:
                    raise ValueError('source ended before expected frame')
            if v19.upstream.prior.pixels_sha(raw) != record['frame_sha256']:
                raise ValueError('source frame pixel hash mismatch')
            canonical = v19.upstream.prior.canonical_frame(raw, side)
            gray = cv2.cvtColor(canonical, cv2.COLOR_BGR2GRAY)
            reasons = []; colors = []
            for k, roi in enumerate(p['rois'][side]):
                result = patch_descriptor(gray, arrays['transforms'][i], roi, bool(arrays['pose_valid'][i]))
                descriptors[i, k] = result['descriptor']; support[i, k] = result['support']
                patches[i, k] = result['patch']; valid[i, k] = result['valid']
                std[i, k] = result['highpass_std_gray']; clipped[i, k] = result['clipped_fraction']
                reasons.append(result['reason'])
                colors.append(color_diagnostics(canonical, arrays['transforms'][i], roi, bool(arrays['pose_valid'][i])))
            records.append(dict(frame_index=record['frame_index'], time_s=record['time_s'],
                frame_sha256=record['frame_sha256'], pose_valid=bool(arrays['pose_valid'][i]),
                valid=valid[i].tolist(), reasons=reasons, color_diagnostics=colors,
                highpass_std_gray=std[i].tolist(), clipped_fraction=clipped[i].tolist()))
            if i % 100 == 0:
                print('V20 appearance extract', side, i, '/', n, flush=True)
    finally:
        cap.release()
    verify(base)
    with (dest/'descriptors.npz').open('xb') as f:
        np.savez_compressed(f, times_s=arrays['times_s'], frame_indices=arrays['frame_indices'],
            descriptors=descriptors, supports=support, valid=valid, patches=patches,
            highpass_std_gray=std, clipped_fraction=clipped)
    result = dict(schema='v20_flange_descriptors_1', status='complete', session=side,
        protocol_sha256=v19.sha(base/'protocol.json'), descriptor_sha256=v19.sha(dest/'descriptors.npz'),
        records=records, video=index['video'], appearance_only=True)
    _write(dest/'index.json', result)
    return result


def compare_arrays(times, descriptors, supports, valid, offsets, profiles, quality):
    """Full causal matching records and separate geometry comparisons."""
    times = _times(times); n, j = np.asarray(offsets).shape
    if n != len(times):
        raise ValueError('geometry timeline length mismatch')
    deltas = np.full((n, 2, 3, j), np.nan, np.float32)
    rawcorr = np.full_like(deltas, np.nan); centercorr = np.full_like(deltas, np.nan)
    common = np.zeros_like(deltas, bool); records = []
    for i, now in enumerate(times):
        histories = []
        for h, history in enumerate(HISTORIES):
            match = select_match(times, descriptors, supports, valid, i, history)
            partners = {}
            for k, (role, key) in enumerate(zip(PARTNERS, ('best_index', 'runner_index', 'control_index'))):
                past = match[key]
                if past is None:
                    partners[role] = None; continue
                pair = compare_pair(offsets, profiles, quality, i, past)
                deltas[i, h, k] = pair['delta']; rawcorr[i, h, k] = pair['raw_profile_cosine']
                centercorr[i, h, k] = pair['centered_profile_cosine']; common[i, h, k] = pair['common']
                partners[role] = dict(past_index=past, past_time_s=float(times[past]),
                    comparison_selection='appearance_only_not_geometry',
                    partitions=pair['partitions'], descriptive_even_if_match_rejected=not match['accepted'])
            histories.append(dict(match=match, comparisons=partners))
        records.append(dict(index=i, time_s=float(now), histories=histories))
    return dict(records=records, row_arrays=dict(normal_offset_delta_px=deltas,
        raw_profile_cosine=rawcorr, centered_profile_cosine=centercorr, common_quality=common))


def summarize(records):
    """Complete-record and fixed 5s bins, including warmup and missing data."""
    def one(selected):
        out = {}
        for h, name in enumerate(HISTORIES):
            cases = [r['histories'][h] for r in selected]
            accepted = [c for c in cases if c['match']['accepted']]
            partner_summary = {}
            for role in PARTNERS:
                parts = []
                for p in range(4):
                    entries = [c['comparisons'][role]['partitions'][p] for c in accepted
                               if c['comparisons'][role] is not None]
                    values = [x['median_offset_delta_px'] for x in entries if x['valid']]
                    parts.append(dict(partition=p, valid_pair_count=len(values),
                        median_of_pair_medians_px=float(np.median(values)) if values else None,
                        minimum_pair_median_px=float(min(values)) if values else None,
                        maximum_pair_median_px=float(max(values)) if values else None))
                partner_summary[role] = parts
            out[name] = dict(frames=len(cases), accepted_pairs=len(accepted),
                reasons=dict(Counter(c['match']['reason'] for c in cases)),
                geometry_only_among_accepted_appearance_pairs=partner_summary)
        return out
    if any(not np.isfinite(r['time_s']) or not 0. <= r['time_s'] <= RULES['summary_end_s'] for r in records):
        raise ValueError('summary records must lie in the frozen 0..60s experiment')
    buckets = []
    for start in np.arange(0., RULES['summary_end_s'], RULES['summary_bucket_seconds']):
        end = start+RULES['summary_bucket_seconds']; inclusive = end == RULES['summary_end_s']
        selected = [r for r in records if start <= r['time_s'] and
                    (r['time_s'] <= end if inclusive else r['time_s'] < end)]
        buckets.append(dict(start_s=float(start), end_s=float(end), end_inclusive=inclusive, summary=one(selected)))
    return dict(complete=one(records), fixed_five_second_buckets=buckets,
        interpretation='descriptive image evidence, not accuracy or physical phase/layer confirmation')


def compare(base=BASE, side='left'):
    if side not in ('left', 'right'):
        raise ValueError('unknown session')
    base = Path(base); p = verify(base); src = base/side/'extract'; dest = base/side/'compare'
    if dest.exists():
        raise FileExistsError('do not overwrite comparison')
    ei = _read(src/'index.json')
    if (ei.get('status') != 'complete' or ei['protocol_sha256'] != v19.sha(base/'protocol.json')
            or ei['descriptor_sha256'] != v19.sha(src/'descriptors.npz')):
        raise ValueError('descriptor output binding invalid')
    index, arrays = _load_source(p['v19_base'], p['upstream_base'], side)
    if ei['video'] != index['video'] or len(ei['records']) != len(index['records']):
        raise ValueError('descriptor source timeline mismatch')
    for a, b in zip(ei['records'], index['records']):
        if any(a[k] != b[k] for k in ('time_s', 'frame_index', 'frame_sha256')):
            raise ValueError('descriptor frame provenance mismatch')
    with np.load(src/'descriptors.npz', allow_pickle=False) as z:
        desc = {k: z[k].copy() for k in ('times_s', 'frame_indices', 'descriptors', 'supports', 'valid')}
    if not np.array_equal(desc['times_s'], arrays['times_s']) or not np.array_equal(desc['frame_indices'], arrays['frame_indices']):
        raise ValueError('descriptor array timeline mismatch')
    dest.mkdir(parents=True); _write(dest/'started.json', {'status': 'started_not_complete'})
    result = compare_arrays(desc['times_s'], desc['descriptors'], desc['supports'], desc['valid'],
                            arrays['normal_offset'], arrays['normal_profiles'], arrays['quality'])
    for record, source in zip(result['records'], index['records']):
        record.update(frame_index=source['frame_index'], frame_sha256=source['frame_sha256'])
    verify(base)
    with (dest/'comparisons.npz').open('xb') as f:
        np.savez_compressed(f, times_s=arrays['times_s'], rows_y=arrays['rows_y'],
            frame_indices=arrays['frame_indices'], **result.pop('row_arrays'))
    result.update(schema='v20_appearance_control_comparison_1', status='complete', session=side,
        protocol_sha256=v19.sha(base/'protocol.json'), descriptor_index_sha256=v19.sha(src/'index.json'),
        descriptor_sha256=v19.sha(src/'descriptors.npz'), comparison_sha256=v19.sha(dest/'comparisons.npz'),
        video=index['video'], summary=summarize(result['records']),
        mechanical_phase_confirmed=False, physical_new_layer_confirmed=False,
        material_identity_confirmed=False, actuator_command=None)
    _write(dest/'report.json', result)
    return result


def preview(rois, output, v19_base=v19.BASE):
    """Only source frame 0: hash-checked canonical overview and native ROI crops."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('fresh preview directory required')
    p19 = v19.verify(v19_base)
    if set(rois) != {'left', 'right'}:
        raise ValueError('both camera ROI sets required')
    output.mkdir(parents=True)
    summary = {}
    for side in ('left', 'right'):
        index = _read(Path(p19['upstream_base'])/side/'index.json')
        r = index['records'][0]
        if r['frame_index'] != 0:
            raise ValueError('preview is restricted to first source frame')
        video = index['video']; checked = validate_rois(rois[side], video['width'], video['height'])
        cap = cv2.VideoCapture(video['path'])
        try:
            ok, raw = cap.read()
        finally:
            cap.release()
        if not ok or v19.upstream.prior.pixels_sha(raw) != r['frame_sha256']:
            raise ValueError('first source frame hash mismatch')
        canonical = v19.upstream.prior.canonical_frame(raw, side); overlay = canonical.copy()
        def save(path, image):
            ok, encoded = cv2.imencode('.png', image)
            if not ok:
                raise ValueError('preview PNG encode failed')
            with Path(path).open('xb') as f:
                f.write(encoded.tobytes())
        for k, (x1, y1, x2, y2) in enumerate(checked):
            save(output/(side+'_roi_'+str(k+1)+'.png'), canonical[y1:y2, x1:x2])
            cv2.rectangle(overlay, (x1, y1), (x2-1, y2-1), (0, 0, 255), 8)
            cv2.putText(overlay, str(k+1), (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 2., (0, 0, 255), 5)
        save(output/(side+'_overview.png'), cv2.resize(overlay, (1365, 1000), interpolation=cv2.INTER_AREA))
        summary[side] = dict(rois=checked, source_frame_sha256=r['frame_sha256'], frame_index=0,
                            roi_scope='manually_selected_first_frame_development_ROIs_not_general_detector')
    _write(output/'index.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('preview', 'freeze', 'verify', 'extract', 'compare'))
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--v19-base', type=Path, default=v19.BASE)
    parser.add_argument('--rois-json', type=Path, help='required for freeze: left/right each two canonical xyxy rectangles')
    parser.add_argument('--preview-output', type=Path)
    parser.add_argument('--session', choices=('left', 'right'), default='left')
    args = parser.parse_args()
    if args.stage == 'preview':
        if args.rois_json is None or args.preview_output is None:
            parser.error('preview requires --rois-json and --preview-output')
        preview(_read(args.rois_json), args.preview_output, args.v19_base)
    elif args.stage == 'freeze':
        if args.rois_json is None:
            parser.error('freeze requires --rois-json; never guess flange ROIs')
        freeze(_read(args.rois_json), args.base, args.v19_base)
    elif args.stage == 'verify':
        verify(args.base); print('V20 sources and dependencies verified')
    elif args.stage == 'extract':
        extract(args.base, args.session)
    else:
        result = compare(args.base, args.session)
        print('V20 complete', args.session, len(result['records']), 'frames; not phase/events')


if __name__ == '__main__':
    main()
