"""H2 free-rope pieces: frozen B response and read-only A7 measurements.

Public seed xy is ORIGINAL image coordinates. All returned path/width fields
are HALF-SIZE H1 crop coordinates: (original_xy - [0, 1050]) / 2.
No labels, hardware, model calls, or nearest-F3-track identity decisions.
Gaps stay unsupported and are not evidence of a physical occluder.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np


PARAMETERS = dict(
    crop_origin_original_xy=[0., 1050.], coordinate_scale=.5,
    maximum_new_entries=2, seed_length_original_px=120.,
    axis_match_angle_deg=6., axis_match_offset_widths=2.,
    step_original_px=6., maximum_axis_original_px=4400.,
    normal_offset_widths=.32, normal_offset_samples=7,
    profile_normal_samples=41, tangent_offsets_widths=[-1., -.5, 0., .5, 1.],
    heat_center_minimum=.22, heat_side_drop_minimum=.035,
    heat_side_distance_asymmetry_widths=.45,
    raw_score_minimum=.25, raw_contrast_floor=.014,
    a7_heat_center_minimum=.10, a7_sample_tolerance_original_px=5.,
    minimum_visible_run_original_px=30., minimum_visible_run_widths=2.5,
    maximum_microgap_widths=.65, minimum_run_supported_fraction=.60,
    maximum_entry_to_run_original_px=80.,
    maximum_reappearance_gap_original_px=180.,
    maximum_a7_reappearance_gap_original_px=400., maximum_a7_gap_widths=20.,
    a7_gap_flank_window_original_px=72., a7_gap_minimum_flank_supported_original_px=30.,
    maximum_total_gap_original_px=600., maximum_visible_runs=12,
    continuation_window_original_px=72., maximum_continuation_starts=6,
)


def _runs(mask):
    z = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(z == 1), np.flatnonzero(z == -1)))


def _sample(image, xy):
    p = np.asarray(xy, np.float32)
    shape = p.shape[:-1]
    flat = p.reshape(-1, 2)
    result = np.empty(len(flat), np.float32)
    for start in range(0, len(flat), 24000):
        q = flat[start:start + 24000]
        result[start:start + len(q)] = cv2.remap(
            image, q[:, 0][None], q[:, 1][None], cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0).ravel()
    return result.reshape(shape)


def _unit(d):
    d = np.asarray(d, float)
    return d / max(float(np.linalg.norm(d)), 1.e-10)


def _match(seed_xy, candidate):
    p = np.asarray(seed_xy, float)
    d = _unit(p[-1] - p[0]); axis = _unit(candidate['direction_xy'])
    angle = math.degrees(math.acos(float(np.clip(abs(d @ axis), 0., 1.))))
    normal = np.array([-axis[1], axis[0]])
    offset = float(np.median(abs((p - np.asarray(candidate['origin_xy'])) @ normal)))
    width = float(candidate['width_px'])
    return dict(compatible=bool(angle <= PARAMETERS['axis_match_angle_deg'] and
        offset <= max(6., PARAMETERS['axis_match_offset_widths'] * width)),
        angle_degrees=angle, offset_original_px=offset,
        cost=offset / max(width, 1.) + angle / 3.)


def _a7_record_path(priors):
    paths = [p for p in priors.get('input_dependency_sha256', {})
             if 'camera_condition_free_straight_20261030_v7' in p and p.endswith('.json')]
    return paths[0] if paths else str(Path('A7_saved_record_not_supplied'))


def _eligible(priors):
    return [c for c in priors.get('a7_candidates', []) if
            c.get('accepted') and c.get('plausible') and
            float(c.get('anchor_span_px', 0)) >= 100.]


def _directions_from_axis_domains(candidate):
    """Infer only projection direction from observed outside/surface sequence.

    This does NOT infer a winding direction or contact point. If both ends
    could be the entry, two hypotheses survive. Never choose the nearer edge.
    """
    p = np.asarray(candidate['center_xy'], float)
    visible = np.asarray(candidate['visible'], bool)
    raw = np.asarray(candidate.get('raw_supported', visible), bool)
    outside = np.asarray(candidate.get('outside', np.zeros(len(p), bool)), bool)
    anchor = np.asarray(candidate['anchor_segment_xy'], float)
    axis = _unit(candidate['direction_xy'])
    s = (p - np.asarray(candidate['origin_xy'])) @ axis
    sa = (anchor - np.asarray(candidate['origin_xy'])) @ axis
    observed = visible & raw
    inside = observed & ~outside
    span = max(float(candidate['width_px']) * 3., 80.)
    forward = int(np.sum(inside & (s > sa.max()) & (s < sa.max() + 900.)))
    backward = int(np.sum(inside & (s < sa.min()) & (s > sa.min() - 900.)))
    required = max(8, int(span / 4.))
    if forward >= required and backward < required / 2:
        return [(1, 'A7_observed_outside_to_surface_projection', False)]
    if backward >= required and forward < required / 2:
        return [(-1, 'A7_observed_outside_to_surface_projection', False)]
    return [(1, 'A7_undirected_axis_two_hypotheses', True),
            (-1, 'A7_undirected_axis_two_hypotheses', True)]


def suggest_a7_entries(priors, existing_seeds):
    """At most two new original-coordinate seeds; existing entries unchanged."""
    seeds = []
    for c in sorted(_eligible(priors), key=lambda x: -float(x.get('score', 0))):
        if any(_match(s['xy'], c)['compatible'] for s in existing_seeds):
            continue
        anchor = np.asarray(c['anchor_segment_xy'], float)
        axis = _unit(c['direction_xy'])
        order = np.argsort(anchor @ axis); anchor = anchor[order]
        length = min(PARAMETERS['seed_length_original_px'], float(np.linalg.norm(anchor[1]-anchor[0])))
        for sign, basis, ambiguous in _directions_from_axis_domains(c):
            start = anchor[0] if sign == 1 else anchor[1]
            end = start + sign * axis * length
            xy = np.array([start, end])
            # The frozen network did not see pixels above its original crop.
            if np.any(xy[:, 1] < PARAMETERS['crop_origin_original_xy'][1]):
                continue
            h, w = priors['original_shape']
            if np.any(xy < 0) or np.any(xy[:, 0] >= w) or np.any(xy[:, 1] >= h):
                continue
            seeds.append(dict(xy=xy.tolist(), source=_a7_record_path(priors),
                entry_id=f"h2_a7_{c['candidate_id']}_{'plus' if sign > 0 else 'minus'}",
                a7_candidate_id=c['candidate_id'], seed_kind='new_A7_automatic_entry',
                direction_basis=basis, direction_ambiguous=ambiguous,
                original_seed_length_px=length,
                appearance_status=c.get('signature', {}).get('status', 'unknown'),
                acceptance_evidence=c.get('acceptance_evidence', {}),
                same_material_identity_confirmed=False))
            if len(seeds) >= PARAMETERS['maximum_new_entries']:
                return seeds
    return seeds


def _profiles(gray, heat, valid, points, direction, width):
    n = np.array([-direction[1], direction[0]])
    u = np.linspace(-1.2, 1.2, PARAMETERS['profile_normal_samples'])
    v = np.asarray(PARAMETERS['tangent_offsets_widths'])
    q = points[:, None, None, :] + width * (
        v[None, :, None, None] * direction + u[None, None, :, None] * n)
    raw = _sample(gray, q).mean(1)
    hp = _sample(heat, q).mean(1)
    good = _sample(valid, q).mean((1, 2)) >= .98
    center = hp[:, abs(u) <= .18].mean(1)
    li = np.flatnonzero((u <= -.45) & (u >= -1.05))
    ri = np.flatnonzero((u >= .45) & (u <= 1.05))
    lidx = li[np.argmin(hp[:, li], axis=1)]
    ridx = ri[np.argmin(hp[:, ri], axis=1)]
    row = np.arange(len(points)); left = hp[row, lidx]; right = hp[row, ridx]
    drop = center - np.maximum(left, right)
    asym = abs(abs(u[lidx]) - abs(u[ridx]))
    heat_ok = good & (center >= PARAMETERS['heat_center_minimum']) & (
        drop >= PARAMETERS['heat_side_drop_minimum']) & (
        asym <= PARAMETERS['heat_side_distance_asymmetry_widths'])
    # Polarity independent ORIGINAL grayscale ribbon; actual zeros are data.
    lb = raw[:, u <= -.9].mean(1); rb = raw[:, u >= .9].mean(1)
    bg = lb[:, None] + (rb-lb)[:, None] * (u[None] + 1.05) / 2.1
    residual = raw - bg
    body = residual[:, abs(u) <= .24].mean(1)
    sign = np.where(body >= 0, 1., -1.)
    scale = np.maximum(residual.std(1), PARAMETERS['raw_contrast_floor'])
    gradient = np.diff(raw, axis=1); mid = (u[:-1] + u[1:]) / 2
    le = gradient[:, (mid > -.8) & (mid < -.22)].sum(1) * sign
    re = -gradient[:, (mid > .22) & (mid < .8)].sum(1) * sign
    pair = np.clip(np.minimum(le, re)/(1.8*scale), 0, 1)
    body_score = np.clip(abs(body)/(2*scale), 0, 1)
    raw_score = (.65*pair + .35*body_score) * np.clip(
        abs(body)/(3*PARAMETERS['raw_contrast_floor']), 0, 1)
    heat_score = np.clip(center, 0, 1) * np.clip(drop/.14, 0, 1) * np.clip(1-asym/1.2, 0, 1)
    return dict(center=center, left_low=left, right_low=right, side_drop=drop,
        side_distance_asymmetry_widths=asym, heat_profile_supported=heat_ok,
        raw_score=raw_score, raw_contrast=abs(body), valid=good,
        score=.65*heat_score+.35*raw_score)


def _a7_flags(candidate, points_original):
    if candidate is None:
        return np.zeros(len(points_original), bool), np.zeros(len(points_original), bool)
    p = np.asarray(candidate['center_xy'], float)
    d = _unit(candidate['direction_xy']); origin = np.asarray(candidate['origin_xy'])
    s = (p-origin) @ d; q = (points_original-origin) @ d
    ix = np.searchsorted(s, q).clip(1, len(s)-1)
    ix -= abs(s[ix-1]-q) < abs(s[ix]-q)
    distance = np.linalg.norm(points_original - p[ix], axis=1)
    close = distance <= max(PARAMETERS['a7_sample_tolerance_original_px'], .35*float(candidate['width_px']))
    visible = np.asarray(candidate['visible'], bool)[ix] & close
    raw = np.asarray(candidate.get('raw_supported', candidate['visible']), bool)[ix] & visible
    return visible, raw


def describe_free(priors, seed, heat):
    """Describe a straight free path and measured pieces, without identity claims.

    ``width_half_px`` is a SCALAR. ``points_xy``, ``supported`` and all evidence
    arrays have length N. ``segment_status`` has N-1 authoritative labels.
    ``continuation_starts`` items include point_index, xy, tangent, width_half_px.
    Raw saved A7 pieces are not extrapolated into a filled straight rope.
    """
    start_time = time.perf_counter()
    cfg = PARAMETERS; scale = cfg['coordinate_scale']; offset = np.asarray(cfg['crop_origin_original_xy'])
    raw_heat = np.asarray(heat, np.float32)
    y0 = int(round(offset[1]*scale)); x0 = int(round(offset[0]*scale))
    image = priors['original_half'][y0:, x0:]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
    if gray.shape != raw_heat.shape:
        raise ValueError(f'H1 crop heat must be {gray.shape}, got {raw_heat.shape}')
    valid = np.isfinite(raw_heat).astype(np.float32)
    finite = raw_heat[np.isfinite(raw_heat)]
    if len(finite) and (finite.min() < -1.e-5 or finite.max() > 1.00001):
        raise ValueError('B must be the original numeric response in [0, 1]')
    heat_f = cv2.GaussianBlur(np.nan_to_num(raw_heat, nan=0.), (0, 0), .7)
    gray_f = cv2.GaussianBlur(gray, (0, 0), .7)
    seed_original = np.asarray(seed['xy'], float)
    if seed_original.ndim != 2 or seed_original.shape[1] != 2 or len(seed_original) < 2:
        raise ValueError('Seed requires at least two ordered original-coordinate points')
    local_seed = (seed_original-offset)*scale
    direction = _unit(local_seed[-1]-local_seed[0]); origin = local_seed[0]
    seed_length = float(np.linalg.norm(local_seed[-1]-local_seed[0]))
    matching = []
    for c in priors.get('a7_candidates', []):
        if not c.get('plausible'):
            continue
        m = _match(seed_original, c)
        if m['compatible']:
            matching.append((not c.get('accepted', False), m['cost'], c, m))
    matching.sort(key=lambda x: (x[0], x[1]))
    candidate = matching[0][2] if matching else None
    accepted_match = candidate is not None and bool(candidate.get('accepted'))
    entry_conflict = bool(_eligible(priors) and not accepted_match)
    width_original = float(candidate['width_px']) if candidate else None
    width_source = 'A7_measured_axis' if candidate else None
    if width_original is None:
        for entry in priors.get('a6_entries', []):
            g = entry.get('free', {}).get('geometry', {})
            if all(k in g for k in ('origin_xy', 'direction_xy', 'width_px')) and _match(seed_original, g)['compatible']:
                width_original = float(g['width_px']); width_source = 'A6_compatible_measurement'; break
    if width_original is None:
        width_original = float(np.clip(priors['pitch_px']*.55, 6., 70.))
        width_source = 'pitch_fallback_not_physical_calibration'
    width = width_original*scale
    if candidate is not None:
        axis = _unit(candidate['direction_xy'])
        if axis @ direction < 0: axis = -axis
        co = (np.asarray(candidate['origin_xy'])-offset)*scale
        # Do not silently move the supplied seed: its points are kept below.
        origin = co + ((local_seed[0]-co) @ axis)*axis
        direction = axis
    step = cfg['step_original_px']*scale
    t = np.arange(0., cfg['maximum_axis_original_px']*scale, step)
    points = origin + t[:, None]*direction
    h, w = gray.shape
    in_domain = (points[:, 0] >= 1)&(points[:, 0] < w-1)&(points[:, 1] >= 1)&(points[:, 1] < h-1)
    # Preserve prefix alignment; a ray leaving the frame cannot return.
    first_bad = np.flatnonzero(~in_domain)
    if len(first_bad): points = points[:first_bad[0]]; t = t[:first_bad[0]]
    base = dict(points_xy=[], width_half_px=width, supported=[], continuation_starts=[],
        entry_conflict=entry_conflict, point_status=[], segment_status=[], visible_runs=[],
        gap_hypotheses=[], disconnected_pieces=[], same_material_identity_confirmed=False,
        coordinate_system='half_size_crop_xy', original_xy_mapping='xy / 0.5 + [0, 1050]',
        actual_seed_xy=seed_original.tolist(), width_source=width_source,
        a7_candidate_id=candidate.get('candidate_id') if candidate else None)
    diagnostics = dict(parameters=cfg, a7_matching_candidates=len(matching),
        matched_accepted_a7=accepted_match, input_nan_pixels=int((valid == 0).sum()),
        real_zero_gray_pixels=int((gray == 0).sum()), labels_read=False,
        physical_contact_confirmed=False, heat_response_is_probability=False,
        entry_conflict_reason='different_accepted_A7_axis' if entry_conflict else None)
    if len(points) < 4:
        return base | dict(available=False, reason='entry_axis_outside_crop', diagnostics=diagnostics)
    n = np.array([-direction[1], direction[0]])
    shifts = np.linspace(-cfg['normal_offset_widths'], cfg['normal_offset_widths'], cfg['normal_offset_samples'])*width
    proposals = (points[:, None, :] + shifts[None, :, None]*n).reshape(-1, 2)
    ev = _profiles(gray_f, heat_f, valid, proposals, direction, width)
    scores = ev['score'].reshape(len(points), -1) - .08*(shifts[None]/max(width, 1))**2
    scores[~ev['valid'].reshape(len(points), -1)] = -1
    choice = scores.argmax(1); ix = np.arange(len(points))*len(shifts)+choice
    ev = {k: np.asarray(v)[ix] for k, v in ev.items()}
    measured = proposals[ix]
    a7_visible, a7_raw = _a7_flags(candidate, measured/scale+offset)
    joint = ev['heat_profile_supported'] & (ev['raw_score'] >= cfg['raw_score_minimum'])
    a7_image = (a7_raw & accepted_match & ev['valid'] &
        (ev['center'] >= cfg['a7_heat_center_minimum']))
    supported = (joint | a7_image) & (not entry_conflict)
    grouping = supported.copy()
    for a, b in _runs(~grouping):
        if a > 0 and b < len(grouping) and (b-a)*step <= width*cfg['maximum_microgap_widths']:
            grouping[a:b] = True
    all_runs = []
    for a, b in _runs(grouping):
        if (b-a)*step < max(cfg['minimum_visible_run_original_px']*scale, width*cfg['minimum_visible_run_widths']): continue
        if supported[a:b].mean() < cfg['minimum_run_supported_fraction']: continue
        all_runs.append(dict(start_index=int(a), end_index=int(b-1),
            span_original_px=float((b-a-1)*step/scale),
            joint_supported_fraction=float(joint[a:b].mean()),
            a7_image_supported_fraction=float(a7_image[a:b].mean())))
    selected = []; gaps = []; total_gap = 0.
    for run in all_runs:
        if not selected:
            if t[run['start_index']] > seed_length + cfg['maximum_entry_to_run_original_px']*scale:
                break
        else:
            lo = selected[-1]['end_index']+1; hi = run['start_index']
            gap = max(0., (hi-lo)*step/scale)
            flank = max(1, int(cfg['a7_gap_flank_window_original_px']*scale/step))
            left_count = int(a7_raw[max(selected[-1]['start_index'],lo-flank):lo].sum())
            right_count = int(a7_raw[hi:min(run['end_index']+1,hi+flank)].sum())
            minimum_count = cfg['a7_gap_minimum_flank_supported_original_px']*scale/step
            a7_reappearance = bool(accepted_match and left_count >= minimum_count and right_count >= minimum_count)
            gap_limit = cfg['maximum_reappearance_gap_original_px']
            if a7_reappearance:
                gap_limit = max(gap_limit, min(cfg['maximum_a7_reappearance_gap_original_px'], cfg['maximum_a7_gap_widths']*width_original))
            if gap > gap_limit or total_gap+gap > cfg['maximum_total_gap_original_px']:
                break
            if gap:
                gaps.append(dict(start_index=lo, end_index=hi-1, length_original_px=gap,
                    kind='A7_same_axis_measured_flanks_unobserved_gap' if a7_reappearance else 'same_axis_unobserved_gap',
                    a7_flank_supported_samples=[left_count,right_count], allowed_gap_original_px=gap_limit,
                    physical_occluder_confirmed=False,
                    supported=False, filled_as_measurement=False))
                total_gap += gap
        selected.append(run)
        if len(selected) >= cfg['maximum_visible_runs']: break
    # Preserve later visible pieces independently. They are NOT attached to
    # the entry by a silent straight fill; a downstream connector must account
    # for the intervening unsupported length and competing continuations.
    disconnected = []
    selected_ends = {r['end_index'] for r in selected}
    for run in all_runs:
        if run['end_index'] in selected_ends: continue
        a, b = run['start_index'], run['end_index']+1
        disconnected.append(dict(points_xy=measured[a:b].tolist(), supported=supported[a:b].tolist(),
            a7_raw_supported=a7_raw[a:b].tolist(), width_half_px=width,
            source_scan_start_index=a, source_scan_end_index=b-1,
            reachable_from_entry=False, gap_filled=False,
            entry_identity_confirmed=False, tangent=direction.tolist(),
            preceding_unobserved_gap_original_px=float(max(0, a-(selected[-1]['end_index']+1))*step/scale) if selected else None,
            evidence={k:np.asarray(v[a:b]).tolist() for k,v in ev.items()}))
        if len(disconnected) >= cfg['maximum_visible_runs']: break
    if not selected:
        diagnostics.update(all_visible_runs=all_runs, elapsed_seconds=time.perf_counter()-start_time)
        return base | dict(available=False, reason='entry_conflict_with_A7' if entry_conflict else 'no_supported_free_run',
            disconnected_pieces=disconnected, diagnostics=diagnostics)
    end = selected[-1]['end_index']+1
    output_points = measured[:end].copy()
    entry = t[:end] <= seed_length
    # Exact original supplied seed is a condition, not newly observed rope.
    output_points[entry] = local_seed[0] + np.clip(t[:end][entry]/max(seed_length, 1.e-6), 0, 1)[:, None]*(local_seed[-1]-local_seed[0])
    keep = np.zeros(end, bool)
    for run in selected: keep[run['start_index']:run['end_index']+1] = supported[run['start_index']:run['end_index']+1]
    statuses = []
    for j in range(end):
        if entry[j]: status = 'entry_condition'
        elif keep[j] and joint[j]: status = 'joint_heat_image_supported'
        elif keep[j] and a7_image[j]: status = 'a7_image_supported'
        else: status = 'ambiguous_gap'
        statuses.append(status)
    segment_status = []
    for j in range(1, end):
        if entry[j-1] and entry[j]: status = 'entry_condition'
        elif keep[j-1] and keep[j]: status = 'image_supported'
        else: status = 'bridge'
        segment_status.append(status)
    starts = []
    for run in selected:
        lo, hi = run['start_index'], run['end_index']
        lo = max(lo, hi-int(cfg['continuation_window_original_px']*scale/step))
        for j in sorted(set([lo, (lo+hi)//2, hi])):
            if j < end and keep[j] and not entry[j]:
                starts.append(dict(point_index=int(j), xy=output_points[j].tolist(),
                    tangent=direction.tolist(), width_half_px=width,
                    a7_raw_supported=bool(a7_raw[j]), heat_profile_supported=bool(ev['heat_profile_supported'][j]),
                    basis='last_visible_piece_near_end_not_contact_truth'))
    starts = starts[-cfg['maximum_continuation_starts']:]
    diagnostics.update(all_visible_runs=all_runs, selected_visible_runs=len(selected),
        disconnected_visible_runs=len(all_runs)-len(selected),
        stop_reason='last_image_supported_axis_piece_not_physical_contact',
        total_gap_original_px=total_gap, elapsed_seconds=time.perf_counter()-start_time)
    evidence = {k: np.asarray(v[:end]).tolist() for k, v in ev.items()}
    evidence.update(a7_visible=a7_visible[:end].tolist(), a7_raw_supported=a7_raw[:end].tolist())
    return base | dict(available=True, reason='bounded_image_supported_free_pieces',
        points_xy=output_points.tolist(), supported=keep.tolist(), point_status=statuses,
        segment_status=segment_status, visible_runs=selected, gap_hypotheses=gaps,
        disconnected_pieces=disconnected,
        continuation_starts=starts, evidence=evidence,
        axis_direction_xy=direction.tolist(), diagnostics=diagnostics)
