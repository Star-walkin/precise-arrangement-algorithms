"""H3 free segment correction on a measured straight axis, with support recheck.

Wraps H2 read-only. Seed and frozen B response are unchanged. Public coordinates
match H2: half-size crop xy = (original_xy - [0, 1050]) / 2. The portion after
the last measured A7 free point is saved separately and never forced straight.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
H2_SOURCE = HERE.parent / 'postprocess_b_20261030_v2/free_candidates.py'
_spec = importlib.util.spec_from_file_location('_h2_readonly_free_for_h3', H2_SOURCE)
H2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H2)

PARAMETERS = dict(
    coordinate_scale=.5, crop_origin_original_xy=[0., 1050.],
    minimum_a7_raw_points=6, minimum_a7_raw_span_original_px=80.,
    fallback_minimum_points=8, fallback_minimum_span_original_px=80.,
    fallback_robust_iterations=4, fallback_maximum_angle_change_deg=3.,
    fallback_residual_width_fraction=.25, fallback_residual_floor_half_px=1.,
    fallback_mad_multiplier=2.5,
    continuation_count=3, continuation_back_widths=[0., 1.5, 3.],
    original_ignore_and_gaps_can_never_be_promoted=True,
    true_zero_gray_is_valid=True, do_not_extend_free_endpoint=True,
)


def _unit(x):
    a = np.asarray(x, float)
    return a/max(float(np.linalg.norm(a)), 1.e-12)


def _angle(a, b):
    return float(np.degrees(np.arccos(np.clip(_unit(a) @ _unit(b), -1., 1.))))


def _geometry_stats(points, origin, direction):
    p = np.asarray(points, float).reshape(-1, 2)
    if not len(p):
        return dict(point_count=0, perpendicular_rms_original_px=None,
                    perpendicular_p95_original_px=None, turn_mean_deg=None, turn_p95_deg=None)
    normal = np.array([-direction[1], direction[0]])
    residual = abs((p-origin) @ normal)/PARAMETERS['coordinate_scale']
    delta = np.diff(p, axis=0)
    delta = delta[np.linalg.norm(delta, axis=1)>1.e-8]
    turns = np.zeros(0)
    if len(delta)>1:
        unit = delta/np.linalg.norm(delta, axis=1)[:, None]
        turns = np.degrees(np.arccos(np.clip((unit[:-1]*unit[1:]).sum(1),-1.,1.)))
    return dict(point_count=len(p), perpendicular_rms_original_px=float(np.sqrt(np.mean(residual**2))),
        perpendicular_p95_original_px=float(np.percentile(residual,95)),
        perpendicular_max_original_px=float(residual.max()),
        turn_mean_deg=float(turns.mean()) if len(turns) else 0.,
        turn_p95_deg=float(np.percentile(turns,95)) if len(turns) else 0.,
        turn_max_deg=float(turns.max()) if len(turns) else 0.)


def _robust_line(points, seed_direction, width):
    p = np.asarray(points, float)
    if len(p)<PARAMETERS['fallback_minimum_points']:
        return None
    keep = np.ones(len(p), bool)
    for _ in range(PARAMETERS['fallback_robust_iterations']):
        origin = p[keep].mean(0)
        _, _, vt = np.linalg.svd(p[keep]-origin, full_matrices=False)
        direction = _unit(vt[0])
        if direction @ seed_direction<0: direction=-direction
        normal = np.array([-direction[1],direction[0]])
        residual = (p-origin) @ normal
        mad = 1.4826*np.median(abs(residual[keep]-np.median(residual[keep])))
        threshold = max(PARAMETERS['fallback_residual_floor_half_px'],
            PARAMETERS['fallback_residual_width_fraction']*width,
            PARAMETERS['fallback_mad_multiplier']*mad)
        updated = abs(residual)<=threshold
        if updated.sum()<PARAMETERS['fallback_minimum_points']: break
        keep = updated
    span = float(np.ptp((p[keep]-origin) @ direction))/PARAMETERS['coordinate_scale']
    angle = _angle(direction,seed_direction)
    if span<PARAMETERS['fallback_minimum_span_original_px'] or angle>PARAMETERS['fallback_maximum_angle_change_deg']:
        return None
    return origin,direction,dict(inliers=int(keep.sum()),input_points=len(p),
        span_original_px=span,angle_from_entry_degrees=angle,method='iterative_TLS_MAD')


def _matching_a7(priors, seed, raw, original_support):
    offset=np.asarray(PARAMETERS['crop_origin_original_xy']); scale=PARAMETERS['coordinate_scale']
    matches=[]
    for c in priors.get('a7_candidates',[]):
        if not(c.get('accepted') and c.get('plausible')): continue
        match=H2._match(seed['xy'],c)
        if not match['compatible']: continue
        visible, measured=H2._a7_flags(c,raw/scale+offset)
        ids=np.flatnonzero(measured & original_support)
        if len(ids)<PARAMETERS['minimum_a7_raw_points']: continue
        direction=_unit(c['direction_xy'])
        span=float(np.ptp(raw[ids] @ direction))/scale
        if span<PARAMETERS['minimum_a7_raw_span_original_px']: continue
        matches.append((match['cost'],c,match,ids))
    matches.sort(key=lambda x:x[0])
    return matches[0] if matches else None


def correct_free(priors, seed, heat, original):
    """Correct a saved H2 free result; used by the wrapper and numeric checks.

    ``raw_points_xy`` and ``raw_supported`` retain the COMPLETE H2 arrays.
    ``points_xy_before_correction`` matches the length of corrected points.
    ``unmodified_response_tail`` holds excluded downstream original points.
    No excluded point is silently reassigned to free or promoted to support.
    """
    tick=time.perf_counter(); result=copy.deepcopy(original)
    raw=np.asarray(original.get('points_xy',[]),float).reshape(-1,2)
    old=np.asarray(original.get('supported',[]),bool)
    if len(raw)!=len(old): raise ValueError('H2 support/path alignment mismatch')
    result['raw_points_xy']=raw.tolist(); result['raw_supported']=old.tolist()
    result['raw_point_status']=copy.deepcopy(original.get('point_status',[]))
    result['unmodified_response_tail']=dict(points_xy=[],supported=[],used_as_free=False)
    receipt=dict(parameters=PARAMETERS,h2_source_path=str(H2_SOURCE),
        h2_source_sha256=hashlib.sha256(H2_SOURCE.read_bytes()).hexdigest(),
        seed_changed=False,B_response_changed=False,labels_read=False,
        support_rule='original_supported AND corrected_location_evidence',
        same_material_identity_confirmed=False)
    if len(raw)<2:
        receipt.update(applied=False,reason='no_H2_free_candidate',elapsed_seconds=time.perf_counter()-tick)
        result['straightening']=receipt
        return result
    scale=PARAMETERS['coordinate_scale']; offset=np.asarray(PARAMETERS['crop_origin_original_xy'])
    seed_xy=np.asarray(seed['xy'],float); seed_direction=_unit(seed_xy[-1]-seed_xy[0])
    width=float(original['width_half_px'])
    chosen=_matching_a7(priors,seed,raw,old)
    candidate=None
    if chosen:
        _,candidate,match,ids=chosen
        origin=(np.asarray(candidate['origin_xy'])-offset)*scale
        direction=_unit(candidate['direction_xy'])
        if direction @ seed_direction<0: direction=-direction
        end=int(ids[-1])+1
        source='accepted_A7_measured_axis'
        receipt.update(a7_candidate_id=candidate['candidate_id'],a7_match=match,
            a7_record_path=H2._a7_record_path(priors),a7_saved_fit=candidate.get('fit'),
            a7_original_origin_xy=candidate['origin_xy'],a7_original_direction_xy=candidate['direction_xy'],
            a7_raw_supported_points=int(len(ids)),free_end_basis='last_originally_supported_A7_raw_measurement',
            direction_basis='unchanged_entry_seed_orients_undirected_A7_axis')
    else:
        runs=original.get('visible_runs',[])
        first=next((r for r in runs if old[r['start_index']:r['end_index']+1].sum()>=PARAMETERS['fallback_minimum_points']),None)
        if first is None:
            receipt.update(applied=False,reason='no_reliable_A7_or_initial_supported_run',elapsed_seconds=time.perf_counter()-tick)
            result['straightening']=receipt
            return result
        ids=np.arange(first['start_index'],first['end_index']+1)
        fit=_robust_line(raw[ids[old[ids]]],seed_direction,width)
        if fit is None:
            receipt.update(applied=False,reason='fallback_robust_line_failed_fixed_support_or_angle_gate',elapsed_seconds=time.perf_counter()-tick)
            result['straightening']=receipt
            return result
        origin,direction,fit_diag=fit; end=int(first['end_index'])+1
        source='H2_initial_supported_run_robust_TLS_fallback_not_A7'
        receipt.update(fallback_fit=fit_diag,a7_candidate_id=None,
            free_end_basis='end_of_first_supported_H2_run_identity_unverified',
            direction_basis='unchanged_entry_seed_orients_robust_fitted_axis')
    before=raw[:end].copy()
    along=(before-origin) @ direction
    corrected=origin+along[:,None]*direction
    receipt.update(axis_source=source,axis_origin_original_xy=(origin/scale+offset).tolist(),
        axis_direction_xy=direction.tolist(),axis_angle_from_seed_degrees=_angle(direction,seed_direction),
        before=_geometry_stats(before,origin,direction),after=_geometry_stats(corrected,origin,direction),
        maximum_point_shift_original_px=float(np.linalg.norm(corrected-before,axis=1).max()/scale),
        modified_point_count=int((np.linalg.norm(corrected-before,axis=1)>1.e-6).sum()),
        original_path_points=len(raw),free_path_points=end,
        excluded_tail_point_count=max(0,len(raw)-end),
        last_free_original_xy=(corrected[-1]/scale+offset).tolist(),
        last_free_point_is_not_confirmed_contact=True)
    response=np.asarray(heat,np.float32)
    y0=int(round(offset[1]*scale)); x0=int(round(offset[0]*scale))
    image=priors['original_half'][y0:,x0:]
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
    if response.shape!=gray.shape: raise ValueError('Heat and H1 crop shapes differ')
    finite=response[np.isfinite(response)]
    if len(finite) and (finite.min() < -1.e-5 or finite.max()>1.00001): raise ValueError('B response must be original 0..1 floats')
    valid=np.isfinite(response).astype(np.float32)
    evidence=H2._profiles(cv2.GaussianBlur(gray,(0,0),.7),
        cv2.GaussianBlur(np.nan_to_num(response,nan=0.),(0,0),.7),valid,corrected,direction,width)
    a7_visible,a7_raw=H2._a7_flags(candidate,corrected/scale+offset)
    joint=evidence['heat_profile_supported'] & (evidence['raw_score']>=H2.PARAMETERS['raw_score_minimum'])
    a7_image=(a7_raw & evidence['valid'] & (evidence['center']>=H2.PARAMETERS['a7_heat_center_minimum'])) if candidate else np.zeros(end,bool)
    supported=old[:end] & (joint|a7_image) & (not bool(original.get('entry_conflict')))
    assert not np.any(supported & ~old[:end]), 'Unknown regions cannot be upgraded'
    old_status=original.get('point_status',['unknown']*len(raw))[:end]
    entry=np.asarray([s=='entry_condition' for s in old_status],bool)
    states=[]
    for i in range(end):
        if entry[i]: status='entry_condition'
        elif supported[i] and joint[i]: status='joint_heat_image_supported'
        elif supported[i] and a7_image[i]: status='a7_image_supported'
        else: status='ambiguous_gap'
        states.append(status)
    seg=[]
    for i in range(1,end):
        seg.append('entry_condition' if entry[i-1] and entry[i] else
                   'image_supported' if supported[i-1] and supported[i] else 'bridge')
    gaps=[]
    for lo,hi in H2._runs(~supported & ~entry):
        gaps.append(dict(start_index=int(lo),end_index=int(hi-1),
            kind='preserved_unknown_or_projection_recheck_failure',supported=False,
            physical_occluder_confirmed=False,filled_as_measurement=False))
    visible=[]
    for lo,hi in H2._runs(supported):
        if hi-lo>=2:
            visible.append(dict(start_index=int(lo),end_index=int(hi-1),
                span_original_px=float(np.linalg.norm(corrected[hi-1]-corrected[lo])/scale)))
    starts=[]; ids=np.flatnonzero(supported & ~entry)
    if len(ids):
        last=int(ids[-1]); s=(corrected-origin) @ direction
        for back in PARAMETERS['continuation_back_widths']:
            j=int(ids[np.argmin(abs(s[ids]-(s[last]-back*width)))])
            if any(v['point_index']==j for v in starts): continue
            starts.append(dict(point_index=j,xy=corrected[j].tolist(),tangent=direction.tolist(),
                width_half_px=width,a7_raw_supported=bool(a7_raw[j]),
                heat_profile_supported=bool(evidence['heat_profile_supported'][j]),
                basis='straight_axis_observation_near_free_end_not_contact_truth'))
    ev={k:np.asarray(v).tolist() for k,v in evidence.items()}
    ev.update(a7_visible=a7_visible.tolist(),a7_raw_supported=a7_raw.tolist())
    receipt.update(applied=True,retained_support_points=int(supported.sum()),
        downgraded_support_points=int((old[:end] & ~supported).sum()),
        promoted_unknown_points=0,original_unsupported_points_in_free=int((~old[:end]).sum()),
        elapsed_seconds=time.perf_counter()-tick)
    tail_start=max(0,end-1)
    tail=dict(points_xy=raw[tail_start:].tolist() if end<len(raw) else [],
        supported=old[tail_start:].tolist() if end<len(raw) else [],
        source_start_index=tail_start,used_as_free=False,forced_straight=False,
        reason='beyond_A7_measured_free_extent' if candidate else 'beyond_initial_supported_fallback_run')
    result.update(points_xy=corrected.tolist(),points_xy_before_correction=before.tolist(),
        supported=supported.tolist(),point_status=states,segment_status=seg,
        available=bool(len(ids)>=2),reason='straight_free_with_fixed_input_support_recheck',
        continuation_starts=starts,visible_runs=visible,gap_hypotheses=gaps,
        original_gap_hypotheses=copy.deepcopy(original.get('gap_hypotheses',[])),
        evidence=ev,axis_direction_xy=direction.tolist(),unmodified_response_tail=tail,
        straightening=receipt)
    result['diagnostics']=dict(result.get('diagnostics',{}),
        stop_reason='last_rechecked_measured_free_point_not_confirmed_contact',
        h3_straightening=receipt)
    return result


def describe_free(priors,seed,heat):
    return correct_free(priors,seed,heat,H2.describe_free(priors,seed,heat))


describe_straight_free=describe_free
