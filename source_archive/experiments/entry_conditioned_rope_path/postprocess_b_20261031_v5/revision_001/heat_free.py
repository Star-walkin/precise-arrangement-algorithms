"""H5 length-first, terminal-domain-bounded entry-linked free hypotheses.

The unchanged entry supplies direction/association, not the full target path.
A7 supplies an initial axis/optical width hint, never a locked line or endpoint.
Original B floats are scanned on a fixed finite angle/offset grid, then a small
fixed refinement grid. Long same-axis pieces and unobserved gaps stay distinct.
All path xy and widths are H1 half-size crop coordinates; raw image zeros are
valid. No labels, model calls, F3 termination or hardware are used here.
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from common import H2, module, sha

BASE = module('h5_readonly_free_measurements', H2/'free_candidates.py')

PARAMETERS = dict(
    coordinate_scale=.5,crop_origin_original_xy=[0.,1050.],
    angle_offsets_degrees=[-3.,-2.625,-2.25,-1.875,-1.5,-1.125,-.75,-.375,0.,.375,.75,1.125,1.5,1.875,2.25,2.625,3.],
    normal_offsets_widths=[-1.,-.75,-.5,-.25,0.,.25,.5,.75,1.],
    refine_angles_degrees=[-.125,0.,.125],refine_offsets_widths=[-.10,0.,.10],
    refine_top_axes=3,coarse_step_original_px=12.,final_step_original_px=6.,
    maximum_axis_original_px=4400.,maximum_total_axis_candidates=180,
    minimum_heat_center=.28,minimum_side_drop=.045,
    maximum_side_distance_asymmetry_widths=.45,
    minimum_raw_score=.25,minimum_image_contrast=.008,
    minimum_visible_run_original_px=36.,minimum_visible_run_widths=2.5,
    maximum_microgap_widths=.65,minimum_run_heat_fraction=.70,
    maximum_entry_to_run_original_px=80.,maximum_gap_original_px=400.,
    maximum_gap_widths=20.,maximum_total_gap_original_px=600.,
    reappearance_minimum_run_original_px=72.,reappearance_center_minimum=.35,
    reappearance_side_drop_minimum=.055,
    length_reference_original_px=4400.,maximum_visible_runs=12,
    length_rank_bin_original_px=48.,joint_gap_flank_original_px=48.,
    joint_gap_flank_window_original_px=96.,
    stable_surface_span_original_px=120.,far_surface_exit_span_original_px=24.,
    far_surface_margin_widths=1.5,far_surface_margin_minimum_original_px=16.,
    retained_axis_hypotheses=3,
    heat_score_weight=.44,image_score_weight=.16,coverage_score_weight=.30,
    entry_score_weight=.10,offset_penalty=.025,angle_penalty=.015,
    unsupported_gap_penalty=.12,continuation_count=3,
    seed_changed=False,old_endpoint_used_as_limit=False,labels_read=False,
)


def _unit(p):
    p=np.asarray(p,float);return p/max(float(np.linalg.norm(p)),1.e-12)


def _rot(d,degrees):
    a=math.radians(degrees);c,s=math.cos(a),math.sin(a)
    return np.array([c*d[0]-s*d[1],s*d[0]+c*d[1]])


def _initial_geometry(priors,seed):
    seed_xy=np.asarray(seed['xy'],float)
    direction=_unit(seed_xy[-1]-seed_xy[0])
    matches=[]
    for c in priors.get('a7_candidates',[]):
        if not c.get('plausible'):continue
        m=BASE._match(seed_xy,c)
        if m['compatible']:matches.append((not bool(c.get('accepted')),m['cost'],c,m))
    matches.sort(key=lambda x:(x[0],x[1]))
    c=matches[0][2] if matches else None
    # Only an accepted A7 orientation is a preferred initialization. An
    # unaccepted plausible candidate can supply width but cannot lock angle.
    accepted=bool(c and c.get('accepted'))
    initial_direction=_unit(c['direction_xy']) if accepted else direction.copy()
    if initial_direction @ direction<0:initial_direction=-initial_direction
    width=float(c['width_px']) if c else None
    source='A7_measured_optical_width_hint' if c else None
    if width is None:
        for entry in priors.get('a6_entries',[]):
            g=entry.get('free',{}).get('geometry',{})
            if all(k in g for k in ('origin_xy','direction_xy','width_px')) and BASE._match(seed_xy,g)['compatible']:
                width=float(g['width_px']);source='A6_compatible_optical_width';break
    if width is None:
        width=float(np.clip(priors['pitch_px']*.55,6.,70.));source='pixel_pitch_fallback_not_physical_width'
    conflict=bool(BASE._eligible(priors) and not accepted)
    return dict(direction=initial_direction,seed_direction=direction,width_original_px=width,
        width_source=source,a7_candidate=c,accepted_a7_match=accepted,entry_conflict=conflict,
        direction_source='accepted_A7_initial_axis' if accepted else 'unchanged_entry_direction',
        a7_match=matches[0][3] if matches else None)


def _measure(gray,heat,valid,points,direction,width):
    ev=BASE._profiles(gray,heat,valid,points,direction,width)
    protect=(ev['valid'] & (ev['center']>=PARAMETERS['minimum_heat_center']) &
        (ev['side_drop']>=PARAMETERS['minimum_side_drop']) &
        (ev['side_distance_asymmetry_widths']<=PARAMETERS['maximum_side_distance_asymmetry_widths']))
    supported=protect & (ev['raw_score']>=PARAMETERS['minimum_raw_score']) & (ev['raw_contrast']>=PARAMETERS['minimum_image_contrast'])
    ev['heat_profile_supported']=protect
    ev['image_heat_supported']=supported
    return ev


def _segments(ev,step,width,seed_length):
    protect=np.asarray(ev['heat_profile_supported'],bool)
    grouping=protect.copy()
    for a,b in BASE._runs(~grouping):
        if a>0 and b<len(grouping) and (b-a)*step<=width*PARAMETERS['maximum_microgap_widths']:
            grouping[a:b]=True
    visible=[]
    for a,b in BASE._runs(grouping):
        span=(b-a-1)*step/.5
        if span<max(PARAMETERS['minimum_visible_run_original_px'],PARAMETERS['minimum_visible_run_widths']*width/.5):continue
        if protect[a:b].mean()<PARAMETERS['minimum_run_heat_fraction']:continue
        visible.append(dict(start_index=int(a),end_index=int(b-1),span_original_px=float(span),
            heat_supported_fraction=float(protect[a:b].mean()),
            joint_supported_fraction=float(ev['image_heat_supported'][a:b].mean()),
            median_center=float(np.median(ev['center'][a:b])),median_side_drop=float(np.median(ev['side_drop'][a:b]))))
    selected=[];gaps=[];total_gap=0.
    for run in visible:
        if not selected:
            if run['start_index']*step>seed_length+PARAMETERS['maximum_entry_to_run_original_px']*.5:break
        else:
            lo=selected[-1]['end_index']+1;hi=run['start_index']
            gap=(hi-lo)*step/.5
            maximum=min(PARAMETERS['maximum_gap_original_px'],PARAMETERS['maximum_gap_widths']*width/.5)
            flank=max(2,int(PARAMETERS['joint_gap_flank_window_original_px']*.5/step))
            left=ev['image_heat_supported'][max(selected[-1]['start_index'],lo-flank):lo]
            right=ev['image_heat_supported'][hi:min(run['end_index']+1,hi+flank)]
            span=lambda q:max([(b-a-1)*step/.5 for a,b in BASE._runs(q)] or [0.])
            joint_bridge=(span(left)>=PARAMETERS['joint_gap_flank_original_px'] and
                          span(right)>=PARAMETERS['joint_gap_flank_original_px'])
            if joint_bridge:maximum=PARAMETERS['maximum_gap_original_px']
            if gap>PARAMETERS['maximum_gap_original_px'] or total_gap+gap>PARAMETERS['maximum_total_gap_original_px']:break
            if gap>maximum:continue
            if not joint_bridge and gap>width/.5*1.5 and (run['span_original_px']<PARAMETERS['reappearance_minimum_run_original_px'] or
                run['median_center']<PARAMETERS['reappearance_center_minimum'] or run['median_side_drop']<PARAMETERS['reappearance_side_drop_minimum']):continue
            if gap>0:
                gaps.append(dict(start_index=lo,end_index=hi-1,length_original_px=float(gap),
                    kind='same_axis_joint_flanks_unknown_gap' if joint_bridge else 'bounded_same_heat_axis_reappearance_hypothesis',supported=False,
                    allowed_gap_original_px=maximum,joint_flank_supported_spans_original_px=[span(left),span(right)],
                    physical_occluder_confirmed=False,filled_as_measurement=False))
                total_gap+=gap
        selected.append(run)
        if len(selected)>=PARAMETERS['maximum_visible_runs']:break
    return selected,gaps,visible,total_gap


def _surface_domain(priors):
    """Pure saved SAM2 outer contour; fill interior holes, no convex extrapolation."""
    if 'sam2_surface_half' not in priors:return None
    mask=(np.asarray(priors['sam2_surface_half'])[525:]>0).astype(np.uint8)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:return None
    largest=max(cv2.contourArea(c) for c in contours)
    substantial=[c for c in contours if cv2.contourArea(c)>=max(100.,largest*.05)]
    filled=np.zeros_like(mask);cv2.drawContours(filled,substantial,-1,1,cv2.FILLED)
    return dict(inside=filled.astype(np.float32),outside_distance=cv2.distanceTransform(1-filled,cv2.DIST_L2,3),
        basis='pure_SAM2_external_contours_internal_holes_filled_not_convex_hull',
        substantial_contours=len(substantial))


def _terminal_domain(domain,points,step,width):
    receipt=dict(applied=False,index=len(points),basis=None,not_physical_contact_or_identity_truth=True)
    if domain is None:return receipt
    inside=BASE._sample(domain['inside'],points)>.5
    stable=[(a,b) for a,b in BASE._runs(inside) if (b-a-1)*step/.5>=PARAMETERS['stable_surface_span_original_px']]
    receipt.update(basis=domain['basis'],stable_surface_entry=bool(stable))
    if not stable:return receipt
    start=stable[0][1]-1
    distance=BASE._sample(domain['outside_distance'],points)/.5
    margin=max(PARAMETERS['far_surface_margin_minimum_original_px'],PARAMETERS['far_surface_margin_widths']*width/.5)
    outside=(np.arange(len(points))>start)&(distance>margin)
    for a,b in BASE._runs(outside):
        if (b-a-1)*step/.5<PARAMETERS['far_surface_exit_span_original_px']:continue
        receipt.update(applied=True,index=int(a),margin_original_px=margin,
            first_stable_surface_index=int(stable[0][0]),
            exit_progress_original_px=float(a*step/.5),exit_point_crop_xy=points[a].tolist(),
            reason='sustained_far_side_SAM2_external_domain_exit_after_surface_passage')
        break
    return receipt


def _evaluate_axis(gray,heat,valid,origin,direction,width,seed_length,angle,offset,step,domain=None):
    t=np.arange(0.,PARAMETERS['maximum_axis_original_px']*.5,step)
    points=origin+t[:,None]*direction
    h,w=heat.shape
    inside=(points[:,0]>=1)&(points[:,0]<w-1)&(points[:,1]>=1)&(points[:,1]<h-1)
    bad=np.flatnonzero(~inside)
    if len(bad):points=points[:bad[0]];t=t[:bad[0]]
    if len(points)<5:return None
    ev=_measure(gray,heat,valid,points,direction,width)
    terminal=_terminal_domain(domain,points,step,width)
    beyond=dict(points_xy=[],heat_supported_points=0,joint_supported_points=0)
    if terminal['applied']:
        end=terminal['index'];beyond=dict(points_xy=points[end:].tolist(),
            heat_supported_points=int(ev['heat_profile_supported'][end:].sum()),
            joint_supported_points=int(ev['image_heat_supported'][end:].sum()),
            not_used_for_length_or_path=True)
        points=points[:end];t=t[:end];ev={k:v[:end] for k,v in ev.items()}
    if len(points)<5:return None
    runs,gaps,all_runs,gap_length=_segments(ev,step,width,seed_length)
    if not runs:return dict(score=-1.,angle_degrees=float(angle),offset_widths=float(offset),points=points,ev=ev,runs=[],gaps=[],all_runs=all_runs)
    end=runs[-1]['end_index']+1
    selected=np.zeros(end,bool)
    for run in runs:selected[run['start_index']:run['end_index']+1]=ev['heat_profile_supported'][run['start_index']:run['end_index']+1]
    span=float(t[end-1]/.5);supported_length=float(selected.sum()*step/.5)
    hs=(ev['center'][:end]*np.clip(ev['side_drop'][:end]/.15,0,1)*np.clip(1-ev['side_distance_asymmetry_widths'][:end]/1.2,0,1))
    heat_mean=float(np.mean(hs[selected])) if selected.any() else 0.
    raw_mean=float(np.mean(ev['raw_score'][:end][selected])) if selected.any() else 0.
    # Full finite original-image/SAM2 terminal domain. Entry condition and
    # unknown gaps do not add observed straight length; no 2000px cutoff.
    reference=PARAMETERS['length_reference_original_px']
    progress=t[:end]/.5
    interval=np.maximum(0.,np.minimum(progress[1:],reference)-np.minimum(progress[:-1],reference))
    supported_interval=selected[:-1] & selected[1:]
    beyond_entry=(t[:end][:-1]>seed_length)&(t[:end][1:]>seed_length)
    supported_interval &= beyond_entry
    visible_length=float(np.sum(interval*supported_interval))
    joint=ev['image_heat_supported'][:end] & selected
    joint_interval=joint[:-1] & joint[1:] & beyond_entry
    joint_length=float(np.sum(interval*joint_interval))
    weights=interval*supported_interval/reference
    heat_score=float(np.sum(.5*(hs[:-1]+hs[1:])*weights))
    rv=ev['raw_score'][:end]
    raw_score=float(np.sum(.5*(rv[:-1]+rv[1:])*weights))
    near=t[:end]<=seed_length+width*2
    entry_score=float(np.mean(hs[near])) if near.any() else 0.
    coverage=float(weights.sum())
    gap_fraction=gap_length/max(span,1.)
    score=(PARAMETERS['heat_score_weight']*heat_score+PARAMETERS['image_score_weight']*raw_score+
        PARAMETERS['coverage_score_weight']*coverage+PARAMETERS['entry_score_weight']*entry_score-
        PARAMETERS['offset_penalty']*abs(offset)-PARAMETERS['angle_penalty']*abs(angle)/3.-
        PARAMETERS['unsupported_gap_penalty']*gap_fraction)
    return dict(score=float(score),angle_degrees=float(angle),offset_widths=float(offset),
        points=points,ev=ev,runs=runs,gaps=gaps,all_runs=all_runs,
        origin=origin,direction=direction,end=end,protected_length_original_px=supported_length,
        span_original_px=span,heat_score=heat_score,raw_score=raw_score,entry_score=entry_score,
        heat_mean_visible=heat_mean,raw_mean_visible=raw_mean,
        common_view_supported_fraction=coverage,common_view_original_px=reference,
        visible_straight_length_original_px=visible_length,joint_straight_length_original_px=joint_length,
        heat_only_straight_length_original_px=visible_length-joint_length,
        length_bin=int(visible_length//PARAMETERS['length_rank_bin_original_px']),
        terminal_domain=terminal,beyond_terminal_diagnostic=beyond,
        total_gap_original_px=gap_length)


def _rank(c):
    return (c.get('length_bin',-1),-abs(c.get('angle_degrees',180.)),c.get('score',-1.),c.get('joint_straight_length_original_px',0.))


def _summary(c):
    keys=['score','angle_degrees','offset_widths','protected_length_original_px','span_original_px',
          'heat_score','raw_score','entry_score','total_gap_original_px',
          'heat_mean_visible','raw_mean_visible','common_view_supported_fraction','common_view_original_px',
          'visible_straight_length_original_px','joint_straight_length_original_px','heat_only_straight_length_original_px','length_bin','terminal_domain']
    return {k:c.get(k) for k in keys}


def describe_free(priors,seed,heat):
    tick=time.perf_counter();cfg=PARAMETERS;offset_image=np.asarray(cfg['crop_origin_original_xy']);scale=.5
    seed_original=np.asarray(seed['xy'],float)
    if len(seed_original)<2 or not np.isfinite(seed_original).all():raise ValueError('Finite ordered original seed required')
    seed_local=(seed_original-offset_image)*scale
    seed_length=float(np.linalg.norm(seed_local[-1]-seed_local[0]))
    if seed_length<1:raise ValueError('Nonzero seed direction required')
    response=np.asarray(heat,np.float32);image=priors['original_half'][525:]
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
    if response.shape!=gray.shape:raise ValueError('Expected the full H1 half-size crop response')
    finite=response[np.isfinite(response)]
    if len(finite) and (finite.min() < -1.e-5 or finite.max()>1.00001):raise ValueError('Response must be frozen B 0..1 floats')
    valid=np.isfinite(response).astype(np.float32)
    gray=cv2.GaussianBlur(gray,(0,0),.7);response=cv2.GaussianBlur(np.nan_to_num(response,nan=0.),(0,0),.7)
    geometry=_initial_geometry(priors,seed);domain=_surface_domain(priors)
    width=geometry['width_original_px']*.5;initial=geometry['direction'];normal=np.array([-initial[1],initial[0]])
    base=dict(points_xy=[],supported=[],protect_support=[],point_status=[],segment_status=[],
        width_half_px=width,continuation_starts=[],visible_runs=[],gap_hypotheses=[],
        available=False,entry_conflict=geometry['entry_conflict'],actual_seed_xy=seed_original.tolist(),
        coordinate_system='half_size_crop_xy',original_xy_mapping='xy / 0.5 + [0, 1050]',
        same_material_identity_confirmed=False)
    diagnostics=dict(parameters=cfg,width_source=geometry['width_source'],
        initial_direction_source=geometry['direction_source'],initial_direction_xy=initial.tolist(),
        accepted_A7_match=geometry['accepted_a7_match'],a7_match=geometry['a7_match'],
        a7_candidate_id=geometry['a7_candidate']['candidate_id'] if geometry['a7_candidate'] else None,
        old_endpoint_used=False,seed_changed=False,B_response_changed=False,
        raw_zero_pixels=int((gray==0).sum()),nan_response_pixels=int((valid==0).sum()),
        evidence_is_identity_probability=False,labels_read=False)
    if geometry['entry_conflict']:
        diagnostics.update(elapsed_seconds=time.perf_counter()-tick,axis_candidates=0,
            reason='different_accepted_A7_entry_axis_original_entry_not_replaced')
        return base|dict(reason='entry_conflict_with_A7',diagnostics=diagnostics)
    coarse=[]
    for angle in cfg['angle_offsets_degrees']:
        d=_rot(initial,angle)
        for offset in cfg['normal_offsets_widths']:
            origin=seed_local[0]+offset*width*normal
            c=_evaluate_axis(gray,response,valid,origin,d,width,seed_length,angle,offset,cfg['coarse_step_original_px']*.5,domain)
            if c is not None:coarse.append(c)
    coarse.sort(key=_rank,reverse=True)
    if not coarse or not coarse[0].get('runs'):
        diagnostics.update(elapsed_seconds=time.perf_counter()-tick,axis_candidates=len(coarse),
            best_candidates=[_summary(c) for c in coarse[:6]])
        return base|dict(reason='no_entry_associated_supported_heat_axis',diagnostics=diagnostics)
    # Refine around at most three distinct coarse axes, including the initial
    # hypothesis when it ranks well. No sample-specific iterations or thresholds.
    longest=max((c for c in coarse if c.get('runs')),key=_rank)
    parents=[longest];parent_reasons=['longest_existing_threshold_passed_heat_axis']
    for c in coarse:
        if not c.get('runs'):continue
        if any(abs(c['angle_degrees']-p['angle_degrees'])<.5 and abs(c['offset_widths']-p['offset_widths'])<.4 for p in parents):continue
        parents.append(c)
        parent_reasons.append('common_view_evidence_score_with_axis_diversity')
        if len(parents)>=cfg['refine_top_axes']:break
    fine=[];seen=set()
    for c in parents:
        for da in cfg['refine_angles_degrees']:
            for dn in cfg['refine_offsets_widths']:
                angle=c['angle_degrees']+da;shift=c['offset_widths']+dn
                if abs(angle)>3.125 or abs(shift)>1.10:continue
                key=(round(angle,6),round(shift,6))
                if key in seen:continue
                seen.add(key)
                f=_evaluate_axis(gray,response,valid,seed_local[0]+shift*width*normal,_rot(initial,angle),
                    width,seed_length,angle,shift,cfg['final_step_original_px']*.5,domain)
                if f is not None:fine.append(f)
    assert len(coarse)+len(fine)<=cfg['maximum_total_axis_candidates']
    fine.sort(key=_rank,reverse=True)
    chosen=fine[0] if fine and fine[0].get('runs') else None
    if chosen is None:
        diagnostics.update(elapsed_seconds=time.perf_counter()-tick,axis_candidates=len(coarse)+len(fine))
        return base|dict(reason='fine_axis_did_not_retain_visible_run',diagnostics=diagnostics)
    end=chosen['end'];points=chosen['points'][:end];ev={k:np.asarray(v[:end]) for k,v in chosen['ev'].items()}
    protect=np.zeros(end,bool)
    for run in chosen['runs']:
        lo,hi=run['start_index'],run['end_index']+1
        protect[lo:hi]=ev['heat_profile_supported'][lo:hi]
    supported=protect & ev['image_heat_supported']
    entry=np.arange(end)*cfg['final_step_original_px']*.5<=seed_length
    states=['entry_condition' if entry[i] else 'image_supported' if supported[i] else
            'heatmap_supported' if protect[i] else 'ambiguous_gap' for i in range(end)]
    segment_status=[]
    for i in range(1,end):
        segment_status.append('entry_condition' if entry[i-1] and entry[i] else
            'image_supported' if supported[i-1] and supported[i] else
            'heatmap_supported' if protect[i-1] and protect[i] else 'bridge')
    ids=np.flatnonzero(protect & ~entry);starts=[]
    if len(ids):
        last=ids[-1]
        for back in (0.,1.5*width,3.*width):
            j=int(ids[np.argmin(abs((ids-last)*cfg['final_step_original_px']*.5+back))])
            if any(s['point_index']==j for s in starts):continue
            starts.append(dict(point_index=j,xy=points[j].tolist(),tangent=chosen['direction'].tolist(),
                width_half_px=width,heat_profile_supported=bool(protect[j]),raw_image_supported=bool(supported[j]),
                basis='last_continuous_B_ridge_piece_not_physical_contact'))
    diagnostics.update(axis_candidates=len(coarse)+len(fine),coarse_axes=len(coarse),refined_axes=len(fine),
        selected_axis=_summary(chosen),best_coarse_candidates=[_summary(c) for c in coarse[:10]],
        refinement_parent_axes=[dict(_summary(c),selection_basis=reason) for c,reason in zip(parents,parent_reasons)],
        score_normalization='primary: full finite terminal-domain observed interval length excluding entry/unknown, 48 original px bins; secondary: smaller angle change then auxiliary image score. No 2000px cutoff.',
        best_refined_candidates=[_summary(c) for c in fine[:8]],
        initial_original_seed_xy=seed_original.tolist(),selected_origin_original_xy=(chosen['origin']/scale+offset_image).tolist(),
        selected_direction_xy=chosen['direction'].tolist(),normal_shift_original_px=float(chosen['offset_widths']*width/scale),
        visible_piece_count=len(chosen['runs']),all_visible_runs=chosen['all_runs'],
        stop_reason='far_side_SAM2_terminal_domain_or_last_supported_axis_piece_fixed_gap_budget_not_contact_truth',
        terminal_domain=chosen['terminal_domain'],
        new_support_source='new_original_gray_and_frozen_B_profile_measurement_not_old_mask',
        protect_support_semantics='B_center_and_two_low_bands_continuity_not_raw_or_identity_truth',
        elapsed_seconds=time.perf_counter()-tick,measurement_source_sha256=sha(H2/'free_candidates.py'))
    result=base|dict(available=True,reason='finite_entry_associated_heat_axis',points_xy=points.tolist(),
        supported=supported.tolist(),protect_support=protect.tolist(),point_status=states,segment_status=segment_status,
        evidence={k:v.tolist() for k,v in ev.items()},center=ev['center'].tolist(),side_drop=ev['side_drop'].tolist(),
        heat_profile_supported=ev['heat_profile_supported'].tolist(),continuation_starts=starts,
        visible_runs=chosen['runs'],gap_hypotheses=chosen['gaps'],axis_direction_xy=chosen['direction'].tolist(),
        free_axis_id='axis_00',terminal_domain=chosen['terminal_domain'],
        beyond_terminal_diagnostic=chosen['beyond_terminal_diagnostic'],
        visible_straight_length_original_px=chosen['visible_straight_length_original_px'],
        joint_straight_length_original_px=chosen['joint_straight_length_original_px'],
        diagnostics=diagnostics)
    # Two additional admissible axes stay available to the connector. The
    # caller must copy the actually selected free axis into entry.free and
    # retain its free_axis_id; another axis's length must not score its path.
    alternatives=[];axes=[chosen]
    for axis in fine:
        if not axis.get('runs'):continue
        if any(abs(axis['angle_degrees']-other['angle_degrees'])<.2 and
               abs(axis['offset_widths']-other['offset_widths'])<.2 for other in axes):continue
        count=axis['end'];ap=axis['points'][:count]
        ae={k:np.asarray(v[:count]) for k,v in axis['ev'].items()}
        ah=np.zeros(count,bool)
        for run in axis['runs']:
            a,b=run['start_index'],run['end_index']+1;ah[a:b]=ae['heat_profile_supported'][a:b]
        aj=ah & ae['image_heat_supported']
        at=np.arange(count)*cfg['final_step_original_px']*.5<=seed_length
        ast=['entry_condition' if at[i] else 'image_supported' if aj[i] else 'heatmap_supported' if ah[i] else 'ambiguous_gap' for i in range(count)]
        seg=['entry_condition' if at[i-1] and at[i] else 'image_supported' if aj[i-1] and aj[i] else 'heatmap_supported' if ah[i-1] and ah[i] else 'bridge' for i in range(1,count)]
        alt_starts=[];valid_ids=np.flatnonzero(ah & ~at)
        if len(valid_ids):
            last=valid_ids[-1]
            for back in (0.,1.5*width,3.*width):
                j=int(valid_ids[np.argmin(abs((valid_ids-last)*cfg['final_step_original_px']*.5+back))])
                if any(s['point_index']==j for s in alt_starts):continue
                alt_starts.append(dict(point_index=j,xy=ap[j].tolist(),tangent=axis['direction'].tolist(),width_half_px=width,
                    heat_profile_supported=bool(ah[j]),raw_image_supported=bool(aj[j]),basis='alternative_observed_axis_end_not_contact_truth'))
        aid=f'axis_{len(alternatives)+1:02d}'
        alternatives.append(base|dict(available=True,reason='alternative_entry_associated_heat_axis',free_axis_id=aid,
            points_xy=ap.tolist(),supported=aj.tolist(),protect_support=ah.tolist(),point_status=ast,segment_status=seg,
            evidence={k:v.tolist() for k,v in ae.items()},center=ae['center'].tolist(),side_drop=ae['side_drop'].tolist(),
            heat_profile_supported=ae['heat_profile_supported'].tolist(),continuation_starts=alt_starts,
            visible_runs=axis['runs'],gap_hypotheses=axis['gaps'],axis_direction_xy=axis['direction'].tolist(),
            visible_straight_length_original_px=axis['visible_straight_length_original_px'],
            joint_straight_length_original_px=axis['joint_straight_length_original_px'],
            terminal_domain=axis['terminal_domain'],beyond_terminal_diagnostic=axis['beyond_terminal_diagnostic'],
            diagnostics=dict(diagnostics,selected_axis=_summary(axis),terminal_domain=axis['terminal_domain'],
                selected_direction_xy=axis['direction'].tolist(),selected_origin_original_xy=(axis['origin']/scale+offset_image).tolist())))
        axes.append(axis)
        if len(axes)>=cfg['retained_axis_hypotheses']:break
    result['alternative_free_axes']=alternatives
    return result
