"""Automatic straight-axis support, with separate unsupported-gap hypotheses.

Inputs are half-size crop arrays and an automatically supplied short entry.
No target coordinates, F3 lane labels or fixed scene-specific points are used.
A short free segment estimates one axis; original paired edges and transverse
network shape can recover another visible piece beyond a bounded weak gap.
The gap is NEVER marked as measured rope and this helper does not claim a
physical occluder. Returned anchors are only a search hypothesis, not identity.
"""
import math
import cv2
import numpy as np

PARAMETERS = dict(step_px=3., normal_search_widths=.45, normal_samples=7,
    refit_span_widths=24., refit_max_angle_degrees=1.5,
    minimum_raw=.34, minimum_heat=.16, minimum_template=.12,
    minimum_visible_run_widths=3., short_gap_widths=1.2,
    maximum_reappearance_gap_widths=16., maximum_reappearance_gap_px=180.,
    minimum_reappearance_widths=4., maximum_width_change_ratio=1.45,
    initial_run_start_extra_widths=3., maximum_axis_length_px=2200.)


def runs(mask):
    values=np.r_[False,np.asarray(mask,bool),False].astype(np.int8)
    return list(zip(np.flatnonzero(np.diff(values)==1),np.flatnonzero(np.diff(values)==-1)))


def _scan(fields, points, direction, width, template):
    # Import only when called: h1_detector may itself use this helper.
    from h1_detector import evaluate
    normal=np.array([-direction[1],direction[0]])
    offsets=np.linspace(-PARAMETERS['normal_search_widths'],PARAMETERS['normal_search_widths'],PARAMETERS['normal_samples'])*width
    q=(points[:,None,:]+offsets[None,:,None]*normal).reshape(-1,2)
    ev=evaluate(fields,q,np.full(len(q),math.atan2(direction[1],direction[0])),width,template)
    score=ev['score'].reshape(len(points),-1)-.06*(offsets[None,:]/max(width,1))**2
    choice=score.argmax(1);ix=np.arange(len(points))*len(offsets)+choice
    return {k:np.asarray(v)[ix] for k,v in ev.items()} | {'offset':offsets[choice], 'points':q[ix]}


def straight_support(gray,response,seed,width):
    """Return finite straight anchors and an unfilled support mask.

    Coordinates/width are in the same half-size crop as gray/response. The
    ``points_xy`` sequence is the refitted axis up to the last supported run;
    ``supported`` and ``entry_condition`` distinguish measurements from seed
    and weak gaps. A downstream search may branch from the final run, while
    also retaining its original non-straight candidate search.
    """
    from h1_detector import prepare_template
    gray=np.asarray(gray,np.float32);response=np.asarray(response,np.float32)
    seed=np.asarray(seed,float);width=float(width)
    assert gray.ndim==2 and gray.shape==response.shape and seed.shape==(2,2) and width>0
    assert np.isfinite(gray).all() and np.isfinite(response).all()
    h,w=gray.shape;step=PARAMETERS['step_px']
    origin=seed[0].copy();direction=seed[1]-seed[0];seed_length=float(np.linalg.norm(direction));direction/=seed_length
    fields=dict(gray=cv2.GaussianBlur(gray,(0,0),.7),heat=cv2.GaussianBlur(response,(0,0),.7),
                fdist=np.full(gray.shape,1e3,np.float32),surface=np.zeros_like(gray),inferred=np.zeros_like(gray))
    template=prepare_template(fields,seed,width)
    t=np.arange(0,min(PARAMETERS['maximum_axis_length_px'],math.hypot(h,w)),step)
    points=origin+t[:,None]*direction
    valid=(points[:,0]>=1)&(points[:,0]<w-1)&(points[:,1]>=1)&(points[:,1]<h-1)
    t=t[valid];points=points[valid]
    if len(t)<5:return dict(available=False,reason='entry_axis_outside_image',points_xy=[],supported=[],gap_hypotheses=[])
    first=_scan(fields,points,direction,width,template)
    near=(t<=max(seed_length,PARAMETERS['refit_span_widths']*width))&(first['raw']>.45)&(first['heat']>.16)&(first['valid']>.9)
    refit=dict(applied=False,anchor_count=int(near.sum()))
    if near.sum()>=8 and np.ptp(t[near])>=max(2*width,24):
        selected=near.copy();coef=None
        for _ in range(3):
            coef=np.polyfit(t[selected],first['offset'][selected],1)
            residual=first['offset']-np.polyval(coef,t)
            selected=near&(abs(residual)<max(width*.18,1.5))
            if selected.sum()<6:break
        if coef is not None and abs(math.degrees(math.atan(coef[0])))<=PARAMETERS['refit_max_angle_degrees'] and abs(coef[1])<width*.7:
            normal=np.array([-direction[1],direction[0]])
            origin=origin+coef[1]*normal;direction=direction+coef[0]*normal;direction/=np.linalg.norm(direction)
            points=origin+t[:,None]*direction
            refit.update(applied=True,slope=float(coef[0]),intercept_px=float(coef[1]))
    ev=_scan(fields,points,direction,width,template)
    # Thresholds describe image support, not calibrated identity probability.
    supported=(ev['raw']>=PARAMETERS['minimum_raw'])&(ev['heat']>=PARAMETERS['minimum_heat'])&(ev['template']>=PARAMETERS['minimum_template'])&(ev['valid']>=.9)
    supported &= (ev['width']>=width/PARAMETERS['maximum_width_change_ratio'])&(ev['width']<=width*PARAMETERS['maximum_width_change_ratio'])
    # Small strand interruptions can belong to one run. Their measured support
    # stays False: the temporary grouping mask is not the output support mask.
    grouping=supported.copy()
    for a,b in runs(~grouping):
        if a>0 and b<len(grouping) and (b-a)*step<=width*PARAMETERS['short_gap_widths']:
            grouping[a:b]=True
    visible=[]
    for a,b in runs(grouping):
        if (b-a)*step<max(width*PARAMETERS['minimum_visible_run_widths'],18):continue
        good=supported[a:b]
        if good.mean()<.60:continue
        visible.append(dict(start_index=int(a),end_index=int(b-1),length_px=float((b-a)*step),
                            raw_median=float(np.median(ev['raw'][a:b][good])),
                            heat_median=float(np.median(ev['heat'][a:b][good])),
                            width_median=float(np.median(ev['width'][a:b][good])),
                            offset_median=float(np.median(ev['offset'][a:b][good])),
                            supported_fraction=float(good.mean())))
    selected=[];gaps=[]
    for r in visible:
        if not selected:
            if t[r['start_index']]>seed_length+width*PARAMETERS['initial_run_start_extra_widths']:break
            selected.append(r);continue
        prev=selected[-1];a=prev['end_index']+1;b=r['start_index'];gap=(b-a)*step
        if gap>min(width*PARAMETERS['maximum_reappearance_gap_widths'],PARAMETERS['maximum_reappearance_gap_px']):break
        long=gap>width*PARAMETERS['short_gap_widths']
        if long and (r['length_px']<width*PARAMETERS['minimum_reappearance_widths'] or r['raw_median']<.42 or r['heat_median']<.18):break
        if abs(r['offset_median']-prev['offset_median'])>width*.65:break
        ratio=r['width_median']/max(prev['width_median'],1e-6)
        if not 1/PARAMETERS['maximum_width_change_ratio']<=ratio<=PARAMETERS['maximum_width_change_ratio']:break
        if b>a:gaps.append(dict(start_index=int(a),end_index=int(b-1),length_px=float(gap),
                               kind='bounded_same_axis_reappearance_hypothesis' if long else 'short_profile_gap',
                               supported=False,filled_as_measurement=False,physical_occluder_confirmed=False))
        selected.append(r)
    if not selected:
        return dict(available=False,reason='no_initial_supported_straight_run',points_xy=[],supported=[],gap_hypotheses=[],
                    all_visible_runs=visible,refit=refit,parameters=PARAMETERS)
    end=selected[-1]['end_index']+1
    accepted=np.zeros(end,bool)
    for r in selected:accepted[r['start_index']:r['end_index']+1]=supported[r['start_index']:r['end_index']+1]
    # Keep the measured transverse offsets separately; the search-anchor axis
    # itself stays straight and cannot gradually drift onto background grooves.
    last=selected[-1]
    branch_start=max(last['start_index'],last['end_index']-int(max(2*width,18)/step))
    return dict(available=True,reason='image_supported_straight_pieces',points_xy=points[:end].tolist(),
                measured_points_xy=ev['points'][:end].tolist(),supported=accepted.tolist(),
                entry_condition=(t[:end]<=seed_length).tolist(),axis_origin_xy=origin.tolist(),axis_direction_xy=direction.tolist(),
                width_px=width,step_px=step,visible_runs=selected,gap_hypotheses=gaps,
                all_visible_runs=visible,refit=refit,
                turn_search_seed_xy=[points[branch_start].tolist(),points[end-1].tolist()],
                raw=ev['raw'][:end].tolist(),heat=ev['heat'][:end].tolist(),template=ev['template'][:end].tolist(),
                parameters=PARAMETERS,force_straight_after_last_anchor=False,same_material_identity_confirmed=False)
