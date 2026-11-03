"""H3: straight free prefix -> finite transition -> ONE saved F3 center track.

No off-track H1 continuation can win. Image support is retained as a gate and
penalty; small direction excursions improve a fixed score, not identity truth.
"""
import cv2
import numpy as np
from common import h2_connect as old
unit=old.unit
sample=old.sample
resample=old.resample
evidence=old.evidence
hermite=old.hermite
_mean=old._mean
_summary=old._summary
_termination=old._termination

PARAMETERS=dict(version='IR-H3.0.0',sample_step_half_px=4.,
    backward_anchor_widths=[0.,2.,4.,8.,12.,16.],maximum_anchor_back_half_px=240.,
    maximum_bridge_half_px=240.,maximum_bridge_widths=16.,
    landing_advances_half_px=[12.,35.,70.,120.,190.],maximum_proposals_per_start=30,
    minimum_tail_half_px=50.,minimum_tail_widths=4.,direction_lookahead_half_px=120.,
    direction_chord_half_px=24.,direction_amplitude_scale_radians=.65,
    direction_total_variation_scale_radians=1.5,join_jump_scale_radians=.35,
    image_weight=.50,direction_weight=.40,lookahead_coverage_weight=.10,
    unsupported_bridge_penalty=.12,abandon_straight_penalty=.25,
    width_log_change_penalty=.06,weak_lane_penalty=.05,
    minimum_bridge_image_quality=.10,minimum_lane_image_quality=.10,
    maximum_abandoned_supported_straight_fraction=.10,
    maximum_turn_degrees=100.,maximum_curvature_per_width=1.2,
    retained_distinct_candidates=3,ambiguity_margin=.045,
    surface_tail_exact_source_points=True,off_track_H1_eligible=False)

def arc(p):return np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]

def direction_metrics(free_tangent,bridge,tail,width):
    """Compare a common120px downstream horizon, not arbitrary total length.

    24px chords suppress raster-level jitter only for measurement. Output track
    points stay exactly unchanged. No straight shortcut replaces a gray lane.
    """
    tail=np.asarray(tail,float);s=arc(tail);limit=PARAMETERS['direction_lookahead_half_px']
    horizon=min(float(s[-1]),limit)
    if s[-1]>limit:
        k=int(np.searchsorted(s,limit));end=tail[k-1]+(tail[k]-tail[k-1])*(limit-s[k-1])/(s[k]-s[k-1])
        tail=np.vstack([tail[:k],end])
    prefix=bridge[0]-np.asarray(free_tangent)*max(24.,2*width)
    path=resample(np.vstack([prefix,bridge,tail[1:]]),4.)
    ss=arc(path);half=PARAMETERS['direction_chord_half_px']*.5
    lo=np.maximum(0,ss-half);hi=np.minimum(ss[-1],ss+half)
    pa=np.stack([np.interp(lo,ss,path[:,j]) for j in range(2)],1)
    pb=np.stack([np.interp(hi,ss,path[:,j]) for j in range(2)],1)
    angles=np.unwrap(np.arctan2((pb-pa)[:,1],(pb-pa)[:,0]))
    amplitude=float(np.ptp(angles));turns=np.abs(np.diff(angles));variation=float(turns.sum())
    ds=np.maximum(np.diff(ss),1.e-6)
    rms_curvature=float(np.sqrt(np.sum((turns/ds)**2*ds)/max(ds.sum(),1))*width)
    # Explicit entrance/landing tangent mismatch remains measured even when
    # chord smoothing would hide a short corner at concatenation.
    entry=unit(bridge[min(2,len(bridge)-1)]-bridge[0]);exit=unit(bridge[-1]-bridge[max(0,len(bridge)-3)])
    lane=unit(tail[min(4,len(tail)-1)]-tail[0])
    jump=float(np.arccos(np.clip(entry@unit(free_tangent),-1,1))+np.arccos(np.clip(exit@lane,-1,1)))
    cost=amplitude/.65+variation/1.5+.9*rms_curvature+jump/.35
    return dict(amplitude_degrees=float(np.degrees(amplitude)),total_variation_degrees=float(np.degrees(variation)),
        junction_jump_degrees=float(np.degrees(jump)),rms_curvature_times_width=rms_curvature,
        direction_cost=float(cost),direction_score=float(np.exp(-cost)),
        measured_lane_horizon_half_px=horizon,lookahead_coverage=horizon/limit,
        measurement_chord_half_px=24.,surface_coordinates_smoothed=False)

def connect(gray,heat,free,tracks,h1_paths,surface):
    p=np.asarray(free.get('points_xy',[]),float).reshape(-1,2)
    support=np.asarray(free.get('supported',[]),bool)
    empty=dict(candidates=[],candidate_pool_count=0,score_margin=None,ambiguous=True,
        complete_identity_verified=False,parameters=PARAMETERS)
    if len(p)<3 or not support.any():return empty|dict(reason='no_supported_free_axis')
    width=max(3.,float(free['width_half_px']));direction=unit(p[-1]-p[0]);sign=1 if direction[0]>=0 else -1
    indices=np.flatnonzero(support);last=int(indices[-1]);distance=arc(p);starts=[]
    # More than the last3 widths: H2 sometimes extended a diagonal past the
    # usable F3 track. Earlier alternatives remain penalized when they abandon
    # supported straight continuation. This does not label a contact point.
    for w in PARAMETERS['backward_anchor_widths']:
        back=min(PARAMETERS['maximum_anchor_back_half_px'],w*width)
        j=int(indices[np.argmin(abs(distance[indices]-(distance[last]-back)))])
        if j>=2 and j not in starts:starts.append(j)
    free_ev=evidence(gray,heat,p[:last+1],width);free_quality=_mean(free_ev,'quality')
    rejected=dict(short_lane=0,geometry=0,appearance=0,width=0,abandons_supported_straight=0)
    proposals=[]
    for tr in tracks:
        pts=np.asarray(tr['points_xy'],float)
        if len(pts)<3:continue
        order=np.arange(len(pts)) if sign>0 else np.arange(len(pts)-1,-1,-1)
        tp=pts[order];tw=np.asarray(tr['spacing_px'])[order]
        pair=np.asarray(tr['f3_pair_supported'],bool)[order];inferred=np.asarray(tr['f3_pair_inferred'],bool)[order]
        remain=arc(tp);remain=remain[-1]-remain
        for idx in starts:
            a=p[idx];delta=tp-a;advance=delta@direction;dist=np.linalg.norm(delta,axis=1)
            lateral=abs(delta[:,0]*direction[1]-delta[:,1]*direction[0])
            valid=(advance>4)&(advance<240)&(dist<min(240,max(90,16*width)))&(lateral<max(55,7*width))
            valid&=remain>=max(50,4*width)
            possible=np.flatnonzero(valid)
            if not len(possible):continue
            landing_ids=sorted(set(int(possible[np.argmin(abs(advance[possible]-t))]) for t in PARAMETERS['landing_advances_half_px']))
            for k in landing_ids:
                db=unit(tp[min(k+6,len(tp)-1)]-tp[max(0,k-3)])
                if direction@db<-.2:rejected['geometry']+=1;continue
                ratio=float(tw[k]/width)
                if not .38<ratio<2.7:rejected['width']+=1;continue
                bridge=hermite(a,tp[k],direction,db)
                _,turn,curvature=old.morphology(bridge,width)
                if turn>100 or curvature>1.2:rejected['geometry']+=1;continue
                # Unknown gaps do not vote as observed straight evidence.
                abandoned=p[idx:last+1][support[idx:last+1]]
                mismatch=float(np.mean(np.min(np.linalg.norm(abandoned[:,None]-bridge[None,:],axis=2),axis=1)>.65*width)) if len(abandoned) else 0.
                if mismatch>PARAMETERS['maximum_abandoned_supported_straight_fraction']:
                    rejected['abandons_supported_straight']+=1;continue
                dm=direction_metrics(direction,bridge,tp[k:],width)
                cheap=dm['direction_score']-.45*mismatch-.001*dist[k]
                proposals.append(dict(cheap=cheap,track=tr,order=order,tp=tp,tw=tw,pair=pair,inferred=inferred,
                    idx=idx,k=k,bridge=bridge,direction=dm,mismatch=mismatch,width_ratio=ratio))
    proposals.sort(key=lambda c:c['cheap'],reverse=True)
    # A bounded diverse pool: one landing per lane first, then remaining options.
    chosen=[];per_start={};covered=set();seen=set()
    for first_pass in (True,False):
        for c in proposals:
            key=(c['idx'],c['track']['track_id'],c['k']);lane_key=key[:2]
            if key in seen or per_start.get(c['idx'],0)>=30:continue
            if first_pass and lane_key in covered:continue
            chosen.append(c);seen.add(key);covered.add(lane_key);per_start[c['idx']]=per_start.get(c['idx'],0)+1
    candidates=[]
    for c in chosen:
        tr=c['track'];idx=c['idx'];k=c['k'];bridge=c['bridge'];tail=c['tp'][k:]
        widths=c['tw'][k:];ps=c['pair'][k:];inf=c['inferred'][k:]
        be=evidence(gray,heat,bridge,np.linspace(width,widths[0],len(bridge)))
        te=evidence(gray,heat,tail,widths,ps);s=arc(tail);near=s<120
        ne={key:v[near] for key,v in te.items()};fe={key:v[~near] for key,v in te.items()}
        if _mean(be,'quality')<.10 or _mean(ne,'quality')<.10:
            rejected['appearance']+=1;continue
        image_score=(.12*free_quality+.25*_mean(be,'quality')+.20*_mean(ne,'quality')+.08*_mean(fe if len(fe['quality']) else ne,'quality'))/.65
        weak_bridge=1-_mean(be,'supported');weak_lane=1-_mean(ne,'supported')
        dm=c['direction'];width_cost=abs(float(np.log(c['width_ratio'])))
        score=.50*image_score+.40*dm['direction_score']+.10*dm['lookahead_coverage']-.12*weak_bridge-.25*c['mismatch']-.06*width_cost-.05*weak_lane
        fp=p[:idx+1];allp=np.vstack([fp,bridge[1:],tail[1:]])
        observed=np.r_[support[:idx+1],be['supported'][1:],te['supported'][1:] & ps[1:] & ~inf[1:]]
        phases=['free']*len(fp)+['transition']*(len(bridge)-1)+['surface']*(len(tail)-1)
        tail_source_indices=c['order'][k+1:]
        assert np.array_equal(allp[np.asarray(phases)=='surface'],np.asarray(tr['points_xy'])[tail_source_indices])
        candidates.append(dict(kind='straight_free_to_F3_center_track',track_id=tr['track_id'],boundary_ids=tr['boundary_ids'],
            points_xy=allp.tolist(),width_half_px=np.r_[np.full(len(fp),width),np.linspace(width,widths[0],len(bridge))[1:],widths[1:]].tolist(),
            supported=observed.tolist(),stage=phases,point_status=['image_supported' if v else 'inferred_or_weak' for v in observed],
            rank_score=float(score),geometry_score=dm['direction_score'],surface_source_indices=tail_source_indices.tolist(),
            surface_max_deviation_half_px=0.,surface_exact_center_track=True,
            metrics=dict(free=_summary(free_ev),bridge=_summary(be),near_lane=_summary(ne),far_lane=_summary(fe),
                appearance_score=image_score,direction=dm,unsupported_bridge_fraction=weak_bridge,
                abandoned_verified_free_fraction=c['mismatch'],width_log_change=width_cost),
            termination=_termination(allp,surface),join_start_xy=p[idx].tolist(),join_end_xy=tail[0].tolist(),identity_verified=False))
    candidates.sort(key=lambda c:c['rank_score'],reverse=True)
    distinct=[]
    for c in candidates:
        if c['track_id'] in {x['track_id'] for x in distinct}:continue
        distinct.append(c)
        if len(distinct)==3:break
    if not distinct:
        distinct=[dict(kind='free_fragment_no_resolved_connection',track_id=None,points_xy=p.tolist(),width_half_px=[width]*len(p),
            supported=support.tolist(),stage=['free']*len(p),point_status=['image_supported' if v else 'inferred_or_weak' for v in support],
            rank_score=free_quality-.25,geometry_score=1.,metrics=dict(free=_summary(free_ev)),
            termination='no_supported_connection_to_existing_center_track',identity_verified=False)]
    margin=float(distinct[0]['rank_score']-distinct[1]['rank_score']) if len(distinct)>1 else None
    return dict(candidates=distinct,candidate_pool_count=len(candidates),geometric_proposal_count=len(proposals),
        scored_bridge_count=len(chosen),rejected=rejected,score_margin=margin,ambiguous=bool(margin is not None and margin<.045),
        selected_index=0,entry_conflict=bool(free.get('entry_conflict',False)),
        direct_H1_paths_diagnostic_only=len(h1_paths),off_track_H1_candidates_selected=0,
        complete_identity_verified=False,ranking_semantics='center-track constraint then image support and direction-change score',parameters=PARAMETERS)
