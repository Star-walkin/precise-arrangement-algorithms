"""H4: heat-observed straight prefix -> finite transition -> ONE F3 center track.

No off-track H1 continuation can win. Image support is retained as a gate and
penalty; small direction excursions improve a fixed score, not identity truth.
"""
import cv2
import numpy as np
from common import h2_connect as old
unit=old.unit
sample=old.sample
resample=old.resample
def evidence(gray,heat,p,width,pair=None):
    ev=old.evidence(gray,heat,p,width,pair)
    # A bright field without two low sides is not a rope. Heat morphology is
    # primary; original grayscale remains an independent corroboration term.
    hc=ev['heat_center'];dual=ev['heat_dual'];raw=ev['raw']
    ev['quality']=.55*hc*np.clip(dual/.30,0,1)+.30*dual+.15*raw
    return ev
hermite=old.hermite
_mean=old._mean
_summary=old._summary
_termination=old._termination

PARAMETERS=dict(version='IR-H4.0.0',sample_step_half_px=4.,
    backward_anchor_widths=[0.,2.,4.,8.,12.,16.],maximum_anchor_back_half_px=240.,
    maximum_bridge_half_px=240.,maximum_bridge_widths=16.,
    landing_advances_half_px=[12.,35.,70.,120.,190.],maximum_proposals_per_start=30,
    minimum_tail_half_px=50.,minimum_tail_widths=4.,direction_lookahead_half_px=120.,
    direction_chord_half_px=24.,direction_amplitude_scale_radians=.65,
    direction_total_variation_scale_radians=1.5,join_jump_scale_radians=.35,
    image_weight=.70,direction_weight=.20,lookahead_coverage_weight=.10,
    unsupported_bridge_penalty=.12,abandon_straight_penalty=.25,
    width_log_change_penalty=.06,weak_lane_penalty=.05,
    minimum_bridge_image_quality=.10,minimum_lane_image_quality=.10,
    maximum_abandoned_supported_straight_fraction=.05,
    protected_ridge_tolerance_widths=.40,maximum_abandoned_run_widths=2.,
    heat_center_gated_weight=.55,heat_dual_weight=.30,raw_weight=.15,
    lane_heat_center_minimum=.20,lane_heat_dual_minimum=.12,
    lane_heat_fraction_minimum=.25,lane_heat_continuous_widths=1.5,
    minimum_connection_score=.25,free_abstention_score=0.,
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
    protected=np.asarray(free.get('protect_support',free.get('supported',[])),bool)
    empty=dict(candidates=[],candidate_pool_count=0,score_margin=None,ambiguous=True,
        complete_identity_verified=False,parameters=PARAMETERS)
    if len(p)<3 or not protected.any():return empty|dict(reason='no_supported_free_axis')
    width=max(3.,float(free['width_half_px']));direction=unit(p[-1]-p[0]);sign=1 if direction[0]>=0 else -1
    indices=np.flatnonzero(protected);last=int(indices[-1]);distance=arc(p);starts=[]
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
                # A long observed B ridge cannot be abandoned for a smooth
                # early turn. Compare the whole continuation, not just bridge.
                # Heat-supported and raw-corroborated states remain distinct.
                abandoned=p[idx:last+1]
                keep=protected[idx:last+1]
                continuation=np.vstack([bridge,tp[k+1:]])
                distances=np.min(np.linalg.norm(abandoned[:,None]-continuation[None,:],axis=2),axis=1)
                missed=(distances>PARAMETERS['protected_ridge_tolerance_widths']*width)&keep
                mismatch=float(missed.sum()/max(keep.sum(),1))
                changes=np.diff(np.r_[False,missed,False].astype(np.int8))
                runs=list(zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)))
                longest=max((float(distance[idx+b-1]-distance[idx+a]) for a,b in runs),default=0.)
                if mismatch>PARAMETERS['maximum_abandoned_supported_straight_fraction']:
                    rejected['abandons_supported_straight']+=1;continue
                if longest>max(12.,PARAMETERS['maximum_abandoned_run_widths']*width):
                    rejected['abandons_supported_straight']+=1;continue
                dm=direction_metrics(direction,bridge,tp[k:],width)
                cheap=dm['direction_score']-.45*mismatch-.001*dist[k]
                proposals.append(dict(cheap=cheap,track=tr,order=order,tp=tp,tw=tw,pair=pair,inferred=inferred,
                    idx=idx,k=k,bridge=bridge,direction=dm,mismatch=mismatch,longest=longest,width_ratio=ratio))
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
        score=.70*image_score+.20*dm['direction_score']+.10*dm['lookahead_coverage']-.12*weak_bridge-.25*c['mismatch']-.06*width_cost-.05*weak_lane
        # A gray lane with almost no compatible B morphology may be retained
        # for review, but cannot defeat the explicit 'do not join yet' option.
        heat_ok=(ne['heat_center']>.20)&(ne['heat_dual']>.12)
        change=np.diff(np.r_[False,heat_ok,False].astype(np.int8))
        near_s=s[near]
        heat_runs=list(zip(np.flatnonzero(change==1),np.flatnonzero(change==-1)))
        heat_run=max((float(near_s[b-1]-near_s[a]) for a,b in heat_runs),default=0.)
        eligible=bool(float(heat_ok.mean())>=.25 and heat_run>=max(12.,1.5*width))
        incremental_score=float(score-PARAMETERS['minimum_connection_score'])
        if not eligible:incremental_score=min(incremental_score,-.05)
        fp=p[:idx+1];allp=np.vstack([fp,bridge[1:],tail[1:]])
        observed=np.r_[support[:idx+1],be['supported'][1:],te['supported'][1:] & ps[1:] & ~inf[1:]]
        phases=['free']*len(fp)+['transition']*(len(bridge)-1)+['surface']*(len(tail)-1)
        statuses=free.get('point_status',['image_supported' if v else 'inferred_or_weak' for v in support])[:idx+1]
        statuses=statuses+['image_supported' if v else 'inferred_or_weak' for v in observed[len(fp):]]
        tail_source_indices=c['order'][k+1:]
        assert np.array_equal(allp[np.asarray(phases)=='surface'],np.asarray(tr['points_xy'])[tail_source_indices])
        candidates.append(dict(kind='straight_free_to_F3_center_track',track_id=tr['track_id'],boundary_ids=tr['boundary_ids'],
            points_xy=allp.tolist(),width_half_px=np.r_[np.full(len(fp),width),np.linspace(width,widths[0],len(bridge))[1:],widths[1:]].tolist(),
            supported=observed.tolist(),stage=phases,point_status=statuses,
            rank_score=incremental_score,geometry_score=dm['direction_score'],surface_source_indices=tail_source_indices.tolist(),
            surface_max_deviation_half_px=0.,surface_exact_center_track=True,
            continuation_eligible=eligible,continuation_heat_fraction=float(heat_ok.mean()),continuation_heat_run_half_px=heat_run,
            metrics=dict(free=_summary(free_ev),bridge=_summary(be),near_lane=_summary(ne),far_lane=_summary(fe),
                appearance_score=image_score,unthresholded_connection_score=float(score),direction=dm,unsupported_bridge_fraction=weak_bridge,
                abandoned_verified_free_fraction=c['mismatch'],abandoned_heat_supported_fraction=c['mismatch'],maximum_abandoned_heat_run_half_px=c['longest'],width_log_change=width_cost),
            termination=_termination(allp,surface),join_start_xy=p[idx].tolist(),join_end_xy=tail[0].tolist(),identity_verified=False))
    candidates.sort(key=lambda c:c['rank_score'],reverse=True)
    distinct=[]
    for c in candidates:
        if c['track_id'] in {x['track_id'] for x in distinct}:continue
        distinct.append(c)
        if len(distinct)==3:break
    # Abstention is a real candidate even when geometric joins exist. Score is
    # a fixed decision utility, not calibrated identity confidence.
    free_candidate=dict(kind='free_fragment_no_resolved_connection',track_id=None,points_xy=p.tolist(),width_half_px=[width]*len(p),
            supported=support.tolist(),stage=['free']*len(p),point_status=free.get('point_status',['image_supported' if v else 'inferred_or_weak' for v in support]),
            rank_score=PARAMETERS['free_abstention_score'],geometry_score=1.,metrics=dict(free=_summary(free_ev)),
            termination='heat_observed_straight_end_connection_not_resolved',identity_verified=False)
    distinct=(distinct[:2]+[free_candidate])
    distinct.sort(key=lambda c:c['rank_score'],reverse=True)
    margin=float(distinct[0]['rank_score']-distinct[1]['rank_score']) if len(distinct)>1 else None
    return dict(candidates=distinct,candidate_pool_count=len(candidates),geometric_proposal_count=len(proposals),
        scored_bridge_count=len(chosen),rejected=rejected,score_margin=margin,ambiguous=bool(margin is not None and margin<.045),
        selected_index=0,entry_conflict=bool(free.get('entry_conflict',False)),
        direct_H1_paths_diagnostic_only=len(h1_paths),off_track_H1_candidates_selected=0,
        complete_identity_verified=False,ranking_semantics='retain observed B straight ridge; center-track constraint; heat morphology primary, grayscale corroboration and direction secondary',parameters=PARAMETERS)
