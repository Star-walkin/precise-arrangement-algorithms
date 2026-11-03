"""Finite, image-scored entry-to-lane hypotheses. Coordinates: crop half pixels.

Projected geometry is a regularizer, never evidence of material identity.
Unobserved bridges are retained explicitly, never converted into annotations.
"""
import cv2
import numpy as np

PARAMETERS = dict(sample_step=4., maximum_bridge_half_px=240., maximum_bridge_widths=16.,
    minimum_lane_after_join_half_px=35., coarse_candidates_per_start=30,
    retained_distinct_candidates=3, ambiguity_margin=.045,
    bridge_image_weight=.36, near_lane_weight=.23, far_lane_weight=.12,
    free_weight=.15, geometry_weight=.14, unsupported_bridge_penalty=.12,
    center_heat_weight=.38, dual_valley_weight=.26, raw_bilateral_weight=.28,
    boundary_pair_weight=.08, maximum_turn_degrees=100., maximum_curvature_per_width=1.2,
    normal_offsets=[-.8,-.6,-.4,-.2,0.,.2,.4,.6,.8], labels_read=False)

def unit(v):
    v=np.asarray(v,float);return v/max(1e-9,float(np.linalg.norm(v)))

def sample(im,p):
    p=np.asarray(p,np.float32);shape=p.shape[:-1];q=p.reshape(-1,2)
    return cv2.remap(np.asarray(im,np.float32),q[:,0][None],q[:,1][None],cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT).reshape(shape)

def resample(points,step=4.):
    p=np.asarray(points,float)
    if len(p)<2:return p
    dist=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    keep=np.r_[True,np.diff(dist)>1e-5];p=p[keep];dist=dist[keep]
    if len(p)<2:return p
    t=np.linspace(0,dist[-1],max(2,int(dist[-1]/step)+1))
    return np.stack([np.interp(t,dist,p[:,j]) for j in range(2)],1)

def evidence(gray,heat,p,width,pair=None):
    p=np.asarray(p,float);n=len(p)
    if n<2:return {k:np.zeros(n) for k in ('quality','raw','heat_center','heat_dual','supported')}
    d=np.gradient(p,axis=0);d/=np.maximum(np.linalg.norm(d,axis=1)[:,None],1e-6)
    normal=np.stack([-d[:,1],d[:,0]],1)
    width=np.broadcast_to(np.asarray(width,float),(n,))
    offsets=np.asarray(PARAMETERS['normal_offsets'])
    q=p[:,None,:]+normal[:,None,:]*width[:,None,None]*offsets[None,:,None]
    # Short along-axis pooling suppresses a single oblique strand or hot pixel.
    gp=sum(sample(gray,q+d[:,None,:]*v*width[:,None,None]) for v in (-.35,0,.35))/3
    hp=sum(sample(heat,q+d[:,None,:]*v*width[:,None,None]) for v in (-.35,0,.35))/3
    gc=gp[:,3:6].mean(1);hc=hp[:,3:6].mean(1)
    gl=gc-gp[:,:3].mean(1);gr=gc-gp[:,6:].mean(1)
    raw=np.clip(np.minimum(abs(gl),abs(gr))/np.maximum(gp.std(1),.025),0,1)*(gl*gr>0)
    hl=hc-hp[:,:3].min(1);hr=hc-hp[:,6:].min(1)
    dual=np.clip(np.minimum(hl,hr)/.18,0,1)*np.clip(1-abs(hl-hr)/(abs(hl)+abs(hr)+.08),0,1)
    pair=np.zeros(n) if pair is None else np.broadcast_to(pair,(n,)).astype(float)
    quality=.38*hc+.26*dual+.28*raw+.08*pair
    valid=(p[:,0]>=width)&(p[:,0]<heat.shape[1]-width)&(p[:,1]>=width)&(p[:,1]<heat.shape[0]-width)
    supported=valid & (raw>.18) & ((hc>.22)|(dual>.22))
    return dict(quality=quality,raw=raw,heat_center=hc,heat_dual=dual,supported=supported)

def hermite(a,b,da,db):
    distance=float(np.linalg.norm(b-a));n=max(3,int(distance/4)+1);t=np.linspace(0,1,n)[:,None]
    ma=unit(da)*distance*.85;mb=unit(db)*distance*.85
    return (2*t**3-3*t**2+1)*a+(t**3-2*t**2+t)*ma+(-2*t**3+3*t**2)*b+(t**3-t**2)*mb

def morphology(p,width):
    d=np.diff(p,axis=0);length=np.linalg.norm(d,axis=1);u=d/np.maximum(length[:,None],1e-8)
    if len(u)<2:return 1.,0.,0.
    turns=np.arccos(np.clip((u[1:]*u[:-1]).sum(1),-1,1))
    total=float(turns.sum());curv=float(np.max(turns/np.maximum((length[1:]+length[:-1])*.5,1))*width)
    stretch=float(length.sum()/max(np.linalg.norm(p[-1]-p[0]),1))
    score=np.exp(-.7*total-.45*max(0,stretch-1)-.5*curv)
    return float(score),float(total*180/np.pi),curv

def _mean(ev,key):return float(np.mean(ev[key])) if len(ev[key]) else 0.

def _summary(ev):return {k:_mean(ev,k) for k in ev}

def _termination(points,surface):
    end=np.asarray(points[-1]);h,w=surface.shape
    if min(end[0],w-1-end[0],h-1-end[1])<7:
        return 'original_image_boundary_out_of_view' if end[1]>7 else 'model_crop_boundary_not_task_endpoint'
    binary=(surface>0).astype(np.uint8)
    boundary=cv2.morphologyEx(binary,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
    boundary_distance=cv2.distanceTransform(1-boundary,cv2.DIST_L2,3)
    if sample(boundary_distance,end[None])[0]<12:return 'saved_SAM2_surface_edge_candidate_not_identity_verified'
    if sample(binary,end[None])[0]<.5:return 'endpoint_outside_saved_surface_not_a_surface_edge'
    return 'F3_pair_support_ends_before_confirmed_surface_edge'

def connect(gray,heat,free,tracks,h1_paths,surface):
    """Rank same-condition direct/Hermite+lane candidates with common evidence.

    Free prefix is allowed only through measured straight continuation. Different
    lane hypotheses receive identical bridge generation, profile and regularizers.
    """
    p=np.asarray(free.get('points_xy',[]),float).reshape(-1,2)
    width=float(np.median(free.get('width_half_px',12.)))
    if len(p)<2:return dict(candidates=[],reason='no_free_axis',complete_identity_verified=False)
    width=max(3.,width);direction=unit(p[-1]-p[0]);sign=1 if direction[0]>=0 else -1
    supported=np.asarray(free.get('supported',[False]*len(p)),bool)
    # Anchor at final observed free section, plus two nearby observations. Never
    # use a low-score extrapolated free end as an observed contact point.
    indices=np.flatnonzero(supported)
    if not len(indices):return dict(candidates=[],reason='no_observed_free_anchor',complete_identity_verified=False)
    last=int(indices[-1])
    starts=[]
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    for back in (0.,1.5*width,3.*width):
        idx=int(indices[np.argmin(abs(distance[indices]-(distance[last]-back)))])
        if idx>=1 and idx not in starts:starts.append(idx)
    free_ev=evidence(gray,heat,p[:last+1],width)
    free_quality=_mean(free_ev,'quality')
    candidates=[];rejected=dict(geometry=0,too_short=0,wrong_direction=0)
    coarse=[]
    for track in tracks:
        tp=np.asarray(track['points_xy'],float)
        if len(tp)<3:continue
        order=np.arange(len(tp)) if sign>0 else np.arange(len(tp)-1,-1,-1)
        tp=tp[order];tw=np.asarray(track['spacing_px'],float)[order]
        pair=np.asarray(track['f3_pair_supported'],bool)[order]
        inferred=np.asarray(track['f3_pair_inferred'],bool)[order]
        for idx in starts:
            a=p[idx];da=unit(p[idx]-p[max(0,idx-4)])
            delta=tp-a;advance=delta@da;lateral=abs(delta[:,0]*da[1]-delta[:,1]*da[0])
            dist=np.linalg.norm(delta,axis=1)
            ok=(advance>6)&(advance<240)&(dist<min(240,max(90,16*width)))&(lateral<max(55,7*width))
            eligible=np.flatnonzero(ok)
            if not len(eligible):continue
            # Several forward landing locations, not just the nearest lane point.
            for target in (25.,65.,115.,185.):
                k=int(eligible[np.argmin(abs(advance[eligible]-target))])
                if np.linalg.norm(tp[-1]-tp[k])<35:rejected['too_short']+=1;continue
                db=unit(tp[min(k+4,len(tp)-1)]-tp[max(0,k-2)])
                if np.dot(da,db)<-.2:rejected['wrong_direction']+=1;continue
                ratio=float(tw[k]/width)
                if not .38<ratio<2.7:continue
                bridge=hermite(a,tp[k],da,db)
                geo,turn,curv=morphology(bridge,width)
                if turn>100 or curv>1.2:rejected['geometry']+=1;continue
                geo*=float(np.exp(-.3*abs(np.log(ratio))))
                # Penalize dropping a verified later straight section; joins can
                # begin earlier only when their bridge agrees with that section.
                abandoned=p[idx:last+1]
                mismatch=float(np.mean(np.min(np.linalg.norm(abandoned[:,None]-bridge[None,:],axis=2),axis=1)>width*.65)) if len(abandoned)>1 else 0.
                cheap=geo-.45*mismatch-.001*dist[k]
                coarse.append((cheap,track,order,tp,tw,pair,inferred,idx,k,bridge,geo,mismatch,turn,curv))
    coarse.sort(key=lambda x:x[0],reverse=True)
    # Fair finite proposal cap applied before inspecting network score.
    kept=[];keys=set();per_start={}
    for c in coarse:
        key=(c[1]['track_id'],c[7],int(c[8]/5))
        if key in keys or per_start.get(c[7],0)>=30:continue
        keys.add(key);per_start[c[7]]=per_start.get(c[7],0)+1;kept.append(c)
    for _,track,order,tp,tw,pair,inferred,idx,k,bridge,geo,mismatch,turn,curv in kept:
        tail=resample(tp[k:]);ws=np.interp(tail[:,0]*sign,tp[:,0]*sign,tw)
        ps=np.interp(tail[:,0]*sign,tp[:,0]*sign,pair.astype(float))>.9
        ins=np.interp(tail[:,0]*sign,tp[:,0]*sign,inferred.astype(float))>.1
        be=evidence(gray,heat,bridge,np.linspace(width,tw[k],len(bridge)))
        te=evidence(gray,heat,tail,ws,ps);near=np.linalg.norm(tail-tail[0],axis=1)<max(70,6*width)
        near_ev={q:v[near] for q,v in te.items()};far_ev={q:v[~near] for q,v in te.items()}
        bgap=1-_mean(be,'supported')
        score=.15*free_quality+.36*_mean(be,'quality')+.23*_mean(near_ev,'quality')+.12*_mean(far_ev if len(far_ev['quality']) else near_ev,'quality')+.14*geo-.12*bgap-.25*mismatch
        fp=p[:idx+1];fe=evidence(gray,heat,fp,width)
        allp=np.concatenate([fp,bridge[1:],tail[1:]])
        support=np.r_[np.asarray(free.get('supported',[False]*len(p)))[:idx+1],be['supported'][1:],te['supported'][1:] & ps[1:] & ~ins[1:]]
        stage=['free']*len(fp)+['transition']*(len(bridge)-1)+['surface']*(len(tail)-1)
        states=['image_supported' if s else 'inferred_or_weak' for s in support]
        candidates.append(dict(kind='free_bridge_adjacent_F3_lane',track_id=track['track_id'],boundary_ids=track['boundary_ids'],points_xy=allp.tolist(),width_half_px=np.r_[np.full(len(fp),width),np.linspace(width,tw[k],len(bridge))[1:],ws[1:]].tolist(),supported=support.tolist(),stage=stage,point_status=states,rank_score=float(score),geometry_score=float(geo),metrics=dict(free=_summary(fe),bridge=_summary(be),near_lane=_summary(near_ev),far_lane=_summary(far_ev),unsupported_bridge_fraction=bgap,abandoned_verified_free_fraction=mismatch,turn_degrees=turn,max_curvature_per_width=curv),termination=_termination(allp,surface),join_start_xy=p[idx].tolist(),join_end_xy=tp[k].tolist(),identity_verified=False))
    # Preserve actual response-following H1 alternatives in the same pool.
    for n,hp in enumerate(h1_paths[:3]):
        q=(np.asarray(hp['points_xy'],float)-[0,1050])*.5
        if len(q)<3:continue
        original_q=q.copy();q=resample(q);dist=np.linalg.norm(q-p[last],axis=1);j=int(np.argmin(dist))
        if dist[j]>max(20,2*width):continue
        fe=evidence(gray,heat,q[:max(2,j+1)],width)
        rest=q[j:]
        if len(rest)<3:continue
        be=evidence(gray,heat,rest[:min(len(rest),25)],width)
        te=evidence(gray,heat,rest[min(len(rest)-2,25):],width)
        geo,turn,curv=morphology(rest[:min(len(rest),30)],width)
        score=.15*_mean(fe,'quality')+.36*_mean(be,'quality')+.35*_mean(te,'quality')+.14*geo-.12*(1-_mean(be,'supported'))
        ev=evidence(gray,heat,q,width)
        original_arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(original_q,axis=0),axis=1))]
        new_arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))]
        prior_observed=np.interp(new_arc,original_arc,np.asarray(hp['supported'],float))>.99
        ev['supported'] &= prior_observed
        termination=_termination(q,surface) if hp['termination']=='image_or_model_roi_edge' else hp['termination']
        candidates.append(dict(kind='direct_H1_response_path',track_id=None,points_xy=q.tolist(),width_half_px=[width]*len(q),supported=ev['supported'].tolist(),stage=['free' if z<=j else 'response_continuation' for z in range(len(q))],point_status=['image_supported' if s else 'inferred_or_weak' for s in ev['supported']],rank_score=float(score),geometry_score=float(geo),metrics=dict(free=_summary(fe),bridge=_summary(be),near_lane=_summary(te),far_lane=_summary(te)),termination=termination,original_h1_stop=hp['termination'],identity_verified=False))
    if not candidates and len(p)>2:
        candidates.append(dict(kind='free_fragment_no_resolved_connection',track_id=None,points_xy=p.tolist(),width_half_px=[width]*len(p),supported=supported.tolist(),stage=['free']*len(p),point_status=['image_supported' if s else 'inferred_or_weak' for s in supported],rank_score=free_quality-.25,geometry_score=0.,metrics=dict(free=_summary(free_ev)),termination='connection_unresolved_at_last_supported_free_piece',identity_verified=False))
    conflict=bool(free.get('entry_conflict',False))
    if conflict:
        for c in candidates:
            c['rank_score']-=.5;c['supported']=[False]*len(c['points_xy']);c['point_status']=['conflicting_entry']*len(c['points_xy'])
    candidates.sort(key=lambda c:c['rank_score'],reverse=True)
    # Keep genuinely different downstream curves; shifts of the same bridge
    # landing on the same lane are not three independent explanations.
    distinct=[]
    for c in candidates:
        q=np.asarray(c['points_xy']);duplicate=False
        for old in distinct:
            r=np.asarray(old['points_xy'])
            if c['track_id'] is not None and c['track_id']==old['track_id']:duplicate=True;break
            a=q[-min(40,len(q)):];b=r[-min(60,len(r)):]
            if np.mean(np.min(np.linalg.norm(a[:,None]-b[None,:],axis=2),axis=1))<width*.65:duplicate=True;break
        if not duplicate:distinct.append(c)
        if len(distinct)==3:break
    margin=float(distinct[0]['rank_score']-distinct[1]['rank_score']) if len(distinct)>1 else None
    ambiguous=conflict or (margin is not None and margin<PARAMETERS['ambiguity_margin'])
    geom=max(candidates,key=lambda c:c['geometry_score']) if candidates else None
    return dict(candidates=distinct,candidate_pool_count=len(candidates),geometric_proposal_count=len(coarse),scored_bridge_count=len(kept),rejected=rejected,score_margin=margin,ambiguous=bool(ambiguous),entry_conflict=conflict,selected_index=0 if distinct else None,geometry_only_best_track_id=geom['track_id'] if geom else None,complete_identity_verified=False,ranking_semantics='fixed heuristic comparison; best candidate not verified identity',parameters=PARAMETERS)
