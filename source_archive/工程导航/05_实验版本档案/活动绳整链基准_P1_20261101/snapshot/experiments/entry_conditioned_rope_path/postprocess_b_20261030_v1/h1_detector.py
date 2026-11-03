"""H1: polarity-invariant ribbon evidence and direction-aware multi-path search.

Only original pixels, frozen B numeric response, and automatic geometric hints
are used. Heatmap values are not probabilities; F3 edges are not identities.
All computation is on the 0.5-scale ROI, output coordinates are full image.
"""
import math
import cv2
import numpy as np

PARAMETERS = dict(step=6., beam_width=48, max_steps=520,
    angle_changes_degrees=[-3,-1.5,0,1.5,3], max_turn_degrees=85.,
    maximum_upward_turn_degrees=5.,
    raw_weight=.25, heat_weight=.65, template_weight=.07, f3_weight=.03,
    turn_penalty=.07, turn_change_penalty=.08, weak_limit_widths=16.,
    minimum_raw=.20, minimum_heat=.14, support_score=.30,
    traversable_heat=.30,
    normal_samples=33, tangent_offsets=[-1.5,-1.,-.5,0,.5,1.,1.5],
    width_factors=[.85,1.,1.15], final_candidates=3,
    local_image_floor=.018, local_heat_floor=.035)

def sample(im, xy):
    q=np.asarray(xy,np.float32); shape=q.shape[:-1]; p=q.reshape(-1,2)
    out=np.empty(len(p),np.float32)
    for lo in range(0,len(p),24000):
        a=p[lo:lo+24000]
        out[lo:lo+len(a)]=cv2.remap(im,a[:,0][None],a[:,1][None],cv2.INTER_LINEAR,
          borderMode=cv2.BORDER_REFLECT_101).ravel()
    return out.reshape(shape)

def wrap(a):return (a+np.pi)%(2*np.pi)-np.pi

def profiles(im, q, angles, widths):
    q=np.asarray(q,np.float32); angles=np.asarray(angles)
    t=np.c_[np.cos(angles),np.sin(angles)]; n=np.c_[-t[:,1],t[:,0]]
    u=np.linspace(-1.35,1.35,PARAMETERS['normal_samples'])
    v=np.asarray(PARAMETERS['tangent_offsets'])
    xy=q[:,None,None,:]+np.asarray(widths)[:,None,None,None]*(
        u[None,None,:,None]*n[:,None,None,:]+v[None,:,None,None]*t[:,None,None,:])
    values=sample(im,xy).mean(1)
    h,w=im.shape; valid=((xy[...,0]>=0)&(xy[...,0]<w-1)&(xy[...,1]>=0)&(xy[...,1]<h-1)).mean((1,2))
    return values,valid

def ribbon_measure(p, floor):
    """Local slopes remove exposure drift. Either bright or dark bodies count.

    Opposite-sign outer gradients and a central band are evaluated together;
    tiny internal strand scratches cannot alone provide both outer edges.
    """
    u=np.linspace(-1.35,1.35,p.shape[1]); left=p[:,u<-.9].mean(1); right=p[:,u>.9].mean(1)
    bg=left[:,None]+(right-left)[:,None]*(u[None]+1.1)/2.2
    z=p-bg; center=z[:,abs(u)<.25].mean(1)
    scale=np.maximum(np.std(z,axis=1),floor)
    sign=np.where(center>=0,1.,-1.)
    grad=np.diff(p,axis=1); mid=(u[:-1]+u[1:])/2
    # Integrate each flank, instead of selecting one strongest strand edge.
    li=(mid>-.75)&(mid<-.25); ri=(mid>.25)&(mid<.75)
    le=grad[:,li].sum(1)*sign; re=-grad[:,ri].sum(1)*sign
    pair=np.clip(np.minimum(le,re)/(scale*1.8),0,1)
    body=np.clip(abs(center)/(scale*2.0),0,1)
    contrast=np.clip(abs(center)/(floor*3.),0,1)
    out=pair*.6+body*.4
    return out*contrast, z*sign[:,None]/scale[:,None], center

def build_fields(gray,response,merged,inferred,surface):
    gray=cv2.GaussianBlur(gray.astype(np.float32),(0,0),.7)
    heat=cv2.GaussianBlur(response.astype(np.float32),(0,0),.7)
    f3=(merged>0).astype(np.uint8)
    fdist=cv2.distanceTransform(1-f3,cv2.DIST_L2,3)
    contours,_=cv2.findContours((surface>0).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    domain=np.zeros_like(surface,np.uint8)
    substantial=[c for c in contours if cv2.contourArea(c)>100]
    if substantial:cv2.fillConvexPoly(domain,cv2.convexHull(np.concatenate(substantial)),1)
    outside_distance=cv2.distanceTransform(1-domain,cv2.DIST_L2,3)
    return dict(gray=gray,heat=heat,fdist=fdist,surface=surface.astype(np.float32),
      inferred=inferred.astype(np.float32),domain=domain.astype(np.float32),outside_distance=outside_distance)

def evaluate(fields,q,angles,width,template,heat_only=False):
    """All proposals get the same raw/image test and width alternatives."""
    q=np.asarray(q); n=len(q); factors=np.array(PARAMETERS['width_factors'])
    p=np.repeat(q,len(factors),axis=0); a=np.repeat(angles,len(factors)); ww=np.tile(width*factors,n)
    raw,valid=profiles(fields['gray'],p,a,ww); hp,_=profiles(fields['heat'],p,a,ww)
    rawscore,z,_=ribbon_measure(raw,PARAMETERS['local_image_floor'])
    hs,hz,_=ribbon_measure(hp,PARAMETERS['local_heat_floor'])
    # A heatmap may have a narrow central peak or two bright margins around a
    # dark center. Local ripple energy is only accepted with transverse shape.
    henergy=np.clip(np.std(hp,axis=1)/.12,0,1)
    hs=np.maximum(hs,.35*henergy)
    sim=np.clip((z*template[None]).mean(1)/2.,0,1) if template is not None else np.zeros(len(p))
    normal=np.c_[-np.sin(a),np.cos(a)]
    ldist=sample(fields['fdist'],p-normal*ww[:,None]*.5)
    rdist=sample(fields['fdist'],p+normal*ww[:,None]*.5)
    # F3 is only a weak paired-boundary preference once path is near horizontal.
    aligned=np.clip((abs(np.cos(a))-.72)/.28,0,1)
    on_surface=sample(fields['surface'],p)>.5
    edge=np.exp(-(ldist+rdist)/np.maximum(ww*.6,2.))*aligned*on_surface
    inf=sample(fields['inferred'],p)>0
    edge[inf]*=.35
    quality=(PARAMETERS['raw_weight']*rawscore+PARAMETERS['heat_weight']*hs+
      PARAMETERS['template_weight']*sim+PARAMETERS['f3_weight']*edge)
    if heat_only:quality=hs
    quality[valid<.75]=0
    ix=quality.reshape(n,-1).argmax(1); ix=np.arange(n)*len(factors)+ix
    return dict(score=quality[ix],raw=rawscore[ix],heat=hs[ix],template=sim[ix],
      f3=edge[ix],width=ww[ix],valid=valid[ix],response=sample(fields['heat'],q))

def prepare_template(fields,seed,width):
    p=np.asarray(seed,float); d=p[1]-p[0]; a=math.atan2(d[1],d[0])
    q=p[0]+np.linspace(.15,.85,9)[:,None]*d
    raw,_=profiles(fields['gray'],q,np.full(len(q),a),np.full(len(q),width))
    _,z,_=ribbon_measure(raw,PARAMETERS['local_image_floor'])
    return np.median(z,axis=0)

def path_distance(a,b):
    pa=np.array(a['points']);pb=np.array(b['points']);n=min(len(pa),len(pb))
    if n<6:return 0.
    return float(np.percentile(np.linalg.norm(pa[:n]-pb[:n],axis=1)[min(10,n//3):],70))

def trace(fields,seed,width,heat_only=False,straight=None):
    seed=np.asarray(seed,float);d=seed[1]-seed[0]; d/=np.linalg.norm(d);a0=math.atan2(d[1],d[0])
    templ=prepare_template(fields,seed,width)
    step=PARAMETERS['step']; count=max(2,int(np.linalg.norm(seed[1]-seed[0])/step)+1)
    pts=np.linspace(seed[0],seed[1],count).tolist()
    beam=[dict(points=pts,angles=[a0]*count,qualities=[.5]*count,ev=[None]*count,
       total=0.,weak=0.,turn=0.,reason=None,surface_steps=0)]
    if straight and straight.get('available'):
        anchor=np.asarray(straight['points_xy']);along=(anchor-seed[0])@d
        ids=np.flatnonzero(along>np.linalg.norm(seed[1]-seed[0])+step)[::2]
        if len(ids)>2:
            q=anchor[ids];angle=math.atan2(*np.asarray(straight['axis_direction_xy'])[::-1])
            e=evaluate(fields,q,np.full(len(q),angle),width,templ,heat_only)
            node=beam[0]
            for j,i in enumerate(ids):
                evidence={k:float(v[j]) for k,v in e.items()}
                evidence['supported']=bool(straight['supported'][i]);evidence['straight_anchor']=True
                node['points'].append(q[j].tolist());node['angles'].append(angle)
                node['qualities'].append(float(e['score'][j]));node['ev'].append(evidence)
                node['total']+=float(e['score'][j])
    beam[0]['surface_steps']=int((sample(fields['domain'],np.asarray(beam[0]['points']))>.5).sum())
    ends=[];h,w=fields['gray'].shape
    offsets=np.deg2rad(PARAMETERS['angle_changes_degrees'])
    for it in range(PARAMETERS['max_steps']):
        proposals=[]
        for node in beam:
            for da in offsets:
                # Weak response cannot justify an arbitrary turn into the
                # occluding guide rod. Coast along the last observed tangent.
                if node['weak']>width*1.5 and da!=0:continue
                a=node['angles'][-1]+da
                if abs(wrap(a-a0))>math.radians(PARAMETERS['max_turn_degrees']):continue
                # Fixed-view projection prior: curve opens down on both
                # cameras. Allow small opposite corrections, not an upward
                # hook around the guide rod. No contact point is assumed.
                if wrap(a-a0)*np.sign(d[0]) < -math.radians(PARAMETERS['maximum_upward_turn_degrees']):continue
                # A wound rope is monotonic across the view; a full U-turn is
                # not invented to collect extra heatmap response on old loops.
                if math.cos(a)*d[0]<.04:continue
                q=np.array(node['points'][-1])+step*np.array([math.cos(a),math.sin(a)])
                if min(q[0],q[1],w-1-q[0],h-1-q[1])<2:
                    if da==0:ends.append({**node,'reason':'image_or_model_roi_edge'})
                    continue
                proposals.append((node,q,a,da))
        if not proposals:break
        q=np.array([p[1] for p in proposals]);angles=np.array([p[2] for p in proposals])
        ev=evaluate(fields,q,angles,width,templ,heat_only)
        domains=sample(fields['domain'],q)>.5
        outside=sample(fields['outside_distance'],q)
        pool=[]
        for j,(node,point,a,da) in enumerate(proposals):
            # The free entry may be outside the spool. After a sustained
            # passage through its visible surface, a route leaving a modest
            # hull margin cannot be continued by network background texture.
            if node['surface_steps']*step>=4*width and outside[j]>max(2*width,12.):
                if da==0:ends.append({**node,'reason':'visible_surface_exit_identity_unknown'})
                continue
            supported=ev['raw'][j]>=PARAMETERS['minimum_raw'] and ev['heat'][j]>=PARAMETERS['minimum_heat']
            # Search may continue on a clear heat-profile ridge when one raw
            # side is clipped or has merged into the spool. This stays orange
            # unless BOTH local image and heat support are present; previously
            # the raw dual-edge stop gate cut off visibly continuous B ridges.
            traversable=supported or ev['heat'][j]>=PARAMETERS['traversable_heat']
            weak=0 if traversable else node['weak']+step
            score=float(ev['score'][j]-PARAMETERS['turn_penalty']*abs(da)/max(abs(offsets))-
                PARAMETERS['turn_change_penalty']*abs(da-node['turn'])/(2*max(abs(offsets))))
            e={k:float(v[j]) for k,v in ev.items()};e['supported']=bool(supported)
            e['traversable_heat_or_joint']=bool(traversable)
            new=dict(points=node['points']+[point.tolist()],angles=node['angles']+[a],
              qualities=node['qualities']+[score],ev=node['ev']+[e],total=node['total']+score,
              weak=weak,turn=float(da),reason=None,surface_steps=node['surface_steps']+int(domains[j]))
            if weak>=max(width*PARAMETERS['weak_limit_widths'],150.):
                new['reason']='long_weak_evidence';ends.append(new)
            else:pool.append(new)
        pool.sort(key=lambda n:n['total'],reverse=True)
        beam=[];keys=set()
        for n in pool:
            p=n['points'][-1];key=(round(p[0]/max(3,width*.3)),round(p[1]/max(3,width*.3)),round(n['angles'][-1]/.10))
            if key in keys:continue
            keys.add(key);beam.append(n)
            if len(beam)>=PARAMETERS['beam_width']:break
        if len(ends)>240:ends=sorted(ends,key=lambda n:rank(n,count,step),reverse=True)[:120]
        if not beam:break
    ends.extend({**n,'reason':'search_length_budget'} for n in beam)
    ends.sort(key=lambda n:rank(n,count,step),reverse=True)
    out=[]
    for n in ends:
        if all(path_distance(n,k)>width*.8 for k in out):out.append(n)
        if len(out)>=PARAMETERS['final_candidates']:break
    for n in out:
        n['rank_score']=rank(n,count,step);n['entry_points']=count
        n['heat_only_baseline']=heat_only
    return out

def rank(n,count,step):
    vals=n['qualities'][count:]
    if not vals:return -1.
    # Bound length incentive; evidence remains the main term. Long weak ends
    # are retained in the record but do not count as observed rope pixels.
    return float(.7*np.mean(vals)+.3*np.mean(vals[-40:])+.14*min(len(vals)*step/1000,1.))

def finalize_path(node,width,shape):
    # Remove unsupported trailing extrapolation, but keep internal gaps and
    # their weak status. A dark gap is not automatically an actual occluder.
    useful=[i for i,e in enumerate(node['ev']) if e is not None and
        (e['supported'] or e.get('traversable_heat_or_joint',False))]
    end=max(node['entry_points'],(useful[-1]+1 if useful else 0))
    p=np.array(node['points'][:end],np.float32)
    # Heat-baseline and fused pools may reference the same search node. Each
    # exported path owns its evidence; later entry-conflict marking must not
    # mutate another path's evidence without its matching support array.
    ev=[None if e is None else dict(e) for e in node['ev'][:end]];n=len(p)
    supported=np.array([False if e is None else e['supported'] for e in ev])
    # Supplied seed is a condition, not a fresh recognition success.
    condition=np.arange(n)<node['entry_points']
    # Smooth only inside a narrow tube about the proposed path. No ellipse
    # extrapolation or forced completion to a flange is performed.
    smooth=cv2.GaussianBlur(p.reshape(-1,1,2),(1,9),1.25).reshape(-1,2) if n>=9 else p.copy()
    smooth[:node['entry_points']]=p[:node['entry_points']];smooth[-1]=p[-1]
    shift=np.linalg.norm(smooth-p,axis=1);bad=shift>width*.20;smooth[bad]=p[bad]
    widths=np.array([width if e is None else e['width'] for e in ev],np.float32)
    widths=cv2.GaussianBlur(widths[:,None],(1,9),2).ravel() if n>=9 else widths
    original=smooth*2+np.array([0,1050])
    return dict(points_xy=original.tolist(),width_px=(widths*2).tolist(),
      supported=supported.tolist(),entry_condition=condition.tolist(),
      evidence=ev,rank_score=node['rank_score'],termination=node['reason'],
      length_px=float(np.linalg.norm(np.diff(original,axis=0),axis=1).sum()),
      physical_contact_confirmed=False,same_root_identity_confirmed=False,
      mask_is_optical_ribbon_estimate=True)

def diagnostic_field(fields,width):
    # Orientation bank shows candidate structure, including wrong old rings.
    # Absolute local DoG admits both dark and bright center responses.
    heat=fields['heat'];gray=fields['gray']
    hs=abs(cv2.GaussianBlur(heat,(0,0),1)-cv2.GaussianBlur(heat,(0,0),max(2,width*.7)))
    gs=abs(cv2.GaussianBlur(gray,(0,0),1)-cv2.GaussianBlur(gray,(0,0),max(2,width*.7)))
    return np.clip(.6*hs/.12+.4*gs/.08,0,1).astype(np.float32)

def prepend_free_extent(paths,back,seed,width,fields):
    """Recover visible free rope behind the unchanged short UNet condition."""
    if not back.get('available'):return
    d=seed[1]-seed[0];d/=np.linalg.norm(d)
    q=np.asarray(back['points_xy']);along=(q-seed[0])@d
    ids=np.flatnonzero(along < -PARAMETERS['step'])[::-1][::2]
    if len(ids)<2:return
    q=q[ids];angle=math.atan2(d[1],d[0]);templ=prepare_template(fields,seed,width)
    ev=evaluate(fields,q,np.full(len(q),angle),width,templ)
    points=(q*2+[0,1050]).tolist();supported=[bool(back['supported'][i]) for i in ids]
    evidence=[]
    for j,i in enumerate(ids):
        e={k:float(v[j]) for k,v in ev.items()};e.update(supported=supported[j],straight_anchor=True,reverse_free_extent=True);evidence.append(e)
    for p in paths:
        p['points_xy']=points+p['points_xy'];p['width_px']=(ev['width']*2).tolist()+p['width_px']
        p['supported']=supported+p['supported'];p['entry_condition']=[False]*len(q)+p['entry_condition']
        p['evidence']=[dict(e) for e in evidence]+p['evidence'];p['reverse_free_extent_points']=len(q)
        p['length_px']=float(np.linalg.norm(np.diff(np.array(p['points_xy']),axis=0),axis=1).sum())

def detect(gray,response,seed,width,merged,inferred,surface):
    fields=build_fields(gray,response,merged,inferred,surface)
    local_seed=(np.asarray(seed,float)-[0,1050])*.5;width=float(np.clip(width*.5,3,35))
    from straight_support import straight_support
    straight=straight_support(gray,response,local_seed,width)
    candidates=trace(fields,local_seed,width)
    if straight.get('available'):
        candidates+=trace(fields,local_seed,width,straight=straight)
    # Heatmap-only candidate search is an explicit comparator, not truth.
    baseline=trace(fields,local_seed,width,True)
    # Retain the independently generated heat-only routes in the same final
    # pool. The first pilot showed fusion could discard the correct straight
    # continuation while the heat-only route still followed it.
    pool=candidates+baseline
    for c in pool:
        q=np.asarray(c['points']);a=np.asarray(c['angles'])
        e=evaluate(fields,q,a,width,prepare_template(fields,local_seed,width))
        c['qualities']=e['score'].tolist()
        c['rank_score']=rank(c,c['entry_points'],PARAMETERS['step'])
        if straight.get('available'):
            # Penalize abandoning measured collinear pieces before their end;
            # weak gap points have zero anchor weight and prove no identity.
            anchors=np.asarray(straight['points_xy'])[np.asarray(straight['supported'])]
            if len(anchors):
                d=(local_seed[1]-local_seed[0]);d/=np.linalg.norm(d)
                aq=(anchors-local_seed[0])@d;pq=(q-local_seed[0])@d
                covered=(aq>=pq.min())&(aq<=pq.max())
                if covered.any():
                    order=np.argsort(pq);pred=np.c_[np.interp(aq[covered],pq[order],q[order,0]),np.interp(aq[covered],pq[order],q[order,1])]
                    mismatch=np.mean(np.clip(np.linalg.norm(pred-anchors[covered],axis=1)/width,0,2))
                    c['rank_score']-=.25*float(mismatch)
                c['rank_score']-=.20*float(np.mean(aq>pq.max()+width))
    pool.sort(key=lambda c:c['rank_score'],reverse=True)
    unique=[]
    for c in pool:
        if all(path_distance(c,o)>width*.8 for o in unique):unique.append(c)
        if len(unique)>=PARAMETERS['final_candidates']:break
    result=[finalize_path(c,width,gray.shape) for c in unique]
    base=[finalize_path(c,width,gray.shape) for c in baseline[:1]]
    back=straight_support(gray,response,local_seed[::-1].copy(),width)
    prepend_free_extent(result+base,back,local_seed,width,fields)
    for r in result:
        r['straight_support']=straight
        r['straight_support_coordinate_system']='half_size_model_crop_xy; original_xy=2*local_xy+[0,1050]'
    return result,base,diagnostic_field(fields,width)
