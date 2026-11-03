"""Finite curve intersections and evidence-limited free projection permission.

All geometry is crop-half xy. Crossings are geometric observations, not proof
of contact. No branch may hide an illegal bridge by changing its stage label.
"""
import copy
import numpy as np

PARAMETERS=dict(intersection_epsilon_half_px=1.e-6,
    free_crossing_continuous_original_px=48.,free_crossing_window_original_px=96.,
    stop_before_unlicensed_crossing_widths=1.,
    original_image_support_required_at_crossing=True,
    unknown_geometry_is_not_no_crossing=True)

def arc(p):return np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]

def crossings(path,tracks):
    """Exact piecewise-linear intersections for monotone-x finite curves.

    The union of both polylines' x knots prevents skipping a crossing between
    coarse samples. Touches and collinear overlaps are separate events.
    """
    p=np.asarray(path,float).reshape(-1,2)
    if len(p)<2:return []
    ds=arc(p);eps=PARAMETERS['intersection_epsilon_half_px']
    dx=np.diff(p[:,0]);nz=dx[abs(dx)>eps]
    if not len(nz) or (nz.min()<0<nz.max()):
        return [dict(kind='nonmonotone_path_unknown',track_id=None,progress_half_px=0.,xy=p[0].tolist())]
    forward=nz[0]>0;ix=np.arange(len(p)) if forward else np.arange(len(p)-1,-1,-1)
    q=p[ix];ss=ds[ix]
    # Consecutive duplicate points are harmless. A true vertical segment is
    # conservatively unknown rather than silently untested.
    keep=np.r_[True,np.linalg.norm(np.diff(q,axis=0),axis=1)>eps];q=q[keep];ss=ss[keep]
    if np.any(np.diff(q[:,0])<=eps):
        return [dict(kind='vertical_path_unknown',track_id=None,progress_half_px=0.,xy=p[0].tolist())]
    found=[];pymin=float(p[:,1].min());pymax=float(p[:,1].max())
    for tr in tracks:
        t=np.asarray(tr['points_xy'],float).reshape(-1,2)
        if len(t)<2:continue
        if pymin>float(t[:,1].max())+eps or pymax<float(t[:,1].min())-eps:continue
        if t[0,0]>t[-1,0]:t=t[::-1]
        if np.any(np.diff(t[:,0])<=eps):
            found.append(dict(kind='nonmonotone_track_unknown',track_id=tr['track_id'],progress_half_px=0.,xy=p[0].tolist()));continue
        lo=max(q[0,0],t[0,0]);hi=min(q[-1,0],t[-1,0])
        if hi<lo:continue
        xs=np.unique(np.r_[lo,hi,q[(q[:,0]>lo)&(q[:,0]<hi),0],t[(t[:,0]>lo)&(t[:,0]<hi),0]])
        yp=np.interp(xs,q[:,0],q[:,1]);yt=np.interp(xs,t[:,0],t[:,1]);delta=yp-yt
        zero=abs(delta)<=eps
        events=[]
        changes=np.diff(np.r_[False,zero,False].astype(np.int8))
        for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
            xa=float(xs[a]);xb=float(xs[b-1]);events.append((xa,xb,'overlap' if xb-xa>eps else 'touch'))
        for k in np.flatnonzero((delta[:-1]*delta[1:]<0)&~zero[:-1]&~zero[1:]):
            x=float(xs[k]-delta[k]*(xs[k+1]-xs[k])/(delta[k+1]-delta[k]));events.append((x,x,'cross'))
        for xa,xb,kind in events:
            sa=float(np.interp(xa,q[:,0],ss));sb=float(np.interp(xb,q[:,0],ss))
            start,end=sorted((sa,sb));x=xa if sa<=sb else xb
            found.append(dict(kind=kind,track_id=tr['track_id'],boundary_ids=tr.get('boundary_ids',[]),
                progress_half_px=start,end_progress_half_px=end,xy=[x,float(np.interp(x,t[:,0],t[:,1]))],
                end_xy=[xb if sa<=sb else xa,float(np.interp(xb if sa<=sb else xa,t[:,0],t[:,1]))]))
    found.sort(key=lambda e:(e['progress_half_px'],str(e['track_id']),e['kind']))
    return found

def _license_event(free,event):
    p=np.asarray(free['points_xy'],float);s=2*arc(p);at=2*event['progress_half_px']
    joint=np.asarray(free['supported'],bool);a7=np.asarray(free.get('raw_a7_supported',np.zeros(len(p))),bool)
    strong=(joint|a7)&(np.asarray(free['point_status'])!='entry_condition')
    k=int(np.clip(np.searchsorted(s,at,side='right')-1,0,len(s)-2))
    at_supported=bool(strong[k] and strong[k+1])
    span=0.
    if at_supported:
        a=k;b=k+1
        while a>0 and strong[a-1]:a-=1
        while b+1<len(s) and strong[b+1]:b+=1
        half=PARAMETERS['free_crossing_window_original_px']/2
        span=max(0.,min(s[b],at+half)-max(s[a],at-half))
    okay=at_supported and span>=PARAMETERS['free_crossing_continuous_original_px']-1.e-6 and event['kind'] in ('cross','touch')
    return dict(permitted_projected_free_crossing=bool(okay),
        original_image_supported_interval_contains_crossing=at_supported,
        continuous_support_original_px=float(span),support_source='original_and_B_or_strict_A7_original',
        physical_free_state_or_contact_verified=False)

def prepare_task_free(free,tracks):
    """Keep the full observation immutable; bound an uncertain crossing tail."""
    f=copy.deepcopy(free);p=np.asarray(f.get('points_xy',[]),float).reshape(-1,2)
    if len(p)<2:return f,dict(events=[],trimmed=False,full_observation_length_original_px=0.)
    events=crossings(p,tracks)
    for e in events:e.update(_license_event(free,e))
    bad=[e for e in events if not e['permitted_projected_free_crossing']]
    s=arc(p);stop=float(s[-1]);reason=None
    if bad:
        stop=max(0.,bad[0]['progress_half_px']-PARAMETERS['stop_before_unlicensed_crossing_widths']*max(3.,float(f['width_half_px'])))
        reason='first_unlicensed_center_crossing_not_a_physical_contact_label'
    last=int(np.searchsorted(s,stop,side='right')-1);last=max(0,min(last,len(p)-1))
    # Do not invent a minimum valid path when the first conflict is at entry.
    n=last+1;trim=n<len(p)
    if trim:
        aligned=('points_xy','supported','protect_support','point_status','raw_a7_supported','observed_straight_support','B_conflict','center','side_drop','heat_profile_supported')
        for key in aligned:
            if key in f:f[key]=f[key][:n]
        f['segment_status']=f.get('segment_status',[])[:max(0,n-1)]
        if 'evidence' in f:f['evidence']={k:v[:n] if isinstance(v,list) and len(v)==len(p) else v for k,v in f['evidence'].items()}
        f['continuation_starts']=[]
        for key in ('visible_runs','gap_hypotheses'):
            clipped=[]
            for ri,r in enumerate(f.get(key,[])):
                if r['start_index']>=n:continue
                a=int(r['start_index']);b=min(int(r['end_index']),last)
                nr=dict(start_index=a,end_index=b,span_original_px=float(2*(s[b]-s[a])),
                    scope='task_prefix_span_between_retained_nodes',
                    source_full_observation_field=key,source_full_observation_index=ri)
                if key=='gap_hypotheses':
                    nr.update(length_original_px=nr['span_original_px'],kind=r.get('kind','unknown_gap'),
                        supported=False,physical_occluder_confirmed=False,filled_as_measurement=False)
                clipped.append(nr)
            f[key]=clipped
        f['task_geometry_trimmed']=True;f['reason']=reason
        f['diagnostics']=dict(f.get('diagnostics',{}),scope='full_free_observation_detector_diagnostics_not_task_prefix')
        f['alternative_axes_scope']='untrimmed_full_observation_diagnostic_only_not_connected'
        # Stored summary scalars must describe the task prefix, not silently
        # retain the longer full-observation values. The latter remain intact
        # in full_free_observation; this does not alter ranking or coordinates.
        ds=2*np.diff(s[:n]);post=np.asarray(f['point_status'])!='entry_condition'
        eligible=post[:-1]&post[1:]
        heat=np.asarray(f['protect_support'],bool);joint=np.asarray(f['supported'],bool)
        raw=np.asarray(f.get('raw_a7_supported',np.zeros(n)),bool)
        observed=heat|raw
        measure=lambda q:float(ds[eligible&q[:-1]&q[1:]].sum())
        hl=measure(heat);jl=measure(joint);ol=measure(observed)
        f.update(visible_straight_length_original_px=hl,joint_straight_length_original_px=jl,
            heat_only_straight_length_original_px=hl-jl,observed_straight_length_original_px=ol,
            raw_a7_straight_length_original_px=measure(raw),raw_a7_only_length_original_px=measure(raw&~heat),
            observed_added_length_original_px=ol-hl,
            scalar_length_scope='task_prefix; full detector diagnostics in full_free_observation')
    retained=[e for e in events if e['progress_half_px']<=s[last]+1.e-7]
    receipt=dict(events=events,retained_prefix_events=retained,trimmed=trim,
        full_observation_length_original_px=float(2*s[-1]),task_prefix_length_original_px=float(2*s[last]),
        task_prefix_last_index=last,unresolved_tail_original_px=float(2*(s[-1]-s[last])),
        tail_state='unresolved_surface_projection' if trim else 'free_projection_not_attachment_verified',
        physical_contact_inferred_from_mask=False,reason=reason,parameters=PARAMETERS)
    f['task_topology']=receipt
    return f,receipt

def check_attachment(bridge,tail,tracks,target_id):
    """No other-rail contact and a single terminal target capture."""
    bridge=np.asarray(bridge,float);tail=np.asarray(tail,float)
    joined=np.vstack([bridge,tail[1:]])
    other=[t for t in tracks if t['track_id']!=target_id]
    blocked=crossings(joined,other)
    target=[t for t in tracks if t['track_id']==target_id]
    own=crossings(bridge,target)
    eps=PARAMETERS['intersection_epsilon_half_px'];total=arc(bridge)[-1]
    early=[]
    for e in own:
        if e['progress_half_px']>=total-eps*8:continue
        # A truly coincident suffix is already captured; it must never leave.
        bs=arc(bridge);at=e['progress_half_px']
        first=np.array([np.interp(at,bs,bridge[:,j]) for j in range(2)])
        suffix=np.vstack([first,bridge[bs>at+eps]])
        t=np.asarray(target[0]['points_xy'],float) if target else np.empty((0,2))
        if len(t) and t[0,0]>t[-1,0]:t=t[::-1]
        if len(suffix)>1 and suffix[0,0]>suffix[-1,0]:suffix=suffix[::-1]
        coincident=False
        if len(t) and len(suffix)>1 and suffix[0,0]>=t[0,0]-eps and suffix[-1,0]<=t[-1,0]+eps:
            knots=np.unique(np.r_[suffix[:,0],t[(t[:,0]>suffix[0,0])&(t[:,0]<suffix[-1,0]),0]])
            coincident=bool(np.all(abs(np.interp(knots,suffix[:,0],suffix[:,1])-np.interp(knots,t[:,0],t[:,1]))<=eps))
        if not coincident:early.append(e)
    okay=not blocked and not early and bool(own)
    return dict(allowed=bool(okay),other_track_events=blocked,own_track_capture_events=own,
        premature_target_events=early,checked_from_bridge_start=True,
        stage_labels_used_to_skip_geometry=False,target_track_id=target_id,
        source_gaps_are_not_bridged_here=True,identity_verified=False)
