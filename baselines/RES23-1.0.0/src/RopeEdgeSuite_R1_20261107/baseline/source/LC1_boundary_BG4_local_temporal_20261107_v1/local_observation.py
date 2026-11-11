"""Local review of existing yellow-point classes; no invented groove rows."""
import math
import numpy as np
from scipy.interpolate import BSpline
from bg4_common import *

PARAMETERS=dict(local_radius_pitch=8.,minimum_points=12,minimum_points_each_side=3,
    minimum_span_pitch=4.,residual_q90_pitch=.22,minimum_local_support=.35,
    supported_bracket_pitch=6.,inferred_bracket_pitch=4.,minimum_separation_pitch=.45,
    crossing_residual_rejection_pitch=.35,crossing_residual_weak_pitch=.22,
    pitch_min_intervals=3,pitch_relative_mad=.12)

def restore_shape(diag):
    k=np.asarray(diag['knots_processing']);s=BSpline(k,diag['coefficients'],3)
    t=BSpline(k,diag['stretch_coefficients'],3)
    def f(x):return s(np.clip(x,k[3],k[-4]))
    f.scale=lambda x:np.clip(1+t(np.clip(x,k[3],k[-4]))-diag['stretch_anchor'],.65,1.5)
    return f

def recover_classes(b,data,d,edge,active):
    p=float(data['pitch']);pts=data['points'];shape=restore_shape(b['diagnostics']['shared_shape'])
    curves=[];logs=[];owned=set()
    for c in b['curves']:
        q=dict(c);q['xy']=np.asarray(c['xy'])*SCALE;q['recovered_local']=False
        curves.append(q);owned.update(c['point_ids'])
    byid={int(v[0]):i for i,v in enumerate(pts)}
    for k,c in enumerate(b['diagnostics']['rejections']):
        dom=c.get('domain',[1,0]);yy=c.get('y_at_d',-1)
        if not dom[0]<=d<=dom[1] or not edge-2*p<=yy<=max(edge,active)+3*p:continue
        log=dict(source_rejection=k,old_reasons=c['reasons'],recovered=False)
        ids=np.array([byid[j] for j in c.get('point_ids',[]) if j not in owned],int)
        ids=ids[np.abs(pts[ids,1]-d)<=PARAMETERS['local_radius_pitch']*p]
        if len(ids)<12 or np.ptp(pts[ids,1])<4*p:
            log['reason']='insufficient_local_points';logs.append(log);continue
        row=Row(0,shape,p);row.fit(pts,ids)
        ids=ids[np.abs(pts[ids,2]-row.predict(pts[ids,1]))<=.30*p]
        if len(ids)<12:log['reason']='insufficient_local_inliers';logs.append(log);continue
        row.fit(pts,ids)
        if row.stats['residual_q90_pitch']>.22:
            log['reason']='local_residual';logs.append(log);continue
        xs=np.arange(math.ceil(pts[ids,1].min()),math.floor(pts[ids,1].max())+1)
        xy=np.c_[xs,row.predict(xs)];valid=sample(data['model_envelope'],xy)>254
        # Keep the single admissible interval containing d. Far-field errors
        # cannot veto it, but no invalid sample or crossed neighbor is bridged.
        for q in curves:
            z=q['xy'];over=(xs>=z[0,0])&(xs<=z[-1,0])
            if not np.any(over):continue
            delta=xy[over,1]-np.interp(xs[over],z[:,0],z[:,1])
            sign=1 if np.median(delta)>0 else -1
            valid[over]&=sign*delta>=.45*p*shape.scale(xs[over])
        leftidx=int(np.searchsorted(xs,d,side='right')-1);rightidx=leftidx+1
        if leftidx<0 or rightidx>=len(xs) or not np.all(valid[leftidx:rightidx+1]):
            log['reason']='local_envelope_or_order_failure';logs.append(log);continue
        lo=leftidx;hi=rightidx
        while lo>0 and valid[lo-1]:lo-=1
        while hi+1<len(xs) and valid[hi+1]:hi+=1
        kept=ids[(pts[ids,1]>=xs[lo])&(pts[ids,1]<=xs[hi])]
        if len(kept)<12 or np.sum(pts[kept,1]<d)<3 or np.sum(pts[kept,1]>d)<3 or np.ptp(pts[kept,1])<4*p:
            log['reason']='insufficient_bilateral_retained_points';logs.append(log);continue
        take=(xs>=pts[kept,1].min())&(xs<=pts[kept,1].max());xy=xy[take]
        inside,unknown,support=supported_path(data,xy)
        near=np.abs(xy[:,0]-d)<=2*p
        if not near.any() or np.mean(support[near])<.35:
            log['reason']='weak_local_image_support';logs.append(log);continue
        bid=f'BG4_r{k:03d}';pointids=pts[kept,0].astype(int).tolist();owned.update(pointids)
        curves.append(dict(boundary_id=bid,xy=xy,point_states=np.where(support,'observed','inferred'),
            point_ids=pointids,fit_point_ids=pts[ids,0].astype(int),model=row.model,
            fit_statistics=row.stats,support_fraction=float(np.mean(support)),
            recovered_local=True,source_rejection=k,local_support_fraction=float(np.mean(support[near]))))
        log.update(recovered=True,boundary_id=bid,domain=xy[[0,-1],0].tolist(),point_count=len(pointids))
        logs.append(log)
    return curves,logs

def local_crossings(curves,points,data,d):
    p=float(data['pitch']);byid={int(pt[0]):pt for pt in points};out=[];rejected=[]
    for c in curves:
        xy=np.asarray(c['xy'])
        if not xy[0,0]<=d<=xy[-1,0]:continue
        y=float(np.interp(d,xy[:,0],xy[:,1]));obs=np.asarray([byid[i][1:3] for i in c['point_ids']])
        near=obs[np.abs(obs[:,0]-d)<=8*p];left=near[near[:,0]<=d];right=near[near[:,0]>=d]
        inside,unknown,support=supported_path(data,np.array([[d,y]]));why=[];gap=None;res=None
        if len(left)<3 or len(right)<3:why.append('insufficient_bilateral_points')
        else:
            gap=float(right[:,0].min()-left[:,0].max())
            if gap>(6 if support[0] else 4)*p:why.append('local_point_gap_too_long')
            res=float(np.percentile(np.abs(near[:,1]-np.interp(near[:,0],xy[:,0],xy[:,1])),90))/p
            if res>.35:why.append('local_residual_too_large')
        if not inside[0] or unknown[0]:why.append('count_intersection_unknown_or_excluded')
        low=bool(c['recovered_local'] or not support[0] or gap is None or gap>3*p or res is None or res>.22)
        rec=dict(boundary_id=c['boundary_id'],track_id=c['boundary_id'],y=y,score=c['support_fraction'],
            status='image_supported_fitted_curve' if support[0] else 'point_bracketed_model_interpolation',
            point_bracket_gap_px=gap,local_residual_pitch=res,lower_confidence=low)
        if why:rejected.append(dict(**rec,reasons=why))
        else:out.append(rec)
    out.sort(key=lambda c:c['y'])
    return out,rejected

def analyze(i,pixelsha,motion):
    path=BG3/'full_001'/'frames'/f'{i:06d}.json';old=read(path)
    assert pixelsha==old['source_pixel_sha256']
    reg=old['region'];s1=old['s1'];edge=reg.get('edge_y');active=old['intersection_y_px']
    r=dict(version='BG4-local-temporal-v1',frame_index=i,nominal_time_s=i/10,
        source_pixel_sha256=pixelsha,baseline_record_sha256=sha(path),fixed_d_original_px=D,
        intersection_y_px=active,region_gate_passed=old['region_gate_passed'],
        region={k:reg.get(k) for k in ('valid','reasons','edge_y','edge_uncertainty_px','top_curve_xy','top_observed_xy')},
        s1={k:s1.get(k) for k in ('valid','current_line_abc','line_points_xy','confidence')},
        masks=old['masks'],camera_motion=motion,reasons=list(old['reasons']),BG4=None,
        observation=None,baseline_count_valid=old['count_valid'],baseline_display_N=old['display_N'])
    if not old['region_gate_passed'] or old['BG3'] is None:return r
    cache=BG3/'cache'/f'{i:06d}.npz'
    with np.load(cache,allow_pickle=False) as f:data={k:f[k] for k in f.files}
    assert str(data['source_pixel_sha256'])==pixelsha
    assert str(data['old_record_sha256'])==sha(BASE/'frames'/f'{i:06d}.json')
    assert str(data['extraction_code_sha256'])==sha(BG3/'inputs.py')
    assert np.array_equal(data['points'],np.asarray(old['BG3']['raw_points']))
    r['cache_sha256']=sha(cache)
    data['model_envelope']=model_envelope(data['surface'],np.asarray(reg['top_curve_xy'])*SCALE)
    d=D*SCALE;p=float(data['pitch']);e=edge*SCALE;a=active*SCALE if active is not None else e
    curves,logs=recover_classes(old['BG3'],data,d,e,a)
    cross,rejected=local_crossings(curves,data['points'],data,d)
    count,reasons,cd=SurfaceObserver(data['surface'],p)._count(d,None if active is None else a,cross,
        dict(edge_y=e,edge_current_frame_confirmed=True),data['exclusion'])
    # A rejected remote class matters only if it represents a genuinely missing
    # local row in the actual measurement interval; padding alone is not a veto.
    recovered={q['source_rejection'] for q in logs if q['recovered']}
    unresolved=[]
    ys=np.array([q['y'] for q in cross])
    possible=[dict(q) for q in rejected]
    for k,q in enumerate(old['BG3']['diagnostics']['rejections']):
        if k not in recovered and q.get('domain',[1,0])[0]<=d<=q.get('domain',[1,0])[1]:
            possible.append(dict(y=q.get('y_at_d',-1),reasons=q['reasons']))
    for q in possible:
        if e<q['y']<a and (not len(ys) or np.min(np.abs(ys-q['y']))>.35*p):unresolved.append(q)
    if unresolved:reasons.append('unresolved_local_class_in_actual_interval')
    if not s1['valid']:reasons.append('S1_not_valid')
    reasons=list(dict.fromkeys(reasons));phase_ok=bool(count['count_valid'] and not reasons)
    lp=float(count['local_pitch_px']);geom=count.get('continuous_pitch_distance_candidate')
    plausible=np.diff(ys[(ys>=e-p)&(ys<=max(a,e)+5*p)])
    plausible=plausible[(plausible>=.65*p)&(plausible<=1.45*p)]
    cv=float(np.median(np.abs(plausible-np.median(plausible)))/np.median(plausible)) if len(plausible) else 1.
    hard=any(z in reasons for z in ('active_y_above_edge','S1_not_valid','sub_pitch_competing_grooves_after_deduplication'))
    # Geometry fallback tolerates missing grooves, not unknown surface coverage
    # or competing sub-pitch rows. It is a low-weight observation, never a new lock.
    if any('unknown' in z or 'outside' in z for z in reasons):hard=True
    geometry_ok=bool(not hard and active is not None and a>=e and len(plausible)>=3 and cv<=.12 and geom is not None)
    sigma=max(.18,min(.75,math.hypot(float(reg.get('edge_uncertainty_px') or 6)/(lp/SCALE),.12)))
    relevant=[q for q in cross if e-p<=q['y']<=a+p]
    weak=any(q['lower_confidence'] for q in relevant)
    obs=None
    if phase_ok:
        obs=dict(kind='estimated' if weak else 'measured',value=count['continuous_pitch_distance'],sigma=sigma*(1.25 if weak else 1),
            phase_anchor=True,geometry_value=geom,pitch_px=lp/SCALE,pitch_intervals=len(plausible),pitch_relative_mad=cv)
    elif geometry_ok:
        obs=dict(kind='estimated',value=geom,sigma=max(.45,sigma*1.5),phase_anchor=False,
            geometry_value=geom,pitch_px=lp/SCALE,pitch_intervals=len(plausible),pitch_relative_mad=cv)
    for c in curves:c['xy']=np.asarray(c['xy'])/SCALE
    for c in cross:c['y']/=SCALE
    r.update(reasons=reasons,observation=obs,BG4=dict(curves=curves,crossings=cross,
        local_recovery=logs,crossing_rejections=rejected,unresolved_local_classes=unresolved,
        phase_valid=phase_ok,phase_value=count['continuous_pitch_distance'] if phase_ok else None,
        geometry_candidate=geom,local_pitch_px=lp/SCALE,count_diagnostics=cd,
        raw_point_count=len(data['points']),local_pitch_intervals=len(plausible),pitch_relative_mad=cv))
    return r
