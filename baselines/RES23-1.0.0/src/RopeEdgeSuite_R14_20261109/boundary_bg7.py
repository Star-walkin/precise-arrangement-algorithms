"""BG7: current unoccluded anchors + causal shape-constrained global fit.

Fitted gaps are explicitly model estimates, not measured pixels. No local
residual interpolation is used. Foreground and active-rope search bands are
excluded before fitting. Partial observations cannot update shape coefficients.
"""
import cv2,numpy as np
from common import D,PITCH,plain

PARAMETERS=dict(column_step=8,bright_width_pitch=.85,bright_reference_cap=200.,
    foreground_run_span_pitch=1.5,active_band_pitch=1.2,minimum_points=30,
    minimum_visible_span_pitch=18.,minimum_shape_span_fraction=.7,
    max_fit_residual_pitch=.5,maximum_d_sensitivity_pitch=.5,maximum_outline_sensitivity_pitch=2.,
    maximum_shape_age_frames=150,minimum_initial_shapes=3)

def intervals(a):
    e=np.diff(np.r_[False,np.asarray(a,bool),False].astype(np.int8))
    return list(zip(np.flatnonzero(e==1),np.flatnonzero(e==-1)))

def foreground(image,proposal,pitch=PITCH):
    curve=np.asarray(proposal.get('top_curve_xy',[]),float).reshape(-1,2)
    if len(curve)<3:return [],[]
    xs=np.arange(curve[0,0]+pitch,curve[-1,0]-pitch,8)
    if not len(xs):return [],[]
    y=np.interp(xs,curve[:,0],curve[:,1]);slope=np.gradient(y,xs)
    offsets=np.arange(-pitch,8*pitch,1.)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)
    vals=[]
    for dx in [-4,0,4]:
        X=np.broadcast_to((xs+dx)[:,None],(len(xs),len(offsets))).astype(np.float32)
        Y=(y[:,None]+offsets[None,:]+dx*slope[:,None]).astype(np.float32)
        vals.append(cv2.remap(gray,X,Y,cv2.INTER_LINEAR))
    profiles=np.median(vals,axis=0);rows=[]
    for x,edge,profile in zip(xs,y,profiles):
        low=profile[offsets>=4*pitch]
        thr=min(200.,max(100.,np.median(low)+4*np.median(abs(low-np.median(low)))+30))
        spans=[(lo,hi) for lo,hi in intervals((profile>thr)&(offsets>-.5*pitch)) if offsets[lo]<6*pitch]
        lo,hi=max(spans,key=lambda v:v[1]-v[0]) if spans else (0,0)
        rows.append(dict(x=float(x),width=float(hi-lo),broad=bool(hi-lo>.85*pitch),
            top=float(edge+offsets[lo]),bottom=float(edge+offsets[min(hi,len(offsets)-1)])))
    polygons=[];ranges=[]
    for lo,hi in intervals([r['broad'] for r in rows]):
        if (hi-lo)*8<1.5*pitch:continue
        q=rows[lo:hi];ranges.append([q[0]['x']-pitch,q[-1]['x']+pitch])
        polygons.append([[r['x'],r['top']-.6*pitch] for r in q]+[[r['x'],r['bottom']+.8*pitch] for r in reversed(q)])
    return ranges,polygons

def design(x,w):
    u=(np.asarray(x)-.42*w)/(.3*w)
    return np.c_[np.ones(len(u)),u,u*u,u**3,u**4]

def robust_fit(X,y,base=None):
    # Full shape when base is absent; current translation and tilt otherwise.
    n=5 if base is None else 2
    fixed=np.zeros(5) if base is None else np.r_[0.,0.,base[2:]]
    target=y-X@fixed;A=X[:,:n];weights=np.ones(len(y));coef=None
    for _ in range(10):
        coef=np.linalg.lstsq(A*np.sqrt(weights[:,None]),target*np.sqrt(weights),rcond=None)[0]
        residual=target-A@coef;scale=max(2.,1.4826*np.median(abs(residual-np.median(residual))))
        weights=np.minimum(1.,1.5*scale/np.maximum(abs(residual),1e-6))
    c=fixed.copy();c[:n]=coef
    keep=abs(y-X@c)<=max(7.,2.5*scale)
    if keep.sum()>=max(12,n+3):
        c[:n]=np.linalg.lstsq(A[keep],target[keep],rcond=None)[0]
    return c,keep

class Boundary:
    def __init__(self):self.shapes=[];self.last_shape_frame=None

    def step(self,image,proposal,track,i,*,forced_exclude=None):
        h,w=image.shape[:2];pitch=PITCH;raw=np.asarray(proposal.get('top_observed_xy',[])+proposal.get('rejected_top_observations_xy',[]),float).reshape(-1,2)
        out=dict(version='BG7',frame=i,fit_valid=False,count_boundary_valid=False,
            reasons=[],edge_y=None,edge_uncertainty_px=None,top_curve_xy=[],top_observed_xy=[],
            observed_segments=[],inferred_segments=[],parameters=PARAMETERS,
            physical_contact_boundary_calibrated=False,history_is_shape_prior_only=True)
        if len(raw)<30:out['reasons']=['current_boundary_candidates_insufficient'];return out
        raw=raw[np.argsort(raw[:,0])];ranges,polygons=foreground(image,proposal)
        exclude=np.array([any(lo<=x<=hi for lo,hi in ranges) for x in raw[:,0]])
        abc=track.get('current_line_abc') if track.get('valid') else None
        active_exclude=np.zeros(len(raw),bool)
        if abc is not None:
            active_exclude=np.abs(raw@np.asarray(abc)[:2]+abc[2])<1.2*pitch
            exclude|=active_exclude
        if forced_exclude:
            exclude|=np.array([any(lo<=x<=hi for lo,hi in forced_exclude) for x in raw[:,0]])
        pts=raw[~exclude];out.update(raw_candidates_xy=raw.tolist(),excluded_candidates_xy=raw[exclude].tolist(),
            foreground_ranges=ranges,foreground_polygons=polygons,active_excluded_points=int(active_exclude.sum()),
            current_clean_points=len(pts),shape_prior_frames=[s['frame'] for s in self.shapes])
        if len(pts)<30 or np.ptp(pts[:,0])<18*pitch:
            out['reasons']=['unoccluded_current_anchor_support_insufficient'];return out
        X=design(pts[:,0],w);start,end=raw[0,0],raw[-1,0]
        grid=np.unique(np.r_[np.arange(start,end+1,8),D]);G=design(grid,w);Xd=design([D],w)[0]
        clean_span=np.ptp(pts[:,0])/(end-start)
        full,fullkeep=robust_fit(X,pts[:,1]);res=pts[:,1]-X@full
        # Never learn a global shape from a one-sided/foreground-covered arc.
        bins=np.floor((pts[:,0]-start)/(end-start+1)*8).astype(int)
        spatial=np.unique(bins[fullkeep]);u=G[:,1]
        curvature=2*full[2]+6*full[3]*u+12*full[4]*u*u
        local_clean=pts[abs(pts[:,0]-D)<2.5*pitch]
        local_shape_support=bool(len(local_clean)>=6 and local_clean[:,0].min()<D-pitch and local_clean[:,0].max()>D+pitch
            and not any(lo-pitch<=D<=hi+pitch for lo,hi in ranges))
        shape_difference=0.
        if self.shapes:
            oldshape=np.median([s['coef'] for s in self.shapes],axis=0)
            diff=G@(full-oldshape);affine=np.linalg.lstsq(G[:,:2],diff,rcond=None)[0]
            shape_difference=float(np.max(abs(diff-G[:,:2]@affine)))
        enough_shape=bool(clean_span>=.70 and len(spatial)>=6 and np.all(curvature>10)
            and np.quantile(abs(res[fullkeep]),.9)<.35*pitch and np.mean(exclude)<.35
            and local_shape_support and shape_difference<.6*pitch)
        out['shape_observability']=dict(clean_span_fraction=float(clean_span),bins=len(spatial),
            min_curvature=float(curvature.min()),residual_q90=float(np.quantile(abs(res[fullkeep]),.9)),
            excluded_fraction=float(np.mean(exclude)),accepted=bool(enough_shape),
            local_shape_support=local_shape_support,shape_difference_px=shape_difference)
        if enough_shape:
            # The current frame is permitted; later frames are never consulted.
            self.shapes=(self.shapes+[dict(frame=i,coef=full.tolist())])[-15:]
            self.last_shape_frame=i
        out['shape_prior_frames']=[s['frame'] for s in self.shapes]
        out['shape_state']=[dict(s) for s in self.shapes]
        if len(self.shapes)<3:
            out['reasons']=['initial_shape_evidence_accumulating'];return out
        age=i-self.last_shape_frame
        base=np.median([s['coef'] for s in self.shapes],axis=0)
        c,keep=robust_fit(X,pts[:,1],base)
        accepted=pts[keep];res=accepted[:,1]-design(accepted[:,0],w)@c
        if len(accepted)<30:
            out['reasons']=['robust_anchor_support_insufficient'];return out
        # Sensitivity to alternative historical shapes and omitted spatial blocks.
        predictions=[]
        for s in self.shapes[::max(1,len(self.shapes)//5)]:
            cc,_=robust_fit(X,pts[:,1],np.asarray(s['coef']));predictions.append(float(Xd@cc))
        for group in np.unique(bins):
            subset=bins!=group
            if subset.sum()<30 or np.ptp(pts[subset,0])<18*pitch:continue
            cc,_=robust_fit(X[subset],pts[subset,1],base);predictions.append(float(Xd@cc))
        yd=float(Xd@c);mad=float(1.4826*np.median(abs(res-np.median(res))))
        uncertainty=max(3.,2*mad,max((abs(v-yd) for v in predictions),default=0.))
        near=np.min(abs(grid[:,None]-accepted[:,0][None,:]),axis=1)<=1.5*pitch
        near&=np.array([not any(lo<=x<=hi for lo,hi in ranges) for x in grid])
        if abc is not None:near&=abs(np.c_[grid,G@c]@np.asarray(abc)[:2]+abc[2])>=1.2*pitch
        xy=np.c_[grid,G@c]
        for key,flag in [('observed_segments',near),('inferred_segments',~near)]:
            out[key]=[xy[lo:hi].tolist() for lo,hi in intervals(flag) if hi-lo>=2]
        reasons=[]
        if age>PARAMETERS['maximum_shape_age_frames']:reasons.append('shape_prior_too_old')
        if np.quantile(abs(res),.9)>.5*pitch:reasons.append('current_anchor_fit_residual_excessive')
        if uncertainty>2*pitch:reasons.append('outline_at_d_model_sensitive')
        # Do not measure roots through a tool even when the outline can be inferred.
        occluded=any(lo-pitch<=D<=hi+pitch for lo,hi in ranges)
        observed_d=bool(np.min(abs(accepted[:,0]-D))<=1.5*pitch and not occluded)
        out.update(fit_valid=not reasons,reasons=reasons,edge_y=yd,edge_uncertainty_px=uncertainty,
            top_curve_xy=xy.tolist(),top_observed_xy=accepted.tolist(),fit_coefficients_ascending=c.tolist(),
            normalized_x=dict(center=.42*w,scale=.3*w),fit_residual_MAD_px=mad,
            shape_age_frames=age,shape_updated_current_frame=enough_shape,
            current_anchor_span_px=float(np.ptp(accepted[:,0])),d_locally_observed=observed_d,
            d_foreground_occluded=occluded,count_boundary_valid=bool(not reasons and not occluded and uncertainty<=.5*pitch),
            d_edge_semantics='current_global_fit_locally_supported' if observed_d else 'shape_constrained_estimate_from_other_visible_anchors',
            sensitivity_is_not_calibrated_confidence_interval=True)
        return plain(out)
