"""Current broad interface can supersede a mismatched historical shape.

Requires current yellow/texture interface, clean convex broad support and
spatial-block holdout stability across the entire calibrated column interval.
No historical samples are fabricated to satisfy the initialization counter.
"""
import numpy as np
from common20 import PITCH_PRIOR as P,plain

def recover(image,proposal,track,old,d_interval,frame):
    from boundary_bg9 import foreground,design,robust_fit,intervals
    check=dict(attempted=False,accepted=False)
    if old.get('fit_valid') or not proposal.get('valid'):return old,check
    if old.get('reasons')!=['outline_at_d_model_sensitive']:return old,check
    check['attempted']=True;h,w=image.shape[:2]
    pts=np.asarray(proposal['top_observed_xy']+proposal.get('rejected_top_observations_xy',[]),float)
    pts=pts[np.argsort(pts[:,0])];ranges,polys=foreground(image,proposal)
    abc=track.get('current_line_abc') if track.get('valid') else None
    keep=np.array([not any(lo<=x<=hi for lo,hi in ranges) for x in pts[:,0]])
    if abc is not None:keep &= abs(pts@np.asarray(abc)[:2]+abc[2])>=1.2*P
    raw=pts.copy();pts=pts[keep]
    if len(pts)<80 or np.ptp(pts[:,0])<30*P:return old,check
    X=design(pts[:,0],w);c,k=robust_fit(X,pts[:,1]);a=pts[k]
    if len(a)<80 or k.mean()<.8 or np.ptp(a[:,0])<.8*np.ptp(raw[:,0]):return old,check
    bins=np.floor((pts[:,0]-pts[:,0].min())/(np.ptp(pts[:,0])+1)*8).astype(int)
    ds=np.linspace(*d_interval,9);DG=design(ds,w);pred=DG@c
    if a[:,0].min()>ds.min()-P or a[:,0].max()<ds.max()+P:return old,check
    grid=np.arange(a[:,0].min(),a[:,0].max()+1,8);G=design(grid,w);u=G[:,1]
    residual=a[:,1]-design(a[:,0],w)@c;q90=float(np.quantile(abs(residual),.9))
    if np.min(2*c[2]+6*c[3]*u+12*c[4]*u*u)<=10 or q90>.20*P:return old,check
    alt=[]
    for group in np.unique(bins):
        sub=(bins!=group)&k
        if sub.sum()<60 or np.ptp(pts[sub,0])<.7*np.ptp(a[:,0]):continue
        cc,_=robust_fit(X[sub],pts[sub,1]);alt.append(DG@cc)
    sensitivity=float(np.max(abs(np.asarray(alt)-pred))) if alt else 1e6
    check.update(points=len(a),residual_q90_px=q90,heldout_blocks=len(alt),max_column_interval_sensitivity_px=sensitivity)
    if len(alt)<6 or sensitivity>.20*P:return old,check
    # Every requested column must have nearby clean measured interface anchors.
    if np.max(np.min(abs(ds[:,None]-a[:,0]),axis=1))>2*P:return old,check
    xy=np.c_[grid,G@c];ranges2,polys2=foreground(image,dict(top_curve_xy=xy.tolist()));ranges+=ranges2;polys+=polys2
    near=np.min(abs(grid[:,None]-a[:,0]),axis=1)<=1.5*P
    near &= np.array([not any(lo<=x<=hi for lo,hi in ranges) for x in grid])
    if abc is not None:near &= abs(xy@np.asarray(abc)[:2]+abc[2])>=1.2*P
    b=dict(old);d=sum(d_interval)/2;yd=float(design([d],w)[0]@c);mad=float(1.4826*np.median(abs(residual-np.median(residual))))
    b.update(version='BG11_current_broad_consensus_R20',fit_valid=True,reasons=[],edge_y=yd,
        edge_uncertainty_px=max(6.,2*mad,sensitivity),fit_residual_MAD_px=mad,
        fit_coefficients_ascending=c.tolist(),normalized_x=dict(center=.42*w,scale=.3*w),
        top_curve_xy=xy.tolist(),top_observed_xy=a.tolist(),shape_state=[dict(frame=frame,coef=c.tolist())],
        shape_prior_frames=[],shape_age_frames=0,shape_updated_current_frame=True,
        foreground_ranges=ranges,foreground_polygons=polys,current_anchor_span_px=float(np.ptp(a[:,0])),
        current_consensus_fraction=float(k.mean()),history_is_shape_prior_only=True,
        count_boundary_valid=False,physical_contact_boundary_calibrated=False,
        current_only_shape_recovery=check,previous_shape_failure=old['reasons'],
        d_edge_semantics='current_broad_convex_curve_with_spatial_holdout_validation')
    for key,flag in [('observed_segments',near),('inferred_segments',~near)]:
        b[key]=[xy[lo:hi].tolist() for lo,hi in intervals(flag) if hi-lo>=2]
    check['accepted']=True
    return plain(b),check
