"""Offline three-regime orientation check around a completed optical event.

It identifies a rope-pose transition, which is compatible with changing layer
but cannot certify a physical layer change from monocular projection alone.
The caller's human reference file is never read here.
"""
import numpy as np

def estimate(rows,event):
    start=event['start_s'];end=event.get('departure_s')
    out=dict(available=False,physical_layer_change_confirmed=False,method='robust_three_regime_current_rope_orientation',uses_future_frames=True)
    if end is None or end-start<2:return out
    points=np.array([[r['time_s'],r['axis_slope']] for r in rows if start-2<=r['time_s']<=end+2
        and r.get('axis_slope') is not None and r.get('current_coil_side_axis') and not r.get('boundary_occluded')],float)
    if len(points)<30:return out
    t,y=points.T;noise=max(.001,1.4826*np.median(abs(np.diff(y)-np.median(np.diff(y))))/np.sqrt(2));delta=2*noise
    def loss(res):
        a=abs(res);return np.sum(np.where(a<=delta,.5*a*a,delta*(a-.5*delta)))
    baseline=loss(y-np.median(y));candidates=[]
    for b1 in t[(t>=start-.7)&(t<=start+2)]:
        for b2 in t[(t>=end-2)&(t<=end+.7)]:
            masks=[t<b1,(t>=b1)&(t<b2),t>=b2]
            if any(np.sum(m)<8 for m in masks) or b2-b1<1:continue
            means=[float(np.median(y[m])) for m in masks]
            a,b,c=means
            if (b-a)*(b-c)<=0 or min(abs(b-a),abs(b-c))<max(.004,2*noise):continue
            score=sum(loss(y[m]-mu) for m,mu in zip(masks,means));candidates.append((float(score),float(b1),float(b2),means))
    if not candidates:return out
    best=min(candidates,key=lambda x:x[0]);gain=1-best[0]/max(baseline,1e-12)
    if gain<.25:return out
    close=[c for c in candidates if c[0]<=best[0]*1.05+1e-12]
    out.update(available=True,start_s=best[1],end_s=best[2],start_sensitivity_range_s=[min(c[1] for c in close),max(c[1] for c in close)],
        end_sensitivity_range_s=[min(c[2] for c in close),max(c[2] for c in close)],regime_slopes=best[3],noise=noise,loss_reduction=gain,
        interpretation='current_rope_pose_transition_compatible_with_layer_transition_but_not_confirmation',sensitivity_range_is_not_confidence_interval=True)
    return out
