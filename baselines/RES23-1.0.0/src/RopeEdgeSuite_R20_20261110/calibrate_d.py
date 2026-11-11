"""Audit optical C1 turn identifiability from current supported P1/H6 paths.

No transition interpolation, historical paths or fixed reference d enter fits.
The returned profile is a model-sensitivity interval, not a calibrated CI.
"""
import numpy as np
from common20 import *

def frame_fit(row):
    p1=row['result'];diag=p1['diagnostics'];checks=diag.get('whole_path_support_checks',{})
    p=np.asarray(p1.get('path_xy',[]),float).reshape(-1,2);st=np.asarray(diag.get('path_stage',[]));sp=np.asarray(diag.get('path_point_support',[]),bool)
    out=dict(frame=row['frame'],eligible=False,fit_available=False,reasons=[],source='current_P1_H6_supported_samples; fitted upstream curves may have correlated errors')
    if len(p)<10 or len(st)!=len(p) or len(sp)!=len(p):out['reasons'].append('path_unavailable');return out
    curve=p[sp&(st=='surface')];free=p[sp&(st=='free')]
    out.update(surface_points=len(curve),free_points=len(free),whole_path_flag=p1.get('whole_path_candidate_supported'),checks=checks)
    if not checks.get('genuine_H6_attached_candidate'):out['reasons'].append('no_image_eligible_H6_attachment')
    if checks.get('ambiguous',True):out['reasons'].append('ambiguous_H6_connection')
    if not checks.get('direction_selection_allowed'):out['reasons'].append('direction_unconfirmed')
    if len(curve)<10 or (len(curve) and np.ptp(curve[:,0])<3*PITCH_PRIOR):out['reasons'].append('curve_span_insufficient')
    if len(free)<15:out['reasons'].append('free_support_insufficient');return out
    if len(curve)<6:return out
    # Count geometry uses the straight segment on the coil side of the guide.
    free=free[free[:,0]>=free[:,0].max()-900]
    obs=np.vstack([free,curve]);obs=obs[np.argsort(obs[:,0])]
    # Spatial bins select actual supported coordinates, not synthetic gap points.
    keys=np.floor(obs[:,0]/12).astype(int);obs=np.array([obs[np.flatnonzero(keys==k)[len(np.flatnonzero(keys==k))//2]] for k in np.unique(keys)])
    x,y=obs.T;x0=float(np.median(free[:,0]));u=(x-x0)/1000
    lo=float(max(x.min()+5*PITCH_PRIOR,curve[:,0].min()-650));hi=float(curve[:,0].max()-2*PITCH_PRIOR)
    if hi<=lo:out['reasons'].append('turn_search_domain_missing');return out
    grid=np.arange(lo,hi+2,2.);fits=[];scores=[]
    for d in grid:
        X=np.c_[np.ones(len(x)),u,np.maximum(0.,x-d)**2/1e6];weights=np.ones(len(x))
        for _ in range(7):
            beta=np.linalg.lstsq(X*np.sqrt(weights[:,None]),y*np.sqrt(weights),rcond=None)[0]
            residual=y-X@beta;weights=np.minimum(1.,4./np.maximum(abs(residual),1e-6))
        e=abs(residual);score=float(np.mean(np.where(e<=4,e*e,8*e-16)))
        fits.append((beta,residual));scores.append(score)
    scores=np.asarray(scores);k=int(scores.argmin());d=float(grid[k]);beta,residual=fits[k]
    q=float(beta[2]/1e6);m=float(beta[1]/1000);b=float(beta[0]-m*x0);uerr=float(np.percentile(abs(residual),90))
    good=scores<=scores[k]+4.;interval=[float(grid[good].min()),float(grid[good].max())]
    closed=bool(not good[0] and not good[-1]);deviation=float(q*max(0.,x.max()-d)**2)
    X=np.c_[np.ones(len(x)),u];linear=X@np.linalg.lstsq(X,y,rcond=None)[0];e=abs(y-linear)
    linear_score=float(np.mean(np.where(e<=4,e*e,8*e-16)));gain=1-float(scores[k])/max(linear_score,1e-9)
    if q<=0:out['reasons'].append('nonpositive_curvature')
    if deviation<.35*PITCH_PRIOR or gain<.3:out['reasons'].append('straight_or_weak_bend')
    if uerr>.3*PITCH_PRIOR:out['reasons'].append('model_residual_excessive')
    if not closed:out['reasons'].append('d_profile_open')
    out.update(fit_available=True,d_candidate=d,d_sensitivity_interval=interval,q=q,line_abc=[-m,1.,-b],
        residual_q90_px=uerr,curved_deviation_px=deviation,curvature_gain=gain,profile_closed=closed,
        fit_points_xy=obs,profile_grid=grid,profile_scores=scores,fit_scores_min=float(scores[k]),
        search_domain=[lo,hi],eligible=not out['reasons'],d_is_optical_not_physical=True)
    return plain(out)

def main():
    out=[]
    for item in plan():
        rows=[frame_fit(read(f)) for f in sorted((ROOT/'probes'/item['id']).glob('p1_*.json'))]
        out.append(dict(id=item['id'],frames=rows))
        print(item['id'],[(r['frame'],round(r.get('d_candidate',0),1),r['reasons']) for r in rows],flush=True)
    dump(ROOT/'d_frame_audit_v1.json',dict(videos=out,calibration_locked=False))
if __name__=='__main__':main()
