"""TC1: infer optical tangent column from measured groove-family geometry.

Projection preserves tangency of a straight line joining a smooth wound path.
This is a conditional geometric model; independent P1 bends/holdouts must test it.
No reference d, timestamp or hand-selected column is used to solve the root.
"""
import numpy as np
from common20 import plain,PITCH_PRIOR

def roots_for_curve(xy,abc,pitch):
    p=np.asarray(xy,float);x,y=p.T
    if len(x)<30 or np.ptp(x)<16*pitch:return None
    center=float(np.median(x));scale=float(np.ptp(x)/2);u=(x-center)/scale
    keep=np.ones(len(x),bool)
    for _ in range(4):
        c=np.polyfit(u[keep],y[keep],4);err=y-np.polyval(c,u);mad=1.4826*np.median(abs(err[keep]-np.median(err[keep])));keep=abs(err)<max(3.,3*mad)
    if keep.sum()<.7*len(x):return None
    slope=-abc[0]/abc[1];der=np.polyder(c);der[-1]-=slope*scale
    candidates=[]
    for v in np.roots(der):
        if abs(v.imag)>.00001:continue
        v=float(v.real);d=v*scale+center
        if not x.min()+2*pitch<d<x.max()-2*pitch:continue
        curvature=float(np.polyval(np.polyder(c,2),v)/scale**2)
        if curvature<=1e-5:continue
        yy=float(np.polyval(c,v));liney=float(-(abc[0]*d+abc[2])/abc[1])
        candidates.append(dict(d=d,y=yy,gap=liney-yy,curvature=curvature))
    if len(candidates)!=1:return None
    result=candidates[0];result.update(polynomial=c,center=center,scale=scale,residual_q90_px=float(np.percentile(abs(err[keep]),90)),domain=[float(x.min()),float(x.max())])
    return result

def estimate(curves,abc,pitch=PITCH_PRIOR):
    out=dict(valid=False,reasons=[],d=None,interval=None,method='TC1_F3_family_tangent_to_current_coil_side_line',physical_d_confirmed=False)
    if abc is None:out['reasons'].append('current_coil_side_line_missing');return out
    candidates=[]
    for c in curves:
        fit=roots_for_curve(c['xy'],abc,pitch)
        if fit is None or fit['residual_q90_px']>.25*pitch:continue
        fit['boundary_id']=c['boundary_id'];candidates.append(fit)
    near=[c for c in candidates if abs(c['gap'])<=4*pitch]
    near=sorted(near,key=lambda c:abs(c['gap']))[:8]
    out['candidate_curves']=plain(near)
    if len(near)<3:out['reasons'].append('insufficient_tangent_neighbor_curves');return out
    ds=np.array([c['d'] for c in near]);d=float(np.median(ds));mad=float(1.4826*np.median(abs(ds-d)))
    # Model spread and optical line-angle sensitivity both survive pooling.
    angle_sensitivity=float(np.median([.002/c['curvature'] for c in near]))
    half=max(12.,2*mad,angle_sensitivity)
    if half>100:out['reasons'].append('tangent_column_spread_excessive')
    if max(ds)-min(ds)>180:out['reasons'].append('groove_family_inconsistent')
    out.update(valid=not out['reasons'],d=d,interval=[d-half,d+half],neighbor_count=len(near),curve_d_mad_px=mad,
        angle_sensitivity_assumption_rad=.002,uncertainty_kind='sensitivity_range_not_calibrated_confidence_interval',
        requires_independent_bend_or_temporal_holdout_validation=True)
    return plain(out)

if __name__=='__main__':
    from common20 import *
    id='dual_20260923_113745__left';i=150;r=record(id,i);g=read(ROOT/'probes'/id/f'groups_{i:06d}.json')
    curves=[dict(c,xy=np.asarray(c['xy'])*2) for c in g['curves']]
    result=estimate(curves,r['track']['line_abc']);dump(ROOT/'probes'/id/'tangent_000150.json',result)
    print({k:v for k,v in result.items() if k!='candidate_curves'})
