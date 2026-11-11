"""Current coil-side axis and column-dependent boundary queries for R20."""
import numpy as np
from common20 import plain,PITCH_PRIOR
from coil_edge_axis_v4 import measure as edge_pairs
from roi20 import masks
from tangent_column import roots_for_curve,estimate

def boundary_at(b,d):
    out=dict(valid=False,y=None,uncertainty_px=None,locally_observed=False,occluded=False,reasons=[])
    if not b.get('fit_valid') or not b.get('fit_coefficients_ascending'):
        out['reasons'].append('current_boundary_shape_unavailable');return out
    n=b['normalized_x'];u=(d-n['center'])/n['scale'];y=float(np.polynomial.polynomial.polyval(u,b['fit_coefficients_ascending']))
    pts=np.asarray(b.get('top_observed_xy',[]),float).reshape(-1,2)
    if len(pts)<8:out['reasons'].append('insufficient_current_boundary_anchors');return out
    xs=pts[:,0];near=float(np.min(abs(xs-d)));in_hull=bool(xs.min()<=d<=xs.max())
    occ=any(lo<=d<=hi for lo,hi in b.get('foreground_ranges',[]))
    local=near<=2*PITCH_PRIOR and not occ
    # Query the fitted coefficient ensemble at THIS d. Its spread and distance
    # from current visible anchors survive, even if many past frames exist.
    vals=[float(np.polynomial.polynomial.polyval(u,h['coef'])) for h in b.get('shape_state',[])]
    mad=float(b.get('fit_residual_MAD_px',3.))
    shape_spread=float(np.percentile(abs(np.array(vals)-np.median(vals)),90)) if len(vals)>=3 else 6.
    unc=max(3.,2*mad,shape_spread)+.025*max(0.,near-2*PITCH_PRIOR)
    if not in_hull:unc+=.01*min(abs(d-xs.min()),abs(d-xs.max()))
    if not local:unc=max(unc,6.)
    out.update(valid=True,y=y,uncertainty_px=float(unc),locally_observed=local,occluded=occ,inside_current_anchor_hull=in_hull,
        nearest_current_anchor_px=near,shape_ensemble_spread_px=shape_spread,
        semantics='current_visible_optical_coil_limit' if local else 'shape_estimate_anchored_elsewhere_in_current_frame',
        physical_contact_boundary_calibrated=False,sensitivity_is_not_confidence_interval=True)
    return out

def measure(image,r,groups):
    t=r['track'];out=dict(axis=None,coil_axis_trial=None,tangent=None,initial_d=None,calibration_far_proxy=None)
    if not t.get('valid'):return out
    if len(r['boundary'].get('top_curve_xy',[]))<30:
        out['reason']='current_boundary_domain_unavailable';return out
    abc=t['line_abc'];curves=(groups or {}).get('curves',[])
    tc=estimate(curves,abc)
    root=roots_for_curve(r['boundary'].get('top_curve_xy',[]),abc,PITCH_PRIOR) if len(r['boundary'].get('top_curve_xy',[]))>=30 else None
    d=tc['d'] if tc.get('valid') else root['d'] if root else float(np.median(np.asarray(r['boundary'].get('top_curve_xy',[[image.shape[1]/2,0]]))[:,0]))
    surface,_,_=masks(image,r)
    trial=edge_pairs(image,t,d,surface=surface);pts=np.asarray(t.get('line_points_xy',[]),float).reshape(-1,2)
    if trial['valid']:
        axis=dict(valid=True,line_abc=trial['line_abc'],points_xy=trial['points_xy'],width_px=trial['width_px'],
            uncertainty_px=max(3.,trial['sensitivity_px']),current_coil_side_measured=True,source='CE4_current_exposed_rope_or_SG5_overlap_checked',
            extrapolation_px=trial['extrapolation_px'],residual_q90_px=trial['residual_q90_px'])
    else:
        ext=max(0.,d-float(pts[:,0].max())) if len(pts) else 1500.
        err=float(np.percentile(abs(pts@np.asarray(abc[:2])+abc[2]),90)) if len(pts) else 10.
        axis=dict(valid=True,line_abc=abc,points_xy=pts,width_px=float((t.get('diagnostics',{}).get('seed_validation') or {}).get('width_px') or PITCH_PRIOR),
            uncertainty_px=max(4.,err)+.02*ext,current_coil_side_measured=False,source='SG5_current_upstream_axis_extrapolation',
            extrapolation_px=ext,residual_q90_px=err)
    tc2=estimate(curves,axis['line_abc']);bd=boundary_at(r['boundary'],float(tc2['d']) if tc2.get('valid') else d)
    dd=float(tc2['d']) if tc2.get('valid') else d;aa,bb,cc=axis['line_abc'];yy=-(aa*dd+cc)/bb
    far=(yy-bd['y']-axis['uncertainty_px']-bd['uncertainty_px'])/PITCH_PRIOR if bd['valid'] else None
    out.update(axis=axis,coil_axis_trial=trial,tangent=tc2,upstream_tangent=tc,initial_d=d,calibration_far_proxy=far,
        far_proxy_is_only_screening_not_F3_count=True,boundary_at_dynamic_tangent=bd)
    return plain(out)
