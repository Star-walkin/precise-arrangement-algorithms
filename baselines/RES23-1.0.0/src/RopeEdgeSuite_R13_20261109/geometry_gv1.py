"""Separate current S1 tracking from conditional fixed-column measurement.

All uncertainty figures are development sensitivity envelopes, not calibrated
coverage probabilities. An uncalibrated d never becomes a certified root count.
"""
import math
import numpy as np
from common import D,PITCH,plain

PARAMETERS=dict(maximum_extrapolation_pitch=12.,maximum_extrapolation_span_ratio=.75,
    maximum_projection_sensitivity_pitch=.35,minimum_effective_blocks=6,
    spatial_block_size_pitch=2.,split_fit_segments=3,reference_d_sensitivity_px=50.)

def assess(track,region,calibration=None,d=D,pitch=PITCH):
    out=dict(tracking_valid=bool(track['valid']),conditional_projection_valid=False,certified_distance_valid=False,
        d_original_px=d,d_calibrated=bool(calibration and calibration.get('calibration_valid')),
        reasons=[],y_at_d=None,raw_gap_px=None,parameters=PARAMETERS,
        uncertainty_semantics='development_sensitivity_envelope_not_calibrated_confidence_interval')
    if not track['valid']:out['reasons']=['current_active_rope_support_missing'];return out
    pts=np.asarray(track['line_points_xy'],float);abc=np.asarray(track['current_line_abc'],float)
    if len(pts)<24 or abs(abc[1])<1e-8:out['reasons']=['current_line_ill_conditioned'];return out
    x,y=pts.T;m=-abc[0]/abc[1];c=-abc[2]/abc[1];yd=m*d+c
    span=float(np.ptp(x));extension=float(max(0,d-x.max(),x.min()-d))
    # Effective spatial blocks avoid pretending every adjacent4px sample is independent.
    group=np.floor((x-x.min())/(2*pitch)).astype(int);blocks=[]
    for k in np.unique(group):
        q=pts[group==k]
        if len(q)>=3:blocks.append(np.median(q,axis=0))
    bp=np.asarray(blocks);line_sigma=None
    if len(bp)>=6:
        x0=float(bp[:,0].mean());X=np.c_[np.ones(len(bp)),bp[:,0]-x0]
        b=np.linalg.lstsq(X,bp[:,1],rcond=None)[0];res=bp[:,1]-X@b
        noise=max(1.,float(1.4826*np.median(abs(res-np.median(res)))))
        cov=np.linalg.inv(X.T@X)*noise**2;v=np.array([1,d-x0]);line_sigma=float(np.sqrt(v@cov@v))
    split=[]
    for ii in np.array_split(np.arange(len(x)),3):
        if len(ii)<8 or np.ptp(x[ii])<2*pitch:continue
        xx=x[ii];yy=y[ii];mid=float(xx.mean());fit=np.polyfit(xx-mid,yy,1)
        split.append(float(fit[0]*(d-mid)+fit[1]))
    sensitivity=max([abs(a-yd) for a in split],default=float('inf'))
    line_bound=max(2.,2*(line_sigma or 0),sensitivity)
    out.update(y_at_d=float(yd),extrapolation_px=extension,observed_span_x_px=span,
        extrapolation_to_span_ratio=extension/max(span,1),effective_spatial_blocks=len(bp),
        split_y_at_d_px=split,split_sensitivity_px=None if not np.isfinite(sensitivity) else sensitivity,
        line_standard_error_development_px=line_sigma,line_sensitivity_bound_px=None if not np.isfinite(line_bound) else line_bound)
    why=out['reasons']
    if extension>12*pitch or extension>.75*span:why.append('unsupported_extrapolation_too_long')
    if len(bp)<6 or len(split)<3:why.append('independent_line_support_insufficient')
    if line_bound>.35*pitch:why.append('projected_position_sensitive_to_support_subset')
    if not region['valid']:why.append('current_target_boundary_unavailable')
    else:
        edge=region['edge_y'];curve=np.asarray(region['top_curve_xy']);eprime=(np.interp(d+16,curve[:,0],curve[:,1])-np.interp(d-16,curve[:,0],curve[:,1]))/32
        rawgap=float(yd-edge);edge_bound=float(region.get('edge_uncertainty_px') or 3.)
        conditional_bound=line_bound+edge_bound
        dscale=float(calibration.get('d_sigma_px') or 0) if out['d_calibrated'] else 50.
        d_sensitivity=abs(m-eprime)*dscale
        out.update(raw_gap_px=rawgap,conditional_gap_interval_px=[rawgap-conditional_bound,rawgap+conditional_bound],
            conditional_gap_sensitivity_px=conditional_bound,d_perturbation_px=dscale,
            d_perturbation_effect_px=float(d_sensitivity),combined_sensitivity_px=float(conditional_bound+d_sensitivity),
            boundary_sensitivity_px=edge_bound,d_perturbation_is_assumed_scenario=not out['d_calibrated'])
        if rawgap+conditional_bound<0:why.append('projection_boundary_incompatible')
        elif rawgap-conditional_bound<=0:why.append('near_edge_position_unresolved')
    out['conditional_projection_valid']=not why
    out['certified_distance_valid']=bool(not why and out['d_calibrated'])
    if not out['d_calibrated']:out['calibration_note']='reference_d_only_no_current_video_bend_calibration'
    return plain(out)
