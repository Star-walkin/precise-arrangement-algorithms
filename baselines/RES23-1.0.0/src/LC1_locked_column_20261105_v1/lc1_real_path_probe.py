"""Experimental S1-anchored turn audit using current supported surface points.

Does not replace lc1_geometry.estimate_turn or authorize physical contact.
H6 Hermite transitions, entry conditions and unsupported points never enter
the fit. No interpolated gap is added as an observation. Uncertainty remains
conditional on the explicit optical-noise/correlation assumptions below.
"""
from __future__ import annotations
import math
import numpy as np
from lc1_geometry import normalize_line


def _mad(v):
    v=np.asarray(v,float)
    return float(1.4826*np.median(abs(v-np.median(v))))


def _spatial_representatives(points,bin_px):
    """Select actual input points (not averaged/synthesized coordinates)."""
    p=np.asarray(points,float)
    order=np.argsort(p[:,0],kind='stable');p=p[order]
    keys=np.floor((p[:,0]-p[0,0])/bin_px).astype(int)
    chosen=[]
    for key in np.unique(keys):
        ids=np.flatnonzero(keys==key)
        # Middle actual sample prevents dense local path sampling gaining weight.
        chosen.append(ids[len(ids)//2])
    return p[chosen]


def _profile(u,y,m,intercept,grid,noise):
    target=y-m*u-intercept
    h=np.minimum(u[None,:]-grid[:,None],0.)
    scale=max(1.,float(np.max(abs(h))))
    design=(h/scale)**2
    weight=np.ones_like(design)
    q=np.zeros(len(grid))
    for _ in range(7):
        q=np.sum(weight*design*target[None,:],axis=1)/np.maximum(np.sum(weight*design**2,axis=1),1.e-15)
        residual=target[None,:]-q[:,None]*design
        weight=np.minimum(1.,2.5*noise/np.maximum(abs(residual),1.e-12))
    e=abs(residual);delta=2.5*noise
    # No global residual trim that can throw out an entire minority branch.
    loss=np.where(e<=delta,e*e,2*delta*e-delta*delta)
    score=np.mean(loss,axis=1)
    return score,q/scale**2,residual


def estimate_turn_observed_anchored(path_xy,path_point_support,path_stage,
                                    free_line_abc,*,free_line_points_xy=None,
                                    free_side='right',pitch_px=20.,
                                    connection_eligible=False,
                                    curve_noise_floor_px=2.,
                                    free_noise_floor_px=1.,
                                    correlation_block_pitch=2.):
    """Audit apparent C1 transition using S1 line + supported H6 surface points.

    `connection_eligible` must be independently established from genuine H6
    attachment/topology, continuation eligibility, branch/direction uniqueness.
    It is deliberately separate from pixel-complete whole_path_observed.

    Returned valid=True means this conditional optical geometry passed its
    stated gates. It neither changes whole_path_candidate_supported nor proves
    same-rope identity or physical contact. Noise floors/correlation length are
    development assumptions, not measured accuracy or confidence coverage.
    """
    def reject(reason,**extra):
        out=dict(valid=False,x=None,y=None,quality=0.,rmse=None,
                 uncertainty_px=None,reason=reason,
                 method='experimental_S1_anchored_current_supported_surface_C1_profile',
                 whole_observed_flag_promoted=False,physical_contact_confirmed=False,
                 uncertainty_is_conditional=True)
        out.update(extra);return out
    if free_side not in ('right','left'):raise ValueError('invalid free_side')
    if not np.isfinite(pitch_px) or pitch_px<=0:return reject('invalid_pitch')
    if connection_eligible is not True:return reject('no_unique_legal_H6_curve_connection')
    try:
        p=np.asarray(path_xy,float).reshape(-1,2)
        support=np.asarray(path_point_support,bool).reshape(-1)
        stage=np.asarray(path_stage).reshape(-1)
        abc=normalize_line(free_line_abc)
        free=np.asarray(free_line_points_xy,float).reshape(-1,2)
    except (ValueError,TypeError):return reject('invalid_or_missing_current_observations')
    if len(support)!=len(p) or len(stage)!=len(p):return reject('support_stage_alignment_mismatch')
    if abs(abc[1])<.1:return reject('steep_y_of_x_model')
    take=support&(stage=='surface')&np.isfinite(p).all(axis=1)
    curve=p[take]
    free=free[np.isfinite(free).all(axis=1)]
    if len(curve)<8:return reject('insufficient_current_supported_surface_points',surface_point_count=len(curve))
    if len(free)<12:return reject('S1_anchor_uncertainty_unavailable_or_insufficient')
    sign=1. if free_side=='right' else -1.
    curve=curve.copy();curve[:,0]*=sign
    free=free.copy();free[:,0]*=sign
    u_span=float(np.ptp(curve[:,0]));free_span=float(np.ptp(free[:,0]))
    if u_span<max(32.,2*pitch_px):return reject('insufficient_spatial_curve_span',curve_span_px=u_span)
    if free_span<4*pitch_px:return reject('insufficient_spatial_S1_anchor_span')
    cbin=_spatial_representatives(curve,max(4.,.5*pitch_px))
    fbin=_spatial_representatives(free,max(8.,pitch_px))
    if len(cbin)<4 or len(fbin)<6:return reject('insufficient_distinct_spatial_bins')
    m=float(-abc[0]/abc[1]*sign);b=float(-abc[2]/abc[1])
    u,y=cbin.T
    # Current supported free and curve domains must flank, not overlap, d.
    lo=float(np.max(curve[:,0]));hi=float(np.min(free[:,0]))
    common=dict(surface_point_count=len(curve),free_point_count=len(free),
                surface_spatial_bins=len(cbin),free_spatial_bins=len(fbin),
                curve_span_px=u_span,free_span_px=free_span,
                omitted_H6_transition_points=int(np.sum(stage=='transition')),
                omitted_unsupported_path_points=int(np.sum(~support)),
                fit_uses_Hermite_or_unsupported_points=False,
                free_abc=abc.tolist(),free_side=free_side,
                apparent_turn_search_interval_x=sorted([sign*lo,sign*hi]),
                interpolation_is_not_observation=True)
    if hi<=lo+2:return reject('S1_and_curve_domains_overlap_or_no_localizable_gap',**common)
    step=min(1.,pitch_px/24.)
    grid=np.linspace(lo,hi,max(81,min(2401,int(math.ceil((hi-lo)/step))+1)))
    # Estimate anchor covariance using spatially separated measured S1 samples.
    center=float(np.mean(fbin[:,0]));fu=fbin[:,0]-center
    ferr=fbin[:,1]-(m*fbin[:,0]+b)
    sigma_free=max(float(free_noise_floor_px),_mad(ferr))
    A=np.column_stack([np.ones(len(fbin)),fu])
    cov=sigma_free**2*np.linalg.pinv(A.T@A)
    # S1 lateral-center quantization/optical offset cannot vanish with more points.
    cov[0,0]+=float(free_noise_floor_px)**2*.25
    sigma_curve=max(float(curve_noise_floor_px),.5*sigma_free)
    scores,q,res=_profile(u,y,m,b,grid,sigma_curve)
    k=int(np.argmin(scores));d=float(grid[k]);quad=float(q[k]);err=res[k]
    rmse=float(np.sqrt(np.mean(err**2)))
    n_eff=min(float(len(cbin)),max(2.,u_span/(float(correlation_block_pitch)*pitch_px)))
    delta=(2.5*sigma_curve)**2/n_eff
    accepted=scores<=scores[k]+delta
    left=float(grid[accepted].min());right=float(grid[accepted].max())
    profile_closed=bool(not accepted[0] and not accepted[-1])
    profile_half=max(d-left,right-d)
    # Carry independent S1 line slope/intercept uncertainty into d; no gap samples.
    eig,vec=np.linalg.eigh(cov)
    shifted=[];anchor_closed=True
    for j in range(2):
        perturb=2.5*math.sqrt(max(0.,float(eig[j])))*vec[:,j]
        for direction in (-1.,1.):
            db,dm=direction*perturb
            mm=m+dm;bb=b+db-dm*center
            ss,_,_=_profile(u,y,mm,bb,grid,sigma_curve)
            kk=int(np.argmin(ss));good=ss<=ss[kk]+delta
            shifted.extend([float(grid[good].min()),float(grid[good].max())])
            anchor_closed &= bool(not good[0] and not good[-1])
    uncertainty=max(profile_half,max(abs(v-d) for v in shifted),float((grid[1]-grid[0])/2))
    anchor_v=np.array([1.,d-center]);anchor_sigma=math.sqrt(max(0.,float(anchor_v@cov@anchor_v)))
    baseline=y-m*u-b
    base_e=abs(baseline);clip=2.5*sigma_curve
    baseline_score=float(np.mean(np.where(base_e<=clip,base_e**2,2*clip*base_e-clip**2)))
    reduction=1-float(scores[k])/max(baseline_score,1.e-12)
    observed_deviation=float(np.max(abs(quad*np.minimum(u-d,0.)**2)))
    details=dict(**common,x=sign*d,y=m*d+b,rmse=rmse,slope=m*sign,
                 intercept=b,quadratic=quad,uncertainty_px=float(uncertainty),
                 conditional_profile_interval_x=sorted([sign*left,sign*right]),
                 conditional_anchor_perturbation_interval_x=sorted([sign*min(shifted),sign*max(shifted)]),
                 profile_closed=profile_closed,anchor_perturbed_profiles_closed=anchor_closed,
                 effective_curve_samples_assumed=n_eff,correlation_block_pitch=float(correlation_block_pitch),
                 curve_noise_floor_px=float(curve_noise_floor_px),free_noise_floor_px=float(free_noise_floor_px),
                 anchor_prediction_sigma_at_d_px=anchor_sigma,curve_fit_noise_assumed_px=sigma_curve,
                 score_reduction=float(reduction),curved_deviation_px=observed_deviation,
                 uncertainty_components_px={'curve_profile_halfwidth':profile_half,'with_anchor_perturbations':float(max(abs(v-d) for v in shifted))},
                 uncertainty_limit_px=.5*pitch_px,
                 uncertainty_caveat='Noise/correlation floors are explicit unvalidated optical assumptions. H6 surface coordinates may inherit correlated fitted F3 center-track error; no calibrated coverage is claimed.')
    if observed_deviation<max(2.,.2*pitch_px,3*_mad(err)) or reduction<.25:
        return reject('straight_or_unresolved_supported_surface_curvature',**details)
    if rmse>max(1.5,.15*pitch_px):return reject('poor_observed_surface_model_fit',**details)
    if not profile_closed or not anchor_closed:return reject('turn_profile_open_or_boundary_limited',**details)
    if uncertainty>.5*pitch_px:return reject('turn_localization_uncertain',**details)
    quality=max(0.,1-rmse/(2*max(1.5,.15*pitch_px)))*max(0.,1-uncertainty/(1.5*pitch_px))
    return dict(valid=True,quality=float(quality),reason='conditional_observed_geometry_passed',
                method='experimental_S1_anchored_current_supported_surface_C1_profile',
                whole_observed_flag_promoted=False,physical_contact_confirmed=False,
                uncertainty_is_conditional=True,**details)
