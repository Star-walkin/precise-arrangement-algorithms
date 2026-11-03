"""Analytic line/ellipse geometry with explicit observation and error limits.

Intersections are NOT tangent points. A single intersection on a visible arc
can be one of TWO intersections with the complete ellipse. All event outputs
are research candidates only; no layer/last-turn/force confirmation is made.
"""
from __future__ import annotations
import math
import numpy as np


def _on_interval(angle, interval, tolerance=1e-7):
    lo, hi = map(float, interval)
    if hi-lo >= 2*math.pi-tolerance:
        return True
    middle = .5*(lo+hi)
    value = angle+2*math.pi*round((middle-angle)/(2*math.pi))
    return lo-tolerance <= value <= hi+tolerance


def _ellipse(model):
    center = np.asarray(model['center'],float)
    axes = np.asarray(model['semi_axes'],float)
    theta = float(model['angle_rad'])
    if center.shape != (2,) or axes.shape != (2,) or not np.isfinite([*center,*axes,theta]).all() or np.min(axes) <= 0:
        raise ValueError('invalid_ellipse_parameters')
    c,s = math.cos(theta),math.sin(theta)
    rotation = np.array([[c,-s],[s,c]])
    return center,axes,rotation


def ellipse_line_intersections(model, normal, rho):
    """0/1/2 complete-ellipse roots, stable under large image translation."""
    center,axes,rotation = _ellipse(model)
    n = np.asarray(normal,float)
    norm = np.linalg.norm(n)
    if n.shape != (2,) or not np.isfinite(norm) or norm < 1e-10 or not np.isfinite(rho):
        raise ValueError('invalid_line')
    n,rho = n/norm,float(rho)/norm
    d = np.array([n[1],-n[0]])
    p0 = center+n*(rho-n@center)
    v = (rotation.T@(p0-center))/axes
    u = (rotation.T@d)/axes
    a,b,c = float(u@u),float(2*(u@v)),float(v@v-1)
    discriminant = b*b-4*a*c
    epsilon = 64*np.finfo(float).eps*max(b*b,abs(4*a*c),a,1e-20)
    coordinate_roundoff = 64*np.finfo(float).eps*max(1.,np.linalg.norm(center),abs(rho))
    epsilon = max(epsilon,8*a*coordinate_roundoff/float(np.min(axes)))
    roots = []
    if discriminant >= -epsilon:
        if discriminant <= epsilon:
            roots = [-b/(2*a)]
        else:
            root = math.sqrt(discriminant)
            q = -.5*(b+math.copysign(root,b))
            roots = sorted([q/a,c/q]) if abs(q)>1e-30 else sorted([(-b-root)/(2*a),(-b+root)/(2*a)])
    points = [p0+t*d for t in roots]
    interval = model.get('arc_interval_rad',[0,2*math.pi])
    visible = []
    for point in points:
        local = (rotation.T@(point-center))/axes
        visible.append(_on_interval(math.atan2(local[1],local[0]),interval))
    return dict(complete_intersection_count=len(points),
                visible_arc_intersection_count=int(sum(visible)),
                intersections=[p.tolist() for p in points],
                intersections_on_modeled_arc=visible,
                chord_length_px=float(np.linalg.norm(points[1]-points[0])) if len(points)==2 else (0.0 if len(points)==1 else None),
                discriminant=discriminant)


def measure_conic_relation(model, rope, *, exclusion=None):
    out = dict(valid=False,status='unavailable',reason='',
               physical_contact_confirmed=False,new_layer_confirmed=False,
               last_turn_confirmed=False,last_turn_candidate=False,force_command=None,
               geometry_kind='line_fitted_ellipse_v2')
    if not model.get('valid') or not rope.get('valid'):
        out['reason'] = 'conic_or_rope_unavailable'
        return out
    try:
        center,axes,rotation = _ellipse(model)
        n = np.asarray(rope['normal'],float)
        norm = np.linalg.norm(n)
        if n.shape != (2,) or not np.isfinite(n).all() or norm < 1e-10:
            raise ValueError('invalid_line')
        rho = float(rope['rho_px'])/norm
        n = n/norm
        complete = ellipse_line_intersections(model,n,rho)
    except (ValueError,KeyError,TypeError,ZeroDivisionError) as error:
        out['reason'] = str(error)
        return out
    out.update(complete)
    # Supporting points have tangent parallel to the measured rope. Choose a
    # relevant one on the modeled/observed angular domain, never the hidden
    # opposite side of the ellipse merely because an infinite line reaches it.
    inverse_q = rotation@np.diag(axes*axes)@rotation.T
    reach = math.sqrt(float(n@inverse_q@n))
    interval = model.get('arc_interval_rad',[0,2*math.pi])
    choices = []
    for side in (-1.,1.):
        point = center+side*(inverse_q@n)/reach
        local = (rotation.T@(point-center))/axes
        angle = math.atan2(local[1],local[0])
        if _on_interval(angle,interval):
            gap = side*(rho-float(n@point))  # positive outside, negative secant
            choices.append((abs(gap),point,gap,angle,side))
    if not choices:
        out['reason'] = 'parallel_tangent_outside_observed_angular_domain'
        return out
    _,point,gap,theta,side = min(choices,key=lambda item:item[0])
    segment = np.asarray(rope.get('observed_segment',[]),float)
    extrapolation = 0.0
    length = 0.0
    angular_lever_arm = 0.0
    if segment.shape == (2,2):
        direction = np.array([n[1],-n[0]])
        endpoints = sorted(segment@direction)
        projection = float(point@direction)
        extrapolation = max(endpoints[0]-projection,projection-endpoints[1],0.)
        length = endpoints[1]-endpoints[0]
        reference = rope.get('diagnostics',{}).get('uncertainty_reference_point')
        reference = np.asarray(reference if reference is not None else np.mean(segment,axis=0),float)
        if reference.shape == (2,) and np.isfinite(reference).all():
            angular_lever_arm = abs(float((point-reference)@direction))
        else:
            angular_lever_arm = length+extrapolation
    line_residual = max(.25,float(rope.get('normal_uncertainty_px') or rope.get('residual_px') or 0))
    angle_error = float(rope.get('angular_uncertainty_deg') or math.degrees(math.atan2(2*line_residual,max(length,1))))
    if not np.isfinite(angle_error) or angle_error < 0:
        angle_error = 90.
    conic_noise = max(.25,float(model.get('residual_px') or 0))
    sensitivity = max(0.,float(model.get('sensitivity_px') or 0))
    # Local sensitivity may be much worse than the whole-arc 95th percentile,
    # especially at an occluded endpoint. Never hide it behind a global score.
    model_points = np.asarray(model.get('model_points',[]),float)
    local_sensitivity = np.asarray(model.get('model_sensitivity_px',[]),float)
    if model_points.ndim == 2 and model_points.shape[1:] == (2,) and len(model_points) == len(local_sensitivity) and len(model_points):
        closest = int(np.argmin(np.linalg.norm(model_points-point,axis=1)))
        nearby = local_sensitivity[max(0,closest-2):closest+3]
        sensitivity = max(sensitivity,float(np.max(nearby))) if np.isfinite(nearby).all() else math.inf
    # Engineering discrepancy budget, NOT a statistical calibrated confidence
    # interval. Capped relative to rope size: a bad model must not make the
    # "near tangent" test easier by increasing its tolerance indefinitely.
    error_budget = 2*math.hypot(line_residual,conic_noise)+sensitivity+angular_lever_arm*math.tan(math.radians(min(angle_error,80)))
    width = rope.get('width_px')
    width_interval = rope.get('width_interval_px')
    upper_width = float(width or 0)
    if width_interval and len(width_interval)==2:
        upper_width = max(upper_width,float(width_interval[1]))
    max_budget = max(3.0,min(8.0,.5*upper_width)) if upper_width>0 else 3.0
    tolerance = max(.75,error_budget)
    adequate = np.isfinite(error_budget) and error_budget <= max_budget and length > 0
    # Exact chord length at a parallel line one tolerance INSIDE this ellipse.
    # It automatically changes with scale and local curvature; no universal
    # 5px/10px chord-distance threshold is assumed.
    threshold_line = float(n@point)-side*min(tolerance,max_budget)
    threshold_intersections = ellipse_line_intersections(model,n,threshold_line)
    chord_limit = threshold_intersections['chord_length_px']
    center_candidate = bool(adequate and abs(gap) <= tolerance)
    out.update(valid=True,reason='modeled_image_geometry_only',
               candidate_point=point.tolist(),signed_center_clearance_px=float(gap),
               center_distance_px=abs(float(gap)),
               diagnostic_tolerance_px=float(min(tolerance,max_budget)),
               estimated_error_budget_px=float(error_budget),quality_sufficient=bool(adequate),
               chord_merge_threshold_px=chord_limit,
               center_tangency_candidate=center_candidate,
               line_extrapolation_px=float(extrapolation),
               angular_lever_arm_px=float(angular_lever_arm),
               candidate_model_sensitivity_px=float(sensitivity),
               angle_uncertainty_deg=angle_error,angle_difference_deg=0.0,
               angle_difference_note='parallel_support_tangent_by_construction_not_independent_evidence',
               support_kind='fitted_conic_not_observed_contact',
               lens_distortion_corrected=False,
               warning='Intersections or tangency do not confirm the last turn or a new layer.')
    if not adequate:
        status = 'uncertainty_too_large'
    elif center_candidate:
        status = 'center_near_tangent_candidate'
    elif gap > 0:
        status = 'line_outside_conic'
    else:
        status = 'line_crosses_conic'
    band_relations = []
    if width is not None and np.isfinite(width) and width>0:
        for shift in [-float(width)/2,float(width)/2]:
            edge = ellipse_line_intersections(model,n,rho+shift)
            edge['signed_clearance_px'] = float(gap+side*shift)
            edge['offset_from_center_px'] = shift
            band_relations.append(edge)
        band_distance = min(abs(item['signed_clearance_px']) for item in band_relations)
        out['band_edge_distances_px'] = [abs(item['signed_clearance_px']) for item in band_relations]
        widths = [float(width),float(width)]
        if width_interval and len(width_interval)==2 and np.isfinite(width_interval).all():
            widths = sorted(map(float,width_interval))
        halfwidth_variation = max(abs(w-float(width)) for w in widths)/2
        band_budget_ok = error_budget+halfwidth_variation <= max_budget
        edge_worst = [max(abs(gap+side*sign*w/2) for w in widths) for sign in [-1,1]]
        out['optical_band_tangency_possible'] = bool(adequate and band_distance<=tolerance+halfwidth_variation)
        out['optical_band_tangency_candidate'] = bool(adequate and band_budget_ok and min(edge_worst)<=tolerance)
        out['optical_halfwidth_variation_px'] = halfwidth_variation
        out['optical_width_applicable_at_contact'] = False
        if not center_candidate and out['optical_band_tangency_candidate']:
            status = 'optical_edge_near_tangent_candidate'
    else:
        out['band_edge_distances_px'] = None
        out['optical_band_tangency_candidate'] = False
    out['band_edges'] = band_relations
    out['status'] = status
    out['last_turn_candidate'] = bool(out['center_tangency_candidate'] or out['optical_band_tangency_candidate'])
    if exclusion:
        out['boundary_under_rope_is_inferred'] = bool(exclusion.get('valid') and abs(float(n@point-rho)) <= exclusion['half_width_px'])
    return out
