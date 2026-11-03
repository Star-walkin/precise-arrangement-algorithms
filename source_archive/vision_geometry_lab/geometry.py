"""Image-plane measurements only. No winding state or actuator decisions.

The fitted rope is an unoriented line n.p = rho. A rope band has TWO edges;
neither a small distance nor apparent tangency establishes a new winding layer.
All exported distances use original image pixels, not millimetres.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import cv2
import numpy as np


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or '.png', image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def measure_relation(arc, rope):
    """Measure stationary-distance contact candidates on OBSERVED arc support.

    A crossing can have distance zero but does not imply tangency. We require
    a local interior minimum of line-normal distance slope, report angle and
    both band-edge distances, and decline to infer a physical layer transition.
    Tolerances below are diagnostic pixel tests, NOT commissioned event gates.
    """
    out = dict(valid=False, status='unavailable', reason='',
               physical_contact_confirmed=False, new_layer_confirmed=False,
               force_command=None)
    if not arc.get('valid') or not rope.get('valid'):
        out['reason'] = 'arc_or_rope_unavailable'
        return out
    pts = np.asarray(arc.get('points', []), np.float64)
    if pts.ndim != 2 or pts.shape[0] < 9 or pts.shape[1] != 2:
        out['reason'] = 'too_few_arc_points'
        return out
    n = np.asarray(rope['normal'], np.float64)
    norm = np.linalg.norm(n)
    if not np.isfinite(norm) or norm < 1e-8:
        out['reason'] = 'invalid_line'
        return out
    n /= norm
    rho = float(rope['rho_px']) / norm
    width = rope.get('width_px')
    if width is not None and (not np.isfinite(width) or width <= 0):
        width = None
    signed = pts @ n - rho
    segments = arc.get('segments')
    if segments is None:
        ranges = arc.get('observed_ranges', [])
        pts = pts[np.argsort(pts[:,0])]
        segments = [pts[(pts[:,0] >= r[0]) & (pts[:,0] <= r[1])] for r in ranges] if ranges else [pts]
    candidates = []
    # Use arc length rather than y(x), so vertical/rotated cameras work too.
    # A tangent stencil never crosses a gap or a missing-support segment.
    for segment in segments:
        local_pts = np.asarray(segment, np.float64)
        if len(local_pts) < 11:
            continue
        travel = np.r_[0, np.cumsum(np.linalg.norm(np.diff(local_pts,axis=0),axis=1))]
        angles = np.full(len(local_pts), np.nan)
        for i in range(4, len(local_pts)-4):
            times = travel[i-4:i+5]-travel[i]
            if np.ptp(times) < 1:
                continue
            fit = np.polynomial.polynomial.polyfit(times, local_pts[i-4:i+5]-local_pts[i],2)
            tangent = fit[1]
            length = np.linalg.norm(tangent)
            if length < 1e-8:
                continue
            angles[i] = np.degrees(np.arcsin(np.clip(abs(tangent @ n)/length,0,1)))
        for i in range(5,len(local_pts)-5):
            if np.isfinite(angles[i]) and angles[i] <= angles[i-1] and angles[i] <= angles[i+1]:
                candidates.append((local_pts[i],float(local_pts[i] @ n-rho),float(angles[i])))
    if not candidates:
        out.update(reason='tangent_outside_observed_arc_or_no_stationary_point',
                   nearest_line_distance_px=float(np.min(np.abs(signed))))
        return out
    # Choose closest band edge among small-angle stationary candidates. We do
    # NOT select the nearest intersection, which would reward crossings.
    best_point, d, angle = min(candidates, key=lambda p: abs(abs(p[1]) - (width or 0)/2) + p[2])
    uncertainty = max(1.0, float(arc.get('residual_px') or 0) + float(rope.get('residual_px') or 0))
    # Distances are unsigned separations to both visible image-band edges.
    edge_distances = [abs(d-width/2), abs(d+width/2)] if width is not None else None
    status = 'not_tangent_in_image'
    if angle <= 3.0:
        status = 'parallel_but_separated'
        if edge_distances is None:
            status = 'parallel_width_unavailable'
        elif min(edge_distances) <= uncertainty:
            status = 'image_tangency_candidate'
        elif abs(d) < width/2:
            status = 'arc_inside_rope_band'
    segment = np.asarray(rope.get('observed_segment', []), np.float64)
    extrapolation = None
    if segment.shape == (2, 2):
        direction = np.array([n[1], -n[0]])
        interval = sorted(segment @ direction)
        position = float(best_point @ direction)
        extrapolation = max(interval[0]-position, position-interval[1], 0.0)
    out.update(valid=True, status=status, reason='image_geometry_only',
               candidate_point=best_point.tolist(),
               signed_center_distance_px=d, center_distance_px=abs(d),
               band_edge_distances_px=edge_distances,
               angle_difference_deg=angle, diagnostic_tolerance_px=uncertainty,
               line_extrapolation_px=extrapolation,
               lens_distortion_corrected=False,
               warning='Apparent tangency does not establish physical contact or a new layer.')
    return out


def rope_exclusion_band(image_shape, rope, margin_px=8.0):
    """Current-frame rope occlusion corridor; never use a stale or invalid line.

    The conservative width proxy is permitted for EXCLUDING pixels only. It is
    not used as a measured rope diameter or a positive contact observation.
    """
    metadata = dict(valid=False,reason='rope_unavailable',half_width_px=None)
    if not rope.get('valid'):
        return None,metadata
    n = np.asarray(rope.get('normal'),float)
    norm = np.linalg.norm(n)
    if n.shape != (2,) or not np.isfinite(norm) or norm < 1e-8:
        metadata['reason']='invalid_rope_line'
        return None,metadata
    n=n/norm
    rho=float(rope['rho_px'])/norm
    if not np.isfinite(rho) or not np.isfinite(margin_px) or margin_px < 0:
        metadata['reason']='invalid_line_or_margin'
        return None,metadata
    diagnostics=rope.get('diagnostics',{})
    width=rope.get('width_px')
    interval=rope.get('width_interval_px')
    if interval and len(interval)==2 and np.isfinite(interval).all():
        upper=float(interval[1]); source='current_frame_width_interval_upper'
    elif width is not None and np.isfinite(width) and width>0:
        upper=float(width); source='current_frame_optical_width'
    else:
        candidate=diagnostics.get('candidate_optical_width_px')
        spread=diagnostics.get('width_mad_px') or 0
        upper=float(candidate)+2*float(spread) if candidate is not None and np.isfinite(candidate) else 8.0
        source='uncertain_width_proxy_for_exclusion_only'
    residual=max(0.,float(rope.get('normal_uncertainty_px') or rope.get('residual_px') or 0))
    half=.5*max(upper,2)+float(margin_px)+3*residual
    if not np.isfinite(half) or half > min(image_shape[:2])*.10:
        metadata['reason']='exclusion_band_too_uncertain'
        return None,metadata
    yy,xx=np.ogrid[:image_shape[0],:image_shape[1]]
    mask=np.abs(n[0]*xx+n[1]*yy-rho)<=half
    metadata.update(valid=True,reason='current_frame_parallel_corridor',normal=n.tolist(),rho_px=rho,
                    half_width_px=float(half),extra_margin_px=float(margin_px),width_source=source,
                    physical_diameter_measured=False)
    return mask,metadata


def analyze_frame(image, *, arc_roi=None, rope_roi=None, rope_mask=None,
                  surface_mask=None, seed_points=None, exclusion_margin_px=8.0,
                  wound_seed_points=None):
    from .arc_detector import detect_arc
    from .rope_detector import detect_rope
    from .conic_fit import fit_conic
    from .conic_geometry import measure_conic_relation
    from .wound_rope_tracker import detect_wound_curve
    start = time.perf_counter()
    rope = detect_rope(image, roi=rope_roi, rope_mask=rope_mask,
                       seed_points=seed_points)
    after_rope = time.perf_counter()
    mask,exclusion = rope_exclusion_band(image.shape,rope,exclusion_margin_px)
    arc = detect_arc(image, roi=arc_roi, surface_mask=surface_mask,exclusion_mask=mask)
    after_arc = time.perf_counter()
    conic = fit_conic(arc,image_shape=image.shape)
    after_conic=time.perf_counter()
    relation = measure_conic_relation(conic,rope,exclusion=exclusion)
    after_relation=time.perf_counter()
    wound=detect_wound_curve(image,rope,arc,surface_mask=surface_mask,seed_points=wound_seed_points)
    after_wound=time.perf_counter()
    guidance_points=[] if wound_seed_points is None else np.asarray(wound_seed_points,float).tolist()
    return dict(schema_version=3, algorithm='local_boundary_wound_curve_v3',
                image_size=[image.shape[1], image.shape[0]], arc=arc, rope=rope,
                conic=conic,exclusion=exclusion,relation=relation,wound_curve=wound,
                wound_guidance=dict(used=bool(guidance_points),
                    kind='manual_search_points' if guidance_points else 'none',
                    points=guidance_points,
                    semantics='search hints only; not verified rope centers or edges'),
                timing_ms=dict(arc=(after_arc-after_rope)*1000,rope=(after_rope-start)*1000,
                               conic=(after_conic-after_arc)*1000,
                               relation=(after_relation-after_conic)*1000,
                               wound_curve=(after_wound-after_relation)*1000,
                               total=(time.perf_counter()-start)*1000),
                hardware_access=False, winding_state_inferred=False)


def _dashed_line(image,a,b,color,thickness=1,dash=12):
    a,b=np.asarray(a,float),np.asarray(b,float)
    distance=np.linalg.norm(b-a)
    if distance<1:
        return
    for t in np.arange(0,distance,2*dash):
        p=a+(b-a)*(t/distance); q=a+(b-a)*(min(t+dash,distance)/distance)
        cv2.line(image,tuple(np.rint(p).astype(int)),tuple(np.rint(q).astype(int)),color,thickness,cv2.LINE_AA)


def draw_overlay(image, result, *, show_arc=True, show_rope=True,
                 show_raw=False,show_exclusion=False,show_conic=False,show_wound=True):
    """Rendering the algorithm's diagnostic geometry, not editing source media."""
    out = image.copy()
    if not result:
        return out
    scale = max(1, round(max(image.shape[:2])/1500))
    arc, rope = result['arc'], result['rope']
    exclusion=result.get('exclusion',{})
    if show_exclusion and exclusion.get('valid'):
        n=np.asarray(exclusion['normal'],float); rho=exclusion['rho_px']; d=np.array([n[1],-n[0]])
        for offset in [-exclusion['half_width_px'],exclusion['half_width_px']]:
            p=n*(rho+offset)
            a,b=p-d*2*max(image.shape[:2]),p+d*2*max(image.shape[:2])
            ok,a,b=cv2.clipLine((0,0,image.shape[1],image.shape[0]),tuple(a.astype(int)),tuple(b.astype(int)))
            if ok:
                _dashed_line(out,a,b,(230,130,70),scale,12*scale)
    conic=result.get('conic',{})
    has_local='local_boundary_segments' in arc
    if show_arc and has_local:
        # Local measurements remain drawable when a globally extrapolated
        # ellipse fails. Never join separate observed fragments across a gap.
        for segment in arc.get('local_boundary_segments',[]):
            points=np.asarray(segment,float)
            if points.ndim==2 and points.shape[1:]==(2,) and len(points)>=2:
                cv2.polylines(out,[np.rint(points).astype(np.int32)],False,(60,235,90),scale,cv2.LINE_AA)
    if (show_conic or (show_arc and not has_local)) and conic.get('valid'):
        points=np.asarray(conic.get('model_points',[]),float)
        supported=conic.get('model_supported',[False]*len(points))
        for i in range(len(points)-1):
            a,b=tuple(np.rint(points[i]).astype(int)),tuple(np.rint(points[i+1]).astype(int))
            if has_local:
                if i%8<4:
                    cv2.line(out,a,b,(240,170,100),scale,cv2.LINE_AA)
            elif supported[i] and supported[i+1]:
                cv2.line(out,a,b,(60,235,90),scale,cv2.LINE_AA)
            elif i%6<3:
                cv2.line(out,a,b,(60,235,90),scale,cv2.LINE_AA)
    elif show_arc and not has_local and not conic and arc.get('valid'):
        # Compatibility for opening saved v1 results only.
        for segment in arc.get('segments', [arc['points']]):
            pts = np.asarray(segment,float)
            if len(pts) >= 2:
                cv2.polylines(out,[np.rint(pts).astype(np.int32)],False,(60,235,90),scale,cv2.LINE_AA)
    if show_raw and arc.get('valid'):
        for point in arc.get('points',[]):
            cv2.circle(out,tuple(np.rint(point).astype(int)),max(1,scale),(245,245,245),-1,cv2.LINE_AA)
    if show_rope and rope.get('valid'):
        n = np.asarray(rope['normal'], float)
        rho = float(rope['rho_px'])
        base = n*rho
        direction = np.array([n[1], -n[0]])
        length = 2*max(image.shape[:2])
        # Unknown width MUST NOT draw two zero-offset yellow lines on top of
        # the orange centerline. That v1 renderer bug hid the actual centerline.
        shifts=[(0,(0,160,255),False)]
        width=rope.get('width_px')
        interval=rope.get('width_interval_px')
        if width is not None and np.isfinite(width) and width>0:
            shifts.extend([(-width/2,(0,230,255),True),(width/2,(0,230,255),True)])
        elif interval and len(interval)==2 and np.isfinite(interval).all():
            representative=.5*(float(interval[0])+float(interval[1]))
            shifts.extend([(-representative/2,(0,230,255),True),(representative/2,(0,230,255),True)])
        for shift, color,dashed in shifts:
            a = tuple(np.rint(base+n*shift-direction*length).astype(int))
            b = tuple(np.rint(base+n*shift+direction*length).astype(int))
            if dashed:
                ok,a,b=cv2.clipLine((0,0,image.shape[1],image.shape[0]),a,b)
                if ok:
                    _dashed_line(out,a,b,color,scale,10*scale)
            else:
                cv2.line(out, a, b, color, scale, cv2.LINE_AA)
        segment = np.asarray(rope.get('observed_segment', []), float)
        if segment.shape == (2, 2):
            if width is not None and np.isfinite(width) and width>0:
                for shift in [-width/2,width/2]:
                    edge=segment+n*shift
                    cv2.line(out,tuple(np.rint(edge[0]).astype(int)),tuple(np.rint(edge[1]).astype(int)),
                             (0,230,255),scale,cv2.LINE_AA)
            cv2.line(out, tuple(segment[0].astype(int)), tuple(segment[1].astype(int)),
                     (255, 230, 0), 2*scale, cv2.LINE_AA)
    relation = result['relation']
    if (show_conic or (show_arc and not has_local)) and show_rope and relation.get('valid'):
        p = tuple(np.rint(relation['candidate_point']).astype(int))
        cv2.circle(out, p, 5*scale, (255, 0, 255), scale+1, cv2.LINE_AA)
        for point,on_arc in zip(relation.get('intersections',[]),relation.get('intersections_on_modeled_arc',[])):
            if on_arc:
                cv2.circle(out,tuple(np.rint(point).astype(int)),4*scale,(255,130,0),scale+1,cv2.LINE_AA)
    if show_wound:
        candidates=result.get('wound_curve',{}).get('candidates',[])
        for index,candidate in enumerate(candidates[:3]):
            color=(230,70,220) if index==0 else (190,110,200)
            # A candidate may be drawable despite unconfirmed rope identity.
            # Draw only its image-supported fragments, never arbitrary bridges.
            for segment in candidate.get('observed_segments',[]):
                points=np.asarray(segment,float)
                if points.ndim==2 and points.shape[1:]==(2,) and len(points)>=2:
                    for j in range(len(points)-1):
                        if j%6<3:
                            cv2.line(out,tuple(np.rint(points[j]).astype(int)),tuple(np.rint(points[j+1]).astype(int)),
                                     color,scale,cv2.LINE_AA)
    return out


def dump_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
