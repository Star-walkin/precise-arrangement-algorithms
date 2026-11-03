"""Low-confidence adjacent-F3-lane tail proposals, in field-local pixels.

This module only proposes additional review paths. It never changes an existing
path, claims a contact point, selects a material identity, or invents F3 pixels.
Input arrays share one coordinate system (normally the half-size model crop).
"""
from __future__ import annotations
import math
import cv2
import numpy as np

PARAMETERS = dict(maximum_candidates=3, column_step_px=3., column_half_width_px=1,
    initial_forward_widths=8., initial_lateral_widths=2.,
    initial_gap_min_widths=.60, initial_gap_max_widths=2.75,
    pair_motion_gate_lane_widths=.40, pair_width_ratio_min=.65, pair_width_ratio_max=1.50,
    maximum_missing_widths=2., maximum_tail_widths=160., maximum_tail_pixels=1000.,
    maximum_angle_deg=65., maximum_turn_deg_per_width=10.,
    minimum_return_span_widths=5., minimum_pair_fraction=.55)


def _sample(image, points):
    p = np.asarray(points, np.float32).reshape(-1, 2)
    return cv2.remap(np.asarray(image, np.float32), p[:, 0][None], p[:, 1][None],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).ravel()


def _peaks(mask, x, lo, hi):
    h, w = mask.shape
    x = int(round(x)); lo = max(0, int(math.floor(lo))); hi = min(h, int(math.ceil(hi)) + 1)
    if not 1 <= x < w-1 or hi <= lo:
        return np.empty(0)
    ys = np.flatnonzero(mask[lo:hi, x-1:x+2].any(axis=1)) + lo
    if not len(ys):
        return np.empty(0)
    groups = np.split(ys, np.flatnonzero(np.diff(ys) > 1) + 1)
    # Multiple adjoining pixels are the raster thickness of one boundary.
    return np.asarray([float(v.mean()) for v in groups])


def _pairs(peaks, width_min, width_max):
    if len(peaks) < 2:
        return np.empty((0, 2))
    pairs = np.c_[peaks[:-1], peaks[1:]]
    widths = np.diff(pairs, axis=1).ravel()
    return pairs[(widths >= width_min) & (widths <= width_max)]


def _slope_at(mask, x, pair, dx):
    center = pair.mean(); gap = pair[1] - pair[0]
    q = _pairs(_peaks(mask, x+dx, center-2*gap, center+2*gap), .65*gap, 1.5*gap)
    if not len(q):
        return None
    dist = np.max(abs(q-pair), axis=1)
    ix = int(np.argmin(dist))
    if dist[ix] > gap*.70:
        return None
    return float((q[ix].mean()-center)/dx)


def _initial_pairs(mask, surface, origin, slope, direction, width):
    proposals = []
    for distance in np.arange(max(6., 2*width), max(24., PARAMETERS['initial_forward_widths']*width)+1, 3.):
        x = origin[0]+direction*distance
        pred = origin[1]+slope*(x-origin[0])
        pairs = _pairs(_peaks(mask, x, pred-4*width, pred+4*width),
                       width*PARAMETERS['initial_gap_min_widths'], width*PARAMETERS['initial_gap_max_widths'])
        for pair in pairs:
            gap = pair[1]-pair[0]; center = pair.mean()
            if abs(center-pred) > PARAMETERS['initial_lateral_widths']*width:
                continue
            if _sample(surface, [[x, center]])[0] <= 0:
                continue
            observed_slope = _slope_at(mask, x, pair, direction*max(6., width))
            if observed_slope is None:
                continue
            turn = abs(math.atan(observed_slope)-math.atan(slope))
            if turn > math.radians(28):
                continue
            cost = abs(center-pred)/width + .20*distance/width + .5*turn
            proposals.append(dict(x=float(x), pair=pair, slope=observed_slope, cost=cost))
    proposals.sort(key=lambda p:p['cost'])
    return proposals


def _connector(origin, tangent, target, target_slope, direction, width):
    """A short monotonic Hermite bridge, always explicitly unsupported by F3."""
    dx = target[0]-origin[0]
    if abs(dx) < 1.:
        return None
    slope0 = tangent[1]/tangent[0]
    count = max(3, int(abs(dx)/PARAMETERS['column_step_px'])+1)
    u = np.linspace(0, 1, count)
    y = ((2*u**3-3*u**2+1)*origin[1] + (u**3-2*u**2+u)*dx*slope0 +
         (-2*u**3+3*u**2)*target[1] + (u**3-u**2)*dx*target_slope)
    p = np.c_[origin[0]+u*dx, y]
    angle = np.arctan2(np.diff(p[:, 1]), direction*np.diff(p[:, 0]))
    # For right-to-left traversal use the forward image-horizontal angle.
    lengths = np.linalg.norm(np.diff(p, axis=0), axis=1)
    if (abs(angle) > math.radians(PARAMETERS['maximum_angle_deg'])).any():
        return None
    if len(angle)>1:
        turn = abs(np.diff(angle))/np.maximum((lengths[1:]+lengths[:-1])*.5, 1e-6)*width
        if (turn > math.radians(PARAMETERS['maximum_turn_deg_per_width'])).any():
            return None
    return p


def _follow(mask, surface, initial, direction, optical_width):
    x = initial['x']; center = float(initial['pair'].mean()); gap = float(np.diff(initial['pair'])[0])
    initial_gap = gap; slope = initial['slope']; step = direction*PARAMETERS['column_step_px']
    points = [[x, center]]; widths = [gap/math.sqrt(1+slope*slope)]
    pairs = [initial['pair'].tolist()]; observed = [True]; missing = 0.; reason = 'length_budget'
    max_steps = int(min(PARAMETERS['maximum_tail_pixels'], PARAMETERS['maximum_tail_widths']*optical_width)/abs(step))
    for _ in range(max_steps):
        x += step; predicted = center+slope*step
        if min(x-2, mask.shape[1]-3-x, predicted-2, mask.shape[0]-3-predicted) < 0:
            reason = 'image_edge'; break
        if _sample(surface, [[x, predicted]])[0] <= 0:
            reason = 'surface_mask_endpoint_not_physical_contact'; break
        expected = np.array([predicted-gap/2, predicted+gap/2])
        options = _pairs(_peaks(mask, x, predicted-2*gap, predicted+2*gap),
                         PARAMETERS['pair_width_ratio_min']*initial_gap, PARAMETERS['pair_width_ratio_max']*initial_gap)
        match = None
        if len(options):
            distances = np.max(abs(options-expected), axis=1)
            ix = int(np.argmin(distances))
            if distances[ix] <= PARAMETERS['pair_motion_gate_lane_widths']*gap:
                match = options[ix]
        if match is None:
            missing += abs(step)*math.sqrt(1+slope*slope)
            if missing > max(8., PARAMETERS['maximum_missing_widths']*optical_width):
                reason = 'paired_F3_boundaries_missing'; break
            center = predicted; measured_pair = expected
        else:
            missing = 0.
            raw_center = float(match.mean()); residual = raw_center-predicted
            # Smooth column quantization with a bounded turn per unit arc.
            proposed_slope = slope+.22*residual/step
            angle0 = math.atan(slope); angle1 = math.atan(proposed_slope)
            max_da = math.radians(PARAMETERS['maximum_turn_deg_per_width'])*abs(step)*math.sqrt(1+slope*slope)/optical_width
            angle1 = angle0+float(np.clip(angle1-angle0, -max_da, max_da))
            slope = math.tan(angle1)
            if abs(angle1) > math.radians(PARAMETERS['maximum_angle_deg']):
                reason = 'curvature_or_side_view_limit'; break
            center = predicted+float(np.clip(.35*residual, -.15*gap, .15*gap))
            gap = .85*gap+.15*float(match[1]-match[0]); measured_pair = match
        points.append([x, center]); widths.append(gap/math.sqrt(1+slope*slope))
        pairs.append(measured_pair.tolist()); observed.append(match is not None)
    # Do not publish a geometric tail after the last actual paired boundary.
    end = max((i+1 for i,v in enumerate(observed) if v), default=1)
    return np.asarray(points[:end]), np.asarray(widths[:end]), pairs[:end], observed[:end], reason


def _appearance(fields, points, widths):
    if len(points) < 2:
        return np.zeros(len(points)), np.zeros(len(points)), np.zeros(len(points))
    tangent = np.gradient(points, axis=0); tangent /= np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-6)
    normal = np.c_[-tangent[:,1],tangent[:,0]]
    offsets = np.array([-1.,-.55,-.25,0,.25,.55,1.])
    q = points[:,None]+normal[:,None]*widths[:,None,None]*offsets[None,:,None]
    raw = _sample(fields['gray'],q).reshape(-1,len(offsets))
    heat = _sample(fields['heat'],q).reshape(-1,len(offsets))
    center = raw[:,2:5].mean(1); outside = (raw[:,0]+raw[:,-1])*.5
    polarity = np.where(center>=outside,1.,-1.)
    pair = np.minimum((center-raw[:,0])*polarity,(center-raw[:,-1])*polarity)
    raw_score = np.clip(pair/.08,0,1)
    heat_shape = np.clip(np.ptp(heat,axis=1)/.18,0,1)
    heat_level = np.max(heat,axis=1)
    return raw_score, heat_shape, heat_level


def propose_f3_tails(fields, last_points_xy, tangent_xy, width_px):
    """Return up to three REVIEW-ONLY tails in the unchanged field coordinates.

    Required fields: gray,heat,merged,surface; inferred is optional. Last points
    are the selected existing path's last about80 pixels; no target annotation.
    Returned points start at its endpoint, with explicit unsupported connectors.
    F3 support and raw-image support are separate; neither proves same identity.
    """
    width = float(width_px); tail = np.asarray(last_points_xy, float).reshape(-1,2)
    tangent = np.asarray(tangent_xy, float); tangent /= max(float(np.linalg.norm(tangent)),1e-8)
    merged = np.asarray(fields['merged'])>0; surface = np.asarray(fields['surface'])
    out = {'candidates': [], 'diagnostics': {'parameters': dict(PARAMETERS), 'same_identity_confirmed': False,
           'F3_is_candidate_evidence_not_truth': True, 'existing_path_modified': False}}
    if len(tail)<2 or width<=1 or not np.isfinite(np.r_[tail.ravel(),tangent,width]).all() or abs(tangent[0])<.35:
        out['diagnostics']['reason']='invalid_or_near_vertical_entry'; return out
    if any(np.asarray(fields[key]).shape!=merged.shape for key in ('gray','heat','surface')):
        raise ValueError('F3 and image fields must have equal local coordinate shape')
    origin=tail[-1]; direction=1. if tangent[0]>0 else -1.; slope=tangent[1]/tangent[0]
    initials=_initial_pairs(merged,surface,origin,slope,direction,width)
    out['diagnostics']['initial_pair_count']=len(initials)
    considered=[]
    for initial in initials:
        points,widths,pairs,observed,reason=_follow(merged,surface,initial,direction,width)
        if len(points)<3 or abs(points[-1,0]-points[0,0])<PARAMETERS['minimum_return_span_widths']*width:
            continue
        bridge=_connector(origin,tangent,points[0],initial['slope'],direction,width)
        if bridge is None or (_sample(surface,bridge[1:])<=0).any():
            continue
        # Same ordered pair predicted back to the entry is one lane family.
        intercept=float(points[0,1]-initial['slope']*(points[0,0]-origin[0]))
        if any(abs(intercept-c['intercept'])<.55*np.median(widths) for c in considered):
            continue
        bridge_count=len(bridge)-1
        all_points=np.vstack([bridge[:-1],points]); all_width=np.r_[np.linspace(width,widths[0],len(bridge))[:-1],widths]
        delta=np.diff(all_points,axis=0); segment_lengths=np.linalg.norm(delta,axis=1)
        angles=np.unwrap(np.arctan2(delta[:,1],delta[:,0]))
        turns=abs(np.diff(angles))/np.maximum((segment_lengths[:-1]+segment_lengths[1:])*.5,1e-6)*width
        max_turn=float(np.degrees(turns).max()) if len(turns) else 0.
        # Check the delivered polyline itself as well as the internal motion
        # predictor, including the bridge-to-F3 seam. Raster jitter may violate
        # this limit even when each predictor angle update was bounded.
        if max_turn>PARAMETERS['maximum_turn_deg_per_width']:
            continue
        obs=np.r_[np.zeros(bridge_count,bool),np.asarray(observed,bool)]
        raw,heat,level=_appearance(fields,all_points,all_width)
        inferred=_sample(fields.get('inferred',np.zeros_like(merged)),all_points)>0
        supported=(raw>=.20)&(heat>=.12)&obs&~inferred
        rank=float(np.mean(.55*raw[bridge_count:]+.35*heat[bridge_count:]+.10*level[bridge_count:]))
        candidate=dict(points_xy=all_points.tolist(),width_px=all_width.tolist(),
            f3_pair_supported=obs.tolist(),image_supported=supported.tolist(),
            raw_ribbon_score=raw.tolist(),heat_shape_score=heat.tolist(),heat_level=level.tolist(),
            f3_inferred_at_center=inferred.tolist(),connector_point_count=bridge_count,
            track_upper_lower_y=pairs,rank_score=rank,termination=reason,
            length_px=float(np.linalg.norm(np.diff(all_points,axis=0),axis=1).sum()),
            optical_reference_width_px=width,initial_lane_width_px=float(widths[0]),
            maximum_polyline_turn_deg_per_optical_width=max_turn,
            confidence='LOW_REVIEW_ONLY_F3_LANE_IDENTITY_UNKNOWN',same_identity_confirmed=False,
            raw_image_supported_fraction=float(supported.mean()),F3_pair_fraction=float(obs.mean()))
        if candidate['F3_pair_fraction']<PARAMETERS['minimum_pair_fraction']:
            continue
        considered.append({'intercept':intercept,'candidate':candidate})
        if len(considered)>=PARAMETERS['maximum_candidates']:
            break
    out['candidates']=sorted([c['candidate'] for c in considered],key=lambda c:-c['rank_score'])
    out['diagnostics']['reason']='review_candidates_only' if considered else 'no_supported_ordered_pair_continuation'
    out['diagnostics']['candidate_count']=len(out['candidates'])
    return out
