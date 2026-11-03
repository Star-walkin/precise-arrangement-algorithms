"""V28 finite image-only candidates and frozen-pool scoring. No label or device IO."""
from __future__ import annotations
import hashlib
import json
import math
import cv2
import numpy as np


DEFAULT_CONFIG = dict(
    max_candidates=8, max_old_candidates=6, max_crossing_candidates=2,
    search_length_widths=90., search_length_max_px=2400., lateral_widths=18.,
    path_step_px=4., transition_max_widths=12., minimum_unique_separation_widths=.45,
    new_trace_step_widths=.6, new_trace_beam=24, new_trace_max_gap_widths=12.,
    distinguishing_separation_widths=.65, distinguishing_start_widths=4.,
    distinguishing_budget_widths=20.,
    g0_weights=dict(distance=.45, direction=.55),
    g1_weights=dict(g0=.40, curvature=.18, width=.16, arrangement=.08, extrapolation=.18),
    g2_weights=dict(g1=.38, bilateral=.32, profile=.15, interruption=.15),
    minimum_available_fraction=.35, minimum_bilateral_fraction=.30,
    minimum_distinguishing_points=12, minimum_score_margin=.035,
    minimum_rank_stability=.60, maximum_geometry_cost=.85,
    arrangement_minimum_separation_widths=.45,
    arrangement_scope='same-phase local arcs after old-row join; new crossing paths exempt',
    perturbations=[dict(normal_widths=0., angle_deg=0., width_factor=1.),
                   dict(normal_widths=.10, angle_deg=.6, width_factor=1.10),
                   dict(normal_widths=-.10, angle_deg=-.6, width_factor=.90),
                   dict(normal_widths=.10, angle_deg=-.6, width_factor=.90),
                   dict(normal_widths=-.10, angle_deg=.6, width_factor=1.10)],
    perturbation_kind='predeclared development sensitivity, NOT confidence intervals',
    missing_evidence_cost=.5, crossing_depth_order='UNKNOWN',
)


def pool_hash(pool):
    return hashlib.sha256(json.dumps(pool, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


def unit(v):
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-9)


def resample(points, step=4.):
    p = np.asarray(points, float)
    p = p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-7]]
    if len(p) < 2:
        return p
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    q = np.unique(np.r_[np.arange(0., s[-1], step), s[-1]])
    return np.column_stack([np.interp(q, s, p[:, k]) for k in (0, 1)])


def tangents(p):
    p = np.asarray(p, float)
    t = np.gradient(p, axis=0)
    if len(t) >= 7:
        t = cv2.GaussianBlur(t.astype(np.float32), (1, 7), 0).astype(float)
    return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)


def sample(image, xy):
    return cv2.remap(image, np.asarray(xy[..., 0], np.float32),
                     np.asarray(xy[..., 1], np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


def profile_evidence(image_bgr, points_xy, widths_px):
    """Two opposite normal gradients on the untouched image, either polarity.

    Dark, saturated, flat and out-of-frame sections remain unavailable. A failed
    edge never becomes a measured boundary; all geometric envelope points are
    separately marked as proposals. These are algorithmic supports, not identity.
    """
    p = np.asarray(points_xy, float)
    widths = np.broadcast_to(np.asarray(widths_px, float), (len(p),)).copy()
    t = tangents(p); n = np.column_stack([-t[:, 1], t[:, 0]])
    offsets = np.linspace(-1.15, 1.15, 47)
    along = np.linspace(-.30, .30, 5)
    xy = (p[:, None, None, :] + n[:, None, None, :] * widths[:, None, None, None] * offsets[None, :, None, None]
          + t[:, None, None, :] * widths[:, None, None, None] * along[None, None, :, None])
    # Remap is limited to 2-D grids; preserve row/profile/tangential indexing.
    raw = sample(image_bgr, xy.reshape(len(p), -1, 2)).reshape(len(p), len(offsets), len(along), 3)
    gray = np.median(raw.min(axis=3), axis=2).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (5, 1), .7)
    step = widths * (offsets[1] - offsets[0])
    grad = np.gradient(smooth, axis=1) / step[:, None]
    li = np.flatnonzero((offsets >= -.72) & (offsets <= -.28))
    ri = np.flatnonzero((offsets >= .28) & (offsets <= .72))
    lidx = li[np.argmax(np.abs(grad[:, li]), axis=1)]
    ridx = ri[np.argmax(np.abs(grad[:, ri]), axis=1)]
    rows = np.arange(len(p)); gl = grad[rows, lidx]; gr = grad[rows, ridx]
    contrast = np.percentile(smooth, 90, axis=1) - np.percentile(smooth, 10, axis=1)
    inside = ((xy[..., 0] >= 1) & (xy[..., 0] < image_bgr.shape[1]-2) &
              (xy[..., 1] >= 1) & (xy[..., 1] < image_bgr.shape[0]-2)).all(axis=(1, 2))
    saturated = (raw.max(axis=3) >= 254).mean(axis=(1, 2))
    dark = np.percentile(smooth, 90, axis=1) < 12
    available = inside & (saturated <= .20) & (~dark) & (contrast >= 8.)
    strength = np.minimum(abs(gl), abs(gr))
    balance = strength / np.maximum(np.maximum(abs(gl), abs(gr)), 1e-6)
    measured_width = (offsets[ridx]-offsets[lidx])*widths
    opposite = gl*gr < 0
    supported = available & opposite & (strength >= 1.2) & (balance >= .2) & (measured_width >= .65*widths) & (measured_width <= 1.35*widths)
    bilateral = np.where(available, np.clip(strength/3., 0, 1)*opposite*np.clip(balance/.4, 0, 1), 0.)
    profile = np.where(available, np.clip(contrast/35., 0, 1)*opposite, 0.)
    left = p + n*(offsets[lidx]*widths)[:, None]
    right = p + n*(offsets[ridx]*widths)[:, None]
    return dict(supported=supported.tolist(), available=available.tolist(),
                inferred=(available & ~supported).tolist(), unavailable=(~available).tolist(),
                left_xy=left.tolist(), right_xy=right.tolist(),
                measured_widths_px=[float(w) if ok else None for w, ok in zip(measured_width, supported)],
                bilateral=bilateral.tolist(), profile=profile.tolist(),
                saturated_fraction=saturated.tolist(), contrast=contrast.tolist(),
                boundary_status=['image_pair' if x else 'unverified_proposal' for x in supported],
                source='untouched original BGR normal profiles; no completion pixels',
                polarity=['bright_body' if a > 0 else 'dark_body_or_gap' for a in gl],
                depth_order='UNKNOWN')


def bezier(a, b, da, db, count=40):
    length = np.linalg.norm(b-a)
    controls = np.asarray([a, a+da*length/3., b-db*length/3., b])
    s = np.linspace(0, 1, count)[:, None]
    return (1-s)**3*controls[0]+3*(1-s)**2*s*controls[1]+3*(1-s)*s*s*controls[2]+s**3*controls[3]


def _crossing_traces(image, origin, d, n, width, length, config):
    """Bounded incoming-direction beam; it may cross old rows without joining.

    Identical fixed width/step budget for both alternatives; original image only.
    No old-row order constraint, no expected target endpoint, no gap inpainting.
    """
    du = width*config['new_trace_step_widths']
    u = np.arange(0., length+du*.5, du)
    dv = width*.25
    v = np.arange(-config['lateral_widths']*width, config['lateral_widths']*width+dv*.5, dv)
    xy = origin + u[:, None, None]*d + v[None, :, None]*n
    gray = image.min(axis=2)
    strip = sample(gray, xy).astype(np.float32)
    # Whole-rope envelope scale; a peak by itself does not establish a rope.
    smooth = cv2.GaussianBlur(strip, (3, 3), .65)
    grad = np.gradient(smooth, dv, axis=1)
    gl = np.roll(grad, 2, axis=1); gr = np.roll(grad, -2, axis=1)
    pair = np.clip(np.minimum(abs(gl), abs(gr))/3., 0, 1)*(gl*gr < 0)
    contrast = abs(smooth-.5*(np.roll(smooth, 4, axis=1)+np.roll(smooth, -4, axis=1)))
    valid = ((xy[..., 0] > width) & (xy[..., 0] < image.shape[1]-width) &
             (xy[..., 1] > width) & (xy[..., 1] < image.shape[0]-width) & (strip < 250) & (strip > 10))
    evidence = (.7*pair+.3*np.clip(contrast/20., 0, 1))*valid
    mid = int(np.argmin(abs(v)))
    beam = [(0., (mid,), 0, 0)]
    completed = []
    maxgap = int(config['new_trace_max_gap_widths']/config['new_trace_step_widths'])
    for i in range(1, len(u)):
        nxt = []
        for cost, path, slope, gap in beam:
            for delta in (-2, -1, 0, 1, 2):
                j = path[-1]+delta
                if not 3 <= j < len(v)-3 or not valid[i, j]:
                    continue
                ev = float(evidence[i, j]); newgap = gap+1 if ev < .20 else 0
                if newgap > maxgap:
                    continue
                # Beginning remains attached to the observed incoming direction.
                if i < 4 and abs(v[j]) > .65*width:
                    continue
                score = cost + (1-ev)*.65 + .10*abs(delta-slope) + .05*abs(delta)
                nxt.append((score, path+(j,), delta, newgap))
        if not nxt:
            break
        # Keep spatial/directional alternatives; do not let duplicate histories
        # consume the entire finite beam budget.
        keys = set(); beam = []
        for state in sorted(nxt, key=lambda a:(a[0], a[1])):
            key = (state[1][-1], state[2])
            if key in keys:
                continue
            keys.add(key); beam.append(state)
            if len(beam) == config['new_trace_beam']:
                break
        if i >= 12:
            completed = beam
    result = []
    for cost, path, slope, gap in sorted(completed, key=lambda a:(a[0]/len(a[1]), a[1])):
        ids = np.asarray(path); points = xy[np.arange(len(ids)), ids]
        ev = evidence[np.arange(len(ids)), ids]
        if len(points) < 12 or np.mean(ev[4:] >= .20) < .20:
            continue
        if any(np.mean(np.linalg.norm(points[:min(len(points),len(q))]-q[:min(len(points),len(q))], axis=1)) < width*.6 for q in result):
            continue
        result.append(points)
        if len(result) == config['max_crossing_candidates']:
            break
    return result


def generate_pool(image_bgr, free_seed, width_px, structure, config=None):
    cfg = config or DEFAULT_CONFIG
    seed = np.asarray(free_seed, float); origin = seed[-1]
    d = unit(seed[-1]-seed[0]); n = np.array([-d[1], d[0]])
    width = float(width_px); length = min(width*cfg['search_length_widths'], cfg['search_length_max_px'])
    rows = []; attempted = []; discarded = []
    for s in structure.get('structures', [])[:cfg['max_old_candidates']]:
        p = np.asarray(s['points_xy'], float)
        if len(p) < 3:
            continue
        u = (p-origin)@d
        if u[-1] < u[0]:
            p = p[::-1]; u = u[::-1]
        keep = (u >= width*3) & (u <= length)
        p = p[keep]; u = u[keep]
        if len(p) < 3:
            discarded.append(dict(structure_id=s['id'], reason='no_forward_arc'))
            continue
        # A long upstream interval remains explicit geometric extrapolation.
        # Only the final bounded transition has cubic freedom.
        join = p[0]; target_d = unit(p[min(4, len(p)-1)]-p[0])
        lead_length = max(0., float(u[0])-cfg['transition_max_widths']*width)
        lead = origin+d*lead_length
        transition = bezier(lead, join, d, target_d)
        line = np.vstack([origin, lead, transition, p[1:]])
        rows.append(dict(id='O'+str(len(rows)+1), kind='old_row_connection',
                         structure_id=s['id'], raw_points=line, join_xy=join.tolist(),
                         structure_phase=s.get('polarity','unknown'),
                         inferred_prefix_px=lead_length, generator='bounded cubic + image local arc'))
    crossing = _crossing_traces(image_bgr, origin, d, n, width, length, cfg)
    for i, p in enumerate(crossing):
        rows.append(dict(id='X'+str(i+1), kind='new_segment_may_cross_old_rows',
                         structure_id=None, raw_points=p, join_xy=p[min(20, len(p)-1)].tolist(),
                         inferred_prefix_px=0., generator='incoming original-image paired-profile beam'))
    candidates = []
    for row in rows:
        p = resample(row.pop('raw_points'), cfg['path_step_px'])
        inside = ((p[:,0]>=1)&(p[:,0]<image_bgr.shape[1]-2)&(p[:,1]>=1)&(p[:,1]<image_bgr.shape[0]-2))
        if not inside.all():
            stop = np.flatnonzero(~inside)[0]; p = p[:stop]
        if len(p) < 3:
            continue
        duplicate = None
        for c in candidates:
            a = np.asarray(c['points_xy']); m = min(len(a), len(p))
            if abs(len(a)-len(p))*cfg['path_step_px'] < width*3 and np.mean(np.linalg.norm(a[:m]-p[:m],axis=1)) < width*cfg['minimum_unique_separation_widths']:
                duplicate = c['id']; break
        if duplicate:
            discarded.append(dict(id=row['id'], reason='duplicate', equivalent_to=duplicate)); continue
        widths = np.full(len(p), width)
        ev = profile_evidence(image_bgr, p, widths)
        row.update(points_xy=p.tolist(), widths_px=widths.tolist(), evidence=ev,
                   supported=ev['supported'], inferred=ev['inferred'], unavailable=ev['unavailable'],
                   left_xy=ev['left_xy'], right_xy=ev['right_xy'],
                   physical_identity_confirmed=False, depth_order='UNKNOWN')
        candidates.append(row)
    truncated = len(candidates) > cfg['max_candidates']
    return dict(schema='v28_pool_1', coordinate_system='full_original_image_xy',
                image_shape_hw=list(image_bgr.shape[:2]), origin_xy=origin.tolist(),
                direction_xy=d.tolist(), normal_xy=n.tolist(), free_seed=seed.tolist(),
                width_px=width, candidates=candidates[:cfg['max_candidates']],
                local_arrangement=[dict(id=s['id'],points_xy=s['points_xy'],
                                        phase=s.get('polarity','unknown')) for s in structure.get('structures',[])],
                candidate_cap=cfg['max_candidates'], truncated=truncated,
                discarded=discarded, crossing_proposals=len(crossing),
                crossing_missing_reason=None if crossing else 'no incoming image-supported beam within fixed budget',
                input_uses_labels=False, geometry_completion_is_evidence=False,
                uncertainty='UNKNOWN is an output, not a forced old-row assignment')


def distinguishing_masks(pool, cfg):
    candidates = pool['candidates']; width=pool['width_px']
    origin=np.asarray(pool['origin_xy']); d=np.asarray(pool['direction_xy']); n=np.asarray(pool['normal_xy'])
    paths = [np.asarray(c['points_xy']) for c in candidates]
    if not paths:
        return [], []
    umax = max(float(np.max((p-origin)@d)) for p in paths)
    grid = np.arange(width*cfg['distinguishing_start_widths'], umax+1., max(2., width*.25))
    vals = []
    for p in paths:
        u=(p-origin)@d; v=(p-origin)@n; order=np.argsort(u)
        vals.append(np.interp(grid,u[order],v[order],left=np.nan,right=np.nan))
    values=np.asarray(vals); eligible=np.isfinite(values).sum(axis=0)>=2
    spread=np.zeros(len(grid))
    for j in np.flatnonzero(eligible):
        z=values[:,j]; z=z[np.isfinite(z)]; spread[j]=np.max(z)-np.min(z)
    different=eligible & (spread>=width*cfg['distinguishing_separation_widths'])
    # Fixed first 20 widths of genuinely separated region. Common long straight
    # prefixes cannot dilute the critical decision; the GT bend is never read.
    chosen=np.flatnonzero(different)
    budget=int(cfg['distinguishing_budget_widths']*width/max(2.,width*.25))
    chosen=chosen[:budget]; selected=np.zeros(len(grid),bool); selected[chosen]=True
    masks=[]
    for p in paths:
        u=(p-origin)@d
        masks.append(np.interp(u,grid,selected.astype(float),left=0,right=0)>.5 if len(grid) else np.zeros(len(p),bool))
    return masks, grid[selected].tolist()


def _costs(pool, masks, evidence, cfg, perturb):
    origin=np.asarray(pool['origin_xy']); d=np.asarray(pool['direction_xy']); n=np.asarray(pool['normal_xy'])
    width=pool['width_px']*perturb['width_factor']; origin=origin+n*pool['width_px']*perturb['normal_widths']
    a=math.radians(perturb['angle_deg']); d=np.array([d[0]*math.cos(a)-d[1]*math.sin(a),d[0]*math.sin(a)+d[1]*math.cos(a)])
    out=[]
    for c, mask, ev in zip(pool['candidates'],masks,evidence):
        p=np.asarray(c['points_xy']); t=tangents(p); join=np.asarray(c['join_xy'])
        distance=float(np.clip(np.linalg.norm(join-origin)/(90*width),0,1))
        ji=int(np.argmin(np.linalg.norm(p-join,axis=1)))
        direction=float(np.arccos(np.clip(t[ji]@d,-1,1))/math.pi)
        g0=cfg['g0_weights']['distance']*distance+cfg['g0_weights']['direction']*direction
        turn=np.arccos(np.clip(np.sum(t[1:]*t[:-1],axis=1),-1,1))
        curvature=float(np.clip(np.percentile(turn,90)*width/cfg['path_step_px'],0,1))
        measured=np.asarray([x if x is not None else np.nan for x in c['evidence']['measured_widths_px']])
        good=np.isfinite(measured)&mask
        width_cost=float(np.clip(np.median(abs(measured[good]/width-1)),0,1)) if good.any() else .5
        consecutive=good[1:]&good[:-1]
        if consecutive.any():
            continuity=float(np.clip(np.median(abs(np.diff(measured)[consecutive]))/(.25*width),0,1))
            width_cost=.7*width_cost+.3*continuity
        inferred=~np.asarray(c['supported'],bool)
        extrapolation=float(np.mean(inferred[mask])) if mask.any() else 1.
        u=(p-origin)@d
        arrangement=float(np.clip(np.mean(np.diff(u)<-1)*5.,0,1))
        # Same-phase local rows should retain their transverse order after an
        # old-row attachment. Dark half-phase alternatives are not treated as
        # neighbouring whole ropes. New-segment paths may cross these rows.
        relation_checks=[]
        if c['kind']=='old_row_connection':
            v=(p-origin)@n; ju=float((join-origin)@d)
            for neighbour in pool.get('local_arrangement',[]):
                if neighbour['id']==c.get('structure_id') or neighbour['phase']!=c.get('structure_phase'):
                    continue
                q=np.asarray(neighbour['points_xy']); qu=(q-origin)@d; qv=(q-origin)@n
                order=np.argsort(qu)
                active=(u>=ju+width)&(u>=qu.min())&(u<=qu.max())
                if active.sum()<4:
                    continue
                delta=v[active]-np.interp(u[active],qu[order],qv[order])
                collision=np.mean(abs(delta)<cfg['arrangement_minimum_separation_widths']*width)
                crossing=bool(delta.min() < -.5*width and delta.max() > .5*width)
                relation_checks.append(float(np.clip(collision+.5*crossing,0,1)))
            if relation_checks:
                arrangement=max(arrangement,float(np.mean(relation_checks)))
        g1=(cfg['g1_weights']['g0']*g0+cfg['g1_weights']['curvature']*curvature+
            cfg['g1_weights']['width']*width_cost+cfg['g1_weights']['arrangement']*arrangement+
            cfg['g1_weights']['extrapolation']*extrapolation)
        available=np.asarray(ev['available'],bool)&mask
        usable_fraction=float(available.sum()/max(1,mask.sum()))
        support=np.asarray(ev['supported'],bool)
        bilateral=float(np.mean(np.asarray(ev['bilateral'])[available])) if available.any() else None
        profile=float(np.mean(np.asarray(ev['profile'])[available])) if available.any() else None
        # Only visible, usable sections may be evidence of a broken pair. Missing
        # or erased boundaries remain unknown, with a neutral cost plus rejection.
        broken=float(np.mean(~support[available])) if available.any() else None
        g2=cfg['g2_weights']['g1']*g1
        g2+=cfg['g2_weights']['bilateral']*((1-bilateral) if bilateral is not None else cfg['missing_evidence_cost'])
        g2+=cfg['g2_weights']['profile']*((1-profile) if profile is not None else cfg['missing_evidence_cost'])
        g2+=cfg['g2_weights']['interruption']*(broken if broken is not None else cfg['missing_evidence_cost'])
        out.append(dict(id=c['id'],G0=g0,G1=g1,G2=g2,
                        components=dict(distance=distance,direction=direction,curvature=curvature,
                          width=width_cost,arrangement=arrangement,extrapolation=extrapolation,
                          bilateral=bilateral,profile=profile,interruption=broken,
                          available_fraction=usable_fraction,
                          bilateral_fraction=float(np.mean(support[mask])) if mask.any() else 0.,
                          distinguishing_points=int(mask.sum()),depth_order='UNKNOWN')))
    return out


def score_pool(image_bgr, pool, config=None):
    cfg=config or DEFAULT_CONFIG; before=pool_hash(pool)
    masks, grid=distinguishing_masks(pool,cfg)
    trials=[]
    for perturb in cfg['perturbations']:
        ev=[]
        for c in pool['candidates']:
            p=np.asarray(c['points_xy']); t=tangents(p); n=np.column_stack([-t[:,1],t[:,0]])
            shifted=p+n*pool['width_px']*perturb['normal_widths']
            ev.append(profile_evidence(image_bgr,shifted,np.asarray(c['widths_px'])*perturb['width_factor']))
        trials.append(_costs(pool,masks,ev,cfg,perturb))
    results={}
    for group in ('G0','G1','G2'):
        ordered=sorted(trials[0],key=lambda r:(r[group],r['id']))
        if not ordered:
            results[group]=dict(best_id=None,selected_id=None,status='UNKNOWN',reason=['empty_candidate_pool'],ordered=[],margin=None,stability=None,pool_sha256=before)
            continue
        best=ordered[0]; second=ordered[1] if len(ordered)>1 else None
        margin=float(second[group]-best[group]) if second else None
        winners=[min(t,key=lambda r:(r[group],r['id']))['id'] for t in trials]
        stability=winners.count(best['id'])/len(winners)
        reasons=[]; state='SINGLE_CANDIDATE'
        if best[group]>cfg['maximum_geometry_cost']:
            reasons.append('excessive_cost'); state='UNKNOWN'
        if second and margin<cfg['minimum_score_margin']:
            reasons.append('near_tied_candidates'); state='MULTI_CANDIDATE'
        if stability<cfg['minimum_rank_stability']:
            reasons.append('perturbation_changes_ranking'); state='MULTI_CANDIDATE'
        if group=='G2':
            comp=best['components']
            if comp['distinguishing_points']<cfg['minimum_distinguishing_points']:
                reasons.append('insufficient_distinguishing_region'); state='UNKNOWN'
            if comp['available_fraction']<cfg['minimum_available_fraction']:
                reasons.append('insufficient_usable_original_evidence'); state='UNKNOWN'
            if comp['bilateral_fraction']<cfg['minimum_bilateral_fraction']:
                reasons.append('insufficient_bilateral_connection_support'); state='UNKNOWN'
        results[group]=dict(best_id=best['id'],selected_id=best['id'] if state=='SINGLE_CANDIDATE' else None,
                            status=state,reason=reasons or ['lowest_fixed_cost_within_gates'],
                            ordered=[dict(id=r['id'],cost=r[group],components=r['components']) for r in ordered],
                            margin=margin,stability=stability,perturbed_best_ids=winners,
                            pool_sha256=before,distinguishing_u_px=grid,
                            distinguishing_masks={c['id']:m.tolist() for c,m in zip(pool['candidates'],masks)},
                            score_is_identity_probability=False,physical_identity_confirmed=False)
    assert pool_hash(pool)==before, 'scoring must not mutate or refit the frozen pool'
    return results
