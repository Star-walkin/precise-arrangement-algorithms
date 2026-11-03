"""Image-only scene geometry for a conditioned incoming-rope experiment.

The yellow/gray interface is an observed spool-edge candidate, never a label
endpoint. Stages are visual proxies, not physical contact measurements.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PARAMETERS = dict(seed_length_px=120., contact_band_px=96.,
                  ray_interface_tolerance_px=48., edge_band_px=96.,
                  minimum_arc_span_px=900., far_side_fraction=.55,
                  far_edge_span_fraction=.12)


def short_seed(points, length_px=120.):
    """Keep the initial arc length only; never expose later seed/path hints."""
    p = np.asarray(points, float).reshape(-1, 2)
    if len(p) < 2:
        raise ValueError('A nondegenerate free-entry segment is required')
    out = [p[0]]; remaining = float(length_px)
    for a, b in zip(p[:-1], p[1:]):
        d = float(np.linalg.norm(b-a))
        if d <= 1e-8:
            continue
        if d >= remaining:
            out.append(a+(b-a)*remaining/d)
            break
        out.append(b); remaining -= d
    out = np.asarray(out)
    if len(out) < 2 or np.linalg.norm(out[-1]-out[0]) < 1:
        raise ValueError('Degenerate free-entry direction')
    return out


def build_scene(image_bgr, entry_xy, *, arc=None):
    """Derive geometry from original image and entry only; no target labels."""
    from vision_geometry_lab.arc_detector import detect_arc
    p = short_seed(entry_xy, PARAMETERS['seed_length_px'])
    origin = p[0]; direction = p[-1]-origin
    direction /= np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    if arc is None:
        arc = detect_arc(image_bgr)
    segments = [np.asarray(s, float).reshape(-1, 2)
                for s in arc.get('local_boundary_segments', []) if len(s) >= 2]
    points = np.concatenate(segments) if segments else np.zeros((0, 2))
    h, w = image_bgr.shape[:2]
    contact = {'status': 'unknown', 'xy': None, 't_from_entry_px': None,
               'source': 'first observed yellow/gray interface near forward entry ray'}
    target = np.zeros((0, 2)); reason = 'insufficient observed interface support'
    if len(points):
        t = (points-origin)@direction; distance = np.abs((points-origin)@normal)
        eligible = (t > PARAMETERS['seed_length_px']) & (distance <= PARAMETERS['ray_interface_tolerance_px'])
        if eligible.any():
            candidates = np.flatnonzero(eligible)
            k = candidates[np.argmin(t[candidates])]
            contact.update(status='visual_proxy', xy=points[k].tolist(),
                           t_from_entry_px=float(t[k]))
        # The distal side must be actually observed on the opposite half of
        # the frame. A short visible arc fragment cannot define a far endpoint.
        span = float(points[:, 0].max()-points[:, 0].min())
        far_is_right = direction[0] > 0
        crosses_far_half = (points[:, 0].max() > w*PARAMETERS['far_side_fraction']
                           if far_is_right else points[:, 0].min() < w*(1-PARAMETERS['far_side_fraction']))
        if span >= PARAMETERS['minimum_arc_span_px'] and crosses_far_half:
            cutoff = float(t.max()-PARAMETERS['far_edge_span_fraction']*(t.max()-t.min()))
            target = points[t >= cutoff]
            reason = 'observed distal yellow-flange/gray-rope interface flank; target identity unconfirmed'
    return dict(schema_version=1, original_shape_hw=[h,w],
                entry_short_xy=p.tolist(), entry_origin_xy=origin.tolist(),
                entry_direction_xy=direction.tolist(),
                edge_points_xy=points.tolist(), edge_segments_xy=[s.tolist() for s in segments],
                target_edge_points_xy=target.tolist(), target_edge_reason=reason,
                contact_proxy=contact, physical_contact_confirmed=False,
                target_identity_at_edge_confirmed=False,
                actual_image_boundary_xyxy=[0,0,w-1,h-1],
                patch_boundary_is_target=False,
                edge_source='arc_detector.detect_arc local_boundary_segments; original RGB only',
                arc_status=arc.get('status'), arc_valid=bool(arc.get('valid',False)),
                parameters=PARAMETERS.copy(),
                stage_definition={'1':'before visual interface proximity',
                  '2':'within +/-96 original px along entry direction around first interface proximity',
                  '3':'after transition, interpreted only as attached-side visual region',
                  '4':'within 96 px of observed distal interface flank, after transition'},
                limitations=['visual stages are not physical contact labels',
                  'flange/rope interface candidate is not independently verified target termination',
                  'missing arc sections stay missing; no fitted/extrapolated endpoint',
                  'unknown contact defaults to stage 1 for bookkeeping and is explicitly reported'])


def stage_at_points(scene, points):
    """Return stages 1..4 for original-coordinate points, shape (...,2)."""
    p = np.asarray(points, float); shape = p.shape[:-1]; p = p.reshape(-1, 2)
    stage = np.ones(len(p), np.uint8)
    origin = np.asarray(scene['entry_origin_xy']); direction = np.asarray(scene['entry_direction_xy'])
    t = (p-origin)@direction
    contact_t = scene['contact_proxy'].get('t_from_entry_px')
    if contact_t is not None:
        band = scene['parameters']['contact_band_px']
        stage[t >= contact_t-band] = 2
        stage[t > contact_t+band] = 3
        edge = np.asarray(scene.get('target_edge_points_xy', []), float).reshape(-1, 2)
        if len(edge):
            # Bounded chunks avoid a full image x all-edge-points allocation.
            for lo in range(0,len(p),16384):
                q=p[lo:lo+16384]
                d2=((q[:,None,:]-edge[None,:,:])**2).sum(2).min(1)
                take=(d2<=scene['parameters']['edge_band_px']**2)&(t[lo:lo+len(q)]>contact_t+band)
                chunk=stage[lo:lo+len(q)];chunk[take]=4
    return stage.reshape(shape)
