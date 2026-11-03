"""Read-only, image-evidence candidates for a free rope entering a wound surface.

This does NOT identify a new winding from repeated rope texture. A free-line
anchor gives a starting direction; a multi-hypothesis ridge tracer then follows
the CURRENT image. The flange arc never contributes to the path score. It is
used to choose the entry-facing end and, afterwards, to compare with directly
observed boundary segments. No ellipse, translated boundary, time rule, or
actuator/winding-state decision is used.
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np


def _empty(reason, **diagnostics):
    return dict(valid=False, identity_confirmed=False, status="unavailable",
                reason=reason, candidates=[], contact_comparison=dict(valid=False, reason="no_supported_curve"),
                trajectory_confirmed=False, edge_contact_confirmed=False, physical_contact_confirmed=False,
                last_turn_confirmed=False, new_layer_confirmed=False,
                force_command=None, diagnostics=diagnostics)


def _points(value):
    try:
        p = np.asarray(value, float)
        if p.ndim == 2 and p.shape[1] == 2 and np.isfinite(p).all():
            return p
    except (ValueError, TypeError):
        pass
    return np.empty((0, 2), float)


def _sample(image, x, y):
    return cv2.remap(image, x.astype(np.float32), y.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def _entry(rope, arc, seeds):
    segment = _points(rope.get("observed_segment"))
    if len(segment) != 2:
        return None, "free_rope_observed_segment_required"
    n = np.asarray(rope.get("normal"), float)
    if n.shape != (2,) or not np.isfinite(n).all() or np.linalg.norm(n) < 1e-8:
        return None, "invalid_free_rope_line"
    n = n / np.linalg.norm(n)
    d = np.array([n[1], -n[0]])
    order = np.argsort(segment @ d)
    segment = segment[order]
    if np.linalg.norm(segment[1] - segment[0]) < 25:
        return None, "free_rope_anchor_too_short"
    observed = _points(arc.get("points", [])) if arc else np.empty((0, 2))
    target = seeds if len(seeds) else observed
    if not len(target):
        return None, "entry_direction_requires_observed_arc_or_seed"
    distance = [float(np.min(np.linalg.norm(target - point, axis=1))) for point in segment]
    if abs(distance[0] - distance[1]) < max(15., .08 * np.linalg.norm(segment[1] - segment[0])):
        return None, "entry_direction_ambiguous"
    which = int(np.argmin(distance))
    advance = d if which == 1 else -d
    return (segment[which], advance), "manual_seed" if len(seeds) else "observed_arc_proximity_only"


def _features(image, origin, d, n, width, resolution, length, guided=False):
    half_height = max(width * 9., length * .24)
    u = np.arange(0., length + resolution, resolution)
    v = np.arange(-half_height, half_height + resolution, resolution)
    xy = origin + u[None, :, None] * d + v[:, None, None] * n
    strip = _sample(image, xy[..., 0], xy[..., 1]).astype(np.float32)
    white = strip.min(axis=2)
    sigma = max(.65, width / resolution * .16)
    # Suppress the short diagonal bright facets of a braided rope before
    # searching for its longer bundle-scale ridge. Otherwise a tracer readily
    # follows a strand highlight diagonally across several wound ropes.
    smooth = cv2.GaussianBlur(white, (0, 0), sigmaX=max(.6, width / resolution * .65), sigmaY=sigma)
    xx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=3) / 4.
    yy = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=3) / 4.
    cross = cv2.Sobel(smooth, cv2.CV_32F, 1, 1, ksize=3) / 4.
    delta = np.sqrt((xx - yy) ** 2 + 4 * cross ** 2)
    small_eigen = .5 * (xx + yy - delta)
    large_eigen = .5 * (xx + yy + delta)
    anisotropy = np.clip((abs(small_eigen) - abs(large_eigen)) /
                        (abs(small_eigen) + abs(large_eigen) + 1e-4), 0., 1.)
    tangent = .5 * np.arctan2(2 * cross, xx - yy)
    offset = max(2, int(round(width * .65 / resolution)))
    flank_a, flank_b = np.roll(smooth, offset, axis=0), np.roll(smooth, -offset, axis=0)
    prominence = smooth - .5 * (flank_a + flank_b)
    dynamic_range = max(25., float(np.percentile(white, 95) - np.percentile(white, 10)))
    contrast_scale = max(12., dynamic_range * .20)
    curvature_scale = max(.8, float(np.percentile(np.maximum(-small_eigen, 0), 90)))
    response = .65 * np.clip(prominence / contrast_scale, 0, 1)
    response += .35 * np.clip(-small_eigen / curvature_scale, 0, 1) * (.35 + .65 * anisotropy)
    polarity = np.ones_like(response)
    minimum_prominence = max(4., dynamic_range * .025)
    if guided:
        # Manual route hints allow a LOCAL photometric search, not an invented
        # path. The incoming bright strand can become a dark body/valley after
        # wrapping; a global bright-ridge threshold suppresses all such pixels.
        # Keep raw contrast as a floor so normalization cannot create evidence
        # on a flat/erased patch.
        local = cv2.GaussianBlur(white, (0, 0), sigmaX=max(.6, width/resolution*.30),
                                 sigmaY=max(.6, width/resolution*.11))
        mean = cv2.GaussianBlur(local, (0, 0), max(2., width/resolution*1.3))
        square = cv2.GaussianBlur(local*local, (0, 0), max(2., width/resolution*1.3))
        local_sigma = np.sqrt(np.maximum(square-mean*mean, 0.))
        signed = local - .5*(np.roll(local, offset, axis=0)+np.roll(local, -offset, axis=0))
        polarity = np.sign(signed)
        prominence = np.abs(signed)
        response = np.clip(prominence / np.maximum(2., local_sigma*.45), 0., 1.)
        response = cv2.GaussianBlur(response, (3, 3), .45)
        minimum_prominence = 1.5
    saturation = cv2.GaussianBlur((strip.min(axis=2) >= 250).astype(np.float32), (0, 0), sigma)
    margin = max(width, resolution * offset)
    valid = ((xy[..., 0] > margin) & (xy[..., 0] < image.shape[1] - margin) &
             (xy[..., 1] > margin) & (xy[..., 1] < image.shape[0] - margin))
    valid[:offset] = False
    valid[-offset:] = False
    response[~valid] = 0
    return dict(u=u, v=v, response=response, prominence=prominence,
                saturation=saturation, tangent=tangent, anisotropy=anisotropy,
                polarity=polarity, minimum_prominence=minimum_prominence)


def _column_peaks(features, column, width, resolution):
    response = features["response"][:, column]
    indices = np.flatnonzero((response[1:-1] >= response[:-2]) &
                             (response[1:-1] > response[2:]) & (response[1:-1] > .23)) + 1
    indices = [int(i) for i in indices if features["prominence"][i, column] > features["minimum_prominence"]]
    # Spatial NMS prevents one broad plateau becoming several identities.
    selected = []
    for i in sorted(indices, key=lambda j: float(response[j]), reverse=True):
        if all(abs(i - old) * resolution > width * .25 for old in selected):
            selected.append(i)
    nodes = []
    for i in sorted(selected):
        f0, f1, f2 = response[i-1:i+2]
        denominator = float(f0 - 2*f1 + f2)
        delta = float(np.clip(.5*(f0-f2)/denominator, -.5, .5)) if abs(denominator) > 1e-5 else 0.
        y = float(features["v"][i] + delta * resolution)
        siblings = sum(width * .65 < abs(features["v"][j] - y) < width * 3.
                       and response[j] > response[i] * .70 for j in selected if j != i)
        nodes.append(dict(u=float(features["u"][column]), v=y, quality=float(response[i]),
                          saturation=float(features["saturation"][i, column]),
                          tangent=float(features["tangent"][i, column]),
                          anisotropy=float(features["anisotropy"][i, column]),
                          polarity=int(features["polarity"][i, column]),
                          observed=True, siblings=int(siblings)))
    return nodes


def _trace(features, width, resolution, anchor_length, seed_uv):
    stride = max(1, int(round(width * .42 / resolution)))
    columns = list(range(1, len(features["u"]) - 1, stride))
    if len(columns) < 8:
        return []
    initial = _column_peaks(features, columns[0], width, resolution)
    initial = [node for node in initial if abs(node["v"]) < width * .85 and node["polarity"] > 0]
    route = np.vstack(([0., 0.], [anchor_length, 0.], seed_uv[seed_uv[:, 0] > anchor_length])) if len(seed_uv) else np.empty((0, 2))
    if len(route):
        route = route[np.argsort(route[:, 0])]
    beam = [dict(path=(node,), slope=0., score=node["quality"] - abs(node["v"]) / width * .2, gaps=0)
            for node in initial]
    if not beam:
        return []
    for col in columns[1:]:
        nodes = _column_peaks(features, col, width, resolution)
        t = float(features["u"][col])
        extensions = []
        for state in beam:
            last = state["path"][-1]
            du = t - last["u"]
            predicted = last["v"] + state["slope"] * du
            guided_v = float(np.interp(t, route[:, 0], route[:, 1])) if len(route) else None
            choices = [node for node in nodes if abs(node["v"] - predicted) < width * (.85 if len(route) else .60)
                       and (guided_v is None or abs(node["v"] - guided_v) < width*1.5)]
            for node in choices:
                slope = (node["v"] - last["v"]) / du
                if abs(slope) > 1.2:
                    continue
                change = abs(slope - state["slope"])
                reward = node["quality"] - .18 * abs(slope) - .45 * change
                reward -= .18 * node["saturation"]
                if t < anchor_length:
                    reward -= .65 * (node["v"] / max(2., width * .7)) ** 2
                if len(seed_uv):
                    # Continuous search corridor/penalty supplies look-ahead
                    # before the first click. Its interpolated locations are
                    # NEVER emitted as observed samples; only image extrema are.
                    reward -= .85 * abs(node["v"] - guided_v) / width
                    closest = int(np.argmin(abs(seed_uv[:, 0] - t)))
                    separation = abs(seed_uv[closest, 0] - t)
                    if separation < width * 2.5:
                        reward -= .5 * (1 - separation / (width * 2.5)) * abs(node["v"] - seed_uv[closest, 1]) / width
                extensions.append(dict(path=state["path"] + (node,), slope=.65*slope+.35*state["slope"],
                                       score=state["score"] + reward, gaps=0))
            if state["gaps"] < 3:
                missing = dict(u=t, v=predicted, quality=0., saturation=0., tangent=0.,
                               anisotropy=0., polarity=0, observed=False, siblings=0)
                extensions.append(dict(path=state["path"] + (missing,), slope=state["slope"],
                                       score=state["score"] - 1.15, gaps=state["gaps"] + 1))
        if not extensions:
            break
        # Retain alternatives in position AND direction. The best path is not
        # promoted to rope identity merely because continuity gave it a lead.
        next_beam, occupied = [], set()
        for state in sorted(extensions, key=lambda s: s["score"], reverse=True):
            cell = (round(state["path"][-1]["v"] / max(1., width * .23)), round(state["slope"] / .16))
            if cell in occupied:
                continue
            occupied.add(cell)
            next_beam.append(state)
            if len(next_beam) >= 20:
                break
        beam = next_beam
    return sorted(beam, key=lambda s: s["score"] / max(1, len(s["path"])), reverse=True)


def _comparison(points, segments, width):
    """Nearest projections onto DIRECT observed links, never over arc gaps."""
    starts, ends = [], []
    for segment in segments:
        p = _points(segment)
        if len(p) < 2:
            continue
        lengths = np.linalg.norm(np.diff(p, axis=0), axis=1)
        limit = max(5., float(np.median(lengths)) * 3.)
        good = (lengths > 1e-6) & (lengths < limit)
        starts.extend(p[:-1][good]); ends.extend(p[1:][good])
    if not starts or not len(points):
        return dict(valid=False, reason="no_direct_observed_boundary_segments", supported_samples=0, samples=[],
                    identity_confirmed=False, edge_contact_confirmed=False, physical_contact_confirmed=False)
    a, b = np.asarray(starts), np.asarray(ends)
    v = b-a
    t = np.einsum("pki,ki->pk", points[:, None, :] - a[None, :, :], v) / np.sum(v*v, axis=1)
    feet = a[None, :, :] + np.clip(t, 0, 1)[..., None] * v[None, :, :]
    distance = np.linalg.norm(points[:, None, :] - feet, axis=2)
    index = np.argmin(distance, axis=1)
    rows = np.arange(len(points))
    dist = distance[rows, index]
    # A nearest endpoint outside its observed link is a gap/end cap, not a
    # direct boundary observation under this trajectory sample.
    keep = (t[rows, index] > 0.) & (t[rows, index] < 1.) & (dist < max(40., width * 3.))
    samples = [dict(point=list(map(float, points[i])), boundary_point=list(map(float, feet[i, index[i]])),
                    center_distance_px=float(dist[i])) for i in np.flatnonzero(keep)]
    return dict(valid=bool(len(samples)), reason="direct_boundary_links_only" if samples else "curve_region_has_no_direct_boundary_support",
                supported_samples=len(samples), unsupported_samples=int(len(points)-len(samples)), samples=samples,
                minimum_center_distance_px=float(dist[keep].min()) if keep.any() else None,
                median_center_distance_px=float(np.median(dist[keep])) if keep.any() else None,
                geometric_proximity_candidate=bool(keep.sum() >= 5 and np.mean(dist[keep] <= width*.5+2.) > .4),
                identity_confirmed=False, edge_contact_confirmed=False, physical_contact_confirmed=False,
                warning="Proximity to observed boundary does not identify the new incoming winding.")


def detect_wound_curve(image_bgr, rope, arc=None, *, surface_mask=None,
                       seed_points=None, max_length_px=None):
    """Return image-supported curves and explicit identity ambiguity.

    Coordinates are original pixels. ``seed_points`` may contain post-entry
    user clicks, but does not fabricate absent ridge observations. ``valid``
    means a displayable extraction candidate exists, NEVER confirmed identity.
    This stateless routine does not consume old frames or infer motion/state.
    """
    start = time.perf_counter()
    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        return _empty("image_must_be_uint8_BGR")
    if not isinstance(rope, dict) or not rope.get("valid"):
        return _empty("independent_free_rope_unavailable")
    seeds = _points(seed_points) if seed_points is not None else np.empty((0, 2))
    if seed_points is not None and not len(seeds):
        return _empty("invalid_post_entry_seed_points")
    if surface_mask is not None and np.asarray(surface_mask).shape != image_bgr.shape[:2]:
        return _empty("surface_mask_must_match_image")
    chosen, source = _entry(rope, arc or {}, seeds)
    if chosen is None:
        return _empty(source)
    endpoint, d = chosen
    n = np.array([-d[1], d[0]])
    width = rope.get("width_px") or rope.get("diagnostics", {}).get("candidate_optical_width_px")
    if width is None or not np.isfinite(width) or width <= 0:
        return _empty("optical_width_scale_unavailable")
    width = float(np.clip(width, 2., 70.))
    anchor_length = max(25., width * 3.)
    origin = endpoint - d * anchor_length
    if max_length_px is not None and (not np.isfinite(max_length_px) or max_length_px <= 0):
        return _empty("invalid_max_length_px")
    length = min(float(max_length_px or min(1700., max(220., min(image_bgr.shape[:2]) * .55))),
                 float(np.hypot(*image_bgr.shape[:2])))
    resolution = float(np.clip(width / 6., 1., 4.))
    seed_uv = np.column_stack(((seeds-origin)@d, (seeds-origin)@n)) if len(seeds) else np.empty((0, 2))
    if len(seed_uv):
        length = min(length, max(anchor_length+width*3., float(np.max(seed_uv[:, 0]))+width*2.))
    features = _features(image_bgr, origin, d, n, width, resolution, length, guided=bool(len(seeds)))
    states = _trace(features, width, resolution, anchor_length, seed_uv)
    if not states:
        return _empty("no_ridge_continuously_anchored_to_free_rope", elapsed_ms=(time.perf_counter()-start)*1000)
    selected = []
    for state in states:
        path = list(state["path"])
        while path and not path[-1]["observed"]:
            path.pop()
        if len(path) < 12:
            continue
        values = np.array([node["v"] for node in path])
        if any(np.mean(abs(values[:min(len(values), len(old))] - old[:min(len(values), len(old))]) > width*.55) < .12
               for _, old in selected):
            continue
        selected.append((path, values))
        if len(selected) >= 3:
            break
    if not selected:
        return _empty("insufficient_continuous_ridge_support", elapsed_ms=(time.perf_counter()-start)*1000)
    candidates = []
    all_flags = set()
    boundary = arc or {}
    arc_segments = boundary["local_boundary_segments"] if "local_boundary_segments" in boundary else boundary.get("segments", [])
    for path, values in selected:
        uv = np.array([[node["u"], node["v"]] for node in path])
        xy = origin + uv[:, :1]*d + uv[:, 1:]*n
        observed = np.array([node["observed"] for node in path])
        quality = np.array([node["quality"] for node in path])
        saturation = np.array([node["saturation"] for node in path])
        post = (uv[:, 0] > anchor_length + width * 2.) & observed
        ids = np.flatnonzero(observed)
        runs = np.split(ids, np.flatnonzero(np.diff(ids) > 1)+1)
        segments = [xy[run].tolist() for run in runs if len(run) >= 2]
        post_points = xy[post]
        repeat = float(np.mean([path[i]["siblings"] >= 1 for i in np.flatnonzero(post)])) if post.any() else 0.
        saturated = float(np.mean(saturation[post] > .35)) if post.any() else 0.
        angular_change = 0.
        terminal_change = 0.
        if post.sum() >= 8:
            pp = uv[post]
            k = max(3, len(pp)//4)
            change = pp[-k:].mean(axis=0) - pp[:k].mean(axis=0)
            angular_change = abs(math.degrees(math.atan2(change[1], change[0])))
            if k >= 4 and np.ptp(pp[-k:, 0]) > width*2.:
                terminal_slope = np.polyfit(pp[-k:, 0]-pp[-k:, 0].mean(), pp[-k:, 1], 1)[0]
                terminal_change = abs(math.degrees(math.atan(terminal_slope)))
        deviation = float(np.percentile(abs(uv[post, 1]), 90)) if post.any() else 0.
        bend = bool(post.sum() >= 12 and deviation > width*.5 and max(angular_change, terminal_change) > 3.)
        flags = []
        if repeat > .18:
            flags.append("repetitive_parallel_rope_texture")
        if saturated > .15:
            flags.append("saturated_or_clipped_ridge_identity")
        if float(observed.mean()) < .90:
            flags.append("gaps_in_continuity")
        if len(selected) > 1:
            flags.append("multiple_connected_ridge_hypotheses")
        if not bend:
            flags.append("no_resolved_post_entry_bending")
        if len(seeds):
            distances = np.min(np.linalg.norm(seeds[:, None, :] - xy[observed][None, :, :], axis=2), axis=1)
            if np.max(distances) > width * 1.5:
                flags.append("manual_seeds_not_all_image_supported")
            if any(node["polarity"] < 0 for node in path if node["observed"]):
                flags.append("guided_bright_dark_polarity_ambiguous")
            if np.max(distances) > width * .35:
                flags.append("guided_body_center_vs_edge_unresolved")
        if surface_mask is not None and len(post_points):
            values_on_surface = _sample((np.asarray(surface_mask)>0).astype(np.float32), post_points[:, 0:1], post_points[:, 1:2])[:, 0]
            if np.mean(values_on_surface>.5) < .25:
                flags.append("insufficient_proposed_wound_surface_support")
        score = float(np.clip(np.mean(quality[observed]) * observed.mean(), 0, 1))
        comparison = _comparison(post_points, arc_segments, width)
        if comparison.get("geometric_proximity_candidate"):
            flags.append("boundary_ridge_cannot_be_independently_identified")
        all_flags.update(flags)
        candidates.append(dict(points=xy[observed].tolist(), observed_segments=segments,
                               score=score, supported=bool(post.sum() >= 12),
                               reason="image_ridge_candidate_identity_unconfirmed", identity_confirmed=False,
                               ambiguity_flags=flags, observed_fraction=float(observed.mean()),
                               supported_length_px=float(uv[-1, 0]-uv[0, 0]),
                               post_entry_samples=int(post.sum()),
                               manual_seed_distances_px=distances.tolist() if len(seeds) else None,
                               dark_feature_fraction=float(np.mean([node["polarity"] < 0 for node in path if node["observed"]])),
                               bend_evidence=dict(resolved=bend, departure_from_free_line_px=deviation,
                                                  mean_direction_change_deg=angular_change,
                                                  terminal_direction_change_deg=terminal_change,
                                                  interpretation="photometric_ridge_bending_only_not_rope_identity"),
                               repetitive_texture_fraction=repeat, saturation_fraction=saturated,
                               contact_comparison=comparison))
    severe = any(flag in all_flags for flag in ("repetitive_parallel_rope_texture", "saturated_or_clipped_ridge_identity",
                 "multiple_connected_ridge_hypotheses", "manual_seeds_not_all_image_supported",
                 "boundary_ridge_cannot_be_independently_identified", "guided_bright_dark_polarity_ambiguous",
                 "guided_body_center_vs_edge_unresolved"))
    status = "identity_ambiguous" if severe else ("curve_candidate_unconfirmed" if candidates[0]["bend_evidence"]["resolved"] else "straight_continuation_only")
    return dict(valid=True, identity_confirmed=False, status=status,
                reason="; ".join(sorted(all_flags)) or "single_frame_curve_does_not_confirm_incoming_winding_identity",
                candidates=candidates, contact_comparison=candidates[0]["contact_comparison"],
                trajectory_confirmed=False, edge_contact_confirmed=False, physical_contact_confirmed=False,
                last_turn_confirmed=False, new_layer_confirmed=False, force_command=None,
                diagnostics=dict(algorithm="free_anchor_multihypothesis_hessian_ridge_v1",
                                 entry_direction_source=source, entry_anchor=endpoint.tolist(),
                                 advance_direction=d.tolist(), search_origin=origin.tolist(),
                                 search_length_px=length, search_resolution_px=resolution,
                                 optical_scale_px=width, manual_post_entry_seeds=seeds.tolist(),
                                 arc_used_in_path_score=False, temporal_identity_used=False,
                                 manual_route_used_as_search_prior=bool(len(seeds)),
                                 manual_route_interpolation_emitted_as_observation=False,
                                 feature_mode="guided_local_contrast_bright_dark_extrema" if len(seeds) else "automatic_positive_hessian_ridge",
                                 score_kind="normalized_ridge_response_times_observed_fraction_not_identity_probability",
                                 observed_segments_kind="sampled_ridge_links_not_confirmed_rope_body_centerline",
                                 elapsed_ms=(time.perf_counter()-start)*1000,
                                 warning="A photometric ridge is not necessarily the wire-rope body center; ridge enhancement and boundary proximity do not confirm new-winding identity."))


track_wound_rope = detect_wound_curve
