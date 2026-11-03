"""Independent, image-only measurement of the rope/yellow-flange inner interface.

The yellow region localizes a concave *inner* boundary, not the outer flange
silhouette. Every reported sample is refined against original-resolution color
profiles. A mask can rank the localization candidates, but never supplies the
reported geometry. No ellipse, count, camera calibration or legacy geometry is
used. ``residual_px`` describes local consistency, NOT calibrated accuracy.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def _invalid(reason: str, **diagnostics: Any) -> dict:
    return {"valid": False, "status": "invalid", "reason": reason, "points": [],
            "observed_points": [], "observed_mask": [], "observed_ranges": [],
            "interpolated_ranges": [], "segments": [], "confidence": 0.0,
            "residual_px": None, "observed_x_range": None,
            "local_boundary_valid": False, "local_boundary_segments": [],
            "conic_support_points": [], "conic_support_segments": [],
            "local_boundary_diagnostics": {"reason": reason},
            "diagnostics": diagnostics}


def _runs(indices: np.ndarray, max_gap: int = 1) -> list[np.ndarray]:
    if len(indices) == 0:
        return []
    return list(np.split(indices, np.flatnonzero(np.diff(indices) > max_gap) + 1))


def _yellow_strength(bgr: np.ndarray) -> np.ndarray:
    # Normalization makes this substantially invariant to ordinary exposure
    # changes. The additive floor suppresses noise in near-black pixels.
    b, g, r = cv2.split(bgr.astype(np.float32))
    return np.clip((np.minimum(r, g) - b) / (np.maximum(r, g) + 16.0), 0.0, 1.0)


def _yellow_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    good = ((hsv[:, :, 0] >= 12) & (hsv[:, :, 0] <= 46)
            & (hsv[:, :, 1] >= 65) & (hsv[:, :, 2] >= 16)
            & (_yellow_strength(bgr) >= 0.22))
    mask = (good.astype(np.uint8) * 255)
    # Opening comes first: closing thin yellow reflections between rope strands
    # would turn them into a false continuation of the painted flange.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


def _bottom(mask: np.ndarray) -> np.ndarray:
    present = mask.any(axis=0)
    ys = mask.shape[0] - 1 - np.argmax(mask[::-1] > 0, axis=0)
    return np.where(present, ys, np.nan).astype(np.float64)


def _median_series(values: np.ndarray, radius: int) -> np.ndarray:
    if radius < 1:
        return values.copy()
    padded = np.pad(values, radius, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, radius * 2 + 1)
    return np.nanmedian(windows, axis=-1)


def _localizations(image: np.ndarray, prior: np.ndarray | None) -> list[dict]:
    mask = _yellow_mask(image)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    min_area = max(100, image.shape[0] * image.shape[1] * 0.001)
    component_ids = sorted(range(1, count), key=lambda i: stats[i, 4], reverse=True)[:6]
    candidates = []
    for component_id in component_ids:
        area = int(stats[component_id, 4])
        if area < min_area:
            continue
        ys, xs = np.where(labels == component_id)
        xy = np.column_stack([xs, ys]).astype(np.float64)
        origin = xy.mean(axis=0)
        covariance = np.cov(xy.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        tangent = eigenvectors[:, -1]
        if tangent[0] < 0:
            tangent = -tangent
        component = (labels == component_id).astype(np.uint8) * 255
        for sign in (1.0, -1.0):
            normal = np.array([-tangent[1], tangent[0]]) * sign
            basis = np.stack([tangent, normal])
            uv = (xy - origin) @ basis.T
            lo = np.floor(uv.min(axis=0)) - 3
            hi = np.ceil(uv.max(axis=0)) + 3
            size = np.ceil(hi - lo + 1).astype(int)
            if size[0] < 45 or size[1] < 15:
                continue
            affine = np.column_stack([basis, -basis @ origin - lo])
            canonical = cv2.warpAffine(component, affine, tuple(size), flags=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(canonical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            hull = cv2.convexHull(np.concatenate(contours))
            hull_mask = np.zeros_like(canonical)
            cv2.fillConvexPoly(hull_mask, hull, 255)
            bottom = _bottom(canonical)
            hull_bottom = _bottom(hull_mask)
            depth = np.nan_to_num(hull_bottom - bottom)
            depth = _median_series(depth, 3)
            max_depth = float(np.max(depth))
            if max_depth < max(4.0, size[0] * 0.012):
                continue
            threshold = max(2.0, max_depth * 0.10)
            region = depth > threshold
            # Small occluder gaps must not create separate localization objects.
            region = cv2.morphologyEx(region.astype(np.uint8)[None, :], cv2.MORPH_CLOSE,
                                     np.ones((1, max(3, int(size[0] * .015))), np.uint8))[0] > 0
            for run in _runs(np.flatnonzero(region)):
                width = int(run[-1] - run[0] + 1)
                if width < max(30, size[0] * .24):
                    continue
                u = run[np.isfinite(bottom[run])]
                if len(u) < width * .8:
                    continue
                v = _median_series(bottom[u], 2)
                # Reject tiny concavities and mechanical cut-outs. An interface
                # must have yellow support above it over a substantial span.
                notch_depth = float(np.percentile(depth[u], 75))
                if notch_depth < max(4.0, width * .015):
                    continue
                midpoint = np.column_stack([u, v]) + lo
                image_xy = midpoint @ basis + origin
                prior_score = 0.0
                if prior is not None:
                    below = image_xy + normal * max(4.0, size[1] * .08)
                    ix = np.clip(np.rint(below[:, 0]).astype(int), 0, prior.shape[1] - 1)
                    iy = np.clip(np.rint(below[:, 1]).astype(int), 0, prior.shape[0] - 1)
                    prior_score = float(np.mean(prior[iy, ix] > 0))
                score = width * notch_depth * (1.0 + prior_score)
                candidates.append({"xy": image_xy, "u": u.astype(float), "v": v,
                                   "basis": basis, "origin": origin, "lo": lo,
                                   "score": score, "width": width, "depth": notch_depth,
                                   "component_area": area, "prior_score": prior_score})
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def _refine_unmasked(image: np.ndarray, seed: np.ndarray, normals: np.ndarray, search: float,
                     exclusion_guard=None) -> dict:
    # Sample original pixels along locally perpendicular profiles. One profile
    # per ~2 original pixels is inexpensive even for a 4k source frame.
    offsets = np.arange(-search, search + .126, .25, dtype=np.float32)
    coords = seed[:, None, :] + normals[:, None, :] * offsets[None, :, None]
    xmap, ymap = coords[:, :, 0].astype(np.float32), coords[:, :, 1].astype(np.float32)
    # Remapping BGR first avoids constructing three full-size floating images.
    sampled = cv2.remap(image, xmap, ymap, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    strength = _yellow_strength(sampled)
    strength = cv2.GaussianBlur(strength, (9, 1), .65)
    # A stable step, not a single rope-texture peak: compare ~2px bands on both sides.
    step = max(2, int(round(1.5 / .25)))
    drop = strength[:, :-2 * step] - strength[:, 2 * step:]
    offset_centers = offsets[step:-step]
    # Localization is only a weak tie-breaker. Strong raw evidence can move the edge.
    penalty = .002 * (offset_centers / max(search, 1.0)) ** 2
    response = drop - penalty[None, :]
    peaks = np.argmax(response, axis=1)
    rows = np.arange(len(seed))
    center = peaks + step
    refined_offset = offsets[center].astype(float)
    # Parabolic interpolation of the symmetric color-step response.
    interior = (peaks > 0) & (peaks < response.shape[1] - 1)
    r = rows[interior]
    k = peaks[interior]
    left, mid, right = response[r, k - 1], response[r, k], response[r, k + 1]
    denominator = left - 2 * mid + right
    correction = np.divide(.5 * (left - right), denominator,
                           out=np.zeros_like(left), where=np.abs(denominator) > 1e-7)
    refined_offset[interior] += np.clip(correction, -.5, .5) * .25
    edge_strength = drop[rows, peaks]
    above = strength[rows, np.maximum(0, center - step)]
    below = strength[rows, np.minimum(strength.shape[1] - 1, center + step)]
    # Exclude range-limited searches and low chromatic contrast; exposure-dependent
    # gray gradients cannot silently substitute for a yellow-to-rope interface.
    observed = (edge_strength > .035) & (above > .07) & (below < above * .72)
    observed &= np.abs(refined_offset) < search - 1.75
    refined = seed + normals * refined_offset[:, None]
    # A free rope crossing yellow gives the same immediate color drop as the
    # desired edge, but yellow returns on the far side of that thin occluder.
    # Require a persistent nonyellow surface beyond the transition.
    far_offsets = np.linspace(max(6.0, search), max(24.0, search * 4.0), 12, dtype=np.float32)
    far_coords = refined[:, None, :] + normals[:, None, :] * far_offsets[None, :, None]
    far_colors = cv2.remap(image, far_coords[:, :, 0].astype(np.float32),
                          far_coords[:, :, 1].astype(np.float32), cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
    far_strength = _yellow_strength(far_colors)
    observed &= np.percentile(far_strength, 65, axis=1) < np.maximum(.18, above * .72)
    observed &= (refined[:, 0] >= 2) & (refined[:, 0] < image.shape[1] - 2)
    observed &= (refined[:, 1] >= 2) & (refined[:, 1] < image.shape[0] - 2)
    # A warm reflection between rope strands can pass a one-dimensional color
    # drop test. Measure actual painted-face support in a small *outside* patch,
    # rather than accepting a single favorable yellow pixel on metal texture.
    normal_offsets, tangent_offsets = np.meshgrid(np.array([-4., -7., -10.]), np.array([-5., 0., 5.]))
    tangent = np.column_stack([normals[:, 1], -normals[:, 0]])
    paint_coords = (refined[:, None, :] + normals[:, None, :]*normal_offsets.reshape(1, -1, 1)
                    + tangent[:, None, :]*tangent_offsets.reshape(1, -1, 1))
    paint_safe = np.ones(len(seed), bool)
    if exclusion_guard is not None:
        touched = cv2.remap(exclusion_guard, paint_coords[:, :, 0].astype(np.float32),
                            paint_coords[:, :, 1].astype(np.float32), cv2.INTER_NEAREST) > 0
        paint_safe &= ~np.any(touched, axis=1)
    paint_median = np.zeros(len(seed))
    paint_low = np.zeros(len(seed))
    if np.any(paint_safe):
        safe_coords = paint_coords[paint_safe]
        paint_colors = cv2.remap(image, safe_coords[:, :, 0].astype(np.float32),
                                safe_coords[:, :, 1].astype(np.float32), cv2.INTER_LINEAR)
        paint_score = _yellow_strength(paint_colors)
        paint_median[paint_safe] = np.median(paint_score, axis=1)
        paint_low[paint_safe] = np.percentile(paint_score, 20, axis=1)
    return {"xy": refined, "observed": observed, "strength": edge_strength,
            "offset": refined_offset, "paint_median": paint_median, "paint_low": paint_low}


def _refine(image, seed, normals, search, exclusion_mask=None):
    """Never use a precision profile whose pixel footprint touches an exclusion.

    Coarse topology is intentionally untouched: painting the corridor black in
    the image/color mask would create precisely the false edge we must avoid.
    """
    blocked = np.zeros(len(seed), dtype=bool)
    guard = None
    offsets = np.arange(-search, search + max(24., search * 4.) + .5, .5, dtype=np.float32)
    def touching(locations, directions):
        coords = locations[:, None, :] + directions[:, None, :] * offsets[None, :, None]
        samples = cv2.remap(guard, coords[:, :, 0].astype(np.float32),
                            coords[:, :, 1].astype(np.float32), cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT)
        return np.any(samples > 0, axis=1)
    if exclusion_mask is not None:
        # The one-pixel guard covers the bilinear interpolation footprint.
        guard = cv2.dilate(exclusion_mask.astype(np.uint8), np.ones((3, 3), np.uint8))
        blocked = touching(seed, normals)
    result = {"xy": seed.copy(), "observed": np.zeros(len(seed), bool),
              "strength": np.zeros(len(seed)), "offset": np.zeros(len(seed)),
              "paint_median": np.zeros(len(seed)), "paint_low": np.zeros(len(seed)),
              "normals": normals.copy(),
              "excluded": blocked.copy(), "recovered": np.zeros(len(seed), bool)}
    allowed = ~blocked
    if np.any(allowed):
        measured = _refine_unmasked(image, seed[allowed], normals[allowed], search, guard)
        for name in ("xy", "observed", "strength", "offset", "paint_median", "paint_low"):
            result[name][allowed] = measured[name]
    # The coarse bottom-color contour can follow the crossing rope rather than
    # the true edge immediately behind it. Search a few separate, entirely safe
    # profiles farther into the surface. No samples traverse the corridor and no
    # pixel within it is used to select/refine an observation.
    if guard is not None and np.any(blocked):
        for shift in (search, search*2, search*3):
            pending = np.flatnonzero(blocked & ~result["observed"])
            if not len(pending):
                break
            shifted = seed[pending] + normals[pending]*shift
            safe = ~touching(shifted, normals[pending])
            indices = pending[safe]
            if not len(indices):
                continue
            measured = _refine_unmasked(image, shifted[safe], normals[indices], search, guard)
            accepted = measured["observed"] & (measured["strength"] >= .06)
            indices = indices[accepted]
            for name in ("xy", "observed", "strength", "offset", "paint_median", "paint_low"):
                result[name][indices] = measured[name][accepted]
            result["recovered"][indices] = True
        result["excluded"] = blocked & ~result["observed"]
    return result


def _smooth_observations(points: np.ndarray, keep: np.ndarray, spacing: float,
                         blocked=None) -> tuple[np.ndarray, np.ndarray, float]:
    """Short local quadratic fits, never a whole-arc ellipse or global polynomial."""
    adjusted = points.copy()
    n = len(points)
    radius = max(3, int(round(9.0 / spacing)))
    residuals = np.zeros(n, dtype=float)
    blocks = np.cumsum(blocked) if blocked is not None else np.zeros(n, dtype=int)
    for i in range(n):
        left, right = max(0, i - radius), min(n, i + radius + 1)
        idx = np.arange(left, right)
        idx = idx[keep[idx] & (blocks[idx] == blocks[i])]
        if len(idx) < 5:
            continue
        t = (idx - i).astype(float)
        weights = np.exp(-.5 * (t / max(radius * .7, 1)) ** 2)
        design = np.column_stack([np.ones_like(t), t, t * t])
        predicted = np.linalg.lstsq(design * weights[:, None], points[idx] * weights[:, None], rcond=None)[0][0]
        residuals[i] = float(np.linalg.norm(points[i] - predicted))
        # Small real local bumps remain in the data: use only a 9px neighborhood
        # and blend, instead of forcing the observations onto a mathematical arc.
        adjusted[i] = .5 * points[i] + .5 * predicted
    valid_residuals = residuals[keep]
    if len(valid_residuals):
        median = float(np.median(valid_residuals))
        mad = float(np.median(np.abs(valid_residuals - median)))
        keep = keep & (residuals <= max(2.0, median + 4.5 * 1.4826 * mad))
    value = float(np.sqrt(np.mean(residuals[keep] ** 2))) if np.any(keep) else math.inf
    return adjusted, keep, value


def _local_boundaries(points, keep, refined, spacing, axis_normal,
                      exclusion_mask=None, image_offset=(0, 0)):
    """Independent local display support; never a full-ellipse extrapolation.

    All decisions use local paint support and orientation relative to the
    detected flange, not camera identity, left/right side or frame time.
    """
    paint_good = (refined["paint_median"] >= .245) & (refined["paint_low"] >= .20)
    # The near-perpendicular continuation can be the cylinder side silhouette,
    # not an inner-face arc in the flange plane. Preserve it in raw audit points
    # but do not let a few vertical pixels determine a global ellipse.
    alignment = np.abs(refined["normals"] @ axis_normal)
    plane_arc = alignment >= math.cos(math.radians(72.))
    selected = keep & paint_good & plane_arc
    selected_runs = _runs(np.flatnonzero(selected))
    smooth_segments, support_segments = [], []
    residuals = []
    guard = (cv2.dilate(exclusion_mask.astype(np.uint8), np.ones((3, 3), np.uint8))
             if exclusion_mask is not None else None)
    for run in selected_runs:
        if len(run) < 4:
            continue
        raw = points[run]
        support_segments.append(np.round(raw, 3).tolist())
        # A drawable local curve needs extent as well as a count. Scattered
        # points can still be kept as conic evidence without drawing a false arc.
        travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))]
        if len(run) < 7 or travel[-1] < 18.:
            continue
        smoothed = raw.copy()
        for j in range(len(raw)):
            indices = np.flatnonzero(np.abs(travel-travel[j]) <= 18.)
            if len(indices) < 5:
                continue
            t = travel[indices]-travel[j]
            design = np.column_stack([np.ones(len(t)), t, t*t])
            weights = np.exp(-.5*(t/11.)**2)
            for _ in range(3):
                fit = np.linalg.lstsq(design*weights[:, None], raw[indices]*weights[:, None], rcond=None)[0]
                distance = np.linalg.norm(raw[indices]-design@fit, axis=1)
                scale = max(.5, 1.4826*float(np.median(np.abs(distance-np.median(distance)))))
                weights *= np.minimum(1., 2.5*scale/np.maximum(distance, 1e-6))
            predicted = fit[0]
            delta = float(np.linalg.norm(predicted-raw[j]))
            if delta <= 2.5:
                smoothed[j] = .25*raw[j]+.75*predicted
                residuals.append(delta)
        if guard is not None:
            fractions = np.linspace(0., 1., 9)[None, :, None]
            links = smoothed[:-1, None, :]*(1-fractions)+smoothed[1:, None, :]*fractions
            links -= np.asarray(image_offset)[None, None, :]
            touches = cv2.remap(guard, links[:, :, 0].astype(np.float32),
                                links[:, :, 1].astype(np.float32), cv2.INTER_NEAREST) > 0
            if np.any(touches):
                # Raw support links were checked before smoothing. Never let a
                # cosmetic smoothing displacement reconnect across a corridor.
                smoothed = raw.copy()
        smooth_segments.append(np.round(smoothed, 3).tolist())
    return dict(local_boundary_valid=bool(smooth_segments), local_boundary_segments=smooth_segments,
                conic_support_points=np.round(points[selected], 3).tolist(),
                conic_support_segments=support_segments,
                local_boundary_diagnostics={
                    "method": "paint_verified_observed_local_quadratic_segments",
                    "extrapolation": False, "bridges_support_gaps": False,
                    "independent_of_global_ellipse_acceptance": True,
                    "input_observed_points": int(np.sum(keep)),
                    "selected_support_points": int(np.sum(selected)),
                    "paint_support_rejected_points": int(np.sum(keep & ~paint_good)),
                    "ambiguous_steep_continuation_points": int(np.sum(keep & ~plane_arc)),
                    "drawable_segments": len(smooth_segments),
                    "smoothing_rms_displacement_px": float(np.sqrt(np.mean(np.square(residuals)))) if residuals else None,
                    "note": "Local observed boundary only; no claim of reliable whole-ellipse extrapolation."})


def detect_arc(image_bgr: np.ndarray, roi=None, surface_mask=None, exclusion_mask=None) -> dict:
    """Return JSON-safe measured inner-arc samples in full original-image pixels.

    ``points`` contains observed samples only, ordered along the flange principal
    axis (not necessarily by image x for a rotated camera). ``segments`` separates
    support gaps; consumers must NOT draw across segments. No invented points are
    returned and ``interpolated_ranges`` is always empty. ``observed_ranges`` gives
    original-image x ranges per segment, while ``support_ranges`` gives ordered
    arc-index ranges, meaningful even after 90-degree rotation. An optional mask
    must match the full image. ``exclusion_mask`` is a full-image boolean corridor
    around an independently found occluder; no original-pixel refinement profile
    or smoothing stencil may cross it. It never alters coarse yellow topology.
    Invalid shape/type inputs return an invalid result.
    """
    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        return _invalid("image_must_be_HxWx3_BGR")
    if image_bgr.dtype != np.uint8 or min(image_bgr.shape[:2]) < 24:
        return _invalid("image_must_be_uint8_and_at_least_24px")
    height, width = image_bgr.shape[:2]
    if roi is None:
        box = [0, 0, width, height]
    else:
        try:
            if len(roi) != 4 or not np.all(np.isfinite(roi)):
                return _invalid("roi_must_be_four_finite_coordinates")
            box = [max(0, int(roi[0])), max(0, int(roi[1])), min(width, int(roi[2])), min(height, int(roi[3]))]
        except (TypeError, ValueError, OverflowError):
            return _invalid("roi_must_be_four_finite_coordinates")
    x0, y0, x1, y1 = box
    if x1 - x0 < 24 or y1 - y0 < 24:
        return _invalid("roi_too_small_or_empty", roi=box)
    if surface_mask is not None:
        surface_mask = np.asarray(surface_mask)
        if surface_mask.shape != (height, width):
            return _invalid("surface_mask_must_match_full_image")
    if exclusion_mask is not None:
        exclusion_mask = np.asarray(exclusion_mask)
        if exclusion_mask.shape != (height, width):
            return _invalid("exclusion_mask_must_match_full_image")
        exclusion_mask = exclusion_mask.astype(bool)[y0:y1, x0:x1]
    cropped = image_bgr[y0:y1, x0:x1]
    scale = min(1.0, 1100.0 / max(cropped.shape[:2]))
    small = cv2.resize(cropped, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else cropped
    prior = None
    if surface_mask is not None:
        prior = cv2.resize((surface_mask[y0:y1, x0:x1] > 0).astype(np.uint8),
                           (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    candidates = _localizations(small, prior)
    if not candidates:
        return _invalid("no_supported_yellow_inner_concavity", roi=box, localization_scale=scale)
    results = []
    for candidate in candidates[:3]:
        u = candidate["u"]
        v = candidate["v"]
        # Denoise color threshold stairs ONLY for search locations/normals.
        v_smooth = cv2.GaussianBlur(v[None, :].astype(np.float32), (0, 0), 2.0)[0]
        sample_step = max(.55, 2.0 * scale)
        ui = np.arange(float(u[0]), float(u[-1]) + .01, sample_step)
        vi = np.interp(ui, u, v_smooth)
        slopes = np.interp(ui, u, np.gradient(v_smooth, u))
        uv = np.column_stack([ui, vi]) + candidate["lo"]
        seed = (uv @ candidate["basis"] + candidate["origin"]) / scale
        canonical_normals = np.column_stack([-slopes, np.ones_like(slopes)])
        canonical_normals /= np.linalg.norm(canonical_normals, axis=1, keepdims=True)
        normals = canonical_normals @ candidate["basis"]
        refined = _refine(cropped, seed, normals, max(5.0, 3.5 / scale), exclusion_mask)
        if exclusion_mask is not None:
            fractions = np.linspace(0., 1., 17)[None, :, None]
            links = refined["xy"][:-1, None, :] * (1-fractions) + refined["xy"][1:, None, :]*fractions
            guard = cv2.dilate(exclusion_mask.astype(np.uint8), np.ones((3, 3), np.uint8))
            touched = cv2.remap(guard, links[:, :, 0].astype(np.float32),
                                links[:, :, 1].astype(np.float32), cv2.INTER_NEAREST) > 0
            bad = np.flatnonzero(touched.any(axis=1))
            refined["observed"][bad] = False
            refined["observed"][bad+1] = False
            refined["excluded"][bad] = True
            refined["excluded"][bad+1] = True
        points, keep, residual = _smooth_observations(refined["xy"], refined["observed"],
                                                      sample_step / scale, refined["excluded"])
        if exclusion_mask is not None:
            # Also forbid a plotted edge joining two safe endpoints over a narrow
            # excluded corridor. This is independent of color/gradient quality.
            fractions = np.linspace(0., 1., 9)[None, :, None]
            links = points[:-1, None, :] * (1 - fractions) + points[1:, None, :] * fractions
            guard = cv2.dilate(exclusion_mask.astype(np.uint8), np.ones((3, 3), np.uint8))
            touched = cv2.remap(guard, links[:, :, 0].astype(np.float32),
                                links[:, :, 1].astype(np.float32), cv2.INTER_NEAREST) > 0
            bad_links = np.flatnonzero(touched.any(axis=1))
            keep[bad_links] = False
            keep[bad_links + 1] = False
        # Remove islands of a few pixels. Long gaps remain missing, never bridged.
        # The conic stage can robustly combine many short, genuinely measured
        # pieces. Do not erase the dim far side merely because rope texture
        # breaks its raw support into shorter runs than the bright near side.
        minimum_run = max(4, int(round(len(points) * .004)))
        for run in _runs(np.flatnonzero(keep)):
            if len(run) < minimum_run:
                keep[run] = False
        support = float(np.mean(keep))
        raw_score = candidate["score"] * support
        results.append((raw_score, candidate, points, keep, residual, refined, sample_step / scale))
    _, candidate, points, keep, residual, refined, spacing = max(results, key=lambda item: item[0])
    supported = np.flatnonzero(keep)
    if (len(supported) < 20 or float(np.mean(keep)) < .25
            or len(supported) * spacing < max(30, min(cropped.shape[:2]) * .08)):
        return _invalid("insufficient_raw_image_edge_support", roi=box,
                        supported_samples=int(len(supported)), candidate_samples=int(len(points)),
                        exclusion_used=exclusion_mask is not None,
                        excluded_profile_count=int(np.sum(refined["excluded"])))
    support_fraction = float(np.mean(keep))
    points += np.array([x0, y0])
    local = _local_boundaries(points, keep, refined, spacing, candidate["basis"][1],
                              exclusion_mask, (x0, y0))
    observed = points[keep]
    segments = []
    support_ranges = []
    observed_ranges = []
    for run in _runs(supported):
        segment = points[run]
        segments.append(np.round(segment, 3).tolist())
        support_ranges.append([int(run[0]), int(run[-1])])
        observed_ranges.append([round(float(segment[:, 0].min()), 3), round(float(segment[:, 0].max()), 3)])
    quality = float(np.median(refined["strength"][keep]))
    confidence = float(np.clip(support_fraction * min(1.0, quality / .13) * math.exp(-residual / 3.0), 0, 1))
    gap_fraction = 1.0 - support_fraction
    partial = gap_fraction > .08 or len(segments) > 1
    exclusion_polygons = []
    if exclusion_mask is not None and np.any(exclusion_mask):
        # A compact, conservative footprint for downstream model rendering.
        # It is an exclusion annotation, not observed object geometry.
        guarded = cv2.dilate(exclusion_mask.astype(np.uint8), np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(guarded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            polygon = cv2.approxPolyDP(contour, .5, True).reshape(-1, 2)
            if len(polygon) >= 3:
                exclusion_polygons.append((polygon + [x0, y0]).astype(int).tolist())
    return {"valid": True, "status": "partial" if partial else "ok",
            **local,
            "reason": "observed_interface_with_missing_support" if partial else "observed_inner_interface",
            "points": np.round(observed, 3).tolist(), "observed_points": np.round(observed, 3).tolist(),
            "observed_mask": [True] * len(observed), "segments": segments,
            "observed_ranges": observed_ranges, "support_ranges": support_ranges,
            "exclusion_polygons": exclusion_polygons,
            "interpolated_ranges": [], "confidence": round(confidence, 4),
            "residual_px": round(residual, 4),
            "observed_x_range": [round(float(observed[:, 0].min()), 3), round(float(observed[:, 0].max()), 3)],
            "diagnostics": {"method": "yellow_concavity_original_pixel_profile",
                            "roi": box, "localization_scale": scale,
                            "candidate_count": len(candidates), "candidate_samples": len(points),
                            "observed_samples": len(observed), "support_fraction": round(support_fraction, 4),
                            "median_yellow_step": round(quality, 4),
                            "coarse_concavity_depth_px": round(candidate["depth"] / scale, 3),
                            "sample_spacing_px": round(spacing, 3), "prior_used": prior is not None,
                            "exclusion_used": exclusion_mask is not None,
                            "excluded_profile_count": int(np.sum(refined["excluded"])),
                            "recovered_outside_corridor_profiles": int(np.sum(refined["recovered"] & keep)),
                            "excluded_pixel_count": int(np.sum(exclusion_mask)) if exclusion_mask is not None else 0,
                            "exclusion_changes_coarse_topology": False,
                            "prior_agreement": round(candidate["prior_score"], 4),
                            "principal_direction": candidate["basis"][0].round(6).tolist(),
                            "accuracy_note": "Subpixel coordinates; residual is consistency, not accuracy certification.",
                            "no_extrapolation": True}}
