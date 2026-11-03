"""Independent, single-frame free-rope line measurement.

SAM masks and clicks are *proposals*: measurements always come from original
image stripe profiles. Endpoints only delimit observed evidence, never physical
rope endpoints or a contact point. No motion, camera-side, or tangency prior is
used. A line can be valid while its optical width remains unavailable.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def _invalid(reason: str, diagnostics: dict | None = None) -> dict:
    return {"valid": False, "reason": reason, "angle_deg": None,
            "normal": None, "rho_px": None, "width_px": None,
            "width_interval_px": None, "width_reference_point": None,
            "angular_uncertainty_deg": None, "normal_uncertainty_px": None,
            "confidence": 0.0, "residual_px": None,
            "observed_segment": None, "observed_support": 0,
            "diagnostics": diagnostics or {}}


def _line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Robust orthogonal fit; insensitive to endpoint choice."""
    p = np.asarray(points, np.float64)
    weights = np.ones(len(p))
    for _ in range(7):
        c = np.average(p, axis=0, weights=weights)
        q = p - c
        cov = (q * weights[:, None]).T @ q / weights.sum()
        _, vectors = np.linalg.eigh(cov)
        d = vectors[:, -1]
        n = np.array([-d[1], d[0]])
        residual = q @ n
        scale = max(0.30, 1.4826 * float(np.median(np.abs(residual))))
        u = residual / (4.685 * scale)
        weights = np.where(np.abs(u) < 1, (1 - u * u) ** 2, 0.0) + 1e-5
    if d[0] < 0 or (abs(d[0]) < 1e-8 and d[1] < 0):
        d = -d
    return c, d, float(np.sqrt(np.average(residual ** 2, weights=weights)))


def _clip_line(c: np.ndarray, d: np.ndarray, box: tuple) -> tuple | None:
    lo, hi = -float("inf"), float("inf")
    for k, (a, b) in enumerate(((box[0], box[2]), (box[1], box[3]))):
        if abs(d[k]) < 1e-8:
            if not a <= c[k] <= b:
                return None
        else:
            v0, v1 = (a - c[k]) / d[k], (b - c[k]) / d[k]
            lo, hi = max(lo, min(v0, v1)), min(hi, max(v0, v1))
    return (lo, hi) if hi > lo else None


def _sample(img: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return cv2.remap(img, xy[..., 0].astype(np.float32),
                     xy[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


def _profiles(channel: np.ndarray, c: np.ndarray, d: np.ndarray, box: tuple,
              yellow: np.ndarray | None, mask: np.ndarray | None,
              search: float, step: float, max_width: float,
              guided: bool, sample_step: float = 5.0,
              projection_range: tuple | None = None) -> dict | None:
    """Locate two half-prominence boundaries with subpixel interpolation.

    Both flanks must be darker than the stripe. Multiple neighbouring peaks
    and broad metal/plastic highlights are rejected before fitting.
    """
    n = np.array([-d[1], d[0]])
    radius = search + max_width * 1.7
    h, w = channel.shape
    bounds = (max(box[0], 1), max(box[1], 1), min(box[2], w - 2), min(box[3], h - 2))
    clipped = _clip_line(c, d, bounds)
    if clipped is None:
        return None
    lo, hi = clipped
    if projection_range is not None:
        lo = max(lo, projection_range[0] - float(c @ d))
        hi = min(hi, projection_range[1] - float(c @ d))
    ts = np.arange(lo, hi, sample_step)
    if len(ts) < 8:
        return None
    centers = c + ts[:, None] * d
    offsets = np.arange(-radius, radius + step * 0.5, step)
    xy = centers[:, None, :] + offsets[None, :, None] * n
    profiles = _sample(channel, xy).astype(np.float32)
    profiles = cv2.GaussianBlur(profiles, (5, 1), 0.6)
    eligible = np.ones(len(ts), bool)
    yellow_support = np.zeros(len(ts), np.float32)
    if yellow is not None:
        side_distance = max_width * 1.6
        sides = centers[:, None, :] + np.array([-side_distance, side_distance])[None, :, None] * n
        ys = _sample(yellow, sides)
        yellow_support = np.max(ys, axis=1)
        if not guided:
            eligible &= yellow_support > 0.3
    peaks = np.flatnonzero(np.abs(offsets) <= search)
    if len(peaks) < 1:
        return None
    k = peaks[np.argmax(profiles[:, peaks], axis=1)]
    rows = np.arange(len(ts))
    peak = profiles[rows, k]
    left_base = np.percentile(profiles[:, :max(2, len(offsets) // 5)], 40, axis=1)
    right_base = np.percentile(profiles[:, -max(2, len(offsets) // 5):], 40, axis=1)
    base = np.maximum(left_base, right_base)
    prominence = peak - base
    half = base + prominence * 0.5
    index_grid = np.arange(len(offsets))[None, :]
    below = profiles <= half[:, None]
    kl = np.max(np.where(below & (index_grid < k[:, None]), index_grid, -1), axis=1)
    kr = np.min(np.where(below & (index_grid > k[:, None]), index_grid, len(offsets)), axis=1)
    eligible &= (kl >= 0) & (kr < len(offsets)) & (prominence >= 17)
    kl = np.clip(kl, 0, len(offsets) - 2)
    kr = np.clip(kr, 1, len(offsets) - 1)
    vl, vl1 = profiles[rows, kl], profiles[rows, kl + 1]
    vr0, vr = profiles[rows, kr - 1], profiles[rows, kr]
    xl = offsets[kl] + step * np.clip((half - vl) / np.maximum(vl1 - vl, 1e-3), 0, 1)
    xr = offsets[kr - 1] + step * np.clip((vr0 - half) / np.maximum(vr0 - vr, 1e-3), 0, 1)
    widths = xr - xl
    shifts = (xl + xr) / 2
    eligible &= (widths >= max(0.8, step * 1.2)) & (widths <= max_width)
    eligible &= np.abs(shifts) <= search + max_width * 0.35
    # Another strong stripe in the outer flanks is evidence of wound-rope
    # texture, not an isolated incoming span.
    outer = np.abs(offsets[None, :] - shifts[:, None]) > np.maximum(widths[:, None] * 1.35, 3 * step)
    neighbours = np.max(np.where(outer, profiles, -1), axis=1)
    isolated = neighbours < base + prominence * 0.90
    eligible &= isolated
    points = centers + shifts[:, None] * n
    if mask is not None:
        # A generous corridor tolerates ragged SAM boundaries while preventing
        # an unrelated parallel machine edge winning a guided search.
        ms = _sample(mask, points[:, None, :])[:, 0]
        eligible &= ms > 0.1
    if eligible.sum() < 8:
        return None
    # Keep a coherent support run; tolerate brief twists/shadows/occlusions.
    indices = np.flatnonzero(eligible)
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > max(5, int(35 / sample_step))) + 1)
    group = max(groups, key=lambda g: (ts[g[-1]] - ts[g[0]]) * min(1., len(g) / 15.))
    if len(group) < 8:
        return None
    return {"points": points[group], "widths": widths[group],
            "prominence": prominence[group], "yellow": yellow_support[group],
            "span": float(ts[group[-1]] - ts[group[0]]),
            "coverage": float(len(group) / (group[-1] - group[0] + 1)),
            "eligible": int(eligible.sum())}


def _free_color_profiles(image: np.ndarray, c: np.ndarray, d: np.ndarray,
                         box: tuple, projection_range: tuple, scale: float,
                         yellow: np.ndarray, brightness: dict) -> dict | None:
    """Measure the whole achromatic optical stripe over coloured free space.

    A bright braid facet is not the cable envelope. Min/max colour ratio also
    includes darker grey facets, but is meaningful only with yellow on BOTH
    sides. This independent semantic check excludes the silver wound surface;
    no contact geometry, camera side, or tangent constraint is used.
    """
    channel = image.min(axis=2).astype(np.float32)
    channel *= 255. / np.maximum(image.max(axis=2), 1)
    result = _profiles(channel, c, d, box, None, None,
                       search=max(3., 3. / scale), step=.5,
                       max_width=12. / scale, guided=True,
                       sample_step=max(3., 4. / scale), projection_range=projection_range)
    if result is None:
        return None
    n = np.array([-d[1], d[0]])
    p = result["points"]
    distance = np.maximum(20., result["widths"] * 1.25)
    sides = p[:, None, :] + np.array([-1., 1.])[None, :, None] * distance[:, None, None] * n
    bilateral = np.min(_sample(yellow, sides * scale), axis=1) > .45
    indices = np.flatnonzero(bilateral)
    if len(indices) < 12:
        return None
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 4) + 1)
    ids = max(groups, key=len)
    if len(ids) < 12:
        return None
    span = float(np.ptp(p[ids] @ d))
    if span < max(150., min(image.shape[:2]) * .065, brightness["span"] * .30):
        return None
    fc, fd, residual = _line(p[ids])
    if abs(float(fd @ d)) < math.cos(math.radians(1.)) or residual > 2.:
        return None
    # Do not let an unrelated grey ridge replace the verified bright stripe.
    if abs(float((fc - c) @ n)) > max(5., float(np.median(brightness["widths"]))):
        return None
    selected = {k: (v[ids] if isinstance(v, np.ndarray) and len(v) == len(p) else v)
                for k, v in result.items()}
    selected["span"] = span
    selected["coverage"] = float(len(ids) / (ids[-1] - ids[0] + 1))
    return selected


def _support_uncertainty(points: np.ndarray, c: np.ndarray,
                         d: np.ndarray, residual: float) -> tuple[float, float]:
    """Empirical sensitivity to the supported segment, NOT calibrated CIs."""
    order = np.argsort((points - c) @ d)
    angular = []
    offsets = []
    n = np.array([-d[1], d[0]])
    # Halves expose sag/photometric bias hidden by a small pooled residual.
    for ids in np.array_split(order, 2):
        if len(ids) < 8:
            continue
        sc, sd, _ = _line(points[ids])
        sn = np.array([-sd[1], sd[0]])
        angular.append(abs(math.degrees(math.atan2(float(d[0] * sd[1] - d[1] * sd[0]), float(d @ sd)))))
        offsets.append(abs(float((c - sc) @ sn)))
    span = max(1., float(np.ptp(points @ d)))
    angular_proxy = max(math.degrees(math.atan2(max(.35, residual), span)), max(angular, default=0.))
    normal_proxy = max(.35, residual, max(offsets, default=0.))
    return float(angular_proxy), float(normal_proxy)


def _hypotheses(gray: np.ndarray, white: np.ndarray, yellow: np.ndarray | None,
                mask: np.ndarray | None, seeds: np.ndarray | None) -> list:
    h, w = gray.shape
    out = []
    yellow_near = cv2.dilate(yellow, np.ones((81, 81), np.uint8)) if yellow is not None else None
    if seeds is not None and len(seeds) >= 2:
        c, d, _ = _line(seeds)
        out.append((c, d, float(np.ptp(seeds @ d)), "clicks"))
    if mask is not None:
        ys, xs = np.nonzero(mask > 0.5)
        if len(xs) >= 12:
            stride = max(1, len(xs) // 8000)
            points = np.column_stack((xs[::stride], ys[::stride])).astype(float)
            c, d, _ = _line(points)
            out.append((c, d, float(np.ptp(points @ d)), "mask_proposal"))
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    sources = [white, gray]
    for source in sources:
        lines = detector.detect(np.ascontiguousarray(source.astype(np.uint8)))[0]
        if lines is None:
            continue
        lines = lines.reshape(-1, 4).astype(float)
        lengths = np.linalg.norm(lines[:, 2:] - lines[:, :2], axis=1)
        order = np.argsort(lengths)[::-1]
        for i in order[:700]:
            if lengths[i] < max(22, min(h, w) * 0.026):
                break
            a, b = lines[i, :2], lines[i, 2:]
            c, d = (a + b) / 2, (b - a) / lengths[i]
            if d[0] < 0:
                d = -d
            n = np.array([-d[1], d[0]])
            if seeds is not None and np.min(np.abs((seeds - c) @ n)) > 14:
                continue
            samples = a + np.linspace(0, 1, 11)[:, None] * (b - a)
            if mask is not None and float(_sample(mask, samples[:, None, :]).max()) < 0.1:
                continue
            if yellow is not None and mask is None and seeds is None:
                ys = _sample(yellow, samples[:, None, :] + np.array([-12., 12.])[None, :, None] * n)
                # A true free span can be in front of wound rope or beyond the
                # flange, so immediate yellow adjacency is only a score, never
                # an exclusion. Broad scene proximity excludes distant rails.
                if float(_sample(yellow_near, samples[:, None, :]).max()) < 0.3:
                    continue
            duplicate = False
            for old_c, old_d, _, _ in out:
                angle = abs(float(d @ old_d))
                if angle > math.cos(math.radians(1.0)) and abs(float((c - old_c) @ np.array([-old_d[1], old_d[0]]))) < 4:
                    duplicate = True
                    break
            if not duplicate:
                out.append((c, d, float(lengths[i]), "lsd"))
    # Keep the longest hypotheses from both intensity channels; the second
    # channel must not be starved by a heavily textured first one.
    proposals = [p for p in out if p[3] != "lsd"]
    return proposals + sorted((p for p in out if p[3] == "lsd"), key=lambda p: p[2], reverse=True)[:140]


def detect_rope(image_bgr: np.ndarray, roi=None, rope_mask=None, arc=None,
                seed_points=None) -> dict[str, Any]:
    """Measure an isolated bright free-rope centerline in original pixels.

    ``roi`` is an optional original-image ``(x0,y0,x1,y1)`` search box.
    ``rope_mask`` is a full-resolution binary SAM proposal. ``seed_points``
    are positive original-image clicks. ``arc`` is accepted for integration,
    but never forces line orientation or tangency. Results are JSON-safe.

    Width denotes an optical stripe width, not a calibrated physical diameter;
    unresolved/thin/unstable highlights return ``width_px=None`` with a reason.
    """
    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        return _invalid("image_must_be_bgr")
    if min(image_bgr.shape[:2]) < 32 or not np.isfinite(image_bgr).all():
        return _invalid("image_too_small_or_nonfinite")
    h, w = image_bgr.shape[:2]
    if roi is None:
        box = (0, 0, w - 1, h - 1)
    else:
        try:
            x0, y0, x1, y1 = map(float, roi)
            if not np.isfinite([x0, y0, x1, y1]).all():
                raise ValueError
            box = (max(0., x0), max(0., y0), min(w - 1., x1), min(h - 1., y1))
        except (ValueError, TypeError):
            return _invalid("invalid_roi")
        if box[2] - box[0] < 20 or box[3] - box[1] < 20:
            return _invalid("roi_too_small")
    seeds = None
    if seed_points is not None:
        try:
            seeds = np.asarray(seed_points, dtype=float).reshape(-1, 2)
        except (ValueError, TypeError):
            return _invalid("invalid_seed_points")
        if not np.isfinite(seeds).all():
            return _invalid("invalid_seed_points")
        if len(seeds) == 0:
            seeds = None
    if rope_mask is not None:
        rope_mask = np.asarray(rope_mask)
        if rope_mask.shape != (h, w):
            return _invalid("mask_must_match_original_image")
        if np.count_nonzero(rope_mask) < 12:
            return _invalid("empty_rope_mask")
    scale = min(1., 1536. / max(h, w))
    small = cv2.resize(image_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    small = np.clip(small, 0, 255).astype(np.uint8)
    # Normalize exposure for hypothesis/semantic thresholds only. The final
    # profile positions and optical widths use untouched original intensities.
    # This is deliberately capped: almost-black/noisy frames must abstain.
    white_level = float(np.percentile(small.min(axis=2), 99.8))
    exposure_gain = float(np.clip(230. / max(white_level, 1.), 1., 4.))
    if exposure_gain > 1.01:
        small = np.clip(small.astype(np.float32) * exposure_gain, 0, 255).astype(np.uint8)
    sh, sw = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    white = small.min(axis=2)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    yellow = ((hsv[:, :, 0] >= 13) & (hsv[:, :, 0] <= 43) & (hsv[:, :, 1] > 65) & (hsv[:, :, 2] > 25)).astype(np.float32)
    yellow = cv2.GaussianBlur(yellow, (5, 5), 1.0)
    has_yellow = int(np.count_nonzero(yellow > 0.5)) > sh * sw * 0.01
    flange_envelope = None
    if has_yellow:
        closed = cv2.morphologyEx((yellow > 0.5).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
        contours = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if contours:
            largest = max(contours, key=cv2.contourArea)
            flange_envelope = np.zeros_like(yellow)
            cv2.fillConvexPoly(flange_envelope, cv2.convexHull(largest), 1.)
    guided = rope_mask is not None or seeds is not None or roi is not None
    # No camera-specific positional crop. The flange colour supplies semantic
    # support when available; a clean synthetic/plain scene also remains usable.
    yellow_context = yellow if has_yellow else None
    mask_small = None
    if rope_mask is not None:
        mask_small = cv2.resize((rope_mask > 0).astype(np.float32), (sw, sh), interpolation=cv2.INTER_NEAREST)
        mask_small = cv2.dilate(mask_small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    seeds_small = seeds * scale if seeds is not None else None
    hypotheses = _hypotheses(gray, white, yellow_context, mask_small, seeds_small)
    small_box = tuple(v * scale for v in box)
    ranked = []
    for c, d, proposal_length, source in hypotheses:
        result = _profiles(white, c, d, small_box, yellow_context, mask_small,
                           search=7., step=0.5, max_width=9., guided=True)
        if result is None or result["span"] < max(45, min(sh, sw) * 0.065):
            continue
        fc, fd, residual = _line(result["points"])
        if not guided and has_yellow and (float(np.median(result["prominence"])) < 120 or result["span"] < 100 or result["coverage"] < 0.65):
            continue
        strong_stripe = float(np.median(result["prominence"])) > 175.
        result["envelope_share"] = float(np.mean(_sample(flange_envelope, result["points"][:, None, :]) > 0.3)) if flange_envelope is not None else 1.
        if not guided and has_yellow and not strong_stripe and float(np.mean(result["yellow"] > 0.3)) < 0.06:
            continue
        if not guided and result["envelope_share"] < (0.05 if strong_stripe else 0.25):
            continue
        if residual > 0.85:
            continue
        if seeds_small is not None:
            sn = np.array([-fd[1], fd[0]])
            seed_distance = np.abs((seeds_small - fc) @ sn)
            if float(np.max(seed_distance)) > max(8., float(np.median(result["widths"])) * 1.5):
                continue
        score = result["span"] * math.sqrt(result["coverage"]) * min(1.0, float(np.median(result["prominence"])) / 85.)
        score *= 0.6 + 0.4 * float(np.mean(result["yellow"])) if has_yellow else 1.
        score /= 1. + residual / 2.
        ranked.append((score, fc, fd, result, source))
    ranked.sort(key=lambda x: x[0], reverse=True)
    diagnostics = {"algorithm": "independent_lsd_free_span_profiles_v2", "proposal_count": len(hypotheses),
                   "accepted_hypotheses": len(ranked), "proposal_scale": scale,
                   "yellow_context": bool(has_yellow), "guided": bool(guided),
                   "proposal_exposure_gain": exposure_gain,
                   "arc_used_as_tangency_constraint": False,
                   "width_kind": "optical_half_prominence", "width_status": "unavailable"}
    diagnostics["candidates"] = [{"score": float(s), "angle_deg": float(math.degrees(math.atan2(d[1], d[0])) % 180),
                                 "center": list(map(float, c / scale)), "span_px": r["span"] / scale,
                                 "prominence": float(np.median(r["prominence"])),
                                 "coverage": r["coverage"], "width_px": float(np.median(r["widths"])) / scale}
                                for s, c, d, r, _ in ranked[:8]]
    if not ranked:
        return _invalid("no_isolated_rope_support", diagnostics)
    best = ranked[0]
    distinct = []
    for candidate in ranked[1:]:
        _, c, d, _, _ = candidate
        n = np.array([-best[2][1], best[2][0]])
        if abs(float(d @ best[2])) < math.cos(math.radians(2)) or abs(float((c - best[1]) @ n)) > 9:
            distinct.append(candidate)
    competitor_ratio = float(distinct[0][0] / best[0]) if distinct else 0.
    diagnostics["competitor_ratio"] = competitor_ratio
    if competitor_ratio > 0.80:
        return _invalid("ambiguous_parallel_or_crossing_candidates", diagnostics)
    if not guided and not has_yellow and len(hypotheses) > 12:
        return _invalid("no_semantic_context_in_clutter", diagnostics)
    _, c, d, coarse, source = best
    c = c / scale
    # Refine only along the independently observed support, not the entire
    # frame: this keeps machine edges and wound rows outside the fit.
    projected = coarse["points"] @ d / scale
    t0, t1 = np.percentile(projected, [1, 99])
    n = np.array([-d[1], d[0]])
    rho = float(c @ n)
    ends = np.array([t0 * d + rho * n, t1 * d + rho * n])
    margin = 20 / scale
    fine_box = (max(box[0], float(ends[:, 0].min()) - margin),
                max(box[1], float(ends[:, 1].min()) - margin),
                min(box[2], float(ends[:, 0].max()) + margin),
                min(box[3], float(ends[:, 1].max()) + margin))
    full_white = np.min(image_bgr, axis=2).astype(np.float32)
    fine = _profiles(full_white, c, d, fine_box, None, None,
                     search=max(3., 3. / scale), step=0.5, max_width=12. / scale,
                     guided=True, sample_step=max(3., 4. / scale), projection_range=(t0, t1))
    if fine is None:
        return _invalid("original_resolution_refinement_failed", diagnostics)
    brightness_support = int(len(fine["points"]))
    free_color = _free_color_profiles(image_bgr, c, d, fine_box, (t0, t1),
                                      scale, yellow, fine) if has_yellow else None
    if free_color is not None:
        fine = free_color
        diagnostics["width_kind"] = "optical_achromatic_envelope_half_prominence"
        diagnostics["line_support_kind"] = "bilateral_yellow_free_span"
    else:
        diagnostics["line_support_kind"] = "isolated_brightness_stripe"
    diagnostics["brightness_support_before_free_span_filter"] = brightness_support
    diagnostics["fine_projection_range_px"] = [float(t0), float(t1)]
    fc, fd, residual = _line(fine["points"])
    normal = np.array([-fd[1], fd[0]])
    if normal[1] < 0 or (abs(normal[1]) < 1e-8 and normal[0] < 0):
        normal = -normal
    signed = (fine["points"] - fc) @ normal
    keep = np.abs(signed) < max(1.5, residual * 3.)
    points = fine["points"][keep]
    if len(points) < 10:
        return _invalid("insufficient_original_resolution_support", diagnostics)
    widths = fine["widths"][keep]
    width = float(np.median(widths))
    width_mad = float(1.4826 * np.median(np.abs(widths - width)))
    if not guided and has_yellow and coarse["envelope_share"] < 0.25 and width_mad < max(0.75, width * 0.06):
        return _invalid("uniform_machine_edge_outside_flange", diagnostics)
    length = float(np.ptp(points @ fd))
    if residual > max(1.7, width * 0.3) or length < max(55., min(h, w) * 0.06):
        return _invalid("nonlinear_or_short_support", diagnostics)
    t0, t1 = np.percentile((points - fc) @ fd, [1, 99])
    segment = [list(map(float, fc + t0 * fd)), list(map(float, fc + t1 * fd))]
    width_interval = [float(max(.5, np.percentile(widths, 10) - .5)),
                      float(np.percentile(widths, 90) + .5)]
    angular_uncertainty, normal_uncertainty = _support_uncertainty(points, fc, fd, residual)
    width_bins = []
    order = np.argsort((points - fc) @ fd)
    for ids in np.array_split(order, min(6, max(1, len(points) // 12))):
        wp = points[ids].mean(axis=0)
        local_width = widths[ids]
        width_bins.append({"point": list(map(float, wp)), "width_px": float(np.median(local_width)),
                           "interval_px": list(map(float, np.percentile(local_width, [10, 90]))),
                           "samples": int(len(ids))})
    width_status = "measured_optical"
    if width < 2.5:
        width_status = "unresolved_highlight"
    elif width_mad > max(1.3, width * 0.25):
        width_status = "unstable_profile_width"
    diagnostics.update({"proposal_source": source, "support_length_px": length,
                        "support_coverage": fine["coverage"], "width_status": width_status,
                        "width_mad_px": width_mad, "candidate_optical_width_px": width,
                        "width_interval_kind": "empirical_observed_profile_10_90_percentiles_with_half_pixel_margin",
                        "width_local_samples": width_bins,
                        "width_warning": ("Optical width varies along support; interval is not a confidence interval or physical diameter."
                                          if width_interval[1] - width_interval[0] > width * .3 else
                                          "Optical image width, not calibrated physical diameter."),
                        "uncertainty_kind": "empirical_half_span_fit_sensitivity_not_calibrated",
                        "uncertainty_reference_point": list(map(float, fc)),
                        "extrapolation_warning": "Position uncertainty grows with distance from observed support; do not force tangency.",
                        "note": "Observed segment is evidence extent, not physical endpoints."})
    confidence = (0.50 + 0.20 * min(1., length / (min(h, w) * 0.35))
                  + 0.15 * min(1., float(np.median(fine["prominence"])) / 100.)
                  + 0.15 * fine["coverage"]) * max(0.4, 1 - residual / max(5., width * 2))
    confidence *= 1 - 0.25 * competitor_ratio
    return {"valid": True, "reason": "ok", "angle_deg": float(math.degrees(math.atan2(fd[1], fd[0])) % 180),
            "normal": list(map(float, normal)), "rho_px": float(fc @ normal),
            "width_px": width if width_status == "measured_optical" else None,
            "width_interval_px": width_interval, "width_reference_point": list(map(float, fc)),
            "angular_uncertainty_deg": angular_uncertainty, "normal_uncertainty_px": normal_uncertainty,
            "confidence": float(np.clip(confidence, 0, 1)), "residual_px": residual,
            "observed_segment": segment, "observed_support": int(len(points)), "diagnostics": diagnostics}
