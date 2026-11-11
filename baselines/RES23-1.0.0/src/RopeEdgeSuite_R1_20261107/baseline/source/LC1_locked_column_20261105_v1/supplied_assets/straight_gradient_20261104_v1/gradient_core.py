"""E1-conditioned straight-rope tracing from blurred original-image gradients.

E1's first point is the far, free-rope point and its second point faces the
spool. E1 is a supplied condition, never counted as a detected segment. This
module does not load a model, a mask, a label, a video, or any device module.
All returned coordinates are in the original, unmirrored image.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


DEFAULTS = {
    "method_id": "IR-GRAD-S1.0.0",
    "blur_sigma_px": 2.4,
    "step_px": 4.0,
    "max_extension_px": 5000.0,
    "width_candidates_px": [10.0, 14.0, 18.0, 23.0, 29.0, 36.0, 45.0],
    "path_width_factors": [0.85, 1.0, 1.15],
    "profile_radius_px": 65.0,
    "profile_normal_step_px": 1.0,
    "profile_tangent_offsets_px": [-6.0, 0.0, 6.0],
    "seed_profile_count": 9,
    "seed_min_supported_profiles": 4,
    "seed_min_edge_gradient": 0.85,
    "seed_min_body_contrast": 3.5,
    "path_min_edge_gradient": 0.8,
    "path_min_body_contrast": 3.0,
    "path_seed_edge_fraction": 0.22,
    "path_seed_contrast_fraction": 0.22,
    "maximum_axis_refine_angle_deg": 2.5,
    "initial_path_refit_span_px": 500.0,
    "initial_path_refit_min_span_px": 400.0,
    "initial_path_refit_min_supported_fraction": 0.75,
    "initial_path_refit_max_angle_deg": 2.0,
    "initial_path_refit_max_intercept_widths": 0.25,
    "initial_path_refit_max_support_loss_fraction": 0.10,
    "maximum_center_shift_widths": 0.30,
    "max_unobserved_gap_px": 48.0,
    "minimum_confirmed_extension_px": 32.0,
    "support_smoothing_window_steps": 5,
}


def _config(config: dict | None) -> dict:
    c = DEFAULTS | ({} if config is None else config)
    for name in ("blur_sigma_px", "step_px", "max_extension_px",
                 "profile_radius_px", "profile_normal_step_px",
                 "max_unobserved_gap_px", "minimum_confirmed_extension_px"):
        if not math.isfinite(float(c[name])) or float(c[name]) <= 0:
            raise ValueError(f"Invalid {name}")
    if not c["width_candidates_px"] or not c["path_width_factors"]:
        raise ValueError("Empty width candidates")
    return c


def _profile_samples(gray: np.ndarray, normal_gradient: np.ndarray,
                     centers: np.ndarray, direction: np.ndarray,
                     normal: np.ndarray, c: dict):
    """Sample an axial average of original-pixel cross sections.

    A sample outside the real source image is invalid even though remap needs
    a numeric border value. No reflected or padded image becomes evidence.
    """
    radius = float(c["profile_radius_px"])
    delta = float(c["profile_normal_step_px"])
    offsets = np.arange(-radius, radius + 0.5 * delta, delta, dtype=np.float32)
    tangential = np.asarray(c["profile_tangent_offsets_px"], np.float32)
    grid = (centers[:, None, None, :].astype(np.float32)
            + tangential[None, :, None, None] * direction[None, None, None, :]
            + offsets[None, None, :, None] * normal[None, None, None, :])
    h, w = gray.shape
    valid = ((grid[..., 0] >= 1) & (grid[..., 0] <= w - 2)
             & (grid[..., 1] >= 1) & (grid[..., 1] <= h - 2)).all((1, 2))
    x = np.ascontiguousarray(grid[..., 0].reshape(-1, len(offsets)))
    y = np.ascontiguousarray(grid[..., 1].reshape(-1, len(offsets)))
    intensity = cv2.remap(gray, x, y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT).reshape(len(centers), len(tangential), -1)
    gradient = cv2.remap(normal_gradient, x, y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT).reshape(len(centers), len(tangential), -1)
    return offsets, intensity.mean(axis=1), gradient.mean(axis=1), valid


def _paired_edges(offsets: np.ndarray, intensity: np.ndarray,
                  gradient: np.ndarray, nominal_width: float,
                  center_shift: float, polarity: int) -> dict:
    """Measure two separated, opposite outer gradients and body contrast.

    The brighter and darker polarities are tested symmetrically. Strong
    internal wire-strand gradients cannot alone satisfy the bilateral test.
    """
    w = float(nominal_width)
    left_band = np.flatnonzero(np.abs(offsets - (center_shift - 0.5 * w)) <= max(1.5, 0.10 * w))
    right_band = np.flatnonzero(np.abs(offsets - (center_shift + 0.5 * w)) <= max(1.5, 0.10 * w))
    body_band = np.flatnonzero(np.abs(offsets - center_shift) <= max(1.0, 0.18 * w))
    left_bg_band = np.flatnonzero((offsets >= center_shift - 1.18 * w)
                                  & (offsets <= center_shift - 0.93 * w))
    right_bg_band = np.flatnonzero((offsets >= center_shift + 0.93 * w)
                                   & (offsets <= center_shift + 1.18 * w))
    if min(map(len, (left_band, right_band, body_band, left_bg_band, right_bg_band))) == 0:
        raise ValueError("profile_radius_px too small for width candidates")
    row = np.arange(len(intensity))
    li = left_band[np.argmax(polarity * gradient[:, left_band], axis=1)]
    ri = right_band[np.argmax(-polarity * gradient[:, right_band], axis=1)]
    left_gradient = polarity * gradient[row, li]
    right_gradient = -polarity * gradient[row, ri]
    edge = np.minimum(left_gradient, right_gradient)
    body = np.median(intensity[:, body_band], axis=1)
    left_bg = np.median(intensity[:, left_bg_band], axis=1)
    right_bg = np.median(intensity[:, right_bg_band], axis=1)
    contrast = np.minimum(polarity * (body - left_bg),
                          polarity * (body - right_bg))
    measured_width = offsets[ri] - offsets[li]
    measured_center = (offsets[ri] + offsets[li]) / 2
    geometry = ((measured_width >= 0.75 * w) & (measured_width <= 1.25 * w)
                & (np.abs(measured_center - center_shift) <= 0.13 * w + 1.0))
    # The score is diagnostic and relative; it is not a rope probability.
    quality = np.minimum(np.maximum(edge, 0), 12.0) / 12.0
    quality += np.minimum(np.maximum(contrast, 0), 28.0) / 28.0
    quality -= 0.14 * abs(center_shift) / max(w, 1.0)
    quality[~geometry] = -1.0
    return {"edge": edge, "contrast": contrast, "quality": quality,
            "center": measured_center, "width": measured_width,
            "geometry": geometry, "polarity": polarity}


def _measure_seed(offsets: np.ndarray, intensity: np.ndarray,
                  gradient: np.ndarray, valid: np.ndarray, c: dict) -> dict | None:
    best = None
    for width in map(float, c["width_candidates_px"]):
        for fraction in (-0.16, -0.08, 0.0, 0.08, 0.16):
            shift = fraction * width
            for polarity in (-1, 1):
                m = _paired_edges(offsets, intensity, gradient, width, shift, polarity)
                support = (valid & m["geometry"]
                           & (m["edge"] >= float(c["seed_min_edge_gradient"])))
                support &= m["contrast"] >= float(c["seed_min_body_contrast"])
                count = int(support.sum())
                if count < int(c["seed_min_supported_profiles"]):
                    continue
                # Prefer sustained paired edges over a single large peak.
                score = (float(np.median(m["quality"][support]))
                         + 0.35 * count / len(valid)
                         - 0.025 * abs(math.log(width / 28.0)))
                if best is None or score > best["score"]:
                    best = {"score": score, "width": width,
                            "median_measured_width": float(np.median(m["width"][support])),
                            "seed_edge_gradient": float(np.median(m["edge"][support])),
                            "seed_body_contrast": float(np.median(m["contrast"][support])),
                            "seed_support_count": count,
                            "seed_support": support,
                            "seed_measured_centers": m["center"],
                            "seed_polarity": polarity}
    return best


def _refine_axis(seed: np.ndarray, direction: np.ndarray, normal: np.ndarray,
                 seed_positions: np.ndarray, measurement: dict, c: dict):
    keep = measurement["seed_support"]
    offsets = measurement["seed_measured_centers"][keep]
    distances = seed_positions[keep]
    slope = 0.0
    intercept = float(np.median(offsets))
    if len(offsets) >= 4 and np.ptp(distances) > 25:
        fit = np.polyfit(distances, offsets, 1)
        residual = offsets - np.polyval(fit, distances)
        robust = np.abs(residual - np.median(residual)) <= max(2.0, 0.18 * measurement["width"])
        if robust.sum() >= 4:
            fit = np.polyfit(distances[robust], offsets[robust], 1)
        if abs(math.degrees(math.atan(float(fit[0])))) <= float(c["maximum_axis_refine_angle_deg"]):
            slope, intercept = map(float, fit)
    refined = direction + slope * normal
    refined /= np.linalg.norm(refined)
    # seed_positions are distances from the first E1 point.
    origin = seed[-1] + (intercept + slope * np.linalg.norm(seed[-1] - seed[0])) * normal
    new_normal = np.array([-refined[1], refined[0]], np.float32)
    return origin, refined.astype(np.float32), new_normal, slope, intercept


def _measure_path(offsets: np.ndarray, intensity: np.ndarray,
                  gradient: np.ndarray, valid: np.ndarray,
                  seed_measurement: dict, c: dict) -> dict:
    n = len(valid)
    edge_threshold = max(float(c["path_min_edge_gradient"]),
                         float(c["path_seed_edge_fraction"]) * seed_measurement["seed_edge_gradient"])
    contrast_threshold = max(float(c["path_min_body_contrast"]),
                             float(c["path_seed_contrast_fraction"]) * seed_measurement["seed_body_contrast"])
    best_rank = np.full(n, -np.inf, np.float32)
    best_quality = np.full(n, -np.inf, np.float32)
    best_edge = np.zeros(n, np.float32)
    best_contrast = np.zeros(n, np.float32)
    best_center = np.zeros(n, np.float32)
    best_width = np.zeros(n, np.float32)
    best_polarity = np.zeros(n, np.int8)
    width = float(seed_measurement["median_measured_width"])
    for factor in map(float, c["path_width_factors"]):
        w = width * factor
        for fraction in (-0.24, -0.16, -0.08, 0.0, 0.08, 0.16, 0.24):
            shift = fraction * width
            for polarity in (-1, 1):
                m = _paired_edges(offsets, intensity, gradient, w, shift, polarity)
                usable = (valid & m["geometry"]
                          & (np.abs(m["center"]) <= float(c["maximum_center_shift_widths"]) * width))
                passes = ((m["edge"] >= edge_threshold)
                          & (m["contrast"] >= contrast_threshold))
                rank = m["quality"] + 10.0 * passes
                better = usable & (rank > best_rank)
                best_rank[better] = rank[better]
                best_quality[better] = m["quality"][better]
                best_edge[better] = m["edge"][better]
                best_contrast[better] = m["contrast"][better]
                best_center[better] = m["center"][better]
                best_width[better] = m["width"][better]
                best_polarity[better] = polarity
    raw = (np.isfinite(best_quality) & (best_quality >= 0)
           & (best_edge >= edge_threshold) & (best_contrast >= contrast_threshold))
    window = int(c["support_smoothing_window_steps"])
    if window < 1 or window % 2 != 1:
        raise ValueError("support_smoothing_window_steps must be positive odd")
    if window > 1:
        count = np.convolve(raw.astype(np.int16), np.ones(window, np.int16), mode="same")
        support = raw & (count >= window // 2 + 1)
    else:
        support = raw
    return {"raw": raw, "support": support, "edge": best_edge,
            "contrast": best_contrast, "center": best_center,
            "width": best_width, "polarity": best_polarity,
            "quality": best_quality, "valid": valid,
            "edge_threshold": edge_threshold,
            "contrast_threshold": contrast_threshold}


def _trace_axis(gray: np.ndarray, normal_gradient: np.ndarray,
                origin: np.ndarray, direction: np.ndarray, normal: np.ndarray,
                seed_measurement: dict, c: dict) -> dict:
    h, w = gray.shape
    tangent_radius = max(abs(float(v)) for v in c["profile_tangent_offsets_px"])
    exit_distances = []
    for coordinate, slope, size, normal_part in zip(origin, direction, (w, h), normal):
        margin = abs(float(normal_part)) * float(c["profile_radius_px"])
        margin += abs(float(slope)) * tangent_radius + 2.0
        if slope > 1e-8:
            exit_distances.append((size - 1 - margin - coordinate) / slope)
        elif slope < -1e-8:
            exit_distances.append((margin - coordinate) / slope)
    border_limit = min(exit_distances) if exit_distances else float(c["max_extension_px"])
    search_limit = max(0.0, min(border_limit, float(c["max_extension_px"])))
    step = float(c["step_px"])
    distances = np.arange(step, search_limit + 1e-4, step, dtype=np.float32)
    if len(distances) == 0:
        return {"distances": distances, "border_limit": border_limit}
    centers = origin + distances[:, None] * direction
    offsets, intensity, gradient, valid = _profile_samples(
        gray, normal_gradient, centers, direction, normal, c)
    path = _measure_path(offsets, intensity, gradient, valid, seed_measurement, c)
    return {"distances": distances, "centers": centers, "path": path,
            "border_limit": border_limit}


def _refit_from_initial_path(trace: dict, origin: np.ndarray,
                             direction: np.ndarray, normal: np.ndarray,
                             seed_width: float, c: dict):
    """One small straight-axis correction from the first 500 px only.

    No later endpoint or evaluation result participates in the fit. The same
    rule applies to every image; a long later continuation cannot earn itself
    a more favorable axis.
    """
    distances = trace["distances"]
    path = trace.get("path")
    if path is None:
        return None, {"applied": False, "reason": "no_search_area"}
    early = distances <= float(c["initial_path_refit_span_px"])
    supported = early & path["support"]
    count = int(supported.sum())
    available = int(early.sum())
    if (count < 20 or available < 20
            or count / available < float(c["initial_path_refit_min_supported_fraction"])):
        return None, {"applied": False, "reason": "insufficient_initial_bilateral_support",
                      "supported_count": count, "available_count": available}
    s = distances[supported].astype(float)
    offset = path["center"][supported].astype(float)
    span = float(np.ptp(s))
    if span < float(c["initial_path_refit_min_span_px"]):
        return None, {"applied": False, "reason": "initial_support_span_too_short",
                      "supported_count": count, "span_px": span}
    fit = np.polyfit(s, offset, 1)
    keep = np.ones(len(s), bool)
    for _ in range(3):
        residual = offset - np.polyval(fit, s)
        center = float(np.median(residual[keep]))
        spread = float(1.4826 * np.median(np.abs(residual[keep] - center)))
        keep = np.abs(residual - center) <= max(1.5, 0.12 * seed_width, 2.5 * spread)
        if keep.sum() < 20 or np.ptp(s[keep]) < float(c["initial_path_refit_min_span_px"]):
            return None, {"applied": False, "reason": "initial_axis_fit_unstable",
                          "supported_count": count, "span_px": span}
        fit = np.polyfit(s[keep], offset[keep], 1)
    slope, intercept = map(float, fit)
    angle = math.degrees(math.atan(slope))
    receipt = {"applied": False, "reason": "candidate",
               "supported_count": count, "available_count": available,
               "span_px": span, "fit_count": int(keep.sum()),
               "angle_delta_deg": angle, "intercept_px": intercept,
               "fit_residual_median_px": float(np.median(np.abs(offset[keep] - np.polyval(fit, s[keep]))))}
    if (abs(angle) > float(c["initial_path_refit_max_angle_deg"])
            or abs(intercept) > float(c["initial_path_refit_max_intercept_widths"]) * seed_width):
        receipt["reason"] = "bounded_axis_correction_exceeded"
        return None, receipt
    if abs(angle) < 0.03 and abs(intercept) < 0.25:
        receipt["reason"] = "negligible_axis_correction"
        return None, receipt
    new_direction = direction + slope * normal
    new_direction /= np.linalg.norm(new_direction)
    new_normal = np.array([-new_direction[1], new_direction[0]], np.float32)
    new_origin = origin + intercept * normal
    return (new_origin.astype(np.float32), new_direction.astype(np.float32), new_normal), receipt


def _no_prediction(seed: np.ndarray, reason: str, c: dict,
                   seed_measurement: dict | None = None) -> dict:
    return {"method_id": c["method_id"], "status": "UNKNOWN",
            "stop_reason": reason, "input_seed_xy": seed.tolist(),
            "seed_is_input_not_detection": True,
            "points_xy": [], "supported_segments_xy": [],
            "endpoint_xy": None, "width_px": None,
            "supported_extension_px": 0.0,
            "seed_measurement": seed_measurement, "step_records": [],
            "physical_contact": None, "same_material_identity": None}


def analyze(image_bgr: np.ndarray, seed_xy: Any,
            config: dict | None = None) -> tuple[dict, dict]:
    """Return image fields and a JSON-compatible E1-conditioned prediction.

    `points_xy` starts at E1's *second* point and contains only subsequent
    supported axial samples. Small failed intervals may be skipped, not filled;
    `supported_segments_xy` keeps those visible runs separate for rendering.
    A disappearance is an optical endpoint, not proven physical contact.
    """
    c = _config(config)
    if (not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8
            or image_bgr.ndim != 3 or image_bgr.shape[2] != 3):
        raise ValueError("Expected original uint8 BGR image")
    sigma = float(c["blur_sigma_px"])
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=sigma, sigmaY=sigma,
                               borderType=cv2.BORDER_REFLECT_101)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
    magnitude = cv2.magnitude(gx, gy)
    fields = {"blurred_bgr": blurred,
              "gradient_magnitude": magnitude.astype(np.float32, copy=False)}
    try:
        seed = np.asarray(seed_xy, np.float32).reshape(2, 2)
    except (ValueError, TypeError):
        seed = np.empty((0, 2), np.float32)
    if seed.shape != (2, 2) or not np.isfinite(seed).all():
        return fields, _no_prediction(seed, "invalid_E1_two_point_input", c)
    vector = seed[1] - seed[0]
    seed_length = float(np.linalg.norm(vector))
    if seed_length < 40:
        return fields, _no_prediction(seed, "E1_too_short", c)
    direction = vector / seed_length
    normal = np.array([-direction[1], direction[0]], np.float32)
    cv2.addWeighted(gx, float(normal[0]), gy, float(normal[1]), 0,
                    dst=gx)
    normal_gradient = gx
    del gy
    positions = np.linspace(0.12, 0.88, int(c["seed_profile_count"]), dtype=np.float32) * seed_length
    centers = seed[0] + positions[:, None] * direction
    offsets, intensity, gradient, valid = _profile_samples(
        gray, normal_gradient, centers, direction, normal, c)
    measured = _measure_seed(offsets, intensity, gradient, valid, c)
    if measured is None:
        return fields, _no_prediction(seed, "E1_bilateral_gradient_not_confirmed", c)
    origin, direction, normal, axis_slope, axis_intercept = _refine_axis(
        seed, direction, normal, positions, measured, c)
    # Reproject the gradient onto the refined normal. The angular correction is
    # small, but this keeps evidence and geometry in the same coordinates.
    if abs(axis_slope) > 1e-5:
        del normal_gradient
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
        cv2.addWeighted(gx, float(normal[0]), gy, float(normal[1]), 0,
                        dst=gx)
        normal_gradient = gx
        del gy
    trace = _trace_axis(gray, normal_gradient, origin, direction, normal, measured, c)
    distances = trace["distances"]
    if len(distances) == 0:
        out = _no_prediction(seed, "no_search_area_after_E1", c)
        out["seed_measurement"] = {"width_px": measured["median_measured_width"],
                                   "supported_profiles": measured["seed_support_count"]}
        return fields, out
    candidate, refit_receipt = _refit_from_initial_path(
        trace, origin, direction, normal, measured["median_measured_width"], c)
    if candidate is not None:
        new_origin, new_direction, new_normal = candidate
        del normal_gradient
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
        cv2.addWeighted(gx, float(new_normal[0]), gy, float(new_normal[1]), 0,
                        dst=gx)
        normal_gradient = gx
        del gy
        retraced = _trace_axis(gray, normal_gradient, new_origin, new_direction,
                               new_normal, measured, c)
        span = float(c["initial_path_refit_span_px"])
        old_early = trace["distances"] <= span
        new_early = retraced["distances"] <= span
        old_fraction = float(trace["path"]["support"][old_early].mean())
        new_fraction = (float(retraced["path"]["support"][new_early].mean())
                        if new_early.any() else 0.0)
        refit_receipt["old_initial_support_fraction"] = old_fraction
        refit_receipt["new_initial_support_fraction"] = new_fraction
        if new_fraction >= old_fraction - float(c["initial_path_refit_max_support_loss_fraction"]):
            trace = retraced
            origin, direction, normal = new_origin, new_direction, new_normal
            refit_receipt["applied"] = True
            refit_receipt["reason"] = "initial_bilateral_axis_refit"
        else:
            refit_receipt["reason"] = "refit_initial_support_loss"
    distances = trace["distances"]
    centers = trace["centers"]
    path = trace["path"]
    border_limit = trace["border_limit"]
    step = float(c["step_px"])
    max_gap_steps = max(1, int(math.floor(float(c["max_unobserved_gap_px"]) / step)))
    last_supported = -1
    gap = 0
    stop_index = len(distances) - 1
    stop_reason = "image_border" if border_limit <= float(c["max_extension_px"]) else "search_budget_reached"
    for k, supported in enumerate(path["support"]):
        if supported:
            last_supported = k
            gap = 0
        else:
            gap += 1
            if gap > max_gap_steps:
                stop_index = k
                stop_reason = "sustained_gradient_disappearance"
                break
    if last_supported < 0:
        out = _no_prediction(seed, "no_supported_extension_after_E1", c)
        out["seed_measurement"] = {"width_px": measured["median_measured_width"],
                                   "supported_profiles": measured["seed_support_count"],
                                   "median_edge_gradient": measured["seed_edge_gradient"],
                                   "median_body_contrast": measured["seed_body_contrast"]}
        return fields, out
    supported_length = float(distances[last_supported])
    supported_count = int(path["support"][:last_supported + 1].sum())
    status = ("PARTIAL" if supported_count * step >= float(c["minimum_confirmed_extension_px"])
              else "UNKNOWN")
    if status == "UNKNOWN":
        stop_reason = "insufficient_confirmed_extension"
    selected = np.flatnonzero(path["support"][:last_supported + 1])
    # Smooth lateral offsets over measured rows only. This affects display
    # geometry, never the evidence decision or the optical end position.
    smoothed_center = path["center"].copy()
    for k in selected:
        local = np.arange(max(0, k - 2), min(len(path["support"]), k + 3))
        nearby = local[path["support"][local]]
        smoothed_center[k] = np.median(path["center"][nearby])
    supported_xy = centers[selected] + smoothed_center[selected, None] * normal
    points_xy = [seed[1].astype(float).tolist()] + supported_xy.astype(float).tolist() if status == "PARTIAL" else []
    segments = []
    if status == "PARTIAL":
        for run in np.split(np.arange(len(selected)), np.flatnonzero(np.diff(selected) > 1) + 1):
            if len(run):
                segments.append(supported_xy[run].astype(float).tolist())
    records = []
    for k in range(stop_index + 1):
        records.append({"distance_after_E1_px": float(distances[k]),
                        "axis_xy": centers[k].astype(float).tolist(),
                        "supported": bool(path["support"][k]),
                        "raw_bilateral_supported": bool(path["raw"][k]),
                        "valid_original_samples": bool(path["valid"][k]),
                        "paired_edge_gradient": float(path["edge"][k]),
                        "body_to_two_background_contrast": float(path["contrast"][k]),
                        "center_offset_px": float(path["center"][k]),
                        "measured_width_px": float(path["width"][k]),
                        "polarity": int(path["polarity"][k])})
    return fields, {"method_id": c["method_id"], "status": status,
                    "stop_reason": stop_reason, "input_seed_xy": seed.astype(float).tolist(),
                    "seed_is_input_not_detection": True,
                    "points_xy": points_xy, "supported_segments_xy": segments,
                    "endpoint_xy": supported_xy[-1].astype(float).tolist() if status == "PARTIAL" else None,
                    "width_px": float(measured["median_measured_width"]),
                    "supported_extension_px": supported_length if status == "PARTIAL" else 0.0,
                    "supported_sample_count": int(len(selected)),
                    "edge_threshold": float(path["edge_threshold"]),
                    "contrast_threshold": float(path["contrast_threshold"]),
                    "axis_direction_xy": direction.astype(float).tolist(),
                    "axis_refine_angle_deg": float(math.degrees(math.atan(axis_slope))
                                                   + (refit_receipt["angle_delta_deg"] if refit_receipt["applied"] else 0.0)),
                    "axis_intercept_px": float(axis_intercept),
                    "initial_path_axis_refit": refit_receipt,
                    "seed_measurement": {
                        "width_px": float(measured["median_measured_width"]),
                        "supported_profiles": int(measured["seed_support_count"]),
                        "median_edge_gradient": float(measured["seed_edge_gradient"]),
                        "median_body_contrast": float(measured["seed_body_contrast"]),
                        "selected_polarity": int(measured["seed_polarity"]),
                        "input_length_px": seed_length},
                    "step_records": records,
                    "physical_contact": None, "same_material_identity": None}
