"""Conditioned straight-rope evidence, with re-observation after missing spans.

The geometry is exactly one infinite line.  Only intervals supported by image
measurements are labelled observed; gaps are UNKNOWN even when a later segment
on the same line is accepted.  The input candidate comes from guidance.py.
No target labels, neural responses, or device modules are loaded here.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


DEFAULTS = {
    "method_id": "IR-GRAD-S2.0.0",
    "blur_sigma_px": 2.1,
    "step_px": 4.0,
    "max_search_px": 5000.0,
    "width_min_px": 8.0,
    "width_max_px": 50.0,
    "width_factors": [0.82, 1.0, 1.18],
    "center_shift_fractions": [-0.38, -0.28, -0.18, -0.09, 0.0, 0.09, 0.18, 0.28, 0.38],
    "cross_step_px": 1.0,
    "cross_radius_widths": 1.8,
    "tangent_offsets_px": [-7.0, 0.0, 7.0],
    "min_weaker_edge": 0.7,
    "min_stronger_edge": 1.5,
    "min_weaker_contrast": 1.8,
    "min_stronger_contrast": 4.0,
    "min_quality": 0.48,
    "support_smooth_steps": 5,
    "min_initial_run_px": 36.0,
    "max_initial_offset_px": 200.0,
    "min_reobserved_run_px": 52.0,
    "max_reobservation_gap_px": 900.0,
    "max_local_gap_px": 28.0,
    "max_lateral_center_widths": 0.42,
    "reobservation_width_tolerance": 0.38,
}


def _unit(x: Any) -> np.ndarray:
    a = np.asarray(x, np.float32).reshape(2)
    n = float(np.linalg.norm(a))
    if not np.isfinite(n) or n < 1e-6:
        raise ValueError("invalid axis direction")
    return a / n


def _sample(gray: np.ndarray, grad_n: np.ndarray, centers: np.ndarray,
            d: np.ndarray, n: np.ndarray, width: float, cfg: dict):
    step = float(cfg["cross_step_px"])
    radius = max(65.0, float(cfg["cross_radius_widths"]) * width)
    offsets = np.arange(-radius, radius + 0.5 * step, step, dtype=np.float32)
    ts = np.asarray(cfg["tangent_offsets_px"], np.float32)
    xy = (centers[:, None, None, :].astype(np.float32)
          + ts[None, :, None, None] * d[None, None, None, :]
          + offsets[None, None, :, None] * n[None, None, None, :])
    h, w = gray.shape
    valid = ((xy[..., 0] >= 1) & (xy[..., 0] <= w - 2)
             & (xy[..., 1] >= 1) & (xy[..., 1] <= h - 2)).all(axis=(1, 2))
    shape = (len(centers), len(ts), len(offsets))
    mapx = np.ascontiguousarray(xy[..., 0].reshape(-1, len(offsets)))
    mapy = np.ascontiguousarray(xy[..., 1].reshape(-1, len(offsets)))
    inten = cv2.remap(gray, mapx, mapy, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT).reshape(shape).mean(axis=1)
    grad = cv2.remap(grad_n, mapx, mapy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT).reshape(shape).mean(axis=1)
    return offsets, inten, grad, valid


def _feature_profiles(offsets: np.ndarray, inten: np.ndarray, grad: np.ndarray,
                      valid: np.ndarray, width: float, cfg: dict) -> dict:
    """Estimate independent flank evidence on every fixed-axis cross section.

    Each side may have a different background contrast. A polarity switch
    between sides or along the rope is allowed and recorded, with a soft cost.
    This is intentionally unlike the S1 requirement for both sides to be
    brighter (or both darker) than their surroundings.
    """
    count = len(valid)
    best_rank = np.full(count, -np.inf, np.float32)
    out = {key: np.zeros(count, np.float32) for key in
           ("quality", "edge_left", "edge_right", "contrast_left",
            "contrast_right", "center_shift_px", "measured_width_px")}
    out["opposite_edge_signs"] = np.zeros(count, bool)
    idx = np.arange(count)
    for factor in cfg["width_factors"]:
        w = width * float(factor)
        for fraction in cfg["center_shift_fractions"]:
            shift = width * float(fraction)
            l = np.flatnonzero(abs(offsets - (shift - 0.5 * w)) <= max(1.5, 0.13 * w))
            r = np.flatnonzero(abs(offsets - (shift + 0.5 * w)) <= max(1.5, 0.13 * w))
            body = np.flatnonzero(abs(offsets - shift) <= max(1.0, 0.18 * w))
            lbg = np.flatnonzero((offsets >= shift - 1.25*w) & (offsets <= shift - 0.90*w))
            rbg = np.flatnonzero((offsets >= shift + 0.90*w) & (offsets <= shift + 1.25*w))
            if min(map(len, (l, r, body, lbg, rbg))) == 0:
                continue
            li = l[np.argmax(abs(grad[:, l]), axis=1)]
            ri = r[np.argmax(abs(grad[:, r]), axis=1)]
            gl = grad[idx, li]
            gr = grad[idx, ri]
            left = abs(gl)
            right = abs(gr)
            center_intensity = np.median(inten[:, body], axis=1)
            cl = abs(center_intensity - np.median(inten[:, lbg], axis=1))
            cr = abs(center_intensity - np.median(inten[:, rbg], axis=1))
            eweak = np.minimum(left, right)
            estrong = np.maximum(left, right)
            cweak = np.minimum(cl, cr)
            cstrong = np.maximum(cl, cr)
            measured_width = offsets[ri] - offsets[li]
            measured_center = 0.5 * (offsets[li] + offsets[ri])
            geometry = ((measured_width >= 0.70*w) & (measured_width <= 1.30*w)
                        & (abs(measured_center) <= float(cfg["max_lateral_center_widths"])*width))
            opposite = gl * gr < 0
            # Scale-limited score avoids a single intense glare dominating.
            quality = (0.37 * np.minimum(eweak / 3.0, 1.0)
                       + 0.16 * np.minimum(estrong / 6.0, 1.0)
                       + 0.20 * np.minimum(cweak / 10.0, 1.0)
                       + 0.12 * np.minimum(cstrong / 20.0, 1.0)
                       + 0.15 * opposite.astype(np.float32)
                       - 0.045 * abs(measured_center) / max(width, 1.0))
            hard = (valid & geometry
                    & (eweak >= float(cfg["min_weaker_edge"]))
                    & (estrong >= float(cfg["min_stronger_edge"]))
                    & (cweak >= float(cfg["min_weaker_contrast"]))
                    & (cstrong >= float(cfg["min_stronger_contrast"])))
            rank = quality + 2.0 * hard.astype(np.float32)
            better = geometry & valid & (rank > best_rank)
            best_rank[better] = rank[better]
            for key, values in (("quality", quality), ("edge_left", left), ("edge_right", right),
                                ("contrast_left", cl), ("contrast_right", cr),
                                ("center_shift_px", measured_center),
                                ("measured_width_px", measured_width)):
                out[key][better] = values[better]
            out["opposite_edge_signs"][better] = opposite[better]
    weaker_e = np.minimum(out["edge_left"], out["edge_right"])
    stronger_e = np.maximum(out["edge_left"], out["edge_right"])
    weaker_c = np.minimum(out["contrast_left"], out["contrast_right"])
    stronger_c = np.maximum(out["contrast_left"], out["contrast_right"])
    raw = (valid & (weaker_e >= float(cfg["min_weaker_edge"]))
           & (stronger_e >= float(cfg["min_stronger_edge"]))
           & (weaker_c >= float(cfg["min_weaker_contrast"]))
           & (stronger_c >= float(cfg["min_stronger_contrast"]))
           & (out["quality"] >= float(cfg["min_quality"])))
    win = int(cfg["support_smooth_steps"])
    if win > 1:
        count_win = np.convolve(raw.astype(np.int16), np.ones(win, np.int16), mode="same")
        support = raw & (count_win >= win // 2 + 1)
    else:
        support = raw
    out["raw_support"] = raw
    out["support"] = support
    out["valid"] = valid
    return out


def _runs(mask: np.ndarray, step: float) -> list[tuple[int, int]]:
    k = np.flatnonzero(mask)
    if len(k) == 0:
        return []
    return [(int(group[0]), int(group[-1]))
            for group in np.split(k, np.flatnonzero(np.diff(k) > 1) + 1)]


def _border_limit(origin: np.ndarray, direction: np.ndarray,
                  normal: np.ndarray, image_shape: tuple[int, int],
                  width: float, cfg: dict) -> float:
    h, w = image_shape
    margin = max(65.0, float(cfg["cross_radius_widths"]) * width)
    margin += max(abs(float(x)) for x in cfg["tangent_offsets_px"]) + 2.0
    limits = []
    for pos, delta, size in ((origin[0], direction[0], w), (origin[1], direction[1], h)):
        if delta > 1e-8:
            limits.append((size - 1 - margin - pos) / delta)
        elif delta < -1e-8:
            limits.append((margin - pos) / delta)
    return max(0.0, min(limits, default=float(cfg["max_search_px"])))


def analyze_straight(image_bgr: np.ndarray, candidate: dict,
                     config: dict | None = None) -> dict:
    """Evaluate one candidate guidance axis without changing its identity.

    Return a single straight line and intervals measured from the projection
    of seed_xy[1] in the forward direction. Each guidance candidate must be
    passed through this same function before any later ranking.
    """
    cfg = DEFAULTS | (config or {})
    if image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("original BGR uint8 image required")
    seed = np.asarray(candidate["seed_xy"], np.float32).reshape(2, 2)
    if not np.isfinite(seed).all():
        raise ValueError("nonfinite seed")
    d = _unit(candidate.get("axis_direction_xy", seed[1] - seed[0]))
    if float(np.dot(d, seed[1] - seed[0])) <= 0:
        d = -d
    n = np.array([-d[1], d[0]], np.float32)
    width = float(candidate.get("width_px") or 22.0)
    if not float(cfg["width_min_px"]) <= width <= float(cfg["width_max_px"]):
        raise ValueError("guidance optical width out of range")
    point = np.asarray(candidate.get("axis_point_xy", seed[1]), np.float32).reshape(2)
    origin = seed[1] + float(np.dot(point - seed[1], n)) * n
    limit = min(_border_limit(origin, d, n, image_bgr.shape[:2], width, cfg),
                float(cfg["max_search_px"]))
    step = float(cfg["step_px"])
    distances = np.arange(step, limit + 1e-4, step, dtype=np.float32)
    base = {"method_id": cfg["method_id"], "candidate_id": candidate.get("candidate_id"),
            "input_seed_xy": seed.astype(float).tolist(),
            "line_origin_xy": origin.astype(float).tolist(),
            "line_direction_xy": d.astype(float).tolist(), "width_px": width,
            "supported_intervals_px": [], "unknown_intervals_px": [],
            "endpoint_xy": None, "supported_length_px": 0.0,
            "status": "UNKNOWN", "stop_reason": "no_search_area",
            "score": 0.0, "step_records": [],
            "straight_line_invariant": True}
    if len(distances) == 0:
        return base
    blur = cv2.GaussianBlur(image_bgr, (0, 0), float(cfg["blur_sigma_px"]))
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
    grad_n = cv2.addWeighted(gx, float(n[0]), gy, float(n[1]), 0)
    centers = origin[None, :] + distances[:, None] * d[None, :]
    offsets, inten, grad, valid = _sample(gray, grad_n, centers, d, n, width, cfg)
    features = _feature_profiles(offsets, inten, grad, valid, width, cfg)
    runs = _runs(features["support"], step)
    min_initial = int(math.ceil(float(cfg["min_initial_run_px"]) / step))
    initial = next(((a, b) for a, b in runs
                    if a * step <= float(cfg["max_initial_offset_px"]) and b-a+1 >= min_initial), None)
    if initial is None:
        base["stop_reason"] = "no_initial_supported_rope"
        return base
    chosen = [initial]
    last = initial[1]
    for a, b in runs:
        if a <= last:
            continue
        gap = (a - last - 1) * step
        length = (b - a + 1) * step
        if gap <= float(cfg["max_local_gap_px"]):
            chosen.append((a, b)); last = b; continue
        if gap > float(cfg["max_reobservation_gap_px"]):
            break
        if length < float(cfg["min_reobserved_run_px"]):
            continue
        # Same straight axis and stable optical width are independent of
        # absolute brightness, so a dark-to-bright change is permitted.
        segment_width = float(np.median(features["measured_width_px"][a:b+1]))
        if abs(segment_width / width - 1) > float(cfg["reobservation_width_tolerance"]):
            continue
        chosen.append((a, b)); last = b
    intervals = [[float(distances[a]), float(distances[b])] for a, b in chosen]
    gaps = [[float(distances[b] + step), float(distances[a] - step)]
            for (_, b), (a, _) in zip(chosen[:-1], chosen[1:]) if a > b + 1]
    selected_mask = np.zeros(len(distances), bool)
    for a, b in chosen:
        selected_mask[a:b+1] = features["support"][a:b+1]
    selected_quality = features["quality"][selected_mask]
    endpoint = origin + float(distances[last]) * d
    base.update({"status": "PARTIAL", "stop_reason": "last_reobserved_gradient_support",
                 "supported_intervals_px": intervals, "unknown_intervals_px": gaps,
                 "endpoint_xy": endpoint.astype(float).tolist(),
                 "supported_length_px": float(selected_mask.sum() * step),
                 "axis_extent_px": float(distances[last]),
                 "score": float(np.mean(selected_quality)) if len(selected_quality) else 0.0,
                 "reobserved_after_gap_count": sum((a-b-1)*step > float(cfg["max_local_gap_px"])
                                                    for (_, b), (a, _) in zip(chosen[:-1], chosen[1:])),
                 "step_records": [{"distance_after_seed_px": float(t),
                                   "supported": bool(features["support"][k]),
                                   "raw_supported": bool(features["raw_support"][k]),
                                   "quality": float(features["quality"][k]),
                                   "left_edge": float(features["edge_left"][k]),
                                   "right_edge": float(features["edge_right"][k]),
                                   "left_contrast": float(features["contrast_left"][k]),
                                   "right_contrast": float(features["contrast_right"][k]),
                                   "opposite_edge_signs": bool(features["opposite_edge_signs"][k]),
                                   "center_shift_px": float(features["center_shift_px"][k]),
                                   "measured_width_px": float(features["measured_width_px"][k])}
                                  for k, t in enumerate(distances[:last+1])]})
    return base
