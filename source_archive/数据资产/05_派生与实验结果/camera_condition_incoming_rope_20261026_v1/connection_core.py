"""活动绳自由直段到卷面转弯的有限、多候选、原图证据研究。

入口只接收原图、已观测自由段、SAM 卷面掩码和 C3 旧卷面圆锥族。
C3 仅提供相邻旧圈的几何竞争结构；它的拟合线、自由段直线外推，
都不能自行变成原图支持。输出仅为二维视觉接续假设，不是三维接触、
同根身份、t0/t1/t2 或设备动作指令。本模块没有文件与设备访问。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class ConnectionParameters:
    max_candidates: int = 6
    search_forward_widths: float = 90.
    search_forward_max_px: float = 1800.
    lateral_half_widths: float = 18.
    beam_size: int = 48
    step_u_widths: float = .40
    step_v_widths: float = .25
    max_image_gap_widths: float = 5.
    minimum_candidate_extent_widths: float = 8.
    minimum_observed_extension_widths: float = 3.
    minimum_pair_contrast_gray: float = 7.
    minimum_strip_pair_score: float = .27
    minimum_original_pair_fraction_for_turn: float = .35
    minimum_candidate_separation_widths: float = .55
    old_band_distance_pitches: float = .65
    maximum_turn_interval_widths: float = 24.
    turn_lateral_deviation_widths: float = .45
    turn_slope_threshold: float = .12
    minimum_score_margin: float = .10


def _unit(value: object) -> np.ndarray | None:
    try:
        vector = np.asarray(value, float)
        if vector.shape != (2,) or not np.isfinite(vector).all():
            return None
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-8 else None
    except (TypeError, ValueError):
        return None


def _empty(image_shape: tuple[int, int], status: str, reason: str) -> dict:
    return {"status": status, "selected_candidate_id": None,
            "g0_baseline_candidate_id": None, "candidates": [],
            "rejected": [{"reason": reason}],
            "diagnostics": {"status": status, "reason": reason,
                            "physical_contact_confirmed": False,
                            "image_identity_confirmed": False}}


def _input(image: np.ndarray, surface_mask: np.ndarray,
           free_segment: dict) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray] | dict:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image_bgr must be full-resolution uint8 BGR")
    if (not isinstance(surface_mask, np.ndarray) or surface_mask.dtype != np.uint8 or
            surface_mask.shape != image.shape[:2] or
            np.any((surface_mask != 0) & (surface_mask != 255))):
        raise ValueError("surface_mask must be aligned uint8 0/255")
    if not isinstance(free_segment, dict):
        raise ValueError("free_segment must be a dictionary")
    if free_segment.get("status") != "observed":
        return _empty(image.shape[:2], "UNKNOWN", "independent_free_segment_not_observed")
    try:
        width = float(free_segment["width_px"])
    except (KeyError, ValueError, TypeError):
        return _empty(image.shape[:2], "UNKNOWN", "free_optical_width_unavailable")
    if not math.isfinite(width) or not 4 <= width <= 100:
        return _empty(image.shape[:2], "UNKNOWN", "free_optical_width_outside_range")
    points = np.asarray(free_segment.get("observed_points_xy", []), float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        points = np.asarray(free_segment.get("observed_segment_xy", []), float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        return _empty(image.shape[:2], "UNKNOWN", "free_observed_points_unavailable")
    h, w = image.shape[:2]
    if ((points[:, 0] < 0).any() or (points[:, 0] >= w).any() or
            (points[:, 1] < 0).any() or (points[:, 1] >= h).any()):
        return _empty(image.shape[:2], "UNKNOWN", "free_points_outside_image")
    direction = _unit(free_segment.get("toward_surface_direction_xy"))
    if direction is None:
        return _empty(image.shape[:2], "UNKNOWN", "free_toward_surface_direction_unknown")
    span = float(np.ptp(points @ direction))
    if span < 3 * width:
        return _empty(image.shape[:2], "UNKNOWN", "free_observed_span_too_short")
    # 取独立实测直段朝卷面的末端；预测方向只确定搜索坐标，不算接触证据。
    entry = points[int(np.argmax(points @ direction))].astype(float)
    if (entry[0] < 2 or entry[0] >= w - 2 or entry[1] < 2 or entry[1] >= h - 2):
        return _empty(image.shape[:2], "UNKNOWN", "free_entry_at_image_border")
    return image, surface_mask > 0, width, entry, direction


def _c3_bands(c3_fit: dict, x_reference: float, shape: tuple[int, int],
              fallback_pitch: float) -> tuple[list[dict], float, dict]:
    if not isinstance(c3_fit, dict):
        raise ValueError("c3_fit must be a dictionary")
    h, _ = shape
    rows = []
    reasons = Counter()
    for curve in c3_fit.get("curves", []):
        model = curve.get("model", {})
        required = ("normalized_k", "normalized_b", "normalized_c",
                    "u_center_x_px", "u_scale_x_px")
        try:
            vals = [float(model[key]) for key in required]
        except (KeyError, TypeError, ValueError):
            reasons["invalid_c3_model"] += 1
            continue
        if not all(math.isfinite(v) for v in vals) or vals[2] < 0 or vals[4] <= 0:
            reasons["invalid_c3_model"] += 1
            continue
        u = (x_reference - vals[3]) / vals[4]
        y = vals[0] + vals[1] * u + vals[2] * u * u
        if not -fallback_pitch <= y <= h + fallback_pitch:
            reasons["c3_model_outside_reference_column"] += 1
            continue
        rows.append({"group_id": curve.get("group_id", curve.get("curve_id")),
                     "model": {key: value for key, value in zip(required, vals)},
                     "y_reference": float(y)})
    rows.sort(key=lambda row: row["y_reference"])
    differences = np.diff([row["y_reference"] for row in rows])
    plausible = differences[(differences >= .55 * fallback_pitch) &
                            (differences <= 1.7 * fallback_pitch)]
    pitch = float(np.median(plausible)) if len(plausible) else fallback_pitch
    bands = []
    for upper, lower in zip(rows[:-1], rows[1:]):
        separation = lower["y_reference"] - upper["y_reference"]
        if not .55 * pitch <= separation <= 1.7 * pitch:
            reasons["c3_neighbor_spacing_implausible"] += 1
            continue
        bands.append({"band_id": f"{upper['group_id']}__{lower['group_id']}",
                      "group_ids": [upper["group_id"], lower["group_id"]],
                      "upper_model": upper["model"], "lower_model": lower["model"],
                      "reference_gap_px": float(separation)})
    return bands, pitch, {"accepted_c3_rows": len(rows), "adjacent_bands": len(bands),
                          "rejection_counts": dict(reasons),
                          "pitch_source": "accepted_C3_adjacent_group_spacing_or_free_optical_width"}


def _predict(model: dict, x: np.ndarray) -> np.ndarray:
    u = (x - model["u_center_x_px"]) / model["u_scale_x_px"]
    return model["normalized_k"] + model["normalized_b"] * u + model["normalized_c"] * u * u


def _strip(image: np.ndarray, surface: np.ndarray, entry: np.ndarray,
           direction: np.ndarray, width: float, p: ConnectionParameters) -> dict:
    h, w = surface.shape
    n = np.array([-direction[1], direction[0]])
    length = min(p.search_forward_widths * width, p.search_forward_max_px)
    du = float(np.clip(p.step_u_widths * width, 3., 9.))
    dv = float(np.clip(p.step_v_widths * width, 1.5, 5.))
    u = np.arange(0., length + du * .5, du)
    v = np.arange(-p.lateral_half_widths * width,
                  p.lateral_half_widths * width + dv * .5, dv)
    xy = entry + u[:, None, None] * direction + v[None, :, None] * n
    mx, my = xy[..., 0].astype(np.float32), xy[..., 1].astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sampled = cv2.remap(gray, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
    source_max = np.max(image, axis=2)
    sampled_max = cv2.remap(source_max, mx, my, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
    inside = (mx >= 2) & (mx <= w - 3) & (my >= 2) & (my <= h - 3)
    on_surface = cv2.remap(surface.astype(np.uint8), mx, my, cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT) > 0
    smooth = cv2.GaussianBlur(sampled, (0, 0), sigmaX=1.0,
                              sigmaY=max(.8, .55 * width / du))
    score = np.zeros_like(smooth, dtype=np.float32)
    phase = np.zeros(smooth.shape, np.int8)
    for factor in (.70, 1.0, 1.30):
        k = max(2, int(round(.5 * width * factor / dv)))
        if 2 * k >= smooth.shape[1]:
            continue
        center = smooth[:, k:-k]
        left, right = smooth[:, :-2 * k], smooth[:, 2 * k:]
        bright = np.minimum(center - left, center - right)
        dark = np.minimum(left - center, right - center)
        best = np.maximum(bright, dark)
        proposed = np.clip((best - p.minimum_pair_contrast_gray) / 22., 0., 1.)
        target = score[:, k:-k]
        improved = proposed > target
        target[improved] = proposed[improved]
        sign = np.where(bright >= dark, 1, -1).astype(np.int8)
        phase[:, k:-k][improved] = sign[improved]
    local_contrast = cv2.blur(sampled, (max(3, int(round(width / dv))) | 1, 1))
    available = (inside & (sampled_max < 253) & (sampled >= 10) &
                 (np.abs(sampled - local_contrast) >= 2.))
    score[~available] = 0.
    return {"u": u, "v": v, "xy": xy, "score": score, "phase": phase,
            "available": available, "inside": inside, "surface": on_surface,
            "du": du, "dv": dv, "normal": n, "direction": direction,
            "original_image_sampling": True}


def _beam_paths(strip: dict, width: float, p: ConnectionParameters) -> tuple[list[dict], dict]:
    u, v = strip["u"], strip["v"]
    mid = int(np.argmin(abs(v)))
    beam = [{"cost": 0., "path": (mid,), "slope": 0, "gap_px": 0.,
             "observed_count": 0, "entered_surface": bool(strip["surface"][0, mid])}]
    terminal = []
    rejected = Counter()
    snapshot_stride = max(1, int(round(10 * width / strip["du"])))
    max_gap = p.max_image_gap_widths * width
    for i in range(1, len(u)):
        expanded = []
        for state in beam:
            previous = state["path"][-1]
            for delta in (-2, -1, 0, 1, 2):
                j = previous + delta
                if j < 3 or j >= len(v) - 3 or not strip["inside"][i, j]:
                    continue
                if i * strip["du"] <= 3 * width and abs(v[j]) > .65 * width:
                    continue
                on_surface = bool(strip["surface"][i, j])
                if not on_surface and (state["entered_surface"] or u[i] > 8 * width or
                                       abs(v[j]) > .65 * width):
                    continue
                evidence = float(strip["score"][i, j])
                observed = bool(on_surface and strip["available"][i, j] and
                                evidence >= p.minimum_strip_pair_score)
                # 入卷前最多8个光学宽度只借直线方向寻找 SAM；这段始终不算观测。
                outside_search_prefix = (not state["entered_surface"] and
                                         not on_surface and u[i] <= 8 * width)
                gap = 0. if observed or outside_search_prefix else state["gap_px"] + strip["du"]
                if gap > max_gap:
                    continue
                curvature_step = abs(delta - state["slope"])
                cost = (state["cost"] + (1. - evidence if on_surface else .7) +
                        .13 * curvature_step + .025 * abs(delta))
                expanded.append({"cost": cost, "path": state["path"] + (j,),
                                 "slope": delta, "gap_px": gap,
                                 "observed_count": state["observed_count"] + int(observed),
                                 "entered_surface": state["entered_surface"] or on_surface})
        if not expanded:
            rejected["beam_exhausted_before_horizon"] += 1
            break
        # 空间格和方向格轮转，防止同一高反光旧圈占满全部 beam。
        cells = defaultdict(list)
        bin_width = max(2, int(round(.75 * width / strip["dv"])))
        for state in sorted(expanded, key=lambda item: item["cost"]):
            key = (state["path"][-1] // bin_width, state["slope"])
            if len(cells[key]) < 2:
                cells[key].append(state)
        beam = []
        for rank in (0, 1):
            for key in sorted(cells, key=lambda item: cells[item][0]["cost"]):
                if rank < len(cells[key]):
                    beam.append(cells[key][rank])
                if len(beam) >= p.beam_size:
                    break
            if len(beam) >= p.beam_size:
                break
        if i % snapshot_stride == 0 and i * strip["du"] >= p.minimum_candidate_extent_widths * width:
            terminal.extend({**state, "terminal_reason": "finite_extent_snapshot"}
                            for state in beam[:min(12, len(beam))])
    terminal.extend({**state, "terminal_reason": "final_supported_extent"} for state in beam)
    retained = []
    for state in terminal:
        extent = (len(state["path"]) - 1) * strip["du"]
        if extent < p.minimum_candidate_extent_widths * width:
            rejected["trace_too_short"] += 1
            continue
        if state["observed_count"] * strip["du"] < p.minimum_observed_extension_widths * width:
            rejected["original_pixel_pair_extension_too_short"] += 1
            continue
        retained.append(state)
    retained.sort(key=lambda state: (-state["observed_count"] * strip["du"],
                                     state["cost"] / len(state["path"])))
    return retained, {"considered_terminal_states": len(terminal),
                      "retained_terminal_states": len(retained),
                      "rejection_counts": dict(rejected), "beam_size": p.beam_size,
                      "trace_horizon_px": float(u[-1])}


def _profile_evidence(image: np.ndarray, path_xy: np.ndarray,
                      width: float, surface: np.ndarray) -> dict:
    """V28 双边剖面的独立原图复核；缺边不补成测量。"""
    count = len(path_xy)
    tangents = np.gradient(path_xy, axis=0)
    if count >= 7:
        tangents = cv2.GaussianBlur(tangents.astype(np.float32), (1, 7), 0).astype(float)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-8)
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    offsets = np.linspace(-1.15, 1.15, 47)
    along = np.linspace(-.30, .30, 5)
    sample_xy = (path_xy[:, None, None, :] +
                 normals[:, None, None, :] * width * offsets[None, :, None, None] +
                 tangents[:, None, None, :] * width * along[None, None, :, None])
    remap = sample_xy.reshape(count, -1, 2)
    raw = cv2.remap(image, remap[..., 0].astype(np.float32),
                    remap[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT).reshape(count, len(offsets), len(along), 3)
    gray = np.median(np.min(raw, axis=3), axis=2).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (5, 1), .7)
    grad = np.gradient(smooth, axis=1) / (width * (offsets[1] - offsets[0]))
    li = np.flatnonzero((offsets >= -.72) & (offsets <= -.28))
    ri = np.flatnonzero((offsets >= .28) & (offsets <= .72))
    lidx = li[np.argmax(np.abs(grad[:, li]), axis=1)]
    ridx = ri[np.argmax(np.abs(grad[:, ri]), axis=1)]
    rows = np.arange(count)
    gl, gr = grad[rows, lidx], grad[rows, ridx]
    strength = np.minimum(abs(gl), abs(gr))
    balance = strength / np.maximum(np.maximum(abs(gl), abs(gr)), 1e-6)
    contrast = np.percentile(smooth, 90, axis=1) - np.percentile(smooth, 10, axis=1)
    valid_source = ((sample_xy[..., 0] >= 1) &
                    (sample_xy[..., 0] < image.shape[1] - 2) &
                    (sample_xy[..., 1] >= 1) &
                    (sample_xy[..., 1] < image.shape[0] - 2)).all(axis=(1, 2))
    saturated = (np.max(raw, axis=3) >= 254).mean(axis=(1, 2))
    dark = np.percentile(smooth, 90, axis=1) < 12
    available = valid_source & (saturated <= .20) & (~dark) & (contrast >= 8.)
    measured_width = (offsets[ridx] - offsets[lidx]) * width
    opposite = gl * gr < 0
    supported = (available & opposite & (strength >= 1.2) & (balance >= .2) &
                 (measured_width >= .65 * width) & (measured_width <= 1.35 * width))
    px = np.rint(path_xy[:, 0]).astype(int)
    py = np.rint(path_xy[:, 1]).astype(int)
    center_inside = ((px >= 0) & (px < image.shape[1]) &
                     (py >= 0) & (py < image.shape[0]))
    on_surface = np.zeros(count, bool)
    on_surface[center_inside] = surface[py[center_inside], px[center_inside]]
    source_supported = supported.copy()
    supported &= on_surface
    left = path_xy + normals * (offsets[lidx] * width)[:, None]
    right = path_xy + normals * (offsets[ridx] * width)[:, None]
    bilateral = np.where(available,
                         np.clip(strength / 3., 0., 1.) * opposite *
                         np.clip(balance / .4, 0., 1.), 0.)
    return {"supported": supported, "source_supported": source_supported,
            "available": available, "on_surface": on_surface,
            "left": left, "right": right, "measured_width": measured_width,
            "bilateral": bilateral, "contrast": contrast,
            "saturated_fraction": saturated,
            "polarity": np.where(gl > 0, "bright_body", "dark_body_or_gap")}


def _band_distance(path: np.ndarray, band: dict, width: float,
                   observed: np.ndarray) -> float:
    indices = np.flatnonzero(observed)
    if len(indices) < 3:
        return math.inf
    last = indices[-1]
    selected = indices[indices >= last - max(5, int(round(4 * width / max(1.,
                                  np.median(np.linalg.norm(np.diff(path, axis=0), axis=1))))))]
    if len(selected) < 3:
        selected = indices[-3:]
    x = path[selected, 0]
    mid = (_predict(band["upper_model"], x) + _predict(band["lower_model"], x)) / 2.
    return float(np.median(np.abs(path[selected, 1] - mid)))


def _transition(path: np.ndarray, v: np.ndarray, observed: np.ndarray,
                width: float, du: float, band_distance: float, pitch: float,
                p: ConnectionParameters) -> tuple[list | None, list | None, dict]:
    if len(path) < 8:
        return None, None, {"reason": "path_too_short_for_turn"}
    smooth_v = cv2.GaussianBlur(v.reshape(1, -1).astype(np.float32), (7, 1), 1.3)[0]
    slope = np.gradient(smooth_v, du)
    departed = ((np.abs(smooth_v) >= p.turn_lateral_deviation_widths * width) |
                (np.abs(slope) >= p.turn_slope_threshold))
    start = None
    for i in range(2, len(path) - 5):
        if np.count_nonzero(departed[i:i + 5]) >= 4 and np.count_nonzero(observed[i:i + 6]) >= 3:
            start = i
            break
    if start is None:
        return None, None, {"reason": "no_original_pair_supported_direction_change"}
    observed_after = np.flatnonzero(observed & (np.arange(len(path)) > start))
    if not len(observed_after):
        return None, None, {"reason": "turn_has_no_supported_tail"}
    end = int(observed_after[-1])
    if (end - start) * du > p.maximum_turn_interval_widths * width:
        end = min(end, start + int(round(p.maximum_turn_interval_widths * width / du)))
    interval = [path[start].round(2).tolist(), path[end].round(2).tolist()]
    portion = observed[start:end + 1]
    pair_fraction = float(np.mean(portion)) if len(portion) else 0.
    joined = (math.isfinite(band_distance) and
              band_distance <= p.old_band_distance_pitches * pitch)
    center = (path[(start + end) // 2].round(2).tolist()
              if joined and pair_fraction >= p.minimum_original_pair_fraction_for_turn and
              (end - start) * du >= width else None)
    return interval, center, {"reason": "supported_turn_interval" if center is not None else
                                     "bend_visible_but_join_ambiguous",
                                     "turn_start_index": start, "turn_end_index": end,
                                     "pair_fraction_in_turn": pair_fraction,
                                     "old_band_alignment_at_tail": joined}


def _masks(shape: tuple[int, int], path: np.ndarray, states: list[str],
           surface: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observed = np.zeros(shape, np.uint8)
    inferred = np.zeros(shape, np.uint8)
    for i in range(1, len(path)):
        if states[i] == "unavailable" or states[i - 1] == "unavailable":
            continue
        xy0 = tuple(np.rint(path[i - 1]).astype(int))
        xy1 = tuple(np.rint(path[i]).astype(int))
        if np.linalg.norm(path[i] - path[i - 1]) > 12:
            continue
        target = observed if states[i] == states[i - 1] == "observed" else inferred
        cv2.line(target, xy0, xy1, 255, 1, cv2.LINE_8)
    observed[~surface] = 0
    inferred[~surface] = 0
    inferred[observed > 0] = 0
    return observed, inferred


def _candidate(state: dict, strip: dict, image: np.ndarray, surface: np.ndarray,
               bands: list[dict], pitch: float, width: float,
               p: ConnectionParameters, candidate_id: str) -> dict:
    indices = np.asarray(state["path"], int)
    path = strip["xy"][np.arange(len(indices)), indices].copy()
    v = strip["v"][indices].copy()
    if len(path) >= 7:
        # 平滑是候选几何，下面仍在未修改原图按平滑路径重测双边。
        smoothed_v = cv2.GaussianBlur(v.reshape(1, -1).astype(np.float32), (7, 1), 1.2)[0]
        shift = (smoothed_v - v)[:, None] * strip["normal"]
        path += shift
        v = smoothed_v.astype(float)
    evidence = _profile_evidence(image, path, width, surface)
    observed = evidence["supported"]
    unavailable = ~evidence["available"]
    states = ["observed" if good else "unavailable" if missing else "inferred"
              for good, missing in zip(observed, unavailable)]
    ordered_bands = sorted(((float(_band_distance(path, band, width, observed)), band)
                            for band in bands), key=lambda item: item[0])
    band_distance, band = ordered_bands[0] if ordered_bands else (math.inf, None)
    aligned = band is not None and band_distance <= p.old_band_distance_pitches * pitch
    kind = "old_row_band" if aligned else "possible_new_crossing"
    interval, center, turn_diag = _transition(path, v, observed, width, strip["du"],
                                               band_distance, pitch, p)
    visible = observed & evidence["on_surface"]
    observed_indices = np.flatnonzero(visible)
    post_start = min(len(path) - 1, max(1, int(round(3 * width / strip["du"]))))
    post_observed_count = int(np.count_nonzero(visible[post_start:]))
    rejection = []
    warnings = []
    if post_observed_count * strip["du"] < p.minimum_observed_extension_widths * width:
        rejection.append("final_original_profile_support_too_short")
    if interval is None:
        rejection.append("turn_not_image_supported")
    elif turn_diag.get("pair_fraction_in_turn", 0.) < p.minimum_original_pair_fraction_for_turn:
        rejection.append("turn_pair_support_insufficient")
    if band is None:
        warnings.append("c3_adjacent_old_band_unavailable")
    if kind == "possible_new_crossing" and center is None:
        warnings.append("possible_crossing_has_no_physical_depth_evidence")
    if not np.any(visible):
        rejection.append("no_inside_surface_observed_pair")
    observed_mask, inferred_mask = _masks(surface.shape, path, states, surface)
    # G0 仅几何；G1 加同族形态；G2 才使用候选分歧区的原图双边。
    first_tangent = path[min(3, len(path) - 1)] - path[0]
    first_tangent /= max(float(np.linalg.norm(first_tangent)), 1e-8)
    incoming = strip["direction"]
    direction_cost = min(1., math.acos(float(np.clip(first_tangent @ incoming, -1, 1))) / .7)
    distance_cost = min(1., band_distance / max(pitch, 1.)) if math.isfinite(band_distance) else .8
    g0_cost = .55 * distance_cost + .45 * direction_cost
    curvature = np.gradient(np.gradient(v, strip["du"]), strip["du"])
    curvature_cost = min(1., float(np.quantile(np.abs(curvature), .9)) * width * 8.)
    inferred_fraction = float(np.mean(np.asarray(states) == "inferred"))
    measured = evidence["measured_width"][visible]
    width_cost = (min(1., float(np.median(np.abs(measured / width - 1.))))
                  if len(measured) else 1.)
    arrangement_cost = distance_cost if kind == "old_row_band" else .5
    g1_cost = (.40 * g0_cost + .20 * curvature_cost + .18 * inferred_fraction +
               .12 * width_cost + .10 * arrangement_cost)
    # 后段/弯段优先计分，不用共同的长自由直段冲淡错误分支。
    disjoint_start = (turn_diag.get("turn_start_index") if interval is not None else
                      max(post_start, int(round(.45 * len(path)))))
    end = min(len(path), disjoint_start + max(12, int(round(20 * width / strip["du"]))))
    available_slice = evidence["available"][disjoint_start:end] & evidence["on_surface"][disjoint_start:end]
    supported_slice = evidence["supported"][disjoint_start:end]
    eligible = int(np.count_nonzero(available_slice))
    bilateral_fraction = (float(np.count_nonzero(supported_slice & available_slice) / eligible)
                          if eligible else None)
    contrast = evidence["contrast"][disjoint_start:end]
    contrast_score = (float(np.mean(np.clip(contrast[available_slice] / 35., 0., 1.)))
                      if eligible else None)
    image_cost = (.65 * (1. - bilateral_fraction) + .35 * (1. - contrast_score)
                  if bilateral_fraction is not None else .5)
    g2_cost = .50 * g1_cost + .50 * image_cost
    left = [xy.round(2).tolist() if ok else None
            for xy, ok in zip(evidence["left"], observed)]
    right = [xy.round(2).tolist() if ok else None
             for xy, ok in zip(evidence["right"], observed)]
    return {"candidate_id": candidate_id, "kind": kind,
            "target_c3_group_ids": band["group_ids"] if band else [],
            "target_band_id": band["band_id"] if band else None,
            "competing_band_distances_px": [
                {"band_id": item["band_id"],
                 "distance_px": round(distance, 3) if math.isfinite(distance) else None}
                for distance, item in ordered_bands[:4]],
            "path_xy": path.round(2).tolist(), "point_states": states,
            "boundary_left_xy": left, "boundary_right_xy": right,
            "transition_interval_xy": interval, "transition_center_xy": center,
            "transition_diagnostics": turn_diag,
            "score_g0": {"cost": float(g0_cost), "distance": float(distance_cost),
                         "entry_direction": float(direction_cost)},
            "score_g1": {"cost": float(g1_cost), "g0": float(g0_cost),
                         "curvature": float(curvature_cost),
                         "inferred_fraction": inferred_fraction, "width": float(width_cost),
                         "old_band_arrangement": float(arrangement_cost)},
            "score_g2": {"cost": float(g2_cost), "g1": float(g1_cost),
                         "distinct_original_profile_cost": float(image_cost),
                         "distinct_available_samples": eligible,
                         "distinct_bilateral_fraction": bilateral_fraction,
                         "distinct_contrast_score": contrast_score,
                         "missing_evidence_is_neutral": eligible == 0,
                         "window_kind": "provisional_turn_or_tail_window"},
            "reject_reasons": rejection, "warnings": warnings,
            "observed_mask": observed_mask, "inferred_mask": inferred_mask,
            "_pool_profile": {"available": evidence["available"] & evidence["on_surface"],
                              "supported": evidence["supported"],
                              "contrast": evidence["contrast"]},
            "observed_samples_in_surface": int(np.count_nonzero(visible)),
            "unsupported_or_unavailable_samples": int(len(path) - np.count_nonzero(visible)),
            "path_extent_px": float((len(path) - 1) * strip["du"]),
            "tail_old_band_distance_px": float(band_distance) if math.isfinite(band_distance) else None,
            "source": "original_Bgr_finite_double_edge_trace_and_original_profile_recheck",
            "physical_contact_confirmed": False, "material_identity_confirmed": False,
            "depth_order": "UNKNOWN", "terminal_reason": state["terminal_reason"]}


def _diverse(states: list[dict], strip: dict, bands: list[dict],
             pitch: float, width: float, p: ConnectionParameters) -> list[dict]:
    selected = []
    used_band = Counter()
    used_new = 0
    for state in states:
        indices = np.asarray(state["path"], int)
        path = strip["xy"][np.arange(len(indices)), indices]
        observed_strip = np.asarray([
            bool(strip["surface"][i, j] and strip["available"][i, j] and
                 strip["score"][i, j] >= p.minimum_strip_pair_score)
            for i, j in enumerate(indices)])
        distances = sorted(((float(_band_distance(path, band, width, observed_strip)), band)
                            for band in bands), key=lambda item: item[0])
        best_distance, best_band = distances[0] if distances else (math.inf, None)
        key = (best_band["band_id"] if best_band is not None and
               best_distance <= p.old_band_distance_pitches * pitch else "NEW_OR_UNRESOLVED")
        if key == "NEW_OR_UNRESOLVED" and used_new >= 2:
            continue
        if key != "NEW_OR_UNRESOLVED" and used_band[key] >= 1:
            continue
        duplicate = False
        for prior in selected:
            a = np.asarray(prior["path"], int)
            common = min(len(a), len(indices))
            if common < 6:
                continue
            tail = max(3, common - int(round(8 * width / strip["du"])))
            lateral = np.median(abs(strip["v"][a[tail:common]] -
                                    strip["v"][indices[tail:common]]))
            if (lateral < p.minimum_candidate_separation_widths * width and
                    abs(len(a) - len(indices)) * strip["du"] < 6 * width):
                duplicate = True
                break
        if duplicate:
            continue
        selected.append({"state": state, "path": indices, "band_key": key})
        if key == "NEW_OR_UNRESOLVED":
            used_new += 1
        else:
            used_band[key] += 1
        if len(selected) >= p.max_candidates:
            break
    return selected


def _score_actual_divergence(candidates: list[dict], width: float,
                             du: float, p: ConnectionParameters) -> None:
    """G2只评价候选真正分离的原图列；共同正确直段不加分。"""
    paths = [np.asarray(candidate["path_xy"], float) for candidate in candidates]
    for i, candidate in enumerate(candidates):
        path = paths[i]
        separated = np.zeros(len(path), bool)
        for j, other in enumerate(paths):
            if i == j:
                continue
            common = min(len(path), len(other))
            if common:
                separation = np.linalg.norm(path[:common] - other[:common], axis=1)
                separated[:common] |= separation >= p.minimum_candidate_separation_widths * width
        separated[:min(len(path), int(round(4 * width / du)))] = False
        distinct_indices = np.flatnonzero(separated)
        if len(distinct_indices):
            first = int(distinct_indices[0])
            end = min(len(path), first + max(12, int(round(20 * width / du))))
            window = separated & (np.arange(len(path)) >= first) & (np.arange(len(path)) < end)
        else:
            window = separated
        evidence = candidate.pop("_pool_profile")
        available = window & evidence["available"]
        available_count = int(np.count_nonzero(available))
        bilateral = (float(np.count_nonzero(evidence["supported"] & available) / available_count)
                     if available_count else None)
        contrast = (float(np.mean(np.clip(evidence["contrast"][available] / 35., 0., 1.)))
                    if available_count else None)
        image_cost = (.65 * (1. - bilateral) + .35 * (1. - contrast)
                      if bilateral is not None else .5)
        g1 = candidate["score_g1"]["cost"]
        candidate["score_g2"] = {
            "cost": float(.50 * g1 + .50 * image_cost), "g1": float(g1),
            "distinct_original_profile_cost": float(image_cost),
            "distinct_available_samples": available_count,
            "distinct_bilateral_fraction": bilateral,
            "distinct_contrast_score": contrast,
            "distinct_start_index": int(distinct_indices[0]) if len(distinct_indices) else None,
            "distinct_evaluated_columns": int(np.count_nonzero(window)),
            "missing_evidence_is_neutral": available_count == 0,
            "window_kind": "actual_candidate_separation_over_half_optical_width"}


def _global_conic(model: dict) -> tuple[float, float, float]:
    """C3 归一化 y=k+b*u+c*u² 换算为原图 y=a*x²+b*x+c。"""
    scale, center = model["u_scale_x_px"], model["u_center_x_px"]
    a = model["normalized_c"] / (scale * scale)
    b = model["normalized_b"] / scale - 2 * a * center
    c = (model["normalized_k"] - model["normalized_b"] * center / scale +
         a * center * center)
    return float(a), float(b), float(c)


def _tangent_hypotheses(bands: list[dict], surface: np.ndarray,
                        entry: np.ndarray, direction: np.ndarray,
                        width: float, pitch: float,
                        p: ConnectionParameters) -> tuple[list[dict], dict]:
    """候选是连续相切的单圆锥尾段；C3 仅给曲率和竞争旧带。"""
    h, w = surface.shape
    reasons = Counter()
    if abs(direction[0]) < .20:
        return [], {"rejection_counts": {"incoming_line_near_vertical_in_x": 1}}
    horizon = min(p.search_forward_widths * width, p.search_forward_max_px)
    slope = float(direction[1] / direction[0])
    intercept = float(entry[1] - slope * entry[0])
    prepared = []
    for band in bands:
        coefs = np.mean(np.asarray([_global_conic(band["upper_model"]),
                                    _global_conic(band["lower_model"])]), axis=0)
        a, b, c = (float(value) for value in coefs)
        if not 1e-6 <= a <= .002:
            reasons["c3_curvature_unidentifiable_or_implausible"] += 1
            continue
        prepared.append((band, a, b, c))
    if not prepared:
        return [], {"rejection_counts": dict(reasons), "horizon_px": horizon}

    def inside_tangent(t: float) -> bool:
        xy = entry + t * direction
        x, y = np.rint(xy).astype(int)
        return bool(0 <= x < w and 0 <= y < h and surface[y, x])

    proposals = []
    for band, a, b, c in prepared:
        xt = (slope - b) / (2 * a)
        t = (xt - entry[0]) / direction[0]
        if not 3 * width <= t <= horizon - 8 * width or not inside_tangent(t):
            reasons["c3_slope_tangent_not_in_visible_forward_surface"] += 1
            continue
        line_at_tangent = slope * xt + intercept
        band_at_tangent = a * xt * xt + b * xt + c
        offset = line_at_tangent - band_at_tangent
        if abs(offset) > 1.8 * pitch:
            reasons["c3_old_band_requires_large_shift"] += 1
            continue
        proposals.append({"kind": "old_row_band" if abs(offset) <= .45 * pitch
                          else "possible_new_crossing",
                          "tangent_t_px": float(t), "tangent_x_px": float(xt),
                          "a_px_per_px2": float(a), "line_slope_dy_dx": slope,
                          "line_intercept_px": intercept,
                          "shift_from_c3_band_px": float(offset),
                          "target_band_id": band["band_id"],
                          "target_c3_group_ids": band["group_ids"],
                          "curvature_source": "adjacent_C3_accepted_conic_midband",
                          "generation_cost": abs(offset) / pitch + .08 * t / horizon})

    # 局部旧圈曲率范围还允许有限的跨旧圈候选。它们的相切位置是离散
    # 几何假设，绝不因此记作观测；原图双边决定哪些段实际可见。
    all_curvature = sorted(a for _, a, _, _ in prepared)
    nearby = sorted(prepared,
                    key=lambda item: abs((item[1] * entry[0]**2 +
                                          item[2] * entry[0] + item[3]) - entry[1]))[:9]
    local_curvature = sorted(item[1] for item in nearby) or all_curvature
    for fraction, quantile in ((.48, .25), (.68, .50), (.82, .75)):
        t = fraction * horizon
        if not 3 * width <= t <= horizon - 8 * width or not inside_tangent(t):
            reasons["new_tangent_not_in_surface"] += 1
            continue
        a = float(np.quantile(local_curvature, quantile))
        xt = float(entry[0] + t * direction[0])
        probe_t = min(horizon, t + 12 * width)
        probe_x = entry[0] + probe_t * direction[0]
        probe_y = slope * probe_x + intercept + a * (probe_x - xt)**2
        closest = min(prepared,
                      key=lambda item: abs(probe_y -
                                           (item[1] * probe_x**2 + item[2] * probe_x + item[3])))
        matched_band = closest[0]
        gap = float(probe_y - (closest[1] * probe_x**2 + closest[2] * probe_x + closest[3]))
        proposals.append({"kind": "possible_new_crossing",
                          "tangent_t_px": float(t), "tangent_x_px": xt,
                          "a_px_per_px2": a, "line_slope_dy_dx": slope,
                          "line_intercept_px": intercept,
                          "shift_from_c3_band_px": gap,
                          "target_band_id": matched_band["band_id"],
                          "target_c3_group_ids": matched_band["group_ids"],
                          "curvature_source": "local_accepted_C3_conic_quantile",
                          "generation_cost": .7 + .08 * fraction + min(1., abs(gap) / pitch)})

    # 同一曲线可能与多条平移旧带相切；按路径而非带名去重。
    proposals.sort(key=lambda item: (item["generation_cost"], item["tangent_t_px"]))
    unique = []
    for item in proposals:
        def modeled_y(t: float) -> float:
            x = entry[0] + direction[0] * t
            extra = (item["a_px_per_px2"] * (x - item["tangent_x_px"])**2
                     if t > item["tangent_t_px"] else 0.)
            return item["line_slope_dy_dx"] * x + item["line_intercept_px"] + extra
        duplicate = False
        for previous in unique:
            distances = []
            for fraction in (.68, .82, .96):
                t = fraction * horizon
                x = entry[0] + direction[0] * t
                extra = (previous["a_px_per_px2"] *
                         (x - previous["tangent_x_px"])**2
                         if t > previous["tangent_t_px"] else 0.)
                py = previous["line_slope_dy_dx"] * x + previous["line_intercept_px"] + extra
                distances.append(abs(modeled_y(t) - py))
            if max(distances) < .45 * width:
                duplicate = True
                break
        if duplicate:
            reasons["same_geometric_curve_from_different_old_bands"] += 1
        else:
            unique.append(item)
        if len(unique) >= p.max_candidates:
            break
    return unique, {"rejection_counts": dict(reasons),
                    "c3_band_count": len(prepared),
                    "geometric_proposals_before_dedup": len(proposals),
                    "unique_geometric_hypotheses": len(unique),
                    "horizon_px": float(horizon),
                    "formula": "before tangent: free y=m*x+q; after tangent: y=m*x+q+a*(x-x_t)^2, a from accepted C3 local conics"}


def _conic_candidate(hypothesis: dict, image: np.ndarray, surface: np.ndarray,
                     entry: np.ndarray, direction: np.ndarray,
                     width: float, pitch: float, p: ConnectionParameters,
                     candidate_id: str) -> dict:
    horizon = min(p.search_forward_widths * width, p.search_forward_max_px)
    t0 = hypothesis["tangent_t_px"]
    du = float(np.clip(p.step_u_widths * width, 3., 9.))
    tail_limit = min(horizon, t0 + 35 * width)
    travel = np.arange(0., tail_limit + du * .5, du)
    x = entry[0] + direction[0] * travel
    baseline_y = hypothesis["line_slope_dy_dx"] * x + hypothesis["line_intercept_px"]
    bend = hypothesis["a_px_per_px2"] * (x - hypothesis["tangent_x_px"])**2
    bend[travel <= t0] = 0.
    path = np.column_stack((x, baseline_y + bend))
    # 原图或卷面以外不能借图像边框/背景继续路径；自由前缀允许在卷面外。
    length = len(path)
    for i, (px, py) in enumerate(path):
        ix, iy = int(round(px)), int(round(py))
        if not (2 <= ix < image.shape[1] - 2 and 2 <= iy < image.shape[0] - 2):
            length = i
            break
        if travel[i] >= t0 and not surface[iy, ix]:
            length = i
            break
    path, travel, bend = path[:length], travel[:length], bend[:length]
    if len(path) < 4:
        raise ValueError("geometric tangent left valid image/surface immediately")
    evidence = _profile_evidence(image, path, width, surface)
    source_pairs = evidence["source_supported"]
    surface_pairs = evidence["supported"]
    states = ["observed" if ok else "unavailable" if not available else "inferred"
              for ok, available in zip(source_pairs, evidence["available"])]
    before = (travel >= max(0., t0 - 3 * width)) & (travel <= t0)
    after = (travel > t0 + .5 * width) & (travel <= t0 + 13 * width)
    before_pairs = int(np.count_nonzero(source_pairs & before))
    after_pairs = int(np.count_nonzero(surface_pairs & after))
    after_available = int(np.count_nonzero(evidence["available"] & evidence["on_surface"] & after))
    after_fraction = after_pairs / after_available if after_available else 0.
    bend_visible = (before_pairs >= 3 and after_pairs >= 4 and
                    after_fraction >= p.minimum_original_pair_fraction_for_turn and
                    np.max(bend) >= p.turn_lateral_deviation_widths * width)
    turn_index = int(np.argmin(abs(travel - t0)))
    end_indices = np.flatnonzero((travel > t0) &
                                 (bend >= p.turn_lateral_deviation_widths * width))
    end_index = int(end_indices[0]) if len(end_indices) else min(len(path) - 1,
                       turn_index + int(round(10 * width / du)))
    interval = ([path[turn_index].round(2).tolist(), path[end_index].round(2).tolist()]
                if bend_visible else None)
    center = path[turn_index].round(2).tolist() if bend_visible else None
    observed_mask, inferred_mask = _masks(surface.shape, path, states, surface)
    rejection = []
    if before_pairs < 3:
        rejection.append("original_pair_missing_before_tangent")
    if after_pairs < 4 or after_fraction < p.minimum_original_pair_fraction_for_turn:
        rejection.append("original_pair_missing_after_tangent")
    if np.max(bend) < p.turn_lateral_deviation_widths * width:
        rejection.append("curvature_not_visibly_separable_from_straight")
    warnings = (["two_dimensional_crossing_has_unknown_depth_order"]
                if hypothesis["kind"] == "possible_new_crossing" else [])
    measured = evidence["measured_width"][surface_pairs]
    width_cost = (min(1., float(np.median(abs(measured / width - 1.))))
                  if len(measured) else 1.)
    shift_cost = min(1., abs(hypothesis["shift_from_c3_band_px"]) / max(pitch, 1.))
    horizon_cost = min(1., t0 / max(horizon, 1.))
    g0_cost = .80 * shift_cost + .20 * horizon_cost
    inferred_fraction = float(np.mean(np.asarray(states) == "inferred"))
    local_curvature_cost = .0 if hypothesis["curvature_source"].startswith("adjacent") else .25
    g1_cost = (.55 * g0_cost + .20 * inferred_fraction +
               .15 * width_cost + .10 * local_curvature_cost)
    left = [point.round(2).tolist() if supported else None
            for point, supported in zip(evidence["left"], source_pairs)]
    right = [point.round(2).tolist() if supported else None
             for point, supported in zip(evidence["right"], source_pairs)]
    return {"candidate_id": candidate_id, "kind": hypothesis["kind"],
            "target_c3_group_ids": hypothesis["target_c3_group_ids"],
            "target_band_id": hypothesis["target_band_id"],
            "path_xy": path.round(2).tolist(), "point_states": states,
            "boundary_left_xy": left, "boundary_right_xy": right,
            "transition_interval_xy": interval, "transition_center_xy": center,
            "geometric_tangent_xy": path[turn_index].round(2).tolist(),
            "geometric_tangent_u_px": float(t0),
            "transition_diagnostics": {"before_original_pair_samples": before_pairs,
                                       "after_surface_pair_samples": after_pairs,
                                       "after_available_samples": after_available,
                                       "after_pair_fraction": after_fraction,
                                       "bend_max_deviation_from_line_px": float(np.max(bend)),
                                       "bend_visible_in_original_image": bool(bend_visible)},
            "conic_model": hypothesis,
            "score_g0": {"cost": float(g0_cost), "c3_band_shift": shift_cost,
                         "tangent_forward_distance": horizon_cost,
                         "free_tangent_direction_error": 0.},
            "score_g1": {"cost": float(g1_cost), "g0": float(g0_cost),
                         "inferred_fraction": inferred_fraction,
                         "width": float(width_cost),
                         "c3_curvature_source_cost": local_curvature_cost},
            "score_g2": {"cost": float(.5 * g1_cost + .25),
                         "window_kind": "pending_actual_candidate_separation"},
            "reject_reasons": rejection, "warnings": warnings,
            "observed_mask": observed_mask, "inferred_mask": inferred_mask,
            "_pool_profile": {"available": evidence["available"] & evidence["on_surface"],
                              "supported": surface_pairs,
                              "contrast": evidence["contrast"]},
            "observed_samples_in_surface": int(np.count_nonzero(surface_pairs)),
            "unsupported_or_unavailable_samples": int(len(path) - np.count_nonzero(source_pairs)),
            "path_extent_px": float(travel[-1]),
            "tail_old_band_distance_px": abs(hypothesis["shift_from_c3_band_px"]),
            "source": "free_line_tangent_to_C3_curvature_conic_then_original_Bgr_double_edge_recheck",
            "physical_contact_confirmed": False, "material_identity_confirmed": False,
            "depth_order": "UNKNOWN", "terminal_reason": "finite_conic_support_domain"}


def find_connections(image_bgr: np.ndarray, surface_mask: np.ndarray,
                     free_segment: dict, c3_fit: dict,
                     parameters: ConnectionParameters | None = None) -> dict:
    """返回最多六条可审计图像接续假设；不宣布物理接触。"""
    p = parameters or ConnectionParameters()
    checked = _input(image_bgr, surface_mask, free_segment)
    if isinstance(checked, dict):
        checked["diagnostics"]["parameters"] = asdict(p)
        return checked
    image, surface, width, entry, direction = checked
    x_reference = float(np.clip(entry[0] + direction[0] * min(25 * width, 500),
                                0, image.shape[1] - 1))
    bands, pitch, band_diag = _c3_bands(c3_fit, x_reference, surface.shape, width)
    strip = _strip(image, surface, entry, direction, width, p)
    states, beam_diag = _beam_paths(strip, width, p)
    hypotheses, tangent_diag = _tangent_hypotheses(bands, surface, entry,
                                                     direction, width, pitch, p)
    candidates = []
    geometry_rejected = []
    for hypothesis in hypotheses:
        try:
            candidate = _conic_candidate(hypothesis, image, surface, entry,
                                         direction, width, pitch, p,
                                         f"K{len(candidates) + 1:02d}")
        except ValueError as error:
            geometry_rejected.append({"target_band_id": hypothesis["target_band_id"],
                                      "reason": str(error)})
            continue
        candidates.append(candidate)
    _score_actual_divergence(candidates, width, strip["du"], p)
    rejected = geometry_rejected + [{"candidate_id": candidate["candidate_id"],
                 "reasons": candidate["reject_reasons"]}
                for candidate in candidates if candidate["reject_reasons"]]
    g0 = (min(candidates, key=lambda item: item["score_g0"]["cost"])["candidate_id"]
          if candidates else None)
    ranked = sorted(candidates, key=lambda item: item["score_g2"]["cost"])
    viable = [candidate for candidate in ranked if
              not candidate["reject_reasons"] and
              candidate["transition_interval_xy"] is not None and
              candidate["score_g2"]["distinct_available_samples"] >= 12]
    if not bands:
        status, selected = "UNKNOWN", None
        reason = "adjacent_c3_old_band_competitors_unavailable"
    elif not viable:
        status, selected = "UNKNOWN", None
        reason = "no_candidate_has_original_pair_supported_turn"
    elif len(viable) == 1:
        # 单一路径仍只是图像假设；它不能排除未生成的竞争旧圈。
        status, selected = "UNKNOWN", None
        reason = "only_one_viable_image_hypothesis_competition_unresolved"
    else:
        margin = viable[1]["score_g2"]["cost"] - viable[0]["score_g2"]["cost"]
        if margin >= p.minimum_score_margin:
            status, selected = "SINGLE_IMAGE_HYPOTHESIS", viable[0]["candidate_id"]
            reason = "distinct_original_profile_and_geometric_margin"
        else:
            status, selected = "MULTI_CANDIDATE", None
            reason = "plausible_competing_image_paths"
    return {"status": status, "selected_candidate_id": selected,
            "g0_baseline_candidate_id": g0, "candidates": candidates,
            "rejected": rejected,
            "diagnostics": {"status": status, "decision_reason": reason,
                            "free_observed_segment_status": "observed",
                            "free_entry_xy": entry.round(2).tolist(),
                            "free_direction_xy": direction.round(5).tolist(),
                            "free_optical_width_px": width,
                            "c3_bands": band_diag, "estimated_pitch_px": pitch,
                            "beam": {**beam_diag,
                                     "role": "diagnostic_only; beam polylines never enter candidate pool"},
                            "tangent_conic_generation": tangent_diag,
                            "generated_candidate_count": len(candidates),
                            "viable_candidate_count": len(viable),
                            "parameters": asdict(p),
                            "g2_ranked_candidate_ids": [c["candidate_id"] for c in ranked],
                            "physical_contact_confirmed": False,
                            "image_identity_confirmed": False,
                            "t1_confirmed": False,
                            "free_straight_projection_is_contact_evidence": False,
                            "c3_conic_is_original_image_evidence": False}}
