"""C4：SAM 全卷面上的暗沟观测、单组圆锥重拟合和有限几何推断。

亮区 C3 接受组是唯一种子；暗点必须由原图双侧亮纹、跨列连续性、
邻组唯一归属共同支持。每组亮/暗点在一条 c>=0 抛物线上联合拟合。
无点区域仅作有限距离几何推断，不赋予真实材料身份或换层含义。
模块纯离线，不导入设备代码，也不读取任何数据文件。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class FullSurfaceParameters:
    column_step_pitches: float = .30
    search_radius_pitches: float = .32
    flank_offset_pitches: float = .38
    min_raw_two_flank_contrast: float = 1.0
    min_enhanced_two_flank_contrast: float = 2.4
    min_normalized_response: float = .80
    max_saturation_fraction: float = .48
    min_dark_track_points: int = 6
    min_dark_track_span_pitches: float = 2.0
    min_dark_track_support_fraction: float = .46
    max_dark_track_mad_pitches: float = .12
    max_dark_seed_residual_pitches: float = .30
    min_neighbor_spacing_pitches: float = .55
    max_extrapolation_pitches: float = 30.0
    max_extrapolation_pixels: int = 750
    weak_seed_max_extrapolation_pitches: float = 2.0
    min_bright_span_for_long_extension_pitches: float = 12.0
    max_model_slope: float = .22
    near_observation_distance_pitches: float = .25


def _binary(mask: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8 HxW")
    if np.any((mask != 0) & (mask != 255)):
        raise ValueError(f"{name} must contain only 0/255")
    return mask > 0


def _validate(image_bgr: np.ndarray | None, surface_mask: np.ndarray,
              bright_mask: np.ndarray, exclusion_mask: np.ndarray,
              pitch_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    surface = _binary(surface_mask, "sam_surface_mask")
    bright = _binary(bright_mask, "bright_roi_mask")
    exclusion = _binary(exclusion_mask, "exclusion_mask")
    if surface.shape != bright.shape or surface.shape != exclusion.shape:
        raise ValueError("all masks must have the same image dimensions")
    if image_bgr is not None and (not isinstance(image_bgr, np.ndarray) or
                                  image_bgr.shape != (*surface.shape, 3) or
                                  image_bgr.dtype != np.uint8):
        raise ValueError("image_bgr must be full-resolution uint8 BGR")
    pitch = float(pitch_px)
    if not math.isfinite(pitch) or not 8 <= pitch <= 70:
        raise ValueError("pitch_px must be finite and within [8,70]")
    return surface, bright, exclusion, pitch


def _seeds(c3_fit: dict, shape: tuple[int, int]) -> tuple[list[dict], dict]:
    """仅使用 C3 已接受的分组、原黄点和其单模型；旧紫点不参与。"""
    if not isinstance(c3_fit, dict):
        raise ValueError("c3_fit must be a dictionary")
    curves = {item["group_id"]: item for item in c3_fit.get("curves", [])}
    seeds = []
    rejected = Counter()
    height, width = shape
    for group in c3_fit.get("groups", []):
        group_id = group.get("group_id")
        curve = curves.get(group_id)
        if not group.get("curve_id") or curve is None:
            rejected["c3_group_rejected_or_missing_curve"] += 1
            continue
        points = np.asarray(group.get("observed_points_xy", []), float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
            rejected["c3_seed_too_few_or_invalid_points"] += 1
            continue
        if ((points[:, 0] < 0).any() or (points[:, 0] >= width).any() or
                (points[:, 1] < 0).any() or (points[:, 1] >= height).any()):
            rejected["c3_seed_outside_image"] += 1
            continue
        model = curve["model"]
        required = ("normalized_k", "normalized_b", "normalized_c",
                    "u_center_x_px", "u_scale_x_px")
        if any(key not in model or not math.isfinite(float(model[key])) for key in required):
            rejected["c3_seed_missing_model"] += 1
            continue
        if model["normalized_c"] < 0 or model["u_scale_x_px"] <= 0:
            rejected["c3_seed_invalid_parabola"] += 1
            continue
        raw_weights = group.get("fit_weights", [])
        weights = (np.asarray(raw_weights, float) if len(raw_weights) == len(points)
                   else np.ones(len(points), float))
        if not np.isfinite(weights).all() or (weights <= 0).any():
            weights = np.ones(len(points), float)
        seeds.append({"group_id": group_id, "source_fragment_ids": group.get("source_fragment_ids", []),
                      "bright_points": points, "bright_weights": weights,
                      "model": {key: float(model[key]) for key in required},
                      "c3_fit_regularization": group.get("fit_regularization"),
                      "c3_curve_id": curve["curve_id"]})
    return seeds, {"seed_count": len(seeds), "rejected_seed_counts": dict(rejected)}


def _predict(model: dict, x: np.ndarray | float) -> np.ndarray | float:
    u = (x - model["u_center_x_px"]) / model["u_scale_x_px"]
    return model["normalized_k"] + model["normalized_b"] * u + model["normalized_c"] * u * u


def _slope(model: dict, x: float) -> float:
    u = (x - model["u_center_x_px"]) / model["u_scale_x_px"]
    return float((model["normalized_b"] + 2 * model["normalized_c"] * u) /
                 model["u_scale_x_px"])


def _image_signals(image_bgr: np.ndarray, surface: np.ndarray, pitch: float) -> tuple[dict, tuple]:
    ys, xs = np.nonzero(surface)
    if len(xs) == 0:
        return {}, (0, 0, 0, 0)
    margin = max(30, int(round(2 * pitch)))
    x0, x1 = max(0, int(xs.min()) - margin), min(surface.shape[1], int(xs.max()) + margin + 1)
    y0, y1 = max(0, int(ys.min()) - margin), min(surface.shape[0], int(ys.max()) + margin + 1)
    gray = cv2.cvtColor(image_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(12, 12)).apply(gray)
    sigma_x, sigma_y = max(3., .48 * pitch), max(.9, .08 * pitch)
    smooth = cv2.GaussianBlur(enhanced.astype(np.float32), (0, 0), sigmaX=sigma_x,
                              sigmaY=sigma_y)
    raw_smooth = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigmaX=sigma_x,
                                  sigmaY=sigma_y)
    broad = cv2.GaussianBlur(smooth, (0, 0), sigmaX=max(2., .25 * pitch),
                             sigmaY=max(2., .45 * pitch))
    local_scale = cv2.blur(np.abs(smooth - broad),
                           (1, max(7, int(round(1.7 * pitch))) | 1))
    saturation = cv2.blur((gray >= 250).astype(np.float32),
                          (max(7, int(round(.85 * pitch))) | 1,
                           max(7, int(round(.65 * pitch))) | 1))
    return {"smooth": smooth, "raw": raw_smooth, "scale": local_scale,
            "saturation": saturation}, (x0, y0, x1, y1)


def _candidate_at(x: int, predicted_y: float, signals: dict,
                  bounds: tuple[int, int, int, int], surface: np.ndarray,
                  bright: np.ndarray, exclusion: np.ndarray, pitch: float,
                  parameters: FullSurfaceParameters) -> dict | None:
    x0, y0, x1, y1 = bounds
    if not x0 <= x < x1:
        return None
    d = max(3, int(round(parameters.flank_offset_pitches * pitch)))
    radius = max(3, int(round(parameters.search_radius_pitches * pitch)))
    lo = max(y0 + d + 3, int(round(predicted_y)) - radius)
    hi = min(y1 - d - 4, int(round(predicted_y)) + radius)
    if lo > hi:
        return None
    c = x - x0
    enhanced = signals["smooth"]
    raw = signals["raw"]
    scale = signals["scale"]
    saturation = signals["saturation"]
    best = None
    for y in range(lo, hi + 1):
        if not (surface[y, x] and not bright[y, x] and not exclusion[y, x]):
            continue
        if not (surface[y - d, x] and surface[y + d, x] and
                not exclusion[y - d, x] and not exclusion[y + d, x]):
            continue
        r = y - y0
        valley = float(enhanced[r, c])
        if not (valley <= enhanced[r - 2, c] and valley < enhanced[r + 2, c]):
            continue
        upper, lower = float(enhanced[r - d, c] - valley), float(enhanced[r + d, c] - valley)
        weak = min(upper, lower)
        strong = max(upper, lower)
        raw_weak = min(float(raw[r - d, c] - raw[r, c]),
                       float(raw[r + d, c] - raw[r, c]))
        normalized = weak / max(2., float(scale[r, c]))
        if (weak < parameters.min_enhanced_two_flank_contrast or
                strong < 4.0 or raw_weak < parameters.min_raw_two_flank_contrast or
                normalized < parameters.min_normalized_response):
            continue
        if (saturation[r, c] > parameters.max_saturation_fraction and
                raw[r, c] >= 235):
            continue
        error = abs(y - predicted_y)
        score = normalized + .04 * weak - .35 * error / pitch
        candidate = {"x": int(x), "y": int(y), "score": round(float(score), 4),
                     "weak_enhanced": round(weak, 3), "weak_raw": round(raw_weak, 3),
                     "normalized_response": round(normalized, 4),
                     "seed_distance_px": round(float(error), 3),
                     "saturation_fraction": round(float(saturation[r, c]), 3)}
        if best is None or score > best[0]:
            best = (score, candidate)
    return best[1] if best else None


def _tracks(points: list[dict], step: int, pitch: float,
            parameters: FullSurfaceParameters) -> tuple[list[dict], Counter]:
    if not points:
        return [], Counter()
    ordered = sorted(points, key=lambda p: p["x"])
    segments = []
    current = [ordered[0]]
    for point in ordered[1:]:
        previous = current[-1]
        if (point["x"] - previous["x"] > 2.5 * step or
                abs(point["y"] - previous["y"]) > max(3., .20 * pitch)):
            segments.append(current)
            current = [point]
        else:
            current.append(point)
    segments.append(current)
    kept = []
    reasons = Counter()
    for segment in segments:
        span = segment[-1]["x"] - segment[0]["x"]
        if len(segment) < parameters.min_dark_track_points or span < parameters.min_dark_track_span_pitches * pitch:
            reasons["dark_track_too_short"] += 1
            continue
        support = len(segment) / (1 + span / step)
        if support < parameters.min_dark_track_support_fraction:
            reasons["dark_track_sparse"] += 1
            continue
        xx = np.asarray([p["x"] for p in segment], float)
        yy = np.asarray([p["y"] for p in segment], float)
        degree = min(2, len(np.unique(xx)) - 1)
        fitted = np.polyval(np.polyfit((xx - np.mean(xx)) / max(1., np.ptp(xx)), yy, degree),
                            (xx - np.mean(xx)) / max(1., np.ptp(xx)))
        mad = float(np.median(np.abs(yy - fitted)))
        if mad > parameters.max_dark_track_mad_pitches * pitch:
            reasons["dark_track_rough"] += 1
            continue
        if np.median([p["seed_distance_px"] for p in segment]) > parameters.max_dark_seed_residual_pitches * pitch:
            reasons["dark_track_seed_disagreement"] += 1
            continue
        for point in segment:
            point["track_span_px"] = int(span)
            point["track_support_fraction"] = round(float(support), 3)
            point["track_mad_px"] = round(mad, 3)
        kept.extend(segment)
    return kept, reasons


def detect_dark_candidates(image_bgr: np.ndarray, sam_surface_mask: np.ndarray,
                           bright_roi_mask: np.ndarray, c3_fit: dict,
                           exclusion_mask: np.ndarray, pitch_px: float,
                           parameters: FullSurfaceParameters | None = None) -> dict:
    """仅识别 SAM∖亮ROI 的原图暗沟点；返回候选供纯几何入口复用。"""
    p = parameters or FullSurfaceParameters()
    surface, bright, exclusion, pitch = _validate(image_bgr, sam_surface_mask,
                                                  bright_roi_mask, exclusion_mask, pitch_px)
    seeds, seed_diag = _seeds(c3_fit, surface.shape)
    result = {"points": [], "diagnostics": {**seed_diag, "status": "completed"},
              "rejected_counts": {}}
    if not seeds or not np.any(surface):
        result["diagnostics"]["status"] = "no_seed_or_empty_surface"
        return result
    signals, bounds = _image_signals(image_bgr, surface, pitch)
    if not signals:
        result["diagnostics"]["status"] = "empty_surface"
        return result
    step = max(4, int(round(p.column_step_pitches * pitch)))
    xs = range(bounds[0], bounds[2], step)
    proposals: dict[str, list[dict]] = defaultdict(list)
    rejection = Counter()
    for x in xs:
        column = []
        for seed in seeds:
            predicted = float(_predict(seed["model"], x))
            if not math.isfinite(predicted) or not 0 <= predicted < surface.shape[0]:
                continue
            candidate = _candidate_at(x, predicted, signals, bounds, surface, bright,
                                      exclusion, pitch, p)
            if candidate is None:
                continue
            candidate["group_id"] = seed["group_id"]
            column.append(candidate)
        column.sort(key=lambda item: item["y"])
        suppressed = set()
        for i in range(len(column) - 1):
            a, b = column[i], column[i + 1]
            if b["y"] - a["y"] >= p.min_neighbor_spacing_pitches * pitch:
                continue
            if abs(a["score"] - b["score"]) <= .5:
                suppressed.update((i, i + 1))
            else:
                suppressed.add(i if a["score"] < b["score"] else i + 1)
        for i, candidate in enumerate(column):
            if i in suppressed:
                rejection["ambiguous_neighbor_assignment"] += 1
            else:
                proposals[candidate["group_id"]].append(candidate)
    accepted = []
    for group_id, points in proposals.items():
        retained, failed = _tracks(points, step, pitch, p)
        accepted.extend(retained)
        rejection.update(failed)
    result["points"] = accepted
    result["rejected_counts"] = dict(rejection)
    result["diagnostics"].update({"sampled_columns": len(xs), "column_step_px": step,
                                   "proposed_point_count": sum(map(len, proposals.values())),
                                   "trusted_dark_point_count": len(accepted),
                                   "surface_bounds_xyxy": list(bounds),
                                   "interpretation": "local image-supported visual valley points only; no physical rope identity"})
    return result


def _joint_fit(seed: dict, dark: list[dict], pitch: float) -> tuple[dict, np.ndarray, np.ndarray, dict]:
    """全部亮点与可信暗点一起进入同一受约束加权抛物线目标。"""
    baseline = seed["model"]
    bright = seed["bright_points"]
    dark_array = np.asarray([[p["x"], p["y"]] for p in dark], float).reshape(-1, 2)
    all_points = np.concatenate((bright, dark_array))
    center, scale = baseline["u_center_x_px"], baseline["u_scale_x_px"]
    u = (all_points[:, 0] - center) / scale
    design = np.column_stack((np.ones(len(u)), u, u * u))
    y = all_points[:, 1]
    bright_weight = np.maximum(.25, seed["bright_weights"])
    dark_weight = np.asarray([np.clip(.20 + .12 * point["normalized_response"], .20, .55)
                              for point in dark], float)
    base_weight = np.concatenate((bright_weight, dark_weight))
    prior = np.array([baseline["normalized_k"], baseline["normalized_b"],
                      baseline["normalized_c"]], float)
    lam_b = max(.05, .025 * len(bright))
    lam_c = max(.05, .045 * len(bright))
    factor = np.ones(len(y))

    def solve(weights: np.ndarray) -> tuple[np.ndarray, bool]:
        normal = design.T @ (weights[:, None] * design)
        rhs = design.T @ (weights * y)
        normal[1, 1] += lam_b
        normal[2, 2] += lam_c
        rhs[1] += lam_b * prior[1]
        rhs[2] += lam_c * prior[2]
        coef = np.linalg.solve(normal, rhs)
        active = bool(coef[2] < 0)
        if active:
            reduced = design[:, :2]
            normal2 = reduced.T @ (weights[:, None] * reduced)
            normal2[1, 1] += lam_b
            rhs2 = reduced.T @ (weights * y)
            rhs2[1] += lam_b * prior[1]
            kb = np.linalg.solve(normal2, rhs2)
            coef = np.array([kb[0], kb[1], 0.], float)
        return coef, active

    for _ in range(5):
        used_weight = base_weight * factor
        coef, _ = solve(used_weight)
        residual = y - design @ coef
        robust = max(.8, 1.4826 * float(np.median(np.abs(residual))))
        factor = np.clip(1.5 * robust / np.maximum(np.abs(residual), 1e-6), .08, 1.)
    used_weight = base_weight * factor
    coef, active = solve(used_weight)
    residual = y - design @ coef
    model = {"type": "vertical_axis_parabola" if coef[2] > 0 else "near_flat_line",
             "normalized_k": float(coef[0]), "normalized_b": float(coef[1]),
             "normalized_c": float(coef[2]), "u_center_x_px": float(center),
             "u_scale_x_px": float(scale), "a_px_per_px2": float(coef[2] / scale**2)}
    regularization = {"parameter_order": ["normalized_k", "normalized_b", "normalized_c"],
                      "prior_b": float(prior[1]), "prior_c": float(prior[2]),
                      "lambda_k": 0., "lambda_b": lam_b, "lambda_c": lam_c,
                      "c_nonnegative_active": active,
                      "objective": "sum positive_w*(all_bright_and_dark_y-k-b*u-c*u²)^2 + lambda_b*(b-prior_b)^2 + lambda_c*(c-prior_c)^2"}
    return model, used_weight, residual, regularization


def _sample_model(model: dict, xlo: int, xhi: int, surface: np.ndarray,
                  exclusion: np.ndarray, parameters: FullSurfaceParameters,
                  observed_points: np.ndarray, pitch: float) -> tuple[list[list[int]], list[bool], dict]:
    height, width = surface.shape
    points = []
    inferred = []
    observed_order = np.argsort(observed_points[:, 0], kind="stable")
    observed = observed_points[observed_order]
    observed_x = observed[:, 0]
    near_radius = max(3., parameters.near_observation_distance_pitches * pitch)
    clipped = Counter()
    for x in range(max(0, xlo), min(width - 1, xhi) + 1):
        if abs(_slope(model, x)) > parameters.max_model_slope:
            clipped["slope_limit_columns"] += 1
            continue
        yf = float(_predict(model, x))
        y = int(round(yf))
        if not 0 <= y < height:
            clipped["outside_image_columns"] += 1
            continue
        if not surface[y, x] or exclusion[y, x]:
            clipped["outside_surface_or_excluded_columns"] += 1
            continue
        # 只看 x 距离会把拟合到错误 y 行的紫线误称为“贴近黄点”。
        # 在 x±门槛的全部真实点中计算二维欧氏距离；无点即明确推断。
        left = int(np.searchsorted(observed_x, x - near_radius, side="left"))
        right = int(np.searchsorted(observed_x, x + near_radius, side="right"))
        nearby = observed[left:right]
        distance = (float(np.min(np.hypot(nearby[:, 0] - x, nearby[:, 1] - y)))
                    if len(nearby) else math.inf)
        points.append([x, y])
        inferred.append(bool(distance > near_radius))
    return points, inferred, dict(clipped)


def _domain(seed: dict, dark: list[dict], surface_x_bounds: tuple[int, int],
            pitch: float, parameters: FullSurfaceParameters) -> tuple[int, int, dict]:
    bright = seed["bright_points"]
    xs = bright[:, 0].tolist() + [p["x"] for p in dark]
    lo, hi = min(xs), max(xs)
    enough = (len(bright) >= 12 and
              np.ptp(bright[:, 0]) >= parameters.min_bright_span_for_long_extension_pitches * pitch)
    extension = (min(parameters.max_extrapolation_pixels,
                     parameters.max_extrapolation_pitches * pitch) if enough else
                 parameters.weak_seed_max_extrapolation_pitches * pitch)
    surface_left, surface_right = surface_x_bounds
    xlo = max(surface_left, int(math.floor(lo - extension)))
    xhi = min(surface_right, int(math.ceil(hi + extension)))
    return xlo, xhi, {"status": "bounded", "bright_seed_strong": bool(enough),
                      "max_extension_from_observed_x_px": float(extension),
                      "observed_x_span_px": [float(lo), float(hi)],
                      "drawable_x_range_px": [xlo, xhi],
                      "left_extension_truncated_by_budget": bool(xlo > surface_left),
                      "right_extension_truncated_by_budget": bool(xhi < surface_right)}


def _frozen_bright_fit(seed: dict) -> tuple[dict, np.ndarray, np.ndarray, dict]:
    """没有通过暗区联合验证时，逐系数保留已冻结 C3 曲线。"""
    bright = seed["bright_points"]
    residual = bright[:, 1] - _predict(seed["model"], bright[:, 0])
    return (dict(seed["model"]), seed["bright_weights"].copy(), residual,
            {"status": "frozen_c3_model_no_trusted_dark_point",
             "source_c3_regularization": seed["c3_fit_regularization"],
             "new_optimization_performed": False})


def _spacing_guard(curves: list[dict], pitch: float) -> list[dict]:
    """只把无直接观测支持的相交/挤靠绘制点转未知，保留模型与诊断。"""
    by_x = [{xy[0]: (xy[1], idx) for idx, xy in enumerate(curve["sampled_points_xy"])}
            for curve in curves]
    remove = [set() for _ in curves]
    remove_details = [defaultdict(set) for _ in curves]
    conflicts = []
    for i in range(len(curves)):
        for j in range(i + 1, len(curves)):
            common = by_x[i].keys() & by_x[j].keys()
            if not common:
                continue
            ordered_x = sorted(common)
            signed = [by_x[i][x][0] - by_x[j][x][0] for x in ordered_x]
            crossing = min(signed) <= 0 <= max(signed)
            close = {x for x in ordered_x if abs(by_x[i][x][0] - by_x[j][x][0]) < .50 * pitch}
            if not crossing and not close:
                continue
            reliable = []
            for x in ordered_x:
                ai, bi = by_x[i][x][1], by_x[j][x][1]
                dy = by_x[i][x][0] - by_x[j][x][0]
                if (not curves[i]["inferred_flags"][ai] and
                        not curves[j]["inferred_flags"][bi] and abs(dy) >= .50 * pitch):
                    reliable.append(dy)
            reference = (float(np.median(reliable)) if reliable else
                         next((float(v) for v in signed if v != 0), 1.))
            reference_sign = 1 if reference > 0 else -1
            affected = close | {x for x in ordered_x
                                if reference_sign * (by_x[i][x][0] - by_x[j][x][0]) <= 0}
            for x in affected:
                aidx, bidx = by_x[i][x][1], by_x[j][x][1]
                a_inferred = curves[i]["inferred_flags"][aidx]
                b_inferred = curves[j]["inferred_flags"][bidx]
                reason = ("order_reversal" if reference_sign *
                          (by_x[i][x][0] - by_x[j][x][0]) <= 0 else
                          "spacing_below_half_pitch")
                if a_inferred or not b_inferred:
                    remove[i].add(x)
                    remove_details[i][(curves[j]["group_id"], reason)].add(x)
                if b_inferred or not a_inferred:
                    remove[j].add(x)
                    remove_details[j][(curves[i]["group_id"], reason)].add(x)
            conflicts.append({"group_ids": [curves[i]["group_id"], curves[j]["group_id"]],
                              "reason": "model_crossing" if crossing else "model_spacing_conflict",
                              "affected_columns": len(affected),
                              "affected_x_columns": sorted(affected)})
    for curve, columns, details in zip(curves, remove, remove_details):
        curve["geometry_conflict_blank_columns"] = len(columns)
        curve["geometry_conflict_blank_details"] = [
            {"other_group_id": peer, "reason": reason, "x_columns": sorted(xs)}
            for (peer, reason), xs in sorted(details.items())]
        if not columns:
            continue
        kept = [(xy, flag) for xy, flag in zip(curve["sampled_points_xy"], curve["inferred_flags"])
                if xy[0] not in columns]
        curve["sampled_points_xy"] = [xy for xy, _ in kept]
        curve["inferred_flags"] = [flag for _, flag in kept]
    return conflicts


def fit_surface_from_candidates(sam_surface_mask: np.ndarray, bright_roi_mask: np.ndarray,
                                c3_fit: dict, exclusion_mask: np.ndarray,
                                pitch_px: float, dark_candidates: dict | list,
                                parameters: FullSurfaceParameters | None = None) -> dict:
    """纯几何入口；不读取像素。供隐藏黄点评价在保存候选后独立调用。"""
    p = parameters or FullSurfaceParameters()
    surface, bright, exclusion, pitch = _validate(None, sam_surface_mask,
                                                  bright_roi_mask, exclusion_mask, pitch_px)
    height, width = surface.shape
    zeros = lambda: np.zeros((height, width), np.uint8)
    result = {"groups": [], "curves": [], "dark_observed_mask": zeros(),
              "modeled_mask": zeros(), "inferred_mask": zeros(),
              "smooth_mask": zeros(), "simple_c3_extension_mask": zeros(),
              "rejected_dark_candidates": [],
              "diagnostics": {}}
    seeds, seed_diag = _seeds(c3_fit, surface.shape)
    if not seeds or not np.any(surface):
        result["diagnostics"] = {**seed_diag, "status": "no_seed_or_empty_surface"}
        return result
    surface_x = np.flatnonzero(np.any(surface, axis=0))
    surface_x_bounds = (int(surface_x[0]), int(surface_x[-1]))
    candidate_list = dark_candidates.get("points", []) if isinstance(dark_candidates, dict) else dark_candidates
    if not isinstance(candidate_list, list):
        raise ValueError("dark_candidates must be a list or a dictionary with points")
    by_group = defaultdict(list)
    reasons = Counter()
    ids = {seed["group_id"] for seed in seeds}
    owned = set()
    for candidate in candidate_list:
        try:
            group_id = candidate["group_id"]
            x, y = int(candidate["x"]), int(candidate["y"])
            score = float(candidate.get("normalized_response", candidate.get("score", 0.)))
        except (KeyError, TypeError, ValueError, OverflowError):
            reasons["malformed_dark_candidate"] += 1
            continue
        if group_id not in ids or not 0 <= x < width or not 0 <= y < height:
            reasons["unknown_group_or_outside_image"] += 1
            continue
        if not surface[y, x] or bright[y, x] or exclusion[y, x]:
            reasons["not_in_dark_valid_surface"] += 1
            continue
        if not math.isfinite(score) or score < 0:
            reasons["nonfinite_or_negative_candidate_score"] += 1
            continue
        if (x, y) in owned:
            reasons["duplicate_dark_candidate"] += 1
            continue
        owned.add((x, y))
        by_group[group_id].append({**candidate, "x": x, "y": y,
                                   "normalized_response": score})
    curves = []
    baseline_curves = []
    for seed in seeds:
        group_id = seed["group_id"]
        bright_points = seed["bright_points"]
        dark = sorted(by_group[group_id], key=lambda item: item["x"])
        bright_x_span = float(np.ptp(bright_points[:, 0]))
        weak_seed = bright_x_span < p.min_bright_span_for_long_extension_pitches * pitch
        weak_seed_dark_limits = None
        if weak_seed and dark:
            # 短亮段给不出可靠的远端曲率；只有贴着亮段的连续暗沟可以
            # 参与同组全点拟合。远端暗点即使局部像暗谷也保留为拒识证据。
            reach = max(2. * pitch, bright_x_span)
            x_left = float(np.min(bright_points[:, 0]) - reach)
            x_right = float(np.max(bright_points[:, 0]) + reach)
            weak_seed_dark_limits = [x_left, x_right]
            nearby = []
            for point in dark:
                if x_left <= point["x"] <= x_right:
                    nearby.append(point)
                else:
                    result["rejected_dark_candidates"].append(
                        {"group_id": group_id, "xy": [point["x"], point["y"]],
                         "reason": "weak_seed_far_dark_candidate"})
                    reasons["weak_seed_far_dark_candidate"] += 1
            dark = nearby
            if dark:
                local_tracks, _ = _tracks(dark, max(4, int(round(p.column_step_pitches * pitch))),
                                           pitch, p)
                trusted_xy = {(point["x"], point["y"]) for point in local_tracks}
                for point in dark:
                    if (point["x"], point["y"]) not in trusted_xy:
                        result["rejected_dark_candidates"].append(
                            {"group_id": group_id, "xy": [point["x"], point["y"]],
                             "reason": "weak_seed_local_track_too_short_or_sparse"})
                        reasons["weak_seed_local_track_too_short_or_sparse"] += 1
                dark = local_tracks
        model, weights, residual, regularization = (_joint_fit(seed, dark, pitch) if dark
                                                     else _frozen_bright_fit(seed))
        bright_residual = np.abs(residual[:len(bright_points)])
        old_bright_residual = np.abs(bright_points[:, 1] -
                                     _predict(seed["model"], bright_points[:, 0]))
        dark_residual = np.abs(residual[len(bright_points):])
        visible_drift_limit = (max(1., .05 * pitch) if weak_seed else
                               max(2., .10 * pitch))
        reject_dark = bool(dark) and (
            np.quantile(bright_residual, .90) > np.quantile(old_bright_residual, .90) + visible_drift_limit or
            np.quantile(dark_residual, .90) > .30 * pitch)
        if reject_dark:
            for point in dark:
                result["rejected_dark_candidates"].append({"group_id": group_id,
                                                             "xy": [point["x"], point["y"]],
                                                             "reason": "joint_fit_visible_or_dark_residual"})
            reasons["dark_group_joint_fit_rejected"] += 1
            dark = []
            model, weights, residual, regularization = _frozen_bright_fit(seed)
        observed = np.concatenate((bright_points,
                                   np.asarray([[d["x"], d["y"]] for d in dark], float).reshape(-1, 2)))
        xlo, xhi, domain_diag = _domain(seed, dark, surface_x_bounds, pitch, p)
        sampled, inferred, sample_diag = _sample_model(model, xlo, xhi, surface, exclusion,
                                                        p, observed, pitch)
        simple, simple_flags, simple_diag = _sample_model(seed["model"], xlo, xhi, surface,
                                                          exclusion, p, bright_points, pitch)
        for point in dark:
            result["dark_observed_mask"][point["y"], point["x"]] = 255
        group = {"group_id": group_id, "source_c3_curve_id": seed["c3_curve_id"],
                 "status": "updated_with_dark_observations" if dark else "bright_seed_only",
                 "source_fragment_ids": seed["source_fragment_ids"],
                 "observed_points_xy": observed.tolist(),
                 "point_kind": ["bright_c3_yellow"] * len(bright_points) + ["trusted_dark_valley"] * len(dark),
                 "fit_weights": weights.tolist(), "residuals_px": residual.tolist(),
                 "fit_regularization": regularization,
                 "model": model, "bright_seed_model": seed["model"],
                 "coefficients": model, "sampled_points_xy": sampled,
                 "extension_x_px": domain_diag.get("max_extension_from_observed_x_px"),
                 "bright_observed_count": len(bright_points), "trusted_dark_observed_count": len(dark),
                 "dark_point_evidence": dark, "drawable_domain": domain_diag,
                 "bright_seed_x_span_pitches": bright_x_span / pitch,
                 "weak_seed_local_dark_limits_x_px": weak_seed_dark_limits,
                 "visible_bright_p90_drift_limit_px": visible_drift_limit,
                 "sample_rejection_counts": sample_diag,
                 "interpretation": "one image-local geometric conic; dark points only where image supports a valley"}
        result["groups"].append(group)
        curves.append({"curve_id": f"surface_{group_id}", "group_id": group_id,
                       "model": model, "source_fragment_ids": seed["source_fragment_ids"],
                       "observed_span_x_px": [float(observed[:, 0].min()), float(observed[:, 0].max())],
                       "drawable_span_x_px": [xlo, xhi],
                       "sampled_points_xy": sampled, "inferred_flags": inferred,
                       "support": {"bright_points": len(bright_points), "dark_points": len(dark),
                                   "model_slope_limit": p.max_model_slope,
                                   "max_extrapolation_px": domain_diag.get("max_extension_from_observed_x_px")}})
        baseline_curves.append({"curve_id": f"simple_{group_id}", "group_id": group_id,
                                "sampled_points_xy": simple, "inferred_flags": simple_flags,
                                "sample_rejection_counts": simple_diag})
    conflicts = _spacing_guard(curves, pitch)
    simple_conflicts = _spacing_guard(baseline_curves, pitch)
    final_by_group = {curve["group_id"]: curve["sampled_points_xy"] for curve in curves}
    for group in result["groups"]:
        group["sampled_points_xy"] = final_by_group[group["group_id"]]
    for curve in curves:
        for (x, y), extrapolated in zip(curve["sampled_points_xy"], curve["inferred_flags"]):
            result["modeled_mask"][y, x] = 255
            if extrapolated:
                result["inferred_mask"][y, x] = 255
    for curve in baseline_curves:
        for x, y in curve["sampled_points_xy"]:
            result["simple_c3_extension_mask"][y, x] = 255
    result["curves"] = curves
    result["smooth_mask"] = result["modeled_mask"]
    result["diagnostics"] = {**seed_diag, "status": "completed", "group_count": len(result["groups"]),
                             "groups_with_trusted_dark_points": sum(g["trusted_dark_observed_count"] > 0
                                                                   for g in result["groups"]),
                             "weak_bright_seed_groups": sum(g["bright_seed_x_span_pitches"] <
                                                            p.min_bright_span_for_long_extension_pitches
                                                            for g in result["groups"]),
                             "trusted_dark_point_count": int(np.count_nonzero(result["dark_observed_mask"])),
                             "modeled_pixels": int(np.count_nonzero(result["modeled_mask"])),
                             "inferred_pixels": int(np.count_nonzero(result["inferred_mask"])),
                             "simple_c3_extension_pixels": int(np.count_nonzero(result["simple_c3_extension_mask"])),
                             "model_spacing_conflicts": conflicts,
                             "simple_spacing_conflicts": simple_conflicts,
                             "dark_candidate_rejection_counts": dict(reasons),
                             "no_seed_policy": "no copying or inventing missing rope rows",
                             "identity_scope": "visual boundary within one image; no physical material identity"}
    return result


def fit_full_surface(image_bgr: np.ndarray, sam_surface_mask: np.ndarray,
                     bright_roi_mask: np.ndarray, c3_fit: dict,
                     exclusion_mask: np.ndarray, pitch_px: float,
                     parameters: FullSurfaceParameters | None = None,
                     candidate_points_override: dict | list | None = None) -> dict:
    """常规入口；override 给定时严格跳过原图候选检测，供遮点评价。"""
    p = parameters or FullSurfaceParameters()
    _validate(image_bgr, sam_surface_mask, bright_roi_mask, exclusion_mask, pitch_px)
    detected = (detect_dark_candidates(image_bgr, sam_surface_mask, bright_roi_mask,
                                       c3_fit, exclusion_mask, pitch_px, p)
                if candidate_points_override is None else candidate_points_override)
    result = fit_surface_from_candidates(sam_surface_mask, bright_roi_mask, c3_fit,
                                         exclusion_mask, pitch_px, detected, p)
    result["diagnostics"]["candidate_mode"] = ("automatic_dark_image_detection"
                                                  if candidate_points_override is None else
                                                  "provided_candidates_no_image_detection")
    if candidate_points_override is None:
        result["diagnostics"]["candidate_detection"] = detected["diagnostics"]
        result["diagnostics"]["candidate_track_rejection_counts"] = detected["rejected_counts"]
    return result
