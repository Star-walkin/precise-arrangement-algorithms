"""D3：由亮区锚点与暗区真实黄点/沟槽场联合寻找有序视觉边界。

仅处理开放开发图的已给数组和记录；不读文件、不连接设备。黄色点原封保留，
新路径只是图像与几何候选，不能证明同根材料或物理接触。
"""
from __future__ import annotations

import math
from collections import defaultdict

import cv2
import numpy as np


PARAMETERS = {
    "column_step_pitch": .30,
    "max_absolute_slope": 2.5,
    "minimum_neighbor_gap_pitch": .50,
    "image_supported_score": .55,
    "yellow_supported_distance_pitch": .28,
    "maximum_inferred_gap_pitch": 3.0,
    "assignment_radius_pitch": .55,
    "assignment_uniqueness_pitch": .10,
    "em_rounds": 2,
    "curve_extension_pitch": 2.0,
    "image_window_pitch": 2.5,
    "image_pseudo_step_pitch": 1.0,
    "image_pseudo_shift_pitch": .35,
    "image_pseudo_weight": .50,
    "image_min_window_support_fraction": .55,
    "image_min_window_score": .35,
    "image_min_gain": .045,
    "minimum_drawable_segment_pitch": 1.5,
    "reverse_bend_min_length_pitch": 2.0,
    "reverse_bend_min_slope_change": .10,
    "reverse_bend_min_bow_pitch": .10,
    "shape_refit_max_residual_increase_pitch": .10,
}


def _validate(roi: np.ndarray, pitch: float, field: dict, variant: str
              ) -> tuple[np.ndarray, np.ndarray, int, int, float]:
    if variant not in ("image", "geometry"):
        raise ValueError("variant must be 'image' or 'geometry'")
    if (not isinstance(roi, np.ndarray) or roi.ndim != 2 or
            roi.dtype != np.uint8 or np.any((roi != 0) & (roi != 255))):
        raise ValueError("roi must be uint8 HxW 0/255")
    pitch = float(pitch)
    if not math.isfinite(pitch) or not 8 <= pitch <= 70:
        raise ValueError("pitch must be finite in [8,70]")
    score = np.asarray(field["score"], dtype=np.float32)
    support = np.asarray(field["support"])
    x0, y0 = int(field["x0"]), int(field["y0"])
    if (score.ndim != 2 or score.shape != support.shape or
            not np.isfinite(score).all() or
            (score.size and (np.min(score) < 0 or np.max(score) > 1.00001)) or
            x0 < 0 or y0 < 0 or
            x0 + score.shape[1] > roi.shape[1] or
            y0 + score.shape[0] > roi.shape[0]):
        raise ValueError("field crop, score and support must align with roi")
    return score, support > 0, x0, y0, pitch


def _yellow_points(previous_record: dict) -> list[dict]:
    points = []
    for line in previous_record.get("observed_lines", []):
        old_id = str(line["line_id"])
        for index, xy in enumerate(line.get("observed_points_xy", [])):
            points.append({"old_line_id": old_id, "point_index": index,
                           "xy": [int(xy[0]), int(xy[1])],
                           "old_c3_group_id": line.get("source_c3_group_id")})
    return points


def _anchors(c3_record: dict, pitch: float) -> tuple[list[dict], dict]:
    fit = c3_record.get("fit", c3_record)
    model = (fit.get("model") or {}).get(
        "shared_shape_for_grouping_and_regularization") or {}
    if model.get("status") not in ("positive_shared_parabola", "near_flat_shared_shape"):
        return [], {"reason": "no_identifiable_c3_shared_shape"}
    try:
        center = float(model["center_x_px"])
        scale = float(model["scale_x_px"])
        b = float(model["b_px_per_u"])
        c = float(model["c_px_per_u2"])
    except (KeyError, ValueError, TypeError):
        return [], {"reason": "invalid_c3_shared_shape"}
    if not all(map(math.isfinite, (center, scale, b, c))) or scale <= 0:
        return [], {"reason": "invalid_c3_shared_shape"}

    def base(x: np.ndarray | float) -> np.ndarray:
        u = (np.asarray(x, float) - center) / scale
        return b * u + c * u * u

    found = []
    for group in fit.get("groups", []):
        if group.get("status") != "accepted":
            continue
        points = np.asarray(group.get("observed_points_xy", []), float)
        if points.ndim != 2 or points.shape[1] < 2 or len(points) < 8:
            continue
        if not np.isfinite(points[:, :2]).all():
            continue
        residual = points[:, 1] - base(points[:, 0])
        offset = float(np.median(residual))
        mad = float(np.median(np.abs(residual - offset)))
        if mad > .30 * pitch:
            continue
        found.append({"c3_group_id": str(group["group_id"]),
                      "offset_px": offset, "bright_point_count": len(points),
                      "bright_span_x_px": [float(points[:, 0].min()),
                                           float(points[:, 0].max())],
                      "bright_mad_px": mad})
    found.sort(key=lambda a: (-a["bright_point_count"], a["bright_mad_px"]))
    kept, duplicate = [], []
    for anchor in found:
        near = next((a for a in kept if abs(a["offset_px"] - anchor["offset_px"])
                     < .60 * pitch), None)
        if near is None:
            kept.append(anchor)
        else:
            duplicate.append(anchor["c3_group_id"])
    kept.sort(key=lambda a: a["offset_px"])
    for index, anchor in enumerate(kept):
        anchor["path_id"] = f"groove_path_{index + 1:04d}"
    return kept, {"status": "ok", "center_x_px": center,
                  "scale_x_px": scale, "b_px_per_u": b,
                  "c_px_per_u2": c, "duplicate_anchor_ids": duplicate}


def _yellow_index(points: list[dict]) -> dict[int, np.ndarray]:
    by_x = defaultdict(list)
    for point in points:
        by_x[point["xy"][0]].append(point["xy"][1])
    return {x: np.asarray(sorted(y), float) for x, y in by_x.items()}


def _near_yellow(index: dict[int, np.ndarray], x: int,
                 radius_x: int) -> np.ndarray:
    values = [index[xx] for xx in range(x - radius_x, x + radius_x + 1)
              if xx in index]
    return np.concatenate(values) if values else np.empty(0, float)


def _field_values(x: int, yy: np.ndarray, score: np.ndarray,
                  support: np.ndarray, x0: int, y0: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    value = np.zeros(len(yy), np.float32)
    valid = np.zeros(len(yy), bool)
    ix = x - x0
    if not 0 <= ix < score.shape[1]:
        return value, valid
    iy = np.rint(yy).astype(int) - y0
    inside = (iy >= 0) & (iy < score.shape[0])
    if np.any(inside):
        valid[inside] = support[iy[inside], ix]
        value[inside] = score[iy[inside], ix]
    # 弱 score 仍可指导搜索；只有严格 support 才能把该处称作图像支持。
    return value, valid


def _base_y(x: int | np.ndarray, anchor: dict, shared: dict) -> np.ndarray:
    u = (np.asarray(x, float) - shared["center_x_px"]) / shared["scale_x_px"]
    return (shared["b_px_per_u"] * u + shared["c_px_per_u2"] * u * u +
            anchor["offset_px"])


def _path_support(x: int, y: float, field_parts: tuple,
                  yellow: dict[int, np.ndarray], pitch: float) -> tuple[bool, str]:
    score, support, x0, y0 = field_parts
    values, valid = _field_values(x, np.asarray([y]), score, support, x0, y0)
    image_ok = bool(valid[0] and values[0] >= PARAMETERS["image_supported_score"])
    near = _near_yellow(yellow, x, max(1, int(round(.25 * pitch))))
    yellow_ok = bool(len(near) and np.min(np.abs(near - y)) <=
                     PARAMETERS["yellow_supported_distance_pitch"] * pitch)
    if yellow_ok:
        return True, "existing_yellow"
    if image_ok:
        return True, "groove_field"
    return False, "no_local_evidence"


def _drawable_segments(path_id: str, side: str, geometry: dict[int, float],
                       roi: np.ndarray, field_parts: tuple,
                       yellow: dict[int, np.ndarray], pitch: float
                       ) -> tuple[list[dict], dict[int, int], dict]:
    eligible = {}
    support_reason = {}
    for x, y in geometry.items():
        yi = int(round(y))
        if 0 <= x < roi.shape[1] and 0 <= yi < roi.shape[0] and roi[yi, x]:
            eligible[x] = yi
            ok, reason = _path_support(x, y, field_parts, yellow, pitch)
            if ok:
                support_reason[x] = reason
    supported_x = sorted(support_reason)
    drawable = set(supported_x)
    for a, b in zip(supported_x, supported_x[1:]):
        if b - a <= PARAMETERS["maximum_inferred_gap_pitch"] * pitch:
            drawable.update(x for x in range(a + 1, b) if x in eligible)
    segments = []
    current_points, current_states = [], []

    def flush() -> None:
        if not current_points:
            return
        if (len(current_points) >= max(6, int(round(
                PARAMETERS["minimum_drawable_segment_pitch"] * pitch))) and
                sum(state == "supported_fit" for state in current_states) >=
                max(4, int(round(.5 * pitch)))):
            segments.append({"curve_id": f"{path_id}_{side}_seg{len(segments) + 1:03d}",
                             "path_id": path_id, "side": side,
                             "sampled_points_xy": current_points.copy(),
                             "point_states": current_states.copy(),
                             "model": "shared_quartic_plus_individual_cubic",
                             "interpretation": "visual groove candidate; gap state is geometry only"})
        current_points.clear()
        current_states.clear()

    for x in sorted(drawable):
        y = eligible[x]
        if current_points and (x != current_points[-1][0] + 1 or
                               abs(y - current_points[-1][1]) > 4):
            flush()
        current_points.append([x, y])
        current_states.append("supported_fit" if x in support_reason
                              else "inferred_short_gap")
    flush()
    return segments, eligible, {"supported_columns": len(supported_x),
                                "drawable_columns": len(drawable),
                                "support_from_yellow": sum(v == "existing_yellow"
                                                           for v in support_reason.values()),
                                "support_from_field": sum(v == "groove_field"
                                                          for v in support_reason.values())}


def _dark_side_columns(roi: np.ndarray, field_parts: tuple, anchors: list[dict],
                  pitch: float) -> dict[str, np.ndarray]:
    score, _, x0, _ = field_parts
    valid_x = np.flatnonzero(np.any(roi[:, x0:x0 + score.shape[1]] > 0, axis=0)) + x0
    if not len(valid_x):
        return {"left": np.empty(0, int), "right": np.empty(0, int)}
    inner_left = min(a["bright_span_x_px"][0] for a in anchors)
    inner_right = max(a["bright_span_x_px"][1] for a in anchors)
    step = max(4, int(round(PARAMETERS["column_step_pitch"] * pitch)))
    result = {}
    for side, available in (("left", valid_x[valid_x < inner_left]),
                            ("right", valid_x[valid_x > inner_right])):
        if not len(available):
            result[side] = np.empty(0, int)
            continue
        lo, hi = int(available[0]), int(available[-1])
        nodes = np.arange(hi, lo - 1, -step) if side == "left" else np.arange(lo, hi + 1, step)
        if nodes[-1] != (lo if side == "left" else hi):
            nodes = np.r_[nodes, lo if side == "left" else hi]
        result[side] = nodes.astype(int)
    return result


def _poly_value(x: np.ndarray | float, coefficients: np.ndarray,
                center: float, scale: float) -> np.ndarray:
    u = (np.asarray(x, float) - center) / scale
    return np.polynomial.polynomial.polyval(u, coefficients)


def _robust_poly(x: np.ndarray, y: np.ndarray, base_weight: np.ndarray,
                 degree: int, center: float, scale: float,
                 pitch: float) -> tuple[np.ndarray, float]:
    """固定低自由度Huber拟合，绝不逐列追峰。"""
    u = (np.asarray(x, float) - center) / scale
    design = np.stack([u ** i for i in range(degree + 1)], axis=1)
    penalty = np.asarray(([1e-6, .003, .025, .075, .12])[:degree + 1], float)
    weight = np.asarray(base_weight, float).copy()
    huber = max(1., .18 * pitch)
    normal = np.zeros((degree + 1, degree + 1), float)
    rhs = np.zeros(degree + 1, float)
    coefficients = np.zeros(degree + 1, float)
    for _ in range(7):
        normal = design.T @ (weight[:, None] * design) + np.diag(penalty)
        rhs = design.T @ (weight * y)
        try:
            coefficients = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(normal, rhs, rcond=None)[0]
        residual = y - design @ coefficients
        updated = base_weight * np.maximum(.08, np.minimum(1., huber /
                                       np.maximum(np.abs(residual), 1e-8)))
        if np.max(np.abs(updated - weight)) < 1e-3:
            break
        weight = updated
    mad = float(np.median(np.abs(y - design @ coefficients)))
    return coefficients, mad


def _lowdim_y(model: dict, x: np.ndarray | float,
              anchor: dict, shared_c3: dict) -> np.ndarray:
    return (_base_y(x, anchor, shared_c3) +
            _poly_value(x, model["shared_coefficients"], model["center"], model["scale"]) +
            _poly_value(x, model["individual_coefficients"], model["center"], model["scale"]) +
            model.get("ordered_global_shift_px", 0.))


def _curvature_components(model: dict, x: np.ndarray,
                          shared_c3: dict) -> tuple[np.ndarray, np.ndarray]:
    """返回总曲率的固定部分和逐行三次部分；y 轴向下，正值为预期弯向。"""
    u = (np.asarray(x, float) - model["center"]) / model["scale"]
    shared = model["shared_coefficients"]
    individual = model["individual_coefficients"]
    fixed = (2 * shared_c3["c_px_per_u2"] /
             shared_c3["scale_x_px"] ** 2 +
             (2 * shared[2] + 6 * shared[3] * u + 12 * shared[4] * u * u) /
             model["scale"] ** 2)
    individual_part = (2 * individual[2] + 6 * individual[3] * u) / model["scale"] ** 2
    return fixed, individual_part


def _reverse_bend(model: dict, anchor: dict, shared_c3: dict,
                  x_first: int, x_last: int, pitch: float) -> dict:
    """只把跨越足够长度和转角的反弯当成结构问题；微小负导数不触发。"""
    xs = np.arange(int(x_first), int(x_last) + 1, dtype=float)
    if len(xs) < 3:
        return {"meaningful": False, "longest_run_px": 0,
                "maximum_reverse_slope_change": 0., "maximum_bow_pitch": 0.}
    fixed, individual = _curvature_components(model, xs, shared_c3)
    negative = fixed + individual < -1e-7
    starts = np.flatnonzero(negative & ~np.r_[False, negative[:-1]])
    ends = np.flatnonzero(negative & ~np.r_[negative[1:], False])
    best_length, best_change, best_bow = 0, 0., 0.
    for start, end in zip(starts, ends):
        length = int(end - start + 1)
        if length < PARAMETERS["reverse_bend_min_length_pitch"] * pitch:
            continue
        x_run = xs[start:end + 1]
        curve = np.asarray(_lowdim_y(model, x_run, anchor, shared_c3), float)
        chord = np.linspace(curve[0], curve[-1], len(curve))
        bow = float(np.max(np.abs(curve - chord)) / pitch)
        change = float(-np.sum((fixed + individual)[start:end + 1]))
        best_length = max(best_length, length)
        best_change = max(best_change, change)
        best_bow = max(best_bow, bow)
    return {"meaningful": bool(best_change >= PARAMETERS["reverse_bend_min_slope_change"]
                                or best_bow >= PARAMETERS["reverse_bend_min_bow_pitch"]),
            "longest_run_px": best_length,
            "maximum_reverse_slope_change": best_change,
            "maximum_bow_pitch": best_bow}


def _constrained_cubic(x: np.ndarray, target: np.ndarray,
                       base_weight: np.ndarray, model: dict,
                       shared_c3: dict, pitch: float) -> np.ndarray | None:
    """带 Huber 权重的四参数三次最小二乘；只约束总曲率，不修改黄点。"""
    center, scale = model["center"], model["scale"]
    u = (np.asarray(x, float) - center) / scale
    design = np.stack([np.ones(len(u)), u, u * u, u ** 3], axis=1)
    penalty = np.asarray([1e-6, .003, .025, .075], float)
    weight = np.asarray(base_weight, float).copy()
    huber = max(1., .18 * pitch)
    grid_x = np.linspace(float(np.min(x)), float(np.max(x)), 9)
    grid_u = (grid_x - center) / scale
    shared = model["shared_coefficients"]
    fixed_scaled = (2 * shared_c3["c_px_per_u2"] *
                    (scale / shared_c3["scale_x_px"]) ** 2 +
                    2 * shared[2] + 6 * shared[3] * grid_u +
                    12 * shared[4] * grid_u ** 2)
    constraints = np.stack([np.full(len(grid_u), 2.), 6 * grid_u], axis=1)
    lower = -fixed_scaled

    def solve(weight_now: np.ndarray, matrix: np.ndarray,
              threshold: np.ndarray) -> np.ndarray | None:
        normal = design.T @ (weight_now[:, None] * design) + np.diag(penalty)
        rhs = design.T @ (weight_now * target)
        nww, nwz = normal[:2, :2], normal[:2, 2:]
        nzw, nzz = normal[2:, :2], normal[2:, 2:]
        try:
            inv_w_nwz = np.linalg.solve(nww, nwz)
            inv_w_rhs = np.linalg.solve(nww, rhs[:2])
            metric = nzz - nzw @ inv_w_nwz
            z0 = np.linalg.solve(metric, rhs[2:] - nzw @ inv_w_rhs)
        except np.linalg.LinAlgError:
            return None
        feasible = lambda z: bool(np.all(matrix @ z >= threshold - 1e-7))
        candidates = [z0] if feasible(z0) else []
        for row, bound in zip(matrix, threshold):
            try:
                direction = np.linalg.solve(metric, row)
            except np.linalg.LinAlgError:
                return None
            denominator = float(row @ direction)
            if denominator <= 1e-12:
                continue
            z = z0 + direction * ((bound - float(row @ z0)) / denominator)
            if feasible(z):
                candidates.append(z)
        for i in range(len(matrix)):
            for j in range(i + 1, len(matrix)):
                pair = matrix[[i, j]]
                if abs(float(np.linalg.det(pair))) < 1e-8:
                    continue
                z = np.linalg.solve(pair, threshold[[i, j]])
                if feasible(z):
                    candidates.append(z)
        if not candidates:
            return None
        z = min(candidates, key=lambda v: float((v - z0) @ metric @ (v - z0)))
        w = inv_w_rhs - inv_w_nwz @ z
        return np.r_[w, z]

    coefficients = None
    for _ in range(5):
        coefficients = solve(weight, constraints, lower)
        if coefficients is None:
            return None
        residual = target - design @ coefficients
        updated = base_weight * np.maximum(.08, np.minimum(1., huber /
                                   np.maximum(np.abs(residual), 1e-8)))
        if np.max(np.abs(updated - weight)) < 1e-3:
            break
        weight = updated
    # 最后一次权重也要真正用于求解；采样遗漏的曲率最低点补为约束再解。
    coefficients = solve(weight, constraints, lower)
    if coefficients is None:
        return None
    for _ in range(2):
        dense_x = np.arange(int(np.floor(np.min(x))),
                            int(np.ceil(np.max(x))) + 1, dtype=float)
        probe = dict(model, individual_coefficients=coefficients)
        fixed, individual = _curvature_components(probe, dense_x, shared_c3)
        worst = int(np.argmin(fixed + individual))
        if float((fixed + individual)[worst]) >= -1e-7:
            return coefficients
        x_bad = dense_x[worst]
        u_bad = (x_bad - center) / scale
        extra = np.asarray([[2., 6 * u_bad]])
        shared = model["shared_coefficients"]
        fixed_bad = (2 * shared_c3["c_px_per_u2"] *
                     (scale / shared_c3["scale_x_px"]) ** 2 +
                     2 * shared[2] + 6 * shared[3] * u_bad +
                     12 * shared[4] * u_bad * u_bad)
        constraints = np.vstack([constraints, extra])
        lower = np.r_[lower, -fixed_bad]
        coefficients = solve(weight, constraints, lower)
        if coefficients is None:
            return None
    return coefficients


def _shape_guard(model: dict, anchor: dict, shared_c3: dict,
                 points: list[dict], pseudo: list[tuple[float, float, float]],
                 pitch: float) -> dict:
    """仅对明显反弯尝试约束拟合；不符合真实黄点时保留并标歧义。"""
    lo, hi = model["yellow_span_x_px"]
    diagnosis = _reverse_bend(model, anchor, shared_c3, lo, hi, pitch)
    if not diagnosis["meaningful"]:
        return {**diagnosis, "status": "no_meaningful_reverse_bend"}
    if len(points) < 6:
        return {**diagnosis, "status": "ambiguous_reverse_bend_too_few_points"}
    yellow_x = np.asarray([point["xy"][0] for point in points], float)
    yellow_y = np.asarray([point["xy"][1] for point in points], float)
    x = np.r_[yellow_x, np.asarray([item[0] for item in pseudo], float)]
    y = np.r_[yellow_y, np.asarray([item[1] for item in pseudo], float)]
    weight = np.r_[np.ones(len(points), float),
                   np.asarray([item[2] for item in pseudo], float)]
    target = (y - _base_y(x, anchor, shared_c3) -
              _poly_value(x, model["shared_coefficients"],
                          model["center"], model["scale"]))
    before = np.abs(yellow_y - _lowdim_y(model, yellow_x, anchor, shared_c3))
    coefficients = _constrained_cubic(x, target, weight, model, shared_c3, pitch)
    if coefficients is None or not np.isfinite(coefficients).all():
        return {**diagnosis, "status": "ambiguous_reverse_bend_infeasible_refit"}
    candidate = dict(model, individual_coefficients=coefficients)
    dense_x = np.arange(lo, hi + 1, dtype=float)
    fixed, individual = _curvature_components(candidate, dense_x, shared_c3)
    if (np.min(fixed + individual) < -1e-6 or
            np.max(np.abs(np.diff(_lowdim_y(candidate, dense_x, anchor,
                                          shared_c3)))) >
            PARAMETERS["max_absolute_slope"]):
        return {**diagnosis, "status": "ambiguous_reverse_bend_constraint_failed"}
    after = np.abs(yellow_y - _lowdim_y(candidate, yellow_x, anchor, shared_c3))
    before_median, after_median = float(np.median(before)), float(np.median(after))
    before_p90, after_p90 = (float(np.quantile(before, .90)),
                             float(np.quantile(after, .90)))
    details = {"yellow_count_used": len(points),
               "same_image_pseudo_count_used": len(pseudo),
               "before_median_yellow_residual_px": before_median,
               "after_median_yellow_residual_px": after_median,
               "before_p90_yellow_residual_px": before_p90,
               "after_p90_yellow_residual_px": after_p90}
    budget = PARAMETERS["shape_refit_max_residual_increase_pitch"] * pitch
    if (after_median - before_median >= budget or
            after_p90 - before_p90 >= budget):
        return {**diagnosis, **details,
                "status": "ambiguous_reverse_bend_refit_hurts_yellow"}
    model["individual_coefficients"] = coefficients
    model["fit_mad_px"] = after_median
    return {**diagnosis, **details, "status": "accepted_constrained_refit"}


def _assign_for_em(points: list[dict], models: dict[str, dict],
                   anchor_by_id: dict[str, dict], shared_c3: dict,
                   pitch: float) -> dict[str, list[dict]]:
    assigned = {path_id: [] for path_id in models}
    for point in points:
        x, y = point["xy"]
        options = []
        for path_id, model in models.items():
            lo, hi = model["yellow_span_x_px"]
            if x < lo - 2 * pitch or x > hi + 2 * pitch:
                continue
            target = float(_lowdim_y(model, x, anchor_by_id[path_id], shared_c3))
            options.append((abs(y - target), path_id))
        options.sort()
        if (options and options[0][0] <= PARAMETERS["assignment_radius_pitch"] * pitch and
                (len(options) == 1 or options[1][0] - options[0][0] >=
                 PARAMETERS["assignment_uniqueness_pitch"] * pitch)):
            assigned[options[0][1]].append(point)
    return assigned


def _image_pseudo(model: dict, anchor: dict, shared_c3: dict,
                  field_parts: tuple, pitch: float) -> list[tuple[float, float, float]]:
    """沿整条当前曲线的2.5p窗口比较平移，窗口内多数原图支持才提名。"""
    lo, hi = model["yellow_span_x_px"]
    half = .5 * PARAMETERS["image_window_pitch"] * pitch
    stride = max(4, int(round(PARAMETERS["image_pseudo_step_pitch"] * pitch)))
    if hi - lo < 2 * half:
        return []
    centers = np.arange(int(math.ceil(lo + half)),
                        int(math.floor(hi - half)) + 1, stride)
    shifts = np.arange(-PARAMETERS["image_pseudo_shift_pitch"] * pitch,
                       PARAMETERS["image_pseudo_shift_pitch"] * pitch + 1,
                       max(1., .10 * pitch))
    if not np.any(np.abs(shifts) < 1e-6):
        shifts = np.r_[shifts, 0.]
    result = []
    field_score, field_support, x0, y0 = field_parts
    for center in centers:
        sample_x = np.rint(np.linspace(center - half, center + half, 11)).astype(int)
        baseline = np.asarray(_lowdim_y(model, sample_x, anchor, shared_c3), float)
        ix = sample_x - x0
        valid_x = (ix >= 0) & (ix < field_score.shape[1])
        merits = []
        for shift in shifts:
            iy = np.rint(baseline + shift).astype(int) - y0
            inside = valid_x & (iy >= 0) & (iy < field_score.shape[0])
            scores = np.zeros(len(sample_x), float)
            supported = np.zeros(len(sample_x), bool)
            if np.any(inside):
                scores[inside] = field_score[iy[inside], ix[inside]]
                supported[inside] = field_support[iy[inside], ix[inside]]
            fraction = float(np.mean(supported))
            merit = float(np.mean(scores * supported)) if fraction >= \
                PARAMETERS["image_min_window_support_fraction"] else -1.
            merits.append((merit, fraction, float(shift)))
        best = max(merits, key=lambda item: item[0])
        zero_merit = min(merits, key=lambda item: abs(item[2]))[0]
        if (best[0] >= PARAMETERS["image_min_window_score"] and
                best[0] - zero_merit >= PARAMETERS["image_min_gain"] and
                abs(best[2]) > .04 * pitch):
            target = float(_lowdim_y(model, center, anchor, shared_c3)) + best[2]
            result.append((float(center), target,
                           PARAMETERS["image_pseudo_weight"] * best[1]))
    return result


def _global_order_guard(models: dict[str, dict], anchor_by_id: dict[str, dict],
                        shared_c3: dict, pitch: float, roi: np.ndarray,
                        side_bounds: tuple[int, int]) -> dict:
    """仅做整行常量位移；严重相交者整行拒识，不制造逐列PAVA台阶。"""
    ordered = sorted(models, key=lambda pid: anchor_by_id[pid]["offset_px"])
    # 绘制时会取整 y；多留一像素，避免连续曲线恰好达标却在像素上近碰。
    min_gap = float(math.ceil(PARAMETERS["minimum_neighbor_gap_pitch"] * pitch) + 1)
    rejected = []
    corrections = []
    for _ in range(3 * max(1, len(ordered))):
        changed = False
        restarted = False
        # 中间编号的曲线可能在本列已结束；此时非连续编号也会成为
        # 局部可见相邻线，因此检查所有绘制范围有重叠的有序线对。
        for upper_index in range(len(ordered) - 1):
            for lower_index in range(upper_index + 1, len(ordered)):
                upper_id, lower_id = ordered[upper_index], ordered[lower_index]
                upper, lower = models[upper_id], models[lower_id]
                # 与下方真正生成 dense_x 的 round/clamp 完全相同。
                extension = PARAMETERS["curve_extension_pitch"] * pitch
                lo = max(side_bounds[0],
                         int(round(upper["yellow_span_x_px"][0] - extension)),
                         int(round(lower["yellow_span_x_px"][0] - extension)))
                hi = min(side_bounds[1],
                         int(round(upper["yellow_span_x_px"][1] + extension)),
                         int(round(lower["yellow_span_x_px"][1] + extension)))
                if hi < lo:
                    continue
                xx = np.arange(lo, hi + 1, dtype=float)
                yy_upper = np.asarray(_lowdim_y(upper, xx, anchor_by_id[upper_id],
                                               shared_c3), float)
                yy_lower = np.asarray(_lowdim_y(lower, xx, anchor_by_id[lower_id],
                                               shared_c3), float)
                xi = xx.astype(int)
                yu, yl = np.rint(yy_upper).astype(int), np.rint(yy_lower).astype(int)
                both_inside = ((xi >= 0) & (xi < roi.shape[1]) &
                               (yu >= 0) & (yu < roi.shape[0]) &
                               (yl >= 0) & (yl < roi.shape[0]))
                if np.any(both_inside):
                    visible_index = np.flatnonzero(both_inside)
                    both_inside[visible_index] = ((roi[yu[visible_index], xi[visible_index]] > 0) &
                                                  (roi[yl[visible_index], xi[visible_index]] > 0))
                if not np.any(both_inside):
                    continue
                separation = yy_lower[both_inside] - yy_upper[both_inside]
                needed = min_gap - float(np.min(separation))
                if needed <= 1e-5:
                    continue
                if needed > .45 * pitch:
                    weak_id = (upper_id if upper["yellow_point_count"] <
                               lower["yellow_point_count"] else lower_id)
                    ordered.remove(weak_id)
                    rejected.append({"path_id": weak_id,
                                     "reason": "severe_lowdim_neighbor_crossing",
                                     "required_shift_px": needed,
                                     "other_path_id": lower_id if weak_id == upper_id
                                     else upper_id})
                    changed = True
                    restarted = True
                    break
                lower["ordered_global_shift_px"] += needed
                corrections.append({"path_id": lower_id, "shift_px": needed,
                                    "other_path_id": upper_id})
                changed = True
            if restarted:
                break
        if not changed:
            break
    for item in rejected:
        models.pop(item["path_id"], None)
    return {"minimum_gap_px": min_gap, "constant_shift_repairs": corrections,
            "rejected_paths": rejected}


def fit_groove_paths(previous_record: dict, c3_record: dict,
                     roi: np.ndarray, pitch: float, field: dict,
                     variant: str = "image") -> dict:
    """联合搜索暗沟路径，原 D2 黄点不动；输出几何候选及每点新归属。"""
    score, support, x0, y0, pitch = _validate(roi, pitch, field, variant)
    field_parts = (score, support, x0, y0)
    observed_lines = previous_record.get("observed_lines", [])
    yellow_points = _yellow_points(previous_record)
    yellow_index = _yellow_index(yellow_points)
    observed_mask = np.zeros_like(roi)
    smooth_mask = np.zeros_like(roi)
    inferred_mask = np.zeros_like(roi)
    for point in yellow_points:
        x, y = point["xy"]
        if 0 <= x < roi.shape[1] and 0 <= y < roi.shape[0] and roi[y, x]:
            cv2.circle(observed_mask, (x, y), 1, 255, -1, cv2.LINE_8)
    observed_mask[roi == 0] = 0
    anchors, shared = _anchors(c3_record, pitch)
    new_groups = [{"path_id": a["path_id"], "c3_group_id": a["c3_group_id"],
                   "bright_anchor_point_count": a["bright_point_count"],
                   "assigned_points": []} for a in anchors]
    group_by_id = {g["path_id"]: g for g in new_groups}
    if not anchors or not yellow_points or not np.any(roi):
        assignments = [{**point, "new_path_id": None, "status": "unassigned",
                        "reason": "no_c3_anchor_or_dark_roi"} for point in yellow_points]
        diagnostics = {"status": "no_anchor_or_observations_or_roi",
                       "reason": "no_c3_anchor" if not anchors else
                       "no_previous_yellow_points" if not yellow_points else
                       "no_dark_roi", "variant": variant,
                       "yellow_input_count": len(yellow_points),
                       "assigned_count": 0, "unassigned_count": len(assignments),
                       "field_diagnostics": field.get("diagnostics", {})}
        return {"observed_lines": observed_lines, "observed_mask": observed_mask,
                "new_groups": new_groups, "point_assignments": assignments,
                "fit": {"model": "shared_quartic_plus_individual_cubic",
                        "curves": [], "smooth_mask": smooth_mask,
                        "diagnostics": diagnostics},
                "inferred_mask": inferred_mask, "diagnostics": diagnostics}
    columns = _dark_side_columns(roi, field_parts, anchors, pitch)
    anchor_by_id = {a["path_id"]: a for a in anchors}
    geometry = {}
    path_diagnostics = []
    rejected_paths = []
    global_order_diagnostics = []
    model_by_side = {}
    for side, xs in columns.items():
        if not len(xs):
            continue
        lo, hi = int(np.min(xs)), int(np.max(xs))
        center = .5 * (lo + hi)
        scale = max(.5 * (hi - lo), pitch)
        side_points = [p for p in yellow_points if lo <= p["xy"][0] <= hi]
        initially_labeled = defaultdict(list)
        for point in side_points:
            initially_labeled[point["old_c3_group_id"]].append(point)
        common_x, common_y = [], []
        for anchor in anchors:
            for point in initially_labeled[anchor["c3_group_id"]]:
                x, y = point["xy"]
                common_x.append(x)
                common_y.append(y - float(_base_y(x, anchor, shared)))
        if len(common_x) >= 20:
            shared_coeff, shared_mad = _robust_poly(
                np.asarray(common_x, float), np.asarray(common_y, float),
                np.ones(len(common_x), float), 4, center, scale, pitch)
        else:
            shared_coeff, shared_mad = np.zeros(5, float), None
        models = {}
        for anchor in anchors:
            points = initially_labeled[anchor["c3_group_id"]]
            if len(points) < 6 or max(p["xy"][0] for p in points) - \
                    min(p["xy"][0] for p in points) < 1.5 * pitch:
                rejected_paths.append({"path_id": anchor["path_id"], "side": side,
                                       "reason": "no_stable_initial_yellow_run"})
                continue
            x = np.asarray([p["xy"][0] for p in points], float)
            y = np.asarray([p["xy"][1] for p in points], float)
            target = (y - _base_y(x, anchor, shared) -
                      _poly_value(x, shared_coeff, center, scale))
            individual, mad = _robust_poly(x, target, np.ones(len(x)), 3,
                                           center, scale, pitch)
            models[anchor["path_id"]] = {
                "center": center, "scale": scale,
                "shared_coefficients": shared_coeff,
                "individual_coefficients": individual,
                "yellow_span_x_px": [int(x.min()), int(x.max())],
                "yellow_point_count": len(points), "fit_mad_px": mad,
                "ordered_global_shift_px": 0., "image_pseudo_count": 0}
        em_assignment = {}
        for _ in range(PARAMETERS["em_rounds"]):
            em_assignment = _assign_for_em(side_points, models, anchor_by_id,
                                            shared, pitch)
            for path_id, model in models.items():
                points = em_assignment[path_id]
                if len(points) < 6:
                    continue
                x = np.asarray([p["xy"][0] for p in points], float)
                if np.ptp(x) < 1.5 * pitch:
                    continue
                y = np.asarray([p["xy"][1] for p in points], float)
                target = (y - _base_y(x, anchor_by_id[path_id], shared) -
                          _poly_value(x, shared_coeff, center, scale))
                individual, mad = _robust_poly(x, target, np.ones(len(x)), 3,
                                               center, scale, pitch)
                model["individual_coefficients"] = individual
                model["yellow_span_x_px"] = [int(x.min()), int(x.max())]
                model["yellow_point_count"] = len(points)
                model["fit_mad_px"] = mad
        pseudo_by_path = {path_id: [] for path_id in models}
        if variant == "image":
            for path_id, model in models.items():
                pseudo = _image_pseudo(model, anchor_by_id[path_id], shared,
                                       field_parts, pitch)
                pseudo_by_path[path_id] = pseudo
                model["image_pseudo_count"] = len(pseudo)
                if not pseudo:
                    continue
                points = em_assignment.get(path_id, [])
                x = np.asarray([p["xy"][0] for p in points] +
                               [p[0] for p in pseudo], float)
                y = np.asarray([p["xy"][1] for p in points] +
                               [p[1] for p in pseudo], float)
                weights = np.asarray([1.] * len(points) + [p[2] for p in pseudo], float)
                if len(x) < 6:
                    continue
                target = (y - _base_y(x, anchor_by_id[path_id], shared) -
                          _poly_value(x, shared_coeff, center, scale))
                model["individual_coefficients"], model["fit_mad_px"] = _robust_poly(
                    x, target, weights, 3, center, scale, pitch)
        for path_id, model in models.items():
            model["reverse_bend_shape_guard"] = _shape_guard(
                model, anchor_by_id[path_id], shared,
                em_assignment.get(path_id, []), pseudo_by_path[path_id], pitch)
        order_diag = _global_order_guard(models, anchor_by_id, shared, pitch, roi,
                                         (lo, hi))
        global_order_diagnostics.append({"side": side, **order_diag})
        rejected_paths.extend({"side": side, **item} for item in
                              order_diag["rejected_paths"])
        model_by_side[side] = models
        for path_id, model in models.items():
            span_lo, span_hi = model["yellow_span_x_px"]
            begin = max(lo, int(round(span_lo - PARAMETERS["curve_extension_pitch"] * pitch)))
            end = min(hi, int(round(span_hi + PARAMETERS["curve_extension_pitch"] * pitch)))
            dense_x = np.arange(begin, end + 1, dtype=int)
            dense_y = _lowdim_y(model, dense_x, anchor_by_id[path_id], shared)
            if (len(dense_x) < 2 or np.max(np.abs(np.diff(dense_y))) >
                    PARAMETERS["max_absolute_slope"]):
                rejected_paths.append({"path_id": path_id, "side": side,
                                       "reason": "lowdim_curve_exceeds_slope_budget"})
                continue
            geometry[(path_id, side)] = {int(x): float(y) for x, y in
                                         zip(dense_x, dense_y)}
            group_by_id[path_id].setdefault("side_models", []).append({
                "side": side, "center_x_px": center, "scale_x_px": scale,
                "shared_quartic_coefficients_px": shared_coeff.tolist(),
                "individual_cubic_coefficients_px":
                    model["individual_coefficients"].tolist(),
                "yellow_point_count": model["yellow_point_count"],
                "yellow_span_x_px": model["yellow_span_x_px"],
                "fit_mad_px": model["fit_mad_px"],
                "coherent_image_pseudo_count": model["image_pseudo_count"],
                "ordered_global_shift_px": model["ordered_global_shift_px"],
                "reverse_bend_shape_guard": model["reverse_bend_shape_guard"]})
            path_diagnostics.append({"path_id": path_id, "side": side,
                                     "yellow_point_count": model["yellow_point_count"],
                                     "fit_mad_px": model["fit_mad_px"],
                                     "coherent_image_pseudo_count":
                                         model["image_pseudo_count"],
                                     "ordered_global_shift_px":
                                         model["ordered_global_shift_px"],
                                     "reverse_bend_shape_guard":
                                         model["reverse_bend_shape_guard"]})
        path_diagnostics.append({"side": side, "shared_quartic_coefficients_px":
                                 shared_coeff.tolist(),
                                 "initial_shared_fit_mad_px": shared_mad,
                                 "initial_labeled_yellow_count": len(common_x)})

    curves = []
    drawable_by_x: dict[int, list[int]] = defaultdict(list)
    support_totals = defaultdict(int)
    for (path_id, side), dense in geometry.items():
        segments, _, support_diag = _drawable_segments(path_id, side, dense,
                                                        roi, field_parts, yellow_index,
                                                        pitch)
        projection_max = abs(model_by_side[side][path_id]["ordered_global_shift_px"])
        for segment in segments:
            segment["ordered_projection_max_adjustment_px"] = projection_max
            model = model_by_side[side][path_id]
            span_x = segment["sampled_points_xy"]
            bend = _reverse_bend(model, anchor_by_id[path_id], shared,
                                 span_x[0][0], span_x[-1][0], pitch)
            segment["reverse_bend_diagnostics"] = bend
            segment["reverse_bend_shape_guard"] = model["reverse_bend_shape_guard"]
            reasons = []
            if projection_max > .2 * pitch:
                reasons.append("ambiguous_global_order_shift")
            if (model["reverse_bend_shape_guard"]["status"].startswith("ambiguous_")
                    or bend["meaningful"]):
                reasons.append("ambiguous_reverse_bend")
            segment["quality_reasons"] = reasons
            segment["quality"] = reasons[0] if reasons else "candidate"
        for key, value in support_diag.items():
            support_totals[key] += value
        curves.extend(segments)
        for segment in segments:
            for x, y in segment["sampled_points_xy"]:
                drawable_by_x[x].append(y)
                cv2.circle(smooth_mask, (x, y), 1, 255, -1, cv2.LINE_8)
            for (x, y), state in zip(segment["sampled_points_xy"],
                                      segment["point_states"]):
                if state == "inferred_short_gap":
                    cv2.circle(inferred_mask, (x, y), 1, 255, -1, cv2.LINE_8)

    assignments = []
    reassigned = 0
    unassigned_reason = defaultdict(int)
    nearest_distances = []
    full_denominator_loss = 0.
    full_denominator_hit = 0
    for point in yellow_points:
        x, y = point["xy"]
        distances = []
        for anchor in anchors:
            path_id = anchor["path_id"]
            for side in ("left", "right"):
                candidate = geometry.get((path_id, side), {}).get(x)
                if candidate is not None:
                    distances.append((abs(y - candidate), path_id,
                                      anchor["c3_group_id"], side))
        distances.sort(key=lambda item: item[0])
        entry = dict(point)
        if not distances or distances[0][0] > PARAMETERS["assignment_radius_pitch"] * pitch:
            entry.update({"new_path_id": None, "status": "unassigned",
                          "reason": "no_nearby_new_path"})
        elif len(distances) > 1 and distances[1][0] - distances[0][0] < \
                PARAMETERS["assignment_uniqueness_pitch"] * pitch:
            entry.update({"new_path_id": None, "status": "unassigned",
                          "reason": "ambiguous_neighbor_paths",
                          "nearest_distance_px": float(distances[0][0])})
        else:
            _, path_id, new_c3_id, side = distances[0]
            entry.update({"new_path_id": path_id, "new_c3_group_id": new_c3_id,
                          "status": "assigned", "side": side,
                          "nearest_distance_px": float(distances[0][0]),
                          "reason": "uniquely_nearest_ordered_path"})
            group_by_id[path_id]["assigned_points"].append({
                "old_line_id": point["old_line_id"],
                "point_index": point["point_index"], "xy": [x, y],
                "distance_px": float(distances[0][0])})
            reassigned += new_c3_id != point["old_c3_group_id"]
        if entry["new_path_id"] is None:
            unassigned_reason[entry["reason"]] += 1
        assignments.append(entry)
        options = drawable_by_x.get(x, [])
        if options:
            nearest = float(min(abs(y - yy) for yy in options))
            nearest_distances.append(nearest)
            full_denominator_loss += min(nearest / pitch, 1.)
            full_denominator_hit += nearest <= .25 * pitch
        else:
            full_denominator_loss += 1.
    # 曲线的“可信”只相对于原黄色图像提名点；保留旧组全分母供复核，
    # 不把重新归类后只剩的易点集当作真实准确率。
    assigned_errors = defaultdict(list)
    original_count = defaultdict(int)
    original_capped_loss = defaultdict(float)
    c3_to_path = {a["c3_group_id"]: a["path_id"] for a in anchors}
    for entry in assignments:
        new_path = entry["new_path_id"]
        if new_path is not None:
            assigned_errors[new_path].append(entry["nearest_distance_px"])
        old_path = c3_to_path.get(entry["old_c3_group_id"])
        if old_path is None:
            continue
        original_count[old_path] += 1
        x, y = entry["xy"]
        values = [geometry[(old_path, side)][x]
                  for side in ("left", "right")
                  if (old_path, side) in geometry and x in geometry[(old_path, side)]]
        original_capped_loss[old_path] += min(abs(y - values[0]) / pitch, 1.) \
            if values else 1.
    for group in new_groups:
        path_id = group["path_id"]
        errors = assigned_errors[path_id]
        p90 = float(np.quantile(errors, .90)) if errors else None
        group["fit_to_yellow"] = {
            "assigned_count": len(errors),
            "assigned_median_px": float(np.median(errors)) if errors else None,
            "assigned_p90_px": p90,
            "original_group_full_denominator": original_count[path_id],
            "original_group_capped_pitch_loss_mean":
                original_capped_loss[path_id] / original_count[path_id]
                if original_count[path_id] else None,
            "meaning": "fit to prior yellow candidates only; no identity ground truth"}
    for curve in curves:
        path_id = curve["path_id"]
        curve["fit_to_yellow"] = group_by_id[path_id]["fit_to_yellow"]
        p90 = curve["fit_to_yellow"]["assigned_p90_px"]
        if p90 is None or p90 > .25 * pitch:
            curve["quality"] = "ambiguous_fit_to_yellow"
    inferred_mask[observed_mask > 0] = 0
    smooth_mask[roi == 0] = 0
    inferred_mask[roi == 0] = 0
    observed_count = len(yellow_points)
    diagnostics = {
        "status": "completed", "variant": variant,
        "field_diagnostics": field.get("diagnostics", {}),
        "parameters": PARAMETERS.copy(),
        "c3_anchor_count": len(anchors), "path_side_count": len(geometry),
        "curve_segment_count": len(curves),
        "yellow_input_count": observed_count,
        "assigned_count": observed_count - sum(unassigned_reason.values()),
        "unassigned_count": sum(unassigned_reason.values()),
        "unassigned_reasons": dict(unassigned_reason),
        "reassigned_c3_group_count": reassigned,
        "field_and_yellow_support": dict(support_totals),
        "global_order_guard": global_order_diagnostics,
        "yellow_full_denominator": {
            "denominator": observed_count,
            "predicted_same_x_count": len(nearest_distances),
            "missing_same_x_count": observed_count - len(nearest_distances),
            "hit_within_quarter_pitch_count": full_denominator_hit,
            "hit_within_quarter_pitch_fraction":
                full_denominator_hit / observed_count if observed_count else None,
            "capped_pitch_loss_mean":
                full_denominator_loss / observed_count if observed_count else None,
            "nearest_distance_p50_px": float(np.median(nearest_distances))
                if nearest_distances else None,
            "nearest_distance_p90_px": float(np.quantile(nearest_distances, .90))
                if nearest_distances else None},
        "path_diagnostics": path_diagnostics,
        "rejected_paths": rejected_paths,
        "identity_scope": "visual_groove_candidate_only_not_physical_rope_identity"}
    return {"observed_lines": observed_lines, "observed_mask": observed_mask,
            "new_groups": new_groups, "point_assignments": assignments,
            "fit": {"model": {"type": "shared_quartic_plus_individual_cubic",
                              "c3_shared_shape_only_for_interface_seed": shared,
                              "image_guidance": "coherent_2.5_pitch_window_pseudo_points"
                              if variant == "image" else "off_geometry_control"},
                    "curves": curves, "smooth_mask": smooth_mask,
                    "diagnostics": diagnostics},
            "inferred_mask": inferred_mask, "diagnostics": diagnostics}


__all__ = ["PARAMETERS", "fit_groove_paths"]
