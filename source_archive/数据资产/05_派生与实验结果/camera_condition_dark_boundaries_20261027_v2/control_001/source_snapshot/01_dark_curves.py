"""D2：暗卷面独立黄点的单调归组与局部可变曲率平滑样条。

输入仅含本图各列的原图候选；不读旧 C3、评价标签、设备或文件。
黄点只表示输入中通过观测模块检验的像素。拟合曲线与短缺口另列，
没有足够证据的长缺口保持空白；视觉边界不证明材料身份。
"""
from __future__ import annotations

from collections import Counter
import math

import cv2
import numpy as np


def _validate(proposals: dict, dark_mask: np.ndarray, pitch_px: float
              ) -> tuple[list[list[dict]], list[int], float, list[dict]]:
    if (not isinstance(dark_mask, np.ndarray) or dark_mask.ndim != 2 or
            dark_mask.dtype != np.uint8 or
            np.any((dark_mask != 0) & (dark_mask != 255))):
        raise ValueError("dark_mask must be HxW uint8 0/255")
    pitch = float(pitch_px)
    if not math.isfinite(pitch) or not 8 <= pitch <= 70:
        raise ValueError("pitch_px must be finite in [8,70]")
    if not isinstance(proposals, dict):
        raise ValueError("proposals must contain columns and xs")
    columns, xs = proposals.get("columns"), proposals.get("xs")
    if not isinstance(columns, (list, tuple)) or not isinstance(xs, (list, tuple, np.ndarray)):
        raise ValueError("proposals.columns and proposals.xs required")
    if len(columns) != len(xs):
        raise ValueError("columns and xs lengths differ")
    xs = [int(x) for x in xs]
    if any(xs[i] >= xs[i + 1] for i in range(len(xs) - 1)):
        raise ValueError("xs must increase strictly")
    h, w = dark_mask.shape
    cleaned: list[list[dict]] = []
    rejected: list[dict] = []
    for ci, column in enumerate(columns):
        if not isinstance(column, (list, tuple)):
            raise ValueError("each proposal column must be a list")
        candidates = []
        for pi, original in enumerate(column):
            if not isinstance(original, dict):
                raise ValueError("candidate must be a dictionary")
            try:
                x, y = int(original["x"]), float(original["y"])
                score = float(original.get("score", 1.))
                slope = float(original.get("slope", original.get("slope_dy_dx", 0.)))
                local_pitch = float(original.get("pitch", pitch))
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"invalid candidate at column {ci}, index {pi}") from error
            if not all(math.isfinite(v) for v in (y, score, slope, local_pitch)):
                raise ValueError("candidate score, slope, pitch must be finite")
            ref = {"column_index": ci, "candidate_index": pi}
            pixel_y = int(round(y))
            if (x != xs[ci] or not 0 <= x < w or not 0 <= pixel_y < h or
                    dark_mask[pixel_y, x] == 0):
                rejected.append({"ref": ref, "xy": [x, y],
                                 "reason": "outside_dark_roi_or_wrong_column"})
                continue
            candidates.append({"x": x, "y": y, "score": max(.001, score),
                               "slope": float(np.clip(slope, -1.2, 1.2)),
                               "pitch": float(np.clip(local_pitch, .55 * pitch, 1.5 * pitch)),
                               "ref": ref})
        # 单列过近响应通常是同根细股纹或重复检测，保留较强的一个，
        # 弱者仍在 rejected 中可追溯，不被悄悄算作另一条整绳边界。
        kept = []
        for item in sorted(candidates, key=lambda p: p["score"], reverse=True):
            rival = next((q for q in kept if abs(q["y"] - item["y"]) <
                          .42 * min(q["pitch"], item["pitch"])), None)
            if rival is not None:
                rejected.append({"ref": item["ref"], "xy": [item["x"], item["y"]],
                                 "reason": "same_column_sub_pitch_duplicate",
                                 "kept_ref": rival["ref"]})
            else:
                kept.append(item)
        cleaned.append(sorted(kept, key=lambda p: p["y"]))
    return cleaned, xs, pitch, rejected


def _track_slope(points: list[dict]) -> float:
    recent = points[-min(5, len(points)):]
    if len(recent) >= 2 and recent[-1]["x"] > recent[0]["x"]:
        measured = ((recent[-1]["y"] - recent[0]["y"]) /
                    (recent[-1]["x"] - recent[0]["x"]))
        return float(np.clip(.75 * measured + .25 * recent[-1]["slope"], -1.2, 1.2))
    return float(np.clip(points[-1]["slope"], -1.2, 1.2))


def _ordered_match(active: list[tuple[int, float, float]], candidates: list[dict],
                   tracks: list[dict], pitch: float) -> list[tuple[int, int]]:
    """单调一对一动态匹配：同一列不能让两条轨迹交换上下次序。"""
    n, m = len(active), len(candidates)
    if n == 0 or m == 0:
        return []
    reward = np.full((n, m), -np.inf, float)
    for i, (ti, predicted, slope) in enumerate(active):
        last = tracks[ti]["points"][-1]
        for j, candidate in enumerate(candidates):
            spacing = min(pitch, last["pitch"], candidate["pitch"])
            distance = abs(candidate["y"] - predicted)
            if distance > max(2.5, .48 * spacing):
                continue
            slope_difference = abs(candidate["slope"] - slope)
            if slope_difference > .75:
                continue
            cost = (distance / spacing + .18 * slope_difference +
                    .08 * max(0, candidate["ref"]["column_index"] -
                              tracks[ti]["last_ci"] - 1))
            reward[i, j] = 1.5 - cost
    dp = np.zeros((n + 1, m + 1), float)
    action = np.zeros((n + 1, m + 1), np.uint8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if dp[i - 1, j] >= dp[i, j - 1]:
                dp[i, j], action[i, j] = dp[i - 1, j], 1
            else:
                dp[i, j], action[i, j] = dp[i, j - 1], 2
            if math.isfinite(reward[i - 1, j - 1]):
                candidate_value = dp[i - 1, j - 1] + reward[i - 1, j - 1]
                if candidate_value > dp[i, j]:
                    dp[i, j], action[i, j] = candidate_value, 3
    matched = []
    i, j = n, m
    while i > 0 and j > 0:
        move = int(action[i, j])
        if move == 3:
            matched.append((active[i - 1][0], j - 1))
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    return list(reversed(matched))


def _make_tracks(columns: list[list[dict]], xs: list[int], pitch: float
                 ) -> list[dict]:
    tracks: list[dict] = []
    for ci, candidates in enumerate(columns):
        active = []
        for ti, track in enumerate(tracks):
            if ci - track["last_ci"] > 3:
                continue
            slope = _track_slope(track["points"])
            last = track["points"][-1]
            predicted = last["y"] + slope * (xs[ci] - last["x"])
            active.append((ti, predicted, slope))
        # 保留上一可见列的上下身份顺序。若外推线互相穿越，宁可断轨，
        # 不能因按穿越后的预测 y 重排而把相邻边界悄悄交换编号。
        active.sort(key=lambda item: tracks[item[0]]["points"][-1]["y"])
        matched = _ordered_match(active, candidates, tracks, pitch)
        used = set()
        for ti, cj in matched:
            tracks[ti]["points"].append(candidates[cj])
            tracks[ti]["last_ci"] = ci
            used.add(cj)
        for cj, candidate in enumerate(candidates):
            if cj not in used:
                tracks.append({"points": [candidate], "last_ci": ci})
    return tracks


def _merge_fragments(tracks: list[dict], pitch: float
                     ) -> tuple[list[dict], list[dict]]:
    """只合并短空档且两端斜率、行序唯一一致的碎段。"""
    parents = list(range(len(tracks)))
    members = {i: [i] for i in range(len(tracks))}
    links = []

    def root(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    proposals = []
    for i, first in enumerate(tracks):
        aa = first["points"]
        if len(aa) < 2:
            continue
        for j, second in enumerate(tracks):
            if i == j or len(second["points"]) < 2:
                continue
            bb = second["points"]
            gap = bb[0]["x"] - aa[-1]["x"]
            if gap <= 0 or gap > 4 * pitch:
                continue
            sa, sb = _track_slope(aa), _track_slope(bb[:min(5, len(bb))])
            mismatch = abs(bb[0]["y"] - (aa[-1]["y"] + sa * gap))
            if mismatch > .38 * pitch or abs(sa - sb) > .35:
                continue
            # 中间若有另一行的可见黄点，则不凭近距强行跨行接续。
            blocked = False
            xmid = .5 * (aa[-1]["x"] + bb[0]["x"])
            target_y = aa[-1]["y"] + sa * (xmid - aa[-1]["x"])
            for k, other in enumerate(tracks):
                if k in (i, j):
                    continue
                pts = other["points"]
                nearby = [p for p in pts if abs(p["x"] - xmid) <= .7 * pitch]
                if nearby and min(abs(p["y"] - target_y) for p in nearby) < .48 * pitch:
                    blocked = True
                    break
            if not blocked:
                proposals.append((mismatch / pitch + .3 * abs(sa - sb) +
                                  .1 * gap / pitch, i, j, gap, mismatch))
    succ, pred = set(), set()
    for cost, i, j, gap, mismatch in sorted(proposals):
        if i in succ or j in pred or root(i) == root(j):
            continue
        ri, rj = root(i), root(j)
        parents[rj] = ri
        members[ri].extend(members.pop(rj))
        succ.add(i)
        pred.add(j)
        links.append({"from_track": i, "to_track": j, "gap_px": gap,
                      "endpoint_projection_error_px": mismatch,
                      "association_cost": cost})
    groups = []
    for indices in members.values():
        pts = [point for ti in indices for point in tracks[ti]["points"]]
        groups.append({"source_track_ids": sorted(indices),
                       "points": sorted(pts, key=lambda p: (p["x"], p["y"]))})
    groups.sort(key=lambda g: (g["points"][0]["x"], g["points"][0]["y"]))
    return groups, links


def _bspline_basis(x: np.ndarray, xlo: float, xhi: float,
                   control_count: int) -> tuple[np.ndarray, np.ndarray]:
    """无 SciPy 的三次 Cox–de Boor 基函数；端点重复四次。"""
    inner = np.linspace(0., 1., control_count - 2)[1:-1]
    knots = np.r_[np.zeros(4), inner, np.ones(4)]
    u = np.clip((np.asarray(x, float) - xlo) / max(xhi - xlo, 1.), 0., 1.)
    near_one = u >= 1.
    u = np.where(near_one, np.nextafter(1., 0.), u)
    basis = ((u[:, None] >= knots[:-1]) &
             (u[:, None] < knots[1:])).astype(float)
    for degree in range(1, 4):
        next_basis = np.zeros((len(u), basis.shape[1] - 1), float)
        for i in range(next_basis.shape[1]):
            left_den = knots[i + degree] - knots[i]
            right_den = knots[i + degree + 1] - knots[i + 1]
            if left_den > 0:
                next_basis[:, i] += ((u - knots[i]) / left_den) * basis[:, i]
            if right_den > 0:
                next_basis[:, i] += ((knots[i + degree + 1] - u) /
                                     right_den) * basis[:, i + 1]
        basis = next_basis
    basis = basis[:, :control_count]
    if np.any(near_one):
        basis[near_one] = 0.
        basis[near_one, -1] = 1.
    return basis, knots


def _fit_spline(points: list[dict], pitch: float) -> dict | None:
    x = np.asarray([p["x"] for p in points], float)
    y = np.asarray([p["y"] for p in points], float)
    if len(points) < 4 or len(np.unique(x)) < 4 or np.ptp(x) < .8 * pitch:
        return None
    xlo, xhi = float(x.min()), float(x.max())
    controls = int(np.clip(math.ceil((xhi - xlo) / (5 * pitch)) + 3,
                            4, min(24, max(4, len(points) // 2))))
    basis, knots = _bspline_basis(x, xlo, xhi, controls)
    d2 = np.diff(np.eye(controls), n=2, axis=0)
    penalty = float(np.clip(.6 * math.sqrt(len(points) / 20.), .5, 2.5))
    huber_delta = max(1., .13 * pitch)
    scores = np.asarray([p["score"] for p in points], float)
    score_middle = max(float(np.median(scores)), .001)
    base_weights = np.clip(scores / score_middle, .3, 2.)
    weights = base_weights.copy()
    regularizer = penalty * (d2.T @ d2) + 1e-7 * np.eye(controls)

    def solve(current: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        normal = basis.T @ (current[:, None] * basis) + regularizer
        rhs = basis.T @ (current * y)
        try:
            coefficients = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(normal, rhs, rcond=None)[0]
        return coefficients, normal, rhs

    for _ in range(7):
        coefficients, _, _ = solve(weights)
        residual = y - basis @ coefficients
        robust = np.minimum(1., huber_delta /
                            np.maximum(np.abs(residual), 1e-9))
        updated = base_weights * np.clip(robust, .08, 1.)
        if np.max(np.abs(updated - weights)) < 1e-3:
            weights = updated
            break
        weights = updated
    # 保存的 IRLS 最终正权重就是产生所记录系数的权重。
    coefficients, normal, rhs = solve(weights)
    residual = y - basis @ coefficients
    if not (np.isfinite(coefficients).all() and np.isfinite(residual).all() and
            np.isfinite(weights).all() and np.all(weights > 0)):
        return None
    return {"degree": 3, "xlo": xlo, "xhi": xhi,
            "knots_normalized": knots.tolist(),
            "control_y_px": coefficients.tolist(),
            "control_count": controls,
            "weights": weights.tolist(), "residuals_px": residual.tolist(),
            "mad_px": float(np.median(np.abs(residual))),
            "p90_px": float(np.quantile(np.abs(residual), .90)),
            "regularization": {"second_difference_lambda": penalty,
                               "huber_delta_px": huber_delta,
                               "score_weight_median": score_middle,
                               "base_weight_rule": "clip(score/median_score,0.3,2.0)",
                               "robust_weight_rule": "clip(min(1,huber_delta/abs(residual)),0.08,1)",
                               "objective": "sum_i weight_i*(y_i-B_i@control)^2 + lambda*sum_j(diff2(control)_j)^2 + 1e-7*sum_j(control_j^2)",
                               "normal_equation_max_abs_residual":
                                   float(np.max(np.abs(normal @ coefficients - rhs)))}}


def _sample_group(fitted: dict, points: list[dict], dark: np.ndarray,
                  pitch: float) -> tuple[list[list[int]], list[str], dict]:
    """观测域内采样；短缺口标 inferred_short_gap，长缺口留白。"""
    xlo, xhi = fitted["xlo"], fitted["xhi"]
    xs = np.arange(max(0, int(math.ceil(xlo))),
                   min(dark.shape[1] - 1, int(math.floor(xhi))) + 1)
    if not len(xs):
        return [], [], {"long_gap_count": 0, "short_gap_count": 0}
    basis, _ = _bspline_basis(xs, xlo, xhi, fitted["control_count"])
    yy = np.rint(basis @ np.asarray(fitted["control_y_px"])).astype(int)
    observed_x = sorted({p["x"] for p in points})
    short_gap_limit = 4. * pitch
    regular_step_limit = 1.25 * pitch
    gap_intervals = [(a, b) for a, b in zip(observed_x[:-1], observed_x[1:])
                     if b - a > regular_step_limit]
    long_gap_count = sum(b - a > short_gap_limit for a, b in gap_intervals)
    sampled: list[list[int]] = []
    states: list[str] = []
    for x, y in zip(xs, yy):
        if not 0 <= y < dark.shape[0] or dark[y, x] == 0:
            continue
        interval = next(((a, b) for a, b in gap_intervals if a < x < b), None)
        if interval is not None and interval[1] - interval[0] > short_gap_limit:
            continue
        sampled.append([int(x), int(y)])
        states.append("inferred_short_gap" if interval is not None
                      else "supported_fit")
    return sampled, states, {"long_gap_count": long_gap_count,
                             "short_gap_count": len(gap_intervals) - long_gap_count,
                             "maximum_drawable_gap_px": short_gap_limit}


def _curve_conflict(first: dict, second: dict, pitch: float) -> dict | None:
    a = {x: y for x, y in first["sampled_points_xy"]}
    b = {x: y for x, y in second["sampled_points_xy"]}
    common = sorted(a.keys() & b.keys())
    if len(common) < max(6, int(round(.5 * pitch))):
        return None
    signed = np.asarray([a[x] - b[x] for x in common], float)
    if (np.min(signed) < -.1 * pitch and np.max(signed) > .1 * pitch):
        return {"reason": "postfit_curves_cross", "common_columns": len(common),
                "minimum_abs_separation_px": float(np.min(np.abs(signed)))}
    minimum = float(np.min(np.abs(signed)))
    if minimum < .45 * pitch:
        return {"reason": "postfit_neighbor_spacing_too_small",
                "common_columns": len(common),
                "minimum_abs_separation_px": minimum}
    return None


def group_and_fit(proposals: dict, dark_mask: np.ndarray,
                  pitch_px: float) -> dict:
    """归组原图候选并拟合局部可变曲率三次样条。

    线号仅是本图视觉边界组号。所有输入点都进入一个组或明确的候选拒识
    记录；只有短且有两端支持的缺口可产生 inferred_short_gap。
    """
    columns, xs, pitch, rejected_measurements = _validate(
        proposals, dark_mask, pitch_px)
    observed_mask = np.zeros(dark_mask.shape, np.uint8)
    smooth_mask = np.zeros(dark_mask.shape, np.uint8)
    inferred_mask = np.zeros(dark_mask.shape, np.uint8)
    empty_fit = {"model": {"type": "local_cubic_B_spline",
                            "degree": 3,
                            "identity_scope": "single_image_visual_boundary_only"},
                 "curves": [], "groups": [], "smooth_mask": smooth_mask,
                 "inferred_mask": inferred_mask, "rejected": [],
                 "diagnostics": {}}
    if not xs or not any(columns):
        empty_fit["diagnostics"] = {"status": "no_valid_observation_candidates",
                                    "input_candidate_count": sum(map(len, columns)),
                                    "accepted_curve_count": 0}
        return {"observed_lines": [], "observed_mask": observed_mask,
                "inferred_mask": inferred_mask, "fit": empty_fit,
                "rejected_measurements": rejected_measurements,
                "diagnostics": {"status": "no_valid_observation_candidates",
                                "input_columns": len(xs),
                                "rejected_measurement_count": len(rejected_measurements)}}
    tracks = _make_tracks(columns, xs, pitch)
    grouped, associations = _merge_fragments(tracks, pitch)
    groups: list[dict] = []
    observed_lines: list[dict] = []
    provisional: list[dict] = []
    rejected: list[dict] = []
    reject_reasons: Counter[str] = Counter()

    for number, associated in enumerate(grouped, 1):
        points = associated["points"]
        gid = f"dark_group_{number:04d}"
        xy = [[int(p["x"]), int(round(p["y"]))] for p in points]
        subpixel_xy = [[int(p["x"]), float(p["y"])] for p in points]
        refs = [p["ref"] for p in points]
        score = [float(p["score"]) for p in points]
        span = int(points[-1]["x"] - points[0]["x"])
        group = {"group_id": gid, "status": "rejected",
                 "source_track_ids": associated["source_track_ids"],
                 "observed_points_xy": xy, "point_refs": refs,
                 "measured_points_xy_subpixel": subpixel_xy,
                 "point_scores": score,
                 "point_local_slope": [float(p["slope"]) for p in points],
                 "point_local_pitch_px": [float(p["pitch"]) for p in points],
                 "observed_point_count": len(points), "observed_span_px": span,
                 "fit_weights": None, "fit_residuals_px": None,
                 "fit_regularization": None, "model": None,
                 "curve_id": None, "reason": None,
                 "postfit_conflicts": []}
        if len(points) < 4 or len({p["x"] for p in points}) < 4 or span < .8 * pitch:
            reason = "short_observation_below_four_columns_or_point_eight_pitch_span"
            group["reason"] = reason
            rejected.append({"group_id": gid, "reason": reason,
                             "observed_point_count": len(points),
                             "point_refs": refs})
            reject_reasons[reason] += 1
            groups.append(group)
            continue
        # 短但证据一致的组也保留为黄色，不再套用旧 B1 的 76px/12点门。
        observed_lines.append({"line_id": gid, "observed_points_xy": xy,
                               "point_scores": score,
                               "source_track_ids": associated["source_track_ids"],
                               "point_count": len(points), "span_px": span,
                               "identity_scope": "visual_boundary_not_material_identity"})
        for x, y in xy:
            cv2.circle(observed_mask, (x, y), 1, 255, -1, cv2.LINE_8)
        fitted = _fit_spline(points, pitch)
        if fitted is None:
            reason = "spline_fit_unidentifiable_or_nonfinite"
        elif (fitted["mad_px"] > .25 * pitch or
              fitted["p90_px"] > .55 * pitch):
            reason = "all_observations_disagree_with_smooth_spline"
        else:
            reason = None
        if fitted is not None:
            group["fit_weights"] = fitted["weights"]
            group["fit_residuals_px"] = fitted["residuals_px"]
            group["fit_regularization"] = fitted["regularization"]
            group["model"] = {key: fitted[key] for key in
                              ("degree", "xlo", "xhi", "knots_normalized",
                               "control_y_px", "control_count")}
            group["fit_mad_px"] = fitted["mad_px"]
            group["fit_p90_px"] = fitted["p90_px"]
        if reason is not None:
            group["reason"] = reason
            rejected.append({"group_id": gid, "reason": reason,
                             "observed_point_count": len(points),
                             "point_refs": refs})
            reject_reasons[reason] += 1
            groups.append(group)
            continue
        sampled, states, gaps = _sample_group(fitted, points, dark_mask > 0, pitch)
        if not sampled:
            reason = "no_drawable_dark_roi_spline_pixels"
            group["reason"] = reason
            rejected.append({"group_id": gid, "reason": reason,
                             "observed_point_count": len(points),
                             "point_refs": refs})
            reject_reasons[reason] += 1
            groups.append(group)
            continue
        model = group["model"]
        # 曲率随局部 x 变化，约束样条斜率而非强制整条抛物线。
        dense_x = np.arange(int(math.ceil(fitted["xlo"])),
                            int(math.floor(fitted["xhi"])) + 1)
        dense_basis, _ = _bspline_basis(dense_x, fitted["xlo"],
                                        fitted["xhi"], fitted["control_count"])
        dense_y = dense_basis @ np.asarray(fitted["control_y_px"])
        slopes = np.diff(dense_y)
        if len(slopes) and np.max(np.abs(slopes)) > 1.25:
            reason = "spline_local_slope_exceeds_dark_boundary_budget"
            group["reason"] = reason
            rejected.append({"group_id": gid, "reason": reason,
                             "observed_point_count": len(points),
                             "point_refs": refs})
            reject_reasons[reason] += 1
            groups.append(group)
            continue
        curve_id = f"dark_curve_{number:04d}"
        curve = {"curve_id": curve_id, "group_id": gid,
                 "model": model, "source_track_ids": associated["source_track_ids"],
                 "observed_span_x_px": [int(points[0]["x"]), int(points[-1]["x"])],
                 "sampled_points_xy": sampled, "point_states": states,
                 "observed_point_count": len(points),
                 "observed_fit_mad_px": fitted["mad_px"],
                 "observed_fit_p90_px": fitted["p90_px"],
                 "gap_diagnostics": gaps,
                 "interpretation": "smooth visual boundary; inferred_short_gap is geometry only"}
        quality = (len(points) * max(1, span) * max(.1, float(np.median(score))) /
                   (1. + fitted["p90_px"]))
        provisional.append({"curve": curve, "group": group, "quality": quality})
        groups.append(group)

    # 共域排序和间距校验放在全部曲线拟合之后。遇到冲突，保留观测黄点，
    # 仅拒绝证据较弱整组的平滑模型，不裁一段曲线来隐藏交叉。
    accepted: list[dict] = []
    for item in sorted(provisional, key=lambda entry: -entry["quality"]):
        conflicts = []
        for stronger in accepted:
            problem = _curve_conflict(item["curve"], stronger["curve"], pitch)
            if problem is not None:
                conflicts.append({"other_group_id": stronger["group"]["group_id"],
                                  **problem})
        if conflicts:
            group = item["group"]
            group["reason"] = "postfit_crossing_or_spacing_conflict"
            group["postfit_conflicts"] = conflicts
            rejected.append({"group_id": group["group_id"],
                             "reason": group["reason"],
                             "observed_point_count": group["observed_point_count"],
                             "conflicts": conflicts,
                             "point_refs": group["point_refs"]})
            reject_reasons[group["reason"]] += 1
            continue
        item["group"]["status"] = "accepted"
        item["group"]["curve_id"] = item["curve"]["curve_id"]
        accepted.append(item)

    curves = sorted((item["curve"] for item in accepted),
                    key=lambda curve: curve["curve_id"])
    for curve in curves:
        for (x, y), state in zip(curve["sampled_points_xy"],
                                 curve["point_states"]):
            cv2.circle(smooth_mask, (x, y), 1, 255, -1, cv2.LINE_8)
            if state == "inferred_short_gap":
                cv2.circle(inferred_mask, (x, y), 1, 255, -1, cv2.LINE_8)
    observed_mask[dark_mask == 0] = 0
    smooth_mask[dark_mask == 0] = 0
    inferred_mask[dark_mask == 0] = 0
    diagnostics = {
        "status": "completed", "input_columns": len(xs),
        "input_candidate_count": sum(len(c) for c in proposals["columns"]),
        "validated_candidate_count": sum(map(len, columns)),
        "rejected_measurement_count": len(rejected_measurements),
        "raw_track_count": len(tracks), "short_gap_association_count": len(associations),
        "stable_observed_line_count": len(observed_lines),
        "accepted_curve_count": len(curves),
        "rejected_group_count": len(rejected),
        "rejected_group_reasons": dict(reject_reasons),
        "observed_pixel_count": int(np.count_nonzero(observed_mask)),
        "smooth_pixel_count": int(np.count_nonzero(smooth_mask)),
        "inferred_short_gap_pixel_count": int(np.count_nonzero(inferred_mask)),
        "point_states": ["supported_fit", "inferred_short_gap"],
        "spline_model": "cubic_clamped_B_spline_with_second_difference_penalty_and_positive_Huber_IRLS",
        "identity_scope": "single_image_visual_boundary_only",
        "long_gap_policy": "no curve samples inside observed x gaps over four local pitches"
    }
    fit = {"model": empty_fit["model"], "curves": curves,
           "groups": groups, "smooth_mask": smooth_mask,
           "inferred_mask": inferred_mask, "rejected": rejected,
           "diagnostics": {**diagnostics,
                           "association_edges": associations}}
    return {"observed_lines": observed_lines,
            "observed_mask": observed_mask,
            "inferred_mask": inferred_mask,
            "fit": fit,
            "rejected_measurements": rejected_measurements,
            "diagnostics": diagnostics}
