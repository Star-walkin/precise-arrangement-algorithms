"""C2：只在两段真实观测之间补局部抛物线，保留 C1-002 的可见段拟合。

图像 y 向下。补线使用 y(x)=两端弦线+a(x-L)(x-R)，a>=0；
这是局部近竖直轴抛物线，不是完整椭圆，也不证明同根材料身份。
每个 visible/gap 段单独记录模型与绘制范围；不会把缺口模型外推到可见区域。
"""
from __future__ import annotations

from collections import Counter
import math

import numpy as np

import base_conic as base


def _visible_intervals(points: np.ndarray, pitch: float) -> list[tuple[int, int]]:
    """只在有连续黄点支撑的列绘制旧模型，片段内真空档也不冒充观测。"""
    if len(points) == 0:
        return []
    steps = np.diff(points[:, 0])
    positive = steps[steps > 0]
    median_step = float(np.median(positive)) if len(positive) else 1.
    limit = max(2.5 * median_step, .8 * pitch)
    starts = [0] + [i + 1 for i, step in enumerate(steps) if step > limit]
    ends = starts[1:] + [len(points)]
    return [(int(math.ceil(points[start, 0])), int(math.floor(points[end - 1, 0])))
            for start, end in zip(starts, ends)]


def _endpoint_local(points: np.ndarray, side: str, pitch: float) -> tuple[float, float, float | None, float]:
    """最近几个黄点估计端头高度与切线，回归端点最多修正少量像素。"""
    end = points[-min(9, len(points)):] if side == "left" else points[:min(9, len(points))]
    x0, raw_y = map(float, points[-1] if side == "left" else points[0])
    if len(end) < 3 or np.ptp(end[:, 0]) < 5.:
        return x0, raw_y, None, 0.
    shifted_x = end[:, 0] - x0
    slope, intercept = np.polyfit(shifted_x, end[:, 1], 1)
    prediction = intercept + slope * shifted_x
    median_residual = float(np.median(np.abs(end[:, 1] - prediction)))
    endpoint_y = float(np.clip(intercept, raw_y - max(2., .12 * pitch),
                               raw_y + max(2., .12 * pitch)))
    return x0, endpoint_y, float(slope), median_residual


def _gap_model(left: np.ndarray, right: np.ndarray, shared_a: float,
               pitch: float) -> tuple[dict | None, str]:
    """只用左右黄点及共享曲率建模，绝不看缺口内的隐藏参考。"""
    xL, yL, slopeL, localL = _endpoint_local(left, "left", pitch)
    xR, yR, slopeR, localR = _endpoint_local(right, "right", pitch)
    gap = xR - xL
    if gap <= 0 or shared_a < 0:
        return None, "invalid_gap_or_shared_curvature"
    if max(localL, localR) > .22 * pitch:
        return None, "noisy_endpoint_neighborhood"
    chord = (yR - yL) / gap
    # 两边短端头的切线只能微调共享曲率，不能从噪声短弧独立估计椭圆。
    a = shared_a
    if shared_a > 0 and slopeL is not None and slopeR is not None:
        local_a = .5 * ((chord - slopeL) / gap + (slopeR - chord) / gap)
        a = float(np.clip(.80 * shared_a + .20 * local_a,
                          .70 * shared_a, 1.30 * shared_a))
    slope_model_L = chord - a * gap
    slope_model_R = chord + a * gap
    if max(abs(slope_model_L), abs(slope_model_R)) > .19:
        return None, "implausible_gap_tangent"
    if ((slopeL is not None and abs(slopeL - slope_model_L) > .15) or
            (slopeR is not None and abs(slopeR - slope_model_R) > .15)):
        return None, "endpoint_tangent_disagreement"
    # 只以端头邻域检验局部延续，不拿长片段整体截距拖动缺口位置。
    tails = np.concatenate((left[-min(9, len(left)):], right[:min(9, len(right))]))
    prediction = (yL + chord * (tails[:, 0] - xL) +
                  a * (tails[:, 0] - xL) * (tails[:, 0] - xR))
    residual = np.abs(tails[:, 1] - prediction)
    mad = float(np.median(residual))
    p90 = float(np.quantile(residual, .90))
    if mad > .24 * pitch or p90 > .45 * pitch:
        return None, "endpoint_visible_residual"
    center = .5 * (xL + xR)
    scale = max(1., .5 * gap)
    b = scale * (chord + a * (2 * center - xL - xR))
    c = a * scale * scale
    k = yL + chord * (center - xL) + a * (center - xL) * (center - xR)
    return {"type": "vertical_axis_parabola" if a > 0 else "near_flat_line",
            "a_px_per_px2": float(a), "normalized_b": float(b),
            "normalized_c": float(c), "normalized_k": float(k),
            "u_center_x_px": float(center), "u_scale_x_px": float(scale),
            "endpoint_xy": [[xL, yL], [xR, yR]],
            "endpoint_slopes_px_per_px": [slopeL, slopeR],
            "median_visible_residual_px": mad, "p90_visible_residual_px": p90}, "ok"


def _evaluate(model: dict, x: float) -> float:
    u = (x - model["u_center_x_px"]) / model["u_scale_x_px"]
    return float(model["normalized_k"] + model["normalized_b"] * u +
                 model["normalized_c"] * u * u)


def fit_conic_family(lines: list[dict], pitch_px: float, roi_mask: np.ndarray,
                     exclusion_mask: np.ndarray, parameters=None) -> dict:
    """返回仅可见段旧拟合与仅内部缺口新拟合，及两者 0/255 联合光栅。"""
    pitch = float(pitch_px)
    if not math.isfinite(pitch) or not 8 <= pitch <= 70:
        raise ValueError("invalid pitch_px")
    roi = base._binary_mask(roi_mask, "roi_mask")
    exclusion = base._binary_mask(exclusion_mask, "exclusion_mask")
    if roi.shape != exclusion.shape or np.any(exclusion & ~roi):
        raise ValueError("exclusion_mask must be inside ROI")
    if not isinstance(lines, list):
        raise ValueError("lines must be a list")
    height, width = roi.shape
    fragments = [base._points(line, width, height) for line in lines]
    ids = [line.get("line_id", f"fragment_{i:04d}") for i, line in enumerate(lines)]
    if len(set(ids)) != len(ids):
        raise ValueError("line_id must be unique")
    fragment_by_id = dict(zip(ids, fragments))
    baseline = base.fit_conic_family(lines, pitch, roi_mask, exclusion_mask, parameters)
    smooth = np.zeros((height, width), np.uint8)
    result = {"model": {"branch": "C2_local_gap_conic", "shared_visible_model": baseline["model"],
                        "meaning": "separate visible and internal-gap models; no whole-curve conic identity"},
              "curves": [], "smooth_mask": smooth, "rejected": [],
              "diagnostics": {"status": "ok", "base_status": baseline["diagnostics"].get("status")}}
    if baseline["model"] is None or baseline["diagnostics"].get("status") != "ok":
        result["diagnostics"]["status"] = "shared_visible_shape_unidentifiable"
        return result
    reasons: Counter[str] = Counter()
    # C1-002 模型只能在各片段实际可见 span 内绘制。即使它把两片段关联，
    # 原模型的跨缺口像素也在这里移除，随后由独立的端点局部模型填补。
    for old in baseline["curves"]:
        for fragment_id in old["source_fragment_ids"]:
            source = fragment_by_id.get(fragment_id)
            if source is None:
                continue
            for part, (lo, hi) in enumerate(_visible_intervals(source, pitch), 1):
                sampled = [[x, y] for x, y in old["sampled_points_xy"] if lo <= x <= hi]
                if not sampled:
                    continue
                model = dict(old["model"])
                model["drawable_span_x_px"] = [lo, hi]
                curve = {"curve_id": f"visible_{old['curve_id']}_{fragment_id}_{part}",
                         "status": "visible_base_conic_segment",
                         "source_fragment_ids": [fragment_id],
                         "model": model, "observed_span_x_px": [lo, hi],
                         "drawable_span_x_px": [lo, hi], "sampled_points_xy": sampled,
                         "support": {**old["support"], "support_level": "visible_yellow_fragment",
                                     "visible_source_fragment_id": fragment_id},
                         "interpretation": "C1-002 smooth model clipped to directly visible fragment span"}
                result["curves"].append(curve)
                for x, y in sampled:
                    smooth[y, x] = 255
    visible_curves = list(result["curves"])
    model_info = baseline["model"]
    center = float(model_info["center_x_px"])
    scale = float(model_info["scale_x_px"])
    shared = np.array([model_info["b_px_per_u"], model_info["c_px_per_u2"]], float)
    shared_a = max(0., float(model_info["a_px_per_px2"]))
    offsets, residuals = [], []
    eligible = []
    for points in fragments:
        if len(points) < 2 or np.ptp(points[:, 0]) < 4.:
            offsets.append(float("nan"))
            residuals.append(float("inf"))
            eligible.append(False)
            continue
        values = points[:, 1] - base._shape(points[:, 0], shared, center, scale)
        offset = float(np.median(values))
        mad = float(np.median(np.abs(values - offset)))
        offsets.append(offset)
        residuals.append(mad)
        # 短片段可作端头，但整体形态严重不符的长片段不参与配对。
        eligible.append(mad <= .35 * pitch)
    spacing, spacing_info = base._estimate_spacing(
        [offsets[i] for i in range(len(lines)) if eligible[i] and len(fragments[i]) >= 8], pitch)
    maximum_gap = min(25. * pitch, 650.)
    candidates = []
    for i, left in enumerate(fragments):
        if not eligible[i]:
            continue
        for j, right in enumerate(fragments):
            if i == j or not eligible[j]:
                continue
            x_gap = float(right[0, 0] - left[-1, 0])
            if not max(3., .25 * pitch) <= x_gap <= maximum_gap:
                continue
            ltail, rhead = left[-min(5, len(left)):], right[:min(5, len(right))]
            llocal = float(np.median(ltail[:, 1] - base._shape(ltail[:, 0], shared, center, scale)))
            rlocal = float(np.median(rhead[:, 1] - base._shape(rhead[:, 0], shared, center, scale)))
            if abs(llocal - rlocal) > .62 * pitch:
                reasons["different_course_endpoint_offset"] += 1
                continue
            local_model, why = _gap_model(left, right, shared_a, pitch)
            if local_model is None:
                reasons[why] += 1
                continue
            midpoint = .5 * (llocal + rlocal)
            neighbors = base._course_neighbors(
                i, midpoint, fragments, offsets, residuals,
                (float(left[-1, 0]), float(right[0, 0])), spacing, pitch)
            needed = 2 if len(left) >= 8 and len(right) >= 8 else 3
            if neighbors < needed:
                reasons["insufficient_neighbor_courses"] += 1
                continue
            support = min(len(left), 80) + min(len(right), 80)
            score = (abs(llocal - rlocal) / pitch + .25 * x_gap / maximum_gap
                     + local_model["median_visible_residual_px"] / pitch
                     - .003 * support - .025 * neighbors)
            candidates.append((score, i, j, x_gap, neighbors, llocal, rlocal, local_model))
    candidates.sort(key=lambda item: item[0])
    successor, predecessor, selected_offsets = {}, {}, {}
    selected_gap_curves = []
    for score, i, j, x_gap, neighbors, llocal, rlocal, model in candidates:
        left, right = fragments[i], fragments[j]
        if i in successor or j in predecessor:
            reasons["one_to_one_conflict"] += 1
            continue
        order_bad = False
        for (k, m), (other_l, other_r) in selected_offsets.items():
            left_sep = llocal - other_l
            right_sep = rlocal - other_r
            if left_sep * right_sep < 0 or (
                    max(abs(left_sep), abs(right_sep)) >= .65 * pitch and
                    min(abs(left_sep), abs(right_sep)) < .45 * pitch):
                order_bad = True
                break
        if order_bad:
            reasons["order_crossing_or_spacing_collapse"] += 1
            continue
        xL, xR = model["endpoint_xy"][0][0], model["endpoint_xy"][1][0]
        x0 = max(0, int(math.floor(xL)) + 1)
        x1 = min(width - 1, int(math.ceil(xR)) - 1)
        sampled = []
        for x in range(x0, x1 + 1):
            y = int(round(_evaluate(model, x)))
            if 0 <= y < height and roi[y, x] and not exclusion[y, x]:
                sampled.append([x, y])
        if len(sampled) < 3:
            reasons["gap_has_too_few_drawable_pixels"] += 1
            continue
        # 与已有可见段及先接受的补线比较，同列太近或相交的候选留白。
        conflict = False
        candidate_by_x = {x: y for x, y in sampled}
        for previous in visible_curves + selected_gap_curves:
            if ids[i] in previous["source_fragment_ids"] or ids[j] in previous["source_fragment_ids"]:
                continue
            previous_by_x = {x: y for x, y in previous["sampled_points_xy"]}
            common = candidate_by_x.keys() & previous_by_x.keys()
            if len(common) < 2:
                continue
            separations = [candidate_by_x[x] - previous_by_x[x] for x in common]
            if (min(separations) < -1 and max(separations) > 1) or any(
                    abs(distance) < .45 * pitch for distance in separations):
                conflict = True
                break
        if conflict:
            reasons["gap_postfit_spacing_or_crossing"] += 1
            continue
        successor[i] = j
        predecessor[j] = i
        selected_offsets[(i, j)] = (llocal, rlocal)
        local_model = dict(model)
        local_model["drawable_span_x_px"] = [x0, x1]
        curve = {"curve_id": f"gap_{len(selected_gap_curves) + 1:04d}",
                 "status": "gap_endpoint_conic_geometric_inference",
                 "source_fragment_ids": [ids[i], ids[j]],
                 "model": local_model,
                 "observed_span_x_px": [int(left[-1, 0]), int(right[0, 0])],
                 "drawable_span_x_px": [x0, x1], "sampled_points_xy": sampled,
                 "support": {"observed_fragment_count": 2,
                             "observed_point_count": int(len(left) + len(right)),
                             "neighbor_course_count": neighbors,
                             "median_residual_px": model["median_visible_residual_px"],
                             "p90_residual_px": model["p90_visible_residual_px"],
                             "geometric_gap_count": 1,
                             "support_level": "two_endpoints_plus_neighbor_courses"},
                 "interpretation": "local gap-only geometric prediction, not directly observed"}
        selected_gap_curves.append(curve)
        for x, y in sampled:
            smooth[y, x] = 255
    result["curves"].extend(selected_gap_curves)
    result["diagnostics"] = {"status": "ok", "base_status": baseline["diagnostics"].get("status"),
                             "base_visible_model_sha256": "58B6AF6D63816FC84BA7F0065BD6076019AFDBD1B42281C0769601C0C3BEA05E",
                             "fragment_count": len(lines), "eligible_endpoints": int(sum(eligible)),
                             "visible_segment_count": len(visible_curves),
                             "gap_candidate_count": len(candidates),
                             "accepted_gap_count": len(selected_gap_curves),
                             "accepted_connections": [{"from": ids[i], "to": ids[j],
                                                       "endpoint_offsets_px": [selected_offsets[(i, j)][0],
                                                                               selected_offsets[(i, j)][1]]}
                                                      for i, j in successor.items()],
                             "spacing": spacing_info, "rejection_counts": dict(reasons),
                             "identity_scope": "one image; no material identity",
                             "overexposure_policy": "only finite two-ended internal gaps; exclusion remains blank"}
    return result
