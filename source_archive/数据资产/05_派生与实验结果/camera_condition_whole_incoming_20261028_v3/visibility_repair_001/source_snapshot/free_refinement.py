"""A1 已观测自由直段的原图双边亚像素精修，纯离线。

只测 A1 已自动找到的卷面外直段；不寻找新绳、不使用卷面绳号、人工标签
或接触点。返回坐标仍是输入原图 (x, y)，宽度是图像光学边缘间距（px），
不能解释为实体绳径、毫米尺寸或三维接触。证据不够时原样返回 A1 几何。
"""
from __future__ import annotations

import copy
import math

import cv2
import numpy as np


# 参数在固定 38 张开放开发 QA 对照前锁定；更改须另开版本和运行记录。
PROFILE_STEP_PX = .5
SAMPLE_MAX = 240
SAMPLE_MIN = 14
SUPPORT_SPAN_MIN_PX = 100.
PROFILE_PROMINENCE_MIN = 17.
GRADIENT_MIN_PER_PX = 2.0
HALF_GRAD_AGREEMENT_MIN_PX = 1.5
WIDTH_RATIO_MIN = .55
WIDTH_RATIO_MAX = 1.85
INLIER_FRACTION_MIN = .45
ANGLE_CHANGE_MAX_DEG = 2.5
CENTER_SHIFT_MAX_WIDTH_FRACTION = .38


def _fallback(a1_free: dict, reason: str, diagnostics: dict | None = None) -> dict:
    """失败时保留所有 A1 数值；调用者能区分无证据与改进被接纳。"""
    result = copy.deepcopy(a1_free)
    result["refinement"] = {"status": "fallback_a1", "reason": reason,
                            "method": "paired_original_half_prominence_and_opposite_gradient_edges",
                            "a1_geometry_preserved": True,
                            "diagnostics": diagnostics or {}}
    return result


def _channel(image_bgr: np.ndarray, a1_free: dict) -> tuple[np.ndarray, str]:
    """只按 A1 事先给出的黄法兰证据决定原图通道，不看本轮拟合结果。"""
    old_kind = (a1_free.get("diagnostics", {})
                .get("selected_profile_refinement", {}).get("channel"))
    yellow_fraction = (a1_free.get("diagnostics", {})
                       .get("selected_candidate", {}).get("approach_evidence", {})
                       .get("bilateral_yellow_background_fraction", 0.))
    bgr = image_bgr.astype(np.float32)
    low = bgr.min(axis=2)
    if old_kind == "bilateral_yellow_free_span" or float(yellow_fraction or 0.) >= .6:
        return low * 255. / np.maximum(bgr.max(axis=2), 1.), "original_achromatic_min_over_max_on_A1_yellow_context"
    return low, "original_BGR_min_brightness"


def _refined_surface_direction(segment: np.ndarray, surface: np.ndarray,
                               direction: np.ndarray) -> tuple[list[float] | None, str,
                                                                list[list[float]], list[dict]]:
    """用精修直段的有限双向射线重新判断朝卷面方向，绝不称真实接触。"""
    length = min(1800., .45 * max(surface.shape))
    distances = np.arange(0., length + 8., 8.)
    h, w = surface.shape
    hits = []
    for end_index, ray in ((0, -direction), (1, direction)):
        xy = segment[end_index][None, :] + distances[:, None] * ray
        ix = np.rint(xy[:, 0]).astype(int)
        iy = np.rint(xy[:, 1]).astype(int)
        inside = np.flatnonzero((ix >= 0) & (ix < w) & (iy >= 0) & (iy < h))
        touching = inside[surface[iy[inside], ix[inside]] != 0]
        if len(touching):
            first = int(touching[0])
            hits.append({"end_index": end_index, "intersects_surface": True,
                         "first_hit_distance_px": float(distances[first]),
                         "first_hit_xy": [int(ix[first]), int(iy[first])]})
        else:
            hits.append({"end_index": end_index, "intersects_surface": False,
                         "first_hit_distance_px": None, "first_hit_xy": None})
    reached = [item["end_index"] for item in hits if item["intersects_surface"]]
    if len(reached) == 1:
        toward = -direction if reached[0] == 0 else direction
        return toward.tolist(), "unique_surface_ray", [toward.tolist()], hits
    alternatives = [direction.tolist(), (-direction).tolist()]
    status = "ambiguous_both_surface_rays" if len(reached) == 2 else "refined_line_ray_does_not_reach_surface"
    return None, status, alternatives, hits


def _quadratic_extremum(values: np.ndarray, index: int, step: float) -> float:
    """梯度峰三点抛物线细化；二阶导近零时回到离散峰。"""
    if index <= 0 or index >= len(values) - 1:
        return float(index * step)
    a, b, c = map(float, values[index - 1:index + 2])
    denominator = a - 2. * b + c
    if abs(denominator) < 1e-6:
        return float(index * step)
    substep = np.clip(.5 * (a - c) / denominator, -1., 1.)
    return float((index + substep) * step)


def _crossing(profile: np.ndarray, offsets: np.ndarray, peak_index: int,
              half: float, side: int) -> float | None:
    """从峰向外找半峰交点，每侧采用自己的背景基线。"""
    if side < 0:
        candidates = np.flatnonzero(profile[:peak_index] <= half)
        if not len(candidates):
            return None
        i = int(candidates[-1])
        p0, p1 = float(profile[i]), float(profile[i + 1])
        fraction = float(np.clip((half - p0) / max(p1 - p0, 1e-6), 0., 1.))
        return float(offsets[i] + fraction * PROFILE_STEP_PX)
    candidates = np.flatnonzero(profile[peak_index + 1:] <= half)
    if not len(candidates):
        return None
    i = int(candidates[0] + peak_index + 1)
    p0, p1 = float(profile[i - 1]), float(profile[i])
    fraction = float(np.clip((p0 - half) / max(p0 - p1, 1e-6), 0., 1.))
    return float(offsets[i - 1] + fraction * PROFILE_STEP_PX)


def _edge_from_gradient(profile: np.ndarray, offsets: np.ndarray,
                        half_edge: float, side: int,
                        width_hint: float) -> tuple[float, float] | None:
    """在半峰边附近找正确符号的最大梯度，拒绝绳股内部反向峰。"""
    gradient = np.gradient(profile, PROFILE_STEP_PX)
    allowance = max(2., width_hint * .18)
    window = np.flatnonzero(np.abs(offsets - half_edge) <= allowance)
    window = window[(window > 0) & (window < len(offsets) - 1)]
    if not len(window):
        return None
    signed = gradient if side < 0 else -gradient
    index = int(window[np.argmax(signed[window])])
    strength = float(signed[index])
    if strength < GRADIENT_MIN_PER_PX:
        return None
    location = float(offsets[0] + _quadratic_extremum(signed, index, PROFILE_STEP_PX))
    return location, strength


def _measure_pairs(image_bgr: np.ndarray, surface: np.ndarray, a1_free: dict,
                   anchors: np.ndarray, center0: np.ndarray,
                   direction: np.ndarray, normal: np.ndarray,
                   width_hint: float) -> tuple[np.ndarray, dict]:
    """每个 A1 原图支撑点独立测两侧交点；不绘制未观测的中间像素。"""
    channel, channel_kind = _channel(image_bgr, a1_free)
    radius = float(np.clip(max(24., width_hint * 2.0), 24., 80.))
    offsets = np.arange(-radius, radius + PROFILE_STEP_PX * .5,
                        PROFILE_STEP_PX, dtype=np.float32)
    xy = anchors[:, None, :] + offsets[None, :, None] * normal
    sampled = cv2.remap(channel, xy[:, :, 0].astype(np.float32),
                        xy[:, :, 1].astype(np.float32), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    profiles = cv2.GaussianBlur(sampled, (5, 1), .75)
    keep = []
    rejects: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejects[reason] = rejects.get(reason, 0) + 1

    min_flank = max(6., width_hint * .85)
    peak_window = max(3., width_hint * .24)
    for j, profile in enumerate(profiles):
        point = anchors[j]
        if not (1 <= point[0] < image_bgr.shape[1] - 1 and
                1 <= point[1] < image_bgr.shape[0] - 1):
            reject("anchor_near_image_border")
            continue
        cx, cy = np.rint(point).astype(int)
        if surface[cy, cx] != 0:
            reject("anchor_inside_work_surface")
            continue
        possible = np.flatnonzero(np.abs(offsets) <= peak_window)
        k = int(possible[np.argmax(profile[possible])])
        peak = float(profile[k])
        left_flank = profile[offsets <= -min_flank]
        right_flank = profile[offsets >= min_flank]
        if len(left_flank) < 4 or len(right_flank) < 4:
            reject("flank_baseline_unavailable")
            continue
        left_base = float(np.percentile(left_flank, 35))
        right_base = float(np.percentile(right_flank, 35))
        left_prom = peak - left_base
        right_prom = peak - right_base
        if min(left_prom, right_prom) < PROFILE_PROMINENCE_MIN:
            reject("one_flank_contrast_insufficient")
            continue
        left_half = _crossing(profile, offsets, k, left_base + .5 * left_prom, -1)
        right_half = _crossing(profile, offsets, k, right_base + .5 * right_prom, 1)
        if left_half is None or right_half is None:
            reject("half_prominence_crossing_missing")
            continue
        left_grad = _edge_from_gradient(profile, offsets, left_half, -1, width_hint)
        right_grad = _edge_from_gradient(profile, offsets, right_half, 1, width_hint)
        if left_grad is None or right_grad is None:
            reject("opposite_signed_gradient_missing")
            continue
        agreement = max(abs(left_grad[0] - left_half), abs(right_grad[0] - right_half))
        if agreement > max(HALF_GRAD_AGREEMENT_MIN_PX, width_hint * .16):
            reject("gradient_and_half_prominence_disagree")
            continue
        # 半峰响应背景色，梯度响应实际过渡；两者达成局部一致后融合。
        left = .6 * left_half + .4 * left_grad[0]
        right = .6 * right_half + .4 * right_grad[0]
        width = right - left
        if not WIDTH_RATIO_MIN * width_hint <= width <= WIDTH_RATIO_MAX * width_hint:
            reject("edge_spacing_implausible_vs_A1")
            continue
        if abs((left + right) * .5) > max(3., width_hint * .38):
            reject("paired_center_too_far_from_A1_support")
            continue
        edge_xy = point[None, :] + np.asarray([left, right])[:, None] * normal
        ex = np.rint(edge_xy[:, 0]).astype(int)
        ey = np.rint(edge_xy[:, 1]).astype(int)
        if (np.any(ex < 0) or np.any(ex >= surface.shape[1]) or
                np.any(ey < 0) or np.any(ey >= surface.shape[0]) or
                np.any(surface[ey, ex] != 0)):
            reject("measured_edge_inside_work_surface_or_frame_border")
            continue
        # 边外数像素仍应偏暗；可挡掉绳内部一条高亮股纹。
        l_outer = profile[(offsets >= left - max(4., width * .3)) & (offsets <= left - 1.)]
        r_outer = profile[(offsets >= right + 1.) & (offsets <= right + max(4., width * .3))]
        if len(l_outer) < 2 or len(r_outer) < 2 or (float(np.median(l_outer)) > peak - .22 * left_prom or
                                                    float(np.median(r_outer)) > peak - .22 * right_prom):
            reject("outer_flanks_not_darker_than_stripe")
            continue
        t = float((point - center0) @ direction)
        base_n = float((point - center0) @ normal)
        keep.append((t, base_n + left, base_n + right,
                     min(left_grad[1], right_grad[1]), agreement))
    array = np.asarray(keep, float).reshape(-1, 5)
    return array, {"channel": channel_kind, "tested_profiles": int(len(anchors)),
                   "paired_profiles": int(len(array)), "reject_counts": rejects,
                   "normal_profile_step_px": PROFILE_STEP_PX,
                   "independent_side_baselines": True}


def _joint_fit(pairs: np.ndarray, width_hint: float) -> tuple[np.ndarray | None, np.ndarray, dict]:
    """两侧点同权估计一条中心直线与平行边间距，Huber 权重抑制绳股噪点。"""
    n = len(pairs)
    t = pairs[:, 0]
    design = np.empty((2 * n, 3), float)
    design[:n] = np.column_stack((np.ones(n), t, -.5 * np.ones(n)))
    design[n:] = np.column_stack((np.ones(n), t, .5 * np.ones(n)))
    values = np.r_[pairs[:, 1], pairs[:, 2]]

    def solve(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
        for _ in range(7):
            error = y - x @ coefficients
            mad = 1.4826 * np.median(np.abs(error - np.median(error)))
            threshold = max(1.1, 1.6 * mad)
            weights = np.minimum(1., threshold / np.maximum(np.abs(error), 1e-8))
            update = np.linalg.lstsq(x * np.sqrt(weights[:, None]),
                                     y * np.sqrt(weights), rcond=None)[0]
            if float(np.linalg.norm(update - coefficients)) < 1e-5:
                coefficients = update
                break
            coefficients = update
        return coefficients

    coefficients = solve(design, values)
    predicted_left = coefficients[0] + coefficients[1] * t - .5 * coefficients[2]
    predicted_right = coefficients[0] + coefficients[1] * t + .5 * coefficients[2]
    per_pair_error = np.maximum(np.abs(pairs[:, 1] - predicted_left),
                                np.abs(pairs[:, 2] - predicted_right))
    inlier_limit = max(2.5, .18 * width_hint)
    inlier = per_pair_error <= inlier_limit
    if int(np.sum(inlier)) < max(SAMPLE_MIN, math.ceil(INLIER_FRACTION_MIN * n)):
        return None, inlier, {"reason": "paired_edge_joint_fit_too_few_inliers",
                                "initial_pairs": n, "inliers": int(np.sum(inlier))}
    ids = np.r_[np.flatnonzero(inlier), np.flatnonzero(inlier) + n]
    coefficients = solve(design[ids], values[ids])
    error = values[ids] - design[ids] @ coefficients
    return coefficients, inlier, {"initial_pairs": n, "inliers": int(np.sum(inlier)),
                                  "fit_edge_residual_median_abs_px": float(np.median(np.abs(error))),
                                  "fit_edge_residual_p90_abs_px": float(np.percentile(np.abs(error), 90)),
                                  "pair_inlier_limit_px": inlier_limit}


def refine_free_segment(image_bgr: np.ndarray, surface_mask: np.ndarray,
                        a1_free: dict) -> dict:
    """在 A1 原图观测直段内做双边共同精修；无证据则完整保留 A1。

    `a1_free` 必须是 A1 `detect_free_segment` 的结果，不读取其输入时使用的
    C3 线号、历史 occlusion 或人工真值。输入输出都在原图坐标系。
    """
    if not isinstance(a1_free, dict):
        raise ValueError("a1_free must be a dictionary")
    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must be HxWx3")
    if not isinstance(surface_mask, np.ndarray) or surface_mask.shape != image_bgr.shape[:2]:
        raise ValueError("surface_mask must match image height and width")
    if image_bgr.dtype != np.uint8:
        raise ValueError("image_bgr must be uint8 original pixels")
    if a1_free.get("status") != "observed":
        return _fallback(a1_free, "a1_has_no_observed_free_segment")
    if a1_free.get("coordinate_system") != "original_unmirrored_image_xy_pixels":
        return _fallback(a1_free, "a1_coordinate_system_not_original_pixels")
    if a1_free.get("observed_points_kind") != "individual_original_pixel_outside_surface_half_prominence_profiles":
        return _fallback(a1_free, "a1_has_no_individual_original_profile_samples")
    try:
        anchors = np.asarray(a1_free["observed_points_xy"], float).reshape(-1, 2)
        segment = np.asarray(a1_free["observed_segment_xy"], float).reshape(2, 2)
        width_hint = float(a1_free["width_px"])
    except (KeyError, TypeError, ValueError):
        return _fallback(a1_free, "a1_geometry_missing_or_malformed")
    if (len(anchors) < SAMPLE_MIN or not np.isfinite(anchors).all() or
            not np.isfinite(segment).all() or not math.isfinite(width_hint) or width_hint <= 1.):
        return _fallback(a1_free, "a1_support_or_width_insufficient")
    center0 = np.mean(segment, axis=0)
    vector = segment[1] - segment[0]
    length0 = float(np.linalg.norm(vector))
    if length0 < SUPPORT_SPAN_MIN_PX:
        return _fallback(a1_free, "a1_observed_span_too_short")
    direction = vector / length0
    if direction[0] < 0:
        direction = -direction
    normal = np.array([-direction[1], direction[0]])
    # 仅为限制计算量均匀抽取已经存在的 A1 原图支撑点，不填空缺、不外推。
    projection = (anchors - center0) @ direction
    order = np.argsort(projection)
    anchors = anchors[order]
    if len(anchors) > SAMPLE_MAX:
        ids = np.unique(np.rint(np.linspace(0, len(anchors) - 1, SAMPLE_MAX)).astype(int))
        anchors = anchors[ids]
    pairs, profile_diag = _measure_pairs(image_bgr, surface_mask, a1_free,
                                         anchors, center0, direction, normal, width_hint)
    if len(pairs) < max(SAMPLE_MIN, math.ceil(INLIER_FRACTION_MIN * len(anchors))):
        return _fallback(a1_free, "too_few_independent_two_edge_profiles", profile_diag)
    coefficients, inlier, fit_diag = _joint_fit(pairs, width_hint)
    diagnostics = {**profile_diag, **fit_diag}
    if coefficients is None:
        return _fallback(a1_free, fit_diag["reason"], diagnostics)
    p = pairs[inlier]
    t = p[:, 0]
    span = float(np.ptp(t))
    center_n = .5 * (p[:, 1] + p[:, 2])
    widths = p[:, 2] - p[:, 1]
    width_median = float(np.median(widths))
    width_mad = float(1.4826 * np.median(np.abs(widths - width_median)))
    edge_residual = float(fit_diag["fit_edge_residual_p90_abs_px"])
    new_direction = direction + coefficients[1] * normal
    new_direction /= np.linalg.norm(new_direction)
    if new_direction[0] < 0:
        new_direction = -new_direction
    old_angle = float(a1_free["angle_deg"])
    new_angle = float(math.degrees(math.atan2(new_direction[1], new_direction[0])) % 180.)
    angle_change = abs((new_angle - old_angle + 90.) % 180. - 90.)
    middle_t = float(np.median(t))
    center_shift = abs(float(coefficients[0] + coefficients[1] * middle_t))
    diagnostics.update({"observed_span_px": span, "median_pair_width_px": width_median,
                        "pair_width_mad_px": width_mad, "fit_width_px": float(coefficients[2]),
                        "angle_change_vs_A1_deg": angle_change,
                        "center_normal_shift_vs_A1_px": center_shift,
                        "a1_width_px": width_hint, "a1_angle_deg": old_angle})
    failures = []
    if span < SUPPORT_SPAN_MIN_PX:
        failures.append("two_edge_support_span_short")
    if not WIDTH_RATIO_MIN * width_hint <= coefficients[2] <= WIDTH_RATIO_MAX * width_hint:
        failures.append("joint_width_implausible_vs_A1")
    if width_mad > max(2., .26 * width_median):
        failures.append("pair_width_variable")
    if edge_residual > max(2.5, .22 * width_hint):
        failures.append("paired_edge_fit_residual_high")
    if angle_change > ANGLE_CHANGE_MAX_DEG:
        failures.append("angle_change_too_large_for_same_A1_span")
    if center_shift > max(3., CENTER_SHIFT_MAX_WIDTH_FRACTION * width_hint):
        failures.append("center_shift_too_large_for_same_A1_span")
    if failures:
        diagnostics["acceptance_failures"] = failures
        return _fallback(a1_free, "two_edge_fit_not_sufficiently_supported", diagnostics)

    # 边缘坐标均为原图每条法向剖面实际测得的亚像素交点；拟合线仅定义
    # 直段整体位置和方向，不把中间缺失像素伪装为观测。
    bases = center0[None, :] + t[:, None] * direction
    left_points = bases + p[:, 1, None] * normal
    right_points = bases + p[:, 2, None] * normal
    observed_points = .5 * (left_points + right_points)
    sort = np.argsort(observed_points[:, 0])
    observed_points = observed_points[sort]
    left_points = left_points[sort]
    right_points = right_points[sort]
    ends_t = np.percentile(t, [1, 99])
    ends = np.asarray([center0 + value * direction +
                       (coefficients[0] + coefficients[1] * value) * normal
                       for value in ends_t])
    if ends[0, 0] > ends[1, 0]:
        ends = ends[::-1]
    toward, toward_status, toward_candidates, surface_ray_hits = _refined_surface_direction(
        ends, surface_mask, new_direction)
    diagnostics["refined_surface_ray_hits"] = surface_ray_hits
    if toward_status == "refined_line_ray_does_not_reach_surface":
        return _fallback(a1_free, "refined_line_lost_A1_surface_approach", diagnostics)
    new_center = np.mean(ends, axis=0)
    new_normal = np.array([-new_direction[1], new_direction[0]])
    if new_normal[1] < 0:
        new_normal = -new_normal
        left_points, right_points = right_points, left_points
    center_error = center_n - (coefficients[0] + coefficients[1] * t)
    normal_scatter = float(1.4826 * np.median(np.abs(center_error - np.median(center_error))))
    angle_scatter = float(math.degrees(normal_scatter / max(span, 1.)))
    result = copy.deepcopy(a1_free)
    result.update({"observed_segment_xy": ends.tolist(), "center_xy": new_center.tolist(),
                   "direction_xy": new_direction.tolist(), "normal_xy": new_normal.tolist(),
                   "toward_surface_direction_xy": toward,
                   "toward_surface_status": toward_status,
                   "toward_surface_candidates_xy": toward_candidates,
                   "rho_px": float(new_center @ new_normal), "angle_deg": new_angle,
                   "width_px": float(coefficients[2]),
                   "width_interval_px": [float(v) for v in np.percentile(widths, [10, 90])],
                   "observed_points_xy": observed_points.tolist(),
                   "boundary_left_xy": left_points.tolist(),
                   "boundary_right_xy": right_points.tolist(),
                   "observed_points_kind": "paired_independent_original_subpixel_edge_midpoints",
                   "boundary_points_kind": "independently_measured_half_prominence_gradient_fused_edges",
                   "support_length_px": float(np.linalg.norm(ends[1] - ends[0])),
                   "observed_support": int(len(p)), "profile_sample_count": int(len(p)),
                   "angular_uncertainty_deg": angle_scatter,
                   "normal_uncertainty_px": normal_scatter,
                   "uncertainty_kind": "empirical_paired_edge_fit_scatter_not_calibrated",
                   "width_kind": "original_optical_two_edge_spacing_not_physical_diameter"})
    result.setdefault("diagnostics", {})["surface_approach_refined"] = {
        "surface_ray_hits": surface_ray_hits,
        "direction_meaning": "finite_refined_line_ray_toward_work_ROI_not_verified_rope_contact"}
    result["refinement"] = {"status": "accepted", "reason": "paired_original_edges_consistent",
                            "method": "paired_original_half_prominence_and_opposite_gradient_edges",
                            "a1_geometry_preserved": False, "diagnostics": diagnostics,
                            "interpretation": "Visible optical stripe edge spacing, not physical rope diameter or contact point"}
    return result
