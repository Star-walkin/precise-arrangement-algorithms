"""用独立黄点片段拟合共享抛物线曲线族，并给出有限几何补全。

图像 y 坐标向下；y=a*(x-h)^2+k 且 a>0 是顶点在上、两臂向下
的近竖直轴抛物线。抛物线属于圆锥曲线，但本模块不声称从短弧识别
完整椭圆或材料身份。输出平滑线是拟合结果；原黄点仍须独立显示。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ConicParameters:
    min_fragment_points: int = 8
    min_fragment_span_pitches: float = 3.0
    max_fragment_mad_pitches: float = .20
    max_pair_offset_pitches: float = .42
    max_pair_gap_pitches: float = 25.0
    max_pair_gap_pixels: int = 650
    max_endpoint_error_pitches: float = .22
    min_neighbor_courses: int = 2
    max_curve_mad_pitches: float = .24
    min_positive_curvature_fraction: float = .68
    near_flat_sag_pitches: float = .10
    max_curvature_change_fraction: float = .18
    max_slope_change_px_per_px: float = .025


def _binary_mask(mask: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8 HxW")
    if np.any((mask != 0) & (mask != 255)):
        raise ValueError(f"{name} must contain only 0/255")
    return mask > 0


def _points(line: dict, width: int, height: int) -> np.ndarray:
    raw = np.asarray(line.get("observed_points_xy", []), dtype=float)
    if raw.size == 0:
        return np.empty((0, 2), dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 2 or not np.isfinite(raw).all():
        raise ValueError("invalid observed_points_xy")
    if ((raw[:, 0] < 0).any() or (raw[:, 0] >= width).any() or
        (raw[:, 1] < 0).any() or (raw[:, 1] >= height).any()):
        raise ValueError("observed_points_xy outside image")
    return raw[np.argsort(raw[:, 0])]


def _fit_shared(fragments: list[np.ndarray], center: float, scale: float,
                pitch: float, parameters: ConicParameters) -> tuple[np.ndarray | None, dict]:
    """片段内先去截距，再稳健联合估计 b、c；绝不使用隐藏点。"""
    bases, targets, local_curvatures = [], [], []
    for pts in fragments:
        if len(pts) < parameters.min_fragment_points or np.ptp(pts[:, 0]) < parameters.min_fragment_span_pitches * pitch:
            continue
        u = (pts[:, 0] - center) / scale
        basis = np.column_stack((u, u * u))
        bases.append(basis)
        targets.append(pts[:, 1].astype(float))
        if np.ptp(pts[:, 0]) >= max(90., 4 * pitch):
            x_local = pts[:, 0] - np.median(pts[:, 0])
            local_curvatures.append(float(np.polyfit(x_local, pts[:, 1], 2)[0]))
    if len(bases) < 3:
        return None, {"status": "too_few_independent_fragments"}
    # y=a_i+b*u+c*u² 的截距 a_i 只在“同一组权重下的均值中心化”后
    # 才会严格消掉。旧版对 y、u、u² 各取中位数并分别相减；抛物线跨顶点
    # 时 median(y) 不等于 b*median(u)+c*median(u²)，会系统偏置斜率。
    centered = [basis - np.mean(basis, axis=0) for basis in bases]
    initial_weight = [np.full(len(basis), 1 / len(basis)) for basis in bases]
    A = np.concatenate(centered)
    base_weight = np.concatenate(initial_weight)
    gram = A.T @ (base_weight[:, None] * A)
    condition = float(np.linalg.cond(gram))
    if not np.isfinite(condition) or condition > 1e5:
        return None, {"status": "shared_curvature_unidentifiable",
                      "condition": condition if math.isfinite(condition) else None}
    coef = np.zeros(2, dtype=float)
    robust_factors = [np.ones(len(basis), dtype=float) for basis in bases]
    centered_residuals = np.array([], dtype=float)
    for _ in range(7):
        design, target, weights = [], [], []
        for basis, values, factor in zip(bases, targets, robust_factors):
            weight = factor / len(factor)
            denom = float(np.sum(weight))
            b_mean = np.sum(weight[:, None] * basis, axis=0) / denom
            y_mean = float(np.sum(weight * values) / denom)
            design.append(basis - b_mean)
            target.append(values - y_mean)
            weights.append(weight)
        A = np.concatenate(design)
        y = np.concatenate(target)
        weight = np.concatenate(weights)
        coef = np.linalg.lstsq(A * np.sqrt(weight[:, None]),
                               y * np.sqrt(weight), rcond=None)[0]
        # 每次更新 b/c 后重新估计各片段截距，再用残差更新 Huber 权重。
        residuals = []
        for basis, values, current_weight in zip(bases, targets, weights):
            raw = values - basis @ coef
            intercept = float(np.sum(current_weight * raw) / np.sum(current_weight))
            residuals.append(raw - intercept)
        centered_residuals = np.concatenate(residuals)
        robust = max(.8, 1.4826 * float(np.median(np.abs(centered_residuals))))
        robust_factors = [np.minimum(1.0, 1.5 * robust / np.maximum(np.abs(r), 1e-6))
                          for r in residuals]
    if not np.isfinite(coef).all():
        return None, {"status": "nonfinite_model"}
    b, c = float(coef[0]), float(coef[1])
    a_px2 = c / (scale * scale)
    sag = abs(c)
    positive_fraction = (float(np.mean(np.asarray(local_curvatures) > 0))
                         if local_curvatures else None)
    # 若可辨弯曲和多条独立短弧一起指向反向，明确拒识；小弯曲可退化直线。
    if sag <= parameters.near_flat_sag_pitches * pitch:
        c = 0.0
        status = "near_flat_linear_fallback"
    elif (a_px2 <= 0 or
          (positive_fraction is not None and positive_fraction < parameters.min_positive_curvature_fraction)):
        return None, {"status": "reverse_or_inconsistent_curvature",
                      "a_px_per_px2": a_px2, "positive_local_fraction": positive_fraction,
                      "sag_px": sag}
    else:
        status = "positive_vertical_axis_parabola"
    coef = np.array([b, c], dtype=float)
    alpha = c / (scale * scale)
    axis = (center - b * scale / (2 * c)) if c else None
    return coef, {"status": status, "formula": "y_i=k_i+b*u+c*u^2; u=(x-center)/scale",
                  "conic_type": "parabola" if c else "near_flat_line",
                  "center_x_px": center, "scale_x_px": scale,
                  "b_px_per_u": b, "c_px_per_u2": c,
                  "a_px_per_px2": alpha, "shared_axis_x_px": axis,
                  "sag_across_half_span_px": abs(c), "condition": condition,
                  "positive_local_curvature_fraction": positive_fraction,
                  "centered_residual_mad_px": float(np.median(np.abs(centered_residuals)))}


def _shape(x: np.ndarray | float, coef: np.ndarray,
           center: float, scale: float) -> np.ndarray | float:
    u = (x - center) / scale
    return coef[0] * u + coef[1] * u * u


def _estimate_spacing(offsets: list[float], pitch: float) -> tuple[float, dict]:
    """相近 offset 先视作同一行的多个片段，再量相邻行的光学间距。"""
    values = sorted(v for v in offsets if math.isfinite(v))
    distinct = []
    for value in values:
        if not distinct or value - distinct[-1] > .28 * pitch:
            distinct.append(value)
    differences = np.diff(distinct)
    plausible = differences[(differences >= .55 * pitch) & (differences <= 1.35 * pitch)]
    if len(plausible) < 3:
        return pitch, {"status": "input_pitch_fallback", "plausible_spacing_count": int(len(plausible))}
    spacing = float(np.median(plausible))
    spacing = min(1.30 * pitch, max(.70 * pitch, spacing))
    return spacing, {"status": "estimated_from_independent_offsets",
                     "plausible_spacing_count": int(len(plausible)),
                     "measured_pitch_px": spacing}


def _course_neighbors(index: int, offset: float, fragments: list[np.ndarray],
                      offsets: list[float], residuals: list[float],
                      interval: tuple[float, float], spacing: float, pitch: float) -> int:
    """数不同的邻行；同一邻行断成多片不能反复计数。"""
    x0, x1 = interval
    neighbor_offsets = []
    for k, pts in enumerate(fragments):
        if k == index or len(pts) < 8 or residuals[k] > .24 * pitch:
            continue
        if pts[-1, 0] < x0 - 2 * pitch or pts[0, 0] > x1 + 2 * pitch:
            continue
        distance = abs(offsets[k] - offset)
        multiple = int(round(distance / spacing))
        if 1 <= multiple <= 6 and abs(distance - multiple * spacing) <= .36 * pitch:
            neighbor_offsets.append(offsets[k])
    unique = []
    for value in sorted(neighbor_offsets):
        if not unique or value - unique[-1] > .46 * pitch:
            unique.append(value)
    return len(unique)


def _fit_course(points: np.ndarray, shared: np.ndarray, center: float, scale: float,
                pitch: float, parameters: ConicParameters) -> tuple[np.ndarray, float, float]:
    """在共享形状周围作有限逐圈微调；不任意扭成每行一条独立曲线。"""
    u = (points[:, 0] - center) / scale
    basis = np.column_stack((u, u * u))
    base_shape = basis @ shared
    k = float(np.median(points[:, 1] - base_shape))
    if len(points) >= 15 and np.ptp(points[:, 0]) >= 6 * pitch:
        # 轻微透视/层厚变化可以造成不完全平移。加岭惩罚，并对变化上限钳制。
        A = np.column_stack((u, u * u, np.ones(len(points))))
        target = points[:, 1] - base_shape
        penalty = np.diag([max(6.0, .20 * len(points)),
                           max(6.0, .20 * len(points)), 0.0])
        solution = np.linalg.solve(A.T @ A + penalty, A.T @ target)
        max_b = parameters.max_slope_change_px_per_px * scale
        delta_b = float(np.clip(solution[0], -max_b, max_b))
        max_c = max(.1 * pitch, parameters.max_curvature_change_fraction * abs(shared[1]))
        delta_c = float(np.clip(solution[1], -max_c, max_c))
        b = float(shared[0] + delta_b)
        c = float(max(0.0, shared[1] + delta_c))
        k = float(np.median(points[:, 1] - b * u - c * u * u))
    else:
        b, c = float(shared[0]), float(shared[1])
    prediction = k + b * u + c * u * u
    mad = float(np.median(np.abs(points[:, 1] - prediction)))
    return np.array([b, c, k], dtype=float), mad, float(np.quantile(np.abs(points[:, 1] - prediction), .90))


def _local_visible_fit(points: np.ndarray, pitch: float) -> tuple[dict | None, str]:
    """仅给共享模型拒识的长可见片段作单段回退，绝不拿它独自跨缺口。"""
    if len(points) < 12 or np.ptp(points[:, 0]) < 4 * pitch:
        return None, "local_fallback_too_short"
    center = float(np.median(points[:, 0]))
    scale = max(1., float(np.ptp(points[:, 0]) / 2))
    u = (points[:, 0] - center) / scale
    A = np.column_stack((u, u * u, np.ones(len(u))))
    weights = np.ones(len(u))
    coef = np.zeros(3)
    for _ in range(5):
        coef = np.linalg.lstsq(A * np.sqrt(weights[:, None]),
                               points[:, 1] * np.sqrt(weights), rcond=None)[0]
        residual = points[:, 1] - A @ coef
        robust = max(.8, 1.4826 * float(np.median(np.abs(residual))))
        weights = np.minimum(1., 1.5 * robust / np.maximum(np.abs(residual), 1e-6))
    b, c, k = map(float, coef)
    a = c / scale**2
    if a < 0 and abs(c) > .10 * pitch:
        return None, "local_fallback_reverse_curvature"
    if abs(c) <= .10 * pitch:
        c = 0.
        k = float(np.median(points[:, 1] - b * u))
    slope_left = (b + 2 * c * float(u.min())) / scale
    slope_right = (b + 2 * c * float(u.max())) / scale
    if max(abs(slope_left), abs(slope_right)) > .19:
        return None, "local_fallback_endpoint_slope"
    residual = np.abs(points[:, 1] - (k + b * u + c * u * u))
    mad = float(np.median(residual))
    p90 = float(np.quantile(residual, .90))
    if mad > .24 * pitch or p90 > .50 * pitch:
        return None, "local_fallback_residual"
    return {"b": b, "c": c, "k": k, "center": center, "scale": scale,
            "mad": mad, "p90": p90}, "ok"


def _endpoint_slope(points: np.ndarray, side: str) -> float | None:
    end = points[-min(6, len(points)):] if side == "right" else points[:min(6, len(points))]
    if len(end) < 3 or np.ptp(end[:, 0]) < 4:
        return None
    return float(np.polyfit(end[:, 0], end[:, 1], 1)[0])


def _anchored_pair_fit(left: np.ndarray, right: np.ndarray,
                       shared: np.ndarray, center: float, scale: float,
                       pitch: float) -> tuple[np.ndarray | None, str, dict]:
    """两端定点，共享正曲率给中间形状；只用可见点检验候选。

    形式 y(x)=linear(L,R)+a*(x-L)*(x-R)。它仍是近竖直轴抛物线，
    且恰好经过两端黄点；a 从邻线共享曲率起步，只允许有限偏移。
    """
    xL, yL = map(float, left[-1])
    xR, yR = map(float, right[0])
    gap = xR - xL
    if gap <= 0:
        return None, "anchor_order", {}
    a_shared = float(shared[1] / scale**2)
    if a_shared < 0:
        return None, "anchor_reverse_shared_curvature", {}
    points = np.concatenate((left, right))
    x = points[:, 0]
    baseline = yL + (yR - yL) * (x - xL) / gap
    q = (x - xL) * (x - xR)
    # 短端头难以单独量二阶导，此时严格采用其他行共享的正曲率。
    if a_shared == 0 or len(left) < 6 or len(right) < 6 or np.ptp(points[:, 0]) < 5 * pitch:
        a = a_shared
    else:
        prior = max(1., .12 * len(points)) * float(np.median(q * q))
        denom = float(np.sum(q * q) + prior)
        a = float((np.sum(q * (points[:, 1] - baseline)) + prior * a_shared) / denom)
        if a_shared > 0 and not (.65 * a_shared <= a <= 1.35 * a_shared):
            return None, "anchor_curvature_disagrees_with_family", {"a_candidate": a}
    chord_slope = (yR - yL) / gap
    slope_left = chord_slope - a * gap
    slope_right = chord_slope + a * gap
    if max(abs(slope_left), abs(slope_right)) > .19:
        return None, "anchor_implausible_tangent", {"slopes": [slope_left, slope_right]}
    for actual, predicted in ((_endpoint_slope(left, "right"), slope_left),
                              (_endpoint_slope(right, "left"), slope_right)):
        if actual is not None and abs(actual - predicted) > .15:
            return None, "anchor_observed_tangent_mismatch", {"observed": actual, "predicted": predicted}
    prediction = baseline + a * q
    residual = np.abs(points[:, 1] - prediction)
    mad = float(np.median(residual))
    p90 = float(np.quantile(residual, .90))
    if mad > .24 * pitch or p90 > .50 * pitch:
        return None, "anchor_visible_residual", {"mad_px": mad, "p90_px": p90}
    # 把端点式转换成模块统一的 u=(x-center)/scale 形式。
    c = a * scale**2
    b = scale * (chord_slope + a * (2 * center - xL - xR))
    k = yL + chord_slope * (center - xL) + a * (center - xL) * (center - xR)
    return np.array([b, c, k], dtype=float), "ok", {"a_px_per_px2": a,
                                                     "mad_px": mad, "p90_px": p90,
                                                     "endpoint_slopes": [slope_left, slope_right]}


def fit_conic_family(lines: list[dict], pitch_px: float, roi_mask: np.ndarray,
                     exclusion_mask: np.ndarray,
                     parameters: ConicParameters | None = None) -> dict:
    """输出 image-local 曲线族、逐列平滑点和 ROI 内可绘制光栅。

    任意原黄点不在此函数中删改。两段跨过曝区的连接允许纯几何，但必须
    有两端黄点、共享走势、相邻绳距和不交叉支持；活动绳/未知区仍留黑。
    """
    p = parameters or ConicParameters()
    pitch = float(pitch_px)
    if not math.isfinite(pitch) or not 8 <= pitch <= 70:
        raise ValueError("invalid pitch_px")
    roi = _binary_mask(roi_mask, "roi_mask")
    exclusion = _binary_mask(exclusion_mask, "exclusion_mask")
    if exclusion.shape != roi.shape or np.any(exclusion & ~roi):
        raise ValueError("exclusion_mask must be inside ROI and match its shape")
    if not isinstance(lines, list):
        raise ValueError("lines must be a list")
    height, width = roi.shape
    fragments = [_points(line, width, height) for line in lines]
    smooth = np.zeros((height, width), dtype=np.uint8)
    result = {"model": None, "curves": [], "smooth_mask": smooth,
              "rejected": [], "diagnostics": {"fragment_count": len(lines)}}
    if len(fragments) < 3 or not np.any(roi):
        result["diagnostics"]["status"] = "insufficient_fragments_or_roi"
        return result
    all_x = np.concatenate([pts[:, 0] for pts in fragments if len(pts)])
    center = float((all_x.min() + all_x.max()) / 2)
    scale = max(1., float((all_x.max() - all_x.min()) / 2))
    shared, model_diag = _fit_shared(fragments, center, scale, pitch, p)
    result["model"] = model_diag
    if shared is None:
        result["diagnostics"]["status"] = "unidentifiable_or_wrong_direction"
        return result
    offsets = []
    residuals = []
    usable = []
    anchor_eligible = []
    local_fallback = []
    initial_reason = []
    reasons: Counter[str] = Counter()
    for i, pts in enumerate(fragments):
        if len(pts) < 2 or np.ptp(pts[:, 0]) < 4:
            offsets.append(float("nan"))
            residuals.append(float("inf"))
            usable.append(False)
            anchor_eligible.append(False)
            local_fallback.append(None)
            initial_reason.append("too_few_endpoint_points")
            continue
        value = pts[:, 1] - _shape(pts[:, 0], shared, center, scale)
        offset = float(np.median(value))
        mad = float(np.median(np.abs(value - offset)))
        offsets.append(offset)
        residuals.append(mad)
        long_enough = (len(pts) >= p.min_fragment_points and
                       np.ptp(pts[:, 0]) >= p.min_fragment_span_pitches * pitch)
        good = long_enough and mad <= p.max_fragment_mad_pitches * pitch
        usable.append(good)
        if good:
            anchor_eligible.append(True)
            local_fallback.append(None)
            initial_reason.append(None)
        elif not long_enough:
            # 短片段没有独立测曲率的能力，但两端锚点与多条长邻线可以
            # 共同证实一次跨缺口连接。单侧短点绝不外推成整条绳。
            anchor_eligible.append(mad <= .20 * pitch)
            local_fallback.append(None)
            initial_reason.append("short_observed_fragment")
        else:
            fallback, why = _local_visible_fit(pts, pitch)
            anchor_eligible.append(fallback is not None)
            local_fallback.append(fallback)
            initial_reason.append("shared_shape_residual" if fallback is not None else why)
    spacing, spacing_diag = _estimate_spacing([offsets[i] for i in range(len(lines)) if usable[i]], pitch)
    max_gap = min(p.max_pair_gap_pitches * pitch, float(p.max_pair_gap_pixels))
    candidates = []
    # 先找任何左右不重叠的同轨迹片段，再按点数、残差、间隔排列；没有旧行号。
    for i, left in enumerate(fragments):
        if not anchor_eligible[i]:
            continue
        for j, right in enumerate(fragments):
            if i == j or not anchor_eligible[j]:
                continue
            gap = float(right[0, 0] - left[-1, 0])
            if not max(3., .25 * pitch) <= gap <= max_gap:
                continue
            offset_delta = abs(offsets[i] - offsets[j])
            if offset_delta > .75 * pitch:
                continue
            # 两端局部黄点到各自共享曲线位置必须接近；可容许整段偏移不同。
            l_tail = left[-min(5, len(left)):]
            r_head = right[:min(5, len(right))]
            l_local = float(np.median(l_tail[:, 1] - _shape(l_tail[:, 0], shared, center, scale)))
            r_local = float(np.median(r_head[:, 1] - _shape(r_head[:, 0], shared, center, scale)))
            left_error = abs(float(left[-1, 1] - _shape(left[-1, 0], shared, center, scale) - l_local))
            right_error = abs(float(right[0, 1] - _shape(right[0, 0], shared, center, scale) - r_local))
            if max(left_error, right_error) > max(2.5, p.max_endpoint_error_pitches * pitch):
                reasons["endpoint_or_local_offset_mismatch"] += 1
                continue
            standard = (usable[i] and usable[j] and
                        offset_delta <= p.max_pair_offset_pitches * pitch and
                        abs(l_local - r_local) <= .50 * pitch)
            anchored_model = None
            anchor_diag = None
            if not standard:
                if abs(l_local - r_local) > .75 * pitch:
                    reasons["anchor_offset_ambiguity"] += 1
                    continue
                anchored_model, why, anchor_diag = _anchored_pair_fit(
                    left, right, shared, center, scale, pitch)
                if anchored_model is None:
                    reasons[why] += 1
                    continue
            midpoint_offset = .5 * (offsets[i] + offsets[j])
            interval = (float(left[-1, 0]), float(right[0, 0]))
            neighbors = _course_neighbors(i, midpoint_offset, fragments, offsets, residuals,
                                          interval, spacing, pitch)
            required_neighbors = p.min_neighbor_courses + (0 if standard else 1)
            if neighbors < required_neighbors:
                reasons["insufficient_neighbor_courses"] += 1
                continue
            # 越多真实黄点、越小错位的连接优先；不把未观测中心当真值。
            support = min(len(left), 80) + min(len(right), 80)
            score = (offset_delta / pitch + .25 * gap / max_gap
                     + (left_error + right_error) / pitch - .003 * support
                     + (0 if standard else .10))
            candidates.append((score, i, j, gap, neighbors,
                               left_error, right_error,
                               "shared" if standard else "endpoint_anchored",
                               anchored_model, anchor_diag))
    predecessor, successor = {}, {}
    accepted_models = {}
    accepted = []
    for (score, i, j, gap, neighbors, left_error, right_error,
         method, anchored_model, anchor_diag) in sorted(candidates, key=lambda row: row[0]):
        if i in successor or j in predecessor:
            reasons["one_to_one_conflict"] += 1
            continue
        # 排序不能翻转。两条真实整绳边界不能在缺口里相互穿越。
        crossing = False
        for k, m in successor.items():
            if min(abs(offsets[i] - offsets[k]), abs(offsets[j] - offsets[m])) < .45 * pitch:
                continue
            if (offsets[i] - offsets[k]) * (offsets[j] - offsets[m]) < 0:
                crossing = True
                break
        if crossing:
            reasons["order_crossing"] += 1
            continue
        successor[i] = j
        predecessor[j] = i
        if anchored_model is not None:
            accepted_models[(i, j)] = anchored_model
        accepted.append({"from": lines[i].get("line_id", f"fragment_{i:04d}"),
                         "to": lines[j].get("line_id", f"fragment_{j:04d}"),
                         "method": method,
                         "gap_px": gap, "neighbor_courses": neighbors,
                         "offset_difference_px": abs(offsets[i] - offsets[j]),
                         "endpoint_errors_px": [left_error, right_error],
                         "score": score,
                         "anchor_diagnostics": anchor_diag})
    families = []
    assigned = set()
    for start in range(len(lines)):
        if not anchor_eligible[start] or start in predecessor or start in assigned:
            continue
        chain = []
        current = start
        while current not in assigned:
            chain.append(current)
            assigned.add(current)
            if current not in successor:
                break
            current = successor[current]
        families.append(chain)
    rendered_indices: set[int] = set()
    for idx, chain in enumerate(families, 1):
        course_points = np.concatenate([fragments[k] for k in chain])
        anchored_two_end = len(chain) == 2 and (chain[0], chain[1]) in accepted_models
        if anchored_two_end:
            fitted = accepted_models[(chain[0], chain[1])]
            u = (course_points[:, 0] - center) / scale
            residual = np.abs(course_points[:, 1] -
                              (fitted[2] + fitted[0] * u + fitted[1] * u * u))
            mad = float(np.median(residual))
            p90 = float(np.quantile(residual, .90))
        else:
            fitted, mad, p90 = _fit_course(course_points, shared, center, scale, pitch, p)
        if mad > p.max_curve_mad_pitches * pitch or p90 > .50 * pitch:
            result["rejected"].append({"source_fragment_ids": [lines[k].get("line_id", f"fragment_{k:04d}") for k in chain],
                                       "reason": "whole_curve_residual", "mad_px": mad,
                                       "p90_residual_px": p90})
            reasons["whole_curve_residual"] += 1
            continue
        b, c, k0 = map(float, fitted)
        x0 = int(math.ceil(min(fragments[k][0, 0] for k in chain)))
        x1 = int(math.floor(max(fragments[k][-1, 0] for k in chain)))
        sampled = []
        for x in range(max(0, x0), min(width - 1, x1) + 1):
            u = (x - center) / scale
            y = int(round(k0 + b * u + c * u * u))
            if 0 <= y < height and roi[y, x] and not exclusion[y, x]:
                sampled.append([x, y])
                smooth[y, x] = 255
        if not sampled:
            result["rejected"].append({"source_fragment_ids": [lines[k].get("line_id", f"fragment_{k:04d}") for k in chain],
                                       "reason": "no_visible_roi_curve_pixels"})
            reasons["no_visible_roi_curve_pixels"] += 1
            continue
        alpha = c / (scale * scale)
        axis = (center - b * scale / (2 * c)) if c else None
        # 顶点式的 k 不是归一化式截距，需代入轴位置重新计算。
        vertex_y = (k0 + b * ((axis - center) / scale) +
                    c * ((axis - center) / scale) ** 2) if axis is not None else None
        ids = [lines[k].get("line_id", f"fragment_{k:04d}") for k in chain]
        result["curves"].append({"curve_id": f"conic_{idx:04d}",
                                 "status": ("two_sided_endpoint_anchored" if anchored_two_end else
                                            "two_sided_geometric_connection" if len(chain) > 1 else
                                            "single_observed_fragment_fit"),
                                 "source_fragment_ids": ids,
                                 "model": {"type": "vertical_axis_parabola" if c else "near_flat_line",
                                           "a_px_per_px2": alpha, "h_px": axis,
                                           "k_vertex_px": vertex_y,
                                           "normalized_b": b, "normalized_c": c,
                                           "normalized_k": k0,
                                           "u_center_x_px": center, "u_scale_x_px": scale},
                                 "observed_span_x_px": [x0, x1],
                                 "sampled_points_xy": sampled,
                                 "support": {"observed_fragment_count": len(chain),
                                             "observed_point_count": int(len(course_points)),
                                             "median_residual_px": mad,
                                             "p90_residual_px": p90,
                                             "geometric_gap_count": max(0, len(chain) - 1),
                                             "support_level": ("two_sided_with_neighbor_courses" if len(chain) > 1
                                                               else "single_visible_fragment")},
                                 "interpretation": "image-local smooth fit, not raw observation or physical rope identity"})
        rendered_indices.update(chain)
    # 共享形状拒识的长可见片段可以独立平滑；它们没有参与凭空跨带补线。
    for i, local in enumerate(local_fallback):
        if local is None or i in rendered_indices:
            continue
        pts = fragments[i]
        x0, x1 = int(math.ceil(pts[0, 0])), int(math.floor(pts[-1, 0]))
        b, c, k0 = local["b"], local["c"], local["k"]
        local_center, local_scale = local["center"], local["scale"]
        sampled = []
        for x in range(max(0, x0), min(width - 1, x1) + 1):
            u = (x - local_center) / local_scale
            y = int(round(k0 + b * u + c * u * u))
            if 0 <= y < height and roi[y, x] and not exclusion[y, x]:
                sampled.append([x, y])
                smooth[y, x] = 255
        if not sampled:
            reasons["local_fallback_no_roi_pixels"] += 1
            continue
        axis = local_center - b * local_scale / (2 * c) if c else None
        vertex = k0 - b * b / (4 * c) if c else None
        result["curves"].append({"curve_id": f"conic_fallback_{i:04d}",
                                 "status": "single_fragment_local_conic_fallback",
                                 "source_fragment_ids": [lines[i].get("line_id", f"fragment_{i:04d}")],
                                 "model": {"type": "vertical_axis_parabola" if c else "near_flat_line",
                                           "a_px_per_px2": c / local_scale**2,
                                           "h_px": axis, "k_vertex_px": vertex,
                                           "normalized_b": b, "normalized_c": c,
                                           "normalized_k": k0,
                                           "u_center_x_px": local_center,
                                           "u_scale_x_px": local_scale},
                                 "observed_span_x_px": [x0, x1],
                                 "sampled_points_xy": sampled,
                                 "support": {"observed_fragment_count": 1,
                                             "observed_point_count": int(len(pts)),
                                             "median_residual_px": local["mad"],
                                             "p90_residual_px": local["p90"],
                                             "geometric_gap_count": 0,
                                             "support_level": "local_visible_only"},
                                 "interpretation": "locally fitted visible span only; no cross-gap identity"})
        rendered_indices.add(i)
        reasons["single_fragment_local_fallback_used"] += 1
    for i, reason in enumerate(initial_reason):
        if reason is not None and i not in rendered_indices:
            result["rejected"].append({"fragment_id": lines[i].get("line_id", f"fragment_{i:04d}"),
                                       "reason": reason,
                                       "shared_shape_mad_px": (residuals[i] if math.isfinite(residuals[i]) else None)})
            reasons[reason] += 1
    # 最终拟合后再检查两条线。独立拟合可能让相邻黄点本有一绳距，
    # 平滑端头却挤到一起；冲突列两条都留白，绝不把相邻绳强行合并。
    by_x = [{x: y for x, y in curve["sampled_points_xy"]} for curve in result["curves"]]
    clipped: list[set[int]] = [set() for _ in by_x]
    conflicts = []
    for i in range(len(by_x)):
        for j in range(i + 1, len(by_x)):
            common = by_x[i].keys() & by_x[j].keys()
            if len(common) < 2:
                continue
            signed = [by_x[i][x] - by_x[j][x] for x in common]
            crossing = min(signed) < -1 and max(signed) > 1
            close_x = {x for x in common if abs(by_x[i][x] - by_x[j][x]) < .45 * pitch}
            if not close_x and not crossing:
                continue
            affected = set(common) if crossing else close_x
            clipped[i].update(affected)
            clipped[j].update(affected)
            conflicts.append({"curve_ids": [result["curves"][i]["curve_id"],
                                            result["curves"][j]["curve_id"]],
                              "reason": "postfit_crossing" if crossing else "postfit_spacing_conflict",
                              "overlap_columns": len(common), "clipped_columns": len(affected)})
    smooth.fill(0)
    kept_curves = []
    for curve, banned in zip(result["curves"], clipped):
        if banned:
            curve["sampled_points_xy"] = [pt for pt in curve["sampled_points_xy"] if pt[0] not in banned]
            curve["support"]["postfit_spacing_clipped_pixels"] = len(banned)
            reasons["postfit_spacing_clipped_pixels"] += len(banned)
        if not curve["sampled_points_xy"]:
            result["rejected"].append({"source_fragment_ids": curve["source_fragment_ids"],
                                       "reason": "postfit_spacing_removed_entire_curve"})
            reasons["postfit_spacing_removed_entire_curve"] += 1
            continue
        for x, y in curve["sampled_points_xy"]:
            smooth[y, x] = 255
        kept_curves.append(curve)
    result["curves"] = kept_curves
    connected_indices = set(successor) | set(predecessor)
    for i, ok in enumerate(usable):
        if ok and i not in connected_indices:
            result["rejected"].append({"fragment_id": lines[i].get("line_id", f"fragment_{i:04d}"),
                                       "reason": "no_unique_cross_fragment_partner; kept_as_single_fragment_curve"})
            reasons["no_cross_fragment_partner"] += 1
    result["diagnostics"] = {"status": "ok", "fragment_count": len(lines),
                             "usable_fragment_count": int(sum(usable)),
                             "spacing": spacing_diag,
                             "accepted_connections": accepted,
                             "connection_count": len(accepted),
                             "curve_count": len(result["curves"]),
                             "smooth_pixels": int(np.count_nonzero(smooth)),
                             "postfit_conflicts": conflicts,
                             "rejection_counts": dict(reasons),
                             "identity_scope": "one image; no physical rope identity",
                             "overexposure_policy": "geometry allowed between supported ends; exclusions blank"}
    return result
