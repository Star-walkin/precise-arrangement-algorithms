"""把全 ROI 独立观测到的暗沟槽片段组成局部曲线族，并有限补齐缺口。

这是离线视觉几何候选，不是同根材料跟踪或完整椭圆拟合。黄色观测点原样保留；
紫色仅标识有两端和邻线支持的推断，活动绳覆盖内也只画紫色。
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math

import cv2
import numpy as np


def _mask(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8 HxW")
    if np.any((value != 0) & (value != 255)):
        raise ValueError(f"{name} must contain only 0/255")
    return value > 0


def _fragment(line: dict, width: int, height: int) -> np.ndarray:
    raw = np.asarray(line.get("observed_points_xy", []), dtype=float)
    if raw.size == 0:
        return np.empty((0, 2), dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 2 or not np.isfinite(raw).all():
        raise ValueError("invalid observed_points_xy")
    if (raw[:, 0] < 0).any() or (raw[:, 0] >= width).any() or (raw[:, 1] < 0).any() or (raw[:, 1] >= height).any():
        raise ValueError("observed point outside image")
    return raw[np.argsort(raw[:, 0])]


def _shared_shape(fragments: list[np.ndarray], center: float, scale: float,
                  pitch: float) -> tuple[np.ndarray | None, dict]:
    """去掉每个片段自己的截距，联合估计共享斜率与弯曲。

    模型为 y_i(x)=a_i+b*u+c*u²，u=(x-center)/scale。a_i 是每条片段
    的竖向位置；b、c 是整幅图局部共享的走势。短弧不能辨认完整椭圆
    的中心、长短轴，所以这里明确使用二次多项式曲线族。
    """
    rows, targets, weights = [], [], []
    spans = []
    for pts in fragments:
        if len(pts) < 5:
            continue
        u = (pts[:, 0] - center) / scale
        basis = np.column_stack((u, u * u))
        basis -= np.median(basis, axis=0)
        target = pts[:, 1] - np.median(pts[:, 1])
        rows.append(basis)
        targets.append(target)
        # 长片段不应凭点数压倒多条其他绳；每个片段总权重相近。
        weights.append(np.full(len(pts), 1.0 / len(pts)))
        spans.append(float(np.ptp(pts[:, 0])))
    if len(rows) < 3 or sum(len(x) for x in rows) < 30:
        return None, {"reason": "too_few_independent_fragments"}
    A = np.concatenate(rows)
    y = np.concatenate(targets)
    base = np.concatenate(weights)
    gram = A.T @ (base[:, None] * A)
    condition = float(np.linalg.cond(gram))
    # 至少跨多个绳距且二次项可辨，才允许画弯曲补线。
    if max(spans) < 3.0 * pitch or not np.isfinite(condition) or condition > 1e5:
        return None, {"reason": "shared_curvature_unidentifiable", "condition": condition,
                      "max_fragment_span_px": max(spans)}
    weight = base.copy()
    coefficient = np.zeros(2)
    for _ in range(5):
        coefficient = np.linalg.lstsq(A * np.sqrt(weight[:, None]),
                                       y * np.sqrt(weight), rcond=None)[0]
        residual = y - A @ coefficient
        robust_scale = max(1.0, 1.4826 * float(np.median(np.abs(residual))))
        weight = base * np.minimum(1.0, (1.5 * robust_scale) /
                                   np.maximum(np.abs(residual), 1e-6))
    if not np.isfinite(coefficient).all():
        return None, {"reason": "nonfinite_shared_shape"}
    return coefficient, {"reason": "ok", "model": "shared_quadratic_with_per_fragment_offset",
                         "formula": "y_i=a_i+b*u+c*u^2; u=(x-center)/scale",
                         "b_px_per_u": float(coefficient[0]),
                         "c_px_per_u2": float(coefficient[1]),
                         "center_x_px": center, "scale_x_px": scale,
                         "condition": condition,
                         "median_centered_residual_px": float(np.median(np.abs(y - A @ coefficient)))}


def _shape(x: np.ndarray | float, coefficient: np.ndarray,
           center: float, scale: float) -> np.ndarray | float:
    u = (x - center) / scale
    return coefficient[0] * u + coefficient[1] * u * u


def _offset(pts: np.ndarray, coefficient: np.ndarray,
            center: float, scale: float) -> tuple[float, float]:
    residual = pts[:, 1] - _shape(pts[:, 0], coefficient, center, scale)
    middle = float(np.median(residual))
    return middle, float(np.median(np.abs(residual - middle)))


def _endpoint_tangent(pts: np.ndarray, side: str) -> float:
    end = pts[-min(7, len(pts)):] if side == "right" else pts[:min(7, len(pts))]
    if len(end) < 3 or np.ptp(end[:, 0]) < 5:
        return float("nan")
    return float(np.polyfit(end[:, 0], end[:, 1], 1)[0])


def _valley_evidence(gray: np.ndarray, near_saturation: np.ndarray,
                     x: int, y: int, pitch: float) -> bool:
    """普通缺口要有原图暗谷；明显亮脊及过曝留白。"""
    h, w = gray.shape
    d = max(3, int(round(.37 * pitch)))
    if not (d + 2 <= y < h - d - 2 and 0 <= x < w):
        return False
    # 对 x 做小窗口平均，避免股纹单列噪声冒充整条沟槽。
    xa, xb = max(0, x - 2), min(w, x + 3)
    valley = float(gray[y - 1:y + 2, xa:xb].mean())
    above = float(gray[y - d - 1:y - d + 2, xa:xb].mean())
    below = float(gray[y + d - 1:y + d + 2, xa:xb].mean())
    # 平滑像素尚未到 247 也可能处于大片 250+ 饱和白带：周围饱和比例
    # 高且沟槽自身已很亮时，微小数值起伏不能当作可见整绳暗沟。
    if near_saturation[y, x] > .35 and valley >= 220:
        return False
    if valley >= 247 or min(above, below) - valley < 1.2:
        return False
    if max(above, below) - valley < 3.0:
        return False
    return True


def infer_family(image_bgr: np.ndarray, reference_roi: np.ndarray,
                 observed_mask: np.ndarray, lines: list[dict],
                 supported_occlusion_mask: np.ndarray, unknown_mask: np.ndarray,
                 pitch_px: float) -> tuple[np.ndarray, list[dict], dict]:
    """共享走势、关联可见片段，并把有限可信缺口画成紫线。

    返回的 family_id 只是当前图像内的几何组号，不是跨帧或跨层的物理绳号。
    未关联片段仍完整保留；无依据的缺口继续为黑色。
    """
    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError("image_bgr must be uint8 HxWx3")
    height, width = image_bgr.shape[:2]
    roi = _mask(reference_roi, (height, width), "reference_roi")
    observed = _mask(observed_mask, (height, width), "observed_mask")
    occluded = _mask(supported_occlusion_mask, (height, width), "supported_occlusion_mask")
    unknown = _mask(unknown_mask, (height, width), "unknown_mask")
    if np.any(occluded & unknown) or np.any((observed | occluded | unknown) & ~roi) or np.any(observed & (occluded | unknown)):
        raise ValueError("input mask contract violation")
    pitch = float(pitch_px)
    if not math.isfinite(pitch) or not 8 <= pitch <= 70:
        raise ValueError("invalid pitch_px")
    if not isinstance(lines, list):
        raise ValueError("lines must be list")
    result_lines = deepcopy(lines)
    fragments = [_fragment(line, width, height) for line in result_lines]
    for line, pts in zip(result_lines, fragments):
        line["family_id"] = None
        line["connected_fragment_ids"] = []
        line["inferred_points_xy"] = []
        for x, y in pts:
            if not observed[int(round(y)), int(round(x))]:
                raise ValueError("observed point is absent from observed raster")
    inferred = np.zeros((height, width), np.uint8)
    reasons: Counter[str] = Counter()
    diag: dict = {"fragment_count": len(lines), "connection_count": 0,
                  "rejection_counts": {}, "connections": [],
                  "model_scope": "image-local quadratic family; no full ellipse or material identity"}
    if not np.any(roi) or len(fragments) < 3:
        diag["status"] = "insufficient_fragments_or_roi"
        return inferred, result_lines, diag
    ys, xs = np.where(roi)
    center = float((xs.min() + xs.max()) / 2)
    scale = max(1.0, float((xs.max() - xs.min()) / 2))
    coefficient, model_diag = _shared_shape(fragments, center, scale, pitch)
    diag["shared_model"] = model_diag
    if coefficient is None:
        diag["status"] = "no_identifiable_shared_shape"
        return inferred, result_lines, diag
    # 片段先保留，各自求曲线族截距；同轨迹关联另行进行。
    offsets, deviations = [], []
    for pts in fragments:
        if len(pts) >= 5:
            a, mad = _offset(pts, coefficient, center, scale)
        else:
            a, mad = float("nan"), float("inf")
        offsets.append(a)
        deviations.append(mad)
    diag["unmatched_fragment_ids"] = []
    # 候选必须是不同片段、左右端分明，并且处于约同一整绳暗沟轨迹。
    candidates = []
    max_gap = min(12.0 * pitch, 240.0)
    for i, left in enumerate(fragments):
        if len(left) < 5:
            reasons["fragment_too_short"] += 1
            continue
        for j, right in enumerate(fragments):
            if i == j or len(right) < 5:
                continue
            gap = float(right[0, 0] - left[-1, 0])
            if gap <= max(5.0, .35 * pitch) or gap > max_gap:
                continue
            if deviations[i] > .23 * pitch or deviations[j] > .23 * pitch:
                reasons["fragment_shape_mismatch"] += 1
                continue
            offset_difference = abs(offsets[i] - offsets[j])
            if offset_difference > .34 * pitch:
                reasons["different_course_offset"] += 1
                continue
            # 全片段截距只用于判断同轨迹。补线必须和当前两端实际黄点贴合：
            # 分别用最后/最前 5 点的局部截距修正，并限制它们相对整段漂移。
            left_tail = left[-min(5, len(left)):]
            right_head = right[:min(5, len(right))]
            left_local = float(np.median(left_tail[:, 1] -
                                         _shape(left_tail[:, 0], coefficient, center, scale)))
            right_local = float(np.median(right_head[:, 1] -
                                          _shape(right_head[:, 0], coefficient, center, scale)))
            if (abs(left_local - offsets[i]) > .24 * pitch or
                abs(right_local - offsets[j]) > .24 * pitch or
                abs(left_local - right_local) > .32 * pitch):
                reasons["endpoint_offset_drift"] += 1
                continue
            left_end_error = abs(float(left[-1, 1] -
                                       _shape(left[-1, 0], coefficient, center, scale) - left_local))
            right_end_error = abs(float(right[0, 1] -
                                        _shape(right[0, 0], coefficient, center, scale) - right_local))
            if max(left_end_error, right_end_error) > max(2.5, .16 * pitch):
                reasons["endpoint_projection_mismatch"] += 1
                continue
            left_tangent = _endpoint_tangent(left, "right")
            right_tangent = _endpoint_tangent(right, "left")
            if not np.isfinite(left_tangent + right_tangent) or abs(left_tangent - right_tangent) > .13:
                reasons["endpoint_tangent_mismatch"] += 1
                continue
            # 桥接曲线的端点切线还需与观测切线相容。截距线性修正的导数
            # 是 (right_local-left_local)/gap，而不是凭完整椭圆强制弯过去。
            du_left = (left[-1, 0] - center) / scale
            du_right = (right[0, 0] - center) / scale
            offset_slope = (right_local - left_local) / gap
            bridge_slope_left = (coefficient[0] + 2 * coefficient[1] * du_left) / scale + offset_slope
            bridge_slope_right = (coefficient[0] + 2 * coefficient[1] * du_right) / scale + offset_slope
            if (abs(left_tangent - bridge_slope_left) > .10 or
                abs(right_tangent - bridge_slope_right) > .10):
                reasons["bridge_tangent_mismatch"] += 1
                continue
            # 相邻整绳的轨迹距约为 p。至少两条邻线共同支持，不要求一上一下。
            xmid = .5 * (left[-1, 0] + right[0, 0])
            neighbors = []
            for k, other in enumerate(fragments):
                if k in (i, j) or len(other) < 7 or deviations[k] > .24 * pitch:
                    continue
                separation = abs(offsets[k] - .5 * (offsets[i] + offsets[j]))
                multiple = round(separation / pitch)
                if not 1 <= multiple <= 5 or abs(separation - multiple * pitch) > .38 * pitch:
                    continue
                # 邻线至少进入左右端附近或缺口范围，不能拿画面远端做佐证。
                if other[0, 0] > right[0, 0] + 2 * pitch or other[-1, 0] < left[-1, 0] - 2 * pitch:
                    continue
                # 若中间确有其他已观测的整绳，则不跨越它接错行。
                near = other[np.abs(other[:, 0] - xmid) <= 2 * pitch]
                if len(near) and np.min(np.abs(near[:, 1] - (_shape(near[:, 0], coefficient, center, scale)
                                                            + .5 * (offsets[i] + offsets[j])))) < .45 * pitch:
                    continue
                neighbors.append(k)
            # 同一条邻绳可能断成多个片段：按相近截距去重。
            unique_offsets = []
            for k in sorted(neighbors, key=lambda k: offsets[k]):
                if not unique_offsets or abs(offsets[k] - unique_offsets[-1]) > .45 * pitch:
                    unique_offsets.append(offsets[k])
            if len(unique_offsets) < 2:
                reasons["fewer_than_two_neighbor_courses"] += 1
                continue
            score = (offset_difference / pitch + abs(left_tangent - right_tangent) / .13
                     + gap / max_gap + (left_end_error + right_end_error) / pitch)
            candidates.append((score, i, j, gap, len(unique_offsets), left_local, right_local,
                               left_end_error, right_end_error))
    raw_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(raw_gray, (0, 0),
                            sigmaX=max(3.0, .45 * pitch), sigmaY=max(1.0, .08 * pitch))
    sat_kx = max(7, int(round(.85 * pitch))) | 1
    sat_ky = max(7, int(round(.65 * pitch))) | 1
    near_saturation = cv2.blur((raw_gray >= 250).astype(np.float32), (sat_kx, sat_ky))
    # 先通过原图像素检查，再占用左右端并建立几何族。没有紫色支持的提案
    # 不会把两个片段误记成已连接，也不会抢掉后续可接受的候选。
    predecessor, successor = {}, {}
    diag["rejected_connections"] = []
    for (score, i, j, gap, neighbor_count, left_local, right_local,
         left_end_error, right_end_error) in sorted(candidates):
        left_id = result_lines[i].get("line_id", f"fragment_{i}")
        right_id = result_lines[j].get("line_id", f"fragment_{j}")
        if i in successor or j in predecessor:
            reasons["branch_conflict"] += 1
            diag["rejected_connections"].append({"from": left_id, "to": right_id,
                                                  "reason": "branch_conflict"})
            continue
        left, right = fragments[i], fragments[j]
        x0, x1 = int(round(left[-1, 0])), int(round(right[0, 0]))
        candidate_xy = []
        under_rope = 0
        rejected_bright = 0
        eligible = 0
        nonoccluded_eligible = 0
        bright_run = 0
        longest_bright_run = 0
        # 两端局部截距线性过渡，保证紫线从末端黄点附近出发并回到起端。
        for x in range(x0 + 1, x1):
            t = (x - x0) / max(1, x1 - x0)
            a = (1 - t) * left_local + t * right_local
            y = int(round(float(_shape(float(x), coefficient, center, scale) + a)))
            if not (0 <= y < height) or not roi[y, x] or unknown[y, x] or observed[y, x]:
                bright_run = 0
                continue
            eligible += 1
            if not occluded[y, x]:
                nonoccluded_eligible += 1
            if not occluded[y, x] and not _valley_evidence(gray, near_saturation, x, y, pitch):
                rejected_bright += 1
                bright_run += 1
                longest_bright_run = max(longest_bright_run, bright_run)
                continue
            bright_run = 0
            candidate_xy.append([x, y])
            if occluded[y, x]:
                under_rope += 1
        if eligible < 4 or len(candidate_xy) < max(4, int(math.ceil(.55 * eligible))):
            reason = "too_little_pixel_support"
        elif (nonoccluded_eligible and
              rejected_bright / nonoccluded_eligible > .35):
            reason = "bright_ridge_or_saturation_contradiction"
        elif longest_bright_run >= max(5, int(round(.55 * pitch))):
            reason = "long_bright_ridge_crossing"
        else:
            reason = None
        if reason is not None:
            reasons[reason] += 1
            diag["rejected_connections"].append({"from": left_id, "to": right_id,
                                                  "reason": reason,
                                                  "eligible_pixels": eligible,
                                                  "supported_pixels": len(candidate_xy),
                                                  "rejected_nonoccluded_pixels": rejected_bright,
                                                  "longest_rejected_run_px": longest_bright_run})
            continue
        # 单个采样点即一像素紫色；不跨无依据区强行连线。
        for x, y in candidate_xy:
            inferred[y, x] = 255
        result_lines[i]["inferred_points_xy"].extend(candidate_xy)
        result_lines[i]["connected_fragment_ids"].append(right_id)
        result_lines[j]["connected_fragment_ids"].append(left_id)
        successor[i], predecessor[j] = j, i
        diag["connections"].append({"from": left_id, "to": right_id,
                                    "gap_px": gap, "neighbor_courses": neighbor_count,
                                    "score": score, "inferred_pixels": len(candidate_xy),
                                    "under_occlusion_pixels": under_rope,
                                    "bright_or_no_valley_rejections": rejected_bright,
                                    "eligible_pixels": eligible,
                                    "left_endpoint_projection_error_px": left_end_error,
                                    "right_endpoint_projection_error_px": right_end_error,
                                    "model": "shared quadratic plus endpoint offset interpolation"})
    family_of = {}
    next_family = 1
    for start in range(len(fragments)):
        if start in predecessor:
            continue
        cur = start
        family = f"family_{next_family:04d}"
        next_family += 1
        while cur not in family_of:
            family_of[cur] = family
            if cur not in successor:
                break
            cur = successor[cur]
    for k, line in enumerate(result_lines):
        line["family_id"] = family_of.get(k, f"family_{next_family+k:04d}")
        if k not in predecessor and k not in successor:
            diag["unmatched_fragment_ids"].append(line.get("line_id", f"fragment_{k}"))
    inferred[~roi | unknown | observed] = 0
    diag["status"] = "ok"
    diag["connection_count"] = len(diag["connections"])
    diag["inferred_pixels"] = int(np.count_nonzero(inferred))
    diag["inferred_under_occlusion_pixels"] = int(np.count_nonzero((inferred > 0) & occluded))
    diag["rejection_counts"] = dict(reasons)
    return inferred, result_lines, diag
