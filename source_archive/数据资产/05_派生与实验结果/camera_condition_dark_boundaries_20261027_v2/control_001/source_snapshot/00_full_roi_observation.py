"""从原图的整个亮带 ROI 独立寻找卷绕绳暗沟槽候选。

本模块不读取 v1/v2 黄色点、线号或人工真值。黄色只在当前原图确实
有暗谷且两侧有亮度对比的采样点落下；两点之间不画几何补线。
输出是视觉边界候选，不证明钢丝绳材料身份或每根绳都已检出。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FullROIParameters:
    # 所有长度都按原图像素计；p 是从现有 ROI 记录取得的光学绳距。
    column_step_pitches: float = 0.30
    flank_offset_pitches: float = 0.38
    min_column_spacing_pitches: float = 0.65
    exclusion_guard_pitches: float = 0.32
    min_span_pitches: float = 4.0
    min_span_pixels: int = 76
    min_points: int = 12
    min_support_fraction: float = 0.57
    max_link_distance_pitches: float = 0.22
    max_link_skip_columns: int = 2
    max_slope: float = 0.19
    max_quadratic_mad_pitches: float = 0.12
    min_weak_contrast: float = 0.8
    min_strong_contrast: float = 2.2
    min_response: float = 2.6
    max_near_saturation_fraction: float = 0.48
    min_saturated_valley_intensity: float = 235.0


def _signals(image_bgr: np.ndarray, pitch: float,
             bounds: tuple[int, int, int, int], flank: float
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算局部暗谷：谷点上、下各约 0.38p 处都应比谷点亮。"""
    x0, y0, x1, y1 = bounds
    gray = cv2.cvtColor(image_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    # 沿绳走势 x 方向平滑多些；跨绳 y 方向平滑少些，避免抹掉绳间暗沟。
    smooth = cv2.GaussianBlur(gray, (0, 0),
                              sigmaX=max(5.0, .58 * pitch),
                              sigmaY=max(1.0, .10 * pitch))
    broad = cv2.GaussianBlur(smooth, (0, 0),
                             sigmaX=max(2.0, .28 * pitch),
                             sigmaY=max(2.0, .46 * pitch))
    d = max(3, int(round(flank * pitch)))
    above = np.zeros_like(smooth)
    below = np.zeros_like(smooth)
    above[d:] = smooth[:-d] - smooth[d:]
    below[:-d] = smooth[d:] - smooth[:-d]
    weak = np.minimum(above, below)
    strong = np.maximum(above, below)
    response = (.68 * np.maximum(weak, 0) + .20 * np.maximum(strong, 0)
                + .12 * np.maximum(broad - smooth, 0))
    # 大面积接近 255 的眩光中，数值上出现的微弱谷值不代表可辨纹理。
    # 仅以局部饱和比例拒识，不补猜被曝光抹掉的线。
    kx = max(7, int(round(.85 * pitch))) | 1
    ky = max(7, int(round(.65 * pitch))) | 1
    near_saturation = cv2.blur((gray >= 250).astype(np.float32), (kx, ky))
    return smooth, weak, strong, response, near_saturation


def _thresholds(local_scale: float, p: FullROIParameters) -> tuple[float, float, float]:
    """亮度纹理较弱时下降到下限；强反光时仍限制门槛上限。"""
    return (max(p.min_weak_contrast, min(2.0, .13 * local_scale)),
            max(p.min_strong_contrast, min(6.0, .32 * local_scale)),
            max(p.min_response, min(5.0, .17 * local_scale)))


def _column_candidates(
    x: int, x0: int, y0: int, pitch: float,
    smooth: np.ndarray, weak: np.ndarray, strong: np.ndarray,
    response: np.ndarray, local_scale: np.ndarray, near_saturation: np.ndarray,
    valid: np.ndarray, spacing: int, flank: int,
    parameters: FullROIParameters,
) -> tuple[list[dict], dict[str, int]]:
    """扫描一整列，不借用任何旧边界预报位置。"""
    c = x - x0
    profile = smooth[:, c]
    if len(profile) < 2 * flank + 7:
        return [], {"minima": 0, "contrast": 0, "kept": 0}
    # y 是原图向下增大的行号。±2 检查是暗谷，±0.38p 检查是两侧亮纹。
    possible = np.flatnonzero((profile[2:-2] <= profile[:-4]) &
                              (profile[2:-2] < profile[4:])) + 2
    ys = possible[(possible > flank + 2) &
                  (possible < len(profile) - flank - 3) &
                  valid[possible, c]]
    if not len(ys):
        return [], {"minima": int(len(possible)), "saturated": 0,
                    "contrast": 0, "kept": 0}
    # 亮纹饱和而沟槽仍很暗时仍有有效证据；只有谷本身也几乎泛白才拒识。
    saturated = ((near_saturation[ys, c] > parameters.max_near_saturation_fraction) &
                 (smooth[ys, c] >= parameters.min_saturated_valley_intensity))
    n_saturated = int(np.count_nonzero(saturated))
    ys = ys[~saturated]
    if not len(ys):
        return [], {"minima": int(len(possible)), "saturated": n_saturated,
                    "contrast": 0, "kept": 0}
    scores = response[ys, c]
    scales = local_scale[ys, c]
    weak_limits = np.maximum(parameters.min_weak_contrast,
                             np.minimum(2.0, .13 * scales))
    strong_limits = np.maximum(parameters.min_strong_contrast,
                               np.minimum(6.0, .32 * scales))
    response_limits = np.maximum(parameters.min_response,
                                 np.minimum(5.0, .17 * scales))
    passed = ((weak[ys, c] >= weak_limits) &
              (strong[ys, c] >= strong_limits) &
              (scores >= response_limits))
    accepted = ys[passed]
    # 同一绳的多条细股暗纹可能同时过门槛：一列中先选高响应暗谷，
    # 然后要求两条候选至少相距 0.65p。此抑制不保证物理整绳身份。
    kept: list[dict] = []
    for ly in sorted(accepted, key=lambda yy: float(response[yy, c]), reverse=True):
        if any(abs(int(ly + y0) - item["y"]) < spacing for item in kept):
            continue
        tw, ts, tr = _thresholds(float(local_scale[ly, c]), parameters)
        kept.append({
            "x": int(x), "y": int(ly + y0),
            "score": round(float(response[ly, c]), 3),
            "weak": round(float(weak[ly, c]), 3),
            "strong": round(float(strong[ly, c]), 3),
            "thresholds": [round(tw, 3), round(ts, 3), round(tr, 3)],
        })
    return sorted(kept, key=lambda item: item["y"]), {
        "minima": int(len(possible)), "saturated": n_saturated,
        "contrast": int(len(accepted)), "kept": len(kept),
    }


def _link_columns(columns: list[list[dict]], xs: list[int], pitch: float,
                  parameters: FullROIParameters) -> list[list[dict]]:
    """多个起点同时增长；每次只连接已被原图支持的相邻列采样。"""
    tracks: list[dict] = []
    for ci, candidates in enumerate(columns):
        edges: list[tuple[float, int, int]] = []
        for ti, track in enumerate(tracks):
            if ci - track["last_ci"] > parameters.max_link_skip_columns:
                continue
            points = track["points"]
            last = points[-1]
            dx = xs[ci] - last["x"]
            if len(points) >= 2:
                prev = points[-2]
                slope = float(np.clip((last["y"] - prev["y"]) /
                                      max(1, last["x"] - prev["x"]),
                                      -parameters.max_slope, parameters.max_slope))
            else:
                slope = 0.0
            target_y = last["y"] + slope * dx
            tolerance = max(2.5, parameters.max_link_distance_pitches * pitch)
            for cj, candidate in enumerate(candidates):
                error = abs(candidate["y"] - target_y)
                if error > tolerance:
                    continue
                if abs(candidate["y"] - last["y"]) > max(3.0, parameters.max_slope * dx + 2.0):
                    continue
                # 几何接近优先；同一候选不能分配给两条轨迹。
                edges.append((error + .10 * (ci - track["last_ci"] - 1), ti, cj))
        used_tracks: set[int] = set()
        used_candidates: set[int] = set()
        for _, ti, cj in sorted(edges):
            if ti in used_tracks or cj in used_candidates:
                continue
            tracks[ti]["points"].append(candidates[cj])
            tracks[ti]["last_ci"] = ci
            used_tracks.add(ti)
            used_candidates.add(cj)
        for cj, candidate in enumerate(candidates):
            if cj not in used_candidates:
                tracks.append({"points": [candidate], "last_ci": ci})
    return [item["points"] for item in tracks]


def detect_full_roi(
    image_bgr: np.ndarray,
    roi_mask: np.ndarray,
    exclusion_mask: np.ndarray | None,
    pitch_px: float,
    parameters: FullROIParameters = FullROIParameters(),
) -> tuple[np.ndarray, list[dict], dict]:
    """返回 `(黄色像素, 独立可见片段, 诊断)`。

    输入图为 BGR 原图；ROI、排除掩码与原图等大，非零即选中。
    每个 `observed_points_xy` 均为原图 `(x, y)` 整数采样点。
    `line_id` 仅是本图片段编号，绝非已验证的相邻绳顺序或材料身份。
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must be a three-channel BGR image")
    h, w = image_bgr.shape[:2]
    if roi_mask.shape != (h, w):
        raise ValueError("ROI and image dimensions differ")
    if exclusion_mask is not None and exclusion_mask.shape != (h, w):
        raise ValueError("exclusion and image dimensions differ")
    if not math.isfinite(pitch_px) or not 8 <= pitch_px <= 70:
        raise ValueError("pitch_px must be a finite optical pixel distance in [8, 70]")
    out = np.zeros((h, w), np.uint8)
    mask = roi_mask > 0
    if not np.any(mask):
        return out, [], {"status": "empty_roi", "accepted_tracks": 0}
    yy, xx = np.nonzero(mask)
    margin = max(40, int(round(2.5 * pitch_px)))
    x0 = max(0, int(xx.min()) - margin)
    x1 = min(w, int(xx.max()) + margin + 1)
    y0 = max(0, int(yy.min()) - margin)
    y1 = min(h, int(yy.max()) + margin + 1)
    bounds = (x0, y0, x1, y1)
    smooth, weak, strong, response, near_saturation = _signals(
        image_bgr, pitch_px, bounds, parameters.flank_offset_pitches)
    kernel_y = max(5, int(round(1.8 * pitch_px))) | 1
    local_scale = cv2.blur(response, (1, kernel_y))
    valid = mask[y0:y1, x0:x1].copy()
    if exclusion_mask is not None:
        radius = max(2, int(round(parameters.exclusion_guard_pitches * pitch_px)))
        forbidden = cv2.dilate((exclusion_mask[y0:y1, x0:x1] > 0).astype(np.uint8),
                               np.ones((2 * radius + 1, 1), np.uint8)) > 0
        valid &= ~forbidden
    flank = max(3, int(round(parameters.flank_offset_pitches * pitch_px)))
    step = max(4, int(round(parameters.column_step_pitches * pitch_px)))
    spacing = max(4, int(round(parameters.min_column_spacing_pitches * pitch_px)))
    # 从左到右扫描亮带的所有列，不在已知黄点附近限定搜索范围。
    xs = list(range(int(xx.min()), int(xx.max()) + 1, step))
    columns: list[list[dict]] = []
    scan_counts = {"minima": 0, "saturated": 0, "contrast": 0, "kept": 0}
    for x in xs:
        candidates, counts = _column_candidates(
            x, x0, y0, pitch_px, smooth, weak, strong, response,
            local_scale, near_saturation, valid, spacing, flank, parameters)
        columns.append(candidates)
        for key in scan_counts:
            scan_counts[key] += counts[key]
    raw_tracks = _link_columns(columns, xs, pitch_px, parameters)
    rejects = {"too_short": 0, "low_support": 0, "steep": 0, "rough": 0}
    accepted: list[dict] = []
    for points in raw_tracks:
        xvalues = np.asarray([p["x"] for p in points], np.float64)
        yvalues = np.asarray([p["y"] for p in points], np.float64)
        span = int(xvalues[-1] - xvalues[0])
        if len(points) < parameters.min_points or span < max(
                parameters.min_span_pixels, parameters.min_span_pitches * pitch_px):
            rejects["too_short"] += 1
            continue
        possible_columns = 1 + span / step
        support_fraction = len(points) / possible_columns
        if support_fraction < parameters.min_support_fraction:
            rejects["low_support"] += 1
            continue
        slope = float(np.polyfit(xvalues, yvalues, 1)[0])
        if abs(slope) > parameters.max_slope:
            rejects["steep"] += 1
            continue
        xspan = max(float(np.ptp(xvalues)), 1.0)
        xn = (xvalues - xvalues.mean()) / xspan
        quadratic = np.polyfit(xn, yvalues, 2)
        residual = float(np.median(np.abs(np.polyval(quadratic, xn) - yvalues)))
        if residual > parameters.max_quadratic_mad_pitches * pitch_px:
            rejects["rough"] += 1
            continue
        accepted.append({
            "observed_points_xy": [[int(p["x"]), int(p["y"])] for p in points],
            "median_score": round(float(np.median([p["score"] for p in points])), 3),
            "median_weak_contrast": round(float(np.median([p["weak"] for p in points])), 3),
            "median_strong_contrast": round(float(np.median([p["strong"] for p in points])), 3),
            "point_count": len(points), "span_px": span,
            "support_fraction": round(float(support_fraction), 3),
            "quadratic_mad_px": round(residual, 3),
            # 局部阈值随画面变化，保存首、中、末三处实际响应以便查漏检。
            "score_threshold_samples": [
                {"xy": [points[j]["x"], points[j]["y"]],
                 "score": points[j]["score"],
                 "weak": points[j]["weak"],
                 "strong": points[j]["strong"],
                 "thresholds": points[j]["thresholds"]}
                for j in sorted({0, len(points) // 2, len(points) - 1})
            ],
        })
    # 排序只便于审阅，不能当作全卷面连续绳号或物理身份。
    accepted.sort(key=lambda line: float(np.median(
        [p[1] for p in line["observed_points_xy"]])))
    for i, line in enumerate(accepted, 1):
        line["line_id"] = f"roi_segment_{i:04d}"
        for x, y in line["observed_points_xy"]:
            cv2.circle(out, (x, y), 1, 255, -1, cv2.LINE_8)
    out[~mask] = 0
    if exclusion_mask is not None:
        out[exclusion_mask > 0] = 0
    diagnostics = {
        "status": "completed", "algorithm": "independent_full_roi_dark_valleys_v3",
        "pitch_px": float(pitch_px), "roi_bounds_xyxy": list(bounds),
        "sampled_columns": len(xs), "sampled_step_px": step,
        "raw_candidates": scan_counts, "raw_tracks": len(raw_tracks),
        "accepted_tracks": len(accepted), "rejected_tracks": rejects,
        "observed_pixels": int(np.count_nonzero(out)),
        "interpretation": "visible dark-valley samples; no old yellow anchors or lattice identity",
    }
    return out, accepted, diagnostics
