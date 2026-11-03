"""D2 暗区逐列、方向自适应的可见暗沟候选；不跨列归组。

增强灰度只提名。每个返回点在原图同一坐标及沿局部斜率的邻近位置
重新检查双侧亮纹和局部谷形。局部证据是图像纹理，不证明同根材料。
所有坐标、绳距与尺度均为输入原图像素；函数不读取文件/标签/设备。
"""
from __future__ import annotations

import math

import cv2
import numpy as np


PARAMETERS = {
    "column_step_pitch": .27,
    "max_abs_slope": 1.2,
    "slope_hypothesis_offset": .35,
    "low_coherence_slope_choices": (-.85, 0., .85),
    "low_coherence_limit": .16,
    "flank_pitch_scales": (.30, .42, .55),
    "min_within_column_spacing_pitch": .42,
    "raw_longitudinal_check_pitch": .34,
    "raw_required_samples": 2,
    "max_candidate_density_per_pitch": 2.6,
}


def _remap(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    mx, my = np.asarray(x, np.float32), np.asarray(y, np.float32)
    shape = mx.shape
    if mx.ndim == 1:
        mx, my = mx.reshape(1, -1), my.reshape(1, -1)
    result = cv2.remap(image, mx, my, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)
    return result.reshape(shape)


def _local_slope(enhanced: np.ndarray, dark: np.ndarray, pitch: float
                ) -> tuple[np.ndarray, np.ndarray, dict]:
    """结构张量估计暗沟切向；相干性差时后续另试三种固定方向。"""
    h, w = enhanced.shape
    half = (max(2, int(math.ceil(w/2))), max(2, int(math.ceil(h/2))))
    small = cv2.resize(enhanced, half, interpolation=cv2.INTER_AREA)
    weight = cv2.resize(dark.astype(np.float32), half,
                        interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (3, 3), .8)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    sigma = max(3., .38*pitch)
    den = cv2.GaussianBlur(weight, (0, 0), sigma)
    def average(value):
        num = cv2.GaussianBlur(value*weight, (0, 0), sigma)
        return num/np.maximum(den, .02)
    jxx = average(gx*gx)
    jyy = average(gy*gy)
    jxy = average(gx*gy)
    angle = .5*np.arctan2(2*jxy, jxx-jyy)
    # 主梯度法向角 angle 的垂直方向是暗沟切向。
    slope_small = -np.cos(angle)/np.where(np.abs(np.sin(angle)) < .08,
                                          np.where(np.sin(angle) < 0, -.08, .08),
                                          np.sin(angle))
    slope_small = np.clip(slope_small, -PARAMETERS["max_abs_slope"],
                          PARAMETERS["max_abs_slope"]).astype(np.float32)
    coherence_small = (np.sqrt((jxx-jyy)**2+4*jxy*jxy)/
                       np.maximum(jxx+jyy, .01)).astype(np.float32)
    # 平坦区域不能因为数值上相干性高就自称方向可信。
    coherence_small[jxx+jyy < 1.] = 0.
    slope = cv2.resize(slope_small, (w, h), interpolation=cv2.INTER_LINEAR)
    coherence = cv2.resize(coherence_small, (w, h), interpolation=cv2.INTER_LINEAR)
    return slope, coherence, {"tensor_half_resolution": True,
                              "tensor_sigma_full_px": float(2*sigma),
                              "coherence_p10_p50_p90": [float(v) for v in
                                  np.percentile(coherence[dark], [10, 50, 90])]}


def _original_support(raw: np.ndarray, noise: np.ndarray,
                      x: np.ndarray, y: np.ndarray, slope: np.ndarray,
                      flank: np.ndarray, pitch: float) -> dict[str, np.ndarray]:
    """沿候选切向取三组原图剖面：弱纹理可通过，孤立块边较难通过。"""
    length = max(3., PARAMETERS["raw_longitudinal_check_pitch"]*pitch)
    den = np.sqrt(1+slope*slope)
    nx, ny = -slope/den, 1./den
    nearest_noise = _remap(noise, x, y)
    weak_limit = np.maximum(.25, .28*nearest_noise)
    strong_limit = np.maximum(.65, .55*nearest_noise)
    supports = []
    weaks = []
    strongs = []
    for shift in (-length, 0., length):
        cx, cy = x+shift, y+slope*shift
        center = _remap(raw, cx, cy)
        upper = _remap(raw, cx+flank*nx, cy+flank*ny)
        lower = _remap(raw, cx-flank*nx, cy-flank*ny)
        weak = np.minimum(upper-center, lower-center)
        strong = np.maximum(upper-center, lower-center)
        # ±2 像素的法向邻点只验谷形，不要求绝对灰度超过常数。
        near = min(2., .11*pitch)
        side_a = _remap(raw, cx+near*nx, cy+near*ny)
        side_b = _remap(raw, cx-near*nx, cy-near*ny)
        valley = center <= np.minimum(side_a, side_b)+.55
        supports.append((weak >= weak_limit) &
                        (strong >= strong_limit) & valley)
        weaks.append(weak)
        strongs.append(strong)
    count = np.sum(np.stack(supports), axis=0)
    return {"supported": supports[1] &
                         (count >= PARAMETERS["raw_required_samples"]),
            "fraction": count.astype(np.float32)/3.,
            "weak": weaks[1], "strong": strongs[1],
            "noise": nearest_noise,
            "weak_limit": weak_limit, "strong_limit": strong_limit}


def _validate(original_bgr: np.ndarray, enhanced_bgr: np.ndarray,
              dark_mask: np.ndarray, pitch_px: float) -> float:
    if (not isinstance(original_bgr, np.ndarray) or original_bgr.dtype != np.uint8 or
            original_bgr.ndim != 3 or original_bgr.shape[2] != 3):
        raise ValueError("original_bgr must be uint8 HxWx3 BGR")
    if (not isinstance(enhanced_bgr, np.ndarray) or
            enhanced_bgr.shape != original_bgr.shape or enhanced_bgr.dtype != np.uint8):
        raise ValueError("enhanced_bgr must match original uint8 BGR image")
    if (not isinstance(dark_mask, np.ndarray) or
            dark_mask.shape != original_bgr.shape[:2] or dark_mask.dtype != np.uint8 or
            np.any((dark_mask != 0) & (dark_mask != 255))):
        raise ValueError("dark_mask must be original-size uint8 0/255")
    pitch = float(pitch_px)
    if not math.isfinite(pitch) or not 8 <= pitch <= 70:
        raise ValueError("pitch_px must be finite optical pixels in [8,70]")
    return pitch


def propose_dark_points(original_bgr: np.ndarray, enhanced_bgr: np.ndarray,
                        dark_mask: np.ndarray, pitch_px: float) -> dict:
    """返回 xs、columns 与诊断；columns[i] 对应原图 x=xs[i] 的全部候选。

    每个候选至少有 x、y、score、slope、pitch、raw_support。y 为
    响应峰抛物线微调后的原图浮点像素；不在此处连接、排序成绳或拟合。
    """
    pitch = _validate(original_bgr, enhanced_bgr, dark_mask, pitch_px)
    mask = dark_mask > 0
    if not np.any(mask):
        return {"xs": [], "columns": [], "diagnostics": {
            "status": "empty_dark_roi", "pitch_px": pitch,
            "candidate_count": 0, "interpretation": "no visible dark groove inferred"}}
    yy, xx = np.nonzero(mask)
    margin = max(24, int(round(1.6*pitch)))
    h, w = mask.shape
    x0, x1 = max(0, int(xx.min())-margin), min(w, int(xx.max())+margin+1)
    y0, y1 = max(0, int(yy.min())-margin), min(h, int(yy.max())+margin+1)
    dark = mask[y0:y1, x0:x1]
    raw = cv2.cvtColor(original_bgr[y0:y1, x0:x1],
                       cv2.COLOR_BGR2GRAY).astype(np.float32)
    enhanced = cv2.cvtColor(enhanced_bgr[y0:y1, x0:x1],
                            cv2.COLOR_BGR2GRAY).astype(np.float32)
    # 仅亚像素尺度去噪；不用 B1 的固定水平大模糊，保留 ±1.2 侧缘斜线。
    raw_smooth = cv2.GaussianBlur(raw, (3, 3), .65)
    enhanced_smooth = cv2.GaussianBlur(enhanced, (3, 3), .75)
    slope_field, coherence_field, tensor_diag = _local_slope(
        enhanced_smooth, dark, pitch)
    raw_noise = cv2.GaussianBlur(
        np.abs(raw_smooth-cv2.GaussianBlur(raw_smooth, (3, 3), .9)),
        (0, 0), max(2., .22*pitch))
    step = max(4, int(round(PARAMETERS["column_step_pitch"]*pitch)))
    xs = list(range(int(xx.min()), int(xx.max())+1, step))
    local_x = np.asarray(xs, np.int32)-x0
    grid_x = np.broadcast_to(local_x[None, :], (y1-y0, len(xs))).astype(np.float32)
    grid_y = np.broadcast_to(np.arange(y1-y0, dtype=np.float32)[:, None],
                             grid_x.shape)
    base = slope_field[:, local_x]
    coherence = coherence_field[:, local_x]
    low = coherence < PARAMETERS["low_coherence_limit"]
    fallback = PARAMETERS["low_coherence_slope_choices"]
    slope_choices = [
        np.where(low, fallback[1], base),
        np.where(low, fallback[0], base-PARAMETERS["slope_hypothesis_offset"]),
        np.where(low, fallback[2], base+PARAMETERS["slope_hypothesis_offset"]),
    ]
    best_response = np.zeros_like(grid_x, np.float32)
    best_weak = np.zeros_like(grid_x, np.float32)
    best_strong = np.zeros_like(grid_x, np.float32)
    best_slope = np.zeros_like(grid_x, np.float32)
    best_flank = np.zeros_like(grid_x, np.float32)
    center = enhanced_smooth[:, local_x]
    for hypothesis in slope_choices:
        slope = np.clip(hypothesis, -PARAMETERS["max_abs_slope"],
                        PARAMETERS["max_abs_slope"]).astype(np.float32)
        den = np.sqrt(1.+slope*slope)
        nx, ny = -slope/den, 1./den
        for scale in PARAMETERS["flank_pitch_scales"]:
            d = float(scale*pitch)
            a = _remap(enhanced_smooth, grid_x+d*nx, grid_y+d*ny)
            b = _remap(enhanced_smooth, grid_x-d*nx, grid_y-d*ny)
            weak = np.minimum(a-center, b-center)
            strong = np.maximum(a-center, b-center)
            response = np.maximum(weak, 0.)*.72 + np.maximum(strong, 0.)*.28
            update = response > best_response
            best_response[update] = response[update]
            best_weak[update] = weak[update]
            best_strong[update] = strong[update]
            best_slope[update] = slope[update]
            best_flank[update] = d

    local_noise = raw_noise[:, local_x]
    # 上限不依赖画面整体亮度；下限又随局部原图噪声升高。
    response_limit = np.maximum(1.15, .90*local_noise)
    weak_limit = np.maximum(.38, .28*local_noise)
    strong_limit = np.maximum(1.05, .70*local_noise)
    peak = np.zeros(best_response.shape, bool)
    peak[2:-2] = ((best_response[2:-2] >= best_response[1:-3]) &
                  (best_response[2:-2] > best_response[3:-1]) &
                  (best_response[2:-2] >= best_response[:-4]) &
                  (best_response[2:-2] > best_response[4:]))
    enhanced_pass = (peak & dark[:, local_x] &
                     (best_response >= response_limit) &
                     (best_weak >= weak_limit) &
                     (best_strong >= strong_limit))
    rows, cols = np.nonzero(enhanced_pass)
    if not len(rows):
        return {"xs": xs, "columns": [[] for _ in xs], "diagnostics": {
            "status": "no_supported_candidates", "pitch_px": pitch,
            "sampled_columns": len(xs), "candidate_count": 0,
            "tensor": tensor_diag, "parameters": PARAMETERS.copy()}}
    # 三点抛物线定位响应峰；将原图 y 亚像素坐标限制在峰值邻域。
    upper = best_response[rows-1, cols]
    middle = best_response[rows, cols]
    lower = best_response[rows+1, cols]
    curvature = upper-2*middle+lower
    delta = np.zeros_like(middle)
    np.divide(.5*(upper-lower), curvature, out=delta,
              where=np.abs(curvature) > .01)
    delta = np.clip(delta, -.5, .5)
    px = grid_x[rows, cols]
    py = rows.astype(np.float32)+delta
    chosen_slope = best_slope[rows, cols]
    chosen_flank = best_flank[rows, cols]
    raw_check = _original_support(raw_smooth, raw_noise, px, py,
                                  chosen_slope, chosen_flank, pitch)
    supported = raw_check["supported"]
    raw_pass_count = int(np.count_nonzero(supported))
    # 分数只用于同列竞争；不是概率/准确率。
    normalized_score = (best_response[rows, cols]/
                        np.maximum(1., local_noise[rows, cols]) +
                        .75*np.maximum(raw_check["weak"], 0.)/
                        np.maximum(.6, raw_check["noise"]) +
                        .25*raw_check["fraction"])
    pool: list[list[dict]] = [[] for _ in xs]
    for i in np.flatnonzero(supported):
        column_index = int(cols[i])
        candidate = {
            "x": int(xs[column_index]), "y": round(float(y0+py[i]), 3),
            "score": round(float(normalized_score[i]), 4),
            "slope": round(float(chosen_slope[i]), 4),
            "pitch": float(pitch),
            "raw_support": round(float(raw_check["fraction"][i]), 3),
            "raw_weak": round(float(raw_check["weak"][i]), 3),
            "raw_strong": round(float(raw_check["strong"][i]), 3),
            "raw_noise": round(float(raw_check["noise"][i]), 3),
            "raw_thresholds": [round(float(raw_check["weak_limit"][i]), 3),
                               round(float(raw_check["strong_limit"][i]), 3)],
            "enhanced_weak": round(float(best_weak[rows[i], cols[i]]), 3),
            "enhanced_strong": round(float(best_strong[rows[i], cols[i]]), 3),
            "enhanced_threshold": round(float(response_limit[rows[i], cols[i]]), 3),
            "orientation_coherence": round(float(coherence[rows[i], cols[i]]), 3),
            "flank_offset_px": round(float(chosen_flank[i]), 3),
        }
        pool[column_index].append(candidate)
    spacing = max(4., PARAMETERS["min_within_column_spacing_pitch"]*pitch)
    columns: list[list[dict]] = []
    rejected_spacing = 0
    for candidates in pool:
        kept: list[dict] = []
        limit = max(1, int(math.ceil((y1-y0)/pitch*
                                       PARAMETERS["max_candidate_density_per_pitch"])))
        for item in sorted(candidates, key=lambda q:q["score"], reverse=True):
            if any(abs(item["y"]-old["y"]) < spacing for old in kept):
                rejected_spacing += 1
                continue
            kept.append(item)
            if len(kept) >= limit:
                break
        kept.sort(key=lambda q:q["y"])
        for i, item in enumerate(kept):
            gaps = [abs(item["y"]-other["y"]) for j, other in enumerate(kept)
                    if j != i and .60*pitch <= abs(item["y"]-other["y"]) <= 1.6*pitch]
            if gaps:
                item["pitch"] = round(float(min(gaps)), 3)
                item["pitch_source"] = "same_column_neighbor_gap"
            else:
                item["pitch_source"] = "input_global_pitch_fallback"
        columns.append(kept)
    slopes = [point["slope"] for col in columns for point in col]
    diagnostics = {
        "status": "completed", "algorithm": "D2_tensor_multiscale_raw_checked_columns",
        "pitch_px": pitch, "dark_pixels": int(np.count_nonzero(mask)),
        "crop_xyxy": [x0, y0, x1, y1], "sampled_columns": len(xs),
        "sampled_column_step_px": step,
        "enhanced_peak_candidates": int(len(rows)),
        "original_supported_candidates_before_spacing": raw_pass_count,
        "suppressed_within_column_spacing": rejected_spacing,
        "candidate_count": int(sum(map(len, columns))),
        "columns_with_candidates": int(sum(bool(col) for col in columns)),
        "slope_p10_p50_p90": [float(v) for v in np.percentile(slopes, [10, 50, 90])] if slopes else [],
        "tensor": tensor_diag, "parameters": PARAMETERS.copy(),
        "interpretation": "original-image supported valley proposals only; no grouping, conic fit, or material identity",
    }
    return {"xs": xs, "columns": columns, "diagnostics": diagnostics}


__all__ = ["PARAMETERS", "propose_dark_points"]
