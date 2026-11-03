"""D3 宽深暗沟图像能量场；不归组、不拟合、不产生新黄点。

原图负责确认真实双亮肩和沟槽宽度，D1增强图只帮助提名弱纹。
输出仍在原图像素尺度，x0/y0是score左上角在原图中的坐标。
方向bank围绕C3亮区族方向搜索局部证据，绝非暗区路径的斜率上限。
"""
from __future__ import annotations

import math

import cv2
import numpy as np


PARAMETERS = {
    "crop_margin_pitch": 2.5,
    "half_resolution_from_pitch_px": 16.,
    "normal_blur_sigma_pitch": .18,
    "tangent_span_pitch": 2.0,
    "tangent_sigma_pitch": 1.20,
    "direction_offsets_dy_dx": (-.8, -.6, -.4, -.2, 0., .2, .4, .6, .8),
    "max_bank_abs_slope": 2.5,
    "shoulder_pitch_scales": (.28, .40, .52),
    "near_shoulder_pitch": .10,
    "min_tangent_valid_fraction": .72,
    "raw_weak_min": .35,
    "raw_strong_min": .75,
    "enhanced_weak_min": .65,
    "min_longitudinal_consistency": .32,
}


def _remap(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    shape = x.shape
    result = cv2.remap(image, x.astype(np.float32), y.astype(np.float32),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return result.reshape(shape)


def _oriented_average(raw: np.ndarray, enhanced: np.ndarray,
                      valid: np.ndarray, slope: np.ndarray,
                      pitch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """沿候选切向长窗平均；ROI外像素不参加平均，防黑边造假暗谷。"""
    h, w = raw.shape
    xx = np.broadcast_to(np.arange(w, dtype=np.float32)[None, :], (h, w))
    yy = np.broadcast_to(np.arange(h, dtype=np.float32)[:, None], (h, w))
    span = PARAMETERS["tangent_span_pitch"]*pitch
    offsets = np.linspace(-span, span, 9, dtype=np.float32)
    sigma = PARAMETERS["tangent_sigma_pitch"]*pitch
    weights = np.exp(-.5*(offsets/sigma)**2).astype(np.float32)
    weights /= weights.sum()
    numerator_raw = np.zeros((h, w), np.float32)
    numerator_enh = np.zeros((h, w), np.float32)
    coverage = np.zeros((h, w), np.float32)
    for dx, weight in zip(offsets, weights):
        sx, sy = xx+dx, yy+slope*dx
        local = _remap(valid, sx, sy)
        numerator_raw += weight*_remap(raw, sx, sy)*local
        numerator_enh += weight*_remap(enhanced, sx, sy)*local
        coverage += weight*local
    denominator = np.maximum(coverage, .01)
    return numerator_raw/denominator, numerator_enh/denominator, coverage


def _median3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return a+b+c-np.minimum(np.minimum(a, b), c)-np.maximum(np.maximum(a, b), c)


def _validate(original_bgr: np.ndarray, enhanced_roi: np.ndarray,
              roi: np.ndarray, pitch_px: float, reference_shape: dict) -> float:
    if (not isinstance(original_bgr, np.ndarray) or original_bgr.ndim != 3 or
            original_bgr.shape[2] != 3 or original_bgr.dtype != np.uint8):
        raise ValueError("original_bgr must be uint8 HxWx3 BGR")
    if (not isinstance(enhanced_roi, np.ndarray) or
            enhanced_roi.shape != original_bgr.shape or enhanced_roi.dtype != np.uint8):
        raise ValueError("enhanced_roi must match original uint8 BGR dimensions")
    if (not isinstance(roi, np.ndarray) or roi.shape != original_bgr.shape[:2] or
            roi.dtype != np.uint8 or np.any((roi != 0) & (roi != 255))):
        raise ValueError("roi must be original-size uint8 binary 0/255")
    pitch = float(pitch_px)
    if not math.isfinite(pitch) or not 8 <= pitch <= 70:
        raise ValueError("pitch_px must be finite optical pixels in [8,70]")
    if not isinstance(reference_shape, dict):
        raise ValueError("reference_shape must be C3 shared shape dictionary")
    needed = ("center_x_px", "scale_x_px", "b_px_per_u", "c_px_per_u2")
    if any(key not in reference_shape for key in needed):
        raise ValueError("reference_shape needs center/scale/b/c")
    values = [float(reference_shape[key]) for key in needed]
    if not all(math.isfinite(v) for v in values) or values[1] <= 0:
        raise ValueError("reference_shape contains invalid coefficients")
    return pitch


def build_groove_field(original_bgr: np.ndarray, enhanced_roi: np.ndarray,
                       roi: np.ndarray, pitch: float,
                       reference_shape: dict) -> dict:
    """返回crop score(float32,0..1)、support(uint8,0/255)及可复核诊断。"""
    pitch = _validate(original_bgr, enhanced_roi, roi, pitch, reference_shape)
    mask = roi > 0
    if not np.any(mask):
        return {"x0": 0, "y0": 0, "score": np.zeros((0, 0), np.float32),
                "support": np.zeros((0, 0), np.uint8),
                "diagnostics": {"status": "empty_roi", "pitch_px": pitch,
                                "parameters": PARAMETERS.copy()}}
    yy, xx = np.nonzero(mask)
    h, w = mask.shape
    margin = max(16, int(round(PARAMETERS["crop_margin_pitch"]*pitch)))
    x0, y0 = max(0, int(xx.min())-margin), max(0, int(yy.min())-margin)
    x1, y1 = min(w, int(xx.max())+margin+1), min(h, int(yy.max())+margin+1)
    valid_full = mask[y0:y1, x0:x1].astype(np.float32)
    raw_full = cv2.cvtColor(original_bgr[y0:y1, x0:x1],
                            cv2.COLOR_BGR2GRAY).astype(np.float32)
    enhanced_full = cv2.cvtColor(enhanced_roi[y0:y1, x0:x1],
                                 cv2.COLOR_BGR2GRAY).astype(np.float32)
    # 已有图的绳距约20px时半分辨率足以刻画0.28p以上亮肩；小绳距用原分辨率。
    factor = .5 if pitch >= PARAMETERS["half_resolution_from_pitch_px"] else 1.
    if factor == .5:
        size = (max(2, int(round(raw_full.shape[1]*factor))),
                max(2, int(round(raw_full.shape[0]*factor))))
        raw = cv2.resize(raw_full, size, interpolation=cv2.INTER_AREA)
        enhanced = cv2.resize(enhanced_full, size, interpolation=cv2.INTER_AREA)
        valid = cv2.resize(valid_full, size, interpolation=cv2.INTER_AREA)
    else:
        raw, enhanced, valid = raw_full, enhanced_full, valid_full
    local_pitch = pitch*factor
    blur = max(.75, PARAMETERS["normal_blur_sigma_pitch"]*local_pitch)
    raw = cv2.GaussianBlur(raw, (0, 0), blur)
    enhanced = cv2.GaussianBlur(enhanced, (0, 0), blur)
    # 原图局部高频量仅用于自适应门槛；不会用绝对灰度硬删暗面。
    highpass = np.abs(raw-cv2.GaussianBlur(raw, (0, 0), max(1., .22*local_pitch)))
    noise = cv2.GaussianBlur(highpass, (0, 0), max(1.5, .42*local_pitch))
    ph, pw = raw.shape
    gx = np.broadcast_to(np.arange(pw, dtype=np.float32)[None, :], (ph, pw))
    gy = np.broadcast_to(np.arange(ph, dtype=np.float32)[:, None], (ph, pw))
    image_x = x0+gx/factor
    u = (image_x-float(reference_shape["center_x_px"])) / float(reference_shape["scale_x_px"])
    base_slope = ((float(reference_shape["b_px_per_u"])+
                   2*float(reference_shape["c_px_per_u2"])*u) /
                  float(reference_shape["scale_x_px"]))
    best_score = np.zeros((ph, pw), np.float32)
    any_support = np.zeros((ph, pw), bool)
    best_raw_prominence = np.zeros((ph, pw), np.float32)
    direction_support_count = []
    raw_limit = np.maximum(PARAMETERS["raw_weak_min"], .30*noise)
    raw_strong_limit = np.maximum(PARAMETERS["raw_strong_min"], .58*noise)
    enhanced_limit = np.maximum(PARAMETERS["enhanced_weak_min"], .45*noise)
    min_valid = PARAMETERS["min_tangent_valid_fraction"]
    for offset in PARAMETERS["direction_offsets_dy_dx"]:
        slope = np.clip(base_slope+offset,
                        -PARAMETERS["max_bank_abs_slope"],
                        PARAMETERS["max_bank_abs_slope"]).astype(np.float32)
        denominator = np.sqrt(1.+slope*slope)
        normal_x, normal_y = -slope/denominator, 1./denominator
        raw_center, enh_center, coverage = _oriented_average(
            raw, enhanced, valid, slope, local_pitch)
        raw_weak = []
        raw_strong = []
        enh_weak = []
        shoulder_coverage = []
        for scale in PARAMETERS["shoulder_pitch_scales"]:
            distance = scale*local_pitch
            ax, ay = gx+distance*normal_x, gy+distance*normal_y
            bx, by = gx-distance*normal_x, gy-distance*normal_y
            ra, rb = _remap(raw_center, ax, ay), _remap(raw_center, bx, by)
            ea, eb = _remap(enh_center, ax, ay), _remap(enh_center, bx, by)
            ca, cb = _remap(coverage, ax, ay), _remap(coverage, bx, by)
            raw_weak.append(np.minimum(ra-raw_center, rb-raw_center))
            raw_strong.append(np.maximum(ra-raw_center, rb-raw_center))
            enh_weak.append(np.minimum(ea-enh_center, eb-enh_center))
            shoulder_coverage.append(np.minimum(ca, cb))
        rw = _median3(*raw_weak)
        rs = _median3(*raw_strong)
        ew = _median3(*enh_weak)
        valid_fraction = np.minimum(coverage,
                                    _median3(*shoulder_coverage))
        # 细股纹的近邻肩常比0.4p处更亮；宽沟的肩亮度通常向外增长。
        near = PARAMETERS["near_shoulder_pitch"]*local_pitch
        na = _remap(raw_center, gx+near*normal_x, gy+near*normal_y)
        nb = _remap(raw_center, gx-near*normal_x, gy-near*normal_y)
        near_rise = np.minimum(na-raw_center, nb-raw_center)
        breadth = np.clip((raw_weak[1]-near_rise)/
                          np.maximum(np.abs(raw_weak[1]), 1.), 0., 1.)
        # 原图中同一方向前后两个位置仍须有宽沟响应，块边短脉冲降分。
        shift = .75*local_pitch
        ra = _remap(rw, gx+shift, gy+slope*shift)
        rb = _remap(rw, gx-shift, gy-slope*shift)
        continuity = np.clip(np.minimum(ra, rb)/np.maximum(rw, .5), 0., 1.)
        raw_strength = 1.-np.exp(-np.maximum(rw/raw_limit-.65, 0.)/2.)
        enh_strength = 1.-np.exp(-np.maximum(ew/enhanced_limit-.65, 0.)/2.)
        score = (.55*raw_strength+.25*enh_strength+
                 .12*continuity+.08*breadth).astype(np.float32)
        score *= np.clip(valid_fraction/min_valid, 0., 1.)
        # 支持图要求原图双肩、增强提名、长向连续与小尺度局部谷形。
        at_center = (raw_center <= np.minimum(na, nb)+.35)
        supported = ((valid > .5) & (valid_fraction >= min_valid) &
                     (rw >= raw_limit) & (rs >= raw_strong_limit) &
                     (ew >= enhanced_limit) &
                     (continuity >= PARAMETERS["min_longitudinal_consistency"]) &
                     at_center)
        any_support |= supported
        direction_support_count.append(int(np.count_nonzero(supported)))
        candidate_score = np.where(supported, score, .12*score)
        update = candidate_score > best_score
        best_score[update] = candidate_score[update]
        best_raw_prominence[update] = rw[update]
    # 半分辨率用于计算，最终输出仍逐原图像素对应；ROI外严格置零。
    if factor != 1.:
        out_size = (x1-x0, y1-y0)
        score_full = cv2.resize(best_score, out_size,
                                interpolation=cv2.INTER_LINEAR).astype(np.float32)
        support_full = cv2.resize(any_support.astype(np.uint8), out_size,
                                  interpolation=cv2.INTER_NEAREST)*255
    else:
        score_full = best_score.astype(np.float32)
        support_full = any_support.astype(np.uint8)*255
    score_full[valid_full == 0] = 0.
    support_full[valid_full == 0] = 0
    score_full = np.clip(score_full, 0., 1.).astype(np.float32)
    in_roi = score_full[valid_full > 0]
    raw_supported = best_raw_prominence[any_support]
    diagnostics = {
        "status": "completed", "crop_xyxy": [x0, y0, x1, y1],
        "input_pitch_px": pitch, "internal_scale": factor,
        "roi_pixel_count": int(np.count_nonzero(mask)),
        "support_pixel_count": int(np.count_nonzero(support_full)),
        "raw_weak_threshold_floor": PARAMETERS["raw_weak_min"],
        "raw_strong_threshold_floor": PARAMETERS["raw_strong_min"],
        "raw_threshold_noise_multipliers": [.30, .58],
        "direction_bank_relative_slopes": list(PARAMETERS["direction_offsets_dy_dx"]),
        "direction_support_counts_internal": direction_support_count,
        "score_roi_p10_p50_p90_p99": [float(v) for v in
            np.percentile(in_roi, [10, 50, 90, 99])],
        "raw_prominence_supported_p10_p50_p90": [float(v) for v in
            np.percentile(raw_supported, [10, 50, 90])] if len(raw_supported) else [],
        "parameters": PARAMETERS.copy(),
        "meaning": "image groove energy only; support is original bilateral contrast, not rope identity or new yellow observations",
    }
    return {"x0": x0, "y0": y0, "score": score_full,
            "support": support_full.astype(np.uint8), "diagnostics": diagnostics}


__all__ = ["PARAMETERS", "build_groove_field"]
