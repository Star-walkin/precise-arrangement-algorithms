"""全卷面欠曝光 ROI 与灰度对比度拉伸，纯离线图像函数。

输入仅为原图、C4 全卷面二值掩码和绳距。低频照明场决定连续暗区，
局部残差增益只改善显示/候选对比；增强产生的纹理不是新的原图证据。
不包含活动绳、接触、绳身份或设备逻辑。
"""
from __future__ import annotations

import math

import cv2
import numpy as np


PARAMETERS = {
    "field_sigma_pitch": 8.0,
    "field_sigma_min_px": 160.0,
    "local_sigma_pitch": 2.5,
    "local_sigma_min_px": 35.0,
    "field_max_side_small_px": 1024,
    "otsu_upper_gray": 110.0,
    "whole_dark_p90_gray": 85.0,
    "illumination_target_gray": 105.0,
    "max_illumination_gain": 2.4,
    "max_contrast_gain": 3.2,
    "local_deviation_floor_gray": 2.5,
    "no_signal_deviation_gray": 1.2,
    "output_ceiling_gray": 235.0,
}


def _masked_blur_small(weighted: np.ndarray, weight: np.ndarray,
                       sigma_small: float, fallback: float) -> np.ndarray:
    """卷面内归一化模糊：法兰/图像黑边不会把卷面均值拉低。"""
    num = cv2.GaussianBlur(weighted, (0, 0), sigmaX=sigma_small,
                           sigmaY=sigma_small)
    den = cv2.GaussianBlur(weight, (0, 0), sigmaX=sigma_small,
                           sigmaY=sigma_small)
    result = np.full(num.shape, fallback, np.float32)
    np.divide(num, den, out=result, where=den > 0.01)
    return result


def _percentiles(values: np.ndarray, levels=(10, 50, 90)) -> list[float]:
    return [float(v) for v in np.percentile(values, levels)] if values.size else []


def _adjacent_vertical_stats(raw: np.ndarray, enhanced: np.ndarray,
                             dark: np.ndarray) -> dict:
    """同位置相邻灰度差，只量化对比度变化，不作识别准确率。"""
    pair = dark[1:] & dark[:-1] & (raw[1:] >= 6) & (raw[:-1] >= 6)
    pair &= (raw[1:] < 250) & (raw[:-1] < 250)
    if not np.any(pair):
        return {"paired_pixel_count": 0, "raw_abs_diff_p50_p90": None,
                "enhanced_abs_diff_p50_p90": None}
    # uint8 直接相减会溢出；升到 int16 后统计同一批位置。
    before = np.abs(raw[1:].astype(np.int16)-raw[:-1].astype(np.int16))[pair]
    after = np.abs(enhanced[1:].astype(np.int16)-enhanced[:-1].astype(np.int16))[pair]
    return {"paired_pixel_count": int(np.count_nonzero(pair)),
            "raw_abs_diff_p50_p90": _percentiles(before, (50, 90)),
            "enhanced_abs_diff_p50_p90": _percentiles(after, (50, 90))}


def process_dark_roi(image_bgr: np.ndarray, surface_mask: np.ndarray,
                     pitch_px: float) -> tuple[np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray, dict]:
    """返回 dark_mask、raw_roi、enhanced_roi、enhanced_full、diagnostics。

    坐标、尺寸与原图相同：dark_mask 为 0/255；raw_roi 是暗区原始 BGR；
    enhanced_roi 是暗区灰度三通道；enhanced_full 在暗区增强，卷面其他
    部分保留原灰度，卷面外一律为 0。灰度输出避免放大彩色噪块。
    """
    if (not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8 or
            image_bgr.ndim != 3 or image_bgr.shape[2] != 3):
        raise ValueError("image_bgr must be uint8 HxWx3 BGR")
    if (not isinstance(surface_mask, np.ndarray) or
            surface_mask.dtype != np.uint8 or
            surface_mask.shape != image_bgr.shape[:2] or
            np.any((surface_mask != 0) & (surface_mask != 255))):
        raise ValueError("surface_mask must be same-size uint8 0/255")
    pitch = float(pitch_px)
    if not math.isfinite(pitch) or not 8 <= pitch <= 70:
        raise ValueError("pitch_px must be finite and in [8,70]")

    h, w = surface_mask.shape
    surface = surface_mask != 0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dark_mask = np.zeros((h, w), np.uint8)
    raw_roi = np.zeros_like(image_bgr)
    enhanced_roi = np.zeros_like(image_bgr)
    enhanced_full = np.zeros_like(image_bgr)
    if not np.any(surface):
        return dark_mask, raw_roi, enhanced_roi, enhanced_full, {
            "status": "empty_surface", "surface_pixels": 0, "dark_pixels": 0,
            "parameters": PARAMETERS.copy(),
            "interpretation": "No surface pixels; no dark-area claim"}

    # 倍数仅决定低频场的计算尺寸；sigma 仍以原图像素为单位。
    factor = max(1, int(math.ceil(max(h, w)/PARAMETERS["field_max_side_small_px"])))
    small_size = (int(math.ceil(w/factor)), int(math.ceil(h/factor)))
    weight = cv2.resize(surface.astype(np.float32), small_size,
                        interpolation=cv2.INTER_AREA)
    gray_small_weighted = cv2.resize(gray.astype(np.float32)*surface,
                                     small_size, interpolation=cv2.INTER_AREA)
    raw_median = float(np.median(gray[surface]))
    field_sigma = max(PARAMETERS["field_sigma_pitch"]*pitch,
                      PARAMETERS["field_sigma_min_px"])
    field_small = _masked_blur_small(gray_small_weighted, weight,
                                     max(2., field_sigma/factor), raw_median)
    field = cv2.resize(field_small, (w, h), interpolation=cv2.INTER_LINEAR)
    field_p10, field_p50, field_p90 = _percentiles(field[surface])

    # Otsu 只看低频场的卷面像素，故逐根暗沟不会被当作独立暗区。
    # 上限 110 防止高曝光/几乎均匀亮面强制分成明暗两半。
    field_u8 = np.rint(np.clip(field[surface], 0, 255)).astype(np.uint8)
    otsu, _ = cv2.threshold(field_u8.reshape(-1, 1), 0, 255,
                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if field_p90 <= PARAMETERS["whole_dark_p90_gray"]:
        threshold = min(PARAMETERS["otsu_upper_gray"], field_p90+5.)
        threshold_rule = "whole_surface_low_light_p90_plus_5"
    else:
        threshold = min(PARAMETERS["otsu_upper_gray"], float(otsu))
        threshold_rule = "masked_low_frequency_otsu_capped"
    dark_small = ((field_small <= threshold) & (weight > .1)).astype(np.uint8)*255
    # 小尺度闭合仅连通照明场中的微小空洞，随后重新与 C4 卷面相交。
    kernel_side = max(3, int(round(pitch/factor))*2+1)
    kernel_side = min(kernel_side, 25)
    if kernel_side % 2 == 0:
        kernel_side += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernel_side, kernel_side))
    dark_small = cv2.morphologyEx(dark_small, cv2.MORPH_CLOSE, kernel)
    dark = (cv2.resize(dark_small, (w, h),
                       interpolation=cv2.INTER_NEAREST) > 0) & surface
    dark_mask[dark] = 255
    raw_roi[dark] = image_bgr[dark]

    # 局部均值和局部平均绝对偏差都只由卷面有效像素贡献。
    # 灰度轻微去噪后将暗区残差放大，低频照明则单独限增益补偿。
    denoised = cv2.GaussianBlur(gray, (3, 3), sigmaX=.65).astype(np.float32)
    local_sigma = max(PARAMETERS["local_sigma_pitch"]*pitch,
                      PARAMETERS["local_sigma_min_px"])
    local_mean_small = _masked_blur_small(
        cv2.resize(denoised*surface, small_size, interpolation=cv2.INTER_AREA),
        weight, max(2., local_sigma/factor), raw_median)
    local_mean = cv2.resize(local_mean_small, (w, h), interpolation=cv2.INTER_LINEAR)
    residual = denoised-local_mean
    deviation_small = _masked_blur_small(
        cv2.resize(np.abs(residual)*surface, small_size,
                   interpolation=cv2.INTER_AREA),
        weight, max(2., local_sigma/factor), 0.)
    deviation = cv2.resize(deviation_small, (w, h), interpolation=cv2.INTER_LINEAR)
    if np.any(dark):
        target_deviation = float(np.clip(np.percentile(deviation[dark], 75), 7., 12.))
    else:
        target_deviation = 7.
    contrast_gain = np.clip(target_deviation/
                            np.maximum(deviation, PARAMETERS["local_deviation_floor_gray"]),
                            1., PARAMETERS["max_contrast_gain"])
    contrast_gain[deviation < PARAMETERS["no_signal_deviation_gray"]] = 1.
    illumination_gain = np.clip(PARAMETERS["illumination_target_gray"]/
                                np.maximum(field, 18.),
                                1., PARAMETERS["max_illumination_gain"])
    improved = np.clip((local_mean + residual*contrast_gain)*illumination_gain,
                       0, PARAMETERS["output_ceiling_gray"]).astype(np.uint8)
    full_gray = gray.copy()
    full_gray[dark] = improved[dark]
    full_gray[~surface] = 0
    enhanced_full = cv2.cvtColor(full_gray, cv2.COLOR_GRAY2BGR)
    enhanced_roi[dark] = enhanced_full[dark]

    diagnostics = {
        "status": "completed", "surface_pixels": int(np.count_nonzero(surface)),
        "dark_pixels": int(np.count_nonzero(dark)),
        "dark_fraction_of_surface": float(np.mean(dark[surface])),
        "dark_threshold_rule": threshold_rule,
        "dark_threshold_gray": float(threshold),
        "masked_field_otsu_gray": float(otsu),
        "field_p10_p50_p90_gray": [field_p10, field_p50, field_p90],
        "field_sigma_px": float(field_sigma),
        "local_sigma_px": float(local_sigma),
        "downsample_factor": factor,
        "small_closing_kernel_side": kernel_side,
        "local_target_deviation_gray": target_deviation,
        "illumination_gain_p10_p50_p90": _percentiles(illumination_gain[dark]),
        "contrast_gain_p10_p50_p90": _percentiles(contrast_gain[dark]),
        "gray_p10_p50_p90_raw_dark": _percentiles(gray[dark]),
        "gray_p10_p50_p90_enhanced_dark": _percentiles(full_gray[dark]),
        "vertical_adjacent_difference": _adjacent_vertical_stats(gray, full_gray, dark),
        "parameters": PARAMETERS.copy(),
        "interpretation": "Contrast change is a display/candidate property, not new raw-image evidence or accuracy",
    }
    return dark_mask, raw_roi, enhanced_roi, enhanced_full, diagnostics


__all__ = ["PARAMETERS", "process_dark_roi"]
