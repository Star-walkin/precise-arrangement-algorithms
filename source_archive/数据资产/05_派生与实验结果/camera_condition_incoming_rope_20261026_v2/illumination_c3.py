"""离线卷面照明对照与局部 B1→C3 边界重建。

增强只帮助提出候选。黄色观测点必须在未增强原图具有可复测的暗谷
和双侧亮纹；C3 对这些新点重新分组，并以每组全部点拟合单一抛物线。
旧 C3 曲线仅供计数对照，不给新点或新模型作锚。这里不判断同根、接触或 t1。
"""
from __future__ import annotations

from collections import Counter
import importlib.util
import math
from pathlib import Path
import sys
from threading import Lock

import cv2
import numpy as np


_ASSET_ROOT = Path(__file__).resolve().parent.parent
_B1_FILE = (_ASSET_ROOT / "camera_condition_bright_rope_boundaries_20261025_v3"
            / "full_roi_observation.py")
_C3_FILE = (_ASSET_ROOT / "camera_condition_grouped_conic_20261026_v6"
            / "conic_family.py")
SOURCE_DEPENDENCIES = [_B1_FILE, _C3_FILE]
_CACHE: dict[str, object] = {}
_IMPORT_LOCK = Lock()


def _module(key: str, path: Path):
    """只动态导入纯算法文件；不导入相机或设备启动代码。"""
    with _IMPORT_LOCK:
        if key not in _CACHE:
            spec = importlib.util.spec_from_file_location(f"incoming_v2_{key}", path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot load pure algorithm module: {path}")
            loaded = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = loaded  # dataclass 需要模块名可解析。
            spec.loader.exec_module(loaded)
            _CACHE[key] = loaded
    return _CACHE[key]


def _check(image_bgr: np.ndarray, surface_mask: np.ndarray,
           pitch_px: float) -> tuple[np.ndarray, np.ndarray, float]:
    if (not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or
            image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8):
        raise ValueError("image_bgr must be uint8 HxWx3 BGR")
    if (not isinstance(surface_mask, np.ndarray) or
            surface_mask.shape != image_bgr.shape[:2] or
            surface_mask.dtype != np.uint8 or
            np.any((surface_mask != 0) & (surface_mask != 255))):
        raise ValueError("surface_mask must be same-size uint8 0/255")
    pitch = float(pitch_px)
    if not math.isfinite(pitch) or not 8 <= pitch <= 70:
        raise ValueError("pitch_px must be finite and between 8 and 70")
    return image_bgr, surface_mask > 0, pitch


def _raw_flags(image_bgr: np.ndarray, surface: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    low = (gray < 7) & surface
    saturated = (gray >= 250) & surface
    valid = surface & ~low & ~saturated
    return gray, low, saturated, valid


def normalize_surface(image_bgr: np.ndarray, surface_mask: np.ndarray,
                      pitch_px: float, mode: str = "raw") -> dict:
    """固定四种单独对照：原图、gamma、全局百分位、平滑照明场补偿。

    `gain_map` 是灰度等效输出/输入比；百分位带偏移，故并非该模式
    的完整线性变换。`valid_mask` 由原图灰度定义，不会因增强而转正。
    """
    image, surface, pitch = _check(image_bgr, surface_mask, pitch_px)
    modes = {"raw", "global_gamma", "gamma_070", "percentile_2_98", "illumination"}
    if mode not in modes:
        raise ValueError(f"mode must be one of {sorted(modes)}")
    applied_mode = "gamma_070" if mode == "global_gamma" else mode
    gray, low, saturated, valid = _raw_flags(image, surface)
    enhanced = image.copy()
    extra: dict = {}
    if not np.any(surface):
        return {"enhanced_bgr": enhanced,
                "gain_map": np.ones(gray.shape, np.float32),
                "valid_mask": np.zeros(gray.shape, np.uint8),
                "raw_low_light_mask": np.zeros(gray.shape, np.uint8),
                "raw_near_saturation_mask": np.zeros(gray.shape, np.uint8),
                "diagnostics": {"mode": mode, "status": "empty_surface", "surface_pixels": 0}}
    if applied_mode == "gamma_070":
        # 固定 LUT 而非逐图选 gamma；大亮面不能用调参反向证明暗纹。
        lut = np.rint(255 * (np.arange(256, dtype=np.float32) / 255) ** .70).astype(np.uint8)
        enhanced = cv2.LUT(image, lut)
        extra = {"gamma": .70}
    elif applied_mode == "percentile_2_98":
        p2, p98 = np.percentile(gray[surface], [2., 98.])
        span = max(float(p98 - p2), 20.)
        alpha = min(4., 225. / span)
        beta = 15. - alpha * float(p2)
        enhanced = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        extra = {"masked_gray_p2": float(p2), "masked_gray_p98": float(p98),
                 "linear_alpha": float(alpha), "linear_beta": float(beta),
                 "max_alpha": 4.}
    elif applied_mode == "illumination":
        # 先下采样再做遮罩归一化卷积，尺度至少为 8p，避免把一根根绳纹
        # 当作照明场。卷面外的金色法兰不进入分子和分母。
        h, w = gray.shape
        reduction = max(1, int(math.ceil(max(h, w) / 900)))
        small_size = (max(1, int(math.ceil(w / reduction))),
                      max(1, int(math.ceil(h / reduction))))
        weight_small = cv2.resize(surface.astype(np.float32), small_size,
                                  interpolation=cv2.INTER_AREA)
        weighted_small = cv2.resize(gray.astype(np.float32) * surface,
                                    small_size, interpolation=cv2.INTER_AREA)
        sigma_full = max(8. * pitch, 160.)
        sigma_small = max(2., sigma_full / reduction)
        numerator = cv2.GaussianBlur(weighted_small, (0, 0), sigma_small)
        denominator = cv2.GaussianBlur(weight_small, (0, 0), sigma_small)
        fallback = float(np.median(gray[surface]))
        field_small = np.full(denominator.shape, fallback, np.float32)
        np.divide(numerator, denominator, out=field_small, where=denominator > .015)
        field = cv2.resize(field_small, (w, h), interpolation=cv2.INTER_LINEAR)
        target = max(95., float(np.median(field[surface])))
        applied_gain = np.clip(target / np.maximum(field, 12.), .70, 3.5).astype(np.float32)
        enhanced = np.clip(image.astype(np.float32) * applied_gain[:, :, None],
                           0, 255).astype(np.uint8)
        extra = {"field_sigma_px": float(sigma_full), "downsample_factor": reduction,
                 "target_gray": float(target), "max_gain": 3.5,
                 "min_gain": .70,
                 "field_p10_p50_p90": [float(x) for x in
                                          np.percentile(field[surface], [10, 50, 90])]}
    # gain_map 只用于报告照明改变量；检测仍将图像与原图分别复测。
    enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    gain_map = (enhanced_gray.astype(np.float32) /
                np.maximum(gray.astype(np.float32), 1.)).astype(np.float32)
    gain_map[~surface] = 1.
    return {"enhanced_bgr": enhanced,
            "gain_map": gain_map,
            "valid_mask": (valid * 255).astype(np.uint8),
            "raw_low_light_mask": (low * 255).astype(np.uint8),
            "raw_near_saturation_mask": (saturated * 255).astype(np.uint8),
            "diagnostics": {"mode": mode, "applied_mode": applied_mode,
                            "status": "completed",
                            "surface_pixels": int(np.count_nonzero(surface)),
                            "valid_raw_pixels": int(np.count_nonzero(valid)),
                            "raw_low_light_pixels": int(np.count_nonzero(low)),
                            "raw_near_saturation_pixels": int(np.count_nonzero(saturated)),
                            "effective_gain_p10_p50_p90": [float(x) for x in
                                np.percentile(gain_map[surface], [10, 50, 90])],
                            **extra}}


def _free_geometry(free_segment: dict, pitch: float,
                   shape: tuple[int, int]) -> tuple[np.ndarray | None, dict]:
    """选择与已观测直段相接且指向卷面的端点，仅限定处理范围。"""
    if not isinstance(free_segment, dict) or free_segment.get("status") != "observed":
        return None, {"status": "free_segment_not_observed"}
    segment = np.asarray(free_segment.get("observed_segment_xy"), float)
    direction = np.asarray(free_segment.get("toward_surface_direction_xy"), float)
    width = free_segment.get("width_px")
    if (segment.shape != (2, 2) or direction.shape != (2,) or
            not np.isfinite(segment).all() or not np.isfinite(direction).all() or
            width is None or not math.isfinite(float(width)) or float(width) < 3):
        return None, {"status": "free_geometry_invalid"}
    norm = float(np.linalg.norm(direction))
    if norm < .5:
        return None, {"status": "free_direction_ambiguous"}
    direction /= norm
    entry = segment[int(np.argmax(segment @ direction))]
    half_width = max(14. * pitch, 7. * float(width))
    # 直线射线横贯整幅图；SAM 交集自行裁切，不只搜入口邻域。
    horizon = float(math.hypot(*shape))
    before = min(150., 6. * float(width))
    start = entry - before * direction
    end = entry + horizon * direction
    mask = np.zeros(shape, np.uint8)
    cv2.line(mask, tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int)),
             255, max(3, int(round(2 * half_width))), cv2.LINE_8)
    return mask, {"status": "completed", "entry_xy": entry.tolist(),
                  "toward_surface_direction_xy": direction.tolist(),
                  "corridor_half_width_px": float(half_width),
                  "forward_budget_px": float(horizon),
                  "backward_budget_px": float(before),
                  "free_optical_width_px": float(width)}


def _raw_recheck(image_bgr: np.ndarray, lines: list[dict], pitch: float,
                 valid: np.ndarray, surface: np.ndarray,
                 exclusion: np.ndarray) -> tuple[list[dict], dict]:
    """在原图相同坐标复测暗谷、两侧亮纹和原始曝光状态。"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (0, 0),
                              sigmaX=max(5., .58 * pitch),
                              sigmaY=max(1., .10 * pitch))
    h, w = gray.shape
    flank = max(3, int(round(.38 * pitch)))
    reasons: Counter[str] = Counter()
    selected = []
    for line in lines:
        kept = []
        evidence = []
        for item in line.get("observed_points_xy", []):
            x, y = map(int, item)
            if not (0 <= x < w and flank + 3 <= y < h - flank - 3 and
                    surface[y, x] and not exclusion[y, x]):
                reasons["outside_valid_surface_or_free_exclusion"] += 1
                continue
            if not valid[y, x]:
                reasons["raw_low_or_near_saturated_center"] += 1
                continue
            valley = float(smooth[y, x])
            upper = float(smooth[y - flank, x]) - valley
            lower = float(smooth[y + flank, x]) - valley
            weak = min(upper, lower)
            strong = max(upper, lower)
            # 只要求原图存在可复测的双侧正对比；更强的跨度、间距与走势门
            # 已在 B1 链中完成。平滑增强可提高提名率，却不能凭空造点。
            if weak < .75 or strong < 1.35:
                reasons["raw_two_flank_contrast_not_reproducible"] += 1
                continue
            if valley > min(float(smooth[y - 2, x]), float(smooth[y + 2, x])) + .5:
                reasons["raw_center_not_local_valley"] += 1
                continue
            kept.append([x, y])
            evidence.append({"xy": [x, y], "raw_valley_gray": round(valley, 3),
                             "raw_upper_contrast": round(upper, 3),
                             "raw_lower_contrast": round(lower, 3),
                             "raw_weak_contrast": round(weak, 3)})
        if len(kept) < 8 or kept[-1][0] - kept[0][0] < max(60., 3. * pitch):
            reasons["fragment_insufficient_after_original_recheck"] += len(kept)
            continue
        selected.append({**line, "line_id": f"enh_{line['line_id']}",
                         "observed_points_xy": kept,
                         "raw_point_evidence": evidence,
                         "point_count": len(kept),
                         "span_px": int(kept[-1][0] - kept[0][0]),
                         "original_image_rechecked": True})
    return selected, {"rejected_points_by_reason": dict(reasons),
                      "retained_fragments": len(selected),
                      "retained_points": sum(len(line["observed_points_xy"]) for line in selected)}


def extract_c3_boundaries(image_bgr: np.ndarray, enhanced_bgr: np.ndarray,
                          surface_mask: np.ndarray, c3_fit: dict, pitch_px: float,
                          free_segment: dict, *, normalization: dict | None = None,
                          exclusion_mask: np.ndarray | None = None) -> dict:
    """增强图重跑完整 B1 检出→原图复核→C3 归组及全点单模型拟合。

    分析走廊沿自由绳入卷方向，避免在无关卷面耗费平方级分组时间；
    走廊仍从每列独立检索多条旧绳暗沟，不能依赖旧 C3 线的位置提名。
    """
    image, surface, pitch = _check(image_bgr, surface_mask, pitch_px)
    if (not isinstance(enhanced_bgr, np.ndarray) or
            enhanced_bgr.shape != image.shape or enhanced_bgr.dtype != np.uint8):
        raise ValueError("enhanced_bgr must match original uint8 BGR shape")
    if exclusion_mask is None:
        external_exclusion = np.zeros(surface.shape, np.uint8)
    elif (not isinstance(exclusion_mask, np.ndarray) or
          exclusion_mask.shape != surface.shape or exclusion_mask.dtype != np.uint8 or
          np.any((exclusion_mask != 0) & (exclusion_mask != 255))):
        raise ValueError("exclusion_mask must be same-size uint8 0/255")
    else:
        external_exclusion = exclusion_mask
    _, _, _, raw_valid = _raw_flags(image, surface)
    if normalization is None:
        normalization_diag = {"status": "raw_flags_recomputed"}
    else:
        supplied_valid = np.asarray(normalization["valid_mask"]) > 0
        if (supplied_valid.shape != surface.shape or
                not np.array_equal(supplied_valid, raw_valid)):
            raise ValueError("normalization.valid_mask must equal original-image flags")
        normalization_diag = normalization.get("diagnostics", {})
    corridor, scope = _free_geometry(free_segment, pitch, surface.shape)
    analysis = np.zeros(surface.shape, np.uint8)
    exclusion = np.zeros(surface.shape, np.uint8)
    observed = np.zeros(surface.shape, np.uint8)
    original_supported = np.zeros(surface.shape, np.uint8)
    old_count = (len(c3_fit.get("curves", [])) if isinstance(c3_fit, dict) else None)
    empty_fit = {"model": None, "groups": [], "curves": [],
                 "smooth_mask": np.zeros(surface.shape, np.uint8),
                 "rejected": [], "diagnostics": {"status": "no_observed_points",
                                                  "input_point_count": 0,
                                                  "grouped_point_count": 0,
                                                  "accepted_group_count": 0}}
    empty = {"analysis_mask": analysis, "exclusion_mask": exclusion,
             "observed_mask": observed, "raw_supported_mask": original_supported,
             "observed_lines": [], "fit": empty_fit,
             "metrics": {"enhanced_proposed_points": 0, "raw_supported_points": 0,
                         "new_c3_accepted_curves": 0, "old_c3_curves_for_count_only": old_count},
             "diagnostics": {"status": scope["status"], "scope": scope,
                             "normalization": normalization_diag,
                             "old_c3_used_as_seed": False}}
    if corridor is None:
        return empty
    analysis[surface & (corridor > 0)] = 255
    if not np.any(analysis):
        empty["diagnostics"]["status"] = "empty_surface_in_free_corridor"
        return empty
    exclusion[(external_exclusion > 0) & (analysis > 0)] = 255
    # 自由直段在卷面上的前向投影只用于拒识潜在遮挡像素；它不是接触或
    # 绳身份的证据。多根旧绳仍可由两侧可见片段在 C3 中关联。
    entry = np.asarray(scope["entry_xy"], float)
    direction = np.asarray(scope["toward_surface_direction_xy"], float)
    width = scope["free_optical_width_px"]
    forward = scope["forward_budget_px"]
    p0 = entry - 2. * width * direction
    p1 = entry + forward * direction
    free_guard = np.zeros(surface.shape, np.uint8)
    cv2.line(free_guard, tuple(np.rint(p0).astype(int)),
             tuple(np.rint(p1).astype(int)), 255,
             max(3, int(round(1.8 * width))), cv2.LINE_8)
    exclusion[(free_guard > 0) & (analysis > 0)] = 255
    b1 = _module("b1_full_roi", _B1_FILE)
    c3 = _module("c3_grouped_conic", _C3_FILE)
    proposed_mask, proposed_lines, b1_diag = b1.detect_full_roi(
        enhanced_bgr, analysis, exclusion, pitch)
    checked_lines, raw_diag = _raw_recheck(image, proposed_lines, pitch,
                                           raw_valid, analysis > 0, exclusion > 0)
    for line in checked_lines:
        for x, y in line["observed_points_xy"]:
            observed[y, x] = 255
            original_supported[y, x] = 255
    fit = c3.fit_conic_family(checked_lines, pitch, analysis, exclusion)
    return {"analysis_mask": analysis, "exclusion_mask": exclusion,
            "observed_mask": observed, "raw_supported_mask": original_supported,
            "observed_lines": checked_lines, "fit": fit,
            "metrics": {"enhanced_proposed_points": sum(len(line["observed_points_xy"])
                                                    for line in proposed_lines),
                        "raw_supported_points": sum(len(line["observed_points_xy"])
                                                    for line in checked_lines),
                        "proposed_fragments": len(proposed_lines),
                        "rechecked_fragments": len(checked_lines),
                        "new_c3_accepted_curves": len(fit.get("curves", [])),
                        "new_c3_rejected_groups": len(fit.get("rejected", [])),
                        "old_c3_curves_for_count_only": old_count,
                        "analysis_pixels": int(np.count_nonzero(analysis)),
                        "free_projection_exclusion_pixels": int(np.count_nonzero(exclusion))},
            "diagnostics": {"status": "completed", "scope": scope,
                            "normalization": normalization_diag,
                            "b1": b1_diag, "raw_recheck": raw_diag,
                            "c3": fit.get("diagnostics", {}),
                            "old_c3_used_as_seed": False,
                            "free_projection_is_contact_evidence": False,
                            "new_curve_is_material_identity": False}}
