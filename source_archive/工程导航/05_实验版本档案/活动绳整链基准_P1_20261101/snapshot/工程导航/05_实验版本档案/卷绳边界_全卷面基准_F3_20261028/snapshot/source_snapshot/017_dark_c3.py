"""整张欠曝光卷面 ROI 的 B1 黄点检测与 C3 分组圆锥拟合。

增强图仅用来提出暗沟槽位置；每个保留黄点都在原图同一坐标通过
双侧亮纹和局部暗谷复核。这里不依赖活动绳、旧 C3 曲线或明亮带种子，
也不以原图灰度 7 为硬门。视觉边界不证明同根材料或真实接触。
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


_ASSETS = Path(__file__).resolve().parent.parent
_B1_SOURCE = (_ASSETS / "camera_condition_bright_rope_boundaries_20261025_v3"
              / "full_roi_observation.py")
_C3_SOURCE = (_ASSETS / "camera_condition_grouped_conic_20261026_v6"
              / "conic_family.py")
SOURCE_DEPENDENCIES = [_B1_SOURCE, _C3_SOURCE]
_MODULES: dict[str, object] = {}
_LOCK = Lock()


def _pure_module(name: str, path: Path):
    """只加载已冻结的纯图像算法源文件，不导入设备相关模块。"""
    with _LOCK:
        if name not in _MODULES:
            spec = importlib.util.spec_from_file_location(f"dark_roi_{name}", path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"pure algorithm source unavailable: {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module  # B1 的 dataclass 解析需要模块登记。
            spec.loader.exec_module(module)
            _MODULES[name] = module
    return _MODULES[name]


def _validate(original: np.ndarray, enhanced: np.ndarray,
              dark_mask: np.ndarray, pitch_px: float) -> float:
    if (not isinstance(original, np.ndarray) or original.ndim != 3 or
            original.shape[2] != 3 or original.dtype != np.uint8):
        raise ValueError("original_bgr must be uint8 HxWx3 BGR")
    if (not isinstance(enhanced, np.ndarray) or enhanced.shape != original.shape or
            enhanced.dtype != np.uint8):
        raise ValueError("enhanced_full_bgr must match original BGR image")
    if (not isinstance(dark_mask, np.ndarray) or
            dark_mask.shape != original.shape[:2] or dark_mask.dtype != np.uint8 or
            np.any((dark_mask != 0) & (dark_mask != 255))):
        raise ValueError("dark_mask must be same-size uint8 0/255")
    pitch = float(pitch_px)
    if not math.isfinite(pitch) or not 8 <= pitch <= 70:
        raise ValueError("pitch_px must be a finite optical spacing in [8,70]")
    return pitch


def _recheck_original(original_bgr: np.ndarray, dark_mask: np.ndarray,
                      proposed_lines: list[dict], pitch: float
                      ) -> tuple[list[dict], dict]:
    """低亮度也可保留；仅拒绝原图确无双侧暗谷形状的增强假点。"""
    yy, xx = np.nonzero(dark_mask)
    if len(xx) == 0 or not proposed_lines:
        return [], {"retained_points": 0, "retained_fragments": 0,
                    "low_gray_retained_points": 0, "rejected": {}}
    h, w = dark_mask.shape
    margin = max(40, int(round(2.5 * pitch)))
    x0, x1 = max(0, int(xx.min()) - margin), min(w, int(xx.max()) + margin + 1)
    y0, y1 = max(0, int(yy.min()) - margin), min(h, int(yy.max()) + margin + 1)
    gray_u8 = cv2.cvtColor(original_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    gray = gray_u8.astype(np.float32)
    # 与 B1 相同的绳向/跨绳平滑尺度；不从增强图取原始可信度。
    smooth = cv2.GaussianBlur(gray, (0, 0),
                              sigmaX=max(5., .58 * pitch),
                              sigmaY=max(1., .10 * pitch))
    flank = max(3, int(round(.38 * pitch)))
    rejected: Counter[str] = Counter()
    kept_lines = []
    dark_raw_count = 0
    for source in proposed_lines:
        points = []
        evidence = []
        line_dark_raw_count = 0
        for xy in source.get("observed_points_xy", []):
            x, y = map(int, xy)
            lx, ly = x - x0, y - y0
            if (not 0 <= x < w or not 0 <= y < h or dark_mask[y, x] == 0 or
                    not 2 <= lx < smooth.shape[1] - 2 or
                    not flank + 3 <= ly < smooth.shape[0] - flank - 3):
                rejected["outside_dark_roi_or_profile_margin"] += 1
                continue
            valley = float(smooth[ly, lx])
            above = float(smooth[ly - flank, lx] - valley)
            below = float(smooth[ly + flank, lx] - valley)
            weak, strong = min(above, below), max(above, below)
            # 允许非常暗的真实谷值：只需原图两侧可复测的弱正对比。
            # 绝对灰度不入条件，弱对比阈值低于 A2 的整图自由绳试验。
            if weak < .35 or strong < .70:
                rejected["original_two_flank_shape_absent"] += 1
                continue
            if valley > min(float(smooth[ly - 2, lx]),
                            float(smooth[ly + 2, lx])) + .5:
                rejected["original_center_not_local_valley"] += 1
                continue
            raw_gray = int(gray_u8[ly, lx])
            if raw_gray < 7:
                line_dark_raw_count += 1
            points.append([x, y])
            evidence.append({"xy": [x, y], "original_gray": raw_gray,
                             "original_weak_contrast": round(weak, 3),
                             "original_strong_contrast": round(strong, 3)})
        # B1 的轨迹门已在增强图运行。这里不重设长度门；仅完全无点或
        # 单点无法由 C3 拟合，记录拒识，其余交由原 C3 全点规则裁决。
        if len(points) < 2 or len({xy[0] for xy in points}) < 2:
            rejected["under_two_distinct_original_columns"] += len(points)
            continue
        line = dict(source)
        line["line_id"] = f"dark_{source['line_id']}"
        line["observed_points_xy"] = points
        line["original_point_evidence"] = evidence
        line["original_image_rechecked"] = True
        line["point_count"] = len(points)
        line["span_px"] = points[-1][0] - points[0][0]
        kept_lines.append(line)
        dark_raw_count += line_dark_raw_count
    return kept_lines, {"retained_points": sum(len(x["observed_points_xy"])
                                                for x in kept_lines),
                        "retained_fragments": len(kept_lines),
                        "low_gray_retained_points": dark_raw_count,
                        "rejected": dict(rejected),
                        "original_shape_thresholds": {"minimum_weak_contrast": .35,
                                                      "minimum_strong_contrast": .70,
                                                      "absolute_gray_floor": None}}


def detect_dark_c3(original_bgr: np.ndarray, enhanced_full_bgr: np.ndarray,
                   dark_mask: np.ndarray, pitch_px: float) -> dict:
    """整张暗 ROI 上运行原 B1→原 C3；全部坐标保持原图分辨率。"""
    pitch = _validate(original_bgr, enhanced_full_bgr, dark_mask, pitch_px)
    b1 = _pure_module("b1_full_roi", _B1_SOURCE)
    c3 = _pure_module("c3_grouped_conic", _C3_SOURCE)
    exclusion = np.zeros(dark_mask.shape, np.uint8)
    enhanced_mask, proposals, b1_diag = b1.detect_full_roi(
        enhanced_full_bgr, dark_mask, exclusion, pitch)
    lines, recheck = _recheck_original(original_bgr, dark_mask, proposals, pitch)
    observed = np.zeros(dark_mask.shape, np.uint8)
    for line in lines:
        for x, y in line["observed_points_xy"]:
            cv2.circle(observed, (x, y), 1, 255, -1, cv2.LINE_8)
    observed[dark_mask == 0] = 0
    fit = c3.fit_conic_family(lines, pitch, dark_mask, exclusion)
    return {"observed_mask": observed, "smooth_mask": fit["smooth_mask"],
            "fit": fit, "observed_lines": lines,
            "diagnostics": {
                "status": "empty_dark_roi" if not np.any(dark_mask) else "completed",
                "roi_pixels": int(np.count_nonzero(dark_mask)),
                "pitch_px": pitch,
                "enhanced_proposal_points": sum(len(x["observed_points_xy"])
                                                for x in proposals),
                "original_rechecked_points": recheck["retained_points"],
                "original_rechecked_fragments": recheck["retained_fragments"],
                "original_low_gray_retained_points": recheck["low_gray_retained_points"],
                "new_c3_accepted_curves": len(fit["curves"]),
                "new_c3_rejected_groups": len(fit["rejected"]),
                "b1": b1_diag, "original_recheck": recheck,
                "c3": fit["diagnostics"],
                "roi_scope": "entire_given_dark_mask_not_free_rope_corridor",
                "old_c3_used_as_seed": False,
                "free_rope_required": False,
                "physical_rope_identity_confirmed": False,
                "source_dependencies": [str(p) for p in SOURCE_DEPENDENCIES]}}
