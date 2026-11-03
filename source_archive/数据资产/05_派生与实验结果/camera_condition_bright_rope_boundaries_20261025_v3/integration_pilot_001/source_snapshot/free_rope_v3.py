"""Offline free-rope occlusion evidence for the bright wound-face ROI.

This module calls the historical ``rope_detector.detect_rope`` on original BGR
pixels.  A camera semantic map supplies an automatic *proposal*, never a truth
label or clicked seed.  Only original-pixel stripe profile samples are called
supported.  The older semantic mask remains UNKNOWN where pixels do not support
the historical profile; zero output does not establish that rope is absent.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import cv2
import numpy as np


_ROOT = next(p for p in Path(__file__).resolve().parents
             if (p / "vision_geometry_lab" / "rope_detector.py").is_file())
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "vision_prototype"))
sys.path.insert(0, str(_ROOT / "数据资产" / "05_派生与实验结果" /
                       "camera_condition_bright_rope_boundaries_20261023_v2"))
from vision_geometry_lab import rope_detector  # noqa: E402
from semantic_locator import semantic_locator_for_camera  # noqa: E402
from free_rope_occlusion import estimate_free_rope_occlusion  # noqa: E402


def _proposal(image: np.ndarray, roi: np.ndarray, role: str,
              pitch: float) -> tuple[np.ndarray, tuple | None, dict]:
    """Find a long oblique automatic semantic span; preserve component evidence."""
    locator = semantic_locator_for_camera(role)
    regions = locator.predict(image)
    prob = regions.probability("incoming_rope_free_span")
    x0, y0, x1, y1 = regions.crop_xyxy
    binary = (prob >= .50).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    selected = np.zeros_like(binary)
    entries = []
    sx, sy = (x1 - x0) / prob.shape[1], (y1 - y0) / prob.shape[0]
    for k in range(1, n):
        area = int(stats[k, cv2.CC_STAT_AREA])
        if area < 30:
            continue
        yy, xx = np.nonzero(labels == k)
        xy = np.column_stack((x0 + (xx + .5) * sx, y0 + (yy + .5) * sy)).astype(np.float32)
        vx, vy, _, _ = (float(v) for v in cv2.fitLine(
            xy, cv2.DIST_HUBER, 0, .01, .01).flatten())
        if vx < 0:
            vx, vy = -vx, -vy
        angle = math.degrees(math.atan2(vy, vx))
        span = float(np.ptp(xy[:, 0]))
        mean_prob = float(np.mean(prob[yy, xx]))
        # This gate only chooses an original-pixel search proposal.  Broad
        # components remain possible here; detect_rope must verify the stripe.
        chosen = (-43.0 <= angle <= -1.0 and
                  span >= max(150.0, 6.0 * pitch) and mean_prob >= .55)
        entries.append({"id": k, "area": area, "angle_deg": round(angle, 2),
                        "x_span_px": round(span, 2),
                        "mean_probability": round(mean_prob, 3), "chosen": chosen})
        if chosen:
            selected[labels == k] = 255
    full = np.zeros(image.shape[:2], np.uint8)
    if np.any(selected):
        full[y0:y1, x0:x1] = cv2.resize(selected, (x1 - x0, y1 - y0),
                                         interpolation=cv2.INTER_NEAREST)
        ys, xs = np.nonzero(full)
        margin = int(round(max(30.0, 2.0 * pitch)))
        box = (max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
               min(image.shape[1] - 1, int(xs.max()) + margin),
               min(image.shape[0] - 1, int(ys.max()) + margin))
    else:
        box = None
    return full, box, {"semantic_components": entries,
                       "proposal_component_ids": [e["id"] for e in entries if e["chosen"]],
                       "semantic_model_path": str(locator.model_path),
                       "semantic_crop_xyxy": [int(v) for v in regions.crop_xyxy]}


def _profile_supported(image: np.ndarray, roi: np.ndarray, proposal: np.ndarray,
                       history: dict, pitch: float) -> tuple[np.ndarray, dict]:
    """Re-sample untouched pixels along the measured line, including near ROI.

    The historical routine selects a coherent observed support run and rejects
    multiple wound-face stripes.  We rasterize only neighbouring valid sample
    intervals.  No gap or geometric extrapolation becomes supported by itself.
    """
    h, w = roi.shape
    result = np.zeros((h, w), np.uint8)
    segment = history.get("observed_segment")
    if not history.get("valid") or segment is None:
        return result, {"status": "history_invalid"}
    endpoints = np.asarray(segment, np.float64)
    vector = endpoints[1] - endpoints[0]
    length = float(np.linalg.norm(vector))
    if length < 1:
        return result, {"status": "observed_segment_too_short"}
    d = vector / length
    c = np.mean(endpoints, axis=0)
    # The source detector's optical width is not a physical rope diameter.
    width = history.get("width_px")
    if width is None:
        return result, {"status": "optical_width_unavailable"}
    width = float(width)
    scale = min(1., 1536. / max(h, w))
    ys, xs = np.nonzero(proposal)
    if len(xs) < 12:
        return result, {"status": "semantic_proposal_missing"}
    # Force this second profile pass to inspect the supplied bright ROI.  If
    # the external free segment is longer, an unrestricted _profiles call
    # selects that external group again and never tests the crossing.
    overlap = (proposal > 0) & (roi > 0)
    ry, rx = np.nonzero(overlap)
    if len(rx) < 12:
        return result, {"status": "semantic_proposal_does_not_touch_roi"}
    # Projection extends only one pitch past semantic support in the ROI.
    # The original-pixel profile test below decides which points survive.
    ps = np.column_stack((rx, ry)).astype(np.float64) @ d
    projection = (float(ps.min() - pitch), float(ps.max() + pitch))
    box = (max(0, int(rx.min() - 2 * pitch)), max(0, int(ry.min() - 2 * pitch)),
           min(w - 1, int(rx.max() + 2 * pitch)),
           min(h - 1, int(ry.max() + 2 * pitch)))
    white = np.min(image, axis=2).astype(np.float32)
    sample_step = max(3., 4. / scale)
    fine = rope_detector._profiles(
        white, c, d, box, None, proposal.astype(np.float32) / 255.,
        search=max(3. / scale, 2.5 * pitch), step=.5, max_width=12. / scale,
        guided=True, sample_step=sample_step, projection_range=projection)
    if fine is None:
        return result, {"status": "original_profile_extension_failed",
                        "projection_range_px": list(projection)}
    points = fine["points"]
    widths = fine["widths"]
    order = np.argsort(points @ d)
    points, widths = points[order], widths[order]
    support_count = 0
    for i in range(len(points) - 1):
        gap = float(np.linalg.norm(points[i + 1] - points[i]))
        if gap > max(18., 2.3 * sample_step):
            continue
        local_width = float(min(widths[i], widths[i + 1], width))
        if local_width < 2.5 or local_width > max(1.4 * pitch, 36.):
            continue
        a = tuple(np.rint(points[i]).astype(int))
        b = tuple(np.rint(points[i + 1]).astype(int))
        cv2.line(result, a, b, 255, max(1, int(round(local_width))), cv2.LINE_8)
        support_count += 1
    result[roi == 0] = 0
    # Allow only a small tolerance around the actual semantic proposal.  This
    # is still an original-pixel verified stripe, never an untested long line.
    tolerance = max(3, int(round(.2 * pitch)))
    near = cv2.dilate(proposal, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1)))
    result[near == 0] = 0
    observed_projection = sorted(float(v) for v in endpoints @ d)
    sample_projection = points @ d
    outside_observed = int(np.count_nonzero(
        (sample_projection < observed_projection[0]) |
        (sample_projection > observed_projection[1])))
    return result, {"status": "original_profile_verified", "profile_samples": int(len(points)),
                    "drawn_sample_intervals": support_count,
                    "profile_support_length_px": round(float(fine["span"]), 2),
                    "samples_beyond_original_observed_segment": outside_observed,
                    "projection_range_px": list(projection),
                    "source": "historical_rope_detector._profiles_on_original_pixels"}


def estimate_occlusion(image_bgr: np.ndarray, reference_roi: np.ndarray,
                       camera_role: str, pitch_px: float
                       ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return supported/unknown 0-255 masks inside the supplied bright ROI.

    ``supported`` means that detect_rope was valid and local original-image
    stripe profiles accepted the pixels. ``unknown`` is a conservative semantic
    exclusion, not a confirmed rope. A zero in both masks is not rope absence.
    """
    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must be HxWx3 BGR")
    if reference_roi.shape != image_bgr.shape[:2]:
        raise ValueError("reference_roi dimensions must match image")
    role = camera_role.replace("_camera", "")
    if role not in {"left", "right"}:
        raise ValueError("camera_role must be left or right")
    pitch = float(pitch_px)
    if not math.isfinite(pitch) or not 6. <= pitch <= 100.:
        raise ValueError("pitch_px must be a plausible finite pixel pitch")
    roi = (reference_roi > 0).astype(np.uint8) * 255
    canonical = cv2.flip(image_bgr, 1) if role == "left" else image_bgr
    canonical_roi = cv2.flip(roi, 1) if role == "left" else roi
    proposal, box, sem_diag = _proposal(canonical, canonical_roi, role, pitch)
    if box is None:
        history = {"valid": False, "reason": "no_automatic_semantic_span",
                   "diagnostics": {}}
        history_called = False
    else:
        history = rope_detector.detect_rope(canonical, roi=box,
                                            rope_mask=proposal)
        history_called = True
    support, profile_diag = _profile_supported(canonical, canonical_roi,
                                                proposal, history, pitch)
    old_mask, old_diag = estimate_free_rope_occlusion(image_bgr, roi, role, pitch)
    old_canonical = cv2.flip(old_mask, 1) if role == "left" else old_mask
    unknown = np.where((old_canonical > 0) & (support == 0), 255, 0).astype(np.uint8)
    support[canonical_roi == 0] = 0
    unknown[canonical_roi == 0] = 0
    if role == "left":
        support, unknown = cv2.flip(support, 1), cv2.flip(unknown, 1)
    diag = {"source": "historical_detect_rope_plus_original_profile_v3",
            "camera_role": role, "mask_value_convention": "uint8_0_or_255",
            "history_module": "vision_geometry_lab.rope_detector.detect_rope",
            "history_module_called": history_called,
            "history_valid": bool(history.get("valid")),
            "history_reason": history.get("reason"),
            "history_width_px": history.get("width_px"),
            "history_width_interval_px": history.get("width_interval_px"),
            "history_angle_deg": history.get("angle_deg"),
            "history_observed_segment_canonical_xy": history.get("observed_segment"),
            "history_observed_support": history.get("observed_support"),
            "history_confidence": history.get("confidence"),
            "history_diagnostics": history.get("diagnostics", {}),
            "semantic_proposal": sem_diag, "profile_extension": profile_diag,
            "v2_fallback_mask_kind": old_diag.get("mask_kind"),
            "v2_fallback_status": old_diag.get("status"),
            "v2_fallback_pixels": int(np.count_nonzero(old_mask)),
            "supported_pixels_in_roi": int(np.count_nonzero(support)),
            "unknown_pixels_in_roi": int(np.count_nonzero(unknown)),
            "supported_meaning": "original-pixel optical stripe evidence within ROI",
            "unknown_meaning": "v2 semantic candidate without sufficient v3 pixel verification",
            "complete_free_rope_claim": False}
    if not np.count_nonzero(support):
        diag["status"] = "v2_unknown_only" if np.count_nonzero(unknown) else "not_detected"
    else:
        diag["status"] = "partial_original_profile_support"
    return support.astype(np.uint8), unknown.astype(np.uint8), diag
