"""Offline candidate mask for a visible incoming rope crossing a bright wound face.

The camera-specific semantic model provides proposals.  A long, oblique,
confident component must overlap the accepted bright ROI before it is used.
Nearby fragments are joined only when their direction and position agree.
The result is an occlusion *candidate*, not an identity or contact label.
"""
from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
import sys

import cv2
import numpy as np


_PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "vision_prototype" / "semantic_locator.py").is_file()
)
sys.path.insert(0, str(_PROJECT_ROOT / "vision_prototype"))
from semantic_locator import semantic_locator_for_camera  # noqa: E402


@lru_cache(maxsize=2)
def _locator(role: str):
    return semantic_locator_for_camera(role)


def _component_candidates(prob: np.ndarray, low_roi: np.ndarray,
                          crop: tuple[int, int, int, int], pitch: float
                          ) -> list[dict]:
    high = (prob >= 0.50).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(high, 8)
    x0, y0, x1, y1 = crop
    sx = (x1 - x0) / prob.shape[1]
    sy = (y1 - y0) / prob.shape[0]
    candidates = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < 30:
            continue
        ys, xs = np.nonzero(labels == index)
        xx = x0 + (xs.astype(np.float32) + .5) * sx
        yy = y0 + (ys.astype(np.float32) + .5) * sy
        points = np.column_stack((xx, yy))
        vx, vy, cx, cy = (float(v) for v in cv2.fitLine(
            points, cv2.DIST_HUBER, 0, .01, .01).flatten())
        if vx < 0:
            vx, vy = -vx, -vy
        angle = math.degrees(math.atan2(vy, vx))
        xspan = float(np.ptp(xx))
        residual = np.abs(-vy * (xx - cx) + vx * (yy - cy))
        median_residual = float(np.median(residual))
        width90 = float(np.percentile(residual, 90))
        roi_overlap = int(np.count_nonzero(low_roi[ys, xs]))
        mean_probability = float(np.mean(prob[ys, xs]))
        eligible = bool(
            -40.0 <= angle <= -1.0
            and xspan >= max(100.0, 0.07 * (x1 - x0))
            and median_residual <= max(8.0, 0.55 * pitch)
            and width90 <= max(16.0, 1.3 * pitch)
            and mean_probability >= .57
        )
        # A short fragment cannot qualify on its own. It may only bridge to
        # an independently long semantic rope outside the current ROI.
        fragment = bool(
            -40.0 <= angle <= -3.0
            and xspan >= max(90.0, 4.5 * pitch)
            and median_residual <= max(8.0, .55 * pitch)
            and width90 <= max(16.0, 1.3 * pitch)
            and mean_probability >= .54
        )
        candidates.append({
            "id": int(index), "area": area, "angle_deg": angle,
            "x_min": float(xx.min()), "x_max": float(xx.max()),
            "x_span_px": xspan, "roi_overlap_low_pixels": roi_overlap,
            "mean_probability": mean_probability,
            "median_line_residual_px": median_residual,
            "width90_px": width90, "eligible_geometry": eligible,
            "fragment_geometry": fragment,
            "points": points, "vx": vx, "vy": vy, "cx": cx, "cy": cy,
        })
    return candidates


def _x_gap(a: dict, b: dict) -> float:
    return max(0.0, a["x_min"] - b["x_max"], b["x_min"] - a["x_max"])


def _baseline_guard(prob: np.ndarray, low_roi: np.ndarray,
                    crop: tuple[int, int, int, int],
                    source_shape: tuple[int, int], pitch: float,
                    allowed_ids: set[int] | None = None
                    ) -> tuple[np.ndarray, list[int]]:
    """Preserve the previously reviewed conservative semantic guard."""
    high = (prob >= .50).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(high, 8)
    selected = np.zeros_like(high)
    ids = []
    for index in range(1, count):
        if allowed_ids is not None and index not in allowed_ids:
            continue
        if int(stats[index, cv2.CC_STAT_AREA]) < 30:
            continue
        ys, xs = np.nonzero(labels == index)
        if len(xs) < 3:
            continue
        covariance = np.cov(np.stack((xs, ys)).astype(np.float64))
        eigen = np.linalg.eigvalsh(covariance)
        length = 4.0 * math.sqrt(max(float(eigen[1]), 0.0))
        elongation = math.sqrt(float(eigen[1]) / max(float(eigen[0]), .01))
        overlap = int(np.count_nonzero(low_roi[ys, xs]))
        if length < 18 or elongation < 4.0 or overlap < 6:
            continue
        selected[labels == index] = 1
        ids.append(index)
    h, w = source_shape
    x0, y0, x1, y1 = crop
    full = np.zeros((h, w), np.uint8)
    full[y0:y1, x0:x1] = cv2.resize(selected, (x1 - x0, y1 - y0),
                                     interpolation=cv2.INTER_NEAREST)
    dilation = max(3, int(round(.35 * pitch)))
    guard = cv2.dilate(full, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilation + 1, 2 * dilation + 1))) > 0
    return guard, ids


def _compatible(candidate: dict, group: list[dict], pitch: float) -> bool:
    if not (candidate["eligible_geometry"] or candidate["fragment_geometry"]):
        return False
    for item in group:
        if abs(candidate["angle_deg"] - item["angle_deg"]) > 5.0:
            continue
        if _x_gap(candidate, item) > max(320.0, 16.0 * pitch):
            continue
        points = candidate["points"]
        dist = np.abs(-item["vy"] * (points[:, 0] - item["cx"]) +
                      item["vx"] * (points[:, 1] - item["cy"]))
        if float(np.median(dist)) <= max(14.0, 0.95 * pitch):
            return True
    return False


def _candidate_groups(candidates: list[dict], pitch: float) -> list[list[dict]]:
    """Build reproducible collinear groups without assuming the largest blob is rope."""
    eligible = [item for item in candidates if
                item["eligible_geometry"] or item["fragment_geometry"]]
    groups = []
    seen = set()
    for seed in eligible:
        group = [seed]
        changed = True
        while changed:
            changed = False
            for item in eligible:
                if any(item is chosen for chosen in group):
                    continue
                if _compatible(item, group, pitch):
                    group.append(item)
                    changed = True
        group.sort(key=lambda item: item["x_min"])
        key = tuple(sorted(item["id"] for item in group))
        if key not in seen:
            groups.append(group)
            seen.add(key)
    return groups


def _ambiguous_long_exclusion_components(candidates: list[dict],
                                         pitch: float) -> list[int]:
    """Long oblique but broad semantic spans: exclude as UNKNOWN, not rope."""
    return [int(item["id"]) for item in candidates if
            -35.0 <= item["angle_deg"] <= -3.0
            and item["x_span_px"] >= max(600.0, 25.0 * pitch)
            and item["area"] >= 200
            and item["roi_overlap_low_pixels"] >= 60
            and item["mean_probability"] >= .70]


def _center_curve(group: list[dict], pitch: float) -> tuple[np.ndarray, float, str]:
    points = np.concatenate([item["points"] for item in group], axis=0)
    xx, yy = points[:, 0], points[:, 1]
    xmin, xmax = float(xx.min()), float(xx.max())
    xcenter = .5 * (xmin + xmax)
    scale = max(100.0, .5 * (xmax - xmin))
    edges = np.arange(xmin, xmax + max(12.0, pitch), max(12.0, pitch))
    mids_x, mids_y = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        choose = (xx >= lo) & (xx < hi)
        if np.count_nonzero(choose) >= 4:
            mids_x.append(float(np.median(xx[choose])))
            mids_y.append(float(np.median(yy[choose])))
    mx, my = np.asarray(mids_x), np.asarray(mids_y)
    if len(mx) < 4:
        raise ValueError("not enough semantic center sections")
    xnorm = (mx - xcenter) / scale
    linear = np.polyfit(xnorm, my, 1)
    linear_y = np.polyval(linear, xnorm)
    fit = linear
    kind = "linear"
    if len(mx) >= 8 and xmax - xmin >= 10.0 * pitch:
        quadratic = np.polyfit(xnorm, my, 2)
        quad_y = np.polyval(quadratic, xnorm)
        slope_change = abs(4.0 * quadratic[0] / scale)
        if (np.median(np.abs(my - quad_y)) + 1.0 <
                np.median(np.abs(my - linear_y)) and slope_change <= .09):
            fit = quadratic
            kind = "quadratic"
    center_y = np.polyval(fit, (xx - xcenter) / scale)
    half_width = max(0.62 * pitch,
                     float(np.percentile(np.abs(yy - center_y), 90)) + .22 * pitch)
    if half_width > 1.35 * pitch:
        raise ValueError("semantic ribbon too wide for local rope pitch")
    # Return coefficients in original x units to avoid recalculating normalization.
    return np.poly1d(fit)(np.poly1d([1.0 / scale, -xcenter / scale])).c, half_width, kind


def estimate_free_rope_occlusion(
        image_bgr: np.ndarray, roi_mask: np.ndarray,
        camera_role: str, pitch_px: float
        ) -> tuple[np.ndarray, dict]:
    """Return a 0/255 candidate occluder limited to ROI, and evidence.

    A zero mask with ``status=not_detected`` or ``ambiguous`` means unknown,
    not proof that an incoming rope is absent. A broad, long, oblique semantic
    block may yield ``mask_kind=ambiguous_exclusion`` solely to keep boundaries
    UNKNOWN there. It is not a confirmed free-rope segmentation.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must be a BGR image")
    h, w = image_bgr.shape[:2]
    if roi_mask.shape != (h, w):
        raise ValueError("roi_mask dimensions differ from image")
    role = camera_role.replace("_camera", "")
    if role not in {"left", "right"}:
        raise ValueError("camera_role must be left or right")
    if not math.isfinite(pitch_px) or not 6.0 <= pitch_px <= 100.0:
        raise ValueError("pitch_px must be a finite plausible image pixel pitch")
    out = np.zeros((h, w), np.uint8)
    canonical = cv2.flip(image_bgr, 1) if role == "left" else image_bgr
    canonical_roi = cv2.flip(roi_mask, 1) if role == "left" else roi_mask
    regions = _locator(role).predict(canonical)
    x0, y0, x1, y1 = regions.crop_xyxy
    prob = regions.probability("incoming_rope_free_span")
    low_roi = cv2.resize(canonical_roi[y0:y1, x0:x1],
                         (prob.shape[1], prob.shape[0]),
                         interpolation=cv2.INTER_NEAREST) > 0
    baseline, baseline_ids = _baseline_guard(
        prob, low_roi, regions.crop_xyxy, (h, w), float(pitch_px))
    baseline &= canonical_roi > 0
    candidates = _component_candidates(prob, low_roi, regions.crop_xyxy,
                                       float(pitch_px))
    public = [{key: value for key, value in item.items()
               if key not in {"points", "vx", "vy", "cx", "cy"}}
              for item in candidates]
    groups = _candidate_groups(candidates, float(pitch_px))
    def qualification(group: list[dict]) -> str | None:
        if any(item["eligible_geometry"] and
               item["roi_overlap_low_pixels"] >= 8 for item in group):
            return "long_oblique_component_intersects_roi"
        external_long = any(item["eligible_geometry"] and
                            item["x_span_px"] >= max(500.0, 20.0 * pitch_px)
                            for item in group)
        internal_fragment = any(item["fragment_geometry"] and
                                item["roi_overlap_low_pixels"] >= 8
                                for item in group)
        if len(group) >= 2 and external_long and internal_fragment:
            return "external_long_oblique_plus_collinear_roi_fragment"
        return None

    qualified_groups = [group for group in groups if qualification(group)]
    diag: dict = {
        "source": "camera_semantic_free_span_plus_collinear_ribbon",
        "status": "candidate" if baseline_ids else "not_detected",
        "mask_kind": "supported_free_rope_candidate" if baseline_ids else "none",
        "refinement_status": "baseline_only" if baseline_ids else "not_attempted",
        "camera_role": role,
        "candidate_components": public, "selected_component_ids": [],
        "candidate_groups": [
            {"component_ids": [int(item["id"]) for item in group],
             "roi_overlap_low_pixels": int(sum(
                 item["roi_overlap_low_pixels"] for item in group)),
             "x_span_px": round(max(item["x_max"] for item in group) -
                                min(item["x_min"] for item in group), 2)}
            for group in groups],
        "baseline_component_ids": [int(v) for v in baseline_ids],
        "baseline_mask_pixels_in_roi": int(np.count_nonzero(baseline)),
        "mask_pixels_in_roi": int(np.count_nonzero(baseline)),
        "model_path": str(_locator(role).model_path),
        "mask_value_convention": "uint8_0_or_255",
        "semantic_threshold": .50,
    }
    def baseline_result():
        result = (cv2.flip(baseline, 1) if role == "left" else baseline)
        return result.astype(np.uint8) * 255, diag

    if not qualified_groups:
        uncertain_ids = _ambiguous_long_exclusion_components(
            candidates, float(pitch_px))
        if uncertain_ids:
            unknown, accepted_ids = _baseline_guard(
                prob, low_roi, regions.crop_xyxy, (h, w), float(pitch_px),
                set(uncertain_ids))
            unknown &= canonical_roi > 0
            if accepted_ids and np.count_nonzero(unknown):
                diag.update(
                    status="ambiguous_exclusion",
                    mask_kind="ambiguous_exclusion",
                    refinement_status="broad_long_semantic_unknown",
                    ambiguity_reason="long oblique semantic block is too broad for supported rope geometry",
                    uncertain_component_ids=[int(v) for v in accepted_ids],
                    mask_pixels_in_roi=int(np.count_nonzero(unknown)),
                )
                result = cv2.flip(unknown, 1) if role == "left" else unknown
                return result.astype(np.uint8) * 255, diag
        if baseline_ids:
            diag.update(
                status="ambiguous",
                mask_kind="none",
                refinement_status="rejected_unqualified_baseline",
                ambiguity_reason="semantic components lack a sufficiently long, narrow, oblique ROI-overlapping group",
                rejected_component_ids=[int(v) for v in baseline_ids],
                rejected_baseline_pixels_in_roi=int(np.count_nonzero(baseline)),
                mask_pixels_in_roi=0,
            )
        return out, diag
    group = max(qualified_groups, key=lambda entries:
                math.sqrt(sum(item["roi_overlap_low_pixels"] for item in entries)) *
                (max(item["x_max"] for item in entries) -
                 min(item["x_min"] for item in entries)) *
                np.average([item["mean_probability"] for item in entries],
                           weights=[item["area"] for item in entries]))
    diag["qualification_reason"] = qualification(group)
    selected_baseline, selected_baseline_ids = _baseline_guard(
        prob, low_roi, regions.crop_xyxy, (h, w), float(pitch_px),
        {item["id"] for item in group})
    selected_baseline &= canonical_roi > 0
    # Once a group is qualified, all unrelated semantic blobs are discarded.
    # If the smoother fails, only this group's narrow original guard remains.
    baseline = selected_baseline
    diag["baseline_component_ids"] = [int(v) for v in selected_baseline_ids]
    diag["baseline_mask_pixels_in_roi"] = int(np.count_nonzero(baseline))
    diag["mask_pixels_in_roi"] = int(np.count_nonzero(baseline))
    diag["selected_component_ids"] = [c["id"] for c in group]
    diag["selected_x_ranges_px"] = [
        [round(c["x_min"], 1), round(c["x_max"], 1)] for c in group]
    max_gap = max((_x_gap(a, b) for a, b in zip(group[:-1], group[1:])),
                  default=0.0)
    diag["max_bridged_gap_px"] = round(max_gap, 2)
    try:
        coeff, half_width, kind = _center_curve(group, float(pitch_px))
    except ValueError as exc:
        diag.update(status="candidate" if selected_baseline_ids else "ambiguous",
                    mask_kind="supported_free_rope_candidate" if selected_baseline_ids else "none",
                    refinement_status="rejected_curve_group_baseline",
                    ambiguity_reason=str(exc))
        return baseline_result()
    min_x = max(0, int(math.floor(group[0]["x_min"] - pitch_px)))
    max_x = min(w - 1, int(math.ceil(group[-1]["x_max"] + pitch_px)))
    x_values = np.arange(min_x, max_x + 1, 3, dtype=np.float64)
    y_values = np.polyval(coeff, x_values)
    valid = (y_values >= 0) & (y_values < h)
    points = np.column_stack((x_values[valid], y_values[valid]))
    if len(points) < 2:
        diag.update(status="candidate" if selected_baseline_ids else "ambiguous",
                    mask_kind="supported_free_rope_candidate" if selected_baseline_ids else "none",
                    refinement_status="rejected_curve_group_baseline",
                    ambiguity_reason="ribbon outside image")
        return baseline_result()
    ribbon = np.zeros((h, w), np.uint8)
    cv2.polylines(ribbon, [np.rint(points).astype(np.int32)], False, 255,
                  max(1, int(round(2.0 * half_width))), cv2.LINE_8)
    ribbon[canonical_roi == 0] = 0
    extra = int(np.count_nonzero((ribbon > 0) & ~baseline))
    union = (ribbon > 0) | baseline
    pixels = int(np.count_nonzero(union))
    if pixels < 40:
        diag.update(status="candidate" if selected_baseline_ids else "ambiguous",
                    mask_kind="supported_free_rope_candidate" if selected_baseline_ids else "none",
                    refinement_status="rejected_curve_group_baseline",
                    ambiguity_reason="tiny ROI intersection")
        return baseline_result()
    result = cv2.flip(union, 1) if role == "left" else union
    diag.update(
        status="candidate", refinement_status="ribbon_added",
        mask_kind="supported_free_rope_candidate",
        curve_type=kind,
        half_width_px=round(float(half_width), 2),
        max_bridged_gap_px=round(max_gap, 2),
        mask_pixels_in_roi=pixels,
        ribbon_extra_pixels_in_roi=extra,
        x_span_px=[int(min_x), int(max_x)],
    )
    return result.astype(np.uint8) * 255, diag


def outer_roi_contour_mask(roi_mask: np.ndarray, pitch_px: float) -> np.ndarray:
    """Build a display-only outer envelope; never use as a valid-pixel ROI.

    In particular, it deliberately bridges holes where an incoming rope
    splits the bright ROI.  The actual segmentation mask remains unchanged.
    """
    if roi_mask.ndim != 2:
        raise ValueError("roi_mask must be 2D")
    binary = roi_mask > 0
    h, w = binary.shape
    envelope = np.zeros((h, w), np.uint8)
    ys, xs = np.nonzero(binary)
    if len(xs) < 300:
        return envelope
    left, right = int(xs.min()), int(xs.max())
    top = np.full(w, np.nan, np.float32)
    bottom = np.full(w, np.nan, np.float32)
    for x in range(left, right + 1):
        rows = np.flatnonzero(binary[:, x])
        if len(rows) >= 5:
            top[x] = float(rows[0])
            bottom[x] = float(rows[-1])
    valid = np.flatnonzero(np.isfinite(top))
    if len(valid) < 4:
        return envelope
    xx = np.arange(valid[0], valid[-1] + 1)
    top_i = np.interp(xx, valid, top[valid]).astype(np.float32)
    bottom_i = np.interp(xx, valid, bottom[valid]).astype(np.float32)
    sigma = max(2.0, min(20.0, .5 * float(pitch_px)))
    top_s = cv2.GaussianBlur(top_i.reshape(1, -1), (0, 0), sigmaX=sigma)[0]
    bottom_s = cv2.GaussianBlur(bottom_i.reshape(1, -1), (0, 0), sigmaX=sigma)[0]
    for j, x in enumerate(xx):
        lo = max(0, min(h - 1, int(round(top_s[j]))))
        hi = max(0, min(h - 1, int(round(bottom_s[j]))))
        if lo <= hi:
            envelope[lo:hi + 1, x] = 255
    return envelope
