"""Low-latency tracking of a previously rebuilt wound-rope centreline family.

The expensive full-family extractor is deliberately kept as an independent
rebuild operation.  Between rebuilds this module searches only narrow bands
around the last observed paths, estimates the current local phase/pitch, and
cross-checks the result with a structure-tensor orientation score.  All
outputs are diagnostic measurements; this module has no device interface.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from semantic_locator import SemanticRegions
from surface_centerlines import (CenterlineFamily, RopeCenterline,
                                 extract_centerline_family,
                                 source_wound_mask)


@dataclass(frozen=True)
class FastTrackResult:
    valid: bool
    reason: str
    family: CenterlineFamily
    rebuilt: bool
    request_rebuild: bool
    orientation_score: float = math.nan
    hessian_support: float = math.nan
    phase_shift_px: float = math.nan
    pitch_ratio: float = math.nan
    observed_fraction: float = 0.0
    tracking_confidence: float = 0.0


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    intersection = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return intersection / max(1, union)


def _mask_geometry(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size < 1000:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1),
            float(ys.max() + 1))


def _ridge_response_roi(gray: np.ndarray, pitch: float) -> np.ndarray:
    """Fast, local vertical periodic response on one cropped surface ROI."""
    broad = cv2.GaussianBlur(gray.astype(np.float32), (0, 0),
                             sigmaX=max(4.0, 1.45 * pitch),
                             sigmaY=max(4.0, 1.45 * pitch))
    normalized = gray.astype(np.float32) - broad
    radius = min(24, max(5, int(round(1.18 * pitch))))
    coordinate = np.arange(-radius, radius + 1, dtype=np.float32)
    vertical = (np.exp(-.5 * (coordinate / max(2.0, .38 * pitch)) ** 2) *
                np.cos(2.0 * np.pi * coordinate / pitch))
    vertical -= float(np.mean(vertical))
    vertical /= float(np.linalg.norm(vertical) + 1e-9)
    horizontal_size = max(3, int(round(.36 * pitch)) | 1)
    horizontal = cv2.getGaussianKernel(
        horizontal_size, max(1.0, .15 * pitch), cv2.CV_32F)[:, 0]
    return cv2.sepFilter2D(normalized, cv2.CV_32F, horizontal, vertical)


def _orientation_score(gray: np.ndarray, mask: np.ndarray,
                       sample_step: int = 4) -> float:
    """Return evidence that ridges run mainly left/right, in [0, 1]."""
    small = gray[::sample_step, ::sample_step].astype(np.float32)
    valid = mask[::sample_step, ::sample_step]
    if int(np.count_nonzero(valid)) < 400:
        return 0.0
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 1.2)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 1.2)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 1.2)
    root = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
    coherence = root / (jxx + jyy + 1e-6)
    # Horizontal rope ridges have a predominantly vertical normal (Gy).
    vertical_normal = np.maximum(0.0, (jyy - jxx) / (jxx + jyy + 1e-6))
    energy = jxx + jyy
    threshold = float(np.percentile(energy[valid], 55))
    selected = valid & (energy >= threshold)
    if int(np.count_nonzero(selected)) < 100:
        return 0.0
    value = np.median((.55 * coherence + .45 * vertical_normal)[selected])
    return float(np.clip(value, 0.0, 1.0))


def _hessian_support(gray: np.ndarray, mask: np.ndarray,
                     family: CenterlineFamily, origin_xy: tuple[int, int],
                     scale: float) -> float:
    """Independent second-derivative support at the tracked line locations."""
    if not family.centerlines:
        return 0.0
    # Half resolution is sufficient for a yes/no cross-check and avoids
    # turning an independent diagnostic into the dominant runtime cost.
    reduction = 4
    gray = gray[::reduction, ::reduction]
    mask = mask[::reduction, ::reduction]
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    dxx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=3)
    dyy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=3)
    dxy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3)
    discriminant = np.sqrt(np.maximum((dxx - dyy) ** 2 + 4.0 * dxy * dxy,
                                      0.0))
    eigen_a = .5 * (dxx + dyy + discriminant)
    eigen_b = .5 * (dxx + dyy - discriminant)
    ridge = np.maximum(np.abs(eigen_a), np.abs(eigen_b))
    background = ridge[mask]
    if background.size < 400:
        return 0.0
    low, high = np.percentile(background, [35, 88])
    samples = []
    ox, oy = origin_xy
    for line in family.centerlines:
        for px, py in line.observed_points_xy[::max(1, len(line.observed_points_xy) // 8)]:
            x = int(round((float(px) - ox) * scale / reduction))
            y = int(round((float(py) - oy) * scale / reduction))
            if 0 <= x < ridge.shape[1] and 0 <= y < ridge.shape[0] and mask[y, x]:
                samples.append(float(ridge[y, x]))
    if len(samples) < 20:
        return 0.0
    normalized = (float(np.median(samples)) - low) / max(high - low, 1e-6)
    return float(np.clip(normalized, 0.0, 1.0))


def _quadratic_peak(values: np.ndarray, index: int) -> float:
    if index <= 0 or index >= len(values) - 1:
        return float(index)
    a, b, c = map(float, values[index - 1:index + 2])
    denominator = a - 2.0 * b + c
    delta = .5 * (a - c) / denominator if abs(denominator) > 1e-9 else 0.0
    return float(index + np.clip(delta, -.5, .5))


class FastCenterlineTracker:
    """Sparse narrow-band tracker with periodic full-family rebuilds."""

    def __init__(self, fallback_pitch_px: float,
                 rebuild_interval_s: float = 1.5,
                 maximum_width: int = 1100):
        self.fallback_pitch_px = float(fallback_pitch_px)
        self.rebuild_interval_s = float(rebuild_interval_s)
        self.maximum_width = int(maximum_width)
        self.family: CenterlineFamily | None = None
        self.mask: np.ndarray | None = None
        self.last_rebuild_s = math.nan

    def reset(self) -> None:
        self.family = None
        self.mask = None
        self.last_rebuild_s = math.nan

    def install_rebuild(self, family: CenterlineFamily, mask: np.ndarray,
                        time_s: float) -> None:
        """Atomically install a completed external/background rebuild."""
        if mask.ndim != 2:
            raise ValueError("rebuild mask must be two-dimensional")
        self.family = family
        self.mask = mask.astype(bool, copy=False)
        self.last_rebuild_s = float(time_s)

    def rebuild(self, frame_bgr: np.ndarray, regions: SemanticRegions,
                time_s: float) -> FastTrackResult:
        family = extract_centerline_family(
            frame_bgr, regions, self.fallback_pitch_px)
        mask = source_wound_mask(regions, frame_bgr.shape)
        self.family = family
        self.mask = mask
        self.last_rebuild_s = float(time_s)
        return FastTrackResult(
            valid=family.valid, reason=family.reason, family=family,
            rebuilt=True, request_rebuild=not family.valid,
            observed_fraction=(float(np.median(
                [line.coverage for line in family.centerlines]))
                if family.centerlines else 0.0),
            tracking_confidence=family.family_confidence)

    def update(self, frame_bgr: np.ndarray, regions: SemanticRegions,
               time_s: float, allow_rebuild: bool = True,
               blocking_recovery: bool = False) -> FastTrackResult:
        if self.family is None or self.mask is None or not self.family.valid:
            if allow_rebuild:
                return self.rebuild(frame_bgr, regions, time_s)
            empty = self.family or CenterlineFamily(False, "tracker_uninitialized")
            return FastTrackResult(False, "tracker_uninitialized", empty,
                                   False, True)
        current_mask = source_wound_mask(regions, frame_bgr.shape)
        elapsed = float(time_s) - self.last_rebuild_s
        mask_iou = _mask_iou(self.mask, current_mask)
        old_geometry = _mask_geometry(self.mask)
        new_geometry = _mask_geometry(current_mask)
        bbox_changed = True
        if old_geometry is not None and new_geometry is not None:
            old_width = old_geometry[2] - old_geometry[0]
            old_height = old_geometry[3] - old_geometry[1]
            center_shift = math.hypot(
                .5 * (new_geometry[0] + new_geometry[2] -
                      old_geometry[0] - old_geometry[2]),
                .5 * (new_geometry[1] + new_geometry[3] -
                      old_geometry[1] - old_geometry[3]))
            scale_change = max(
                abs((new_geometry[2] - new_geometry[0]) /
                    max(old_width, 1.0) - 1.0),
                abs((new_geometry[3] - new_geometry[1]) /
                    max(old_height, 1.0) - 1.0))
            bbox_changed = (center_shift > .06 * math.hypot(old_width, old_height)
                            or scale_change > .15)
        periodic = elapsed >= self.rebuild_interval_s
        # Semantic boundaries naturally shimmer by several pixels. Only a
        # large mask loss together with a bbox displacement is a geometry
        # change; otherwise the narrow-band tracker is the stabilizer.
        geometry_changed = mask_iou < .40 or bbox_changed
        if allow_rebuild and (periodic or geometry_changed):
            return self.rebuild(frame_bgr, regions, time_s)
        tracked = self._track(frame_bgr, current_mask)
        if tracked.request_rebuild and allow_rebuild and blocking_recovery:
            return self.rebuild(frame_bgr, regions, time_s)
        if tracked.valid:
            self.family = tracked.family
            self.mask = current_mask
        return tracked

    def _track(self, frame_bgr: np.ndarray,
               current_mask: np.ndarray) -> FastTrackResult:
        assert self.family is not None
        old = self.family
        x1, y1, x2, y2 = old.bbox_xyxy
        height, width = frame_bgr.shape[:2]
        margin = max(8, int(round(old.pitch_px)))
        x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
        x2, y2 = min(width, x2 + margin), min(height, y2 + margin)
        if x2 - x1 < 80 or y2 - y1 < 80:
            return FastTrackResult(False, "tracking_roi_too_small", old,
                                   False, True)
        scale = min(1.0, self.maximum_width / float(x2 - x1))
        roi = frame_bgr[y1:y2, x1:x2]
        mask_roi = current_mask[y1:y2, x1:x2]
        size = (max(2, int(round(roi.shape[1] * scale))),
                max(2, int(round(roi.shape[0] * scale))))
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if scale < .999:
            gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
            mask_small = cv2.resize(mask_roi.astype(np.uint8), size,
                                    interpolation=cv2.INTER_NEAREST) > 0
        else:
            mask_small = mask_roi
        pitch = old.pitch_px * scale
        if pitch < 4.0:
            return FastTrackResult(False, "tracking_pitch_too_small", old,
                                   False, True)
        response = _ridge_response_roi(gray, pitch)
        orientation = _orientation_score(gray, mask_small)
        tracked_lines = []
        shifts = []
        pitch_samples = []
        total_samples = 0
        observed_samples = 0
        search_radius = max(2, int(round(.42 * pitch)))
        positions = np.empty(0, np.float64)
        for line in old.centerlines:
            prior = line.points_xy
            if len(prior) < 4:
                continue
            stride = max(1, len(prior) // 10)
            observations = []
            for px, py in prior[::stride]:
                sx = int(round((float(px) - x1) * scale))
                sy = (float(py) - y1) * scale
                if sx < 1 or sx >= response.shape[1] - 1:
                    continue
                total_samples += 1
                lo = max(1, int(round(sy)) - search_radius)
                hi = min(response.shape[0] - 1,
                         int(round(sy)) + search_radius + 1)
                if hi - lo < 3:
                    continue
                column = response[lo:hi, max(0, sx - 1):sx + 2].mean(axis=1)
                index = int(np.argmax(column))
                peak = float(column[index])
                baseline = float(np.median(column))
                robust = float(np.median(np.abs(column - baseline)) * 1.4826 + 1e-6)
                yy = lo + _quadratic_peak(column, index)
                global_y = y1 + yy / scale
                global_x = x1 + sx / scale
                iy = int(np.clip(round(yy), 0, mask_small.shape[0] - 1))
                if (not mask_small[iy, sx] or
                        (peak - baseline) / robust < .65):
                    continue
                observations.append((global_x, global_y,
                                     (peak - baseline) / robust))
                shifts.append(global_y - float(py))
                observed_samples += 1
            if len(observations) < 6:
                continue
            points = np.asarray(observations, np.float64)
            degree = 2 if len(points) >= 10 else 1
            coefficients = np.polyfit(points[:, 0], points[:, 1], degree)
            residual = np.abs(np.polyval(coefficients, points[:, 0]) - points[:, 1])
            keep = residual <= max(2.0, .24 * old.pitch_px)
            if int(keep.sum()) < 6:
                continue
            source_x = prior[:, 0].astype(np.float64)
            fitted_y = np.polyval(coefficients, source_x)
            inside_x = np.clip(np.rint(source_x).astype(int), 0, width - 1)
            inside_y = np.clip(np.rint(fitted_y).astype(int), 0, height - 1)
            inside = current_mask[inside_y, inside_x]
            fitted = np.column_stack((source_x[inside], fitted_y[inside])).astype(np.float32)
            observed = points[keep, :2].astype(np.float32)
            if len(fitted) >= 4:
                tracked_lines.append(RopeCenterline(
                    fitted, observed, len(observed) / max(1, len(prior[::stride])),
                    float(np.median(points[keep, 2]))))
        if len(tracked_lines) >= 4:
            center_x = .5 * (old.bbox_xyxy[0] + old.bbox_xyxy[2])
            positions = []
            for line in tracked_lines:
                if line.points_xy[0, 0] <= center_x <= line.points_xy[-1, 0]:
                    positions.append(float(np.interp(center_x,
                                                     line.points_xy[:, 0],
                                                     line.points_xy[:, 1])))
            positions = np.asarray(sorted(positions), np.float64)
            if positions.size >= 4:
                differences = np.diff(positions)
                plausible = differences[(differences >= .55 * old.pitch_px) &
                                        (differences <= 1.45 * old.pitch_px)]
                if plausible.size:
                    pitch_samples.extend(plausible.tolist())
        observed_fraction = observed_samples / max(1, total_samples)
        new_pitch = (float(np.median(pitch_samples)) if pitch_samples
                     else old.pitch_px)
        pitch_ratio = new_pitch / max(old.pitch_px, 1e-6)
        phase_shift = float(np.median(shifts)) if shifts else math.nan
        pitch_agreement = math.exp(-abs(pitch_ratio - 1.0) / .08)
        line_fraction = len(tracked_lines) / max(1, len(old.centerlines))
        provisional = CenterlineFamily(
            True, "provisional", old.bbox_xyxy, new_pitch,
            old.pitch_confidence, tuple(tracked_lines), math.nan, 0.0,
            old.sampled_columns)
        hessian = _hessian_support(gray, mask_small, provisional,
                                   (x1, y1), scale)
        confidence = float(np.clip(
            .33 * observed_fraction + .24 * min(1.0, line_fraction) +
            .18 * orientation + .08 * hessian +
            .17 * pitch_agreement, 0.0, 1.0))
        request = (len(tracked_lines) < 4 or observed_fraction < .18 or
                   orientation < .16 or not .84 <= pitch_ratio <= 1.18 or
                   confidence < .35)
        valid = not request
        reason = "ok" if valid else "narrow_band_tracking_unreliable"
        if tracked_lines:
            regularity = math.nan
            if len(positions) >= 4:
                regularity = float(np.median(
                    np.abs(np.diff(positions) / max(new_pitch, 1e-6) - 1.0)))
            family = CenterlineFamily(
                valid, reason, old.bbox_xyxy, new_pitch,
                old.pitch_confidence, tuple(tracked_lines), regularity,
                confidence, old.sampled_columns)
        else:
            family = CenterlineFamily(False, reason, old.bbox_xyxy,
                                      old.pitch_px, old.pitch_confidence)
        return FastTrackResult(
            valid, reason, family, False, request, orientation, hessian,
            phase_shift, pitch_ratio, observed_fraction, confidence)
