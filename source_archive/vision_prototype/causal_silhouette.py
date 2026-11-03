"""Past-only wound-surface silhouette anomaly for live diagnostics."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class SilhouetteObservation:
    valid: bool
    bump_px: float = math.nan
    reference_frames: int = 0
    valid_fraction: float = 0.0
    reason: str = "unavailable"


def _largest(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == index


def refine_target_silhouette(frame_bgr: np.ndarray, regions,
                             source_rows: np.ndarray) -> np.ndarray:
    """Refine canonical target-edge silhouette at sparse source rows."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mask = _largest(regions.mask("wound_rope_surface"))
    x1, y1, x2, y2 = regions.crop_xyxy
    low_h, low_w = mask.shape
    result = np.full(source_rows.shape, np.nan, np.float64)
    for index, source_y in enumerate(source_rows):
        low_y = int(round((source_y - y1) * low_h / max(y2 - y1, 1)))
        if low_y < 0 or low_y >= low_h:
            continue
        occupied = np.flatnonzero(mask[low_y])
        if occupied.size == 0:
            continue
        predicted = x1 + float(occupied[0]) * (x2 - x1) / low_w
        xa = max(2, int(round(predicted - 65)))
        xb = min(gray.shape[1] - 2, int(round(predicted + 66)))
        if xb - xa < 8:
            continue
        ya = max(0, int(source_y) - 3)
        yb = min(gray.shape[0], int(source_y) + 4)
        # Only a narrow strip around the semantic prediction is required.
        # A full 12 MP Sobel image added hundreds of milliseconds per camera
        # and made otherwise healthy live feeds look disconnected.
        strip = gray[ya:yb, max(0, xa - 1):min(gray.shape[1], xb + 1)]
        intensity = np.mean(strip.astype(np.float32), axis=0)
        if intensity.size < 5:
            continue
        local_gradient = np.abs(np.gradient(intensity))
        profile = local_gradient[1:-1]
        if profile.size == 0 or float(np.max(profile)) < 8.0:
            continue
        coordinates = np.arange(xa, xb, dtype=np.float64)
        weight = np.exp(-.5 * ((coordinates - predicted) / 28.0) ** 2)
        result[index] = float(xa + int(np.argmax(profile * weight)))
    return result


class CausalSilhouetteTracker:
    """Compare the current contour only with sufficiently old contours."""

    def __init__(self, history_s: float = 7.0, guard_s: float = .6,
                 minimum_reference_frames: int = 20):
        self.history_s = float(history_s)
        self.guard_s = float(guard_s)
        self.minimum_reference_frames = int(minimum_reference_frames)
        self.history: deque[tuple[float, np.ndarray]] = deque(maxlen=512)
        self.last_time_s: float | None = None

    def reset(self) -> None:
        self.history.clear()
        self.last_time_s = None

    def update(self, frame_bgr: np.ndarray, regions,
               time_s: float) -> SilhouetteObservation:
        if not math.isfinite(time_s):
            self.reset()
            return SilhouetteObservation(False, reason="invalid_timestamp")
        if self.last_time_s is not None and time_s <= self.last_time_s:
            self.reset()
        self.last_time_s = float(time_s)
        height = frame_bgr.shape[0]
        rows = np.arange(int(round(.633 * height)),
                         int(round(.974 * height)) + 1,
                         max(10, int(round(height / 250))), dtype=np.int32)
        contour = refine_target_silhouette(frame_bgr, regions, rows)
        while self.history and time_s - self.history[0][0] > self.history_s:
            self.history.popleft()
        references = [item[1] for item in self.history
                      if item[0] <= time_s - self.guard_s]
        self.history.append((float(time_s), contour))
        valid_fraction = float(np.mean(np.isfinite(contour)))
        if len(references) < self.minimum_reference_frames:
            return SilhouetteObservation(
                False, reference_frames=len(references),
                valid_fraction=valid_fraction,
                reason="silhouette_history_warming_up")
        matrix = np.stack(references)
        reference = np.full(contour.shape, np.nan, np.float64)
        supported = np.any(np.isfinite(matrix), axis=0)
        reference[supported] = np.nanmedian(matrix[:, supported], axis=0)
        lower = rows >= .833 * height
        upper = (rows >= .673 * height) & (rows <= .807 * height)
        anchor = contour[lower] - reference[lower]
        anchor = anchor[np.isfinite(anchor)]
        if anchor.size < 10:
            return SilhouetteObservation(
                False, reference_frames=len(references),
                valid_fraction=valid_fraction,
                reason="silhouette_anchor_unavailable")
        global_shift = float(np.median(anchor))
        outward = reference - (contour - global_shift)
        values = outward[upper & np.isfinite(outward)]
        if values.size < 10:
            return SilhouetteObservation(
                False, reference_frames=len(references),
                valid_fraction=valid_fraction,
                reason="silhouette_event_band_unavailable")
        return SilhouetteObservation(
            True, float(np.percentile(values, 90)), len(references),
            valid_fraction, "past_only_silhouette_bump")
