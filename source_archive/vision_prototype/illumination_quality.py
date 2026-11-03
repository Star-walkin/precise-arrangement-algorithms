"""Read-only, camera-local illumination quality measurements.

The monitor evaluates only pixels inside the semantic wound-rope surface and
reuses the photometric observability rule used by ``surface_centerlines``.
It has no camera, serial, network, PLC, relay, or light-controller interface;
its result is diagnostic data and cannot authorize an actuator command.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import cv2
import numpy as np

from semantic_locator import SemanticRegions
from surface_centerlines import photometric_observation_mask


@dataclass(frozen=True)
class IlluminationQualityResult:
    """Objective per-frame scores on a 0 (bad) to 100 (good) scale."""

    valid: bool
    status: str
    reasons: tuple[str, ...]
    camera_role: str = 'unknown'
    overall_score: float = 0.0
    exposure_score: float = 0.0
    dark_region_score: float = 0.0
    texture_observability_score: float = 0.0
    photometric_observability_score: float = 0.0
    photometric_observable_fraction: float = math.nan
    texture_observable_fraction: float = math.nan
    dark_fraction: float = math.nan
    highlight_fraction: float = math.nan
    clipped_fraction: float = math.nan
    largest_dark_run_fraction: float = math.nan
    largest_unobservable_run_fraction: float = math.nan
    gray_p05: float = math.nan
    gray_median: float = math.nan
    gray_p95: float = math.nan
    local_contrast_median: float = math.nan
    wound_pixel_count: int = 0
    # These init=False safety facts cannot be changed through configuration.
    read_only: bool = field(default=True, init=False)
    automatic_control_enabled: bool = field(default=False, init=False)


def _descending_score(value: float, good: float, bad: float) -> float:
    """Return 1 at/below ``good`` and 0 at/above ``bad``."""
    if bad <= good:
        raise ValueError('bad threshold must be greater than good threshold')
    return float(1.0 - np.clip((value - good) / (bad - good), 0.0, 1.0))


def _ascending_score(value: float, bad: float, good: float) -> float:
    """Return 0 at/below ``bad`` and 1 at/above ``good``."""
    if good <= bad:
        raise ValueError('good threshold must be greater than bad threshold')
    return float(np.clip((value - bad) / (good - bad), 0.0, 1.0))


def _largest_true_run_fraction(values: np.ndarray, denominator: int) -> float:
    indices = np.flatnonzero(values)
    if indices.size == 0:
        return 0.0
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    return float(max(len(group) for group in groups) / max(1, denominator))


def _prepare_analysis_crop(
        frame_bgr: np.ndarray, regions: SemanticRegions,
        pitch_px: float, maximum_dimension: int
        ) -> tuple[np.ndarray, np.ndarray, float, int] | None:
    if (frame_bgr is None or frame_bgr.ndim != 3 or
            frame_bgr.shape[2] != 3 or frame_bgr.size == 0):
        raise ValueError('frame_bgr must be a nonempty BGR image')
    low_wound = regions.mask('wound_rope_surface').astype(np.uint8)
    component_count, labels, stats, _centroids = (
        cv2.connectedComponentsWithStats(low_wound, 8))
    if component_count <= 1:
        return None
    component_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = labels == component_index
    low_pixel_count = int(stats[component_index, cv2.CC_STAT_AREA])
    if low_pixel_count < 16:
        return None
    crop_x1, crop_y1, crop_x2, crop_y2 = regions.crop_xyxy
    crop_width, crop_height = crop_x2 - crop_x1, crop_y2 - crop_y1
    low_height, low_width = low_wound.shape
    wound_pixel_count = round(
        low_pixel_count * crop_width * crop_height /
        max(1, low_width * low_height))
    if wound_pixel_count < 256:
        return None
    low_x = int(stats[component_index, cv2.CC_STAT_LEFT])
    low_y = int(stats[component_index, cv2.CC_STAT_TOP])
    low_w = int(stats[component_index, cv2.CC_STAT_WIDTH])
    low_h = int(stats[component_index, cv2.CC_STAT_HEIGHT])
    low_x2, low_y2 = low_x + low_w, low_y + low_h
    x1 = crop_x1 + int(math.floor(low_x * crop_width / low_width))
    x2 = crop_x1 + int(math.ceil(low_x2 * crop_width / low_width))
    y1 = crop_y1 + int(math.floor(low_y * crop_height / low_height))
    y2 = crop_y1 + int(math.ceil(low_y2 * crop_height / low_height))
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(frame_bgr.shape[1], max(x1 + 1, x2))
    y2 = min(frame_bgr.shape[0], max(y1 + 1, y2))
    frame = frame_bgr[y1:y2, x1:x2]
    mask = component[low_y:low_y2, low_x:low_x2]
    maximum_dimension = max(128, int(maximum_dimension))
    scale = min(1.0, maximum_dimension / max(frame.shape[:2]))
    size = (max(1, round(frame.shape[1] * scale)),
            max(1, round(frame.shape[0] * scale)))
    if size != (frame.shape[1], frame.shape[0]):
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    if size != (mask.shape[1], mask.shape[0]):
        mask = cv2.resize(mask.astype(np.uint8), size,
                          interpolation=cv2.INTER_NEAREST)
    analysis_pitch = max(4.0, float(pitch_px) * scale)
    return frame, mask.astype(bool), analysis_pitch, wound_pixel_count


def assess_illumination_quality(
        frame_bgr: np.ndarray, regions: SemanticRegions, pitch_px: float,
        *, camera_role: str = 'unknown', maximum_dimension: int = 720
        ) -> IlluminationQualityResult:
    """Measure exposure, dark regions, and centerline observability.

    Scores are camera-local and recipe-independent.  Raw fractions and gray
    percentiles are retained so a later, stopped-line illumination sweep can
    compare steps without treating this score as a light-control command.
    """
    if not math.isfinite(pitch_px) or pitch_px <= 0:
        raise ValueError('pitch_px must be finite and positive')
    prepared = _prepare_analysis_crop(
        frame_bgr, regions, pitch_px, maximum_dimension)
    if prepared is None:
        return IlluminationQualityResult(
            False, 'unavailable', ('wound_surface_unavailable',),
            camera_role=str(camera_role))
    frame, wound, analysis_pitch, wound_pixel_count = prepared

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    values = gray[wound]
    p05, median, p95 = (float(value) for value in
                        np.percentile(values, (5, 50, 95)))
    dark_fraction = float(np.mean(values <= 24.0))
    highlight_fraction = float(np.mean(values >= 245.0))
    clipped_fraction = float(np.mean((values <= 3.0) | (values >= 252.0)))

    sigma = max(3.0, .45 * analysis_pitch)
    local_mean = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    square_mean = cv2.GaussianBlur(
        gray * gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    local_std = np.sqrt(np.maximum(square_mean - local_mean * local_mean, 0.0))
    local_contrast_median = float(np.median(local_std[wound]))
    texture_observable_fraction = float(np.mean(local_std[wound] >= 2.5))
    texture_fraction_score = _ascending_score(
        texture_observable_fraction, .15, .85)
    texture_strength_score = _ascending_score(
        local_contrast_median, 2.0, 10.0)
    texture_score = 100.0 * (
        .72 * texture_fraction_score + .28 * texture_strength_score)

    quality_mask = photometric_observation_mask(
        frame, wound, analysis_pitch)
    observable_fraction = float(np.mean(quality_mask[wound]))
    observability_score = 100.0 * observable_fraction

    # Column runs expose a local black/clipped band that a global mean hides.
    wound_counts = np.count_nonzero(wound, axis=0)
    quality_counts = np.count_nonzero(quality_mask, axis=0)
    dark_counts = np.count_nonzero(wound & (gray <= 24.0), axis=0)
    active_columns = wound_counts > 0
    active_indices = np.flatnonzero(active_columns)
    if active_indices.size:
        xa, xb = int(active_indices[0]), int(active_indices[-1]) + 1
        width = xb - xa
        inset = max(1, int(round(.22 * width)))
        ca, cb = min(xb, xa + inset), max(xa, xb - inset)
        column_slice = slice(ca, cb) if cb > ca else slice(xa, xb)
        column_observable = (quality_counts[column_slice] /
                             np.maximum(wound_counts[column_slice], 1))
        column_dark = (dark_counts[column_slice] /
                       np.maximum(wound_counts[column_slice], 1))
        largest_unobservable_run = _largest_true_run_fraction(
            (wound_counts[column_slice] == 0) | (column_observable < .18),
            width)
        largest_dark_run = _largest_true_run_fraction(
            (wound_counts[column_slice] > 0) & (column_dark >= .55), width)
    else:  # Defensive; the minimum pixel check above normally makes this impossible.
        largest_unobservable_run = largest_dark_run = 1.0

    if 55.0 <= median <= 205.0:
        median_exposure_score = 1.0
    elif median < 55.0:
        median_exposure_score = _ascending_score(median, 3.0, 55.0)
    else:
        median_exposure_score = _descending_score(median, 205.0, 252.0)
    tail_score = min(
        _descending_score(dark_fraction, .03, .35),
        _descending_score(highlight_fraction, .02, .25),
        _descending_score(clipped_fraction, .01, .20))
    exposure_score = 100.0 * median_exposure_score * (.55 + .45 * tail_score)

    dark_region_score = 100.0 * min(
        _descending_score(dark_fraction, .04, .50),
        _descending_score(largest_dark_run, .06, .40))

    weighted = (.30 * exposure_score + .20 * dark_region_score +
                .30 * texture_score + .20 * observability_score)
    bottleneck = min(exposure_score, dark_region_score,
                     texture_score, observability_score)
    overall_score = float(np.clip(min(weighted, bottleneck + 25.0), 0.0, 100.0))

    reasons: list[str] = []
    if dark_fraction >= .25 or median < 45.0:
        reasons.append('underexposed')
    if highlight_fraction >= .12 or median > 215.0:
        reasons.append('overexposed')
    if clipped_fraction >= .05:
        reasons.append('clipped_pixels')
    if largest_dark_run >= .15:
        reasons.append('broad_dark_region')
    if texture_score < 55.0:
        reasons.append('low_texture_observability')
    if largest_unobservable_run >= .15:
        reasons.append('broad_unobservable_region')
    if observable_fraction < .60:
        reasons.append('low_photometric_observability')
    if not reasons:
        reasons.append('quality_good')

    status = ('good' if overall_score >= 78.0 and reasons == ['quality_good']
              else 'caution' if overall_score >= 50.0 else 'poor')
    return IlluminationQualityResult(
        valid=True, status=status, reasons=tuple(reasons),
        camera_role=str(camera_role), overall_score=overall_score,
        exposure_score=float(exposure_score),
        dark_region_score=float(dark_region_score),
        texture_observability_score=float(texture_score),
        photometric_observability_score=float(observability_score),
        photometric_observable_fraction=observable_fraction,
        texture_observable_fraction=texture_observable_fraction,
        dark_fraction=dark_fraction, highlight_fraction=highlight_fraction,
        clipped_fraction=clipped_fraction,
        largest_dark_run_fraction=largest_dark_run,
        largest_unobservable_run_fraction=largest_unobservable_run,
        gray_p05=p05, gray_median=median, gray_p95=p95,
        local_contrast_median=local_contrast_median,
        wound_pixel_count=wound_pixel_count)
