"""Read-only visual rotation-period and first-wrap timing helpers.

The image estimator converts several flange-texture measurements into scalar
time series and looks for a period that is supported by both autocorrelation
and a frequency spectrum.  It deliberately reports ``unavailable`` for a
static or weakly periodic signal.  No class in this module imports a device
driver or authorizes motion.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class RotationEstimate:
    status: str = "unavailable"
    period_s: float = math.nan
    rpm: float = math.nan
    confidence: float = 0.0
    samples: int = 0
    reason: str = "insufficient_history"
    candidate_period_s: float = math.nan
    spectral_period_s: float = math.nan
    autocorrelation_peak: float = math.nan
    autocorrelation_prominence: float = math.nan
    spectral_contrast: float = math.nan
    method_agreement: float = math.nan


def flange_texture_features(image_bgr: np.ndarray, flange_mask: np.ndarray,
                            rows: int = 2, columns: int = 5
                            ) -> np.ndarray | None:
    """Return illumination-normalized texture features from the flange face.

    Features are measured in a grid over the *current* semantic flange mask.
    Per-frame centering removes uniform lamp/exposure changes.  The method is
    markerless and therefore remains a diagnostic signal, not a certified
    revolution pulse.
    """
    if image_bgr.ndim != 3 or flange_mask.ndim != 2:
        return None
    height, width = flange_mask.shape
    if image_bgr.shape[:2] != (height, width):
        return None
    mask = flange_mask.astype(np.uint8)
    if int(mask.sum()) < max(200, height * width // 500):
        return None
    ys, xs = np.nonzero(mask)
    x1, x2 = int(np.percentile(xs, 3)), int(np.percentile(xs, 97)) + 1
    y1, y2 = int(np.percentile(ys, 3)), int(np.percentile(ys, 97)) + 1
    if x2 - x1 < columns * 3 or y2 - y1 < rows * 3:
        return None
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Gradient energy helps when the yellow flange brightness is nearly flat
    # but tape wrinkles/marks move through the fixed camera view.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    values: list[float] = []
    for row in range(rows):
        ya = y1 + (y2 - y1) * row // rows
        yb = y1 + (y2 - y1) * (row + 1) // rows
        for column in range(columns):
            xa = x1 + (x2 - x1) * column // columns
            xb = x1 + (x2 - x1) * (column + 1) // columns
            cell = mask[ya:yb, xa:xb] > 0
            if int(cell.sum()) < 20:
                values.extend((math.nan, math.nan))
                continue
            values.append(float(np.mean(gray[ya:yb, xa:xb][cell])))
            values.append(float(np.mean(gradient[ya:yb, xa:xb][cell])))
    vector = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(vector)
    if int(finite.sum()) < max(6, vector.size // 2):
        return None
    # Fill missing grid cells without inventing a temporal pattern.
    vector[~finite] = float(np.median(vector[finite]))
    brightness = vector[0::2]
    edges = vector[1::2]
    brightness -= float(np.mean(brightness))
    edges -= float(np.mean(edges))
    brightness /= float(np.std(brightness) + 1e-6)
    edges /= float(np.std(edges) + 1e-6)
    return vector


def flange_edge_features(image_bgr: np.ndarray, flange_mask: np.ndarray,
                         columns: int = 12) -> np.ndarray | None:
    """Track original-image flange boundary positions in several columns.

    This follows the markerless edge-position idea used for conveyor idlers:
    small geometric/tape imperfections create a periodic boundary signal as
    the part rotates.  The semantic mask only limits the search; the measured
    position comes from the original grayscale gradient.
    """
    if image_bgr.ndim != 3 or flange_mask.ndim != 2:
        return None
    height, width = flange_mask.shape
    if image_bgr.shape[:2] != (height, width):
        return None
    mask = flange_mask.astype(bool)
    if int(mask.sum()) < max(200, height * width // 500):
        return None
    ys, xs = np.nonzero(mask)
    x1, x2 = int(np.percentile(xs, 8)), int(np.percentile(xs, 92))
    if x2 - x1 < columns * 2:
        return None
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    vertical_gradient = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    result: list[float] = []
    for x in np.linspace(x1, x2, columns).round().astype(int):
        occupied = np.flatnonzero(mask[:, x])
        if occupied.size < 3:
            result.extend((math.nan, math.nan))
            continue
        for expected in (int(occupied[0]), int(occupied[-1])):
            start, stop = max(0, expected - 7), min(height, expected + 8)
            profile = vertical_gradient[start:stop, x]
            if profile.size == 0 or float(np.max(profile)) < 4.0:
                result.append(math.nan)
            else:
                result.append(float(start + int(np.argmax(profile))))
    vector = np.asarray(result, np.float32)
    finite = np.isfinite(vector)
    if int(finite.sum()) < max(8, vector.size // 2):
        return None
    vector[~finite] = float(np.median(vector[finite]))
    return vector


def flange_rotation_features(image_bgr: np.ndarray,
                             flange_mask: np.ndarray) -> np.ndarray | None:
    """Combine complementary texture and boundary-position measurements."""
    texture = flange_texture_features(image_bgr, flange_mask)
    edges = flange_edge_features(image_bgr, flange_mask)
    available = [part for part in (texture, edges) if part is not None]
    return np.concatenate(available) if available else None


class MarkerlessRotationEstimator:
    """Estimate a slow spool period from flange texture over a causal window."""

    def __init__(self, minimum_period_s: float = 1.2,
                 maximum_period_s: float = 6.0, history_s: float = 16.0,
                 minimum_cycles: float = 2.0) -> None:
        if not 0 < minimum_period_s < maximum_period_s:
            raise ValueError("invalid period bounds")
        self.minimum_period_s = float(minimum_period_s)
        self.maximum_period_s = float(maximum_period_s)
        self.history_s = max(float(history_s), self.maximum_period_s * minimum_cycles)
        self.minimum_cycles = float(minimum_cycles)
        self.reset()

    def reset(self) -> None:
        self.samples: deque[tuple[float, np.ndarray]] = deque(maxlen=4096)
        self.last_time_s: float | None = None
        self.estimate = RotationEstimate()

    def update(self, time_s: float, features: np.ndarray | None, *,
               stationary: bool = False) -> RotationEstimate:
        if not math.isfinite(time_s):
            self.reset()
            return self.estimate
        if stationary:
            self.reset()
            self.estimate = RotationEstimate(reason='scene_stationary')
            return self.estimate
        if self.last_time_s is not None and (time_s <= self.last_time_s or
                                            time_s - self.last_time_s > .75):
            self.reset()
        self.last_time_s = float(time_s)
        if features is None:
            self.estimate = RotationEstimate(samples=len(self.samples),
                                             reason="texture_features_unavailable")
            return self.estimate
        vector = np.asarray(features, dtype=np.float64).reshape(-1)
        if vector.size < 2 or not np.all(np.isfinite(vector)):
            self.estimate = RotationEstimate(samples=len(self.samples),
                                             reason="texture_features_invalid")
            return self.estimate
        if self.samples and vector.size != self.samples[-1][1].size:
            self.reset()
            self.last_time_s = float(time_s)
        self.samples.append((float(time_s), vector.copy()))
        while self.samples and time_s - self.samples[0][0] > self.history_s:
            self.samples.popleft()
        self.estimate = self._estimate()
        return self.estimate

    def _estimate(self) -> RotationEstimate:
        if len(self.samples) < 24:
            return RotationEstimate(samples=len(self.samples),
                                    reason="insufficient_samples")
        times = np.asarray([sample[0] for sample in self.samples], np.float64)
        duration = float(times[-1] - times[0])
        required = self.maximum_period_s * self.minimum_cycles
        if duration < required:
            return RotationEstimate(samples=len(self.samples),
                                    reason="insufficient_cycles")
        differences = np.diff(times)
        if np.max(differences) > .75:
            return RotationEstimate(samples=len(self.samples),
                                    reason='sampling_gap_too_large')
        valid_dt = differences[(differences > 1e-4) & (differences < 1.0)]
        if not valid_dt.size:
            return RotationEstimate(samples=len(self.samples),
                                    reason="invalid_timestamps")
        dt = float(np.median(valid_dt))
        grid = np.arange(times[0], times[-1] + .25 * dt, dt)
        if grid.size < 24:
            return RotationEstimate(samples=len(self.samples),
                                    reason="insufficient_resampled_samples")
        raw = np.stack([sample[1] for sample in self.samples], axis=0)
        recent = raw[times >= times[-1] - max(.6, 4 * dt)]
        if len(recent) >= 4 and np.count_nonzero(np.std(recent, axis=0) > .02) < 3:
            return RotationEstimate(samples=len(self.samples),
                                    reason='static_or_low_texture')
        matrix = np.column_stack([
            np.interp(grid, times, raw[:, index])
            for index in range(raw.shape[1])
        ])
        # Remove mean and linear drift per feature. Uniform illumination changes
        # are already reduced by flange_texture_features; this rejects slow lamp
        # drift and camera auto-exposure remnants.
        x = np.linspace(-1.0, 1.0, matrix.shape[0])
        design = np.column_stack((np.ones_like(x), x))
        matrix -= design @ np.linalg.lstsq(design, matrix, rcond=None)[0]
        standard = np.std(matrix, axis=0)
        usable = standard > 0.08
        if int(np.count_nonzero(usable)) < 3:
            return RotationEstimate(samples=len(self.samples),
                                    reason="static_or_low_texture")
        matrix = matrix[:, usable] / standard[usable]
        minimum_lag = max(2, int(round(self.minimum_period_s / dt)))
        maximum_lag = min(matrix.shape[0] // 2,
                          int(round(self.maximum_period_s / dt)))
        if maximum_lag <= minimum_lag + 2:
            return RotationEstimate(samples=len(self.samples),
                                    reason="period_search_window_too_short")
        correlations = []
        for lag in range(minimum_lag, maximum_lag + 1):
            products = matrix[:-lag] * matrix[lag:]
            correlations.append(float(np.median(np.mean(products, axis=0))))
        correlations = np.asarray(correlations)
        # Prefer a genuine local maximum; selecting the search boundary is a
        # common false positive for drift/non-periodic motion.
        local = np.flatnonzero((correlations[1:-1] > correlations[:-2]) &
                               (correlations[1:-1] >= correlations[2:])) + 1
        if not local.size:
            return RotationEstimate(samples=len(self.samples),
                                    reason="no_autocorrelation_peak")
        peak_index = int(local[np.argmax(correlations[local])])
        peak = float(correlations[peak_index])
        baseline = float(np.median(correlations))
        prominence = peak - baseline
        # Refine the ACF peak instead of quantizing the period to one frame.
        left, centre, right = correlations[peak_index - 1:peak_index + 2]
        denominator = left - 2.0 * centre + right
        fraction = (float(np.clip(.5 * (left - right) / denominator, -.5, .5))
                    if abs(denominator) > 1e-9 else 0.0)
        period = (minimum_lag + peak_index + fraction) * dt

        frequencies = np.fft.rfftfreq(matrix.shape[0], dt)
        spectrum = np.median(np.abs(np.fft.rfft(matrix, axis=0)) ** 2, axis=1)
        band = ((frequencies >= 1.0 / self.maximum_period_s) &
                (frequencies <= 1.0 / self.minimum_period_s))
        if not np.any(band):
            return RotationEstimate(samples=len(self.samples),
                                    reason="frequency_band_unavailable")
        band_indices = np.flatnonzero(band)
        spectral_index = int(band_indices[np.argmax(spectrum[band])])
        spectral_period = 1.0 / float(frequencies[spectral_index])
        agreement = math.exp(-abs(spectral_period - period) /
                             max(.15, .22 * period))
        spectral_contrast = float(spectrum[spectral_index] /
                                  (np.median(spectrum[band]) + 1e-9))
        confidence = max(0.0, min(1.0,
            .45 * max(0.0, min(1.0, prominence / .35)) +
            .30 * max(0.0, min(1.0, (peak - .15) / .55)) +
            .25 * agreement))
        if peak < .25 or prominence < .10 or spectral_contrast < 1.35 or agreement < .35:
            return RotationEstimate(samples=len(self.samples),
                                    confidence=confidence,
                                    reason="periodicity_not_reliable",
                                    candidate_period_s=period,
                                    spectral_period_s=spectral_period,
                                    autocorrelation_peak=peak,
                                    autocorrelation_prominence=prominence,
                                    spectral_contrast=spectral_contrast,
                                    method_agreement=agreement)
        return RotationEstimate(status="measured_markerless", period_s=period,
                                rpm=60.0 / period, confidence=confidence,
                                samples=len(self.samples), reason="ok",
                                candidate_period_s=period,
                                spectral_period_s=spectral_period,
                                autocorrelation_peak=peak,
                                autocorrelation_prominence=prominence,
                                spectral_contrast=spectral_contrast,
                                method_agreement=agreement)


class ArucoRotationEstimator:
    """Measure complete turns from one uniquely identified flange marker.

    The orientation of the marker's first edge winds by 2*pi per mechanical
    turn even when the marker stays visible for the whole revolution.  Missing
    detections never add angle or fabricate a pulse.
    """

    def __init__(self, marker_id: int = 17, minimum_period_s: float = .8,
                 maximum_period_s: float = 10.0) -> None:
        self.marker_id = int(marker_id)
        self.minimum_period_s = float(minimum_period_s)
        self.maximum_period_s = float(maximum_period_s)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detector = cv2.aruco.ArucoDetector(
            dictionary, cv2.aruco.DetectorParameters())
        self.reset()

    def reset(self) -> None:
        self.last_time_s: float | None = None
        self.last_angle: float | None = None
        self.accumulated_angle = 0.0
        self.last_turn_angle = 0.0
        self.last_turn_time_s: float | None = None
        self.periods: deque[float] = deque(maxlen=5)
        self.detected_frames = 0
        self.total_frames = 0
        self.estimate = RotationEstimate(reason='marker_not_yet_detected')

    def update(self, time_s: float, image_bgr: np.ndarray) -> RotationEstimate:
        if not math.isfinite(time_s):
            self.reset()
            return self.estimate
        if self.last_time_s is not None and time_s <= self.last_time_s:
            self.reset()
        self.total_frames += 1
        corners, ids, _ = self.detector.detectMarkers(image_bgr)
        selected = None
        if ids is not None:
            for marker_corners, marker in zip(corners, ids.reshape(-1)):
                if int(marker) == self.marker_id:
                    selected = marker_corners.reshape(4, 2)
                    break
        if selected is None:
            # A long loss invalidates angular continuity; retaining old phase
            # across an unobserved turn could create a false revolution.
            if self.last_time_s is not None and time_s - self.last_time_s > .4:
                self.last_angle = None
            self.estimate = RotationEstimate(samples=self.detected_frames,
                                             reason='marker_not_visible')
            return self.estimate
        self.detected_frames += 1
        edge = selected[1] - selected[0]
        angle = math.atan2(float(edge[1]), float(edge[0]))
        if self.last_angle is not None and self.last_time_s is not None:
            delta = (angle - self.last_angle + math.pi) % (2 * math.pi) - math.pi
            dt = time_s - self.last_time_s
            # At the expected site speed/frame rate a real step is far below
            # this limit. Larger jumps mean a lost/corner-swapped detection.
            if 0 < dt <= .4 and abs(delta) <= 1.2:
                previous = self.accumulated_angle
                self.accumulated_angle += delta
                turns_before = math.floor(abs(previous) / (2 * math.pi))
                turns_after = math.floor(abs(self.accumulated_angle) / (2 * math.pi))
                if turns_after > turns_before:
                    if self.last_turn_time_s is not None:
                        period = time_s - self.last_turn_time_s
                        if self.minimum_period_s <= period <= self.maximum_period_s:
                            self.periods.append(period)
                    self.last_turn_time_s = time_s
                    self.last_turn_angle = self.accumulated_angle
        self.last_angle = angle
        self.last_time_s = time_s
        if not self.periods:
            self.estimate = RotationEstimate(samples=self.detected_frames,
                                             reason='marker_waiting_for_two_passes')
            return self.estimate
        period = float(np.median(np.asarray(self.periods)))
        variation = (float(np.std(np.asarray(self.periods))) / period
                     if len(self.periods) > 1 else .12)
        visibility = min(1.0, self.detected_frames / max(1, self.total_frames))
        confidence = max(0.0, min(1.0, visibility * (1.0 - min(.8, variation))))
        self.estimate = RotationEstimate(
            status='measured_marker', period_s=period, rpm=60.0 / period,
            confidence=confidence, samples=self.detected_frames,
            reason='aruco_full_turns')
        return self.estimate


def fuse_rotation_estimates(estimates: list[RotationEstimate],
                            maximum_relative_disagreement: float = .18
                            ) -> RotationEstimate:
    """Fuse two camera estimates, rejecting a contradictory pair."""
    valid = [estimate for estimate in estimates
             if estimate.status in ('measured_markerless', 'measured_marker')
             and math.isfinite(estimate.period_s) and estimate.period_s > 0]
    if not valid:
        if estimates and all(item.reason == 'scene_stationary' for item in estimates):
            return RotationEstimate(reason='scene_stationary')
        return RotationEstimate(reason='no_camera_has_reliable_period')
    if len(valid) == 1:
        return valid[0]
    shortest, longest = min(valid, key=lambda item: item.period_s), max(
        valid, key=lambda item: item.period_s)
    disagreement = ((longest.period_s - shortest.period_s) /
                    max(shortest.period_s, 1e-9))
    if disagreement > maximum_relative_disagreement:
        return RotationEstimate(
            confidence=min(item.confidence for item in valid),
            samples=sum(item.samples for item in valid),
            reason='camera_period_conflict')
    weights = np.asarray([max(.05, item.confidence) for item in valid])
    period = float(np.average([item.period_s for item in valid], weights=weights))
    status = ('measured_marker' if any(item.status == 'measured_marker'
                                      for item in valid)
              else 'measured_markerless')
    return RotationEstimate(status, period, 60.0 / period,
                            float(min(item.confidence for item in valid)),
                            sum(item.samples for item in valid),
                            'two_cameras_agree')


@dataclass(frozen=True)
class FirstWrapDecision:
    phase: str
    advisory: bool
    reason: str
    edge: str | None = None
    new_layer_start_s: float = math.nan
    expected_force_time_s: float = math.nan
    time_to_force_s: float = math.nan
    rotation_period_s: float = math.nan
    rotation_confidence: float = 0.0


@dataclass(frozen=True)
class EdgeGapTransitionDecision:
    """Causal event evidence from one camera-relative axial gap trajectory."""

    phase: str
    edge: str
    edge_armed_s: float = math.nan
    edge_arrival_s: float = math.nan
    new_layer_start_s: float = math.nan
    reversal_confirmed_s: float = math.nan
    filtered_gap: float = math.nan
    slope_rope_s: float = math.nan
    reason: str = "waiting_for_approach"


class EdgeGapTransitionTracker:
    """Detect approach, edge dwell and departure without requiring direction first.

    The long, median-stabilized gap supplies approach/edge evidence.  Departure
    uses a short upper envelope of raw candidates because flange overlap tends
    to create intermittent false zero gaps after the rope has started leaving.
    No missing frame is converted into a measurement, and confirmation remains
    delayed until a sustained positive trend exists.
    """

    def __init__(self, edge: str, edge_arm_gap: float = 1.5,
                 near_edge_gap: float = .6, history_s: float = 12.0,
                 upper_envelope_s: float = 1.0,
                 minimum_trend_span_s: float = 1.5,
                 minimum_slope_rope_s: float = .12,
                 minimum_departure_confirm_s: float = 1.5,
                 departure_coverage_window_s: float = 2.0,
                 minimum_departure_coverage: float = .70,
                 maximum_event_duration_s: float = 12.0,
                 maximum_missing_s: float = 8.0) -> None:
        if edge not in ("left", "right"):
            raise ValueError("edge must be left or right")
        self.edge = edge
        self.edge_arm_gap = float(edge_arm_gap)
        self.near_edge_gap = float(near_edge_gap)
        self.history_s = float(history_s)
        self.upper_envelope_s = float(upper_envelope_s)
        self.minimum_trend_span_s = float(minimum_trend_span_s)
        self.minimum_slope_rope_s = float(minimum_slope_rope_s)
        self.minimum_departure_confirm_s = float(minimum_departure_confirm_s)
        self.departure_coverage_window_s = float(departure_coverage_window_s)
        self.minimum_departure_coverage = float(minimum_departure_coverage)
        self.maximum_event_duration_s = float(maximum_event_duration_s)
        self.maximum_missing_s = float(maximum_missing_s)
        self.reset()

    def reset(self) -> None:
        self.phase = "SEEKING_EDGE"
        self.stable_history: deque[tuple[float, float]] = deque(maxlen=512)
        self.raw_history: deque[tuple[float, float]] = deque(maxlen=512)
        self.upper_history: deque[tuple[float, float]] = deque(maxlen=512)
        # Keep invalid observations as explicit zeros.  A sparse sequence of
        # plausible-looking rope candidates must not be promoted to a physical
        # reversal when the contact branch is missing in most nearby frames.
        self.coverage_history: deque[tuple[float, float]] = deque(maxlen=512)
        self.last_time_s: float | None = None
        self.last_measurement_s: float | None = None
        self.edge_armed_s = math.nan
        self.edge_arrival_s = math.nan
        self.last_near_edge_s = math.nan
        self.minimum_gap = math.inf
        self.minimum_gap_s = math.nan
        self.new_layer_start_s = math.nan
        self.reversal_confirmed_s = math.nan
        self.departure_candidate_s = math.nan

    @staticmethod
    def _robust_slope(samples: list[tuple[float, float]],
                      minimum_span_s: float,
                      minimum_samples: int = 6) -> float:
        if (len(samples) < minimum_samples or
                samples[-1][0] - samples[0][0] < minimum_span_s):
            return math.nan
        slopes = [(yb - ya) / (tb - ta)
                  for index, (ta, ya) in enumerate(samples[:-1])
                  for tb, yb in samples[index + 1:]
                  if tb - ta >= .4]
        return float(np.median(np.asarray(slopes))) if slopes else math.nan

    @staticmethod
    def _valid_count(value: float | None) -> bool:
        return value is not None and math.isfinite(value) and 0 <= value <= 100

    def _prune(self, time_s: float) -> None:
        for history in (self.stable_history, self.raw_history, self.upper_history,
                        self.coverage_history):
            while history and time_s - history[0][0] > self.history_s:
                history.popleft()

    def _window(self, history: deque[tuple[float, float]], time_s: float,
                duration_s: float) -> list[tuple[float, float]]:
        return [item for item in history if time_s - item[0] <= duration_s]

    def update(self, time_s: float, raw_gap: float | None,
               stable_gap: float | None, measured: bool = True
               ) -> EdgeGapTransitionDecision:
        if not math.isfinite(time_s):
            self.reset()
            return self._decision(reason="invalid_timestamp")
        if self.last_time_s is not None and time_s <= self.last_time_s:
            self.reset()
        self.last_time_s = float(time_s)
        raw_valid = measured and self._valid_count(raw_gap)
        stable_valid = measured and self._valid_count(stable_gap)
        self.coverage_history.append(
            (time_s, 1.0 if raw_valid and stable_valid else 0.0))
        if raw_valid:
            self.raw_history.append((time_s, float(raw_gap)))
            recent_raw = self._window(
                self.raw_history, time_s, self.upper_envelope_s)
            if len(recent_raw) >= 2:
                upper = float(np.percentile(
                    np.asarray([value for _, value in recent_raw]), 75.0))
                self.upper_history.append((time_s, upper))
            self.last_measurement_s = time_s
        if stable_valid:
            stable = float(stable_gap)
            self.stable_history.append((time_s, stable))
            self.last_measurement_s = time_s
            if stable < self.minimum_gap:
                self.minimum_gap = stable
                self.minimum_gap_s = time_s
        self._prune(time_s)
        if (self.last_measurement_s is not None and
                time_s - self.last_measurement_s > self.maximum_missing_s):
            self.reset()
            self.last_time_s = float(time_s)
            return self._decision(reason="measurement_gap_reset")

        stable_window = self._window(self.stable_history, time_s, 5.0)
        approach_slope = self._robust_slope(
            stable_window, self.minimum_trend_span_s)
        current_stable = stable_window[-1][1] if stable_window else math.nan
        if self.phase in ("SEEKING_EDGE", "COMPLETE"):
            if (stable_valid and current_stable <= self.edge_arm_gap and
                    math.isfinite(approach_slope) and
                    approach_slope <= -self.minimum_slope_rope_s):
                self.phase = "EDGE_REACHED"
                self.edge_armed_s = time_s
                self.edge_arrival_s = (time_s if current_stable <= self.near_edge_gap
                                       else math.nan)
                self.last_near_edge_s = (time_s if current_stable <= self.near_edge_gap
                                         else math.nan)
                self.minimum_gap = current_stable
                self.minimum_gap_s = time_s

        if self.phase == "EDGE_REACHED":
            if time_s - self.edge_armed_s > self.maximum_event_duration_s:
                # Latch the failed cycle until the rope is visibly well clear.
                # Immediately returning to SEEKING_EDGE lets residual texture
                # near the same flange arm a second, entirely fictitious event.
                self.phase = "EXPIRED_WAIT_CLEAR"
                self.stable_history.clear()
                self.raw_history.clear()
                self.upper_history.clear()
                self.coverage_history.clear()
                return self._decision(reason="edge_event_expired_without_reversal")
            if stable_valid and current_stable <= self.near_edge_gap:
                if not math.isfinite(self.edge_arrival_s):
                    self.edge_arrival_s = time_s
                self.last_near_edge_s = time_s
                self.departure_candidate_s = math.nan
            upper_window = self._window(self.upper_history, time_s, 5.0)
            # Only fit the rising branch after the most recent local minimum.
            # Including the preceding descent/edge dwell can move the zero
            # crossing several seconds early and can even flip the slope.
            minimum_index = (min(range(len(upper_window)),
                                 key=lambda index: upper_window[index][1])
                             if upper_window else 0)
            departure_window = upper_window[minimum_index:]
            departure_slope = self._robust_slope(
                departure_window, 1.2, minimum_samples=4)
            current_upper = upper_window[-1][1] if upper_window else math.nan
            coverage_window = self._window(
                self.coverage_history, time_s,
                self.departure_coverage_window_s)
            departure_coverage = (float(np.mean(np.asarray(
                [value for _, value in coverage_window], dtype=np.float64)))
                if coverage_window else 0.0)
            if (stable_valid and raw_valid and current_stable >= 1.2 and
                    math.isfinite(current_upper) and
                    current_upper >= self.edge_arm_gap):
                if not math.isfinite(self.departure_candidate_s):
                    self.departure_candidate_s = time_s
            elif (math.isfinite(self.departure_candidate_s) and
                  (self.last_measurement_s is None or
                   time_s - self.last_measurement_s > .8)):
                self.departure_candidate_s = math.nan
            if (time_s - self.edge_armed_s >= 1.0 and
                    raw_valid and upper_window and
                    time_s - upper_window[-1][0] <= .4 and
                    math.isfinite(self.departure_candidate_s) and
                    time_s - self.departure_candidate_s >=
                    self.minimum_departure_confirm_s and
                    departure_coverage >= self.minimum_departure_coverage and
                    math.isfinite(departure_slope) and
                    departure_slope >= self.minimum_slope_rope_s and
                    math.isfinite(current_upper) and
                    current_upper >= self.edge_arm_gap):
                self.phase = "REVERSAL_CONFIRMED"
                self.reversal_confirmed_s = time_s
                # A plateau at the flange is common.  The last trustworthy
                # near-edge observation is a better layer-start estimate than
                # the much later direction-confirmation frame.
                departure_intercept = float(np.median(np.asarray([
                    value - departure_slope * sample_time
                    for sample_time, value in departure_window
                ])))
                crossing = -departure_intercept / departure_slope
                lower_bound = (self.edge_arrival_s
                               if math.isfinite(self.edge_arrival_s)
                               else self.edge_armed_s)
                crossing = min(time_s, max(lower_bound, crossing))
                # Use the rising-branch zero crossing itself.  A late false
                # zero from flange texture must not drag the start forward;
                # the crossing is already bounded by arrival and confirmation.
                self.new_layer_start_s = crossing
                return self._decision(departure_slope,
                                      "sustained_departure_after_edge_dwell")
            if (math.isfinite(self.departure_candidate_s) and
                    departure_coverage < self.minimum_departure_coverage):
                return self._decision(
                    departure_slope,
                    "departure_measurement_coverage_too_low")
            return self._decision(departure_slope,
                                  "edge_reached_waiting_for_sustained_departure")
        if self.phase == "REVERSAL_CONFIRMED":
            if stable_valid and current_stable >= 4.0:
                self.phase = "COMPLETE"
            return self._decision(reason="reversal_confirmed")
        if self.phase == "EXPIRED_WAIT_CLEAR":
            if stable_valid and current_stable >= 4.0:
                self.reset()
                self.last_time_s = float(time_s)
                return self._decision(reason="expired_edge_cycle_cleared")
            return self._decision(reason="expired_edge_cycle_waiting_for_clearance")
        return self._decision(approach_slope, "waiting_for_measured_edge_approach")

    def _decision(self, slope: float = math.nan, reason: str = "waiting"):
        filtered = (self.stable_history[-1][1]
                    if self.stable_history else math.nan)
        return EdgeGapTransitionDecision(
            self.phase, self.edge, self.edge_armed_s, self.edge_arrival_s,
            self.new_layer_start_s, self.reversal_confirmed_s,
            filtered, slope, reason)


class RotationEstimateHold:
    """Briefly bridge intermittent feature failures after a real period estimate."""

    def __init__(self, maximum_age_s: float = 1.0) -> None:
        if maximum_age_s <= 0:
            raise ValueError("maximum_age_s must be positive")
        self.maximum_age_s = float(maximum_age_s)
        self.reset()

    def reset(self) -> None:
        self.last_time_s: float | None = None
        self.last_valid_time_s: float | None = None
        self.last_valid = RotationEstimate()

    def update(self, time_s: float, estimate: RotationEstimate) -> RotationEstimate:
        if not math.isfinite(time_s):
            self.reset()
            return RotationEstimate(reason="invalid_timestamp")
        if self.last_time_s is not None and time_s <= self.last_time_s:
            self.reset()
        self.last_time_s = float(time_s)
        if estimate.reason in ('scene_stationary', 'camera_period_conflict',
                               'sampling_gap_too_large', 'harmonic_ambiguity'):
            self.reset()
            return estimate
        valid = (estimate.status in ("measured_markerless", "measured_marker")
                 and math.isfinite(estimate.period_s) and estimate.period_s > 0
                 and estimate.confidence > 0)
        if valid:
            self.last_valid_time_s = time_s
            self.last_valid = estimate
            return estimate
        if self.last_valid_time_s is None:
            return estimate
        age = time_s - self.last_valid_time_s
        if age > self.maximum_age_s:
            return estimate
        confidence = self.last_valid.confidence * max(
            0.0, 1.0 - .35 * age / self.maximum_age_s)
        return RotationEstimate(
            self.last_valid.status, self.last_valid.period_s,
            self.last_valid.rpm, confidence, self.last_valid.samples,
            f"held_recent_{age:.3f}s_after:{estimate.reason}")


class FirstWrapTimingTracker:
    """Fuse an observed edge reversal with a measured visual turn period.

    ``remaining_gap`` excludes the active rope while approaching.  After the
    reversal ``departed_count`` uses the operator convention and includes it.
    The tracker never substitutes axial pitch for 360 degrees.
    """

    def __init__(self, edge_arm_gap: float = 1.5,
                 window_before_s: float = .12, window_after_s: float = .30,
                 minimum_rotation_confidence: float = .55) -> None:
        self.edge_arm_gap = float(edge_arm_gap)
        self.window_before_s = float(window_before_s)
        self.window_after_s = float(window_after_s)
        self.minimum_rotation_confidence = float(minimum_rotation_confidence)
        self.reset()

    def reset(self) -> None:
        self.phase = "SEEKING_EDGE"
        self.edge: str | None = None
        self.closest_time_s = math.nan
        self.closest_gap = math.inf
        self.new_layer_start_s = math.nan
        self.completed = False
        self.last_time_s: float | None = None
        self.approach_samples: deque[tuple[float, float]] = deque(maxlen=80)
        self.leaving_samples: deque[tuple[float, float]] = deque(maxlen=80)

    @staticmethod
    def _robust_line(samples: list[tuple[float, float]]) -> tuple[float, float] | None:
        """Theil-Sen-like line fit for a short causal event window."""
        if len(samples) < 3:
            return None
        slopes = []
        for index, (ta, ya) in enumerate(samples[:-1]):
            for tb, yb in samples[index + 1:]:
                if tb - ta >= .04:
                    slopes.append((yb - ya) / (tb - ta))
        if not slopes:
            return None
        slope = float(np.median(np.asarray(slopes)))
        intercept = float(np.median(
            np.asarray([value - slope * time_s for time_s, value in samples])))
        return slope, intercept

    def _refine_transition_time(self) -> None:
        before = list(self.approach_samples)[-20:]
        after = list(self.leaving_samples)[-20:]
        approaching = self._robust_line(before)
        leaving = self._robust_line(after)
        if approaching is None or leaving is None:
            return
        approach_slope, approach_intercept = approaching
        leave_slope, leave_intercept = leaving
        if approach_slope >= -.03 or leave_slope <= .03:
            return
        denominator = approach_slope - leave_slope
        if abs(denominator) < 1e-6:
            return
        crossing = (leave_intercept - approach_intercept) / denominator
        earliest = before[-1][0] - 1.0
        latest = after[0][0] + 1.0
        if not earliest <= crossing <= latest:
            return
        crossing_gap = approach_slope * crossing + approach_intercept
        if crossing_gap < -1.0 or crossing_gap > self.edge_arm_gap + .5:
            return
        self.new_layer_start_s = float(crossing)

    def update(self, time_s: float, global_direction: str,
               remaining_gap: float | None, departed_count: float | None,
               period: RotationEstimate) -> FirstWrapDecision:
        if not math.isfinite(time_s):
            self.reset()
            return self._decision(False, "invalid_timestamp", period)
        if period.reason == 'scene_stationary':
            self.reset()
            return self._decision(False, 'scene_stationary', period)
        if self.last_time_s is not None and time_s <= self.last_time_s:
            self.reset()
        self.last_time_s = float(time_s)
        direction = global_direction if global_direction in ("left", "right") else None
        valid_remaining = (remaining_gap is not None and
                           math.isfinite(remaining_gap) and remaining_gap >= 0)
        valid_departed = (departed_count is not None and
                          math.isfinite(departed_count) and departed_count >= 1)

        if direction and valid_remaining and (
                self.edge is None or direction == self.edge):
            self.approach_samples.append((time_s, float(remaining_gap)))
            while (self.approach_samples and
                   time_s - self.approach_samples[0][0] > 4.0):
                self.approach_samples.popleft()
        if self.edge and direction and direction != self.edge and valid_departed:
            self.leaving_samples.append((time_s, max(0.0, float(departed_count) - 1.0)))
            while (self.leaving_samples and
                   time_s - self.leaving_samples[0][0] > 4.0):
                self.leaving_samples.popleft()

        if (self.phase == 'COMPLETE' and direction and valid_remaining and
                remaining_gap <= self.edge_arm_gap and direction != self.edge):
            self.phase = 'EDGE_REACHED'
            self.edge = direction
            self.closest_time_s = time_s
            self.closest_gap = float(remaining_gap)
            self.new_layer_start_s = math.nan
            self.completed = False
            self.approach_samples.clear()
            self.leaving_samples.clear()
            self.approach_samples.append((time_s, float(remaining_gap)))

        if self.phase == "SEEKING_EDGE" and direction and valid_remaining:
            if remaining_gap <= self.edge_arm_gap:
                self.phase = "EDGE_REACHED"
                self.edge = direction
                self.closest_time_s = time_s
                self.closest_gap = float(remaining_gap)
        elif self.phase == "EDGE_REACHED":
            if direction == self.edge and valid_remaining:
                if remaining_gap <= self.closest_gap:
                    self.closest_gap = float(remaining_gap)
                    self.closest_time_s = time_s
            elif direction and self.edge and direction != self.edge and valid_departed:
                # The minimum observed near-edge gap is a less delayed estimate
                # of the physical layer transition than the eventual robust
                # direction confirmation.
                self.new_layer_start_s = (self.closest_time_s if
                                          math.isfinite(self.closest_time_s)
                                          else time_s)
                self.phase = "FIRST_REVOLUTION"
                self._refine_transition_time()

        if self.phase == "FIRST_REVOLUTION":
            self._refine_transition_time()
            if period.status not in ("measured_markerless", "measured_marker") or not math.isfinite(period.period_s):
                return self._decision(False, "rotation_period_unavailable", period)
            if period.confidence < self.minimum_rotation_confidence:
                return self._decision(False, "rotation_period_low_confidence", period)
            expected = self.new_layer_start_s + period.period_s
            delta = expected - time_s
            if delta < -self.window_after_s:
                self.phase = "COMPLETE"
                self.completed = True
                return self._decision(False, "first_wrap_window_missed", period)
            if -self.window_after_s <= delta <= self.window_before_s:
                self.phase = "FORCE_WINDOW"
                return self._decision(True, "one_visual_revolution_completed", period)
            return self._decision(False, "first_visual_revolution_in_progress", period)
        if self.phase == "FORCE_WINDOW":
            if (period.status not in ('measured_markerless', 'measured_marker')
                    or not math.isfinite(period.period_s)
                    or period.confidence < self.minimum_rotation_confidence):
                self.phase = 'FIRST_REVOLUTION'
                return self._decision(False, 'rotation_period_unavailable', period)
            expected = self.new_layer_start_s + period.period_s
            if time_s - expected <= self.window_after_s:
                return self._decision(True, "one_visual_revolution_completed", period)
            self.phase = "COMPLETE"
            self.completed = True
        return self._decision(False, {
            "SEEKING_EDGE": "waiting_for_measured_edge_approach",
            "EDGE_REACHED": "edge_reached_waiting_for_measured_reversal",
            "COMPLETE": "first_wrap_window_complete",
        }.get(self.phase, "not_ready"), period)

    def update_from_edge_transition(
            self, time_s: float, transition: EdgeGapTransitionDecision | None,
            period: RotationEstimate) -> FirstWrapDecision:
        """Advance from an independently measured per-edge gap transition.

        This path avoids the former circular dependency where a global motion
        direction had to be confirmed before the target-edge gap was allowed
        to arm the event tracker.  It still cannot open the force window unless
        an independent, sufficiently confident rotation period is available.
        """
        if not math.isfinite(time_s):
            self.reset()
            return self._decision(False, "invalid_timestamp", period)
        if period.reason == 'scene_stationary':
            self.reset()
            return self._decision(False, 'scene_stationary', period)
        if self.last_time_s is not None and time_s <= self.last_time_s:
            self.reset()
        self.last_time_s = float(time_s)
        if transition is not None:
            if (transition.phase == "EXPIRED_WAIT_CLEAR" and
                    self.phase == "EDGE_REACHED" and
                    (self.edge is None or transition.edge == self.edge)):
                # The per-edge detector explicitly rejected this cycle.  Do
                # not leave the downstream first-wrap timer armed forever.
                self.reset()
                self.last_time_s = float(time_s)
                return self._decision(
                    False, "edge_event_expired_without_reversal", period)
            if (transition.phase == "EDGE_REACHED" and
                    (self.phase in ("SEEKING_EDGE", "COMPLETE") or
                     transition.edge != self.edge)):
                self.phase = "EDGE_REACHED"
                self.edge = transition.edge
                self.closest_time_s = (
                    transition.edge_arrival_s
                    if math.isfinite(transition.edge_arrival_s)
                    else transition.edge_armed_s)
                self.closest_gap = transition.filtered_gap
                self.new_layer_start_s = math.nan
                self.completed = False
            if (transition.phase in ("REVERSAL_CONFIRMED", "COMPLETE") and
                    math.isfinite(transition.new_layer_start_s) and
                    self.phase in ("EDGE_REACHED", "SEEKING_EDGE")):
                self.edge = transition.edge
                self.new_layer_start_s = transition.new_layer_start_s
                self.phase = "FIRST_REVOLUTION"

        if self.phase == "FIRST_REVOLUTION":
            if (period.status not in ("measured_markerless", "measured_marker")
                    or not math.isfinite(period.period_s)):
                return self._decision(False, "rotation_period_unavailable", period)
            if period.confidence < self.minimum_rotation_confidence:
                return self._decision(False, "rotation_period_low_confidence", period)
            expected = self.new_layer_start_s + period.period_s
            delta = expected - time_s
            if delta < -self.window_after_s:
                self.phase = "COMPLETE"
                self.completed = True
                return self._decision(False, "first_wrap_window_missed", period)
            if -self.window_after_s <= delta <= self.window_before_s:
                self.phase = "FORCE_WINDOW"
                return self._decision(True, "one_visual_revolution_completed", period)
            return self._decision(False, "first_visual_revolution_in_progress", period)
        if self.phase == "FORCE_WINDOW":
            if (period.status not in ('measured_markerless', 'measured_marker')
                    or not math.isfinite(period.period_s)
                    or period.confidence < self.minimum_rotation_confidence):
                self.phase = 'FIRST_REVOLUTION'
                return self._decision(False, 'rotation_period_unavailable', period)
            expected = self.new_layer_start_s + period.period_s
            if time_s - expected <= self.window_after_s:
                return self._decision(True, "one_visual_revolution_completed", period)
            self.phase = "COMPLETE"
            self.completed = True
        return self._decision(False, {
            "SEEKING_EDGE": "waiting_for_measured_edge_approach",
            "EDGE_REACHED": "edge_reached_waiting_for_measured_reversal",
            "COMPLETE": "first_wrap_window_complete",
        }.get(self.phase, "not_ready"), period)

    def _decision(self, advisory: bool, reason: str,
                  period: RotationEstimate) -> FirstWrapDecision:
        expected = (self.new_layer_start_s + period.period_s
                    if math.isfinite(self.new_layer_start_s) and
                    math.isfinite(period.period_s) else math.nan)
        time_to = (expected - self.last_time_s if math.isfinite(expected) and
                   self.last_time_s is not None else math.nan)
        return FirstWrapDecision(
            self.phase, advisory, reason, self.edge, self.new_layer_start_s,
            expected, time_to, period.period_s, period.confidence)
