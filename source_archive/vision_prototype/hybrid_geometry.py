"""Original-resolution geometry constrained by coarse semantic regions.

This module is intentionally read-only.  Results are measurements for offline
and shadow-mode validation and are never actuator authorization.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math

import cv2
import numpy as np

from semantic_locator import SemanticRegions


@dataclass
class HybridMeasurement:
    valid: bool
    tangent_x: float = math.nan
    tangent_y: float = math.nan
    edge_x: float = math.nan
    edge_y: float = math.nan
    texture_period_px: float = math.nan
    pitch_px: float = math.nan
    remaining_rope_count: float = math.nan
    line_angle_deg: float = math.nan
    tangent_confidence: float = 0.0
    edge_confidence: float = 0.0
    pitch_confidence: float = 0.0
    # Independent bright/dark wound-row periodicity. Diagnostic only until
    # validated across rope gauges, layer depths and both cameras.
    row_pitch_candidate_px: float = math.nan
    row_pitch_candidate_confidence: float = 0.0
    row_pitch_count_candidate: float = math.nan
    source_line_end_x: float = math.nan
    source_line_end_y: float = math.nan
    reason: str = 'not_evaluated'


class HybridTemporalFilter:
    """Short causal median; invalid evidence is never converted into a value."""

    def __init__(self, window_frames: int = 5,
                 live_window_s: float = .7):
        if window_frames < 1:
            raise ValueError('window_frames must be positive')
        if live_window_s <= 0:
            raise ValueError('live_window_s must be positive')
        self.window_frames = int(window_frames)
        # Timestamped live processing is bounded by elapsed time, not by a
        # fixed frame count.  A five-item maxlen made the same 0.7 s filter
        # cover 1.0 s at 5 fps but only 0.5 s at 10 fps.
        self.history = deque()
        self.timestamps = deque()
        self.live_window_s = live_window_s

    def reset(self):
        self.history.clear()
        self.timestamps.clear()

    def update(self, measurement: HybridMeasurement,
               time_s: float | None = None) -> HybridMeasurement:
        if not measurement.valid:
            if time_s is None or not math.isfinite(time_s):
                self.reset()
            else:
                while self.timestamps and time_s - self.timestamps[0] > self.live_window_s:
                    self.timestamps.popleft()
                    self.history.popleft()
                if self.timestamps and time_s - self.timestamps[-1] > .6:
                    self.reset()
            return measurement
        if time_s is None and self.timestamps:
            self.reset()
        if time_s is not None:
            if self.history and not self.timestamps:
                self.reset()
            if not math.isfinite(time_s):
                self.reset()
                return replace(measurement, valid=False, reason='invalid_timestamp')
            if self.timestamps and (time_s <= self.timestamps[-1]
                                    or time_s - self.timestamps[-1] > .6):
                self.reset()
        self.history.append(measurement)
        if time_s is not None:
            self.timestamps.append(time_s)
            while self.timestamps and time_s - self.timestamps[0] > self.live_window_s:
                self.timestamps.popleft()
                self.history.popleft()
        else:
            while len(self.history) > self.window_frames:
                self.history.popleft()
        median = lambda field: float(np.median(
            [getattr(item, field) for item in self.history]))
        return replace(
            measurement,
            tangent_x=median('tangent_x'), tangent_y=median('tangent_y'),
            edge_x=median('edge_x'), edge_y=median('edge_y'),
            texture_period_px=median('texture_period_px'),
            pitch_px=median('pitch_px'),
            remaining_rope_count=median('remaining_rope_count'),
        )


class FrameStationarityDetector:
    """Require several low-motion frames before declaring a scene stationary.

    The detector deliberately reacts asymmetrically: motion clears the state on
    the first frame, while stillness has to persist.  It is only a gate for the
    diagnostic tangent branch stabilizer; it is not a machine-state or safety
    signal.
    """

    def __init__(self, maximum_mean_difference: float = 1.10,
                 maximum_p95_difference: float = 5.0,
                 confirmation_s: float = .45,
                 required_low_motion_pairs: int = 2,
                 maximum_gap_s: float = 2.5,
                 comparison_size: tuple[int, int] = (160, 120)):
        if maximum_mean_difference <= 0 or maximum_p95_difference <= 0:
            raise ValueError('stationarity thresholds must be positive')
        if confirmation_s < 0 or required_low_motion_pairs < 1:
            raise ValueError('stationarity confirmation must be nonnegative')
        self.maximum_mean_difference = float(maximum_mean_difference)
        self.maximum_p95_difference = float(maximum_p95_difference)
        self.confirmation_s = float(confirmation_s)
        self.required_low_motion_pairs = int(required_low_motion_pairs)
        self.maximum_gap_s = float(maximum_gap_s)
        self.comparison_size = tuple(map(int, comparison_size))
        self.reset()

    def reset(self):
        self.previous = None
        self.previous_time_s = None
        self.candidate_reference = None
        self.low_motion_since_s = None
        self.low_motion_pairs = 0
        self.last_mean_difference = math.nan
        self.last_p95_difference = math.nan

    def update(self, frame: np.ndarray, time_s: float | None) -> bool:
        if (time_s is None or not math.isfinite(time_s)
                or frame is None or frame.size == 0):
            self.reset()
            return False
        current = cv2.resize(frame, self.comparison_size,
                             interpolation=cv2.INTER_AREA)
        if current.ndim == 3:
            current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        current = cv2.GaussianBlur(current, (3, 3), 0)
        time_s = float(time_s)
        if (self.previous_time_s is not None
                and (time_s <= self.previous_time_s
                     or time_s - self.previous_time_s > self.maximum_gap_s)):
            self.reset()
        previous = self.previous
        previous_time_s = self.previous_time_s
        self.previous = current
        self.previous_time_s = time_s
        if previous is None or previous_time_s is None:
            return False
        difference = cv2.absdiff(current, previous)
        mean_difference = float(np.mean(difference))
        p95_difference = float(np.percentile(difference, 95))
        self.last_mean_difference = mean_difference
        self.last_p95_difference = p95_difference
        low_motion = (
            mean_difference <= self.maximum_mean_difference and
            p95_difference <= self.maximum_p95_difference)
        if not low_motion:
            # A single clear motion frame must immediately release any branch
            # lock so a real approach/reversal is not delayed by old history.
            self.candidate_reference = None
            self.low_motion_since_s = None
            self.low_motion_pairs = 0
            return False
        if self.low_motion_since_s is None:
            self.candidate_reference = previous.copy()
            self.low_motion_since_s = previous_time_s
            self.low_motion_pairs = 1
        else:
            # Adjacent frames can look similar during slow or periodically
            # sampled motion.  A true stopped scene must also remain close to
            # the fixed frame at which the stillness candidate began.
            reference_difference = cv2.absdiff(
                current, self.candidate_reference)
            if (float(np.mean(reference_difference)) >
                    self.maximum_mean_difference or
                    float(np.percentile(reference_difference, 95)) >
                    self.maximum_p95_difference):
                self.candidate_reference = None
                self.low_motion_since_s = None
                self.low_motion_pairs = 0
                return False
            self.low_motion_pairs += 1
        return (
            self.low_motion_pairs >= self.required_low_motion_pairs and
            time_s - self.low_motion_since_s >= self.confirmation_s)


class StaticTangentBranchStabilizer:
    """Resolve periodic tangent branches only after independent stillness.

    On a stopped cylinder the ridge endpoint detector can alternate between
    adjacent wound-rope ridges.  Measurements are clustered in the normalized
    edge-to-tangent direction (rope pitches), then the earliest *supported*
    branch is selected.  A remote, weak cluster cannot win merely because it
    has the smallest value.  No numerical truth value is embedded here.

    Invalid current evidence is always returned invalid.  Any lack of explicit
    stationarity immediately clears the history and returns the current raw
    path, preserving moving-scene latency.
    """

    def __init__(self, history_s: float = 5.0,
                 cluster_tolerance_rope: float = .60,
                 minimum_support: int = 3,
                 minimum_support_span_s: float = .15,
                 maximum_supported_branch_separation_rope: float = 3.25,
                 maximum_gap_s: float = 2.5):
        if history_s <= 0 or cluster_tolerance_rope <= 0:
            raise ValueError('static branch history and tolerance must be positive')
        if minimum_support < 2 or minimum_support_span_s < 0:
            raise ValueError('static branch support is invalid')
        self.history_s = float(history_s)
        self.cluster_tolerance_rope = float(cluster_tolerance_rope)
        self.minimum_support = int(minimum_support)
        self.minimum_support_span_s = float(minimum_support_span_s)
        self.maximum_supported_branch_separation_rope = float(
            maximum_supported_branch_separation_rope)
        self.maximum_gap_s = float(maximum_gap_s)
        self.reset()

    def reset(self):
        self.history = deque()
        self.previous_time_s = None
        self.locked_normalized_gap = math.nan
        self.locked_measurement = None

    @staticmethod
    def _normalized_gap(measurement: HybridMeasurement) -> float:
        if (not measurement.valid or not math.isfinite(measurement.tangent_y)
                or not math.isfinite(measurement.edge_y)
                or not math.isfinite(measurement.pitch_px)
                or measurement.pitch_px <= 0):
            return math.nan
        gap = (measurement.tangent_y - measurement.edge_y) / measurement.pitch_px
        return float(gap) if gap >= 0 and math.isfinite(gap) else math.nan

    def _clusters(self):
        ordered = sorted(self.history, key=lambda item: item[1])
        clusters = []
        for sample in ordered:
            if (not clusters or
                    sample[1] - float(np.median(
                        [item[1] for item in clusters[-1]])) >
                    self.cluster_tolerance_rope):
                clusters.append([sample])
            else:
                clusters[-1].append(sample)
        return clusters

    def _supported_clusters(self):
        supported = []
        for cluster in self._clusters():
            times = [item[0] for item in cluster]
            if (len(cluster) >= self.minimum_support and
                    max(times) - min(times) >= self.minimum_support_span_s):
                supported.append(cluster)
        return supported

    @staticmethod
    def _median_field(cluster, field: str) -> float:
        values = [getattr(item[2], field) for item in cluster]
        finite = [float(value) for value in values if math.isfinite(value)]
        return float(np.median(finite)) if finite else math.nan

    def _measurement_from_cluster(self, current: HybridMeasurement,
                                  cluster) -> HybridMeasurement:
        center = float(np.median([item[1] for item in cluster]))
        self.locked_normalized_gap = center
        self.locked_measurement = replace(
            current,
            tangent_x=self._median_field(cluster, 'tangent_x'),
            tangent_y=self._median_field(cluster, 'tangent_y'),
            edge_x=self._median_field(cluster, 'edge_x'),
            edge_y=self._median_field(cluster, 'edge_y'),
            texture_period_px=self._median_field(cluster, 'texture_period_px'),
            pitch_px=self._median_field(cluster, 'pitch_px'),
            remaining_rope_count=self._median_field(
                cluster, 'remaining_rope_count'),
            tangent_confidence=self._median_field(
                cluster, 'tangent_confidence'),
            edge_confidence=self._median_field(cluster, 'edge_confidence'),
            pitch_confidence=self._median_field(cluster, 'pitch_confidence'),
            row_pitch_candidate_px=self._median_field(
                cluster, 'row_pitch_candidate_px'),
            row_pitch_candidate_confidence=self._median_field(
                cluster, 'row_pitch_candidate_confidence'),
            row_pitch_count_candidate=self._median_field(
                cluster, 'row_pitch_count_candidate'),
            reason='static_supported_tangent_branch')
        return self.locked_measurement

    def _apply_locked_measurement(self,
                                  current: HybridMeasurement) -> HybridMeasurement:
        locked = self.locked_measurement
        if locked is None:
            return current
        return replace(
            current,
            tangent_x=locked.tangent_x, tangent_y=locked.tangent_y,
            edge_x=locked.edge_x, edge_y=locked.edge_y,
            texture_period_px=locked.texture_period_px,
            pitch_px=locked.pitch_px,
            remaining_rope_count=locked.remaining_rope_count,
            tangent_confidence=locked.tangent_confidence,
            edge_confidence=locked.edge_confidence,
            pitch_confidence=locked.pitch_confidence,
            row_pitch_candidate_px=locked.row_pitch_candidate_px,
            row_pitch_candidate_confidence=(
                locked.row_pitch_candidate_confidence),
            row_pitch_count_candidate=locked.row_pitch_count_candidate,
            reason='static_supported_tangent_branch')

    def update(self, measurement: HybridMeasurement,
               time_s: float | None,
               stationary_confirmed: bool) -> HybridMeasurement:
        if not stationary_confirmed:
            self.reset()
            return measurement
        if time_s is None or not math.isfinite(time_s):
            self.reset()
            return measurement
        time_s = float(time_s)
        if (self.previous_time_s is not None
                and (time_s <= self.previous_time_s
                     or time_s - self.previous_time_s > self.maximum_gap_s)):
            self.reset()
        self.previous_time_s = time_s
        # Never turn an occluded/invalid current frame into a valid value held
        # from history.  This is important even in the read-only shadow path,
        # because downstream event logic must see missing evidence as missing.
        if not measurement.valid:
            return measurement
        normalized_gap = self._normalized_gap(measurement)
        if not math.isfinite(normalized_gap):
            return measurement
        self.history.append((time_s, normalized_gap, measurement))
        while self.history and time_s - self.history[0][0] > self.history_s:
            self.history.popleft()
        supported = self._supported_clusters()
        if not supported:
            return self._apply_locked_measurement(measurement)
        # Establish the locally credible family from the best-supported mode.
        # Among similarly supported nearby periodic branches, choose the first
        # contact in normalized edge-to-cylinder direction (minimum gap).
        dominant = max(
            supported,
            key=lambda cluster: (
                len(cluster),
                max(item[0] for item in cluster) - min(item[0] for item in cluster)))
        dominant_center = float(np.median([item[1] for item in dominant]))
        credible = [
            cluster for cluster in supported
            if abs(float(np.median([item[1] for item in cluster])) -
                   dominant_center) <=
            self.maximum_supported_branch_separation_rope]
        selected = min(
            credible,
            key=lambda cluster: float(np.median([item[1] for item in cluster])))
        selected_center = float(np.median([item[1] for item in selected]))
        # Once the earliest supported branch is established in an independently
        # stationary scene, a later periodic ridge cannot replace it merely
        # because the earlier ridge was missed for a few frames.  A newly
        # supported *earlier* nearby branch can still refine the contact.
        if (self.locked_measurement is None or
                selected_center < (self.locked_normalized_gap -
                                   self.cluster_tolerance_rope) or
                abs(selected_center - self.locked_normalized_gap) <=
                self.cluster_tolerance_rope):
            return self._measurement_from_cluster(measurement, selected)
        return self._apply_locked_measurement(measurement)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8)
    if count <= 1:
        return mask.astype(bool)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == label


def _fit_source_line(regions: SemanticRegions):
    raw = regions.mask('incoming_rope_free_span').astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    if count <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    x1, y1, x2, y2 = regions.crop_xyxy
    low_h, low_w = raw.shape

    def source_points(label: int) -> np.ndarray:
        ys, xs = np.nonzero(labels == label)
        sx = x1 + (xs + .5) * (x2 - x1) / low_w
        sy = y1 + (ys + .5) * (y2 - y1) / low_h
        return np.column_stack([sx, sy]).astype(np.float32)

    # The network can produce horizontal false positives on the control box or
    # frame rails. Fitting every nontrivial component together drags the line
    # endpoint hundreds of pixels away from the actual rope. First select the
    # longest thin component with a plausible canonical rope angle, then only
    # merge fragments that are geometrically collinear with it.
    candidates = []
    minimum_span = .08 * (x2 - x1)
    for label, area in enumerate(areas, start=1):
        if area < 8:
            continue
        component = source_points(label)
        if len(component) < 8:
            continue
        vx, vy, cx, cy = cv2.fitLine(
            component, cv2.DIST_HUBER, 0, .01, .01).flatten()
        if vx < 0:
            vx, vy = -vx, -vy
        angle = math.degrees(math.atan2(float(vy), float(vx)))
        span = float(np.ptp(component[:, 0]))
        perpendicular = np.abs(
            -vy * (component[:, 0] - cx) + vx * (component[:, 1] - cy))
        residual = float(np.median(perpendicular))
        if (-38.0 <= angle <= -1.0 and span >= minimum_span
                and residual <= 28.0):
            candidates.append((span * math.exp(-residual / 20.0), label,
                               float(vx), float(vy), float(cx), float(cy),
                               residual))
    if not candidates:
        return None
    _score, best_label, vx, vy, cx, cy, residual = max(candidates)
    selected = [source_points(best_label)]
    minimum_area = max(8, int(round(.01 * areas.max())))
    merge_tolerance = max(14.0, 2.5 * residual)
    for label, area in enumerate(areas, start=1):
        if label == best_label or area < minimum_area:
            continue
        component = source_points(label)
        perpendicular = np.abs(
            -vy * (component[:, 0] - cx) + vx * (component[:, 1] - cy))
        if float(np.median(perpendicular)) <= merge_tolerance:
            selected.append(component)
    points = np.concatenate(selected, axis=0)
    if len(points) < 80:
        return None
    vx, vy, x0, y0 = cv2.fitLine(points[::2], cv2.DIST_HUBER, 0, .01, .01).flatten()
    if vx < 0:
        vx, vy = -vx, -vy
    slope = float(vy / max(vx, 1e-6))
    predicted = y0 + slope * (points[:, 0] - x0)
    residual = float(np.median(np.abs(points[:, 1] - predicted)))
    # The left tail of a thin semantic mask is unstable.  A low percentile is
    # a reproducible coarse search anchor, not the final tangent.
    coarse_x = float(np.percentile(points[:, 0], .20))
    return float(vx), float(vy), float(x0), float(y0), coarse_x, residual


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return cv2.blur(values.reshape(1, -1).astype(np.float32), (window, 1))[0]


def _refine_tangent(frame_bgr: np.ndarray, line):
    vx, vy, x0, y0, coarse_x, line_residual = line
    height, width = frame_bgr.shape[:2]
    slope = vy / max(vx, 1e-6)
    start = max(0, int(round(coarse_x - 120)))
    stop = min(width - 1, int(round(coarse_x + 60)))
    if stop - start < 80:
        return None
    x_values = np.arange(start, stop + 1, dtype=np.float32)
    center_y = y0 + slope * (x_values - x0)
    normal = np.asarray([-vy, vx], dtype=np.float32)
    offsets = np.arange(-44, 45, dtype=np.float32)
    map_x = x_values[:, None] + normal[0] * offsets[None, :]
    map_y = center_y[:, None].astype(np.float32) + normal[1] * offsets[None, :]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    gray_profiles = cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    sat_profiles = cv2.remap(saturation, map_x, map_y, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    center = gray_profiles.shape[1] // 2
    core = slice(center - 10, center + 11)
    side = np.r_[0:center - 22, center + 23:gray_profiles.shape[1]]
    ridge = np.percentile(gray_profiles[:, core], 80, axis=1) - np.median(
        gray_profiles[:, side], axis=1)
    neutral = np.median(sat_profiles[:, side], axis=1) - np.percentile(
        sat_profiles[:, core], 30, axis=1)
    score = _rolling_mean(np.maximum(ridge, 0), 17) + .18 * _rolling_mean(
        np.maximum(neutral, 0), 17)
    smooth = _rolling_mean(score, 13)
    offset = 14
    before = np.r_[np.repeat(smooth[0], offset), smooth[:-offset]]
    after = np.r_[smooth[offset:], np.repeat(smooth[-1], offset)]
    change = after - before
    # Prefer the earliest credible onset near the coarse endpoint.  A strong
    # highlight farther right often marks the fully exposed span, not tangency.
    robust_scale = float(np.median(np.abs(change - np.median(change))) * 1.4826)
    threshold = max(1.2, 2.2 * robust_scale)
    candidates = np.flatnonzero((change >= threshold) &
                                (x_values >= coarse_x - 100) &
                                (x_values <= coarse_x + 25))
    if len(candidates):
        groups = np.split(candidates, np.flatnonzero(np.diff(candidates) > 1) + 1)
        sustained = [group for group in groups if len(group) >= 3]
        index = int((sustained[0] if sustained else groups[0])[0])
    else:
        search = np.flatnonzero((x_values >= coarse_x - 100) &
                                (x_values <= coarse_x + 25))
        if not len(search):
            return None
        index = int(search[np.argmax(change[search])])
    tangent_x = float(x_values[index])
    tangent_y = float(y0 + slope * (tangent_x - x0))
    signal_confidence = float(np.clip(change[index] / max(threshold, 1e-6), 0, 2) / 2)
    residual_confidence = math.exp(-line_residual / 14.0)
    return tangent_x, tangent_y, signal_confidence * residual_confidence, line_residual


def _full_wound_mask(regions: SemanticRegions, frame_shape) -> np.ndarray:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = regions.crop_xyxy
    low_mask = _largest_component(regions.mask('wound_rope_surface')).astype(np.uint8) * 255
    full = np.zeros((height, width), dtype=np.uint8)
    full[y1:y2, x1:x2] = cv2.resize(low_mask, (x2 - x1, y2 - y1),
                                    interpolation=cv2.INTER_NEAREST)
    return full


def _semantic_inner_edge(regions: SemanticRegions, source_x: float):
    """Fit the wound-surface upper envelope in semantic coordinates.

    The old colour-only fitter can lock onto a lower gold reflection band. The
    network was labelled specifically at the flange/wound interface, so this
    envelope is the primary reference. Original-resolution rope texture still
    supplies the pitch and therefore the physical scale.
    """
    probability = regions.probability('wound_rope_surface')
    mask = _largest_component(regions.mask('wound_rope_surface'))
    column_valid = mask.any(axis=0)
    xs = np.flatnonzero(column_valid).astype(np.float64)
    if len(xs) < 60:
        return None
    tops = mask[:, column_valid].argmax(axis=0).astype(np.float64)
    center = float(np.mean(xs)); scale = max(float(np.ptp(xs)), 1.0)
    normalized = (xs - center) / scale
    matrix = np.column_stack([normalized * normalized, normalized, np.ones(len(xs))])
    keep = np.ones(len(xs), dtype=bool)
    for _ in range(4):
        if keep.sum() < 40:
            return None
        coefficients = np.linalg.lstsq(matrix[keep], tops[keep], rcond=None)[0]
        residuals = np.abs(matrix @ coefficients - tops)
        tolerance = max(2.0, float(np.percentile(residuals[keep], 75)) * 2.2)
        keep = residuals <= tolerance
    low_x, _ = regions.point_from_source(source_x, 0)
    if low_x < xs[keep].min() - 12 or low_x > xs[keep].max() + 12:
        return None
    z = (low_x - center) / scale
    low_y = float(np.asarray([z * z, z, 1.0]) @ coefficients)
    _, source_y = regions.point_to_source(low_x, low_y)
    boundary_rows = np.clip(np.rint(tops[keep]).astype(int), 0, probability.shape[0] - 1)
    boundary_prob = probability[boundary_rows, xs[keep].astype(int)]
    residual = float(np.sqrt(np.mean((matrix[keep] @ coefficients - tops[keep]) ** 2)))
    confidence = float(np.clip(np.median(boundary_prob), 0, 1) *
                       math.exp(-residual / 4.0))
    return float(source_y), confidence, residual


def _estimate_texture_period(gray: np.ndarray, x: float, edge_y: float,
                             tangent_y: float, fallback: float):
    """Measure the repeat period of the visible rope texture over many columns.

    This period is deliberately kept separate from axial row pitch. The helical
    strand texture repeats at a slightly different interval from adjacent wound
    rope centres; a small anchor-derived ratio converts between them.
    """
    minimum = max(8, int(round(.45 * fallback)))
    maximum = max(minimum + 6, int(round(1.35 * fallback)))
    y1 = max(0, round(edge_y + 1.5 * fallback))
    y2 = min(gray.shape[0], round(tangent_y + 14 * fallback))
    curves = []
    for dx in (-240, -180, -120, -60, 0, 60, 120, 180, 240):
        column = max(4, min(gray.shape[1] - 5, round(x + dx)))
        signal = gray[y1:y2, column - 3:column + 4].mean(axis=1).astype(np.float64)
        if len(signal) < 5 * maximum or np.std(signal) < 3:
            continue
        trend = cv2.GaussianBlur(signal.reshape(-1, 1), (1, 0),
                                 sigmaX=0, sigmaY=35)[:, 0]
        signal -= trend; signal -= signal.mean()
        curve = []
        for lag in range(minimum, maximum + 1):
            left, right = signal[:-lag], signal[lag:]
            curve.append(float(np.dot(left, right) /
                               max(np.linalg.norm(left) * np.linalg.norm(right), 1e-9)))
        curves.append(curve)
    if len(curves) < 4:
        return fallback, 0.0
    curve = np.median(np.asarray(curves), axis=0)
    peaks = [index for index in range(1, len(curve) - 1)
             if curve[index] > curve[index - 1] and curve[index] >= curve[index + 1]]
    if not peaks:
        return fallback, 0.0
    index = max(peaks, key=lambda candidate: curve[candidate])
    denominator = curve[index - 1] - 2 * curve[index] + curve[index + 1]
    delta = (.5 * (curve[index - 1] - curve[index + 1]) / denominator
             if abs(denominator) > 1e-9 else 0.0)
    period = minimum + index + float(np.clip(delta, -.5, .5))
    return float(period), float(max(0.0, min(1.0, curve[index])))


def measure_hybrid_geometry(frame_bgr: np.ndarray, regions: SemanticRegions,
                            fallback_pitch_px: float = 20.08,
                            texture_to_row_pitch_ratio: float = 0.887,
                            maximum_texture_period_deviation: float = 0.03,
                            tangent_x_range: tuple[float, float] | None = None
                            ) -> HybridMeasurement:
    line = _fit_source_line(regions)
    if line is None:
        return HybridMeasurement(False, reason='semantic_incoming_invalid')
    tangent = _refine_tangent(frame_bgr, line)
    if tangent is None:
        return HybridMeasurement(False, reason='original_ridge_endpoint_invalid')
    tangent_x, tangent_y, tangent_confidence, _ = tangent
    if tangent_x_range is not None:
        lower_x, upper_x = tangent_x_range
        constrained_x = float(np.clip(tangent_x, lower_x, upper_x))
        if constrained_x != tangent_x:
            vx, vy, x0, y0, _coarse_x, _residual = line
            tangent_x = constrained_x
            tangent_y = float(y0 + (vy / max(vx, 1e-6)) * (tangent_x - x0))
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edge_fit = _semantic_inner_edge(regions, tangent_x)
    if edge_fit is None:
        return HybridMeasurement(False, tangent_x=tangent_x, tangent_y=tangent_y,
                                 tangent_confidence=tangent_confidence,
                                 reason='semantic_inner_edge_unreliable')
    edge_y, edge_confidence, _ = edge_fit
    from geometry_measurement import local_pitch
    row_pitch, row_pitch_confidence = local_pitch(
        gray, tangent_x, edge_y + fallback_pitch_px,
        fallback_pitch_px * texture_to_row_pitch_ratio)
    row_pitch_count = (max(0.0, (tangent_y - edge_y) / row_pitch)
                       if row_pitch_confidence >= .35 else math.nan)
    texture_period, pitch_confidence = _estimate_texture_period(
        gray, tangent_x, edge_y, tangent_y, fallback_pitch_px)
    # A wrong autocorrelation harmonic produces a large count jump even when
    # all geometry is stable. The fixed camera/rope calibration supplies a
    # narrow physical plausibility band; independent thin/thick-layer data is
    # required before widening or adapting this band.
    lower = fallback_pitch_px * (1.0 - maximum_texture_period_deviation)
    upper = fallback_pitch_px * (1.0 + maximum_texture_period_deviation)
    texture_period = float(np.clip(texture_period, lower, upper))
    measured_pitch = texture_period * texture_to_row_pitch_ratio
    if pitch_confidence < .30:
        return HybridMeasurement(False, tangent_x=tangent_x, tangent_y=tangent_y,
                                 edge_x=tangent_x, edge_y=edge_y,
                                 tangent_confidence=tangent_confidence,
                                 edge_confidence=edge_confidence,
                                 texture_period_px=texture_period,
                                 pitch_px=measured_pitch,
                                 pitch_confidence=pitch_confidence,
                                 reason='local_pitch_unreliable')
    distance = tangent_y - edge_y
    if distance < -measured_pitch:
        return HybridMeasurement(False, tangent_x=tangent_x, tangent_y=tangent_y,
                                 edge_x=tangent_x, edge_y=edge_y,
                                 reason='inner_edge_below_tangent')
    angle = math.degrees(math.atan2(line[1], line[0]))
    return HybridMeasurement(
        True, tangent_x=tangent_x, tangent_y=tangent_y,
        edge_x=tangent_x, edge_y=edge_y, texture_period_px=texture_period,
        pitch_px=measured_pitch,
        remaining_rope_count=max(0.0, distance / measured_pitch),
        line_angle_deg=angle, tangent_confidence=tangent_confidence,
        edge_confidence=edge_confidence, pitch_confidence=pitch_confidence,
        row_pitch_candidate_px=row_pitch,
        row_pitch_candidate_confidence=row_pitch_confidence,
        row_pitch_count_candidate=row_pitch_count,
        reason='ok',
    )


def _near_edge_local_boundary_support(
        frame_bgr: np.ndarray, regions: SemanticRegions, source_x: float,
        edge_y: float, pitch: float) -> bool:
    """Confirm a sustained flange-to-rope transition in seven nearby columns.

    This detector is deliberately specific to the current yellow-flange scene.
    A new flange appearance must fail closed rather than inherit a colour
    calibration.  Semantic curvature only predicts relative column heights;
    the transitions themselves are measured in the original image.
    """
    reference = _semantic_inner_edge(regions, source_x)
    if reference is None:
        return False
    height, width = frame_bgr.shape[:2]
    half_band = max(3, round(.5 * pitch))
    offsets = []
    for dx in (-3, -2, -1, 0, 1, 2, 3):
        x = round(source_x + dx * pitch)
        curve = _semantic_inner_edge(regions, x)
        if curve is None or x < 3 or x >= width - 3:
            continue
        expected_y = edge_y + curve[0] - reference[0]
        start = round(expected_y - 1.5 * pitch)
        stop = round(expected_y + 1.5 * pitch)
        if start - half_band < 0 or stop + half_band >= height:
            continue
        patch = frame_bgr[start-half_band:stop+half_band+1, x-3:x+4]
        saturation = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 1]
        signal = saturation.mean(axis=1)
        cumulative = np.r_[0., np.cumsum(signal)]
        starts = np.arange(stop - start + 1)
        before = (cumulative[starts+half_band] - cumulative[starts]) / half_band
        after = (cumulative[starts+2*half_band] -
                 cumulative[starts+half_band]) / half_band
        change = before - after
        index = int(np.argmax(change))
        if change[index] >= 25 and before[index] >= 65:
            offsets.append((start + index - expected_y) / pitch)
    return (len(offsets) >= 5 and
            abs(float(np.median(offsets))) <= .75 and
            float(np.percentile(np.abs(offsets), 80)) <= 1.0)


def measure_guarded_near_edge_geometry(
        frame_bgr: np.ndarray, regions: SemanticRegions, legacy_result,
        failed_measurement: HybridMeasurement, *, analysis_scale: float = .5
        ) -> HybridMeasurement | None:
    """Recover a close-range endpoint failure from independent current geometry.

    A semantic free-span endpoint is sometimes hundreds of pixels away from
    the contact.  Extending its short ridge search or clamping a negative gap
    produces unjustified counts.  Instead this branch accepts an already
    measured original-image contact only when its free span agrees with the
    semantic line and its inner edge agrees with the semantic surface at the
    *new* contact column.  It is limited to five pitches and does not accept
    predicted points, absent ropes, invalid original geometry or conflicts.
    """
    if (failed_measurement.valid or failed_measurement.reason not in (
            'inner_edge_below_tangent', 'local_pitch_unreliable',
            'original_tangent_signal_unreliable') or legacy_result is None
            or not 0 < analysis_scale <= 1):
        return None
    mask_pixels = int(regions.mask('incoming_rope_free_span').sum())
    if mask_pixels < 800:
        return None
    line = legacy_result.line
    tangent = legacy_result.tangent
    if (not getattr(legacy_result, 'geometry_localized', False)
            or not line.valid or line.confidence < .70
            or not tangent.valid or tangent.provenance != 'measured'
            or tangent.confidence < .35 or tangent.pitch_confidence < .35
            or not math.isfinite(legacy_result.distance_rope)
            or not 0 <= legacy_result.distance_rope <= 5.0):
        return None
    pitch = tangent.local_pitch_px / analysis_scale
    source_x = tangent.tangent_x / analysis_scale
    source_y = tangent.tangent_y / analysis_scale
    edge_x = tangent.edge_x / analysis_scale
    edge_y = tangent.edge_y / analysis_scale
    if (not all(math.isfinite(value) for value in
                (pitch, source_x, source_y, edge_x, edge_y)) or pitch <= 0
            or source_y < edge_y):
        return None
    semantic_line = _fit_source_line(regions)
    if semantic_line is None:
        return None
    vx, vy, x0, y0, _coarse_x, _residual = semantic_line
    semantic_angle = math.degrees(math.atan2(vy, vx))
    if abs(semantic_angle - line.angle_deg) > 3.0:
        return None
    # Compare axes over the visible free span, not the low-confidence mask tip.
    probe_x = max(source_x + 12 * pitch, frame_bgr.shape[1] * .65)
    probe_x = min(probe_x, frame_bgr.shape[1] * .90)
    original_slope = (line.y2 - line.y1) / max(line.x2 - line.x1, 1e-6)
    original_y = (line.y1 / analysis_scale + original_slope *
                  (probe_x - line.x1 / analysis_scale))
    semantic_y = y0 + vy / max(vx, 1e-6) * (probe_x - x0)
    if abs(original_y - semantic_y) > pitch:
        return None
    edge_fit = _semantic_inner_edge(regions, source_x)
    if edge_fit is None or abs(edge_fit[0] - edge_y) > 1.25 * pitch:
        return None
    if not _near_edge_local_boundary_support(
            frame_bgr, regions, source_x, edge_y, pitch):
        return None
    # Original-image support is separate from the low-resolution masks.  The
    # import is local because the general weak-rope helper imports this module.
    from weak_rope_fallback import free_span_ridge_support
    ridge_score, bright_positions = free_span_ridge_support(
        frame_bgr, line, source_x, analysis_scale)
    if ridge_score < 60 or bright_positions < 9:
        return None
    return HybridMeasurement(
        True, tangent_x=source_x, tangent_y=source_y,
        edge_x=edge_x, edge_y=edge_y, pitch_px=pitch,
        remaining_rope_count=float(legacy_result.distance_rope),
        line_angle_deg=float(line.angle_deg),
        tangent_confidence=float(tangent.confidence),
        edge_confidence=float(min(tangent.confidence, .50)),
        pitch_confidence=float(tangent.pitch_confidence),
        row_pitch_candidate_px=pitch,
        row_pitch_candidate_confidence=float(tangent.pitch_confidence),
        row_pitch_count_candidate=float(legacy_result.distance_rope),
        source_line_end_x=line.x2 / analysis_scale,
        source_line_end_y=line.y2 / analysis_scale,
        reason='guarded_near_edge_original_geometry')
