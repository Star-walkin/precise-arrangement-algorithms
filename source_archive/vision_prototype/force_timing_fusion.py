"""Causal fusion for the first-new-layer-revolution force time.

The module has no device interface.  A direct visual layer transition is
preferred.  Edge-arrival extrapolation is allowed only as an explicitly
shadow-only bridge when a measured occlusion covers the expected transition.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math

import numpy as np


@dataclass(frozen=True)
class OcclusionInterval:
    start_s: float
    confirmed_s: float
    end_s: float = math.nan
    peak_value: float = math.nan
    threshold: float = math.nan


@dataclass(frozen=True)
class ForceTimingEstimate:
    valid: bool
    reason: str
    source: str = "unavailable"
    new_layer_start_s: float = math.nan
    expected_force_s: float = math.nan
    force_window_start_s: float = math.nan
    force_window_end_s: float = math.nan
    available_at_s: float = math.nan
    period_s: float = math.nan
    confidence: float = 0.0
    shadow_only: bool = True
    device_advisory: bool = False


class CausalOcclusionTracker:
    """Incremental past-only wrapper around the batch interval definition."""

    def __init__(self, history_s: float = 4.0,
                 minimum_history: int = 15,
                 consecutive_on: int = 2,
                 consecutive_off: int = 3):
        self.history_s = float(history_s)
        self.minimum_history = int(minimum_history)
        self.consecutive_on = int(consecutive_on)
        self.consecutive_off = int(consecutive_off)
        self.samples: deque[tuple[float, float]] = deque(maxlen=512)
        self.last_time_s: float | None = None

    def reset(self) -> None:
        self.samples.clear()
        self.last_time_s = None

    def update(self, time_s: float, value: float | None
               ) -> list[OcclusionInterval]:
        if not math.isfinite(time_s):
            self.reset()
            return []
        if self.last_time_s is not None and time_s <= self.last_time_s:
            self.reset()
        self.last_time_s = float(time_s)
        if value is not None and math.isfinite(value):
            self.samples.append((float(time_s), float(value)))
        # Retain enough clean prehistory and the complete active event.
        while (self.samples and time_s - self.samples[0][0] >
               max(20.0, 3.0 * self.history_s)):
            self.samples.popleft()
        return detect_causal_occlusion_intervals(
            list(self.samples), self.history_s, self.minimum_history,
            self.consecutive_on, self.consecutive_off)


def detect_causal_occlusion_intervals(
        samples: list[tuple[float, float]],
        history_s: float = 4.0,
        minimum_history: int = 15,
        consecutive_on: int = 2,
        consecutive_off: int = 3) -> list[OcclusionInterval]:
    """Detect gross silhouette replacement using past samples only."""
    history: list[tuple[float, float]] = []
    intervals = []
    on_buffer: list[tuple[float, float, float]] = []
    active_start = active_confirmed = math.nan
    active_peak = -math.inf
    active_threshold = math.nan
    off_count = 0
    last_time = math.nan
    for time_s, value in samples:
        if not (math.isfinite(time_s) and math.isfinite(value)):
            continue
        if math.isfinite(last_time) and time_s <= last_time:
            history.clear(); on_buffer.clear()
            active_start = active_confirmed = math.nan
            active_peak = -math.inf; off_count = 0
        last_time = time_s
        history[:] = [item for item in history
                      if time_s - item[0] <= history_s]
        if math.isfinite(active_start):
            active_peak = max(active_peak, value)
            if value < active_threshold:
                off_count += 1
            else:
                off_count = 0
            if off_count >= consecutive_off:
                end_s = time_s
                intervals.append(OcclusionInterval(
                    active_start, active_confirmed, end_s, active_peak,
                    active_threshold))
                active_start = active_confirmed = math.nan
                active_peak = -math.inf; off_count = 0
                history.clear()
            continue
        if len(history) >= minimum_history:
            baseline = np.asarray([item[1] for item in history], np.float64)
            median = float(np.median(baseline))
            mad = float(np.median(np.abs(baseline - median)))
            threshold = median + max(60.0, 8.0 * mad)
            if value >= threshold:
                on_buffer.append((time_s, value, threshold))
                if len(on_buffer) >= consecutive_on:
                    active_start = on_buffer[0][0]
                    active_confirmed = time_s
                    active_peak = max(item[1] for item in on_buffer)
                    active_threshold = max(item[2] for item in on_buffer)
                    on_buffer.clear()
                continue
        on_buffer.clear()
        history.append((time_s, value))
    if math.isfinite(active_start):
        intervals.append(OcclusionInterval(
            active_start, active_confirmed, math.nan, active_peak,
            active_threshold))
    return intervals


def fuse_force_timing(
        direct_transition: dict | None,
        edge_arrival_s: float | None,
        period: dict | None,
        occlusions: list[OcclusionInterval],
        direct_window: tuple[float, float] = (-.15, .30),
        bridged_window: tuple[float, float] = (-.30, .35),
        minimum_period_confidence: float = .45) -> ForceTimingEstimate:
    """Fuse direct transition or a strictly gated causal occlusion bridge."""
    if direct_transition and direct_transition.get("valid"):
        start = float(direct_transition.get(
            "new_layer_start_candidate_s", math.nan))
        rotation = direct_transition.get("rotation") or {}
        period_s = float(rotation.get("period_s", math.nan))
        confidence = float(rotation.get("confidence", 0.0))
        available = float(direct_transition.get("confirmed_at_s", math.nan))
        if (math.isfinite(start) and math.isfinite(period_s) and period_s > 0
                and confidence >= minimum_period_confidence):
            expected = start + period_s
            if not math.isfinite(available) or available >= expected:
                return ForceTimingEstimate(False, 'direct_not_available_before_force')
            return ForceTimingEstimate(
                True, "direct_transition_plus_causal_full_period",
                "direct_centerline_gap_reversal", start, expected,
                expected + direct_window[0], expected + direct_window[1],
                available, period_s, confidence, True, False)
    if edge_arrival_s is None or period is None:
        return ForceTimingEstimate(False, "direct_and_bridge_inputs_unavailable")
    edge = float(edge_arrival_s)
    period_s = float(period.get("period_s", math.nan))
    confidence = float(period.get("confidence", 0.0))
    period_available = float(period.get("available_at_s", math.nan))
    if (not math.isfinite(edge) or not math.isfinite(period_s) or period_s <= 0
            or confidence < minimum_period_confidence):
        return ForceTimingEstimate(False, "edge_or_period_unreliable")
    predicted_start = edge + period_s
    covering = []
    for interval in occlusions:
        end = interval.end_s if math.isfinite(interval.end_s) else math.inf
        # The interval must be confirmed before the force time and must cover
        # the expected layer transition. A distant/late occlusion cannot arm
        # an extrapolated event.
        if (interval.start_s <= predicted_start <= end and
                interval.confirmed_s <= edge + 2.0 * period_s and
                .25 * period_s <= interval.start_s - edge <=
                1.25 * period_s):
            covering.append(interval)
    if not covering:
        return ForceTimingEstimate(
            False, "no_causal_occlusion_covering_expected_transition")
    interval = min(covering, key=lambda item: item.confirmed_s)
    available = max(period_available, interval.confirmed_s)
    expected = edge + 2.0 * period_s
    if not math.isfinite(available) or available >= expected:
        return ForceTimingEstimate(False, "bridge_not_available_before_force")
    # Confidence is deliberately discounted: the transition itself was not
    # visible, even though both its anchor and turn period were measured.
    bridge_confidence = confidence * .72
    return ForceTimingEstimate(
        True, "occlusion_gated_edge_plus_two_periods",
        "causal_occlusion_bridge", predicted_start, expected,
        expected + bridged_window[0], expected + bridged_window[1],
        available, period_s, bridge_confidence, True, False)
