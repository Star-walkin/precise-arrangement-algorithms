"""Causal, unconfirmed timing diagnostics for an occluded layer change.

This module does not detect a reversal, count rotations, or authorize a motor.
It extrapolates measured approach motion under explicit assumptions. A worker's
action, filename, video event label and future frame are not model inputs.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace
from statistics import median


ASSUMPTIONS = (
    "unconfirmed: approach speed remains approximately constant until the edge",
    "unconfirmed: reversal occurs immediately on reaching the edge",
    "unconfirmed: one new-layer revolution takes one projected rope pitch / approach speed",
    "diagnostic forecast only; neither a witnessed reversal nor a motor permission",
)


@dataclass(frozen=True)
class ForecastFitDiagnostics:
    """Last causal fit attempt; absent values are not zero measurements."""

    reason: str = "not_attempted"
    evaluated_at_s: float | None = None
    sample_count: int = 0
    span_s: float = 0.0
    slope_pair_count: int = 0
    speed_rope_s: float | None = None
    residual_rope: float | None = None
    latest_sample_error_rope: float | None = None
    speed_uncertainty_rope_s: float | None = None
    relative_speed_uncertainty: float | None = None


@dataclass(frozen=True)
class ForceTimingForecast:
    status: str = "unavailable"
    earliest_s: float | None = None
    latest_s: float | None = None
    target_s: float | None = None
    force_direction: str = ""
    reason: str = "insufficient_measured_approach"
    assumptions: tuple[str, ...] = ASSUMPTIONS
    measurement_age_s: float = 0.0
    approach_speed_rope_s: float | None = None
    fit_residual_rope: float | None = None
    unconfirmed: bool = True
    input_reason: str = "no_measurement_yet"
    fit_diagnostics: ForecastFitDiagnostics = field(default_factory=ForecastFitDiagnostics)


class CounterfactualForceForecaster:
    """Estimate a *conditional* first-wrap time from measured approach samples.

    Samples must already have passed the caller's image-measurement checks.
    ``measured=False`` includes held positions and all predictions: those can
    advance the clock, but never train or renew this forecast. Time is video
    time, not playback wall-clock time. Backwards seeks clear all history.

    An expired/cancelled window cannot be scheduled again in the same cycle.
    Rearming requires measured departure to >=3 pitches, followed by a fresh
    approach from >=3 pitches. This is deliberately more conservative than
    merely seeing direction jitter next to the edge.
    """

    def __init__(
        self,
        force_direction: str = "右",
        horizon_s: float = 6.0,
        history_s: float = 5.0,
        minimum_span_s: float = 1.0,
        minimum_samples: int = 6,
        maximum_sample_gap_s: float = 0.75,
        minimum_speed_rope_s: float = 0.1,
        maximum_speed_rope_s: float = 2.0,
        maximum_fit_residual_rope: float = 0.35,
        maximum_relative_speed_uncertainty: float = 0.35,
        trend_bin_s: float = 0.4,
    ) -> None:
        if horizon_s <= 0 or history_s < minimum_span_s or minimum_span_s <= 0:
            raise ValueError("positive horizon and history >= positive minimum span required")
        if minimum_samples < 3 or maximum_sample_gap_s <= 0:
            raise ValueError("at least three samples and a positive sample gap required")
        if not 0 < minimum_speed_rope_s < maximum_speed_rope_s:
            raise ValueError("invalid speed bounds")
        if maximum_fit_residual_rope <= 0:
            raise ValueError("positive residual limit required")
        if not 0 < maximum_relative_speed_uncertainty < 1 or trend_bin_s <= 0:
            raise ValueError("relative uncertainty must be between zero and one; positive trend bin required")
        self.force_direction = force_direction
        self.horizon_s = horizon_s
        self.history_s = history_s
        self.minimum_span_s = minimum_span_s
        self.minimum_samples = minimum_samples
        self.maximum_sample_gap_s = maximum_sample_gap_s
        self.minimum_speed_rope_s = minimum_speed_rope_s
        self.maximum_speed_rope_s = maximum_speed_rope_s
        self.maximum_fit_residual_rope = maximum_fit_residual_rope
        self.maximum_relative_speed_uncertainty = maximum_relative_speed_uncertainty
        self.trend_bin_s = trend_bin_s
        self.reset()

    def reset(self) -> None:
        self.samples: deque[tuple[float, float]] = deque()
        self.last_time_s: float | None = None
        self.anchor_time_s: float | None = None
        self.forecast = ForceTimingForecast(force_direction=self.force_direction)
        self.fit_diagnostics = ForecastFitDiagnostics()
        self.cycle_locked = False
        self.departure_seen = False
        self.window_started = False

    def _expire(self, reason: str) -> None:
        self.forecast = replace(self.forecast, status="expired", reason=reason)
        self.cycle_locked = True
        self.samples.clear()

    def _fit(self, time_s: float) -> tuple[float, float, float, float] | None:
        self.fit_diagnostics = ForecastFitDiagnostics(
            evaluated_at_s=time_s, sample_count=len(self.samples),
            span_s=self.samples[-1][0] - self.samples[0][0] if self.samples else 0.0,
        )
        if len(self.samples) < self.minimum_samples:
            self.fit_diagnostics = replace(self.fit_diagnostics, reason="insufficient_sample_count")
            return None
        raw_data = list(self.samples)
        span_s = raw_data[-1][0] - raw_data[0][0]
        if span_s < self.minimum_span_s:
            self.fit_diagnostics = replace(self.fit_diagnostics, reason="insufficient_time_span")
            return None
        data = raw_data
        if span_s >= max(2.0, 5 * self.trend_bin_s):
            # All entries here are already in the past. Aggregate local jitter
            # before estimating a multi-second trend; never aggregate future
            # frames or fill an empty/occluded bin with a predicted position.
            bins: dict[int, list[tuple[float, float]]] = {}
            for t, distance in raw_data:
                bins.setdefault(math.floor(t / self.trend_bin_s), []).append((t, distance))
            aggregated = [
                (median(t for t, _ in points), median(d for _, d in points))
                for points in bins.values() if len(points) >= 2
            ]
            if len(aggregated) >= 5:
                data = aggregated
        # Bound per-frame cost even if called at full camera frame rate.
        if len(data) > 40:
            data = [data[round(i * (len(data) - 1) / 39)] for i in range(40)]
        # Close frames mostly measure localization jitter divided by a tiny dt.
        # Restrict pairwise slopes to at least half the observed time span so
        # their median/MAD measures an actual approach trend, not frame noise.
        minimum_pair_dt = max(0.25, 0.5 * (data[-1][0] - data[0][0]))
        slopes = [
            (d2 - d1) / (t2 - t1)
            for i, (t1, d1) in enumerate(data)
            for t2, d2 in data[i + 1 :]
            if t2 - t1 >= minimum_pair_dt
        ]
        if not slopes:
            self.fit_diagnostics = replace(self.fit_diagnostics, reason="no_long_baseline_slopes")
            return None
        slope = median(slopes)
        speed = -slope
        self.fit_diagnostics = replace(self.fit_diagnostics, speed_rope_s=speed,
                                       slope_pair_count=len(slopes))
        if not self.minimum_speed_rope_s <= speed <= self.maximum_speed_rope_s:
            self.fit_diagnostics = replace(self.fit_diagnostics, reason="approach_speed_out_of_bounds")
            return None
        distance_now = median(d - slope * (t - time_s) for t, d in data)
        # Evaluate uncertainty and outliers against original measurements, not
        # the smoother bin medians: averaging must not hide image uncertainty.
        residuals = sorted(abs(d - (distance_now + slope * (t - time_s))) for t, d in raw_data)
        robust_residual = residuals[int(0.8 * (len(residuals) - 1))]
        latest_error = abs(raw_data[-1][1] - distance_now)
        self.fit_diagnostics = replace(self.fit_diagnostics, residual_rope=robust_residual,
                                       latest_sample_error_rope=latest_error)
        if robust_residual > self.maximum_fit_residual_rope:
            self.fit_diagnostics = replace(self.fit_diagnostics, reason="fit_residual_too_large")
            return None
        # A newly arrived bad localization must not refresh the expiry clock.
        if latest_error > max(0.5, 3 * robust_residual):
            self.fit_diagnostics = replace(self.fit_diagnostics, reason="latest_measurement_outlier")
            return None
        slope_mad = median(abs(value - slope) for value in slopes)
        speed_uncertainty = max(0.015, 0.05 * speed, 1.4826 * slope_mad)
        self.fit_diagnostics = replace(
            self.fit_diagnostics, speed_uncertainty_rope_s=speed_uncertainty,
            relative_speed_uncertainty=speed_uncertainty / speed,
        )
        if speed_uncertainty > self.maximum_relative_speed_uncertainty * speed:
            self.fit_diagnostics = replace(self.fit_diagnostics, reason="speed_uncertainty_too_large")
            return None
        if distance_now < 0:
            self.fit_diagnostics = replace(self.fit_diagnostics, reason="negative_fitted_distance")
            return None
        self.fit_diagnostics = replace(self.fit_diagnostics, reason="fit_accepted")
        return distance_now, speed, speed_uncertainty, robust_residual

    def update(
        self,
        time_s: float,
        distance_rope: float | None,
        direction: str,
        measured: bool,
    ) -> ForceTimingForecast:
        if not math.isfinite(time_s):
            self.reset()
            self.forecast = replace(self.forecast, reason="invalid_video_time",
                                    input_reason="invalid_video_time")
            return self.forecast
        if self.last_time_s is not None and time_s < self.last_time_s:
            self.reset()
        duplicate_time = self.last_time_s == time_s
        self.last_time_s = time_s
        valid = measured and distance_rope is not None and math.isfinite(distance_rope) and distance_rope >= 0
        input_reason = ("measurement_not_eligible" if not valid else
                        "duplicate_video_time" if duplicate_time else
                        "measured_approach" if direction == "APPROACHING" else
                        "measured_departure" if direction == "LEAVING" else
                        "direction_not_confirmed_approaching")

        if valid and direction == "LEAVING":
            if distance_rope >= 3.0:
                self.departure_seen = True
            if self.forecast.status in ("scheduled", "due"):
                self._expire("measured_departure_use_observed_state_machine")
            self.samples.clear()
        if self.cycle_locked and self.departure_seen and valid and direction == "APPROACHING" and distance_rope >= 3.0:
            self.reset()
            self.last_time_s = time_s

        if self.forecast.status in ("scheduled", "due"):
            if time_s - self.anchor_time_s > self.horizon_s:
                self._expire("measurement_horizon_expired")
            elif time_s > self.forecast.latest_s:
                self._expire("conditional_window_passed")

        if not self.cycle_locked and valid and direction == "APPROACHING" and not duplicate_time:
            if self.samples and time_s - self.samples[-1][0] > self.maximum_sample_gap_s:
                self.samples.clear()
            self.samples.append((time_s, float(distance_rope)))
            while self.samples and time_s - self.samples[0][0] > self.history_s:
                self.samples.popleft()
            fit = self._fit(time_s)
            if fit is not None and not self.window_started:
                distance, speed, speed_error, residual = fit
                # This half-pitch floor is a conservative engineering bound,
                # NOT a calibrated statistical confidence interval. It includes
                # an allowance for the one-pitch/one-wrap model mismatch.
                position_error = max(0.5, 3 * residual)
                earliest = time_s + max(0.0, distance + 1 - position_error) / (speed + speed_error)
                latest = time_s + (distance + 1 + position_error) / max(self.minimum_speed_rope_s, speed - speed_error)
                target = time_s + (distance + 1) / speed
                if latest - time_s <= self.horizon_s:
                    self.anchor_time_s = time_s
                    self.forecast = ForceTimingForecast(
                        status="scheduled", earliest_s=earliest, latest_s=latest,
                        target_s=target, force_direction=self.force_direction,
                        reason="conditional_constant_speed_edge_plus_one_pitch",
                        approach_speed_rope_s=speed, fit_residual_rope=residual,
                    )
                elif self.forecast.status == "unavailable":
                    self.forecast = replace(self.forecast, reason="target_beyond_bounded_horizon")
            elif fit is None and self.forecast.status == "unavailable":
                self.forecast = replace(self.forecast, reason=self.fit_diagnostics.reason)

        if self.forecast.status == "scheduled" and time_s >= self.forecast.earliest_s:
            self.window_started = True
            self.forecast = replace(self.forecast, status="due", reason="unconfirmed_conditional_window_only")
        if self.anchor_time_s is not None:
            self.forecast = replace(self.forecast, measurement_age_s=max(0.0, time_s - self.anchor_time_s))
        self.forecast = replace(self.forecast, input_reason=input_reason,
                                fit_diagnostics=self.fit_diagnostics)
        return self.forecast
