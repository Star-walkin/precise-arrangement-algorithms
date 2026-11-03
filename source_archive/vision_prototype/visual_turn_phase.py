"""Causal, read-only turn-phase integration from visual period estimates.

This is an experimental alternative to adding one constant period to a layer
anchor. A visual texture period is not necessarily one mechanical revolution;
integration cannot resolve that ambiguity. Outputs never authorize equipment.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class VisualTurnPhaseDecision:
    status: str = 'WAITING_FOR_ANCHOR'
    reason: str = 'new_layer_anchor_unavailable'
    anchor_time_s: float = math.nan
    anchor_available_at_s: float = math.nan
    observed_until_s: float = math.nan
    available_at_s: float = math.nan
    phase_turns: float = math.nan
    predicted_turn_end_s: float = math.nan
    integrated_turn_end_s: float = math.nan
    integrated_turn_end_available_at_s: float = math.nan
    period_source: str = 'unavailable'
    mechanical_turn_confirmed: bool = False
    device_advisory: bool = False


@dataclass(frozen=True)
class _PeriodObservation:
    time_s: float
    period_s: float
    source: str
    valid: bool
    reason: str


class VisualTurnPhaseIntegrator:
    """Integrate reciprocal period, rejecting unknown temporal segments.

    Samples must arrive with strictly increasing acquisition timestamps. Both
    endpoints must have valid periods and be nonstationary. On each fully
    observed segment, reciprocal period is integrated with the trapezoid rule.
    A first-turn crossing is linearly interpolated *after* that segment arrives.

    ``anchor`` may refer to an earlier layer-start time, but only uses already
    received samples. The result records the later availability time separately
    from the inferred event time. Existing frozen decisions are never rewritten.
    An aborted cycle requires a new explicit anchor; invalid intervals are not
    bridged. Period-source changes also invalidate a segment.
    """

    def __init__(self, *, maximum_gap_s: float = .75,
                 maximum_period_jump_fraction: float = .18,
                 history_s: float = 60.):
        if maximum_gap_s <= 0 or not math.isfinite(maximum_gap_s):
            raise ValueError('maximum_gap_s must be positive and finite')
        if not 0 < maximum_period_jump_fraction < 1:
            raise ValueError('maximum_period_jump_fraction must be between 0 and 1')
        if not math.isfinite(history_s) or history_s <= maximum_gap_s:
            raise ValueError('history_s must exceed maximum_gap_s')
        self.maximum_gap_s = float(maximum_gap_s)
        self.maximum_period_jump_fraction = float(maximum_period_jump_fraction)
        self.history_s = float(history_s)
        self.reset()

    def reset(self) -> None:
        self.history: deque[_PeriodObservation] = deque()
        self.anchor_time_s = math.nan
        self.anchor_available_at_s = math.nan
        self.decision = VisualTurnPhaseDecision()

    def _abort(self, reason: str, available_at_s: float) -> VisualTurnPhaseDecision:
        self.decision = VisualTurnPhaseDecision(
            status='ABORTED', reason=reason,
            anchor_time_s=self.anchor_time_s,
            anchor_available_at_s=self.anchor_available_at_s,
            observed_until_s=(self.history[-1].time_s if self.history else math.nan),
            available_at_s=available_at_s)
        return self.decision

    def update(self, time_s: float, period_s: float | None, *, valid: bool = True,
               stationary: bool = False,
               source: str = 'markerless') -> VisualTurnPhaseDecision:
        time_s = float(time_s)
        if not math.isfinite(time_s):
            return self._abort('invalid_timestamp', math.nan)
        if self.history and time_s <= self.history[-1].time_s:
            self.history.clear()
            return self._abort('non_increasing_timestamp', time_s)
        period = float(period_s) if period_s is not None else math.nan
        period_valid = bool(valid and not stationary and
                            math.isfinite(period) and period > 0)
        reason = ('stationary' if stationary else
                  'ok' if period_valid else 'period_unavailable')
        self.history.append(_PeriodObservation(
            time_s, period, str(source), period_valid, reason))
        cutoff = time_s - self.history_s
        while len(self.history) > 1 and self.history[1].time_s < cutoff:
            self.history.popleft()
        if self.decision.status in ('ABORTED', 'INTEGRATED_TURN_CANDIDATE'):
            # The completed event is immutable. A later bad period cannot make
            # an earlier candidate disappear or change its availability time.
            return self.decision
        if not math.isfinite(self.anchor_time_s):
            self.decision = VisualTurnPhaseDecision(
                observed_until_s=time_s, available_at_s=time_s,
                period_source=str(source))
            return self.decision
        return self._integrate(time_s)

    def anchor(self, new_layer_start_s: float, *, available_at_s: float | None = None
               ) -> VisualTurnPhaseDecision:
        """Supply a causal layer-start estimate, including a delayed estimate.

        Availability is the timestamp of the latest already received frame.
        Allowing callers to backdate availability would turn retrospective
        recovery into a falsely early force prediction, so that is rejected.
        """
        self.anchor_time_s = float(new_layer_start_s)
        latest = self.history[-1].time_s if self.history else math.nan
        available = latest if available_at_s is None else float(available_at_s)
        self.anchor_available_at_s = available
        if not math.isfinite(self.anchor_time_s) or not math.isfinite(available):
            return self._abort('anchor_or_history_unavailable', available)
        if not self.history or abs(available - latest) > 1e-8:
            return self._abort('anchor_availability_must_equal_current_observation', available)
        if self.anchor_time_s > latest:
            return self._abort('anchor_is_in_the_future', available)
        if self.anchor_time_s < self.history[0].time_s:
            return self._abort('anchor_outside_retained_history', available)
        return self._integrate(available)

    def _segment_failure(self, left: _PeriodObservation,
                         right: _PeriodObservation) -> str | None:
        if not left.valid:
            return left.reason
        if not right.valid:
            return right.reason
        if right.time_s - left.time_s > self.maximum_gap_s + 1e-9:
            return 'observation_gap_exceeded'
        if abs(right.period_s / left.period_s - 1.) > (
                self.maximum_period_jump_fraction + 1e-9):
            return 'period_jump_or_harmonic_switch'
        if left.source != right.source:
            return 'period_source_changed'
        return None

    def _integrate(self, available_at_s: float) -> VisualTurnPhaseDecision:
        if self.anchor_time_s < self.history[0].time_s:
            return self._abort('anchor_outside_retained_history', available_at_s)
        latest = self.history[-1]
        # A new anchor placed exactly at the latest frame still requires a
        # valid, moving observation; otherwise there is no phase-zero evidence.
        if not latest.valid and self.anchor_time_s == latest.time_s:
            return self._abort(latest.reason, available_at_s)
        phase = 0.
        samples = list(self.history)
        sources = set()
        for left, right in zip(samples, samples[1:]):
            if right.time_s <= self.anchor_time_s:
                continue
            failure = self._segment_failure(left, right)
            if failure:
                return self._abort(failure, available_at_s)
            sources.add(left.source)
            start = max(self.anchor_time_s, left.time_s)
            span = right.time_s - left.time_s
            fraction = (start - left.time_s) / span
            frequency_left, frequency_right = 1. / left.period_s, 1. / right.period_s
            frequency_start = frequency_left + fraction * (
                frequency_right - frequency_left)
            increment = .5 * (frequency_start + frequency_right) * (right.time_s - start)
            if phase + increment >= 1. - 1e-12:
                crossing_fraction = max(0., min(1., (1. - phase) / increment))
                crossing = start + crossing_fraction * (right.time_s - start)
                self.decision = VisualTurnPhaseDecision(
                    status='INTEGRATED_TURN_CANDIDATE',
                    reason='one_visual_period_integrated_not_mechanical_confirmation',
                    anchor_time_s=self.anchor_time_s,
                    anchor_available_at_s=self.anchor_available_at_s,
                    observed_until_s=latest.time_s,
                    available_at_s=available_at_s,
                    phase_turns=1., integrated_turn_end_s=crossing,
                    integrated_turn_end_available_at_s=available_at_s,
                    period_source=left.source)
                return self.decision
            phase += increment
        self.decision = VisualTurnPhaseDecision(
            status='INTEGRATING', reason='awaiting_one_integrated_visual_period',
            anchor_time_s=self.anchor_time_s,
            anchor_available_at_s=self.anchor_available_at_s,
            observed_until_s=latest.time_s,
            available_at_s=available_at_s, phase_turns=phase,
            predicted_turn_end_s=latest.time_s + (1. - phase) * latest.period_s,
            period_source=latest.source)
        return self.decision
