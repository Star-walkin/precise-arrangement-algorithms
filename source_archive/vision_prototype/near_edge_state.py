"""Causal, read-only state inference for the last 1.5 rope pitches.

The exact geometric counter is deliberately kept separate from this state
tracker.  Near a flange, the incoming rope mask can merge with the wound
surface and make an exact tangent-to-edge distance unavailable even though a
recent, directly measured approach is still useful evidence that the rope is
at the edge.  This module may bridge that *state* for a bounded time; it never
fills in a missing exact count and it has no device-control dependencies.

Cold-start invalid images remain ``UNKNOWN``.  A future spatial contact
detector can supply an independent cold-start observation, but weakening the
exact geometry gates here would turn wound-rope texture into false active-rope
detections on the opposite camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from collections import deque

import numpy as np


class NearEdgeState(str, Enum):
    """Mutually exclusive states reported by :class:`NearEdgeStateTracker`."""

    UNKNOWN = "UNKNOWN"
    FAR = "FAR"
    NEAR_EDGE_MEASURED = "NEAR_EDGE_MEASURED"
    CONTACT_HELD = "CONTACT_HELD"
    CONTACT_TOPOLOGY_CANDIDATE = "CONTACT_TOPOLOGY_CANDIDATE"
    LEAVING_CANDIDATE = "LEAVING_CANDIDATE"
    LEAVING_CONFIRMED = "LEAVING_CONFIRMED"


CONTACT_COMPATIBLE_FAILURES = frozenset({
    "semantic_incoming_invalid",
    "semantic_inner_edge_unreliable",
    "inner_edge_below_tangent",
    "local_pitch_unreliable",
    "original_ridge_endpoint_invalid",
    "semantic_rope_presence_unconfirmed",
})


@dataclass(frozen=True)
class NearEdgeDecision:
    """One read-only near-edge decision.

    ``exact_count_valid`` is the only permission to consume
    ``exact_gap_roots`` as a measurement.  During ``CONTACT_HELD`` the latter
    is NaN and ``near_edge_interval_roots`` is the deliberately coarse
    ``(0, threshold)`` interval.  In particular, a held contact is never
    exported as a fabricated exact zero.
    """

    state: str = NearEdgeState.UNKNOWN.value
    exact_count_valid: bool = False
    exact_gap_roots: float = math.nan
    near_edge_interval_roots: tuple[float, float] | None = None
    confidence: float = 0.0
    reason: str = "no_near_edge_evidence"
    direction: str = "UNKNOWN"
    evidence_age_s: float = math.nan

    @property
    def near_edge(self) -> bool:
        return self.state in (
            NearEdgeState.NEAR_EDGE_MEASURED.value,
            NearEdgeState.CONTACT_HELD.value,
            NearEdgeState.CONTACT_TOPOLOGY_CANDIDATE.value,
        )


def topology_contact_confidence(shadow) -> float:
    """Return guarded single-frame contact-topology evidence.

    At the flange the active-rope semantic component can remain strong while
    the fitted wound-surface top edge falls below the rope endpoint.  That
    ordering inversion is useful as a *state candidate*, not as a distance.
    Empty/opposite-camera frames are rejected by the independent presence and
    mask-area checks.  The result is diagnostic-only and never validates an
    exact count.
    """
    if shadow is None or getattr(shadow, "geometry_conflict", False):
        return 0.0
    if getattr(shadow, "presence_evidence", "unconfirmed") not in (
            "supported", "supported_by_semantic_and_original_ridge"):
        return 0.0
    if int(getattr(shadow, "incoming_mask_pixels", 0)) < 800:
        return 0.0
    if getattr(shadow, "reason", "") not in (
            "inner_edge_below_tangent", "local_pitch_unreliable"):
        return 0.0
    tangent_y = getattr(shadow, "tangent_y", math.nan)
    edge_y = getattr(shadow, "edge_y", math.nan)
    if not (math.isfinite(tangent_y) and math.isfinite(edge_y)
            and edge_y > tangent_y):
        return 0.0
    mask_score = min(1.0, int(shadow.incoming_mask_pixels) / 2000.0)
    tangent_score = max(0.0, min(
        1.0, float(getattr(shadow, "tangent_confidence", 0.0))))
    return float(min(.80, .35 + .25 * mask_score + .20 * tangent_score))


class NearEdgeStateTracker:
    """Bridge contact-compatible geometry failures after direct observation.

    Parameters are expressed in rope pitches (``roots`` in the existing
    project terminology).  A directly measured gap at or below
    ``near_edge_threshold_roots`` arms the tracker regardless of whether a
    motion direction is available.  An increasing, directly measured gap
    beyond the threshold confirms departure from the edge.

    The tracker intentionally cannot arm from an invalid frame alone.  That
    fail-closed behavior prevents a static texture line or the opposite
    camera from being promoted to a contact event.
    """

    def __init__(
        self,
        near_edge_threshold_roots: float = 1.5,
        maximum_hold_s: float = 8.0,
        departure_rise_roots: float = 0.45,
        threshold_hysteresis_roots: float = 0.20,
        leaving_latch_s: float = 2.0,
        minimum_departure_samples: int = 4,
        minimum_departure_span_s: float = 0.55,
        minimum_departure_slope_roots_s: float = 0.12,
    ) -> None:
        if not math.isfinite(near_edge_threshold_roots) or near_edge_threshold_roots <= 0:
            raise ValueError("near_edge_threshold_roots must be positive")
        if not math.isfinite(maximum_hold_s) or maximum_hold_s <= 0:
            raise ValueError("maximum_hold_s must be positive")
        if not math.isfinite(departure_rise_roots) or departure_rise_roots <= 0:
            raise ValueError("departure_rise_roots must be positive")
        if threshold_hysteresis_roots < 0 or not math.isfinite(threshold_hysteresis_roots):
            raise ValueError("threshold_hysteresis_roots must be non-negative")
        if leaving_latch_s < 0 or not math.isfinite(leaving_latch_s):
            raise ValueError("leaving_latch_s must be non-negative")
        if minimum_departure_samples < 3:
            raise ValueError("minimum_departure_samples must be at least 3")
        if minimum_departure_span_s <= 0:
            raise ValueError("minimum_departure_span_s must be positive")
        if minimum_departure_slope_roots_s <= 0:
            raise ValueError("minimum_departure_slope_roots_s must be positive")
        self.near_edge_threshold_roots = float(near_edge_threshold_roots)
        self.maximum_hold_s = float(maximum_hold_s)
        self.departure_rise_roots = float(departure_rise_roots)
        self.threshold_hysteresis_roots = float(threshold_hysteresis_roots)
        self.leaving_latch_s = float(leaving_latch_s)
        self.minimum_departure_samples = int(minimum_departure_samples)
        self.minimum_departure_span_s = float(minimum_departure_span_s)
        self.minimum_departure_slope_roots_s = float(
            minimum_departure_slope_roots_s)
        self.reset()

    def reset(self) -> None:
        self.last_time_s: float | None = None
        self.armed = False
        self.last_direct_near_s: float | None = None
        self.last_direct_near_confidence = 0.0
        self.minimum_near_gap_roots = math.inf
        self.last_measured_gap_roots = math.nan
        self.leaving_confirmed_s: float | None = None
        # Departure needs a continuous measured branch.  A single false
        # geometry point after a long contact dropout must not create a layer
        # reversal; that exact failure occurs in the existing 111500 replay.
        self.departure_history: deque[tuple[float, float]] = deque(maxlen=32)
        self.decision = NearEdgeDecision()

    @staticmethod
    def _robust_slope(samples: list[tuple[float, float]]) -> float:
        slopes = [
            (right_value - left_value) / (right_time - left_time)
            for index, (left_time, left_value) in enumerate(samples[:-1])
            for right_time, right_value in samples[index + 1:]
            if right_time - left_time >= .15
        ]
        return float(np.median(np.asarray(slopes))) if slopes else math.nan

    @staticmethod
    def _confidence(value: float) -> float:
        if not math.isfinite(value):
            return 0.0
        return min(1.0, max(0.0, float(value)))

    @staticmethod
    def _direction(value: str | None) -> str:
        text = str(value or "UNKNOWN").strip().upper()
        return text or "UNKNOWN"

    @staticmethod
    def _failure_key(reason: str | None) -> str:
        text = str(reason or "").strip()
        # Some callers prepend a stage name (for example
        # ``hybrid:semantic_incoming_invalid``).  Preserve strict membership
        # while accepting that harmless diagnostic wrapper.
        return text.rsplit(":", 1)[-1]

    def update(
        self,
        time_s: float,
        gap_roots: float | None,
        measured: bool,
        *,
        confidence: float = 1.0,
        failure_reason: str = "",
        direction: str | None = None,
        spatial_contact_confidence: float = 0.0,
    ) -> NearEdgeDecision:
        """Consume one causal observation and return a state-only decision."""

        if not math.isfinite(time_s):
            self.reset()
            self.decision = NearEdgeDecision(reason="invalid_timestamp")
            return self.decision
        time_s = float(time_s)
        if self.last_time_s is not None and time_s < self.last_time_s:
            self.reset()
        self.last_time_s = time_s
        direction_text = self._direction(direction)

        exact_valid = bool(
            measured
            and gap_roots is not None
            and math.isfinite(float(gap_roots))
            and float(gap_roots) >= 0.0
        )
        if exact_valid:
            self.decision = self._update_measured(
                time_s, float(gap_roots), self._confidence(confidence),
                direction_text,
            )
        else:
            # Require consecutive re-acquired measurements for departure.
            # Contact-compatible invalid frames may hold the contact state,
            # but they interrupt a would-be leaving confirmation.
            self.departure_history.clear()
            invalid_reason = failure_reason or (
                "exact_gap_invalid" if measured else "exact_gap_unavailable"
            )
            self.decision = self._update_invalid(
                time_s, invalid_reason, direction_text,
                self._confidence(spatial_contact_confidence),
            )
        return self.decision

    def _update_measured(
        self,
        time_s: float,
        gap_roots: float,
        confidence: float,
        direction: str,
    ) -> NearEdgeDecision:
        threshold = self.near_edge_threshold_roots
        within_hysteresis = gap_roots <= threshold + self.threshold_hysteresis_roots

        if gap_roots <= threshold:
            self.armed = True
            self.last_direct_near_s = time_s
            self.last_direct_near_confidence = confidence
            self.minimum_near_gap_roots = min(
                self.minimum_near_gap_roots, gap_roots)
            self.last_measured_gap_roots = gap_roots
            self.leaving_confirmed_s = None
            self.departure_history.clear()
            return NearEdgeDecision(
                state=NearEdgeState.NEAR_EDGE_MEASURED.value,
                exact_count_valid=True,
                exact_gap_roots=gap_roots,
                near_edge_interval_roots=(0.0, threshold),
                confidence=confidence,
                reason="direct_near_edge_measurement",
                direction=direction,
                evidence_age_s=0.0,
            )

        if self.armed:
            rise = gap_roots - self.minimum_near_gap_roots
            if not within_hysteresis:
                self.departure_history.append((time_s, gap_roots))
            else:
                self.departure_history.clear()
            departure_samples = list(self.departure_history)
            departure_span = (
                departure_samples[-1][0] - departure_samples[0][0]
                if len(departure_samples) >= 2 else 0.0)
            departure_slope = self._robust_slope(departure_samples)
            if (rise >= self.departure_rise_roots and
                    len(departure_samples) >= self.minimum_departure_samples and
                    departure_span >= self.minimum_departure_span_s and
                    math.isfinite(departure_slope) and
                    departure_slope >= self.minimum_departure_slope_roots_s):
                self.armed = False
                self.leaving_confirmed_s = time_s
                self.last_measured_gap_roots = gap_roots
                self.departure_history.clear()
                return NearEdgeDecision(
                    state=NearEdgeState.LEAVING_CONFIRMED.value,
                    exact_count_valid=True,
                    exact_gap_roots=gap_roots,
                    confidence=confidence,
                    reason="direct_gap_increase_after_near_edge",
                    direction=direction,
                    evidence_age_s=0.0,
                )
            # A value just above 1.5 roots is not enough evidence to discard a
            # directly observed contact; keep hysteresis while retaining the
            # exact measurement as exact.
            if within_hysteresis:
                self.last_measured_gap_roots = gap_roots
                return NearEdgeDecision(
                    state=NearEdgeState.NEAR_EDGE_MEASURED.value,
                    exact_count_valid=True,
                    exact_gap_roots=gap_roots,
                    near_edge_interval_roots=(0.0, threshold + self.threshold_hysteresis_roots),
                    confidence=confidence,
                    reason="direct_near_edge_measurement_hysteresis",
                    direction=direction,
                    evidence_age_s=0.0,
                )
            # Keep the tracker armed while collecting a continuous rising
            # branch.  Export the current value as an exact measurement, but
            # do not call it a confirmed reversal yet.
            self.last_measured_gap_roots = gap_roots
            return NearEdgeDecision(
                state=NearEdgeState.LEAVING_CANDIDATE.value,
                exact_count_valid=True,
                exact_gap_roots=gap_roots,
                confidence=confidence,
                reason="collecting_sustained_departure_measurements",
                direction=direction,
                evidence_age_s=0.0,
            )

        if (self.leaving_confirmed_s is not None and
                time_s - self.leaving_confirmed_s <= self.leaving_latch_s):
            self.last_measured_gap_roots = gap_roots
            return NearEdgeDecision(
                state=NearEdgeState.LEAVING_CONFIRMED.value,
                exact_count_valid=True,
                exact_gap_roots=gap_roots,
                confidence=confidence,
                reason="leaving_confirmation_latched",
                direction=direction,
                evidence_age_s=time_s - self.leaving_confirmed_s,
            )

        self.armed = False
        self.last_direct_near_s = None
        self.last_direct_near_confidence = 0.0
        self.minimum_near_gap_roots = math.inf
        self.leaving_confirmed_s = None
        self.departure_history.clear()
        self.last_measured_gap_roots = gap_roots
        return NearEdgeDecision(
            state=NearEdgeState.FAR.value,
            exact_count_valid=True,
            exact_gap_roots=gap_roots,
            confidence=confidence,
            reason="direct_far_measurement",
            direction=direction,
            evidence_age_s=0.0,
        )

    def _update_invalid(
        self,
        time_s: float,
        failure_reason: str,
        direction: str,
        spatial_contact_confidence: float,
    ) -> NearEdgeDecision:
        if (self.leaving_confirmed_s is not None and
                time_s - self.leaving_confirmed_s <= self.leaving_latch_s):
            age = time_s - self.leaving_confirmed_s
            confidence = max(0.0, 1.0 - age / max(self.leaving_latch_s, 1e-9))
            return NearEdgeDecision(
                state=NearEdgeState.LEAVING_CONFIRMED.value,
                exact_count_valid=False,
                confidence=confidence,
                reason="leaving_confirmation_held",
                direction=direction,
                evidence_age_s=age,
            )

        if self.armed and self.last_direct_near_s is not None:
            age = max(0.0, time_s - self.last_direct_near_s)
            compatible = self._failure_key(failure_reason) in CONTACT_COMPATIBLE_FAILURES
            if compatible and age < self.maximum_hold_s:
                confidence = self.last_direct_near_confidence * (
                    1.0 - age / self.maximum_hold_s)
                return NearEdgeDecision(
                    state=NearEdgeState.CONTACT_HELD.value,
                    exact_count_valid=False,
                    exact_gap_roots=math.nan,
                    near_edge_interval_roots=(
                        0.0, self.near_edge_threshold_roots),
                    confidence=max(0.0, confidence),
                    reason=f"contact_compatible_dropout:{self._failure_key(failure_reason)}",
                    direction=direction,
                    evidence_age_s=age,
                )
            if age >= self.maximum_hold_s:
                self.armed = False
                self.last_direct_near_s = None
                self.last_direct_near_confidence = 0.0
                self.minimum_near_gap_roots = math.inf
                return NearEdgeDecision(
                    reason="near_edge_hold_expired",
                    direction=direction,
                    evidence_age_s=age,
                )

        if spatial_contact_confidence > 0.0:
            return NearEdgeDecision(
                state=NearEdgeState.CONTACT_TOPOLOGY_CANDIDATE.value,
                exact_count_valid=False,
                exact_gap_roots=math.nan,
                near_edge_interval_roots=None,
                confidence=spatial_contact_confidence,
                reason=("single_frame_contact_topology:"
                        f"{self._failure_key(failure_reason)}"),
                direction=direction,
                evidence_age_s=0.0,
            )

        return NearEdgeDecision(
            reason=f"unconfirmed_invalid:{self._failure_key(failure_reason)}",
            direction=direction,
        )


__all__ = [
    "CONTACT_COMPATIBLE_FAILURES",
    "NearEdgeDecision",
    "NearEdgeState",
    "NearEdgeStateTracker",
    "topology_contact_confidence",
]
