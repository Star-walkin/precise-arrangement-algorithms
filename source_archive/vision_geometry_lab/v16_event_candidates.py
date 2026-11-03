"""Offline, past-only geometric event candidates; never equipment commands.

``gap_proxy`` is optical-rope-width-normalized boundary/centre separation,
NOT an axial rope count. Contact and a trajectory reversal are only proxies
for last-turn and new-layer onset. No human action, annotation or video ID is
accepted by the tracker. Ground truth belongs to the separate scoring call.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import copy
import math

from vision_prototype.centerline_event_fusion import detect_gap_reversal, robust_slope
from vision_prototype.visual_turn_phase import VisualTurnPhaseIntegrator


PROTOCOL = {
    "schema": "v16_causal_event_candidates_1",
    "input_units": "optical_width_normalized_gap_proxy_not_axial_rope_count",
    "history_s": 14.0,
    "maximum_missing_s": .65,
    "approach_window_s": 1.4,
    "approach_minimum_points": 3,
    "approach_minimum_span_s": .6,
    "approach_minimum_drop_widths": .25,
    "contact_maximum_gap_widths": .25,
    "contact_consecutive_frames": 3,
    "reversal_maximum_gap_widths": 2.2,
    "reversal_window_s": 1.4,
    "reversal_minimum_slope_widths_s": .05,
    "departure_minimum_points": 3,
    "departure_minimum_span_s": .6,
    "rearm_clear_gap_widths": 3.,
    "maximum_event_duration_s": 12.,
    "event_matching_tolerance_s": .2,
    "deployment_permitted": False,
    "stationary_policy": "explicit_scene_evidence_only_not_constant_gap",
    "period_policy": "independent_full_turn_confirmed_unambiguous_only",
}


def _finite(value, default=math.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


@dataclass(frozen=True)
class EventCandidate:
    event: str
    cycle_id: int
    event_time_s: float
    first_available_at_s: float
    source_frame: int
    confirmation_frame: int
    confirmation_delay_s: float
    source: str
    evidence: dict
    candidate_only: bool = True
    physical_layer_confirmed: bool = False
    mechanical_turn_confirmed: bool = False
    device_advisory: bool = False


class EventCandidateTracker:
    """Bounded streaming state with immutable first reports per cycle.

    update(record, period=None) returns newly emitted candidate dictionaries.
    record: time_s, source_frame, usable, gap_proxy, processing_ms; optional
    stationary is *independent scene-motion evidence*, not a constant gap.

    Optional period: time_s, available_at_s, period_s, valid,
    full_turn_confirmed, ambiguous, source. The observation must be available
    at this frame and must explicitly establish a complete mechanical turn.
    An image texture periodicity without that evidence cannot enable t2.
    """

    def __init__(self):
        self.history = deque(maxlen=512)
        self.contact = []
        self.candidates = []
        self.resets = []
        self.last_time_s = None
        self.last_usable_s = None
        self.approach_armed = False
        self.arm_time_s = None
        self.cycle_id = 0
        self.emitted = set()
        self.clear_seen_s = None
        self.cycle_expired = False
        self.phase = VisualTurnPhaseIntegrator(maximum_gap_s=.65)
        self.phase_anchored = False
        self.period_rejection_reason = "independent_full_turn_period_unavailable"
        self.rows_received = 0
        self.rows_usable = 0
        self.latest_reason = "waiting_for_approach"

    def _reset_continuity(self, reason, now):
        # Never erase past candidates or re-arm the same cycle after missing
        # evidence. A new cycle requires visible clearance and new approach.
        self.history.clear()
        self.contact.clear()
        self.approach_armed = False
        self.arm_time_s = None
        self.last_usable_s = None
        self.clear_seen_s = None
        self.phase.reset()
        self.phase_anchored = False
        self.period_rejection_reason = reason
        self.latest_reason = reason
        self.resets.append({"time_s": now if math.isfinite(now) else None,
                            "reason": reason})

    def _approach_ready(self, now):
        rows = [r for r in self.history if r["usable"] and
                now - r["time_s"] <= PROTOCOL["approach_window_s"] + 1e-9]
        if self.clear_seen_s is not None:
            rows = [r for r in rows if r["time_s"] >= self.clear_seen_s]
        return bool(len(rows) >= 3 and
                    rows[-1]["time_s"] - rows[0]["time_s"] >= .6 - 1e-9 and
                    rows[0]["gap_proxy"] - rows[-1]["gap_proxy"] > .25 and
                    robust_slope([(r["time_s"], r["gap_proxy"]) for r in rows]) < -.05)

    def _period_update(self, now, period, stationary):
        source = str((period or {}).get("source", "unavailable"))
        ptime = _finite((period or {}).get("time_s"))
        available = _finite((period or {}).get("available_at_s"))
        value = _finite((period or {}).get("period_s"))
        valid = bool(period and period.get("valid") is True and
                     period.get("full_turn_confirmed") is True and
                     period.get("ambiguous") is False and value > 0 and
                     math.isfinite(value) and abs(ptime - now) <= 1e-8 and
                     math.isfinite(available) and ptime <= available <= now + 1e-8)
        self.period_rejection_reason = ("ok" if valid else
            "independent_full_turn_period_unavailable_or_ambiguous")
        decision = self.phase.update(now, value, valid=valid,
                                     stationary=stationary, source=source)
        if self.phase_anchored and decision.status == "ABORTED":
            self.period_rejection_reason = decision.reason
        return decision

    def _emit(self, event, event_time, source_frame, row, source, evidence):
        if event in self.emitted:
            return None
        self.emitted.add(event)
        item = asdict(EventCandidate(
            event=event, cycle_id=self.cycle_id, event_time_s=float(event_time),
            first_available_at_s=row["time_s"], source_frame=int(source_frame),
            confirmation_frame=row["source_frame"],
            confirmation_delay_s=row["time_s"] - event_time,
            source=source, evidence=copy.deepcopy(evidence)))
        self.candidates.append(item)
        return copy.deepcopy(item)

    def update(self, record, period=None):
        self.rows_received += 1
        now = _finite(record.get("time_s"))
        if not math.isfinite(now):
            self._reset_continuity("invalid_timestamp", now)
            self.last_time_s = None
            return []
        if self.last_time_s is not None and now <= self.last_time_s:
            self._reset_continuity("non_increasing_timestamp", now)
            self.last_time_s = now
            return []
        if self.last_time_s is not None and now - self.last_time_s > .65 + 1e-9:
            self._reset_continuity("frame_gap_exceeded", now)
        self.last_time_s = now
        stationary = bool(record.get("stationary", False))
        if stationary:
            self._reset_continuity("scene_stationary", now)
            return []
        usable = bool(record.get("usable", False))
        gap = _finite(record.get("gap_proxy"))
        usable = usable and math.isfinite(gap)
        row = {"time_s": now, "source_frame": int(record.get("source_frame", -1)),
               "usable": usable, "gap_proxy": gap,
               "processing_ms": _finite(record.get("processing_ms"), 0.)}
        if self.last_usable_s is not None and now - self.last_usable_s > .65 + 1e-9:
            self._reset_continuity("usable_geometry_gap_exceeded", now)
        self.history.append(row)
        while self.history and now - self.history[0]["time_s"] > 14.:
            self.history.popleft()
        phase_decision = self._period_update(now, period, False)
        new = []
        if usable:
            self.rows_usable += 1
            self.last_usable_s = now
            if (self.emitted or self.cycle_expired) and gap > 3. and self.clear_seen_s is None:
                self.clear_seen_s = now
            if self._approach_ready(now):
                if self.clear_seen_s is not None:
                    # This clear->approach motion starts a distinct cycle;
                    # its first event cannot steal the previous cycle's valley.
                    self.cycle_id += 1
                    self.emitted.clear()
                    self.cycle_expired = False
                    self.history = deque([r for r in self.history
                        if r["time_s"] >= self.clear_seen_s], maxlen=512)
                    self.contact.clear()
                    self.phase.reset()
                    self.phase_anchored = False
                    self._period_update(now, period, False)
                    self.clear_seen_s = None
                    self.arm_time_s = now
                if not self.approach_armed and not self.cycle_expired:
                    self.approach_armed = True
                    self.arm_time_s = now
            if self.arm_time_s is not None and now - self.arm_time_s > 12.:
                self.cycle_expired = True
                self.approach_armed = False
                self.contact.clear()
                self.latest_reason = "cycle_expired_waiting_for_clearance"
            if self.approach_armed and not self.cycle_expired:
                if gap <= .25 and "t0" not in self.emitted:
                    self.contact.append(row)
                    if len(self.contact) >= 3:
                        first = self.contact[0]
                        emitted = self._emit("t0", first["time_s"], first["source_frame"], row,
                            "sustained_contact_proxy_not_last_turn_proof",
                            {"gap_proxy": first["gap_proxy"], "consecutive_frames": len(self.contact),
                             "unit": PROTOCOL["input_units"]})
                        if emitted:
                            new.append(emitted)
                else:
                    self.contact.clear()
                if "t1" not in self.emitted:
                    prefix = [{"time_s": r["time_s"], "fused_valid": r["usable"],
                               "fused_count": r["gap_proxy"]} for r in self.history]
                    reversal = detect_gap_reversal(prefix)
                    if reversal.get("valid"):
                        event_time = reversal["new_layer_start_candidate_s"]
                        after = [r for r in self.history if r["usable"] and
                                 event_time + .2 - 1e-9 <= r["time_s"] <= event_time + 1.4 + 1e-9]
                        if (len(after) >= 3 and
                                after[-1]["time_s"] - after[0]["time_s"] >= .6 - 1e-9 and
                                robust_slope([(r["time_s"], r["gap_proxy"]) for r in after]) >= .05):
                            event_row = next(r for r in self.history if r["time_s"] == event_time)
                            emitted = self._emit("t1", event_time, event_row["source_frame"], row,
                                "gap_valley_and_sustained_departure_not_radial_layer_proof",
                                {"minimum_gap_proxy": reversal["minimum_gap_count"],
                                 "approach_slope": reversal["approach_slope_count_per_s"],
                                 "departure_slope": reversal["departure_slope_count_per_s"],
                                 "departure_span_s": after[-1]["time_s"] - after[0]["time_s"],
                                 "unit": PROTOCOL["input_units"]})
                            if emitted:
                                new.append(emitted)
                                self.phase_anchored = True
                                phase_decision = self.phase.anchor(event_time, available_at_s=now)
        else:
            # A missing frame cannot count towards consecutive contact.
            self.contact.clear()
            self.latest_reason = "current_geometry_unavailable"
        if (self.phase_anchored and not self.cycle_expired and
                phase_decision.status == "INTEGRATED_TURN_CANDIDATE"):
            emitted = self._emit("t2", phase_decision.integrated_turn_end_s, row["source_frame"], row,
                "independent_full_turn_integrated_after_t1_proxy",
                {"period_source": phase_decision.period_source,
                 "anchor_time_s": phase_decision.anchor_time_s,
                 "warning": "phase_is_measured_but_layer_anchor_is_only_a_candidate"})
            if emitted:
                new.append(emitted)
        if self.phase_anchored and phase_decision.status == "ABORTED":
            self.period_rejection_reason = phase_decision.reason
        if new:
            self.latest_reason = "candidate_emitted_not_physical_confirmation"
        return new

    def snapshot(self):
        return {"protocol": copy.deepcopy(PROTOCOL), "candidates": copy.deepcopy(self.candidates),
                "rows_received": self.rows_received, "rows_usable": self.rows_usable,
                "cycle_id": self.cycle_id, "latest_reason": self.latest_reason,
                "t2_reason": self.period_rejection_reason,
                "continuity_resets": copy.deepcopy(self.resets),
                "device_advisory": False, "deployment_permitted": False}

    def finalize(self):
        return self.snapshot()


def score_candidates(candidates, manual_truth, *, tolerance_s=.2):
    """Post-inference evaluation only; no closest/best candidate cherry-pick.

    Each source clip has one approximately labelled reversal. Report the
    *first* candidate for each node and count extras; do not match to whichever
    candidate happens to be closest to truth. Positive first-availability lag
    is confirmation delay, not future leakage and not an advance forecast.
    """
    names = {"t0": "previous_layer_last_revolution_start_s",
             "t1": "new_layer_start_s", "t2": "new_layer_first_revolution_end_s"}
    scored = {}
    for event, key in names.items():
        rows = [r for r in candidates if r["event"] == event]
        first = rows[0] if rows else None
        truth = _finite(manual_truth.get(key))
        item = {"detected": first is not None, "candidate_count": len(rows),
                "extra_candidate_count": max(0, len(rows) - 1),
                "truth_s": truth if math.isfinite(truth) else None,
                "missed": first is None, "within_research_tolerance": False}
        if first is not None and math.isfinite(truth):
            error = first["event_time_s"] - truth
            item.update(event_time_s=first["event_time_s"],
                        first_available_at_s=first["first_available_at_s"],
                        signed_error_s=error, absolute_error_s=abs(error),
                        confirmation_delay_s=first["first_available_at_s"] - first["event_time_s"],
                        first_availability_minus_truth_s=first["first_available_at_s"] - truth,
                        within_research_tolerance=abs(error) <= tolerance_s + 1e-9)
        scored[event] = item
    target = _finite(manual_truth.get(names["t2"]))
    for item in scored.values():
        item["lead_before_true_first_turn_end_s"] = (
            target - item["first_available_at_s"]
            if math.isfinite(target) and "first_available_at_s" in item else None)
    return {"events": scored, "research_tolerance_s": tolerance_s,
            "truth_status": "operator_approximate_development_video_not_blind_validation",
            "matching_policy": "first_emitted_not_closest_to_truth",
            "hardware_timing_tolerance_agreed": False, "deployment_permitted": False}
