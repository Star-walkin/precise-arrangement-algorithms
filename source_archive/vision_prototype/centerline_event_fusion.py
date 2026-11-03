"""Read-only fusion of multi-method gap trajectories into reversal timing."""
from __future__ import annotations

import math

import numpy as np


def annotate_causal_gap_direction(records: list[dict], observed_edge: str,
                                  window_s: float = 1.4,
                                  minimum_slope: float = .08,
                                  hold_s: float = 5.0) -> None:
    """Add approach/leave state and the operator's count convention.

    Approaching counts exclude the active rope. Departed counts include it.
    When motion becomes stationary, the last measured direction is retained as
    an explicitly held state; missing geometry never receives a new count.
    """
    history: list[tuple[float, float]] = []
    last_motion = "unknown"
    last_motion_time = math.nan
    for row in records:
        time_s = float(row["time_s"])
        valid = bool(row.get("fused_valid"))
        count = float(row.get("fused_count", math.nan))
        if valid and math.isfinite(count):
            history.append((time_s, count))
        history[:] = [item for item in history if time_s - item[0] <= window_s]
        slope = robust_slope(history)
        evidence = "insufficient_history"
        if math.isfinite(slope) and slope <= -minimum_slope:
            last_motion = "approaching"
            last_motion_time = time_s
            evidence = "measured_decreasing_gap"
        elif math.isfinite(slope) and slope >= minimum_slope:
            last_motion = "leaving"
            last_motion_time = time_s
            evidence = "measured_increasing_gap"
        elif (last_motion != "unknown" and math.isfinite(last_motion_time)
              and time_s - last_motion_time <= hold_s):
            evidence = "held_during_stationary_or_weak_trend"
        else:
            last_motion = "unknown"
        row["gap_slope_count_per_s"] = slope
        row["trend_motion"] = last_motion
        row["trend_target_edge"] = observed_edge
        row["trend_evidence"] = evidence
        row["count_mode"] = ("remaining" if last_motion == "approaching"
                             else "departed" if last_motion == "leaving"
                             else "unknown")
        row["displayed_count"] = (
            count if valid and last_motion == "approaching" else
            count + 1.0 if valid and last_motion == "leaving" else math.nan)


def robust_slope(samples: list[tuple[float, float]]) -> float:
    slopes = []
    for index, (ta, ya) in enumerate(samples[:-1]):
        for tb, yb in samples[index + 1:]:
            if tb - ta >= .08:
                slopes.append((yb - ya) / (tb - ta))
    return float(np.median(slopes)) if slopes else math.nan


def select_causal_full_period(records: list[dict], event_time_s: float,
                              history_s: float = 8.0) -> dict | None:
    """Select a past-only stable period; a 2:1 pair is NOT full-turn proof.

    This is still a periodic-appearance candidate, not a mechanically verified
    revolution. Reject octave conflicts rather than guessing the larger one.
    """
    candidates = []
    for row in records:
        time_s = float(row["time_s"])
        period = float(row.get("rotation_period_s", math.nan))
        confidence = float(row.get("rotation_confidence", 0.0))
        if (event_time_s - history_s <= time_s <= event_time_s
                and math.isfinite(period) and period > 0
                and confidence >= .45):
            candidates.append((time_s, period, confidence))
    if not candidates:
        return None
    last_time, last_period, last_confidence = candidates[-1]
    if event_time_s - last_time > 1.0:
        return None
    ambiguous = [item for item in candidates
                 if (1.82 <= max(item[1], last_period) / min(item[1], last_period) <= 2.18)
                 and item[2] >= .75 * last_confidence]
    if ambiguous:
        return None
    selected = candidates[-1]
    nearby = [item[1] for item in candidates
              if abs(item[1] - selected[1]) <= .08 * selected[1]]
    period = float(np.median(nearby)) if nearby else selected[1]
    return {"period_s": period, "confidence": selected[2],
            "last_observed_s": selected[0], "causal_samples": len(nearby)}


def detect_gap_reversal(records: list[dict], maximum_gap_count: float = 2.2,
                        window_s: float = 1.4,
                        minimum_slope: float = .05) -> dict:
    """Confirm a local gap minimum using independent past/future slopes.

    The returned event timestamp is the observed minimum, while
    ``confirmed_at_s`` is when enough departure evidence would have existed in
    a causal real-time implementation.  Long missing intervals are rejected.
    """
    usable = [row for row in records if bool(row.get("fused_valid"))
              and math.isfinite(float(row.get("fused_count", math.nan)))]
    candidates = []
    for row in usable:
        time_s = float(row["time_s"])
        count = float(row["fused_count"])
        if count > maximum_gap_count:
            continue
        before = [(float(item["time_s"]), float(item["fused_count"]))
                  for item in usable
                  if time_s - window_s <= float(item["time_s"])
                  <= time_s - .2]
        after = [(float(item["time_s"]), float(item["fused_count"]))
                 for item in usable
                 if time_s + .2 <= float(item["time_s"])
                 <= time_s + window_s]
        if len(before) < 3 or len(after) < 3:
            continue
        combined_times = np.asarray([item[0] for item in before + after])
        if (np.max(np.diff(np.sort(combined_times))) > .65
                or before[-1][0] < time_s - .65
                or after[0][0] > time_s + .65):
            continue
        slope_before = robust_slope(before)
        slope_after = robust_slope(after)
        if (not math.isfinite(slope_before) or not math.isfinite(slope_after)
                or slope_before > -minimum_slope
                or slope_after < minimum_slope):
            continue
        local = [float(item["fused_count"]) for item in usable
                 if abs(float(item["time_s"]) - time_s) <= .6]
        if count > min(local) + .12:
            continue
        score = ((-slope_before + slope_after) *
                 math.exp(-count / max(maximum_gap_count, 1e-6)))
        candidates.append((score, row, slope_before, slope_after, after[-1][0]))
    if not candidates:
        return {"valid": False, "reason": "no_sustained_gap_reversal"}
    score, row, slope_before, slope_after, confirmed_at = max(
        candidates, key=lambda item: (item[0], float(item[1]['time_s'])))
    event_time = float(row["time_s"])
    period = select_causal_full_period(records, event_time)
    result = {
        "valid": True,
        "reason": "multi_method_gap_minimum_with_sustained_departure",
        "new_layer_start_candidate_s": event_time,
        "confirmed_at_s": confirmed_at,
        "minimum_gap_count": float(row["fused_count"]),
        "approach_slope_count_per_s": slope_before,
        "departure_slope_count_per_s": slope_after,
        "score": score,
        "rotation": period,
        "force_advisory": False,
        "status": "offline_timing_candidate_not_actuation_permission",
    }
    if period:
        expected = event_time + period["period_s"]
        result["first_revolution_end_candidate_s"] = expected
        result["force_window_candidate_s"] = [expected - .15,
                                               expected + .30]
    return result
