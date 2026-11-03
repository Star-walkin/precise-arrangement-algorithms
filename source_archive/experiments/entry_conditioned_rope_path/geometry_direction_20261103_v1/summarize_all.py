"""Summarize the frozen 292-image G0/G1 geometry run without reading images.

Only run this after all_001/run_receipt.json has been written. Prediction JSON
and the run receipt are read-only; summary.json is created exclusively once.
Lengths and direction bonuses describe algorithm output, not target identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_RUN = Path(r"C:\precise_arrangement_results\geometry_direction_20261103_v1\all_001")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "maximum": None}
    ordered = sorted(values)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    return {"count": len(values), "mean": statistics.fmean(values),
            "median": statistics.median(values), "p95": p95, "maximum": ordered[-1]}


def candidate_summary(candidate: dict | None) -> dict | None:
    if candidate is None:
        return None
    evidence = candidate.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []
    known = [item for item in evidence if item.get("background_direction", {}).get("known") is True]
    unknown = [item.get("background_direction", {}).get("reason", "unspecified")
               for item in evidence if item.get("background_direction", {}).get("known") is not True]
    bonuses = [float(item.get("direction_soft", 0.0)) for item in evidence]
    angles = [finite_number(item.get("background_direction", {}).get("relative_angle_deg")) for item in known]
    angles = [value for value in angles if value is not None]
    return {
        "candidate_id": candidate.get("id"),
        "termination": candidate.get("termination"),
        "point_count": len(candidate.get("points_xy", [])),
        "supported_after_entry_px": candidate.get("supported_after_entry_px"),
        "unsupported_after_entry_px": candidate.get("unsupported_after_entry_px"),
        "rank_score": candidate.get("rank_score"),
        "evidence_steps": len(evidence),
        "background_direction_known_steps": len(known),
        "background_direction_unknown_reasons": dict(Counter(unknown)),
        "direction_positive_bonus_steps": sum(value > 0 for value in bonuses),
        "direction_bonus_sum": sum(bonuses),
        "direction_bonus_max": max(bonuses, default=0.0),
        "known_relative_angle_deg": distribution(angles),
    }


def selected_candidate(prediction: dict) -> dict | None:
    selected = prediction.get("selected")
    if not isinstance(selected, dict):
        return None
    return selected


def after_entry(candidate: dict) -> list[list[float]]:
    points = candidate.get("points_xy", [])
    status = candidate.get("point_status", [])
    if len(points) != len(status):
        return []
    first = next((i for i, state in enumerate(status) if state not in ("input_entry", "entry_condition")), len(points))
    return points[max(first - 1, 0):] if first < len(points) else []


def path_length(points: list[list[float]]) -> float:
    return sum(math.dist(a, b) for a, b in zip(points[:-1], points[1:]))


def interpolate_path(points: list[list[float]], distance: float) -> tuple[float, float]:
    remaining = distance
    for a, b in zip(points[:-1], points[1:]):
        segment = math.dist(a, b)
        if remaining <= segment or b is points[-1]:
            fraction = 0.0 if segment == 0 else min(max(remaining / segment, 0.0), 1.0)
            return (a[0] + (b[0] - a[0]) * fraction, a[1] + (b[1] - a[1]) * fraction)
        remaining -= segment
    return tuple(points[-1])


def compare_paths(g0: dict, g1: dict) -> dict:
    a = selected_candidate(g0)
    b = selected_candidate(g1)
    if a is None or b is None:
        return {"both_selected": False, "exact_same_points_and_states": None,
                "common_arclength_px": None, "median_separation_px": None,
                "maximum_separation_px": None, "endpoint_separation_px": None}
    pa, pb = after_entry(a), after_entry(b)
    exact = a.get("points_xy") == b.get("points_xy") and a.get("point_status") == b.get("point_status")
    endpoint = math.dist(a["points_xy"][-1], b["points_xy"][-1]) if a.get("points_xy") and b.get("points_xy") else None
    common = min(path_length(pa), path_length(pb)) if len(pa) >= 2 and len(pb) >= 2 else 0.0
    if common > 0:
        distances = [i * min(16.0, common / max(1, math.ceil(common / 16.0)))
                     for i in range(math.ceil(common / 16.0) + 1)]
        separations = [math.dist(interpolate_path(pa, position), interpolate_path(pb, position))
                       for position in distances]
    else:
        separations = []
    return {"both_selected": True, "exact_same_points_and_states": exact,
            "common_arclength_px": common,
            "median_separation_px": statistics.median(separations) if separations else None,
            "maximum_separation_px": max(separations) if separations else None,
            "endpoint_separation_px": endpoint}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run.resolve()
    receipt_path = run / "run_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError("Wait until all_001/run_receipt.json exists")
    inputs_path = HERE / "inputs.json"
    inputs = load(inputs_path)
    receipt = load(receipt_path)
    if receipt.get("phase") != "all" or receipt.get("input_rows") != 292 or receipt.get("completed_rows") != 292:
        raise ValueError("The full 292-image run is not complete")
    input_rows = inputs["rows"]
    if len(input_rows) != 292 or len({row["source_id"] for row in input_rows}) != 292:
        raise ValueError("Frozen inputs are not 292 unique source images")
    receipt_by_id = {row["source_id"]: row for row in receipt["rows"]}
    if len(receipt_by_id) != 292 or set(receipt_by_id) != {row["source_id"] for row in input_rows}:
        raise ValueError("Run receipt does not match frozen source IDs")

    errors: list[dict] = []
    rows: list[dict] = []
    for item in input_rows:
        source_id = item["source_id"]
        run_row = receipt_by_id[source_id]
        predictions: dict[str, dict] = {}
        for arm in ("G0", "G1"):
            path = run / "predictions" / arm / (source_id + ".json")
            try:
                pred = load(path)
                if pred.get("source_id") != source_id or pred.get("arm") != arm:
                    raise ValueError("Prediction source/arm mismatch")
                predictions[arm] = pred
            except Exception as exc:
                errors.append({"source_id": source_id, "arm": arm, "error": f"{type(exc).__name__}: {exc}"})
        arm_rows: dict[str, dict] = {}
        for arm in ("G0", "G1"):
            pred = predictions.get(arm)
            if pred is None:
                arm_rows[arm] = {"status": "missing_prediction", "stop_reason": "missing_prediction",
                                 "selected": None, "best_candidate": None}
                continue
            chosen = selected_candidate(pred)
            candidates = pred.get("candidates", [])
            arm_rows[arm] = {
                "status": pred.get("status"),
                "stop_reason": pred.get("stop_reason"),
                "candidate_count": len(candidates),
                "selected": candidate_summary(chosen),
                "best_candidate": candidate_summary(candidates[0] if candidates else None),
                "elapsed_seconds": pred.get("elapsed_seconds"),
            }
            if run_row.get(f"status_{arm}") != pred.get("status") or run_row.get(f"stop_{arm}") not in (None, pred.get("stop_reason")):
                errors.append({"source_id": source_id, "arm": arm, "error": "run_receipt_prediction_mismatch"})
        g0, g1 = predictions.get("G0"), predictions.get("G1")
        if g0 is None or g1 is None:
            pair_category = "missing_prediction"
            path_difference = None
        else:
            has0, has1 = selected_candidate(g0) is not None, selected_candidate(g1) is not None
            pair_category = ("both_selected" if has0 and has1 else "G0_only" if has0 else "G1_only" if has1 else "neither_selected")
            path_difference = compare_paths(g0, g1)
        rows.append({
            "source_id": source_id,
            "source_group": item["source_group"],
            "source_use": item["source_use"],
            "run_failure": bool(run_row.get("failure")),
            "run_elapsed_seconds": run_row.get("elapsed_seconds"),
            "G0": arm_rows["G0"],
            "G1": arm_rows["G1"],
            "pair_category": pair_category,
            "status_changed": arm_rows["G0"]["status"] != arm_rows["G1"]["status"],
            "stop_changed": arm_rows["G0"]["stop_reason"] != arm_rows["G1"]["stop_reason"],
            "path_difference": path_difference,
        })

    summary: dict = {
        "schema_version": "geometry_direction_summary_v1",
        "run_directory": str(run),
        "run_receipt_sha256": sha(receipt_path),
        "inputs_sha256": sha(inputs_path),
        "source_count": len(rows),
        "run_exit_code": receipt.get("exit_code"),
        "run_failure_count": sum(row["run_failure"] for row in rows),
        "network_response_used": False,
        "status_counts": {arm: dict(Counter(row[arm]["status"] for row in rows)) for arm in ("G0", "G1")},
        "stop_reason_counts": {arm: dict(Counter(row[arm]["stop_reason"] for row in rows)) for arm in ("G0", "G1")},
        "pair_category_counts": dict(Counter(row["pair_category"] for row in rows)),
        "status_changed_count": sum(row["status_changed"] for row in rows),
        "stop_changed_count": sum(row["stop_changed"] for row in rows),
        "both_selected_path_changed_count": sum(
            row["path_difference"] is not None and row["path_difference"]["exact_same_points_and_states"] is False
            for row in rows
        ),
        "run_elapsed_seconds": distribution([value for row in rows if (value := finite_number(row["run_elapsed_seconds"])) is not None]),
        "arm_elapsed_seconds": {
            arm: distribution([value for row in rows if (value := finite_number(row[arm].get("elapsed_seconds"))) is not None])
            for arm in ("G0", "G1")
        },
        "selected_supported_after_entry_px": {
            arm: distribution([value for row in rows if row[arm]["selected"] is not None
                               if (value := finite_number(row[arm]["selected"].get("supported_after_entry_px"))) is not None])
            for arm in ("G0", "G1")
        },
        "selected_unsupported_after_entry_px": {
            arm: distribution([value for row in rows if row[arm]["selected"] is not None
                               if (value := finite_number(row[arm]["selected"].get("unsupported_after_entry_px"))) is not None])
            for arm in ("G0", "G1")
        },
        "G1_selected_direction": {
            "evidence_steps": sum(row["G1"]["selected"]["evidence_steps"] for row in rows if row["G1"]["selected"]),
            "known_steps": sum(row["G1"]["selected"]["background_direction_known_steps"] for row in rows if row["G1"]["selected"]),
            "positive_bonus_steps": sum(row["G1"]["selected"]["direction_positive_bonus_steps"] for row in rows if row["G1"]["selected"]),
            "bonus_sum": sum(row["G1"]["selected"]["direction_bonus_sum"] for row in rows if row["G1"]["selected"]),
            "bonus_max": max((row["G1"]["selected"]["direction_bonus_max"] for row in rows if row["G1"]["selected"]), default=0.0),
            "unknown_reason_counts": dict(sum((Counter(row["G1"]["selected"]["background_direction_unknown_reasons"])
                                               for row in rows if row["G1"]["selected"]), Counter())),
        },
        "path_endpoint_separation_px": distribution([
            value for row in rows if row["path_difference"] is not None
            if (value := finite_number(row["path_difference"].get("endpoint_separation_px"))) is not None
        ]),
        "source_use_counts": dict(Counter(row["source_use"] for row in rows)),
        "by_source_group": {},
        "errors": errors,
        "error_count": len(errors),
        "rows": rows,
        "interpretation_boundary": "Algorithm output statistics only; 292 gallery images are not independent identity ground truth, and a longer selected path is not proof of correct rope identity.",
    }
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["source_group"]].append(row)
    for group, grouped in sorted(by_group.items()):
        summary["by_source_group"][group] = {
            "source_count": len(grouped),
            "source_use_counts": dict(Counter(row["source_use"] for row in grouped)),
            "status_counts": {arm: dict(Counter(row[arm]["status"] for row in grouped)) for arm in ("G0", "G1")},
            "stop_reason_counts": {arm: dict(Counter(row[arm]["stop_reason"] for row in grouped)) for arm in ("G0", "G1")},
            "pair_category_counts": dict(Counter(row["pair_category"] for row in grouped)),
            "run_failure_count": sum(row["run_failure"] for row in grouped),
        }

    out = run / "summary.json"
    with out.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(out), "source_count": len(rows), "error_count": len(errors),
                      "status_counts": summary["status_counts"], "pair_category_counts": summary["pair_category_counts"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
