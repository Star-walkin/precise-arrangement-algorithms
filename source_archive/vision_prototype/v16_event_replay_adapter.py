"""Convert V16 past-only predictions to a strict offline replay bundle.

This is historical candidate playback on the source AVI time base, NOT live
wall-clock performance or a controller. Candidate expiry equals first image
availability: candidates are point-in-time audit marks, never invented ongoing
countdowns. They remain in event_history after that instant, not current state.
Processing lower bounds stay in the provenance sidecar and cannot be mistaken
for measured camera-to-display latency. No human scoring file is an input.

Example (from repository root)::

    python -m vision_prototype.v16_event_replay_adapter --protocol PROTOCOL.json \
      --measurement MEASURE.json --prediction PREDICTION.json --output REPLAY.json

No main-UI entry point, video decoder, model or equipment interface is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

try:  # module invocation and the existing prototype's flat imports
    from .offline_event_replay import (OfflineEventReplay, SourceIdentity, TIME_BASE,
                                      schema_example, source_identity)
except ImportError:
    from offline_event_replay import (OfflineEventReplay, SourceIdentity, TIME_BASE,
                                     schema_example, source_identity)


PREDICTION_TIME_BASE = "source_video_frame_index_over_fps"
FORBIDDEN_TRUTH_KEYS = frozenset({"manual_truth", "ground_truth", "manual_review", "truth_s",
                                "signed_error_s", "absolute_error_s", "evaluation_scores",
                                "within_research_tolerance"})


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name}: finite number required")
    return float(value)


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name}: nonnegative integer required")
    return value


def _no_truth(value):
    if isinstance(value, dict):
        if FORBIDDEN_TRUTH_KEYS.intersection(value):
            raise ValueError("human truth or scoring fields must not enter an inference replay")
        for key in ("ground_truth_used", "labels_or_reference_event_times_used"):
            if key in value and value[key] is not False:
                raise ValueError("prediction must not use labels or reference times")
        for child in value.values():
            _no_truth(child)
    elif isinstance(value, list):
        for child in value:
            _no_truth(child)


def _source_from_video(video):
    return SourceIdentity.from_dict({"path": video["path"], "sha256": video["sha256"],
                                    "fps": video["fps"], "frame_count": video["frame_count"],
                                    "time_base": TIME_BASE})


def build_replay_bundle(prediction: dict, *, actual_source: SourceIdentity,
                        protocol_sha256: str, selected_indices: list[int],
                        step_frames: int) -> dict:
    """Pure data conversion; source identity must be computed from the real video.

    Every selected frame is mandatory, including unusable measurements. This
    never fills an absent t2 and never converts a candidate into confirmation.
    """
    _no_truth(prediction)
    if prediction.get("timebase") != PREDICTION_TIME_BASE:
        raise ValueError("V16 source AVI timebase is required, not host receive time")
    source = _source_from_video(prediction["video"])
    if source != SourceIdentity.from_dict(actual_source.as_dict()):
        raise ValueError("prediction source differs from verified video")
    step_frames = _integer(step_frames, "step_frames")
    if step_frames < 1:
        raise ValueError("step_frames must be positive")
    expected = [_integer(frame, "selected frame") for frame in selected_indices]
    if (not expected or expected != sorted(set(expected)) or
            any(frame % step_frames or frame >= source.frame_count for frame in expected)):
        raise ValueError("selected frame grid must be nonempty, unique and increasing")
    rows = prediction["prediction_rows"]
    if not isinstance(rows, list) or [r["source_frame"] for r in rows] != expected:
        raise ValueError("prediction frame sequence incomplete or reordered; failed frames must be retained")
    arm = prediction.get("arm")
    if arm not in ("center_only", "joint"):
        raise ValueError("unexpected V16 model arm")
    result = schema_example(source, protocol_sha256=protocol_sha256,
        algorithm_name=f"V16 {arm}: historical candidates on capture-time axis; not live deployment")
    result["sampling"] = {"step_frames": step_frames,
                           "observation_max_age_s": 1.25 * step_frames / source.fps}
    result["records"] = []
    emitted = []
    for raw in rows:
        frame = _integer(raw["source_frame"], "source_frame")
        time_s = _number(raw["time_s"], "time_s")
        if abs(time_s - frame / source.fps) > 1e-6:
            raise ValueError("V16 time_s must equal actual source frame / fps")
        inputs = raw["input"]
        if inputs["source_frame"] != frame or abs(_number(inputs["time_s"], "input.time_s")-time_s) > 1e-6:
            raise ValueError("event input does not belong to this source frame")
        if not isinstance(inputs["usable"], bool) or not isinstance(inputs.get("stationary", False), bool):
            raise ValueError("usable and stationary must be booleans")
        if inputs["usable"]:
            _number(inputs["gap_proxy"], "usable gap_proxy")
        reason = raw["measurement_reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("measurement_reason required for every frame")
        stationary = inputs.get("stationary", False)
        output = {"source_frame": frame, "source_time_s": time_s, "available_at_s": time_s,
                  "status": "available" if inputs["usable"] and not stationary else "unavailable",
                  "reason": "scene_stationary" if stationary else reason, "events": []}
        for candidate in raw["new_candidates"]:
            if (candidate.get("candidate_only") is not True or
                    any(candidate.get(flag) is not False for flag in
                        ("physical_layer_confirmed", "mechanical_turn_confirmed", "device_advisory"))):
                raise ValueError("only non-physical, non-device event candidates are accepted")
            kind = candidate["event"]
            if kind not in ("t0", "t1", "t2"):
                raise ValueError("unknown event kind")
            confirmation = _integer(candidate["confirmation_frame"], "confirmation_frame")
            occurred = _integer(candidate["source_frame"], "candidate.source_frame")
            available = _number(candidate["first_available_at_s"], "first_available_at_s")
            estimate = _number(candidate["event_time_s"], "event_time_s")
            if (confirmation != frame or occurred > confirmation or estimate < 0 or
                    estimate > available + 1e-9 or abs(available-time_s) > 1e-6):
                raise ValueError("event must be emitted on its actual confirmation frame, without future evidence")
            delay = _number(candidate["confirmation_delay_s"], "confirmation_delay_s")
            if abs(delay - (available-estimate)) > 1e-6:
                raise ValueError("confirmation delay inconsistent with candidate times")
            cycle = _integer(candidate["cycle_id"], "cycle_id")
            event_source = candidate["source"]
            if not isinstance(event_source, str) or not event_source.strip():
                raise ValueError("candidate source explanation required")
            output["events"].append({
                "event_id": f"cycle{cycle}_{kind}_source{occurred}_confirm{confirmation}",
                "kind": kind, "status": "candidate", "estimated_time_s": estimate,
                "available_at_s": available, "expires_at_s": available,
                "reason": event_source + "; historical mark only; no ongoing force window"})
            emitted.append(candidate)
            lower = candidate.get("first_available_processing_lower_bound_s")
            if lower is not None:
                if _number(lower, "processing lower bound") + 1e-9 < available:
                    raise ValueError("processing lower bound cannot precede capture availability")
        if raw.get("cumulative_candidate_count") != len(emitted):
            raise ValueError("cumulative candidate count does not match emitted prefix")
        result["records"].append(output)
    if prediction.get("candidates") != emitted:
        raise ValueError("summary candidates must exactly equal chronological newly emitted candidates")
    # Use the consumer validator too; adapter output cannot bypass its contract.
    OfflineEventReplay(result, actual_source=actual_source)
    return result


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def convert_files(protocol_path: Path, measurement_path: Path, prediction_path: Path,
                  output_path: Path) -> dict:
    """Bind the EVENT protocol, its measurement protocol, and source video."""
    protocol_path, measurement_path, prediction_path, output_path = map(
        Path, (protocol_path, measurement_path, prediction_path, output_path))
    sidecar = output_path.with_suffix(output_path.suffix + ".provenance.json")
    if output_path.exists() or sidecar.exists():
        raise FileExistsError("replay output or provenance exists; choose a fresh path")
    protocol, measurement, prediction = map(_read, (protocol_path, measurement_path, prediction_path))
    _no_truth(measurement)
    protocol_hash, measurement_hash, prediction_hash = map(_sha, (protocol_path, measurement_path, prediction_path))
    if (protocol.get("schema") != "v16_event_trial_1" or protocol.get("period_input") != "none" or
            protocol.get("ground_truth_used") is not False or
            prediction.get("schema") != "v16_event_prediction_1" or
            prediction.get("ground_truth_used") is not False or
            prediction.get("protocol_sha256") != protocol_hash):
        raise ValueError("unrecognized or mismatched frozen V16 event protocol")
    reference = protocol["measurement_protocol"]
    if _sha(reference["path"]) != reference["sha256"]:
        raise ValueError("measurement protocol changed")
    measurement_protocol = _read(reference["path"])
    session = prediction["session"]
    if (measurement["session"] != session or measurement["arm"] != prediction["arm"] or
            measurement["protocol_sha256"] != reference["sha256"] or
            prediction.get("measurement_protocol_sha256") != reference["sha256"] or
            prediction.get("model_sha256") != measurement_protocol["models"][prediction["arm"]]["sha256"]):
        raise ValueError("measurement provenance does not match video protocol or prediction")
    if (prediction.get("measurement_sha256") != measurement_hash or
            Path(prediction["measurement_path"]).resolve() != measurement_path.resolve()):
        raise ValueError("prediction must bind the exact measurement SHA256")
    if any(c["event"] == "t2" for c in prediction["candidates"]):
        raise ValueError("this frozen trial supplies no independent full-turn phase; t2 is forbidden")
    video = measurement_protocol["videos"][session]
    actual = source_identity(video["path"], fps=video["fps"], frame_count=video["frame_count"])
    if actual != _source_from_video(video):
        raise ValueError("original video no longer matches its frozen protocol")
    expected = video["selected_indices"]
    if protocol["selected_indices"][session] != expected:
        raise ValueError("event and measurement sampling schedules differ")
    if [r["frame_index"] for r in measurement["records"]] != expected:
        raise ValueError("measurement frame sequence incomplete")
    output = build_replay_bundle(prediction, actual_source=actual, protocol_sha256=protocol_hash,
                                 selected_indices=expected, step_frames=measurement_protocol["sampling"]["stride_frames"])
    # Verify the adapter has not been handed usable flags from another run.
    for measured, predicted in zip(measurement["records"], prediction["prediction_rows"]):
        if (measured["features"]["usable"] != predicted["input"]["usable"] or
                measured["features"]["reason"] != predicted["measurement_reason"] or
                measured["features"].get("image_gap_proxy") != predicted["input"]["gap_proxy"] or
                abs(measured["time_s"]-predicted["time_s"]) > 1e-6):
            raise ValueError("prediction rows do not preserve measurement validity/reasons/time")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against a concurrently produced artifact.
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2, allow_nan=False)
    provenance = {"schema": "v16_replay_adapter_provenance/v1", "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": protocol_hash, "measurement_path": str(measurement_path.resolve()),
        "measurement_protocol_sha256": reference["sha256"],
        "measurement_sha256": measurement_hash, "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": prediction_hash, "replay_sha256": _sha(output_path),
        "adapter_sha256": _sha(__file__), "records": len(output["records"]),
        "unavailable_records": sum(r["status"] == "unavailable" for r in output["records"]),
        "candidate_counts": {kind: sum(e["kind"] == kind for r in output["records"] for e in r["events"])
                             for kind in ("t0", "t1", "t2")},
        "event_policy": "point_in_time_history_only_no_inferred_ttl_or_force_window",
        "availability_policy": "offline_capture_time_axis_not_measured_end_to_end_latency",
        "processing_lower_bounds_retained_in_prediction_file_only": True,
        "ground_truth_used": False, "device_advisory": False, "deployment_allowed": False}
    with sidecar.open("x", encoding="utf-8") as handle:
        json.dump(provenance, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "measurement", "prediction", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(convert_files(args.protocol, args.measurement, args.prediction, args.output),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
