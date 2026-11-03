"""Frozen input inventory and post-prediction scoring for the geometry G0/G1 arms.

This module never supplies target reference geometry to the predictor. The
``inputs`` and ``freeze`` commands do not load evaluation labels. ``score``
checks the complete 8-image x 2-mode prediction set and its SHA-256 receipt
before the first trusted partial reference is opened.

Run each arm independently, for example::

    python evaluation_adapter.py inputs --output evaluation_inputs.json
    python evaluation_adapter.py freeze --manifest evaluation_inputs.json \
      --predictions-dir evaluation_predictions/G0 --arm G0
    python evaluation_adapter.py score --manifest evaluation_inputs.json \
      --predictions-dir evaluation_predictions/G0 --output-dir evaluation/G0 --arm G0

The historical eight are previously open development images, not a blind set.
All distances and lengths are original-image pixels. No endpoint identity or
physical contact success is inferred from this partial centerline reference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.entry_conditioned_rope_path import compare_evaluate as comparison
from experiments.entry_conditioned_rope_path import evaluate as established
from experiments.entry_conditioned_rope_path import scene as scene_module
from vision_geometry_lab.centerline_path_metrics_v6 import _nearest, clip_scoring_segments
from vision_geometry_lab.experiments.v28_local_spool_connection import data_eval


INDEX = BASE / "runs" / "run_20261030_002" / "data" / "index.json"
MODES = ("AUTO", "CONDITIONAL")
STATUS_MAP = {
    "entry_condition": "entry_condition",
    "visible_supported": "visible_supported",
    "ambiguous": "ambiguous",
    "inferred": "inferred",
    # The geometry runner's evidence vocabulary maps to the established
    # partial-reference evaluator without changing support semantics.
    "input_entry": "entry_condition",
    "image_supported": "visible_supported",
    "unsupported_gap": "inferred",
}
TOLERANCE_PX = established.PROTOCOL["matching_tolerance_px"]
FROZEN_NAME = "prediction_hashes.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant in {path}: {value}")

    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle, parse_constant=reject_constant)


def write_new(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _xy(value, *, allow_empty: bool = True):
    points = np.asarray(value, dtype=np.float64)
    if points.size == 0 and allow_empty:
        return np.empty((0, 2), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Expected finite Nx2 original-image coordinates")
    if not allow_empty and len(points) < 2:
        raise ValueError("A path requires at least two points")
    return points


def build_inputs():
    """Only fixed image/entry metadata; no annotation or quality files."""
    samples = {row["sample_id"]: row for row in data_eval.list_inputs(verify_images=False)}
    source = read_json(INDEX)
    evaluated = {row["id"]: row for row in source["items"] if row["split"] == "eval"}
    if set(evaluated) != set(data_eval.SAMPLE_IDS) or len(evaluated) != 8:
        raise ValueError("The historical eight-image input split changed")

    conditions = []
    for mode in MODES:
        for sample_id in data_eval.SAMPLE_IDS:
            item, sample = evaluated[sample_id], samples[sample_id]
            if (item["image_path"] != sample["image_path"] or
                    item["image_sha256"] != sample["image_sha256"] or
                    item["source_group"] != sample["group_id"]):
                raise ValueError(f"Input identity/group mismatch: {sample_id}")
            shape = tuple(sample["image_shape_hw"])
            if shape != (3000, 4096):
                raise ValueError(f"Unexpected historical image shape: {sample_id}: {shape}")
            bbox = item["bbox"]
            if len(bbox) != 4 or not (0 <= bbox[0] < bbox[2] <= 4096 and
                                          0 <= bbox[1] < bbox[3] <= 3000):
                raise ValueError(f"Invalid frozen search box: {sample_id}")
            if mode == "AUTO":
                entry_path = Path(item["auto_entry_path"])
                entry = read_json(entry_path)
                if entry.get("image_sha256") != item["image_sha256"]:
                    raise ValueError(f"AUTO entry/image mismatch: {sample_id}")
                available = bool(entry["available"])
                seed = entry["free_seed"] if available else []
                seed_source = str(entry_path)
                seed_source_sha256 = sha256(entry_path)
                seed_origin = entry.get("mode", "AUTO_FROZEN_ENTRY")
            else:
                seed = item["known_seed_xy"]
                available = True
                seed_source = str(INDEX) + "#items/" + sample_id + "/known_seed_xy"
                seed_source_sha256 = sha256(INDEX)
                seed_origin = "CONDITIONAL_KNOWN_ENTRY"
            seed = _xy(seed).tolist()
            if available and len(seed) < 2:
                raise ValueError(f"Missing declared entry: {mode}/{sample_id}")
            if not available and seed:
                raise ValueError(f"Unavailable AUTO entry has coordinates: {sample_id}")
            conditions.append({
                "id": sample_id, "mode": mode,
                "source_path": item["image_path"],
                "image_sha256": item["image_sha256"],
                "image_shape_hw": list(shape),
                "eval_group": item["source_group"],
                "bbox_original_xyxy": bbox,
                "input_seed_xy": seed,
                "entry_available": available,
                "seed_source": seed_source,
                "seed_source_sha256": seed_source_sha256,
                "seed_origin": seed_origin,
            })
    groups = {row["eval_group"] for row in conditions}
    if len(conditions) != 16 or len(groups) != 3:
        raise ValueError("Expected 16 conditions from three source groups")
    return {
        "schema": "geometry_direction_evaluation_inputs_v1",
        "population": "fixed historical eight previously open development images, not blind",
        "modes": list(MODES),
        "sample_ids": list(data_eval.SAMPLE_IDS),
        "source_index": str(INDEX),
        "source_index_sha256": sha256(INDEX),
        "entry_is_input_not_detector_output": True,
        "labels_read": False,
        "conditions": conditions,
    }


def load_manifest(path: Path):
    manifest = read_json(path)
    if manifest != build_inputs():
        raise ValueError("Evaluation input manifest differs from frozen sources")
    return manifest


def _candidate(candidate, shape):
    if not isinstance(candidate, dict) or not isinstance(candidate.get("id"), str) or not candidate["id"]:
        raise ValueError("Each candidate needs a nonempty string ID")
    points = _xy(candidate.get("points_xy", []), allow_empty=False)
    if (points[:, 0] < 0).any() or (points[:, 0] >= shape[1]).any() or \
            (points[:, 1] < 0).any() or (points[:, 1] >= shape[0]).any():
        raise ValueError(f"Candidate {candidate['id']} leaves the original image")
    status = candidate.get("point_status")
    if not isinstance(status, list) or len(status) != len(points) or set(status) - STATUS_MAP.keys():
        raise ValueError(f"Candidate {candidate['id']} has invalid point_status")
    return points, status


def validate_prediction(path: Path, condition):
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Prediction is not a JSON object: {path}")
    if value.get("id", condition["id"]) != condition["id"] or \
            value.get("mode", condition["mode"]) != condition["mode"]:
        raise ValueError(f"Prediction ID/mode mismatch: {path}")
    seed = _xy(value.get("input_seed_xy", []))
    expected = _xy(condition["input_seed_xy"])
    if seed.shape != expected.shape or not np.allclose(seed, expected, rtol=0, atol=1e-6):
        raise ValueError(f"Prediction used a different supplied entry: {path}")
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or len(candidates) > 3:
        raise ValueError(f"Expected at most three candidates: {path}")
    ids = []
    for candidate in candidates:
        _candidate(candidate, condition["image_shape_hw"])
        ids.append(candidate["id"])
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate candidate IDs: {path}")
    selected = value.get("selected_candidate_id")
    if selected is not None and selected not in ids:
        raise ValueError(f"Selected candidate absent from pool: {path}")
    if not condition["entry_available"] and (seed.size or selected is not None or candidates):
        raise ValueError(f"AUTO entry failure cannot yield a selected path: {path}")
    if not isinstance(value.get("status"), str) or not isinstance(value.get("stop_reason"), str):
        raise ValueError(f"Prediction requires status and stop_reason: {path}")
    if "image_sha256" in value and value["image_sha256"] != condition["image_sha256"]:
        raise ValueError(f"Prediction source image mismatch: {path}")
    return value


def prediction_paths(predictions_dir: Path, manifest):
    predictions_dir = Path(predictions_dir)
    if not predictions_dir.is_dir():
        raise FileNotFoundError(predictions_dir)
    expected = {f"{row['mode']}/{row['id']}.json" for row in manifest["conditions"]}
    actual = {f"{mode}/{p.name}" for mode in MODES
              for p in (predictions_dir / mode).glob("*.json")}
    if actual != expected:
        raise ValueError(f"Prediction inventory mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    return [(row, predictions_dir / row["mode"] / (row["id"] + ".json"))
            for row in manifest["conditions"]]


def freeze(manifest_path: Path, predictions_dir: Path, arm: str):
    manifest = load_manifest(manifest_path)
    rows = []
    for condition, path in prediction_paths(predictions_dir, manifest):
        validate_prediction(path, condition)
        rows.append({"relative_path": f"{condition['mode']}/{condition['id']}.json",
                     "sha256": sha256(path)})
    receipt = {
        "schema": "geometry_direction_prediction_hashes_v1", "arm": arm,
        "case_count": 16, "image_count": 8, "source_group_count": 3,
        "input_manifest_sha256": sha256(manifest_path),
        "adapter_sha256": sha256(Path(__file__)),
        "all_predictions_saved_before_reference": True,
        "reference_read": False,
        "files": rows,
    }
    output = Path(predictions_dir) / FROZEN_NAME
    write_new(output, receipt)
    return output


def verify_frozen(manifest_path: Path, predictions_dir: Path, arm: str):
    manifest = load_manifest(manifest_path)
    rows = prediction_paths(predictions_dir, manifest)
    receipt_path = Path(predictions_dir) / FROZEN_NAME
    if not receipt_path.is_file():
        raise FileNotFoundError("Freeze all 16 predictions before scoring: " + str(receipt_path))
    receipt = read_json(receipt_path)
    if receipt.get("schema") != "geometry_direction_prediction_hashes_v1" or \
            receipt.get("arm") != arm or receipt.get("case_count") != 16 or \
            receipt.get("input_manifest_sha256") != sha256(manifest_path) or \
            receipt.get("adapter_sha256") != sha256(Path(__file__)) or \
            not receipt.get("all_predictions_saved_before_reference"):
        raise ValueError("Prediction freeze receipt does not match this run")
    expected = [{"relative_path": f"{condition['mode']}/{condition['id']}.json",
                 "sha256": sha256(path)} for condition, path in rows]
    if receipt.get("files") != expected:
        raise ValueError("Saved prediction hashes changed after freeze")
    validated = [(condition, path, validate_prediction(path, condition)) for condition, path in rows]
    return manifest, receipt_path, validated


def _supported_segments(candidate):
    """Only path edges explicitly supported by source-image evidence."""
    if candidate is None:
        return []
    points, status = candidate["points_xy"], candidate["point_status"]
    segments = []
    for a, b, state_a, state_b in zip(points[:-1], points[1:], status[:-1], status[1:]):
        if state_b == "visible_supported" and state_a in ("entry_condition", "visible_supported"):
            segments.append([a, b])
    return segments


def _longest_match(pieces, predicted_segments):
    """Longest 5px matched run within each clipped trusted reference piece."""
    longest = 0.0
    for piece in pieces:
        distance, _ = _nearest(piece["points"], predicted_segments)
        run = 0.0
        for matched, weight in zip(distance <= TOLERANCE_PX, piece["weights"]):
            run = run + float(weight) if matched else 0.0
            longest = max(longest, run)
    return longest


def _scored_segments(candidate, reference, input_polygons):
    if candidate is None:
        return []
    return clip_scoring_segments(established._without_input_edges(candidate),
                                 ignore_polygons=[*reference["ignore_polygons"], *input_polygons])


def score(manifest_path: Path, predictions_dir: Path, output_dir: Path, arm: str):
    # No call to load_evaluation_reference may precede this complete gate.
    manifest, receipt_path, validated = verify_frozen(manifest_path, predictions_dir, arm)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Choose a new evaluation output directory: {output_dir}")
    source_items = {row["id"]: row for row in read_json(INDEX)["items"] if row["split"] == "eval"}

    references = {}
    detailed = []
    established_rows = []
    for condition, path, prediction in validated:
        sample_id = condition["id"]
        if sample_id not in references:
            references[sample_id] = data_eval.load_evaluation_reference(sample_id)
        reference = references[sample_id]
        scoring_scene = read_json(Path(source_items[sample_id]["scene_path"]))
        candidates = []
        for candidate in prediction["candidates"]:
            points = _xy(candidate["points_xy"], allow_empty=False)
            stages = scene_module.stage_at_points(scoring_scene, points).tolist()
            candidates.append({"id": candidate["id"], "kind": candidate.get("kind"),
                               "points_xy": candidate["points_xy"],
                               "point_status": [STATUS_MAP[state] for state in candidate["point_status"]],
                               "stage": stages})
        normalized_prediction = {"candidates": candidates,
                                 "selected_candidate_id": prediction["selected_candidate_id"]}
        case = {"sample_id": sample_id, "mode": condition["mode"],
                "prediction": normalized_prediction, "scene": scoring_scene,
                "scoring_scene": scoring_scene,
                "actual_input_seed_xy": condition["input_seed_xy"],
                "response_distribution": {"status": "not_applicable_geometry_only"},
                "prediction_files": {"path": str(path), "sha256": sha256(path)}}
        evaluated = established.evaluate_case(case, reference, scene_module.stage_at_points)
        established_rows.append(evaluated)

        input_polygons = established._entry_polygons(condition["input_seed_xy"])
        quality_segments = clip_scoring_segments([reference["centerline"]],
                                                 ignore_polygons=reference["ignore_polygons"])
        trusted_segments = clip_scoring_segments(quality_segments, ignore_polygons=input_polygons)
        pieces = established._reference_samples(trusted_segments, scoring_scene, scene_module.stage_at_points)
        selected = next((c for c in candidates if c["id"] == prediction["selected_candidate_id"]), None)
        all_segments = _scored_segments(selected, reference, input_polygons)
        supported_segments = clip_scoring_segments(_supported_segments(selected),
            ignore_polygons=[*reference["ignore_polygons"], *input_polygons])
        supported = {"all": established._summarize_reference_rows(pieces, supported_segments)}
        supported["stages"] = {name: established._summarize_reference_rows(pieces, supported_segments, number)
                               for number, name in established.STAGES.items()}
        first_deviation = comparison.first_deviation(evaluated["selected"], reference)
        detail = {
            "arm": arm, "id": sample_id, "mode": condition["mode"],
            "eval_group": condition["eval_group"],
            "entry_available": condition["entry_available"],
            "entry_origin": condition["seed_origin"],
            "prediction_path": str(path), "prediction_sha256": sha256(path),
            "status": prediction["status"], "stop_reason": prediction["stop_reason"],
            "selected_candidate_id": prediction["selected_candidate_id"],
            "candidate_count": len(candidates),
            "selected_partial_reference": evaluated["selected"],
            "supported_only_partial_reference": supported,
            "longest_continuous_5px_match_px": _longest_match(pieces, all_segments),
            "longest_continuous_supported_5px_match_px": _longest_match(pieces, supported_segments),
            "first_sustained_reference_gap_ge60px": first_deviation,
            "visible_ribbon_diagnostic": evaluated["legacy_selected_visible_connection_status"],
            "reference_annotation_sha256": reference["annotation_sha256"],
            "reference_quality_sha256": reference["quality_sha256"],
            "unknown_tail_policy": "unannotated/quality-excluded regions are unknown, not negative",
            "whole_arc_identity_success": None,
            "physical_contact_confirmed": False,
        }
        detailed.append(detail)

    aggregate = established._aggregate(established_rows)
    supported_summary = {}
    for mode in MODES:
        rows = [r for r in detailed if r["mode"] == mode]
        by_stage = {}
        for key in ("all", *established.STAGES.values()):
            if key == "all":
                scores = [r["supported_only_partial_reference"]["all"] for r in rows]
            else:
                scores = [r["supported_only_partial_reference"]["stages"][key] for r in rows]
            length = sum(s["reference_length_px"] for s in scores)
            covered = sum(s["covered_reference_length_px"] or 0.0 for s in scores)
            by_stage[key] = {"reference_length_px": length,
                             "covered_reference_length_px": covered,
                             "coverage_5px": covered / length if length else None,
                             "image_population_denominator": 8}
        supported_summary[mode] = by_stage
    summary = {
        "status": "completed", "arm": arm, "image_population_denominator_per_mode": 8,
        "source_group_count": 3, "input_manifest": str(manifest_path),
        "input_manifest_sha256": sha256(manifest_path),
        "prediction_freeze_receipt": str(receipt_path),
        "prediction_freeze_sha256": sha256(receipt_path),
        "adapter_sha256": sha256(Path(__file__)),
        "scoring_protocol": {
            "reference": "same eight previously open development images, trusted partial centerline",
            "matching_tolerance_original_px": TOLERANCE_PX,
            "maximum_reference_sampling_spacing_px": established.PROTOCOL["maximum_sampling_spacing_px"],
            "weighting": established.PROTOCOL["weighting"],
            "quality_exclusion": "existing ignore and precision-exclude polygons",
            "actual_input_seed_exclusion_radius_px": established.PROTOCOL["input_seed_exclusion_radius_px"],
            "first_sustained_gap_threshold_px": 60.0,
            "supported_only": "only edges labelled visible_supported after input exclusion",
            "stages": "frozen visual location proxies; not physical contact stages",
            "unsupported_or_unlabelled_tail": "unknown, not background",
            "full_identity_or_endpoint_accuracy": None,
        },
        "selected_partial_reference": aggregate,
        "supported_only_partial_reference": supported_summary,
        "rows": detailed,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    for row in detailed:
        write_new(output_dir / row["mode"] / (row["id"] + ".json"), row)
    write_new(output_dir / "summary.json", summary)
    fields = ["arm", "mode", "id", "eval_group", "entry_available", "status", "stop_reason",
              "selected_candidate_id", "candidate_count", "all_coverage_5px", "attached_coverage_5px",
              "edge_near_coverage_5px", "supported_only_all_coverage_5px",
              "longest_continuous_5px_match_px", "longest_continuous_supported_5px_match_px",
              "longest_uncovered_run_px", "first_sustained_gap_status", "visible_ribbon_diagnostic"]
    with (output_dir / "per_case_metrics.csv").open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in detailed:
            chosen = row["selected_partial_reference"]
            writer.writerow({
                "arm": arm, "mode": row["mode"], "id": row["id"],
                "eval_group": row["eval_group"], "entry_available": row["entry_available"],
                "status": row["status"], "stop_reason": row["stop_reason"],
                "selected_candidate_id": row["selected_candidate_id"],
                "candidate_count": row["candidate_count"],
                "all_coverage_5px": chosen["all"]["covered_fraction"],
                "attached_coverage_5px": chosen["stages"]["attached"]["covered_fraction"],
                "edge_near_coverage_5px": chosen["stages"]["edge_near"]["covered_fraction"],
                "supported_only_all_coverage_5px": row["supported_only_partial_reference"]["all"]["covered_fraction"],
                "longest_continuous_5px_match_px": row["longest_continuous_5px_match_px"],
                "longest_continuous_supported_5px_match_px": row["longest_continuous_supported_5px_match_px"],
                "longest_uncovered_run_px": chosen["all"]["longest_uncovered_run_px"],
                "first_sustained_gap_status": row["first_sustained_reference_gap_ge60px"]["status"],
                "visible_ribbon_diagnostic": row["visible_ribbon_diagnostic"],
            })
    return output_dir / "summary.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inputs = commands.add_parser("inputs", help="Export fixed image and entry metadata, without labels")
    inputs.add_argument("--output", required=True, help="New JSON path or '-' for stdout")
    for name in ("freeze", "score"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--predictions-dir", type=Path, required=True)
        command.add_argument("--arm", choices=("G0", "G1"), required=True)
        if name == "score":
            command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inputs":
        manifest = build_inputs()
        if args.output == "-":
            print(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
        else:
            write_new(Path(args.output), manifest)
            print(args.output)
    elif args.command == "freeze":
        print(freeze(args.manifest, args.predictions_dir, args.arm))
    else:
        print(score(args.manifest, args.predictions_dir, args.output_dir, args.arm))


if __name__ == "__main__":
    main()
