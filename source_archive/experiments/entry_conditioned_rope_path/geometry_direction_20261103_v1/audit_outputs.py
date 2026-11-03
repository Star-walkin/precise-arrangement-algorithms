"""Independent, read-only verification of the geometry gallery outputs.

For the full run this checks all 292 source/PNG/prediction/evidence contracts
and exactly redraws a deterministic sample with the saved G1 evidence. A pilot
can be rehearsed with --read-only, which writes no receipt. The audit never
recomputes or changes a predicted path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageChops

import run_geometry


HERE = Path(__file__).resolve().parent
DEFAULT_RUN = Path(r"C:\precise_arrangement_results\geometry_direction_20261103_v1\all_001")
INPUTS = HERE / "inputs.json"
EXPECTED_IMAGE_SHAPE = (3000, 4096)
EXPECTED_BOARD_SIZE = (
    2 * EXPECTED_IMAGE_SHAPE[1] + run_geometry.GAP + run_geometry.SIDE,
    EXPECTED_IMAGE_SHAPE[0] + run_geometry.HEADER + run_geometry.FOOTER,
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_ids(ids: list[str], selected: dict[str, dict], count: int) -> list[str]:
    if not ids:
        return []
    n = min(len(ids), max(count, 1))
    positions = {round(k * (len(ids) - 1) / max(n - 1, 1)) for k in range(n)}
    by_status: dict[str, str] = {}
    for source_id in ids:
        row = selected.get(source_id, {})
        key = str(row.get("status_G1"))
        by_status.setdefault(key, source_id)
        if row.get("failure"):
            by_status.setdefault("failed", source_id)
        if row.get("status_G0") != row.get("status_G1"):
            by_status.setdefault("arm_difference", source_id)
    chosen = {ids[p] for p in positions} | set(by_status.values())
    return [source_id for source_id in ids if source_id in chosen]


def read_source(row: dict, errors: list[str]) -> np.ndarray | None:
    path = Path(row["source_path"])
    try:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["source_sha256"]:
            errors.append("source_sha256_mismatch")
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            errors.append("source_decode_failed")
            return None
        if image.shape[:2] != EXPECTED_IMAGE_SHAPE:
            errors.append("source_shape_mismatch")
        return image
    except Exception as exc:
        errors.append(f"source_read_error:{type(exc).__name__}:{exc}")
        return None


def load_evidence(path: Path, config: dict, errors: list[str]) -> dict | None:
    if not path.is_file():
        errors.append("evidence_missing")
        return None
    expected = {
        "geometry_response": np.float32,
        "valid_domain": np.uint8,
        "direction_cos2": np.float32,
        "direction_sin2": np.float32,
        "direction_confidence": np.float32,
        "field_scale": np.float32,
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(expected):
                errors.append("evidence_keys_mismatch")
            arrays = {key: archive[key] for key in expected if key in archive.files}
        for key, dtype in expected.items():
            if key not in arrays:
                continue
            if arrays[key].dtype != dtype:
                errors.append(f"evidence_dtype_mismatch:{key}")
        if any(key not in arrays for key in expected):
            return None
        scale = float(arrays["field_scale"])
        if not math.isfinite(scale) or abs(scale - float(config["field_scale"])) > 1e-6:
            errors.append("evidence_field_scale_config_mismatch")
        expected_shape = tuple(int(round(d * scale)) for d in EXPECTED_IMAGE_SHAPE)
        for key in expected:
            if key == "field_scale":
                continue
            if arrays[key].shape != expected_shape:
                errors.append(f"evidence_shape_mismatch:{key}")
        if arrays["field_scale"].shape not in ((), (1,)):
            errors.append("evidence_field_scale_shape_mismatch")
        valid = arrays["valid_domain"]
        if not np.isin(valid, (0, 1)).all():
            errors.append("valid_domain_nonbinary")
        valid = valid.astype(bool)
        response = arrays["geometry_response"]
        if valid.shape == response.shape:
            if not valid.any():
                errors.append("valid_domain_empty")
            if not np.isfinite(response[valid]).all():
                errors.append("valid_response_nonfinite")
            if valid.any() and (np.min(response[valid]) < -1e-6 or np.max(response[valid]) > 1 + 1e-6):
                errors.append("valid_response_outside_zero_one")
            if np.isfinite(response[~valid]).any():
                errors.append("uncomputed_response_not_nan")
        for key in ("direction_cos2", "direction_sin2", "direction_confidence"):
            if not np.isfinite(arrays[key]).all():
                errors.append(f"direction_field_nonfinite:{key}")
        confidence = arrays["direction_confidence"]
        if np.isfinite(confidence).all() and (confidence.min() < -1e-6 or confidence.max() > 1 + 1e-6):
            errors.append("direction_confidence_outside_zero_one")
        return {"geometry_response": response, "valid_domain": valid}
    except Exception as exc:
        errors.append(f"evidence_decode_error:{type(exc).__name__}:{exc}")
        return None


def validate_prediction(pred: dict, row: dict, arm: str, config_sha: str, errors: list[str]) -> None:
    source_id = row["source_id"]
    for key in ("id", "source_id"):
        if pred.get(key) != source_id:
            errors.append(f"prediction_{arm}_{key}_mismatch")
    if pred.get("image_path") != row["source_path"] or pred.get("image_sha256") != row["source_sha256"]:
        errors.append(f"prediction_{arm}_source_mismatch")
    if pred.get("arm") != arm or pred.get("use_direction") is not (arm == "G1"):
        errors.append(f"prediction_{arm}_arm_mismatch")
    if pred.get("config_sha256") != config_sha:
        errors.append(f"prediction_{arm}_config_mismatch")
    if pred.get("network_response_used") is not False or pred.get("entry_is_input_condition") is not True:
        errors.append(f"prediction_{arm}_input_semantics_mismatch")
    seed = row.get("e1", {})
    expected_seed = seed.get("xy") if seed.get("valid") else []
    if pred.get("input_seed_xy") != expected_seed or pred.get("entry_source") != seed.get("source"):
        errors.append(f"prediction_{arm}_entry_mismatch")
    candidates = pred.get("candidates")
    if not isinstance(candidates, list):
        errors.append(f"prediction_{arm}_candidates_not_list")
        return
    candidate_ids: set[str] = set()
    for candidate in candidates:
        candidate_id = candidate.get("id")
        if not isinstance(candidate_id, str) or candidate_id in candidate_ids:
            errors.append(f"prediction_{arm}_candidate_id_invalid")
        candidate_ids.add(candidate_id)
        points = candidate.get("points_xy", [])
        states = candidate.get("point_status", [])
        if not isinstance(points, list) or len(points) != len(states):
            errors.append(f"prediction_{arm}_path_state_length_mismatch")
            continue
        for point in points:
            if not isinstance(point, list) or len(point) != 2 or not all(
                isinstance(v, (int, float)) and math.isfinite(v) for v in point
            ):
                errors.append(f"prediction_{arm}_invalid_path_coordinate")
                break
    chosen = pred.get("selected_candidate_id")
    if chosen is not None and chosen not in candidate_ids:
        errors.append(f"prediction_{arm}_selected_missing_from_candidates")
    selected = pred.get("selected")
    if selected is not None and selected.get("id") != chosen:
        errors.append(f"prediction_{arm}_selected_record_mismatch")
    if not isinstance(pred.get("status"), str) or not isinstance(pred.get("stop_reason"), str):
        errors.append(f"prediction_{arm}_status_missing")


def audit_one(
    row: dict, receipt: dict, run: Path, gallery: Path, config: dict, config_sha: str,
    redraw: bool,
) -> dict:
    source_id = row["source_id"]
    errors: list[str] = []
    failure_recorded = bool(receipt.get("failure"))
    if receipt.get("source_id") != source_id or receipt.get("record_key") != source_id:
        errors.append("run_receipt_source_id_mismatch")
    if receipt.get("source_sha256") not in (None, row["source_sha256"]):
        errors.append("run_receipt_source_hash_mismatch")
    image = read_source(row, errors)
    png_path = gallery / (source_id + ".png")
    board = None
    if not png_path.is_file():
        errors.append("png_missing")
    else:
        if receipt.get("png_path") != str(png_path):
            errors.append("run_receipt_png_path_mismatch")
        if file_sha256(png_path) != receipt.get("png_sha256"):
            errors.append("png_sha256_mismatch")
        try:
            with Image.open(png_path) as opened:
                opened.load()
                if opened.format != "PNG" or opened.mode != "RGB":
                    errors.append("png_format_or_mode_mismatch")
                if opened.size != EXPECTED_BOARD_SIZE:
                    errors.append("png_board_size_mismatch")
                board = opened.copy() if redraw else None
                if image is not None and opened.size == EXPECTED_BOARD_SIZE:
                    crop = opened.crop((0, run_geometry.HEADER, 4096, run_geometry.HEADER + 3000))
                    source_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if not np.array_equal(np.asarray(crop), source_rgb):
                        errors.append("left_panel_pixels_differ_from_source")
        except Exception as exc:
            errors.append(f"png_decode_error:{type(exc).__name__}:{exc}")

    predictions: dict[str, dict] = {}
    for arm in ("G0", "G1"):
        path = run / "predictions" / arm / (source_id + ".json")
        try:
            pred = read_json(path)
            predictions[arm] = pred
            validate_prediction(pred, row, arm, config_sha, errors)
            if receipt.get(f"status_{arm}") != pred.get("status"):
                errors.append(f"run_receipt_{arm}_status_mismatch")
            if receipt.get(f"stop_{arm}") not in (None, pred.get("stop_reason")):
                errors.append(f"run_receipt_{arm}_stop_mismatch")
        except Exception as exc:
            errors.append(f"prediction_{arm}_read_error:{type(exc).__name__}:{exc}")

    evidence = None
    evidence_path = run / "evidence_float32" / (source_id + ".npz")
    if failure_recorded:
        # A late exception may leave a valid earlier field file. The failure
        # renderer deliberately displays neutral heat, so that file is not
        # used for exact redraw and its presence is not a false success.
        if any(predictions.get(arm, {}).get("status") != "single_image_failure" for arm in ("G0", "G1")):
            errors.append("failed_case_prediction_status_mismatch")
    else:
        if receipt.get("evidence_path") != str(evidence_path):
            errors.append("run_receipt_evidence_path_mismatch")
        if evidence_path.is_file() and file_sha256(evidence_path) != receipt.get("evidence_sha256"):
            errors.append("evidence_sha256_mismatch")
        evidence = load_evidence(evidence_path, config, errors)

    redraw_match: bool | None = None
    if redraw and image is not None and board is not None and "G1" in predictions:
        try:
            fields = evidence if evidence is not None else {
                "geometry_response": np.zeros(EXPECTED_IMAGE_SHAPE, np.float32),
                "valid_domain": np.zeros(EXPECTED_IMAGE_SHAPE, bool),
            }
            expected = run_geometry.render(image, fields, predictions["G1"], row, config)
            redraw_match = ImageChops.difference(board, expected).getbbox() is None
            if not redraw_match:
                errors.append("g1_png_exact_redraw_mismatch")
        except Exception as exc:
            errors.append(f"g1_png_redraw_error:{type(exc).__name__}:{exc}")
    return {
        "source_id": source_id,
        "errors": errors,
        "status_G0": predictions.get("G0", {}).get("status"),
        "status_G1": predictions.get("G1", {}).get("status"),
        "failed_in_run": failure_recorded,
        "redraw_checked": redraw,
        "redraw_exact_match": redraw_match,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--inputs", type=Path, default=INPUTS)
    parser.add_argument("--read-only", action="store_true", help="Print a pilot audit; do not write audit_receipt.json")
    parser.add_argument("--redraw-samples", type=int, default=8)
    args = parser.parse_args()
    run = args.run.resolve()
    inputs = read_json(args.inputs)
    start = read_json(run / "run_start.json")
    execution = read_json(run / "run_receipt.json")
    phase = execution.get("phase")
    if phase not in ("all", "pilot"):
        raise ValueError("This gallery audit supports all_001 and pilot_001 only")
    gallery = run / ("final_292" if phase == "all" else "pilot_png")
    input_by_id = {row["source_id"]: row for row in inputs["rows"]}
    expected_ids = [row["source_id"] for row in inputs["rows"]] if phase == "all" else [
        item["source_id"] for item in execution["rows"]
    ]
    errors: list[str] = []
    warnings: list[str] = []
    if len(input_by_id) != 292 or inputs.get("source_count") != 292:
        errors.append("inputs_not_exactly_292_unique_sources")
    if len(set(expected_ids)) != len(expected_ids) or any(source_id not in input_by_id for source_id in expected_ids):
        errors.append("run_ids_not_unique_or_not_in_inputs")
    if phase == "all" and len(expected_ids) != 292:
        errors.append("all_run_receipt_not_292_rows")
    if start.get("row_count") != len(expected_ids) or execution.get("input_rows") != len(expected_ids):
        errors.append("run_start_or_receipt_row_count_mismatch")
    if execution.get("completed_rows") != len(expected_ids):
        errors.append("run_receipt_incomplete")
    if start.get("inputs_sha256") != file_sha256(args.inputs):
        errors.append("inputs_sha256_mismatch")
    if start.get("network_response_used") is not False or execution.get("network_response_used") is not False:
        errors.append("run_network_response_flag_mismatch")
    config_path = Path(start["config_path"])
    config_sha = file_sha256(config_path)
    if config_sha != start.get("config_sha256") or config_sha != execution.get("config_sha256"):
        errors.append("run_config_sha256_mismatch")
    runner_sha = file_sha256(HERE / "run_geometry.py")
    if runner_sha != execution.get("script_sha256"):
        warnings.append("current_runner_source_differs_from_run_receipt; exact redraw still checked")
    config = read_json(config_path)
    expected_files = {source_id + ".png" for source_id in expected_ids}
    actual_files = {path.name for path in gallery.glob("*.png")}
    if actual_files != expected_files:
        errors.append(f"gallery_file_set_mismatch:missing={len(expected_files - actual_files)},extra={len(actual_files - expected_files)}")
    for arm in ("G0", "G1"):
        actual_predictions = {path.name for path in (run / "predictions" / arm).glob("*.json")}
        expected_predictions = {source_id + ".json" for source_id in expected_ids}
        if actual_predictions != expected_predictions:
            errors.append(
                f"prediction_{arm}_file_set_mismatch:missing={len(expected_predictions - actual_predictions)},"
                f"extra={len(actual_predictions - expected_predictions)}"
            )
    unexpected_evidence = {
        path.stem for path in (run / "evidence_float32").glob("*.npz")
    } - set(expected_ids)
    if unexpected_evidence:
        errors.append(f"evidence_extra_source_ids:{len(unexpected_evidence)}")
    receipt_by_id = {row["source_id"]: row for row in execution["rows"]}
    if len(receipt_by_id) != len(expected_ids):
        errors.append("run_receipt_duplicate_source_id")
    selected = sample_ids(expected_ids, receipt_by_id, args.redraw_samples)
    rows: list[dict] = []
    for source_id in expected_ids:
        if source_id not in input_by_id or source_id not in receipt_by_id:
            rows.append({"source_id": source_id, "errors": ["missing_input_or_run_receipt"]})
            continue
        rows.append(audit_one(input_by_id[source_id], receipt_by_id[source_id], run, gallery, config, config_sha, source_id in selected))
    error_rows = [{"source_id": row["source_id"], "errors": row["errors"]} for row in rows if row["errors"]]
    result = {
        "schema_version": "geometry_direction_audit_v1",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "run_directory": str(run),
        "inputs_path": str(args.inputs.resolve()),
        "inputs_sha256": file_sha256(args.inputs),
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "current_runner_sha256": runner_sha,
        "run_runner_sha256": execution.get("script_sha256"),
        "expected_count": len(expected_ids),
        "png_count": len(actual_files),
        "prediction_G0_count": len(list((run / "predictions" / "G0").glob("*.json"))),
        "prediction_G1_count": len(list((run / "predictions" / "G1").glob("*.json"))),
        "evidence_count": len(list((run / "evidence_float32").glob("*.npz"))),
        "redraw_sample_ids": selected,
        "redraw_exact_matches": sum(row.get("redraw_exact_match") is True for row in rows),
        "run_failure_count": sum(bool(row.get("failed_in_run")) for row in rows),
        "status_G1_counts": dict(Counter(row.get("status_G1") for row in rows)),
        "global_errors": errors,
        "warnings": warnings,
        "image_errors": error_rows,
        "checked_count": len(rows),
        "passed_count": sum(not row["errors"] for row in rows),
        "exit_code": int(bool(errors or error_rows)),
    }
    if args.read_only:
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        output = run / "audit_receipt.json"
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"output": str(output), "checked": len(rows), "passed": result["passed_count"],
                          "exit_code": result["exit_code"]}, ensure_ascii=False))
    return result["exit_code"]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
