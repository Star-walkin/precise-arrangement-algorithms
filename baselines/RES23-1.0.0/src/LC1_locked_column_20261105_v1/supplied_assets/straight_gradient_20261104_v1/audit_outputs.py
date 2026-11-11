"""Independent, read-only audit of the E1-conditioned straight-gradient gallery.

This checks the frozen source manifest, every full-resolution four-panel PNG,
and every machine-readable prediction.  It never changes a prediction and
does not load labels, neural outputs, held-out media, or device modules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs.json"
DEFAULT_RUN = Path(r"C:\precise_arrangement_results\straight_gradient_20261104_v1\all_001")
EXPECTED_COUNT = 292
EXPECTED_HW = (3000, 4096)
PANEL_NAMES = ("original", "blurred", "gradient", "overlay")
Image.MAX_IMAGE_PIXELS = 300_000_000


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_file_set(directory: Path, extension: str, ids: list[str], name: str,
                   errors: list[str]) -> int:
    expected = {source_id + extension for source_id in ids}
    found = {path.name for path in directory.glob("*" + extension)} if directory.is_dir() else set()
    if found != expected:
        errors.append(f"{name}_file_set_mismatch:missing={len(expected - found)}:extra={len(found - expected)}")
    return len(found)


def valid_xy(point: object, width: int, height: int) -> bool:
    return (isinstance(point, (list, tuple)) and len(point) == 2
            and all(isinstance(value, (int, float)) and math.isfinite(value) for value in point)
            and -0.5 <= point[0] < width + 0.5 and -0.5 <= point[1] < height + 0.5)


def validate_prediction(pred: dict, row: dict, config_sha256: str, errors: list[str]) -> None:
    source_id = row["source_id"]
    for field in ("source_id", "id"):
        if field in pred and pred[field] != source_id:
            errors.append(f"prediction_{field}_mismatch")
    if "source_id" not in pred and "id" not in pred:
        errors.append("prediction_source_id_absent")
    for field, expected in (("source_path", row["source_path"]),
                            ("image_path", row["source_path"]),
                            ("source_sha256", row["source_sha256"]),
                            ("image_sha256", row["source_sha256"])):
        if field in pred and pred[field] != expected:
            errors.append(f"prediction_{field}_mismatch")
    expected_e1 = row["e1"]["xy"]
    if pred.get("e1") != row["e1"]:
        errors.append("prediction_e1_mismatch")
    if pred.get("config_sha256") != config_sha256:
        errors.append("prediction_config_sha256_mismatch")
    if pred.get("entry_is_input_condition") is not True:
        errors.append("prediction_entry_semantics_invalid")
    for field in ("e1_xy", "input_seed_xy", "seed_xy"):
        if field in pred:
            try:
                match = np.allclose(np.asarray(pred[field], float), np.asarray(expected_e1, float),
                                    rtol=0, atol=0.001)
            except (ValueError, TypeError):
                match = False
            if not match:
                errors.append(f"prediction_{field}_mismatch")
    if pred.get("network_response_used") is True or pred.get("complete_identity_confirmed") is True:
        errors.append("prediction_semantics_invalid")
    if pred.get("seed_is_input_not_detection") is not True:
        errors.append("prediction_seed_semantics_invalid")
    if pred.get("physical_contact") is not None or pred.get("same_material_identity") is not None:
        errors.append("prediction_unwarranted_physical_claim")
    if not isinstance(pred.get("status"), str):
        errors.append("prediction_status_missing")
    if not isinstance(pred.get("stop_reason"), str):
        errors.append("prediction_stop_reason_missing")

    height, width = row["shape_hw"]
    # Supported polyline and endpoint names are intentionally simple so the
    # independent audit can accept a frozen runner's JSON without importing it.
    checked = False
    for field in ("points_xy", "path_xy", "centerline_xy", "segment_xy", "straight_points_xy"):
        if field not in pred:
            continue
        points = pred[field]
        if not isinstance(points, list) or not all(valid_xy(p, width, height) for p in points):
            errors.append(f"prediction_{field}_invalid")
        checked = True
    for field in ("end_xy", "end_point_xy", "straight_end_xy", "endpoint_xy"):
        if field in pred and pred[field] is not None:
            if not valid_xy(pred[field], width, height):
                errors.append(f"prediction_{field}_invalid")
            checked = True
    if not checked and pred.get("status") not in ("single_image_failure", "invalid_e1", "no_supported_straight"):
        errors.append("prediction_no_auditable_geometry")
    segments = pred.get("supported_segments_xy")
    if not isinstance(segments, list) or not all(
        isinstance(segment, list) and segment and all(valid_xy(p, width, height) for p in segment)
        for segment in segments
    ):
        errors.append("prediction_supported_segments_invalid")
    records = pred.get("step_records")
    if not isinstance(records, list):
        errors.append("prediction_step_records_missing")
    else:
        distances = [item.get("distance_after_E1_px") for item in records if isinstance(item, dict)]
        if len(distances) != len(records) or any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            for value in distances
        ) or any(a >= b for a, b in zip(distances, distances[1:])):
            errors.append("prediction_step_distances_invalid")
        for item in records:
            if not isinstance(item, dict) or not valid_xy(item.get("axis_xy"), width, height):
                errors.append("prediction_step_axis_invalid")
                break
    length = pred.get("supported_extension_px")
    if not isinstance(length, (int, float)) or not math.isfinite(length) or length < 0:
        errors.append("prediction_supported_extension_invalid")
    elif pred.get("status") == "PARTIAL":
        if length <= 0 or len(pred.get("points_xy", [])) < 2 or pred.get("endpoint_xy") is None:
            errors.append("prediction_partial_geometry_missing")
        elif not np.allclose(pred["endpoint_xy"], pred["points_xy"][-1], atol=1e-4, rtol=0):
            errors.append("prediction_endpoint_not_last_supported_point")
        if (isinstance(segments, list) and all(isinstance(segment, list) for segment in segments)
                and isinstance(pred.get("points_xy"), list) and len(pred["points_xy"]) >= 2):
            flat = [point for segment in segments for point in segment]
            if not np.array_equal(np.asarray(flat), np.asarray(pred["points_xy"][1:])):
                errors.append("prediction_supported_segments_do_not_partition_points")
            if not np.allclose(pred["points_xy"][0], expected_e1[-1], atol=0.001, rtol=0):
                errors.append("prediction_first_point_not_E1_end")
            if pred.get("supported_sample_count") != len(flat):
                errors.append("prediction_supported_sample_count_mismatch")
        if isinstance(records, list) and records and all(isinstance(item, dict) for item in records):
            supported_distances = [item["distance_after_E1_px"] for item in records if item.get("supported")]
            if supported_distances and abs(max(supported_distances) - length) > 1e-4:
                errors.append("prediction_extension_not_last_supported_distance")
    elif pred.get("status") == "UNKNOWN":
        if length != 0 or pred.get("points_xy") != [] or pred.get("endpoint_xy") is not None:
            errors.append("prediction_unknown_contains_claimed_straight")


def parse_layout(run_receipt: dict, errors: list[str]) -> tuple[tuple[int, int], dict[str, tuple[int, int, int, int]]]:
    layout = run_receipt.get("layout", {})
    board_size = layout.get("board_size_wh")
    rects = layout.get("panel_rects_xyxy")
    if not isinstance(board_size, list) or len(board_size) != 2 or not all(isinstance(v, int) for v in board_size):
        errors.append("run_layout_board_size_missing")
        return (0, 0), {}
    if not isinstance(rects, dict) or set(rects) != set(PANEL_NAMES):
        errors.append("run_layout_panel_rects_missing")
        return tuple(board_size), {}
    parsed = {}
    width, height = board_size
    for name in PANEL_NAMES:
        rect = rects[name]
        if (not isinstance(rect, list) or len(rect) != 4 or not all(isinstance(v, int) for v in rect)):
            errors.append(f"run_layout_{name}_invalid")
            continue
        x0, y0, x1, y1 = rect
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            errors.append(f"run_layout_{name}_out_of_board")
        if (y1 - y0, x1 - x0) != EXPECTED_HW:
            errors.append(f"run_layout_{name}_not_original_resolution")
        parsed[name] = tuple(rect)
    for first in PANEL_NAMES:
        for second in PANEL_NAMES:
            if first >= second or first not in parsed or second not in parsed:
                continue
            a, b = parsed[first], parsed[second]
            if max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3]):
                errors.append(f"run_layout_panels_overlap:{first}:{second}")
    return tuple(board_size), parsed


def audit_one(row: dict, run_row: dict | None, run: Path, gallery_dir: str,
              config_sha256: str,
              board_size: tuple[int, int], rects: dict[str, tuple[int, int, int, int]]) -> dict:
    source_id = row["source_id"]
    errors: list[str] = []
    info: dict = {"source_id": source_id, "errors": errors}
    source = Path(row["source_path"])
    try:
        if sha256(source) != row["source_sha256"]:
            errors.append("source_sha256_mismatch")
        with Image.open(source) as opened:
            opened.load()
            if opened.size != (EXPECTED_HW[1], EXPECTED_HW[0]):
                errors.append("source_size_mismatch")
            source_rgb = np.asarray(opened.convert("RGB"))
    except Exception as exc:
        errors.append(f"source_error:{type(exc).__name__}:{exc}")
        source_rgb = None

    png = run / gallery_dir / (source_id + ".png")
    try:
        if run_row is not None:
            if run_row.get("source_id") != source_id:
                errors.append("run_row_source_id_mismatch")
            if run_row.get("source_sha256") not in (None, row["source_sha256"]):
                errors.append("run_row_source_sha256_mismatch")
            if "png_path" in run_row and Path(run_row["png_path"]) != png:
                errors.append("run_row_png_path_mismatch")
            if "png_sha256" in run_row and sha256(png) != run_row["png_sha256"]:
                errors.append("png_sha256_mismatch")
        with Image.open(png) as opened:
            opened.load()
            if opened.format != "PNG" or opened.mode != "RGB":
                errors.append("png_format_or_mode_mismatch")
            if opened.size != board_size:
                errors.append("png_board_size_mismatch")
            if source_rgb is not None and "original" in rects:
                original = np.asarray(opened.crop(rects["original"]))
                if not np.array_equal(original, source_rgb):
                    errors.append("original_panel_pixels_differ_from_source")
            for panel_name in ("blurred", "gradient", "overlay"):
                if panel_name not in rects:
                    continue
                panel = np.asarray(opened.crop(rects[panel_name]))
                if panel.shape != (EXPECTED_HW[0], EXPECTED_HW[1], 3):
                    errors.append(f"{panel_name}_panel_shape_mismatch")
                elif np.max(panel) == np.min(panel):
                    errors.append(f"{panel_name}_panel_is_constant")
                else:
                    info[panel_name + "_std"] = float(np.std(panel[::32, ::32]))
                    if source_rgb is not None and not np.any(panel != source_rgb):
                        errors.append(f"{panel_name}_panel_identical_to_original")
                    if panel_name == "gradient" and not (
                        np.array_equal(panel[:, :, 0], panel[:, :, 1])
                        and np.array_equal(panel[:, :, 0], panel[:, :, 2])
                    ):
                        errors.append("gradient_panel_not_grayscale")
    except Exception as exc:
        errors.append(f"png_error:{type(exc).__name__}:{exc}")

    pred_path = run / "predictions" / (source_id + ".json")
    try:
        if run_row is not None and run_row.get("pred_path") != str(pred_path):
            errors.append("run_row_pred_path_mismatch")
        if run_row is not None and run_row.get("pred_sha256") != sha256(pred_path):
            errors.append("prediction_sha256_mismatch")
        pred = read_json(pred_path)
        validate_prediction(pred, row, config_sha256, errors)
        info["status"] = pred.get("status")
        if run_row is not None and "status" in run_row and run_row["status"] != pred.get("status"):
            errors.append("run_row_prediction_status_mismatch")
    except Exception as exc:
        errors.append(f"prediction_error:{type(exc).__name__}:{exc}")
    info["failed_in_run"] = bool(run_row and run_row.get("failure"))
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--inputs", type=Path, default=INPUTS)
    parser.add_argument("--config", type=Path, help="Config used for the run; inferred from run_start.argv by default")
    parser.add_argument("--read-only", action="store_true", help="Do not write audit_receipt.json")
    args = parser.parse_args()
    run = args.run.resolve()
    inputs = read_json(args.inputs)
    execution = read_json(run / "run_receipt.json")
    errors: list[str] = []
    warnings: list[str] = []
    rows = inputs.get("rows", [])
    ids = [row["source_id"] for row in rows]
    if len(ids) != EXPECTED_COUNT or len(set(ids)) != EXPECTED_COUNT or inputs.get("source_count") != EXPECTED_COUNT:
        errors.append("input_manifest_not_292_unique")
    if inputs.get("e1_valid_count") != EXPECTED_COUNT:
        errors.append("input_manifest_e1_not_292_valid")
    for row in rows:
        if row.get("shape_hw") != list(EXPECTED_HW) or row.get("independent_blind_test") is not False:
            errors.append("input_manifest_shape_or_use_invalid")
            break
        if row.get("e1", {}).get("valid") is not True or len(row.get("e1", {}).get("xy", [])) != 2:
            errors.append("input_manifest_invalid_e1")
            break
    if inputs.get("development_evaluation_policy", {}).get("reserved_media_and_labels_access") is not False:
        errors.append("input_manifest_reserved_policy_invalid")
    start = execution.get("start", {})
    phase = execution.get("phase")
    if phase not in ("all", "pilot") or start.get("phase") != phase:
        errors.append("run_receipt_phase_invalid")
    if phase == "pilot":
        ids = [item.get("source_id") for item in execution.get("rows", [])]
        lookup = {row["source_id"]: row for row in rows}
        if not ids or len(set(ids)) != len(ids) or any(source_id not in lookup for source_id in ids):
            errors.append("pilot_ids_invalid")
        rows = [lookup[source_id] for source_id in ids if source_id in lookup]
        gallery_dir = "pilot_png"
    else:
        gallery_dir = "final_292"
    if execution.get("input_count") != len(ids) or execution.get("completed_rows") != len(ids):
        errors.append("run_receipt_row_count_mismatch")
    if start.get("input_count") != len(ids):
        errors.append("run_start_count_mismatch")
    if start.get("source_index_sha256") != sha256(args.inputs):
        errors.append("run_receipt_inputs_sha256_mismatch")
    if args.config is not None:
        config_path = args.config.resolve()
    else:
        command = start.get("argv", [])
        if "--config" in command:
            entered = Path(command[command.index("--config") + 1])
            config_path = entered if entered.is_absolute() else HERE.parents[2] / entered
        else:
            config_path = HERE / "config.json"
    if start.get("config_sha256") != sha256(config_path):
        errors.append("run_config_sha256_mismatch")
    if start.get("core_sha256") != sha256(HERE / "gradient_core.py"):
        (warnings if phase == "pilot" else errors).append("current_core_differs_from_run_record")
    if start.get("runner_sha256") != sha256(HERE / "run_gradient.py"):
        (warnings if phase == "pilot" else errors).append("current_runner_differs_from_run_record")
    if start.get("network_response_used") is not False:
        errors.append("run_receipt_network_flag_invalid")
    board_size, rects = parse_layout(start, errors)
    png_count = check_file_set(run / gallery_dir, ".png", ids, "png", errors)
    pred_count = check_file_set(run / "predictions", ".json", ids, "predictions", errors)
    run_rows = execution.get("rows", [])
    if not isinstance(run_rows, list):
        run_rows = []
        errors.append("run_receipt_rows_missing")
    by_id = {item.get("source_id"): item for item in run_rows if isinstance(item, dict)}
    if len(run_rows) != len(ids) or len(by_id) != len(ids):
        errors.append("run_receipt_ids_not_unique")
    audited = [audit_one(row, by_id.get(row["source_id"]), run, gallery_dir,
                         start.get("config_sha256", ""), board_size, rects) for row in rows]
    bad = [{"source_id": item["source_id"], "errors": item["errors"]} for item in audited if item["errors"]]
    result = {
        "schema_version": "straight_gradient_audit_v1",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "run_directory": str(run),
        "inputs_path": str(args.inputs.resolve()),
        "inputs_sha256": sha256(args.inputs),
        "expected_count": len(ids),
        "png_count": png_count,
        "prediction_count": pred_count,
        "checked_count": len(audited),
        "passed_count": sum(not item["errors"] for item in audited),
        "status_counts": dict(Counter(item.get("status") for item in audited)),
        "run_failure_count": sum(item["failed_in_run"] for item in audited),
        "global_errors": errors,
        "warnings": warnings,
        "image_errors": bad,
        "exit_code": int(bool(errors or bad)),
    }
    if args.read_only:
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        path = run / "audit_receipt.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"path": str(path), "passed": result["passed_count"],
                          "exit_code": result["exit_code"]}, ensure_ascii=False))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
