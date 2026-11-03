"""D3独立结构/像素审计；只读现有预测，绝不评价边界真实准确率。

用法：python audit.py --run pilot_001 --output pilot_001/independent_audit.json
所有文件由固定400清单显式构造，不递归跟随junction或扫描保留集。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import struct
import sys

import cv2
import numpy as np

from data_io import HERE, D2, D1, C3, DATA, old, selection, read_json, read_image


def digest(path: Path) -> str:
    hashed = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hashed.update(block)
    return hashed.hexdigest()


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if (len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or
            header[12:16] != b"IHDR"):
        raise ValueError("missing PNG IHDR")
    return struct.unpack(">II", header[16:24])


def in_project(path: Path) -> bool:
    try:
        return path.resolve().is_relative_to(HERE.parents[2].resolve())
    except (OSError, RuntimeError):
        return False


def issue(errors: list[dict], code: str, detail: object = None) -> None:
    errors.append({"code": code, "detail": detail})


def binary_mask(mask: np.ndarray | None, shape: tuple[int, int],
                errors: list[dict], name: str) -> bool:
    if mask is None or mask.shape != shape or mask.dtype != np.uint8:
        issue(errors, "mask_shape_or_dtype", name)
        return False
    if np.any((mask != 0) & (mask != 255)):
        issue(errors, "mask_not_binary", name)
        return False
    return True


def expected_deps(role: str, stem: str, name: str) -> list[Path]:
    c4 = old.C4
    return [
        c4 / "roi/surface_masks" / role / name,
        C3 / "boundaries/records" / role / (stem + ".json"),
        c4 / "boundaries/records" / role / (stem + ".json"),
        D1 / "all_001/记录/masks/dark_roi" / role / name,
        D1 / "all_001/图像结果/02_对比度增强ROI" / role / name,
        D1 / "all_001/记录/cases" / role / (stem + ".json"),
        D2 / "all_002/记录/cases" / role / (stem + ".json"),
    ]


def audit_ledger(previous: dict, current: dict, errors: list[dict]) -> None:
    old_lines = previous.get("observed_lines", [])
    new_lines = current.get("observed_lines", [])
    if new_lines != old_lines:
        issue(errors, "old_observed_lines_changed", {
            "old_lines": len(old_lines), "new_lines": len(new_lines)})
    expected: dict[tuple[str, int], tuple[list[int], str | None]] = {}
    for line in old_lines:
        line_id = str(line.get("line_id"))
        for i, xy in enumerate(line.get("observed_points_xy", [])):
            key = (line_id, i)
            if key in expected:
                issue(errors, "duplicate_old_point_reference", [line_id, i])
            expected[key] = ([int(xy[0]), int(xy[1])],
                             line.get("source_c3_group_id"))
    assignments = current.get("point_assignments", [])
    seen: Counter = Counter()
    paths = {group.get("path_id"): group for group in current.get("new_groups", [])}
    if len(paths) != len(current.get("new_groups", [])):
        issue(errors, "duplicate_new_path_id")
    assigned_refs = Counter()
    for entry in assignments:
        key = (str(entry.get("old_line_id")), entry.get("point_index"))
        seen[key] += 1
        if key not in expected:
            issue(errors, "invented_assignment_reference", list(key))
            continue
        xy, old_group = expected[key]
        if entry.get("xy") != xy or entry.get("old_c3_group_id") != old_group:
            issue(errors, "assignment_changed_old_point", list(key))
        if entry.get("status") == "assigned":
            path_id = entry.get("new_path_id")
            if path_id not in paths or entry.get("new_c3_group_id") != paths[path_id].get("c3_group_id"):
                issue(errors, "assigned_path_missing_or_group_wrong", list(key))
            assigned_refs[key] += 1
        elif entry.get("status") == "unassigned":
            if entry.get("new_path_id") is not None or not entry.get("reason"):
                issue(errors, "unassigned_point_contract", list(key))
        else:
            issue(errors, "invalid_assignment_status", [*key, entry.get("status")])
    for key in expected:
        if seen[key] != 1:
            issue(errors, "old_point_assignment_multiplicity", [*key, seen[key]])
    group_refs = Counter()
    for group in paths.values():
        for point in group.get("assigned_points", []):
            key = (str(point.get("old_line_id")), point.get("point_index"))
            group_refs[key] += 1
            if key not in expected or point.get("xy") != expected[key][0]:
                issue(errors, "new_group_changed_point", list(key))
    if group_refs != assigned_refs:
        issue(errors, "group_assignment_ledger_diff", {
            "assigned_in_ledger": sum(assigned_refs.values()),
            "assigned_in_groups": sum(group_refs.values()),
            "mismatched_refs": len(set(group_refs) ^ set(assigned_refs))})


def audit_curves(case: dict, roi: np.ndarray, masks: dict[str, np.ndarray],
                 pitch: float, errors: list[dict]) -> None:
    curves = case.get("fit", {}).get("curves", [])
    rank = {g["path_id"]: i for i, g in enumerate(case.get("new_groups", []))}
    seen_ids = set()
    from_curves_smooth = np.zeros(roi.shape, np.uint8)
    from_curves_inferred = np.zeros(roi.shape, np.uint8)
    per_x: dict[int, dict[str, int]] = defaultdict(dict)
    gap = math.ceil(.5 * pitch)
    for curve in curves:
        curve_id, path_id = curve.get("curve_id"), curve.get("path_id")
        if curve_id in seen_ids:
            issue(errors, "duplicate_curve_id", curve_id)
        seen_ids.add(curve_id)
        if path_id not in rank or curve.get("side") not in ("left", "right"):
            issue(errors, "curve_path_or_side_invalid", curve_id)
        points = curve.get("sampled_points_xy", [])
        states = curve.get("point_states", [])
        if len(points) != len(states):
            issue(errors, "curve_state_length", curve_id)
            continue
        previous = None
        inference_run = 0
        for xy, state in zip(points, states):
            if (not isinstance(xy, list) or len(xy) != 2 or
                    not all(isinstance(v, (int, float)) and math.isfinite(v) for v in xy)):
                issue(errors, "curve_nonfinite_or_nonxy", curve_id)
                continue
            x, y = xy
            if int(x) != x or int(y) != y:
                issue(errors, "curve_not_integer_original_pixel", [curve_id, xy])
                continue
            x, y = int(x), int(y)
            if not (0 <= x < roi.shape[1] and 0 <= y < roi.shape[0] and roi[y, x]):
                issue(errors, "curve_outside_dark_roi", [curve_id, x, y])
                continue
            if state not in ("supported_fit", "inferred_short_gap"):
                issue(errors, "invalid_point_state", [curve_id, state])
            if previous is not None and (x != previous[0] + 1 or abs(y - previous[1]) > 4):
                issue(errors, "curve_discontinuous_segment", [curve_id, previous, [x, y]])
            previous = [x, y]
            cv2.circle(from_curves_smooth, (x, y), 1, 255, -1, cv2.LINE_8)
            if state == "inferred_short_gap":
                inference_run += 1
                cv2.circle(from_curves_inferred, (x, y), 1, 255, -1, cv2.LINE_8)
            else:
                if inference_run > math.ceil(3 * pitch) + 1:
                    issue(errors, "inferred_gap_too_long", [curve_id, inference_run])
                inference_run = 0
            if path_id in rank:
                if path_id in per_x[x]:
                    issue(errors, "same_path_duplicate_x", [path_id, x])
                per_x[x][path_id] = y
        if inference_run > math.ceil(3 * pitch) + 1:
            issue(errors, "inferred_gap_too_long", [curve_id, inference_run])
    for x, ys in per_x.items():
        ordered = sorted(ys.items(), key=lambda item: rank[item[0]])
        for (left_id, left_y), (right_id, right_y) in zip(ordered, ordered[1:]):
            if right_y - left_y < gap:
                issue(errors, "ordered_curve_intersection_or_close_pair", {
                    "x": x, "upper": left_id, "lower": right_id,
                    "distance_px": right_y - left_y, "required_px": gap})
    from_curves_smooth[roi == 0] = 0
    from_curves_inferred[roi == 0] = 0
    from_curves_inferred[masks["observed"] > 0] = 0
    if not np.array_equal(from_curves_smooth, masks["smooth"]):
        issue(errors, "smooth_mask_records_raster_mismatch", int(np.count_nonzero(
            from_curves_smooth != masks["smooth"])))
    if not np.array_equal(from_curves_inferred, masks["inferred"]):
        issue(errors, "inferred_mask_records_raster_mismatch", int(np.count_nonzero(
            from_curves_inferred != masks["inferred"])))


def audit_one(row: dict, run: Path, d2_inventory: dict[str, str],
              mode: str) -> dict:
    errors: list[dict] = []
    rel = Path(row["output_file"])
    role, name, stem = rel.parent.name, rel.name, rel.stem
    if (rel.parts[0] != "images" or role not in ("left_camera", "right_camera") or
            ".." in rel.parts):
        return {"output_file": row["output_file"], "status": "failed",
                "errors": [{"code": "invalid_manifest_relative_path"}]}
    source_png = DATA / rel
    case_path = run / "记录/cases" / role / (stem + ".json")
    old_path = D2 / "all_002/记录/cases" / role / (stem + ".json")
    roi_path = D1 / "all_001/记录/masks/dark_roi" / role / name
    mask_paths = {key: run / "记录/masks" / key / role / name
                  for key in ("observed", "smooth", "inferred")}
    overlay_path = run / "图像结果" / role / name
    expected_paths = [case_path, overlay_path, *mask_paths.values()]
    if mode == "pilot":
        expected_paths += [run / "记录" / key / role / name
                           for key in ("field_score", "field_support")]
    for path in expected_paths:
        if not path.is_file():
            issue(errors, "missing_output", str(path))
    if not case_path.is_file():
        return {"output_file": row["output_file"], "status": "failed", "errors": errors}
    try:
        case = read_json(case_path)
        previous = read_json(old_path)
        old_rel = (Path("记录/cases") / role / (stem + ".json")).as_posix()
        if digest(old_path) != d2_inventory.get(old_rel):
            issue(errors, "frozen_d2_record_manifest_hash_mismatch", old_rel)
        if digest(source_png) != row["output_sha256"]:
            issue(errors, "original_png_hash_mismatch", row["output_file"])
        if png_dimensions(source_png) != (4096, 3000):
            issue(errors, "original_png_dimensions", row["output_file"])
    except Exception as exc:
        issue(errors, "input_or_json_unreadable", repr(exc))
        return {"output_file": row["output_file"], "status": "failed", "errors": errors}
    rec = case.get("record", {})
    if (rec.get("status") != "completed" or rec.get("output_file") != row["output_file"] or
            rec.get("source_png_sha256") != row["output_sha256"]):
        issue(errors, "case_record_identity_or_status", rec.get("status"))
    deps = rec.get("input_dependency_sha256", {})
    expected = {str(p): p for p in expected_deps(role, stem, name)}
    if set(deps) != set(expected) or len(deps) != 7:
        issue(errors, "dependency_key_set", {"actual": len(deps), "expected": 7})
    for key, path in expected.items():
        try:
            if deps.get(key) != digest(path):
                issue(errors, "dependency_hash_mismatch", key)
        except (OSError, ValueError) as exc:
            issue(errors, "dependency_missing_or_unreadable", [key, repr(exc)])
    audit_ledger(previous, case, errors)
    try:
        roi = read_image(roi_path, 0)
        masks = {key: read_image(path, 0) for key, path in mask_paths.items()}
        overlay = read_image(overlay_path)
        old_observed = read_image(D2 / "all_002/记录/masks/observed" / role / name, 0)
        shape = (3000, 4096)
        if not binary_mask(roi, shape, errors, "frozen_dark_roi"):
            raise ValueError("invalid dark ROI")
        for key, array in masks.items():
            if not binary_mask(array, shape, errors, key):
                raise ValueError("invalid prediction mask")
            if np.any(array[roi == 0]):
                issue(errors, "mask_outside_dark_roi", key)
        if not np.array_equal(masks["observed"], old_observed):
            issue(errors, "old_observed_mask_changed", int(np.count_nonzero(
                masks["observed"] != old_observed)))
        expected_observed = np.zeros(shape, np.uint8)
        for line in case.get("observed_lines", []):
            for x, y in line.get("observed_points_xy", []):
                cv2.circle(expected_observed, (int(x), int(y)), 1, 255, -1, cv2.LINE_8)
        expected_observed[roi == 0] = 0
        if not np.array_equal(expected_observed, masks["observed"]):
            issue(errors, "observed_record_raster_mismatch", int(np.count_nonzero(
                expected_observed != masks["observed"])))
        if np.any((masks["inferred"] > 0) & (masks["observed"] > 0)):
            issue(errors, "inferred_observed_overlap")
        if overlay is None or overlay.shape != (3000, 4096, 3):
            issue(errors, "overlay_dimensions")
        elif np.any(overlay[roi == 0]):
            issue(errors, "overlay_nonblack_outside_roi")
        audit_curves(case, roi, masks, float(rec.get("pitch_px", 0)), errors)
        for key in ("observed", "smooth", "inferred"):
            count = int(np.count_nonzero(masks[key]))
            if rec.get(key + "_pixels") != count:
                issue(errors, "record_mask_pixel_count", [key, rec.get(key + "_pixels"), count])
        if rec.get("curve_count") != len(case.get("fit", {}).get("curves", [])):
            issue(errors, "record_curve_count")
        count = sum(len(line.get("observed_points_xy", []))
                    for line in case.get("observed_lines", []))
        if rec.get("observed_point_count") != count:
            issue(errors, "record_observed_point_count")
        if mode == "pilot":
            field_score = read_image(run / "记录/field_score" / role / name, 0)
            field_support = read_image(run / "记录/field_support" / role / name, 0)
            if field_score is None or field_score.shape != shape or np.any(field_score[roi == 0]):
                issue(errors, "field_score_raster_contract")
            if binary_mask(field_support, shape, errors, "field_support") and np.any(field_support[roi == 0]):
                issue(errors, "field_support_outside_roi")
    except Exception as exc:
        issue(errors, "output_decode_or_pixel_audit_error", repr(exc))
    return {"output_file": row["output_file"],
            "status": "passed" if not errors else "failed", "errors": errors}


def audit_run(run: Path) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    global_errors: list[dict] = []
    context = read_json(run / "run_context.json")
    summary = read_json(run / "summary.json")
    mode = context.get("mode")
    if mode not in ("pilot", "all"):
        issue(global_errors, "run_mode_invalid", mode)
        mode = "all"
    rows, _ = selection(mode)
    expected_selection = [row["output_file"] for row in rows]
    if context.get("selection") != expected_selection:
        issue(global_errors, "fixed_selection_mismatch", {
            "expected": len(expected_selection),
            "recorded": len(context.get("selection", []))})
    if (summary.get("planned") != len(rows) or summary.get("failed") != 0 or
            summary.get("completed") != len(rows) or summary.get("exit_code") != 0 or
            summary.get("source_unchanged") is not True):
        issue(global_errors, "run_summary_not_successful", summary)
    protocol = read_json(HERE / "protocol.json")
    d2_manifest = D2 / "all_002/output_manifest.json"
    if digest(d2_manifest) != protocol.get("d2_output_manifest_sha256"):
        issue(global_errors, "d2_output_manifest_hash_mismatch")
    d2_data = read_json(d2_manifest)
    d2_inventory = {item["path"]: item["sha256"]
                    for case in d2_data["cases"] for item in case["files"]}
    if d2_data.get("case_count") != 400:
        issue(global_errors, "d2_manifest_case_count")
    d1_protocol = read_json(D1 / "protocol.json")
    if digest(DATA / "manifest.csv") != d1_protocol.get("input_manifest_sha256"):
        issue(global_errors, "original_manifest_hash_mismatch")
    if digest(D1 / "all_001/output_manifest.json") != read_json(D2 / "protocol.json").get("d1_output_manifest_sha256"):
        issue(global_errors, "d1_output_manifest_hash_mismatch")
    frozen = protocol.get("frozen_d2_source_sha256", {})
    for source, expected_hash in frozen.items():
        if digest(Path(source)) != expected_hash:
            issue(global_errors, "frozen_d2_source_hash_mismatch", source)
    source = context.get("source_sha256", {})
    snapshots = context.get("source_snapshot_paths", {})
    if set(source) != set(snapshots):
        issue(global_errors, "source_snapshot_mapping_keys")
    for original, expected_hash in source.items():
        relative = Path(snapshots.get(original, ""))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            issue(global_errors, "invalid_snapshot_relative_path", original)
            continue
        snapshot = run / relative
        if not snapshot.is_file() or digest(snapshot) != expected_hash:
            issue(global_errors, "source_snapshot_hash_mismatch", original)
    if mode == "all":
        frozen_run = read_json(HERE / "freeze_all.json")
        if source != frozen_run.get("source_sha256"):
            issue(global_errors, "full_run_source_not_frozen")
    log_lines = (run / "run_log.jsonl").read_text(encoding="utf-8").splitlines()
    if len(log_lines) != len(rows):
        issue(global_errors, "run_log_denominator", [len(log_lines), len(rows)])
    else:
        for i, (line, row) in enumerate(zip(log_lines, rows)):
            entry = json.loads(line)
            if entry.get("output_file") != row["output_file"] or entry.get("status") != "completed":
                issue(global_errors, "run_log_status_or_order", i)
    cases = []
    for i, row in enumerate(rows, 1):
        cases.append(audit_one(row, run, d2_inventory, mode))
        if i % 25 == 0 or i == len(rows):
            print(json.dumps({"audited": i, "planned": len(rows),
                "failed": sum(c["status"] == "failed" for c in cases)},
                ensure_ascii=False), flush=True)
    passed = sum(case["status"] == "passed" for case in cases)
    return {"audit_name": "D3 independent structural and pixel audit",
            "run": str(run), "mode": mode, "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "planned": len(rows), "passed": passed,
            "failed": len(rows)-passed, "global_errors": global_errors,
            "cases": cases, "ground_truth_accuracy": None,
            "meaning": "hashes, fixed yellow ledger, masks, geometry and ROI only; no visual or physical accuracy claim"}


def immutable_output_manifest(run: Path, rows: list[dict]) -> dict:
    """仅五项逐图预测产物；排除审计、QA、诊断图和后生成报告。"""
    cases = []
    for row in rows:
        rel = Path(row["output_file"])
        role, stem, name = rel.parent.name, rel.stem, rel.name
        parts = [Path("图像结果") / role / name,
                 *(Path("记录/masks") / key / role / name
                   for key in ("observed", "smooth", "inferred")),
                 Path("记录/cases") / role / (stem + ".json")]
        files = []
        for part in parts:
            path = run / part
            files.append({"path": part.as_posix(), "size": path.stat().st_size,
                          "sha256": digest(path)})
        record = read_json(run / parts[-1])["record"]
        cases.append({"output_file": row["output_file"], "files": files,
                      "observed_points": record["observed_point_count"],
                      "curves": record["curve_count"]})
    return {"version": read_json(HERE / "protocol.json")["version"],
            "run": str(run), "planned": len(rows), "cases": cases,
            "case_count": len(cases), "file_count": sum(len(c["files"]) for c in cases),
            "incomplete_cases": [],
            "scope": "only original five immutable outputs per case; excludes generated audit, QA and diagnostics"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output.absolute()
    if run.parent != HERE.resolve() or not run.is_dir():
        raise ValueError("--run must be an existing direct child of D3 root")
    if run.is_symlink() or (hasattr(os.path, "isjunction") and os.path.isjunction(run)):
        raise ValueError("run must not be a junction or symlink")
    if output.parent.resolve() != run or output.exists():
        raise ValueError("--output must be a new JSON directly inside --run")
    cv2.setNumThreads(1)
    report = audit_run(run)
    if report["failed"] == 0 and not report["global_errors"]:
        rows, _ = selection(report["mode"])
        manifest = immutable_output_manifest(run, rows)
        manifest_path = run / "output_manifest.json"
        if manifest_path.exists():
            if read_json(manifest_path) != manifest:
                issue(report["global_errors"], "existing_output_manifest_differs")
        else:
            with manifest_path.open("x", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2,
                          allow_nan=False)
                handle.write("\n")
        if manifest_path.exists():
            report["output_manifest"] = {"path": str(manifest_path),
                "sha256": digest(manifest_path),
                "case_count": manifest["case_count"],
                "file_count": manifest["file_count"]}
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    result = {key: report[key] for key in ("run", "planned", "passed", "failed")}
    result["global_error_count"] = len(report["global_errors"])
    result["report"] = str(output)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return int(report["failed"] > 0 or bool(report["global_errors"]))


if __name__ == "__main__":
    sys.exit(main())
