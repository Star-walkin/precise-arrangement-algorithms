"""独立审计 IR-A1 离线产物的来源、坐标、状态和证据措辞。

此脚本不运行检测器，也不把自动产物当真值。它只读取已完成的 pilot/all
运行和固定输入，输出新的 JSON；明显绳路是否正确仍需看原图。
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import sys

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent
ROOT = ASSETS.parents[1]
DATA = ASSETS / "camera_condition_images_20261009_v2"
MANIFEST = DATA / "manifest.csv"
QA = ASSETS / "camera_condition_bright_rope_boundaries_20261009_v1" / "qa_plan.csv"
BASELINE = ROOT / "工程导航/05_实验版本档案/卷绳边界_整链基准_C3_20261026/baseline_manifest.json"
C3 = ASSETS / "camera_condition_grouped_conic_20261026_v6" / "all_001"
C4 = ASSETS / "camera_condition_full_surface_20261026_v7" / "all_001"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_image(path: Path, flags: int):
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, flags)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def safe_output_path(run: Path, raw: str) -> Path:
    path = run / raw
    if not path.resolve().is_relative_to(run.resolve()):
        raise ValueError(f"Path escapes run directory: {raw}")
    return path


def finite_point(point, width: int, height: int) -> bool:
    return (isinstance(point, (list, tuple)) and len(point) == 2
            and all(isinstance(value, (int, float)) and math.isfinite(value) for value in point)
            and 0 <= point[0] < width and 0 <= point[1] < height)


def check_points(value, width: int, height: int, label: str, errors: list[str]):
    if not isinstance(value, list):
        errors.append(f"{label}: points are not a list")
        return
    for index, point in enumerate(value):
        if point is not None and not finite_point(point, width, height):
            errors.append(f"{label}: invalid image coordinate at {index}: {point}")
            break


def binary_mask(path: Path, width: int, height: int, errors: list[str]):
    if not path.is_file():
        errors.append(f"missing mask: {path}")
        return None
    try:
        mask = read_image(path, cv2.IMREAD_UNCHANGED)
    except Exception as error:
        errors.append(str(error))
        return None
    if mask.shape != (height, width) or mask.dtype != np.uint8:
        errors.append(f"wrong shape/dtype: {path} {mask.shape} {mask.dtype}")
        return None
    if np.any((mask != 0) & (mask != 255)):
        errors.append(f"non-binary mask: {path}")
        return None
    return mask


def fail_if(condition: bool, message: str, errors: list[str]):
    if condition:
        errors.append(message)


def expected_inventory(protocol: dict, errors: list[str]):
    files = [(MANIFEST, "input_manifest_sha256"), (QA, "qa_plan_sha256"),
             (BASELINE, "baseline_manifest_sha256"),
             (C3 / "integrity_audit.json", "c3_integrity_sha256"),
             (C4 / "output_manifest.json", "c4_output_manifest_sha256")]
    hashes = {}
    for path, field in files:
        if not path.is_file():
            errors.append(f"missing fixed input inventory: {path}")
            continue
        digest = sha256(path)
        hashes[field] = digest
        fail_if(digest != protocol[field], f"fixed inventory hash differs: {path}", errors)
    known = {}
    if (C3 / "integrity_audit.json").is_file():
        for item in read_json(C3 / "integrity_audit.json").get("file_manifest", []):
            known[str(Path(item["path"]).resolve()).casefold()] = item["sha256"]
    if (C4 / "output_manifest.json").is_file():
        for case in read_json(C4 / "output_manifest.json").get("cases", []):
            for item in case.get("files", []):
                known[str(Path(item["absolute_path"]).resolve()).casefold()] = item["sha256"]
    return hashes, known


def audit(run: Path):
    errors: list[str] = []
    cases = []
    protocol = read_json(HERE / "protocol.json")
    context = read_json(run / "run_context.json")
    summary = read_json(run / "summary.json")
    scope = read_json(HERE / "evaluation_scope.json")
    inventory_hashes, known_dependencies = expected_inventory(protocol, errors)

    mode = context.get("mode")
    fail_if(mode not in {"pilot", "all"}, f"unknown run mode: {mode}", errors)
    all_rows = read_csv(MANIFEST)
    qa_rows = read_csv(QA)
    by_name = {row["output_file"]: row for row in all_rows}
    qa_names = {row["output_file"] for row in qa_rows}
    expected = [row for row in all_rows if mode == "all" or row["output_file"] in qa_names]
    expected_names = [row["output_file"] for row in expected]
    fail_if(len(all_rows) != 400 or len(by_name) != 400, "fixed manifest must have 400 unique images", errors)
    fail_if(len(qa_rows) != 38 or len(qa_names) != 38, "fixed QA must have 38 unique images", errors)
    fail_if(len(expected) != (38 if mode == "pilot" else 400), "run denominator differs", errors)
    fail_if(context.get("selection") != expected_names, "run selection/order differs from frozen manifest and QA", errors)
    fail_if(context.get("manifest_sha256") != inventory_hashes.get("input_manifest_sha256"), "context manifest hash differs", errors)
    fail_if(context.get("qa_sha256") != inventory_hashes.get("qa_plan_sha256"), "context QA hash differs", errors)
    fail_if(context.get("baseline_sha256") != inventory_hashes.get("baseline_manifest_sha256"), "context baseline hash differs", errors)
    fail_if(context.get("equipment_access") is not False or context.get("reserved_inputs") is not False,
            "context indicates equipment/reserved access", errors)
    fail_if(summary.get("planned") != len(expected), "summary planned differs", errors)
    fail_if(summary.get("completed", 0) + summary.get("failed", 0) != len(expected), "summary outcome denominator differs", errors)
    fail_if(summary.get("presence_accuracy") is not None or summary.get("physical_contact_accuracy") is not None,
            "summary falsely provides target accuracy", errors)
    fail_if(summary.get("source_unchanged") is not True, "runner reported source changed during run", errors)

    # 运行期间的快照是本轮权威源码；之后工作目录允许发生有记录的下一轮改动。
    source_hashes = context.get("source_sha256", {})
    for source, digest in source_hashes.items():
        snapshot = run / "source_snapshot" / Path(source).name
        fail_if(not snapshot.is_file(), f"missing source snapshot: {snapshot}", errors)
        if snapshot.is_file():
            fail_if(sha256(snapshot) != digest, f"source snapshot hash differs: {snapshot}", errors)
    for model, digest in context.get("model_sha256", {}).items():
        path = Path(model)
        fail_if(not path.is_file(), f"missing model: {path}", errors)
        if path.is_file():
            fail_if(sha256(path) != digest, f"model hash differs: {path}", errors)

    selection_rows = read_csv(run / "selection.csv")
    fail_if([row["output_file"] for row in selection_rows] != expected_names,
            "selection.csv differs from fixed order", errors)
    log_rows = [json.loads(line) for line in (run / "run_log.jsonl").read_text(encoding="utf-8").splitlines() if line]
    fail_if([row["output_file"] for row in log_rows] != expected_names,
            "run_log.jsonl differs from fixed order", errors)
    status_counts = Counter()
    free_counts = Counter()
    connection_counts = Counter()
    selected_count = 0
    candidates_total = 0
    hashes_checked = 0
    dependency_hashes_checked = 0

    for row in expected:
        rel = Path(row["output_file"])
        role = rel.parent.name
        stem = rel.stem
        item_errors: list[str] = []
        original = DATA / rel
        actual_source_sha = sha256(original) if original.is_file() else None
        fail_if(actual_source_sha != row["output_sha256"], "source PNG hash differs", item_errors)
        hashes_checked += actual_source_sha is not None
        fail_if(row.get("dataset_use_status") != "unlabeled_open_development_no_new_test_split",
                "not an allowed open-development image", item_errors)
        fail_if("20260925" in row.get("source_file", "") + row["output_file"],
                "reserved-date source in fixed manifest", item_errors)

        record_path = run / "records" / role / f"{stem}.json"
        if not record_path.is_file():
            item_errors.append("missing record JSON")
            cases.append({"output_file": row["output_file"], "errors": item_errors})
            errors.extend(f"{row['output_file']}: {message}" for message in item_errors)
            continue
        saved = read_json(record_path)
        record = saved.get("record", {})
        free = saved.get("free_segment", {})
        connection = saved.get("connection", {})
        comparison = saved.get("comparison", {})
        fail_if(record.get("output_file") != row["output_file"], "record image identity differs", item_errors)
        fail_if(record.get("camera_role") != row["camera_role"], "record camera role differs", item_errors)
        fail_if(record.get("source_png_sha256") != row["output_sha256"], "record source hash differs", item_errors)
        fail_if(record.get("source_group") != row["source_group"], "record source group differs", item_errors)
        fail_if(record.get("qa_selected") is not (row["output_file"] in qa_names), "record QA flag differs", item_errors)
        fail_if(record.get("physical_contact_verified") is not False, "physical contact claim is not false", item_errors)
        fail_if(record.get("image_presence_ground_truth") is not None, "presence label/claim injected", item_errors)
        fail_if(record.get("selected_candidate_id") != connection.get("selected_candidate_id"),
                "record and connection selection differ", item_errors)
        fail_if(record.get("free_status") != free.get("status"), "record and free status differ", item_errors)
        fail_if(record.get("connection_status") != connection.get("status"), "record and connection status differ", item_errors)
        status = record.get("status")
        status_counts[status] += 1
        free_counts[record.get("free_status")] += 1
        connection_counts[record.get("connection_status")] += 1
        fail_if(record.get("version") != protocol["version"], "record protocol version differs", item_errors)
        fail_if(status not in {"completed", "failed"}, "invalid case status", item_errors)
        fail_if(record.get("free_status") not in {"observed", "direction_only", "unknown", "error"},
                "invalid free status", item_errors)
        fail_if(record.get("connection_status") not in {"MULTI_CANDIDATE", "SINGLE_IMAGE_HYPOTHESIS", "UNKNOWN", "ERROR"},
                "invalid connection status", item_errors)

        dependencies = record.get("input_dependency_sha256", {})
        if status == "completed":
            fail_if(len(dependencies) != 3, "expected exactly three frozen C3/C4 dependencies", item_errors)
            for path_text, digest in dependencies.items():
                path = Path(path_text)
                if not path.is_file():
                    item_errors.append(f"missing dependency: {path_text}")
                    continue
                actual_digest = sha256(path)
                inventory_digest = known_dependencies.get(str(path.resolve()).casefold())
                fail_if(actual_digest != digest or actual_digest != inventory_digest,
                        f"dependency differs from both record and parent inventory: {path_text}", item_errors)
                dependency_hashes_checked += 1

            width = int(row["width"])
            height = int(row["height"])
            masks = {}
            for kind in ("free_observed", "selected_observed", "selected_inferred"):
                masks[kind] = binary_mask(run / "masks" / kind / role / rel.name, width, height, item_errors)
                if masks[kind] is not None:
                    fail_if(int(np.count_nonzero(masks[kind])) != record.get(f"{kind}_pixels"),
                            f"{kind} pixel count differs", item_errors)
            if masks["selected_observed"] is not None and masks["selected_inferred"] is not None:
                fail_if(np.any((masks["selected_observed"] > 0) & (masks["selected_inferred"] > 0)),
                        "selected observed/inferred masks overlap", item_errors)

            # 原图只经字节哈希；叠图必须完整原图尺寸，缩略图只验证可读取。
            overlay_path = run / "overlays" / role / rel.name
            if not overlay_path.is_file():
                item_errors.append("missing overlay")
            else:
                overlay = read_image(overlay_path, cv2.IMREAD_COLOR)
                fail_if(overlay.shape[:2] != (height, width), "overlay dimensions differ from original", item_errors)
            for folder in ("previews", "baseline_previews"):
                preview_path = run / folder / role / f"{stem}.jpg"
                if not preview_path.is_file():
                    item_errors.append(f"missing {folder}")
                else:
                    preview = read_image(preview_path, cv2.IMREAD_COLOR)
                    fail_if(preview.size == 0, f"empty {folder}", item_errors)

            if free.get("status") == "observed":
                fail_if(not free.get("observed_points_xy"), "observed status lacks profile points", item_errors)
                for key in ("observed_points_xy", "boundary_left_xy", "boundary_right_xy",
                            "observed_segment_xy"):
                    check_points(free.get(key), width, height, f"free.{key}", item_errors)
                fail_if(len(free.get("boundary_left_xy", [])) != len(free.get("observed_points_xy", []))
                        or len(free.get("boundary_right_xy", [])) != len(free.get("observed_points_xy", [])),
                        "profile/boundary lengths differ", item_errors)
                fail_if(not isinstance(free.get("width_px"), (int, float)) or free.get("width_px", 0) <= 0,
                        "observed free span has no positive optical width", item_errors)
            fail_if(free.get("complete_free_rope_claim") is True or free.get("contact_or_tangency_claim") is True,
                    "free span overstates completeness/contact", item_errors)
            baseline = comparison.get("straight_extension", {})
            fail_if(baseline.get("contact_point_xy") is not None, "straight baseline asserts contact", item_errors)
            check_points(baseline.get("path_xy", []), width, height, "straight.path_xy", item_errors)

            candidates = connection.get("candidates", [])
            ids = [candidate.get("candidate_id") for candidate in candidates]
            fail_if(len(candidates) > protocol["candidate_limit"], "candidate budget exceeded", item_errors)
            fail_if(len(ids) != len(set(ids)), "candidate IDs duplicate", item_errors)
            fail_if(record.get("candidate_count") != len(candidates), "candidate count differs", item_errors)
            candidates_total += len(candidates)
            chosen = record.get("selected_candidate_id")
            fail_if(chosen is not None and chosen not in ids, "selected candidate absent from pool", item_errors)
            selected_count += chosen is not None
            if chosen is None:
                for kind in ("selected_observed", "selected_inferred"):
                    fail_if(masks[kind] is not None and np.any(masks[kind]),
                            f"{kind} nonempty without selected candidate", item_errors)
            for candidate in candidates:
                label = f"candidate {candidate.get('candidate_id')}"
                path = candidate.get("path_xy", [])
                states = candidate.get("point_states", [])
                check_points(path, width, height, label + ".path_xy", item_errors)
                fail_if(len(path) != len(states), label + " path/state lengths differ", item_errors)
                fail_if(any(state not in {"observed", "inferred", "unavailable"} for state in states),
                        label + " has invalid evidence state", item_errors)
                fail_if(candidate.get("physical_contact_confirmed") is not False
                        or candidate.get("material_identity_confirmed") is not False,
                        label + " asserts contact/material identity", item_errors)
                for key in ("transition_interval_xy", "transition_center_xy"):
                    value = candidate.get(key)
                    if value is not None:
                        check_points(value if key.endswith("interval_xy") else [value],
                                     width, height, label + "." + key, item_errors)
                for kind, selected_kind in (("observed_mask", "selected_observed"),
                                            ("inferred_mask", "selected_inferred")):
                    mask_rel = candidate.get(kind + "_path")
                    if mask_rel is None:
                        if chosen == candidate.get("candidate_id"):
                            fail_if(masks[selected_kind] is not None and np.any(masks[selected_kind]),
                                    label + f" selected {kind} missing", item_errors)
                        continue
                    try:
                        mask_path = safe_output_path(run, mask_rel)
                    except ValueError as error:
                        item_errors.append(str(error))
                        continue
                    cmask = binary_mask(mask_path, width, height, item_errors)
                    if chosen == candidate.get("candidate_id") and cmask is not None and masks[selected_kind] is not None:
                        fail_if(not np.array_equal(cmask, masks[selected_kind]),
                                label + f" selected {kind} differs from saved selected mask", item_errors)

        cases.append({"output_file": row["output_file"], "camera_role": row["camera_role"],
                      "status": status, "free_status": record.get("free_status"),
                      "connection_status": record.get("connection_status"),
                      "candidate_count": record.get("candidate_count"),
                      "selected_candidate_id": record.get("selected_candidate_id"),
                      "errors": item_errors})
        errors.extend(f"{row['output_file']}: {message}" for message in item_errors)

    fail_if(dict(status_counts) != {key: count for key, count in (("completed", summary.get("completed")),
                                                                  ("failed", summary.get("failed"))) if count},
            "summary completed/failed counts differ from records", errors)
    fail_if(dict(free_counts) != summary.get("free_status"), "summary free counts differ", errors)
    fail_if(dict(connection_counts) != summary.get("connection_status"), "summary connection counts differ", errors)
    result = {
        "audit_version": "IR-A1-independent-structure-1",
        "run": str(run), "mode": mode, "passed": not errors, "error_count": len(errors),
        "errors": errors, "scope": "file/coordinate/provenance/state audit only; no incoming-rope or contact accuracy",
        "expected_images": len(expected), "case_results": cases,
        "counts": {"status": dict(status_counts), "free_status": dict(free_counts),
                   "connection_status": dict(connection_counts), "candidates_total": candidates_total,
                   "selected_count": selected_count, "source_png_hashes_checked": hashes_checked,
                   "c3_c4_dependency_hashes_checked": dependency_hashes_checked},
        "fixed_manifest_hashes": inventory_hashes,
        "source_snapshot_hashes_checked": len(source_hashes),
        "script_sha256": sha256(Path(__file__)),
        "evaluation_scope_sha256": sha256(HERE / "evaluation_scope.json"),
        "protocol_sha256": sha256(HERE / "protocol.json"),
        "fixed_visual_review_selection": scope["fixed_review"]["selected_output_files"],
        "limitations": ["No directly matched independent single-frame geometry/contact labels for fixed 400",
                        "All candidate and mask counts are output descriptions, not correctness rates",
                        "Original-image visual review is a separate post-prediction diagnostic"],
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    if not run.is_dir() or run.parent != HERE:
        raise ValueError("Run must be an existing direct child of this experiment directory")
    if output.parent != run:
        raise ValueError("Audit JSON must be a new file inside the selected run")
    result = audit(run)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "error_count": result["error_count"],
                      "counts": result["counts"], "output": str(output)}, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
