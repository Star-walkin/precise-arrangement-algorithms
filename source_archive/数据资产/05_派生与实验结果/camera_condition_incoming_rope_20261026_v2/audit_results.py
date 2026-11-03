"""A2 独立结构审计：固定开放图片、冻结输入、像素及记录契约。

本脚本不导入推理模块、不读取人工标签，也不评价识别准确率。它仅按
protocol.json 的固定 400 图清单读指定 run 的明确文件路径，不递归扫描目录。
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent
ROOT = ASSETS.parents[1]
DATA = ASSETS / "camera_condition_images_20261009_v2"
C3 = ASSETS / "camera_condition_grouped_conic_20261026_v6" / "all_001"
C4 = ASSETS / "camera_condition_full_surface_20261026_v7" / "all_001"
A1 = ASSETS / "camera_condition_incoming_rope_20261026_v1"
MANIFEST = DATA / "manifest.csv"
QA = ASSETS / "camera_condition_bright_rope_boundaries_20261009_v1" / "qa_plan.csv"
BASELINE = ROOT / "工程导航" / "05_实验版本档案" / "卷绳边界_整链基准_C3_20261026" / "baseline_manifest.json"
MASK_NAMES = ("c3_observed", "c3_smooth", "free_observed",
              "connection_observed", "connection_inferred")
OUTPUT_FOLDERS = ("overlays", "enhanced_overlays", "previews", "enhanced_previews")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def strict_json(path: Path):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def csv_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_image(path: Path, flag: int):
    data = np.fromfile(str(path), np.uint8)
    image = cv2.imdecode(data, flag)
    if image is None:
        raise ValueError(f"Image cannot be decoded: {path}")
    return image


def check(condition: bool, issues: list[str], message: str):
    if not condition:
        issues.append(message)


def finite_json(value, label: str, issues: list[str]):
    if isinstance(value, dict):
        for key, item in value.items():
            finite_json(item, f"{label}.{key}", issues)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_json(item, f"{label}[{index}]", issues)
    elif isinstance(value, float) and not math.isfinite(value):
        issues.append(f"{label}: non-finite number")


def xy_array(value, label: str, width: int, height: int,
             issues: list[str], *, count: int | None = None):
    if value is None:
        check(count is None, issues, f"{label}: missing coordinates")
        return None
    try:
        points = np.asarray(value, dtype=np.float64)
        if points.size == 0:
            points = np.empty((0, 2), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("expected Nx2")
        if count is not None and len(points) != count:
            issues.append(f"{label}: point count {len(points)} != {count}")
        if not np.all(np.isfinite(points)):
            issues.append(f"{label}: non-finite coordinate")
        elif len(points) and (np.any(points[:, 0] < -1) or
                              np.any(points[:, 0] > width + 1) or
                              np.any(points[:, 1] < -1) or
                              np.any(points[:, 1] > height + 1)):
            issues.append(f"{label}: outside original-image coordinates")
        return points
    except (TypeError, ValueError) as exc:
        issues.append(f"{label}: invalid Nx2 array ({exc})")
        return None


def unit_vector(value, label: str, issues: list[str]):
    try:
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (2,) or not np.all(np.isfinite(vector)):
            raise ValueError("expected finite [dx,dy]")
        norm = float(np.linalg.norm(vector))
        check(abs(norm - 1) <= 0.02, issues, f"{label}: not unit length ({norm:.5f})")
        return vector
    except (TypeError, ValueError) as exc:
        issues.append(f"{label}: {exc}")
        return None


def verify_free(free: dict, rec: dict, width: int, height: int,
                issues: list[str]):
    check(free.get("coordinate_system") == "original_unmirrored_image_xy_pixels",
          issues, "free_segment: coordinate_system is not original image pixels")
    check(free.get("camera_role") == rec.get("camera_role"),
          issues, "free_segment: camera role mismatch")
    check(free.get("status") == rec.get("free_status"),
          issues, "free_segment: status mismatch")
    check(free.get("complete_free_rope_claim") is False and
          free.get("contact_or_tangency_claim") is False,
          issues, "free_segment: unsupported physical claim")
    status = free.get("status")
    check(status in ("observed", "direction_only", "unknown"), issues,
          f"free_segment: unexpected status {status}")
    for key in ("observed_segment_xy", "observed_points_xy", "boundary_left_xy",
                "boundary_right_xy"):
        xy_array(free.get(key), f"free_segment.{key}", width, height, issues,
                 count=2 if key == "observed_segment_xy" and status == "observed" else None)
    if status == "unknown":
        check(free.get("center_xy") is None and free.get("width_px") is None,
              issues, "unknown free_segment has center or width")
        check(not free.get("observed_points_xy"), issues,
              "unknown free_segment has observed points")
        return
    seg = xy_array(free.get("observed_segment_xy"), "free_segment.segment",
                   width, height, issues, count=2)
    center = xy_array([free.get("center_xy")], "free_segment.center",
                      width, height, issues, count=1)
    direction = unit_vector(free.get("direction_xy"), "free_segment.direction", issues)
    normal = unit_vector(free.get("normal_xy"), "free_segment.normal", issues)
    points = xy_array(free.get("observed_points_xy"), "free_segment.observed_points",
                      width, height, issues)
    left = xy_array(free.get("boundary_left_xy"), "free_segment.left_edge",
                    width, height, issues)
    right = xy_array(free.get("boundary_right_xy"), "free_segment.right_edge",
                     width, height, issues)
    if seg is not None and center is not None and len(seg) == 2 and len(center) == 1:
        check(np.linalg.norm(center[0] - np.mean(seg, axis=0)) <= 0.2,
              issues, "free_segment: center does not match segment midpoint")
        length = float(np.linalg.norm(seg[1] - seg[0]))
        support = free.get("support_length_px")
        check(isinstance(support, (int, float)) and abs(support - length) <= 1,
              issues, "free_segment: support_length differs from segment")
        if direction is not None and length > 0:
            check(abs(float(np.dot(direction, (seg[1]-seg[0]) / length))) >= 0.98,
                  issues, "free_segment: segment and direction disagree")
    if direction is not None and normal is not None:
        check(abs(float(np.dot(direction, normal))) <= 0.02,
              issues, "free_segment: normal is not perpendicular")
        angle = free.get("angle_deg")
        if isinstance(angle, (int, float)):
            expected = math.degrees(math.atan2(direction[1], direction[0]))
            difference = (float(angle) - expected + 90) % 180 - 90
            check(abs(difference) <= 1, issues,
                  "free_segment: angle disagrees with direction modulo 180 degrees")
        else:
            issues.append("free_segment: observed angle missing")
    if status == "direction_only":
        # A1 冻结的方向可信但原图双边宽度不足：保留线位，不编造直径/边界。
        check(free.get("width_px") is None and free.get("width_interval_px") is None,
              issues, "direction_only free_segment has unsupported optical width")
        check(points is not None and left is not None and right is not None and
              len(points) == len(left) == len(right) == 0,
              issues, "direction_only free_segment has unsupported paired edges")
        check(rec.get("width_px") is None and rec.get("angle_deg") == free.get("angle_deg"),
              issues, "record: direction_only width/angle mismatch")
        return
    width_px = free.get("width_px")
    interval = free.get("width_interval_px")
    check(isinstance(width_px, (int, float)) and width_px > 0,
          issues, "free_segment: observed optical width must be positive")
    check(isinstance(interval, list) and len(interval) == 2 and
          all(isinstance(x, (int, float)) for x in interval) and
          0 < interval[0] <= width_px <= interval[1],
          issues, "free_segment: invalid width interval")
    if all(array is not None for array in (points, left, right)):
        check(len(points) > 0 and len(points) == len(left) == len(right),
              issues, "free_segment: measured center/edge sample count mismatch")
        if len(points) == len(left) == len(right) and len(points):
            max_gap = float(np.max(np.linalg.norm(points - (left+right)/2, axis=1)))
            check(max_gap <= 0.25, issues,
                  f"free_segment: center is not midpoint of paired optical edges ({max_gap:.3f}px)")
            edge_width = np.linalg.norm(left-right, axis=1)
            if isinstance(width_px, (int, float)) and width_px > 0:
                check(abs(float(np.median(edge_width))-width_px) <= max(2, width_px*0.3),
                      issues, "free_segment: sampled optical edges conflict with width")
    check(rec.get("width_px") == width_px and rec.get("angle_deg") == free.get("angle_deg"),
          issues, "record: free width or angle differs from free_segment")


def verify_connection(connection: dict, rec: dict, width: int, height: int,
                      issues: list[str]):
    candidates = connection.get("candidates", [])
    if not isinstance(candidates, list):
        issues.append("connection: candidates is not a list")
        return None
    ids = [c.get("candidate_id") for c in candidates]
    check(len(ids) == len(set(ids)) and all(isinstance(i, str) and i for i in ids),
          issues, "connection: candidate ids are missing or duplicate")
    selected_id = connection.get("selected_candidate_id")
    check(selected_id == rec.get("selected_candidate_id"),
          issues, "connection: selected id differs from record")
    check(connection.get("status") == rec.get("connection_status"),
          issues, "connection: status differs from record")
    check(connection.get("physical_contact_confirmed") is False and
          connection.get("contact_point_xy") is None,
          issues, "connection: unsupported physical contact claim")
    check(selected_id is None or selected_id in ids, issues,
          "connection: selected candidate is absent from pool")
    status = connection.get("status")
    if status in ("UNKNOWN", "AMBIGUOUS"):
        check(selected_id is None, issues,
              "connection: unknown/ambiguous status has a selected candidate")
    elif status == "STRAIGHT_SUPPORTED":
        check(selected_id == "S0", issues,
              "connection: straight status must select S0")
    elif status == "CURVE_CANDIDATE":
        check(selected_id is not None and selected_id != "S0", issues,
              "connection: curved status must select a curved candidate")
    else:
        issues.append(f"connection: unsupported status {status}")
    params = connection.get("diagnostics", {}).get("parameters", {})
    for candidate in candidates:
        label = f"candidate.{candidate.get('candidate_id')}"
        path = xy_array(candidate.get("path_xy"), label+".path", width, height, issues)
        n = len(path) if path is not None else None
        left = xy_array(candidate.get("boundary_left_xy"), label+".left",
                        width, height, issues, count=n)
        right = xy_array(candidate.get("boundary_right_xy"), label+".right",
                         width, height, issues, count=n)
        states = candidate.get("point_states")
        widths = candidate.get("widths_px")
        check(isinstance(states, list) and len(states) == n and
              set(states).issubset({"observed", "inferred", "unavailable"}),
              issues, label+": state length or value invalid")
        check(isinstance(widths, list) and len(widths) == n and
              all(isinstance(x, (int, float)) and x > 0 for x in widths),
              issues, label+": width length or value invalid")
        if path is not None and left is not None and right is not None and n:
            check(len(path) >= 2, issues, label+": fewer than two points")
            # path 是候选几何中心；左右边是独立梯度峰，允许原图中不对称。
            # 只验证双侧确实分居 path 两边且偏移量没有远超名义宽度。
            nominal = np.asarray(widths, dtype=np.float64)
            left_offset, right_offset = left-path, right-path
            opposite = np.sum(left_offset*right_offset, axis=1) < 0
            left_distance = np.linalg.norm(left_offset, axis=1)
            right_distance = np.linalg.norm(right_offset, axis=1)
            check(bool(np.all(opposite)), issues,
                  label+": measured edges are not on opposite sides of path")
            check(bool(np.all((left_distance >= nominal*.2) &
                              (left_distance <= nominal*.85) &
                              (right_distance >= nominal*.2) &
                              (right_distance <= nominal*.85))), issues,
                  label+": measured edge offsets inconsistent with nominal width")
        if isinstance(states, list):
            check(candidate.get("observed_count") == states.count("observed"), issues,
                  label+": observed_count disagrees with states")
        check(isinstance(candidate.get("surface_sample_count"), int) and
              candidate["surface_sample_count"] >= candidate.get("observed_count", 0),
              issues, label+": surface sample count below observed count")
        reasons = candidate.get("reject_reasons")
        check(isinstance(reasons, list) and len(reasons) == len(set(reasons)),
              issues, label+": reject reasons invalid/duplicate")
        if candidate.get("kind") == "straight_continuation":
            check(candidate.get("candidate_id") == "S0" and
                  candidate.get("join_index") is None and
                  candidate.get("transition_center_xy") is None and
                  not reasons, issues, label+": straight candidate metadata invalid")
        elif candidate.get("kind") == "c3_paired_groove_connection":
            join = candidate.get("join_index")
            check(isinstance(join, int) and n is not None and 0 <= join < n,
                  issues, label+": transition join index outside candidate path")
            groups = candidate.get("c3_group_ids")
            check(isinstance(groups, list) and len(groups) == 2 and
                  all(isinstance(g, str) and g for g in groups) and groups[0] != groups[1],
                  issues, label+": paired C3 group IDs invalid")
            if reasons:
                check(candidate.get("transition_center_xy") is None, issues,
                      label+": rejected candidate has transition center")
            elif path is not None and isinstance(join, int) and 0 <= join < len(path):
                center = xy_array([candidate.get("transition_center_xy")],
                                  label+".transition_center", width, height, issues, count=1)
                if center is not None and len(center):
                    check(np.linalg.norm(center[0]-path[join]) <= .01, issues,
                          label+": transition center differs from path at join index")
            comparison = candidate.get("branch_comparison", {})
            if isinstance(comparison, dict) and comparison:
                gate = {
                    "insufficient_distinct_branch_samples":
                        comparison.get("samples", -1) < params.get("min_branch_samples", 12),
                    "paired_curve_edges_not_sustained":
                        comparison.get("curve_support", -1) < params.get("min_branch_support", .45),
                    "curve_does_not_beat_straight":
                        comparison.get("enhanced_and_raw_score_advantage", -1) <
                        params.get("min_curve_advantage", .14),
                    "visible_straight_continuation_contradicts_bend":
                        comparison.get("straight_support", 2) >
                        params.get("max_supported_straight_fraction", .30) or
                        comparison.get("straight_supported_run_samples", 99) >= 8,
                }
                for reason, triggered in gate.items():
                    check((reason in reasons) == bool(triggered), issues,
                          label+f": branch gate/reject reason mismatch: {reason}")
                check(abs(candidate.get("selection_score", 1e9)-
                          (comparison.get("enhanced_and_raw_score_advantage", 0)+
                           .2*comparison.get("curve_support", 0))) <= 1e-5,
                      issues, label+": selection_score differs from stored branch evidence")
            else:
                issues.append(label+": missing branch comparison")
            # 旧运行可没有此第三轮门；只在冻结源码声明该门时逐候选核验。
            if "min_transition_support" in params:
                evidence = candidate.get("transition_evidence", {})
                if not isinstance(evidence, dict) or not evidence:
                    issues.append(label+": missing transition evidence")
                else:
                    support = evidence.get("paired_support_fraction", -1)
                    gap = evidence.get("longest_unsupported_gap_px", float("inf"))
                    max_gap = evidence.get("required_max_gap_px", -1)
                    check(isinstance(evidence.get("sample_count"), int) and
                          evidence["sample_count"] > 0 and 0 <= support <= 1 and
                          gap >= 0 and max_gap > 0, issues,
                          label+": transition evidence values invalid")
                    if isinstance(rec.get("width_px"), (int, float)):
                        check(abs(max_gap-params["max_transition_gap_widths"]*
                                  rec["width_px"]) <= 1e-5, issues,
                              label+": transition gap gate differs from fixed width/parameter")
                    trigger = support < params["min_transition_support"] or gap > max_gap
                    check(("transition_itself_not_continuously_observed" in reasons) == trigger,
                          issues, label+": transition gate/reject reason mismatch")
            if selected_id == candidate.get("candidate_id"):
                check(not reasons, issues,
                      label+": rejected curved candidate was selected")
        else:
            issues.append(label+": unknown candidate kind")
    return next((c for c in candidates if c.get("candidate_id") == selected_id), None)


def raster_ribbon(shape, left, right, states=None, kind="observed"):
    mask = np.zeros(shape, np.uint8)
    left, right = np.asarray(left, float), np.asarray(right, float)
    if left.ndim != 2 or right.shape != left.shape or left.shape[1] != 2:
        return mask
    for i in range(len(left)-1):
        if states is not None and not (states[i] == kind and states[i+1] == kind):
            continue
        if np.linalg.norm(left[i+1]-left[i]) > 45 or np.linalg.norm(right[i+1]-right[i]) > 45:
            continue
        polygon = np.rint([left[i], right[i], right[i+1], left[i+1]]).astype(np.int32)
        cv2.fillConvexPoly(mask, polygon, 255)
    return mask


def safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or any(part in ("..", ".") for part in path.parts):
        raise ValueError(f"Unsafe relative path: {value}")
    return path


def expected_source_paths():
    v3 = ASSETS / "camera_condition_bright_rope_boundaries_20261025_v3"
    c3dir = ASSETS / "camera_condition_grouped_conic_20261026_v6"
    return {
        str(v3 / "full_roi_observation.py"), str(c3dir / "conic_family.py"),
        str(A1 / "experiment_io.py"), str(A1 / "protocol.json"),
        *(str(HERE / name) for name in ("experiment_io.py", "run_experiment.py",
            "paired_connection.py", "free_refinement.py", "illumination_c3.py",
            "render_results.py", "protocol.json")),
    }


def source_checks(run: Path, context: dict, issues: list[str]):
    source = context.get("source_sha256", {})
    snapshots = context.get("source_snapshot_paths", {})
    if not isinstance(source, dict) or not isinstance(snapshots, dict):
        issues.append("run_context: invalid source hash/snapshot mapping")
        return {"snapshot_count": 0, "current_source_drift": []}
    check(set(source) == expected_source_paths(), issues,
          "run_context: source dependency set incomplete or unexpected")
    check(set(source) == set(snapshots), issues,
          "run_context: snapshot mapping differs from source mapping")
    drift = []
    for original, digest in source.items():
        try:
            rel = safe_relative(snapshots[original])
            check(rel.parts[0] == "source_snapshot", issues,
                  f"source snapshot outside source_snapshot: {rel}")
            snapshot = run / rel
            check(sha256(snapshot) == digest, issues,
                  f"source snapshot hash mismatch: {rel}")
            current = Path(original)
            if not current.is_file() or sha256(current) != digest:
                drift.append(original)
        except Exception as exc:
            issues.append(f"source snapshot invalid for {original}: {exc}")
    return {"snapshot_count": len(snapshots), "current_source_drift": drift}


def fixed_inputs(issues: list[str]):
    protocol = strict_json(HERE / "protocol.json")
    a1_protocol = strict_json(A1 / "protocol.json")
    inventory = [
        (MANIFEST, protocol["input_manifest_sha256"], "original manifest"),
        (BASELINE, protocol["baseline_manifest_sha256"], "C3 baseline manifest"),
        (QA, a1_protocol["qa_plan_sha256"], "fixed QA plan"),
        (C3 / "integrity_audit.json", a1_protocol["c3_integrity_sha256"], "C3 inventory"),
        (C4 / "output_manifest.json", a1_protocol["c4_output_manifest_sha256"], "C4 inventory"),
        (A1 / "all_001" / "output_manifest.json", protocol["a1_output_manifest_sha256"], "A1 inventory"),
    ]
    for path, frozen_hash, name in inventory:
        check(sha256(path) == frozen_hash, issues, f"{name}: frozen hash mismatch")
    rows = csv_rows(MANIFEST)
    qa = {r["output_file"] for r in csv_rows(QA)}
    pilot = set(protocol["pilot_images"])
    check(len(rows) == protocol["denominator"] == 400, issues,
          "fixed input manifest must contain 400 rows")
    check(len({r["output_file"] for r in rows}) == len(rows), issues,
          "fixed input manifest contains duplicate output_file")
    check(len(qa) == 38 and len(pilot) == 49, issues,
          "fixed QA38 or pilot49 denominator changed")
    check(pilot.issubset({r["output_file"] for r in rows}), issues,
          "pilot selection contains an image outside the fixed 400")
    for row in rows:
        rel = safe_relative(row["output_file"])
        check(len(rel.parts) == 3 and rel.parts[:2] == ("images", row["camera_role"]+"_camera")
              and rel.suffix.lower() == ".png", issues,
              f"unsafe or unexpected original image path: {rel}")
        check("20260925" not in row["source_file"] + row["output_file"] and
              not any(s in row["dataset_use_status"].lower() for s in ("reserved", "heldout")),
              issues, f"reserved input forbidden: {rel}")
    c3data = strict_json(C3 / "integrity_audit.json")
    c4data = strict_json(C4 / "output_manifest.json")
    a1data = strict_json(A1 / "all_001" / "output_manifest.json")
    check(c3data.get("passed") == 400 and c3data.get("failed") == 0 and
          c3data.get("exit_code") == 0, issues, "C3 parent audit is not complete")
    check(c4data.get("case_count") == 400, issues, "C4 parent inventory incomplete")
    check(a1data.get("image_count") == 400, issues, "A1 parent inventory incomplete")
    c3hash = {r["path"]: r["sha256"] for r in c3data["file_manifest"]}
    c4hash = {r["absolute_path"]: r["sha256"]
              for case in c4data["cases"] for r in case["files"]}
    a1hash = {r["path"]: r["sha256"] for r in a1data["files"]}
    return protocol, rows, qa, pilot, c3hash, c4hash, a1hash


def verify_case(run: Path, row: dict, qa: set[str], pilot: set[str],
                variant: str, c3hash: dict, c4hash: dict, a1hash: dict):
    issues: list[str] = []
    rel = safe_relative(row["output_file"])
    role, name, stem = rel.parent.name, rel.name, rel.stem
    original_path = DATA / rel
    record_path = run / "records" / role / (stem + ".json")
    try:
        check(sha256(original_path) == row["output_sha256"], issues,
              "original PNG hash differs from manifest")
        record_obj = strict_json(record_path)
        finite_json(record_obj, "record_file", issues)
        rec = record_obj["record"]
        check(rec.get("status") == "completed", issues, "record is not completed")
        for field in ("output_file", "camera_role", "source_group", "source_file",
                      "source_time_s", "dataset_use_status"):
            check(rec.get(field) == row[field], issues, f"record.{field} differs from input manifest")
        check(rec.get("source_png_sha256") == row["output_sha256"], issues,
              "record original PNG hash differs from manifest")
        check(rec.get("variant") == variant and rec.get("version") == "IR-A2.0.0", issues,
              "record variant/version mismatch")
        check(rec.get("qa_selected") == (row["output_file"] in qa) and
              rec.get("pilot_selected") == (row["output_file"] in pilot),
              issues, "record QA/pilot selection mismatch")
        input_paths = [
            C4 / "roi" / "surface_masks" / role / name,
            C3 / "boundaries" / "records" / role / (stem + ".json"),
            C4 / "boundaries" / "records" / role / (stem + ".json"),
            A1 / "all_001" / "records" / role / (stem + ".json"),
        ]
        frozen = [c4hash.get(str(input_paths[0])), c3hash.get(str(input_paths[1])),
                  c4hash.get(str(input_paths[2])),
                  a1hash.get((Path("records")/role/(stem+".json")).as_posix())]
        reported = rec.get("input_dependency_sha256", {})
        check(isinstance(reported, dict) and set(reported) == {str(p) for p in input_paths},
              issues, "record parent dependency path set mismatch")
        for path, expected in zip(input_paths, frozen):
            check(expected is not None, issues, f"frozen inventory lacks {path}")
            actual = sha256(path)
            check(expected == actual == reported.get(str(path)), issues,
                  f"parent dependency hash mismatch: {path}")
        original = read_image(original_path, cv2.IMREAD_COLOR)
        height, width = original.shape[:2]
        check(original.shape == (int(row["height"]), int(row["width"]), 3), issues,
              "original PNG dimensions differ from manifest")
        surface = read_image(input_paths[0], cv2.IMREAD_GRAYSCALE)
        check(surface.shape == (height, width) and
              np.all((surface == 0) | (surface == 255)), issues,
              "C4 surface mask invalid binary/size")
        enhanced = read_image(run / "enhanced_roi" / role / name, cv2.IMREAD_UNCHANGED)
        check(enhanced.shape == original.shape and enhanced.dtype == np.uint8,
              issues, "enhanced ROI dimensions/channels/dtype differ from original")
        if enhanced.shape == original.shape and surface.shape == (height, width):
            check(not np.any(enhanced[surface == 0]), issues,
                  "enhanced ROI has nonzero pixels outside frozen C4 surface")
            if variant == "raw":
                check(np.array_equal(enhanced[surface != 0], original[surface != 0]), issues,
                      "raw variant modified original pixels inside surface")
        free = record_obj["free_segment"]
        connection = record_obj["connection"]
        verify_free(free, rec, width, height, issues)
        chosen = verify_connection(connection, rec, width, height, issues)
        masks = {}
        for kind in MASK_NAMES:
            mask = read_image(run / "masks" / kind / role / name, cv2.IMREAD_UNCHANGED)
            masks[kind] = mask
            check(mask.shape == (height, width) and mask.dtype == np.uint8 and
                  np.all((mask == 0) | (mask == 255)), issues,
                  f"{kind}: not original-size binary uint8 mask")
            if mask.shape == (height, width):
                check(rec.get("mask_pixels", {}).get(kind) == int(np.count_nonzero(mask)),
                      issues, f"{kind}: pixel count differs from record")
        observed, inferred = masks["connection_observed"], masks["connection_inferred"]
        if observed.shape == inferred.shape == surface.shape:
            check(not np.any((observed > 0) & (inferred > 0)), issues,
                  "connection observed and inferred masks overlap")
            check(not np.any(observed[surface == 0]) and
                  not np.any(inferred[surface == 0]), issues,
                  "connection masks extend beyond frozen surface")
        expected_free = raster_ribbon((height, width), free.get("boundary_left_xy", []),
                                      free.get("boundary_right_xy", []))
        check(np.array_equal(masks["free_observed"], expected_free), issues,
              "free_observed does not rasterize reported optical boundaries")
        for kind in ("observed", "inferred"):
            expected = np.zeros((height, width), np.uint8)
            if chosen is not None:
                expected = raster_ribbon((height, width), chosen["boundary_left_xy"],
                                         chosen["boundary_right_xy"],
                                         chosen["point_states"], kind)
                expected[surface == 0] = 0
                if kind == "inferred":
                    expected[observed > 0] = 0
            check(np.array_equal(masks["connection_"+kind], expected), issues,
                  f"connection_{kind} does not match selected candidate raster")
        for folder in OUTPUT_FOLDERS:
            suffix = ".jpg" if folder.endswith("previews") else ".png"
            path = run / folder / role / (stem + suffix)
            check(path.is_file() and path.stat().st_size > 0, issues,
                  f"missing/empty display output: {path}")
        return issues, rec
    except Exception as exc:
        issues.append(f"case audit exception: {type(exc).__name__}: {exc}")
        return issues, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Existing direct child run directory or its name")
    parser.add_argument("--output", required=True, type=Path,
                        help="New JSON report path inside the run directory; never overwritten")
    args = parser.parse_args()
    run = Path(args.run)
    if not run.is_absolute():
        run = HERE / run
    if run.parent != HERE or not run.is_dir() or run.is_junction() or run.is_symlink():
        parser.error("--run must be an existing, non-junction direct child of A2 root")
    output = args.output if args.output.is_absolute() else run / args.output
    if output.parent != run or output.exists():
        parser.error("--output must be a new direct child JSON file of the run")
    cv2.setNumThreads(1)
    global_errors: list[str] = []
    findings: list[dict] = []
    source_info = {}
    planned = 0
    try:
        protocol, rows, qa, pilot, c3hash, c4hash, a1hash = fixed_inputs(global_errors)
        context = strict_json(run / "run_context.json")
        summary = strict_json(run / "summary.json")
        finite_json(context, "run_context", global_errors)
        finite_json(summary, "summary", global_errors)
        mode, variant = context.get("mode"), context.get("variant")
        check(mode in ("pilot", "all") and variant in
              ("raw", "global_gamma", "illumination"), global_errors,
              "run_context: invalid mode/variant")
        selected = [r for r in rows if mode == "all" or r["output_file"] in pilot]
        planned = len(selected)
        check(planned == (49 if mode == "pilot" else 400), global_errors,
              "selection denominator is not fixed 49/400")
        expected_files = [r["output_file"] for r in selected]
        check(context.get("selection") == expected_files, global_errors,
              "run_context selection differs from frozen input order")
        check(summary.get("mode") == mode and summary.get("variant") == variant and
              summary.get("planned") == planned and summary.get("completed") == planned and
              summary.get("failed") == 0 and summary.get("source_unchanged") is True and
              summary.get("exit_code") == 0 and summary.get("accuracy") is None,
              global_errors, "summary: denominator/completion/source/accuracy contract failed")
        check(context.get("equipment_access") is False and
              context.get("reserved_inputs") is False,
              global_errors, "run_context claims equipment or reserved inputs")
        source_info = source_checks(run, context, global_errors)
        if mode == "all":
            freeze = strict_json(HERE / "freeze_all.json")
            check(freeze.get("variant") == variant and
                  freeze.get("source_sha256") == context.get("source_sha256"),
                  global_errors, "full run differs from frozen variant/source choice")
            pilot_run = HERE / freeze["pilot_run"]
            check(pilot_run.parent == HERE and pilot_run.is_dir() and
                  sha256(pilot_run / "summary.json") == freeze["pilot_summary_sha256"] and
                  sha256(pilot_run / "independent_audit.json") == freeze["pilot_audit_sha256"],
                  global_errors, "full run pilot selection evidence changed")
        log_lines = (run / "run_log.jsonl").read_text(encoding="utf-8").splitlines()
        log = [json.loads(line) for line in log_lines]
        check(len(log) == planned and [r.get("output_file") for r in log] == expected_files,
              global_errors, "run_log missing/duplicate/reordered input entries")
        selection_csv = csv_rows(run / "selection.csv")
        check(len(selection_csv) == planned and
              [r.get("output_file") for r in selection_csv] == expected_files,
              global_errors, "selection.csv missing/duplicate/reordered input entries")
        counts = {"free_status": Counter(), "refinement_status": Counter(),
                  "connection_status": Counter()}
        for index, row in enumerate(selected):
            issues, rec = verify_case(run, row, qa, pilot, variant,
                                      c3hash, c4hash, a1hash)
            if rec is not None:
                if index < len(log):
                    logged = dict(log[index]); logged.pop("elapsed_s", None)
                    check(logged == rec, issues,
                          "run_log record fields differ from individual record")
                if index < len(selection_csv):
                    csv_rec = selection_csv[index]
                    for field in ("output_file", "camera_role", "source_group", "status",
                                  "free_status", "refinement_status", "connection_status"):
                        check(str(rec.get(field)) == csv_rec.get(field), issues,
                              f"selection.csv {field} differs from record")
                for key in counts:
                    counts[key][rec.get(key)] += 1
            findings.append({"output_file": row["output_file"], "passed": not issues,
                             "issues": issues})
        for key, counter in counts.items():
            check(dict(counter) == summary.get(key), global_errors,
                  f"summary {key} counts differ from records")
        check(summary.get("by_camera", {}).get("left", {}).get("images") ==
              sum(r["camera_role"] == "left" for r in selected) and
              summary.get("by_camera", {}).get("right", {}).get("images") ==
              sum(r["camera_role"] == "right" for r in selected),
              global_errors, "summary by-camera denominator mismatch")
    except Exception as exc:
        global_errors.append(f"global audit exception: {type(exc).__name__}: {exc}")
    passed = sum(item["passed"] for item in findings)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run": str(run), "script_sha256": sha256(Path(__file__)),
        "planned": planned, "checked": len(findings), "passed": passed,
        "failed": len(findings) - passed, "global_errors": global_errors,
        "source_snapshot": source_info, "findings": findings,
        "meaning": "Structural and byte-integrity audit only; no accuracy or physical-contact claim.",
        "reserved_inputs_read": False, "equipment_access": False,
        "exit_code": int(bool(global_errors) or len(findings) != planned or passed != planned),
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in
                      ("run", "planned", "checked", "passed", "failed", "global_errors", "exit_code")},
                     ensure_ascii=False), flush=True)
    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
