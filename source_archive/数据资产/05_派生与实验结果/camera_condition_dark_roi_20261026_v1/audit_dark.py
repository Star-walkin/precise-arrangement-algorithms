"""暗区三步离线结果的独立结构审计；不评价边界准确率。

只按冻结 manifest 明确打开 12/400 张开放图片、C3/C4 冻结依赖及指定
运行的逐图文件。不导入实验算法，不递归访问目录，不读取人工标签。
"""
from __future__ import annotations

import argparse
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
BASELINE = ROOT / "工程导航" / "05_实验版本档案" / "卷绳边界_整链基准_C3_20261026" / "baseline_manifest.json"
MANIFEST = DATA / "manifest.csv"
QA_PLAN = ASSETS / "camera_condition_bright_rope_boundaries_20261009_v1" / "qa_plan.csv"
IMAGE_GROUPS = ("01_欠曝光ROI", "02_对比度增强ROI", "03_C3边界结果")
MASK_GROUPS = ("dark_roi", "observed", "smooth")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path: Path):
    def reject(value):
        raise ValueError(f"Non-finite JSON constant {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def image(path: Path, flag: int):
    result = cv2.imdecode(np.fromfile(str(path), np.uint8), flag)
    if result is None:
        raise ValueError(f"Cannot decode image: {path}")
    return result


def require(condition: bool, issues: list[str], message: str):
    if not condition:
        issues.append(message)


def finite(value, prefix: str, issues: list[str]):
    if isinstance(value, dict):
        for key, item in value.items():
            finite(item, prefix+"."+str(key), issues)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite(item, f"{prefix}[{index}]", issues)
    elif isinstance(value, float) and not math.isfinite(value):
        issues.append(prefix+": non-finite number")


def relative(value: str):
    path = Path(value)
    if path.is_absolute() or not path.parts or any(part in (".", "..") for part in path.parts):
        raise ValueError("Unsafe relative path: "+value)
    return path


def frozen_inputs(issues: list[str]):
    protocol = read_json(HERE / "protocol.json")
    a1_protocol = read_json(A1 / "protocol.json")
    inventory = (
        (MANIFEST, protocol["input_manifest_sha256"], "400 original manifest"),
        (BASELINE, protocol["baseline_manifest_sha256"], "C3 baseline manifest"),
        (QA_PLAN, a1_protocol["qa_plan_sha256"], "QA38 plan"),
        (C3 / "integrity_audit.json", a1_protocol["c3_integrity_sha256"], "C3 frozen inventory"),
        (C4 / "output_manifest.json", a1_protocol["c4_output_manifest_sha256"], "C4 frozen inventory"),
    )
    for path, expected, label in inventory:
        require(sha256(path) == expected, issues, label+": hash differs from frozen protocol")
    rows = read_csv(MANIFEST)
    pilot = set(protocol["pilot_images"])
    qa = {row["output_file"] for row in read_csv(QA_PLAN)}
    require(len(rows) == protocol.get("input_count") == 400, issues,
            "input manifest is not fixed400")
    require(len({row["output_file"] for row in rows}) == 400, issues,
            "input manifest duplicate output_file")
    require(len(qa) == 38 and len(pilot) == 12 and
            pilot.issubset(qa) and pilot.issubset({r["output_file"] for r in rows}),
            issues, "pilot is not the frozen QA12 subset")
    for row in rows:
        rel = relative(row["output_file"])
        require(len(rel.parts) == 3 and rel.parts[0] == "images" and
                rel.parts[1] == row["camera_role"]+"_camera" and rel.suffix == ".png",
                issues, f"unapproved original PNG path {rel}")
        require("20260925" not in row["source_file"]+row["output_file"] and
                not any(word in row["dataset_use_status"].lower()
                        for word in ("reserved", "heldout")),
                issues, f"reserved input in manifest: {rel}")
    c3_audit = read_json(C3 / "integrity_audit.json")
    c4_manifest = read_json(C4 / "output_manifest.json")
    require(c3_audit.get("passed") == 400 and c3_audit.get("failed") == 0 and
            c3_audit.get("exit_code") == 0, issues, "C3 frozen run is incomplete")
    require(c4_manifest.get("case_count") == 400, issues, "C4 inventory is incomplete")
    c3_hashes = {item["path"]: item["sha256"] for item in c3_audit["file_manifest"]}
    c4_hashes = {item["absolute_path"]: item["sha256"]
                 for case in c4_manifest["cases"] for item in case["files"]}
    return protocol, rows, pilot, c3_hashes, c4_hashes


def expected_sources():
    b1 = ASSETS / "camera_condition_bright_rope_boundaries_20261025_v3" / "full_roi_observation.py"
    c3 = ASSETS / "camera_condition_grouped_conic_20261026_v6" / "conic_family.py"
    return {str(b1), str(c3), str(A1 / "experiment_io.py"), str(A1 / "protocol.json"),
            *(str(HERE / name) for name in ("data_io.py", "run.py", "dark_roi.py",
                                           "dark_c3.py", "protocol.json"))}


def source_snapshots(run: Path, context: dict, issues: list[str]):
    sources = context.get("source_sha256", {})
    snapshots = context.get("source_snapshot_paths", {})
    if not isinstance(sources, dict) or not isinstance(snapshots, dict):
        issues.append("source/snapshot mappings missing")
        return {"snapshot_count": 0, "current_source_drift": []}
    require(set(sources) == expected_sources(), issues,
            "source snapshot dependency set incomplete/unexpected")
    require(set(sources) == set(snapshots), issues,
            "source and snapshot maps use different paths")
    drift = []
    for original, digest in sources.items():
        try:
            rel = relative(snapshots[original])
            require(rel.parts[0] == "source_snapshot", issues,
                    f"snapshot not inside source_snapshot: {rel}")
            require(sha256(run / rel) == digest, issues,
                    f"source snapshot hash mismatch: {rel}")
            if not Path(original).is_file() or sha256(Path(original)) != digest:
                drift.append(original)
        except Exception as exc:
            issues.append(f"snapshot {original}: {type(exc).__name__}: {exc}")
    return {"snapshot_count": len(snapshots), "current_source_drift": drift}


def observed_raster(shape, lines: list[dict], issues: list[str]):
    raster = np.zeros(shape, np.uint8)
    h, w = shape
    for index, line in enumerate(lines):
        require(line.get("original_image_rechecked") is True, issues,
                f"observed_lines[{index}] lacks original-image recheck")
        points = line.get("observed_points_xy", [])
        require(line.get("point_count") == len(points), issues,
                f"observed_lines[{index}] point count mismatch")
        evidence = line.get("original_point_evidence", [])
        require(len(evidence) == len(points), issues,
                f"observed_lines[{index}] original evidence count mismatch")
        for j, xy in enumerate(points):
            try:
                x, y = map(int, xy[:2])
                require(0 <= x < w and 0 <= y < h, issues,
                        f"observed_lines[{index}][{j}] outside original image")
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(raster, (x, y), 1, 255, -1, cv2.LINE_8)
                if j < len(evidence):
                    require(evidence[j].get("xy") == [x, y], issues,
                            f"observed_lines[{index}][{j}] evidence coordinate mismatch")
            except Exception as exc:
                issues.append(f"observed_lines[{index}][{j}] invalid coordinate: {exc}")
    return raster


def audit_case(run: Path, row: dict, pilot: set[str], c3_hashes: dict,
               c4_hashes: dict):
    issues: list[str] = []
    rec = None
    rel = relative(row["output_file"])
    role, name, stem = rel.parent.name, rel.name, rel.stem
    try:
        original_path = DATA / rel
        require(sha256(original_path) == row["output_sha256"], issues,
                "original PNG hash differs from manifest")
        case_path = run / "记录" / "cases" / role / (stem+".json")
        case = read_json(case_path)
        finite(case, "case", issues)
        rec = case["record"]
        require(rec.get("status") == "completed", issues,
                "case failed or incomplete; preserved in output")
        for key in ("output_file", "camera_role", "source_group", "source_file",
                    "source_time_s", "dataset_use_status"):
            require(rec.get(key) == row[key], issues,
                    f"case record {key} differs from original manifest")
        require(rec.get("source_png_sha256") == row["output_sha256"] and
                rec.get("pilot_selected") == (row["output_file"] in pilot),
                issues, "original hash or pilot_selected differs from manifest/protocol")
        input_paths = (
            C4 / "roi" / "surface_masks" / role / name,
            C3 / "boundaries" / "records" / role / (stem+".json"),
            C4 / "boundaries" / "records" / role / (stem+".json"),
        )
        expected = (c4_hashes.get(str(input_paths[0])),
                    c3_hashes.get(str(input_paths[1])),
                    c4_hashes.get(str(input_paths[2])))
        dependency = rec.get("input_dependency_sha256", {})
        require(isinstance(dependency, dict) and
                set(dependency) == {str(p) for p in input_paths}, issues,
                "input dependency path set differs from C3/C4 frozen inputs")
        for path, digest in zip(input_paths, expected):
            require(digest is not None and sha256(path) == digest == dependency.get(str(path)),
                    issues, f"dependency hash differs from frozen inventory/record: {path}")
        original = image(original_path, cv2.IMREAD_COLOR)
        h, w = original.shape[:2]
        require(original.shape == (int(row["height"]), int(row["width"]), 3), issues,
                "original PNG dimensions differ from manifest")
        surface = image(input_paths[0], cv2.IMREAD_UNCHANGED)
        require(surface.shape == (h, w) and surface.dtype == np.uint8 and
                np.all((surface == 0) | (surface == 255)), issues,
                "C4 surface mask is not original-size uint8 binary")
        c3_record = read_json(input_paths[1])
        require(isinstance(rec.get("pitch_px"), (int, float)) and
                rec["pitch_px"] == c3_record["record"]["pitch_px"], issues,
                "pitch differs from C3 frozen record")
        masks = {}
        for group in MASK_GROUPS:
            path = run / "记录" / "masks" / group / role / name
            mask = image(path, cv2.IMREAD_UNCHANGED)
            masks[group] = mask
            require(mask.shape == (h, w) and mask.dtype == np.uint8 and
                    np.all((mask == 0) | (mask == 255)), issues,
                    f"{group} mask is not original-size uint8 binary")
        dark, observed, smooth = (masks[group] for group in MASK_GROUPS)
        if dark.shape == observed.shape == smooth.shape == surface.shape:
            require(not np.any(dark[surface == 0]), issues,
                    "dark ROI extends outside frozen C4 surface")
            require(not np.any(observed[dark == 0]), issues,
                    "observed mask extends outside dark ROI")
            require(not np.any(smooth[dark == 0]), issues,
                    "smooth mask extends outside dark ROI")
            require(rec.get("surface_pixels") == int(np.count_nonzero(surface)) and
                    rec.get("dark_roi_pixels") == int(np.count_nonzero(dark)) and
                    rec.get("observed_pixels") == int(np.count_nonzero(observed)) and
                    rec.get("smooth_pixels") == int(np.count_nonzero(smooth)), issues,
                    "case record pixel counts disagree with PNG masks")
        results = {}
        for group in IMAGE_GROUPS:
            path = run / "图像结果" / group / role / name
            result = image(path, cv2.IMREAD_UNCHANGED)
            results[group] = result
            require(result.shape == original.shape and result.dtype == np.uint8,
                    issues, f"{group} is not original-size uint8 BGR")
            if result.shape == original.shape and dark.shape == (h, w):
                require(not np.any(result[dark == 0]), issues,
                        f"{group} has nonzero pixels outside dark ROI")
        raw = results[IMAGE_GROUPS[0]]
        enhanced = results[IMAGE_GROUPS[1]]
        if raw.shape == original.shape and dark.shape == (h, w):
            require(np.array_equal(raw[dark > 0], original[dark > 0]), issues,
                    "raw ROI changed original pixels inside dark mask")
        if enhanced.shape == original.shape:
            require(np.array_equal(enhanced[:, :, 0], enhanced[:, :, 1]) and
                    np.array_equal(enhanced[:, :, 1], enhanced[:, :, 2]), issues,
                    "enhanced ROI is not equal-channel grayscale BGR")
        diag = rec.get("roi_and_enhancement", {})
        c3_diag = rec.get("c3_diagnostics", {})
        require(diag.get("dark_pixels") == rec.get("dark_roi_pixels") and
                diag.get("surface_pixels") == rec.get("surface_pixels"), issues,
                "ROI diagnostics pixel counts disagree with record")
        require(c3_diag.get("roi_pixels") == rec.get("dark_roi_pixels") and
                c3_diag.get("original_rechecked_points") ==
                sum(len(line.get("observed_points_xy", [])) for line in case.get("observed_lines", [])),
                issues, "C3 diagnostics disagree with ROI/observed records")
        require(c3_diag.get("roi_scope") == "entire_given_dark_mask_not_free_rope_corridor" and
                c3_diag.get("free_rope_required") is False and
                c3_diag.get("old_c3_used_as_seed") is False and
                c3_diag.get("physical_rope_identity_confirmed") is False,
                issues, "C3 diagnostic scope/physical claim invalid")
        require(rec.get("curve_count") == len(case.get("fit", {}).get("curves", [])),
                issues, "curve count differs from fit curves")
        if observed.shape == (h, w) and dark.shape == (h, w):
            regenerated = observed_raster((h, w), case.get("observed_lines", []), issues)
            regenerated[dark == 0] = 0
            require(np.array_equal(observed, regenerated), issues,
                    "observed PNG does not match rasterized original-rechecked points")
        return issues, rec
    except Exception as exc:
        issues.append(f"case audit exception: {type(exc).__name__}: {exc}")
        return issues, rec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Existing direct child pilot/all run or its absolute path")
    parser.add_argument("--output", required=True, type=Path,
                        help="New JSON report inside run; refusal to overwrite")
    args = parser.parse_args()
    run = Path(args.run)
    if not run.is_absolute():
        run = HERE / run
    if run.parent != HERE or not run.is_dir() or run.is_junction() or run.is_symlink():
        parser.error("--run must be a physical direct child of experiment root")
    output = args.output if args.output.is_absolute() else run / args.output
    if output.parent != run or output.exists() or output.suffix.lower() != ".json":
        parser.error("--output must be a new direct-child JSON path in run directory")
    cv2.setNumThreads(1)
    globals_: list[str] = []
    findings = []
    planned = 0
    snapshot = {}
    try:
        protocol, rows, pilot, c3_hashes, c4_hashes = frozen_inputs(globals_)
        context = read_json(run / "run_context.json")
        summary = read_json(run / "summary.json")
        finite(context, "run_context", globals_)
        finite(summary, "summary", globals_)
        mode = context.get("mode")
        require(mode in ("pilot", "all"), globals_, "invalid run mode")
        chosen = [r for r in rows if mode == "all" or r["output_file"] in pilot]
        planned = len(chosen)
        require(planned == (12 if mode == "pilot" else 400), globals_,
                "run denominator is not fixed12/400")
        names = [r["output_file"] for r in chosen]
        require(context.get("selection") == names, globals_,
                "run_context selection differs from frozen manifest order")
        require(context.get("version") == summary.get("version") == protocol.get("version") and
                summary.get("planned") == planned and
                summary.get("completed") == planned and summary.get("failed") == 0 and
                summary.get("source_unchanged") is True and summary.get("exit_code") == 0 and
                summary.get("images_per_output_group") == planned and
                summary.get("output_groups") == list(IMAGE_GROUPS) and
                summary.get("accuracy") is None and summary.get("html_generated") is False and
                summary.get("equipment_access") is False and summary.get("reserved_access") is False,
                globals_, "summary completion/scope contract failed")
        snapshot = source_snapshots(run, context, globals_)
        if mode == "all":
            freeze = read_json(HERE / "freeze_all.json")
            require(context.get("source_sha256") == freeze.get("source_sha256"),
                    globals_, "full run source differs from frozen choice")
        log = [json.loads(line) for line in
               (run / "run_log.jsonl").read_text(encoding="utf-8").splitlines()]
        require(len(log) == planned and [item.get("output_file") for item in log] == names,
                globals_, "run_log missing/duplicate/reordered input")
        for i, row in enumerate(chosen):
            issues, rec = audit_case(run, row, pilot, c3_hashes, c4_hashes)
            if rec is not None and i < len(log):
                logged = dict(log[i]); logged.pop("elapsed_s", None)
                require(logged == rec, issues, "run_log fields differ from case record")
            findings.append({"output_file": row["output_file"],
                             "passed": not issues, "issues": issues})
        require(summary.get("by_camera") == {
            camera: sum(row["camera_role"] == camera for row in chosen)
            for camera in ("left", "right")}, globals_,
            "summary camera denominators disagree with selection")
    except Exception as exc:
        globals_.append(f"global audit exception: {type(exc).__name__}: {exc}")
    passed = sum(case["passed"] for case in findings)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run": str(run), "audit_script_sha256": sha256(Path(__file__)),
        "planned": planned, "checked": len(findings), "passed": passed,
        "failed": len(findings)-passed, "global_errors": globals_,
        "source_snapshot": snapshot, "findings": findings,
        "meaning": "Structure, frozen bytes and pixel contracts only; no boundary accuracy or rope identity claim.",
        "reserved_inputs_read": False, "equipment_access": False,
        "exit_code": int(bool(globals_) or len(findings) != planned or passed != planned),
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in
                      ("planned", "checked", "passed", "failed", "global_errors", "exit_code")},
                     ensure_ascii=False), flush=True)
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
