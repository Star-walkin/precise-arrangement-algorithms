"""D2 暗纹实验的独立只读审计；不导入检测算法，也不评价视觉准确率。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent
D1 = ASSETS / "camera_condition_dark_roi_20261026_v1"
A1 = ASSETS / "camera_condition_incoming_rope_20261026_v1"
C3 = ASSETS / "camera_condition_grouped_conic_20261026_v6" / "all_001"
C4 = ASSETS / "camera_condition_full_surface_20261026_v7" / "all_001"
DATA = ASSETS / "camera_condition_images_20261009_v2"
SHAPE = (3000, 4096)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def image(path: Path, flags: int) -> np.ndarray:
    result = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    if result is None:
        raise ValueError(f"Cannot decode image: {path}")
    return result


def expect(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def file_entry(path: Path, run: Path) -> dict:
    return {"path": path.relative_to(run).as_posix(), "size": path.stat().st_size,
            "sha256": sha(path)}


def point_array(points: list, label: str) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if not len(values):
        return np.empty((0, 2), dtype=np.int32)
    expect(values.ndim == 2 and values.shape[1] == 2, f"{label}: xy shape")
    expect(np.isfinite(values).all(), f"{label}: nonfinite coordinates")
    rounded = np.rint(values).astype(np.int32)
    expect(np.allclose(values, rounded, atol=1e-6), f"{label}: nonpixel coordinates")
    expect(bool(np.all((rounded[:, 0] >= 0) & (rounded[:, 0] < SHAPE[1]) &
                       (rounded[:, 1] >= 0) & (rounded[:, 1] < SHAPE[0]))),
           f"{label}: outside image")
    return rounded


def check_sources(run: Path, context: dict, protocol: dict, failures: list[str]) -> None:
    source = context.get("source_sha256", {})
    snapshots = context.get("source_snapshot_paths", {})
    if not isinstance(source, dict) or not isinstance(snapshots, dict):
        failures.append("run_context source maps missing")
        return
    if set(source) != set(snapshots):
        failures.append("source/snapshot key mismatch")
    for raw_path, frozen_hash in source.items():
        try:
            original = Path(raw_path)
            snap_rel = Path(snapshots[raw_path])
            expect(not snap_rel.is_absolute() and ".." not in snap_rel.parts,
                   "unsafe snapshot path")
            expect(sha(original) == frozen_hash, f"changed source: {original}")
            expect(sha(run / snap_rel) == frozen_hash,
                   f"snapshot mismatch: {snap_rel}")
        except Exception as exc:
            failures.append(str(exc))
    for raw_path, frozen_hash in protocol["upstream_source_sha256"].items():
        try:
            expect(source.get(raw_path) == frozen_hash,
                   f"upstream source not in frozen run: {raw_path}")
            expect(sha(Path(raw_path)) == frozen_hash,
                   f"upstream source changed: {raw_path}")
        except Exception as exc:
            failures.append(str(exc))
    if context.get("mode") == "all":
        try:
            freeze_path = Path(context.get("freeze_manifest_path", "freeze_all.json"))
            freeze_path = freeze_path if freeze_path.is_absolute() else HERE / freeze_path
            expect(freeze_path.resolve().parent == HERE and
                   freeze_path.name.startswith("freeze_all") and
                   freeze_path.suffix == ".json",
                   "unsafe or missing full freeze path")
            expect(read_json(freeze_path)["source_sha256"] == source,
                   "full source differs from freeze_all")
        except Exception as exc:
            failures.append(str(exc))


def check_case(run: Path, row: dict, d1_inventory: dict, c3_inventory: dict,
               c4_inventory: dict, context: dict) -> dict:
    rel = Path(row["output_file"])
    role, name, stem = rel.parent.name, rel.name, rel.stem
    expect(role in ("left_camera", "right_camera"), "unexpected role path")
    expect(row["camera_role"] == role.split("_")[0], "role metadata mismatch")
    paths = {
        "overlay": run / "图像结果" / role / name,
        "observed": run / "记录" / "masks" / "observed" / role / name,
        "smooth": run / "记录" / "masks" / "smooth" / role / name,
        "inferred": run / "记录" / "masks" / "inferred" / role / name,
        "case": run / "记录" / "cases" / role / f"{stem}.json",
    }
    for label, path in paths.items():
        expect(path.is_file(), f"missing {label}: {path}")
    case = read_json(paths["case"])
    rec = case["record"]
    expect(rec["status"] == "completed", "case failed")
    expect(rec["output_file"] == row["output_file"], "case output_file mismatch")
    expect(rec["camera_role"] == row["camera_role"], "case camera mismatch")
    expect(rec["source_png_sha256"] == row["output_sha256"],
           "case original hash mismatch")
    expect(rec["version"] == context["version"], "case version mismatch")
    expect(rec["variant"] == context["variant"], "case variant mismatch")
    source_image = DATA / rel
    expect(sha(source_image) == row["output_sha256"], "original PNG changed")

    deps = rec["input_dependency_sha256"]
    expect(len(deps) == 6, "expected exactly six input dependencies")
    surface = C4 / "roi" / "surface_masks" / role / name
    c3case = C3 / "boundaries" / "records" / role / f"{stem}.json"
    c4case = C4 / "boundaries" / "records" / role / f"{stem}.json"
    dark_rel = Path("记录/masks/dark_roi") / role / name
    enhanced_rel = Path("图像结果/02_对比度增强ROI") / role / name
    d1case_rel = Path("记录/cases") / role / f"{stem}.json"
    expected_deps = {
        str(surface): c4_inventory[str(surface)],
        str(c3case): c3_inventory[str(c3case)],
        str(c4case): c4_inventory[str(c4case)],
        str(D1 / "all_001" / dark_rel): d1_inventory[dark_rel.as_posix()],
        str(D1 / "all_001" / enhanced_rel): d1_inventory[enhanced_rel.as_posix()],
        str(D1 / "all_001" / d1case_rel): d1_inventory[d1case_rel.as_posix()],
    }
    expect(deps == expected_deps, "input dependency map differs from inventories")
    for raw_path, frozen_hash in expected_deps.items():
        expect(sha(Path(raw_path)) == frozen_hash,
               f"input dependency changed: {raw_path}")

    dark = image(D1 / "all_001" / dark_rel, cv2.IMREAD_GRAYSCALE)
    overlay = image(paths["overlay"], cv2.IMREAD_COLOR)
    masks = {key: image(paths[key], cv2.IMREAD_GRAYSCALE)
             for key in ("observed", "smooth", "inferred")}
    expect(dark.shape == SHAPE and dark.dtype == np.uint8, "dark ROI shape")
    expect(overlay.shape == (*SHAPE, 3) and overlay.dtype == np.uint8,
           "overlay shape")
    expect(not np.any(overlay[dark == 0]), "overlay outside dark ROI nonzero")
    expect(rec["dark_roi_pixels"] == int(np.count_nonzero(dark)),
           "dark ROI pixel count mismatch")
    for label, mask in masks.items():
        expect(mask.shape == SHAPE and mask.dtype == np.uint8,
               f"{label} mask shape")
        expect(not np.any((mask != 0) & (mask != 255)),
               f"{label} nonbinary mask")
        expect(not np.any(mask[dark == 0]), f"{label} outside dark ROI")
        expect(rec[f"{label}_pixels"] == int(np.count_nonzero(mask)),
               f"{label} pixel count mismatch")
    expect(not np.any(masks["inferred"] & ~masks["smooth"]),
           "inferred pixels missing from smooth mask")

    lines = case.get("observed_lines", [])
    observed_count = sum(len(line.get("observed_points_xy", [])) for line in lines)
    expect(observed_count == rec["observed_point_count"],
           "observed point count mismatch")
    reconstructed = np.zeros(SHAPE, dtype=np.uint8)
    line_map = {}
    for line in lines:
        line_id = line["line_id"]
        expect(line_id not in line_map, "duplicate line_id")
        pts = point_array(line.get("observed_points_xy", []), line_id)
        line_map[line_id] = pts
        for x, y in pts:
            expect(dark[y, x] != 0, f"observed point outside dark ROI: {line_id}")
            cv2.circle(reconstructed, (int(x), int(y)), 1, 255, -1,
                       cv2.LINE_8)
    reconstructed[dark == 0] = 0
    expect(np.array_equal(reconstructed, masks["observed"]),
           "observed PNG differs from recorded observed points")

    curves = case.get("fit", {}).get("curves", [])
    expect(len(curves) == rec["curve_count"], "curve count mismatch")
    pitch = float(rec["pitch_px"])
    expect(math.isfinite(pitch) and pitch > 0, "invalid pitch")
    if rec["variant"] == "anchored":
        if "parameters" in rec["fit_diagnostics"]:
            expect(float(rec["fit_diagnostics"]["parameters"]["max_inferred_gap_pitch"]) == 3.0,
                   "anchored long-gap rule differs from frozen contract")
        else:
            expect(rec["fit_diagnostics"].get("status") == "no_anchors_or_dark_roi" and
                   not lines and not curves,
                   "missing gap parameters without explicit empty-anchor rejection")
    anchored_smooth = np.zeros(SHAPE, dtype=np.uint8)
    anchored_inferred = np.zeros(SHAPE, dtype=np.uint8)
    for curve in curves:
        curve_id = curve["curve_id"]
        pts = point_array(curve.get("sampled_points_xy", []), curve_id)
        states = curve.get("point_states", ["supported_fit"] * len(pts))
        expect(len(states) == len(pts), f"{curve_id}: states length mismatch")
        permitted = (("observed_supported_fit", "inferred_short_gap")
                     if rec["variant"] == "anchored" else
                     ("supported_fit", "inferred_short_gap"))
        expect(all(s in permitted for s in states),
               f"{curve_id}: unknown state")
        model = curve.get("model", {})
        if isinstance(model, dict):
            for field in ("xlo", "xhi"):
                if field in model:
                    expect(math.isfinite(float(model[field])), f"{curve_id}: {field}")
            for value in model.get("control_y_px", []):
                expect(math.isfinite(float(value)), f"{curve_id}: control nonfinite")
        if len(pts):
            expect(np.all(np.diff(pts[:, 0]) > 0),
                   f"{curve_id}: x not strictly increasing")
            expect(np.all(dark[pts[:, 1], pts[:, 0]] != 0),
                   f"{curve_id}: sample outside dark ROI")
            if rec["variant"] != "anchored":
                expect(np.all(masks["smooth"][pts[:, 1], pts[:, 0]] != 0),
                       f"{curve_id}: sample absent from smooth mask")
                infer = np.array([s == "inferred_short_gap" for s in states])
                if np.any(infer):
                    expect(np.all(masks["inferred"][pts[infer, 1], pts[infer, 0]]),
                           f"{curve_id}: inferred sample absent from inferred mask")
        if rec["variant"] == "d2":
            group_id = curve["group_id"]
            expect(group_id in line_map, f"{curve_id}: no observed group")
            x_observed = sorted(set(line_map[group_id][:, 0].tolist()))
            expect(len(x_observed) >= 2, f"{curve_id}: no observed span")
            expect(int(pts[0, 0]) >= x_observed[0] and
                   int(pts[-1, 0]) <= x_observed[-1],
                   f"{curve_id}: outside observed x span")
            max_gap = 4 * pitch
            recorded_limit = curve["gap_diagnostics"]["maximum_drawable_gap_px"]
            expect(abs(float(recorded_limit) - max_gap) < 1e-6,
                   f"{curve_id}: gap limit mismatch")
            for a, b in zip(x_observed[:-1], x_observed[1:]):
                if b - a > max_gap:
                    expect(not np.any((pts[:, 0] > a) & (pts[:, 0] < b)),
                           f"{curve_id}: sampled across long unsupported gap")
            for (x, _), state in zip(pts, states):
                in_gap = any(a < x < b and b - a > 1.25 * pitch
                             for a, b in zip(x_observed[:-1], x_observed[1:]))
                expect((state == "inferred_short_gap") == in_gap,
                       f"{curve_id}: inferred state inconsistent with support")
        elif rec["variant"] == "anchored":
            expect(curve_id in line_map, f"{curve_id}: no observed line")
            observed_x = np.unique(line_map[curve_id][:, 0])
            expect(len(observed_x) >= 2, f"{curve_id}: no observed span")
            max_gap = 3.0 * pitch
            if len(pts):
                expect(pts[0, 0] >= observed_x[0] and
                       pts[-1, 0] <= observed_x[-1],
                       f"{curve_id}: outside observed x span")
                right = np.searchsorted(observed_x, pts[:, 0], side="left")
                right = np.clip(right, 1, len(observed_x) - 1)
                left_x, right_x = observed_x[right - 1], observed_x[right]
                inside = (pts[:, 0] > left_x) & (pts[:, 0] < right_x)
                expect(np.all(~inside | (right_x - left_x <= max_gap + 1e-6)),
                       f"{curve_id}: sample in unsupported long gap")
            for i in range(len(pts) - 1):
                a, b = pts[i], pts[i + 1]
                if b[0] - a[0] != 1 or abs(int(b[1]) - int(a[1])) > 4:
                    continue
                cv2.line(anchored_smooth, tuple(map(int, a)),
                         tuple(map(int, b)), 255, 1)
                if states[i] == "inferred_short_gap" or states[i + 1] == "inferred_short_gap":
                    cv2.line(anchored_inferred, tuple(map(int, a)),
                             tuple(map(int, b)), 255, 1)

    if rec["variant"] == "anchored":
        anchored_smooth[dark == 0] = 0
        anchored_inferred[dark == 0] = 0
        anchored_inferred[masks["observed"] > 0] = 0
        expect(np.array_equal(anchored_smooth, masks["smooth"]),
               "anchored smooth PNG differs from recorded curve segments")
        expect(np.array_equal(anchored_inferred, masks["inferred"]),
               "anchored inferred PNG differs from recorded curve states")

    return {"output_file": row["output_file"], "files":
            [file_entry(paths[k], run) for k in
             ("overlay", "observed", "smooth", "inferred", "case")],
            "observed_points": observed_count, "curves": len(curves)}


def audit(run: Path) -> tuple[dict, dict]:
    context = read_json(run / "run_context.json")
    summary = read_json(run / "summary.json")
    snapshot_rel = context["source_snapshot_paths"].get(str(HERE / "protocol.json"))
    expect(snapshot_rel is not None, "run protocol snapshot missing")
    protocol = read_json(run / snapshot_rel)
    d1_protocol = read_json(D1 / "protocol.json")
    a1_protocol = read_json(A1 / "protocol.json")
    global_errors: list[str] = []
    expect(context["mode"] in ("pilot", "all"), "unexpected run mode")
    expect(context["variant"] in ("d2", "threshold_only", "anchored"),
           "unexpected variant")
    expect(context["version"] in ("DR-D2.0.0", "DR-D2.1.0", "DR-D2.1.1"),
           "unexpected run version")
    expect(protocol["version"] == context["version"],
           "run version differs from protocol snapshot")
    expect(summary["version"] == context["version"], "summary version mismatch")
    check_sources(run, context, protocol, global_errors)
    fixed_files = [
        (DATA / "manifest.csv", d1_protocol["input_manifest_sha256"]),
        (DATA / "manifest.csv", a1_protocol["input_manifest_sha256"]),
        (D1 / "all_001" / "output_manifest.json",
         protocol["d1_output_manifest_sha256"]),
        (C3 / "integrity_audit.json", a1_protocol["c3_integrity_sha256"]),
        (C4 / "output_manifest.json", a1_protocol["c4_output_manifest_sha256"]),
    ]
    for path, frozen_hash in fixed_files:
        try:
            expect(sha(path) == frozen_hash, f"frozen dependency changed: {path}")
        except Exception as exc:
            global_errors.append(str(exc))
    with (DATA / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    expect(len(all_rows) == 400, "fixed manifest does not contain 400 inputs")
    expect(Counter(row["camera_role"] for row in all_rows) ==
           {"left": 200, "right": 200}, "fixed camera split differs from 200+200")
    expect(all("20260925" not in row["output_file"] + row["source_file"]
               for row in all_rows), "reserved date in manifest")
    expect(all("reserved" not in row["dataset_use_status"].lower() and
               "heldout" not in row["dataset_use_status"].lower()
               for row in all_rows), "reserved status in manifest")
    if context["mode"] == "pilot":
        expected_set = set(protocol["pilot_images"])
        expect(expected_set == set(d1_protocol["pilot_images"]),
               "D2 and D1 pilot lists differ")
        rows = [row for row in all_rows if row["output_file"] in expected_set]
        expect(len(rows) == 12, "fixed pilot is not 12 inputs")
    else:
        rows = all_rows
    expect(context["selection"] == [r["output_file"] for r in rows],
           "run selection differs from fixed input order")
    expect(summary["planned"] == len(rows), "summary planned mismatch")
    expect(summary["completed"] + summary["failed"] == len(rows),
           "summary denominator mismatch")
    expect(summary["variant"] == context["variant"], "summary variant mismatch")
    expect(summary["by_camera"] ==
           {r: sum(row["camera_role"] == r for row in rows)
            for r in ("left", "right")}, "summary camera counts mismatch")
    if summary["failed"] != 0 or not summary["source_unchanged"] or summary["exit_code"] != 0:
        global_errors.append("run summary reports failed source/inputs/exit")

    d1_inventory = {entry["path"]: entry["sha256"] for entry in
                    read_json(D1 / "all_001" / "output_manifest.json")["files"]}
    c3_inventory = {entry["path"]: entry["sha256"] for entry in
                    read_json(C3 / "integrity_audit.json")["file_manifest"]}
    c4_inventory = {entry["absolute_path"]: entry["sha256"]
                    for case in read_json(C4 / "output_manifest.json")["cases"]
                    for entry in case["files"]}
    passed, failed, output_cases = 0, [], []
    for index, row in enumerate(rows, 1):
        try:
            output_cases.append(check_case(run, row, d1_inventory,
                                           c3_inventory, c4_inventory,
                                           context))
            passed += 1
        except Exception as exc:
            failed.append({"output_file": row["output_file"], "error": str(exc)})
        if index % 25 == 0 or index == len(rows):
            print(f"audit {index}/{len(rows)} passed={passed} failed={len(failed)}",
                  flush=True)
    report = {
        "auditor": "independent D2 file/pixel/hash audit; no accuracy claim",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run": str(run), "mode": context["mode"], "variant": context["variant"],
        "planned": len(rows), "passed": passed, "failed": len(failed),
        "global_errors": global_errors, "failures": failed,
        "status": "passed" if passed == len(rows) and not global_errors else "failed",
        "input_manifest_sha256": sha(DATA / "manifest.csv"),
        "d1_output_manifest_sha256": sha(D1 / "all_001" / "output_manifest.json"),
        "source_snapshot_checked": len(context["source_snapshot_paths"]),
        "scope": "Structural integrity only; visual grooves and physical identity are not ground truth validated.",
    }
    output_manifest = {
        "version": context["version"], "run": str(run), "planned": len(rows),
        "cases": output_cases, "case_count": len(output_cases),
        "file_count": sum(len(c["files"]) for c in output_cases),
        "incomplete_cases": [f["output_file"] for f in failed],
    }
    return report, output_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        help="New audit JSON (default: RUN/independent_audit.json)")
    args = parser.parse_args()
    run = args.run.resolve()
    expect(run.parent == HERE and run.is_dir(), "run must be an existing direct child")
    output = args.output.resolve() if args.output else run / "independent_audit.json"
    manifest = run / "output_manifest.json"
    expect(output.parent == run, "audit output must be inside run directory")
    expect(output != manifest, "audit and manifest paths must differ")
    expect(not output.exists() and not manifest.exists(), "refuse to overwrite audit/manifest")
    try:
        report, files = audit(run)
    except Exception as exc:
        report = {"auditor": "independent D2 file/pixel/hash audit",
                  "created_utc": datetime.now(timezone.utc).isoformat(),
                  "run": str(run), "status": "failed", "planned": None,
                  "passed": 0, "failed": None, "global_errors": [str(exc)],
                  "failures": [], "scope": "Structural integrity only; no accuracy claim."}
        files = {"version": None, "run": str(run), "planned": None,
                 "cases": [], "case_count": 0, "file_count": 0,
                 "incomplete_cases": [], "error": str(exc)}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(files, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in
                      ("run", "planned", "passed", "failed", "status", "global_errors")},
                     ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
