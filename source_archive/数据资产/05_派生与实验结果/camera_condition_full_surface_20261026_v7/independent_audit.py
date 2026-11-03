"""Independent file, pixel, and whole-group audit of a frozen C4 run.

This script reads existing open-development outputs.  It never calls the
detector or fitter, modifies images, or accepts a pre-existing report path.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent
PROTOCOL = HERE / "evaluation_protocol.json"
QA = ASSETS / "camera_condition_bright_rope_boundaries_20261009_v1" / "qa_plan.csv"
MANIFEST = ASSETS / "camera_condition_images_20261009_v2" / "manifest.csv"
SAM_MANIFEST = (ASSETS / "camera_condition_sam2_20261009_v1" / "all_001" /
                "independent_final_audit_001" / "output_manifest.csv")
B1 = ASSETS / "camera_condition_bright_rope_boundaries_20261025_v3" / "all_001"
C3 = ASSETS / "camera_condition_grouped_conic_20261026_v6" / "all_001"
MASK_DIRS = ("roi/effective_masks", "boundaries/dark_observed", "boundaries/modeled",
             "boundaries/inferred", "boundaries/simple_c3_extension",
             "boundaries/incoming_rope_occlusion", "boundaries/unknown_exclusion")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path, flag: int) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), flag)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return image


def mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    value = read(path, cv2.IMREAD_GRAYSCALE)
    if value.dtype != np.uint8 or (shape is not None and value.shape != shape):
        raise ValueError(f"Invalid mask shape/type: {path}")
    if np.any((value != 0) & (value != 255)):
        raise ValueError(f"Mask is not uniformly 0/255: {path}")
    return value


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_model(group: dict, pitch: float, errors: list[str]) -> float:
    name = group.get("group_id", "unnamed_group")
    points = np.asarray(group.get("observed_points_xy", []), dtype=float)
    weights = np.asarray(group.get("fit_weights", []), dtype=float)
    saved_residual = np.asarray(group.get("residuals_px", []), dtype=float)
    kinds = group.get("point_kind", [])
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) == 0 or
            len(weights) != len(points) or len(saved_residual) != len(points) or
            len(kinds) != len(points)):
        errors.append(f"{name}: point/weight/residual/kind lengths differ")
        return math.inf
    if not (np.isfinite(points).all() and np.isfinite(weights).all() and
            np.isfinite(saved_residual).all() and np.all(weights > 0)):
        errors.append(f"{name}: nonfinite point/residual or nonpositive fit weight")
        return math.inf
    model = group.get("model", {})
    try:
        k, b, c = (float(model[key]) for key in
                   ("normalized_k", "normalized_b", "normalized_c"))
        center, scale = (float(model[key]) for key in
                         ("u_center_x_px", "u_scale_x_px"))
    except (KeyError, TypeError, ValueError):
        errors.append(f"{name}: missing whole-group model coefficients")
        return math.inf
    if not all(math.isfinite(value) for value in (k, b, c, center, scale)) or scale <= 0 or c < -1e-9:
        errors.append(f"{name}: invalid conic coefficients")
        return math.inf
    u = (points[:, 0] - center) / scale
    design = np.column_stack((np.ones(len(points)), u, u * u))
    residual = points[:, 1] - design @ np.array([k, b, c])
    residual_difference = float(np.max(np.abs(residual - saved_residual)))
    if residual_difference > 1e-5:
        errors.append(f"{name}: recorded residuals do not match its one model")
    regularization = group.get("fit_regularization") or {}
    dark_count = sum(kind == "trusted_dark_valley" for kind in kinds)
    bright_count = sum(kind == "bright_c3_yellow" for kind in kinds)
    if dark_count != group.get("trusted_dark_observed_count") or bright_count != group.get("bright_observed_count"):
        errors.append(f"{name}: bright/dark point-kind accounting differs")
    if dark_count:
        try:
            lb = float(regularization["lambda_b"])
            lc = float(regularization["lambda_c"])
            pb = float(regularization["prior_b"])
            pc = float(regularization["prior_c"])
            active = bool(regularization["c_nonnegative_active"])
            gradient = design.T @ (weights * (design @ np.array([k, b, c]) - points[:, 1]))
            gradient[1] += lb * (b - pb)
            gradient[2] += lc * (c - pc)
            free = gradient[:2] if active else gradient
            normalized = float(np.max(np.abs(free)) / max(1.0, float(np.sum(weights))))
            if lb < 0 or lc < 0 or normalized > 1e-4 or (active and abs(c) > 1e-8):
                errors.append(f"{name}: all-point weighted normal equations/KKT not satisfied")
        except (KeyError, TypeError, ValueError):
            errors.append(f"{name}: missing joint-fit objective provenance")
    else:
        seed = group.get("bright_seed_model")
        if regularization.get("new_optimization_performed") is not False or model != seed:
            errors.append(f"{name}: bright-only group did not retain C3 model")
    return residual_difference


def audit_case(run: Path, row: dict, sam_row: dict, roi_policy: str) -> tuple[dict, np.ndarray]:
    rel = Path(row["output_file"])
    role, name, stem = rel.parent.name, rel.name, rel.stem
    errors: list[str] = []
    warnings: list[str] = []
    source = ASSETS / "camera_condition_images_20261009_v2" / rel
    surface_source = Path(sam_row["mask_path"])
    masked_source = Path(sam_row["masked_path"])
    paths = {directory: run / directory / role / name for directory in MASK_DIRS}
    paths.update({"surface": run / "roi/surface_masks" / role / name,
                  "surface_masked": run / "roi/surface_masked" / role / name,
                  "bright": run / "roi/bright_reference_masks" / role / name,
                  "old_observed": run / "boundaries/original_observed" / role / name,
                  "record": run / "boundaries/records" / role / f"{stem}.json"})
    if roi_policy == "SAM2_v2_union_B1_bright":
        paths.update({"sam_reference": run / "roi/sam_reference_masks" / role / name,
                      "sam_reference_masked": run / "roi/sam_reference_masked" / role / name})
    for directory, filename in (("overlays", name), ("previews", stem + ".jpg"),
                                ("smooth_previews", stem + ".jpg"),
                                ("group_overlays", stem + ".jpg")):
        visual_path = run / directory / role / filename
        if not visual_path.is_file() or visual_path.stat().st_size == 0:
            errors.append(f"missing/empty visual output {directory}")
    if digest(source) != row["output_sha256"]:
        errors.append("input PNG hash differs from fixed 400 manifest")
    if digest(surface_source) != sam_row["mask_sha256"]:
        errors.append("SAM2 source mask differs from frozen SAM output manifest")
    if digest(masked_source) != sam_row["masked_sha256"]:
        errors.append("SAM2 source masked RGB differs from frozen SAM output manifest")
    required_copies = [("bright", B1 / "roi/reference_masks" / role / name),
                       ("old_observed", B1 / "boundaries/observed" / role / name)]
    if roi_policy == "SAM2_v2_union_B1_bright":
        required_copies += [("sam_reference", surface_source),
                            ("sam_reference_masked", masked_source)]
    else:
        required_copies += [("surface", surface_source), ("surface_masked", masked_source)]
    for output_key, upstream in required_copies:
        if digest(paths[output_key]) != digest(upstream):
            errors.append(f"{output_key} copy is not byte-exact")
    image = read(source, cv2.IMREAD_COLOR)
    shape = image.shape[:2]
    surface = mask(paths["surface"], shape)
    bright = mask(paths["bright"], shape)
    raw_sam = mask(surface_source, shape)
    union_added = int(np.count_nonzero((bright > 0) & (raw_sam == 0)))
    if roi_policy == "SAM2_v2_union_B1_bright":
        if not np.array_equal(surface, np.maximum(raw_sam, bright)):
            errors.append("working surface is not exact SAM2 union B1 bright ROI")
    elif roi_policy == "SAM2_v2_raw":
        if not np.array_equal(surface, raw_sam):
            errors.append("raw-policy surface differs from SAM2 mask")
    else:
        errors.append(f"unrecognized ROI policy: {roi_policy}")
    old_observed = mask(paths["old_observed"], shape)
    values = {key: mask(paths[key], shape) for key in MASK_DIRS}
    masked = read(paths["surface_masked"], cv2.IMREAD_COLOR)
    if masked.shape != image.shape:
        errors.append("surface masked RGB dimensions differ from original")
    elif (not np.array_equal(masked[surface > 0], image[surface > 0]) or
          np.any(masked[surface == 0] != 0)):
        errors.append("surface masked RGB does not preserve original pixels/black background")
    if np.any((bright > 0) & (surface == 0)):
        errors.append("B1 bright ROI extends outside SAM2 surface")
    old_rope = mask(B1 / "boundaries/incoming_rope_occlusion" / role / name, shape)
    old_unknown = mask(B1 / "boundaries/unknown_exclusion" / role / name, shape)
    rope, unknown = values["boundaries/incoming_rope_occlusion"], values["boundaries/unknown_exclusion"]
    if not np.array_equal(rope[bright > 0], old_rope[bright > 0]):
        errors.append("old supported rope mask changed in bright ROI")
    if not np.array_equal(unknown[bright > 0], old_unknown[bright > 0]):
        errors.append("old unknown rope exclusion changed in bright ROI")
    if np.any((rope > 0) & (unknown > 0)) or np.any(((rope > 0) | (unknown > 0)) & (surface == 0)):
        errors.append("supported/unknown overlap or leave SAM2 ROI")
    expected_effective = np.where((surface > 0) & (rope == 0) & (unknown == 0), 255, 0).astype(np.uint8)
    effective = values["roi/effective_masks"]
    if not np.array_equal(effective, expected_effective):
        errors.append("effective ROI differs from SAM2 minus exclusions")
    for key in ("boundaries/dark_observed", "boundaries/modeled", "boundaries/inferred",
                "boundaries/simple_c3_extension"):
        if np.any((values[key] > 0) & (effective == 0)):
            errors.append(f"{key} leaves effective surface or crosses rope exclusion")
    if np.any((values["boundaries/dark_observed"] > 0) & (bright > 0)):
        errors.append("dark observed point appears in B1 bright ROI")
    if np.any((values["boundaries/inferred"] > 0) & (values["boundaries/modeled"] == 0)):
        errors.append("inferred mask not subset of modeled mask")
    if np.any((old_observed > 0) & (bright == 0)):
        warnings.append("B1 original yellow appears outside its reference ROI")
    record = load_json(paths["record"])
    meta, fit = record["record"], record["fit"]
    pitch = float(meta["pitch_px"])
    if not 8 <= pitch <= 70 or not math.isfinite(pitch):
        errors.append("pitch is implausible")
    if meta["output_file"] != row["output_file"] or meta["camera_role"] != row["camera_role"]:
        errors.append("record identity differs from input manifest")
    if roi_policy == "SAM2_v2_union_B1_bright":
        if meta.get("roi_policy") != roi_policy or meta.get("sam_raw_pixels") != int(np.count_nonzero(raw_sam)) or \
                meta.get("union_added_bright_pixels") != union_added:
            errors.append("ROI policy/raw/union provenance differs from actual masks")
    for field, actual in (("surface_pixels", surface), ("expanded_pixels", np.where(bright == 0, surface, 0)),
                          ("dark_observed_pixels", values["boundaries/dark_observed"]),
                          ("modeled_pixels", values["boundaries/modeled"]),
                          ("inferred_pixels", values["boundaries/inferred"]),
                          ("rope_pixels", rope), ("unknown_pixels", unknown)):
        if int(np.count_nonzero(actual)) != meta.get(field):
            errors.append(f"record {field} differs from actual mask")
    if meta.get("complete_free_rope_claim") is not False:
        errors.append("record incorrectly claims complete free-rope detection")
    groups, curves = fit.get("groups", []), fit.get("curves", [])
    group_ids = [group.get("group_id") for group in groups]
    curve_ids = [curve.get("group_id") for curve in curves]
    if len(set(group_ids)) != len(group_ids) or len(set(curve_ids)) != len(curve_ids):
        errors.append("group or accepted-curve ID repeated")
    if set(group_ids) != set(curve_ids):
        errors.append("accepted group and curve IDs are not one-to-one")
    by_group = {group.get("group_id"): group for group in groups}
    dark_raster = np.zeros(shape, np.uint8)
    modeled_raster = np.zeros(shape, np.uint8)
    inferred_raster = np.zeros(shape, np.uint8)
    max_residual_difference = 0.0
    for group in groups:
        max_residual_difference = max(max_residual_difference, _check_model(group, pitch, errors))
        for kind, point in zip(group.get("point_kind", []), group.get("observed_points_xy", [])):
            x, y = int(point[0]), int(point[1])
            if not (0 <= x < shape[1] and 0 <= y < shape[0]):
                errors.append(f"{group.get('group_id')}: observed point outside image")
                continue
            if kind == "trusted_dark_valley":
                if bright[y, x] or not effective[y, x]:
                    errors.append(f"{group.get('group_id')}: dark point outside dark effective ROI")
                dark_raster[y, x] = 255
            elif kind != "bright_c3_yellow":
                errors.append(f"{group.get('group_id')}: unrecognized observed point kind")
    for curve in curves:
        group_id = curve.get("group_id")
        group = by_group.get(group_id)
        if group is None or group.get("model") != curve.get("model"):
            errors.append(f"{group_id}: group/curve coefficient set differs")
        points = curve.get("sampled_points_xy", [])
        flags = curve.get("inferred_flags", [])
        if len(points) != len(flags) or len({int(point[0]) for point in points}) != len(points):
            errors.append(f"{group_id}: curve sample/flag length or unique columns invalid")
        for point, inferred in zip(points, flags):
            x, y = map(int, point)
            if not (0 <= x < shape[1] and 0 <= y < shape[0]):
                errors.append(f"{group_id}: sampled curve point outside image")
                continue
            if not effective[y, x]:
                errors.append(f"{group_id}: sampled curve point outside effective ROI")
            modeled_raster[y, x] = 255
            if inferred:
                inferred_raster[y, x] = 255
    for label, raster in (("dark_observed", dark_raster), ("modeled", modeled_raster),
                          ("inferred", inferred_raster)):
        if not np.array_equal(raster, values[f"boundaries/{label}"]):
            errors.append(f"{label} mask differs from retained group/curve samples")
    columns: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for curve in curves:
        for x, y in curve.get("sampled_points_xy", []):
            columns[int(x)].append((curve["group_id"], int(y)))
    close_pairs = 0
    exact_collisions = 0
    for items in columns.values():
        items.sort(key=lambda item: item[1])
        for a, b in zip(items[:-1], items[1:]):
            exact_collisions += int(a[1] == b[1])
            close_pairs += int(b[1] - a[1] < .45 * pitch)
    if exact_collisions:
        errors.append(f"accepted curves share {exact_collisions} identical pixels")
    if close_pairs:
        warnings.append(f"accepted curves have {close_pairs} too-close same-column pairs")
    curve_columns = {curve["group_id"]: {int(x): int(y) for x, y in curve.get("sampled_points_xy", [])}
                     for curve in curves}
    crossing_pairs = []
    for i, first in enumerate(curves):
        for second in curves[i + 1:]:
            a, b = curve_columns[first["group_id"]], curve_columns[second["group_id"]]
            signs = {int(a[x] > b[x]) for x in a.keys() & b.keys() if a[x] != b[x]}
            if len(signs) > 1:
                crossing_pairs.append([first["group_id"], second["group_id"]])
    if crossing_pairs:
        errors.append(f"accepted curves reverse vertical order: {crossing_pairs[:8]}")
    dark = (surface > 0) & (bright == 0)
    outside_modeled = (values["boundaries/modeled"] > 0) & dark
    distances = np.empty(0, dtype=np.float32)
    if np.any(outside_modeled):
        distance_map = cv2.distanceTransform((bright == 0).astype(np.uint8), cv2.DIST_L2, 5)
        distances = (distance_map[outside_modeled] / pitch).astype(np.float32)
    finding = {"output_file": row["output_file"], "camera_role": row["camera_role"],
               "source_group": row["source_group"], "status": "pass" if not errors else "fail",
               "errors": errors, "warnings": warnings, "pitch_px": pitch,
               "roi_policy": roi_policy, "sam_raw_pixels": int(np.count_nonzero(raw_sam)),
               "union_added_bright_pixels": union_added,
               "surface_pixels": int(np.count_nonzero(surface)),
               "expanded_dark_surface_pixels": int(np.count_nonzero(dark)),
               "dark_observed_pixels": int(np.count_nonzero(values["boundaries/dark_observed"])),
               "modeled_outside_bright_pixels": int(np.count_nonzero(outside_modeled)),
               "outside_bright_supported_pixels": int(np.count_nonzero((rope > 0) & (bright == 0))),
               "outside_bright_unknown_pixels": int(np.count_nonzero((unknown > 0) & (bright == 0))),
               "accepted_groups": len(groups), "accepted_curves": len(curves),
               "group_residual_replay_max_px": max_residual_difference,
               "close_pair_columns": close_pairs,
               "crossing_pairs": crossing_pairs,
               "dark_extension_distance_pitch_median": float(np.median(distances)) if len(distances) else None,
               "dark_extension_distance_pitch_p90": float(np.quantile(distances, .9)) if len(distances) else None,
               "dark_extension_distance_pitch_max": float(np.max(distances)) if len(distances) else None}
    return finding, distances


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    protocol = load_json(PROTOCOL)
    fixed = protocol["fixed_inputs"]
    for path, expected in ((QA, fixed["qa_plan_sha256"]),
                           (MANIFEST, fixed["full_400_manifest_sha256"]),
                           (SAM_MANIFEST, fixed["sam2_final_output_manifest_sha256"])):
        if digest(path) != expected:
            raise ValueError(f"Frozen input manifest changed: {path}")
    context, summary = load_json(run / "run_context.json"), load_json(run / "summary.json")
    mode = context["mode"]
    roi_policy = context.get("roi_policy", "SAM2_v2_raw")
    if mode not in ("pilot", "all"):
        raise ValueError(f"Unknown run mode {mode}")
    with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with QA.open(encoding="utf-8-sig", newline="") as handle:
        qa = {row["output_file"] for row in csv.DictReader(handle)}
    with SAM_MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        sam = {"images/" + row["output_file"]: row for row in csv.DictReader(handle)}
    expected = [row for row in rows if mode == "all" or row["output_file"] in qa]
    expected_count = 400 if mode == "all" else fixed["fixed_qa_images"]
    global_errors: list[str] = []
    if len(expected) != expected_count or len({row["output_file"] for row in expected}) != expected_count:
        global_errors.append("manifest selection count/identity differs from frozen protocol")
    if len(sam) != 400:
        global_errors.append("SAM final manifest does not have 400 distinct images")
    if summary.get("planned") != expected_count or summary.get("completed") != expected_count or summary.get("failed") != 0:
        global_errors.append("runner did not complete all fixed images")
    if summary.get("source_unchanged") is not True or summary.get("exit_code") != 0:
        global_errors.append("runner sources changed or exited nonzero")
    log_path = run / "run_log.jsonl"
    logged = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    if len(logged) != expected_count or {item["output_file"] for item in logged} != {row["output_file"] for row in expected}:
        global_errors.append("run log does not cover the exact fixed image set")
    findings = []
    distance_groups: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for index, row in enumerate(expected, 1):
        source = row["output_file"]
        try:
            finding, distances = audit_case(run, row, sam[source], roi_policy)
        except Exception as exc:
            finding = {"output_file": source, "camera_role": row["camera_role"],
                       "source_group": row["source_group"], "status": "fail",
                       "errors": [repr(exc)], "warnings": []}
            distances = np.empty(0, dtype=np.float32)
        findings.append(finding)
        if len(distances):
            distance_groups[(row["camera_role"], row["source_group"])].append(distances)
        if index % 10 == 0 or index == len(expected):
            print(f"Audited {index}/{len(expected)}", flush=True)
    grouped = []
    for (role, group), arrays in sorted(distance_groups.items()):
        values = np.concatenate(arrays)
        grouped.append({"camera_role": role, "source_group": group,
                        "modeled_dark_pixels": int(len(values)),
                        "distance_pitch_median": float(np.median(values)),
                        "distance_pitch_p90": float(np.quantile(values, .9)),
                        "distance_pitch_max": float(np.max(values))})
    passed = sum(item["status"] == "pass" for item in findings)
    report = {"run": str(run), "mode": mode, "roi_policy": roi_policy, "planned": expected_count,
              "passed": passed, "failed": len(findings) - passed,
              "global_errors": global_errors,
              "protocol_sha256": digest(PROTOCOL), "audit_script_sha256": digest(Path(__file__)),
              "input_manifest_sha256": digest(MANIFEST), "sam_manifest_sha256": digest(SAM_MANIFEST),
              "source_photographs_decoded": True, "reserved_inputs_accessed": False,
              "equipment_connected": False,
              "by_camera": {role: {"images": sum(item["camera_role"] == role for item in findings),
                                   "passed": sum(item["camera_role"] == role and item["status"] == "pass"
                                                 for item in findings),
                                   "modeled_dark_pixels": sum(item.get("modeled_outside_bright_pixels", 0)
                                                              for item in findings if item["camera_role"] == role),
                                   "outside_bright_supported_pixels": sum(item.get("outside_bright_supported_pixels", 0)
                                                                           for item in findings if item["camera_role"] == role),
                                   "outside_bright_unknown_pixels": sum(item.get("outside_bright_unknown_pixels", 0)
                                                                         for item in findings if item["camera_role"] == role)}
                            for role in ("left", "right")},
              "dark_extension_distance_by_camera_group": grouped,
              "findings": findings,
              "interpretation": "Structural and pixel integrity only. Dark rope-edge correctness, material identity, and physical tangency are not labeled."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": passed, "failed": report["failed"],
                      "global_errors": global_errors}, ensure_ascii=False), flush=True)
    return int(report["failed"] != 0 or bool(global_errors))


if __name__ == "__main__":
    sys.exit(main())
