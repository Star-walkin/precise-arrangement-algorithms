"""Frozen QA38 C4 evaluation: old bright compatibility and one-sided yellow holdout.

The edge proxy fits C3 again from freshly numbered visible yellow fragments. It
passes no image pixels or dark observations to C4. Predictions are written and
hashed before the withheld-reference file is opened by the scorer. This is an
open-development consistency test, not a label for dark boundaries or tangency.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import shutil
import sys

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent
B1 = ASSETS / "camera_condition_bright_rope_boundaries_20261025_v3" / "all_001"
C3_ROOT = ASSETS / "camera_condition_grouped_conic_20261026_v6"
C3 = C3_ROOT / "all_001"
C3_MODULE = C3 / "source_snapshot" / "conic_family.py"
CORE = HERE / "full_surface_core.py"
PROTOCOL = HERE / "evaluation_protocol.json"
QA = ASSETS / "camera_condition_bright_rope_boundaries_20261009_v1" / "qa_plan.csv"
MANIFEST = ASSETS / "camera_condition_images_20261009_v2" / "manifest.csv"
SAM_MANIFEST = (ASSETS / "camera_condition_sam2_20261009_v1" / "all_001" /
                "independent_final_audit_001" / "output_manifest.csv")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(1 << 20), b""):
            result.update(data)
    return result.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_line(handle, value: dict) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def binary(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    value = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
    if value is None or value.dtype != np.uint8 or (shape is not None and value.shape != shape):
        raise ValueError(f"Invalid mask: {path}")
    if np.any((value != 0) & (value != 255)):
        raise ValueError(f"Nonbinary mask: {path}")
    return value


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reference_points(record: dict, roi: np.ndarray, exclusion: np.ndarray) -> list[tuple[int, int]]:
    height, width = roi.shape
    points = sorted({(int(x), int(y)) for line in record["lines"]
                     for x, y in line.get("observed_points_xy", [])
                     if (0 <= int(x) < width and 0 <= int(y) < height and
                         roi[int(y), int(x)] and not exclusion[int(y), int(x)])})
    return points


def split_visible(lines: list[dict], held: set[tuple[int, int]], pitch: float,
                  roi: np.ndarray, exclusion: np.ndarray) -> list[dict]:
    """Only visible coordinates enter the fitter; IDs and old family caches vanish."""
    raw = []
    height, width = roi.shape
    for line in lines:
        points = sorted({(int(x), int(y)) for x, y in line.get("observed_points_xy", [])
                         if (int(x), int(y)) not in held and 0 <= int(x) < width and
                         0 <= int(y) < height and roi[int(y), int(x)] and
                         not exclusion[int(y), int(x)]})
        if not points:
            continue
        increments = [b[0] - a[0] for a, b in zip(points[:-1], points[1:]) if b[0] > a[0]]
        step = float(np.median(increments)) if increments else 7.
        gap_limit = max(2.5 * step, .8 * pitch)
        current = [points[0]]
        for point in points[1:]:
            last = current[-1]
            if point[0] - last[0] > gap_limit or abs(point[1] - last[1]) > .7 * pitch:
                raw.append(current)
                current = [point]
            else:
                current.append(point)
        raw.append(current)
    raw.sort(key=lambda p: (p[0][0], float(np.median([xy[1] for xy in p])), p[-1][0]))
    return [{"line_id": f"visible_{i:05d}",
             "observed_points_xy": [[x, y] for x, y in points]}
            for i, points in enumerate(raw, 1)]


def x_to_ys(mask: np.ndarray, low: int | None = None, high: int | None = None) -> dict[str, list[int]]:
    if low is None:
        low, high = 0, mask.shape[1] - 1
    result = {}
    for x in range(max(0, low), min(mask.shape[1] - 1, high) + 1):
        ys = np.flatnonzero(mask[:, x]).tolist()
        if ys:
            result[str(x)] = ys
    return result


def dedup_y(values: list[int], pitch: float) -> list[float]:
    groups = []
    for y in sorted(float(v) for v in values):
        if groups and y - groups[-1][-1] <= .15 * pitch:
            groups[-1].append(y)
        else:
            groups.append([y])
    return [float(np.median(group)) for group in groups]


def nearest(y: float, ys: list[float]) -> float | None:
    if not ys:
        return None
    index = bisect_left(ys, y)
    candidates = []
    if index < len(ys):
        candidates.append(abs(ys[index] - y))
    if index:
        candidates.append(abs(ys[index - 1] - y))
    return min(candidates)


def score(points: list[tuple[int, int]], predicted: dict[str, list[int]], pitch: float) -> dict:
    reference = defaultdict(list)
    for x, y in points:
        reference[x].append(y)
    columns = {int(x): dedup_y(ys, pitch) for x, ys in predicted.items()}
    threshold = .25 * pitch
    errors = []
    hit = missing = 0
    losses = []
    for x, y in points:
        error = nearest(float(y), columns.get(x, []))
        if error is None:
            missing += 1
            losses.append(1.)
        else:
            errors.append(error)
            losses.append(min(error / pitch, 1.))
            hit += int(error <= threshold)
    reverse_n = reverse_bad = 0
    for x, ys in reference.items():
        for y in columns.get(x, []):
            reverse_n += 1
            reverse_bad += int(nearest(y, sorted(ys)) > threshold)
    return {"reference_points": len(points), "same_column_predictions": len(errors),
            "missing_points": missing, "hits": hit, "sum_capped_loss": float(sum(losses)),
            "absolute_errors_px": errors, "reverse_predictions": reverse_n,
            "reverse_unmatched": reverse_bad}


def aggregate(items: list[dict], method: str) -> dict:
    values = [item["methods"][method] for item in items]
    n = sum(v["reference_points"] for v in values)
    hits = sum(v["hits"] for v in values)
    predicted = sum(v["same_column_predictions"] for v in values)
    loss = sum(v["sum_capped_loss"] for v in values)
    errors = [error for v in values for error in v["absolute_errors_px"]]
    reverse_n = sum(v["reverse_predictions"] for v in values)
    reverse_bad = sum(v["reverse_unmatched"] for v in values)
    return {"cases": len(items), "reference_points": n, "zero_reference_cases": sum(v["reference_points"] == 0 for v in values),
            "same_column_predictions": predicted, "missing_points": n - predicted,
            "hits_within_quarter_pitch": hits, "all_point_hit_rate": hits / n if n else None,
            "all_point_capped_loss": loss / n if n else None,
            "prediction_coverage": predicted / n if n else None,
            "conditional_mae_px": float(np.mean(errors)) if errors else None,
            "conditional_p95_px": float(np.quantile(errors, .95)) if errors else None,
            "reverse_predictions": reverse_n, "reverse_unmatched_predictions": reverse_bad,
            "reverse_unmatched_fraction": reverse_bad / reverse_n if reverse_n else None}


def light_case(row: dict, run: Path, input_hashes: list[dict]) -> tuple[dict, dict, dict, dict, float]:
    rel = Path(row["output_file"])
    role, name, stem = rel.parent.name, rel.name, rel.stem
    paths = {"b1_record": B1 / "boundaries/records" / role / (stem + ".json"),
             "bright": B1 / "roi/reference_masks" / role / name,
             "old_rope": B1 / "boundaries/incoming_rope_occlusion" / role / name,
             "old_unknown": B1 / "boundaries/unknown_exclusion" / role / name,
             "c3_smooth": C3 / "boundaries/smooth_boundaries" / role / name,
             "surface": run / "roi/surface_masks" / role / name,
             "v7_modeled": run / "boundaries/modeled" / role / name,
             "new_rope": run / "boundaries/incoming_rope_occlusion" / role / name,
             "new_unknown": run / "boundaries/unknown_exclusion" / role / name}
    input_hashes.append({"output_file": row["output_file"],
                         "sha256": {key: digest(path) for key, path in paths.items()}})
    record = load_json(paths["b1_record"])
    bright = binary(paths["bright"])
    old_rope = binary(paths["old_rope"], bright.shape)
    old_unknown = binary(paths["old_unknown"], bright.shape)
    old_exclusion = np.maximum(old_rope, old_unknown)
    surface = binary(paths["surface"], bright.shape)
    new_exclusion = np.maximum(binary(paths["new_rope"], bright.shape),
                               binary(paths["new_unknown"], bright.shape))
    old_smooth = binary(paths["c3_smooth"], bright.shape)
    new_smooth = binary(paths["v7_modeled"], bright.shape)
    pitch = float(record["record"]["pitch_px"])
    points = reference_points(record, bright, old_exclusion)
    prediction = {"case_id": row["output_file"], "camera_role": row["camera_role"],
                  "source_group": row["source_group"], "pitch_px": pitch,
                  "methods": {"c3": x_to_ys(old_smooth), "v7": x_to_ys(new_smooth)}}
    reference = {"case_id": row["output_file"], "points_xy": points}
    return prediction, reference, record, {"bright": bright, "old_exclusion": old_exclusion,
                                           "surface": surface, "new_exclusion": new_exclusion}, pitch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    run, out = args.run.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    protocol = load_json(PROTOCOL)
    fixed = protocol["fixed_inputs"]
    for path, expected in ((QA, fixed["qa_plan_sha256"]), (MANIFEST, fixed["full_400_manifest_sha256"]),
                           (SAM_MANIFEST, fixed["sam2_final_output_manifest_sha256"]),
                           (B1 / "integrity_audit.json", fixed["bright_b1_integrity_sha256"]),
                           (C3 / "integrity_audit.json", fixed["selected_c3_integrity_sha256"]),
                           (C3_ROOT / "evaluation_002/summary.json", fixed["c3_frozen_proxy_summary_sha256"])):
        if digest(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    context = load_json(run / "run_context.json")
    if context.get("mode") != "pilot":
        raise ValueError("The fixed 38-image evaluation accepts pilot mode only")
    with QA.open(encoding="utf-8-sig", newline="") as handle:
        qa = list(csv.DictReader(handle))
    with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        manifest = {row["output_file"]: row for row in csv.DictReader(handle)}
    if len(qa) != fixed["fixed_qa_images"] or Counter(row["camera_role"] for row in qa) != {"left": 19, "right": 19}:
        raise ValueError("QA38 identity or camera balance changed")
    if any(row["output_file"] not in manifest for row in qa):
        raise ValueError("QA item absent from frozen image manifest")
    if len({row["output_file"] for row in qa}) != 38:
        raise ValueError("QA image identity repeated")
    out.mkdir(parents=True)
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    sources = {"evaluation_full_surface.py": Path(__file__).resolve(),
               "full_surface_core.py": CORE, "conic_family.py": C3_MODULE,
               "evaluation_protocol.json": PROTOCOL}
    started = {key: digest(path) for key, path in sources.items()}
    for key, path in sources.items():
        shutil.copyfile(path, snapshot / key)
    evaluator_context = {"started_utc": datetime.now(timezone.utc).isoformat(),
                         "command": [str(Path(sys.executable).resolve()), str(Path(__file__).resolve()),
                                     "--run", str(run), "--output", str(out)],
                         "run": str(run), "run_context_sha256": digest(run / "run_context.json"),
                         "protocol_sha256": digest(PROTOCOL), "source_sha256_start": started,
                         "python": platform.python_version(), "opencv": cv2.__version__,
                         "source_photographs_decoded": False, "dark_image_pixels_used_in_holdout": False,
                         "reserved_inputs_accessed": False, "equipment_connected": False}
    dump(out / "run_context.json", evaluator_context)
    c3_module = import_module(C3_MODULE, "frozen_c3_for_v7_evaluation")
    core = import_module(CORE, "frozen_core_for_v7_evaluation")
    edge_references = []
    bright_references = []
    input_hashes = []
    edge_prediction_path = out / "edge_predictions.jsonl"
    bright_prediction_path = out / "bright_predictions.jsonl"
    with edge_prediction_path.open("w", encoding="utf-8") as edge_file, \
            bright_prediction_path.open("w", encoding="utf-8") as bright_file:
        for index, qa_row in enumerate(qa, 1):
            row = manifest[qa_row["output_file"]]
            bright_prediction, bright_reference, record, data, pitch = light_case(row, run, input_hashes)
            write_line(bright_file, bright_prediction)
            bright_references.append(bright_reference)
            xx = np.flatnonzero(np.any(data["bright"] > 0, axis=0))
            if len(xx) < 2:
                raise ValueError("Bright ROI lacks horizontal extent")
            xmin, xmax = int(xx[0]), int(xx[-1])
            span = xmax - xmin
            all_points = bright_reference["points_xy"]
            for side in ("left", "right"):
                hidden = {(x, y) for x, y in all_points if
                          ((x - xmin) / span < .15 if side == "left" else (x - xmin) / span > .85)}
                fragments = split_visible(record["lines"], hidden, pitch,
                                          data["bright"], data["old_exclusion"])
                c3_fit = c3_module.fit_conic_family(fragments, pitch, data["bright"],
                                                    data["old_exclusion"])
                fit = core.fit_surface_from_candidates(data["surface"], data["bright"], c3_fit,
                                                       data["new_exclusion"], pitch, [])
                low = xmin if side == "left" else int(math.floor(xmin + .85 * span)) + 1
                high = int(math.ceil(xmin + .15 * span)) - 1 if side == "left" else xmax
                prediction = {"case_id": f"{row['output_file']}:{side}", "output_file": row["output_file"],
                              "camera_role": row["camera_role"], "source_group": row["source_group"],
                              "hidden_side": side, "hidden_x_range_inclusive": [low, high],
                              "pitch_px": pitch, "visible_fragment_count": len(fragments),
                              "truncated_c3_group_count": len(c3_fit.get("groups", [])),
                              "new_group_count": len(fit.get("groups", [])),
                              "predictors": {"v7": x_to_ys(fit["smooth_mask"], low, high),
                                             "simple_c3": x_to_ys(fit["simple_c3_extension_mask"], low, high)}}
                write_line(edge_file, prediction)
                edge_references.append({"case_id": prediction["case_id"],
                                        "points_xy": sorted(hidden)})
            print(f"Predicted {index}/38 QA images", flush=True)
    # The prediction files are closed and hashed before withheld references are
    # serialized or opened for scoring. Their points never entered either fitter.
    pred_hashes = {"edge": digest(edge_prediction_path), "bright": digest(bright_prediction_path)}
    edge_ref_path = out / "withheld_edge_reference.jsonl"
    bright_ref_path = out / "bright_reference.jsonl"
    with edge_ref_path.open("w", encoding="utf-8") as handle:
        for ref in edge_references:
            write_line(handle, ref)
    with bright_ref_path.open("w", encoding="utf-8") as handle:
        for ref in bright_references:
            write_line(handle, ref)
    dump(out / "input_sha256.json", {"inputs": input_hashes, "prediction_sha256_before_scoring": pred_hashes,
                                     "withheld_reference_sha256": digest(edge_ref_path),
                                     "bright_reference_sha256": digest(bright_ref_path)})
    edge_predictions = [json.loads(line) for line in edge_prediction_path.read_text(encoding="utf-8").splitlines()]
    edge_truth = {item["case_id"]: item for item in
                  (json.loads(line) for line in edge_ref_path.read_text(encoding="utf-8").splitlines())}
    bright_predictions = [json.loads(line) for line in bright_prediction_path.read_text(encoding="utf-8").splitlines()]
    bright_truth = {item["case_id"]: item for item in
                    (json.loads(line) for line in bright_ref_path.read_text(encoding="utf-8").splitlines())}
    if len(edge_predictions) != 76 or len(edge_truth) != 76 or len(bright_predictions) != 38:
        raise ValueError("Prediction/reference case count differs from frozen protocol")
    edge_results = []
    for prediction in edge_predictions:
        ref = edge_truth[prediction["case_id"]]
        points = [tuple(point) for point in ref["points_xy"]]
        methods = {name: score(points, columns, prediction["pitch_px"])
                   for name, columns in prediction["predictors"].items()}
        edge_results.append({key: prediction[key] for key in ("case_id", "output_file", "camera_role", "source_group", "hidden_side")} |
                            {"methods": methods})
    bright_results = []
    for prediction in bright_predictions:
        ref = bright_truth[prediction["case_id"]]
        points = [tuple(point) for point in ref["points_xy"]]
        methods = {name: score(points, columns, prediction["pitch_px"])
                   for name, columns in prediction["methods"].items()}
        bright_results.append({key: prediction[key] for key in ("case_id", "camera_role", "source_group")} |
                              {"methods": methods})
    edge_n = sum(item["methods"]["v7"]["reference_points"] for item in edge_results)
    if edge_n != protocol["edge_holdout_proxy"]["expected_denominator_total"]:
        raise ValueError(f"Edge denominator {edge_n} differs from frozen 38200")
    bright_n = sum(item["methods"]["v7"]["reference_points"] for item in bright_results)
    if bright_n != protocol["fixed_bright_reference"]["expected_denominator_total"]:
        raise ValueError(f"Bright denominator {bright_n} differs from frozen 122068")
    summarize = lambda items, methods: {method: aggregate(items, method) for method in methods}
    edge_overall = summarize(edge_results, ("v7", "simple_c3"))
    bright_overall = summarize(bright_results, ("v7", "c3"))
    edge_groups = {name: summarize([item for item in edge_results if predicate(item)], ("v7", "simple_c3"))
                   for name, predicate in (("left_camera", lambda x: x["camera_role"] == "left"),
                                           ("right_camera", lambda x: x["camera_role"] == "right"),
                                           ("left_hidden_side", lambda x: x["hidden_side"] == "left"),
                                           ("right_hidden_side", lambda x: x["hidden_side"] == "right"))}
    bright_groups = {role: summarize([item for item in bright_results if item["camera_role"] == role], ("v7", "c3"))
                     for role in ("left", "right")}
    edge_loss_new = edge_overall["v7"]["all_point_capped_loss"]
    edge_loss_old = edge_overall["simple_c3"]["all_point_capped_loss"]
    edge_loss_gate = edge_loss_new <= 1.10 * edge_loss_old + 1e-12
    edge_hit_gate = all(block["v7"]["all_point_hit_rate"] >= block["simple_c3"]["all_point_hit_rate"] - .02 - 1e-12
                        for block in [edge_overall, *edge_groups.values()])
    bright_hit_gate = all(block["v7"]["all_point_hit_rate"] >= block["c3"]["all_point_hit_rate"] - .02 - 1e-12
                          for block in [bright_overall, *bright_groups.values()])
    edge_identical = all(item["predictors"]["v7"] == item["predictors"]["simple_c3"]
                         for item in edge_predictions)
    compact = lambda item: {key: value for key, value in item.items() if key != "absolute_errors_px"}
    def case_table(items: list[dict]) -> list[dict]:
        return [{**{k: v for k, v in item.items() if k != "methods"},
                 "methods": {name: compact(value) for name, value in item["methods"].items()}}
                for item in items]
    dump(out / "edge_cases.json", {"cases": case_table(edge_results)})
    dump(out / "bright_cases.json", {"cases": case_table(bright_results)})
    ended = {key: digest(path) for key, path in sources.items()}
    unchanged = started == ended
    gates = {"bright_hit_nonregression": bright_hit_gate,
             "edge_loss_nonregression": edge_loss_gate,
             "edge_hit_nonregression": edge_hit_gate,
             "source_unchanged": unchanged}
    overall_pass = all(gates.values())
    summary = {"run": str(run), "protocol_sha256": digest(PROTOCOL),
               "source_sha256_start": started, "source_sha256_end": ended,
               "source_unchanged": unchanged, "prediction_sha256_before_scoring": pred_hashes,
               "fixed_qa_images": 38, "edge_cases": len(edge_results),
               "bright_reference_points": bright_n, "withheld_edge_points": edge_n,
               "bright": {"overall": bright_overall, "by_camera": bright_groups},
               "edge": {"overall": edge_overall, "by_camera_or_side": edge_groups,
                        "predictions_identical_to_simple_when_no_dark_candidates": edge_identical,
                        "dark_gain_tested_by_edge_proxy": False},
               "gates": gates, "metric_gate_pass": overall_pass,
               "interpretation": "Same-image yellow agreement and finite extrapolation only; there is no dark or cutpoint ground truth.",
               "exit_code": 0 if overall_pass else 1}
    dump(out / "summary.json", summary)
    evaluator_context.update({"ended_utc": datetime.now(timezone.utc).isoformat(),
                              "source_sha256_end": ended, "source_unchanged": unchanged,
                              "prediction_sha256_before_scoring": pred_hashes,
                              "input_sha256_file": str(out / "input_sha256.json"),
                              "exit_code": summary["exit_code"]})
    dump(out / "run_context.json", evaluator_context)
    print(json.dumps({"output": str(out), "bright": bright_overall, "edge": edge_overall,
                      "gates": gates}, ensure_ascii=False), flush=True)
    return summary["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
