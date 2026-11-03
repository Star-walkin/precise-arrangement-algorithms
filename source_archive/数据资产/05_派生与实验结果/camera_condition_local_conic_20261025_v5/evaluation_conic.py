"""Frozen two-band hide-and-predict evaluation for the open v3 QA38 images.

The conic fitter and straight-line baseline receive only visible, freshly
renumbered fragments, the original ROI, the v3 exclusion mask and image pitch.
No original photograph or hidden point enters either predictor.  Predictions
are saved before withheld yellow points are scored.  This measures agreement
with earlier detections, not overexposure truth or material identity.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import traceback

import cv2
import numpy as np

from conic_family import fit_conic_family


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
V3 = BASE / "camera_condition_bright_rope_boundaries_20261025_v3" / "all_001"
QA = BASE / "camera_condition_bright_rope_boundaries_20261009_v1" / "qa_plan.csv"
PROTOCOL = HERE / "evaluation_protocol.json"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mask(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2 or np.any((image != 0) & (image != 255)):
        raise ValueError(f"Expected 0/255 mask: {path}")
    return image


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_line(handle, data: dict) -> None:
    handle.write(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")


def _parse_bands(protocol: dict) -> list[tuple[float, float]]:
    bands = [tuple(map(float, pair)) for pair in protocol["bands"]]
    if bands != [(0.25, 0.40), (0.60, 0.75)]:
        raise ValueError("Evaluation bands differ from the predeclared C1 protocol")
    return bands


def _roi_x_bounds(roi: np.ndarray) -> tuple[int, int]:
    xx = np.flatnonzero(np.any(roi > 0, axis=0))
    if len(xx) < 2:
        raise ValueError("ROI has no horizontal extent")
    return int(xx[0]), int(xx[-1])


def _split_fragments(lines: list[dict], held: set[tuple[int, int]],
                     pitch: float) -> list[dict]:
    """Erase original IDs/metadata and split gaps using visible pixels only."""
    raw = []
    for line in lines:
        pts = sorted({(int(x), int(y)) for x, y in line.get("observed_points_xy", [])
                      if (int(x), int(y)) not in held})
        if not pts:
            continue
        increments = [b[0] - a[0] for a, b in zip(pts[:-1], pts[1:]) if b[0] > a[0]]
        median_step = float(np.median(increments)) if increments else 7.
        gap_limit = max(2.5 * median_step, .8 * pitch)
        current = [pts[0]]
        for point in pts[1:]:
            prev = current[-1]
            if point[0] - prev[0] > gap_limit or abs(point[1] - prev[1]) > .7 * pitch:
                raw.append(current)
                current = [point]
            else:
                current.append(point)
        raw.append(current)
    # Sorting uses solely visible geometry. Fresh IDs contain no old line,
    # family or hidden-band identity and are the only names seen by the fitter.
    raw.sort(key=lambda pts: (pts[0][0], float(np.median([p[1] for p in pts])), pts[-1][0]))
    return [{"line_id": f"visible_{i:05d}",
             "observed_points_xy": [[x, y] for x, y in pts]}
            for i, pts in enumerate(raw, 1)]


def _side_anchors(fragments: list[dict], edge: float, side: str,
                  reach: float, pitch: float) -> list[tuple[float, float]]:
    anchors = []
    for fragment in fragments:
        pts = fragment["observed_points_xy"]
        if len(pts) < 2:
            continue
        x, y = pts[-1] if side == "left" else pts[0]
        if side == "left":
            if not (edge - reach <= x < edge):
                continue
        elif not (edge <= x <= edge + reach):
            continue
        anchors.append((float(x), float(y)))
    anchors.sort(key=lambda p: (p[1], p[0]))
    # Close duplicates can arise from fragmented observations of one row.
    # Dedup uses only visible endpoint geometry, never original line IDs.
    unique = []
    for point in anchors:
        if unique and abs(point[1] - unique[-1][1]) <= .30 * pitch:
            if abs(point[0] - edge) < abs(unique[-1][0] - edge):
                unique[-1] = point
        else:
            unique.append(point)
    return unique


def _linear_predict(fragments: list[dict], xlo: int, xhi: int,
                    roi: np.ndarray, exclusion: np.ndarray, pitch: float,
                    roi_width: float) -> tuple[dict[int, list[int]], dict]:
    """Simple ordered nearest-endpoint pairing, then straight interpolation."""
    reach = max(3. * pitch, .08 * roi_width)
    left = _side_anchors(fragments, xlo, "left", reach, pitch)
    right = _side_anchors(fragments, xhi, "right", reach, pitch)
    pairs = []
    last_right = -1
    for lx, ly in left:
        candidates = [(abs(ry - ly), j, rx, ry)
                      for j, (rx, ry) in enumerate(right)
                      if j > last_right and abs(ry - ly) <= 1.25 * pitch]
        if not candidates:
            continue
        _, j, rx, ry = min(candidates)
        last_right = j
        pairs.append(((lx, ly), (rx, ry)))
    result: dict[int, list[int]] = defaultdict(list)
    height, width = roi.shape
    for (lx, ly), (rx, ry) in pairs:
        if rx <= lx:
            continue
        for x in range(max(0, xlo), min(width - 1, xhi - 1) + 1):
            if not lx <= x <= rx:
                continue
            y = int(round(ly + (ry - ly) * (x - lx) / (rx - lx)))
            if 0 <= y < height and roi[y, x] and not exclusion[y, x]:
                result[x].append(y)
    return dict(result), {"left_anchors": len(left), "right_anchors": len(right),
                          "matched_pairs": len(pairs), "reach_px": reach,
                          "match_y_limit_px": 1.25 * pitch}


def _conic_predict(fragments: list[dict], roi: np.ndarray,
                   exclusion: np.ndarray, xlo: int, xhi: int,
                   pitch: float) -> tuple[dict[int, list[int]], dict]:
    fit = fit_conic_family(fragments, pitch, roi, exclusion)
    smooth = fit["smooth_mask"]
    if smooth.shape != roi.shape or smooth.dtype != np.uint8:
        raise ValueError("Conic smooth_mask violates image contract")
    violations = int(np.count_nonzero((smooth > 0) & ((roi == 0) | (exclusion > 0))))
    result: dict[int, list[int]] = defaultdict(list)
    height, width = roi.shape
    for curve in fit.get("curves", []):
        for x, y in curve.get("sampled_points_xy", []):
            x, y = int(x), int(y)
            if not (0 <= x < width and 0 <= y < height):
                violations += 1
            elif xlo <= x < xhi:
                if roi[y, x] == 0 or exclusion[y, x] != 0:
                    violations += 1
                else:
                    result[x].append(y)
    if violations:
        raise ValueError(f"Conic ROI/exclusion violation: {violations} pixels")
    return dict(result), {"fit_diagnostics": fit.get("diagnostics"),
                          "rejected": fit.get("rejected"),
                          "curve_count": len(fit.get("curves", [])),
                          "smooth_pixels": int(np.count_nonzero(smooth)),
                          "roi_exclusion_violations": violations}


def _predictions_one(source: str, role: str, group: str, band_index: int,
                     band: tuple[float, float], record: dict,
                     roi: np.ndarray, exclusion: np.ndarray) -> tuple[dict, dict]:
    pitch = float(record["record"]["pitch_px"])
    left, right = _roi_x_bounds(roi)
    width = right - left + 1
    xlo = int(math.ceil(left + band[0] * width))
    xhi = int(math.ceil(left + band[1] * width))
    hidden = sorted({(int(x), int(y))
                     for line in record["lines"]
                     for x, y in line.get("observed_points_xy", [])
                     if xlo <= int(x) < xhi})
    for x, y in hidden:
        if not (0 <= x < roi.shape[1] and 0 <= y < roi.shape[0]
                and roi[y, x] and not exclusion[y, x]):
            raise ValueError(f"Original yellow point outside effective ROI: {source}, {(x,y)}")
    fragments = _split_fragments(record["lines"], set(hidden), pitch)
    if hidden:
        conic, conic_diag = _conic_predict(fragments, roi, exclusion, xlo, xhi, pitch)
        linear, linear_diag = _linear_predict(fragments, xlo, xhi, roi,
                                             exclusion, pitch, width)
    else:
        conic, linear = {}, {}
        conic_diag = {"status": "zero_hidden_reference", "curve_count": 0,
                      "roi_exclusion_violations": 0}
        linear_diag = {"status": "zero_hidden_reference", "matched_pairs": 0}
    key = f"{role}:{Path(source).stem}:band{band_index}"
    prediction = {"case_id": key, "source": source, "camera_role": role,
                  "source_group": group, "band_index": band_index,
                  "band": list(band), "band_x_px": [xlo, xhi],
                  "pitch_px": pitch, "visible_fragment_count": len(fragments),
                  "predictors": {"conic": {str(x): ys for x, ys in sorted(conic.items())},
                                 "linear": {str(x): ys for x, ys in sorted(linear.items())}},
                  "conic_diagnostics": conic_diag,
                  "linear_diagnostics": linear_diag}
    reference = {"case_id": key, "hidden_points_xy": [[x, y] for x, y in hidden]}
    return prediction, reference


def _dedup_y(values: list[int], pitch: float) -> list[float]:
    ordered = sorted(float(v) for v in values)
    groups = []
    for value in ordered:
        if groups and value - groups[-1][-1] <= .15 * pitch:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [float(np.median(group)) for group in groups]


def _score_prediction(points: list[list[int]], prediction: dict,
                      pitch: float) -> dict:
    by_x: dict[int, list[int]] = defaultdict(list)
    for x, y in points:
        by_x[int(x)].append(int(y))
    emitted = {int(x): _dedup_y(ys, pitch) for x, ys in prediction.items()}
    threshold = .25 * pitch
    errors = []
    misses = 0
    hits = 0
    all_losses = []
    for x, y in points:
        ys = emitted.get(int(x), [])
        if not ys:
            misses += 1
            all_losses.append(1.)
            continue
        error = min(abs(float(y) - candidate) for candidate in ys)
        errors.append(error)
        all_losses.append(min(error / pitch, 1.))
        hits += int(error <= threshold)
    reverse_total = 0
    reverse_unmatched = 0
    for x, truth_ys in by_x.items():
        for y in emitted.get(x, []):
            reverse_total += 1
            reverse_unmatched += int(min(abs(y - v) for v in truth_ys) > threshold)
    return {"hidden_points": len(points), "predicted_points": len(errors),
            "missing_points": misses, "hits_within_quarter_pitch": hits,
            "sum_capped_normalized_error": float(sum(all_losses)),
            "accepted_absolute_errors_px": errors,
            "reverse_predictions_at_hidden_x": reverse_total,
            "reverse_unmatched_predictions": reverse_unmatched}


def _aggregate(items: list[dict], method: str) -> dict:
    relevant = [item["methods"][method] for item in items]
    n = sum(item["hidden_points"] for item in relevant)
    emitted = sum(item["predicted_points"] for item in relevant)
    hit = sum(item["hits_within_quarter_pitch"] for item in relevant)
    loss = sum(item["sum_capped_normalized_error"] for item in relevant)
    errors = [value for item in relevant for value in item["accepted_absolute_errors_px"]]
    reverse_n = sum(item["reverse_predictions_at_hidden_x"] for item in relevant)
    reverse_bad = sum(item["reverse_unmatched_predictions"] for item in relevant)
    return {"image_band_cases": len(items),
            "zero_hidden_reference_cases": sum(item["hidden_points"] == 0 for item in relevant),
            "hidden_points": n, "predicted_points": emitted,
            "missing_points": n - emitted, "hits_within_quarter_pitch": hit,
            "all_point_hit_rate": hit / n if n else None,
            "prediction_coverage": emitted / n if n else None,
            "all_point_capped_loss": loss / n if n else None,
            "conditional_predicted_point_mae_px": float(np.mean(errors)) if errors else None,
            "conditional_predicted_point_p95_px": float(np.quantile(errors, .95)) if errors else None,
            "reverse_predictions_at_hidden_x": reverse_n,
            "reverse_unmatched_predictions": reverse_bad,
            "reverse_unmatched_fraction": reverse_bad / reverse_n if reverse_n else None}


def _summarize(items: list[dict]) -> dict:
    return {method: _aggregate(items, method) for method in ("conic", "linear")}


def _comparison(aggregate: dict) -> dict:
    c, b = aggregate["conic"], aggregate["linear"]
    if c["hidden_points"] != b["hidden_points"]:
        raise ValueError("Prediction methods have different denominators")
    c_loss, b_loss = c["all_point_capped_loss"], b["all_point_capped_loss"]
    c_hit, b_hit = c["all_point_hit_rate"], b["all_point_hit_rate"]
    return {"same_hidden_point_denominator": c["hidden_points"],
            "all_point_hit_rate_delta_pp": 100 * (c_hit - b_hit) if c_hit is not None else None,
            "relative_capped_loss_reduction": (b_loss - c_loss) / b_loss
            if b_loss is not None and b_loss > 0 else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New evaluation directory; never overwrites earlier runs")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    bands = _parse_bands(protocol)
    if _hash(QA) != protocol["qa_manifest_sha256"]:
        raise ValueError("Fixed QA manifest SHA256 differs from frozen evaluation protocol")
    with QA.open(encoding="utf-8-sig", newline="") as handle:
        qa = list(csv.DictReader(handle))
    if len(qa) != protocol["fixed_qa_count"]:
        raise ValueError("Unexpected QA image count")
    out.mkdir(parents=True)
    source_paths = [HERE / "evaluation_conic.py", HERE / "conic_family.py",
                    HERE / "base_conic.py", PROTOCOL, QA]
    source_start = {str(path): _hash(path) for path in source_paths}
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for path in source_paths:
        shutil.copyfile(path, snapshot / path.name)
    input_hashes = []
    for row in qa:
        rel = Path(row["output_file"])
        camera_dir = row["camera_role"] + "_camera"
        name = rel.name
        paths = {
            "v3_record": V3 / "boundaries" / "records" / camera_dir / (rel.stem + ".json"),
            "reference_roi": V3 / "roi" / "reference_masks" / camera_dir / name,
            "supported_occlusion": V3 / "boundaries" / "incoming_rope_occlusion" / camera_dir / name,
            "unknown_exclusion": V3 / "boundaries" / "unknown_exclusion" / camera_dir / name,
        }
        input_hashes.append({"output_file": row["output_file"],
                             "sha256": {key: _hash(path) for key, path in paths.items()}})
    context = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": _hash(PROTOCOL), "qa_manifest_sha256": _hash(QA),
        "conic_module_sha256": _hash(HERE / "conic_family.py"),
        "evaluation_script_sha256": _hash(Path(__file__).resolve()),
        "v3_run_context_sha256": _hash(V3 / "run_context.json"),
        "source_sha256_start": source_start,
        "input_sha256": input_hashes,
        "source_snapshot": str(snapshot),
        "python": platform.python_version(), "opencv": cv2.__version__,
        "source_photographs_decoded": False, "reserved_data_accessed": False,
        "equipment_connected": False,
    }
    _write_json(out / "run_context.json", context)
    predictions_path = out / "predictions.jsonl"
    references_path = out / "withheld_reference.jsonl"
    with predictions_path.open("w", encoding="utf-8") as prediction_file, \
            references_path.open("w", encoding="utf-8") as reference_file:
        for image_index, row in enumerate(qa, 1):
            source = row["output_file"]
            rel = Path(source)
            role = row["camera_role"]
            camera_dir = role + "_camera"
            name = rel.name
            path = V3 / "boundaries" / "records" / camera_dir / (rel.stem + ".json")
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["record"]["camera_role"] != role:
                raise ValueError(f"Camera mismatch: {source}")
            roi = _mask(V3 / "roi" / "reference_masks" / camera_dir / name)
            orange = _mask(V3 / "boundaries" / "incoming_rope_occlusion" / camera_dir / name)
            blue = _mask(V3 / "boundaries" / "unknown_exclusion" / camera_dir / name)
            if roi.shape != orange.shape or roi.shape != blue.shape:
                raise ValueError(f"ROI/exclusion shape mismatch: {source}")
            exclusion = np.maximum(orange, blue)
            if np.any((exclusion > 0) & (roi == 0)):
                raise ValueError(f"Exclusion outside reference ROI: {source}")
            for band_index, band in enumerate(bands, 1):
                prediction, reference = _predictions_one(
                    source, role, row["source_group"], band_index, band,
                    record, roi, exclusion)
                _write_line(prediction_file, prediction)
                _write_line(reference_file, reference)
            print(f"Predicted {image_index}/{len(qa)} fixed QA images", flush=True)
    # All predictions are now durable on disk.  Only this second phase reads
    # withheld points for matching and metrics.
    predictions = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
    references = [json.loads(line) for line in references_path.read_text(encoding="utf-8").splitlines()]
    if len(predictions) != len(references) or len(predictions) != 2 * len(qa):
        raise ValueError("Prediction/reference case count mismatch")
    cases = []
    for prediction, reference in zip(predictions, references):
        if prediction["case_id"] != reference["case_id"]:
            raise ValueError("Prediction/reference ordering mismatch")
        hidden = reference["hidden_points_xy"]
        result = {"case_id": prediction["case_id"],
                  "source": prediction["source"], "camera_role": prediction["camera_role"],
                  "source_group": prediction["source_group"],
                  "band_index": prediction["band_index"],
                  "hidden_points": len(hidden),
                  "methods": {
                      method: _score_prediction(hidden, prediction["predictors"][method],
                                                prediction["pitch_px"])
                      for method in ("conic", "linear")}}
        cases.append(result)
    overall = _summarize(cases)
    by_camera = {role: _summarize([case for case in cases if case["camera_role"] == role])
                 for role in ("left", "right")}
    by_band = {str(index): _summarize([case for case in cases if case["band_index"] == index])
               for index in (1, 2)}
    by_group = []
    for role, group in sorted({(case["camera_role"], case["source_group"]) for case in cases}):
        group_cases = [case for case in cases
                       if case["camera_role"] == role and case["source_group"] == group]
        by_group.append({"camera_role": role, "source_group": group,
                         "metrics": _summarize(group_cases)})
    comparisons = {"overall": _comparison(overall),
                   "by_camera": {key: _comparison(value) for key, value in by_camera.items()},
                   "by_band": {key: _comparison(value) for key, value in by_band.items()}}
    overall_loss = comparisons["overall"]["relative_capped_loss_reduction"]
    hit_deltas = [comparisons["overall"]["all_point_hit_rate_delta_pp"]] + [
        value["all_point_hit_rate_delta_pp"] for value in comparisons["by_camera"].values()] + [
        value["all_point_hit_rate_delta_pp"] for value in comparisons["by_band"].values()]
    criterion_met = (overall_loss is not None and overall_loss >= .10
                     and all(value is not None and value >= -2. for value in hit_deltas))
    source_end = {str(path): _hash(path) for path in source_paths}
    source_unchanged = source_end == source_start
    exit_code = int(not source_unchanged)
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "interpretation": protocol["truth_limit"],
               "fixed_qa_images": len(qa), "image_band_cases": len(cases),
               "predictions_sha256": _hash(predictions_path),
               "withheld_reference_sha256": _hash(references_path),
               "protocol_sha256": _hash(PROTOCOL),
               "input_sha256_reference": "run_context.json:input_sha256",
               "source_unchanged": source_unchanged,
               "exit_code": exit_code,
               "overall": overall, "by_camera": by_camera, "by_band": by_band,
               "by_camera_source_group": by_group,
               "comparisons": comparisons,
               "predeclared_continue_criterion_met": criterion_met,
               "case_metrics": cases}
    _write_json(out / "summary.json", summary)
    context.update({"finished_utc": datetime.now(timezone.utc).isoformat(),
                    "source_sha256_end": source_end,
                    "source_unchanged": source_unchanged,
                    "predictions_sha256": _hash(predictions_path),
                    "withheld_reference_sha256": _hash(references_path),
                    "summary_sha256": _hash(out / "summary.json"),
                    "exit_code": exit_code})
    _write_json(out / "run_context.json", context)
    print(json.dumps({"output": str(out / "summary.json"),
                      "cases": len(cases),
                      "hidden_points": overall["conic"]["hidden_points"],
                      "conic_loss": overall["conic"]["all_point_capped_loss"],
                      "linear_loss": overall["linear"]["all_point_capped_loss"],
                      "continue_criterion_met": criterion_met}, ensure_ascii=False), flush=True)
    return exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        output_arg = None
        if "--output" in sys.argv:
            index = sys.argv.index("--output")
            if index + 1 < len(sys.argv):
                output_arg = Path(sys.argv[index + 1]).resolve()
        if output_arg is not None and output_arg.is_dir():
            _write_json(output_arg / "failure_context.json", {
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "exit_code": 1, "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
        raise
