"""Independent, post-prediction scoring of the fixed eight open development images.

This file never trains, proposes candidates, changes a selection, or decodes an
image. Every mode's eight response/path pairs are validated and hashed before
the first evaluation annotation is opened. Partial reference is not an
exhaustive negative mask, and an observed surface boundary is not a known
target-identity endpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision_geometry_lab.centerline_path_metrics_v6 import (
    _nearest, _sampling, clip_scoring_segments,
)
from vision_geometry_lab.experiments.v28_local_spool_connection import data_eval

STAGES = {1: "free", 2: "transition", 3: "attached", 4: "edge_near"}
STATUSES = ("entry_condition", "visible_supported", "ambiguous", "inferred")
PROTOCOL = {
    "version": 1,
    "population": "fixed eight previously open development images; not blind",
    "coordinate_system": "original full-image xy; all distances in original pixels",
    "matching_tolerance_px": 5.0,
    "maximum_sampling_spacing_px": 4.0,
    "weighting": "polyline arclength Voronoi cells; stage assigned at cell sample",
    "phase_definition": "same frozen data/scenes known-entry visual proxies for both modes, scoring only; not physical contact annotation or detector input",
    "labels_opened_only_after_all_requested_predictions_validated_and_hashed": True,
    "empty_selection": "rejection; zero coverage where reference exists; error undefined",
    "partial_reference": "unannotated path is unknown, never false-positive background",
    "candidate_pool": "oracle posthoc diagnostics only; does not change selected path",
    "edge_endpoint_identity": "unknown: no independent annotated target endpoint",
    "physical_identity_confirmed": False,
    "input_seed_exclusion_radius_px": 5.0,
    "input_seed_exclusion": "remove the actual supplied short-seed 5px capsule from reference and prediction; also remove explicitly input-labelled path edges; disclose original and remaining lengths",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_new(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _line(points):
    a = np.asarray(points, dtype=np.float64)
    if not a.size:
        return np.empty((0, 2), dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all():
        raise ValueError("Path must contain finite Nx2 original-image coordinates")
    return a


def _validate_case(run: Path, mode: str, sid: str, shape_hw):
    folder = run / "predictions" / mode / sid
    paths_file, response_file = folder / "paths.json", folder / "response.npy"
    # No reference loader is invoked before this entire gate has passed.
    if not paths_file.is_file() or not response_file.is_file():
        raise FileNotFoundError(f"Prediction-save gate failed: {folder}")
    hashes = {"paths_sha256": sha256(paths_file), "response_sha256": sha256(response_file)}
    value = _read(paths_file)
    if value["mode"] != mode:
        raise ValueError(f"Mode mismatch: {paths_file}")
    candidates = value["candidates"]
    if len(candidates) > 3:
        raise ValueError("At most three saved candidates are permitted")
    ids = [c["id"] for c in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate candidate IDs")
    selected = value.get("selected_candidate_id")
    if selected is not None and selected not in ids:
        raise ValueError("Selection references an absent candidate")
    height, width = shape_hw
    for candidate in candidates:
        points = _line(candidate["points_xy"])
        if len(points) < 2:
            raise ValueError("Nonempty candidate needs at least two points")
        if (points < 0).any() or (points[:, 0] > width-1).any() or (points[:, 1] > height-1).any():
            raise ValueError("Candidate points lie outside original image")
        statuses, stages = candidate["point_status"], candidate["stage"]
        if len(statuses) != len(points) or len(stages) != len(points):
            raise ValueError("Per-point status/stage lengths do not match candidate")
        if set(statuses) - set(STATUSES) or set(stages) - set(STAGES):
            raise ValueError("Unknown status or stage")
    response = np.load(response_file, mmap_mode="r", allow_pickle=False)
    if response.shape != tuple(shape_hw) or response.dtype != np.dtype("float32"):
        raise ValueError(f"Expected original-shape float32 response: {response_file}")
    if np.isinf(response).any():
        raise ValueError("Response contains infinity")
    finite = np.isfinite(response)
    values = np.asarray(response[finite])
    if values.size and (values.min() < -1e-6 or values.max() > 1.0+1e-6):
        raise ValueError("Response is outside the fixed [0,1] intensity range")
    distribution = dict(finite_pixels=int(values.size), nan_pixels=int((~finite).sum()),
        minimum=float(values.min()) if values.size else None,
        maximum=float(values.max()) if values.size else None,
        mean=float(values.mean()) if values.size else None,
        std=float(values.std()) if values.size else None,
        quantiles={str(q): float(np.quantile(values, q)) for q in (.01, .1, .5, .9, .99)} if values.size else {},
        threshold_fraction={str(q): float((values >= q).mean()) for q in (.1, .5, .9)} if values.size else {},
        interpretation="uncalibrated response intensity; finite search domain only")
    del response
    scene_file = run / "data" / "scenes" / f"{sid}.json"
    # Freeze the scene input before labels as well. The entry-specific scene
    # in saved prediction JSON is preferred to a separate default scene.
    scene = value.get("edge_target")
    if not isinstance(scene, dict) or "contact_proxy" not in scene:
        if not scene_file.is_file():
            raise ValueError(f"Frozen scene missing: {scene_file}")
        scene = _read(scene_file)
        hashes["scene_sha256"] = sha256(scene_file)
    else:
        hashes["embedded_scene_sha256"] = hashlib.sha256(json.dumps(
            scene, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()
    if mode == "AUTO":
        entry_file = run / "data" / "auto_entries" / f"{sid}.json"
        entry = _read(entry_file)
        seed = entry.get("free_seed", []) if entry["available"] else []
    else:
        entry_file = run / "data" / "entry_hints" / f"{sid}.json"
        seed = _read(entry_file)["free_seed_entry_only_xy"]
    hashes["entry_input_sha256"] = sha256(entry_file)
    scoring_scene = _read(scene_file)
    hashes["common_scoring_scene_sha256"] = sha256(scene_file)
    return dict(sample_id=sid, mode=mode, prediction=value, scene=scene, scoring_scene=scoring_scene, actual_input_seed_xy=seed,
        response_distribution=distribution, prediction_files={
            "paths": str(paths_file), "response": str(response_file), **hashes})


def _weighted_quantile(values, weights, q):
    if not len(values) or not np.isfinite(values).all():
        return None
    order = np.argsort(values)
    return float(np.interp(q*np.sum(weights), np.cumsum(weights[order]), values[order]))


def _reference_samples(segments, scene, stage_at_points):
    result = []
    for piece in _sampling(segments):
        stages = np.asarray(stage_at_points(scene, piece["points"]), dtype=np.uint8)
        if stages.shape != (len(piece["points"]),) or set(stages.tolist()) - set(STAGES):
            raise ValueError("Scene classification must return one stage 1..4 per point")
        result.append({**piece, "stages": stages})
    return result


def _summarize_reference_rows(pieces, predicted_segments, stage=None):
    distances, normal_errors, weights = [], [], []
    uncovered = []
    for piece in pieces:
        distance, closest = _nearest(piece["points"], predicted_segments)
        normal = np.column_stack((-piece["tangents"][:, 1], piece["tangents"][:, 0]))
        normal_error = np.abs(np.sum((closest-piece["points"])*normal, axis=1))
        mask = np.ones(len(distance), dtype=bool) if stage is None else piece["stages"] == stage
        distances.extend(distance[mask]); normal_errors.extend(normal_error[mask]); weights.extend(piece["weights"][mask])
        # Each gap remains within its original clipped reference piece and stage.
        bad = mask & (distance > PROTOCOL["matching_tolerance_px"])
        starts = np.flatnonzero(bad & ~np.r_[False, bad[:-1]])
        ends = np.flatnonzero(bad & ~np.r_[bad[1:], False]) + 1
        for lo, hi in zip(starts, ends):
            uncovered.append(dict(length_px=float(piece["weights"][lo:hi].sum()),
                from_xy=piece["points"][lo].tolist(), to_xy=piece["points"][hi-1].tolist()))
    distance, normal_error, weight = map(np.asarray, (distances, normal_errors, weights))
    length = float(weight.sum())
    matched = distance <= PROTOCOL["matching_tolerance_px"]
    covered = float(weight[matched].sum())
    finite = bool(len(distance) and np.isfinite(distance).all())
    return dict(reference_length_px=length, reference_samples=int(len(weight)),
        status="scored_partial_reference" if length else "unknown_no_reference_in_stage",
        covered_reference_length_px=covered if length else None,
        covered_fraction=covered/length if length else None,
        reference_to_prediction_mean_px=float(np.sum(distance*weight)/length) if finite and length else None,
        reference_to_prediction_median_px=_weighted_quantile(distance, weight, .5),
        reference_to_prediction_p95_px=_weighted_quantile(distance, weight, .95),
        normal_error_p95_on_5px_matches_px=_weighted_quantile(normal_error[matched], weight[matched], .95),
        longest_uncovered_run_px=max((g["length_px"] for g in uncovered), default=0.) if length else None,
        uncovered_runs=uncovered,
        precision=None, identity_accuracy=None,
        reason="partial target centerline; no exhaustive negative/endpoint identity reference")


def _status_lengths(candidate):
    if candidate is None:
        return {status: 0. for status in STATUSES}
    points = _line(candidate["points_xy"])
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    labels = candidate["point_status"]
    totals = {status: 0. for status in STATUSES}
    priority = {"visible_supported": 0, "inferred": 1, "ambiguous": 2, "entry_condition": 3}
    for length, a, b in zip(lengths, labels[:-1], labels[1:]):
        # The first segment beyond the seed is predicted, even though its
        # initial endpoint is the last supplied seed point.
        pair = (a, b) if a == b == "entry_condition" else tuple(s for s in (a, b) if s != "entry_condition")
        totals[max(pair, key=priority.get)] += float(length)
    return totals


def _entry_polygons(seed, radius=5.):
    """Vector capsules, independent of all target geometry, no mask rasterization."""
    line = _line(seed)
    if not len(line):
        return []
    polygons = []
    angles = np.linspace(0., 2*np.pi, 33)[:-1]
    # Circumscribed regular polygons conservatively cover the true 5px disk.
    offsets = radius/np.cos(np.pi/32)*np.column_stack([np.cos(angles), np.sin(angles)])
    for point in line:
        polygons.append((point+offsets).tolist())
    for a, b in zip(line[:-1], line[1:]):
        delta = b-a; length = np.linalg.norm(delta)
        if length <= 1e-9:
            continue
        normal = radius*np.array([-delta[1], delta[0]])/length
        polygons.append(np.array([a+normal, b+normal, b-normal, a-normal]).tolist())
    return polygons


def _without_input_edges(candidate):
    if candidate is None:
        return []
    points, states = candidate["points_xy"], candidate["point_status"]
    runs, run = [], []
    for i, (a, b) in enumerate(zip(points[:-1], points[1:])):
        if states[i] == states[i+1] == "entry_condition":
            if len(run) >= 2:
                runs.append(run)
            run = []
        else:
            if not run:
                run = [a]
            run.append(b)
    if len(run) >= 2:
        runs.append(run)
    return runs


def _length(segments):
    return float(sum(np.linalg.norm(np.diff(_line(s), axis=0), axis=1).sum() for s in segments))


def _score_candidate(candidate, pieces, reference, scene, entry_polygons=()):
    points = [] if candidate is None else candidate["points_xy"]
    quality_only_segments = clip_scoring_segments([points], ignore_polygons=reference["ignore_polygons"])
    predicted_segments = clip_scoring_segments(_without_input_edges(candidate),
        ignore_polygons=[*reference["ignore_polygons"], *entry_polygons])
    all_score = _summarize_reference_rows(pieces, predicted_segments)
    stages = {name: _summarize_reference_rows(pieces, predicted_segments, number)
              for number, name in STAGES.items()}
    edge = scene.get("target_edge_points_xy", [])
    end = _line(points)[-1:] if len(points) else np.empty((0, 2))
    geometric_edge_distance = None
    if len(end) and len(edge) >= 2:
        geometric_edge_distance = float(_nearest(end, [edge])[0][0])
    return dict(candidate_id=None if candidate is None else candidate["id"],
        all=all_score, stages=stages,
        predicted_arclength_px=float(np.linalg.norm(np.diff(_line(points), axis=0), axis=1).sum()),
        prediction_length_before_seed_exclusion_px=_length(quality_only_segments),
        prediction_length_scored_after_seed_exclusion_px=_length(predicted_segments),
        prediction_length_removed_as_input_px=_length(quality_only_segments)-_length(predicted_segments),
        claimed_status_lengths_px=_status_lengths(candidate),
        claimed_status_note="algorithm's labels, not independent visibility/identity truth",
        endpoint_to_scene_edge_distance_px=geometric_edge_distance,
        target_edge_identity_success=None,
        endpoint_status="unknown_no_independent_target_terminal_reference",
        whole_arc_identity_success=None)


def evaluate_case(case, reference, stage_at_points):
    """Pure posthoc scorer; scene and prediction are immutable evaluator inputs."""
    value, scene = case["prediction"], case["scene"]
    quality_only_segments = clip_scoring_segments([reference["centerline"]], ignore_polygons=reference["ignore_polygons"])
    input_polygons = _entry_polygons(case.get("actual_input_seed_xy", []))
    segments = clip_scoring_segments(quality_only_segments,
        ignore_polygons=input_polygons)
    pieces = _reference_samples(segments, case.get("scoring_scene", scene), stage_at_points)
    candidates = value["candidates"]
    scores = [_score_candidate(c, pieces, reference, scene, input_polygons) for c in candidates]
    selected_id = value.get("selected_candidate_id")
    selected = next((c for c in candidates if c["id"] == selected_id), None)
    selected_score = _score_candidate(selected, pieces, reference, scene, input_polygons)
    best = max(scores, key=lambda s: s["all"]["covered_fraction"] or 0., default=None)
    # Reuse the original V28 reviewed paired-edge diagnostic. Its scoring
    # geometry cannot influence scene, pool, ranking, stopping, or input seed.
    ranking = {key: {"selected_id": selected_id, "best_id": None,
                    "status": "rejected" if selected is None else "selected"}
               for key in ("G0", "G1", "G2")}
    ribbon = data_eval.evaluate_reference(reference, value, ranking, case["mode"])
    return {key: case[key] for key in ("sample_id", "mode", "prediction_files", "response_distribution")} | dict(
        selection_status="rejected" if selected is None else "selected",
        selected_candidate_id=selected_id, selected=selected_score,
        candidate_scores=scores,
        first_candidate_diagnostic=scores[0] if scores else None,
        oracle_best_pool_diagnostic=best,
        oracle_note="posthoc only; never an automatic selection and not a correct-identity proof",
        source_reference=dict(annotation_sha256=reference["annotation_sha256"],
            quality_sha256=reference["quality_sha256"], reviewed=reference["reviewed"],
            status=reference["status"], trusted_clipped_centerline=segments,
            reference_length_before_seed_exclusion_px=_length(quality_only_segments),
            reference_length_after_seed_exclusion_px=_length(segments),
            reference_length_removed_as_input_px=_length(quality_only_segments)-_length(segments),
            actual_input_seed_xy=case.get("actual_input_seed_xy", []),
            input_seed_exclusion_radius_px=5.,
            true_edge_endpoint=None, physical_identity_confirmed=False),
        legacy_v28_ribbon_diagnostic={key: ribbon[key] for key in (
            "visible_connection_evaluable", "candidate_scores", "GT_departure",
            "trusted_bend_segments", "visible_branch_segments", "branch_cross_sections")},
        legacy_selected_visible_connection_status=ribbon["choices"]["G0"]["selected_visible_connection_status"],
        scene_phase_semantics="same frozen known-entry data/scenes stage classifier for both modes; posthoc scoring only, not physical contact truth; actual input exclusion may change per-mode denominators",
        legacy_diagnostic_note="unchanged V28 departure/bend metric downstream of its longer old free_seed; primary new scores explicitly exclude the actual short input",
        true_whole_arc_identity_success=None)


def _aggregate(rows):
    summary = {}
    for mode in sorted({r["mode"] for r in rows}):
        mode_rows = [r for r in rows if r["mode"] == mode]
        if {r["sample_id"] for r in mode_rows} != set(data_eval.SAMPLE_IDS) or len(mode_rows) != 8:
            raise ValueError("All eight open-development cases, including failures, are required")
        selected = [r["selected"] for r in mode_rows]
        groups = {"all": [r["all"] for r in selected]}
        groups.update({stage: [r["stages"][stage] for r in selected] for stage in STAGES.values()})
        coverage = {}
        for name, scores in groups.items():
            length = sum(s["reference_length_px"] for s in scores)
            covered = sum(s["covered_reference_length_px"] or 0. for s in scores)
            coverage[name] = dict(reference_length_px=length, covered_reference_length_px=covered,
                coverage_5px=covered/length if length else None,
                images_with_reference=sum(s["reference_length_px"] > 0 for s in scores),
                image_population_denominator=8)
        summary[mode] = dict(image_population_denominator=8,
            selected_images=sum(r["selection_status"] == "selected" for r in mode_rows),
            rejected_images=sum(r["selection_status"] == "rejected" for r in mode_rows),
            source_group_count=3, selected_reference_coverage=coverage,
            reference_length_before_seed_exclusion_px=sum(r["source_reference"]["reference_length_before_seed_exclusion_px"] for r in mode_rows),
            reference_length_removed_as_input_px=sum(r["source_reference"]["reference_length_removed_as_input_px"] for r in mode_rows),
            visible_connection_evaluable_images=sum(r["legacy_v28_ribbon_diagnostic"]["visible_connection_evaluable"] for r in mode_rows),
            selected_legacy_ribbon_status_counts={name: sum(r["legacy_selected_visible_connection_status"] == name for r in mode_rows)
                                                for name in ("CORRECT", "WRONG", "UNKNOWN")},
            true_edge_identity_evaluable_images=0, whole_arc_identity_success=None)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=("AUTO", "CONDITIONAL"), default=["AUTO", "CONDITIONAL"])
    parser.add_argument("--output", type=Path, help="New evaluation output folder; default run/evaluation")
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output.resolve() if args.output else run / "evaluation"
    if output.exists():
        raise FileExistsError(f"Preserve earlier evaluation; choose a new output directory: {output}")
    inputs = {s["sample_id"]: s for s in data_eval.list_inputs(verify_images=False)}
    frozen = [_validate_case(run, mode, sid, inputs[sid]["image_shape_hw"])
              for mode in args.modes for sid in data_eval.SAMPLE_IDS]
    scene_module = importlib.import_module("experiments.entry_conditioned_rope_path.scene")
    # Only after the entire save boundary succeeds are target labels opened.
    rows = []
    references = {}
    for case in frozen:
        sid = case["sample_id"]
        if sid not in references:
            references[sid] = data_eval.load_evaluation_reference(sid)
        rows.append(evaluate_case(case, references[sid], scene_module.stage_at_points))
    summary = _aggregate(rows)
    output.mkdir(parents=True, exist_ok=False)
    for row in rows:
        _write_new(output / row["mode"] / (row["sample_id"] + ".json"), row)
    payload = dict(status="completed", exit_code=0, created_utc=datetime.now(timezone.utc).isoformat(),
        command=[sys.executable, *sys.argv], protocol=PROTOCOL,
        evaluator_sha256=sha256(Path(__file__)), scene_code_sha256=sha256(Path(scene_module.__file__)),
        summary=summary,
        prediction_save_boundary=[{k: c[k] for k in ("sample_id", "mode", "prediction_files")} for c in frozen])
    _write_new(output / "summary.json", payload)
    print(json.dumps({"status": "completed", "output": str(output), "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
