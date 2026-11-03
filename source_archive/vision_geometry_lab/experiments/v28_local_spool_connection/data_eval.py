"""V28 fixed, already-open development inventory and post-prediction evaluation.

Only list_inputs is needed by AUTO. Labels are loaded only after a saved pool is
verified. A visible annotated ribbon match is not physical material identity.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from vision_geometry_lab.centerline_path_metrics_v6 import (
    _nearest, _sampling, _score_scope, clip_scoring_segments, evaluate_paths,
)

LAB = Path(__file__).resolve().parents[2]
MANIFEST = LAB / "outputs/v11_path_identity/development_manifest.json"
WIDTH_INDEX = LAB / "outputs/v11_path_identity/development_cache/index.json"
MANIFEST_SHA256 = "a6da6a323ea9bb3fdc67978e0fe65f97790acc0add76d4758d39bc3ae3f9fad7"
WIDTH_INDEX_SHA256 = "18cfa270628ae226b774e00d5bc7180fe03cba6c185fa3441ad307c13aac62ca"
SAMPLE_IDS = (
    "dual_20260923_113745__left__f00450",
    "dual_20260923_113745__left__f00550",
    "dual_20260923_113745__left__f00580",
    "dual_20260923_113305__right__f00250",
    "dual_20260923_113305__right__f00500",
    "dual_20260923_113305__right__f00550",
    "C04__dual_20260923_113029__t05__left",
    "C05__dual_20260923_113029__t25__left",
)
EVALUATION_PROTOCOL = dict(
    version=1, population="fixed historical eight open development images; not blind",
    old_geometry_tolerance_px=5., departure_from_free_line_widths=.5,
    legacy_bend_neighborhood_half_widths=2.,
    visible_branch_interval_widths=[-1., 8.],
    minimum_reference_pair_fraction=.75,
    minimum_branch_difference_widths=.5,
    minimum_distinct_branch_length_widths=1.,
    distinct_branch_requires="at least one centered candidate and another outside both reviewed target edges by 2px; geometric separation alone is insufficient",
    target_center_band_fraction=.3,
    minimum_match_fraction=.8, minimum_wrong_fraction=.5,
    outside_ribbon_margin_px=2., maximum_scored_offset_widths=3.,
    candidate_bend_coverage_threshold=.8,
    identity_policy="paired reviewed target edges plus a separated competing branch are required for visible connection assessment; no whole-rope or physical identity assertion",
    unknown_policy="quality exclusions and missing/ambiguous cross-sections are unknown, never background",
    labels_used_only_after_pool_saved=True, physical_identity_confirmed=False,
    bend_prediction_precision_policy="undefined: partial trusted bend is not an exhaustive negative reference; coverage only",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_bound(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Historical source hash changed: " + str(path))
    return json.loads(raw.decode("utf-8-sig"))


def _samples():
    manifest = _read_bound(MANIFEST, MANIFEST_SHA256)
    selected = {s["sample_id"]: s for s in manifest["samples"] if s["sample_id"] in SAMPLE_IDS}
    if set(selected) != set(SAMPLE_IDS):
        raise ValueError("Fixed V28 inventory changed")
    for sample in selected.values():
        if sample["split"] != "holdout" or "20260925" in json.dumps(sample):
            raise ValueError("Only the historical open 20260923 development subset is permitted")
    return [selected[sid] for sid in SAMPLE_IDS]


def list_inputs(verify_images=True):
    """Return image-only allowlist metadata; do not parse or hash annotations."""
    result = []
    for sample in _samples():
        if verify_images and sha256(sample["image_path"]) != sample["image_sha256"]:
            raise ValueError("Input image hash changed: " + sample["sample_id"])
        result.append({key: sample[key] for key in (
            "sample_id", "group_id", "image_path", "image_sha256", "image_shape_hw", "source")})
    return result


def _sample(sample_id):
    if sample_id not in SAMPLE_IDS:
        raise ValueError("Sample is outside V28's fixed image allowlist")
    return next(s for s in _samples() if s["sample_id"] == sample_id)


def _label(sample):
    label = _read_bound(sample["annotation_path"], sample["annotation_sha256"])
    if (label["sample_id"] != sample["sample_id"] or label["image_sha256"] != sample["image_sha256"]
            or label["coordinate_system"] != "full_image_xy" or label["reviewed"] is not True):
        raise ValueError("Reviewed label provenance mismatch")
    return label


def load_conditional_seed(sample_id):
    """Existing independent free_seed only; caller remeasures width from RGB.

    This explicit CONDITIONAL API parses the old annotation to select exactly
    the independent free_seed field. It never returns centerline/bend/edges.
    """
    sample = _sample(sample_id)
    return _label(sample)["polylines"]["free_seed"]


def load_evaluation_reference(sample_id):
    """Evaluator-only trusted geometry, with all quality exclusions retained."""
    from vision_geometry_lab.centerline_quality_v5 import load_label_quality
    sample = _sample(sample_id)
    label = _label(sample)
    quality = load_label_quality(sample)
    if quality["quality_sha256"] != sample.get("quality_sha256"):
        raise ValueError("Historical annotation quality changed")
    lines = quality.get("supervision_polylines", label["polylines"])
    width_rows = _read_bound(WIDTH_INDEX, WIDTH_INDEX_SHA256)["entries"]
    entry = next(r for r in width_rows if r["sample_id"] == sample_id)
    if entry["image_sha256"] != sample["image_sha256"] or entry["width"].get("target_geometry_used") is not False:
        raise ValueError("Historical independent width provenance mismatch")
    return dict(sample_id=sample_id, centerline=lines["centerline"],
                edge_flange=lines["edge_flange"], edge_surface=lines["edge_surface"],
                free_seed=label["polylines"]["free_seed"], width_px=entry["width"]["width_px"],
                ignore_polygons=label["ignore_polygons"] + quality["precision_exclude_polygons"],
                reviewed=label["reviewed"], status=label["status"],
                annotation_sha256=sample["annotation_sha256"], quality_sha256=quality["quality_sha256"],
                width_source="unchanged V11 original-image + independent free_seed measurement",
                branch_annotation_status="reviewed target center and both edges; adjacent rope IDs not annotated",
                physical_identity_confirmed=False)


def _arc(line):
    line = np.asarray(line, float)
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]


def _project_arc(points, line):
    line = np.asarray(line, float)
    arcs = _arc(line)
    best = np.full(len(points), np.inf)
    result = np.zeros(len(points))
    for i, (a, b) in enumerate(zip(line[:-1], line[1:])):
        delta = b - a
        length = np.linalg.norm(delta)
        if length <= 1e-10:
            continue
        t = np.clip((points - a) @ delta / length**2, 0, 1)
        distances = np.linalg.norm(points - (a + t[:, None] * delta), axis=1)
        keep = distances < best
        best[keep], result[keep] = distances[keep], arcs[i] + t[keep] * length
    return result


def _slice(line, start, end):
    line = np.asarray(line, float)
    arc = _arc(line)
    start, end = max(0., start), min(float(arc[-1]), end)
    if end <= start + 1e-8:
        return []
    first = [np.interp(start, arc, line[:, k]) for k in (0, 1)]
    last = [np.interp(end, arc, line[:, k]) for k in (0, 1)]
    return np.vstack([first, line[(arc > start) & (arc < end)], last]).tolist()


def _departure(reference, segments):
    seed = np.asarray(reference["free_seed"], float)
    direction = seed[-1] - seed[0]
    direction /= np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    width = reference["width_px"]
    if width is None or width <= 0:
        return None
    candidates = []
    for segment in segments:
        for a, b in zip(np.asarray(segment)[:-1], np.asarray(segment)[1:]):
            first, last = (np.array([a, b]) - seed[-1]) @ normal
            fraction = None
            if abs(first) >= .5 * width:
                fraction = 0.
            elif abs(last) >= .5 * width and abs(last-first) > 1e-10:
                fraction = ((.5 * width if last > 0 else -.5 * width) - first) / (last-first)
            if fraction is not None:
                point = a + fraction * (b-a)
                candidates.append((float(_project_arc(point[None], reference["centerline"])[0]), point))
    if not candidates:
        return None
    arc, point = min(candidates, key=lambda c: c[0])
    return dict(arc_px=arc, point_xy=point.tolist(), is_physical_contact=False,
                localization="GT posthoc only; never used to crop or generate candidates")


def _crossing(polyline, point, tangent):
    """One actual segment/cross-section intersection; never extrapolate edges."""
    line = np.asarray(polyline, float)
    values = []
    for a, b in zip(line[:-1], line[1:]):
        aa, bb = (np.array([a, b]) - point) @ tangent
        if abs(bb-aa) < 1e-8:
            continue
        t = -aa / (bb-aa)
        if -1e-8 <= t <= 1+1e-8:
            xy = a + np.clip(t, 0, 1) * (b-a)
            if not any(np.linalg.norm(xy-old) < 1e-5 for old in values):
                values.append(xy)
    return values[0] if len(values) == 1 else None


def _sections(reference, pieces, candidates):
    result = []
    edge_segments = [clip_scoring_segments([reference[key]], ignore_polygons=reference["ignore_polygons"])
                     for key in ("edge_flange", "edge_surface")]
    prediction_segments = {
        c["id"]: clip_scoring_segments([c["points_xy"]], ignore_polygons=reference["ignore_polygons"])
        for c in candidates
    }
    for piece in _sampling(pieces):
        for point, tangent, weight in zip(piece["points"], piece["tangents"], piece["weights"]):
            normal = np.array([-tangent[1], tangent[0]])
            edges = []
            for pieces_of_edge in edge_segments:
                hits = [_crossing(line, point, tangent) for line in pieces_of_edge]
                hits = [p for p in hits if p is not None]
                edges.append(float((hits[0]-point) @ normal) if len(hits) == 1 else None)
            valid = (all(e is not None for e in edges) and min(edges) < 0 < max(edges)
                     and max(edges)-min(edges) >= 4.)
            offsets = {}
            if valid:
                width = max(edges) - min(edges)
                for candidate in candidates:
                    hits = [_crossing(line, point, tangent) for line in prediction_segments[candidate["id"]]]
                    hits = [hit for hit in hits if hit is not None]
                    hit = hits[0] if len(hits) == 1 else None
                    offset = float((hit-point) @ normal) if hit is not None else None
                    offsets[candidate["id"]] = offset if offset is not None and abs(offset) <= 3*width else None
            result.append(dict(point_xy=point.tolist(), weight_px=float(weight),
                               paired_edges_valid=valid, edge_offsets_px=edges,
                               candidate_offsets_px=offsets))
    return result


def _bend_coverage(reference_segments, predicted_segments):
    """Legacy 5px reference coverage without scoring unannotated path as false.

    A short, trusted bend has no exhaustive negative annotation outside it.
    Therefore no prediction precision or extra-path error is computed here.
    """
    pieces = _sampling(reference_segments)
    length = float(sum(piece["length"] for piece in pieces))
    covered = float(sum(piece["weights"][_nearest(piece["points"], predicted_segments)[0] <= 5.].sum()
                        for piece in pieces))
    return dict(status="scored_reference" if length else "no_scored_reference",
                reference_length_px=length, reference_segment_count=len(pieces),
                reference_samples=sum(len(piece["points"]) for piece in pieces),
                covered_reference_length_px=covered if length else None,
                covered_fraction=covered/length if length else None,
                prediction_precision_at_5px=None, extra_predicted_fraction=None,
                extra_predicted_length_px=None, precision_score_available=False,
                precision_reason="partial trusted bend has no exhaustive negatives; unknown path is not background",
                identity_confirmed=False, physical_precision_validated=False)


def evaluate_reference(reference, pool, rankings, mode="AUTO"):
    """Pure evaluator useful for synthetic contracts; no IO or prediction edits."""
    candidates = pool["candidates"]
    ids = [c["id"] for c in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate IDs must be unique")
    empty_metrics = evaluate_paths(reference["centerline"], [], reference["free_seed"],
                                   ignore_polygons=reference["ignore_polygons"])
    support = empty_metrics["score_support"]["downstream_reference_segments"]
    departure = _departure(reference, support)
    bend, branch = [], []
    if departure:
        width, arc = reference["width_px"], departure["arc_px"]
        for lo, hi, target in ((arc-2*width, arc+2*width, bend), (arc-width, arc+8*width, branch)):
            line = _slice(reference["centerline"], lo, hi)
            if line:
                target.extend(clip_scoring_segments([line], free_seed=reference["free_seed"],
                                                    downstream=True, ignore_polygons=reference["ignore_polygons"]))
    sections = _sections(reference, branch, candidates)
    total = sum(s["weight_px"] for s in sections)
    paired = sum(s["weight_px"] for s in sections if s["paired_edges_valid"])
    distinct_length = 0.
    for s in sections:
        if s["paired_edges_valid"]:
            offsets = [v for v in s["candidate_offsets_px"].values() if v is not None]
            width = max(s["edge_offsets_px"])-min(s["edge_offsets_px"])
            low, high = min(s["edge_offsets_px"]), max(s["edge_offsets_px"])
            has_centered = any(abs(v) <= .3*width for v in offsets)
            has_excluded = any(v < low-2. or v > high+2. for v in offsets)
            if has_centered and has_excluded and max(offsets)-min(offsets) >= .5*width:
                distinct_length += s["weight_px"]
    evaluable = bool(reference.get("reviewed") and total and paired/total >= .75 and
                     distinct_length >= reference["width_px"])
    candidate_scores = []
    for candidate in candidates:
        pred = clip_scoring_segments([candidate["points_xy"]], ignore_polygons=reference["ignore_polygons"])
        bend_score = _bend_coverage(bend, pred)
        matched = wrong = covered = 0.
        for s in sections:
            if not s["paired_edges_valid"]:
                continue
            offset = s["candidate_offsets_px"].get(candidate["id"])
            if offset is None:
                continue
            weight = s["weight_px"]
            covered += weight
            low, high = min(s["edge_offsets_px"]), max(s["edge_offsets_px"])
            width = high-low
            matched += weight * (abs(offset) <= .3*width)
            wrong += weight * (offset < low-2. or offset > high+2.)
        match_fraction = matched/paired if paired else None
        wrong_fraction = wrong/paired if paired else None
        geom = "UNKNOWN"
        if paired and covered/paired >= .8:
            if match_fraction >= .8:
                geom = "MATCHES_ANNOTATED_RIBBON"
            elif wrong_fraction >= .5:
                geom = "DEPARTS_ANNOTATED_RIBBON"
        visible = {"MATCHES_ANNOTATED_RIBBON": "CORRECT", "DEPARTS_ANNOTATED_RIBBON": "WRONG"}.get(geom, "UNKNOWN") if evaluable else "UNKNOWN"
        candidate_scores.append(dict(id=candidate["id"], kind=candidate.get("kind"),
            bend_geometry_5px=bend_score, visible_ribbon_geometry_status=geom,
            visible_connection_status=visible, cross_section_coverage_fraction=covered/paired if paired else None,
            centered_target_ribbon_fraction=match_fraction, outside_target_ribbon_fraction=wrong_fraction,
            physical_identity_confirmed=False))
    by_id = {r["id"]: r for r in candidate_scores}
    choices = {}
    for group in ("G0", "G1", "G2"):
        ranking = rankings[group]
        selected = ranking.get("selected_id")
        if selected is not None and selected not in by_id:
            raise ValueError("Ranking references a candidate outside the saved pool")
        row = by_id.get(selected)
        reason = ("algorithm_rejected: " + str(ranking.get("reason"))) if selected is None else (
            "reviewed_target_ribbon_and_separated_competing_path" if evaluable else
            "insufficient_paired_target_edges_or_separated_competing_branch")
        choices[group] = dict(selected_id=selected, best_id=ranking.get("best_id"),
            algorithm_status=ranking.get("status"), rejected=selected is None,
            selected_visible_connection_status=row["visible_connection_status"] if row else "UNKNOWN",
            selected_visible_geometry_status=row["visible_ribbon_geometry_status"] if row else "UNKNOWN",
            bend_geometry_5px=row["bend_geometry_5px"] if row else _bend_coverage(bend, []),
            reason=reason, physical_identity_confirmed=False, whole_rope_identity_status="NOT_EVALUABLE")
    fractions = [r["bend_geometry_5px"]["covered_fraction"] for r in candidate_scores]
    best_coverage = max((v for v in fractions if v is not None), default=None)
    return dict(sample_id=reference.get("sample_id"), mode=mode,
        total_population_denominator=8,
        automatic_entry_success=bool(pool.get("automatic_entry_success", pool.get("entry_available", bool(candidates)))),
        candidate_generation_success=bool(candidates),
        reference_status=reference.get("status"), GT_departure=departure,
        trusted_bend_segments=bend, visible_branch_segments=branch,
        reference_length_px=empty_metrics["downstream"]["reference_length_px"],
        bend_reference_length_px=_bend_coverage(bend, [])["reference_length_px"],
        branch_reference_length_px=total, paired_reference_length_px=paired,
        distinct_competitor_length_px=distinct_length, visible_connection_evaluable=evaluable,
        candidate_pool_contains_trusted_bend_geometry=best_coverage is not None and best_coverage >= .8,
        best_pool_bend_coverage_5px=best_coverage,
        candidate_pool_contains_confirmed_rope_identity=None,
        candidate_scores=candidate_scores, choices=choices, branch_cross_sections=sections,
        evaluation_only_reference=reference, protocol=EVALUATION_PROTOCOL,
        physical_identity_confirmed=False, whole_rope_identity_status="NOT_EVALUABLE")


def evaluate_case(sample_id, pool, rankings, mode="AUTO", *, persisted_pool_path):
    """Enforce the prediction-save boundary before opening any target label."""
    saved = json.loads(Path(persisted_pool_path).read_text("utf-8-sig"))
    if saved != pool:
        raise ValueError("Saved candidate pool differs from the evaluator argument")
    return evaluate_reference(load_evaluation_reference(sample_id), pool, rankings, mode)


def summarize(rows):
    result = {}
    for mode in sorted({r["mode"] for r in rows}):
        selected = [r for r in rows if r["mode"] == mode]
        if len(selected) != 8 or {r["sample_id"] for r in selected} != set(SAMPLE_IDS):
            raise ValueError("Summary requires every fixed image including entry failures")
        result[mode] = dict(total_images=8,
            automatic_entry_failures=sum(not r["automatic_entry_success"] for r in selected),
            visible_connection_evaluable_images=sum(r["visible_connection_evaluable"] for r in selected),
            candidate_pool_contains_trusted_bend_geometry_images=sum(r["candidate_pool_contains_trusted_bend_geometry"] for r in selected),
            whole_rope_identity_evaluable_images=0, scores={})
        for group in ("G0", "G1", "G2"):
            choices = [r["choices"][group] for r in selected]
            denominator = sum(r["bend_reference_length_px"] for r in selected)
            covered = sum(c["bend_geometry_5px"]["covered_reference_length_px"] or 0 for c in choices)
            result[mode]["scores"][group] = dict(
                correct=sum(c["selected_visible_connection_status"] == "CORRECT" for c in choices),
                wrong=sum(c["selected_visible_connection_status"] == "WRONG" for c in choices),
                rejected=sum(c["rejected"] for c in choices),
                unevaluable_output=sum(not c["rejected"] and c["selected_visible_connection_status"] == "UNKNOWN" for c in choices),
                output_images=sum(not c["rejected"] for c in choices),
                effective_output_fraction=sum(not c["rejected"] for c in choices)/8,
                bend_coverage_5px=covered/denominator if denominator else None,
                bend_reference_length_px=denominator, physical_identity_confirmed=False)
        result[mode]["G1_to_G2_repairs"] = [r["sample_id"] for r in selected if r["choices"]["G1"]["selected_visible_connection_status"] == "WRONG" and r["choices"]["G2"]["selected_visible_connection_status"] == "CORRECT"]
        result[mode]["G1_to_G2_degradations"] = [r["sample_id"] for r in selected if r["choices"]["G1"]["selected_visible_connection_status"] == "CORRECT" and r["choices"]["G2"]["selected_visible_connection_status"] != "CORRECT"]
    return result
