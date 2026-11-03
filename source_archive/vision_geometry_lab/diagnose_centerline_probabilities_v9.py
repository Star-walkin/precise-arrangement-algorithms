"""Post-hoc V9 probability diagnostic; never a model-selection protocol.

This read-only diagnostic explains an already-frozen evaluation.  It does not
train, decode, tune a threshold, or reinterpret high response outside partial
annotations as a false positive.  Ground-truth reference segments are used
only here to sample exported probabilities after inference.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json

import cv2
import numpy as np

from .centerline_path_metrics_v6 import _sampling, _weighted_percentile
from .centerline_review_v5 import read_clean_image
from .evaluate_centerline_multiscale_v9 import (
    V9_PROTOCOL, _dataset, canonical_hash, load_prediction,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "vision_geometry_lab/outputs/centerline_v9"
MODES = ("native_detail", "lowpass_control")
FIXED_THRESHOLD = 0.5
MAX_REFERENCE_SPACING_PX = 4.0
EXPECTED_SPLIT_COUNTS = {"train": 10, "holdout": 8}
COHORTS = ("all", "old_pilot", "supplement")
ALLOWED_DECODER_REASONS = {
    "free_seed_required",
    "independent_free_width_unavailable",
    "free_seed_too_short",
    "competing_ridge_stop",
    "fixed_local_monotone_candidate_stopped_at_missing_evidence",
    "missing_prediction",
    "rejected_prediction",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _probability_summary(values, weights, threshold=FIXED_THRESHOLD):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (values.ndim != 1 or weights.shape != values.shape or not np.isfinite(values).all()
            or not np.isfinite(weights).all() or np.any(weights < 0)):
        raise ValueError("Probability samples/weights must be finite matching vectors")
    count = int(len(values))
    above = int(np.count_nonzero(values >= float(threshold)))
    if not count:
        return dict(sample_count=0, p10=None, median=None, p90=None,
                    arc_length_weight_px=0.0, samples_ge_0_5=0,
                    sample_fraction_ge_0_5=None,
                    arc_length_ge_0_5_px=0.0, arc_length_weighted_fraction_ge_0_5=None)
    weight = float(weights.sum())
    above_weight = float(weights[values >= float(threshold)].sum())
    return dict(
        sample_count=count,
        p10=_weighted_percentile(values, weights, .10),
        median=_weighted_percentile(values, weights, .50),
        p90=_weighted_percentile(values, weights, .90),
        percentile_weighting="arc_length_voronoi",
        arc_length_weight_px=weight,
        samples_ge_0_5=above, sample_fraction_ge_0_5=above/count,
        arc_length_ge_0_5_px=above_weight,
        arc_length_weighted_fraction_ge_0_5=(above_weight/weight if weight else None),
    )


def _covered_map_summary(probability, coverage):
    values = np.asarray(probability)[np.asarray(coverage, dtype=bool)]
    return dict(covered_pixel_count=int(len(values)),
                max=float(values.max()) if len(values) else None,
                pixels_ge_0_5=int(np.count_nonzero(values >= FIXED_THRESHOLD)))


def _reference_samples(segments):
    pieces = _sampling(segments)
    if not pieces:
        return (np.empty((0, 2), dtype=np.float64),
                np.empty(0, dtype=np.float64))
    return (np.ascontiguousarray(np.concatenate([piece["points"] for piece in pieces]),
                                 dtype=np.float64),
            np.ascontiguousarray(np.concatenate([piece["weights"] for piece in pieces]),
                                 dtype=np.float64))


def sample_reference_probabilities(segments, body, center, coverage, full_image_hw):
    """Sample <=4 px trusted-reference points without dropping unsupported ones."""
    body = np.asarray(body, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    coverage = np.asarray(coverage)
    if body.ndim != 2 or center.shape != body.shape or coverage.shape != body.shape:
        raise ValueError("body, center and coverage must have one identical 2D shape")
    height, width = map(int, full_image_hw)
    if height <= 0 or width <= 0:
        raise ValueError("full_image_hw must be positive")
    points, weights = _reference_samples(segments)
    count = len(points)
    if not count:
        empty = _probability_summary(np.empty(0), np.empty(0))
        return dict(
            max_sampling_spacing_px=MAX_REFERENCE_SPACING_PX,
            probability_available=True,
            total_samples=0, covered_samples=0, samples_outside_coverage=0,
            outside_coverage_fraction=None, total_reference_arc_length_px=0.0,
            covered_reference_arc_length_px=0.0, outside_coverage_arc_length_px=0.0,
            outside_coverage_arc_length_fraction=None,
            body=dict(all_reference_including_unsupported_as_zero=empty,
                      covered_reference_only=empty),
            center=dict(all_reference_including_unsupported_as_zero=empty,
                        covered_reference_only=empty),
            joint_samples_ge_0_5=0, joint_fraction_ge_0_5=None,
            unsupported_probability_policy="retained in denominator and assigned zero",
        )
    scale = np.asarray([body.shape[1]/width, body.shape[0]/height], dtype=np.float64)
    grid = (points+0.5)*scale-0.5
    inside = ((grid[:, 0] >= 0) & (grid[:, 0] < body.shape[1]-1)
              & (grid[:, 1] >= 0) & (grid[:, 1] < body.shape[0]-1))
    map_x = grid[:, 0].astype(np.float32)[None]
    map_y = grid[:, 1].astype(np.float32)[None]

    def sample(array):
        return cv2.remap(np.asarray(array, dtype=np.float32), map_x, map_y,
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=0)[0].astype(np.float64)

    coverage_value = sample(coverage)
    supported = inside & (coverage_value >= 1.0-1e-6)
    body_raw, center_raw = sample(body), sample(center)
    body_all = np.where(supported, body_raw, 0.0)
    center_all = np.where(supported, center_raw, 0.0)
    joint = supported & (body_raw >= FIXED_THRESHOLD) & (center_raw >= FIXED_THRESHOLD)
    outside = int(count-np.count_nonzero(supported))
    total_weight = float(weights.sum())
    covered_weight = float(weights[supported].sum())
    joint_weight = float(weights[joint].sum())
    return dict(
        max_sampling_spacing_px=MAX_REFERENCE_SPACING_PX,
        probability_available=True,
        total_samples=int(count), covered_samples=int(np.count_nonzero(supported)),
        samples_outside_coverage=outside, outside_coverage_fraction=outside/count,
        total_reference_arc_length_px=total_weight,
        covered_reference_arc_length_px=covered_weight,
        outside_coverage_arc_length_px=total_weight-covered_weight,
        outside_coverage_arc_length_fraction=(
            (total_weight-covered_weight)/total_weight if total_weight else None),
        body=dict(
            all_reference_including_unsupported_as_zero=_probability_summary(body_all, weights),
            covered_reference_only=_probability_summary(body_raw[supported], weights[supported])),
        center=dict(
            all_reference_including_unsupported_as_zero=_probability_summary(center_all, weights),
            covered_reference_only=_probability_summary(center_raw[supported], weights[supported])),
        joint_samples_ge_0_5=int(np.count_nonzero(joint)),
        joint_fraction_ge_0_5=float(np.count_nonzero(joint)/count),
        joint_arc_length_ge_0_5_px=joint_weight,
        joint_arc_length_weighted_fraction_ge_0_5=(
            joint_weight/total_weight if total_weight else None),
        unsupported_probability_policy="retained in denominator and assigned zero",
    )


def _load_dataset_context(manifest_path):
    manifest_path = Path(manifest_path)
    manifest, records, snapshot = _dataset(manifest_path)
    identifiers = [record["sample"]["sample_id"] for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Manifest records are not unique")
    return dict(
        manifest=manifest, manifest_path=manifest_path,
        manifest_sha256=file_sha256(manifest_path), snapshot=snapshot,
        records_by_id={record["sample"]["sample_id"]: record for record in records},
    )


def _load_verified_export(predictions_dir, record, context, mode):
    """Reuse the frozen evaluator's full image/input/provenance/coverage gate."""
    image_rgb = np.asarray(read_clean_image(record["sample"]))
    loaded = load_prediction(
        predictions_dir, record, image_rgb, context["snapshot"],
        context["manifest_sha256"], mode)
    if loaded is None:
        raise ValueError(f"Loaded evaluation has no export: {record['sample']['sample_id']}")
    return loaded


def _unavailable_reference_sampling(segments):
    points, weights = _reference_samples(segments)
    total_weight = float(weights.sum())
    return dict(
        max_sampling_spacing_px=MAX_REFERENCE_SPACING_PX,
        probability_available=False,
        total_samples=int(len(points)), covered_samples=None,
        samples_outside_coverage=None, outside_coverage_fraction=None,
        total_reference_arc_length_px=total_weight,
        covered_reference_arc_length_px=None, outside_coverage_arc_length_px=None,
        outside_coverage_arc_length_fraction=None,
        body=None, center=None, joint_samples_ge_0_5=None,
        joint_fraction_ge_0_5=None, joint_arc_length_ge_0_5_px=None,
        joint_arc_length_weighted_fraction_ge_0_5=None,
        unsupported_probability_policy="unavailable map is not reclassified as zero response",
    )


def _fixed_explanation(prediction_status, decoded_count, decoded_reason, width, sampled):
    """Fixed explanatory label using only the existing 0.5 decoder threshold."""
    if prediction_status != "loaded":
        return "prediction_unavailable_or_rejected"
    if decoded_reason == "independent_free_width_unavailable" or width.get("width_px") is None:
        return "independent_width_gate_blocked_decode"
    if decoded_count:
        return "decoded_segments_present"
    if not sampled["total_samples"]:
        return "no_trusted_reference_samples"
    if not sampled["covered_samples"]:
        return "trusted_reference_outside_window_coverage"
    body_count = sampled["body"]["covered_reference_only"]["samples_ge_0_5"]
    center_count = sampled["center"]["covered_reference_only"]["samples_ge_0_5"]
    if sampled["joint_samples_ge_0_5"]:
        return "both_heads_reach_fixed_0_5_on_reference_but_decoder_returns_no_segment"
    if not body_count and not center_count:
        return "both_heads_below_fixed_0_5_on_covered_reference"
    if not body_count:
        return "body_head_below_fixed_0_5_on_covered_reference"
    if not center_count:
        return "center_head_below_fixed_0_5_on_covered_reference"
    return "heads_individually_reach_0_5_but_not_at_same_reference_samples"


def diagnose_arm(evaluation_path, predictions_dir, mode, manifest_path, *,
                 _context=None, _prediction_loader=None):
    if mode not in MODES:
        raise ValueError(f"Unknown V9 mode: {mode}")
    evaluation_path = Path(evaluation_path)
    report = _read_json(evaluation_path)
    context = _context if _context is not None else _load_dataset_context(manifest_path)
    prediction_loader = _prediction_loader or _load_verified_export
    if (report.get("schema_version") != 1 or report.get("experiment_version") != 9
            or report.get("mode") != mode or report.get("no_rows_dropped") is not True
            or report.get("coverage_excluded_from_denominator") is not False
            or report.get("protocol") != V9_PROTOCOL
            or report.get("protocol_sha256") != canonical_hash(V9_PROTOCOL)
            or report.get("manifest_sha256") != context["manifest_sha256"]):
        raise ValueError(f"Not a complete V9 evaluation for {mode}")
    rows = report.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Evaluation rows must be a list")
    identifiers = [row.get("sample_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in identifiers) or len(set(identifiers)) != len(rows):
        raise ValueError("Evaluation sample IDs must be unique nonempty strings")
    if set(identifiers) != set(context["records_by_id"]):
        raise ValueError("Evaluation/manifest sample inventories differ")
    diagnosed = []
    for row in rows:
        record = context["records_by_id"][row["sample_id"]]
        sample, quality = record["sample"], record["quality"]
        expected_row = dict(
            split=sample["split"], group_id=sample["group_id"],
            image_sha256=sample["image_sha256"],
            annotation_sha256=record["annotation_sha256"],
            quality_sha256=quality["quality_sha256"],
            cohort=record["cohort"],
        )
        for key, value in expected_row.items():
            if row.get(key) != value:
                raise ValueError(f"Evaluation/dataset mismatch ({key}): {row['sample_id']}")
        metrics = row.get("metrics") or {}
        protocol = metrics.get("protocol") or {}
        if (protocol.get("max_sampling_spacing_px") != MAX_REFERENCE_SPACING_PX
                or metrics.get("protocol_sha256") != report.get("path_protocol_sha256")):
            raise ValueError(f"Reference sampling protocol mismatch: {row['sample_id']}")
        segments = (metrics.get("score_support") or {}).get("full_reference_segments")
        if not isinstance(segments, list):
            raise ValueError(f"Missing trusted reference support: {row['sample_id']}")
        decoded = row.get("decoded") or {}
        decoded_segments = decoded.get("segments")
        decoded_reason = decoded.get("reason")
        if not isinstance(decoded_segments, list) or decoded_reason not in ALLOWED_DECODER_REASONS:
            raise ValueError(f"Unknown decoded result/reason: {row['sample_id']}")
        prediction_status = row.get("prediction_status")
        if prediction_status not in ("loaded", "missing_prediction", "rejected_prediction"):
            raise ValueError(f"Unknown prediction status: {row['sample_id']}")
        probability = None
        probability_source = None
        sampled = _unavailable_reference_sampling(segments)
        if prediction_status == "loaded":
            metadata, body, center, coverage = prediction_loader(
                predictions_dir, record, context, mode)
            source = row.get("prediction_source") or {}
            for key, value in source.items():
                if metadata.get(key) != value:
                    raise ValueError(
                        f"Evaluation/export provenance mismatch ({key}): {row['sample_id']}")
            sampled = sample_reference_probabilities(
                segments, body, center, coverage, sample["image_shape_hw"])
            sampled["probability_available"] = True
            probability = dict(
                body_covered_window=_covered_map_summary(body, coverage),
                center_covered_window=_covered_map_summary(center, coverage),
                reference=sampled,
                high_response_outside_reference_interpretation=(
                    "not_false_positive_without_exhaustive_annotation"),
            )
            probability_source = {key: metadata.get(key) for key in (
                "output_npz", "output_sha256", "checkpoint_sha256",
                "input_contract_sha256", "model_source_sha256",
                "contract_source_sha256", "training_report_sha256")}
        width = row.get("independent_width") or {}
        decoded_count = len(decoded_segments)
        diagnosed.append(dict(
            sample_id=row["sample_id"], split=row["split"], cohort=row["cohort"],
            group_id=row.get("group_id"), prediction_status=prediction_status,
            image_sha256=row["image_sha256"],
            annotation_sha256=row["annotation_sha256"],
            quality_sha256=row["quality_sha256"],
            reference_source=row.get("reference_source"),
            reference_support_sha256=hashlib.sha256(json.dumps(
                segments, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")).hexdigest(),
            trusted_reference_segment_count=len(segments),
            probability=probability, probability_source=probability_source,
            reference_sampling=sampled,
            decoded_segment_count=decoded_count,
            fixed_decoder_stop_reason=decoded_reason,
            independent_width=dict(width_px=width.get("width_px"), reason=width.get("reason"),
                                   rope_valid=width.get("rope_valid")),
            fixed_explanation=_fixed_explanation(
                prediction_status, decoded_count, decoded_reason, width, sampled),
            causal_claim=False,
        ))
    return dict(
        mode=mode, evaluation=str(evaluation_path.resolve()),
        evaluation_sha256=file_sha256(evaluation_path),
        predictions_dir=str(Path(predictions_dir).resolve()),
        manifest_sha256=report.get("manifest_sha256"),
        evaluation_protocol_sha256=report.get("protocol_sha256"),
        path_protocol_sha256=report.get("path_protocol_sha256"),
        rows=diagnosed, stratum_summaries=_aggregate(diagnosed),
    )


def _aggregate(rows):
    summaries = []
    for split in ("train", "holdout"):
        for cohort in COHORTS:
            selected = [row for row in rows
                        if row["split"] == split and (cohort == "all" or row["cohort"] == cohort)]
            reference_total = sum(row["reference_sampling"]["total_samples"] for row in selected)
            available = [row for row in selected if row["reference_sampling"]["probability_available"]]
            available_reference_total = sum(
                row["reference_sampling"]["total_samples"] for row in available)
            reference_covered = sum(row["reference_sampling"]["covered_samples"] for row in available)
            body_above = sum(
                row["reference_sampling"]["body"]["all_reference_including_unsupported_as_zero"]
                ["samples_ge_0_5"] for row in available)
            center_above = sum(
                row["reference_sampling"]["center"]["all_reference_including_unsupported_as_zero"]
                ["samples_ge_0_5"] for row in available)
            reference_arc = sum(row["reference_sampling"]["total_reference_arc_length_px"]
                                for row in selected)
            available_arc = sum(row["reference_sampling"]["total_reference_arc_length_px"]
                                for row in available)
            covered_arc = sum(row["reference_sampling"]["covered_reference_arc_length_px"]
                              for row in available)
            body_above_arc = sum(
                row["reference_sampling"]["body"]["all_reference_including_unsupported_as_zero"]
                ["arc_length_ge_0_5_px"] for row in available)
            center_above_arc = sum(
                row["reference_sampling"]["center"]["all_reference_including_unsupported_as_zero"]
                ["arc_length_ge_0_5_px"] for row in available)
            reasons = {}
            explanations = {}
            for row in selected:
                reasons[row["fixed_decoder_stop_reason"]] = reasons.get(row["fixed_decoder_stop_reason"], 0)+1
                explanations[row["fixed_explanation"]] = explanations.get(row["fixed_explanation"], 0)+1
            summaries.append(dict(
                split=split, cohort=cohort, sample_count=len(selected),
                loaded_prediction_count=sum(row["prediction_status"] == "loaded" for row in selected),
                decoded_segment_count=sum(row["decoded_segment_count"] for row in selected),
                empty_decoded_sample_count=sum(row["decoded_segment_count"] == 0 for row in selected),
                decoder_stop_reason_counts=reasons, fixed_explanation_counts=explanations,
                trusted_reference_samples=reference_total,
                trusted_reference_arc_length_px=reference_arc,
                probability_available_reference_samples=available_reference_total,
                probability_available_reference_arc_length_px=available_arc,
                probability_unavailable_reference_samples=reference_total-available_reference_total,
                probability_unavailable_reference_arc_length_px=reference_arc-available_arc,
                covered_reference_samples=reference_covered,
                samples_outside_coverage=available_reference_total-reference_covered,
                covered_reference_arc_length_px=covered_arc,
                outside_coverage_arc_length_px=available_arc-covered_arc,
                body_reference_samples_ge_0_5=body_above,
                body_reference_fraction_ge_0_5=(
                    body_above/available_reference_total if available_reference_total else None),
                body_reference_arc_length_ge_0_5_px=body_above_arc,
                body_reference_arc_length_weighted_fraction_ge_0_5=(
                    body_above_arc/available_arc if available_arc else None),
                center_reference_samples_ge_0_5=center_above,
                center_reference_fraction_ge_0_5=(
                    center_above/available_reference_total if available_reference_total else None),
                center_reference_arc_length_ge_0_5_px=center_above_arc,
                center_reference_arc_length_weighted_fraction_ge_0_5=(
                    center_above_arc/available_arc if available_arc else None),
                high_response_outside_reference_is_false_positive=False,
            ))
    return summaries


def diagnose_pair(manifest_path, native_evaluation, native_predictions, lowpass_evaluation,
                  lowpass_predictions, *, require_fixed_inventory=True,
                  _context=None, _prediction_loader=None):
    context = _context if _context is not None else _load_dataset_context(manifest_path)
    arms = {
        "native_detail": diagnose_arm(
            native_evaluation, native_predictions, "native_detail", manifest_path,
            _context=context, _prediction_loader=_prediction_loader),
        "lowpass_control": diagnose_arm(
            lowpass_evaluation, lowpass_predictions, "lowpass_control", manifest_path,
            _context=context, _prediction_loader=_prediction_loader),
    }
    native_rows = {row["sample_id"]: row for row in arms["native_detail"]["rows"]}
    control_rows = {row["sample_id"]: row for row in arms["lowpass_control"]["rows"]}
    if set(native_rows) != set(control_rows):
        raise ValueError("Paired V9 diagnostics require identical sample inventories")
    for sample_id in native_rows:
        first, second = native_rows[sample_id], control_rows[sample_id]
        if (first["split"], first["cohort"], first["group_id"]) != (
                second["split"], second["cohort"], second["group_id"]):
            raise ValueError(f"Paired sample strata differ: {sample_id}")
        for key in ("image_sha256", "annotation_sha256", "quality_sha256",
                    "reference_source", "reference_support_sha256"):
            if first[key] != second[key]:
                raise ValueError(f"Paired source/reference differs ({key}): {sample_id}")
    if arms["native_detail"]["manifest_sha256"] != arms["lowpass_control"]["manifest_sha256"]:
        raise ValueError("Paired evaluations use different manifests")
    if arms["native_detail"]["path_protocol_sha256"] != arms["lowpass_control"]["path_protocol_sha256"]:
        raise ValueError("Paired evaluations use different path protocols")
    if require_fixed_inventory:
        counts = {split: sum(row["split"] == split for row in native_rows.values())
                  for split in EXPECTED_SPLIT_COUNTS}
        if counts != EXPECTED_SPLIT_COUNTS or len(native_rows) != sum(EXPECTED_SPLIT_COUNTS.values()):
            raise ValueError(f"Expected fixed V9 10/8 inventory, got {counts}")
    return dict(
        schema_version=1, experiment_version=9,
        status="posthoc_explanatory_not_model_selection",
        diagnostic_source=str(Path(__file__).resolve()),
        diagnostic_source_sha256=file_sha256(__file__),
        manifest=str(Path(manifest_path).resolve()),
        fixed_probability_threshold=FIXED_THRESHOLD,
        reference_sampling_max_spacing_px=MAX_REFERENCE_SPACING_PX,
        path_protocol_sha256=arms["native_detail"]["path_protocol_sha256"],
        manifest_sha256=arms["native_detail"]["manifest_sha256"],
        paired_inventory_verified=True, no_rows_dropped=True,
        arms=arms,
        interpretation=dict(
            purpose=("separate missing width/decode gating, fixed-0.5 heatmap support, and "
                     "seed-window coverage after the frozen evaluation"),
            not_model_selection=True, threshold_tuning_performed=False,
            decoder_rerun=False, training_run=False,
            ground_truth_use="offline probability sampling only; never an inference input",
            unsupported_reference="retained in every all-reference denominator and reported separately",
            generic_decoder_reason=(
                "fixed_local_monotone_candidate_stopped_at_missing_evidence combines several "
                "fixed decoder exits and is not a unique causal gate"),
            unlabelled_high_response=(
                "not called false positive because partial references are not exhaustive negatives"),
            fixed_explanation_is_causal_proof=False,
            field_accuracy_or_automation_claim=False,
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=BASE/"dataset/manifest.json")
    parser.add_argument("--native-evaluation", type=Path,
                        default=BASE/"native_detail_evaluation/summary.json")
    parser.add_argument("--native-predictions", type=Path,
                        default=BASE/"native_detail_exports/predictions")
    parser.add_argument("--lowpass-evaluation", type=Path,
                        default=BASE/"lowpass_control_evaluation/summary.json")
    parser.add_argument("--lowpass-predictions", type=Path,
                        default=BASE/"lowpass_control_exports/predictions")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Use a new output directory; post-hoc diagnostics are immutable")
    cv2.setNumThreads(2)
    report = diagnose_pair(args.manifest, args.native_evaluation, args.native_predictions,
                           args.lowpass_evaluation, args.lowpass_predictions)
    args.output.mkdir(parents=True)
    destination = args.output/"summary.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(dict(report=str(destination), samples=len(report["arms"]["native_detail"]["rows"]),
                          status=report["status"]), ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
