from pathlib import Path
import json
import os
import tempfile
import unittest

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from vision_geometry_lab.centerline_path_metrics_v6 import PROTOCOL, protocol_sha256
from vision_geometry_lab.evaluate_centerline_multiscale_v9 import V9_PROTOCOL, canonical_hash
from vision_geometry_lab.diagnose_centerline_probabilities_v9 import (
    FIXED_THRESHOLD,
    diagnose_arm,
    diagnose_pair,
    file_sha256,
    sample_reference_probabilities,
)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _make_arm(root, mode, sample_specs):
    root = Path(root)
    predictions = root/f"{mode}_predictions"
    predictions.mkdir()
    rows = []
    for index, spec in enumerate(sample_specs):
        sample_id = spec["sample_id"]
        coverage = np.zeros((4, 6), np.uint8)
        coverage[:, :3] = 1
        body = np.zeros((4, 6), np.float32)
        center = np.zeros((4, 6), np.float32)
        body[coverage == 1] = spec.get("body_probability", .8)
        center[coverage == 1] = spec.get("center_probability", .7)
        npz = predictions/f"{sample_id}.npz"
        np.savez_compressed(npz, body_probability=body,
                            center_probability=center, coverage_mask=coverage)
        output_hash = file_sha256(npz)
        metadata = dict(
            schema_version=1, experiment_version=9,
            format="centerline_multiscale_v9_seed_conditioned_pair",
            sample_id=sample_id, mode=mode,
            output_npz=str(npz.resolve()), output_sha256=output_hash,
            probability_map_hw=[4, 6], full_image_hw=[8, 12],
            ground_truth_center_or_edges_used_as_input=False,
            quality_used_for_input=False,
            coverage_restricts_reference_denominator=False,
        )
        _write_json(predictions/f"{sample_id}.json", metadata)
        decoded_reason = spec.get(
            "decoded_reason", "fixed_local_monotone_candidate_stopped_at_missing_evidence")
        decoded_segments = spec.get("decoded_segments", [])
        reference = spec.get("reference", [[[1., 3.], [10., 3.]]])
        rows.append(dict(
            sample_id=sample_id, split=spec["split"], cohort=spec["cohort"],
            group_id=spec.get("group_id", f"g{index}"),
            image_sha256=f"image-{sample_id}",
            annotation_sha256=f"annotation-{sample_id}",
            quality_sha256=f"quality-{sample_id}",
            prediction_status="loaded", prediction_source=dict(
                output_sha256=output_hash, mode=mode),
            reference_source="synthetic_trusted_reference",
            independent_width=dict(width_px=spec.get("width_px", 12.),
                                   reason=spec.get("width_reason", "synthetic"),
                                   rope_valid=spec.get("width_px", 12.) is not None),
            decoded=dict(segments=decoded_segments, widths=[], reason=decoded_reason,
                         identity_confirmed=False),
            metrics=dict(protocol=dict(PROTOCOL), protocol_sha256=protocol_sha256(),
                         score_support=dict(full_reference_segments=reference)),
        ))
    evaluation = root/f"{mode}_evaluation.json"
    _write_json(evaluation, dict(
        schema_version=1, experiment_version=9, mode=mode,
        manifest_sha256="same-manifest", protocol=V9_PROTOCOL,
        protocol_sha256=canonical_hash(V9_PROTOCOL),
        path_protocol_sha256=protocol_sha256(), rows=rows,
        no_rows_dropped=True, coverage_excluded_from_denominator=False,
    ))
    return evaluation, predictions


def _fake_context(sample_specs):
    records = {}
    for index, spec in enumerate(sample_specs):
        sample_id = spec["sample_id"]
        sample = dict(sample_id=sample_id, split=spec["split"],
                      group_id=spec.get("group_id", f"g{index}"),
                      image_shape_hw=[8, 12], image_sha256=f"image-{sample_id}")
        records[sample_id] = dict(
            sample=sample, annotation_sha256=f"annotation-{sample_id}",
            quality={"quality_sha256": f"quality-{sample_id}"},
            cohort=spec["cohort"],
        )
    return dict(manifest={}, manifest_path=Path("unused"),
                manifest_sha256="same-manifest", snapshot={}, records_by_id=records)


def _fake_prediction_loader(predictions_dir, record, context, mode):
    sample_id = record["sample"]["sample_id"]
    metadata = json.loads((Path(predictions_dir)/f"{sample_id}.json").read_text("utf-8"))
    path = Path(metadata["output_npz"])
    if file_sha256(path) != metadata["output_sha256"]:
        raise ValueError(f"V9 prediction hash mismatch: {sample_id}")
    with np.load(path, allow_pickle=False) as arrays:
        body = arrays["body_probability"].copy()
        center = arrays["center_probability"].copy()
        coverage = arrays["coverage_mask"].copy()
    if (not np.isfinite(body).all() or not np.isfinite(center).all()
            or np.any((body < 0) | (body > 1)) or np.any((center < 0) | (center > 1))):
        raise ValueError(f"Invalid probability/coverage values: {sample_id}")
    return metadata, body, center, coverage


class DiagnoseCenterlineProbabilitiesV9Tests(unittest.TestCase):
    def test_reference_sampling_retains_outside_coverage_in_denominator(self):
        coverage = np.zeros((4, 6), np.uint8)
        coverage[:, :3] = 1
        body = np.zeros((4, 6), np.float32)
        center = np.zeros((4, 6), np.float32)
        body[coverage == 1] = .8
        center[coverage == 1] = .7
        result = sample_reference_probabilities(
            [[[1., 3.], [10., 3.]]], body, center, coverage, [8, 12])
        self.assertEqual(result["max_sampling_spacing_px"], 4.)
        self.assertGreater(result["total_samples"], result["covered_samples"])
        self.assertEqual(result["samples_outside_coverage"],
                         result["total_samples"]-result["covered_samples"])
        self.assertEqual(
            result["body"]["all_reference_including_unsupported_as_zero"]["sample_count"],
            result["total_samples"])
        self.assertEqual(
            result["body"]["covered_reference_only"]["sample_count"],
            result["covered_samples"])
        all_fraction = result["body"]["all_reference_including_unsupported_as_zero"][
            "arc_length_weighted_fraction_ge_0_5"]
        covered_fraction = result["body"]["covered_reference_only"][
            "arc_length_weighted_fraction_ge_0_5"]
        self.assertLess(all_fraction, covered_fraction)
        self.assertEqual(covered_fraction, 1.)

    def test_pair_reports_every_row_stratum_heatmap_and_fixed_reason(self):
        specs = [
            dict(sample_id="old_train", split="train", cohort="old_pilot",
                 width_px=None, width_reason="no_width",
                 decoded_reason="independent_free_width_unavailable"),
            dict(sample_id="new_holdout", split="holdout", cohort="supplement",
                 body_probability=.9, center_probability=.8),
        ]
        with tempfile.TemporaryDirectory() as directory:
            native_eval, native_pred = _make_arm(directory, "native_detail", specs)
            low_eval, low_pred = _make_arm(directory, "lowpass_control", specs)
            context = _fake_context(specs)
            report = diagnose_pair("unused", native_eval, native_pred, low_eval, low_pred,
                                   require_fixed_inventory=False, _context=context,
                                   _prediction_loader=_fake_prediction_loader)
            self.assertEqual(report["status"], "posthoc_explanatory_not_model_selection")
            self.assertEqual(report["fixed_probability_threshold"], FIXED_THRESHOLD)
            self.assertTrue(report["no_rows_dropped"])
            self.assertFalse(report["interpretation"]["threshold_tuning_performed"])
            self.assertFalse(report["interpretation"]["decoder_rerun"])
            self.assertEqual(report["interpretation"]["ground_truth_use"],
                             "offline probability sampling only; never an inference input")
            for arm in report["arms"].values():
                self.assertEqual(len(arm["rows"]), 2)
                by_id = {row["sample_id"]: row for row in arm["rows"]}
                self.assertEqual(by_id["old_train"]["fixed_explanation"],
                                 "independent_width_gate_blocked_decode")
                self.assertEqual(
                    by_id["new_holdout"]["fixed_explanation"],
                    "both_heads_reach_fixed_0_5_on_reference_but_decoder_returns_no_segment")
                self.assertAlmostEqual(
                    by_id["new_holdout"]["probability"]["body_covered_window"]["max"], .9)
                self.assertEqual(
                    by_id["new_holdout"]["probability"]
                    ["high_response_outside_reference_interpretation"],
                    "not_false_positive_without_exhaustive_annotation")
                summaries = {(row["split"], row["cohort"]): row
                             for row in arm["stratum_summaries"]}
                self.assertEqual(summaries[("train", "all")]["sample_count"], 1)
                self.assertEqual(summaries[("holdout", "supplement")]["sample_count"], 1)
                self.assertGreater(
                    summaries[("holdout", "all")]["samples_outside_coverage"], 0)

    def test_export_hash_tamper_and_paired_inventory_mismatch_fail_closed(self):
        specs = [dict(sample_id="sample_a", split="train", cohort="old_pilot")]
        with tempfile.TemporaryDirectory() as directory:
            native_eval, native_pred = _make_arm(directory, "native_detail", specs)
            low_eval, low_pred = _make_arm(directory, "lowpass_control", specs)
            npz = native_pred/"sample_a.npz"
            with npz.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                diagnose_arm(native_eval, native_pred, "native_detail", "unused",
                             _context=_fake_context(specs),
                             _prediction_loader=_fake_prediction_loader)

            # Rebuild the native arm, then make the other evaluation inventory differ.
        with tempfile.TemporaryDirectory() as directory:
            native_eval, native_pred = _make_arm(directory, "native_detail", specs)
            low_eval, low_pred = _make_arm(
                directory, "lowpass_control",
                [dict(sample_id="sample_b", split="train", cohort="old_pilot")])
            with self.assertRaisesRegex(ValueError, "inventories differ"):
                diagnose_pair("unused", native_eval, native_pred, low_eval, low_pred,
                              require_fixed_inventory=False, _context=_fake_context(specs),
                              _prediction_loader=_fake_prediction_loader)

    def test_invalid_probability_or_unknown_decoder_reason_is_rejected(self):
        specs = [dict(sample_id="sample_a", split="train", cohort="old_pilot")]
        with tempfile.TemporaryDirectory() as directory:
            evaluation, predictions = _make_arm(directory, "native_detail", specs)
            report = json.loads(evaluation.read_text("utf-8"))
            report["rows"][0]["decoded"]["reason"] = "new_unfrozen_reason"
            _write_json(evaluation, report)
            with self.assertRaisesRegex(ValueError, "Unknown decoded"):
                diagnose_arm(evaluation, predictions, "native_detail", "unused",
                             _context=_fake_context(specs),
                             _prediction_loader=_fake_prediction_loader)

    def test_missing_export_is_unavailable_not_zero_heatmap(self):
        specs = [dict(sample_id="sample_a", split="train", cohort="old_pilot")]
        with tempfile.TemporaryDirectory() as directory:
            evaluation, predictions = _make_arm(directory, "native_detail", specs)
            report = json.loads(evaluation.read_text("utf-8"))
            row = report["rows"][0]
            row["prediction_status"] = "missing_prediction"
            row["prediction_source"] = None
            row["decoded"] = dict(segments=[], widths=[], reason="missing_prediction",
                                  identity_confirmed=False)
            _write_json(evaluation, report)
            result = diagnose_arm(
                evaluation, predictions, "native_detail", "unused",
                _context=_fake_context(specs), _prediction_loader=_fake_prediction_loader)
            diagnosed = result["rows"][0]
            self.assertIsNone(diagnosed["probability"])
            self.assertFalse(diagnosed["reference_sampling"]["probability_available"])
            self.assertIsNone(diagnosed["reference_sampling"]["covered_samples"])
            self.assertEqual(diagnosed["fixed_explanation"],
                             "prediction_unavailable_or_rejected")
            all_train = next(row for row in result["stratum_summaries"]
                             if row["split"] == "train" and row["cohort"] == "all")
            self.assertEqual(all_train["probability_available_reference_samples"], 0)
            self.assertGreater(all_train["probability_unavailable_reference_samples"], 0)

        with tempfile.TemporaryDirectory() as directory:
            evaluation, predictions = _make_arm(directory, "native_detail", specs)
            metadata = json.loads((predictions/"sample_a.json").read_text("utf-8"))
            with np.load(predictions/"sample_a.npz", allow_pickle=False) as arrays:
                body = arrays["body_probability"].copy()
                center = arrays["center_probability"].copy()
                coverage = arrays["coverage_mask"].copy()
            body[0, 0] = np.nan
            np.savez_compressed(predictions/"sample_a.npz", body_probability=body,
                                center_probability=center, coverage_mask=coverage)
            metadata["output_sha256"] = file_sha256(predictions/"sample_a.npz")
            _write_json(predictions/"sample_a.json", metadata)
            report = json.loads(evaluation.read_text("utf-8"))
            report["rows"][0]["prediction_source"]["output_sha256"] = metadata["output_sha256"]
            _write_json(evaluation, report)
            with self.assertRaisesRegex(ValueError, "Invalid probability"):
                diagnose_arm(evaluation, predictions, "native_detail", "unused",
                             _context=_fake_context(specs),
                             _prediction_loader=_fake_prediction_loader)


if __name__ == "__main__":
    unittest.main()
