"""Verify the two human-confirmed C7 supervision cutoffs without training a model.

This is an integration/provenance audit.  Its self-scores only prove that the
quality half-plane is applied consistently; they are not recognition accuracy.
"""
from pathlib import Path
import argparse
import copy
import gc
import hashlib
import json
import os
import subprocess

import numpy as np
import torch

from .centerline_context_v6 import (
    letterbox_native_input,
    rasterize_letterbox_record,
)
from .centerline_model_v5 import (
    _resize_native_model_input,
    inspect_manifest,
    manifest_native_input,
    masked_two_head_loss,
    rasterize_manifest_record,
)
BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
PROTOTYPE_PY = ROOT / "vision_prototype/.venv/Scripts/python.exe"
DEFAULT_MANIFEST = BASE / "outputs/centerline_v7/review_batch01/manifest.json"
DEFAULT_AUDIT = BASE / "outputs/centerline_v8_intake_audit/audit.json"
DEFAULT_OUTPUT = BASE / "outputs/centerline_v8_quality_confirmed/integration_verification.json"
TARGET_KEYS = ("centerline", "edge_flange", "edge_surface")


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _evaluate_paths(reference, predictions, seed, ignored):
    """Run the frozen scorer in its existing OpenCV-capable environment."""
    if not PROTOTYPE_PY.is_file():
        raise FileNotFoundError(f"Prototype Python not found: {PROTOTYPE_PY}")
    payload = dict(reference=np.asarray(reference, float).tolist(),
                   predicted_segments=[np.asarray(line, float).tolist() for line in predictions],
                   free_seed=np.asarray(seed, float).tolist(),
                   ignore_polygons=[np.asarray(polygon, float).tolist() for polygon in ignored])
    code = r'''
import json,sys
from vision_geometry_lab.centerline_path_metrics_v6 import evaluate_paths
p=json.load(sys.stdin)
print(json.dumps(evaluate_paths(p['reference'],p['predicted_segments'],p['free_seed'],
                                ignore_polygons=p['ignore_polygons']),allow_nan=False))
'''
    environment = dict(os.environ)
    environment["PYTHONUTF8"] = "1"
    result = subprocess.run(
        [str(PROTOTYPE_PY), "-c", code], cwd=ROOT, env=environment,
        input=json.dumps(payload, allow_nan=False), text=True, capture_output=True, timeout=60,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode:
        raise RuntimeError(f"Path scoring subprocess failed:\n{result.stderr}")
    return json.loads(result.stdout)


def _array_hash(*arrays):
    digest = hashlib.sha256()
    for source in arrays:
        array = np.ascontiguousarray(source)
        header = json.dumps(
            {"shape": list(array.shape), "dtype": str(array.dtype)},
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(array.tobytes())
    return digest.hexdigest()


def _input_hashes(record):
    rgb, seed = manifest_native_input(record)
    native_hash = _array_hash(rgb, seed)
    square_rgb, square_seed = _resize_native_model_input(rgb, seed, [512, 512])
    square_hash = _array_hash(square_rgb, square_seed)
    letterbox, geometry = letterbox_native_input(rgb, seed, [512, 1024])
    letterbox_hash = _array_hash(letterbox.numpy())
    result = dict(
        native_rgb_seed_sha256=native_hash,
        v5_square_model_input_sha256=square_hash,
        v6_letterbox_model_input_sha256=letterbox_hash,
        v6_letterbox_geometry=geometry,
    )
    del rgb, seed, square_rgb, square_seed, letterbox
    gc.collect()
    return result


def _gradient_audit(sample):
    body_logits = torch.zeros_like(sample["body_target"][None], requires_grad=True)
    center_logits = torch.zeros_like(sample["center_target"][None], requires_grad=True)
    losses = masked_two_head_loss(
        {"body_logits": body_logits, "center_logits": center_logits},
        sample["body_target"][None], sample["center_target"][None],
        body_valid_mask=sample["body_valid_mask"][None],
        center_valid_mask=sample["center_valid_mask"][None],
    )
    losses["loss"].backward()
    body_valid = sample["body_valid_mask"][None] > 0
    center_valid = sample["center_valid_mask"][None] > 0
    result = dict(
        finite_loss=bool(torch.isfinite(losses["loss"])),
        body_valid_pixels=int(body_valid.sum()),
        center_valid_pixels=int(center_valid.sum()),
        body_invalid_gradient_abs_sum=float(body_logits.grad[~body_valid].abs().sum()),
        center_invalid_gradient_abs_sum=float(center_logits.grad[~center_valid].abs().sum()),
    )
    if result["body_invalid_gradient_abs_sum"] != 0.0:
        raise AssertionError("V8 excluded V5/V6 body pixels contributed loss gradient")
    if result["center_invalid_gradient_abs_sum"] != 0.0:
        raise AssertionError("V8 excluded V5/V6 center pixels contributed loss gradient")
    return result


def _raster_audits(record):
    square = rasterize_manifest_record(record, 512)
    square_result = _gradient_audit(square)
    del square
    gc.collect()
    letterbox = rasterize_letterbox_record(record, [512, 1024])
    letterbox_result = _gradient_audit(letterbox)
    del letterbox
    gc.collect()
    return dict(v5_square_512=square_result, v6_letterbox_512x1024=letterbox_result)


def _sample_result(record, audit_row):
    annotation_path = Path(record["annotation_path"])
    annotation_hash = _file_hash(annotation_path)
    if annotation_hash != audit_row["annotation_sha256"]:
        raise AssertionError("Raw annotation changed relative to the frozen V8 intake audit")
    if annotation_hash != record["annotation_sha256"]:
        raise AssertionError("inspect_manifest annotation provenance mismatch")

    supervision = record.get("parsed_supervision_polylines")
    if not isinstance(supervision, dict) or set(supervision) != set(TARGET_KEYS):
        raise AssertionError("Trusted supervision must contain exactly the three target curves")
    if len(supervision["centerline"]) != 7:
        raise AssertionError("Trusted centre supervision must stop at one-based C7")
    if len(record["parsed_polylines"]["centerline"]) != 8:
        raise AssertionError("Raw C8 annotation was unexpectedly changed")

    regions = [region for region in record["quality_regions"]
               if region["region_id"] == "human_confirmed_after_center_c7"]
    if len(regions) != 1:
        raise AssertionError("Expected one C7 cutoff quality region")
    region = regions[0]
    origin = np.asarray(region["cutoff_origin_xy"], float)
    tangent = np.asarray(region["cutoff_unit_tangent_xy"], float)
    normal = np.asarray([-tangent[1], tangent[0]])
    if not np.allclose(supervision["centerline"][-1], origin, rtol=0., atol=1e-7):
        raise AssertionError("Trusted centre does not end at cutoff origin C7")
    edge_endpoints = {name: np.asarray(supervision[name][-1], float)
                      for name in ("edge_flange", "edge_surface")}
    signed = {name: float((point-origin) @ tangent)
              for name, point in edge_endpoints.items()}
    offsets = {name: float((point-origin) @ normal)
               for name, point in edge_endpoints.items()}
    if max(abs(value) for value in signed.values()) > 1e-7:
        raise AssertionError("Trusted edge endpoints are not in the C7 normal cross-section")
    if offsets["edge_flange"] * offsets["edge_surface"] >= 0:
        raise AssertionError("Trusted edge endpoints do not bracket centre C7")

    baseline_input = _input_hashes(record)
    input_control = copy.copy(record)
    input_control.pop("parsed_supervision_polylines", None)
    input_control["parsed_quality_exclude_polygons"] = []
    control_input = _input_hashes(input_control)
    input_equal = baseline_input == control_input
    if not input_equal:
        raise AssertionError("Quality/target supervision changed RGB/free-seed model input")

    ignored = [*record["parsed_ignore_polygons"],
               *record["parsed_quality_exclude_polygons"]]
    raw = record["parsed_polylines"]
    metrics = _evaluate_paths(raw["centerline"], [raw["centerline"]],
                              raw["free_seed"], ignored)
    support = metrics["score_support"]["full_reference_segments"]
    if len(support) != 1 or not np.allclose(support[0][-1], origin, rtol=0., atol=1e-7):
        raise AssertionError("Scored reference did not stop at the C7 cutoff")

    return dict(
        sample_id=record["sample_id"], split=record["split"], group_id=record["group_id"],
        annotation_path=str(annotation_path.resolve()), annotation_sha256=annotation_hash,
        annotation_sha256_matches_intake_audit=True,
        quality_sha256=record["quality_sha256"],
        raw_center_point_count=len(raw["centerline"]),
        trusted_supervision_keys=sorted(supervision),
        trusted_center_point_count=len(supervision["centerline"]),
        trusted_center_ends_at_c7=True,
        edge_endpoint_cut_plane_signed_distance_px=signed,
        edge_endpoint_cross_section_offsets_px=offsets,
        edge_endpoints_bracket_center=True,
        input_hashes=baseline_input,
        input_hash_equal_after_removing_quality_and_trusted_targets=input_equal,
        raster_loss_mask_audit=_raster_audits(record),
        scoring_cutoff_self_consistency_only=dict(
            protocol_sha256=metrics["protocol_sha256"],
            full_reference_length_px=metrics["full_path"]["reference_length_px"],
            downstream_reference_length_px=metrics["downstream"]["reference_length_px"],
            full_prediction_length_px=metrics["full_path"]["predicted_length_px"],
            downstream_prediction_length_px=metrics["downstream"]["predicted_length_px"],
            scored_reference_endpoint_xy=support[0][-1],
            endpoint_is_c7=True,
            unknown_tail_excluded_from_primary_score=True,
            prediction_is_raw_reference_for_cutoff_self_check=True,
            recognition_accuracy_measurement=False,
        ),
    )


def verify(manifest_path, audit_path):
    torch.set_num_threads(2)
    inspection = inspect_manifest(manifest_path)
    audit = json.loads(Path(audit_path).read_text("utf-8-sig"))
    audit_rows = {row["sample_id"]: row for row in audit["samples"]}
    records = [record for split in ("train", "holdout")
               for record in inspection["usable"][split]
               if record["sample_id"].startswith(("C07__", "C08__"))]
    if len(records) != 2:
        raise AssertionError("The fixed eight-sample manifest must contain C07 and C08 exactly once")
    results = []
    for record in records:
        results.append(_sample_result(record, audit_rows[record["sample_id"]]))
        print(f"verified {record['sample_id']}", flush=True)
    return dict(
        schema_version=1,
        artifact_type="centerline_v8_quality_cutoff_integration_verification",
        status="passed",
        manifest=str(Path(manifest_path).resolve()),
        manifest_sha256=_file_hash(manifest_path),
        intake_audit=str(Path(audit_path).resolve()),
        intake_audit_sha256=_file_hash(audit_path),
        path_protocol_sha256=results[0]["scoring_cutoff_self_consistency_only"]["protocol_sha256"],
        torch_threads=2,
        samples=results,
        regression_contract=dict(
            test_file=str((BASE / "tests/test_centerline_cutoff_integration_v8.py").resolve()),
            v5_and_v6_unknown_raw_tail_invariance=True,
            quality_unknown_pixels_zero_loss_gradient=True,
            quality_and_target_geometry_absent_from_model_input=True,
            legacy_missing_field_behavior_preserved=True,
            primary_path_scores_clip_reference_and_prediction=True,
        ),
        scope_and_claims=dict(
            model_training_performed=False,
            checkpoint_created_or_modified=False,
            hardware_or_production_access=False,
            this_verification_modified_labels_or_quality_sidecars=False,
            recognition_accuracy_established=False,
            physical_contact_or_layer_transition_established=False,
            purpose="schema, provenance, raster-loss masking, input isolation, and score-cutoff integration only",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--intake-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = verify(args.manifest, args.intake_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                           encoding="utf-8")
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
