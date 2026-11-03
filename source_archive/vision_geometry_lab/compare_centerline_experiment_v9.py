"""Summarize the frozen paired experiment; never choose or tune a model."""
from pathlib import Path
import argparse
import json
import statistics

from .run_centerline_experiment_v9 import ARMS, BASE, PROTOCOL, read, sha, verify_snapshot


def compare():
    protocol = read(PROTOCOL)
    verify_snapshot(protocol)
    reports = {mode: read(BASE/f"{mode}_300/report.json") for mode in ARMS}
    evaluations = {mode: read(BASE/f"{mode}_evaluation/summary.json") for mode in ARMS}
    exports = {mode: read(BASE/f"{mode}_exports/summary.json") for mode in ARMS}
    first, second = (reports[mode] for mode in ARMS)
    paired_checks = {}
    for key in ("initial_head_state_sha256", "sampler_sequence_sha256", "sampler_sequence",
                "steps_completed", "batch_size", "seed", "learning_rate", "center_weight"):
        paired_checks[key] = first["training"][key] == second["training"][key]
    paired_checks["data"] = first["data"] == second["data"]
    paired_checks["architecture"] = first["architecture"] == second["architecture"]
    for mode in ARMS:
        if not evaluations[mode]["provenance_complete"] or evaluations[mode]["invalid_prediction_count"]:
            raise ValueError(f"Incomplete provenance: {mode}")
        if len(evaluations[mode]["rows"]) != 18 or len(exports[mode]["exported"]) != 18:
            raise ValueError("Required samples missing")
        if reports[mode]["training"]["steps_completed"] != 300:
            raise ValueError("Fixed budget not followed")
    input_checks = []
    export_by_mode = {mode:{row["sample_id"]:row for row in exports[mode]["exported"]} for mode in ARMS}
    for sample_id in export_by_mode[ARMS[0]]:
        left, right = (read(export_by_mode[mode][sample_id]["metadata"]) for mode in ARMS)
        record = dict(sample_id=sample_id)
        for key in ("global_input_sha256", "free_seed_sha256", "tile_layout", "expected_coverage_mask_sha256",
                    "annotation_sha256", "quality_sha256"):
            record[key] = left[key] == right[key]
        record["local_rgb_changed"] = left["local_inputs_sha256"] != right["local_inputs_sha256"]
        input_checks.append(record)
    if not all(paired_checks.values()) or not all(all(v for k,v in row.items() if k != "sample_id") for row in input_checks):
        raise ValueError("Paired design mismatch; do not compare scores")
    summary = {}
    for mode in ARMS:
        times = [float(row["end_to_end_ms"]) for row in exports[mode]["exported"]]
        report = reports[mode]
        summary[mode] = dict(
            cohorts=evaluations[mode]["split_summaries"],
            cpu_export_end_to_end_ms=dict(median=statistics.median(times), min=min(times), max=max(times)),
            training_seconds=report["elapsed_seconds_before_report"],
            training_loop_seconds=report["training"]["loop_seconds"],
            zero_supervision_items=report["training"]["zero_supervision_items_sampled"],
            pure_zero_supervision_batches=report["training"]["pure_zero_supervision_batches"],
            train_pixel_diagnostic=report["train_pixel_diagnostic"],
            holdout_pixel_diagnostic=report["holdout_pixel_diagnostic"],
            source_files={suffix:sha(BASE/f"{mode}_{suffix}/summary.json")
                          for suffix in ("evaluation","exports")},
            checkpoint_sha256=report["artifacts"]["checkpoint_sha256"],
        )
    return dict(schema_version=1, protocol_sha256=sha(PROTOCOL), paired_checks=paired_checks,
                per_image_input_checks=input_checks, summaries=summary,
                status="frozen_pilot_completed_not_field_acceptance",
                limitations=["human free seed required", "partial references, not every rope identity",
                             "downstream is a seed half-plane, not measured physical contact",
                             "single seed and fixed training budget; reused small development holdout",
                             "timing includes file IO/preprocessing/model/stitch, excludes SDK/display/width/decoder",
                             "No layer or revolution-end timestamps tested"],
                deployment_authorized=False, hardware_access=False, force_command=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE/"paired_comparison.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite paired comparison")
    result = compare()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(dict(output=str(args.output), paired_design_verified=True)))


if __name__ == "__main__":
    main()
