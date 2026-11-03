"""Freeze and execute the finite V9 offline paired experiment, without tuning.

The immutable protocol is written before either real-data training run. Each
stage rechecks source/data snapshots and refuses to overwrite prior outputs.
This is an experiment runner, not a live detector or a hardware controller.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "vision_geometry_lab"
BASE = LAB / "outputs/centerline_v9"
PROTOCOL = BASE / "paired_protocol.json"
MANIFEST = BASE / "dataset/manifest.json"
INVENTORY = BASE / "dataset/inventory.json"
ARMS = ("native_detail", "lowpass_control")
SOURCE_NAMES = (
    "run_centerline_experiment_v9.py", "prepare_centerline_manifest_v9.py",
    "centerline_multiscale_v9.py", "centerline_multiscale_contract_v9.py",
    "evaluate_centerline_multiscale_v9.py", "centerline_model_v5.py",
    "centerline_review_v5.py", "centerline_quality_v5.py", "prepare_centerline_v5.py",
    "centerline_quality_cutoff_v8.py", "centerline_benchmark_v5.py",
    "centerline_path_metrics_v6.py", "rope_detector.py",
)


def sha(path):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dataset_snapshot():
    inventory = read(INVENTORY)
    snapshot = {str(MANIFEST.resolve()): sha(MANIFEST), str(INVENTORY.resolve()): sha(INVENTORY)}
    for sample in inventory["samples"]:
        for key in ("image", "annotation", "quality"):
            path_key, hash_key = key + "_path", key + "_sha256"
            if path_key not in sample:
                if key == "quality":
                    path = str(Path(sample["annotation_path"]).with_suffix(".quality.json"))
                else:
                    raise ValueError(f"Inventory missing {path_key}")
            else:
                path = sample[path_key]
            expected = sample.get(hash_key)
            if key == "quality":
                path = sample["quality_snapshot"]["path"]
                expected = sample["quality_snapshot"]["sha256"]
            if sha(path) != expected:
                raise ValueError(f"Inventory source changed: {path}")
            snapshot[str(Path(path).resolve())] = expected
    return snapshot


def verify_snapshot(protocol):
    for category in ("source_snapshot", "dataset_snapshot"):
        for path, expected in protocol[category].items():
            if sha(path) != expected:
                raise ValueError(f"Frozen {category} changed: {path}")


def commands(mode, stage):
    train_python = str(ROOT / "vision_training/.venv/Scripts/python.exe")
    cv_python = str(ROOT / "vision_prototype/.venv/Scripts/python.exe")
    run_dir, export_dir = BASE / f"{mode}_300", BASE / f"{mode}_exports"
    common = ["-X", "utf8", "-m"]
    if stage == "train":
        return [train_python, *common, "vision_geometry_lab.centerline_multiscale_v9",
                "train-manifest", "--manifest", str(MANIFEST), "--output-dir", str(run_dir),
                "--mode", mode, "--steps", "300", "--seed", "20260927", "--threads", "2",
                "--batch-size", "2", "--base-channels", "8", "--learning-rate", "0.001",
                "--center-weight", "1.0", "--cache-samples"]
    if stage == "export":
        return [train_python, *common, "vision_geometry_lab.centerline_multiscale_v9",
                "infer-manifest", "--manifest", str(MANIFEST), "--checkpoint", str(run_dir / "checkpoint.pt"),
                "--output-dir", str(export_dir), "--split", "all", "--threads", "2", "--tile-batch-size", "1"]
    return [cv_python, *common, "vision_geometry_lab.evaluate_centerline_multiscale_v9",
            "--manifest", str(MANIFEST), "--predictions-dir", str(export_dir / "predictions"),
            "--mode", mode, "--output", str(BASE / f"{mode}_evaluation")]


def freeze():
    if PROTOCOL.exists():
        raise FileExistsError("Protocol already exists; do not rewrite it after seeing results")
    manifest = read(MANIFEST)
    samples = manifest["samples"]
    if [sum(s["split"] == split for s in samples) for split in ("train", "holdout")] != [10, 8]:
        raise ValueError("Expected the reviewed 10/8 split")
    group_splits = {}
    for sample in samples:
        group_splits.setdefault(sample["group_id"], set()).add(sample["split"])
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("Source group leakage")
    if any(s["sample_id"].startswith(("C03__", "C06__")) for s in samples):
        raise ValueError("Diagnostic-only data admitted to geometry experiment")
    if any((BASE / f"{mode}_300").exists() for mode in ARMS):
        raise ValueError("Real-data run directories already exist; cannot claim preregistration")
    from .evaluate_centerline_multiscale_v9 import V9_PROTOCOL, canonical_hash, validate_frozen_protocol
    frozen_path_protocol = BASE / "path_protocol_v1/protocol.json"
    validate_frozen_protocol(frozen_path_protocol)
    protocol = dict(
        schema_version=1, experiment_version=9,
        frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        question="Does retaining local RGB detail improve seed-conditioned wound-path recognition in this paired pilot?",
        real_training_authorized=True, hardware_access=False, force_command=None,
        training=dict(arms=list(ARMS), steps=300, seed=20260927, batch_size=2,
                      threads=2, base_channels=8, learning_rate=0.001, center_weight=1.0,
                      optimizer="Adam", holdout_early_stopping=False, augmentation=None, cache_samples=True),
        identical_between_arms=["initial weights", "deterministic sample/tile sequence", "supervision",
                                "global RGB+seed input", "local seed", "architecture", "tile layout", "decoder"],
        changed_between_arms="Local RGB only: native versus fit512x1024/down-up lowpass",
        scope=dict(full_roi=[0,0,4096,3000], global_input_hw=[256,384], local_native_hw=[768,768],
                   local_model_hw=[384,384], tiles_per_image=9, target_based_windows=False,
                   human_free_seed_required=True, fully_automatic=False,
                   global_context="8-channel pooled descriptor, not HRNet or proven spatial reasoning",
                   c07_c08_last_trusted_center_index=7, c03_c06="inventory diagnostics only",
                   zero_supervision_tiles="retained; zero masked loss, not resampled",
                   unknown_quality_pixels="excluded, not positive or negative"),
        evaluation=dict(protocol=V9_PROTOCOL, protocol_sha256=canonical_hash(V9_PROTOCOL),
                        primary="5 native-pixel arc-length weighted downstream coverage and prediction precision",
                        secondary=["full-path score", "empty/zero coverage counts", "window support", "uncached CPU latency"],
                        width="RGB + human free_seed only; never annotated two-edge width",
                        denominator="All trusted references, including outside tile coverage and failed predictions",
                        cohorts=["train", "holdout", "old held-out 6", "new held-out C04/C05"],
                        caveat="Downstream is seed-defined, not a physical contact point; labels are partial; small reused development holdout, not new blind acceptance"),
        decision_policy=["No threshold, training-step, window or decoder changes after inspecting holdout scores",
                         "Report negative/failed results without selecting the prettier arm",
                         "A change relative to V7 cannot isolate resolution because target/model/width differ",
                         "No field acceptance, automatic layer recognition or force timing claimed from path scores"],
        known_limitations=["Old f246/f114 quality is legacy polygon masking after target rasterization; both arms identical, unlike the explicit pre-raster C7 cutoff",
                           "Zero-supervision batches have zero loss/gradient but Adam may apply existing momentum; same sequence in both arms",
                           "Single fixed 300-step seed is an exploratory budget, not a convergence or generalization guarantee"],
        source_snapshot={str((LAB/name).resolve()): sha(LAB/name) for name in SOURCE_NAMES},
        dataset_snapshot={**dataset_snapshot(), str(frozen_path_protocol.resolve()): sha(frozen_path_protocol)},
        group_splits={group:next(iter(splits)) for group,splits in sorted(group_splits.items())},
        planned_commands={mode:{stage:commands(mode,stage) for stage in ("train","export","evaluate")}
                          for mode in ARMS},
    )
    if any(value is None for value in protocol["source_snapshot"].values()):
        raise ValueError("Missing source module")
    verify_snapshot(protocol)
    with PROTOCOL.open("x", encoding="utf-8") as stream:
        json.dump(protocol, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(dict(protocol=str(PROTOCOL), sha256=sha(PROTOCOL)), ensure_ascii=False))


def run_stage(mode, stage):
    protocol = read(PROTOCOL)
    verify_snapshot(protocol)
    command = commands(mode, stage)
    if command != protocol["planned_commands"][mode][stage]:
        raise ValueError("Command differs from preregistration")
    print(json.dumps(dict(stage=stage, mode=mode, command=command, protocol_sha256=sha(PROTOCOL))), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    verify_snapshot(protocol)
    print(json.dumps(dict(stage=stage, mode=mode, frozen_sources_unchanged=True)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "verify", "train", "export", "evaluate"))
    parser.add_argument("--mode", choices=ARMS)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
    elif args.stage == "verify":
        verify_snapshot(read(PROTOCOL))
        print("Frozen source/data snapshots verified")
    else:
        if args.mode is None:
            parser.error("--mode required for an execution stage")
        run_stage(args.mode, args.stage)


if __name__ == "__main__":
    main()
