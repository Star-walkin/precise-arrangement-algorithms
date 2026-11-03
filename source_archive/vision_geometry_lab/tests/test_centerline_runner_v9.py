import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from vision_geometry_lab import run_centerline_experiment_v9 as runner


def test_runner_commands_have_identical_fixed_budget():
    first = runner.commands("native_detail", "train")
    second = runner.commands("lowpass_control", "train")
    assert "--cache-samples" in first and "--cache-samples" in second
    for flag in ("--steps", "--seed", "--threads", "--batch-size", "--learning-rate"):
        assert first[first.index(flag)+1] == second[second.index(flag)+1]
    assert "train-manifest" in first
    assert runner.commands("native_detail", "evaluate")[-1].endswith("native_detail_evaluation")


def test_runner_snapshot_checks_nested_quality_and_missing_null():
    with TemporaryDirectory() as temp:
        root = Path(temp)
        image, label, quality = root/"image.png", root/"label.json", root/"label.quality.json"
        image.write_bytes(b"synthetic image")
        label.write_text("{}", encoding="utf-8")
        quality.write_text("{\"trusted\":true}", encoding="utf-8")
        manifest, inventory = root/"manifest.json", root/"inventory.json"
        manifest.write_text("{}", encoding="utf-8")
        row = dict(image_path=str(image), image_sha256=runner.sha(image),
                   annotation_path=str(label), annotation_sha256=runner.sha(label),
                   quality_snapshot=dict(path=str(quality), sha256=runner.sha(quality)))
        inventory.write_text(json.dumps(dict(samples=[row])), encoding="utf-8")
        with patch.object(runner, "INVENTORY", inventory), patch.object(runner, "MANIFEST", manifest):
            snapshot = runner.dataset_snapshot()
            assert snapshot[str(quality.resolve())] == runner.sha(quality)
            quality.unlink()
            try:
                runner.dataset_snapshot()
            except ValueError:
                pass
            else:
                raise AssertionError("Removing confirmed quality must fail closed")
            row["quality_snapshot"]["sha256"] = None
            inventory.write_text(json.dumps(dict(samples=[row])), encoding="utf-8")
            snapshot = runner.dataset_snapshot()
            assert snapshot[str(quality.resolve())] is None
            quality.write_text("{}", encoding="utf-8")
            try:
                runner.verify_snapshot(dict(dataset_snapshot=snapshot, source_snapshot={}))
            except ValueError:
                pass
            else:
                raise AssertionError("Creating previously absent quality invalidates frozen data")
