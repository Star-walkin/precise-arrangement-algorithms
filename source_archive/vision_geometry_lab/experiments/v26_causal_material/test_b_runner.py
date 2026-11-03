"""B driver tests: synthetic arrays, fake adapter interface, no real video/model."""
import copy
import json
import os

import numpy as np
from PIL import Image
import pytest

from vision_geometry_lab.experiments.v26_causal_material import b_runner as b


BOX = [100, 200, 1024, 1024]
SEEDS = [[250., 500.], [350., 501.], [450., 510.], [550., 509.]]
PROTOCOL = "b" * 64


def meta(available=True, side="left", start=0):
    return dict(side=side, start_index=start, group_id=f"{side}_{start:06d}", available=available,
                seeds_native=SEEDS if available else [], box=BOX,
                reason="ok" if available else "no_automatic_seed")


def data(start=0, count=21):
    return {"frames256": np.zeros((count, 256, 256, 3), np.uint8),
            "gray_native": np.zeros((count, 1024, 1024), np.uint8),
            "times": np.arange(count) / 10., "source_frames": np.arange(start, start + count),
            "pixel_hashes": np.asarray([str(index) for index in range(count)])}


def fake_lk(frames, times, sources, hashes, seeds, box, mode):
    return [{"time_s": float(t), "t_emit": float(t), "max_source_timestamp": float(t),
             "source_frame": int(source), "age_s": float(t - times[0]),
             "points_xy_native": copy.deepcopy(seeds), "valid_candidate": [True] * len(seeds),
             "rejection_reason": None, "status": "candidate", "material_identity_confirmed": False,
             "processing_ms": 1., "waiting_future_frames": 0}
            for t, source in zip(times, sources)]


def test_seed_failure_keeps_all_frames_without_calling_tracker(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("failed seed must not enter tracker")
    monkeypatch.setattr(b, "track_lk", forbidden)
    result = b.replay_cohort(meta(False), data(), "lk256")
    assert len(result["rows"]) == 21
    assert result["summary"]["scheduled_continuation_frames"] == 20
    assert result["summary"]["initialized_points"] == 0
    assert all(row["status"] == "unknown" and not row["valid_candidate"] for row in result["rows"])
    assert json.dumps(result, allow_nan=False)


def test_fixed_frame_denominator_and_conditional_point_denominator_are_separate(monkeypatch):
    monkeypatch.setattr(b, "track_lk", fake_lk)
    good = b.replay_cohort(meta(True), data(), "lk256")
    failed = b.replay_cohort(meta(False), data(), "lk256")
    summary = b.summarize_method([good, failed])
    assert summary["scheduled_cohort_denominator"] == 2
    assert summary["scheduled_frames"] == 42
    assert summary["frame_candidate_fraction_fixed_denominator"] == .5
    assert summary["point_frame_opportunities_conditional_on_seed"] == 84
    assert summary["point_survival_fraction_conditional_on_seed"] == 1.
    assert summary["material_identity_accuracy"] is None


def test_deformation_diagnostics_remove_translation_scale_rotation_but_not_warp():
    seed = np.asarray(SEEDS)
    angle = .2
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    current = seed @ rotation.T * 1.2 + [40., 70.]
    rigid = b.point_deformation(seed, current, [True] * 4)
    assert rigid["similarity_residual_rms_native_px"] < 1e-8
    assert rigid["median_pair_distance_ratio"] == pytest.approx(1.2)
    current[1, 1] += 25.
    deformed = b.point_deformation(seed, current, [True] * 4)
    assert deformed["similarity_residual_rms_native_px"] > 1.
    assert deformed["pair_distance_ratio_mad"] > 0.
    assert b.point_deformation(seed, current, [True, False, False, False])["status"] == "insufficient_current_points"


def test_overlay_uses_pixel_center_native_mapping_and_never_stale_points():
    native = b.map_points([[0., 0.], [255., 255.], [100., 100.]], BOX, 256, to_native=True)
    row = {"points_xy_native": native.tolist(), "valid_candidate": [True, True, False]}
    assert b.drawable_points(row, BOX) == [(0, 0., 0.), (1, 255., 255.)]
    image = b.overlay_current(np.zeros((256, 256, 3), np.uint8), row, BOX, available=True)
    assert image.size == (256, 256)
    assert np.asarray(image)[95:106, 95:106].sum() == 0  # no stale marker or trail


class FakeTap:
    def __init__(self):
        self.resets = 0
        self.calls = []

    def reset(self):
        self.resets += 1

    def start(self, frame, points, timestamp, group_id, source):
        self.seed = np.asarray(points)
        self.index = 0
        return self.update(frame, timestamp, group_id, source)

    def update(self, frame, timestamp, group_id, source):
        self.calls.append((group_id, source))
        points = self.seed.copy()
        visible = np.ones(len(points), bool)
        if self.index == 1:
            points[0] = np.nan
            visible[0] = False
        self.index += 1
        return dict(points_xy=points.tolist(), visibility_candidate=visible.tolist(),
                    visible_probability=[.9] * len(points), status=["candidate"] * len(points),
                    inference_ms=1., max_source_timestamp=timestamp)


def test_optional_adapter_sanitizes_nan_and_never_reacquires_lost_identity():
    adapter = FakeTap()
    result = b.replay_cohort(meta(), data(count=3), "tapnext", adapter=adapter)
    assert result["rows"][1]["points_xy_native"][0] == [None, None]
    assert result["rows"][2]["valid_candidate"][0] is False
    assert result["rows"][2]["model_visibility_candidate"][0] is True
    assert json.dumps(result, allow_nan=False)


def test_optional_adapter_duplicate_hash_or_gap_stops_group(monkeypatch):
    inputs = data(count=3)
    inputs["pixel_hashes"][1] = inputs["pixel_hashes"][0]
    adapter = FakeTap()
    result = b.replay_cohort(meta(), inputs, "tapnext", adapter=adapter)
    assert len(adapter.calls) == 1
    assert not any(result["rows"][1]["valid_candidate"])
    assert not any(result["rows"][2]["valid_candidate"])


def test_gt_fields_do_not_enter_tracking_or_deformation(monkeypatch):
    monkeypatch.setattr(b, "track_lk", fake_lk)
    first = b.replay_cohort(meta(), data(), "lk256")
    modified = dict(meta(), confirmed_t1=27.3, label_path="never_read", future_ground_truth=[[1., 2.]])
    second = b.replay_cohort(modified, data(), "lk256")
    assert first["rows"] == second["rows"]


def test_missing_optional_weights_fail_before_output_or_torch_load(tmp_path):
    out = tmp_path / "output"
    with pytest.raises(FileNotFoundError, match="weights unavailable"):
        b.run(tmp_path / "input", out, PROTOCOL, ["tapnext"], weights_path=tmp_path / "missing.pt")
    assert not out.exists()


def make_fixed_inputs(base):
    # One compressed synthetic archive, hard-linked per group. No real pixels.
    template = base / "synthetic_template.npz"
    base.mkdir()
    contents = data()
    with template.open("xb") as handle:
        np.savez_compressed(handle, **contents)
    hashes = {}
    for side in b.CONFIG["sides"]:
        for start in b.CONFIG["cohort_start_indices"]:
            folder = base / "cohorts" / f"{side}_{start:06d}"
            folder.mkdir(parents=True)
            record = meta(start == 0, side, start)
            (folder / "meta.json").write_text(json.dumps(record), encoding="utf-8")
            # The loader requires each cohort's actual video indices. Tests
            # patch the loader's array return, not production validation.
            os.link(template, folder / "data.npz")
            for name in ("meta.json", "data.npz"):
                path = folder / name
                hashes[path.relative_to(base).as_posix()] = b.sha256(path)
    return hashes


def test_fixed60_integration_outputs_all_groups_hashes_and_no_overwrite(tmp_path, monkeypatch):
    base, out = tmp_path / "input", tmp_path / "output"
    hashes = make_fixed_inputs(base)
    monkeypatch.setattr(b, "track_lk", fake_lk)
    monkeypatch.setattr(b, "_load_data", lambda path, start: data(start))
    report = b.run(base, out, PROTOCOL, expected_input_sha256=hashes)
    for method in ("lk256", "lk_native"):
        summary = report["methods"][method]
        assert summary["scheduled_cohort_denominator"] == 60
        assert summary["scheduled_frames"] == 1260
        assert summary["scheduled_continuation_frames"] == 1200
        assert summary["seed_failure_cohorts"] == 58
        assert len(list((out / method).glob("*.json"))) == 60
    assert len(list((out / "overlays").glob("*.png"))) == 26
    with Image.open(out / "overlays" / "left_fixed_last_frames.png") as image:
        assert image.size == (1720, 775)
    with pytest.raises(FileExistsError):
        b.run(base, out, PROTOCOL)


def test_missing_cohort_or_hash_mismatch_cannot_reduce_denominator(tmp_path):
    base = tmp_path / "input"
    hashes = make_fixed_inputs(base)
    first = next(iter(hashes))
    with pytest.raises(ValueError, match="hash mismatch"):
        b.run(base, tmp_path / "output", PROTOCOL, expected_input_sha256=dict(hashes, **{first: "0" * 64}))
    assert not (tmp_path / "output").exists()
    with pytest.raises(FileNotFoundError):
        b.run(tmp_path / "missing", tmp_path / "output2", PROTOCOL)


def test_array_loader_preserves_21_even_if_last_age_exceeds_limit(tmp_path):
    values = data(start=100)
    values["times"][-1] = 2.0001
    path = tmp_path / "data.npz"
    with path.open("xb") as handle:
        np.savez_compressed(handle, **values)
    loaded = b._load_data(path, 100)
    assert len(loaded["times"]) == 21
    assert loaded["times"][-1] == 2.0001
    with pytest.raises(ValueError, match="frame identities"):
        b._load_data(path, 0)
