"""Synthetic-only coordinate/LK contract audit; never reads footage or labels."""
import copy
import hashlib
import inspect
import json

import cv2
import numpy as np
import pytest

from vision_geometry_lab.experiments.v26_causal_material.common import (
    CONFIG, crop_box, map_points, stream_guard,
)
from vision_geometry_lab.experiments.v26_causal_material.point_tracks import seed_points, track_lk


BOX = [100, 200, 1024, 1024]
MODEL_POINTS = np.array([[48., 64.], [100., 80.], [176., 164.], [160., 96.]], dtype=np.float32)
SEEDS = map_points(MODEL_POINTS, BOX, 256, to_native=True)


def frames(count=3, size=256):
    return [np.full((size, size), i, dtype=np.uint8) for i in range(count)]


def fake_identity(monkeypatch):
    calls = []
    def flow(before, after, points, ignored, **kwargs):
        calls.append((before, after, points.copy(), kwargs))
        return points.copy(), np.ones((len(points), 1), np.uint8), np.zeros((len(points), 1), np.float32)
    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", flow)
    return calls


def stripped(rows):
    return [{key: value for key, value in row.items() if key != "processing_ms"} for row in rows]


def test_pixel_center_map_exact_anchors_and_roundtrip():
    # A resize from 1024 to 256 maps native centers origin+1.5 to output 0,
    # NOT origin to output 0. Native integer 0 maps to -.375, not a corner.
    np.testing.assert_allclose(map_points([[101.5, 201.5]], BOX, 256), [[0., 0.]])
    np.testing.assert_allclose(map_points([[100., 200.]], BOX, 256), [[-.375, -.375]])
    np.testing.assert_allclose(map_points([[255., 255.]], BOX, 256, to_native=True), [[1121.5, 1221.5]])
    rng = np.random.default_rng(CONFIG["seed"])
    points = rng.uniform([-50., -60.], [4300., 3200.], (100, 2))
    for size in (256, 1024):
        mapped = map_points(points, BOX, size)
        np.testing.assert_allclose(map_points(mapped, BOX, size, to_native=True), points, atol=1e-12)


def test_coordinate_mapping_matches_opencv_half_pixel_linear_resize():
    source = np.arange(1024, dtype=np.float32)[None, :].repeat(1024, axis=0)
    resized = cv2.resize(source, (256, 256), interpolation=cv2.INTER_LINEAR)
    expected = map_points([[0., 0.], [255., 255.]], [0, 0, 1024, 1024], 256, to_native=True)
    assert resized[0, 0] == pytest.approx(expected[0, 0])
    assert resized[-1, -1] == pytest.approx(expected[-1, 0])


def test_crop_box_clips_at_image_edges_without_changing_size():
    shape = (3000, 4096, 3)
    for seed in ([[10., 20.], [80., 20.]], [[4000., 2980.], [4090., 2980.]],
                 [[40., 2900.], [40., 2990.]], [[3000., 10.], [3000., 0.]]):
        x, y, width, height = crop_box(shape, seed)
        assert width == height == 1024
        assert 0 <= x <= shape[1] - width and 0 <= y <= shape[0] - height
    with pytest.raises(ValueError, match="too small"):
        crop_box((512, 512), [[0., 0.], [1., 0.]])


@pytest.mark.parametrize("seed", [[], [[1., 1.]], [[1., 1.], [1., 1.]],
                                 [[0., 0.], [np.nan, 1.]]])
def test_degenerate_crop_seed_is_explicitly_rejected(seed):
    with pytest.raises(ValueError):
        crop_box((3000, 4096), seed)


def test_fixed_automatic_texture_seeds_only_inside_supported_past_span():
    rng = np.random.default_rng(CONFIG["seed"])
    frame = rng.integers(16, 240, (1024, 1024, 3), dtype=np.uint8)
    seed = [[200., 512.], [800., 512.]]
    first, diagnostic = seed_points(frame, seed, 40., [0, 0, 1024, 1024])
    second, diagnostic2 = seed_points(frame, seed, 40., [0, 0, 1024, 1024])
    assert first == second and diagnostic == diagnostic2
    assert CONFIG["min_points"] <= len(first) <= CONFIG["max_points"]
    xy = np.asarray(first)
    assert np.all((xy[:, 0] >= 320.) & (xy[:, 0] <= 760.))
    assert np.all(np.abs(xy[:, 1] - 512.) <= 10.)
    assert diagnostic["material_identity_confirmed"] is False
    assert "label" not in inspect.signature(seed_points).parameters
    assert "ground_truth" not in inspect.signature(track_lk).parameters


def test_saturated_or_flat_texture_is_not_replaced_with_invented_points():
    for value in (128, 255):
        frame = np.full((1024, 1024, 3), value, np.uint8)
        seeds, diagnostic = seed_points(frame, [[200., 512.], [800., 512.]], 40., [0, 0, 1024, 1024])
        assert seeds == []
        assert diagnostic["reason"] == "insufficient_unsaturated_texture"


@pytest.mark.parametrize("mode,size", [("lk256", 256), ("lk_native", 1024)])
def test_forward_backward_error_is_in_native_pixels_and_thresholded(monkeypatch, mode, size):
    calls = 0
    def flow(before, after, points, ignored, **kwargs):
        nonlocal calls
        calls += 1
        out = points.copy()
        if calls % 2 == 0:
            out[:, 0, 0] += np.array([3.6, 4.4, 0., 0.]) * size / CONFIG["crop_size"]
        return out, np.ones((len(points), 1), np.uint8), np.zeros((len(points), 1), np.float32)
    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", flow)
    result = track_lk(frames(2, size), [0., .1], [0, 1], ["a", "b"], SEEDS, BOX, mode)
    assert result[1]["fb_error_native_px"][:2] == pytest.approx([3.6, 4.4], abs=1e-4)
    assert result[1]["valid_candidate"] == [True, False, True, True]


def test_both_forward_and_backward_status_and_gray_error_are_required(monkeypatch):
    calls = 0
    def flow(before, after, points, ignored, **kwargs):
        nonlocal calls
        calls += 1
        status = np.ones((len(points), 1), np.uint8)
        error = np.zeros((len(points), 1), np.float32)
        if calls == 1:
            status[0] = 0
            error[2] = CONFIG["lk_max_gray_error"] + 1
        else:
            status[1] = 0
        return points.copy(), status, error
    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", flow)
    result = track_lk(frames(2), [0., .1], [0, 1], ["a", "b"], SEEDS, BOX, "lk256")
    assert result[1]["valid_candidate"] == [False, False, False, True]


def test_forward_outside_crop_is_rejected_not_clipped_onto_border(monkeypatch):
    calls = 0
    def flow(before, after, points, ignored, **kwargs):
        nonlocal calls
        calls += 1
        q = points.copy()
        if calls == 1:
            q[0, 0] = [-.01, 50.]
            q[1, 0] = [255.01, 50.]
        else:
            q[:] = MODEL_POINTS.reshape(-1, 1, 2)
        return q, np.ones((len(q), 1), np.uint8), np.zeros((len(q), 1), np.float32)
    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", flow)
    result = track_lk(frames(2), [0., .1], [0, 1], ["a", "b"], SEEDS, BOX, "lk256")
    assert result[1]["valid_candidate"][:2] == [False, False]


@pytest.mark.parametrize("failure", ["duplicate_time", "duplicate_source", "duplicate_pixels", "gap", "age"])
def test_failure_clears_cohort_and_is_absorbing(monkeypatch, failure):
    calls = fake_identity(monkeypatch)
    times, sources, hashes = [0., .1, .2], [0, 1, 2], ["a", "b", "c"]
    if failure == "duplicate_time":
        times = [0., 0., .2]
    elif failure == "duplicate_source":
        sources = [0, 0, 2]
    elif failure == "duplicate_pixels":
        hashes = ["a", "a", "c"]
    elif failure == "gap":
        times = [0., .251, .3]
    elif failure == "age":
        times = [0., 2.001, 2.1]
    result = track_lk(frames(), times, sources, hashes, SEEDS, BOX, "lk256")
    assert all(result[0]["valid_candidate"])
    assert not any(result[1]["valid_candidate"])
    assert not any(result[2]["valid_candidate"])
    assert result[1]["rejection_reason"] is not None
    assert calls == []


def test_max_age_exact_boundary_survives_and_next_fresh_frame_expires(monkeypatch):
    fake_identity(monkeypatch)
    times = list(np.arange(21) / 10) + [2.0001]
    source = list(range(len(times)))
    result = track_lk(frames(len(times)), times, source, [str(x) for x in source], SEEDS, BOX, "lk256")
    assert all(result[20]["valid_candidate"])
    assert not any(result[21]["valid_candidate"])
    assert result[21]["rejection_reason"] == "cohort_age_limit"


def test_actual_lk_translation_uses_causal_adjacent_frames_and_native_units():
    rng = np.random.default_rng(2610)
    first = cv2.GaussianBlur(rng.integers(0, 256, (256, 256), dtype=np.uint8), (5, 5), 0)
    matrix = np.array([[1., 0., 1.25], [0., 1., .75]], dtype=np.float32)
    second = cv2.warpAffine(first, matrix, (256, 256), flags=cv2.INTER_LINEAR)
    hashes = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in [first, second]]
    result = track_lk([first, second], [0., .1], [0, 1], hashes, SEEDS, BOX, "lk256")
    assert all(result[-1]["valid_candidate"])
    displacement = np.asarray(result[-1]["points_xy_native"]) - SEEDS
    np.testing.assert_allclose(displacement, np.tile([5., 3.], (4, 1)), atol=.8)
    assert result[-1]["max_source_timestamp"] == result[-1]["t_emit"] == .1
    assert result[-1]["waiting_future_frames"] == 0


def test_actual_lk_future_pixels_cannot_change_earlier_predictions():
    rng = np.random.default_rng(2611)
    first = cv2.GaussianBlur(rng.integers(0, 256, (256, 256), dtype=np.uint8), (5, 5), 0)
    sequence = [cv2.warpAffine(first, np.array([[1., 0., i], [0., 1., i / 2]], np.float32),
                               (256, 256)) for i in range(4)]
    changed = sequence[:3] + [rng.integers(0, 256, (256, 256), dtype=np.uint8)]
    def run_sequence(seq):
        return track_lk(seq, [i * .1 for i in range(len(seq))], list(range(len(seq))),
                        [hashlib.sha256(frame.tobytes()).hexdigest() for frame in seq],
                        SEEDS, BOX, "lk256")
    assert stripped(run_sequence(sequence)[:3]) == stripped(run_sequence(changed)[:3])
    assert stripped(run_sequence(sequence)[:3]) == stripped(run_sequence(sequence[:3]))


def test_future_replacement_and_separate_cohorts_do_not_rewrite_prefix(monkeypatch):
    fake_identity(monkeypatch)
    original = frames(4)
    changed = copy.deepcopy(original)
    changed[3][:] = 255
    a = track_lk(original, [0., .1, .2, .3], [0, 1, 2, 3], ["a", "b", "c", "d"], SEEDS, BOX, "lk256")
    b = track_lk(changed, [0., .1, .2, .3], [0, 1, 2, 3], ["a", "b", "c", "changed"], SEEDS, BOX, "lk256")
    prefix = track_lk(original[:3], [0., .1, .2], [0, 1, 2], ["a", "b", "c"], SEEDS, BOX, "lk256")
    assert stripped(a[:3]) == stripped(b[:3]) == stripped(prefix)
    killed = track_lk(original[:2], [0., .3], [0, 1], ["a", "b"], SEEDS, BOX, "lk256")
    assert not any(killed[-1]["valid_candidate"])
    fresh = track_lk(original[:2], [0., .1], [0, 1], ["a", "b"], SEEDS, BOX, "lk256")
    assert all(fresh[-1]["valid_candidate"])


@pytest.mark.parametrize("field", ["times", "source_frames", "pixel_hashes"])
def test_mismatched_sequence_lengths_are_not_silently_zipped(field):
    args = {"times": [0., .1, .2], "source_frames": [0, 1, 2], "pixel_hashes": ["a", "b", "c"]}
    args[field] = args[field][:-1]
    with pytest.raises(ValueError):
        track_lk(frames(), seeds_native=SEEDS, box=BOX, mode="lk256", **args)


def test_lk_nan_output_is_invalid_and_json_safe(monkeypatch):
    def flow(before, after, points, ignored, **kwargs):
        q = points.copy()
        q[0, 0, 0] = np.nan
        return q, np.ones((len(q), 1), np.uint8), np.zeros((len(q), 1), np.float32)
    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", flow)
    result = track_lk(frames(2), [0., .1], [0, 1], ["a", "b"], SEEDS, BOX, "lk256")
    assert not result[1]["valid_candidate"][0]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("invalid", ["seed", "timestamp"])
def test_initial_nan_seed_and_nonfinite_timestamp_are_explicitly_rejected(invalid):
    bad = SEEDS.copy()
    if invalid == "seed":
        bad[0, 0] = np.nan
    timestamp = np.nan if invalid == "timestamp" else 0.
    with pytest.raises(ValueError):
        track_lk(frames(1), [timestamp], [0], ["a"], bad, BOX, "lk256")


def test_nonfinite_guard_is_explicit_and_pixel_repeat_is_not_motion_proof():
    assert stream_guard(None, np.nan, 0, "x") == "nonfinite_timestamp"
    previous = dict(time_s=0., source_frame=0, pixel_hash="x")
    assert stream_guard(previous, .1, 1, "x") == "duplicate_decoded_pixels"
    assert stream_guard(previous, .1, 1, "y") is None
    assert stream_guard(previous, .25, 1, "y") is None
    assert stream_guard(previous, .250001, 1, "y") == "frame_gap_exceeded"
