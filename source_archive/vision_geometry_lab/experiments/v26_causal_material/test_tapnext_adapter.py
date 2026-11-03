"""Adapter contracts without optional weights/torch; model smoke is separate."""
import importlib
import numpy as np
import pytest

from vision_geometry_lab.experiments.v26_causal_material.tapnext_adapter import (
    TapNextAdapter, make_queries_xy, from_official_yx, validate_frame,
)


def test_pixel_centres_not_corners_and_yx_order():
    xy = np.array([[0, 255], [71.25, 120.75]], np.float32)
    q = make_queries_xy(xy)
    assert q[0, 0].tolist() == [0, 255.5, .5]
    np.testing.assert_array_equal(from_official_yx(q[0, :, 1:]), xy)


@pytest.mark.parametrize("xy", [[], [[-1, 5]], [[256, 9]], [[np.nan, 1]], [[1, 2, 3]]])
def test_invalid_queries(xy):
    with pytest.raises(ValueError):
        make_queries_xy(xy)


def test_shared_exact_input_contract():
    image = np.zeros((256, 256, 3), np.uint8)
    assert validate_frame(image) is image
    with pytest.raises(ValueError):
        validate_frame(image.astype(np.float32))
    with pytest.raises(ValueError):
        validate_frame(image[:255])


def mock_started():
    obj = TapNextAdapter.__new__(TapNextAdapter)
    obj.state = object()
    obj.stream_id = "left/group1"
    obj.last_timestamp = .1
    obj.seed_timestamp = 0.
    obj.last_frame_index = 1
    obj.occluded_ever = np.zeros(1, bool)
    obj.max_gap_s = .25
    obj.max_age_s = 2.
    obj._forward = lambda frame, timestamp, index, queries: {
        "timestamp": timestamp, "frame_index": index, "queries": queries}
    return obj


@pytest.mark.parametrize("timestamp,stream,index", [(.1, "left/group1", 2),
    (.05, "left/group1", 2), (.2, "right/group1", 2), (.2, "left/group1", 1),
    (np.nan, "left/group1", 2)])
def test_bad_source_rejected(timestamp, stream, index):
    with pytest.raises(ValueError):
        mock_started().update(np.zeros((256, 256, 3), np.uint8), timestamp, stream, index)


def test_long_missing_invalidates_state():
    obj = mock_started()
    with pytest.raises(ValueError, match="missing"):
        obj.update(np.zeros((256, 256, 3), np.uint8), .5, "left/group1", 5)
    assert obj.state is None and obj.stream_id is None


def test_old_material_age_invalidates_state():
    obj = mock_started()
    obj.last_timestamp = 2.
    with pytest.raises(ValueError, match="age"):
        obj.update(np.zeros((256, 256, 3), np.uint8), 2.1, "left/group1", 21)
    assert obj.state is None


def test_no_future_needed():
    obj = mock_started()
    row = obj.update(np.zeros((256, 256, 3), np.uint8), .2, "left/group1", 2)
    assert row == {"timestamp": .2, "frame_index": 2, "queries": None}


def test_weights_missing_explicit_not_random_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="weights missing"):
        TapNextAdapter(weights_path=tmp_path / "missing.pt")


def test_wrong_weights_hash_before_import(tmp_path):
    fake = tmp_path / "fake.pt"
    fake.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="SHA256"):
        TapNextAdapter(weights_path=fake, expected_sha256="0" * 64)


def test_import_does_not_import_hardware(monkeypatch):
    original = importlib.import_module
    attempted = []
    def guarded(name, *args, **kwargs):
        attempted.append(name)
        assert not any(word in name.lower() for word in ("mvimport", "serial", "modbus", "device_control"))
        return original(name, *args, **kwargs)
    monkeypatch.setattr(importlib, "import_module", guarded)
    module = original("vision_geometry_lab.experiments.v26_causal_material.tapnext_adapter")
    importlib.reload(module)
    assert not attempted
