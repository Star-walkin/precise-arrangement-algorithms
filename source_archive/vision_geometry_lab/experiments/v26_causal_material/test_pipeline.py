"""Synthetic measurement/import tests only; no source AVI, cache or labels read."""
import copy
import json
import math
import subprocess
import sys

import numpy as np
import pytest

from vision_geometry_lab.experiments.v26_causal_material import pipeline as p
from vision_geometry_lab.experiments.v26_causal_material.common import timestamps


def image(height=200, width=300, value=128):
    return np.full((height, width, 3), value, np.uint8)


def measurement(theta=.2, x_ref=120., *, height=200, scale=1.):
    normal = np.array([-math.sin(theta), math.cos(theta)])
    point = np.array([x_ref, p.CONFIG["reference_y_fraction"] * height])
    direction = np.array([math.cos(theta), math.sin(theta)])
    return {"rope": {"valid": True, "reason": "ok", "normal": normal.tolist(),
            "rho_px": float(normal @ point), "angle_deg": math.degrees(theta),
            "observed_segment": [(point - 80 * scale * direction).tolist(),
                                  (point + 80 * scale * direction).tolist()],
            "width_px": 20. * scale, "residual_px": .5, "observed_support": 50,
            "diagnostics": {"support_coverage": .9}}, "timing_ms": {"rope": 10.}}


def feature(meta=None, frame=None, *, side="left", index=20, ts=None, pixel_hash="new", previous_hash="old"):
    return p.feature_row(side, index, ts or {"host_time_s": 1234.567, "source_frame": 789},
                         meta or measurement(), {"features": {"usable": True, "image_gap_proxy": 1.25, "reason": "ok"}},
                         frame if frame is not None else image(), pixel_hash, previous_hash)


def without_runtime(row):
    return {key: value for key, value in row.items() if key != "quality_and_feature_ms"}


def test_line_orientation_and_position_are_two_degrees_of_freedom_not_one_scalar():
    original = feature(measurement(theta=.2, x_ref=120.))
    translated = feature(measurement(theta=.2, x_ref=140.))
    rotated = feature(measurement(theta=.35, x_ref=120.))
    assert original["available"] and translated["available"] and rotated["available"]
    assert original["angle_proxy_rad"] == pytest.approx(translated["angle_proxy_rad"])
    assert translated["x_ref_px"] - original["x_ref_px"] == pytest.approx(20.)
    assert rotated["x_ref_px"] == pytest.approx(original["x_ref_px"])
    assert rotated["angle_proxy_rad"] - original["angle_proxy_rad"] == pytest.approx(.15)


def test_scanline_uses_full_original_height_and_preserves_native_units_no_mirror():
    small = feature(measurement(), image())
    large = feature(measurement(height=400, x_ref=240., scale=2.), image(400, 600))
    right = feature(measurement(), image(), side="right")
    assert small["x_ref_px"] == pytest.approx(120.)
    assert large["x_ref_px"] == pytest.approx(240.)
    assert large["angle_proxy_rad"] == pytest.approx(small["angle_proxy_rad"])
    assert right["x_ref_px"] == pytest.approx(small["x_ref_px"])
    assert right["angle_proxy_rad"] == pytest.approx(small["angle_proxy_rad"])
    assert large["free_span_length_px"] == pytest.approx(2 * small["free_span_length_px"])


def test_unoriented_line_sign_and_pi_are_equivalent():
    a = feature(measurement(theta=.2))
    b = feature(measurement(theta=.2 + math.pi))
    assert a["angle_proxy_rad"] == pytest.approx(b["angle_proxy_rad"])
    assert a["x_ref_px"] == pytest.approx(b["x_ref_px"])


def test_receive_time_is_preserved_not_fps_or_sdk_counter_or_device_ticks():
    timestamp = {"host_time_s": 98765.4321, "source_frame": 87654, "device_ticks": 9999999999}
    row = feature(ts=timestamp, index=20)
    assert row["time_s"] == row["max_source_timestamp"] == row["available_at_s"] == 98765.4321
    assert row["nominal_time_s"] == 2.
    assert row["source_frame"] == 20 and row["sdk_source_frame"] == 87654
    assert row["timestamp_basis"] == "host_monotonic_sdk_receive_not_exposure"
    assert row["emit_clock"] == "causal_simulation_not_live_completion"
    assert row["motion_status"] == "unknown" and row["motion_source"] == "unavailable"


def test_saturation_current_quality_failure_and_duplicate_pixels_reject_measurements():
    saturated = feature(frame=image(value=255))
    assert not saturated["available"] and saturated["reason"] == "free_span_saturated"
    assert saturated["angle_proxy_rad"] is None and saturated["x_ref_px"] is None
    failed = measurement()
    failed["rope"].update(valid=False, observed_segment=None, width_px=None)
    missing = feature(failed)
    assert not missing["available"] and missing["reason"] == "missing_current_quality"
    duplicate = feature(pixel_hash="same", previous_hash="same")
    assert not duplicate["available"] and duplicate["reason"] == "duplicate_decoded_pixels"
    assert not duplicate["old_gap_usable"]


def test_quality_threshold_equality_is_not_tuned_after_results(monkeypatch):
    monkeypatch.setattr(p, "optical_quality", lambda frame, rope: (p.CONFIG["signal_max_saturation"], 10.))
    assert feature()["available"]
    monkeypatch.setattr(p, "optical_quality", lambda frame, rope: (p.CONFIG["signal_max_saturation"] + 1e-6, 10.))
    assert not feature()["available"]


def test_nearly_parallel_scanline_and_malformed_line_are_explicitly_rejected():
    parallel = feature(measurement(theta=.01))
    assert not parallel["available"]
    assert parallel["reason"] == "near_parallel_reference_scanline"
    malformed = measurement()
    malformed["rope"]["normal"] = None
    output = feature(malformed)
    assert not output["available"] and output["reason"] == "malformed_line"


def test_quality_reads_only_observed_free_span_not_full_image_or_future_pixels():
    first = image()
    second = first.copy()
    second[:50, :] = 255  # well outside the observed span near y=140
    a, b = feature(frame=first), feature(frame=second)
    assert without_runtime(a) == without_runtime(b)
    assert a["inlier_ratio"] is None
    assert a["observed_support"] == 50
    assert a["fit_support_coverage"] == .9


def test_unrelated_annotations_cannot_change_measurement_or_confirm_t1():
    original = measurement()
    modified = copy.deepcopy(original)
    modified.update(label_path="must_not_open", confirmed_t1=27.3, true_contact=[100, 100])
    modified["rope"]["manual_layer_label"] = "confirmed"
    assert without_runtime(feature(original)) == without_runtime(feature(modified))
    assert not feature(modified)["physical_t1_confirmed"]


def test_outside_quality_support_is_unknown_not_black_padding_observation():
    meta = measurement()
    meta["rope"]["observed_segment"] = [[-200., -150.], [-40., -130.]]
    saturation, contrast = p.optical_quality(image(), meta["rope"])
    assert saturation is None and contrast is None
    result = feature(meta)
    assert not result["available"]


@pytest.mark.parametrize("field", ["width", "segment"])
def test_nonfinite_optical_geometry_does_not_make_fresh_quality(field):
    meta = measurement()
    if field == "width":
        meta["rope"]["width_px"] = math.nan
    else:
        meta["rope"]["observed_segment"][0][0] = math.nan
    saturation, contrast = p.optical_quality(image(), meta["rope"])
    assert saturation is None and contrast is None
    result = feature(meta)
    assert not result["available"]
    json.dumps(result, allow_nan=False)


def test_nonfinite_optional_fit_diagnostics_are_json_safe_unknown_not_fake_zero():
    meta = measurement()
    meta["rope"]["residual_px"] = math.nan
    meta["rope"]["diagnostics"]["support_coverage"] = math.inf
    result = feature(meta)
    json.dumps(result, allow_nan=False)
    assert result["fit_residual_px"] is None
    assert result["fit_support_coverage"] is None


def test_array_hash_binds_pixels_shape_and_dtype_and_handles_noncontiguous_array():
    source = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    assert p.array_hash(source) == p.array_hash(source.copy())
    assert p.array_hash(source) != p.array_hash(source.reshape(4, 2, 3))
    assert p.array_hash(source) != p.array_hash(source.astype(np.int16))
    view = source[:, ::-1]
    assert p.array_hash(view) == p.array_hash(np.ascontiguousarray(view))


def test_synthetic_csv_preserves_receive_jitter_and_rejects_duplicate_source(tmp_path):
    path = tmp_path / "timestamps.csv"
    path.write_text("video_frame,source_frame,host_time_s,device_ticks\n0,100,12.0,1000\n1,101,12.103,2000\n2,102,12.199,3000\n", encoding="utf-8")
    rows = timestamps(path)
    assert [row["host_time_s"] for row in rows] == [12., 12.103, 12.199]
    assert [row["source_frame"] for row in rows] == [100, 101, 102]
    path.write_text("video_frame,source_frame,host_time_s,device_ticks\n0,100,12.0,1000\n1,100,12.103,2000\n", encoding="utf-8")
    with pytest.raises(ValueError, match="repeated source"):
        timestamps(path)


def test_fresh_pipeline_import_cannot_open_video_hardware_network_or_device_library():
    code = r'''
import builtins, ctypes, socket
import cv2, numpy, PIL.Image
original_import = builtins.__import__
forbidden_names = {'serial','pymodbus','snap7','pyads','pycomm3','gxipy','pypylon',
                   'MvCameraControl_class','MvCamera','camera','camera_control',
                   'motion_control','plc','lighting','servo'}
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if any(part in forbidden_names for part in name.split('.')):
        raise AssertionError('hardware import: '+name)
    return original_import(name, globals, locals, fromlist, level)
def forbidden(*args, **kwargs):
    raise AssertionError('import attempted live IO/device initialization')
builtins.__import__ = guarded_import
cv2.VideoCapture = forbidden
socket.create_connection = forbidden
socket.socket = forbidden
ctypes.CDLL = forbidden
if hasattr(ctypes, 'WinDLL'): ctypes.WinDLL = forbidden
class NoLibraries:
    def __getattr__(self, name):
        raise AssertionError('import attempted dynamic library load: '+name)
ctypes.cdll = NoLibraries()
if hasattr(ctypes, 'windll'): ctypes.windll = NoLibraries()
from vision_geometry_lab.experiments.v26_causal_material import pipeline
assert pipeline.CONFIG['hardware_access'] is False
assert pipeline.GATES['lack_independent_labels'] == 'not_evaluable_not_pass'
print('safe_import')
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=p.ROOT,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "safe_import"
