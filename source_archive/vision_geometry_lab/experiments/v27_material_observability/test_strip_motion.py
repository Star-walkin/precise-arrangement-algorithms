"""Synthetic contracts only. No videos, labels, hardware, or physical acceptance."""
import inspect
import json
import time

import numpy as np
import pytest

import strip_motion as sm


def translated_random(shift=37, rows=5, length=401):
    rng = np.random.default_rng(891)
    a = rng.uniform(40, 200, (rows, length))
    b = np.zeros_like(a); ma = np.ones_like(a, bool); mb = np.zeros_like(ma)
    b[:, shift:] = a[:, :-shift]; mb[:, shift:] = True
    return a, b, ma, mb


def test_masked_zncc_exact_overlap_sign_and_no_circular_wrap():
    a, b, ma, mb = translated_random()
    c = sm.masked_zncc(a, b, ma, mb)
    index = np.nanargmax(c["scores"])
    assert c["lags"][index] == 37
    assert c["scores"][index] == pytest.approx(1.0)
    assert c["overlap_samples"][index] == 5 * (401 - 37)
    assert c["overlap_fraction"][index] == pytest.approx(1.0)
    assert sm._nfft(401) >= 2 * 401 - 1


def test_masked_zncc_matches_direct_nonuniform_mask_and_offset():
    a, b, ma, mb = translated_random(23)
    ma[:, 30:55] = False; mb[1:3, 190:217] = False
    b = 1.7 * b + 60
    lags = [-40, -3, 0, 23, 100]
    c = sm.masked_zncc(a, b, ma, mb, lags)
    for k, lag in enumerate(lags):
        aa, bb = (a[:, :401-lag], b[:, lag:]) if lag >= 0 else (a[:, -lag:], b[:, :401+lag])
        m = (ma[:, :401-lag] & mb[:, lag:]) if lag >= 0 else (ma[:, -lag:] & mb[:, :401+lag])
        expected = np.corrcoef(aa[m], bb[m])[0, 1]
        assert c["scores"][k] == pytest.approx(expected, abs=1e-10)


def test_all_invalid_and_flat_mask_borders_do_not_create_zncc_match():
    a = np.ones((5, 401)) * 100
    ma = np.zeros_like(a, bool); ma[:, 60:320] = True
    report, costs = sm.compare_strips(a, a, ma, ma, .1)
    assert report["status"] == "unavailable"
    assert report["candidate_displacements_px"] == []
    assert np.isnan(costs["zncc"]).all()
    assert "mask_edge_zero_bias_possible" in report["phase_warnings"]


def test_periodic_translation_preserves_all_integer_branches():
    a, ma, _ = sm.render_helical_cylinder()
    b, mb, _ = sm.render_helical_cylinder(translation_px=77)
    report, costs = sm.compare_strips(a, b, ma, mb, .1)
    assert report["status"] == "phase_ambiguous"
    assert {5., 29., 53., 77., 101.}.issubset(set(report["candidate_displacements_px"]))
    assert report["selected_displacement_px"] is None
    assert report["metric_velocity_m_s"] is None
    assert report["P_texture_px"] == 24
    assert report["correlation_is_probability"] is False
    assert costs["zncc"].shape == (257,)


def test_pure_spin_no_feed_indistinguishable_from_translation():
    theta = .8; shift = 24 * 3 * theta / (2*np.pi)
    a, ma, _ = sm.render_helical_cylinder()
    translation, mt, truth_t = sm.render_helical_cylinder(translation_px=shift)
    spin, mr, truth_r = sm.render_helical_cylinder(rotation_rad=theta)
    assert truth_r["translation_px"] == 0
    assert np.allclose(translation, spin, atol=1e-5)
    report_t, cost_t = sm.compare_strips(a, translation, ma, mt, .1)
    report_r, cost_r = sm.compare_strips(a, spin, ma, mr, .1)
    assert np.allclose(cost_t["zncc"], cost_r["zncc"], atol=1e-6, equal_nan=True)
    assert report_r["candidate_displacements_px"] == report_t["candidate_displacements_px"]
    assert "translation_spin_phase_coupling_unresolved" in report_r["ambiguity_reason"]
    assert report_r["metric_velocity_m_s"] is None


def test_mixed_rotation_and_feed_same_phase_not_unique_truth():
    mixed, _, _ = sm.render_helical_cylinder(translation_px=11, rotation_rad=.9)
    pure, _, _ = sm.render_helical_cylinder(translation_px=11+24*3*.9/(2*np.pi))
    assert np.allclose(mixed, pure, atol=1e-5)


def test_defect_glare_occlusion_render_mechanism_and_diagnostics():
    a, ma, _ = sm.render_helical_cylinder(defect=True, glare=True, occlusion=True)
    b, mb, truth = sm.render_helical_cylinder(translation_px=29, defect=True, glare=True, occlusion=True)
    assert truth["defect_material_attached"] and truth["glare_camera_fixed"]
    assert not ma.all() and not mb.all()
    report, costs = sm.compare_strips(a, b, ma, mb, .1)
    assert 29 in report["candidate_displacements_px"]
    assert report["metric_velocity_m_s"] is None
    assert np.isfinite(costs["overlap"]).all()
    assert "fixed_hann_window_is_not_material_evidence" in report["phase_warnings"]


def test_fft_phase_positive_displacement_and_wide_search():
    a, b, ma, mb = translated_random(93, rows=5, length=701)
    p = sm.fft_phase_multi(a, b, ma, mb, np.arange(-128, 129))
    assert p["peaks"][0]["displacement_px"] == 93
    assert p["padding"] == "linear_at_least_2L_minus_1"


def test_static_mask_actually_creates_false_zero_on_moving_texture():
    x = np.arange(401)[None, :]
    a = np.repeat(120 + 40*np.cos(2*np.pi*x/24), 5, axis=0)
    b = np.repeat(120 + 40*np.cos(2*np.pi*(x-5)/24), 5, axis=0)
    mask = np.repeat(((x>80)&(x<170))|((x>250)&(x<330)), 5, axis=0)
    report, cost = sm.compare_strips(a,b,mask,mask,.1)
    assert report["phase_correlation_peaks"][0]["displacement_px"] == 0
    assert "mask_edge_zero_bias_possible" in report["phase_warnings"]
    assert 5 in report["candidate_displacements_px"]
    assert report["metric_velocity_m_s"] is None


def test_lateral_profile_without_axial_texture_is_not_motion_evidence():
    a = np.repeat(np.arange(10,210,20)[:,None], 401, axis=1)
    mask = np.ones_like(a, bool)
    cost = sm.masked_zncc(a,a,mask,mask)
    assert np.isnan(cost["scores"]).all()
    assert cost["axial_std_a"] == 0


def test_period_difference_is_only_arithmetic_no_branch_selected():
    result = sm.period_multiple_diagnostic(47, [24, 48])
    assert result["comparisons"][0]["nearest_integer_multiple"] == 2
    assert result["comparisons"][0]["residual_px"] == -1
    assert "correct_branch" in result["conclusion"]
    assert sm.period_multiple_diagnostic(None, [24])["available"] is False


def synthetic_cohort(count=4):
    rng = np.random.default_rng(911)
    plane = rng.integers(40, 210, (1024, 1024), dtype=np.uint8)
    frames = [np.roll(plane, k * 7, axis=1) for k in range(count)]
    meta = {"group_id": "synthetic_0", "side": "synthetic", "box": [0,0,1024,1024],
            "free_seed": [[80.,500.],[960.,500.]], "optical_width_px": 22.}
    data = {"gray_native": np.stack(frames), "times": np.arange(count)*.1,
            "source_frames": np.arange(count), "pixel_hashes": np.array([str(k) for k in range(count)])}
    geometry = [{"session":"synthetic", "camera":"synthetic", "source_frame": k, "time_s": k*.1, "available_at_s": k*.1,
        "max_source_timestamp": k*.1, "timestamp_kind": sm.TIMESTAMP_KIND,
        "available": True, "reason":"ok", "normal_xy":[0.,1.], "rho_px":500.,
        "segment_xy":[[80.,500.],[960.,500.]], "width_px":22.} for k in range(count)]
    return meta, data, geometry


def test_fixed_origin_does_not_follow_changing_visible_endpoints():
    meta, _, geometry = synthetic_cohort()
    initial = sm.initial_frame(meta)
    changed = {**geometry[0], "segment_xy": [[230.,500.],[800.,500.]]}
    first = sm.line_frame(geometry[0], initial["reference_xy"], initial["d_xy"])
    second = sm.line_frame(changed, initial["reference_xy"], initial["d_xy"])
    assert first["origin_xy"] == pytest.approx(second["origin_xy"])
    assert first["origin_xy"] == pytest.approx([511.5,500.])
    assert np.array_equal(sm.sampling_grid(first, [-10,0,10], [-1,0,1]),
                          sm.sampling_grid(second, [-10,0,10], [-1,0,1]))


def test_geometry_compensation_preserves_axial_shift_removes_independent_swing():
    meta, data, geometry = synthetic_cohort(count=2)
    data["gray_native"][1] = np.roll(np.roll(data["gray_native"][0], 7, axis=1), 3, axis=0)
    geometry[1]["rho_px"] = 503
    geometry[1]["segment_xy"] = [[80.,503.],[960.,503.]]
    report, arrays = sm.process_cohort(meta,data,geometry)
    raw, comp = [p["bands"][0] for p in report["pairs"]]
    assert 7 in comp["candidate_displacements_px"]
    assert max(comp["correlation_scores"]) > .99
    assert not raw["correlation_scores"] or max(raw["correlation_scores"]) < .5
    assert np.nanmedian(arrays["geometry_compensated_original_xy_maps"][1,...,1] -
                        arrays["geometry_compensated_original_xy_maps"][0,...,1]) == 3
    assert np.array_equal(arrays["raw_fixed_original_xy_maps"][0], arrays["raw_fixed_original_xy_maps"][1])


def test_process_keeps_2d_three_bands_maps_and_no_unique_velocity():
    meta, data, geometry = synthetic_cohort()
    report, arrays = sm.process_cohort(meta, data, geometry)
    assert report["pair_denominator_per_configuration"] == 5
    assert len(report["pairs"]) == 10
    assert len(report["frames"]) == 8
    assert arrays["raw_fixed_strips"].shape[0] == 4
    assert arrays["raw_fixed_original_xy_maps"].shape[-1] == 2
    assert arrays["raw_fixed_narrow_band_spacetime"].shape == (3,4,1025)
    for pair in report["pairs"]:
        assert len(pair["bands"]) == 4
        assert pair["metric_velocity_m_s"] is None
        assert pair["t_available"] is None
        assert pair["source_available_lower_bound_s"] == data["times"][pair["local_frame_pair"][1]]
        for band in pair["bands"]:
            assert band["selected_displacement_px"] is None
    first = report["pairs"][0]["bands"][0]
    assert 7 in first["candidate_displacements_px"]
    json.dumps(report, allow_nan=False)
    assert report["t_available_is_measured"] is False


def test_future_image_and_geometry_replacement_does_not_change_past():
    meta, data, geometry = synthetic_cohort()
    report, arrays = sm.process_cohort(meta, data, geometry)
    changed_data = {**data, "gray_native": data["gray_native"].copy()}
    changed_data["gray_native"][3] = 0
    changed_geometry = geometry[:3] + [{**geometry[3], "rho_px": 900.}]
    other, other_arrays = sm.process_cohort(meta, changed_data, changed_geometry)
    past = [p for p in report["pairs"] if p["local_frame_pair"][1] <= 2]
    other_past = [p for p in other["pairs"] if p["local_frame_pair"][1] <= 2]
    assert past == other_past
    assert np.array_equal(arrays["geometry_compensated_original_xy_maps"][:3],
                          other_arrays["geometry_compensated_original_xy_maps"][:3])


def test_geometry_age_and_invalid_no_fallback():
    meta, data, geometry = synthetic_cohort()
    # Last-arrived invalid observation must not fall back to an earlier valid pose.
    geometry[1]["available"] = False
    geometry[1]["reason"] = "synthetic_invalid"
    report, _ = sm.process_cohort(meta, data, geometry[:2])
    rows = [r for r in report["frames"] if r["configuration"] == "geometry_compensated"]
    assert rows[0]["geometry_available"]
    assert all(not r["geometry_available"] for r in rows[1:])
    assert all(r["valid_fraction"] == 0 for r in rows[1:])


def test_bilinear_saturated_pixel_cannot_become_valid_midgray_texture():
    image=np.full((10,10),120,np.uint8); image[4,4]=255; image[4,5]=30
    grid=np.array([[[4.5,4.0]]],np.float32)
    pose={"d_xy":[1,0],"n_xy":[0,1],"origin_xy":[5,4],
          "segment_xy":[[1,4],[8,4]],"width_px":4}
    strip,mask=sm.extract_strip(image,[0,0,10,10],grid,pose)
    assert 15 < strip[0,0] < 245
    assert not mask[0,0]


@pytest.mark.parametrize("kind", ["camera", "session", "clock"])
def test_wrong_stream_or_clock_rejected(kind):
    meta,data,geometry=synthetic_cohort()
    if kind=="camera": meta["side"]="wrong"
    if kind=="session": meta["session"]="wrong"
    if kind=="clock": data["timestamp_kind"]="exposure_clock_unverified"
    with pytest.raises(ValueError,match="mismatch"):
        sm.process_cohort(meta,data,geometry)


@pytest.mark.parametrize("kind", ["time", "gap", "source", "duplicate"])
def test_guards_do_not_invent_correspondence(kind):
    _, data, _ = synthetic_cohort()
    if kind == "time": data["times"][1] = 0
    if kind == "gap": data["times"][1] = .5
    if kind == "source": data["source_frames"][1] = 5
    if kind == "duplicate": data["pixel_hashes"][1] = data["pixel_hashes"][0]
    assert sm._pair_guard(data, 0, 1) is not None


def test_failed_group_stays_in_summary_denominator():
    meta, data, geometry = synthetic_cohort()
    meta["free_seed"] = []
    report, arrays = sm.process_cohort(meta, data, geometry)
    summary = sm.summarize_reports([report], expected_groups=60)
    assert report["status"] == "unavailable"
    assert summary["received_group_count"] == 1
    assert summary["missing_groups"] == 59
    assert summary["configurations"]["raw_fixed"]["uncomputed_pairs"] == 5
    assert summary["material_identity_accuracy"] is None


def test_artifacts_write_complete_costs_and_no_overwrite(tmp_path):
    meta, data, geometry = synthetic_cohort(count=2)
    report, arrays = sm.process_cohort(meta, data, geometry)
    out = tmp_path / "synthetic"
    sm.write_artifacts(out, report, arrays)
    assert (out / "raw_fixed_all_strips.png").is_file()
    assert (out / "geometry_compensated_spacetime.png").is_file()
    with np.load(out / "strip_maps_full_costs.npz") as saved:
        assert saved["raw_fixed_zncc_costs"].shape == (1,4,257)
        assert saved["raw_fixed_original_xy_maps"].shape[-1] == 2
    with pytest.raises(FileExistsError):
        sm.write_artifacts(out, report, arrays)


def test_synthetic_mechanism_suite_output(tmp_path):
    report = sm.run_synthetic_suite(tmp_path / "synthetic_mechanisms")
    assert len(report["cases"]) == 4
    assert report["nonidentifiability_demonstration"]["max_absolute_image_difference_gray"] < 1e-4
    assert report["static_mask_false_zero"]["diagnostic"]["phase_correlation_peaks"][0]["displacement_px"] == 0
    assert all(row["diagnostic"]["metric_velocity_m_s"] is None for row in report["cases"])


def test_no_annotation_or_hardware_input_import():
    assert list(inspect.signature(sm.process_cohort).parameters) == ["meta", "data", "geometry_records"]
    source = inspect.getsource(sm)
    for forbidden in ["import serial", "import socket", "MvCamera", "plc_control", "import tkinter"]:
        assert forbidden not in source
    assert len(sm.CONFIG["configurations"]) == 2
    assert sm.CONFIG["max_shift_px"] >= 128
