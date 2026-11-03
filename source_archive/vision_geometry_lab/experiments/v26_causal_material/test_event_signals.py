"""Synthetic only: no footage, annotations, model weights or hardware imports."""
import copy
import ast
import inspect
import json
import math

import numpy as np
import pytest

from vision_geometry_lab.experiments.v26_causal_material.event_signals import (
    CausalEventDetector, DEFAULT_CONFIG, StudentTBOCPD, run_stream, unwrap_mod_pi,
)


def synthetic(values, *, angles=None, dt=.1, stream="synthetic"):
    if angles is None:
        angles = np.full(len(values), .2)
    return [{"stream_id": stream, "source_frame": i, "time_s": round(i * dt, 9),
             "max_source_timestamp": round(i * dt, 9),
             "available_at_s": round(i * dt, 9), "available": value is not None,
             "reason": "synthetic" if value is not None else "synthetic_missing",
             "x_ref_px": value, "angle_proxy_rad": float(angles[i])}
            for i, value in enumerate(values)]


def synthetic_case_summary():
    """Reproducible synthetic diagnostics, including retained false alarms.

    Expected scenario semantics are for this test only. Inference receives
    measurements alone; a signal change is never labelled a physical t1.
    """
    rng = np.random.default_rng(2601)
    n = 300
    times = np.arange(n) * .1
    noise = rng.normal(0, .2, n)
    mean = 1000 + noise.copy()
    mean[150:] += 12
    variance = 1000 + noise.copy()
    variance[150:] = 1000 + rng.normal(0, 5, n - 150)
    outlier = np.full(n, 1000.)
    outlier[150] += 100
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = .95 * ar[i - 1] + rng.normal(0, .5)
    cases = {
        "constant_noise": 1000 + noise,
        "mean_step_at_15s": mean,
        "variance_step_at_15s": variance,
        "slow_linear_drift": 1000 + .2 * times + noise,
        "periodic_not_physical_t1": 1000 + 5 * np.sin(2 * math.pi * times / 3),
        "autocorrelated_noise": 1000 + ar,
        "isolated_outlier_at_15s": outlier,
        "long_missing_then_new_level": [1000.] * 100 + [None] * 30 + [1040.] * 170,
    }
    results = {}
    for name, values in cases.items():
        results[name] = {}
        for method in ("trend_cusum", "bocpd"):
            output = run_stream(synthetic(values, stream=name), method)
            results[name][method] = {
                "candidate_count": len(output["events"]),
                "t_emit": [event["t_emit"] for event in output["events"]],
                "physical_t1_confirmed": False,
                "unknown_rows": output["status_counts"].get("unknown", 0),
            }
    return {"seed": 2601, "sampling_interval_s": .1, "configuration_count": 1,
            "warning": "synthetic_model_diagnostics_not_physical_event_accuracy",
            "cases": results}


def test_nig_update_and_empty_reset_prior():
    model = StudentTBOCPD()
    result = model.update(2., 0.)
    assert result["hazard"] == 0 and result["p_run_zero"] == 0
    assert model.mu.tolist() == [0., 1.]
    assert model.kappa.tolist() == [1., 2.]
    assert model.alpha.tolist() == [2., 2.5]
    assert model.beta.tolist() == [1., 2.]
    assert model.posterior.tolist() == [0., 1.]


def test_constant_hazard_p0_is_data_independent_and_posterior_normalized():
    streams = [np.zeros(120), np.r_[np.zeros(60), np.full(60, 8.)],
               np.random.default_rng(23).normal(0, 3, 120)]
    trajectories = []
    for values in streams:
        model, zeros = StudentTBOCPD(), []
        for i, value in enumerate(values):
            row = model.update(value, i * .2)
            assert np.isclose(model.posterior.sum(), 1., atol=1e-12)
            assert row["p_run_zero"] == pytest.approx(row["hazard"], abs=1e-15)
            assert row["posterior_states"] == i + 2  # no tail truncation
            zeros.append(row["p_run_zero"])
        trajectories.append(zeros)
    np.testing.assert_allclose(trajectories[0], trajectories[1], atol=1e-15)
    np.testing.assert_allclose(trajectories[0], trajectories[2], atol=1e-15)


def test_elapsed_time_hazard_not_frame_count_and_long_gap_requires_reset():
    model = StudentTBOCPD()
    model.update(0., 0.)
    for now, dt in [(.1, .1), (.4, .3), (.6, .2)]:
        result = model.update(0., now)
        assert result["hazard"] == pytest.approx(1 - math.exp(-dt / 30.))
    with pytest.raises(ValueError, match="long observation gap"):
        model.update(0., 1.251)


@pytest.mark.parametrize("method", ["trend_cusum", "bocpd"])
def test_constant_noise_linear_drift_and_mod_pi_do_not_make_reversal(method):
    times = np.arange(240) * .1
    rng = np.random.default_rng(37)
    values = 1000. + .2 * times + rng.normal(0, .2, len(times))
    true_angles = 1.4 + .02 * times
    wrapped = (true_angles + math.pi / 2) % math.pi - math.pi / 2
    result = run_stream(synthetic(values, angles=wrapped), method)
    assert result["events"] == []
    actual = [row["channels"]["angle_proxy_rad"]["unwrapped_or_native_value"]
              for row in result["rows"]]
    np.testing.assert_allclose(actual, true_angles, atol=1e-12)
    assert unwrap_mod_pi(0., math.pi) == pytest.approx(0., abs=1e-12)


@pytest.mark.parametrize("method", ["trend_cusum", "bocpd"])
def test_mean_step_is_candidate_with_immutable_first_emit(method):
    rng = np.random.default_rng(12)
    values = 1000. + rng.normal(0, .2, 240)
    values[100:] += 12.
    rows = synthetic(values)
    result = run_stream(rows, method)
    assert result["events"], method
    first = result["events"][0]
    assert 10. <= first["t_emit"] <= 12.
    assert first["max_source_timestamp"] <= first["t_emit"]
    assert first["t_est_hi"] <= first["t_emit"]
    assert first["status"] == "candidate" and not first["physical_t1_confirmed"]
    emitted_prefix = [r for r in rows if r["available_at_s"] <= first["t_emit"]]
    assert run_stream(emitted_prefix, method)["events"][0] == first


def test_bocpd_variance_step_changes_short_run_mass_not_p0():
    rng = np.random.default_rng(5)
    model = StudentTBOCPD()
    values = np.r_[rng.normal(0, .2, 100), rng.normal(0, 5., 50)]
    results = [model.update(value, i * .1) for i, value in enumerate(values)]
    assert max(r["short_run_mass_excluding_zero"] for r in results[100:110]) > .5
    for r in results[1:]:
        assert r["p_run_zero"] == pytest.approx(1 - math.exp(-.1 / 30.))


@pytest.mark.parametrize("method", ["trend_cusum", "bocpd"])
def test_single_outlier_limit_is_explicit_not_t1_confirmation(method):
    values = np.full(200, 1000.)
    values[100] += 100.
    result = run_stream(synthetic(values), method)
    if method == "trend_cusum":
        assert not result["events"]
    else:
        # Standard BOCPD may start a new segment after an isolated outlier;
        # its short-run mass then persists. This is a retained synthetic
        # false candidate, NOT evidence that outlier robustness passed.
        assert result["events"]
        assert all(e["status"] == "candidate" and not e["physical_t1_confirmed"]
                   for e in result["events"])
        assert "single_outlier_can_trigger" in DEFAULT_CONFIG["bocpd_outlier_policy"]


@pytest.mark.parametrize("method", ["trend_cusum", "bocpd"])
def test_long_missing_restarts_warmup_and_never_holds_observations(method):
    values = [1000.] * 80 + [None] * 20 + [1040.] * 80
    rows = synthetic(values)
    for row in rows[80:100]:
        row["x_ref_px"] = 1.e9  # stale payload is ignored when unavailable
    result = run_stream(rows, method)
    assert not result["events"]
    assert result["status_counts"] == {"candidate": 160, "unknown": 20}
    for row in result["rows"][80:100]:
        assert not row["available"]
        assert all(c["status"] == "unknown" for c in row["channels"].values())
    for row in result["rows"][100:150]:
        assert not row["channels"]["x_ref_px"]["ready"]


def test_point_six_five_gap_boundary_and_dropped_frame_indices():
    rows = synthetic([1000.] * 100)
    detector = CausalEventDetector("synthetic")
    for row in rows[:80]:
        detector.update(row)
    last = copy.deepcopy(rows[80])
    last.update(time_s=8.55, max_source_timestamp=8.55, available_at_s=8.55,
                source_frame=85)
    assert not detector.update(last)["channels"]["x_ref_px"]["continuity_reset"]
    last.update(time_s=9.2001, max_source_timestamp=9.2001, available_at_s=9.2001,
                source_frame=92)
    output = detector.update(last)
    assert output["channels"]["x_ref_px"]["continuity_reset"]
    assert not output["channels"]["x_ref_px"]["ready"]


@pytest.mark.parametrize("method", ["trend_cusum", "bocpd"])
def test_sparse_valid_frames_can_make_residual_unavailable_without_crash(method):
    rows = synthetic([1000.] * 200)
    # Every raw observation gap stays <=.5s, but the trend temporarily has
    # fewer than six past points. No synthetic residual observation is added.
    rows = [r for r in rows if r["time_s"] < 8 or r["time_s"] > 11
            or r["source_frame"] % 5 == 0]
    result = run_stream(rows, method)
    assert not result["events"]
    assert any(r["channels"]["x_ref_px"]["continuity_reset"] for r in result["rows"])


@pytest.mark.parametrize("method", ["trend_cusum", "bocpd"])
def test_future_replacement_and_annotation_fields_do_not_change_prefix(method):
    rng = np.random.default_rng(41)
    rows = synthetic(1000 + rng.normal(size=220))
    changed = copy.deepcopy(rows)
    for i, row in enumerate(changed):
        row["ground_truth"] = {"t1": i / 2, "confirmed": True}
        row["label_path"] = "must_not_be_read"
        if i >= 130:
            row["x_ref_px"] = 1.e4 + i
            row["angle_proxy_rad"] = .9
    original, alternative = run_stream(rows, method), run_stream(changed, method)
    assert original["rows"][:130] == alternative["rows"][:130]
    assert original["rows"][:130] == run_stream(rows[:130], method)["rows"]
    assert json.dumps(original, allow_nan=False)


def test_separate_stream_state_and_reject_mixed_video():
    a = synthetic([1000.] * 160, stream="video_A")
    b = synthetic([2000.] * 160, stream="video_B")
    first, second = CausalEventDetector("video_A"), CausalEventDetector("video_B")
    interleaved_a, interleaved_b = [], []
    for row_a, row_b in zip(a, b):
        interleaved_a.append(first.update(row_a))
        interleaved_b.append(second.update(row_b))
    assert interleaved_a == run_stream(a)["rows"]
    assert interleaved_b == run_stream(b)["rows"]
    with pytest.raises(ValueError, match="cross-stream"):
        first.update(b[-1])


def test_duplicate_out_of_order_future_dependency_and_invalid_timestamp_rejected():
    detector = CausalEventDetector("synthetic")
    rows = synthetic([1., 2., 3.])
    detector.update(rows[0])
    with pytest.raises(ValueError, match="duplicate/out-of-order"):
        detector.update(rows[0])
    bad = dict(rows[1], max_source_timestamp=100.)
    with pytest.raises(ValueError, match="future input"):
        detector.update(bad)
    with pytest.raises(ValueError, match="invalid timestamp"):
        detector.update(dict(rows[1], time_s=math.nan))
    detector.update(rows[1])  # invalid packets did not advance/reverse state
    with pytest.raises(ValueError, match="duplicate/out-of-order"):
        detector.update(dict(rows[2], source_frame=1))


def test_inference_module_has_no_device_or_file_io_imports():
    from vision_geometry_lab.experiments.v26_causal_material import event_signals
    tree = ast.parse(inspect.getsource(event_signals))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module.split(".")[0])
    assert imports <= {"__future__", "collections", "copy", "math", "numpy"}
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id in {"open", "eval", "exec", "__import__"}
                   for node in ast.walk(tree))


@pytest.mark.parametrize("method", ["trend_cusum", "bocpd"])
def test_periodicity_is_tested_but_never_physical_confirmation(method):
    times = np.arange(300) * .1
    rows = synthetic(1000. + 5. * np.sin(2 * math.pi * times / 3.))
    result = run_stream(rows, method)
    assert result == run_stream(rows, method)
    assert result["evaluation_status"].startswith("not_evaluable")
    assert all(e["status"] == "candidate" and not e["physical_t1_confirmed"]
               for e in result["events"])
    assert json.dumps(result, allow_nan=False)


def test_fixed_configuration_and_detached_append_only_event_payloads():
    with pytest.raises(ValueError, match="one fixed configuration"):
        CausalEventDetector("synthetic", config=dict(DEFAULT_CONFIG, warmup_s=0.))
    rows = synthetic([1000.] * 100 + [1020.] * 100)
    detector = CausalEventDetector("synthetic")
    for row in rows:
        result = detector.update(row)
        if result["event"]:
            result["event"]["status"] = "confirmed"
    assert detector.events and detector.events[0]["status"] == "candidate"
    events = detector.events
    events.clear()
    assert detector.events


if __name__ == "__main__":
    print(json.dumps(synthetic_case_summary(), ensure_ascii=False, indent=2))
