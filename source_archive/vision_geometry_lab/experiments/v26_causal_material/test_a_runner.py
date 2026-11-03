"""A runner tests use only generated measurements and temporary directories."""
import ast
import copy
import json

from PIL import Image
import pytest

from vision_geometry_lab.experiments.v26_causal_material.a_runner import (
    CONFIG, METHODS, ROOT, SOURCE_FILES, replay_rows, run, sha256,
)
from vision_geometry_lab.experiments.v26_causal_material.test_event_signals import synthetic


PROTOCOL = "a" * 64


def rows(stream="left", count=180):
    result = synthetic([1000.] * 80 + [1020.] * (count - 80), dt=.2, stream=stream)
    for row in result:
        row.update(old_gap_usable=False, old_gap_proxy=None,
                   timestamp_basis="synthetic_host_receive_not_exposure")
    return result


def write_features(base, *, extra=None):
    folder = base / "features"
    folder.mkdir(parents=True)
    for side in ("left", "right"):
        payload = {"rows": rows(side), "protocol_sha256": PROTOCOL}
        if extra:
            payload.update(extra)
        (folder / f"{side}.json").write_text(json.dumps(payload), encoding="utf-8")
    return {side: sha256(folder / f"{side}.json") for side in ("left", "right")}


def deterministic_payload(replay):
    return {method: {"rows": replay["methods"][method]["rows"],
                     "events": replay["methods"][method]["events"]} for method in METHODS}


def test_old_baseline_keeps_true_gap_unavailable_not_substituted_x():
    result = replay_rows(rows())
    old = result["methods"]["legacy_gap_baseline"]
    assert old["summary"]["source_row_denominator"] == 180
    assert old["summary"]["available_rows"] == 0
    assert not old["events"]
    for method in METHODS:
        summary = result["methods"][method]["summary"]
        assert summary["physical_t1"]["status"] == "not_evaluable"
        assert summary["physical_t2"]["status"] == "not_evaluable"
        assert summary["false_alarms_per_verified_negative_minute"] is None


def test_old_t1_name_is_only_a_proxy_and_first_emit_uses_arrival_clock():
    gaps = [1.2, 1., .8, .6, .4, .2, .15, .1, .1, .2, .3, .4, .5, .6, .7, .8]
    source = synthetic([1000.] * len(gaps), dt=.2)
    for row, gap in zip(source, gaps):
        row.update(old_gap_usable=True, old_gap_proxy=gap)
        row["time_s"] += 100.
        row["max_source_timestamp"] += 100.
        row["available_at_s"] += 100.25
    result = replay_rows(source)
    legacy = result["methods"]["legacy_gap_baseline"]
    assert {e["event_type"] for e in legacy["events"]} == {"legacy_gap_t0_proxy", "legacy_gap_t1_proxy"}
    for event in legacy["events"]:
        assert not event["physical_t1_confirmed"]
        assert event["t_emit"] - event["max_source_timestamp"] == pytest.approx(.25)
        assert event["t_est_hi"] >= 100.
        assert event["status"] == "candidate"


def test_labels_and_future_changes_cannot_rewrite_first_events_or_prefix():
    source = rows()
    modified = copy.deepcopy(source)
    for i, row in enumerate(modified):
        row["confirmed_t1"] = 27.3
        row["label_path"] = "must-not-open"
        if i >= 130:
            row["x_ref_px"] = 9.e4
    first = deterministic_payload(replay_rows(source))
    second = deterministic_payload(replay_rows(modified))
    prefix = deterministic_payload(replay_rows(source[:130]))
    for method in METHODS:
        assert first[method]["rows"][:130] == second[method]["rows"][:130]
        assert first[method]["rows"][:130] == prefix[method]["rows"]
        assert "confirmed_t1" not in json.dumps(first[method])


def test_event_sink_is_called_on_first_emission_and_payload_detached():
    seen = {method: [] for method in METHODS}
    def sink_for(method):
        def sink(event):
            seen[method].append(copy.deepcopy(event))
            event["status"] = "confirmed"
        return sink
    result = replay_rows(rows(), event_sinks={m: sink_for(m) for m in METHODS})
    for method in METHODS:
        assert seen[method] == result["methods"][method]["events"]
        assert all(e["status"] == "candidate" for e in seen[method])


def test_runner_writes_all_denominators_exclusive_event_logs_and_figures(tmp_path):
    base, output = tmp_path / "base", tmp_path / "new_run"
    inputs = write_features(base)
    sources = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    report = run(base, output, PROTOCOL, inputs, sources)
    assert report["physical_t1"]["status"] == "not_evaluable"
    assert report["physical_t2"]["status"] == "not_evaluable"
    for side in ("left", "right"):
        for method in METHODS:
            saved = json.loads((output / side / f"{method}.json").read_text(encoding="utf-8"))
            log = (output / side / f"{method}.events.jsonl").read_text(encoding="utf-8")
            events = [json.loads(line) for line in log.splitlines()]
            assert events == saved["events"]
            assert saved["summary"]["source_row_denominator"] == 180
        with Image.open(output / side / "timeline.png") as image:
            assert image.size == (1900, 1600)
    assert report["production_end_to_end_latency_ms"] is None
    receipt = (output / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        run(base, output, PROTOCOL, inputs, sources)
    assert (output / "report.json").read_bytes() == receipt


def test_feature_and_source_hash_mismatch_fail_before_output_creation(tmp_path):
    base = tmp_path / "base"
    inputs = write_features(base)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="feature hash"):
        run(base, output, PROTOCOL, dict(inputs, left="0" * 64))
    assert not output.exists()
    with pytest.raises(ValueError, match="source freeze"):
        run(base, output, PROTOCOL, inputs, {})
    assert not output.exists()


def test_feature_protocol_empty_stream_and_mixed_video_rejected(tmp_path):
    base = tmp_path / "base"
    write_features(base, extra={"protocol_sha256": "b" * 64})
    with pytest.raises(ValueError, match="protocol mismatch"):
        run(base, tmp_path / "output", PROTOCOL)
    with pytest.raises(ValueError, match="empty streams"):
        replay_rows([])
    mixed = rows()[:10]
    mixed[-1]["stream_id"] = "other_video"
    with pytest.raises(ValueError, match="cross-stream"):
        replay_rows(mixed)


def test_legacy_import_closure_contains_no_hardware_import_or_top_level_calls():
    approved = {"__future__", "collections", "dataclasses", "copy", "math", "numpy",
                "vision_prototype"}
    for relative in SOURCE_FILES:
        if relative not in ("vision_geometry_lab/v16_event_candidates.py",
                             "vision_prototype/centerline_event_fusion.py",
                             "vision_prototype/visual_turn_phase.py"):
            continue
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] in approved for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert node.module.split(".")[0] in approved
        assert not any(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                       for node in tree.body)
    assert CONFIG["legacy_period_input"] is None
    assert CONFIG["legacy_processing_ms"] == 0
    assert CONFIG["confirmed_physical_labels"] == []
