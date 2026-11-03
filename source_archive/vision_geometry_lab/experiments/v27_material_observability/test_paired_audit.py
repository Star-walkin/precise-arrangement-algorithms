"""Synthetic V27 pairing/geometry/lifecycle tests. No videos or real labels."""
from __future__ import annotations

import ast
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from vision_geometry_lab.experiments.v27_material_observability import paired_audit as pa


def geometry(frame=0, time_s=0., *, rho=220., normal=(0., 1.), available=True,
             segment=((100., 220.), (164., 220.)), camera="left"):
    return dict(session=pa.SESSIONS[camera], camera=camera, source_frame=frame,
        time_s=time_s, available_at_s=time_s, max_source_timestamp=time_s,
        timestamp_kind=pa.TIME_KIND, available=available, reason="synthetic",
        normal_xy=list(normal), rho_px=rho, width_px=10., segment_xy=list(map(list, segment)),
        cache_sha256="a"*64)


def fixture(n=6, masks=None, shifts=None):
    seed = [[110., 220.], [120., 220.], [130., 220.], [140., 220.]]
    meta = dict(group_id="left_000000", side="left", start_index=0, available=True,
                reason="synthetic", box=[100, 200, 64, 64], seeds_native=seed,
                free_seed=[[100., 220.], [164., 220.]])
    yy, xx = np.mgrid[:64, :64]
    gray = np.asarray(((xx+yy) % 2)*180+20, np.uint8)
    data = dict(gray_native=np.repeat(gray[None], n, axis=0), times=np.arange(n)*.1,
                source_frames=np.arange(n), pixel_hashes=np.array([f"pixels-{i}" for i in range(n)]))
    tracks = {}
    for method in pa.METHODS:
        rows = []
        for slot in range(n):
            delta = np.tile([float(slot), 0.], (4, 1)) if shifts is None else np.asarray(shifts[method][slot], float)
            valid = [True]*4 if masks is None else masks[method][slot]
            rows.append(dict(frame_slot_index=slot, source_frame=slot,
                source_frame_basis="video_frame_index_not_camera_counter", time_s=float(data["times"][slot]),
                age_s=float(data["times"][slot]), t_emit=float(data["times"][slot]),
                max_source_timestamp=float(data["times"][slot]), point_ids=list(range(4)),
                points_xy_native=(np.asarray(seed)+delta).tolist(), valid_candidate=valid,
                rejection_reason=None if any(valid) else "all_points_rejected"))
        tracks[method] = dict(group_id=meta["group_id"], side="left", method=method,
                              seeds_native=seed, box=meta["box"], rows=rows)
    records = [geometry(frame=i, time_s=float(data["times"][i])) for i in range(0, n, 2)]
    return meta, data, tracks, records


def _set_times(data, tracks, times):
    data["times"] = np.asarray(times, float)
    for track in tracks.values():
        for i, row in enumerate(track["rows"]):
            row.update(time_s=float(times[i]), t_emit=float(times[i]), max_source_timestamp=float(times[i]),
                       age_s=float(times[i]-times[0]))


def test_all_seed_states_and_four_disjoint_acceptance_sets():
    masks = {"lk256": [[True]*4, [True, True, False, False]],
             "lk_native": [[True]*4, [True, False, True, False]]}
    result = pa.audit_group(*fixture(2, masks=masks))
    assert len(result["seed_states"]) == 8
    assert [p["state"] for p in result["pairs"]] == ["both", "only_lk256", "only_lk_native", "neither"]
    assert result["summary"]["original_seed_pair_denominator"] == 4
    assert result["summary"]["both_accepted_pairs"] == 1
    assert all(p["paired_difference"] is None for p in result["pairs"][1:])


def test_medians_are_computed_from_paired_ids_not_separate_survivors():
    masks = {"lk256": [[True]*4, [True, True, False, False]],
             "lk_native": [[True]*4, [True, False, True, False]]}
    zero = np.zeros((4, 2)).tolist()
    shifts = {"lk256": [zero, [[2, 0], [100, 0], [0, 0], [0, 0]]],
              "lk_native": [zero, [[1, 0], [0, 0], [-100, 0], [0, 0]]]}
    result = pa.audit_group(*fixture(2, masks=masks, shifts=shifts))
    assert result["summary"]["paired_difference_initial_tangent_px"]["median"] == 1.
    assert result["summary"]["same_common_ID_method_motion"]["lk256"]["initial_tangent_px"]["median"] == 2.
    assert result["summary"]["same_common_ID_method_motion"]["lk_native"]["initial_tangent_px"]["median"] == 1.


def test_point_ids_join_by_identity_not_array_order():
    args = fixture(3)
    expected = pa.audit_group(*copy.deepcopy(args))
    order = [2, 0, 3, 1]
    for row in args[2]["lk_native"]["rows"]:
        for field in ("point_ids", "points_xy_native", "valid_candidate"):
            row[field] = [row[field][i] for i in order]
    assert pa.audit_group(*args) == expected


@pytest.mark.parametrize("fault", ["duplicate_id", "missing_point", "shifted_initial_native", "different_group", "different_camera", "different_seed", "different_box"])
def test_mismatched_point_or_native_coordinate_contract_rejected(fault):
    meta, data, tracks, records = fixture(2)
    track = tracks["lk_native"]
    if fault == "duplicate_id": track["rows"][1]["point_ids"] = [0, 0, 2, 3]
    elif fault == "missing_point": track["rows"][1]["points_xy_native"].pop()
    elif fault == "shifted_initial_native": track["rows"][0]["points_xy_native"][0][0] += .5
    elif fault == "different_group": track["group_id"] = "left_000020"
    elif fault == "different_camera": track["side"] = "right"
    elif fault == "different_seed": track["seeds_native"] = [[10, 20]]*4
    else: track["box"] = [0, 0, 64, 64]
    with pytest.raises(ValueError): pa.audit_group(meta, data, tracks, records)


def test_no_identity_revival_or_gap_bridging():
    masks = {m: [[True]*4, [False, True, True, True], [False, True, True, True]] for m in pa.METHODS}
    args = fixture(3, masks=masks)
    result = pa.audit_group(*args)
    assert [p["state"] for p in result["pairs"] if p["point_id"] == 0] == ["neither", "neither"]
    args[2]["lk_native"]["rows"][2]["valid_candidate"][0] = True
    with pytest.raises(ValueError, match="reidentification"): pa.audit_group(*args)


@pytest.mark.parametrize("fault", ["missing", "duplicate_time", "reordered", "nonfinite", "nominal_clock", "counter_basis", "backdated_emit"])
def test_missing_and_invalid_temporal_contract_fail_closed(fault):
    meta, data, tracks, records = fixture(3)
    if fault == "missing": tracks["lk256"]["rows"].pop()
    elif fault == "duplicate_time": data["times"][2] = data["times"][1]
    elif fault == "reordered": data["source_frames"] = np.array([0, 2, 1])
    elif fault == "nonfinite": data["times"][1] = np.nan
    elif fault == "nominal_clock": data["timestamp_kind"] = "avi_frame_index_over_fps"
    elif fault == "counter_basis": tracks["lk256"]["rows"][0]["source_frame_basis"] = "device_counter"
    else: tracks["lk256"]["rows"][1]["t_emit"] = 0.
    with pytest.raises(ValueError): pa.audit_group(meta, data, tracks, records)


def test_geometry_selection_uses_only_arrived_past_and_records_age():
    records = [geometry(), geometry(2, .2, rho=230.)]
    selected = pa.select_geometry(records, .1, 1)
    assert selected["record"]["source_frame"] == 0 and selected["age_s"] == .1
    assert pa.select_geometry(records, .2, 2)["record"]["source_frame"] == 2
    records[1]["available_at_s"] = .24
    assert pa.select_geometry(records, .21, 2)["record"]["source_frame"] == 0
    assert not pa.select_geometry(records, -.1, 0)["available"]


def test_latest_invalid_geometry_is_not_backfilled_and_expiry_is_explicit():
    records = [geometry(), geometry(2, .2, available=False)]
    selected = pa.select_geometry(records, .21, 2)
    assert not selected["available"] and selected["record"]["source_frame"] == 2
    assert pa.select_geometry(records[:1], .25, 3)["available"]
    expired = pa.select_geometry(records[:1], .25001, 3)
    assert not expired["available"] and expired["reason"] == "geometry_expired"


@pytest.mark.parametrize("fault", ["wrong_clock", "duplicate", "mixed_video", "future_dependency"])
def test_geometry_stream_time_and_session_integrity(fault):
    args = fixture(3)
    records = args[-1]
    if fault == "wrong_clock": records[0]["timestamp_kind"] = "exposure"
    elif fault == "duplicate": records[1]["time_s"] = 0.
    elif fault == "mixed_video": records[1]["session"] = "another_session"
    else: records[0]["max_source_timestamp"] = 10.
    with pytest.raises(ValueError): pa.audit_group(*args)


def test_fixed_reference_does_not_follow_segment_endpoint_or_width():
    record = geometry()
    frame = pa.line_frame(record, [131.5, 231.5], [1, 0])
    altered = copy.deepcopy(record)
    altered.update(segment_xy=[[120, 220], [184, 220]], width_px=20.)
    other = pa.line_frame(altered, [131.5, 231.5], [1, 0])
    assert other["origin_xy"] == frame["origin_xy"] == [131.5, 220.]
    motion = pa.motion_diagnostic([120, 220], [127, 220], .1, [1, 0], frame, other)
    assert motion["tangent_px"] == motion["compensated"]["tangent_px"] == 7.
    assert motion["optical_width_ratio"] == 2. and motion["metric_velocity_m_s"] is None


def test_independent_normal_translation_removed_without_removing_axial_motion():
    first = pa.line_frame(geometry(), [131.5, 231.5], [1, 0])
    second = pa.line_frame(geometry(rho=225., segment=((100, 225), (164, 225))), [131.5, 231.5], [1, 0])
    motion = pa.motion_diagnostic([120, 220], [127, 225], .1, [1, 0], first, second)
    assert motion["delta_xy_native_px"] == [7., 5.]
    assert motion["compensated"]["delta_xy_native_px"] == [7., 0.]
    assert motion["compensated"]["apparent_tangent_px_s"] == 70.


def test_unoriented_line_normal_flip_gives_identical_mapping():
    first = pa.line_frame(geometry(), [131.5, 231.5], [1, 0])
    second = pa.line_frame(geometry(rho=-220., normal=(0, -1)), [131.5, 231.5], [1, 0])
    assert first == second


def test_independent_rotation_about_fixed_reference_is_conditional_only():
    r = np.array([131.5, 231.5])
    angle = .2
    d = np.array([np.cos(angle), np.sin(angle)])
    n = np.array([-d[1], d[0]])
    g0 = geometry(rho=r[1], segment=(r-[20, 0], r+[20, 0]))
    g1 = geometry(rho=float(n @ r), normal=n, segment=(r-20*d, r+20*d))
    f0, f1 = pa.line_frame(g0, r, [1, 0]), pa.line_frame(g1, r, [1, 0])
    result = pa.motion_diagnostic(r+[10, 0], r+10*d, .1, [1, 0], f0, f1)
    assert np.linalg.norm(result["delta_xy_native_px"]) > 1.
    assert np.linalg.norm(result["compensated"]["delta_xy_native_px"]) < 1e-10
    assert result["compensated"]["conditional_only"] and not result["compensated"]["material_motion_confirmed"]


def test_gray_quality_failure_does_not_change_original_acceptance_denominator():
    args = fixture(2)
    args[1]["gray_native"][:] = 255
    result = pa.audit_group(*args)
    assert result["summary"]["both_accepted_pairs"] == 4
    assert result["summary"]["both_with_geometric_patch_condition"] == 0
    item = result["seed_states"][0]["methods"]["lk256"]
    assert item["patch_quality"]["saturation"] == 1.
    assert item["accepted"] and not item["patch_quality"]["visibility_verified"]
    assert not result["pairs"][0]["methods"]["lk256"]["motion"]["compensated"]["geometric_and_patch_condition"]


def test_predicted_exit_or_unusable_texture_does_not_prove_physical_exit():
    args = fixture(3)
    for track in args[2].values():
        for row in track["rows"][1:]:
            row["points_xy_native"][0] = [9999., 9999.]
            row["valid_candidate"][0] = False
    result = pa.audit_group(*args)
    lost = [f for f in result["failures"] if f["point_id"] == 0]
    assert all(f["category"] == "algorithm_rejected" for f in lost)
    assert all(f["physical_cause"] == "unknown" and not f["crop_exit_observed"] and not f["occluded_verified"] for f in lost)


@pytest.mark.parametrize("age,expected", [(2., "algorithm_rejected"), (2.0000000005, "algorithm_rejected"), (2.000002, "ttl_expired")])
def test_ttl_boundary_without_source_gap(age, expected):
    times = np.linspace(0., age, 21)
    args = fixture(21)
    _set_times(args[1], args[2], times)
    args = (*args[:3], [geometry(i, float(times[i])) for i in range(0, 21, 2)])
    for track in args[2].values():
        track["rows"][-1]["valid_candidate"] = [False]*4
    result = pa.audit_group(*args)
    assert {f["category"] for f in result["failures"]} == {expected}


def test_later_ttl_cannot_overwrite_first_algorithm_loss():
    args = fixture(21)
    _set_times(args[1], args[2], np.linspace(0., 2.01, 21))
    args = (*args[:3], [geometry(i, float(args[1]["times"][i])) for i in range(0, 21, 2)])
    for track in args[2].values():
        for row in track["rows"][3:]: row["valid_candidate"][0] = False
        track["rows"][-1].update(valid_candidate=[False]*4, rejection_reason="cohort_age_limit")
    result = pa.audit_group(*args)
    first = [f for f in result["failures"] if f["point_id"] == 0]
    assert all(f["category"] == "algorithm_rejected" and f["first_invalid_slot"] == 3 for f in first)
    assert all(f["final_row_reason"] == "cohort_age_limit" for f in first)
    assert all(f["actual_observable_duration_s"] is None for f in first)


@pytest.mark.parametrize("fault", ["receive_gap", "saved_frame_gap", "duplicate_pixels"])
def test_source_gap_class_preserves_specific_observation_failure(fault):
    args = fixture(2)
    if fault == "receive_gap": _set_times(args[1], args[2], [0., .26])
    elif fault == "saved_frame_gap":
        args[1]["source_frames"][1] = 2
        for track in args[2].values(): track["rows"][1]["source_frame"] = 2
    else: args[1]["pixel_hashes"][1] = args[1]["pixel_hashes"][0]
    for track in args[2].values(): track["rows"][1]["valid_candidate"] = [False]*4
    assert {f["category"] for f in pa.audit_group(*args)["failures"]} == {"source_gap"}


def test_unavailable_seed_group_keeps_frame_denominator_and_unknown():
    meta, data, tracks, records = fixture(3)
    meta.update(available=False, seeds_native=[], reason="insufficient_texture")
    for track in tracks.values():
        track["seeds_native"] = []
        for row in track["rows"]:
            row.update(point_ids=[], points_xy_native=[], valid_candidate=[])
    result = pa.audit_group(meta, data, tracks, records)
    assert result["summary"]["scheduled_frames"] == 3
    assert result["summary"]["scheduled_frame_pairs"] == 2
    assert result["summary"]["initialized_points"] == 0
    assert result["pairs"] == [] and not result["summary"]["seed_available"]


def test_future_replacement_cannot_change_prefix_pairs_or_seed_states():
    args = fixture(6)
    baseline = pa.audit_group(*copy.deepcopy(args))
    for track in args[2].values():
        for row in track["rows"][3:]:
            row["points_xy_native"] = [[150., 240.]]*4
            row["valid_candidate"] = [False]*4
    args[1]["gray_native"][3:] = 255
    args[-1][-1] = geometry(4, .4, rho=300.)
    altered = pa.audit_group(*args)
    assert baseline["pairs"][:8] == altered["pairs"][:8]
    assert baseline["seed_states"][:12] == altered["seed_states"][:12]


def test_extra_annotation_fields_never_affect_diagnostic_and_meters_stay_null():
    args = fixture(2)
    result = pa.audit_group(*copy.deepcopy(args))
    args[0]["ground_truth"] = {"true_velocity": 10000, "t1": 27.3}
    args[2]["lk256"]["truth"] = {"correct_point": [999, 999]}
    assert pa.audit_group(*args) == result
    assert all(p["metric_velocity_m_s"] is None and not p["material_identity_confirmed"] for p in result["pairs"])


def test_receive_timestamp_is_only_input_lower_bound_not_live_result_availability():
    args = fixture(3)
    result = pa.audit_group(*args)
    for pair in result["pairs"]:
        assert pair["t_available"] is None and pair["actual_live_t_available"] is None
        assert pair["source_available_lower_bound_s"] == args[1]["times"][pair["frame_slot1"]]
        assert pair["max_source_timestamp"] == pair["source_available_lower_bound_s"]
        assert "unmeasured_live_completion" in pair["availability_clock"]


def test_csv_and_jsonl_roundtrip_keep_each_seed_pair_and_no_nonfinite(tmp_path):
    result = pa.audit_group(*fixture(2))
    pa._csv(tmp_path / "pairs.csv", result["pairs"])
    pa._jsonl(tmp_path / "states.jsonl", result["seed_states"])
    with (tmp_path / "pairs.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert json.loads(rows[2]["detail_json"]) == result["pairs"][2]
    assert len((tmp_path / "states.jsonl").read_text(encoding="utf-8").splitlines()) == 8
    with pytest.raises(FileExistsError): pa._csv(tmp_path / "pairs.csv", [])


def test_hash_mismatch_rejected_before_any_output(tmp_path):
    path = tmp_path / "one.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        pa._record(path, {}, {"one.json": "0"*64}, tmp_path)
    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(FileExistsError): pa.run(tmp_path, out, "a"*64, {})


def _geometry_cache_fixture(tmp_path):
    base = tmp_path / "v26"
    (base / "features").mkdir(parents=True)
    rows, hashes = [], {}
    for i in range(0, 601, 2):
        key = f"vision_geometry_lab/outputs/v16_event_feasibility/extract/left/frames/f{i:06d}.json"
        path = tmp_path / key
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(session="left", frame_index=i, frame_sha256=f"pixels-{i}", time_s=i/10,
            rope=dict(valid=True, reason="synthetic", normal=[0, 1], rho_px=220.,
                      observed_segment=[[100, 220], [164, 220]], width_px=10.))
        path.write_text(json.dumps(payload), encoding="utf-8")
        hashes[key] = pa.sha(path)
        timestamp = 100. + i/10
        rows.append(dict(stream_id=f"{pa.SESSIONS['left']}:left", source_frame=i,
            timestamp_basis=pa.TIME_KIND, time_s=timestamp, available_at_s=timestamp,
            max_source_timestamp=timestamp, available=True, reason="synthetic", pixel_sha256=f"pixels-{i}"))
    (base / "protocol.json").write_text(json.dumps(dict(data_sha256=hashes)), encoding="utf-8")
    feature_path = base / "features/left.json"
    feature_path.write_text(json.dumps(dict(protocol_sha256=pa.sha(base / "protocol.json"), rows=rows)), encoding="utf-8")
    # V26 pipeline.prepare receipt: features are keyed by side in their own map;
    # input_hashes is exclusively for cohort meta.json/data.npz inputs to route B.
    receipt = dict(protocol_sha256=pa.sha(base / "protocol.json"),
        input_hashes={"cohorts/left_000000/meta.json": "a"*64,
                      "cohorts/left_000000/data.npz": "b"*64},
        feature_sha256={"left": pa.sha(feature_path), "right": "c"*64},
        audits={}, preparation_wall_s=0., label_files_read=[], hardware_access=False)
    (base / "prepare_manifest.json").write_text(json.dumps(receipt), encoding="utf-8")
    return base


def test_geometry_loader_uses_verified_host_time_not_nominal_cache_time(tmp_path):
    base = _geometry_cache_fixture(tmp_path)
    hashes = {}
    records = pa.load_geometry_records(base, "left", workspace_root=tmp_path, input_hashes=hashes)
    assert len(records) == 301 and records[0]["time_s"] == 100.
    assert records[1]["source_frame"] == 2 and records[1]["time_s"] == 100.2
    assert len(hashes) == 304
    path = tmp_path / records[1]["cache_path"]
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="cached fit changed"):
        pa.load_geometry_records(base, "left", workspace_root=tmp_path)


@pytest.mark.parametrize("defect", ["missing_map", "missing_side", "wrong_digest",
                                   "obsolete_input_hashes_only", "tampered_feature"])
def test_geometry_loader_requires_actual_prepare_feature_hash_schema(tmp_path, defect):
    base = _geometry_cache_fixture(tmp_path)
    path = base / "prepare_manifest.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    feature_path = base / "features/left.json"
    if defect == "missing_map":
        del receipt["feature_sha256"]
    elif defect == "missing_side":
        del receipt["feature_sha256"]["left"]
    elif defect == "wrong_digest":
        receipt["feature_sha256"]["left"] = "0"*64
        # A matching digest in the wrong map must never bypass the true receipt.
        receipt["input_hashes"]["features/left.json"] = pa.sha(feature_path)
    elif defect == "obsolete_input_hashes_only":
        del receipt["feature_sha256"]
        receipt["input_hashes"]["features/left.json"] = pa.sha(feature_path)
    else:
        feature_path.write_text(feature_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="prepared feature hash mismatch"):
        pa.load_geometry_records(base, "left", workspace_root=tmp_path)


def test_module_import_closure_has_no_tracker_ui_or_hardware_and_import_is_inert():
    source = Path(pa.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {node.names[0].name.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.Import)}
    imports |= {node.module.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imports <= {"__future__", "argparse", "bisect", "collections", "csv", "hashlib", "json", "math", "pathlib", "numpy"}
    code = """import numpy, builtins, importlib, socket
original = builtins.__import__
def guard(name, *args, **kwargs):
    if any(word in name.lower() for word in ('camera','plc','relay','serial','live_window','ctypes','cv2','torch','point_tracks')):
        raise RuntimeError('forbidden import '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guard
socket.socket=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('network forbidden'))
importlib.import_module('vision_geometry_lab.experiments.v27_material_observability.paired_audit')
print('inert import passed')
"""
    completed = subprocess.run([sys.executable, "-B", "-c", code], cwd=pa.ROOT, capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
    assert "inert import passed" in completed.stdout
