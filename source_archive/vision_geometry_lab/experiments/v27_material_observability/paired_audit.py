"""V27 same-ID paired audit of saved V26 outputs; no tracker, labels, or device.

Line-frame compensation is conditional on independently detected current/past
free geometry, not registration of material or a physical motion measurement.
No tracked-point transform is fitted. A point on a fitted line with a usable
gray patch is still only a geometric/appearance candidate, never verified rope.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
TIME_KIND = "host_monotonic_sdk_receive_not_exposure"
METHODS = ("lk256", "lk_native")
CONFIG = dict(schema="v27_paired_material_audit_1", methods=list(METHODS),
    sides=["left", "right"], starts=list(range(0, 600, 20)), cohort_frames=21,
    source_shape_hw=[3000, 4096], ttl_s=2., source_gap_s=.25, max_geometry_age_s=.25,
    min_points=4, patch_half=4, patch_max_saturation=.25, patch_min_contrast=8.,
    saturation_gray_threshold=250, free_band_half_widths=.5,
    pair_key=["session", "camera", "group_id", "point_id", "source_frame0", "source_frame1"],
    pair_acceptance="same_ID_accepted_at_both_adjacent_endpoints_per_method",
    comparison="paired_difference_lk256_minus_lk_native_only_on_common_IDs",
    geometry_selection="latest_arrived_5Hz_observation_no_future_interpolation_no_invalid_backfill",
    compensation="independent_line_rotation_normal_shift_about_fixed_crop_center_no_scale",
    timestamp_kind=TIME_KIND, metric_velocity_m_s=None, metric_status="metric_uncertain",
    result_availability_clock="unmeasured_live_completion_source_arrival_is_lower_bound_only",
    physical_exit_classification="unknown_without_independent_image_review",
    example_groups=["left_000240", "right_000140"], detail_frame_pair=[241, 242],
    candidate_only=True, material_identity_confirmed=False, deployment_permitted=False)
SOURCE_FILES = (
    "vision_geometry_lab/experiments/v27_material_observability/paired_audit.py",
    "vision_geometry_lab/experiments/v27_material_observability/test_paired_audit.py",
)
SESSIONS = dict(left="dual_20260924_111500", right="dual_20260924_111143")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def _key(path, root=ROOT):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def _record(path, hashes, expected=None, root=ROOT):
    key, digest = _key(path, root), sha(path)
    if expected is not None and expected.get(key) != digest:
        raise ValueError("frozen input hash mismatch: " + key)
    if hashes is not None:
        hashes[key] = digest
    return digest


def input_paths(v26_base, workspace_root=ROOT):
    """Exact source-relative path list for the parent V27 pre-run freeze."""
    base, root = Path(v26_base), Path(workspace_root)
    paths = [base / name for name in ("protocol.json", "prepare_manifest.json", "B/manifest.json", "B/report.json")]
    for side in CONFIG["sides"]:
        paths.append(base / f"features/{side}.json")
        for frame in range(0, 601, 2):
            paths.append(root / f"vision_geometry_lab/outputs/v16_event_feasibility/extract/{side}/frames/f{frame:06d}.json")
        for start in CONFIG["starts"]:
            group = f"{side}_{start:06d}"
            paths.extend(base / f"cohorts/{group}/{name}" for name in ("meta.json", "data.npz"))
            paths.extend(base / f"B/{method}/{group}.json" for method in METHODS)
    return {_key(path, root): path for path in paths}


def load_geometry_records(v26_base, side, *, input_hashes=None, expected_input_sha256=None, workspace_root=ROOT):
    """Load only SHA-bound V16 fits; map host times from the matching V26 rows.

    The loader contains the whole frozen stream for offline audit. Selection is
    independently past-only for each queried frame. No source AVI is opened.
    """
    if side not in SESSIONS:
        raise ValueError("unknown camera")
    base, root = Path(v26_base), Path(workspace_root)
    protocol = _read(base / "protocol.json")
    prepare = _read(base / "prepare_manifest.json")
    feature_hashes = prepare.get("feature_sha256")
    for name in ("protocol.json", "prepare_manifest.json", f"features/{side}.json"):
        digest = _record(base / name, input_hashes, expected_input_sha256, root)
        if name.startswith("features/") and (not isinstance(feature_hashes, dict) or
                                             feature_hashes.get(side) != digest):
            raise ValueError("prepared feature hash mismatch")
    features = _read(base / f"features/{side}.json")
    if features["protocol_sha256"] != sha(base / "protocol.json"):
        raise ValueError("feature protocol mismatch")
    if [r["source_frame"] for r in features["rows"]] != list(range(0, 601, 2)):
        raise ValueError("full frozen 5Hz sequence required")
    records = []
    for feature in features["rows"]:
        index = feature["source_frame"]
        if feature["timestamp_basis"] != TIME_KIND or feature["stream_id"] != f"{SESSIONS[side]}:{side}":
            raise ValueError("feature stream/timebase mismatch")
        key = f"vision_geometry_lab/outputs/v16_event_feasibility/extract/{side}/frames/f{index:06d}.json"
        digest = _record(root / key, input_hashes, expected_input_sha256, root)
        if protocol["data_sha256"].get(key) != digest:
            raise ValueError("V16 cached fit changed")
        cached = _read(root / key)
        if cached["frame_index"] != index or cached["session"] != side or cached["frame_sha256"] != feature["pixel_sha256"]:
            raise ValueError("fit not tied to the exact feature frame pixels")
        rope = cached.get("rope") or {}
        records.append(dict(session=SESSIONS[side], camera=side, source_frame=index,
            time_s=float(feature["time_s"]), available_at_s=float(feature["available_at_s"]),
            max_source_timestamp=float(feature["max_source_timestamp"]), timestamp_kind=TIME_KIND,
            geometry_availability_clock="inherited_V26_causal_simulation_not_measured_fit_completion",
            available=bool(rope.get("valid") and feature.get("available")),
            reason=feature["reason"] if not feature.get("available") else rope.get("reason", "unknown"),
            normal_xy=rope.get("normal"), rho_px=rope.get("rho_px"),
            segment_xy=rope.get("observed_segment"), width_px=rope.get("width_px"),
            cache_path=key, cache_sha256=digest, frame_sha256=cached["frame_sha256"],
            feature_quality={k: feature.get(k) for k in ("saturation", "contrast", "fit_support_coverage", "quality_reason")}))
    _validate_geometry_stream(records)
    return records


def _validate_geometry_stream(records):
    previous = None
    for row in records:
        fields = [row[k] for k in ("time_s", "available_at_s", "max_source_timestamp")]
        if row["timestamp_kind"] != TIME_KIND or not np.isfinite(fields).all():
            raise ValueError("geometry timestamp kind/finite check failed")
        if row["max_source_timestamp"] > row["available_at_s"] + 1e-9:
            raise ValueError("geometry uses future information")
        if previous is not None and (row["time_s"] <= previous["time_s"] or
                row["source_frame"] <= previous["source_frame"] or
                row["available_at_s"] < previous["available_at_s"] or
                row["camera"] != previous["camera"] or row["session"] != previous["session"]):
            raise ValueError("unordered/duplicate/mixed geometry stream")
        previous = row


def select_geometry(records, time_s, source_frame):
    """Latest arrived observation, including an unavailable latest observation."""
    if not math.isfinite(time_s):
        raise ValueError("invalid query time")
    index = bisect_right([r["time_s"] for r in records], float(time_s)) - 1
    while index >= 0 and (records[index]["source_frame"] > source_frame or
                         records[index]["available_at_s"] > time_s + 1e-9 or
                         records[index]["max_source_timestamp"] > time_s + 1e-9):
        index -= 1
    if index < 0:
        return dict(available=False, reason="no_arrived_geometry", age_s=None, record=None)
    record = records[index]
    age = float(time_s - record["time_s"])
    valid = bool(record["available"] and age <= CONFIG["max_geometry_age_s"] + 1e-9)
    reason = "ok" if valid else ("geometry_expired" if age > CONFIG["max_geometry_age_s"] + 1e-9 else "latest_geometry_unavailable")
    return dict(available=valid, reason=reason, age_s=age, record=record)


def line_frame(record, reference_xy, direction_hint):
    """Orthonormal frame at fixed R's projection onto an independent free line."""
    normal = np.asarray(record["normal_xy"], float)
    rho = float(record["rho_px"])
    reference = np.asarray(reference_xy, float)
    hint = np.asarray(direction_hint, float)
    segment = np.asarray(record["segment_xy"], float)
    width = float(record["width_px"])
    if (normal.shape != (2,) or reference.shape != (2,) or hint.shape != (2,) or segment.shape != (2, 2) or
            not np.isfinite(np.r_[normal, rho, reference, hint, segment.ravel(), width]).all() or
            np.linalg.norm(normal) < 1e-9 or np.linalg.norm(hint) < 1e-9 or width <= 0):
        raise ValueError("invalid independent line geometry")
    norm = np.linalg.norm(normal)
    normal, rho = normal / norm, rho / norm
    direction = np.array([normal[1], -normal[0]])
    if direction @ hint < 0:
        direction, normal, rho = -direction, -normal, -rho
    origin = reference + (rho - normal @ reference) * normal
    return dict(origin_xy=origin.tolist(), d_xy=direction.tolist(), n_xy=normal.tolist(),
                reference_xy=reference.tolist(), width_px=width, segment_xy=segment.tolist())


def patch_quality(gray, point_xy, box):
    """Current candidate-location appearance, not verified material visibility."""
    result = dict(available=False, reason="point_or_patch_outside_crop", saturation=None,
                  contrast=None, texture_available=False, visibility_verified=False)
    if point_xy is None:
        return result
    point = np.asarray(point_xy, float)
    if point.shape != (2,) or not np.isfinite(point).all():
        return result
    x, y = np.rint(point - np.asarray(box[:2])).astype(int)
    half = CONFIG["patch_half"]
    if not (half <= x < gray.shape[1] - half and half <= y < gray.shape[0] - half):
        return result
    patch = gray[y-half:y+half+1, x-half:x+half+1]
    saturation = float(np.mean(patch >= CONFIG["saturation_gray_threshold"]))
    contrast = float(np.percentile(patch, 95) - np.percentile(patch, 5))
    return dict(available=True, reason="current_candidate_patch_only", saturation=saturation,
                contrast=contrast, texture_available=bool(saturation <= CONFIG["patch_max_saturation"] and
                    contrast >= CONFIG["patch_min_contrast"]), visibility_verified=False)


def geometric_membership(point, frame):
    if point is None or frame is None:
        return dict(available=False, in_observed_free_band=False, material_membership_verified=False)
    p = np.asarray(point, float)
    if p.shape != (2,) or not np.isfinite(p).all():
        return dict(available=False, in_observed_free_band=False, material_membership_verified=False)
    d, n, origin = (np.asarray(frame[k]) for k in ("d_xy", "n_xy", "origin_xy"))
    segment = np.asarray(frame["segment_xy"])
    along, cross = float((p-origin) @ d), float((p-origin) @ n)
    bounds = sorted(((segment-origin) @ d).tolist())
    inside = bounds[0] <= along <= bounds[1] and abs(cross) <= CONFIG["free_band_half_widths"] * frame["width_px"]
    return dict(available=True, in_observed_free_band=bool(inside), line_coordinate_s_px=along,
                signed_line_distance_px=cross, material_membership_verified=False)


def motion_diagnostic(p0, p1, dt_s, initial_d, frame0=None, frame1=None):
    """Raw and conditional residual; units stay image pixels and receive px/s."""
    a, b, d0 = np.asarray(p0, float), np.asarray(p1, float), np.asarray(initial_d, float)
    if not np.isfinite(np.r_[a, b, d0, dt_s]).all() or dt_s <= 0:
        raise ValueError("invalid accepted displacement inputs")
    delta = b - a
    d = np.asarray(frame1["d_xy"]) if frame1 else d0
    n = np.array([-d[1], d[0]])
    result = dict(delta_xy_native_px=delta.tolist(), distance_px=float(np.linalg.norm(delta)),
        initial_tangent_px=float(delta @ d0), tangent_px=float(delta @ d), normal_px=float(delta @ n),
        apparent_tangent_px_s=float(delta @ d / dt_s), apparent_normal_px_s=float(delta @ n / dt_s),
        tangent_basis="current_or_past_independent_geometry" if frame1 else "initial_seed_fallback",
        compensated=None, optical_width_ratio=None, metric_status="metric_uncertain", metric_velocity_m_s=None)
    if frame0 is not None and frame1 is not None:
        q0, q1 = np.asarray(frame0["origin_xy"]), np.asarray(frame1["origin_xy"])
        mapped = q1 + ((a-q0) @ frame0["d_xy"]) * d + ((a-q0) @ frame0["n_xy"]) * n
        residual = b - mapped
        result["compensated"] = dict(delta_xy_native_px=residual.tolist(),
            tangent_px=float(residual @ d), normal_px=float(residual @ n),
            apparent_tangent_px_s=float(residual @ d / dt_s), apparent_normal_px_s=float(residual @ n / dt_s),
            independent_line_predicted_shift_xy=(mapped-a).tolist(),
            conditional_only=True, material_motion_confirmed=False)
        result["optical_width_ratio"] = float(frame1["width_px"] / frame0["width_px"])
    return result


def _stats(values):
    a = np.asarray([x for x in values if x is not None], float)
    if not len(a):
        return dict(n=0, median=None, mad=None, p05=None, p95=None)
    median = float(np.median(a))
    return dict(n=len(a), median=median, mad=float(np.median(abs(a-median))),
                p05=float(np.percentile(a, 5)), p95=float(np.percentile(a, 95)))


def _state_label(a, b):
    return "both" if a and b else "only_lk256" if a else "only_lk_native" if b else "neither"


def _json_xy(value):
    a = np.asarray(value, float)
    return a.tolist() if a.shape == (2,) and np.isfinite(a).all() else None


def _rows_by_id(rows, count):
    result = []
    previous_valid = None
    for row in rows:
        ids = row["point_ids"]
        if len(ids) != count or set(ids) != set(range(count)) or len(set(ids)) != count:
            raise ValueError("original seed ID denominator changed")
        if len(row["points_xy_native"]) != count or len(row["valid_candidate"]) != count:
            raise ValueError("point arrays changed size")
        values = {i: dict(xy=_json_xy(p), accepted=bool(v)) for i, p, v in zip(
            ids, row["points_xy_native"], row["valid_candidate"])}
        valid = {i for i, item in values.items() if item["accepted"]}
        if any(values[i]["xy"] is None for i in valid):
            raise ValueError("accepted nonfinite point")
        if previous_valid is not None and not valid <= previous_valid:
            raise ValueError("reidentification is forbidden")
        result.append(values)
        previous_valid = valid
    return result


def _source_issue(data, slot):
    if slot == 0:
        return None
    if data["source_frames"][slot] != data["source_frames"][slot-1] + 1:
        return "saved_source_frame_gap"
    if data["times"][slot] - data["times"][slot-1] > CONFIG["source_gap_s"]:
        return "receive_time_gap"
    if str(data["pixel_hashes"][slot]) == str(data["pixel_hashes"][slot-1]):
        return "duplicate_decoded_pixels"
    return None


def _failure_category(row, data, slot):
    issue = _source_issue(data, slot)
    if issue or row.get("rejection_reason") in ("frame_gap_exceeded", "non_increasing_input", "duplicate_decoded_pixels"):
        return "source_gap", issue or row["rejection_reason"]
    if row["age_s"] > CONFIG["ttl_s"] + 1e-9:
        return "ttl_expired", "prespecified_receive_time_age_limit"
    return "algorithm_rejected", row.get("rejection_reason") or "point_gate_details_not_recorded"


def _geometry_snapshot(selected):
    record = selected["record"]
    return dict(available=selected["available"], age_s=selected["age_s"], reason=selected["reason"],
                source_frame=record["source_frame"] if record else None,
                observation_time_s=record["time_s"] if record else None,
                cache_sha256=record.get("cache_sha256") if record else None)


def audit_group(meta, data, tracks_by_method, geometry_records):
    """Pure array/JSON audit. No annotation input or tracker call is accepted."""
    times, frames = np.asarray(data["times"], float), np.asarray(data["source_frames"])
    gray = np.asarray(data["gray_native"])
    if data.get("timestamp_kind", TIME_KIND) != TIME_KIND:
        raise ValueError("source timebase must be SDK receive, not nominal/exposure/device ticks")
    if (not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or
            len(frames) != len(times) or np.any(np.diff(frames) <= 0) or
            len(data["pixel_hashes"]) != len(times) or len(gray) != len(times) or gray.dtype != np.uint8):
        raise ValueError("missing, unordered, duplicate, or invalid source schedule")
    _validate_geometry_stream(geometry_records)
    session, side, group = SESSIONS[meta["side"]], meta["side"], meta["group_id"]
    if any(g["session"] != session or g["camera"] != side for g in geometry_records):
        raise ValueError("geometry from another video is forbidden")
    seeds = meta["seeds_native"] if meta["available"] else []
    count = len(seeds)
    indices = {}
    for method in METHODS:
        output = tracks_by_method[method]
        if output["group_id"] != group or output["side"] != side or output["method"] != method or output["seeds_native"] != seeds or output["box"] != meta["box"]:
            raise ValueError("paired group/seed/source coordinate contract mismatch")
        rows = output["rows"]
        if len(rows) != len(times) or [r["source_frame"] for r in rows] != frames.tolist():
            raise ValueError("missing or reordered tracker frame")
        for slot, row in enumerate(rows):
            if (row.get("source_frame_basis") != "video_frame_index_not_camera_counter" or
                    row.get("frame_slot_index") != slot or abs(row["time_s"]-times[slot]) > 1e-9 or
                    abs(row["age_s"]-(times[slot]-times[0])) > 1e-9 or
                    row["max_source_timestamp"] > row["t_emit"] + 1e-9 or row["max_source_timestamp"] > times[slot] + 1e-9):
                raise ValueError("tracker timestamp/source basis mismatch")
        indices[method] = _rows_by_id(rows, count)
        if count and (not all(v["accepted"] for v in indices[method][0].values()) or
                not np.allclose([indices[method][0][i]["xy"] for i in range(count)], seeds, atol=1e-3, rtol=0)):
            raise ValueError("initial native coordinates do not match shared seeds")
    seed = np.asarray(meta.get("free_seed") or [[0, 0], [1, 0]], float)
    initial_d = seed[-1]-seed[0]
    if not np.isfinite(initial_d).all() or np.linalg.norm(initial_d) < 1e-9:
        raise ValueError("invalid initial direction")
    initial_d /= np.linalg.norm(initial_d)
    box = meta["box"]
    reference = np.asarray(box[:2], float) + (np.asarray(box[2:], float)-1)/2
    geometries, line_frames = [], []
    for timestamp, frame in zip(times, frames):
        selected = select_geometry(geometry_records, float(timestamp), int(frame))
        geometries.append(selected)
        line_frames.append(line_frame(selected["record"], reference, initial_d) if selected["available"] else None)
    states, pairs, failures = [], [], []
    for slot, (timestamp, frame) in enumerate(zip(times, frames)):
        for point_id, seed_xy in enumerate(seeds):
            methods = {}
            for method in METHODS:
                item = indices[method][slot][point_id]
                methods[method] = dict(**item,
                    patch_quality=patch_quality(gray[slot], item["xy"] if item["accepted"] else None, box),
                    geometric_membership=geometric_membership(item["xy"] if item["accepted"] else None, line_frames[slot]),
                    saved_row_reason=tracks_by_method[method]["rows"][slot].get("rejection_reason"))
            states.append(dict(session=session, camera=side, group_id=group, point_id=point_id,
                frame_slot=slot, source_frame=int(frame), time_s=float(timestamp), timestamp_kind=TIME_KIND,
                initial_seed_xy=seed_xy, state=_state_label(methods[METHODS[0]]["accepted"], methods[METHODS[1]]["accepted"]),
                geometry=_geometry_snapshot(geometries[slot]), methods=methods,
                material_identity_confirmed=False, visibility_verified=False))
    state_lookup = {(r["frame_slot"], r["point_id"]): r for r in states}
    for slot in range(1, len(times)):
        dt = float(times[slot]-times[slot-1])
        for point_id, seed_xy in enumerate(seeds):
            methods = {}
            for method in METHODS:
                before, after = (indices[method][i][point_id] for i in (slot-1, slot))
                accepted = before["accepted"] and after["accepted"]
                motion = motion_diagnostic(before["xy"], after["xy"], dt, initial_d,
                    line_frames[slot-1], line_frames[slot]) if accepted else None
                condition = accepted and all(
                    state_lookup[(i, point_id)]["methods"][method]["patch_quality"]["texture_available"] and
                    state_lookup[(i, point_id)]["methods"][method]["geometric_membership"]["in_observed_free_band"]
                    for i in (slot-1, slot))
                if motion is not None and motion["compensated"] is not None:
                    motion["compensated"]["geometric_and_patch_condition"] = bool(condition)
                methods[method] = dict(accepted_pair=bool(accepted), accepted0=before["accepted"],
                    accepted1=after["accepted"], point_xy0=before["xy"], point_xy1=after["xy"],
                    motion=motion, geometric_and_patch_condition=bool(condition),
                    condition_is_not_verified_material=True)
            state = _state_label(*(methods[m]["accepted_pair"] for m in METHODS))
            difference = None
            if state == "both":
                a, b = (methods[m]["motion"] for m in METHODS)
                difference = dict(delta_xy_native_px=(np.asarray(a["delta_xy_native_px"])-b["delta_xy_native_px"]).tolist(),
                    initial_tangent_px=a["initial_tangent_px"]-b["initial_tangent_px"],
                    tangent_px=a["tangent_px"]-b["tangent_px"], normal_px=a["normal_px"]-b["normal_px"],
                    compensated_tangent_px=(a["compensated"]["tangent_px"]-b["compensated"]["tangent_px"]
                        if a["compensated"] and b["compensated"] else None),
                    sign="lk256_minus_lk_native_same_ID_same_frame_pair")
            pair = dict(session=session, camera=side, group_id=group, point_id=point_id, initial_seed_xy=seed_xy,
                source_frame0=int(frames[slot-1]), source_frame1=int(frames[slot]),
                frame_pair=[int(frames[slot-1]), int(frames[slot])], frame_slot0=slot-1, frame_slot1=slot,
                time_s0=float(times[slot-1]), time_s1=float(times[slot]), dt_s=dt, timestamp_kind=TIME_KIND,
                state=state, source_issue=_source_issue(data, slot),
                geometry0=_geometry_snapshot(geometries[slot-1]), geometry1=_geometry_snapshot(geometries[slot]),
                reference_xy=reference.tolist(), initial_d_xy=initial_d.tolist(), methods=methods,
                paired_difference=difference, P_texture_px=None, texture_join_pending=True,
                metric_velocity_m_s=None, material_identity_confirmed=False,
                t_available=None, actual_live_t_available=None,
                source_available_lower_bound_s=float(times[slot]),
                availability_clock=CONFIG["result_availability_clock"],
                max_source_timestamp=float(times[slot]))
            pairs.append(pair)
    for method in METHODS:
        for point_id in range(count):
            valid = [indices[method][slot][point_id]["accepted"] for slot in range(len(times))]
            lost = next((i for i, good in enumerate(valid) if not good), None)
            category, detail = ("unknown", "right_censored_at_last_scheduled_frame") if lost is None else _failure_category(
                tracks_by_method[method]["rows"][lost], data, lost)
            last_accepted = (len(times)-1) if lost is None else max(0, lost-1)
            failures.append(dict(session=session, camera=side, group_id=group, point_id=point_id, method=method,
                first_invalid_slot=lost, first_invalid_age_s=float(times[lost]-times[0]) if lost is not None else None,
                last_accepted_age_s=float(times[last_accepted]-times[0]), category=category, detail=detail,
                right_censored=lost is None, actual_observable_duration_s=None,
                duration_semantics="saved candidate accepted duration only; actual material visibility unverified",
                physical_cause="unknown", crop_exit_observed=False, free_segment_exit_observed=False,
                occluded_verified=False, final_row_reason=tracks_by_method[method]["rows"][-1].get("rejection_reason")))
    summary = summarize(pairs, failures)
    summary.update(group_id=group, camera=side, session=session, initialized_points=count,
                   seed_available=meta["available"], seed_reason=meta["reason"], scheduled_frames=len(times),
                   scheduled_frame_pairs=len(times)-1, original_seed_frame_denominator=count*len(times))
    return dict(seed_states=states, pairs=pairs, failures=failures, summary=summary)


def summarize(pairs, failures):
    both = [p for p in pairs if p["state"] == "both"]
    conditions = [p for p in both if all(p["methods"][m]["geometric_and_patch_condition"] for m in METHODS)]
    return dict(original_seed_pair_denominator=len(pairs), states=dict(Counter(p["state"] for p in pairs)),
        both_accepted_pairs=len(both), both_with_geometric_patch_condition=len(conditions),
        paired_difference_initial_tangent_px=_stats([p["paired_difference"]["initial_tangent_px"] for p in both]),
        paired_difference_current_tangent_px=_stats([p["paired_difference"]["tangent_px"] for p in both]),
        paired_difference_compensated_tangent_px=_stats([p["paired_difference"]["compensated_tangent_px"] for p in both]),
        condition_subset_paired_difference_compensated_tangent_px=_stats([
            p["paired_difference"]["compensated_tangent_px"] for p in conditions]),
        same_common_ID_method_motion={m: dict(initial_tangent_px=_stats([
            p["methods"][m]["motion"]["initial_tangent_px"] for p in both]),
            current_tangent_px=_stats([p["methods"][m]["motion"]["tangent_px"] for p in both])) for m in METHODS},
        failure_first_cause={m: dict(Counter(f["category"] for f in failures if f["method"] == m and not f["right_censored"])) for m in METHODS},
        right_censored_points={m: sum(f["right_censored"] for f in failures if f["method"] == m) for m in METHODS},
        selection_bias="common accepted IDs are selected by both methods; repeated frames/overlapping cohorts are correlated",
        identity_accuracy=None, physical_motion_evaluation="not_evaluable")


def _jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _csv(path, pairs):
    fields = CONFIG["pair_key"] + ["timestamp_kind", "time_s0", "time_s1", "dt_s", "state",
        "lk256_accepted0", "lk256_accepted1", "lk_native_accepted0", "lk_native_accepted1",
        "lk256_raw_tangent_px", "lk_native_raw_tangent_px", "paired_raw_tangent_difference_px",
        "paired_compensated_tangent_difference_px", "geometry_age0_s", "geometry_age1_s", "detail_json"]
    with Path(path).open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for pair in pairs:
            row = {k: pair[k] for k in fields if k in pair}
            for method in METHODS:
                item = pair["methods"][method]
                row[method+"_accepted0"], row[method+"_accepted1"] = item["accepted0"], item["accepted1"]
                row[method+"_raw_tangent_px"] = item["motion"]["tangent_px"] if item["motion"] else None
            diff = pair["paired_difference"]
            row.update(paired_raw_tangent_difference_px=diff["tangent_px"] if diff else None,
                       paired_compensated_tangent_difference_px=diff["compensated_tangent_px"] if diff else None,
                       geometry_age0_s=pair["geometry0"]["age_s"], geometry_age1_s=pair["geometry1"]["age_s"],
                       detail_json=json.dumps(pair, ensure_ascii=False, allow_nan=False))
            writer.writerow(row)


def run(v26_base, outdir, protocol_sha256, expected_input_sha256):
    """Run once only after the parent freezes exact source/config/input hashes."""
    base, outdir = Path(v26_base).resolve(), Path(outdir).resolve()
    if outdir.exists():
        raise FileExistsError("audit refuses overwrite")
    if len(protocol_sha256) != 64 or expected_input_sha256 is None:
        raise ValueError("V27 frozen protocol and expected input hashes required")
    hashes = {}
    for key, path in input_paths(base).items():
        _record(path, hashes, expected_input_sha256)
    protocol, prepare = _read(base / "protocol.json"), _read(base / "prepare_manifest.json")
    bmanifest, breport = _read(base / "B/manifest.json"), _read(base / "B/report.json")
    old_sha = sha(base / "protocol.json")
    if any(p["protocol_sha256"] != old_sha for p in (prepare, bmanifest, breport)):
        raise ValueError("V26 provenance mismatch")
    if (protocol.get("schema") != "v26_causal_material_1" or
            bmanifest["config"]["shared_config"]["max_age_s"] != CONFIG["ttl_s"] or
            bmanifest["config"]["shared_config"]["max_frame_gap_s"] != CONFIG["source_gap_s"] or
            bmanifest["methods"] != list(METHODS)):
        raise ValueError("saved V26 configuration mismatch")
    sources = {name: sha(ROOT / name) for name in SOURCE_FILES}
    geometries = {side: load_geometry_records(base, side, input_hashes=hashes,
        expected_input_sha256=expected_input_sha256) for side in CONFIG["sides"]}
    all_states, all_pairs, all_failures, group_summaries = [], [], [], []
    for side in CONFIG["sides"]:
        for start in CONFIG["starts"]:
            group = f"{side}_{start:06d}"
            directory = f"cohorts/{group}"
            meta = _read(base / directory / "meta.json")
            with np.load(base / directory / "data.npz", allow_pickle=False) as archive:
                data = {k: archive[k] for k in ("gray_native", "times", "source_frames", "pixel_hashes")}
            if len(data["times"]) != 21 or not np.array_equal(data["source_frames"], np.arange(start, start+21)):
                raise ValueError("changed frozen cohort schedule")
            for name in ("meta.json", "data.npz"):
                key = f"{directory}/{name}"
                digest = hashes[_key(base / key)]
                if any(receipt[field].get(key) != digest for receipt, field in
                       ((prepare, "input_hashes"), (bmanifest, "input_sha256"), (breport, "input_sha256"))):
                    raise ValueError("V26 prepared input receipt mismatch")
            tracks = {m: _read(base / f"B/{m}/{group}.json") for m in METHODS}
            for track in tracks.values():
                if track["protocol_sha256"] != old_sha or track["input_sha256"] != {
                        name: hashes[_key(base / directory / name)] for name in ("meta.json", "data.npz")}:
                    raise ValueError("trajectory input hash mismatch")
            result = audit_group(meta, data, tracks, geometries[side])
            all_states.extend(result["seed_states"])
            all_pairs.extend(result["pairs"])
            all_failures.extend(result["failures"])
            group_summaries.append(result["summary"])
    if len(group_summaries) != 60 or sum(s["initialized_points"] for s in group_summaries) != 556:
        raise ValueError("V26 original 60-cohort/556-seed denominator changed")
    outdir.mkdir(parents=True, exist_ok=False)
    _write(outdir / "manifest.json", dict(config=CONFIG, protocol_sha256=protocol_sha256,
        v26_protocol_sha256=old_sha, input_sha256=hashes, source_sha256=sources,
        label_files_read=[], hardware_access=False, tracker_reexecuted=False))
    _jsonl(outdir / "seed_states.jsonl", all_states)
    _jsonl(outdir / "paired_points.jsonl", all_pairs)
    _csv(outdir / "paired_points.csv", all_pairs)
    _write(outdir / "failure_censoring.json", dict(rows=all_failures,
        unavailable_categories=["crop_exit_observed", "free_segment_exit_observed", "occluded_verified"],
        reason="no independent image evidence; these physical causes must remain unknown"))
    detailed = [p for p in all_pairs if p["group_id"] == "left_000240" and p["frame_pair"] == CONFIG["detail_frame_pair"]]
    _write(outdir / "left240_frame241_242.json", dict(pairs=detailed, summary=summarize(detailed, []),
        warning="common point IDs only; not the difference of separate surviving-set medians"))
    for group in CONFIG["example_groups"]:
        subset = [p for p in all_pairs if p["group_id"] == group]
        _csv(outdir / (group + "_paired_points.csv"), subset)
    _write(outdir / "summary.json", dict(config=CONFIG, all_cohorts=summarize(all_pairs, all_failures),
        cohorts=group_summaries, fixed_cohort_denominator=60, initialized_seed_denominator=556,
        scheduled_frame_denominator=1260, scheduled_continuation_frame_denominator=1200,
        original_seed_frame_denominator=len(all_states), original_seed_pair_denominator=len(all_pairs),
        by_camera={s: summarize([p for p in all_pairs if p["camera"] == s],
                               [f for f in all_failures if f["camera"] == s]) for s in CONFIG["sides"]},
        geometry_condition_is_not_material_confirmation=True, metric_velocity_m_s=None,
        no_fitted_point_field=True, old_v26_statistics_unchanged=True, deployment_permitted=False))
    if any(sha(ROOT / key) != value for key, value in hashes.items()) or any(sha(ROOT / key) != value for key, value in sources.items()):
        raise RuntimeError("frozen input/source changed during audit")
    outputs = {p.name: sha(p) for p in outdir.iterdir() if p.is_file()}
    _write(outdir / "output_receipt.json", dict(input_sha256=hashes, source_sha256=sources,
        output_sha256=outputs, self_hash_excluded=True, completed=True))
    return _read(outdir / "summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v26-base", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--expected-input-sha256", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.v26_base, args.outdir, args.protocol_sha256, _read(args.expected_input_sha256))
    print(json.dumps(result["all_cohorts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
