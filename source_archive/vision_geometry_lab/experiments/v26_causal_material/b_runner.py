"""V26 B-only fixed-cohort replay. Point persistence is NOT rope identity.

Only the frozen 60 prepared meta.json/data.npz pairs are experimental inputs.
Seed failures keep all 21 scheduled frame slots. No GT, original video, device,
reidentification, historical trajectory line or event label is consumed here.
TAPNext is an optional explicit method: missing/unverified weights fail before
creating results; it is never silently replaced by LK or a synthetic model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .common import CONFIG as SHARED_CONFIG, ROOT, map_points, stream_guard
from .point_tracks import track_lk


SOURCE_FILES = (
    "vision_geometry_lab/experiments/v26_causal_material/b_runner.py",
    "vision_geometry_lab/experiments/v26_causal_material/test_b_runner.py",
    "vision_geometry_lab/experiments/v26_causal_material/common.py",
    "vision_geometry_lab/experiments/v26_causal_material/point_tracks.py",
    "vision_geometry_lab/experiments/v26_causal_material/test_point_tracks.py",
    "vision_geometry_lab/experiments/v26_causal_material/tapnext_adapter.py",
)
CONFIG = {
    "maximum_processing_s": 600,
    "schema": "v26_route_b_runner_1", "shared_config": copy.deepcopy(SHARED_CONFIG),
    "default_methods": ["lk256", "lk_native"], "optional_method": "tapnext",
    "sides": ["left", "right"], "cohort_start_indices": list(SHARED_CONFIG["cohort_start_indices"]),
    "cohort_frames": SHARED_CONFIG["cohort_frames"],
    "fixed_overlay_start_indices": [0, 100, 200, 300, 400, 500],
    "tapnext_visibility_threshold": .5, "tapnext_point_loss_absorbing": True,
    "seed_failures": "retain_all_scheduled_frame_slots_no_tracker_call",
    "point_denominator": "actual_initialized_points_only_explicitly_conditional_on_seed_success",
    "deformation": "current_corresponding_points_vs_seed_similarity_residual_and_pair_distance_ratios",
    "overlay": "last_frame_current_valid_points_and_ids_only_no_history_lines",
    "material_identity_evaluation": "not_evaluable_without_independent_identity_labels",
    "physical_t1_t2_evaluation": "not_evaluable", "deployment_permitted": False,
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def _write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _stats(values):
    values = [v for value in values if (v := _number(value)) is not None]
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    data = np.asarray(values)
    return {"count": len(values), "mean": float(data.mean()), "p50": float(np.percentile(data, 50)),
            "p95": float(np.percentile(data, 95)), "max": float(data.max())}


def point_deformation(seeds, points, valid):
    """Same-frame geometric consistency only; no historical curve is built."""
    seed = np.asarray(seeds, dtype=float).reshape(-1, 2)
    current = np.asarray(points, dtype=float).reshape(-1, 2)
    valid = np.asarray(valid, dtype=bool)
    if len(seed) != len(current) or len(valid) != len(seed):
        raise ValueError("point correspondence count mismatch")
    keep = valid & np.isfinite(seed).all(1) & np.isfinite(current).all(1)
    a, b = seed[keep], current[keep]
    result = {"status": "insufficient_current_points", "current_points": int(keep.sum()),
              "similarity_residual_rms_native_px": None, "median_pair_distance_ratio": None,
              "pair_distance_ratio_mad": None,
              "semantics": "diagnostic_not_material_identity_or_visibility_confirmation"}
    if len(a) < 3:
        return result
    design = np.zeros((2 * len(a), 4), dtype=float)
    design[0::2, 0], design[0::2, 1], design[0::2, 2] = a[:, 0], -a[:, 1], 1.
    design[1::2, 0], design[1::2, 1], design[1::2, 3] = a[:, 1], a[:, 0], 1.
    coefficients, _, rank, _ = np.linalg.lstsq(design, b.ravel(), rcond=None)
    ratios = []
    for i in range(len(a) - 1):
        for j in range(i + 1, len(a)):
            distance = float(np.linalg.norm(a[i] - a[j]))
            if distance > 1e-9:
                ratios.append(float(np.linalg.norm(b[i] - b[j]) / distance))
    if rank < 4 or not ratios:
        result["status"] = "degenerate_seed_geometry"
        return result
    median = float(np.median(ratios))
    residual = (design @ coefficients).reshape(-1, 2) - b
    result.update(status="diagnostic_only", similarity_residual_rms_native_px=float(
        np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))), median_pair_distance_ratio=median,
        pair_distance_ratio_mad=float(np.median(np.abs(np.asarray(ratios) - median))))
    return result


def _invalid_row(t, source, age, count, reason, *, status="invalid"):
    return {"time_s": float(t), "t_emit": float(t), "max_source_timestamp": float(t),
            "source_frame": int(source), "age_s": float(age),
            "points_xy_native": [[None, None] for _ in range(count)],
            "valid_candidate": [False] * count, "rejection_reason": str(reason),
            "status": status, "visibility": "unknown_not_confirmed",
            "material_identity_confirmed": False, "current_new_rope_membership": "unknown",
            "processing_ms": 0., "waiting_future_frames": 0}


def _tap_rows(adapter, data, seeds, box, group_id):
    """Same guard/age/loss contract as LK; never revive a lost point identity."""
    points = map_points(seeds, box, 256)
    active = np.ones(len(points), bool)
    previous = None
    stopped = None
    rows = []
    for index, (frame, timestamp, source, pixel_hash) in enumerate(zip(
            data["frames256"], data["times"], data["source_frames"], data["pixel_hashes"])):
        start = time.perf_counter()
        timestamp, source = float(timestamp), int(source)
        age = timestamp - float(data["times"][0])
        reason = stream_guard(previous, timestamp, source, str(pixel_hash))
        previous = {"time_s": timestamp, "source_frame": source, "pixel_hash": str(pixel_hash)}
        if reason or age > SHARED_CONFIG["max_age_s"] + 1e-9:
            stopped = stopped or reason or "cohort_age_limit"
            active[:] = False
            adapter.reset()
        if stopped:
            rows.append(_invalid_row(timestamp, source, age, len(points), stopped))
            continue
        if index == 0:
            raw = adapter.start(frame, points, timestamp, group_id, source)
        else:
            raw = adapter.update(frame, timestamp, group_id, source)
        predicted = np.asarray(raw["points_xy"], dtype=float).reshape(-1, 2)
        visibility = np.asarray(raw["visibility_candidate"], dtype=bool)
        if len(predicted) != len(points) or len(visibility) != len(points):
            raise ValueError("TAPNext output point count changed")
        finite = np.isfinite(predicted).all(1)
        inside = finite & (predicted >= 0).all(1) & (predicted <= 255).all(1)
        active &= visibility & inside
        native = map_points(predicted, box, 256, to_native=True)
        row = _invalid_row(timestamp, source, age, len(points), "all_points_rejected")
        row.update(points_xy_native=[[_number(v) for v in point] for point in native],
                   valid_candidate=active.tolist(), status="candidate" if np.any(active) else "invalid",
                   rejection_reason=None if np.any(active) else "all_points_rejected",
                   processing_ms=1000 * (time.perf_counter() - start),
                   model_inference_ms=float(raw["inference_ms"]),
                   model_visibility_probability=[_number(v) for v in raw["visible_probability"]],
                   model_visibility_candidate=visibility.tolist(), model_status=raw["status"],
                   gpu_peak_allocated_bytes=raw.get("gpu_peak_allocated_bytes"),
                   max_source_timestamp=float(raw["max_source_timestamp"]))
        rows.append(row)
        if not np.any(active):
            stopped = "all_points_rejected_absorbing"
            adapter.reset()
    return rows


def replay_cohort(meta, data, method, *, adapter=None):
    """No labels in this interface; callers provide the current fixed crop."""
    count = len(data["times"])
    seeds = meta["seeds_native"] if meta["available"] else []
    start = time.perf_counter()
    if not meta["available"]:
        rows = [_invalid_row(t, s, t - data["times"][0], 0, meta["reason"], status="unknown")
                for t, s in zip(data["times"], data["source_frames"])]
    elif method in ("lk256", "lk_native"):
        frame_key = "frames256" if method == "lk256" else "gray_native"
        rows = track_lk(data[frame_key], data["times"], data["source_frames"], data["pixel_hashes"],
                        seeds, meta["box"], method)
    elif method == "tapnext":
        if adapter is None:
            raise ValueError("TAPNext requires the official initialized adapter")
        rows = _tap_rows(adapter, data, seeds, meta["box"], meta["group_id"])
    else:
        raise ValueError(f"unknown method: {method}")
    tracking_wall_ms = 1000 * (time.perf_counter() - start)
    if len(rows) != count:
        raise RuntimeError("tracker silently changed the scheduled frame denominator")
    for index, row in enumerate(rows):
        if len(row["valid_candidate"]) != len(seeds) or len(row["points_xy_native"]) != len(seeds):
            raise ValueError("tracker silently changed point count")
        if float(row["max_source_timestamp"]) > float(row["t_emit"]) + 1e-9:
            raise ValueError("tracker consumed a future source frame")
        native = [[_number(value) for value in point] for point in row["points_xy_native"]]
        for point, valid in zip(native, row["valid_candidate"]):
            if valid and any(value is None for value in point):
                raise ValueError("nonfinite point cannot be a valid candidate")
        row["points_xy_native"] = native
        row.update(group_id=meta["group_id"], point_ids=list(range(len(seeds))),
                   source_frame_basis="video_frame_index_not_camera_counter",
                   frame_slot_index=index, seed_available=meta["available"])
        row["deformation"] = point_deformation(seeds, native, row["valid_candidate"])
        if not any(row["valid_candidate"]) and row.get("rejection_reason") is None:
            row["rejection_reason"] = "previously_invalid_absorbing"
    current_counts = [sum(row["valid_candidate"]) for row in rows]
    min_points = SHARED_CONFIG["min_points"]
    summary = {
        "scheduled_frames": count, "scheduled_continuation_frames": max(0, count - 1),
        "seed_available": bool(meta["available"]), "initialized_points": len(seeds),
        "frames_with_any_current_candidate": sum(n > 0 for n in current_counts),
        "continuation_frames_with_min_points": sum(n >= min_points for n in current_counts[1:]),
        "last_frame_has_min_points": bool(current_counts[-1] >= min_points),
        "point_frame_opportunities_conditional_on_seed": count * len(seeds),
        "current_point_frame_candidates": sum(current_counts),
        "continuation_point_opportunities_conditional_on_seed": max(0, count - 1) * len(seeds),
        "continuation_point_candidates": sum(current_counts[1:]),
        "frame_rejection_reason_counts": dict(Counter(row["rejection_reason"] for row in rows if row.get("rejection_reason"))),
        "tracker_per_frame_ms": _stats(row["processing_ms"] for row in rows),
        "tracking_call_wall_ms": tracking_wall_ms,
        "identity_accuracy": None, "identity_evaluation": "not_evaluable",
    }
    return {"group_id": meta["group_id"], "side": meta["side"], "start_index": meta["start_index"],
            "available": meta["available"], "reason": meta["reason"], "box": meta["box"],
            "seeds_native": seeds, "method": method, "rows": rows, "summary": summary,
            "material_identity_confirmed": False, "deployment_permitted": False}


def summarize_method(groups):
    n = len(groups)
    summaries = [g["summary"] for g in groups]
    keys = ("scheduled_frames", "scheduled_continuation_frames", "initialized_points",
            "frames_with_any_current_candidate", "continuation_frames_with_min_points",
            "point_frame_opportunities_conditional_on_seed", "current_point_frame_candidates",
            "continuation_point_opportunities_conditional_on_seed", "continuation_point_candidates")
    totals = {key: sum(s[key] for s in summaries) for key in keys}
    divide = lambda a, b: a / b if b else None
    deformations = [row["deformation"] for group in groups for row in group["rows"][1:]]
    return {
        "scheduled_cohort_denominator": n, "seed_available_cohorts": sum(g["available"] for g in groups),
        "seed_failure_cohorts": sum(not g["available"] for g in groups), **totals,
        "frame_candidate_fraction_fixed_denominator": divide(totals["frames_with_any_current_candidate"], totals["scheduled_frames"]),
        "continuation_min_points_fraction_fixed_denominator": divide(totals["continuation_frames_with_min_points"], totals["scheduled_continuation_frames"]),
        "last_frame_min_points_cohorts": sum(s["last_frame_has_min_points"] for s in summaries),
        "last_frame_min_points_fraction_fixed_cohorts": divide(sum(s["last_frame_has_min_points"] for s in summaries), n),
        "point_survival_fraction_conditional_on_seed": divide(totals["current_point_frame_candidates"], totals["point_frame_opportunities_conditional_on_seed"]),
        "continuation_point_survival_fraction_conditional_on_seed": divide(totals["continuation_point_candidates"], totals["continuation_point_opportunities_conditional_on_seed"]),
        "fraction_semantics": "descriptive_candidate_persistence_not_identity_accuracy",
        "seed_failure_reason_counts": dict(Counter(g["reason"] for g in groups if not g["available"])),
        "frame_rejection_reason_counts": dict(sum((Counter(s["frame_rejection_reason_counts"]) for s in summaries), Counter())),
        "tracking_per_frame_ms_including_inactive_rows": _stats(row["processing_ms"] for g in groups for row in g["rows"]),
        "tracking_group_wall_ms": _stats(s["tracking_call_wall_ms"] for s in summaries),
        "similarity_residual_rms_native_px": _stats(d["similarity_residual_rms_native_px"] for d in deformations),
        "median_pair_distance_ratio": _stats(d["median_pair_distance_ratio"] for d in deformations),
        "pair_distance_ratio_mad": _stats(d["pair_distance_ratio_mad"] for d in deformations),
        "deformation_semantics": "same_frame_vs_seed_geometry_camera_projection_and_selection_sensitive_not_identity",
        "material_identity_accuracy": None, "identity_evaluation": "not_evaluable",
        "physical_t1": "not_evaluable", "physical_t2": "not_evaluable",
        "production_end_to_end_latency_ms": None,
    }


def _font(size):
    path = Path("C:/Windows/Fonts/msyh.ttc")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default(size=size)


def drawable_points(row, box):
    result = []
    for index, (point, valid) in enumerate(zip(row["points_xy_native"], row["valid_candidate"])):
        if valid and all(_number(v) is not None for v in point):
            xy = map_points([point], box, 256)[0]
            if np.isfinite(xy).all() and np.all(xy >= 0) and np.all(xy <= 255):
                result.append((index, float(xy[0]), float(xy[1])))
    return result


def overlay_current(frame_bgr, row, box, *, available):
    canvas = Image.fromarray(np.asarray(frame_bgr)[..., ::-1].copy(), "RGB")
    draw = ImageDraw.Draw(canvas)
    for index, x, y in drawable_points(row, box):
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline="#ff45ce", width=1)
        draw.text((min(240, x + 4), max(0, y - 10)), str(index), font=_font(12), fill="#ffeb76", stroke_width=1, stroke_fill="black")
    if not available:
        draw.rectangle((0, 0, 255, 22), fill="black")
        draw.text((5, 2), "自动种子不可用 / 无预测", font=_font(13), fill="#ffac80")
    return canvas


def _contact_sheet(outdir, side, methods, tiles):
    starts = CONFIG["fixed_overlay_start_indices"]
    canvas = Image.new("RGB", (1720, 125 + 325 * len(methods)), "#141f2e")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 15), f"V26 B | {side} | 固定六组末帧：当前点与ID，无历史线", font=_font(27), fill="white")
    draw.text((20, 58), "品红圈=当前候选；不确认可见性/整绳/新材料身份。原图坐标按像素中心映射到同一256输入。", font=_font(20), fill="#e4cba0")
    for row_index, method in enumerate(methods):
        top = 115 + row_index * 325
        for col, start in enumerate(starts):
            item = tiles[(side, start, method)]
            left = 20 + col * 283
            draw.text((left, top), f"{method} / 起帧 {start}", font=_font(16), fill="#9bd8ef")
            canvas.paste(item["image"], (left, top + 29))
            draw.text((left, top + 290), f"末帧 {item['source_frame']} | 当前 {item['count']} 点", font=_font(15), fill="#d2dce8")
    with (outdir / f"{side}_fixed_last_frames.png").open("xb") as handle:
        canvas.save(handle, format="PNG")


def _validate_meta(meta, side, start):
    if meta.get("side") != side or meta.get("start_index") != start or not meta.get("group_id"):
        raise ValueError("cohort identity mismatch")
    if not isinstance(meta.get("available"), bool) or not isinstance(meta.get("reason"), str):
        raise ValueError("cohort availability/reason must be explicit")
    box = np.asarray(meta.get("box"), dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or list(box[2:]) != [1024., 1024.]:
        raise ValueError("frozen 1024 square crop required")
    if np.any(box[:2] < 0) or box[0] + 1024 > 4096 or box[1] + 1024 > 3000:
        raise ValueError("crop outside source image")
    seed = np.asarray(meta.get("seeds_native"), dtype=float)
    if meta["available"] and (seed.ndim != 2 or seed.shape[1] != 2 or
            not SHARED_CONFIG["min_points"] <= len(seed) <= SHARED_CONFIG["max_points"] or not np.isfinite(seed).all()):
        raise ValueError("available cohort has invalid automatic seeds")


def _load_data(path, start):
    with np.load(path, allow_pickle=False) as loaded:
        required = ("frames256", "gray_native", "times", "source_frames", "pixel_hashes")
        if any(key not in loaded.files for key in required):
            raise ValueError("missing prepared cohort array")
        data = {key: loaded[key] for key in required}
    n = CONFIG["cohort_frames"]
    if (data["frames256"].shape != (n, 256, 256, 3) or data["frames256"].dtype != np.uint8 or
            data["gray_native"].shape != (n, 1024, 1024) or data["gray_native"].dtype != np.uint8):
        raise ValueError("unmatched frozen cohort image shape/dtype")
    if any(data[key].shape != (n,) for key in ("times", "source_frames", "pixel_hashes")):
        raise ValueError("cohort metadata length mismatch")
    if not np.isfinite(data["times"]).all() or not np.array_equal(data["source_frames"], np.arange(start, start + n)):
        raise ValueError("invalid cohort times/video frame identities")
    return data


def run(base, outdir, protocol_sha256, methods=("lk256", "lk_native"), expected_input_sha256=None,
        *, weights_path=None, expected_weights_sha256=None, device="cuda", vendor_path=None):
    base, outdir = Path(base).resolve(), Path(outdir).resolve()
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods) or any(m not in ("lk256", "lk_native", "tapnext") for m in methods):
        raise ValueError("explicit unique LK or TAPNext methods required")
    if len(protocol_sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in protocol_sha256):
        raise ValueError("invalid protocol SHA256")
    if outdir.exists():
        raise FileExistsError(f"refusing to overwrite {outdir}")
    adapter, initialization_ms = None, 0.
    if "tapnext" in methods:
        from .tapnext_adapter import TapNextAdapter
        weight = Path(weights_path) if weights_path else Path(__file__).parent / "weights/tapnextpp_verified.pt"
        if not weight.is_file():
            raise FileNotFoundError(f"official TAPNext weights unavailable: {weight}")
        if expected_weights_sha256 is None:
            raise ValueError("TAPNext requires an explicitly frozen weight SHA256")
        began = time.perf_counter()
        adapter = TapNextAdapter(device=device, weights_path=weight, vendor_path=vendor_path,
                                 expected_sha256=expected_weights_sha256,
                                 max_gap_s=SHARED_CONFIG["max_frame_gap_s"], max_age_s=SHARED_CONFIG["max_age_s"])
        initialization_ms = 1000 * (time.perf_counter() - began)
    inputs, cohorts, seen = {}, [], set()
    for side in CONFIG["sides"]:
        for start in CONFIG["cohort_start_indices"]:
            relative = f"cohorts/{side}_{start:06d}"
            for name in ("meta.json", "data.npz"):
                key = f"{relative}/{name}"
                inputs[key] = sha256(base / key)
                if expected_input_sha256 is not None and inputs[key] != expected_input_sha256.get(key):
                    raise ValueError(f"prepared input hash mismatch: {key}")
            meta = json.loads((base / relative / "meta.json").read_text(encoding="utf-8-sig"))
            _validate_meta(meta, side, start)
            if meta["group_id"] in seen:
                raise ValueError("duplicate group identity across independent cohorts")
            seen.add(meta["group_id"])
            cohorts.append((relative, meta))
    sources = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    outdir.mkdir(parents=True, exist_ok=False)
    for method in methods:
        (outdir / method).mkdir()
    (outdir / "overlays").mkdir()
    _write(outdir / "manifest.json", {"protocol_sha256": protocol_sha256, "config": CONFIG,
        "methods": methods, "input_sha256": inputs, "source_sha256": sources,
        "expected_input_hashes_verified": expected_input_sha256 is not None,
        "model_weight_sha256": adapter.weight_sha256 if adapter is not None else None,
        "model_initialization_ms": initialization_ms if adapter is not None else None,
        "label_files_read": [], "completion_marker": "report.json_only_after_success"})
    groups = {method: [] for method in methods}
    tiles, cache_load_ms, footprints = {}, [], []
    begin = time.perf_counter()
    for relative, meta in cohorts:
        if time.perf_counter() - begin > CONFIG["maximum_processing_s"]:
            raise TimeoutError("frozen B processing budget exceeded; incomplete output retained")
        started = time.perf_counter()
        data = _load_data(base / relative / "data.npz", meta["start_index"])
        cache_load_ms.append(1000 * (time.perf_counter() - started))
        footprints.append(sum(array.nbytes for array in data.values()))
        for method in methods:
            result = replay_cohort(meta, data, method, adapter=adapter)
            result.update(protocol_sha256=protocol_sha256,
                          input_sha256={name: inputs[f"{relative}/{name}"] for name in ("meta.json", "data.npz")})
            _write(outdir / method / f"{meta['side']}_{meta['start_index']:06d}.json", result)
            groups[method].append(result)
            if meta["start_index"] in CONFIG["fixed_overlay_start_indices"]:
                row = result["rows"][-1]
                overlay = overlay_current(data["frames256"][-1], row, meta["box"], available=meta["available"])
                filename = f"{meta['side']}_{meta['start_index']:06d}_{method}.png"
                with (outdir / "overlays" / filename).open("xb") as handle:
                    overlay.save(handle, format="PNG")
                tiles[(meta["side"], meta["start_index"], method)] = {
                    "image": overlay, "source_frame": row["source_frame"],
                    "count": len(drawable_points(row, meta["box"]))}
    for side in CONFIG["sides"]:
        _contact_sheet(outdir / "overlays", side, methods, tiles)
    if any(sha256(base / key) != value for key, value in inputs.items()):
        raise RuntimeError("prepared input changed during replay; incomplete result retained")
    if any(sha256(ROOT / name) != value for name, value in sources.items()):
        raise RuntimeError("source changed during replay; incomplete result retained")
    report = {"schema": CONFIG["schema"], "protocol_sha256": protocol_sha256,
              "input_sha256": inputs, "source_sha256": sources,
              "methods": {method: summarize_method(results) for method, results in groups.items()},
              "per_side": {side: {method: summarize_method([g for g in results if g["side"] == side])
                                   for method, results in groups.items()} for side in CONFIG["sides"]},
              "offline_wall_ms_including_cache_json_and_plots": 1000 * (time.perf_counter() - begin),
              "cache_load_ms_per_cohort": _stats(cache_load_ms),
              "loaded_input_array_bytes_per_cohort_measured": _stats(footprints),
              "process_peak_rss_bytes": None, "production_end_to_end_latency_ms": None,
              "candidate_persistence_is_not_material_identity_accuracy": True,
              "overlapping_cohorts_are_not_independent_events": True,
              "physical_t1": "not_evaluable", "physical_t2": "not_evaluable",
              "deployment_permitted": False}
    _write(outdir / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--method", action="append", choices=("lk256", "lk_native", "tapnext"))
    parser.add_argument("--expected-input-hashes")
    parser.add_argument("--weights")
    parser.add_argument("--expected-weights-sha256")
    parser.add_argument("--vendor")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    expected = json.loads(Path(args.expected_input_hashes).read_text(encoding="utf-8-sig")) if args.expected_input_hashes else None
    report = run(args.base, args.outdir, args.protocol_sha256, args.method or CONFIG["default_methods"], expected,
                 weights_path=args.weights, expected_weights_sha256=args.expected_weights_sha256,
                 vendor_path=args.vendor, device=args.device)
    print(json.dumps({"output": str(Path(args.outdir).resolve()), "methods": report["methods"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
