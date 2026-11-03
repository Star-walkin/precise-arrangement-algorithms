"""V26 A-only cached-feature replay. No labels or devices enter inference.

The existing V16 import closure was read before reuse: v16_event_candidates
imports centerline_event_fusion (math/numpy) and visual_turn_phase (stdlib).
These modules do not initialize cameras, motion, serial or network devices.
Old t0/t1 names are preserved only inside legacy diagnostics; exported events
are explicitly renamed gap proxies, never physical t1/t2 confirmations.

Only features/left.json and features/right.json are read as experimental data.
Every row, including an unavailable one, remains in the denominator. An output
directory must be new. Events are written once in arrival order to a freshly
created JSONL file and never edited after emission. Processing time is measured
separately from the logical source/availability timestamps supplied by prepare.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vision_geometry_lab.v16_event_candidates import EventCandidateTracker, PROTOCOL as V16_CONFIG
from .event_signals import CausalEventDetector, DEFAULT_CONFIG, QUALITY_FIELDS


ROOT = Path(__file__).resolve().parents[3]
SOURCE_FILES = (
    "vision_geometry_lab/experiments/v26_causal_material/a_runner.py",
    "vision_geometry_lab/experiments/v26_causal_material/event_signals.py",
    "vision_geometry_lab/experiments/v26_causal_material/test_a_runner.py",
    "vision_geometry_lab/experiments/v26_causal_material/test_event_signals.py",
    "vision_geometry_lab/v16_event_candidates.py",
    "vision_prototype/centerline_event_fusion.py",
    "vision_prototype/visual_turn_phase.py",
    "vision_geometry_lab/__init__.py",
)
METHODS = ("trend_cusum", "bocpd", "legacy_gap_baseline")
CONFIG = {
    "maximum_replay_s": 300,
    "schema": "v26_route_a_runner_1",
    "signal_config": copy.deepcopy(DEFAULT_CONFIG),
    "legacy_gap_config": copy.deepcopy(V16_CONFIG),
    "sides": ["left", "right"],
    "methods": list(METHODS),
    "old_gap_input": "existing_gap_proxy_optical_width_units_never_x_ref_px",
    "time_axis": "supplied_host_monotonic_receive_time_not_exposure_or_device_ticks",
    "legacy_processing_ms": 0.0,
    "legacy_period_input": None,
    "legacy_stationary_policy": "explicit_stationary_with_named_source_only",
    "event_log": "exclusive_create_append_in_arrival_order_no_revisions",
    "emit_clock": "logical_available_at_s_excludes_compute_transport_and_hardware",
    "confirmed_physical_labels": [],
    "physical_t1_evaluation": "not_evaluable",
    "physical_t2_evaluation": "not_evaluable",
    "false_alarm_rate_evaluation": "not_evaluable_without_verified_negative_duration",
    "figure_selection": "both_complete_streams_no_truth_or_candidate_based_window",
    "pipeline_timing_scope": "cached_feature_three_detectors_only_not_feature_extraction",
    "deployment_permitted": False,
}
ROW_FIELDS = ("stream_id", "source_frame", "time_s", "max_source_timestamp",
              "available_at_s", "available", "reason", "angle_proxy_rad", "x_ref_px",
              "old_gap_usable", "old_gap_proxy", "nominal_time_s") + QUALITY_FIELDS
COLORS = {"trend_cusum": "#efaa54", "bocpd": "#d58afa", "legacy_gap_baseline": "#61c6ed"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, payload):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (ValueError, TypeError, OverflowError):
        return False


def stats(values):
    data = np.array([float(x) for x in values if _finite(x)], dtype=np.float64)
    if not len(data):
        return {"count": 0, "min": None, "mean": None, "p50": None, "p95": None, "max": None}
    return {"count": len(data), "min": float(data.min()), "mean": float(data.mean()),
            "p50": float(np.percentile(data, 50)), "p95": float(np.percentile(data, 95)),
            "max": float(data.max())}


class _LegacyGapAdapter:
    """No period is supplied, so a physical or proxy t2 cannot be invented."""

    def __init__(self, stream_id):
        self.stream_id = stream_id
        self.tracker = EventCandidateTracker()
        self.max_source_timestamp = None
        self.events = []

    def update(self, row):
        now = float(row["time_s"])
        self.max_source_timestamp = max(float(row["max_source_timestamp"]),
                                        self.max_source_timestamp if
                                        self.max_source_timestamp is not None else -math.inf)
        usable = row.get("old_gap_usable") is True and _finite(row.get("old_gap_proxy"))
        motion_source = row.get("motion_source")
        stationary = (row.get("motion_status") == "stationary" and
                      motion_source not in (None, "", "unknown", "unavailable"))
        native = self.tracker.update({
            "time_s": now, "source_frame": row["source_frame"], "usable": usable,
            "gap_proxy": float(row["old_gap_proxy"]) if usable else None,
            "processing_ms": 0.0, "stationary": stationary,
        }, period=None)
        events = []
        for item in native:
            if item["event"] == "t2":
                raise RuntimeError("legacy t2 is forbidden without an independent period")
            event = {
                "event_id": f"{self.stream_id}:legacy_gap_baseline:{len(self.events) + 1}",
                "stream_id": self.stream_id, "method": "legacy_gap_baseline",
                "event_type": f"legacy_gap_{item['event']}_proxy", "status": "candidate",
                "t_est_lo": item["event_time_s"], "t_est_hi": item["event_time_s"],
                "t_emit": float(row["available_at_s"]),
                "max_source_timestamp": self.max_source_timestamp,
                "score": None, "score_semantics": "legacy_has_no_calibrated_scalar_score",
                "evidence_source": ["old_gap_proxy"], "legacy_native": copy.deepcopy(item),
                "interval_semantics": "legacy_point_proxy_zero_width_is_not_precision_or_CI",
                "rejection_reason": "gap_contact_or_reversal_does_not_confirm_physical_layer",
                "physical_t1_confirmed": False, "mechanical_turn_confirmed": False,
                "device_advisory": False, "deployment_permitted": False,
            }
            self.events.append(copy.deepcopy(event))
            events.append(event)
        return {
            "stream_id": self.stream_id, "source_frame": row["source_frame"],
            "time_s": now, "available_at_s": float(row["available_at_s"]),
            "max_source_timestamp": float(row["max_source_timestamp"]),
            "available": usable, "status": "candidate" if usable else "unknown",
            "old_gap_proxy": float(row["old_gap_proxy"]) if usable else None,
            "stationary_from_explicit_evidence": stationary,
            "reason": self.tracker.latest_reason if usable else "old_gap_unavailable",
            "events": events,
        }


def summarize_method(rows, events, compute_ms):
    n = len(rows)
    duration = float(rows[-1]["time_s"] - rows[0]["time_s"]) if n > 1 else 0.
    available = sum(row["available"] for row in rows)
    observed_duration = 0.
    for before, after in zip(rows, rows[1:]):
        gap = after["time_s"] - before["time_s"]
        if before["available"] and after["available"] and gap <= DEFAULT_CONFIG["maximum_missing_s"] + 1e-9:
            observed_duration += gap
    return {
        "source_row_denominator": n, "available_rows": available,
        "unavailable_rows": n - available, "available_row_fraction": available / n if n else None,
        "stream_duration_s": duration,
        "adjacent_available_duration_s": observed_duration,
        "adjacent_duration_policy": "both_endpoints_available_gap_at_most_0.65s_no_filled_intervals",
        "candidate_count": len(events),
        "candidate_type_counts": dict(Counter(e["event_type"] for e in events)),
        "descriptive_candidates_per_stream_minute": len(events) * 60 / duration if duration else None,
        "rate_is_false_alarm_rate": False,
        "candidate_estimate_width_s": stats([e["t_est_hi"] - e["t_est_lo"] for e in events]),
        "emit_minus_max_source_s": stats([e["t_emit"] - e["max_source_timestamp"] for e in events]),
        "emit_minus_estimate_high_s": stats([e["t_emit"] - e["t_est_hi"] for e in events]),
        "delay_semantics": "statistical_proxy_confirmation_delay_not_physical_t1_latency",
        "detector_compute_ms": stats(compute_ms),
        "detector_compute_total_ms": float(sum(compute_ms)),
        "physical_t1": {"status": "not_evaluable", "confirmed_labels": 0,
                         "accuracy": None, "misses": None, "duplicates": None, "delay_s": None},
        "physical_t2": {"status": "not_evaluable", "confirmed_labels": 0,
                         "accuracy": None, "misses": None, "duplicates": None, "delay_s": None},
        "false_alarms_per_verified_negative_minute": None,
        "independent_negative_duration_s": None,
    }


def replay_rows(rows, *, event_sinks=None):
    """Public in-memory API for synthetic causality tests and the file runner.

    Each event sink receives a detached event immediately after first emission.
    Runtime measurements are separate from deterministic prediction payloads.
    Input order is not sorted, and extra annotation keys are not projected in.
    """
    if not rows:
        raise ValueError("empty streams cannot silently disappear from evaluation")
    stream_id = rows[0]["stream_id"]
    start = time.perf_counter()
    engines = {method: CausalEventDetector(stream_id, method) for method in METHODS[:2]}
    engines["legacy_gap_baseline"] = _LegacyGapAdapter(stream_id)
    init_ms = 1000 * (time.perf_counter() - start)
    output = {method: {"rows": [], "events": [], "compute_ms": []} for method in METHODS}
    pipeline_ms = []
    sinks = event_sinks or {}
    for source in rows:
        row = {key: copy.deepcopy(source[key]) for key in ROW_FIELDS if key in source}
        compute_total = 0.
        for method in METHODS:
            started = time.perf_counter()
            result = engines[method].update(row)
            elapsed = 1000 * (time.perf_counter() - started)
            compute_total += elapsed
            events = result.get("events", [result["event"]] if result.get("event") else [])
            output[method]["rows"].append(result)
            output[method]["events"].extend(copy.deepcopy(events))
            output[method]["compute_ms"].append(elapsed)
            for event in events:
                if event["max_source_timestamp"] > event["t_emit"] + 1e-9:
                    raise RuntimeError("noncausal event emission")
                if method in sinks:
                    sinks[method](copy.deepcopy(event))
        pipeline_ms.append(compute_total)
    for method in METHODS:
        item = output[method]
        item["summary"] = summarize_method(item["rows"], item["events"], item.pop("compute_ms"))
    # This queue is an explicitly hypothetical single-thread cache replay; it
    # excludes image decoding, free-rope extraction, transport and actuators.
    completion, backlog = None, []
    for row, elapsed in zip(rows, pipeline_ms):
        arrival = float(row["available_at_s"])
        completion = max(arrival, completion if completion is not None else arrival) + elapsed / 1000
        backlog.append(1000 * (completion - arrival))
    return {
        "stream_id": stream_id, "methods": output,
        "timing": {"initialization_ms": init_ms, "pipeline_detector_compute_ms": stats(pipeline_ms),
                   "pipeline_detector_compute_total_ms": float(sum(pipeline_ms)),
                   "source_arrival_interval_ms": stats([1000 * (b["available_at_s"] - a["available_at_s"])
                                                         for a, b in zip(rows, rows[1:])]),
                   "simulated_cached_detector_queue_delay_ms": stats(backlog),
                   "production_end_to_end_latency_ms": None,
                   "feature_extraction_cost_included": False,
                   "queue_status": "simulation_only_not_production_measurement"},
    }


def _font(size):
    path = Path("C:/Windows/Fonts/msyh.ttc")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default(size=size)


def render_timeline(path, side, inputs, replay):
    """Full-stream plots; no annotation/time-of-interest lines or windowing."""
    canvas = Image.new("RGB", (1900, 1600), "#121c2a")
    draw = ImageDraw.Draw(canvas)
    white, muted = "#e9eff6", "#aab9c9"
    origin = float(inputs[0]["time_s"])
    end = max(float(inputs[-1]["time_s"]), float(inputs[-1]["available_at_s"]))
    span = max(end - origin, 1e-6)
    draw.text((35, 22), f"V26 A | {side} | 完整输入时间线，不是物理换层确认", font=_font(31), fill=white)
    draw.text((35, 70), f"横轴：相对本流起点的接收秒；起点={origin:.6f}。非曝光时间；无27.3秒真值线。",
              font=_font(23), fill=muted)
    draw.text((35, 105), "灰色：当前量不可用；缺测不插值。刻线=首次发出，横线=统计估计区间，不是物理t1置信区间。",
              font=_font(22), fill="#ffd180")
    primary = replay["methods"]["trend_cusum"]["rows"]
    x0, x1 = 180, 1840

    def x(time_s):
        return x0 + (float(time_s) - origin) / span * (x1 - x0)

    def panel(top, label, values, color, height=180):
        bottom = top + height
        draw.text((30, top + 3), label.replace("\n", " / "), font=_font(21), fill=white)
        finite = [float(v) for v in values if _finite(v)]
        if finite:
            lower, upper = min(finite), max(finite)
            margin = max((upper - lower) * .08, 1e-6)
            lower, upper = lower - margin, upper + margin
        else:
            lower, upper = -1., 1.
        yt, yb = top + 39, bottom - 23
        draw.rectangle((x0, yt, x1, yb), fill="#19283a", outline="#476079")
        draw.text((35, yt), f"{upper:.4g}", font=_font(17), fill=muted)
        draw.text((35, yb - 20), f"{lower:.4g}", font=_font(17), fill=muted)
        previous = None
        for i, (row, value) in enumerate(zip(inputs, values)):
            if not _finite(value):
                nxt = inputs[min(i + 1, len(inputs) - 1)]["time_s"]
                draw.rectangle((x(row["time_s"]), yt, max(x(row["time_s"]) + 2, x(nxt)), yb), fill="#424853")
                previous = None
                continue
            point = (x(row["time_s"]), yb - (float(value) - lower) / (upper - lower) * (yb - yt))
            if previous is not None and row["time_s"] - inputs[i - 1]["time_s"] <= .65 + 1e-9:
                draw.line((previous, point), fill=color, width=2)
            else:
                draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill=color)
            previous = point
        for k in range(7):
            time_s = origin + span * k / 6
            draw.text((x(time_s) - 18, bottom - 19), f"{time_s - origin:.1f}", font=_font(16), fill=muted)

    for top, channel, key, label, color in (
        (155, "angle_proxy_rad", "unwrapped_or_native_value", "图像线方向\n模π展开(rad)", "#76d4e7"),
        (355, "x_ref_px", "unwrapped_or_native_value", "参考线交点\nx (px)", "#82d49a"),
        (555, "angle_proxy_rad", "standardized_residual", "角度残差\n标准化z", "#76d4e7"),
        (755, "x_ref_px", "standardized_residual", "位置残差\n标准化z", "#82d49a"),
    ):
        panel(top, label, [row["channels"][channel].get(key) for row in primary], color)
    legacy = replay["methods"]["legacy_gap_baseline"]["rows"]
    panel(955, "旧gap代理\n光学宽度单位", [row["old_gap_proxy"] for row in legacy], COLORS["legacy_gap_baseline"])
    timeline_top = 1225
    draw.text((30, 1158), "首次候选时间线 | 仅描述，不把候选率当误报率；未知区间既非正例也非负例。", font=_font(23), fill=white)
    for index, method in enumerate(METHODS):
        y = timeline_top + index * 65
        item = replay["methods"][method]
        short_name = {"trend_cusum": "CUSUM", "bocpd": "BOCPD", "legacy_gap_baseline": "旧gap基线"}[method]
        draw.text((30, y - 10), short_name, font=_font(20), fill=COLORS[method])
        draw.line((x0, y, x1, y), fill="#476079", width=1)
        for event in item["events"]:
            left, right, emit = x(event["t_est_lo"]), x(event["t_est_hi"]), x(event["t_emit"])
            draw.line((left, y - 5, max(left + 1, right), y - 5), fill=COLORS[method], width=4)
            draw.line((emit, y - 14, emit, y + 14), fill=COLORS[method], width=3)
        summary = item["summary"]
        draw.text((x0, y + 19), f"可用 {summary['available_rows']}/{summary['source_row_denominator']} | 候选 {summary['candidate_count']} | 物理t1/t2: not_evaluable",
                  font=_font(17), fill=muted)
    draw.text((35, 1427), "两种统计方法会对周期/相关噪声发出候选；标准BOCPD也可能响应孤立离群点。合成失败已保留。", font=_font(22), fill="#ffd180")
    draw.text((35, 1470), "本图不显示真实接触、爬层或整圈真值；旧gap的t0/t1只是历史代理命名，未接设备。", font=_font(22), fill=muted)
    draw.text((35, 1513), "全部图为事后诊断展示；滤波与事件推理仍逐帧只用当时可用前缀。", font=_font(22), fill=muted)
    with Path(path).open("xb") as handle:
        canvas.save(handle, format="PNG")


def run(base, outdir, protocol_sha256, expected_feature_sha256=None, expected_source_sha256=None):
    base, outdir = Path(base).resolve(), Path(outdir).resolve()
    if len(protocol_sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in protocol_sha256):
        raise ValueError("protocol_sha256 must be a SHA256 hex digest")
    if outdir.exists():
        raise FileExistsError(f"refusing to overwrite {outdir}")
    sources = {relative: sha256(ROOT / relative) for relative in SOURCE_FILES}
    if expected_source_sha256 is not None and sources != expected_source_sha256:
        raise ValueError("source freeze hash mismatch")
    features, input_hashes = {}, {}
    for side in CONFIG["sides"]:
        path = base / "features" / f"{side}.json"
        input_hashes[side] = sha256(path)
        if expected_feature_sha256 is not None and input_hashes[side] != expected_feature_sha256.get(side):
            raise ValueError(f"feature hash mismatch: {side}")
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("protocol_sha256", protocol_sha256) != protocol_sha256:
            raise ValueError(f"feature protocol mismatch: {side}")
        if not isinstance(document.get("rows"), list) or not document["rows"]:
            raise ValueError(f"empty/missing feature rows: {side}")
        features[side] = document["rows"]
    if features["left"][0]["stream_id"] == features["right"][0]["stream_id"]:
        raise ValueError("left and right must have separate stream identities")
    outdir.mkdir(parents=True, exist_ok=False)
    _write_json(outdir / "manifest.json", {
        "schema": CONFIG["schema"], "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_sha256": protocol_sha256, "config": CONFIG,
        "input_feature_sha256": input_hashes, "source_sha256": sources,
        "expected_feature_hashes_verified": expected_feature_sha256 is not None,
        "expected_source_hashes_verified": expected_source_sha256 is not None,
        "source_import_audit": "V16 event closure is pure CPU/stdlib/numpy; no hardware initialization",
        "label_files_read": [], "completion_marker": "report.json_written_only_after_success",
    })
    reports = {}
    start = time.perf_counter()
    for side, rows in features.items():
        if time.perf_counter() - start > CONFIG["maximum_replay_s"]:
            raise TimeoutError("frozen A replay budget exceeded; incomplete output retained")
        side_dir = outdir / side
        side_dir.mkdir()
        handles = {method: (side_dir / f"{method}.events.jsonl").open("x", encoding="utf-8", newline="\n")
                   for method in METHODS}
        try:
            def make_sink(handle):
                def emit(event):
                    handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                return emit
            replay = replay_rows(rows, event_sinks={method: make_sink(handle) for method, handle in handles.items()})
        finally:
            for handle in handles.values():
                handle.close()
        for method in METHODS:
            payload = replay["methods"][method]
            _write_json(side_dir / f"{method}.json", {
                "protocol_sha256": protocol_sha256, "input_sha256": input_hashes[side],
                "stream_id": replay["stream_id"], "method": method, **payload,
            })
        render_timeline(side_dir / "timeline.png", side, rows, replay)
        reports[side] = {
            "stream_id": replay["stream_id"], "source_row_denominator": len(rows),
            "timestamp_basis_values": sorted({str(row.get("timestamp_basis", "unknown")) for row in rows}),
            "methods": {method: replay["methods"][method]["summary"] for method in METHODS},
            "timing": replay["timing"], "timeline": str(side_dir / "timeline.png"),
        }
    if any(sha256(base / "features" / f"{side}.json") != value for side, value in input_hashes.items()):
        raise RuntimeError("feature input changed during replay; incomplete output retained")
    if any(sha256(ROOT / relative) != value for relative, value in sources.items()):
        raise RuntimeError("source changed during replay; incomplete output retained")
    report = {
        "schema": CONFIG["schema"], "protocol_sha256": protocol_sha256,
        "input_feature_sha256": input_hashes, "source_sha256": sources, "streams": reports,
        "offline_driver_wall_ms_including_event_io_json_and_plots": 1000 * (time.perf_counter() - start),
        "physical_t1": {"status": "not_evaluable", "reason": "independent_confirmed_labels_empty"},
        "physical_t2": {"status": "not_evaluable", "reason": "no_confirmed_t1_or_directed_mechanical_2pi"},
        "candidate_counts_are_not_true_or_false_event_counts": True,
        "unknown_intervals_are_not_negative_labels": True,
        "production_end_to_end_latency_ms": None, "deployment_permitted": False,
    }
    _write_json(outdir / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--expected-feature-hashes", help="JSON mapping left/right to SHA256, not labels")
    parser.add_argument("--expected-source-hashes", help="JSON mapping SOURCE_FILES to SHA256, not labels")
    args = parser.parse_args()
    load = lambda path: json.loads(Path(path).read_text(encoding="utf-8")) if path else None
    result = run(args.base, args.outdir, args.protocol_sha256,
                 load(args.expected_feature_hashes), load(args.expected_source_hashes))
    print(json.dumps({"output": str(Path(args.outdir).resolve()),
                      "physical_t1": result["physical_t1"], "physical_t2": result["physical_t2"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
