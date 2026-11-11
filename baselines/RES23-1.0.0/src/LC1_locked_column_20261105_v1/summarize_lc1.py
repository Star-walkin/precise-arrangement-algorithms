"""Summarize existing development outputs; no source video decoding or inference."""
from __future__ import annotations

from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS = [
    ("左相机 0–8 s", "left111500_fixed_reference1300_0_8s"),
    ("右相机 24–42 s", "right113305_fixed_reference_24_42s"),
]


def number(s):
    try:
        value = float(s)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def span(rows, key):
    values = [number(r.get(key)) for r in rows]
    values = [v for v in values if v is not None]
    return None if not values else dict(first=values[0], last=values[-1], minimum=min(values), maximum=max(values))


def summarize(label, name):
    folder = ROOT / "runs" / name
    receipt = json.loads((folder / "run_receipt.json").read_text(encoding="utf-8"))
    with (folder / "measurements.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with gzip.open(folder / "diagnostics.jsonl.gz", "rt", encoding="utf-8") as stream:
        details = [json.loads(line) for line in stream if line.strip()]
    assert len(rows) == len(details) == receipt["processed_frames"]
    assert [int(r["frame_index"]) for r in rows] == [r["frame_index"] for r in details]
    assert not any(r["count_valid"].lower() == "true" for r in rows)
    assert not any(r["d_original_px"] for r in rows)
    assert all(r["diagnostic_reference_column_original_px"] for r in rows)

    # This is a diagnostic of the user's constant-angle assumption, not a
    # measurement of true contact or rope-count error. Calibration uses only
    # the first >=1 s, never future frames, and only the LC1-T measured lines.
    start = details[0]["time_s"]
    calibration = []
    calibration_end = None
    for item in details:
        abc = item["front"].get("line_abc")
        if abc is not None:
            abc = np.asarray(abc, float)
            abc /= np.hypot(abc[0], abc[1])
            if abc[1] < 0:
                abc = -abc
            calibration.append(float(np.arctan2(-abc[0], abc[1])))
        if item["time_s"] - start >= 1.0 and len(calibration) >= 8:
            calibration_end = item["frame_index"]
            break
    fixed_angle = float(np.median(calibration)) if calibration_end is not None else None
    discrepancies = []
    if fixed_angle is not None:
        normal = np.array([-math.sin(fixed_angle), math.cos(fixed_angle)])
        width = 4096
        reference = receipt["diagnostic_reference_column_original_px"]
        column = width - 1 - reference if receipt["internal_left_mirror"] else reference
        for item in details:
            if item["frame_index"] <= calibration_end:
                continue
            points = np.asarray(item["front"].get("line_points_xy", []), float).reshape(-1, 2)
            abc = item["front"].get("line_abc")
            if len(points) < 2 or abc is None or abs(abc[1]) < .05:
                continue
            c = -float(np.median(points @ normal))
            frozen_y = -(normal[0] * column + c) / normal[1]
            raw_y = -(abc[0] * column + abc[2]) / abc[1]
            discrepancies.append(abs(frozen_y - raw_y))
    reasons = Counter(reason for item in details for reason in item["surface"].get("reasons", []))
    hypothesis = dict(
        basis="LC1-T measured free-line points; fixed diagnostic column, not actual contact d",
        is_physical_count_error=False,
        first_one_second_calibration_end_frame=calibration_end,
        tested_frames=len(discrepancies),
        absolute_intersection_disagreement_px_median=float(np.median(discrepancies)) if discrepancies else None,
        absolute_intersection_disagreement_px_p95=float(np.percentile(discrepancies, 95)) if discrepancies else None,
        absolute_intersection_disagreement_px_max=float(max(discrepancies)) if discrepancies else None,
    )
    return dict(
        label=label, run=name, video=f"runs/{name}/overlay.mp4",
        source_video=receipt["source_video"], source_sha256=receipt["source_sha256"],
        source_frame_range_exclusive=receipt["frame_range_exclusive"],
        processed_frames=len(rows), duration_s=len(rows)/receipt["output_fps"],
        source_stable=receipt["source_stable"], encoded_verification=receipt["encoded_output_verification"],
        diagnostic_reference_column_original_px=receipt["diagnostic_reference_column_original_px"],
        automatically_locked_d_frames=0, accepted_automatic_count_frames=0,
        complete_path_supported_frames=receipt["complete_path_candidate_frames"],
        candidate_pitch_distance=span(rows, "continuous_pitch_distance_candidate"),
        observed_grooves=span(rows, "observed_grooves"),
        raw_angle_original_deg=span(rows, "raw_angle_deg"),
        surface_quality_counts=dict(Counter(r["surface_quality"] for r in rows)),
        surface_rejection_reasons=dict(reasons),
        median_algorithm_ms=receipt["median_algorithm_ms"], wall_seconds=receipt["wall_seconds"],
        first_second_frozen_angle_diagnostic=hypothesis,
        video_sha256=receipt["output_video_sha256"],
        source_snapshot_exists=(folder / "source_snapshot" / "manifest.json").exists(),
    )


def main():
    results = [summarize(*run) for run in RUNS]
    output = dict(status="diagnostic_preview_only_missing_P1_weights_and_S1", algorithm="LC1", preview="LC1-T.0.0",
                  formal_requested_pipeline_completed=False, runs=results)
    (ROOT / "experiment_summary_LC1T.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# LC1 视频试跑结果", "", "本页记录实际结果。两段均为 **LC1-T 诊断预演**；P1/H6、S1 尚未接入。固定列由人工指定，不是自动 d。", "",
             "| 片段 | 处理帧 | 人工参考列（原图 px） | 有效自动 d / 计数帧 | 算法中位耗时 |", "|---|---:|---:|---:|---:|"]
    for result in results:
        lines.append(f"| [{result['label']}]({result['video']}) | {result['processed_frames']} | {result['diagnostic_reference_column_original_px']:.0f} | 0 / 0 | {result['median_algorithm_ms']:.0f} ms/帧 |")
    lines += ["", "视频为源帧逐帧离线处理，输出以 10 FPS 播放；目前算法没有达到 10 FPS 推理速度。源图为 4096×3000，输出为 1920×1406；画面上的方程和列坐标均已换回原始未镜像坐标。", ""]
    for result in results:
        angle=result["raw_angle_original_deg"]
        hyp=result["first_second_frozen_angle_diagnostic"]
        candidate=result["candidate_pitch_distance"]
        lines += [f"## {result['label']}", "",
                  f"实测自由直段角度：{angle['first']:.3f}° → {angle['last']:.3f}°，全段范围 {angle['minimum']:.3f}° 至 {angle['maximum']:.3f}°。这些是传统备用前端观测，不是 S1 输出。",
                  "",
                  f"参考列候选距离：{candidate['first']:.2f} → {candidate['last']:.2f} 个绳距；这不是已确认的剩余根数，也不构成识别精度结果。",
                  "",
                  f"以最初 ≥1 s 的实测角度中位数冻结斜率、逐帧重估 C，与各帧独立直线在同一人工参考列比较：交点差中位数 {hyp['absolute_intersection_disagreement_px_median']:.2f} px，95 分位 {hyp['absolute_intersection_disagreement_px_p95']:.2f} px，最大 {hyp['absolute_intersection_disagreement_px_max']:.2f} px。它只检验冻结斜率带来的模型差异，不是真实接触点误差；未验证 d 恒定。",
                  "", "当前计数质量拒绝原因：", ""]
        lines += [f"- `{reason}`：{count} 帧" for reason,count in result["surface_rejection_reasons"].items()]
        lines += ["", f"[逐帧数值](runs/{result['run']}/measurements.csv) · [完整运行记录](runs/{result['run']}/run_receipt.json)", ""]
    lines += ["## 结果的边界", "", "两段没有一帧满足自动标定 d 的完整活动绳支持条件；没有生成正式整数根数，也没有以周期补线填充漏检。F3 当前只是恢复的传统观测核心，未运行完整旧版融合链。", "",
              "源文件下载完整性、逐帧时间索引和输出首末帧已核验，源视频未改动。左片段保留了运行时源文件快照；右片段在增加快照功能之前运行，没有运行时快照，应以其运行记录和输出哈希审阅。", "",
              "接入实际 P1/H6 权重及可运行 S1 后，需要重新跑自动标定版本，人工标注转折点与剩余根数，再报告成功率、误差、换层/换向重置情况。"]
    (ROOT / "RESULTS_LC1T.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
