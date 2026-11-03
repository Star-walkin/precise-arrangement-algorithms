"""Post-hoc V17 diagnostic figures; no inference changes, GUI or equipment.

The requested reference times are chosen AFTER the experiment. They are not
algorithm detections and never enter a model or tracker. Green denotes current
direct grid support; blue denotes historical prediction; magenta denotes an
identity-unconfirmed V16 model path. Sigma is a heuristic scale, NOT a calibrated
confidence interval. The full 0--60 s support strip includes every sampled frame.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .v16_video_measurement import read, sha, BASE as V16_BASE
from .v17_boundary_trial import verify, BASE, MODES, validate_inputs, validate_metadata
from .v11_path_contract import array_sha256


REFERENCE_TIMES = {"left": (24.6, 27.2, 30.0), "right": (11.4, 14.4, 16.8)}
COLORS = {"direct": (0, 225, 90), "predicted": (40, 135, 255), "model": (240, 55, 220)}
FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")


def font(size):
    return ImageFont.truetype(str(FONT_PATH), size) if FONT_PATH.is_file() else ImageFont.load_default()


def boundary_runs(boundary, max_step=4.01):
    """Split gaps and provenance changes; never draw a green bridge over blue."""
    points, states = boundary.get("points", []), boundary.get("states", [])
    if len(points) != len(states):
        raise ValueError("boundary states do not align with points")
    runs, current, old_state = [], [], None
    for point, state in zip(points, states):
        xy = np.asarray(point, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all() or state not in ("direct", "predicted"):
            raise ValueError("invalid boundary point or state")
        if current and (state != old_state or xy[0] <= current[-1][0] or xy[0]-current[-1][0] > max_step):
            runs.append((old_state, current)); current = []
        current.append(xy.tolist()); old_state = state
    if current:
        runs.append((old_state, current))
    return runs


def validate_decoded_frame(frame, actual_index, expected_index, meta, fps):
    validate_metadata(meta, expected_index, fps)
    if actual_index != expected_index:
        raise ValueError("decoded frame index differs from requested original frame")
    if array_sha256(frame) != meta["frame_sha256"]:
        raise ValueError("decoded original frame pixels differ from extraction hash")


def overlay(frame, boundary, decoded):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for state, points in boundary_runs(boundary):
        p = np.rint(points).astype(np.int32)
        if len(p) > 1:
            cv2.polylines(rgb, [p], False, COLORS[state], 8, cv2.LINE_AA)
        else:
            cv2.circle(rgb, tuple(p[0]), 4, COLORS[state], -1, cv2.LINE_AA)
    for segment in decoded.get("observed_segments", []):
        points = np.asarray(segment, float)
        if points.ndim != 2 or points.shape[1:] != (2,) or not np.isfinite(points).all():
            raise ValueError("invalid decoded model path")
        if len(points) >= 2:
            cv2.polylines(rgb, [np.rint(points).astype(np.int32)], False, COLORS["model"], 6, cv2.LINE_AA)
    return Image.fromarray(rgb)


def shared_crop(frame_shape, direct, temporal, decoded):
    """Display-only union crop; never used for fitting or experimental inputs."""
    points = list(direct.get("points", [])) + list(temporal.get("points", []))
    points += [p for segment in decoded.get("observed_segments", []) for p in segment]
    h, w = frame_shape[:2]
    if not points:
        return (0, 0, w, h)
    values = np.asarray(points, float)
    margin = max(35., .025 * min(w, h))
    lo = np.maximum(0, np.floor(values.min(axis=0)-margin)).astype(int)
    hi = np.minimum([w, h], np.ceil(values.max(axis=0)+margin)).astype(int)
    if np.any(hi <= lo):
        return (0, 0, w, h)
    return (int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1]))


def _paste_fit(canvas, picture, box):
    x, y, w, h = box
    image = ImageOps.contain(picture, (w, h), Image.Resampling.LANCZOS)
    canvas.paste(image, (x+(w-image.width)//2, y+(h-image.height)//2))


def _fmt(value, suffix=""):
    return f"{float(value):.3f}{suffix}" if value is not None and math.isfinite(float(value)) else "不可用"


def make_triptych(frame, session, index, fps, direct, temporal, decoded):
    cell_w = 720
    canvas = Image.new("RGB", (3*cell_w, 1160), "#111820")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 8), f"V17 后验诊断 · {session} · 实际帧 {index} · AVI {index/fps:.1f}s", font=font(26), fill="white")
    draw.text((16, 46), "本组时刻由人工参考后验选取，不是算法检出节点；只读、不接设备。", font=font(21), fill="#ffd580")
    draw.text((16, 80), "绿色：当前直接网格支持    蓝色：历史边界预测    品红：V16 模型路径（身份未确认）", font=font(22), fill="white")
    pictures = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                overlay(frame, direct["boundary"], decoded), overlay(frame, temporal["boundary"], decoded)]
    crop = shared_crop(frame.shape, direct["boundary"], temporal["boundary"], decoded)
    titles = ["原始画面（无叠加）", "当前直接网格（无历史传播）", "时序边界（实测支持 / 预测分色）"]
    ambiguity = decoded.get("diagnostics", {}).get("ambiguous")
    for i, picture in enumerate(pictures):
        x = i*cell_w
        draw.text((x+14, 120), titles[i], font=font(23), fill="white")
        _paste_fit(canvas, picture, (x+8, 158, cell_w-16, 420))
        draw.text((x+14, 588), "同一原图坐标区域放大", font=font(21), fill="#ced8e2")
        _paste_fit(canvas, picture.crop(crop), (x+8, 626, cell_w-16, 340))
        if i == 0:
            lines = ["原图指纹及实际解码帧号均已校验",
                     f"模型 valid={decoded.get('valid', False)}；ambiguous={ambiguity}",
                     "valid / 网络概率不代表活动绳可见性已确认",
                     f"共同放大区域：{crop}"]
        else:
            row = direct if i == 1 else temporal
            b, f = row["boundary"], row["features"]
            lines = [f"直接点 {b.get('direct_point_count', 0)} / 预测点 {b.get('predicted_point_count', 0)}；usable={f.get('usable', False)}",
                     f"最老支持龄期 {_fmt(b.get('boundary_max_age_s'), 's')}；sigma {_fmt(b.get('boundary_sigma_px'), 'px')}",
                     "sigma 为启发式不确定尺度，不是统计置信区间",
                     f"几何代理间隙 {_fmt(f.get('image_gap_proxy'))}；不是实际根数"]
        for j, line in enumerate(lines):
            draw.text((x+14, 981+j*37), line, font=font(19), fill="#ced8e2")
    return canvas, crop


def support_strip(measurements):
    canvas = Image.new("RGB", (1900, 720), "#111820")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), "V17 完整 0–60s 支持状态（全部采样帧，不挑成功片段）", font=font(26), fill="white")
    draw.text((16, 50), "绿：几何可用候选；蓝：存在历史预测支持；品红：模型歧义；灰：无此支持。t0/t1无候选；t2未评估（未提供独立整圈相位）", font=font(20), fill="#ced8e2")
    left, width = 320, 1520
    for session_index, session in enumerate(("left", "right")):
        y = 130+session_index*275
        draw.text((16, y-38), session+" 原视频", font=font(23), fill="white")
        direct, temporal = (measurements[session][mode]["records"] for mode in MODES)
        bands = [("直接网格 usable", [r["features"]["usable"] for r in direct], COLORS["direct"]),
                 ("时序模式 usable", [r["features"]["usable"] for r in temporal], COLORS["direct"]),
                 ("存在历史预测点", [r["boundary"].get("predicted_point_count", 0)>0 for r in temporal], COLORS["predicted"]),
                 ("模型路径 ambiguous", [bool(r["decoded_path_ambiguous"]) for r in temporal], COLORS["model"])]
        for j, (label, flags, color) in enumerate(bands):
            by = y+j*43
            draw.text((16, by), f"{label}  {sum(flags)}/{len(flags)}", font=font(20), fill="white")
            for k, (r, flag) in enumerate(zip(temporal, flags)):
                x0 = left+width*r["time_s"]/60
                next_time = temporal[k+1]["time_s"] if k+1<len(temporal) else 60.2
                x1 = min(left+width, left+width*next_time/60)
                draw.rectangle((int(x0), by+1, max(int(x0), int(x1)), by+29), fill=color if flag else "#34414c")
        for t in range(0, 61, 10):
            x = left+width*t/60
            draw.text((x-12, y+178), str(t)+"s", font=font(18), fill="#ced8e2")
    draw.text((16, 675), "usable 仅满足当前几何比较门槛，不证明绳身份、最后一圈、换层或机械整圈。", font=font(20), fill="#ffd580")
    return canvas


def run(trial=BASE, output=None):
    trial = Path(trial).resolve()
    output = Path(output) if output else trial/"diagnostics"/("posthoc_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    if output.exists():
        raise FileExistsError("diagnostics output exists; never overwrite")
    p = verify(trial, verify_videos=True)
    protocol_hash = sha(trial/"protocol.json")
    input_hashes, measurements, source_rows, extraction = {}, {}, {}, {}
    for session in ("left", "right"):
        measurements[session] = {}
        video = p["videos"][session]
        ex_path = V16_BASE/"extract"/session/"index.json"
        old_path = V16_BASE/"measure"/session/"center_only.json"
        extraction[session], old = read(ex_path), read(old_path)
        validate_inputs(extraction[session], old, video["selected_indices"], video["fps"], p["input_protocol"]["sha256"], sha(ex_path))
        source_rows[session] = {r["frame_index"]: r for r in old["records"]}
        input_hashes.update({str(ex_path): sha(ex_path), str(old_path): sha(old_path)})
        for mode in MODES:
            path = trial/"measure"/session/(mode+".json")
            data = read(path)
            if (data["protocol_sha256"] != protocol_hash or data["session"] != session or data["mode"] != mode or
                    data["video"] != video or [r["frame_index"] for r in data["records"]] != video["selected_indices"]):
                raise ValueError("V17 measurement identity or sequence mismatch")
            for row in data["records"]:
                if abs(row["time_s"]-row["frame_index"]/video["fps"]) > 1e-8:
                    raise ValueError("V17 record does not use the source AVI time base")
            input_hashes[str(path)] = sha(path); measurements[session][mode] = data
    output.mkdir(parents=True)
    visuals = []
    for session, times in REFERENCE_TIMES.items():
        video = p["videos"][session]; cap = cv2.VideoCapture(video["path"])
        if not cap.isOpened():
            raise ValueError("cannot open source video")
        try:
            ex = {r["frame_index"]: r for r in extraction[session]["records"]}
            by_mode = {m: {r["frame_index"]: r for r in measurements[session][m]["records"]} for m in MODES}
            for time_s in times:
                index = round(time_s*video["fps"])
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok: raise ValueError("source frame decoding failed")
                actual = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))-1
                info = ex[index]
                if sha(info["metadata_path"]) != info["metadata_sha256"]:
                    raise ValueError("original extraction metadata hash changed")
                meta = read(info["metadata_path"])
                validate_decoded_frame(frame, actual, index, meta, video["fps"])
                direct, temporal = (by_mode[m][index] for m in MODES)
                decoded = source_rows[session][index]["decoded"]
                picture, crop = make_triptych(frame, session, index, video["fps"], direct, temporal, decoded)
                path = output/f"{session}_f{index:06d}_t{time_s:05.1f}.png"
                picture.save(path)
                visuals.append({"session": session, "source_frame": index, "actual_decoded_frame": actual,
                    "source_time_s": time_s, "frame_sha256": meta["frame_sha256"], "crop_xyxy": crop,
                    "image": path.name, "image_sha256": sha(path), "selection": "posthoc_user_reference_time_not_detected_event",
                    "decoded_valid": decoded.get("valid", False), "model_ambiguous": decoded.get("diagnostics", {}).get("ambiguous"),
                    "modes": {m: {"usable": by_mode[m][index]["features"]["usable"],
                                **{k: by_mode[m][index]["boundary"].get(k) for k in
                                   ("boundary_state", "direct_point_count", "predicted_point_count", "boundary_max_age_s", "boundary_sigma_px", "reason")}}
                              for m in MODES}})
        finally:
            cap.release()
    strip_path = output/"full_video_support.png"
    support_strip(measurements).save(strip_path)
    verify(trial, verify_videos=False)
    for path, digest in input_hashes.items():
        if sha(path) != digest: raise ValueError("input changed during plotting")
    manifest = {"schema": "v17_posthoc_visual_diagnostics_1", "protocol_sha256": protocol_hash,
        "script_sha256": sha(__file__), "input_hashes": input_hashes,
        "original_videos": p["videos"], "selection": "posthoc_reference_times; no contribution to inference, thresholds or event scoring",
        "reference_times_s": REFERENCE_TIMES, "visuals": visuals,
        "support_strip": {"image": strip_path.name, "sha256": sha(strip_path), "all_sampled_frames_included": True},
        "colors": {"green": "current direct grid support", "blue": "historical predicted boundary",
                   "magenta": "V16 model path candidate, identity and visibility unconfirmed"},
        "sigma_definition": "heuristic uncertainty scale in native pixels, not a calibrated confidence interval",
        "ground_truth_files_opened": False, "hardware_access": False, "main_ui_modified": False, "deployment_allowed": False}
    (output/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "figures": len(visuals)+1, "manifest": str(output/"manifest.json")}, ensure_ascii=False))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, default=BASE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.trial, args.output)


if __name__ == "__main__":
    main()
