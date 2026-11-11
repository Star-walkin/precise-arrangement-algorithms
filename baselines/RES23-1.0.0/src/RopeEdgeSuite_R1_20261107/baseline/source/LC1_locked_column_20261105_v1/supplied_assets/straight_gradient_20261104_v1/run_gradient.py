"""Run the E1-conditioned, original-image straight-rope gradient experiment.

Only the frozen open-image index and source pixels are prediction inputs.  This
module has no model, label, device, or controller imports.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gradient_core import analyze

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = Path(r"C:\precise_arrangement_results\straight_gradient_20261104_v1")
PANEL_HEADER = 108
ROW_GAP = 132
COLUMN_GAP = 36
FOOTER = 174
Image.MAX_IMAGE_PIXELS = 160_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"image decode failed: {path}")
    return image


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", size)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _draw_polyline(draw: ImageDraw.ImageDraw, points, origin, color, width=4):
    if len(points) < 2:
        return
    xy = [(origin[0] + float(x), origin[1] + float(y)) for x, y in points]
    draw.line(xy, fill=(0, 0, 0), width=width + 4, joint="curve")
    draw.line(xy, fill=color, width=width, joint="curve")


def render_four(image_bgr: np.ndarray, fields: dict, pred: dict, row: dict, config: dict) -> Image.Image:
    h, w = image_bgr.shape[:2]
    if (h, w) != (3000, 4096):
        raise ValueError(f"unexpected source shape {(h, w)}")
    original = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    blurred_bgr = np.asarray(fields["blurred_bgr"])
    grad = np.asarray(fields["gradient_magnitude"], dtype=np.float32)
    if blurred_bgr.shape != image_bgr.shape or blurred_bgr.dtype != np.uint8:
        raise ValueError("blurred image shape or dtype mismatch")
    if grad.shape != (h, w) or not np.isfinite(grad).all() or (grad < 0).any():
        raise ValueError("gradient field invalid")
    blurred = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2RGB)
    grad_max = float(config["display"]["gradient_magnitude_max"])
    gradient_view = np.rint(np.clip(grad / grad_max, 0, 1) * 255).astype(np.uint8)
    gradient_view = cv2.cvtColor(gradient_view, cv2.COLOR_GRAY2RGB)

    x2 = w + COLUMN_GAP
    y1 = PANEL_HEADER
    y2 = PANEL_HEADER + h + ROW_GAP
    board = Image.new("RGB", (2 * w + COLUMN_GAP, y2 + h + FOOTER), (244, 246, 249))
    board.paste(Image.fromarray(original), (0, y1))
    board.paste(Image.fromarray(blurred), (x2, y1))
    board.paste(Image.fromarray(gradient_view), (0, y2))
    board.paste(Image.fromarray(original), (x2, y2))
    draw = ImageDraw.Draw(board)
    failed = pred.get("status") == "PROCESS_FAILURE"
    headings = [
        (0, 0, "① 原图（未绘制）"),
        (x2, 0, "② 处理失败：仅复制原图" if failed else f"② 轻度高斯模糊（σ={config['blur_sigma_px']} px）"),
        (0, y2 - PANEL_HEADER, "③ 处理失败：梯度未生成" if failed else f"③ 模糊后梯度幅值（固定 0–{grad_max:g}）"),
        (x2, y2 - PANEL_HEADER, "④ 原图＋固定 E1 和直段检测结果"),
    ]
    for x, y, caption in headings:
        draw.text((x + 24, y + 19), caption, font=_font(43), fill=(18, 25, 36))
        draw.text((x + 24, y + 72), row["source_id"], font=_font(25), fill=(56, 66, 80))

    origin = (x2, y2)
    seed = row["e1"]["xy"]
    _draw_polyline(draw, seed, origin, (0, 246, 247), width=5)
    segments = pred.get("supported_segments_xy") or []
    width_px = pred.get("width_px")
    d = np.asarray(pred.get("axis_direction_xy") or
                   (np.asarray(seed[-1], dtype=float) - np.asarray(seed[0], dtype=float)), dtype=float)
    d /= max(np.linalg.norm(d), 1e-9)
    n = np.array([-d[1], d[0]])
    for segment in segments:
        if len(segment) >= 2:
            _draw_polyline(draw, segment, origin, (255, 221, 0), width=6)
        elif len(segment) == 1:
            sx, sy = origin[0] + float(segment[0][0]), origin[1] + float(segment[0][1])
            draw.ellipse((sx-4, sy-4, sx+4, sy+4), fill=(255, 221, 0))
        if len(segment) >= 2 and isinstance(width_px, (int, float)) and 5 <= width_px <= 100:
            p = np.asarray(segment, dtype=float)
            for sign in (-1, 1):
                side = p + sign * 0.5 * float(width_px) * n
                _draw_polyline(draw, side, origin, (37, 252, 77), width=3)
    end = pred.get("endpoint_xy")
    if end is not None:
        ex, ey = origin[0] + float(end[0]), origin[1] + float(end[1])
        r = 15
        draw.ellipse((ex-r-4, ey-r-4, ex+r+4, ey+r+4), fill=(0, 0, 0))
        draw.ellipse((ex-r, ey-r, ex+r, ey+r), fill=(255, 51, 46))

    fy = y2 + h + 14
    status = pred.get("status", "UNKNOWN")
    stop = pred.get("stop_reason", "unknown")
    length = float(pred.get("supported_extension_px") or 0.0)
    draw.text((20, fy), f"E1 为既有入口条件；黄线=有双边支持的直段中心，绿线=估计两侧，红点=最后有支持处。状态 {status}；终止 {stop}。", font=_font(31), fill=(20, 28, 38))
    draw.text((20, fy + 51), f"新检长度 {length:.1f} px；入口 ID {row['e1'].get('entry_id', 'unknown')}；全图梯度按固定尺度显示，像素单位。", font=_font(29), fill=(20, 28, 38))
    draw.text((20, fy + 100), "仅为离线图像直线证据；红点不表示真实入卷接触或同根身份，入口错误会传递到本结果。", font=_font(29), fill=(130, 40, 38))
    return board


def process_one(row: dict, output: Path, config: dict, config_hash: str, phase: str) -> dict:
    cv2.setNumThreads(1)
    start = time.perf_counter()
    source = Path(row["source_path"])
    if sha256(source) != row["source_sha256"]:
        raise ValueError(f"source SHA mismatch: {source}")
    image = read_bgr(source)
    if list(image.shape[:2]) != row["shape_hw"]:
        raise ValueError(f"source dimensions mismatch: {row['source_id']}")
    fields, prediction = analyze(image, row["e1"]["xy"], config)
    prediction = _json_ready(prediction)
    prediction.update({
        "source_id": row["source_id"], "source_group": row["source_group"],
        "camera_role": row["camera_role"], "source_use": row["source_use"],
        "source_path": str(source), "source_sha256": row["source_sha256"],
        "e1": row["e1"], "entry_is_input_condition": True,
        "method_id": config["method_id"], "config_sha256": config_hash,
        "network_response_used": False, "complete_identity_confirmed": False,
        "elapsed_seconds": time.perf_counter() - start,
    })
    pred_path = output / "predictions" / (row["source_id"] + ".json")
    save_json(pred_path, prediction)
    png_path = output / ("final_292" if phase == "all" else "pilot_png") / (row["source_id"] + ".png")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    render_four(image, fields, prediction, row, config).save(png_path, compress_level=1)
    return {"source_id": row["source_id"], "status": prediction.get("status"),
            "stop_reason": prediction.get("stop_reason"),
            "supported_extension_px": prediction.get("supported_extension_px"),
            "pred_path": str(pred_path), "pred_sha256": sha256(pred_path),
            "png_path": str(png_path), "png_sha256": sha256(png_path),
            "elapsed_seconds": time.perf_counter() - start, "failure": None}


def render_failure(row: dict, output: Path, config: dict, config_hash: str,
                   phase: str, error: str) -> dict:
    """Keep a visible failure in the gallery; never count it as detection."""
    image = read_bgr(Path(row["source_path"]))
    fields = {"blurred_bgr": image.copy(),
              "gradient_magnitude": np.zeros(image.shape[:2], np.float32)}
    prediction = {
        "source_id": row["source_id"], "source_group": row["source_group"],
        "camera_role": row["camera_role"], "source_use": row["source_use"],
        "source_path": row["source_path"], "source_sha256": row["source_sha256"],
        "e1": row["e1"], "entry_is_input_condition": True,
        "method_id": config["method_id"], "config_sha256": config_hash,
        "network_response_used": False, "complete_identity_confirmed": False,
        "status": "PROCESS_FAILURE", "stop_reason": "exception_saved_in_prediction",
        "failure": error, "points_xy": [], "supported_segments_xy": [],
        "endpoint_xy": None, "width_px": None, "supported_extension_px": 0.0,
        "step_records": [],
    }
    pred_path = output / "predictions" / (row["source_id"] + ".json")
    save_json(pred_path, prediction)
    png_path = output / ("final_292" if phase == "all" else "pilot_png") / (row["source_id"] + ".png")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    render_four(image, fields, prediction, row, config).save(png_path, compress_level=1)
    return {"source_id": row["source_id"], "status": "PROCESS_FAILURE",
            "stop_reason": "exception_saved_in_prediction", "supported_extension_px": 0.0,
            "pred_path": str(pred_path), "pred_sha256": sha256(pred_path),
            "png_path": str(png_path), "png_sha256": sha256(png_path),
            "elapsed_seconds": None, "failure": error}


def pilot_rows(index: dict) -> list[dict]:
    permitted = set(index["development_evaluation_policy"]["development_source_groups"])
    groups: dict[str, list[dict]] = {}
    for row in index["rows"]:
        if row["source_group"] in permitted:
            groups.setdefault(row["source_group"], []).append(row)
    return [rows[len(rows) // 2] for _, rows in sorted(groups.items())]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pilot", "all"), required=True)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_json(args.config)
    index = load_json(HERE / "inputs.json")
    rows = pilot_rows(index) if args.phase == "pilot" else index["rows"]
    if args.phase == "all" and (len(rows) != 292 or len({r["source_id"] for r in rows}) != 292):
        raise ValueError("full run must contain exactly 292 unique sources")
    if any(not r["e1"]["valid"] or r["e1"]["index"] != 0 for r in rows):
        raise ValueError("missing or changed E1 in frozen index")
    output = args.output / (args.run_name or ("pilot_001" if args.phase == "pilot" else "all_001"))
    if output.exists() and not args.resume:
        raise FileExistsError(f"refuse to overwrite run: {output}")
    output.mkdir(parents=True, exist_ok=args.resume)
    config_hash = sha256(args.config)
    start_record = {
        "phase": args.phase, "method_id": config["method_id"],
        "input_count": len(rows), "source_index_sha256": sha256(HERE / "inputs.json"),
        "config_sha256": config_hash, "core_sha256": sha256(HERE / "gradient_core.py"),
        "runner_sha256": sha256(Path(__file__)), "argv": sys.argv,
        "python_executable": sys.executable, "opencv_version": cv2.__version__,
        "numpy_version": np.__version__, "PIL_version": Image.__version__,
        "workers": args.workers, "network_response_used": False,
        "original_panel_xy": [0, PANEL_HEADER],
        "panel_size_wh": [4096, 3000],
        "canvas_size_wh": [2*4096+COLUMN_GAP, 2*3000+PANEL_HEADER+ROW_GAP+FOOTER],
        "panel_origins_xy": [[0, PANEL_HEADER], [4096+COLUMN_GAP, PANEL_HEADER],
                             [0, PANEL_HEADER+3000+ROW_GAP], [4096+COLUMN_GAP, PANEL_HEADER+3000+ROW_GAP]],
        "layout": {
            "board_size_wh": [2*4096+COLUMN_GAP, 2*3000+PANEL_HEADER+ROW_GAP+FOOTER],
            "panel_rects_xyxy": {
                "original": [0, PANEL_HEADER, 4096, PANEL_HEADER+3000],
                "blurred": [4096+COLUMN_GAP, PANEL_HEADER, 2*4096+COLUMN_GAP, PANEL_HEADER+3000],
                "gradient": [0, PANEL_HEADER+3000+ROW_GAP, 4096, PANEL_HEADER+6000+ROW_GAP],
                "overlay": [4096+COLUMN_GAP, PANEL_HEADER+3000+ROW_GAP,
                            2*4096+COLUMN_GAP, PANEL_HEADER+6000+ROW_GAP],
            },
        },
    }
    start_path = output / "run_start.json"
    if args.resume:
        old = load_json(start_path)
        for field in ("phase", "input_count", "source_index_sha256", "config_sha256", "core_sha256", "runner_sha256"):
            if old[field] != start_record[field]:
                raise ValueError(f"resume mismatch: {field}")
    else:
        save_json(start_path, start_record)
    done: dict[str, dict] = {}
    progress = output / "progress.jsonl"
    if args.resume and progress.is_file():
        for line in progress.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("failure") is None and Path(rec["png_path"]).is_file() and Path(rec["pred_path"]).is_file():
                done[rec["source_id"]] = rec
    remaining = [r for r in rows if r["source_id"] not in done]
    tick = time.perf_counter()
    with progress.open("a", encoding="utf-8") as log, concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(process_one, row, output, config, config_hash, args.phase): row for row in remaining}
        for job in concurrent.futures.as_completed(jobs):
            row = jobs[job]
            try:
                receipt = job.result()
            except Exception:
                error = traceback.format_exc()
                try:
                    receipt = render_failure(row, output, config, config_hash, args.phase, error)
                except Exception:
                    receipt = {"source_id": row["source_id"], "status": "PROCESS_FAILURE",
                               "failure": error + "\nFAILURE_RENDER: " + traceback.format_exc(),
                               "elapsed_seconds": None}
            done[row["source_id"]] = receipt
            log.write(json.dumps(receipt, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
            if len(done) % 10 == 0 or len(done) == len(rows):
                print(json.dumps({"phase": args.phase, "done": len(done), "total": len(rows),
                                  "failures": sum(bool(x.get("failure")) for x in done.values()),
                                  "elapsed_s": round(time.perf_counter() - tick, 1)}), flush=True)
    ordered = [done[row["source_id"]] for row in rows]
    counts: dict[str, int] = {}
    stops: dict[str, int] = {}
    for rec in ordered:
        counts[rec.get("status", "UNKNOWN")] = counts.get(rec.get("status", "UNKNOWN"), 0) + 1
        stops[rec.get("stop_reason", "unknown")] = stops.get(rec.get("stop_reason", "unknown"), 0) + 1
    receipt = {"start": start_record, "phase": args.phase, "input_count": len(rows),
               "completed_rows": len(ordered), "exit_code": 0 if all(not x.get("failure") for x in ordered) else 1,
               "status_counts": counts, "stop_counts": stops,
               "elapsed_seconds": time.perf_counter() - tick, "rows": ordered}
    save_json(output / "run_receipt.json", receipt)
    return receipt["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
