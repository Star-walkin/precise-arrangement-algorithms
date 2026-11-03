"""Run the independent original-image G0/G1 geometry experiment.

The only prediction inputs are source RGB pixels and the frozen entry.  This
module deliberately does not import the historical neural inference modules.
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

from geometry_core import compute_fields, track

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = Path(r"C:\precise_arrangement_results\geometry_direction_20261103_v1")
HEADER, FOOTER, GAP, SIDE = 148, 172, 32, 142
Image.MAX_IMAGE_PIXELS = 150_000_000


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_color(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"image decode failed: {path}")
    return image


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", size)


def _line(draw: ImageDraw.ImageDraw, points, origin, style: str) -> None:
    if len(points) < 2:
        return
    shifted = [(float(x) + origin[0], float(y) + origin[1]) for x, y in points]
    if style == "supported":
        draw.line(shifted, fill=(0, 0, 0), width=5, joint="curve")
        draw.line(shifted, fill=(255, 255, 255), width=2, joint="curve")
    else:
        for i in range(len(shifted) - 1):
            if i % 2 == 0:
                draw.line(shifted[i:i + 2], fill=(0, 0, 0), width=5)
                draw.line(shifted[i:i + 2], fill=(255, 193, 77), width=2)


def _draw_selected(draw: ImageDraw.ImageDraw, result: dict, origin: tuple[int, int]) -> None:
    chosen = result.get("selected_candidate_id")
    if chosen is None:
        return
    row = next((c for c in result.get("candidates", []) if c.get("id") == chosen), None)
    if row is None:
        return
    points = row.get("points_xy", [])
    states = row.get("point_status", [])
    if len(points) != len(states):
        raise ValueError("selected points/status length mismatch")
    for i in range(1, len(points)):
        if states[i] in ("entry_condition", "input_entry"):
            continue
        style = "supported" if states[i] in ("visible_supported", "image_supported") else "unsupported"
        _line(draw, points[i - 1:i + 1], origin, style)


def _right_panel(fields: dict, shape_hw: tuple[int, int], display: dict) -> np.ndarray:
    h, w = shape_hw
    raw = np.asarray(fields["geometry_response"], np.float32)
    valid = np.asarray(fields["valid_domain"], bool)
    if raw.shape != valid.shape:
        raise ValueError("response/valid shape mismatch")
    lo, hi = display["response_range"]
    if not np.isfinite(raw[valid]).all():
        raise ValueError("valid geometry response contains nonfinite values")
    value = np.rint(np.clip((np.where(valid, raw, lo) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    color = cv2.cvtColor(cv2.applyColorMap(value, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    color[~valid] = display["neutral_rgb"]
    if color.shape[:2] != (h, w):
        color = cv2.resize(color, (w, h), interpolation=cv2.INTER_LINEAR)
        full_valid = cv2.resize(valid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        color[~full_valid] = display["neutral_rgb"]
    return color


def render(image_bgr: np.ndarray, fields: dict, result: dict, row: dict, config: dict) -> Image.Image:
    h, w = image_bgr.shape[:2]
    original = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    right = _right_panel(fields, (h, w), config["display"])
    board = Image.new("RGB", (2 * w + GAP + SIDE, h + HEADER + FOOTER), (247, 248, 250))
    board.paste(Image.fromarray(original), (0, HEADER))
    right_x = w + GAP
    board.paste(Image.fromarray(right), (right_x, HEADER))
    draw = ImageDraw.Draw(board)
    _draw_selected(draw, result, (right_x, HEADER))
    seed = row.get("e1", {}).get("xy")
    if seed and len(seed) >= 2:
        draw.line([(right_x + x, HEADER + y) for x, y in seed], fill=(0, 0, 0), width=6)
        draw.line([(right_x + x, HEADER + y) for x, y in seed], fill=(0, 255, 230), width=3)
    draw.text((20, 14), "原始彩色图  " + row["source_id"], font=font(38), fill=(0, 0, 0))
    draw.text((20, 80), f"原图像素 {w}×{h}，左栏无绘制", font=font(29), fill=(0, 0, 0))
    draw.text((right_x + 20, 14), "G1 原图几何响应及活动绳中心候选", font=font(38), fill=(0, 0, 0))
    draw.text((right_x + 20, 80), "几何响应，非概率；固定色标；入口是输入条件", font=font(29), fill=(0, 0, 0))
    bar_top = HEADER + 500
    bar = cv2.cvtColor(cv2.applyColorMap(np.linspace(255, 0, 1400, dtype=np.uint8)[:, None], cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    bar = np.repeat(bar, 36, axis=1)
    board.paste(Image.fromarray(bar), (2 * w + GAP + 12, bar_top))
    for y, label in [(bar_top, "高"), (bar_top + 700, "中"), (bar_top + 1360, "低")]:
        draw.text((2 * w + GAP + 53, y), label, font=font(27), fill=(0, 0, 0))
    text_y = h + HEADER + 11
    e1 = row.get("e1", {})
    entry_text = f"E1 来源：{e1.get('entry_id', 'unknown')} / {e1.get('source', 'unknown')}"
    if len(entry_text) > 175:
        entry_text = entry_text[:175] + "…"
    draw.text((20, text_y), entry_text, font=font(27), fill=(0, 0, 0))
    draw.text((20, text_y + 45), f"方法 {config['method_id']}  |  状态 {result.get('status', 'unknown')}  |  停止 {result.get('stop_reason', 'unknown')}", font=font(30), fill=(0, 0, 0))
    draw.text((20, text_y + 93), "右栏：白实线=原图几何支持候选；黄虚线=有限无支持段；青线=固定输入入口；灰色=未计算区域。身份未独立确认。", font=font(28), fill=(0, 0, 0))
    return board


def _serializable(value):
    if isinstance(value, dict):
        return {str(k): _serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _prediction(row: dict, result: dict, arm: str, config_hash: str, elapsed: float) -> dict:
    r = _serializable(result)
    selected = r.get("selected")
    r["selected_candidate_id"] = selected.get("id") if isinstance(selected, dict) else None
    r.update({
        "id": row["source_id"], "source_id": row["source_id"], "source_group": row.get("source_group"),
        "image_path": row["source_path"], "image_sha256": row["source_sha256"],
        "input_seed_xy": row.get("e1", {}).get("xy") if row.get("e1", {}).get("valid") else [],
        "entry_source": row.get("e1", {}).get("source"), "entry_id": row.get("e1", {}).get("entry_id"),
        "entry_is_input_condition": True, "arm": arm, "method_id": "IR-GEOM-G1.0.0",
        "config_sha256": config_hash, "elapsed_seconds": elapsed,
        "network_response_used": False, "complete_identity_confirmed": False,
    })
    return r


def original_background_domain(image_bgr: np.ndarray, config: dict) -> np.ndarray:
    """Conservative original-pixel texture domain for background direction only.

    This is never a candidate mask. Texture and low chroma identify regions
    where a spool direction *may* be measured. The bilateral direction checks
    in geometry_core still decide whether a local direction is known.
    """
    scale = float(config["field_scale"])
    small = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    x = small.astype(np.float32) / 255.0
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mean = cv2.GaussianBlur(gray, (0, 0), 5.0)
    var = cv2.GaussianBlur((gray - mean) ** 2, (0, 0), 5.0)
    chroma = x.max(axis=2) - x.min(axis=2)
    p = config["background_domain"]
    domain = (var >= p["minimum_local_variance"]) & (chroma <= p["maximum_chroma"]) & (gray >= p["minimum_gray"])
    start_y = int(np.floor(config["bbox_xyxy"][1] * scale))
    domain[:start_y] = False
    domain = cv2.morphologyEx(domain.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.resize(domain, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)


def process(row: dict, output: Path, config: dict, config_hash: str, kind: str) -> dict:
    cv2.setNumThreads(1)
    tick = time.perf_counter()
    image_path = Path(row["source_path"])
    if sha(image_path) != row["source_sha256"]:
        raise ValueError("source SHA mismatch: " + str(image_path))
    image = read_color(image_path)
    if list(image.shape[:2]) != row["shape_hw"]:
        raise ValueError("source dimensions mismatch: " + row["source_id"])
    background = original_background_domain(image, config)
    fields = compute_fields(image, dict(config, background_surface_mask=background))
    record_key = row["source_id"] + ("__" + row["mode"] if kind == "eval" else "")
    evidence_path = output / "evidence_float32" / (record_key + ".npz")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(evidence_path, geometry_response=np.asarray(fields["geometry_response"], np.float32),
                        valid_domain=np.asarray(fields["valid_domain"], np.uint8),
                        direction_cos2=np.asarray(fields["direction_cos2"], np.float32),
                        direction_sin2=np.asarray(fields["direction_sin2"], np.float32),
                        direction_confidence=np.asarray(fields["direction_confidence"], np.float32),
                        field_scale=np.asarray(fields["field_scale"], np.float32))
    predictions = {}
    for arm, use_direction in [("G0", False), ("G1", True)]:
        t = time.perf_counter()
        result = track(fields, image, row["e1"]["xy"] if row.get("e1", {}).get("valid") else [], config, use_direction)
        predictions[arm] = _prediction(row, result, arm, config_hash, time.perf_counter() - t)
        pred_dir = output / "predictions" / arm
        if kind == "eval":
            pred_dir = pred_dir / row["mode"]
        pred_dir.mkdir(parents=True, exist_ok=True)
        dump(pred_dir / (row["source_id"] + ".json"), predictions[arm])
    if kind in ("pilot", "all"):
        png_dir = output / ("final_292" if kind == "all" else "pilot_png")
        png_dir.mkdir(parents=True, exist_ok=True)
        png = png_dir / (row["source_id"] + ".png")
        board = render(image, fields, predictions["G1"], row, config)
        board.save(png, compress_level=1)
        output_sha = sha(png)
    else:
        png, output_sha = None, None
    return {"source_id": row["source_id"], "record_key": record_key, "source_sha256": row["source_sha256"], "status_G0": predictions["G0"].get("status"),
            "status_G1": predictions["G1"].get("status"), "stop_G0": predictions["G0"].get("stop_reason"),
            "stop_G1": predictions["G1"].get("stop_reason"), "evidence_path": str(evidence_path),
            "evidence_sha256": sha(evidence_path), "png_path": str(png) if png else None,
            "png_sha256": output_sha, "elapsed_seconds": time.perf_counter() - tick, "failure": None}


def failure(row: dict, output: Path, config: dict, config_hash: str, kind: str, error: str) -> dict:
    # A failed single image still has a truthful neutral-heat result image.
    image = read_color(Path(row["source_path"]))
    h, w = image.shape[:2]
    fields = {"geometry_response": np.zeros((h, w), np.float32), "valid_domain": np.zeros((h, w), bool)}
    result = {"selected_candidate_id": None, "candidates": [], "status": "single_image_failure", "stop_reason": "exception_saved_in_log"}
    for arm in ("G0", "G1"):
        pred_dir = output / "predictions" / arm
        if kind == "eval":
            pred_dir = pred_dir / row["mode"]
        pred_dir.mkdir(parents=True, exist_ok=True)
        dump(pred_dir / (row["source_id"] + ".json"), _prediction(row, result, arm, config_hash, 0.0))
    png = None
    if kind in ("pilot", "all"):
        png_dir = output / ("final_292" if kind == "all" else "pilot_png")
        png_dir.mkdir(parents=True, exist_ok=True)
        png = png_dir / (row["source_id"] + ".png")
        render(image, fields, result, row, config).save(png, compress_level=1)
    return {"source_id": row["source_id"], "record_key": row["source_id"] + ("__" + row["mode"] if kind == "eval" else ""), "failure": error, "status_G0": "single_image_failure", "status_G1": "single_image_failure",
            "png_path": str(png) if png else None, "png_sha256": sha(png) if png else None, "elapsed_seconds": None}


def pilot_rows(index: dict) -> list[dict]:
    allowed = set(index["development_evaluation_policy"]["development_source_groups"])
    groups = {}
    for row in index["rows"]:
        if row["source_group"] in allowed:
            groups.setdefault(row["source_group"], []).append(row)
    selected = [v[len(v) // 2] for _, v in sorted(groups.items())]
    return selected


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["pilot", "all", "eval"], required=True)
    p.add_argument("--config", type=Path, default=HERE / "config.json")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--eval-inputs", type=Path, default=HERE / "evaluation_inputs.json")
    p.add_argument("--run-name")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    cfg = load(a.config)
    config_hash = sha(a.config)
    index = load(HERE / "inputs.json")
    if a.phase == "pilot":
        rows = pilot_rows(index)
        output = a.output / (a.run_name or "pilot_001")
    elif a.phase == "all":
        rows = index["rows"]
        output = a.output / (a.run_name or "all_001")
    else:
        ev = load(a.eval_inputs)
        rows = []
        for c in ev["conditions"]:
            rows.append({"source_id": c["id"], "source_group": c["eval_group"], "source_path": c["source_path"],
                         "source_sha256": c["image_sha256"], "shape_hw": c["image_shape_hw"],
                         "bbox_original_xyxy": c["bbox_original_xyxy"], "mode": c["mode"],
                         "e1": {"valid": c["entry_available"], "xy": c["input_seed_xy"],
                                "source": c["seed_source"], "entry_id": c["seed_origin"],
                                "failure_reasons": [] if c["entry_available"] else ["frozen_auto_entry_missing"]}})
        output = a.output / (a.run_name or "eval_001")
    if output.exists() and not a.resume:
        raise FileExistsError(f"Refuse to overwrite existing run: {output}")
    output.mkdir(parents=True, exist_ok=a.resume)
    start_path = output / "run_start.json"
    expected_start = {"phase": a.phase, "config_path": str(a.config.resolve()), "config_sha256": config_hash,
                      "inputs_sha256": sha(HERE / "inputs.json") if a.phase != "eval" else sha(a.eval_inputs),
                      "row_count": len(rows), "workers": a.workers, "argv": sys.argv, "network_response_used": False,
                      "core_sha256": sha(HERE / "geometry_core.py"), "runner_sha256": sha(Path(__file__))}
    if a.resume:
        prior = load(start_path)
        for key in ("phase", "config_sha256", "inputs_sha256", "row_count", "core_sha256", "runner_sha256"):
            if prior[key] != expected_start[key]:
                raise ValueError(f"Resume source/config mismatch: {key}")
    else:
        dump(start_path, expected_start)
    tick = time.perf_counter()
    receipts = []
    progress = output / "progress.jsonl"
    completed = {}
    if a.resume and progress.exists():
        for line in progress.read_text(encoding="utf-8").splitlines():
            old = json.loads(line)
            key = old.get("record_key")
            if old.get("failure") is None and key:
                png = old.get("png_path")
                evidence = old.get("evidence_path")
                if (not png or Path(png).is_file()) and evidence and Path(evidence).is_file():
                    completed[key] = old
    pending_rows = [row for row in rows if row["source_id"] + ("__" + row["mode"] if a.phase == "eval" else "") not in completed]
    receipts.extend(completed.values())
    with progress.open("a", encoding="utf-8") as log, concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        future = {pool.submit(process, row, output, cfg, config_hash, "eval" if a.phase == "eval" else a.phase): row for row in pending_rows}
        for f in concurrent.futures.as_completed(future):
            row = future[f]
            try:
                receipt = f.result()
            except Exception:
                err = traceback.format_exc()
                try:
                    receipt = failure(row, output, cfg, config_hash, "eval" if a.phase == "eval" else a.phase, err)
                except Exception:
                    receipt = {"source_id": row["source_id"], "failure": err + "\nFAILURE_RENDER: " + traceback.format_exc()}
            receipts.append(receipt)
            log.write(json.dumps(receipt, ensure_ascii=False) + "\n")
            log.flush()
            if len(receipts) % 10 == 0 or len(receipts) == len(rows):
                print(json.dumps({"phase": a.phase, "done": len(receipts), "total": len(rows), "failures": sum(bool(x.get("failure")) for x in receipts), "elapsed_s": round(time.perf_counter() - tick, 1)}), flush=True)
    by_id = {r["record_key"]: r for r in receipts}
    ordered = [by_id[r["source_id"] + ("__" + r["mode"] if a.phase == "eval" else "")] for r in rows]
    result = {"phase": a.phase, "exit_code": 0 if all(not r.get("failure") for r in ordered) else 1,
              "completed_rows": len(ordered), "input_rows": len(rows), "rows": ordered, "elapsed_seconds": time.perf_counter() - tick,
              "network_response_used": False, "config_sha256": config_hash, "script_sha256": sha(Path(__file__))}
    dump(output / "run_receipt.json", result)
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
