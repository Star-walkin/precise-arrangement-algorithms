"""Run the fixed, open 38-image QA list with v3 occlusion evidence only."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "AGENTS.md").is_file())
BASE = ROOT / "数据资产" / "05_派生与实验结果"
V1 = BASE / "camera_condition_bright_rope_boundaries_20261009_v1"
IMAGES = BASE / "camera_condition_images_20261009_v2"
sys.path.insert(0, str(HERE))
from free_rope_v3 import estimate_occlusion  # noqa: E402


def load_image(path: Path, mode: int) -> np.ndarray:
    raw = np.fromfile(str(path), np.uint8)
    image = cv2.imdecode(raw, mode)
    if image is None:
        raise RuntimeError(f"cannot decode {path}")
    return image


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"cannot encode {path}")
    encoded.tofile(str(path))


def main() -> None:
    out = HERE / os.environ.get("PILOT_FREE_ROPE_V3_RUN", "pilot_free_rope_v3_001")
    out.mkdir(parents=True, exist_ok=True)
    names = set(sys.argv[1:])
    rows = list(csv.DictReader((V1 / "qa_plan.csv").open(encoding="utf-8-sig", newline="")))
    summary = []
    for row in rows:
        rel = Path(row["output_file"])
        stem = rel.stem
        if names and stem not in names:
            continue
        role = row["camera_role"]
        camera_dir = role + "_camera"
        source = IMAGES / rel
        record_file = V1 / "all_001" / "boundaries" / "records" / camera_dir / (stem + ".json")
        roi_file = V1 / "all_001" / "roi" / "masks" / camera_dir / (stem + ".png")
        record = json.loads(record_file.read_text(encoding="utf-8"))
        pitch = record["record"]["boundaries"]["pitch_px"]
        image = load_image(source, cv2.IMREAD_COLOR)
        roi = load_image(roi_file, cv2.IMREAD_GRAYSCALE)
        t0 = time.perf_counter()
        try:
            support, unknown, diag = estimate_occlusion(image, roi, role, pitch)
            assert support.shape == roi.shape == unknown.shape
            assert not np.any((support > 0) & (unknown > 0))
            assert not np.any((support > 0) & (roi == 0))
            assert not np.any((unknown > 0) & (roi == 0))
            assert set(np.unique(support)).issubset({0, 255})
            assert set(np.unique(unknown)).issubset({0, 255})
            status = "ok"
        except Exception as exc:
            diag = {"error": repr(exc)}
            status = "error"
            support = np.zeros_like(roi)
            unknown = np.zeros_like(roi)
        elapsed = time.perf_counter() - t0
        summary.append({"name": stem, "role": role, "status": status,
                        "elapsed_sec": round(elapsed, 3),
                        "history_called": diag.get("history_module_called"),
                        "history_valid": diag.get("history_valid"),
                        "history_reason": diag.get("history_reason"),
                        "v3_status": diag.get("status"),
                        "support_pixels": int(np.count_nonzero(support)),
                        "unknown_pixels": int(np.count_nonzero(unknown))})
        save_png(out / "supported" / camera_dir / (stem + ".png"), support)
        save_png(out / "unknown" / camera_dir / (stem + ".png"), unknown)
        (out / "diagnostics" / camera_dir).mkdir(parents=True, exist_ok=True)
        (out / "diagnostics" / camera_dir / (stem + ".json")).write_text(
            json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
        # Overview only; ROI and image are never altered on disk.
        small = cv2.resize(image, (round(image.shape[1] * .4), round(image.shape[0] * .4)))
        sup_small = cv2.resize(support, small.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
        unk_small = cv2.resize(unknown, small.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
        overlay = small.copy()
        overlay[sup_small] = (.45 * overlay[sup_small] + .55 * np.array([0, 180, 255])).astype(np.uint8)
        overlay[unk_small] = (.45 * overlay[unk_small] + .55 * np.array([255, 90, 30])).astype(np.uint8)
        save_png(out / "qa" / camera_dir / (stem + ".png"), overlay)
        print(json.dumps(summary[-1], ensure_ascii=False), flush=True)
    (out / "summary.json").write_text(json.dumps({
        "input_list": str(V1 / "qa_plan.csv"), "rows": summary,
        "source_sha256": hashlib.sha256((HERE / "free_rope_v3.py").read_bytes()).hexdigest(),
        "device_connected": False, "reserved_data_used": False},
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
