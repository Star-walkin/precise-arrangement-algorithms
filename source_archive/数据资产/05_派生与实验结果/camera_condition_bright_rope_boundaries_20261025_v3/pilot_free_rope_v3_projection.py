"""One-off diagnostic for semantic crossing offsets from historical free span."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from free_rope_v3 import _proposal

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
BASE = ROOT / "数据资产" / "05_派生与实验结果"
V1 = BASE / "camera_condition_bright_rope_boundaries_20261009_v1"
V3 = Path(__file__).resolve().parent
NAMES = {"ruler_left_01", "ruler_right_02",
         "L_run_stop_run_20260922_01__f002370__left",
         "R_run_stop_run_20260922_01__f000026__right"}
for row in csv.DictReader((V1 / "qa_plan.csv").open(encoding="utf-8-sig", newline="")):
    stem = Path(row["output_file"]).stem
    if stem not in NAMES:
        continue
    role = row["camera_role"]
    camera = role + "_camera"
    source = BASE / "camera_condition_images_20261009_v2" / row["output_file"]
    mask = V1 / "all_001" / "roi" / "masks" / camera / (stem + ".png")
    rec = V1 / "all_001" / "boundaries" / "records" / camera / (stem + ".json")
    diag = V3 / "pilot_free_rope_v3_006" / "diagnostics" / camera / (stem + ".json")
    image = cv2.imdecode(np.fromfile(str(source), np.uint8), cv2.IMREAD_COLOR)
    roi = cv2.imdecode(np.fromfile(str(mask), np.uint8), cv2.IMREAD_GRAYSCALE)
    pitch = json.loads(rec.read_text(encoding="utf-8"))["record"]["boundaries"]["pitch_px"]
    data = json.loads(diag.read_text(encoding="utf-8"))
    if role == "left":
        image, roi = cv2.flip(image, 1), cv2.flip(roi, 1)
    proposal, _, _ = _proposal(image, roi, role, pitch)
    segment = np.asarray(data["history_observed_segment_canonical_xy"], np.float32)
    m = (segment[1, 1] - segment[0, 1]) / (segment[1, 0] - segment[0, 0])
    yy, xx = np.nonzero((proposal > 0) & (roi > 0))
    q = yy - (segment[0, 1] + m * (xx - segment[0, 0]))
    print(stem, "overlap", len(q), "q_range", np.percentile(q, [0, 5, 25, 50, 75, 95, 100]).round(1).tolist()
          if len(q) else [], "roi_x", [int(xx.min()), int(xx.max())] if len(q) else [])
