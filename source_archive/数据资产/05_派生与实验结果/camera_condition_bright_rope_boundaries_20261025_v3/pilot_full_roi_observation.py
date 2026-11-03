"""仅在固定 38 张开放开发 QA 图上运行 v3 全 ROI 黄色候选试验。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from full_roi_observation import detect_full_roi


HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent
DATASET = ASSETS / "camera_condition_images_20261009_v2"
V1 = ASSETS / "camera_condition_bright_rope_boundaries_20261009_v1"
V2 = ASSETS / "camera_condition_bright_rope_boundaries_20261023_v2"
QA_PLAN = V1 / "qa_plan.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path, flag: int) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), flag)
    if image is None:
        raise ValueError(f"Cannot decode: {path}")
    return image


def write(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix,
                            image, [cv2.IMWRITE_JPEG_QUALITY, 93]
                            if path.suffix.lower() == ".jpg" else [])
    if not ok:
        raise ValueError(f"Cannot encode: {path}")
    data.tofile(str(path))


def overlay(image: np.ndarray, roi: np.ndarray, yellow: np.ndarray,
            old: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(roi)
    x0, x1 = max(0, int(xs.min())-70), min(image.shape[1], int(xs.max())+71)
    y0, y1 = max(0, int(ys.min())-70), min(image.shape[0], int(ys.max())+71)
    panel = image[y0:y1, x0:x1].copy()
    panel[old[y0:y1, x0:x1] > 0] = (255, 255, 0)  # 青色=原 v2 黄色层，仅供对照
    panel[yellow[y0:y1, x0:x1] > 0] = (0, 255, 255)  # 黄色=独立全 ROI 新扫描
    edge = cv2.morphologyEx(roi[y0:y1, x0:x1], cv2.MORPH_GRADIENT,
                            np.ones((3, 3), np.uint8))
    panel[edge > 0] = (0, 255, 0)
    scale = min(1.0, 1500.0 / max(1, panel.shape[1]))
    if scale < 1:
        panel = cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 35), (0, 0, 0), -1)
    cv2.putText(panel, "yellow = independent v3   cyan = existing v2   green = ROI",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0,
                        help="zero-based offset within the fixed 38-image QA plan")
    parser.add_argument("--limit", type=int, default=38)
    args = parser.parse_args()
    if not 0 <= args.start < 38 or not 1 <= args.limit <= 38 - args.start:
        raise ValueError("start/limit must stay within the fixed 38 QA images")
    if args.output.exists():
        raise FileExistsError(args.output)
    with QA_PLAN.open("r", encoding="utf-8-sig", newline="") as stream:
        qa_rows = list(csv.DictReader(stream))
    if len(qa_rows) != 38 or sha256(QA_PLAN) != \
            "8b3b73673ebddda9a3bf1ee6fd45ac2fd3bda958cba472be78c27dcea351d913":
        raise ValueError("Fixed QA plan changed")
    args.output.mkdir(parents=True)
    context = {
        "argv": sys.argv, "qa_plan_sha256": sha256(QA_PLAN),
        "source_sha256": sha256(HERE / "full_roi_observation.py"),
        "purpose": "open development only; independent visual candidates, no accuracy claim",
    }
    (args.output / "run_context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    records: list[dict] = []
    for i, row in enumerate(qa_rows[args.start:args.start + args.limit], 1):
        name = Path(row["output_file"]).name
        role = row["camera_role"] + "_camera"
        result: dict = {"output_file": row["output_file"], "camera_role": row["camera_role"]}
        tic = time.perf_counter()
        try:
            if "20260925" in row["output_file"]:
                raise ValueError("Reserved input in QA plan")
            source = DATASET / row["output_file"]
            if sha256(source) != row["input_sha256"]:
                raise ValueError("Source image hash differs from fixed QA plan")
            image = read(source, cv2.IMREAD_COLOR)
            roi = read(V1 / "all_001" / "roi" / "masks" / role / name,
                       cv2.IMREAD_GRAYSCALE)
            occlusion = read(V2 / "all_002" / "boundaries" /
                             "incoming_rope_occlusion" / role / name,
                             cv2.IMREAD_GRAYSCALE)
            unknown = read(V2 / "all_002" / "boundaries" /
                           "unknown_exclusion" / role / name,
                           cv2.IMREAD_GRAYSCALE)
            exclusion = np.maximum(occlusion, unknown)
            old = read(V2 / "all_002" / "boundaries" /
                       "observed" / role / name, cv2.IMREAD_GRAYSCALE)
            v1_record = json.loads((V1 / "all_001" / "boundaries" /
                                    "records" / role / name.replace(".png", ".json")
                                    ).read_text(encoding="utf-8"))
            pitch = v1_record["record"]["boundaries"]["pitch_px"]
            if pitch is None:
                result["status"] = "skipped_no_pitch"
            else:
                yellow, lines, diag = detect_full_roi(
                    image, roi, exclusion, float(pitch))
                # 像素审计只验证输出约束，不等于逐根语义正确。
                if yellow.shape != roi.shape or not set(np.unique(yellow)).issubset({0, 255}):
                    raise AssertionError("Yellow mask shape or 0/255 convention failed")
                if np.any((yellow > 0) & ((roi == 0) | (exclusion > 0))):
                    raise AssertionError("Yellow pixel outside ROI or inside exclusion")
                for line in lines:
                    for x, y in line["observed_points_xy"]:
                        if yellow[y, x] != 255:
                            raise AssertionError("Recorded observation absent from raster")
                # 旧黄点只是既有算法对照，并非人工真值；这两项不能叫准确率。
                near_px = max(3, int(round(.20 * float(pitch))))
                old_distance = cv2.distanceTransform((old == 0).astype(np.uint8),
                                                     cv2.DIST_L2, 3)
                new_distance = cv2.distanceTransform((yellow == 0).astype(np.uint8),
                                                     cv2.DIST_L2, 3)
                new_near_old = float(np.mean(old_distance[yellow > 0] <= near_px)) \
                    if np.any(yellow) else 0.0
                old_near_new = float(np.mean(new_distance[old > 0] <= near_px)) \
                    if np.any(old) else 0.0
                write(args.output / "observed" / role / name, yellow)
                write(args.output / "qa" / role / name.replace(".png", ".jpg"),
                      overlay(image, roi, yellow, old))
                dest = args.output / "records" / role / name.replace(".png", ".json")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps({"source": row["output_file"],
                                            "lines": lines, "diagnostics": diag},
                                           ensure_ascii=False) + "\n", encoding="utf-8")
                result.update(status="completed", pitch_px=pitch,
                              lines=len(lines), observed_pixels=diag["observed_pixels"],
                              raw_candidates=diag["raw_candidates"]["kept"],
                              raw_tracks=diag["raw_tracks"],
                              new_near_v2_fraction=round(new_near_old, 4),
                              v2_near_new_fraction=round(old_near_new, 4),
                              comparison_radius_px=near_px)
        except Exception as exc:
            result.update(status="error", error=f"{type(exc).__name__}: {exc}")
        result["elapsed_s"] = round(time.perf_counter()-tic, 3)
        records.append(result)
        print(f"{i}/{args.limit} {name}: {result['status']} "
              f"lines={result.get('lines', 0)} t={result['elapsed_s']}s", flush=True)
    summary = {"planned": args.limit, "qa_start_zero_based": args.start,
               "completed": sum(r["status"] == "completed" for r in records),
               "errors": sum(r["status"] == "error" for r in records),
               "line_count": sum(r.get("lines", 0) for r in records),
               "interpretation": "candidate dark valleys from full ROI; no truth labels"}
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = list(dict.fromkeys(key for record in records for key in record))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
