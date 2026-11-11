"""Two permitted development first-frame checks of lc1_surface.

The Hough line used here only supplies an explicit API smoke-test active_y;
it is not the user's S1 module or a validation of locked-column estimation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from lc1_surface import SurfaceObserver

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
VIDEO_BASE = ROOT.parents[3] / "双相机同步"


def read_image(path, flags=0):
    image = cv2.imdecode(np.fromfile(path, np.uint8), flags)
    if image is None:
        raise ValueError(f"Cannot read {path}")
    return image


def probe_line(image, mask, d):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray[1600:], 70, 170)
    raw = cv2.HoughLinesP(edge, 1, np.pi / 720, 100,
                          minLineLength=400, maxLineGap=120)
    choices = []
    for segment in np.asarray(raw if raw is not None else []).reshape(-1, 4):
        x0, y0, x1, y1 = [float(v) for v in segment]
        y0 += 1600
        y1 += 1600
        if x1 < x0:
            x0, x1, y0, y1 = x1, x0, y1, y0
        if x1 - x0 < 300:
            continue
        slope = (y1 - y0) / (x1 - x0)
        if not -.55 <= slope <= -.06 or x1 < .91 * image.shape[1]:
            continue
        p = np.rint(np.c_[np.linspace(x0, x1, 100), np.linspace(y0, y1, 100)]).astype(int)
        inside = (p[:, 0] >= 0) & (p[:, 0] < mask.shape[1]) & (p[:, 1] >= 0) & (p[:, 1] < mask.shape[0])
        surface_fraction = float(np.mean(mask[p[inside, 1], p[inside, 0]] > 0))
        y = y0 + slope * (d - x0)
        if not 1700 < y < image.shape[0]:
            continue
        choices.append(dict(points_xy=[[x0, y0], [x1, y1]], y_at_d=y,
                            rank=float(x1 - x0) + 300 * (1 - surface_fraction),
                            surface_fraction=surface_fraction))
    choices.sort(key=lambda value: -value["rank"])
    return choices[0] if choices else None


def overlay(image, output):
    frame = image.copy()
    for line in output["fit_polylines_xy"]:
        if len(line) >= 2:
            cv2.polylines(frame, [np.rint(line).astype(np.int32)], False, (200, 120, 30), 1)
    for line in output["boundary_polylines_xy"]:
        if len(line) >= 2:
            cv2.polylines(frame, [np.rint(line).astype(np.int32)], False, (0, 235, 255), 2)
    x = int(round(output["measurement_x"]))
    cv2.line(frame, (x, 1600), (x, image.shape[0] - 1), (40, 220, 40), 2)
    if output["edge_y"] is not None:
        y = int(round(output["edge_y"]))
        cv2.line(frame, (max(0, x - 120), y), (min(image.shape[1] - 1, x + 120), y), (255, 60, 220), 3)
    for y in output["crossings_y"]:
        cv2.circle(frame, (x, int(round(y))), 4, (0, 180, 255), -1)
    cv2.putText(frame, "F3 observation core; yellow=raw support; blue=fit", (80, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(frame, f"valid={output['count_valid']} grooves={output['count_observed_grooves']} "
                f"distance={output['continuous_pitch_distance']} candidate={output['continuous_pitch_distance_candidate']}",
                (80, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255, 255, 255), 2)
    return frame


def main():
    cv2.setNumThreads(1)
    destination = ROOT / "surface_smoke_results"
    destination.mkdir(exist_ok=True)
    rows = []
    for session, role in (("dual_20260923_113305", "right"), ("dual_20260924_111500", "left")):
        serial = "DA9626975" if role == "right" else "DA9627065"
        video = VIDEO_BASE / session / f"{role}_{serial}.avi"
        cap = cv2.VideoCapture(str(video))
        okay, image = cap.read()
        cap.release()
        if not okay:
            raise ValueError(f"Cannot decode permitted first frame: {video}")
        stem = f"{session}__pair0015__{role}"
        mask_path = BASE / "camera_condition_sam2_20261009_v1/all_001/body_branches_002/masks" / f"{role}_camera" / (stem + ".png")
        record_path = BASE / "camera_condition_bright_rope_boundaries_20261023_v2/all_002/boundaries/records" / f"{role}_camera" / (stem + ".json")
        mask = read_image(mask_path)
        pitch = float(json.loads(record_path.read_text(encoding="utf-8"))["record"]["pitch_px"])
        if role == "left":
            image, mask = cv2.flip(image, 1), cv2.flip(mask, 1)
        columns = np.flatnonzero(np.any(mask > 0, axis=0))
        d = float(np.median(columns))
        line = probe_line(image, mask, d)
        active_y = line["y_at_d"] if line else None
        observer = SurfaceObserver(mask, pitch)
        tick = time.perf_counter()
        output = observer.step(image, d=d, active_y=active_y)
        elapsed = time.perf_counter() - tick
        value = dict(session=session, camera_role=role, video=str(video), frame=0,
                     analysis_horizontal_flip=role == "left", surface_mask_source=str(mask_path),
                     pitch_source=str(record_path), pitch_px=pitch,
                     d=d, d_source="mask_mid_column_for_API_smoke_only_not_locked_bend",
                     active_y=active_y, active_y_source="independent_Hough_smoke_probe_not_S1",
                     line_probe=line, seconds=elapsed, output=output)
        (destination / f"{session}_{role}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        preview = overlay(image, output)
        cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 94])[1].tofile(destination / f"{session}_{role}.jpg")
        rows.append(dict(session=session, role=role, mirrored=role == "left", d=d, active_y=active_y,
                         seconds=elapsed, core_seconds=output["diagnostics"]["core_seconds"],
                         boundary_segments=len(output["boundary_polylines_xy"]),
                         fit_curves=len(output["fit_polylines_xy"]), crossings=len(output["crossings_y"]),
                         edge_y=output["edge_y"], edge_source=output["edge_source"],
                         edge_confidence=output["edge_confidence"],
                         count_observed_grooves=output["count_observed_grooves"],
                         continuous_pitch_distance=output["continuous_pitch_distance"],
                         continuous_pitch_distance_candidate=output["continuous_pitch_distance_candidate"],
                         count_valid=output["count_valid"], reasons=output["reasons"]))
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    receipt = dict(status="executed", scope="Two permitted real first-frame API smokes; no S1/d validation",
                   lc1_surface_sha256=hashlib.sha256((ROOT / "lc1_surface.py").read_bytes()).hexdigest(),
                   rows=rows, reserved_media_read=False, neural_model_calls=0,
                   vendor_source_modified=False)
    (destination / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
