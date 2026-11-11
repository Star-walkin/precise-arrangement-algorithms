"""Make a navigation CSV and descriptive counts from an already finished run.

No reference labels are loaded, so these counts cannot be accuracy metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_RUN = Path(r"C:\precise_arrangement_results\straight_gradient_20261104_v1\all_001")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def describe(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    a = np.asarray(values, dtype=float)
    return {"count": len(a), "min": float(a.min()), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)), "max": float(a.max())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    index = read(HERE / "inputs.json")
    receipt = read(args.run / "run_receipt.json")
    if receipt["exit_code"] != 0 or receipt["completed_rows"] != 292:
        raise ValueError("full run is incomplete or contains processing failures")
    rows = []
    for row in index["rows"]:
        source_id = row["source_id"]
        pred_path = args.run / "predictions" / (source_id + ".json")
        pred = read(pred_path)
        png_path = args.run / "final_292" / (source_id + ".png")
        if pred["source_id"] != source_id or not png_path.is_file():
            raise ValueError(f"missing or mismatched result: {source_id}")
        rows.append({
            "order_index": row["order_index"], "source_id": source_id,
            "camera_role": row["camera_role"], "source_group": row["source_group"],
            "source_use": row["source_use"], "e1_entry_id": row["e1"]["entry_id"],
            "status": pred["status"], "stop_reason": pred["stop_reason"],
            "supported_extension_px": pred.get("supported_extension_px", 0.0),
            "measured_width_px": pred.get("width_px"),
            "supported_sample_count": pred.get("supported_sample_count", 0),
            "support_segment_count": len(pred.get("supported_segments_xy", [])),
            "source_path": row["source_path"],
            "png_path": str(png_path), "prediction_path": str(pred_path),
        })
    csv_path = args.run / "gallery_index.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    partial = [r for r in rows if r["status"] == "PARTIAL"]
    data = {
        "method_id": receipt["start"]["method_id"],
        "input_unique_source_count": len(rows),
        "left_count": sum(r["camera_role"] == "left" for r in rows),
        "right_count": sum(r["camera_role"] == "right" for r in rows),
        "source_group_count": len({r["source_group"] for r in rows}),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "stop_reason_counts": dict(Counter(r["stop_reason"] for r in rows)),
        "by_camera": {camera: dict(Counter(r["status"] for r in rows if r["camera_role"] == camera))
                      for camera in ("left", "right")},
        "supported_extension_px_on_partial": describe([float(r["supported_extension_px"]) for r in partial]),
        "measured_width_px_on_partial": describe([float(r["measured_width_px"]) for r in partial
                                                   if r["measured_width_px"] is not None]),
        "support_segment_count_on_partial": describe([float(r["support_segment_count"]) for r in partial]),
        "network_response_used": False,
        "accuracy_metrics_available": False,
        "note": "Status and length are output properties, not verified straight-rope accuracy or physical contact.",
        "gallery_index_csv": str(csv_path),
    }
    (args.run / "summary.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(data, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
