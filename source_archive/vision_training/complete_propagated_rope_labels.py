"""Create geometry-completed copies of propagated pseudo labels.

Raw SAM 2 outputs are never overwritten.  Only the incoming-rope channel is
extended, using the anchor's manually reviewed tangent x coordinate and the
current frame's robust rope-line fit.  This is a semi-supervised teacher label,
not a production-time tangent detector.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import cv2

from rope_span_completion import complete_rope_span


CLASSES = ("wound_rope_surface", "flange_inner_face", "incoming_rope_free_span")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()
    task_dir = args.task_dir.resolve()
    root = task_dir / "propagation_windows"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for window in manifest["windows"]:
        window_dir = root / window["window_id"]
        report = json.loads((window_dir / "propagation_report.json").read_text(encoding="utf-8"))
        anchor_dir = Path(window["anchor_label_dir"])
        annotation = json.loads((anchor_dir / "annotation.json").read_text(encoding="utf-8"))
        tangent_x = float(annotation["tangent_point"][0])
        valid = 0
        added = []
        for frame in report["frames"]:
            index = int(frame["index"])
            source = window_dir / "pseudo_labels" / f"{index:05d}"
            target = window_dir / "pseudo_labels_completed" / f"{index:05d}"
            target.mkdir(parents=True, exist_ok=True)
            wound = cv2.imread(str(source / "wound_rope_surface.png"), cv2.IMREAD_GRAYSCALE)
            incoming = cv2.imread(str(source / "incoming_rope_free_span.png"), cv2.IMREAD_GRAYSCALE)
            if wound is None or incoming is None:
                result = None
            else:
                completed, result = complete_rope_span(
                    incoming > 127, wound > 127, (tangent_x, float("nan")))
                cv2.imwrite(str(target / "incoming_rope_free_span.png"),
                            completed.astype("uint8") * 255)
            for name in ("wound_rope_surface", "flange_inner_face"):
                shutil.copy2(source / f"{name}.png", target / f"{name}.png")
            if result is None:
                shutil.copy2(source / "incoming_rope_free_span.png",
                             target / "incoming_rope_free_span.png")
                detail = {"valid": False, "reason": "mask_missing"}
            else:
                detail = asdict(result)
                valid += int(result.valid)
                if result.valid:
                    added.append(result.added_pixels)
            rows.append({
                "window_id": window["window_id"], "index": index,
                "source_frame": frame["source_frame"], "time_s": frame["time_s"],
                "is_anchor": frame["is_anchor"], "completion": detail,
                "mask_dir": str(target.resolve()),
            })
        print(f"{window['window_id']}: completed={valid}/{len(report['frames'])}")
    valid_rows = [r for r in rows if r["completion"].get("valid")]
    payload = {
        "schema_version": 1,
        "status": "teacher_labels_not_production_measurements",
        "source": "raw SAM2 propagation plus anchor-x/current-line completion",
        "frames": rows,
        "summary": {
            "total": len(rows), "valid": len(valid_rows),
            "added_pixels_median": (float(sorted(
                r["completion"]["added_pixels"] for r in valid_rows)[len(valid_rows)//2])
                if valid_rows else None),
        },
    }
    (task_dir / "rope_completion_propagation_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
