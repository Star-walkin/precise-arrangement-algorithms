"""只复算已冻结 38 张 integration_pilot_001 的曲线族模块，不重跑观测。

用途是快速检查补线模块修订量。原图/旧输出只读，诊断另存 JSON。
这个试跑不是准确率评价，也不访问设备或保留集。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from curve_family_v3 import infer_family


HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent
DATA = ASSETS / "camera_condition_images_20261009_v2"


def read_image(path: Path, flag: int) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), flag)
    if image is None:
        raise ValueError(f"无法解码：{path}")
    return image


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=HERE / "integration_pilot_001")
    parser.add_argument("--output", type=Path, default=HERE / "family_pilot_002.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"不覆盖既有试跑：{args.output}")
    records = sorted((args.input / "boundaries" / "records").rglob("*.json"))
    if len(records) != 38:
        raise ValueError(f"预期固定 38 张，得到 {len(records)} 张")
    rows = []
    for path in records:
        old = json.loads(path.read_text(encoding="utf-8"))
        record = old["record"]
        rel = Path(record["output_file"])
        role, name = rel.parent.name, rel.name
        image = read_image(DATA / rel, cv2.IMREAD_COLOR)
        roi = read_image(args.input / "roi" / "reference_masks" / role / name,
                         cv2.IMREAD_GRAYSCALE)
        observed = read_image(args.input / "boundaries" / "observed" / role / name,
                              cv2.IMREAD_GRAYSCALE)
        rope = read_image(args.input / "boundaries" / "incoming_rope_occlusion" / role / name,
                          cv2.IMREAD_GRAYSCALE)
        unknown = read_image(args.input / "boundaries" / "unknown_exclusion" / role / name,
                             cv2.IMREAD_GRAYSCALE)
        inferred, lines, diagnostics = infer_family(
            image, roi, observed, old["lines"], rope, unknown, record["pitch_px"])
        if np.any((inferred > 0) & ((roi == 0) | (unknown > 0) | (observed > 0))):
            raise ValueError(f"紫线契约不成立：{rel}")
        for line in lines:
            for x, y in line.get("inferred_points_xy", []):
                if inferred[y, x] != 255:
                    raise ValueError(f"推断点与光栅不一致：{rel}")
        rows.append({"output_file": str(rel),
                     "old_inferred_pixels": record["inferred_pixels"],
                     "new_inferred_pixels": diagnostics["inferred_pixels"],
                     "old_connections": record["family_diagnostics"]["connection_count"],
                     "new_connections": diagnostics["connection_count"],
                     "new_under_occlusion_pixels": diagnostics["inferred_under_occlusion_pixels"],
                     "rejection_counts": diagnostics["rejection_counts"],
                     "rejected_connections": diagnostics["rejected_connections"]})
        print(role, name, "old/new pixels", rows[-1]["old_inferred_pixels"],
              rows[-1]["new_inferred_pixels"], flush=True)
    summary = {"input": str(args.input), "input_run_context_sha256": digest(args.input / "run_context.json"),
               "family_module_sha256": digest(HERE / "curve_family_v3.py"),
               "fixed_image_count": len(rows),
               "old_total_inferred_pixels": sum(r["old_inferred_pixels"] for r in rows),
               "new_total_inferred_pixels": sum(r["new_inferred_pixels"] for r in rows),
               "old_total_connections": sum(r["old_connections"] for r in rows),
               "new_total_connections": sum(r["new_connections"] for r in rows),
               "new_total_under_occlusion_pixels": sum(r["new_under_occlusion_pixels"] for r in rows),
               "interpretation": "只复算冻结观测上的曲线族；像素数不是准确率，需看原图"}
    args.output.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False,
                                      indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
