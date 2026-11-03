"""Render the 108 open-development images lacking an existing saved entry.

This is a failure inventory only: no entry selection, network inference, labels,
or response arrays are read. Blank A/B panels do not mean zero response.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import time

from PIL import Image, ImageDraw, ImageFont


BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
DATA = ROOT / "数据资产/05_派生与实验结果/camera_condition_images_20261009_v2"
DEFAULT_RUN = BASE / "postprocess_20261030_001"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def font(size: int, bold: bool = False):
    return ImageFont.truetype(
        str(Path("C:/Windows/Fonts") / ("msyhbd.ttc" if bold else "msyh.ttc")), size
    )


def centered(draw, box, lines, color=(65, 65, 65), size=56):
    x0, y0, x1, y1 = box
    f = font(size)
    step = int(size * 1.65)
    yy = (y0 + y1 - step * len(lines)) / 2
    for line in lines:
        bounds = draw.textbbox((0, 0), line, font=f)
        xx = (x0 + x1 - (bounds[2] - bounds[0])) / 2
        draw.text((xx, yy), line, font=f, fill=color)
        yy += step


def render(source: Image.Image, sample_id: str, width: int) -> Image.Image:
    image_height = round(source.height * width / source.width)
    header, footer = 120, 124
    canvas = Image.new("RGB", (width * 4, header + image_height + footer), "white")
    original = source.resize((width, image_height), Image.Resampling.LANCZOS)
    canvas.paste(original, (0, header))
    canvas.paste(original, (width * 3, header))
    draw = ImageDraw.Draw(canvas)
    titles = ["原图", "方案 A：未推理（无模型热图）", "方案 B：未推理（无模型热图）", "原图诊断：缺少既有有效入口"]
    for i, title in enumerate(titles):
        x0 = i * width
        draw.rectangle((x0, 0, x0 + width - 1, header - 1), fill=(27, 35, 45))
        draw.text((x0 + 32, 16), title, font=font(42, True), fill="white")
        draw.text((x0 + 32, 76), sample_id, font=font(25), fill=(220, 225, 232))
        if i in (1, 2):
            draw.rectangle((x0, header, x0 + width - 1, header + image_height - 1), fill=(232, 234, 237))
            centered(draw, (x0, header, x0 + width, header + image_height), [
                "无既有有效入口", "未执行网络推理", "此空白不是全零响应", "未生成热力图或目标路径"
            ])
        if i > 0:
            draw.line((x0, 0, x0, canvas.height), fill=(100, 100, 100), width=3)
    # Keep the diagnostic original recognizable; add only a text status banner.
    x0 = width * 3
    draw.rectangle((x0 + 24, header + 24, x0 + width - 24, header + 158), fill=(80, 53, 20))
    draw.text((x0 + 48, header + 39), "入口缺失：保留失败，不新增或猜测入口", font=font(42, True), fill="white")
    draw.text((x0 + 48, header + 100), "缺少入口 ≠ 图中无绳；后续身份与末端均未评价", font=font(32), fill=(255, 230, 190))
    yy = header + image_height
    draw.rectangle((0, yy, canvas.width, canvas.height), fill=(246, 247, 248))
    draw.text((32, yy + 15), "状态：NO_SAVED_OBSERVED_ACTIVE_ROPE_ENTRY | 仅使用开放 400 图清单与既有入口清单；无新入口、无网络前向。", font=font(34), fill=(50, 50, 50))
    draw.text((32, yy + 66), "原图保留完整视野并等比缩小供查看。A/B 空白仅标记未推理；没有浮点响应、阈值、连线或真实定位结论。", font=font(32), fill=(50, 50, 50))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--panel-width", type=int, default=2048)
    args = parser.parse_args()
    run = args.run.resolve()
    assert run.parent == BASE.resolve() and run.name == "postprocess_20261030_001"
    assert args.panel_width >= 1024
    manifest_path = DATA / "manifest.csv"
    jobs_path = BASE / "batch_20261030_001/jobs.json"
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8-sig")))
    jobs = read_json(jobs_path)
    source_ids = [Path(r["output_file"]).stem for r in rows]
    job_ids = [j["id"] for j in jobs["jobs"]]
    assert len(rows) == len(set(source_ids)) == 400
    assert len(job_ids) == len(set(job_ids)) == 292
    assert sum(len(j["seeds"]) for j in jobs["jobs"]) == 299
    assert set(job_ids) <= set(source_ids)
    missing = [r for r in rows if Path(r["output_file"]).stem not in set(job_ids)]
    assert len(missing) == 108
    skipped = {r["file"]: r["reason"] for r in jobs["skipped"]}
    assert set(skipped) == {r["output_file"] for r in missing}
    assert set(skipped.values()) == {"no_saved_observed_active_rope_entry"}
    for row in rows:
        assert row["dataset_use_status"] == "unlabeled_open_development_no_new_test_split"
        assert "20260925" not in row["output_file"] + row["source_file"]
    outdir = run / "无入口诊断_108"
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    records = []
    for i, row in enumerate(missing, 1):
        path = (DATA / row["output_file"]).resolve()
        assert path.is_relative_to(DATA.resolve())
        sample_id = path.stem
        data = path.read_bytes()
        digest = sha(data)
        assert digest == row["output_sha256"]
        with Image.open(BytesIO(data)) as source:
            source.load()
            assert source.size == (int(row["width"]), int(row["height"]))
            canvas = render(source.convert("RGB"), sample_id, args.panel_width)
        output = outdir / (sample_id + ".jpg")
        canvas.save(output, quality=90, subsampling=0, optimize=True)
        with Image.open(output) as verify:
            assert verify.size == canvas.size and verify.mode == "RGB"
            verify.verify()
        records.append({
            "id": sample_id, "source_group": row["source_group"],
            "dataset_use_status": row["dataset_use_status"], "historical_use": row["historical_use"],
            "source_image": str(path), "source_sha256": digest,
            "reason": skipped[row["output_file"]], "interpretation": "missing_saved_entry_not_absence_of_rope",
            "network_inference": False, "response_available": False,
            "output": str(output), "output_sha256": sha(output.read_bytes()), "output_size_wh": list(canvas.size)
        })
        if i == 1 or i % 12 == 0 or i == len(missing):
            print(json.dumps({"completed": i, "total": len(missing), "last": sample_id}), flush=True)
    actual = {p.stem for p in outdir.glob("*.jpg")}
    assert actual == {r["id"] for r in records} and len(actual) == 108
    assert not (actual & set(job_ids))
    receipt = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "exit_code": 0,
        "command": ".\\vision_prototype\\.venv\\Scripts\\python.exe -B experiments/entry_conditioned_rope_path/postprocess_missing_entries.py",
        "script": str(Path(__file__).resolve()), "script_sha256": sha(Path(__file__).read_bytes()),
        "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path.read_bytes()),
        "saved_entry_jobs": str(jobs_path), "saved_entry_jobs_sha256": sha(jobs_path.read_bytes()),
        "allowed_source_count": 400, "with_saved_entries_image_count": 292, "saved_entry_count": 299,
        "no_saved_entry_image_count": 108, "rendered_image_count": len(records),
        "missing_source_groups": dict(Counter(r["source_group"] for r in records)),
        "output_directory": str(outdir), "panel_order": ["original", "A_not_inferred_blank", "B_not_inferred_blank", "original_entry_missing_diagnostic"],
        "panel_width": args.panel_width, "elapsed_seconds": time.monotonic() - started,
        "network_forward_count": 0, "new_entries_selected": 0, "ground_truth_read": False, "reserved_data_accessed": False,
        "interpretation": "108 images have no existing valid saved entry; this is not a claim of no rope in their images. A/B blank panels are missing outputs, not zero-valued responses.",
        "records": records,
    }
    (run / "missing_entries_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"exit_code": 0, "rendered": len(records), "network_forwards": 0, "elapsed_seconds": round(receipt["elapsed_seconds"], 2)}), flush=True)


if __name__ == "__main__":
    main()
