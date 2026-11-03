"""Create a layout-only report revision without changing frozen V28 results.

The frozen renderer draws transformed curves on a panel canvas. Restore the
original title/margins after overlays, clipping those overlays to the actual
photograph. No coordinates, candidates, scores, or evaluation are changed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sys

from . import render_results


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restore_photo_margins(panel):
    """Keep only the overlay pixels inside the real displayed photograph."""
    base = panel.info.pop("_v28_clean_frame")
    x0, y0, x1, y1 = panel.info.pop("_v28_photo_rect")
    for box in ((0, 0, panel.width, y0), (0, y1, panel.width, panel.height),
                (0, y0, x0, y1), (x1, y0, panel.width, y1)):
        if box[2] > box[0] and box[3] > box[1]:
            panel.paste(base.crop(box), box[:2])


def render_layout_revision(source_dir, output_dir):
    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source == output:
        raise ValueError("Layout revision requires a new output directory")
    protocol = json.loads((source / "protocol.json").read_text("utf-8"))
    for path, expected in protocol["source_hashes"].items():
        if _sha(path) != expected:
            raise ValueError("Frozen source changed before layout revision: " + path)
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for name in ("input_manifest.json", "records.json", "evaluation.json"):
        shutil.copyfile(source / name, output / name)
        artifacts[name] = _sha(source / name)
    original_panel, original_footer = render_results._panel, render_results._footer

    def clipped_panel(image, crop, *args, **kwargs):
        panel, transform = original_panel(image, crop, *args, **kwargs)
        p0, p1 = transform([crop[:2], crop[2:]])
        box = tuple(int(round(v)) for v in (*p0, *p1))
        panel.info["_v28_clean_frame"] = panel.copy()
        panel.info["_v28_photo_rect"] = box
        return panel, transform

    def clipped_footer(panel, text):
        restore_photo_margins(panel)
        original_footer(panel, text)

    render_results._panel, render_results._footer = clipped_panel, clipped_footer
    try:
        result = render_results.render_run(output)
    finally:
        render_results._panel, render_results._footer = original_panel, original_footer
    provenance = dict(source_run=str(source), source_protocol_sha256=_sha(source / "protocol.json"),
        unchanged_artifacts=artifacts, layout_fix_script=str(Path(__file__).resolve()),
        layout_fix_script_sha256=_sha(__file__), frozen_renderer_sha256=_sha(render_results.__file__),
        change="clip overlays to the actual photo rectangle; restore titles and margins",
        inference_rerun=False, coordinates_changed=False, scores_changed=False, evaluation_changed=False,
        result=result)
    with (output / "layout_provenance.json").open("x", encoding="utf-8") as handle:
        json.dump(provenance, handle, ensure_ascii=False, indent=2)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    command = dict(argv=[sys.executable, "-m", __spec__.name, *sys.argv[1:]],
                   cwd=str(Path.cwd()),
                   started_at=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat())
    try:
        result = render_layout_revision(args.source, args.output)
        command.update(exit_code=0, status="layout_revision_complete")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        command.update(exit_code=1, status="failed_preserved", error=str(exc))
        raise
    finally:
        if args.output.is_dir():
            with (args.output / "layout_command_result.json").open("x", encoding="utf-8") as handle:
                json.dump(command, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
