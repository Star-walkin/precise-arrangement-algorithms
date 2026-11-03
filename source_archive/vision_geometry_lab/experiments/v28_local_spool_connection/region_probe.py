"""Read-only probe of existing coarse regions on the two open V28 debug images."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "vision_prototype"))
import cv2
import numpy as np
from semantic_locator import semantic_locator_for_camera
from surface_centerlines import source_wound_mask


def encode(path, image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("PNG encoding failed")
    encoded.tofile(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--fullwidth",action="store_true")
    args=parser.parse_args()
    source = ROOT / "vision_geometry_lab/outputs/v28_local_spool_connection/v28_20261007_002"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    items = json.loads((source / "input_manifest.json").read_text(encoding="utf-8-sig"))
    for item in items:
        sid = item["sample_id"]
        if not sid.endswith(("left__f00450", "left__f00580")):
            continue
        image = cv2.imdecode(np.fromfile(item["image_path"], np.uint8), cv2.IMREAD_COLOR)
        locator = semantic_locator_for_camera("left")
        if args.fullwidth:
            locator.crop_norm = (0., locator.crop_norm[1], 1., locator.crop_norm[3])
        regions = locator.predict(image)
        wound = source_wound_mask(regions, image.shape)
        x0, y0, x1, y1 = regions.crop_xyxy
        masks = {"wound": wound}
        for name in ("flange_inner_face", "incoming_rope_free_span"):
            full = np.zeros(image.shape[:2], np.uint8)
            full[y0:y1, x0:x1] = cv2.resize(regions.mask(name).astype(np.uint8), (x1-x0, y1-y0), interpolation=cv2.INTER_NEAREST)
            masks[name] = full.astype(bool)
        folder = output / sid
        folder.mkdir()
        np.savez_compressed(folder / "existing_semantic_masks.npz", **masks)
        original = cv2.resize(image, (2048, 1500))
        canvas = image.copy()
        colors = {"wound": (40,210,40), "flange_inner_face": (20,180,255), "incoming_rope_free_span": (255,140,20)}
        summary = {}
        for name, mask in masks.items():
            tint = np.asarray(colors[name], np.float32)
            canvas[mask] = (.60*canvas[mask] + .40*tint).astype(np.uint8)
            contour,_ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contour, -1, colors[name], 5)
            ys,xs=np.nonzero(mask)
            summary[name] = dict(area_px=int(mask.sum()), bbox_xyxy=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if len(xs) else None)
        canvas = cv2.resize(canvas,(2048,1500))
        for row,name in enumerate(masks):
            cv2.putText(canvas,name,(25,50+40*row),cv2.FONT_HERSHEY_SIMPLEX,1.1,colors[name],3,cv2.LINE_AA)
        encode(folder / "existing_regions_overlay.png", canvas)
        encode(folder / "original.png", original)
        (folder / "summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
        print(json.dumps(dict(sample_id=sid,summary=summary,overlay=str(folder/"existing_regions_overlay.png"))),flush=True)


if __name__ == "__main__":
    main()
