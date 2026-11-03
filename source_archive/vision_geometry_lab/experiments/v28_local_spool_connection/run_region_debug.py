"""Run V28 coarse-region repair on fixed, already-open left development frames."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from vision_geometry_lab.experiments.v28_local_spool_connection.run_experiment import detect_input
from vision_geometry_lab.experiments.v28_local_spool_connection.regions_repair import detect_regions,save_regions


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"vision_geometry_lab/outputs/v28_local_spool_connection/repair_debug_20261007_001/regions")
    args=parser.parse_args()
    source=ROOT/"vision_geometry_lab/outputs/v28_local_spool_connection/v28_20261007_002/input_manifest.json"
    manifest=json.loads(source.read_text(encoding="utf-8-sig"))
    for item in manifest:
        sid=item["sample_id"]
        if not sid.endswith(("left__f00450","left__f00580")):
            continue
        image=cv2.imdecode(np.fromfile(item["image_path"],np.uint8),cv2.IMREAD_COLOR)
        entry=detect_input(image,sid,"AUTO")
        regions=detect_regions(image,entry["free_seed"],entry["width_px"],camera=item["source"]["camera"],arc=entry["arc"])
        folder=args.output/sid
        save_regions(folder,image,regions)
        (folder/"fresh_auto_entry.json").write_text(json.dumps(entry,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps(dict(sample_id=sid,entry=entry["available"],summary=regions["summary"],overlay=str(folder/"regions_overlay.png")),ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
