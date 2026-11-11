"""Check restored source bytes and optionally smoke-test on synthetic arrays.

Writes only f3_restore_receipt.json and, when requested,
f3_core_smoke_receipt.json beside this script. Never opens research images.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_sources():
    source_manifest = ROOT / "f3_source_manifest.json"
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    records = []
    for item in manifest["files"]:
        path = ROOT / item["local_relative_path"]
        payload = path.read_bytes()
        actual_blob = hashlib.sha1(b"blob " + str(len(payload)).encode("ascii")
                                   + b"\0" + payload).hexdigest()
        if actual_blob != item["git_blob_sha"]:
            raise ValueError(f"Restored bytes differ from Git blob: {path}")
        sha = hashlib.sha256(payload).hexdigest()
        expected_sha = item.get("p1_manifest_sha256")
        if expected_sha and expected_sha != sha:
            raise ValueError(f"Restored source differs from P1 manifest SHA-256: {path}")
        if path.suffix == ".py":
            ast.parse(payload.decode("utf-8"), filename=str(path))
        records.append(dict(path=item["local_relative_path"], bytes=len(payload),
                            git_blob_sha=actual_blob, sha256=sha,
                            p1_manifest_sha256_checked=bool(expected_sha)))
    result = dict(status="passed", checked_utc=datetime.now(timezone.utc).isoformat(),
                  repository=manifest["repository"],
                  repository_commit_sha=manifest["repository_commit_sha"],
                  repository_tree_sha=manifest["repository_tree_sha"],
                  source_manifest_sha256=digest(source_manifest),
                  verified_files=len(records),
                  python_syntax_files=sum(r["path"].endswith(".py") for r in records),
                  sources=records, vendor_logic_modified=False,
                  full_historical_dataset_runner_reproduced=False)
    dump(ROOT / "f3_restore_receipt.json", result)
    return result


def smoke():
    import numpy as np
    import cv2
    from f3_core_adapter import detect_surface_core

    cv2.setNumThreads(1)
    height, width, pitch = 640, 960, 20.0
    y, x = np.mgrid[:height, :width]
    gray = np.rint(128 + 55 * np.cos(2 * np.pi * (y + 0.012 * x) / pitch)).astype(np.uint8)
    image = np.repeat(gray[:, :, None], 3, axis=2)
    surface = np.zeros((height, width), np.uint8)
    surface[100:600, 80:900] = 255
    exclusion = np.zeros_like(surface)
    cases = []
    for name, mask in (("empty_surface", np.zeros_like(surface)), ("synthetic_periodic_surface", surface)):
        before = hashlib.sha256(image.tobytes()).hexdigest()
        result = detect_surface_core(image, mask, pitch, exclusion_mask=exclusion, c3_record=None)
        if hashlib.sha256(image.tobytes()).hexdigest() != before:
            raise ValueError("Core changed input image")
        if result["normalized_bgr"].shape != image.shape:
            raise ValueError("Wrong normalized image shape")
        if result["exclusion_mask"].shape != surface.shape:
            raise ValueError("Wrong output exclusion shape")
        for point in result["points"]:
            if not (0 <= point["x"] < width and 0 <= point["y"] < height):
                raise ValueError("Observation outside source image")
        cases.append(dict(name=name, status="executed",
                          points=len(result["points"]), tracks=len(result["tracks"]),
                          accepted_curves=len(result["fit"]["curves"]),
                          source_image_unchanged=True))
    value = dict(status="passed", checked_utc=datetime.now(timezone.utc).isoformat(),
                 scope="Synthetic execution only; no accuracy or full F3 fusion claim",
                 python=sys.version, numpy=np.__version__, opencv=cv2.__version__,
                 cv2_file=cv2.__file__, image_shape=[height, width, 3],
                 pitch_original_px=pitch, coordinate_system="input_full_image_xy",
                 mirrored=False, models_loaded=False, research_images_read=False,
                 cases=cases)
    dump(ROOT / "f3_core_smoke_receipt.json", value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        result = verify_sources()
        print(json.dumps({k: v for k, v in result.items() if k != "sources"}, ensure_ascii=False))
        if args.smoke:
            print(json.dumps(smoke(), ensure_ascii=False))
        return 0
    except Exception as exc:
        traceback.print_exc()
        dump(ROOT / "f3_verification_failure.json",
             dict(status="failed", error_type=type(exc).__name__, error=str(exc),
                  checked_utc=datetime.now(timezone.utc).isoformat()))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
