"""Finite geometry checks plus three preselected open-development smoke cases."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import cv2
import numpy as np
from surface_tracks import build_tracks, draw_tracks, load_surface_priors

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
IDS = ["dual_20260923_101517__pair0075__right", "dual_20260924_111500__pair0015__left", "R_run_20260923_01__f000055__right"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry_checks():
    shape = (1500, 100)
    f3 = np.zeros(shape, np.uint8)
    f3[625, 3:94] = 255
    f3[645, 3:94] = 255
    f3[665, 3:94] = 255
    f3[645, 42:54] = 0  # Remove the middle boundary; never invent it again.
    surface = np.zeros(shape, np.uint8)
    surface[525:] = 255
    priors = dict(f3_merged_half=f3, f3_inferred_half=np.zeros_like(f3), surface_half=surface,
                  sam2_surface_half=surface, original_half=np.zeros((*shape, 3), np.uint8))
    heat = np.full((975, 100), .7, np.float32)
    result = build_tracks(priors, heat)
    assert result["tracks"]
    for t in result["tracks"]:
        p = np.asarray(t["midpoint_xy"])
        assert not np.any((p[:, 0] >= 42) & (p[:, 0] < 54)), "Doubled missing-boundary gap was filled"
        assert np.max(np.asarray(t["heat_score"])) < 1e-5, "Constant response became structured evidence"
        assert not any(t["supported"]), "Raw all-zero ROI became false image support"
        assert np.max(np.asarray(t["spacing_px"])) < 21
    return {"missing_middle_boundary_not_filled": True, "constant_heat_has_no_shape_support": True,
            "ROI_zero_gray_retained_not_excluded": True, "source_boundary_count": result["diagnostics"]["f3_component_count"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="surface_checks_002")
    args = parser.parse_args()
    out = HERE / args.out
    out.mkdir(exist_ok=False)
    checks = geometry_checks()
    jobs = json.loads((BASE / "batch_ab_20261030_001/jobs.json").read_text(encoding="utf-8"))["jobs"]
    rows = []
    for identity in IDS:
        job = next(x for x in jobs if x["id"] == identity)
        priors = load_surface_priors(job)
        entry = 1 if identity == IDS[0] else 0
        response = BASE / "postprocess_b_20261030_v1/responses" / identity / f"entry_{entry:02d}.npz"
        before = digest(response)
        with np.load(response, allow_pickle=False) as data:
            heat = data["response"].copy()
        started = time.monotonic()
        result = build_tracks(priors, heat)
        elapsed = time.monotonic() - started
        for track in result["tracks"]:
            p = np.asarray(track["points_xy"])
            m = np.asarray(track["midpoint_xy"])
            width = np.asarray(track["spacing_px"])
            assert np.all(np.linalg.norm(p - m, axis=1) <= width * .15001)
            assert np.all(np.diff(m[:, 0]) == 3)
            assert len(set(track["boundary_ids"])) == 2
            assert len(p) == len(track["supported"])
        assert digest(response) == before
        original = priors["original_half"][525:]
        overlay = draw_tracks(original, result)
        cv2.imencode(".jpg", np.concatenate([original, overlay], axis=1), [cv2.IMWRITE_JPEG_QUALITY, 94])[1].tofile(str(out / (identity + ".jpg")))
        row = dict(id=identity, entry_index=entry, response_sha256=before,
                   input_dependency_sha256=priors["input_dependency_sha256"], seconds=elapsed,
                   diagnostics=result["diagnostics"], bounds_and_adjacency_checks_pass=True,
                   longest_track_half_px=max((t["length_px"] for t in result["tracks"]), default=0))
        rows.append(row)
        print(json.dumps({"id": identity, "tracks": len(result["tracks"]), "seconds": elapsed}), flush=True)
    receipt = dict(command=sys.argv, python=sys.executable, exit_code=0, geometry_checks=checks,
                   source_sha256=digest(HERE / "surface_tracks.py"), check_source_sha256=digest(__file__),
                   labels_read=False, parameter_selection_from_evaluation=False, rows=rows)
    (out / "summary.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
