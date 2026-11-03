"""Read-only, post-hoc V26 output audit; never reruns a tracker or loads labels.

The fixed two examples and saved tracker outputs are descriptive diagnostics,
not new primary metrics, material identity evidence, or physical motion truth.
Only B metadata/results and prepared cohort inputs are read. Native gray PNGs
preserve the exact cached 1024-square pixels; point images are separate files.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CONFIG = dict(schema="v26_posthoc_review_1", groups=["left_000240", "right_000140"],
              frame_slots=[0, 1, 2, 5, 10], methods=["lk256", "lk_native"],
              coordinate_rule="native crop xy = stored full-image xy - box[:2]; no resampling",
              displacement_rule="same point ID valid at both adjacent stored frames; no gap bridging",
              projection_sign="positive along initial free_seed[-1] - free_seed[0]",
              posthoc=True, new_primary_metric=False, tracker_reexecuted=False,
              material_identity_confirmed=False, physical_motion_confirmed=False,
              physical_t1_t2="not_evaluable", deployment_permitted=False)
FONT = Path("C:/Windows/Fonts/msyh.ttc")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(array):
    array = np.ascontiguousarray(array)
    header = json.dumps(dict(shape=list(array.shape), dtype=str(array.dtype)),
                        sort_keys=True, separators=(",", ":")).encode("ascii")
    digest = hashlib.sha256(header)
    digest.update(array.tobytes())
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def stats(values):
    values = np.asarray(values, float)
    if not len(values):
        return dict(count=0, min=None, mean=None, p50=None, p95=None, max=None)
    if not np.isfinite(values).all():
        raise ValueError("diagnostic statistic must be finite")
    return dict(count=len(values), min=float(values.min()), mean=float(values.mean()),
                p50=float(np.percentile(values, 50)), p95=float(np.percentile(values, 95)),
                max=float(values.max()))


def _points(row):
    points = np.asarray(row["points_xy_native"], dtype=float).reshape(-1, 2)
    valid = np.asarray(row["valid_candidate"], dtype=bool)
    if len(points) != len(valid) or row["point_ids"] != list(range(len(valid))):
        raise ValueError("point correspondence changed")
    if not np.isfinite(points[valid]).all():
        raise ValueError("accepted point is nonfinite")
    return points, valid


def accepted_steps(rows, free_seed):
    seed = np.asarray(free_seed, dtype=float)
    direction = seed[-1] - seed[0]
    length = np.linalg.norm(direction)
    if not np.isfinite(seed).all() or length < 1e-9:
        raise ValueError("invalid initial free-seed direction")
    direction /= length
    normal = np.array([-direction[1], direction[0]])
    records, by_step = [], []
    for previous, current in zip(rows[:-1], rows[1:]):
        before, keep_before = _points(previous)
        after, keep_after = _points(current)
        if len(before) != len(after) or np.any(keep_after & ~keep_before):
            raise ValueError("point ID revival violates saved absorbing contract")
        dt = float(current["time_s"] - previous["time_s"])
        if current["source_frame"] != previous["source_frame"] + 1 or dt <= 0:
            raise ValueError("not an adjacent chronological saved frame pair")
        local = []
        for point_id in np.flatnonzero(keep_before & keep_after):
            delta = after[point_id] - before[point_id]
            record = dict(point_id=int(point_id), from_slot=previous["frame_slot_index"],
                          to_slot=current["frame_slot_index"],
                          from_source_frame=previous["source_frame"], to_source_frame=current["source_frame"],
                          to_age_s=current["age_s"], dt_host_receive_s=dt,
                          delta_xy_native_px=delta.tolist(), displacement_native_px=float(np.linalg.norm(delta)),
                          along_initial_free_seed_native_px=float(delta @ direction),
                          across_initial_free_seed_native_px=float(delta @ normal))
            records.append(record)
            local.append(record)
        by_step.append(dict(to_slot=current["frame_slot_index"], age_s=current["age_s"],
                            accepted_adjacent_pairs=len(local), current_points=int(keep_after.sum()),
                            displacement_native_px=stats([r["displacement_native_px"] for r in local]),
                            along_initial_free_seed_native_px=stats([
                                r["along_initial_free_seed_native_px"] for r in local])))
    return dict(direction_xy=direction.tolist(), rows=records, by_step=by_step,
                displacement_native_px=stats([r["displacement_native_px"] for r in records]),
                along_initial_free_seed_native_px=stats([
                    r["along_initial_free_seed_native_px"] for r in records]),
                selection_bias="accepted pairs only; rejected motion is unknown; repeated point pairs are correlated",
                semantics="saved tracker displacement, NOT measured true material motion or velocity")


def failures(rows, min_points):
    masks = [_points(row)[1] for row in rows]
    count = len(masks[0])
    counts = [int(mask.sum()) for mask in masks]

    def first(indices):
        if not indices:
            return None
        index = indices[0]
        row = rows[index]
        return dict(frame_slot=index, source_frame=row["source_frame"], age_s=row["age_s"],
                    last_prior_scheduled_age_s=rows[index - 1]["age_s"] if index else None,
                    current_point_count=counts[index], row_rejection_reason=row.get("rejection_reason"))

    return dict(initialized_points=count, current_counts=counts,
                first_any_point_invalid=first([i for i, n in enumerate(counts) if n < count]),
                first_below_fixed_min_points=first([i for i, n in enumerate(counts) if n < min_points]),
                first_all_invalid=first([i for i, n in enumerate(counts) if n == 0]),
                first_age_limit_row=first([i for i, r in enumerate(rows) if r.get("rejection_reason") == "cohort_age_limit"]),
                per_point=[dict(point_id=j, first_invalid=first([i for i, m in enumerate(masks) if not m[j]]),
                                right_censored=all(m[j] for m in masks)) for j in range(count)],
                semantics="first saved rejection age, not independently observed physical disappearance; partial-loss cause may be unrecorded")


def current_point_image(gray, row, box):
    canvas = Image.fromarray(gray).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(FONT), 16) if FONT.is_file() else ImageFont.load_default(size=16)
    points, valid = _points(row)
    for point_id in np.flatnonzero(valid):
        x, y = points[point_id] - np.asarray(box[:2], float)
        if not (0 <= x <= gray.shape[1] - 1 and 0 <= y <= gray.shape[0] - 1):
            raise ValueError("accepted native point outside the exact displayed crop")
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline="#ff35d4", width=2)
        draw.text((min(gray.shape[1] - 30, x + 7), max(0, y - 20)), str(point_id), font=font,
                  fill="#fff478", stroke_width=1, stroke_fill="black")
    return canvas


def save_png(path, image):
    with Path(path).open("xb") as stream:
        image.save(stream, format="PNG")


def comparison_page(images, group_id, slot, source_frame, age, counts):
    # Image panels are pasted at native resolution, never resampled.
    page = Image.new("RGB", (3 * 1024, 1144), "#102030")
    draw = ImageDraw.Draw(page)
    font = ImageFont.truetype(str(FONT), 24) if FONT.is_file() else ImageFont.load_default(size=24)
    draw.text((15, 10), f"POSTHOC | {group_id} | slot {slot} / frame {source_frame} | host age {age:.6f}s", font=font, fill="white")
    draw.text((15, 42), "Cached gray_native 1024x1024 at 1:1. Current candidate IDs only; NO trajectory / NO identity confirmation.", font=font, fill="#ffdda0")
    for i, title in enumerate(("CLEAN native gray", f"lk256 saved current points: {counts[0]}",
                               f"lk_native saved current points: {counts[1]}")):
        draw.text((i * 1024 + 15, 82), title, font=font, fill="#94def5")
        page.paste(images[i], (i * 1024, 120))
    return page


def synthetic_checks():
    def row(slot, points, valid):
        return dict(frame_slot_index=slot, source_frame=10 + slot, time_s=slot * .1,
                    age_s=slot * .1, point_ids=[0, 1], points_xy_native=points,
                    valid_candidate=valid, rejection_reason=None)
    rows = [row(0, [[10, 20], [15, 20]], [True, True]),
            row(1, [[13, 24], [float("nan"), float("nan")]], [True, False]),
            row(2, [[16, 28], [100, 100]], [True, False])]
    result = accepted_steps(rows, [[0, 0], [3, 4]])
    assert len(result["rows"]) == 2
    assert np.allclose([r["displacement_native_px"] for r in result["rows"]], 5.)
    assert np.allclose([r["along_initial_free_seed_native_px"] for r in result["rows"]], 5.)
    assert failures(rows, 2)["first_any_point_invalid"]["frame_slot"] == 1
    assert failures(rows, 2)["per_point"][0]["right_censored"]
    gray = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    test = row(0, [[110, 220], [115, 220]], [True, False])
    painted = np.asarray(current_point_image(gray, test, [100, 200, 64, 64]))
    assert tuple(painted[20, 5]) == (255, 53, 212)
    assert tuple(painted[60, 60]) == (int(gray[60, 60]),) * 3
    revival = [rows[0], rows[1], row(2, [[16, 28], [18, 24]], [True, True])]
    try:
        accepted_steps(revival, [[0, 0], [3, 4]])
    except ValueError:
        pass
    else:
        raise AssertionError("revival accepted")
    return dict(status="passed", checks=["3-4-5 displacement/projection", "invalid pair exclusion",
        "first-loss age/right censoring", "native crop point mapping", "clean background preservation",
        "point revival rejected"], real_tracker_executed=False)


def run(base, outdir):
    base, outdir = Path(base).resolve(), Path(outdir).resolve()
    if outdir.exists():
        raise FileExistsError(f"refusing to overwrite {outdir}")
    if outdir.parent != base or outdir.name != "posthoc":
        raise ValueError("this bounded audit only writes a new base/posthoc directory")
    inputs = {}

    def tracked(relative):
        path = base / relative
        inputs[relative] = sha(path)
        return path

    manifest = read(tracked("B/manifest.json"))
    report = read(tracked("B/report.json"))
    if manifest["protocol_sha256"] != report["protocol_sha256"]:
        raise ValueError("B result protocol mismatch")
    if manifest["methods"] != CONFIG["methods"]:
        raise ValueError("unexpected completed methods")
    shared = manifest["config"]["shared_config"]
    threshold, min_points = float(shared["max_age_s"]), int(shared["min_points"])
    group_ids = [f"{side}_{start:06d}" for side in ("left", "right")
                 for start in shared["cohort_start_indices"]]
    if len(group_ids) != 60:
        raise ValueError("fixed 60-cohort denominator required")
    cases, selected = [], {}
    for group_id in group_ids:
        directory = f"cohorts/{group_id}"
        for name in ("meta.json", "data.npz"):
            key = f"{directory}/{name}"
            tracked(key)
            if inputs[key] != manifest["input_sha256"][key] or inputs[key] != report["input_sha256"][key]:
                raise ValueError(f"prepared cohort SHA mismatch: {key}")
        meta = read(base / directory / "meta.json")
        with np.load(base / directory / "data.npz", allow_pickle=False) as archive:
            times, frames = archive["times"], archive["source_frames"]
            gray = archive["gray_native"] if group_id in CONFIG["groups"] else None
        if len(times) != 21 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
            raise ValueError("invalid prepared timestamps")
        age = times - times[0]
        case = dict(group_id=group_id, seed_available=meta["available"], seed_reason=meta["reason"],
                    final_age_s=float(age[-1]), final_age_exceeds_limit=bool(age[-1] > threshold + 1e-9),
                    final_age_minus_limit_ms=float((age[-1] - threshold) * 1000), methods={})
        outputs = {}
        for method in CONFIG["methods"]:
            key = f"B/{method}/{group_id}.json"
            output = read(tracked(key))
            if (output["group_id"] != group_id or output["method"] != method or
                    output["protocol_sha256"] != manifest["protocol_sha256"] or output["box"] != meta["box"] or
                    output["seeds_native"] != (meta["seeds_native"] if meta["available"] else [])):
                raise ValueError("saved trajectory identity mismatch")
            if output["input_sha256"] != {name: inputs[f"{directory}/{name}"] for name in ("meta.json", "data.npz")}:
                raise ValueError("saved trajectory input binding mismatch")
            rows = output["rows"]
            if (len(rows) != 21 or [r["source_frame"] for r in rows] != frames.tolist() or
                    not np.allclose([r["time_s"] for r in rows], times, rtol=0, atol=1e-9) or
                    not np.allclose([r["age_s"] for r in rows], age, rtol=0, atol=1e-9)):
                raise ValueError("saved output/source frame-time alignment mismatch")
            failure = failures(rows, min_points)
            case["methods"][method] = dict(trajectory_sha256=inputs[key], failures=failure,
                final_current_points=sum(rows[-1]["valid_candidate"]),
                final_min_points=sum(rows[-1]["valid_candidate"]) >= min_points,
                penultimate_min_points=sum(rows[-2]["valid_candidate"]) >= min_points,
                final_row_rejection_reason=rows[-1].get("rejection_reason"))
            outputs[method] = output
        cases.append(case)
        if group_id in CONFIG["groups"]:
            if meta.get("rotation_or_mirror") is not False or gray.shape != (21, 1024, 1024) or gray.dtype != np.uint8:
                raise ValueError("not the requested exact native gray crop")
            selected[group_id] = (meta, gray, outputs)
    synthetic = synthetic_checks()
    source_hash = sha(__file__)
    outdir.mkdir(exist_ok=False)
    write(outdir / "source_receipt.json", dict(source_path=str(Path(__file__).resolve()), source_sha256=source_hash,
        input_sha256=inputs, config=CONFIG, protocol_sha256=manifest["protocol_sha256"],
        font_path=str(FONT), font_sha256=sha(FONT) if FONT.is_file() else None,
        identity_labels_read=[], original_videos_opened=[], models_loaded=[]))
    write(outdir / "synthetic_case_summary.json", synthetic)
    examples, image_checks = {}, []
    for group_id in CONFIG["groups"]:
        meta, gray, outputs = selected[group_id]
        folder = outdir / group_id
        folder.mkdir()
        examples[group_id] = dict(box=meta["box"], initial_free_seed=meta["free_seed"],
                                 optical_width_px=meta["optical_width_px"], methods={})
        for method in CONFIG["methods"]:
            examples[group_id]["methods"][method] = dict(
                trajectory_sha256=inputs[f"B/{method}/{group_id}.json"],
                accepted_adjacent_steps=accepted_steps(outputs[method]["rows"], meta["free_seed"]),
                failures=failures(outputs[method]["rows"], min_points))
        for slot in CONFIG["frame_slots"]:
            stem = f"slot_{slot:02d}"
            clean_path = folder / f"{stem}_clean_native.png"
            clean = Image.fromarray(gray[slot])
            save_png(clean_path, clean)
            restored = np.asarray(Image.open(clean_path))
            if not np.array_equal(restored, gray[slot]):
                raise RuntimeError("native clean PNG changed cached pixels")
            images, counts = [clean.convert("RGB")], []
            for method in CONFIG["methods"]:
                row = outputs[method]["rows"][slot]
                painted = current_point_image(gray[slot], row, meta["box"])
                save_png(folder / f"{stem}_{method}_current_points_native.png", painted)
                images.append(painted)
                counts.append(sum(row["valid_candidate"]))
            row = outputs[CONFIG["methods"][0]]["rows"][slot]
            save_png(folder / f"{stem}_comparison_1to1.png", comparison_page(
                images, group_id, slot, row["source_frame"], row["age_s"], counts))
            image_checks.append(dict(group_id=group_id, slot=slot, source_frame=row["source_frame"],
                cached_gray_array_sha256=array_sha(gray[slot]), decoded_clean_png_array_sha256=array_sha(restored),
                pixel_equal=True, shape_hw=list(gray[slot].shape),
                clean_path=str(clean_path.relative_to(outdir)), current_counts=dict(zip(CONFIG["methods"], counts))))
    age_summary = dict(fixed_cohorts=len(cases), age_limit_s=threshold,
        final_age_exceeds_limit_cohorts=sum(c["final_age_exceeds_limit"] for c in cases),
        over_limit_excess_ms=stats([c["final_age_minus_limit_ms"] for c in cases if c["final_age_exceeds_limit"]]),
        interpretation="receive-time age guard is prespecified administrative expiry, NOT evidence of natural tracking loss",
        methods={})
    for method in CONFIG["methods"]:
        items = [c["methods"][method] for c in cases]
        age_summary["methods"][method] = dict(
            final_min_points_cohorts=sum(v["final_min_points"] for v in items),
            final_any_points_cohorts=sum(v["final_current_points"] > 0 for v in items),
            final_reason_counts=dict(Counter(v["final_row_rejection_reason"] or "candidate" for v in items)),
            first_all_invalid_reason_counts=dict(Counter(
                (v["failures"]["first_all_invalid"]["row_rejection_reason"] or "unrecorded")
                if v["failures"]["first_all_invalid"] is not None else "right_censored_still_has_points"
                for v in items)),
            first_below_min_reason_counts=dict(Counter(
                (v["failures"]["first_below_fixed_min_points"]["row_rejection_reason"] or "partial_loss_cause_unrecorded")
                if v["failures"]["first_below_fixed_min_points"] is not None else "right_censored_still_at_least_min"
                for v in items)),
            over_age_previous_point_counts=dict(Counter(
                c["methods"][method]["failures"]["current_counts"][-2]
                for c in cases if c["final_age_exceeds_limit"])),
            over_age_with_min_points_on_previous_frame=sum(c["final_age_exceeds_limit"] and
                c["methods"][method]["penultimate_min_points"] for c in cases),
            over_age_already_below_min_on_previous_frame=sum(c["final_age_exceeds_limit"] and
                not c["methods"][method]["penultimate_min_points"] for c in cases),
            warning="final age-limit reason can overwrite earlier loss; preceding survival and first-loss ages must be inspected")
    write(outdir / "review.json", dict(config=CONFIG, source_sha256=source_hash,
        input_sha256=inputs, protocol_sha256=manifest["protocol_sha256"],
        age_limit_audit=age_summary, cohorts=cases, fixed_examples=examples, image_validation=image_checks,
        limitations=["Post-hoc chosen examples, not new primary metrics or an unbiased sample.",
            "Saved LK acceptance is not correct material correspondence; apparent displacement may be matching drift.",
            "Signed direction is initial free-seed orientation, not verified winding motion.",
            "No rerun, threshold adjustment, model, video, external image, or label access.",
            "Black/gray views are already cached native crop, not full 4096x3000 frames.",
            "First-invalid sample brackets detector rejection, not true visibility or physical failure time."]))
    if sha(__file__) != source_hash or any(sha(base / key) != digest for key, digest in inputs.items()):
        raise RuntimeError("input or posthoc source changed during audit")
    hashes = {str(path.relative_to(outdir)).replace("\\", "/"): sha(path)
              for path in outdir.rglob("*") if path.is_file()}
    write(outdir / "output_manifest.json", dict(source_sha256=source_hash, input_sha256=inputs,
        output_sha256=hashes, manifest_self_hash="excluded_to_avoid_recursive_hash", completed=True))
    print(json.dumps(dict(outdir=str(outdir), age_limit_audit=age_summary,
                          source_sha256=source_hash, output_files=len(hashes)), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(synthetic_checks()))
    else:
        if args.base is None:
            parser.error("--base required")
        run(args.base, args.base / "posthoc")


if __name__ == "__main__":
    main()
