"""Observed adjacent-F3-boundary lanes for H2, not material identities.

All returned coordinates are H1 half-size model-crop xy:
original_xy = 2 * crop_xy + [0, 1050]. Nothing is extrapolated to missing rows.
Boundary IDs identify connected components in the saved F3 raster, not rope IDs.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib.util
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARAMETERS = {
    "column_step_half_px": 3,
    "minimum_column_gap_half_px": 3.0,
    "maximum_column_gap_half_px": 60.0,
    "minimum_samples": 6,
    "minimum_span_half_px": 15.0,
    "maximum_boundary_raster_thickness_half_px": 6,
    "local_spacing_ratio_min": 0.45,
    "local_spacing_ratio_max": 1.8,
    "spacing_neighborhood_pairs": 3,
    "refine_max_fraction_spacing": 0.15,
    "refine_max_half_px": 2.0,
    "refine_candidate_count": 7,
    "refine_required_score_gain": 0.02,
    "gray_contrast_std_floor": 0.02,
    "heat_shape_scale": 0.15,
    "raw_support_threshold": 0.25,
    "heat_support_threshold": 0.15,
    "heat_level_support_threshold": 0.15,
}
COORDINATES = "half_size_model_crop_xy; original_xy=2*xy+[0,1050]"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_surface_priors(job):
    """Read only an allowed open400 job, including the original SAM2 mask.

    H1's surface_half is SAM2 union B1. This adapter additionally supplies
    sam2_surface_half, without replacing or silently renaming that union.
    A6/A7 remain auxiliary records and are never used to choose lane identity.
    The parent may instead call H1 load_priors and pass its result to build_tracks.
    """
    path = HERE.parent / "postprocess_b_20261030_v1/priors.py"
    spec = importlib.util.spec_from_file_location("h2_readonly_h1_priors", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    priors = module.load_priors(job)
    mask_path = (ROOT / "数据资产/05_派生与实验结果/camera_condition_sam2_20261009_v1/all_001/body_branches_002/masks"
                 / (priors["camera_role"] + "_camera") / (job["id"] + ".png"))
    mask = cv2.imdecode(np.fromfile(mask_path, np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None or tuple(mask.shape) != tuple(job["shape"]):
        raise ValueError("Original SAM2 mask missing or shape mismatch")
    priors["sam2_surface_half"] = cv2.resize(mask, (mask.shape[1] // 2, mask.shape[0] // 2), interpolation=cv2.INTER_NEAREST)
    priors["input_dependency_sha256"][str(mask_path)] = _sha(mask_path)
    priors["surface_policy"] = "saved_original_SAM2_v2_for_tracks; H1_union_B1_retained_separately"
    return priors


def _sample(field, xy):
    xy = np.asarray(xy, np.float32)
    shape = xy.shape[:-1]
    p = xy.reshape(-1, 2)
    return cv2.remap(np.asarray(field, np.float32), p[:, 0][None], p[:, 1][None],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).reshape(shape)


def _smooth(values, count=5):
    values = np.asarray(values, float)
    return np.convolve(np.pad(values, (count // 2, count // 2), mode="edge"), np.ones(count) / count, mode="valid")


def _appearance(gray, heat, center, normal, widths):
    # These samples test local profile support; their peaks are not ground truth.
    offsets = np.array([-.50, -.40, -.20, 0., .20, .40, .50], np.float32)
    q = center[:, None, :] + normal[:, None, :] * widths[:, None, None] * offsets[None, :, None]
    raw = _sample(gray, q)
    response = _sample(heat, q)
    rc = raw[:, 2:5].mean(1)
    hc = response[:, 2:5].mean(1)
    raw_dual = np.minimum(rc - raw[:, :2].mean(1), rc - raw[:, -2:].mean(1))
    heat_dual = np.minimum(hc - response[:, :2].mean(1), hc - response[:, -2:].mean(1))
    raw_score = np.clip(raw_dual / np.maximum(raw.std(1), PARAMETERS["gray_contrast_std_floor"]), 0, 1)
    heat_score = np.clip(heat_dual / PARAMETERS["heat_shape_scale"], 0, 1)
    # Raw gray supports both boundaries and a brighter body. Dark phase remains
    # unsupported rather than reversing polarity to manufacture a positive score.
    return raw_score, heat_score, np.clip(hc, 0, 1), raw_dual, heat_dual


def _finish_track(samples, pair, serial, gray, heat, inferred):
    a = np.asarray(samples, float)
    mid = np.c_[a[:, 0], (a[:, 1] + a[:, 2]) * .5]
    upper = np.c_[a[:, 0], a[:, 1]]
    lower = np.c_[a[:, 0], a[:, 2]]
    tangent = np.c_[np.gradient(mid[:, 0]), np.gradient(_smooth(mid[:, 1], 5))]
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-6)
    normal = np.c_[-tangent[:, 1], tangent[:, 0]]
    widths = (a[:, 2] - a[:, 1]) * np.abs(normal[:, 1])
    widths = np.maximum(widths, 1.)
    raw0, hs0, hl0, _, _ = _appearance(gray, heat, mid, normal, widths)
    base = .35 * raw0 + .65 * hs0
    bound = np.minimum(PARAMETERS["refine_max_half_px"], PARAMETERS["refine_max_fraction_spacing"] * widths)
    candidates = np.linspace(-1, 1, PARAMETERS["refine_candidate_count"])
    score = []
    for factor in candidates:
        rp, hp, _, _, _ = _appearance(gray, heat, mid + normal * (bound * factor)[:, None], normal, widths)
        score.append(.35 * rp + .65 * hp - .02 * abs(factor))
    scores = np.asarray(score)
    best = np.argmax(scores, axis=0)
    shift = bound * candidates[best]
    shift[scores[best, np.arange(len(best))] < base + PARAMETERS["refine_required_score_gain"]] = 0
    shift = np.clip(_smooth(shift, 5), -bound, bound)
    points = mid + normal * shift[:, None]
    # Numeric confinement to the source pair's narrow interior is a hard gate.
    if np.any(np.abs(shift) > .150001 * widths):
        raise AssertionError("Refinement escaped its adjacent boundary pair")
    raw, heat_shape, heat_level, raw_dual, heat_dual = _appearance(gray, heat, points, normal, widths)
    boundary_inferred = (a[:, 3] > 0) | (a[:, 4] > 0)
    raw_ok = raw >= PARAMETERS["raw_support_threshold"]
    heat_ok = ((heat_shape >= PARAMETERS["heat_support_threshold"]) &
               (heat_level >= PARAMETERS["heat_level_support_threshold"]))
    observed = ~boundary_inferred
    supported = raw_ok & heat_ok & observed
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    score_values = .45 * raw + .35 * heat_shape + .20 * heat_level
    return {
        "track_id": f"lane_{pair[0]:04d}_{pair[1]:04d}_{serial:02d}",
        "boundary_ids": [f"f3cc_{pair[0]:04d}", f"f3cc_{pair[1]:04d}"],
        "points_xy": points.tolist(), "midpoint_xy": mid.tolist(),
        "upper_boundary_xy": upper.tolist(), "lower_boundary_xy": lower.tolist(),
        "spacing_px": widths.tolist(), "vertical_spacing_px": (a[:, 2] - a[:, 1]).tolist(),
        "width_px": widths.tolist(), "width_is_calibrated_rope_diameter": False,
        "refine_offset_px": shift.tolist(), "f3_pair_supported": [True] * len(a),
        "f3_pair_observed": observed.tolist(), "f3_pair_inferred": boundary_inferred.tolist(),
        "upper_boundary_inferred": (a[:, 3] > 0).tolist(), "lower_boundary_inferred": (a[:, 4] > 0).tolist(),
        "raw_score": raw.tolist(), "heat_score": heat_shape.tolist(), "heat_level": heat_level.tolist(),
        "raw_signed_center_minus_sides": raw_dual.tolist(), "heat_signed_center_minus_sides": heat_dual.tolist(),
        "image_supported": raw_ok.tolist(), "heat_supported": heat_ok.tolist(), "supported": supported.tolist(),
        "score": float(np.mean(score_values)), "length_px": length,
        "support_summary": {"sample_count": len(a), "paired_raster_fraction": 1.,
                            "observed_pair_fraction": float(observed.mean()), "raw_supported_fraction": float(raw_ok.mean()),
                            "heat_supported_fraction": float(heat_ok.mean()), "joint_supported_fraction": float(supported.mean()),
                            "spacing_median_half_px": float(np.median(widths)),
                            "spacing_range_half_px": [float(widths.min()), float(widths.max())],
                            "maximum_refine_offset_half_px": float(np.max(np.abs(shift)))},
        "stop_reason": "saved_adjacent_boundary_pair_or_surface_support_ends; no_extrapolation",
        "identity": "unknown", "same_material_identity_confirmed": False,
        "coordinate_system": COORDINATES,
    }


def build_tracks(priors, heat, gray=None):
    """Return all locally adjacent supported-domain F3 lane fragments.

    Inputs: H1 load_priors output; heat is finite crop-half gray975x2048.
    gray is optional gray0..1 of the same crop; otherwise derive from original.
    The presence of a lane is geometric candidate evidence, never target identity.
    Raw/B support fields stay separate. Missing F3 sections are never filled.
    """
    merged = np.asarray(priors["f3_merged_half"])[525:] > 0
    inferred = np.asarray(priors["f3_inferred_half"])[525:] > 0
    chosen_surface = priors.get("sam2_surface_half", priors["surface_half"])
    surface = np.asarray(chosen_surface)[525:] > 0
    surface_policy = "original_SAM2_v2" if "sam2_surface_half" in priors else "H1_SAM2_union_B1"
    heat = np.asarray(heat, np.float32)
    if gray is None:
        gray = cv2.cvtColor(np.asarray(priors["original_half"])[525:], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
    gray = np.asarray(gray, np.float32)
    if any(a.shape != merged.shape for a in (gray, heat, inferred, surface)):
        raise ValueError("Expected consistent H1 crop-half fields")
    if not np.isfinite(gray).all() or not np.isfinite(heat).all() or gray.min() < 0 or gray.max() > 1.0001 or heat.min() < 0 or heat.max() > 1.0001:
        raise ValueError("Expected finite raw gray and frozen response in [0,1]")
    count, components = cv2.connectedComponents(merged.astype(np.uint8), connectivity=8)
    # Dilating inferred only affects the status flag; it never adds a boundary.
    inf_near = cv2.dilate(inferred.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    pairs = defaultdict(list)
    boundary_samples = defaultdict(list)
    rejected = defaultdict(int)
    columns = []
    raster_gaps = []
    step = PARAMETERS["column_step_half_px"]
    for x in range(0, merged.shape[1], step):
        ys = np.flatnonzero(merged[:, x])
        if not len(ys):
            continue
        runs = np.split(ys, np.flatnonzero(np.diff(ys) > 1) + 1)
        entries = []
        for run in runs:
            if len(run) > PARAMETERS["maximum_boundary_raster_thickness_half_px"]:
                # A junction/thick uncertain strip is not split into fake lines.
                entries.append(None)
                rejected["thick_raster_run"] += 1
                continue
            label = int(components[run[0], x])
            y = float(np.mean(run))
            inf = bool(inf_near[run, x].any())
            entries.append((y, label, inf))
            boundary_samples[label].append([float(x), y, inf])
        valid_gaps = [entries[k + 1][0] - entries[k][0] if entries[k] is not None and entries[k + 1] is not None else np.nan
                      for k in range(len(entries) - 1)]
        columns.append((x, entries, valid_gaps))
        raster_gaps.extend(g for g in valid_gaps if np.isfinite(g) and
                           PARAMETERS["minimum_column_gap_half_px"] <= g <= PARAMETERS["maximum_column_gap_half_px"])
    # Sparse columns cannot define their own pitch from one possible doubled
    # gap. Use only the image's actually present F3 gap distribution as fallback.
    # This number rejects dubious candidates; it never fills a missing boundary.
    raster_median_gap = float(np.median(raster_gaps)) if raster_gaps else None
    for x, entries, valid_gaps in columns:
        for k in range(len(entries) - 1):
            upper, lower = entries[k:k + 2]
            if upper is None or lower is None:
                continue
            y0, b0, inf0 = upper
            y1, b1, inf1 = lower
            gap = y1 - y0
            if b0 == b1:
                rejected["same_component_branch_or_loop"] += 1
                continue
            if not PARAMETERS["minimum_column_gap_half_px"] <= gap <= PARAMETERS["maximum_column_gap_half_px"]:
                rejected["gap_outside_pixel_sanity_bounds"] += 1
                continue
            n = PARAMETERS["spacing_neighborhood_pairs"]
            neighborhood = np.asarray(valid_gaps[max(0, k - n):k + n + 1])
            neighborhood = neighborhood[np.isfinite(neighborhood) & (neighborhood >= PARAMETERS["minimum_column_gap_half_px"])]
            median = float(np.median(neighborhood)) if len(neighborhood) >= 3 else raster_median_gap
            if median is None:
                rejected["insufficient_spacing_evidence"] += 1
                continue
            if not PARAMETERS["local_spacing_ratio_min"] <= gap / median <= PARAMETERS["local_spacing_ratio_max"]:
                rejected["local_spacing_ambiguity"] += 1
                continue
            ys_test = np.rint(np.linspace(y0, y1, 5)).astype(int)
            if not surface[ys_test, x].all():
                rejected["outside_surface_support"] += 1
                continue
            pairs[(b0, b1)].append([float(x), y0, y1, inf0, inf1])
    tracks = []
    for pair in sorted(pairs):
        samples = np.asarray(pairs[pair], float)
        # No bridging at all: both saved boundaries must exist at every column.
        breaks = np.flatnonzero(np.diff(samples[:, 0]) > step + .1) + 1
        for serial, run in enumerate(np.split(samples, breaks)):
            if len(run) < PARAMETERS["minimum_samples"] or run[-1, 0] - run[0, 0] < PARAMETERS["minimum_span_half_px"]:
                rejected["too_short_pair_fragment"] += 1
                continue
            tracks.append(_finish_track(run, pair, serial, gray, heat, inferred))
    boundaries = []
    for label in sorted(boundary_samples):
        values = np.asarray(boundary_samples[label], float)
        boundaries.append({"boundary_id": f"f3cc_{label:04d}", "points_xy": values[:, :2].tolist(),
                           "inferred": (values[:, 2] > 0).tolist(),
                           "id_definition": "8-connected component of saved F3 raster; not physical rope identity"})
    return {
        "tracks": tracks, "boundaries": boundaries, "coordinate_system": COORDINATES,
        "diagnostics": {"parameters": dict(PARAMETERS), "f3_component_count": int(count - 1),
                        "sampled_boundary_count": len(boundaries), "track_fragment_count": len(tracks),
                        "paired_sample_count": sum(len(t["points_xy"]) for t in tracks),
                        "total_track_length_half_px": sum(t["length_px"] for t in tracks),
                        "rejected": dict(rejected), "surface_policy": surface_policy,
                        "observed_raster_median_vertical_gap_half_px": raster_median_gap,
                        "sparse_column_spacing_fallback": "same_image_existing_F3_gaps_not_calibrated_width",
                        "pitch_prior_used_to_generate_or_fill_lanes": False,
                        "fixed_spacing_lines_generated": 0, "extrapolated_points": 0,
                        "center_refinement_uses_frozen_B_and_original_gray": True,
                        "all_source_geometry_is_candidate_not_truth": True,
                        "same_material_identity_confirmed": False, "labels_read": False},
    }


def draw_tracks(crop_bgr, result, thickness=1):
    """Background lane visualization; green supported, orange inferred/unknown."""
    image = np.asarray(crop_bgr).copy()
    for track in result["tracks"]:
        points = np.rint(np.asarray(track["points_xy"])).astype(np.int32)
        for k in range(1, len(points)):
            color = (80, 170, 80) if track["supported"][k] and track["supported"][k - 1] else (65, 110, 180)
            cv2.line(image, tuple(points[k - 1]), tuple(points[k]), color, thickness, cv2.LINE_AA)
    return image
