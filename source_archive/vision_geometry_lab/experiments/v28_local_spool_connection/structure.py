"""Fixed-budget, image-only local winding structure proposals for V28.

Only current original pixels, an independent free-span measurement and optional
automatic flange points enter this module. The rotated working strip is a
sampling coordinate system, not a cylinder reconstruction. Ridge paths propose
where to inspect; V12 paired profiles inspect the *unchanged original image*.
No inferred path point becomes observed merely because the path is smooth.
"""
from __future__ import annotations

from copy import deepcopy
import math

import cv2
import numpy as np

from vision_geometry_lab.v10_ridge_candidates import extract_ridge_candidates
from vision_geometry_lab.v12_profile_refine import refine_paths


DEFAULT_CONFIG = {
    "search_forward_widths": 90.0,
    "search_backward_widths": 2.0,
    "search_half_widths": 18.0,
    "maximum_forward_px": 2400.0,
    "grid_step_widths": 0.25,
    "grid_step_min_px": 1.5,
    "grid_step_max_px": 4.0,
    "ridge_sigma_width_ratios": [0.22, 0.38, 0.55],
    "ridge_method": "frangi_inspired",
    "seed_slices_relative_widths": [-6.0, 0.0, 6.0],
    "seed_peaks_per_slice": 6,
    "seed_phase_budget": "split_equally_bright_dark",
    "seed_nms_widths": 0.65,
    "seed_minimum_response": 0.16,
    "seed_maximum_abs_normal_widths": 16.0,
    "trace_step_widths": 0.45,
    "trace_search_widths": 0.8,
    "trace_minimum_response": 0.12,
    "trace_maximum_abs_slope": 4.0,
    "trace_maximum_each_side_widths": 32.0,
    "trace_maximum_gap_widths": 2.0,
    "trace_minimum_length_widths": 6.0,
    "trace_minimum_observed_nodes": 8,
    "trace_minimum_observed_fraction": 0.5,
    "duplicate_distance_widths": 0.35,
    "maximum_structures": 6,
    "free_span_mask_half_widths": 0.8,
    "profile_method": "paired_edges",
    "minimum_reported_pair_tangential_support": 0.7,
    "maximum_reported_pair_saturation": 0.15,
}


def strip_to_original(uv, entry_xy, direction_xy):
    """Map original-pixel u/v offsets into full-image x/y pixel centers."""
    d = np.asarray(direction_xy, dtype=float)
    d = d / np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    p = np.asarray(uv, dtype=float)
    return np.asarray(entry_xy, dtype=float) + p[..., :1] * d + p[..., 1:] * n


def original_to_strip(xy, entry_xy, direction_xy):
    """Inverse coordinate mapping; no image bounds clipping is performed."""
    d = np.asarray(direction_xy, dtype=float)
    d = d / np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    relative = np.asarray(xy, dtype=float) - np.asarray(entry_xy, dtype=float)
    return np.stack((relative @ d, relative @ n), axis=-1)


def _runs(mask):
    ids = np.flatnonzero(mask)
    if not len(ids):
        return []
    return [[int(run[0]), int(run[-1]) + 1]
            for run in np.split(ids, np.flatnonzero(np.diff(ids) != 1) + 1)]


def _peaks(values, threshold, radius):
    indices = np.flatnonzero((values[1:-1] >= values[:-2]) &
                            (values[1:-1] > values[2:]) &
                            (values[1:-1] >= threshold)) + 1
    selected = []
    for i in sorted(indices, key=lambda j: (-float(values[j]), int(j))):
        if all(abs(int(i) - previous) >= radius for previous in selected):
            selected.append(int(i))
    return selected


def _make_strip(image, entry, direction, width, cfg, roi):
    step = float(np.clip(width * cfg["grid_step_widths"],
                         cfg["grid_step_min_px"], cfg["grid_step_max_px"]))
    length = min(cfg["search_forward_widths"] * width, cfg["maximum_forward_px"])
    u = np.arange(-cfg["search_backward_widths"] * width, length + step, step)
    v = np.arange(-cfg["search_half_widths"] * width,
                  cfg["search_half_widths"] * width + step, step)
    uv = np.stack(np.broadcast_arrays(u[None, :], v[:, None]), axis=-1)
    xy = strip_to_original(uv, entry, direction)
    x, y = xy[..., 0], xy[..., 1]
    # The mask covers the full bilinear footprint. Reflected or constant border
    # pixels never count as observations or become endpoints of local curves.
    valid = ((x >= 1) & (x <= image.shape[1] - 2) &
             (y >= 1) & (y <= image.shape[0] - 2))
    if roi is not None:
        x0, y0, x1, y1 = roi
        valid &= (x >= x0) & (x < x1 - 1) & (y >= y0) & (y < y1 - 1)
    sampled = cv2.remap(image, x.astype(np.float32), y.astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    free_mask = ((u[None, :] < -width * .2) &
                 (np.abs(v[:, None]) <= cfg["free_span_mask_half_widths"] * width))
    # Free-span exclusion is a feature validity mask. No black stripe is painted
    # before filtering, which would introduce a false edge at its boundary.
    feature_valid = valid & ~free_mask
    corners = strip_to_original(np.array([[u[0], v[0]], [u[-1], v[0]],
                                          [u[-1], v[-1]], [u[0], v[-1]]]),
                                entry, direction)
    box = [max(0, int(math.floor(corners[:, 0].min()))),
           max(0, int(math.floor(corners[:, 1].min()))),
           min(image.shape[1], int(math.ceil(corners[:, 0].max())) + 1),
           min(image.shape[0], int(math.ceil(corners[:, 1].max())) + 1)]
    if roi is not None:
        box = [max(box[0], roi[0]), max(box[1], roi[1]),
               min(box[2], roi[2]), min(box[3], roi[3])]
    return sampled, valid, feature_valid, u, v, step, box


def _seed_focus(u, entry, direction, width, arc_points):
    """Choose a search slice using automatic flange proximity, never a label."""
    if arc_points is not None:
        points = np.asarray(arc_points, dtype=float)
        if points.ndim == 2 and points.shape[1] == 2 and len(points):
            points = points[np.isfinite(points).all(axis=1)]
            uv = original_to_strip(points, entry, direction)
            keep = ((uv[:, 0] >= 2 * width) & (uv[:, 0] <= u[-1] - 2 * width) &
                    (np.abs(uv[:, 1]) <= 16 * width))
            if np.any(keep):
                candidates = uv[keep]
                target = candidates[np.argmin(np.abs(candidates[:, 1]))]
                return float(target[0]), "automatic_flange_nearest_free_ray"
    return float(max(3 * width, .5 * u[-1])), "fixed_middle_of_forward_search"


def _trace(seed, response, tangent, polarity, valid, u, v, step, width, cfg):
    row0, col0 = seed
    stride = max(1, int(round(cfg["trace_step_widths"] * width / step)))
    radius = max(2, int(math.ceil(cfg["trace_search_widths"] * width / step)))
    limit = max(1, int(math.ceil(cfg["trace_maximum_each_side_widths"] * width /
                                 (stride * step))))

    def walk(sign):
        row = float(row0)
        initial = tangent[row0, col0]
        slope = float(np.clip(initial[1] / max(abs(initial[0]), .15) *
                              (1 if initial[0] >= 0 else -1),
                              -cfg["trace_maximum_abs_slope"], cfg["trace_maximum_abs_slope"]))
        previous_polarity = int(polarity[row0, col0])
        gap = 0.0
        result = []
        for offset in range(1, limit + 1):
            col = col0 + sign * stride * offset
            if not 1 <= col < len(u) - 1:
                break
            predicted = row + sign * stride * slope
            lo, hi = max(1, int(round(predicted)) - radius), min(len(v) - 1, int(round(predicted)) + radius + 1)
            if hi <= lo:
                break
            values = response[lo - 1:hi + 1, col].copy()
            indices = _peaks(values, cfg["trace_minimum_response"], 1)
            options = []
            for local in indices:
                candidate_row = lo - 1 + local
                if not valid[candidate_row, col]:
                    continue
                candidate_tangent = tangent[candidate_row, col]
                # Near-vertical ridges in the u/v strip have no well-defined
                # next u-column. Keep them out rather than jumping across rows.
                if abs(candidate_tangent[0]) < .15:
                    continue
                observed_slope = (candidate_row - row) / (sign * stride)
                if abs(observed_slope) > cfg["trace_maximum_abs_slope"]:
                    continue
                travel = np.array([1., observed_slope])
                travel /= np.linalg.norm(travel)
                alignment = abs(float(travel @ candidate_tangent))
                score = (float(response[candidate_row, col]) + .30 * alignment -
                         .25 * abs(candidate_row - predicted) / max(radius, 1) -
                         .05 * (int(polarity[candidate_row, col]) != previous_polarity))
                options.append((score, candidate_row, observed_slope))
            if options:
                _, next_row, observed_slope = max(options, key=lambda item: (item[0], -item[1]))
                row = float(next_row)
                slope = .65 * slope + .35 * observed_slope
                previous_polarity = int(polarity[next_row, col])
                gap = 0.
                result.append((float(u[col]), float(v[next_row]), True,
                               float(response[next_row, col]), previous_polarity))
            else:
                gap += stride * step
                next_row = int(round(predicted))
                if (gap > cfg["trace_maximum_gap_widths"] * width or
                        not 0 <= next_row < len(v) or not valid[next_row, col]):
                    break
                row = predicted
                result.append((float(u[col]), float(v[0] + row * step), False, 0., previous_polarity))
        while result and not result[-1][2]:
            result.pop()
        return result

    center = (float(u[col0]), float(v[row0]), True,
              float(response[row0, col0]), int(polarity[row0, col0]))
    nodes = list(reversed(walk(-1))) + [center] + walk(1)
    observed = np.array([node[2] for node in nodes], dtype=bool)
    length = nodes[-1][0] - nodes[0][0]
    if (len(nodes) < 2 or int(observed.sum()) < cfg["trace_minimum_observed_nodes"] or
            observed.mean() < cfg["trace_minimum_observed_fraction"] or
            length < cfg["trace_minimum_length_widths"] * width):
        return None
    return dict(uv=np.array([node[:2] for node in nodes]),
                ridge_observed=observed,
                response=np.array([node[3] for node in nodes]),
                polarity=np.array([node[4] for node in nodes]),
                seed_uv=[float(u[col0]), float(v[row0])])


def _duplicate(first, second, width, cfg):
    a, b = first["uv"], second["uv"]
    lo, hi = max(a[0, 0], b[0, 0]), min(a[-1, 0], b[-1, 0])
    if hi - lo < 3 * width:
        return False
    coordinates = np.linspace(lo, hi, 24)
    distance = np.abs(np.interp(coordinates, a[:, 0], a[:, 1]) -
                      np.interp(coordinates, b[:, 0], b[:, 1]))
    return bool(np.median(distance) < cfg["duplicate_distance_widths"] * width)


def _profile_structure(image, path, width, entry, direction, index, cfg):
    xy = strip_to_original(path["uv"], entry, direction)
    result = refine_paths(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), [xy], width,
                          cfg["profile_method"])
    records = result["diagnostics"]["records"]
    points = np.asarray(result["segments"][0], dtype=float)
    tangents = np.gradient(points, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-8)
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    accepted, pair_supported, unavailable = [], [], []
    left, right, widths, evidence = [], [], [], []
    for i, record in enumerate(records):
        current = record["evidence"]
        pair = current.get("edge_pair")
        pair_ok = bool(pair is not None and
                       current.get("tangential_support", 0.) >= cfg["minimum_reported_pair_tangential_support"] and
                       current.get("saturated_fraction", 1.) <= cfg["maximum_reported_pair_saturation"])
        accepted.append(bool(record["accepted"]))
        pair_supported.append(pair_ok)
        reason = record["reason"]
        bad = (reason in {"profile_footprint_outside_source_image", "local_tangent_unavailable", "saturated_profile_center"})
        unavailable.append(bad)
        # These two coordinates are measurements only when pair_supported is
        # true. Absent boundaries stay null; nominal +/- width is never emitted
        # as a measured edge. V12 final_xy can be an unrefined hypothesis.
        if pair_ok:
            base = np.asarray(record["input_xy"], dtype=float)
            proposed_offset = float(record.get("proposed_offset_px") or 0.)
            shift = np.asarray(record["proposed_xy"], dtype=float) - base
            n = shift / proposed_offset if abs(proposed_offset) > 1e-5 else normals[i]
            left.append((base + float(pair[0]) * n).tolist())
            right.append((base + float(pair[1]) * n).tolist())
            widths.append(float(pair[1] - pair[0]))
        else:
            left.append(None)
            right.append(None)
            widths.append(None)
        evidence.append(dict(reason=reason, accepted=bool(record["accepted"]),
                             pair_supported=pair_ok, candidate_count=record["candidate_count"],
                             contrast_gray=current.get("contrast_gray"),
                             tangential_support=current.get("tangential_support"),
                             saturated_fraction=current.get("saturated_fraction"),
                             polarity=(record["top_candidates"][0]["polarity"]
                                       if record["top_candidates"] else None)))
    supported = np.asarray(accepted, dtype=bool)
    unavailable_mask = np.asarray(unavailable, dtype=bool)
    inferred = ~supported & ~unavailable_mask
    sample_uv = original_to_strip(points, entry, direction)
    ridge_at_sample = np.interp(sample_uv[:, 0], path["uv"][:, 0],
                                path["ridge_observed"].astype(float)) >= .999
    polarity = "bright_phase" if np.mean(path["polarity"] > 0) >= .5 else "dark_phase_rope_or_gap"
    return dict(id=f"S{index + 1:02d}", points_xy=points.tolist(),
                widths_px=widths, nominal_width_px=float(width), left_xy=left, right_xy=right,
                supported=supported.tolist(), candidate_supported=pair_supported,
                inferred=inferred.tolist(), unavailable=unavailable_mask.tolist(),
                ridge_supported=ridge_at_sample.tolist(),
                supported_ranges=_runs(supported), inferred_ranges=_runs(inferred),
                unavailable_ranges=_runs(unavailable_mask),
                polarity=polarity, source="v10_multiscale_ridge_then_v12_original_pixel_paired_profiles",
                identity_confirmed=False, physical_center_confirmed=False,
                image_pair_does_not_resolve_rope_vs_gap=True,
                extraction_quality=float(path["extraction_quality"]),
                ridge_support_fraction=float(path["ridge_observed"].mean()),
                paired_refinement_fraction=float(supported.mean()),
                paired_candidate_fraction=float(np.mean(pair_supported)),
                seed_uv=path["seed_uv"], profile_records=evidence)


def extract_local_structure(image_bgr, entry_xy, direction_xy, width_px,
                            roi_xyxy=None, config=None, arc_points_xy=None):
    """Return at most six local curve hypotheses with full-image coordinates.

    ``supported`` is strict V12 photometric refinement, not identity truth.
    ``candidate_supported`` also retains plausible dark/gap or competing-pair
    explanations. Missing width/boundaries remain None. The optional roi and
    automatic flange points must not come from target-path annotations.
    """
    cfg = deepcopy(DEFAULT_CONFIG)
    if config:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError(f"unknown structure config keys: {sorted(unknown)}")
        cfg.update(config)
    image = np.asarray(image_bgr)
    entry, direction = np.asarray(entry_xy, dtype=float), np.asarray(direction_xy, dtype=float)
    width = float(width_px)
    if (image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or
            min(image.shape[:2]) < 8):
        raise ValueError("image_bgr must be original HxWx3 uint8")
    if (entry.shape != (2,) or direction.shape != (2,) or
            not np.isfinite(entry).all() or not np.isfinite(direction).all() or
            np.linalg.norm(direction) < 1e-8 or not math.isfinite(width) or not 1 <= width <= 300):
        raise ValueError("finite entry/direction and independent width [1,300] required")
    direction = direction / np.linalg.norm(direction)
    if roi_xyxy is not None:
        roi_xyxy = list(map(int, roi_xyxy))
        if not (len(roi_xyxy) == 4 and 0 <= roi_xyxy[0] < roi_xyxy[2] <= image.shape[1] and
                0 <= roi_xyxy[1] < roi_xyxy[3] <= image.shape[0]):
            raise ValueError("roi_xyxy must be half-open source image bounds")
    strip, valid, feature_valid, u, v, step, box = _make_strip(image, entry, direction, width, cfg, roi_xyxy)
    scaled_width = width / step
    sigmas = np.clip(np.asarray(cfg["ridge_sigma_width_ratios"]) * scaled_width, .5, 32.).tolist()
    fields = extract_ridge_candidates(strip, method=cfg["ridge_method"], width_px=scaled_width,
                                      sigmas_px=sigmas, polarity="both")
    response = fields["response"].copy()
    # Require the smoothing context to contain real image samples. Broad
    # responses beside the image/ROI border cannot create an artificial ridge.
    guard = max(1, int(math.ceil(2 * max(sigmas))))
    safe = cv2.erode(feature_valid.astype(np.uint8),
                     np.ones((2 * guard + 1, 2 * guard + 1), np.uint8)) > 0
    response[~safe] = 0.
    phase_responses = {1: fields["bright_response"].copy(),
                       -1: fields["dark_response"].copy()}
    for value in phase_responses.values():
        value[~safe] = 0.
    focus, focus_source = _seed_focus(u, entry, direction, width, arc_points_xy)
    seeds = []
    seed_columns = []
    for relative in cfg["seed_slices_relative_widths"]:
        column = int(np.argmin(abs(u - np.clip(focus + relative * width, 2 * width, u[-1] - 2 * width))))
        if column in seed_columns:
            continue
        seed_columns.append(column)
        for phase in (1, -1):
            values = phase_responses[phase][:, column].copy()
            values[abs(v) > cfg["seed_maximum_abs_normal_widths"] * width] = 0.
            values[np.abs(fields["tangent_xy"][:, column, 0]) < .15] = 0.
            peaks = _peaks(values, cfg["seed_minimum_response"],
                           max(1, int(round(cfg["seed_nms_widths"] * width / step))))
            # Each phase gets the same fixed budget. Strong dark grooves cannot
            # consume every seed and silently remove the bright-body family.
            peaks.sort(key=lambda row: (-(float(values[row]) - .18 * abs(v[row]) / (16 * width)), abs(v[row]), row))
            budget = max(1, int(cfg["seed_peaks_per_slice"]) // 2)
            seeds.extend((row, column, phase) for row in peaks[:budget])
    paths = []
    for seed in seeds:
        phase = seed[2]
        path = _trace(seed[:2], phase_responses[phase], fields["tangent_xy"],
                       np.full_like(fields["polarity_map"], phase),
                       safe, u, v, step, width, cfg)
        if path is None:
            continue
        observed = path["ridge_observed"]
        quality = float(np.mean(path["response"][observed]) * observed.mean())
        proximity = min(1., float(np.min(np.linalg.norm(path["uv"], axis=1))) / (40 * width))
        path["extraction_quality"] = quality - .15 * proximity
        paths.append(path)
    paths.sort(key=lambda p: (-p["extraction_quality"], p["seed_uv"][0], p["seed_uv"][1]))
    unique = []
    for path in paths:
        if not any(_duplicate(path, previous, width, cfg) for previous in unique):
            unique.append(path)
    # Preserve both phase explanations at the output budget too. This selection
    # does not prefer the phase later accepted by profile refinement.
    phases = {phase: [p for p in unique if int(p["polarity"][0]) == phase]
              for phase in (1, -1)}
    selected = []
    for rank in range(max(len(phases[1]), len(phases[-1]))):
        for phase in (1, -1):
            if rank < len(phases[phase]) and len(selected) < int(cfg["maximum_structures"]):
                selected.append(phases[phase][rank])
    structures = [_profile_structure(image, path, width, entry, direction, i, cfg)
                  for i, path in enumerate(selected)]
    return dict(roi_xyxy=box, structures=structures, config=cfg,
                coordinate_system="original_image_pixel_centers_xy",
                status="local_structure_candidates" if structures else "NO_SUPPORTED_LOCAL_STRUCTURE",
                diagnostics=dict(entry_xy=entry.tolist(), direction_xy=direction.tolist(),
                                 independent_width_px=width, strip_step_px=step,
                                 strip_u_range_px=[float(u[0]), float(u[-1])],
                                 strip_v_range_px=[float(v[0]), float(v[-1])],
                                 focus_u_px=focus, focus_source=focus_source,
                                 seed_u_px=[float(u[column]) for column in seed_columns],
                                 seed_budget=int(cfg["seed_peaks_per_slice"] * len(seed_columns)),
                                 seeds_found=len(seeds), linked_paths=len(paths),
                                 unique_paths=len(unique), retained_paths=len(selected),
                                 truncated_structures=max(0, len(unique) - len(selected)),
                                 ridge_metadata=fields["metadata"],
                                 ridge_scales_original_px=[float(sigma * step) for sigma in sigmas],
                                 original_image_used_for_final_profiles=True,
                                 bright_dark_and_half_phase_not_resolved_by_period=True,
                                 no_global_constant_pitch_constraint=True,
                                 labels_used=False, hardware_access=False,
                                 inference_is_image_evidence=False))
