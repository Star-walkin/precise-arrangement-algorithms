"""Offline, original-pixel refinement of a *proposed rope-body* centerline.

The input polyline identifies a candidate, not a confirmed rope instance. Local
normal profiles seek two opposite, independently supported body edges. Their
midpoint is the apparent image-body center; it is not a measured 3-D axis.

A width prior is necessary here: an isolated dark rope and an equally wide dark
groove can have IDENTICAL pixels. Neither contrast polarity nor subpixel fitting
resolves that semantic ambiguity. With no width prior this module abstains.
Even with a prior, ``identity_confirmed`` is always false. A learned body mask,
independent identity evidence and validation remain separate responsibilities.

No point insertion, interpolation through missing observations, old frames,
flange attraction, camera access, or physical/control decision is performed.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def _record(seed, reason):
    return dict(seed_xy=[float(v) for v in seed], refined_xy=None,
                edge_a_xy=None, edge_b_xy=None, width_px=None,
                valid=False, reason=reason, confidence=0.,
                identity_confirmed=False, diagnostics={})


def _result(records, reason=""):
    count = sum(record["valid"] for record in records)
    return dict(valid=bool(count), status="complete_candidate" if count == len(records) and count else
                ("partial_candidate" if count else "unavailable"),
                reason=reason or ("paired_edge_candidates_identity_unconfirmed" if count else "no_valid_two_sided_body_candidates"),
                identity_confirmed=False, physical_contact_confirmed=False,
                records=records,
                diagnostics=dict(method="paired_original_pixel_normal_edges_v5",
                                 input_sample_count=len(records), valid_sample_count=count,
                                 inserted_points=0, interpolated_points=0,
                                 center_definition="midpoint_of_two_apparent_body_edges_in_image_normal_section",
                                 confidence_is_calibrated_probability=False,
                                 width_prior_is_not_identity_confirmation=True,
                                 unresolved_case="same_width_dark_body_and_dark_groove_can_be_photometrically_identical"))


def _per_point(value, count):
    try:
        array = np.asarray(value, float)
        if array.ndim == 0:
            array = np.full(count, float(array))
        if array.shape != (count,) or not np.isfinite(array).all() or np.any(array <= 0):
            return None
        return array
    except (ValueError, TypeError):
        return None


def _tangent(points, index):
    before = index - 1
    while before >= 0 and np.linalg.norm(points[index] - points[before]) <= 1e-6:
        before -= 1
    after = index + 1
    while after < len(points) and np.linalg.norm(points[after] - points[index]) <= 1e-6:
        after += 1
    directions = []
    if before >= 0:
        difference = points[index] - points[before]
        directions.append(difference / np.linalg.norm(difference))
    if after < len(points):
        difference = points[after] - points[index]
        directions.append(difference / np.linalg.norm(difference))
    if not directions:
        return None
    direction = np.sum(directions, axis=0)
    length = np.linalg.norm(direction)
    if length < .3:
        return None
    return direction / length


def _profiles(image, gray, clipped_image, point, tangent, width, shift):
    normal = np.array([-tangent[1], tangent[0]])
    step = .5
    # Fixed width ratios, independent of the source frame or any manual dots.
    half_range = shift + .825 * width + max(3., .30 * width)
    cross = np.arange(-half_range, half_range + step * .5, step)
    along = np.linspace(-.55 * width, .55 * width, 11)
    xy = point + along[:, None, None] * tangent + cross[None, :, None] * normal
    if (np.min(xy[..., 0]) < 1 or np.max(xy[..., 0]) > image.shape[1] - 2 or
            np.min(xy[..., 1]) < 1 or np.max(xy[..., 1]) > image.shape[0] - 2):
        return None
    map_x, map_y = xy[..., 0].astype(np.float32), xy[..., 1].astype(np.float32)
    sample = cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR)
    clipped = cv2.remap(clipped_image, map_x, map_y, cv2.INTER_LINEAR)
    # Tangential median suppresses moving braid facets; it does not make a
    # strong isolated groove semantically different from a dark rope body.
    smooth_rows = cv2.GaussianBlur(sample, (0, 0), sigmaX=1.3, sigmaY=0.01)
    profile = np.median(smooth_rows, axis=0)
    derivative = np.gradient(profile, step)
    row_derivative = np.gradient(smooth_rows, step, axis=1)
    noise = max(.5, float(np.median(np.abs(np.diff(profile))) / .954))
    return dict(normal=normal, offsets=cross, profile=profile, derivative=derivative,
                row_derivative=row_derivative, clipped=clipped, step=step, noise=noise)


def _peaks(profile):
    gradient = profile["derivative"]
    magnitude = np.abs(gradient)
    threshold = max(2., profile["noise"] * 1.5)
    indices = np.flatnonzero((magnitude[1:-1] > magnitude[:-2]) &
                             (magnitude[1:-1] >= magnitude[2:]) &
                             (magnitude[1:-1] >= threshold)) + 1
    peaks = []
    for index in indices:
        a, b, c = magnitude[index - 1:index + 2]
        denominator = float(a - 2 * b + c)
        delta = float(np.clip(.5 * (a - c) / denominator, -.5, .5)) if abs(denominator) > 1e-8 else 0.
        offset = float(profile["offsets"][index] + delta * profile["step"])
        local = profile["row_derivative"][:, max(0, index - 2):index + 3]
        polarity = 1 if gradient[index] > 0 else -1
        strength = np.max(local * polarity, axis=1)
        support = float(np.mean(strength >= max(2., magnitude[index] * .30)))
        peaks.append(dict(offset=offset, index=int(index), gradient=float(gradient[index]), support=support))
    return peaks


def _interval_median(profile, left, right):
    keep = (profile["offsets"] >= left) & (profile["offsets"] <= right)
    return float(np.median(profile["profile"][keep])) if np.any(keep) else None


def _edge_pairs(profile, peaks, width, shift):
    candidates = []
    diagnostics = dict(edge_peak_count=len(peaks), opposite_pair_count=0,
                       width_compatible_pair_count=0, corridor_pair_count=0,
                       unsupported_edge_pairs=0, weak_contrast_pairs=0, saturated_pairs=0)
    for i, first in enumerate(peaks):
        for second in peaks[i + 1:]:
            if first["gradient"] * second["gradient"] >= 0:
                continue
            diagnostics["opposite_pair_count"] += 1
            measured = second["offset"] - first["offset"]
            if measured < .55 * width or measured > 1.65 * width:
                continue
            diagnostics["width_compatible_pair_count"] += 1
            center = (first["offset"] + second["offset"]) / 2
            if abs(center) > shift:
                continue
            diagnostics["corridor_pair_count"] += 1
            support = min(first["support"], second["support"])
            balance = min(abs(first["gradient"]), abs(second["gradient"])) / max(abs(first["gradient"]), abs(second["gradient"]))
            if support < .70 or balance < .20:
                diagnostics["unsupported_edge_pairs"] += 1
                continue
            low, high = first["offset"], second["offset"]
            inside = _interval_median(profile, low + measured * .22, high - measured * .22)
            flank = max(1., measured * .10)
            outside_a = _interval_median(profile, low - 1.75 * flank, low - .25 * flank)
            outside_b = _interval_median(profile, high + .25 * flank, high + 1.75 * flank)
            if inside is None or outside_a is None or outside_b is None:
                continue
            polarity = 1 if first["gradient"] > 0 else -1
            contrast = min(polarity * (inside - outside_a), polarity * (inside - outside_b))
            if contrast < max(6., profile["noise"] * 3.):
                diagnostics["weak_contrast_pairs"] += 1
                continue
            inside_columns = (profile["offsets"] >= low) & (profile["offsets"] <= high)
            saturation = float(np.mean(profile["clipped"][:, inside_columns]))
            if saturation > .25:
                diagnostics["saturated_pairs"] += 1
                continue
            width_consistency = math.exp(-.5 * (math.log(measured / width) / .42)**2)
            confidence = (.45 * min(1., contrast / 40.) + .25 * support +
                          .15 * balance + .15 * width_consistency)
            candidates.append(dict(offset_a=low, offset_b=high, center_offset=center,
                                   width_px=measured, score=float(confidence),
                                   polarity="bright_body" if polarity > 0 else "dark_body",
                                   contrast_gray=float(contrast), edge_support_fraction=support,
                                   gradient_balance=float(balance), saturation_fraction=saturation))
    return sorted(candidates, key=lambda candidate: candidate["score"], reverse=True), diagnostics


def refine_centerline(image_bgr, seed_points_xy, *, expected_width_px=None, max_shift_px=None):
    """Refine each supplied sample independently; return JSON-safe candidates.

    ``seed_points_xy`` is an ordered, image-coordinate polyline. Width and shift
    may be positive scalars or one value per input sample. Width is a SEMANTIC
    body-scale prior, not a physical diameter. Measured width may vary within
    0.55--1.65 times that prior; exceeding this range is a rejection, not a clamp.
    Default search displacement is 0.6 times the local expected width. A shift
    exceeding half the measured width is flagged as possible identity change.

    At least two distinct seeds are needed to define a local normal. The routine
    emits exactly one record per input point and never fills unsupported gaps.
    ``valid`` denotes a two-edge photometric candidate, never rope identity.
    """
    try:
        seeds = np.asarray(seed_points_xy, float)
    except (ValueError, TypeError):
        return _result([], "seed_points_must_be_finite_Nx2")
    if seeds.ndim != 2 or seeds.shape[1:] != (2,) or not np.isfinite(seeds).all():
        return _result([], "seed_points_must_be_finite_Nx2")
    records = [_record(point, "not_evaluated") for point in seeds]
    if not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or min(image_bgr.shape[:2]) < 3:
        for record in records:
            record["reason"] = "image_must_be_uint8_BGR"
        return _result(records, "image_must_be_uint8_BGR")
    if expected_width_px is None:
        for record in records:
            record["reason"] = "body_width_prior_required_to_disambiguate_internal_grooves"
        return _result(records, "body_width_prior_required")
    widths = _per_point(expected_width_px, len(seeds))
    shifts = .6 * widths if widths is not None and max_shift_px is None else _per_point(max_shift_px, len(seeds))
    if widths is None or shifts is None or np.any(widths < 4.) or np.any(widths > 200.) or np.any(shifts > max(image_bgr.shape[:2])):
        for record in records:
            record["reason"] = "invalid_width_or_shift_prior"
        return _result(records, "width_must_be_4_to_200px_and_shift_positive")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    clipped_image = cv2.inRange(image_bgr, (250, 250, 250), (255, 255, 255)).astype(np.float32) / 255.
    for index, (point, width, shift, record) in enumerate(zip(seeds, widths, shifts, records)):
        tangent = _tangent(seeds, index)
        if tangent is None:
            record["reason"] = "local_tangent_unavailable"
            continue
        profile = _profiles(image_bgr, gray, clipped_image, point, tangent, width, shift)
        if profile is None:
            record["reason"] = "insufficient_image_support_for_two_sided_profile"
            continue
        peaks = _peaks(profile)
        candidates, diagnostic = _edge_pairs(profile, peaks, width, shift)
        record["diagnostics"] = dict(diagnostic, expected_width_px=float(width), max_shift_px=float(shift),
                                      tangent_xy=tangent.tolist(), normal_xy=profile["normal"].tolist(),
                                      candidate_pairs=candidates[:8],
                                      photometric_noise_proxy_gray=profile["noise"])
        if not candidates:
            if diagnostic["saturated_pairs"]:
                record["reason"] = "overexposed_body_edges_not_precise"
            elif len(peaks) < 2 or not diagnostic["opposite_pair_count"]:
                record["reason"] = "insufficient_two_sided_edge_evidence"
            elif not diagnostic["width_compatible_pair_count"]:
                record["reason"] = "no_body_width_pair_possible_internal_groove"
            elif not diagnostic["corridor_pair_count"]:
                record["reason"] = "no_body_pair_within_seed_corridor"
            else:
                record["reason"] = "paired_edge_evidence_inconsistent_or_weak"
            continue
        best = candidates[0]
        alternatives = [candidate for candidate in candidates[1:]
                        if (abs(candidate["center_offset"] - best["center_offset"]) > max(1., width * .12) or
                            abs(candidate["width_px"] - best["width_px"]) > width * .20)]
        if any(candidate["score"] >= best["score"] * .85 for candidate in alternatives):
            record["reason"] = "multiple_comparable_body_pairs_identity_ambiguous"
            continue
        if abs(best["center_offset"]) > best["width_px"] * .5:
            record["reason"] = "shift_may_change_rope_identity"
            continue
        normal = profile["normal"]
        record.update(valid=True, reason="paired_body_edges_candidate_identity_unconfirmed",
                      refined_xy=(point + best["center_offset"] * normal).tolist(),
                      edge_a_xy=(point + best["offset_a"] * normal).tolist(),
                      edge_b_xy=(point + best["offset_b"] * normal).tolist(),
                      width_px=float(best["width_px"]), confidence=float(best["score"]))
    return _result(records)
