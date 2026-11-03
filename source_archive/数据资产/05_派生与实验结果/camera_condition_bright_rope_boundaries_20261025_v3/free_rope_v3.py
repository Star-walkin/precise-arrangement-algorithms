"""Offline free-rope occlusion evidence for the bright wound-face ROI.

This module calls the historical ``rope_detector.detect_rope`` on original BGR
pixels.  A camera semantic map supplies an automatic *proposal*, never a truth
label or clicked seed.  Only original-pixel stripe profile samples are called
supported.  The older semantic mask remains UNKNOWN where pixels do not support
the historical profile; zero output does not establish that rope is absent.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import cv2
import numpy as np


_ROOT = next(p for p in Path(__file__).resolve().parents
             if (p / "vision_geometry_lab" / "rope_detector.py").is_file())
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "vision_prototype"))
sys.path.insert(0, str(_ROOT / "数据资产" / "05_派生与实验结果" /
                       "camera_condition_bright_rope_boundaries_20261023_v2"))
from vision_geometry_lab import rope_detector  # noqa: E402
from semantic_locator import semantic_locator_for_camera  # noqa: E402
from free_rope_occlusion import estimate_free_rope_occlusion  # noqa: E402


def _proposal(image: np.ndarray, roi: np.ndarray, role: str,
              pitch: float) -> tuple[np.ndarray, tuple | None, dict]:
    """Find a long oblique automatic semantic span; preserve component evidence."""
    locator = semantic_locator_for_camera(role)
    regions = locator.predict(image)
    prob = regions.probability("incoming_rope_free_span")
    x0, y0, x1, y1 = regions.crop_xyxy
    binary = (prob >= .50).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    selected = np.zeros_like(binary)
    entries = []
    sx, sy = (x1 - x0) / prob.shape[1], (y1 - y0) / prob.shape[0]
    for k in range(1, n):
        area = int(stats[k, cv2.CC_STAT_AREA])
        if area < 30:
            continue
        yy, xx = np.nonzero(labels == k)
        xy = np.column_stack((x0 + (xx + .5) * sx, y0 + (yy + .5) * sy)).astype(np.float32)
        vx, vy, _, _ = (float(v) for v in cv2.fitLine(
            xy, cv2.DIST_HUBER, 0, .01, .01).flatten())
        if vx < 0:
            vx, vy = -vx, -vy
        angle = math.degrees(math.atan2(vy, vx))
        span = float(np.ptp(xy[:, 0]))
        mean_prob = float(np.mean(prob[yy, xx]))
        # This gate only chooses an original-pixel search proposal.  Broad
        # components remain possible here; detect_rope must verify the stripe.
        chosen = (-43.0 <= angle <= -1.0 and
                  span >= max(150.0, 6.0 * pitch) and mean_prob >= .55)
        entries.append({"id": k, "area": area, "angle_deg": round(angle, 2),
                        "x_span_px": round(span, 2),
                        "mean_probability": round(mean_prob, 3), "chosen": chosen})
        if chosen:
            selected[labels == k] = 255
    full = np.zeros(image.shape[:2], np.uint8)
    if np.any(selected):
        full[y0:y1, x0:x1] = cv2.resize(selected, (x1 - x0, y1 - y0),
                                         interpolation=cv2.INTER_NEAREST)
        ys, xs = np.nonzero(full)
        margin = int(round(max(30.0, 2.0 * pitch)))
        box = (max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
               min(image.shape[1] - 1, int(xs.max()) + margin),
               min(image.shape[0] - 1, int(ys.max()) + margin))
    else:
        box = None
    return full, box, {"semantic_components": entries,
                       "proposal_component_ids": [e["id"] for e in entries if e["chosen"]],
                       "semantic_model_path": str(locator.model_path),
                       "semantic_crop_xyxy": [int(v) for v in regions.crop_xyxy]}


def _profile_supported(image: np.ndarray, roi: np.ndarray, proposal: np.ndarray,
                       history: dict, pitch: float) -> tuple[np.ndarray, dict]:
    """Re-sample untouched pixels along the measured line, including near ROI.

    The historical routine selects a coherent observed support run and rejects
    multiple wound-face stripes.  We rasterize only neighbouring valid sample
    intervals.  No gap or geometric extrapolation becomes supported by itself.
    """
    h, w = roi.shape
    result = np.zeros((h, w), np.uint8)
    segment = history.get("observed_segment")
    if not history.get("valid") or segment is None:
        return result, {"status": "history_invalid"}
    endpoints = np.asarray(segment, np.float64)
    vector = endpoints[1] - endpoints[0]
    length = float(np.linalg.norm(vector))
    if length < 1:
        return result, {"status": "observed_segment_too_short"}
    d = vector / length
    c = np.mean(endpoints, axis=0)
    # The source detector's optical width is not a physical rope diameter.
    width = history.get("width_px")
    if width is None:
        return result, {"status": "optical_width_unavailable"}
    width = float(width)
    scale = min(1., 1536. / max(h, w))
    ys, xs = np.nonzero(proposal)
    if len(xs) < 12:
        return result, {"status": "semantic_proposal_missing"}
    # Force this second profile pass to inspect the supplied bright ROI.  If
    # the external free segment is longer, an unrestricted _profiles call
    # selects that external group again and never tests the crossing.
    overlap = (proposal > 0) & (roi > 0)
    ry, rx = np.nonzero(overlap)
    if len(rx) < 12:
        return result, {"status": "semantic_proposal_does_not_touch_roi"}
    # Projection extends only one pitch past semantic support in the ROI.
    # The original-pixel profile test below decides which points survive.
    ps = np.column_stack((rx, ry)).astype(np.float64) @ d
    projection = (float(ps.min() - pitch), float(ps.max() + pitch))
    box = (max(0, int(rx.min() - 2 * pitch)), max(0, int(ry.min() - 2 * pitch)),
           min(w - 1, int(rx.max() + 2 * pitch)),
           min(h - 1, int(ry.max() + 2 * pitch)))
    white = np.min(image, axis=2).astype(np.float32)
    sample_step = max(3., 4. / scale)
    fine = rope_detector._profiles(
        white, c, d, box, None, proposal.astype(np.float32) / 255.,
        search=max(3. / scale, 2.5 * pitch), step=.5, max_width=12. / scale,
        guided=True, sample_step=sample_step, projection_range=projection)
    if fine is None:
        return result, {"status": "original_profile_extension_failed",
                        "projection_range_px": list(projection)}
    points = fine["points"]
    widths = fine["widths"]
    order = np.argsort(points @ d)
    points, widths = points[order], widths[order]
    support_count = 0
    for i in range(len(points) - 1):
        gap = float(np.linalg.norm(points[i + 1] - points[i]))
        if gap > max(18., 2.3 * sample_step):
            continue
        local_width = float(min(widths[i], widths[i + 1], width))
        if local_width < 2.5 or local_width > max(1.4 * pitch, 36.):
            continue
        a = tuple(np.rint(points[i]).astype(int))
        b = tuple(np.rint(points[i + 1]).astype(int))
        cv2.line(result, a, b, 255, max(1, int(round(local_width))), cv2.LINE_8)
        support_count += 1
    result[roi == 0] = 0
    # Allow only a small tolerance around the actual semantic proposal.  This
    # is still an original-pixel verified stripe, never an untested long line.
    tolerance = max(3, int(round(.2 * pitch)))
    near = cv2.dilate(proposal, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1)))
    result[near == 0] = 0
    observed_projection = sorted(float(v) for v in endpoints @ d)
    sample_projection = points @ d
    outside_observed = int(np.count_nonzero(
        (sample_projection < observed_projection[0]) |
        (sample_projection > observed_projection[1])))
    return result, {"status": "original_profile_verified", "profile_samples": int(len(points)),
                    "drawn_sample_intervals": support_count,
                    "profile_support_length_px": round(float(fine["span"]), 2),
                    "samples_beyond_original_observed_segment": outside_observed,
                    "projection_range_px": list(projection),
                    "source": "historical_rope_detector._profiles_on_original_pixels"}


def _oriented_ribbon_supported(image: np.ndarray, roi: np.ndarray,
                               proposal: np.ndarray, history: dict,
                               pitch: float) -> tuple[np.ndarray, dict]:
    """Check the crossing with dark/bright double flanks in original pixels.

    The historical measurement supplies slope and optical width.  Longitudinal
    smoothing suppresses nearly horizontal wound-row highlights, which drift
    across the rectified oblique rope coordinate.  A short semantic fragment
    alone cannot pass the length, orientation and local contrast tests.
    """
    h, w = roi.shape
    out = np.zeros((h, w), np.uint8)
    if not history.get("valid") or history.get("width_px") is None:
        return out, {"status": "history_line_or_width_unavailable"}
    segment = np.asarray(history["observed_segment"], np.float64)
    dx = segment[1, 0] - segment[0, 0]
    if abs(dx) < 10:
        return out, {"status": "history_line_nearly_vertical"}
    slope = float((segment[1, 1] - segment[0, 1]) / dx)
    if not -.84 <= slope <= -.018:
        return out, {"status": "history_direction_outside_oblique_gate",
                     "history_slope": slope}
    width = float(history["width_px"])
    if not 4. <= width <= max(1.5 * pitch, 38.):
        return out, {"status": "history_optical_width_outside_gate",
                     "optical_width_px": width}
    # Select x only from the existing bright ROI.  Semantic support may be
    # shifted toward the shadow, so it is tested with a finite 1.2-pitch reach.
    ry, rx = np.nonzero(roi)
    if len(rx) < 100:
        return out, {"status": "roi_too_small"}
    xlo, xhi = int(rx.min()), int(rx.max())
    if xhi - xlo < max(100., 4. * pitch):
        return out, {"status": "roi_too_narrow"}
    step_x = 3
    xx = np.arange(xlo, xhi + 1, step_x, dtype=np.float32)
    center_x = float(np.mean(xx))
    anchor = segment[0]
    base_y = anchor[1] + slope * (xx - anchor[0])
    search = int(round(min(115., max(65., 4.0 * pitch))))
    q = np.arange(-search, search + 1, dtype=np.float32)
    map_x = np.broadcast_to(xx[None, :], (len(q), len(xx))).copy()
    map_y = (base_y[None, :] + q[:, None]).astype(np.float32)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rect = cv2.remap(grey, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
    # Averaging along the measured oblique direction attenuates the rows of
    # wound rope because their slope differs from the free span's slope.
    smooth = cv2.GaussianBlur(rect, (0, 0),
                              sigmaX=max(4., .9 * pitch / step_x),
                              sigmaY=1.2)
    sem = cv2.remap(proposal, map_x, map_y, cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT)
    sem_reach = max(4, int(round(1.2 * pitch)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (1, 2 * sem_reach + 1))
    near_sem = cv2.dilate(sem, kernel) > 0
    roi_here = cv2.remap(roi, map_x, map_y, cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT)
    near_roi = cv2.dilate(roi_here, cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, 2 * max(3, int(round(pitch))) + 1))) > 0
    half = int(round(max(4., min(width * .62, .9 * pitch))))
    score = np.zeros_like(smooth)
    polarity = np.zeros_like(smooth, np.uint8)
    bright = np.minimum(smooth[half:-half] - smooth[:-2 * half],
                        smooth[half:-half] - smooth[2 * half:])
    dark = np.minimum(smooth[:-2 * half] - smooth[half:-half],
                      smooth[2 * half:] - smooth[half:-half])
    score[half:-half] = np.maximum(bright, dark)
    polarity[half:-half] = np.where(bright >= dark, 1, 2).astype(np.uint8)
    score[~near_sem | ~near_roi] = 0
    # Candidate lines can bend slightly relative to the external free segment,
    # but may not rotate into the nearly horizontal wound-row direction.
    deltas = np.linspace(-.07, .07, 15)
    contrast_threshold = 9.0
    best = None
    competitors = []
    col = np.arange(len(xx))
    for delta in deltas:
        shift = np.rint(delta * (xx - center_x)).astype(np.int32)
        for q0 in range(-search + half, search - half + 1, 2):
            qi = q0 + shift + search
            good_index = (qi >= half) & (qi < len(q) - half)
            values = np.zeros(len(xx), np.float32)
            values[good_index] = score[qi[good_index], col[good_index]]
            passed = values >= contrast_threshold
            count = int(np.count_nonzero(passed))
            if count < max(20, int(.23 * len(xx))):
                continue
            # Reward repeated original-image edge pairs, not a single peak.
            quality = float(np.sum(np.minimum(values[passed], 45.)))
            quality *= math.sqrt(count / len(xx))
            item = (quality, q0, float(delta), count, values)
            if best is None or quality > best[0]:
                if best is not None:
                    competitors.append(best[:4])
                best = item
            else:
                competitors.append(item[:4])
    if best is None:
        return out, {"status": "no_oblique_double_flank_support",
                     "history_slope": slope, "optical_width_px": width,
                     "contrast_threshold_gray": contrast_threshold,
                     "maximum_double_flank_contrast_gray": round(float(np.max(score)), 2),
                     "score_pixels_over_threshold": int(np.count_nonzero(score >= contrast_threshold)),
                     "semantic_near_pixels": int(np.count_nonzero(near_sem & near_roi)),
                     "semantic_rect_pixels": int(np.count_nonzero(sem)),
                     "roi_rect_pixels": int(np.count_nonzero(roi_here)),
                     "rectified_y_range": [float(np.min(map_y)), float(np.max(map_y))]}
    quality, q0, delta, count, _ = best
    # A second parallel ridge one pitch away can be a wound-rope row.  If it
    # is comparable, the geometry does not uniquely identify the free rope.
    distinct = [c for c in competitors if abs(c[1] - q0) >= .8 * pitch]
    second_ratio = max((c[0] / quality for c in distinct), default=0.)
    if second_ratio > .88:
        return out, {"status": "ambiguous_competing_ribbons",
                     "candidate_quality": round(quality, 2),
                     "competitor_ratio": round(float(second_ratio), 3)}
    line_q = q0 + delta * (xx - center_x)
    # Refine each position by at most one quarter pitch.  Large missing gaps
    # stay UNKNOWN rather than being connected by a smooth line.
    radius = max(2, int(round(.25 * pitch)))
    chosen_q = np.full(len(xx), np.nan, np.float32)
    chosen_score = np.zeros(len(xx), np.float32)
    chosen_polarity = np.zeros(len(xx), np.uint8)
    for j in range(len(xx)):
        middle = int(round(line_q[j])) + search
        lo, hi = max(half, middle - radius), min(len(q) - half - 1, middle + radius)
        if hi < lo:
            continue
        k = lo + int(np.argmax(score[lo:hi + 1, j]))
        if score[k, j] >= contrast_threshold:
            chosen_q[j] = q[k]
            chosen_score[j] = score[k, j]
            chosen_polarity[j] = polarity[k, j]
    valid = np.isfinite(chosen_q)
    if int(np.count_nonzero(valid)) < max(25, int(.30 * len(xx))):
        return out, {"status": "insufficient_local_profile_coverage",
                     "candidate_quality": round(quality, 2)}
    # Keep only coherent runs at least four rope pitches long; no infinite
    # extension from the historical line or semantic centre curve.
    groups = np.split(np.flatnonzero(valid),
                      np.flatnonzero(np.diff(np.flatnonzero(valid)) > 1) + 1)
    groups = [g for g in groups if len(g) * step_x >= max(4. * pitch, 90.)]
    if not groups:
        return out, {"status": "no_long_original_pixel_run",
                     "candidate_quality": round(quality, 2),
                     "candidate_valid_samples": int(np.count_nonzero(valid))}
    drawn = 0
    for group in groups:
        for a, b in zip(group[:-1], group[1:]):
            p0 = (int(round(xx[a])), int(round(base_y[a] + chosen_q[a])))
            p1 = (int(round(xx[b])), int(round(base_y[b] + chosen_q[b])))
            if abs(p1[1] - p0[1]) > 5:
                continue
            cv2.line(out, p0, p1, 255, max(2, int(round(.75 * width))),
                     cv2.LINE_8)
            drawn += 1
    out[roi == 0] = 0
    if drawn == 0 or np.count_nonzero(out) < 50:
        return np.zeros_like(out), {"status": "ribbon_no_roi_intersection",
                                    "drawn_intervals": drawn}
    return out, {"status": "original_pixel_double_flank_supported",
                 "source": "historical_detect_rope_direction_width_plus_rectified_original_bgr",
                 "history_slope": slope, "local_slope_delta": float(delta),
                 "optical_width_px": width, "sample_step_x_px": step_x,
                 "contrast_threshold_gray": contrast_threshold,
                 "supporting_samples": int(sum(len(g) for g in groups)),
                 "drawn_intervals": drawn,
                 "candidate_quality": round(quality, 2),
                 "competitor_ratio": round(float(second_ratio), 3),
                 "polarity_counts": {"bright": int(np.count_nonzero(chosen_polarity == 1)),
                                     "dark": int(np.count_nonzero(chosen_polarity == 2))}}


def estimate_occlusion(image_bgr: np.ndarray, reference_roi: np.ndarray,
                       camera_role: str, pitch_px: float
                       ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return supported/unknown 0-255 masks inside the supplied bright ROI.

    ``supported`` means that detect_rope was valid and local original-image
    stripe profiles accepted the pixels. ``unknown`` is a conservative semantic
    exclusion, not a confirmed rope. A zero in both masks is not rope absence.
    """
    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must be HxWx3 BGR")
    if reference_roi.shape != image_bgr.shape[:2]:
        raise ValueError("reference_roi dimensions must match image")
    role = camera_role.replace("_camera", "")
    if role not in {"left", "right"}:
        raise ValueError("camera_role must be left or right")
    pitch = float(pitch_px)
    if not math.isfinite(pitch) or not 6. <= pitch <= 100.:
        raise ValueError("pitch_px must be a plausible finite pixel pitch")
    roi = (reference_roi > 0).astype(np.uint8) * 255
    canonical = cv2.flip(image_bgr, 1) if role == "left" else image_bgr
    canonical_roi = cv2.flip(roi, 1) if role == "left" else roi
    proposal, box, sem_diag = _proposal(canonical, canonical_roi, role, pitch)
    if box is None:
        history = {"valid": False, "reason": "no_automatic_semantic_span",
                   "diagnostics": {}}
        history_called = False
    else:
        history = rope_detector.detect_rope(canonical, roi=box,
                                            rope_mask=proposal)
        history_called = True
    support, profile_diag = _profile_supported(canonical, canonical_roi,
                                                proposal, history, pitch)
    ribbon_diag = {"status": "not_attempted"}
    if np.count_nonzero(support) == 0 and history.get("valid"):
        support, ribbon_diag = _oriented_ribbon_supported(
            canonical, canonical_roi, proposal, history, pitch)
    old_mask, old_diag = estimate_free_rope_occlusion(image_bgr, roi, role, pitch)
    old_canonical = cv2.flip(old_mask, 1) if role == "left" else old_mask
    unknown = np.where((old_canonical > 0) & (support == 0), 255, 0).astype(np.uint8)
    support[canonical_roi == 0] = 0
    unknown[canonical_roi == 0] = 0
    if role == "left":
        support, unknown = cv2.flip(support, 1), cv2.flip(unknown, 1)
    diag = {"source": "historical_detect_rope_plus_original_profile_v3",
            "camera_role": role, "mask_value_convention": "uint8_0_or_255",
            "history_module": "vision_geometry_lab.rope_detector.detect_rope",
            "history_module_called": history_called,
            "history_valid": bool(history.get("valid")),
            "history_reason": history.get("reason"),
            "history_width_px": history.get("width_px"),
            "history_width_interval_px": history.get("width_interval_px"),
            "history_angle_deg": history.get("angle_deg"),
            "history_observed_segment_canonical_xy": history.get("observed_segment"),
            "history_observed_support": history.get("observed_support"),
            "history_confidence": history.get("confidence"),
            "history_diagnostics": history.get("diagnostics", {}),
            "semantic_proposal": sem_diag, "profile_extension": profile_diag,
            "oriented_ribbon": ribbon_diag,
            "v2_fallback_mask_kind": old_diag.get("mask_kind"),
            "v2_fallback_status": old_diag.get("status"),
            "v2_fallback_pixels": int(np.count_nonzero(old_mask)),
            "supported_pixels_in_roi": int(np.count_nonzero(support)),
            "unknown_pixels_in_roi": int(np.count_nonzero(unknown)),
            "supported_meaning": "original-pixel optical stripe evidence within ROI",
            "unknown_meaning": "v2 semantic candidate without sufficient v3 pixel verification",
            "complete_free_rope_claim": False}
    if not np.count_nonzero(support):
        diag["status"] = "v2_unknown_only" if np.count_nonzero(unknown) else "not_detected"
    else:
        diag["status"] = "partial_original_profile_support"
    return support.astype(np.uint8), unknown.astype(np.uint8), diag
