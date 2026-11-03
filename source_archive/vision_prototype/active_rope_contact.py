"""Active-rope centreline refinement and multi-evidence contact candidates.

SAM/semantic masks provide a coarse region. Precision geometry is measured on
source pixels. The result is diagnostic only and cannot authorize actuation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from surface_centerlines import CenterlineFamily


@dataclass(frozen=True)
class ActiveRopeGeometry:
    valid: bool
    reason: str
    point_xy: tuple[float, float] = (math.nan, math.nan)
    direction_xy: tuple[float, float] = (math.nan, math.nan)
    segment_start_xy: tuple[float, float] = (math.nan, math.nan)
    segment_end_xy: tuple[float, float] = (math.nan, math.nan)
    angle_deg: float = math.nan
    width_px: float = math.nan
    fit_residual_px: float = math.nan
    refined_points_xy: np.ndarray | None = None
    confidence: float = 0.0


@dataclass(frozen=True)
class ContactCandidate:
    valid: bool
    method: str
    point_xy: tuple[float, float] = (math.nan, math.nan)
    confidence: float = 0.0
    reason: str = "unavailable"


@dataclass(frozen=True)
class ContactEstimate:
    valid: bool
    reason: str
    point_xy: tuple[float, float] = (math.nan, math.nan)
    confidence: float = 0.0
    spread_roots: float = math.nan
    candidates: tuple[ContactCandidate, ...] = ()


@dataclass(frozen=True)
class PhysicalPhaseEstimate:
    valid: bool
    reason: str
    selected_phase: str = "unresolved"
    bright_distance_roots: float = math.nan
    dark_distance_roots: float = math.nan
    confidence: float = 0.0


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(mask, bool)
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == index


def refine_active_rope_geometry(frame_bgr: np.ndarray,
                                coarse_mask: np.ndarray
                                ) -> ActiveRopeGeometry:
    """Fit and photometrically refine a coarse SAM/semantic rope mask."""
    if coarse_mask.shape != frame_bgr.shape[:2]:
        coarse_mask = cv2.resize(
            coarse_mask.astype(np.uint8),
            (frame_bgr.shape[1], frame_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST) > 0
    component = _largest_component(coarse_mask)
    ys, xs = np.nonzero(component)
    if xs.size < 80:
        return ActiveRopeGeometry(False, "incoming_mask_too_small")
    points = np.column_stack((xs, ys)).astype(np.float32)
    vx, vy, cx, cy = cv2.fitLine(
        points[::max(1, len(points) // 8000)], cv2.DIST_HUBER,
        0, .01, .01).flatten()
    if vx < 0:
        vx, vy = -vx, -vy
    direction = np.asarray([float(vx), float(vy)], np.float64)
    direction /= max(np.linalg.norm(direction), 1e-9)
    normal = np.asarray([-direction[1], direction[0]])
    center = np.asarray([float(cx), float(cy)])
    relative = points.astype(np.float64) - center
    longitudinal = relative @ direction
    perpendicular = relative @ normal
    low, high = np.percentile(longitudinal, [2, 98])
    width = float(np.percentile(perpendicular, 95) -
                  np.percentile(perpendicular, 5))
    if high - low < 120 or width < 3 or width > 180:
        return ActiveRopeGeometry(False, "incoming_mask_not_long_thin")
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    sample_t = np.linspace(low + .10 * (high - low),
                           high - .08 * (high - low), 32)
    radius = int(np.clip(round(2.3 * width), 18, 100))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    refined = []
    widths = []
    contrasts = []
    height, image_width = gray.shape
    for value in sample_t:
        base = center + value * direction
        map_x = (base[0] + offsets * normal[0]).reshape(1, -1).astype(np.float32)
        map_y = (base[1] + offsets * normal[1]).reshape(1, -1).astype(np.float32)
        if (map_x.min() < 1 or map_x.max() >= image_width - 1 or
                map_y.min() < 1 or map_y.max() >= height - 1):
            continue
        profile = cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR)[0].astype(np.float32)
        sat = cv2.remap(saturation, map_x, map_y,
                        cv2.INTER_LINEAR)[0].astype(np.float32)
        side = np.abs(offsets) >= max(width, .55 * radius)
        baseline = float(np.median(profile[side]))
        sat_baseline = float(np.median(sat[side]))
        likelihood = np.maximum(profile - baseline, 0.0) + .15 * np.maximum(
            sat_baseline - sat, 0.0)
        likelihood *= np.exp(-.5 * (offsets / max(.72 * width, 3.0)) ** 2)
        peak = float(np.max(likelihood))
        if peak < 3.0:
            continue
        support = likelihood >= .34 * peak
        if int(np.count_nonzero(support)) < 2:
            continue
        weights = likelihood[support]
        offset = float(np.sum(offsets[support] * weights) /
                       max(np.sum(weights), 1e-9))
        refined.append(base + offset * normal)
        widths.append(float(np.ptp(offsets[support])))
        contrasts.append(peak)
    if len(refined) < 8:
        return ActiveRopeGeometry(False, "source_pixel_refinement_failed")
    refined_array = np.asarray(refined, np.float32)
    rvx, rvy, rcx, rcy = cv2.fitLine(
        refined_array, cv2.DIST_HUBER, 0, .01, .01).flatten()
    if rvx < 0:
        rvx, rvy = -rvx, -rvy
    refined_direction = np.asarray([float(rvx), float(rvy)])
    refined_direction /= max(np.linalg.norm(refined_direction), 1e-9)
    refined_center = np.asarray([float(rcx), float(rcy)])
    refined_normal = np.asarray([-refined_direction[1], refined_direction[0]])
    refined_longitudinal = (refined_array - refined_center) @ refined_direction
    residual = np.abs((refined_array - refined_center) @ refined_normal)
    fit_residual = float(np.median(residual))
    start = refined_center + low * refined_direction
    end = refined_center + high * refined_direction
    refined_width = float(np.median(widths)) if widths else width
    angle = math.degrees(math.atan2(refined_direction[1],
                                    refined_direction[0]))
    span_score = min(1.0, (high - low) / 700.0)
    residual_score = math.exp(-fit_residual / max(.35 * refined_width, 2.0))
    contrast_score = min(1.0, float(np.median(contrasts)) / 35.0)
    confidence = float(np.clip(span_score * residual_score * contrast_score,
                               0.0, 1.0))
    return ActiveRopeGeometry(
        True, "ok", tuple(refined_center), tuple(refined_direction),
        tuple(start), tuple(end), angle, refined_width, fit_residual,
        refined_array, confidence)


def _line_y(geometry: ActiveRopeGeometry, x: np.ndarray | float):
    vx, vy = geometry.direction_xy
    cx, cy = geometry.point_xy
    return cy + vy / max(vx, 1e-9) * (np.asarray(x) - cx)


def semantic_endpoint_candidate(geometry: ActiveRopeGeometry
                                ) -> ContactCandidate:
    if not geometry.valid:
        return ContactCandidate(False, "semantic_endpoint",
                                reason=geometry.reason)
    return ContactCandidate(True, "semantic_endpoint",
                            geometry.segment_start_xy,
                            .55 * geometry.confidence, "ok")


def silhouette_intersection_candidate(
        geometry: ActiveRopeGeometry, wound_mask: np.ndarray,
        pitch_px: float) -> ContactCandidate:
    if not geometry.valid or not math.isfinite(pitch_px):
        return ContactCandidate(False, "surface_silhouette",
                                reason="geometry_or_pitch_unavailable")
    component = _largest_component(wound_mask)
    ys, xs = np.nonzero(component)
    if xs.size < 1000:
        return ContactCandidate(False, "surface_silhouette",
                                reason="wound_surface_unavailable")
    start_x = geometry.segment_start_xy[0]
    minimum_x = max(int(xs.min()), int(round(start_x - 9 * pitch_px)))
    maximum_x = min(int(xs.max()), int(round(start_x + 9 * pitch_px)))
    candidates = []
    for x in range(minimum_x, maximum_x + 1):
        column = np.flatnonzero(component[:, x])
        if column.size:
            top = float(column.min())
            line = float(_line_y(geometry, x))
            candidates.append((abs(line - top), float(x), top, line))
    if not candidates:
        return ContactCandidate(False, "surface_silhouette",
                                reason="no_overlap_with_surface")
    distance, x, top, line = min(candidates)
    if distance > 1.8 * pitch_px:
        return ContactCandidate(False, "surface_silhouette", (x, line), 0.0,
                                "line_does_not_meet_surface_envelope")
    confidence = geometry.confidence * math.exp(-distance / pitch_px)
    return ContactCandidate(True, "surface_silhouette", (x, .5 * (top + line)),
                            float(np.clip(confidence, 0.0, 1.0)), "ok")


def photometric_contact_candidate(
        frame_bgr: np.ndarray, geometry: ActiveRopeGeometry,
        pitch_px: float) -> ContactCandidate:
    """Trace the free rope backwards until source-pixel evidence disappears.

    SAM often stops far before physical contact. Its fitted direction remains
    useful, so this branch samples normal cross-sections up to 60 rope rows
    before the mask endpoint and finds the left boundary of the bright,
    low-saturation ridge connected to the known free span.
    """
    if not geometry.valid or not math.isfinite(pitch_px):
        return ContactCandidate(False, "photometric_backtrace",
                                reason="geometry_or_pitch_unavailable")
    direction = np.asarray(geometry.direction_xy, np.float64)
    normal = np.asarray([-direction[1], direction[0]])
    center = np.asarray(geometry.point_xy, np.float64)
    start = np.asarray(geometry.segment_start_xy, np.float64)
    end = np.asarray(geometry.segment_end_xy, np.float64)
    t_start = float((start - center) @ direction)
    t_end = float((end - center) @ direction)
    step = float(np.clip(.24 * pitch_px, 3.0, 7.0))
    extension = max(24.0 * pitch_px, min(70.0 * pitch_px, 1200.0))
    samples_t = np.arange(t_start - extension, t_end, step)
    width = max(geometry.width_px, .45 * pitch_px)
    radius = int(np.clip(round(2.1 * width), 18, 90))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    values = np.full(len(samples_t), np.nan, np.float32)
    height, image_width = gray.shape
    for index, value in enumerate(samples_t):
        base = center + value * direction
        map_x = (base[0] + offsets * normal[0]).reshape(1, -1).astype(np.float32)
        map_y = (base[1] + offsets * normal[1]).reshape(1, -1).astype(np.float32)
        if (map_x.min() < 1 or map_x.max() >= image_width - 1 or
                map_y.min() < 1 or map_y.max() >= height - 1):
            continue
        profile = cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR)[0].astype(np.float32)
        sat = cv2.remap(saturation, map_x, map_y,
                        cv2.INTER_LINEAR)[0].astype(np.float32)
        core = np.abs(offsets) <= max(3.0, .42 * width)
        side = np.abs(offsets) >= max(.85 * width, .52 * radius)
        brightness = float(np.percentile(profile[core], 78) -
                           np.median(profile[side]))
        neutrality = float(np.median(sat[side]) -
                           np.percentile(sat[core], 35))
        # A coherent ridge is also locally symmetric. This suppresses many
        # one-sided highlights from wound-rope texture.
        left = profile[(offsets >= -.95 * width) &
                       (offsets <= -.40 * width)]
        right = profile[(offsets >= .40 * width) &
                        (offsets <= .95 * width)]
        symmetry_penalty = (abs(float(np.median(left)) -
                                float(np.median(right))) if len(left) and
                            len(right) else 0.0)
        values[index] = (brightness + .18 * max(neutrality, 0.0) -
                         .12 * symmetry_penalty)
    finite = np.isfinite(values)
    known = finite & (samples_t >= t_start + .15 * (t_end - t_start)) & \
        (samples_t <= t_start + .85 * (t_end - t_start))
    if int(np.count_nonzero(known)) < 10:
        return ContactCandidate(False, "photometric_backtrace",
                                reason="free_span_reference_unavailable")
    reference = float(np.median(values[known]))
    noise = float(np.median(np.abs(values[known] - reference)) * 1.4826)
    threshold = max(2.0, .30 * reference, reference - 3.0 * noise)
    support = finite & (values >= threshold)
    # Close gaps shorter than roughly one rope row, caused by strand twist.
    kernel = max(3, int(round(pitch_px / step)) | 1)
    closed = cv2.morphologyEx(support.astype(np.uint8).reshape(1, -1),
                              cv2.MORPH_CLOSE,
                              np.ones((1, kernel), np.uint8))[0] > 0
    anchor_candidates = np.flatnonzero(known & closed)
    if not len(anchor_candidates):
        return ContactCandidate(False, "photometric_backtrace",
                                reason="known_free_span_not_supported")
    anchor = int(np.median(anchor_candidates))
    groups = np.split(np.flatnonzero(closed),
                      np.flatnonzero(np.diff(np.flatnonzero(closed)) > 1) + 1)
    connected = next((group for group in groups if anchor in group), None)
    if connected is None or len(connected) < max(5, kernel):
        return ContactCandidate(False, "photometric_backtrace",
                                reason="no_connected_rope_ridge")
    contact_index = int(connected[0])
    contact_t = float(samples_t[contact_index])
    point = center + contact_t * direction
    margin = max(0.0, reference - threshold)
    confidence = (geometry.confidence *
                  min(1.0, len(connected) * step / max(12 * pitch_px, 1.0)) *
                  min(1.0, margin / max(noise + 2.0, 1e-6)))
    return ContactCandidate(True, "photometric_backtrace", tuple(point),
                            float(np.clip(confidence, 0.0, 1.0)), "ok")


def family_continuation_candidate(
        geometry: ActiveRopeGeometry, family: CenterlineFamily
        ) -> ContactCandidate:
    if not geometry.valid or not family.valid:
        return ContactCandidate(False, "centerline_continuation",
                                reason="geometry_or_family_unavailable")
    start = np.asarray(geometry.segment_start_xy)
    vx, vy = geometry.direction_xy
    normal = np.asarray([-vy, vx])
    best = None
    for line in family.centerlines:
        points = line.observed_points_xy
        if len(points) < 3:
            continue
        relative = points - np.asarray(geometry.point_xy)
        perpendicular = np.abs(relative @ normal)
        longitudinal = relative @ np.asarray([vx, vy])
        start_longitudinal = ((start - np.asarray(geometry.point_xy)) @
                              np.asarray([vx, vy]))
        cost = (perpendicular / max(family.pitch_px, 1e-6) +
                .025 * np.abs(longitudinal - start_longitudinal) /
                max(family.pitch_px, 1e-6))
        index = int(np.argmin(cost))
        item = (float(cost[index]), points[index], float(perpendicular[index]))
        if best is None or item[0] < best[0]:
            best = item
    if best is None or best[2] > 1.25 * family.pitch_px:
        return ContactCandidate(False, "centerline_continuation",
                                reason="no_nearby_observed_centerline")
    confidence = (geometry.confidence * family.family_confidence *
                  math.exp(-best[2] / family.pitch_px))
    point = tuple(float(value) for value in best[1])
    return ContactCandidate(True, "centerline_continuation", point,
                            float(np.clip(confidence, 0.0, 1.0)), "ok")


def fuse_contact_candidates(candidates: list[ContactCandidate],
                            pitch_px: float) -> ContactEstimate:
    valid = [item for item in candidates if item.valid and item.confidence > 0]
    if len(valid) < 2 or not math.isfinite(pitch_px):
        return ContactEstimate(False, "fewer_than_two_contact_methods",
                               candidates=tuple(candidates))
    points = np.asarray([item.point_xy for item in valid], np.float64)
    weights = np.asarray([max(.05, item.confidence) for item in valid])
    fused = np.average(points, axis=0, weights=weights)
    distances = np.linalg.norm(points - fused, axis=1) / pitch_px
    spread = float(np.max(distances))
    if spread > 2.5:
        return ContactEstimate(False, "contact_methods_conflict",
                               tuple(fused), 0.0, spread, tuple(candidates))
    confidence = float(np.clip(np.mean(weights) * math.exp(-spread / 1.4),
                               0.0, 1.0))
    return ContactEstimate(True, "multiple_contact_methods_agree",
                           tuple(fused), confidence, spread,
                           tuple(candidates))


def _phase_distance(geometry: ActiveRopeGeometry, family: CenterlineFamily,
                    point_xy: tuple[float, float]) -> float:
    if not geometry.valid or not family.valid:
        return math.nan
    x, _y = point_xy
    active_y = float(_line_y(geometry, x))
    values = []
    for line in family.centerlines:
        points = line.points_xy
        if points[0, 0] <= x <= points[-1, 0]:
            values.append(abs(float(np.interp(x, points[:, 0], points[:, 1])) -
                              active_y) / family.pitch_px)
    return min(values) if values else math.nan


def resolve_physical_phase(
        geometry: ActiveRopeGeometry, bright: CenterlineFamily,
        dark: CenterlineFamily, contact_xy: tuple[float, float]
        ) -> PhysicalPhaseEstimate:
    bright_distance = _phase_distance(geometry, bright, contact_xy)
    dark_distance = _phase_distance(geometry, dark, contact_xy)
    if not (math.isfinite(bright_distance) and math.isfinite(dark_distance)):
        return PhysicalPhaseEstimate(False, "both_phases_not_available",
                                     bright_distance_roots=bright_distance,
                                     dark_distance_roots=dark_distance)
    difference = abs(bright_distance - dark_distance)
    selected = "bright" if bright_distance < dark_distance else "dark"
    if difference < .16:
        return PhysicalPhaseEstimate(False, "phase_still_ambiguous", "unresolved",
                                     bright_distance, dark_distance,
                                     float(np.clip(difference / .16, 0.0, 1.0)))
    confidence = float(np.clip((difference / .50) *
                               math.exp(-min(bright_distance, dark_distance)),
                               0.0, 1.0))
    return PhysicalPhaseEstimate(True, "active_rope_continuation_selects_phase",
                                 selected, bright_distance, dark_distance,
                                 confidence)
