"""Visible half-cylinder rope-centerline extraction for offline/shadow use.

The semantic model selects the wound body.  Rope pitch and centre paths are
measured on original camera pixels with an oriented filter bank, column-wise
non-maximum suppression and bidirectional path linking.  Nothing in this
module imports device interfaces or grants actuator permission.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from semantic_locator import SemanticRegions


@dataclass(frozen=True)
class RopeCenterline:
    points_xy: np.ndarray
    observed_points_xy: np.ndarray
    coverage: float
    response: float


@dataclass(frozen=True)
class CenterlineFamily:
    valid: bool
    reason: str
    bbox_xyxy: tuple[int, int, int, int] = (0, 0, 0, 0)
    pitch_px: float = math.nan
    pitch_confidence: float = 0.0
    centerlines: tuple[RopeCenterline, ...] = ()
    regularity_error: float = math.nan
    family_confidence: float = 0.0
    sampled_columns: int = 0


@dataclass(frozen=True)
class FamilyGapEstimate:
    valid: bool
    reason: str
    continuous_count: float = math.nan
    indexed_count: float = math.nan
    local_pitch_px: float = math.nan
    confidence: float = 0.0


@dataclass(frozen=True)
class MultiScalePitchEstimate:
    valid: bool
    reason: str
    pitch_px: float = math.nan
    confidence: float = 0.0
    proposals: tuple[tuple[float, float], ...] = ()
    supporting_proposals: int = 0


@dataclass(frozen=True)
class MultiPhaseCenterlineResult:
    valid: bool
    reason: str
    pitch: MultiScalePitchEstimate
    bright_family: CenterlineFamily
    dark_family: CenterlineFamily
    selected_phase: str = "unresolved"
    selected_family: CenterlineFamily | None = None
    phase_ambiguous: bool = True
    phase_score_ratio: float = math.nan
    median_phase_offset_roots: float = math.nan


@dataclass(frozen=True)
class QualityGatedCenterlineResult:
    result: MultiPhaseCenterlineResult
    used_quality_gate: bool
    reason: str
    ungated_score: float = 0.0
    gated_score: float = 0.0
    observable_fraction: float = math.nan
    largest_bad_run_fraction: float = math.nan
    ungated_result: MultiPhaseCenterlineResult | None = None
    gated_result: MultiPhaseCenterlineResult | None = None


@dataclass(frozen=True)
class QualityGatedSingleFamilyResult:
    family: CenterlineFamily
    used_quality_gate: bool
    reason: str
    ungated_score: float = 0.0
    gated_score: float = 0.0
    observable_fraction: float = math.nan
    largest_bad_run_fraction: float = math.nan


@dataclass(frozen=True)
class CylindricalLattice:
    """Empirical coordinates on the camera-visible half cylinder.

    ``u`` is an orthographic half-cylinder angle in [-1, 1]. ``v`` is an
    axial rope-row coordinate relative to the first visible centreline.  This
    is a view normalization, not a metric 3-D reconstruction.
    """
    valid: bool
    reason: str
    center_x: float = math.nan
    projected_radius_px: float = math.nan
    pitch_px: float = math.nan
    reference_line_xy: np.ndarray | None = None
    normalized_regularity_error: float = math.nan


def build_cylindrical_lattice(family: CenterlineFamily) -> CylindricalLattice:
    if not family.valid or len(family.centerlines) < 4:
        return CylindricalLattice(False, "centerline_family_unavailable")
    x1, _y1, x2, _y2 = family.bbox_xyxy
    radius = .5 * (x2 - x1)
    if radius < 20 or not math.isfinite(family.pitch_px):
        return CylindricalLattice(False, "invalid_projected_cylinder")
    reference = family.centerlines[0].points_xy.copy()
    center_x = .5 * (x1 + x2)
    residuals = []
    # A linked family may legitimately miss a row. Comparing with the tuple
    # index would then label every following row as irregular. The lattice
    # residual is phase-to-nearest-integer, so missing rows remain missing
    # observations instead of corrupting the cylindrical normalization.
    for line in family.centerlines[1:]:
        points = line.points_xy
        if len(points) < 4:
            continue
        reference_y = np.interp(points[:, 0], reference[:, 0], reference[:, 1])
        v = (points[:, 1] - reference_y) / family.pitch_px
        residuals.extend(np.abs(v - np.rint(v)).tolist())
    regularity = float(np.median(residuals)) if residuals else math.nan
    return CylindricalLattice(True, "ok", center_x, radius,
                              family.pitch_px, reference, regularity)


def points_to_cylindrical(lattice: CylindricalLattice,
                          points_xy: np.ndarray) -> np.ndarray:
    """Map source pixels to empirical visible-cylinder (u, v) coordinates."""
    if not lattice.valid or lattice.reference_line_xy is None:
        return np.empty((0, 2), np.float32)
    points = np.asarray(points_xy, np.float64).reshape(-1, 2)
    normalized_x = np.clip((points[:, 0] - lattice.center_x) /
                           lattice.projected_radius_px, -1.0, 1.0)
    u = np.arcsin(normalized_x) / (.5 * np.pi)
    reference = lattice.reference_line_xy
    reference_y = np.interp(points[:, 0], reference[:, 0], reference[:, 1])
    v = (points[:, 1] - reference_y) / lattice.pitch_px
    return np.column_stack((u, v)).astype(np.float32)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == index


def source_wound_mask(regions: SemanticRegions,
                      source_shape: tuple[int, ...]) -> np.ndarray:
    height, width = source_shape[:2]
    x1, y1, x2, y2 = regions.crop_xyxy
    low = _largest_component(regions.mask("wound_rope_surface"))
    full = np.zeros((height, width), np.uint8)
    full[y1:y2, x1:x2] = cv2.resize(
        low.astype(np.uint8), (x2 - x1, y2 - y1),
        interpolation=cv2.INTER_NEAREST)
    return full.astype(bool)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size < 1000:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _column_run(mask_column: np.ndarray) -> tuple[int, int] | None:
    indices = np.flatnonzero(mask_column)
    if indices.size < 4:
        return None
    splits = np.flatnonzero(np.diff(indices) > 1) + 1
    groups = np.split(indices, splits)
    group = max(groups, key=len)
    return int(group[0]), int(group[-1]) + 1


def estimate_family_pitch(gray: np.ndarray, mask: np.ndarray,
                          fallback_pitch_px: float = 18.0
                          ) -> tuple[float, float]:
    """Robust axial period from many original-resolution cylinder columns."""
    bbox = _mask_bbox(mask)
    if bbox is None or not math.isfinite(fallback_pitch_px):
        return math.nan, 0.0
    x1, _y1, x2, _y2 = bbox
    minimum = max(5, int(round(.58 * fallback_pitch_px)))
    maximum = max(minimum + 5, int(round(1.48 * fallback_pitch_px)))
    curves = []
    for x in np.linspace(x1 + .12 * (x2 - x1),
                         x2 - .12 * (x2 - x1), 23).round().astype(int):
        run = _column_run(mask[:, x])
        if run is None:
            continue
        ya, yb = run
        signal = gray[ya:yb, max(0, x - 2):x + 3].mean(axis=1).astype(np.float64)
        if signal.size < 7 * maximum or float(np.std(signal)) < 3.0:
            continue
        trend = cv2.GaussianBlur(signal.reshape(-1, 1), (1, 0),
                                 sigmaX=0, sigmaY=max(10., 2.5 * fallback_pitch_px))[:, 0]
        signal -= trend
        signal -= float(np.mean(signal))
        curve = []
        for lag in range(minimum, maximum + 1):
            a, b = signal[:-lag], signal[lag:]
            curve.append(float(np.dot(a, b) /
                               max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9)))
        curves.append(curve)
    if len(curves) < 5:
        return math.nan, 0.0
    curve = np.median(np.asarray(curves), axis=0)
    peaks = [index for index in range(1, len(curve) - 1)
             if curve[index] > curve[index - 1]
             and curve[index] >= curve[index + 1]]
    if not peaks:
        return math.nan, 0.0
    best_value = max(float(curve[index]) for index in peaks)
    plausible = [index for index in peaks if curve[index] >= .82 * best_value]
    if not plausible:
        # With an implausible search window every local maximum can be below
        # zero; multiplying the negative best value by .82 then produces a
        # threshold above the best sample. Treat this as no periodic evidence
        # rather than raising from min([]).
        return math.nan, 0.0
    index = min(plausible)  # reject a strong two-row harmonic when comparable
    denominator = curve[index - 1] - 2 * curve[index] + curve[index + 1]
    delta = (.5 * (curve[index - 1] - curve[index + 1]) / denominator
             if abs(denominator) > 1e-9 else 0.0)
    pitch = minimum + index + float(np.clip(delta, -.5, .5))
    column_curves = np.asarray(curves)
    support = float(np.mean(
        np.argmax(column_curves, axis=1) == index))
    prominence = float(curve[index] - np.median(curve))
    confidence = float(np.clip(.55 * max(0.0, curve[index]) +
                               .25 * max(0.0, prominence / .35) +
                               .20 * support, 0.0, 1.0))
    return float(pitch), confidence


def estimate_family_pitch_multiscale(
        gray: np.ndarray, mask: np.ndarray,
        minimum_pitch_px: float | None = None,
        maximum_pitch_px: float | None = None) -> MultiScalePitchEstimate:
    """Estimate rope-row pitch without a camera/rope-specific initial value.

    Several overlapping autocorrelation windows vote for the same physical
    period. A lone 2x harmonic cannot beat a repeatedly observed fundamental.
    Limits scale from the current wound-surface height and remain overrideable
    for a future camera resolution change.
    """
    bbox = _mask_bbox(mask)
    if bbox is None:
        return MultiScalePitchEstimate(False, "wound_surface_unavailable")
    surface_height = float(bbox[3] - bbox[1])
    minimum = (max(4.0, .006 * surface_height)
               if minimum_pitch_px is None else float(minimum_pitch_px))
    maximum = (min(96.0, max(24.0, .085 * surface_height))
               if maximum_pitch_px is None else float(maximum_pitch_px))
    if maximum <= minimum * 1.4:
        return MultiScalePitchEstimate(False, "invalid_pitch_search_range")
    # These are search-window centres, not assumed rope pitches.
    centers = np.geomspace(max(7.0, minimum / .72),
                           max(9.0, maximum / 1.15), 7)
    proposals = []
    for center in centers:
        pitch, confidence = estimate_family_pitch(
            gray, mask, float(center))
        if (math.isfinite(pitch) and minimum <= pitch <= maximum
                and confidence >= .14):
            proposals.append((float(pitch), float(confidence)))
    if not proposals:
        return MultiScalePitchEstimate(False, "no_consistent_period")
    ordered = sorted(proposals)
    clusters: list[list[tuple[float, float]]] = []
    for proposal in ordered:
        for cluster in clusters:
            center = float(np.median([item[0] for item in cluster]))
            if abs(proposal[0] / center - 1.0) <= .075:
                cluster.append(proposal)
                break
        else:
            clusters.append([proposal])

    def cluster_value(cluster):
        count = len(cluster)
        confidence = float(np.mean([item[1] for item in cluster]))
        consistency = math.exp(-8.0 * float(np.std(
            [item[0] for item in cluster])) /
                               max(np.mean([item[0] for item in cluster]), 1e-6))
        return count * confidence * consistency

    best = max(clusters, key=cluster_value)
    # If a 2x cluster has a slightly larger score, retain the lower fundamental
    # when it is supported by at least two windows and remains competitive.
    best_pitch = float(np.median([item[0] for item in best]))
    for cluster in clusters:
        pitch = float(np.median([item[0] for item in cluster]))
        if (len(cluster) >= 2 and 1.82 * pitch <= best_pitch <= 2.18 * pitch
                and cluster_value(cluster) >= .55 * cluster_value(best)):
            best = cluster
            best_pitch = pitch
    spread = float(np.std([item[0] for item in best]) /
                   max(best_pitch, 1e-6))
    confidence = float(np.clip(
        np.mean([item[1] for item in best]) *
        min(1.0, len(best) / 3.0) * math.exp(-7.0 * spread), 0.0, 1.0))
    if len(best) < 2 or confidence < .16:
        return MultiScalePitchEstimate(
            False, "period_lacks_multiscale_consensus", best_pitch,
            confidence, tuple(proposals), len(best))
    return MultiScalePitchEstimate(True, "ok", best_pitch, confidence,
                                   tuple(proposals), len(best))


def _ridge_response(gray: np.ndarray, pitch: float) -> np.ndarray:
    broad = cv2.GaussianBlur(gray.astype(np.float32), (0, 0),
                             sigmaX=max(6., 2.2 * pitch),
                             sigmaY=max(6., 2.2 * pitch))
    normalized = gray.astype(np.float32) - broad
    # A separable near-horizontal Gabor is much faster than three full 2-D
    # convolutions on the 4096x3000 source.  Path linking below absorbs the
    # observed small row slope/curvature instead of paying for a dense angle
    # bank on every frame.
    radius = min(30, max(7, int(round(1.5 * pitch))))
    coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
    vertical = (np.exp(-.5 * (coordinates / max(2.5, .42 * pitch)) ** 2) *
                np.cos(2 * np.pi * coordinates / pitch))
    vertical -= float(np.mean(vertical))
    vertical /= float(np.linalg.norm(vertical) + 1e-9)
    horizontal_size = max(5, int(round(.55 * pitch)) | 1)
    horizontal = cv2.getGaussianKernel(
        horizontal_size, max(1.5, .20 * pitch), cv2.CV_32F)[:, 0]
    return cv2.sepFilter2D(normalized, cv2.CV_32F,
                           horizontal, vertical)


def photometric_observation_mask(frame_bgr: np.ndarray,
                                 wound_mask: np.ndarray,
                                 pitch_px: float) -> np.ndarray:
    """Mark pixels whose local exposure and texture can support an observation.

    Rejected pixels may still receive an extrapolated family curve, but they
    are never drawn or counted as source observations.  Thresholds are local
    and pitch-scaled so this is not a fixed gray-level steel classifier.
    """
    if wound_mask.shape != frame_bgr.shape[:2]:
        wound_mask = cv2.resize(
            wound_mask.astype(np.uint8),
            (frame_bgr.shape[1], frame_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST) > 0
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sigma = max(3.0, .45 * float(pitch_px))
    mean = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    square_mean = cv2.GaussianBlur(gray * gray, (0, 0),
                                   sigmaX=sigma, sigmaY=sigma)
    local_std = np.sqrt(np.maximum(square_mean - mean * mean, 0.0))
    clipped = ((gray <= 3.0) | (gray >= 252.0)).astype(np.float32)
    clipped_density = cv2.GaussianBlur(
        clipped, (0, 0), sigmaX=max(2.0, .30 * pitch_px),
        sigmaY=max(2.0, .30 * pitch_px))
    values = local_std[wound_mask]
    illumination_values = mean[wound_mask]
    if values.size < 1000:
        return np.zeros_like(wound_mask, bool)
    # Darkness alone is not a failure: a dim but periodic region may be more
    # useful than a bright flat one. Reject only when exposure is extremely
    # low relative to the current wound surface or local texture collapses.
    illumination_reference = float(np.median(illumination_values))
    low_exposure = max(7.0, .18 * illumination_reference)
    high_exposure = min(249.0, max(220.0, 2.10 * illumination_reference))
    texture_floor = 2.5
    quality = (wound_mask & (local_std >= texture_floor) &
               (mean >= low_exposure) & (mean <= high_exposure) &
               (clipped_density <= .32))
    # Remove isolated speckles without closing broad invalid illumination bands.
    radius = max(2, int(round(.12 * pitch_px)))
    kernel = np.ones((radius, radius), np.uint8)
    return cv2.morphologyEx(quality.astype(np.uint8), cv2.MORPH_OPEN,
                            kernel) > 0


def _column_candidates(response: np.ndarray, mask: np.ndarray, x: int,
                       pitch: float,
                       observation_quality: np.ndarray | None = None
                       ) -> list[tuple[float, float]]:
    run = _column_run(mask[:, x])
    if run is None:
        return []
    ya, yb = run
    if yb - ya < 5 * pitch:
        return []
    values = response[ya:yb, max(0, x - 2):x + 3].mean(axis=1)
    if values.size < 3:
        return []
    if observation_quality is None:
        observable = np.ones(values.shape, bool)
    else:
        observable = (observation_quality[
            ya:yb, max(0, x - 2):x + 3].mean(axis=1) >= .50)
    if int(np.count_nonzero(observable)) < max(8, int(2.5 * pitch)):
        return []
    threshold = float(np.percentile(values[observable], 52))
    maxima = np.flatnonzero((values[1:-1] > values[:-2]) &
                            (values[1:-1] >= values[2:])) + 1
    maxima = [int(index) for index in maxima
              if observable[index] and values[index] >= threshold]
    minimum_distance = max(3, int(round(.62 * pitch)))
    selected: list[int] = []
    for index in sorted(maxima, key=lambda item: float(values[item]), reverse=True):
        if all(abs(index - other) >= minimum_distance for other in selected):
            selected.append(index)
    selected.sort()
    scale = float(np.std(values) + 1e-6)
    return [(float(ya + index), float((values[index] - threshold) / scale))
            for index in selected]


def _link_paths(columns: list[int], candidates: list[list[tuple[float, float]]],
                pitch: float,
                observable_columns: list[bool] | None = None
                ) -> list[list[tuple[float, float, float]]]:
    if not columns:
        return []
    if observable_columns is not None:
        # A clipped band often borders a strong filter transient with the
        # opposite optical phase. Seed in the interior of a trustworthy run,
        # not merely at the image centre or the first post-gap column.
        observable = np.asarray(observable_columns, bool)
        distance = np.zeros(len(columns), np.int32)
        for index in range(len(columns)):
            if observable[index]:
                invalid_left = np.flatnonzero(~observable[:index])
                invalid_right = np.flatnonzero(~observable[index + 1:])
                left = (index - int(invalid_left[-1])
                        if len(invalid_left) else index + 1)
                right = (int(invalid_right[0]) + 1
                         if len(invalid_right) else len(columns) - index)
                distance[index] = min(left, right)
        available = [index for index, items in enumerate(candidates)
                     if items and observable[index]]
        seed_index = (max(available,
                          key=lambda i: (distance[i], len(candidates[i]),
                                         -abs(i - len(columns) // 2)))
                      if available else len(columns) // 2)
    else:
        seed_index = min(range(len(columns)),
                         key=lambda i: (abs(i - len(columns) // 2),
                                        -len(candidates[i])))
    if not candidates[seed_index]:
        available = [i for i, items in enumerate(candidates) if items]
        if not available:
            return []
        seed_index = min(available, key=lambda i: abs(i - len(columns) // 2))
    paths = [[(float(columns[seed_index]), y, score)]
             for y, score in candidates[seed_index]]

    def extend(indices: list[int], prepend: bool) -> None:
        state = [{"y": path[0][1], "slope": 0.0, "gaps": 0,
                  "active": True} for path in paths]
        for column_index in indices:
            x = float(columns[column_index])
            pool = candidates[column_index]
            possibilities = []
            for track_index, track in enumerate(state):
                if not track["active"]:
                    continue
                last_x = paths[track_index][0 if prepend else -1][0]
                predicted = track["y"] + track["slope"] * (x - last_x)
                for candidate_index, (y, score) in enumerate(pool):
                    distance = abs(y - predicted)
                    if distance <= .72 * pitch:
                        possibilities.append((distance - .08 * score,
                                              track_index, candidate_index,
                                              predicted))
            used_tracks, used_candidates = set(), set()
            for _cost, track_index, candidate_index, _predicted in sorted(possibilities):
                if track_index in used_tracks or candidate_index in used_candidates:
                    continue
                y, score = pool[candidate_index]
                last = paths[track_index][0 if prepend else -1]
                dx = x - last[0]
                slope = ((y - last[1]) / dx) if abs(dx) > 1e-6 else 0.0
                state[track_index]["slope"] = float(np.clip(
                    .65 * state[track_index]["slope"] + .35 * slope,
                    -.45, .45))
                state[track_index]["y"] = y
                state[track_index]["gaps"] = 0
                point = (x, y, score)
                if prepend:
                    paths[track_index].insert(0, point)
                else:
                    paths[track_index].append(point)
                used_tracks.add(track_index)
                used_candidates.add(candidate_index)
            column_is_observable = (observable_columns is None or
                                    observable_columns[column_index])
            for track_index, track in enumerate(state):
                if track_index not in used_tracks and track["active"]:
                    if not column_is_observable:
                        continue
                    track["gaps"] += 1
                    if track["gaps"] > 3:
                        track["active"] = False

    extend(list(range(seed_index + 1, len(columns))), False)
    extend(list(range(seed_index - 1, -1, -1)), True)
    return paths


def extract_centerline_family(frame_bgr: np.ndarray, regions: SemanticRegions,
                              fallback_pitch_px: float = 18.0,
                              ridge_polarity: str = "bright",
                              pitch_override_px: float | None = None,
                              pitch_override_confidence: float = 0.0,
                              observation_quality_mask: np.ndarray | None = None,
                              ) -> CenterlineFamily:
    mask = source_wound_mask(regions, frame_bgr.shape)
    bbox = _mask_bbox(mask)
    if bbox is None:
        return CenterlineFamily(False, "wound_surface_unavailable")
    if ridge_polarity not in {"bright", "dark"}:
        raise ValueError("ridge_polarity must be 'bright' or 'dark'")
    if pitch_override_px is not None:
        pitch = float(pitch_override_px)
        pitch_confidence = float(pitch_override_confidence)
    else:
        pitch, pitch_confidence = estimate_family_pitch(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), mask,
            fallback_pitch_px)
    minimum_pitch_confidence = .16 if pitch_override_px is not None else .18
    if (not math.isfinite(pitch) or
            pitch_confidence < minimum_pitch_confidence):
        return CenterlineFamily(False, "family_pitch_unreliable",
                                bbox_xyxy=bbox, pitch_px=pitch,
                                pitch_confidence=pitch_confidence)
    erosion = max(3, int(round(.18 * pitch)))
    core = cv2.erode(mask.astype(np.uint8),
                     np.ones((erosion, erosion), np.uint8)) > 0
    x1, _y1, x2, _y2 = bbox
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    margin = max(12, int(round(2.0 * pitch)))
    rx1, ry1 = max(0, x1 - margin), max(0, _y1 - margin)
    rx2 = min(gray.shape[1], x2 + margin)
    ry2 = min(gray.shape[0], _y2 + margin)
    response = np.zeros(gray.shape, np.float32)
    response[ry1:ry2, rx1:rx2] = _ridge_response(
        gray[ry1:ry2, rx1:rx2], pitch)
    if ridge_polarity == "dark":
        response[ry1:ry2, rx1:rx2] *= -1.0
    step = max(5, int(round(.38 * pitch)))
    columns = list(range(x1 + step, x2 - step, step))
    quality = None
    if observation_quality_mask is not None:
        quality = observation_quality_mask
        if quality.shape != core.shape:
            quality = cv2.resize(
                quality.astype(np.uint8), (core.shape[1], core.shape[0]),
                interpolation=cv2.INTER_NEAREST) > 0
        quality = quality & core
    candidates = [_column_candidates(response, core, x, pitch, quality)
                  for x in columns]
    observable_columns = None
    if quality is not None:
        observable_columns = []
        for x in columns:
            run = _column_run(core[:, x])
            observable_columns.append(
                bool(run is not None and np.mean(quality[run[0]:run[1], x]) >= .18))
    paths = _link_paths(columns, candidates, pitch, observable_columns)
    lines = []
    for path in paths:
        coverage = len(path) / max(1, len(columns))
        if len(path) < 8 or coverage < .28:
            continue
        array = np.asarray(path, np.float64)
        keep = np.ones(len(array), dtype=bool)
        degree = 2 if len(array) >= 12 else 1
        for _ in range(3):
            coefficients = np.polyfit(array[keep, 0], array[keep, 1], degree)
            residual = np.abs(np.polyval(coefficients, array[:, 0]) - array[:, 1])
            keep = residual <= max(2.0, .30 * pitch)
            if int(keep.sum()) < max(6, degree + 2):
                break
        if int(keep.sum()) < 6:
            continue
        coefficients = np.polyfit(array[keep, 0], array[keep, 1], degree)
        xs = np.asarray(columns, np.float64)
        ys = np.polyval(coefficients, xs)
        height, width = mask.shape
        xi = np.clip(np.rint(xs).astype(int), 0, width - 1)
        yi = np.clip(np.rint(ys).astype(int), 0, height - 1)
        inside = mask[yi, xi]
        points = np.column_stack((xs[inside], ys[inside])).astype(np.float32)
        if len(points) < 6:
            continue
        lines.append(RopeCenterline(
            points, array[keep, :2].astype(np.float32), coverage,
            float(np.median(array[keep, 2]))))
    if len(lines) < 4:
        return CenterlineFamily(False, "too_few_linked_centerlines",
                                bbox, pitch, pitch_confidence, tuple(lines),
                                sampled_columns=len(columns))
    center_x = .5 * (x1 + x2)
    lines.sort(key=lambda line: float(np.interp(
        center_x, line.points_xy[:, 0], line.points_xy[:, 1])))
    center_y = np.asarray([np.interp(center_x, line.points_xy[:, 0],
                                    line.points_xy[:, 1]) for line in lines])
    spacings = np.diff(center_y)
    regularity = float(np.median(np.abs(spacings / pitch - 1.0)))
    coverage = float(np.median([line.coverage for line in lines]))
    confidence = float(np.clip(pitch_confidence *
                               min(1.0, len(lines) / 12.0) *
                               min(1.0, coverage / .55) *
                               math.exp(-1.8 * regularity), 0.0, 1.0))
    return CenterlineFamily(True, "ok", bbox, pitch, pitch_confidence,
                            tuple(lines), regularity, confidence,
                            len(columns))


def extract_centerline_family_multiphase(
        frame_bgr: np.ndarray, regions: SemanticRegions,
        minimum_pitch_px: float | None = None,
        maximum_pitch_px: float | None = None,
        use_photometric_quality: bool = False) -> MultiPhaseCenterlineResult:
    """Extract bright and dark lattices while preserving phase ambiguity."""
    mask = source_wound_mask(regions, frame_bgr.shape)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    pitch = estimate_family_pitch_multiscale(
        gray, mask, minimum_pitch_px, maximum_pitch_px)
    empty = CenterlineFamily(False, "automatic_pitch_unavailable")
    if not pitch.valid:
        return MultiPhaseCenterlineResult(
            False, pitch.reason, pitch, empty, empty)
    quality = (photometric_observation_mask(frame_bgr, mask, pitch.pitch_px)
               if use_photometric_quality else None)
    bright = extract_centerline_family(
        frame_bgr, regions, pitch.pitch_px, "bright",
        pitch.pitch_px, pitch.confidence, quality)
    dark = extract_centerline_family(
        frame_bgr, regions, pitch.pitch_px, "dark",
        pitch.pitch_px, pitch.confidence, quality)
    valid = [("bright", bright), ("dark", dark)]
    valid = [(name, family) for name, family in valid if family.valid]
    if not valid:
        return MultiPhaseCenterlineResult(
            False, "neither_phase_has_linked_family", pitch, bright, dark)

    def quality(family: CenterlineFamily) -> float:
        coverage = (float(np.median([line.coverage
                                    for line in family.centerlines]))
                    if family.centerlines else 0.0)
        regularity = (family.regularity_error
                      if math.isfinite(family.regularity_error) else 1.0)
        return (family.family_confidence * math.sqrt(max(coverage, 0.0)) *
                math.exp(-1.5 * regularity))

    selected_name, selected = max(valid, key=lambda item: quality(item[1]))
    if len(valid) == 1:
        ratio = math.inf
        ambiguous = False
    else:
        scores = sorted([quality(item[1]) for item in valid], reverse=True)
        ratio = scores[0] / max(scores[1], 1e-9)
        # Image quality can select the cleaner optical phase, but not prove
        # that it is the physical rope centre. Close scores remain explicit.
        ambiguous = ratio < 1.22
    phase_offset = math.nan
    if bright.valid and dark.valid:
        x = .5 * (bright.bbox_xyxy[0] + bright.bbox_xyxy[2])
        bright_y = [float(np.interp(x, line.points_xy[:, 0], line.points_xy[:, 1]))
                    for line in bright.centerlines
                    if line.points_xy[0, 0] <= x <= line.points_xy[-1, 0]]
        dark_y = [float(np.interp(x, line.points_xy[:, 0], line.points_xy[:, 1]))
                  for line in dark.centerlines
                  if line.points_xy[0, 0] <= x <= line.points_xy[-1, 0]]
        offsets = []
        for value in bright_y:
            if dark_y:
                offsets.append(min(abs(value - other) for other in dark_y) /
                               pitch.pitch_px)
        if offsets:
            phase_offset = float(np.median(offsets))
    return MultiPhaseCenterlineResult(
        True, "ok_phase_unresolved" if ambiguous else "ok_optical_phase_selected",
        pitch, bright, dark, selected_name, selected, ambiguous, ratio,
        phase_offset)


def _multiphase_family_score(result: MultiPhaseCenterlineResult) -> float:
    family = result.selected_family
    if family is None or not family.valid:
        return 0.0
    coverage = (float(np.median([line.coverage
                                for line in family.centerlines]))
                if family.centerlines else 0.0)
    return float(family.family_confidence * math.sqrt(max(coverage, 0.0)))


def _single_family_score(family: CenterlineFamily) -> float:
    if not family.valid:
        return 0.0
    coverage = (float(np.median([line.coverage
                                for line in family.centerlines]))
                if family.centerlines else 0.0)
    return float(family.family_confidence * math.sqrt(max(coverage, 0.0)))


def _quality_summary(quality: np.ndarray, wound: np.ndarray,
                     bbox: tuple[int, int, int, int] | None
                     ) -> tuple[float, float]:
    observable_fraction = (float(np.mean(quality[wound]))
                           if np.any(wound) else math.nan)
    if bbox is None:
        return observable_fraction, 1.0
    x1, _y1, x2, _y2 = bbox
    width = x2 - x1
    # Cylinder sides are naturally dark/foreshortened. Only internal holes
    # should trigger a second full reconstruction.
    inset = max(1, int(round(.22 * width)))
    wound_counts = np.count_nonzero(
        wound[:, x1 + inset:x2 - inset], axis=0)
    quality_counts = np.count_nonzero(
        quality[:, x1 + inset:x2 - inset], axis=0)
    fractions = quality_counts / np.maximum(wound_counts, 1)
    bad_columns = (wound_counts == 0) | (fractions < .18)
    indices = np.flatnonzero(bad_columns)
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    largest_bad_run = max((len(group) for group in groups), default=0)
    return observable_fraction, largest_bad_run / max(1, width)


def _quick_internal_bad_run_fraction(frame_bgr: np.ndarray,
                                     wound: np.ndarray,
                                     bbox: tuple[int, int, int, int] | None
                                     ) -> float:
    """Cheap prefilter for broad clipped/black holes in the cylinder centre."""
    if bbox is None or not np.any(wound):
        return 1.0
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    x1, _y1, x2, _y2 = bbox
    width = x2 - x1
    inset = max(1, int(round(.22 * width)))
    xa, xb = x1 + inset, x2 - inset
    sampled_gray = gray[::6, xa:xb].astype(np.float32)
    sampled_mask = wound[::6, xa:xb]
    global_values = gray[::6, :][wound[::6, :]]
    if global_values.size < 200:
        return 1.0
    reference = float(np.median(global_values))
    counts = np.count_nonzero(sampled_mask, axis=0)
    weighted = sampled_gray * sampled_mask
    means = np.sum(weighted, axis=0) / np.maximum(counts, 1)
    squares = np.sum(sampled_gray * sampled_gray * sampled_mask, axis=0)
    standard = np.sqrt(np.maximum(
        squares / np.maximum(counts, 1) - means * means, 0.0))
    clipped_counts = np.count_nonzero(
        sampled_mask & ((sampled_gray <= 3.0) | (sampled_gray >= 252.0)),
        axis=0)
    clipped = clipped_counts / np.maximum(counts, 1)
    very_dark_flat = (
        (means < max(6.0, .14 * reference)) &
        (standard < max(2.5, .025 * reference)))
    bad = (counts < 8) | (clipped >= .55) | very_dark_flat
    indices = np.flatnonzero(bad)
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    largest = max((len(group) for group in groups), default=0)
    return largest / max(1, width)


def extract_centerline_family_quality_hybrid(
        frame_bgr: np.ndarray, regions: SemanticRegions,
        minimum_pitch_px: float | None = None,
        maximum_pitch_px: float | None = None,
        required_score_ratio: float = 1.08) -> QualityGatedCenterlineResult:
    """Use quality completion only when it has clearly stronger evidence.

    A valid dim region remains available to the ordinary branch. Clipped or
    flat regions may be rejected and completed from the global regular family.
    Predicted paths remain distinct from source observations.
    """
    ungated = extract_centerline_family_multiphase(
        frame_bgr, regions, minimum_pitch_px, maximum_pitch_px, False)
    wound = source_wound_mask(regions, frame_bgr.shape)
    bbox = _mask_bbox(wound)
    if ungated.pitch.valid:
        pitch = ungated.pitch.pitch_px
    elif bbox is not None:
        pitch = max(8.0, .035 * max(1, bbox[3] - bbox[1]))
    else:
        pitch = 18.0
    quality = photometric_observation_mask(frame_bgr, wound, pitch)
    observable_fraction, largest_bad_run_fraction = _quality_summary(
        quality, wound, bbox)
    ungated_score = _multiphase_family_score(ungated)
    needs_gated_branch = (
        (not ungated.valid and ungated.pitch.valid) or
        largest_bad_run_fraction >= .235 or
        (ungated.valid and ungated_score < .18 and
         largest_bad_run_fraction >= .20))
    if not needs_gated_branch:
        return QualityGatedCenterlineResult(
            result=ungated, used_quality_gate=False,
            reason="quality_or_ungated_evidence_sufficient_skip_second_branch",
            ungated_score=ungated_score, gated_score=0.0,
            observable_fraction=observable_fraction,
            largest_bad_run_fraction=largest_bad_run_fraction,
            ungated_result=ungated)
    gated = extract_centerline_family_multiphase(
        frame_bgr, regions, minimum_pitch_px, maximum_pitch_px, True)
    gated_score = _multiphase_family_score(gated)
    use_gated = gated.valid and (
        not ungated.valid or
        gated_score > required_score_ratio * ungated_score or
        (largest_bad_run_fraction >= .235 and
         gated_score >= .80 * ungated_score))
    if use_gated:
        reason = ("ungated_invalid_quality_completion"
                  if not ungated.valid else "quality_completion_score_gain")
        selected = gated
    else:
        reason = ("ungated_evidence_retained" if ungated.valid
                  else "both_branches_invalid")
        selected = ungated
    return QualityGatedCenterlineResult(
        result=selected, used_quality_gate=use_gated, reason=reason,
        ungated_score=ungated_score, gated_score=gated_score,
        observable_fraction=observable_fraction,
        largest_bad_run_fraction=largest_bad_run_fraction,
        ungated_result=ungated, gated_result=gated)


def extract_single_family_quality_hybrid(
        frame_bgr: np.ndarray, regions: SemanticRegions,
        fallback_pitch_px: float = 18.0,
        ridge_polarity: str = "bright",
        required_score_ratio: float = 1.08
        ) -> QualityGatedSingleFamilyResult:
    """Conditional same-phase completion for background/runtime rebuilds."""
    ungated = extract_centerline_family(
        frame_bgr, regions, fallback_pitch_px, ridge_polarity)
    wound = source_wound_mask(regions, frame_bgr.shape)
    bbox = _mask_bbox(wound)
    pitch = (ungated.pitch_px if math.isfinite(ungated.pitch_px)
             else fallback_pitch_px)
    quick_bad_run = _quick_internal_bad_run_fraction(
        frame_bgr, wound, bbox)
    if ((ungated.valid and quick_bad_run < .08) or
            (not ungated.valid and
             (not math.isfinite(ungated.pitch_px) or
              ungated.pitch_confidence < .18))):
        return QualityGatedSingleFamilyResult(
            ungated, False, "quick_photometric_prefilter_skipped_quality_map",
            _single_family_score(ungated), 0.0, math.nan, quick_bad_run)
    quality = photometric_observation_mask(frame_bgr, wound, pitch)
    observable_fraction, largest_bad_run_fraction = _quality_summary(
        quality, wound, bbox)
    ungated_score = _single_family_score(ungated)
    needs_gated_branch = (
        (not ungated.valid and math.isfinite(ungated.pitch_px) and
         ungated.pitch_confidence >= .18) or
        largest_bad_run_fraction >= .235 or
        (ungated.valid and ungated_score < .18 and
         largest_bad_run_fraction >= .20))
    if not needs_gated_branch:
        return QualityGatedSingleFamilyResult(
            ungated, False,
            "quality_or_ungated_evidence_sufficient_skip_second_branch",
            ungated_score, 0.0, observable_fraction,
            largest_bad_run_fraction)
    if math.isfinite(ungated.pitch_px) and ungated.pitch_confidence >= .18:
        gated = extract_centerline_family(
            frame_bgr, regions, fallback_pitch_px, ridge_polarity,
            ungated.pitch_px, ungated.pitch_confidence, quality)
    else:
        gated = extract_centerline_family(
            frame_bgr, regions, fallback_pitch_px, ridge_polarity,
            observation_quality_mask=quality)
    gated_score = _single_family_score(gated)
    use_gated = gated.valid and (
        not ungated.valid or
        gated_score > required_score_ratio * ungated_score or
        (largest_bad_run_fraction >= .235 and
         gated_score >= .80 * ungated_score))
    return QualityGatedSingleFamilyResult(
        gated if use_gated else ungated, use_gated,
        ("quality_completion_selected" if use_gated
         else "ungated_evidence_retained"),
        ungated_score, gated_score, observable_fraction,
        largest_bad_run_fraction)


def estimate_gap_from_family(family: CenterlineFamily, x: float,
                             edge_y: float, tangent_y: float
                             ) -> FamilyGapEstimate:
    """Independent pitch/index estimate between flange edge and active tangent."""
    if (not family.valid or not all(math.isfinite(value) for value in
                                    (x, edge_y, tangent_y))):
        return FamilyGapEstimate(False, "family_or_geometry_unavailable")
    if tangent_y < edge_y - family.pitch_px:
        return FamilyGapEstimate(False, "tangent_above_edge")
    positions = []
    for line in family.centerlines:
        points = line.points_xy
        if points[0, 0] <= x <= points[-1, 0]:
            positions.append(float(np.interp(x, points[:, 0], points[:, 1])))
    positions = np.asarray(sorted(positions), np.float64)
    if positions.size < 4:
        return FamilyGapEstimate(False, "too_few_centerlines_at_tangent")
    local_differences = np.diff(positions)
    plausible = local_differences[
        (local_differences >= .55 * family.pitch_px) &
        (local_differences <= 1.55 * family.pitch_px)]
    local_pitch = (float(np.median(plausible)) if plausible.size >= 2
                   else family.pitch_px)
    continuous = max(0.0, (tangent_y - edge_y) / local_pitch)
    indexed = float(np.count_nonzero(
        (positions > min(edge_y, tangent_y)) &
        (positions <= max(edge_y, tangent_y))))
    disagreement = abs(indexed - continuous)
    confidence = family.family_confidence * math.exp(-disagreement / 2.0)
    return FamilyGapEstimate(True, "ok", continuous, indexed, local_pitch,
                             float(np.clip(confidence, 0.0, 1.0)))


def draw_centerline_family(frame_bgr: np.ndarray, family: CenterlineFamily,
                           gap: FamilyGapEstimate | None = None,
                           tangent: tuple[float, float] | None = None,
                           edge: tuple[float, float] | None = None
                           ) -> np.ndarray:
    shown = frame_bgr.copy()
    if family.bbox_xyxy != (0, 0, 0, 0):
        x1, y1, x2, y2 = family.bbox_xyxy
        cv2.rectangle(shown, (x1, y1), (x2, y2), (0, 180, 255), 3)
    for index, line in enumerate(family.centerlines):
        points = np.rint(line.points_xy).astype(np.int32).reshape(-1, 1, 2)
        color = (50, 220, 50) if index % 2 == 0 else (255, 180, 0)
        # The fitted family is a thin, muted guide. Only linked source-ridge
        # samples are drawn brightly; gaps through a hand/tool remain visibly
        # unsupported and can never masquerade as an observation.
        cv2.polylines(shown, [points], False, (90, 90, 90), 1, cv2.LINE_AA)
        observed = line.observed_points_xy
        if len(observed) >= 2:
            differences = np.diff(observed[:, 0])
            normal_step = float(np.median(differences)) if differences.size else 1.0
            splits = np.flatnonzero(differences > 2.2 * max(normal_step, 1.0)) + 1
            for segment in np.split(observed, splits):
                if len(segment) >= 2:
                    segment_points = np.rint(segment).astype(
                        np.int32).reshape(-1, 1, 2)
                    cv2.polylines(shown, [segment_points], False, color, 3,
                                  cv2.LINE_AA)
    if tangent is not None:
        cv2.circle(shown, tuple(np.rint(tangent).astype(int)), 10,
                   (255, 0, 255), -1, cv2.LINE_AA)
    if edge is not None:
        cv2.circle(shown, tuple(np.rint(edge).astype(int)), 10,
                   (0, 255, 255), -1, cv2.LINE_AA)
    text = (f"CENTERLINES={len(family.centerlines)} "
            f"PITCH={family.pitch_px:.2f}px "
            f"CONF={family.family_confidence:.2f}")
    cv2.putText(shown, text, (40, 65), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (0, 255, 255), 3, cv2.LINE_AA)
    if gap is not None and gap.valid:
        cv2.putText(shown,
                    f"GAP continuous={gap.continuous_count:.2f} indexed={gap.indexed_count:.0f}",
                    (40, 112), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 255), 3, cv2.LINE_AA)
    return shown
