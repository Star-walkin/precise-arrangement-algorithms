"""Conservative, read-only near-edge fallback for a partially segmented rope.

This is deliberately not a general replacement for the semantic model. The
old Hough detector alone reports false ropes in empty camera views; it can be
used here only when independent original-image and semantic-edge evidence
agree on the same physical line.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from geometry_measurement import adaptive_pitch_profile, local_pitch
from hybrid_geometry import HybridMeasurement, _semantic_inner_edge


def guarded_legacy_rescue_allowed(reason: str, mask_pixels: int,
                                  line_confidence: float,
                                  tangent_confidence: float,
                                  ridge_score: float,
                                  bright_positions: int) -> bool:
    """Gate original-resolution geometry with independent semantic/ridge evidence.

    A low-area semantic rope needs exceptionally strong evidence along the
    complete original-image line.  If the semantic rope is already strong and
    only its coarse inner edge failed, a lower ridge threshold is sufficient.
    """
    if reason == 'semantic_rope_presence_unconfirmed':
        return (mask_pixels >= 300 and line_confidence >= .58
                and tangent_confidence >= .28 and ridge_score >= 100
                and bright_positions >= 10)
    if reason in ('semantic_inner_edge_unreliable', 'local_pitch_unreliable',
                  'semantic_geometry_conflict'):
        return (mask_pixels >= 900 and line_confidence >= .70
                and tangent_confidence >= .40 and ridge_score >= 45
                and bright_positions >= 7)
    return False


def free_span_ridge_support(frame_bgr: np.ndarray, line, tangent_x: float,
                            analysis_scale: float) -> tuple[float, int]:
    """Measure rope-like contrast along the *detected* free-span axis.

    A long, neutral, bright ridge is required at many separated source pixels.
    Counting independent positions suppresses a single flange glint or spot.
    Only small crops are converted, keeping this fallback inexpensive at 12 MP.
    """
    if not line.valid or not 0 < analysis_scale <= 1:
        return 0.0, 0
    x1, y1 = line.x1 / analysis_scale, line.y1 / analysis_scale
    x2, y2 = line.x2 / analysis_scale, line.y2 / analysis_scale
    if x2 - x1 < 200:
        return 0.0, 0
    slope = (y2 - y1) / (x2 - x1)
    height, width = frame_bgr.shape[:2]
    start = max(tangent_x + 250, width * .61)
    stop = width * .94
    if stop <= start:
        return 0.0, 0
    scores = []
    for x in np.linspace(start, stop, 12):
        y = y1 + slope * (x - x1)
        ix, iy = round(x), round(y)
        if ix < 80 or ix >= width - 80 or iy < 80 or iy >= height - 80:
            continue
        patch = frame_bgr[iy-60:iy+61, ix-10:ix+11]
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        saturation = cv2.cvtColor(patch[55:66], cv2.COLOR_BGR2HSV)[:, :, 1]
        core = gray[55:66]
        sides = np.r_[gray[:30].ravel(), gray[90:120].ravel()]
        scores.append(float(np.percentile(core, 70) - np.median(sides) -
                            .10 * np.median(saturation)))
    return (float(np.median(scores)), sum(score >= 45 for score in scores)) if scores else (0.0, 0)


def measure_guarded_weak_rope(frame_bgr: np.ndarray, regions, legacy_result,
                             *, mask_pixels: int, analysis_scale: float,
                             fallback_row_pitch_px: float,
                             minimum_mask_pixels: int = 80,
                             maximum_near_edge_count: float = 6.0
                             ) -> HybridMeasurement | None:
    """Return a measurement only if five independent near-edge checks pass."""
    if mask_pixels < minimum_mask_pixels or legacy_result is None:
        return None
    line = legacy_result.line
    tangent = legacy_result.tangent
    if (not line.valid or line.confidence < .70 or not tangent.valid
            or tangent.provenance != 'measured' or tangent.confidence < .35
            or not math.isfinite(legacy_result.distance_rope)
            or not 0 <= legacy_result.distance_rope <= maximum_near_edge_count):
        return None
    if not 0 < analysis_scale <= 1:
        return None
    source_x = tangent.tangent_x / analysis_scale
    source_y = tangent.tangent_y / analysis_scale
    old_edge_y = tangent.edge_y / analysis_scale
    edge = _semantic_inner_edge(regions, source_x)
    if edge is None:
        return None
    edge_y, edge_confidence, _ = edge
    # The 2026-09-23 ruler/hand audit produced a superficially plausible line
    # but only 0.203 combined confidence and shifted the inner edge by roughly
    # half a row.  Weak semantic recovery is useful only when the semantic
    # wound boundary itself is convincing; reject such partial occlusions.
    if edge_confidence < .25 or abs(edge_y - old_edge_y) > 20:
        return None
    ridge_score, bright_positions = free_span_ridge_support(
        frame_bgr, line, source_x, analysis_scale)
    if ridge_score < 45 or bright_positions < 8:
        return None
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    pitch, pitch_confidence = local_pitch(
        gray, source_x, edge_y + fallback_row_pitch_px,
        fallback_row_pitch_px)
    if pitch_confidence < .35 or not math.isfinite(pitch) or pitch <= 0:
        return None
    gap = (source_y - edge_y) / pitch
    if not 0 <= gap <= maximum_near_edge_count + .5:
        return None
    return HybridMeasurement(
        True, tangent_x=source_x, tangent_y=source_y,
        edge_x=source_x, edge_y=edge_y, pitch_px=pitch,
        remaining_rope_count=gap, line_angle_deg=line.angle_deg,
        tangent_confidence=float(tangent.confidence),
        edge_confidence=float(edge_confidence),
        pitch_confidence=float(pitch_confidence),
        row_pitch_candidate_px=pitch,
        row_pitch_candidate_confidence=float(pitch_confidence),
        row_pitch_count_candidate=gap,
        source_line_end_x=line.x2 / analysis_scale,
        source_line_end_y=line.y2 / analysis_scale,
        reason='guarded_original_ridge_fallback')


def measure_guarded_legacy_geometry(frame_bgr: np.ndarray, legacy_result,
                                    *, failure_reason: str, mask_pixels: int,
                                    analysis_scale: float
                                    ) -> HybridMeasurement | None:
    """Rescue a current-frame legacy measurement after a specific semantic failure.

    This is not a Hough-only fallback: semantic rope pixels must exist and a
    long, bright, neutral ridge must independently support the same original-
    resolution line.  Predicted/history measurements are never accepted.
    """
    if legacy_result is None or not 0 < analysis_scale <= 1:
        return None
    line = legacy_result.line
    tangent = legacy_result.tangent
    if (not getattr(legacy_result, 'geometry_localized', False)
            or not line.valid or not tangent.valid
            or tangent.provenance != 'measured'
            or not math.isfinite(legacy_result.distance_rope)
            or not 0 <= legacy_result.distance_rope <= 40
            or not math.isfinite(tangent.local_pitch_px)
            or tangent.local_pitch_px <= 0):
        return None
    source_x = tangent.tangent_x / analysis_scale
    ridge_score, bright_positions = free_span_ridge_support(
        frame_bgr, line, source_x, analysis_scale)
    if not guarded_legacy_rescue_allowed(
            failure_reason, mask_pixels, float(line.confidence),
            float(tangent.confidence), ridge_score, bright_positions):
        return None
    adaptive = legacy_adaptive_pitch_candidate(
        frame_bgr, legacy_result, analysis_scale)
    if adaptive is None:
        adaptive_count = float(legacy_result.distance_rope)
        adaptive_pitch = tangent.local_pitch_px / analysis_scale
        adaptive_confidence = float(tangent.pitch_confidence)
    else:
        adaptive_count, adaptive_pitch, adaptive_confidence = adaptive
    return HybridMeasurement(
        True,
        tangent_x=tangent.tangent_x / analysis_scale,
        tangent_y=tangent.tangent_y / analysis_scale,
        edge_x=tangent.edge_x / analysis_scale,
        edge_y=tangent.edge_y / analysis_scale,
        pitch_px=adaptive_pitch,
        remaining_rope_count=adaptive_count,
        line_angle_deg=line.angle_deg,
        tangent_confidence=float(tangent.confidence),
        edge_confidence=float(tangent.confidence),
        pitch_confidence=adaptive_confidence,
        row_pitch_candidate_px=adaptive_pitch,
        row_pitch_candidate_confidence=adaptive_confidence,
        row_pitch_count_candidate=adaptive_count,
        source_line_end_x=line.x2 / analysis_scale,
        source_line_end_y=line.y2 / analysis_scale,
        reason='guarded_traditional_geometry_rescue')


def legacy_adaptive_pitch_candidate(frame_bgr: np.ndarray, legacy_result,
                                    analysis_scale: float,
                                    minimum_confidence: float = .20
                                    ) -> tuple[float, float, float] | None:
    """Return a current-frame count using an integrated local pitch profile."""
    if legacy_result is None or not 0 < analysis_scale <= 1:
        return None
    tangent = legacy_result.tangent
    if (not tangent.valid or tangent.provenance != 'measured'
            or not math.isfinite(tangent.local_pitch_px)
            or tangent.local_pitch_px <= 0):
        return None
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    pitch, confidence = adaptive_pitch_profile(
        gray,
        tangent.tangent_x / analysis_scale,
        tangent.edge_y / analysis_scale,
        tangent.tangent_y / analysis_scale,
        tangent.local_pitch_px / analysis_scale)
    if not math.isfinite(pitch) or confidence < minimum_confidence:
        return None
    distance = tangent.axis_distance_px / analysis_scale
    return float(distance / pitch), float(pitch), float(confidence)
