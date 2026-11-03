"""Read-only camera-specific semantic shadow pipeline.

This module has no actuator imports.  Its output is diagnostic only and must
not be used as an authorization signal for PLC, relay, or servo commands.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from async_centerline_runtime import AsyncCenterlineRuntime
from causal_silhouette import CausalSilhouetteTracker
from fast_centerline_tracker import FastCenterlineTracker
from illumination_quality import (IlluminationQualityResult,
                                  assess_illumination_quality)
from hybrid_geometry import (FrameStationarityDetector, HybridMeasurement,
                             HybridTemporalFilter,
                             StaticTangentBranchStabilizer,
                             measure_guarded_near_edge_geometry,
                             measure_hybrid_geometry)
from direction_estimation import ShortWindowDirectionTracker
from semantic_locator import SemanticLocator, SemanticRegions
from weak_rope_fallback import (measure_guarded_legacy_geometry,
                                measure_guarded_weak_rope,
                                legacy_adaptive_pitch_candidate)
from rotation_timing import (ArucoRotationEstimator, MarkerlessRotationEstimator,
                             RotationEstimate, flange_rotation_features)
from surface_centerlines import estimate_gap_from_family


def guarded_upper_envelope_count(semantic_count: float,
                                 traditional_count: float,
                                 maximum_disagreement: float
                                 ) -> tuple[float, bool, bool]:
    """Fuse two measured counts without allowing a large outlier to win."""
    disagreement = abs(traditional_count - semantic_count)
    conflict = disagreement > maximum_disagreement
    applied = not conflict and traditional_count > semantic_count
    return (traditional_count if applied else semantic_count), applied, conflict


@dataclass
class SemanticShadowResult:
    enabled: bool
    valid: bool
    reason: str
    remaining_rope_count: float = math.nan
    stabilized_remaining_rope_count: float = math.nan
    tangent_x: float = math.nan
    tangent_y: float = math.nan
    edge_x: float = math.nan
    edge_y: float = math.nan
    display_tangent_x: float = math.nan
    display_edge_x: float = math.nan
    confidence: float = 0.0
    processing_ms: float = 0.0
    views: dict[str, np.ndarray] | None = None
    presence_evidence: str = 'unconfirmed'
    incoming_mask_pixels: int = 0
    direction: str = 'UNKNOWN'
    fast_direction: str = 'UNKNOWN'
    row_pitch_candidate_px: float = math.nan
    row_pitch_candidate_confidence: float = 0.0
    row_pitch_count_candidate: float = math.nan
    count_method: str = 'unavailable'
    wound_high_saturation_fraction: float = math.nan
    wound_top_band_chroma_fraction: float = math.nan
    semantic_only_count: float = math.nan
    traditional_geometry_count: float = math.nan
    candidate_disagreement_rope: float = math.nan
    fused_candidate_applied: bool = False
    geometry_conflict: bool = False
    rotation_period_s: float = math.nan
    rotation_rpm: float = math.nan
    rotation_confidence: float = 0.0
    rotation_status: str = 'unavailable'
    rotation_reason: str = 'insufficient_history'
    tangent_confidence: float = 0.0
    edge_confidence: float = 0.0
    pitch_confidence: float = 0.0
    centerline_valid: bool = False
    centerline_reason: str = 'disabled'
    centerline_gap_count: float = math.nan
    centerline_confidence: float = 0.0
    centerline_pitch_px: float = math.nan
    centerline_snapshot_age_s: float = math.nan
    centerline_rebuild_pending: bool = False
    timing_count_valid: bool = False
    timing_gap_count: float = math.nan
    timing_count_confidence: float = 0.0
    timing_count_spread: float = math.nan
    silhouette_bump_px: float = math.nan
    silhouette_valid: bool = False
    silhouette_reason: str = 'disabled'
    silhouette_sample_new: bool = False
    illumination_quality: IlluminationQualityResult | None = None
    # The quality metric is intentionally cached for several analyzed frames.
    # Consumers must only add a sample when this flag is true and deduplicate
    # with ``illumination_quality_sample_id``.  Otherwise a slow cached metric
    # is accidentally counted as many independent measurements.
    illumination_quality_sample_new: bool = False
    illumination_quality_sample_id: str = ''
    illumination_quality_source_frame: int = -1
    illumination_quality_source_time_s: float = math.nan
    illumination_quality_computed_at_s: float = math.nan
    illumination_stream_epoch: int = 0
    illumination_quality_pitch_px: float = math.nan
    illumination_quality_pitch_source: str = 'unavailable'


def _fuse_live_count_candidates(
        candidates: list[tuple[str, float, float]]) -> tuple[bool, float, float, float]:
    usable = [(value, max(.02, confidence))
              for _name, value, confidence in candidates
              if math.isfinite(value) and value >= 0 and confidence > 0]
    if len(usable) < 2:
        return False, math.nan, 0.0, math.nan
    values = np.asarray([item[0] for item in usable], np.float64)
    spread = float(np.max(values) - np.min(values))
    if spread > 2.0:
        return False, math.nan, 0.0, spread
    weights = np.asarray([item[1] for item in usable], np.float64)
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    selected = order[int(np.searchsorted(cumulative, .5 * cumulative[-1]))]
    confidence = float(np.clip(
        np.mean(weights) * math.exp(-spread / 1.5), 0.0, 1.0))
    return True, float(values[selected]), confidence, spread


def _render(frame: np.ndarray, regions: SemanticRegions,
            measurement: HybridMeasurement,
            display_scale: float) -> dict[str, np.ndarray]:
    """Render at UI resolution instead of allocating three full 12 MP masks."""
    height, width = frame.shape[:2]
    display_width = max(1, round(width * display_scale))
    display_height = max(1, round(height * display_scale))
    semantic = cv2.resize(frame, (display_width, display_height),
                          interpolation=cv2.INTER_AREA)
    layer = semantic.copy()
    x1, y1, x2, y2 = regions.crop_xyxy
    dx1, dy1 = round(x1 * display_scale), round(y1 * display_scale)
    dx2, dy2 = round(x2 * display_scale), round(y2 * display_scale)
    crop_width, crop_height = max(1, dx2 - dx1), max(1, dy2 - dy1)
    colors = ((45, 175, 45), (0, 205, 255), (0, 140, 255))
    for index, color in enumerate(colors):
        mask = cv2.resize(regions.masks[index].astype(np.uint8),
                          (crop_width, crop_height), interpolation=cv2.INTER_NEAREST) > 0
        crop = layer[dy1:dy2, dx1:dx2]
        crop[mask] = color
    semantic = cv2.addWeighted(semantic, .68, layer, .32, 0)
    line_width = max(2, round(4 * display_scale))
    cv2.rectangle(semantic, (dx1, dy1), (dx2, dy2), (255, 255, 255), line_width)
    cv2.putText(semantic, 'SEMANTIC SHADOW - DISPLAY ONLY',
                (round(35 * display_scale), round(70 * display_scale)),
                cv2.FONT_HERSHEY_SIMPLEX, max(.55, 1.25 * display_scale),
                (255, 255, 255), max(1, round(3 * display_scale)), cv2.LINE_AA)

    geometry = semantic.copy()
    if measurement.valid:
        tangent = (round(measurement.tangent_x * display_scale),
                   round(measurement.tangent_y * display_scale))
        edge = (round(measurement.edge_x * display_scale),
                round(measurement.edge_y * display_scale))
        if (math.isfinite(measurement.source_line_end_x)
                and math.isfinite(measurement.source_line_end_y)):
            cv2.line(geometry, tangent,
                     (round(measurement.source_line_end_x * display_scale),
                      round(measurement.source_line_end_y * display_scale)),
                     (255, 255, 0), max(2, round(6 * display_scale)), cv2.LINE_AA)
        cv2.circle(geometry, tangent, max(6, round(18 * display_scale)),
                   (255, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(geometry, edge, max(5, round(14 * display_scale)),
                   (100, 255, 100), -1, cv2.LINE_AA)
        cv2.line(geometry, tangent, edge, (100, 255, 100),
                 max(3, round(8 * display_scale)), cv2.LINE_AA)
        label = f'SHADOW GAP {measurement.remaining_rope_count:.2f} PITCHES - NO ACTUATION'
        color = (100, 255, 100)
    else:
        label = f'SHADOW INVALID: {measurement.reason}'
        color = (0, 180, 255)
    cv2.rectangle(geometry, (round(25 * display_scale), round(85 * display_scale)),
                  (min(display_width - 1, round(1750 * display_scale)),
                   round(150 * display_scale)), (0, 0, 0), -1)
    cv2.putText(geometry, label,
                (round(45 * display_scale), round(132 * display_scale)),
                cv2.FONT_HERSHEY_SIMPLEX, max(.52, 1.15 * display_scale), color,
                max(1, round(3 * display_scale)), cv2.LINE_AA)
    return {'semantic': semantic, 'geometry': geometry}


class SemanticShadowProcessor:
    def __init__(self, locator: SemanticLocator, calibration: dict,
                 mirror_input_horizontal: bool, replace_panels: bool = True,
                 stabilization_window_frames: int = 5,
                 display_scale: float = 0.5,
                 minimum_incoming_mask_pixels: int = 0,
                 minimum_tangent_confidence: float = 0.0,
                 guarded_weak_rope_fallback: bool = False,
                 guarded_traditional_geometry_rescue: bool = False,
                 guarded_dual_geometry_count_fusion: bool = False,
                 maximum_fusion_disagreement_rope: float = 2.0,
                 legacy_analysis_scale: float = .5,
                 rotation_minimum_period_s: float = 1.0,
                 rotation_maximum_period_s: float = 7.0,
                 rotation_history_s: float = 18.0,
                 rotation_marker_id: int = 17,
                 use_fixed_tangent_x_range: bool = True,
                 live_centerline_enabled: bool = False,
                 centerline_rebuild_interval_s: float = 1.5,
                 centerline_maximum_width: int = 600,
                 centerline_foreground_interval_s: float = 1.0,
                 silhouette_interval_s: float = .25,
                 camera_role: str = 'unknown',
                 illumination_quality_interval_s: float = .5,
                 illumination_quality_maximum_dimension: int = 720):
        self.locator = locator
        self.calibration = calibration
        self.mirror_input_horizontal = mirror_input_horizontal
        self.replace_panels = replace_panels
        self.display_scale = min(1.0, max(0.1, float(display_scale)))
        self.minimum_incoming_mask_pixels = max(0, int(minimum_incoming_mask_pixels))
        self.minimum_tangent_confidence = max(0.0, float(minimum_tangent_confidence))
        self.guarded_weak_rope_fallback = bool(guarded_weak_rope_fallback)
        self.guarded_traditional_geometry_rescue = bool(
            guarded_traditional_geometry_rescue)
        self.guarded_dual_geometry_count_fusion = bool(
            guarded_dual_geometry_count_fusion)
        self.maximum_fusion_disagreement_rope = max(
            0.0, float(maximum_fusion_disagreement_rope))
        self.legacy_analysis_scale = float(legacy_analysis_scale)
        self.camera_role = str(camera_role)
        self.automatic_actuation_enabled = False
        self.illumination_quality_interval_s = max(
            .1, float(illumination_quality_interval_s))
        self.illumination_quality_maximum_dimension = max(
            128, int(illumination_quality_maximum_dimension))
        self.filter = HybridTemporalFilter(window_frames=stabilization_window_frames)
        self.frame_stationarity_detector = FrameStationarityDetector()
        self.static_tangent_branch_stabilizer = StaticTangentBranchStabilizer()
        self.direction_tracker = ShortWindowDirectionTracker(
            window_s=5.0, minimum_displacement_rope=.8,
            minimum_speed_rope_s=.10, required_support=3,
            minimum_span_s=2.5)
        self.fast_direction_tracker = ShortWindowDirectionTracker(
            window_s=1.2, minimum_displacement_rope=.25,
            minimum_speed_rope_s=.10, required_support=3,
            maximum_missing_s=.35, minimum_span_s=.35)
        self.rotation_estimator = MarkerlessRotationEstimator(
            minimum_period_s=rotation_minimum_period_s,
            maximum_period_s=rotation_maximum_period_s,
            history_s=rotation_history_s)
        self.marker_rotation_estimator = ArucoRotationEstimator(
            marker_id=rotation_marker_id,
            minimum_period_s=rotation_minimum_period_s,
            maximum_period_s=rotation_maximum_period_s)
        self.use_fixed_tangent_x_range = bool(use_fixed_tangent_x_range)
        self.centerline_runtime = None
        self.silhouette_tracker = None
        # ``0`` means: run the lightweight narrow-band tracker for every
        # frame that already reached this analysis worker.  The expensive
        # full-family rebuild is still scheduled and throttled internally by
        # ``AsyncCenterlineRuntime`` / ``FastCenterlineTracker``.  Keeping an
        # outer multi-second throttle here would freeze the tracked phase and
        # defeat the purpose of the fast path near a layer transition.
        self.centerline_foreground_interval_s = max(
            0.0, float(centerline_foreground_interval_s))
        self.silhouette_interval_s = max(.1, float(silhouette_interval_s))
        self.last_centerline_wall_s = -math.inf
        self.last_silhouette_wall_s = -math.inf
        self.cached_centerline_update = None
        self.cached_silhouette = None
        self.last_illumination_quality_wall_s = -math.inf
        self.cached_illumination_quality = None
        self.illumination_quality_sequence = 0
        self.illumination_stream_epoch = 0
        self.cached_illumination_quality_sample_id = ''
        self.cached_illumination_quality_source_frame = -1
        self.cached_illumination_quality_source_time_s = math.nan
        self.cached_illumination_quality_computed_at_s = math.nan
        self.cached_illumination_quality_pitch_px = math.nan
        self.cached_illumination_quality_pitch_source = 'unavailable'
        if live_centerline_enabled:
            row_pitch = (float(calibration['fallback_texture_period_px']) *
                         float(calibration['texture_to_row_pitch_ratio']))
            self.centerline_runtime = AsyncCenterlineRuntime(
                FastCenterlineTracker(
                    row_pitch,
                    rebuild_interval_s=float(centerline_rebuild_interval_s),
                    maximum_width=int(centerline_maximum_width)),
                use_quality_gate=False)
            self.silhouette_tracker = CausalSilhouetteTracker()

    @classmethod
    def from_config(cls, config: dict, config_path: Path | str,
                    enable_live_centerline: bool = False):
        settings = config.get('semantic_shadow', {})
        if not settings.get('enabled', False):
            return None
        if settings.get('automatic_actuation_enabled', False):
            raise ValueError(
                'semantic shadow automatic actuation must remain disabled')
        base = Path(config_path).resolve().parent
        model = (base / settings['model']).resolve()
        metadata = (base / settings['metadata']).resolve()
        calibration_path = (base / settings['calibration']).resolve()
        for path in (model, metadata, calibration_path):
            if not path.is_file():
                raise FileNotFoundError(f'语义影子文件不存在：{path}')
        calibration = json.loads(calibration_path.read_text(encoding='utf-8'))
        if calibration.get('status') != 'offline_shadow_calibration_not_for_device_control':
            raise ValueError('影子标定缺少只读安全标记')
        if calibration.get('camera_role') != config.get('camera', {}).get('role'):
            raise ValueError('语义影子标定与相机角色不匹配')
        rotation = settings.get('rotation_period', {})
        return cls(
            SemanticLocator(model, metadata), calibration,
            bool(config.get('camera', {}).get('mirror_input_horizontal', False)),
            bool(settings.get('replace_diagnostic_panels', True)),
            int(settings.get('stabilization_window_frames', 5)),
            float(settings.get('display_scale', 0.5)),
            int(settings.get('minimum_incoming_mask_pixels', 0)),
            float(settings.get('minimum_tangent_confidence', 0.0)),
            bool(settings.get('guarded_weak_rope_fallback', False)),
            bool(settings.get('guarded_traditional_geometry_rescue', False)),
            bool(settings.get('guarded_dual_geometry_count_fusion', False)),
            float(settings.get('maximum_fusion_disagreement_rope', 2.0)),
            float(config['video']['analysis_scale']),
            float(rotation.get('minimum_period_s', 1.0)),
            float(rotation.get('maximum_period_s', 7.0)),
            float(rotation.get('history_s', 18.0)),
            int(rotation.get('aruco_marker_id', 17)),
            bool(settings.get('use_fixed_tangent_x_range', True)),
            bool(enable_live_centerline),
            float(settings.get('live_centerline_rebuild_interval_s', 1.5)),
            int(settings.get('live_centerline_maximum_width', 600)),
            float(settings.get('live_centerline_foreground_interval_s', 1.0)),
            float(settings.get('live_silhouette_interval_s', .25)),
            camera_role=str(config.get('camera', {}).get('role', 'unknown')),
            illumination_quality_interval_s=float(
                settings.get('illumination_quality_interval_s', .5)),
            illumination_quality_maximum_dimension=int(
                settings.get('illumination_quality_maximum_dimension', 720)))

    def reset(self, preserve_centerline_structure: bool = False):
        self.filter.reset()
        # The live worker deliberately analyzes only the newest frame.  Under
        # dual-camera CPU load the interval between analyzed frames can exceed
        # the motion/event continuity limit even though acquisition itself is
        # healthy.  Always reset direction/event filters, but keep the
        # *diagnostic-only* static geometry candidate across such intentional
        # skips.  Its next image is still compared with the fixed reference;
        # any visible motion releases the lock immediately.  A real camera or
        # frame-number reset passes preserve_centerline_structure=False and
        # clears everything.
        if not preserve_centerline_structure:
            self.frame_stationarity_detector.reset()
            self.static_tangent_branch_stabilizer.reset()
        self.direction_tracker.reset()
        self.fast_direction_tracker.reset()
        self.rotation_estimator.reset()
        self.marker_rotation_estimator.reset()
        if self.centerline_runtime is not None and not preserve_centerline_structure:
            self.centerline_runtime.reset()
        if self.silhouette_tracker is not None:
            self.silhouette_tracker.reset()
        if not preserve_centerline_structure:
            self.last_centerline_wall_s = -math.inf
            self.cached_centerline_update = None
        self.last_silhouette_wall_s = -math.inf
        self.cached_silhouette = None
        self.last_illumination_quality_wall_s = -math.inf
        self.cached_illumination_quality = None
        self.illumination_stream_epoch += 1
        self.cached_illumination_quality_sample_id = ''
        self.cached_illumination_quality_source_frame = -1
        self.cached_illumination_quality_source_time_s = math.nan
        self.cached_illumination_quality_computed_at_s = math.nan
        self.cached_illumination_quality_pitch_px = math.nan
        self.cached_illumination_quality_pitch_source = 'unavailable'

    def close(self):
        if self.centerline_runtime is not None:
            self.centerline_runtime.close()
            self.centerline_runtime = None

    def process(self, frame_bgr: np.ndarray, *, render: bool = True,
                 time_s: float | None = None,
                 legacy_result=None,
                 source_frame: int | None = None) -> SemanticShadowResult:
        started = time.perf_counter()
        canonical = cv2.flip(frame_bgr, 1) if self.mirror_input_horizontal else frame_bgr
        regions = self.locator.predict(canonical)
        wall_now = time.monotonic()
        illumination_quality_sample_new = False
        if (self.cached_illumination_quality is None or
                wall_now - self.last_illumination_quality_wall_s >=
                self.illumination_quality_interval_s):
            row_pitch = (float(self.calibration['fallback_texture_period_px']) *
                         float(self.calibration['texture_to_row_pitch_ratio']))
            pitch_source = 'camera_calibration_fallback'
            # Winding radius changes the projected row spacing.  Reuse the
            # preceding high-confidence full-resolution line family when it is
            # available; a one-sample delay is preferable to making the light
            # metric itself chase noisy current-frame geometry.
            prior_centerline = self.cached_centerline_update
            if (prior_centerline is not None and
                    prior_centerline.tracking.valid and
                    prior_centerline.tracking.family.valid and
                    math.isfinite(prior_centerline.tracking.family.pitch_px) and
                    prior_centerline.tracking.family.pitch_px >= 4.0):
                row_pitch = float(prior_centerline.tracking.family.pitch_px)
                pitch_source = 'previous_live_centerline'
            self.cached_illumination_quality = assess_illumination_quality(
                canonical, regions, row_pitch, camera_role=self.camera_role,
                maximum_dimension=self.illumination_quality_maximum_dimension)
            computed_at_s = time.monotonic()
            self.last_illumination_quality_wall_s = computed_at_s
            self.illumination_quality_sequence += 1
            self.cached_illumination_quality_sample_id = (
                f'{self.illumination_stream_epoch}:'
                f'{self.illumination_quality_sequence}')
            self.cached_illumination_quality_source_frame = (
                int(source_frame) if source_frame is not None else -1)
            self.cached_illumination_quality_source_time_s = (
                float(time_s) if time_s is not None else math.nan)
            self.cached_illumination_quality_computed_at_s = computed_at_s
            self.cached_illumination_quality_pitch_px = row_pitch
            self.cached_illumination_quality_pitch_source = pitch_source
            illumination_quality_sample_new = True
        illumination_quality = self.cached_illumination_quality
        silhouette_updated = False
        if (self.silhouette_tracker is not None and time_s is not None and
                wall_now - self.last_silhouette_wall_s >=
                self.silhouette_interval_s):
            self.cached_silhouette = self.silhouette_tracker.update(
                canonical, regions, float(time_s))
            self.last_silhouette_wall_s = time.monotonic()
            silhouette_updated = True
        silhouette = self.cached_silhouette
        x1, y1, x2, y2 = regions.crop_xyxy
        low_height, low_width = regions.masks.shape[1:]
        low_crop = cv2.resize(canonical[y1:y2, x1:x2],
                              (low_width, low_height), interpolation=cv2.INTER_AREA)
        stationary_confirmed = self.frame_stationarity_detector.update(
            low_crop, time_s)
        low_saturation = cv2.cvtColor(low_crop, cv2.COLOR_BGR2HSV)[:, :, 1]
        wound_mask = regions.mask('wound_rope_surface')
        rotation_features = flange_rotation_features(
            low_crop, regions.mask('flange_inner_face'))
        if time_s is not None:
            marker_rotation = self.marker_rotation_estimator.update(time_s, low_crop)
            markerless_rotation = self.rotation_estimator.update(
                time_s, rotation_features, stationary=stationary_confirmed)
            rotation = (marker_rotation if marker_rotation.status == 'measured_marker'
                        else markerless_rotation)
            if stationary_confirmed:
                # Stillness is visual evidence, not a hardware stop signal.
                # Never continue an old periodic countdown over a still scene.
                self.marker_rotation_estimator.reset()
                rotation = RotationEstimate(reason='scene_stationary')
        else:
            rotation = RotationEstimate(reason='timestamp_unavailable')
        wound_high_saturation_fraction = (
            float(np.mean(low_saturation[wound_mask] >= 80))
            if np.any(wound_mask) else math.nan)
        top_band = np.zeros_like(wound_mask, dtype=bool)
        for column in np.flatnonzero(wound_mask.any(axis=0)):
            top = int(np.argmax(wound_mask[:, column]))
            top_band[top:min(low_height, top + 48), column] = True
        top_band &= wound_mask
        wound_top_band_chroma_fraction = (
            float(np.mean(low_saturation[top_band] >= 60))
            if np.any(top_band) else math.nan)
        mask_pixels = int(regions.mask('incoming_rope_free_span').sum())
        presence = ('supported' if mask_pixels >= self.minimum_incoming_mask_pixels
                    else 'unconfirmed')
        if presence == 'supported':
            tangent_x_range = (tuple(self.calibration['tangent_x_valid_range'])
                               if (self.use_fixed_tangent_x_range and
                                   'tangent_x_valid_range' in self.calibration)
                               else None)
            # Absolute pixel ranges belong to the old camera pose. If the
            # current-frame original-resolution geometry places the tangent
            # clearly outside that range, do not force the semantic path back
            # into the old coordinates; let the guarded rescue arbitrate.
            if (tangent_x_range is not None and legacy_result is not None
                    and legacy_result.tangent.valid
                    and legacy_result.tangent.provenance == 'measured'):
                legacy_source_x = (legacy_result.tangent.tangent_x /
                                   self.legacy_analysis_scale)
                margin = 2.0 * float(self.calibration['fallback_texture_period_px'])
                if not (tangent_x_range[0] - margin <= legacy_source_x <=
                        tangent_x_range[1] + margin):
                    tangent_x_range = None
            measurement = measure_hybrid_geometry(
                canonical, regions,
                fallback_pitch_px=float(self.calibration['fallback_texture_period_px']),
                texture_to_row_pitch_ratio=float(self.calibration['texture_to_row_pitch_ratio']),
                maximum_texture_period_deviation=float(
                    self.calibration['maximum_texture_period_deviation']),
                tangent_x_range=tangent_x_range)
            if (measurement.valid and measurement.tangent_confidence <
                    self.minimum_tangent_confidence):
                measurement = replace(measurement, valid=False,
                                      reason='original_tangent_signal_unreliable')
        else:
            measurement = HybridMeasurement(False, reason='semantic_rope_presence_unconfirmed')
        correction = self.calibration.get('count_linear_correction', {})
        count_method = 'unavailable'
        if not measurement.valid and self.guarded_weak_rope_fallback:
            fallback = measure_guarded_weak_rope(
                canonical, regions, legacy_result,
                mask_pixels=mask_pixels,
                analysis_scale=self.legacy_analysis_scale,
                fallback_row_pitch_px=(
                    float(self.calibration['fallback_texture_period_px']) *
                    float(self.calibration['texture_to_row_pitch_ratio'])))
            if fallback is not None:
                measurement = fallback
                presence = 'supported_by_original_ridge'
                count_method = 'guarded_original_ridge_row_pitch'
        if not measurement.valid and self.guarded_traditional_geometry_rescue:
            rescue = measure_guarded_near_edge_geometry(
                canonical, regions, legacy_result, measurement,
                analysis_scale=self.legacy_analysis_scale)
            if rescue is not None:
                measurement = rescue
                presence = 'supported_by_semantic_and_original_ridge'
                count_method = 'guarded_near_edge_original_geometry'
        if not measurement.valid and self.guarded_traditional_geometry_rescue:
            rescue = measure_guarded_legacy_geometry(
                canonical, legacy_result,
                failure_reason=measurement.reason,
                mask_pixels=mask_pixels,
                analysis_scale=self.legacy_analysis_scale)
            if rescue is not None:
                measurement = rescue
                presence = 'supported_by_semantic_and_original_ridge'
                count_method = 'guarded_traditional_geometry_rescue'
        if measurement.valid and count_method == 'unavailable':
            # Independent light/dark wound-row period can avoid carrying a
            # 4.15 mm strand-texture calibration into a new rope gauge. Require
            # a strong periodic signal; otherwise keep the established camera
            # calibration. This choice is diagnostic and is evaluated on held-out
            # annotations before any wider claim is made.
            if (measurement.row_pitch_candidate_confidence >= .55
                    and math.isfinite(measurement.row_pitch_count_candidate)):
                measurement = replace(
                    measurement,
                    remaining_rope_count=measurement.row_pitch_count_candidate,
                    pitch_px=measurement.row_pitch_candidate_px)
                count_method = 'bright_dark_row_pitch'
            else:
                corrected = max(0.0, float(correction.get('scale', 1.0)) *
                                measurement.remaining_rope_count +
                                float(correction.get('offset', 0.0)))
                measurement = replace(measurement, remaining_rope_count=corrected)
                count_method = 'calibrated_texture_period'
        semantic_only_count = (float(measurement.remaining_rope_count)
                               if measurement.valid else math.nan)
        traditional_count = math.nan
        candidate_disagreement = math.nan
        fused_candidate_applied = False
        geometry_conflict = False
        if (measurement.valid and self.guarded_dual_geometry_count_fusion
                and count_method != 'guarded_near_edge_original_geometry'
                and legacy_result is not None
                and getattr(legacy_result, 'geometry_localized', False)
                and legacy_result.tangent.valid
                and legacy_result.tangent.provenance == 'measured'
                and math.isfinite(legacy_result.distance_rope)
                and legacy_result.distance_rope >= 0):
            adaptive_candidate = legacy_adaptive_pitch_candidate(
                canonical, legacy_result, self.legacy_analysis_scale)
            raw_traditional_count = float(legacy_result.distance_rope)
            # An adaptive pitch may only refine a legacy geometry candidate
            # that already agrees with the independent semantic count.  If the
            # raw geometries conflict, retain the conflict for explicit
            # arbitration instead of using pitch refinement to hide it.
            traditional_count = (
                adaptive_candidate[0]
                if (adaptive_candidate is not None
                    and abs(raw_traditional_count - semantic_only_count) <=
                    self.maximum_fusion_disagreement_rope
                    and adaptive_candidate[0] - semantic_only_count <= 1.0)
                else raw_traditional_count)
            candidate_disagreement = abs(traditional_count - semantic_only_count)
            selected, fused_candidate_applied, geometry_conflict = (
                guarded_upper_envelope_count(
                    semantic_only_count, traditional_count,
                    self.maximum_fusion_disagreement_rope))
            # Both edge estimators are biased toward under-counting when their
            # visible boundary is partly hidden by the flange.  Use the upper
            # envelope only under close agreement.  A large disagreement is
            # exposed as a conflict rather than silently selecting either one.
            if fused_candidate_applied:
                measurement = replace(
                    measurement, remaining_rope_count=selected,
                    reason='guarded_dual_geometry_fusion')
                count_method = 'guarded_dual_geometry_upper_envelope'
            elif geometry_conflict:
                semantic_geometry_confidence = min(
                    measurement.tangent_confidence,
                    measurement.edge_confidence,
                    measurement.pitch_confidence)
                if (semantic_geometry_confidence <= .02
                        and self.guarded_traditional_geometry_rescue):
                    rescue = measure_guarded_legacy_geometry(
                        canonical, legacy_result,
                        failure_reason='semantic_geometry_conflict',
                        mask_pixels=mask_pixels,
                        analysis_scale=self.legacy_analysis_scale)
                    if rescue is not None:
                        measurement = rescue
                        presence = 'supported_by_semantic_and_original_ridge'
                        count_method = 'guarded_traditional_geometry_rescue'
                        fused_candidate_applied = True
                        geometry_conflict = False
        measurement = self.static_tangent_branch_stabilizer.update(
            measurement, time_s, stationary_confirmed)
        centerline_update = None
        centerline_gap = None
        if (self.centerline_runtime is not None and time_s is not None and
                (self.centerline_foreground_interval_s <= 0.0 or
                 time.monotonic() - self.last_centerline_wall_s >=
                 self.centerline_foreground_interval_s)):
            self.cached_centerline_update = self.centerline_runtime.update(
                canonical, regions, float(time_s))
            self.last_centerline_wall_s = time.monotonic()
        centerline_update = self.cached_centerline_update
        if centerline_update is not None:
            family = centerline_update.tracking.family
            centerline_gap = estimate_gap_from_family(
                family, measurement.tangent_x, measurement.edge_y,
                measurement.tangent_y)
        timing_candidates = []
        if measurement.valid:
            timing_candidates.append((
                'hybrid', measurement.remaining_rope_count,
                min(measurement.edge_confidence,
                    measurement.pitch_confidence)))
        if (measurement.valid and not geometry_conflict
                and math.isfinite(measurement.row_pitch_count_candidate)
                and not math.isclose(measurement.row_pitch_count_candidate,
                                     measurement.remaining_rope_count,
                                     rel_tol=1e-9, abs_tol=1e-9)
                and measurement.row_pitch_candidate_confidence >= .25):
            # The selected measurement may already be this very row-pitch
            # result (including guarded original geometry). Counting a copy
            # twice is not multi-method support. All candidates still share
            # the checked tangent/edge and are only pitch-consistency evidence.
            timing_candidates.append((
                'row_pitch', measurement.row_pitch_count_candidate,
                measurement.row_pitch_candidate_confidence))
        if (measurement.valid and not geometry_conflict
                and centerline_gap is not None and centerline_gap.valid):
            timing_candidates.append((
                'centerline_family', centerline_gap.continuous_count,
                centerline_gap.confidence))
        timing_valid, timing_gap, timing_confidence, timing_spread = (
            _fuse_live_count_candidates(timing_candidates))
        stabilized = self.filter.update(measurement, time_s=time_s)
        direction = (self.direction_tracker.update(
            time_s, measurement.remaining_rope_count if measurement.valid else None,
            1.0)
            if time_s is not None else 'UNKNOWN')
        fast_direction = (self.fast_direction_tracker.update(
            time_s, measurement.remaining_rope_count if measurement.valid else None,
            1.0)
            if time_s is not None else 'UNKNOWN')
        views = _render(canonical, regions, measurement, self.display_scale) if render else None
        if views is not None and self.mirror_input_horizontal:
            views = {name: cv2.flip(view, 1) for name, view in views.items()}
        return SemanticShadowResult(
            enabled=True, valid=measurement.valid, reason=measurement.reason,
            remaining_rope_count=measurement.remaining_rope_count,
            stabilized_remaining_rope_count=stabilized.remaining_rope_count,
            tangent_x=measurement.tangent_x, tangent_y=measurement.tangent_y,
            edge_x=measurement.edge_x, edge_y=measurement.edge_y,
            display_tangent_x=(canonical.shape[1] - 1 - measurement.tangent_x
                               if self.mirror_input_horizontal else measurement.tangent_x),
            display_edge_x=(canonical.shape[1] - 1 - measurement.edge_x
                            if self.mirror_input_horizontal else measurement.edge_x),
            confidence=min(measurement.tangent_confidence,
                           measurement.edge_confidence,
                           measurement.pitch_confidence) if measurement.valid else 0.0,
            processing_ms=(time.perf_counter() - started) * 1000.0,
            views=views, presence_evidence=presence,
            incoming_mask_pixels=mask_pixels, direction=direction,
            fast_direction=fast_direction,
            row_pitch_candidate_px=measurement.row_pitch_candidate_px,
            row_pitch_candidate_confidence=measurement.row_pitch_candidate_confidence,
            row_pitch_count_candidate=measurement.row_pitch_count_candidate,
            count_method=count_method,
            wound_high_saturation_fraction=wound_high_saturation_fraction,
            wound_top_band_chroma_fraction=wound_top_band_chroma_fraction,
            semantic_only_count=semantic_only_count,
            traditional_geometry_count=traditional_count,
            candidate_disagreement_rope=candidate_disagreement,
            fused_candidate_applied=fused_candidate_applied,
            geometry_conflict=geometry_conflict,
            rotation_period_s=rotation.period_s,
            rotation_rpm=rotation.rpm,
            rotation_confidence=rotation.confidence,
            rotation_status=rotation.status,
            rotation_reason=rotation.reason,
            tangent_confidence=measurement.tangent_confidence,
            edge_confidence=measurement.edge_confidence,
            pitch_confidence=measurement.pitch_confidence,
            centerline_valid=bool(
                centerline_update is not None and
                centerline_update.tracking.valid and centerline_gap is not None
                and centerline_gap.valid),
            centerline_reason=(
                centerline_gap.reason if centerline_gap is not None else
                'live_centerline_disabled'),
            centerline_gap_count=(
                centerline_gap.continuous_count
                if centerline_gap is not None and centerline_gap.valid
                else math.nan),
            centerline_confidence=(
                centerline_gap.confidence
                if centerline_gap is not None and centerline_gap.valid else 0.0),
            centerline_pitch_px=(
                centerline_update.tracking.family.pitch_px
                if centerline_update is not None and
                centerline_update.tracking.valid else math.nan),
            centerline_snapshot_age_s=(
                centerline_update.snapshot_age_s
                if centerline_update is not None else math.nan),
            centerline_rebuild_pending=bool(
                centerline_update is not None and
                centerline_update.rebuild_pending),
            timing_count_valid=timing_valid,
            timing_gap_count=timing_gap,
            timing_count_confidence=timing_confidence,
            timing_count_spread=timing_spread,
            silhouette_bump_px=(
                silhouette.bump_px if silhouette is not None and silhouette.valid
                else math.nan),
            silhouette_valid=bool(silhouette is not None and silhouette.valid),
            silhouette_reason=(silhouette.reason if silhouette is not None
                               else 'live_silhouette_disabled'),
            silhouette_sample_new=silhouette_updated,
            illumination_quality=illumination_quality,
            illumination_quality_sample_new=illumination_quality_sample_new,
            illumination_quality_sample_id=(
                self.cached_illumination_quality_sample_id),
            illumination_quality_source_frame=(
                self.cached_illumination_quality_source_frame),
            illumination_quality_source_time_s=(
                self.cached_illumination_quality_source_time_s),
            illumination_quality_computed_at_s=(
                self.cached_illumination_quality_computed_at_s),
            illumination_stream_epoch=self.illumination_stream_epoch,
            illumination_quality_pitch_px=(
                self.cached_illumination_quality_pitch_px),
            illumination_quality_pitch_source=(
                self.cached_illumination_quality_pitch_source))
