"""Read-only absolute background registration against one immutable image.

GFTT, bidirectional LK and robust partial-affine estimation use only the upper
background crop. Every measurement starts at the same reference, never the
previous result. A valid registration is NOT proof that a spool is stopped,
that a rope is visible, or that a physical winding event occurred. No physical
timestamps, video decoding, trained model, or device access are involved.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class FixedBackgroundPoseParameters:
    maximum_width: int = 1024
    background_top_fraction: float = .40
    maximum_features: int = 400
    feature_quality: float = .01
    feature_minimum_distance_px: float = 10.
    lk_window_size: int = 21
    lk_pyramid_levels: int = 3
    maximum_lk_error: float = 25.
    maximum_forward_backward_px: float = 1.
    ransac_reprojection_px: float = 1.5
    minimum_tracks: int = 40
    minimum_inliers: int = 30
    minimum_inlier_fraction: float = .65
    maximum_median_reprojection_px: float = .75
    minimum_grid_cells: int = 6
    grid_columns: int = 4
    grid_rows: int = 3
    maximum_center_shift_diagonal_fraction: float = .01
    maximum_rotation_deg: float = .7
    maximum_scale_change: float = .008


DEFAULT_PARAMETERS = FixedBackgroundPoseParameters()


def _validate_parameters(p):
    if not isinstance(p, FixedBackgroundPoseParameters):
        raise ValueError('parameters_must_be_FixedBackgroundPoseParameters')
    values = asdict(p)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in values.values()):
        raise ValueError('invalid_parameters')
    integer_names = ('maximum_width', 'maximum_features', 'lk_window_size',
                     'lk_pyramid_levels', 'minimum_tracks', 'minimum_inliers',
                     'minimum_grid_cells', 'grid_columns', 'grid_rows')
    if any(not isinstance(values[k], int) for k in integer_names):
        raise ValueError('invalid_integer_parameters')
    if not (64 <= p.maximum_width <= 1024 and 0 < p.background_top_fraction <= .40
            and 0 < p.feature_quality <= 1 and p.feature_minimum_distance_px > 0
            and p.maximum_features >= p.minimum_tracks >= 3
            and p.minimum_tracks >= p.minimum_inliers >= 3
            and p.lk_window_size >= 3 and p.lk_window_size % 2 == 1
            and p.lk_pyramid_levels >= 0 and p.maximum_lk_error > 0
            and p.maximum_forward_backward_px > 0 and p.ransac_reprojection_px > 0
            and 0 < p.minimum_inlier_fraction <= 1
            and p.maximum_median_reprojection_px > 0
            and p.grid_columns >= 1 and p.grid_rows >= 1
            and 1 <= p.minimum_grid_cells <= p.grid_columns * p.grid_rows
            and 0 < p.maximum_center_shift_diagonal_fraction <= .01
            and 0 < p.maximum_rotation_deg <= .7
            and 0 < p.maximum_scale_change <= .008):
        raise ValueError('invalid_parameters')


def _frame(frame, p):
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise ValueError('frame_must_be_uint8_array')
    if frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 2:
        gray = frame
    else:
        raise ValueError('frame_must_be_gray_or_BGR')
    h, w = gray.shape
    if min(h, w) < 80:
        raise ValueError('frame_too_small')
    scale = min(1., p.maximum_width / w)
    sw, sh = round(w * scale), round(h * scale)
    if min(sw, sh) < 1:
        raise ValueError('resized_frame_too_small')
    small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA) if scale < 1 else gray
    top = int(sh * p.background_top_fraction)
    if top < 8:
        raise ValueError('background_crop_too_small')
    # Copy even for a grayscale input: subsequent caller mutations cannot
    # change the fixed reference. No object-identity shortcut is permitted.
    return small[:top].copy(), tuple(frame.shape), (h, w), (sw / w, sh / h)


def _occupied_cells(points, shape, columns, rows):
    h, w = shape
    x = np.clip((points[:, 0] / w * columns).astype(int), 0, columns - 1)
    y = np.clip((points[:, 1] / h * rows).astype(int), 0, rows - 1)
    return len(set(zip(x.tolist(), y.tolist())))


def _original_affine(matrix, scale_xy):
    resize = np.diag([*scale_xy, 1.])
    homogeneous = np.eye(3)
    homogeneous[:2] = matrix
    return (np.linalg.inv(resize) @ homogeneous @ resize)[:2]


def _pose_metrics(affine, shape):
    center = np.array([(shape[1] - 1) / 2, (shape[0] - 1) / 2])
    displacement = affine[:, :2] @ center + affine[:, 2] - center
    scale = math.sqrt(abs(float(np.linalg.det(affine[:, :2]))))
    return dict(center_shift_xy_px=displacement.tolist(),
                center_shift_px=float(np.linalg.norm(displacement)),
                center_shift_diagonal_fraction=float(np.linalg.norm(displacement) / np.hypot(*shape)),
                rotation_deg=math.degrees(math.atan2(affine[1, 0], affine[0, 0])),
                scale=scale, scale_change=abs(scale - 1.))


class FixedBackgroundPose:
    """Map original reference pixels to current pixels without temporal drift.

    ``measure`` returns ``{reference_to_current, valid, diagnostics}``.
    ``reference_to_current`` is a 2x3 list only when all guards pass; otherwise
    it is None. A rejected fit, if available, is diagnostic-only. Feature/LK/
    RANSAC pixel thresholds refer to the resized crop, while the returned
    matrix and pose displacement refer to original image coordinates.

    Parameters mirror V17 quality limits but cap *total reference-relative*
    displacement at 1% of the full-image diagonal, rotation at 0.7 degrees,
    and scale change at 0.008. Parameters can tighten but not loosen these
    three motion caps. No reference refresh API is intentionally provided.
    """

    def __init__(self, reference_bgr, parameters=DEFAULT_PARAMETERS):
        _validate_parameters(parameters)
        self.parameters = parameters
        self._reference, self._input_shape, self._shape, self._scale_xy = _frame(reference_bgr, parameters)
        self._reference.setflags(write=False)
        self._points = cv2.goodFeaturesToTrack(
            self._reference, maxCorners=parameters.maximum_features,
            qualityLevel=parameters.feature_quality,
            minDistance=parameters.feature_minimum_distance_px, blockSize=7)
        if self._points is not None:
            self._points = np.asarray(self._points, np.float32).reshape(-1, 1, 2).copy()
            self._points.setflags(write=False)

    def measure(self, frame_bgr):
        started = time.perf_counter()
        p = self.parameters
        diagnostics = dict(reason='uninitialized', reference_is_fixed=True,
            reference_refreshed=False, cumulative_transform_chaining=False,
            reference_shape=list(self._input_shape), registration_crop_shape=list(self._reference.shape),
            resize_scale_xy=list(self._scale_xy), detected_features=0 if self._points is None else len(self._points),
            forward_tracks=0, validated_tracks=0, inliers=0, inlier_fraction=None,
            forward_backward_median_px=None, reprojection_median_px=None, reprojection_p90_px=None,
            occupied_reference_cells=0, occupied_current_cells=0, pose=None,
            fit_available=False, estimated_reference_to_current=None,
            parameters=asdict(p), physical_time_inferred=False, spool_motion_inferred=False,
            rope_visibility_inferred=False, hardware_access=False,
            warning='Background registration only; not spool motion, rope visibility, or a winding event.')
        result = dict(reference_to_current=None, valid=False, diagnostics=diagnostics)

        def finish(reason):
            diagnostics['reason'] = reason
            diagnostics['processing_elapsed_ms'] = (time.perf_counter() - started) * 1000
            return result

        try:
            current, input_shape, _, _ = _frame(frame_bgr, p)
        except (ValueError, TypeError, OverflowError, cv2.error) as exc:
            return finish(str(exc))
        if input_shape != self._input_shape:
            return finish('image_shape_changed')
        if self._points is None:
            return finish('no_background_features')
        if len(self._points) < p.minimum_tracks:
            return finish('insufficient_background_features')
        options = dict(winSize=(p.lk_window_size, p.lk_window_size), maxLevel=p.lk_pyramid_levels,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        try:
            tracked, status, errors = cv2.calcOpticalFlowPyrLK(
                self._reference, current, self._points, None, **options)
            if tracked is None or status is None or errors is None:
                return finish('forward_tracking_failed')
            source = self._points.reshape(-1, 2)
            target = tracked.reshape(-1, 2)
            good = (np.asarray(status).reshape(-1) > 0) & (np.asarray(errors).reshape(-1) <= p.maximum_lk_error)
            good &= np.isfinite(target).all(axis=1)
            good &= ((target[:, 0] >= 0) & (target[:, 0] < current.shape[1])
                     & (target[:, 1] >= 0) & (target[:, 1] < current.shape[0]))
            diagnostics['forward_tracks'] = int(good.sum())
            if good.sum() < p.minimum_tracks:
                return finish('insufficient_forward_tracks')
            source, target = source[good], target[good]
            returned, reverse_status, reverse_errors = cv2.calcOpticalFlowPyrLK(
                current, self._reference, target.reshape(-1, 1, 2), None, **options)
            if returned is None or reverse_status is None or reverse_errors is None:
                return finish('backward_tracking_failed')
            fb = np.linalg.norm(returned.reshape(-1, 2) - source, axis=1)
            keep = (np.asarray(reverse_status).reshape(-1) > 0) & np.isfinite(fb)
            keep &= np.asarray(reverse_errors).reshape(-1) <= p.maximum_lk_error
            keep &= fb <= p.maximum_forward_backward_px
            source, target, fb = source[keep], target[keep], fb[keep]
            diagnostics['validated_tracks'] = len(source)
            if len(source) < p.minimum_tracks:
                return finish('insufficient_forward_backward_tracks')
            diagnostics['forward_backward_median_px'] = float(np.median(fb))
            affine, mask = cv2.estimateAffinePartial2D(
                source, target, method=cv2.RANSAC, ransacReprojThreshold=p.ransac_reprojection_px,
                maxIters=2000, confidence=.99, refineIters=10)
        except cv2.error:
            return finish('opencv_registration_failed')
        if affine is None or mask is None or not np.isfinite(affine).all():
            return finish('affine_estimation_failed')
        inliers = np.asarray(mask).reshape(-1) > 0
        diagnostics.update(inliers=int(inliers.sum()), inlier_fraction=float(inliers.mean()))
        if inliers.sum() < p.minimum_inliers or inliers.mean() < p.minimum_inlier_fraction:
            return finish('insufficient_affine_inliers')
        residual = np.linalg.norm(source @ affine[:, :2].T + affine[:, 2] - target, axis=1)[inliers]
        source_cells = _occupied_cells(source[inliers], self._reference.shape, p.grid_columns, p.grid_rows)
        target_cells = _occupied_cells(target[inliers], current.shape, p.grid_columns, p.grid_rows)
        full = _original_affine(affine, self._scale_xy)
        pose = _pose_metrics(full, self._shape)
        diagnostics.update(fit_available=True, estimated_reference_to_current=full.tolist(),
            reprojection_median_px=float(np.median(residual)), reprojection_p90_px=float(np.percentile(residual, 90)),
            occupied_reference_cells=source_cells, occupied_current_cells=target_cells, pose=pose)
        if min(source_cells, target_cells) < p.minimum_grid_cells:
            return finish('background_support_too_local')
        if np.median(residual) > p.maximum_median_reprojection_px:
            return finish('background_reprojection_unstable')
        checks = dict(translation=pose['center_shift_diagonal_fraction'] <= p.maximum_center_shift_diagonal_fraction,
                      rotation=abs(pose['rotation_deg']) <= p.maximum_rotation_deg,
                      scale=pose['scale_change'] <= p.maximum_scale_change)
        diagnostics['motion_checks'] = checks
        if not all(checks.values()):
            return finish('background_pose_change_' + '_'.join(k for k, v in checks.items() if not v))
        result.update(reference_to_current=full.tolist(), valid=True)
        return finish('supported_fixed_reference_registration')
