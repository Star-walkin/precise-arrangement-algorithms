"""Background-pose continuity guard for offline historical-boundary research.

Only the upper image region is tracked. This module has no rope/event labels,
no trained model, no video/file access, and no hardware commands. Stable means
small, supported camera/background displacement, NOT a stopped winding spool.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class SceneGuardParameters:
    maximum_width: int = 1024
    background_top_fraction: float = .40
    maximum_frame_gap_s: float = .60
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
    maximum_center_shift_diagonal_fraction: float = .005
    maximum_rotation_deg: float = .35
    maximum_scale_change: float = .004


DEFAULT_PARAMETERS = SceneGuardParameters()


def _frame(frame, parameters):
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
    scale = min(1., parameters.maximum_width / w)
    sw, sh = round(w * scale), round(h * scale)
    small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA) if scale < 1 else gray.copy()
    top = max(32, int(sh * parameters.background_top_fraction))
    return np.ascontiguousarray(small[:top]), (h, w), (sw/w, sh/h)


def occupied_cells(points, crop_shape, columns=4, rows=3):
    points = np.asarray(points, float).reshape(-1, 2)
    if not len(points):
        return 0
    h, w = crop_shape
    x = np.clip((points[:, 0] / w * columns).astype(int), 0, columns-1)
    y = np.clip((points[:, 1] / h * rows).astype(int), 0, rows-1)
    return len(set(zip(x.tolist(), y.tolist())))


def original_affine(small_affine, previous_scale_xy, current_scale_xy):
    """Convert cropped-origin-zero resized coordinates into original pixels."""
    matrix = np.asarray(small_affine, float)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError('invalid_affine')
    before = np.diag([*previous_scale_xy, 1.])
    after_inverse = np.diag([1/current_scale_xy[0], 1/current_scale_xy[1], 1.])
    homogeneous = np.eye(3); homogeneous[:2] = matrix
    return (after_inverse @ homogeneous @ before)[:2]


def pose_metrics(affine, image_shape):
    affine = np.asarray(affine, float)
    center = np.array([(image_shape[1]-1)/2, (image_shape[0]-1)/2])
    displacement = affine[:, :2] @ center + affine[:, 2] - center
    scale = math.sqrt(abs(float(np.linalg.det(affine[:, :2]))))
    rotation = math.degrees(math.atan2(affine[1, 0], affine[0, 0]))
    return dict(center_shift_xy_px=displacement.tolist(), center_shift_px=float(np.linalg.norm(displacement)),
        center_shift_diagonal_fraction=float(np.linalg.norm(displacement)/np.hypot(*image_shape)),
        rotation_deg=rotation, scale=scale, scale_change=abs(scale-1.))


class ScenePoseGuard:
    """Consume chronological frames; every rejection breaks historical support.

    A rejected but decodable frame becomes the next background baseline. An
    invalid/missing frame clears the baseline. Consumers must reset their own
    historical geometry whenever history_action is reset. A diagnostic affine
    may still exist for excessive motion; that never authorizes continuation.
    """

    def __init__(self, parameters=DEFAULT_PARAMETERS):
        if not isinstance(parameters, SceneGuardParameters):
            raise ValueError('parameters must be SceneGuardParameters')
        if not (64 <= parameters.maximum_width <= 1024 and 0 < parameters.background_top_fraction <= .5
                and parameters.maximum_frame_gap_s > 0 and parameters.minimum_tracks >= 3
                and parameters.minimum_inliers >= 3 and 0 < parameters.minimum_inlier_fraction <= 1
                and parameters.minimum_grid_cells <= parameters.grid_columns * parameters.grid_rows):
            raise ValueError('invalid_parameters')
        self.parameters = parameters
        self.reset()

    def reset(self):
        self._gray = None
        self._shape = None
        self._scale = None
        self._time_s = None

    def update(self, frame, time_s):
        started = time.perf_counter(); p = self.parameters
        out = dict(pose_stable=False, measurement_available=False, history_action='reset',
            reason='uninitialized', previous_to_current=None, previous_time_s=self._time_s,
            current_time_s=None, time_gap_s=None, detected_features=0, forward_tracks=0,
            validated_tracks=0, inliers=0, inlier_fraction=None, forward_backward_median_px=None,
            reprojection_median_px=None, reprojection_p90_px=None, occupied_background_cells=0,
            pose=None, frame_pair_contiguous=False, spool_motion_inferred=False,
            hardware_access=False, parameters=asdict(p),
            warning='Background pose stability does not indicate spool stopped or rope visibility.')

        def finish(reason):
            out['reason'] = reason
            out['elapsed_ms'] = (time.perf_counter()-started)*1000
            return out

        try:
            timestamp = float(time_s)
            if not math.isfinite(timestamp):
                raise ValueError('nonfinite_timestamp')
            gray, shape, scale = _frame(frame, p)
        except (ValueError, TypeError, OverflowError) as exc:
            self.reset()
            return finish(str(exc))
        out['current_time_s'] = timestamp
        previous, prev_shape, prev_scale, previous_time = self._gray, self._shape, self._scale, self._time_s
        self._gray, self._shape, self._scale, self._time_s = gray, shape, scale, timestamp
        if previous is None:
            return finish('first_frame_requires_new_history')
        gap = timestamp-previous_time; out['time_gap_s'] = gap
        if gap <= 0:
            return finish('nonincreasing_timestamp')
        if gap > p.maximum_frame_gap_s:
            return finish('frame_gap_exceeds_limit')
        if prev_shape != shape:
            return finish('image_shape_changed')
        out['frame_pair_contiguous'] = True
        points = cv2.goodFeaturesToTrack(previous, maxCorners=p.maximum_features,
            qualityLevel=p.feature_quality, minDistance=p.feature_minimum_distance_px, blockSize=7)
        if points is None:
            return finish('no_background_features')
        points = np.asarray(points, np.float32).reshape(-1, 1, 2)
        out['detected_features'] = len(points)
        if len(points) < p.minimum_tracks:
            return finish('insufficient_background_features')
        options = dict(winSize=(p.lk_window_size, p.lk_window_size), maxLevel=p.lk_pyramid_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        next_points, status, errors = cv2.calcOpticalFlowPyrLK(previous, gray, points, None, **options)
        if next_points is None or status is None or errors is None:
            return finish('forward_tracking_failed')
        source = points.reshape(-1, 2); target = next_points.reshape(-1, 2)
        good = (np.asarray(status).reshape(-1) > 0) & (np.asarray(errors).reshape(-1) <= p.maximum_lk_error)
        good &= np.isfinite(target).all(axis=1)
        good &= (target[:, 0] >= 0) & (target[:, 0] < gray.shape[1]) & (target[:, 1] >= 0) & (target[:, 1] < gray.shape[0])
        out['forward_tracks'] = int(good.sum())
        if good.sum() < p.minimum_tracks:
            return finish('insufficient_forward_tracks')
        source, target = source[good], target[good]
        returned, reverse_status, _ = cv2.calcOpticalFlowPyrLK(gray, previous, target.reshape(-1,1,2), None, **options)
        if returned is None or reverse_status is None:
            return finish('backward_tracking_failed')
        reverse = returned.reshape(-1, 2); fb = np.linalg.norm(reverse-source, axis=1)
        keep = (np.asarray(reverse_status).reshape(-1) > 0) & np.isfinite(fb) & (fb <= p.maximum_forward_backward_px)
        source, target, fb = source[keep], target[keep], fb[keep]
        out['validated_tracks'] = len(source)
        if len(source) < p.minimum_tracks:
            return finish('insufficient_forward_backward_tracks')
        out['forward_backward_median_px'] = float(np.median(fb))
        affine, mask = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC,
            ransacReprojThreshold=p.ransac_reprojection_px, maxIters=2000, confidence=.99, refineIters=10)
        if affine is None or mask is None or not np.isfinite(affine).all():
            return finish('affine_estimation_failed')
        inliers = np.asarray(mask).reshape(-1) > 0
        out['inliers'] = int(inliers.sum()); out['inlier_fraction'] = float(inliers.mean())
        if inliers.sum() < p.minimum_inliers or inliers.mean() < p.minimum_inlier_fraction:
            return finish('insufficient_affine_inliers')
        predicted = source @ affine[:, :2].T + affine[:, 2]
        residual = np.linalg.norm(predicted-target, axis=1)[inliers]
        cells = occupied_cells(source[inliers], previous.shape, p.grid_columns, p.grid_rows)
        full = original_affine(affine, prev_scale, scale)
        out.update(previous_to_current=full.tolist(), measurement_available=True,
            reprojection_median_px=float(np.median(residual)), reprojection_p90_px=float(np.percentile(residual,90)),
            occupied_background_cells=cells, pose=pose_metrics(full, shape))
        if cells < p.minimum_grid_cells:
            return finish('background_support_too_local')
        if np.median(residual) > p.maximum_median_reprojection_px:
            return finish('background_reprojection_unstable')
        pose = out['pose']
        checks = dict(translation=pose['center_shift_diagonal_fraction'] <= p.maximum_center_shift_diagonal_fraction,
            rotation=abs(pose['rotation_deg']) <= p.maximum_rotation_deg,
            scale=pose['scale_change'] <= p.maximum_scale_change)
        out['motion_checks'] = checks
        if not all(checks.values()):
            return finish('background_pose_change_'+'_'.join(k for k,v in checks.items() if not v))
        out.update(pose_stable=True, history_action='continue')
        return finish('supported_small_background_displacement')
