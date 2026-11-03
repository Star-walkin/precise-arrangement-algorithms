"""Short-lived, causal boundary support; predicted pixels never become observed.

Only direct flange observations update their observation timestamps. Rope paths
are NOT an input. Background registration and overlapping direct boundary checks
gate propagation, but cannot prove the hidden flange has not changed radially.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numpy as np


@dataclass(frozen=True)
class BoundaryParameters:
    grid_step_px: float = 4.
    maximum_segment_link_px: float = 8.
    minimum_direct_sigma_px: float = 1.5
    direct_sigma_width_fraction: float = .08
    maximum_history_s: float = 2.
    maximum_frame_gap_s: float = .60
    sigma_growth_widths_per_second: float = .12
    maximum_sigma_width_fraction: float = .35
    minimum_overlap_points: int = 12
    minimum_overlap_span_widths: float = 5.
    maximum_overlap_shift_widths: float = .35
    maximum_overlap_p90_widths: float = .30


PARAMETERS = BoundaryParameters()


def _runs(indices):
    return [v for v in np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1) if len(v)]


def raster_direct(segments, grid, p=PARAMETERS):
    """Interpolate inside each observed adjacent link, not across a support gap."""
    y = np.full(len(grid), np.nan)
    conflict = np.zeros(len(grid), bool)
    for segment in segments:
        pts = np.asarray(segment, float)
        if pts.ndim != 2 or pts.shape[1:] != (2,):
            continue
        for a, b in zip(pts[:-1], pts[1:]):
            if not np.isfinite([a, b]).all() or abs(b[0]-a[0]) < 1e-8:
                continue
            if np.linalg.norm(b-a) > p.maximum_segment_link_px:
                continue
            selected = (grid >= min(a[0], b[0])) & (grid <= max(a[0], b[0]))
            estimate = a[1] + (grid[selected]-a[0]) * (b[1]-a[1])/(b[0]-a[0])
            old = y[selected]
            # Multiple distant branches cannot be silently averaged.
            conflict[selected] |= np.isfinite(old) & (abs(old-estimate) > p.minimum_direct_sigma_px)
            y[selected] = estimate
    y[conflict] = np.nan
    return y


class TemporalBoundary:
    def __init__(self, parameters=PARAMETERS):
        self.p = parameters
        if parameters.grid_step_px <= 0 or parameters.maximum_history_s <= 0:
            raise ValueError('invalid boundary parameters')
        self.reset()

    def reset(self):
        self.grid = self.y = self.sigma = self.seen = None
        self.last_time = self.shape = None

    def _warped(self, grid, matrix):
        """Keep oldest contributing observation and largest error when resampling."""
        out_y, out_sigma, out_seen = (np.full(len(grid), np.nan) for _ in range(3))
        finite = np.isfinite(self.y) & np.isfinite(self.sigma) & np.isfinite(self.seen)
        for run in _runs(np.flatnonzero(finite)):
            if len(run) < 2:
                continue
            xy = np.column_stack([self.grid[run], self.y[run]])
            warped = xy @ matrix[:, :2].T + matrix[:, 2]
            for k in range(len(run)-1):
                a, b = warped[k], warped[k+1]
                if b[0] <= a[0] or b[0]-a[0] > self.p.grid_step_px*1.1:
                    continue
                ids = np.flatnonzero((grid >= a[0]-1e-7) & (grid <= b[0]+1e-7))
                if not len(ids):
                    continue
                vals = a[1]+(grid[ids]-a[0])*(b[1]-a[1])/(b[0]-a[0])
                stamp = min(self.seen[run[k]], self.seen[run[k+1]])
                error = max(self.sigma[run[k]], self.sigma[run[k+1]])
                existing = np.isfinite(out_y[ids])
                if np.any(existing):
                    out_seen[ids[existing]] = np.minimum(out_seen[ids[existing]], stamp)
                    out_sigma[ids[existing]] = np.maximum(out_sigma[ids[existing]], error)
                out_y[ids] = vals
                out_seen[ids[~existing]] = stamp
                out_sigma[ids[~existing]] = error
        return out_y, out_sigma, out_seen

    def update(self, time_s, frame_shape, direct_segments, width_px, scene, *, temporal=True):
        p = self.p
        try:
            time_s, width_px = float(time_s), float(width_px)
        except (ValueError, TypeError, OverflowError):
            time_s, width_px = float('nan'), float('nan')
        if not math.isfinite(time_s) or not math.isfinite(width_px) or width_px <= 0:
            self.reset()
            return {'valid': False, 'reason': 'invalid_time_or_width', 'points': [],
                    'segments': [], 'states': [], 'observed_at_s': [], 'sigma_px': [],
                    'boundary_state': 'unknown', 'registration_valid': False, 'overlap_valid': False,
                    'direct_point_count': 0, 'predicted_point_count': 0, 'oldest_observed_at_s': None,
                    'boundary_max_age_s': None, 'boundary_sigma_px': None,
                    'physical_boundary_confirmed': False}
        shape = tuple(frame_shape[:2])
        grid = np.arange(0., shape[1], p.grid_step_px)
        direct = raster_direct(direct_segments, grid, p)
        valid_direct = np.isfinite(direct)
        sigma0 = max(p.minimum_direct_sigma_px, width_px*p.direct_sigma_width_fraction)
        y, sigma, seen = direct.copy(), np.full(len(grid), sigma0), np.full(len(grid), time_s)
        sigma[~valid_direct] = np.nan; seen[~valid_direct] = np.nan
        states = np.where(valid_direct, 'direct', 'unknown').astype('<U9')
        elapsed = time_s-self.last_time if self.last_time is not None else None
        registration = bool(scene.get('pose_stable') is True and scene.get('history_action') == 'continue')
        overlap_valid = False
        diag = dict(reason='direct_only', overlap_points=0, overlap_span_px=0.,
                    overlap_shift_px=None, overlap_p90_residual_px=None, propagated_points=0)
        can = temporal and self.y is not None and self.shape == shape and registration
        can = can and elapsed is not None and 0 < elapsed <= p.maximum_frame_gap_s+1e-9
        if scene.get('previous_time_s') is not None:
            can = can and abs(float(scene['previous_time_s'])-self.last_time) <= 1e-8
        matrix = np.asarray(scene.get('previous_to_current'), float) if registration else None
        can = can and matrix.shape == (2, 3) and np.isfinite(matrix).all()
        if can:
            prior, error, stamps = self._warped(grid, matrix)
            age = time_s-stamps
            ok = np.isfinite(prior) & (age >= 0) & (age <= p.maximum_history_s+1e-9)
            overlap = ok & valid_direct
            ids = np.flatnonzero(overlap)
            span = float(np.ptp(grid[ids])) if len(ids) else 0.
            diag.update(overlap_points=len(ids), overlap_span_px=span)
            if len(ids) >= p.minimum_overlap_points and span >= p.minimum_overlap_span_widths*width_px:
                delta = direct[ids]-prior[ids]
                shift = float(np.median(delta))
                residual = float(np.percentile(abs(delta-shift), 90))
                overlap_valid = abs(shift) <= p.maximum_overlap_shift_widths*width_px and residual <= p.maximum_overlap_p90_widths*width_px
                diag.update(overlap_shift_px=shift, overlap_p90_residual_px=residual,
                            reason='overlap_consistent' if overlap_valid else 'overlap_disagrees_reset_history')
                if overlap_valid:
                    # Vertical deformation is a model constrained by current
                    # overlap. It does not refresh the hidden point's age.
                    registration_error = float(scene.get('reprojection_p90_px') or 0.)
                    small_width = scene.get('parameters', {}).get('maximum_width', shape[1])
                    registration_error /= min(1., small_width/shape[1])
                    error = np.sqrt(error**2 + residual**2 + registration_error**2) + elapsed*width_px*p.sigma_growth_widths_per_second
                    take = ok & ~valid_direct & (error <= p.maximum_sigma_width_fraction*width_px)
                    y[take] = prior[take]+shift; sigma[take] = error[take]; seen[take] = stamps[take]
                    states[take] = 'predicted'
                    diag['propagated_points'] = int(take.sum())
            else:
                diag['reason'] = 'insufficient_current_overlap_reset_history'
        elif temporal:
            diag['reason'] = 'no_valid_pose_or_time_history'
        self.grid, self.y, self.sigma, self.seen = grid, y, sigma, seen
        self.last_time, self.shape = time_s, shape
        valid = np.isfinite(y)
        ids = np.flatnonzero(valid)
        segments = [np.column_stack([grid[r], y[r]]).tolist() for r in _runs(ids) if len(r) >= 2]
        all_points = np.column_stack([grid[ids], y[ids]]).tolist()
        state = 'predicted' if np.any(states == 'predicted') else ('direct' if len(ids) else 'unknown')
        return dict(valid=bool(segments), reason=diag['reason'], points=all_points, segments=segments,
            states=states[ids].tolist(), observed_at_s=seen[ids].tolist(), sigma_px=sigma[ids].tolist(),
            boundary_state=state, registration_valid=registration, overlap_valid=overlap_valid,
            direct_point_count=int(valid_direct.sum()), predicted_point_count=int(np.sum(states=='predicted')),
            oldest_observed_at_s=float(np.min(seen[ids])) if len(ids) else None,
            boundary_max_age_s=float(np.max(time_s-seen[ids])) if len(ids) else None,
            boundary_sigma_px=float(np.max(sigma[ids])) if len(ids) else None,
            diagnostics=diag, parameters=asdict(p), physical_boundary_confirmed=False)


def boundary_path_features(decoded, boundary, geometry):
    """Reuse unchanged geometric gates; every use of prior support is labelled."""
    from .v16_video_measurement import path_gap_features
    common = dict(boundary_state='unknown', boundary_max_age_s=None, boundary_sigma_px=None,
                  oldest_observed_at_s=None, registration_valid=boundary.get('registration_valid', False),
                  overlap_valid=boundary.get('overlap_valid', False), path_visibility_assessed=False,
                  path_identity_confirmed=False, physical_event_confirmed=False, force_command=None)
    if geometry is None:
        return dict(common, usable=False, image_gap_proxy=None, reason='no_geometry')
    f = path_gap_features(decoded, {'local_boundary_segments': boundary['segments']}, geometry)
    f.update(common)
    f['boundary_source_counts'] = {name: boundary.get('states', []).count(name)
                                  for name in ('direct', 'predicted', 'unknown')}
    if not f['usable'] or not boundary['points']:
        return f
    # Conservatively attribute to every supplied point. This may overstate
    # uncertainty but can never label a mixed/prior curve as wholly observed.
    f.update(boundary_state=boundary['boundary_state'], boundary_max_age_s=boundary['boundary_max_age_s'],
        boundary_sigma_px=boundary['boundary_sigma_px'], oldest_observed_at_s=boundary['oldest_observed_at_s'],
        reason='predicted_boundary_gap_candidate' if boundary['boundary_state']=='predicted' else 'direct_grid_gap_candidate')
    uncertainty = boundary['boundary_sigma_px']/float(geometry['width_px'])
    gap = f['image_gap_proxy']
    f.update(gap_proxy_interval=[gap-uncertainty, gap+uncertainty],
             interval_is_calibrated_confidence=False, uncertainty_attribution='conservative_all_boundary_support')
    return f
