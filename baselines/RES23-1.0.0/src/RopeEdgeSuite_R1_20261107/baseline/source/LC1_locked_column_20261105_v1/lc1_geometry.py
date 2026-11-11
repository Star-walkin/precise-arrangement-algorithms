"""Causal locked-column geometry, independent of image segmentation.

Coordinates are full-frame pixels after the caller's documented normalization.
The apparent C1 transition is an image measurement, not proof of 3-D contact.
All thresholds below are explicit development defaults, not validated accuracy.
Only NumPy and the Python standard library are required.
"""

from collections import deque
import math
import numpy as np


def normalize_line(abc):
    """Return unit-normal [a,b,c], with b>=0 (a>=0 when b==0).

    Invalid/degenerate coefficients raise ValueError. Equivalent scaled lines,
    including negative scale, have the same canonical representation.
    """
    v = np.asarray(abc, dtype=float).reshape(-1)
    if v.size != 3 or not np.all(np.isfinite(v)):
        raise ValueError("line requires three finite coefficients")
    norm = float(np.hypot(v[0], v[1]))
    if norm < 1e-12:
        raise ValueError("line normal is degenerate")
    v = v / norm
    if v[1] < 0 or (v[1] == 0 and v[0] < 0):
        v = -v
    return v


def _points(points):
    a = np.asarray(points, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("points must have shape (N,2)")
    return a[np.all(np.isfinite(a), axis=1)]


def _angle(abc):
    return float(np.degrees(np.arctan2(-abc[0], abc[1])))


def _angle_delta(a, b):
    return float((a - b + 90.0) % 180.0 - 90.0)


def _mad(a):
    a = np.asarray(a, dtype=float)
    return float(1.4826 * np.median(np.abs(a - np.median(a))))


def _weighted_tls(p, weights):
    center = np.average(p, axis=0, weights=weights)
    z = p - center
    cov = (z * weights[:, None]).T @ z / np.sum(weights)
    values, vectors = np.linalg.eigh(cov)
    normal = vectors[:, 0]
    if values[-1] < 1e-12:
        raise ValueError("points have no spatial extent")
    return normalize_line([normal[0], normal[1], -normal @ center])


def fit_line(points):
    """Robust total-least-squares line; rmse is perpendicular inlier RMSE.

    This fits the supplied FREE segment. A whole curved path must first be
    separated by the caller; robust fitting cannot establish material identity.
    """
    try:
        p = _points(points)
        if len(p) < 2:
            raise ValueError("fewer than two finite points")
        weights = np.ones(len(p))
        abc = _weighted_tls(p, weights)
        for _ in range(10):
            residual = p @ abc[:2] + abc[2]
            scale = max(_mad(residual), 0.25)
            weights = np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(residual), 1e-12))
            new = _weighted_tls(p, weights)
            if np.max(np.abs(new - abc)) < 1e-8:
                abc = new
                break
            abc = new
        residual = p @ abc[:2] + abc[2]
        inliers = np.abs(residual) <= max(1.0, 3.0 * _mad(residual))
        if np.count_nonzero(inliers) < 2:
            raise ValueError("insufficient line support")
        rmse = float(np.sqrt(np.mean(residual[inliers] ** 2)))
        return dict(valid=True, abc=abc, angle_deg=_angle(abc), rmse=rmse,
                    total_rmse=float(np.sqrt(np.mean(residual ** 2))),
                    inlier_count=int(inliers.sum()), point_count=len(p),
                    inlier_ratio=float(np.mean(inliers)), reason="ok")
    except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
        return dict(valid=False, abc=None, angle_deg=None, rmse=None, reason=str(exc))


def _robust_regression(design, y):
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(6):
        residual = y - design @ beta
        scale = max(_mad(residual), 0.30)
        weights = np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(residual), 1e-12))
        root = np.sqrt(weights)
        new = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)[0]
        if np.max(np.abs(new - beta)) < 1e-8:
            beta = new
            break
        beta = new
    residual = y - design @ beta
    # A small number of trace outliers cannot decide the change point.
    sorted_squared = np.sort(residual ** 2)
    score = float(np.mean(sorted_squared[:max(3, int(0.90 * len(y)))]))
    return beta, residual, score


def estimate_turn(points, free_side="right", pitch_px=20):
    """Fit y=m*x+b+q*hinge(x-d)^2, with C1 continuity at x=d.

    free_side='right': hinge=min(x-d,0); free_side='left': max(x-d,0).
    Requires a single-valued y(x) centerline with observed curved AND free
    branches. Pure lines and poorly localized smooth transitions are rejected.
    uncertainty_px is a conservative profile/linearized fit diagnostic, not a
    calibrated coverage guarantee. No geometry is learned from rejected paths.
    """
    def rejected(reason, **extra):
        result = dict(valid=False, x=None, y=None, quality=0.0, rmse=None,
                      uncertainty_px=None, reason=reason)
        result.update(extra)
        return result

    if free_side not in ("right", "left"):
        raise ValueError("free_side must be 'right' or 'left'")
    if not np.isfinite(pitch_px) or pitch_px <= 0:
        return rejected("invalid_pitch")
    try:
        p = _points(points)
    except (ValueError, TypeError):
        return rejected("invalid_points")
    if len(p) < 24:
        return rejected("insufficient_points")
    p = p[np.argsort(p[:, 0], kind="stable")]
    if len(p) > 600:
        p = p[np.linspace(0, len(p) - 1, 600).astype(int)]
    x, y = p.T
    span = float(np.ptp(x))
    if span < max(40.0, 4.0 * pitch_px):
        return rejected("insufficient_x_span")
    center = float((x[0] + x[-1]) / 2)
    scale = span / 2
    z = (x - center) / scale
    base_design = np.column_stack([np.ones(len(x)), z])
    _, base_residual, base_score = _robust_regression(base_design, y)
    if math.sqrt(base_score) < max(0.8, 0.06 * pitch_px):
        return rejected("straight_path", straight_rmse=math.sqrt(base_score))

    margin = max(12.0, 0.8 * pitch_px)
    lo, hi = float(x[0] + margin), float(x[-1] - margin)
    if lo >= hi:
        return rejected("insufficient_branch_span")

    def fit_candidate(d):
        dz = (d - center) / scale
        h = np.minimum(z - dz, 0) if free_side == "right" else np.maximum(z - dz, 0)
        design = np.column_stack([np.ones(len(x)), z, h * h])
        beta, residual, score = _robust_regression(design, y)
        return dict(d=float(d), beta=beta, residual=residual, score=score, h=h,
                    design=design)

    candidates = [fit_candidate(d) for d in np.linspace(lo, hi, 81)]
    best = min(candidates, key=lambda r: r["score"])
    coarse_step = (hi - lo) / 80
    refine_lo, refine_hi = max(lo, best["d"] - coarse_step), min(hi, best["d"] + coarse_step)
    candidates.extend(fit_candidate(d) for d in np.linspace(refine_lo, refine_hi, 25))
    best = min(candidates, key=lambda r: r["score"])
    d, beta, residual = best["d"], best["beta"], best["residual"]
    free = x >= d if free_side == "right" else x <= d
    curved = ~free
    if free.sum() < 8 or curved.sum() < 8:
        return rejected("insufficient_branch_support")
    free_span, curve_span = float(np.ptp(x[free])), float(np.ptp(x[curved]))
    if min(free_span, curve_span) < margin * 0.85:
        return rejected("insufficient_branch_span")
    fit_gate = max(1.5, 0.15 * pitch_px)
    inliers = np.abs(residual) <= max(fit_gate, 3 * _mad(residual))
    ratio = float(inliers.mean())
    rmse = float(np.sqrt(np.mean(residual[inliers] ** 2)))
    reduction = 1.0 - best["score"] / max(base_score, 1e-12)
    m = float(beta[1] / scale)
    intercept = float(beta[0] - m * center)
    q = float(beta[2] / scale ** 2)
    bend = abs(q) * curve_span ** 2
    details = dict(x=d, y=m * d + intercept, rmse=rmse, slope=m,
                   intercept=intercept, quadratic=q, inlier_ratio=ratio,
                   score_reduction=float(reduction), curved_deviation_px=bend,
                   free_span_px=free_span, curve_span_px=curve_span,
                   free_side=free_side, point_count=len(p),
                   free_abc=normalize_line([-m, 1, -intercept]))
    if abs(m) > math.tan(math.radians(80)):
        return rejected("steep_y_of_x_model", **details)
    if bend < max(2.0, 0.20 * pitch_px, 3.0 * _mad(residual)) or reduction < 0.25:
        return rejected("straight_or_unresolved_curvature", **details)
    if ratio < 0.75 or rmse > fit_gate:
        return rejected("poor_model_fit", **details)

    # Profile width reflects ambiguity after refitting the other parameters.
    tolerance = max(0.25, rmse ** 2 / math.sqrt(len(x)))
    acceptable = [r["d"] for r in candidates if r["score"] <= best["score"] + tolerance]
    profile_width = max(abs(d - min(acceptable)), abs(max(acceptable) - d))
    # Marginal covariance includes slope/intercept/curvature coupling to d.
    derivative_dz = -2.0 * beta[2] * best["h"]
    jac = np.column_stack([best["design"], derivative_dz])[inliers]
    cov = max(rmse, 0.35) ** 2 * np.linalg.pinv(jac.T @ jac)
    stderr_d = scale * math.sqrt(max(float(cov[-1, -1]), 0.0))
    uncertainty = max(coarse_step / 12, profile_width, 2.5 * stderr_d)
    details["uncertainty_px"] = float(uncertainty)
    if uncertainty > 0.50 * pitch_px:
        return rejected("turn_localization_uncertain", **details)
    quality = ratio * max(0.0, 1 - rmse / (2 * fit_gate))
    quality *= max(0.0, 1 - uncertainty / (1.5 * pitch_px)) * min(1.0, reduction / 0.5)
    return dict(valid=True, quality=float(np.clip(quality, 0, 1)), reason="ok", **details)


class ColumnLock:
    """Causal per-camera/per-layer/per-direction locked-column state machine.

    Calibration requires complete supported paths. After locking, current S1
    lines alone may drive counts for max_unverified_lock_age_s; missing turn
    measurements are explicitly labelled and never move d. This does NOT gate
    F3 image quality, which the caller must independently check.
    External layer/camera changes must call reset=True. Reversal detection here
    is apparent axial motion, not encoder-confirmed physical direction.
    When current supported line points are provided, C is refit at the frozen
    unit normal. The resulting intersection must agree with the independently
    measured raw line at d within 0.4 pitch (an explicit development default).
    """

    def __init__(self, min_frames=8, min_calibration_seconds=1.0,
                 min_sample_dt=0.08, min_turn_quality=0.40,
                 max_angle_mad_deg=0.75, max_angle_spread_deg=2.0,
                 max_angle_trend_deg=1.0, max_d_mad_pitch=0.20,
                 max_d_spread_pitch=0.60, max_d_trend_pitch=0.30,
                 tracking_angle_limit_deg=2.0, tracking_d_limit_pitch=0.40,
                 mismatch_frames=3, max_line_gap_s=1.0,
                 max_unverified_lock_age_s=15.0,
                 min_reverse_excursion_pitch=0.35, reverse_frames=3,
                 tracking_intersection_limit_pitch=0.40):
        if min_frames < 8 or min_calibration_seconds < 1.0:
            raise ValueError("calibration requires at least 8 frames and 1.0 second")
        if min_sample_dt <= 0 or max_unverified_lock_age_s <= 0 or mismatch_frames < 1:
            raise ValueError("invalid lock settings")
        if tracking_intersection_limit_pitch <= 0:
            raise ValueError("intersection limit must be positive")
        self.config = dict(locals())
        self.config.pop("self")
        self.epoch = -1
        self._clear("initializing")

    def _clear(self, state):
        self.epoch += 1
        self.state = state
        self.samples = deque(maxlen=240)
        self.d = self.angle = self.lock_time = None
        self.last_t = self.last_line_t = self.last_audit_t = None
        self.pitch = self.last_y = self.last_y_t = self.extreme_y = None
        self.direction = 0
        self.motion_anchor_y = None
        self.motion_votes = self.reverse_votes = 0
        self.angle_bad = self.d_bad = self.position_bad = 0
        self.intersection_bad = 0
        self._latest_turn_supported = False
        self._current_line_points = None
        self._current_points_error = False

    def _frozen_model(self, raw):
        if raw is None or self.d is None or self.angle is None:
            return None, "not_available", None, None
        if self._current_points_error:
            return None, "invalid_current_line_points", None, None
        y = self._line_y(raw, self.d)
        phi = math.radians(self.angle)
        a, b = -math.sin(phi), math.cos(phi)
        if self._current_line_points is not None:
            projections = self._current_line_points @ np.array([a, b])
            c = -float(np.median(projections))
            fit_rmse = float(np.sqrt(np.mean((projections + c) ** 2)))
            basis = "robust_current_measured_points_fixed_unit_normal"
        else:
            c = -a * self.d - b * y
            fit_rmse = None
            basis = "raw_line_at_locked_column_no_source_points_available"
        model = normalize_line([a, b, c])
        delta = self._line_y(model, self.d) - y
        return model, basis, float(delta), fit_rmse

    def _intersection_gate(self, raw, turn):
        _, _, delta, _ = self._frozen_model(raw)
        if delta is None:
            return None
        limit = self.config["tracking_intersection_limit_pitch"] * self.pitch
        self.intersection_bad = self.intersection_bad + 1 if abs(delta) > limit else 0
        if self.intersection_bad >= self.config["mismatch_frames"]:
            return self._invalidate("persistent_frozen_intersection_drift", raw, turn, "assumption_failed")
        if self.intersection_bad:
            self.state = "degraded"
            return self._output("frozen_intersection_mismatch", raw, turn)
        return None

    def _output(self, reason, raw=None, turn=None, count_allowed=False):
        raw_angle = _angle(raw) if raw is not None else None
        drift = _angle_delta(raw_angle, self.angle) if raw is not None and self.angle is not None else None
        d_drift = None
        if self.d is not None and self._usable_turn(turn, require_complete=False):
            d_drift = float(turn["x"] - self.d)
        locked, basis, intersection_drift, fit_rmse = self._frozen_model(raw)
        age = None if self.last_audit_t is None or self.last_t is None else self.last_t - self.last_audit_t
        return dict(state=self.state, d=self.d, locked_abc=locked,
                    raw_angle_deg=raw_angle, angle_drift_deg=drift,
                    d_drift_px=d_drift, lock_time=self.lock_time, reason=reason,
                    count_allowed=bool(count_allowed), epoch=self.epoch,
                    calibration_frames=len(self.samples),
                    audit_missing=not self._latest_turn_supported,
                    unverified_age_s=age, apparent_direction=self.direction,
                    slope_frozen=self.angle is not None,
                    C_fit_basis=basis,
                    frozen_C_fit_point_count=0 if self._current_line_points is None else len(self._current_line_points),
                    frozen_C_fit_rmse_px=fit_rmse,
                    frozen_vs_raw_intersection_drift_px=intersection_drift,
                    frozen_vs_raw_intersection_drift_pitch=(intersection_drift / self.pitch
                        if intersection_drift is not None and self.pitch is not None else None),
                    frozen_intersection_gate_limit_px=(self.config["tracking_intersection_limit_pitch"] * self.pitch
                        if self.pitch is not None else None),
                    frozen_intersection_mismatch_frames=self.intersection_bad)

    @staticmethod
    def _line_y(raw, d):
        if abs(raw[1]) < 0.10:
            return None
        return float(-(raw[0] * d + raw[2]) / raw[1])

    def _usable_turn(self, turn, require_complete=True, complete=False):
        if not isinstance(turn, dict) or not turn.get("valid", False):
            return False
        try:
            uncertainty = float(turn.get("uncertainty_px", float("inf")))
            good = (np.isfinite(turn["x"]) and np.isfinite(uncertainty) and
                    float(turn.get("quality", 0)) >= self.config["min_turn_quality"] and
                    uncertainty >= 0)
        except (TypeError, ValueError, KeyError):
            return False
        return bool(good and (complete or not require_complete))

    def _invalidate(self, reason, raw, turn, state="recalibrating"):
        old = self._output(reason, raw, turn)
        previous_d, previous_angle = self.d, self.angle
        self._clear(state)
        result = self._output(reason, raw, turn)
        # Preserve the independent evidence on the exact frame that invalidates
        # a lock, while clearing d/locked_abc for all downstream calculations.
        result.update(angle_drift_deg=old["angle_drift_deg"],
                      d_drift_px=old["d_drift_px"],
                      frozen_vs_raw_intersection_drift_px=old["frozen_vs_raw_intersection_drift_px"],
                      frozen_vs_raw_intersection_drift_pitch=old["frozen_vs_raw_intersection_drift_pitch"],
                      frozen_intersection_gate_limit_px=old["frozen_intersection_gate_limit_px"],
                      frozen_intersection_mismatch_frames=old["frozen_intersection_mismatch_frames"],
                      C_fit_basis=old["C_fit_basis"],
                      frozen_C_fit_point_count=old["frozen_C_fit_point_count"],
                      frozen_C_fit_rmse_px=old["frozen_C_fit_rmse_px"],
                      invalidated_d=previous_d, invalidated_angle_deg=previous_angle)
        return result

    def update(self, t, line_abc, turn, pitch_px, complete_path_supported=False,
               reset=False, line_points_xy=None):
        """Process one timestamp once; optionally refit C from current S1 points.

        line_points_xy must contain observed center points on the CURRENT free
        straight segment, in the same full-frame coordinates as line_abc. Empty
        or absent points keep the labelled raw-at-d approximation. Supplied
        malformed/nonfinite/single-point evidence cannot authorize a count.
        """
        if reset:
            self._clear("calibrating")
        self._latest_turn_supported = False
        self._current_line_points = None
        self._current_points_error = False
        if line_points_xy is not None:
            try:
                points = np.asarray(line_points_xy, dtype=float)
                if points.size:
                    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or
                            not np.isfinite(points).all() or np.max(np.ptp(points, axis=0)) < 1e-6):
                        raise ValueError("invalid current supported line points")
                    self._current_line_points = points
            except (TypeError, ValueError):
                self._current_points_error = True
        try:
            t = float(t)
            pitch_px = float(pitch_px)
            if not np.isfinite(t) or not np.isfinite(pitch_px) or pitch_px <= 0:
                raise ValueError("invalid timestamp or pitch")
        except (TypeError, ValueError):
            return self._output("invalid_timestamp_or_pitch")
        if self.last_t is not None and t <= self.last_t:
            return self._invalidate("nonmonotonic_timestamp", None, turn)
        self.last_t = t
        try:
            raw = normalize_line(line_abc)
            if abs(raw[1]) < 0.10:
                raise ValueError("near_vertical_line")
        except (ValueError, TypeError):
            raw = None
        full_turn = self._usable_turn(turn, complete=complete_path_supported)
        self._latest_turn_supported = full_turn

        if self.d is None:
            self.state = "calibrating"
            if self.samples and t - self.samples[-1][0] > self.config["max_line_gap_s"]:
                self.samples.clear()
            if raw is None or not full_turn or self._current_points_error:
                return self._output("calibration_requires_line_and_complete_turn", raw, turn)
            if self.samples and t - self.samples[-1][0] + 1e-9 < self.config["min_sample_dt"]:
                return self._output("calibration_sample_too_close", raw, turn)
            if turn["uncertainty_px"] > 0.50 * pitch_px:
                return self._output("calibration_turn_uncertain", raw, turn)
            self.samples.append((t, raw.copy(), float(turn["x"]), pitch_px))
            duration = self.samples[-1][0] - self.samples[0][0]
            if len(self.samples) < self.config["min_frames"] or duration + 1e-9 < self.config["min_calibration_seconds"]:
                return self._output("calibrating", raw, turn)
            times = np.array([r[0] for r in self.samples])
            times -= times[0]
            reference_angle = _angle(self.samples[0][1])
            angles = np.array([reference_angle + _angle_delta(_angle(r[1]), reference_angle) for r in self.samples])
            ds = np.array([r[2] for r in self.samples])
            pitch = float(np.median([r[3] for r in self.samples]))
            angle_trend = abs(float(np.polyfit(times, angles, 1)[0]) * duration)
            d_trend = abs(float(np.polyfit(times, ds, 1)[0]) * duration)
            if (_mad(angles) > self.config["max_angle_mad_deg"] or
                    np.ptp(np.percentile(angles, [10, 90])) > self.config["max_angle_spread_deg"] or
                    angle_trend > self.config["max_angle_trend_deg"]):
                return self._invalidate("constant_angle_assumption_failed", raw, turn, "assumption_failed")
            if (_mad(ds) > self.config["max_d_mad_pitch"] * pitch or
                    np.ptp(np.percentile(ds, [10, 90])) > self.config["max_d_spread_pitch"] * pitch or
                    d_trend > self.config["max_d_trend_pitch"] * pitch):
                return self._invalidate("constant_column_assumption_failed", raw, turn, "assumption_failed")
            self.d, self.angle, self.pitch = float(np.median(ds)), float(np.median(angles)), pitch
            self.lock_time = self.last_line_t = self.last_audit_t = t
            ys = np.array([self._line_y(r[1], self.d) for r in self.samples])
            motion = float(np.polyfit(times, ys, 1)[0])
            if abs(ys[-1] - ys[0]) >= 0.40 * pitch and abs(motion) >= 0.05 * pitch:
                self.direction = 1 if motion > 0 else -1
            self.last_y = self.extreme_y = self.motion_anchor_y = float(ys[-1])
            self.last_y_t = t
            self.state = "locked"
            gated = self._intersection_gate(raw, turn)
            if gated is not None:
                return gated
            return self._output("locked", raw, turn, True)

        if raw is None:
            self.intersection_bad = 0
            self.state = "degraded"
            if t - self.last_line_t > self.config["max_line_gap_s"]:
                return self._invalidate("prolonged_line_missing", raw, turn)
            return self._output("current_line_invalid", raw, turn)

        if self._current_points_error:
            self.intersection_bad = 0
            self.state = "degraded"
            if t - self.last_line_t > self.config["max_line_gap_s"]:
                return self._invalidate("prolonged_line_points_invalid", raw, turn)
            return self._output("current_line_points_invalid", raw, turn)

        angle_drift = abs(_angle_delta(_angle(raw), self.angle))
        self.angle_bad = self.angle_bad + 1 if angle_drift > self.config["tracking_angle_limit_deg"] else 0
        d_mismatch = full_turn and abs(float(turn["x"]) - self.d) > max(2.0, self.config["tracking_d_limit_pitch"] * self.pitch)
        self.d_bad = self.d_bad + 1 if d_mismatch else 0
        if self.angle_bad >= self.config["mismatch_frames"]:
            return self._invalidate("persistent_angle_drift", raw, turn, "assumption_failed")
        if self.d_bad >= self.config["mismatch_frames"]:
            return self._invalidate("persistent_column_drift", raw, turn, "assumption_failed")
        if self.angle_bad or self.d_bad:
            self.intersection_bad = 0
            self.state = "degraded"
            return self._output("angle_mismatch" if self.angle_bad else "column_mismatch", raw, turn)

        gated = self._intersection_gate(raw, turn)
        if gated is not None:
            return gated

        y = self._line_y(raw, self.d)
        dt = t - self.last_y_t
        jump_limit = max(4.0 * self.pitch, 10.0 * self.pitch * dt)
        jump = abs(y - self.last_y) > jump_limit
        self.position_bad = self.position_bad + 1 if jump else 0
        if self.position_bad >= self.config["mismatch_frames"]:
            return self._invalidate("persistent_line_position_jump", raw, turn)
        if jump:
            self.state = "degraded"
            return self._output("line_position_jump", raw, turn)

        velocity = (y - self.last_y) / dt
        moving_sign = 0 if abs(velocity) < 0.05 * self.pitch else (1 if velocity > 0 else -1)
        if self.direction == 0:
            excursion = y - self.motion_anchor_y
            if abs(excursion) >= 0.40 * self.pitch and moving_sign == (1 if excursion > 0 else -1):
                self.motion_votes += 1
                if self.motion_votes >= self.config["reverse_frames"]:
                    self.direction = moving_sign
                    self.extreme_y = y
            else:
                self.motion_votes = 0
        else:
            self.extreme_y = max(self.extreme_y, y) if self.direction > 0 else min(self.extreme_y, y)
            opposite_excursion = (self.extreme_y - y) * self.direction
            if moving_sign == -self.direction and opposite_excursion >= self.config["min_reverse_excursion_pitch"] * self.pitch:
                self.reverse_votes += 1
            else:
                self.reverse_votes = 0
            if self.reverse_votes >= self.config["reverse_frames"]:
                return self._invalidate("apparent_direction_reversal", raw, turn)
        self.last_y, self.last_y_t, self.last_line_t = y, t, t
        if full_turn:
            self.last_audit_t = t  # Audit refresh NEVER re-estimates the locked column.
        age = t - self.last_audit_t
        if age > self.config["max_unverified_lock_age_s"]:
            self.state = "degraded"
            return self._output("turn_audit_expired", raw, turn)
        self.state = "locked"
        result = self._output("locked" if full_turn else "locked_turn_audit_missing", raw, turn, True)
        result["audit_missing"] = not full_turn
        result["measurement_basis"] = "current_line_and_frozen_column"
        return result


def count_crossings(boundary_y, edge_y, active_y, pitch_px):
    """Count OBSERVED groove/row-center crossings strictly between endpoints.

    boundary_y is the caller's detected crossing coordinates, not inferred
    evenly spaced lines. Their phase convention must be specified externally.
    continuous_pitch_distance deliberately has NO automatic half-pitch offset.
    Suspicious gaps are flagged, never filled and counted as observations.
    """
    try:
        edge, active, pitch = float(edge_y), float(active_y), float(pitch_px)
        crossings = np.asarray(boundary_y, dtype=float).reshape(-1)
        if not np.all(np.isfinite([edge, active, pitch])) or pitch <= 0:
            raise ValueError("invalid endpoints or pitch")
    except (ValueError, TypeError):
        return dict(valid=False, observed_grooves=0, continuous_pitch_distance=None,
                    reason="invalid_input", inferred_grooves=0)
    lo, hi = sorted((edge, active))
    finite = crossings[np.isfinite(crossings)]
    selected = np.unique(finite[(finite > lo + 1e-6) & (finite < hi - 1e-6)])
    distances = np.diff(selected)
    ratios = distances / pitch
    reasons = []
    if np.any(ratios < 0.65):
        reasons.append("too_close_or_harmonic_crossings")
    if np.any(ratios > 1.50):
        reasons.append("missing_or_irregular_crossings")
    continuous = abs(active - edge) / pitch
    if abs(len(selected) - continuous) > 1.50:
        reasons.append("observed_count_distance_inconsistent")
    if len(finite) != len(crossings):
        reasons.append("nonfinite_crossings")
    return dict(valid=not reasons, observed_grooves=int(len(selected)),
                observed_y=selected, continuous_pitch_distance=float(continuous),
                inferred_grooves=0, phase_offset_applied=0.0,
                spacing_in_pitch=ratios, reason="ok" if not reasons else ";".join(reasons),
                quality_reasons=reasons)
