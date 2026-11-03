"""V26 route A: offline-only causal signal-change candidates, never t1 proof.

Inference accepts measurements, NOT annotations, images, files or device handles.
The image angle is an unoriented line angle modulo pi, not a 3-D lag angle.
Angle and x_ref come from one fit: their likelihoods are never multiplied.

BOCPD follows Adams & MacKay (2007), Eq. 3 / Algorithm 1:
https://arxiv.org/pdf/0710.3742 . r=0 denotes a boundary AFTER the current
observation, with empty sufficient statistics for the next prediction. The
growth states incorporate the current observation. With a hazard constant
across run hypotheses, P(r=0 | observations) == H(dt); this is NOT an alarm.
The alarm uses posterior mass of positive, <=1-second runs, excluding r=0.
The reported boundary interval is conditional on that short-run subset, NOT
a calibrated interval for physical t1. No run-length truncation is performed.
This standard segmentation model has no contamination/outlier component: an
isolated large outlier can seed a short-run candidate even with persistence.
That synthetic failure is retained, not hidden by a tuned rejection gate.

All transformations use previous observations plus the current measurement.
No interpolation, full-record normalization, label-based prior or held-value
observation is used. Gaps > .65 s restart all affected channel state and warmup.
CUSUM is in observation-sum units; its cadence must be frozen by the caller.
"""
from __future__ import annotations

from collections import Counter, deque
import copy
import math

import numpy as np


DEFAULT_CONFIG = {
    "schema": "v26_causal_signals_1",
    "warmup_s": 5.0,
    "warmup_points": 20,
    "residual_initial_run_guard": "residual_span_must_exceed_short_run_window",
    "trend_window_s": 2.0,
    "trend_minimum_points": 6,
    "trend_estimator": "past_only_theil_sen_median_intercept",
    "scale_window_s": 5.0,
    "angle_scale_floor_rad": math.radians(.1),
    "x_scale_floor_px": 1.0,
    "maximum_missing_s": .65,
    "cusum_allowance": .5,
    "cusum_threshold": 5.0,
    "cusum_clip_z": 6.0,
    "bocpd_mean_change_s": 30.0,
    "bocpd_prior_mu": 0.0,
    "bocpd_prior_kappa": 1.0,
    "bocpd_prior_alpha": 2.0,
    "bocpd_prior_beta": 1.0,
    "bocpd_short_run_s": 1.0,
    "bocpd_short_mass_threshold": .5,
    "bocpd_outlier_policy": "no_contamination_model_single_outlier_can_trigger",
    "persistence_points": 3,
    "persistence_s": .3,
    "cooldown_s": 5.0,
    "cooldown_scope": "per_channel_union_may_have_nearby_correlated_candidates",
    "channel_combination": "union_not_independent_likelihood_product",
    "candidate_only": True,
    "deployment_permitted": False,
}
CHANNELS = ("angle_proxy_rad", "x_ref_px")
QUALITY_FIELDS = (
    "width_px", "free_span_length_px", "fit_residual_px", "inlier_ratio",
    "saturation", "contrast", "motion_status", "motion_source",
    "timestamp_basis", "measurement_source", "quality_reason",
)


def _finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def unwrap_mod_pi(previous_unwrapped, angle_rad):
    """Nearest modulo-pi increment; no physical rotation/direction claim."""
    angle = float(angle_rad)
    if not math.isfinite(angle):
        raise ValueError("angle must be finite")
    if previous_unwrapped is None:
        return angle
    delta = angle - float(previous_unwrapped)
    return float(previous_unwrapped) + .5 * math.atan2(math.sin(2 * delta),
                                                    math.cos(2 * delta))


def _logsumexp(values):
    maximum = float(np.max(values))
    return maximum + math.log(float(np.exp(values - maximum).sum()))


class StudentTBOCPD:
    """Univariate Normal-Inverse-Gamma BOCPD; exact, untruncated posterior.

    dt is elapsed measurement time, not wall-clock compute time. First update
    uses H=0 because no elapsed interval is known. Missing measurements must
    not call update; the owning channel resets after a long observation gap.
    """

    def __init__(self):
        cfg = DEFAULT_CONFIG
        self.mu = np.array([cfg["bocpd_prior_mu"]], dtype=np.float64)
        self.kappa = np.array([cfg["bocpd_prior_kappa"]], dtype=np.float64)
        self.alpha = np.array([cfg["bocpd_prior_alpha"]], dtype=np.float64)
        self.beta = np.array([cfg["bocpd_prior_beta"]], dtype=np.float64)
        self.log_posterior = np.array([0.0], dtype=np.float64)
        self.boundary_times = None
        self.last_time = None

    @property
    def posterior(self):
        return np.exp(self.log_posterior).copy()

    def update(self, value, time_s):
        value, time_s = float(value), float(time_s)
        if not math.isfinite(value) or not math.isfinite(time_s):
            raise ValueError("BOCPD requires finite fresh observations")
        dt = 0.0 if self.last_time is None else time_s - self.last_time
        if self.last_time is not None and dt <= 0:
            raise ValueError("BOCPD timestamps must strictly increase")
        if dt > DEFAULT_CONFIG["maximum_missing_s"] + 1e-9:
            raise ValueError("reset BOCPD before a long observation gap")
        if self.boundary_times is None:
            self.boundary_times = np.array([time_s], dtype=np.float64)
        hazard = -math.expm1(-dt / DEFAULT_CONFIG["bocpd_mean_change_s"])
        df = 2.0 * self.alpha
        scale2 = self.beta * (self.kappa + 1) / (self.alpha * self.kappa)
        gamma_ratio = np.array([math.lgamma((d + 1) / 2) - math.lgamma(d / 2)
                                for d in df])
        log_predictive = (gamma_ratio - .5 * np.log(df * math.pi * scale2)
                          - .5 * (df + 1) * np.log1p(
                              (value - self.mu) ** 2 / (df * scale2)))
        log_joint = self.log_posterior + log_predictive
        log_evidence = _logsumexp(log_joint)
        # Algebraically identical to Algorithm 1; preserves P0=H exactly up to
        # floating-point roundoff. It cannot contain evidence of a change.
        log_zero = math.log(hazard) if hazard > 0 else -math.inf
        self.log_posterior = np.concatenate((
            [log_zero], log_joint - log_evidence + math.log1p(-hazard)))
        cfg = DEFAULT_CONFIG
        new_kappa = self.kappa + 1
        new_mu = (self.kappa * self.mu + value) / new_kappa
        new_beta = self.beta + self.kappa * (value - self.mu) ** 2 / (2 * new_kappa)
        self.mu = np.concatenate(([cfg["bocpd_prior_mu"]], new_mu))
        self.kappa = np.concatenate(([cfg["bocpd_prior_kappa"]], new_kappa))
        self.alpha = np.concatenate(([cfg["bocpd_prior_alpha"]], self.alpha + .5))
        self.beta = np.concatenate(([cfg["bocpd_prior_beta"]], new_beta))
        self.boundary_times = np.concatenate(([time_s], self.boundary_times))
        self.last_time = time_s
        posterior = self.posterior
        short = ((time_s - self.boundary_times <= cfg["bocpd_short_run_s"] + 1e-9)
                 & (np.arange(len(posterior)) > 0))
        mass = float(posterior[short].sum())
        interval = None
        if mass > 1e-15:
            times, weights = self.boundary_times[short], posterior[short]
            order = np.argsort(times, kind="stable")
            cumulative = np.cumsum(weights[order]) / mass
            low = min(int(np.searchsorted(cumulative, .05)), len(order) - 1)
            high = min(int(np.searchsorted(cumulative, .95)), len(order) - 1)
            interval = [float(times[order[low]]), float(times[order[high]])]
        return {
            "hazard": hazard, "p_run_zero": float(posterior[0]),
            "short_run_mass_excluding_zero": mass,
            "map_run_length_observations": int(np.argmax(posterior)),
            "predictive_surprise_nats": -log_evidence,
            "conditional_short_boundary_interval_90": interval,
            "posterior_states": len(posterior),
        }


class _Channel:
    def __init__(self, name, method):
        self.name, self.method = name, method
        self.reset()

    def reset(self):
        self.history = deque()
        self.residuals = deque()
        self.previous_angle = None
        self.last_observed = None
        self.last_residual = None
        self.first_residual = None
        self.first_observed = None
        self.valid_points = 0
        self.positive = self.negative = 0.0
        self.bocpd = StudentTBOCPD()
        self.pending = []
        self.latched = False
        self.last_emission = None

    def update(self, time_s, value):
        cfg = DEFAULT_CONFIG
        raw_gap = (self.last_observed is not None and
                   time_s - self.last_observed > cfg["maximum_missing_s"] + 1e-9)
        residual_gap = (self.last_residual is not None and
                        time_s - self.last_residual > cfg["maximum_missing_s"] + 1e-9)
        reset = raw_gap or residual_gap
        if reset:
            self.reset()
        if not _finite(value):
            self.pending.clear()
            return {"status": "unknown", "reason": "channel_unavailable",
                    "continuity_reset": reset, "ready": False}, None
        value = float(value)
        if self.name == "angle_proxy_rad":
            value = unwrap_mod_pi(self.previous_angle, value)
            self.previous_angle = value
        if self.first_observed is None:
            self.first_observed = time_s
        previous_time = self.last_observed
        self.last_observed = time_s
        self.valid_points += 1
        while self.history and time_s - self.history[0][0] > cfg["trend_window_s"] + 1e-9:
            self.history.popleft()
        while self.residuals and time_s - self.residuals[0][0] > cfg["scale_window_s"] + 1e-9:
            self.residuals.popleft()
        result = {"status": "candidate", "reason": "warmup", "ready": False,
                  "continuity_reset": reset, "unwrapped_or_native_value": value}
        prediction = residual = z = None
        if len(self.history) >= cfg["trend_minimum_points"]:
            if self.first_residual is None:
                self.first_residual = time_s
            self.last_residual = time_s
            times = np.array([t - time_s for t, _ in self.history])
            values = np.array([v for _, v in self.history])
            # A plain least-squares window turns one gross outlier into many
            # subsequent residuals. Use a fixed robust linear estimator, with
            # no RANSAC randomness, extra rejection threshold or future point.
            slopes = [(values[j] - values[i]) / (times[j] - times[i])
                      for i in range(len(times) - 1)
                      for j in range(i + 1, len(times))]
            trend = float(np.median(slopes))
            intercept = float(np.median(values - trend * times))
            prediction, residual = float(intercept), value - float(intercept)
            old = np.array([r for _, r in self.residuals])
            center = float(np.median(old)) if len(old) else 0.0
            mad = float(np.median(np.abs(old - center))) if len(old) else 0.0
            floor = (cfg["angle_scale_floor_rad"] if self.name == "angle_proxy_rad"
                     else cfg["x_scale_floor_px"])
            scale = max(floor, 1.4826 * mad)
            z = (residual - center) / scale
            result.update(prediction=prediction, past_trend_per_s=float(trend),
                          residual=residual, past_residual_center=center,
                          past_residual_scale=scale, standardized_residual=z)
            if self.method == "trend_cusum":
                clipped = max(-cfg["cusum_clip_z"], min(cfg["cusum_clip_z"], z))
                self.positive = max(0.0, self.positive + clipped - cfg["cusum_allowance"])
                self.negative = max(0.0, self.negative - clipped - cfg["cusum_allowance"])
                score = max(self.positive, self.negative) / cfg["cusum_threshold"]
                result.update(cusum_positive=self.positive, cusum_negative=self.negative,
                              score=score, score_semantics="cusum_over_fixed_threshold")
                interval = [previous_time, time_s]
            else:
                diagnostic = self.bocpd.update(z, time_s)
                result.update(diagnostic)
                score = diagnostic["short_run_mass_excluding_zero"]
                result.update(score=score, score_semantics="short_run_posterior_mass_not_P0")
                interval = diagnostic["conditional_short_boundary_interval_90"]
            ready = (time_s - self.first_observed >= cfg["warmup_s"] - 1e-9
                     and self.valid_points >= cfg["warmup_points"]
                     # If cadence only recently became dense enough for a
                     # trend, the initial run cannot count as a new change.
                     and time_s - self.first_residual > cfg["bocpd_short_run_s"] + 1e-9)
            result.update(ready=ready, reason="ok" if ready else "warmup")
            threshold = 1.0 if self.method == "trend_cusum" else cfg["bocpd_short_mass_threshold"]
            if score < threshold:
                self.pending.clear()
                self.latched = False
            elif ready and not self.latched:
                self.pending.append({"time_s": time_s, "interval": interval})
            else:
                self.pending.clear()
            # A short missing row clears pending immediately; a large gap also
            # clears the probabilistic/filter state. Old values never add time.
            persistent = (len(self.pending) >= cfg["persistence_points"] and
                          time_s - self.pending[0]["time_s"] >= cfg["persistence_s"] - 1e-9)
            cooldown = (self.last_emission is None or
                        time_s - self.last_emission >= cfg["cooldown_s"] - 1e-9)
            signal = None
            if persistent and cooldown:
                self.latched = True
                self.last_emission = time_s
                selected = self.pending[0]["interval"] if self.method == "trend_cusum" else interval
                signal = {"channel": self.name, "score": score,
                          "t_est_lo": selected[0], "t_est_hi": selected[1],
                          "first_threshold_time_s": self.pending[0]["time_s"],
                          "persistence_points": len(self.pending)}
            self.residuals.append((time_s, residual))
        else:
            self.pending.clear()
            signal = None
        self.history.append((time_s, value))
        return result, signal


class CausalEventDetector:
    """One immutable method/configuration and one stream per instance.

    Required row keys: stream_id, source_frame, time_s, max_source_timestamp,
    available_at_s, available, reason. available_at_s is a replay availability
    clock, NOT measured production end-to-end latency. Valid rows may have one
    or both CHANNELS. Missing/quality-rejected channels use None, not old values.
    Extra keys (including annotations) are ignored, never used as features.
    """

    def __init__(self, stream_id, method="trend_cusum", config=None):
        if method not in ("trend_cusum", "bocpd"):
            raise ValueError("unknown method")
        if config is not None and config != DEFAULT_CONFIG:
            raise ValueError("V26 has one fixed configuration; no run-time tuning")
        if not isinstance(stream_id, str) or not stream_id:
            raise ValueError("a nonempty stream_id is required")
        self.stream_id, self.method = stream_id, method
        self.channels = {name: _Channel(name, method) for name in CHANNELS}
        self.last_time = self.last_frame = self.last_available = None
        self.max_source_timestamp = None
        self._events = []
        self.counts = Counter()

    @property
    def events(self):
        return copy.deepcopy(self._events)

    def update(self, row):
        required = ("stream_id", "source_frame", "time_s", "max_source_timestamp",
                    "available_at_s", "available", "reason")
        if any(key not in row for key in required):
            raise ValueError("missing causal measurement contract fields")
        if row["stream_id"] != self.stream_id:
            raise ValueError("cross-stream state reuse is forbidden")
        if not all(_finite(row[key]) for key in ("time_s", "max_source_timestamp", "available_at_s")):
            raise ValueError("invalid timestamp")
        now, source, available_at = (float(row[key]) for key in
                                     ("time_s", "max_source_timestamp", "available_at_s"))
        frame = row["source_frame"]
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ValueError("source_frame must be a nonnegative integer")
        if not now <= source + 1e-9 <= available_at + 2e-9:
            raise ValueError("future input or inconsistent source/availability timestamps")
        if (self.last_time is not None and
                (now <= self.last_time or frame <= self.last_frame or available_at < self.last_available)):
            raise ValueError("duplicate/out-of-order observation; state was not rewound")
        if not isinstance(row["available"], bool):
            raise ValueError("available must explicitly be boolean")
        self.last_time, self.last_frame, self.last_available = now, frame, available_at
        self.max_source_timestamp = max(source, self.max_source_timestamp if
                                        self.max_source_timestamp is not None else source)
        usable = row["available"] and any(_finite(row.get(name)) for name in CHANNELS)
        signals, diagnostics = [], {}
        for name, channel in self.channels.items():
            diagnostic, signal = channel.update(now, row.get(name) if usable else None)
            diagnostics[name] = diagnostic
            if signal is not None:
                signals.append(signal)
        status = "candidate" if usable else "unknown"
        self.counts[status] += 1
        event = None
        if signals:
            # Correlated channels only form a union. The score is not a joint
            # probability, and the union has not been false-alarm calibrated.
            chosen = max(signals, key=lambda s: (s["score"], s["channel"]))
            event = {
                "event_id": f"{self.stream_id}:{self.method}:{len(self._events) + 1}",
                "stream_id": self.stream_id, "method": self.method,
                "event_type": "free_rope_statistical_change", "status": "candidate",
                "t_est_lo": chosen["t_est_lo"], "t_est_hi": chosen["t_est_hi"],
                "t_emit": available_at, "max_source_timestamp": self.max_source_timestamp,
                "score": chosen["score"], "evidence_source": [s["channel"] for s in signals],
                "evidence": signals,
                "interval_semantics": ("threshold_onset_bracket_not_physical_t1" if
                    self.method == "trend_cusum" else
                    "conditional_short_run_boundary_90pct_not_physical_t1_CI"),
                "rejection_reason": "physical_layer_transition_unverified",
                "physical_t1_confirmed": False, "mechanical_turn_confirmed": False,
                "device_advisory": False, "deployment_permitted": False,
            }
            self._events.append(copy.deepcopy(event))
        return {
            "stream_id": self.stream_id, "method": self.method, "source_frame": frame,
            "time_s": now, "max_source_timestamp": source, "available_at_s": available_at,
            "status": status, "available": usable,
            "measurement_reason": str(row["reason"]),
            "reason": str(row["reason"]) if not usable else "signal_candidate_not_physical_event",
            "quality": {key: copy.deepcopy(row.get(key)) for key in QUALITY_FIELDS},
            "channels": diagnostics, "event": event,
        }


def run_stream(rows, method="trend_cusum", *, stream_id=None, config=None):
    """Process in arrival order; sorting would conceal out-of-order inputs."""
    iterator = iter(rows)
    first = next(iterator, None)
    if first is None:
        if stream_id is None:
            raise ValueError("empty stream requires an explicit stream_id")
        detector = CausalEventDetector(stream_id, method, config)
        output = []
    else:
        detector = CausalEventDetector(stream_id or first.get("stream_id"), method, config)
        output = [detector.update(first)]
        output.extend(detector.update(row) for row in iterator)
    return {"stream_id": detector.stream_id, "method": method,
            "config": copy.deepcopy(DEFAULT_CONFIG), "rows": output,
            "events": detector.events, "status_counts": dict(detector.counts),
            "evaluation_status": "not_evaluable_without_independent_physical_labels",
            "deployment_permitted": False}
