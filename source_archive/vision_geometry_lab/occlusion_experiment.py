"""Offline, causal occlusion ablation. This is NOT a production completion gate.

Current-frame right-hand candidate points are held out from both model fitting
and model selection. A second method may use a genuinely EARLIER full observation
from a fixed acquisition schedule. Right-hand candidates are pseudo-references,
not independent ground truth. No physical contact, layer, or control output is
produced. The existing arc, conic, geometry and app modules are not modified.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .conic_fit import _orthogonal, _rotation, _serialize, fit_conic
from .geometry import dump_json, rope_exclusion_band, write_image


@dataclass(frozen=True)
class Protocol:
    """Fixed experimental choices, declared before inspecting hidden errors."""

    split_width_fractions: tuple = (.45, .50, .55)
    history_times_s: tuple = tuple(float(i) for i in range(0, 25, 2))
    probe_times_s: tuple = (1., 5., 9., 13., 17., 21., 22.5, 23.5,
                            24.5, 24.6, 24.7, 25.1, 26.5)
    max_prior_age_s: float = 5.
    min_visible_points: int = 35
    min_reference_points: int = 35
    min_reference_width_fraction: float = .06
    min_clear_side_points: int = 50
    min_clear_left_width_fraction: float = .12
    min_clear_right_width_fraction: float = .06
    min_clear_total_width_fraction: float = .32
    max_translation_px: float = 20.
    max_scale_change: float = .015
    max_before_update_p95_px: float = 12.
    max_after_update_p95_px: float = 4.
    max_update_condition: float = 1e4
    scoring_failure_threshold_px: float = 5.
    rope_exclusion_margin_px: float = 8.
    random_seed: int = 20260926


def _points(arc):
    points = np.asarray(arc.get("conic_support_points", arc.get("points", [])), float)
    if points.ndim != 2 or points.shape[1:] != (2,):
        return np.empty((0, 2))
    return points[np.isfinite(points).all(axis=1)]


def split_support(arc, cut_x):
    """Only the left return value may be passed to either prediction method.

    Explicitly build a clean input, rather than copying right-derived fields,
    confidences, fit residuals, or model selection diagnostics from the full arc.
    Image-x, not physical flange/camera handedness, defines 'left' here.
    """
    points = _points(arc)
    visible = points[points[:, 0] < cut_x]
    hidden = points[points[:, 0] >= cut_x]
    segments = []
    for segment in arc.get("conic_support_segments", arc.get("segments", [])):
        part = np.asarray(segment, float)
        if part.ndim != 2 or len(part) < 2:
            continue
        indices = np.flatnonzero(part[:, 0] < cut_x)
        for run in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
            if len(run) >= 2:
                segments.append(part[run].tolist())
    if not segments and len(visible):
        # The fallback has no gap-bridging role: fit_conic re-splits angular gaps.
        segments = [visible.tolist()]
    left = dict(valid=len(visible) >= 6, points=visible.tolist(),
                conic_support_points=visible.tolist(), segments=segments,
                conic_support_segments=segments,
                exclusion_polygons=arc.get("exclusion_polygons", []))
    return left, hidden


def observation_quality(points, image_shape, protocol=Protocol()):
    """Observed coverage proxy, deliberately independent of ellipse success.

    'Clear enough to try a historical shape' means paint-verified candidates on
    both image sides, with sufficient horizontal extent. It does NOT mean that
    those candidates are truth, that the object is circular, or that a fit passes.
    """
    points = np.asarray(points, float).reshape(-1, 2)
    width = float(image_shape[1])
    left, right = points[points[:, 0] < width / 2], points[points[:, 0] >= width / 2]
    extent = lambda p: float(np.ptp(p[:, 0])) if len(p) > 1 else 0.
    conditions = dict(
        left_count=len(left) >= protocol.min_clear_side_points,
        right_count=len(right) >= protocol.min_clear_side_points,
        left_extent=extent(left) >= width * protocol.min_clear_left_width_fraction,
        right_extent=extent(right) >= width * protocol.min_clear_right_width_fraction,
        total_extent=extent(points) >= width * protocol.min_clear_total_width_fraction)
    return dict(coverage_eligible=all(conditions.values()), conditions=conditions,
                left_count=len(left), right_count=len(right),
                left_x_extent_px=extent(left), right_x_extent_px=extent(right),
                total_x_extent_px=extent(points),
                definition="paint_verified_candidate_count_and_image_x_coverage_not_fit_success",
                is_ground_truth=False)


def _compact(model):
    keys = ("valid", "status", "reason", "center", "semi_axes", "angle_rad", "angle_deg",
            "implicit", "arc_interval_rad", "observed_angular_span_deg", "residual_px",
            "sensitivity_px", "sensitivity_models", "diagnostics")
    return {key: model[key] for key in keys if key in model}


def _parameters(model):
    return np.r_[model["center"], np.log(model["semi_axes"]), model["angle_rad"]]


def _transformed(prior, delta, protocol):
    out = _parameters(prior).copy()
    out[:2] += np.asarray(delta[:2]) * protocol.max_translation_px
    out[2:4] += delta[2] * math.log1p(protocol.max_scale_change)
    return out


def _update_parameters(visible, prior, protocol):
    """Robust 3-DOF update; ratio and angle stay fixed, limits are NOT tuned."""
    delta = np.zeros(3)
    condition = math.inf
    damping = 1e-3
    for _ in range(35):
        residual = _orthogonal(visible, _transformed(prior, delta, protocol))
        sigma = max(.5, 1.4826 * float(np.median(np.abs(residual - np.median(residual)))))
        weights = np.minimum(1., 1.5 * sigma / np.maximum(np.abs(residual), 1e-8))
        jacobian = []
        for j in range(3):
            shifted = delta.copy()
            shifted[j] += 1e-4
            trial = _orthogonal(visible, _transformed(prior, shifted, protocol))
            jacobian.append((trial - residual) / 1e-4)
        jacobian = np.column_stack(jacobian)
        weighted = jacobian * np.sqrt(weights[:, None])
        singular = np.linalg.svd(weighted, compute_uv=False)
        condition = float(singular[0] / max(singular[-1], 1e-12))
        # One weak unit Gaussian penalty per normalized update coordinate.
        # This is a declared regularizer, NOT a calibrated probabilistic prior.
        hessian = weighted.T @ weighted + np.eye(3)
        rhs = jacobian.T @ (weights * residual) + delta
        try:
            step = np.linalg.solve(hessian + damping * np.diag(np.diag(hessian)), -rhs)
        except np.linalg.LinAlgError:
            return None, condition
        candidate = delta + np.clip(step, -.3, .3)
        if not np.isfinite(candidate).all():
            return None, condition
        trial = _orthogonal(visible, _transformed(prior, candidate, protocol))
        cost = float(np.sum(weights * residual**2) + delta @ delta)
        new_cost = float(np.sum(weights * trial**2) + candidate @ candidate)
        if new_cost <= cost:
            delta = candidate
            damping = max(1e-9, damping * .3)
            if np.linalg.norm(step) < 1e-7:
                break
        else:
            damping = min(1e7, damping * 8)
    return delta, condition


def update_prior(visible_points, history_entry, time_s, protocol=Protocol()):
    """No hidden points accepted by this API; reject future/stale/bad histories."""
    out = dict(valid=False, reason="no_eligible_earlier_shape", model=None,
               source_time_s=None, age_s=None, physical_contact_confirmed=False)
    if history_entry is None:
        return out
    source_time = float(history_entry["time_s"])
    age = float(time_s - source_time)
    out.update(source_time_s=source_time, age_s=age)
    if age <= 0:
        out["reason"] = "prior_is_not_strictly_earlier"
        return out
    if age > protocol.max_prior_age_s:
        out["reason"] = "prior_is_stale"
        return out
    prior = history_entry.get("model", {})
    if not history_entry.get("quality", {}).get("coverage_eligible") or not prior.get("valid"):
        out["reason"] = "history_lacks_coverage_or_original_ellipse_gate"
        return out
    visible = np.asarray(visible_points, float).reshape(-1, 2)
    if len(visible) < protocol.min_visible_points or not np.isfinite(visible).all():
        out["reason"] = "insufficient_current_visible_support"
        return out
    before = np.abs(_orthogonal(visible, _parameters(prior)))
    out["before_update_p95_px"] = float(np.percentile(before, 95))
    if out["before_update_p95_px"] > protocol.max_before_update_p95_px:
        out["reason"] = "prior_inconsistent_with_current_visible_arc"
        return out
    delta, condition = _update_parameters(visible, prior, protocol)
    out["update_condition"] = condition if np.isfinite(condition) else None
    if delta is None or condition > protocol.max_update_condition:
        out["reason"] = "low_dof_update_is_ill_conditioned"
        return out
    translation = delta[:2] * protocol.max_translation_px
    scale = math.exp(delta[2] * math.log1p(protocol.max_scale_change))
    out.update(translation_px=translation.tolist(), uniform_scale=scale,
               fixed_parameters=["axis_ratio", "angle"], update_degrees_of_freedom=3)
    if np.linalg.norm(translation) > protocol.max_translation_px or abs(scale - 1) > protocol.max_scale_change:
        out["reason"] = "update_exceeds_declared_small_motion_limits"
        return out
    parameters = _transformed(prior, delta, protocol)
    after = np.abs(_orthogonal(visible, parameters))
    out.update(after_update_p95_px=float(np.percentile(after, 95)),
               after_update_rms_px=float(np.sqrt(np.mean(after**2))))
    if out["after_update_p95_px"] > protocol.max_after_update_p95_px:
        out["reason"] = "updated_shape_inconsistent_with_current_visible_arc"
        return out
    model = _serialize(parameters, np.zeros(2), 1.)
    model.update(valid=True, arc_interval_rad=prior.get("arc_interval_rad"),
                 sensitivity_models=prior.get("sensitivity_models", []))
    # Propagate history shape variants through THIS visible-only transform.
    # Shape uncertainty that is invisible on the left may survive the update.
    variants = []
    for variant in prior.get("sensitivity_models", []):
        alternative = _parameters(variant)
        alternative[:2] += translation
        alternative[2:4] += math.log(scale)
        variants.append(_serialize(alternative, np.zeros(2), 1.))
    model["sensitivity_models"] = variants
    out.update(valid=True, reason="causal_shape_prior_visible_only_small_similarity_update",
               model=model, history_model_status=prior.get("status"),
               uncertainty_is_not_calibrated=True,
               hidden_geometry_verified=False)
    return out


def select_history(history, time_s):
    """Most recent eligible EARLIER sample, never ranked using the held-out side."""
    earlier = [entry for entry in history if entry["time_s"] < time_s and
               entry.get("quality", {}).get("coverage_eligible") and entry.get("model", {}).get("valid")]
    return max(earlier, key=lambda entry: entry["time_s"]) if earlier else None


def curve_points(model, angles):
    axes = np.asarray(model["semi_axes"], float)
    local = np.column_stack([axes[0] * np.cos(angles), axes[1] * np.sin(angles)])
    return local @ _rotation(model["angle_rad"]).T + model["center"]


def nearest_curve_distance(points, model, interval=None):
    """Global nearest-distance scoring, also safe far from a wrong ellipse.

    The core Newton projection is intended for near-inliers. Held-out errors can
    be hundreds of pixels, so initialize every point by a full curve search and
    then bounded golden-section minimization. No scoring influences prediction.
    """
    points = np.asarray(points, float).reshape(-1, 2)
    if not len(points):
        return np.empty(0)
    low, high = (0., 2 * math.pi) if interval is None else interval
    grid = np.linspace(low, high, 2049)
    samples = curve_points(model, grid)
    distances = []
    step = (high - low) / 2048
    ratio = (math.sqrt(5) - 1) / 2
    for start in range(0, len(points), 128):
        part = points[start:start + 128]
        squared = np.sum((part[:, None] - samples[None])**2, axis=2)
        index = np.argmin(squared, axis=1)
        left, right = grid[index] - step, grid[index] + step
        if interval is not None:
            left, right = np.maximum(left, low), np.minimum(right, high)
        for _ in range(30):
            a, b = right - ratio * (right - left), left + ratio * (right - left)
            da = np.sum((curve_points(model, a) - part)**2, axis=1)
            db = np.sum((curve_points(model, b) - part)**2, axis=1)
            lower = da < db
            right, left = np.where(lower, b, right), np.where(lower, left, a)
        projection = curve_points(model, (left + right) / 2)
        distances.extend(np.linalg.norm(projection - part, axis=1).tolist())
    return np.asarray(distances)


def score_hidden(hidden, model, image_shape, protocol=Protocol()):
    """Post-hoc evaluator. No field returned here can select or update a model."""
    hidden = np.asarray(hidden, float).reshape(-1, 2)
    width = float(np.ptp(hidden[:, 0])) if len(hidden) > 1 else 0.
    adequate = len(hidden) >= protocol.min_reference_points and width >= image_shape[1] * protocol.min_reference_width_fraction
    score = dict(reference_count=len(hidden), reference_x_extent_px=width,
                 reference_coverage_adequate=bool(adequate), independent_ground_truth=False,
                 reference="held_out_current_paint_verified_candidates",
                 valid_prediction=bool(model and model.get("valid")),
                 nearest_full_ellipse_is_optimistic_distance=True)
    if not score["valid_prediction"] or not len(hidden):
        score.update(median_px=None, p95_px=None, max_px=None, rms_px=None,
                     p95_exceeds_experimental_5px_threshold=None)
        return score
    distance = nearest_curve_distance(hidden, model)
    score.update(median_px=float(np.median(distance)), p95_px=float(np.percentile(distance, 95)),
                 max_px=float(np.max(distance)), rms_px=float(np.sqrt(np.mean(distance**2))),
                 p95_exceeds_experimental_5px_threshold=bool(np.percentile(distance, 95) > protocol.scoring_failure_threshold_px))
    interval = model.get("arc_interval_rad")
    if interval:
        clipped = nearest_curve_distance(hidden, model, interval)
        score["previously_observed_or_currently_fitted_interval_p95_px"] = float(np.percentile(clipped, 95))
    # This AFTER-THE-FACT diagnostic probes extension, not the observed-arc
    # sensitivity exported by fit_conic. It is neither a gate nor a true bound.
    variants = model.get("sensitivity_models", [])
    if variants:
        differences = [np.abs(nearest_curve_distance(hidden, v) - distance) for v in variants]
        envelope = np.max(differences, axis=0)
        score["hidden_error_leave_block_out_sensitivity_p95_px"] = float(np.percentile(envelope, 95))
    return score


def predict_case(arc, image_shape, time_s, fraction, history, protocol=Protocol()):
    cut = float(image_shape[1] * fraction)
    visible_arc, hidden = split_support(arc, cut)
    visible = _points(visible_arc)
    # Predictions are finalized before the held-out evaluator is called.
    free = _compact(fit_conic(visible_arc, image_shape))
    prior_entry = select_history(history, time_s)
    prior_only = prior_entry["model"] if prior_entry and 0 < time_s - prior_entry["time_s"] <= protocol.max_prior_age_s else None
    updated = update_prior(visible, prior_entry, time_s, protocol)
    out = dict(time_s=float(time_s), cut_width_fraction=float(fraction), cut_x=cut,
               visible_count=len(visible), hidden_count=len(hidden),
               visible_points=visible.tolist(), hidden_reference_points=hidden.tolist(),
               free_left_model=free,
               free_model_validity_scope="original_gate_on_visible_left_interval_only_not_hidden_completion",
               selected_history_time_s=None if prior_entry is None else prior_entry["time_s"],
               prior_only_model=prior_only, prior_update=updated,
               prior_only_validity_scope="diagnostic_history_baseline_not_checked_against_current_visible_arc",
               unchanged_physical_radius_or_identity_independently_verified=False,
               prediction_used_current_hidden_points=False,
               prediction_used_future_frames=False,
               scores={})
    out["scores"] = dict(free_left=score_hidden(hidden, free, image_shape, protocol),
                         historical_prior_only=score_hidden(hidden, prior_only, image_shape, protocol),
                         historical_prior_updated=score_hidden(hidden, updated.get("model"), image_shape, protocol))
    return out


def _draw_case(image, case):
    out = image.copy()
    height, width = out.shape[:2]
    cut = int(round(case["cut_x"]))
    cv2.line(out, (cut, 0), (cut, height - 1), (180, 180, 180), 3)
    for field, color in (("visible_points", (40, 240, 40)), ("hidden_reference_points", (0, 200, 255))):
        for point in case[field]:
            cv2.circle(out, tuple(np.round(point).astype(int)), 4, color, -1)
    models = ((case["free_left_model"], (255, 100, 220)),
              (case["prior_update"].get("model"), (255, 210, 60)))
    for model, color in models:
        if not model or not model.get("valid"):
            continue
        # Diagnostic candidates only: full conic dots explicitly show how an
        # observed-left fit can have an incorrect rightmost extent. Not an app
        # overlay, not measurements, not a physical completion confirmation.
        points = curve_points(model, np.linspace(0, 2 * math.pi, 1500))
        for i, point in enumerate(points):
            if i % 8 >= 4 or not (0 <= point[0] < width and 0 <= point[1] < height):
                continue
            cv2.circle(out, tuple(np.round(point).astype(int)), 3, color, -1)
    labels = [f"t={case['time_s']:.1f}s  hide x >= {case['cut_width_fraction']:.0%} width",
              "GREEN: fit input; ORANGE: held-out candidates, NOT truth",
              "DOTTED MAGENTA: left-only conic; CYAN: causal prior update",
              f"Prior: {case['selected_history_time_s']}s; update: {case['prior_update']['reason']}",
              "DIAGNOSTIC HYPOTHESES ONLY - no contact/layer confirmation"]
    for i, label in enumerate(labels):
        cv2.putText(out, label, (35, 70 + 57 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.25,
                    (0, 0, 0), 8, cv2.LINE_AA)
        cv2.putText(out, label, (35, 70 + 57 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.25,
                    (245, 245, 245), 3, cv2.LINE_AA)
    return cv2.resize(out, (1600, round(height * 1600 / width)), interpolation=cv2.INTER_AREA)


def _aggregate(cases):
    report = {}
    for method in ("free_left", "historical_prior_only", "historical_prior_updated"):
        scores = [c["scores"][method] for c in cases]
        adequate = [s for s in scores if s["reference_coverage_adequate"]]
        accepted = [s for s in adequate if s["valid_prediction"]]
        failed = [s for s in accepted if s["p95_exceeds_experimental_5px_threshold"]]
        report[method] = dict(total_probes=len(scores), adequate_candidate_reference_cases=len(adequate),
                              predictions=len([s for s in scores if s["valid_prediction"]]),
                              accepted_adequate_cases=len(accepted),
                              rejected_adequate_cases=len(adequate) - len(accepted),
                              accepted_p95_over_5px=len(failed),
                              accepted_bad_rate=len(failed) / len(accepted) if accepted else None,
                              rejection_or_bad_rate=(len(adequate) - len(accepted) + len(failed)) / len(adequate) if adequate else None,
                              median_of_case_medians_px=float(np.median([s["median_px"] for s in accepted])) if accepted else None,
                              median_of_case_p95_px=float(np.median([s["p95_px"] for s in accepted])) if accepted else None,
                              maximum_case_error_px=max((s["max_px"] for s in accepted), default=None))
    report["sampling_units"] = dict(
        distinct_probe_times=len({c["time_s"] for c in cases}),
        adequate_reference_distinct_times=len({c["time_s"] for c in cases if c["scores"]["free_left"]["reference_coverage_adequate"]}),
        repeated_cuts_are_not_independent_frames=True)
    paired = [c for c in cases if c["scores"]["free_left"]["reference_coverage_adequate"]
              and c["scores"]["free_left"]["valid_prediction"]
              and c["scores"]["historical_prior_updated"]["valid_prediction"]]
    report["paired_available_predictions"] = dict(
        cases=len(paired),
        prior_update_lower_p95=sum(c["scores"]["historical_prior_updated"]["p95_px"] < c["scores"]["free_left"]["p95_px"] for c in paired),
        cases_are_not_a_success_probability_estimate=True)
    return report


def synthetic_experiment(protocol=Protocol()):
    """Known ellipse truth; noise and motion are fixed, not tuned to real results."""
    rng = np.random.default_rng(protocol.random_seed)
    truth = dict(valid=True, center=[2000., 1800.], semi_axes=[900., 260.], angle_rad=.15)
    angles = np.linspace(math.radians(195), math.radians(345), 600)
    clean = curve_points(truth, angles)
    earlier = clean + rng.normal(0, .5, clean.shape)
    arc = dict(valid=True, points=earlier.tolist(), segments=[earlier.tolist()])
    model = _compact(fit_conic(arc, (3000, 4096)))
    entry = dict(time_s=0., quality=observation_quality(earlier, (3000, 4096), protocol), model=model)
    moved_truth = dict(truth, center=[2004., 1797.], semi_axes=[904.5, 261.3])
    current = curve_points(moved_truth, angles)
    noisy = current + rng.normal(0, .5, current.shape)
    current_arc = dict(valid=True, points=noisy.tolist(), segments=[noisy.tolist()])
    case = predict_case(current_arc, (3000, 4096), 1., .5, [entry], protocol)
    true_hidden = current[current[:, 0] >= 2048.]
    actual_truth_scores = {
        "free_left": score_hidden(true_hidden, case["free_left_model"], (3000, 4096), protocol),
        "prior_updated": score_hidden(true_hidden, case["prior_update"].get("model"), (3000, 4096), protocol)}
    for score in actual_truth_scores.values():
        score.update(independent_ground_truth=True, reference="noise_free_known_synthetic_ellipse")
    return dict(history=entry, case=case, known_truth=moved_truth, ground_truth_scores=actual_truth_scores,
                observation_noise_sigma_px=.5, actual_translation_px=[4., -3.], actual_uniform_scale=1.005)


def ambiguous_prior_counterexample(protocol=Protocol()):
    """A wrong shape can pass a short visible arc: a declared failure example.

    Both ellipses share the left vertex and its curvature. This intentionally
    violates the unchanged-object-shape premise between the earlier and current
    observation. No image-only left test can generally certify that premise.
    """
    truth = dict(valid=True, center=[2000., 1800.], semi_axes=[900., 260.], angle_rad=0.)
    wrong = dict(valid=True, center=[2300., 1800.],
                 semi_axes=[1200., float(math.sqrt(1200 * 260**2 / 900))], angle_rad=0.)
    previous_points = curve_points(wrong, np.deg2rad(np.linspace(195, 345, 600)))
    previous_arc = dict(valid=True, points=previous_points.tolist(), segments=[previous_points.tolist()])
    history = dict(time_s=0., quality=observation_quality(previous_points, (3000, 4096), protocol),
                   model=_compact(fit_conic(previous_arc, (3000, 4096))))
    visible = curve_points(truth, np.deg2rad(np.linspace(193, 215, 150)))
    hidden = curve_points(truth, np.deg2rad(np.linspace(290, 345, 180)))
    current_arc = dict(valid=True, points=np.vstack([visible, hidden]).tolist(),
                       segments=[visible.tolist(), hidden.tolist()])
    case = predict_case(current_arc, (3000, 4096), 1., .5, [history], protocol)
    score = score_hidden(hidden, case["prior_update"].get("model"), (3000, 4096), protocol)
    score.update(independent_ground_truth=True, reference="noise_free_known_changed_synthetic_ellipse")
    return dict(history=history, case=case, current_truth=truth, previous_truth=wrong,
                ground_truth_score=score, deliberately_violated_assumption="same_ellipse_shape_across_time",
                conclusion="a_small_visible_residual_does_not_prove_a_history_shape_is_still_correct")


def run_experiment(video_path, output_dir, protocol=Protocol()):
    from .arc_detector import detect_arc
    from .rope_detector import detect_rope

    cv2.setNumThreads(2)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Persist declared design before any video-derived predictions/evaluation.
    dump_json(output_dir / "protocol.json", dict(
        protocol=asdict(protocol), source=str(Path(video_path).resolve()),
        split_rule="fixed image-width fractions; no current hidden points choose model or parameters",
        historical_access="separate single-outage episodes: fixed earlier acquisition frames are visible before outage",
        chronological_rule="strict source_time < probe_time; same video/camera only",
        extraction_limitation="shared full-image feature detector; held-out geometric candidates, not pixel-blind segmentation",
        reference_limitation="candidate consistency, not independent accuracy or contact truth",
        thresholds="fixed exploratory pixel limits, not commissioned accuracy requirements",
        physical_control=False))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    histories, frames, observations = [], {}, {}
    times = sorted(set(protocol.history_times_s) | set(protocol.probe_times_s))
    try:
        for time_s in times:
            index = int(round(time_s * fps))
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                observations[time_s] = dict(time_s=time_s, frame_index=index, available=False)
                continue
            rope = detect_rope(frame)
            mask, exclusion = rope_exclusion_band(frame.shape, rope, protocol.rope_exclusion_margin_px)
            arc = detect_arc(frame, exclusion_mask=mask)
            quality = observation_quality(_points(arc), frame.shape, protocol)
            observation = dict(time_s=time_s, frame_index=index, available=True,
                               image_shape=list(frame.shape), quality=quality,
                               arc=arc, rope=rope, exclusion=exclusion)
            observations[time_s] = observation
            dump_json(output_dir / "observations" / f"t_{time_s:06.2f}.json", observation)
            if time_s in protocol.history_times_s:
                # Coverage selection is frozen BEFORE ellipse fitting; retain
                # both results so no one can redefine 'clear' as successful fit.
                model = _compact(fit_conic(arc, frame.shape)) if quality["coverage_eligible"] else dict(valid=False, reason="observed_coverage_not_eligible")
                histories.append(dict(time_s=time_s, frame_index=index, quality=quality, model=model))
            if time_s in protocol.probe_times_s:
                frames[time_s] = frame
            print(json.dumps(dict(stage="observe", time_s=time_s, left=quality["left_count"],
                                  right=quality["right_count"], coverage=quality["coverage_eligible"])), flush=True)
    finally:
        capture.release()
    dump_json(output_dir / "history.json", histories)
    cases = []
    for time_s in protocol.probe_times_s:
        if time_s not in frames:
            continue
        frame, observation = frames[time_s], observations[time_s]
        for fraction in protocol.split_width_fractions:
            case = predict_case(observation["arc"], frame.shape, time_s, fraction, histories, protocol)
            case.update(frame_index=observation["frame_index"], source_image_size=[frame.shape[1], frame.shape[0]])
            name = f"t_{time_s:06.2f}_hide_{int(round(fraction * 100)):02d}"
            dump_json(output_dir / "cases" / f"{name}.json", case)
            write_image(output_dir / "figures" / f"{name}.jpg", _draw_case(frame, case))
            cases.append(case)
            print(json.dumps(dict(stage="score", time_s=time_s, fraction=fraction,
                                  free=case["free_left_model"].get("reason"), prior=case["prior_update"]["reason"])), flush=True)
        del frames[time_s]
    synthetic = synthetic_experiment(protocol)
    dump_json(output_dir / "synthetic.json", synthetic)
    counterexample = ambiguous_prior_counterexample(protocol)
    dump_json(output_dir / "synthetic_wrong_lookalike_prior.json", counterexample)
    canvas = np.full((3000, 4096, 3), 45, np.uint8)
    write_image(output_dir / "figures" / "synthetic_correct_prior.jpg", _draw_case(canvas, synthetic["case"]))
    write_image(output_dir / "figures" / "synthetic_wrong_lookalike_prior.jpg", _draw_case(canvas, counterexample["case"]))
    summary = dict(protocol=asdict(protocol), source=str(Path(video_path).resolve()), fps=fps,
                   history_samples=len(histories),
                   history_coverage_eligible=sum(h["quality"]["coverage_eligible"] for h in histories),
                   history_model_available=sum(h["model"].get("valid", False) for h in histories),
                   history_models=[dict(time_s=h["time_s"], coverage_eligible=h["quality"]["coverage_eligible"],
                                        valid=h["model"].get("valid", False), reason=h["model"].get("reason")) for h in histories],
                   aggregate=_aggregate(cases),
                   prior_update_reasons={reason: sum(c["prior_update"]["reason"] == reason for c in cases)
                                         for reason in sorted({c["prior_update"]["reason"] for c in cases})},
                   independent_real_ground_truth=False,
                   no_hidden_side_optimization=True, no_future_prior=True,
                   case_files=[f"cases/t_{c['time_s']:06.2f}_hide_{int(round(c['cut_width_fraction'] * 100)):02d}.json" for c in cases])
    dump_json(output_dir / "summary.json", summary)
    return summary


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=root / "现场视频/训练采集/双相机同步/dual_20260924_111500/left_DA9627065.avi")
    parser.add_argument("--output", type=Path, default=root / "vision_geometry_lab/outputs/occlusion_v4")
    args = parser.parse_args()
    summary = run_experiment(args.video, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
