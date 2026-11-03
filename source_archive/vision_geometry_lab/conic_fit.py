"""Robust, explicitly modeled ellipse completion of observed inner-edge support.

An ellipse is a hypothesis, not a replacement for measured pixels. Short,
ill-conditioned and unstable support is rejected. Hidden portions are modeled
only within the angular interval spanned by observations, never a full invented
bottom half. Residuals and leave-block-out sensitivity are NOT accuracy bounds.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def _invalid(reason, **diagnostics):
    return dict(valid=False, status="invalid", reason=reason, center=None,
                semi_axes=None, angle_rad=None, angle_deg=None, implicit=None,
                arc_interval_rad=None, observed_angular_span_deg=0.,
                model_points=[], model_supported=[], model_sensitivity_px=[],
                observed_model_segments=[], modeled_segments=[], inlier_points=[],
                residual_px=None, sensitivity_px=None, sensitivity_models=[],
                diagnostics=diagnostics)


def _rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def _ellipse_parameters(coefficients):
    a, b, c, d, e, f = coefficients
    quadratic = np.array([[a, b / 2], [b / 2, c]])
    if np.linalg.det(quadratic) <= 0:
        return None
    try:
        center = -.5 * np.linalg.solve(quadratic, np.array([d, e]))
        rhs = float(center @ quadratic @ center - f)
        eigenvalues, eigenvectors = np.linalg.eigh(quadratic / rhs)
    except np.linalg.LinAlgError:
        return None
    if np.min(eigenvalues) <= 0 or not np.all(np.isfinite(eigenvalues)):
        return None
    axes = 1 / np.sqrt(eigenvalues)
    angle = math.atan2(eigenvectors[1, 0], eigenvectors[0, 0]) % math.pi
    return np.array([center[0], center[1], np.log(axes[0]), np.log(axes[1]), angle])


def _direct(points, weights=None):
    """Normalized Halir/Flusser reduced direct ellipse least-squares system."""
    if len(points) < 6:
        return None
    x, y = points.T
    d1 = np.column_stack([x*x, x*y, y*y])
    d2 = np.column_stack([x, y, np.ones(len(points))])
    if weights is not None:
        w = np.sqrt(np.maximum(0, weights))[:, None]
        d1, d2 = d1*w, d2*w
    s1, s2, s3 = d1.T @ d1, d1.T @ d2, d2.T @ d2
    try:
        transform = -np.linalg.solve(s3, s2.T)
        reduced = s1 + s2 @ transform
        constraint = np.array([[0., 0., 2.], [0., -1., 0.], [2., 0., 0.]])
        _, vectors = np.linalg.eig(np.linalg.solve(constraint, reduced))
    except np.linalg.LinAlgError:
        return None
    choices = []
    for vector in vectors.T:
        if np.max(np.abs(np.imag(vector))) > 1e-7:
            continue
        vector = np.real(vector)
        if 4*vector[0]*vector[2] - vector[1]**2 <= 1e-10:
            continue
        parameters = _ellipse_parameters(np.r_[vector, transform @ vector])
        if parameters is not None:
            choices.append(parameters)
    if not choices:
        return None
    return min(choices, key=lambda p: float(np.median(np.abs(_sampson(points, p)))))


def _sampson(points, parameters):
    local = (points - parameters[:2]) @ _rotation(parameters[4])
    inverse = np.exp(-2*parameters[2:4])
    value = np.sum(local*local*inverse, axis=1) - 1
    gradient = 2*np.sqrt(np.sum(local*local*inverse*inverse, axis=1))
    return value / np.maximum(gradient, 1e-12)


def _orthogonal(points, parameters, return_angles=False):
    """Signed orthogonal distances to the closest local ellipse projection.

    Refinement inputs are already within a narrow inlier tube. Newton's nearest
    parameter solve is therefore initialized in that local basin, not at a far
    algebraic minimum. RANSAC scoring uses the cheaper first-order distance.
    """
    local = (points - parameters[:2]) @ _rotation(parameters[4])
    axes = np.exp(parameters[2:4])
    angles = np.arctan2(local[:, 1] / axes[1], local[:, 0] / axes[0])
    for _ in range(9):
        cosine, sine = np.cos(angles), np.sin(angles)
        projected = np.column_stack([axes[0]*cosine, axes[1]*sine])
        derivative = np.column_stack([-axes[0]*sine, axes[1]*cosine])
        second = -projected
        difference = projected - local
        numerator = np.sum(difference*derivative, axis=1)
        denominator = np.sum(derivative*derivative + difference*second, axis=1)
        delta = np.divide(numerator, denominator, out=np.zeros_like(numerator),
                          where=np.abs(denominator) > 1e-12)
        angles -= np.clip(delta, -.35, .35)
    projected = np.column_stack([axes[0]*np.cos(angles), axes[1]*np.sin(angles)])
    signs = np.sign(np.sum((local/axes)**2, axis=1)-1)
    distance = np.linalg.norm(local-projected, axis=1)*signs
    return (distance, angles) if return_angles else distance


def _geometric_refine(points, initial, pixel_scale, iterations=25):
    parameters = initial.copy()
    damping = 1e-4
    for _ in range(iterations):
        residual = _orthogonal(points, parameters)
        median = np.median(residual)
        sigma = max(.35/pixel_scale, 1.4826*np.median(np.abs(residual-median)))
        u = residual / (4.685*sigma)
        weights = np.maximum(0., 1-u*u)**2
        if np.sum(weights > .01) < 12:
            return None, math.inf
        jacobian = []
        for j in range(5):
            delta = 1e-5
            shifted = parameters.copy()
            shifted[j] += delta
            jacobian.append((_orthogonal(points, shifted)-residual)/delta)
        jacobian = np.column_stack(jacobian)
        weighted = jacobian*np.sqrt(weights[:, None])
        rhs = jacobian.T @ (weights*residual)
        matrix = weighted.T @ weighted
        try:
            step = np.linalg.solve(matrix + damping*np.diag(np.maximum(np.diag(matrix), 1e-6)), -rhs)
        except np.linalg.LinAlgError:
            return None, math.inf
        step[:2] = np.clip(step[:2], -.3, .3)
        step[2:] = np.clip(step[2:], -.2, .2)
        candidate = parameters + step
        if not np.all(np.isfinite(candidate)) or np.any(np.abs(candidate[2:4]) > 12):
            return None, math.inf
        trial = _orthogonal(points, candidate)
        if float(np.sum(weights*trial*trial)) <= float(np.sum(weights*residual*residual)):
            parameters = candidate
            damping = max(1e-9, damping*.3)
            if np.linalg.norm(step) < 1e-7:
                break
        else:
            damping = min(1e6, damping*8)
    # A circle has no identifiable major-axis angle, but its *geometry* is not
    # degenerate. Remove that gauge direction from the conditioning diagnostic.
    conditioned = weighted[:, :4] if abs(parameters[2]-parameters[3]) < .01 else weighted
    singular = np.linalg.svd(conditioned, compute_uv=False)
    condition = float(singular[0] / max(singular[-1], 1e-15))
    return parameters, condition


def _canonical(parameters):
    out = parameters.copy()
    if out[2] < out[3]:
        out[2], out[3] = out[3], out[2]
        out[4] += math.pi/2
    out[4] %= math.pi
    return out


def _serialize(parameters, origin, scale):
    parameters = _canonical(parameters)
    center = parameters[:2]*scale + origin
    axes = np.exp(parameters[2:4])*scale
    angle = float(parameters[4])
    rotation = _rotation(angle)
    q = rotation @ np.diag(1/(axes*axes)) @ rotation.T
    linear = -2*q @ center
    implicit = [q[0, 0], 2*q[0, 1], q[1, 1], linear[0], linear[1], center @ q @ center-1]
    return dict(center=center.tolist(), semi_axes=axes.tolist(), angle_rad=angle,
                angle_deg=math.degrees(angle), implicit=[float(x) for x in implicit])


def _angular_interval(angles):
    ordered = np.sort(angles % (2*math.pi))
    gaps = np.diff(np.r_[ordered, ordered[0]+2*math.pi])
    cut = int(np.argmax(gaps))
    start = float(ordered[(cut+1) % len(ordered)])
    unwrapped = (angles-start) % (2*math.pi) + start
    return start, float(np.max(unwrapped)), unwrapped


def _segment_runs(points, flags, value):
    indices = np.flatnonzero(flags == value)
    if not len(indices):
        return []
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1)
    return [np.round(points[run], 4).tolist() for run in runs if len(run) >= 2]


def _balanced_indices(points, bins=90, per_bin=3):
    """Prevent densely lit near-side pixels from voting away the dim far side."""
    centered = points-points.mean(axis=0)
    _, _, basis = np.linalg.svd(centered, full_matrices=False)
    coordinate = centered @ basis[0]
    width = max(float(np.ptp(coordinate)), 1e-9)
    sectors = np.minimum(bins-1, ((coordinate-coordinate.min())/width*bins).astype(int))
    selected = []
    for sector in range(bins):
        indices = np.flatnonzero(sectors == sector)
        if len(indices):
            selected.extend(indices[np.linspace(0, len(indices)-1, min(per_bin, len(indices))).astype(int)])
    return np.sort(np.asarray(selected, dtype=int))


def _inside_exclusions(points, polygons):
    inside = np.zeros(len(points), bool)
    for polygon in polygons:
        contour = np.asarray(polygon, np.float32).reshape(-1, 1, 2)
        if len(contour) >= 3:
            inside |= np.array([cv2.pointPolygonTest(contour, (float(p[0]), float(p[1])), False) >= 0
                                for p in points], dtype=bool)
    return inside


def fit_conic(arc, image_shape=None):
    """Fit an ellipse using only ``arc['points']`` measured outside exclusions.

    R=[[cos,-sin],[sin,cos]], p=c+R@[a*cos(t),b*sin(t)], a>=b. Positive
    angle rotates +x toward image +y (clockwise on screen). The interval is an
    unwrapped eccentric-anomaly interval, not a polar angle. ``model_supported``
    says nearby observations support that modeled portion; model points NEVER
    become measured data. ``sensitivity_models`` can propagate fitted-shape
    variation through a downstream line/conic intersection calculation.
    """
    if not isinstance(arc, dict) or not arc.get("valid"):
        return _invalid("observed_arc_unavailable")
    try:
        points = np.asarray(arc.get("conic_support_points", arc.get("points", [])), dtype=float)
    except (TypeError, ValueError):
        return _invalid("invalid_observed_points")
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 35 or not np.all(np.isfinite(points)):
        return _invalid("insufficient_finite_observed_points")
    exclusions = arc.get("exclusion_polygons", [])
    points = points[~_inside_exclusions(points, exclusions)]
    if len(points) < 35:
        return _invalid("insufficient_observations_outside_exclusion")
    # Equalize the number of samples per arc-distance bin: repeated/dense points
    # must not manufacture angular support or overwhelm another observed section.
    _, unique_indices = np.unique(np.round(points, 4), axis=0, return_index=True)
    points = points[np.sort(unique_indices)]
    if len(points) < 35:
        return _invalid("insufficient_unique_observed_points")
    origin = np.mean(points, axis=0)
    scale = float(np.sqrt(np.mean(np.sum((points-origin)**2, axis=1))))
    if scale < 12:
        return _invalid("observed_extent_too_small")
    normalized = (points-origin)/scale
    spread = np.linalg.svd(normalized, compute_uv=False)
    if spread[-1]/spread[0] < .025:
        return _invalid("insufficient_observed_curvature", spread_ratio=float(spread[-1]/spread[0]))
    sample_indices = _balanced_indices(normalized)
    sample = normalized[sample_indices]
    threshold_px = max(2.0, min(5.0, scale*.005))
    threshold = threshold_px/scale
    rng = np.random.default_rng(21917)
    hypotheses = [_direct(sample)]
    # Stratified samples retain both ends of a partial arc. A small island cannot
    # win merely because a five-point conic happens to pass through it exactly.
    blocks = np.array_split(np.arange(len(sample)), 8)
    for _ in range(72):
        chosen = [int(rng.choice(block)) for block in blocks if len(block)]
        hypotheses.append(_direct(sample[chosen]))
    best, best_score = None, -math.inf
    for hypothesis in hypotheses:
        if hypothesis is None:
            continue
        axes = np.exp(hypothesis[2:4])
        if min(axes) < .025 or max(axes) > 30 or min(axes)/max(axes) < .025:
            continue
        distance = np.abs(_sampson(sample, hypothesis))
        inliers = distance < threshold
        fraction = float(np.mean(inliers))
        score = fraction - .15*float(np.median(np.minimum(distance/threshold, 1)))
        if score > best_score:
            best_score, best = score, hypothesis
    if best is None:
        return _invalid("no_nondegenerate_ellipse_hypothesis")
    initial_inliers = np.abs(_sampson(sample, best)) < threshold*1.5
    if np.mean(initial_inliers) < .6:
        return _invalid("ellipse_inconsistent_with_observed_shape", inlier_fraction=float(np.mean(initial_inliers)))
    refined, condition = _geometric_refine(sample[initial_inliers], best, scale)
    if refined is None:
        return _invalid("geometric_refinement_failed")
    refined = _canonical(refined)
    residual, angles = _orthogonal(normalized, refined, return_angles=True)
    residual_px = residual*scale
    robust_sigma = max(.5, 1.4826*float(np.median(np.abs(residual_px-np.median(residual_px)))))
    inliers = np.abs(residual_px) < max(2., min(threshold_px*2, robust_sigma*3.5))
    fraction = float(np.mean(inliers))
    if fraction < .6 or np.sum(inliers) < 35:
        return _invalid("ellipse_inconsistent_with_observed_shape", inlier_fraction=fraction)
    rms = float(np.sqrt(np.mean(residual_px[inliers]**2)))
    start, end, supported_angles = _angular_interval(angles[inliers])
    span = end-start
    axes_px = np.exp(refined[2:4])*scale
    diagnostics = dict(method="normalized_direct_ellipse_robust_orthogonal_refinement",
                       support_source="paint_verified_segmented_support" if "conic_support_points" in arc else "observed_arc_points",
                       observed_count=len(points), inlier_count=int(np.sum(inliers)),
                       inlier_fraction=fraction, normalized_jacobian_condition=condition,
                       observed_angular_span_deg=math.degrees(span),
                       rms_orthogonal_residual_px=rms, robust_sigma_px=robust_sigma,
                       ransac_threshold_px=threshold_px)
    if span < math.radians(80):
        return _invalid("observed_arc_too_short_for_ellipse", **diagnostics)
    if span > math.radians(260):
        return _invalid("observations_do_not_form_a_single_visible_inner_arc", **diagnostics)
    if condition > 1e5 or not np.isfinite(condition):
        return _invalid("ill_conditioned_partial_ellipse", **diagnostics)
    if axes_px[1]/axes_px[0] < .035 or axes_px[0] > scale*18:
        return _invalid("degenerate_or_unbounded_ellipse", **diagnostics)
    if image_shape is not None:
        try:
            diagonal = float(np.hypot(image_shape[0], image_shape[1]))
        except (TypeError, IndexError):
            return _invalid("image_shape_must_start_with_height_width")
        if axes_px[0] > diagonal*2:
            return _invalid("ellipse_extent_inconsistent_with_image", **diagnostics)
    count = int(np.clip(math.ceil(span*axes_px[0]/2), 100, 1200))
    model_angles = np.linspace(start, end, count)
    model_local = np.column_stack([np.exp(refined[2])*np.cos(model_angles),
                                   np.exp(refined[3])*np.sin(model_angles)])
    model_normalized = model_local @ _rotation(refined[4]).T + refined[:2]
    model_points = model_normalized*scale+origin
    # Model support is restricted to input observed segments and accepted
    # residuals. Absent an explicit segmentation, large angular gaps split runs.
    ordered_angles = np.sort(supported_angles)
    angular_step = float(np.median(np.diff(ordered_angles)))
    gap_limit = max(angular_step*3, .004)
    support_intervals = []
    for segment in arc.get("conic_support_segments", arc.get("segments")) or [points.tolist()]:
        segment = np.asarray(segment, float)
        if segment.ndim != 2 or len(segment) < 2:
            continue
        segment_dist, segment_angles = _orthogonal((segment-origin)/scale, refined, True)
        good = np.abs(segment_dist*scale) < max(2., min(threshold_px*2, robust_sigma*3.5))
        sector = np.sort(((segment_angles[good]-start) % (2*math.pi))+start)
        sector = sector[(sector >= start-1e-8) & (sector <= end+1e-8)]
        if len(sector) < 2:
            continue
        for run in np.split(sector, np.flatnonzero(np.diff(sector) > gap_limit)+1):
            if len(run) >= 2:
                support_intervals.append([float(run[0]), float(run[-1])])
    supported = np.zeros(count, dtype=bool)
    for left, right in support_intervals:
        supported |= (model_angles >= left-angular_step) & (model_angles <= right+angular_step)
    # Angular-neighbor padding must never re-label the rope corridor as observed.
    supported &= ~_inside_exclusions(model_points, exclusions)
    # Remove consecutive angular blocks, not random isolated points: this probes
    # extrapolation/occlusion sensitivity that ordinary residuals cannot reveal.
    inlier_sample = normalized[inliers]
    # Equal *angular* sectors, not equal point counts: otherwise deleting one
    # block from unevenly lit data can delete an entire camera-side half while
    # another block deletes only a few degrees of the densely sampled highlight.
    edges = np.linspace(start, end+1e-9, 9)
    blocks = [np.flatnonzero((supported_angles >= edges[i]) &
                            (supported_angles < edges[i+1])) for i in range(8)]
    variants, displacements = [], []
    attempted_blocks = 0
    failed_blocks = 0
    for block in blocks:
        if len(block) < 3:
            continue
        attempted_blocks += 1
        keep = np.ones(len(inlier_sample), dtype=bool)
        keep[block] = False
        subsample = inlier_sample[keep]
        subsample = subsample[_balanced_indices(subsample)]
        alternative, alt_condition = _geometric_refine(subsample, refined, scale, iterations=18)
        if alternative is None or alt_condition > 1e6:
            failed_blocks += 1
            continue
        variants.append(_serialize(alternative, origin, scale))
        displacements.append(_orthogonal(model_normalized, alternative)*scale)
    diagnostics.update(
        leave_block_out_block_sample_counts=[int(len(block)) for block in blocks],
        leave_block_out_empty_blocks=sum(len(block) == 0 for block in blocks),
        leave_block_out_sparse_blocks=sum(0 < len(block) < 3 for block in blocks),
        leave_block_out_attempted_blocks=attempted_blocks,
        leave_block_out_failed_blocks=failed_blocks,
        leave_block_out_fit_count=len(variants),
        leave_block_out_success_fraction=len(variants)/max(attempted_blocks, 1))
    # A masked angular gap is intentionally unobserved, not a failed fit. Use
    # only populated sectors as the denominator while retaining both minimum
    # diversity (four sectors) and a substantial success fraction. The geometric
    # sensitivity gate below is unchanged and still rejects unstable support.
    if attempted_blocks < 4:
        return _invalid("insufficient_observed_angular_blocks", **diagnostics)
    if len(variants) < 4 or len(variants)/attempted_blocks < .75:
        return _invalid("insufficient_stable_leave_block_out_fits", **diagnostics)
    displacements = np.asarray(displacements)
    model_sensitivity = np.max(np.abs(displacements), axis=0)
    sensitivity = float(np.percentile(model_sensitivity, 95))
    diagnostics["leave_block_out_p95_displacement_px"] = sensitivity
    diagnostics["leave_block_out_max_displacement_px"] = float(np.max(model_sensitivity))
    # A drawable model can still be far too uncertain for a contact decision.
    # Keep that distinction explicit; reject genuinely unstable geometry rather
    # than disguising it with a smooth line or a low in-sample residual.
    if sensitivity > max(8., scale*.02):
        return _invalid("partial_ellipse_is_unstable_to_support_removal", **diagnostics)
    if rms > max(2., scale*.006):
        return _invalid("geometric_model_residual_too_large", **diagnostics)
    output = _serialize(refined, origin, scale)
    uncertain = sensitivity > max(2., scale*.004)
    output.update(valid=True, status="uncertain_model" if uncertain else "modeled_partial_ellipse",
                  reason="observed_support_with_explicit_modeled_gaps",
                  arc_interval_rad=[start, end], observed_angular_span_deg=math.degrees(span),
                  observed_angular_coverage_deg=math.degrees(float(np.sum(supported))*(span/max(count-1, 1))),
                  observed_angular_intervals_rad=support_intervals,
                  model_points=np.round(model_points, 4).tolist(), model_supported=supported.tolist(),
                  model_angles_rad=model_angles.tolist(), model_sensitivity_px=model_sensitivity.tolist(),
                  observed_model_segments=_segment_runs(model_points, supported, True),
                  modeled_segments=_segment_runs(model_points, supported, False),
                  inlier_points=np.round(points[inliers], 4).tolist(),
                  outlier_points=np.round(points[~inliers], 4).tolist(),
                  residual_px=rms, sensitivity_px=sensitivity, sensitivity_models=variants,
                  uncertainty_kind="leave_block_out_sensitivity_not_calibrated_accuracy",
                  exclusion_polygons=exclusions,
                  model_points_are_observations=False, diagnostics=diagnostics)
    return output
