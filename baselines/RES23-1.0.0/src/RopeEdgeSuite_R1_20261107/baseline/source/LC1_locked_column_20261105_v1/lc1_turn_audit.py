"""Experimental local/multi-model turn audit; NOT the LC1 default estimator.

The free straight line anchors a tangent/normal coordinate system. Fits use
only 3..6 pitch of curved support near a candidate C1 transition, plus local
free support. Quadratic and cubic hypotheses must agree. This is a conditional
image-geometry audit, not proof of contact or calibrated uncertainty coverage.
No real data/model is loaded and no thresholds are learned in this module.
"""
import math
import numpy as np

from lc1_geometry import fit_line, normalize_line


def _mad(a):
    return float(1.4826*np.median(np.abs(a-np.median(a))))


def _huber(residual, sigma):
    delta = 1.5*sigma
    a = np.abs(residual)
    return np.where(a <= delta, residual**2, 2*delta*a-delta**2)


def _regress(design, v, sigma):
    beta = np.linalg.lstsq(design, v, rcond=None)[0]
    for _ in range(6):
        residual = v-design@beta
        weights = np.minimum(1., 1.5*sigma/np.maximum(np.abs(residual), 1e-12))
        root = np.sqrt(weights)
        updated = np.linalg.lstsq(design*root[:, None], v*root, rcond=None)[0]
        if np.max(np.abs(updated-beta)) < 1e-9:
            beta = updated
            break
        beta = updated
    return beta, v-design@beta


def _reject(reason, **kwargs):
    result = dict(valid=False, x=None, y=None, uncertainty_px=None, quality=0.,
                  reason=reason, experimental=True, default_estimator_changed=False,
                  physical_contact_validated=False, anchor_uncertainty_included=False)
    result.update(kwargs)
    return result


def audit_turn_local(points, free_line_abc=None, free_side="right", pitch_px=20.,
                     windows_pitch=(3., 4.5, 6.), max_disagreement_pitch=.40):
    """Return conditional local quadratic/cubic transition consensus.

    A supplied free_line_abc is preferable to a line estimated from the far
    quarter of this path. Parameter uncertainty is conditional on that anchor;
    anchor error is NOT silently treated as known to be zero in interpretation.
    The prototype rejects missing branch evidence, weak curvature, uncertain
    transitions, and >0.4 pitch model/window disagreement. It never substitutes
    a straight-segment optical endpoint for the curved transition.
    """
    if free_side not in ("right", "left"):
        raise ValueError("free_side must be right or left")
    try:
        p = np.asarray(points, float)
        pitch = float(pitch_px)
        windows = sorted(set(float(w) for w in windows_pitch))
        if (p.ndim != 2 or p.shape[1] != 2 or len(p) < 32 or
                not np.isfinite(p).all() or not np.isfinite(pitch) or pitch <= 0 or
                len(windows) < 2 or min(windows) < 3 or max(windows) > 6):
            return _reject("invalid_points_pitch_or_windows")
    except (ValueError, TypeError):
        return _reject("invalid_points_pitch_or_windows")
    p = p[np.argsort(p[:, 0], kind="stable")]
    far_n = max(12, int(math.ceil(.25*len(p))))
    far = p[-far_n:] if free_side == "right" else p[:far_n]
    if np.ptp(far[:, 0]) < 2*pitch:
        return _reject("insufficient_far_free_anchor_span")
    if free_line_abc is None:
        line = fit_line(far)
        if not line["valid"]:
            return _reject("free_anchor_fit_failed")
        abc = line["abc"]
        source = "far_quarter_path_TLS_conditional_anchor"
    else:
        try:
            abc = normalize_line(free_line_abc)
        except (ValueError, TypeError):
            return _reject("invalid_supplied_anchor")
        source = "supplied_independent_free_line"
    a, b, c = abc
    if abs(b) < .10:
        return _reject("near_vertical_anchor")
    tangent = np.array([b, -a])
    normal = np.array([a, b])
    uv = np.column_stack([p@tangent, p@normal+c])
    uv = uv[np.argsort(uv[:, 0], kind="stable")]
    # Spatial bins prevent a densely rasterized contour from manufacturing
    # independent precision. Retain the median observed u/v in each bin.
    resolution = max(2., .25*pitch)
    bin_ids = np.floor((uv[:, 0]-uv[0, 0])/resolution).astype(int)
    uv = np.array([[np.median(uv[bin_ids == k, 0]), np.median(uv[bin_ids == k, 1])]
                   for k in np.unique(bin_ids)])
    u, v = uv.T
    far_v = far@normal+c
    sigma = max(.35, _mad(far_v))
    anchor_rmse = float(np.sqrt(np.mean(far_v**2)))
    if anchor_rmse > max(1.5, .15*pitch):
        return _reject("far_free_anchor_not_supported", anchor_rmse_px=anchor_rmse)
    if np.max(np.abs(v)) < max(2., .15*pitch, 3*sigma):
        return _reject("straight_or_unobservable_curvature", anchor_rmse_px=anchor_rmse)
    bounds = ((float(u.min()+3*pitch), float(u.max()-2*pitch)) if free_side == "right"
              else (float(u.min()+2*pitch), float(u.max()-3*pitch)))
    if bounds[0] >= bounds[1]:
        return _reject("insufficient_local_branch_span")
    candidates = []

    for window_pitch in windows:
        length = window_pitch*pitch
        for degree in (2, 3):
            def fit_at(join):
                h = u-join
                curve = ((h < 0) & (h >= -length) if free_side == "right"
                         else (h > 0) & (h <= length))
                free = ((h >= 0) & (h <= length) if free_side == "right"
                        else (h <= 0) & (h >= -length))
                if (curve.sum() < 8 or free.sum() < 8 or
                        np.ptp(u[curve]) < .75*length or np.ptp(u[free]) < 2*pitch):
                    return None
                z = h[curve]/length
                design = np.column_stack([z**power for power in range(2, degree+1)])
                beta, residual = _regress(design, v[curve], sigma)
                # Both losses are in physical normal pixels; each observed
                # branch contributes equally and no unsupported gap is filled.
                score = float(.5*np.mean(_huber(residual, sigma)) +
                              .5*np.mean(_huber(v[free], sigma)))
                free_fraction = float(np.mean(np.abs(v[free]) <= max(1., 3*sigma)))
                rmse = float(np.sqrt(np.mean(residual**2)))
                sign = -1. if free_side == "right" else 1.
                end_deviation = abs(float(sum(beta[k]*sign**(k+2) for k in range(len(beta)))))
                # A local circular/elliptical arc has one curvature sign.
                # Noise-fitted cubic inflections cannot certify the contact.
                curvature_ends = [2*beta[0], 2*beta[0]+(6*beta[1]*sign if degree == 3 else 0)]
                convex = curvature_ends[0]*curvature_ends[1] > 0
                return dict(join=float(join), score=score, beta=beta, rmse=rmse,
                            curve_points=int(curve.sum()), free_points=int(free.sum()),
                            free_fraction=free_fraction, end_deviation=end_deviation,
                            convex=bool(convex))

            fits = [r for join in np.linspace(*bounds, 101) if (r := fit_at(join)) is not None]
            admissible = [r for r in fits if r["free_fraction"] >= .90 and
                          r["end_deviation"] >= max(2., .12*pitch, 3*sigma) and r["convex"]]
            if not admissible:
                candidates.append(dict(degree=degree, window_pitch=window_pitch,
                                       valid=False, reason="no_supported_local_curve"))
                continue
            best = min(admissible, key=lambda r: r["score"])
            step = (bounds[1]-bounds[0])/100
            refined = [r for join in np.linspace(max(bounds[0], best["join"]-step),
                                                   min(bounds[1], best["join"]+step), 31)
                       if (r := fit_at(join)) is not None]
            fits.extend(refined)
            admissible.extend(r for r in refined if r["free_fraction"] >= .90 and
                              r["end_deviation"] >= max(2., .12*pitch, 3*sigma) and r["convex"])
            best = min(admissible, key=lambda r: r["score"])
            # A 3-sigma profile-loss diagnostic with independent spatial bins.
            # Include unconstrained neighboring joins, not only those that
            # already passed curvature/sign gates, to avoid conditional shrinkage.
            n_eff = min(best["curve_points"], best["free_points"])
            tolerance = 9*sigma**2/max(n_eff, 1)
            # Use the connected profile basin of THIS observed transition.
            # A distant all-free window can fit q=0 with zero loss but represents
            # no observed turn; it must not be pooled as a second confidence
            # interval endpoint across an intervening high-loss region.
            by_join = {}
            for r in fits:
                if r["join"] not in by_join or r["score"] < by_join[r["join"]]["score"]:
                    by_join[r["join"]] = r
            ordered = sorted(by_join.values(), key=lambda r: r["join"])
            k = min(range(len(ordered)), key=lambda j: abs(ordered[j]["join"]-best["join"]))
            left = right = k
            while (left and ordered[left-1]["convex"] and
                   ordered[left-1]["score"] <= best["score"]+tolerance):
                left -= 1
            while (right+1 < len(ordered) and ordered[right+1]["convex"] and
                   ordered[right+1]["score"] <= best["score"]+tolerance):
                right += 1
            support = [r["join"] for r in ordered[left:right+1]]
            join_unc = max(abs(best["join"]-min(support)), abs(max(support)-best["join"]), step/15)
            xy = tangent*best["join"]-c*normal
            x_unc = abs(b)*join_unc
            valid = best["rmse"] <= max(1.5, .15*pitch) and x_unc <= .50*pitch
            candidates.append(dict(degree=degree, window_pitch=window_pitch,
                valid=bool(valid), x=float(xy[0]), y=float(xy[1]), uncertainty_px=float(x_unc),
                rmse=best["rmse"], normal_deviation_px=best["end_deviation"],
                free_fraction=best["free_fraction"], curve_points=best["curve_points"],
                free_points=best["free_points"], score=best["score"],
                beta_local_normal=best["beta"].tolist(), reason="ok" if valid else "local_fit_uncertain_or_poor"))

    common = dict(models=candidates, free_line_abc=abc, anchor_source=source,
                  anchor_rmse_px=anchor_rmse, independent_spatial_bin_px=resolution,
                  uncertainty_method="conditional_fixed_anchor_3sigma_profile_loss_spatial_bins",
                  profile_model_constraint="one_curvature_sign_local_round_surface",
                  max_model_disagreement_px=max_disagreement_pitch*pitch,
                  experimental=True, default_estimator_changed=False,
                  physical_contact_validated=False, anchor_uncertainty_included=False)
    valid_models = [r for r in candidates if r["valid"]]
    if any(sum(r["degree"] == degree for r in valid_models) < 2 for degree in (2, 3)):
        return _reject("insufficient_two_model_two_window_support", **common)
    xs = np.array([r["x"] for r in valid_models])
    spread = float(np.ptp(xs))
    if spread > max_disagreement_pitch*pitch:
        return _reject("local_model_or_window_disagreement", model_disagreement_px=spread, **common)
    x, y = float(np.median(xs)), float(np.median([r["y"] for r in valid_models]))
    uncertainty = max(r["uncertainty_px"]+abs(r["x"]-x) for r in valid_models)
    if uncertainty > .50*pitch:
        return _reject("consensus_uncertainty_exceeds_half_pitch", model_disagreement_px=spread, **common)
    quality = min(max(0., 1-r["rmse"]/max(1.5, .15*pitch)) for r in valid_models)
    quality *= max(0., 1-uncertainty/pitch)
    return dict(valid=True, x=x, y=y, uncertainty_px=float(uncertainty), quality=float(quality),
                reason="conditional_local_multimodel_consensus", model_disagreement_px=spread, **common)
