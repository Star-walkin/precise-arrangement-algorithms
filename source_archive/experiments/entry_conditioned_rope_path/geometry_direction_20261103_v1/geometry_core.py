"""Original-image geometry search with an optional spool-direction soft term.

The supplied entry is an input condition. Nothing here imports a model, reads a
network response, uses a label, or assigns material identity. ``compute_fields``
returns arrays at ``field_scale``; ``track`` returns JSON-compatible records.
All path coordinates and optical widths are in original, unmirrored pixels.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


DEFAULTS = {
    "field_scale": 0.25,
    "rope_width_hint_px": 28.0,
    "field_border_widths": 0.8,
    "background_erosion_widths": 0.4,
    "background_min_coherence": 0.34,
    "background_min_energy": 0.002,
    "background_min_confidence": 0.28,
    "background_min_resultant": 0.72,
    "background_max_side_disagreement_deg": 27.0,
    "background_offsets_widths": [1.6, 2.5, 3.5],
    "direction_soft_cap": 0.08,
    "direction_parallel_neutral_deg": 12.0,
    "step_px": 12.0,
    "max_steps": 305,
    "beam_width": 8,
    "angle_offsets_deg": [-14.0, -7.0, 0.0, 7.0, 14.0],
    "width_candidates_px": [10.0, 14.0, 18.0, 23.0, 29.0, 36.0, 45.0],
    "width_factors": [0.86, 1.0, 1.16],
    "profile_positions": 33,
    "profile_tangent_offsets_widths": [-0.25, 0.0, 0.25],
    "profile_std_floor": 0.025,
    "seed_min_bilateral": 0.07,
    "support_bilateral_min": 0.16,
    "support_template_min": 0.43,
    "support_window_steps": 10,
    "minimum_recent_supported_fraction": 0.60,
    "max_consecutive_unsupported_px": 72.0,
    "max_total_unsupported_px": 240.0,
    "max_log_width_step": 0.20,
    "initial_forward_projection_min": -0.12,
    "merge_endpoint_px": 7.0,
    "merge_angle_deg": 12.0,
    "final_candidate_count": 3,
    "distinct_median_distance_px": 18.0,
    "selection_score_margin": 0.035,
    "minimum_supported_after_entry_px": 48.0,
    "edge_arrival_distance_px": 12.0,
}


def _cfg(config: dict | None) -> dict:
    return DEFAULTS | ({} if config is None else config)


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    if sigma < 0.45:
        return a
    return cv2.GaussianBlur(a, (0, 0), sigmaX=sigma, sigmaY=sigma,
                            borderType=cv2.BORDER_REFLECT_101)


def _sample(a: np.ndarray, xy: np.ndarray, *, border: float = 0.0) -> np.ndarray:
    """Bilinear samples; callers also check support against the real image."""
    p = np.asarray(xy, np.float32)
    shape = p.shape[:-1]
    q = p.reshape(-1, 2)
    out = np.empty(len(q), np.float32)
    for lo in range(0, len(q), 16000):
        block = q[lo:lo + 16000]
        out[lo:lo + len(block)] = cv2.remap(
            a, block[:, 0][None], block[:, 1][None], cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=float(border)).ravel()
    return out.reshape(shape)


def _tensor(gray: np.ndarray, pre_sigma: float, integration_sigma: float):
    smooth = _blur(gray, pre_sigma)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
    jxx = _blur(gx * gx, integration_sigma)
    jxy = _blur(gx * gy, integration_sigma)
    jyy = _blur(gy * gy, integration_sigma)
    energy = jxx + jyy
    anis = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4 * jxy ** 2, 0))
    coherence = np.clip(anis / np.maximum(energy, 1e-7), 0, 1)
    # The dominant tensor eigenvector is the EDGE NORMAL. Rotate by 90 deg
    # to obtain the undirected axis of a broad rope / groove structure.
    normal = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    tangent = normal + np.pi / 2
    return energy, coherence, np.cos(2 * tangent), np.sin(2 * tangent)


def compute_fields(image_bgr: np.ndarray, config: dict | None = None) -> dict:
    """Compute path-independent geometry and background directions.

    ``background_surface_mask`` is an optional same-size uint8/bool mask. It
    restricts *only* background direction availability; the free rope and the
    geometry response are computed across the full original image. If absent,
    direction assistance stays unknown everywhere. An original gray value of
    zero remains valid. The arrays are at the returned ``field_scale``.
    """
    c = _cfg(config)
    if (not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8
            or image_bgr.ndim != 3 or image_bgr.shape[2] != 3):
        raise ValueError("Expected original uint8 BGR image")
    scale = float(c["field_scale"])
    if not 0.1 <= scale <= 1.0:
        raise ValueError("field_scale must be in [0.1, 1]")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    width = max(4.0, float(c["rope_width_hint_px"])) * scale
    mean = _blur(small, max(2.0, 3.0 * width))
    variance = _blur((small - mean) ** 2, max(2.0, 3.0 * width))
    normalized = np.clip((small - mean) / np.maximum(np.sqrt(variance), 0.035), -4, 4)
    normalized = normalized.astype(np.float32)
    e, coh, _, _ = _tensor(normalized, max(0.55, 0.12 * width), max(1.0, 0.35 * width))
    response = (coh * np.sqrt(e / (e + 0.025))).astype(np.float32)
    be, bc, cos2, sin2 = _tensor(normalized, max(1.0, 0.34 * width), max(1.2, 0.85 * width))
    confidence = (bc * np.sqrt(be / (be + 0.02))).astype(np.float32)
    h, w = small.shape
    valid = np.ones((h, w), bool)
    margin = max(2, int(math.ceil(float(c["field_border_widths"]) * width)))
    valid[:margin] = False
    valid[-margin:] = False
    valid[:, :margin] = False
    valid[:, -margin:] = False
    supplied = c.get("background_surface_mask")
    if supplied is None:
        background_domain = np.zeros((h, w), bool)
        domain_source = "absent_direction_unknown"
    else:
        supplied = np.asarray(supplied)
        if supplied.shape != gray.shape:
            raise ValueError("background_surface_mask must match original image")
        background_domain = cv2.resize((supplied > 0).astype(np.uint8), (w, h),
                                       interpolation=cv2.INTER_NEAREST) > 0
        radius = max(1, int(round(float(c["background_erosion_widths"]) * width)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        background_domain = cv2.erode(background_domain.astype(np.uint8), kernel) > 0
        domain_source = "supplied_surface_mask_direction_only"
    background_valid = (valid & background_domain
                        & (bc >= float(c["background_min_coherence"]))
                        & (be >= float(c["background_min_energy"]))
                        & (confidence >= float(c["background_min_confidence"])))
    confidence[~background_valid] = 0.0
    response[~valid] = np.nan
    return {
        "gray": gray, "geometry_response": response,
        "valid_domain": valid, "background_domain": background_domain,
        "background_valid": background_valid,
        "direction_cos2": cos2.astype(np.float32),
        "direction_sin2": sin2.astype(np.float32),
        "direction_confidence": confidence,
        "field_scale": scale, "original_shape_hw": list(gray.shape),
        "background_domain_source": domain_source,
        "response_semantics": "path-independent original-gray tensor coherence and contrast; not probability",
        "response_display_range": [0.0, 1.0],
    }


def _field_sample(fields: dict, name: str, xy: np.ndarray) -> np.ndarray:
    scale = float(fields["field_scale"])
    p = (np.asarray(xy, np.float32) + 0.5) * scale - 0.5
    return _sample(fields[name].astype(np.float32, copy=False), p)


def _profiles(gray: np.ndarray, xy: np.ndarray, angles: np.ndarray,
              widths: np.ndarray, c: dict):
    q = np.asarray(xy, np.float32).reshape(-1, 2)
    a = np.broadcast_to(np.asarray(angles, np.float32), (len(q),))
    widths = np.broadcast_to(np.asarray(widths, np.float32), (len(q),))
    normal = np.stack([-np.sin(a), np.cos(a)], 1)
    tangent = np.stack([np.cos(a), np.sin(a)], 1)
    u = np.linspace(-1.25, 1.25, int(c["profile_positions"]), dtype=np.float32)
    t = np.asarray(c["profile_tangent_offsets_widths"], np.float32)
    points = (q[:, None, None, :] + normal[:, None, None, :] * widths[:, None, None, None] * u[None, None, :, None]
              + tangent[:, None, None, :] * widths[:, None, None, None] * t[None, :, None, None])
    h, w = gray.shape
    # A reflected, zero-padded or cropped sample is never original evidence.
    valid = ((points[..., 0] >= 1) & (points[..., 0] <= w - 2)
             & (points[..., 1] >= 1) & (points[..., 1] <= h - 2)).all((1, 2))
    sampled = _sample(gray, points).mean(1)
    return sampled, valid


def _profile_evidence(profiles: np.ndarray, template: np.ndarray | None, c: dict):
    u = np.linspace(-1.25, 1.25, profiles.shape[1])
    std = np.maximum(profiles.std(1), float(c["profile_std_floor"]))
    li = profiles[:, (u >= -0.35) & (u <= -0.15)].mean(1)
    lo = profiles[:, (u >= -0.90) & (u <= -0.65)].mean(1)
    ri = profiles[:, (u >= 0.15) & (u <= 0.35)].mean(1)
    ro = profiles[:, (u >= 0.65) & (u <= 0.90)].mean(1)
    dl = (li - lo) / std
    dr = (ro - ri) / std
    bilateral = np.where(dl * dr < 0, np.clip(np.minimum(abs(dl), abs(dr)) / 2.0, 0, 1), 0)
    z = (profiles - profiles.mean(1, keepdims=True)) / std[:, None]
    signed = np.zeros(len(z)) if template is None else (z * template).mean(1)
    # The same paired flanks can reverse polarity under changing metal glare.
    # Preserve the sign for diagnosis, but compare the normalized shape without
    # forcing the rope to remain brighter or darker than its surroundings.
    corr = np.clip(np.abs(signed), 0, 1)
    return bilateral, corr, z, signed


def _seed_profile(gray: np.ndarray, seed: np.ndarray, c: dict):
    vec = seed[-1] - seed[0]
    angle = math.atan2(float(vec[1]), float(vec[0]))
    centers = seed[0] + np.linspace(0.12, 0.88, 7)[:, None] * vec
    choices = []
    for width in c["width_candidates_px"]:
        prof, valid = _profiles(gray, centers, np.full(7, angle), np.full(7, width), c)
        two, _, z, _ = _profile_evidence(prof, None, c)
        two[~valid] = 0
        choices.append((float(np.median(two)), float(width), z.mean(0), prof.mean(0), int(valid.sum())))
    best = max(choices, key=lambda x: x[0])
    template = best[2] / max(float(np.std(best[2])), 0.25)
    receipt = {"optical_width_px": best[1], "median_bilateral": best[0],
               "valid_profiles": best[4], "profile_mean": best[3].tolist(),
               "width_candidates": [{"width_px": q[1], "median_bilateral": q[0]} for q in choices],
               "seed_is_input_not_new_detection": True}
    return best[1], template.astype(np.float32), receipt


def _background_direction(fields: dict, q: np.ndarray, angle: float,
                          width: float, c: dict) -> dict:
    normal = np.array([-math.sin(angle), math.cos(angle)])
    tangent = np.array([math.cos(angle), math.sin(angle)])
    offsets = np.asarray(c["background_offsets_widths"], float) * width
    along = np.array([-2.0, 0.0, 2.0]) * width
    positive = q + offsets[:, None, None] * normal + along[None, :, None] * tangent
    negative = q - offsets[:, None, None] * normal + along[None, :, None] * tangent
    samples = np.concatenate([positive.reshape(-1, 2), negative.reshape(-1, 2)])
    valid = _field_sample(fields, "background_valid", samples) > 0.8
    conf = _field_sample(fields, "direction_confidence", samples)
    z = _field_sample(fields, "direction_cos2", samples) + 1j * _field_sample(fields, "direction_sin2", samples)
    side = []
    side_size = len(offsets) * len(along)
    for take in (slice(0, side_size), slice(side_size, 2 * side_size)):
        use = valid[take] & (conf[take] >= c["background_min_confidence"])
        if use.sum() < 4:
            return {"known": False, "reason": "insufficient_bilateral_background_support", "bonus": 0.0}
        weights = conf[take][use]
        mean = np.sum(weights * z[take][use]) / np.sum(weights)
        if abs(mean) < c["background_min_resultant"]:
            return {"known": False, "reason": "mixed_directions_on_one_side", "bonus": 0.0}
        side.append((mean / abs(mean), float(np.mean(weights)), float(abs(mean))))
    disagreement = 0.5 * math.degrees(math.acos(float(np.clip(np.real(side[0][0] * np.conj(side[1][0])), -1, 1))))
    if disagreement > c["background_max_side_disagreement_deg"]:
        return {"known": False, "reason": "opposite_sides_disagree", "side_disagreement_deg": disagreement, "bonus": 0.0}
    mean = side[0][0] * side[0][1] + side[1][0] * side[1][1]
    mean /= max(abs(mean), 1e-9)
    candidate = complex(math.cos(2 * angle), math.sin(2 * angle))
    relative = 0.5 * math.degrees(math.acos(float(np.clip(np.real(candidate * np.conj(mean)), -1, 1))))
    confidence = min(side[0][1], side[1][1]) * min(side[0][2], side[1][2])
    return {"known": True, "axis_angle_deg": float(math.degrees(np.angle(mean)) / 2 % 180),
            "relative_angle_deg": relative, "confidence": float(np.clip(confidence, 0, 1)),
            "side_disagreement_deg": disagreement, "bonus": 0.0}


def _node_rank(node: dict) -> float:
    scores = node["scores"]
    if not scores:
        return -1e9
    avg = float(np.mean(scores))
    tail = float(np.mean(scores[-min(24, len(scores)):]))
    supported = node["supported_px"]
    total = supported + node["unsupported_px"]
    return 0.65 * avg + 0.35 * tail + 0.035 * min(supported / 1800.0, 1.0) - 0.08 * node["unsupported_px"] / max(total, 1.0)


def _path_distance(a: list, b: list) -> float:
    aa = np.asarray(a, float)
    bb = np.asarray(b, float)
    n = max(len(aa), len(bb))
    ix = np.arange(max(0, n - 80), n)
    return float(np.median(np.linalg.norm(aa[np.minimum(ix, len(aa) - 1)] - bb[np.minimum(ix, len(bb) - 1)], axis=1)))


def _public_config(c: dict) -> dict:
    return {k: v for k, v in c.items() if k not in ("background_surface_mask", "edge_points_xy")
            and isinstance(v, (str, int, float, bool, list, type(None)))}


def track(fields: dict, image_bgr: np.ndarray, seed_xy: Any,
          config: dict | None = None, use_direction: bool = False) -> dict:
    """Bounded G0/G1 search; direction affects score only, never generation/gates."""
    c = _cfg(config)
    if image_bgr.shape[:2] != tuple(fields["original_shape_hw"]):
        raise ValueError("Image and fields differ")
    try:
        seed = np.asarray(seed_xy, float).reshape(-1, 2)
    except (TypeError, ValueError):
        seed = np.empty((0, 2), float)
    failure = lambda reason: {"selected": None, "candidates": [], "status": "UNKNOWN", "stop_reason": reason,
                              "use_direction": bool(use_direction), "input_seed_xy": seed.tolist(),
                              "configuration": _public_config(c), "material_identity_verified": False}
    if len(seed) < 2 or not np.isfinite(seed).all():
        return failure("entry_missing_or_invalid")
    vec = seed[-1] - seed[0]
    length = float(np.linalg.norm(vec))
    if length < 4:
        return failure("entry_too_short")
    gray = fields["gray"]
    h, w = gray.shape
    bbox = np.asarray(c.get("bbox_xyxy", [0, 0, w, h]), float)
    if bbox.shape != (4,) or not (0 <= bbox[0] < bbox[2] <= w and 0 <= bbox[1] < bbox[3] <= h):
        raise ValueError("Invalid original-coordinate search bbox")
    if np.any(seed[:, 0] < bbox[0]) or np.any(seed[:, 0] >= bbox[2]) or np.any(seed[:, 1] < bbox[1]) or np.any(seed[:, 1] >= bbox[3]):
        return failure("entry_outside_fixed_search_bbox")
    width, template, seed_receipt = _seed_profile(gray, seed, c)
    if seed_receipt["median_bilateral"] < c["seed_min_bilateral"]:
        out = failure("weak_entry_original_profile")
        out["seed_measurement"] = seed_receipt
        return out
    a0 = math.atan2(float(vec[1]), float(vec[0]))
    step = float(c["step_px"])
    initial = np.linspace(seed[0], seed[-1], max(2, int(math.ceil(length / step)) + 1)).tolist()
    root = {"points": initial, "angle": a0, "width": width, "scores": [], "evidence": [],
            "weak_px": 0.0, "supported_px": 0.0, "unsupported_px": 0.0, "reason": None}
    beam = [root]
    finished = []
    angle_offsets = np.deg2rad(c["angle_offsets_deg"])
    edges = np.asarray(c.get("edge_points_xy", []), float).reshape(-1, 2)
    iterations = 0
    for iteration in range(int(c["max_steps"])):
        proposals = []
        for parent in beam:
            for da in angle_offsets:
                a = parent["angle"] + float(da)
                if math.cos(a - a0) < c["initial_forward_projection_min"]:
                    continue
                q = np.asarray(parent["points"][-1]) + step * np.array([math.cos(a), math.sin(a)])
                if q[0] < bbox[0] + 2 or q[0] >= bbox[2] - 2 or q[1] < bbox[1] + 2 or q[1] >= bbox[3] - 2:
                    stopped = dict(parent)
                    stopped["reason"] = ("image_border" if q[0] < 2 or q[0] >= w - 2 or q[1] < 2 or q[1] >= h - 2
                                         else "search_bbox_boundary_unknown")
                    finished.append(stopped)
                    continue
                proposals.append((parent, q, a, float(da)))
        if not proposals:
            break
        q = np.stack([item[1] for item in proposals])
        angles = np.array([item[2] for item in proposals])
        widths = np.array([item[0]["width"] for item in proposals])
        factors = np.asarray(c["width_factors"], float)
        pq = np.repeat(q, len(factors), axis=0)
        pa = np.repeat(angles, len(factors))
        pw = (widths[:, None] * factors[None, :]).reshape(-1)
        prof, valid = _profiles(gray, pq, pa, pw, c)
        two, corr, _, signed = _profile_evidence(prof, template, c)
        n = len(proposals)
        two = two.reshape(n, -1)
        corr = corr.reshape(n, -1)
        signed = signed.reshape(n, -1)
        valid = valid.reshape(n, -1)
        width_step = np.abs(np.log(factors))[None, :]
        width_choice = 0.70 * two + 0.30 * corr - 0.04 * width_step
        width_choice[~valid] = -1e6
        chosen = np.argmax(width_choice, axis=1)
        take = np.arange(n)
        bilateral = two[take, chosen]
        similarity = corr[take, chosen]
        signed_similarity = signed[take, chosen]
        width_valid = valid[take, chosen]
        chosen_width = pw[take * len(factors) + chosen]
        resp = _field_sample(fields, "geometry_response", q)
        field_valid = _field_sample(fields, "valid_domain", q) > 0.9
        resp = np.where(field_valid & np.isfinite(resp), np.clip(resp, 0, 1), 0.0)
        pool = []
        for i, (parent, pos, angle, da) in enumerate(proposals):
            # All support and stop decisions are identical in G0 and G1.
            # A strong two-flank response alone can be a flange, tool edge or
            # neighboring course. Require both the paired flanks and a
            # polarity-invariant match to the measured entry profile.
            supported = bool(width_valid[i]
                             and bilateral[i] >= c["support_bilateral_min"]
                             and similarity[i] >= c["support_template_min"])
            weak = 0.0 if supported else parent["weak_px"] + step
            total_weak = parent["unsupported_px"] + (0.0 if supported else step)
            curvature = abs(da) / max(float(np.max(abs(angle_offsets))), 1e-6)
            width_change = abs(math.log(float(chosen_width[i]) / parent["width"]))
            base = (0.55 * float(bilateral[i]) + 0.25 * float(similarity[i])
                    + 0.12 * float(resp[i]) - 0.06 * curvature - 0.04 * width_change)
            direction = {"known": False, "reason": "G0_disabled", "bonus": 0.0}
            bonus = 0.0
            if use_direction:
                direction = _background_direction(fields, pos, angle, float(chosen_width[i]), c)
                if (supported and direction["known"]
                        and direction["relative_angle_deg"] > c["direction_parallel_neutral_deg"]):
                    relative = math.radians(direction["relative_angle_deg"])
                    quality = min(1.0, max(0.0, 0.7 * bilateral[i] + 0.3 * similarity[i]))
                    cap = min(0.08, max(0.0, float(c["direction_soft_cap"])))
                    bonus = float(cap * direction["confidence"] * quality * math.sin(relative) ** 2)
                direction["bonus"] = bonus
            terms = {"bilateral": float(bilateral[i]), "entry_profile_similarity": float(similarity[i]),
                     "entry_profile_signed_similarity": float(signed_similarity[i]),
                     "entry_profile_polarity_flipped": bool(signed_similarity[i] < 0),
                     "geometry_response": float(resp[i]), "curvature_penalty": float(0.06 * curvature),
                     "width_change_penalty": float(0.04 * width_change), "direction_soft": bonus,
                     "base_score": base, "total_step_score": base + bonus,
                     "profile_inside_original_image": bool(width_valid[i]),
                     "optical_width_px": float(chosen_width[i]), "supported": supported,
                     "background_direction": direction}
            node = {"points": parent["points"] + [pos.tolist()], "angle": angle,
                    "width": float(chosen_width[i]), "scores": parent["scores"] + [base + bonus],
                    "evidence": parent["evidence"] + [terms], "weak_px": weak,
                    "supported_px": parent["supported_px"] + (step if supported else 0.0),
                    "unsupported_px": total_weak, "reason": None}
            window = int(c["support_window_steps"])
            recent = node["evidence"][-window:]
            recent_fraction = float(np.mean([ev["supported"] for ev in recent]))
            terms["recent_supported_fraction"] = recent_fraction
            if len(node["evidence"]) >= window and recent_fraction < c["minimum_recent_supported_fraction"]:
                node["reason"] = "recent_structure_support_collapsed"
            if node["reason"] is None and supported and len(edges) and node["supported_px"] >= 48:
                if float(np.min(np.linalg.norm(edges - pos, axis=1))) <= c["edge_arrival_distance_px"]:
                    node["reason"] = "observed_edge_candidate_reached_identity_unconfirmed"
            if node["reason"] is None and weak > c["max_consecutive_unsupported_px"]:
                node["reason"] = "unsupported_gap_limit"
            if node["reason"] is None and total_weak > c["max_total_unsupported_px"]:
                node["reason"] = "total_unsupported_budget"
            (finished if node["reason"] else pool).append(node)
        iterations = iteration + 1
        pool.sort(key=_node_rank, reverse=True)
        next_beam = []
        keys = set()
        for node in pool:
            p = node["points"][-1]
            key = (round(p[0] / c["merge_endpoint_px"]), round(p[1] / c["merge_endpoint_px"]),
                   round(math.degrees(node["angle"]) / c["merge_angle_deg"]))
            if key in keys:
                continue
            keys.add(key)
            next_beam.append(node)
            if len(next_beam) >= c["beam_width"]:
                break
        beam = next_beam
        if len(finished) > 96:
            finished = sorted(finished, key=_node_rank, reverse=True)[:96]
        if not beam:
            break
    for node in beam:
        node["reason"] = "arc_length_budget_unresolved"
        finished.append(node)
    if not finished:
        out = failure("no_viable_geometric_proposal")
        out["seed_measurement"] = seed_receipt
        return out
    finished.sort(key=_node_rank, reverse=True)
    distinct = []
    for node in finished:
        if all(_path_distance(node["points"], other["points"]) >= c["distinct_median_distance_px"] for other in distinct):
            distinct.append(node)
        if len(distinct) >= c["final_candidate_count"]:
            break
    rows = []
    for index, node in enumerate(distinct):
        statuses = ["input_entry"] * len(initial) + ["image_supported" if ev["supported"] else "unsupported_gap"
                                                     for ev in node["evidence"]]
        rows.append({"id": f"g{index}", "points_xy": node["points"], "point_status": statuses,
                     "evidence": node["evidence"], "rank_score": _node_rank(node),
                     "termination": node["reason"], "supported_after_entry_px": node["supported_px"],
                     "unsupported_after_entry_px": node["unsupported_px"],
                     "maximum_gap_limit_px": c["max_consecutive_unsupported_px"],
                     "input_entry_length_px": length, "optical_width_px": node["width"],
                     "material_identity_verified": False})
    margin = rows[0]["rank_score"] - rows[1]["rank_score"] if len(rows) > 1 else None
    enough = rows[0]["supported_after_entry_px"] >= c["minimum_supported_after_entry_px"]
    ambiguous = bool(margin is not None and margin < c["selection_score_margin"])
    if not enough:
        status, reason, selected = "UNKNOWN", "insufficient_supported_path_after_input_entry", None
    elif ambiguous:
        status, reason, selected = "AMBIGUOUS", "distinct_candidates_not_separable", None
    else:
        status, reason, selected = "PARTIAL", rows[0]["termination"], rows[0]
        if reason == "observed_edge_candidate_reached_identity_unconfirmed":
            status = "EDGE_CANDIDATE_IDENTITY_UNCONFIRMED"
    return {"selected": selected, "candidates": rows, "status": status, "stop_reason": reason,
            "use_direction": bool(use_direction), "input_seed_xy": seed.tolist(),
            "seed_measurement": seed_receipt, "score_margin": margin, "iterations": iterations,
            "background_domain_source": fields["background_domain_source"],
            "configuration": _public_config(c), "material_identity_verified": False,
            "direction_alters_only_score": True,
            "support_definition": "original full-resolution two-flank/profile evidence; no network gate",
            "response_definition": fields["response_semantics"]}
