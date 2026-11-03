"""固定开放图上的单帧自由活动绳直段检测，纯离线、无设备导入。

ONNX 语义图只提出候选；位置、方向和光学宽度来自未增强原图的双侧
半峰值剖面。所有坐标返回输入原朝向的 (x, y) 原图像素。直段支持端点
不是入卷切点，光学宽度不是毫米绳径，单帧不能证明同根材料身份。
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import cv2
import numpy as np


# 仅导入已经审阅的纯视觉模块；两者在导入时均不初始化设备。
_ROOT = next(path for path in Path(__file__).resolve().parents
             if (path / "vision_geometry_lab/rope_detector.py").is_file())
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "vision_prototype"))
from vision_geometry_lab import rope_detector  # noqa: E402
from semantic_locator import semantic_locator_for_camera  # noqa: E402


# 本模块固定门槛；后续变更须另开版本并在运行上下文记录源码哈希。
SEMANTIC_THRESHOLD = .50
SEMANTIC_MEAN_MIN = .55
SEMANTIC_COMPONENT_AREA_MIN = 30
SEMANTIC_COMPONENT_MAX = 6
SEMANTIC_X_SPAN_MIN_PX = 120.
SEMANTIC_ANGLE_RANGE_DEG = (-60., 12.)  # 左相机先镜像到历史统一朝向。
SEMANTIC_OBSERVED_LENGTH_MIN_PX = 140.
SEMANTIC_OUTSIDE_SURFACE_MIN_FRACTION = .12
SEMANTIC_OUTSIDE_SURFACE_MIN_LENGTH_PX = 75.
COMPETITOR_SCORE_RATIO_LIMIT = .82
RESAMPLE_STEP_PX = 6.
RESAMPLE_MIN_POINTS = 12
RESAMPLE_MIN_SPAN_PX = 105.
FALLBACK_MIN_CONFIDENCE = .82
FALLBACK_MIN_LENGTH_PX = 250.
FALLBACK_MIN_OUTSIDE_FRACTION = .38
FALLBACK_MAX_COMPETITOR_RATIO = .55
SURFACE_RAY_MAX_PX = 1800.
SURFACE_RAY_STEP_PX = 8.
BACKGROUND_YELLOW_MIN_FRACTION = .35
BACKGROUND_NEAR_SURFACE_MAX_PX = 95.
OUTSIDE_RETRY_GUARD_PX = 20.
OUTSIDE_RETRY_BOX_MARGIN_PX = 15
OUTSIDE_RETRY_MAX_PER_SEMANTIC_COMPONENT = 1


def _unknown(reason: str, camera_role: str, diagnostics: dict | None = None) -> dict:
    """失败也返回同一字段契约；统计400分母时不得丢弃此记录。"""
    return {"status": "unknown", "reason": reason, "camera_role": camera_role,
            "coordinate_system": "original_unmirrored_image_xy_pixels",
            "observed_segment_xy": None, "center_xy": None, "direction_xy": None,
            "toward_surface_direction_xy": None, "toward_surface_status": "unknown",
            "toward_surface_candidates_xy": [], "normal_xy": None, "rho_px": None,
            "angle_deg": None, "width_px": None, "width_interval_px": None,
            "observed_points_xy": [], "boundary_left_xy": [], "boundary_right_xy": [],
            "observed_points_kind": "none", "boundary_points_kind": "none",
            "support_length_px": 0., "observed_support": 0,
            "angular_uncertainty_deg": None, "normal_uncertainty_px": None,
            "uncertainty_kind": "empirical_not_calibrated", "confidence": 0.,
            "width_kind": "optical_half_prominence_not_physical_diameter",
            "source": "none", "complete_free_rope_claim": False,
            "contact_or_tangency_claim": False, "diagnostics": diagnostics or {}}


def _validate(image: np.ndarray, surface: np.ndarray, role: str, prior: dict | None) -> str:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image_bgr must be uint8 HxWx3")
    if not isinstance(surface, np.ndarray) or surface.shape != image.shape[:2] or surface.dtype != np.uint8:
        raise ValueError("surface_mask must be same-size uint8 HxW")
    if np.any((surface != 0) & (surface != 255)):
        raise ValueError("surface_mask must contain only 0/255")
    if np.count_nonzero(surface) < 100:
        raise ValueError("surface_mask is empty or implausibly small")
    role = role.replace("_camera", "").strip().lower()
    if role not in {"left", "right"}:
        raise ValueError("camera_role must be left or right")
    if prior is not None and not isinstance(prior, dict):
        raise ValueError("prior must be None or a dictionary")
    return role


def _fit_component(prob: np.ndarray, component: np.ndarray,
                   crop: tuple[int, int, int, int]) -> dict:
    yy, xx = np.nonzero(component)
    x0, y0, x1, y1 = crop
    sx = (x1 - x0) / prob.shape[1]
    sy = (y1 - y0) / prob.shape[0]
    xy = np.column_stack((x0 + (xx + .5) * sx, y0 + (yy + .5) * sy)).astype(np.float32)
    vx, vy, _, _ = map(float, cv2.fitLine(xy, cv2.DIST_HUBER, 0, .01, .01).flatten())
    if vx < 0:
        vx, vy = -vx, -vy
    return {"area_semantic_px": int(len(xx)),
            "angle_deg_canonical": float(math.degrees(math.atan2(vy, vx))),
            "x_span_px": float(np.ptp(xy[:, 0])),
            "mean_probability": float(np.mean(prob[yy, xx])),
            "bbox_canonical_xyxy": [float(np.min(xy[:, 0])), float(np.min(xy[:, 1])),
                                    float(np.max(xy[:, 0])), float(np.max(xy[:, 1]))]}


def _semantic_candidates(image: np.ndarray, role: str) -> tuple[list[dict], dict]:
    locator = semantic_locator_for_camera(role)
    regions = locator.predict(image)
    prob = np.asarray(regions.probability("incoming_rope_free_span"), np.float32)
    binary = (prob >= SEMANTIC_THRESHOLD).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    components = []
    eligible = []
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < SEMANTIC_COMPONENT_AREA_MIN:
            continue
        component = labels == component_id
        info = _fit_component(prob, component, regions.crop_xyxy)
        info["component_id"] = component_id
        reasons = []
        if info["mean_probability"] < SEMANTIC_MEAN_MIN:
            reasons.append("semantic_probability_low")
        if info["x_span_px"] < SEMANTIC_X_SPAN_MIN_PX:
            reasons.append("semantic_span_short")
        if not SEMANTIC_ANGLE_RANGE_DEG[0] <= info["angle_deg_canonical"] <= SEMANTIC_ANGLE_RANGE_DEG[1]:
            reasons.append("semantic_angle_outside_fixed_gate")
        info["reject_reasons"] = reasons
        components.append(info)
        if not reasons:
            eligible.append((info, component))
    eligible.sort(key=lambda item: item[0]["x_span_px"] * item[0]["mean_probability"], reverse=True)
    accepted = []
    x0, y0, x1, y1 = regions.crop_xyxy
    for info, component in eligible[:SEMANTIC_COMPONENT_MAX]:
        selected = np.zeros_like(binary, np.uint8)
        selected[component] = 255
        proposal = np.zeros(image.shape[:2], np.uint8)
        proposal[y0:y1, x0:x1] = cv2.resize(selected, (x1 - x0, y1 - y0),
                                               interpolation=cv2.INTER_NEAREST)
        py, px = np.nonzero(proposal)
        if len(px) < 12:
            info["reject_reasons"].append("upsampled_proposal_too_small")
            continue
        margin = 100
        box = (max(0, int(px.min()) - margin), max(0, int(py.min()) - margin),
               min(image.shape[1] - 1, int(px.max()) + margin),
               min(image.shape[0] - 1, int(py.max()) + margin))
        accepted.append({"info": info, "mask": proposal, "box": box})
    for info, _ in eligible[SEMANTIC_COMPONENT_MAX:]:
        info["reject_reasons"].append("beyond_fixed_candidate_budget")
    diag = {"source": "camera_specific_ONNX_incoming_rope_free_span",
            "model_path": str(locator.model_path), "metadata_path": str(locator.metadata_path),
            "crop_canonical_xyxy": [int(v) for v in regions.crop_xyxy],
            "threshold": SEMANTIC_THRESHOLD, "candidate_limit": SEMANTIC_COMPONENT_MAX,
            "components": components, "eligible_components": len(eligible),
            "tested_components": len(accepted)}
    return accepted, diag


def _segment_context(segment: np.ndarray, surface: np.ndarray) -> dict:
    h, w = surface.shape
    length = float(np.linalg.norm(segment[1] - segment[0]))
    amount = max(32, min(240, int(length / 6)))
    xy = segment[0] + np.linspace(0., 1., amount)[:, None] * (segment[1] - segment[0])
    x = np.clip(np.rint(xy[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(int), 0, h - 1)
    outside = surface[y, x] == 0
    return {"outside_surface_fraction": float(np.mean(outside)),
            "outside_surface_length_estimate_px": float(length * np.mean(outside)),
            "surface_overlap_fraction": float(1. - np.mean(outside)),
            "segment_length_px": length}


def _surface_distance_map(surface: np.ndarray) -> tuple[np.ndarray, float]:
    """粗分辨率距离图只服务于候选拒识，不影响原图宽度测量。"""
    scale = min(1., 1024. / max(surface.shape))
    small = cv2.resize(surface, (round(surface.shape[1] * scale),
                                 round(surface.shape[0] * scale)),
                       interpolation=cv2.INTER_NEAREST)
    return cv2.distanceTransform((small == 0).astype(np.uint8), cv2.DIST_L2, 3), scale


def _surface_ray_hits(segment: np.ndarray, surface: np.ndarray) -> list[dict]:
    """从观测直段两个外端各延长有限射线；交到工作ROI不等于真实切点。"""
    vector = segment[1] - segment[0]
    unit = vector / max(float(np.linalg.norm(vector)), 1e-6)
    h, w = surface.shape
    limit = min(SURFACE_RAY_MAX_PX, .45 * max(h, w))
    distances = np.arange(0., limit + SURFACE_RAY_STEP_PX, SURFACE_RAY_STEP_PX)
    results = []
    for end_index, ray in ((0, -unit), (1, unit)):
        xy = segment[end_index][None, :] + distances[:, None] * ray[None, :]
        ix = np.rint(xy[:, 0]).astype(int)
        iy = np.rint(xy[:, 1]).astype(int)
        inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        sample_index = np.flatnonzero(inside)
        hits = sample_index[surface[iy[sample_index], ix[sample_index]] != 0]
        if len(hits):
            first = int(hits[0])
            results.append({"end_index": end_index, "intersects_surface": True,
                            "first_hit_distance_px": float(distances[first]),
                            "first_hit_xy": [int(ix[first]), int(iy[first])]})
        else:
            results.append({"end_index": end_index, "intersects_surface": False,
                            "first_hit_distance_px": None, "first_hit_xy": None})
    return results


def _flange_background_fraction(image: np.ndarray, segment: np.ndarray,
                                width_px: float | None) -> float:
    """原图观测线两侧的黄法兰背景比例；非黄色真绳仍可由近卷面证据支持。"""
    vector = segment[1] - segment[0]
    unit = vector / max(float(np.linalg.norm(vector)), 1e-6)
    normal = np.array([-unit[1], unit[0]])
    side = max(20., 1.25 * float(width_px or 0.))
    center = (segment[0][None, :] + np.linspace(.06, .94, 20)[:, None] * vector[None, :])
    probes = center[:, None, :] + np.array([-1., 1.])[None, :, None] * side * normal
    ix = np.rint(probes[:, :, 0]).astype(int)
    iy = np.rint(probes[:, :, 1]).astype(int)
    h, w = image.shape[:2]
    valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    paired = np.all(valid, axis=1)
    if int(np.sum(paired)) < 10:
        return 0.
    bgr = image[iy[paired], ix[paired]]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    yellow = ((hsv[:, :, 0] >= 13) & (hsv[:, :, 0] <= 43) &
              (hsv[:, :, 1] > 65) & (hsv[:, :, 2] > 25))
    return float(np.mean(np.all(yellow, axis=1)))


def _surface_approach_evidence(image: np.ndarray, surface: np.ndarray,
                               segment: np.ndarray, width_px: float | None,
                               distance_map: np.ndarray, scale: float) -> dict:
    """需要观测直段确实朝卷面延续，且不只是机架上一条孤立亮边。"""
    rays = _surface_ray_hits(segment, surface)
    endpoint_distances = []
    for point in segment:
        ix = int(np.clip(round(point[0] * scale), 0, distance_map.shape[1] - 1))
        iy = int(np.clip(round(point[1] * scale), 0, distance_map.shape[0] - 1))
        endpoint_distances.append(float(distance_map[iy, ix] / scale))
    yellow_fraction = _flange_background_fraction(image, segment, width_px)
    near_surface = min(endpoint_distances) <= BACKGROUND_NEAR_SURFACE_MAX_PX
    return {"surface_ray_hits": rays,
            "finite_ray_enters_surface": bool(any(ray["intersects_surface"] for ray in rays)),
            "bilateral_yellow_background_fraction": yellow_fraction,
            "endpoint_distance_to_surface_px": endpoint_distances,
            "background_supported": bool(yellow_fraction >= BACKGROUND_YELLOW_MIN_FRACTION or
                                         near_surface),
            "background_support_kind": "bilateral_yellow_flange" if
            yellow_fraction >= BACKGROUND_YELLOW_MIN_FRACTION else
            ("near_surface_original_stripe" if near_surface else "unconfirmed_structure_or_background"),
            "surface_intersection_meaning": "geometric_work_ROI_intersection_not_material_contact"}


def _canonical_to_original(points: np.ndarray, role: str, width: int) -> np.ndarray:
    result = np.asarray(points, float).copy()
    if role == "left":
        result[..., 0] = width - 1 - result[..., 0]
    return result


def _resample_original_profiles(image: np.ndarray, surface: np.ndarray,
                                history: dict) -> tuple[dict | None, dict]:
    """只在卷面外直段重采样；_profiles 给真正半峰交点的中心及宽度。"""
    segment = np.asarray(history["observed_segment"], float)
    vector = segment[1] - segment[0]
    length = float(np.linalg.norm(vector))
    if length < RESAMPLE_MIN_SPAN_PX:
        return None, {"status": "historical_segment_too_short"}
    d = vector / length
    c = np.mean(segment, axis=0)
    candidate_width = history.get("width_px") or history.get("diagnostics", {}).get("candidate_optical_width_px")
    if candidate_width is None or not math.isfinite(float(candidate_width)):
        return None, {"status": "historical_width_unavailable"}
    candidate_width = float(candidate_width)
    search = max(3., min(10., candidate_width * .35))
    max_width = max(12., min(70., candidate_width * 1.6))
    margin = int(max(25., candidate_width * 2.5))
    box = (max(0, int(segment[:, 0].min()) - margin),
           max(0, int(segment[:, 1].min()) - margin),
           min(image.shape[1] - 1, int(segment[:, 0].max()) + margin),
           min(image.shape[0] - 1, int(segment[:, 1].max()) + margin))
    kind = history.get("diagnostics", {}).get("line_support_kind")
    if kind == "bilateral_yellow_free_span":
        channel = image.min(axis=2).astype(np.float32)
        channel *= 255. / np.maximum(image.max(axis=2), 1)
    else:
        channel = image.min(axis=2).astype(np.float32)
    outside = (surface == 0).astype(np.float32)
    projection = (float(min(segment @ d)), float(max(segment @ d)))
    result = rope_detector._profiles(channel, c, d, box, None, outside,
                                      search=search, step=.5, max_width=max_width,
                                      guided=True, sample_step=RESAMPLE_STEP_PX,
                                      projection_range=projection)
    if result is None:
        return None, {"status": "no_coherent_outside_surface_half_prominence_profiles",
                      "channel": kind or "isolated_brightness_stripe"}
    points = np.asarray(result["points"], float)
    widths = np.asarray(result["widths"], float)
    normal = np.array([-d[1], d[0]])
    residual = np.abs((points - c) @ normal)
    acceptable = (residual <= max(3., candidate_width * .33)) & np.isfinite(widths)
    if kind == "bilateral_yellow_free_span":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        yellow = ((hsv[:, :, 0] >= 13) & (hsv[:, :, 0] <= 43) &
                  (hsv[:, :, 1] > 65) & (hsv[:, :, 2] > 25)).astype(np.float32)
        yellow = cv2.GaussianBlur(yellow, (5, 5), 1.)
        side_distance = np.maximum(20., widths * 1.25)
        sides = points[:, None, :] + np.asarray([-1., 1.])[None, :, None] * side_distance[:, None, None] * normal
        bilateral = np.min(rope_detector._sample(yellow, sides), axis=1) > .45
        acceptable &= bilateral
    points, widths = points[acceptable], widths[acceptable]
    if len(points) < RESAMPLE_MIN_POINTS or float(np.ptp(points @ d)) < RESAMPLE_MIN_SPAN_PX:
        return None, {"status": "resample_sparse_or_drifted", "samples": int(len(points)),
                      "span_px": float(np.ptp(points @ d)) if len(points) else 0.}
    median = float(np.median(widths))
    mad = float(1.4826 * np.median(np.abs(widths - median)))
    if (abs(median - candidate_width) > max(2.5, candidate_width * .38) or
            mad > max(1.8, median * .30)):
        return None, {"status": "resampled_width_disagrees_or_unstable",
                      "samples": int(len(points)), "median_width_px": median,
                      "width_mad_px": mad, "historical_width_px": candidate_width}
    return ({"points": points, "widths": widths, "normal": normal,
             "center": c, "direction": d, "width_median": median,
             "width_interval": [float(np.percentile(widths, 10)),
                                float(np.percentile(widths, 90))]},
            {"status": "original_pixel_outside_surface_half_prominence_profiles",
             "samples": int(len(points)), "span_px": float(np.ptp(points @ d)),
             "width_mad_px": mad, "channel": kind or "isolated_brightness_stripe",
             "sample_step_px": RESAMPLE_STEP_PX,
             "profile_source": "rope_detector._profiles_original_BGR_intensity_and_surface_exclusion"})


def _direction_toward_surface(segment: np.ndarray, surface: np.ndarray,
                              width_px: float | None) -> tuple[list[float] | None, str, list[list[float]], dict]:
    vec = segment[1] - segment[0]
    unit = vec / max(float(np.linalg.norm(vec)), 1e-6)
    distance, scale = _surface_distance_map(surface)
    values = []
    for point in segment:
        x = int(np.clip(round(point[0] * scale), 0, distance.shape[1] - 1))
        y = int(np.clip(round(point[1] * scale), 0, distance.shape[0] - 1))
        values.append(float(distance[y, x] / scale))
    rays = _surface_ray_hits(segment, surface)
    reached = [ray["end_index"] for ray in rays if ray["intersects_surface"]]
    diag = {"endpoint_distance_to_surface_px": values, "surface_ray_hits": rays,
            "direction_meaning": "line_ray_toward_work_ROI_not_verified_rope_contact"}
    if len(reached) == 1:
        toward = -unit if reached[0] == 0 else unit
        return toward.tolist(), "unique_surface_ray", [toward.tolist()], diag
    if len(reached) == 0:
        status = "refined_line_ray_does_not_reach_surface"
    else:
        status = "ambiguous_both_surface_rays"
    alternatives = [unit.tolist(), (-unit).tolist()]
    return None, status, alternatives, diag


def _same_line(a: dict, b: dict) -> bool:
    sa = np.asarray(a["history"]["observed_segment"], float)
    sb = np.asarray(b["history"]["observed_segment"], float)
    da = (sa[1] - sa[0]) / np.linalg.norm(sa[1] - sa[0])
    db = (sb[1] - sb[0]) / np.linalg.norm(sb[1] - sb[0])
    normal = np.array([-da[1], da[0]])
    separation = abs(float((np.mean(sa, axis=0) - np.mean(sb, axis=0)) @ normal))
    return abs(float(da @ db)) >= math.cos(math.radians(3.)) and separation <= max(10.,
           1.5 * float(a["history"].get("width_px") or 8.))


def _retry_semantic_only_outside_surface(image: np.ndarray, surface: np.ndarray,
                                         proposal: dict, distance_full: np.ndarray) -> tuple[dict | None, dict]:
    """原候选跨卷面精测失败时，只从同一自动候选的自由侧重测一次。"""
    restricted = np.where((proposal["mask"] != 0) &
                          (distance_full > OUTSIDE_RETRY_GUARD_PX), 255, 0).astype(np.uint8)
    yy, xx = np.nonzero(restricted)
    info = {"attempted": True, "source_component_id": proposal["info"]["component_id"],
            "distance_from_surface_guard_px": OUTSIDE_RETRY_GUARD_PX,
            "mask_pixels_outside_guard": int(len(xx)),
            "original_mask_pixels": int(np.count_nonzero(proposal["mask"]))}
    if len(xx) < 12:
        info["reject_reason"] = "same_component_outside_surface_too_sparse"
        return None, info
    pad = OUTSIDE_RETRY_BOX_MARGIN_PX
    box = (max(0, int(xx.min()) - pad), max(0, int(yy.min()) - pad),
           min(image.shape[1] - 1, int(xx.max()) + pad),
           min(image.shape[0] - 1, int(yy.max()) + pad))
    info["restricted_box_canonical_xyxy"] = list(box)
    history = rope_detector.detect_rope(image, roi=box, rope_mask=restricted)
    info["historical_valid"] = bool(history.get("valid"))
    info["historical_reason"] = history.get("reason")
    info["historical_width_status"] = history.get("diagnostics", {}).get("width_status")
    if not history.get("valid"):
        info["reject_reason"] = "same_component_original_detector_invalid"
        return None, info
    # 旧检测器可因局部反光把宽度标成不稳；本模块的原图剖面重新量测仍须
    # 满足固定的样本数、跨度、宽度一致性门槛，否则不把方向提升为观测绳。
    profile, profile_diag = _resample_original_profiles(image, surface, history)
    info["original_profile_recheck"] = profile_diag
    if profile is None:
        info["reject_reason"] = "same_component_original_double_edge_unconfirmed"
        return None, info
    info["status"] = "accepted_for_common_spatial_and_competition_gates"
    return history, info


def _finish(selected: dict, image: np.ndarray, surface: np.ndarray,
            role: str, diagnostics: dict) -> dict:
    history = selected["history"]
    h, w = image.shape[:2]
    canonical_image = cv2.flip(image, 1) if role == "left" else image
    canonical_surface = cv2.flip(surface, 1) if role == "left" else surface
    profile, profile_diag = _resample_original_profiles(canonical_image, canonical_surface, history)
    diagnostics["selected_profile_refinement"] = profile_diag
    if profile is not None:
        center, direction, residual = rope_detector._line(profile["points"])
        if direction[0] < 0:
            direction = -direction
        projection = (profile["points"] - center) @ direction
        t0, t1 = np.percentile(projection, [1, 99])
        segment = np.asarray([center + t0 * direction, center + t1 * direction])
        angular_unc, normal_unc = rope_detector._support_uncertainty(profile["points"],
                                                                     center, direction, residual)
        points = profile["points"]
        widths = profile["widths"]
        edge_normal = profile["normal"]
        width = profile["width_median"]
        width_interval = profile["width_interval"]
        observed_kind = "individual_original_pixel_outside_surface_half_prominence_profiles"
        boundary_kind = "individual_half_prominence_crossings_reconstructed_from_measured_center_and_width"
    else:
        segment = np.asarray(history["observed_segment"], float)
        direction = (segment[1] - segment[0]) / np.linalg.norm(segment[1] - segment[0])
        bins = history.get("diagnostics", {}).get("width_local_samples", [])
        candidate_points = np.asarray([item["point"] for item in bins], float).reshape(-1, 2)
        candidate_widths = np.asarray([item["width_px"] for item in bins], float)
        if len(candidate_points):
            ix = np.clip(np.rint(candidate_points[:, 0]).astype(int), 0, w - 1)
            iy = np.clip(np.rint(candidate_points[:, 1]).astype(int), 0, h - 1)
            keep = canonical_surface[iy, ix] == 0
            candidate_points, candidate_widths = candidate_points[keep], candidate_widths[keep]
        points, widths = candidate_points, candidate_widths
        edge_normal = np.array([-direction[1], direction[0]])
        width = history.get("width_px") if len(points) >= 2 else None
        width_interval = history.get("width_interval_px") if width is not None else None
        angular_unc = history.get("angular_uncertainty_deg")
        normal_unc = history.get("normal_uncertainty_px")
        observed_kind = "sparse_aggregated_historical_original_profile_bin_centers" if len(points) else "none"
        boundary_kind = "symmetric_from_sparse_aggregated_profile_bin_width" if width is not None else "none"
        diagnostics["sparse_fallback_bins_outside_surface"] = int(len(points))
    if segment[0, 0] > segment[1, 0]:
        segment = segment[::-1]
    original_segment = _canonical_to_original(segment, role, w)
    original_points = _canonical_to_original(points, role, w)
    if original_segment[0, 0] > original_segment[1, 0]:
        original_segment = original_segment[::-1]
    vector = original_segment[1] - original_segment[0]
    unit = vector / max(float(np.linalg.norm(vector)), 1e-6)
    normal = np.array([-unit[1], unit[0]])
    if normal[1] < 0:
        normal = -normal
    center_xy = np.mean(original_segment, axis=0)
    if width is not None and len(points):
        left = points - widths[:, None] * .5 * edge_normal
        right = points + widths[:, None] * .5 * edge_normal
        left = _canonical_to_original(left, role, w)
        right = _canonical_to_original(right, role, w)
        # 左右仅指法向负/正侧；在镜像还原后重新统一成原图 normal。
        original_edge_normal = _canonical_to_original(
            np.asarray([np.zeros(2), edge_normal]), role, w)
        side_vector = original_edge_normal[1] - original_edge_normal[0]
        if float(side_vector @ normal) < 0:
            left, right = right, left
        order = np.argsort(original_points[:, 0])
        original_points, left, right = original_points[order], left[order], right[order]
    else:
        left = right = np.empty((0, 2), float)
    toward, toward_status, alternatives, approach_diag = _direction_toward_surface(
        original_segment, surface, width)
    diagnostics["surface_approach"] = approach_diag
    status = "observed" if width is not None and len(points) >= 2 else "direction_only"
    return {"status": status, "reason": "original_pixel_profiles_supported" if status == "observed" else
            "direction_supported_width_unavailable", "camera_role": role,
            "coordinate_system": "original_unmirrored_image_xy_pixels",
            "observed_segment_xy": original_segment.tolist(), "center_xy": center_xy.tolist(),
            "direction_xy": unit.tolist(), "toward_surface_direction_xy": toward,
            "toward_surface_status": toward_status,
            "toward_surface_candidates_xy": alternatives,
            "normal_xy": normal.tolist(), "rho_px": float(center_xy @ normal),
            "angle_deg": float(math.degrees(math.atan2(unit[1], unit[0])) % 180.),
            "width_px": float(width) if width is not None else None,
            "width_interval_px": [float(v) for v in width_interval] if width_interval is not None else None,
            "observed_points_xy": original_points.tolist() if width is not None else [],
            "boundary_left_xy": left.tolist(), "boundary_right_xy": right.tolist(),
            "observed_points_kind": observed_kind, "boundary_points_kind": boundary_kind,
            "boundary_side_convention": "left=negative_original_image_normal; right=positive_original_image_normal",
            "support_length_px": float(np.linalg.norm(vector)),
            "observed_support": int(history["observed_support"]),
            "profile_sample_count": int(len(points)),
            "angular_uncertainty_deg": float(angular_unc) if angular_unc is not None else None,
            "normal_uncertainty_px": float(normal_unc) if normal_unc is not None else None,
            "uncertainty_kind": "empirical_segment_halves_not_calibrated",
            "confidence": float(history["confidence"]),
            "width_kind": "optical_half_prominence_not_physical_diameter",
            "source": selected["source"], "complete_free_rope_claim": False,
            "contact_or_tangency_claim": False, "diagnostics": diagnostics}


def detect_free_segment(image_bgr: np.ndarray, surface_mask: np.ndarray,
                        camera_role: str, prior: dict | None = None) -> dict:
    """自动找自由直段，返回 JSON 安全证据；无可靠支持时保留 unknown。

    `prior` 仅接受供调用方登记的字典，本版不读取其中的卷面线号、黄点、
    人工标签或方向来选择候选。每次最多测 6 个语义分量；某分量整段
    原图精测无效时，最多在同一自动分量的卷面外部分重试一次。若所有
    语义候选仍无支持，才执行一次无提示全局检测。固定门槛见文件顶部。
    """
    role = _validate(image_bgr, surface_mask, camera_role, prior)
    canonical = cv2.flip(image_bgr, 1) if role == "left" else image_bgr
    canonical_surface = cv2.flip(surface_mask, 1) if role == "left" else surface_mask
    distance_map, distance_scale = _surface_distance_map(canonical_surface)
    diagnostics = {"candidate_budget": SEMANTIC_COMPONENT_MAX,
                   "outside_surface_retry_budget_per_component": OUTSIDE_RETRY_MAX_PER_SEMANTIC_COMPONENT,
                   "outside_surface_retry_guard_px": OUTSIDE_RETRY_GUARD_PX,
                   "outside_surface_retries_attempted": 0,
                   "prior_used_for_selection": False,
                   "prior_keys_ignored": sorted(str(key) for key in prior) if prior else [],
                   "surface_meaning": "work_surface_proposal_not_ground_truth",
                   "semantic": {}, "candidates": [], "global_fallback_called": False}
    try:
        proposals, semantic_diag = _semantic_candidates(canonical, role)
        diagnostics["semantic"] = semantic_diag
    except (FileNotFoundError, ValueError, cv2.error) as error:
        proposals = []
        diagnostics["semantic"] = {"status": "unavailable", "error": type(error).__name__ + ": " + str(error)}
    viable = []
    full_outside_distance = None
    for proposal in proposals:
        info = proposal["info"]
        history = rope_detector.detect_rope(canonical, roi=proposal["box"],
                                             rope_mask=proposal["mask"])
        initial_reason = history.get("reason")
        source = "semantic_ONNX_component_then_original_profiles"
        retry_info = None
        if not history.get("valid"):
            if full_outside_distance is None:
                full_outside_distance = cv2.distanceTransform(
                    (canonical_surface == 0).astype(np.uint8), cv2.DIST_L2, 3)
            diagnostics["outside_surface_retries_attempted"] += 1
            retried, retry_info = _retry_semantic_only_outside_surface(
                canonical, canonical_surface, proposal, full_outside_distance)
            if retried is not None:
                history = retried
                source = "same_semantic_component_outside_surface_retry_then_original_profiles"
        item = {"source": source, "initial_historical_reason": initial_reason,
                "outside_surface_retry": retry_info,
                "component_id": info["component_id"], "semantic_mean_probability": info["mean_probability"],
                "historical_valid": bool(history.get("valid")), "historical_reason": history.get("reason"),
                "historical_confidence": history.get("confidence"),
                "historical_observed_support": history.get("observed_support"),
                "historical_width_status": history.get("diagnostics", {}).get("width_status"),
                "reject_reasons": []}
        if not history.get("valid"):
            item["reject_reasons"].append("historical_original_profile_invalid")
        else:
            segment = np.asarray(history["observed_segment"], float)
            context = _segment_context(segment, canonical_surface)
            item.update(context)
            if context["segment_length_px"] < SEMANTIC_OBSERVED_LENGTH_MIN_PX:
                item["reject_reasons"].append("original_observed_segment_short")
            if (context["outside_surface_fraction"] < SEMANTIC_OUTSIDE_SURFACE_MIN_FRACTION or
                    context["outside_surface_length_estimate_px"] < SEMANTIC_OUTSIDE_SURFACE_MIN_LENGTH_PX):
                item["reject_reasons"].append("free_span_outside_surface_support_insufficient")
            approach = _surface_approach_evidence(canonical, canonical_surface, segment,
                                                  history.get("width_px"), distance_map,
                                                  distance_scale)
            item["approach_evidence"] = approach
            if not approach["finite_ray_enters_surface"]:
                item["reject_reasons"].append("finite_straight_ray_misses_work_surface")
            if not approach["background_supported"]:
                item["reject_reasons"].append("isolated_structure_without_flange_or_near_surface_support")
            if not item["reject_reasons"]:
                score = (float(history["confidence"]) *
                         min(1., context["segment_length_px"] / 500.) *
                         (.65 + .35 * info["mean_probability"]) *
                         (.85 + .15 * context["outside_surface_fraction"]))
                item["selection_score"] = score
                viable.append({"history": history, "source": source,
                               "score": score, "item": item})
        diagnostics["candidates"].append(item)
    viable.sort(key=lambda item: item["score"], reverse=True)
    if viable:
        winner = viable[0]
        distinct = [candidate for candidate in viable[1:] if not _same_line(winner, candidate)]
        ratio = distinct[0]["score"] / max(winner["score"], 1e-9) if distinct else 0.
        diagnostics["distinct_competitor_ratio"] = float(ratio)
        if ratio >= COMPETITOR_SCORE_RATIO_LIMIT:
            return _unknown("ambiguous_competing_free_spans", role, diagnostics)
        diagnostics["selected_candidate"] = winner["item"]
        return _finish(winner, image_bgr, surface_mask, role, diagnostics)
    # 无语义支持时才试一次历史全图无提示搜索；工具/手套/机架不可仅因
    # 有长直线就被升级为活动绳，所有拒绝证据保留在 diagnostics。
    diagnostics["global_fallback_called"] = True
    history = rope_detector.detect_rope(canonical)
    fallback = {"source": "historical_global_unguided_detector", "historical_valid":
                bool(history.get("valid")), "historical_reason": history.get("reason"),
                "reject_reasons": []}
    if history.get("valid"):
        context = _segment_context(np.asarray(history["observed_segment"], float), canonical_surface)
        fallback.update(context)
        diag = history.get("diagnostics", {})
        fallback.update({"historical_confidence": history.get("confidence"),
                         "historical_width_status": diag.get("width_status"),
                         "historical_line_support_kind": diag.get("line_support_kind"),
                         "historical_competitor_ratio": diag.get("competitor_ratio")})
        if float(history["confidence"]) < FALLBACK_MIN_CONFIDENCE:
            fallback["reject_reasons"].append("fallback_confidence_low")
        if context["segment_length_px"] < FALLBACK_MIN_LENGTH_PX:
            fallback["reject_reasons"].append("fallback_observed_segment_short")
        if context["outside_surface_fraction"] < FALLBACK_MIN_OUTSIDE_FRACTION:
            fallback["reject_reasons"].append("fallback_not_isolated_outside_surface")
        approach = _surface_approach_evidence(canonical, canonical_surface,
                                              np.asarray(history["observed_segment"], float),
                                              history.get("width_px"), distance_map, distance_scale)
        fallback["approach_evidence"] = approach
        if not approach["finite_ray_enters_surface"]:
            fallback["reject_reasons"].append("finite_straight_ray_misses_work_surface")
        if not approach["background_supported"]:
            fallback["reject_reasons"].append("isolated_structure_without_flange_or_near_surface_support")
        if float(diag.get("competitor_ratio", 1.)) > FALLBACK_MAX_COMPETITOR_RATIO:
            fallback["reject_reasons"].append("fallback_competing_line")
        if diag.get("line_support_kind") != "bilateral_yellow_free_span" or history.get("width_px") is None:
            fallback["reject_reasons"].append("fallback_lacks_bilateral_color_and_stable_width")
        angle = float(history["angle_deg"])
        if not SEMANTIC_ANGLE_RANGE_DEG[0] <= (angle - 180. if angle > 90. else angle) <= SEMANTIC_ANGLE_RANGE_DEG[1]:
            fallback["reject_reasons"].append("fallback_angle_outside_fixed_gate")
    else:
        fallback["reject_reasons"].append("historical_global_detection_invalid")
    diagnostics["candidates"].append(fallback)
    if fallback["reject_reasons"]:
        return _unknown("no_original_supported_free_segment", role, diagnostics)
    diagnostics["selected_candidate"] = fallback
    return _finish({"history": history, "source": fallback["source"]}, image_bgr,
                   surface_mask, role, diagnostics)
