"""C3：先按视觉边界归类全部黄点，再给每组拟合一条抛物线。

图像 y 向下，模型 y=k+b*u+c*u²、c>=0，u=(x-center)/scale。
共享形状只帮助片段归类并作为明确记录的弱正则；最终每组的唯一
模型以该组全部原始黄点和严格正的稳健权重求解。平滑线不证明材料身份。
本模块不读原图、旧紫线、旧 family_id、评价隐藏点或设备。
"""
from __future__ import annotations

from collections import Counter
import math

import numpy as np


def _mask(value: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != 2 or value.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8 HxW")
    if np.any((value != 0) & (value != 255)):
        raise ValueError(f"{name} must contain only 0/255")
    return value > 0


def _input_points(line: dict, width: int, height: int) -> np.ndarray:
    points = np.asarray(line.get("observed_points_xy", []), dtype=float)
    if points.size == 0:
        return np.empty((0, 2), dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("invalid observed_points_xy")
    if ((points[:, 0] < 0).any() or (points[:, 0] >= width).any() or
            (points[:, 1] < 0).any() or (points[:, 1] >= height).any()):
        raise ValueError("observed_points_xy outside image")
    return points


def _shared_shape(fragments: list[np.ndarray], pitch: float,
                  center: float, scale: float) -> tuple[np.ndarray, dict]:
    """每片段消掉独立截距，联合估计仅供关联用的 b、c。"""
    usable = [p for p in fragments if len(p) >= 8 and np.ptp(p[:, 0]) >= 3 * pitch]
    if len(usable) < 3:
        return np.zeros(2, float), {"status": "insufficient_shared_fragments",
                                   "fragment_count": len(usable), "b_px_per_u": 0.,
                                   "c_px_per_u2": 0., "a_px_per_px2": 0.}
    bases = [np.column_stack(((p[:, 0] - center) / scale,
                              ((p[:, 0] - center) / scale) ** 2)) for p in usable]
    targets = [p[:, 1] for p in usable]
    factors = [np.ones(len(p)) for p in usable]
    coefficient = np.zeros(2)
    condition = None
    for iteration in range(7):
        designs, values, weights = [], [], []
        for basis, y, factor in zip(bases, targets, factors):
            weight = factor / len(factor)
            bmean = np.average(basis, axis=0, weights=weight)
            ymean = float(np.average(y, weights=weight))
            designs.append(basis - bmean)
            values.append(y - ymean)
            weights.append(weight)
        design = np.concatenate(designs)
        target = np.concatenate(values)
        weight = np.concatenate(weights)
        gram = design.T @ (weight[:, None] * design)
        if iteration == 0:
            condition = float(np.linalg.cond(gram))
            if not math.isfinite(condition) or condition > 1e6:
                return np.zeros(2, float), {"status": "shared_shape_unidentifiable",
                                           "fragment_count": len(usable),
                                           "condition": condition if math.isfinite(condition) else None,
                                           "b_px_per_u": 0., "c_px_per_u2": 0.,
                                           "a_px_per_px2": 0.}
        coefficient = np.linalg.lstsq(design * np.sqrt(weight[:, None]),
                                      target * np.sqrt(weight), rcond=None)[0]
        if not np.isfinite(coefficient).all():
            return np.zeros(2, float), {"status": "nonfinite_shared_shape",
                                       "fragment_count": len(usable),
                                       "b_px_per_u": 0., "c_px_per_u2": 0.,
                                       "a_px_per_px2": 0.}
        residuals = []
        for basis, y, factor in zip(bases, targets, factors):
            raw = y - basis @ coefficient
            intercept = float(np.average(raw, weights=factor))
            residuals.append(raw - intercept)
        pooled = np.concatenate(residuals)
        robust_scale = max(.8, 1.4826 * float(np.median(np.abs(pooled))))
        factors = [np.clip(1.5 * robust_scale / np.maximum(np.abs(r), 1e-6), .08, 1.)
                   for r in residuals]
    b, c = map(float, coefficient)
    if c <= .10 * pitch:
        c = 0.
        status = "near_flat_shared_shape"
    else:
        status = "positive_shared_parabola"
    if not math.isfinite(b) or not math.isfinite(c):
        b, c, status = 0., 0., "nonfinite_shared_shape"
    return np.array([b, c], float), {"status": status, "fragment_count": len(usable),
                                    "center_x_px": center, "scale_x_px": scale,
                                    "b_px_per_u": b, "c_px_per_u2": c,
                                    "a_px_per_px2": c / scale**2,
                                    "condition": condition,
                                    "centered_residual_mad_px": float(np.median(np.abs(pooled)))}


def _shape(x: np.ndarray | float, shared: np.ndarray,
           center: float, scale: float) -> np.ndarray | float:
    u = (x - center) / scale
    return shared[0] * u + shared[1] * u * u


def _profile(points: np.ndarray, shared: np.ndarray, center: float,
             scale: float, pitch: float) -> dict:
    """局部中心/斜率只用于判断片段是否相接，不进入最终 group 模型。"""
    ordered = points[np.argsort(points[:, 0])]
    mid = float(np.median(ordered[:, 0]))
    raw = ordered[:, 1] - _shape(ordered[:, 0], shared, center, scale)
    dx = ordered[:, 0] - mid
    if len(ordered) >= 6 and np.ptp(ordered[:, 0]) >= 2 * pitch:
        slope = float(np.clip(np.sum(dx * (raw - np.median(raw))) /
                              max(1., np.sum(dx * dx)), -.03, .03))
    else:
        slope = 0.
    offset = float(np.median(raw - slope * dx))
    deviation = raw - (offset + slope * dx)
    return {"xlo": float(ordered[0, 0]), "xhi": float(ordered[-1, 0]),
            "mid": mid, "offset": offset, "slope": slope,
            "mad": float(np.median(np.abs(deviation))),
            "count": int(len(points))}


def _profile_y(profile: dict, x: float, shared: np.ndarray,
               center: float, scale: float) -> float:
    return float(_shape(x, shared, center, scale) + profile["offset"] +
                 profile["slope"] * (x - profile["mid"]))


def _fit_all_points(points: np.ndarray, shared: np.ndarray,
                    center: float, scale: float, pitch: float) -> dict | None:
    """IRLS 每次用所有点；最终权重再解一次并原样输出，供 normal equation 复算。"""
    if len(points) == 0:
        return None
    u = (points[:, 0] - center) / scale
    design = np.column_stack((np.ones(len(points)), u, u * u))
    y = points[:, 1]
    span = float(np.ptp(points[:, 0]))
    coverage = max(.25, min(2., span / max(1., 6 * pitch)))
    # 先验仅在 b/c 系数上施加岭惩罚，不添加任何合成观测点，也不惩罚 k。
    lam_b = max(.05, .04 * len(points) / coverage**2)
    lam_c = max(.05, .08 * len(points) / coverage**4)
    prior_b, prior_c = float(shared[0]), float(shared[1])
    weight = np.ones(len(points), float)

    def solve(current_weight: np.ndarray) -> tuple[np.ndarray, bool, float]:
        normal = design.T @ (current_weight[:, None] * design)
        normal[1, 1] += lam_b
        normal[2, 2] += lam_c
        rhs = design.T @ (current_weight * y)
        rhs[1] += lam_b * prior_b
        rhs[2] += lam_c * prior_c
        coef = np.linalg.solve(normal, rhs)
        active = bool(coef[2] < 0)
        if active:
            # KKT 边界 c=0：重新用所有点求 k/b，绝不先拟合再将 c 截断。
            short_design = design[:, :2]
            normal2 = short_design.T @ (current_weight[:, None] * short_design)
            normal2[1, 1] += lam_b
            rhs2 = short_design.T @ (current_weight * y)
            rhs2[1] += lam_b * prior_b
            short_coef = np.linalg.solve(normal2, rhs2)
            coef = np.array([short_coef[0], short_coef[1], 0.], float)
        gradient = design.T @ (current_weight * (design @ coef - y))
        gradient[1] += lam_b * (coef[1] - prior_b)
        gradient[2] += lam_c * (coef[2] - prior_c)
        free_gradient = gradient[:2] if active else gradient
        kkt = float(np.max(np.abs(free_gradient)) / max(1., float(np.sum(current_weight))))
        return coef, active, kkt

    for _ in range(6):
        coef, _, _ = solve(weight)
        residual = y - design @ coef
        scale_px = max(.8, 1.4826 * float(np.median(np.abs(residual))))
        weight = np.clip(1.5 * scale_px / np.maximum(np.abs(residual), 1e-6), .08, 1.)
    coef, active, kkt = solve(weight)
    residual = y - design @ coef
    if not np.isfinite(coef).all() or not np.isfinite(residual).all():
        return None
    return {"coefficients": coef, "weights": weight, "residuals": residual,
            "mad_px": float(np.median(np.abs(residual))),
            "p90_px": float(np.quantile(np.abs(residual), .90)),
            "regularization": {"parameter_order": ["normalized_k", "normalized_b", "normalized_c"],
                               "prior_b": prior_b, "prior_c": prior_c,
                               "lambda_b": lam_b, "lambda_c": lam_c,
                               "lambda_k": 0., "c_nonnegative_active": active,
                               "objective": "sum_i w_i*(y_i-k-b*u_i-c*u_i^2)^2 + lambda_b*(b-prior_b)^2 + lambda_c*(c-prior_c)^2",
                               "normal_equation_free_gradient_max_per_weight": kkt}}


def fit_conic_family(lines: list[dict], pitch_px: float, roi_mask: np.ndarray,
                     exclusion_mask: np.ndarray, parameters=None) -> dict:
    """按视觉边界分组，输出每组唯一全点抛物线及可审计点分配。"""
    if parameters is not None:
        raise ValueError("C3 has frozen built-in parameters; custom parameters unsupported")
    pitch = float(pitch_px)
    if not math.isfinite(pitch) or not 8 <= pitch <= 70:
        raise ValueError("invalid pitch_px")
    roi = _mask(roi_mask, "roi_mask")
    exclusion = _mask(exclusion_mask, "exclusion_mask")
    if roi.shape != exclusion.shape or np.any(exclusion & ~roi):
        raise ValueError("exclusion_mask must be inside ROI")
    if not isinstance(lines, list):
        raise ValueError("lines must be a list")
    height, width = roi.shape
    fragments = [_input_points(line, width, height) for line in lines]
    ids = [line.get("line_id", f"fragment_{i:04d}") for i, line in enumerate(lines)]
    if len(set(ids)) != len(ids):
        raise ValueError("line_id must be unique")
    smooth = np.zeros((height, width), np.uint8)
    result = {"model": None, "groups": [], "curves": [], "smooth_mask": smooth,
              "rejected": [], "diagnostics": {}}
    nonempty = [p for p in fragments if len(p)]
    if not nonempty:
        result["diagnostics"] = {"status": "no_observed_points", "input_point_count": 0,
                                 "grouped_point_count": 0, "accepted_group_count": 0}
        return result
    all_x = np.concatenate([p[:, 0] for p in nonempty])
    center = .5 * float(all_x.min() + all_x.max())
    scale = max(1., .5 * float(all_x.max() - all_x.min()))
    shared, shared_info = _shared_shape(fragments, pitch, center, scale)
    result["model"] = {"type": "vertical_axis_parabola_family",
                       "shared_shape_for_grouping_and_regularization": shared_info,
                       "group_curve_formula": "y=k+b*u+c*u^2; u=(x-center)/scale; c>=0",
                       "one_model_per_accepted_group": True}
    parent = list(range(len(lines)))
    members = {i: [i] for i in range(len(lines))}
    reasons: Counter[str] = Counter()
    accepted_edges = []

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def merge(a: int, b: int) -> int:
        ra, rb = find(a), find(b)
        if ra == rb:
            return ra
        if min(members[rb]) < min(members[ra]):
            ra, rb = rb, ra
        parent[rb] = ra
        members[ra].extend(members.pop(rb))
        members[ra].sort()
        return ra

    def gathered(indices: list[int]) -> np.ndarray:
        pieces = [fragments[i] for i in indices if len(fragments[i])]
        return np.concatenate(pieces) if pieces else np.empty((0, 2), float)

    def fit_consistent(indices: list[int], strict: bool = False) -> tuple[bool, dict | None]:
        fitted = _fit_all_points(gathered(indices), shared, center, scale, pitch)
        if fitted is None:
            return False, None
        mad_limit, p90_limit = ((.18, .38) if strict else (.22, .45))
        return (fitted["mad_px"] <= mad_limit * pitch and
                fitted["p90_px"] <= p90_limit * pitch), fitted

    # 阶段1a：相同列域中的重叠碎段需在多个列位置贴合才合并。
    initial_profiles = [_profile(p, shared, center, scale, pitch) if len(p) >= 2 else None
                        for p in fragments]
    overlap_edges = []
    for i in range(len(lines)):
        pi = initial_profiles[i]
        if pi is None or pi["mad"] > .25 * pitch:
            continue
        for j in range(i + 1, len(lines)):
            pj = initial_profiles[j]
            if pj is None or pj["mad"] > .25 * pitch:
                continue
            lo = max(pi["xlo"], pj["xlo"])
            hi = min(pi["xhi"], pj["xhi"])
            if hi - lo < max(4., .4 * pitch):
                continue
            difference = [abs(_profile_y(pi, x, shared, center, scale) -
                              _profile_y(pj, x, shared, center, scale))
                          for x in (lo, .5 * (lo + hi), hi)]
            if max(difference) <= .34 * pitch and float(np.median(difference)) <= .18 * pitch:
                overlap_edges.append((float(np.median(difference)) / pitch, i, j))
    for score, i, j in sorted(overlap_edges):
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        okay, _ = fit_consistent(members[ri] + members[rj], strict=True)
        if not okay:
            reasons["overlap_all_point_fit_disagrees"] += 1
            continue
        merge(ri, rj)
        accepted_edges.append({"from": ids[i], "to": ids[j], "type": "multi_column_overlap",
                               "score": score})

    # 阶段1b：跨真正空档关联。只用可见左右组的端头、共享走势、邻行和
    # 合并后全部黄点拟合残差；隐藏带绝不进入这里。
    roots = sorted(members)
    group_points = {root: gathered(members[root]) for root in roots}
    profiles = {root: _profile(group_points[root], shared, center, scale, pitch)
                for root in roots if len(group_points[root]) >= 2}
    spacing_values = sorted(p["offset"] for p in profiles.values() if p["count"] >= 8)
    distinct = []
    for value in spacing_values:
        if not distinct or value - distinct[-1] > .28 * pitch:
            distinct.append(value)
    differences = np.diff(distinct)
    plausible = differences[(differences >= .55 * pitch) & (differences <= 1.35 * pitch)]
    spacing = float(np.median(plausible)) if len(plausible) >= 3 else pitch
    spacing = float(np.clip(spacing, .70 * pitch, 1.30 * pitch))

    def neighbors(a: int, b: int, x0: float, x1: float,
                  midpoint_y: float) -> int:
        positions = []
        xmid = .5 * (x0 + x1)
        for other, profile in profiles.items():
            if other in (a, b) or profile["count"] < 8 or profile["mad"] > .24 * pitch:
                continue
            if profile["xhi"] < x0 - 2 * pitch or profile["xlo"] > x1 + 2 * pitch:
                continue
            y = _profile_y(profile, xmid, shared, center, scale)
            distance = abs(y - midpoint_y)
            multiple = int(round(distance / spacing))
            if 1 <= multiple <= 6 and abs(distance - multiple * spacing) <= .36 * pitch:
                positions.append(y)
        unique = []
        for value in sorted(positions):
            if not unique or value - unique[-1] > .46 * pitch:
                unique.append(value)
        return len(unique)

    candidates = []
    maximum_gap = min(25 * pitch, 650.)
    for a in roots:
        pa = profiles.get(a)
        if pa is None or pa["mad"] > .30 * pitch:
            continue
        left = group_points[a][np.argsort(group_points[a][:, 0])]
        for b in roots:
            if a == b:
                continue
            pb = profiles.get(b)
            if pb is None or pb["mad"] > .30 * pitch:
                continue
            right = group_points[b][np.argsort(group_points[b][:, 0])]
            gap = float(right[0, 0] - left[-1, 0])
            if not max(3., .25 * pitch) <= gap <= maximum_gap:
                continue
            ltail = left[-min(7, len(left)):]
            rhead = right[:min(7, len(right))]
            loffset = float(np.median(ltail[:, 1] - _shape(ltail[:, 0], shared, center, scale)))
            roffset = float(np.median(rhead[:, 1] - _shape(rhead[:, 0], shared, center, scale)))
            if abs(loffset - roffset) > .62 * pitch:
                reasons["gap_endpoint_different_course"] += 1
                continue
            xmid = .5 * (left[-1, 0] + right[0, 0])
            profile_difference = abs(_profile_y(pa, xmid, shared, center, scale) -
                                     _profile_y(pb, xmid, shared, center, scale))
            if profile_difference > .60 * pitch:
                reasons["gap_multi_column_projection_disagrees"] += 1
                continue
            near = neighbors(a, b, float(left[-1, 0]), float(right[0, 0]),
                             .5 * (_profile_y(pa, xmid, shared, center, scale) +
                                   _profile_y(pb, xmid, shared, center, scale)))
            required = 2 if len(left) >= 8 and len(right) >= 8 else 3
            if near < required:
                reasons["gap_insufficient_neighbor_courses"] += 1
                continue
            okay, fit = fit_consistent(members[a] + members[b])
            if not okay:
                reasons["gap_all_point_fit_disagrees"] += 1
                continue
            support = min(len(left), 80) + min(len(right), 80)
            score = (abs(loffset - roffset) / pitch + .25 * profile_difference / pitch +
                     .25 * gap / maximum_gap + fit["mad_px"] / pitch -
                     .003 * support - .025 * near)
            candidates.append((score, a, b, loffset, roffset, near, gap))
    candidates.sort(key=lambda row: row[0])
    successor, predecessor, endpoint_offsets = {}, {}, {}
    for score, a, b, loffset, roffset, near, gap in candidates:
        if a in successor or b in predecessor or find(a) == find(b):
            reasons["gap_one_to_one_conflict"] += 1
            continue
        crossing = False
        for other_l, other_r in endpoint_offsets.values():
            left_sep, right_sep = loffset - other_l, roffset - other_r
            if (left_sep * right_sep < 0 or
                    (max(abs(left_sep), abs(right_sep)) >= .65 * pitch and
                     min(abs(left_sep), abs(right_sep)) < .45 * pitch)):
                crossing = True
                break
        if crossing:
            reasons["gap_order_crossing_or_collapse"] += 1
            continue
        ra, rb = find(a), find(b)
        okay, _ = fit_consistent(members[ra] + members[rb])
        if not okay:
            reasons["gap_full_chain_fit_disagrees"] += 1
            continue
        from_id, to_id = ids[min(members[ra])], ids[min(members[rb])]
        from_members = [ids[index] for index in members[ra]]
        to_members = [ids[index] for index in members[rb]]
        merge(ra, rb)
        successor[a], predecessor[b] = b, a
        endpoint_offsets[(a, b)] = (loffset, roffset)
        accepted_edges.append({"from": from_id, "to": to_id,
                               "from_group_members_before_merge": from_members,
                               "to_group_members_before_merge": to_members,
                               "type": "two_side_multi_column_gap",
                               "score": score, "gap_px": gap,
                               "neighbor_courses": near,
                               "endpoint_offsets_px": [loffset, roffset]})

    # 首轮失败发现：短至2–6个黄点的孤片段常与已由两端归为同一行的
    # 长组在相同可见列贴靠。若直接把这两条模型都拒掉，就丢失整行。
    # 只在该短片段的每个真实黄点与某唯一长组的全点模型贴合时，
    # 将其作为同一视觉边界的成员，随后用合并后全部黄点重新拟合。
    # 这里不生成任何端点补片，也不使用评价隐藏点。
    short_roots = [r for r in sorted(members, key=lambda key: min(members[key]))
                   if len(gathered(members[r])) <= 8 and
                   (len(gathered(members[r])) == 0 or
                    np.ptp(gathered(members[r])[:, 0]) <= 1.5 * pitch)]
    for short_root in short_roots:
        if short_root not in members:
            continue
        short_points = gathered(members[short_root])
        if len(short_points) == 0:
            continue
        compatible = []
        for long_root in sorted(members):
            if long_root == short_root:
                continue
            long_points = gathered(members[long_root])
            if len(long_points) < 20 or np.ptp(long_points[:, 0]) < 4 * pitch:
                continue
            if (short_points[:, 0].min() < long_points[:, 0].min() - .5 * pitch or
                    short_points[:, 0].max() > long_points[:, 0].max() + .5 * pitch):
                continue
            long_fit = _fit_all_points(long_points, shared, center, scale, pitch)
            if long_fit is None or long_fit["mad_px"] > .22 * pitch or long_fit["p90_px"] > .45 * pitch:
                continue
            u = (short_points[:, 0] - center) / scale
            k0, b0, c0 = long_fit["coefficients"]
            errors = np.abs(short_points[:, 1] - (k0 + b0 * u + c0 * u * u))
            median_error = float(np.median(errors))
            maximum_error = float(np.max(errors))
            if median_error <= .25 * pitch and maximum_error <= .40 * pitch:
                compatible.append((median_error, maximum_error, long_root,
                                   long_fit["mad_px"], long_fit["p90_px"]))
        compatible.sort()
        if not compatible:
            reasons["short_fragment_no_close_whole_group"] += 1
            continue
        best_median, best_maximum, destination, before_mad, before_p90 = compatible[0]
        if len(compatible) > 1 and compatible[1][0] - best_median < .12 * pitch:
            reasons["short_fragment_attachment_ambiguous"] += 1
            continue
        okay, combined_fit = fit_consistent(members[short_root] + members[destination])
        if not okay:
            reasons["short_attachment_all_point_fit_disagrees"] += 1
            continue
        short_ids = [ids[index] for index in members[short_root]]
        destination_ids = [ids[index] for index in members[destination]]
        merge(short_root, destination)
        accepted_edges.append({"from": short_ids[0], "to": destination_ids[0],
                               "from_group_members_before_merge": short_ids,
                               "to_group_members_before_merge": destination_ids,
                               "type": "unique_short_visible_fragment_attachment",
                               "median_point_to_whole_conic_px": best_median,
                               "max_point_to_whole_conic_px": best_maximum,
                               "before_group_all_point_mad_px": before_mad,
                               "before_group_all_point_p90_px": before_p90,
                               "after_group_all_point_mad_px": combined_fit["mad_px"],
                               "after_group_all_point_p90_px": combined_fit["p90_px"],
                               "second_best_median_px": (compatible[1][0] if len(compatible) > 1 else None)})

    # 阶段2：每个最终 group 以全部原点求一次唯一系数；点的顺序和来源可复核。
    curves = []
    for group_number, root in enumerate(sorted(members, key=lambda r: min(members[r])), 1):
        indices = sorted(members[root])
        source_ids = [ids[i] for i in indices]
        raw_points = []
        refs = []
        for i in indices:
            for point_index, original in enumerate(lines[i].get("observed_points_xy", [])):
                raw_points.append(list(original))
                refs.append({"fragment_id": ids[i], "point_index": point_index})
        points = gathered(indices)
        fit = _fit_all_points(points, shared, center, scale, pitch)
        group_id = f"group_{group_number:04d}"
        group = {"group_id": group_id, "status": "rejected", "source_fragment_ids": source_ids,
                 "observed_points_xy": raw_points, "point_refs": refs,
                 "fit_weights": fit["weights"].tolist() if fit else [None] * len(points),
                 "residuals_px": fit["residuals"].tolist() if fit else [None] * len(points),
                 "fit_regularization": fit["regularization"] if fit else None,
                 "fitted_coefficients_kbc": fit["coefficients"].tolist() if fit else None,
                 "curve_id": None, "reason": None,
                 "observed_point_count": len(points)}
        if len(points) < 2 or len(np.unique(points[:, 0])) < 2:
            reason = "insufficient_distinct_observed_columns"
        elif np.ptp(points[:, 0]) < 4.:
            reason = "observed_span_under_4px"
        elif fit is None:
            reason = "nonfinite_or_unsolved_all_point_fit"
        elif fit["mad_px"] > .22 * pitch or fit["p90_px"] > .45 * pitch:
            reason = "whole_group_visible_residual_too_high"
        else:
            reason = None
        if reason is not None:
            group["reason"] = reason
            result["rejected"].append({"group_id": group_id, "source_fragment_ids": source_ids,
                                       "reason": reason, "point_count": len(points)})
            reasons[reason] += 1
            result["groups"].append(group)
            continue
        k, b, c = map(float, fit["coefficients"])
        if c < 0:
            raise AssertionError("nonnegative curvature solver failed")
        xlo, xhi = float(points[:, 0].min()), float(points[:, 0].max())
        sampled = []
        for x in range(max(0, int(math.ceil(xlo))), min(width - 1, int(math.floor(xhi))) + 1):
            u = (x - center) / scale
            y = int(round(k + b * u + c * u * u))
            if 0 <= y < height and roi[y, x] and not exclusion[y, x]:
                sampled.append([x, y])
        if not sampled:
            group["reason"] = "no_drawable_roi_pixels"
            result["rejected"].append({"group_id": group_id, "source_fragment_ids": source_ids,
                                       "reason": group["reason"], "point_count": len(points)})
            reasons[group["reason"]] += 1
            result["groups"].append(group)
            continue
        curve_id = f"conic_{group_number:04d}"
        axis = center - b * scale / (2 * c) if c > 0 else None
        vertex = k - b * b / (4 * c) if c > 0 else None
        curve = {"curve_id": curve_id, "group_id": group_id,
                 "status": "single_whole_group_conic" if c > 0 else "near_flat_group_line",
                 "source_fragment_ids": source_ids,
                 "model": {"type": "vertical_axis_parabola" if c > 0 else "near_flat_line",
                           "normalized_k": k, "normalized_b": b, "normalized_c": c,
                           "u_center_x_px": center, "u_scale_x_px": scale,
                           "a_px_per_px2": c / scale**2, "h_px": axis,
                           "k_vertex_px": vertex},
                 "observed_span_x_px": [xlo, xhi], "sampled_points_xy": sampled,
                 "support": {"observed_fragment_count": len(indices),
                             "observed_point_count": len(points),
                             "median_residual_px": fit["mad_px"],
                             "p90_residual_px": fit["p90_px"],
                             "all_points_positive_weight": True},
                 "interpretation": "one fitted parabola for all yellow points in this visual group; material identity unknown"}
        group["status"] = "accepted"
        group["curve_id"] = curve_id
        group["reason"] = ("single_fragment_no_unique_partner" if len(indices) == 1 else None)
        group["fit_mad_px"] = fit["mad_px"]
        group["fit_p90_px"] = fit["p90_px"]
        result["groups"].append(group)
        curves.append(curve)

    # 独立组重新拟合后若相交/挤靠，则整组拒识并保留其系数与原点。
    # 不能裁掉交叉列后假装整条接受的单一模型得到了验证。
    by_x = [{x: y for x, y in c["sampled_points_xy"]} for c in curves]
    active = [True for _ in curves]
    conflicts = []
    for i in range(len(curves)):
        for j in range(i + 1, len(curves)):
            if not active[i] or not active[j]:
                continue
            common = by_x[i].keys() & by_x[j].keys()
            if len(common) < 2:
                continue
            signed = [by_x[i][x] - by_x[j][x] for x in common]
            crossing = min(signed) <= 0 <= max(signed)
            close = {x for x in common if abs(by_x[i][x] - by_x[j][x]) < .45 * pitch}
            if not crossing and not close:
                continue
            support_i, support_j = curves[i]["support"], curves[j]["support"]
            # 可见全点残差和点数只能在差异足够明显时选保留者；
            # 近似并列时拒识双方，不凭曲线编号判断真假。
            def quality(support: dict) -> float:
                return ((support["median_residual_px"] + .3 * support["p90_residual_px"]) / pitch
                        - .02 * math.log1p(support["observed_point_count"]))
            difference = quality(support_i) - quality(support_j)
            if abs(difference) > .08:
                losers = [i if difference > 0 else j]
                decision = "weaker_whole_group_rejected"
            else:
                losers = [i, j]
                decision = "ambiguous_both_whole_groups_rejected"
            for loser in losers:
                active[loser] = False
            conflicts.append({"curve_ids": [curves[i]["curve_id"], curves[j]["curve_id"]],
                              "reason": "postfit_group_crossing" if crossing else "postfit_group_spacing",
                              "conflict_column_count": len(common) if crossing else len(close),
                              "rejected_curve_ids": [curves[index]["curve_id"] for index in losers],
                              "decision": decision,
                              "quality_scores": [quality(support_i), quality(support_j)]})
    group_by_id = {g["group_id"]: g for g in result["groups"]}
    kept = []
    for index, curve in enumerate(curves):
        if not active[index]:
            group = group_by_id[curve["group_id"]]
            group["status"] = "rejected"
            group["curve_id"] = None
            group["reason"] = "postfit_whole_group_crossing_or_spacing"
            group["rejected_fitted_model"] = curve["model"]
            result["rejected"].append({"group_id": group["group_id"],
                                       "source_fragment_ids": group["source_fragment_ids"],
                                       "reason": group["reason"],
                                       "fitted_model": curve["model"],
                                       "point_count": group["observed_point_count"]})
            reasons[group["reason"]] += 1
            continue
        for x, y in curve["sampled_points_xy"]:
            smooth[y, x] = 255
        kept.append(curve)
    result["curves"] = kept
    input_count = sum(len(p) for p in fragments)
    group_count = sum(len(g["point_refs"]) for g in result["groups"])
    if input_count != group_count:
        raise AssertionError("not all input points were grouped exactly once")
    result["diagnostics"] = {"status": "ok", "input_fragment_count": len(lines),
                             "input_point_count": input_count, "grouped_point_count": group_count,
                             "group_count": len(result["groups"]),
                             "accepted_group_count": len(kept),
                             "rejected_group_count": sum(g["curve_id"] is None for g in result["groups"]),
                             "accepted_edges": accepted_edges,
                             "fragment_assignment": {fragment_id: group["group_id"]
                                                     for group in result["groups"]
                                                     for fragment_id in group["source_fragment_ids"]},
                             "candidate_overlap_edge_count": len(overlap_edges),
                             "candidate_gap_edge_count": len(candidates),
                             "postfit_conflicts": conflicts, "rejection_counts": dict(reasons),
                             "shared_shape_status": shared_info["status"],
                             "identity_scope": "visual boundary within one image; not physical material identity"}
    return result
