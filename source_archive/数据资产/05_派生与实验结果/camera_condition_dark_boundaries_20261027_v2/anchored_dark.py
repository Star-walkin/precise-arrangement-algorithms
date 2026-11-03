"""D2 锚定分支：用 C3 亮区真实黄点限定暗区边界的行序。

此模块不读文件、标签或设备。C3 曲线只给预测位置；暗区黄点必须重新
通过本图原图的双侧暗谷检验。长无证据区留空，任何曲线都不证明材料身份。
"""
from __future__ import annotations

import math

import cv2
import numpy as np


PARAMETERS = {
    "column_step_pitch": .25,
    "normal_search_pitch": .40,
    "tangent_span_pitch": 1.5,
    "flank_pitch": .35,
    "anchor_nms_pitch": .60,
    "max_shared_step_pitch": .10,
    "max_individual_shift_pitch": .20,
    "max_inferred_gap_pitch": 3.0,
    "reentry_gap_pitch": 5.0,
    "minimum_valid_tangent_fraction": .68,
    "minimum_segment_points": 6,
    "minimum_segment_span_pitch": 1.5,
}


def _map(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    shape = x.shape
    a = cv2.remap(image, x.astype(np.float32).reshape(1, -1),
                  y.astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_REFLECT_101)
    return a.reshape(shape)


def _tangent_average(image: np.ndarray, valid: np.ndarray | None,
                     x: np.ndarray, y: np.ndarray, slope: float,
                     pitch: float) -> tuple[np.ndarray, np.ndarray]:
    """沿共同切向取7点；增强图只用暗ROI内值，原图不改动。"""
    offsets = np.linspace(-1.5, 1.5, 7, dtype=np.float32)*pitch
    weights = np.exp(-.5*(offsets/(.9*pitch))**2).astype(np.float32)
    weights /= weights.sum()
    numerator = np.zeros_like(x, np.float32)
    coverage = np.zeros_like(x, np.float32)
    for dx, weight in zip(offsets, weights):
        xx, yy = x+dx, y+slope*dx
        if valid is None:
            local = 1.
        else:
            local = _map(valid, xx, yy)
        numerator += weight*_map(image, xx, yy)*local
        coverage += weight*local
    return numerator/np.maximum(coverage, .01), coverage


def _anchors(c3_record: dict, pitch: float) -> tuple[list[dict], dict]:
    fit = c3_record.get("fit", c3_record)
    # C3可合法拒识并保存model=None；此时没有可信亮区锚，暗区也应拒识。
    model = (fit.get("model") or {}).get("shared_shape_for_grouping_and_regularization", {})
    if model.get("status") not in ("positive_shared_parabola", "near_flat_shared_shape"):
        return [], {"reason": "no_accepted_c3_shared_shape"}
    center = float(model["center_x_px"])
    scale = float(model["scale_x_px"])
    b = float(model["b_px_per_u"])
    c = float(model["c_px_per_u2"])
    if not all(math.isfinite(v) for v in (center, scale, b, c)) or scale <= 0:
        return [], {"reason": "invalid_c3_shared_shape"}
    def shape(x):
        u = (np.asarray(x, float)-center)/scale
        return b*u+c*u*u
    groups = []
    for group in fit.get("groups", []):
        if group.get("status") != "accepted":
            continue
        points = np.asarray(group.get("observed_points_xy", []), float)
        if points.ndim != 2 or points.shape[1] < 2 or len(points) < 8:
            continue
        if not np.isfinite(points[:, :2]).all():
            continue
        offset = float(np.median(points[:, 1]-shape(points[:, 0])))
        mad = float(np.median(np.abs(points[:, 1]-shape(points[:, 0])-offset)))
        if mad > .30*pitch:
            continue
        groups.append({"group_id": str(group["group_id"]), "offset": offset,
                       "anchor_point_count": len(points), "anchor_mad_px": mad,
                       "x_span": [float(points[:, 0].min()), float(points[:, 0].max())]})
    groups.sort(key=lambda g: g["offset"])
    # 近于半绳距的相邻C3组可能重复落在同一绳的股纹上，仅保留较稳定者。
    kept = []
    rejected = []
    for group in sorted(groups, key=lambda g: (-g["anchor_point_count"], g["anchor_mad_px"])):
        near = next((other for other in kept if
                     abs(group["offset"]-other["offset"]) <
                     PARAMETERS["anchor_nms_pitch"]*pitch), None)
        if near is None:
            kept.append(group)
        else:
            rejected.append({"group_id": group["group_id"],
                             "reason": "near_duplicate_c3_anchor", "kept": near["group_id"]})
    kept.sort(key=lambda g: g["offset"])
    return kept, {"status": "ok", "shared_shape": {"center_x_px": center,
            "scale_x_px": scale, "b_px_per_u": b, "c_px_per_u2": c},
            "accepted_c3_groups": len(groups), "retained_anchor_count": len(kept),
            "rejected_anchors": rejected}


def _measure_column(raw: np.ndarray, enhanced: np.ndarray, valid: np.ndarray,
                    noise: np.ndarray, x: int, predicted: np.ndarray,
                    slope: float, pitch: float) -> list[dict | None]:
    """各已知行只在±0.4p内搜索；双侧原图暗谷是唯一黄色观测入口。"""
    h, w = raw.shape
    radius = int(math.ceil(PARAMETERS["normal_search_pitch"]*pitch))
    offsets = np.arange(-radius, radius+1, dtype=np.float32)
    rows = len(predicted)
    xx = np.full((rows, len(offsets)), float(x), np.float32)
    yy = predicted[:, None].astype(np.float32)+offsets[None, :]
    flank = PARAMETERS["flank_pitch"]*pitch
    en_center, cv = _tangent_average(enhanced, valid, xx, yy, slope, pitch)
    en_a, av = _tangent_average(enhanced, valid, xx, yy-flank, slope, pitch)
    en_b, bv = _tangent_average(enhanced, valid, xx, yy+flank, slope, pitch)
    ra, _ = _tangent_average(raw, None, xx, yy-flank, slope, pitch)
    rb, _ = _tangent_average(raw, None, xx, yy+flank, slope, pitch)
    rc, _ = _tangent_average(raw, None, xx, yy, slope, pitch)
    rnear_a, _ = _tangent_average(raw, None, xx, yy-2., slope, pitch)
    rnear_b, _ = _tangent_average(raw, None, xx, yy+2., slope, pitch)
    local_noise = _map(noise, xx, yy)
    en_weak = np.minimum(en_a-en_center, en_b-en_center)
    en_strong = np.maximum(en_a-en_center, en_b-en_center)
    raw_weak = np.minimum(ra-rc, rb-rc)
    raw_strong = np.maximum(ra-rc, rb-rc)
    response = .72*np.maximum(en_weak, 0.)+.28*np.maximum(en_strong, 0.)
    valid_fraction = PARAMETERS["minimum_valid_tangent_fraction"]
    inside = ((yy >= 1) & (yy < h-1) & (xx >= 0) & (xx < w) &
              (_map(valid, xx, yy) > .5))
    en_response_limit = np.maximum(1.15, .9*local_noise)
    en_weak_limit = np.maximum(.38, .28*local_noise)
    en_strong_limit = np.maximum(1.05, .70*local_noise)
    raw_weak_limit = np.maximum(.25, .28*local_noise)
    raw_strong_limit = np.maximum(.65, .55*local_noise)
    passed = (inside & (cv >= valid_fraction) & (av >= valid_fraction) &
              (bv >= valid_fraction) & (response >= en_response_limit) &
              (en_weak >= en_weak_limit) & (en_strong >= en_strong_limit) &
              (raw_weak >= raw_weak_limit) & (raw_strong >= raw_strong_limit) &
              (rc <= np.minimum(rnear_a, rnear_b)+.55))
    score = (response/np.maximum(1., local_noise) +
             .6*np.maximum(raw_weak, 0.)/np.maximum(.6, local_noise) -
             .5*(offsets[None, :]/pitch)**2)
    score[~passed] = -np.inf
    result: list[dict | None] = []
    for i in range(rows):
        j = int(np.argmax(score[i]))
        if not np.isfinite(score[i, j]):
            result.append(None)
        else:
            result.append({"x": x, "y": float(yy[i, j]), "score": float(score[i, j]),
                           "search_residual_px": float(offsets[j]),
                           "raw_weak": float(raw_weak[i, j]),
                           "raw_strong": float(raw_strong[i, j]),
                           "raw_weak_threshold": float(raw_weak_limit[i, j]),
                           "raw_strong_threshold": float(raw_strong_limit[i, j]),
                           "enhanced_weak": float(en_weak[i, j]),
                           "enhanced_strong": float(en_strong[i, j]),
                           "enhanced_response_threshold": float(en_response_limit[i, j]),
                           "enhanced_weak_threshold": float(en_weak_limit[i, j]),
                           "enhanced_strong_threshold": float(en_strong_limit[i, j]),
                           "local_raw_noise": float(local_noise[i, j]),
                           "valid_fraction": float(min(cv[i, j], av[i, j], bv[i, j]))})
    return result


def _bspline_basis(x: np.ndarray, xlo: float, xhi: float,
                   controls: int) -> tuple[np.ndarray, np.ndarray]:
    """三次端点夹持B样条基；与D2主分支同一Cox–de Boor定义。"""
    inner = np.linspace(0., 1., controls-2)[1:-1]
    knots = np.r_[np.zeros(4), inner, np.ones(4)]
    u = np.clip((np.asarray(x, float)-xlo)/max(xhi-xlo, 1.), 0., 1.)
    end = u >= 1.
    u = np.where(end, np.nextafter(1., 0.), u)
    basis = ((u[:, None] >= knots[:-1]) &
             (u[:, None] < knots[1:])).astype(float)
    for degree in range(1, 4):
        next_basis = np.zeros((len(u), basis.shape[1]-1), float)
        for i in range(next_basis.shape[1]):
            left = knots[i+degree]-knots[i]
            right = knots[i+degree+1]-knots[i+1]
            if left > 0:
                next_basis[:, i] += (u-knots[i])/left*basis[:, i]
            if right > 0:
                next_basis[:, i] += (knots[i+degree+1]-u)/right*basis[:, i+1]
        basis = next_basis
    basis = basis[:, :controls]
    if np.any(end):
        basis[end] = 0.
        basis[end, -1] = 1.
    return basis, knots


def _shared_spline(x: np.ndarray, y: np.ndarray, support: np.ndarray,
                   pitch: float) -> dict | None:
    """逐列多行中位位移，用行数加权Huber及二阶差分正则拟合。"""
    if len(x) < 4 or np.ptp(x) < 1.5*pitch:
        return None
    xlo, xhi = float(x.min()), float(x.max())
    controls = int(np.clip(math.ceil((xhi-xlo)/(6*pitch))+3,
                           4, min(24, len(x))))
    basis, knots = _bspline_basis(x, xlo, xhi, controls)
    d2 = np.diff(np.eye(controls), n=2, axis=0)
    count_mid = max(float(np.median(support)), 1.)
    base = np.clip(support/count_mid, .3, 3.)
    penalty = float(np.clip(4.*math.sqrt(len(x)/30.), 4., 20.))
    huber = max(1., .20*pitch)
    regularizer = penalty*(d2.T@d2)+1e-7*np.eye(controls)
    weights = base.copy()
    for _ in range(7):
        normal = basis.T@(weights[:, None]*basis)+regularizer
        rhs = basis.T@(weights*y)
        try:
            coefficients = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(normal, rhs, rcond=None)[0]
        residual = y-basis@coefficients
        updated = base*np.clip(np.minimum(1., huber/np.maximum(np.abs(residual), 1e-9)),
                               .08, 1.)
        if np.max(np.abs(updated-weights)) < 1e-3:
            weights = updated
            break
        weights = updated
    normal = basis.T@(weights[:, None]*basis)+regularizer
    rhs = basis.T@(weights*y)
    try:
        coefficients = np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(normal, rhs, rcond=None)[0]
    if not np.isfinite(coefficients).all():
        return None
    return {"type": "cubic_clamped_B_spline", "degree": 3,
            "xlo": xlo, "xhi": xhi, "control_count": controls,
            "knots_normalized": knots.tolist(),
            "coefficients_px": coefficients.tolist(),
            "support_columns": len(x), "support_rows_median": count_mid,
            "regularization": {"control_spacing_pitch": 6.,
                "second_difference_lambda": penalty, "huber_delta_px": huber,
                "weight_rule": "clip(column_support_rows / median_rows, .3, 3)",
                "robust_rule": "clip(min(1,huber/abs(residual)),.08,1)"}}


def _row_kernel(points: list[dict], xlo: int, xhi: int,
                family, shared_curve: np.ndarray, scale_curve: np.ndarray,
                offset: float, reference_offset: float,
                pitch: float, amplitude: float) -> tuple[np.ndarray, dict, np.ndarray]:
    """一行原图点的3p高斯核回归；只约束导数，相邻行后续有序投影。"""
    length = xhi-xlo+1
    signal = np.zeros(length, np.float32)
    weights = np.zeros(length, np.float32)
    for p in points:
        x = int(p["x"])
        if not xlo <= x <= xhi:
            continue
        signal[x-xlo] += (p["y"]-float(family(x))-offset-shared_curve[x]-
                          scale_curve[x]*(offset-reference_offset))
        weights[x-xlo] += 1.
    sigma = max(2., 3.*pitch)
    numerator = cv2.GaussianBlur(signal.reshape(1, -1), (0, 0),
                                 sigmaX=sigma, borderType=cv2.BORDER_CONSTANT).ravel()
    denominator = cv2.GaussianBlur(weights.reshape(1, -1), (0, 0),
                                   sigmaX=sigma, borderType=cv2.BORDER_CONSTANT).ravel()
    values = np.divide(numerator, denominator, out=np.zeros_like(numerator),
                       where=denominator > 1e-6)
    # 不截断振幅；原图黄点若相对共享模型偏移较大，也应让拟合靠近它。
    # 不允许相邻像素的独立行形变突然反向；共同样条承担大曲率。
    max_step = .05
    for i in range(1, length):
        values[i] = np.clip(values[i], values[i-1]-max_step,
                            values[i-1]+max_step)
    for i in range(length-2, -1, -1):
        values[i] = np.clip(values[i], values[i+1]-max_step,
                            values[i+1]+max_step)
    nodes = np.flatnonzero(weights > 0)
    metadata = {"type": "gaussian_kernel_regression_with_slope_projection",
                "xlo": xlo, "xhi": xhi, "sigma_px": sigma,
                "kernel_input_x_px": (nodes+xlo).tolist(),
                "kernel_input_sum_residual_px": signal[nodes].tolist(),
                "kernel_input_count": weights[nodes].tolist(),
                "former_amplitude_limit_px_diagnostic_only": amplitude,
                "amplitude_clip_applied": False,
                "max_abs_individual_derivative_px_per_px": max_step,
                "border_rule": "constant_zero; numerator / denominator"}
    return values, metadata, denominator


def _ordered_projection(y: np.ndarray, confidence: np.ndarray,
                        min_gap: float) -> np.ndarray:
    """带权PAVA：投影到不换序、相邻边界间距至少min_gap的集合。"""
    transformed = y-min_gap*np.arange(len(y))
    blocks = []
    for i, (value, weight) in enumerate(zip(transformed, confidence)):
        blocks.append([i, i, float(value*weight), float(weight)])
        while (len(blocks) > 1 and
               blocks[-2][2]/blocks[-2][3] > blocks[-1][2]/blocks[-1][3]):
            current = blocks.pop()
            previous = blocks[-1]
            previous[1] = current[1]
            previous[2] += current[2]
            previous[3] += current[3]
    result = np.empty(len(y), float)
    for first, last, numerator, denominator in blocks:
        result[first:last+1] = numerator/denominator
    return result+min_gap*np.arange(len(y))


def detect_anchored(original_bgr: np.ndarray, enhanced_bgr: np.ndarray,
                    dark_mask: np.ndarray, pitch_px: float,
                    c3_record: dict) -> dict:
    """返回旧runner可画/存的 observed_lines、fit及3个二值mask。"""
    if (original_bgr.ndim != 3 or original_bgr.shape[2] != 3 or
            original_bgr.dtype != np.uint8 or enhanced_bgr.shape != original_bgr.shape or
            enhanced_bgr.dtype != np.uint8 or dark_mask.shape != original_bgr.shape[:2] or
            dark_mask.dtype != np.uint8 or np.any((dark_mask != 0) & (dark_mask != 255))):
        raise ValueError("original/enhanced BGR and dark uint8 0/255 mask must align")
    pitch = float(pitch_px)
    if not math.isfinite(pitch) or not 8 <= pitch <= 70:
        raise ValueError("pitch_px outside [8,70]")
    h, w = dark_mask.shape
    observed = np.zeros((h, w), np.uint8)
    smooth = np.zeros_like(observed)
    inferred = np.zeros_like(observed)
    anchors, anchor_diag = _anchors(c3_record, pitch)
    empty = {"observed_lines": [], "observed_mask": observed,
             "fit": {"model": "c3_anchor_shared_deformation", "curves": [],
                     "smooth_mask": smooth, "diagnostics": {"status": "no_anchors_or_dark_roi"}},
             "inferred_mask": inferred, "diagnostics": {"anchors": anchor_diag,
                     "status": "no_anchors_or_dark_roi", "parameters": PARAMETERS.copy()}}
    if not anchors or not np.any(dark_mask):
        return empty
    for i, anchor in enumerate(anchors):
        gaps = []
        if i:
            gaps.append(anchor["offset"]-anchors[i-1]["offset"])
        if i+1 < len(anchors):
            gaps.append(anchors[i+1]["offset"]-anchor["offset"])
        nearest = min(gaps) if gaps else float("inf")
        anchor["individual_limit_px"] = float(max(0., min(.2*pitch,
                                                (nearest-.5*pitch)/2.)))
    m = anchor_diag["shared_shape"]
    center, scale = m["center_x_px"], m["scale_x_px"]
    b, c = m["b_px_per_u"], m["c_px_per_u2"]
    def family(x):
        u = (np.asarray(x, float)-center)/scale
        return b*u+c*u*u
    def derivative(x):
        return (b+2*c*(np.asarray(x, float)-center)/scale)/scale
    raw = cv2.GaussianBlur(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
                           .astype(np.float32), (3, 3), .65)
    enhanced = cv2.GaussianBlur(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
                                .astype(np.float32), (0, 0), max(1.2, .13*pitch))
    valid = (dark_mask > 0).astype(np.float32)
    noise = cv2.GaussianBlur(np.abs(raw-cv2.GaussianBlur(raw, (3, 3), .9)),
                             (0, 0), max(2., .22*pitch))
    valid_x = np.flatnonzero(np.any(dark_mask > 0, axis=0))
    step = max(4, int(round(PARAMETERS["column_step_pitch"]*pitch)))
    anchor_min = min(a["x_span"][0] for a in anchors)
    anchor_max = max(a["x_span"][1] for a in anchors)
    # 亮区中部不产生暗区观测；从两侧离亮区最近的列分别走向暗面外缘。
    left = valid_x[valid_x < anchor_min][::-1][::step]
    right = valid_x[valid_x > anchor_max][::step]
    all_points = {a["group_id"]: [] for a in anchors}
    side_diagnostics = []
    for side_name, columns in (("left", left), ("right", right)):
        shared = 0.
        individual = np.zeros(len(anchors), float)
        last_seen = np.full(len(anchors), np.nan)
        accepted_columns = 0
        ambiguity_rejects = 0
        no_support_columns = 0
        shared_history = []
        for x_value in columns:
            x = int(x_value)
            predicted = (float(family(x))+np.asarray([a["offset"] for a in anchors])+
                         shared+individual)
            # 最近多列共同位移的变化也是整组切向；所有行使用同一斜率。
            if len(shared_history) >= 4:
                recent = np.asarray(shared_history[-5:], float)
                shared_slope = float(np.median(np.diff(recent[:, 1]) /
                                               np.diff(recent[:, 0])))
            else:
                shared_slope = 0.
            slope = float(np.clip(derivative(x)+shared_slope, -1.2, 1.2))
            proposed = _measure_column(raw, enhanced, valid, noise, x,
                                       predicted, slope, pitch)
            # 同列一对一；间距不足时只保留更强证据，不能让两行交换身份。
            order = sorted((i for i, p in enumerate(proposed) if p is not None),
                           key=lambda i: proposed[i]["score"], reverse=True)
            selected = []
            for i in order:
                point = proposed[i]
                if any(abs(point["y"]-proposed[j]["y"]) < .60*pitch
                       for j in selected):
                    ambiguity_rejects += 1
                else:
                    selected.append(i)
            if len(selected) < 4:
                # 单根孤立谷不足以重启长时间失测的身份轨。
                selected = [i for i in selected if not math.isfinite(last_seen[i]) or
                            abs(x-last_seen[i]) <= PARAMETERS["reentry_gap_pitch"]*pitch]
            residuals = np.asarray([proposed[i]["y"]-predicted[i] for i in selected], float)
            if len(residuals) >= 3 and np.median(np.abs(residuals-np.median(residuals))) <= .2*pitch:
                step_shift = float(np.clip(np.median(residuals),
                               -PARAMETERS["max_shared_step_pitch"]*pitch,
                               PARAMETERS["max_shared_step_pitch"]*pitch))
                shared += step_shift
            else:
                step_shift = 0.
            if selected:
                accepted_columns += 1
            else:
                no_support_columns += 1
            for i in selected:
                point = proposed[i]
                residual = point["y"]-(float(family(x))+anchors[i]["offset"]+shared)
                individual[i] = float(np.clip(.75*individual[i]+.25*residual,
                                  -anchors[i]["individual_limit_px"],
                                  anchors[i]["individual_limit_px"]))
                point["side"] = side_name
                point["c3_group_id"] = anchors[i]["group_id"]
                point["shared_shift_px"] = round(shared, 3)
                point["individual_shift_px"] = round(individual[i], 3)
                all_points[anchors[i]["group_id"]].append(point)
                last_seen[i] = x
            shared_history.append([x, round(shared, 3), len(selected)])
        side_diagnostics.append({"side": side_name, "sampled_columns": len(columns),
                                 "supported_columns": accepted_columns,
                                 "no_support_columns": no_support_columns,
                                 "same_column_spacing_rejects": ambiguity_rejects,
                                 "shared_shift_history": shared_history})
    # 原图提名仍全部保留；孤立短段不进入有效黄点/拟合。
    accepted_points: dict[str, list[dict]] = {a["group_id"]: [] for a in anchors}
    rejected_segments = []
    for anchor in anchors:
        for side_name in ("left", "right"):
            points = sorted((p for p in all_points[anchor["group_id"]]
                             if p["side"] == side_name), key=lambda p:p["x"])
            runs = []
            current = []
            for p in points:
                if current and p["x"]-current[-1]["x"] > 3.*pitch:
                    runs.append(current)
                    current = []
                current.append(p)
            if current:
                runs.append(current)
            for run in runs:
                span = run[-1]["x"]-run[0]["x"]
                if (len(run) >= PARAMETERS["minimum_segment_points"] and
                        span >= PARAMETERS["minimum_segment_span_pitch"]*pitch):
                    accepted_points[anchor["group_id"]].extend(run)
                else:
                    rejected_segments.append({"c3_group_id": anchor["group_id"],
                        "side": side_name, "reason": "short_isolated_original_candidate_run",
                        "point_count": len(run), "span_px": span,
                        "original_candidate_evidence": run})
    # 逐列多行稳健估计共享平移与轻微行距缩放；不是每行独立追细股。
    reference_offset = float(np.median([a["offset"] for a in anchors]))
    offset_scale = max(float(np.std([a["offset"] for a in anchors])), 3.*pitch)
    shared_curve = np.zeros(w, np.float32)
    scale_curve = np.zeros(w, np.float32)
    shared_models = []
    for side_name in ("left", "right"):
        bins: list[list[tuple[float, float]]] = [[] for _ in range(w)]
        for anchor in anchors:
            for p in accepted_points[anchor["group_id"]]:
                if p["side"] == side_name:
                    x = p["x"]
                    bins[x].append((anchor["offset"],
                                    p["y"]-float(family(x))-anchor["offset"]))
        translation_x, translation_y, translation_count = [], [], []
        scale_x, scale_y, scale_count = [], [], []
        for x, values in enumerate(bins):
            if len(values) < 3:
                continue
            pairs = np.asarray(values, float)
            offsets = pairs[:, 0]
            residual = pairs[:, 1]
            centered = (offsets-reference_offset)/offset_scale
            if len(values) >= 5 and np.ptp(offsets) >= 4.*pitch:
                design = np.c_[np.ones(len(values)), centered]
                beta = np.linalg.lstsq(design, residual, rcond=None)[0]
                for _ in range(4):
                    error = residual-design@beta
                    robust = np.minimum(1., max(1., .16*pitch)/
                                        np.maximum(np.abs(error), 1e-6))
                    beta = np.linalg.lstsq(design*np.sqrt(robust[:, None]),
                                           residual*np.sqrt(robust), rcond=None)[0]
                intercept, coefficient = map(float, beta)
                scale_x.append(x)
                scale_y.append(coefficient)
                scale_count.append(len(values))
            else:
                intercept = float(np.median(residual))
            translation_x.append(x)
            translation_y.append(intercept)
            translation_count.append(len(values))
        sx = np.asarray(translation_x, float)
        model = _shared_spline(sx, np.asarray(translation_y, float),
                               np.asarray(translation_count, float), pitch)
        if model is None:
            shared_models.append({"side": side_name, "status": "insufficient_multiline_support",
                                  "support_columns": len(sx)})
            continue
        sample_x = np.arange(max(0, int(math.ceil(model["xlo"]))),
                             min(w-1, int(math.floor(model["xhi"])))+1)
        basis, _ = _bspline_basis(sample_x, model["xlo"], model["xhi"],
                                   model["control_count"])
        shared_curve[sample_x] = basis@np.asarray(model["coefficients_px"])
        scale_model = _shared_spline(np.asarray(scale_x, float),
                   np.asarray(scale_y, float), np.asarray(scale_count, float), pitch)
        if scale_model is not None:
            scale_xs = np.arange(max(0, int(math.ceil(scale_model["xlo"]))),
                                 min(w-1, int(math.floor(scale_model["xhi"])))+1)
            scale_basis, _ = _bspline_basis(scale_xs, scale_model["xlo"],
                                           scale_model["xhi"],
                                           scale_model["control_count"])
            scale_curve[scale_xs] = np.clip(
                scale_basis@np.asarray(scale_model["coefficients_px"])/offset_scale,
                -.15, .15)
        shared_models.append({"side": side_name, "status": "fitted",
                              "translation": model,
                              "row_scale": scale_model if scale_model is not None
                              else {"status": "insufficient_multiline_span", "scale": 0.},
                              "row_scale_bounds": [.85, 1.15]})
    observed_lines = []
    curves = []
    max_gap = PARAMETERS["max_inferred_gap_pitch"]*pitch
    for anchor in anchors:
        points = sorted(accepted_points[anchor["group_id"]], key=lambda p: p["x"])
        if len(points) < PARAMETERS["minimum_segment_points"]:
            continue
        points = [p for p in points if 0 <= int(round(p["y"])) < h and
                  dark_mask[int(round(p["y"])), p["x"]]]
        coords = [[int(p["x"]), int(round(p["y"]))] for p in points]
        if len(coords) < PARAMETERS["minimum_segment_points"]:
            continue
        line_id = "dark_"+anchor["group_id"]
        evidence = [{"x": xy[0], "y": xy[1], "c3_group_id": anchor["group_id"],
                     **{k: p[k] for k in ("score", "search_residual_px", "raw_weak",
                        "raw_strong", "raw_weak_threshold", "raw_strong_threshold",
                        "enhanced_weak", "enhanced_strong", "enhanced_response_threshold",
                        "enhanced_weak_threshold", "enhanced_strong_threshold",
                        "local_raw_noise", "valid_fraction", "side")}}
                    for xy, p in zip(coords, points)]
        observed_lines.append({"line_id": line_id, "source_c3_group_id": anchor["group_id"],
                               "observed_points_xy": coords,
                               "original_point_evidence": evidence,
                               "anchor_yellow_points": anchor["anchor_point_count"],
                               "individual_limit_px": anchor["individual_limit_px"],
                               "raw_support": "original_grayscale_bilateral_valley"})
        for x, y in coords:
            cv2.circle(observed, (x, y), 1, 255, -1)
        xp = np.asarray([p["x"] for p in points], int)
        unique_x = np.unique(xp)
        row_correction = np.zeros(w, np.float32)
        row_density = np.zeros(w, np.float32)
        row_models = []
        for side_name in ("left", "right"):
            side_points = [p for p in points if p["side"] == side_name]
            if not side_points:
                continue
            xlo = min(p["x"] for p in side_points)
            xhi = max(p["x"] for p in side_points)
            position = anchors.index(anchor)
            gaps = []
            if position:
                gaps.append(anchor["offset"]-anchors[position-1]["offset"])
            if position+1 < len(anchors):
                gaps.append(anchors[position+1]["offset"]-anchor["offset"])
            nearest_gap = min(gaps) if gaps else float("inf")
            corrected_min_gap = nearest_gap*float(np.min(1.+scale_curve[xlo:xhi+1]))
            amplitude = float(max(0., min(.2*pitch,
                                          (corrected_min_gap-.5*pitch)/2.)))
            values, row_model, density = _row_kernel(
                side_points, xlo, xhi, family, shared_curve, scale_curve,
                anchor["offset"], reference_offset, pitch, amplitude)
            row_correction[xlo:xhi+1] = values
            row_density[xlo:xhi+1] = density
            row_models.append({"side": side_name,
                               "corrected_min_neighbor_gap_px": corrected_min_gap,
                               **row_model})
        samples = []
        states = []
        projection_density = []
        for a, z in zip(unique_x[:-1], unique_x[1:]):
            if z-a > max_gap:
                continue
            for x in range(int(a), int(z)+1):
                if samples and x <= samples[-1][0]:
                    continue
                delta = float(row_correction[x])
                y = (float(family(x))+anchor["offset"]+shared_curve[x]+
                     scale_curve[x]*(anchor["offset"]-reference_offset)+delta)
                yi = int(round(y))
                if not (0 <= yi < h and dark_mask[yi, x]):
                    continue
                samples.append([x, float(y)])
                projection_density.append(float(row_density[x]))
                nearest = min(x-a, z-x)
                states.append("observed_supported_fit" if nearest <= .45*pitch
                              else "inferred_short_gap")
        curves.append({"curve_id": line_id, "source_c3_group_id": anchor["group_id"],
                       "sampled_points_xy": samples, "point_states": states,
                       "observed_point_count": len(coords),
                       "_projection_density": projection_density,
                       "model": {"type": "C3_shared_parabola_plus_dark_B_spline_plus_row_kernel",
                                 "c3_offset_px": anchor["offset"],
                                 "shared_row_reference_offset_px": reference_offset,
                                 "shared_row_offset_scale_px": offset_scale,
                                 "row_models": row_models},
                       "interpretation": "dark observations in yellow; short supported interpolation only"})
    # 所有行的黄点原封不动；仅对待显示的平滑线做置信度加权有序投影。
    by_x: dict[int, list[tuple[float, int, int, float, float]]] = {}
    line_by_id = {line["line_id"]: line for line in observed_lines}
    scores = [float(np.median([e["score"] for e in line["original_point_evidence"]]))
              for line in observed_lines]
    score_reference = max(float(np.median(scores)), .01) if scores else 1.
    for ci, curve in enumerate(curves):
        evidence = line_by_id[curve["curve_id"]]["original_point_evidence"]
        score = float(np.median([e["score"] for e in evidence]))
        density = np.asarray(curve.pop("_projection_density"), float)
        positive = density[density > 0]
        middle = max(float(np.median(positive)), 1e-6) if len(positive) else 1.
        for pi, ((x, y), local_density) in enumerate(
                zip(curve["sampled_points_xy"], density)):
            weight = float(np.clip(score/score_reference, .5, 2.)*
                           np.clip(local_density/middle, .5, 2.))
            by_x.setdefault(x, []).append((curve["model"]["c3_offset_px"],
                                             ci, pi, y, weight))
    projection_gap = float(math.ceil(.5*pitch))
    projection_changes = []
    projection_violations = 0
    projection_pairs = 0
    for x, entries in by_x.items():
        entries.sort(key=lambda e:e[0])
        yy = np.asarray([e[3] for e in entries], float)
        confidence = np.asarray([e[4] for e in entries], float)
        if len(yy) > 1:
            projection_violations += int(np.count_nonzero(np.diff(yy) < projection_gap))
            projection_pairs += len(yy)-1
        projected = _ordered_projection(yy, confidence, projection_gap)
        for entry, new_y in zip(entries, projected):
            _, ci, pi, old_y, _ = entry
            change = float(new_y-old_y)
            projection_changes.append(abs(change))
            curves[ci]["sampled_points_xy"][pi][1] = float(new_y)
            if abs(change) > 1e-5:
                curves[ci]["model"].setdefault("ordered_projection_adjustments", []).append(
                    [x, round(change, 5)])
    projection_removed_outside_roi = 0
    for curve in curves:
        retained = []
        retained_states = []
        fitted_y_by_x = {}
        for (x, y), state in zip(curve["sampled_points_xy"], curve["point_states"]):
            yi = int(round(y))
            if not (0 <= yi < h and dark_mask[yi, x]):
                projection_removed_outside_roi += 1
                continue
            fitted_y_by_x[x] = float(y)
            retained.append([x, yi])
            retained_states.append(state)
        curve["sampled_points_xy"] = retained
        curve["point_states"] = retained_states
        for k in range(len(retained)-1):
            a, z = retained[k], retained[k+1]
            if z[0]-a[0] != 1 or abs(z[1]-a[1]) > 4:
                continue
            cv2.line(smooth, tuple(a), tuple(z), 255, 1)
            if (retained_states[k] == "inferred_short_gap" or
                    retained_states[k+1] == "inferred_short_gap"):
                cv2.line(inferred, tuple(a), tuple(z), 255, 1)
        line = line_by_id[curve["curve_id"]]
        errors = [abs(y-fitted_y_by_x[x]) for x, y in line["observed_points_xy"]
                  if x in fitted_y_by_x]
        if errors:
            error = np.asarray(errors, float)
            curve["fit_to_observed_residual"] = {
                "median_px": float(np.median(error)),
                "p90_px": float(np.quantile(error, .90)),
                "max_px": float(np.max(error)),
                "median_pitch": float(np.median(error)/pitch),
                "p90_pitch": float(np.quantile(error, .90)/pitch),
                "max_pitch": float(np.max(error)/pitch),
                "support_point_count": len(error),
                "unpaired_yellow_point_count": len(line["observed_points_xy"])-len(error),
                "quality": "unreliable" if np.quantile(error, .90) > .25*pitch
                           else "within_smoothing_budget"}
        else:
            curve["fit_to_observed_residual"] = {"quality": "unreliable",
                "reason": "no_drawable_curve_at_original_yellow_points",
                "support_point_count": 0,
                "unpaired_yellow_point_count": len(line["observed_points_xy"])}
        curve["model"]["ordered_projection"] = {
            "method": "confidence_weighted_PAVA_at_each_sample_x",
            "min_gap_px": projection_gap,
            "confidence": "clip(median_original_score/image_median_score,.5,2)*clip(local_kernel_density/line_median_density,.5,2)",
            "model_reconstruction": "reconstruct row kernel, add listed per-x projection adjustments, then round"}
    observed[dark_mask == 0] = 0
    smooth[dark_mask == 0] = 0
    inferred[dark_mask == 0] = 0
    inferred[observed > 0] = 0
    diagnostics = {"status": "completed", "anchors": anchor_diag,
                   "sides": side_diagnostics, "observed_line_count": len(observed_lines),
                   "observed_point_count": sum(len(z["observed_points_xy"]) for z in observed_lines),
                   "unreliable_fitted_curve_count": sum(
                       curve["fit_to_observed_residual"]["quality"] == "unreliable"
                       for curve in curves),
                   "ordered_projection": {"min_gap_px": projection_gap,
                       "preprojection_close_pair_count": projection_violations,
                       "preprojection_pair_count": projection_pairs,
                       "sampled_point_count": len(projection_changes),
                       "moved_over_half_pixel_count": int(np.count_nonzero(
                           np.asarray(projection_changes) > .5)),
                       "max_adjustment_px": float(max(projection_changes, default=0.)),
                       "removed_outside_dark_roi_count": projection_removed_outside_roi},
                   "rejected_original_candidate_segments": rejected_segments,
                   "rejected_original_candidate_point_count": sum(
                       item["point_count"] for item in rejected_segments),
                   "parameters": PARAMETERS.copy(),
                   "meaning": "C3 yellow anchors propose identity order; dark yellow requires local original image evidence"}
    return {"observed_lines": observed_lines, "observed_mask": observed,
            "fit": {"model": {"type": "c3_anchor_shared_deformation",
                      "shared_deformation_models": shared_models,
                      "row_reference_offset_px": reference_offset,
                      "row_offset_scale_px": offset_scale,
                      "row_scale_formula": "y=f0(x)+anchor_offset+translation_spline(x)+row_scale_delta(x)*(anchor_offset-reference_offset)+unclipped_3p_row_kernel(x)+ordered_projection_adjustment(x)"}, "curves": curves,
                    "smooth_mask": smooth, "diagnostics": diagnostics},
            "inferred_mask": inferred, "diagnostics": diagnostics}


__all__ = ["PARAMETERS", "detect_anchored"]
