"""V27 causal free-rope *appearance* diagnostics, never material speed or t1.

Coordinates are original-image pixels. Each cohort has one fixed geometric
reference (the native crop centre), one fixed s sampling, and no resizing.
The two predeclared configurations differ ONLY in whether the independently
observed line pose is used for conditional swing compensation. Current/past
geometry is an observation, not material identity or metric calibration.

Positive displacement means image texture moved in the initial seed's direction
towards its visible entry endpoint. Periodic peaks are preserved, not unwrapped.
The phase-spectrum implementation retains the full linear-lag multi-peak curve;
unlike cv2.phaseCorrelate it does not select one argmax as a speed measurement.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

try:  # Both root's package import and standalone pytest/CLI are supported.
    from .paired_audit import select_geometry, line_frame, _validate_geometry_stream, SESSIONS
except ImportError:
    from paired_audit import select_geometry, line_frame, _validate_geometry_stream, SESSIONS


CONFIG = {
    "version": "v27_strip_v1",
    "configurations": ["raw_fixed", "geometry_compensated"],
    "s_min_px": -512, "s_max_px": 512, "sample_spacing_original_px": 1.0,
    "half_width_fraction": 0.40, "max_half_width_px": 32,
    "narrow_band_centres_width_fraction": [-0.25, 0.0, 0.25],
    "narrow_band_half_width_px": 1.5,
    "narrow_band_representation": "valid_mean_1d_min_half_rows",
    "intensity_min_exclusive": 15, "intensity_max_exclusive": 245,
    "source_sampling_validity": "all_four_native_bilinear_neighbours_unsaturated_and_nonblack",
    "max_shift_px": 128, "pair_steps": [1, 2],
    "max_adjacent_gap_s": 0.25, "max_geometry_age_s": 0.25,
    "min_overlap_fraction": 0.35, "min_overlap_samples": 96,
    "min_std_gray": 4.0, "min_candidate_zncc": 0.30,
    "candidate_score_drop": 0.12, "max_reported_peaks": 16,
    "min_period_px": 4, "max_period_px": 128,
    "min_period_zncc": 0.60,
    "phase_peak_fraction": 0.35,
    "band_agreement_tolerance_px": 2.0,
    "metric_calibration": None,
}
CONFIG_SHA256 = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()
SOURCE_FILES = (
    "vision_geometry_lab/experiments/v27_material_observability/strip_motion.py",
    "vision_geometry_lab/experiments/v27_material_observability/test_strip_motion.py",
    "vision_geometry_lab/experiments/v27_material_observability/paired_audit.py",
)
TIMESTAMP_KIND = "host_monotonic_sdk_receive_not_exposure"
AVAILABILITY_CLOCK = "causal_input_availability_simulation_excludes_compute"
BAND_NAMES = ["full_2d", "low", "center", "high"]


def _native(value):
    """Strict-JSON conversion: NaNs belong in NPZ cost arrays, not JSON."""
    if isinstance(value, dict):
        return {str(k): _native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_native(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _nfft(length):
    return 1 << (2 * int(length) - 2).bit_length()


def _corr(a, b, nfft, indices):
    """Sum rowwise linear correlation sum_s a[s]*b[s+lag]."""
    cross = np.fft.rfft(b, n=nfft, axis=-1) * np.conj(np.fft.rfft(a, n=nfft, axis=-1))
    return np.fft.irfft(cross.sum(axis=0), n=nfft)[indices]


def masked_zncc(a, b, mask_a, mask_b, lags=None):
    """Exact per-lag overlap-normalized ZNCC; no circular wrap or filled-mask mean.

    FFT evaluates six masked sums. Means/variance are re-estimated over the
    SAME valid overlap at each lag, so static black mask borders do not become
    artificial image edges. Returned overlap is relative to the smaller valid
    support, NOT total strip area. It still cannot certify material identity.
    """
    a, b = np.atleast_2d(a).astype(np.float64), np.atleast_2d(b).astype(np.float64)
    ma, mb = np.atleast_2d(mask_a).astype(bool), np.atleast_2d(mask_b).astype(bool)
    if a.shape != b.shape or a.shape != ma.shape or a.shape != mb.shape:
        raise ValueError("same_shape_strips_and_masks_required")
    ma &= np.isfinite(a); mb &= np.isfinite(b)
    a = np.where(ma, a, 0.0); b = np.where(mb, b, 0.0)
    ma, mb = ma.astype(np.float64), mb.astype(np.float64)
    def axial_contrast(image, mask):
        counts = mask.sum(axis=1)
        means = image.sum(axis=1) / np.maximum(counts, 1)
        row_var = np.maximum(0, (image * image).sum(axis=1) / np.maximum(counts, 1) - means * means)
        return float(np.sqrt(np.sum(row_var * counts) / max(1.0, counts.sum())))
    axial_std_a, axial_std_b = axial_contrast(a, ma), axial_contrast(b, mb)
    if lags is None:
        lags = np.arange(-CONFIG["max_shift_px"], CONFIG["max_shift_px"] + 1)
    lags = np.asarray(lags, dtype=np.int64)
    nfft = _nfft(a.shape[-1]); indices = lags % nfft
    count = np.maximum(0, np.rint(_corr(ma, mb, nfft, indices)))
    sa = _corr(a, mb, nfft, indices); sb = _corr(ma, b, nfft, indices)
    saa = _corr(a * a, mb, nfft, indices); sbb = _corr(ma, b * b, nfft, indices)
    sab = _corr(a, b, nfft, indices)
    safe = np.maximum(count, 1.0)
    va = np.maximum(0, saa - sa * sa / safe)
    vb = np.maximum(0, sbb - sb * sb / safe)
    fraction = count / max(1.0, min(ma.sum(), mb.sum()))
    std_a, std_b = np.sqrt(va / safe), np.sqrt(vb / safe)
    valid = ((count >= CONFIG["min_overlap_samples"]) &
             (fraction >= CONFIG["min_overlap_fraction"]) &
             (std_a >= CONFIG["min_std_gray"]) & (std_b >= CONFIG["min_std_gray"]) &
             (axial_std_a >= CONFIG["min_std_gray"]) & (axial_std_b >= CONFIG["min_std_gray"]) &
             (np.abs(lags) < a.shape[-1]))
    score = np.full(len(lags), np.nan, dtype=np.float64)
    score[valid] = np.clip((sab[valid] - sa[valid] * sb[valid] / safe[valid]) /
                           np.sqrt(va[valid] * vb[valid]), -1, 1)
    return {"lags": lags, "scores": score, "overlap_fraction": fraction,
            "overlap_samples": count, "std_a": std_a, "std_b": std_b, "valid": valid,
            "axial_std_a": axial_std_a, "axial_std_b": axial_std_b}


def local_peaks(lags, values, minimum=-np.inf, max_peaks=None):
    """Return ranked local maxima, including flagged search-boundary peaks."""
    lags, v = np.asarray(lags), np.asarray(values)
    peaks = []
    for i, score in enumerate(v):
        if not np.isfinite(score) or score < minimum:
            continue
        left = v[i - 1] if i > 0 and np.isfinite(v[i - 1]) else -np.inf
        right = v[i + 1] if i + 1 < len(v) and np.isfinite(v[i + 1]) else -np.inf
        if score >= left and score >= right and (score > left or score > right):
            peaks.append({"displacement_px": float(lags[i]), "score": float(score),
                          "search_boundary": i in (0, len(v) - 1), "cost_index": i})
    peaks.sort(key=lambda p: (-p["score"], p["displacement_px"]))
    return peaks[:max_peaks] if max_peaks is not None else peaks


def fft_phase_multi(a, b, ma, mb, lags):
    """Hann + >=2L-1 zero padding + multi-peak normalized cross spectrum.

    Masked zeros and a fixed window can manufacture a stationary phase feature.
    Export the mask-only phase curve and flag its zero peak; do not use a zero
    peak as proof of stopped material. FFT zero-padding prevents circular
    pixel-wrap matching, but DOES NOT resolve an intrinsically periodic branch.
    """
    a, b = np.atleast_2d(a).astype(float), np.atleast_2d(b).astype(float)
    ma, mb = np.atleast_2d(ma).astype(bool), np.atleast_2d(mb).astype(bool)
    if a.shape != b.shape or a.shape != ma.shape or a.shape != mb.shape:
        raise ValueError("same_shape_strips_and_masks_required")
    window = np.hanning(a.shape[-1])[None, :]
    nfft = _nfft(a.shape[-1]); idx = np.asarray(lags, int) % nfft
    def phase(x, y):
        cross = (np.fft.rfft(y, n=nfft, axis=-1) *
                 np.conj(np.fft.rfft(x, n=nfft, axis=-1))).sum(axis=0)
        mag = np.abs(cross)
        norm = np.divide(cross, mag, out=np.zeros_like(cross),
                         where=mag > max(1e-10, mag.max(initial=0) * 1e-12))
        return np.fft.irfft(norm, n=nfft)[idx]
    def prepare(x, m):
        count = m.sum(axis=-1, keepdims=True)
        mean = np.where(m, x, 0).sum(axis=-1, keepdims=True) / np.maximum(count, 1)
        return np.where(m & np.isfinite(x), x - mean, 0) * window
    aa, bb = prepare(a, ma), prepare(b, mb)
    costs = phase(aa, bb)
    mask_costs = phase(ma.astype(float) * window, mb.astype(float) * window)
    meaningful = bool(aa.size and np.std(aa) >= 1 and np.std(bb) >= 1 and min(ma.sum(), mb.sum()) >= 96)
    peaks = local_peaks(lags, costs, max(0, costs.max(initial=0) * CONFIG["phase_peak_fraction"]),
                        CONFIG["max_reported_peaks"]) if meaningful else []
    mask_peaks = local_peaks(lags, mask_costs, 0, 4)
    missing = 1 - min(float(ma.mean()), float(mb.mean()))
    zero_mask = bool(mask_peaks and abs(mask_peaks[0]["displacement_px"]) <= 1)
    zero_image = bool(peaks and abs(peaks[0]["displacement_px"]) <= 1)
    warnings = ["fixed_hann_window_is_not_material_evidence"]
    if missing > 0.05 and (zero_mask or zero_image):
        warnings.append("mask_edge_zero_bias_possible")
    if not meaningful:
        warnings.append("low_variance_or_insufficient_valid_texture")
    return {"scores": costs, "mask_scores": mask_costs, "peaks": peaks,
            "mask_peaks": mask_peaks, "warnings": warnings, "nfft": nfft,
            "padding": "linear_at_least_2L_minus_1", "window": "fixed_Hann_along_s"}


def texture_period(strip, mask):
    lags = np.arange(CONFIG["min_period_px"], CONFIG["max_period_px"] + 1)
    costs = masked_zncc(strip, strip, mask, mask, lags)
    peaks = local_peaks(lags, costs["scores"], CONFIG["min_period_zncc"])
    # A boundary maximum cannot establish a local period. Keep it as a warning.
    interior = [p for p in peaks if not p["search_boundary"]]
    period = min((p["displacement_px"] for p in interior), default=None)
    relations = []
    if period is not None:
        for peak in interior:
            k = int(round(peak["displacement_px"] / period))
            relations.append({"peak_px": peak["displacement_px"], "multiple_of_shortest": k,
                              "residual_px": peak["displacement_px"] - k * period})
    return ({"P_texture_px": period, "period_candidates_px": [p["displacement_px"] for p in interior],
             "peaks": peaks, "harmonic_relations": relations,
             "meaning": "shortest_observed_strong_recurrence_not_physical_lay_or_rope_spacing",
             "material_identity_confirmed": False}, costs)


def period_multiple_diagnostic(displacement_difference_px, period_candidates_px):
    """Arithmetic proximity only, not proof which LK branch is correct."""
    if displacement_difference_px is None or not np.isfinite(displacement_difference_px):
        return {"available": False, "reason": "no_paired_displacement_difference", "comparisons": []}
    comparisons = []
    for p in period_candidates_px:
        if p is None or not np.isfinite(p) or p <= 0:
            continue
        k = int(round(float(displacement_difference_px) / p))
        residual = float(displacement_difference_px) - k * p
        comparisons.append({"P_texture_px": float(p), "nearest_integer_multiple": k,
                            "residual_px": residual, "residual_fraction_period": residual / p})
    return {"available": bool(comparisons), "comparisons": comparisons,
            "conclusion": "proximity_is_not_identity_or_correct_branch_evidence"}


def compare_strips(a, b, ma, mb, dt_s, period=None):
    corr = masked_zncc(a, b, ma, mb)
    all_peaks = local_peaks(corr["lags"], corr["scores"], CONFIG["min_candidate_zncc"])
    best = all_peaks[0]["score"] if all_peaks else None
    candidates = [p for p in all_peaks if p["score"] >= best - CONFIG["candidate_score_drop"]] if all_peaks else []
    # Candidate list is NOT capped: never silently discard a periodic branch.
    phase = fft_phase_multi(a, b, ma, mb, corr["lags"])
    if period is None:
        period, _ = texture_period(a, ma)
    reasons = ["appearance_not_independent_material_correspondence", "translation_spin_phase_coupling_unresolved"]
    if len(candidates) > 1:
        status = "phase_ambiguous"; reasons.append("multiple_correlation_peaks")
    elif len(candidates) == 1:
        status = "single_appearance_peak_not_material"
    else:
        status = "unavailable"; reasons.append("low_variance_overlap_or_no_accepted_peak")
    if any(p["search_boundary"] for p in candidates):
        reasons.append("search_boundary_peak_motion_may_exceed_range")
    report = {"status": status, "candidate_displacements_px": [p["displacement_px"] for p in candidates],
              "correlation_scores": [p["score"] for p in candidates],
              "valid_overlap_fraction": [corr["overlap_fraction"][p["cost_index"]] for p in candidates],
              "all_local_peaks": all_peaks, "phase_correlation_peaks": phase["peaks"],
              "phase_mask_peaks": phase["mask_peaks"], "phase_warnings": phase["warnings"],
              "best_minus_second_peak_score": all_peaks[0]["score"] - all_peaks[1]["score"] if len(all_peaks) > 1 else None,
              "P_texture_px": period["P_texture_px"], "period_diagnostic": period,
              "ambiguity_reason": reasons, "apparent_velocity_px_s":
              [p["displacement_px"] / dt_s for p in candidates] if dt_s > 0 else [],
              "apparent_velocity_kind": "candidate_set_under_receive_clock_not_material_velocity",
              "selected_displacement_px": None, "metric_velocity_m_s": None,
              "calibration_id": None, "reference_measurement_id": None,
              "material_identity_confirmed": False, "correlation_is_probability": False}
    report["axial_texture_std_gray"] = [corr["axial_std_a"], corr["axial_std_b"]]
    report["input_valid_fraction"] = [float(np.mean(ma)) if np.size(ma) else 0,
                                      float(np.mean(mb)) if np.size(mb) else 0]
    return _native(report), {"zncc": corr["scores"], "overlap": corr["overlap_fraction"],
                            "overlap_samples": corr["overlap_samples"], "phase": phase["scores"],
                            "mask_phase": phase["mask_scores"]}


def initial_frame(meta):
    seed = np.asarray(meta.get("free_seed", []), dtype=float)
    box = np.asarray(meta["box"], dtype=float)
    width = meta.get("optical_width_px")
    if seed.shape != (2, 2) or not np.isfinite(seed).all() or width is None or not np.isfinite(width) or width <= 0:
        raise ValueError("no_trustworthy_initial_free_geometry")
    d = seed[1] - seed[0]; norm = np.linalg.norm(d)
    if norm <= 1:
        raise ValueError("degenerate_initial_free_segment")
    d /= norm; n = np.array([-d[1], d[0]])
    reference = box[:2] + (box[2:] - 1) / 2
    origin = reference + n * (n @ seed[0] - n @ reference)
    return {"origin_xy": origin, "d_xy": d, "n_xy": n,
            "reference_xy": reference, "width_px": float(width), "segment_xy": seed}


def sampling_grid(frame, s, n):
    return (np.asarray(frame["origin_xy"])[None, None, :] +
            np.asarray(s)[None, :, None] * np.asarray(frame["d_xy"])[None, None, :] +
            np.asarray(n)[:, None, None] * np.asarray(frame["n_xy"])[None, None, :]).astype(np.float32)


def extract_strip(gray, box, grid, observed_frame):
    local = grid - np.asarray(box[:2], np.float32)
    x, y = local[..., 0], local[..., 1]
    image = cv2.remap(gray, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    valid = ((x >= 0) & (x <= gray.shape[1] - 1) & (y >= 0) & (y <= gray.shape[0] - 1))
    # A saturated pixel mixed with darkness can yield a plausible mid-gray.
    # Do not let bilinear resampling rehabilitate such source pixels as texture.
    source_valid = ((gray > CONFIG["intensity_min_exclusive"]) & (gray < CONFIG["intensity_max_exclusive"]))
    x0 = np.clip(np.floor(x).astype(int), 0, gray.shape[1] - 1)
    y0 = np.clip(np.floor(y).astype(int), 0, gray.shape[0] - 1)
    x1 = np.minimum(x0 + 1, gray.shape[1] - 1); y1 = np.minimum(y0 + 1, gray.shape[0] - 1)
    valid &= source_valid[y0, x0] & source_valid[y0, x1] & source_valid[y1, x0] & source_valid[y1, x1]
    if observed_frame is None:
        return image, np.zeros_like(valid)
    d = np.asarray(observed_frame["d_xy"]); normal = np.asarray(observed_frame["n_xy"])
    origin = np.asarray(observed_frame["origin_xy"]); segment = np.asarray(observed_frame["segment_xy"])
    axial = (grid - origin) @ d; ends = (segment - origin) @ d
    lateral = (grid - origin) @ normal
    valid &= ((axial >= min(ends)) & (axial <= max(ends)) &
              (np.abs(lateral) <= float(observed_frame["width_px"]) / 2))
    valid &= ((image > CONFIG["intensity_min_exclusive"]) & (image < CONFIG["intensity_max_exclusive"]))
    return image, valid


def _band_selectors(n, width):
    selectors = [np.ones(len(n), bool)]
    for fraction in CONFIG["narrow_band_centres_width_fraction"]:
        selectors.append(np.abs(n - fraction * width) <= CONFIG["narrow_band_half_width_px"])
    return selectors


def _band_arrays(strips, masks, selector, full_2d=False):
    images, valid = strips[:, selector], masks[:, selector]
    if full_2d:
        return images, valid
    count = valid.sum(axis=1, keepdims=True)
    # Average only a fixed ~3-pixel narrow band, NOT the full rope width.
    image = np.where(valid, images, 0).sum(axis=1, keepdims=True) / np.maximum(count, 1)
    mask = count >= max(1, int(math.ceil(images.shape[1] / 2)))
    return image.astype(np.float32), mask


def _pair_guard(data, i, j):
    times = np.asarray(data["times"], float); frames = np.asarray(data["source_frames"], int)
    if not np.isfinite(times[i:j + 1]).all() or np.any(np.diff(times[i:j + 1]) <= 0):
        return "nonmonotonic_or_nonfinite_receive_time"
    if np.any(np.diff(times[i:j + 1]) > CONFIG["max_adjacent_gap_s"]):
        return "source_gap"
    if np.any(np.diff(frames[i:j + 1]) != 1):
        return "nonconsecutive_saved_source_frames"
    if "pixel_hashes" in data:
        hashes = data["pixel_hashes"]
        if any(hashes[k] == hashes[k - 1] for k in range(i + 1, j + 1)):
            return "duplicate_pixels_not_proof_of_stationary_material"
    return None


def _band_agreement(reports):
    sets = [r["candidate_displacements_px"] for r in reports]
    available = [s for s in sets if s]
    if len(available) < 2:
        return {"available": False, "reason": "fewer_than_two_available_bands"}
    # Report every cluster supported by all available narrow bands; no branch selection.
    common = [x for x in available[0] if all(min(abs(x - y) for y in other) <=
               CONFIG["band_agreement_tolerance_px"] for other in available[1:])]
    return {"available": True, "number_of_available_bands": len(available),
            "common_candidate_displacements_px": common,
            "independent_evidence": False, "resolves_integer_branch": False}


def process_cohort(meta, data, geometry_records):
    """No I/O or GT. Return report + NPZ-ready arrays for one ALL-frame cohort.

    data is V26 data.npz: gray_native, times, source_frames, optional pixel_hashes.
    geometry_records follow paired_audit.load_geometry_records's independent V16
    pose schema; select_geometry enforces past-only, no stale-invalid fallback.
    This diagnostic does not alter the V26 point TTL or its result denominators.
    """
    start = time.perf_counter()
    _validate_geometry_stream(geometry_records)
    if "timestamp_kind" in data and str(np.asarray(data["timestamp_kind"]).item()) != TIMESTAMP_KIND:
        raise ValueError("cohort_timestamp_kind_mismatch")
    if geometry_records:
        stream = geometry_records[0]
        if stream["camera"] != meta.get("side"):
            raise ValueError("cohort_geometry_camera_mismatch")
        expected_session = meta.get("session", SESSIONS.get(meta.get("side")))
        if expected_session is not None and stream["session"] != expected_session:
            raise ValueError("cohort_geometry_session_mismatch")
    gray = np.asarray(data["gray_native"]); times = np.asarray(data["times"], float)
    sources = np.asarray(data["source_frames"], int)
    if gray.ndim != 3 or len(gray) != len(times) or len(times) != len(sources):
        raise ValueError("malformed_native_cohort")
    s = np.arange(CONFIG["s_min_px"], CONFIG["s_max_px"] + 1, dtype=np.float32)
    first_record = geometry_records[0] if geometry_records else {}
    base = {"group_id": meta["group_id"], "side": meta.get("side"),
            "session": first_record.get("session"), "camera": first_record.get("camera"),
            "config_sha256": CONFIG_SHA256, "timestamp_kind": TIMESTAMP_KIND,
            "frame_denominator": len(times), "pair_denominator_per_configuration":
            sum(max(0, len(times) - k) for k in CONFIG["pair_steps"]),
            "material_identity_confirmed": False, "metric_velocity_m_s": None,
            "t1": None, "t2": None,
            "interpretation": "current_past_appearance_diagnostic_not_online_latency_benchmark",
            "availability_clock": AVAILABILITY_CLOCK, "t_available_is_measured": False,
            "actual_result_available_at_s": None, "actual_live_t_available": None,
            "point_ttl_unchanged": True, "pairs": [], "frames": []}
    arrays = {"s_original_px": s, "times": times, "source_frames": sources,
              "cost_lags_px": np.arange(-CONFIG["max_shift_px"], CONFIG["max_shift_px"] + 1)}
    try:
        initial = initial_frame(meta)
    except (KeyError, ValueError) as exc:
        base.update(status="unavailable", reason=str(exc), elapsed_s=time.perf_counter() - start)
        # Whole-cohort failure remains in ALL-60 summaries, never dropped.
        return _native(base), arrays
    width = initial["width_px"]
    half = min(CONFIG["max_half_width_px"], max(2, int(math.ceil(CONFIG["half_width_fraction"] * width))))
    n = np.arange(-half, half + 1, dtype=np.float32); arrays["n_original_px"] = n
    selectors = _band_selectors(n, width)
    base["coordinate_definition"] = _native({"fixed_reference_xy": initial["reference_xy"],
        "initial_direction_xy": initial["d_xy"], "s_origin": "fixed_crop_centre_projected_to_independent_line",
        "sample_spacing_original_px": 1, "no_per_frame_rescale": True, "initial_optical_width_px": width,
        "aperture_half_width_px": half, "aperture_clipped": half < CONFIG["half_width_fraction"] * width,
        "positive_direction": "initial_seed_to_visible_entry", "pixel_arc_length_is_not_physical_arc_length": True})
    for config in CONFIG["configurations"]:
        frame_infos = []
        strips = np.zeros((len(gray), len(n), len(s)), dtype=np.uint8)
        masks = np.zeros_like(strips, dtype=bool)
        maps = np.full(strips.shape + (2,), np.nan, dtype=np.float32)
        for j, (image, timestamp, source) in enumerate(zip(gray, times, sources)):
            selection = (select_geometry(geometry_records, float(timestamp), int(source)) if np.isfinite(timestamp)
                         else {"available": False, "reason": "invalid_receive_time", "age_s": None, "record": None})
            obs = line_frame(selection["record"], initial["reference_xy"], initial["d_xy"]) if selection["available"] else None
            grid_frame = initial if config == "raw_fixed" else obs
            if grid_frame is not None:
                maps[j] = sampling_grid(grid_frame, s, n)
                strips[j], masks[j] = extract_strip(image, meta["box"], maps[j], obs)
            record = selection.get("record") or {}
            frame_info = {"configuration": config, "source_frame": int(source),
                "time_s": float(timestamp), "geometry_age_s": selection["age_s"],
                "geometry_source_frame": record.get("source_frame"),
                "geometry_available_at_s": record.get("available_at_s"),
                "geometry_available": selection["available"], "geometry_reason": selection["reason"],
                "valid_fraction": float(masks[j].mean()),
                "strip_transform_id": f"{meta['group_id']}:{config}:{int(source)}",
                "t_available": None, "source_available_lower_bound_s": float(timestamp),
                "availability_clock": AVAILABILITY_CLOCK,
                "t_available_is_measured": False, "actual_result_available_at_s": None,
                "actual_live_t_available": None}
            base["frames"].append(frame_info); frame_infos.append(frame_info)
        arrays[config + "_strips"] = strips
        arrays[config + "_valid_masks"] = masks
        arrays[config + "_original_xy_maps"] = maps
        band_arrays = [_band_arrays(strips, masks, select, b == 0) for b, select in enumerate(selectors)]
        # Cache spatial period for EACH source frame/band using only that frame.
        periods, period_costs = [], []
        for j in range(len(gray)):
            frame_periods, frame_costs = [], []
            for band_image, band_mask in band_arrays:
                p, cost = texture_period(band_image[j], band_mask[j])
                frame_periods.append(p); frame_costs.append(cost["scores"])
            periods.append(frame_periods); period_costs.append(frame_costs)
        arrays[config + "_texture_period_costs"] = np.asarray(period_costs, np.float32)
        curves = {key: [] for key in ("zncc", "overlap", "overlap_samples", "phase", "mask_phase")}
        pair_indices = []; lookup = {}
        for j in range(1, len(gray)):
            for step in CONFIG["pair_steps"]:
                i = j - step
                if i < 0:
                    continue
                dt = float(times[j] - times[i]); guard = _pair_guard(data, i, j)
                bands, cost_rows = [], {key: [] for key in curves}
                for b, (band_image, band_mask) in enumerate(band_arrays):
                    mask_a = band_mask[i] if guard is None else np.zeros_like(band_mask[i])
                    mask_b = band_mask[j] if guard is None else np.zeros_like(band_mask[j])
                    report, cost = compare_strips(band_image[i], band_image[j], mask_a, mask_b,
                                                  dt, periods[i][b])
                    report["band"] = BAND_NAMES[b]
                    report["representation"] = "full_2d_texture_row_preserved" if b == 0 else CONFIG["narrow_band_representation"]
                    if guard:
                        report["ambiguity_reason"].append(guard)
                    bands.append(report)
                    for key in curves:
                        cost_rows[key].append(cost[key])
                pair = {"group_id": meta["group_id"], "side": meta.get("side"),
                    "session": base["session"], "camera": base["camera"],
                    "configuration": config, "frame_pair": [int(sources[i]), int(sources[j])],
                    "local_frame_pair": [i, j], "timestamp_kind": TIMESTAMP_KIND, "dt_s": dt,
                    "strip_transform_id": [f"{meta['group_id']}:{config}:{int(sources[k])}" for k in (i, j)],
                    "t_available": None, "source_available_lower_bound_s": float(times[j]),
                    "max_source_timestamp": float(times[j]),
                    "availability_clock": AVAILABILITY_CLOCK, "t_available_is_measured": False,
                    "actual_result_available_at_s": None, "actual_live_t_available": None,
                    "geometry_age_s": [frame_infos[k]["geometry_age_s"] for k in (i,j)],
                    "geometry_source_frames": [frame_infos[k]["geometry_source_frame"] for k in (i,j)],
                    "guard_failure": guard, "bands": bands,
                    "narrow_band_agreement": _band_agreement(bands[1:]),
                    "metric_velocity_m_s": None, "calibration_id": None, "reference_measurement_id": None,
                    "ttl_rule": "V26_tracking_TTL_unchanged_this_is_pairwise_appearance_not_identity"}
                if step == 2:
                    first = lookup.get((i, j - 1), [])
                    second = lookup.get((j - 1, j), [])
                    target = bands[0]["candidate_displacements_px"]
                    combinations = [(a, b, c, c - a - b) for a in first for b in second for c in target]
                    best = min((abs(row[3]) for row in combinations), default=None)
                    pair["composition_diagnostic"] = {"minimum_branch_residual_px": best,
                        "candidate_triplet_count": len(combinations), "independent_evidence": False,
                        "branch_selected": False, "future_frames_used": False}
                lookup[(i, j)] = bands[0]["candidate_displacements_px"]
                base["pairs"].append(pair); pair_indices.append([i, j])
                for key in curves:
                    curves[key].append(cost_rows[key])
        arrays[config + "_pair_indices"] = np.asarray(pair_indices, np.int32)
        for key, values in curves.items():
            arrays[config + "_" + key + "_costs"] = np.asarray(values, np.float32)
        spaces = []
        for band_image, band_mask in band_arrays[1:]:
            spaces.append(np.where(band_mask[:, 0], band_image[:, 0], np.nan))
        arrays[config + "_narrow_band_spacetime"] = np.asarray(spaces, np.float32)
    base.update(status="diagnostic_completed", elapsed_s=time.perf_counter() - start)
    return _native(base), arrays


def write_artifacts(directory, report, arrays):
    """Create NEW group directory with full maps/costs and visual diagnostics."""
    from PIL import Image, ImageDraw
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "report.json").write_text(json.dumps(_native(report), ensure_ascii=False, indent=2,
                                                     allow_nan=False), encoding="utf-8")
    np.savez_compressed(directory / "strip_maps_full_costs.npz", **arrays)
    for config in CONFIG["configurations"]:
        key = config + "_strips"
        if key not in arrays:
            continue
        strips, masks = arrays[key], arrays[config + "_valid_masks"]
        # All frames shown, original s pixel spacing retained, invalid tinted red.
        height = strips.shape[1] + 18
        canvas = Image.new("RGB", (strips.shape[2], len(strips) * height), "#111111")
        draw = ImageDraw.Draw(canvas)
        for j in range(len(strips)):
            rgb = np.repeat(strips[j, :, :, None], 3, axis=-1)
            rgb[~masks[j], 0] = np.maximum(rgb[~masks[j], 0], 90)
            rgb[~masks[j], 1:] //= 3
            canvas.paste(Image.fromarray(rgb), (0, j * height + 16))
            draw.text((3, j * height), f"source {int(arrays['source_frames'][j])}: red=invalid; pixel scale 1", fill="white")
        canvas.save(directory / (config + "_all_strips.png"))
        space = arrays[config + "_narrow_band_spacetime"]
        canvas = Image.new("RGB", (space.shape[-1], 3 * (len(strips) * 8 + 22)), "#111111")
        draw = ImageDraw.Draw(canvas)
        for k in range(3):
            valid = np.isfinite(space[k]); rgb = np.repeat(np.nan_to_num(space[k], nan=0).astype(np.uint8)[..., None], 3, axis=-1)
            rgb[~valid] = [110, 10, 10]
            rendered = Image.fromarray(rgb).resize((space.shape[-1], len(strips) * 8), Image.Resampling.NEAREST)
            top = k * (len(strips) * 8 + 22)
            draw.text((3, top), BAND_NAMES[k + 1] + ": y=receive sample index, NOT uniform exposure time", fill="white")
            canvas.paste(rendered, (0, top + 20))
        canvas.save(directory / (config + "_spacetime.png"))


def summarize_reports(reports, expected_groups=60):
    """Fixed all-group/pair denominators. This summary intentionally has no accuracy."""
    output = {"expected_group_count": expected_groups, "received_group_count": len(reports),
              "missing_groups": max(0, expected_groups - len(reports)), "config_sha256": CONFIG_SHA256,
              "group_failures": [{"group_id": r["group_id"], "reason": r.get("reason")} for r in reports
                                 if r.get("status") != "diagnostic_completed"],
              "metric_velocity_m_s": None, "material_identity_accuracy": None, "t1_accuracy": None,
              "t2": None, "configurations": {}}
    for config in CONFIG["configurations"]:
        counts = {}; periods = []; candidates = []
        for report in reports:
            for pair in report.get("pairs", []):
                if pair["configuration"] != config:
                    continue
                band = pair["bands"][0]; status = band["status"]
                counts[status] = counts.get(status, 0) + 1
                if band["P_texture_px"] is not None:
                    periods.append(band["P_texture_px"])
                candidates.append(len(band["candidate_displacements_px"]))
        denominator = sum(r["pair_denominator_per_configuration"] for r in reports)
        output["configurations"][config] = {"pair_denominator_received_groups": denominator,
            "status_counts_full_2d": counts, "uncomputed_pairs": denominator - sum(counts.values()),
            "pairs_with_spatial_recurrence": len(periods),
            "median_observed_texture_recurrence_px": float(np.median(periods)) if periods else None,
            "median_candidate_count": float(np.median(candidates)) if candidates else None,
            "not_independent_material_or_t1_evidence": True}
    return output


def render_helical_cylinder(length=1025, rows=25, period_px=24.0, radius_px=18.0,
                             strand_count=3, translation_px=0.0, rotation_rad=0.0,
                             defect=False, glare=False, occlusion=False):
    """Orthographic physical-surface texture renderer, not arbitrary brightness roll.

    Camera samples cylinder azimuth phi=asin(n/R). Material coordinates are
    (s-u, phi-theta), intrinsic helical phase 2pi(s-u)/P+k(phi-theta).
    Thus ideal periodic surface translation u and spin theta are nonidentifiable
    when 2pi*u/P+k*theta is equal (mod 2pi). A nonperiodic surface mark MAY break
    this symmetry if visible; highlights and occluders belong to the camera.
    """
    s = np.arange(length, dtype=float)[None, :]
    n = (np.arange(rows, dtype=float) - (rows - 1) / 2)[:, None]
    phi = np.arcsin(np.clip(n / radius_px, -1, 1))
    material_s, material_phi = s - translation_px, phi - rotation_rad
    phase = 2 * np.pi * material_s / period_px + strand_count * material_phi
    value = 120 + 48 * np.cos(phase) + 12 * np.cos(2 * phase + 0.4)
    value = np.broadcast_to(value, (rows, length)).copy()
    if defect:
        # A nonperiodic material mark, moving and rotating WITH the surface.
        value -= 85 * np.exp(-0.5 * ((material_s - length * 0.46) / 17) ** 2 -
                             0.5 * ((material_phi - 0.12) / 0.45) ** 2)
    mask = np.ones_like(value, bool)
    if glare:
        highlight = np.exp(-0.5 * ((s - length * .72) / 32) ** 2 - 0.5 * (n / 7) ** 2)
        value += 240 * highlight
    if occlusion:
        block = (s > length * .25) & (s < length * .31)
        mask &= ~np.broadcast_to(block, value.shape)
        value[~mask] = 0
    value = np.clip(value, 0, 255).astype(np.float32)
    mask &= (value > CONFIG["intensity_min_exclusive"]) & (value < CONFIG["intensity_max_exclusive"])
    truth = {"renderer": "orthographic_material_helical_cylinder",
             "translation_px": translation_px, "rotation_rad": rotation_rad,
             "period_px": period_px, "radius_px": radius_px, "strand_count": strand_count,
             "equivalent_axial_phase_shift_px": translation_px + period_px * strand_count * rotation_rad / (2 * np.pi),
             "defect_material_attached": defect, "glare_camera_fixed": glare,
             "occlusion_camera_fixed": occlusion, "synthetic_not_field_validation": True}
    return value, mask, truth


def run_synthetic_suite(directory):
    """Save four mechanism-declared cases and a genuine moving-texture false-zero case.

    No field data/labels are used. Exports physical renderer parameters, both
    images/masks and all curves. Success is software/observability evidence only.
    """
    from PIL import Image, ImageDraw
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    cases = [
        ("pure_translation", {"translation_px": 77.0}),
        ("pure_spin_no_feed", {"rotation_rad": 0.8}),
        ("translation_and_spin", {"translation_px": 11.0, "rotation_rad": 0.9}),
        ("material_defect_glare_occlusion", {"translation_px": 29.0, "defect": True, "glare": True, "occlusion": True}),
    ]
    output = {"config_sha256": CONFIG_SHA256, "synthetic_not_field_validation": True, "cases": []}
    arrays = {"lags": np.arange(-128, 129)}
    panels = []
    for name, kwargs in cases:
        initial_kwargs = {k: v for k, v in kwargs.items() if k not in ("translation_px", "rotation_rad")}
        a, ma, truth_a = render_helical_cylinder(**initial_kwargs)
        b, mb, truth_b = render_helical_cylinder(**kwargs)
        report, cost = compare_strips(a, b, ma, mb, .1)
        output["cases"].append({"name": name, "truth_a": truth_a, "truth_b": truth_b, "diagnostic": report})
        for key, value in {"a": a, "b": b, "mask_a": ma, "mask_b": mb, **cost}.items():
            arrays[name + "_" + key] = value
        panels.append((name, a, b))
    # Same rendered images from two physically distinct motions.
    theta = .8; displacement = 24 * 3 * theta / (2 * np.pi)
    translated, _, _ = render_helical_cylinder(translation_px=displacement)
    spun, _, _ = render_helical_cylinder(rotation_rad=theta)
    output["nonidentifiability_demonstration"] = {
        "translation_only_px": displacement, "rotation_only_rad": theta,
        "rotation_case_axial_feed_px": 0.0,
        "max_absolute_image_difference_gray": float(np.max(np.abs(translated - spun))),
        "conclusion": "same_image_phase_does_not_identify_axial_feed_vs_axial_spin",
    }
    x = np.arange(401)[None, :]
    a = np.repeat(120 + 40 * np.cos(2*np.pi*x/24), 5, axis=0)
    b = np.repeat(120 + 40 * np.cos(2*np.pi*(x-5)/24), 5, axis=0)
    mask = np.repeat(((x > 80) & (x < 170)) | ((x > 250) & (x < 330)), 5, axis=0)
    report, cost = compare_strips(a, b, mask, mask, .1)
    output["static_mask_false_zero"] = {"actual_synthetic_texture_shift_px": 5.0,
        "diagnostic": report, "explanation": "fixed_mask_edges_induce_zero_phase_peak_despite_moving_periodic_texture"}
    for key, value in {"a": a, "b": b, "mask": mask, **cost}.items():
        arrays["false_zero_" + key] = value
    np.savez_compressed(directory / "synthetic_full_costs.npz", **arrays)
    (directory / "report.json").write_text(json.dumps(_native(output), ensure_ascii=False, indent=2,
                                                     allow_nan=False), encoding="utf-8")
    canvas = Image.new("RGB", (1025, len(panels) * 150), "#111111")
    draw = ImageDraw.Draw(canvas)
    for k, (name, a, b) in enumerate(panels):
        draw.text((5, k*150+4), name + " | synthetic surface model, not field validation", fill="white")
        for offset, image in [(25, a), (85, b)]:
            rgb = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).convert("RGB")
            canvas.paste(rgb.resize((1025, 50), Image.Resampling.NEAREST), (0, k*150+offset))
    canvas.save(directory / "mechanism_pairs.png")
    return _native(output)
