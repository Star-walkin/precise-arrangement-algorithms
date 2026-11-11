"""Real B750 + frozen H6 with a current-frame A7/D4 scene adapter.

Coordinate contract: original, unmirrored 4096 x 3000 image xy throughout the
public API.  The frozen neural/H6 crop is internal: original=2*xy+[0,1050].
The supplied mask is a static SAM2 scene search domain, not a current target.
Historical target paths, response caches, selected tracks and labels are never
loaded.  Current D4 observations replace the unavailable historical F3 fusion
cache explicitly; this is an LC1 upstream adaptation, not frozen P1 replay.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "p1_runtime" / "source_archive"
P1 = ARCHIVE / "experiments" / "entry_conditioned_rope_path"
A7 = (ARCHIVE / "数据资产" / "05_派生与实验结果" /
      "camera_condition_free_straight_20261030_v7")
PINNED_COMMIT = "8d30826fbef2df423ccfc56047995e6194f12b6c"
B750_SHA256 = "ed4bfa12c5b3081a19d056265b053d516bdba5ec05dce658e1e32339b47fb882"
_IMPORT_LOCK = threading.RLock()
SUPPORT_POLICY = dict(
    minimum_free_joint_fraction=.45, minimum_free_observed_fraction=.65,
    minimum_transition_joint_fraction=.50, minimum_surface_joint_fraction=.50,
    minimum_free_observed_length_px=120., minimum_surface_joint_length_px=60.,
    maximum_unsupported_gap_px=80., maximum_unsupported_gap_widths=4.,
    minimum_direction_distance_difference_px=20.,
    note="LC1 current-frame evidence gate; frozen H6 ranking/geometry unchanged",
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _isolated_modules(directory, aliases, final_name, final_path):
    """Resolve frozen bare imports without leaking aliases into LC1/S1."""
    with _IMPORT_LOCK:
        previous = {name: sys.modules.get(name) for name in aliases}
        old_path = list(sys.path)
        try:
            sys.path.insert(0, str(directory))
            for name in aliases:
                sys.modules.pop(name, None)
            return _load(final_name, final_path)
        finally:
            sys.path[:] = old_path
            for name, value in previous.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value


@lru_cache(maxsize=1)
def _a7():
    return _isolated_modules(
        A7, ["free_rope", "free_seed_recovery", "strand_signature"],
        "_lc1_p1_A7", A7 / "straight_detector.py")


@lru_cache(maxsize=1)
def _h6():
    folder = P1 / "postprocess_b_20261031_v6"
    return _isolated_modules(
        folder, ["common", "surface_tracks", "topology", "heat_free", "connect_paths"],
        "_lc1_p1_H6", folder / "run_h6.py")


@lru_cache(maxsize=1)
def _f3():
    return _load("_lc1_p1_current_f3",
                 ROOT / "vendor_archive" / "f3_core_adapter.py")


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _sample(array, xy):
    p = np.asarray(xy, np.float32)
    shape = p.shape[:-1]
    q = p.reshape(-1, 2)
    return cv2.remap(
        np.asarray(array, np.float32), q[:, 0][None], q[:, 1][None],
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).reshape(shape)


def _original(points):
    p = np.asarray(points, float).reshape(-1, 2)
    return (2. * p + [0., 1050.]).tolist()


def _maxhalf(mask):
    a = np.asarray(mask, np.uint8)
    h, w = a.shape
    return a.reshape(h // 2, 2, w // 2, 2).max(axis=(1, 3))


def _longest_gap(points, supported):
    p = np.asarray(points, float).reshape(-1, 2)
    okay = np.asarray(supported, bool)
    if len(p) < 2:
        return 0.
    ds = 2 * np.linalg.norm(np.diff(p, axis=0), axis=1)
    bad = ~(okay[:-1] & okay[1:])
    best = value = 0.
    for length, flag in zip(ds, bad):
        value = value + float(length) if flag else 0.
        best = max(value, best)
    return best


def _supported_length(points, supported):
    p = np.asarray(points, float).reshape(-1, 2)
    okay = np.asarray(supported, bool)
    if len(p) < 2:
        return 0.
    ds = 2 * np.linalg.norm(np.diff(p, axis=0), axis=1)
    return float(ds[okay[:-1] & okay[1:]].sum())


def _raw_seed_fraction(half_gray, xy, width):
    p = np.linspace(np.asarray(xy)[0], np.asarray(xy)[-1], 24) * .5
    direction = p[-1] - p[0]
    direction /= max(float(np.linalg.norm(direction)), 1.e-12)
    normal = np.array([-direction[1], direction[0]])
    offsets = np.array([-1., -.6, 0., .6, 1.]) * float(width) * .5
    v = _sample(half_gray, p[:, None, :] + normal * offsets[None, :, None])
    bright = np.minimum(v[:, 2] - v[:, :2].mean(1),
                        v[:, 2] - v[:, 3:].mean(1))
    dark = np.minimum(v[:, :2].mean(1) - v[:, 2],
                      v[:, 3:].mean(1) - v[:, 2])
    return float((np.maximum(bright, dark) >= .005).mean())


class P1H6Adapter:
    """Cold-start current-frame A7 -> real B750 -> genuine H6.

    The class does not mirror either camera.  It owns no temporal target cache.
    last_heatmaps is diagnostic in-memory data only and is excluded from step().
    """
    def __init__(self, surface_mask_original, pitch_px, model_path, threads=2,
                 camera=None, surface_mode="full", entry_band_halfwidth_pitch=8.,
                 entry_band_halo_pitch=2.):
        import torch
        self.torch = torch
        torch.set_num_threads(int(threads))
        cv2.setNumThreads(1)
        mask = np.asarray(surface_mask_original)
        if mask.shape != (3000, 4096):
            raise ValueError("P1 model contract requires original 4096x3000 mask")
        self.surface = (mask > 0).astype(np.uint8) * 255
        self.pitch = float(pitch_px)
        if not 8 <= self.pitch <= 70:
            raise ValueError("Expected observed original image pitch 8..70 px")
        self.camera = camera
        if surface_mode not in ("full", "entry_band"):
            raise ValueError("surface_mode must be full or explicitly selected entry_band")
        self.surface_mode = surface_mode
        self.entry_band_halfwidth_pitch = float(entry_band_halfwidth_pitch)
        self.entry_band_halo_pitch = float(entry_band_halo_pitch)
        if self.entry_band_halfwidth_pitch < 8. or self.entry_band_halo_pitch < 2.:
            raise ValueError("Conservative entry_band requires at least 8pitch halfwidth plus 2pitch halo")
        self.model_path = Path(model_path)
        actual = _sha(self.model_path)
        if actual != B750_SHA256:
            raise ValueError("Weight SHA does not match frozen P1 B750")
        network = _load("_lc1_p1_entry_unet", P1 / "model.py")
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        self.net = network.EntryUNet(int(checkpoint["base_channels"]))
        self.net.load_state_dict(checkpoint["model_state"])
        self.net.eval()
        self.checkpoint_step = checkpoint.get("step")
        self.model_sha256 = actual
        self.a7 = _a7()
        self.h6 = _h6()
        self.f3 = _f3()
        self.last_heatmaps = []
        self.last_scene_arrays = None

    def _surface_observation_mask(self, seeds):
        """Optional finite-width current-A7 search domain, without any resize.

        Only D4's observation mask is restricted. Neural input, H6's task domain
        and optical terminal domain remain the full static SAM2 search region.
        The two-pitch halo is retained as genuine detector context, not silently
        trimmed into a claimed complete surface. A bent branch outside this
        domain is missing, never extrapolated back into the observation.
        """
        stats = dict(mode=self.surface_mode, original_shape=[3000, 4096],
                     resized=False, full_static_mask_pixels=int((self.surface > 0).sum()),
                     downstream_H6_terminal_domain_restricted=False)
        if self.surface_mode == "full":
            stats.update(observation_mask_pixels=stats["full_static_mask_pixels"],
                         fraction_of_static_surface=1., current_entry_axes_used=0)
            return self.surface, stats
        yy, xx = np.ogrid[:3000, :4096]
        keep = np.zeros(self.surface.shape, bool)
        halfwidth = self.entry_band_halfwidth_pitch * self.pitch
        halo = self.entry_band_halo_pitch * self.pitch
        for seed in seeds:
            q = np.asarray(seed["xy"], float)
            direction = q[-1] - q[0]
            direction /= max(float(np.linalg.norm(direction)), 1.e-12)
            distance = np.abs((xx - q[0, 0]) * (-direction[1]) +
                              (yy - q[0, 1]) * direction[0])
            keep |= distance <= halfwidth + halo
        mask = ((self.surface > 0) & keep).astype(np.uint8) * 255
        pixels = int((mask > 0).sum())
        stats.update(observation_mask_pixels=pixels,
                     fraction_of_static_surface=pixels / max(stats["full_static_mask_pixels"], 1),
                     current_entry_axes_used=len(seeds),
                     halfwidth_pitch=self.entry_band_halfwidth_pitch,
                     halo_pitch=self.entry_band_halo_pitch,
                     halfwidth_original_px=halfwidth, halo_original_px=halo,
                     entry_source="current_frame_A7_automatic_axes_only",
                     candidate_scope="finite_current_entry_band_plus_halo_not_whole_surface")
        return mask, stats

    def _entry_seeds(self, bgr, a7):
        full_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        half_gray = cv2.resize(full_gray, (2048, 1500),
                               interpolation=cv2.INTER_AREA).astype(np.float32) / 255.
        distance = cv2.distanceTransform((self.surface == 0).astype(np.uint8),
                                         cv2.DIST_L2, 3)
        seeds, ledger = [], []
        for candidate in a7.get("candidates", []):
            cid = candidate.get("candidate_id")
            reason = None
            if not candidate.get("plausible"):
                reason = "A7_candidate_not_plausible"
            elif float(candidate.get("anchor_span_px", 0)) < 100:
                reason = "outside_anchor_shorter_than_100px"
            segment = np.asarray(candidate.get("anchor_segment_xy", []), float).reshape(-1, 2)
            if reason is None and len(segment) != 2:
                reason = "no_finite_A7_outside_anchor"
            if reason is not None:
                ledger.append(dict(candidate_id=cid, used=False, reason=reason))
                continue
            distances = _sample(distance, segment)
            if float(max(distances)) < 20:
                ledger.append(dict(candidate_id=cid, used=False,
                                   reason="anchor_not_20px_outside_static_surface"))
                continue
            if distances[0] < distances[-1]:
                segment = segment[::-1].copy()
                distances = distances[::-1].copy()
            direction = segment[-1] - segment[0]
            length = float(np.linalg.norm(direction))
            if length < 60:
                continue
            direction /= length
            start = segment[0] + direction * min(12., length * .05)
            end = start + direction * min(120., length - 20.)
            xy = np.vstack([start, end])
            if not ((xy[:, 0] >= 0) & (xy[:, 0] < 4096) &
                    (xy[:, 1] >= 1050) & (xy[:, 1] < 3000)).all():
                ledger.append(dict(candidate_id=cid, used=False,
                                   reason="A7_seed_outside_frozen_model_crop"))
                continue
            raw = _raw_seed_fraction(half_gray, xy, candidate["width_px"])
            if raw < .30:
                ledger.append(dict(candidate_id=cid, used=False,
                                   reason="seed_current_raw_bilateral_support_below_0.30",
                                   raw_support_fraction=raw))
                continue
            seed = dict(xy=xy.tolist(), source="current_frame_P1_A7_automatic_far_to_spool",
                        entry_id="current_" + str(cid), a7_candidate_id=cid,
                        width_px=float(candidate["width_px"]),
                        a7_accepted=bool(candidate.get("accepted")),
                        automatic=True, current_raw_support_fraction=raw,
                        direction_ambiguous=bool(abs(distances[0] - distances[-1]) < 20.),
                        outside_distance_endpoints_px=distances.tolist(),
                        direction_basis="finite_current_A7_outside_anchor_toward_static_SAM2_domain",
                        identity_confirmed=False)
            seeds.append(seed)
            ledger.append(dict(candidate_id=cid, used=True,
                               raw_support_fraction=raw, direction_ambiguous=seed["direction_ambiguous"]))
            if len(seeds) == 2:
                break
        return seeds, ledger

    def _response(self, bgr, seed):
        torch = self.torch
        full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(full[1050:].astype(np.float32) / 255., (2048, 975),
                          interpolation=cv2.INTER_AREA)
        condition = np.zeros((975, 2048), np.uint8)
        q = np.rint((np.asarray(seed["xy"]) - [0, 1050]) * .5).astype(np.int32)
        cv2.polylines(condition, [q], False, 1, 1, cv2.LINE_8)
        if not condition.any():
            raise ValueError("Empty current-frame entry condition")
        distance = cv2.distanceTransform(1 - condition, cv2.DIST_L2,
                                        cv2.DIST_MASK_PRECISE)
        seed_map = np.exp(-.5 * (distance / 2.) ** 2).astype(np.float32)
        g = torch.from_numpy(gray.copy())
        s = torch.from_numpy(seed_map)
        x = torch.nn.functional.pad(
            torch.stack([(g - g.mean()) / g.std(unbiased=False).clamp_min(.08), s]),
            (0, 0, 0, 17))[None]
        with torch.inference_mode():
            response = self.net(x)[0, 0, :975, :2048].numpy().copy()
        if response.shape != (975, 2048) or not np.isfinite(response).all():
            raise ValueError("Invalid real B750 response")
        return response

    def _priors(self, bgr, a7, core):
        fit = core["fit"]
        merged = fit["smooth_mask"]
        inferred = fit.get("inferred_mask", np.zeros_like(merged))
        exclusion = core.get("exclusion_mask", np.zeros_like(self.surface))
        surface_half = cv2.resize(self.surface, (2048, 1500), interpolation=cv2.INTER_NEAREST)
        return dict(
            original_half=cv2.resize(bgr, (2048, 1500), interpolation=cv2.INTER_AREA),
            surface_half=surface_half, sam2_surface_half=surface_half.copy(),
            original_shape=[3000, 4096], half_shape=[1500, 2048],
            pitch_px=self.pitch, pitch=self.pitch, pitch_half_px=self.pitch * .5,
            camera_role=self.camera, a7_candidates=a7.get("candidates", []), a6_entries=[],
            f3_merged_half=_maxhalf(merged), f3_inferred_half=_maxhalf(inferred),
            f3_exclusion_half=_maxhalf(exclusion),
            f3_vector_records={"C3": {}, "D4": {"fit": fit}, "F2": {}, "F3": {"groups": []}},
            f3_vector_source_paths={"C3": None, "D4": "current_frame_F3_observation_core_D4_fit",
                                    "F2": None, "F3": None},
            input_dependency_sha256={},
            surface_policy="current_frame_D4_accepted_fits_and_current_raster_not_historical_F3_cache")

    def _support(self, entry, candidate, all_ranked):
        reasons, checks = [], {}
        p = np.asarray(candidate["points_xy"], float).reshape(-1, 2)
        joint = np.asarray(candidate["supported"], bool)
        stages = np.asarray(candidate["stage"])
        status = np.asarray(candidate["point_status"])
        observed = joint | np.isin(status, ["raw_a7_supported", "image_supported",
                                           "joint_heat_image_supported"])
        # H6's image_supported on bridge/tail is joint raw+heat in this frozen connector.
        # entry_condition is supplied condition, never an observation.
        observed &= status != "entry_condition"
        joint &= status != "entry_condition"
        evaluation_scope = status != "entry_condition"
        # The supplied condition is explicitly not an observation. Exclude its
        # leading extent from evidence denominators/gap tests as well; otherwise
        # a 120px input seed would force rejection against the 80px gap budget
        # even on an otherwise fully observed trajectory. Its rendered segments
        # remain unsupported, and it never contributes observed length.
        checks["evaluation_scope"] = "current_candidate_after_input_entry_condition"
        checks["input_condition_point_count_excluded"] = int((~evaluation_scope).sum())
        for stage in ("free", "transition", "surface"):
            take = (stages == stage) & evaluation_scope
            checks[stage] = dict(
                point_count=int(take.sum()),
                joint_fraction=float(joint[take].mean()) if take.any() else 0.,
                observed_fraction=float(observed[take].mean()) if take.any() else 0.,
                observed_length_original_px=_supported_length(p, observed & take),
                joint_length_original_px=_supported_length(p, joint & take),
                maximum_unsupported_run_original_px=_longest_gap(p[take], observed[take]) if take.any() else 0.)
        attached = (candidate.get("kind") == "straight_free_to_F3_center_track" and
                    candidate.get("continuation_eligible") is True and
                    candidate.get("topology", {}).get("allowed") is True)
        checks["genuine_H6_attached_candidate"] = attached
        if not attached:
            reasons.append("H6_did_not_select_an_image_eligible_legal_curve_connection")
        for stage, threshold in (("free", SUPPORT_POLICY["minimum_free_joint_fraction"]),
                                 ("transition", SUPPORT_POLICY["minimum_transition_joint_fraction"]),
                                 ("surface", SUPPORT_POLICY["minimum_surface_joint_fraction"])):
            if checks[stage]["point_count"] < 3 or checks[stage]["joint_fraction"] < threshold:
                reasons.append(stage + "_current_joint_support_insufficient")
        if checks["free"]["observed_fraction"] < SUPPORT_POLICY["minimum_free_observed_fraction"]:
            reasons.append("free_current_observed_fraction_insufficient")
        if checks["free"]["observed_length_original_px"] < SUPPORT_POLICY["minimum_free_observed_length_px"]:
            reasons.append("free_current_observed_length_insufficient")
        if checks["surface"]["joint_length_original_px"] < SUPPORT_POLICY["minimum_surface_joint_length_px"]:
            reasons.append("curve_current_joint_length_insufficient")
        width = 2 * float(np.median(candidate.get("width_half_px", [self.pitch * .25])))
        maximum_gap = max(SUPPORT_POLICY["maximum_unsupported_gap_px"],
                          SUPPORT_POLICY["maximum_unsupported_gap_widths"] * width)
        evaluated = np.flatnonzero(evaluation_scope)
        gap = _longest_gap(p[evaluated[0]:evaluated[-1]+1],
                           observed[evaluated[0]:evaluated[-1]+1]) if len(evaluated) else 0.
        checks["maximum_unsupported_run_original_px"] = gap
        checks["maximum_allowed_unsupported_run_original_px"] = maximum_gap
        if gap > maximum_gap:
            reasons.append("unsupported_path_gap_exceeds_LC1_gate")
        branch_margin = entry["connection"].get("score_margin")
        cross_margin = None
        if len(all_ranked) > 1 and all_ranked[0][1] != all_ranked[1][1]:
            cross_margin = float(all_ranked[0][0] - all_ranked[1][0])
        ambiguous = bool(entry["connection"].get("ambiguous", True) or
                         entry["seed"].get("direction_ambiguous") or
                         cross_margin is not None and cross_margin < .045)
        checks["branch_score_margin"] = branch_margin
        checks["cross_entry_score_margin"] = cross_margin
        checks["ambiguous"] = ambiguous
        checks["direction_selection_allowed"] = entry.get("task_direction", {}).get("selection_allowed", False)
        if ambiguous:
            reasons.append("H6_branch_or_A7_direction_ambiguous")
        if not checks["direction_selection_allowed"]:
            reasons.append("current_entry_direction_not_task_eligible")
        if entry["seed"].get("a7_accepted") is not True:
            reasons.append("A7_axis_not_uniquely_accepted")
        segment_support = (observed[:-1] & observed[1:]).tolist()
        return not reasons, checks, reasons, segment_support, observed.tolist()

    def step(self, bgr_original):
        start = time.perf_counter()
        bgr = np.asarray(bgr_original)
        if bgr.shape != (3000, 4096, 3) or bgr.dtype != np.uint8:
            raise ValueError("Expected original unmirrored uint8 3000x4096x3 BGR")
        self.last_heatmaps = []
        timings = {}
        a7 = self.a7.detect(dict(original=bgr, surface=self.surface, pitch=self.pitch))
        seeds, seed_ledger = self._entry_seeds(bgr, a7)
        timings["current_A7_seconds"] = time.perf_counter() - start
        base = dict(
            source="real_P1_B750_H6_with_current_A7_D4_scene_adapter",
            coordinate_system="original_unmirrored_image_xy_pixels",
            camera=self.camera, model_sha256=self.model_sha256, model_step=self.checkpoint_step,
            frozen_source_commit=PINNED_COMMIT,
            frozen_P1_scene_chain_reproduced=False, current_scene_adapter=True,
            seed_xy=None, seed_source=None, line_points_xy=[], path_xy=[],
            whole_path_candidate_supported=False, same_rope_identity_confirmed=False,
            model_forward_count=0, status="UNKNOWN", reasons=[],
            diagnostics=dict(
                A7_status=a7["status"], current_A7_seed_ledger=seed_ledger,
                scene_prior="static_SAM2_mask_and_observed_pitch_only",
                historical_target_cache_read=False, labels_read=False,
                source_manifest=str(ROOT / "p1_runtime" / "source_manifest.json"),
                support_policy=SUPPORT_POLICY, path_segment_support=[],
                surface_observation_mode=self.surface_mode,
                timings=timings))
        if not seeds:
            base["reasons"] = ["no_current_A7_raw_supported_entry_in_B750_crop"]
            timings["total_seconds"] = time.perf_counter() - start
            return _plain(base)
        tick = time.perf_counter()
        observation_mask, observation_mask_stats = self._surface_observation_mask(seeds)
        core = self.f3.detect_surface_core(
            bgr, observation_mask, self.pitch, exclusion_mask=np.zeros_like(self.surface), c3_record=None)
        timings["current_D4_seconds"] = time.perf_counter() - tick
        priors = self._priors(bgr, a7, core)
        self.last_scene_arrays = {"merged": core["fit"]["smooth_mask"],
                                  "inferred": core["fit"].get("inferred_mask")}
        gray = cv2.cvtColor(priors["original_half"][525:], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
        entries, tracksets = [], []
        for seed in seeds:
            tick = time.perf_counter()
            heat = self._response(bgr, seed)
            neural_seconds = time.perf_counter() - tick
            self.last_heatmaps.append(heat)
            tick = time.perf_counter()
            tracks = self.h6.build_tracks(priors, heat, gray)
            full_free = self.h6.describe_free(priors, seed, heat)
            free, projection = self.h6.prepare_task_free(full_free, tracks["tracks"])
            linked = self.h6.connect(gray, heat, free, tracks["tracks"], [],
                                     priors["surface_half"][525:])
            entries.append(dict(seed=seed, free=free, full_free_observation=full_free,
                                free_projection=projection, connection=linked,
                                neural_seconds=neural_seconds,
                                H6_seconds=time.perf_counter() - tick))
            tracksets.append(tracks)
        self.h6.task_direction_feasibility(entries, priors["sam2_surface_half"][525:])
        ranked = []
        for i, entry in enumerate(entries):
            for j, candidate in enumerate(entry["connection"]["candidates"]):
                ranked.append((float(candidate["rank_score"]), i, j, candidate))
        ranked.sort(key=lambda value: (
            entries[value[1]]["task_direction"]["selection_allowed"],
            value[0], value[3].get("metrics", {}).get("free", {}).get("quality", 0.)), reverse=True)
        selected = ranked[0][1] if ranked else 0
        entry = entries[selected]
        seed = entry["seed"]
        full_free = entry["full_free_observation"]
        free_points = np.asarray(full_free.get("points_xy", []), float).reshape(-1, 2)
        statuses = np.asarray(full_free.get("point_status", []))
        evidence_mask = (np.asarray(full_free.get("supported", [False] * len(free_points)), bool) |
                         np.asarray(full_free.get("raw_a7_supported", [False] * len(free_points)), bool))
        if len(statuses) == len(evidence_mask):
            evidence_mask &= statuses != "entry_condition"
        base.update(seed_xy=seed["xy"], seed_source=seed["source"], selected_entry=selected,
                    line_points_xy=_original(free_points[evidence_mask]),
                    model_forward_count=len(entries),
                    status="CURRENT_FREE_ONLY")
        diagnostics = base["diagnostics"]
        diagnostics.update(
            current_D4_observation_mask=observation_mask_stats,
            current_A7_seed=seed, A7_candidate_count=len(a7.get("candidates", [])),
            current_D4_curve_count=len(core["fit"].get("curves", [])),
            current_D4_point_count=len(core.get("points", [])),
            current_D4_track_count=len(core.get("tracks", [])),
            H6_track_count=len(tracksets[selected]["tracks"]),
            entry_summary=[dict(seed=e["seed"], neural_seconds=e["neural_seconds"],
                                H6_seconds=e["H6_seconds"], task_direction=e["task_direction"],
                                free_point_count=len(e["free"].get("points_xy", [])),
                                connection_candidates=len(e["connection"]["candidates"]),
                                score_margin=e["connection"].get("score_margin"),
                                rejected=e["connection"].get("rejected"),
                                geometric_proposal_count=e["connection"].get("geometric_proposal_count"))
                           for e in entries],
            ranked_candidates=[dict(entry=i, candidate=j, kind=c["kind"], rank_score=score,
                                    track_id=c.get("track_id"), continuation_eligible=c.get("continuation_eligible"),
                                    task_direction_allowed=entries[i]["task_direction"]["selection_allowed"])
                               for score, i, j, c in ranked],
            free_diagnostics=full_free.get("diagnostics", {}),
            free_evidence_summary=full_free.get("support_summary", {}),
            current_D4_is_complete_historical_F3_fusion=False,
            inherited_H6_track_diagnostics=tracksets[selected].get("diagnostics", {}))
        if ranked:
            candidate = ranked[0][3]
            good, checks, reasons, segments, observed = self._support(entry, candidate, ranked)
            base.update(path_xy=_original(candidate["points_xy"]),
                        whole_path_candidate_supported=good,
                        selected_kind=candidate["kind"],
                        status="WHOLE_PATH_CANDIDATE_SUPPORTED" if good else "CURRENT_PATH_DIAGNOSTIC",
                        reasons=reasons)
            diagnostics.update(
                path_segment_support=segments, path_point_support=observed,
                path_stage=candidate["stage"], path_point_status=candidate["point_status"],
                whole_path_support_checks=checks, H6_candidate_metrics=candidate.get("metrics", {}),
                H6_topology=candidate.get("topology"),
                H6_termination=candidate.get("termination"),
                selected_track_id=candidate.get("track_id"),
                join_start_xy=_original([candidate["join_start_xy"]])[0] if candidate.get("join_start_xy") is not None else None,
                join_end_xy=_original([candidate["join_end_xy"]])[0] if candidate.get("join_end_xy") is not None else None)
        else:
            base["reasons"] = ["H6_no_scored_path_candidate"]
        timings["total_seconds"] = time.perf_counter() - start
        return _plain(base)

