"""Original-image proposals for a straight active-rope guide.

E1, any frozen alternate entries, and independently detected image axes are
competing proposals.  All are measured by the same image-only diagnostics.
Selection may return AMBIGUOUS; an E1 label is never treated as ground truth.
The optional surface mask is a proposal search ROI, never a material label.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
A7_RECOVERY = (HERE.parents[2] / "数据资产" / "05_派生与实验结果"
               / "camera_condition_free_straight_20261030_v7" / "free_seed_recovery.py")
DEFAULTS = {
    "auto_pitch_px": 24.0,
    "auto_candidates": 5,
    "minimum_score": 0.27,
    "minimum_score_margin": 0.11,
    "deduplicate_axis_offset_px": 22.0,
    "deduplicate_angle_deg": 3.0,
    "guide_span_px": 120.0,
    "texture_minimum_energy": 1.0,
}
_RECOVERY = None


def _unit(value: Any) -> np.ndarray:
    a = np.asarray(value, np.float64).reshape(2)
    norm = float(np.linalg.norm(a))
    if norm < 1e-6 or not np.isfinite(norm):
        raise ValueError("degenerate direction")
    return a / norm


def _recovery_module():
    global _RECOVERY
    if _RECOVERY is None:
        if not A7_RECOVERY.is_file():
            raise FileNotFoundError(A7_RECOVERY)
        spec = importlib.util.spec_from_file_location("_straight_guidance_a7_proposals", A7_RECOVERY)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        _RECOVERY = module
    return _RECOVERY


def _fallback_surface(shape: tuple[int, int]) -> np.ndarray:
    """Loose original-frame winding search ROI, not a semantic spool mask."""
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (int(.50 * w), int(.91 * h)),
                (int(.34 * w), int(.32 * h)), 0, 0, 360, 255, -1)
    return mask


def _seed_from_axis(segment: Any, target_xy: np.ndarray, length: float) -> list[list[float]]:
    q = np.asarray(segment, np.float64).reshape(2, 2)
    d = _unit(q[1] - q[0])
    center = q.mean(axis=0)
    # The end toward the lower central winding area is the second guide point.
    if float(d @ (target_xy - center)) < 0:
        d = -d
    return [list(map(float, center - .5 * length * d)),
            list(map(float, center + .5 * length * d))]


def _texture(gray: np.ndarray, seed: np.ndarray, width: float) -> dict:
    """Oblique strand modulation inside the candidate, excluding outer edges."""
    d = _unit(seed[1] - seed[0])
    n = np.array([-d[1], d[0]])
    center = seed.mean(axis=0)
    t = np.arange(-52, 53, 1.0, dtype=np.float32)
    q = np.arange(-max(2.0, .29 * width), max(2.0, .29 * width) + .5, 1.0, dtype=np.float32)
    xy = center + t[:, None, None] * d + q[None, :, None] * n
    h, w = gray.shape
    valid = (xy[..., 0] >= 2) & (xy[..., 0] < w - 3) & (xy[..., 1] >= 2) & (xy[..., 1] < h - 3)
    if valid.mean() < .9:
        return {"status": "border", "energy": 0.0, "oblique": 0.0, "coherence": 0.0}
    patch = cv2.remap(gray, xy[..., 0].astype(np.float32),
                      xy[..., 1].astype(np.float32), cv2.INTER_LINEAR)
    high = patch - cv2.GaussianBlur(patch, (0, 0), sigmaX=1.2, sigmaY=max(.8, .13 * width))
    gt = cv2.Sobel(high, cv2.CV_32F, 0, 1, ksize=3, scale=.125)[2:-2, 1:-1]
    gn = cv2.Sobel(high, cv2.CV_32F, 1, 0, ksize=3, scale=.125)[2:-2, 1:-1]
    jtt = float(np.mean(gt * gt)); jnn = float(np.mean(gn * gn)); jtn = float(np.mean(gt * gn))
    energy = jtt + jnn
    oblique = abs(jtn) / max(math.sqrt(jtt * jnn), 1e-6)
    coherence = math.hypot(jtt - jnn, 2 * jtn) / max(energy, 1e-6)
    return {"status": "supported" if energy >= 1.0 else "weak",
            "energy": energy, "oblique": float(oblique),
            "coherence": float(coherence), "body_mean": float(patch.mean())}


def _paired_profile(gray: np.ndarray, seed: np.ndarray, width: float) -> dict:
    d = _unit(seed[1] - seed[0]); n = np.array([-d[1], d[0]])
    centers = seed[0] + np.linspace(.12, .88, 9)[:, None] * (seed[1] - seed[0])
    offsets = np.arange(-1.3 * width, 1.3 * width + .5, .5, np.float32)
    grid = centers[:, None, :] + offsets[None, :, None] * n
    h, w = gray.shape
    valid = (grid[..., 0] >= 2) & (grid[..., 0] < w - 3) & (grid[..., 1] >= 2) & (grid[..., 1] < h - 3)
    if valid.mean() < .92:
        return {"status": "border", "fraction": 0.0, "strength": 0.0, "contrast": 0.0}
    patch = cv2.remap(gray, grid[..., 0].astype(np.float32),
                      grid[..., 1].astype(np.float32), cv2.INTER_LINEAR)
    patch = cv2.GaussianBlur(patch, (5, 1), sigmaX=.8)
    derivative = np.gradient(patch, .5, axis=1)
    left = (offsets >= -.7 * width) & (offsets <= -.3 * width)
    right = (offsets >= .3 * width) & (offsets <= .7 * width)
    body = abs(offsets) <= .22 * width
    lb = (offsets >= -1.2 * width) & (offsets <= -.9 * width)
    rb = (offsets >= .9 * width) & (offsets <= 1.2 * width)
    best = None
    for polarity in (-1, 1):
        l = np.max(polarity * derivative[:, left], axis=1)
        r = np.max(-polarity * derivative[:, right], axis=1)
        strength = np.minimum(l, r)
        contrast = np.minimum(polarity * (np.median(patch[:, body], axis=1) - np.median(patch[:, lb], axis=1)),
                              polarity * (np.median(patch[:, body], axis=1) - np.median(patch[:, rb], axis=1)))
        good = (strength >= .8) & (contrast >= 3.0)
        score = float(good.mean() * min(1.0, np.median(np.maximum(strength, 0)) / 4))
        item = {"status": "supported" if good.sum() >= 4 else "weak",
                "fraction": float(good.mean()), "strength": float(np.median(strength)),
                "contrast": float(np.median(contrast)), "polarity": polarity,
                "score": score}
        if best is None or item["score"] > best["score"]:
            best = item
    return best


def _score_candidate(gray: np.ndarray, candidate: dict, target: np.ndarray) -> dict:
    seed = np.asarray(candidate["seed_xy"], np.float64).reshape(2, 2)
    d = _unit(seed[1] - seed[0]); width = float(candidate.get("width_px") or 22.0)
    tex = _texture(gray, seed, width)
    paired = _paired_profile(gray, seed, width)
    oblique = min(1.0, tex["oblique"] / .42) * min(1.0, tex["energy"] / 4.0)
    paired_score = paired.get("score", 0.0)
    approach = max(0.0, float(d @ (target - seed.mean(axis=0)))) / max(float(np.linalg.norm(target - seed.mean(axis=0))), 1e-6)
    # A7 proposal score is a bounded original-image prior, never a label.
    proposal = min(1.0, max(0.0, float(candidate.get("proposal_score", 0.0))))
    score = .43 * oblique + .33 * paired_score + .12 * approach + .12 * proposal
    candidate["score"] = round(float(score), 6)
    candidate["evidence"] = {"texture": tex, "paired_edges": paired,
                              "winding_approach_cosine": float(approach),
                              "proposal_score": proposal}
    candidate["axis_direction_xy"] = d.astype(float).tolist()
    candidate["axis_point_xy"] = seed.mean(axis=0).astype(float).tolist()
    return candidate


def propose_guidance(image_bgr: np.ndarray, e1_xy: Any,
                     *, extra_seeds: list[dict] | None = None,
                     surface_mask: np.ndarray | None = None,
                     pitch_px: float | None = None,
                     config: dict | None = None) -> dict:
    """Return ranked image-checked axes; ambiguity is an explicit outcome.

    `extra_seeds` rows may contain `xy` or `seed_xy`, `entry_id`, `source`.
    Their order has no selection privilege. The returned selected axis is
    still only an optical straight-rope proposal.
    """
    c = DEFAULTS | (config or {})
    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError("expected original BGR uint8 image")
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    target = np.array([.5 * w, .88 * h], np.float64)
    rows = []
    def add_seed(value, source, entry_id, proposal_score=0.0, width=None):
        try:
            xy = np.asarray(value, np.float64).reshape(2, 2)
            if not np.isfinite(xy).all() or np.linalg.norm(xy[1]-xy[0]) < 40:
                return
            d = _unit(xy[1] - xy[0])
            if d @ (target - xy.mean(axis=0)) < 0:
                xy = xy[::-1]
            rows.append({"candidate_id": f"guide_{len(rows):02d}",
                         "seed_xy": xy.astype(float).tolist(), "source": source,
                         "entry_id": entry_id, "width_px": width,
                         "proposal_score": proposal_score})
        except (ValueError, TypeError):
            return
    add_seed(e1_xy, "frozen_e1", "E1")
    for row in extra_seeds or []:
        add_seed(row.get("xy", row.get("seed_xy")), "frozen_alternate",
                 row.get("entry_id", "alternate"), row.get("proposal_score", 0.0), row.get("width_px"))
    mask = np.asarray(surface_mask, np.uint8) if surface_mask is not None else _fallback_surface((h, w))
    if mask.shape != (h, w):
        raise ValueError("surface_mask shape mismatch")
    pitch = float(pitch_px or c["auto_pitch_px"])
    recovered = _recovery_module().recover_free_seeds({"original": image_bgr,
                                                        "surface": mask, "pitch": pitch})
    for proposal in recovered["candidates"][:int(c["auto_candidates"])]:
        seed = proposal["seed"]
        segment = seed["observed_segment_xy"]
        guide = _seed_from_axis(segment, target, float(c["guide_span_px"]))
        add_seed(guide, "auto_image", proposal["candidate_id"],
                 proposal.get("score", 0.0), seed.get("width_px"))
    # Deduplicate optical axes after each candidate has been independently measured.
    graded = [_score_candidate(gray, row, target) for row in rows]
    graded.sort(key=lambda row: -row["score"])
    unique = []
    for row in graded:
        d = np.asarray(row["axis_direction_xy"])
        n = np.array([-d[1], d[0]])
        p = np.asarray(row["axis_point_xy"])
        same = next((u for u in unique if abs(float(d @ np.asarray(u["axis_direction_xy"]))) >= math.cos(math.radians(c["deduplicate_angle_deg"]))
                     and abs(float((p - np.asarray(u["axis_point_xy"])) @ n)) <= float(c["deduplicate_axis_offset_px"])), None)
        if same is None:
            unique.append(row)
        else:
            same.setdefault("equivalent_sources", []).append({"source": row["source"],
                                                                 "entry_id": row["entry_id"],
                                                                 "score": row["score"]})
    selected = None
    status = "UNKNOWN"
    if unique and unique[0]["score"] >= float(c["minimum_score"]):
        if len(unique) == 1 or unique[0]["score"] - unique[1]["score"] >= float(c["minimum_score_margin"]):
            selected = unique[0]["candidate_id"]
            status = "SELECTED"
        else:
            status = "AMBIGUOUS"
    return {"status": status, "selected_candidate_id": selected,
            "candidates": unique, "raw_candidate_count": len(rows),
            "fallback_surface_used": surface_mask is None,
            "auto_recovery_status": recovered["status"],
            "e1_is_input_not_truth": True,
            "complete_material_identity_confirmed": False,
            "config": c}
