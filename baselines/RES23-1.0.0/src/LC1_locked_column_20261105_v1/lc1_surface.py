"""Current-frame traditional surface observations for the LC1 locked column.

This adapter runs the unchanged RB-F3 observation core on a narrow mask, not
the historical C3/F2/F3 fusion/cache chain. Raw D4 observation tracks and fitted
curves remain separate. Only short, image-supported raw brackets cross x=d.
The caller must put frame, masks, d and active_y in one coordinate system.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

VENDOR = Path(__file__).resolve().parent / "vendor_archive"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
from f3_core_adapter import detect_surface_core


def _finite(value):
    return value is not None and math.isfinite(float(value))


def _longest_true(values):
    flags = np.asarray(values, bool)
    edges = np.diff(np.r_[False, flags, False].astype(np.int8))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return int(max(stops - starts, default=0))


class SurfaceObserver:
    """Measure visible groove geometry in the input frame's original pixels.

    No resize or mirror is performed. For a mirrored-left analysis, mirror the
    surface/exclusion masks along with the frame before calling this class.
    count_valid means its stated observation/continuity gates passed; it does
    not assert independently validated physical rope identity/count accuracy.
    """

    def __init__(self, surface_mask_original, pitch_px, c3_record=None, *,
                 band_halfwidth_pitches=4.0):
        mask = np.asarray(surface_mask_original)
        if mask.ndim != 2 or mask.dtype != np.uint8:
            raise ValueError("surface mask must be a uint8 HxW image")
        if np.any((mask != 0) & (mask != 255)):
            raise ValueError("surface mask values must be 0/255")
        self.pitch = float(pitch_px)
        if not math.isfinite(self.pitch) or not 8 <= self.pitch <= 70:
            raise ValueError("pitch_px must be 8..70 original pixels")
        self.surface = mask.copy()
        self.c3_record = c3_record
        self.band_halfwidth_pitches = max(2.0, float(band_halfwidth_pitches))
        columns = np.any(mask > 0, axis=0)
        self.top_by_x = np.full(mask.shape[1], np.nan, float)
        if np.any(columns):
            self.top_by_x[columns] = np.argmax(mask[:, columns] > 0, axis=0)
            self.default_x = float(np.median(np.flatnonzero(columns)))
        else:
            self.default_x = float((mask.shape[1] - 1) / 2)

    def _edge_prior(self, x):
        lo = max(0, int(math.floor(x - self.pitch)))
        hi = min(self.surface.shape[1], int(math.ceil(x + self.pitch)) + 1)
        cols = np.arange(lo, hi, max(1, int(round(self.pitch / 12))))
        good = np.isfinite(self.top_by_x[cols])
        cols = cols[good]
        if not len(cols):
            return None, cols
        # Interpolate the static top only inside its local observed x range.
        # It is a search prior, never a current-frame measurement.
        top = self.top_by_x[cols]
        prior = float(np.interp(x, cols, top)) if cols[0] <= x <= cols[-1] else float(np.median(top))
        return prior, cols

    def _edge(self, image, x, active_y, exclusion):
        prior, cols = self._edge_prior(x)
        result = dict(edge_y=prior, edge_source="static_surface_top_candidate" if prior is not None else "unavailable",
                      edge_confidence=0.0, edge_current_frame_confirmed=False)
        diag = dict(search_prior_y=prior, search_radius_pitch=2.0,
                    method="yellow_above_silver_below_plus_current_color_gradient",
                    columns=int(len(cols)))
        if prior is None or len(cols) < 5:
            diag["reason"] = "insufficient_static_top_support"
            return result, diag
        pitch = self.pitch
        offsets = np.arange(-int(math.ceil(2 * pitch)), int(math.ceil(2 * pitch)) + 1)
        centers = np.rint(self.top_by_x[cols][None, :] + offsets[:, None]).astype(int)
        band = np.arange(max(2, int(round(.20 * pitch))), max(3, int(round(.65 * pitch))) + 1)
        h = image.shape[0]
        safe = (centers - band[-1] >= 0) & (centers + band[-1] < h)
        clipped = np.clip(centers, band[-1], h - band[-1] - 1)
        above = image[clipped[:, :, None] - band[None, None, :], cols[None, :, None]].mean(axis=2).astype(float)
        below = image[clipped[:, :, None] + band[None, None, :], cols[None, :, None]].mean(axis=2).astype(float)

        def chroma(a):
            b, g, r = a[..., 0], a[..., 1], a[..., 2]
            light = .114 * b + .587 * g + .299 * r
            yellow = (.5 * (r + g) - b) / np.maximum(light, 32)
            saturation = (a.max(axis=-1) - a.min(axis=-1)) / np.maximum(a.max(axis=-1), 32)
            return light, yellow, saturation

        la, ya, sa = chroma(above)
        lb, yb, sb = chroma(below)
        color_jump = np.sqrt(np.mean((above - below) ** 2, axis=-1))
        clear = exclusion[np.clip(centers, 0, h - 1), cols[None, :]] == 0
        if _finite(active_y):
            # An active rope exactly over the rim cannot certify the hidden rim.
            clear &= np.abs(centers - float(active_y)) > .45 * pitch
        support = (safe & clear & (ya > .12) & (ya - yb > .08)
                   & (sb < .48) & (lb > 28) & (color_jump > 8))
        score = (.50 * np.clip((ya - yb) / .30, 0, 1)
                 + .20 * np.clip((sa - sb) / .30, 0, 1)
                 + .30 * np.clip(color_jump / 45, 0, 1))
        score[~support] = 0
        fraction = support.mean(axis=1)
        response = np.median(score, axis=1) * fraction - .035 * np.abs(offsets) / pitch
        k = int(np.argmax(response))
        nearby = np.abs(offsets - offsets[k]) <= max(2, .18 * pitch)
        # Saturated color evidence can form a broad equal-score plateau. Use
        # the same small prior-distance tie break as the aggregate response;
        # taking the first maximum would manufacture column disagreement.
        col_best = offsets[np.argmax(score - .035 * np.abs(offsets[:, None]) / pitch, axis=0)]
        agreement = float(np.mean(np.abs(col_best - offsets[k]) <= .4 * pitch))
        confidence = float(np.clip(.65 * np.median(score[k]) + .20 * fraction[k] + .15 * agreement, 0, 1))
        diag.update(best_offset_px=int(offsets[k]), supported_column_fraction=float(fraction[k]),
                    offset_agreement=agreement, yellow_drop_median=float(np.median((ya - yb)[k])),
                    color_jump_median=float(np.median(color_jump[k])), confidence=confidence,
                    supported_offset_width_px=int(np.count_nonzero(nearby & (fraction >= .5))))
        if fraction[k] >= .55 and agreement >= .5 and confidence >= .42:
            edge_y = float(prior + offsets[k])
            result.update(edge_y=edge_y, edge_source="current_yellow_silver_gradient",
                          edge_confidence=confidence, edge_current_frame_confirmed=True)
            diag["reason"] = "current_color_transition_supported"
        else:
            diag["reason"] = "current_color_transition_not_confirmed"
        return result, diag

    def _observed_segments(self, result):
        segments, records = [], []
        max_gap = max(3.0, .32 * self.pitch)
        for track in result.get("tracks", []):
            points = sorted(track.get("points", []), key=lambda p: (float(p["x"]), -float(p.get("score", 0))))
            part = []
            for point in points:
                if part:
                    dx = float(point["x"]) - float(part[-1]["x"])
                    if dx <= 0 or dx > max_gap:
                        if len(part) >= 3:
                            segments.append([[float(p["x"]), float(p["y"])] for p in part])
                            records.append((track["track_id"], part))
                        part = []
                part.append(point)
            if len(part) >= 3:
                segments.append([[float(p["x"]), float(p["y"])] for p in part])
                records.append((track["track_id"], part))
        return segments, records

    def _crossings(self, records, d, result):
        candidates, rejected = [], dict(no_bracket=0, excluded_bracket=0, duplicate=0)
        unknown = result["exclusion_mask"]
        for track_id, part in records:
            xy = np.asarray([[p["x"], p["y"]] for p in part], float)
            xs = xy[:, 0]
            if not xs[0] <= d <= xs[-1]:
                rejected["no_bracket"] += 1
                continue
            j = int(np.searchsorted(xs, d, side="left"))
            if j < len(xs) and abs(xs[j] - d) < 1e-7:
                a = b = j
                y = float(xy[j, 1])
            else:
                a, b = j - 1, j
                if a < 0 or b >= len(xs):
                    rejected["no_bracket"] += 1
                    continue
                y = float(np.interp(d, xs[a:b + 1], xy[a:b + 1, 1]))
            line = np.linspace(xy[a], xy[b], max(2, int(math.ceil(np.linalg.norm(xy[b] - xy[a]))) + 1))
            rr = np.rint(line).astype(int)
            if (np.any(unknown[rr[:, 1], rr[:, 0]] > 0)
                    or np.any(self.surface[rr[:, 1], rr[:, 0]] == 0)):
                rejected["excluded_bracket"] += 1
                continue
            candidates.append(dict(y=y, track_id=str(track_id),
                                   bracket_xy=xy[a:b + 1].tolist(),
                                   source_point_ids=[int(part[k]["point_id"]) for k in range(a, b + 1)],
                                   score=float(np.mean([part[k].get("score", 0) for k in range(a, b + 1)])),
                                   observed_segment_span_px=float(xs[-1] - xs[0]),
                                   status="raw_observed_short_bracket_interpolation" if a != b else "raw_observed_point"))
        kept = []
        for crossing in sorted(candidates, key=lambda p: (p["score"], p["observed_segment_span_px"]), reverse=True):
            if any(abs(crossing["y"] - other["y"]) < .45 * self.pitch for other in kept):
                rejected["duplicate"] += 1
                continue
            kept.append(crossing)
        kept.sort(key=lambda p: p["y"])
        return kept, rejected

    @staticmethod
    def _phase(y, observed_y, pitch):
        # No boundary is invented. Only an endpoint's fractional phase uses
        # the neighboring measured pitch, explicitly reported by the caller.
        if not len(observed_y):
            return None
        if y < observed_y[0]:
            return float((y - observed_y[0]) / pitch)
        if y > observed_y[-1]:
            return float(len(observed_y) - 1 + (y - observed_y[-1]) / pitch)
        return float(np.interp(y, observed_y, np.arange(len(observed_y), dtype=float)))

    def _count(self, d, active_y, crossings, edge, exclusion):
        reasons = []
        output = dict(count_observed_grooves=0, observed_groove_count_lower_bound=0,
                      continuous_pitch_distance=None, continuous_pitch_distance_candidate=None,
                      count_valid=False, local_pitch_px=self.pitch, local_pitch_source="static_pitch_prior",
                      count_status="unavailable")
        diag = dict(duplicate_distance_pitch=.45, maximum_gap_pitch=1.6,
                    count_interpolation="phase between observed grooves; endpoint fraction uses local measured pitch",
                    physical_identity_validated=False)
        if not _finite(d):
            return output, ["measurement_column_not_locked"], diag
        if not _finite(active_y):
            return output, ["active_y_unavailable"], diag
        if edge["edge_y"] is None:
            return output, ["edge_unavailable"], diag
        y0, y1 = float(edge["edge_y"]), float(active_y)
        if not 0 <= y1 < self.surface.shape[0]:
            return output, ["active_y_outside_image"], diag
        if y1 < y0:
            return output, ["active_y_above_edge"], diag
        ys = np.asarray([p["y"] for p in crossings], float)
        nearby = ys[(ys >= y0 - self.pitch) & (ys <= y1 + 5 * self.pitch)]
        gaps = np.diff(nearby)
        plausible = gaps[(gaps >= .65 * self.pitch) & (gaps <= 1.45 * self.pitch)]
        if len(plausible) >= 2:
            local_pitch = float(np.median(plausible))
            output.update(local_pitch_px=local_pitch, local_pitch_source="current_observed_crossing_median")
        else:
            local_pitch = self.pitch
            reasons.append("local_pitch_not_confirmed_by_two_observed_intervals")
        inner = ys[(ys > y0 + 1e-6) & (ys < y1 - 1e-6)]
        output["count_observed_grooves"] = int(len(inner))
        output["observed_groove_count_lower_bound"] = int(len(inner))
        output["continuous_pitch_distance_candidate"] = float((y1 - y0) / local_pitch)
        if not edge["edge_current_frame_confirmed"]:
            reasons.append("edge_is_static_prior_only")
        sequence = np.r_[y0, inner, y1]
        all_gaps = np.diff(sequence)
        largest = float(all_gaps.max(initial=0))
        if largest > 1.6 * local_pitch:
            reasons.append("unsupported_count_gap_gt_1_6_pitch")
        actual_gaps = np.diff(ys[(ys >= y0) & (ys <= y1)])
        if np.any(actual_gaps < .65 * local_pitch):
            reasons.append("sub_pitch_competing_grooves_after_deduplication")
        ix = int(np.clip(round(float(d)), 0, self.surface.shape[1] - 1))
        lo = max(0, int(math.ceil(y0 + .2 * local_pitch)))
        hi = min(self.surface.shape[0], int(math.floor(y1 - .2 * local_pitch)) + 1)
        unknown_run = _longest_true(exclusion[lo:hi, ix] > 0) if hi > lo else 0
        if unknown_run > .4 * local_pitch:
            reasons.append("unknown_or_excluded_region_on_count_segment")
        if not len(ys):
            reasons.append("no_observed_groove_crossings")
        endpoint_support = []
        for name, y in (("edge", y0), ("active", y1)):
            if not len(ys):
                endpoint_support.append(dict(endpoint=name, nearest_observed_distance_px=None))
                continue
            nearest = float(np.min(np.abs(ys - y)))
            item = dict(endpoint=name, nearest_observed_distance_px=nearest)
            if nearest > 1.6 * local_pitch:
                reasons.append(name + "_endpoint_far_from_observed_grooves")
            j = int(np.searchsorted(ys, y, side="right"))
            if 0 < j < len(ys) and abs(y - ys[j - 1]) > 1e-7:
                bracket_gap = float(ys[j] - ys[j - 1])
                item["observed_bracket_gap_px"] = bracket_gap
                if bracket_gap > 1.6 * local_pitch:
                    reasons.append(name + "_phase_bracket_has_unobserved_large_gap")
            endpoint_support.append(item)
        diag.update(interval_y=[y0, y1], local_pitch_intervals=plausible.tolist(),
                    count_segment_gaps_px=all_gaps.tolist(), maximum_count_segment_gap_px=largest,
                    maximum_unknown_run_px=unknown_run, observed_crossings_between_endpoints=int(len(inner)),
                    endpoint_support=endpoint_support)
        if not reasons:
            phase0, phase1 = self._phase(y0, ys, local_pitch), self._phase(y1, ys, local_pitch)
            output.update(continuous_pitch_distance=float(phase1 - phase0), count_valid=True,
                          count_status="continuous_observed_pitch_coordinate")
        else:
            output["count_status"] = "observed_grooves_lower_bound_and_distance_candidate"
        return output, reasons, diag

    def step(self, bgr, d=None, active_y=None, exclusion_mask=None):
        tick = time.perf_counter()
        image = np.asarray(bgr)
        if image.dtype != np.uint8 or image.shape != (*self.surface.shape, 3):
            raise ValueError("bgr must be same-size uint8 HxWx3 as the supplied surface mask")
        if exclusion_mask is None:
            exclusion = np.zeros_like(self.surface)
        else:
            exclusion = np.asarray(exclusion_mask)
            if exclusion.shape != self.surface.shape or exclusion.dtype != np.uint8 or np.any((exclusion != 0) & (exclusion != 255)):
                raise ValueError("exclusion mask must be same-size uint8 0/255")
        h, w = self.surface.shape
        if _finite(d) and not 0 <= float(d) < w:
            raise ValueError("d must be within the image")
        x = float(d) if _finite(d) else self.default_x
        edge, edge_diag = self._edge(image, x, active_y, exclusion)
        half = max(48, int(math.ceil(self.band_halfwidth_pitches * self.pitch)))
        x0, x1 = max(0, int(math.floor(x - half))), min(w, int(math.ceil(x + half)) + 1)
        # Crop only the mask. Every vendor array and returned coordinate remains
        # at original resolution; the pitch is never rescaled.
        band_mask = np.zeros_like(self.surface)
        y_stop = h
        if _finite(active_y) and edge["edge_y"] is not None and float(active_y) >= edge["edge_y"]:
            y_stop = min(h, max(0, int(math.ceil(float(active_y) + 6 * self.pitch))))
        band_mask[:y_stop, x0:x1] = self.surface[:y_stop, x0:x1]
        core_tick = time.perf_counter()
        result = detect_surface_core(image, band_mask, self.pitch,
                                     exclusion_mask=exclusion, c3_record=self.c3_record)
        core_seconds = time.perf_counter() - core_tick
        observed, records = self._observed_segments(result)
        if _finite(d):
            crossings, crossing_diag = self._crossings(records, float(d), result)
        else:
            crossings, crossing_diag = [], dict(no_bracket=0, excluded_bracket=0, duplicate=0)
        fit = result.get("fit", {}).get("curves", [])
        counts, reasons, count_diag = self._count(d, active_y, crossings, edge, result["exclusion_mask"])
        quality = "current_observation_geometry_passed" if counts["count_valid"] else "partial_observation" if crossings else "unresolved"
        return dict(version="LC1_surface_F3_observation_adapter_1", coordinate_system="input_full_image_xy",
                    method="unchanged_RB_F3_D4_observation_core_with_narrow_surface_mask_not_full_F3_fusion",
                    measurement_x=float(d) if _finite(d) else None,
                    boundary_polylines_xy=observed,
                    boundary_track_ids=[str(track_id) for track_id, _ in records],
                    fit_polylines_xy=[curve.get("sampled_points_xy", []) for curve in fit],
                    fit_curve_ids=[str(curve.get("curve_id", "")) for curve in fit],
                    fit_point_states=[curve.get("point_states", []) for curve in fit],
                    crossings_y=[float(item["y"]) for item in crossings], crossing_records=crossings,
                    **edge, **counts, quality=quality, reasons=reasons,
                    diagnostics=dict(edge=edge_diag, count=count_diag, crossings=crossing_diag,
                                     narrow_mask_x_range=[x0, x1], narrow_mask_y_stop=y_stop,
                                     narrowed_surface_pixels=int(np.count_nonzero(band_mask)),
                                     core_seconds=float(core_seconds), total_seconds=float(time.perf_counter() - tick),
                                     raw_point_count=len(result.get("points", [])), raw_track_count=len(result.get("tracks", [])),
                                     observed_segment_count=len(observed), fit_curve_count=len(fit),
                                     frame_resize=False, frame_mirror=False, neural_models_invoked=False,
                                     core_observation_diagnostics=result.get("diagnostics", {})))
