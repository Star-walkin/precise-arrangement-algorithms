"""Fixed seed-anchored beam for V11 column distributions, never a force command.

Concentrated learned probability is only model evidence; it is not independent
visibility/rope identity. Uniform or diffuse columns cannot be renormalized into
observations. Internal bounded gaps are always split in exported image paths.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .v11_path_contract import HEIGHT, WIDTH, STEP_WIDTHS, _checked_geometry, strip_to_image


DECODE_PARAMETERS = dict(anchor_u_widths=-2., anchor_half_widths=.75,
    minimum_window_mass=.45, minimum_peak_to_column_median=2.,
    window_radius_target_sigmas=2., maximum_absolute_slope=1.2,
    maximum_missing_columns=2, beam_size=16, maximum_peaks_per_column=24,
    peak_spacing_widths=.25, slope_penalty=.06, curvature_penalty=.15,
    learned_slope_penalty=.06, missing_penalty=1., slope_smoothing=.65,
    position_cell_rows=.75, slope_cell=.20, alternative_count=3,
    ambiguity_average_reward_gap=.10)


def _column_nodes(probability, valid, column, radius):
    values = np.where(valid[:, column], probability[:, column], 0.)
    cumulative = np.r_[0., np.cumsum(values)]
    rows = np.arange(HEIGHT)
    lo, hi = np.maximum(0, rows-radius), np.minimum(HEIGHT, rows+radius+1)
    mass = cumulative[hi]-cumulative[lo]
    # Window must be genuinely observed, not a tiny surviving fragment on which
    # a masked softmax has artificially concentrated an otherwise flat map.
    valid_count = np.r_[0, np.cumsum(valid[:, column])]
    window_valid = ((valid_count[hi]-valid_count[lo]) == 2*radius+1)
    local_max = ((mass[1:-1] >= mass[:-2]) & (mass[1:-1] > mass[2:]))
    indices = np.flatnonzero(local_max)+1
    median = float(np.median(values[valid[:, column]])) if valid[:, column].any() else 0.
    selected = []
    for row in sorted(indices, key=lambda item: (-float(mass[item]), int(item))):
        if not window_valid[row] or mass[row] < .45:
            continue
        peak = float(np.max(values[lo[row]:hi[row]]))
        if peak < max(1e-10, median*2):
            continue
        if any(abs(row-old['row']) < .25/STEP_WIDTHS for old in selected):
            continue
        local = values[lo[row]:hi[row]]
        center = float(np.sum(np.arange(lo[row], hi[row])*local)/max(float(local.sum()), 1e-12))
        selected.append(dict(column=column, row=center, mass=float(mass[row]), observed=True))
        if len(selected) == 24:
            break
    return selected


def _segments(state, geometry):
    segments, current = [], []
    for node in state['path']:
        if node['observed']:
            current.append([node['column'], node['row']])
        else:
            if len(current) >= 2:
                segments.append(strip_to_image(np.asarray(current), geometry).tolist())
            current = []
    if len(current) >= 2:
        segments.append(strip_to_image(np.asarray(current), geometry).tolist())
    return segments


def decode_path(probability, slope, geometry, valid_mask):
    """Decode row probability distributions with fixed synthetic-tested gates.

    ``slope`` is optional per-column dv/du. Masking never renormalizes model
    probabilities. Predicted mass away from valid pixels is lost, not silently
    reassigned. No target/quality/ground-truth parameters exist in this API.
    """
    started = time.perf_counter()
    _checked_geometry(geometry)
    probability = np.asarray(probability, np.float64)
    valid = np.asarray(valid_mask, bool)
    if probability.shape != (HEIGHT, WIDTH) or valid.shape != (HEIGHT, WIDTH):
        raise ValueError('probability and valid_mask must be 256x512')
    if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1+1e-6:
        raise ValueError('probability must be finite in [0,1]')
    if np.any(probability.sum(axis=0) > 1+1e-4):
        raise ValueError('probability must be a row distribution, not independent sigmoid pixels')
    if slope is not None:
        slope = np.asarray(slope, np.float64)
        if slope.shape != (WIDTH,) or not np.isfinite(slope).all():
            raise ValueError('slope must be a finite 512-vector or None')
    sigma = max(1., 2./float(geometry['native_px_per_row']))
    radius = max(1, int(math.ceil(2*sigma)))
    anchor = int(round((-2.-(-8.))/STEP_WIDTHS-.5))
    anchor_row = (-0.-(-24.))/STEP_WIDTHS-.5
    columns = list(range(anchor, WIDTH))
    peak_lists = [_column_nodes(probability, valid, column, radius) for column in columns]
    initial = [node for node in peak_lists[0] if abs(node['row']-anchor_row) <= .75/STEP_WIDTHS]
    beam = [dict(path=(node,), score=node['mass']-.05*abs(node['row']-anchor_row),
                 slope=0., gaps=0) for node in initial]
    finished = []
    for column, nodes in zip(columns[1:], peak_lists[1:]):
        extensions = []
        for state in beam:
            last = state['path'][-1]
            predicted = last['row']+state['slope']
            accepted = 0
            for node in nodes:
                measured_slope = node['row']-last['row']
                if abs(measured_slope) > 1.2:
                    continue
                change = abs(measured_slope-state['slope'])
                reward = node['mass']-.06*abs(measured_slope)-.15*change
                if slope is not None:
                    reward -= .06*abs(measured_slope-float(slope[column]))
                extensions.append(dict(path=state['path']+(node,), score=state['score']+reward,
                    slope=.65*measured_slope+.35*state['slope'], gaps=0))
                accepted += 1
            if state['gaps'] < 2:
                missing = dict(column=column, row=predicted, mass=0., observed=False)
                extensions.append(dict(path=state['path']+(missing,),
                    score=state['score']-1., slope=state['slope'], gaps=state['gaps']+1))
            elif not accepted:
                finished.append(state)
        if not extensions:
            beam = []
            break
        occupied, next_beam = set(), []
        for state in sorted(extensions, key=lambda item: item['score'], reverse=True):
            cell = (round(state['path'][-1]['row']/.75), round(state['slope']/.2), state['gaps'])
            if cell not in occupied:
                occupied.add(cell)
                next_beam.append(state)
            if len(next_beam) == 16:
                break
        beam = next_beam
    finished.extend(beam)
    ranked = []
    for state in sorted(finished, key=lambda item: item['score'], reverse=True):
        segments = _segments(state, geometry)
        if not segments:
            continue
        # Do not multiply nearly identical beam cells into false alternatives.
        end = np.asarray(segments[-1][-1])
        if any(np.linalg.norm(end-np.asarray(old['segments'][-1][-1])) < geometry['width_px']*.25
               for old in ranked):
            continue
        ranked.append(dict(segments=segments, score=float(state['score']),
            average_reward=float(state['score']/len(state['path'])),
            observed_columns=sum(node['observed'] for node in state['path']),
            missing_columns=sum(not node['observed'] for node in state['path']),
            identity_confirmed=False))
        if len(ranked) == 3:
            break
    segments = ranked[0]['segments'] if ranked else []
    ambiguous = (len(ranked) > 1 and abs(ranked[0]['average_reward']-ranked[1]['average_reward']) < .10)
    reason = ('model_path_candidate_identity_unconfirmed' if segments else
              'no_seed_anchored_concentrated_distribution')
    return dict(segments=segments, observed_segments=segments, candidates=ranked,
        valid=bool(segments), reason=reason, identity_confirmed=False,
        physical_contact_confirmed=False, new_layer_confirmed=False, force_command=None,
        diagnostics=dict(parameters=dict(DECODE_PARAMETERS), anchor_column=anchor,
            window_radius_rows=radius, target_sigma_rows=sigma,
            candidate_columns=sum(bool(nodes) for nodes in peak_lists),
            ambiguous=bool(ambiguous), probability_was_renormalized=False,
            interpolated_gap_points_emitted=False, quality_or_targets_used=False,
            probability_is_visibility_confirmation=False, elapsed_ms=(time.perf_counter()-started)*1000))
