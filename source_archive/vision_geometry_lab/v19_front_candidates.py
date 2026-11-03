"""Past-only side-contour expansion candidates, never physical layer events.

Spatial order is increasing canonical image-row index, not calibrated world
position. Thresholds are supplied from the frozen startup profile MAD; no truth
times, mechanical periods or device inputs are used. A delayed periodic wave
can share a persistent front's finite prefix: this module does not resolve that
physical ambiguity and must not drive an actuator.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


RULES = dict(low_duration_s=.4, low_minimum_points=3,
             high_duration_s=.3, high_minimum_points=3,
             maximum_observation_gap_s=.3, minimum_adjacent_rows=3,
             minimum_first_onset_age_s=.8, recent_confirmation_s=.11,
             maximum_chain_history_s=6., low_threshold_fraction=.5,
             minimum_high_threshold_px=3.,
             episode_release='same_emitted_three_rows_new_stable_direct_low',
             candidate_kind='side_contour_persistent_expansion_only')
EPS = 1e-9


@dataclass
class _Row:
    phase: str = 'unarmed'
    armed: bool = False
    low_start: float | None = None
    low_count: int = 0
    last_low: float | None = None
    first_high: float | None = None
    high_count: int = 0
    confirmed_at: float | None = None
    last_direct: float | None = None


class FrontTracker:
    """One bounded-memory image-front tracker with explicit rejection states.

    ``threshold_px`` is scalar or one high threshold per original row, ordinarily
    max(3, 3 * startup_sigma). NaN thresholds retain unsupported original rows;
    they cannot be skipped when finding a three-row chain. Quality accepts bool
    or finite numeric 0/1 arrays; NaN/nonfinite displacement is always unknown.

    running=None means unknown, not stopped/running inferred from camera pose.
    running=False resets evidence; this is an optional external state, not a
    visual motion detector. Repeated static high samples cannot initiate or
    mature an event without the required ordered spatial confirmations.
    """

    def __init__(self, rows_y, threshold_px):
        self.rows = np.asarray(rows_y, dtype=np.float64)
        if (self.rows.ndim != 1 or not np.isfinite(self.rows).all()
                or (len(self.rows) > 1 and not np.all(np.diff(self.rows) > 0))):
            raise ValueError('rows_y must be finite, one-dimensional and strictly increasing')
        threshold = np.asarray(threshold_px, dtype=np.float64)
        if threshold.ndim == 0:
            threshold = np.full(len(self.rows), float(threshold))
        if threshold.shape != self.rows.shape or np.isinf(threshold).any():
            raise ValueError('threshold_px must match rows and contain only finite values or NaN')
        if np.any(threshold[np.isfinite(threshold)] <= 0):
            raise ValueError('finite thresholds must be positive')
        self.high = np.maximum(threshold, RULES['minimum_high_threshold_px'])
        self.low = self.high * RULES['low_threshold_fraction']
        self._candidate_counter = 0
        self.reset()

    def reset(self):
        self._rows = [_Row() for _ in self.rows]
        self.last_time = None
        self.episode_locked = False
        self.episode_rows = None
        self.episode_reported_at = None

    def _clear_row(self, j):
        self._rows[j] = _Row()

    def _snapshot(self, now, displacement, quality, running, state, reason, candidates):
        row_states = []
        for j, row in enumerate(self._rows):
            row_states.append(dict(index=j, row_y=float(self.rows[j]), phase=row.phase,
                armed=row.armed, quality=bool(quality[j]),
                displacement_px=float(displacement[j]) if np.isfinite(displacement[j]) else None,
                high_threshold_px=float(self.high[j]) if np.isfinite(self.high[j]) else None,
                low_threshold_px=float(self.low[j]) if np.isfinite(self.low[j]) else None,
                low_start_s=row.low_start, low_count=row.low_count,
                last_low_s=row.last_low, first_high_s=row.first_high,
                high_count=row.high_count, confirmed_at_s=row.confirmed_at,
                last_direct_s=row.last_direct))
        return dict(time_s=now, state=state, reason=reason, row_states=row_states,
                    episode_locked=self.episode_locked,
                    running_state=('unknown' if running is None else
                                   'explicit_running' if running else 'explicit_stopped'),
                    new_candidates=candidates, material_identity_confirmed=False,
                    new_layer_confirmed=False, mechanical_revolution_confirmed=False,
                    actuator_command=None)

    def _observe_row(self, j, now, value, quality):
        row = self._rows[j]
        history_origin = row.first_high if row.first_high is not None else row.last_low
        if history_origin is not None and now-history_origin > RULES['maximum_chain_history_s']+EPS:
            self._clear_row(j)
            row = self._rows[j]
        if row.last_direct is not None and now-row.last_direct > RULES['maximum_observation_gap_s']+EPS:
            self._clear_row(j)
            row = self._rows[j]
        if not quality:
            # Pending/arming evidence cannot survive a missing sample. Previously
            # confirmed history may remain for <= .3s but is not usable this frame.
            if row.phase != 'confirmed':
                self._clear_row(j)
            return
        row.last_direct = now
        if row.phase == 'confirmed' and value >= self.low[j]:
            return
        if value <= self.low[j]:
            if row.phase != 'low':
                row = _Row(phase='low', low_start=now, last_direct=now)
                self._rows[j] = row
            row.low_count += 1
            row.last_low = now
            row.armed = (row.low_count >= RULES['low_minimum_points']
                         and now-row.low_start >= RULES['low_duration_s']-EPS)
            return
        if value < self.high[j]:
            # A mid-band sample is not a high observation and cannot age a
            # pending high run to confirmation. Previously stable low evidence
            # may arm a future, newly observed high run within the history bound.
            row.phase = 'armed' if row.armed else 'unarmed'
            row.low_start = None
            row.low_count = 0
            row.first_high = None
            row.high_count = 0
            row.confirmed_at = None
            return
        if not row.armed:
            row.phase = 'unarmed'
            row.low_start = None
            row.low_count = 0
            return
        if row.phase != 'pending':
            row.phase = 'pending'
            row.first_high = now
            row.high_count = 0
            row.low_start = None
            row.low_count = 0
            row.confirmed_at = None
        row.high_count += 1
        if (row.high_count >= RULES['high_minimum_points']
                and now-row.first_high >= RULES['high_duration_s']-EPS):
            row.phase = 'confirmed'
            row.confirmed_at = now

    def _chain(self, now, displacement, quality):
        size = RULES['minimum_adjacent_rows']
        eligible = []
        for start in range(len(self.rows)-size+1):
            ids = list(range(start, start+size))
            rows = [self._rows[j] for j in ids]
            if not all(quality[j] and self._rows[j].phase == 'confirmed'
                       and displacement[j] >= self.low[j] for j in ids):
                continue
            onsets = [row.first_high for row in rows]
            if not all(b-a > EPS for a, b in zip(onsets, onsets[1:])):
                continue
            if now-onsets[0] < RULES['minimum_first_onset_age_s']-EPS:
                continue
            if now-onsets[0] > RULES['maximum_chain_history_s']+EPS:
                continue
            if not 0. <= now-rows[-1].confirmed_at <= RULES['recent_confirmation_s']+EPS:
                continue
            eligible.append((onsets[0], start, ids))
        # Deterministic first available image candidate, never nearest truth.
        return min(eligible)[2] if eligible else None

    def _release_episode(self, now, quality):
        if not self.episode_locked:
            return False
        for j in self.episode_rows:
            row = self._rows[j]
            if not (quality[j] and row.phase == 'low' and row.armed
                    and row.low_start is not None
                    and row.low_start > self.episode_reported_at+EPS):
                return False
        # Clear other mature front fragments too: they cannot be recycled as a
        # new event immediately after the reported front's return to baseline.
        self._rows = [_Row() for _ in self.rows]
        self.episode_locked = False
        self.episode_rows = None
        self.episode_reported_at = None
        return True

    def update(self, time_s, displacement, quality, running=None):
        displacement = np.asarray(displacement, dtype=np.float64)
        q = np.asarray(quality)
        if displacement.shape != self.rows.shape or q.shape != self.rows.shape:
            raise ValueError('displacement and quality must match original rows')
        if q.dtype.kind not in 'biuf':
            raise ValueError('quality must be bool or numeric 0/1')
        if running is not None and not isinstance(running, (bool, np.bool_)):
            raise ValueError('running must be True, False or None')
        effective = np.isfinite(q) & (q == 1) & np.isfinite(displacement) & np.isfinite(self.high)
        now = float(time_s)
        if not math.isfinite(now):
            self.reset()
            return self._snapshot(None, displacement, effective, running,
                                  'reset', 'nonfinite_time', [])
        reset_reason = None
        if self.last_time is not None:
            if now <= self.last_time:
                reset_reason = 'nonincreasing_time_reset'
            elif now-self.last_time > RULES['maximum_observation_gap_s']+EPS:
                reset_reason = 'frame_gap_reset'
        if reset_reason:
            self.reset()
        self.last_time = now
        if running is not None and not bool(running):
            self.reset()
            self.last_time = now
            return self._snapshot(now, displacement, effective, running,
                                  'stopped_reset', 'explicit_stop_no_event_clock', [])
        for j in range(len(self.rows)):
            self._observe_row(j, now, displacement[j], bool(effective[j]))
        if self._release_episode(now, effective):
            return self._snapshot(now, displacement, effective, running,
                                  'episode_rearmed', 'reported_rows_returned_to_stable_low', [])
        new = []
        if not self.episode_locked:
            ids = self._chain(now, displacement, effective)
            if ids is not None:
                first = self._rows[ids[0]]
                provenance = [dict(index=j, row_y=float(self.rows[j]),
                    high_threshold_px=float(self.high[j]), low_threshold_px=float(self.low[j]),
                    first_high_s=self._rows[j].first_high,
                    last_low_s=self._rows[j].last_low,
                    confirmed_at_s=self._rows[j].confirmed_at,
                    current_displacement_px=float(displacement[j]), current_quality=True)
                    for j in ids]
                self._candidate_counter += 1
                new.append(dict(candidate_id=f'side_expansion_{self._candidate_counter:04d}',
                    kind=RULES['candidate_kind'], estimated_onset=first.first_high,
                    first_available=now, delay=now-first.first_high,
                    time_units='seconds_on_input_capture_axis',
                    lastlow_firsthigh_interval=[first.last_low, first.first_high],
                    rows=provenance, spatial_direction='increasing_canonical_image_row_not_world_axis',
                    material_identity_confirmed=False, new_layer_confirmed=False,
                    mechanical_revolution_confirmed=False, actuator_command=None,
                    ambiguity='a_long_period_wave_can_share_the_same_finite_prefix'))
                self.episode_locked = True
                self.episode_rows = ids
                self.episode_reported_at = now
        state = ('candidate_emitted' if new else 'episode_locked' if self.episode_locked else
                 'armed_waiting_for_ordered_front' if any(row.armed for row in self._rows)
                 else 'waiting_for_stable_direct_low')
        return self._snapshot(now, displacement, effective, running, state,
                              reset_reason or 'past_only_image_candidate_rules', new)
