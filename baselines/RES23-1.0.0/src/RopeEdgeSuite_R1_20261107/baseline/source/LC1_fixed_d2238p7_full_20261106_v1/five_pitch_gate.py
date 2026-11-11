"""Causal current-distance gate at the user's provisional fixed image column.

Only CURRENT valid original-F3 continuous distance <= threshold triggers inside.
Invalid distance candidates, observed-groove LOWER bounds, and history never
turn an unknown current frame into inside/outside. No integer phase, physical
contact, velocity, future horizon or automatically calibrated d is asserted.

Native F3 bindings: measurement_x must match fixed_d_original_px, and
diagnostics.count.interval_y[1] must match the current S1 line intersection.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real


def _finite_real(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


class FivePitchGate:
    """One-pass state; input timestamps/indices must strictly increase.

    `armed` remembers that inside was observed at least once. It is an audit
    latch, never a substitute for the CURRENT classification. `emit_distance`
    is true only for a currently valid inside observation. Entry event time is
    first confirmation, not a claim about the physical crossing during gaps.
    """

    def __init__(self, threshold_pitch=5.0, fixed_d_original_px=2238.7,
                 binding_tolerance_px=1.e-3):
        for key, value in [('threshold_pitch', threshold_pitch),
                           ('fixed_d_original_px', fixed_d_original_px),
                           ('binding_tolerance_px', binding_tolerance_px)]:
            if not _finite_real(value) or float(value) < 0:
                raise ValueError(key+' must be finite and nonnegative')
        self.threshold = float(threshold_pitch)
        self.fixed_d = float(fixed_d_original_px)
        self.binding_tolerance = float(binding_tolerance_px)
        self.epoch = 0
        self._clear()

    def _clear(self):
        self.previous_t = None
        self.previous_index = None
        self.armed = False
        self.first_inside_confirmation_time = None
        self.first_inside_confirmation_frame = None
        self.last_known_region = None
        self.last_known_N = None
        self.last_known_time = None
        self.last_known_frame = None
        self.unknown_run_frames = 0

    def _measurement(self, surface, s1_valid, intersection_y_px):
        if s1_valid is not True:
            return None, 'current_S1_not_valid'
        if not _finite_real(intersection_y_px):
            return None, 'current_S1_intersection_unavailable_or_nonfinite'
        if not isinstance(surface, Mapping):
            return None, 'current_F3_unavailable'
        if surface.get('count_valid') is not True:
            return None, 'current_F3_distance_not_valid_candidate_ignored'
        if surface.get('edge_current_frame_confirmed') is not True:
            return None, 'current_F3_edge_not_confirmed'
        column = surface.get('measurement_x')
        if not _finite_real(column):
            return None, 'current_F3_column_binding_unavailable'
        if abs(float(column)-self.fixed_d) > self.binding_tolerance:
            return None, 'current_F3_column_does_not_match_user_fixed_d'
        diag = surface.get('diagnostics', {})
        count = diag.get('count', {}) if isinstance(diag, Mapping) else {}
        endpoints = count.get('interval_y') if isinstance(count, Mapping) else None
        if not isinstance(endpoints, (list, tuple)) or len(endpoints) != 2 or not all(_finite_real(v) for v in endpoints):
            return None, 'current_F3_active_endpoint_binding_unavailable'
        if abs(float(endpoints[1])-float(intersection_y_px)) > self.binding_tolerance:
            return None, 'current_F3_active_endpoint_does_not_match_current_S1'
        value = surface.get('continuous_pitch_distance')
        if not _finite_real(value) or float(value) < 0:
            return None, 'current_F3_continuous_distance_invalid'
        return float(value), 'current_valid_conditional_continuous_distance'

    def update(self, t, surface, *, s1_valid, intersection_y_px,
               frame_index=None, reset=False):
        """Classify one CURRENT frame; no smoothing or future buffering.

        Invalid measurements return unknown and preserve last-known audit
        state. Invalid chronology raises before mutating state. An explicit
        reset starts a new epoch and clears the armed latch and prior history.
        """
        if type(reset) is not bool:
            raise ValueError('reset must be a literal boolean')
        if not _finite_real(t):
            raise ValueError('Timestamp must be finite real')
        t = float(t)
        if frame_index is not None and (type(frame_index) is not int or frame_index < 0):
            raise ValueError('frame_index must be a nonnegative integer')
        if not reset:
            if self.previous_t is not None and t <= self.previous_t:
                raise ValueError('Timestamps must strictly increase; repeated/backward updates are not accepted')
            if frame_index is not None and self.previous_index is not None and frame_index <= self.previous_index:
                raise ValueError('Frame indices must strictly increase')
        if reset:
            self.epoch += 1
            self._clear()
        value, reason = self._measurement(surface, s1_valid, intersection_y_px)
        old_region, old_time = self.last_known_region, self.last_known_time
        first_inside = False
        inside_confirmation = False
        entry_observation_interval = None
        if value is None:
            region = 'unknown'
            self.unknown_run_frames += 1
        else:
            region = 'inside' if value <= self.threshold else 'outside'
            self.unknown_run_frames = 0
            if region == 'inside':
                inside_confirmation = old_region != 'inside'
                if inside_confirmation:
                    entry_observation_interval = [old_time if old_region == 'outside' else None, t]
                if not self.armed:
                    first_inside = True
                    self.armed = True
                    self.first_inside_confirmation_time = t
                    self.first_inside_confirmation_frame = frame_index
            self.last_known_region = region
            self.last_known_N = value
            self.last_known_time = t
            self.last_known_frame = frame_index
        self.previous_t = t
        if frame_index is not None:
            self.previous_index = frame_index
        unknown_age = t-self.last_known_time if region == 'unknown' and self.last_known_time is not None else None
        return dict(version='five_pitch_gate_1', epoch=self.epoch, frame_index=frame_index, time_s=t,
            current_region=region, current_unknown=region == 'unknown',
            current_N=value, current_distance_pitch=value,
            current_measurement_valid=value is not None, reason=reason,
            threshold_pitch=self.threshold, threshold_inclusive=True,
            armed=self.armed, armed_since_first_inside=self.armed,
            emit_distance_estimate=region == 'inside',
            first_inside_confirmed_this_frame=first_inside,
            inside_confirmation_event=inside_confirmation,
            entry_observation_interval_s=entry_observation_interval,
            entry_time_is_confirmation_not_exact_physical_crossing=True,
            first_inside_confirmation_time_s=self.first_inside_confirmation_time,
            first_inside_confirmation_frame=self.first_inside_confirmation_frame,
            last_known_region=self.last_known_region, last_known_N=self.last_known_N,
            last_known_time_s=self.last_known_time, last_known_frame=self.last_known_frame,
            current_unknown_age_s=unknown_age, unknown_run_frames=self.unknown_run_frames,
            fixed_d_original_px=self.fixed_d,
            fixed_d_is_user_provisional_condition=True,
            fixed_d_automatically_calibrated=False,
            distance_condition='user_provisional_fixed_original_d_and_current_S1_line_intersection',
            distance_semantics='current_continuous_pitch_distance_not_future_time_forecast',
            physical_rope_intersection_independently_confirmed=False,
            physical_contact_confirmed=False, integer_phase_calibrated=False,
            integer_remaining_rope_count=None, future_frames_used=False,
            invalid_candidate_or_groove_lower_bound_used=False)
