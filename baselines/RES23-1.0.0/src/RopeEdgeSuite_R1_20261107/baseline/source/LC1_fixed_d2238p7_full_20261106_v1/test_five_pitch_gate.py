"""Temporal contracts for current valid-N gating; no real media required."""
from copy import deepcopy
import json
import math
import unittest

from five_pitch_gate import FivePitchGate


def current_surface(n, **changes):
    result = dict(count_valid=True, continuous_pitch_distance=n,
                  continuous_pitch_distance_candidate=n,
                  edge_current_frame_confirmed=True, measurement_x=2238.7,
                  diagnostics=dict(count=dict(interval_y=[100., 300.])))
    result.update(changes)
    return result


class FivePitchGateTests(unittest.TestCase):
    def setUp(self):
        self.gate = FivePitchGate()

    def update(self, t, n, **kwargs):
        surface = kwargs.pop('surface', current_surface(n))
        return self.gate.update(t, surface, s1_valid=kwargs.pop('s1_valid', True),
                                intersection_y_px=kwargs.pop('intersection_y_px', 300.), **kwargs)

    def test_invalid_candidate_never_arms(self):
        r = self.update(0., None, surface=current_surface(None, count_valid=False,
                        continuous_pitch_distance_candidate=2.))
        self.assertEqual(r['current_region'], 'unknown')
        self.assertIsNone(r['current_N'])
        self.assertFalse(r['armed'])
        self.assertFalse(r['emit_distance_estimate'])

    def test_exact_threshold_is_inside_but_next_float_is_outside(self):
        outside = self.update(0., math.nextafter(5., math.inf))
        self.assertEqual(outside['current_region'], 'outside')
        self.assertFalse(outside['armed'])
        inside = self.update(1., 5.)
        self.assertEqual(inside['current_region'], 'inside')
        self.assertTrue(inside['first_inside_confirmed_this_frame'])
        self.assertEqual(inside['entry_observation_interval_s'], [0., 1.])
        self.assertEqual(inside['current_N'], 5.)
        self.assertIsNone(inside['integer_remaining_rope_count'])

    def test_unknown_preserves_history_without_filling_current_region(self):
        self.update(0., 4.25, frame_index=0)
        r = self.update(.1, None, surface=None, frame_index=1)
        self.assertEqual(r['current_region'], 'unknown')
        self.assertIsNone(r['current_distance_pitch'])
        self.assertTrue(r['armed'])
        self.assertEqual(r['last_known_region'], 'inside')
        self.assertEqual(r['last_known_N'], 4.25)
        self.assertAlmostEqual(r['current_unknown_age_s'], .1)
        self.assertFalse(r['emit_distance_estimate'])
        r = self.update(1.5, None, surface=None, frame_index=15)
        self.assertEqual(r['unknown_run_frames'], 2)
        self.assertAlmostEqual(r['current_unknown_age_s'], 1.5)

    def test_valid_outside_remains_outside_after_arming(self):
        self.update(0., 4.)
        r = self.update(1., 6.)
        self.assertEqual(r['current_region'], 'outside')
        self.assertEqual(r['current_N'], 6.)
        self.assertTrue(r['armed'])
        self.assertFalse(r['emit_distance_estimate'])
        self.assertEqual(r['last_known_region'], 'outside')

    def test_gap_entry_brackets_observations_not_actual_crossing_time(self):
        self.update(0., 6.)
        self.update(.1, None, surface=None)
        r = self.update(2., 4.)
        self.assertTrue(r['inside_confirmation_event'])
        self.assertEqual(r['entry_observation_interval_s'], [0., 2.])
        self.assertTrue(r['entry_time_is_confirmation_not_exact_physical_crossing'])
        r = self.update(2.1, 3.9)
        self.assertFalse(r['inside_confirmation_event'])
        self.assertFalse(r['first_inside_confirmed_this_frame'])

    def test_current_S1_and_intersection_required_even_if_F3_claims_valid(self):
        for i, values in enumerate((dict(s1_valid=False), dict(s1_valid='true'),
                                    dict(intersection_y_px=None), dict(intersection_y_px=float('nan')))):
            with self.subTest(values=values):
                r = self.update(float(i), 3., **values)
                self.assertEqual(r['current_region'], 'unknown')
                self.assertFalse(r['armed'])

    def test_count_flag_and_N_require_literal_valid_finite_nonnegative_values(self):
        bad = [current_surface(3., count_valid='true'), current_surface(3., count_valid=1),
               current_surface(float('inf')), current_surface(float('nan')),
               current_surface(-.01), current_surface(True), current_surface('3')]
        for i, surface in enumerate(bad):
            with self.subTest(surface=surface):
                r = self.update(float(i), None, surface=surface)
                self.assertEqual(r['current_region'], 'unknown')
                self.assertFalse(r['armed'])

    def test_wrong_column_or_stale_active_endpoint_not_accepted(self):
        bad = [current_surface(3., measurement_x=2237.7),
               current_surface(3., measurement_x=None),
               current_surface(3., diagnostics={}),
               current_surface(3., diagnostics=dict(count=dict(interval_y=[100., 301.]))),
               current_surface(3., edge_current_frame_confirmed=False)]
        for i, surface in enumerate(bad):
            with self.subTest(i=i):
                r = self.update(float(i), None, surface=surface)
                self.assertEqual(r['current_region'], 'unknown')
                self.assertFalse(r['emit_distance_estimate'])

    def test_groove_lower_bound_does_not_establish_inside_upper_bound(self):
        r = self.update(0., None, surface=current_surface(None, count_valid=False,
                         observed_groove_count_lower_bound=2,
                         continuous_pitch_distance_candidate=2.4))
        self.assertEqual(r['current_region'], 'unknown')
        self.assertFalse(r['armed'])

    def test_explicit_reset_starts_new_epoch_and_clears_latch(self):
        self.update(10., 4., frame_index=100)
        r = self.update(0., 7., frame_index=0, reset=True)
        self.assertEqual(r['epoch'], 1)
        self.assertFalse(r['armed'])
        self.assertIsNone(r['first_inside_confirmation_time_s'])
        self.assertEqual(r['current_region'], 'outside')

    def test_repeated_backwards_time_or_frame_does_not_mutate_state(self):
        self.update(1., 7., frame_index=10)
        for t, index in ((1., 11), (.5, 11), (2., 10), (2., 9), (2., True), (2., 10.5)):
            with self.subTest(t=t, index=index), self.assertRaises(ValueError):
                self.update(t, 2., frame_index=index)
        r = self.update(2., 6., frame_index=11)
        self.assertFalse(r['armed'])
        self.assertEqual(r['last_known_N'], 6.)

    def test_prefix_is_causal_and_json_output_has_no_nonfinite_history(self):
        sequence = [current_surface(7.), current_surface(None, count_valid=False), current_surface(4.3)]
        a = FivePitchGate(); b = FivePitchGate()
        prefix = [a.update(i/10., s, s1_valid=True, intersection_y_px=300., frame_index=i)
                  for i, s in enumerate(sequence)]
        for i, surface in enumerate(sequence):
            self.assertEqual(prefix[i], b.update(i/10., surface, s1_valid=True, intersection_y_px=300., frame_index=i))
        frozen_prefix = deepcopy(prefix)
        a.update(.3, current_surface(8.), s1_valid=True, intersection_y_px=300., frame_index=3)
        self.assertEqual(prefix, frozen_prefix)
        json.dumps(prefix, allow_nan=False)
        self.assertTrue(all(r['fixed_d_is_user_provisional_condition'] for r in prefix))
        self.assertTrue(all(not r['future_frames_used'] for r in prefix))


if __name__ == '__main__':
    unittest.main()
