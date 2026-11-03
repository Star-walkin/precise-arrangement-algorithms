"""V13 only synthetic patches: no field files, labels, scores or model access."""
import json
import unittest

import numpy as np

from vision_geometry_lab.v13_dark_texture_features import PARAMETERS, analyze_patch


def _periodic(rows=129, columns=33, period=12., phase=0., right_offset=0., right_period=None):
    t = np.arange(rows)
    left = 100+35*np.sin(2*np.pi*t/period)
    right = 100+right_offset+35*np.sin(2*np.pi*t/(right_period or period)+phase)
    patch = np.zeros((rows, columns), float)
    patch[:, :columns//2] = left[:, None]
    patch[:, columns//2:] = right[:, None]
    return np.clip(np.round(patch), 0, 255).astype(np.uint8)


class DarkTextureFeaturesV13Tests(unittest.TestCase):
    def test_same_period_has_measured_period_and_no_boundary_verdict(self):
        output = analyze_patch(_periodic(), .5, 1.25)
        self.assertTrue(output['available'])
        for side in ('left', 'right'):
            feature = output['tangent_texture'][side]
            self.assertTrue(feature['available'])
            self.assertEqual(feature['peak_lag_samples'], 12)
            self.assertEqual(feature['peak_period_px'], 15.)
            self.assertGreaterEqual(feature['observed_periods'], 3.)
            self.assertGreater(feature['peak_correlation'], .99)
        self.assertAlmostEqual(output['cross_boundary_texture']['zero_lag_correlation'], 1.)
        self.assertNotIn('boundary_valid', output)
        self.assertFalse(output['parameters']['combined_boundary_score'])
        json.dumps(output, allow_nan=False)
        self.assertEqual(json.loads(json.dumps(PARAMETERS)), PARAMETERS)

    def test_phase_shift_does_not_mean_different_ropes(self):
        output = analyze_patch(_periodic(phase=np.pi/2), 1., .75)
        cross = output['cross_boundary_texture']
        self.assertTrue(cross['available'])
        self.assertLess(abs(cross['zero_lag_correlation']), .05)
        self.assertGreater(cross['maximum_shifted_correlation'], .99)
        self.assertEqual(abs(cross['best_lag_samples']) % 12, 3)
        self.assertGreaterEqual(cross['best_overlap_samples'], cross['minimum_overlap_samples'])

    def test_brightness_offset_preserves_texture_but_reports_signed_transition(self):
        patch = _periodic(right_offset=35.)
        forward = analyze_patch(patch, .5, 1.)
        reverse = analyze_patch(patch[:, ::-1], .5, 1.)
        self.assertGreater(forward['transition']['signed_gradient'], 0)
        self.assertLess(reverse['transition']['signed_gradient'], 0)
        self.assertAlmostEqual(forward['transition']['absolute_gradient'], reverse['transition']['absolute_gradient'])
        self.assertAlmostEqual(forward['transition']['signed_side_intensity_change'], 35.)
        self.assertGreater(forward['cross_boundary_texture']['zero_lag_correlation'], .999)

    def test_different_frequencies_can_be_low_correlation_without_boundary_claim(self):
        output = analyze_patch(_periodic(period=8., right_period=11.), 1., 1.)
        self.assertTrue(output['tangent_texture']['left']['available'])
        self.assertTrue(output['tangent_texture']['right']['available'])
        self.assertLess(output['cross_boundary_texture']['maximum_shifted_correlation'], .2)
        self.assertTrue(any('does not prove a boundary' in text for text in output['ambiguities']))

    def test_flat_patch_is_unknown_texture_not_a_negative_boundary_example(self):
        output = analyze_patch(np.full((40, 15), 80, np.uint8), 1., 1.)
        self.assertTrue(output['available'])
        self.assertEqual(output['transition']['absolute_gradient'], 0.)
        self.assertIsNone(output['transition']['relative_to_patch_mean'])
        self.assertFalse(output['dark_valley']['available'])
        self.assertFalse(output['tangent_texture']['left']['available'])
        self.assertFalse(output['tangent_texture']['right']['available'])
        self.assertFalse(output['cross_boundary_texture']['available'])
        self.assertIsNone(output['cross_boundary_texture']['maximum_shifted_correlation'])

    def test_nonperiodic_signal_and_insufficient_cycles_remain_unknown(self):
        rng = np.random.default_rng(102)
        profile = np.clip(100+20*rng.standard_normal(129), 0, 255).astype(np.uint8)
        patch = np.repeat(profile[:, None], 33, axis=1)
        random_output = analyze_patch(patch, 1., 1.)
        self.assertFalse(random_output['tangent_texture']['left']['available'])
        slow = analyze_patch(_periodic(period=80.), 1., 1.)
        self.assertFalse(slow['tangent_texture']['left']['available'])
        self.assertEqual(slow['tangent_texture']['left']['lag_search_samples'], [2, 42])

    def test_forty_by_fifteen_patch_supports_five_short_periods(self):
        output = analyze_patch(_periodic(rows=40, columns=15, period=8.), .8, 1.5)
        self.assertTrue(output['tangent_texture']['left']['available'])
        self.assertEqual(output['tangent_texture']['left']['peak_lag_samples'], 8)
        self.assertEqual(output['tangent_texture']['left']['observed_periods'], 39/8)

    def test_weak_period_retains_raw_peak_but_is_unknown_for_supported_comparison(self):
        t = np.arange(129)
        profile = np.round(100+np.sin(2*np.pi*t/12)).astype(np.uint8)
        output = analyze_patch(np.repeat(profile[:, None], 33, axis=1), 1., 1.)
        feature = output['tangent_texture']['left']
        self.assertFalse(feature['available'])
        self.assertIsNone(feature['peak_correlation'])
        self.assertGreater(feature['raw_peak_correlation'], .99)
        self.assertIsNotNone(feature['largest_tested_correlation'])

    def test_side_averaging_cancellation_is_reported_not_recovered_using_gt(self):
        t, x = np.indices((129, 33))
        patch = np.clip(np.round(120+30*np.sin(2*np.pi*t/12.+np.pi*x)), 0, 255).astype(np.uint8)
        output = analyze_patch(patch, 1., 1.)
        self.assertLess(output['tangent_texture']['left']['side_averaging_retained_rms_ratio'], .1)
        self.assertFalse(output['tangent_texture']['left']['available'])

    def test_dark_valley_offset_uses_supported_samples_and_native_units(self):
        x = np.arange(33)
        values = 130-50*np.exp(-.5*((x-19.)/1.7)**2)
        patch = np.repeat(np.round(values)[None], 129, axis=0).astype(np.uint8)
        output = analyze_patch(patch, .8, 1.)
        self.assertTrue(output['dark_valley']['available'])
        self.assertAlmostEqual(output['dark_valley']['offset_samples'], 3.)
        self.assertAlmostEqual(output['dark_valley']['offset_px'], 2.4)
        ramp = np.repeat(np.arange(33, dtype=np.uint8)[None]*5, 129, axis=0)
        self.assertFalse(analyze_patch(ramp, 1., 1.)['dark_valley']['available'])

    def test_no_padding_filter_or_artificial_endpoint_valley(self):
        patch = np.full((129, 33), 80, np.uint8)
        patch[:, 0] = 0
        output = analyze_patch(patch, 1., 1.)
        self.assertFalse(output['dark_valley']['available'])
        self.assertEqual(output['dark_valley']['profile_columns'][0], 1.)
        self.assertEqual(output['dark_valley']['profile_columns'][-1], 31.)
        self.assertFalse(output['geometry']['input_support_checked_by_module'])

    def test_invalid_and_too_small_inputs(self):
        output = analyze_patch(np.zeros((39, 15), np.uint8), 1., 1.)
        self.assertFalse(output['available'])
        self.assertIsNone(output['transition'])
        for normal, tangent in ((0, 1), (1, np.nan), (-1, 1)):
            with self.assertRaises(ValueError):
                analyze_patch(np.zeros((40, 15), np.uint8), normal, tangent)
        with self.assertRaises(ValueError):
            analyze_patch(np.zeros((40, 15), float), 1., 1.)


if __name__ == '__main__':
    unittest.main()
