"""Analytic counterexamples for the experimental observed-turn estimator.

These tests check observation provenance, sampling behavior and equivariance.
They do not establish real-video accuracy or uncertainty coverage.
Run: python -m unittest -v test_lc1_real_path_probe.py
"""
import unittest
import numpy as np

from lc1_geometry import estimate_turn
from lc1_real_path_probe import estimate_turn_observed_anchored


def analytic_case(free_count=800, curved=True):
    d = 720.0
    xf = np.linspace(900.0, 2700.0, free_count)
    xc = np.linspace(100.0, 700.0, 40)
    free = np.column_stack((xf, -0.2 * xf + 1000.0))
    yc = -0.2 * xc + 1000.0
    if curved:
        yc = yc + 0.0015 * np.minimum(xc - d, 0.0) ** 2
    curve = np.column_stack((xc, yc))
    path = np.vstack((free, curve))
    return dict(path_xy=path, path_point_support=np.ones(len(path), bool),
                path_stage=['free'] * len(free) + ['surface'] * len(curve),
                free_line_abc=[0.2, 1.0, -1000.0],
                free_line_points_xy=free, free_side='right', pitch_px=20.0,
                connection_eligible=True)


class ObservedTurnTests(unittest.TestCase):
    def test_minority_curved_branch_is_not_trimmed_as_outliers(self):
        case = analytic_case()
        old = estimate_turn(case['path_xy'], free_side='right', pitch_px=20.0)
        self.assertFalse(old['valid'])
        self.assertEqual(old['reason'], 'straight_path')
        new = estimate_turn_observed_anchored(**case)
        self.assertTrue(new['valid'], new)
        self.assertAlmostEqual(new['x'], 720.0, places=7)
        self.assertLessEqual(new['uncertainty_px'], 10.0)

    def test_free_sampling_density_cannot_erase_curve_or_shrink_profile(self):
        results = [estimate_turn_observed_anchored(**analytic_case(n))
                   for n in (200, 800, 1600)]
        for result in results:
            self.assertTrue(result['valid'], result)
            self.assertAlmostEqual(result['x'], 720.0, places=7)
            self.assertAlmostEqual(result['uncertainty_px'],
                                   results[0]['uncertainty_px'], places=7)
            self.assertEqual(result['surface_spatial_bins'],
                             results[0]['surface_spatial_bins'])

    def test_unsupported_points_never_become_curve_observations(self):
        case = analytic_case()
        baseline = estimate_turn_observed_anchored(**case)
        # Deliberately extreme false surface/transition samples, interspersed
        # spatially across the gap. None may change the measured curve fit.
        fake = np.column_stack((np.linspace(100.0, 1500.0, 99),
                                np.full(99, -9000.0)))
        case['path_xy'] = np.vstack((case['path_xy'], fake, fake))
        case['path_point_support'] = np.r_[case['path_point_support'],
                                           np.zeros(198, bool)]
        case['path_stage'] += ['surface'] * 99 + ['transition'] * 99
        actual = estimate_turn_observed_anchored(**case)
        self.assertEqual(actual['valid'], baseline['valid'])
        for key in ('x', 'y', 'uncertainty_px', 'rmse', 'quadratic'):
            self.assertAlmostEqual(actual[key], baseline[key], places=10)
        self.assertEqual(actual['surface_point_count'], 40)
        self.assertEqual(actual['omitted_unsupported_path_points'], 198)
        self.assertFalse(actual['fit_uses_Hermite_or_unsupported_points'])

    def test_supported_hermite_transition_is_also_excluded(self):
        case = analytic_case()
        baseline = estimate_turn_observed_anchored(**case)
        fake = np.column_stack((np.linspace(701.0, 899.0, 80),
                                np.full(80, 20000.0)))
        case['path_xy'] = np.vstack((case['path_xy'], fake))
        case['path_point_support'] = np.r_[case['path_point_support'],
                                           np.ones(80, bool)]
        case['path_stage'] += ['transition'] * 80
        actual = estimate_turn_observed_anchored(**case)
        self.assertTrue(actual['valid'], actual)
        self.assertEqual(actual['omitted_H6_transition_points'], 80)
        self.assertAlmostEqual(actual['x'], baseline['x'], places=10)
        self.assertAlmostEqual(actual['uncertainty_px'],
                               baseline['uncertainty_px'], places=10)

    def test_horizontal_mirror_preserves_geometry_and_uncertainty(self):
        case = analytic_case()
        baseline = estimate_turn_observed_anchored(**case)
        reflected = dict(case)
        reflected['path_xy'] = case['path_xy'].copy()
        reflected['path_xy'][:, 0] = 4095.0 - reflected['path_xy'][:, 0]
        reflected['free_line_points_xy'] = case['free_line_points_xy'].copy()
        reflected['free_line_points_xy'][:, 0] = (
            4095.0 - reflected['free_line_points_xy'][:, 0])
        a, b, c = case['free_line_abc']
        reflected['free_line_abc'] = [-a, b, c + a * 4095.0]
        reflected['free_side'] = 'left'
        actual = estimate_turn_observed_anchored(**reflected)
        self.assertTrue(actual['valid'], actual)
        self.assertAlmostEqual(actual['x'], 4095.0 - baseline['x'], places=6)
        self.assertAlmostEqual(actual['y'], baseline['y'], places=6)
        self.assertAlmostEqual(actual['uncertainty_px'],
                               baseline['uncertainty_px'], places=6)

    def test_no_legal_H6_connection_cannot_authorize_a_turn(self):
        case = analytic_case()
        case['connection_eligible'] = False
        result = estimate_turn_observed_anchored(**case)
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'no_unique_legal_H6_curve_connection')
        self.assertIsNone(result['x'])

    def test_straight_surface_rejected_even_with_legal_topology(self):
        result = estimate_turn_observed_anchored(**analytic_case(curved=False))
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'],
                         'straight_or_unresolved_supported_surface_curvature')

    def test_missing_S1_support_cannot_claim_anchor_uncertainty(self):
        case = analytic_case()
        case['free_line_points_xy'] = []
        result = estimate_turn_observed_anchored(**case)
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'],
                         'S1_anchor_uncertainty_unavailable_or_insufficient')

    def test_support_stage_alignment_is_required(self):
        case = analytic_case()
        case['path_point_support'] = case['path_point_support'][:-1]
        result = estimate_turn_observed_anchored(**case)
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'support_stage_alignment_mismatch')

    def test_geometry_never_promotes_observation_or_physical_identity(self):
        result = estimate_turn_observed_anchored(**analytic_case())
        self.assertTrue(result['valid'], result)
        self.assertFalse(result['whole_observed_flag_promoted'])
        self.assertFalse(result['physical_contact_confirmed'])
        self.assertTrue(result['uncertainty_is_conditional'])


if __name__ == '__main__':
    unittest.main()
