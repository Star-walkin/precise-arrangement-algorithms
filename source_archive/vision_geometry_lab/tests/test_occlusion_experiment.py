"""Leakage, causal-time, rejection and known-truth checks for offline ablation."""
import copy
import json
import math
import unittest

import numpy as np

from vision_geometry_lab.occlusion_experiment import (
    Protocol, ambiguous_prior_counterexample, curve_points, nearest_curve_distance, observation_quality,
    predict_case, select_history, split_support, synthetic_experiment, update_prior)


class OcclusionExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.synthetic = synthetic_experiment()
        cls.prior = cls.synthetic["history"]
        cls.visible = np.asarray(cls.synthetic["case"]["visible_points"])

    def test_known_truth_and_small_causal_update(self):
        result = self.synthetic
        update = result["case"]["prior_update"]
        self.assertTrue(result["history"]["quality"]["coverage_eligible"])
        self.assertTrue(result["history"]["model"]["valid"])
        self.assertTrue(update["valid"], update)
        self.assertLess(result["ground_truth_scores"]["prior_updated"]["p95_px"], 2.)
        self.assertLess(np.linalg.norm(np.asarray(update["translation_px"]) - [4, -3]), 3.)
        self.assertLess(abs(update["uniform_scale"] - 1.005), .004)
        json.dumps(result, allow_nan=False)

    def test_future_and_current_priors_rejected(self):
        for time_s in (-1., 0.):
            result = update_prior(self.visible, self.prior, time_s)
            self.assertFalse(result["valid"])
            self.assertEqual(result["reason"], "prior_is_not_strictly_earlier")
        earlier = dict(self.prior, time_s=-2.)
        later = dict(self.prior, time_s=2.)
        self.assertIs(select_history([later, self.prior, earlier], 0.), earlier)
        self.assertIsNone(select_history([later, self.prior], 0.))

    def test_stale_prior_and_absent_history_rejected(self):
        result = update_prior(self.visible, self.prior, 5.01)
        self.assertEqual(result["reason"], "prior_is_stale")
        self.assertFalse(result["valid"])
        self.assertEqual(update_prior(self.visible, None, 1.)["reason"], "no_eligible_earlier_shape")

    def test_visibly_wrong_prior_is_rejected(self):
        wrong = copy.deepcopy(self.prior)
        wrong["model"]["center"][1] += 50.
        result = update_prior(self.visible, wrong, 1.)
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "prior_inconsistent_with_current_visible_arc")

    def test_clear_definition_does_not_depend_on_fit_success(self):
        x = np.linspace(500., 3300., 300)
        straight = np.column_stack([x, 1800 + x * .03])
        quality = observation_quality(straight, (3000, 4096))
        self.assertTrue(quality["coverage_eligible"])
        bad = dict(self.prior, quality=quality, model=dict(valid=False))
        self.assertIsNone(select_history([bad], 1.))
        self.assertFalse(update_prior(self.visible, bad, 1.)["valid"])

    def test_holdout_values_cannot_change_prediction(self):
        # Perturb only withheld y, add thousands of held-out points. Neither
        # pure-left fit nor prior selection/update may observe those changes.
        visible = self.visible.tolist()
        hidden = [[2500. + i, 1600.] for i in range(80)]
        one = dict(valid=True, points=visible + hidden, segments=[visible, hidden])
        corrupted = [[2500. + i % 100, -10000. + i] for i in range(500)]
        two = dict(valid=True, points=visible + corrupted, segments=[visible, corrupted],
                   residual_px=1000., confidence=0., diagnostics={"should_never_leak": True})
        first = predict_case(one, (3000, 4096), 1., .5, [self.prior])
        second = predict_case(two, (3000, 4096), 1., .5, [self.prior])
        self.assertEqual(first["free_left_model"], second["free_left_model"])
        self.assertEqual(first["prior_update"], second["prior_update"])
        self.assertNotEqual(first["scores"], second["scores"])

    def test_holdout_cannot_be_restored_through_segments(self):
        points = [[1800., 2.], [2200., 1.], [1700., 3.], [1600., 4.]]
        left, right = split_support(dict(points=points, segments=[points]), 2048)
        self.assertEqual(len(right), 1)
        self.assertTrue(all(p[0] < 2048 for segment in left["segments"] for p in segment))
        self.assertNotIn("residual_px", left)

    def test_far_scoring_is_global_and_clipped_scoring_is_distinct(self):
        circle = dict(center=[0., 0.], semi_axes=[10., 10.], angle_rad=0.)
        points = np.array([[0., 0.], [100., 0.], [-100., 0.]])
        distance = nearest_curve_distance(points, circle)
        np.testing.assert_allclose(distance, [10., 90., 90.], atol=1e-6)
        clipped = nearest_curve_distance(np.array([[-10., 0.]]), circle, [-.1, .1])
        self.assertGreater(clipped[0], 19.)

    def test_shape_ratio_angle_preserved_not_free_ellipse_refit(self):
        update = self.synthetic["case"]["prior_update"]
        self.assertEqual(update["update_degrees_of_freedom"], 3)
        old = self.prior["model"]
        model = update["model"]
        self.assertAlmostEqual(model["semi_axes"][0] / model["semi_axes"][1],
                               old["semi_axes"][0] / old["semi_axes"][1], places=10)
        self.assertAlmostEqual(model["angle_rad"], old["angle_rad"], places=12)

    def test_lookalike_wrong_history_can_pass_left_but_fail_hidden_truth(self):
        # An important limitation regression, not a request to tune gates until
        # this carefully hidden shape change becomes detectable.
        result = ambiguous_prior_counterexample()
        self.assertTrue(result["history"]["model"]["valid"])
        self.assertTrue(result["case"]["prior_update"]["valid"])
        self.assertLess(result["case"]["prior_update"]["after_update_p95_px"], 1.)
        self.assertGreater(result["ground_truth_score"]["p95_px"], 100.)
        self.assertFalse(result["case"]["prior_update"]["hidden_geometry_verified"])


if __name__ == "__main__":
    unittest.main()
