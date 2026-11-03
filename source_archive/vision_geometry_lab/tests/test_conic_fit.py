"""Known-conic geometry, occlusion and model-rejection regressions."""
import json
import math
import unittest

import numpy as np

from vision_geometry_lab.conic_fit import fit_conic


def ellipse_scene(start=195., end=345., angle=.2, noise=.4, gap=None, outliers=False):
    t = np.linspace(np.deg2rad(start), np.deg2rad(end), 450)
    local = np.column_stack([900*np.cos(t), 260*np.sin(t)])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = local @ rotation.T + [2000., 1800.]
    points += np.random.default_rng(42).normal(0, noise, points.shape)
    keep = np.ones(len(points), bool)
    if gap:
        keep &= (t < np.deg2rad(gap[0])) | (t > np.deg2rad(gap[1]))
    indices = np.flatnonzero(keep)
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1)
    points = points[keep]
    if outliers:
        chosen = np.arange(11, len(points), 11)
        points[chosen] += np.random.default_rng(5).normal(0, 35, (len(chosen), 2))
    segments, offset = [], 0
    for run in runs:
        segments.append(points[offset:offset+len(run)].tolist())
        offset += len(run)
    return dict(valid=True, points=points.tolist(), segments=segments), rotation


class ConicFitTests(unittest.TestCase):
    def test_known_ellipse_and_implicit_equation(self):
        arc, _ = ellipse_scene()
        result = fit_conic(arc, (3000, 4096))
        self.assertTrue(result["valid"], result)
        np.testing.assert_allclose(result["center"], [2000, 1800], atol=2.)
        np.testing.assert_allclose(result["semi_axes"], [900, 260], atol=2.)
        self.assertLess(abs(result["angle_rad"]-.2), .003)
        self.assertLess(result["residual_px"], .7)
        points = np.asarray(result["model_points"])
        a, b, c, d, e, f = result["implicit"]
        values = a*points[:, 0]**2 + b*points[:, 0]*points[:, 1] + c*points[:, 1]**2 + d*points[:, 0] + e*points[:, 1] + f
        self.assertLess(np.max(np.abs(values)), 2e-6)
        self.assertLess(result["observed_angular_span_deg"], 170.)
        self.assertGreaterEqual(len(result["sensitivity_models"]), 6)
        self.assertEqual(len(result["model_points"]), len(result["model_sensitivity_px"]))
        json.dumps(result, allow_nan=False)

    def test_occlusion_outliers_are_modeled_not_observed(self):
        arc, _ = ellipse_scene(gap=(256., 287.), outliers=True)
        result = fit_conic(arc)
        self.assertTrue(result["valid"], result)
        np.testing.assert_allclose(result["center"], [2000, 1800], atol=5.)
        self.assertFalse(result["model_points_are_observations"])
        self.assertGreater(len(result["modeled_segments"]), 0)
        self.assertGreater(len(result["outlier_points"]), 10)
        angles = np.asarray(result["model_angles_rad"])
        central_gap = (angles > np.deg2rad(260)) & (angles < np.deg2rad(283))
        self.assertTrue(np.any(central_gap))
        self.assertFalse(np.any(np.asarray(result["model_supported"])[central_gap]))

    def test_rotated_model_is_equivalent(self):
        arc, _ = ellipse_scene(angle=1.6)
        result = fit_conic(arc)
        self.assertTrue(result["valid"], result)
        self.assertLess(abs(result["angle_rad"]-1.6), .003)
        np.testing.assert_allclose(result["semi_axes"], [900, 260], atol=2.)

    def test_short_arc_is_not_forced_to_an_ellipse(self):
        arc, _ = ellipse_scene(start=258, end=283, noise=.5)
        result = fit_conic(arc)
        self.assertFalse(result["valid"], result)
        self.assertEqual(result["model_points"], [])

    def test_straight_support_is_invalid(self):
        x = np.linspace(0., 1000., 300)
        points = np.column_stack([x, 400 + .2*x])
        result = fit_conic(dict(valid=True, points=points.tolist()))
        self.assertFalse(result["valid"], result)

    def test_circle_angle_gauge_is_not_geometric_degeneracy(self):
        angle = np.linspace(np.deg2rad(190), np.deg2rad(350), 300)
        points = np.column_stack([500*np.cos(angle), 500*np.sin(angle)]) + [900, 900]
        result = fit_conic(dict(valid=True, points=points.tolist(), segments=[points.tolist()]))
        self.assertTrue(result["valid"], result)
        np.testing.assert_allclose(result["semi_axes"], [500, 500], atol=.01)

    def test_bad_inputs(self):
        self.assertFalse(fit_conic(None)["valid"])
        self.assertFalse(fit_conic(dict(valid=False))["valid"])
        self.assertFalse(fit_conic(dict(valid=True, points=[[0, 0]]*100))["valid"])

    def test_supported_arc_interval_does_not_show_invisible_bottom(self):
        arc, _ = ellipse_scene(angle=0., noise=.1)
        result = fit_conic(arc)
        self.assertTrue(result["valid"], result)
        points = np.asarray(result["model_points"])
        self.assertLess(float(points[:, 1].max()), 1800.)
        self.assertLess(result["arc_interval_rad"][1]-result["arc_interval_rad"][0], math.pi)

    def test_exclusion_cannot_be_restored_by_angular_support_padding(self):
        arc, _ = ellipse_scene(angle=0.)
        arc["exclusion_polygons"] = [[[1970, 1400], [2040, 1400], [2040, 1600], [1970, 1600]]]
        result = fit_conic(arc)
        self.assertTrue(result["valid"], result)
        points = np.asarray(result["model_points"])
        excluded = (points[:, 0] >= 1970) & (points[:, 0] <= 2040) & (points[:, 1] >= 1400) & (points[:, 1] <= 1600)
        self.assertTrue(np.any(excluded))
        self.assertFalse(np.any(np.asarray(result["model_supported"])[excluded]))

    def test_empty_occluded_angular_blocks_are_not_failed_refits(self):
        # Three of eight sectors contain no observations by construction. The
        # five populated sectors still constrain this known ellipse; an empty
        # masked sector must not count as a failed leave-block-out experiment.
        arc, _ = ellipse_scene(gap=(245., 310.), noise=.15)
        result = fit_conic(arc)
        self.assertTrue(result["valid"], result)
        diagnostics = result["diagnostics"]
        self.assertGreaterEqual(diagnostics["leave_block_out_empty_blocks"], 2)
        self.assertGreaterEqual(diagnostics["leave_block_out_attempted_blocks"], 4)
        self.assertLess(diagnostics["leave_block_out_attempted_blocks"], 6)
        self.assertEqual(diagnostics["leave_block_out_failed_blocks"], 0)
        self.assertEqual(diagnostics["leave_block_out_success_fraction"], 1.)
        self.assertEqual(diagnostics["leave_block_out_fit_count"], diagnostics["leave_block_out_attempted_blocks"])
        np.testing.assert_allclose(result["semi_axes"], [900, 260], atol=2.)

    def test_explicit_segment_selection_is_used_without_raw_fallback(self):
        arc, _ = ellipse_scene()
        support = list(arc["points"])
        arc["conic_support_points"] = support
        arc["conic_support_segments"] = [support]
        arc["points"] = [[p[0], p[1]+100*np.sin(p[0]/20)] for p in support]
        result = fit_conic(arc)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["diagnostics"]["support_source"], "paint_verified_segmented_support")
        arc["conic_support_points"] = []
        rejected = fit_conic(arc)
        self.assertFalse(rejected["valid"])


if __name__ == "__main__":
    unittest.main()
