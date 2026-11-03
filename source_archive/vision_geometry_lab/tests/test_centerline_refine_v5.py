"""Known image-body centers; no model weights, hardware, or pytest fixtures."""
import json
import unittest

import cv2
import numpy as np

from vision_geometry_lab.centerline_refine_v5 import refine_centerline


def _scene(*, dark=False, texture=False, variable_width=False, saturation=False):
    height, width = 260, 500
    yy, xx = np.mgrid[:height, :width]
    center = 105. + .0003 * (xx - 250.)**2
    body_width = 20. + (5. * np.sin(xx / 110.) if variable_width else 0.)
    slope = .0006 * (xx - 250.)
    distance = (yy - center) / np.sqrt(1. + slope**2)
    # Soft edge is the known apparent width, not a jagged rounded pixel mask.
    amount = .5 * (np.tanh((distance + body_width / 2) / .8) -
                   np.tanh((distance - body_width / 2) / .8))
    background, interior = (190., 55.) if dark else (35., 185.)
    if saturation:
        interior = 255.
    image = background + (interior - background) * amount
    if texture:
        # Diagonal bright/dark braid facets inside the same body envelope.
        image += 32. * np.sin(xx * .75 + distance * .9) * amount
    image = cv2.GaussianBlur(image.astype(np.float32), (0, 0), .4)
    image = np.repeat(np.clip(image, 0, 255)[..., None], 3, axis=2).astype(np.uint8)
    seed_x = np.linspace(70., 430., 19)
    true_y = 105. + .0003 * (seed_x - 250.)**2
    seeds = np.column_stack([seed_x, true_y + 2.4])
    return image, seeds


class CenterlineRefineV5Tests(unittest.TestCase):
    def test_known_bright_curved_body_subpixel_center(self):
        image, seeds = _scene()
        result = refine_centerline(image, seeds, expected_width_px=20.)
        self.assertEqual(len(result["records"]), len(seeds))
        self.assertGreaterEqual(sum(r["valid"] for r in result["records"]), 17)
        refined = np.asarray([r["refined_xy"] for r in result["records"] if r["valid"]])
        error = np.abs(refined[:, 1] - (105. + .0003 * (refined[:, 0] - 250.)**2))
        self.assertLess(float(np.percentile(error, 95)), .6)
        self.assertLess(max(abs(r["width_px"] - 20.) for r in result["records"] if r["valid"]), .8)
        self.assertFalse(result["identity_confirmed"])
        json.dumps(result, allow_nan=False)

    def test_periodic_strands_are_not_the_refined_center(self):
        image, seeds = _scene(texture=True)
        result = refine_centerline(image, seeds, expected_width_px=20.)
        good = [r for r in result["records"] if r["valid"]]
        self.assertGreaterEqual(len(good), 15)
        points = np.asarray([r["refined_xy"] for r in good])
        error = abs(points[:, 1] - (105. + .0003 * (points[:, 0] - 250.)**2))
        self.assertLess(float(np.percentile(error, 95)), .8)

    def test_dark_body_and_local_width_change(self):
        image, seeds = _scene(dark=True, variable_width=True)
        result = refine_centerline(image, seeds, expected_width_px=20.)
        good = [r for r in result["records"] if r["valid"]]
        self.assertGreaterEqual(len(good), 17)
        widths = np.asarray([r["width_px"] for r in good])
        self.assertGreater(float(np.ptp(widths)), 7.)
        points = np.asarray([r["refined_xy"] for r in good])
        error = abs(points[:, 1] - (105. + .0003 * (points[:, 0] - 250.)**2))
        self.assertLess(float(np.percentile(error, 95)), .6)

    def test_blank_and_single_edge_are_not_completed(self):
        seeds = [[90., 100.], [150., 100.], [210., 100.]]
        blank = np.full((220, 300, 3), 80, np.uint8)
        single = blank.copy()
        single[100:] = 190
        for image in (blank, single):
            result = refine_centerline(image, seeds, expected_width_px=20.)
            self.assertFalse(result["valid"])
            for record in result["records"]:
                self.assertIsNone(record["refined_xy"])
                self.assertIsNone(record["edge_a_xy"])
                self.assertIsNone(record["edge_b_xy"])

    def test_narrow_dark_groove_not_whole_rope_center(self):
        image = np.full((220, 300, 3), 180, np.uint8)
        image[98:103] = 35
        result = refine_centerline(image, [[80., 100.], [150., 100.], [220., 100.]], expected_width_px=20.)
        self.assertFalse(result["valid"])
        self.assertTrue(all(r["reason"] == "no_body_width_pair_possible_internal_groove" for r in result["records"]))

    def test_two_body_pairs_are_ambiguous_not_nearest_peak(self):
        image = np.full((240, 320, 3), 30, np.uint8)
        image[80:100] = 185
        image[112:132] = 185
        seeds = [[90., 105.5], [160., 105.5], [230., 105.5]]
        result = refine_centerline(image, seeds, expected_width_px=20., max_shift_px=20.)
        self.assertFalse(result["valid"])
        self.assertTrue(all(r["reason"] == "multiple_comparable_body_pairs_identity_ambiguous" for r in result["records"]))

    def test_overexposed_body_is_not_called_precise(self):
        image, seeds = _scene(saturation=True)
        result = refine_centerline(image, seeds, expected_width_px=20.)
        self.assertFalse(result["valid"])
        self.assertTrue(any(r["reason"] == "overexposed_body_edges_not_precise" for r in result["records"]))

    def test_erased_interval_stays_invalid_without_interpolation(self):
        image, seeds = _scene()
        image[:, 215:285] = 35
        result = refine_centerline(image, seeds, expected_width_px=20.)
        at_middle = int(np.argmin(abs(seeds[:, 0] - 250.)))
        self.assertFalse(result["records"][at_middle]["valid"])
        self.assertIsNone(result["records"][at_middle]["refined_xy"])
        self.assertEqual(result["diagnostics"]["interpolated_points"], 0)
        self.assertTrue(result["valid"])

    def test_no_width_prior_cannot_resolve_dark_body_vs_groove(self):
        image, seeds = _scene(dark=True)
        result = refine_centerline(image, seeds)
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "body_width_prior_required")

    def test_semantically_wrong_same_width_groove_is_unidentifiable(self):
        # Identical image evidence can describe a dark rope OR a dark groove.
        # Width-compatible edges are only a candidate, never confirmed identity.
        image, seeds = _scene(dark=True)
        result = refine_centerline(image, seeds, expected_width_px=20.)
        self.assertTrue(result["valid"])
        self.assertFalse(result["identity_confirmed"])
        self.assertTrue(all(not r["identity_confirmed"] for r in result["records"]))

    def test_input_errors_and_local_width_priors(self):
        image, seeds = _scene()
        for bad in (None, [1, 2], [[np.nan, 0]]):
            self.assertFalse(refine_centerline(image, bad, expected_width_px=20.)["valid"])
        self.assertFalse(refine_centerline(image, [[100, 100]], expected_width_px=20.)["valid"])
        self.assertFalse(refine_centerline(image, seeds, expected_width_px=-2.)["valid"])
        self.assertFalse(refine_centerline(image, seeds, expected_width_px=20., max_shift_px=0.)["valid"])
        result = refine_centerline(image, seeds, expected_width_px=np.full(len(seeds), 20.))
        self.assertTrue(result["valid"])

    def test_rotated_polyline_uses_local_normals(self):
        image, seeds = _scene(texture=True)
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotated_seeds = np.column_stack([image.shape[0] - 1 - seeds[:, 1], seeds[:, 0]])
        result = refine_centerline(rotated, rotated_seeds, expected_width_px=20.)
        points = np.asarray([r["refined_xy"] for r in result["records"] if r["valid"]])
        self.assertGreaterEqual(len(points), 15)
        original_x, original_y = points[:, 1], image.shape[0] - 1 - points[:, 0]
        error = abs(original_y - (105. + .0003 * (original_x - 250.)**2))
        self.assertLess(float(np.percentile(error, 95)), .8)

    def test_large_jump_to_only_nearby_body_is_rejected(self):
        image = np.full((230, 320, 3), 30, np.uint8)
        image[120:140] = 185
        seeds = [[90., 107.], [160., 107.], [230., 107.]]
        result = refine_centerline(image, seeds, expected_width_px=20., max_shift_px=30.)
        self.assertFalse(result["valid"])
        self.assertTrue(all(r["reason"] == "shift_may_change_rope_identity" for r in result["records"]))

    def test_image_border_does_not_supply_fake_opposite_edge(self):
        image = np.full((100, 200, 3), 140, np.uint8)
        result = refine_centerline(image, [[60., 5.], [100., 5.], [140., 5.]], expected_width_px=20.)
        self.assertFalse(result["valid"])
        self.assertTrue(all(r["reason"] == "insufficient_image_support_for_two_sided_profile" for r in result["records"]))


if __name__ == "__main__":
    unittest.main()
