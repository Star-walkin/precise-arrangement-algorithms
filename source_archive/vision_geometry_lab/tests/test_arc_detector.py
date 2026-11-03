"""Image-geometry tests, using synthetic truth rather than SAM mask precision."""
import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from vision_geometry_lab.arc_detector import detect_arc


def make_scene(bump=False):
    """Pixel-center sampled, smoothly antialiased painted flange / rope interface."""
    height, width = 480, 720
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    image = np.full((height, width, 3), 16.0, np.float32)
    # Full flange oval. Its lower edge is *not* the target.
    ellipse = ((xx - 360) / 290) ** 2 + ((yy - 264) / 112) ** 2 <= 1
    image[ellipse] = (35, 165, 205)
    truth = 245.35 + .0011 * (np.arange(width) - 360.0) ** 2
    if bump:
        truth += 4.0 * np.exp(-.5 * ((np.arange(width) - 420.0) / 9.0) ** 2)
    coverage = np.clip(yy - truth[None, :] + .5, 0, 1)
    coverage *= ((xx > 170) & (xx < 551))
    rope = np.empty_like(image)
    gray = 64 + 22 * np.sin(yy * 1.1 + xx * .08)
    rope[:] = gray[:, :, None]
    image = image * (1 - coverage[:, :, None]) + rope * coverage[:, :, None]
    # Bright achromatic paper and UI boxes must not become a flange.
    image[28:115, 440:690] = 248
    return np.clip(image, 0, 255).astype(np.uint8), truth


class ArcDetectorTest(unittest.TestCase):
    def test_raw_subpixel_inner_boundary_not_outer_ellipse(self):
        image, truth = make_scene()
        result = detect_arc(image)
        self.assertTrue(result["valid"], result)
        points = np.asarray(result["points"])
        central = points[(points[:, 0] > 190) & (points[:, 0] < 530)]
        error = central[:, 1] - np.interp(central[:, 0], np.arange(len(truth)), truth)
        self.assertGreater(len(central), 100)
        self.assertLess(float(np.median(np.abs(error))), .55)
        self.assertLess(float(np.percentile(np.abs(error), 95)), 1.2)
        self.assertEqual(result["interpolated_ranges"], [])
        json.dumps(result, allow_nan=False)

    def test_preserves_local_nonelliptical_bump(self):
        image, truth = make_scene(bump=True)
        result = detect_arc(image)
        self.assertTrue(result["valid"], result)
        points = np.asarray(result["points"])
        bump = points[np.abs(points[:, 0] - 420) < 5]
        error = bump[:, 1] - np.interp(bump[:, 0], np.arange(len(truth)), truth)
        self.assertGreater(len(bump), 2)
        self.assertLess(float(np.max(np.abs(error))), 1.1)

    def test_rotation_exposure_and_contrast(self):
        image, _ = make_scene()
        variants = [np.rot90(image).copy(), np.rot90(image, 2).copy(),
                    np.clip(image.astype(float) * .50, 0, 255).astype(np.uint8),
                    np.clip(image.astype(float) * .78 + 18, 0, 255).astype(np.uint8)]
        base = np.asarray(detect_arc(image)["points"])
        for i, variant in enumerate(variants):
            with self.subTest(variant=i):
                result = detect_arc(variant)
                self.assertTrue(result["valid"], result)
                points = np.asarray(result["points"])
                if i == 0:
                    points = np.column_stack([image.shape[1] - 1 - points[:, 1], points[:, 0]])
                elif i == 1:
                    points = np.array([image.shape[1] - 1, image.shape[0] - 1]) - points
                distances = np.sqrt(np.min(np.sum((points[:, None] - base[None]) ** 2, axis=2), axis=1))
                self.assertLess(float(np.median(distances)), 1.3)

    def test_white_paper_and_solid_flange_are_invalid(self):
        image = np.zeros((360, 640, 3), np.uint8)
        image[80:230, 70:570] = 255
        self.assertFalse(detect_arc(image)["valid"])
        cv2.ellipse(image, (320, 250), (180, 60), 0, 0, 360, (30, 155, 210), -1)
        self.assertFalse(detect_arc(image)["valid"])

    def test_mask_is_only_prior_and_roi_coordinates_are_full_image(self):
        image, truth = make_scene()
        wrong_mask = np.zeros(image.shape[:2], np.uint8)
        wrong_mask[20:90, 20:90] = 255
        result = detect_arc(image, roi=[40, 130, 680, 450], surface_mask=wrong_mask)
        self.assertTrue(result["valid"], result)
        points = np.asarray(result["points"])
        central = points[(points[:, 0] > 210) & (points[:, 0] < 510)]
        error = central[:, 1] - np.interp(central[:, 0], np.arange(len(truth)), truth)
        self.assertLess(float(np.median(np.abs(error))), .6)

    def test_occluded_support_is_not_interpolated(self):
        image, _ = make_scene()
        image[190:310, 337:351] = 22
        result = detect_arc(image)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["interpolated_ranges"], [])
        # No segment may cross the deliberately missing observation.
        for segment in result["segments"]:
            xs = np.asarray(segment)[:, 0]
            self.assertFalse(xs.min() < 338 and xs.max() > 350)

    def test_invalid_arguments(self):
        self.assertFalse(detect_arc(None)["valid"])
        self.assertFalse(detect_arc(np.zeros((40, 40), np.uint8))["valid"])
        image, _ = make_scene()
        self.assertFalse(detect_arc(image, roi=[0, 0, 10, 20])["valid"])
        self.assertFalse(detect_arc(image, surface_mask=np.zeros((10, 10)))["valid"])
        self.assertFalse(detect_arc(image, exclusion_mask=np.zeros((10, 10)))["valid"])

    def test_exclusion_corridor_is_not_sampled_or_bridged(self):
        image, _ = make_scene()
        mask = np.zeros(image.shape[:2], bool)
        mask[:, 334:367] = True
        # Change all excluded pixel values to an adversarial bright false edge.
        changed = image.copy()
        changed[mask] = (0, 255, 255)
        for source in (image, changed):
            result = detect_arc(source, exclusion_mask=mask)
            self.assertTrue(result["valid"], result)
            self.assertGreater(result["diagnostics"]["excluded_profile_count"], 0)
            points = np.asarray(result["points"])
            self.assertFalse(np.any((points[:, 0] >= 333) & (points[:, 0] <= 368)))
            for segment in result["segments"]:
                xs = np.asarray(segment)[:, 0]
                self.assertFalse(xs.min() < 334 and xs.max() > 367)

    def test_full_exclusion_fails_explicitly(self):
        image, _ = make_scene()
        result = detect_arc(image, exclusion_mask=np.ones(image.shape[:2], bool))
        self.assertFalse(result["valid"])
        self.assertGreater(result["diagnostics"]["excluded_profile_count"], 0)

    def test_local_boundary_does_not_depend_on_a_global_ellipse(self):
        image, _ = make_scene(bump=True)
        result = detect_arc(image)
        self.assertTrue(result["local_boundary_valid"], result)
        self.assertGreater(len(result["local_boundary_segments"]), 0)
        self.assertTrue(result["local_boundary_diagnostics"]["independent_of_global_ellipse_acceptance"])
        self.assertFalse(result["local_boundary_diagnostics"]["bridges_support_gaps"])
        for segment in result["local_boundary_segments"]:
            self.assertGreaterEqual(len(segment), 7)

    def test_local_selection_ignores_weak_paint_reflections(self):
        from vision_geometry_lab.arc_detector import _local_boundaries
        x = np.arange(80, dtype=float)*2
        points = np.column_stack([x, .003*(x-75)**2])
        refined = dict(paint_median=np.r_[np.full(60, .38), np.full(20, .21)],
                       paint_low=np.r_[np.full(60, .32), np.full(20, .14)],
                       normals=np.tile([0., 1.], (80, 1)))
        result = _local_boundaries(points, np.ones(80, bool), refined, 2., np.array([0., 1.]))
        self.assertTrue(result["local_boundary_valid"])
        self.assertEqual(len(result["conic_support_points"]), 60)
        self.assertEqual(result["local_boundary_diagnostics"]["paint_support_rejected_points"], 20)


REPOSITORY_VIDEO = Path(__file__).resolve().parents[2] / "现场视频/训练采集/双相机同步/dual_20260924_111500/left_DA9627065.avi"


@unittest.skipUnless(REPOSITORY_VIDEO.exists(), "repository video fixture unavailable")
class ArcCrossingSequenceRegression(unittest.TestCase):
    def test_local_support_survives_global_model_refusals(self):
        from vision_geometry_lab.rope_detector import detect_rope
        from vision_geometry_lab.geometry import rope_exclusion_band
        from vision_geometry_lab.conic_fit import fit_conic
        capture = cv2.VideoCapture(str(REPOSITORY_VIDEO))
        try:
            for index in (245, 246, 247, 251, 265):
                with self.subTest(frame_index=index):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, image = capture.read()
                    self.assertTrue(ok)
                    rope = detect_rope(image)
                    mask, _ = rope_exclusion_band(image.shape, rope)
                    self.assertIsNotNone(mask)
                    arc = detect_arc(image, exclusion_mask=mask)
                    self.assertTrue(arc["local_boundary_valid"], arc["local_boundary_diagnostics"])
                    self.assertGreater(len(arc["local_boundary_segments"]), 0)
                    self.assertLess(len(arc["conic_support_points"]), len(arc["points"]))
                    for segment in arc["local_boundary_segments"]:
                        points = np.asarray(segment)
                        links = points[:-1, None, :]*(1-np.linspace(0, 1, 9)[None, :, None]) + points[1:, None, :]*np.linspace(0, 1, 9)[None, :, None]
                        inside = cv2.remap(mask.astype(np.uint8), links[:, :, 0].astype(np.float32), links[:, :, 1].astype(np.float32), cv2.INTER_NEAREST)
                        self.assertFalse(np.any(inside))
                    model = fit_conic(arc, image.shape)
                    if index == 265:
                        self.assertTrue(model["valid"], model)
                    if index == 251:
                        selected = np.asarray(arc["conic_support_points"])
                        # Reference-frame regression: the visible warm strand
                        # reflections inside the wound surface are not support.
                        wrong_cluster = ((selected[:, 0] > 2180) & (selected[:, 0] < 2340)
                                         & (selected[:, 1] > 2350) & (selected[:, 1] < 2420))
                        self.assertFalse(np.any(wrong_cluster))
                        self.assertFalse(model["valid"])
                        self.assertTrue(arc["local_boundary_valid"])
        finally:
            capture.release()


if __name__ == "__main__":
    unittest.main()
