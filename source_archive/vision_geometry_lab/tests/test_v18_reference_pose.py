import unittest
from dataclasses import replace

import cv2
import numpy as np

from vision_geometry_lab.v18_reference_pose import FixedBackgroundPose, DEFAULT_PARAMETERS


def scene(height=480, width=640):
    rng = np.random.default_rng(20260930)
    image = np.full((height, width), 80, np.uint8)
    for _ in range(700):
        x = int(rng.integers(10, width - 10))
        y = int(rng.integers(10, height - 10))
        cv2.circle(image, (x, y), int(rng.integers(2, 7)), int(rng.integers(20, 240)), -1)
    return cv2.cvtColor(cv2.GaussianBlur(image, (3, 3), .7), cv2.COLOR_GRAY2BGR)


def warped(image, dx=0, dy=0, rotation=0, scale=1):
    h, w = image.shape[:2]
    affine = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), rotation, scale)
    affine[:, 2] += [dx, dy]
    return cv2.warpAffine(image, affine, (w, h)), affine


class FixedBackgroundPoseTests(unittest.TestCase):
    def test_identity_is_supported_by_texture_even_for_same_object(self):
        reference = scene()
        result = FixedBackgroundPose(reference).measure(reference)
        self.assertTrue(result['valid'], result['diagnostics']['reason'])
        np.testing.assert_allclose(result['reference_to_current'], [[1, 0, 0], [0, 1, 0]], atol=.001)
        self.assertGreaterEqual(result['diagnostics']['inliers'], 30)
        self.assertGreaterEqual(result['diagnostics']['occupied_reference_cells'], 6)
        self.assertFalse(result['diagnostics']['spool_motion_inferred'])
        self.assertFalse(result['diagnostics']['rope_visibility_inferred'])
        self.assertFalse(result['diagnostics']['physical_time_inferred'])

    def test_translation_maps_reference_to_current(self):
        reference = scene()
        current, affine = warped(reference, dx=3, dy=2)
        result = FixedBackgroundPose(reference).measure(current)
        self.assertTrue(result['valid'], result['diagnostics']['reason'])
        np.testing.assert_allclose(result['reference_to_current'], affine, atol=.15)

    def test_resize_preserves_original_pixel_units(self):
        reference = scene(960, 1280)
        current, affine = warped(reference, dx=5, dy=3)
        result = FixedBackgroundPose(reference).measure(current)
        self.assertTrue(result['valid'], result['diagnostics']['reason'])
        self.assertEqual(result['diagnostics']['registration_crop_shape'][1], 1024)
        np.testing.assert_allclose(result['reference_to_current'], affine, atol=.2)

    def test_identical_blank_object_never_bypasses_texture_validation(self):
        blank = np.zeros((480, 640, 3), np.uint8)
        result = FixedBackgroundPose(blank).measure(blank)
        self.assertFalse(result['valid'])
        self.assertIsNone(result['reference_to_current'])
        self.assertEqual(result['diagnostics']['reason'], 'no_background_features')

    def test_bottom_texture_cannot_support_background_pose(self):
        image = scene()
        image[:192] = 80
        result = FixedBackgroundPose(image).measure(image.copy())
        self.assertFalse(result['valid'])
        self.assertEqual(result['diagnostics']['reason'], 'no_background_features')

    def test_bottom_motion_does_not_change_background_result(self):
        reference = scene()
        current = reference.copy()
        current[192:] = 255 - current[192:]
        result = FixedBackgroundPose(reference).measure(current)
        self.assertTrue(result['valid'], result['diagnostics']['reason'])
        np.testing.assert_allclose(result['reference_to_current'], [[1, 0, 0], [0, 1, 0]], atol=.001)

    def test_excess_rotation_is_rejected(self):
        reference = scene()
        current, _ = warped(reference, rotation=1.3)
        result = FixedBackgroundPose(reference).measure(current)
        self.assertFalse(result['valid'])
        self.assertIsNone(result['reference_to_current'])
        self.assertIn('rotation', result['diagnostics']['reason'])

    def test_excess_translation_and_scale_are_rejected(self):
        reference = scene()
        for kwargs, reason in ((dict(dx=12), 'translation'), (dict(scale=1.015), 'scale')):
            with self.subTest(kwargs=kwargs):
                current, _ = warped(reference, **kwargs)
                result = FixedBackgroundPose(reference).measure(current)
                self.assertFalse(result['valid'])
                self.assertIn(reason, result['diagnostics']['reason'])

    def test_shape_change_does_not_replace_reference(self):
        reference = scene()
        pose = FixedBackgroundPose(reference)
        result = pose.measure(scene(500, 640))
        self.assertFalse(result['valid'])
        self.assertEqual(result['diagnostics']['reason'], 'image_shape_changed')
        self.assertTrue(pose.measure(reference)['valid'])

    def test_success_does_not_refresh_or_chain_reference(self):
        reference = scene()
        pose = FixedBackgroundPose(reference)
        for dx in (3, 6, 2):
            current, affine = warped(reference, dx=dx)
            result = pose.measure(current)
            self.assertTrue(result['valid'], result['diagnostics']['reason'])
            np.testing.assert_allclose(result['reference_to_current'], affine, atol=.15)
            self.assertFalse(result['diagnostics']['reference_refreshed'])
        # Full diagonal is 800 px: 9 px must fail, despite a small step from
        # preceding accepted observations. This is not a chained LK estimate.
        current, _ = warped(reference, dx=9)
        result = pose.measure(current)
        self.assertFalse(result['valid'])
        self.assertIn('translation', result['diagnostics']['reason'])

    def test_failed_measurements_do_not_refresh_reference(self):
        reference = scene()
        pose = FixedBackgroundPose(reference)
        current, _ = warped(reference, rotation=1.3)
        self.assertFalse(pose.measure(current)['valid'])
        self.assertFalse(pose.measure(None)['valid'])
        current, affine = warped(reference, dx=2)
        result = pose.measure(current)
        self.assertTrue(result['valid'], result['diagnostics']['reason'])
        np.testing.assert_allclose(result['reference_to_current'], affine, atol=.15)

    def test_constructor_copies_reference_pixels(self):
        reference = scene()
        original = reference.copy()
        pose = FixedBackgroundPose(reference)
        reference[:] = 0
        result = pose.measure(original)
        self.assertTrue(result['valid'], result['diagnostics']['reason'])
        np.testing.assert_allclose(result['reference_to_current'], [[1, 0, 0], [0, 1, 0]], atol=.001)

    def test_bad_parameters_and_reference_are_rejected(self):
        for kwargs in (dict(background_top_fraction=.5), dict(maximum_width=2048),
                       dict(maximum_rotation_deg=.8), dict(maximum_features=2),
                       dict(minimum_grid_cells=13), dict(maximum_lk_error=float('nan'))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FixedBackgroundPose(scene(), replace(DEFAULT_PARAMETERS, **kwargs))
        with self.assertRaises(ValueError):
            FixedBackgroundPose(None)


if __name__ == '__main__':
    unittest.main()
