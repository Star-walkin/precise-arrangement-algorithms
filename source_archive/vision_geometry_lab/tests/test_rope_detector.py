"""Synthetic geometry and repository-frame regressions for the fresh detector.

Run: vision_prototype/.venv/Scripts/python.exe -m unittest
     vision_geometry_lab.tests.test_rope_detector -v
"""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from vision_geometry_lab.rope_detector import detect_rope


ROOT = Path(__file__).resolve().parents[2]


def stripe(angle=23., width=6., exposure=1., rho=None, yellow=False):
    h, w = 480, 720
    a = np.deg2rad(angle)
    normal = np.array([-np.sin(a), np.cos(a)])
    center = np.array([w * .5 + .31, h * .5 + .27])
    rho = float(center @ normal) if rho is None else float(rho)
    yy, xx = np.mgrid[:h, :w]
    distance = xx * normal[0] + yy * normal[1] - rho
    alpha = 1 / (1 + np.exp(np.clip((np.abs(distance) - width / 2) * 4, -60, 60)))
    background = np.array([20., 95., 125.]) if yellow else np.array([15., 15., 15.])
    image = (background + alpha[..., None] * (235. - background)) * exposure
    return np.clip(image, 0, 255).astype(np.uint8), normal, rho


class RopeSyntheticTests(unittest.TestCase):
    def assert_line(self, result, normal, rho, angle_tol=.2, offset_tol=.8):
        self.assertTrue(result["valid"], result)
        actual = np.array(result["normal"])
        if normal[1] < 0 or (abs(normal[1]) < 1e-8 and normal[0] < 0):
            normal, rho = -normal, -rho
        error = np.rad2deg(np.arccos(np.clip(abs(actual @ normal), 0, 1)))
        self.assertLess(error, angle_tol, result)
        self.assertLess(abs(result["rho_px"] - rho), offset_tol, result)
        self.assertGreaterEqual(result["normal"][1], -1e-8)
        json.dumps(result, allow_nan=False)

    def test_arbitrary_orientations_and_exposure(self):
        for angle in (0., 23., 76., 90., 137., 169.):
            for exposure in (.35, .6, 1.):
                with self.subTest(angle=angle, exposure=exposure):
                    im, normal, rho = stripe(angle, exposure=exposure)
                    result = detect_rope(im)
                    self.assert_line(result, normal, rho)
                    self.assertAlmostEqual(result["width_px"], 6., delta=.7)

    def test_roi_returns_original_coordinates(self):
        im, normal, rho = stripe(23.)
        self.assert_line(detect_rope(im, roi=(150, 100, 650, 400)), normal, rho)

    def test_mask_is_proposal_not_line_truth(self):
        im, normal, rho = stripe(23.)
        yy, xx = np.mgrid[:im.shape[0], :im.shape[1]]
        # Deliberately biased/over-wide SAM mask: the image must determine rho.
        mask = (np.abs(xx * normal[0] + yy * normal[1] - rho - 3.) < 11).astype(np.uint8)
        self.assert_line(detect_rope(im, rope_mask=mask), normal, rho)

    def test_seeds_do_not_determine_offset(self):
        im, normal, rho = stripe(23.)
        d = np.array([normal[1], -normal[0]])
        c = rho * normal + 270 * d
        seeds = [c + 30 * d + 2 * normal, c + 290 * d + 2 * normal]
        self.assert_line(detect_rope(im, seed_points=seeds), normal, rho)

    def test_flip_orientation_and_offset(self):
        im, normal, rho = stripe(23.)
        flipped_normal = np.array([-normal[0], normal[1]])
        flipped_rho = rho - normal[0] * (im.shape[1] - 1)
        self.assert_line(detect_rope(im[:, ::-1].copy()), flipped_normal, flipped_rho)

    def test_parallel_ambiguity_abstains(self):
        im, n, rho = stripe(0.)
        other, _, _ = stripe(0., rho=rho + 65.)
        result = detect_rope(np.maximum(im, other))
        self.assertFalse(result["valid"], result)
        self.assertEqual(result["reason"], "ambiguous_parallel_or_crossing_candidates")

    def test_thin_highlight_width_is_unavailable(self):
        im, n, rho = stripe(0., width=1.2)
        r = detect_rope(im, seed_points=[[40., 240.27], [670., 240.27]])
        self.assert_line(r, n, rho)
        self.assertIsNone(r["width_px"], r)
        self.assertEqual(r["diagnostics"]["width_status"], "unresolved_highlight")
        self.assertIsNotNone(r["width_interval_px"])
        self.assertGreater(r["diagnostics"]["candidate_optical_width_px"], 0.)

    def test_color_envelope_ignores_off_center_bright_facet(self):
        h, w = 480, 720
        yy, xx = np.mgrid[:h, :w]
        n = np.array([-.2, np.sqrt(.96)])
        rho = float(np.array([360.3, 240.2]) @ n)
        q = xx * n[0] + yy * n[1] - rho
        body = 1 / (1 + np.exp(np.clip((abs(q) - 5.) * 5, -60, 60)))
        facet = 1 / (1 + np.exp(np.clip((abs(q - 2.) - 1.3) * 6, -60, 60)))
        background = np.array([25., 85., 120.])
        im = background + body[..., None] * (75. - background)
        im += facet[..., None] * 155.
        r = detect_rope(np.clip(im, 0, 255).astype(np.uint8))
        self.assert_line(r, n, rho, offset_tol=1.)
        self.assertEqual(r["diagnostics"]["line_support_kind"], "bilateral_yellow_free_span")
        self.assertGreater(r["width_px"], 8.)
        self.assertGreater(r["width_interval_px"][1], r["width_interval_px"][0])
        self.assertGreater(r["normal_uncertainty_px"], 0.)
        self.assertGreater(r["angular_uncertainty_deg"], 0.)

    def test_empty_and_invalid_inputs(self):
        self.assertFalse(detect_rope(np.zeros((480, 720, 3), np.uint8))["valid"])
        self.assertEqual(detect_rope(None)["reason"], "image_must_be_bgr")
        im, _, _ = stripe()
        self.assertEqual(detect_rope(im, rope_mask=np.ones((10, 10)))["reason"], "mask_must_match_original_image")
        self.assertEqual(detect_rope(im, roi=(20, 20, 10, 10))["reason"], "roi_too_small")


@unittest.skipUnless((ROOT / "vision_training/task_dual_midlayer_v1").exists(), "repository image fixture unavailable")
class RopeRepositoryTests(unittest.TestCase):
    def test_positive_negative_exposure_and_mirror(self):
        base = ROOT / "vision_training/task_dual_midlayer_v1/holdout/frames"
        positive = cv2.imread(str(base / "dual_20260923_101643__pair0100__right.jpg"))
        negative = cv2.imread(str(base / "dual_20260923_101643__pair0100__left.jpg"))
        reference = detect_rope(positive)
        self.assertTrue(reference["valid"], reference)
        for exposure in (.5, 1., 1.3):
            for mirror in (False, True):
                with self.subTest(exposure=exposure, mirror=mirror):
                    pos = np.clip(positive.astype(float) * exposure, 0, 255).astype(np.uint8)
                    neg = np.clip(negative.astype(float) * exposure, 0, 255).astype(np.uint8)
                    if mirror:
                        pos, neg = pos[:, ::-1].copy(), neg[:, ::-1].copy()
                    result = detect_rope(pos)
                    self.assertTrue(result["valid"], result)
                    expected = (180 - reference["angle_deg"]) % 180 if mirror else reference["angle_deg"]
                    self.assertLess(abs(result["angle_deg"] - expected), .25, result)
                    self.assertFalse(detect_rope(neg)["valid"])

    def test_mask_recovers_conservative_auto_miss(self):
        root = ROOT / "vision_training/task_dual_thicklayer_v1"
        image = root / "overview/dual_20260923_113029__t55__left.jpg"
        if not image.exists():
            self.skipTest("thick-layer fixture unavailable")
        im = cv2.imread(str(image))
        mask = cv2.imread(str(root / "train/labels/dual_20260923_113029__t55__left/incoming_rope_free_span.png"), cv2.IMREAD_GRAYSCALE)
        r = detect_rope(im, rope_mask=mask)
        self.assertTrue(r["valid"], r)
        self.assertGreater(r["angle_deg"], 170.)
        self.assertLess(r["angle_deg"], 175.)

    def test_near_edge_video_free_span_envelope(self):
        video = ROOT / "现场视频/训练采集/双相机同步/dual_20260924_111500/left_DA9627065.avi"
        if not video.exists():
            self.skipTest("near-edge video fixture unavailable")
        capture = cv2.VideoCapture(str(video))
        try:
            for seconds in (24.3, 26.5, 27.3, 30.):
                with self.subTest(seconds=seconds):
                    capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
                    ok, im = capture.read()
                    self.assertTrue(ok)
                    r = detect_rope(im)
                    self.assertTrue(r["valid"], r)
                    self.assertEqual(r["diagnostics"]["line_support_kind"], "bilateral_yellow_free_span")
                    self.assertLess(r["residual_px"], 1.)
                    self.assertGreater(r["width_px"], 15.)
                    self.assertLess(r["width_px"], 30.)
                    self.assertLess(r["width_interval_px"][0], r["width_px"])
                    self.assertGreater(r["width_interval_px"][1], r["width_px"])
                    self.assertGreater(len(r["diagnostics"]["width_local_samples"]), 2)
                    self.assertGreater(r["angular_uncertainty_deg"], 0.)
                    self.assertLess(r["angular_uncertainty_deg"], 1.)
                    self.assertFalse(r["diagnostics"]["arc_used_as_tangency_constraint"])
                    json.dumps(r, allow_nan=False)
        finally:
            capture.release()

    def test_short_color_support_preserves_verified_brightness_fit(self):
        root = ROOT / "vision_training/task_dual_thicklayer_v1"
        image = root / "overview/dual_20260923_113305__t25__right.jpg"
        if not image.exists():
            self.skipTest("thick-layer fixture unavailable")
        im = cv2.imread(str(image))
        mask = cv2.imread(str(root / "train/labels/dual_20260923_113305__t25__right/incoming_rope_free_span.png"), cv2.IMREAD_GRAYSCALE)
        for proposal in (None, mask):
            r = detect_rope(im, rope_mask=proposal)
            self.assertTrue(r["valid"], r)
            self.assertGreater(r["diagnostics"]["support_length_px"], 180.)


if __name__ == "__main__":
    unittest.main()
