"""Image extraction, ambiguity and no-boundary-attraction regressions."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from vision_geometry_lab.wound_rope_tracker import detect_wound_curve, _comparison


ROOT = Path(__file__).resolve().parents[2]


def scene(kind="curve", mirror=False):
    h, w = 600, 800
    yy, xx = np.mgrid[:h, :w]
    curve = 200. + .0007 * np.maximum(xx-240., 0.)**2
    if kind == "straight" or kind == "parallel":
        curve = np.full_like(xx, 200., dtype=float)
    response = np.exp(-.5*((yy-curve)/2.7)**2)
    if kind == "parallel":
        response = np.maximum(response, np.exp(-.5*((yy-curve-17.)/2.7)**2))
    if kind == "fork":
        lower = 200. - .0007 * np.maximum(xx-300., 0.)**2
        response = np.maximum(response, np.exp(-.5*((yy-lower)/2.7)**2))
    im = np.repeat((20.+response*210.)[..., None], 3, axis=2).astype(np.uint8)
    rope = dict(valid=True, normal=[0., 1.], rho_px=200., width_px=7.,
                observed_segment=[[55., 200.], [230., 200.]], diagnostics={})
    arc = dict(valid=True, points=[[350., 400.], [400., 410.], [500., 420.], [650., 450.]], segments=[])
    if kind == "empty":
        im[:] = 20
    if mirror:
        im = im[:, ::-1].copy()
        rope["observed_segment"] = [[w-1-p[0], p[1]] for p in rope["observed_segment"]]
        arc["points"] = [[w-1-p[0], p[1]] for p in arc["points"]]
    return im, rope, arc


class WoundCurveTests(unittest.TestCase):
    def test_single_curve_is_extracted_but_not_physically_confirmed(self):
        im, rope, arc = scene()
        r = detect_wound_curve(im, rope, arc, max_length_px=420.)
        self.assertTrue(r["valid"], r)
        p = np.array(r["candidates"][0]["points"])
        expected = 200.+.0007*np.maximum(p[:, 0]-240., 0.)**2
        self.assertLess(np.median(abs(p[:, 1]-expected)), 2.)
        self.assertTrue(r["candidates"][0]["bend_evidence"]["resolved"])
        self.assertFalse(r["identity_confirmed"])
        self.assertFalse(r["edge_contact_confirmed"])
        self.assertIsNone(r["force_command"])
        json.dumps(r, allow_nan=False)

    def test_mirrored_camera_does_not_need_angle_sign_rule(self):
        im, rope, arc = scene(mirror=True)
        r = detect_wound_curve(im, rope, arc, max_length_px=420.)
        self.assertTrue(r["valid"], r)
        p = np.array(r["candidates"][0]["points"])
        expected = 200.+.0007*np.maximum(799.-p[:, 0]-240., 0.)**2
        self.assertLess(np.median(abs(p[:, 1]-expected)), 2.)

    def test_moving_arc_cannot_attract_image_curve(self):
        im, rope, arc = scene()
        a = detect_wound_curve(im, rope, arc, max_length_px=420.)
        shifted = dict(arc, points=(np.array(arc["points"])+[0., 55.]).tolist())
        b = detect_wound_curve(im, rope, shifted, max_length_px=420.)
        self.assertEqual(a["candidates"][0]["points"], b["candidates"][0]["points"])
        self.assertFalse(a["diagnostics"]["arc_used_in_path_score"])

    def test_two_equal_parallel_ridges_are_identity_ambiguous(self):
        im, rope, arc = scene("parallel")
        r = detect_wound_curve(im, rope, arc, max_length_px=420.)
        self.assertTrue(r["valid"], r)
        self.assertEqual(r["status"], "identity_ambiguous", r)
        self.assertIn("repetitive_parallel_rope_texture", r["reason"])
        self.assertFalse(r["identity_confirmed"])

    def test_fork_is_not_promoted_to_identity(self):
        im, rope, arc = scene("fork")
        r = detect_wound_curve(im, rope, arc, max_length_px=420.)
        self.assertTrue(r["valid"], r)
        self.assertFalse(r["identity_confirmed"])
        self.assertEqual(r["status"], "identity_ambiguous", r)

    def test_no_texture_or_no_free_anchor_abstains(self):
        im, rope, arc = scene("empty")
        self.assertFalse(detect_wound_curve(im, rope, arc)["valid"])
        self.assertFalse(detect_wound_curve(im, dict(valid=False), arc)["valid"])
        self.assertFalse(detect_wound_curve(im, rope, arc, seed_points=[[350., 210.], [600., 290.]])["valid"])

    def test_guidance_cannot_bridge_erased_image_evidence(self):
        im, rope, arc = scene()
        im[:, 420:500] = 20
        seeds = [[350., 200.+.0007*(350-240)**2], [600., 200.+.0007*(600-240)**2]]
        r = detect_wound_curve(im, rope, arc, seed_points=seeds, max_length_px=450.)
        if r["valid"]:
            self.assertIn("manual_seeds_not_all_image_supported", r["reason"])
            self.assertLess(max(p[0] for p in r["candidates"][0]["points"]), 500.)
        self.assertFalse(r["identity_confirmed"])

    def test_explicit_empty_local_boundary_does_not_use_raw_arc(self):
        im, rope, arc = scene()
        raw = [[float(x), float(200+.0007*max(x-240, 0)**2)] for x in np.arange(300, 650, 5)]
        arc["segments"] = [raw]
        arc["local_boundary_segments"] = []
        r = detect_wound_curve(im, rope, arc, max_length_px=420.)
        self.assertTrue(r["valid"], r)
        self.assertFalse(r["contact_comparison"]["valid"])
        self.assertEqual(r["contact_comparison"]["reason"], "no_direct_observed_boundary_segments")

    def test_boundary_gap_endcaps_are_not_observations(self):
        r = _comparison(np.array([[40., 1.], [50., 0.], [60., 1.]]),
                        [[[0., 0.], [10., 0.], [20., 0.]], [[80., 0.], [90., 0.], [100., 0.]]], 10.)
        self.assertFalse(r["valid"], r)
        self.assertEqual(r["supported_samples"], 0)


class WoundCurveVideoTests(unittest.TestCase):
    def test_clean_manual_reference_is_guidance_not_identity_confirmation(self):
        reference = ROOT / "vision_geometry_lab/outputs/wound_curve_v3/manual_247/reference.json"
        if not reference.exists():
            self.skipTest("user manual-reference fixture unavailable")
        ref = json.loads(reference.read_text(encoding="utf-8"))
        from vision_geometry_lab.rope_detector import detect_rope
        from vision_geometry_lab.arc_detector import detect_arc
        from vision_geometry_lab.geometry import rope_exclusion_band
        capture = cv2.VideoCapture(ref["clean_source"])
        capture.set(cv2.CAP_PROP_POS_FRAMES, ref["frame_index"])
        ok, im = capture.read()
        capture.release()
        self.assertTrue(ok)
        rope = detect_rope(im)
        mask, _ = rope_exclusion_band(im.shape, rope)
        arc = detect_arc(im, exclusion_mask=mask)
        r = detect_wound_curve(im, rope, arc, seed_points=ref["points_original_px"])
        self.assertTrue(r["valid"], r)
        distances = r["candidates"][0]["manual_seed_distances_px"]
        # Coverage of rough user hints, NOT automatic accuracy or body-center GT.
        self.assertLess(max(distances), 25., distances)
        self.assertGreater(r["candidates"][0]["supported_length_px"], 1000.)
        self.assertFalse(r["identity_confirmed"])
        self.assertFalse(r["diagnostics"]["manual_route_interpolation_emitted_as_observation"])
        self.assertTrue(r["diagnostics"]["manual_route_used_as_search_prior"])

    def test_requested_frames_never_claim_incoming_identity(self):
        video = ROOT / "现场视频/训练采集/双相机同步/dual_20260924_111500/left_DA9627065.avi"
        if not video.exists():
            self.skipTest("video fixture unavailable")
        from vision_geometry_lab.rope_detector import detect_rope
        from vision_geometry_lab.arc_detector import detect_arc
        from vision_geometry_lab.geometry import rope_exclusion_band
        capture = cv2.VideoCapture(str(video))
        try:
            for index in (245, 246, 247, 251):
                with self.subTest(index=index):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, im = capture.read()
                    self.assertTrue(ok)
                    rope = detect_rope(im)
                    mask, _ = rope_exclusion_band(im.shape, rope)
                    arc = detect_arc(im, exclusion_mask=mask)
                    r = detect_wound_curve(im, rope, arc)
                    self.assertTrue(r["valid"], r)
                    self.assertEqual(r["status"], "identity_ambiguous")
                    self.assertFalse(r["identity_confirmed"])
                    self.assertFalse(r["edge_contact_confirmed"])
                    self.assertIn("repetitive_parallel_rope_texture", r["reason"])
                    self.assertGreater(len(r["candidates"][0]["observed_segments"]), 0)
                    json.dumps(r, allow_nan=False)
        finally:
            capture.release()


if __name__ == "__main__":
    unittest.main()
