import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from vision_geometry_lab import v20_appearance_control as v20


class PatchTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2020)
        self.image = self.rng.integers(20, 180, (100, 240), dtype=np.uint8)
        self.roi = [32, 20, 96, 52]
        self.A = np.array([[1., 0., 0.], [0., 1., 0.]])

    def test_rois_nonoverlap_bounds_size_and_integer(self):
        good = [[0, 0, 64, 32], [64, 0, 128, 32]]
        self.assertEqual(v20.validate_rois(good, 128, 32), good)
        for bad in ([[0, 0, 64, 32], [63, 0, 127, 32]],
                    [[0, 0, 63, 32], [64, 0, 128, 32]],
                    [[0, 0, 64, 32], [65, 0, 129, 32]],
                    [[0., 0., 64., 32.], [64., 0., 128., 32.]]):
            with self.assertRaises(ValueError):
                v20.validate_rois(bad, 128, 32)

    def test_identical_and_constant_offset_invariance(self):
        a = v20.patch_descriptor(self.image, self.A, self.roi)
        b = v20.patch_descriptor(self.image.astype(np.float32)+30., self.A, self.roi)
        self.assertTrue(a['valid']); self.assertTrue(b['valid'])
        self.assertAlmostEqual(float(np.linalg.norm(a['descriptor'])), 1., places=6)
        self.assertGreater(v20.cosine_common(a['descriptor'], b['descriptor'], a['support'], b['support']), .999999)

    def test_pose_translation_compensates(self):
        moved = np.zeros_like(self.image); moved[:, 7:] = self.image[:, :-7]
        A = self.A.copy(); A[0, 2] = 7
        a = v20.patch_descriptor(self.image, self.A, self.roi)
        b = v20.patch_descriptor(moved, A, self.roi)
        np.testing.assert_allclose(a['descriptor'], b['descriptor'], atol=1e-7)

    def test_no_pose_constant_and_clipping_are_unknown(self):
        self.assertFalse(v20.patch_descriptor(self.image, self.A, self.roi, False)['valid'])
        self.assertEqual(v20.patch_descriptor(np.full_like(self.image, 100), self.A, self.roi)['reason'], 'weak_texture')
        A = self.A.copy(); A[0, 2] = -40
        r = v20.patch_descriptor(self.image, A, self.roi)
        self.assertEqual(r['reason'], 'patch_clipped'); self.assertGreater(r['clipped_fraction'], .05)

    def test_small_clipping_supported_pixels_and_no_fabricated_texture(self):
        A = self.A.copy(); A[0, 2] = -33
        result = v20.patch_descriptor(np.full_like(self.image, 100), A, self.roi)
        self.assertLess(result['clipped_fraction'], .05)
        self.assertFalse(result['valid']); self.assertEqual(result['reason'], 'weak_texture')

    def test_nonfinite_pose_unknown(self):
        A = self.A.copy(); A[0, 2] = np.nan
        self.assertEqual(v20.patch_descriptor(self.image, A, self.roi)['reason'], 'invalid_pose')

    def test_color_summary_is_diagnostic_not_gate(self):
        bgr = np.zeros((*self.image.shape, 3), np.uint8); bgr[:, :, 1:] = 255
        result = v20.color_diagnostics(bgr, self.A, self.roi)
        self.assertTrue(result['available']); self.assertFalse(result['material_identity_or_occlusion_classified'])
        np.testing.assert_allclose(result['bgr_mean'], [0, 255, 255])


def fixture():
    times = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.])
    d = np.zeros((len(times), 2, 3)); d[:, :, 1] = 1.
    d[5, :, :] = [1., 0., 0.]; d[13, :, :] = [1., 0., 0.]
    d[6, :, :] = [.9, np.sqrt(1.-.9**2), 0.]
    return times, d, np.ones_like(d, bool), np.ones((len(times), 2), bool)


class MatchingTests(unittest.TestCase):
    def test_best_runner_control_and_acceptance(self):
        t, d, s, v = fixture()
        result = v20.select_match(t, d, s, v, 13, v20.HISTORIES[0])
        self.assertTrue(result['accepted']); self.assertEqual(result['best_index'], 5)
        self.assertEqual(result['runner_index'], 6); self.assertAlmostEqual(result['runner_margin'], .1)
        self.assertNotIn(result['control_index'], (5, 6))
        self.assertFalse(result['control_is_guaranteed_nonmatch'])

    def test_two_roi_argmax_disagreement_rejects(self):
        t, d, s, v = fixture(); d[6, 1] = [1., 0., 0.]; d[5, 1] = [.98, .199, 0.]
        result = v20.select_match(t, d, s, v, 13, v20.HISTORIES[0])
        self.assertFalse(result['accepted']); self.assertIn('ROI_partner_disagreement', result['reason'])

    def test_runner_ambiguity_rejects(self):
        t, d, s, v = fixture(); d[6] = d[5]
        result = v20.select_match(t, d, s, v, 13, v20.HISTORIES[0])
        self.assertFalse(result['accepted']); self.assertIn('appearance_partner_ambiguous', result['reason'])

    def test_current_unknown_never_reuses_old_descriptor(self):
        t, d, s, v = fixture(); v[13, 1] = False
        result = v20.select_match(t, d, s, v, 13, v20.HISTORIES[0])
        self.assertFalse(result['accepted']); self.assertIsNone(result['best_index'])

    def test_history_boundaries(self):
        t, d, s, v = fixture()
        np.testing.assert_array_equal(v20.eligible_history(t, 13, v20.HISTORIES[0]), np.arange(1, 12))
        np.testing.assert_array_equal(v20.eligible_history(t, 13, v20.HISTORIES[1]), np.arange(5, 11))
        self.assertEqual(len(v20.eligible_history(t, 12, v20.HISTORIES[1])), 0)

    def test_runner_exclusion_and_missing_margin_fail_closed(self):
        t = np.array([5., 5.1, 5.2, 8.]); d = np.ones((4, 2, 3)); d /= np.sqrt(3)
        result = v20.select_match(t, d, np.ones_like(d, bool), np.ones((4, 2), bool), 3, v20.HISTORIES[0])
        self.assertIsNone(result['runner_index']); self.assertFalse(result['accepted'])

    def test_future_does_not_change_output(self):
        t, d, s, v = fixture()
        expected = v20.select_match(t, d, s, v, 13, v20.HISTORIES[0])
        tf = np.append(t, [13.1, 14.]); df = np.concatenate([d, np.repeat(d[13:14], 2, axis=0)])
        sf = np.ones_like(df, bool); vf = np.ones(df.shape[:2], bool)
        self.assertEqual(expected, v20.select_match(tf, df, sf, vf, 13, v20.HISTORIES[0]))

    def test_finite_times_and_shape_validation(self):
        t, d, s, v = fixture(); t[2] = t[1]
        with self.assertRaises(ValueError):
            v20.select_match(t, d, s, v, 13, v20.HISTORIES[0])


class ComparisonTests(unittest.TestCase):
    def arrays(self):
        off = np.zeros((2, 48)); off[1] = 8.
        profiles = np.tile(np.linspace(20., 100., 131), (2, 48, 1))
        return off, profiles, np.ones((2, 48), bool)

    def test_same_original_rows_positive_offset_and_profiles(self):
        off, pro, q = self.arrays(); result = v20.compare_pair(off, pro, q, 1, 0)
        self.assertEqual([x['common_rows'] for x in result['partitions']], [12]*4)
        self.assertEqual([x['median_offset_delta_px'] for x in result['partitions']], [8.]*4)
        np.testing.assert_allclose(result['raw_profile_cosine'], 1.)

    def test_both_endpoints_quality_required_no_repacking_holes(self):
        off, pro, q = self.arrays(); q[0, :3] = False
        result = v20.compare_pair(off, pro, q, 1, 0)
        self.assertFalse(result['partitions'][0]['valid'])
        self.assertEqual(result['partitions'][0]['common_rows'], 9)
        self.assertTrue(result['partitions'][1]['valid'])
        self.assertTrue(np.isnan(result['delta'][:3]).all())

    def test_constant_profile_centered_similarity_unknown(self):
        off, pro, q = self.arrays(); pro[:] = 80.
        result = v20.compare_pair(off, pro, q, 1, 0)
        self.assertTrue(np.isnan(result['centered_profile_cosine']).all())
        np.testing.assert_allclose(result['raw_profile_cosine'], 1.)

    def test_geometry_does_not_change_appearance_partner(self):
        t, d, s, v = fixture(); off = np.zeros((len(t), 48))
        pro = np.tile(np.arange(131, dtype=float), (len(t), 48, 1)); q = np.ones_like(off, bool)
        a = v20.compare_arrays(t, d, s, v, off, pro, q)
        off[:] = np.random.default_rng(42).normal(size=off.shape)*1000.; q[:] = False
        b = v20.compare_arrays(t, d, s, v, off, pro, q)
        for ar, br in zip(a['records'], b['records']):
            for ah, bh in zip(ar['histories'], br['histories']):
                self.assertEqual(ah['match'], bh['match'])

    def test_prefix_records_and_fixed_bins_keep_missing(self):
        t, d, s, v = fixture(); off = np.zeros((len(t), 48))
        pro = np.tile(np.arange(131, dtype=float), (len(t), 48, 1)); q = np.ones_like(off, bool)
        a = v20.compare_arrays(t, d, s, v, off, pro, q)
        b = v20.compare_arrays(t[:8], d[:8], s[:8], v[:8], off[:8], pro[:8], q[:8])
        self.assertEqual(a['records'][:8], b['records'])
        summary = v20.summarize(a['records'])
        self.assertEqual(summary['complete'][v20.HISTORIES[0]]['frames'], len(t))
        self.assertEqual([x['start_s'] for x in summary['fixed_five_second_buckets']], list(np.arange(0., 60., 5.)))

    def test_twelve_bins_last_includes_exact_sixty_without_extra_bucket(self):
        t = np.arange(601)/10.; d = np.zeros((601, 2, 3)); s = np.ones_like(d, bool)
        v = np.zeros((601, 2), bool); off = np.zeros((601, 48))
        pro = np.zeros((601, 48, 131)); q = np.zeros_like(off, bool)
        result = v20.compare_arrays(t, d, s, v, off, pro, q)
        summary = v20.summarize(result['records'])
        bins = summary['fixed_five_second_buckets']
        self.assertEqual(len(bins), 12)
        self.assertEqual([b['summary'][v20.HISTORIES[0]]['frames'] for b in bins], [50]*11+[51])
        self.assertEqual(bins[-1]['end_s'], 60.); self.assertTrue(bins[-1]['end_inclusive'])
        self.assertFalse(bins[0]['end_inclusive'])

    def test_stop_rule_explicitly_blocks_threshold_tuning_and_physical_claims(self):
        self.assertFalse(v20.STOP_RULE['threshold_tuning'])
        self.assertIn('physical_new_layer_start_t1_confirmation', v20.STOP_RULE['forbidden_inferences'])
        self.assertIn('first_revolution_end_t2_confirmation', v20.STOP_RULE['forbidden_inferences'])
        self.assertIn('first-frame', v20.STOP_RULE['ROI_scope'])

    def test_no_overwrite_freeze_before_source_access(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileExistsError):
                v20.freeze({}, Path(td), Path(td)/'missing')


if __name__ == '__main__':
    unittest.main()
