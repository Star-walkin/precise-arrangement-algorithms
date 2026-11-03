import copy
import hashlib
import unittest

import numpy as np

from vision_geometry_lab.v18_front_diagnostics import (
    KEY_TIMES, MISSING_RGB, candidate_points, diagnostic_masks, displacement_rgb,
    baseline_status, diagnostic_roi, fixed_roi, mirror_roi, overlay_raw, pixels_sha, reference_to_raw,
    spacetime_figure, validate_measurement, validate_original_frame,
)


def fixture():
    times = np.array([0., 2., 5., 6., 10., 60.])
    rows = np.array([80., 100., 120.])
    t, j = len(times), len(rows)
    affine = np.array([[1., 0., 0.], [0., 1., 0.]])
    arrays = dict(times_s=times, rows_y=rows,
        aligned_semantic_x=np.full((t, j), 70.), semantic_seed_x_current=np.full((t, j), 70.), pose_valid=np.ones(t, bool),
        transforms=np.repeat(affine[None], t, axis=0),
        baseline_x=np.full(j, 70.), baseline_nx=np.ones(j), baseline_ny=np.zeros(j),
        normal_offset=np.full((t, j), 4.), normal_gradient=np.full((t, j), 10.),
        normal_peak_ratio=np.full((t, j), 1.5), normal_clipped_fraction=np.zeros((t, j)),
        normal_profiles=np.zeros((t, j, 131)), baseline_normal_offset=np.full(j, 2.),
        normal_residual=np.full((t, j), 2.))
    records = [dict(frame_index=round(time * 10), time_s=float(time), frame_sha256='a' * 64,
                    pose=dict(valid=True, reference_to_current=affine.tolist())) for time in times]
    index = dict(protocol_sha256='b' * 64,
        video=dict(path='dummy.avi', fps=10., frame_count=601, width=300, height=200, camera='left'),
        records=records)
    return arrays, index


class FrontDiagnosticsTests(unittest.TestCase):
    def test_contract_accepts_finite_data_and_preserves_missing_measurements(self):
        arrays, index = fixture()
        arrays['normal_offset'][4, 1] = np.nan
        self.assertTrue(validate_measurement(arrays, index, 'b' * 64))
        self.assertTrue(np.isnan(arrays['normal_offset'][4, 1]))

    def test_contract_rejects_shape_time_hash_and_pose_mismatches(self):
        for kind in ('shape', 'time', 'protocol', 'framehash', 'pose', 'transform'):
            arrays, index = fixture()
            digest = 'b' * 64
            if kind == 'shape': arrays['normal_profiles'] = np.zeros((6, 3, 130))
            elif kind == 'time': index['records'][3]['time_s'] += .1
            elif kind == 'protocol': digest = 'c' * 64
            elif kind == 'framehash': index['records'][3]['frame_sha256'] = 'bad'
            elif kind == 'pose': arrays['pose_valid'][3] = False
            elif kind == 'transform': arrays['transforms'][3, 0, 2] = 1.
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_measurement(arrays, index, digest)

    def test_reference_transform_precedes_left_mirror(self):
        points = np.array([[10., 20.], [25., 40.]])
        affine = np.array([[1., 0., 3.], [0., 1., 4.]])
        np.testing.assert_allclose(reference_to_raw(points, affine, 100, 'right'), [[13, 24], [28, 44]])
        np.testing.assert_allclose(reference_to_raw(points, affine, 100, 'left'), [[86, 24], [71, 44]])
        # A wrong raw-first order would put the first x at 92 rather than 86.
        self.assertNotEqual(reference_to_raw(points, affine, 100, 'left')[0, 0], 92)

    def test_candidate_uses_absolute_offset_not_residual(self):
        arrays, _ = fixture()
        np.testing.assert_allclose(candidate_points(arrays, 3)[:, 0], 74.)
        np.testing.assert_allclose(candidate_points(arrays, 3, baseline=True)[:, 0], 72.)

    def test_fixed_roi_is_clamped_and_independent_of_current_candidates(self):
        arrays, _ = fixture()
        before = fixed_roi(arrays, 300, 200)
        arrays['normal_offset'][:] = 10000.
        arrays['aligned_semantic_x'][:] = -1000.
        arrays['transforms'][:, 0, 2] = 50.
        self.assertEqual(before, fixed_roi(arrays, 300, 200))
        self.assertEqual(before, (0, 40, 176, 161))
        self.assertEqual(mirror_roi(before, 300, 'left'), (124, 40, 300, 161))
        self.assertEqual(mirror_roi(before, 300, 'right'), before)

    def test_half_open_roi_mirrors_pixels_exactly(self):
        image = np.arange(7 * 11).reshape(7, 11)
        canonical_roi = (2, 1, 6, 5)
        x0, y0, x1, y1 = canonical_roi
        a, b, c, d = mirror_roi(canonical_roi, 11, 'left')
        np.testing.assert_array_equal(image[b:d, a:c][:, ::-1], image[:, ::-1][y0:y1, x0:x1])

    def test_missing_baseline_uses_startup_seeds_for_crop_not_measurement(self):
        arrays, index = fixture()
        for name in ('baseline_x', 'baseline_nx', 'baseline_ny', 'baseline_normal_offset'):
            arrays[name][:] = np.nan
        arrays['normal_offset'][:] = np.nan
        arrays['normal_residual'][:] = np.nan
        self.assertTrue(validate_measurement(arrays, index, 'b' * 64))
        count, status = baseline_status(arrays)
        self.assertEqual(count, 0)
        self.assertIn('法向变化未评估', status)
        crop, info = diagnostic_roi(arrays, 300, 200)
        self.assertEqual(crop, (0, 40, 176, 161))
        self.assertFalse(info['creates_measurement_baseline'])
        arrays['semantic_seed_x_current'][3:] = 250.
        self.assertEqual(crop, diagnostic_roi(arrays, 300, 200)[0])
        self.assertTrue(np.isnan(arrays['normal_offset']).all())

    def test_seed_crop_inverse_maps_pose_and_falls_back_without_guessing_front(self):
        arrays, _ = fixture()
        arrays['transforms'][:, 0, 2] = 10.
        arrays['semantic_seed_x_current'][:] = 80.
        self.assertEqual(diagnostic_roi(arrays, 300, 200)[0], (0, 40, 176, 161))
        for name in ('baseline_x', 'baseline_nx', 'baseline_ny', 'baseline_normal_offset', 'semantic_seed_x_current'):
            arrays[name][:] = np.nan
        crop, info = diagnostic_roi(arrays, 300, 200)
        self.assertEqual(crop, (0, 40, 300, 161))
        self.assertEqual(info['roi_source'], 'full_width_protocol_row_band_no_spatial_baseline_or_seed')

    def test_masks_do_not_treat_ambiguity_as_validity_or_visibility(self):
        arrays, _ = fixture()
        arrays['normal_peak_ratio'][3] = [1.19, 1.2, np.nan]
        arrays['normal_gradient'][3] = [-10., 10., 0.]
        arrays['pose_valid'][4] = False
        masks = diagnostic_masks(arrays)
        np.testing.assert_array_equal(masks['ambiguous'][3], [True, False, False])
        np.testing.assert_array_equal(masks['undefined'][3], [False, False, True])
        np.testing.assert_array_equal(masks['polarity_changed'][3], [True, False, False])
        self.assertTrue(masks['normal'][3].all())
        self.assertFalse(masks['normal'][4].any())
        np.testing.assert_array_equal(masks['warmup'], [True, True, True, False, False, False])

    def test_color_scale_is_fixed_and_missing_is_opaque_gray(self):
        values = np.array([[-80., -40., 0., 40., 80., np.nan]])
        rgb = displacement_rgb(values, np.ones_like(values, bool))
        np.testing.assert_array_equal(rgb[0, 0], rgb[0, 1])
        np.testing.assert_array_equal(rgb[0, 3], rgb[0, 4])
        np.testing.assert_array_equal(rgb[0, 5], MISSING_RGB)
        invalid = displacement_rgb(np.array([[25.]]), np.array([[False]]))
        np.testing.assert_array_equal(invalid[0, 0], MISSING_RGB)

    def test_invalid_pose_draws_no_overlay_and_raw_is_not_modified(self):
        arrays, _ = fixture()
        image = np.zeros((200, 300, 3), np.uint8)
        arrays['pose_valid'][3] = False
        np.testing.assert_array_equal(overlay_raw(image, arrays, 3, 'left'), image)
        arrays['pose_valid'][3] = True
        overlay = overlay_raw(image, arrays, 3, 'left')
        self.assertGreater(np.count_nonzero(overlay), 0)
        self.assertEqual(np.count_nonzero(image), 0)
        # Current canonical x=74 maps to raw x=225 (not 74).
        self.assertGreater(np.count_nonzero(overlay[80, 225]), 0)
        self.assertEqual(np.count_nonzero(overlay[80, 74]), 0)

    def test_v18_digest_and_actual_decoded_frame_are_required(self):
        frame = np.zeros((20, 30, 3), np.uint8)
        independent = hashlib.sha256(str((frame.shape, str(frame.dtype))).encode('ascii') + frame.tobytes()).hexdigest()
        self.assertEqual(pixels_sha(frame), independent)
        record = dict(frame_index=10, frame_sha256=independent)
        video = dict(width=30, height=20)
        validate_original_frame(frame, 10, record, video)
        with self.assertRaises(ValueError): validate_original_frame(frame, 11, record, video)
        wrong = frame.copy(); wrong[0, 0, 0] = 1
        with self.assertRaises(ValueError): validate_original_frame(wrong, 10, record, video)

    def test_fixed_time_selection_and_chart_render(self):
        self.assertEqual(KEY_TIMES, (0, 10, 20, 22, 24, 26, 28, 30, 32, 34, 40, 50, 60))
        arrays, _ = fixture()
        chart = spacetime_figure(arrays, 'synthetic-only')
        self.assertEqual(chart.size, (2100, 1590))


if __name__ == '__main__':
    unittest.main()
