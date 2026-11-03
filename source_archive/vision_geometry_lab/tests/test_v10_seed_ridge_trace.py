"""One connector, synthetic pixels only; no field targets or hardware."""
import json
import unittest
from unittest.mock import patch

import numpy as np

from vision_geometry_lab.v10_seed_ridge_trace import TRACE_METHODS, trace_seed_ridges


def _scene(kind='curve', mirror=False):
    yy, xx = np.indices((440, 640), dtype=np.float32)
    center = 170+.0008*np.maximum(xx-200, 0)**2
    if kind in ('straight', 'parallel'):
        center[:] = 170
    response = np.exp(-.5*((yy-center)/3.)**2)
    if kind == 'parallel':
        response = np.maximum(response, np.exp(-.5*((yy-center-8)/3.)**2))
    image = np.repeat((35+170*response)[..., None], 3, axis=2).astype(np.uint8)
    if kind == 'blank':
        image[:] = 35
    if kind == 'gap':
        image[:, 300:345] = 35
    if kind == 'dark':
        image = np.repeat((210-160*response)[..., None], 3, axis=2).astype(np.uint8)
    seed = [[80., 170.], [180., 170.]]
    if mirror:
        image = image[:, ::-1].copy()
        seed = [[639-x, y] for x, y in seed]
    return image, seed


class SeedRidgeTraceV10Tests(unittest.TestCase):
    def test_four_methods_share_curve_connector_without_identity_claim(self):
        image, seed = _scene()
        for method in TRACE_METHODS:
            result = trace_seed_ridges(image, seed, 9., method=method, max_length_px=320)
            self.assertTrue(result['valid'], (method, result['diagnostics']))
            points = np.concatenate(result['observed_segments'])
            expected = 170+.0008*np.maximum(points[:, 0]-200, 0)**2
            self.assertLess(np.median(abs(points[:, 1]-expected)), 2.)
            self.assertGreater(points[:, 0].max(), 430)
            self.assertFalse(result['identity_confirmed'])
            self.assertFalse(result['new_layer_confirmed'])
            self.assertIsNone(result['force_command'])
            json.dumps(result, allow_nan=False)

    def test_mirrored_direction_follows_ordered_free_seed(self):
        image, seed = _scene(mirror=True)
        result = trace_seed_ridges(image, seed, 9., method='steger_taylor', max_length_px=320)
        self.assertTrue(result['valid'])
        points = np.concatenate(result['observed_segments'])
        expected = 170+.0008*np.maximum(639-points[:, 0]-200, 0)**2
        self.assertLess(np.median(abs(points[:, 1]-expected)), 2.)
        self.assertLess(result['geometry']['direction_xy'][0], 0)

    def test_blank_and_dark_seed_do_not_supply_bright_rope_identity(self):
        for kind in ('blank', 'dark'):
            image, seed = _scene(kind)
            for method in TRACE_METHODS:
                result = trace_seed_ridges(image, seed, 9., method=method, max_length_px=300)
                if kind == 'blank':
                    self.assertFalse(result['valid'])
                self.assertFalse(result['metadata']['dark_ridges_used_as_centers'])
                self.assertFalse(result['identity_confirmed'])

    def test_long_blank_gap_is_not_filled_or_crossed(self):
        image, seed = _scene('gap')
        for method in TRACE_METHODS:
            result = trace_seed_ridges(image, seed, 9., method=method, max_length_px=350)
            self.assertTrue(result['valid'])
            for segment in result['observed_segments']:
                points = np.asarray(segment)
                self.assertFalse(np.any((points[:, 0] > 315) & (points[:, 0] < 330)))
                self.assertLess(points[:, 0].max(), 345)
            self.assertFalse(result['metadata']['missing_samples_emitted'])

    def test_one_missing_column_breaks_observed_segments_even_after_resume(self):
        # Connector-only unit case: a short missing sample is allowed in its
        # internal state, but cannot be emitted as a straight observed link.
        u = np.arange(0., 200., 2.)
        v = np.arange(-40., 42., 2.)
        response = np.zeros((len(v), len(u)), np.float32)
        response[20, :] = .8
        response[:, 21] = 0  # selected columns are 1,3,5,... at width=8
        tangent = np.zeros(response.shape+(2,), np.float32)
        tangent[..., 0] = 1
        features = dict(u=u, v=v, bright=response, dark=np.zeros_like(response),
                        tangent=tangent, valid=np.ones(response.shape, bool),
                        photometric_supported=response > 0,
                        detector_metadata=dict(method='synthetic_connector_only'))
        image, seed = _scene('straight')
        with patch('vision_geometry_lab.v10_seed_ridge_trace._make_features', return_value=features):
            result = trace_seed_ridges(image, seed, 8., method='steger_taylor', max_length_px=150)
        self.assertTrue(result['valid'])
        self.assertGreaterEqual(len(result['observed_segments']), 2)
        missing_x = seed[-1][0]-32+42
        for segment in result['observed_segments']:
            xs = np.asarray(segment)[:, 0]
            self.assertFalse(xs.min() < missing_x < xs.max())
            self.assertFalse(np.any(xs == missing_x))
        self.assertGreater(max(point[0] for segment in result['observed_segments'] for point in segment), missing_x+20)

    def test_geometry_masks_and_legacy_support_are_reported(self):
        image, seed = _scene('straight')
        for method in TRACE_METHODS:
            result = trace_seed_ridges(image, seed, 9., method=method,
                                      max_length_px=300, return_maps=True)
            maps = result['ridge_maps']
            self.assertEqual(maps['bright_response'].shape, maps['valid_mask'].shape)
            self.assertFalse(maps['bright_response'][~maps['valid_mask']].any())
            self.assertFalse(maps['dark_response'][~maps['valid_mask']].any())
            self.assertEqual(result['geometry']['support_differs_from_new_methods'], method=='legacy_v3')
            if method != 'legacy_v3':
                self.assertEqual(result['geometry']['v_range_px'], [-768., 768.])
            else:
                self.assertLess(result['geometry']['v_range_px'][1], 768.)
            self.assertFalse(result['metadata']['original_targets_used'])
            self.assertFalse(result['metadata']['boundary_used'])
            self.assertFalse(result['metadata']['exported_path_uses_taylor_subpixel_coordinates'])

    def test_dense_response_without_bright_localized_support_cannot_enter_graph(self):
        image, seed = _scene('straight')
        for support, sign in ((False, 1), (True, -1)):
            def fake_detector(strip, **_kwargs):
                roi = _kwargs.get('roi', [0, 0, strip.shape[1], strip.shape[0]])
                shape = (roi[3]-roi[1], roi[2]-roi[0])
                tangent = np.zeros(shape+(2,), np.float32)
                tangent[..., 0] = 1
                return dict(bright_response=np.full(shape, .9, np.float32),
                            dark_response=np.full(shape, .8, np.float32),
                            peak_response=np.full(shape, .8, np.float32),
                            supported_mask=np.full(shape, support, bool),
                            polarity_map=np.full(shape, sign, np.int8),
                            tangent_xy=tangent, observation_mask=np.ones(shape, bool),
                            metadata=dict(method='synthetic_gate_test'))
            for method in TRACE_METHODS[:-1]:
                with patch('vision_geometry_lab.v10_seed_ridge_trace.extract_ridge_candidates',
                           side_effect=fake_detector):
                    result = trace_seed_ridges(image, seed, 9., method=method,
                                              max_length_px=300, return_maps=True)
                self.assertFalse(result['valid'])
                self.assertFalse(result['ridge_maps']['response'].any())
                self.assertFalse(result['ridge_maps']['photometric_supported'].any())
                self.assertEqual(result['metadata']['peak_gate'],
                    'method_peak_response_supported_mask_and_positive_polarity')

    def test_constant_image_oblique_border_cannot_generate_padding_ridge(self):
        image = np.full((800, 1000, 3), 100, np.uint8)
        seed = [[250., 230.], [470., 390.]]
        for method in TRACE_METHODS[:-1]:
            result = trace_seed_ridges(image, seed, 20., method=method,
                                      max_length_px=700, return_maps=True)
            self.assertFalse(result['valid'])
            self.assertFalse(result['ridge_maps']['response'].any())
            self.assertFalse(result['ridge_maps']['photometric_supported'].any())
            detector = result['metadata']['detector']
            expected = int(np.ceil(4*detector['normalization_sigma_strip_px'] +
                                  4*detector['largest_derivative_sigma_strip_px'] + 3))
            self.assertEqual(detector['image_remap_full_filter_guard_strip_px'], expected)
            self.assertFalse(detector['lower_quality_padded_context_is_emitted'])

    def test_real_halo_preserves_visible_anchor_at_core_start(self):
        yy, xx = np.indices((1000, 1400), dtype=np.float32)
        line = 430+.25*(xx-500)
        normal_distance = (yy-line)/np.sqrt(1+.25**2)
        response = np.exp(-.5*(normal_distance/5.)**2)
        image = np.repeat((40+150*response)[..., None], 3, axis=2).astype(np.uint8)
        seed = [[360., 395.], [560., 445.]]
        for method in TRACE_METHODS[:-1]:
            result = trace_seed_ridges(image, seed, 16., method=method,
                                      max_length_px=400, return_maps=True)
            self.assertTrue(result['valid'], (method, result['diagnostics']))
            # Full source context exists before u=0: do not erode the prescribed
            # core boundary and silently begin late in the free-span anchor.
            self.assertEqual(result['diagnostics']['starting_column'], 1)
            first = np.asarray(result['observed_segments'][0][0])
            along = np.dot(first-np.asarray(result['geometry']['origin_xy']),
                           result['geometry']['d_xy'])
            self.assertLess(along, 4.)
            self.assertTrue(result['ridge_maps']['observation_mask'][384, 1])
            points = np.concatenate(result['observed_segments'])
            error = abs(points[:, 1]-(430+.25*(points[:, 0]-500)))
            self.assertLess(np.median(error), 2.)

    def test_invalid_seed_width_and_method_fail_closed(self):
        image, seed = _scene()
        cases = [(seed, 0., 'steger_taylor'), ([[1., 2.]], 9., 'steger_taylor'),
                 ([[1., 2.], [1., 2.]], 9., 'steger_taylor'),
                 ([[1., 2.], [700., 2.]], 9., 'steger_taylor'), (seed, 9., 'unknown')]
        for points, width, method in cases:
            with self.assertRaises(ValueError):
                trace_seed_ridges(image, points, width, method=method)

    def test_alternatives_never_use_target_selected_best(self):
        image, seed = _scene('parallel')
        result = trace_seed_ridges(image, seed, 12., method='steger_taylor', max_length_px=300)
        self.assertLessEqual(len(result['candidates']), 3)
        if result['candidates']:
            self.assertEqual(result['observed_segments'], result['candidates'][0]['observed_segments'])
        self.assertFalse(result['identity_confirmed'])
        self.assertTrue(result['ambiguity']['not_a_calibrated_probability'])


if __name__ == '__main__':
    unittest.main()
