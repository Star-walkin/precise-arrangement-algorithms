"""Synthetic V12 profile checks; no real image, holdout score or fitting loop."""
import json
import unittest

import numpy as np

from vision_geometry_lab.v12_profile_refine import (
    METHODS, PARAMETERS, _profile_layout, _profiles, _bright_candidates,
    _paired_candidates, refine_paths,
)


def _scene(kind='gaussian', offset_widths=0., width=20., mirror=False):
    height, columns = 300, 560
    yy, xx = np.indices((height, columns), dtype=np.float32)
    center = 130.+.0002*(xx-280.)**2
    distance = yy-center
    if kind == 'edge':
        amount = .5*(1+np.tanh(distance/.7))
    elif kind == 'rectangle':
        amount = .5*(np.tanh((distance+width/2)/.7)-np.tanh((distance-width/2)/.7))
    else:
        amount = np.exp(-.5*(distance/(width/2.355))**2)
    amplitude = 4. if kind == 'weak' else 155.
    image = 40.+amplitude*amount
    if kind == 'parallel':
        image += 155.*np.exp(-.5*((distance-width)/(width/2.355))**2)
    if kind == 'dark':
        image = 210.-155.*amount
    if kind == 'saturated':
        amount = .5*(np.tanh((distance+width/2)/.3)-np.tanh((distance-width/2)/.3))
        image = 40.+260.*amount
    if kind == 'texture':
        image += 22.*np.sin(xx*.8+distance*.7)*amount
    if kind == 'illumination':
        image = image*(.65+.3*xx/columns)+.10*yy
    if kind == 'asymmetric_highlight':
        body = .5*(np.tanh((distance+width/2)/.7)-np.tanh((distance-width/2)/.7))
        image = 35.+65.*body+110.*np.exp(-.5*((distance-4.)/4.5)**2)*body
    if kind == 'blank':
        image[:] = 80.
    image = np.repeat(np.clip(image, 0, 255)[..., None], 3, axis=2).astype(np.uint8)
    xs = np.linspace(80., 480., 17)
    ys = 130.+.0002*(xs-280.)**2+offset_widths*width
    segment = np.column_stack([xs, ys]).tolist()
    if mirror:
        image = image[:, ::-1].copy()
        segment = [[columns-1-x, y] for x, y in segment]
    return image, [segment]


def _error(result, mirror=False):
    points = np.concatenate(result['segments'])
    xs = 559-points[:, 0] if mirror else points[:, 0]
    return abs(points[:, 1]-(130.+.0002*(xs-280.)**2))


class ProfileRefineV12Tests(unittest.TestCase):
    def test_all_filtered_queries_have_real_gaussian_context_at_all_width_scales(self):
        gray = np.full((512, 512), 80., np.float32)
        clipped = np.zeros_like(gray)
        for width in (.5, 2., 12., 20., 28., 80.):
            for method in METHODS:
                offsets, sigma, radius, kernels = _profile_layout(width, method)
                self.assertEqual(radius, (kernels[0].shape[1]//2)*.5)
                self.assertGreaterEqual(offsets[-1], 2.2*width-1e-9)
                self.assertGreaterEqual(offsets[-1]-radius, 1.705*width+.5-1e-9)
                self.assertLessEqual(offsets[0]+radius, -1.705*width-.5+1e-9)
                profile = _profiles(gray, clipped, np.array([256., 256.]),
                                    np.array([.6, .8]), width, method)
                self.assertIsNotNone(profile)
                self.assertEqual(profile['sigma_px'], sigma)
                lo, hi = profile['supported_profile_interval_px']
                supported = (offsets >= lo) & (offsets <= hi)
                self.assertLess(float(np.max(abs(profile['first'][supported]))), 1e-5)
                self.assertLess(float(np.max(abs(profile['second'][supported]))), 1e-5)
                candidates, _ = (_bright_candidates(profile, width) if method == 'bright_ridge'
                                 else _paired_candidates(profile, width))
                self.assertFalse(candidates)
        self.assertEqual(json.loads(json.dumps(PARAMETERS)), PARAMETERS)

    def test_extended_filter_context_requires_real_source_pixels(self):
        gray = np.full((160, 200), 80., np.float32)
        # The old +/-1.8*w footprint fitted here, but complete Gaussian
        # context does not. Reject instead of reflecting nonexistent pixels.
        for method in METHODS:
            profile = _profiles(gray, np.zeros_like(gray), np.array([100., 40.]),
                                np.array([1., 0.]), 20., method)
            self.assertIsNone(profile)

    def test_correct_center_is_not_shifted_to_body_edges(self):
        image, segments = _scene()
        for method in METHODS:
            result = refine_paths(image, segments, 20., method)
            self.assertLess(np.median(_error(result)), .6)
            self.assertGreater(result['diagnostics']['accepted_points'], 90)
            self.assertFalse(result['identity_confirmed'])
            self.assertFalse(result['physical_center_confirmed'])
            self.assertTrue(result['candidate_path_available'])
            self.assertTrue(result['cv_refinement_available'])
            self.assertIn(result['status'], ('fully_refined_candidate', 'partially_refined_candidate'))
            self.assertIsNone(result['force_command'])
            json.dumps(result, allow_nan=False)

    def test_plus_minus_edge_bias_is_corrected_by_pixels_not_fixed_translation(self):
        for offset in (-.45, .45):
            image, segments = _scene(offset_widths=offset)
            for method in METHODS:
                result = refine_paths(image, segments, 20., method)
                self.assertGreater(result['diagnostics']['accepted_points'], 90, (offset, method, result['diagnostics']['reason_counts']))
                self.assertLess(np.median(_error(result)), 1.)
                accepted = [r for r in result['diagnostics']['records'] if r['accepted']]
                self.assertTrue(all(abs(r['applied_offset_px']) <= 12. for r in accepted))
                self.assertTrue(np.median([r['applied_offset_px'] for r in accepted])*offset < 0)

    def test_width_scaled_and_mirrored_refinement(self):
        for width in (12., 28.):
            image, segments = _scene(offset_widths=.4, width=width, mirror=True)
            for method in METHODS:
                result = refine_paths(image, segments, width, method)
                self.assertGreater(result['diagnostics']['accepted_points'], 80)
                self.assertLess(np.median(_error(result, mirror=True)), 1.)

    def test_periodic_strand_texture_uses_tangential_aggregation(self):
        image, segments = _scene('texture', offset_widths=.25)
        for method in METHODS:
            result = refine_paths(image, segments, 20., method)
            self.assertGreater(result['diagnostics']['accepted_points'], 70)
            self.assertLess(np.median(_error(result)), 1.5)

    def test_moderate_uneven_lighting_preserves_bounded_recovery(self):
        image, segments = _scene('illumination', offset_widths=-.4)
        for method in METHODS:
            result = refine_paths(image, segments, 20., method)
            self.assertGreater(result['diagnostics']['accepted_points'], 90)
            self.assertLess(np.median(_error(result)), 1.)

    def test_offcenter_specular_ridge_is_not_claimed_as_physical_body_center(self):
        image, segments = _scene('asymmetric_highlight')
        result = refine_paths(image, segments, 20., 'bright_ridge')
        # This is an explicit limitation, not a success-accuracy assertion:
        # an accepted optical extremum can be displaced by an off-center glint.
        self.assertGreater(result['diagnostics']['accepted_points'], 90)
        self.assertGreater(np.median(_error(result)), 2.)
        self.assertFalse(result['physical_center_confirmed'])
        self.assertFalse(result['identity_confirmed'])
        self.assertFalse(result['diagnostics']['subpixel_profile_estimate_is_physical_accuracy'])

    def test_blank_weak_single_edge_and_saturation_do_not_become_confirmed_centers(self):
        for kind in ('blank', 'weak', 'edge', 'saturated'):
            image, segments = _scene(kind)
            for method in METHODS:
                result = refine_paths(image, segments, 20., method)
                self.assertEqual(result['diagnostics']['accepted_points'], 0, (kind, method, result['diagnostics']['reason_counts']))
                self.assertTrue(all(r['input_xy'] == r['final_xy'] for r in result['diagnostics']['records']))
                self.assertFalse(result['accepted_segments'])
                self.assertTrue(result['valid'])  # Compatibility: a candidate exists, not a measurement.
                self.assertTrue(result['candidate_path_available'])
                self.assertFalse(result['cv_refinement_available'])
                self.assertEqual(result['status'], 'fallback_only')
                self.assertEqual(result['valid_meaning'], 'candidate_path_available_only_not_measured_validity')

    def test_dark_body_groove_identity_is_uncertain(self):
        image, segments = _scene('dark', offset_widths=.2)
        for method in METHODS:
            result = refine_paths(image, segments, 20., method)
            self.assertEqual(result['diagnostics']['accepted_points'], 0)
            self.assertFalse(result['identity_confirmed'])

    def test_adjacent_rope_competition_does_not_pick_an_instance(self):
        image, segments = _scene('parallel', offset_widths=.5)
        for method in METHODS:
            result = refine_paths(image, segments, 20., method)
            self.assertEqual(result['diagnostics']['accepted_points'], 0, (method, result['diagnostics']['reason_counts']))
            self.assertTrue(all(not record['accepted'] for record in result['diagnostics']['records']))

    def test_padding_and_existing_disconnected_segments_stay_uncertain_separate(self):
        image, _ = _scene()
        segments = [[[80., 8.], [180., 8.]], [[360., 8.], [480., 8.]]]
        for method in METHODS:
            result = refine_paths(image, segments, 20., method)
            self.assertEqual(len(result['segments']), 2)
            self.assertEqual(result['diagnostics']['accepted_points'], 0)
            self.assertEqual(result['segments'][0][0], segments[0][0])
            self.assertEqual(result['segments'][0][-1], segments[0][-1])
            self.assertEqual(result['segments'][1][0], segments[1][0])
            self.assertEqual(result['segments'][1][-1], segments[1][-1])
            self.assertFalse(result['diagnostics']['gaps_bridged'])

    def test_empty_and_invalid_inputs(self):
        image, segments = _scene()
        result = refine_paths(image, [], 20., 'bright_ridge')
        self.assertEqual(result['segments'], [])
        self.assertFalse(result['valid'])
        self.assertFalse(result['candidate_path_available'])
        self.assertFalse(result['cv_refinement_available'])
        self.assertEqual(result['status'], 'no_candidate_path')
        for width, method in ((0, 'bright_ridge'), (20, 'other')):
            with self.assertRaises(ValueError):
                refine_paths(image, segments, width, method)
        with self.assertRaises(ValueError):
            refine_paths(image, [[[20, 40], [20, 40]]], 20, 'paired_edges')


if __name__ == '__main__':
    unittest.main()
