"""Synthetic fixed-threshold probability path tests; no model or field data."""
import json
import unittest

import numpy as np

from vision_geometry_lab.v11_path_contract import HEIGHT, WIDTH, _geometry, image_to_strip
from vision_geometry_lab.v11_path_decode import decode_path


def _geom():
    return _geometry((2000, 3000), [[700., 1000.], [900., 1000.]], 10.)


def _probability(centers=None, sigma=1.3):
    if centers is None:
        centers = 127.5+.00045*np.maximum(np.arange(WIDTH)-50, 0)**2
    output = np.exp(-.5*((np.arange(HEIGHT)[:, None]-centers[None, :])/sigma)**2)
    output /= output.sum(axis=0, keepdims=True)
    return output, centers


class PathDecodeV11Tests(unittest.TestCase):
    def test_known_probability_curve_decodes_but_not_physical_identity(self):
        probability, centers = _probability()
        geometry = _geom()
        result = decode_path(probability, None, geometry, np.ones((HEIGHT, WIDTH), bool))
        self.assertTrue(result['valid'])
        pixels = image_to_strip(np.concatenate(result['segments']), geometry)
        truth = np.interp(pixels[:, 0], np.arange(WIDTH), centers)
        self.assertLess(np.median(abs(pixels[:, 1]-truth)), .15)
        self.assertGreater(pixels[:, 0].max(), 500)
        self.assertFalse(result['identity_confirmed'])
        self.assertFalse(result['new_layer_confirmed'])
        self.assertIsNone(result['force_command'])
        json.dumps(result, allow_nan=False)

    def test_uniform_diffuse_and_zero_distributions_abstain(self):
        diffuse, _ = _probability(np.full(WIDTH, 127.5), sigma=40.)
        for probability in (np.full((HEIGHT, WIDTH), 1/HEIGHT),
                            np.zeros((HEIGHT, WIDTH)), diffuse):
            result = decode_path(probability, None, _geom(), np.ones((HEIGHT, WIDTH), bool))
            self.assertFalse(result['valid'])
            self.assertEqual(result['segments'], [])

    def test_seed_disconnected_distant_path_is_not_selected(self):
        probability, _ = _probability(np.full(WIDTH, 160.))
        result = decode_path(probability, None, _geom(), np.ones((HEIGHT, WIDTH), bool))
        self.assertFalse(result['valid'])

    def test_short_gap_recovery_is_split_not_filled(self):
        probability, _ = _probability(np.full(WIDTH, 127.5))
        probability[:, 140:142] = 0
        geometry = _geom()
        result = decode_path(probability, None, geometry, np.ones((HEIGHT, WIDTH), bool))
        self.assertTrue(result['valid'])
        self.assertGreaterEqual(len(result['segments']), 2)
        for segment in result['segments']:
            columns = image_to_strip(segment, geometry)[:, 0]
            self.assertFalse(np.any((columns > 139.5) & (columns < 141.5)))
            self.assertFalse(columns.min() < 140 and columns.max() > 141)
        self.assertGreater(image_to_strip(result['segments'][-1], geometry)[:, 0].max(), 400)
        self.assertFalse(result['diagnostics']['interpolated_gap_points_emitted'])

    def test_long_gap_stops_before_unseen_segment(self):
        probability, _ = _probability(np.full(WIDTH, 127.5))
        probability[:, 140:145] = 0
        geometry = _geom()
        result = decode_path(probability, None, geometry, np.ones((HEIGHT, WIDTH), bool))
        columns = image_to_strip(np.concatenate(result['segments']), geometry)[:, 0]
        self.assertLess(columns.max(), 140)

    def test_large_jump_to_parallel_rope_is_not_connected(self):
        centers = np.full(WIDTH, 127.5)
        centers[160:] += 15
        probability, _ = _probability(centers)
        geometry = _geom()
        result = decode_path(probability, None, geometry, np.ones((HEIGHT, WIDTH), bool))
        columns = image_to_strip(np.concatenate(result['segments']), geometry)[:, 0]
        self.assertLess(columns.max(), 160)

    def test_probability_on_invalid_pixels_cannot_be_renormalized_to_a_path(self):
        probability, _ = _probability(np.full(WIDTH, 127.5))
        valid = np.ones((HEIGHT, WIDTH), bool)
        valid[:, 300:] = False
        geometry = _geom()
        result = decode_path(probability, None, geometry, valid)
        columns = image_to_strip(np.concatenate(result['segments']), geometry)[:, 0]
        self.assertLess(columns.max(), 300)
        self.assertFalse(result['diagnostics']['probability_was_renormalized'])

    def test_narrow_valid_uniform_band_is_not_false_high_confidence(self):
        valid = np.zeros((HEIGHT, WIDTH), bool)
        valid[124:132] = True
        probability = valid.astype(float)/8
        result = decode_path(probability, None, _geom(), valid)
        self.assertFalse(result['valid'])

    def test_invalid_probability_and_slope_fail_closed(self):
        probability, _ = _probability()
        valid = np.ones((HEIGHT, WIDTH), bool)
        for bad in (np.full((HEIGHT, WIDTH), .8), np.full((HEIGHT, WIDTH), np.nan),
                    np.full((HEIGHT, WIDTH), -.1)):
            with self.assertRaises(ValueError):
                decode_path(bad, None, _geom(), valid)
        with self.assertRaises(ValueError):
            decode_path(probability, np.full(WIDTH, np.nan), _geom(), valid)


if __name__ == '__main__':
    unittest.main()
