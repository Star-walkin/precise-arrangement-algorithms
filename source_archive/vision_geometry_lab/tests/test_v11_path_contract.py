"""Synthetic V11 coordinate/supervision regressions; no field data."""
import json
import unittest

import numpy as np

from vision_geometry_lab.v11_path_contract import (
    HEIGHT, WIDTH, _geometry, array_sha256, build_input, build_targets,
    image_to_strip, strip_to_image,
)


def _geom():
    return _geometry((2000, 3000), [[700., 1000.], [900., 1000.]], 10.)


class PathContractV11Tests(unittest.TestCase):
    def test_pixel_center_mapping_roundtrip_and_seed_origin(self):
        geometry = _geom()
        pixels = np.array([[0., 0.], [511., 255.], [42.1666666666667, 127.5], [201.3, 91.7]])
        native = strip_to_image(pixels, geometry)
        np.testing.assert_allclose(native[0], [900-79.0625, 1000-239.0625], atol=1e-10)
        np.testing.assert_allclose(native[2], [900., 1000.], atol=1e-10)
        np.testing.assert_allclose(image_to_strip(native, geometry), pixels, atol=1e-10)
        json.dumps(geometry, allow_nan=False)

    def test_rotation_and_width_are_input_geometry_not_fixed_pixel_crop(self):
        first = _geom()
        rotated = _geometry((3000, 3000), [[1100., 500.], [1100., 700.]], 20.)
        point = [[70., 91.]]
        offset1 = strip_to_image(point, first)[0]-first['origin_xy']
        offset2 = strip_to_image(point, rotated)[0]-rotated['origin_xy']
        np.testing.assert_allclose(offset2, 2*np.array([-offset1[1], offset1[0]]))

    def test_eight_channels_rgb_order_and_float_shapes(self):
        image = np.zeros((700, 1000, 3), np.uint8)
        image[..., 0], image[..., 1], image[..., 2] = 40, 100, 180
        data = build_input(image, [[150., 330.], [250., 330.]], 6.)
        self.assertEqual(data['input'].shape, (8, HEIGHT, WIDTH))
        self.assertEqual(data['input'].dtype, np.float32)
        self.assertEqual(data['valid_mask'].dtype, np.bool_)
        valid = data['valid_mask']
        for index, value in enumerate((40, 100, 180)):
            np.testing.assert_allclose(data['input'][index, valid], value/255., atol=1e-6)
        self.assertGreater(float(data['input'][4].max()), .5)
        self.assertFalse(data['geometry']['target_or_quality_used'])

    def test_constant_image_diagonal_padding_cannot_create_normalized_ridge(self):
        image = np.full((500, 650, 3), 107, np.uint8)
        result = build_input(image, [[80., 90.], [230., 180.]], 20.)
        valid, tensor = result['valid_mask'], result['input']
        self.assertTrue(valid.any())
        self.assertTrue((~valid).any())
        self.assertLess(float(abs(tensor[3, valid]).max()), 1e-4)
        self.assertFalse(tensor[:5, ~valid].any())
        np.testing.assert_array_equal(tensor[7], valid.astype(np.float32))

    def test_targets_are_unique_column_distributions_and_tail_unknown(self):
        geometry = _geom()
        line = strip_to_image([[25., 110.], [200., 127.5], [420., 149.5]], geometry)
        result = build_targets(geometry, line, [], np.ones((HEIGHT, WIDTH), bool))
        self.assertEqual(int(result['column_valid'].sum()), 396)
        np.testing.assert_allclose(result['target_distribution'][:, 25:421].sum(0), 1., atol=1e-6)
        self.assertFalse(result['target_distribution'][:, 421:].any())
        self.assertFalse(result['row_valid'][:, :25].any())
        self.assertFalse(result['row_valid'][:, 421:].any())
        self.assertFalse(result['slope_valid'][421:].any())
        np.testing.assert_allclose(result['slope_target'][result['slope_valid']], .1, atol=1e-6)
        expectation = np.arange(HEIGHT)@result['target_distribution'][:, 200]
        self.assertAlmostEqual(expectation, 127.5, places=4)
        self.assertFalse(result['diagnostics']['unknown_is_background'])

    def test_ignore_region_and_its_neighborhood_are_not_negative_supervision(self):
        geometry = _geom()
        line = strip_to_image([[20., 127.5], [480., 127.5]], geometry)
        polygon = strip_to_image([[150., 120.], [210., 120.], [210., 135.], [150., 135.]], geometry)
        result = build_targets(geometry, line, [polygon], np.ones((HEIGHT, WIDTH), bool))
        self.assertFalse(result['column_valid'][150:211].any())
        self.assertFalse(result['row_valid'][:, 150:211].any())
        self.assertFalse(result['target_distribution'][:, 150:211].any())
        self.assertTrue(result['column_valid'][140])
        self.assertTrue(result['column_valid'][220])

    def test_multiple_intersections_are_unknown_not_nearest_target_selection(self):
        geometry = _geom()
        line = strip_to_image([[30., 90.], [300., 120.], [30., 160.]], geometry)
        result = build_targets(geometry, line, [], np.ones((HEIGHT, WIDTH), bool))
        self.assertFalse(result['column_valid'][30:300].any())
        self.assertGreater(result['diagnostics']['ambiguous_columns'], 200)
        self.assertFalse(result['target_distribution'][:, 30:300].any())

    def test_target_changes_do_not_change_input_or_roi(self):
        image = np.full((600, 1000, 3), 90, np.uint8)
        data = build_input(image, [[100., 290.], [250., 290.]], 6.)
        original = array_sha256(data['input'])
        a = strip_to_image([[30., 127.5], [300., 140.]], data['geometry'])
        b = strip_to_image([[30., 110.], [200., 95.]], data['geometry'])
        build_targets(data['geometry'], a, [], data['valid_mask'])
        build_targets(data['geometry'], b, [], data['valid_mask'])
        self.assertEqual(original, array_sha256(data['input']))

    def test_empty_and_padded_target_are_all_unknown(self):
        geometry = _geom()
        line = strip_to_image([[20., 127.5], [480., 127.5]], geometry)
        for centerline, valid in (([], np.ones((HEIGHT, WIDTH), bool)),
                                  (line, np.zeros((HEIGHT, WIDTH), bool))):
            result = build_targets(geometry, centerline, [], valid)
            for key in ('target_distribution', 'column_valid', 'row_valid', 'slope_target', 'slope_valid'):
                self.assertFalse(result[key].any())

    def test_invalid_input_fails_closed(self):
        image = np.zeros((50, 60, 3), np.uint8)
        cases = [dict(free_seed=[[1, 2]], width_px=10),
                 dict(free_seed=[[1, 2], [1, 2]], width_px=10),
                 dict(free_seed=[[1, 2], [70, 2]], width_px=10),
                 dict(free_seed=[[1, 2], [30, 2]], width_px=0)]
        for arguments in cases:
            with self.assertRaises(ValueError):
                build_input(image, **arguments)
        with self.assertRaises(ValueError):
            build_input(image.astype(float), [[1, 2], [30, 2]], 10)


if __name__ == '__main__':
    unittest.main()
