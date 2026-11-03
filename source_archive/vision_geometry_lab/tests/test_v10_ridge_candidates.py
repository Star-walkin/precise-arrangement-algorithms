"""Synthetic candidate tests; no field images or target-guided parameter tuning."""
import json
import unittest

import numpy as np

from vision_geometry_lab.v10_ridge_candidates import METHODS, extract_ridge_candidates


def _gaussian_line(angle=0., offset=.3, dark=False, illumination=False, repeated=False):
    yy, xx = np.indices((192, 256), dtype=np.float32)
    tangent = np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([-tangent[1], tangent[0]])
    distance = (xx-128)*normal[0] + (yy-96)*normal[1] - offset
    body = np.exp(-.5*(distance/2.)**2)
    if repeated:
        body += np.exp(-.5*((distance-18)/2.)**2)
        body += np.exp(-.5*((distance+18)/2.)**2)
    base = 170. if dark else 40.
    image = base + (-70. if dark else 70.)*body
    if illumination:
        image = image*(.65+.35*xx/256) + 16*yy/192
    return image.astype(np.float32), tangent, normal


def _near_true(result, normal, offset=.3):
    points = result['subpixel_xy']
    inner = (points[:, 0] > 40) & (points[:, 0] < 216) & (points[:, 1] > 30) & (points[:, 1] < 162)
    points = points[inner]
    distance = (points-np.array([128, 96])) @ normal - offset
    return points[np.abs(distance) < 3], distance[np.abs(distance) < 3]


class RidgeCandidatesV10Tests(unittest.TestCase):
    def test_bright_and_dark_polarity_are_separate(self):
        for method in METHODS:
            for dark in (False, True):
                image, _, normal = _gaussian_line(dark=dark)
                output = extract_ridge_candidates(image, method=method, width_px=8,
                                                  polarity='dark' if dark else 'bright')
                points, error = _near_true(output, normal)
                self.assertGreater(len(points), 100)
                self.assertLess(np.median(abs(error)), .8)
                signs = output['subpixel_polarity']
                self.assertTrue(np.all(signs == (-1 if dark else 1)))
                self.assertFalse(output['metadata']['identity_confirmed'])
                self.assertFalse(output['metadata']['subpixel_bias_corrected'])
                json.dumps(output['metadata'], allow_nan=False)

    def test_steger_subpixel_offsets_and_rotation(self):
        for angle in (0., .36, 1.25, np.pi/2):
            for offset in (-.4, .1, .35):
                image, tangent, normal = _gaussian_line(angle, offset)
                result = extract_ridge_candidates(image, method='steger_taylor', width_px=8,
                                                  polarity='bright')
                points, error = _near_true(result, normal, offset)
                self.assertGreater(len(points), 70)
                self.assertLess(np.percentile(abs(error), 90), .15)
                directions = result['tangent_xy'][result['supported_mask']]
                self.assertGreater(np.median(abs(directions @ tangent)), .99)

    def test_broad_illumination_does_not_erase_known_ridge(self):
        for method in METHODS:
            for dark in (False, True):
                image, _, normal = _gaussian_line(.25, dark=dark, illumination=True)
                result = extract_ridge_candidates(image, method=method, width_px=8,
                                                  polarity='dark' if dark else 'bright')
                points, error = _near_true(result, normal)
                self.assertGreater(len(points), 80)
                self.assertLess(np.percentile(abs(error), 90), 1.)

    def test_flat_saturated_and_dark_images_abstain(self):
        for method in METHODS:
            for value in (0., 80., 255.):
                result = extract_ridge_candidates(np.full((128, 160), value, np.float32),
                                                  method=method, width_px=8.)
                self.assertFalse(result['supported_mask'].any())
                self.assertEqual(result['subpixel_xy'].shape, (0, 2))
                self.assertFalse(result['response'].any())

    def test_repeated_texture_stays_multiple_unidentified_ridges(self):
        image, _, normal = _gaussian_line(repeated=True)
        result = extract_ridge_candidates(image, method='steger_taylor', width_px=8,
                                          polarity='bright')
        points = result['subpixel_xy']
        distance = (points-np.array([128, 96])) @ normal
        for expected in (-17.7, .3, 18.3):
            self.assertGreater(np.count_nonzero(abs(distance-expected) < .3), 100)
        self.assertFalse(result['metadata']['connected_trajectory'])
        self.assertFalse(result['metadata']['physical_contact_confirmed'])

    def test_roi_coordinates_match_full_image_and_maps_are_finite(self):
        image, _, normal = _gaussian_line(.22)
        roi = [48, 48, 212, 148]
        for method in METHODS:
            full = extract_ridge_candidates(image, method=method, width_px=8)
            crop = extract_ridge_candidates(image, method=method, width_px=8, roi=roi)
            self.assertEqual(crop['response'].shape, (100, 164))
            np.testing.assert_allclose(crop['response'][3:-3, 3:-3],
                full['response'][51:145, 51:209], atol=2e-5)
            self.assertTrue(np.isfinite(crop['response']).all())
            self.assertTrue((crop['response'] >= 0).all())
            self.assertTrue((crop['response'] <= 1).all())
            points, error = _near_true(crop, normal)
            self.assertGreater(len(points), 70)
            self.assertLess(np.median(abs(error)), .8)

    def test_invalid_inputs_fail_closed(self):
        image = np.zeros((32, 40), np.uint8)
        cases = [dict(method='unknown', width_px=8), dict(method='steger_taylor', width_px=0),
                 dict(method='steger_taylor', width_px=8, roi=[0, 0, 41, 32]),
                 dict(method='steger_taylor', width_px=8, roi=[0, 0, 10.3, 20]),
                 dict(method='steger_taylor', width_px=8, sigmas_px=[0.]),
                 dict(method='steger_taylor', width_px=8, sigmas_px=[1., 2., 3., 4., 5.]),
                 dict(method='steger_taylor', width_px=8, polarity='rope')]
        for arguments in cases:
            with self.assertRaises(ValueError):
                extract_ridge_candidates(image, **arguments)
        with self.assertRaises(ValueError):
            extract_ridge_candidates(np.full((32, 40), np.nan), method='steger_taylor', width_px=8)

    def test_no_interpolation_across_blank_gap(self):
        image, _, _ = _gaussian_line()
        image[:, 92:164] = 40
        for method in METHODS:
            result = extract_ridge_candidates(image, method=method, width_px=8, polarity='bright')
            self.assertFalse(result['supported_mask'][90:104, 112:144].any())
            self.assertFalse(result['metadata']['cached'])


if __name__ == '__main__':
    unittest.main()
