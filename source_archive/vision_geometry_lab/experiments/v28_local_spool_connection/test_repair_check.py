"""Synthetic checks of post-save repair diagnostics; no field claims or labels."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import check_repair as check
from . import pipeline


def fixture():
    reference = dict(sample_id='synthetic', centerline=[[20., 60.], [60., 60.], [100., 60.], [100., 180.]],
                     free_seed=[[20., 60.], [60., 60.]],
                     edge_flange=[[20., 50.], [110., 50.], [110., 180.]],
                     edge_surface=[[20., 70.], [90., 70.], [90., 180.]],
                     width_px=20., ignore_polygons=[], reviewed=True, status='partial')
    points = pipeline.resample(reference['centerline'], 4.).tolist()
    count = len(points)
    candidate = dict(id='A', kind='synthetic_geometric_completion', points_xy=points,
                     supported=[False]*count, strict_body_supported=[False]*count,
                     raw_pair_supported=[True]*count,
                     inferred=[True]*count, unavailable=[False]*count)
    pool = dict(candidates=[candidate], origin_xy=[60., 60.], direction_xy=[1., 0.],
                width_px=20., image_shape_hw=[240, 240], input_mode='AUTO')
    return reference, pool


class RepairCheckTests(unittest.TestCase):
    def write_pool(self, folder, pool):
        target = Path(folder) / 'pool.json'
        target.write_text(json.dumps(pool), encoding='utf-8')
        return target

    def test_missing_or_changed_saved_pool_never_loads_reference(self):
        _, pool = fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'pool.json'
            with patch.object(check.data_eval, 'load_evaluation_reference', side_effect=AssertionError('label opened early')):
                with self.assertRaises(FileNotFoundError):
                    check.evaluate_saved_pool('synthetic', path, {})
                self.write_pool(folder, pool)
                validate = check._validate_candidate
                def change_after_first_read(candidate):
                    result = validate(candidate)
                    path.write_text(json.dumps(dict(pool, changed=True)), encoding='utf-8')
                    return result
                with patch.object(check, '_validate_candidate', side_effect=change_after_first_read):
                    with self.assertRaisesRegex(ValueError, 'Saved candidate pool differs'):
                        check.evaluate_saved_pool('synthetic', path, {})

    def test_geometric_coverage_does_not_become_strict_body_coverage(self):
        reference, pool = fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_pool(folder, pool)
            before = path.read_bytes()
            with patch.object(check.data_eval, 'load_evaluation_reference', return_value=reference):
                result = check.evaluate_saved_pool('synthetic', path, {})
            self.assertEqual(path.read_bytes(), before)
        self.assertEqual(result['best_bend5px_coverage'], 1.)
        self.assertEqual(result['best_supported_only_bend5px_coverage'], 0.)
        self.assertTrue(result['candidate_pool_contains_trusted_bend_geometry'])
        self.assertFalse(result['candidate_pool_contains_strict_supported_bend'])
        self.assertEqual(result['candidates'][0]['raw_pair_supported_fraction'], 1.)
        self.assertEqual(result['candidates'][0]['strict_body_supported_fraction'], 0.)
        self.assertEqual(result['tolerance_px'], 5.)
        self.assertEqual(result['coverage_threshold'], .8)
        self.assertFalse(result['ranking_performed'])

    def test_strict_support_never_bridges_a_missing_sample(self):
        points = [[0., 0.], [4., 0.], [8., 0.], [12., 0.], [16., 0.]]
        self.assertEqual(check.strict_supported_segments(points, [True, True, False, True, True]),
                         [points[:2], points[3:]])
        self.assertEqual(check.strict_supported_segments(points, [True, False, True, False, True]), [])

    def test_unknown_region_is_not_background_and_absent_strict_field_is_unknown(self):
        reference, pool = fixture()
        pool['candidates'][0].pop('strict_body_supported')
        mask = np.ones((240, 240), bool)
        regions = dict(explicit_background=mask.copy(), unknown_mask=mask.copy(),
                       search_mask=np.zeros_like(mask))
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_pool(folder, pool)
            with patch.object(check.data_eval, 'load_evaluation_reference', return_value=reference):
                result = check.evaluate_saved_pool('synthetic', path, regions)
        row = result['candidates'][0]
        self.assertEqual(row['explicit_background_fraction'], 0.)
        self.assertEqual(row['outside_search_fraction'], 1.)
        self.assertEqual(row['unknown_region_fraction'], 1.)
        self.assertIsNone(row['supported_only_bend5px_coverage'])
        self.assertIsNone(result['best_supported_only_bend5px_coverage'])
        self.assertFalse(result['unknown_is_background'])

    def test_inconsistent_support_partition_rejected_before_labels(self):
        _, pool = fixture()
        pool['candidates'][0]['supported'][0] = True
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_pool(folder, pool)
            with patch.object(check.data_eval, 'load_evaluation_reference', side_effect=AssertionError('label opened early')):
                with self.assertRaisesRegex(ValueError, 'disjoint complete partition'):
                    check.evaluate_saved_pool('synthetic', path, {})

    def test_wrong_coordinate_mask_shape_rejected_before_labels(self):
        _, pool = fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_pool(folder, pool)
            with patch.object(check.data_eval, 'load_evaluation_reference', side_effect=AssertionError('label opened early')):
                with self.assertRaisesRegex(ValueError, 'original image coordinates'):
                    check.evaluate_saved_pool('synthetic', path, dict(explicit_background=np.zeros((12, 12), bool)))


if __name__ == '__main__':
    unittest.main()
