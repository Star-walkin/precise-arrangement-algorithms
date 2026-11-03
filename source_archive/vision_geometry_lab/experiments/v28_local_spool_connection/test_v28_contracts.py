"""V28 safety and evidence contracts, using synthetic data only.

These tests do not establish field accuracy and never load repository images,
annotations, models, videos, or equipment interfaces.
"""
from __future__ import annotations

import hashlib
import importlib
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
PREFIX = 'vision_geometry_lab.experiments.v28_local_spool_connection'
REUSED_HASHES = {
    'rope_detector.py': '6d90427900284332f032efd06abd2951d23b8b08df32e6a47a3fc4dd8bcdebf0',
    'arc_detector.py': 'd0cde99802a3fe6af23736878927ace8fe90adac92a2d09b1407bb6085953bae',
    'geometry.py': '1557b81ac00a851785d62427195be46e5b92ff5d6eff1c3e74f80f6603668992',
    'wound_rope_tracker.py': '1b4c9d803bafbfed4433d5a2e96441b8afdae4ba08c9e6fd9e741d5fc578d11a',
    'v16_video_measurement.py': 'dddd0a4c6faeedd1cc58e2fcac163ac5c1d256bb1cea8a5a6c8ffb5ccf7017aa',
}


class ImportAndIsolationTests(unittest.TestCase):
    def test_reused_sources_unchanged(self):
        for filename, expected in REUSED_HASHES.items():
            with self.subTest(filename=filename):
                actual = hashlib.sha256((ROOT / 'vision_geometry_lab' / filename).read_bytes()).hexdigest()
                self.assertEqual(actual, expected)

    def test_imports_do_not_access_capture_models_or_hardware(self):
        program = r'''
import builtins, importlib, sys
import cv2
def forbidden(*args, **kwargs):
    raise AssertionError('import attempted video/device access')
cv2.VideoCapture = forbidden
original_import = builtins.__import__
blocked = ('serial', 'pymodbus', 'pyads', 'pypylon', 'MvCameraControl',
           'MVS', 'camera_controller', 'plc_controller', 'torch')
def guarded(name, *args, **kwargs):
    if any(name == item or name.startswith(item + '.') for item in blocked):
        raise AssertionError('forbidden dependency: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
for name in (
    'vision_geometry_lab.rope_detector',
    'vision_geometry_lab.arc_detector',
    'vision_geometry_lab.geometry',
    'vision_geometry_lab.v16_video_measurement',
    'vision_geometry_lab.experiments.v28_local_spool_connection.structure',
    'vision_geometry_lab.experiments.v28_local_spool_connection.pipeline',
    'vision_geometry_lab.experiments.v28_local_spool_connection.data_eval',
    'vision_geometry_lab.experiments.v28_local_spool_connection.run_experiment',
):
    importlib.import_module(name)
from vision_geometry_lab.v16_video_measurement import automatic_seed
seed, reason = automatic_seed(
    dict(valid=True, width_px=8., normal=[0., 1.], observed_segment=[[10., 30.], [80., 30.]]),
    dict(points=[[110., 35.], [140., 35.]]))
assert seed[-1] == [80., 30.]
print('safe_imports_ok')
'''
        result = subprocess.run([sys.executable, '-c', program], cwd=ROOT,
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('safe_imports_ok', result.stdout)

    def test_auto_inventory_does_not_return_or_read_label_fields(self):
        ev = importlib.import_module(PREFIX + '.data_eval')
        sample = dict(sample_id='synthetic', group_id='test', image_path='no_image_opened',
                      image_sha256='test', image_shape_hw=[10, 10], source={},
                      annotation_path='must_not_open', annotation_sha256='secret',
                      centerline='must_not_expose', free_seed='must_not_expose')
        with patch.object(ev, '_samples', return_value=[sample]), \
             patch.object(ev, 'sha256', side_effect=AssertionError('no image or label IO requested')):
            rows = ev.list_inputs(verify_images=False)
        self.assertEqual(set(rows[0]), {'sample_id', 'group_id', 'image_path',
                                       'image_sha256', 'image_shape_hw', 'source'})


def paired_scene():
    image = np.full((260, 420, 3), 45, np.uint8)
    image[90:111, 20:400] = 200
    return image


def synthetic_pool(image):
    pipeline = importlib.import_module(PREFIX + '.pipeline')
    paths = [pipeline.resample([[40., 100.], [360., 100.]]),
             pipeline.resample([[40., 100.], [120., 100.], [360., 160.]])]
    candidates = []
    for cid, path, join in zip(('A', 'B'), paths, ([100., 100.], [220., 125.])):
        widths = np.full(len(path), 20.)
        evidence = pipeline.profile_evidence(image, path, widths)
        candidates.append(dict(id=cid, kind='synthetic_competing_connection',
                               points_xy=path.tolist(), widths_px=widths.tolist(),
                               join_xy=join, evidence=evidence,
                               supported=evidence['supported'], inferred=evidence['inferred'],
                               unavailable=evidence['unavailable']))
    return dict(candidates=candidates, origin_xy=[40., 100.], direction_xy=[1., 0.],
                normal_xy=[0., 1.], width_px=20., free_seed=[[10., 100.], [40., 100.]])


class GeometryAndEvidenceTests(unittest.TestCase):
    def test_rotated_strip_original_coordinate_roundtrip(self):
        structure = importlib.import_module(PREFIX + '.structure')
        uv = np.array([[-4.7, 2.3], [0., 0.], [231.2, -67.3]])
        for direction in ([1., 0.], [0., -1.], [-.7, .3]):
            with self.subTest(direction=direction):
                xy = structure.strip_to_original(uv, [313.25, 227.75], direction)
                np.testing.assert_allclose(structure.original_to_strip(xy, [313.25, 227.75], direction), uv, atol=1e-10)
                np.testing.assert_allclose(xy[1], [313.25, 227.75], atol=1e-10)

    def test_blank_structure_has_no_supported_arcs_and_preserves_image(self):
        structure = importlib.import_module(PREFIX + '.structure')
        image = np.full((240, 360, 3), 128, np.uint8)
        before = image.copy()
        result = structure.extract_local_structure(image, [60., 120.], [1., 0.], 8.,
                                                   roi_xyxy=[20, 30, 340, 220])
        self.assertEqual(result['structures'], [])
        self.assertEqual(result['status'], 'NO_SUPPORTED_LOCAL_STRUCTURE')
        np.testing.assert_array_equal(image, before)
        x0, y0, x1, y1 = result['roi_xyxy']
        self.assertTrue(20 <= x0 < x1 <= 340 and 30 <= y0 < y1 <= 220)

    def test_missing_or_saturated_boundary_never_becomes_supported(self):
        pipeline = importlib.import_module(PREFIX + '.pipeline')
        points = np.column_stack((np.linspace(70, 330, 31), np.full(31, 100.)))
        for value in (0, 128, 255):
            image = np.full((240, 400, 3), value, np.uint8)
            with self.subTest(value=value):
                evidence = pipeline.profile_evidence(image, points, 20.)
                self.assertFalse(any(evidence['supported']))
                self.assertFalse(any(evidence['available']))
                self.assertTrue(all(x is None for x in evidence['measured_widths_px']))
        one_edge = np.full((240, 400, 3), 40, np.uint8)
        one_edge[100:] = 200
        evidence = pipeline.profile_evidence(one_edge, points, 20.)
        self.assertFalse(any(evidence['supported']))
        self.assertTrue(all(x is None for x in evidence['measured_widths_px']))

    def test_bilateral_profiles_support_both_polarities_without_image_edits(self):
        pipeline = importlib.import_module(PREFIX + '.pipeline')
        points = np.column_stack((np.linspace(70, 330, 31), np.full(31, 100.)))
        for dark in (False, True):
            image = np.full((240, 400, 3), 200 if dark else 40, np.uint8)
            image[90:111] = 40 if dark else 200
            before = image.copy()
            evidence = pipeline.profile_evidence(image, points, 20.)
            self.assertTrue(all(evidence['supported']))
            self.assertTrue(all(evidence['available']))
            self.assertEqual(set(evidence['polarity']), {'dark_body_or_gap' if dark else 'bright_body'})
            np.testing.assert_array_equal(image, before)

    def test_scores_share_pool_are_order_invariant_and_use_only_distinct_region(self):
        pipeline = importlib.import_module(PREFIX + '.pipeline')
        image = paired_scene()
        pool = synthetic_pool(image)
        before = copy.deepcopy(pool)
        first = pipeline.score_pool(image, pool)
        self.assertEqual(pool, before)
        self.assertEqual({row['pool_sha256'] for row in first.values()}, {pipeline.pool_hash(pool)})
        reverse = copy.deepcopy(pool)
        reverse['candidates'].reverse()
        second = pipeline.score_pool(image, reverse)
        for group in ('G0', 'G1', 'G2'):
            self.assertEqual(first[group]['best_id'], second[group]['best_id'])
            self.assertEqual(first[group]['selected_id'], second[group]['selected_id'])
            self.assertEqual(first[group]['ordered'], second[group]['ordered'])
        for candidate in pool['candidates']:
            mask = np.asarray(first['G2']['distinguishing_masks'][candidate['id']])
            points = np.asarray(candidate['points_xy'])
            self.assertFalse(mask[points[:, 0] <= 120.].any(), 'shared free span must not count as distinguishing evidence')
            self.assertTrue(mask.any())

    def test_frozen_claimed_support_cannot_validate_itself_on_erased_original(self):
        pipeline = importlib.import_module(PREFIX + '.pipeline')
        pool = synthetic_pool(paired_scene())
        for candidate in pool['candidates']:
            candidate['supported'] = [True] * len(candidate['points_xy'])
        blank = np.full((260, 420, 3), 128, np.uint8)
        result = pipeline.score_pool(blank, pool)['G2']
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertIsNone(result['selected_id'])
        self.assertIn('insufficient_usable_original_evidence', result['reason'])
        self.assertTrue(all(r['components']['available_fraction'] == 0 for r in result['ordered']))

    def test_new_segment_can_cross_old_row_and_auto_never_reads_labels(self):
        pipeline = importlib.import_module(PREFIX + '.pipeline')
        ev = importlib.import_module(PREFIX + '.data_eval')
        image = np.full((260, 420, 3), 45, np.uint8)
        cv2.line(image, (10, 100), (400, 100), (200, 200, 200), 20)
        cv2.line(image, (80, 40), (350, 180), (180, 180, 180), 20)
        before = image.copy()
        local = {'structures': [{'id': 'S1', 'points_xy':
                 [[80., 40.], [150., 76.], [220., 112.], [290., 148.], [350., 180.]]}]}
        with patch.object(ev, 'load_evaluation_reference', side_effect=AssertionError('AUTO tried target label')), \
             patch.object(ev, 'load_conditional_seed', side_effect=AssertionError('AUTO tried conditional hint')):
            pool = pipeline.generate_pool(image, [[10., 100.], [40., 100.]], 20., local)
            pipeline.score_pool(image, pool)
        self.assertTrue(any(c['kind'] == 'new_segment_may_cross_old_rows' for c in pool['candidates']))
        self.assertTrue(any(c['kind'] == 'old_row_connection' for c in pool['candidates']))
        self.assertLessEqual(len(pool['candidates']), pipeline.DEFAULT_CONFIG['max_candidates'])
        self.assertFalse(pool['input_uses_labels'])
        np.testing.assert_array_equal(image, before)


def reference_fixture():
    reference = dict(sample_id='synthetic', centerline=[[20., 60.], [60., 60.], [100., 60.], [100., 180.]],
                     free_seed=[[20., 60.], [60., 60.]],
                     edge_flange=[[20., 50.], [110., 50.], [110., 180.]],
                     edge_surface=[[20., 70.], [90., 70.], [90., 180.]],
                     width_px=20., ignore_polygons=[], reviewed=True, status='partial')
    pool = dict(candidates=[dict(id='A', kind='test', points_xy=reference['centerline']),
                            dict(id='B', kind='test', points_xy=[[20., 60.], [125., 60.], [125., 180.]])])
    rankings = {g: dict(best_id='A', selected_id='A', status='SINGLE_CANDIDATE', reason='test')
                for g in ('G0', 'G1', 'G2')}
    return reference, pool, rankings


class EvaluationBoundaryTests(unittest.TestCase):
    def test_reviewed_ribbon_with_separate_branch_can_distinguish_right_and_wrong(self):
        ev = importlib.import_module(PREFIX + '.data_eval')
        reference, pool, rankings = reference_fixture()
        result = ev.evaluate_reference(reference, pool, rankings)
        by_id = {row['id']: row for row in result['candidate_scores']}
        self.assertTrue(result['visible_connection_evaluable'])
        self.assertEqual(by_id['A']['visible_connection_status'], 'CORRECT')
        self.assertEqual(by_id['B']['visible_connection_status'], 'WRONG')
        self.assertFalse(result['physical_identity_confirmed'])

    def test_successful_entry_with_empty_pool_is_not_an_entry_failure(self):
        ev = importlib.import_module(PREFIX + '.data_eval')
        reference, _, rankings = reference_fixture()
        for choice in rankings.values():
            choice.update(best_id=None, selected_id=None, status='UNKNOWN')
        result = ev.evaluate_reference(reference, dict(candidates=[], automatic_entry_success=True), rankings)
        self.assertTrue(result['automatic_entry_success'])
        self.assertFalse(result['candidate_pool_contains_trusted_bend_geometry'])

    def test_unknown_tail_is_not_positive_or_negative_reference(self):
        ev = importlib.import_module(PREFIX + '.data_eval')
        reference, pool, rankings = reference_fixture()
        reference['ignore_polygons'] = [[[0., 120.], [220., 120.], [220., 240.], [0., 240.]]]
        result = ev.evaluate_reference(reference, pool, rankings)
        self.assertAlmostEqual(result['reference_length_px'], 100., places=6)
        changed = copy.deepcopy(pool)
        changed['candidates'][0]['points_xy'][-1] = [170., 220.]
        # Change only the excluded tail, keeping the trusted portion identical.
        changed['candidates'][0]['points_xy'] = [[20., 60.], [60., 60.], [100., 60.], [100., 120.], [170., 220.]]
        other = ev.evaluate_reference(reference, changed, rankings)
        self.assertEqual(result['choices']['G0']['bend_geometry_5px'], other['choices']['G0']['bend_geometry_5px'])
        self.assertEqual(result['choices']['G0']['selected_visible_connection_status'], other['choices']['G0']['selected_visible_connection_status'])

    def test_centerline_nearness_without_reviewed_both_edges_is_unknown(self):
        ev = importlib.import_module(PREFIX + '.data_eval')
        reference, pool, rankings = reference_fixture()
        reference['edge_surface'] = []
        result = ev.evaluate_reference(reference, pool, rankings)
        self.assertFalse(result['visible_connection_evaluable'])
        self.assertEqual(result['choices']['G0']['selected_visible_connection_status'], 'UNKNOWN')
        self.assertFalse(result['physical_identity_confirmed'])

    def test_unsaved_or_changed_pool_is_rejected_before_any_label_load(self):
        ev = importlib.import_module(PREFIX + '.data_eval')
        reference, pool, rankings = reference_fixture()
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'pool.json'
            with patch.object(ev, 'load_evaluation_reference', side_effect=AssertionError('GT opened before pool validation')):
                with self.assertRaises(FileNotFoundError):
                    ev.evaluate_case('synthetic', pool, rankings, persisted_pool_path=target)
                target.write_text(json.dumps(dict(candidates=[])), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Saved candidate pool differs'):
                    ev.evaluate_case('synthetic', pool, rankings, persisted_pool_path=target)


class RunnerIsolationTests(unittest.TestCase):
    def test_auto_detector_does_not_call_conditional_or_target_loaders(self):
        runner = importlib.import_module(PREFIX + '.run_experiment')
        image = np.zeros((80, 120, 3), np.uint8)
        with patch.object(runner.data_eval, 'load_conditional_seed', side_effect=AssertionError('conditional leakage')), \
             patch.object(runner.data_eval, 'load_evaluation_reference', side_effect=AssertionError('GT leakage')):
            entry = runner.detect_input(image, 'synthetic_not_in_manifest', 'AUTO')
        self.assertFalse(entry['available'])
        self.assertIsNone(entry['free_seed'])
        self.assertEqual(entry['mode'], 'AUTO')

    def test_output_directory_and_files_are_exclusive(self):
        runner = importlib.import_module(PREFIX + '.run_experiment')
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'new_run'
            runner.reserve_output(output)
            target = output / 'pool.json'
            runner.write_json(target, {'sentinel': 'original'})
            before = target.read_bytes()
            with self.assertRaises(FileExistsError):
                runner.reserve_output(output)
            with self.assertRaises(FileExistsError):
                runner.write_json(target, {'sentinel': 'replacement'})
            self.assertEqual(target.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
