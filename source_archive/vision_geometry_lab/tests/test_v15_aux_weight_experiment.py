"""Synthetic V15 orchestration checks; no field data or training executed."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from vision_geometry_lab import v14_joint_experiment as old
from vision_geometry_lab import v15_aux_weight_experiment as trial


RULES = dict(center_min=.90, center_max_drop=.02, synthetic_min=.90,
    each_edge_hit_min=.85, width_mean_relative_max=.25, crossing_max=.05,
    strict_human_center_bracket_min=.85, width_prediction_fraction_min=.99,
    each_edge_reference_support_min=.90)


def _evaluation(arm):
    return dict(phase='train', arm=arm, protocol_sha256='hash', checkpoint_sha256='hash', exports_sha256='hash',
        summaries=[dict(split='train', cohort='all', scope='downstream',
                        covered_fraction=.99, prediction_precision_at_5px=.99)],
        synthetic_pair_pass_fraction=1.,
        boundary_summary=dict(edges={edge: dict(hit_5px_fraction=.95) for edge in ('edge_flange', 'edge_surface')},
            width=dict(mean_absolute_relative_error=.12, trusted_columns=100, predicted_trusted_columns=100),
            joint=dict(crossing_fraction_among_available=.01, strict_center_bracket_fraction=.95)),
        rows=[dict(boundary_reference_support=dict(edges={edge:dict(scorable_fraction_of_trusted=.99)
              for edge in ('edge_flange', 'edge_surface')}))])


def _reports():
    base = {key:'same' for key in trial.MATCH_FIELDS}
    return {arm:copy.deepcopy(base) for arm in trial.ARMS}


class AuxiliaryWeightV15Tests(unittest.TestCase):
    def test_new_arm_and_references_are_separate_and_reference_training_is_rejected(self):
        self.assertEqual(trial.NEW_ARM, 'joint_low')
        self.assertEqual(trial.REFERENCE_ARMS, ('center_only', 'joint'))
        with patch.object(trial, 'verify') as verify:
            for phase, arm in (('single','center_only'), ('train','joint'), ('holdout','joint_low')):
                with self.assertRaises(ValueError):
                    trial.train(Path('does_not_exist'), phase, arm)
            verify.assert_not_called()
        for arm in trial.REFERENCE_ARMS:
            with self.assertRaises(ValueError):
                trial._check_new_evaluation_scope('train', arm)
        for arm in trial.HOLDOUT_ARMS:
            trial._check_new_evaluation_scope('holdout', arm)
        with self.assertRaises(ValueError):
            trial._check_new_evaluation_scope('holdout', 'joint')
        with self.assertRaises(ValueError):
            trial.checkpoint_source(Path('no_files'), 'holdout', 'joint')
        with self.assertRaises(ValueError):
            trial.evaluation_source(Path('no_files'), 'holdout', 'joint')
        self.assertNotEqual(trial.location(Path('new'), 'single', trial.NEW_ARM),
                            trial.location(Path('new'), 'train', trial.NEW_ARM))

    def test_cannot_write_into_old_v14_tree(self):
        with self.assertRaises(ValueError):
            trial._new_root(old.BASE)
        with self.assertRaises(ValueError):
            trial._new_root(old.BASE/'nested_new_trial')

    def test_all_matched_report_fields_required_including_versions_and_sequence(self):
        trial.assert_matched_reports(_reports())
        for key in trial.MATCH_FIELDS:
            reports = _reports(); reports['joint_low'][key] = 'changed'
            with self.assertRaisesRegex(ValueError, key):
                trial.assert_matched_reports(reports)
        reports = _reports(); reports['joint']['initial_state_sha256'] = 'different_reference'
        with self.assertRaises(ValueError):
            trial.assert_matched_reports(reports)

    def test_gate_checks_exactly_match_original_v14_for_same_comparison(self):
        for failure in ('none', 'center', 'precision', 'width', 'support', 'synthetic', 'edge'):
            values = {arm:_evaluation(arm) for arm in trial.ARMS}
            if failure == 'center': values['joint_low']['summaries'][0]['covered_fraction'] = .969
            elif failure == 'precision': values['joint_low']['summaries'][0]['prediction_precision_at_5px'] = .89
            elif failure == 'width': values['joint_low']['boundary_summary']['width']['mean_absolute_relative_error'] = .251
            elif failure == 'support': values['joint_low']['rows'][0]['boundary_reference_support']['edges']['edge_surface']['scorable_fraction_of_trusted'] = .89
            elif failure == 'synthetic': values['joint_low']['synthetic_pair_pass_fraction'] = .89
            elif failure == 'edge': values['joint_low']['boundary_summary']['edges']['edge_surface']['hit_5px_fraction'] = .84
            new = trial.matched_gate_decision(values, _reports(), RULES)
            # Feed exactly the same baseline/new values through the unchanged
            # old gate, mapping the new arm to its original 'joint' role.
            old_values = dict(center_only=values['center_only'], joint=values['joint_low'])
            with patch.object(old, 'verify', return_value={'gate':RULES}), \
                 patch.object(old, 'read', side_effect=lambda path:old_values[Path(path).parent.name]), \
                 patch.object(old, 'checked_report', side_effect=lambda output,phase,arm:_reports()['center_only']), \
                 patch.object(old, 'sha', return_value='hash'):
                legacy = old.gate_decision(Path('synthetic_trial'), 'train')
            self.assertEqual(new['checks'], legacy['checks'])
            self.assertEqual(new['passed'], legacy['passed'])
            self.assertFalse(new['deployment_allowed'])

    def test_joint_point_five_is_context_not_gate_driver(self):
        values = {arm:_evaluation(arm) for arm in trial.ARMS}
        values['joint']['summaries'][0]['covered_fraction'] = .01
        values['joint']['synthetic_pair_pass_fraction'] = 0.
        decision = trial.matched_gate_decision(values, _reports(), RULES)
        self.assertTrue(decision['passed'])
        self.assertEqual(decision['center']['joint']['covered_fraction'], .01)
        self.assertIn('context only', decision['comparison'])

    def test_holdout_loader_calls_gate_before_loading_any_arrays(self):
        with patch.object(trial, 'require_gate', side_effect=ValueError('failed train gate')) as gate, \
             patch.object(old, 'load_inputs') as loader:
            with self.assertRaisesRegex(ValueError, 'failed train gate'):
                trial.load_inputs(Path('new_trial'), 'holdout')
            gate.assert_called_once_with(Path('new_trial'), 'train')
            loader.assert_not_called()
        with patch.object(trial, 'require_gate') as gate, \
             patch.object(old, 'load_inputs', return_value=['scoped']) as loader:
            self.assertEqual(trial.load_inputs(Path('new_trial'), 'holdout'), ['scoped'])
            gate.assert_called_once()
            loader.assert_called_once_with(Path('new_trial'), 'holdout', False)

    def test_checkpoint_descriptor_preserves_reference_original_protocol(self):
        p = dict(reference_root='old_trial', edge_weight=dict(center_only=0.,joint=.5,joint_low=.1))
        report = dict(checkpoint_sha256='cp_hash', steps=2400)
        with patch.object(trial, 'read', return_value=p), \
             patch.object(trial, 'checked_report', return_value=report), \
             patch.object(trial, 'sha', side_effect=lambda path:'hash:'+str(path)):
            old_source = trial.checkpoint_source(Path('new_trial'), 'holdout', 'center_only')
            new_source = trial.checkpoint_source(Path('new_trial'), 'holdout', 'joint_low')
        self.assertEqual(old_source['checkpoint_protocol'], str(Path('old_trial/protocol.json')))
        self.assertEqual(new_source['checkpoint_protocol'], str(Path('new_trial/protocol.json')))
        self.assertIn('reused', old_source['origin'])
        self.assertIn('new_V15', new_source['origin'])
        self.assertEqual(old_source['training_phase'], 'train')

    def test_manifest_snapshot_outputs_are_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'artifact.json'
            trial.write_json(path, dict(synthetic=True))
            with self.assertRaises(FileExistsError):
                trial.write_json(path, dict(synthetic=False))

    def test_reuse_array_comparison_is_fixed_finite_and_shape_sensitive(self):
        source = np.array([[.1, .5], [.7, 1.]], np.float32)
        same = trial.compare_reused_array(source, source.copy())
        self.assertTrue(same['passed'])
        self.assertTrue(same['bit_equal'])
        close = trial.compare_reused_array(source+np.float32(2e-8), source)
        self.assertTrue(close['passed'])
        self.assertFalse(trial.compare_reused_array(source+.01, source)['passed'])
        self.assertFalse(trial.compare_reused_array(source[:1], source)['passed'])
        self.assertFalse(trial.compare_reused_array(np.full_like(source, np.nan), source)['passed'])
        self.assertEqual(same['rtol'], 1e-6)
        self.assertEqual(same['atol'], 1e-7)

    def test_gate_must_first_require_a_passed_reuse_audit(self):
        with patch.object(trial, 'verify', return_value={'gate':RULES}), \
             patch.object(trial, 'require_reuse_audit', side_effect=ValueError('audit missing or failed')) as audit, \
             patch.object(trial, 'evaluation_source') as evaluation:
            with self.assertRaisesRegex(ValueError, 'audit missing or failed'):
                trial.gate_decision(Path('new_trial'), 'single')
            audit.assert_called_once()
            evaluation.assert_not_called()


if __name__ == '__main__':
    unittest.main()
