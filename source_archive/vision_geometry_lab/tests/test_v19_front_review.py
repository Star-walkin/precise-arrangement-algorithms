import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from vision_geometry_lab.v19_front_review import (
    UNKNOWN, LOW, MID, HIGH, WARMUP, direct_state_matrix, quality_summary,
    score_candidates, score_from_base, timeline_figure, validate_prediction,
)


def candidate(onset=5.1, report=5.3, name='candidate_1'):
    return dict(candidate_id=name, estimated_onset_s=onset, first_available_s=report,
        confirmation_delay_s=report-onset, estimated_onset=onset, first_available=report,
        delay=report-onset, last_low_first_high_interval_s=[onset-.1, onset],
        lastlow_firsthigh_interval=[onset-.1, onset], first_available_frame_index=round(report*10),
        material_identity_confirmed=False, new_layer_confirmed=False,
        mechanical_revolution_confirmed=False, actuator_command=None)


def prediction():
    records = []
    values = [[None]*3, [None]*3, [1., 2., 4.], [None, 4., 4.], [4., 4., 4.]]
    for i, (frame, displacement) in enumerate(zip((0, 50, 51, 52, 53), values)):
        warm = frame <= 50
        states = []
        for j, value in enumerate(displacement):
            states.append(dict(index=j, row_y=100.+10*j, phase='warmup' if warm else 'confirmed',
                quality=value is not None, displacement_px=value,
                low_threshold_px=None if warm else 1.5, high_threshold_px=None if warm else 3.))
        tracker = dict(time_s=frame/10., row_states=states, new_candidates=[],
            material_identity_confirmed=False, new_layer_confirmed=False,
            mechanical_revolution_confirmed=False, actuator_command=None)
        records.append(dict(frame_index=frame, time_s=frame/10.,
            quality_mask=[r['quality'] for r in states], profile_residual=displacement,
            quality_reason=['synthetic_only']*3, tracker=tracker))
    return dict(status='complete', session='left', protocol_sha256='a'*64,
        video=dict(indices=[0, 50, 51, 52, 53], fps=10., frame_count=601), records=records,
        candidates=[], quality_counts=dict(frames=5, rows=3, after_startup_cells=9,
            profile_consistent_cells=8, unknown_after_startup_cells=1),
        physical_new_layer_confirmed=False, new_layer_confirmed=False,
        material_identity_confirmed=False, mechanical_revolution_confirmed=False,
        hardware_access=False, deployment_allowed=False, actuator_command=None)


class FrontReviewTests(unittest.TestCase):
    def test_complete_prediction_and_current_unknown_mask(self):
        p = prediction()
        self.assertEqual(validate_prediction(p, 'left', p['video'], 'a'*64), [100., 110., 120.])
        matrix = direct_state_matrix(p)
        np.testing.assert_array_equal(matrix, [[WARMUP]*3, [WARMUP]*3,
            [LOW, MID, HIGH], [UNKNOWN, HIGH, HIGH], [HIGH]*3])
        self.assertEqual(p['records'][3]['tracker']['row_states'][0]['phase'], 'confirmed')
        self.assertEqual(matrix[3, 0], UNKNOWN)

    def test_quality_and_adjacent_support_are_not_accuracy(self):
        summary = quality_summary(prediction())
        self.assertEqual(summary['direct_state_cell_counts'], dict(unknown=1, low=1, mid=1, high=6))
        self.assertEqual(summary['frames_with_three_adjacent_direct_high'], 1)
        self.assertEqual(summary['maximum_adjacent_direct_high'], 3)
        self.assertEqual(summary['historical_confirmed_without_current_quality_cells'], 1)
        self.assertFalse(summary['unknown_means_no_thickening'])
        self.assertFalse(summary['adjacent_high_is_sufficient_to_confirm_front'])

    def test_missing_candidate_has_unknown_not_zero_errors(self):
        score = score_candidates([], {'new_layer_start_s': 20., 'new_layer_first_revolution_end_s': 23.})
        self.assertTrue(score['t1']['no_candidate'])
        for key in ('onset_error_s', 'first_report_minus_t1_s', 'source_time_margin_before_manual_t2_s'):
            self.assertIsNone(score['t1'][key])
        self.assertEqual(score['t0']['evaluation_status'], 'not_implemented_not_evaluated')
        self.assertEqual(score['t2']['evaluation_status'], 'not_implemented_not_evaluated')

    def test_first_emitted_not_closest_all_extras_and_delay_reported(self):
        candidates = [candidate(10., 12., 'early'), candidate(20., 24., 'closer')]
        score = score_candidates(candidates, {'new_layer_start_s': 20., 'new_layer_first_revolution_end_s': 23.})
        self.assertEqual(score['t1']['onset_error_s'], -10.)
        self.assertEqual(score['t1']['extra_candidate_count'], 1)
        self.assertTrue(score['all_candidates'][0]['report_is_earlier_than_manual_t1'])
        self.assertEqual(score['all_candidates'][1]['onset_error_s'], 0.)
        self.assertFalse(score['all_candidates'][1]['report_not_later_than_manual_t2_on_source_axis'])
        self.assertEqual(score['all_candidates'][1]['source_time_margin_before_manual_t2_s'], -1.)
        self.assertFalse(score['all_candidates'][1]['safe_actuation_timing_established'])

    def test_incomplete_missing_frame_warmup_and_device_claims_rejected(self):
        for mutation in ('status', 'frames', 'warmup', 'hardware', 'quality'):
            p = prediction()
            if mutation == 'status': p['status'] = 'started'
            elif mutation == 'frames': p['records'].pop(2)
            elif mutation == 'warmup': p['records'][0]['tracker']['row_states'][0]['high_threshold_px'] = 3.
            elif mutation == 'hardware': p['hardware_access'] = True
            elif mutation == 'quality': p['records'][2]['quality_mask'][0] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_prediction(p, 'left', p['video'], 'a'*64)

    def test_candidate_must_be_bound_to_emission_record(self):
        p = prediction(); c = candidate()
        p['candidates'] = [copy.deepcopy(c)]
        with self.assertRaises(ValueError): validate_prediction(p, 'left', p['video'], 'a'*64)
        p['records'][-1]['tracker']['new_candidates'] = [copy.deepcopy(c)]
        validate_prediction(p, 'left', p['video'], 'a'*64)
        p['candidates'][0]['first_available_s'] = 5.2
        with self.assertRaises(ValueError): validate_prediction(p, 'left', p['video'], 'a'*64)

    def test_failed_complete_cohort_gate_never_reads_truth(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch('vision_geometry_lab.v19_front_review.load_complete_cohort', side_effect=ValueError('missing second complete prediction')):
                with patch('vision_geometry_lab.v19_front_review.read') as reader:
                    with self.assertRaises(ValueError): score_from_base(Path(temp), Path(temp)/'must_not_be_opened.json')
                    reader.assert_not_called()

    def test_render_without_truth_and_no_prediction_change(self):
        p = prediction(); original = copy.deepcopy(p)
        image = timeline_figure(p, 'synthetic-only')
        self.assertEqual(image.size, (2100, 1290))
        self.assertEqual(p, original)


if __name__ == '__main__':
    unittest.main()
