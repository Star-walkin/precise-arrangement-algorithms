"""Small synthetic S2 contracts; no real media/model inference."""
import unittest
from unittest import mock
from types import SimpleNamespace
import json
import numpy as np

from lc1_s2_adapter import S2Adapter, _sha


class S2Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = S2Adapter(retain_step_records=True)

    def image_with_gap(self):
        image = np.zeros((320, 480, 3), np.uint8)
        image[150:171, 25:465] = 210
        image[:, 180:241] = 0
        return image

    def test_unknown_gap_remains_segment_break_and_never_full_authorization(self):
        result = self.adapter.step(self.image_with_gap(), [[410., 160.], [350., 160.]],
                                   seed_source='SYNTHETIC TEST, explicit condition', width_px=20.)
        self.assertTrue(result['valid'])
        self.assertGreaterEqual(len(result['supported_segments_xy']), 2)
        self.assertTrue(result['unknown_intervals_px'])
        self.assertTrue(any(not f for f in result['diagnostics']['path_segment_support']))
        for segment in result['supported_segments_xy']:
            if len(segment)>1:
                self.assertLessEqual(float(np.linalg.norm(np.diff(segment, axis=0), axis=1).max()), 4.01)
        for a,b in result['unknown_intervals_px']:
            records = result['diagnostics']['original_prediction']['step_records']
            self.assertFalse(any(r['supported'] for r in records if a <= r['distance_after_seed_px'] <= b))
        self.assertFalse(result['whole_path_candidate_supported'])
        self.assertFalse(result['same_rope_identity_confirmed'])

    def test_source_hashes_unchanged_after_real_original_core_execution(self):
        before = dict(self.adapter.hashes)
        self.adapter.step(self.image_with_gap(), [[410.,160.],[350.,160.]],
                          seed_source='SYNTHETIC TEST', width_px=20.)
        self.assertEqual(before['core_sha256'], _sha(self.adapter.source_dir/'straight_core.py'))
        self.assertEqual(before['guidance_sha256'], _sha(self.adapter.source_dir/'guidance.py'))

    def test_nonfinite_seed_and_invalid_width_rejected(self):
        for seed, width in (([[410.,160.],[350.,np.nan]],20.),
                            ([[410.,160.],[350.,160.]],np.inf),
                            ([[410.,160.],[350.,160.]],0.)):
            with self.subTest(width=width):
                with self.assertRaises(ValueError):
                    self.adapter.step(self.image_with_gap(), seed, seed_source='SYNTHETIC TEST', width_px=width)

    def test_no_input_seed_produces_no_evidence(self):
        result = self.adapter.step(self.image_with_gap(), None,
                                   seed_source='missing explicit seed', width_px=20.)
        self.assertFalse(result['valid'])
        self.assertIsNone(result['line_abc'])
        self.assertEqual(result['line_points_xy'], [])

    def test_auto_ambiguity_cannot_be_overridden_by_supported_S2_candidate(self):
        candidate = dict(candidate_id='synthetic_0', seed_xy=[[410.,160.],[350.,160.]],
                         source='synthetic_auto_image', width_px=20.)
        proposals = dict(status='AMBIGUOUS', selected_candidate_id=None, candidates=[candidate])
        fake = SimpleNamespace(propose_guidance=lambda *args, **kwargs: proposals)
        with mock.patch.object(self.adapter, '_load_guidance', return_value=fake):
            result = self.adapter.step_auto(self.image_with_gap())
        self.assertFalse(result['valid'])
        self.assertIsNone(result['line_abc'])
        self.assertFalse(result['whole_path_candidate_supported'])
        self.assertTrue(result['diagnostics']['s2_auto_candidate_results'][0]['result']['valid'])

    def test_selected_auto_result_is_serializable_and_does_not_authorize_whole(self):
        candidate = dict(candidate_id='synthetic_0', seed_xy=[[410.,160.],[350.,160.]],
                         source='synthetic_auto_image', width_px=20.)
        proposals = dict(status='SELECTED', selected_candidate_id='synthetic_0', candidates=[candidate])
        fake = SimpleNamespace(propose_guidance=lambda *args, **kwargs: proposals)
        with mock.patch.object(self.adapter, '_load_guidance', return_value=fake):
            result = self.adapter.step_auto(self.image_with_gap())
        self.assertTrue(result['valid'])
        self.assertFalse(result['whole_path_candidate_supported'])
        json.dumps(result, allow_nan=False)  # No selected-result/candidate-list reference cycle.


if __name__ == '__main__':
    unittest.main()
