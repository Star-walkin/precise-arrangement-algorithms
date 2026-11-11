"""Causal tracker contracts using the real frozen S1 on synthetic pixels."""
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from live_s1_tracker import LiveS1Tracker,pixel_sha256


def image(shift=0.,blank=False):
    a=np.full((480,720,3),25,np.uint8)
    if not blank:
        cv2.line(a,(20,int(153+shift)),(700,int(255+shift)),(220,220,220),20,cv2.LINE_AA)
    return a


def seed(shift=0.):return [[70.,160.5+shift],[190.,178.5+shift]]


def mask():
    m=np.zeros((480,720),np.uint8);m[330:,400:]=255
    return m


class LiveTrackerTests(unittest.TestCase):
    def tracker(self,**kw):
        return LiveS1Tracker(mask(),initial_seed_xy=seed(),initial_seed_source='synthetic_current_E1',**kw)

    def test_real_S1_fresh_current_line_tracks_transverse_motion(self):
        tracker=self.tracker()
        for i in range(5):
            result=tracker.step(image(2*i),i/10.,i)
            self.assertTrue(result['valid'],result['diagnostics'])
            self.assertTrue(result['S1']['diagnostics']['s1_core_executed'])
            a,b,c=result['current_line_abc']
            self.assertAlmostEqual(-(a*300+c)/b,195+2*i,delta=2.)
            self.assertEqual(result['diagnostics']['S1_calls'],1)
            self.assertFalse(result['diagnostics']['callback_called'])

    def test_current_blank_frame_cannot_hold_previous_ABC(self):
        tracker=self.tracker();self.assertTrue(tracker.step(image(),0.,0)['valid'])
        result=tracker.step(image(blank=True),.1,1)
        self.assertFalse(result['valid'])
        self.assertIsNone(result['current_line_abc'])
        self.assertIsNone(result['line_abc'])
        self.assertIsNone(tracker.prev)
        self.assertFalse(result['S1']['diagnostics']['s1_core_executed'])

    def test_flow_is_optional_current_bilateral_validation_can_recover(self):
        tracker=self.tracker();tracker.step(image(),0.,0)
        with patch('live_s1_tracker.cv2.calcOpticalFlowPyrLK',side_effect=cv2.error('synthetic LK failure')):
            result=tracker.step(image(2.),.1,1)
        self.assertTrue(result['valid'],result['diagnostics'])
        self.assertFalse(result['diagnostics']['tracking']['LK_used'])

    def test_pixel_bound_current_callback_recovers_without_historical_line(self):
        called=[]
        def callback(current,t,index):
            called.append(index)
            return dict(seed_xy=seed(4.),seed_source='current_pixel_bound_test_reacquire',
                        source_pixel_sha256=pixel_sha256(current),model_forward_count=0)
        tracker=LiveS1Tracker(mask(),reacquire_callback=callback)
        result=tracker.step(image(4.),.2,2)
        self.assertTrue(result['valid'])
        self.assertEqual(called,[2])
        self.assertEqual(result['model_forward_count'],0)
        self.assertEqual(result['seed_source'],'current_pixel_bound_test_reacquire')

    def test_wrong_pixel_hash_reacquisition_is_rejected(self):
        def callback(current,t,index):
            return dict(seed_xy=seed(),seed_source='wrong_frame',source_pixel_sha256='not_this_image',model_forward_count=0)
        tracker=LiveS1Tracker(mask(),reacquire_callback=callback)
        result=tracker.step(image(),0.,0)
        self.assertFalse(result['valid'])
        self.assertIsNone(result['current_line_abc'])
        self.assertTrue(result['diagnostics']['reacquire_seed_pixel_hash_mismatch_or_missing'])
        self.assertEqual(result['diagnostics']['S1_calls'],0)

    def test_no_seed_returns_faithful_S1_UNKNOWN_with_provenance(self):
        tracker=LiveS1Tracker(mask())
        result=tracker.step(image(),0.,0)
        self.assertFalse(result['valid'])
        self.assertEqual(result['S1']['s1_status'],'UNKNOWN')
        self.assertEqual(result['S1']['diagnostics']['stop_reason'],'missing_explicit_E1')
        self.assertFalse(result['S1']['diagnostics']['s1_core_executed'])

    def test_missing_or_out_of_order_frame_does_not_track_stale_support(self):
        tracker=self.tracker();tracker.step(image(),0.,0)
        result=tracker.step(image(4.),.2,2)
        self.assertFalse(result['valid'])
        self.assertEqual(result['diagnostics']['tracking']['reason'],'nonconsecutive_or_stale_previous_frame')
        self.assertIsNone(result['current_line_abc'])

    def test_seed_endpoints_are_not_output_line_samples(self):
        result=self.tracker().step(image(),0.,0)
        self.assertTrue(result['valid'])
        points=np.asarray(result['line_points_xy'])
        endpoints=np.asarray(result['seed_xy_original'])
        self.assertTrue(all(np.min(np.linalg.norm(points-p,axis=1))>1. for p in endpoints))
        self.assertFalse(result['S1']['diagnostics']['line_fit_uses_E1_points'])

    def test_seed_witness_is_measured_and_agrees_with_original_seed_predicate(self):
        result=self.tracker().step(image(),0.,0)
        self.assertTrue(result['diagnostics']['seed_witness_matches_primary_S1_seed_measurement'])
        witness=result['diagnostics']['seed_validation']
        self.assertGreaterEqual(len(witness['points_xy']),4)
        self.assertIn('not_E1_endpoints',witness['source'])

    def test_initial_seed_hash_mismatch_does_not_authorize_a_line(self):
        result=self.tracker(initial_seed_pixel_sha256='incorrect').step(image(),0.,0)
        self.assertFalse(result['valid'])
        self.assertTrue(result['diagnostics']['initial_seed_pixel_hash_mismatch'])

    def test_short_current_support_can_be_flagged_without_holding_ABC(self):
        result=self.tracker(minimum_S1_span_px=10000.).step(image(),0.,0)
        self.assertFalse(result['valid'])
        self.assertTrue(result['S1']['valid'])
        self.assertEqual(result['diagnostics']['accept_reason'],'current_S1_support_span_insufficient')
        self.assertIsNone(result['current_line_abc'])

    def test_original_coordinates_and_no_whole_or_identity_promotion(self):
        result=self.tracker().step(image(),0.,0)
        self.assertEqual(result['diagnostics']['coordinate_system'],'original_unmirrored_image_xy')
        self.assertFalse(result['whole_path_candidate_supported'])
        self.assertFalse(result['same_rope_identity_confirmed'])


if __name__=='__main__':unittest.main()
