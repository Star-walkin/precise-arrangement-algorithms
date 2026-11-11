"""Anchored E1 regressions using frozen real S1 on current synthetic pixels."""
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from live_s1_tracker_anchored import LiveS1Tracker, pixel_sha256, BASE_SHA256, _base
from test_live_s1_tracker import image, mask, seed


class AnchoredTrackerTests(unittest.TestCase):
    def tracker(self,**kw):
        return LiveS1Tracker(mask(),initial_seed_xy=seed(),initial_seed_source='synthetic_current_E1',**kw)

    def test_defaults_match_A7_length_and_original_refit_support(self):
        t=self.tracker()
        self.assertEqual(t.seed_length,120.)
        self.assertEqual(t.minimum_span,400.)

    def test_one_hundred_advancing_material_proposals_cannot_march_spatial_anchor(self):
        t=self.tracker();self.assertTrue(t.step(image(),0.,0)['valid'])
        direction=np.array([1.,.15]);direction/=np.linalg.norm(direction)
        # Isolate the dangerous case: an otherwise valid LK material minimum
        # advances along the SAME visible line, without moving its spatial E1.
        def moving_material(*args):
            index=args[-1];x=190.+2.5*index
            p=np.array([x,150.+.15*x])
            return np.stack((p,p+120*direction)),None,dict(LK_used=True)
        with patch.object(_base.LiveS1Tracker,'_track_seed',side_effect=moving_material):
            for i in range(1,101):
                r=t.step(image(),i/10.,i)
                self.assertTrue(r['valid'],(i,r['diagnostics']))
                self.assertEqual(r['seed_xy_original'][0][0],70.)
                self.assertEqual(r['diagnostics']['spatial_anchor']['reset_count'],1)
                self.assertEqual(r['diagnostics']['S1_calls'],1)
        self.assertEqual(t.anchor_x_original,70.)

    def test_current_transverse_motion_changes_y_without_changing_x(self):
        t=self.tracker()
        for i in range(6):
            r=t.step(image(2*i),i/10.,i)
            self.assertTrue(r['valid'],r['diagnostics'])
            self.assertEqual(r['seed_xy_original'][0][0],70.)
            a,b,c=r['current_line_abc']
            self.assertAlmostEqual(-(a*300+c)/b,195+2*i,delta=2.)
            self.assertEqual(r['diagnostics']['spatial_anchor']['reset_count'],1)

    def test_normal_search_keeps_fixed_x(self):
        t=self.tracker();self.assertTrue(t.step(image(),0.,0)['valid'])
        p=np.array(seed(),float);p[:,1]-=8.
        with patch.object(_base.LiveS1Tracker,'_track_seed',return_value=(p,None,dict(LK_used=False))):
            r=t.step(image(),.1,1)
        self.assertTrue(r['valid'],r['diagnostics'])
        self.assertEqual(r['seed_xy_original'][0][0],70.)

    def test_current_occlusion_never_holds_previous_ABC(self):
        t=self.tracker();self.assertTrue(t.step(image(),0.,0)['valid'])
        r=t.step(image(blank=True),.1,1)
        self.assertFalse(r['valid']);self.assertIsNone(r['current_line_abc'])
        self.assertIsNone(r['line_abc']);self.assertIsNone(t.prev)
        self.assertEqual(t.anchor_x_original,70.)
        self.assertEqual(r['diagnostics']['S1_calls'],0)

    def test_only_explicit_current_pixel_bound_reacquisition_can_reset_anchor(self):
        calls=[]
        def callback(current,time,index):
            calls.append(index)
            if index!=2:return None
            return dict(seed_xy=[[150.,176.5],[270.,194.5]],
                seed_source='current_frame_reacquisition',source_pixel_sha256=pixel_sha256(current),model_forward_count=0)
        t=self.tracker(reacquire_callback=callback)
        self.assertTrue(t.step(image(),0.,0)['valid'])
        self.assertFalse(t.step(image(blank=True),.1,1)['valid'])
        r=t.step(image(4.),.2,2)
        self.assertTrue(r['valid'],r['diagnostics'])
        self.assertEqual(calls,[1,2]);self.assertEqual(t.anchor_x_original,150.)
        self.assertEqual(t.anchor_reset_count,2)
        self.assertEqual(r['diagnostics']['spatial_anchor']['reason'],'explicit_pixel_bound_current_reacquisition')

    def test_wrong_frame_callback_cannot_reset_anchor_or_output_line(self):
        def callback(*args):
            return dict(seed_xy=[[150.,172.5],[270.,190.5]],seed_source='wrong_frame',
                source_pixel_sha256='wrong',model_forward_count=0)
        t=self.tracker(reacquire_callback=callback)
        self.assertTrue(t.step(image(),0.,0)['valid'])
        r=t.step(image(blank=True),.1,1)
        self.assertFalse(r['valid']);self.assertIsNone(r['current_line_abc'])
        self.assertEqual(t.anchor_x_original,70.);self.assertEqual(t.anchor_reset_count,1)

    def test_near_vertical_fixed_x_proposal_is_unknown(self):
        t=self.tracker();self.assertTrue(t.step(image(),0.,0)['valid'])
        proposal=np.array([[70.,160.],[71.,280.]])
        with patch.object(_base.LiveS1Tracker,'_track_seed',return_value=(proposal,None,{})):
            r=t.step(image(),.1,1)
        self.assertFalse(r['valid']);self.assertIsNone(r['current_line_abc'])
        self.assertEqual(r['diagnostics']['tracking']['reason'],'near_vertical_fixed_x_anchor_degenerate')

    def test_nonconsecutive_input_cannot_track_stale_line(self):
        t=self.tracker();t.step(image(),0.,0)
        r=t.step(image(),.2,2)
        self.assertFalse(r['valid']);self.assertIsNone(r['current_line_abc'])

    def test_frozen_original_tracker_is_unchanged(self):
        p=Path(__file__).with_name('live_s1_tracker.py')
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),BASE_SHA256)


if __name__=='__main__':unittest.main()
