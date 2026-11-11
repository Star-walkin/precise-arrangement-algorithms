"""Regression checks for loss handling and false-identity rejection."""
import unittest, json
import numpy as np
from tracker_sg2 import Tracker,PARAMETERS,ROOT

class Tests(unittest.TestCase):
    def make(self):
        a=Tracker(np.zeros((600,900),np.uint8))
        a.trusted=dict(t=1.,frame=10,abc=[-.16,1.,-200.],width=24.,reference_x=100.,direction=np.array([1.,.16]))
        a.anchor_x_original=100.;a.history=[(.9,217.),(1.,216.)]
        return a
    def fake_result(self,abc):return dict(valid=True,line_abc=abc,line_points_xy=[[100.,216.],[800.,328.]])
    def test_reject_winding_width(self):
        a=self.make();ok,why,_=a.identity(self.fake_result([-.16,1.,-199.]),np.array([[100.,215.],[220.,234.2]]),{'width_px':12.},1.1)
        self.assertFalse(ok);self.assertIn('rope_width_identity_mismatch',why)
    def test_reject_winding_direction(self):
        a=self.make();ok,why,_=a.identity(self.fake_result([.0525,1.,-220.]),np.array([[100.,215.],[220.,234.2]]),{'width_px':24.},1.1)
        self.assertFalse(ok);self.assertIn('rope_angle_identity_mismatch',why)
    def test_accept_small_motion(self):
        a=self.make();ok,why,_=a.identity(self.fake_result([-.16,1.,-199.]),np.array([[100.,215.],[220.,234.2]]),{'width_px':24.},1.1)
        self.assertTrue(ok,why)
    def test_no_current_evidence_no_output_and_no_anchor_commit(self):
        a=self.make();old=a.trusted.copy()
        a._track_seed=lambda *args:(None,None,{'reason':'test_missing_current_evidence'})
        a._seed_witness=lambda *args:dict(valid=False,reason='no_edges')
        r=a.step(np.zeros((600,900,3),np.uint8),1.1,11)
        self.assertFalse(r['valid']);self.assertIsNone(r['line_abc']);self.assertEqual(r['line_points_xy'],[])
        self.assertEqual(a.trusted['frame'],old['frame']);self.assertEqual(a.anchor_x_original,100.)
        self.assertEqual(r['diagnostics']['accept_reason'],'SG2_short_loss_search_memory_retained')
    def test_memory_expires(self):
        a=self.make();a.prev={'not':'to be reused'}
        r=a.step(np.zeros((600,900,3),np.uint8),1.6,16)
        self.assertFalse(r['valid']);self.assertIsNone(a.prev)
        self.assertEqual(r['diagnostics']['accept_reason'],'SG2_reacquisition_required')
    def test_cold_start_needs_current_surface(self):
        a=Tracker(np.zeros((600,900),np.uint8))
        ok,why,_=a.identity(self.fake_result([-.16,1.,-200.]),np.array([[100.,216.],[220.,235.2]]),{'width_px':24.},1.)
        self.assertFalse(ok);self.assertIn('cold_start_current_region_unavailable',why)
    def test_profile_sampler_overflow_rejects_candidate_without_commit(self):
        a=self.make();a.entry_seed=None
        a._seed_witness=lambda *args:dict(valid=True,width_px=55.)
        def failure(*args,**kwargs):raise ValueError('profile_radius_px too small for width candidates')
        a.s1.step=failure
        got=a.trial(np.zeros((600,900,3),np.uint8),[[100.,216.],[220.,235.2]],'bad_width',1.1)
        self.assertIsNone(got);self.assertEqual(a.trusted['frame'],10);self.assertEqual(a.anchor_x_original,100.)
        self.assertEqual(a.attempts[-1]['reasons'],['candidate_width_exceeds_frozen_profile_sampler'])

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    r=unittest.TextTestRunner(verbosity=2).run(suite)
    (ROOT/'test_receipt.json').write_text(json.dumps(dict(passed=r.wasSuccessful(),tests=r.testsRun,failures=len(r.failures),errors=len(r.errors)),indent=2))
    raise SystemExit(0 if r.wasSuccessful() else 1)
