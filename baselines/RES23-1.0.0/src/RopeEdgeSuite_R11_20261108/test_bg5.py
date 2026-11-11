"""Meaningful region gate regressions, including real-frame perturbations."""
from pathlib import Path
import unittest,json,time
import cv2,numpy as np
import region_bg5 as g
from diagnose import ROOT,OLD,read

class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.p=10.;self.d=400.;self.xy=np.c_[np.arange(200,601,8),np.full(51,300.)]
        self.yellow=np.zeros((500,800),np.uint8);self.yellow[200:300,150:650]=255
        self.outer=dict(valid=False,reason='outer_rim_local_evidence_incomplete',points_xy=[])
    def run_gate(self,yellow=None):return g.flange_presence(self.outer,self.xy,self.p,self.yellow if yellow is None else yellow,1.,self.d)
    def test_missing_outer_does_not_veto_actual_adjacent_flange(self):self.assertTrue(self.run_gate()['valid'])
    def test_flange_below_coil_is_rejected(self):
        y=np.zeros_like(self.yellow);y[300:400,150:650]=255;self.assertFalse(self.run_gate(y)['valid'])
    def test_local_color_occlusion_cannot_be_filled_from_global_evidence(self):
        y=self.yellow.copy();y[:,360:441]=0;self.assertFalse(self.run_gate(y)['valid'])
    def test_no_yellow_does_not_pass_with_fake_outer_points(self):
        self.outer.update(valid=True,points_xy=self.xy.tolist());self.assertFalse(self.run_gate(np.zeros_like(self.yellow))['valid'])
    def test_sparse_interface_rejected(self):
        self.xy=self.xy[:8];self.assertFalse(self.run_gate()['valid'])

class ActualFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1);run=OLD/'transfer_runs/dual_20260923_113745'
        cap=cv2.VideoCapture(read(run/'binding.json')['source']);cap.set(1,20);ok,cls.im=cap.read();cap.release();assert ok
        cls.base=read(run/'bg1/frames/000020.json')['region'];cls.result=g.segment(cls.im)
    def test_recovers_outer_only_failure_with_current_evidence(self):
        self.assertEqual(self.base['reasons'],['outer_flange_not_confirmed']);self.assertTrue(self.result['record']['valid'])
    def test_count_origin_not_replaced_by_outer_rim(self):
        self.assertLess(abs(self.result['record']['edge_y']-self.base['edge_y']),2.)
    def test_surface_never_includes_above_count_boundary(self):
        yy=self.result['record']['edge_y'];self.assertFalse(self.result['surface_mask'][:int(yy),2239].any())
    def test_local_boundary_and_drawn_curve_agree(self):
        r=self.result['record'];xy=np.array(r['top_curve_xy']);self.assertLess(abs(r['edge_y']-np.interp(2238.7,xy[:,0],xy[:,1])),1e-9)
    def test_subpixel_origin_is_explicit_curve_node_regression_f26(self):
        run=OLD/'transfer_runs/dual_20260923_113745';cap=cv2.VideoCapture(read(run/'binding.json')['source'])
        cap.set(1,26);ok,im=cap.read();cap.release();self.assertTrue(ok)
        r=g.segment(im)['record'];self.assertTrue(r['valid']);xy=np.array(r['top_curve_xy'])
        self.assertLess(abs(r['edge_y']-np.interp(2238.7,xy[:,0],xy[:,1])),1e-9)
    def test_gray_scene_cannot_invent_yellow_flange(self):
        im=cv2.cvtColor(cv2.cvtColor(self.im,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR);self.assertFalse(g.segment(im)['record']['valid'])
    def test_current_d_occlusion_still_rejects(self):
        im=self.im.copy();im[1850:2550,2110:2370]=0;self.assertFalse(g.segment(im)['record']['valid'])
    def test_blank_frame_rejected(self):self.assertFalse(g.segment(np.zeros_like(self.im))['record']['valid'])
    def test_no_temporal_mask_carryover(self):
        a=g.segment(self.im)['record'];b=g.segment(np.zeros_like(self.im))['record'];self.assertTrue(a['valid']);self.assertFalse(b['valid'])

if __name__=='__main__':
    tick=time.perf_counter();suite=unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    (ROOT/'test_receipt.json').write_text(json.dumps(dict(passed=result.wasSuccessful(),tests=result.testsRun,elapsed_s=time.perf_counter()-tick,
        failures=[str(x) for x in result.failures+result.errors]),indent=2),encoding='utf8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
