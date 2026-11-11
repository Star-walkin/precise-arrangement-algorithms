import copy,unittest
import numpy as np
import boundary_bg7 as m
from common import *

class Tests(unittest.TestCase):
    def setUp(self):
        self.im=np.zeros((3000,4096,3),np.uint8);self.x=np.arange(600.,2801.,8.)
        self.c=np.array([2100.,30.,210.,20.,40.]);self.y=m.design(self.x,4096)@self.c
        self.p=dict(top_observed_xy=np.c_[self.x,self.y].tolist(),top_curve_xy=np.c_[self.x,self.y].tolist())
        self.t=dict(valid=False);self.b=m.Boundary()
        for i in range(3):self.b.step(self.im,self.p,self.t,i)
    def test_clean_curve(self):
        r=self.b.step(self.im,self.p,self.t,3)
        self.assertTrue(r['fit_valid']);self.assertLess(abs(r['edge_y']-m.design([D],4096)[0]@self.c),1e-6)
    def test_occluded_bad_points_do_not_bend_curve(self):
        p=copy.deepcopy(self.p)
        for v in p['top_observed_xy']:
            if 1500<v[0]<2700:v[1]=1800+.2*v[0]
            else:v[1]+=4
        r=self.b.step(self.im,p,self.t,3,forced_exclude=[[1500,2700]])
        self.assertTrue(r['fit_valid']);self.assertFalse(r['d_locally_observed'])
        self.assertLess(abs(r['edge_y']-(m.design([D],4096)[0]@self.c+4)),1e-5)
        self.assertFalse(r['shape_updated_current_frame'])
    def test_history_cannot_replace_current_points(self):
        r=self.b.step(self.im,self.p,self.t,3,forced_exclude=[[0,4096]])
        self.assertFalse(r['fit_valid']);self.assertIsNone(r['edge_y'])
    def test_stale_shape_rejected(self):
        r=self.b.step(self.im,self.p,self.t,200,forced_exclude=[[1900,2600]])
        self.assertFalse(r['fit_valid']);self.assertIn('shape_prior_too_old',r['reasons'])
    def test_model_not_direct_at_d(self):
        r=self.b.step(self.im,self.p,self.t,3,forced_exclude=[[D-200,D+200]])
        self.assertFalse(r['d_locally_observed']);self.assertTrue(r['inferred_segments'])
    def test_foreground_bright_bar(self):
        im=self.im.copy()
        import cv2
        yy=int(np.interp(D,self.x,self.y));cv2.rectangle(im,(int(D-100),yy),(int(D+100),yy+60),(245,245,245),-1)
        r=self.b.step(im,self.p,self.t,3)
        self.assertTrue(r['d_foreground_occluded']);self.assertFalse(r['count_boundary_valid'])

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests);r=unittest.TextTestRunner(verbosity=2).run(suite)
    dump(ROOT/'test_receipt.json',dict(tests=r.testsRun,passed=r.wasSuccessful(),failures=len(r.failures),errors=len(r.errors),code_sha256=sha(ROOT/'boundary_bg7.py')))
    raise SystemExit(0 if r.wasSuccessful() else 1)
