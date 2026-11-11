import unittest
import cv2,numpy as np
from ecw_v2 import local_axis
from ecw_v3 import local_boundary
from local_measurement import D,ROOT,BG4,read
from window_detector import evaluate
class Tests(unittest.TestCase):
    def test_bilateral_free_rope_known_slope(self):
        im=np.empty((3000,4096,3),np.uint8);im[:]=(40,140,170)
        p=np.array([[500,1850],[2500,2250]],np.int32)
        cv2.polylines(im,[p],False,(220,220,220),18,cv2.LINE_AA)
        a=local_axis(im,[-.2,1,-1750]);self.assertTrue(a['valid'])
        self.assertLess(abs(a['y_at_d']-(1750+.2*D)),2)
        self.assertLess(abs(a['slope']-.2),.003)
    def test_no_rope_is_not_measurement(self):
        im=np.empty((3000,4096,3),np.uint8);im[:]=(40,140,170)
        self.assertFalse(local_axis(im,[-.2,1,-1750])['valid'])
    def test_gray_surface_is_not_free_rope(self):
        im=np.full((3000,4096,3),120,np.uint8)
        cv2.line(im,(500,1850),(2500,2250),(240,240,240),20)
        self.assertFalse(local_axis(im,[-.2,1,-1750])['valid'])
    def test_current_interface_known_curve(self):
        im=np.empty((3000,4096,3),np.uint8);im[:]=(40,140,170)
        x=np.arange(4096);y=2320+.25*(x-D)+.0002*(x-D)**2
        for xx in range(4096):im[int(y[xx]):,xx]=(120,120,120)
        c=np.c_[np.arange(2000,2501,8),np.interp(np.arange(2000,2501,8),x,y)]
        b=local_boundary(im,dict(region=dict(top_curve_xy=c)))
        self.assertTrue(b['valid']);self.assertLess(abs(b['y_at_d']-2320),4)
    def test_missing_interface_is_invalid(self):
        im=np.full((3000,4096,3),120,np.uint8)
        c=np.c_[np.arange(2000,2501,8),np.full(63,2320)]
        self.assertFalse(local_boundary(im,dict(region=dict(top_curve_xy=c)))['valid'])
    def test_online_prefix_independent_of_future(self):
        a=read(ROOT/'v3_full/records.json');b=read(BG4/'full_003/compact_validation.json')
        full,_=evaluate(a,b);prefix,_=evaluate(a[:280],b[:280]);self.assertEqual(full[:280],prefix)
if __name__=='__main__':unittest.main()
