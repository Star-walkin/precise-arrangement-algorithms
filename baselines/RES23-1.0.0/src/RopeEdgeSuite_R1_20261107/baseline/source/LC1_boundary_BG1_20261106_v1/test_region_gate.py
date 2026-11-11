"""Fail-closed integration checks with current permitted-source perturbations."""
import unittest,json,cv2,numpy as np
from pathlib import Path
from gated_rope import analyze
from region_gate import segment
from inspect_regions import SOURCE
ROOT=Path(__file__).resolve().parent

class RegionGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1);c=cv2.VideoCapture(str(SOURCE));c.set(1,240);ok,cls.image=c.read();c.release();assert ok
        cls.old=json.loads((ROOT.parent/'LC1_fixed_d2238p7_full_20261106_v1/analysis_anchored/frames/frame_000240.json').read_text(encoding='utf8'))
    def reject_without_detector(self,image):
        called=[]
        def forbidden(data):called.append(True);raise AssertionError('F3 must not execute on an invalid region')
        r,m=analyze(image,self.old,detector=forbidden)
        self.assertFalse(called);self.assertFalse(r['region']['valid'])
        self.assertFalse(r['F3_called']);self.assertFalse(r['count_valid']);self.assertIsNone(r['display_N'])
    def test_black_image_cannot_reuse_previous_S1_to_emit_count(self):
        self.reject_without_detector(np.zeros_like(self.image))
    def test_uniform_yellow_cannot_be_a_wound_surface(self):
        a=np.zeros_like(self.image);a[:]=[40,110,125];self.reject_without_detector(a)
    def test_missing_yellow_flange_blocks_detector(self):
        a=cv2.cvtColor(cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
        self.reject_without_detector(a)
    def test_black_occlusion_at_count_interface_blocks_detector(self):
        a=self.image.copy();a[2180:2500,2120:2360]=0;self.reject_without_detector(a)
    def test_known_clear_frame_is_current_pixel_supported(self):
        r=segment(self.image);self.assertTrue(r['record']['valid'])
        self.assertGreater(r['record']['count_neighborhood_surface_coverage'],.95)
        self.assertFalse(np.any((r['surface_mask']>0)&(r['yellow_mask']>0)))
    def test_boundary_position_follows_translated_current_image(self):
        r=segment(self.image)['record'];a=cv2.warpAffine(self.image,np.float32([[1,0,0],[0,1,24]]),(4096,3000))
        q=segment(a)['record'];self.assertTrue(q['valid'])
        self.assertAlmostEqual(q['edge_y']-r['edge_y'],24,delta=3)
    def test_surface_does_not_include_background_above_flange(self):
        r=segment(self.image);self.assertFalse(np.any(r['surface_mask'][:1800]))

if __name__=='__main__':unittest.main(verbosity=2)
