import unittest,json,time
import cv2,numpy as np
import region_bg5 as v1
import region_bg5_refined as v2
from diagnose import ROOT,OLD,read

class RefinedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1);cls.source=read(OLD/'transfer_runs/dual_20260923_113745/binding.json')['source']
    def frame(self,i):
        cap=cv2.VideoCapture(self.source);cap.set(1,i);ok,im=cap.read();cap.release();assert ok;return im
    def test_disconnected_fragments_merge_without_painting_gap(self):
        mask=np.zeros((500,800),np.uint8);mask[230:280,100:300]=255;mask[230:280,400:600]=255
        merged,d=v2.yellow_fragments(mask,20.,1.);self.assertTrue(d['used']);np.testing.assert_array_equal(merged,mask)
    def test_far_lower_yellow_distractor_excluded(self):
        mask=np.zeros((500,800),np.uint8);mask[100:140,100:350]=255;mask[110:140,400:600]=255;mask[400:420,100:350]=255
        merged,d=v2.yellow_fragments(mask,10.,1.);self.assertFalse(merged[400:].any());self.assertTrue(d['used'])
    def test_actual_highlight_f280_recovers(self):
        im=self.frame(280);self.assertEqual(v1.segment(im)['record']['reasons'],['insufficient_yellow_flange'])
        r=v2.segment(im)['record'];self.assertTrue(r['valid']);self.assertTrue(r['yellow_fragment_association']['used'])
    def test_unaffected_branch_masks_and_count_origin_are_exact(self):
        im=self.frame(20);a=v1.segment(im);b=v2.segment(im)
        for k in ['surface_mask','yellow_mask']:np.testing.assert_array_equal(a[k],b[k])
        for k in ['edge_y','top_curve_xy','top_observed_xy','reasons','valid']:self.assertEqual(a['record'][k],b['record'][k])
    def test_recovered_frame_local_occlusion_remains_rejected(self):
        im=self.frame(280);im[1850:2550,2110:2370]=0;self.assertFalse(v2.segment(im)['record']['valid'])
    def test_grayscale_remains_rejected(self):
        im=self.frame(280);im=cv2.cvtColor(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR);self.assertFalse(v2.segment(im)['record']['valid'])

if __name__=='__main__':
    start=time.perf_counter();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    (ROOT/'refined_test_receipt.json').write_text(json.dumps(dict(passed=result.wasSuccessful(),tests=result.testsRun,elapsed_s=time.perf_counter()-start,
        failures=[str(x) for x in result.failures+result.errors]),indent=2),encoding='utf8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
