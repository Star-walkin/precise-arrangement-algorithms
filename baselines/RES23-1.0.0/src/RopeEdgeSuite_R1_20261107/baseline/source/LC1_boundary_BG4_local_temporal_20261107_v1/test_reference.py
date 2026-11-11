import unittest
import cv2,numpy as np
from camera_reference import CameraContinuity
import test_bg4
from test_bg4 import frame

class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image=np.zeros((3000,4096,3),np.uint8);rng=np.random.default_rng(991)
        for x,y,r in zip(rng.integers(100,3996,120),rng.integers(100,1300,120),rng.integers(4,13,120)):
            cv2.circle(cls.image,(int(x),int(y)),int(r),(220,220,220),-1)
    def test_fixed_reference_jitter_does_not_accumulate(self):
        c=CameraContinuity();c.step(self.image)
        for j in range(12):
            moved=cv2.warpAffine(self.image,np.array([[1,0,(-1)**j],[0,1,0]],float),(4096,3000))
            r=c.step(moved);self.assertFalse(r['camera_changed']);self.assertTrue(r['confirmed'])
            self.assertLess(r['translation_px'],3)
    def test_persistent_real_camera_shift_invalidates_d(self):
        c=CameraContinuity();c.step(self.image)
        moved=cv2.warpAffine(self.image,np.array([[1,0,20],[0,1,0]],float),(4096,3000))
        for j in range(3):r=c.step(moved)
        self.assertTrue(r['camera_changed'])
    def test_weak_pitch_alias_does_not_destroy_valid_lock(self):
        c,_=test_bg4.TemporalTests().locked();r=frame(15,2.2,False);r['observation']['pitch_px']=30
        result=c.step(r);self.assertEqual(result['state'],'predicted')
        self.assertIn('geometry_pitch_step_observation_rejected',result['events'])

if __name__=='__main__':unittest.main(verbosity=2)
