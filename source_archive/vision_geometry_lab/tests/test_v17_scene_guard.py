import unittest

import cv2
import numpy as np

from vision_geometry_lab.v17_scene_guard import ScenePoseGuard, original_affine, occupied_cells


def scene(height=480,width=640):
    rng=np.random.default_rng(20260930)
    image=np.full((height,width),80,np.uint8)
    for _ in range(350):
        x=int(rng.integers(10,width-10));y=int(rng.integers(10,height-10))
        cv2.circle(image,(x,y),int(rng.integers(2,7)),int(rng.integers(20,240)),-1)
    return cv2.GaussianBlur(image,(3,3),.7)


def warped(image,dx=0,dy=0,rotation=0,scale=1):
    h,w=image.shape
    a=cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2),rotation,scale)
    a[:,2]+=[dx,dy]
    return cv2.warpAffine(image,a,(w,h)),a


class SceneGuardTests(unittest.TestCase):
    def pair(self,first,second):
        guard=ScenePoseGuard(); initial=guard.update(first,0.)
        self.assertFalse(initial['pose_stable']);self.assertEqual(initial['history_action'],'reset')
        return guard.update(second,.2)

    def test_small_translation_full_pixel_affine(self):
        first=scene();second,a=warped(first,2.,1.)
        result=self.pair(first,second)
        self.assertTrue(result['pose_stable'],result['reason'])
        np.testing.assert_allclose(result['previous_to_current'],a,atol=.13)
        self.assertFalse(result['spool_motion_inferred'])

    def test_small_rotation_is_measured(self):
        first=scene();second,a=warped(first,rotation=.15)
        result=self.pair(first,second)
        self.assertTrue(result['pose_stable'],result['reason'])
        self.assertAlmostEqual(result['pose']['rotation_deg'],-.15,delta=.08)

    def test_large_translation_resets_history(self):
        first=scene();second,_=warped(first,20.,0.)
        result=self.pair(first,second)
        self.assertFalse(result['pose_stable']);self.assertEqual(result['history_action'],'reset')
        self.assertIn('translation',result['reason'])

    def test_large_rotation_resets_history(self):
        first=scene();second,_=warped(first,rotation=2.)
        result=self.pair(first,second)
        self.assertFalse(result['pose_stable']);self.assertIn('rotation',result['reason'])

    def test_scale_change_resets_history(self):
        first=scene();second,_=warped(first,scale=1.02)
        result=self.pair(first,second)
        self.assertFalse(result['pose_stable']);self.assertIn('scale',result['reason'])

    def test_empty_image_rejected(self):
        blank=np.zeros((480,640),np.uint8);result=self.pair(blank,blank)
        self.assertFalse(result['pose_stable']);self.assertEqual(result['reason'],'no_background_features')

    def test_missing_frame_clears_baseline(self):
        guard=ScenePoseGuard();im=scene();guard.update(im,0.)
        self.assertEqual(guard.update(None,.2)['history_action'],'reset')
        self.assertEqual(guard.update(im,.4)['reason'],'first_frame_requires_new_history')

    def test_gap_and_time_reversal_reset(self):
        guard=ScenePoseGuard();im=scene();guard.update(im,0.)
        self.assertEqual(guard.update(im,1.)['reason'],'frame_gap_exceeds_limit')
        self.assertEqual(guard.update(im,.5)['reason'],'nonincreasing_timestamp')

    def test_shape_change_reset(self):
        result=self.pair(scene(),scene(500,640))
        self.assertFalse(result['pose_stable']);self.assertEqual(result['reason'],'image_shape_changed')

    def test_bottom_motion_does_not_mean_camera_motion(self):
        first=scene();second=first.copy();second[220:]=255-second[220:]
        result=self.pair(first,second)
        self.assertTrue(result['pose_stable'],result['reason'])
        self.assertFalse(result['spool_motion_inferred'])

    def test_original_affine_scales_translation(self):
        matrix=np.array([[1.,0.,2.],[0.,1.,3.]])
        np.testing.assert_allclose(original_affine(matrix,(.25,.25),(.25,.25)),[[1,0,8],[0,1,12]])

    def test_grid_coverage_counts_cells_not_points(self):
        self.assertEqual(occupied_cells(np.repeat([[1.,1.]],100,axis=0),(400,1000)),1)


if __name__=='__main__':unittest.main()
