"""Contracts for original-camera inference and downstream canonical geometry."""
import unittest
import numpy as np

from lc1_pipeline import P1S1Frontend, line_to_canonical, points_to_canonical


class StubP1:
    def __init__(self,result):
        self.result=result
        self.image=None
    def step(self,image):
        self.image=image
        return self.result


class StubS1:
    def __init__(self,result):
        self.result=result
        self.image=None
        self.seed=None
    def step(self,image,seed,*,seed_source):
        self.image=image
        self.seed=seed
        self.seed_source=seed_source
        return self.result


class StubS2:
    def __init__(self,result):
        self.result=result
    def step_auto(self,image,**kwargs):
        self.image=image;self.conditions=kwargs
        return self.result


class LivePipelineContracts(unittest.TestCase):
    def pipeline(self,mirror=False,*,p1_supported=True,s1_valid=True):
        # No pretrained/real algorithm is mocked in a claimed video run. These
        # stubs check only data routing, provenance and source gating contracts.
        front=object.__new__(P1S1Frontend)
        front.width=200
        front.camera='left' if mirror else 'right'
        front.mirror=mirror
        path=np.c_[np.arange(30)+80,np.arange(30)+40].tolist()
        front.p1=StubP1(dict(seed_xy=[[10,20],[20,30]],path_xy=path,
                             whole_path_candidate_supported=p1_supported,
                             diagnostics=dict(path_segment_support=[True]*29)))
        front.s1=StubS1(dict(valid=s1_valid,line_abc=[1,2,-50],
                             line_points_xy=[[10,20],[12,19],[14,18]]))
        return front

    def test_both_detectors_receive_original_unmirrored_frame_and_original_seed(self):
        front=self.pipeline(mirror=True)
        original=np.arange(12,dtype=np.uint8).reshape(2,2,3)
        result=front.step(original,1.,12)
        self.assertIs(front.p1.image,original)
        self.assertIs(front.s1.image,original)
        self.assertEqual(front.s1.seed,[[10,20],[20,30]])
        self.assertEqual(result['diagnostics']['seed_xy_canonical'],[[189.,20.],[179.,30.]])
        self.assertEqual(result['path_xy'][0],[119.,40.])
        self.assertEqual(result['line_points_xy'][0],[189.,20.])

    def test_mirrored_line_satisfies_original_measured_point(self):
        abc=line_to_canonical([1,2,-50],200,True)
        point=np.array(points_to_canonical([[10,20]],200,True)[0])
        self.assertAlmostEqual(np.dot(point,abc[:2])+abc[2],0.)

    def test_unknown_S1_cannot_authorize_complete_path_or_line(self):
        front=self.pipeline(s1_valid=False)
        result=front.step(np.zeros((2,2,3),np.uint8),0.,0)
        self.assertIsNone(result['line_abc'])
        self.assertEqual(result['line_points_xy'],[])
        self.assertFalse(result['whole_path_candidate_supported'])

    def test_string_true_P1_flag_cannot_authorize_calibration(self):
        result=self.pipeline(p1_supported='true').step(np.zeros((2,2,3),np.uint8),0.,0)
        self.assertFalse(result['whole_path_candidate_supported'])

    def test_unobserved_path_segment_flags_preserved(self):
        front=self.pipeline()
        front.p1.result['diagnostics']['path_segment_support'][10]=False
        result=front.step(np.zeros((2,2,3),np.uint8),0.,0)
        self.assertFalse(result['diagnostics']['path_segment_support'][10])
        front.p1.result['diagnostics']['path_segment_support']=[]
        with self.assertRaises(ValueError):
            front.step(np.zeros((2,2,3),np.uint8),0.,0)

    def test_nonfinite_model_path_rejected(self):
        front=self.pipeline()
        front.p1.result['path_xy'][10][0]=float('nan')
        with self.assertRaises(ValueError):
            front.step(np.zeros((2,2,3),np.uint8),0.,0)

    def test_valid_S2_audit_cannot_replace_S1_or_promote_partial_P1(self):
        front=self.pipeline(p1_supported=False)
        front.p1.surface=np.zeros((2,2),np.uint8);front.p1.pitch=20.
        front.s2=StubS2(dict(valid=True,line_abc=[0,1,-100],whole_path_candidate_supported=True))
        original=np.zeros((2,2,3),np.uint8)
        result=front.step(original,0.,0)
        self.assertIs(front.s2.image,original)
        self.assertIsNone(front.s2.conditions['e1_xy'])
        self.assertEqual(result['line_abc'],line_to_canonical([1,2,-50],200,False))
        self.assertFalse(result['whole_path_candidate_supported'])
        self.assertFalse(result['diagnostics']['S2_used_for_C_or_whole_path_authorization'])

    def test_S2_valid_does_not_authorize_missing_S1_line(self):
        front=self.pipeline(s1_valid=False)
        front.p1.surface=np.zeros((2,2),np.uint8);front.p1.pitch=20.
        front.s2=StubS2(dict(valid=True,line_abc=[0,1,-100]))
        result=front.step(np.zeros((2,2,3),np.uint8),0.,0)
        self.assertIsNone(result['line_abc'])
        self.assertFalse(result['whole_path_candidate_supported'])


if __name__=='__main__':
    unittest.main()
