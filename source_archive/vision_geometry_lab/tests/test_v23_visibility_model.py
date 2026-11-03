import unittest
import numpy as np
try:
    import torch
except ImportError as exc:
    raise unittest.SkipTest('Run this torch-only module with vision_training/.venv and unittest') from exc
from vision_geometry_lab.v23_visibility_model import (VisibilityHead, local_features,
    gate_probabilities, intervention, supervision, masked_loss)


class VisibilityTests(unittest.TestCase):
    def setUp(self):
        self.x = np.zeros((8,256,512), np.float32)
        self.x[:3] = .45
        self.x[4,127,:40] = 1
        self.x[7] = 1

    def test_gate_mass_not_renormalized(self):
        p=torch.ones(1,10,5)/10; v=torch.tensor([[.9,.5,.49,0.,1.]])
        got=gate_probabilities(p,v)
        torch.testing.assert_close(got.sum(1),torch.tensor([[.9,.5,0,0,1.]]))

    def test_gate_unknown_no_coordinates(self):
        self.assertEqual(float(gate_probabilities(torch.ones(1,3,4)/3,torch.zeros(1,4)).sum()),0)

    def test_same_crop_zero_query(self):
        result=intervention(self.x,'zero_query')['input']
        np.testing.assert_array_equal(result[[0,1,2,3,5,6,7]],self.x[[0,1,2,3,5,6,7]])
        self.assertFalse(result[4].any())

    def test_synthetic_never_changes_geometry_or_query(self):
        result=intervention(self.x,'synthetic_occlusion',44)
        np.testing.assert_array_equal(result['input'][4:],self.x[4:])
        self.assertTrue(result['erased'].any())
        self.assertTrue(np.isfinite(result['input']).all())

    def test_synthetic_reproducible_and_heldout_texture_distinct(self):
        a=intervention(self.x,'synthetic_occlusion',55)
        b=intervention(self.x,'synthetic_occlusion',55)
        c=intervention(self.x,'synthetic_occlusion',55,heldout=True)
        np.testing.assert_array_equal(a['input'],b['input'])
        np.testing.assert_array_equal(a['erased'],c['erased'])
        self.assertFalse(np.array_equal(a['input'],c['input']))

    def test_unknown_tail_never_negative(self):
        trusted=np.zeros(512,bool); trusted[30:200]=True
        for kind in ('original','zero_query','synthetic_occlusion'):
            v=intervention(self.x,kind,1); y,m=supervision(trusted,v)
            self.assertFalse(m[200:].any())
            self.assertFalse(m[:30].any())
            if kind=='synthetic_occlusion':
                self.assertFalse(m[v['guard']].any())
                self.assertTrue((y[v['erased']]==0).all())

    def test_loss_unknown_nan_does_not_leak(self):
        x=torch.tensor([[1.,float('nan')]],requires_grad=True)
        y=torch.tensor([[1.,float('nan')]])
        loss=masked_loss(x,y,torch.tensor([[1,0]]))
        loss.backward(); self.assertEqual(float(x.grad[0,1]),0.)

    def test_features_detached_no_gt_api(self):
        x=torch.from_numpy(self.x[None]); dense=torch.rand(1,8,256,512,requires_grad=True)
        p=torch.ones(1,256,512)/256
        features=local_features(dense,p,x)
        self.assertEqual(tuple(features.shape),(1,30,512))
        self.assertFalse(features.requires_grad)
        out=VisibilityHead()(features); self.assertEqual(tuple(out.shape),(1,512))

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError): intervention(self.x,'bad')
        with self.assertRaises(ValueError): gate_probabilities(torch.ones(1,3,4),torch.ones(1,3))


if __name__=='__main__': unittest.main()
