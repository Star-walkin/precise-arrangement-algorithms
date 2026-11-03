"""Synthetic-only pure four-column/authorized-phase comparison tests."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from vision_geometry_lab.v15_weight_visuals import figure,phase_arms


def references():
    return dict(centerline=[[[50.,120.],[170.,120.]],[[210.,125.],[330.,140.]]],
        edge_flange=[[[50.,110.],[170.,110.]],[[210.,115.],[330.,130.]]],
        edge_surface=[[[50.,130.],[170.,130.]],[[210.,135.],[330.,150.]]])


class WeightVisualsV15Tests(unittest.TestCase):
    def test_four_columns_have_identical_crops_scale_and_fixed_arm_order(self):
        called=[]
        image=Image.new('RGB',(400,300),(80,90,100))
        ref=references()
        arms=phase_arms('single')
        pred={arm:dict(tag=arm) for arm in arms}
        def fake_panel(rgb,crop,gt,prediction=None,size=(600,325),zoom=False):
            self.assertIs(rgb,image)
            self.assertIs(gt,ref)
            called.append((list(crop),None if prediction is None else prediction['tag'],zoom,size))
            return Image.new('RGB',size,(80,90,100))
        with tempfile.TemporaryDirectory() as folder,patch('vision_geometry_lab.v15_weight_visuals.panel',side_effect=fake_panel):
            path=Path(folder)/'comparison.png'
            result=figure(image,dict(sample_id='synthetic'),'single',ref,ref['centerline'],pred,path)
            self.assertEqual(Image.open(path).size,(2450,1270))
        self.assertEqual(len(called),12)
        for row in range(3):
            block=called[row*4:row*4+4]
            self.assertEqual([v[1] for v in block],[None,*arms])
            self.assertTrue(all(v[0]==block[0][0] and v[2:]==block[0][2:] for v in block))
        self.assertEqual(result['arm_order'],['center_only','joint','joint_low'])
        self.assertFalse(result['raw_edges_joined'])

    def test_holdout_has_only_authorized_zero_and_point_one(self):
        self.assertEqual(phase_arms('holdout'),('center_only','joint_low'))
        ref=references();image=Image.new('RGB',(400,300))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'held_synthetic.png'
            valid={arm:dict(center_segments=[],raw_edge_points=[[],[]]) for arm in phase_arms('holdout')}
            result=figure(image,dict(sample_id='synthetic'),'holdout',ref,ref['centerline'],valid,path)
            self.assertEqual(Image.open(path).size,(1840,1270))
            self.assertFalse(result['old_weight_05_holdout_evaluated'])
            invalid=dict(valid,joint=dict(center_segments=[],raw_edge_points=[[],[]]))
            with self.assertRaises(ValueError):
                figure(image,dict(sample_id='synthetic'),'holdout',ref,ref['centerline'],invalid,path)

    def test_missing_arm_is_not_silently_drawn_as_empty_prediction(self):
        ref=references()
        with self.assertRaises(ValueError):
            figure(Image.new('RGB',(400,300)),dict(sample_id='synthetic'),'train',ref,ref['centerline'],{},Path('never_created.png'))
        with self.assertRaises(ValueError):
            phase_arms('unknown')


if __name__=='__main__':
    unittest.main()
