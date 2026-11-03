"""Synthetic-only renderer and fail-closed gate tests, no field images."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image

from vision_geometry_lab.v11_path_contract import _geometry
from vision_geometry_lab.v14_joint_visuals import raw_edge_points,panel,_check_gate,crop_for


class JointVisualsV14Tests(unittest.TestCase):
    def test_raw_edges_keep_order_and_missing_points_without_connections(self):
        geometry=_geometry((2000,3000),[[500.,1000.],[700.,1000.]],10.)
        rows=[[None]*512 for _ in range(2)]
        valid=[[False]*512 for _ in range(2)]
        rows[0][100],rows[1][100]=150.,100.
        rows[0][102]=120.
        valid[0][100]=valid[1][100]=valid[0][102]=True
        points=raw_edge_points(dict(edge_rows=rows,edge_valid=valid),geometry)
        self.assertEqual(len(points[0]),2)
        self.assertEqual(len(points[1]),1)
        self.assertGreater(points[0][0,1],points[1][0,1])

    def test_missing_or_failed_gate_blocks_before_any_field_access(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            with self.assertRaisesRegex(ValueError,'absent'):_check_gate(path,'train','p')
            (path/'gate_train.json').write_text(json.dumps(dict(protocol_sha256='p',
                decision=dict(passed=False,checks=dict(center=False)))),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'did not pass'):_check_gate(path,'train','p')

    def test_native_crop_is_same_and_letterbox_cannot_receive_geometry(self):
        image=Image.new('RGB',(100,100),(80,90,100))
        references=dict(centerline=[[[0.,50.],[100.,50.]]],edge_flange=[],edge_surface=[])
        crop=crop_for([50.,50.],40,80,image.size)
        a=panel(image,crop,references,size=(100,100))
        b=panel(image,crop,references,prediction=dict(center_segments=[],raw_edge_points=[[],[]]),size=(100,100))
        np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
        self.assertEqual(a.getpixel((5,50)),(20,27,38))
        self.assertNotEqual(a.getpixel((50,50)),(20,27,38))


if __name__=='__main__':
    unittest.main()
