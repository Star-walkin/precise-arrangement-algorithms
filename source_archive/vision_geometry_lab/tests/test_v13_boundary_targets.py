"""Synthetic-only tests: independent boundaries, no extrapolation or unknown fill."""
import unittest
import numpy as np

from vision_geometry_lab.v11_path_contract import (
    HEIGHT,WIDTH,_geometry,strip_to_image,build_targets,array_sha256)
from vision_geometry_lab.v13_boundary_targets import build_boundary_targets,targets_for_annotation,_crosses_polygon


def geometry():
    return _geometry((2000,3000),[[700.,1000.],[900.,1000.]],10.)


def line(row,start=20.,end=300.):
    return strip_to_image([[start,row],[end,row]],geometry()).tolist()


def run_targets(center=None,flange=None,surface=None,ignored=(),valid=None):
    return build_boundary_targets(geometry(),line(120.) if center is None else center,
        line(115.) if flange is None else flange,line(125.) if surface is None else surface,
        list(ignored),np.ones((HEIGHT,WIDTH),bool) if valid is None else valid)


class BoundaryTargetsV13Tests(unittest.TestCase):
    def test_exact_center_sorted_edges_width_and_v11_compatibility(self):
        data=run_targets()
        self.assertEqual(int(data['center_valid'].sum()),281)
        self.assertEqual(int(data['width_valid'].sum()),281)
        np.testing.assert_allclose(data['edge_rows'][:,100],[115.,125.])
        self.assertAlmostEqual(float(data['local_width_px'][100]),18.75)
        self.assertAlmostEqual(float(data['local_width_free_units'][100]),1.875)
        previous=build_targets(geometry(),line(120.),[],np.ones((HEIGHT,WIDTH),bool))
        for key in ('target_distribution','column_valid','row_valid','slope_target','slope_valid'):
            np.testing.assert_array_equal(data[key],previous[key])
        for side in (0,1):
            np.testing.assert_allclose(data['edge_distribution'][side,:,data['edge_valid'][side]].sum(1),1.,atol=1e-6)

    def test_semantic_names_do_not_define_normal_order(self):
        data=run_targets(flange=line(125.),surface=line(115.))
        np.testing.assert_array_equal(data['edge_source'][:,100],[2,1])
        np.testing.assert_allclose(data['edge_rows'][:,100],[115.,125.])

    def test_single_edge_independent_without_invented_other_edge(self):
        data=run_targets(surface=[])
        self.assertTrue(data['center_valid'][100])
        self.assertTrue(data['edge_valid'][0,100])
        self.assertFalse(data['edge_valid'][1].any())
        self.assertFalse(data['edge_distribution'][1].any())
        self.assertFalse(data['width_valid'].any())
        self.assertFalse(data['local_width_px'].any())

    def test_single_edge_without_center_retains_semantic_but_unknown_side(self):
        data=run_targets(center=[],surface=[])
        self.assertTrue(data['edge_flange_valid'][100])
        self.assertFalse(data['edge_valid'].any())
        self.assertFalse(data['center_valid'].any())
        self.assertFalse(data['width_valid'].any())

    def test_two_edges_independent_of_missing_center_but_no_width(self):
        data=run_targets(center=[])
        self.assertTrue(data['edge_valid'][:,100].all())
        self.assertFalse(data['width_valid'].any())

    def test_finite_edge_ends_are_not_extrapolated(self):
        data=run_targets(surface=line(125.,60.,140.))
        self.assertTrue(data['center_valid'][200])
        self.assertFalse(data['edge_surface_valid'][200])
        self.assertFalse(data['width_valid'][200])
        self.assertEqual(int(data['width_valid'].sum()),81)

    def test_multiple_edge_intersections_unknown_not_nearest_selected(self):
        edge=strip_to_image([[20.,115.],[250.,113.],[20.,100.]],geometry()).tolist()
        data=run_targets(flange=edge)
        self.assertFalse(data['edge_flange_valid'][20:250].any())
        self.assertTrue(data['edge_surface_valid'][100])
        self.assertFalse(data['width_valid'][100])

    def test_ignore_crossing_between_valid_targets_invalidates_width(self):
        polygon=strip_to_image([[80.,134.2],[140.,134.2],[140.,134.3],[80.,134.3]],geometry()).tolist()
        data=run_targets(flange=line(100.),surface=line(145.),ignored=[polygon])
        self.assertTrue(data['center_valid'][100])
        self.assertTrue(data['edge_valid'][:,100].all())
        self.assertFalse(data['width_valid'][100])
        self.assertTrue(data['width_valid'][200])
        self.assertTrue(_crosses_polygon(100.,100.,145.,np.array([[80.,134.2],[140.,134.2],[140.,134.3],[80.,134.3]])))

    def test_unknown_one_edge_does_not_erase_center_or_other_edge(self):
        polygon=strip_to_image([[80.,98.],[140.,98.],[140.,102.],[80.,102.]],geometry()).tolist()
        data=run_targets(flange=line(100.),surface=line(145.),ignored=[polygon])
        self.assertTrue(data['center_valid'][100])
        self.assertFalse(data['edge_flange_valid'][100])
        self.assertTrue(data['edge_surface_valid'][100])
        self.assertTrue(data['edge_valid'][1,100])
        self.assertFalse(data['width_valid'][100])

    def test_padding_between_valid_targets_invalidates_width(self):
        valid=np.ones((HEIGHT,WIDTH),bool)
        valid[133,80:140]=False
        snapshot=array_sha256(valid)
        data=run_targets(flange=line(100.),surface=line(145.),valid=valid)
        self.assertTrue(data['center_valid'][100])
        self.assertTrue(data['edge_valid'][:,100].all())
        self.assertFalse(data['width_valid'][100])
        self.assertEqual(snapshot,array_sha256(valid))

    def test_center_outside_edges_no_width_but_independent_supervision(self):
        data=run_targets(flange=line(130.),surface=line(140.))
        self.assertTrue(data['center_valid'][100])
        self.assertTrue(data['edge_valid'][:,100].all())
        self.assertFalse(data['width_valid'].any())

    def test_quality_c7_style_cutoff_applies_to_center_and_both_edges(self):
        original={name:line(row,end=300.) for name,row in [('centerline',120.),('edge_flange',115.),('edge_surface',125.)]}
        trusted={name:line(row,end=170.) for name,row in [('centerline',120.),('edge_flange',115.),('edge_surface',125.)]}
        label=dict(status='partial',polylines=original,ignore_polygons=[])
        quality=dict(supervision_polylines=trusted,precision_exclude_polygons=[])
        data=targets_for_annotation(geometry(),label,quality,np.ones((HEIGHT,WIDTH),bool))
        self.assertTrue(data['width_valid'][170])
        self.assertFalse(data['center_valid'][171:].any())
        self.assertFalse(data['edge_valid'][:,171:].any())
        self.assertFalse(data['width_valid'][171:].any())
        self.assertEqual(len(label['polylines']['centerline']),2)

    def test_empty_nonvisible_annotations_and_bad_geometry_fail_closed(self):
        data=run_targets(center=[],flange=[],surface=[])
        for key in ('center_valid','edge_valid','width_valid','target_distribution','edge_distribution'):
            self.assertFalse(data[key].any())
        label=dict(status='absent',polylines=dict(centerline=line(120.),edge_flange=line(115.),edge_surface=line(125.)))
        data=targets_for_annotation(geometry(),label,{},np.ones((HEIGHT,WIDTH),bool))
        self.assertFalse(data['center_valid'].any())
        self.assertFalse(data['edge_valid'].any())
        with self.assertRaises(ValueError):
            run_targets(valid=np.ones((10,10),bool))


if __name__=='__main__':
    unittest.main()
