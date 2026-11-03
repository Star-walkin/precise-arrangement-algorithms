"""Synthetic-only denominator/unknown-scope regressions; no field data."""
import unittest
import numpy as np

from vision_geometry_lab.v11_path_contract import HEIGHT,WIDTH,_geometry,strip_to_image
from vision_geometry_lab.v13_boundary_targets import targets_for_annotation
from vision_geometry_lab.v14_reference_support import boundary_reference_support,_input_supported


def geometry():
    return _geometry((3000,4000),[[500.,1000.],[700.,1000.]],10.)


def line(row,start=50.,end=300.):
    return strip_to_image([[start,row],[end,row]],geometry()).tolist()


def label(start=50.,end=300.):
    return dict(status='clear',ignore_polygons=[],polylines=dict(free_seed=geometry()['free_seed_xy'],
        centerline=line(120.,start,end),edge_flange=line(110.,start,end),edge_surface=line(130.,start,end)))


def targets():
    return dict(edge_flange_valid=np.ones(WIDTH,bool),edge_surface_valid=np.ones(WIDTH,bool),
        edge_source=np.repeat(np.array([[1],[2]]),WIDTH,axis=1),edge_valid=np.ones((2,WIDTH),bool),
        diagnostics=dict(edge_flange=dict(ambiguous_columns=0),edge_surface=dict(ambiguous_columns=0)))


def score(l=None,q=None,t=None,image_valid=None):
    return boundary_reference_support(label() if l is None else l,{} if q is None else q,
        geometry(),np.ones((HEIGHT,WIDTH),bool) if image_valid is None else image_valid,
        targets() if t is None else t)


class ReferenceSupportV14Tests(unittest.TestCase):
    def test_semantic_target_without_head_assignment_is_not_scorable(self):
        t=targets();t['edge_source'][0,100:201]=0
        result=score(t=t)['edges']['edge_flange']
        self.assertLess(result['scorable_fraction_of_trusted'],.61)
        self.assertAlmostEqual(result['semantic_scorable_fraction_of_trusted'],1.)
        self.assertEqual(result['semantic_valid_but_unassigned_columns'],101)

    def test_complete_trusted_reference_all_supported(self):
        result=score()
        for edge in result['edges'].values():
            self.assertAlmostEqual(edge['trusted_downstream_length_px'],250*1.875)
            self.assertAlmostEqual(edge['input_support_fraction'],1.)
            self.assertAlmostEqual(edge['scorable_fraction_of_trusted'],1.)
            self.assertFalse(edge['denominator_restricted_to_target_success'])

    def test_out_of_window_reference_stays_in_denominator(self):
        result=score(l=label(400.,650.))
        edge=result['edges']['edge_flange']
        self.assertAlmostEqual(edge['trusted_downstream_length_px'],250*1.875)
        self.assertLess(edge['input_support_fraction'],.46)
        self.assertGreater(edge['input_support_fraction'],.43)
        self.assertGreater(edge['outside_strip_midpoints'],0)
        self.assertAlmostEqual(edge['input_support_fraction'],edge['scorable_fraction_of_trusted'])

    def test_failed_target_does_not_remove_denominator_or_select_nearest_valid_column(self):
        t=targets()
        t['edge_flange_valid'][100:201]=False
        result=score(t=t)
        edge=result['edges']['edge_flange']
        self.assertAlmostEqual(edge['trusted_downstream_length_px'],250*1.875)
        self.assertAlmostEqual(edge['input_support_fraction'],1.)
        self.assertGreater(edge['scorable_fraction_of_trusted'],.58)
        self.assertLess(edge['scorable_fraction_of_trusted'],.61)
        self.assertAlmostEqual(result['edges']['edge_surface']['scorable_fraction_of_trusted'],1.)

    def test_multiple_intersections_remain_denominator_with_unknown_targets(self):
        l=label()
        l['polylines']['edge_flange']=strip_to_image([[50.,100.],[250.,110.],[50.,115.]],geometry()).tolist()
        valid=np.ones((HEIGHT,WIDTH),bool)
        t=targets_for_annotation(geometry(),l,{},valid)
        self.assertFalse(t['edge_flange_valid'][50:250].any())
        result=score(l=l,t=t)
        edge=result['edges']['edge_flange']
        self.assertGreater(edge['trusted_downstream_length_px'],750.)
        self.assertAlmostEqual(edge['input_support_fraction'],1.)
        self.assertLess(edge['scorable_fraction_of_trusted'],.01)
        self.assertGreater(result['target_builder_diagnostics']['edge_flange']['ambiguous_columns'],100)

    def test_quality_hole_splits_reference_without_unknown_bridge(self):
        l=label(50.,250.)
        hole=strip_to_image([[100.,90.],[200.,90.],[200.,145.],[100.,145.]],geometry()).tolist()
        result=score(l=l,q=dict(precision_exclude_polygons=[hole]))
        for edge in result['edges'].values():
            self.assertEqual(edge['trusted_piece_count'],2)
            self.assertAlmostEqual(edge['trusted_downstream_length_px'],100*1.875)
            np.testing.assert_allclose(edge['trusted_piece_lengths_px'],[50*1.875,50*1.875])
            self.assertAlmostEqual(edge['scorable_fraction_of_trusted'],1.)

    def test_original_ignore_and_quality_exclusion_both_apply(self):
        l=label(50.,250.)
        def polygon(a,b):
            return strip_to_image([[a,90.],[b,90.],[b,145.],[a,145.]],geometry()).tolist()
        l['ignore_polygons']=[polygon(80.,100.)]
        result=score(l=l,q=dict(precision_exclude_polygons=[polygon(170.,200.)]))
        edge=result['edges']['edge_flange']
        self.assertEqual(edge['trusted_piece_count'],3)
        self.assertAlmostEqual(edge['trusted_downstream_length_px'],150*1.875)

    def test_c7_override_replaces_both_full_edges(self):
        l=label(50.,300.)
        q=dict(supervision_polylines=dict(centerline=line(120.,50.,170.),
            edge_flange=line(110.,50.,170.),edge_surface=line(130.,50.,170.)),precision_exclude_polygons=[])
        result=score(l=l,q=q)
        self.assertEqual(result['reference_source'],'hash_bound_quality_polylines')
        for edge in result['edges'].values():
            self.assertAlmostEqual(edge['trusted_downstream_length_px'],120*1.875)

    def test_seed_plane_clips_in_native_geometry_not_target_success(self):
        result=score(l=label(0.,100.))
        origin_column=8./.1875-.5
        edge=result['edges']['edge_flange']
        self.assertAlmostEqual(edge['trusted_downstream_length_px'],(100.-origin_column)*1.875)

    def test_bilinear_mask_requires_full_support_not_nearest_pixel(self):
        valid=np.ones((HEIGHT,WIDTH),bool)
        valid[110,100]=False
        points=np.array([[99.1,110.],[99.,110.],[100.1,110.],[100.,109.5],[101.,110.]])
        np.testing.assert_array_equal(_input_supported(points,valid),[False,True,False,False,True])
        result=score(image_valid=valid)
        self.assertLess(result['edges']['edge_flange']['input_support_fraction'],1.)
        self.assertAlmostEqual(result['edges']['edge_surface']['input_support_fraction'],1.)

    def test_empty_or_absent_reference_not_success(self):
        l=label();l['polylines']['edge_flange']=[]
        result=score(l=l)
        self.assertIsNone(result['edges']['edge_flange']['scorable_fraction_of_trusted'])
        self.assertEqual(result['edges']['edge_flange']['trusted_downstream_length_px'],0.)
        l=label();l['status']='absent'
        result=score(l=l)
        self.assertIsNone(result['combined']['scorable_fraction_of_trusted'])

    def test_bad_mask_fails_closed(self):
        with self.assertRaises(ValueError):
            score(image_valid=np.ones((10,10),bool))
        t=targets();t['edge_flange_valid']=np.full(WIDTH,.5)
        with self.assertRaises(ValueError):
            score(t=t)


if __name__=='__main__':
    unittest.main()
