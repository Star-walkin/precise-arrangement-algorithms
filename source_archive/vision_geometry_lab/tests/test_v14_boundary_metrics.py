"""Synthetic-only anti-leakage and unit-scale checks for V14 boundary scoring."""
import unittest
import numpy as np

from vision_geometry_lab.v11_path_contract import HEIGHT,WIDTH,_geometry
from vision_geometry_lab.v14_boundary_metrics import decode_boundaries,score_boundaries,aggregate_boundary_scores


def geometry():
    return _geometry((2000,3000),[[700.,1000.],[900.,1000.]],10.)


def targets(columns=(10,11,12),low=100.,high=120.,center=110.):
    edge_rows=np.zeros((2,WIDTH),np.float32)
    edge_valid=np.zeros((2,WIDTH),bool)
    center_rows=np.zeros(WIDTH,np.float32)
    center_valid=np.zeros(WIDTH,bool)
    local_width=np.zeros(WIDTH,np.float32)
    for c in columns:
        edge_rows[:,c]=[low,high]
        edge_valid[:,c]=True
        center_rows[c],center_valid[c]=center,True
        local_width[c]=(high-low)*1.875
    return dict(edge_rows=edge_rows,edge_valid=edge_valid,center_rows=center_rows,center_valid=center_valid,
        width_valid=center_valid.copy(),local_width_px=local_width,
        edge_row_valid=np.zeros((2,HEIGHT,WIDTH),bool),edge_distribution=np.zeros((2,HEIGHT,WIDTH),np.float32))


def prediction(low=100,high=120):
    prob=np.zeros((2,HEIGHT,WIDTH),np.float32)
    prob[0,low,:]=1.
    prob[1,high,:]=1.
    return prob,np.ones((HEIGHT,WIDTH),bool)


class BoundaryMetricsV14Tests(unittest.TestCase):
    def test_perfect_edges_and_unknown_columns_are_not_background(self):
        p,v=prediction()
        result=score_boundaries(p,v,targets(),geometry())
        for edge in result['edges'].values():
            self.assertEqual(edge['trusted_columns'],3)
            self.assertEqual(edge['hit_5px_fraction'],1.)
            self.assertEqual(edge['error_median_px'],0.)
            self.assertEqual(edge['column_status'][0],'unknown')
        self.assertEqual(result['width']['mae_px'],0.)
        self.assertEqual(result['joint']['strict_center_bracket_fraction'],1.)
        self.assertIsNotNone(result['decoded']['edge_rows'][0][0])

    def test_gt_masks_and_targets_cannot_change_predicted_argmax(self):
        p,v=prediction(80,140)
        p[0,100,:]=.5
        p[1,120,:]=.5
        a=targets()
        a['edge_row_valid'][:,98:123,:]=True
        a['edge_distribution'][:,110,:]=1.
        b=targets(columns=(11,12),low=80.,high=140.)
        b['edge_row_valid'][:]=True
        b['edge_distribution'][:]=1.
        first=score_boundaries(p,v,a,geometry())
        second=score_boundaries(p,v,b,geometry())
        self.assertEqual(first['decoded'],second['decoded'])
        self.assertEqual(first['decoded']['edge_rows'][0][10],80.)
        self.assertEqual(first['edges']['low_normal']['hit_5px_fraction'],0.)
        self.assertEqual(second['edges']['low_normal']['hit_5px_fraction'],1.)

    def test_image_valid_is_the_only_spatial_argmax_mask(self):
        p,v=prediction(80,140)
        p[0,100,:]=.5
        p[1,120,:]=.5
        v[80,:]=False
        v[140,:]=False
        result=score_boundaries(p,v,targets(),geometry())
        self.assertEqual(result['decoded']['edge_rows'][0][10],100.)
        self.assertEqual(result['decoded']['edge_rows'][1][10],120.)
        self.assertEqual(result['width']['mae_px'],0.)

    def test_crossed_edges_never_sorted_or_repaired(self):
        p,v=prediction(120,100)
        result=score_boundaries(p,v,targets(),geometry())
        self.assertEqual(result['joint']['crossing_fraction_among_available'],1.)
        self.assertEqual(result['joint']['strict_center_bracket_fraction'],0.)
        self.assertEqual(result['width']['raw_signed_predicted_widths_px'],[-37.5]*3)
        self.assertEqual(result['width']['mae_px'],75.)
        self.assertEqual(result['width']['mean_absolute_relative_error'],2.)
        self.assertEqual(result['width']['nonpositive_raw_prediction_count'],3)

    def test_coincident_edges_not_valid_center_bracketing(self):
        p,v=prediction(110,110)
        result=score_boundaries(p,v,targets(),geometry())
        self.assertEqual(result['joint']['crossing_count'],0)
        self.assertEqual(result['joint']['coincident_count'],3)
        self.assertEqual(result['joint']['strict_center_bracket_count'],0)
        self.assertEqual(result['width']['nonpositive_raw_prediction_count'],3)

    def test_original_pixel_tolerance_not_strip_pixel_tolerance(self):
        p,v=prediction(103,122)
        result=score_boundaries(p,v,targets(),geometry())
        self.assertEqual(result['edges']['low_normal']['error_median_px'],5.625)
        self.assertEqual(result['edges']['low_normal']['hit_5px_fraction'],0.)
        self.assertEqual(result['edges']['high_normal']['hit_5px_fraction'],1.)
        self.assertEqual(result['width']['mae_px'],1.875)

    def test_missing_predictions_remain_in_hit_denominator(self):
        p,v=prediction()
        p[:,:,10]=0.
        result=score_boundaries(p,v,targets(),geometry())
        self.assertEqual(result['edges']['low_normal']['hit_5px_fraction'],2/3)
        self.assertEqual(result['edges']['low_normal']['missing_prediction_columns'],1)
        self.assertEqual(result['edges']['low_normal']['column_status'][10],'missing_prediction')
        self.assertEqual(result['joint']['strict_center_bracket_fraction'],2/3)
        self.assertEqual(result['width']['predicted_trusted_columns'],2)
        self.assertEqual(result['width']['missing_prediction_columns'],1)

    def test_all_image_invalid_has_unknown_prediction_not_success(self):
        p,v=prediction()
        v[:]=False
        result=score_boundaries(p,v,targets(),geometry())
        self.assertIsNone(result['decoded']['edge_rows'][0][10])
        self.assertEqual(result['edges']['low_normal']['hit_5px_fraction'],0.)
        self.assertIsNone(result['edges']['low_normal']['error_median_px'])
        self.assertIsNone(result['width']['mae_px'])
        self.assertIsNone(result['joint']['crossing_fraction_among_available'])

    def test_nan_untrusted_coordinates_ignored_and_empty_scope_unknown(self):
        p,v=prediction()
        t=targets(columns=())
        for k in ('edge_rows','center_rows','local_width_px'):
            t[k][:]=np.nan
        result=score_boundaries(p,v,t,geometry())
        self.assertIsNone(result['edges']['low_normal']['hit_5px_fraction'])
        self.assertIsNone(result['width']['mae_px'])
        self.assertEqual(result['edges']['low_normal']['column_status'][10],'unknown')

    def test_each_side_has_independent_trusted_columns(self):
        p,v=prediction()
        t=targets()
        t['edge_valid'][1,11]=False
        t['width_valid'][11]=False
        result=score_boundaries(p,v,t,geometry())
        self.assertEqual(result['edges']['low_normal']['trusted_columns'],3)
        self.assertEqual(result['edges']['high_normal']['trusted_columns'],2)
        self.assertEqual(result['joint']['trusted_both_columns'],2)
        self.assertEqual(result['width']['trusted_columns'],2)

    def test_pool_observations_not_per_image_medians(self):
        first=score_boundaries(*prediction(),targets(columns=(10,)),geometry())
        second=score_boundaries(*prediction(104,124),targets(),geometry())
        pooled=aggregate_boundary_scores([first,second])
        edge=pooled['edges']['low_normal']
        self.assertEqual(edge['trusted_columns'],4)
        self.assertEqual(edge['hit_5px_fraction'],.25)
        self.assertEqual(edge['error_median_px'],7.5)
        self.assertEqual(edge['error_p95_px'],7.5)
        empty=aggregate_boundary_scores([])
        self.assertIsNone(empty['width']['mae_px'])

    def test_invalid_probability_or_target_contract_fails_closed(self):
        p,v=prediction()
        with self.assertRaises(ValueError):
            decode_boundaries(-p,v)
        invalid=p.copy();invalid[0,0,0]=np.nan
        with self.assertRaises(ValueError):
            decode_boundaries(invalid,v)
        with self.assertRaises(ValueError):
            decode_boundaries(p,v[:10])
        t=targets();t['local_width_px'][10]=1.
        with self.assertRaises(ValueError):
            score_boundaries(p,v,t,geometry())
        t=targets();t['edge_rows'][0,10]=np.nan
        with self.assertRaises(ValueError):
            score_boundaries(p,v,t,geometry())


if __name__=='__main__':
    unittest.main()
