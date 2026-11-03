import numpy as np
from vision_geometry_lab.v23_path_prefix_audit import audit_coverage,departure_diagnostic,seed_plane_origin,cell_overlap


def test_bins_preserve_length_and_old_sample_weight_coverage():
    ref=[[-5,0],[100,0]];seed=[[-5,0],[0,0]];pred=[[[0,0],[100,0]]]
    result=audit_coverage(ref,seed,2.,[[[0,0],[100,0]]],pred)
    assert [c['reference_length_px'] for c in result['bins']]==[8,8,16,32,36]
    assert sum(c['covered_reference_length_px'] for c in result['bins'])==100
    assert all(c['coverage_5px']==1 for c in result['bins'])


def test_ignored_gap_keeps_original_arc_position_not_compressed():
    result=audit_coverage([[-5,0],[100,0]],[[-5,0],[0,0]],2.,[[[0,0],[10,0]],[[20,0],[100,0]]],[])
    assert [c['reference_length_px'] for c in result['bins']]==[8,2,12,32,36]
    assert sum(c['reference_length_px'] for c in result['bins'])==90
    assert all(c['coverage_5px']==0 for c in result['bins'])


def test_missing_reference_bins_are_unknown_not_success():
    result=audit_coverage([[-5,0],[5,0]],[[-5,0],[0,0]],2.,[[[0,0],[5,0]]],[])
    assert result['bins'][0]['coverage_5px']==0
    assert all(c['coverage_5px'] is None for c in result['bins'][1:])


def test_missing_width_does_not_remove_reference_denominator():
    r=audit_coverage([[-5,0],[20,0]],[[-5,0],[0,0]],None,[[[0,0],[20,0]]],[])
    assert r['unknown_binned_reference_length_px']==20
    assert all(c['coverage_5px'] is None for c in r['bins'])


def test_turn_back_and_missing_seed_plane_do_not_guess_origin():
    assert seed_plane_origin([[-1,0],[5,0],[3,2]],[[-1,0],[0,0]])[0] is None
    assert seed_plane_origin([[5,0],[10,0]],[[-1,0],[0,0]])[0] is None


def test_departure_is_gt_interpolated_diagnostic_not_contact():
    ref=[[-10,0],[10,0],[20,10]];seed=[[-10,0],[0,0]];support=[[[0,0],[10,0],[20,10]]]
    origin,_=seed_plane_origin(ref,seed)
    r=departure_diagnostic(ref,seed,4.,support,support,origin)
    np.testing.assert_allclose(r['point_xy'],[12,2])
    assert r['prediction_hits_5px'] and not r['is_contact'] and not r['used_to_crop_inference']


def test_cell_intersection_conserves_voronoi_weights():
    edges=[0,2,6,9]
    np.testing.assert_allclose(cell_overlap(edges,0,4)+cell_overlap(edges,4,np.inf),np.diff(edges))
