import math
import numpy as np
from vision_geometry_lab.conic_geometry import ellipse_line_intersections,measure_conic_relation


def model(center=(100,200),axes=(80,40),angle=0):
    return dict(valid=True,center=list(center),semi_axes=list(axes),angle_rad=angle,
                arc_interval_rad=[-math.pi,0],residual_px=.15,sensitivity_px=.2)


def rope(rho=160,width=6):
    return dict(valid=True,normal=[0,1],rho_px=rho,width_px=width,
                residual_px=.2,observed_segment=[[10,rho],[180,rho]])


def test_two_one_zero_intersections_are_not_all_tangencies():
    m = model()
    assert ellipse_line_intersections(m,[0,1],170)['complete_intersection_count']==2
    assert ellipse_line_intersections(m,[0,1],160)['complete_intersection_count']==1
    assert ellipse_line_intersections(m,[0,1],150)['complete_intersection_count']==0
    assert not measure_conic_relation(m,rope(180))['last_turn_candidate']


def test_one_visible_intersection_is_not_a_tangent():
    m = model()
    result = ellipse_line_intersections(m,[1,0],100)
    assert result['complete_intersection_count']==2
    assert result['visible_arc_intersection_count']==1


def test_gap_tolerance_does_not_reward_bad_fit():
    m = model(); m['sensitivity_px']=40
    result = measure_conic_relation(m,rope(160))
    assert result['status']=='uncertainty_too_large' and not result['last_turn_candidate']


def test_exact_and_near_no_root_have_qualified_candidate_only():
    for rho in [160,159.9,160.1]:
        result = measure_conic_relation(model(),rope(rho))
        assert result['center_tangency_candidate'],result
        assert not result['last_turn_confirmed'] and not result['new_layer_confirmed']
        assert result['force_command'] is None


def test_rope_width_is_not_required_for_centerline_geometry():
    result = measure_conic_relation(model(),rope(160,None))
    assert result['center_tangency_candidate']
    assert result['band_edge_distances_px'] is None


def test_large_translation_stability_and_rotation():
    for theta in [0,.3,1.5]:
        R = np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
        n = R@np.array([0.,1.])
        center = np.array([100000.,200000.])
        rho = float(n@center)-40
        result = ellipse_line_intersections(model(center,angle=theta),n,rho)
        assert result['complete_intersection_count']==1,result


def test_chord_threshold_changes_with_scale():
    small = measure_conic_relation(model(),rope(160))
    large = model(axes=(160,80))
    bigger = measure_conic_relation(large,rope(120))
    assert bigger['chord_merge_threshold_px']>small['chord_merge_threshold_px']


def test_hidden_opposite_tangent_not_used():
    # A line very close to the hidden bottom does not become a near-edge event.
    result = measure_conic_relation(model(),rope(240))
    assert not result['last_turn_candidate']


def test_angle_error_propagates_inside_observed_segment():
    r = rope(160)
    r.update(observed_segment=[[100,160],[1100,160]],angular_uncertainty_deg=1,
             diagnostics={'uncertainty_reference_point':[600,160]})
    result = measure_conic_relation(model(),r)
    assert result['line_extrapolation_px']==0
    assert result['angular_lever_arm_px']==500
    assert not result['quality_sufficient'] and not result['last_turn_candidate']


def test_local_model_sensitivity_not_hidden_by_global_percentile():
    m = model()
    m.update(model_points=[[100,160]],model_sensitivity_px=[50])
    result = measure_conic_relation(m,rope())
    assert result['candidate_model_sensitivity_px']==50
    assert not result['last_turn_candidate']


def test_varying_optical_width_is_not_a_precise_edge():
    r = rope(150,20); r['width_interval_px']=[10,30]
    result = measure_conic_relation(model(),r)
    assert result['optical_band_tangency_possible']
    assert not result['optical_band_tangency_candidate'] and not result['last_turn_candidate']


def test_invalid_zero_normal_abstains():
    r = rope(); r['normal']=[0,0]
    assert not measure_conic_relation(model(),r)['valid']
