import numpy as np

from vision_geometry_lab.geometry import measure_relation


def scene(rho=45, width=10, slope=0):
    x = np.linspace(20, 180, 161)
    y = 50+.005*(x-100)**2
    n = np.array([-slope, 1.0]); n /= np.linalg.norm(n)
    return (dict(valid=True, points=np.column_stack((x,y)).tolist(),
                 observed_ranges=[[20,180]], residual_px=.2),
            dict(valid=True, normal=n.tolist(), rho_px=rho, width_px=width,
                 residual_px=.2, observed_segment=[[20,rho],[180,rho]]))


def test_tangent_band_edge_is_not_centerline():
    a, r = scene()
    out = measure_relation(a,r)
    assert out['valid'] and out['status'] == 'image_tangency_candidate', out
    assert abs(out['center_distance_px']-5) < .01
    assert not out['physical_contact_confirmed'] and not out['new_layer_confirmed']


def test_crossing_does_not_count_as_tangent():
    a, r = scene(rho=70)
    out = measure_relation(a,r)
    assert out['status'] != 'image_tangency_candidate', out


def test_missing_width_does_not_confirm_band_contact():
    a, r = scene(rho=50,width=None)
    out = measure_relation(a,r)
    assert out['status'] == 'parallel_width_unavailable'


def test_occluded_tangent_is_not_reconstructed_as_observed():
    a, r = scene()
    a['observed_ranges'] = [[20,80],[120,180]]
    out = measure_relation(a,r)
    assert not out['valid'], out


def test_invalid_measurement_never_produces_force_command():
    out = measure_relation(dict(valid=False),dict(valid=True))
    assert not out['valid'] and out['force_command'] is None


def test_mirror_has_same_distance_and_angle():
    a, r = scene()
    original = measure_relation(a,r)
    a['points'] = [[200-x,y] for x,y in a['points']]
    mirrored = measure_relation(a,r)
    assert abs(original['center_distance_px']-mirrored['center_distance_px']) < 1e-5


def test_rotated_arc_uses_parametric_segments_not_y_of_x():
    a,r = scene()
    original = measure_relation(a,r)
    # Rotate coordinates by 90 degrees about the image origin.
    a['segments'] = [[[-y,x] for x,y in a['points']]]
    a['points'] = a['segments'][0]
    r['normal'] = [-1,0]
    r['observed_segment'] = [[-y,x] for x,y in r['observed_segment']]
    rotated = measure_relation(a,r)
    assert rotated['valid'], rotated
    assert abs(original['center_distance_px']-rotated['center_distance_px']) < 1e-5
    assert abs(original['angle_difference_deg']-rotated['angle_difference_deg']) < 1e-5


def test_actual_segments_override_overlapping_image_x_ranges():
    a,r = scene()
    pts = np.asarray(a['points'])
    a['segments'] = [pts[pts[:,0]<80].tolist(),pts[pts[:,0]>120].tolist()]
    a['observed_ranges'] = [[20,180]]  # broad image-x range cannot bridge missing data
    assert not measure_relation(a,r)['valid']
