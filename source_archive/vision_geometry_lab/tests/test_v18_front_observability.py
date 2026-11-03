import numpy as np
import pytest

from vision_geometry_lab.v18_front_observability import (
    build_fixed_baseline, sample_normal_edges,
)


IDENTITY = np.array([[1., 0., 0.], [0., 1., 0.]])


def baseline(rows=None, x=140.):
    rows = np.arange(60., 141., 10.) if rows is None else np.asarray(rows, float)
    times = np.arange(26, dtype=float) / 5.
    values = np.broadcast_to(np.asarray(x, float), (len(times), len(rows))).copy()
    return rows, build_fixed_baseline(times, rows, values)


def image_edge(edge_x=140., bias=0., slope=0., height=220, width=300):
    yy, xx = np.mgrid[:height, :width]
    # Smooth, unsaturated boundary with dark exterior on its left.
    image = 25. + 180. / (1. + np.exp(np.clip(-(xx-edge_x-slope*yy), -80., 80.)))
    return (image+bias).astype(np.float32)


def test_same_frame_observation_is_near_zero():
    rows, b = baseline()
    r = sample_normal_edges(image_edge(), IDENTITY, rows, b)
    assert r['supported'].all()
    assert np.max(np.abs(r['offset'])) <= 1.
    assert (r['gradient_signed'] < 0.).all()
    assert r['profile'].shape == (len(rows), 131)
    assert not r['boundary_identity_confirmed'] and not r['new_layer_confirmed']


def test_additive_brightness_does_not_move_peak():
    rows, b = baseline()
    a = sample_normal_edges(image_edge(), IDENTITY, rows, b)
    r = sample_normal_edges(image_edge(bias=20.), IDENTITY, rows, b)
    np.testing.assert_array_equal(a['offset'], r['offset'])
    np.testing.assert_allclose(a['gradient_signed'], r['gradient_signed'], atol=2e-5)
    np.testing.assert_allclose(a['contrast'], r['contrast'], atol=2e-5)


def test_black_image_has_no_evidence_even_with_saturation_diagnostic():
    rows, b = baseline()
    r = sample_normal_edges(np.zeros((220, 300), np.uint8), IDENTITY, rows, b)
    assert not r['supported'].any()
    assert np.isnan(r['offset']).all()
    assert (r['reason'] == 'gradient_below_threshold').all()
    assert (r['clipped_fraction'] == 1.).all()


def test_future_and_invalid_timestamps_cannot_change_baseline():
    times = np.r_[np.arange(26)/5., 5.2, 30., np.nan, np.inf]
    rows = np.arange(7.)*10.
    x = np.full((len(times), len(rows)), 100.)
    x[26:] = 9000.
    b = build_fixed_baseline(times, rows, x)
    assert b['baseline_frame_count'] == 26
    np.testing.assert_array_equal(b['x_ref'], 100.)
    np.testing.assert_array_equal(b['baseline_mad_px'], 0.)
    assert np.max(b['baseline_times_s']) <= 5.


def test_minimum_global_and_per_location_baseline_support():
    rows = np.arange(7.)*10.
    b = build_fixed_baseline(np.arange(19)/5., rows, np.ones((19, 7))*100.)
    assert not b['baseline_ready'] and not b['supported'].any()
    x = np.ones((26, 7))*100.
    x[:7, 3] = np.nan
    b = build_fixed_baseline(np.arange(26)/5., rows, x)
    assert b['support_count'][3] == 19
    assert not b['location_supported'][3] and np.isnan(b['x_ref'][3])


def test_normal_fit_does_not_cross_nan_gap():
    rows = np.arange(9.)*10.
    x = np.r_[100.+.2*rows[:4], np.nan, 800.+3.*rows[5:]]
    _, b = baseline(rows, x)
    assert not b['supported'][4]
    np.testing.assert_allclose(b['ny'][:4]/-b['nx'][:4], .2)
    np.testing.assert_allclose(b['ny'][5:]/-b['nx'][5:], 3.)


def test_short_isolated_runs_leave_normal_unknown():
    rows = np.arange(5.)*10.
    _, b = baseline(rows, [100., 100., np.nan, 100., 100.])
    assert not b['supported'].any()
    assert np.isfinite(b['x_ref'][[0, 1, 3, 4]]).all()


def test_affine_camera_translation_is_compensated():
    rows, b = baseline()
    affine = np.array([[1., 0., 13.], [0., 1., 9.]])
    r = sample_normal_edges(image_edge(edge_x=153.), affine, rows, b)
    assert r['supported'].all()
    assert np.max(np.abs(r['offset'])) <= 1.


def test_physical_outward_image_expansion_is_positive_eight():
    rows, b = baseline()
    r = sample_normal_edges(image_edge(edge_x=132.), IDENTITY, rows, b)
    assert r['supported'].all()
    np.testing.assert_allclose(r['offset'], 8., atol=1.)


def test_affine_translation_and_expansion_are_not_conflated():
    rows, b = baseline()
    affine = np.array([[1., 0., 13.], [0., 1., 9.]])
    r = sample_normal_edges(image_edge(edge_x=145.), affine, rows, b)
    assert r['supported'].all()
    np.testing.assert_allclose(r['offset'], 8., atol=1.)


def test_slanted_contour_uses_unit_outward_normal():
    rows = np.arange(60., 141., 10.)
    _, b = baseline(rows, 100.+.3*rows)
    np.testing.assert_allclose(b['nx'], -1./np.hypot(1., .3))
    np.testing.assert_allclose(b['ny'], .3/np.hypot(1., .3))
    r = sample_normal_edges(image_edge(edge_x=100., slope=.3), IDENTITY, rows, b)
    assert r['supported'].all()
    np.testing.assert_allclose(r['offset'], 0., atol=1.)


def test_entire_strip_must_be_in_bounds():
    rows, b = baseline(x=50.)
    r = sample_normal_edges(image_edge(edge_x=50.), IDENTITY, rows, b)
    assert not r['supported'].any()
    assert np.isnan(r['profile']).all()
    assert (r['reason'] == 'sampling_out_of_bounds').all()


def test_nonfinite_image_samples_are_unknown():
    rows, b = baseline()
    image = image_edge()
    image[:, 140] = np.nan
    r = sample_normal_edges(image, IDENTITY, rows, b)
    assert not r['supported'].any()
    assert (r['reason'] == 'nonfinite_image_samples').all()


def test_second_edge_reduces_peak_ratio_without_identity_claim():
    rows, b = baseline()
    a = sample_normal_edges(image_edge(), IDENTITY, rows, b)
    x = np.arange(300, dtype=float)
    p = 25.+100./(1.+np.exp(np.clip(-(x-140.), -80., 80.)))
    p -= 90./(1.+np.exp(np.clip(-(x-160.), -80., 80.)))
    r = sample_normal_edges(np.tile(p, (220, 1)).astype(np.float32), IDENTITY, rows, b)
    assert r['supported'].all()
    assert (r['peak_ratio'] < a['peak_ratio']).all()
    assert not r['boundary_identity_confirmed']


def test_bad_shapes_coordinates_and_affine_are_rejected():
    with pytest.raises(ValueError):
        build_fixed_baseline([0.], [2., 1.], [[0., 0.]])
    rows, b = baseline()
    with pytest.raises(ValueError):
        sample_normal_edges(image_edge(), np.zeros((2, 3)), rows, b)
    with pytest.raises(ValueError):
        sample_normal_edges(image_edge(), IDENTITY, rows+1., b)
