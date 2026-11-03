import cv2
import numpy as np
import pytest

from vision_geometry_lab.v18_front_observability import build_fixed_baseline, sample_normal_edges
from vision_geometry_lab.v18_unit_profile import (
    GRADIENT_UNITS, SEMANTIC_GRADIENT_UNITS, THRESHOLD,
    semantic_from_seeds, sample_unit_normal,
)


IDENTITY = np.array([[1., 0., 0.], [0., 1., 0.]])
ROWS = np.arange(60., 141., 10.)


def baseline():
    return build_fixed_baseline(np.arange(26)/5., ROWS, np.full((26, len(ROWS)), 140.))


def edge(x=140., contrast=180.):
    xs = np.arange(300, dtype=float)
    p = 25.+contrast/(1.+np.exp(np.clip(-(xs-x), -80., 80.)))
    return np.tile(p, (220, 1)).astype(np.float32)


@pytest.mark.parametrize('profile', [
    np.arange(131, dtype=np.float32),
    (np.arange(131) >= 65).astype(np.float32)*10.,
    (np.arange(131) >= 65).astype(np.float32)*-10.+90.,
])
def test_sobel_gain_is_eight_for_vertical_constant_ramp_and_step(profile):
    image = np.tile(profile, (9, 1))
    sobel = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)[4, 1:-1]
    derivative = np.gradient(profile)[1:-1]
    np.testing.assert_allclose(sobel, 8.*derivative, atol=1e-6)


@pytest.mark.parametrize('amplitude,accepted', [(1.5, False), (2., True), (3., True)])
def test_unit_conversion_rejects_and_accepts_like_sobel_8(amplitude, accepted):
    p = 25.+(np.arange(300)>=140).astype(np.float32)*amplitude
    image = np.tile(p, (220, 1))
    old_max = float(np.abs(cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)[100, 75:206]).max())
    out = semantic_from_seeds(image, np.full(len(ROWS), 140.), ROWS)
    assert bool(old_max >= 8.) is accepted
    assert bool(np.isfinite(out['x']).all()) is accepted
    normal = sample_unit_normal(image, IDENTITY, ROWS, baseline())
    assert bool(normal['supported'].all()) is accepted


def test_units_are_explicit_and_not_a_confidence_claim():
    assert THRESHOLD == 1.
    assert GRADIENT_UNITS == 'gray_levels_per_reference_pixel'
    assert SEMANTIC_GRADIENT_UNITS == 'gray_levels_per_current_pixel'
    r = sample_unit_normal(edge(), IDENTITY, ROWS, baseline())
    assert not r['new_layer_confirmed'] and not r['boundary_identity_confirmed']


def test_constant_image_stays_unknown():
    image = np.full((220, 300), 92., np.float32)
    a = semantic_from_seeds(image, np.full(len(ROWS), 140.), ROWS)
    b = sample_unit_normal(image, IDENTITY, ROWS, baseline())
    assert np.isnan(a['x']).all() and np.isnan(b['offset']).all()
    assert not b['supported'].any()


def test_winning_peak_itself_must_reach_threshold():
    # Near-reference derivative 0.75 wins its Gaussian-weighted competition,
    # while the far derivative 1.1 passes the unweighted global maximum gate.
    p = 25.+(np.arange(300)>=140)*1.5+(np.arange(300)>=195)*2.2
    image = np.tile(p, (220, 1)).astype(np.float32)
    a = semantic_from_seeds(image, np.full(len(ROWS), 140.), ROWS)
    b = sample_unit_normal(image, IDENTITY, ROWS, baseline())
    assert np.isnan(a['x']).all()
    assert np.isnan(b['offset']).all()
    assert (b['reason'] == 'selected_gradient_below_unit_threshold').all()


def test_unit_normal_outward_positive_eight_and_affine_translation():
    a = sample_unit_normal(edge(x=132., contrast=12.), IDENTITY, ROWS, baseline())
    affine = np.array([[1., 0., 13.], [0., 1., 9.]])
    b = sample_unit_normal(edge(x=145., contrast=12.), affine, ROWS, baseline())
    assert a['supported'].all() and b['supported'].all()
    np.testing.assert_allclose(a['offset'], 8., atol=1.)
    np.testing.assert_allclose(b['offset'], 8., atol=1.)
    np.testing.assert_array_equal(a['offset'], b['offset'])


def test_high_contrast_matches_frozen_threshold_8_sampling():
    a = sample_normal_edges(edge(x=132.), IDENTITY, ROWS, baseline())
    b = sample_unit_normal(edge(x=132.), IDENTITY, ROWS, baseline())
    assert a.keys() == b.keys()
    for key in ('offset', 'gradient_signed', 'peak_ratio', 'contrast', 'clipped_fraction',
                'profile', 'offsets_px', 'supported'):
        np.testing.assert_array_equal(a[key], b[key])


def test_old_missing_offset_is_not_preserved_when_profile_supports_unit_peak():
    image = edge(x=132., contrast=12.)
    a = sample_normal_edges(image, IDENTITY, ROWS, baseline())
    b = sample_unit_normal(image, IDENTITY, ROWS, baseline())
    assert np.isnan(a['offset']).all() and not a['supported'].any()
    assert b['supported'].all() and np.isfinite(b['offset']).all()


def test_out_of_bounds_remains_unknown_without_filling_profile():
    b = baseline()
    b['x_ref'][:] = 50.
    r = sample_unit_normal(edge(x=50.), IDENTITY, ROWS, b)
    assert np.isnan(r['profile']).all() and not r['supported'].any()
    assert (r['reason'] == 'sampling_out_of_bounds').all()


def test_archived_nan_seeds_are_not_invented():
    seeds = np.full(len(ROWS), 140.)
    seeds[3] = np.nan
    r = semantic_from_seeds(edge(), seeds, ROWS)
    assert set(r) == {'x', 'seed', 'gradient', 'contrast'}
    assert np.isnan(r['x'][3]) and np.isnan(r['seed'][3])
    assert np.isfinite(r['x'][np.arange(len(ROWS)) != 3]).all()


def test_nonfinite_pixels_do_not_supply_evidence():
    image = edge()
    image[:, 140] = np.nan
    a = semantic_from_seeds(image, np.full(len(ROWS), 140.), ROWS)
    b = sample_unit_normal(image, IDENTITY, ROWS, baseline())
    assert np.isnan(a['x']).all() and np.isnan(b['offset']).all()
