import copy
import inspect

import cv2
import numpy as np
import pytest

from vision_geometry_lab import v25_local_contract as contract


def pair(rgb=None, seed=None, width=16.):
    if rgb is None:
        y, x = np.indices((240, 480))
        rgb = np.stack([(17*x+3*y)%256, (7*x+13*y)%256, (11*x+5*y)%256], axis=-1).astype(np.uint8)
    return contract.build_input_pair(rgb, seed or [[100, 120], [164, 120]], width)


def probability(rows, sigma=2.):
    values = np.exp(-.5*((np.arange(contract.HEIGHT)[:, None]-np.asarray(rows)[None])/sigma)**2)
    return (values/values.sum(axis=0, keepdims=True)).astype(np.float32)


def straight_probability():
    return probability(np.full(contract.WIDTH, 47.5))


def output_local(decoded, geometry):
    return [contract.image_to_local(np.asarray(s), geometry) for s in decoded['segments']]


def test_pair_shape_dtype_same_nonimage_channels_and_source_not_changed():
    rgb = np.random.default_rng(7).integers(0, 256, (240, 480, 3), dtype=np.uint8)
    old = rgb.copy()
    built = pair(rgb)
    assert built['high'].shape == built['low'].shape == (8, 96, 192)
    assert built['high'].dtype == built['low'].dtype == np.float32
    assert built['valid_mask'].dtype == bool and built['valid_mask'].all()
    assert np.isfinite(built['high']).all() and np.isfinite(built['low']).all()
    np.testing.assert_array_equal(built['high'][4:], built['low'][4:])
    np.testing.assert_array_equal(built['high'][7], built['valid_mask'])
    np.testing.assert_array_equal(rgb, old)


def test_low_rgb_is_exact_three_by_three_area_then_bilinear_and_gray_recomputed():
    built = pair()
    high = built['high'][:3].transpose(1, 2, 0)
    coarse = high.reshape(32, 3, 64, 3, 3).mean(axis=(1, 3))
    expected = cv2.resize(coarse, (192, 96), interpolation=cv2.INTER_LINEAR)
    np.testing.assert_allclose(built['low'][:3].transpose(1, 2, 0), expected, atol=2e-7)
    gray = contract._normalized_gray(built['low'][:3].transpose(1, 2, 0), built['valid_mask'])
    np.testing.assert_allclose(built['low'][3], gray.astype(np.float32), atol=1e-7)
    assert np.max(np.abs(built['high'][3]-built['low'][3])) > .01
    assert built['geometry']['low_resolution']['input_width_pixels'] == pytest.approx(16/3)


def test_constant_image_padding_does_not_become_normalized_gray_evidence():
    rgb = np.full((80, 120, 3), 128, np.uint8)
    built = pair(rgb, [[2, 8], [20, 8]], 16.)
    assert built['valid_mask'].any() and not built['valid_mask'].all()
    for arm in ('high', 'low'):
        assert np.max(np.abs(built[arm][3])) < 1e-5
        assert np.all(built[arm][:5, ~built['valid_mask']] == 0)
    np.testing.assert_array_equal(built['high'][4:], built['low'][4:])


@pytest.mark.parametrize('seed', [[[100, 120], [164, 120]], [[220, 100], [180, 130]], [[180, 180], [180, 116]]])
def test_coordinate_roundtrip_and_physical_bounds(seed):
    geometry = pair(seed=seed)['geometry']
    points = np.array([[-.5, -.5], [191.5, 95.5], [35.7, 52.2], [63.5, 47.5]])
    xy = contract.local_to_image(points, geometry)
    np.testing.assert_allclose(contract.image_to_local(xy, geometry), points, atol=1e-10)
    np.testing.assert_allclose(xy[-1], seed[-1], atol=1e-10)
    assert geometry['step_widths'] == 1/16


def test_coordinate_channels_use_new_local_extents():
    built = pair()
    expected_u = (-4+(np.arange(192)+.5)/16)/8
    expected_v = (-3+(np.arange(96)+.5)/16)/3
    np.testing.assert_allclose(built['high'][5, 0], expected_u, atol=1e-8)
    np.testing.assert_allclose(built['high'][6, :, 0], expected_v, atol=1e-8)


def test_targets_truncated_tail_is_unknown_and_does_not_modify_input():
    built = pair(); before = copy.deepcopy(built)
    result = contract.targets([[100, 120], [196, 120]], [], built['geometry'], built['valid_mask'])
    assert set(contract.TARGET_KEYS).issubset(result)
    assert result['target_distribution'].shape == result['row_valid'].shape == (96, 192)
    assert result['column_valid'][80]
    assert not result['column_valid'][110:].any()
    assert not result['slope_valid'][110:].any()
    assert not result['row_valid'][:, 110:].any()
    assert not result['target_distribution'][:, 110:].any()
    np.testing.assert_allclose(result['target_distribution'][:, result['column_valid']].sum(axis=0), 1, atol=1e-6)
    assert result['diagnostics']['unknown_is_background'] is False
    assert result['diagnostics']['null_class_trained'] is False
    np.testing.assert_array_equal(built['high'], before['high'])
    np.testing.assert_array_equal(built['low'], before['low'])
    assert built['geometry'] == before['geometry']


def test_target_sigma_is_two_native_pixels_at_different_optical_widths():
    for width in (8., 16., 32.):
        built = pair(width=width)
        result = contract.targets([[100, 120], [220, 120]], [], built['geometry'], built['valid_mask'])
        assert result['diagnostics']['sigma_local_rows']*built['geometry']['native_px_per_row'] == pytest.approx(2.)


def test_quality_hole_and_multivalued_columns_receive_no_loss():
    built = pair(); geometry, valid = built['geometry'], built['valid_mask']
    exclusion = [[[184, 100], [204, 100], [204, 140], [184, 140]]]
    result = contract.targets([[100, 120], [260, 120]], exclusion, geometry, valid)
    assert result['column_valid'][70] and result['column_valid'][120]
    assert not result['column_valid'][90:100].any()
    loop = [[164, 112], [228, 112], [164, 128], [228, 128]]
    result = contract.targets(loop, [], geometry, valid)
    assert result['diagnostics']['ambiguous_columns'] > 0
    assert not result['column_valid'][80:120].any()


def test_empty_targets_all_unknown_not_null_background():
    built = pair()
    result = contract.targets([], [], built['geometry'], built['valid_mask'])
    assert all(not result[key].any() for key in contract.TARGET_KEYS)


def test_decoder_straight_outputs_only_new_segment_and_never_confirms_identity():
    built = pair()
    result = contract.decode_local(straight_probability(), np.zeros(192), built['geometry'], built['valid_mask'])
    assert result['valid'] and result['segments']
    for segment in output_local(result, built['geometry']):
        u = -4+(segment[:, 0]+.5)/16
        assert u.min() >= -1e-10 and u.max() <= 4
        assert u.min() == pytest.approx(0.)
        assert u.max() > 3.9
    assert not result['identity_confirmed'] and not result['physical_contact_confirmed']
    assert result['force_command'] is None
    assert result['diagnostics']['quality_or_targets_used'] is False
    assert result['candidates'][0]['stop_reason'] == 'fixed_four_width_horizon'


def test_decoder_follows_smooth_bend_without_straight_line_extrapolation():
    built = pair()
    u = -4+(np.arange(192)+.5)/16
    v = .08*np.maximum(u, 0)**2
    rows = 47.5+v*16
    slope = .16*np.maximum(u, 0)
    result = contract.decode_local(probability(rows), slope, built['geometry'], built['valid_mask'])
    assert result['valid']
    end = output_local(result, built['geometry'])[0][-1]
    assert -4+(end[0]+.5)/16 > 3.9
    assert -3+(end[1]+.5)/16 > 1.


def test_decoder_rejects_neighbor_only_peak_at_anchor():
    built = pair()
    neighbor = probability(np.full(192, 47.5+16))
    result = contract.decode_local(neighbor, None, built['geometry'], built['valid_mask'])
    assert not result['valid'] and not result['segments']
    assert result['reason'] == 'no_concentrated_query_anchored_distribution'


def test_decoder_does_not_cross_unknown_column_or_restart_after_gap():
    built = pair(); p = straight_probability(); p[:, 80] = 0
    result = contract.decode_local(p, None, built['geometry'], built['valid_mask'])
    assert result['valid']
    end = output_local(result, built['geometry'])[0][-1]
    assert end[0] == pytest.approx(79.)
    assert result['candidates'][0]['unobserved_columns'] == 0
    assert result['candidates'][0]['stop_reason'] == 'no_reliable_connected_next_column'
    assert not result['diagnostics']['interpolated_unobserved_gap_points_emitted']
    p[:, 60] = 0
    stopped_before_new = contract.decode_local(p, None, built['geometry'], built['valid_mask'])
    assert not stopped_before_new['segments']


def test_decoder_stops_on_immediate_neighbor_jump_after_known_prefix():
    built = pair()
    rows = np.full(192, 47.5); rows[70:] += 16
    result = contract.decode_local(probability(rows), None, built['geometry'], built['valid_mask'])
    end = output_local(result, built['geometry'])[0][-1]
    assert end[0] < 70


def test_decoder_requires_actual_query_support_not_guessed_anchor():
    built = pair(seed=[[156, 120], [164, 120]])
    result = contract.decode_local(straight_probability(), None, built['geometry'], built['valid_mask'])
    assert result['reason'] == 'query_does_not_uniquely_support_anchor'
    assert not result['segments']


def test_uniform_probability_is_not_reliable_evidence():
    built = pair()
    result = contract.decode_local(np.full((96, 192), 1/96), None, built['geometry'], built['valid_mask'])
    assert not result['segments']


def test_multiple_query_compatible_paths_are_marked_ambiguous_not_confirmed():
    built = pair()
    p = .5*probability(np.full(192, 47.5-5.8), sigma=.5)
    p += .5*probability(np.full(192, 47.5+5.8), sigma=.5)
    result = contract.decode_local(p, None, built['geometry'], built['valid_mask'])
    assert len(result['candidates']) >= 2
    assert result['diagnostics']['ambiguous']
    assert not result['identity_confirmed']


def test_input_and_decode_apis_have_no_GT_or_quality_parameter():
    assert tuple(inspect.signature(contract.build_input_pair).parameters) == ('image_rgb_uint8', 'past_seed', 'width_px')
    assert tuple(inspect.signature(contract.decode_local).parameters) == ('probability', 'slope', 'geometry', 'valid')


def test_invalid_probability_mask_geometry_are_rejected():
    built = pair(); p = straight_probability()
    with pytest.raises(ValueError, match='row distribution'):
        contract.decode_local(p*2, None, built['geometry'], built['valid_mask'])
    with pytest.raises(ValueError, match='binary'):
        contract.decode_local(p, None, built['geometry'], np.full((96, 192), .5))
    wrong = copy.deepcopy(built['geometry']); wrong['step_widths'] = .1875
    with pytest.raises(ValueError, match='sampling contract'):
        contract.image_to_local([[1, 2]], wrong)
