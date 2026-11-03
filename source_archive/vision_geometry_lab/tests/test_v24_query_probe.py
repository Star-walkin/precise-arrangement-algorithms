import inspect

import numpy as np
import pytest

from vision_geometry_lab import v24_query_probe as m


def fixture():
    rng = np.random.default_rng(4)
    a = rng.random((8, 256, 512), dtype=np.float32)
    a[7] = 1
    a[4] = 0
    for col in (10, 20, 70, 300): a[4, 100:140, col] = .8
    return a


def test_three_interventions_only_change_query_and_do_not_expand_support():
    x = fixture(); before = x.copy(); variants = m.interventions(x)
    for name, (meta, a) in variants.items():
        assert meta['available']
        np.testing.assert_array_equal(a[[0, 1, 2, 3, 5, 6, 7]], x[[0, 1, 2, 3, 5, 6, 7]])
        assert np.all(a[4] <= x[4])
    assert not variants['zero_query'][1][4].any()
    assert variants['query_first_half'][0]['retained_effective_columns'] == [10, 20]
    assert variants['query_last_half'][0]['retained_effective_columns'] == [70, 300]
    np.testing.assert_array_equal(x, before)


def test_effective_column_count_split_not_span_midpoint():
    x = fixture(); x[4] = 0
    for col in (20, 21, 22, 480, 490): x[4, 128, col] = 1
    variants = m.interventions(x)
    assert variants['query_first_half'][0]['retained_effective_columns'] == [20, 21]
    assert variants['query_last_half'][0]['retained_effective_columns'] == [22, 480, 490]


def test_invalid_pixels_cannot_establish_query_support():
    x = fixture(); x[7] = 0
    assert all(a is None and not meta['available'] for meta, a in m.interventions(x).values())


def test_low_amplitude_tail_is_not_effective_support():
    x = fixture(); x[4] = .049
    assert all(a is None for meta, a in m.interventions(x).values())


def test_one_effective_column_zero_probe_allowed_halves_unknown():
    x = fixture(); x[4] = 0; x[4, 100, 50] = .5
    out = m.interventions(x)
    assert out['zero_query'][0]['available']
    assert out['query_first_half'][1] is None
    assert out['query_last_half'][1] is None


@pytest.mark.parametrize('corruption', ('nan', 'shape', 'negative_query', 'aboveone'))
def test_invalid_original_input_rejected(corruption):
    x = fixture()
    if corruption == 'nan': x[0, 0, 0] = np.nan
    elif corruption == 'shape': x = x[:7]
    elif corruption == 'negative_query': x[4, 0, 0] = -.2
    else: x[4, 0, 0] = 1.2
    with pytest.raises(ValueError): m.interventions(x)


def test_probability_difference_keeps_identical_geometry_native_scale():
    a = np.zeros((8, 10)); a[2] = 1
    b = np.zeros_like(a); b[5] = 1
    result = m.distribution_change(a, b, np.ones_like(a, bool), 4.)
    assert result['mean_total_variation_per_valid_column'] == 1.
    assert result['peak_row_shift_mean_native_px'] == 12.
    assert result['target_presence_or_identity_assessed'] is False
    same = m.distribution_change(a, a, np.ones_like(a, bool), 4.)
    assert same['mean_total_variation_per_valid_column'] == 0
    assert same['peak_row_shift_mean_native_px'] == 0


def test_invalid_image_columns_excluded_not_counted_as_unchanged():
    a = np.zeros((8, 10)); a[2] = 1
    b = a.copy(); b[:, 0] = 0; b[5, 0] = 1
    valid = np.zeros_like(a, bool); valid[:, 0] = True
    result = m.distribution_change(a, b, valid, 1.)
    assert result['valid_columns'] == 1
    assert result['mean_total_variation_per_valid_column'] == 1


def test_missing_visibility_cannot_appear_as_identity_accuracy():
    result = m.distribution_change(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2), bool), 1.)
    assert result == dict(available=False, reason='no_valid_image_columns')


def test_path_length_does_not_bridge_gaps():
    assert m.path_length({'segments': [[[0, 0], [3, 4]], [[50, 50], [50, 62]]]}) == 17


def test_infer_uses_no_target_reader_and_does_not_rebuild_crop():
    source = inspect.getsource(m.infer)
    for forbidden in ('build_input(', 'build_targets(', '_dataset(', 'load_train_target(', 'load_label_quality('):
        assert forbidden not in source


def test_freeze_will_not_overwrite_existing_probe(tmp_path):
    with pytest.raises(FileExistsError): m.freeze(tmp_path/'missing_parent', tmp_path)
