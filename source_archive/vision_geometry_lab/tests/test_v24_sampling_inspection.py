import numpy as np
import pytest

from vision_geometry_lab.v11_path_contract import build_input
from vision_geometry_lab.v24_sampling_inspection import (
    compare_same_region, inspect_sampling, sampling_record, summarize, viewport_bounds)


def identity_grid_fixture():
    y, x = np.indices((120, 180))
    rgb = np.stack([x, y, (x+y)//2], axis=-1).astype(np.uint8)
    # Width chosen for one native pixel per strip step. Subpixel phase also
    # checks that the inverse mapping does not accidentally add another .5.
    built = build_input(rgb, [[32+1/6, 32+1/6], [42+1/6, 32+1/6]], 1/.1875)
    return rgb, built


def test_same_region_preserves_original_crop_and_native_coordinate_grid():
    rgb, built = identity_grid_fixture()
    bounds = [40, 24, 72, 56]
    native, reconstructed, support, diagnostic = compare_same_region(
        rgb, built['input'][:3], built['valid_mask'], built['geometry'], bounds)
    np.testing.assert_array_equal(native, rgb[24:56, 40:72])
    assert diagnostic['both_panels_same_xy_grid']
    assert diagnostic['native_xy_strip_roundtrip_max_error_px'] < 1e-8
    assert support.all()
    # Affine RGB ramp is preserved within uint8 rounding despite subpixel phase.
    np.testing.assert_allclose(reconstructed.astype(float), native, atol=1)


def test_invalid_cache_support_is_explicit_and_never_clips_native_region():
    rgb, built = identity_grid_fixture()
    native, reconstructed, support, diagnostic = compare_same_region(
        rgb, built['input'][:3], np.zeros_like(built['valid_mask']), built['geometry'], [40, 24, 72, 56])
    np.testing.assert_array_equal(native, rgb[24:56, 40:72])
    assert not support.any() and diagnostic['supported_fraction'] == 0
    assert (reconstructed == 90).all()


def test_cache_from_different_source_shape_is_rejected():
    rgb, built = identity_grid_fixture()
    built['geometry']['source_shape_hw'] = [120, 181]
    with pytest.raises(ValueError, match='source shape'):
        compare_same_region(rgb, built['input'][:3], built['valid_mask'], built['geometry'], [40, 24, 72, 56])


def test_width_ratio_fixed_unavailable_measurements_stay_unknown():
    _, built = identity_grid_fixture()
    entry = dict(sample_id='sample', split='holdout', variant='auto', available=True, reason='ok', geometry=built['geometry'])
    measured = sampling_record('sample::auto', entry)
    assert measured['optical_width_input_pixels'] == pytest.approx(5+1/3)
    entry.update(available=False, reason='missing', geometry=None)
    missing = sampling_record('other::auto', entry)
    assert missing['native_px_per_row'] is None
    summary = next(s for s in summarize([measured, missing]) if s['variant'] == 'auto' and s['split'] == 'holdout')
    assert summary['total'] == 2 and summary['available'] == 1
    assert summary['native_px_per_row']['count'] == 1


def test_bounds_clip_to_source_without_changing_window_size():
    assert viewport_bounds([1.4, 2.5], [100, 160], 32) == [0, 0, 32, 32]
    assert viewport_bounds([159, 99], [100, 160], 32) == [128, 68, 160, 100]
    with pytest.raises(ValueError):
        viewport_bounds([1, 2], [10, 10], 32)


def test_output_refuses_overwrite_before_loading_frozen_sources(tmp_path):
    (tmp_path/'sampling_inspection').mkdir()
    with pytest.raises(FileExistsError):
        inspect_sampling(tmp_path)
