"""Synthetic-only driver/coordinate audit; no videos, labels or hardware read."""
import cv2
import numpy as np
import pytest

from vision_geometry_lab.v18_front_trial import (
    align_x_to_reference, canonical_frame, semantic_candidates,
    supported_median, validate_video,
)
from vision_geometry_lab.v18_front_observability import (
    build_fixed_baseline, sample_normal_edges,
)


IDENTITY = np.array([[1., 0., 0.], [0., 1., 0.]])


class Regions:
    def __init__(self, mask, crop=None):
        self._mask = np.asarray(mask, bool)
        h, w = self._mask.shape
        self.crop_xyxy = (0, 0, w, h) if crop is None else crop

    def mask(self, name):
        assert name == 'wound_rope_surface'
        return self._mask


def regions_at(seed=200, h=240, w=400):
    return Regions(np.tile(np.arange(w) >= seed, (h, 1)))


@pytest.mark.parametrize('fps', [np.nan, np.inf, -np.inf, 0., -10.])
def test_invalid_video_fps_explicitly_rejected(fps):
    with pytest.raises(ValueError, match='invalid video fps'):
        validate_video(dict(fps=fps, frame_count=601, width=4096, height=3000))


def test_valid_video_metadata_and_fractional_shapes():
    validate_video(dict(fps=10., frame_count=601, width=4096, height=3000))
    with pytest.raises(ValueError, match='noninteger'):
        validate_video(dict(fps=10., frame_count=601.5, width=4096, height=3000))


def test_canonical_mirror_is_camera_specific():
    raw = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
    before = raw.copy()
    np.testing.assert_array_equal(canonical_frame(raw, 'left'), raw[:, ::-1])
    np.testing.assert_array_equal(canonical_frame(raw, 'right'), raw)
    np.testing.assert_array_equal(raw, before)
    with pytest.raises(ValueError, match='unknown camera'):
        canonical_frame(raw, 'guessed')


def test_empty_semantic_mask_remains_unknown_even_with_strong_image_edge():
    image = np.full((240, 400), 20., dtype=np.float32)
    image[:, 200:] = 220.
    result = semantic_candidates(image, Regions(np.zeros_like(image)), [80, 100, 120])
    for value in result.values():
        assert np.isnan(value).all()


@pytest.mark.parametrize('level', [0., 40., 128., 255.])
def test_constant_intensity_does_not_create_semantic_boundary(level):
    image = np.full((240, 400), level, dtype=np.float32)
    result = semantic_candidates(image, regions_at(), [80, 100, 120])
    np.testing.assert_array_equal(result['seed'], 200.)
    assert np.isnan(result['x']).all()
    assert np.isnan(result['gradient']).all()


def test_semantic_candidate_is_invariant_to_unsaturated_brightness_offset():
    image = np.full((240, 400), 20., dtype=np.float32)
    image[:, 209:] = 180.
    a = semantic_candidates(image, regions_at(), [80, 100, 120])
    b = semantic_candidates(image + 25., regions_at(), [80, 100, 120])
    assert np.isfinite(a['x']).all()
    assert np.max(np.abs(a['x']-209.)) <= 1.
    for key in ('x', 'gradient', 'contrast'):
        np.testing.assert_array_equal(a[key], b[key])


def test_semantic_crop_coordinate_conversion():
    mask = np.tile(np.arange(100) >= 30, (80, 1))
    region = Regions(mask, (100, 40, 300, 200))
    image = np.full((240, 400), 20., dtype=np.float32)
    image[:, 160:] = 180.
    result = semantic_candidates(image, region, [80, 100, 120])
    np.testing.assert_array_equal(result['seed'], 160.)
    assert np.max(np.abs(result['x']-160.)) <= 1.


def test_far_strong_edge_cannot_validate_selected_weak_semantic_peak():
    image = np.full((240, 400), 20., dtype=np.float32)
    image[:, 200:] += 14.  # Chosen near seed, gradient 7 < required 8.
    image[:, 263:] += 160.  # Window maximum 80 does not support the near peak.
    result = semantic_candidates(image, regions_at(), [80, 100, 120])
    np.testing.assert_array_equal(result['seed'], 200.)
    assert np.isnan(result['x']).all()
    assert np.isnan(result['gradient']).all()


def test_alignment_never_interpolates_across_unknown_input_row():
    ys = np.array([20., 28., 36., 44., 52.])
    xs = np.array([120., 122., np.nan, 126., 128.])
    grid = np.arange(20., 53.)
    aligned = align_x_to_reference(xs, ys, IDENTITY, grid)
    assert np.isnan(aligned[(grid > 28.) & (grid < 44.)]).all()
    assert np.isfinite(aligned[(grid <= 28.) | (grid >= 44.)]).all()


def test_alignment_never_interpolates_excessively_wide_row_gap():
    grid = np.arange(20., 54.)
    aligned = align_x_to_reference([120., 124.], [20., 53.], IDENTITY, grid)
    assert np.isnan(aligned).all()


def test_alignment_translation_uses_inverse_not_forward_transform():
    rows = np.arange(40., 121., 8.)
    reference_x = 140. + .2*rows
    transform = np.array([[1., 0., 17.], [0., 1., -9.]])
    result = align_x_to_reference(reference_x+17., rows-9., transform, rows)
    np.testing.assert_allclose(result, reference_x, atol=1e-10)


def test_alignment_rotation_regrids_y_as_well_as_x():
    rows = np.arange(40., 161., 4.)
    reference = np.column_stack([140.+.15*rows, rows])
    transform = cv2.getRotationMatrix2D((160., 110.), .6, 1.003)
    transform[:, 2] += [3., -2.]
    current = reference@transform[:, :2].T+transform[:, 2]
    grid = np.arange(42., 159., 4.)
    result = align_x_to_reference(current[:, 0], current[:, 1], transform, grid)
    np.testing.assert_allclose(result, 140.+.15*grid, atol=1e-9)


@pytest.mark.parametrize('matrix', [np.full((2, 3), np.nan), np.zeros((3, 3))])
def test_bad_alignment_matrix_cannot_create_observation(matrix):
    result = align_x_to_reference([100., 102.], [20., 28.], matrix, np.arange(20., 29.))
    assert np.isnan(result).all()


def test_normal_profiles_compensate_affine_rotation_and_translation():
    height, width = 280, 400
    yy, xx = np.mgrid[:height, :width]
    reference_image = (30.+180./(1.+np.exp(np.clip(-(xx-160.-.2*yy), -80., 80.)))).astype(np.float32)
    rows = np.arange(80., 161., 8.)
    times = np.arange(26)/5.
    baseline = build_fixed_baseline(times, rows, np.tile(160.+.2*rows, (26, 1)))
    identity = sample_normal_edges(reference_image, IDENTITY, rows, baseline)
    transform = cv2.getRotationMatrix2D((200., 140.), .6, 1.003)
    transform[:, 2] += [3., -2.]
    current_image = cv2.warpAffine(reference_image, transform, (width, height), flags=cv2.INTER_LINEAR)
    result = sample_normal_edges(current_image, transform, rows, baseline)
    assert result['supported'].all()
    np.testing.assert_allclose(result['offset'], identity['offset'], atol=1.)
    assert not result['boundary_identity_confirmed']
    assert not result['new_layer_confirmed']


def test_supported_median_does_not_make_unknown_row_finite():
    values = np.full((25, 3), 5.)
    values[:6, 1] = np.nan
    values[:, 2] = np.nan
    result = supported_median(values)
    assert result[0] == 5.
    assert np.isnan(result[1:]).all()

