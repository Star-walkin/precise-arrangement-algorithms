"""Mechanical checks for the explicitly GT-guided, non-predictive diagnostic."""
import numpy as np

from vision_geometry_lab.context300_gt_response_posthoc import (
    query_support, reference_samples, sample_maps,
)


def test_reference_cells_preserve_length_and_do_not_join_segments():
    points, normals, weights = reference_samples([
        [[0, 0], [30, 0], [30, 40]],
        [[100, 0], [130, 0]],
    ])
    assert np.isclose(weights.sum(), 100.0)
    assert np.all((weights > 0) & (weights <= 4.0))
    assert not np.any((points[:, 0] > 30) & (points[:, 0] < 100))
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)


def test_native_map_sampling_uses_full_image_roi_origin():
    plane = np.arange(48, dtype=np.float32).reshape(6, 8) / 48.0
    body, center, valid = sample_maps(
        plane, plane, np.array([[101, 202], [106, 203]], float), [100, 200, 108, 206]
    )
    assert valid.all()
    assert np.allclose(body, [plane[2, 1], plane[3, 6]])
    assert np.array_equal(body, center)


def test_unpadded_map_half_pixel_sampling_and_border_rejection():
    plane = np.arange(50, dtype=np.float32).reshape(5, 10) / 50.0
    # Native ROI 20 x 10, map 10 x 5. Native-local (4.5, 2.5) -> map (2, 1).
    body, _, valid = sample_maps(
        plane, plane, np.array([[104.5, 202.5], [100, 200]], float), [100, 200, 120, 210]
    )
    assert valid.tolist() == [True, False]
    assert np.isclose(body[0], plane[1, 2])


def test_posthoc_query_rejects_upstream_and_quality_boundaries():
    points = np.array([[-1, 1], [0, 1], [2, 1], [3, 1], [5, 1]], float)
    polygon = [[2, 0], [4, 0], [4, 2], [2, 2]]
    valid = query_support(points, [polygon], np.array([0, 1]), np.array([1, 0]))
    assert valid.tolist() == [False, True, False, False, True]
