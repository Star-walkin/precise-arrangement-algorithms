import numpy as np
import pytest

from vision_geometry_lab.v21_box_audit import _inside, _save_png, image_read


def test_inclusive_box_and_nonfinite_points():
    points = np.array([[2, 3], [5, 7], [1, 4], [3, 8], [np.nan, 4], [3, np.inf]])
    assert _inside(points, [2, 3, 5, 7]).tolist() == [True, True, False, False, False, False]


def test_strip_array_retains_row_axes():
    points = np.zeros((2, 7, 131, 2))
    points[1, 3, 65] = [10, 10]
    touches = _inside(points, [9, 9, 11, 11]).any(axis=(1, 2))
    assert touches.tolist() == [False, True]


def test_png_exact_pixels_and_no_overwrite(tmp_path):
    path = tmp_path / '核查.png'
    image = np.arange(90, dtype=np.uint8).reshape(5, 6, 3)
    _save_png(path, image)
    np.testing.assert_array_equal(image_read(path), image)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        _save_png(path, np.zeros_like(image))
    assert path.read_bytes() == original
