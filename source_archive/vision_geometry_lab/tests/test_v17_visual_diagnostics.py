import numpy as np
import pytest

from vision_geometry_lab.v17_visual_diagnostics import boundary_runs, shared_crop, validate_decoded_frame
from vision_geometry_lab.v11_path_contract import array_sha256


def test_boundary_runs_split_unknown_gaps_and_provenance_changes():
    boundary = {"points": [[0, 2], [4, 3], [8, 3], [12, 4], [24, 5]],
                "states": ["direct", "direct", "predicted", "predicted", "direct"]}
    assert [(state, len(points)) for state, points in boundary_runs(boundary)] == [
        ("direct", 2), ("predicted", 2), ("direct", 1)]


def test_boundary_bad_provenance_rejected():
    with pytest.raises(ValueError):
        boundary_runs({"points": [[0, 0]], "states": []})
    with pytest.raises(ValueError):
        boundary_runs({"points": [[0, 0]], "states": ["physical_confirmed"]})


def test_exact_source_frame_and_pixels_required():
    image = np.zeros((30, 40, 3), np.uint8)
    meta = {"frame_index": 246, "fps": 10., "time_s": 24.6, "frame_sha256": array_sha256(image)}
    validate_decoded_frame(image, 246, 246, meta, 10.)
    with pytest.raises(ValueError, match="index"):
        validate_decoded_frame(image, 245, 246, meta, 10.)
    changed = image.copy(); changed[0, 0] = 255
    with pytest.raises(ValueError, match="pixels"):
        validate_decoded_frame(changed, 246, 246, meta, 10.)


def test_display_crop_is_shared_and_clamped():
    assert shared_crop((100, 200, 3), {}, {}, {}) == (0, 0, 200, 100)
    box = shared_crop((100, 200, 3), {"points": [[10, 20], [40, 70]]}, {},
                      {"observed_segments": [[[80, 30], [120, 50]]]})
    assert box == (0, 0, 155, 100)
