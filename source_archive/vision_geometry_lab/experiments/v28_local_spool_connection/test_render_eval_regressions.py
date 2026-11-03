"""V28 run001 fixes: unknown outside a partial bend and missing edge vertices."""
import numpy as np
from PIL import Image

from vision_geometry_lab.experiments.v28_local_spool_connection.data_eval import _bend_coverage
from vision_geometry_lab.experiments.v28_local_spool_connection.render_results import (
    _candidate, _valid_edge_runs,
)


def test_partial_bend_has_no_false_positive_background():
    reference = [[[20., 0.], [40., 0.]]]
    short = _bend_coverage(reference, reference)
    extended = _bend_coverage(reference, [[[0., 0.], [1000., 0.]]])
    assert short["covered_fraction"] == extended["covered_fraction"] == 1.
    assert short["reference_length_px"] == extended["reference_length_px"] == 20.
    for row in (short, extended):
        assert row["prediction_precision_at_5px"] is None
        assert row["extra_predicted_length_px"] is None
        assert row["extra_predicted_fraction"] is None
        assert row["precision_score_available"] is False


def test_missing_boundary_points_split_measured_edges():
    points = [[0., 0.], [1., 0.], None, [3., 0.], [4., 0.], None]
    assert _valid_edge_runs(points) == [(0, 2), (3, 5)]
    assert _valid_edge_runs([None, [1., 0.], None]) == []


def test_render_missing_edges_without_bridging():
    canvas = Image.new("RGB", (100, 100), "black")
    candidate = dict(points_xy=[[10., 60.], [20., 60.], [30., 60.], [40., 60.], [50., 60.]],
                     left_xy=[[10., 20.], [20., 20.], None, [40., 20.], [50., 20.]],
                     right_xy=[None]*5, supported=[True]*5, unavailable=[False]*5)
    _candidate(canvas, lambda p: np.asarray(p), candidate, "white", boundary=True)
    pixels = np.asarray(canvas)
    assert pixels[20, 15].sum() > 0
    assert pixels[20, 30].sum() == 0
    assert pixels[20, 45].sum() > 0
