import unittest
import numpy as np

from vision_geometry_lab.v16_video_measurement import automatic_seed, path_gap_features, selected_frames


class VideoMeasurementTests(unittest.TestCase):
    def geometry(self):
        return dict(width_px=10., origin_xy=[0., 0.], d_xy=[1., 0.])

    def line(self, y, x0=0, x1=101):
        return np.c_[np.arange(x0, x1, 2.), np.full(len(np.arange(x0, x1, 2.)), y)].tolist()

    def arc(self, x0=0, x1=101):
        # Offset sampling avoids the old helper's deliberately strict interior
        # foot test rejecting every exactly coincident polyline vertex.
        return self.line(0, x0-.5, x1+.5)

    def test_sampling_complete_0_to_60_5hz(self):
        result = selected_frames(601, 10.)
        self.assertEqual(len(result), 301)
        self.assertEqual((result[0], result[-1]), (0, 600))

    def test_auto_seed_orients_by_observed_arc(self):
        rope = dict(valid=True, width_px=10., normal=[0., 1.], observed_segment=[[100., 0.], [0., 0.]])
        arc = dict(points=[[150., 10.], [170., 10.]])
        seed, _ = automatic_seed(rope, arc)
        self.assertEqual(seed, [[0., 0.], [100., 0.]])

    def test_unresolved_width_never_defaulted(self):
        self.assertIsNone(automatic_seed(dict(valid=True, width_px=None), {})[0])

    def test_negative_proxy_preserved_not_physical_contact(self):
        result = path_gap_features(dict(observed_segments=[self.line(2)]),
            dict(local_boundary_segments=[self.arc()]), self.geometry())
        self.assertTrue(result['usable'])
        self.assertAlmostEqual(result['image_gap_proxy'], -.3)
        self.assertFalse(result['physical_contact_confirmed'])

    def test_arc_gap_cannot_be_bridged_for_span(self):
        result = path_gap_features(dict(observed_segments=[self.line(2)]),
            dict(local_boundary_segments=[self.arc(0,21),self.arc(80,101)]), self.geometry())
        self.assertFalse(result['usable'])
        self.assertLess(result['contiguous_supported_span_px'], 30)

    def test_crossing_not_parallel_is_unusable(self):
        points=np.c_[np.arange(0,101,2.),np.arange(0,101,2.)-50].tolist()
        result=path_gap_features(dict(observed_segments=[points]),dict(local_boundary_segments=[self.arc()]),self.geometry())
        self.assertFalse(result['usable'])
        self.assertGreater(result['median_tangent_angle_deg'],15)

    def test_missing_boundary_does_not_become_zero(self):
        result=path_gap_features(dict(observed_segments=[self.line(2)]),dict(local_boundary_segments=[]),self.geometry())
        self.assertFalse(result['usable']); self.assertIsNone(result['image_gap_proxy'])

    def test_separate_model_segments_do_not_meet_span_gate(self):
        result=path_gap_features(dict(observed_segments=[self.line(2,0,21),self.line(2,80,101)]),
            dict(local_boundary_segments=[self.arc()]),self.geometry())
        self.assertFalse(result['usable']);self.assertLess(result['contiguous_supported_span_px'],30)


if __name__=='__main__':
    unittest.main()
