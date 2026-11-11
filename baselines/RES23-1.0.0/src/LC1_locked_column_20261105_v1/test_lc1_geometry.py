"""Synthetic contracts, not validation of real-video accuracy.

Run: python -m unittest -v test_lc1_geometry.py
"""
import unittest
import numpy as np

from lc1_geometry import ColumnLock, count_crossings, estimate_turn, fit_line, normalize_line


def line_at(d=100., y=200., angle=-12.):
    phi = np.radians(angle)
    a, b = -np.sin(phi), np.cos(phi)
    return np.array([a, b, -a * d - b * y])


def good_turn(x=100.):
    return dict(valid=True, x=x, y=200., quality=.95, rmse=.3,
                uncertainty_px=.5, reason="synthetic")


def points_on_line(abc, xs=None):
    xs = np.linspace(2000., 3000., 11) if xs is None else np.asarray(xs, float)
    abc = np.asarray(abc, float)
    return np.column_stack([xs, -(abc[0]*xs+abc[2])/abc[1]])


def calibrated(**kwargs):
    lock = ColumnLock(**kwargs)
    result = None
    for t in np.linspace(0., 1.2, 9):
        result = lock.update(t, line_at(y=200 + 10*t), good_turn(), 20., True)
    return lock, result


class LineTests(unittest.TestCase):
    def test_scaled_equivalent_lines_and_sign(self):
        expected = normalize_line([2, -3, 5])
        for factor in [.001, 7, -2, -1e5]:
            np.testing.assert_allclose(normalize_line(np.array([2, -3, 5])*factor), expected)
        self.assertGreaterEqual(expected[1], 0)
        np.testing.assert_allclose(normalize_line([-1, 0, 5]), [1, 0, -5])

    def test_degenerate_rejected(self):
        for abc in ([0, 0, 1], [np.nan, 1, 0], [1, 2]):
            with self.assertRaises(ValueError):
                normalize_line(abc)

    def test_robust_tls(self):
        rng = np.random.default_rng(4)
        x = np.linspace(0, 300, 150)
        y = -.22*x + 150 + rng.normal(0, .25, len(x))
        y[[20, 45, 70]] += 15
        fit = fit_line(np.column_stack([x, y]))
        self.assertTrue(fit["valid"])
        self.assertLess(abs(fit["angle_deg"] - np.degrees(np.arctan(-.22))), .15)
        self.assertLess(fit["rmse"], .5)
        self.assertFalse(fit_line([[2, 2], [2, 2]])["valid"])


class TurnTests(unittest.TestCase):
    def make_path(self, side="right", noise=.25):
        rng = np.random.default_rng(9)
        x = np.linspace(0, 260, 261)
        h = np.minimum(x-120, 0) if side == "right" else np.maximum(x-120, 0)
        y = 300-.2*x + .004*h*h + rng.normal(0, noise, len(x))
        return np.column_stack([x, y])

    def test_right_free_c1_turn(self):
        turn = estimate_turn(self.make_path(), "right", 20)
        self.assertTrue(turn["valid"], turn)
        self.assertLess(abs(turn["x"]-120), 3.)
        self.assertLess(abs(turn["slope"]+.2), .02)
        self.assertGreater(turn["quality"], .4)
        self.assertGreater(turn["uncertainty_px"], 0)

    def test_left_free_mirror(self):
        turn = estimate_turn(self.make_path("left"), "left", 20)
        self.assertTrue(turn["valid"], turn)
        self.assertLess(abs(turn["x"]-120), 3.)

    def test_straight_and_short_paths_rejected(self):
        x = np.arange(200.)
        self.assertFalse(estimate_turn(np.column_stack([x, 300-.2*x]))["valid"])
        self.assertFalse(estimate_turn(self.make_path()[:15])["valid"])

    def test_weak_curvature_must_not_force_lock(self):
        rng = np.random.default_rng(2)
        x = np.linspace(0, 260, 261)
        y = 300-.2*x + .00001*np.minimum(x-120, 0)**2 + rng.normal(0, .5, len(x))
        self.assertFalse(estimate_turn(np.column_stack([x, y]))["valid"])

    def test_trace_outliers(self):
        path = self.make_path()
        path[[10, 60, 190, 225], 1] += 20
        turn = estimate_turn(path)
        self.assertTrue(turn["valid"], turn)
        self.assertLess(abs(turn["x"]-120), 4.)


class LockTests(unittest.TestCase):
    def test_fixed_normal_C_uses_robust_current_points(self):
        lock, _ = calibrated()
        raw = line_at(y=214.)
        points = points_on_line(raw)
        points[-1, 1] += 200.  # A single bad supported point cannot decide median C.
        result = lock.update(1.4, raw, good_turn(), 20., True, line_points_xy=points)
        self.assertTrue(result["count_allowed"])
        model = np.asarray(result["locked_abc"])
        self.assertAlmostEqual(model[2], -float(np.median(points @ model[:2])))
        self.assertEqual(result["C_fit_basis"], "robust_current_measured_points_fixed_unit_normal")
        self.assertEqual(result["frozen_C_fit_point_count"], len(points))
        self.assertAlmostEqual(result["frozen_vs_raw_intersection_drift_px"], 0., places=8)

    def test_small_angle_far_extrapolation_is_blocked_by_intersection_gate(self):
        lock, _ = calibrated()
        raw = line_at(y=214., angle=-11.5)
        result = lock.update(1.4, raw, good_turn(), 20., True, line_points_xy=points_on_line(raw))
        self.assertLess(abs(result["angle_drift_deg"]), 2.)
        self.assertGreater(abs(result["frozen_vs_raw_intersection_drift_px"]), 20.)
        self.assertEqual(result["frozen_intersection_gate_limit_px"], 8.)
        self.assertFalse(result["count_allowed"])
        self.assertEqual(result["reason"], "frozen_intersection_mismatch")

    def test_persistent_frozen_intersection_failure_preserves_evidence(self):
        lock, _ = calibrated()
        for t in [1.4, 1.6, 1.8]:
            raw = line_at(y=200+10*t, angle=-11.5)
            result = lock.update(t, raw, good_turn(), 20., True, line_points_xy=points_on_line(raw))
        self.assertIsNone(result["d"])
        self.assertIsNone(result["locked_abc"])
        self.assertEqual(result["state"], "assumption_failed")
        self.assertEqual(result["reason"], "persistent_frozen_intersection_drift")
        self.assertGreater(abs(result["frozen_vs_raw_intersection_drift_px"]), 20.)
        self.assertEqual(result["frozen_intersection_mismatch_frames"], 3)
        self.assertEqual(result["invalidated_d"], 100.)

    def test_frozen_intersection_gate_recovers_and_requires_consecutive_failures(self):
        lock, _ = calibrated()
        for t in [1.4, 1.6]:
            raw = line_at(y=200+10*t, angle=-11.5)
            result = lock.update(t, raw, None, 20., line_points_xy=points_on_line(raw))
        self.assertEqual(result["frozen_intersection_mismatch_frames"], 2)
        # A missing current S1 line interrupts the consecutive mismatch run.
        result = lock.update(1.8, None, None, 20.)
        self.assertFalse(result["count_allowed"])
        raw = line_at(y=220., angle=-11.5)
        result = lock.update(2., raw, None, 20., line_points_xy=points_on_line(raw))
        self.assertEqual(result["frozen_intersection_mismatch_frames"], 1)
        raw = line_at(y=222.)
        result = lock.update(2.2, raw, good_turn(), 20., True, line_points_xy=points_on_line(raw))
        self.assertTrue(result["count_allowed"])
        self.assertEqual(result["frozen_intersection_mismatch_frames"], 0)

    def test_no_points_keeps_labelled_approximation(self):
        lock, _ = calibrated()
        result = lock.update(1.4, line_at(y=214., angle=-11.5), None, 20.)
        self.assertTrue(result["count_allowed"])
        self.assertEqual(result["C_fit_basis"], "raw_line_at_locked_column_no_source_points_available")
        self.assertAlmostEqual(result["frozen_vs_raw_intersection_drift_px"], 0., places=8)
        self.assertEqual(result["frozen_C_fit_point_count"], 0)

    def test_invalid_or_sparse_supplied_points_cannot_authorize_count(self):
        for points in ([[1., np.nan], [2., 3.]], [[1., 2.]], [[1., 2.], [1., 2.]], [1., 2., 3.]):
            with self.subTest(points=points):
                lock, _ = calibrated()
                result = lock.update(1.4, line_at(y=214.), good_turn(), 20., True,
                                     line_points_xy=points)
                self.assertFalse(result["count_allowed"])
                self.assertIsNone(result["locked_abc"])
                self.assertEqual(result["reason"], "current_line_points_invalid")

    def test_requires_complete_and_distributed_calibration(self):
        lock = ColumnLock()
        for i in range(20):
            result = lock.update(i*.1, line_at(), good_turn(), 20., False)
        self.assertIsNone(result["d"])
        lock = ColumnLock()
        for i in range(50):
            result = lock.update(i*.01, line_at(), good_turn(), 20., True)
        self.assertIsNone(result["d"])
        self.assertLess(result["calibration_frames"], 8)

    def test_lock_and_missing_turn_current_line_count(self):
        lock, result = calibrated()
        self.assertEqual(result["state"], "locked")
        self.assertAlmostEqual(result["d"], 100)
        self.assertGreaterEqual(result["lock_time"], 1.)
        result = lock.update(1.4, line_at(y=214), None, 20., False)
        self.assertTrue(result["count_allowed"])
        self.assertTrue(result["audit_missing"])
        self.assertAlmostEqual(result["d"], 100)
        np.testing.assert_allclose(result["locked_abc"][:2], normalize_line(line_at())[:2])
        result = lock.update(1.6, line_at(y=216), good_turn(), 20., False)
        self.assertTrue(result["count_allowed"])
        self.assertTrue(result["audit_missing"])

    def test_audit_refresh_does_not_move_column_and_expiry(self):
        lock, _ = calibrated(max_unverified_lock_age_s=2.)
        result = lock.update(1.5, line_at(y=215), good_turn(102), 20., True)
        self.assertEqual(result["d"], 100)
        for t in [2., 2.5, 3., 3.6]:
            result = lock.update(t, line_at(y=200+10*t), None, 20., False)
        self.assertFalse(result["count_allowed"])
        self.assertEqual(result["reason"], "turn_audit_expired")
        result = lock.update(3.8, line_at(y=238), good_turn(101), 20., True)
        self.assertTrue(result["count_allowed"])
        self.assertEqual(result["d"], 100)

    def test_angle_drift_during_calibration(self):
        lock = ColumnLock()
        reasons = []
        for t in np.linspace(0, 1.2, 9):
            result = lock.update(t, line_at(angle=-12+5*t), good_turn(), 20., True)
            reasons.append(result["reason"])
        self.assertIn("constant_angle_assumption_failed", reasons)
        self.assertIsNone(result["d"])

    def test_column_drift_during_calibration(self):
        lock = ColumnLock()
        reasons = []
        for t in np.linspace(0, 1.2, 9):
            result = lock.update(t, line_at(), good_turn(100+20*t), 20., True)
            reasons.append(result["reason"])
        self.assertIn("constant_column_assumption_failed", reasons)
        self.assertIsNone(result["d"])

    def test_persistent_tracking_mismatch_resets(self):
        for mode in ("angle", "column"):
            lock, _ = calibrated()
            for t in [1.4, 1.6, 1.8]:
                result = lock.update(t, line_at(y=220, angle=-17 if mode=="angle" else -12),
                                     good_turn(125 if mode=="column" else 100), 20., True)
            self.assertIsNone(result["d"])
            self.assertFalse(result["count_allowed"])
            self.assertIn("persistent_", result["reason"])
            self.assertIsNotNone(result["invalidated_d"])
            self.assertIsNotNone(result["angle_drift_deg"])

    def test_nonfinite_uncertainty_cannot_refresh_audit(self):
        for uncertainty in (np.inf, np.nan, None, "missing"):
            with self.subTest(uncertainty=uncertainty):
                lock, _ = calibrated(max_unverified_lock_age_s=1.)
                bad = good_turn()
                if uncertainty == "missing":
                    bad.pop("uncertainty_px")
                else:
                    bad["uncertainty_px"] = uncertainty
                for t in [1.4, 1.8, 2.4]:
                    result = lock.update(t, line_at(y=200+10*t), bad, 20., True)
                self.assertFalse(result["count_allowed"])
                self.assertEqual(result["reason"], "turn_audit_expired")

    def test_stop_is_not_reverse_but_persistent_reverse_resets(self):
        lock, _ = calibrated()
        for t, y in [(1.4, 214), (1.6, 216), (1.8, 218), (2., 218), (2.2, 218)]:
            result = lock.update(t, line_at(y=y), good_turn(), 20., True)
        self.assertTrue(result["count_allowed"])
        for t, y in [(2.4, 214), (2.6, 210), (2.8, 206), (3., 202)]:
            result = lock.update(t, line_at(y=y), good_turn(), 20., True)
        self.assertIsNone(result["d"])
        self.assertEqual(result["reason"], "apparent_direction_reversal")

    def test_invalid_current_line_and_reset(self):
        lock, _ = calibrated()
        result = lock.update(1.4, [1, 0, -100], None, 20.)
        self.assertFalse(result["count_allowed"])
        self.assertEqual(result["state"], "degraded")
        result = lock.update(1.6, line_at(), good_turn(), 20., True, reset=True)
        self.assertIsNone(result["d"])
        self.assertEqual(result["calibration_frames"], 1)

    def test_causal_prefix_is_future_invariant(self):
        a, b = ColumnLock(), ColumnLock()
        def serial(r):
            return (r["state"], r["d"], r["lock_time"], r["count_allowed"], r["reason"])
        prefix_a, prefix_b = [], []
        for t in np.linspace(0, 1.8, 13):
            args = (t, line_at(y=200+10*t), good_turn(), 20., True)
            prefix_a.append(serial(a.update(*args)))
            prefix_b.append(serial(b.update(*args)))
        for t in [2., 2.2, 2.4]:
            a.update(t, line_at(angle=-40), good_turn(150), 20., True)
            b.update(t, line_at(y=200+10*t), None, 20., False)
        self.assertEqual(prefix_a, prefix_b)
        self.assertIsNone(a.d)
        self.assertIsNotNone(b.d)

    def test_nonmonotonic_timestamp_resets(self):
        lock, _ = calibrated()
        result = lock.update(1.0, line_at(), good_turn(), 20., True)
        self.assertEqual(result["reason"], "nonmonotonic_timestamp")
        self.assertIsNone(result["d"])


class CountTests(unittest.TestCase):
    def test_observations_separate_from_pitch_distance(self):
        result = count_crossings([10, 30, 50], 0, 60, 20)
        self.assertEqual(result["observed_grooves"], 3)
        self.assertEqual(result["continuous_pitch_distance"], 3)
        self.assertEqual(result["phase_offset_applied"], 0)
        self.assertTrue(result["valid"])

    def test_missing_lines_not_invented(self):
        result = count_crossings([10, 30, 70], 0, 80, 20)
        self.assertEqual(result["observed_grooves"], 3)
        self.assertEqual(result["inferred_grooves"], 0)
        self.assertFalse(result["valid"])
        self.assertIn("missing_or_irregular", result["reason"])

    def test_half_phase_endpoint_and_harmonic(self):
        result = count_crossings([10, 30, 50], 0, 50, 20)
        self.assertEqual(result["observed_grooves"], 2)
        self.assertEqual(result["continuous_pitch_distance"], 2.5)
        result = count_crossings([10, 20, 30, 40, 50], 0, 60, 20)
        self.assertFalse(result["valid"])
        self.assertIn("harmonic", result["reason"])
        self.assertEqual(count_crossings([], 0, 0, 20)["observed_grooves"], 0)


if __name__ == "__main__":
    unittest.main()
