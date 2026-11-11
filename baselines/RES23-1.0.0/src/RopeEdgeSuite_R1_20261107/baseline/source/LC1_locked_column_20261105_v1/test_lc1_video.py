"""Runner input/coordinate contracts using synthetic JSON/CSV only.

No real video, image, model, reserved dataset or runner.main() is opened.
The artificial observations below test interfaces, not detector accuracy.
Run: python -m unittest -v test_lc1_video.py
"""
import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import lc1_video as video
from lc1_geometry import ColumnLock, normalize_line


def observation(frame_index=0):
    return dict(
        frame_index=frame_index,
        coordinate_system="original_unmirrored_xy",
        P1_H6=dict(source="SYNTHETIC TEST INPUT, not actual P1/H6 inference",
                   complete_supported=True,
                   path_xy=[[float(x), 200. - .2*x] for x in range(0, 160, 20)]),
        S1=dict(source="SYNTHETIC TEST INPUT, not actual S1 inference",
                valid=True, abc=[2., -10., 1000.]),
    )


class TemporaryInputs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="lc1_input_contract_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def observations(self, rows, name="observations.jsonl"):
        path = self.root / name
        text = "\n".join(json.dumps(row, allow_nan=False) for row in rows) + "\n"
        if name.endswith(".gz"):
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write(text)
        else:
            path.write_text(text, encoding="utf-8")
        return path

    def time_inputs(self, indices=(0, 1, 2), times=(.037, .107, .201), expected=3):
        # The AVI path is deliberately nonexistent. timestamps() reads only its
        # neighboring sidecars; any accidental decoding would fail immediately.
        source = self.root / "right_SYNTHETIC_NO_MEDIA.avi"
        (self.root / "session.json").write_text(
            json.dumps(dict(frames=dict(right=expected))), encoding="utf-8")
        path = self.root / "right_frames.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["video_frame", "host_time_s"])
            writer.writeheader()
            for index, t in zip(indices, times):
                writer.writerow(dict(video_frame=index, host_time_s=t))
        return source


class ExternalFrontendTests(TemporaryInputs):
    def test_missing_s1_source_cannot_supply_line(self):
        for source in (None, "", "   "):
            with self.subTest(source=source):
                row = observation()
                if source is None:
                    row["S1"].pop("source")
                else:
                    row["S1"]["source"] = source
                frontend = video.ExternalFrontend(self.observations([row]), 4096, False)
                result = frontend.step(None, 0., 0)
                self.assertIsNone(result["line_abc"])
                self.assertFalse(result["whole_path_candidate_supported"])

    def test_missing_s1_source_disables_current_count_after_lock(self):
        rows = [observation(i) for i in range(10)]
        rows[-1]["S1"].pop("source")
        frontend = video.ExternalFrontend(self.observations(rows), 4096, False)
        lock = ColumnLock()
        turn = dict(valid=True, x=100., y=120., quality=.99,
                    uncertainty_px=.2, reason="synthetic geometry fixture")
        for i, t in enumerate(np.linspace(0., 1.2, 9)):
            front = frontend.step(None, float(t), i)
            result = lock.update(t, front["line_abc"], turn, 20.,
                                 complete_path_supported=front["whole_path_candidate_supported"])
        self.assertTrue(result["count_allowed"])
        self.assertIsNotNone(result["d"])
        front = frontend.step(None, 1.4, 9)
        result = lock.update(1.4, front["line_abc"], None, 20.,
                             complete_path_supported=front["whole_path_candidate_supported"])
        self.assertFalse(result["count_allowed"])
        self.assertEqual(result["reason"], "current_line_invalid")

    def test_complete_supported_requires_literal_true(self):
        for flag in ("false", "true", 1, 0, False, None):
            with self.subTest(flag=flag):
                row = observation()
                row["P1_H6"]["complete_supported"] = flag
                frontend = video.ExternalFrontend(self.observations([row]), 4096, False)
                result = frontend.step(None, 0., 0)
                self.assertFalse(result["whole_path_candidate_supported"])
                self.assertIsNotNone(result["line_abc"])
        frontend = video.ExternalFrontend(self.observations([observation()]), 4096, False)
        self.assertTrue(frontend.step(None, 0., 0)["whole_path_candidate_supported"])

    def test_explicit_invalid_s1_is_not_current_measurement(self):
        for flag in (False, "true", 1):
            row = observation()
            row["S1"]["valid"] = flag
            frontend = video.ExternalFrontend(self.observations([row]), 4096, False)
            self.assertIsNone(frontend.step(None, 0., 0)["line_abc"])

    def test_duplicate_and_negative_external_frame_indices_rejected(self):
        for rows in ([observation(2), observation(2)], [observation(-1)]):
            with self.subTest(indices=[row["frame_index"] for row in rows]):
                with self.assertRaises(ValueError):
                    video.ExternalFrontend(self.observations(rows), 4096, False)

    def test_external_indices_must_be_json_integers_without_coercion(self):
        for index in (0.5, -0.5, 0.0, True, False, "0", None):
            with self.subTest(index=index):
                with self.assertRaisesRegex(ValueError, "JSON integer"):
                    video.ExternalFrontend(self.observations([observation(index)]), 4096, False)

    def test_valid_video_hash_and_camera_binding(self):
        digest = "ab" * 32
        rows = [observation(0), observation(1)]
        for row in rows:
            row.update(video_sha256=digest, camera_role="right")
        frontend = video.ExternalFrontend(self.observations(rows), 4096, False,
                                          source_sha256=digest, camera="right")
        self.assertIsNotNone(frontend.step(None, .1, 1)["line_abc"])

    def test_missing_or_wrong_video_hash_binding_rejected(self):
        digest = "ab" * 32
        for recorded in (None, "cd" * 32):
            with self.subTest(recorded=recorded):
                row = observation()
                row["camera_role"] = "right"
                if recorded is not None:
                    row["video_sha256"] = recorded
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    video.ExternalFrontend(self.observations([row]), 4096, False,
                                            source_sha256=digest, camera="right")
        rows = [observation(0), observation(1)]
        rows[0].update(video_sha256=digest, camera_role="right")
        rows[1].update(video_sha256="cd"*32, camera_role="right")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            video.ExternalFrontend(self.observations(rows), 4096, False,
                                    source_sha256=digest, camera="right")

    def test_missing_or_wrong_camera_binding_rejected(self):
        digest = "ab" * 32
        for recorded in (None, "left"):
            with self.subTest(recorded=recorded):
                row = observation()
                row["video_sha256"] = digest
                if recorded is not None:
                    row["camera_role"] = recorded
                with self.assertRaisesRegex(ValueError, "camera_role"):
                    video.ExternalFrontend(self.observations([row]), 4096, False,
                                            source_sha256=digest, camera="right")

    def test_nonfinite_external_geometry_never_reaches_overlay(self):
        row = observation()
        row["S1"]["abc"] = [1., 2., "NaN"]
        frontend = video.ExternalFrontend(self.observations([row]), 4096, False)
        self.assertIsNone(frontend.step(None, 0., 0)["line_abc"])
        row = observation()
        row["P1_H6"]["path_xy"][0][1] = "NaN"
        frontend = video.ExternalFrontend(self.observations([row]), 4096, False)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            frontend.step(None, 0., 0)

    def test_missing_frame_is_explicit_missing_not_previous_observation(self):
        frontend = video.ExternalFrontend(self.observations([observation(0)]), 4096, False)
        self.assertIsNotNone(frontend.step(None, 0., 0)["line_abc"])
        result = frontend.step(None, .1, 1)
        self.assertIsNone(result["line_abc"])
        self.assertFalse(result["whole_path_candidate_supported"])
        self.assertEqual(result["source"], "missing_external_observation")

    def test_gzip_and_wrong_coordinate_contract(self):
        frontend = video.ExternalFrontend(self.observations([observation()], "input.jsonl.gz"), 4096, True)
        result = frontend.step(None, 0., 0)
        self.assertTrue(result["whole_path_candidate_supported"])
        self.assertAlmostEqual(result["path_xy"][0, 0], 4095.)
        row = observation()
        row["coordinate_system"] = "resized_or_unspecified"
        frontend = video.ExternalFrontend(self.observations([row]), 4096, False)
        with self.assertRaises(ValueError):
            frontend.step(None, 0., 0)

    def test_s1_measured_points_follow_same_mirror_contract(self):
        row = observation()
        row["S1"]["points_xy"] = [[200., 140.], [300., 160.]]
        frontend = video.ExternalFrontend(self.observations([row]), 4096, True)
        result = frontend.step(None, 0., 0)
        np.testing.assert_allclose(result["line_points_xy"], [[3895., 140.], [3795., 160.]])


class CoordinateTests(unittest.TestCase):
    def test_nonfinite_geometry_is_rejected_before_overlay(self):
        for coefficients in ([1., 2., np.nan], [1., 2., np.inf], [np.nan, 1., 0.]):
            self.assertIsNone(video.original_abc(coefficients, 4096, True))
        with self.assertRaises(ValueError):
            video.original_points([[100., np.nan]], 4096, True)

    def test_mirror_equation_and_points_preserve_projection(self):
        width = 4096
        abc = np.array([.35, -1., 340.])
        x = np.linspace(0., width-1., 21)
        points = np.column_stack([x, .35*x+340.])
        canonical_points = video.original_points(points, width, True)
        canonical_abc = np.array(video.original_abc(abc, width, True))
        residual = canonical_points @ canonical_abc[:2] + canonical_abc[2]
        self.assertLess(float(np.max(np.abs(residual))), 1e-9)
        np.testing.assert_allclose(video.original_points(canonical_points, width, True), points)
        restored = video.original_abc(canonical_abc, width, True)
        np.testing.assert_allclose(restored, normalize_line(abc), atol=1e-10)

    def test_equation_display_normalized_for_both_orientations(self):
        for mirror in (False, True):
            expected = np.array(video.original_abc([2., -10., 1000.], 4096, mirror))
            self.assertGreaterEqual(expected[1], 0.)
            self.assertAlmostEqual(float(np.hypot(*expected[:2])), 1.)
            for factor in (.001, -7., 500.):
                actual = video.original_abc(np.array([2., -10., 1000.])*factor, 4096, mirror)
                np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_y_at_invariant_after_mirror(self):
        abc = [.2, -1., 100.]
        mirrored = video.original_abc(abc, 4096, True)
        for original_x in (0., 99., 2048., 4095.):
            self.assertAlmostEqual(video.y_at(abc, original_x),
                                   video.y_at(mirrored, 4095.-original_x))


class TimestampTests(TemporaryInputs):
    def test_valid_sidecar_retains_receipt_times(self):
        source = self.time_inputs()
        times, basis, path = video.timestamps(source, "right", 3, 10.)
        self.assertEqual(times, [.037, .107, .201])
        self.assertEqual(basis, "shared_PC_host_receipt_time_not_exposure")
        self.assertEqual(Path(path), self.root/"right_frames.csv")
        self.assertFalse(source.exists())

    def test_session_frame_count_mismatch_rejected(self):
        source = self.time_inputs(expected=4)
        with self.assertRaisesRegex(ValueError, "session"):
            video.timestamps(source, "right", 3, 10.)

    def test_csv_frame_count_mismatch_never_falls_back(self):
        source = self.time_inputs(indices=(0, 1), times=(.0, .1), expected=3)
        with self.assertRaisesRegex(ValueError, "CSV"):
            video.timestamps(source, "right", 3, 10.)

    def test_noncontinuous_duplicate_reordered_indices_rejected(self):
        for indices in ((0, 2, 3), (0, 1, 1), (1, 0, 2), (-1, 0, 1)):
            with self.subTest(indices=indices):
                source = self.time_inputs(indices=indices)
                with self.assertRaisesRegex(ValueError, "indices"):
                    video.timestamps(source, "right", 3, 10.)

    def test_nonincreasing_and_nonfinite_times_rejected(self):
        for times in ((.0, .1, .1), (.0, .2, .1), (.0, np.nan, .2), (.0, .1, np.inf)):
            with self.subTest(times=times):
                source = self.time_inputs(times=times)
                with self.assertRaisesRegex(ValueError, "times"):
                    video.timestamps(source, "right", 3, 10.)

    def test_absent_sidecars_nominal_fallback_is_labelled(self):
        times, basis, path = video.timestamps(self.root/"NO_MEDIA.avi", "right", 3, 10.)
        self.assertEqual(times, [0., .1, .2])
        self.assertEqual(basis, "AVI_frame_number_over_nominal_fps")
        self.assertIsNone(path)


if __name__ == "__main__":
    unittest.main()
