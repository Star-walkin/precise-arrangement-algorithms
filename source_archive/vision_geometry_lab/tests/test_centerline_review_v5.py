"""Pure validation / atomic persistence tests; no pytest fixtures or user files."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from PIL import Image

from vision_geometry_lab.centerline_review_v5 import (
    ReviewApp, STATUS_CODES, STATUS_LABELS, ViewTransform, blank_annotation, file_sha256, load_annotation, load_manifest,
    read_clean_image, save_annotation, validate_annotation)


class CenterlineReviewV5Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.image_path = self.root / "clean.png"
        Image.new("RGB", (240, 160), (80, 100, 120)).save(self.image_path)
        self.sample = dict(sample_id="sample_1", group_id="capture_left", split="holdout",
                           image_path=str(self.image_path.resolve()), image_sha256=file_sha256(self.image_path),
                           image_shape_hw=[160, 240], roi_xyxy=[20, 20, 220, 140],
                           annotation_path=str((self.root / "labels/sample_1.json").resolve()),
                           source=dict(video_path=str((self.root / "original.avi").resolve()), frame_index=246,
                                       time_s=24.6, camera="left"))

    def tearDown(self):
        self.temporary.cleanup()

    def label(self):
        value = blank_annotation(self.sample)
        value.update(reviewed=True, status="partial")
        value["polylines"]["centerline"] = [[90., 80.], [140., 85.], [180., 100.]]
        value["polylines"]["free_seed"] = [[25., 70.], [65., 75.]]
        return value

    def test_roundtrip_binds_clean_image_without_modifying_it(self):
        before = self.image_path.read_bytes()
        label = save_annotation(self.sample, self.label(), expected_existing_sha256=None)
        loaded = load_annotation(self.sample)
        self.assertEqual(loaded, label)
        self.assertEqual(self.image_path.read_bytes(), before)
        self.assertEqual(loaded["coordinate_system"], "full_image_xy")
        self.assertTrue(loaded["reviewed"])
        self.assertEqual(loaded["polylines"]["edge_flange"], [])
        self.assertEqual(loaded["polylines"]["edge_surface"], [])

    def test_clear_or_partial_requires_centerline_and_free_anchor(self):
        for status in ("clear", "partial"):
            for missing in ("centerline", "free_seed"):
                label = self.label()
                label["status"] = status
                label["polylines"][missing] = []
                with self.assertRaises(ValueError):
                    validate_annotation(label, self.sample)

    def test_absent_and_unidentifiable_need_notes_not_invented_points(self):
        for status in ("absent", "unidentifiable"):
            label = blank_annotation(self.sample)
            label.update(status=status, reviewed=True)
            with self.assertRaises(ValueError):
                validate_annotation(label, self.sample)
            label["notes"] = "Target identity is not discernible in the clean image."
            self.assertEqual(validate_annotation(label, self.sample)["polylines"]["centerline"], [])

    def test_sample_hash_coordinates_and_review_flag_are_required(self):
        changes = (("sample_id", "different"), ("image_sha256", "0" * 64),
                   ("coordinate_system", "crop_xy"), ("reviewed", False))
        for key, value in changes:
            label = self.label()
            label[key] = value
            with self.assertRaises(ValueError):
                validate_annotation(label, self.sample)
        for point in ([240, 80], [-1, 10], [40, float("nan")], [True, 20]):
            label = self.label()
            label["polylines"]["centerline"][1] = point
            with self.assertRaises(ValueError):
                validate_annotation(label, self.sample)

    def test_ignore_region_cannot_be_bridged_as_observed_line(self):
        label = self.label()
        label["ignore_polygons"] = [[[110, 65], [130, 65], [130, 110], [110, 110]]]
        with self.assertRaisesRegex(ValueError, "crosses the ignore"):
            validate_annotation(label, self.sample)
        label["ignore_polygons"] = [[[30, 110], [50, 110], [50, 130], [30, 130]]]
        self.assertEqual(len(validate_annotation(label, self.sample)["ignore_polygons"]), 1)
        label["ignore_polygons"] *= 2
        with self.assertRaises(ValueError):
            validate_annotation(label, self.sample)

    def test_image_change_blocks_load_and_save(self):
        Image.new("RGB", (240, 160), "red").save(self.image_path)
        with self.assertRaisesRegex(ValueError, "SHA"):
            read_clean_image(self.sample)
        with self.assertRaisesRegex(ValueError, "Image changed"):
            save_annotation(self.sample, self.label())
        self.assertFalse(Path(self.sample["annotation_path"]).exists())

    def test_failed_atomic_replace_preserves_previous_label_and_cleans_temp(self):
        target = Path(self.sample["annotation_path"])
        save_annotation(self.sample, self.label())
        old = target.read_bytes()
        label = self.label()
        label["notes"] = "New edit"
        with mock.patch("vision_geometry_lab.centerline_review_v5.os.replace", side_effect=OSError("simulated failure")):
            with self.assertRaises(OSError):
                save_annotation(self.sample, label)
        self.assertEqual(target.read_bytes(), old)
        self.assertEqual(list(target.parent.glob("*.tmp")), [])

    def test_external_annotation_edit_is_not_overwritten(self):
        save_annotation(self.sample, self.label(), expected_existing_sha256=None)
        old_revision = file_sha256(self.sample["annotation_path"])
        changed = self.label()
        changed["notes"] = "Another reviewer changed this file"
        save_annotation(self.sample, changed)
        with self.assertRaisesRegex(ValueError, "changed on disk"):
            save_annotation(self.sample, self.label(), expected_existing_sha256=old_revision)
        self.assertEqual(load_annotation(self.sample)["notes"], changed["notes"])

    def test_manifest_requires_unique_safe_paths_and_image_shape(self):
        path = self.root / "manifest.json"
        path.write_text(json.dumps(dict(schema_version=1, samples=[self.sample])), encoding="utf8")
        self.assertEqual(load_manifest(path)["samples"][0]["split"], "holdout")
        duplicate = copy.deepcopy(self.sample)
        duplicate["sample_id"] = "different_id"
        path.write_text(json.dumps(dict(schema_version=1, samples=[self.sample, duplicate])), encoding="utf8")
        with self.assertRaises(ValueError):
            load_manifest(path)
        bad = dict(self.sample, annotation_path=str(path.resolve()))
        path.write_text(json.dumps(dict(schema_version=1, samples=[bad])), encoding="utf8")
        with self.assertRaisesRegex(ValueError, "overwrite the manifest"):
            load_manifest(path)
        bad_shape = dict(self.sample, image_shape_hw=[150, 240])
        with self.assertRaises(ValueError):
            read_clean_image(bad_shape)

    def test_holdout_has_no_prefilled_algorithm_or_old_click_labels(self):
        sample = dict(self.sample, algorithm_hint=[[1, 2], [3, 4]], old_six_points=[[50, 60]] * 6)
        label = load_annotation(sample)
        self.assertFalse(label["reviewed"])
        self.assertTrue(all(points == [] for points in label["polylines"].values()))
        self.assertEqual(label["ignore_polygons"], [])
        self.assertNotIn("algorithm_hint", label)

    def test_zoom_keeps_mouse_original_coordinate_and_points_do_not_move(self):
        view = ViewTransform(.4, 30., 50.)
        original = view.canvas_to_image(270., 140.)
        view.zoom_at(270., 140., 1.2)
        after = view.canvas_to_image(270., 140.)
        self.assertAlmostEqual(original[0], after[0], places=10)
        self.assertAlmostEqual(original[1], after[1], places=10)
        view.fit_rect([20, 20, 220, 140], 1000, 600)
        x, y = view.image_to_canvas(120, 80)
        self.assertAlmostEqual(x, 500.)
        self.assertAlmostEqual(y, 300.)

    def test_empty_ignore_undo_query_does_not_create_an_invalid_polygon(self):
        app = ReviewApp.__new__(ReviewApp)
        app.annotation = blank_annotation(self.sample)
        app.class_var = mock.Mock()
        app.class_var.get.return_value = "ignore"
        self.assertEqual(app._active_points(), [])
        self.assertEqual(app.annotation["ignore_polygons"], [])
        app._active_points(create=True).append([30., 30.])
        self.assertEqual(app.annotation["ignore_polygons"], [[[30., 30.]]])

    def test_explicit_polygon_closure_is_saved_without_duplicate_vertex(self):
        label = self.label()
        label["ignore_polygons"] = [[[30, 110], [50, 110], [50, 130], [30, 130], [30, 110]]]
        self.assertEqual(len(validate_annotation(label, self.sample)["ignore_polygons"][0]), 4)

    def test_unsaved_cancel_or_failed_save_prevents_leaving(self):
        app = ReviewApp.__new__(ReviewApp)
        app.root = mock.Mock()
        app.dirty = True
        app._save = mock.Mock(return_value=False)
        with mock.patch("vision_geometry_lab.centerline_review_v5.messagebox.askyesnocancel", return_value=None):
            self.assertFalse(app._confirm_leave())
            app._save.assert_not_called()
        with mock.patch("vision_geometry_lab.centerline_review_v5.messagebox.askyesnocancel", return_value=True):
            self.assertFalse(app._confirm_leave())
        with mock.patch("vision_geometry_lab.centerline_review_v5.messagebox.askyesnocancel", return_value=False):
            self.assertTrue(app._confirm_leave())
        app._save.return_value = True
        with mock.patch("vision_geometry_lab.centerline_review_v5.messagebox.askyesnocancel", return_value=True):
            self.assertTrue(app._confirm_leave())

    def test_cancelled_navigation_and_close_preserve_current_sample(self):
        app = ReviewApp.__new__(ReviewApp)
        app.root, app.help_var = mock.Mock(), mock.Mock()
        app.samples, app.index = [{}, {}], 0
        app._confirm_leave = mock.Mock(return_value=False)
        app._load_current = mock.Mock()
        app._navigate(1)
        self.assertEqual(app.index, 0)
        app._load_current.assert_not_called()
        app._close()
        app.root.destroy.assert_not_called()
        app._confirm_leave.return_value = True
        app._navigate(1)
        self.assertEqual(app.index, 1)
        app._load_current.assert_called_once()

    def test_all_polylines_including_free_seed_must_stay_inside_roi(self):
        for name in ("centerline", "edge_flange", "edge_surface", "free_seed"):
            for outside in ([19., 60.], [220., 60.], [80., 140.]):
                label = self.label()
                label["polylines"][name] = [outside, [90., 75.]]
                with self.assertRaisesRegex(ValueError, "roi_xyxy"):
                    validate_annotation(label, self.sample)

    def test_absent_rejects_every_nonempty_target_line_including_free_seed(self):
        for name in ("centerline", "edge_flange", "edge_surface", "free_seed"):
            label = blank_annotation(self.sample)
            label.update(reviewed=True, status="absent", notes="No incoming rope visible")
            label["polylines"][name] = [[40., 70.], [65., 75.]]
            with self.assertRaisesRegex(ValueError, "absent"):
                validate_annotation(label, self.sample)

    def test_chinese_status_display_serializes_the_original_english_schema(self):
        app = ReviewApp.__new__(ReviewApp)
        app.annotation = self.label()
        app.status_var, app.notes = mock.Mock(), mock.Mock()
        app.notes.get.return_value = "Manual note"
        for code, display in STATUS_LABELS.items():
            self.assertEqual(STATUS_CODES[display], code)
            app.status_var.get.return_value = display
            self.assertEqual(app._working_label()["status"], code)

    def test_canvas_click_outside_roi_adds_no_hidden_invalid_point(self):
        app = ReviewApp.__new__(ReviewApp)
        app.annotation, app.image = self.label(), read_clean_image(self.sample)
        app.samples, app.index, app.view = [self.sample], 0, ViewTransform()
        app.class_var, app.show_var = mock.Mock(), mock.Mock()
        app.class_var.get.return_value, app.show_var.get.return_value = "free_seed", True
        app.help_var, app.canvas = mock.Mock(), mock.Mock()
        app._mark_dirty, app._redraw = mock.Mock(), mock.Mock()
        before = copy.deepcopy(app.annotation)
        app._add_point(SimpleNamespace(x=5, y=70))
        self.assertEqual(app.annotation, before)
        app._mark_dirty.assert_not_called()
        app._add_point(SimpleNamespace(x=80, y=75))
        self.assertEqual(app.annotation["polylines"]["free_seed"][-1], [80., 75.])
        app._mark_dirty.assert_called_once()


if __name__ == "__main__":
    unittest.main()
