"""Synthetic UI-data tests only; does not read project frames or start hardware."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v27_blind_review", HERE / "blind_review.py")
review = importlib.util.module_from_spec(spec); spec.loader.exec_module(review)


@pytest.fixture
def case():
    return {"case_id": "R123456", "kind": "material", "source_file_sha256": "0"*64,
            "session": "synthetic", "camera": "synthetic", "crop_xywh": [100, 200, 1024, 1024],
            "frames": [{"frame_id": i+240, "timestamp_s": i*.1+24., "source_pixel_sha256": str(i),
                        "display_pixel_sha256": str(i)} for i in range(21)]}


@pytest.mark.parametrize("scale", [.1, .5, 1., 2., 12.])
def test_pixel_centre_roundtrip_and_no_mirror(scale):
    crop = [1094, 1708, 1024, 1024]; offset = [-72., 44.]
    for native in [[1094, 1708], [1277.25, 2148.5], [2117, 2731]]:
        canvas = review.native_to_canvas(*native, scale, offset, crop)
        restored = review.canvas_to_native(*canvas, scale, offset, crop)
        assert restored == pytest.approx(native)
    a = review.native_to_canvas(1200, 1800, scale, offset, crop)
    b = review.native_to_canvas(1210, 1810, scale, offset, crop)
    assert b[0] > a[0] and b[1] > a[1]


@pytest.mark.parametrize("visibility", ["unknown", "not_visible", "similar_texture"])
def test_unknown_not_forced_to_coordinate(case, visibility):
    result = review.material_label(case, 0, "P01", [120, 220], visibility, None, "unknown", "tester", "draft")
    assert result["x_px"] is None and result["y_px"] is None
    assert not result["material_identity_confirmed"]


def test_material_roundtrip_sparse_same_id(case, tmp_path):
    first = review.material_label(case, 0, "P01", [120.25, 220.75], "visible", 2.5, "unique_texture", "tester", "reviewed")
    later = review.material_label(case, 2, "P01", [123.25, 221.], "visible", 3., "unique_texture", "tester", "reviewed")
    destination = tmp_path / "annotations.json"
    review.save_json(destination, {"material": [first, later]})
    result = review.read_json(destination)
    assert result["material"] == [first, later]
    assert result["material"][0]["point_id"] == result["material"][1]["point_id"]
    assert result["material"][1]["frame_id"]-result["material"][0]["frame_id"] == 2
    assert all(not x["material_identity_confirmed"] for x in result["material"])


@pytest.mark.parametrize("xy", [None, [0., 0.], [float("nan"), 300]])
def test_visible_requires_valid_click(case, xy):
    with pytest.raises(ValueError):
        review.material_label(case, 0, "P01", xy, "visible", None, "unique_texture", "tester", "draft")


def tcase(case):
    case["kind"] = "t1"; case["crop_xywh"] = [0, 0, 4096, 3000]
    return case


def test_t1_direct_interval_roundtrip_not_auto_confirmed(case, tmp_path):
    case = tcase(case)
    label = review.t1_label(case, 3, 6, [100, 200, 40, 60], "actual_entry", None, "direct_climb", "tester", "reviewed")
    assert label["time_interval"] == pytest.approx([24.3, 24.6])
    assert label["last_definitely_old_frame"] == 243
    assert label["clock_mapping_uncertainty"]["status"] == "unknown"
    assert label["independent_t1_confirmed"] is False
    review.save_json(tmp_path/"t1.json", label)
    assert review.read_json(tmp_path/"t1.json") == label


@pytest.mark.parametrize("cue", ["later_side_protrusion", "operator_button_or_hammer", "unknown"])
def test_delayed_and_operator_cue_not_t1(case, cue):
    label = review.t1_label(tcase(case), 3, 6, [100, 200, 40, 60], "delayed_view", .2, cue, "tester", "reviewed")
    assert label["observation_cue"] == cue and not label["independent_t1_confirmed"]
    with pytest.raises(ValueError):
        review.t1_label(case, 3, 6, [100, 200, 40, 60], "actual_entry", .2, cue, "tester", "reviewed")


def test_unknown_t1_allowed(case):
    label = review.t1_label(tcase(case), None, None, None, "unknown", None, "unknown", "tester", "draft")
    assert label["time_interval"] is None
    assert label["first_definitely_new_frame"] is None


@pytest.mark.parametrize("old,new", [(4, 4), (5, 4), (-1, 4), (1, 21), (None, 4)])
def test_invalid_intervals_rejected(case, old, new):
    with pytest.raises(ValueError):
        review.t1_label(tcase(case), old, new, None, "unknown", None, "unknown", "tester", "draft")


def test_sources_allowlisted_no_reserved_data():
    review.assert_allowed_source("D:/x/dual_20260924_111500/left_DA9627065.avi")
    for path in ["D:/x/20260925/left.avi", "D:/other.avi", "D:/x/dual_20260924_111500/right_DA9626975.avi"]:
        with pytest.raises(ValueError): review.assert_allowed_source(path)


def test_static_no_inference_hardware_imports():
    tree = ast.parse((HERE/"blind_review.py").read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): imported.extend(x.name for x in node.names)
        if isinstance(node, ast.ImportFrom): imported.append(node.module or "")
    assert all(not any(word in x.lower() for word in ["serial", "socket", "camera", "mvs", "torch", "tracker", "bocpd", "servo"]) for x in imported)


def test_exclusive_save_and_nonfinite_rejection(tmp_path):
    path = tmp_path/"labels.json"; review.save_json(path, {"a": 1}, exclusive=True)
    with pytest.raises(FileExistsError): review.save_json(path, {"a": 2}, exclusive=True)
    assert review.read_json(path) == {"a": 1}
    with pytest.raises(ValueError): review.save_json(tmp_path/"bad.json", {"x": float("nan")})


def test_synthetic_export_keeps_clean_pixels_and_hides_algorithms(tmp_path):
    import csv
    import numpy as np
    v26 = tmp_path/"v26"; sources = {}
    names = {"left": ("dual_20260924_111500", "DA9627065"), "right": ("dual_20260924_111143", "DA9626975")}
    for side, (session, serial) in names.items():
        folder = tmp_path/session; folder.mkdir()
        source = folder/f"{side}_{serial}.avi"; source.write_bytes(b"synthetic_not_video")
        sources[side] = source
        with (folder/f"{side}_frames.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["video_frame", "host_time_s"]); writer.writeheader()
            writer.writerows({"video_frame": i, "host_time_s": i*.1} for i in range(601))
        review.save_json(v26/"pixel_hashes"/f"{side}.json", {"pixel_hashes": ["f"*64]*601})
    pattern = np.arange(21*16*16, dtype=np.uint8).reshape(21, 16, 16)
    for group in review.CONFIG["material_groups"]:
        folder = v26/"cohorts"/group; folder.mkdir(parents=True)
        review.save_json(folder/"meta.json", {"box": [10, 20, 16, 16], "seeds_native": [[11, 21]], "method": "NEVER_EXPOSE"})
        np.savez_compressed(folder/"data.npz", gray_native=pattern, source_frames=np.arange(21),
                            times=np.arange(21)*.1, pixel_hashes=np.array(["f"*64]*21))
    output = tmp_path/"review"
    package = review.export_package(v26, output, sources, "protocol_hash")
    assert len(package["cases"]) == 6
    assert "NEVER_EXPOSE" not in json.dumps(package)
    assert "seeds_native" not in json.dumps(package)
    assert len([c for c in package["cases"] if c["kind"] == "material"]) == 4
    result = review.validate_package(output/"manifest.json")
    assert result["material_clean_frames"] == 84
    assert result["t1_full_frames"] == 1202
    assert result["source_files_verified"] == 2
    with pytest.raises(FileExistsError): review.export_package(v26, output, sources, "protocol_hash")


def test_launcher_ascii_crlf_exact_package_path():
    launcher = HERE.parents[1]/"复核V27材料与换层.cmd"
    raw = launcher.read_bytes()
    raw.decode("ascii")
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")
    assert b"v27_20261006_002\\blind_review\\manifest.json" in raw
    assert b"%*" in raw and b"pause" in raw


def test_tk_widgets_construct_hidden_without_video_or_hardware(tmp_path, monkeypatch):
    try:
        import tkinter as tk
    except ImportError:
        pytest.skip("Tk is unavailable in this interpreter")
    real_tk = tk.Tk
    created = []
    def hidden_tk(*args, **kwargs):
        try:
            root = real_tk(*args, **kwargs)
        except tk.TclError as exc:
            pytest.skip(f"Tk display unavailable: {exc}")
        root.withdraw(); created.append(root); return root
    monkeypatch.setattr(tk, "Tk", hidden_tk)
    monkeypatch.setattr(review.ReviewApp, "load_case", lambda self: None)
    review.save_json(tmp_path/"manifest.json", {"cases": [{"case_id": "R123456"}]})
    review.save_json(tmp_path/"annotations.json", {"material": [], "t1": []})
    app = None
    try:
        app = review.ReviewApp(tmp_path/"manifest.json")
        app.root.update_idletasks()
        assert app.root.state() == "withdrawn"
        assert app.canvas.bind("<Control-MouseWheel>")
        assert app.canvas.bind("<Button-1>")
        assert app.canvas.bind("<B3-Motion>")
        assert app.casevar.get() == "R123456"
        assert app.visibility.get() == "unknown"
        assert app.capture is None
        # Fixed control rows must fit the declared initial 1400px width.
        for widget in app.root.winfo_children():
            if widget is not app.canvas:
                assert widget.winfo_reqwidth() <= 1400
    finally:
        if app is not None: app.close()
        elif created: created[0].destroy()
