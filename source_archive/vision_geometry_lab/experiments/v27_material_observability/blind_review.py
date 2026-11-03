"""V27 offline blind-to-algorithm material/t1 review; no inference or devices.

The UI reads only this package manifest, clean pixels, and human labels. It does
not read tracker results. Labels are never loaded by experimental inference.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from datetime import datetime, timezone

CONFIG = {"schema": "v27_blind_review_1", "random_seed": 20261005,
          "material_groups": ["left_000240", "right_000140", "left_000000", "right_000300"],
          "material_frames": 21, "t1_full_sources": ["left", "right"],
          "resize_export": False, "initial_frame": 0,
          "labels_for_evaluation_only": True, "automatic_confirmation": False,
          "hardware_access": False, "inference_access": False}
VISIBILITY = ["visible", "unknown", "not_visible", "similar_texture"]
EVIDENCE = ["unique_texture", "repeated_texture", "geometry_only", "inferred", "unknown"]
REVIEW_STATUS = ["draft", "needs_independent_review", "reviewed"]
ENTRY_KIND = ["unknown", "actual_entry", "delayed_view"]
CUE_KIND = ["direct_climb", "later_side_protrusion", "operator_button_or_hammer", "unknown"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        temporary.replace(path)


def assert_allowed_source(path):
    # Defence in depth: this tool is constrained to the two explicit old runs.
    normalized = str(path).replace("\\", "/")
    allowed = ("dual_20260924_111500/left_DA9627065.avi",
               "dual_20260924_111143/right_DA9626975.avi")
    if not any(normalized.endswith(x) for x in allowed):
        raise ValueError("Source is not one of the two frozen development videos")


def canvas_to_native(x, y, scale, offset, crop):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid display scale")
    # A displayed pixel centre maps to the same source pixel centre.
    return [(float(x) - offset[0] + .5) / scale - .5 + crop[0],
            (float(y) - offset[1] + .5) / scale - .5 + crop[1]]


def native_to_canvas(x, y, scale, offset, crop):
    return [(float(x) - crop[0] + .5) * scale - .5 + offset[0],
            (float(y) - crop[1] + .5) * scale - .5 + offset[1]]


def case_source_fields(case, slot):
    if not 0 <= slot < len(case["frames"]):
        raise ValueError("Frame is outside the review case")
    frame = case["frames"][slot]
    return {"source_file_sha256": case["source_file_sha256"],
            "session": case["session"], "camera": case["camera"],
            "frame_id": frame["frame_id"], "timestamp_kind": "sdk_host_receive_seconds",
            "timestamp_s": frame["timestamp_s"], "case_id": case["case_id"],
            "source_pixel_sha256": frame["source_pixel_sha256"],
            "display_pixel_sha256": frame.get("display_pixel_sha256"),
            "coordinate_system": "original_decoded_frame_pixel_centres_xy"}


def material_label(case, slot, point_id, xy, visibility, uncertainty, evidence, annotator, status):
    if not str(point_id).strip() or not str(annotator).strip():
        raise ValueError("Point ID and annotator are required")
    if visibility not in VISIBILITY or evidence not in EVIDENCE or status not in REVIEW_STATUS:
        raise ValueError("Invalid label category")
    if visibility != "visible":
        xy = None  # Do not invent a coordinate for absence/ambiguous identity.
    if xy is not None:
        if len(xy) != 2 or not all(math.isfinite(v) for v in xy):
            raise ValueError("Nonfinite coordinates")
        x0, y0, width, height = case["crop_xywh"]
        if not x0 <= xy[0] < x0 + width or not y0 <= xy[1] < y0 + height:
            raise ValueError("Point outside original image/crop")
    elif visibility == "visible":
        raise ValueError("Click a visible material location first")
    if uncertainty is not None and (not math.isfinite(uncertainty) or uncertainty < 0):
        raise ValueError("Uncertainty must be a nonnegative pixel radius or unknown")
    result = case_source_fields(case, slot)
    result.update(point_id=str(point_id).strip(), x_px=None if xy is None else float(xy[0]),
                  y_px=None if xy is None else float(xy[1]), visibility=visibility,
                  annotation_uncertainty_px=uncertainty, evidence_kind=evidence,
                  annotator=str(annotator).strip(), review_status=status,
                  material_identity_confirmed=False, use="evaluation_only_not_inference")
    return result


def t1_label(case, old_slot, new_slot, region, entry_kind, clock_uncertainty,
             evidence, annotator, status, notes=""):
    if not annotator.strip():
        raise ValueError("Annotator required")
    if entry_kind not in ENTRY_KIND or evidence not in CUE_KIND or status not in REVIEW_STATUS:
        raise ValueError("Invalid t1 category")
    if (old_slot is None) != (new_slot is None):
        raise ValueError("Record both interval endpoints, or leave both unknown")
    if old_slot is not None and not 0 <= old_slot < new_slot < len(case["frames"]):
        raise ValueError("Last definitely old must precede first definitely new")
    if region is not None:
        if len(region) != 4 or not all(math.isfinite(x) for x in region):
            raise ValueError("Invalid observed region")
        x, y, w, h = region
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x+w > case["crop_xywh"][2] or y+h > case["crop_xywh"][3]:
            raise ValueError("Region outside original frame")
    if entry_kind == "actual_entry" and (region is None or old_slot is None or evidence != "direct_climb"):
        raise ValueError("Direct entry needs an old/new interval, region, and direct-climb evidence")
    if clock_uncertainty is not None and (not math.isfinite(clock_uncertainty) or clock_uncertainty < 0):
        raise ValueError("Clock uncertainty must be nonnegative or unknown")
    old = case["frames"][old_slot] if old_slot is not None else None
    new = case["frames"][new_slot] if new_slot is not None else None
    return {"case_id": case["case_id"], "source_file_sha256": case["source_file_sha256"],
            "session": case["session"], "camera": case["camera"],
            "last_definitely_old_frame": old["frame_id"] if old else None,
            "first_definitely_new_frame": new["frame_id"] if new else None,
            "observed_region": region, "actual_entry_or_delayed_view": entry_kind,
            "time_interval": [old["timestamp_s"], new["timestamp_s"]] if old else None,
            "timestamp_kind": "sdk_host_receive_seconds_not_exposure",
            "clock_mapping_uncertainty": {"seconds": clock_uncertainty,
                                           "status": "unknown" if clock_uncertainty is None else "annotator_supplied"},
            "evidence_source": evidence, "observation_cue": evidence if evidence != "direct_climb" else None,
            "annotator": annotator.strip(), "review_status": status, "notes": notes,
            "independent_t1_confirmed": False, "use": "evaluation_only_not_inference"}


def export_package(v26, output, sources, protocol_sha256):
    """Export fixed clean cases after parent freeze. Never reads prediction JSON."""
    import cv2
    import numpy as np
    v26, output = Path(v26), Path(output)
    if output.exists():
        raise FileExistsError("Refuse to overwrite an existing review package")
    output.mkdir(parents=True)
    source_info = {}
    for side in CONFIG["t1_full_sources"]:
        source = Path(sources[side]); assert_allowed_source(source)
        with source.with_name(f"{side}_frames.csv").open(encoding="utf-8-sig", newline="") as f:
            times = list(csv.DictReader(f))
        hashes = read_json(v26 / "pixel_hashes" / f"{side}.json")["pixel_hashes"]
        # V26 publishes a list with per-frame decoded full-frame SHA256.
        source_info[side] = {"path": str(source), "sha": file_sha(source), "times": times, "hashes": hashes}
    specs = [("material", group) for group in CONFIG["material_groups"]] + [("t1", side) for side in CONFIG["t1_full_sources"]]
    rng = random.Random(CONFIG["random_seed"]); rng.shuffle(specs)
    ids = rng.sample(range(100000, 999999), len(specs)); cases = []
    mapping = []
    for (kind, name), number in zip(specs, ids):
        case_id = f"R{number}"; side = name.split("_")[0] if kind == "material" else name
        info = source_info[side]
        case = {"case_id": case_id, "kind": kind, "source_file_sha256": info["sha"],
                "session": Path(info["path"]).parent.name, "camera": Path(info["path"]).stem.split("_")[1],
                "source_path": info["path"], "frames": []}
        if kind == "material":
            folder = v26 / "cohorts" / name; meta = read_json(folder / "meta.json")
            case["crop_xywh"] = meta["box"]
            with np.load(folder / "data.npz", allow_pickle=False) as data:
                if len(data["gray_native"]) != CONFIG["material_frames"]:
                    raise ValueError("Unexpected material cohort length")
                for slot, pixels in enumerate(data["gray_native"]):
                    relative = f"clean/{case_id}/{slot:03d}.png"
                    path = output / relative; path.parent.mkdir(parents=True, exist_ok=True)
                    okay, encoded = cv2.imencode(".png", pixels)
                    if not okay: raise RuntimeError("PNG encoding failed")
                    with path.open("xb") as f: f.write(encoded.tobytes())
                    case["frames"].append({"frame_id": int(data["source_frames"][slot]),
                        "timestamp_s": float(data["times"][slot]),
                        "source_pixel_sha256": str(data["pixel_hashes"][slot]),
                        "display_pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                        "image": relative, "image_file_sha256": file_sha(path)})
        else:
            case["crop_xywh"] = [0, 0, 4096, 3000]
            for slot, row in enumerate(info["times"]):
                hashed = info["hashes"][slot]
                case["frames"].append({"frame_id": int(row["video_frame"]),
                    "timestamp_s": float(row["host_time_s"]),
                    "source_pixel_sha256": hashed if isinstance(hashed, str) else hashed.get("pixel_hash", hashed.get("sha256"))})
        cases.append(case); mapping.append({"case_id": case_id, "kind": kind, "original_group": name})
    manifest = {"schema": CONFIG["schema"], "protocol_sha256": protocol_sha256, "config": CONFIG,
                "blinding": "algorithm outputs hidden; reviewer may recognize previously seen footage; not fully double blind",
                "coordinate_convention": "original decoded pixel centres; no image mirroring or resize on disk",
                "material_display": "V26 gray_native exact pixels; not original colour",
                "t1_display": "full original clean video; starts at frame 0; no event time recommendations",
                "cases": cases}
    save_json(output / "manifest.json", manifest, exclusive=True)
    save_json(output / "audit_mapping.json", mapping, exclusive=True)
    save_json(output / "annotations.json", {"schema": CONFIG["schema"], "material": [], "t1": [],
              "independent_confirmation_required": True}, exclusive=True)
    return manifest


def validate_package(manifest_path, verify_sources=True):
    """Read-only package/environment audit. Does not construct Tk or decode video."""
    import cv2
    import numpy as np
    import PIL
    path = Path(manifest_path); package = read_json(path)
    if package["schema"] != CONFIG["schema"]:
        raise ValueError("Unexpected review schema")
    if len(package["cases"]) != 6:
        raise ValueError("Expected four material cases and two full t1 cases")
    verified = set(); material_frames = 0; t1_frames = 0
    for case in package["cases"]:
        assert_allowed_source(case["source_path"])
        if verify_sources and case["source_path"] not in verified:
            if file_sha(case["source_path"]) != case["source_file_sha256"]:
                raise ValueError("Source-file SHA256 mismatch")
            verified.add(case["source_path"])
        if case["kind"] == "material":
            if len(case["frames"]) != CONFIG["material_frames"]:
                raise ValueError("Incomplete material case")
            for row in case["frames"]:
                image = path.parent / row["image"]
                if file_sha(image) != row["image_file_sha256"]:
                    raise ValueError("PNG hash mismatch")
                pixels = cv2.imdecode(np.frombuffer(image.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
                if pixels is None or hashlib.sha256(pixels.tobytes()).hexdigest() != row["display_pixel_sha256"]:
                    raise ValueError("Gray native pixels changed")
                material_frames += 1
        elif case["kind"] == "t1":
            if len(case["frames"]) != 601 or case["frames"][0]["frame_id"] != 0:
                raise ValueError("t1 review is not the full original recording")
            t1_frames += len(case["frames"])
        else:
            raise ValueError("Unknown case type")
    labels = read_json(path.parent / "annotations.json")
    if labels["schema"] != CONFIG["schema"]:
        raise ValueError("Unexpected annotation schema")
    return {"status": "valid", "material_clean_frames": material_frames, "t1_full_frames": t1_frames,
            "source_files_verified": len(verified), "label_material_records": len(labels["material"]),
            "label_t1_records": len(labels["t1"]), "hardware_initialized": False,
            "opencv": cv2.__version__, "numpy": np.__version__, "pillow": PIL.__version__}


class ReviewApp:
    def __init__(self, manifest_path):
        import tkinter as tk
        from tkinter import ttk, messagebox
        from PIL import Image, ImageTk
        self.tk, self.ttk, self.messagebox, self.Image, self.ImageTk = tk, ttk, messagebox, Image, ImageTk
        self.path = Path(manifest_path); self.base = self.path.parent
        self.manifest = read_json(self.path); self.cases = self.manifest["cases"]
        self.labelpath = self.base / "annotations.json"; self.labels = read_json(self.labelpath)
        self.root = tk.Tk(); self.root.title("V27 干净画面独立复核 · 不显示算法结果 · 不连接设备")
        self.root.geometry("1400x940"); self.capture = None; self.slot = 0
        self.offset = [0., 0.]; self.scale = 1.; self.mark = None; self.region = None; self.region_start = None
        self.old = self.new = None; self.region_mode = False; self.playing = False
        bar = ttk.Frame(self.root); bar.pack(fill="x")
        self.casevar = tk.StringVar(value=self.cases[0]["case_id"])
        combo = ttk.Combobox(bar, textvariable=self.casevar, values=[c["case_id"] for c in self.cases], state="readonly", width=12)
        combo.pack(side="left"); combo.bind("<<ComboboxSelected>>", lambda e: self.load_case())
        for title, call in [("上一帧 ←", lambda: self.move(-1)), ("下一帧 →", lambda: self.move(1)),
                            ("播放/暂停", self.toggle_play), ("适应窗口", self.fit)]:
            ttk.Button(bar, text=title, command=call).pack(side="left", padx=2)
        ttk.Button(bar, text="操作说明", command=self.help).pack(side="left", padx=2)
        ttk.Label(bar, text="复核帧序号(从0)：").pack(side="left")
        self.jump = tk.StringVar(value="0"); ttk.Entry(bar, textvariable=self.jump, width=7).pack(side="left")
        ttk.Button(bar, text="跳转", command=self.seek).pack(side="left")
        ttk.Label(bar, text="Ctrl+滚轮缩放；右键拖动；所有标记为人工草稿，非自动真值").pack(side="left", padx=8)
        self.status = tk.StringVar(); ttk.Label(self.root, textvariable=self.status).pack(fill="x")
        form = ttk.Frame(self.root); form.pack(fill="x")
        def field(label, values=None, default="", width=15):
            ttk.Label(form, text=label).pack(side="left", padx=(6, 1)); var = tk.StringVar(value=default)
            widget = ttk.Combobox(form, textvariable=var, values=values, state="readonly", width=width) if values else ttk.Entry(form, textvariable=var, width=width)
            widget.pack(side="left"); return var
        self.annotator = field("复核人", default="", width=12)
        self.review = field("复核状态", REVIEW_STATUS, REVIEW_STATUS[0], 23)
        self.point = field("同一材料点ID", default="P01", width=8)
        self.visibility = field("可见性", VISIBILITY, "unknown", 15)
        self.uncertainty = field("误差半径px", default="unknown", width=8)
        self.evidence = field("材料依据", EVIDENCE, "unknown", 18)
        row = ttk.Frame(self.root); row.pack(fill="x")
        ttk.Button(row, text="保存当前材料点/未知", command=self.save_material).pack(side="left")
        ttk.Button(row, text="删除当前点本帧标记", command=self.delete_material).pack(side="left", padx=3)
        ttk.Button(row, text="当前帧=最后明确旧层", command=lambda: self.set_endpoint("old")).pack(side="left", padx=3)
        ttk.Button(row, text="当前帧=首次明确新层", command=lambda: self.set_endpoint("new")).pack(side="left", padx=3)
        ttk.Button(row, text="框选观察区域(两次点击)", command=self.set_region_mode).pack(side="left", padx=3)
        ttk.Button(row, text="清空t1区间/区域", command=self.clear_interval).pack(side="left", padx=3)
        ttk.Button(row, text="保存t1复核/未知", command=self.save_t1).pack(side="left", padx=3)
        trow = ttk.Frame(self.root); trow.pack(fill="x")
        def tfield(label, values=None, default="", width=20):
            ttk.Label(trow, text=label).pack(side="left", padx=5); var = tk.StringVar(value=default)
            widget = ttk.Combobox(trow, textvariable=var, values=values, state="readonly", width=width) if values else ttk.Entry(trow, textvariable=var, width=width)
            widget.pack(side="left"); return var
        self.entry = tfield("t1性质", ENTRY_KIND, "unknown")
        self.cue = tfield("事件依据", CUE_KIND, "unknown", 26)
        self.clock = tfield("接收→曝光时钟不确定度(秒)", default="unknown", width=9)
        self.notes = tfield("说明", default="", width=30)
        self.canvas = tk.Canvas(self.root, bg="#171717", highlightthickness=0); self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Control-MouseWheel>", self.zoom)
        self.canvas.bind("<ButtonPress-3>", self.pan_start)
        self.canvas.bind("<B3-Motion>", self.pan_move)
        self.canvas.bind("<Button-1>", self.click)
        self.root.bind("<Left>", lambda e: self.move(-1) if not isinstance(e.widget, (ttk.Entry, ttk.Combobox)) else None)
        self.root.bind("<Right>", lambda e: self.move(1) if not isinstance(e.widget, (ttk.Entry, ttk.Combobox)) else None)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self.load_case)

    def load_case(self):
        self.playing = False
        if self.capture is not None: self.capture.release(); self.capture = None
        self.case = next(c for c in self.cases if c["case_id"] == self.casevar.get())
        self.slot = 0; self.mark = None; self.region = None; self.old = self.new = None
        self.region_start = None; self.region_mode = False
        self.entry.set("unknown"); self.cue.set("unknown"); self.clock.set("unknown"); self.notes.set("")
        old = next((r for r in self.labels["t1"] if r["case_id"] == self.case["case_id"]), None)
        if old:
            lookup = {r["frame_id"]: i for i, r in enumerate(self.case["frames"])}
            self.old = lookup.get(old["last_definitely_old_frame"]); self.new = lookup.get(old["first_definitely_new_frame"])
            self.region = old["observed_region"]; self.entry.set(old["actual_entry_or_delayed_view"])
            self.cue.set(old["evidence_source"]); self.notes.set(old.get("notes", ""))
            self.clock.set(str(old["clock_mapping_uncertainty"]["seconds"]) if old["clock_mapping_uncertainty"]["seconds"] is not None else "unknown")
        if self.case["kind"] == "t1":
            import cv2
            assert_allowed_source(self.case["source_path"])
            self.capture = cv2.VideoCapture(self.case["source_path"])
            if not self.capture.isOpened(): raise RuntimeError("Cannot open frozen video")
        self.load_frame(); self.fit()

    def load_frame(self):
        row = self.case["frames"][self.slot]
        if self.case["kind"] == "material":
            import numpy as np
            path = self.base / row["image"]
            if file_sha(path) != row["image_file_sha256"]: raise RuntimeError("Clean image file hash mismatch")
            self.pixels = self.Image.open(path).copy()
            if hashlib.sha256(np.asarray(self.pixels).tobytes()).hexdigest() != row["display_pixel_sha256"]:
                raise RuntimeError("Clean image pixels mismatch")
        else:
            import cv2
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, row["frame_id"])
            ok, pixels = self.capture.read()
            if not ok: raise RuntimeError("Video frame decode failed")
            if row["source_pixel_sha256"] and hashlib.sha256(pixels.tobytes()).hexdigest() != row["source_pixel_sha256"]:
                raise RuntimeError("Original frame SHA mismatch; cannot silently seek wrong frame")
            self.pixels = self.Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
        self.jump.set(str(self.slot)); self.mark = None; self.render()

    def render(self):
        if not hasattr(self, "pixels"): return
        width = max(1, round(self.pixels.width * self.scale)); height = max(1, round(self.pixels.height * self.scale))
        # Viewport-only rendering avoids huge zoom allocations without changing coordinates.
        cw = max(1, self.canvas.winfo_width()); ch = max(1, self.canvas.winfo_height())
        left = max(0, int(-self.offset[0] / self.scale)); top = max(0, int(-self.offset[1] / self.scale))
        right = min(self.pixels.width, int((cw-self.offset[0])/self.scale)+2)
        bottom = min(self.pixels.height, int((ch-self.offset[1])/self.scale)+2)
        self.canvas.delete("all")
        if right > left and bottom > top:
            crop = self.pixels.crop((left, top, right, bottom))
            resized = crop.resize((max(1, round(crop.width*self.scale)), max(1, round(crop.height*self.scale))), self.Image.Resampling.NEAREST if self.scale >= 1 else self.Image.Resampling.LANCZOS)
            self.photo = self.ImageTk.PhotoImage(resized)
            self.canvas.create_image(self.offset[0]+left*self.scale, self.offset[1]+top*self.scale, image=self.photo, anchor="nw")
        for label in self.labels["material"]:
            if label["case_id"] == self.case["case_id"] and label["frame_id"] == self.case["frames"][self.slot]["frame_id"] and label["x_px"] is not None:
                self.draw_point([label["x_px"], label["y_px"]], label["point_id"], "#2fffd5")
        if self.mark: self.draw_point(self.mark, self.point.get(), "#ff7030")
        if self.region:
            x, y, w, h = self.region
            a = native_to_canvas(x, y, self.scale, self.offset, self.case["crop_xywh"])
            b = native_to_canvas(x+w, y+h, self.scale, self.offset, self.case["crop_xywh"])
            self.canvas.create_rectangle(*a, *b, outline="#2fffd5", width=2)
        self.status.set(f"{self.case['case_id']} | {'材料跨帧对应' if self.case['kind']=='material' else '完整录像t1独立复核'} | 帧 {self.slot}/{len(self.case['frames'])-1} | 缩放 {self.scale:.2f} | t1旧/新复核帧: {self.old}/{self.new} | 不显示算法时间/轨迹；unknown是有效答案")

    def draw_point(self, xy, name, color):
        x, y = native_to_canvas(*xy, self.scale, self.offset, self.case["crop_xywh"])
        self.canvas.create_oval(x-4, y-4, x+4, y+4, outline=color, width=2)
        self.canvas.create_text(x+8, y-8, text=name, fill=color, anchor="w")

    def fit(self):
        if not hasattr(self, "pixels"): return
        self.scale = min(self.canvas.winfo_width()/self.pixels.width, self.canvas.winfo_height()/self.pixels.height)
        self.offset = [(self.canvas.winfo_width()-self.pixels.width*self.scale)/2, (self.canvas.winfo_height()-self.pixels.height*self.scale)/2]
        self.render()

    def zoom(self, event):
        old = self.scale; self.scale = min(12., max(.05, old*(1.2 if event.delta > 0 else 1/1.2)))
        self.offset = [event.x-(event.x-self.offset[0])*self.scale/old, event.y-(event.y-self.offset[1])*self.scale/old]
        self.render()

    def pan_start(self, event): self.pan = (event.x, event.y, *self.offset)
    def pan_move(self, event):
        self.offset = [self.pan[2]+event.x-self.pan[0], self.pan[3]+event.y-self.pan[1]]; self.render()
    def click(self, event):
        xy = canvas_to_native(event.x, event.y, self.scale, self.offset, self.case["crop_xywh"])
        x, y, w, h = self.case["crop_xywh"]
        if not x <= xy[0] < x+w or not y <= xy[1] < y+h: return
        if self.region_mode:
            if self.region_start is None: self.region_start = xy
            else:
                self.region = [min(xy[0], self.region_start[0]), min(xy[1], self.region_start[1]), abs(xy[0]-self.region_start[0]), abs(xy[1]-self.region_start[1])]
                self.region_start = None; self.region_mode = False
        else: self.mark = xy; self.visibility.set("visible")
        self.render()
    def move(self, delta):
        self.slot = max(0, min(len(self.case["frames"])-1, self.slot+delta)); self.load_frame()
    def seek(self):
        try:
            index = int(self.jump.get())
            if not 0 <= index < len(self.case["frames"]): raise ValueError("Frame out of range")
            self.slot = index; self.load_frame()
        except Exception as exc: self.messagebox.showerror("不能跳转", str(exc))
    def toggle_play(self):
        self.playing = not self.playing
        if self.playing: self.play_tick()
    def play_tick(self):
        if not self.playing: return
        if self.slot >= len(self.case["frames"])-1: self.playing = False; return
        self.move(1); self.root.after(100, self.play_tick)
    def set_endpoint(self, which):
        if self.case["kind"] != "t1": self.messagebox.showinfo("选择t1录像", "请切换到完整录像case，材料短段不用于完整事件区间"); return
        setattr(self, which, self.slot); self.render()
    def set_region_mode(self): self.region_mode = True; self.region_start = None
    def clear_interval(self): self.old = self.new = self.region = None; self.render()
    @staticmethod
    def optional_number(text): return None if text.strip().lower() in ("", "unknown") else float(text)
    def persist(self):
        self.labels["saved_utc"] = datetime.now(timezone.utc).isoformat(); save_json(self.labelpath, self.labels)
    def save_material(self):
        try:
            if self.case["kind"] != "material": raise ValueError("材料点请在材料短段中记录")
            label = material_label(self.case, self.slot, self.point.get(), self.mark, self.visibility.get(), self.optional_number(self.uncertainty.get()), self.evidence.get(), self.annotator.get(), self.review.get())
            self.labels["material"] = [r for r in self.labels["material"] if not (r["case_id"] == label["case_id"] and r["frame_id"] == label["frame_id"] and r["point_id"] == label["point_id"] and r["annotator"] == label["annotator"])]
            self.labels["material"].append(label); self.persist(); self.mark = None; self.render()
        except Exception as exc: self.messagebox.showerror("未保存", str(exc))
    def delete_material(self):
        self.labels["material"] = [r for r in self.labels["material"] if not (r["case_id"] == self.case["case_id"] and r["frame_id"] == self.case["frames"][self.slot]["frame_id"] and r["point_id"] == self.point.get() and r["annotator"] == self.annotator.get())]
        self.persist(); self.render()
    def save_t1(self):
        try:
            if self.case["kind"] != "t1": raise ValueError("请在完整录像中记录t1")
            label = t1_label(self.case, self.old, self.new, self.region, self.entry.get(), self.optional_number(self.clock.get()), self.cue.get(), self.annotator.get(), self.review.get(), self.notes.get())
            self.labels["t1"] = [r for r in self.labels["t1"] if not (r["case_id"] == label["case_id"] and r["annotator"] == label["annotator"])]
            self.labels["t1"].append(label); self.persist(); self.messagebox.showinfo("已保存", "仅保存人工复核记录，未自动确认为独立t1")
        except Exception as exc: self.messagebox.showerror("未保存", str(exc))
    def close(self):
        self.playing = False
        if self.capture is not None: self.capture.release()
        self.root.destroy()
    def help(self):
        self.messagebox.showinfo("独立复核说明", "材料短段：先填复核人。左键点击一处可确认的绳纹，命名P01；下一帧如果能认出同一处，仍用P01。不同材料用不同ID。\n\n看不清请选unknown；已不可见选not_visible；只能看到相似周期纹理选similar_texture。这三类不要求点坐标，不要求跨完整21帧。误差半径填原图像素，不是放大屏幕像素；不确定填unknown。\n\nunique_texture=可辨独特纹理；repeated_texture=重复纹理；geometry_only=仅几何；inferred=经验推测。reviewed表示已人工检查，不等于独立材料真值。\n\nt1完整录像：分别定位最后明确旧层、首次明确新层并按对应按钮；框选观察区域后点击两个角。actual_entry只用于入卷位置直接爬上旧层的证据；delayed_view用于侧缘稍后凸起。按钮或敲击只能记录线索。看不清可以保存unknown。时钟不确定度未知时保持unknown。\n\nCtrl+滚轮缩放，右键拖动；左右键逐帧。保存按钮写到本包annotations.json。页面不显示算法轨迹、候选时间或得分；但熟悉录像的人可能认出片段，所以不称完全双盲。")
    def run(self): self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(validate_package(args.manifest), ensure_ascii=False, indent=2))
    else:
        ReviewApp(args.manifest).run()


if __name__ == "__main__":
    main()
