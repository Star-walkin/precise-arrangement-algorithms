"""Manual, offline review of clean full-resolution centerline pilot images.

No detector, old click set, checkpoint, or algorithm hint is loaded, including
for training images. Holdout uses exactly the same unprompted clean-image UI.
Coordinates are always original full-image pixels, independent of display zoom.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile

from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, ttk


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "vision_geometry_lab/outputs/centerline_v5/pilot/manifest.json"
LINE_CLASSES = ("centerline", "edge_flange", "edge_surface", "free_seed")
STATUSES = ("clear", "partial", "unidentifiable", "absent")
STATUS_LABELS = dict(clear="清晰", partial="部分可辨", unidentifiable="无法区分", absent="活动绳不在画面")
STATUS_CODES = {label: code for code, label in STATUS_LABELS.items()}
COLORS = dict(centerline="#41ff70", edge_flange="#ff80ee", edge_surface="#50dfff",
              free_seed="#ffb347", ignore="#ffff55")
CLASS_LABELS = dict(centerline="中心线", edge_flange="靠法兰侧绳体边", edge_surface="另一侧绳体边",
                    free_seed="自由段锚", ignore="忽略区")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_schema(sample):
    if not isinstance(sample, dict) or not isinstance(sample.get("sample_id"), str) or not sample["sample_id"].strip():
        raise ValueError("sample_id must be a nonempty string")
    if not isinstance(sample.get("group_id"), str) or not sample["group_id"]:
        raise ValueError("group_id is required")
    if sample.get("split") not in ("train", "holdout"):
        raise ValueError("split must be train or holdout")
    for field, suffix in (("image_path", ".png"), ("annotation_path", ".json")):
        if not isinstance(sample.get(field), str):
            raise ValueError(f"{field} must be an absolute path string")
        path = Path(sample.get(field, ""))
        if not path.is_absolute() or path.suffix.lower() != suffix:
            raise ValueError(f"{field} must be an absolute {suffix} path")
    digest = sample.get("image_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
        raise ValueError("image_sha256 must contain 64 hexadecimal characters")
    shape = sample.get("image_shape_hw")
    if not isinstance(shape, list) or len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError("image_shape_hw must be positive integer [height, width]")
    roi = sample.get("roi_xyxy")
    if not isinstance(roi, list) or len(roi) != 4 or any(type(v) is not int for v in roi):
        raise ValueError("roi_xyxy must contain four integer coordinates")
    h, w = shape
    if not (0 <= roi[0] < roi[2] <= w and 0 <= roi[1] < roi[3] <= h):
        raise ValueError("roi_xyxy lies outside its full image")
    source = sample.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("video_path"), str) or not isinstance(source.get("camera"), str):
        raise ValueError("source.video_path and source.camera are required")
    if type(source.get("frame_index")) is not int or source["frame_index"] < 0:
        raise ValueError("source.frame_index must be a nonnegative integer")
    if not isinstance(source.get("time_s"), (int, float)) or not math.isfinite(source["time_s"]) or source["time_s"] < 0:
        raise ValueError("source.time_s must be finite and nonnegative")
    return sample


def load_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("samples"), list) or not data["samples"]:
        raise ValueError("Manifest requires schema_version=1 and a nonempty samples list")
    identifiers, annotations = set(), set()
    for sample in data["samples"]:
        _sample_schema(sample)
        target = Path(sample["annotation_path"]).resolve()
        if sample["sample_id"] in identifiers or target in annotations:
            raise ValueError("Duplicate sample_id or annotation_path")
        if target == path:
            raise ValueError("annotation_path must not overwrite the manifest")
        identifiers.add(sample["sample_id"])
        annotations.add(target)
    return data


def read_clean_image(sample):
    """Decode precisely the bytes whose SHA is checked; never write the image."""
    _sample_schema(sample)
    raw = Path(sample["image_path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sample["image_sha256"].lower():
        raise ValueError("Image SHA mismatch: do not annotate a changed or different image")
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != "PNG" or [image.height, image.width] != sample["image_shape_hw"]:
            raise ValueError("PNG image dimensions differ from the manifest")
        image.load()
        return image.convert("RGB")


def blank_annotation(sample):
    return dict(schema_version=1, sample_id=sample["sample_id"], image_sha256=sample["image_sha256"].lower(),
                coordinate_system="full_image_xy", status="partial", reviewed=False,
                polylines={name: [] for name in LINE_CLASSES}, ignore_polygons=[], notes="")


def _coordinates(points, shape):
    if not isinstance(points, list):
        raise ValueError("Point lists must be JSON arrays")
    h, w = shape
    out = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Every point must be [x, y]")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in point):
            raise ValueError("Point coordinates must be finite numbers")
        x, y = map(float, point)
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError("Point lies outside original full-image coordinates")
        out.append([x, y])
    return out


def _in_roi(point, sample):
    x0, y0, x1, y1 = sample["roi_xyxy"]
    return x0 <= point[0] < x1 and y0 <= point[1] < y1


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _touching(a, b, c, d):
    c1, c2, c3, c4 = _cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b)
    if c1 * c2 < 0 and c3 * c4 < 0:
        return True
    for value, point, left, right in ((c1, c, a, b), (c2, d, a, b), (c3, a, c, d), (c4, b, c, d)):
        if abs(value) <= 1e-9 and min(left[0], right[0]) - 1e-9 <= point[0] <= max(left[0], right[0]) + 1e-9 and min(left[1], right[1]) - 1e-9 <= point[1] <= max(left[1], right[1]) + 1e-9:
            return True
    return False


def _inside(point, polygon):
    inside = False
    x, y = point
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if (a[1] > y) != (b[1] > y):
            intersection = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if intersection > x:
                inside = not inside
    return inside


def validate_annotation(annotation, sample):
    """Return only schema fields, with no algorithm hints carried through."""
    _sample_schema(sample)
    if not isinstance(annotation, dict) or annotation.get("schema_version") != 1:
        raise ValueError("Annotation requires schema_version=1")
    annotation_hash = annotation.get("image_sha256")
    if annotation.get("sample_id") != sample["sample_id"] or not isinstance(annotation_hash, str) or annotation_hash.lower() != sample["image_sha256"].lower():
        raise ValueError("Annotation sample_id/image SHA does not match this sample")
    if annotation.get("coordinate_system") != "full_image_xy" or annotation.get("reviewed") is not True:
        raise ValueError("Save requires reviewed=true and full_image_xy coordinates")
    status = annotation.get("status")
    if status not in STATUSES:
        raise ValueError("Choose clear, partial, unidentifiable, or absent")
    notes = annotation.get("notes", "")
    if not isinstance(notes, str):
        raise ValueError("Notes must be text")
    if status in ("unidentifiable", "absent") and not notes.strip():
        raise ValueError("unidentifiable/absent requires a note explaining the observation")
    source_lines = annotation.get("polylines")
    if not isinstance(source_lines, dict) or set(source_lines) != set(LINE_CLASSES):
        raise ValueError("Exactly centerline, edge_flange, edge_surface, free_seed polylines are required")
    lines = {name: _coordinates(source_lines[name], sample["image_shape_hw"]) for name in LINE_CLASSES}
    if status == "absent" and any(lines.values()):
        raise ValueError("活动绳不在画面（absent）要求四类折线全部为空，包括自由段锚。若仍有自由绳但入卷身份不明，请选‘无法区分’并备注。")
    for name, line in lines.items():
        if any(not _in_roi(point, sample) for point in line):
            raise ValueError(f"{CLASS_LABELS[name]} 超出标注范围 ROI（roi_xyxy）。请只在框内标注；自由段锚也只取框内一小截。")
        if line and (len(line) < 2 or len({tuple(point) for point in line}) < 2):
            raise ValueError(f"{name}: use at least two distinct points or clear this optional line")
    if status in ("clear", "partial") and (len(lines["centerline"]) < 2 or len(lines["free_seed"]) < 2):
        raise ValueError("clear/partial requires centerline >= 2 and free_seed >= 2 points")
    polygons = annotation.get("ignore_polygons", [])
    if not isinstance(polygons, list) or len(polygons) > 1:
        raise ValueError("This tool supports at most one ignore polygon")
    clean_polygons = []
    for polygon in polygons:
        polygon = _coordinates(polygon, sample["image_shape_hw"])
        if len(polygon) > 1 and polygon[-1] == polygon[0]:
            polygon.pop()  # Accept explicit manual closure; serialize one vertex once.
        if len(polygon) < 3:
            raise ValueError("An ignore polygon requires at least three vertices")
        edges = list(zip(polygon, polygon[1:] + polygon[:1]))
        area2 = abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in edges))
        if area2 < 1e-6:
            raise ValueError("Ignore polygon must enclose a nonzero area")
        for i, (a, b) in enumerate(edges):
            for j in range(i + 1, len(edges)):
                if j == i + 1 or (i == 0 and j == len(edges) - 1):
                    continue
                if _touching(a, b, *edges[j]):
                    raise ValueError("Ignore polygon must not intersect itself")
        for name, line in lines.items():
            if any(_inside(point, polygon) for point in line) or any(_touching(a, b, c, d) for a, b in zip(line, line[1:]) for c, d in edges):
                raise ValueError(f"{name} crosses the ignore region; stop at the visible segment, do not bridge a gap")
        clean_polygons.append(polygon)
    return dict(schema_version=1, sample_id=sample["sample_id"], image_sha256=sample["image_sha256"].lower(),
                coordinate_system="full_image_xy", status=status, reviewed=True,
                polylines=lines, ignore_polygons=clean_polygons, notes=notes)


def _read_annotation_with_revision(sample):
    path = Path(sample["annotation_path"])
    if not path.exists():
        return blank_annotation(sample), None
    raw = path.read_bytes()
    return validate_annotation(json.loads(raw.decode("utf-8-sig")), sample), hashlib.sha256(raw).hexdigest()


def load_annotation(sample):
    read_clean_image(sample)
    return _read_annotation_with_revision(sample)[0]


def save_annotation(sample, annotation, *, expected_existing_sha256=Ellipsis):
    """Atomic label-only save with image binding and optional revision check."""
    clean = validate_annotation(annotation, sample)
    if file_sha256(sample["image_path"]) != sample["image_sha256"].lower():
        raise ValueError("Image changed since review; refusing to save against a different image")
    target = Path(sample["annotation_path"])

    def check_revision():
        if expected_existing_sha256 is not Ellipsis:
            actual = file_sha256(target) if target.exists() else None
            if actual != expected_existing_sha256:
                raise ValueError("Annotation changed on disk; reload it before saving to avoid overwriting another review")

    check_revision()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(clean, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        check_revision()
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return clean


@dataclass
class ViewTransform:
    scale: float = 1.
    offset_x: float = 0.
    offset_y: float = 0.

    def image_to_canvas(self, x, y):
        return x * self.scale + self.offset_x, y * self.scale + self.offset_y

    def canvas_to_image(self, x, y):
        return (x - self.offset_x) / self.scale, (y - self.offset_y) / self.scale

    def zoom_at(self, x, y, factor):
        image_x, image_y = self.canvas_to_image(x, y)
        self.scale = min(32., max(.01, self.scale * factor))
        self.offset_x, self.offset_y = x - image_x * self.scale, y - image_y * self.scale

    def fit_rect(self, rectangle, canvas_width, canvas_height):
        x0, y0, x1, y1 = rectangle
        self.scale = max(.01, min(max(1., canvas_width - 32) / (x1 - x0), max(1., canvas_height - 32) / (y1 - y0)))
        self.offset_x = canvas_width / 2 - (x0 + x1) / 2 * self.scale
        self.offset_y = canvas_height / 2 - (y0 + y1) / 2 * self.scale


class ReviewApp:
    def __init__(self, root, samples, *, smoke=False):
        self.root, self.samples, self.smoke = root, samples, smoke
        self.index, self.image, self.annotation = 0, None, None
        self.revision, self.dirty, self._loading, self._pan_start = None, False, False, None
        self.view, self.photo = ViewTransform(), None
        self.class_var = tk.StringVar(value="centerline")
        self.status_var = tk.StringVar(value=STATUS_LABELS["partial"])
        self.show_var = tk.BooleanVar(value=True)
        self.info_var, self.help_var = tk.StringVar(), tk.StringVar()
        root.title("V5 新入卷绳中心线人工复核（离线，无算法提示）")
        root.geometry("1380x920")
        root.minsize(950, 680)
        root.protocol("WM_DELETE_WINDOW", self._close)
        self._build()
        self.status_var.trace_add("write", self._mark_dirty)
        root.bind("<F2>", self._toggle_annotations)
        root.bind("<Control-s>", lambda event: self._save())
        root.bind("<Control-Right>", lambda event: self._navigate(1))
        root.bind("<Control-Left>", lambda event: self._navigate(-1))
        if samples:
            self._load_current()
        else:
            self.info_var.set("GUI smoke test: no image or annotation files will be created")

    @property
    def sample(self):
        return self.samples[self.index] if self.samples else None

    def _build(self):
        navigation = ttk.Frame(self.root, padding=6)
        navigation.pack(fill="x")
        for text, command in (("上一张", lambda: self._navigate(-1)), ("下一张", lambda: self._navigate(1)),
                              ("保存 Ctrl+S", self._save), ("保存并下一张", self._save_next),
                              ("适应全图", lambda: self._fit(False)), ("关注 ROI", lambda: self._fit(True))):
            ttk.Button(navigation, text=text, command=command).pack(side="left", padx=3)
        ttk.Checkbutton(navigation, text="显示人工标注 F2", variable=self.show_var, command=self._redraw).pack(side="left", padx=8)
        ttk.Label(self.root, textvariable=self.info_var, padding=(8, 2), foreground="#163c67").pack(fill="x")
        classes = ttk.Frame(self.root, padding=(8, 3))
        classes.pack(fill="x")
        for name in (*LINE_CLASSES, "ignore"):
            ttk.Radiobutton(classes, text=CLASS_LABELS[name], variable=self.class_var, value=name, command=self._redraw).pack(side="left", padx=5)
        ttk.Button(classes, text="撤销本类最后点", command=self._undo).pack(side="left", padx=7)
        ttk.Button(classes, text="清空当前类", command=self._clear_class).pack(side="left", padx=3)
        instruction = ("ROI 框仅为标注范围，不是算法答案；折线只在框内点。自由段锚也只标框内一小截。\n"
                       "所有线按 自由段 → 卷面 方向；每类只标一条连续可辨段，缺口即停，不跨遮挡。\n"
                       "两侧边都属于目标绳，不是邻绳或法兰交界；都标时须与中心线覆盖同一可见区间、同向配对。缺一侧则仅中心监督。\n"
                       "左键逐点；Ctrl+滚轮缩放；右键拖动；F2开关人工标注。忽略区≥3点自动闭合，不能与折线相交。")
        ttk.Label(self.root, text=instruction, padding=(8, 5), foreground="#753718").pack(fill="x")
        self.canvas = tk.Canvas(self.root, background="#22262d", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=6)
        self.canvas.bind("<Button-1>", self._add_point)
        self.canvas.bind("<ButtonPress-3>", self._pan_begin)
        self.canvas.bind("<B3-Motion>", self._pan_move)
        self.canvas.bind("<ButtonRelease-3>", lambda event: setattr(self, "_pan_start", None))
        self.canvas.bind("<Control-MouseWheel>", self._zoom)
        self.canvas.bind("<Configure>", lambda event: self._redraw())
        self.canvas.bind("<Motion>", self._position_hint)
        self.canvas.bind("<Control-z>", lambda event: self._undo())
        bottom = ttk.Frame(self.root, padding=6)
        bottom.pack(fill="x")
        ttk.Label(bottom, text="可辨状态").pack(side="left")
        ttk.Combobox(bottom, values=tuple(STATUS_LABELS.values()), textvariable=self.status_var, state="readonly", width=18).pack(side="left", padx=6)
        ttk.Label(bottom, text="清晰/部分可辨：中心线及自由段锚各≥2点。\n无法区分/不在画面：必须备注；不在画面须清空全部折线。").pack(side="left", padx=6)
        ttk.Label(self.root, text="备注（不可辨/缺席时必填；可记录遮挡、反光或身份不确定原因）", padding=(8, 0)).pack(fill="x")
        self.notes = tk.Text(self.root, height=3, wrap="word", undo=True)
        self.notes.pack(fill="x", padx=8, pady=(0, 4))
        self.notes.bind("<<Modified>>", self._notes_modified)
        ttk.Label(self.root, textvariable=self.help_var, padding=(8, 3)).pack(fill="x")

    def _mark_dirty(self, *unused):
        if not self._loading and self.annotation is not None and self.image is not None:
            self.dirty = True
            self._update_info()

    def _notes_modified(self, event=None):
        if self.notes.edit_modified():
            self._mark_dirty()
            self.notes.edit_modified(False)

    def _progress(self):
        count = 0
        for sample in self.samples:
            try:
                label, revision = _read_annotation_with_revision(sample)
                count += bool(revision and label.get("reviewed"))
            except (OSError, ValueError, TypeError, AttributeError):
                pass
        return count

    def _update_info(self):
        if not self.sample:
            return
        sample = self.sample
        split = "HOLDOUT：禁止算法提示" if sample["split"] == "holdout" else "TRAIN：同样从干净图人工标注"
        counts = " / ".join(f"{CLASS_LABELS[name]} {len(self.annotation['polylines'][name])}" for name in LINE_CLASSES) if self.annotation else ""
        self.info_var.set(f"{self.index + 1}/{len(self.samples)}  {sample['sample_id']}  |  {split}  |  已保存 {self._saved_count}/{len(self.samples)}  {'● 未保存' if self.dirty else ''}\n{counts}")
        self.root.title(f"V5 人工复核 — {sample['sample_id']} {'*' if self.dirty else ''}")

    def _load_current(self):
        self._loading = True
        self.image, self.annotation, self.revision = None, None, None
        self.dirty = False
        try:
            image = read_clean_image(self.sample)
            annotation, revision = _read_annotation_with_revision(self.sample)
            self.image, self.annotation, self.revision = image, annotation, revision
            self.status_var.set(STATUS_LABELS[annotation["status"]])
            self.notes.delete("1.0", "end")
            self.notes.insert("1.0", annotation["notes"])
            self.notes.edit_modified(False)
            self.show_var.set(True)
            self.help_var.set("本图 PNG SHA 已核对。未自动填充任何旧点或算法线；保存表示已人工复核。")
        except (OSError, ValueError, TypeError, AttributeError) as error:
            if self.smoke:
                raise
            self.help_var.set(f"本图不可编辑：{error}")
            messagebox.showerror("读取失败：未加载旧状态", str(error), parent=self.root)
        finally:
            self._loading = False
        self._saved_count = self._progress()
        self._update_info()
        self.root.after_idle(lambda: self._fit(True))

    def _working_label(self):
        label = dict(self.annotation)
        displayed_status = self.status_var.get()
        label.update(status=STATUS_CODES.get(displayed_status, displayed_status), notes=self.notes.get("1.0", "end-1c"), reviewed=True)
        return label

    def _save(self):
        if self.image is None or self.annotation is None:
            return False
        try:
            label = save_annotation(self.sample, self._working_label(), expected_existing_sha256=self.revision)
        except (OSError, ValueError, TypeError) as error:
            messagebox.showerror("未保存，请修正", str(error), parent=self.root)
            return False
        self.annotation, self.revision = label, file_sha256(self.sample["annotation_path"])
        self.dirty = False
        self._saved_count = self._progress()
        self._update_info()
        self.help_var.set(f"已原子保存人工标注：{self.sample['annotation_path']}")
        return True

    def _confirm_leave(self):
        if not self.dirty:
            return True
        answer = messagebox.askyesnocancel("存在未保存修改", "是否先保存当前图？\n是：校验并保存；否：丢弃本次修改；取消：留在当前图。", parent=self.root)
        return self._save() if answer is True else answer is False

    def _navigate(self, delta, *, already_saved=False):
        target = self.index + delta
        if target < 0 or target >= len(self.samples):
            self.help_var.set("已到第一张/最后一张；当前图不会被丢弃。")
            return
        if already_saved or self._confirm_leave():
            self.index = target
            self._load_current()

    def _save_next(self):
        if self._save():
            self._navigate(1, already_saved=True)

    def _close(self):
        if self._confirm_leave():
            self.root.destroy()

    def _active_points(self, *, create=False):
        if self.annotation is None:
            return None
        name = self.class_var.get()
        if name == "ignore":
            if not self.annotation["ignore_polygons"]:
                if not create:
                    return []
                self.annotation["ignore_polygons"] = [[]]
            return self.annotation["ignore_polygons"][0]
        return self.annotation["polylines"][name]

    def _add_point(self, event):
        if self.image is None or self.annotation is None:
            return
        if not self.show_var.get():
            self.help_var.set("当前仅查看原图。请按 F2 显示人工标注后再加点，避免隐形修改。")
            return
        x, y = self.view.canvas_to_image(event.x, event.y)
        if 0 <= x < self.image.width and 0 <= y < self.image.height:
            if self.class_var.get() != "ignore" and not _in_roi((x, y), self.sample):
                self.help_var.set("未加点：折线必须在黄色 ROI 框内；自由段锚只标框内的一小截。全图仍可查看。")
                self.canvas.focus_set()
                return
            self._active_points(create=True).append([float(x), float(y)])
            self._mark_dirty()
            self._redraw()
        self.canvas.focus_set()

    def _undo(self):
        points = self._active_points()
        if points:
            points.pop()
            if self.class_var.get() == "ignore" and not points:
                self.annotation["ignore_polygons"] = []
            self._mark_dirty()
            self._redraw()

    def _clear_class(self):
        points = self._active_points()
        if points and messagebox.askyesno("清空当前类", f"清空 {CLASS_LABELS[self.class_var.get()]} 的 {len(points)} 个点？", parent=self.root):
            points.clear()
            if self.class_var.get() == "ignore":
                self.annotation["ignore_polygons"] = []
            self._mark_dirty()
            self._redraw()

    def _fit(self, roi):
        if self.image is None:
            self._redraw()
            return
        rectangle = self.sample["roi_xyxy"] if roi and self.sample else [0, 0, self.image.width, self.image.height]
        self.view.fit_rect(rectangle, self.canvas.winfo_width(), self.canvas.winfo_height())
        self._redraw()

    def _zoom(self, event):
        if self.image is not None:
            self.view.zoom_at(event.x, event.y, 1.2 if event.delta > 0 else 1 / 1.2)
            self._redraw()
        return "break"

    def _pan_begin(self, event):
        self._pan_start = (event.x, event.y, self.view.offset_x, self.view.offset_y)

    def _pan_move(self, event):
        if self._pan_start is not None:
            x, y, ox, oy = self._pan_start
            self.view.offset_x, self.view.offset_y = ox + event.x - x, oy + event.y - y
            self._redraw()

    def _toggle_annotations(self, event=None):
        self.show_var.set(not self.show_var.get())
        self._redraw()
        return "break"

    def _position_hint(self, event):
        if self.image is not None:
            x, y = self.view.canvas_to_image(event.x, event.y)
            self.help_var.set(f"原图坐标 ({x:.2f}, {y:.2f})  |  缩放 {self.view.scale:.2f}×  |  当前类：{CLASS_LABELS[self.class_var.get()]}  |  不跨缺口、不能凭纹理猜身份")

    def _redraw(self):
        self.canvas.delete("all")
        if self.image is None:
            self.canvas.create_text(25, 30, anchor="nw", fill="white", text="没有可编辑图像；请选择有效清晰原图及 manifest。")
            return
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        left, top = self.view.canvas_to_image(0, 0)
        right, bottom = self.view.canvas_to_image(width, height)
        box = (max(0, int(math.floor(left))), max(0, int(math.floor(top))),
               min(self.image.width, int(math.ceil(right))), min(self.image.height, int(math.ceil(bottom))))
        if box[2] > box[0] and box[3] > box[1]:
            crop = self.image.crop(box)
            size = (max(1, round(crop.width * self.view.scale)), max(1, round(crop.height * self.view.scale)))
            resampling = Image.Resampling.NEAREST if self.view.scale >= 4 else Image.Resampling.BILINEAR
            self.photo = ImageTk.PhotoImage(crop.resize(size, resampling))
            x, y = self.view.image_to_canvas(box[0], box[1])
            self.canvas.create_image(x, y, image=self.photo, anchor="nw")
        # A task-range box, never a detector result. Keep it visible when F2
        # hides manual annotations so out-of-range clicks remain understandable.
        if self.sample:
            x0, y0, x1, y1 = self.sample["roi_xyxy"]
            left, top = self.view.image_to_canvas(x0, y0)
            right, bottom = self.view.image_to_canvas(x1, y1)
            self.canvas.create_rectangle(left, top, right, bottom, outline="#ffe168", width=2, dash=(8, 4))
            self.canvas.create_text(left + 7, top + 7, text="ROI 标注范围（非算法答案）", fill="black", anchor="nw")
            self.canvas.create_text(left + 6, top + 6, text="ROI 标注范围（非算法答案）", fill="#ffe168", anchor="nw")
        if not self.show_var.get() or self.annotation is None:
            return
        for name in (*LINE_CLASSES, "ignore"):
            points = self.annotation["ignore_polygons"][0] if name == "ignore" and self.annotation["ignore_polygons"] else ([] if name == "ignore" else self.annotation["polylines"][name])
            if not points:
                continue
            coordinates = [value for point in points for value in self.view.image_to_canvas(*point)]
            color = COLORS[name]
            if name == "ignore" and len(points) >= 3:
                self.canvas.create_polygon(*coordinates, fill=color, stipple="gray25", outline=color, width=2)
            elif len(points) >= 2:
                self.canvas.create_line(*coordinates, fill="#111111", width=5)
                self.canvas.create_line(*coordinates, fill=color, width=2, arrow="last", arrowshape=(9, 11, 4))
            for i, point in enumerate(points):
                x, y = self.view.image_to_canvas(*point)
                self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=color, outline="#111111")
                if i in (0, len(points) - 1):
                    self.canvas.create_text(x + 7, y - 7, text=f"{i + 1}", fill=color, anchor="sw")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--smoke-gui", action="store_true", help="Build a hidden Tk GUI briefly, save nothing, then exit")
    args = parser.parse_args(argv)
    try:
        samples = load_manifest(args.manifest)["samples"] if args.manifest.exists() else []
        if not samples and not args.smoke_gui:
            raise ValueError(f"Manifest not found: {args.manifest}")
        root = tk.Tk()
        if args.smoke_gui:
            root.withdraw()
        app = ReviewApp(root, samples, smoke=args.smoke_gui)
        if args.smoke_gui:
            root.update_idletasks()
            if not samples:
                app.image = Image.new("RGB", (240, 160), "#454545")
                app._fit(False)
            root.after(180, root.destroy)
        root.mainloop()
        if args.smoke_gui:
            print(f"SMOKE_GUI_OK samples={len(samples)}; no labels saved")
        return 0
    except (OSError, ValueError, TypeError, tk.TclError) as error:
        print(f"Centerline review failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
