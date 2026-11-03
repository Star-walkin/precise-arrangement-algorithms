"""Small offline SAM 2 anchor annotation tool for the winding project."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import tkinter as tk
from tkinter import messagebox, ttk
import warnings

import numpy as np
from PIL import Image, ImageTk


ROOT = Path(__file__).resolve().parent
DEFAULT_TASK_ROOT = ROOT / 'task_v1'
MANIFEST = DEFAULT_TASK_ROOT / 'manifest.json'
LABEL_ROOT = DEFAULT_TASK_ROOT / 'labels'
SAM2_ROOT = ROOT / 'vendor' / 'sam2'
CHECKPOINT = ROOT / 'checkpoints' / 'sam2.1_hiera_tiny.pt'
CONFIG = 'configs/sam2.1/sam2.1_hiera_t.yaml'

warnings.filterwarnings('ignore', message='`torch.jit.script` is not supported in Python 3.14')

CLASSES = {
    'wound_rope_surface': ('卷绕绳表面', (30, 220, 100)),
    'flange_inner_face': ('法兰内侧面', (0, 205, 255)),
    'incoming_rope_free_span': ('活动绳可见段', (255, 145, 0)),
}
TANGENT_MODE = 'tangent_point'
VISIBILITY_LABELS = {
    '可见': 'visible',
    '不在本相机视野内': 'out_of_view',
    '遮挡／无法确定': 'uncertain',
}
VISIBILITY_NAMES = {code: label for label, code in VISIBILITY_LABELS.items()}


def missing_annotation_fields(masks, failed_classes, tangent_point,
                              count_valid, require_tangent, require_count,
                              incoming_visibility):
    missing = [CLASSES[name][0] for name, mask in masks.items()
               if (name != 'incoming_rope_free_span' or incoming_visibility == 'visible')
               and name not in failed_classes and (mask is None or not np.any(mask))]
    if incoming_visibility == 'visible':
        if require_tangent and tangent_point is None:
            missing.append('切点')
        if require_count and not count_valid:
            missing.append('人工剩余根数（请输入数字）')
    return missing


class AnnotationCanvas(tk.Canvas):
    def __init__(self, parent, app):
        super().__init__(parent, bg='black', highlightthickness=0)
        self.app = app
        self.source = None
        self.photo = None
        self.scale = 1.0
        self.zoom_factor = 1.0
        self.render_size = (1, 1)
        self._resize_job = None
        self._right_press_xy = None
        self._right_dragged = False
        self.bind('<Button-1>', lambda e: self.app.add_prompt(e, 1))
        self.bind('<ButtonPress-3>', self._right_press)
        self.bind('<B3-Motion>', self._right_move)
        self.bind('<ButtonRelease-3>', self._right_release)
        self.bind('<ButtonPress-2>', self._pan_start)
        self.bind('<B2-Motion>', self._pan_move)
        self.bind('<Control-MouseWheel>', self._wheel)
        self.bind('<Configure>', self._schedule_render)

    def _pan_start(self, event):
        self.scan_mark(event.x, event.y)

    def _pan_move(self, event):
        self.scan_dragto(event.x, event.y, gain=1)

    def _right_press(self, event):
        self._right_press_xy = (event.x, event.y)
        self._right_dragged = False
        self.scan_mark(event.x, event.y)
        return 'break'

    def _right_move(self, event):
        if self._right_press_xy is None:
            return 'break'
        start_x, start_y = self._right_press_xy
        if abs(event.x - start_x) >= 5 or abs(event.y - start_y) >= 5:
            self._right_dragged = True
        if self._right_dragged:
            self.scan_dragto(event.x, event.y, gain=1)
        return 'break'

    def _right_release(self, event):
        clicked = self._right_press_xy is not None and not self._right_dragged
        self._right_press_xy = None
        self._right_dragged = False
        if clicked:
            self.app.add_prompt(event, 0)
        return 'break'

    def _wheel(self, event):
        self.zoom(1.25 if event.delta > 0 else 0.8)
        return 'break'

    def _schedule_render(self, _event=None):
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(80, self.render)

    def set_image(self, image):
        self.source = image
        self.zoom_factor = 1.0
        self.xview_moveto(0)
        self.yview_moveto(0)
        self.render(preserve_center=False)

    def fit_scale(self):
        if self.source is None:
            return 1.0
        return min(max(1, self.winfo_width()) / self.source.width,
                   max(1, self.winfo_height()) / self.source.height)

    def zoom(self, factor):
        if self.source is None:
            return
        limit = max(1.0, 1.0 / max(self.fit_scale(), 1e-6))
        self.zoom_factor = max(1.0, min(limit, self.zoom_factor * factor))
        self.render(preserve_center=True)

    def reset_zoom(self):
        self.zoom_factor = 1.0
        self.xview_moveto(0)
        self.yview_moveto(0)
        self.render(preserve_center=False)

    def event_to_image(self, event):
        if self.source is None:
            return None
        x = self.canvasx(event.x) / self.scale
        y = self.canvasy(event.y) / self.scale
        if 0 <= x < self.source.width and 0 <= y < self.source.height:
            return [round(x, 2), round(y, 2)]
        return None

    def render(self, preserve_center=True):
        self._resize_job = None
        if self.source is None:
            return
        cw, ch = max(1, self.winfo_width()), max(1, self.winfo_height())
        old_w, old_h = self.render_size
        if preserve_center and old_w > 1 and old_h > 1:
            center_x = self.canvasx(cw / 2) / old_w
            center_y = self.canvasy(ch / 2) / old_h
        else:
            center_x = center_y = 0.5

        self.scale = min(1.0, self.fit_scale() * self.zoom_factor)
        width = max(1, round(self.source.width * self.scale))
        height = max(1, round(self.source.height * self.scale))
        base = self.source.resize((width, height), Image.Resampling.LANCZOS).convert('RGBA')
        if self.app.overlay_visible.get():
            for class_name, (_label, color) in CLASSES.items():
                if (class_name == 'incoming_rope_free_span'
                        and self.app.collect_incoming_visibility
                        and VISIBILITY_LABELS[self.app.incoming_visibility.get()] != 'visible'):
                    continue
                mask = self.app.masks.get(class_name)
                if mask is None:
                    continue
                alpha = Image.fromarray(mask.astype(np.uint8) * 90).resize(
                    (width, height), Image.Resampling.NEAREST)
                layer = Image.new('RGBA', (width, height), (*color, 0))
                layer.putalpha(alpha)
                base = Image.alpha_composite(base, layer)

        self.photo = ImageTk.PhotoImage(base.convert('RGB'))
        self.delete('all')
        self.create_image(0, 0, image=self.photo, anchor='nw')
        self.render_size = (width, height)
        scroll_w, scroll_h = max(cw, width), max(ch, height)
        self.configure(scrollregion=(0, 0, scroll_w, scroll_h))
        if preserve_center:
            left = (center_x * width - cw / 2) / scroll_w
            top = (center_y * height - ch / 2) / scroll_h
            self.xview_moveto(max(0.0, min(max(0.0, (scroll_w-cw)/scroll_w), left)))
            self.yview_moveto(max(0.0, min(max(0.0, (scroll_h-ch)/scroll_h), top)))
        else:
            self.xview_moveto(0); self.yview_moveto(0)
        self._draw_markers()
        self.app.update_zoom_text()

    def _draw_markers(self):
        if not self.app.overlay_visible.get():
            return
        radius = max(3, round(6 / max(self.scale, .2))) * self.scale
        for class_name, prompt_list in self.app.prompts.items():
            if (class_name == 'incoming_rope_free_span'
                    and self.app.collect_incoming_visibility
                    and VISIBILITY_LABELS[self.app.incoming_visibility.get()] != 'visible'):
                continue
            color = '#%02x%02x%02x' % CLASSES[class_name][1]
            for prompt in prompt_list:
                x, y = prompt['point'][0] * self.scale, prompt['point'][1] * self.scale
                if prompt['label'] == 1:
                    self.create_oval(x-radius, y-radius, x+radius, y+radius,
                                     fill=color, outline='white', width=2)
                else:
                    self.create_line(x-radius, y-radius, x+radius, y+radius,
                                     fill='#ff1744', width=3)
                    self.create_line(x-radius, y+radius, x+radius, y-radius,
                                     fill='#ff1744', width=3)
        if (self.app.tangent_point is not None
                and (not self.app.collect_incoming_visibility
                     or VISIBILITY_LABELS[self.app.incoming_visibility.get()] == 'visible')):
            x = self.app.tangent_point[0] * self.scale
            y = self.app.tangent_point[1] * self.scale
            r = 8
            self.create_line(x-r, y, x+r, y, fill='#ff00ff', width=3)
            self.create_line(x, y-r, x, y+r, fill='#ff00ff', width=3)


class LabelApp:
    def __init__(self, root):
        self.root = root
        self.root.title('SAM 2排线锚点辅助标注（离线，不控制设备）')
        self.root.geometry('1500x960')
        self.task_payload = json.loads(MANIFEST.read_text(encoding='utf-8'))
        self.tasks = self.task_payload['tasks']
        self.require_remaining_count = bool(
            self.task_payload.get('require_remaining_rope_count', False))
        self.require_tangent_point = bool(
            self.task_payload.get('require_tangent_point', True))
        self.collect_physical_direction = bool(
            self.task_payload.get('collect_physical_direction', False))
        self.collect_incoming_visibility = bool(
            self.task_payload.get('collect_incoming_visibility', False))
        self.index = 0
        self.predictor = None
        self.torch = None
        self.device = 'cpu'
        self.image = None
        self.prompts = {name: [] for name in CLASSES}
        self.masks = {name: None for name in CLASSES}
        self.candidates = {name: None for name in CLASSES}
        self.candidate_scores = {name: [] for name in CLASSES}
        self.candidate_index = {name: 0 for name in CLASSES}
        self.failed_classes = set()
        self.tangent_point = None
        self.mode = tk.StringVar(value='wound_rope_surface')
        self.overlay_visible = tk.BooleanVar(value=True)
        self.title_text = tk.StringVar()
        self.status_text = tk.StringVar(value='正在初始化SAM 2……')
        self.zoom_text = tk.StringVar(value='适应窗口')
        self.remaining_count = tk.StringVar(value='unknown')
        self.physical_direction = tk.StringVar(value='unknown')
        self.incoming_visibility = tk.StringVar(value='可见')
        self._build()
        self.root.after(120, self.initialize_model)

    def _build(self):
        tk.Label(self.root,
                 text='左键=目标点　右键短按=背景点　按住右键拖动=平移　Ctrl+滚轮=缩放　F2=隐藏/显示掩膜',
                 bg='#17324d', fg='white', pady=7).pack(fill='x')
        top = ttk.Frame(self.root, padding=5); top.pack(fill='x')
        ttk.Button(top, text='上一张', command=lambda: self.move(-1)).pack(side='left', padx=3)
        ttk.Button(top, text='保存草稿', command=lambda: self.save(False)).pack(side='left', padx=3)
        ttk.Button(top, text='保存并下一张', command=lambda: self.save(True)).pack(side='left', padx=3)
        ttk.Separator(top, orient='vertical').pack(side='left', fill='y', padx=8)
        ttk.Checkbutton(top, text='显示掩膜与点击（F2）', variable=self.overlay_visible,
                        command=self.canvas_render).pack(side='left')
        ttk.Button(top, text='适应窗口（Ctrl+0）', command=self.reset_zoom).pack(side='left', padx=5)
        ttk.Label(top, textvariable=self.zoom_text).pack(side='left', padx=5)
        ttk.Separator(top, orient='vertical').pack(side='left', fill='y', padx=8)
        ttk.Label(top, text='人工剩余根数：').pack(side='left')
        ttk.Entry(top, textvariable=self.remaining_count, width=8).pack(side='left', padx=3)
        if self.collect_physical_direction:
            ttk.Label(top, text='排线方向：').pack(side='left', padx=(10, 0))
            ttk.Combobox(top, textvariable=self.physical_direction, width=9,
                         state='readonly', values=('unknown', '向左', '向右', '静止'))\
                .pack(side='left', padx=3)
        if self.collect_incoming_visibility:
            ttk.Label(top, text='本相机活动绳：').pack(side='left', padx=(10, 0))
            visibility_box = ttk.Combobox(
                top, textvariable=self.incoming_visibility, width=19,
                state='readonly', values=tuple(VISIBILITY_LABELS))
            visibility_box.pack(side='left', padx=3)
            visibility_box.bind('<<ComboboxSelected>>', self.visibility_changed)
        ttk.Label(top, textvariable=self.title_text).pack(side='left', padx=15)

        tools = ttk.LabelFrame(self.root, text='当前标注目标', padding=5); tools.pack(fill='x', padx=5)
        for class_name, (label, color) in CLASSES.items():
            ttk.Radiobutton(tools, text=label, variable=self.mode, value=class_name,
                            command=self.mode_changed).pack(side='left', padx=8)
        ttk.Radiobutton(tools, text='紫色切点（只需左键一点）', variable=self.mode,
                        value=TANGENT_MODE, command=self.mode_changed).pack(side='left', padx=8)
        ttk.Separator(tools, orient='vertical').pack(side='left', fill='y', padx=8)
        ttk.Button(tools, text='撤销最后一点（Backspace）', command=self.undo).pack(side='left', padx=3)
        ttk.Button(tools, text='清空当前目标', command=self.clear_current).pack(side='left', padx=3)
        ttk.Button(tools, text='SAM无法正确分割', command=self.mark_sam_failure).pack(side='left', padx=3)
        ttk.Button(tools, text='◀ 上一候选', command=lambda: self.cycle_candidate(-1)).pack(side='left', padx=3)
        ttk.Button(tools, text='下一候选 ▶', command=lambda: self.cycle_candidate(1)).pack(side='left', padx=3)

        self.canvas = AnnotationCanvas(self.root, self)
        self.canvas.pack(fill='both', expand=True, padx=5, pady=3)
        ttk.Label(self.root, textvariable=self.status_text, wraplength=1460,
                  padding=5).pack(fill='x')
        self.root.bind('<F2>', lambda _e: self.toggle_overlay())
        self.root.bind('<Control-Key-0>', lambda _e: self.reset_zoom())
        self.root.bind('<BackSpace>', lambda _e: self.undo())
        self.root.bind('<Control-s>', lambda _e: self.save(False))
        self.root.bind('<Control-Return>', lambda _e: self.save(True))

    def initialize_model(self):
        try:
            if not CHECKPOINT.is_file():
                raise FileNotFoundError(f'缺少模型文件：{CHECKPOINT}')
            sys.path.insert(0, str(SAM2_ROOT))
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            self.torch = torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = build_sam2(CONFIG, str(CHECKPOINT), device=self.device,
                               apply_postprocessing=False)
            self.predictor = SAM2ImagePredictor(model)
            self.load_task()
        except Exception as exc:
            self.status_text.set(f'初始化失败：{exc}')
            messagebox.showerror('SAM 2初始化失败',
                                 f'{exc}\n\n请先运行“安装SAM2辅助标注环境.cmd”。', parent=self.root)

    def task_dir(self):
        return LABEL_ROOT / Path(self.tasks[self.index]['image_name']).stem

    def load_task(self):
        task = self.tasks[self.index]
        self.status_text.set('正在读取原图并计算SAM 2图像特征，请稍候……')
        self.root.update_idletasks()
        with Image.open(task['image']) as source:
            self.image = source.convert('RGB')
        self.prompts = {name: [] for name in CLASSES}
        self.masks = {name: None for name in CLASSES}
        self.candidates = {name: None for name in CLASSES}
        self.candidate_scores = {name: [] for name in CLASSES}
        self.candidate_index = {name: 0 for name in CLASSES}
        self.failed_classes = set()
        self.tangent_point = None
        self.remaining_count.set(str(task.get('remaining_rope_count') or 'unknown'))
        self.physical_direction.set('unknown')
        self.incoming_visibility.set('可见')
        self._load_existing()
        with self.torch.inference_mode():
            context = (self.torch.autocast('cuda', dtype=self.torch.bfloat16)
                       if self.device == 'cuda' else nullcontext())
            with context:
                self.predictor.set_image(np.array(self.image, copy=True))
        self.canvas.set_image(self.image)
        self._update_title()
        self.mode_changed()

    def _load_existing(self):
        folder = self.task_dir()
        metadata = folder / 'annotation.json'
        if not metadata.is_file():
            return
        data = json.loads(metadata.read_text(encoding='utf-8'))
        self.tangent_point = data.get('tangent_point')
        self.remaining_count.set(str(data.get('remaining_rope_count') or 'unknown'))
        self.physical_direction.set(str(data.get('physical_direction') or 'unknown'))
        saved_visibility = data.get('incoming_visibility')
        if saved_visibility is None:
            # Old files did not ask the annotator about visibility.  Missing
            # rope labels alone must never be interpreted as a negative sample.
            saved_visibility = 'visible'
        self.incoming_visibility.set(VISIBILITY_NAMES.get(saved_visibility, '遮挡／无法确定'))
        for class_name in CLASSES:
            class_data = data.get('classes', {}).get(class_name, {})
            self.prompts[class_name] = class_data.get('prompts', [])
            if class_data.get('sam2_unusable', False):
                self.failed_classes.add(class_name)
            mask_path = folder / f'{class_name}.png'
            if mask_path.is_file():
                with Image.open(mask_path) as mask_image:
                    self.masks[class_name] = np.asarray(mask_image.convert('L')) > 127

    def _update_title(self):
        task = self.tasks[self.index]
        count = self.remaining_count.get().strip() or 'unknown'
        done = sum((LABEL_ROOT / Path(t['image_name']).stem / 'annotation.json').is_file()
                   for t in self.tasks)
        self.title_text.set(
            f'{self.index+1}/{len(self.tasks)}｜{task["clip"]}｜{task["time_s"]:.3f}s｜'
            f'人工剩余{count}根｜已保存{done}/{len(self.tasks)}')

    def mode_changed(self):
        mode = self.mode.get()
        task = self.tasks[self.index]
        if (mode in ('incoming_rope_free_span', TANGENT_MODE)
                and self.collect_incoming_visibility
                and VISIBILITY_LABELS[self.incoming_visibility.get()] != 'visible'):
            self.status_text.set('本相机活动绳不可标注；请先标卷绕面和法兰面，或改选“可见”。')
            self.canvas.render()
            return
        if mode == TANGENT_MODE:
            guidance = '切点模式：左键点击活动绳第一次接触卷绕表面的位置；再次点击会替换。'
        else:
            label = CLASSES[mode][0]
            guidance = (f'{label}：左键点目标内部，右键点误选的背景。先用1～3个左键点；'
                        '候选不理想再补背景点或切换候选。')
            if mode in self.failed_classes:
                guidance += ' 当前已标记为“SAM无法正确分割”；再次点击目标可取消该标记并重试。'
        note = task['quick_review'].get('note') or '无'
        self.status_text.set(f'{guidance}｜此前快速复核备注：{note}')
        self.canvas.render()

    def visibility_changed(self, _event=None):
        visibility = VISIBILITY_LABELS[self.incoming_visibility.get()]
        if visibility != 'visible':
            self.mode.set('wound_rope_surface')
            self.status_text.set(
                '已记录本相机活动绳不可标注。请继续标卷绕面和法兰面；'
                '活动绳、切点和根数可留空。保存时会将当前活动绳标签排除。')
        else:
            self.mode_changed()
        self.canvas.render()

    def add_prompt(self, event, label):
        if self.predictor is None:
            return 'break'
        point = self.canvas.event_to_image(event)
        if point is None:
            return 'break'
        mode = self.mode.get()
        if (mode in ('incoming_rope_free_span', TANGENT_MODE)
                and self.collect_incoming_visibility
                and VISIBILITY_LABELS[self.incoming_visibility.get()] != 'visible'):
            self.status_text.set('本相机活动绳已设为不可标注；如画面中确实可见，请先改选“可见”。')
            return 'break'
        if mode == TANGENT_MODE:
            if label == 1:
                self.tangent_point = point
                self.status_text.set(f'切点已设为({point[0]:.0f}, {point[1]:.0f})。')
                self.canvas.render()
            return 'break'
        if label == 0 and not any(p['label'] == 1 for p in self.prompts[mode]):
            self.status_text.set('请先在目标内部点至少一个左键正样本，再添加右键背景点。')
            return 'break'
        self.failed_classes.discard(mode)
        self.prompts[mode].append({'point': point, 'label': label})
        self.predict_current()
        return 'break'

    def predict_current(self):
        mode = self.mode.get()
        if mode not in CLASSES or not self.prompts[mode]:
            return
        coords = np.asarray([p['point'] for p in self.prompts[mode]], dtype=np.float32)
        labels = np.asarray([p['label'] for p in self.prompts[mode]], dtype=np.int32)
        self.status_text.set('SAM 2正在根据点击更新候选掩膜……')
        self.root.update_idletasks()
        with self.torch.inference_mode():
            context = (self.torch.autocast('cuda', dtype=self.torch.bfloat16)
                       if self.device == 'cuda' else nullcontext())
            with context:
                masks, scores, _logits = self.predictor.predict(
                    point_coords=coords, point_labels=labels, multimask_output=True)
        order = np.argsort(scores)[::-1]
        self.candidates[mode] = masks[order].astype(bool)
        self.candidate_scores[mode] = [float(scores[i]) for i in order]
        self.candidate_index[mode] = 0
        self.masks[mode] = self.candidates[mode][0]
        self.status_text.set(
            f'{CLASSES[mode][0]}已更新：候选1/{len(order)}，模型分数{self.candidate_scores[mode][0]:.3f}。'
            '请以轮廓是否贴合为准，不要只看分数。')
        self.canvas.render()

    def cycle_candidate(self, delta):
        mode = self.mode.get()
        candidates = self.candidates.get(mode)
        if candidates is None or len(candidates) < 2:
            self.status_text.set('当前目标还没有多个候选；先添加目标点。')
            return
        index = (self.candidate_index[mode] + delta) % len(candidates)
        self.candidate_index[mode] = index
        self.masks[mode] = candidates[index]
        score = self.candidate_scores[mode][index]
        self.status_text.set(f'{CLASSES[mode][0]}：候选{index+1}/{len(candidates)}，模型分数{score:.3f}。')
        self.canvas.render()

    def undo(self):
        mode = self.mode.get()
        if mode == TANGENT_MODE:
            self.tangent_point = None
            self.canvas.render()
            return
        if not self.prompts[mode]:
            return
        self.prompts[mode].pop()
        if self.prompts[mode] and any(p['label'] == 1 for p in self.prompts[mode]):
            self.predict_current()
        else:
            self.prompts[mode] = []
            self.masks[mode] = None
            self.candidates[mode] = None
            self.canvas.render()

    def clear_current(self):
        mode = self.mode.get()
        if mode == TANGENT_MODE:
            self.tangent_point = None
        else:
            self.failed_classes.discard(mode)
            self.prompts[mode] = []
            self.masks[mode] = None
            self.candidates[mode] = None
            self.candidate_scores[mode] = []
        self.canvas.render()

    def mark_sam_failure(self):
        mode = self.mode.get()
        if (mode == 'incoming_rope_free_span' and self.collect_incoming_visibility
                and VISIBILITY_LABELS[self.incoming_visibility.get()] != 'visible'):
            self.status_text.set('活动绳不在视野或被遮挡时，使用上方可见性选项；这不是SAM分割失败。')
            return
        if mode == TANGENT_MODE:
            self.status_text.set('切点不使用SAM分割，请直接左键标出；不能确定时不要猜，并在完成前告诉我。')
            return
        self.failed_classes.add(mode)
        self.masks[mode] = None
        self.candidates[mode] = None
        self.candidate_scores[mode] = []
        self.status_text.set(
            f'已记录“{CLASSES[mode][0]}：SAM无法正确分割”。这也是有效测试结论，不要为了完成任务接受错误掩膜。')
        self.canvas.render()

    def save(self, advance, draft=False):
        visibility = (VISIBILITY_LABELS[self.incoming_visibility.get()]
                      if self.collect_incoming_visibility else 'visible')
        count_text = self.remaining_count.get().strip()
        count_valid = False
        try:
            count_value = float(count_text)
            count_valid = math.isfinite(count_value) and count_value >= 0
        except (TypeError, ValueError):
            count_value = None
        missing = missing_annotation_fields(
            self.masks, self.failed_classes, self.tangent_point, count_valid,
            self.require_tangent_point, self.require_remaining_count, visibility)
        if advance and missing:
            messagebox.showwarning('尚未完成', '请先完成：' + '、'.join(missing), parent=self.root)
            return 'break'
        folder = self.task_dir()
        folder.mkdir(parents=True, exist_ok=True)
        if visibility != 'visible':
            previous_mask = folder / 'incoming_rope_free_span.png'
            previous_annotation = folder / 'annotation.json'
            old_data = (json.loads(previous_annotation.read_text(encoding='utf-8'))
                        if previous_annotation.exists() else {})
            old_incoming = old_data.get('classes', {}).get('incoming_rope_free_span', {})
            had_previous_rope_label = (previous_mask.exists() or
                                       bool(old_incoming.get('mask_file')) or
                                       old_data.get('tangent_point') is not None)
            if had_previous_rope_label:
                stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                if previous_annotation.exists():
                    (folder / f'annotation.before_visibility_{stamp}.json').write_bytes(
                        previous_annotation.read_bytes())
                if previous_mask.exists():
                    previous_mask.rename(
                        folder / f'incoming_rope_free_span.before_visibility_{stamp}.png')
            self.prompts['incoming_rope_free_span'] = []
            self.masks['incoming_rope_free_span'] = None
            self.candidates['incoming_rope_free_span'] = None
            self.candidate_scores['incoming_rope_free_span'] = []
            self.failed_classes.discard('incoming_rope_free_span')
            self.tangent_point = None
            self.remaining_count.set('unknown')
        classes = {}
        for name in CLASSES:
            mask = self.masks[name]
            if mask is not None:
                Image.fromarray(mask.astype(np.uint8) * 255).save(folder / f'{name}.png')
            elif name in self.failed_classes:
                (folder / f'{name}.png').unlink(missing_ok=True)
            idx = self.candidate_index[name]
            scores = self.candidate_scores[name]
            classes[name] = {
                'prompts': self.prompts[name],
                'selected_candidate': idx if scores else None,
                'sam2_score': scores[idx] if scores else None,
                'mask_file': f'{name}.png' if mask is not None else None,
                'sam2_unusable': name in self.failed_classes,
                'not_visible': name == 'incoming_rope_free_span' and visibility == 'out_of_view',
                'visibility_uncertain': (name == 'incoming_rope_free_span'
                                         and visibility == 'uncertain'),
            }
        task = self.tasks[self.index]
        if visibility != 'visible':
            saved_count = 'unknown'
        elif count_valid:
            saved_count = f'{count_value:g}'
        else:
            saved_count = count_text or 'unknown'
        task['remaining_rope_count'] = saved_count
        data = {
            'schema_version': 1,
            'image_name': task['image_name'],
            'clip': task['clip'],
            'time_s': task['time_s'],
            'remaining_rope_count': saved_count,
            'incoming_visibility': visibility,
            'physical_direction': (self.physical_direction.get()
                                   if self.collect_physical_direction else 'unknown'),
            'classes': classes,
            'tangent_point': self.tangent_point,
            'completed': not missing,
            'model': 'sam2.1_hiera_tiny',
            'checkpoint': CHECKPOINT.name,
        }
        (folder / 'annotation.json').write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        self._update_title()
        self.status_text.set(f'已保存到：{folder}')
        if advance and self.index < len(self.tasks) - 1:
            self.index += 1
            self.load_task()
        elif advance:
            messagebox.showinfo('本批锚点完成',
                                f'{len(self.tasks)}张锚点已经保存。请关闭工具并告诉我本批标注已完成。',
                                parent=self.root)
        return 'break'

    def move(self, delta):
        new_index = max(0, min(len(self.tasks)-1, self.index + delta))
        if new_index != self.index:
            self.index = new_index
            self.load_task()

    def toggle_overlay(self):
        self.overlay_visible.set(not self.overlay_visible.get())
        self.canvas.render()
        return 'break'

    def canvas_render(self):
        self.canvas.render()

    def reset_zoom(self):
        self.canvas.reset_zoom()
        return 'break'

    def update_zoom_text(self):
        if self.image is not None:
            self.zoom_text.set(f'相对适应 {self.canvas.zoom_factor:.2f}×｜原图{self.canvas.scale*100:.0f}%')


def main():
    global MANIFEST, LABEL_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-dir', type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    task_root = args.task_dir.resolve()
    MANIFEST = task_root / 'manifest.json'
    LABEL_ROOT = task_root / 'labels'
    if not MANIFEST.is_file():
        raise SystemExit(f'缺少任务清单：{MANIFEST}')
    root = tk.Tk()
    app = LabelApp(root)
    smoke_test = args.smoke_test
    if smoke_test:
        root.after(3500, root.destroy)
    root.mainloop()
    if smoke_test:
        if app.predictor is None or app.image is None:
            raise SystemExit(2)
        print('SAM2_LABELER_SMOKE_OK')


if __name__ == '__main__':
    main()
