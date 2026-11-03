"""Review only physical direction and edge distance; never rerun SAM 2.

This tool is offline. It cannot send PLC, relay, or servo commands.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image

from edge_count_semantics import (OBSERVED_EDGE, count_kind, display_count,
                                  parse_count, update_directional_count)
from sam2_labeler import AnnotationCanvas, CLASSES


ROOT = Path(__file__).resolve().parent / 'task_dual_thicklayer_v1'


def load_items(task_dirs):
    items = []
    for task_dir in task_dirs:
        manifest = json.loads((task_dir / 'manifest.json').read_text(encoding='utf-8'))
        for task in manifest['tasks']:
            annotation_path = task_dir / 'labels' / Path(task['image_name']).stem / 'annotation.json'
            if not annotation_path.is_file():
                continue
            data = json.loads(annotation_path.read_text(encoding='utf-8'))
            if data.get('incoming_visibility') == 'visible':
                items.append((task_dir.name, task, annotation_path))
    return items


class ReviewApp:
    def __init__(self, root, task_dirs):
        self.root = root
        self.items = load_items(task_dirs)
        if not self.items:
            raise ValueError('没有已完成且活动绳可见的标注；请先完成SAM锚点标注。')
        self.index = 0
        self.image = None
        self.tangent_point = None
        self.prompts = {name: [] for name in CLASSES}
        self.masks = {name: None for name in CLASSES}
        self.overlay_visible = tk.BooleanVar(value=True)
        self.collect_incoming_visibility = False
        self.incoming_visibility = tk.StringVar(value='可见')
        self.direction_var = tk.StringVar(value='unknown')
        self.count_var = tk.StringVar(value='unknown')
        self.heading = tk.StringVar()
        self.status = tk.StringVar()
        self.kind_text = tk.StringVar()
        self.zoom_text = tk.StringVar(value='适应窗口')
        self.root.title('双机边缘根距与方向复核（不运行SAM／不控制设备）')
        self.root.geometry('1450x930')
        self._build()
        self.load_current()

    def _build(self):
        ttk.Label(self.root, text='本工具只修改方向与距本相机所拍边缘的根距；掩膜和切点保持原样。Ctrl+滚轮放大，按住右键拖动。',
                  padding=7).pack(fill='x')
        controls = ttk.Frame(self.root, padding=6)
        controls.pack(fill='x')
        ttk.Button(controls, text='上一张', command=lambda: self.move(-1)).pack(side='left', padx=3)
        ttk.Button(controls, text='保存当前', command=self.save).pack(side='left', padx=3)
        ttk.Button(controls, text='保存并下一张', command=self.save_next).pack(side='left', padx=3)
        ttk.Button(controls, text='下一张（不保存）', command=lambda: self.move(1)).pack(side='left', padx=3)
        ttk.Label(controls, text='物理排线方向：').pack(side='left', padx=(18, 2))
        direction_box = ttk.Combobox(
            controls, textvariable=self.direction_var, state='readonly', width=9,
            values=('unknown', '向左', '向右', '静止'))
        direction_box.pack(side='left')
        direction_box.bind('<<ComboboxSelected>>', self.update_kind)
        ttk.Label(controls, text='人工根数（按下方口径）：').pack(side='left', padx=(18, 2))
        ttk.Entry(controls, textvariable=self.count_var, width=11).pack(side='left')
        ttk.Label(controls, textvariable=self.zoom_text).pack(side='right', padx=8)
        ttk.Button(controls, text='适应窗口', command=lambda: self.canvas.reset_zoom())\
            .pack(side='right')
        ttk.Label(self.root, textvariable=self.heading, padding=5).pack(fill='x')
        ttk.Label(self.root, textvariable=self.kind_text, padding=5).pack(fill='x')
        self.canvas = AnnotationCanvas(self.root, self)
        self.canvas.pack(fill='both', expand=True, padx=5, pady=4)
        ttk.Label(self.root, textvariable=self.status, padding=5).pack(fill='x')

    def add_prompt(self, _event, _label):
        self.status.set('此窗口不编辑掩膜或切点。请只复核方向与根距。')
        return 'break'

    def update_zoom_text(self):
        if self.image is not None:
            self.zoom_text.set(f'缩放 {self.canvas.zoom_factor:.2f}×')

    def load_current(self):
        split, task, annotation_path = self.items[self.index]
        data = json.loads(annotation_path.read_text(encoding='utf-8'))
        with Image.open(task['image']) as source:
            self.image = source.convert('RGB')
        self.tangent_point = data.get('tangent_point')
        self.direction_var.set(data.get('physical_direction') or 'unknown')
        self.count_var.set(display_count(data))
        self.canvas.set_image(self.image)
        self.heading.set(
            f'{self.index+1}/{len(self.items)}｜{split}｜{task["clip"]}｜'
            f'{task["time_s"]:.2f}s｜{task["camera"]}相机拍{OBSERVED_EDGE[task["camera"]]}边缘')
        self.update_kind()
        self.status.set('紫色点为此前人工切点；图像与SAM掩膜不在此工具修改。')

    def update_kind(self, _event=None):
        camera = self.items[self.index][1]['camera']
        kind = count_kind(camera, self.direction_var.get())
        explanation = {
            'remaining': '靠近本侧边缘：填写“剩余约多少根距”，不包含当前活动绳。',
            'departed': '远离本侧边缘：填写“已排约多少根（包含当前活动绳）”；最小为1。这不是实际工字轮转数。',
            'undetermined': '方向未确定：根距可暂留unknown，勿依据预计换向秒数猜测。',
        }[kind]
        self.kind_text.set(explanation)

    def move(self, delta):
        index = max(0, min(len(self.items)-1, self.index + delta))
        if index != self.index:
            self.index = index
            self.load_current()

    def save(self):
        _split, task, annotation_path = self.items[self.index]
        raw_count = self.count_var.get().strip()
        if raw_count.lower() not in ('', 'unknown') and parse_count(raw_count) is None:
            messagebox.showwarning('根距无效', '请输入非负数字，或填写unknown。', parent=self.root)
            return False
        kind = count_kind(task['camera'], self.direction_var.get())
        if kind == 'departed' and parse_count(raw_count) is not None and parse_count(raw_count) < 1:
            messagebox.showwarning('已排根数无效',
                                   '远离时包含当前活动绳，所以已排根数至少为1；看不清请填unknown。',
                                   parent=self.root)
            return False
        original_bytes = annotation_path.read_bytes()
        data = json.loads(original_bytes.decode('utf-8'))
        revised = update_directional_count(
            data, task['camera'], self.direction_var.get(), raw_count)
        if revised != data:
            backup = annotation_path.with_name('annotation.before_edge_count_review.json')
            if not backup.exists():
                backup.write_bytes(original_bytes)
            annotation_path.write_text(json.dumps(revised, ensure_ascii=False, indent=2),
                                       encoding='utf-8')
        self.status.set('已保存方向与根距；原始JSON首次修改前已备份，SAM掩膜和切点未改。')
        return True

    def save_next(self):
        if self.save():
            if self.index + 1 < len(self.items):
                self.move(1)
            else:
                messagebox.showinfo('完成', '本批可见活动绳图像的方向与根距已复核。',
                                    parent=self.root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-dir', type=Path, action='append')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    task_dirs = args.task_dir or [ROOT / 'train', ROOT / 'holdout']
    root = tk.Tk()
    app = ReviewApp(root, task_dirs)
    if args.smoke_test:
        root.after(900, root.destroy)
    root.mainloop()
    if args.smoke_test:
        print(f'EDGE_COUNT_REVIEW_SMOKE_OK items={len(app.items)}')


if __name__ == '__main__':
    main()
