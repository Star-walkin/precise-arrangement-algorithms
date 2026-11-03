"""Standalone offline inspection UI. Never imports camera/control software."""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import queue
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .geometry import analyze_frame, draw_overlay, dump_json, read_image, write_image


INITIAL_STATUS_TEXT = ('Ctrl+滚轮缩放；右键拖动；F2显隐几何标注。默认只自动生成候选，'
                       '不自动确认绳身份；困难画面可限定搜索框或用SAM2辅助。')
OVERLAY_LEGEND_TEXT = ('绿：有像素支持的局部边界（不跨缺口）；淡蓝虚线：可选椭圆模型；'
                       '白点：原始候选；橙/青：自由绳延长线/可见段；黄：参考光学宽度；'
                       '编号紫点：人工搜索提示，不是绳体中心线或实体边真值；'
                       '紫色虚线：新入卷候选，身份未确认。')
INVALIDATED_STATUS_TEXT = '当前画面、搜索框、提示点或参数已变化；旧结果已失效，请重新检测后再保存。'
WOUND_GUIDANCE_SEMANTICS = 'search hints only; not verified rope centers or edges'
WOUND_REASON_TRANSLATIONS = {
    'repetitive_parallel_rope_texture': '相邻绳纹重复',
    'saturated_or_clipped_ridge_identity': '高光饱和',
    'gaps_in_continuity': '存在断续',
    'multiple_connected_ridge_hypotheses': '多条连通路径',
    'no_resolved_post_entry_bending': '未识别可靠弯曲',
    'manual_seeds_not_all_image_supported': '部分提示点附近缺少像素支持',
    'boundary_ridge_cannot_be_independently_identified': '与边界纹理身份混淆',
    'no_ridge_continuously_anchored_to_free_rope': '没有与自由绳连续相连的纹理',
    'independent_free_rope_unavailable': '自由绳不可用',
    'entry_direction_requires_observed_arc_or_seed': '缺少入卷方向提示',
    'optical_width_scale_unavailable': '缺少纹理搜索尺度',
    'insufficient_continuous_ridge_support': '连续纹理证据不足',
    'guided_body_center_vs_edge_unresolved': '人工提示仅限定搜索，绳体中心与实体边缘仍未区分',
}


def _wound_guidance(points):
    points = [list(map(float, point)) for point in (points or [])]
    return dict(used=bool(points),
                kind='manual_search_points' if points else 'none',
                points=points,
                semantics=WOUND_GUIDANCE_SEMANTICS)


def compose_display_image(frame, result, display_settings, *, arc_roi=None,
                          rope_roi=None, prompts=None, prompt_labels=None,
                          wound_prompts=None):
    """Render the saved UI view in original pixels, including interaction marks.

    Geometry switches reproduce the current display. Search boxes and prompt
    points remain visible even when the F2 geometry overlay is off, matching the
    live canvas. They are search guidance, never measured geometry.
    """
    settings = dict(display_settings or {})
    if settings.get('overlay', True):
        shown = draw_overlay(
            frame, result,
            show_arc=bool(settings.get('show_arc', True)),
            show_rope=bool(settings.get('show_rope', True)),
            show_raw=bool(settings.get('show_raw', False)),
            show_exclusion=bool(settings.get('show_exclusion', False)),
            show_conic=bool(settings.get('show_conic', False)),
            show_wound=bool(settings.get('show_wound', True)))
    else:
        shown = frame.copy()
    scale = max(1, round(max(frame.shape[:2]) / 1500))
    for roi, color in ((arc_roi, (121, 232, 79)), (rope_roi, (68, 170, 255))):
        if roi is not None and len(roi) == 4 and np.isfinite(roi).all():
            a = tuple(np.rint(roi[:2]).astype(int))
            b = tuple(np.rint(roi[2:]).astype(int))
            cv2.rectangle(shown, a, b, color, 2 * scale, cv2.LINE_AA)
    for point, label in zip(prompts or [], prompt_labels or []):
        p = tuple(np.rint(point).astype(int))
        color = (98, 255, 83) if label else (69, 69, 255)
        cv2.circle(shown, p, 5 * scale, color, -1, cv2.LINE_AA)
        cv2.circle(shown, p, 5 * scale, (255, 255, 255), scale, cv2.LINE_AA)
    for index, point in enumerate(wound_prompts or []):
        p = tuple(np.rint(point).astype(int))
        cv2.circle(shown, p, 6 * scale, (255, 99, 237), -1, cv2.LINE_AA)
        cv2.circle(shown, p, 6 * scale, (255, 255, 255), scale, cv2.LINE_AA)
        cv2.putText(shown, str(index + 1), (p[0] + 9 * scale, p[1] - 7 * scale),
                    cv2.FONT_HERSHEY_SIMPLEX, .55 * scale, (255, 99, 237),
                    max(1, scale), cv2.LINE_AA)
    return shown


class ZoomCanvas(tk.Canvas):
    def __init__(self, parent, app):
        super().__init__(parent, bg='#151b24', highlightthickness=0)
        self.app = app
        self.source = None
        self.photo = None
        self.zoom = 1.0
        self.scale = 1.0
        self.drag_start = None
        self.bind('<Configure>', lambda e: self.render())
        self.bind('<Control-MouseWheel>', self.wheel)
        self.bind('<ButtonPress-3>', lambda e: self.scan_mark(e.x, e.y))
        self.bind('<B3-Motion>', lambda e: self.scan_dragto(e.x, e.y, gain=1))
        self.bind('<ButtonPress-1>', self.left_down)
        self.bind('<B1-Motion>', self.left_move)
        self.bind('<ButtonRelease-1>', self.left_up)

    def point(self, event):
        return [self.canvasx(event.x)/self.scale, self.canvasy(event.y)/self.scale]

    def left_down(self, event):
        if self.source is None:
            return
        point = self.point(event)
        if not (0 <= point[0] < self.source.width and 0 <= point[1] < self.source.height):
            return
        mode = self.app.mode.get()
        if mode == '入卷轨迹提示点':
            if self.app.busy or self.app.playing:
                return
            self.app.wound_prompts.append(point)
            self.app.invalidate()
            self.app.render()
        elif mode == '活动绳提示点':
            if self.app.busy or self.app.playing:
                return
            self.app.prompts.append(point)
            self.app.labels.append(0 if event.state & 0x0001 else 1)
            self.app.invalidate()
            self.app.render()
        elif mode in ('弧线搜索框', '活动绳搜索框'):
            self.drag_start = point

    def left_move(self, event):
        if self.drag_start is None:
            return
        self.delete('selection')
        a, b = self.drag_start, self.point(event)
        self.create_rectangle(a[0]*self.scale, a[1]*self.scale,
                              b[0]*self.scale, b[1]*self.scale,
                              outline='#5fcfff', width=2, tags='selection')

    def left_up(self, event):
        if self.drag_start is None:
            return
        a, b = self.drag_start, self.point(event)
        self.drag_start = None
        self.delete('selection')
        if self.app.busy or self.app.playing:
            return
        w, h = self.source.size
        roi = [max(0, min(a[0], b[0])), max(0, min(a[1], b[1])),
               min(w, max(a[0], b[0])), min(h, max(a[1], b[1]))]
        if roi[2]-roi[0] < 16 or roi[3]-roi[1] < 16:
            return
        if self.app.mode.get() == '弧线搜索框':
            self.app.arc_roi = roi
        else:
            self.app.rope_roi = roi
        self.app.invalidate()
        self.app.render()

    def wheel(self, event):
        if self.source is None:
            return 'break'
        x, y = self.point(event)
        self.zoom = min(12, max(1, self.zoom * (1.25 if event.delta > 0 else .8)))
        self.render()
        rw, rh = self.source.width*self.scale, self.source.height*self.scale
        self.xview_moveto(max(0, (x*self.scale-event.x)/max(1, rw)))
        self.yview_moveto(max(0, (y*self.scale-event.y)/max(1, rh)))
        return 'break'

    def render(self):
        if self.source is None:
            return
        cw, ch = max(1, self.winfo_width()), max(1, self.winfo_height())
        fit = min(cw/self.source.width, ch/self.source.height)
        self.scale = fit*self.zoom
        size = (max(1, round(self.source.width*self.scale)), max(1, round(self.source.height*self.scale)))
        self.photo = ImageTk.PhotoImage(self.source.resize(size, Image.Resampling.BILINEAR))
        self.delete('all')
        self.create_image(0, 0, image=self.photo, anchor='nw')
        self.configure(scrollregion=(0, 0, size[0], size[1]))
        for roi, color in [(self.app.arc_roi, '#4fe879'), (self.app.rope_roi, '#ffaa44')]:
            if roi:
                self.create_rectangle(*[v*self.scale for v in roi], outline=color, width=2)
        for point, label in zip(self.app.prompts, self.app.labels):
            x, y = [v*self.scale for v in point]
            self.create_oval(x-4, y-4, x+4, y+4, fill='#53ff62' if label else '#ff4545', outline='white')
        for index,point in enumerate(self.app.wound_prompts):
            x,y=[v*self.scale for v in point]
            self.create_oval(x-4,y-4,x+4,y+4,fill='#ed63ff',outline='white')
            self.create_text(x+8,y-7,text=str(index+1),fill='#ed63ff',anchor='w')


class App:
    def __init__(self, root):
        self.root = root
        root.title('独立几何实验 V3｜局部边界 + 新入卷绳轨迹候选｜仅离线，不控制设备')
        root.geometry('1380x880')
        root.minsize(960, 650)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.queue = queue.Queue()
        self.busy = False
        self.closed = False
        self.version = 0
        self.playing = False
        self.cap = None
        self.path = None
        self.frame = None
        self.index = 0
        self.fps = 0
        self.total = 0
        self.result = None
        self.arc_roi = self.rope_roi = None
        self.prompts, self.labels = [], []
        self.wound_prompts=[]
        self.mode = tk.StringVar(value='查看／拖动')
        self.overlay = tk.BooleanVar(value=True)
        self.show_arc = tk.BooleanVar(value=True)
        self.show_rope = tk.BooleanVar(value=True)
        self.show_raw = tk.BooleanVar(value=False)
        self.show_exclusion = tk.BooleanVar(value=False)
        self.show_conic = tk.BooleanVar(value=False)
        self.show_wound = tk.BooleanVar(value=True)
        self.exclusion_margin = tk.StringVar(value='8')
        self.seek_s = tk.StringVar(value='0')
        self.source_text = tk.StringVar(value='请打开图片或视频。两侧相机均可使用，无需指定相机编号。')
        self.arc_text = tk.StringVar(value='弧线：等待图像')
        self.rope_text = tk.StringVar(value='活动绳：等待图像')
        self.contact_text = tk.StringVar(value='相切：仅图像几何，不能确认新层')
        self.wound_text = tk.StringVar(value='新入卷绳轨迹：等待图像；人工提示仅限定搜索，不是中心线或实体边真值。')
        self.status = tk.StringVar(value=INITIAL_STATUS_TEXT)
        self._build()
        self.exclusion_margin.trace_add('write',self.parameter_changed)
        root.bind('<F2>', lambda e: self.toggle())
        root.bind('<Control-Key-0>', lambda e: self.fit())
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(50, self.poll)

    def _build(self):
        ttk.Label(self.root, text='独立测量实验  |  像素几何，不输出根数／新层确认／伺服指令',
                  anchor='center', padding=8, background='#822f2f', foreground='white').pack(fill='x')
        bar = ttk.Frame(self.root, padding=6)
        bar.pack(fill='x')
        for label, command in [('打开图片／视频', self.open), ('重新检测', self.analyze),
                               ('逐帧分析播放', self.play), ('暂停', self.pause),
                               ('上一帧', lambda: self.step(-1)), ('下一帧', lambda: self.step(1)),
                               ('保存本帧结果', self.save)]:
            ttk.Button(bar, text=label, command=command).pack(side='left', padx=3)
        ttk.Label(bar, text='跳转秒：').pack(side='left')
        ttk.Entry(bar, textvariable=self.seek_s, width=8).pack(side='left')
        ttk.Button(bar, text='跳转', command=self.seek).pack(side='left', padx=3)
        ttk.Label(self.root, textvariable=self.source_text, padding=(9, 2)).pack(fill='x')
        tools = ttk.Frame(self.root, padding=6)
        tools.pack(fill='x')
        ttk.Combobox(tools, textvariable=self.mode, state='readonly', width=17,
                     values=['查看／拖动', '弧线搜索框', '活动绳搜索框', '活动绳提示点','入卷轨迹提示点']).pack(side='left')
        ttk.Button(tools, text='SAM2辅助本帧', command=lambda: self.analyze(use_sam=True)).pack(side='left', padx=4)
        ttk.Button(tools, text='清除搜索框／提示', command=self.clear).pack(side='left', padx=4)
        ttk.Checkbutton(tools, text='标注 F2', variable=self.overlay, command=self.render).pack(side='left', padx=4)
        ttk.Checkbutton(tools, text='局部边界', variable=self.show_arc, command=self.render).pack(side='left')
        ttk.Checkbutton(tools, text='原始边界点', variable=self.show_raw, command=self.render).pack(side='left')
        ttk.Checkbutton(tools, text='绳屏蔽带', variable=self.show_exclusion, command=self.render).pack(side='left')
        ttk.Checkbutton(tools, text='活动绳及绳宽', variable=self.show_rope, command=self.render).pack(side='left')
        ttk.Button(tools, text='适应窗口', command=self.fit).pack(side='left', padx=4)
        ttk.Label(tools,text='屏蔽余量px：').pack(side='left')
        ttk.Spinbox(tools,from_=0,to=80,increment=1,textvariable=self.exclusion_margin,width=4).pack(side='left')
        research=ttk.Frame(self.root,padding=(6,0,6,3))
        research.pack(fill='x')
        ttk.Checkbutton(research,text='完整椭圆模型（可失败，不影响局部边界）',variable=self.show_conic,command=self.render).pack(side='left')
        ttk.Checkbutton(research,text='新入卷轨迹候选（不等于已确认活动绳）',variable=self.show_wound,command=self.render).pack(side='left',padx=12)
        ttk.Button(research,text='导入本帧轨迹提示',command=self.import_wound_reference).pack(side='left',padx=4)
        info = ttk.Frame(self.root, padding=6)
        info.pack(fill='x')
        for i, (title, var) in enumerate([('内侧弧线', self.arc_text), ('活动绳直线', self.rope_text), ('几何关系', self.contact_text)]):
            info.columnconfigure(i, weight=1, uniform='cards')
            card = ttk.LabelFrame(info, text=title, padding=6)
            card.grid(row=0, column=i, sticky='nsew', padx=3)
            ttk.Label(card, textvariable=var, width=40, wraplength=420, anchor='nw').pack(fill='both')
        ttk.Label(self.root,textvariable=self.wound_text,padding=(10,3),wraplength=1340).pack(fill='x')
        self.canvas = ZoomCanvas(self.root, self)
        self.canvas.pack(fill='both', expand=True, padx=8, pady=5)
        ttk.Label(self.root, text=OVERLAY_LEGEND_TEXT, padding=5,wraplength=1340).pack(fill='x')
        ttk.Label(self.root, textvariable=self.status, wraplength=1340, padding=6).pack(fill='x')

    def invalidate(self):
        self.version += 1
        self.result = None
        self.arc_text.set('弧线：结果已失效，等待重新检测')
        self.rope_text.set('活动绳：结果已失效，等待重新检测')
        self.contact_text.set('新层：不判断；本帧结果尚未生成')
        if hasattr(self,'wound_text'):
            self.wound_text.set('新入卷绳轨迹：结果已失效，等待重新检测')
        if hasattr(self, 'status'):
            self.status.set(INVALIDATED_STATUS_TEXT)

    def parameter_changed(self,*args):
        self.pause()
        self.invalidate()
        self.render()
        self.status.set('屏蔽余量已变化，请点击重新检测。旧结果已失效，不能保存为新参数结果。')

    def import_wound_reference(self,path=None):
        if self.frame is None or self.busy:
            return
        path=path or filedialog.askopenfilename(title='导入本帧轨迹参考JSON（校验原图指纹）',
                initialdir=str(Path(__file__).parent/'outputs/wound_curve_v3/manual_247'),
                filetypes=[('轨迹参考','*.json')])
        if not path:
            return
        try:
            data=json.loads(Path(path).read_text(encoding='utf-8-sig'))
            if not isinstance(data, dict):
                raise ValueError('参考JSON顶层必须是对象。')
            digest=hashlib.sha256(np.ascontiguousarray(self.frame).tobytes()).hexdigest()
            if data.get('original_frame_sha256') != digest:
                raise ValueError('参考点不属于当前原始画面。请先打开同一视频，并跳转到参考对应帧。')
            points=np.asarray(data.get('points_original_px'),float)
            if points.ndim!=2 or points.shape[1:]!=(2,) or not 1<=len(points)<=64 or not np.isfinite(points).all():
                raise ValueError('参考点格式无效，应为1～64个原图[x,y]坐标。')
            if np.any(points<0) or np.any(points[:,0]>=self.frame.shape[1]) or np.any(points[:,1]>=self.frame.shape[0]):
                raise ValueError('参考点超出原图范围。')
        except (OSError,ValueError,TypeError) as exc:
            messagebox.showerror('无法导入提示',str(exc),parent=self.root)
            return
        self.pause()
        self.wound_prompts=points.tolist()
        self.invalidate()
        self.render()
        self.analyze()

    def open(self, path=None):
        if self.busy:
            return
        path = path or filedialog.askopenfilename(title='选择原始图片或视频',
            initialdir=str(Path(__file__).resolve().parents[1]/'现场图片'),
            filetypes=[('图像与视频', '*.bmp *.png *.jpg *.jpeg *.avi *.mp4 *.mkv'), ('所有文件', '*.*')])
        if not path:
            return
        self.pause()
        self.invalidate()
        if self.cap:
            self.cap.release()
        self.cap = None
        self.path = Path(path)
        self.frame = None
        self.index, self.fps, self.total = 0, 0, 0
        self.canvas.source = None
        self.canvas.delete('all')
        self.source_text.set(str(self.path)+'  正在读取……')
        self.arc_roi = self.rope_roi = None
        self.prompts, self.labels = [], []
        self.wound_prompts=[]
        self.canvas.zoom = 1.0
        try:
            if self.path.suffix.lower() in ('.avi', '.mp4', '.mkv', '.mov'):
                self.cap = cv2.VideoCapture(str(path))
                if not self.cap.isOpened():
                    raise ValueError('视频无法打开')
                self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
                self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.read_frame(0)
            else:
                self.frame = read_image(path)
                self.index, self.fps, self.total = 0, 0, 1
                self.show_source()
                self.render()
                self.analyze()
        except Exception as exc:
            self.frame = None
            self.canvas.source = None
            self.canvas.delete('all')
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.status.set('读取失败，未显示旧文件结果：'+str(exc))
            messagebox.showerror('读取失败', str(exc), parent=self.root)

    def show_source(self):
        suffix = f'  {self.index/max(self.fps, 1):.3f} s  第{self.index+1}/{self.total}帧' if self.cap else ''
        self.source_text.set(str(self.path)+suffix)

    def read_frame(self, index):
        if self.busy or self.cap is None:
            return
        index = max(0, min(self.total-1, int(index)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            self.pause()
            self.invalidate()
            self.frame = None
            self.index = index
            self.canvas.source = None
            self.canvas.delete('all')
            self.show_source()
            self.status.set('该帧读取失败，未更新测量。')
            return
        self.frame, self.index = frame, index
        self.prompts, self.labels = [], []  # never carry a stale mask/prompt across frames
        self.wound_prompts=[]
        self.invalidate()
        self.show_source()
        self.render()
        self.analyze()

    def step(self, delta):
        self.pause()
        self.read_frame(self.index+delta)

    def seek(self):
        try:
            seconds = float(self.seek_s.get())
            if not np.isfinite(seconds):
                raise ValueError()
        except ValueError:
            messagebox.showerror('秒数错误', '请输入有效秒数。', parent=self.root)
            return
        self.pause()
        self.read_frame(round(seconds*self.fps))

    def play(self):
        if self.cap is None or self.busy:
            return
        self.playing = True
        self.read_frame(self.index+1)

    def pause(self):
        self.playing = False

    def clear(self):
        if self.busy:
            return
        self.pause()
        self.arc_roi = self.rope_roi = None
        self.prompts, self.labels = [], []
        self.wound_prompts=[]
        self.invalidate()
        self.render()
        self.analyze()

    def analyze(self, use_sam=False):
        if self.frame is None or self.busy:
            return
        if use_sam:
            self.pause()
            if not any(self.labels):
                messagebox.showinfo('先指定活动绳', '选择“活动绳提示点”，在清楚可见的自由绳段左键点1～3点。\nShift+左键可点不属于绳的背景。\n建议先拖出仅包含可见自由段的“活动绳搜索框”。', parent=self.root)
                return
        try:
            margin=float(self.exclusion_margin.get()) if hasattr(self,'exclusion_margin') else 8.0
            if not np.isfinite(margin) or not 0<=margin<=80:
                raise ValueError()
        except (ValueError,TypeError):
            messagebox.showerror('屏蔽余量错误','请输入0～80之间的原图像素值。它是在半绳宽及定位裕量以外附加的屏蔽宽度。',parent=self.root)
            return
        self.invalidate()
        self.render()
        self.busy = True
        version = self.version
        frame = self.frame.copy()
        arc_roi, rope_roi = self.arc_roi, self.rope_roi
        points, labels = list(self.prompts), list(self.labels)
        wound_points=list(getattr(self,'wound_prompts',[]))
        source, index, fps = str(self.path), self.index, self.fps
        self.status.set('SAM2正在CPU/GPU上辅助定位本帧；不代表实时速度……' if use_sam else '正在测量本帧弧线与活动绳……')

        def work():
            try:
                mask, metadata = None, None
                start = time.perf_counter()
                if use_sam:
                    from .sam_bridge import sam_rope_proposal
                    mask, metadata = sam_rope_proposal(frame, points, labels, rope_roi)
                positive = [p for p, label in zip(points, labels) if label]
                result = analyze_frame(frame, arc_roi=arc_roi, rope_roi=rope_roi,
                    rope_mask=mask, seed_points=positive or None,exclusion_margin_px=margin,
                    wound_seed_points=wound_points or None)
                guidance=result.get('wound_guidance')
                if not isinstance(guidance,dict):
                    guidance=_wound_guidance(wound_points)
                result.update(source=source, frame_index=index,
                              video_time_s=index/fps if fps else None,
                              prompts=points, prompt_labels=labels, arc_roi=arc_roi,
                              rope_roi=rope_roi, sam2=metadata,
                              wound_prompts=wound_points,
                              wound_guidance=guidance,
                              full_request_ms=(time.perf_counter()-start)*1000)
                self.queue.put((version, result, None))
            except Exception as exc:
                self.queue.put((version, None, str(exc)))
        self.executor.submit(work)

    def poll(self):
        if self.closed:
            return
        try:
            version, result, error = self.queue.get_nowait()
            self.busy = False
            if version == self.version:
                if error:
                    self.pause()
                    self.status.set('检测失败：'+error)
                else:
                    self.result = result
                    self.update_labels()
                    self.render()
            if self.playing:
                if self.index+1 >= self.total:
                    self.pause()
                else:
                    self.root.after(10, lambda: self.read_frame(self.index+1) if self.playing else None)
        except queue.Empty:
            pass
        self.root.after(50, self.poll)

    def update_labels(self):
        arc, rope, relation = [self.result[k] for k in ('arc', 'rope', 'relation')]
        conic=self.result.get('conic',{})
        exclusion=self.result.get('exclusion',{})
        if conic.get('valid'):
            quality='（模型不确定性较大）' if conic.get('status')=='uncertain_model' else ''
            self.arc_text.set(f"已拟合椭圆弧{quality}／原始{len(arc['points'])}点\n残差 {conic.get('residual_px',0):.2f} px；敏感性 {conic.get('sensitivity_px',0):.2f} px（非精度）")
        else:
            self.arc_text.set('椭圆弧不可用：'+conic.get('reason',arc.get('reason',''))+'\n可开启原始边界点诊断')
        if 'local_boundary_segments' in arc:
            segments=arc.get('local_boundary_segments',[])
            if arc.get('local_boundary_valid'):
                description=f"局部边界可用：{len(segments)}段（绿线不跨缺口）"
            else:
                description='局部边界证据不足'
            if conic.get('valid'):
                description+=f"\n椭圆仅模型；残差{conic.get('residual_px',0):.2f}px／敏感性{conic.get('sensitivity_px',0):.2f}px"
            else:
                reasons={'partial_ellipse_is_unstable_to_support_removal':'支持变化时椭圆不稳定',
                         'observed_arc_too_short_for_ellipse':'局部弧不足以确定完整椭圆',
                         'insufficient_observed_angular_blocks':'角度覆盖不足',
                         'ellipse_inconsistent_with_observed_shape':'各段不符合统一椭圆'}
                description+='\n完整椭圆未通过：'+reasons.get(conic.get('reason'),conic.get('reason','不可用'))
            self.arc_text.set(description)
        interval=rope.get('width_interval_px')
        width_text=(f"光学宽度 {rope['width_px']:.2f} px（非标定绳径）" if rope.get('width_px') else
                    (f"宽度不确定：约{interval[0]:.1f}～{interval[1]:.1f} px，黄虚线" if interval else '绳宽不可用，不绘制伪造的边线'))
        if rope.get('width_px') and interval:
            width_text+=f"\n观测变化范围{interval[0]:.1f}～{interval[1]:.1f}px；接触处为外推"
        self.rope_text.set((f"角度 {rope['angle_deg']:.3f}°  偏移ρ {rope['rho_px']:.2f} px\n"
                            +width_text
                            if rope.get('valid') else '不可用：'+rope.get('reason', '')))
        statuses={'center_near_tangent_candidate':'中心线近切候选／最后一圈待验证',
                  'optical_edge_near_tangent_candidate':'光学绳边近切候选／最后一圈待验证',
                  'line_outside_conic':'中心线在弧外，未接近相切',
                  'line_crosses_conic':'中心线与椭圆相交，未接近相切',
                  'uncertainty_too_large':'拟合或绳线不确定性过大，不能判近切'}
        if relation.get('valid'):
            chord=relation.get('chord_length_px')
            chord_text=f"交点间距 {chord:.1f}px" if chord is not None else '无实交点'
            text=statuses.get(relation['status'],relation['status'])
            if relation.get('optical_band_tangency_possible') and not relation.get('optical_band_tangency_candidate'):
                text+='；光学绳边可能近切但宽度证据不足'
            self.contact_text.set(text+
                f"\n全椭圆{relation['complete_intersection_count']}交点／当前弧{relation['visible_arc_intersection_count']}；{chord_text}"+
                f"\n法向间隙 {relation['signed_center_clearance_px']:.2f}px；诊断容差 {relation['diagnostic_tolerance_px']:.2f}px")
        else:
            self.contact_text.set('无法测量：'+relation.get('reason',''))
        wound=self.result.get('wound_curve',{})
        if hasattr(self,'wound_text'):
            statuses={'unavailable':'未取得可用轨迹',
                      'identity_ambiguous':'重复纹理/路径歧义，尚未确认绳身份',
                      'curve_candidate_unconfirmed':'取得弯曲候选，尚未确认绳身份',
                      'straight_continuation_only':'仅找到直线延续，入卷曲线未证实'}
            status=statuses.get(wound.get('status'),wound.get('status','尚未分析'))
            count=len(wound.get('candidates',[]))
            mode='人工提示模式' if self.result.get('wound_prompts') else '自动候选模式'
            reasons=[WOUND_REASON_TRANSLATIONS.get(item.strip(),item.strip()) for item in wound.get('reason','').split(';')]
            detail='／'.join(reasons[:3])
            self.wound_text.set(f'新入卷绳轨迹｜{mode}｜{status}（{count}条候选）｜'+detail+
                                '；不确认贴边/最后一圈/新层。')
        mask_text=f"绳屏蔽半宽{exclusion['half_width_px']:.1f}px；" if exclusion.get('valid') else '无可靠绳屏蔽带；'
        self.status.set(mask_text+f"本帧几何 {self.result['timing_ms']['total']:.0f} ms；总耗时 {self.result['full_request_ms']:.0f} ms。交点合并只显示研究候选，不确认最后一圈、新层或施力。")

    def render(self):
        if self.frame is None:
            return
        shown = draw_overlay(self.frame, self.result,
            show_arc=self.show_arc.get(), show_rope=self.show_rope.get(),
            show_raw=self.show_raw.get(),show_exclusion=self.show_exclusion.get(),
            show_conic=self.show_conic.get(),show_wound=self.show_wound.get()) if self.overlay.get() else self.frame
        self.canvas.source = Image.fromarray(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB))
        self.canvas.render()

    def toggle(self):
        self.overlay.set(not self.overlay.get())
        self.render()

    def fit(self):
        self.canvas.zoom = 1
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self.canvas.render()

    def display_settings(self):
        defaults = dict(overlay=True, show_arc=True, show_rope=True, show_raw=False,
                        show_exclusion=False, show_conic=False, show_wound=True)
        settings = {}
        for name, default in defaults.items():
            variable = getattr(self, name, None)
            try:
                settings[name] = bool(variable.get())
            except (AttributeError, TypeError, tk.TclError):
                settings[name] = default
        return settings

    def save(self):
        if self.result is None or self.busy:
            return
        folder = Path(__file__).parent/'outputs'/'manual'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        pending=folder.with_name(folder.name+'.incomplete')
        try:
            settings=self.display_settings()
            display=compose_display_image(
                self.frame,self.result,settings,
                arc_roi=getattr(self,'arc_roi',None),rope_roi=getattr(self,'rope_roi',None),
                prompts=getattr(self,'prompts',[]),prompt_labels=getattr(self,'labels',[]),
                wound_prompts=getattr(self,'wound_prompts',[]))
            write_image(pending/'display.png',display)
            write_image(pending/'overlay.png',draw_overlay(
                self.frame,self.result,show_arc=True,show_rope=True,show_raw=False,
                show_exclusion=False,show_conic=False,show_wound=True))
            write_image(pending/'diagnostics.png',draw_overlay(
                self.frame,self.result,show_arc=True,show_rope=True,show_raw=True,
                show_exclusion=True,show_conic=True,show_wound=True))
            write_image(pending/'source.png', self.frame)
            guidance=self.result.get('wound_guidance')
            if not isinstance(guidance,dict):
                guidance=_wound_guidance(self.result.get('wound_prompts',getattr(self,'wound_prompts',[])))
            manifest=dict(self.result,bundle_complete=True,display_settings=settings,
                          wound_guidance=guidance,
                          artifact_semantics={
                              'display.png':'current UI switches plus search boxes and prompt points',
                              'overlay.png':'canonical local-boundary/free-rope/wound-candidate overlay; optional conic and raw diagnostics off',
                              'diagnostics.png':'canonical full diagnostic overlay; all geometry layers on',
                              'source.png':'clean source pixels; no annotations'})
            dump_json(pending/'measurement.json',manifest)
            pending.rename(folder)
        except Exception as exc:
            self.status.set('保存失败，未完成文件仅保留在.incomplete目录：'+str(exc))
            messagebox.showerror('保存失败',str(exc),parent=self.root)
            return
        self.status.set('已保存界面快照、规范叠加图、诊断图、原图和测量JSON：'+str(folder))

    def close(self):
        self.closed = True
        self.playing = False
        if self.cap:
            self.cap.release()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', nargs='?')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    cv2.setNumThreads(2)
    root = tk.Tk()
    if args.smoke_test:
        root.withdraw()
    app = App(root)
    if args.source:
        root.after(150, lambda: app.open(args.source))
    if args.smoke_test:
        root.after(300, app.close)
    root.mainloop()


if __name__ == '__main__':
    main()
