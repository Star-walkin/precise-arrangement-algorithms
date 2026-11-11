"""LC1 causal locked-column runner with explicit detector provenance.

The `traditional-preview` backend is a separately named LC1-T preview. It is
never reported as P1/H6/S1. The external backend accepts actual frame-indexed
P1/H6 and S1 observations; absent observations cannot authorize calibration.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT.parent
cv2.setNumThreads(2)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def dump(path, obj):
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def imread(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(path, dtype=np.uint8)
    result = cv2.imdecode(data, flags)
    if result is None:
        raise ValueError(f'Cannot decode {path}')
    return result


def original_points(points, width, mirror):
    if points is None:
        return np.empty((0,2),float)
    q = np.asarray(points, dtype=float).reshape(-1, 2).copy()
    if not np.isfinite(q).all():
        raise ValueError('Non-finite point coordinates are not image observations')
    if mirror:
        q[:, 0] = width - 1 - q[:, 0]
    return q


def original_abc(abc, width, mirror):
    if abc is None:
        return None
    a, b, c = map(float, abc)
    if not all(math.isfinite(v) for v in (a,b,c)):
        return None
    if mirror:
        a, c = -a, c + a * (width - 1)
    if b < 0:
        a, b, c = -a, -b, -c
    norm=math.hypot(a,b)
    if not math.isfinite(norm) or norm<1e-9:
        return None
    return [a/norm, b/norm, c/norm]


def y_at(abc, x):
    if abc is None or x is None or abs(float(abc[1])) < .05:
        return None
    return float(-(abc[0] * x + abc[2]) / abc[1])


def calibration(session, camera):
    stem = f'{session}__pair0015__{camera}'
    mask_path = (ASSETS/'camera_condition_sam2_20261009_v1'/'all_001'/'body_branches_002'
                 /'masks'/f'{camera}_camera'/f'{stem}.png')
    record_path = (ASSETS/'camera_condition_bright_rope_boundaries_20261023_v2'/'all_002'
                   /'boundaries'/'records'/f'{camera}_camera'/f'{stem}.json')
    record = json.loads(record_path.read_text(encoding='utf-8'))['record']
    mask = imread(mask_path, cv2.IMREAD_GRAYSCALE)
    pitch = float(record['pitch_px'])
    return mask, pitch, dict(
        scene_roi_source='previously_open_development_image_static_SAM2_search_domain',
        target_paths_or_saved_line_coordinates_used=False,
        roi_is_not_a_fresh_per_frame_segmentation=True,
        scene_calibration_is_offline=True,
        mask_path=str(mask_path), mask_sha256=sha(mask_path),
        pitch_reference_path=str(record_path), pitch_original_px=pitch,
        pitch_is_uncalibrated_image_spacing=True,
    )


def timestamps(video, camera, count, fps):
    session_path=video.parent/'session.json'
    if session_path.exists():
        metadata=json.loads(session_path.read_text(encoding='utf-8-sig'))
        expected=metadata.get('frames',{}).get(camera)
        if expected is not None and int(expected)!=count:
            raise ValueError(f'Source AVI {count} frames disagrees with session {expected}; download may be incomplete')
    path = video.parent / f'{camera}_frames.csv'
    if path.exists():
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows)!=count:
            raise ValueError('Frame CSV and source AVI disagree; refusing nominal-FPS fallback')
        indices=[int(r['video_frame']) for r in rows]
        times=[float(r['host_time_s']) for r in rows]
        if indices!=list(range(count)):
            raise ValueError('Frame CSV video_frame indices are not unique continuous 0..N-1')
        if not np.isfinite(times).all() or np.any(np.diff(times)<=0):
            raise ValueError('Frame CSV host times are not finite/strictly increasing')
        return times, 'shared_PC_host_receipt_time_not_exposure', str(path)
    return [i/fps for i in range(count)], 'AVI_frame_number_over_nominal_fps', None


def make_lock(max_age):
    from lc1_geometry import ColumnLock
    signature = inspect.signature(ColumnLock)
    params = {}
    if 'max_unverified_lock_age_s' in signature.parameters:
        params['max_unverified_lock_age_s'] = max_age
    return ColumnLock(**params)


def is_finite_number(value):
    return value is not None and np.isscalar(value) and math.isfinite(float(value))


class ExternalFrontend:
    """Actual module outputs, never synthetic or substituted target paths."""
    def __init__(self, path, width, mirror, *, source_sha256=None, camera=None):
        self.width, self.mirror = width, mirror
        opener = gzip.open if str(path).endswith('.gz') else open
        with opener(path, 'rt', encoding='utf-8-sig') as stream:
            data=[json.loads(l) for l in stream if l.strip()]
        ids=[r['frame_index'] for r in data]
        if any(type(i) is not int or i<0 for i in ids):
            raise ValueError('External frame index must be a nonnegative JSON integer')
        if len(ids)!=len(set(ids)) or any(i<0 for i in ids):
            raise ValueError('Duplicate or negative external frame index')
        if source_sha256 is not None and any(r.get('video_sha256')!=source_sha256 for r in data):
            raise ValueError('External observations are not bound to this source video SHA-256')
        if camera is not None and any(r.get('camera_role')!=camera for r in data):
            raise ValueError('External observations camera_role disagrees with the source stream')
        self.rows=dict(zip(ids,data))

    def step(self, bgr, t, frame_index):
        row = self.rows.get(frame_index)
        if not row:
            return dict(line_abc=None, path_xy=[], whole_path_candidate_supported=False,
                        source='missing_external_observation', diagnostics={})
        if row.get('coordinate_system') != 'original_unmirrored_xy':
            raise ValueError('External observations must declare original_unmirrored_xy')
        ph, s1 = row.get('P1_H6', {}), row.get('S1', {})
        path = original_points(ph.get('path_xy', []), self.width, self.mirror)
        s1_source = s1.get('source')
        s1_valid = isinstance(s1_source,str) and bool(s1_source.strip()) and s1.get('valid',True) is True
        abc = original_abc(s1.get('abc'), self.width, self.mirror) if s1_valid else None
        ph_source = ph.get('source')
        supported = bool(ph.get('complete_supported') is True and
                         isinstance(ph_source,str) and ph_source.strip() and s1_valid and len(path)>=8)
        measured=original_points(s1.get('points_xy',[]),self.width,self.mirror) if s1_valid else np.empty((0,2))
        return dict(line_abc=abc, line_points_xy=measured, path_xy=path,
                    whole_path_candidate_supported=supported,
                    source=f"external:{ph.get('source', 'missing P1/H6')} + {s1.get('source','missing S1')}",
                    diagnostics=dict(actual_source_record=row, identity_is_not_ground_truth=True))


class Overlay:
    def __init__(self, width, height, camera, backend, output_width, reference_column=None):
        self.w, self.h, self.camera, self.backend = width, height, camera, backend
        self.mirror = camera == 'left'
        self.reference_column=reference_column
        self.ow = int(output_width)//2*2
        self.oh = int(round(height*self.ow/width))//2*2
        self.scale = np.array([self.ow/width, self.oh/height])
        self.font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 23)
        self.small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 19)

    def points(self, p):
        q = original_points(p, self.w, self.mirror)
        return np.rint(q*self.scale).astype(np.int32)

    def draw(self, original, frame_index, t, front, turn, lock, surface, eval_x, eval_y, processing_ms):
        frame = cv2.resize(original, (self.ow, self.oh), interpolation=cv2.INTER_AREA)
        # Fresh image-supported observations are yellow; fitted proposals are
        # deliberately not drawn as if they were observed physical boundaries.
        for line in surface.get('boundary_polylines_xy', []):
            q = self.points(line)
            if len(q)>1:
                cv2.polylines(frame,[q],False,(0,220,255),1,cv2.LINE_AA)
        q = self.points(front.get('path_xy', []))
        supported_segments=front.get('diagnostics',{}).get('path_segment_support')
        if len(q)>1:
            if supported_segments is not None and len(supported_segments)==len(q)-1:
                for k, supported in enumerate(supported_segments):
                    if supported:
                        cv2.line(frame,tuple(q[k]),tuple(q[k+1]),(230,60,230),2,cv2.LINE_AA)
            elif self.backend=='external':
                cv2.polylines(frame,[q],False,(230,60,230),2,cv2.LINE_AA)
        raw = front.get('line_abc')
        measured = np.asarray(front.get('line_points_xy', []),float).reshape(-1,2)
        if raw is not None:
            lo = float(np.min(measured[:,0])) if len(measured) else .1*self.w
            hi = float(np.max(measured[:,0])) if len(measured) else .95*self.w
            points = [[lo,y_at(raw,lo)],[hi,y_at(raw,hi)]]
            if all(p[1] is not None for p in points):
                r=self.points(points);cv2.line(frame,tuple(r[0]),tuple(r[1]),(255,230,0),2,cv2.LINE_AA)
                if eval_x is not None and (eval_x<lo or eval_x>hi):
                    # Analytic extrapolation is explicitly dashed. It is not
                    # counted as observed S1 support or a complete P1 path.
                    near=lo if eval_x<lo else hi
                    xx=np.linspace(near,float(eval_x),60)
                    yy=np.array([y_at(raw,float(x)) for x in xx])
                    if all(y is not None for y in yy):
                        rr=self.points(np.c_[xx,yy.astype(float)])
                        for k in range(0,len(rr)-1,2):
                            cv2.line(frame,tuple(rr[k]),tuple(rr[k+1]),(255,230,0),1,cv2.LINE_AA)
                model=lock.get('locked_abc')
                if model is not None and lock.get('count_allowed'):
                    # The white dotted line is the explicit frozen-slope model;
                    # cyan remains the independently measured raw straight line.
                    xx=np.linspace(lo,hi,70)
                    yy=np.array([y_at(model,x) for x in xx])
                    valid=np.array([v is not None for v in yy])
                    if valid.all():
                        rr=self.points(np.c_[xx,yy.astype(float)])
                        for k in range(0,len(rr)-1,2):
                            cv2.line(frame,tuple(rr[k]),tuple(rr[k+1]),(245,245,245),1,cv2.LINE_AA)
        locked = bool(lock.get('d') is not None and str(lock.get('state','')).lower() not in ('calibrating','initializing','invalid','invalidated'))
        if eval_x is not None:
            x_orig=self.w-1-eval_x if self.mirror else eval_x
            x=int(round(x_orig*self.scale[0]));color=(40,240,40) if locked else (40,150,255)
            if locked:
                cv2.line(frame,(x,int(.45*self.oh)),(x,self.oh-1),color,2,cv2.LINE_AA)
            else:
                for yy in range(int(.45*self.oh),self.oh,18):
                    cv2.line(frame,(x,yy),(x,min(yy+9,self.oh-1)),color,2)
            if eval_y is not None:
                cy=int(round(eval_y*self.scale[1]));cv2.circle(frame,(x,cy),7,(0,40,255),-1,cv2.LINE_AA)
                edge=surface.get('edge_y')
                if is_finite_number(edge):
                    ey=int(round(float(edge)*self.scale[1]));cv2.circle(frame,(x,ey),6,(255,150,20),-1)
                    cv2.line(frame,(x,ey),(x,cy),color,4,cv2.LINE_AA)
                for yy in surface.get('crossings_y',[]):
                    if edge is not None and float(edge)<float(yy)<eval_y:
                        py=int(round(float(yy)*self.scale[1]));cv2.circle(frame,(x,py),3,(0,220,255),-1)
        raw_o=original_abc(raw,self.w,self.mirror)
        locked_o=original_abc(lock.get('locked_abc'),self.w,self.mirror)
        abc=locked_o if locked_o is not None and lock.get('count_allowed') else raw_o
        d_orig=(self.w-1-eval_x if self.mirror else eval_x) if eval_x is not None else None
        count_ok=bool(lock.get('count_allowed') and surface.get('count_valid'))
        continuous=surface.get('continuous_pitch_distance')
        if continuous is None:
            continuous=surface.get('continuous_pitch_distance_candidate')
        count_text=(f"{continuous:.2f} 个绳距" if is_finite_number(continuous) else '不可测')
        if not count_ok and is_finite_number(continuous):
            count_text+='（候选，未通过质量门槛）'
        if self.backend=='traditional-preview':
            label='LC1-T 技术预演：P1/H6、S1 未接入；F3 传统观测核心'
        elif self.backend=='p1-h6-s1':
            label='LC1：真实 B750/H6 + S1；当前 F3/D4 场景适配'
            if front.get('diagnostics',{}).get('S2') is not None:
                label+='；S2 独立审计'
        else:
            label='LC1：P1/H6 + S1 外部逐帧结果 + F3 观测核心'
        equations='直线未检出' if abc is None else f'{abc[0]:+.6f} x {abc[1]:+.6f} y {abc[2]:+.3f} = 0'
        dtext='d 待校准' if d_orig is None else f'x = d = {d_orig:.1f} px   '+('已锁列' if locked else '仅当前候选')
        if not locked and self.reference_column is not None and eval_x is not None:
            dtext=f'固定诊断参考列 x = {d_orig:.1f} px；转折 d 尚未自动校准'
        if not locked and eval_x is not None and turn.get('column_proposal_source')=='H6_join_reference_not_d':
            dtext=f'H6 连接参考列 x = {d_orig:.1f} px；此列尚不是已标定的 d'
        elif not locked and eval_x is not None and turn.get('column_proposal_source')=='rejected_turn_fit':
            dtext=f'未通过定位检验的 d 候选 x = {d_orig:.1f} px；不授权计数'
        turn_labels={'no_unique_legal_H6_curve_connection':'缺少唯一合法曲段',
                     'turn_profile_open_or_boundary_limited':'定位范围未收敛',
                     'insufficient_current_supported_surface_points':'当前曲段观测不足',
                     'S1_and_curve_domains_overlap_or_no_localizable_gap':'直曲观测域不足以定位',
                     'turn_localization_uncertain':'d 定位不确定度过大',
                     'poor_observed_surface_model_fit':'曲段拟合未通过'}
        detail=f"转折：{turn_labels.get(turn.get('reason'),turn.get('reason','—'))}｜d不确定度 ±{fmt(turn.get('uncertainty_px'))} px｜整绳证据 {'通过' if front.get('whole_path_candidate_supported') else '未通过'}"
        lines=[label,
               f"原{'左' if self.camera=='left' else '右'}相机｜帧 {frame_index}｜t={t:.3f}s｜原图坐标 {self.w}×{self.h}",
               equations,dtext,
               f"距挡边：{count_text}；已观测沟槽 {surface.get('count_observed_grooves','—')}",
               f"状态 {lock.get('state','—')}｜{lock.get('reason','')} ",
               f"角度漂移 {fmt((-1 if self.mirror else 1)*lock['angle_drift_deg'] if is_finite_number(lock.get('angle_drift_deg')) else None)}°｜d漂移 {fmt((-1 if self.mirror else 1)*lock['d_drift_px'] if is_finite_number(lock.get('d_drift_px')) else None)} px｜模型交点差 {fmt(lock.get('frozen_vs_raw_intersection_drift_px'))} px｜{processing_ms:.0f} ms",
               '黄：当前绳沟  青实线：实测直线／虚线：外推  绿：锁定列  红：交点',
               detail,
        ]
        panel=frame.copy();cv2.rectangle(panel,(12,12),(min(self.ow-12,1130),300),(12,16,22),-1)
        cv2.addWeighted(panel,.90,frame,.10,0,frame)
        pil=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(pil)
        for j,line in enumerate(lines):
            draw.text((24,20+30*j),line,font=self.font if j<5 else self.small,
                      fill=('#80ff91' if j==4 and count_ok else '#f5f7fa' if j!=4 else '#ffd37b'))
        return cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)


def fmt(v):
    return f'{float(v):.2f}' if is_finite_number(v) else '—'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--camera',choices=['left','right'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backend',choices=['traditional-preview','external','p1-h6-s1'],default='external')
    parser.add_argument('--external-observations',type=Path)
    parser.add_argument('--model-path',type=Path,default=ROOT/'supplied_assets'/'step_0750.pt',
                        help='Actual frozen B750 step_0750.pt for live P1/H6')
    parser.add_argument('--p1-surface-mode',choices=['full','entry_band'],default='full',
                        help='Current D4 observation domain; entry_band is an explicit upstream adaptation')
    parser.add_argument('--s2-audit',action='store_true',help='Run original S2 guidance/core as a separate optical proposal audit')
    parser.add_argument('--s1-source-dir',type=Path,help='Supplied frozen S1 directory; default uses the supplied temporary asset directory')
    parser.add_argument('--start-frame',type=int,default=0)
    parser.add_argument('--stop-frame',type=int,default=None,help='Exclusive source frame index')
    parser.add_argument('--width',type=int,default=1920)
    parser.add_argument('--max-unverified-lock-age-s',type=float,default=15.)
    parser.add_argument('--preview-stride',type=int,default=0,help='Diagnostic sampling only; full result defaults to every frame')
    parser.add_argument('--diagnostic-column-original',type=float,
                        help='Explicit fixed reference column for a labelled LC1-T ablation, NEVER an automatic d')
    args=parser.parse_args()
    video=args.video.resolve();out=args.output.resolve()
    if any(s in str(video) for s in ['20260925','dual_20260923_113029']):
        raise ValueError('Reserved/heldout media excluded from this development run')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Refusing to overwrite an existing run')
    if args.backend=='external' and not args.external_observations:
        raise ValueError('Actual P1/H6/S1 observations required for the external backend')
    if args.backend=='p1-h6-s1' and args.model_path is None:
        raise ValueError('--model-path to the actual frozen B750 checkpoint is required')
    if args.diagnostic_column_original is not None and args.backend!='traditional-preview':
        raise ValueError('Manual diagnostic columns cannot authorize the actual P1/H6/S1 backend')
    out.mkdir(parents=True,exist_ok=True);(out/'samples').mkdir()
    snapshot=out/'source_snapshot';snapshot.mkdir()
    source_freeze={}
    for path in ROOT.glob('lc1_*.py'):
        shutil.copyfile(path,snapshot/path.name);source_freeze[path.name]=sha(path)
    dump(snapshot/'manifest.json',source_freeze)
    initial_stat=video.stat();source_sha=sha(video)
    cap=cv2.VideoCapture(str(video));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not cap.isOpened() or n<=0 or fps<=0:raise ValueError('Source video not complete/readable')
    if args.diagnostic_column_original is not None and not 0<=args.diagnostic_column_original<w:
        raise ValueError('Diagnostic reference column outside the source image')
    stop=n if args.stop_frame is None else min(n,args.stop_frame)
    if not 0<=args.start_frame<stop:raise ValueError('Invalid frame range')
    mask,pitch,calib=calibration(video.parent.name,args.camera)
    if mask.shape!=(h,w):raise ValueError('Scene calibration shape mismatch')
    mask_original=mask.copy()
    mirror=args.camera=='left'
    if mirror:mask=cv2.flip(mask,1)
    from lc1_geometry import estimate_turn
    from lc1_surface import SurfaceObserver
    surface_engine=SurfaceObserver(mask,pitch)
    if args.backend=='traditional-preview':
        from lc1_frontend import TraditionalFrontend
        frontend=TraditionalFrontend(mask,pitch_px=pitch,free_side='right')
    elif args.backend=='p1-h6-s1':
        from lc1_pipeline import P1S1Frontend
        frontend=P1S1Frontend(mask_original,pitch,args.model_path,camera=args.camera,
                             s1_source_dir=args.s1_source_dir,surface_mode=args.p1_surface_mode,s2_audit=args.s2_audit)
    else:frontend=ExternalFrontend(args.external_observations,w,mirror,source_sha256=source_sha,camera=args.camera)
    lock_engine=make_lock(args.max_unverified_lock_age_s)
    frame_times,time_basis,time_path=timestamps(video,args.camera,n,fps)
    overlay=Overlay(w,h,args.camera,args.backend,args.width,args.diagnostic_column_original)
    stride=max(1,args.preview_stride);output_fps=fps/stride
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-hide_banner','-loglevel','error',
         '-f','rawvideo','-pix_fmt','bgr24','-s',f'{overlay.ow}x{overlay.oh}',
         '-r',str(output_fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast',
         '-crf','20','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(out/'overlay.mp4')]
    log=(out/'encoder.log').open('wb')
    process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log,
                             creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    fields=['frame_index','time_s','backend','state','reason','count_valid','d_original_px',
            'd_canonical_px','d_candidate_original_px','diagnostic_reference_column_original_px','evaluation_column_original_px',
            'A','B','C','A_raw','B_raw','C_raw',
            'intersection_y_px','edge_y_px','observed_grooves','continuous_pitch_distance','continuous_pitch_distance_candidate',
            'pitch_reference_px','raw_angle_deg','angle_drift_deg','d_drift_px',
            'raw_angle_canonical_deg','angle_drift_canonical_deg','d_drift_canonical_px','lock_time_s',
            'frozen_vs_raw_intersection_drift_px','C_fit_basis',
            'whole_path_candidate_supported','turn_valid','turn_reason','turn_uncertainty_px',
            'P1_model_forward_count','S1_core_executed','S2_audit_valid',
            'surface_quality','processing_ms','surface_seconds','front_source']
    rows=[];events=[];started=time.perf_counter();last_state=None
    seek_ok=cap.set(cv2.CAP_PROP_POS_FRAMES,args.start_frame)
    seek_position=cap.get(cv2.CAP_PROP_POS_FRAMES)
    if not seek_ok or abs(seek_position-args.start_frame)>.1:
        cap.release();cap=cv2.VideoCapture(str(video))
        for _ in range(args.start_frame):
            if not cap.grab():raise RuntimeError('Cannot reach requested start frame')
        if abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-args.start_frame)>.1:
            raise RuntimeError('Source frame seek cannot be verified')
    status='running';error=None
    try:
        with (out/'measurements.csv').open('w',encoding='utf-8-sig',newline='') as csvfile, gzip.open(out/'diagnostics.jsonl.gz','wt',encoding='utf-8') as detail:
            writer=csv.DictWriter(csvfile,fieldnames=fields);writer.writeheader()
            for frame_index in range(args.start_frame,stop):
                ok,original=cap.read()
                if not ok:raise RuntimeError(f'Unexpected video EOF at {frame_index}/{stop}')
                if (frame_index-args.start_frame)%stride:continue
                tick=time.perf_counter();t=frame_times[frame_index]
                bgr=cv2.flip(original,1) if mirror else original
                if args.backend=='p1-h6-s1':
                    front=frontend.step(original,t,frame_index)
                elif args.backend=='external':
                    front=frontend.step(bgr,t,frame_index)
                else:
                    front=frontend.step(bgr,t)
                raw=front.get('line_abc');path=front.get('path_xy',[])
                if args.backend=='p1-h6-s1':
                    from lc1_pipeline import estimate_front_turn
                    turn=estimate_front_turn(front,pitch)
                else:
                    turn=estimate_turn(path,free_side='right',pitch_px=pitch) if len(path)>=8 else dict(valid=False,reason='no_whole_path')
                lock=lock_engine.update(t,raw,turn,pitch,
                       complete_path_supported=bool(front.get('whole_path_candidate_supported')),
                       line_points_xy=front.get('line_points_xy'))
                frozen_d=lock.get('d')
                candidate_d=turn.get('x') if is_finite_number(turn.get('x')) else None
                if candidate_d is not None and not turn.get('valid'):
                    turn['column_proposal_source']='rejected_turn_fit'
                eval_x=float(frozen_d) if is_finite_number(frozen_d) else float(candidate_d) if is_finite_number(candidate_d) else None
                if frozen_d is None and args.diagnostic_column_original is not None:
                    eval_x=w-1-args.diagnostic_column_original if mirror else args.diagnostic_column_original
                active_abc=lock.get('locked_abc') if lock.get('count_allowed') and lock.get('locked_abc') is not None else raw
                eval_y=y_at(active_abc,eval_x)
                st=time.perf_counter()
                surface=surface_engine.step(bgr,d=eval_x,active_y=eval_y,exclusion_mask=None)
                surface_seconds=time.perf_counter()-st
                elapsed=(time.perf_counter()-tick)*1000
                rendered=overlay.draw(original,frame_index,t,front,turn,lock,surface,eval_x,eval_y,elapsed)
                process.stdin.write(rendered.tobytes())
                raw_o=original_abc(raw,w,mirror);abc_o=original_abc(active_abc,w,mirror)
                locked_original=(w-1-float(frozen_d) if mirror else float(frozen_d)) if is_finite_number(frozen_d) else None
                candidate_original=(w-1-float(candidate_d) if mirror else float(candidate_d)) if is_finite_number(candidate_d) else None
                row=dict(frame_index=frame_index,time_s=t,backend=args.backend,state=lock.get('state'),reason=lock.get('reason'),
                         count_valid=bool(lock.get('count_allowed') and surface.get('count_valid')),
                         d_original_px=locked_original,d_canonical_px=frozen_d,d_candidate_original_px=candidate_original,
                         diagnostic_reference_column_original_px=args.diagnostic_column_original,
                         evaluation_column_original_px=(w-1-eval_x if mirror else eval_x) if eval_x is not None else None,
                         A=None if abc_o is None else abc_o[0],B=None if abc_o is None else abc_o[1],C=None if abc_o is None else abc_o[2],
                         A_raw=None if raw_o is None else raw_o[0],B_raw=None if raw_o is None else raw_o[1],C_raw=None if raw_o is None else raw_o[2],
                         intersection_y_px=eval_y,edge_y_px=surface.get('edge_y'),
                         observed_grooves=surface.get('count_observed_grooves'),continuous_pitch_distance=surface.get('continuous_pitch_distance'),
                         continuous_pitch_distance_candidate=surface.get('continuous_pitch_distance_candidate'),
                         pitch_reference_px=pitch,
                         raw_angle_deg=(-1 if mirror else 1)*lock['raw_angle_deg'] if is_finite_number(lock.get('raw_angle_deg')) else None,
                         angle_drift_deg=(-1 if mirror else 1)*lock['angle_drift_deg'] if is_finite_number(lock.get('angle_drift_deg')) else None,
                         d_drift_px=(-1 if mirror else 1)*lock['d_drift_px'] if is_finite_number(lock.get('d_drift_px')) else None,
                         raw_angle_canonical_deg=lock.get('raw_angle_deg'),angle_drift_canonical_deg=lock.get('angle_drift_deg'),
                         d_drift_canonical_px=lock.get('d_drift_px'),lock_time_s=lock.get('lock_time'),
                         frozen_vs_raw_intersection_drift_px=lock.get('frozen_vs_raw_intersection_drift_px'),
                         C_fit_basis=lock.get('C_fit_basis'),
                         whole_path_candidate_supported=bool(front.get('whole_path_candidate_supported')),
                         turn_valid=turn.get('valid') is True,turn_reason=turn.get('reason'),turn_uncertainty_px=turn.get('uncertainty_px'),
                         P1_model_forward_count=front.get('diagnostics',{}).get('P1_model_forward_count',0),
                         S1_core_executed=front.get('diagnostics',{}).get('S1_core_actually_executed',False),
                         S2_audit_valid=(front.get('diagnostics',{}).get('S2') or {}).get('valid',False),
                         surface_quality=surface.get('quality'),processing_ms=elapsed,surface_seconds=surface_seconds,front_source=front.get('source'))
                row=clean(row);writer.writerow(row);csvfile.flush();rows.append(row)
                # Geometry records preserve actual crossings and fit diagnostics,
                # not hundreds of full-size masks per frame.
                diagnostic=dict(frame_index=frame_index,time_s=t,line_abc_original=abc_o,line_abc_raw_original=raw_o,
                                front=front,turn=turn,lock=lock,
                                surface={k:v for k,v in surface.items() if k not in ('boundary_polylines_xy','fit_polylines_xy','raw_core','fields')})
                detail.write(json.dumps(clean(diagnostic),ensure_ascii=False,separators=(',',':'))+'\n')
                if lock.get('state')!=last_state:
                    events.append(dict(frame_index=frame_index,time_s=t,state=lock.get('state'),reason=lock.get('reason'),d_original_px=locked_original))
                    last_state=lock.get('state')
                if len(rows)==1 or frame_index%100==0 or lock.get('just_locked'):
                    cv2.imencode('.jpg',rendered,[cv2.IMWRITE_JPEG_QUALITY,94])[1].tofile(str(out/'samples'/f'frame_{frame_index:06d}.jpg'))
                    dump(out/'samples'/f'frame_{frame_index:06d}.json',diagnostic)
                if len(rows)%20==0:
                    dump(out/'progress.json',dict(status='running',processed=len(rows),last_frame=frame_index,
                         total_requested=math.ceil((stop-args.start_frame)/stride),wall_s=time.perf_counter()-started,state=last_state))
                    print(f'{video.parent.name} {args.camera}: {len(rows)} frames; source {frame_index}/{stop-1}; {last_state}',flush=True)
        status='completed'
    except Exception as exc:
        status='failed';error=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        cap.release()
        try:
            if process.stdin:process.stdin.close()
        except (BrokenPipeError,OSError):
            pass
        exit_code=process.wait();log.close()
        if exit_code and status=='completed':status='encoder_failed'
        encoded=dict(expected_frames=len(rows),checked=False)
        if status=='completed':
            verify=cv2.VideoCapture(str(out/'overlay.mp4'))
            encoded.update(checked=True,opened=verify.isOpened(),frames=int(verify.get(cv2.CAP_PROP_FRAME_COUNT)),
                           width=int(verify.get(cv2.CAP_PROP_FRAME_WIDTH)),height=int(verify.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                           fps=verify.get(cv2.CAP_PROP_FPS))
            first_ok,_=verify.read()
            if len(rows):verify.set(cv2.CAP_PROP_POS_FRAMES,len(rows)-1)
            last_ok,_=verify.read();eof_ok,_=verify.read();verify.release()
            encoded.update(first_frame_decoded=bool(first_ok),last_frame_decoded=bool(last_ok),eof_after_last=not eof_ok)
            if not (encoded['opened'] and encoded['frames']==len(rows) and first_ok and last_ok and not eof_ok):
                status='encoded_output_verification_failed'
        final_stat=video.stat()
        stable=initial_stat.st_size==final_stat.st_size and initial_stat.st_mtime_ns==final_stat.st_mtime_ns
        if not stable:status='source_changed_during_processing'
        timing=[r['processing_ms'] for r in rows if is_finite_number(r['processing_ms'])]
        algorithm_version=('LC1.2.0 live B750/H6/S1 with observed-branch turn fitting' if args.backend=='p1-h6-s1' else
                           'LC1.0.0 external observations' if args.backend=='external' else 'LC1-T.0.0 technical preview')
        report=dict(status=status,error=error,algorithm=algorithm_version,
                    actual_live_P1_H6_S1_algorithms_connected=args.backend=='p1-h6-s1',
                    historical_frozen_P1_scene_cache_reproduced=False,
                    requested_pipeline_P1_H6_S1_external_success_conditions_met=bool(args.backend=='external' and
                        sum(bool(r['whole_path_candidate_supported']) for r in rows)>=8 and
                        sum(is_finite_number(r['A_raw']) for r in rows)>0),
                    externally_sourced_valid_line_frames=sum(is_finite_number(r['A_raw']) for r in rows) if args.backend=='external' else 0,
                    external_observations_are_supplied_not_new_model_inference=args.backend=='external',
                    f3_scope='unchanged RB-F3.0.0 traditional observation core; not complete frozen F3 fusion',
                    no_target_identity_accuracy_claim=True,no_physical_contact_or_count_ground_truth=True,
                    diagnostic_reference_column_original_px=args.diagnostic_column_original,
                    diagnostic_reference_column_is_not_an_automatically_estimated_d=args.diagnostic_column_original is not None,
                    created_utc=datetime.now(timezone.utc).isoformat(),source_video=str(video),source_sha256=source_sha,
                    source_size=initial_stat.st_size,source_stable=stable,source_frames=n,source_fps=fps,
                    frame_range_exclusive=[args.start_frame,stop],processed_frames=len(rows),stride=stride,
                    causal=True,future_target_frames_used=False,static_scene_calibration=calib,
                    time_basis=time_basis,frame_time_csv=time_path,internal_left_mirror=mirror,
                    rendered_orientation='original_unmirrored',equation_coordinate_system='original_4096x3000_xy',
                    output_shape=[overlay.oh,overlay.ow],output_fps=output_fps,encoder_exit=exit_code,
                    encoded_output_verification=encoded,
                    valid_count_frames=sum(bool(r['count_valid']) for r in rows),
                    complete_path_candidate_frames=sum(bool(r['whole_path_candidate_supported']) for r in rows),
                    state_counts={s:sum(r['state']==s for r in rows) for s in sorted({str(r['state']) for r in rows})},
                    median_algorithm_ms=float(np.median(timing)) if timing else None,
                    wall_seconds=time.perf_counter()-started,events=events,command=sys.argv,
                    output_video_sha256=sha(out/'overlay.mp4') if (out/'overlay.mp4').exists() else None,
                    source_freeze_sha256=source_freeze,
                    live_model_path=str(args.model_path.resolve()) if args.backend=='p1-h6-s1' else None,
                    live_model_sha256=sha(args.model_path) if args.backend=='p1-h6-s1' else None,
                    S1_source_dir=str(frontend.s1.source_dir) if args.backend=='p1-h6-s1' else None,
                    S1_frozen_hashes=frontend.s1.hashes if args.backend=='p1-h6-s1' else None,
                    P1_surface_mode=args.p1_surface_mode if args.backend=='p1-h6-s1' else None,
                    S2_independent_audit_enabled=args.s2_audit if args.backend=='p1-h6-s1' else False,
                    S2_used_for_C_or_whole_path_authorization=False,
                    actual_P1_model_forward_count=sum(r['P1_model_forward_count'] for r in rows),
                    actual_S1_core_execution_frames=sum(bool(r['S1_core_executed']) for r in rows),
                    turn_estimator=('observed_curve_with_independent_current_S1_anchor' if args.backend=='p1-h6-s1' else
                                    'legacy_global_hinge_fit_not_used_for_live_backend'),
                    software=dict(python=sys.version,opencv=cv2.__version__,numpy=np.__version__))
        dump(out/'run_receipt.json',report);dump(out/'progress.json',dict(status=status,processed=len(rows)))
        print(json.dumps(clean({k:report[k] for k in ['status','algorithm','processed_frames','valid_count_frames','state_counts','wall_seconds']}),ensure_ascii=False),flush=True)
    return 0 if status=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
