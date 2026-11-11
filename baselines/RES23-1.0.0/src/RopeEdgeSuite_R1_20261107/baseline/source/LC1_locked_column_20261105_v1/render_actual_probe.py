"""Render verified real-model observations, plus freshly measured F3.

P1/H6 and S1 are replayed from the documented current-frame inference probe.
Only those actual sampled images are rendered. Repeated output frames are
explicit holds, never fabricated detections or a claim of per-frame inference.
"""
from __future__ import annotations
import csv
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lc1_geometry import ColumnLock, estimate_turn
from lc1_pipeline import line_to_canonical, points_to_canonical, estimate_front_turn
from lc1_surface import SurfaceObserver
from lc1_video import Overlay, calibration, clean, dump, sha, timestamps, y_at

ROOT=Path(__file__).resolve().parent
PROBE=ROOT/'p1_runtime'/'multiframe_probe'
OUT=ROOT/'runs'/'actual_models_sampled_20261105'
TURN_METHOD='legacy'


def frame_sha(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def caption(image,lines):
    pil=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    draw=ImageDraw.Draw(pil)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21)
    draw.rectangle((12,305,min(image.shape[1]-12,1500),410),fill=(16,21,29))
    for i,line in enumerate(lines):
        draw.text((24,310+31*i),line,font=font,fill='#ffe1a6')
    return cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)


def run_role(role,records):
    selected=sorted([r for r in records if r['P1_H6']['camera']==role],key=lambda r:r['frame_index'])
    assert len(selected)==6
    folder=OUT/role
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'samples').mkdir()
    video=Path(selected[0]['source_video'])
    source_hash=sha(video)
    source_initial=video.stat()
    cap=cv2.VideoCapture(str(video))
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    host_times,time_basis,time_csv=timestamps(video,role,count,fps)
    mask,pitch,scene=calibration(video.parent.name,role)
    mirror=role=='left'
    canonical_mask=cv2.flip(mask,1) if mirror else mask
    surface_engine=SurfaceObserver(canonical_mask,pitch)
    lock_engine=ColumnLock()
    overlay=Overlay(width,height,role,'p1-h6-s1',1920)
    gap=selected[1]['frame_index']-selected[0]['frame_index']
    assert all(b['frame_index']-a['frame_index']==gap for a,b in zip(selected,selected[1:]))
    output_fps=10
    hold_frames=round(output_fps*gap/fps)
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-hide_banner','-loglevel','error',
             '-f','rawvideo','-pix_fmt','bgr24','-s',f'{overlay.ow}x{overlay.oh}',
             '-r',str(output_fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast',
             '-crf','20','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(folder/'overlay.mp4')]
    rows=[];details=[];started=time.perf_counter()
    with (folder/'encoder.log').open('wb') as log:
        encoder=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log,
                                 creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            for record in selected:
                frame=record['frame_index']
                assert cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
                assert abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-frame)<.1
                ok,original=cap.read()
                assert ok and frame_sha(original)==record['original_pixel_sha256']
                p1,s1=record['P1_H6'],record['S1']
                path=points_to_canonical(p1.get('path_xy'),width,mirror)
                line=line_to_canonical(s1.get('line_abc'),width,mirror) if s1.get('valid') is True else None
                measured=points_to_canonical(s1.get('line_points_xy'),width,mirror) if line is not None else []
                support=p1['diagnostics'].get('path_segment_support',[])
                assert len(support)==max(0,len(path)-1)
                front=dict(line_abc=line,line_points_xy=measured,path_xy=path,
                           whole_path_candidate_supported=p1.get('whole_path_candidate_supported') is True,
                           source='replayed_verified_current_frame_real_B750_H6_and_frozen_S1',
                           diagnostics=dict(path_segment_support=support,P1_H6=p1))
                if TURN_METHOD=='observed':
                    turn=estimate_front_turn(front,pitch)
                else:
                    turn=estimate_turn(path,free_side='right',pitch_px=pitch) if len(path)>=24 else dict(valid=False,reason='insufficient_path')
                lock=lock_engine.update(host_times[frame],line,turn,pitch,
                    complete_path_supported=front['whole_path_candidate_supported'],line_points_xy=measured)
                column=lock.get('d')
                kind='locked_d' if column is not None else None
                if column is None and turn.get('x') is not None and np.isfinite(turn['x']):
                    column=float(turn['x']);kind='valid_turn_candidate' if turn.get('valid') else 'rejected_turn_fit'
                if column is None:
                    join=p1['diagnostics'].get('join_start_xy')
                    if join is not None:
                        column=float(points_to_canonical([join],width,mirror)[0][0]);kind='H6_join_reference_not_d'
                turn['column_proposal_source']=kind
                active_abc=lock.get('locked_abc') if lock.get('count_allowed') else line
                active=y_at(active_abc,column)
                image=cv2.flip(original,1) if mirror else original
                tick=time.perf_counter()
                surface=surface_engine.step(image,d=column,active_y=active)
                surface_seconds=time.perf_counter()-tick
                elapsed=(record.get('total_seconds',0)+surface_seconds)*1000
                rendered=overlay.draw(original,frame,host_times[frame],front,turn,lock,surface,column,active,elapsed)
                whole='通过' if front['whole_path_candidate_supported'] else '未通过'
                rendered=caption(rendered,[
                    f'抽样核验：每 {gap/fps:.1f} 秒取一真实帧；重复画面仅为停留显示，不是连续逐帧推理',
                    f'完整绳证据：{whole}；转折拟合：{turn.get("reason")}；本视频自动锁列：未通过',
                    '真实 P1/H6、S1 结果按原始帧重放；F3 对该原始帧重新计算；不输出已确认根数',
                ])
                for _ in range(hold_frames):
                    encoder.stdin.write(rendered.tobytes())
                cv2.imencode('.jpg',rendered,[cv2.IMWRITE_JPEG_QUALITY,94])[1].tofile(str(folder/'samples'/f'frame_{frame:06d}.jpg'))
                row=dict(frame_index=frame,time_s=host_times[frame],source_nominal_time_s=frame/fps,
                         complete_path_supported=front['whole_path_candidate_supported'],turn_valid=bool(turn.get('valid')),
                         turn_reason=turn.get('reason'),turn_uncertainty_px=turn.get('uncertainty_px'),
                         d_original_px=lock.get('d'),reference_column_kind=kind,
                         reference_column_original_px=(width-1-column if mirror else column) if column is not None else None,
                         S1_valid=s1.get('valid'),S1_abc_original=s1.get('line_abc'),
                         intersection_y_px=active,edge_y_px=surface.get('edge_y'),
                         observed_grooves=surface.get('count_observed_grooves'),
                         continuous_pitch_distance_candidate=surface.get('continuous_pitch_distance_candidate'),
                         surface_count_geometry_valid=surface.get('count_valid'),
                         accepted_automatic_count=bool(lock.get('count_allowed') and surface.get('count_valid')),
                         state=lock.get('state'),model_inference_seconds=record.get('total_seconds'),
                         F3_seconds=surface_seconds,source_pixel_sha256=record['original_pixel_sha256'])
                rows.append(clean(row))
                details.append(clean(dict(**row,turn=turn,lock=lock,
                    surface={k:v for k,v in surface.items() if k not in ('boundary_polylines_xy','fit_polylines_xy')},
                    P1_inference_record_frame=frame,P1_source_mode=record['surface_mode'])))
                print(f'{role} frame {frame}: complete={whole}, turn={turn.get("reason")}, reference={kind}',flush=True)
        finally:
            cap.release();encoder.stdin.close();returncode=encoder.wait()
    assert returncode==0
    with (folder/'measurements.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    dump(folder/'diagnostics.json',details)
    verify=cv2.VideoCapture(str(folder/'overlay.mp4'))
    expected=len(rows)*hold_frames
    actual=int(verify.get(cv2.CAP_PROP_FRAME_COUNT));first,_=verify.read()
    verify.set(cv2.CAP_PROP_POS_FRAMES,expected-1);last,_=verify.read();eof,_=verify.read();verify.release()
    assert first and last and not eof and actual==expected
    assert not any(row['accepted_automatic_count'] or row['d_original_px'] is not None for row in rows)
    final=video.stat();assert final.st_size==source_initial.st_size and final.st_mtime_ns==source_initial.st_mtime_ns
    report=dict(status='completed_diagnostic_sampling_not_automatic_calibration',camera=role,
        original_video=str(video),source_sha256=source_hash,source_unchanged=True,
        source_frames_processed=[r['frame_index'] for r in selected],
        P1_H6_S1_observations_replayed_from_verified_actual_inference=True,
        P1_H6_S1_inference_rerun_by_this_renderer=False,F3_fresh_for_each_sample=True,
        input_probe_jsonl=str(PROBE/'frames.jsonl'),input_probe_jsonl_sha256=sha(PROBE/'frames.jsonl'),
        model_sha256=selected[0]['P1_H6']['model_sha256'],
        per_source_sample_hold_frames=hold_frames,output_fps=output_fps,encoded_frames=actual,
        sampled_source_frames=len(rows),automatic_locked_d_frames=0,accepted_automatic_count_frames=0,
        reference_columns_are_not_accepted_d=True,rendered_orientation='original_unmirrored_xy',
        output_size=[overlay.ow,overlay.oh],time_basis=time_basis,host_time_csv=time_csv,
        static_scene_calibration=scene,F3_scope='traditional_observation_core_not_complete_historical_F3_fusion',
        source_observation_adapter_mode=selected[0]['surface_mode'],turn_method=TURN_METHOD,
        video_sha256=sha(folder/'overlay.mp4'),wall_seconds=time.perf_counter()-started)
    dump(folder/'receipt.json',report)
    return report


def main():
    global OUT,TURN_METHOD
    parser=argparse.ArgumentParser()
    parser.add_argument('--turn-method',choices=['legacy','observed'],default='observed')
    args=parser.parse_args()
    TURN_METHOD=args.turn_method
    OUT=ROOT/'runs'/('actual_models_observed_turn_20261105' if TURN_METHOD=='observed' else 'actual_models_sampled_20261105')
    OUT.mkdir(parents=True,exist_ok=False)
    snapshot=OUT/'source_snapshot';snapshot.mkdir()
    sources={}
    for path in [Path(__file__),*ROOT.glob('lc1_*.py')]:
        shutil.copyfile(path,snapshot/path.name);sources[path.name]=sha(path)
    dump(snapshot/'manifest.json',sources)
    records=[json.loads(line) for line in (PROBE/'frames.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    reports=[run_role(role,records) for role in ('left','right')]
    dump(OUT/'receipt.json',dict(scope='actual-model sampled diagnostic overlays; not a successful automatic d result',runs=reports))
    print(json.dumps(clean(reports),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
