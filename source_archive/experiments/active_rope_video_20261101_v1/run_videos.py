"""Frozen-B offline video adapter. Causal scene initialization and path gating."""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name,'4')
import sys,json,time,hashlib,argparse,traceback,math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.append('D:/precise_arrangement/vision_prototype/.venv/Lib/site-packages')
import cv2,numpy as np
from frame_engine.engine import FrameEngine,anchor_job
from temporal import TemporalRopeGate,GateConfig
from rendering.render_panels import render_frame

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def line(f,d):f.write(json.dumps(d,ensure_ascii=False,separators=(',',':'))+'\n');f.flush()

def adapt_candidates(raw):
    """Keep upstream order; no newly generated path and no identity probability."""
    out=[]
    for i,c in enumerate(raw[:3]):
        p=np.asarray(c['points_xy'],float)
        statuses=np.asarray(c.get('point_status',['unknown']*len(p)))
        observed=np.isin(statuses,['image_supported','joint_heat_image_supported','raw_a7_supported'])
        length=float(np.linalg.norm(np.diff(p,axis=0),axis=1)[observed[:-1]&observed[1:]].sum())
        supported=bool(c.get('task_direction_allowed',False) and length>=80.)
        out.append(dict(id=f'candidate_{i}',entry_index=c['entry_index'],points_xy=c['points_xy'],
            image_score=1.-.1*i,image_supported=supported,current_raw_supported_length_px=length,
            upstream_rank_score=c.get('rank_score'),score_meaning='monotonic upstream rank utility, not probability'))
    return out

def run_one(video,run_dir,pilot,threads):
    cv2.setNumThreads(1)
    ident=video['video_id'];out=Path(run_dir)/ident;out.mkdir(parents=True,exist_ok=False)
    (out/'responses').mkdir();(out/'previews').mkdir()
    started=time.perf_counter();cap=None;writer=None
    try:
        times=np.asarray(read(video['frame_times_path'])['observation_times_s'],float)
        if not np.all(np.diff(times)>0):raise ValueError('Non-monotone source observation times')
        fps=float(video['nominal_fps']);n=int(video['frame_count_metadata'])
        a=video['earliest_dataset_anchor'];anchor_index=int(a['source_frame']);anchor_t=float(a['mapped_observation_time_s'])
        duration=n/fps
        if pilot:duration=min(duration,(anchor_index/fps)+5.5)
        last=min(n-1,int(math.ceil(duration*fps))-1)
        analysis=np.unique(np.searchsorted(times,np.arange(anchor_t,times[last]+1e-8,.5)))
        analysis=set(int(x) for x in analysis if anchor_index<=x<=last)
        render_indices=np.minimum(np.rint(np.arange(int(math.ceil(duration*5)))*fps/5).astype(int),last)
        render_counts={int(i):int((render_indices==i).sum()) for i in np.unique(render_indices)}
        events=analysis|set(render_counts)
        cap=cv2.VideoCapture(video['canonical_path'])
        if not cap.isOpened():raise RuntimeError('Cannot open allowed source video')
        movie=out/(ident+'__四子图.mp4')
        writer=cv2.VideoWriter(str(movie),cv2.VideoWriter_fourcc(*'mp4v'),5.,(1600,1280))
        if not writer.isOpened():raise RuntimeError('Video writer failed')
        gate=TemporalRopeGate();eng=None;latest=None;gate_result={'status':'UNKNOWN','points_xy':None}
        heat=None;last_analysis_index=None;selected_entry=None;counts={};analysis_count=render_count=forwards=0
        records=(out/'detections.jsonl').open('w',encoding='utf8');renders=(out/'render_frames.jsonl').open('w',encoding='utf8')
        preview_targets=set([0,anchor_index,min(last,anchor_index+round(3*fps)),last//2,last])
        for index in range(last+1):
            if not cap.grab():raise RuntimeError(f'Failed source grab at {index}')
            if index not in events:continue
            ok,frame=cap.retrieve()
            if not ok:raise RuntimeError(f'Failed source decode at {index}')
            if index in analysis:
                if eng is None:
                    eng=FrameEngine(anchor_job(video),anchor_time=anchor_t,threads=threads)
                    write(out/'engine_receipt.json',eng.receipt)
                tick=time.perf_counter();latest=eng.step(frame,float(times[index]))
                candidates=adapt_candidates(latest['raw_candidates'])
                gate_result=gate.update(float(times[index]),candidates,pitch_px=latest['pitch_original_px'])
                accepted=gate_result.get('accepted_candidate_index')
                selected_entry=int(candidates[accepted]['entry_index']) if accepted is not None else latest['diagnostics'].get('selected_entry')
                maps=latest.get('heatmaps_by_entry',[])
                heat=maps[selected_entry] if maps and selected_entry is not None else None
                response_path=None;response_sha=None
                if maps:
                    response_path=out/'responses'/f'{index:06d}.npz'
                    np.savez_compressed(response_path,**{f'entry_{i}':m.astype(np.float32) for i,m in enumerate(maps)},
                        bbox_xyxy=np.array([0,1050,4096,3000]),source_frame_index=index,observation_time_s=times[index])
                    response_sha=sha(response_path)
                row=dict(video_id=ident,source_frame_index=index,media_time_s=index/fps,source_time_s=float(times[index]),
                    source_anchor_frame=anchor_index,source_anchor_time_s=anchor_t,analysis_index=analysis_count,
                    processing_elapsed_s=time.perf_counter()-tick,raw_candidates=latest['raw_candidates'],gate_candidates=candidates,
                    single_frame_selected_index=0 if candidates else None,temporal=gate_result,
                    image_guard_only_selected_index=next((i for i,c in enumerate(candidates) if c['image_supported']),None),
                    heatmap_entry_index=selected_entry,response_path=str(response_path) if response_path else None,
                    response_sha256=response_sha,heat_shape=list(heat.shape) if heat is not None else None,
                    engine_status=latest['status'],diagnostics=latest['diagnostics'],seeds=latest['seeds'],
                    normalization=latest.get('normalization_by_entry'),pitch_original_px=latest['pitch_original_px'])
                line(records,row);last_analysis_index=index;analysis_count+=1
                forwards+=latest['diagnostics']['forward_count'];status=gate_result['status'];counts[status]=counts.get(status,0)+1
                line(sys.stdout,dict(video=ident,analysis=analysis_count,frame=index,status=status,seconds=round(row['processing_elapsed_s'],2)))
            if index in render_counts:
                age=None if last_analysis_index is None else float(times[index]-times[last_analysis_index])
                display_status=gate_result['status']
                reference_age=gate_result.get('reference_age_s')
                if display_status=='STALE' and reference_age is not None and reference_age+(age or 0)>1.:display_status='UNKNOWN'
                meta=dict(video_id=ident,media_time_s=index/fps,detection_time_s=None if last_analysis_index is None else last_analysis_index/fps,
                    heat_age_s=age,detection_age_s=age,source_frame_index=index,analysis_hz=2,playback_fps=5,
                    observation_time_s=float(times[index]),detection_observation_time_s=None if last_analysis_index is None else float(times[last_analysis_index]),
                    heatmap_entry_index=selected_entry,entry_count=0 if latest is None else len(latest.get('seeds',[])),
                    heat_bbox_xyxy=[0,1050,4096,3000],display_status=display_status)
                panel=render_frame(frame,heat,latest or {'raw_candidates':[],'surface_tracks':[],'seeds':[]},gate_result,meta)
                for _ in range(render_counts[index]):
                    writer.write(panel)
                    line(renders,dict(output_frame_index=render_count,source_frame_index=index,media_time_s=index/fps,
                        source_time_s=float(times[index]),last_analysis_frame=last_analysis_index,detection_age_s=age,
                        display_status=display_status,heatmap_entry_index=selected_entry,
                        current_geometry_drawn=display_status in ['ACCEPTED','REACQUIRED'],
                        reference_geometry_drawn=display_status=='STALE',
                        raw_candidate_index=gate_result.get('accepted_candidate_index') if display_status in ['ACCEPTED','REACQUIRED'] else None,
                        heat_available=heat is not None,
                        reference_total_age_s=None if reference_age is None else reference_age+(age or 0)))
                    render_count+=1
                if any(abs(index-t)<=max(1,round(fps/10)) for t in preview_targets):
                    cv2.imencode('.jpg',panel,[cv2.IMWRITE_JPEG_QUALITY,92])[1].tofile(str(out/'previews'/f'{index:06d}.jpg'))
        records.close();renders.close();cap.release();writer.release();writer=None
        summary=dict(video_id=ident,status='complete',source_path=video['canonical_path'],source_sha256=video['sha256'],
            output_path=str(movie),output_sha256=sha(movie),analysis_frames=analysis_count,forward_count=forwards,
            output_frames=render_count,output_fps=5,output_shape=[1280,1600],source_frames_traversed=last+1,
            source_nominal_fps=fps,source_duration_s=duration,output_duration_s=render_count/5,
            anchor_frame=anchor_index,anchor_time_s=anchor_t,counts=counts,wall_seconds=time.perf_counter()-started,
            future_records_used=False,training=False,devices=False,identity_accuracy_evaluated=False)
        write(out/'summary.json',summary);return summary
    except Exception:
        failure=dict(video_id=ident,status='failed',traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-started)
        write(out/'failure.json',failure);return failure
    finally:
        if cap is not None:cap.release()
        if writer is not None:writer.release()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--pilot',action='store_true')
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--threads',type=int,default=4);args=ap.parse_args()
    out=HERE/args.run;out.mkdir(exist_ok=False)
    trace=read(HERE/'source_trace/source_trace.json');videos=trace['videos']
    if args.pilot:videos=[v for v in videos if v['video_id'] in ['dual_20260923_101517__right','L_run_20260923_01__left','dual_20260923_101517__left']]
    sources=['run_videos.py','frame_engine/engine.py','temporal/causal_gate.py','rendering/render_panels.py']
    protocol=dict(version='IR-V1.0.0',baseline='IR-P1.0.0',command=sys.argv,python=sys.executable,
        source_trace_sha256=sha(HERE/'source_trace/source_trace.json'),videos=[v['video_id'] for v in videos],
        analysis_hz=2,playback_fps=5,workers=args.workers,torch_threads_per_worker=args.threads,
        gate=asdict(GateConfig()),minimum_current_raw_supported_length_px=80,
        input_model='original B750, no H7',scene_initialization='earliest dataset anchor only; no processing before anchor',
        scene_update='static machinery causal LK and current image support; not fresh full F3/SAM2',
        heat_values='raw float32, fixed display 0..1, never filled from temporal paths',
        sources_sha256={s:sha(HERE/s) for s in sources},pilot=args.pilot,
        no_hardware=True,no_reserved_sources=True,no_training=True,output_is_offline_original_speed=True)
    write(out/'protocol.json',protocol);started=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks={pool.submit(run_one,v,str(out),args.pilot,args.threads):v['video_id'] for v in videos}
        for f in as_completed(tasks):
            result=f.result();results.append(result);write(out/'progress_summary.json',results)
            line(sys.stdout,dict(completed_video=tasks[f],status=result['status'],completed=len(results),total=len(videos)))
    summary=dict(planned_videos=len(videos),completed_videos=sum(r['status']=='complete' for r in results),
        failures=[r for r in results if r['status']!='complete'],wall_seconds=time.perf_counter()-started,results=results,
        protocol_sha256=sha(out/'protocol.json'))
    write(out/'summary.json',summary)
    return 0 if not summary['failures'] else 1
if __name__=='__main__':raise SystemExit(main())
