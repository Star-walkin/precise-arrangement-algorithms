"""Every-source-frame offline rerun; parallel raw measurement, causal final gate."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(name,'2')
import sys,json,time,gzip,hashlib,math,argparse,traceback,copy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from engine import FrameEngine,anchor_job,cv2,np,V1
sys.path.insert(0,str(V1))
from temporal import TemporalRopeGate,GateConfig
from run_videos import adapt_candidates
spec=__import__('importlib.util',fromlist=['']).spec_from_file_location('_v2_renderer',HERE/'render_panels.py')
renderer=__import__('importlib.util',fromlist=['']).module_from_spec(spec);spec.loader.exec_module(renderer)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def line(f,d):f.write(json.dumps(d,ensure_ascii=False,separators=(',',':'))+'\n');f.flush()
def config(v):
    n=max(5,int(math.ceil(.5*v['nominal_fps'])))
    return GateConfig(history_frames=n,reacquire_frames=max(3,n))

def measure(job):
    v=job['video'];out=Path(job['run_dir'])/v['video_id'];chunk=out/'chunks'/f"{job['start']:06d}"
    chunk.mkdir(parents=True,exist_ok=False);started=time.perf_counter();cap=None
    try:
        times=np.asarray(read(v['frame_times_path'])['observation_times_s']);fps=v['nominal_fps']
        first=max(0,job['start']-int(math.ceil(fps))) # one second causal entry warmup
        cap=cv2.VideoCapture(v['canonical_path']);cap.set(cv2.CAP_PROP_POS_FRAMES,first)
        eng=FrameEngine(anchor_job(v),float(times[first]),threads=job['threads'])
        write(chunk/'engine_receipt.json',eng.receipt)
        records=gzip.open(chunk/'measurements.jsonl.gz','wt',encoding='utf8',compresslevel=1)
        forwards=detected=0;states={};previous_signature=None;changes=0;maximum_seconds=0
        for index in range(first,job['stop']):
            ok,bgr=cap.read()
            if not ok:raise RuntimeError(f'Failed decode {index}')
            result=eng.step(bgr,float(times[index]));diag=result['diagnostics']
            if diag['fresh_gray_detection_calls']!=index-first+1:raise ValueError('Gray detection skipped or repeated')
            if index<job['start']:continue
            maps=result['heatmaps_by_entry'];heat_paths=[]
            for ei,heat in enumerate(maps):
                # Only display caching is resized/quantized; detection used the
                # original float975x2048 response and its hash is recorded below.
                display=cv2.resize(heat,(800,381),interpolation=cv2.INTER_LINEAR)
                path=out/'display_heat'/f'{index:06d}_E{ei}.png'
                cv2.imencode('.png',np.rint(np.clip(display,0,1)*65535).astype(np.uint16),[cv2.IMWRITE_PNG_COMPRESSION,1])[1].tofile(str(path))
                heat_paths.append(str(path))
            keyframe=index % max(1,int(round(5*fps)))==0
            if maps and keyframe:
                np.savez_compressed(out/'float_keyframes'/f'{index:06d}.npz',**{f'entry_{i}':a for i,a in enumerate(maps)},
                    bbox_xyxy=np.array([0,1050,4096,3000]),source_frame_index=index)
            signature=diag['gray_center_sha256'];changes+=int(previous_signature is not None and signature!=previous_signature);previous_signature=signature
            status=result['status'];states[status]=states.get(status,0)+1
            detected+=int(bool(result['surface_tracks']));forwards+=diag['forward_count'];maximum_seconds=max(maximum_seconds,diag['seconds'])
            row=dict(source_frame_index=index,source_time_s=float(times[index]),media_time_s=index/fps,
                raw_candidates=result['raw_candidates'],surface_tracks=result['surface_tracks'],seeds=result['seeds'],
                pitch_original_px=result['pitch_original_px'],engine_status=status,diagnostics=diag,
                gate_candidates=adapt_candidates(result['raw_candidates']),heat_paths=heat_paths,
                float_heat_sha256=[hashlib.sha256(h.astype(np.float32).tobytes()).hexdigest() for h in maps],
                heat_selected_default=diag['selected_entry'],gray_measurement_frame=index,gray_measurement_time_s=float(times[index]),
                chunk_start=job['start'],warmup_start=first,current_centers_used_for_candidates=True)
            line(records,row)
            if (index-job['start'])%30==0:line(sys.stdout,dict(phase='measure',video=v['video_id'],frame=index,end=job['stop'],centers=len(result['surface_tracks']),seconds=round(diag['seconds'],2)))
        records.close();cap.release()
        receipt=dict(status='complete',start=job['start'],stop=job['stop'],count=job['stop']-job['start'],
            warmup_start=first,fresh_gray_detections=job['stop']-job['start'],frames_with_gray_centers=detected,
            gray_changes_within_chunk=changes,forward_count=forwards,states=states,wall_seconds=time.perf_counter()-started,
            measurement_sha256=sha(chunk/'measurements.jsonl.gz'),maximum_frame_seconds=maximum_seconds)
        write(chunk/'summary.json',receipt);return dict(video_id=v['video_id'],chunk=str(chunk),**receipt)
    except Exception:
        receipt=dict(status='failed',video_id=v['video_id'],start=job['start'],traceback=traceback.format_exc());write(chunk/'failure.json',receipt);return receipt
    finally:
        if cap is not None:cap.release()

def render_video(v,run_dir,frame_count):
    cv2.setNumThreads(1);out=Path(run_dir)/v['video_id'];fps=v['nominal_fps'];cap=cv2.VideoCapture(v['canonical_path'])
    movie=out/(v['video_id']+'__逐帧四子图.mp4')
    writer=cv2.VideoWriter(str(movie),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1600,1280))
    if not writer.isOpened():raise RuntimeError('VideoWriter failed')
    gate=TemporalRopeGate(config(v));counts={};next_frame=0;started=time.perf_counter();changed=0;last_signature=None
    selected_other_entry=0;stats=(out/'detections.jsonl.gz');records=gzip.open(stats,'wt',encoding='utf8',compresslevel=1)
    previews=set([0,min(frame_count-1,round(3*fps)),frame_count//2,frame_count-1])
    for chunk in sorted((out/'chunks').iterdir()):
        s=read(chunk/'summary.json');assert s['status']=='complete'
        with gzip.open(chunk/'measurements.jsonl.gz','rt',encoding='utf8') as f:
            for text in f:
                row=json.loads(text);index=row['source_frame_index'];assert index==next_frame
                assert row['gray_measurement_frame']==index and row['current_centers_used_for_candidates']
                ok,bgr=cap.read()
                if not ok:raise RuntimeError(f'Decode for render failed {index}')
                temporal=gate.update(row['source_time_s'],row['gate_candidates'],pitch_px=row['pitch_original_px'])
                ai=temporal['accepted_candidate_index'];ei=row['gate_candidates'][ai]['entry_index'] if ai is not None else row['heat_selected_default']
                heat=None
                if ei is not None and row['heat_paths']:
                    heat=cv2.imdecode(np.fromfile(row['heat_paths'][ei],np.uint8),cv2.IMREAD_UNCHANGED).astype(np.float32)/65535
                engine=dict(raw_candidates=row['raw_candidates'],surface_tracks=row['surface_tracks'],seeds=row['seeds'])
                meta=dict(video_id=v['video_id'],media_time_s=index/fps,detection_time_s=index/fps,heat_age_s=0.,detection_age_s=0.,
                    source_frame_index=index,analysis_hz=fps,playback_fps=fps,heatmap_entry_index=ei,
                    heat_bbox_xyxy=[0,1050,4096,3000],display_status=temporal['status'])
                panel=renderer.render_frame(bgr,heat,engine,temporal,meta);writer.write(panel)
                if index in previews:cv2.imencode('.jpg',panel,[cv2.IMWRITE_JPEG_QUALITY,92])[1].tofile(str(out/'previews'/f'{index:06d}.jpg'))
                status=temporal['status'];counts[status]=counts.get(status,0)+1
                sig=row['diagnostics']['gray_center_sha256'];changed+=int(last_signature is not None and sig!=last_signature);last_signature=sig
                # Rejects must not become history, and accepted geometry must be
                # an unchanged current-frame raw candidate, not a cached path.
                assert all(t<row['source_time_s'] for t in temporal['consulted_accepted_history_times_s'])
                if temporal['accepted']:
                    assert np.array_equal(temporal['points_xy'],row['raw_candidates'][ai]['points_xy'])
                    assert row['gate_candidates'][ai]['image_supported']
                    selected_other_entry+=int(ei!=row['heat_selected_default'])
                else:assert not temporal['history_updated']
                line(records,dict(source_frame_index=index,source_time_s=row['source_time_s'],gray_measurement_frame=index,
                    gray_center_sha256=sig,gray_centers=len(row['surface_tracks']),temporal=temporal,heatmap_entry_index=ei,
                    heat_available=heat is not None,current_surface_used_for_candidates=True,
                    single_frame_selected_index=0 if row['raw_candidates'] else None,
                    image_guard_only_selected_index=next((i for i,c in enumerate(row['gate_candidates']) if c['image_supported']),None),
                    raw_candidates=row['raw_candidates'],display_source_frame=index,display_heat_age_s=0.,gray_age_s=0.))
                next_frame+=1
    cap.release();writer.release();records.close();assert next_frame==frame_count
    summary=dict(video_id=v['video_id'],status='complete',source_frames=frame_count,detected_frames=next_frame,output_frames=next_frame,
        output_fps=fps,output_shape=[1280,1600],source_duration_s=frame_count/fps,output_duration_s=next_frame/fps,
        gray_changes=changed,states=counts,temporal_config=asdict(config(v)),output_path=str(movie),output_sha256=sha(movie),
        wall_seconds=time.perf_counter()-started,selected_different_heat_entry=selected_other_entry,processing_is_offline=True)
    write(out/'summary.json',summary);line(sys.stdout,dict(phase='render_complete',video=v['video_id'],frames=next_frame));return summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--pilot',action='store_true')
    ap.add_argument('--workers',type=int,default=8);ap.add_argument('--threads',type=int,default=2);args=ap.parse_args()
    out=HERE/args.run;out.mkdir(exist_ok=False);started=time.perf_counter()
    trace=read(V1/'source_trace/source_trace.json');videos=trace['videos']
    if args.pilot:videos=[v for v in videos if v['video_id'] in ['dual_20260923_101517__right','L_run_20260923_01__left','dual_20260923_101517__left']]
    counts={v['video_id']:min(15,v['frame_count_metadata']) if args.pilot else v['frame_count_metadata'] for v in videos}
    sources=[HERE/'run.py',HERE/'engine.py',HERE/'fresh_surface.py',HERE/'render_panels.py',V1/'frame_engine/engine.py',V1/'temporal/causal_gate.py']
    write(out/'protocol.json',dict(version='IR-V2.0.0',command=sys.argv,python=sys.executable,frame_policy='EVERY original source frame, no detection subsampling, no held heat/centers',
        workers=args.workers,torch_threads=args.threads,video_ids=list(counts),planned_source_frames=sum(counts.values()),
        source_trace_sha256=sha(V1/'source_trace/source_trace.json'),sources_sha256={str(p):sha(p) for p in sources},
        chunk_seconds=20,warmup_seconds=1,temporal_pass='source-order global per video; never uses future raw results',
        scene_calibration='existing allowed offline fixed ROI and broad C3 direction only; not future target paths or gray centers',
        detector='fresh frame LCN, tangent integration, double shoulders, ordered finite grooves, adjacent boundary midpoint; lighter observer, not original full F3 fusion',
        heat_model='original frozen B750 unchanged',float_keyframes_seconds=5,display_heat_cache='800x381 uint16 fixed0..1; original full float used for detection',
        temporal_configs={v['video_id']:asdict(config(v)) for v in videos},training=False,devices=False,reserved_sources=False))
    jobs=[]
    for v in videos:
        folder=out/v['video_id'];folder.mkdir()
        for name in ['chunks','display_heat','float_keyframes','previews']:(folder/name).mkdir()
        n=counts[v['video_id']];size=max(1,round(20*v['nominal_fps']))
        for start in range(0,n,size):jobs.append(dict(video=v,run_dir=str(out),start=start,stop=min(n,start+size),threads=args.threads))
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks=[pool.submit(measure,j) for j in jobs]
        for f in as_completed(tasks):
            r=f.result();results.append(r);write(out/'measurement_progress.json',results)
            line(sys.stdout,dict(phase='chunk_complete',completed=len(results),total=len(jobs),status=r['status']))
    write(out/'measurement_summary.json',dict(results=results,wall_seconds=time.perf_counter()-started))
    if any(r['status']!='complete' for r in results):raise RuntimeError('Measurement failure; preserved outputs, no fake complete video')
    videos_done=[]
    with ProcessPoolExecutor(max_workers=min(4,args.workers)) as pool:
        tasks=[pool.submit(render_video,v,str(out),counts[v['video_id']]) for v in videos]
        for f in as_completed(tasks):videos_done.append(f.result());write(out/'render_progress.json',videos_done)
    write(out/'summary.json',dict(status='complete',videos=len(videos_done),frames=sum(v['output_frames'] for v in videos_done),
        fresh_gray_detections=sum(r['count'] for r in results),new_B_forwards=sum(r['forward_count'] for r in results),
        source_frames_planned=sum(counts.values()),results=videos_done,wall_seconds=time.perf_counter()-started,
        protocol_sha256=sha(out/'protocol.json'),training=False,devices=False))
if __name__=='__main__':main()
