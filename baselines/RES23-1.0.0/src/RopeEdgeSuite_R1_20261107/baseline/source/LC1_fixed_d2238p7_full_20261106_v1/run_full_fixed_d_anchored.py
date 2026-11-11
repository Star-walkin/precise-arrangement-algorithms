"""Full current-frame S1 trial with spatially anchored E1 and independent F3 workers.

S1 remains strictly chronological. F3 tasks depend only on that task's image,
current S1 intersection, and frozen static search ROI. FIFO consumption keeps
the gate causal even when later tasks happen to finish earlier.
"""
from pathlib import Path
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import argparse,csv,hashlib,json,math,sys,time,types
import cv2,numpy as np

ROOT=Path(__file__).resolve().parent
from run_full_fixed_d import BASE,JOINT,SOURCE,Reacquire,dump,calibration,timestamps,sha,clean
from lc1_surface import SurfaceObserver
from five_pitch_gate import FivePitchGate

_observer=None
def init_surface_worker(mask,pitch):
    global _observer
    cv2.setNumThreads(1)
    _observer=SurfaceObserver(mask,pitch)

def current_surface(image,y,frame_index):
    tick=time.perf_counter()
    pixel=hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
    result=_observer.step(image,d=2238.7,active_y=y)
    return dict(frame_index=frame_index,source_pixel_sha256=pixel,intersection_y_px=y,
                surface=result,surface_ms=1000*(time.perf_counter()-tick))

def run(stop=None,workers=3):
    if not 1<=workers<=4:raise ValueError('workers must be 1..4')
    anchor_path=ROOT/'live_s1_tracker_anchored.py'
    tracker_source=anchor_path.read_bytes()
    tracker_digest=hashlib.sha256(tracker_source).hexdigest()
    base_digest=sha(ROOT/'live_s1_tracker.py')
    mod=types.ModuleType('_exact_anchored_tracker_snapshot');mod.__file__=str(anchor_path)
    exec(compile(tracker_source,str(anchor_path),'exec'),mod.__dict__)
    plan=json.loads((ROOT/'run_plan.json').read_text(encoding='utf-8'))
    previous=json.loads((JOINT/'collection_receipt.json').read_text(encoding='utf-8'))
    frozen=previous['detector_freeze']['detector_hashes']
    for name,digest in frozen.items():
        if sha(BASE/name)!=digest:raise RuntimeError('Frozen original module changed: '+name)
    source_hash=sha(SOURCE)
    if source_hash!=plan['expected_source_sha256']:raise RuntimeError('Source video hash changed')
    cap=cv2.VideoCapture(str(SOURCE))
    if not cap.isOpened():raise RuntimeError('Cannot open permitted full source')
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
    if n!=plan['expected_frames']:raise RuntimeError('Unexpected source frame count')
    end=n if stop is None else min(stop,n)
    times,basis,time_csv=timestamps(SOURCE,'left',n,fps);origin=times[0]
    mask,pitch,scene=calibration(SOURCE.parent.name,'left')
    reacquire=Reacquire(mask,pitch)
    tracker=mod.LiveS1Tracker(mask,reacquire_callback=reacquire)
    gate=FivePitchGate(threshold_pitch=5.,fixed_d_original_px=2238.7)
    out=ROOT/('analysis_anchored' if stop is None else 'pilot_anchored_'+tracker_digest[:8])
    out.mkdir(exist_ok=True);(out/'frames').mkdir(exist_ok=True)
    binding=dict(source_sha256=source_hash,source_frames=n,source_fps=fps,frame_stop_exclusive=end,
        fixed_d_original_px=2238.7,entry_threshold_pitch=5.,surface_coordinates='original_unmirrored_xy',
        detector_hashes=frozen,tracker_sha256=tracker_digest,tracker_base_sha256=base_digest,
        gate_sha256=sha(ROOT/'five_pitch_gate.py'),mask_sha256=scene['mask_sha256'],pitch_px=pitch,
        script_sha256=sha(Path(__file__)),reacquisition_module_sha256=sha(ROOT/'run_full_fixed_d.py'),
        workers=workers,max_inflight_F3=workers*2,seed_length_px=120.,minimum_S1_support_span_px=400.)
    if (out/'binding.json').exists():
        if json.loads((out/'binding.json').read_text(encoding='utf-8'))!=binding:
            raise RuntimeError('Existing run has other inputs/code')
        if list((out/'frames').glob('frame_*.json')):raise RuntimeError('Preserve existing run before retry')
    dump(out/'binding.json',binding)
    (out/'tracker_source_snapshot.py').write_bytes(tracker_source)
    (out/'tracker_base_source_snapshot.py').write_bytes((ROOT/'live_s1_tracker.py').read_bytes())
    (out/'runner_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (out/'gate_source_snapshot.py').write_bytes((ROOT/'five_pitch_gate.py').read_bytes())
    (out/'reacquisition_source_snapshot.py').write_bytes((ROOT/'run_full_fixed_d.py').read_bytes())
    source_stat=SOURCE.stat();started=time.perf_counter();rows=[];events=[];pending=deque()
    fields=['frame_index','time_s','source_nominal_time_s','s1_valid','S1_calls','S1_source','A','B','C',
        'fixed_d_original_px','intersection_y_px','edge_y_px','F3_count_valid','F3_continuous_N',
        'F3_geometric_candidate_N_not_gate_input','current_region','emit_distance','display_distance_N',
        'armed','first_inside_confirmed_this_frame','observed_grooves','local_pitch_px',
        'tracker_ms','surface_ms','total_ms','gate_reason','F3_reasons','anchor_x_px','seed_x_drift_px']
    with ProcessPoolExecutor(max_workers=workers,initializer=init_surface_worker,initargs=(mask,pitch)) as pool:
        with (out/'measurements.csv').open('w',newline='',encoding='utf-8-sig') as csvfile:
            writer=csv.DictWriter(csvfile,fieldnames=fields);writer.writeheader()
            def consume():
                meta,future=pending.popleft();work=future.result()
                i=meta['frame_index'];t=meta['time_s'];track=meta['S1_tracker']
                if (work['frame_index']!=i or work['source_pixel_sha256']!=meta['source_pixel_sha256']
                    or work['intersection_y_px']!=meta['intersection_y_px']):
                    raise RuntimeError('F3 worker source binding mismatch')
                surface=work['surface'];valid=track.get('valid') is True
                y=meta['intersection_y_px'];abc=track.get('current_line_abc') if valid else None
                prediction=gate.update(t,surface,s1_valid=valid,intersection_y_px=y,frame_index=i)
                record=dict(meta,surface=surface,prediction=prediction,surface_ms=work['surface_ms'],
                    total_ms=1000*(time.perf_counter()-meta.pop('_tick')))
                record.pop('_tick',None)
                dump(out/'frames'/f'frame_{i:06d}.json',record)
                anchor=track.get('diagnostics',{}).get('spatial_anchor',{})
                row=dict(frame_index=i,time_s=t,source_nominal_time_s=i/fps,s1_valid=valid,
                    S1_calls=int(track.get('diagnostics',{}).get('S1_calls',0)),S1_source=track.get('source'),
                    A=None if abc is None else abc[0],B=None if abc is None else abc[1],C=None if abc is None else abc[2],
                    fixed_d_original_px=2238.7,intersection_y_px=y,edge_y_px=surface.get('edge_y'),
                    F3_count_valid=surface.get('count_valid') is True,F3_continuous_N=surface.get('continuous_pitch_distance'),
                    F3_geometric_candidate_N_not_gate_input=surface.get('continuous_pitch_distance_candidate'),
                    current_region=prediction['current_region'],emit_distance=prediction['emit_distance_estimate'],
                    display_distance_N=prediction['current_N'] if prediction['emit_distance_estimate'] else None,
                    armed=prediction['armed'],first_inside_confirmed_this_frame=prediction['first_inside_confirmed_this_frame'],
                    observed_grooves=surface.get('count_observed_grooves'),local_pitch_px=surface.get('local_pitch_px'),
                    tracker_ms=record['tracker_ms'],surface_ms=work['surface_ms'],total_ms=record['total_ms'],
                    gate_reason=prediction['reason'],F3_reasons='|'.join(surface.get('reasons',[])),
                    anchor_x_px=anchor.get('x_original'),seed_x_drift_px=anchor.get('seed_start_x_drift_px'))
                writer.writerow(clean(row));csvfile.flush();rows.append(row)
                if prediction['inside_confirmation_event']:
                    events.append(dict(frame_index=i,time_s=t,N=prediction['current_N'],
                        event='current_valid_inside_confirmation',confirmation_interval_s=prediction['entry_observation_interval_s']))
                if i%10==0 or prediction['first_inside_confirmed_this_frame']:
                    progress=dict(status='running',processed_frames=i+1,total_requested=end,last_source_frame=i,
                        last_time_s=t,current_region=prediction['current_region'],current_N=prediction['current_N'],
                        S1_valid_frames=sum(r['s1_valid'] for r in rows),state_counts=dict(Counter(r['current_region'] for r in rows)),
                        first_inside_confirmation_frame=gate.first_inside_confirmation_frame,wall_seconds=time.perf_counter()-started)
                    dump(out/'progress.json',progress);print(json.dumps(progress,ensure_ascii=False),flush=True)
            for i in range(end):
                ok,image=cap.read()
                if not ok or abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-(i+1))>.1:
                    raise RuntimeError(f'Invalid sequential frame {i}')
                tick=time.perf_counter();t=float(times[i]-origin)
                track=tracker.step(image,t,i);tracker_ms=1000*(time.perf_counter()-tick)
                abc=track.get('current_line_abc') if track.get('valid') is True else None;y=None
                if abc is not None and abs(float(abc[1]))>.05:
                    y=float(-(abc[0]*2238.7+abc[2])/abc[1])
                    if not math.isfinite(y):y=None
                meta=dict(frame_index=i,time_s=t,source_nominal_time_s=i/fps,
                    source_pixel_sha256=hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest(),
                    fixed_d_original_px=2238.7,fixed_d_source='user_provisional_parameter_not_automatic_calibration',
                    S1_tracker=track,intersection_y_px=y,tracker_ms=tracker_ms,_tick=tick)
                pending.append((meta,pool.submit(current_surface,image,y,i)))
                if len(pending)>=workers*2:consume()
            while pending:consume()
    cap.release();after=SOURCE.stat()
    assert (source_stat.st_size,source_stat.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    assert sha(anchor_path)==tracker_digest and sha(ROOT/'live_s1_tracker.py')==base_digest
    assert sha(Path(__file__))==binding['script_sha256']
    assert sha(ROOT/'five_pitch_gate.py')==binding['gate_sha256']
    assert sha(ROOT/'run_full_fixed_d.py')==binding['reacquisition_module_sha256']
    receipt=dict(status='completed',source_video=str(SOURCE),source_sha256=source_hash,source_stable=True,
        source_frames=n,source_fps=fps,processed_frames=len(rows),frame_stride=1,source_frame_range_exclusive=[0,end],
        time_basis=basis,time_csv=time_csv,current_frame_S1_requires_verified_E1=True,
        actual_S1_core_calls=sum(r['S1_calls'] for r in rows),frames_with_S1_core_call=sum(r['S1_calls']>0 for r in rows),
        frames_without_verified_seed_or_S1_call=sum(r['S1_calls']==0 for r in rows),
        S1_valid_frames=sum(r['s1_valid'] for r in rows),current_F3_calls=len(rows),
        F3_valid_count_frames=sum(r['F3_count_valid'] for r in rows),prediction_frames=sum(r['emit_distance'] for r in rows),
        state_counts=dict(Counter(r['current_region'] for r in rows)),
        first_inside_confirmation_frame=gate.first_inside_confirmation_frame,
        first_inside_confirmation_time_s=gate.first_inside_confirmation_time,inside_confirmation_events=events,
        fixed_d_original_px=2238.7,d_automatically_calibrated=False,integer_count_phase_calibrated=False,
        no_future_frames_used=True,no_unknown_distance_hold_as_current=True,
        causal_meaning='estimate i uses no parameters or measurements from frames later than i; bounded F3 prefetch is enabled',
        parallel_F3_independent_current_frame_only=True,gate_consumption='FIFO_source_frame_order',
        actual_P1_reacquisition_forwards=reacquire.live_model_forwards,reused_current_frame_P1_seed_count=reacquire.cache_seeds,
        reacquisition_callback_calls=reacquire.calls,
        median_tracker_ms=float(np.median([r['tracker_ms'] for r in rows])),
        median_surface_ms=float(np.median([r['surface_ms'] for r in rows])),
        median_pipeline_latency_ms=float(np.median([r['total_ms'] for r in rows])),
        wall_seconds=time.perf_counter()-started,static_scene_calibration=scene,binding=binding,
        measurements_sha256=sha(out/'measurements.csv'))
    dump(out/'receipt.json',receipt);dump(out/'progress.json',dict(status='completed',processed_frames=len(rows)))
    print(json.dumps({k:receipt[k] for k in ('status','processed_frames','S1_valid_frames','prediction_frames','state_counts','wall_seconds')},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pilot-stop',type=int);p.add_argument('--workers',type=int,default=3)
    args=p.parse_args();run(args.pilot_stop,args.workers)
