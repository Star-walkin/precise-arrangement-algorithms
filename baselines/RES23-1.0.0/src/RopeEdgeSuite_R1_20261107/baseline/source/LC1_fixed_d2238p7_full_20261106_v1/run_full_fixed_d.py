"""Full source-rate current S1/F3 trial at the user's fixed original x=d.

Analysis and rendering are separate: persisted current-frame records can be
reviewed/re-encoded without rerunning detection. No line/distance is held as
a current estimate across failed frames. No source frame is skipped.
"""
from pathlib import Path
import argparse,csv,hashlib,json,math,sys,time,types
from collections import Counter
import cv2,numpy as np

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'LC1_locked_column_20261105_v1'
JOINT=ROOT.parent/'LC1_joint_d_20261105_v1'
SCREEN=ROOT.parent/'LC1_screened_d_20261106_v1'
sys.path.insert(0,str(BASE))
from lc1_video import calibration,timestamps,sha,clean
from lc1_surface import SurfaceObserver
from five_pitch_gate import FivePitchGate

SOURCE=Path('E:/study/program/济宁/双相机同步/dual_20260924_111500/left_DA9627065.avi')


def dump(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


class Reacquire:
    def __init__(self,mask,pitch):
        self.mask,self.pitch=mask,pitch
        self.adapter=None;self.previous_live_frame=-10000
        self.calls=0;self.cache_seeds=0;self.live_model_forwards=0;self.cooldown_skips=0
        self.paths={}
        for folder in (JOINT/'observations',SCREEN/'scout_detections',SCREEN/'far_window_detections'):
            for path in folder.glob('frame_*.json'):
                self.paths.setdefault(int(path.stem.split('_')[-1]),path)

    def __call__(self,image,t,frame_index):
        self.calls+=1;pixel=hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
        path=self.paths.get(frame_index)
        if path:
            row=json.loads(path.read_text(encoding='utf-8'))
            if row['source_pixel_sha256']!=pixel:raise RuntimeError('Current cached seed source pixel mismatch')
            seed=row['P1_H6'].get('seed_xy')
            if seed:
                self.cache_seeds+=1
                return dict(seed_xy=seed,seed_source='same_frame_source_pixel_bound_cached_P1_A7_seed_only',
                    source_pixel_sha256=pixel,model_forward_count=0,
                    diagnostics=dict(cached_observation=str(path),cached_observation_sha256=sha(path),
                        old_line_or_path_used_as_measurement=False))
        if frame_index-self.previous_live_frame<10:
            self.cooldown_skips+=1
            return None
        self.previous_live_frame=frame_index
        if self.adapter is None:
            from lc1_p1_adapter import P1H6Adapter
            self.adapter=P1H6Adapter(self.mask,self.pitch,BASE/'supplied_assets/step_0750.pt',threads=2,
                camera='left',surface_mode='entry_band')
        p1=self.adapter.step(image);n=int(p1.get('model_forward_count',0));self.live_model_forwards+=n
        return dict(seed_xy=p1.get('seed_xy'),seed_source='current_frame_live_P1_A7_reacquisition_seed_only',
            source_pixel_sha256=pixel,model_forward_count=n,
            diagnostics=dict(P1_status=p1.get('status'),P1_reasons=p1.get('reasons'),
                P1_source=p1.get('source'),whole_path_not_used_to_calibrate_d=True))


def run(stop=None):
    tracker_source=(ROOT/'live_s1_tracker.py').read_bytes()
    tracker_digest=hashlib.sha256(tracker_source).hexdigest()
    module=types.ModuleType('_exact_live_s1_tracker_snapshot');module.__file__=str(ROOT/'live_s1_tracker.py')
    exec(compile(tracker_source,module.__file__,'exec'),module.__dict__)
    LiveS1Tracker=module.LiveS1Tracker
    plan=json.loads((ROOT/'run_plan.json').read_text(encoding='utf-8'))
    previous=json.loads((JOINT/'collection_receipt.json').read_text(encoding='utf-8'))
    frozen=previous['detector_freeze']['detector_hashes']
    for name,digest in frozen.items():
        if sha(BASE/name)!=digest:raise RuntimeError('Frozen original module changed: '+name)
    source_hash=sha(SOURCE)
    if source_hash!=plan['expected_source_sha256']:raise RuntimeError('Source video hash changed')
    cap=cv2.VideoCapture(str(SOURCE))
    if not cap.isOpened():raise RuntimeError('Cannot open permitted full development video')
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
    if n!=plan['expected_frames']:raise RuntimeError('Video is incomplete or has unexpected frame count')
    end=n if stop is None else min(stop,n)
    times,basis,time_csv=timestamps(SOURCE,'left',n,fps);origin=times[0]
    mask,pitch,scene=calibration(SOURCE.parent.name,'left')
    reacquire=Reacquire(mask,pitch)
    tracker=LiveS1Tracker(mask,reacquire_callback=reacquire)
    observer=SurfaceObserver(mask,pitch)
    gate=FivePitchGate(threshold_pitch=5.,fixed_d_original_px=2238.7)
    out=ROOT/('analysis' if stop is None else 'pilot_'+tracker_digest[:8]);out.mkdir(exist_ok=True)
    (out/'frames').mkdir(exist_ok=True)
    (out/'tracker_source_snapshot.py').write_bytes(tracker_source)
    binding=dict(source_sha256=source_hash,source_frames=n,source_fps=fps,frame_stop_exclusive=end,
        fixed_d_original_px=2238.7,entry_threshold_pitch=5.,surface_coordinates='original_unmirrored_xy',
        detector_hashes=frozen,tracker_sha256=tracker_digest,gate_sha256=sha(ROOT/'five_pitch_gate.py'),
        mask_sha256=scene['mask_sha256'],pitch_px=pitch,script_sha256=sha(Path(__file__)))
    if (out/'binding.json').exists():
        old=json.loads((out/'binding.json').read_text(encoding='utf-8'))
        if old!=binding:raise RuntimeError('Existing run belongs to other inputs/code; preserve it before a new run')
    dump(out/'binding.json',binding)
    source_stat=SOURCE.stat();started=time.perf_counter();rows=[];events=[]
    fields=['frame_index','time_s','source_nominal_time_s','s1_valid','S1_calls','S1_source','A','B','C',
        'fixed_d_original_px','intersection_y_px','edge_y_px','F3_count_valid','F3_continuous_N',
        'F3_geometric_candidate_N_not_gate_input','current_region','emit_distance','display_distance_N',
        'armed','first_inside_confirmed_this_frame','observed_grooves','local_pitch_px',
        'tracker_ms','surface_ms','total_ms','gate_reason','F3_reasons']
    with (out/'measurements.csv').open('w',newline='',encoding='utf-8-sig') as csvfile:
        writer=csv.DictWriter(csvfile,fieldnames=fields);writer.writeheader()
        for i in range(end):
            ok,image=cap.read()
            if not ok or abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-(i+1))>.1:raise RuntimeError(f'Invalid sequential frame {i}')
            tick=time.perf_counter();t=float(times[i]-origin)
            track=tracker.step(image,t,i);tracker_ms=1000*(time.perf_counter()-tick)
            valid=track.get('valid') is True;abc=track.get('current_line_abc') if valid else None
            y=None
            if abc is not None and abs(float(abc[1]))>.05:
                y=float(-(abc[0]*2238.7+abc[2])/abc[1])
                if not math.isfinite(y):y=None
            st=time.perf_counter();surface=observer.step(image,d=2238.7,active_y=y)
            surface_ms=1000*(time.perf_counter()-st)
            prediction=gate.update(t,surface,s1_valid=valid,intersection_y_px=y,frame_index=i)
            record=dict(frame_index=i,time_s=t,source_nominal_time_s=i/fps,
                source_pixel_sha256=hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest(),
                fixed_d_original_px=2238.7,fixed_d_source='user_provisional_parameter_not_automatic_calibration',
                S1_tracker=track,surface=surface,prediction=prediction,intersection_y_px=y,
                tracker_ms=tracker_ms,surface_ms=surface_ms,total_ms=1000*(time.perf_counter()-tick))
            dump(out/'frames'/f'frame_{i:06d}.json',record)
            row=dict(frame_index=i,time_s=t,source_nominal_time_s=i/fps,s1_valid=valid,
                S1_calls=int(track.get('diagnostics',{}).get('S1_calls',0)),
                S1_source=track.get('source'),A=None if abc is None else abc[0],B=None if abc is None else abc[1],C=None if abc is None else abc[2],
                fixed_d_original_px=2238.7,intersection_y_px=y,edge_y_px=surface.get('edge_y'),
                F3_count_valid=surface.get('count_valid') is True,F3_continuous_N=surface.get('continuous_pitch_distance'),
                F3_geometric_candidate_N_not_gate_input=surface.get('continuous_pitch_distance_candidate'),
                current_region=prediction['current_region'],emit_distance=prediction['emit_distance_estimate'],
                display_distance_N=prediction['current_N'] if prediction['emit_distance_estimate'] else None,
                armed=prediction['armed'],first_inside_confirmed_this_frame=prediction['first_inside_confirmed_this_frame'],
                observed_grooves=surface.get('count_observed_grooves'),local_pitch_px=surface.get('local_pitch_px'),
                tracker_ms=tracker_ms,surface_ms=surface_ms,total_ms=record['total_ms'],gate_reason=prediction['reason'],
                F3_reasons='|'.join(surface.get('reasons',[])))
            writer.writerow(clean(row));csvfile.flush();rows.append(row)
            if prediction['inside_confirmation_event']:
                events.append(dict(frame_index=i,time_s=t,N=prediction['current_N'],
                    event='current_valid_inside_confirmation',confirmation_interval_s=prediction['entry_observation_interval_s']))
            if i%10==0 or prediction['first_inside_confirmed_this_frame']:
                progress=dict(status='running',processed_frames=i+1,total_requested=end,last_source_frame=i,
                    last_time_s=t,current_region=prediction['current_region'],current_N=prediction['current_N'],
                    S1_valid_frames=sum(r['s1_valid'] for r in rows),state_counts=dict(Counter(r['current_region'] for r in rows)),
                    first_inside_confirmation_frame=gate.first_inside_confirmation_frame,wall_seconds=time.perf_counter()-started)
                dump(out/'progress.json',progress)
                print(json.dumps(progress,ensure_ascii=False),flush=True)
    cap.release()
    after=SOURCE.stat();assert (source_stat.st_size,source_stat.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    receipt=dict(status='completed',source_video=str(SOURCE),source_sha256=source_hash,source_stable=True,
        source_frames=n,source_fps=fps,processed_frames=len(rows),frame_stride=1,source_frame_range_exclusive=[0,end],
        time_basis=basis,time_csv=time_csv,current_frame_S1_requires_verified_E1=True,
        actual_S1_core_calls=sum(r['S1_calls'] for r in rows),
        frames_with_S1_core_call=sum(r['S1_calls']>0 for r in rows),
        frames_without_verified_seed_or_S1_call=sum(r['S1_calls']==0 for r in rows),
        S1_valid_frames=sum(r['s1_valid'] for r in rows),current_F3_calls=len(rows),
        F3_valid_count_frames=sum(r['F3_count_valid'] for r in rows),
        prediction_frames=sum(r['emit_distance'] for r in rows),state_counts=dict(Counter(r['current_region'] for r in rows)),
        first_inside_confirmation_frame=gate.first_inside_confirmation_frame,
        first_inside_confirmation_time_s=gate.first_inside_confirmation_time,inside_confirmation_events=events,
        fixed_d_original_px=2238.7,d_automatically_calibrated=False,integer_count_phase_calibrated=False,
        no_future_frames_used=True,no_unknown_distance_hold_as_current=True,actual_P1_reacquisition_forwards=reacquire.live_model_forwards,
        reused_current_frame_P1_seed_count=reacquire.cache_seeds,reacquisition_callback_calls=reacquire.calls,
        median_tracker_ms=float(np.median([r['tracker_ms'] for r in rows])),
        median_surface_ms=float(np.median([r['surface_ms'] for r in rows])),
        median_total_ms=float(np.median([r['total_ms'] for r in rows])),
        wall_seconds=time.perf_counter()-started,static_scene_calibration=scene,binding=binding,
        measurements_sha256=sha(out/'measurements.csv'))
    dump(out/'receipt.json',receipt);dump(out/'progress.json',dict(status='completed',processed_frames=len(rows)))
    print(json.dumps({k:receipt[k] for k in ('status','processed_frames','S1_valid_frames','prediction_frames','state_counts','wall_seconds')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-stop',type=int)
    args=parser.parse_args();run(args.pilot_stop)
