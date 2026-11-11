"""Recover a committed frame prefix without modifying frozen numeric logic.

Replays and checks all causal frontend and temporal states before continuing.
The only I/O change is retrying transient Windows sharing violations.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,csv,time,gzip,json,hashlib
from collections import deque,Counter as Tally
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2,numpy as np
import transfer_engine as e
from run_transfer import ALLOWED
def replace_with_retry(src,dst):
    for k in range(12):
        try:Path(src).replace(dst);return
        except PermissionError:
            if k==11:raise
            time.sleep(min(.05*2**k,1.))
def safe_dump(p,x):
    p=Path(p);tmp=p.with_suffix('.retrytmp');tmp.write_text(json.dumps(e.plain(x),ensure_ascii=False,allow_nan=False),encoding='utf8');replace_with_retry(tmp,p)
e.dump=safe_dump
def frame_read(out,i):
    with gzip.open(out/'frames'/f'{i:06d}.json.gz','rt',encoding='utf8') as f:return json.load(f)
def close(a,b):
    if isinstance(a,dict) and isinstance(b,dict):return set(a)==set(b) and all(close(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):return len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
    if isinstance(a,(float,int)) and not isinstance(a,bool) and isinstance(b,(float,int)) and not isinstance(b,bool):return bool(np.isclose(a,b,rtol=1e-9,atol=1e-7))
    return a==b
def compact(r):
    f=r['frame_index'];ec=r['ECW'];state=r['ECW_decision']['temporal'];ev=r['ECW_decision']['evidence']
    return dict(frame=f,nominal_time_s=f/10,host_time_s=r['host_time_s'],source_pixel_sha256=r['source_pixel_sha256'],s1_valid=r['s1']['valid'],
        s1_accept_reason=r['S1_tracker_full']['diagnostics'].get('accept_reason'),region_valid=r['region']['valid'],region_reasons=r['region']['reasons'],
        F3_called=r['fresh_F3']['called'],raw_points=r['fresh_F3']['point_count'],BG4_curves=0 if not r['BG4'] else len(r['BG4']['curves']),
        count_state=r['temporal']['state'],display_N=r['temporal']['display_N'],count_value=r['temporal']['value'],ecw_valid=ec['valid'],
        ecw_state=state['state'],ecw_near=ev['near'],ecw_projection_gap_px=ec.get('center_gap_proxy_px'),camera_changed=r['camera_motion']['camera_changed'],
        reasons=r['reasons'],frontend_s=r['frontend_elapsed_s'],spatial_s=r['spatial_elapsed_s'])
def worker(im,track,cam,i,t,source_sha,out,host,elapsed):return e.spatial_frame(im,track,cam,i,t,source_sha,out),host,elapsed
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--session',choices=ALLOWED,required=True);ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
    out=e.ROOT/'transfer_runs'/args.session;bind=e.read(out/'binding.json');assert not (out/'receipt.json').exists()
    prefix=sorted((out/'frames').glob('*.json.gz'));assert [p.name for p in prefix]==[f'{i:06d}.json.gz' for i in range(len(prefix))]
    old_hashes=[dict(frame=i,sha256=e.sha(p)) for i,p in enumerate(prefix)];start=len(prefix)
    source=Path(bind['source']);assert e.sha(source)==bind['source_sha256'];assert e.verify_baseline()==bind['baseline_manifest_sha256']
    assert e.sha(e.ROOT/'transfer_engine.py')==bind['adapter_sha256']
    recovery=dict(cause='Windows sharing violation while replacing progress.json',committed_prefix_frames=start,
        original_runner_sha256=bind['runner_sha256'],resume_runner_sha256=e.sha(__file__),frozen_algorithm_changed=False,
        interface_change='retry atomic file replacement only; replay prefix to reconstruct causal states',prefix_file_hashes=old_hashes,
        replay_verified=False,started_unix=time.time())
    e.dump(out/'recovery.json',recovery)
    cap=cv2.VideoCapture(str(source));n=int(cap.get(7));assert n==bind['source_frames'];fps=cap.get(5);assert fps==10
    stamps=list(csv.DictReader((source.parent/'left_frames.csv').open(encoding='utf-8-sig')));times=[float(x['host_time_s']) for x in stamps];t0=times[0]
    front=e.Frontend();counter=e.Counter();window=e.WindowDetector();rows=[];pending=deque();tick_replay=time.perf_counter()
    for i in range(start):
        ok,im=cap.read();assert ok;r=frame_read(out,i);assert e.pixelsha(im)==r['source_pixel_sha256']
        track,cam=front.step(im,times[i]-t0,i)
        for key in ['valid','current_line_abc','line_points_xy','confidence','seed_xy_original','seed_source']:
            assert close(e.plain(track.get(key)),r['S1_tracker_full'].get(key)),f'Frontend replay mismatch frame {i} key {key}'
        assert close(cam,r['camera_motion']),f'Camera replay mismatch {i}'
        temporal=counter.step(r);assert close(temporal,r['temporal']),f'Counter replay mismatch {i}'
        ec=r['ECW'];ev=e.evidence(ec,r);state=window.step(i,times[i],ev['near'],ev['far'],ec.get('camera_valid',True))
        assert close(ev,r['ECW_decision']['evidence']) and close(state,r['ECW_decision']['temporal']),f'Window replay mismatch {i}'
        rows.append(compact(r))
        if (i+1)%50==0:print('REPLAY VERIFIED',i+1,'/',start,round(time.perf_counter()-tick_replay,1),flush=True)
    recovery.update(replay_verified=True,replay_frames=start,replay_elapsed_s=time.perf_counter()-tick_replay);e.dump(out/'recovery.json',recovery)
    print('RESUMING AT',start,flush=True)
    def consume():
        r,host,elapsed=pending.popleft().result();f=r['frame_index'];r['host_time_s']=host;r['temporal']=counter.step(r);ec=r['ECW'];ec['host_time_s']=host
        ev=e.evidence(ec,r);state=window.step(f,host,ev['near'],ev['far'],ec.get('camera_valid',True));r['ECW_decision']=dict(evidence=ev,temporal=state);r['frontend_elapsed_s']=elapsed
        path=out/'frames'/f'{f:06d}.json.gz';assert not path.exists();tmp=path.with_suffix('.part')
        with gzip.open(tmp,'wb',compresslevel=3) as dest:dest.write(json.dumps(e.plain(r),ensure_ascii=False,allow_nan=False).encode('utf8'))
        replace_with_retry(tmp,path);rows.append(compact(r));progress=dict(processed=len(rows),planned=n,elapsed_s=time.time()-(out/'binding.json').stat().st_ctime,
            S1_valid=sum(x['s1_valid'] for x in rows),region_valid=sum(x['region_valid'] for x in rows),count_displayed=sum(x['display_N'] is not None for x in rows),
            ecw_valid=sum(x['ecw_valid'] for x in rows),near_evidence=sum(x['ecw_near'] for x in rows),P1_reacquisitions=front.reacquire.live_model_forwards,
            counts=dict(Tally(x['count_state'] for x in rows)),recovery_prefix_replay_verified=True)
        e.dump(out/'progress.json',progress)
        if len(rows)%20==0 or len(rows)==n:print(args.session,progress,flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i in range(start,n):
            ok,im=cap.read();assert ok and round(cap.get(1))==i+1
            tick=time.perf_counter();track,cam=front.step(im,times[i]-t0,i);elapsed=time.perf_counter()-tick
            pending.append(pool.submit(worker,im,track,cam,i,i/10,bind['source_sha256'],str(out),times[i],elapsed))
            if len(pending)>=args.workers+2:consume()
        while pending:consume()
    cap.release();window.finish();e.dump(out/'compact.json',rows);e.dump(out/'windows.json',window.events)
    assert e.sha(source)==bind['source_sha256'];assert e.verify_baseline()==bind['baseline_manifest_sha256']
    assert all(e.sha(out/'frames'/f"{r['frame']:06d}.json.gz")==r['sha256'] for r in old_hashes)
    e.dump(out/'receipt.json',dict(status='complete',frames=len(rows),source_unchanged=True,baseline_unchanged=True,events=window.events,
        elapsed_s=time.time()-(out/'binding.json').stat().st_ctime,P1_model_forwards=front.reacquire.live_model_forwards,frame_skips=0,
        absolute_accuracy_validated=False,recovery_prefix_replay_verified=True,committed_prefix_unchanged=True,
        elapsed_includes_interruption_and_replay=True))
    print('COMPLETE',args.session,flush=True)
if __name__=='__main__':main()
