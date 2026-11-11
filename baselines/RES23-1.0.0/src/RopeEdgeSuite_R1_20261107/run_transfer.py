"""Run every original frame through the frozen RES-1.0 on a new video."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,csv,time,gzip,json,hashlib
from collections import deque,Counter as Tally
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2
import transfer_engine as e
ALLOWED=['dual_20260923_113745','dual_20260924_111143']
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--session',choices=ALLOWED,required=True);ap.add_argument('--workers',type=int,default=3);ap.add_argument('--limit',type=int)
    args=ap.parse_args();assert 1<=args.workers<=4
    assert e.read(e.ROOT/'parity/receipt.json')['passed'];baseline_hash=e.verify_baseline()
    source=e.ROOT.parents[3]/'双相机同步'/args.session/'left_DA9627065.avi'
    source_hash=e.sha(source);cap=cv2.VideoCapture(str(source));n=int(cap.get(7));fps=float(cap.get(5))
    assert fps==10 and cap.get(3)==4096 and cap.get(4)==3000
    stamps=list(csv.DictReader((source.parent/'left_frames.csv').open(encoding='utf-8-sig')));assert len(stamps)==n
    times=[float(x['host_time_s']) for x in stamps];assert all(b>a for a,b in zip(times,times[1:]))
    end=n if args.limit is None else min(n,args.limit)
    name=args.session+('__pilot_'+str(end) if args.limit else '')
    out=e.ROOT/'transfer_runs'/name;out.mkdir(parents=True,exist_ok=False);(out/'frames').mkdir()
    binding=dict(algorithm='RES-1.0',baseline_manifest_sha256=baseline_hash,source=str(source),source_sha256=source_hash,
        source_frames=n,processed_frames_planned=end,fps=fps,d_original_px=e.D,pitch_prior_px=e.PITCH,static_search_mask='frozen_reference_left_mask',
        no_transfer_tuning=True,new_video_calibration_validated=False,adapter_sha256=e.sha(e.ROOT/'transfer_engine.py'),runner_sha256=e.sha(__file__),
        frame_selection='all_original_frames_in_chronological_order',source_time_basis='host_time_s for tracker/ECW; original 10fps timestamps for frozen BG4')
    e.dump(out/'binding.json',binding)
    front=e.Frontend();counter=e.Counter();window=e.WindowDetector();pending=deque();rows=[];started=time.perf_counter();t0=times[0]
    def consume():
        r,host,elapsed=pending.popleft().result();f=r['frame_index'];r['host_time_s']=host;r['temporal']=counter.step(r)
        ec=r['ECW'];ec['host_time_s']=host
        ev=e.evidence(ec,r);state=window.step(f,host,ev['near'],ev['far'],ec.get('camera_valid',True))
        r['ECW_decision']=dict(evidence=ev,temporal=state);r['frontend_elapsed_s']=elapsed
        payload=json.dumps(e.plain(r),ensure_ascii=False,allow_nan=False).encode('utf8')
        path=out/'frames'/f'{f:06d}.json.gz'
        with gzip.open(path,'wb',compresslevel=3) as dest:dest.write(payload)
        row=dict(frame=f,nominal_time_s=f/fps,host_time_s=host,source_pixel_sha256=r['source_pixel_sha256'],s1_valid=r['s1']['valid'],
            s1_accept_reason=r['S1_tracker_full']['diagnostics'].get('accept_reason'),
            region_valid=r['region']['valid'],region_reasons=r['region']['reasons'],F3_called=r['fresh_F3']['called'],
            raw_points=r['fresh_F3']['point_count'],BG4_curves=0 if not r['BG4'] else len(r['BG4']['curves']),
            count_state=r['temporal']['state'],display_N=r['temporal']['display_N'],count_value=r['temporal']['value'],
            ecw_valid=ec['valid'],ecw_state=state['state'],ecw_near=ev['near'],ecw_projection_gap_px=ec.get('center_gap_proxy_px'),
            camera_changed=r['camera_motion']['camera_changed'],reasons=r['reasons'],frontend_s=elapsed,spatial_s=r['spatial_elapsed_s'])
        rows.append(row)
        progress=dict(processed=len(rows),planned=end,elapsed_s=time.perf_counter()-started,S1_valid=sum(x['s1_valid'] for x in rows),
            region_valid=sum(x['region_valid'] for x in rows),count_displayed=sum(x['display_N'] is not None for x in rows),
            ecw_valid=sum(x['ecw_valid'] for x in rows),near_evidence=sum(x['ecw_near'] for x in rows),
            P1_reacquisitions=front.reacquire.live_model_forwards,counts=dict(Tally(x['count_state'] for x in rows)))
        e.dump(out/'progress.json',progress)
        if len(rows)%20==0 or len(rows)==end:print(args.session,progress,flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i in range(end):
            ok,im=cap.read();assert ok and round(cap.get(1))==i+1
            tick=time.perf_counter();track,cam=front.step(im,times[i]-t0,i);elapsed=time.perf_counter()-tick
            pending.append(pool.submit(worker,im,track,cam,i,i/fps,source_hash,str(out),times[i],elapsed))
            if len(pending)>=args.workers+2:consume()
        while pending:consume()
    cap.release();window.finish();e.dump(out/'compact.json',rows);e.dump(out/'windows.json',window.events)
    e.verify_baseline();assert e.sha(source)==source_hash
    e.dump(out/'receipt.json',dict(status='complete',frames=len(rows),source_unchanged=True,baseline_unchanged=True,events=window.events,
        elapsed_s=time.perf_counter()-started,P1_model_forwards=front.reacquire.live_model_forwards,
        frame_skips=0,absolute_accuracy_validated=False))
    print('COMPLETE',args.session,flush=True)
def worker(im,track,cam,i,t,source_sha,out,host,elapsed):
    return e.spatial_frame(im,track,cam,i,t,source_sha,out),host,elapsed
if __name__=='__main__':main()
