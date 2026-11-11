"""Boundary-only ablation using source-bound same-frame baseline S1/camera.

BG5, raw 01/F3 points, BG3/BG4 spatial and temporal counting, and ECW are
recomputed. No previous count, curve, mask or future observation is reused.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,gzip,time,argparse
from collections import deque,Counter as Tally
from concurrent.futures import ProcessPoolExecutor
import cv2
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'RopeEdgeSuite_R1_20261107'
sys.path.insert(0,str(OLD))
import transfer_engine as e
sys.path.insert(0,str(ROOT))
import region_bg5

def replace_retry(tmp,path):
    for k in range(12):
        try:os.replace(tmp,path);return
        except PermissionError:
            if k==11:raise
            time.sleep(min(.05*2**k,1))

def dump(path,x):
    path=Path(path);temp=path.with_suffix(path.suffix+'.part')
    temp.write_text(json.dumps(e.plain(x),ensure_ascii=False,allow_nan=False),encoding='utf8');replace_retry(temp,path)

def record(run,i):
    with gzip.open(run/'frames'/f'{i:06d}.json.gz','rt',encoding='utf8') as f:return json.load(f)

def worker(im,track,cam,i,source_sha,out):
    captured={}
    def segment(image):
        result=region_bg5.segment(image);captured.update(result['record']);return result
    e.segment=segment;e.dump=dump
    r=e.spatial_frame(im,track,cam,i,i/10,source_sha,out)
    r['region_BG5']=captured
    r['region']['top_curve_supported_segments']=captured.get('top_curve_supported_segments',[])
    r['version']='RES-1.1-BG5/boundary-ablation'
    return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--session',choices=['dual_20260923_113745','dual_20260924_111143'],default='dual_20260923_113745')
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--limit',type=int);a=ap.parse_args()
    cv2.setNumThreads(1);assert 1<=a.workers<=10
    baseline_hash=e.verify_baseline();prior=OLD/'transfer_runs'/a.session;binding=e.read(prior/'binding.json')
    assert e.sha(binding['source'])==binding['source_sha256'];assert e.read(prior/'receipt.json')['status']=='complete'
    cap=cv2.VideoCapture(binding['source']);n=int(cap.get(7));end=n if a.limit is None else min(n,a.limit)
    out=ROOT/'runs'/(a.session if a.limit is None else a.session+f'_pilot{end}')
    out.mkdir(parents=True,exist_ok=False);(out/'frames').mkdir()
    meta=dict(binding);meta.update(algorithm='RES-1.1-BG5',experiment='boundary_only_controlled_ablation',
        processed_frames_planned=end,no_transfer_tuning=False,
        S1_and_camera='reused_source_pixel_bound_same_frame_outputs_of_RES-1.0; no S1 fix claimed',
        recomputed=['BG5 region','raw 01/F3 points','BG3/BG4 spatial','causal counter from empty state','ECW measurement and causal windows'],
        region_code_sha256=e.sha(ROOT/'region_bg5.py'),runner_sha256=e.sha(__file__),baseline_manifest_sha256=baseline_hash,
        development_frames=[0,20,35,190,240,280,286,316,336,526,599],
        absolute_accuracy_validated=False)
    dump(out/'binding.json',meta)
    counter=e.Counter();window=e.WindowDetector();rows=[];pending=deque();start=time.perf_counter()
    def consume():
        future,host,priorhash=pending.popleft();r=future.result();i=r['frame_index']
        r['host_time_s']=host;r['frontend_elapsed_s']=0.;r['reused_frontend_record_sha256']=priorhash
        r['temporal']=counter.step(r);r['ECW']['host_time_s']=host
        ev=e.evidence(r['ECW'],r);state=window.step(i,host,ev['near'],ev['far'],r['ECW'].get('camera_valid',True))
        r['ECW_decision']=dict(evidence=ev,temporal=state)
        path=out/'frames'/f'{i:06d}.json.gz';temp=path.with_suffix('.part')
        with gzip.open(temp,'wb',compresslevel=3) as f:f.write(json.dumps(e.plain(r),ensure_ascii=False,allow_nan=False).encode('utf8'))
        replace_retry(temp,path)
        rows.append(dict(frame=i,time_s=i/10,s1_valid=r['s1']['valid'],region_valid=r['region']['valid'],region_reasons=r['region']['reasons'],
             outer_local_valid=r['region_BG5'].get('outer_rim',{}).get('valid',False),
             region_edge_y=r['region']['edge_y'],raw_points=r['fresh_F3']['point_count'],curves=len(r['BG4']['curves']) if r['BG4'] else 0,
             count_state=r['temporal']['state'],display_N=r['temporal']['display_N'],count_value=r['temporal']['value'],
             count_events=r['temporal']['events'],spatial_reasons=r['reasons'],ecw_valid=r['ECW']['valid'],ecw_near=ev['near'],
             spatial_s=r['spatial_elapsed_s']))
        if len(rows)%20==0 or len(rows)==end:
            progress=dict(processed=len(rows),planned=end,elapsed_s=round(time.perf_counter()-start,1),
                region_valid=sum(x['region_valid'] for x in rows),displayed=sum(x['display_N'] is not None for x in rows),
                count_states=dict(Tally(x['count_state'] for x in rows)))
            dump(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i in range(end):
            ok,im=cap.read();assert ok and round(cap.get(1))==i+1
            prevpath=prior/'frames'/f'{i:06d}.json.gz';prev=record(prior,i)
            assert prev['frame_index']==i and prev['source_pixel_sha256']==e.pixelsha(im)
            future=pool.submit(worker,im,prev['S1_tracker_full'],prev['camera_motion'],i,binding['source_sha256'],str(out))
            pending.append((future,prev['host_time_s'],e.sha(prevpath)))
            if len(pending)>=a.workers+2:consume()
        while pending:consume()
    cap.release();window.finish();dump(out/'compact.json',rows);dump(out/'windows.json',window.events)
    assert e.verify_baseline()==baseline_hash and e.sha(binding['source'])==binding['source_sha256']
    dump(out/'receipt.json',dict(status='complete',frames=len(rows),source_unchanged=True,baseline_unchanged=True,
        frame_skips=0,elapsed_s=time.perf_counter()-start,events=window.events,
        method='boundary-only ablation; source-bound baseline frontend reused',absolute_accuracy_validated=False))
    print('COMPLETE',str(out),flush=True)

if __name__=='__main__':main()
