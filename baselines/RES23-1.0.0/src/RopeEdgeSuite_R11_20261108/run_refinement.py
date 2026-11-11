"""Exact conditional BG5.1 refinement with chronological full-sequence replay.

If BG5 rejects only due to the largest yellow component's area, recompute the
region and every spatial downstream stage with current fragment association.
Else reuse its byte-bound, same-frame spatial evidence. Always replay counting
and event windows from empty state. This is an explicit dependency graph,
not mixed or stale counting caches.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import sys,json,gzip,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,Future
from collections import deque,Counter as Tally
import cv2
from run_bg5 import ROOT,OLD,e,dump,replace_retry,record
import region_bg5_refined
SESSION='dual_20260923_113745'

def worker(im,track,cam,i,source_sha,out):
    captured={}
    def segment(image):
        result=region_bg5_refined.segment(image);captured.update(result['record']);return result
    e.segment=segment;e.dump=dump;r=e.spatial_frame(im,track,cam,i,i/10,source_sha,out)
    r['region_BG5']=captured;r['region']['top_curve_supported_segments']=captured.get('top_curve_supported_segments',[])
    return r

def main():
    cv2.setNumThreads(1);assert e.read(ROOT/'refined_test_receipt.json')['passed']
    first=ROOT/'runs'/SESSION;meta=e.read(first/'binding.json');source=meta['source'];source_sha=meta['source_sha256']
    assert e.sha(ROOT/'region_bg5.py')==meta['region_code_sha256'];baseline=e.verify_baseline()
    out=ROOT/'runs'/(SESSION+'_refined');out.mkdir(parents=True,exist_ok=False);(out/'frames').mkdir();(out/'bg1/frames').mkdir(parents=True)
    binding=dict(meta);binding.update(algorithm='RES-1.1-BG5.1',first_pass_binding_sha256=e.sha(first/'binding.json'),
        refinement_code_sha256=e.sha(ROOT/'region_bg5_refined.py'),refinement_runner_sha256=e.sha(__file__),
        experiment='conditional color-fragment refinement of source-bound BG5 spatial evidence; full causal replay',
        branch='only largest-color-component-area failures recomputed; all other spatial branches identical to BG5')
    dump(out/'binding.json',binding)
    cap=cv2.VideoCapture(source);n=int(cap.get(7));assert n==600;pending=deque();rows=[];counter=e.Counter();window=e.WindowDetector();start=time.perf_counter()
    def consume():
        future,host,priorhash,frontendhash,used=pending.popleft();r=future.result();i=r['frame_index']
        r.update(version='RES-1.1-BG5.1/conditional-refinement',BG5_first_pass_record_sha256=priorhash,BG5_refinement_used=used,
                 reused_frontend_record_sha256=frontendhash,frontend_elapsed_s=0.,host_time_s=host)
        r['temporal']=counter.step(r);r['ECW']['host_time_s']=host
        ev=e.evidence(r['ECW'],r);state=window.step(i,host,ev['near'],ev['far'],r['ECW'].get('camera_valid',True))
        r['ECW_decision']=dict(evidence=ev,temporal=state)
        path=out/'frames'/f'{i:06d}.json.gz';tmp=path.with_suffix('.part')
        with gzip.open(tmp,'wb',compresslevel=3) as f:f.write(json.dumps(e.plain(r),ensure_ascii=False,allow_nan=False).encode('utf8'))
        replace_retry(tmp,path)
        rows.append(dict(frame=i,time_s=i/10,refined=used,s1_valid=r['s1']['valid'],region_valid=r['region']['valid'],region_reasons=r['region']['reasons'],
            region_edge_y=r['region']['edge_y'],raw_points=r['fresh_F3']['point_count'],curves=len(r['BG4']['curves']) if r['BG4'] else 0,
            count_state=r['temporal']['state'],display_N=r['temporal']['display_N'],count_value=r['temporal']['value'],
            count_events=r['temporal']['events'],spatial_reasons=r['reasons'],ecw_valid=r['ECW']['valid'],ecw_near=ev['near'],spatial_s=r['spatial_elapsed_s']))
        if len(rows)%20==0 or len(rows)==n:
            p=dict(processed=len(rows),planned=n,elapsed_s=round(time.perf_counter()-start,1),region_valid=sum(x['region_valid'] for x in rows),
                recomputed_frames=sum(x['refined'] for x in rows),displayed=sum(x['display_N'] is not None for x in rows),count_states=dict(Tally(x['count_state'] for x in rows)))
            dump(out/'progress.json',p);print(json.dumps(p),flush=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i in range(n):
            path=first/'frames'/f'{i:06d}.json.gz';deadline=time.monotonic()+600
            while not path.exists():
                assert time.monotonic()<deadline,'First pass stalled';time.sleep(.25)
            ok,im=cap.read();assert ok;r=record(first,i);assert e.pixelsha(im)==r['source_pixel_sha256'] and r['frame_index']==i
            used=r['region']['reasons']==['insufficient_yellow_flange']
            if used:future=pool.submit(worker,im,r['S1_tracker_full'],r['camera_motion'],i,source_sha,str(out))
            else:
                # Preserve the exact first-pass spatial fields and BG1 binding.
                p=out/'bg1/frames'/f'{i:06d}.json';tmp=p.with_suffix('.part');tmp.write_bytes((first/'bg1/frames'/p.name).read_bytes());replace_retry(tmp,p)
                future=Future();future.set_result(r)
            pending.append((future,r['host_time_s'],e.sha(path),r['reused_frontend_record_sha256'],used))
            if len(pending)>=4:consume()
        while pending:consume()
    cap.release();window.finish();deadline=time.monotonic()+60
    while not (first/'receipt.json').exists():
        assert time.monotonic()<deadline;time.sleep(.25)
    assert e.read(first/'receipt.json')['status']=='complete';assert e.sha(source)==source_sha and e.verify_baseline()==baseline
    dump(out/'compact.json',rows);dump(out/'windows.json',window.events)
    dump(out/'receipt.json',dict(status='complete',frames=n,frame_skips=0,source_unchanged=True,baseline_unchanged=True,
        spatial_recomputed_frames=sum(x['refined'] for x in rows),spatial_reused_frames=sum(not x['refined'] for x in rows),
        temporal_replayed_frames=n,events=window.events,elapsed_s=time.perf_counter()-start,
        elapsed_includes_wait_for_first_pass=True,first_pass_elapsed_s=e.read(first/'receipt.json')['elapsed_s'],
        absolute_accuracy_validated=False))
    print('REFINEMENT COMPLETE',flush=True)

if __name__=='__main__':main()
