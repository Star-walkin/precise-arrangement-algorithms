"""Full fresh-S1/F3/curve/count run. BG5 images are bound to the same source frame."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,gzip,time,argparse
from collections import deque,Counter as Tally
from concurrent.futures import ProcessPoolExecutor
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'RopeEdgeSuite_R11_20261108'))
from run_bg5 import e,dump,replace_retry,record

def extract_current(image,mask,track):
    exclusion=np.zeros_like(mask);exclusion[mask==0]=255
    for pts in track.get('supported_segments_xy',[]):
        if len(pts)>1:cv2.polylines(exclusion,[np.rint(pts).astype(np.int32)],False,255,round(.8*e.PITCH))
    im=cv2.resize(image,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
    mm=((cv2.resize(mask,(im.shape[1],im.shape[0]),interpolation=cv2.INTER_AREA)==255)*255).astype(np.uint8)
    ee=((cv2.resize(exclusion,(im.shape[1],im.shape[0]),interpolation=cv2.INTER_AREA)>0)*255).astype(np.uint8)
    p=e.PITCH*.5;norm,_=e.core._module('normalization')._normalize_surface(im,mm,p)
    shadow=e.core._module('shadow').build_shadow_field(im,norm,mm,p,None);shadow['excluded_mask']=ee
    guard=e.core._module('period').build_period_guard(im,norm,mm,p,None,shadow)
    observed=e.core._module('tracks').detect_and_group(shadow,guard,mm,p)
    track_by_point={pt['point_id']:k for k,t in enumerate(observed['tracks']) for pt in t['points']}
    points=np.asarray([[pt['point_id'],pt['x'],pt['y'],pt['score'],pt['width'],pt['slope'],pt['pitch'],track_by_point[pt['point_id']]]
        for pt in observed['points']],np.float64).reshape(-1,8)
    x0,y0=shadow['x0'],shadow['y0'];hh,ww=shadow['body_score'].shape;ee[y0:y0+hh,x0:x0+ww]|=guard['unknown_mask']
    return dict(points=points,surface=mm,exclusion=ee,broad=shadow['broad_score'],clean=shadow['cleaned_shadow'],body=guard['body_barrier'],
        x0=np.asarray(x0),y0=np.asarray(y0),pitch=np.asarray(p),source_pixel_sha256=np.asarray(e.pixelsha(image)),
        raw_evidence_cache_origin=np.asarray('fresh_current_frame_SG2_observed_segments_only_exclusion'))

def worker(image,item,mask,source_sha,out):
    assert e.pixelsha(image)==item['source_pixel_sha256']
    e.segment=lambda im:dict(record=item['region'],surface_mask=mask)
    e.extract_current=extract_current;e.dump=dump
    r=e.spatial_frame(image,item['track'],item['camera'],item['frame_index'],item['nominal_time_s'],source_sha,out)
    r['region_BG5']=item['region'];r['region']['top_curve_supported_segments']=item['region'].get('top_curve_supported_segments',[])
    r['version']='RES-1.2/SG2-BG5.1';return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);a=ap.parse_args()
    cv2.setNumThreads(1);frontend=ROOT/'frontend/dual_20260923_113745_v2'
    meta=e.read(frontend/'binding.json');source=meta['source'];source_sha=meta['source_sha256'];n=meta['planned_end']
    out=ROOT/'runs/dual_20260923_113745';out.mkdir(parents=True,exist_ok=False);(out/'frames').mkdir()
    binding=dict(meta);binding.update(processed_frames_planned=n,experiment='fresh SG2 frontend and F3 spatial + chronological count',
        reused_current_camera_only=True,processed_source_frames='all at original10fps',
        recomputed=['SG2 activity','BG5.1 region','01/F3 edges','BG3/4 class curves','causal counts','ECW'],
        runner_sha256=e.sha(__file__),frontend_binding_sha256=e.sha(frontend/'binding.json'))
    dump(out/'binding.json',binding);baseline=e.verify_baseline()
    counter=e.Counter();window=e.WindowDetector();cap=cv2.VideoCapture(source);pending=deque();rows=[];start=time.perf_counter()
    def consume():
        future,item,frontend_hash,mask_hash=pending.popleft();r=future.result();i=r['frame_index'];host=item['host_time_s']
        r.update(host_time_s=host,frontend_record_sha256=frontend_hash,current_mask_sha256=mask_hash,
            frontend_elapsed_s=item['track']['diagnostics']['total_seconds'])
        r['temporal']=counter.step(r);r['ECW']['host_time_s']=host
        ev=e.evidence(r['ECW'],r);state=window.step(i,host,ev['near'],ev['far'],r['ECW'].get('camera_valid',True))
        r['ECW_decision']=dict(evidence=ev,temporal=state)
        path=out/'frames'/f'{i:06d}.json.gz';tmp=path.with_suffix('.part')
        with gzip.open(tmp,'wb',compresslevel=3) as f:f.write(json.dumps(e.plain(r),ensure_ascii=False,allow_nan=False).encode())
        replace_retry(tmp,path)
        rows.append(dict(frame=i,time_s=i/10,s1_valid=r['s1']['valid'],region_valid=r['region']['valid'],
            count_state=r['temporal']['state'],display_N=r['temporal']['display_N'],count_value=r['temporal']['value'],
            count_events=r['temporal']['events'],spatial_reasons=r['reasons'],ecw_valid=r['ECW']['valid'],ecw_near=ev['near'],
            s1_reason=item['track']['diagnostics']['accept_reason'],raw_points=r['fresh_F3']['point_count'],
            curves=len(r['BG4']['curves']) if r['BG4'] else 0,spatial_s=r['spatial_elapsed_s']))
        if len(rows)%10==0 or len(rows)==n:
            progress=dict(processed=len(rows),planned=n,elapsed_s=round(time.perf_counter()-start,1),
                s1_valid=sum(v['s1_valid'] for v in rows),region_valid=sum(v['region_valid'] for v in rows),
                displayed=sum(v['display_N'] is not None for v in rows),count_states=dict(Tally(v['count_state'] for v in rows)))
            dump(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i in range(n):
            path=frontend/'frames'/f'{i:06d}.json.gz';maskpath=frontend/'regions'/f'{i:06d}.png';deadline=time.monotonic()+600
            while not path.exists() or not maskpath.exists():
                assert time.monotonic()<deadline,'Frontend failed to publish a frame';time.sleep(.2)
            item=record(frontend,i);mask=cv2.imdecode(np.fromfile(maskpath,np.uint8),0)
            ok,im=cap.read();assert ok and mask is not None
            pending.append((pool.submit(worker,im,item,mask,source_sha,str(out)),item,e.sha(path),e.pixelsha(mask)))
            if len(pending)>=a.workers+1:consume()
        while pending:consume()
    cap.release();window.finish();dump(out/'compact.json',rows);dump(out/'windows.json',window.events)
    assert e.sha(source)==source_sha and e.verify_baseline()==baseline
    assert e.read(frontend/'receipt.json')['status']=='complete'
    dump(out/'receipt.json',dict(status='complete',frames=len(rows),source_unchanged=True,baseline_unchanged=True,
        skipped_frames=0,elapsed_s=time.perf_counter()-start,events=window.events,absolute_accuracy_validated=False))

if __name__=='__main__':main()
