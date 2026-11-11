"""Fresh current-pixel frontend and BG5.1 regions, chronological processing."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse, sys, json, gzip, time
import cv2,numpy as np
from tracker_sg2 import ROOT,RES,Tracker,DynamicReacquire,line_y
sys.path.insert(0,str(ROOT.parent/'RopeEdgeSuite_R11_20261108'))
import region_bg5_refined as bg5
from run_bg5 import e,dump,replace_retry,record

SESSION='dual_20260923_113745'
PRIOR=RES/'transfer_runs'/SESSION

def restore_probe(tracker,im,r):
    sys.path.insert(0,str(ROOT.parent/'S1_loss_113745_20261109'))
    from diagnose_loss import restore
    restore(tracker,im,r)
    s=r['S1_tracker_full'];seed=np.asarray(s['seed_xy_original']);d=seed[1]-seed[0];d/=np.linalg.norm(d)
    tracker.trusted=dict(t=r['nominal_time_s'],frame=r['frame_index'],abc=s['line_abc'],
        width=s['diagnostics']['seed_validation']['width_px'],reference_x=seed[0,0],direction=d)
    tracker.history=[(r['nominal_time_s'],line_y(s['line_abc'],seed[0,0]))]

def restore_current(tracker,im,items):
    r=items[-1];s=r['track'];assert s['valid']
    d=s['diagnostics'];seed=np.asarray(s['seed_xy_original'],float);direction=seed[1]-seed[0];direction/=np.linalg.norm(direction)
    pts=np.asarray(s['line_points_xy'],float)
    if d['seed_witness_matches_primary_S1_seed_measurement']:pts=np.vstack([pts,d['seed_validation']['points_xy']])
    anchor=d['spatial_anchor'];x=anchor['x_original']
    tracker.prev=dict(gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),tracking_support_xy=pts,direction=direction,t=r['nominal_time_s'],frame_index=r['frame_index'])
    tracker.trusted=dict(t=r['nominal_time_s'],frame=r['frame_index'],abc=s['line_abc'],width=d['seed_validation']['width_px'],reference_x=x,direction=direction)
    tracker.anchor_x_original=x;tracker.anchor_frame=anchor['established_frame'];tracker.anchor_reset_count=anchor['reset_count']
    tracker.anchor_source=s['seed_source'];tracker.anchor_reason='SG2_accepted_current_identity'
    tracker.history=[(q['nominal_time_s'],line_y(q['track']['line_abc'],x)) for q in items if q['track']['valid'] and q['frame_index']>=anchor['established_frame']][-9:]
    if tracker.reacquirer:
        reacq=[q['frame_index'] for q in items if q['track']['diagnostics']['reacquire'] is not None]
        tracker.reacquirer.last=max(reacq,default=-100);tracker.reacquirer.calls=len(reacq)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--probe',action='store_true');ap.add_argument('--end',type=int,default=600);ap.add_argument('--name');ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    cv2.setNumThreads(2)
    source=e.read(PRIOR/'binding.json');baseline=e.verify_baseline()
    name=a.name or ('probe315' if a.probe else SESSION+'_v2')
    out=ROOT/'frontend'/name;out.mkdir(parents=True,exist_ok=a.resume)
    (out/'frames').mkdir(exist_ok=a.resume);(out/'regions').mkdir(exist_ok=a.resume)
    binding=dict(source);binding.update(algorithm='RES-1.2/SG2',start_frame=316 if a.probe else 0,planned_end=a.end,
        probe_initialization='saved verified state at315, no future seeds' if a.probe else 'cold current-pixel P1 with current BG5',
        tracker_code_sha256=e.sha(ROOT/'tracker_sg2.py'),region_code_sha256=e.sha(bg5.__file__),baseline_manifest_sha256=baseline)
    if not a.resume:dump(out/'binding.json',binding)
    cap=cv2.VideoCapture(source['source']);mask=cv2.imdecode(np.fromfile(RES/'baseline/calibration/left_search_mask.png',np.uint8),0)
    tracker=Tracker(mask,None if a.probe else DynamicReacquire(e.PITCH))
    begin=0;items=[]
    if a.resume:
        files=sorted((out/'frames').glob('*.json.gz'));assert files
        items=[record(out,i) for i in range(len(files))]
        begin=len(items);cap.set(1,begin-1);ok,im=cap.read();assert ok and e.pixelsha(im)==items[-1]['source_pixel_sha256']
        restore_current(tracker,im,items)
        lastregion=next(q for q in reversed(items) if q['region']['valid'])
        m=cv2.imdecode(np.fromfile(out/'regions'/f"{lastregion['frame_index']:06d}.png",np.uint8),0)
        tracker.set_region(dict(record=lastregion['region'],surface_mask=m))
        oldbinding=e.read(out/'binding.json')
        oldbinding.setdefault('resumptions',[]).append(dict(first_resumed_frame=begin,restored_current_state_frame=begin-1,
            tracker_code_sha256=e.sha(ROOT/'tracker_sg2.py'),change='catch only frozen sampler width overflow; all successful and rejected numeric paths unchanged',
            previous_tracker_source_sha256=e.sha(ROOT/'tracker_sg2_before_profile_guard.py')))
        dump(out/'binding.json',oldbinding)
    elif a.probe:
        cap.set(1,315);ok,im=cap.read();assert ok and e.pixelsha(im)==record(PRIOR,315)['source_pixel_sha256']
        restore_probe(tracker,im,record(PRIOR,315));begin=316
    start=time.perf_counter();rows=[]
    for r in items:
        s=r['track'];rows.append(dict(frame=r['frame_index'],valid=s['valid'],reason=s['diagnostics']['accept_reason'],
            attempts=len(s['diagnostics']['attempts']),abc=s['line_abc'],source=s['seed_source'],region_valid=r['region']['valid'],model_forwards=s['model_forward_count']))
    for i in range(begin,a.end):
        ok,im=cap.read();assert ok and round(cap.get(1))==i+1
        old=record(PRIOR,i);pixel=e.pixelsha(im);assert pixel==old['source_pixel_sha256']
        region=bg5.segment(im);tracker.set_region(region)
        track=tracker.step(im,i/10,i)
        item=dict(frame_index=i,nominal_time_s=i/10,source_pixel_sha256=pixel,track=track,
            region=region['record'],camera=old['camera_motion'],host_time_s=old['host_time_s'],
            camera_provenance='unchanged source-bound same-frame camera diagnostic only')
        path=out/'frames'/f'{i:06d}.json.gz';tmp=path.with_suffix('.part')
        with gzip.open(tmp,'wb',compresslevel=3) as f:f.write(json.dumps(e.plain(item),ensure_ascii=False,allow_nan=False).encode())
        replace_retry(tmp,path)
        cv2.imencode('.png',region['surface_mask'])[1].tofile(out/'regions'/f'{i:06d}.png')
        rows.append(dict(frame=i,valid=track['valid'],reason=track['diagnostics']['accept_reason'],
            attempts=len(track['diagnostics']['attempts']),abc=track['line_abc'],
            source=track['seed_source'],region_valid=region['record']['valid'],model_forwards=track['model_forward_count']))
        if i%10==0 or i==a.end-1:
            progress=dict(last_frame=i,processed=len(rows),valid=sum(r['valid'] for r in rows),elapsed_s=round(time.perf_counter()-start,1),
                last_reason=rows[-1]['reason'],model_forwards=sum(r['model_forwards'] for r in rows))
            dump(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    cap.release();assert e.verify_baseline()==baseline
    dump(out/'compact.json',rows);dump(out/'receipt.json',dict(status='complete',frames=len(rows),elapsed_s=time.perf_counter()-start,
        valid=sum(r['valid'] for r in rows),source_sha256=e.sha(source['source']),baseline_unchanged=True))

if __name__=='__main__':main()
