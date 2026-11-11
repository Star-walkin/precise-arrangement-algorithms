"""Causal BG9 replay with exact same-frame inputs and verified stage provenance."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import argparse,time,traceback
import cv2
from PIL import Image
from common import *
from boundary_bg9 import Boundary
from render import panel,writer
from tracker_sg5 import Fusion
RELAXED=PARENT/'RopeEdgeSuite_R15_20261109'
STRICT=PARENT/'RopeEdgeSuite_Batch_RES14_20261109'

def wait_for(path,follow):
    start=time.time()
    while not path.exists():
        if not follow or time.time()-start>1800:raise RuntimeError('Missing stage record: '+str(path))
        time.sleep(.5)

def process(item,follow=False):
    cv2.setNumThreads(1);id=item['id'];source=RELAXED/'runs'/id;strict_source=STRICT/'runs'/id;out=ROOT/'runs'/id
    out.mkdir(parents=True,exist_ok=False);wait_for(source/'binding.json',follow)
    files=[ROOT/'boundary_bg9.py',ROOT/'run_one.py',ROOT/'render.py',ROOT/'common.py',ROOT/'tracker_sg5.py',R12/'tracker_sg2.py']
    hashes={str(p):sha(p) for p in files};dump(out/'binding.json',dict(algorithm='RES-1.7/SG5-BG9-CG2',strict_stage_run=str(strict_source),strict_binding_sha=sha(strict_source/'binding.json'),
        source_stage_run=str(source),source_stage_binding_sha=sha(source/'binding.json'),source_item=item,
        boundary_cold_start=True,every_frame_in_order=True,core_hashes=hashes,
        branches='strict SG2 and relaxed SG3 cached current-image candidates, pixel hashes verified; SG5 and BG9 recomputed causally',
        d_recalibrated=False,F3_run=False,count_run=False))
    cap=cv2.VideoCapture(item['source']);n=item['planned_frames'];fps=item['fps'];bdy=Boundary();fusion=Fusion();rows=[]
    video=out/(id+'__RES17_SG5_BG9.mp4');enc=writer(video,fps);start=time.time()
    shots=set([0,min(10,n-1),n//4,n//2,3*n//4,n-1])
    try:
        for i in range(n):
            wait_for(source/'frames'/f'{i:06d}.json.gz',follow)
            r=record(source,i);assert not r['algorithm_errors'];ok,raw=cap.read();assert ok
            assert pixelsha(raw)==r['source_pixel_sha256']
            im=cv2.flip(raw,1) if item['camera']=='right' else raw
            assert pixelsha(im)==r['analysis_pixel_sha256']
            s=record(strict_source,i)
            assert s['analysis_pixel_sha256']==r['analysis_pixel_sha256'] and s['source_pixel_sha256']==r['source_pixel_sha256']
            relaxed_track=r['track'];r['track']=fusion.step(im,s['track'],relaxed_track,i/fps,i)
            abc=r['track']['line_abc']
            r['original_line_abc']=None if abc is None else ([-abc[0],abc[1],abc[0]*(raw.shape[1]-1)+abc[2]] if item['camera']=='right' else abc)
            r['strict_stage_record_sha256']=sha(strict_source/'frames'/f'{i:06d}.json.gz')
            b=bdy.step(im,r['proposal'],r['track'],i)
            if not r['camera']['confirmed'] or r['camera']['camera_changed']:
                b['fit_valid']=False;b['count_boundary_valid']=False;b['reasons'].append('camera_motion_invalid')
                if r['camera']['camera_changed']:bdy=Boundary()
            r['boundary']=b;r['stages']=dict(BG5='verified_same_frame_cache',SG3='verified_same_frame_cache',SG2='verified_same_frame_cache',SG5='fresh_current_entry_validation_and_branch_selection',
                CG2='verified_same_frame_cache',BG9='fresh_current_image_causal_fit',F3='not_run',counter='not_run',physical_contact='not_certified')
            r['source_stage_record_sha256']=sha(source/'frames'/f'{i:06d}.json.gz')
            save_record(out,i,r);rgb=panel(raw,r,item);enc.stdin.write(rgb.tobytes())
            if i in shots:Image.fromarray(rgb).save(out/f'preview_{i:06d}.jpg',quality=93)
            rows.append(dict(frame=i,time_s=i/fps,tracking_valid=r['track']['valid'],proposal_valid=r['proposal']['valid'],
                boundary_fit=b['fit_valid'],count_boundary_eligible=b['count_boundary_valid'],d_locally_observed=b.get('d_locally_observed'),
                foreground_at_d=b.get('d_foreground_occluded'),boundary_reasons=b['reasons'],
                tracking_reason=r['track']['diagnostics']['accept_reason'],selected_branch=r['track']['diagnostics']['SG5']['branch'],fallback_rejection=(r['track']['diagnostics']['SG5'].get('fallback_check') or {}).get('reasons',[]),branch_disagreement=r['track']['diagnostics']['SG5']['branch_disagreement'],algorithm_errors=[],model_forwards=0))
            if i%20==0 or i==n-1:
                state=dict(id=id,processed=i+1,planned=n,tracking=sum(x['tracking_valid'] for x in rows),
                    boundary=sum(x['boundary_fit'] for x in rows),errors=0,elapsed_s=round(time.time()-start,1))
                dump(out/'progress.json',state);print(state,flush=True)
        cap.release();enc.stdin.close();error=enc.stderr.read();assert enc.wait()==0,error
        wait_for(source/'receipt.json',follow);sr=read(source/'receipt.json');assert sr['status']=='complete' and sr['algorithm_error_frames']==0
        assert sr['frames']==n and sr['source_unchanged'] and sr['core_unchanged']
        dump(out/'compact.json',rows);check=cv2.VideoCapture(str(video));count=0
        while True:
            ok,frame=check.read()
            if not ok:break
            assert frame.shape==(1080,1920,3);count+=1
        outputfps=check.get(5);check.release();assert count==n and abs(outputfps-fps)<.001
        assert all(sha(p)==h for p,h in hashes.items());assert sha(item['source'])==item['source_sha256']
        dump(out/'receipt.json',dict(status='complete',frames=n,decoded_output_frames=count,fps=outputfps,video=str(video),
            video_sha256=sha(video),source_unchanged=True,core_unchanged=True,algorithm_error_frames=0,strict_stage_binding_sha256=sha(strict_source/'binding.json'),
            pixel_hash_verified_frames=n,base_stage_receipt_sha256=sha(source/'receipt.json'),skipped_frames=0,
            F3_run=False,count_run=False,elapsed_s=time.time()-start,partial_source=item.get('partial_source',False)))
    except BaseException:
        cap.release()
        if enc.stdin and not enc.stdin.closed:enc.stdin.close()
        enc.wait(timeout=60);dump(out/'failure.json',dict(error=traceback.format_exc(),completed_frames=len(rows)));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--id',required=True);ap.add_argument('--follow',action='store_true');a=ap.parse_args()
    item=next(x for x in read(ROOT/'plan.json')['videos'] if x['id']==a.id);process(item,a.follow)
