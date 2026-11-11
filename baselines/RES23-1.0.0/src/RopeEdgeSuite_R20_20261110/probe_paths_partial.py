"""Partial AVI files have no reliable frame index: decode sequentially."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import cv2,time
from common20 import *
from roi20 import masks
from probe_paths_original import AuditedP1,normalized

def main():
    for item in plan():
        if not item.get('partial_source'):continue
        job=next(j for j in read(ROOT/'path_sampling_plan.json')['jobs'] if j['id']==item['id'])
        out=ROOT/'paths_partial_sequential'/item['id'];out.mkdir(parents=True,exist_ok=True);cap=cv2.VideoCapture(item['source']);p1=None
        for i in range(item['planned_frames']):
            ok,raw=cap.read();assert ok;r=record(item['id'],i);assert pixelsha(raw)==r['source_pixel_sha256']
            # Three diagnostic frames suffice: duration is too short for the
            # required multi-second calibration even if all bends are clean.
            if i not in [job['frames'][0],job['frames'][len(job['frames'])//2],job['frames'][-1]]:continue
            dest=out/f'p1_{i:06d}.json'
            if dest.exists():continue
            im=cv2.flip(raw,1);surface,_,meta=masks(im,r);surface_original=cv2.flip(surface,1)
            if p1 is None:p1=AuditedP1(surface_original,PITCH_PRIOR,LC1/'supplied_assets/step_0750.pt',threads=2,camera=item['camera'],surface_mode='entry_band')
            else:p1.surface=surface_original
            p1.candidates=[];tick=time.time();res=p1.step(raw);res['candidate_paths_audit']=p1.candidates
            dump(dest,dict(id=item['id'],frame=i,time_s=i/item['fps'],source_pixel_sha256=pixelsha(raw),analysis_pixel_sha256=pixelsha(im),
                raw_result=res,result=normalized(res,True),decode_mode='sequential_no_seeking',code_sha256=sha(__file__),elapsed_s=time.time()-tick))
            print(item['id'],i,res['status'],flush=True)
        cap.release()
if __name__=='__main__':main()
