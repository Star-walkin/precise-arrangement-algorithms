"""Untuned region regression samples on the second transfer video."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,hashlib,time
import cv2
from diagnose import ROOT,OLD,read
from region_bg5 import segment

def main():
    cv2.setNumThreads(1);run=OLD/'transfer_runs/dual_20260924_111143';b=read(run/'binding.json')
    cap=cv2.VideoCapture(b['source']);rows=[];start=time.perf_counter()
    for i in range(0,601,30):
        cap.set(1,i);ok,im=cap.read();assert ok
        old=read(run/'bg1/frames'/f'{i:06d}.json');assert hashlib.sha256(im.tobytes()).hexdigest()==old['source_pixel_sha256']
        new=segment(im)['record'];rows.append(dict(frame=i,old_valid=old['region']['valid'],new_valid=new['valid'],
            old_edge=old['region'].get('edge_y'),new_edge=new.get('edge_y'),reasons=new['reasons'],
            S1_valid=old['s1']['valid'],flange_presence=new.get('flange_presence')))
    cap.release();result=dict(session=run.name,method='BG5 region-only 21 equally spaced frames; no F3/full-pipeline rerun',
        frames=len(rows),old_valid=sum(r['old_valid'] for r in rows),new_valid=sum(r['new_valid'] for r in rows),rows=rows,
        elapsed_s=time.perf_counter()-start,accuracy_validated=False)
    (ROOT/'control_probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print({k:v for k,v in result.items() if k!='rows'})

if __name__=='__main__':main()
