"""Recompute only changed boundary stage from exact current-frame stage records."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,sys,time
import cv2
from PIL import Image
from common import *
from boundary_bg9_candidate import Boundary
from render_batch import panel

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--id',required=True);ap.add_argument('--end',type=int,required=True)
    ap.add_argument('--revision',required=True);ap.add_argument('--source-root');a=ap.parse_args();cv2.setNumThreads(1)
    source_root=Path(a.source_root) if a.source_root else ROOT
    item=next(x for x in read(ROOT/'plan.json')['videos'] if x['id']==a.id)
    out=ROOT/'experiments'/a.revision/a.id;out.mkdir(parents=True,exist_ok=False)
    cap=cv2.VideoCapture(item['source']);boundary=Boundary();rows=[];start=time.time()
    for i in range(a.end):
        ok,raw=cap.read();assert ok;r=record(source_root/'runs'/a.id,i);assert pixelsha(raw)==r['source_pixel_sha256']
        im=cv2.flip(raw,1) if item['camera']=='right' else raw
        old=r['boundary'];b=boundary.step(im,r['proposal'],r['track'],i)
        if not r['camera']['confirmed'] or r['camera']['camera_changed']:
            b['fit_valid']=False;b['count_boundary_valid']=False;b['reasons'].append('camera_motion_invalid')
            if r['camera']['camera_changed']:boundary=Boundary()
        r['boundary']=b;r['boundary_replay_current_frame_cache_verified']=True
        save_record(out,i,r)
        if i in [10,120,140,150,160,a.end-1]:Image.fromarray(panel(raw,r,item)).save(out/f'preview_{i:06d}.jpg',quality=94)
        rows.append(dict(frame=i,old=old['fit_valid'],new=b['fit_valid'],old_y=old['edge_y'],new_y=b['edge_y'],reasons=b['reasons']))
    cap.release();dump(out/'summary.json',dict(rows=rows,old=sum(r['old'] for r in rows),new=sum(r['new'] for r in rows),elapsed_s=time.time()-start,
        changed_stage='BG9 candidate only',unchanged_stages=f'exact current-frame records from {source_root}; source pixel hash checked',code_sha=sha(ROOT/'boundary_bg9_candidate.py')))
    print(a.id,len(rows),'old',sum(r['old'] for r in rows),'new',sum(r['new'] for r in rows),'seconds',time.time()-start)
if __name__=='__main__':main()
