import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,cv2
from PIL import Image
from common import *
from tracker_sg5 import Fusion
from boundary_bg10 import Boundary
from render import panel
RELAXED=PARENT/'RopeEdgeSuite_R15_20261109';STRICT=PARENT/'RopeEdgeSuite_Batch_RES14_20261109';PREVIOUS=PARENT/'RopeEdgeSuite_R17_20261110'
def main(id,n):
    cv2.setNumThreads(1);item=next(x for x in read(ROOT/'plan.json')['videos'] if x['id']==id)
    cap=cv2.VideoCapture(item['source']);fusion=Fusion();bdy=Boundary();rows=[];out=ROOT/'probes'/id;out.mkdir(parents=True,exist_ok=True)
    for i in range(min(n,item['planned_frames'])):
        ok,raw=cap.read();assert ok;im=cv2.flip(raw,1) if item['camera']=='right' else raw
        r=record(RELAXED/'runs'/id,i);s=record(STRICT/'runs'/id,i)
        assert pixelsha(im)==r['analysis_pixel_sha256']==s['analysis_pixel_sha256']
        r['track']=fusion.step(im,s['track'],r['track'],i/item['fps'],i)
        b=bdy.step(im,r['proposal'],r['track'],i)
        if not r['camera']['confirmed'] or r['camera']['camera_changed']:
            b['fit_valid']=False;b['count_boundary_valid']=False;b['reasons'].append('camera_motion_invalid')
            if r['camera']['camera_changed']:bdy=Boundary()
        r['boundary']=b;abc=r['track']['line_abc'];r['original_line_abc']=None if abc is None else ([-abc[0],abc[1],abc[0]*(raw.shape[1]-1)+abc[2]] if item['camera']=='right' else abc)
        save_record(out,i,r)
        oldpath=PREVIOUS/'runs'/id/'frames'/f'{i:06d}.json.gz'
        old=record(PREVIOUS/'runs'/id,i)['boundary']['fit_valid'] if oldpath.exists() else None
        rows.append(dict(frame=i,valid=b['fit_valid'],old_valid=old,initialization=b['convex_initialization'],reasons=b['reasons']))
        if i in [0,2,5,12,n-1]:Image.fromarray(panel(raw,r,item)).save(out/f'preview_{i:06d}.jpg',quality=95)
    cap.release();dump(out/'summary.json',dict(id=id,frames=len(rows),valid=sum(r['valid'] for r in rows),
        old_valid=sum(bool(r['old_valid']) for r in rows),rows=rows));print(id,len(rows),'new',sum(r['valid'] for r in rows),'old',sum(bool(r['old_valid']) for r in rows),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--id',required=True);ap.add_argument('--frames',type=int,default=80);a=ap.parse_args();main(a.id,a.frames)
