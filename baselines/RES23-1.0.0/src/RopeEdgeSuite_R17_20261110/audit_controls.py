import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import cv2
from common import *
from tracker_sg5 import Fusion
OLD=PARENT/'RopeEdgeSuite_R15_20261109';STRICT=PARENT/'RopeEdgeSuite_Batch_RES14_20261109'
def main():
    cv2.setNumThreads(1);rows=[];scope=read(ROOT/'control_scope.json')
    for case in scope['sources']:
        item=next(x for x in read(ROOT/'plan.json')['control_sources'] if x['id']==case['id'])
        n=case['available_frame_count'];cap=cv2.VideoCapture(item['source']);fusion=Fusion();positive=[];checks=[]
        for i in range(n):
            ok,raw=cap.read();assert ok;im=cv2.flip(raw,1) if item['camera']=='right' else raw
            r=record(OLD/'runs'/item['id'],i);s=record(STRICT/'runs'/item['id'],i)
            assert pixelsha(im)==r['analysis_pixel_sha256']==s['analysis_pixel_sha256']
            o=fusion.step(im,s['track'],r['track'],i/item['fps'],i)
            if o['valid']:positive.append(i)
            checks.append(dict(frame=i,strict=bool(s['track']['valid']),relaxed=bool(r['track']['valid']),chosen=bool(o['valid'])))
        cap.release();rows.append(dict(id=item['id'],tested_frames=n,source_frames=item['planned_frames'],
            frame_range=[0,n-1],detected_frames=positive,pixel_verified=True,checks=checks))
        print(item['id'],n,'detections',positive,flush=True)
    dump(ROOT/'control_audit.json',dict(scope='Continuous initial samples only; not a full-video specificity estimate',
        frames=sum(r['tested_frames'] for r in rows),detections=sum(len(r['detected_frames']) for r in rows),sources=rows))
if __name__=='__main__':main()
