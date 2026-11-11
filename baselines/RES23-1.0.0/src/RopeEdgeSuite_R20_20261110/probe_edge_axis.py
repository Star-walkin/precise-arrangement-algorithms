import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,cv2
from common20 import *
from coil_edge_axis import measure
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--id',required=True);ap.add_argument('--frames',required=True);ap.add_argument('--d',type=float,required=True);a=ap.parse_args()
    item=next(x for x in plan() if x['id']==a.id);cap=cv2.VideoCapture(item['source']);out=ROOT/'axis_edge_probes'/a.id;out.mkdir(parents=True,exist_ok=True)
    for i in map(int,a.frames.split(',')):
        cap.set(1,i);ok,im=cap.read();assert ok
        if item['camera']=='right':im=cv2.flip(im,1)
        r=record(a.id,i);q=measure(im,r['track'],a.d);dump(out/f'{i:06d}.json',dict(d=a.d,result=q));print(i,q['valid'],q.get('differs_from_upstream_y_px'),q['reasons'],flush=True)
        ann=im.copy()
        for p in q.get('points_xy',[]):cv2.circle(ann,tuple(np.rint(p).astype(int)),3,(0,255,255),-1)
        cv2.polylines(ann,[np.rint(r['boundary']['top_curve_xy']).astype(np.int32)],False,(255,255,0),3)
        crop=lambda x:cv2.resize(x[1700:3000,800:3400],(1300,650))
        cv2.imencode('.jpg',np.vstack([crop(im),crop(ann)]))[1].tofile(out/f'{i:06d}.jpg')
    cap.release()
if __name__=='__main__':main()
