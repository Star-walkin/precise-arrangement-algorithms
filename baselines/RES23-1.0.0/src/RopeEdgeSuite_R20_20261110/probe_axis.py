import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import cv2,argparse,time
from common20 import *
from local_axis22 import CoilAxis
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--id',required=True);ap.add_argument('--frames',required=True);ap.add_argument('--d',type=float,required=True);a=ap.parse_args()
    item=next(x for x in plan() if x['id']==a.id);cap=cv2.VideoCapture(item['source']);det=CoilAxis();out=ROOT/'axis_probes_v3'/a.id;out.mkdir(parents=True,exist_ok=True)
    for i in map(int,a.frames.split(',')):
        cap.set(1,i);ok,raw=cap.read();assert ok;im=cv2.flip(raw,1) if item['camera']=='right' else raw;r=record(a.id,i)
        tick=time.time();q=det.step(im,r['track'],a.d);dump(out/f'{i:06d}.json',dict(d=a.d,result=q,elapsed_s=time.time()-tick))
        print(i,q['valid'],q.get('differs_from_upstream_y_px'),q['reasons'],flush=True)
        ann=im.copy()
        for pt in q.get('points_xy',[]):cv2.circle(ann,tuple(np.rint(pt).astype(int)),4,(0,255,255),-1)
        for w in q.get('windows',[]):
            if 'selected' in w:
                for pt in w['selected']['points_xy']:cv2.circle(ann,tuple(np.rint(pt).astype(int)),4,(0,255,255),-1)
        if q['valid']:
            abc=q['line_abc'];xx=np.array([a.d-800,a.d]);yy=-(abc[0]*xx+abc[2])/abc[1]
            cv2.polylines(ann,[np.rint(np.c_[xx,yy]).astype(np.int32)],False,(0,255,0),3)
        cv2.polylines(ann,[np.rint(r['boundary']['top_curve_xy']).astype(np.int32)],False,(255,255,0),3)
        c=lambda x:cv2.resize(x[1700:3000,800:3400],(1300,650))
        cv2.imencode('.jpg',np.vstack([c(im),c(ann)]))[1].tofile(out/f'{i:06d}.jpg')
    cap.release()
if __name__=='__main__':main()
