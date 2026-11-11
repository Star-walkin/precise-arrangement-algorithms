import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,sys,time,cv2
from common20 import *
sys.path.insert(0,str(BASE))
import transfer_engine as e

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--id',required=True);ap.add_argument('--frame',type=int,required=True);ap.add_argument('--d',type=float,default=2238.7);a=ap.parse_args()
    folder=ROOT/'probes'/a.id;path=folder/f'f3_{a.frame:06d}.npz';r=record(a.id,a.frame)
    with np.load(path) as f:data={k:f[k] for k in f.files}
    data['model_envelope']=e.model_envelope(data['surface'],np.asarray(r['boundary']['top_curve_xy'])*.5)
    tick=time.time();curves,diag,points=e.group_and_fit(data,a.d*.5);cross,rejected=e.intersections(curves,points,data,a.d*.5)
    dump(folder/f'groups_{a.frame:06d}.json',dict(d=a.d,curves=curves,crossings=cross,rejected=rejected,diag=diag,elapsed_s=time.time()-tick,coordinates='half_resolution'))
    print('groups',len(curves),'crossings',len(cross),'time',time.time()-tick,flush=True)
    item=next(x for x in plan() if x['id']==a.id);cap=cv2.VideoCapture(item['source']);cap.set(1,a.frame);ok,raw=cap.read();assert ok;cap.release()
    im=cv2.flip(raw,1) if item['camera']=='right' else raw;ann=im.copy()
    for c in curves:cv2.polylines(ann,[np.rint(np.asarray(c['xy'])*2).astype(np.int32)],False,(0,255,0),2)
    for p in points[::3,1:3]*2:cv2.circle(ann,tuple(np.rint(p).astype(int)),2,(0,255,255),-1)
    for seg in r['track'].get('supported_segments_xy',[]):cv2.polylines(ann,[np.rint(seg).astype(np.int32)],False,(255,0,255),3)
    cv2.polylines(ann,[np.rint(r['boundary']['top_curve_xy']).astype(np.int32)],False,(255,255,0),3)
    cv2.line(ann,(round(a.d),1700),(round(a.d),2999),(255,0,255),2)
    c=lambda x:cv2.resize(x[1800:3000,1200:3000],(1200,800))
    cv2.imencode('.jpg',np.hstack([c(im),c(ann)]))[1].tofile(folder/f'groups_{a.frame:06d}.jpg')
if __name__=='__main__':main()
