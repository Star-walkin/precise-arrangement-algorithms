import cv2,sys,argparse
from PIL import Image,ImageDraw,ImageFont
from common20 import *
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--id',required=True);ap.add_argument('--frame',type=int,required=True);ap.add_argument('--folder',default='probes');a=ap.parse_args()
    item=next(x for x in plan() if x['id']==a.id);out=ROOT/a.folder/a.id;r=record(a.id,a.frame)
    cap=cv2.VideoCapture(item['source']);cap.set(1,a.frame);ok,raw=cap.read();cap.release();assert ok
    im=cv2.flip(raw,1) if item['camera']=='right' else raw;ann=im.copy()
    p=out/f'p1_{a.frame:06d}.json';f=out/f'f3_{a.frame:06d}.npz'
    if p.exists():
        rr=read(p)['result'];pts=np.asarray(rr['path_xy'],float).reshape(-1,2);d=rr['diagnostics']
        for j,q in enumerate(pts):
            sup=d.get('path_point_support',[False]*len(pts))[j];stage=d.get('path_stage',['free']*len(pts))[j]
            cv2.circle(ann,tuple(np.rint(q).astype(int)),4,(0,255,0) if sup and stage=='surface' else ((0,255,255) if sup else (0,0,255)),-1)
    if f.exists():
        z=np.load(f);pts=z['points']
        for x,y in pts[::2,1:3]*2:cv2.circle(ann,(round(x),round(y)),2,(0,255,255),-1)
    for seg in r['track'].get('supported_segments_xy',[]):cv2.polylines(ann,[np.rint(seg).astype(np.int32)],False,(255,0,255),3)
    if r['boundary']['fit_valid']:cv2.polylines(ann,[np.rint(r['boundary']['top_curve_xy']).astype(np.int32)],False,(255,255,0),3)
    crop=lambda x:cv2.resize(x[1700:3000,:3600],(1800,650))
    cv2.imencode('.jpg',np.vstack([crop(im),crop(ann)]))[1].tofile(out/f'review_{a.frame:06d}.jpg')
    print(str(out/f'review_{a.frame:06d}.jpg'))
if __name__=='__main__':main()
