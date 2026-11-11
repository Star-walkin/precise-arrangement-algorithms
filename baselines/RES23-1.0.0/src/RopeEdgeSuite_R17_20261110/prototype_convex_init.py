import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import cv2,numpy as np
from PIL import Image,ImageDraw
from common import *
from boundary_bg9 import design
from render import pic,polyline,font

def consensus(pts,w):
    X=design(pts[:,0],w);y=pts[:,1];rng=np.random.default_rng(2311);best=None
    bins=np.floor((pts[:,0]-pts[:,0].min())/(np.ptp(pts[:,0])+1)*8).astype(int)
    for _ in range(500):
        ids=rng.choice(len(pts),5,replace=False)
        if len(set(bins[ids]))<4 or np.linalg.cond(X[ids])>1e4:continue
        c=np.linalg.solve(X[ids],y[ids]);keep=abs(y-X@c)<.35*PITCH
        if keep.sum()<.65*len(pts) or np.ptp(pts[keep,0])<.7*np.ptp(pts[:,0]) or len(set(bins[keep]))<6:continue
        for j in range(2):
            c=np.linalg.lstsq(X[keep],y[keep],rcond=None)[0];keep=abs(y-X@c)<.35*PITCH
        if keep.sum()<.65*len(pts) or np.ptp(pts[keep,0])<.7*np.ptp(pts[:,0]) or len(set(bins[keep]))<6:continue
        u=design(np.linspace(pts[keep,0].min(),pts[keep,0].max(),80),w)[:,1]
        if np.min(2*c[2]+6*c[3]*u+12*c[4]*u*u)<=10:continue
        score=(int(keep.sum()),-float(np.quantile(abs(y[keep]-X[keep]@c),.9)))
        if best is None or score>best[0]:best=(score,c,keep)
    return best

def main():
    cv2.setNumThreads(1);id='dual_20260923_113305__right';item=next(x for x in read(ROOT/'plan.json')['videos'] if x['id']==id)
    cap=cv2.VideoCapture(item['source']);rows=[]
    for i in [0,3,5,12,20,23,24,40,70]:
        r=record(ROOT/'runs'/id,i);b=r['boundary'];excluded={tuple(p) for p in b['excluded_candidates_xy']}
        pts=np.array([p for p in b['raw_candidates_xy'] if tuple(p) not in excluded]);got=consensus(pts,4096)
        row=dict(frame=i,found=got is not None)
        if got:
            score,c,keep=got;row.update(inliers=int(keep.sum()),total=len(pts),residual_q90=-score[1],coefficient=c.tolist())
            cap.set(1,i);ok,raw=cap.read();assert ok;im=cv2.flip(raw,1);ann=im.copy()
            xx=np.arange(pts[keep,0].min(),pts[keep,0].max(),4);polyline(ann,np.c_[xx,design(xx,4096)@c],(255,200,20),5)
            for p in pts[keep]:cv2.circle(ann,tuple(np.rint(p).astype(int)),4,(20,255,20),-1)
            for p in pts[~keep]:cv2.circle(ann,tuple(np.rint(p).astype(int)),5,(20,20,255),-1)
            img=Image.new('RGB',(1400,1030),'#111b25');d=ImageDraw.Draw(img)
            d.text((15,5),f'113305 right {i/10:.1f}s | convex consensus {keep.sum()}/{len(pts)}',font=font(25),fill='white')
            img.paste(pic(im[1700:2700,700:3400],(1400,490)),(0,45));img.paste(pic(ann[1700:2700,700:3400],(1400,490)),(0,540))
            path=ROOT/'experiments'/f'convex_initial_{i:06d}.jpg';path.parent.mkdir(exist_ok=True);img.save(path,quality=94)
        rows.append(row);print(row,flush=True)
    cap.release();dump(ROOT/'experiments/convex_initial_probe.json',rows)
if __name__=='__main__':main()
