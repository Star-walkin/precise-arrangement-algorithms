import cv2,numpy as np
from scipy.ndimage import gaussian_filter1d
from common import *
IDS=[20,100,250,316,336,369,396,450,526,599]
def features(im,reg):
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)
    curve=np.asarray(reg['top_curve_xy']);col=[]
    if not len(curve):return []
    for x in np.arange(D-3*PITCH,D+3*PITCH,8):
        edge=np.interp(x,curve[:,0],curve[:,1]);ys=np.arange(-PITCH,8*PITCH,.5)
        xs=np.arange(x-6,x+7,2);slope=(np.interp(x+12,curve[:,0],curve[:,1])-np.interp(x-12,curve[:,0],curve[:,1]))/24
        X=np.tile(xs,(len(ys),1)).astype(np.float32);Y=(edge+ys[:,None]+slope*(xs[None,:]-x)).astype(np.float32)
        a=np.median(cv2.remap(gray,X,Y,cv2.INTER_LINEAR),axis=1)
        below=a[ys>=4*PITCH];threshold=max(100,float(np.median(below)+4*np.median(abs(below-np.median(below)))+30))
        hits=(a>threshold)&(ys<3*PITCH)&(ys>-.5*PITCH)
        change=np.diff(np.r_[False,hits,False].astype(int));runs=(np.flatnonzero(change==-1)-np.flatnonzero(change==1))*.5
        # Band-passed rope profile strength near the boundary vs deeper surface.
        band=gaussian_filter1d(a,1.0)-gaussian_filter1d(a,PITCH*.7)
        near=np.sqrt(np.mean(band[(ys>.15*PITCH)&(ys<2.5*PITCH)]**2))
        deep=np.sqrt(np.mean(band[(ys>4*PITCH)&(ys<7*PITCH)]**2))
        col.append(dict(x=float(x),broad_bright_width=max(runs,default=0.),threshold=threshold,
            near_band=float(near),deep_band=float(deep),profile=a.tolist(),offset=ys.tolist()))
    return col
def main():
    out=ROOT/'boundary_development';out.mkdir(exist_ok=True)
    cap=cv2.VideoCapture(read(RUN12/'binding.json')['source']);cv2.setNumThreads(2)
    for i in IDS:
        cap.set(1,i);ok,im=cap.read();assert ok;r=record(i);v=features(im,r['region']);dump(out/f'{i:06d}.json',v)
        print(i,'width',np.round([q['broad_bright_width'] for q in v],1),'band ratio',round(np.median([q['near_band']/max(q['deep_band'],.01) for q in v]),2),flush=True)
    cap.release()
if __name__=='__main__':main()
