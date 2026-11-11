"""Exploratory texture motion/recurrence check. No production decisions."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import cv2,numpy as np,json
from scipy.signal import find_peaks
R=Path(__file__).resolve().parent
frames=np.load(R/'inspection/flange_probe.npz')['frames'];cv2.setNumThreads(1)
# Gold face above coil, clear of the changing incoming rope.
rois=[(80,115,250,150),(250,120,410,165),(400,155,530,190),(100,80,350,115)]
out=[]
for roi in rois:
    x0,y0,x1,y1=roi
    gr=np.array([cv2.cvtColor(f[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY) for f in frames])
    feat=np.array([cv2.resize(g.astype(float)-cv2.GaussianBlur(g.astype(float),(0,0),3),(60,20)).ravel() for g in gr])
    feat-=feat.mean(axis=1,keepdims=True);feat/=np.maximum(np.linalg.norm(feat,axis=1,keepdims=True),1e-6)
    raw=[float(np.mean(np.sum(feat[k:]*feat[:-k],axis=1))) for k in range(1,101)]
    ff=feat-feat.mean(axis=0);ff/=np.maximum(np.linalg.norm(ff,axis=1,keepdims=True),1e-6)
    scores=[float(np.mean(np.sum(ff[k:]*ff[:-k],axis=1))) for k in range(1,101)]
    peaks=find_peaks(scores,distance=5,prominence=.01)[0]
    flows=[]
    for i in range(200,300):
        p=cv2.goodFeaturesToTrack(gr[i],80,.02,5)
        if p is None:continue
        q,st,e=cv2.calcOpticalFlowPyrLK(gr[i],gr[i+1],p,None,winSize=(15,15),maxLevel=2)
        d=(q-p).reshape(-1,2)[st.ravel()>0]
        flows.append(np.median(d,axis=0).tolist())
    out.append(dict(roi=roi,raw_correlation_lags_1_10_25_50=[raw[k-1] for k in [1,10,25,50]],
        peaks=[dict(lag_s=float((k+1)/10),score=scores[k]) for k in sorted(peaks,key=lambda k:scores[k],reverse=True)[:8]],
        flow_median=np.median(flows,axis=0).tolist(),flow_absolute_median=np.median(np.abs(flows),axis=0).tolist(),scores=scores))
(R/'inspection/texture_probe.json').write_text(json.dumps(out,indent=2))
for r in out:print({k:v for k,v in r.items() if k!='scores'})
