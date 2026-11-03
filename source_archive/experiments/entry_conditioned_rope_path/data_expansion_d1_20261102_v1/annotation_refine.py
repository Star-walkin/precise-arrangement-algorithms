"""Original-gray bounded center refinement. No U-Net heat or target labels.

A7 geometry may initialize a coarse axis, never labels a segment accepted.
Caller supplies visually checked finite anchors and checks resulting overlays.
"""
from pathlib import Path
import json,hashlib
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[3]
A7=ROOT/'数据资产/05_派生与实验结果/camera_condition_free_straight_20261030_v7/all_001/记录'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sample(a,p):
    p=np.asarray(p,np.float32);shape=p.shape[:-1]
    return cv2.remap(a,p[...,0].reshape(len(p),-1),p[...,1].reshape(len(p),-1),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101).reshape(shape)
def coarse_axis(job):
    seed=np.asarray(job['seeds'][0]['xy'],float);d=seed[-1]-seed[0];d/=np.linalg.norm(d)
    p=A7/(job['camera_role']+'_camera')/(job['id']+'.json')
    candidates=load(p)['result']['candidates'];choices=[]
    for c in candidates:
        dc=np.asarray(c['direction_xy'],float);dc/=np.linalg.norm(dc)
        if dc@d<0:dc=-dc
        o=np.asarray(c['origin_xy'],float);n=np.array([-dc[1],dc[0]])
        angle=np.degrees(np.arccos(np.clip(dc@d,-1,1)));offset=abs((seed-o)@n).mean()
        if angle<4 and offset<max(c['width_px'],12):choices.append((angle+offset*.10,c,dc))
    if choices:
        cost,c,dc=min(choices,key=lambda v:v[0]);o=np.asarray(c['origin_xy'],float)
        return dict(origin=o,direction=dc,width=float(c['width_px']),source=str(p),source_sha256=sha(p),candidate_id=c['candidate_id'],A7_is_raw_proposal_not_identity_truth=True)
    return dict(origin=seed[0],direction=d,width=14.,source='frozen entry coarse axis fallback; width requires visual/raw check',candidate_id=None)
def axis_xy(axis,x):
    o=axis['origin'];d=axis['direction'];return np.c_[np.asarray(x,float),o[1]+(np.asarray(x)-o[0])*d[1]/d[0]]
def resample(points,step=4.):
    p=np.asarray(points,float);s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];ss=np.arange(0,s[-1]+.1,step)
    if ss[-1]<s[-1]:ss=np.r_[ss,s[-1]]
    return np.c_[np.interp(ss,s,p[:,0]),np.interp(ss,s,p[:,1])]
def refine_segment(image,anchors,width,max_shift=4.,step=4.,minimum_strength=.8):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)
    p=resample(anchors,step);t=np.gradient(p,axis=0);t/=np.maximum(np.linalg.norm(t,axis=1)[:,None],1e-8)
    n=np.c_[-t[:,1],t[:,0]];span=max(16.,width*1.4);ns=np.arange(-span,span+.1,.5)
    tang=np.linspace(-max(8.,width),max(8.,width),15)
    xy=p[:,None,None,:]+t[:,None,None,:]*tang[None,:,None,None]+n[:,None,None,:]*ns[None,None,:,None]
    prof=np.median(sample(gray,xy),axis=1);prof=cv2.GaussianBlur(prof,(5,1),.7)
    gradient=np.gradient(prof,.5,axis=1);offsets=np.arange(-max_shift,max_shift+.1,.5)
    widths=np.linspace(max(5.,width*.65),max(7.,width*1.35),9)
    opts=[];raw_scores=[];edgewidths=[];centers=[]
    rows=np.arange(len(p))
    for delta in offsets:
        for w in widths:
            li=int(np.argmin(abs(ns-(delta-w*.5))));ri=int(np.argmin(abs(ns-(delta+w*.5))))
            ci=int(np.argmin(abs(ns-delta)));lo=int(np.argmin(abs(ns-(delta-w*.8))));hi=int(np.argmin(abs(ns-(delta+w*.8))))
            for polarity in [1.,-1.]:
                strength=np.minimum(polarity*gradient[:,li],-polarity*gradient[:,ri])
                contrast=np.minimum(polarity*(prof[:,ci]-prof[:,lo]),polarity*(prof[:,ci]-prof[:,hi]))
                value=strength+.04*np.clip(contrast,0,50)-.12*abs(delta)-.10*abs(np.log(w/width))
                opts.append(value);raw_scores.append(strength);edgewidths.append(w);centers.append(delta)
    opts=np.stack(opts,axis=1);ix=opts.argmax(1);shift=np.asarray(centers)[ix]
    # Local median suppresses strand phase while preserving bounded corridor.
    shift=np.asarray([np.median(shift[max(0,k-3):k+4]) for k in range(len(shift))])
    corrected=p+n*shift[:,None];strength=np.stack(raw_scores,axis=1)[rows,ix]
    local_width=np.asarray(edgewidths)[ix]
    valid=(strength>=minimum_strength)&np.isfinite(corrected).all(1)&(abs(shift)<max_shift-.25)
    valid[:2]=False;valid[-2:]=False
    # Never interpolate unsupported gaps into observed supervision.
    diag=dict(method='native original grayscale tangential median; opposing derivative edges; bounded normal offset and width prior',
        network_heat_used=False,step_original_px=step,tangent_half_context_px=max(8.,width),max_normal_shift_px=max_shift,
        minimum_bilateral_strength=minimum_strength,initial_optical_width_px=width,width_is_not_calibrated_diameter=True,
        median_abs_shift_px=float(np.median(abs(shift))),p95_abs_shift_px=float(np.quantile(abs(shift),.95)),
        median_observed_width_px=float(np.median(local_width)),supported_fraction=float(valid.mean()),
        boundary_hit_fraction=float((abs(shift)>=max_shift-.25).mean()),boundary_hitting_centers_ignored=True)
    return corrected,valid,diag,dict(strength=strength.tolist(),shift=shift.tolist(),observed_width=local_width.tolist(),coarse_points=p.tolist())
def runs(valid,min_points=8):
    diff=np.diff(np.r_[False,valid,False].astype(np.int8))
    return [(a,b) for a,b in zip(np.flatnonzero(diff==1),np.flatnonzero(diff==-1)) if b-a>=min_points]
