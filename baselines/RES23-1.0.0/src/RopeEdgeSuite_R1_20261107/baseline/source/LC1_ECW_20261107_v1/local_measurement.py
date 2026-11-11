"""ECW local optical measurements. All public coordinates are original pixels.

Neither the yellow/coil interface nor a fitted S1 axis is physical-contact truth.
The current source frame and its byte-bound BG4 record are required together.
"""
from pathlib import Path
import json,hashlib
import cv2,numpy as np
from scipy.ndimage import gaussian_filter,uniform_filter1d
from scipy.signal import find_peaks
ROOT=Path(__file__).resolve().parent
BG4=ROOT.parent/'LC1_boundary_BG4_local_temporal_20261107_v1'
SOURCE=ROOT.parents[3]/'双相机同步/dual_20260924_111500/left_DA9627065.avi'
D=2238.7
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def plain(a):
    if isinstance(a,np.ndarray):return a.tolist()
    if isinstance(a,np.generic):return a.item()
    if isinstance(a,dict):return {k:plain(v) for k,v in a.items()}
    if isinstance(a,(list,tuple)):return [plain(v) for v in a]
    return a
def dump(path,a):Path(path).write_text(json.dumps(plain(a),ensure_ascii=False,allow_nan=False),encoding='utf8')
def line_y(abc,x):a,b,c=abc;return -(a*np.asarray(x)+c)/b
def robust_poly(x,y,degree):
    u=(np.asarray(x)-1900)/1000;keep=np.ones(len(u),bool)
    for _ in range(5):
        if keep.sum()<degree+5:return None,None,None
        c=np.polyfit(u[keep],np.asarray(y)[keep],degree)
        e=np.asarray(y)-np.polyval(c,u);mad=1.4826*np.median(abs(e[keep]-np.median(e[keep])))
        keep=abs(e)<max(5,2.5*mad)
    return c,keep,float(max(1,mad))

def reference_boundary(record,abc):
    """Exclude the straight-rope corridor before fitting visible surface reference.

    This is an interpolated visible interface, not the invisible metal contact line.
    Degree disagreement is a model sensitivity diagnostic, not a calibrated CI.
    """
    p=np.asarray(record['region'].get('top_observed_xy',[]),float).reshape(-1,2)
    if len(p)<20:return dict(valid=False,reason='no_boundary_observations')
    use=(p[:,0]>850)&(p[:,0]<2850)&(abs(p[:,1]-line_y(abc,p[:,0]))>48)
    q=p[use]
    if len(q)<25:return dict(valid=False,reason='few_points_outside_rope_corridor')
    c,k,mad=robust_poly(q[:,0],q[:,1],4);c3,_,_=robust_poly(q[:,0],q[:,1],3)
    if c is None or c3 is None:return dict(valid=False,reason='fit_failed')
    q=q[k];pred=float(np.polyval(c,(D-1900)/1000));disagree=abs(pred-np.polyval(c3,(D-1900)/1000))
    bracket=bool(q[:,0].min()<D-80 and q[:,0].max()>D+80)
    left=q[q[:,0]<D];right=q[q[:,0]>D]
    gap=float(right[:,0].min()-left[:,0].max()) if len(left) and len(right) else None
    xs=np.arange(900,2801,8);ys=np.polyval(c,(xs-1900)/1000)
    return dict(valid=bracket and mad<8 and disagree<12,polynomial=c,points_xy=q,
                curve_xy=np.c_[xs,ys],y_at_d=pred,residual_mad_px=mad,degree_disagreement_px=float(disagree),
                unsupported_span_at_d_px=gap,bracketed=bracket,reference_type='interpolated_visible_coil_interface')

def rope_trace(image,abc):
    xs=np.arange(1100,2441,4,dtype=np.float32);offset=np.arange(-100,101,dtype=np.float32)
    mx=np.broadcast_to(xs[None,:],(len(offset),len(xs))).copy()
    my=(line_y(abc,xs)[None,:]+offset[:,None]).astype(np.float32)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)
    g=cv2.remap(gray,mx,my,cv2.INTER_LINEAR)
    # Rectified along-axis averaging suppresses cable strand texture.
    g=gaussian_filter(g,(1.1,3.0))
    scores=[];widths=np.arange(12,37,4)
    for w in widths:
        center=uniform_filter1d(g,size=max(3,w//2),axis=0,mode='nearest')
        sides=(np.roll(g,w,axis=0)+np.roll(g,-w,axis=0))/2
        scores.append((center-sides)/np.maximum(25,(center+sides)/2))
    ss=np.stack(scores);wi=ss.argmax(0);score=ss.max(0)
    score[:38]=-1;score[-38:]=-1
    # Seeded first-order path, continuous but only supported sections are observations.
    value=np.clip(score[:,0],-1,1)-.06*abs(offset);back=np.zeros(score.shape,np.int16)
    for j in range(1,len(xs)):
        cand=[]
        for step in range(-3,4):
            v=np.roll(value,step)-.028*abs(step)
            if step>0:v[:step]=-1e5
            elif step<0:v[step:]=-1e5
            cand.append(v)
        cand=np.stack(cand);a=cand.argmax(0);back[:,j]=np.arange(len(offset))-(a-3)
        value=cand.max(0)+np.clip(score[:,j],-.4,.8)-.0006*abs(offset)
    idx=np.zeros(len(xs),int);idx[-1]=value.argmax()
    for j in range(len(xs)-1,0,-1):idx[j-1]=back[idx[j],j]
    yy=line_y(abc,xs)+offset[idx];chosen=score[idx,np.arange(len(xs))]
    ww=widths[wi[idx,np.arange(len(xs))]]
    second=[]
    for j,k in enumerate(idx):
        peaks,_=find_peaks(score[:,j],distance=12)
        other=peaks[abs(peaks-k)>10]
        second.append(float(score[other,j].max()) if len(other) else -1)
    supported=(chosen>.075)&(abs(offset[idx])<58)
    local=(xs>=D-120)&(xs<=D+80)
    near=np.flatnonzero(local&supported)
    valid=len(near)>=30
    y=float(np.interp(D,xs,yy));w=float(np.median(ww[local&supported])) if near.size else None
    return dict(valid=bool(valid),path_xy=np.c_[xs,yy],supported=supported,score=chosen,widths_px=ww,
                second_peak_score=second,local_supported_fraction=float(supported[local].mean()),
                y_at_d=y,optical_width_px=w,method='seeded_bright_ridge_DP_v1')

def measure(image,f):
    r=read(BG4/'full_003/frames'/f'{f:06d}.json')
    assert hashlib.sha256(image.tobytes()).hexdigest()==r['source_pixel_sha256']
    out=dict(frame=f,time_video_s=f/10,source_pixel_sha256=r['source_pixel_sha256'],s1_valid=r['s1']['valid'],s1_abc=r['s1']['current_line_abc'])
    if not out['s1_valid']:return out
    abc=out['s1_abc'];out['boundary']=reference_boundary(r,abc);out['rope']=rope_trace(image,abc)
    b=out['boundary'];a=out['rope']
    out['old_center_gap_px']=r['intersection_y_px']-r['region']['edge_y'] if r['region'].get('edge_y') is not None else None
    if b.get('y_at_d') is not None:out['new_center_gap_px']=a['y_at_d']-b['y_at_d']
    out['contact_confirmed']=False
    return plain(out)

def overlay(im,r):
    a=im.copy();b=r.get('boundary',{});p=r.get('rope',{})
    if b.get('curve_xy'):
        cv2.polylines(a,[np.rint(b['curve_xy']).astype(np.int32)],False,(255,200,0),2,cv2.LINE_AA)
        for pt in b['points_xy']:cv2.circle(a,tuple(np.rint(pt).astype(int)),3,(0,255,255),-1)
    if p:
        xy=np.rint(p['path_xy']).astype(np.int32)
        for j in range(1,len(xy)):
            if p['supported'][j] and p['supported'][j-1]:cv2.line(a,tuple(xy[j-1]),tuple(xy[j]),(0,220,0),3)
    cv2.line(a,(int(D),2000),(int(D),2650),(255,0,255),2)
    return a

if __name__=='__main__':
    from PIL import Image,ImageDraw
    out=ROOT/'v1';out.mkdir(exist_ok=True)
    cap=cv2.VideoCapture(str(SOURCE));rows=[];ids=[150,220,245,260,290,320]
    sheet=Image.new('RGB',(2000,len(ids)*290),'#151820');draw=ImageDraw.Draw(sheet)
    for j,f in enumerate(ids):
        cap.set(1,f);ok,im=cap.read();assert ok;r=measure(im,f);rows.append(r)
        a=overlay(im,r);crop=a[2070:2580,850:2850]
        pic=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)).resize((1000,255))
        raw=Image.fromarray(cv2.cvtColor(im[2070:2580,850:2850],cv2.COLOR_BGR2RGB)).resize((1000,255))
        sheet.paste(raw,(0,j*290+30));sheet.paste(pic,(1000,j*290+30))
        brief={k:r.get(k) for k in ['frame','old_center_gap_px','new_center_gap_px']};brief['boundary_ok']=r.get('boundary',{}).get('valid');brief['rope_ok']=r.get('rope',{}).get('valid')
        draw.text((8,j*290+8),str(brief),fill='white');print(brief,flush=True)
    cap.release();dump(out/'records.json',rows);sheet.save(out/'review.jpg',quality=96)
