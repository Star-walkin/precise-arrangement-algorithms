"""Conservative local free-rope fit and current visible-interface proximity.

v1's unconstrained ridge path and long-hole polynomial were rejected on review.
v2 measures only a neutral cable between two yellow background strips, then
reports finite extrapolation and fit-window sensitivity explicitly.
"""
import sys,time
from pathlib import Path
import cv2,numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from local_measurement import ROOT,BG4,SOURCE,D,read,dump,plain,line_y

def local_axis(image,abc):
    xs=np.arange(900,2101,4,dtype=np.float32);off=np.arange(-50,51,dtype=np.float32)
    mx=np.broadcast_to(xs[None,:],(len(off),len(xs))).copy();my=(line_y(abc,xs)[None,:]+off[:,None]).astype(np.float32)
    patch=cv2.remap(image,mx,my,cv2.INTER_LINEAR).astype(np.float32)
    # Average along the line, preserving normal width and two flanking colors.
    patch=gaussian_filter(patch,(1,3,0));b,g,r=cv2.split(patch)
    light=(b+g+r)/3;chroma=((r+g)/2-b)/np.maximum(light,12)
    rows=[]
    for j,x in enumerate(xs):
        neutral=(chroma[:,j]<.32)&(light[:,j]>24)
        edges=np.diff(np.r_[False,neutral,False].astype(int))
        choices=[]
        for lo,hi in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            w=hi-lo;c=(lo+hi-1)/2;delta=c-50
            if not(8<=w<=38 and abs(delta)<13 and lo>=8 and hi+8<len(off)):continue
            ca=float(np.median(chroma[lo-8:lo-3,j]));cb=float(np.median(chroma[hi+3:hi+8,j]))
            if ca<.43 or cb<.43:continue
            contrast=float(np.mean(light[lo:hi,j])-(np.mean(light[lo-8:lo-3,j])+np.mean(light[hi+3:hi+8,j]))/2)
            if contrast<4:continue
            choices.append((abs(delta),delta,w,contrast))
        if len(choices)!=1:continue
        _,delta,w,c=choices[0];rows.append([float(x),float(line_y(abc,x)+delta),float(w),float(c)])
    p=np.asarray(rows,float).reshape(-1,4)
    if len(p)<35 or np.ptp(p[:,0])<250:return dict(valid=False,reason='two_yellow_flanks_insufficient',points_xy=p[:,:2],count=len(p))
    # Stop at a large gap; no jumping to a different object on the surface.
    gaps=np.flatnonzero(np.diff(p[:,0])>28)
    if len(gaps):p=p[:gaps[0]+1]
    if len(p)<35 or np.ptp(p[:,0])<250:return dict(valid=False,reason='continuous_free_support_insufficient',points_xy=p[:,:2],count=len(p))
    xmax=float(p[-1,0]);local=p[p[:,0]>=xmax-500];keep=np.ones(len(local),bool)
    for _ in range(4):
        c=np.polyfit(local[keep,0]-D,local[keep,1],1)
        residual=local[:,1]-np.polyval(c,local[:,0]-D)
        keep=abs(residual)<max(2,3*1.4826*np.median(abs(residual)))
    local=local[keep];c=np.polyfit(local[:,0]-D,local[:,1],1)
    mad=float(max(.5,1.4826*np.median(abs(local[:,1]-np.polyval(c,local[:,0]-D)))))
    predictions=[]
    for span in [300,400,500,650]:
        q=p[p[:,0]>=xmax-span]
        if len(q)<25 or np.ptp(q[:,0])<200:continue
        cc=np.polyfit(q[:,0]-D,q[:,1],1);predictions.append(float(cc[1]))
    sensitivity=max(abs(np.asarray(predictions)-c[1])) if predictions else 20
    width=float(np.median(local[:,2]));spread=float(np.percentile(local[:,2],90)-np.percentile(local[:,2],10))
    # Sensitivity is not a probability or a calibrated confidence interval.
    uncertainty=float(max(3,2*mad,sensitivity)+.003*max(0,D-xmax))
    return dict(valid=bool(len(local)>=35 and D-xmax<950 and sensitivity<9),
                points_xy=local[:,:2],all_points_xy=p[:,:2],count=len(local),xmax=xmax,
                y_at_d=float(c[1]),slope=float(c[0]),abc=[float(-c[0]),1.,float(c[0]*D-c[1])],
                optical_width_vertical_px=width,width_spread_px=spread,
                residual_mad_px=mad,window_sensitivity_px=float(sensitivity),
                extrapolation_to_d_px=float(max(0,D-xmax)),sensitivity_bound_px=uncertainty,
                role='current_unambiguous_free_rope_with_reported_extrapolation')

def local_boundary(image,record):
    # BG1 is only a search proposal. Every local point is remeasured in pixels.
    curve=np.asarray(record['region'].get('top_curve_xy',[]),float).reshape(-1,2)
    if len(curve)<20 or curve[:,0].min()>D-160 or curve[:,0].max()<D+160:return dict(valid=False,reason='reference_search_domain_missing')
    xs=np.arange(int(D)-160,int(D)+161,8,dtype=np.float32)
    pred=np.interp(xs,curve[:,0],curve[:,1]);off=np.arange(-55,96,dtype=np.float32)
    mx=np.broadcast_to(xs[None,:],(len(off),len(xs))).copy();my=(pred[None,:]+off[:,None]).astype(np.float32)
    a=cv2.remap(image,mx,my,cv2.INTER_LINEAR).astype(np.float32)
    a=gaussian_filter(a,(1,1,0));b,g,r=cv2.split(a);light=(b+g+r)/3
    ch=((r+g)/2-b)/np.maximum(light,12);pts=[]
    for j,x in enumerate(xs):
        candidates=[]
        for k in range(20,len(off)-45):
            above=np.median(ch[k-18:k-5,j]);below=np.median(ch[k+5:k+30,j])
            persistent=np.mean(ch[k+6:k+40,j]<.38)
            if above>.42 and below<.33 and persistent>.78:
                candidates.append((float(above-below-.001*abs(off[k])),k))
        if not candidates:continue
        _,k=max(candidates);pts.append([float(x),float(pred[j]+off[k])])
    p=np.asarray(pts,float).reshape(-1,2)
    if len(p)<24:return dict(valid=False,reason='local_color_transition_insufficient',points_xy=p)
    keep=np.ones(len(p),bool)
    for _ in range(4):
        c=np.polyfit((p[keep,0]-D)/160,p[keep,1],2);e=p[:,1]-np.polyval(c,(p[:,0]-D)/160)
        mad=1.4826*np.median(abs(e[keep]-np.median(e[keep])))
        keep=abs(e)<max(4,2.5*mad)
    q=p[keep];bracket=len(q)>0 and q[:,0].min()<D-100 and q[:,0].max()>D+100
    return dict(valid=bool(len(q)>=24 and bracket and mad<5),points_xy=q,
                y_at_d=float(c[2]),polynomial=c,residual_mad_px=float(mad),
                sensitivity_bound_px=float(max(3,2*mad)),support_fraction=float(len(q)/len(xs)),
                role='visible_yellow_coil_transition_not_calibrated_inner_flange')

def measure_v2(image,f,record=None):
    import hashlib
    r=record or read(BG4/'full_003/frames'/f'{f:06d}.json')
    assert hashlib.sha256(image.tobytes()).hexdigest()==r['source_pixel_sha256']
    o=dict(frame=f,time_video_s=f/10,source_pixel_sha256=r['source_pixel_sha256'],contact_confirmed=False,
           s1_valid=r['s1']['valid'],camera_valid=bool(r['camera_motion'].get('valid',True)))
    if not o['s1_valid']:o['valid']=False;o['reason']='no_S1_current_axis';return o
    o['s1_abc']=r['s1']['current_line_abc'];a=local_axis(image,o['s1_abc']);b=local_boundary(image,r)
    o.update(rope=a,boundary=b,valid=bool(a['valid'] and b['valid']))
    if 'y_at_d' in a and 'y_at_d' in b:
        q=a['y_at_d']-b['y_at_d'];w=a['optical_width_vertical_px'];u=a['sensitivity_bound_px']+b['sensitivity_bound_px']+.25*a['width_spread_px']
        o.update(center_gap_px=q,edge_gap_proxy_px=q-w/2,sensitivity_bound_px=u,
                 edge_gap_proxy_width=(q-w/2)/w,center_gap_width=q/w)
    return plain(o)

def overlay(im,r):
    out=im.copy();a=r.get('rope',{});b=r.get('boundary',{})
    for p in a.get('points_xy',[]):cv2.circle(out,tuple(np.rint(p).astype(int)),3,(0,230,0),-1)
    if 'abc' in a:
        xs=np.arange(int(a['xmax']),int(D)+60,12)
        for i in range(1,len(xs),2):
            p=np.c_[xs[i-1:i+1],line_y(a['abc'],xs[i-1:i+1])]
            cv2.polylines(out,[np.rint(p).astype(np.int32)],False,(0,160,255),2,cv2.LINE_AA)
    for p in b.get('points_xy',[]):cv2.circle(out,tuple(np.rint(p).astype(int)),3,(255,200,0),-1)
    cv2.line(out,(int(D),2100),(int(D),2600),(255,0,255),2)
    return out

def main():
    from PIL import Image,ImageDraw
    out=ROOT/'v2';out.mkdir(exist_ok=True);cap=cv2.VideoCapture(str(SOURCE));rows=[]
    ids=[150,220,245,260,290,320];sheet=Image.new('RGB',(2000,len(ids)*290),'#151820');draw=ImageDraw.Draw(sheet)
    for j,f in enumerate(ids):
        cap.set(1,f);ok,im=cap.read();assert ok;r=measure_v2(im,f);rows.append(r)
        a=overlay(im,r)
        for k,bgr in enumerate([im,a]):
            pic=Image.fromarray(cv2.cvtColor(bgr[2070:2580,850:2850],cv2.COLOR_BGR2RGB)).resize((1000,255));sheet.paste(pic,(1000*k,j*290+30))
        brief={k:r.get(k) for k in ['frame','valid','center_gap_px','edge_gap_proxy_px','sensitivity_bound_px']}
        draw.text((8,j*290+8),str(brief),fill='white');print(brief,flush=True)
        print('axis', {k:v for k,v in r.get('rope',{}).items() if k not in ['points_xy','all_points_xy']},flush=True)
    cap.release();dump(out/'records.json',rows);sheet.save(out/'review.jpg',quality=96)
if __name__=='__main__':main()
