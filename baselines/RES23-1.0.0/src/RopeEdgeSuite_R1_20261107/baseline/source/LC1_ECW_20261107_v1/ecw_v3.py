"""v3: associate current color-transition points before a short local curve fit.

Bounded paths select evidence; unsupported path sections are never measurements.
Retains v2's two-yellow-flank free-rope measurement without widening its gate.
"""
import hashlib
import cv2,numpy as np
from scipy.ndimage import gaussian_filter,uniform_filter1d
from ecw_v2 import local_axis
from local_measurement import ROOT,BG4,SOURCE,D,read,dump,plain

def local_boundary(image,record):
    curve=np.asarray(record['region'].get('top_curve_xy',[]),float).reshape(-1,2)
    if len(curve)<20 or curve[:,0].min()>D-160 or curve[:,0].max()<D+160:return dict(valid=False,reason='search_domain_missing')
    xs=np.arange(int(D)-160,int(D)+161,8,dtype=np.float32);pred=np.interp(xs,curve[:,0],curve[:,1]);off=np.arange(-55,76,dtype=np.float32)
    mx=np.broadcast_to(xs[None,:],(len(off),len(xs))).copy();my=(pred[None,:]+off[:,None]).astype(np.float32)
    a=cv2.remap(image,mx,my,cv2.INTER_LINEAR).astype(np.float32);a=gaussian_filter(a,(1.5,1,0))
    b,g,r=cv2.split(a);ch=((r+g)/2-b)/np.maximum((r+g+b)/3,12)
    mean=uniform_filter1d(ch,9,axis=0,mode='nearest')
    above=np.roll(mean,10,axis=0);below=np.roll(mean,-12,axis=0)
    persistence=np.roll(uniform_filter1d((ch<.40).astype(float),23,axis=0),-20,axis=0)
    grad=-np.gradient(ch,axis=0)
    eligible=(above>.40)&(below<.36)&(persistence>.73)&(grad>.004)
    eligible[:20]=False;eligible[-35:]=False
    score=np.clip((above-below),0,1)+np.clip(grad*4,0,.5)-.002*abs(off[:,None])
    score=np.where(eligible,score,-.30)
    # Link short local transitions. The path is a proposal until its own pixel
    # evidence and the final residual gate pass. No prior-frame edge is used.
    val=score[:,0].copy();back=np.zeros(score.shape,np.int16)
    for j in range(1,len(xs)):
        options=[]
        for step in range(-5,6):
            v=np.roll(val,step)-.035*abs(step)
            if step>0:v[:step]=-1e6
            elif step<0:v[step:]=-1e6
            options.append(v)
        options=np.stack(options);best=options.argmax(0)
        back[:,j]=np.arange(len(off))-(best-5);val=options.max(0)+score[:,j]
    k=np.zeros(len(xs),int);k[-1]=val.argmax()
    for j in range(len(xs)-1,0,-1):k[j-1]=back[k[j],j]
    okay=eligible[k,np.arange(len(xs))];p=np.c_[xs[okay],pred[okay]+off[k[okay]]]
    if len(p)<24:return dict(valid=False,reason='linked_transition_support_insufficient',points_xy=p)
    keep=np.ones(len(p),bool)
    for _ in range(4):
        c=np.polyfit((p[keep,0]-D)/160,p[keep,1],2);e=p[:,1]-np.polyval(c,(p[:,0]-D)/160)
        mad=1.4826*np.median(abs(e[keep]-np.median(e[keep])))
        keep=abs(e)<max(3.5,2.5*mad)
    q=p[keep];bracket=len(q)>0 and q[:,0].min()<D-100 and q[:,0].max()>D+100
    around=q[abs(q[:,0]-D)<55];local_both=len(around)>=7 and around[:,0].min()<D-16 and around[:,0].max()>D+16
    return dict(valid=bool(len(q)>=24 and bracket and local_both and mad<4),points_xy=q,
                y_at_d=float(c[2]),polynomial=c,residual_mad_px=float(mad),
                sensitivity_bound_px=float(max(3,2*mad)),support_fraction=float(len(q)/len(xs)),
                role='visible_yellow_coil_transition_not_calibrated_inner_flange',
                method='linked_current_chroma_inflection_then_robust_local_quadratic')

def measure_v3(image,f,record=None,rope=None):
    r=record or read(BG4/'full_003/frames'/f'{f:06d}.json')
    assert hashlib.sha256(image.tobytes()).hexdigest()==r['source_pixel_sha256']
    cam=r['camera_motion'];camera_good=bool(cam.get('confirmed') and not cam.get('camera_changed'))
    o=dict(frame=f,time_video_s=f/10,source_pixel_sha256=r['source_pixel_sha256'],physical_contact_confirmed=False,
           s1_valid=r['s1']['valid'],camera_valid=camera_good,projection_only=True)
    if not o['s1_valid']:o['valid']=False;o['reason']='no_S1_current_axis';return o
    o['s1_abc']=r['s1']['current_line_abc'];a=rope or local_axis(image,o['s1_abc']);b=local_boundary(image,r)
    o.update(rope=a,boundary=b,valid=bool(a['valid'] and b['valid'] and camera_good))
    if 'y_at_d' in a and 'y_at_d' in b:
        q=a['y_at_d']-b['y_at_d'];w=a['optical_width_vertical_px'];u=a['sensitivity_bound_px']+b['sensitivity_bound_px']
        o.update(center_gap_proxy_px=q,sensitivity_bound_px=u,center_gap_width=q/w)
        # Do not subtract half a rope width: this interface may contain the
        # active rope itself, so a physical surface-to-surface gap is unidentified.
    return plain(o)

def main():
    from PIL import Image,ImageDraw
    from ecw_v2 import overlay
    out=ROOT/'v3';out.mkdir(exist_ok=True);cap=cv2.VideoCapture(str(SOURCE));rows=[]
    ids=[150,220,245,260,290,320];sheet=Image.new('RGB',(2000,len(ids)*290),'#151820');draw=ImageDraw.Draw(sheet)
    for j,f in enumerate(ids):
        cap.set(1,f);ok,im=cap.read();assert ok;r=measure_v3(im,f);rows.append(r)
        a=overlay(im,r)
        for k,bgr in enumerate([im,a]):
            pic=Image.fromarray(cv2.cvtColor(bgr[2070:2580,850:2850],cv2.COLOR_BGR2RGB)).resize((1000,255));sheet.paste(pic,(1000*k,j*290+30))
        brief={k:r.get(k) for k in ['frame','valid','center_gap_proxy_px','sensitivity_bound_px']}
        draw.text((8,j*290+8),str(brief),fill='white');print(brief,flush=True)
    cap.release();dump(out/'records.json',rows);sheet.save(out/'review.jpg',quality=96)
if __name__=='__main__':main()
