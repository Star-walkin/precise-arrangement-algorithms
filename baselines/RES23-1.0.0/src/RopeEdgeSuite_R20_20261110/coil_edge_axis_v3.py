"""Current opposite-gradient edge-pair consensus in an identity-bound rope tube."""
import cv2,numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from common20 import plain,PITCH_PRIOR

def measure(image,track,d,pitch=PITCH_PRIOR,surface=None):
    out=dict(valid=False,reasons=[],line_abc=None,points_xy=[],source='current_opposite_gradient_pairs_consensus',history_as_measurement=False)
    if not track.get('valid'):out['reasons'].append('upstream_identity_unavailable');return out
    a,b,c=track['line_abc'];m=-a/b;intercept=-c/b
    width=float((track.get('diagnostics',{}).get('seed_validation') or {}).get('width_px') or pitch)
    xs=np.arange(max(80.,d-1700),min(image.shape[1]-80.,d-80),8.)
    off=np.arange(-60.,61.);yy=m*xs+intercept
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)
    profiles=[]
    for dx in [-3.,0.,3.]:
        mx=np.broadcast_to((xs+dx)[:,None],(len(xs),len(off))).astype(np.float32)
        my=(yy[:,None]+m*dx+off[None,:]).astype(np.float32)
        profiles.append(cv2.remap(gray,mx,my,cv2.INTER_LINEAR))
    intensity=np.median(profiles,axis=0);grad=np.gradient(gaussian_filter1d(intensity,1.,axis=1),axis=1)
    pairs=[]
    for j,(x,g) in enumerate(zip(xs,grad)):
        peaks=find_peaks(abs(g),height=max(2.5,np.percentile(abs(g),60)),distance=3)[0]
        for lo in peaks:
            for hi in peaks:
                w=hi-lo
                if not max(8.,.6*width)<=w<=min(40.,1.5*width) or g[lo]*g[hi]>=0:continue
                score=float(min(abs(g[lo]),abs(g[hi])))
                center=float((off[lo]+off[hi])/2)
                y=yy[j]+center
                outside=bool(surface is not None and 0<=round(y)<surface.shape[0] and surface[round(y),round(x)]==0)
                # The exposed incoming rope is a bright body on its dark
                # background. Dark shadow pairs on the coil cannot anchor it.
                external_witness=outside and g[lo]>0 and g[hi]<0 and intensity[j,int((lo+hi)//2)]>60
                pairs.append([x,y,w,score,j,center,float(external_witness)])
    p=np.asarray(pairs,float).reshape(-1,7)
    if len(p)<40:out['reasons'].append('insufficient_edge_pairs');return out
    support=np.asarray(track.get('line_points_xy',[]),float).reshape(-1,2)
    anchor=support[(support[:,0]>=xs[0])&(support[:,0]<=xs[-1])] if len(support) else support
    # Evidence behind a guide may have a different axis. Only overlapping
    # support near/on the current coil constrains the coil-side fit.
    if surface is not None and len(anchor):
        dist=cv2.distanceTransform((surface==0).astype(np.uint8),cv2.DIST_L2,3)
        anchor=anchor[np.array([dist[int(np.clip(round(y),0,dist.shape[0]-1)),int(np.clip(round(x),0,dist.shape[1]-1))]<120 for x,y in anchor])]
    anchor_x=float(np.max(anchor[:,0])) if len(anchor) else float(xs[0])
    anchor_y=m*anchor_x+intercept
    rng=np.random.default_rng(20261110);models=[(m,intercept)];early=p[(p[:,0]<=xs[0]+700)&(p[:,6]>0)];late=p[p[:,0]>=xs[-1]-500]
    if len(early)<10:early=p[p[:,0]<=xs[0]+500]
    if len(early) and len(late):
        for _ in range(300):
            aa=early[rng.integers(len(early))];bb=late[rng.integers(len(late))]
            if bb[0]==aa[0]:continue
            mm=(bb[1]-aa[1])/(bb[0]-aa[0]);cc=aa[1]-mm*aa[0]
            if abs(np.degrees(np.arctan(mm)-np.arctan(m)))>2.5:continue
            if len(anchor)>=12 and abs(mm*anchor_x+cc-anchor_y)>6.:continue
            models.append((mm,cc))
    best=None
    for mm,cc in models:
        # One endpoint alone permits a wrong groove to pivot through the
        # anchor. The whole overlapping current SG5 evidence must agree.
        if len(anchor)>=12 and np.ptp(anchor[:,0])>=100:
            if np.percentile(abs(anchor[:,1]-mm*anchor[:,0]-cc),90)>6.:continue
        err=abs(p[:,1]-mm*p[:,0]-cc);keep=np.flatnonzero(err<3.)
        if len(keep)<30:continue
        selected=[]
        for j in np.unique(p[keep,4]):
            q=keep[p[keep,4]==j];rank=np.log1p(p[q,3])-.3*err[q]-.5*abs(np.log(p[q,2]/width))
            selected.append(q[int(rank.argmax())])
        pts=p[selected]
        if len(pts)<35 or np.ptp(pts[:,0])<450:continue
        witness=pts[pts[:,6]>0]
        overlap_ok=len(anchor)>=12 and np.ptp(anchor[:,0])>=100
        if not overlap_ok and (len(witness)<15 or np.ptp(witness[:,0])<100):continue
        rank=len(pts)+.001*np.ptp(pts[:,0])+.2*np.sum(np.log1p(pts[:,3]))-.1*abs(mm*anchor_x+cc-anchor_y)
        if best is None or rank>best[0]:best=(rank,pts)
    if best is None:out['reasons'].append('no_current_edge_pair_consensus');return out
    pts=best[1];v=cv2.fitLine(pts[:,:2].astype(np.float32),cv2.DIST_HUBER,0,.01,.01).ravel();q=np.array([-v[1],v[0],v[1]*v[2]-v[0]*v[3]])
    err=float(np.percentile(abs(pts[:,:2]@q[:2]+q[2]),90));ext=max(0.,d-pts[:,0].max());span=np.ptp(pts[:,0]);unc=max(2.5,err)*(1+ext/span)
    fraction=len(pts)/len(xs);max_gap=float(np.diff(pts[:,0]).max(initial=0));width_cv=float(np.median(abs(pts[:,2]-np.median(pts[:,2])))/np.median(pts[:,2]))
    if fraction<.45 or max_gap>250:out['reasons'].append('edge_pair_coverage_insufficient')
    if width_cv>.3 or err>3.5 or unc>.3*pitch:out['reasons'].append('edge_pair_fit_inconsistent')
    overlap_error=float(np.percentile(abs(anchor@q[:2]+q[2]),90)) if len(anchor)>=12 else None
    if overlap_error is not None and overlap_error>6.:out['reasons'].append('disagrees_with_current_SG5_overlap')
    witness=pts[pts[:,6]>0]
    out['external_current_bright_rope_witnesses']=len(witness)
    out['external_witness_span_px']=float(np.ptp(witness[:,0])) if len(witness) else 0.
    y=-(q[0]*d+q[2])/q[1]
    out.update(valid=not out['reasons'],line_abc=q,points_xy=pts[:,:2],y_at_d=float(y),width_px=float(np.median(pts[:,2])),
        sensitivity_px=float(unc),extrapolation_px=float(ext),residual_q90_px=err,support_fraction=fraction,max_gap_px=max_gap,width_relative_mad=width_cv,
        differs_from_upstream_y_px=float(y-(m*d+intercept)),anchor_x=anchor_x,anchor_on_current_supported_line=bool(len(anchor)),
        overlap_support_q90_px=overlap_error,physical_contact_confirmed=False)
    return plain(out)
