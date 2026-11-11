"""Current opposite-gradient edge-pair consensus in an identity-bound rope tube."""
import cv2,numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from common20 import plain,PITCH_PRIOR

def measure(image,track,d,pitch=PITCH_PRIOR):
    out=dict(valid=False,reasons=[],line_abc=None,points_xy=[],source='current_opposite_gradient_pairs_consensus',history_as_measurement=False)
    if not track.get('valid'):out['reasons'].append('upstream_identity_unavailable');return out
    a,b,c=track['line_abc'];m=-a/b;intercept=-c/b
    width=float((track.get('diagnostics',{}).get('seed_validation') or {}).get('width_px') or pitch)
    xs=np.arange(max(80.,d-1100),min(image.shape[1]-80.,d-40),8.)
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
                if not max(6.,.4*width)<=w<=min(42.,1.6*width) or g[lo]*g[hi]>=0:continue
                score=float(min(abs(g[lo]),abs(g[hi])))
                center=float((off[lo]+off[hi])/2)
                pairs.append([x,yy[j]+center,w,score,j,center])
    p=np.asarray(pairs,float).reshape(-1,6)
    if len(p)<40:out['reasons'].append('insufficient_edge_pairs');return out
    support=np.asarray(track.get('line_points_xy',[]),float).reshape(-1,2)
    anchor=support[(support[:,0]>=xs[0])&(support[:,0]<=xs[-1])] if len(support) else support
    anchor_x=float(np.max(anchor[:,0])) if len(anchor) else float(xs[0])
    anchor_y=m*anchor_x+intercept
    rng=np.random.default_rng(20261110);models=[(m,intercept)];early=p[p[:,0]<=xs[0]+300];late=p[p[:,0]>=xs[-1]-400]
    if len(early) and len(late):
        for _ in range(180):
            aa=early[rng.integers(len(early))];bb=late[rng.integers(len(late))]
            mm=(bb[1]-aa[1])/(bb[0]-aa[0]);cc=aa[1]-mm*aa[0]
            if abs(np.degrees(np.arctan(mm)-np.arctan(m)))>2.5:continue
            if abs(mm*anchor_x+cc-anchor_y)>(5. if len(anchor) else 28.):continue
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
    y=-(q[0]*d+q[2])/q[1]
    out.update(valid=not out['reasons'],line_abc=q,points_xy=pts[:,:2],y_at_d=float(y),width_px=float(np.median(pts[:,2])),
        sensitivity_px=float(unc),extrapolation_px=float(ext),residual_q90_px=err,support_fraction=fraction,max_gap_px=max_gap,width_relative_mad=width_cv,
        differs_from_upstream_y_px=float(y-(m*d+intercept)),anchor_x=anchor_x,anchor_on_current_supported_line=bool(len(anchor)),
        overlap_support_q90_px=overlap_error,physical_contact_confirmed=False)
    return plain(out)
