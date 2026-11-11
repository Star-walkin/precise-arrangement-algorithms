"""Independent, current-frame classical candidate detector (LC1-T).

This module loads no P1/H6/S1 weights, paths, historical detections or labels.
The supplied static surface mask is a search-domain prior, not a rope path.
All coordinates returned by step are in the input image coordinate system.
"""
from __future__ import annotations
import math
import cv2
import numpy as np

SOURCE = 'LC1-T traditional candidate, not P1/H6/S1'


def _sample(gray, x, y):
    x=np.asarray(x,np.float32); y=np.asarray(y,np.float32)
    return cv2.remap(gray,x.reshape(1,-1),y.reshape(1,-1),cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT101).reshape(x.shape)


def _profile(gray,x,y,m,pitch):
    """Measured bright-bar contrast on both sides of the local normal."""
    norm=math.sqrt(1+m*m); nx=-m/norm; ny=1/norm
    r=max(5.,pitch*.80)
    c=_sample(gray,x,y)
    lo=_sample(gray,np.asarray(x)-r*nx,np.asarray(y)-r*ny)
    hi=_sample(gray,np.asarray(x)+r*nx,np.asarray(y)+r*ny)
    contrast=c-.5*(lo+hi)
    both=np.minimum(c-lo,c-hi)
    return contrast,both,c


def _fit_yx(points):
    # Robust IRLS fit using current profile-supported points only.
    x=points[:,0]; y=points[:,1]
    a=np.column_stack((x,np.ones_like(x)))
    mb=np.linalg.lstsq(a,y,rcond=None)[0]
    for _ in range(6):
        e=y-a@mb
        scale=max(1.,1.4826*np.median(np.abs(e-np.median(e))))
        w=np.minimum(1.,(2.5*scale)/(np.abs(e)+1e-6))
        mb=np.linalg.lstsq(a*w[:,None],y*w,rcond=None)[0]
    e=y-a@mb
    return float(mb[0]),float(mb[1]),np.abs(e)


class TraditionalFrontend:
    def __init__(self,surface_mask_original,pitch_px=20,free_side='right'):
        mask=np.asarray(surface_mask_original)
        if mask.ndim==3: mask=mask[...,0]
        if mask.ndim!=2 or not np.any(mask):
            raise ValueError('surface_mask_original must be a nonempty 2D mask')
        if free_side not in ('right','left'):
            raise ValueError("free_side must be 'right' or 'left'")
        self.original_mask=mask>0
        self.mask=np.fliplr(self.original_mask).copy() if free_side=='left' else self.original_mask.copy()
        self.pitch=float(pitch_px)
        self.free_side=free_side
        yy,xx=np.where(self.mask)
        self.box=(int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max()))

    def step(self,bgr,t):
        if bgr.shape[:2]!=self.mask.shape:
            raise ValueError('frame and supplied static surface mask have different shapes')
        im=cv2.flip(bgr,1) if self.free_side=='left' else bgr
        h,w=im.shape[:2]
        gray=cv2.GaussianBlur(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32),(0,0),1.25)
        candidates=self._free_candidates(gray)
        diag={'time_s':float(t),'current_frame_only':True,'static_mask_role':'search domain only',
              'free_line_method':'Canny + probabilistic Hough + bilateral normal-profile refinement + robust IRLS fit',
              'curve_method':'second-order dynamic programming on current bright-ridge profiles; not semantic segmentation',
              'pitch_px':self.pitch,'surface_bbox_xyxy':list(self.box),
              'line_candidate_count':len(candidates),'line_candidates':[{k:v for k,v in c.items() if k!='points'} for c in candidates[:8]],
              'limitations':['No learned semantic rope identity.', 'Coil texture and specular ridges can imitate the entering rope.',
                            'No historical path is used to replace missing current-frame evidence.']}
        result={'line_abc':None,'line_points_xy':[],'path_xy':[],
                'whole_path_candidate_supported':False,'diagnostics':diag,'source':SOURCE}
        if not candidates:
            diag['status']='no_current_frame_free_line'
            return result
        best=candidates[0]
        m,b=best['m'],best['b']
        line_pts=best['points']
        diag['free_line_supported']=best['free_supported_fraction']>=.65 and best['free_supported_span_px']>=max(220,w*.07)
        # Compare spatially distinct candidate lines at the surface middle x.
        distinct=[c for c in candidates[1:] if abs((c['m']-m)*(.5*(self.box[0]+self.box[2]))+c['b']-b)>self.pitch*.65]
        diag['free_line_candidate_disagreement']=bool(distinct and distinct[0]['score']>.75*best['score'])
        trace,td=self._trace(gray,m,b,line_pts)
        diag.update(td)
        # A path is only a supported *candidate*, and must survive all evidence gates.
        whole=(diag['free_line_supported'] and not diag['free_line_candidate_disagreement']
               and td.get('trace_supported_fraction',0)>=.85
               and td.get('trace_inside_span_px',0)>=self.pitch*6
               and td.get('trace_max_unsupported_run',999)<=2
               and td.get('trace_bend_slope_change',0)>=.075
               and not td.get('trace_candidate_disagreement',True))
        diag['status']='supported_whole_path_candidate' if whole else 'free_line_with_unconfirmed_incoil_candidate'
        point_support=diag.get('path_point_support',[True]*len(trace))
        # A fitted straight line may use separated measured fragments. A gap between
        # their points is explicitly unsupported; renderers must not connect it solid.
        distances=np.linalg.norm(np.diff(trace,axis=0),axis=1) if len(trace)>1 else np.array([])
        segment_support=[bool(point_support[k] and point_support[k+1] and distances[k]<=max(25.,self.pitch*1.4)) for k in range(len(distances))]
        diag['path_segment_support']=segment_support
        diag['path_unsupported_geometric_gap_count']=int(np.sum(distances>max(25.,self.pitch*1.4)))
        diag['path_rendering_rule']='Draw only supported segments solid; unsupported samples/gaps must be dashed or omitted.'
        if diag['path_unsupported_geometric_gap_count']:
            whole=False
            diag['status']='free_line_with_unconfirmed_incoil_candidate'
        # Coordinates are input coordinates; optional left-entry conversion stays internal.
        abc=np.array([m,-1.,b],float); abc/=np.linalg.norm(abc[:2])
        if self.free_side=='left':
            abc=np.array([-abc[0],abc[1],abc[2]+abc[0]*(w-1)])
            line_pts=line_pts.copy(); line_pts[:,0]=w-1-line_pts[:,0]
            trace=trace.copy(); trace[:,0]=w-1-trace[:,0]
        result.update(line_abc=abc.tolist(),line_points_xy=line_pts.tolist(),path_xy=trace.tolist(),whole_path_candidate_supported=bool(whole))
        return result

    def _free_candidates(self,gray):
        h,w=gray.shape; xmin,ymin,xmax,ymax=self.box
        x0=max(0,int(xmin+.18*(xmax-xmin)))
        y0=max(0,int(ymin-max(420,.22*h)))
        y1=min(h,int(ymin+max(550,.28*h)))
        scale=min(1.,1800./w)
        crop=gray[y0:y1,x0:]
        small=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA).astype(np.uint8)
        edge=cv2.Canny(small,35,100,L2gradient=True)
        lines=cv2.HoughLinesP(edge,1,np.pi/720,threshold=45,
                              minLineLength=max(65,int(w*scale*.055)),maxLineGap=max(12,int(self.pitch*scale)))
        if lines is None:return []
        raw=[]
        # OpenCV 5 returns (N,4); OpenCV 4 commonly returns (N,1,4).
        for v in lines.reshape(-1,4):
            xa,ya,xb,yb=v.astype(float)/scale+np.array([x0,y0,x0,y0])
            if xa>xb:xa,xb,ya,yb=xb,xa,yb,ya
            if xb-xa<150:continue
            m=(yb-ya)/(xb-xa)
            if not (-.68<m<-.025):continue
            b=ya-m*xa
            # A free-line extension has to intersect the supplied search mask.
            xx=np.arange(xmin,min(w,xmax+1),10,dtype=float); yy=m*xx+b
            good=(yy>=0)&(yy<h)
            hit=np.zeros(len(xx),bool)
            hit[good]=self.mask[yy[good].astype(int),xx[good].astype(int)]
            if hit.sum()<4:continue
            xhit=float(xx[hit].max())
            xs=np.arange(max(xa,xhit+4),min(w-1,xb)+1,12.,dtype=float)
            if len(xs)<15:continue
            ys=m*xs+b
            offsets=np.arange(-self.pitch*.70,self.pitch*.70+1,1.5)
            norm=math.sqrt(1+m*m); nx=-m/norm; ny=1/norm
            cx=xs[:,None]+offsets[None,:]*nx
            cy=ys[:,None]+offsets[None,:]*ny
            con,both,bright=_profile(gray,cx,cy,m,self.pitch)
            score=con+.25*both-.08*np.abs(offsets[None,:])
            j=np.argmax(score,axis=1)
            sel=(np.arange(len(xs)),j)
            pts=np.column_stack((cx[sel],cy[sel]))
            vals=con[sel]; bothv=both[sel]
            supported=(vals>=9)&(bothv>=1)&(bright[sel]>=75)
            outside=~self.mask[np.clip(pts[:,1],0,h-1).astype(int),np.clip(pts[:,0],0,w-1).astype(int)]
            supported &= outside
            if supported.sum()<14:continue
            pp=pts[supported]
            mm,bb,res=_fit_yx(pp)
            keep=res<max(4,self.pitch*.23)
            pp=pp[keep]
            if len(pp)<12:continue
            span=float(np.ptp(pp[:,0])); frac=float(supported.mean())
            if span<max(180,w*.05) or frac<.45:continue
            # Prefer long, bilateral supported bright bars outside the surface.
            q=span*frac*min(2.,max(.1,float(np.median(vals[supported]))/30.))
            raw.append({'m':mm,'b':bb,'points':pp,'score':float(q),'free_supported_fraction':frac,
                        'free_supported_span_px':span,'median_bilateral_contrast':float(np.median(vals[supported])),
                        'median_fit_residual_px':float(np.median(res)),'mask_entry_x':xhit})
        raw.sort(key=lambda c:c['score'],reverse=True)
        # Retain one representative per line; do not count the two rope edges twice.
        result=[]; xm=.5*(xmin+xmax)
        for c in raw:
            if any(abs((c['m']-z['m'])*xm+c['b']-z['b'])<self.pitch*.45 and abs(c['m']-z['m'])<.035 for z in result):continue
            result.append(c)
            if len(result)>=10:break
        return result

    def _trace(self,gray,m,b,line_pts):
        h,w=gray.shape; xmin,ymin,xmax,ymax=self.box; p=self.pitch
        xx=np.arange(xmin,xmax+1,5.,dtype=float); yy=m*xx+b
        valid=(yy>=0)&(yy<h)
        inside=np.zeros(len(xx),bool)
        inside[valid]=self.mask[yy[valid].astype(int),xx[valid].astype(int)]
        if not inside.any():return line_pts,{'trace_status':'line_does_not_enter_mask'}
        entry=float(xx[inside].max())
        start=min(w-4,float(line_pts[:,0].min()),entry+p*2)
        # Start is on measured current free-line data, not an invented contact point.
        x=np.arange(start,max(xmin+10,start-min(1500.,(xmax-xmin)*.82)),-8.,dtype=float)
        if len(x)<8:return line_pts,{'trace_status':'insufficient_tracking_span'}
        # q offsets around current free-line extrapolation; that extrapolation is only
        # a search coordinate, not itself returned as detected support.
        half=min(120.,max(p*3.5,60.)); q=np.arange(-half,half+1,2.)
        y=m*x[:,None]+b+q[None,:]
        xs=np.broadcast_to(x[:,None],y.shape)
        con,both,bright=_profile(gray,xs,y,m,p)
        supported=(con>=7)&(both>=-1)&(bright>=60)&(y>=2)&(y<h-2)
        clipped_y=np.clip(y,0,h-1).astype(int)
        mask_here=self.mask[clipped_y,np.clip(xs,0,w-1).astype(int)]
        # Directional profile evidence, plus a cost for missing bilateral support.
        data=2.5-np.clip(con/18.,-1,4)+1.5*(~supported)
        data+=.0008*q[None,:]**2
        data+=3.*((y<2)|(y>=h-2))
        # Second-order DP: q state and q-step state. Changes in slope are penalized.
        ds=np.arange(-2,3); nq=len(q); nd=len(ds); nt=len(x)
        inf=1e9
        dp=np.full((nq,nd),inf,np.float32)
        i0=int(np.argmin(np.abs(q)))
        dp[i0,:]=data[0,i0]+np.abs(ds)*.3
        parents=np.full((nt,nq,nd),-1,np.int16)
        for k in range(1,nt):
            nxt=np.full_like(dp,inf)
            for j,d in enumerate(ds):
                # current q = previous q + 2*d
                dst=np.arange(max(0,d),min(nq,nq+d))
                prev=dst-d
                trans=dp[prev,:]+.7*(ds[None,:]-d)**2+.04*d*d
                arg=np.argmin(trans,axis=1)
                nxt[dst,j]=trans[np.arange(len(dst)),arg]+data[k,dst]
                parents[k,dst,j]=arg
            dp=nxt
        ends=np.min(dp,axis=1)
        best_i=int(np.argmin(ends)); best_j=int(np.argmin(dp[best_i]))
        distant=np.abs(q-q[best_i])>p*.65
        alt=float(np.min(ends[distant])) if distant.any() else inf
        margin=float((alt-ends[best_i])/nt)
        qi=[]; i,j=best_i,best_j
        for k in range(nt-1,-1,-1):
            qi.append(i)
            if k:
                oldj=int(parents[k,i,j]); i-=int(ds[j]); j=oldj
        qi=np.asarray(qi[::-1],int)
        path=np.column_stack((x,y[np.arange(nt),qi]))
        sup=supported[np.arange(nt),qi]
        onmask=mask_here[np.arange(nt),qi]
        # Never return a long sequence sustained only by smoothness/search priors.
        run=0; maxrun=0; stop=nt
        for k,s in enumerate(sup):
            run=0 if s else run+1;maxrun=max(maxrun,run)
            if run>=4:
                stop=max(0,k-run+1);break
        path=path[:stop];sup=sup[:stop];onmask=onmask[:stop]
        pts=line_pts[line_pts[:,0]>start]
        pts=pts[np.argsort(-pts[:,0])]
        full=np.vstack((pts,path)) if len(path) else pts
        slopes=np.diff(path[:,1])/np.diff(path[:,0]) if len(path)>1 else np.array([])
        # The per-step median is quantized by the DP grid; fit a finite window to
        # measure accumulated bending rather than requiring a one-step jump.
        nfit=min(25,len(path))
        tail_m=float(np.polyfit(path[-nfit:,0],path[-nfit:,1],1)[0]) if nfit>=5 else m
        change=float(abs(tail_m-m))
        span=float(np.ptp(path[onmask,0])) if onmask.sum()>1 else 0.
        td={'trace_status':'dp_candidate_current_pixel_profiles','trace_sample_step_x_px':8,
            'trace_attempted_points':nt,'trace_returned_points':len(path),'trace_truncated_for_missing_current_support':stop<nt,
            'trace_supported_fraction':float(sup.mean()) if len(sup) else 0.,'trace_inside_span_px':span,
            'trace_max_unsupported_run':int(maxrun),'trace_bend_slope_change':change,
            'trace_distinct_endpoint_cost_margin_per_step':margin,'trace_candidate_disagreement':margin<.12,
            'path_point_support':[True]*len(pts)+sup.tolist(),
            'path_support_semantics':'True means current bright-center bilateral-profile contrast passed; it does not establish semantic rope identity.',
            'path_candidate_warning':'DP may follow a wound-rope/specular ridge when the entering rope becomes indistinguishable; complete-chain gate rejects ambiguous candidates.'}
        return full,td
