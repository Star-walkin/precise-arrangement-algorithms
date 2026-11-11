"""SG3: measurable line components and independent classical entrance proposals.

The supplied S1 sampler is unchanged. Gaps remain gaps in rendered measurements.
No cached line is returned as a current detection. d is not used for discovery.
"""
import sys, math
from pathlib import Path
import cv2,numpy as np
from common import R12,PITCH
sys.path.insert(0,str(R12))
from tracker_sg2 import Tracker as SG2Tracker, DynamicReacquire as P1Reacquire
from tracker_sg2 import S1Adapter, pixel_sha256

PARAMETERS=dict(minimum_span_px=400,minimum_points=60,minimum_inlier_fraction=.75,
    minimum_sample_coverage=.65,residual_widths=.22,fit_prefixes_px=[1400,1000,700,500],
    classical_entry_every_frames=5,maximum_entry_proposals=48)

def fit_line(pts):
    vx,vy,x,y=map(float,cv2.fitLine(np.asarray(pts,np.float32),cv2.DIST_HUBER,0,.01,.01).ravel())
    a=np.array([-vy,vx,vy*x-vx*y]);a/=np.linalg.norm(a[:2])
    return -a if a[1]<0 else a

class SegmentedS1(S1Adapter):
    def __init__(self):
        super().__init__(retain_step_records=True);self.current_width=24.

    def step(self,image,seed_xy,*,seed_source):
        gap=min(120.,4*self.current_width);self.config['max_unobserved_gap_px']=gap
        self.overrides={'max_unobserved_gap_px':gap}
        out=super().step(image,seed_xy,seed_source=seed_source)
        diag=out['diagnostics'];p=diag.get('s1_prediction',{});rows=p.pop('step_records',[])
        seg=dict(valid=False,reasons=[],current_supported_points_only=True,
            extrapolated_points_used_in_fit=False,bridges=[],parameters=PARAMETERS)
        diag['SG2_segments']=seg;diag['SG3_components']=seg
        if not out['valid']:
            seg['reasons']=['frozen_S1_insufficient_current_bilateral_extension']
            return out
        ids=np.array([i for i,v in enumerate(rows) if v['supported']],int)
        points=np.asarray(out['line_points_xy'],float)
        assert len(ids)==len(points)
        widths=np.array([rows[k]['measured_width_px'] for k in ids])
        distances=np.array([rows[k]['distance_after_E1_px'] for k in ids])
        width=float(p['width_px']);direction=np.diff(np.asarray(seed_xy,float),axis=0)[0]
        direction/=np.linalg.norm(direction);selected=None;trials=[]
        for limit in PARAMETERS['fit_prefixes_px']:
            eligible=(distances<=limit)&(widths>=.6*width)&(widths<=1.6*width)
            jj=np.flatnonzero(eligible)
            if len(jj)<60:continue
            abc=fit_line(points[jj]);cut=max(2.5,.22*width)
            keep=np.ones(len(jj),bool)
            for _ in range(3):
                keep=abs(points[jj]@abc[:2]+abc[2])<=cut
                if keep.sum()<60:break
                abc=fit_line(points[jj[keep]])
            kk=jj[keep]
            if len(kk)<60:continue
            pts=points[kk];span=float(np.ptp(pts@direction))
            coverage=len(kk)/max(1,ids[kk[-1]]-ids[kk[0]]+1)
            residual=np.abs(pts@abc[:2]+abc[2]);q90=float(np.quantile(residual,.9))
            delta=abs(math.degrees(math.atan2(-abc[0],abc[1])-math.atan2(direction[1],direction[0])))
            delta=min(delta,abs(180-delta))
            good=bool(span>=400 and coverage>=.65 and len(kk)/len(jj)>=.75 and
                q90<=cut and delta<=2. and ids[kk[0]]<=ids[jj[0]]+20)
            trials.append(dict(prefix_limit_px=limit,span_px=span,inlier_fraction=len(kk)/len(jj),
                sample_coverage=coverage,residual_q90_px=q90,angle_delta_deg=delta,accepted=good))
            if good:selected=(kk,abc,span,coverage,q90);break
        seg['prefix_trials']=trials
        if selected is None:
            out['valid']=False;seg['reasons']=['no_current_straight_component_with_sufficient_support'];return out
        kk,abc,span,coverage,q90=selected;pts=points[kk];sid=ids[kk]
        runs=np.split(np.arange(len(kk)),np.flatnonzero(np.diff(sid)>1)+1)
        segments=[pts[r].tolist() for r in runs if len(r)>=2]
        seg.update(valid=True,support_span_px=span,support_fraction=coverage,residual_q90_px=q90,
            observed_count=len(pts),width_px=width,discarded_points=len(points)-len(pts),
            independent_measured_runs=len(segments),polarity_changes_not_bridged=True)
        out.update(valid=True,line_abc=abc.tolist(),line_points_xy=pts.tolist(),path_xy=pts.tolist(),
            supported_segments_xy=segments,optical_endpoint_xy=pts[-1].tolist())
        diag['adapter_gate']='SG3_current_robust_measured_components'
        return out

def classical_entries(image,region):
    """Propose finite free-rope axes; S1 bilateral evidence and identity still gate them."""
    rec=region['record'];top=np.asarray(rec.get('top_curve_xy',[]),float).reshape(-1,2)
    if len(top)<30:return [],dict(reason='no_current_coil_extent')
    h,w=image.shape[:2];left=float(np.min(top[:,0]));right=float(np.max(top[:,0]));cx=(left+right)/2
    # A guide may hide the entrance. Search the exposed rope after the guide
    # as well; the actual surface mask, not a rectangular x cutoff, separates
    # free-space evidence from wound-rope texture.
    xstop=min(w,int(cx-120),int(left+1200));y0=max(0,int(np.min(top[:,1])-320))
    mask=region['surface_mask']>0
    distance=cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,3)
    gray=cv2.cvtColor(image[y0:,:xstop],cv2.COLOR_BGR2GRAY)
    small=cv2.resize(gray,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
    edges=cv2.Canny(cv2.GaussianBlur(small,(5,5),1.),35,100)
    lines=cv2.HoughLinesP(edges,1,np.pi/720,45,minLineLength=65,maxLineGap=14)
    candidates=[]
    for v in ([] if lines is None else np.asarray(lines).reshape(-1,4)):
        q=np.asarray(v,float).reshape(2,2)*2+[0,y0]
        q=q[np.argsort(q[:,0])];vec=q[1]-q[0];length=np.linalg.norm(vec)
        # Standardized left-entry geometry of this fixed camera pair. Negative
        # slopes here are chiefly flange arcs, not the incoming free rope.
        if vec[0]<120 or not .035<vec[1]/vec[0]<.65 or q[0,0]>=cx-200:continue
        slope=vec[1]/vec[0];at_entry=q[0,1]+slope*(left-q[0,0])
        edge=float(np.interp(left,top[:,0],top[:,1]))
        if not edge-180<=at_entry<h-10:continue
        vdir=vec/length;start=q[0]+max(8.,(55-q[0,0])/vdir[0])*vdir
        if start[0]>=cx-180 or start[1]<y0 or start[1]>h-45:continue
        sample=q[0]+np.linspace(0,1,48)[:,None]*vec
        ij=np.rint(sample).astype(int)
        free=distance[np.clip(ij[:,1],0,h-1),np.clip(ij[:,0],0,w-1)]>=12
        free_span=float(np.ptp(sample[free,0])) if free.sum()>1 else 0.
        seed=np.array([start,start+120*vdir]);abc=fit_line(q)
        if any((free_span>=60)==(c['free_span']>=60) and
            abs(float(abc[:2]@c['seed'][0]+abc[2]))<15 and abs(slope-c['slope'])<.025 for c in candidates):continue
        candidates.append(dict(seed=seed,slope=slope,length=float(length),abc=abc,free_span=free_span))
    anchors=[c for c in candidates if c['free_span']>=60]
    linked=[]
    for c in candidates:
        if c['free_span']>=60:
            c['entry_seed']=c['seed'];linked.append(c);continue
        # A downstream visible axis can be proposed across a guide, provided a
        # separate current free-entry segment corroborates it. S1 then samples
        # both seeds on this frame; the hidden guide interval supplies no points.
        eligible=[a for a in anchors if a['seed'][0,0]<c['seed'][0,0] and
            abs(float(c['abc'][:2]@a['seed'][0]+c['abc'][2]))<2.5*PITCH and
            abs(math.atan(c['slope'])-math.atan(a['slope']))<math.radians(4)]
        if eligible:
            a=min(eligible,key=lambda a:abs(float(c['abc'][:2]@a['seed'][0]+c['abc'][2])))
            c['entry_seed']=a['seed'];linked.append(c)
    candidates=linked
    candidates.sort(key=lambda q:-(q['length']+min(q['free_span'],400)))
    seeds=[]
    for c in candidates[:8]:
        for offset in (0.,160.,320.,480.,640.,800.):
            delta=np.array([1.,c['slope']]);delta/=np.linalg.norm(delta)
            q=c['seed']+offset*delta
            if q[0,0]>=cx-120:continue
            seeds.append(dict(seed_xy=q.tolist(),entry_seed=c['entry_seed'].tolist()))
    return seeds,dict(source='current_Canny_Hough_free_entry',raw_line_count=0 if lines is None else len(lines),
        geometric_candidates=len(candidates),candidate_count=len(seeds),search_box=[0,y0,xstop,h],
        candidates=[dict(seed_xy=c['seed'].tolist(),length_px=c['length'],free_span_px=c['free_span']) for c in candidates[:8]])

class Reacquire:
    def __init__(self):self.last=-100;self.calls=0;self.fallback=P1Reacquire(PITCH);self.source=None
    def __call__(self,image,t,i,region):
        if i-self.last<5 or region is None or not region['record']['valid']:return None
        self.last=i;self.calls+=1;seeds,diag=classical_entries(image,region)
        # Failed classical proposals must never starve the independent P1 path.
        if seeds and self.calls%2==1:
            self.source='SG3_current_Canny_Hough_candidate'
            return dict(seeds=seeds,source_pixel_sha256=pixel_sha256(image),model_forward_count=0,diagnostics=diag)
        self.source='SG3_current_BG5_P1_candidate'
        return self.fallback(image,t,i,region)

class Tracker(SG2Tracker):
    def __init__(self,mask):
        super().__init__(mask,Reacquire());self.s1=SegmentedS1()
    def set_region(self,region):
        r=region['record'];top=np.asarray(r.get('top_observed_xy',[]),float).reshape(-1,2)
        usable=bool(len(top)>=30 and np.ptp(top[:,0])>=600 and np.count_nonzero(region['surface_mask'])>=50000)
        self.count_region_valid=bool(r['valid'])
        search_region=dict(region,record=dict(r,valid=usable))
        super().set_region(search_region)
    def trial(self,image,seed,source,t):
        if source=='SG2_current_BG5_P1_candidate':source=self.reacquirer.source
        return super().trial(image,seed,source,t)
    def step(self,image,t,i):
        out=super().step(image,t,i);out['source']='RES-1.5/SG3'
        out['diagnostics']['count_region_valid']=self.count_region_valid
        out['diagnostics']['search_region_independent_of_count_column']=True
        return out
