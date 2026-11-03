"""Spatial transition descriptors and image-only proposals; no references."""
import numpy as np,cv2,math
from common import load,BASE
OLD=None
def old():
    global OLD
    if OLD is None:
        import sys
        if str(BASE) not in sys.path:sys.path.append(str(BASE))
        from common import mod
        OLD=mod('_r1_readonly_profiles',BASE/'connect.py')
    return OLD
def unit(x):return np.asarray(x,float)/max(float(np.linalg.norm(x)),1e-8)
def sample(a,p):return old().sample(a,p)
def angles(x):return np.arctan2(x[...,1],x[...,0])
def wrap(x):return (x+np.pi)%(2*np.pi)-np.pi
P=dict(step=24.,max_steps=150,beam=8,angles=[-36,-24,-16,-8,-3,0,3,8,16,24,36],max_proposals=30,
    profile_positions=33,profile_tangent_offsets_px=[-6,0,6],profile_std_floor=.015,
    width_candidates_px=[12,16,20,24,32,40,48,64],support_bilateral_min=.18,
    max_weak_original_px=72.,max_bridge_original_px=48.,relation_weight=.30,
    response_weight=.45,bilateral_weight=.25,template_weight=.20,continuity_weight=.10,
    maximum_turn_degrees=42.,forward_cos_min=-.10,merge_xy=8.,merge_angle_degrees=12.,
    supported_heat_min=.22,supported_raw_min=.14,image_only_bilateral_min=.35,image_only_template_min=.35,
    final_candidates=3,selected_margin=.015,selected_min_new_supported_original_px=100.)

class Context:
    def __init__(self,gray,response,features,seed,tracks=None):
        self.gray=gray;self.heat=response;self.features=np.asarray(features,np.float32);self.seed=np.asarray(seed,float)
        self.d=unit(self.seed[-1]-self.seed[0]);self.a0=float(angles(self.d))
        self.width,self.template,self.width_meta=old().free_template(gray,seed,P)
        self.tracks=tracks or [];self.trackfield=None
        if self.tracks:self._trackfield()
    def _trackfield(self):
        h,w=self.heat.shape;m=np.ones((h,w),np.uint8)
        vals=np.zeros((h,w,7),np.float32)
        for serial,tr in enumerate(self.tracks):
            p=np.asarray(tr['points_xy'],float);t=np.gradient(p,axis=0);t/=np.maximum(np.linalg.norm(t,axis=1)[:,None],1e-6)
            observed=~np.asarray(tr['f3_pair_inferred'],bool);wid=np.asarray(tr['width_px'],float)
            for q,tt,oo,ww in zip(p,t,observed,wid):
                x,y=np.rint(q).astype(int)
                if 0<=x<w and 0<=y<h:m[y,x]=0;vals[y,x]=[q[0],q[1],tt[0],tt[1],float(oo),2*ww,serial+1]
        distance,labels=cv2.distanceTransformWithLabels(m,cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
        yy,xx=np.where(m==0)
        if not len(xx):return
        lookup=np.zeros((len(xx)+1,7),np.float32);lookup[1:]=vals[yy,xx]
        field=np.dstack([lookup[labels],distance*2]).astype(np.float32)
        # Contiguous per-channel planes avoid OpenCV copying a full2M-pixel
        # strided plane on every tiny query. Values and coordinate rules match.
        self.trackfield=np.ascontiguousarray(np.moveaxis(field,2,0))
    def track_values(self,p):
        p=np.asarray(p,float).reshape(-1,2);q=(p-[0,1050])*.5
        if self.trackfield is None:return np.zeros((len(p),8),np.float32)
        return np.column_stack([sample(self.trackfield[i],q) for i in range(8)])
    def heat_at(self,p):return sample(self.heat,(np.asarray(p)-[0,1050])*.5)
    def profiles(self,p,a,width=None):
        width=self.width if width is None else width
        return old().profiles(self.gray,p,a,width,P)
    def descriptor(self,history,candidates):
        """One row per edge; retain six ordered spatial feature positions."""
        hist=np.asarray(history,float);q=np.asarray(candidates,float).reshape(-1,2);n=len(q);end=hist[-1]
        d0=unit(hist[-1]-hist[max(0,len(hist)-3)]);delta=q-end;length=np.linalg.norm(delta,axis=1);direc=delta/np.maximum(length[:,None],1e-8)
        a=angles(direc);a0=float(angles(d0));normal=np.c_[-direc[:,1],direc[:,0]]
        pos=np.stack([np.broadcast_to(end-48*d0,(n,2)),np.broadcast_to(end-24*d0,(n,2)),np.broadcast_to(end,(n,2)),
            (end+q)*.5,q,q+24*direc],axis=1)
        pooled=(pos-[0,1050])*.125-.375
        feats=np.stack([sample(self.features[c],pooled) for c in range(8)],-1).reshape(n,-1)
        heat=self.heat_at(pos)
        where=np.stack([np.broadcast_to(end,(n,2)),q,q+24*direc],axis=1)
        prof,cov=self.profiles(where.reshape(-1,2),np.repeat(a,3));two,corr,z=old().evidence(prof,P,self.template)
        prof=prof.reshape(n,3,33);z=z.reshape(n,3,33);two=two.reshape(n,3);corr=corr.reshape(n,3);cov=cov.reshape(n,3)
        raw=np.c_[prof.mean(2),prof.std(2),two,corr,cov]
        bend=wrap(a-a0);seed_rel=(q-self.seed[0])/2000.
        prev=np.zeros((4,2),float);recent=hist[-4:];prev[-len(recent):]=(recent-end)/max(self.width,8)
        track=self.track_values(q);rel=(track[:,:2]*2+[0,1050]-q)/max(self.width,8)
        if self.trackfield is None:rel[:]=0
        trackvec=np.c_[rel.clip(-4,4),track[:,2:4],track[:,4],track[:,5]/64,track[:,7]/64,np.full(n,bool(self.trackfield is not None))]
        pool_center=heat[:,4];rank=(pool_center[:,None]>pool_center[None,:]).mean(1)
        geometry=np.c_[np.sin(bend),np.cos(bend),length/48.,delta@d0/48.,(d0[0]*delta[:,1]-d0[1]*delta[:,0])/48.,
            seed_rel,(q-self.seed[-1])/2000.,np.full(n,self.width/64.),rank,
            np.broadcast_to(prev.reshape(1,-1),(n,8)),np.broadcast_to(self.d,(n,2))]
        desc=np.c_[feats,heat,z.reshape(n,-1),raw,geometry,trackvec].astype(np.float32)
        assert np.isfinite(desc).all()
        supports=(heat[:,4]>=P['supported_heat_min'])&((two[:,1]>=P['supported_raw_min'])|(corr[:,1]>=.45))
        rawonly=(two[:,1]>=P['image_only_bilateral_min'])&(corr[:,1]>=P['image_only_template_min'])
        continuity=np.minimum(corr[:,1],corr[:,2]);quality=.45*heat[:,4]+.25*two[:,1]+.20*corr[:,1]+.10*continuity
        quality-=.04*abs(bend)/np.deg2rad(36)
        # High-quality lane disagreement is a bounded penalty, not an identity gate.
        if self.trackfield is not None:quality-=.025*track[:,4]*np.clip(track[:,7]/max(self.width,8),0,1)
        return desc,dict(heat=heat[:,4],bilateral=two[:,1],template=corr[:,1],coverage=cov[:,1],
            supported=supports|rawonly,raw_only=rawonly&~supports,quality=quality,turn=abs(bend),track_distance=track[:,7])
    def propose(self,history):
        p=np.asarray(history,float);end=p[-1];d=unit(p[-1]-p[max(0,len(p)-3)]);a=float(angles(d))
        aa=a+np.deg2rad(P['angles']);q=end+P['step']*np.c_[np.cos(aa),np.sin(aa)]
        out=[(v,'angle_grid') for v in q]
        center=end+P['step']*d;normal=np.array([-d[1],d[0]])
        shifts=np.linspace(-min(24.,self.width*.75),min(24.,self.width*.75),25)
        cand=center+shifts[:,None]*normal
        prof,cov=self.profiles(cand,np.full(len(cand),a));two,corr,_=old().evidence(prof,P,self.template)
        heat=self.heat_at(cand)
        # Distinct local maxima, not all image edges and not B heat thresholding.
        for signal,source in [(.65*two+.35*corr,'native_gray_bilateral_peak'),(heat,'A_response_peak')]:
            loc=np.flatnonzero((signal>=np.r_[-np.inf,signal[:-1]])&(signal>=np.r_[signal[1:],-np.inf]))
            loc=sorted(loc,key=lambda i:signal[i],reverse=True)[:3]
            for i in loc:
                if cov[i]>=.99 and (source=='A_response_peak' or two[i]>=.12):out.append((cand[i],source))
        tv=self.track_values(center[None])[0]
        if self.trackfield is not None and tv[7]<=max(20.,self.width):
            target=tv[:2]*2+[0,1050]
            if np.linalg.norm(target-end)>8:out.append((target,'F3_position_hint'))
        result=[];seen=set()
        for v,source in out:
            key=tuple(np.rint(v/2).astype(int));delta=v-end;ln=np.linalg.norm(delta)
            if key in seen or ln<12 or ln>48:continue
            da=abs(float(wrap(angles(delta)-a)))
            if da>np.deg2rad(P['maximum_turn_degrees']) or unit(delta)@self.d<P['forward_cos_min']:continue
            if not(2<v[0]<4093 and 1052<v[1]<2997):continue
            if len(p)>5 and np.linalg.norm(p[:-4]-v,axis=1).min()<12:continue
            seen.add(key);result.append((v,source))
            if len(result)>=P['max_proposals']:break
        return result
