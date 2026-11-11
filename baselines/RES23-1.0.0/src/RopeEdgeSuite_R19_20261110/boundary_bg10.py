"""BG10: convex spatial consensus for contaminated cold initialization.

Once a shape is initialized, BG9 current-frame anchor consensus is unchanged.
No future frames or per-video constants are used. Rejected startup candidates
remain recorded, and the accepted current hull bounds the displayed curve.
"""
import numpy as np
from boundary_bg9 import Boundary as BG9,design,robust_fit,foreground
from common import PITCH,plain

def convex_consensus(pts,w):
    X=design(pts[:,0],w);y=pts[:,1];rng=np.random.default_rng(2311);best=None
    bins=np.floor((pts[:,0]-pts[:,0].min())/(np.ptp(pts[:,0])+1)*8).astype(int)
    for _ in range(500):
        ids=rng.choice(len(pts),5,replace=False)
        if len(set(bins[ids]))<4 or np.linalg.cond(X[ids])>1e4:continue
        c=np.linalg.solve(X[ids],y[ids]);keep=abs(y-X@c)<.35*PITCH
        if keep.sum()<.65*len(pts) or np.ptp(pts[keep,0])<.7*np.ptp(pts[:,0]) or len(set(bins[keep]))<6:continue
        for j in range(2):
            c=np.linalg.lstsq(X[keep],y[keep],rcond=None)[0];keep=abs(y-X@c)<.35*PITCH
        if keep.sum()<.65*len(pts) or np.ptp(pts[keep,0])<.7*np.ptp(pts[:,0]) or len(set(bins[keep]))<6:continue
        u=design(np.linspace(pts[keep,0].min(),pts[keep,0].max(),80),w)[:,1]
        if np.min(2*c[2]+6*c[3]*u+12*c[4]*u*u)<=10:continue
        score=(int(keep.sum()),-float(np.quantile(abs(y[keep]-X[keep]@c),.9)))
        if best is None or score>best[0]:best=(score,c,keep)
    return best

class Boundary(BG9):
    def step(self,image,proposal,track,i,*,forced_exclude=None):
        selected=proposal;check=dict(attempted=False,used=False)
        if len(self.shapes)<3:
            raw=np.asarray(proposal.get('top_observed_xy',[])+proposal.get('rejected_top_observations_xy',[]),float).reshape(-1,2)
            if len(raw)>=60:
                raw=raw[np.argsort(raw[:,0])];ranges,_=foreground(image,proposal)
                excluded=np.array([any(lo<=x<=hi for lo,hi in ranges) for x in raw[:,0]])
                abc=track.get('current_line_abc') if track.get('valid') else None
                if abc is not None:excluded|=abs(raw@np.asarray(abc)[:2]+abc[2])<1.2*PITCH
                if forced_exclude:excluded|=np.array([any(lo<=x<=hi for lo,hi in forced_exclude) for x in raw[:,0]])
                pts=raw[~excluded]
                if len(pts)>=60 and np.ptp(pts[:,0])>=18*PITCH:
                    full,keep=robust_fit(design(pts[:,0],image.shape[1]),pts[:,1])
                    u=design(np.linspace(pts[:,0].min(),pts[:,0].max(),80),image.shape[1])[:,1]
                    curvature=2*full[2]+6*full[3]*u+12*full[4]*u*u
                    if np.min(curvature)<=10:
                        got=convex_consensus(pts,image.shape[1]);check['attempted']=True
                        if got is not None:
                            score,c,keep=got;group=pts[keep]
                            original_fraction=len(group)/len(raw);span=np.ptp(group[:,0])/max(1,np.ptp(raw[:,0]))
                            if original_fraction>=.65 and span>=.7:
                                selected=dict(proposal,top_observed_xy=group.tolist(),rejected_top_observations_xy=[])
                                check.update(used=True,inlier_fraction_of_all_candidates=original_fraction,
                                    span_fraction_of_all_candidates=span,selected_current_points_xy=group.tolist(),
                                    group_rejected_points_xy=pts[~keep].tolist(),raw_candidates_xy=raw.tolist(),
                                    residual_q90_px=-score[1],reason='nonconvex_global_initial_fit_replaced_by_broad_current_convex_group')
        out=super().step(image,selected,track,i,forced_exclude=forced_exclude)
        out['version']='BG10';out['convex_initialization']=plain(check)
        return out
