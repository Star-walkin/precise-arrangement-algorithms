"""Recover a missing local F3 class only from unowned current yellow points."""
import sys,numpy as np
from common20 import BASE
sys.path.insert(0,str(BASE))
import transfer_engine as e

def recover(curves,groups,data,d,edge,active):
    pitch=float(data['pitch']);points=data['points'];shape=e.local.restore_shape(groups['diagnostics']['shared_shape'])
    owned={int(k) for c in curves for k in c['point_ids']};logs=[];out=list(curves)
    current,_=e.local.local_crossings(out,points,data,d)
    lookup={c['boundary_id']:c for c in out}
    for upper,lower in zip(current[:-1],current[1:]):
        gap=lower['y']-upper['y']
        if not 1.65*pitch<gap<3.6*pitch or not edge-pitch<upper['y']<max(active,edge)+pitch:continue
        missing=int(round(gap/pitch))-1
        if missing not in (1,2):continue
        top=np.asarray(lookup[upper['boundary_id']]['xy']);bot=np.asarray(lookup[lower['boundary_id']]['xy'])
        lo=max(d-8*pitch,top[0,0],bot[0,0]);hi=min(d+8*pitch,top[-1,0],bot[-1,0])
        for k in range(1,missing+1):
            fraction=k/(missing+1);pred=(1-fraction)*np.interp(points[:,1],top[:,0],top[:,1])+fraction*np.interp(points[:,1],bot[:,0],bot[:,1])
            ids=np.array([j for j,p in enumerate(points) if lo<=p[1]<=hi and int(p[0]) not in owned and abs(p[2]-pred[j])<=.30*pitch],int)
            log=dict(gap_upper=upper['boundary_id'],gap_lower=lower['boundary_id'],candidate_fraction=fraction,raw_points=len(ids),accepted=False)
            if len(ids)<12 or np.ptp(points[ids,1])<4*pitch or min(np.sum(points[ids,1]<d),np.sum(points[ids,1]>d))<3:
                log['reason']='insufficient_bilateral_unowned_current_points';logs.append(log);continue
            row=e.local.Row(0,shape,pitch);row.fit(points,ids)
            ids=ids[abs(points[ids,2]-row.predict(points[ids,1]))<=.3*pitch]
            if len(ids)<12:log['reason']='insufficient_row_inliers';logs.append(log);continue
            row.fit(points,ids)
            if row.stats['residual_q90_pitch']>.22:log['reason']='row_residual_excessive';logs.append(log);continue
            xs=np.arange(np.ceil(points[ids,1].min()),np.floor(points[ids,1].max())+1);xy=np.c_[xs,row.predict(xs)]
            good=np.ones(len(xs),bool)
            for c in out:
                other=np.asarray(c['xy']);over=(xs>=other[0,0])&(xs<=other[-1,0])
                if not over.any():continue
                delta=xy[over,1]-np.interp(xs[over],other[:,0],other[:,1]);sign=1 if np.median(delta)>0 else -1
                good[over]&=sign*delta>=.45*pitch
            inside,unknown,support=e.local.supported_path(data,xy);near=abs(xs-d)<=2*pitch
            if not np.all(good) or not near.any() or np.mean(support[near])<.35 or np.any(unknown[abs(xs-d)<1.]) or not np.all(inside):
                log['reason']='local_support_envelope_or_order_failure';logs.append(log);continue
            bid=f'R20_gap_{len(logs):03d}';candidate=dict(boundary_id=bid,xy=xy,point_states=np.where(support,'observed','inferred'),
                point_ids=points[ids,0].astype(int).tolist(),fit_statistics=row.stats,model=row.model,support_fraction=float(np.mean(support)),
                recovered_local=True,recovery_source='current_unowned_F3_points_with_bilateral_support')
            accepted,rejected=e.local.local_crossings([candidate],points,data,d)
            if not accepted:log['reason']='current_intersection_gate';log['rejections']=rejected;logs.append(log);continue
            out.append(candidate);owned.update(candidate['point_ids']);log.update(accepted=True,boundary_id=bid,fit_points=len(ids),y=accepted[0]['y']);logs.append(log)
    return out,logs
