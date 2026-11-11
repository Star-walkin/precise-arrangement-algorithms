"""Conservative video-only distance smoothing and distinct contact evidence."""
import numpy as np

class DistanceFilter:
    def __init__(self):self.history=[];self.epoch=None;self.last=None;self.bad_run=0
    def step(self,t,epoch,measurement):
        if epoch!=self.epoch:self.history=[];self.last=None;self.bad_run=0;self.epoch=epoch
        out=dict(available=False,N=None,interval=None,source='unavailable',outlier=False,physical_contact_confirmed=False)
        good=bool(measurement.get('available') and measurement.get('N') is not None)
        if not good:
            if self.last and t-self.last['time']<=.5 and measurement.get('d') is not None:
                dt=t-self.last['time'];value=self.last['N']+self.last['velocity']*dt;unc=self.last['uncertainty']+.5*dt
                out.update(available=True,N=float(value),interval=[float(value-unc),float(value+unc)],source='short_prediction_max_0_5s',age_s=dt)
            return out
        n=float(measurement['N']);low,high=measurement['N_interval'];unc=max(.2,(high-low)/2)
        self.history=[x for x in self.history if t-x[0]<=3.]
        if self.last and t-self.last['time']<=.5:
            pred=self.last['N']+self.last['velocity']*(t-self.last['time']);threshold=max(.6,unc,self.last['uncertainty'])
            if abs(n-pred)>threshold:
                self.bad_run+=1
                if self.bad_run<3:
                    out.update(available=True,N=float(np.clip(pred,low,high)),interval=[low,high],source='current_interval_with_temporal_outlier',outlier=True)
                    return out
                self.history=[]
            else:self.bad_run=0
        self.history.append((t,n,unc));velocity=0.;value=n;fit_residual=None
        if len(self.history)>=6 and self.history[-1][0]-self.history[0][0]>=.5:
            points=np.array(self.history);tt=points[:,0]-t;X=np.c_[np.ones(len(tt)),tt];w=1/np.maximum(points[:,2],.25)**2
            weights=w.copy()
            for _ in range(5):
                beta=np.linalg.lstsq(X*np.sqrt(weights[:,None]),points[:,1]*np.sqrt(weights),rcond=None)[0]
                err=points[:,1]-X@beta;scale=max(.15,1.4826*np.median(abs(err-np.median(err))))
                weights=w*np.minimum(1.,1.5*scale/np.maximum(abs(err),1e-8))
            velocity=float(beta[1]);fit_residual=float(np.percentile(abs(err),90))
            if abs(velocity)<=3.:value=float(np.clip(beta[0],low,high))
            else:velocity=0.;value=n
            unc=max(unc,fit_residual)
        self.last=dict(time=t,N=value,velocity=velocity,uncertainty=unc)
        out.update(available=True,N=value,interval=[min(low,value-unc),max(high,value+unc)],source='current_measurement_robust_temporal_fit',
            velocity_pitch_per_s=velocity,current_model_uncertainty_not_averaged_away=True,fit_residual_q90=fit_residual)
        return out

def contact_evidence(measurement,axis):
    out=dict(state='unavailable',physical_contact_confirmed=False,physical_contact_ground_truth_available=False,uses_prediction=False,reasons=[])
    if not measurement.get('available'):
        out['reasons']=measurement.get('reasons',[]);return out
    n=measurement['N'];interval=measurement['N_interval'];gap=measurement.get('contact_gap_interval');c=measurement['central'];b=c['boundary']
    out.update(center_pitch_distance=n,projected_contact_gap_interval=gap,boundary_locally_observed=b['locally_observed'],
        boundary_occluded=b['occluded'],current_coil_side_axis=bool(axis and axis['current_coil_side_measured']))
    if interval[0]>1.25:out['state']='separated';return out
    if n<-1.25 or interval[1]<0:out['state']='projection_conflict';out['reasons'].append('projection_crossing_is_not_physical_contact');return out
    if n>1.25:out['state']='near_edge_approaching';return out
    if not b['locally_observed'] or interval[1]-interval[0]>2.5:
        out['state']='near_edge_uncertain';out['reasons'].append('local_boundary_hidden_or_model_range_too_wide');return out
    if axis and axis['current_coil_side_measured'] and gap is not None and gap[0]<=0<=gap[1] and max(abs(gap[0]),abs(gap[1]))<=1.25:
        out['state']='visual_contact_candidate';out['reasons'].append('monocular_optical_evidence_not_material_contact_confirmation')
    else:out['state']='near_edge_uncertain';out['reasons'].append('current_endpoint_evidence_incomplete')
    return out

def windows(rows,fps):
    result=[];start=None;last=None;kind=None
    for i,row in enumerate(rows+[dict(contact=dict(state='end'))]):
        state=row['contact']['state'];target=state if state in ('visual_contact_candidate','near_edge_uncertain','projection_conflict') else None
        if target!=kind:
            if kind is not None and last-start+1>=3:
                result.append(dict(state=kind,start_frame=start,end_frame=last,start_s=start/fps,end_s=(last+1)/fps,
                    current_evidence_frames=last-start+1,physical_contact_confirmed=False,bridged_missing_frames=0))
            start=i if target else None;kind=target
        if target:last=i
    return result
