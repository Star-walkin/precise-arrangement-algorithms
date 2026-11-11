"""Current tangency measurements are distinct from a certified fixed column.

The >=5-rope calibration rule is unchanged. A short sequence of current
geometry can support a moving-column estimate without certifying fixed d.
No elapsed-video timestamp or video identity participates in these gates.
"""
import copy
import numpy as np
from count20 import measure

class DynamicGate:
    def __init__(self):self.history=[];self.epoch=None
    def step(self,t,epoch,q,axis,tangent):
        if epoch!=self.epoch:self.history=[];self.epoch=epoch
        self.history=[x for x in self.history if t-x[0]<=.7+1e-6]
        reasons=[];c=q.get('central') or {};b=c.get('boundary') or {}
        if not q.get('available'):reasons.append('current_dynamic_measurement_unavailable')
        if not axis or not axis.get('current_coil_side_measured'):reasons.append('current_coil_side_axis_required')
        if not tangent.get('valid'):reasons.append('current_F3_tangency_required')
        if b.get('occluded') or not b.get('inside_current_anchor_hull'):reasons.append('dynamic_boundary_occluded_or_outside_anchor_hull')
        if q.get('N_interval') and np.ptp(q['N_interval'])>4.:reasons.append('dynamic_model_range_exceeds_four_pitch')
        if tangent.get('interval') and np.ptp(tangent['interval'])>120.:reasons.append('dynamic_column_range_exceeds_120px')
        if reasons:return False,reasons
        self.history.append((t,float(q['d']),float(q['N']),max(.2,np.ptp(q['N_interval'])/2)))
        h=np.array(self.history)
        if len(h)<4 or h[-1,0]-h[0,0]<.3-1e-6:return False,['dynamic_current_geometry_warmup']
        if np.max(np.diff(h[:,0]))>.3+1e-6:return False,['dynamic_current_evidence_gap']
        if np.percentile(abs(h[:,1]-np.median(h[:,1])),90)>80:return False,['dynamic_column_inconsistent_across_current_frames']
        # Frame-to-frame differencing amplifies sub-pitch image noise at 10 Hz.
        # Test the robust trajectory against each frame's existing uncertainty.
        slopes=[(b[2]-a[2])/(b[0]-a[0]) for k,a in enumerate(h) for b in h[k+1:] if b[0]-a[0]>=.2-1e-6]
        velocity=float(np.median(slopes));intercept=float(np.median(h[:,2]-velocity*(h[:,0]-t)))
        residual=abs(h[:,2]-(intercept+velocity*(h[:,0]-t)))/h[:,3]
        if abs(velocity)>3. or np.percentile(residual,90)>2.:
            return False,['dynamic_trajectory_exceeds_motion_or_uncertainty_envelope']
        return True,[]

def evaluate(groups,data,boundary,geometry,calibration,gate,t,epoch):
    axis=geometry.get('axis');tc=geometry.get('tangent') or {}
    q=measure(groups,data,boundary,axis,calibration)
    q.update(measurement_mode='fixed_column',fixed_d=calibration.get('d'),fixed_d_calibrated=calibration.get('d') is not None)
    if calibration.get('d') is not None:return q,None
    if not tc.get('valid') or not axis or not axis.get('current_coil_side_measured'):
        q['measurement_mode']='unavailable';return q,None
    local=measure(groups,data,boundary,axis,dict(status='current_local_geometry_not_fixed',d=tc['d'],interval=tc['interval']))
    accepted,reasons=gate.step(t,epoch,local,axis,tc)
    diagnostic={k:v for k,v in local.items() if k!='central'}
    diagnostic.update(gate_passed=accepted,gate_reasons=reasons)
    if not accepted:
        q['measurement_mode']='unavailable';q['reasons']+=reasons;return q,diagnostic
    local.update(measurement_mode='current_dynamic_column',fixed_d=None,fixed_d_calibrated=False,
        calibration_status='current_local_geometry_not_fixed',status='current_dynamic_pitch_estimate',
        dynamic_gate_support=len(gate.history),physical_contact_confirmed=False,
        reasons=local['reasons']+['current_column_estimate_d_not_locked'])
    return local,diagnostic
