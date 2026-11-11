"""Causal [remaining pitch, velocity] filter with bounded evidence lifetime."""
import math
from collections import deque
import numpy as np

PARAMETERS=dict(maximum_prediction_s=.30,maximum_geometry_only_s=1.0,
    maximum_state_sigma=.65,acceleration_sigma=.45,innovation_minimum_gate=.80,
    initialization_points=3,initialization_window_s=1.,reversal_confirmation_points=5,
    reversal_minimum_span_s=.4,reversal_minimum_net_rise=.25,direction_evidence_lifetime_s=1.)

class Counter:
    def __init__(self):
        self.x=None;self.P=None;self.last_time=None;self.last_obs=None;self.last_phase=None
        self.history=deque(maxlen=14);self.bias=deque(maxlen=12);self.seeds=deque(maxlen=6)
        self.direction='unconfirmed';self.reverse=[];self.outliers=[];self.needs_d_relock=False
        self.last_edge=None;self.last_pitch=None;self.last_angle=None;self.pitch_pending=[];self.direction_time=None
    def clear(self):
        self.x=None;self.P=None;self.last_obs=None;self.last_phase=None
        self.history.clear();self.bias.clear();self.seeds.clear();self.reverse=[];self.outliers=[]
        self.direction='unconfirmed';self.last_edge=None;self.last_pitch=None;self.last_angle=None;self.pitch_pending=[];self.direction_time=None
    def initialize(self,t,obs):
        if not obs or not obs['phase_anchor']:return False
        self.seeds.append((t,float(obs['value']),float(obs['sigma'])))
        seeds=[q for q in self.seeds if t-q[0]<=1.000001]
        if len(seeds)<3:return False
        tt=np.array([q[0]-t for q in seeds]);yy=np.array([q[1] for q in seeds])
        v,b=np.polyfit(tt,yy,1);res=yy-(v*tt+b)
        if abs(v)>3 or np.max(np.abs(res))>.45:return False
        self.x=np.array([float(b),float(v)]);self.P=np.diag([max(.18,obs['sigma'])**2,.7**2])
        self.last_obs=t;self.last_phase=t
        self.history.extend((q[0],q[1]) for q in seeds)
        return True
    def step(self,r):
        t=float(r['nominal_time_s']);dt=.1 if self.last_time is None else t-self.last_time
        if not .001<dt<=1.00001:raise ValueError('Frames must be chronological with bounded time gaps')
        self.last_time=t;obs=r.get('observation');events=[];accepted=False
        motion=r.get('camera_motion',{});region=r.get('region',{});s1=r.get('s1',{})
        hard=[]
        if motion.get('camera_changed'):
            self.needs_d_relock=True;hard.append('camera_changed_requires_new_d')
        if self.needs_d_relock:hard.append('d_calibration_invalid_after_camera_change')
        if motion.get('unconfirmed_duration_s',0)>.5:hard.append('camera_continuity_unconfirmed')
        active=r.get('intersection_y_px');edge=region.get('edge_y')
        if s1.get('valid') and region.get('valid') and active is not None and edge is not None and active<edge:
            hard.append('active_at_or_beyond_reference_edge')
        pitch=obs.get('pitch_px') if obs else None
        abc=s1.get('current_line_abc');angle=math.atan2(-abc[0],abc[1]) if abc else None
        if region.get('valid') and self.last_edge is not None and edge is not None and abs(edge-self.last_edge)>1.2*(pitch or 23.3):
            hard.append('surface_geometry_step_requires_reacquisition')
        if pitch and self.last_pitch and abs(pitch/self.last_pitch-1)>.25:
            if obs['phase_anchor']:
                self.pitch_pending.append(pitch);self.pitch_pending=self.pitch_pending[-3:]
                if len(self.pitch_pending)==3 and np.ptp(self.pitch_pending)/np.median(self.pitch_pending)<.08:
                    hard.append('confirmed_pitch_step_requires_reacquisition')
                else:obs=None;events.append('unconfirmed_pitch_step_observation_rejected')
            else:obs=None;events.append('geometry_pitch_step_observation_rejected')
        elif pitch:self.pitch_pending=[]
        if angle is not None and self.last_angle is not None and abs(math.atan2(math.sin(angle-self.last_angle),math.cos(angle-self.last_angle)))>.035:
            hard.append('S1_angle_step_requires_reacquisition')
        if hard:self.clear();events.extend(hard)
        if region.get('valid'):self.last_edge=edge
        if angle is not None and s1.get('valid'):self.last_angle=angle
        if self.x is not None:
            F=np.array([[1,dt],[0,1.]])
            g=np.array([dt*dt/2,dt]);Q=np.outer(g,g)*PARAMETERS['acceleration_sigma']**2
            previous=float(self.x[0]);self.x=F@self.x;self.P=F@self.P@F.T+Q
        else:previous=None
        effective=None;innovation=None;used_kind=None
        if obs is not None and not hard:
            value=float(obs['value']);sigma=float(obs['sigma']);anchor=obs['phase_anchor']
            if anchor:
                # A few noisy rising frames must not declare a physical turn.
                # Require >=0.4 s and a net rise before reacquiring direction.
                if self.direction=='approaching' and self.history:
                    reference=self.history[-1][1]
                    if value>reference+.06 or (self.reverse and value>self.reverse[-1][1]-.04):
                        self.reverse.append((t,value,sigma))
                    else:self.reverse=[]
                    self.reverse=[q for q in self.reverse if t-q[0]<=1.0]
                    if len(self.reverse)>=5 and self.reverse[-1][0]-self.reverse[0][0]>=.399999 and self.reverse[-1][1]-self.reverse[0][1]>.25 and np.mean(np.diff([q[1] for q in self.reverse])>0)>=.8:
                        saved=list(self.reverse[-3:]);self.clear()
                        for st,sv,ss in saved:self.initialize(st,dict(value=sv,sigma=ss,phase_anchor=True))
                        self.direction='departing';self.direction_time=t;events.append('reversal_reacquired_from_current_observations')
                        previous=None
                if obs.get('geometry_value') is not None:
                    # Bias is recorded only AFTER innovation acceptance below.
                    pass
            else:
                if not self.bias or self.last_phase is None or t-self.last_phase>1.000001:
                    events.append('geometry_without_recent_phase_anchor');obs=None
                else:value+=float(np.median(self.bias));sigma=max(sigma,.45)
            if obs is not None:
                effective=value
                if self.x is None:
                    if self.initialize(t,obs):accepted=True;used_kind=obs['kind'];events.append('initialized_from_three_phase_frames')
                    else:events.append('collecting_initial_phase_frames')
                else:
                    innovation=value-float(self.x[0]);var=float(self.P[0,0])+sigma*sigma
                    gate=max(.8,min(1.3,3*math.sqrt(var)))
                    if abs(innovation)<=gate:
                        K=self.P[:,0]/var;self.x+=K*innovation
                        M=np.eye(2)-np.outer(K,[1.,0.]);self.P=M@self.P@M.T+np.outer(K,K)*sigma*sigma
                        accepted=True;used_kind=obs['kind'];self.outliers=[]
                    else:
                        events.append('measurement_innovation_rejected')
                        if anchor:
                            self.outliers.append((t,value,sigma))
                            recent=[q for q in self.outliers if t-q[0]<=.6]
                            if len(recent)>=3 and np.ptp([q[1] for q in recent[-3:]])<.6:
                                self.clear()
                                for st,sv,ss in recent[-3:]:self.initialize(st,dict(value=sv,sigma=ss,phase_anchor=True))
                                accepted=self.x is not None;used_kind=obs['kind'];previous=None
                                events.append('consistent_new_phase_reacquired')
                if accepted:
                    self.last_obs=t
                    if anchor:
                        self.last_phase=t;self.history.append((t,value))
                        if pitch:self.last_pitch=pitch
                        if obs.get('geometry_value') is not None:self.bias.append(value-obs['geometry_value'])
        if self.x is not None:
            hist=[q for q in self.history if t-q[0]<=1.5]
            if len(hist)>=5 and hist[-1][0]-hist[0][0]>=.5:
                slopes=[(b[1]-a[1])/(b[0]-a[0]) for j,a in enumerate(hist) for b in hist[j+1:] if b[0]-a[0]>=.3]
                if slopes:
                    med=float(np.median(slopes));agreement=float(np.mean(np.array(slopes)<-.03))
                    if med<-.08 and agreement>=.8:
                        self.direction='approaching';self.direction_time=t
                    elif med>.08 and np.mean(np.array(slopes)>.03)>=.8:
                        self.direction='departing';self.direction_time=t
            if self.direction_time is None or t-self.direction_time>1.000001:
                self.direction='unconfirmed'
            if self.direction=='approaching' and not self.reverse:
                if self.x[1]>0:self.x[1]=0;events.append('approach_velocity_constraint')
                if previous is not None and self.x[0]>previous:
                    self.x[0]=previous;events.append('approach_position_constraint')
            age=t-self.last_obs if self.last_obs is not None else 1e9
            phase_age=t-self.last_phase if self.last_phase is not None else 1e9
            sigma=math.sqrt(max(0,float(self.P[0,0])))
            if age>.300001 or phase_age>1.000001 or sigma>.65:
                events.append('prediction_or_uncertainty_limit');self.clear()
        if self.x is None:
            state='unknown';value=None;sigma=None;age=None;phase_age=None;velocity=None
        else:
            state=used_kind if accepted else 'predicted';value=float(self.x[0]);velocity=float(self.x[1])
            sigma=math.sqrt(max(0,float(self.P[0,0])));age=max(0,t-self.last_obs);phase_age=max(0,t-self.last_phase)
        shown=value if value is not None and 0<=value<=5 else None
        return dict(state=state,value=value,display_N=shown,display_mode='within_five' if shown is not None else ('unknown' if value is None else 'outside_five'),
            velocity_pitch_per_s=velocity,nominal_filter_sigma=sigma,last_observation_age_s=age,last_phase_age_s=phase_age,
            direction=self.direction,observation_accepted=accepted,effective_observation=effective,innovation=innovation,
            geometry_bias=float(np.median(self.bias)) if self.bias else None,events=events,
            uncertainty_is_not_calibrated_absolute_error=True,causal=True)
