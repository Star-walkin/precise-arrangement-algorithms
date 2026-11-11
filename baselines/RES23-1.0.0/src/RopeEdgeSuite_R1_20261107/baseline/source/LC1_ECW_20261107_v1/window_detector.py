"""Causal, timestamp-based projected-proximity event detector; no force-contact claims."""
from collections import deque
import numpy as np

class WindowDetector:
    def __init__(self,enter_duration=.30,hold_duration=.40,exit_duration=.20):
        self.enter_duration=enter_duration;self.hold_duration=hold_duration;self.exit_duration=exit_duration
        self.history=deque();self.active=None;self.events=[];self.last_far=None;self.far_run=[];self.last_time=None

    def finish(self,reason='end_of_video',departure=None):
        if self.active:
            event=self.active.copy();event.update(end_reason=reason,departure_confirmation=departure)
            self.events.append(event);self.active=None
        self.far_run=[]

    def step(self,frame,t,near=False,far=False,camera_valid=True):
        if self.last_time is not None and t<=self.last_time:raise ValueError('timestamps must strictly increase')
        self.last_time=t
        if not camera_valid:
            self.finish('camera_unconfirmed_or_changed');self.history.clear()
            return dict(state='unknown',current_evidence='camera_unconfirmed',event_active=False)
        kind='near' if near else 'far' if far else 'unresolved'
        self.history.append((frame,t,kind))
        while self.history and t-self.history[0][1]>.61:self.history.popleft()
        if far:self.last_far=dict(frame=frame,t=t)
        if self.active:
            if near:
                self.active['last_near']=dict(frame=frame,t=t);self.active['near_frames'].append(frame);self.far_run=[]
            elif far:
                self.far_run.append((frame,t))
                if self.far_run[-1][1]-self.far_run[0][1]>=self.exit_duration-1e-6:
                    self.active['first_departure_evidence']=dict(frame=self.far_run[0][0],t=self.far_run[0][1])
                    self.finish('sustained_separation',dict(frame=frame,t=t));self.history.clear()
            else:self.far_run=[]
            if self.active and t-self.active['last_near']['t']>self.hold_duration+1e-6:
                self.finish('evidence_expired');self.history.clear()
        if not self.active and near:
            recent=list(self.history)
            last_far_index=max([i for i,x in enumerate(recent) if x[2]=='far'],default=-1)
            votes=[x for x in recent[last_far_index+1:] if x[2]=='near']
            if len(votes)>=4 and votes[-1][1]-votes[0][1]>=self.enter_duration-1e-6:
                self.active=dict(first_near=dict(frame=votes[0][0],t=votes[0][1]),last_near=dict(frame=frame,t=t),
                                 detection=dict(frame=frame,t=t),previous_far=self.last_far,
                                 near_frames=[x[0] for x in votes],physical_contact_confirmed=False)
        state=('near_candidate' if near else 'candidate_pending_evidence') if self.active else ('separated' if far else 'near_pending' if near else 'unknown')
        return dict(state=state,current_evidence=kind,event_active=self.active is not None,
                    physical_contact_confirmed=False)

def evidence(row,baseline,level=.60):
    """Correlated geometric consistency check; this is not independent voting."""
    p=(baseline.get('observation') or {}).get('pitch_px')
    p=float(p) if p is not None and 12<p<35 else None
    reg=baseline['region'];a=baseline.get('intersection_y_px');b=reg.get('edge_y')
    coarse=bool(reg['valid'] and baseline['s1']['valid'] and a is not None and b is not None)
    coarse_gap=float(a-b) if coarse else None
    rope=row.get('rope',{});w=rope.get('optical_width_vertical_px')
    q=row.get('center_gap_proxy_px',row.get('center_gap_px'));u=row.get('sensitivity_bound_px')
    valid=bool(row['valid'] and q is not None and w and u is not None and u<=.65*w)
    agreement=not coarse or not valid or abs(q-coarse_gap)<=w
    near=bool(valid and agreement and -.60*w<=q<=level*w and (not coarse or coarse_gap<1.15*(p or w)))
    far=bool((valid and q>.90*w) or (coarse and p is not None and coarse_gap>1.5*p))
    if near:far=False
    return dict(near=near,far=far,local_valid=valid,geometric_consistency=agreement,
                current_pitch_px=p,coarse_center_gap_px=coarse_gap,
                proxy_center_gap_px=q if valid else None,
                normalized_proxy=q/w if valid else None,
                physical_gap_px=None,physical_contact_confirmed=False)

def evaluate(rows,baseline,level=.60,enter_duration=.30):
    tracker=WindowDetector(enter_duration=enter_duration);out=[]
    for r,b in zip(rows,baseline):
        e=evidence(r,b,level);t=r['host_time_s']
        s=tracker.step(r['frame'],t,e['near'],e['far'],r.get('camera_valid',True))
        out.append(dict(frame=r['frame'],time_video_s=r['frame']/10,host_time_s=t,evidence=e,temporal=s))
    tracker.finish()
    return out,tracker.events
