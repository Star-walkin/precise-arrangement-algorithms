"""SG4: strict current S1 first, verified segmented S1 fallback.

Both branches must be outputs from the same current image. A remembered line
only vetoes inconsistent fallback identity; it is never emitted as a detection.
No video IDs, timestamps, or manual target trajectories enter the policy.
"""
import sys,copy,math
import cv2,numpy as np
from common import R12
sys.path.insert(0,str(R12))
from tracker_sg2 import Tracker as WitnessSampler,angle,line_y

PARAMETERS=dict(anchor_length_px=360.,anchor_min_points=7,anchor_min_span_px=220.,
    anchor_max_angle_deg=1.0,anchor_max_median_residual_widths=.15,
    anchor_max_fit_residual_px=2.5,strict_memory_ttl_s=1.,
    strict_memory_max_angle_deg=1.,strict_memory_width_ratio=[.65,1.45])

def line_angle_delta(a,b):
    d=math.degrees(angle(a)-angle(b))
    return abs((d+90)%180-90)

def width_of(track):
    w=track.get('diagnostics',{}).get('seed_validation') or {}
    return float(w.get('width_px') or 24.)

def anchor_of(track):
    d=track.get('diagnostics',{}).get('spatial_anchor') or {}
    x=d.get('x_original')
    seed=track.get('seed_xy_original')
    return float(x) if x is not None else (float(seed[0][0]) if seed else None)

class Fusion:
    def __init__(self):
        self.sampler=WitnessSampler(np.zeros((3000,4096),np.uint8))
        self.anchor_x=None;self.last_strict=None

    def verify_fallback(self,image,track,t):
        abc=track['line_abc'];width=width_of(track);why=[]
        x=self.anchor_x if self.anchor_x is not None else anchor_of(track)
        log=dict(valid=False,anchor_x=x,reasons=why,current_pixels_only=True,
            remeasured_at_fixed_entry_not_reused_component_points=True)
        if x is None:why.append('no_entry_anchor');return log
        q=np.array([[x,line_y(abc,x)],[x+PARAMETERS['anchor_length_px'],line_y(abc,x+PARAMETERS['anchor_length_px'])]])
        wit=self.sampler._seed_witness(image,q);log['witness']=wit
        if not wit.get('valid'):why.append('current_entry_bilateral_support_missing')
        else:
            p=np.asarray(wit['points_xy'],float);span=float(np.ptp(p[:,0]))
            fit=cv2.fitLine(p.astype(np.float32),cv2.DIST_HUBER,0,.01,.01).ravel()
            fitted=np.array([-fit[1],fit[0],fit[1]*fit[2]-fit[0]*fit[3]],float)
            residual=float(np.median(abs(p@np.asarray(abc)[:2]+abc[2])))
            q90=float(np.percentile(abs(p@fitted[:2]+fitted[2]),90))
            da=line_angle_delta(fitted,abc);ratio=wit['width_px']/width
            log.update(span_px=span,median_candidate_residual_px=residual,fit_residual_q90_px=q90,
                angle_difference_deg=da,width_ratio=ratio,independent_anchor_line_abc=fitted.tolist())
            if len(p)<PARAMETERS['anchor_min_points'] or span<PARAMETERS['anchor_min_span_px']:why.append('entry_anchor_support_insufficient')
            if residual>max(2.,PARAMETERS['anchor_max_median_residual_widths']*wit['width_px']):why.append('entry_anchor_position_disagrees')
            if da>PARAMETERS['anchor_max_angle_deg']:why.append('entry_anchor_direction_disagrees')
            if q90>PARAMETERS['anchor_max_fit_residual_px']:why.append('entry_anchor_not_straight')
            if not .65<=ratio<=1.45:why.append('entry_anchor_width_disagrees')
        old=self.last_strict
        if old and 0<=t-old['t']<=PARAMETERS['strict_memory_ttl_s']:
            dt=t-old['t'];da=line_angle_delta(abc,old['abc']);ratio=width/old['width']
            delta=abs(line_y(abc,x)-line_y(old['abc'],x))
            log['strict_identity_memory']=dict(age_s=dt,angle_difference_deg=da,width_ratio=ratio,position_difference_px=delta)
            if da>PARAMETERS['strict_memory_max_angle_deg']:why.append('fallback_disagrees_with_recent_strict_direction')
            if not .65<=ratio<=1.45:why.append('fallback_disagrees_with_recent_strict_width')
            if delta>max(8.,.4*old['width']+10*dt):why.append('fallback_disagrees_with_recent_strict_position')
        log['valid']=not why
        return log

    def step(self,image,strict,relaxed,t,i):
        branch='none';check=None;disagreement=None
        if strict['valid'] and relaxed['valid']:
            a,b=strict['line_abc'],relaxed['line_abc']
            disagreement=dict(angle_difference_deg=line_angle_delta(a,b),
                y_difference_at_x1600_px=abs(line_y(a,1600)-line_y(b,1600)))
        if strict['valid']:
            chosen=copy.deepcopy(strict);branch='strict_SG2'
            self.last_strict=dict(t=t,abc=strict['line_abc'],width=width_of(strict))
            # Use the independent strict entry once available; it cannot march
            # downstream with relaxed component starts.
            self.anchor_x=anchor_of(strict)
        elif relaxed['valid']:
            check=self.verify_fallback(image,relaxed,t)
            if check['valid']:
                chosen=copy.deepcopy(relaxed);branch='verified_segmented_SG3'
                if self.anchor_x is None:self.anchor_x=anchor_of(relaxed)
            else:chosen=self.empty('SG4_current_entry_verification_rejected')
        else:chosen=self.empty('SG4_both_current_branches_unavailable')
        chosen['source']='RES-1.6/SG4';chosen['diagnostics']['SG4']=dict(branch=branch,
            fallback_check=check,branch_disagreement=disagreement,anchor_x=self.anchor_x,
            strict_current_valid=bool(strict['valid']),relaxed_current_valid=bool(relaxed['valid']),
            cached_history_emitted_as_measurement=False)
        if chosen['valid']:chosen['diagnostics']['accept_reason']='SG4_'+branch+'_current_evidence'
        return chosen

    @staticmethod
    def empty(reason):
        return dict(valid=False,current_line_abc=None,line_abc=None,line_points_xy=[],supported_segments_xy=[],
            confidence=0.,model_forward_count=0,diagnostics=dict(accept_reason=reason,attempts=[]))
