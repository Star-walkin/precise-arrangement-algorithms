"""Causal candidate gating, not material identity verification.

All coordinates are ORIGINAL image pixels. Paths must be ordered from the same
entrance toward the visible end. The caller supplies current-image candidates;
this module never adds a path to a frame or fabricates an image response.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
import numpy as np


@dataclass(frozen=True)
class GateConfig:
    history_frames: int = 5
    history_seconds: float = 2.0
    candidate_limit: int = 3
    sample_count: int = 96
    minimum_common_length_px: float = 80.0
    minimum_shorter_path_overlap: float = 0.55
    fallback_pitch_px: float = 24.0
    pitch_min_px: float = 8.0
    pitch_max_px: float = 64.0
    nominal_seconds_per_pitch: float = 3.0
    speed_multiplier: float = 1.5
    lateral_jitter_pitches: float = 0.35
    maximum_motion_dt_seconds: float = 0.75
    maximum_direction_change_degrees: float = 15.0
    entry_lateral_multiplier: float = 1.5
    maximum_entry_axial_pitches: float = 4.0
    maximum_axial_residual_pitches: float = 2.0
    stale_seconds: float = 1.0
    reset_gap_seconds: float = 2.5
    reacquire_frames: int = 3
    reacquire_minimum_seconds: float = 0.4
    pending_gap_seconds: float = 1.0
    temporal_penalty_weight: float = 0.25
    minimum_image_score: float = 0.0

    def __post_init__(self):
        if self.history_frames < 2 or not 1 <= self.candidate_limit <= 3:
            raise ValueError('history_frames>=2 and1<=candidate_limit<=3 required')
        if not self.stale_seconds < self.reset_gap_seconds:
            raise ValueError('stale limit must precede gap reset')
        if self.reacquire_frames < 2 or self.sample_count < 16:
            raise ValueError('insufficient confirmation/sampling')
        if min(self.pitch_min_px, self.nominal_seconds_per_pitch,
               self.maximum_motion_dt_seconds, self.history_seconds) <= 0:
            raise ValueError('time/scale parameters must be positive')


def _path(points):
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError('path must contain at least2 finite xy points')
    p = p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-6]]
    if len(p) < 2:
        raise ValueError('zero-length path')
    return p


def _arc(p):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def _sample(p, s):
    a = _arc(p)
    return np.column_stack([np.interp(s, a, p[:, k]) for k in range(2)])


def _entry_direction(p):
    distance = min(200., max(20., _arc(p)[-1] * .25))
    q = _sample(p, [0., min(distance, _arc(p)[-1])])
    v = q[1]-q[0]
    return v/max(np.linalg.norm(v), 1e-9)


def compare_paths(reference, current, translation=(0., 0.), config=None):
    """Ordered common-prefix comparison with bounded axial-start alignment.

    The entry tangent's projected start offset accommodates different visible
    entrance endpoints. Normal separation is *not* aligned away. No nearest-
    curve matching or lane identity is inferred. Non-overlapping tails are
    explicitly untested, not silently declared continuous.
    """
    cfg = config or GateConfig()
    r, c = _path(reference), _path(current)
    predicted_r = r + np.asarray(translation, dtype=np.float64)
    axis = _entry_direction(r)
    raw_entry_delta = c[0]-predicted_r[0]
    axial_start = float(raw_entry_delta @ axis)
    r0, c0 = max(axial_start, 0.), max(-axial_start, 0.)
    rl, cl = float(_arc(r)[-1]), float(_arc(c)[-1])
    common = min(rl-r0, cl-c0)
    direction_dot = float(np.clip(_entry_direction(r) @ _entry_direction(c), -1, 1))
    result = dict(common_length_px=max(0., common),
                  common_fraction_of_shorter=max(0., common)/max(min(rl,cl),1e-9),
                  reference_length_px=rl, current_length_px=cl,
                  reference_uncompared_length_px=max(0., rl-common),
                  current_uncompared_length_px=max(0., cl-common),
                  axial_start_offset_px=axial_start,
                  entry_lateral_px=abs(float(raw_entry_delta @ np.array([-axis[1], axis[0]]))),
                  direction_change_degrees=math.degrees(math.acos(direction_dot)))
    if common < 1.:
        return result | dict(valid_comparison=False)
    s = np.linspace(0.,common,cfg.sample_count)
    rs, cs = _sample(predicted_r, r0+s), _sample(c, c0+s)
    tangent = np.gradient(rs, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-9)
    normal = np.column_stack([-tangent[:,1],tangent[:,0]])
    delta = cs-rs
    lateral = np.sum(delta*normal,axis=1)
    axial = np.sum(delta*tangent,axis=1)
    return result | dict(valid_comparison=True,
        lateral_median_abs_px=float(np.median(np.abs(lateral))),
        lateral_p95_abs_px=float(np.quantile(np.abs(lateral),.95)),
        lateral_max_abs_px=float(np.max(np.abs(lateral))),
        axial_p95_abs_px=float(np.quantile(np.abs(axial),.95)),
        median_translation_xy=np.median(delta,axis=0).tolist())


class TemporalRopeGate:
    """Use accepted history only; rejected frames never update that history.

    A separate pending track can establish a *new epoch* after old output has
    expired. REACQUIRED does not claim identity continuity through the gap.
    """
    def __init__(self, config=None):
        self.config = config or GateConfig()
        self.history = deque(maxlen=self.config.history_frames)
        self.pending = []
        self.last_time = None
        self.epoch = 0
        self.events = dict(accepted=0,rejected_frames=0,reacquired=0,resets=0)

    def reset(self, reason='explicit'):
        self.history.clear(); self.pending.clear(); self.last_time=None
        self.epoch += 1; self.events['resets'] += 1
        return dict(reason=reason,epoch=self.epoch)

    def _pitch(self, value):
        if value is not None and np.isfinite(value) and value>0:
            raw=float(value); source='current_projected_spacing_not_physical_calibration'
        elif self.history:
            raw=float(np.median([r['pitch_px'] for r in self.history])); source='accepted_history_projected_spacing'
        else:
            raw=self.config.fallback_pitch_px;source='explicit_uncalibrated_fallback'
        if self.history:
            # One candidate/frame cannot enlarge all gates by supplying a huge pitch.
            old=float(np.median([r['pitch_px'] for r in self.history]))
            raw=float(np.clip(raw,.8*old,1.2*old))
        return float(np.clip(raw,self.config.pitch_min_px,self.config.pitch_max_px)),source

    def _velocity(self,t,pitch):
        recent=[r for r in self.history if t-r['time_s']<=self.config.history_seconds]
        vectors=[]
        for a,b in zip(recent[:-1],recent[1:]):
            dt=b['time_s']-a['time_s']
            if dt<=0:continue
            q=compare_paths(a['points_xy'],b['points_xy'],config=self.config)
            if q['valid_comparison']:vectors.append(np.asarray(q['median_translation_xy'])/dt)
        v=np.median(vectors,axis=0) if vectors else np.zeros(2)
        cap=pitch/self.config.nominal_seconds_per_pitch*self.config.speed_multiplier
        norm=np.linalg.norm(v)
        if norm>cap:v=v*cap/norm
        return v

    def _check(self,reference,current,dt,pitch,velocity):
        cfg=self.config
        clipped_dt=min(dt,cfg.maximum_motion_dt_seconds)
        translation=np.asarray(velocity)*clipped_dt
        q=compare_paths(reference,current,translation,config=cfg)
        limit=pitch*(cfg.lateral_jitter_pitches+cfg.speed_multiplier*clipped_dt/cfg.nominal_seconds_per_pitch)
        q.update(predicted_translation_xy=translation.tolist(),motion_dt_seconds=clipped_dt,
                 actual_dt_seconds=dt,lateral_limit_px=limit,pitch_px=pitch)
        reasons=[]
        if not q['valid_comparison']: reasons.append('no_common_path')
        else:
            if q['common_length_px']<cfg.minimum_common_length_px:reasons.append('insufficient_common_length')
            if q['common_fraction_of_shorter']<cfg.minimum_shorter_path_overlap:reasons.append('insufficient_common_fraction')
            if q['lateral_p95_abs_px']>limit:reasons.append('excessive_lateral_jump')
            if q['lateral_max_abs_px']>2*limit:reasons.append('localized_large_departure')
            if q['axial_p95_abs_px']>cfg.maximum_axial_residual_pitches*pitch:reasons.append('incompatible_shape')
        if q['entry_lateral_px']>cfg.entry_lateral_multiplier*limit:reasons.append('entry_jump')
        if abs(q['axial_start_offset_px'])>cfg.maximum_entry_axial_pitches*pitch:reasons.append('entrance_axial_change')
        if q['direction_change_degrees']>cfg.maximum_direction_change_degrees:reasons.append('direction_jump')
        q.update(passed=not reasons,reasons=reasons)
        return q

    def _eligible(self,candidates):
        if len(candidates)>self.config.candidate_limit:
            raise ValueError('caller must retain at most3 current-image candidates')
        good=[]; diagnostics=[]
        for index,c in enumerate(candidates):
            ident=str(c.get('id',index));reasons=[]
            try:p=_path(c.get('points_xy',[]))
            except (ValueError,TypeError):p=None;reasons.append('invalid_geometry')
            if p is not None and _arc(p)[-1]<self.config.minimum_common_length_px:
                reasons.append('insufficient_path_length')
            score=c.get('image_score',0.)
            if not isinstance(score,(float,int,np.floating,np.integer)) or not np.isfinite(score) or not 0<=score<=1:
                reasons.append('invalid_normalized_image_score')
            elif score<self.config.minimum_image_score:reasons.append('weak_image_score')
            if not c.get('image_supported',False):reasons.append('no_current_image_support')
            diagnostics.append(dict(id=ident,input_index=index,image_score=float(score) if isinstance(score,(float,int,np.floating,np.integer)) and np.isfinite(score) else None,eligible=not reasons,reasons=reasons))
            if not reasons:good.append(dict(id=ident,input_index=index,points_xy=p,image_score=float(score),source=c))
        return good,diagnostics

    def _accept(self,t,c,pitch,status,reason,diagnostics,reset_reason=None):
        if status=='REACQUIRED':
            self.history.clear();self.epoch+=1;self.events['reacquired']+=1
        self.history.append(dict(time_s=t,points_xy=c['points_xy'].copy(),pitch_px=pitch,id=c['id']))
        self.pending.clear();self.events['accepted']+=1
        return dict(time_s=t,status=status,reason=reason,epoch=self.epoch,
            accepted=True,accepted_candidate_id=c['id'],accepted_candidate_index=c['input_index'],
            points_xy=c['points_xy'].tolist(),reference_points_xy=None,reference_age_s=0.,
            temporal_history_count=len(self.history),temporal_confirmed=len(self.history)>1,
            identity_verified=False,identity_continuity_through_gap=False if status=='REACQUIRED' or reset_reason else None,
            candidate_diagnostics=diagnostics,reset_reason=reset_reason,
            output_is_current_image_geometry=True,history_updated=True)

    def _pending_update(self,t,eligible,pitch):
        if not eligible:self.pending=[];return None
        if self.pending and t-self.pending[-1]['time_s']>self.config.pending_gap_seconds:self.pending=[]
        choices=[]
        for c in eligible:
            if self.pending:
                q=self._check(self.pending[-1]['candidate']['points_xy'],c['points_xy'],t-self.pending[-1]['time_s'],pitch,np.zeros(2))
                if not q['passed']:continue
                cost=q['lateral_p95_abs_px']/max(q['lateral_limit_px'],1e-9)
            else:cost=0.
            choices.append((c['image_score']-self.config.temporal_penalty_weight*cost,c))
        if not choices:
            self.pending=[]
            chosen=max(eligible,key=lambda c:c['image_score'])
        else:chosen=max(choices,key=lambda x:x[0])[1]
        self.pending.append(dict(time_s=t,candidate=chosen))
        self.pending=self.pending[-max(self.config.reacquire_frames,8):]
        if len(self.pending)>=self.config.reacquire_frames and t-self.pending[0]['time_s']>=self.config.reacquire_minimum_seconds:
            return chosen
        return None

    def update(self,time_s,candidates,pitch_px=None,scene_cut=False):
        t=float(time_s)
        if not np.isfinite(t):raise ValueError('finite source timestamp required')
        if self.last_time is not None and t<=self.last_time:raise ValueError('timestamps must strictly increase; reset for another video')
        eligible,diagnostics=self._eligible(candidates)
        reset_reason=None
        if scene_cut:reset_reason='scene_cut'
        elif self.last_time is not None and t-self.last_time>self.config.reset_gap_seconds:reset_reason='large_input_time_gap'
        if reset_reason:self.reset(reset_reason)
        self.last_time=t
        pitch,scale_source=self._pitch(pitch_px)
        reference_fresh=bool(self.history and t-self.history[-1]['time_s']<=self.config.stale_seconds)
        consulted_history_times=[r['time_s'] for r in self.history if reference_fresh and t-r['time_s']<=self.config.history_seconds]
        consulted_pending_times=[r['time_s'] for r in self.pending]
        if not self.history:
            if eligible:
                chosen=max(eligible,key=lambda c:c['image_score'])
                result=self._accept(t,chosen,pitch,'ACCEPTED','initial_current_image',diagnostics,reset_reason)
            else:
                result=dict(time_s=t,status='UNKNOWN',reason='no_current_image_candidate',epoch=self.epoch,
                    accepted=False,accepted_candidate_id=None,accepted_candidate_index=None,points_xy=None,
                    reference_points_xy=None,reference_age_s=None,temporal_history_count=0,temporal_confirmed=False,
                    identity_verified=False,candidate_diagnostics=diagnostics,reset_reason=reset_reason,history_updated=False)
        else:
            latest=self.history[-1];dt=t-latest['time_s'];velocity=self._velocity(t,pitch)
            passed=[]
            for c in eligible:
                if dt>self.config.stale_seconds:
                    diagnostics[c['input_index']]['temporal']=dict(passed=False,reasons=['accepted_reference_expired'],reference_age_s=dt)
                    continue
                q=self._check(latest['points_xy'],c['points_xy'],dt,pitch,velocity)
                diagnostics[c['input_index']]['temporal']=q
                if q['passed']:
                    cost=q['lateral_p95_abs_px']/max(q['lateral_limit_px'],1e-9)
                    passed.append((c['image_score']-self.config.temporal_penalty_weight*cost,c))
            if passed and dt<=self.config.stale_seconds:
                chosen=max(passed,key=lambda x:x[0])[1]
                result=self._accept(t,chosen,pitch,'ACCEPTED','current_image_and_history_gate',diagnostics)
            else:
                reacquire=self._pending_update(t,eligible,pitch)
                if reacquire is not None and dt>self.config.stale_seconds:
                    result=self._accept(t,reacquire,pitch,'REACQUIRED','new_epoch_after_consecutive_image_supported_candidates',diagnostics)
                else:
                    self.events['rejected_frames']+=1
                    stale=dt<=self.config.stale_seconds
                    result=dict(time_s=t,status='STALE' if stale else 'UNKNOWN',reason='no_eligible_continuous_candidate' if not passed else 'old_reference_expired_pending_reacquisition',epoch=self.epoch,
                        accepted=False,accepted_candidate_id=None,accepted_candidate_index=None,points_xy=None,
                        reference_points_xy=latest['points_xy'].tolist() if stale else None,reference_age_s=dt,
                        temporal_history_count=len(self.history),temporal_confirmed=False,identity_verified=False,
                        candidate_diagnostics=diagnostics,reset_reason=reset_reason,history_updated=False,
                        pending_confirmation_frames=len(self.pending))
        result.update(pitch_px=pitch,pitch_source=scale_source,
            consulted_accepted_history_times_s=consulted_history_times,
            consulted_pending_history_times_s=consulted_pending_times,
            speed_prior='Uncalibrated projected-spacing heuristic: approximately one pitch per3s; not material velocity.',
            source_timestamp_used=True,uses_future=False)
        return result

    def snapshot(self):
        return dict(config=asdict(self.config),epoch=self.epoch,last_time=self.last_time,
            accepted_history=[dict(time_s=r['time_s'],id=r['id'],pitch_px=r['pitch_px']) for r in self.history],
            pending_times=[r['time_s'] for r in self.pending],events=dict(self.events))
