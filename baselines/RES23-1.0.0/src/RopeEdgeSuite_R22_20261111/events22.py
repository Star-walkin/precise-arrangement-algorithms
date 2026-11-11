"""Separate proximity from motion; causal robust trend and bounded display hold.

Raw distance/F3 and current-evidence event timestamps are never changed.
History-held displays cannot contribute to physical/event evidence.
"""
import numpy as np
from common22 import *
from events21b import analyze as near_events,eligible

CONFIG=dict(window_s=2.4,min_span_s=.8,min_points=7,pair_min_lag_s=.5,
    min_speed=.08,min_progress=.22,direction_support=.72,switch_dwell_s=.4,
    history_hold_s=.5,max_trend_age_s=3.,outlier_min_pitch=.8,outlier_mad_scale=4.,
    reacquire_points=3)

def fit(hist):
    if len(hist)<CONFIG['min_points']:return None
    tt=np.array([x[0] for x in hist]);nn=np.array([x[1] for x in hist])
    if tt[-1]-tt[0]<CONFIG['min_span_s']-1e-8:return None
    dt=tt[:,None]-tt[None,:];dn=nn[:,None]-nn[None,:]
    ix=dt>=CONFIG['pair_min_lag_s'];slopes=dn[ix]/dt[ix]
    if not len(slopes):return None
    v=float(np.median(slopes));intercept=float(np.median(nn-v*(tt-tt[-1])))
    res=nn-(intercept+v*(tt-tt[-1]));mad=float(1.4826*np.median(abs(res-np.median(res))))
    support=float(max(np.mean(slopes< -CONFIG['min_speed']),np.mean(slopes>CONFIG['min_speed'])))
    direction='unknown'
    if support>=CONFIG['direction_support'] and abs(v)*(tt[-1]-tt[0])>=max(CONFIG['min_progress'],.8*mad):
        direction='approaching' if v< -CONFIG['min_speed'] else ('departing' if v>CONFIG['min_speed'] else 'unknown')
    return dict(slope=v,N=intercept,residual_mad=mad,support=support,direction=direction,span_s=float(tt[-1]-tt[0]),t=tt[-1])

def analyze(rows,fps):
    base=near_events(rows,fps)
    hist=[];motion='unknown';last_motion_evidence=None;last_good=None;last_output=None;model=None
    candidate=None;candidate_since=None;rejected=[];out=[]
    for r,event in zip(rows,base['frames']):
        t=r['time_s'];hist=[x for x in hist if t-x[0]<=CONFIG['window_s']+1e-8]
        distance=r.get('motion_distance')
        if distance is None:distance=r['N']
        accepted=eligible(r);reason=None
        if accepted and model is not None and last_good is not None and t-last_good<=CONFIG['history_hold_s']+1e-8:
            expected=model['N']+model['slope']*(t-model['t'])
            half=(r['N_interval'][1]-r['N_interval'][0])/2
            threshold=max(CONFIG['outlier_min_pitch'],CONFIG['outlier_mad_scale']*model['residual_mad'],half)
            if abs(distance-expected)>threshold:
                rejected=[q for q in rejected if t-q['time_s']<=.4+1e-8];rejected.append(r)
                vals=[q.get('motion_distance') if q.get('motion_distance') is not None else q['N'] for q in rejected]
                if len(rejected)>=CONFIG['reacquire_points'] and np.ptp(vals)<threshold:
                    hist=[(q['time_s'],v) for q,v in zip(rejected[:-1],vals[:-1])];rejected=[];model=None
                else:accepted=False;reason='isolated_temporal_outlier'
            else:rejected=[]
        rowout=dict(state='unknown',motion='unknown',proximity='unknown',evidence_mode='unavailable',age_s=None,
            current_evidence=False,uses_prediction=False,physical_contact_confirmed=False,
            near_evidence_state=event['state'],rejection_reason=reason)
        if not accepted:
            age=t-last_good if last_good is not None else None
            if age is not None and age<=CONFIG['history_hold_s']+1e-8 and last_output is not None:
                rowout.update({k:last_output[k] for k in ('state','motion','proximity')})
                rowout.update(evidence_mode='history_hold',uses_prediction=True,age_s=round(age,6))
            else:
                motion='unknown';candidate=None;candidate_since=None;model=None;hist=[]
            out.append(rowout);continue
        last_good=t;hist.append((t,distance));currentfit=fit(hist)
        if currentfit is not None:
            model=currentfit;new=model['direction']
            if new!='unknown':
                if new!=candidate:candidate=new;candidate_since=t
                if t-candidate_since>=CONFIG['switch_dwell_s']-1e-8:
                    motion=new;last_motion_evidence=t
            else:candidate=None;candidate_since=None
        if last_motion_evidence is not None and t-last_motion_evidence>CONFIG['max_trend_age_s']:motion='unknown'
        # A short fitted slope is not allowed to override a qualified near event.
        proximity='near' if event['state']=='near' else ('outside_near_zone' if r['N']>2 else 'near_zone')
        state='near' if event['state']=='near' else motion
        # Departure event has independent repeated current support; retain that direction,
        # but do not continuously relatch it after its first confirmation.
        if event['state']=='departing' and (not out or out[-1]['near_evidence_state']!='departing'):
            state=motion='departing';last_motion_evidence=t;candidate='departing';candidate_since=t
        if state=='unknown':state='steady'
        rowout.update(state=state,motion=motion,proximity=proximity,evidence_mode='current',age_s=0.,
            current_evidence=True,trend=currentfit,motion_signal='pixel_gap_over_fixed_reference_pitch' if r.get('motion_distance') is not None else 'N_fallback',
            active_event_start_s=event.get('active_event_start_s'))
        last_output=rowout;out.append(rowout)
    return dict(config=CONFIG,events=base['events'],frames=out,near_detector_config=base['config'],
        physical_contact_confirmed=False,uses_future_frames=False,
        historical_onset_timestamps_separate_from_causal_confirmation=True)
