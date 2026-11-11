"""Video-only proximity events, distinct from per-frame contact evidence.

All thresholds use pitch units or durations, never video names/timestamps.
An event is an optical near-edge interval, not a physical-contact certificate.
Missing measurements and occlusion cannot provide entry/departure evidence.
"""
import numpy as np

CONFIG=dict(entry_center_pitch=1.25,entry_support=4,history_s=.5,entry_min_span_s=.3,
    departure_rise_pitch=.5,departure_speed_pitch_s=.15,departure_support=3,
    max_gap_s=.5,rearm_rise_pitch=2.,reapproach_drop_pitch=.75)

def eligible(r):
    return bool(r.get('available') and r.get('N') is not None and r.get('current_coil_side_axis')
        and not r.get('boundary_occluded') and r.get('boundary_anchor_hull',True)
        and r.get('N_interval') and r['N_interval'][1]-r['N_interval'][0]<=4.
        and r['N']>=-1.25)

def analyze(rows,fps):
    history=[];state='uninitialized';events=[];outputs=[];event=None;floor=None;peak=None;last_good=None;departure=[];max_missing=0
    for r in rows:
        t=r['time_s'];out=dict(state='unknown',current_evidence=False,uses_prediction=False,physical_contact_confirmed=False)
        history=[q for q in history if t-q['time_s']<=CONFIG['history_s']+1e-6]
        if not eligible(r):
            if event is not None:
                event['unknown_frames'].append(r['frame'])
                if last_good is None or t-last_good>CONFIG['max_gap_s']+1e-6:
                    event.update(end_s=last_good,termination='evidence_lost_not_departure',departure_s=None);events.append(event);event=None;state='uninitialized';floor=None;departure=[]
            outputs.append(out);continue
        last_good=t;history.append(r);n=r['N'];out.update(current_evidence=True)
        med=float(np.median([q['N'] for q in history]));slope=None
        if len(history)>=4 and t-history[0]['time_s']>=.3-1e-6:
            tt=np.array([q['time_s'] for q in history]);nn=np.array([q['N'] for q in history]);slope=float(np.polyfit(tt-tt.mean(),nn,1)[0])
        near=[q for q in history if q['N']<=CONFIG['entry_center_pitch']]
        enter=len(near)>=CONFIG['entry_support'] and near[-1]['time_s']-near[0]['time_s']>=CONFIG['entry_min_span_s']-1e-6
        if state=='departing':
            peak=max(peak or med,med)
            if med>CONFIG['rearm_rise_pitch']:state='separated'
            elif peak-med>=CONFIG['reapproach_drop_pitch'] and slope is not None and slope<-.15:state='uninitialized'
        if event is None and state!='departing' and enter:
            event=dict(start_s=near[0]['time_s'],entry_confirmed_s=t,start_frame=near[0]['frame'],current_evidence_frames=0,
                unknown_frames=[],end_s=None,departure_s=None,physical_contact_confirmed=False,semantics='optical_near_edge_event')
            floor=med;departure=[];state='near'
        if event is not None:
            event['current_evidence_frames']+=1;floor=min(floor,med);state='near'
            rising=slope is not None and slope>=CONFIG['departure_speed_pitch_s'] and med-floor>=CONFIG['departure_rise_pitch']
            if rising:departure.append(t)
            else:departure=[]
            if len(departure)>=CONFIG['departure_support']:
                event.update(end_s=t,departure_s=t,departure_evidence_start_s=departure[0],termination='current_distance_sustained_rise',
                    minimum_center_pitch=floor,departure_center_pitch=med)
                events.append(event);event=None;state='departing';peak=med;departure=[]
        if state not in ('near','departing'):state='separated' if n>2 else 'approaching'
        out.update(state=state,median_current_N=med,current_slope_pitch_per_s=slope,active_event_start_s=event['start_s'] if event else None)
        outputs.append(out)
    if event is not None:
        event.update(end_s=last_good+1/fps,termination='video_ends_during_near_evidence');events.append(event)
    return dict(config=CONFIG,events=events,frames=outputs,physical_contact_confirmed=False,offline_onset_backdating=True,
        onset_timestamps_separate_from_causal_confirmation=True,layer_change_confirmed=False)
