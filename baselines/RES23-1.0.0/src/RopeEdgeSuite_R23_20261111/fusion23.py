"""MFT2: two-cycle phase reference with separately gated current image support."""
import numpy as np
from common23 import *
from period23 import estimate_count_period,combine_periods,timeline
from rotation22 import estimate_period,normalize

def current(r):
    return bool(r.get('available') and r.get('current_coil_side_axis') and not r.get('boundary_occluded')
        and r.get('state22',{}).get('evidence_mode')=='current')

def analyze(rows,near_events,features,fps):
    n=len(rows);events=[];outputs=[dict(state='waiting_near_anchor',confirmation_s=None,event_index=None) for r in rows]
    if not n:return dict(events=events,frames=outputs)
    times=np.array([r['time_s'] for r in rows])
    for near in near_events:
        t0=near['entry_confirmed_s']
        if t0>times[-1]+1e-8:continue
        ref=int(np.searchsorted(times,t0-1e-8));assert ref<n
        count=estimate_count_period(rows[:ref+1],t0);texture=estimate_period(features[:ref+1],fps)
        combined=combine_periods(count,texture)
        ev=dict(t0=t0,count_period=count,texture_period=texture,combined_period=combined,
            timeline=None,image_supported=False,candidate_s=None,confirmation_s=None,
            physical_layer_entry_confirmed=False,physical_turn_confirmed=False,
            semantics='user_proposed_two_cycle_reference_and_current_image_checks')
        if combined['available']:
            clock=timeline(near,combined);ev['timeline']=clock
            T=combined['period_s'];margin=max(.4,.2*T)
            low_time=clock['t2']-margin;high_time=clock['t2']+margin
            low=max(ref+3,int(np.ceil((low_time-times[0])*fps-1e-8)))
            high=int(np.floor((high_time-times[0])*fps+1e-8))
            confirmation_index=high+2
            ev['texture_search_s']=[float(times[0]+low/fps),float(times[0]+high/fps)]
            # Do not turn an incomplete clip into an early confirmation window.
            if texture.get('available') and n>confirmation_index and ref>=2:
                _,mean=normalize(features[max(0,ref-round(18*fps)):ref+1]);z,_=normalize(features,mean)
                regions=texture['regions'];indices=list(range(low,high+1));scores=[]
                for j in indices:
                    scores.append(np.mean(np.sum(z[j-2:j+1]*z[ref-2:ref+1],axis=-1),axis=0))
                scores=np.array(scores);score=np.mean(scores[:,regions],axis=1);best=int(np.argmax(score));idx=indices[best]
                region_peaks=[indices[int(np.argmax(scores[:,k]))] for k in regions]
                support=[k for k,j in zip(regions,region_peaks) if abs(j-idx)/fps<=.3+1e-8]
                local=[r for r in rows[max(0,idx-round(.6*fps)):idx+1] if current(r)]
                departure=[r for r in local if r['state22'].get('motion')=='departing']
                ev.update(recurrence_peak_s=float(times[idx]),recurrence_score=float(score[best]),support_regions=support,
                    region_peak_s=[float(times[j]) for j in region_peaks],current_departure_support=len(departure),
                    recurrence_curve=[dict(time_s=float(times[j]),score=float(v)) for j,v in zip(indices,score)])
                qualifies=bool(0<best<len(indices)-1 and score[best]>.18 and len(support)>=2
                    and combined.get('dual_support') and current(rows[idx]) and len(departure)>=3)
                if qualifies:
                    supporting_times=[float(times[j]) for j in region_peaks if abs(j-idx)/fps<=.3+1e-8]
                    ev.update(image_supported=True,candidate_s=float(times[idx]),confirmation_s=float(times[confirmation_index]),
                        image_time_range_s=[min(supporting_times)-1/fps,max(supporting_times)+1/fps])
        k=len(events);events.append(ev)
        for j in range(ref,n):
            t=times[j];clock=ev['timeline'];feedback=ev['confirmation_s'] is not None and t>=ev['confirmation_s']-1e-8
            out=dict(event_index=k,state='period_unavailable',timeline=clock,confirmation_s=None,
                model_is_prediction=True,current_rope_evidence=current(rows[j]),physical_confirmed=False,
                period_source=combined.get('evidence'),period_reason=combined.get('reason'))
            if clock is not None:
                if t<clock['t1']:state='near_transition_reference'
                elif t<clock['t2_interval'][0]:state='new_layer_first_turn_reference'
                elif t<=clock['t2_interval'][1]:state='motor_time_window_reference'
                else:state='model_window_passed_without_image_confirmation'
                if feedback:
                    state='image_supported_motor_candidate'
                    out.update(candidate_s=ev['candidate_s'],confirmation_s=ev['confirmation_s'],image_time_range_s=ev['image_time_range_s'])
                elif t>clock['t2_interval'][1]+clock['T']:
                    state='reference_expired'
                out['state']=state
            outputs[j]=out
    return dict(events=events,frames=outputs,algorithm='CPL1 + MFT2',uses_human_times=False,
        primary_R22_motion_unchanged=True,physical_turn_confirmed=False)
