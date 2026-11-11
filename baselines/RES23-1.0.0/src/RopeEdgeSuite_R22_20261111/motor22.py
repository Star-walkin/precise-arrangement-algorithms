"""Motor-force timing candidate = new-layer start candidate + one visual cycle.

No hardware command is sent. Visual row overlap is not a physical new-layer
certificate; a repeating image is not independent proof of 360 degrees.
"""
import numpy as np
from common22 import *
from rotation22 import estimate_period,normalize

def target_time(start,period):
    if not start or not period.get('available'):return None
    p=period['period_s'];lo,hi=period['period_interval_s']
    return dict(time_s=start['start_s']+p,
        interval_s=[start['start_s']+lo,start['confirmed_s']+hi],
        physical_layer_change_confirmed=False,physical_revolution_confirmed=False,
        source='first_row_overlap_candidate_plus_visual_period',estimated=True)

def axial_check(rows,start,period):
    pre=[r for r in rows if start-14<=r['time_s']<=start-2 and r.get('available') and r.get('N') is not None
        and r['N']>2 and r.get('current_coil_side_axis') and not r.get('boundary_occluded')]
    out=dict(available=False,consistent=False,assumption='steady_level_winding_advances_one_pitch_per_revolution')
    if len(pre)<20 or pre[-1]['time_s']-pre[0]['time_s']<4 or not period.get('available'):return out
    t=np.array([r['time_s'] for r in pre]);v=np.array([r['N'] for r in pre])
    dt=t[:,None]-t[None,:];dv=v[:,None]-v[None,:];mask=dt>=2.
    slope=float(np.median(dv[mask]/dt[mask]));advance=-slope*period['period_s']
    return dict(out,available=True,consistent=bool(.65<=advance<=1.35),pitch_advance_per_visual_cycle=advance,
        slope_pitch_per_s=slope,support_frames=len(pre))

def analyze(rows,overlap,motion,features,fps):
    candidates=[];n=len(rows);frames=[dict(state='waiting_layer_start',event_index=None) for _ in rows]
    used_episodes=set()
    for start in overlap['candidates']:
        ref=round(start['start_s']*fps);confirm=round(start['confirmed_s']*fps)
        if ref<0 or confirm>=n:continue
        # The pre-event approach and current near event must both be present.
        pre=[q for q in motion['frames'][max(0,ref-round(8*fps)):ref] if q['evidence_mode']=='current' and q['motion']=='approaching']
        near=any(q['state']=='near' and q['evidence_mode']=='current' for q in motion['frames'][ref:confirm+1])
        if len(pre)<round(fps) or not near:continue
        episode=next((j for j,e in enumerate(motion.get('events',[])) if e['start_s']<=start['start_s']<=(e.get('end_s') or float('inf'))),None)
        if episode is not None:
            if episode in used_episodes:continue
            used_episodes.add(episode)
        period=estimate_period(features[:confirm+1],fps)
        target=target_time(start,period)
        check=axial_check(rows[:confirm+1],start['start_s'],period)
        event=dict(layer_start_candidate=start,period=period,target=target,
            physical_layer_change_confirmed=False,physical_revolution_confirmed=False,
            hardware_control=False,confirmation_s=None,recurrence_s=None,recurrence_supported=False,axial_cycle_check=check)
        if target is not None:
            # Pose corroboration uses only frames already seen at its timestamp.
            pre_slopes=[r['axis_slope'] for r in rows[max(0,ref-round(5*fps)):ref] if r.get('axis_slope') is not None and r.get('current_coil_side_axis')]
            base=float(np.median(pre_slopes)) if pre_slopes else None
            noise=float(1.4826*np.median(np.abs(np.array(pre_slopes)-base))) if base is not None else None
            pose_hits=[];pose_confirm=None
            for j in range(confirm,min(n,ref+round(2*period['period_s']*fps))):
                r=rows[j]
                if base is not None and r.get('axis_slope') is not None and r.get('current_coil_side_axis') and abs(r['axis_slope']-base)>max(.005,2.5*noise):
                    pose_hits=[k for k in pose_hits if j-k<=round(.5*fps)];pose_hits.append(j)
                    if len(pose_hits)>=4:pose_confirm=j;break
            event['pose_corroboration_s']=pose_confirm/fps if pose_confirm is not None else None
            # Freeze centering from past frames; no future frames enter the period prior.
            _,mean=normalize(features[max(0,confirm-round(18*fps)):confirm+1])
            norm,_=normalize(features,mean)
            center=target['time_s'];margin=max(.3,.2*period['period_s'])
            low=max(confirm+1,round((center-margin)*fps));requested_high=round((center+margin)*fps)
            complete_search=n>=requested_high+3
            high=min(n-3,requested_high)
            indices=list(range(low,high+1));scores=[];regions=period['regions']
            for j in indices:
                per=np.mean(np.sum(norm[j-2:j+1]*norm[ref-2:ref+1],axis=-1),axis=0)
                scores.append(per)
            event['recurrence_search_s']=[low/fps,high/fps]
            if scores:
                scores=np.array(scores);mean_scores=np.mean(scores[:,regions],axis=1)
                jbest=int(np.argmax(mean_scores));framebest=indices[jbest]
                sector_peaks=[indices[int(np.argmax(scores[:,region]))]/fps for region in regions]
                good_sectors=sum(abs(t-framebest/fps)<=.3+1e-8 for t in sector_peaks)
                event.update(recurrence_score=float(mean_scores[jbest]),sector_peak_s=sector_peaks,
                    recurrence_curve=[dict(time_s=j/fps,score=float(s)) for j,s in zip(indices,mean_scores)])
                # Peak must be interior with observed falling side. This confirmation is retrospective.
                if complete_search and 0<jbest<len(indices)-1 and mean_scores[jbest]>.18 and good_sectors>=2 and pose_confirm is not None and check['consistent']:
                    event.update(recurrence_supported=True,recurrence_s=framebest/fps,
                        confirmation_s=max((high+2)/fps,pose_confirm/fps),
                        observed_cycle_duration_s=(framebest-ref)/fps)
                    target['refined_time_s']=framebest/fps
                    target['refined_interval_s']=[max(low/fps,min(sector_peaks)-.1),min(high/fps,max(sector_peaks)+.1)]
        idx=len(candidates);candidates.append(event)
        for j in range(confirm,n):
            t=rows[j]['time_s']
            if target is None:state='period_unresolved'
            elif event.get('pose_corroboration_s') is None or t<event['pose_corroboration_s']:state='layer_start_unverified'
            elif event['confirmation_s'] is not None and t>=event['confirmation_s']:state='cycle_return_candidate'
            elif t<target['interval_s'][0]:state='first_turn_prediction'
            elif t<=target['interval_s'][1]:state='motor_time_window_candidate'
            else:state='cycle_evidence_pending'
            visible_target=dict(target) if target is not None else None
            confirmed=event['confirmation_s'] is not None and t>=event['confirmation_s']
            if visible_target is not None and not confirmed:
                visible_target.pop('refined_time_s',None);visible_target.pop('refined_interval_s',None)
            if state=='layer_start_unverified':visible_target=None
            frames[j]=dict(state=state,event_index=idx,target=visible_target,period_s=period.get('period_s'),
                confirmation_s=event['confirmation_s'] if confirmed else None,physical_layer_change_confirmed=False,hardware_control=False)
    return dict(events=candidates,frames=frames,algorithm='MFT1 visual first-turn timing candidate',
        uses_prediction=True,physical_layer_change_confirmed=False,physical_revolution_confirmed=False)
