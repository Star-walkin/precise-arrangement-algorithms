"""CPL1: causal count-derived winding cycle, cross-checked against texture.

The one-pitch-per-revolution assumption applies only to steady level winding.
The two-cycle timeline is a user-proposed phase model, not physical ground truth.
"""
import numpy as np
CONFIG=dict(windows_s=[12.,16.,24.,30.],min_distance_pitch=3.,min_span_s=6.,
    min_bins=12,bin_s=.5,min_advance_pitch=1.8,min_pair_dt_s=2.,max_interval_width=4.,
    max_residual_pitch=.65,max_speed_change_fraction=.6,max_period_disagreement=.25)

def _slope(t,n):
    dt=t[:,None]-t[None,:];dn=n[:,None]-n[None,:];mask=dt>=CONFIG['min_pair_dt_s']
    if not np.any(mask):return None
    return float(np.median(dn[mask]/dt[mask]))

def estimate_count_period(rows,t0):
    fits=[];seen=set();eligible=[]
    causal=[r for r in rows if r['time_s']<=t0]
    if not causal:return dict(available=False,reason='no_history',fits=[])
    epoch=causal[-1].get('epoch')
    for r in causal:
        if r['time_s']>=t0-.5 or not r.get('available') or r.get('N') is None:continue
        if not r.get('current_coil_side_axis') or r.get('boundary_occluded'):continue
        if r.get('epoch')!=epoch:continue
        if r['N']<=CONFIG['min_distance_pitch'] or not np.isfinite(r['N']):continue
        if not r.get('N_interval') or np.ptp(r['N_interval'])>CONFIG['max_interval_width']:continue
        if r.get('state22',{}).get('evidence_mode')!='current':continue
        eligible.append(r)
    for w in CONFIG['windows_s']:
        rr=[r for r in eligible if r['time_s']>=t0-w]
        ids=tuple(r['frame'] for r in rr)
        if ids in seen:continue
        seen.add(ids)
        if len(rr)<20:continue
        # Temporal medians prevent high frame rate from overweighting one instant.
        bins={}
        for r in rr:bins.setdefault(int(np.floor((r['time_s']-t0)/CONFIG['bin_s'])),[]).append(r)
        tt=np.array([np.median([r['time_s'] for r in b]) for b in bins.values()])
        nn=np.array([np.median([r['N'] for r in b]) for b in bins.values()])
        if len(tt)<CONFIG['min_bins'] or tt[-1]-tt[0]<CONFIG['min_span_s']:continue
        v=_slope(tt,nn)
        if v is None or v>=-.08:continue
        intercept=float(np.median(nn-v*(tt-t0)));res=nn-(intercept+v*(tt-t0))
        mad=float(1.4826*np.median(np.abs(res-np.median(res))))
        advance=-v*(tt[-1]-tt[0]);mid=len(tt)//2
        v1=_slope(tt[:mid],nn[:mid]);v2=_slope(tt[mid:],nn[mid:])
        if v1 is None or v2 is None or v1>=0 or v2>=0:continue
        speedchange=abs(v1-v2)/max(abs(v),.01)
        if mad>CONFIG['max_residual_pitch'] or advance<CONFIG['min_advance_pitch'] or speedchange>CONFIG['max_speed_change_fraction']:continue
        period=-1/v
        if not .8<=period<=8:continue
        fits.append(dict(window_s=w,period_s=period,slope_pitch_per_s=v,residual_mad_pitch=mad,
            support_frames=len(rr),support_bins=len(tt),span_s=float(tt[-1]-tt[0]),advance_pitch=advance,
            speed_change_fraction=speedchange,first_s=rr[0]['time_s'],last_s=rr[-1]['time_s'],
            fit_intercept_at_t0=intercept,support_frames_indices=[r['frame'] for r in rr]))
    if not fits:return dict(available=False,reason='insufficient_stable_far_winding',fits=[])
    pp=np.array([f['period_s'] for f in fits]);p=float(np.median(pp))
    if np.ptp(pp)/p>.3:return dict(available=False,reason='window_period_disagreement',fits=fits)
    half=max(.08*p,float(np.max(np.abs(pp-p))),.1)
    return dict(available=True,period_s=p,period_interval_s=[max(.1,p-half),p+half],
        fits=fits,reason=None,source='CPL1_robust_count_progress',assumption='one_pitch_axial_advance_per_revolution_during_steady_level_winding')

def combine_periods(count,texture):
    aa=count.get('available');bb=texture.get('available')
    if not aa and not bb:return dict(available=False,reason='both_periods_unavailable')
    if not (aa and bb):
        p=count if aa else texture
        return dict(available=True,period_s=p['period_s'],period_interval_s=p['period_interval_s'],
            evidence='count_only' if aa else 'texture_only',dual_support=False,reason=None)
    a=count['period_s'];b=texture['period_s'];difference=abs(a-b)/min(a,b)
    if difference>CONFIG['max_period_disagreement']:return dict(available=False,reason='count_texture_period_conflict',relative_difference=difference)
    # Fixed equal frequency weighting; do not tune weights against a target timestamp.
    p=2/(1/a+1/b)
    lo=min(count['period_interval_s'][0],texture['period_interval_s'][0]);hi=max(count['period_interval_s'][1],texture['period_interval_s'][1])
    return dict(available=True,period_s=p,period_interval_s=[lo,hi],evidence='count_and_texture',
        dual_support=True,relative_difference=difference,reason=None,interval_is_model_envelope_not_confidence_interval=True)

def timeline(event,p):
    t0=event['entry_confirmed_s'];lo0=event['start_s'];T=p['period_s'];lo,hi=p['period_interval_s']
    return dict(t0=t0,t0_interval=[lo0,t0],T=T,t1=t0+T,t2=t0+2*T,
        t1_interval=[lo0+lo,t0+hi],t2_interval=[lo0+2*lo,t0+2*hi],
        semantics='two_cycle_phase_reference_not_observed_physical_layer_entry',physical_confirmed=False)
