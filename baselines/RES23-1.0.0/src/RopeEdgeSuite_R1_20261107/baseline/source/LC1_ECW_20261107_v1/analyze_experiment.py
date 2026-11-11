import hashlib
from collections import Counter
import numpy as np
from scipy.optimize import least_squares
from local_measurement import ROOT,BG4,read,dump
from window_detector import evaluate

def fit_plateau(rows,omit=None):
    use=[r for r in rows if r['valid'] and 20<=r['time_video_s']<=35 and
         r.get('center_gap_width') is not None and (omit is None or not omit[0]<=r['time_video_s']<omit[1])]
    t=np.array([r['time_video_s'] for r in use]);y=np.array([r['center_gap_width'] for r in use])
    def model(t,z):return z[2]+z[3]*np.maximum(z[0]-t,0)+z[4]*np.maximum(t-z[0]-z[1],0)
    fits=[least_squares(lambda z:model(t,z)-y,[s,d,0,.4,.4],bounds=([21,0,-.6,.01,.01],[30,10,1,2,2]),loss='soft_l1',f_scale=.18)
          for s,d in [(24,6),(25,4),(23,7)]]
    f=min(fits,key=lambda f:f.cost);z=f.x
    return dict(start_s=float(z[0]),end_s=float(z[0]+z[1]),offset=float(z[2]),parameters=z,
                median_abs_residual=float(np.median(abs(model(t,z)-y))),samples=len(use),
                offline_uses_future=True,physical_contact_confirmed=False)

def main():
    out=ROOT/'delivery';out.mkdir(exist_ok=True)
    rows=read(ROOT/'v3_full/records.json');v2=read(ROOT/'v2_full/records.json');base=read(BG4/'full_003/compact_validation.json')
    states,events=evaluate(rows,base)
    for ev in events:
        lo=ev['first_near']['frame'];hi=ev['last_near']['frame']
        ev.update(first_sample_video_s=lo/10,last_near_sample_video_s=hi/10,
                  detection_video_s=ev['detection']['frame']/10,
                  supported_frames=len(ev['near_frames']),total_span_frames=hi-lo+1,
                  unsupported_frames=[i for i in range(lo,hi+1) if not states[i]['evidence']['near']])
        ev['start_evidence_bracket_video_s']=[ev['previous_far']['frame']/10,lo/10] if ev['previous_far'] else None
        later=[s for s in states[hi+1:hi+16] if s['evidence']['far']]
        if len(later)>=3:
            ev['next_separation_sample_video_s']=later[0]['time_video_s']
            ev['end_evidence_bracket_video_s']=[hi/10,later[0]['time_video_s']]
            ev['retrospective_separation_confirmation_video_s']=later[2]['time_video_s']
            ev['departure_confirmation_note']='later fresh separation samples; causal expired state is not retroactively changed'
    sens=[]
    for threshold in [.45,.60,.75]:
        for dur in [.30,.40,.50]:
            _,ee=evaluate(rows,base,threshold,dur)
            sens.append(dict(threshold_rope_width=threshold,entry_duration_s=dur,
                windows=[dict(start_s=e['first_near']['frame']/10,last_near_s=e['last_near']['frame']/10,detected_s=e['detection']['frame']/10) for e in ee]))
    fit=fit_plateau(rows);fits=[fit_plateau(rows,(k,k+1)) for k in range(22,33)]
    visual=read(ROOT/'visual_review_before_temporal_tuning.json')
    checks=[]
    for v in visual['rows']:
        i=v['frame'];checks.append(dict(**v,automatic_state=states[i]['temporal']['state'],current_evidence=states[i]['evidence']))
    stats=dict(total_frames=len(rows),v2_local_geometry_valid=sum(r['valid'] for r in v2),v3_local_geometry_valid=sum(r['valid'] for r in rows),
        evaluation_interval_s=[20,35],v2_valid_20_35=sum(r['valid'] for r in v2[200:351]),v3_valid_20_35=sum(r['valid'] for r in rows[200:351]),
        state_counts=dict(Counter(s['temporal']['state'] for s in states)),events=events,parameter_sensitivity=sens,
        plateau_fit=fit,leave_one_second_block_out_fit=dict(start_range_s=[min(f['start_s'] for f in fits),max(f['start_s'] for f in fits)],
          end_range_s=[min(f['end_s'] for f in fits),max(f['end_s'] for f in fits)],not_confidence_interval=True),
        assistant_spot_checks=checks,physical_contact_accuracy=None,physical_contact_confirmed=False,
        whole_path_P1_H6_probes=read(ROOT/'p1_probes/summary.json'))
    dump(out/'frame_decisions.json',states);dump(out/'experiment_results.json',stats)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(12,7),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[3,1,1]})
    tt=np.arange(601)/10
    for values,label,color in [(v2,'v2 local geometry','#b8aa94'),(rows,'v3 local geometry','#177b8e')]:
        ii=[i for i,r in enumerate(values) if r['valid'] and r.get('center_gap_width') is not None]
        axes[0].scatter(tt[ii],[values[i]['center_gap_width'] for i in ii],s=9,label=label,c=color,alpha=.65)
    z=fit['parameters'];t=np.linspace(20,35,300);q=z[2]+z[3]*np.maximum(z[0]-t,0)+z[4]*np.maximum(t-z[0]-z[1],0)
    axes[0].plot(t,q,color='#b04c40',label='Offline robust plateau fit (diagnostic)')
    axes[0].axhline(.6,ls='--',lw=1,color='#8b6f35');axes[0].axhline(-.6,ls='--',lw=1,color='#8b6f35')
    axes[0].set(ylim=(-1,4),ylabel='Center/interface proxy / optical rope width',title='ECW: visual proximity evidence; physical contact is uncalibrated')
    axes[0].legend(loc='upper right',fontsize=8)
    for k,(data,label) in enumerate([(v2,'v2 geometry'),(rows,'v3 geometry')]):
        axes[1].scatter(tt,[k+.1 if r['valid'] else np.nan for r in data],s=8,label=label)
    axes[1].set(yticks=[.1,1.1],yticklabels=['v2 valid','v3 valid'],ylim=(-.3,1.5))
    near=[s['evidence']['near'] for s in states];pending=[s['temporal']['state']=='candidate_pending_evidence' for s in states]
    axes[2].scatter(tt,np.where(near,1,np.nan),color='#21877b',s=15,label='Current near evidence')
    axes[2].scatter(tt,np.where(pending,.4,np.nan),color='#a98a40',s=15,label='Pending; not evidence')
    axes[2].set(ylim=(0,1.5),yticks=[],xlabel='AVI playback time (s)');axes[2].legend(fontsize=8,loc='upper left')
    for ax in axes:
        ax.set_xlim(18,37);ax.grid(alpha=.15)
        for e in events:ax.axvspan(e['first_sample_video_s'],e['last_near_sample_video_s'],color='#258b78',alpha=.08)
    fig.savefig(out/'timeline.png',dpi=180);plt.close(fig)
    print({k:v for k,v in stats.items() if k not in ['assistant_spot_checks','whole_path_P1_H6_probes','parameter_sensitivity']})
if __name__=='__main__':main()
