"""Independent geometry/temporal audit and same-video comparison, not accuracy GT."""
import os,sys
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import math,copy
from concurrent.futures import ProcessPoolExecutor
from collections import Counter as Tally
from bg4_common import *
from local_observation import restore_shape
from temporal_robust import Counter

def audit_frame(i):
    r=read(ROOT/'full_003/frames'/f'{i:06d}.json');baseline=BG3/'full_001/frames'/f'{i:06d}.json'
    prior=ROOT/'full_001/frames'/f'{i:06d}.json';assert sha(prior)==r['prior_spatial_record_sha256']
    prior_record=read(prior)
    camera_record=ROOT/'full_002/frames'/f'{i:06d}.json'
    assert sha(camera_record)==r['prior_camera_record_sha256']
    assert read(camera_record)['camera_motion']==r['camera_motion']
    assert all(prior_record[k]==r[k] for k in ('BG4','observation','region','s1','reasons'))
    assert sha(baseline)==r['baseline_record_sha256'];old=read(baseline)
    assert old['source_pixel_sha256']==r['source_pixel_sha256']
    tr=r['temporal'];b=r['BG4'];ncurves=0;nrecovered=0;ncross=0;nchecks=0
    assert tr['causal'];assert tr['display_N'] is None or 0<=tr['display_N']<=5
    if tr['state']=='unknown':assert tr['value'] is None and tr['display_N'] is None
    else:
        assert tr['nominal_filter_sigma']<=.650001
        assert tr['last_observation_age_s']<=.300001 and tr['last_phase_age_s']<=1.000001
        if tr['state']=='predicted':assert not tr['observation_accepted']
    if not r['region_gate_passed']:
        assert b is None and r['observation'] is None
        assert tr['state'] in ('unknown','predicted')
    if b:
        assert r['region_gate_passed'];cache=BG3/'cache'/f'{i:06d}.npz';assert sha(cache)==r['cache_sha256']
        with np.load(cache,allow_pickle=False) as f:data={k:f[k] for k in f.files}
        pts=data['points'];lookup={int(q[0]):q[1:3]*2 for q in pts};owned=set();ids=set()
        env=model_envelope(data['surface'],np.array(r['region']['top_curve_xy'])*.5)
        shape=restore_shape(old['BG3']['diagnostics']['shared_shape']);basecurves={q['boundary_id']:q for q in old['BG3']['curves']}
        if b['curves']:
            assert np.all(sample(env,np.vstack([np.asarray(c['xy'])*.5 for c in b['curves']]))>254)
        for c in b['curves']:
            ncurves+=1;xy=np.array(c['xy']);assert len(xy)==len(c['point_states'])
            assert np.allclose(np.diff(xy[:,0]),2);assert c['boundary_id'] not in ids;ids.add(c['boundary_id'])
            assert not owned.intersection(c['point_ids']);owned.update(c['point_ids'])
            xx=np.array([lookup[j][0] for j in c['point_ids']]);assert xy[0,0]>=xx.min()-1e-6 and xy[-1,0]<=xx.max()+1e-6
            if not c['recovered_local']:assert c==dict(basecurves[c['boundary_id']],recovered_local=False)
            else:
                nrecovered+=1;assert xy[0,0]<D<xy[-1,0];assert len(c['point_ids'])>=12
                assert sum(xx<D)>=3 and sum(xx>D)>=3
                assert c['fit_statistics']['residual_q90_pitch']<=.220001 and c['local_support_fraction']>=.35
                for other in b['curves']:
                    if other['boundary_id']==c['boundary_id']:continue
                    z=np.array(other['xy']);lo=max(xy[0,0],z[0,0]);hi=min(xy[-1,0],z[-1,0])
                    if hi<lo:continue
                    xs=np.arange(math.ceil(lo/2),math.floor(hi/2)+1)*2
                    delta=np.interp(xs,xy[:,0],xy[:,1])-np.interp(xs,z[:,0],z[:,1]);nchecks+=1
                    assert np.all(np.abs(delta)>=.45*PITCH*shape.scale(xs*.5)-1e-6)
                    assert not np.min(delta)<0<np.max(delta)
        seen=set();curve_byid={q['boundary_id']:q for q in b['curves']}
        if b['crossings']:
            checkxy=np.array([[D*.5,c['y']*.5] for c in b['crossings']])
            inside,excluded,support=supported_path(data,checkxy)
            assert np.all(inside) and not np.any(excluded)
        for c in b['crossings']:
            ncross+=1;assert c['boundary_id'] not in seen;seen.add(c['boundary_id'])
            xy=np.array(curve_byid[c['boundary_id']]['xy']);assert xy[0,0]<=D<=xy[-1,0]
            assert abs(np.interp(D,xy[:,0],xy[:,1])-c['y'])<1e-8
        if r['observation'] and r['observation']['phase_anchor']:assert b['phase_valid']
    return dict(i=i,curves=ncurves,recovered=nrecovered,crossings=ncross,recovered_order_checks=nchecks)

def stats(values):
    values=np.asarray(values,float)
    if not len(values):return dict(n=0)
    return dict(n=len(values),median=float(np.median(values)),mean=float(np.mean(values)),p95=float(np.percentile(values,95)),maximum=float(np.max(values)))
def runs(mask):
    out=[];count=0
    for q in list(mask)+[False]:
        if q:count+=1
        elif count:out.append(count);count=0
    return out

def metrics(records):
    tr=[r['temporal'] for r in records]
    base=[r['baseline_display_N'] for r in records];new=[t['display_N'] for t in tr]
    def summary(lo,hi):
        rr=records[lo:hi];tt=tr[lo:hi]
        unknown_old=[not r['baseline_count_valid'] for r in rr];unknown_new=[t['state']=='unknown' for t in tt]
        matched=[i for i in range(lo+1,hi) if all(a[j] is not None for a in (base,new) for j in (i-1,i))]
        delta_old=[abs(base[i]-base[i-1]) for i in matched];delta_new=[abs(new[i]-new[i-1]) for i in matched]
        agree=[abs(t['value']-r['observation']['value']) for t,r in zip(tt,rr) if t['value'] is not None and r['observation'] and r['observation']['phase_anchor']]
        return dict(frames=len(rr),old_unknown=sum(unknown_old),new_unknown=sum(unknown_new),
            old_unknown_runs=runs(unknown_old),new_unknown_runs=runs(unknown_new),
            states=dict(Tally(t['state'] for t in tt)),old_displayed=sum(r['baseline_display_N'] is not None for r in rr),
            new_displayed=sum(t['display_N'] is not None for t in tt),
            displayed_states=dict(Tally(t['state'] for t in tt if t['display_N'] is not None)),
            paired_adjacent_count=len(matched),old_paired_abs_delta=stats(delta_old),new_paired_abs_delta=stats(delta_new),
            old_paired_increases=sum(base[i]>base[i-1]+1e-6 for i in matched),new_paired_increases=sum(new[i]>new[i-1]+1e-6 for i in matched),
            deviation_from_current_phase_observation=stats(agree))
    # Deterministically hide three consecutive observations. The model processes
    # the modified prefix only. Evaluate against hidden image estimates, not GT.
    forecast=Counter();held=[]
    for i,r in enumerate(records):
        take=i>=171 and i%23 in (9,10,11);q={k:copy.deepcopy(r[k]) for k in ('nominal_time_s','region','s1','intersection_y_px','camera_motion','observation')}
        if take:q['observation']=None
        result=forecast.step(q)
        obs=r['observation']
        if take and obs and obs['phase_anchor'] and result['state']=='predicted':
            held.append(dict(frame=i,age=result['last_observation_age_s'],forecast=result['value'],hidden_image_estimate=obs['value'],
                absolute_deviation=abs(result['value']-obs['value'])))
    return dict(full=summary(0,len(records)),approach_121_270=summary(121,min(271,len(records))),
        maximum_prediction_age_s=max((t['last_observation_age_s'] for t in tr if t['state']=='predicted'),default=0),
        maximum_internal_sigma=max((t['nominal_filter_sigma'] for t in tr if t['value'] is not None),default=0),
        maximum_camera_displacement_px=max(r['camera_motion'].get('translation_px') or 0 for r in records),
        temporal_events=dict(Tally(e for t in tr for e in t['events'])),
        held_out_forecast_deviation=stats([q['absolute_deviation'] for q in held]),held_out_forecasts=held,
        comparison_is_same_video_no_physical_ground_truth=True,processing_is_offline=True)

def plot(records,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tt=np.arange(len(records))/10
    old=np.array([r['baseline_display_N'] if r['baseline_display_N'] is not None else np.nan for r in records])
    new=np.array([r['temporal']['display_N'] if r['temporal']['display_N'] is not None else np.nan for r in records])
    fig,ax=plt.subplots(2,1,figsize=(12,6.5),sharex=True,gridspec_kw={'height_ratios':[3,1]},layout='constrained')
    ax[0].plot(tt,old,'.',color='#777777',label='BG3 displayed image estimate',alpha=.7)
    ax[0].plot(tt,new,'-',color='#357bb5',lw=1.4,label='BG4 causal output')
    palette={'measured':'#29955b','estimated':'#d19520','predicted':'#377eb8','unknown':'#bd4859'}
    for kind,color in palette.items():
        ix=np.array([r['temporal']['state']==kind for r in records])
        ax[0].scatter(tt[ix],new[ix],s=14,color=color,label='BG4 '+kind)
        ax[1].scatter(tt[ix],np.full(np.sum(ix),list(palette).index(kind)),s=8,c=color)
    ax[0].set_ylim(-.1,5.3);ax[0].set_ylabel('Remaining rope pitches (0–5)');ax[0].legend(ncols=3,fontsize=8)
    ax[1].set_yticks(range(4),list(palette));ax[1].set_xlabel('Video time (s)');ax[0].grid(alpha=.2)
    ax[0].set_title('Same-video comparison; smoothness and availability are not physical accuracy')
    fig.savefig(out/'comparison_timeline.png',dpi=170);plt.close(fig)

def main():
    out=ROOT/'delivery_complete';out.mkdir(exist_ok=True)
    receipt=read(ROOT/'full_003/receipt.json');assert receipt['status']=='completed' and receipt['frames']==601
    binding=read(ROOT/'full_003/binding.json');assert all(sha(p)==v for p,v in binding['code_sha256'].items())
    checks=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for q in pool.map(audit_frame,range(601)):
            checks.append(q)
            if len(checks)%100==0:print('Audited',len(checks),flush=True)
    audit=dict(status='passed',frames=len(checks),failures=0,**{k:sum(q[k] for q in checks) for k in ('curves','recovered','crossings','recovered_order_checks')})
    dump(out/'geometry_temporal_audit.json',audit)
    records=read(ROOT/'full_003/compact_validation.json')
    # Deterministic replay also checks that saved output contains no future use.
    counter=Counter()
    for r in records:assert counter.step(r)==r['temporal']
    result=metrics(records);result['deterministic_causal_replay_passed']=True
    dump(out/'metrics.json',result);plot(records,out)
    print('AUDIT',audit,flush=True)
    print('FULL',result['full'],flush=True);print('APPROACH',result['approach_121_270'],flush=True)
    print('HELD_OUT',result['held_out_forecast_deviation'],flush=True)
if __name__=='__main__':main()
