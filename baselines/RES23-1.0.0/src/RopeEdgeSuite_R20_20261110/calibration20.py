"""Per-video/per-motion-phase d with explicit evidence levels and holdout audit."""
import argparse
from common20 import *
from calibrate_original import fit_row

def collect(item,phase,allowed_frames=None):
    id=item['id'];lo=phase['start'];hi=phase['end'];rows=[];direct=[]
    for f in sorted((ROOT/'geometry_v5'/id/'frames').glob('*.json.gz')):
        i=int(f.name.split('.')[0])
        if not lo<=i<hi:continue
        with gzip.open(f,'rt',encoding='utf8') as h:r=json.load(h)
        t=r.get('tangent') or {};far=r.get('calibration_far_proxy')
        # The static-pitch proxy is only a generous pre-screen. The production
        # guard subsequently requires current F3's lower distance bound >=5.
        if t.get('valid') and far is not None and far>=3 and i%5==0 and (allowed_frames is None or i in allowed_frames):
            rows.append(dict(frame=i,d=t['d'],interval=t['interval'],far_screening_pitch=far,
                current_coil_side_measured=r['axis']['current_coil_side_measured'],geometry_record_sha256=sha(f)))
    for folder in ('paths_original','paths_sg5','paths_partial_sequential'):
        for f in sorted((ROOT/folder/id).glob('p1_*.json')):
            row=read(f);i=row['frame']
            if not lo<=i<hi:continue
            fit=fit_row(row);p1=row['result'];fit['path_file']=str(f);fit['path_sha256']=sha(f)
            fit['model_profile_eligible']=fit['eligible']
            # A geometrically clean fit through distant pieces is not the
            # user's required complete observed bend. Keep that distinction.
            if not p1.get('whole_path_candidate_supported'):
                fit['reasons'].append('whole_current_path_not_supported')
            if not any(abs(x['frame']-i)<=10 for x in rows):fit['reasons'].append('far_five_rope_screen_not_confirmed')
            fit['eligible']=not fit['reasons'];direct.append(fit)
    return rows,direct

def calibrate(item,phase,allowed_frames=None):
    rows,direct=collect(item,phase,allowed_frames);out=dict(**phase,d=None,interval=None,status='unidentified',reasons=[],
        direct_candidates=direct,far_samples=rows,physical_d_confirmed=False,calibration_uses_future_frames=True,
        run_mode='offline_video_validation_not_causal_startup',sensitivity_not_statistical_confidence=True)
    qualified=[x for x in direct if x['eligible']]
    # Hold out whole 5-second blocks, not adjacent highly correlated frames.
    train=[x for x in rows if (x['frame']//50)%2==0];test=[x for x in rows if (x['frame']//50)%2==1]
    holdout_design='alternating_five_second_blocks'
    if (len(train)<3 or len(test)<3) and len(rows)>=6 and rows[-1]['frame']-rows[0]['frame']>=20:
        mid=len(rows)//2;train=rows[:mid];test=rows[mid:];holdout_design='nonoverlapping_early_and_late_blocks_at_least_two_seconds_total'
    if len(qualified)>=3 and np.ptp([x['frame'] for x in qualified])>=20:
        ds=np.array([x['d_candidate'] for x in qualified]);center=float(np.median(ds))
        half=max(15.,2.5*1.4826*np.median(abs(ds-center)),max(max(abs(np.array(x['d_sensitivity_interval'])-center)) for x in qualified))
        if half<=100:out.update(d=center,interval=[center-half,center+half],status='observed_bend_calibrated')
        else:out['reasons'].append('direct_bend_d_scatter_excessive')
    if out['d'] is None and len(train)>=3 and len(test)>=3 and np.ptp([x['frame'] for x in rows])>=20:
        ds=np.array([x['d'] for x in train]);center=float(np.median(ds));mad=float(1.4826*np.median(abs(ds-center)))
        half=max(25.,2.5*mad,float(np.percentile([(x['interval'][1]-x['interval'][0])/2 for x in train],90)))
        errors=np.abs(np.array([x['d'] for x in test])-center);passfrac=float(np.mean(errors<=half))
        out['temporal_holdout']=dict(design=holdout_design,train_frames=[x['frame'] for x in train],test_frames=[x['frame'] for x in test],
            absolute_d_error_median_px=float(np.median(errors)),absolute_d_error_q90_px=float(np.percentile(errors,90)),fraction_in_train_range=passfrac,
            same_geometry_method_not_independent_ground_truth=True)
        if half<=125 and np.percentile(errors,90)<=150:
            # Held-out model disagreement remains in the sensitivity range.
            half=max(half,float(np.percentile(errors,90)))
            # A clean partial optical bend can widen but never certify TC1.
            partial=[x for x in direct if x.get('model_profile_eligible')]
            if partial:half=max(half,max(abs(x['d_candidate']-center) for x in partial))
            if half<=220:out.update(d=center,interval=[center-half,center+half],status='conditional_geometry_estimate')
            else:out['reasons'].append('bend_and_tangent_models_disagree')
        else:out['reasons'].append('tangent_model_not_stable_on_temporal_holdout')
    if out['d'] is None:out['reasons'].append('insufficient_verified_far_bend_or_stable_geometry_samples')
    return plain(out)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');a=ap.parse_args()
    if not a.partial:raise RuntimeError('Use calibrate_guard.py for production calibration with the current-F3 >=5 rope audit')
    phases=read(ROOT/'motion_phases_v2.json')['videos'];videos=[]
    for item in plan():
        ph=next(x for x in phases if x['id']==item['id']);epochs=[calibrate(item,p) for p in ph['epochs']]
        videos.append(dict(id=item['id'],camera=item['camera'],normalized_orientation='left_entering',epochs=epochs))
        print(item['id'],[(p['epoch'],p['status'],None if p['d'] is None else round(p['d'],1),p['interval'],len(p['far_samples'])) for p in epochs],flush=True)
    dump(ROOT/('d_calibration_partial.json' if a.partial else 'd_calibration.json'),dict(videos=videos,partial=a.partial,code_sha256=sha(__file__)))
if __name__=='__main__':main()
