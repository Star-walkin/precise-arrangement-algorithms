"""Independent finite H4 artifact audit. No detector, model, labels or hardware imports."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse, gzip, hashlib, json, sys, traceback
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
H3=HERE.parent/'postprocess_b_20261031_v3'
H2=HERE.parent/'postprocess_b_20261030_v2'

def load(path):
    with (gzip.open if Path(path).suffix=='.gz' else open)(path,'rt',encoding='utf-8-sig') as f:return json.load(f)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def length(p):
    p=np.asarray(p,float).reshape(-1,2)
    return float(2*np.linalg.norm(np.diff(p,axis=0),axis=1).sum()) if len(p)>1 else 0.

def line_residual(p):
    p=np.asarray(p,float).reshape(-1,2)
    if len(p)<3:return 0.
    q=p-p.mean(0);_,_,v=np.linalg.svd(q,full_matrices=False)
    return float(abs(q@v[-1]).max())

def aligned_points(value):
    p=np.asarray(value,float).reshape(-1,2)
    assert np.isfinite(p).all()
    assert ((p[:,0]>=0)&(p[:,0]<2048)&(p[:,1]>=0)&(p[:,1]<975)).all()
    return p

def audit(name,out):
    run=HERE/name; start=load(run/'run_start.json'); summary=load(run/'summary.json')
    config=load(HERE/'inputs.json'); old_inputs=load(H2/'inputs.json')
    jobs={j['id']:j for j in config['jobs']}; old_jobs={j['id']:j for j in old_inputs['jobs']}
    errors=[]; rows=[]; checked={}; counts=Counter(); max_residual=0.; points_exact=0
    try:
        assert len(jobs)==292 and sum(len(j['seeds']) for j in jobs.values())==301
        assert set(jobs)==set(old_jobs)
        assert summary['exit_code']==0 and summary['completed']==summary['planned']==len(start['input_ids'])
        assert summary['source_unchanged']
        assert sha(HERE/'inputs.json')==start['input_protocol_sha256']
        for p,digest in (start['sources']|start['reused_sources']).items():assert sha(p)==digest,('changed_source',p)
    except Exception as e:errors.append(dict(scope='run',error=str(e),traceback=traceback.format_exc()))
    for sid in start['input_ids']:
        try:
            r=load(run/'记录'/(sid+'.json.gz')); old=load(H3/'all_001/记录'/(sid+'.json.gz'))
            old2=load(H2/'all_001/记录'/(sid+'.json.gz')); job=jobs[sid]
            assert len(r['entries'])==len(old['entries'])==len(job['seeds'])
            entry_rows=[]
            for ei,(entry,before) in enumerate(zip(r['entries'],old['entries'])):
                counts['entry_conditions']+=1
                assert entry['seed']==before['seed']==job['seeds'][ei]==old_jobs[sid]['seeds'][ei]
                rp=entry['response_file']; assert rp==before['response_file']==job['response_files'][ei]
                if rp not in checked:checked[rp]=sha(rp)
                assert checked[rp]==entry['response_sha256']==before['response_sha256']
                f=entry['free']; p=aligned_points(f['points_xy']); n=len(p)
                for key in ('supported','protect_support','point_status'):assert len(f[key])==n,(key,n)
                assert len(f['segment_status'])==max(0,n-1)
                supported=np.asarray(f['supported'],bool); protect=np.asarray(f['protect_support'],bool)
                assert not (supported&~protect).any()
                diag=f['diagnostics']; assert not diag['seed_changed'] and not diag['B_response_changed'] and not diag['old_endpoint_used']
                if n:
                    counts['nonempty_free_conditions']+=1
                    residual=line_residual(p); max_residual=max(max_residual,residual); assert residual<1e-7
                    hp=diag['parameters']; ev={k:np.asarray(v) for k,v in f['evidence'].items()}
                    assert all(len(v)==n for v in ev.values())
                    expected_heat=(ev['valid'] & (ev['center']>=hp['minimum_heat_center']) & (ev['side_drop']>=hp['minimum_side_drop']) & (ev['side_distance_asymmetry_widths']<=hp['maximum_side_distance_asymmetry_widths']))
                    expected_image=expected_heat & (ev['raw_score']>=hp['minimum_raw_score']) & (ev['raw_contrast']>=hp['minimum_image_contrast'])
                    assert np.array_equal(ev['heat_profile_supported'],expected_heat)
                    assert np.array_equal(ev['image_heat_supported'],expected_image)
                    assert not (protect&~expected_heat).any()
                    assert np.array_equal(supported,protect&expected_image)
                    assert diag['new_support_source']=='new_original_gray_and_frozen_B_profile_measurement_not_old_mask'
                    assert diag['axis_candidates']<=hp['maximum_total_axis_candidates']
                    assert length(p)<=hp['maximum_axis_original_px']+1e-5
                    for j,status in enumerate(f['point_status']):
                        if status=='entry_condition':continue
                        if status in ('image_supported','joint_heat_image_supported'):assert supported[j]
                        elif status=='heatmap_supported':assert protect[j] and not supported[j]
                        else:assert not protect[j],status
                else:counts['empty_free_conditions']+=1
                cs=entry['connection'].get('candidates',[]); assert len(cs)<=3
                if n and protect.any():assert sum(c['kind']=='free_fragment_no_resolved_connection' for c in cs)==1
                else:assert not cs
                for ci,c in enumerate(cs):
                    counts['candidates']+=1; cp=aligned_points(c['points_xy']); m=len(cp); assert m>=2
                    assert all(len(c[k])==m for k in ('width_half_px','supported','point_status','stage'))
                    assert not c['identity_verified']
                    stages=np.asarray(c['stage']); fp=cp[stages=='free']; nf=len(fp)
                    assert np.array_equal(fp,p[:nf])
                    assert c['point_status'][:nf]==f['point_status'][:nf]
                    assert c['supported'][:nf]==f['supported'][:nf]
                    if c['kind']=='free_fragment_no_resolved_connection':
                        assert c['rank_score']==0 and np.array_equal(cp,p)
                    else:
                        assert c['kind']=='straight_free_to_F3_center_track'
                        counts['surface_candidates']+=1
                        surf=cp[stages=='surface']; tr=entry['candidate_track_geometry'][c['track_id']]
                        tr=tr['points_xy'] if isinstance(tr,dict) else tr
                        ix=np.asarray(c['surface_source_indices'],int)
                        assert len(ix)==len(surf) and (ix>=0).all() and (ix<len(tr)).all()
                        assert np.array_equal(surf,np.asarray(tr)[ix])
                        assert np.all(np.diff(ix)==1) or np.all(np.diff(ix)==-1)
                        points_exact+=len(surf)
                        if ei==old2['selected_entry']:
                            old_track=next(t for t in old2['surface_tracks']['tracks'] if t['track_id']==c['track_id'])
                            assert tr==old_track['points_xy']
                        params=entry['connection']['parameters']; metrics=c['metrics']; width=max(3.,f['width_half_px'])
                        assert metrics['abandoned_heat_supported_fraction']<=params['maximum_abandoned_supported_straight_fraction']+1e-12
                        assert metrics['maximum_abandoned_heat_run_half_px']<=max(12.,params['maximum_abandoned_run_widths']*width)+1e-12
                        eligible=(c['continuation_heat_fraction']>=params['lane_heat_fraction_minimum'] and c['continuation_heat_run_half_px']>=max(12.,params['lane_heat_continuous_widths']*width))
                        assert c['continuation_eligible']==eligible
                        if c['rank_score']>0:assert eligible
                entry_rows.append(dict(entry=ei,H3_free_length_original_px=length(before['free']['points_xy']),H4_free_length_original_px=length(p),
                    H4_heat_supported_points=int(protect.sum()),H4_raw_and_heat_supported_points=int(supported.sum()),
                    H4_heat_only_points=int((protect&~supported).sum()),H4_unknown_points=int((~protect).sum()),
                    selected_axis=diag.get('selected_axis'),normal_shift_original_px=diag.get('normal_shift_original_px'),
                    free_endpoint_original_xy=(p[-1]*2+[0,1050]).tolist() if n else None,
                    condition_candidate_count=len(cs)))
            selected=r['selected_candidate']
            if selected is None:
                counts['selected_none_images']+=1; assert not any(e['connection'].get('candidates') for e in r['entries'])
                selected_kind=None
            else:
                c=r['entries'][r['selected_entry']]['connection']['candidates'][selected]
                selected_kind=c['kind']; counts['selected_'+selected_kind]+=1
                if selected_kind=='straight_free_to_F3_center_track':assert c['continuation_eligible'] and c['rank_score']>0
            with Image.open(run/'四栏检测结果'/(sid+'.jpg')) as image:assert image.size==(8192,1655); image.verify()
            rows.append(dict(id=sid,entry_count=len(r['entries']),selected_kind=selected_kind,selected_entry=r['selected_entry'],entries=entry_rows))
        except Exception as e:errors.append(dict(id=sid,error=str(e),traceback=traceback.format_exc()))
    report=dict(status='passed' if not errors else 'failed',exit_code=int(bool(errors)),created_utc=datetime.now(timezone.utc).isoformat(),
        command=sys.argv,audit_sha256=sha(__file__),run=str(run),checked_images=len(rows),planned_images=len(start['input_ids']),
        counts=dict(counts),surface_points_exact_source_indices=points_exact,maximum_free_line_residual_half_px=max_residual,
        response_files_rehashed=len(checked),original_images_rehashed=False,new4_and_reused4_sources_unchanged=not any(x.get('scope')=='run' for x in errors),
        errors=errors,records=rows,scope='Finite artifact and support-provenance audit; no labels, protected data, training or devices.',
        caveat='New B/image support may differ from H3 unknown states. This is remeasurement, not new truth. Gray-track adherence, heat response and heuristic utility do not establish target identity.')
    with (run/out).open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','errors')},ensure_ascii=False))
    if errors:print(json.dumps(errors,ensure_ascii=False));raise SystemExit(1)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf8'); parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--out',default='independent_geometry_audit.json');args=parser.parse_args();audit(args.run,args.out)
