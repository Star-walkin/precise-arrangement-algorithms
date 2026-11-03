"""Independent finite H5 artifact and length-priority audit. No detector, model, labels or hardware imports."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse, gzip, hashlib, json, sys, traceback
import numpy as np
import cv2
from PIL import Image

HERE=Path(__file__).resolve().parent
H4=HERE.parent/'postprocess_b_20261031_v4'
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

def validate_score(c,params,free):
    credit=c['metrics']['straight_credit']; detail=c['metrics']['score']
    fp=np.asarray(free['points_xy'],float); d=fp[-1]-fp[0]; d/=np.linalg.norm(d)
    t=(fp-fp[0])@d; protect=np.asarray(free.get('observed_straight_support',free['protect_support']),bool)
    post=np.asarray(free['point_status'])!='entry_condition'
    good=protect[:-1]&protect[1:]&post[:-1]&post[1:]
    full=float((2*np.diff(t))[good].sum())
    assert np.isclose(credit['full_free_supported_original_px'],full,atol=1e-6)
    limit=credit['geometric_straight_limit_original_px']/2
    measured=float((2*np.maximum(0,np.minimum(t[1:],limit)-np.minimum(t[:-1],limit)))[good].sum())
    assert np.isclose(credit['supported_straight_original_px'],measured,atol=1e-6)
    assert 0<=measured<=full+1e-6
    assert not credit['input_seed_length_counted'] and not credit['unknown_length_counted']
    assert np.isclose(credit['supported_straight_loss_original_px'],max(0,full-measured),atol=1e-6)
    tier=int(np.floor((measured+1e-7)/params['length_priority_bin_original_px']))
    assert detail['length_priority_tier']==tier
    raw=sum(detail['terms'].values())
    secondary=float(np.clip((raw+params['secondary_score_shift'])/params['secondary_score_scale'],0,params['secondary_score_maximum']))
    assert np.isclose(detail['secondary_raw'],raw,atol=1e-9)
    assert np.isclose(detail['secondary_utility'],secondary,atol=1e-9) and 0<=secondary<1
    assert np.isclose(detail['terms']['straight'],params['straight_weight']*min(1,measured/params['length_reference_original_px']),atol=1e-9)
    connection=c['kind']=='straight_free_to_F3_center_track'
    if connection:
        assert credit['supported_straight_loss_original_px']<=max(params['maximum_early_supported_loss_original_px'],2*max(3.,free['width_half_px']))+1e-6
        assert np.isclose(detail['terms']['direction'],params['direction_weight']*c['geometry_score'],atol=1e-9)
        expected=tier+secondary if c['continuation_eligible'] else -1+secondary
    else:
        assert np.isclose(measured,full,atol=1e-5)
        assert detail['terms']['direction']==0 and detail['terms']['center_track']==0
        expected=tier+secondary
    assert np.isclose(c['rank_score'],expected,atol=1e-9)
    assert np.isclose(detail['loss'],-c['rank_score'],atol=1e-9)
    assert detail['utility_is_not_probability']

def validate_raw_a7(f,record):
    if 'raw_a7_supported' not in f:return
    p=np.asarray(f['points_xy'],float).reshape(-1,2); n=len(p)
    for key in ('raw_a7_supported','observed_straight_support','B_conflict'):assert len(f[key])==n
    raw=np.asarray(f['raw_a7_supported'],bool); heat=np.asarray(f['protect_support'],bool)
    observed=np.asarray(f['observed_straight_support'],bool)
    assert np.array_equal(observed,heat|raw)
    assert np.array_equal(np.asarray(f['B_conflict']),raw&~heat)
    if not n:return
    ev={k:np.asarray(v) for k,v in f['evidence'].items()}; hp=f['diagnostics']['parameters']
    assert not (raw&~ev['raw_a7_supported']).any()
    if raw.any():
        assert f['diagnostics']['raw_a7_eligible'] and f['diagnostics']['accepted_A7_match']
        source=next(q for q in record['input_dependency_sha256'] if 'camera_condition_free_straight_20261030_v7' in q and q.endswith('.json'))
        assert sha(source)==record['input_dependency_sha256'][source]
        result=load(source)['result']; candidate=next(c for c in result['candidates'] if c['candidate_id']==f['diagnostics']['a7_candidate_id'])
        assert candidate['accepted'] and candidate['signature']['status']=='periodic_strand_evidence'
        q=p*2+[0,1050]; cp=np.asarray(candidate['center_xy'],float); direction=np.asarray(candidate['direction_xy'],float);direction/=np.linalg.norm(direction)
        origin=np.asarray(candidate['origin_xy'],float); order=np.argsort((cp-origin)@direction);cp=cp[order];s=(cp-origin)@direction
        flags=(np.asarray(candidate['visible'],bool)&np.asarray(candidate['raw_supported'],bool)&np.asarray(candidate['local_repeat_supported'],bool))[order]
        progress=(q-origin)@direction; ix=np.searchsorted(s,progress).clip(1,len(s)-1);ix-=abs(s[ix-1]-progress)<abs(s[ix]-progress)
        normal=np.array([-direction[1],direction[0]]);distance=abs((q-cp[ix])@normal)
        close=(progress>=s[0])&(progress<=s[-1])&(abs(progress-s[ix])<=max(1.,float(np.median(np.diff(s)))*.55))
        expected=flags[ix]&close&ev['valid']&(distance<=candidate['width_px']*hp['raw_a7_maximum_axis_distance_widths'])&(ev['raw_score']>=hp['raw_a7_minimum_raw_score'])&(ev['raw_contrast']>=hp['raw_a7_minimum_contrast'])
        assert np.array_equal(ev['raw_a7_candidate'],expected)
        assert not (raw&~expected).any()
    post=np.asarray(f['point_status'])!='entry_condition'; ds=2*np.linalg.norm(np.diff(p,axis=0),axis=1)
    hi=heat[:-1]&heat[1:]&post[:-1]&post[1:]; oi=observed[:-1]&observed[1:]&post[:-1]&post[1:]; ri=raw[:-1]&raw[1:]&post[:-1]&post[1:]
    assert np.isclose(f['observed_straight_length_original_px'],float(ds[oi].sum()),atol=1e-5)
    assert np.isclose(f['raw_a7_straight_length_original_px'],float(ds[ri].sum()),atol=1e-5)
    raw_only=raw&~heat
    strict_raw_only=raw_only[:-1]&raw_only[1:]&post[:-1]&post[1:]
    assert np.isclose(f['raw_a7_only_length_original_px'],float(ds[strict_raw_only].sum()),atol=1e-5)
    if 'observed_added_length_original_px' in f:assert np.isclose(f['observed_added_length_original_px'],float(ds[oi&~hi].sum()),atol=1e-5)

def validate_direction(record,params):
    if not all('task_direction' in e for e in record['entries']):return 0
    path=next(q for q in record['input_dependency_sha256'] if 'camera_condition_sam2_' in q and '\\masks\\' in q)
    assert sha(path)==record['input_dependency_sha256'][path]
    mask=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_GRAYSCALE)
    mask=cv2.resize(mask,(mask.shape[1]//2,mask.shape[0]//2),interpolation=cv2.INTER_NEAREST)[525:]
    facts=[]
    for e in record['entries']:
        f=e['free'];p=np.asarray(f['points_xy'],float).reshape(-1,2)
        fact=dict(heat_inside_original_px=0.,joint_inside_original_px=0.,longest_heat_inside_original_px=0.,observed_outside_then_inside=False)
        if len(p)>1:
            inside=cv2.remap(mask.astype(np.float32),p[:,0].astype(np.float32)[None],p[:,1].astype(np.float32)[None],cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT).ravel()>.5
            post=np.asarray(f['point_status'])!='entry_condition'; heat=np.asarray(f['protect_support'],bool)&post; joint=np.asarray(f['supported'],bool)&post; ds=2*np.linalg.norm(np.diff(p,axis=0),axis=1)
            hi=inside&heat;ji=inside&joint;ho=~inside&heat
            change=np.diff(np.r_[False,hi,False].astype(int));runs=zip(np.flatnonzero(change==1),np.flatnonzero(change==-1))
            fact.update(heat_inside_original_px=float(ds[hi[:-1]&hi[1:]].sum()),joint_inside_original_px=float(ds[ji[:-1]&ji[1:]].sum()),longest_heat_inside_original_px=max((float(ds[a:b-1].sum()) for a,b in runs),default=0.),observed_outside_then_inside=bool(ho.any() and hi.any() and np.flatnonzero(ho)[0]<np.flatnonzero(hi)[0]))
        for key,value in fact.items():assert np.isclose(e['task_direction'][key],value,atol=1e-6),(key,value,e['task_direction'][key])
        facts.append(fact)
    excluded=0
    for i,e in enumerate(record['entries']):
        expected=True; aid=e['seed'].get('a7_candidate_id'); db=np.diff(np.asarray(e['seed']['xy']),axis=0).sum(0); db/=np.linalg.norm(db)
        if aid:
            for j,a in enumerate(record['entries']):
                if i==j or a['seed'].get('a7_candidate_id')!=aid:continue
                good=facts[j]; da=np.diff(np.asarray(a['seed']['xy']),axis=0).sum(0); da/=np.linalg.norm(da)
                if good['observed_outside_then_inside'] and good['longest_heat_inside_original_px']>=params['direction_reach_continuous_heat_original_px'] and good['joint_inside_original_px']>=params['direction_reach_joint_original_px'] and da@db<-.8 and facts[i]['heat_inside_original_px']<params['opposite_direction_maximum_heat_inside_original_px']:expected=False
        assert e['task_direction']['selection_allowed']==expected
        excluded+=not expected
    if record['selected_candidate'] is not None:assert record['entries'][record['selected_entry']]['task_direction']['selection_allowed']
    return excluded

def validate_r4_pool(f,reference=None):
    """Audit recorded pool contract without generating or rescoring detection."""
    d=f['diagnostics']
    if 'selected_sample_step_original_px' not in f:return
    p=np.asarray(f['points_xy'],float); step=f['selected_sample_step_original_px']
    assert step in (6.,12.) and np.allclose(2*np.linalg.norm(np.diff(p,axis=0),axis=1),step,atol=1e-6)
    assert step==d['selected_sample_step_original_px']==d['selected_axis']['step_original_px']
    assert d['profile_evaluations']==d['axis_candidates']==d['coarse_axes']+d['refined_axes']<=180
    assert d['shadow_profile_evaluations']==0 and d['coarse_candidates_eligible_for_final_selection']
    assert d['selected_axis']['length_bin']==d['maximum_generated_union_length_bin']
    assert all(x['length_bin']<=d['maximum_generated_union_length_bin'] for x in d['best_coarse_candidates']+d['best_refined_candidates'])
    assert len(d['refinement_parent_axes'])<=3
    for parent in d['refinement_parent_axes']:
        b=parent['b_only_shadow']
        assert parent['angle_degrees']==b['angle_degrees'] and parent['offset_widths']==b['offset_widths']
        assert b['raw_a7_straight_length_original_px']==b['raw_a7_only_length_original_px']==b['observed_added_length_original_px']==0
        assert b['observed_straight_length_original_px']==b['visible_straight_length_original_px']
    if reference and reference.get('points_xy'):
        old=reference['diagnostics']['refinement_parent_axes']; new=d['refinement_parent_axes']
        assert len(old)==len(new)
        for prior,current in zip(old,new):
            for k in ('angle_degrees','offset_widths','score','visible_straight_length_original_px','length_bin'):
                assert np.isclose(prior[k],current['b_only_shadow'][k],atol=1e-9),('R2_B_parent_changed',k)

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
        for p,digest in (start['sources']|start['reused_sources']).items():
            snapshot=run/'source_snapshot'/Path(p).name if p in start['sources'] else Path(p)
            assert sha(snapshot if snapshot.is_file() else p)==digest,('changed_recorded_source',p)
    except Exception as e:errors.append(dict(scope='run',error=str(e),traceback=traceback.format_exc()))
    for sid in start['input_ids']:
        try:
            r=load(run/'记录'/(sid+'.json.gz')); old=load(H4/'all_001/记录'/(sid+'.json.gz'))
            old2=load(H2/'all_001/记录'/(sid+'.json.gz')); job=jobs[sid]
            r2_path=HERE/'pilot_002/记录'/(sid+'.json.gz');r2=load(r2_path) if r2_path.is_file() else None
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
                observed=np.asarray(f.get('observed_straight_support',f['protect_support']),bool)
                raw_a7=np.asarray(f.get('raw_a7_supported',[False]*n),bool)
                validate_raw_a7(f,r)
                validate_r4_pool(f,r2['entries'][ei]['free'] if r2 else None)
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
                    post=np.asarray(f['point_status'])!='entry_condition'
                    good=protect[:-1]&protect[1:]&post[:-1]&post[1:]
                    observed_length=float((2*np.linalg.norm(np.diff(p,axis=0),axis=1))[good].sum())
                    assert np.isclose(f['visible_straight_length_original_px'],observed_length,atol=1e-5)
                    for j,status in enumerate(f['point_status']):
                        if status=='entry_condition':continue
                        if status in ('image_supported','joint_heat_image_supported'):assert supported[j]
                        elif status=='heatmap_supported':assert protect[j] and not supported[j]
                        elif status=='raw_a7_supported':assert raw_a7[j] and not protect[j]
                        else:assert not observed[j],status
                else:counts['empty_free_conditions']+=1
                cs=entry['connection'].get('candidates',[]); assert len(cs)<=3
                if n and observed.any():assert sum(c['kind']=='free_fragment_no_resolved_connection' for c in cs)==1
                else:assert not cs
                assert all(cs[j]['rank_score']>=cs[j+1]['rank_score'] for j in range(len(cs)-1))
                for ci,c in enumerate(cs):
                    counts['candidates']+=1; cp=aligned_points(c['points_xy']); m=len(cp); assert m>=2
                    assert all(len(c[k])==m for k in ('width_half_px','supported','point_status','stage'))
                    assert not c['identity_verified']
                    validate_score(c,entry['connection']['parameters'],f)
                    stages=np.asarray(c['stage']); fp=cp[stages=='free']; nf=len(fp)
                    assert np.array_equal(fp,p[:nf])
                    assert c['point_status'][:nf]==f['point_status'][:nf]
                    assert c['supported'][:nf]==f['supported'][:nf]
                    if c['kind']=='free_fragment_no_resolved_connection':
                        assert np.array_equal(cp,p)
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
                entry_rows.append(dict(entry=ei,H4_free_length_original_px=length(before['free']['points_xy']),H5_free_length_original_px=length(p),
                    H5_heat_supported_points=int(protect.sum()),H5_raw_and_heat_supported_points=int(supported.sum()),
                    H5_heat_only_points=int((protect&~supported).sum()),H5_unknown_points=int((~observed).sum()),
                    selected_axis=diag.get('selected_axis'),normal_shift_original_px=diag.get('normal_shift_original_px'),
                    free_endpoint_original_xy=(p[-1]*2+[0,1050]).tolist() if n else None,
                    condition_candidate_count=len(cs)))
            counts['task_direction_excluded_conditions']+=validate_direction(r,start['parameters'])
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
        response_files_rehashed=len(checked),original_images_rehashed=False,new4_and_reused4_recorded_sources_verified=not any(x.get('scope')=='run' for x in errors),source_check='Frozen per-run source_snapshot when present; reused source file otherwise. Current development files may belong to later revisions.',
        errors=errors,records=rows,scope='Finite artifact and support-provenance audit; no labels, protected data, training or devices.',
        pool_scope='R4 checks actual 6/12px coordinate spacing, recorded maximum generated length bin, finite budget, B-only shadow semantics, and exact R2 parent comparison where prior pilot exists. Full unreported coarse arrays are not independently replayed.',
        caveat='New B/image support may differ from H3 unknown states. This is remeasurement, not new truth. Length priority uses48px bins; geometric departure is not physical contact or identity error. Gray-track adherence, heat response and heuristic utility do not establish target identity.')
    with (run/out).open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','errors')},ensure_ascii=False))
    if errors:print(json.dumps(errors,ensure_ascii=False));raise SystemExit(1)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf8'); parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--out',default='independent_geometry_audit.json');args=parser.parse_args();audit(args.run,args.out)
