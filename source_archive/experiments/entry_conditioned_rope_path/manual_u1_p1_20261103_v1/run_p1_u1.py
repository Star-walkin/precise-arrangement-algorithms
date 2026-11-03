"""Latest U1 heat with frozen P1/H6 postprocessing; no model/label imports.

The original run_h6 driver tied inputs to old B/H1/H2 response receipts.
This driver replaces ONLY that IO binding. All geometric, appearance, topology,
free-axis, connection and entry-ranking functions/parameters remain frozen.
"""
from pathlib import Path
import argparse,ast,concurrent.futures,gzip,hashlib,json,sys,time,traceback,platform
import cv2
import numpy as np

HERE=Path(__file__).resolve().parent;BASE=HERE.parent;ROOT=HERE.parents[2]
H6DIR=BASE/'postprocess_b_20261031_v6'
U1DIR=BASE/'manual_r1_20261103_v1'
P1DIR=ROOT/'工程导航/05_实验版本档案/活动绳整链基准_P1_20261101'
BULK=Path('C:/precise_arrangement_results/manual_u1_p1_20261103_v1')
sys.path.insert(0,str(H6DIR))
import run_h6 as H6

PILOT_IDS=['L_run_20260923_01__f000585__left','R_static_20260922_01__f000095__right',
           'dual_20260923_101517__pair0075__right','ruler_left_01']

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def freeze():
    task=load(HERE/'TASK.json');manifest=load(P1DIR/'baseline_manifest.json')
    expected={str(Path(x['origin_path']).resolve()):x for x in manifest['files']}
    pure=[H6DIR/n for n in ['run_h6.py','common.py','surface_tracks.py','heat_free.py','connect_paths.py','topology.py']]
    pure += [BASE/'postprocess_b_20261030_v2'/n for n in ['surface_tracks.py','free_candidates.py','connect_paths.py']]
    pure += [BASE/'postprocess_b_20261030_v1/priors.py']
    checked=[]
    for path in pure:
        saved=expected[str(path.resolve())];snapshot=P1DIR/saved['snapshot']
        assert sha(path)==saved['sha256']==sha(snapshot),'P1 frozen source mismatch: '+str(path)
        checked.append(dict(path=str(path),sha256=saved['sha256'],snapshot=str(snapshot)))
    tree=ast.parse((H6DIR/'connect_paths.py').read_text(encoding='utf-8-sig'))
    func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='connect')
    refs=[n for n in ast.walk(func) if isinstance(n,ast.Name) and n.id=='h1_paths' and isinstance(n.ctx,ast.Load)]
    assert len(refs)==1
    parent={id(c):n for n in ast.walk(func) for c in ast.iter_child_nodes(n)}
    call=parent[id(refs[0])];assert isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='len'
    keyword=parent[id(call)];assert isinstance(keyword,ast.keyword) and keyword.arg=='direct_H1_paths_diagnostic_only'
    selected=load(U1DIR/'selected_checkpoints.json')
    assert selected['common_step']==task['U1_step']==800
    assert selected['paths']['U1']==task['U1_model'] and sha(task['U1_model']['path'])==task['U1_model']['sha256']
    assert sha(H6DIR/'inputs.json')==task['H6_inputs_sha256']
    assert sha(U1DIR/'protocol.json')==task['U1_protocol_sha256']
    assert sha(U1DIR/'inference/gallery/receipt.json')==task['U1_inference_receipt_sha256']
    jobs=load(H6DIR/'inputs.json')['jobs'];latest=load(U1DIR/'protocol.json')['gallery_jobs']
    assert len(jobs)==len(latest)==292
    for a,b in zip(jobs,latest):
        assert a['id']==b['id'] and a['image_sha256']==b['image_sha256']
        assert len(a['seeds'])==len(b['seeds'])
        assert all(np.array_equal(np.asarray(x['xy']),np.asarray(y['xy'])) for x,y in zip(a['seeds'],b['seeds']))
    receipt=dict(P1_source_checks=checked,H1_trace_use='AST proves len(h1_paths) diagnostic only; passed empty, no old B paths read',
        model=task['U1_model'],step=800,input_protocol_sha256=sha(H6DIR/'inputs.json'),TASK_sha256=sha(HERE/'TASK.json'),
        U1_inference_receipt_sha256=task['U1_inference_receipt_sha256'],
        parameters=H6.PARAMETERS,heat_parameters=H6.HEAT_PARAMETERS,surface_parameters=H6.SURFACE_PARAMETERS,
        topology_parameters=H6.TOPOLOGY_PARAMETERS,new_training=False,new_model_forward=False,R1_used=False,
        old_B_heat_or_paths_loaded=False,old_B_dependent_evidence_loaded=False,
        all_heat_dependent_tracks_free_projection_connection_recomputed=True,source_sha256=sha(__file__))
    path=HERE/'adapter_freeze.json'
    if path.exists():assert load(path)==receipt,'Frozen adapter changed'
    else:dump(path,receipt)
    return jobs,receipt

def input_heat(job,k,seed):
    path=U1DIR/'inference/gallery/U1'/job['id']/f'entry_{k:02d}.npz'
    meta=load(path.with_suffix('.json'));model=load(HERE/'TASK.json')['U1_model']
    assert meta['id']==job['id'] and meta['entry_index']==k and meta['status']=='predicted'
    assert meta['model_sha256']==model['sha256'] and meta['model_step']==800 and meta['branch']=='U1'
    assert meta['image_sha256']==job['image_sha256'] and sha(path)==meta['response_sha256']
    assert meta['target_labels_read'] is False and meta['new_model_forward'] is True
    assert np.array_equal(np.asarray(meta['seed_xy']),np.asarray(seed['xy']))
    with np.load(path,allow_pickle=False) as data:heat=data['response'].copy()
    assert heat.shape==(975,2048) and heat.dtype==np.float32 and np.isfinite(heat).all() and 0<=heat.min()<=heat.max()<=1
    return path,meta,heat

def process(job,out):
    started=time.perf_counter()
    priors=H6.load_surface_priors(dict(job,seeds=job['original_seeds']))
    original=priors['original_half'];gray=cv2.cvtColor(original[525:],cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    entries=[];tracksets=[]
    for k,seed in enumerate(job['seeds']):
        path,meta,heat=input_heat(job,k,seed)
        # Every heat-dependent observation is measured on CURRENT U1.
        tracks=H6.build_tracks(priors,heat,gray)
        full_free=H6.describe_free(priors,seed,heat)
        free,projection=H6.prepare_task_free(full_free,tracks['tracks'])
        linked=H6.connect(gray,heat,free,tracks['tracks'],[],priors['surface_half'][525:])
        for c in linked['candidates']:c['free_axis_id']=free.get('free_axis_id','axis_00')
        linked['alternative_axes_used_for_connection']=False
        linked['retained_axis_count']=int(bool(free['points_xy']))+len(free.get('alternative_free_axes',[]))
        entries.append(dict(entry_index=k,seed=seed,response_file=str(path),response_sha256=sha(path),
            response_metadata=meta,free=free,full_free_observation=full_free,free_projection=projection,
            connection=linked,track_diagnostics=tracks['diagnostics'],
            candidate_track_geometry={t['track_id']:t['points_xy'] for t in tracks['tracks']
                if t['track_id'] in {c['track_id'] for c in linked['candidates']}}))
        tracksets.append(tracks)
    H6.task_direction_feasibility(entries,priors['sam2_surface_half'][525:])
    ranked=[]
    for k,e in enumerate(entries):
        for j,c in enumerate(e['connection']['candidates']):ranked.append((c['rank_score'],k,j,c))
    ranked.sort(key=lambda x:(entries[x[1]]['task_direction']['selection_allowed'],x[0],x[3]['metrics'].get('free',{}).get('quality',0)),reverse=True)
    best=ranked[0] if ranked else None;selected=best[1] if best else 0;shown=ranked[:3]
    ambiguous=entries[selected]['connection'].get('ambiguous',True) or bool(job['seeds'][selected].get('direction_ambiguous',False))
    if len(ranked)>1 and ranked[0][1]!=ranked[1][1] and ranked[0][0]-ranked[1][0]<H6.PARAMETERS['ambiguity_margin']:ambiguous=True
    record=dict(version='IR-U1-P1-POST.0.0',P1_postprocess='frozen IR-P1.0.0/H6, new U1 replaces B response only',
        id=job['id'],source_image=job['image_path'],source_image_sha256=job['image_sha256'],
        camera_role=job['camera_role'],source_group=job['source_group'],
        coordinate_system='crop_half_xy; original=2*xy+[0,1050]',entries=entries,selected_entry=selected,
        selected_candidate=best[2] if best else None,displayed_candidates=[dict(entry=k,candidate=j,score=s,
            task_direction_allowed=entries[k]['task_direction']['selection_allowed']) for s,k,j,c in shown],
        surface_tracks=tracksets[selected],input_dependency_sha256=priors['input_dependency_sha256'],
        response_entry_conditions_merged=False,complete_identity_verified=False,labels_read=False,device_connected=False,
        ambiguous=bool(ambiguous),old_B_heat_or_paths_loaded=False,old_H1_paths_diagnostic_count=0,
        all_heat_dependent_evidence_recomputed=True,U1_model=load(HERE/'TASK.json')['U1_model'],
        source_freeze_sha256=sha(HERE/'adapter_freeze.json'),adapter_source_sha256=sha(__file__),
        elapsed_seconds=time.perf_counter()-started)
    path=out/'记录'/(job['id']+'.json.gz')
    with gzip.open(path,'wt',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,allow_nan=False)
    return dict(id=job['id'],entry_count=len(entries),selected_entry=selected,
        selected_candidate=record['selected_candidate'],has_path=best is not None,ambiguous=bool(ambiguous),
        score=best[0] if best else None,kind=best[3]['kind'] if best else None,
        length_original_px=float(2*np.linalg.norm(np.diff(best[3]['points_xy'],axis=0),axis=1).sum()) if best else 0.,
        termination=best[3]['termination'] if best else None,track_count=len(tracksets[selected]['tracks']),
        record=str(path),record_sha256=sha(path),seconds=record['elapsed_seconds'])

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True)
    parser.add_argument('--subset',choices=['pilot','all'],default='pilot');parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();sys.stdout.reconfigure(encoding='utf-8');cv2.setNumThreads(1)
    jobs,frozen=freeze()
    if args.subset=='pilot':jobs=[j for j in jobs if j['id'] in PILOT_IDS];assert len(jobs)==4
    out=BULK/args.run;out.mkdir(parents=True,exist_ok=False);(out/'记录').mkdir()
    dump(out/'run_start.json',dict(command=sys.argv,python=sys.executable,platform=platform.platform(),
        input_ids=[j['id'] for j in jobs],planned=len(jobs),expected_entries=sum(len(j['seeds']) for j in jobs),
        frozen_adapter_sha256=sha(HERE/'adapter_freeze.json'),source_sha256=sha(__file__),
        pure_P1_parameters_unchanged=True,old_B_heat_paths_read=False,labels_read=False,device_connected=False))
    started=time.perf_counter();rows=[];failures=[]
    with (out/'progress.jsonl').open('x',encoding='utf-8') as log,concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        pending={pool.submit(process,j,out):j['id'] for j in jobs}
        for future in concurrent.futures.as_completed(pending):
            try:
                row=future.result();rows.append(row);log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
            except Exception:
                row=dict(id=pending[future],traceback=traceback.format_exc());failures.append(row)
                dump(out/(row['id']+'__failure.json'),row)
            if len(rows)+len(failures)==len(jobs) or (len(rows)+len(failures))%12==0:
                print(dict(done=len(rows),failures=len(failures),planned=len(jobs),seconds=round(time.perf_counter()-started,1)),flush=True)
    summary=dict(exit_code=int(bool(failures)),completed=len(rows),planned=len(jobs),rows=rows,failures=failures,
        entry_conditions=sum(r['entry_count'] for r in rows),seconds=time.perf_counter()-started,
        source_unchanged=all(sha(x['path'])==x['sha256'] for x in frozen['P1_source_checks']),
        frozen_adapter_sha256=sha(HERE/'adapter_freeze.json'),source_sha256=sha(__file__),
        identity_accuracy_not_measured=True,source_denominator=400,old_entry_unresolved_count=108)
    dump(out/'summary.json',summary)
    if failures:print(failures[:1],flush=True)
    return summary['exit_code']

if __name__=='__main__':raise SystemExit(main())
