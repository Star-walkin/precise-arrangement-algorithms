"""Independent saved A5 outputs audit; no detector, device or scoring imports.

Verifies fixed inputs, bytes, image invariants and saved candidate bookkeeping.
Passing does not establish visual accuracy, identity or physical disappearance.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,math,shutil,sys,time,traceback
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import cv2,numpy as np
import whole_io as io

FOLDERS=('01_整根候选掩膜','02_原图叠加','03_图像支持','04_接续待确认')
CORE=('whole_io.py','free_rope.py','photometric_ribbon.py','attached_groove.py','lane_candidates.py','photometric_trace.py','run_experiment.py','protocol.json')
POINT_FIELDS=('widths','phase','point_inferred','formal_boundary_xy','formal_boundary_distance_px','formal_boundary_sources','formal_boundary_inferred_mask','boundary_snap_distance_px','supported','inside_search','groove_score','groove_supported','groove_available')

def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def normalized(path):return str(Path(path).resolve())
def canonical_hash(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()

def finite_paths(value,path='$'):
    if isinstance(value,float) and not math.isfinite(value):return [path]
    if isinstance(value,dict):return [q for k,v in value.items() for q in finite_paths(v,path+'.'+k)]
    if isinstance(value,list):return [q for i,v in enumerate(value) for q in finite_paths(v,path+'['+str(i)+']')]
    return []

def independent_domain(surface):
    binary=np.where(surface>0,255,0).astype(np.uint8);n=int(np.count_nonzero(binary))
    cs,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cs=[c for c in cs if cv2.contourArea(c)>=max(50.,.0001*n)]
    domain=np.zeros_like(binary)
    if not cs:return domain
    hull=np.zeros_like(binary);cv2.fillConvexPoly(hull,cv2.convexHull(np.concatenate(cs)),255)
    if np.count_nonzero(hull)<=1.8*max(1,n):return hull
    cv2.drawContours(domain,cs,-1,255,cv2.FILLED);return domain

def parent_inventory(protocol):
    inventory={};checks=[]
    for name,target in protocol['parent_manifest_sha256'].items():
        p=Path(name);actual=io.sha(p);checks.append({'path':name,'expected':target,'actual':actual,'ok':actual==target})
        if actual!=target:raise ValueError('Parent manifest changed: '+name)
        j=load(p)
        entries=[v for c in j['cases'] for v in c['files']] if 'cases' in j else j.get('files',j.get('file_manifest',[]))
        for v in entries:
            q=v.get('absolute_path')
            if q is None:
                q=Path(v.get('path',v.get('relative_path')))
                if not q.is_absolute():q=p.parent/q
            inventory[normalized(q)]=v['sha256']
    return inventory,checks

def expected_dependencies(row):
    rel=Path(row['output_file']);role=rel.parent.name;stem=rel.stem
    return [io.C4/'roi/surface_masks'/role/rel.name,io.A1/'records'/role/(stem+'.json'),io.A2/'records'/role/(stem+'.json'),io.C3/'boundaries/records'/role/(stem+'.json'),io.F3/'记录/cases'/role/(stem+'.json')]+[io.F3/'记录/masks'/k/role/rel.name for k in ('merged','inferred','exclusion')]

def source_checks(run,start,protocol):
    errors=[];checks=[];frozen=start.get('sources_sha256',{})
    for name in CORE:
        if str(io.HERE/name) not in frozen:errors.append('missing_core_source_hash:'+name)
    external=io.ASSETS/'camera_condition_incoming_rope_20261026_v2/free_refinement.py'
    if str(external) not in frozen:errors.append('missing_external_free_refinement_source')
    names=[Path(k).name for k in frozen]
    if len(names)!=len(set(names)):errors.append('flat_snapshot_collision')
    for name,target in frozen.items():
        source=Path(name);snapshot=run/'source_snapshot'/source.name
        a=io.sha(source) if source.is_file() else None;b=io.sha(snapshot) if snapshot.is_file() else None
        checks.append({'source':name,'expected':target,'current':a,'snapshot':b,'ok':a==b==target})
        if a!=target:errors.append('current_source_changed:'+name)
        if b!=target:errors.append('snapshot_source_changed:'+name)
    if (run/'freeze_snapshot.json').is_file():
        freeze=load(run/'freeze_snapshot.json')
        if freeze.get('sources_sha256')!=frozen:errors.append('freeze_snapshot_vs_executed_sources_mismatch')
    baseline=dict(protocol['source_parent_sha256'])
    f3path=io.F3.parent/'freeze_all.json'
    if io.sha(f3path)!=protocol['F3_sources_freeze_sha256']:errors.append('F3_freeze_changed')
    baseline.update(load(f3path)['source_sha256'])
    baseline[str(Path(protocol['A4_output_root'])/'output_manifest.json')]=protocol['A4_output_manifest_sha256']
    oldchecks=[]
    for name,target in baseline.items():
        actual=io.sha(name) if Path(name).is_file() else None;oldchecks.append({'path':name,'expected':target,'actual':actual,'ok':actual==target})
        if actual!=target:errors.append('old_baseline_changed:'+name)
    return checks,oldchecks,errors

def g0_source_order(run):
    """Inspect frozen source, never import/execute the detector or recompute G0."""
    source=(run/'source_snapshot/photometric_trace.py').read_text(encoding='utf-8')
    tree=ast.parse(source);fn=next((n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='trace_photometric'),None)
    if fn is None:return {'passed':False,'error':'trace_function_not_found'}
    calls={}
    for n in ast.walk(fn):
        if isinstance(n,ast.Call):
            name=n.func.id if isinstance(n.func,ast.Name) else (n.func.attr if isinstance(n.func,ast.Attribute) else '')
            calls.setdefault(name,[]).append(n.lineno)
    assignments=[]
    for n in ast.walk(fn):
        if isinstance(n,ast.Assign):
            for target in n.targets:
                if isinstance(target,ast.Name) and target.id=='g0':assignments.append(n.lineno)
    geometry_line=min(calls.get('generate_lane_candidates',[10**9]));rgb_line=min([i for k in ('build_fields','_straight_candidate','_score') for i in calls.get(k,[])],default=-1)
    g0_line=min(assignments,default=10**9)
    lane_source=(run/'source_snapshot/lane_candidates.py').read_text(encoding='utf-8');lane_tree=ast.parse(lane_source)
    original_reads=[]
    for n in ast.walk(lane_tree):
        if isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Constant) and n.slice.value=='original':original_reads.append(n.lineno)
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='get' and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='original':original_reads.append(n.lineno)
    return {'passed':geometry_line<g0_line<rgb_line and not original_reads,'geometry_pool_call_line':geometry_line,'g0_assignment_line':g0_line,'first_RGB_measurement_call_line':rgb_line,'lane_module_original_RGB_read_lines':original_reads,'scope':'Conditional geometry-only choice from image-derived F3/free geometry; not an entirely image-free pipeline.'}

def candidate_checks(candidate,shape,domain,scored):
    errors=[];q=np.asarray(candidate.get('xy',[]),float);n=len(q);h,w=shape
    if q.ndim!=2 or q.shape[1]!=2 or n<2:return ['invalid_xy_shape'],{}
    if not np.isfinite(q).all() or np.any(q[:,0]<0) or np.any(q[:,0]>=w) or np.any(q[:,1]<0) or np.any(q[:,1]>=h):errors.append('center_path_outside_original_or_nonfinite')
    for k in POINT_FIELDS:
        if k in candidate and (not isinstance(candidate[k],list) or len(candidate[k])!=n):errors.append('point_field_length:'+k)
    widths=np.asarray(candidate.get('widths',[]),float)
    if widths.shape!=(n,) or np.any(widths<=0):errors.append('invalid_widths')
    if 'photometric' in candidate:
        for k,v in candidate['photometric'].items():
            if not isinstance(v,list) or len(v)!=n:errors.append('photometric_length:'+k)
    for k in ('join_index','transition_start_index'):
        if k in candidate and not 0<=candidate[k]<=n:errors.append('invalid_index:'+k)
    sample=cv2.remap((domain>0).astype(np.float32),q[:,0].astype(np.float32)[None,:],q[:,1].astype(np.float32)[None,:],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).ravel()>.3
    if not np.any(sample):errors.append('candidate_never_entered_significant_search_domain')
    if scored:
        for k in ('score','supported','inside_search'):
            if k not in candidate:errors.append('missing_scored_field:'+k)
    return errors,{'point_count':n,'search_domain_samples':int(sample.sum()),'lane_id':candidate.get('lane_id')}

def g0_record_check(whole):
    errors=[];ledger=whole.get('geometry_pool_ledger');g0=whole.get('g0_candidate')
    diag={'ledger_present':ledger is not None,'g0_present':g0 is not None,'scope':'Checks saved pre-image ledger and hashes; source order checked independently.'}
    if ledger is None:
        # Early no-entry return needs no pool. An executed generator must save one.
        if whole.get('diagnostics',{}).get('lane_generation') is not None:errors.append('missing_pre_RGB_geometry_pool_ledger')
        return errors,diag
    ids=[v['candidate_id'] for v in ledger]
    if len(ids)!=len(set(ids)):errors.append('duplicate_pre_RGB_pool_id')
    if not ledger:
        if g0 is not None:errors.append('G0_present_with_empty_geometry_pool')
        return errors,diag
    if g0 is None:return ['G0_missing_with_nonempty_geometry_pool'],diag
    best=min(ledger,key=lambda c:c['geometry_cost']);item=next((v for v in ledger if v['candidate_id']==g0['candidate_id']),None)
    if item is None:errors.append('G0_not_in_pre_RGB_pool')
    else:
        if g0['candidate_id']!=best['candidate_id']:errors.append('G0_not_minimum_geometry_cost')
        if g0.get('geometry_cost')!=item['geometry_cost']:errors.append('G0_geometry_cost_mismatch')
        if len(g0['xy'])!=item['point_count']:errors.append('G0_pool_point_count_mismatch')
        if canonical_hash(g0['xy'])!=item['xy_sha256']:errors.append('G0_pre_RGB_xy_hash_mismatch')
        if canonical_hash(g0['widths'])!=item['widths_sha256']:errors.append('G0_pre_RGB_width_hash_mismatch')
        if any(k in g0 for k in ('score','image_score','photometric','supported')):errors.append('G0_contains_RGB_scored_fields')
        # Recompute the saved geometry-only cost from its interpretable scalar
        # terms, without running candidate generation or reading image pixels.
        params=whole.get('diagnostics',{}).get('lane_generation',{}).get('parameters',{})
        keys=('join_offset_px','transition_scale_px','join_angle_deg','lane_free_width_ratio','transition_length_width','transition_turn_per_width')
        if not all(k in g0 for k in keys):errors.append('G0_missing_geometry_cost_terms')
        elif g0['transition_scale_px']<=0 or g0['lane_free_width_ratio']<=0:errors.append('G0_invalid_geometry_scale')
        else:
            ratio=g0['lane_free_width_ratio'];weight=params.get('free_width_mismatch_cost_weight',.15);cap=params.get('maximum_free_width_mismatch_cost',1.5)
            reconstructed=abs(g0['join_offset_px'])/g0['transition_scale_px']+math.radians(g0['join_angle_deg'])/.35+weight*min(cap,abs(math.log(max(ratio,1e-6))))+.04*g0['transition_length_width']+.15*g0['transition_turn_per_width']
            diag['geometry_cost_reconstructed']=reconstructed
            if not math.isclose(reconstructed,g0['geometry_cost'],rel_tol=1e-9,abs_tol=1e-9):errors.append('G0_geometry_cost_formula_mismatch')
    chosen=whole.get('g0_selected_candidate_id')
    if chosen is not None and chosen!=g0['candidate_id']:errors.append('G0_selected_id_mismatch')
    diag.update(pool_count=len(ledger),minimum_geometry_candidate_id=best['candidate_id'],g0_candidate_id=g0['candidate_id'])
    return errors,diag

def audit_case(row,run,inventory,manifest):
    rel=Path(row['output_file']);role=rel.parent.name;paths=[run/k/role/rel.name for k in FOLDERS]+[run/'记录'/role/(rel.stem+'.json')]
    answer={'input':row['output_file'],'camera_role':row['camera_role'],'errors':[],'files':[],'input_checks':[]};fail=answer['errors'].append
    try:
        for p in paths:
            relative=p.relative_to(run).as_posix();entry=manifest.get(relative)
            if not p.is_file():raise ValueError('Missing output:'+relative)
            digest=io.sha(p);size=p.stat().st_size;answer['files'].append({'path':relative,'size':size,'sha256':digest})
            if entry is None or entry.get('sha256')!=digest or entry.get('size')!=size:fail('output_manifest_mismatch:'+relative)
        saved=load(paths[-1]);rec=saved['record'];whole=saved['whole'];answer['status']=rec.get('status')
        for k in ('output_file','camera_role','source_group','source_time_s','dataset_use_status'):
            if str(rec.get(k))!=str(row.get(k)):fail('input_metadata_mismatch:'+k)
        expected={normalized(p):inventory.get(normalized(p)) for p in expected_dependencies(row)};original_path=io.DATA/rel;expected[normalized(original_path)]=row['output_sha256']
        saveddeps={normalized(k):v for k,v in rec['input_dependency_sha256'].items()}
        if set(saveddeps)!=set(expected):fail('input_dependency_set_mismatch')
        for name,target in expected.items():
            actual=io.sha(name);ok=target is not None and actual==target and saveddeps.get(name)==target
            answer['input_checks'].append({'path':name,'expected':target,'actual':actual,'ok':ok})
            if not ok:fail('input_dependency_changed:'+name)
        original=io.read_image(original_path);shape=original.shape[:2];masks=[io.read_image(paths[i],cv2.IMREAD_UNCHANGED) for i in (0,2,3)];overlay=io.read_image(paths[1],cv2.IMREAD_UNCHANGED)
        for key,mask in zip(('whole','support','review'),masks):
            if mask.dtype!=np.uint8 or mask.shape!=shape:fail('mask_shape_dtype:'+key)
            elif np.any((mask!=0)&(mask!=255)):fail('mask_not_binary:'+key)
        mask,support,review=masks
        if all(a.shape==shape for a in masks):
            if np.any((support>0)&(mask==0)):fail('support_not_subset_whole')
            if np.any((review>0)&(mask==0)):fail('review_not_subset_whole')
            answer.update(whole_pixels=int(np.count_nonzero(mask)),support_pixels=int(np.count_nonzero(support)),review_pixels=int(np.count_nonzero(review)))
        if overlay.shape!=original.shape or overlay.dtype!=np.uint8:fail('overlay_shape_dtype')
        elif mask.shape==shape:
            allowed=(mask>0).astype(np.uint8);cs,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(allowed,cs,-1,1,1);cv2.rectangle(allowed,(12,12),(1700,112),1,-1)
            changed=np.any(overlay!=original,axis=2);outside=int(np.count_nonzero(changed&(allowed==0)));answer['overlay_background_changes']=outside
            if outside:fail('original_pixels_changed_outside_mask_contour_banner')
        surface=io.read_image(io.C4/'roi/surface_masks'/role/rel.name,0);domain=independent_domain(surface)
        candidates=whole.get('candidates',[]);ids=[c.get('candidate_id') for c in candidates]
        if len(ids)!=len(set(ids)):fail('duplicate_scored_candidate_id')
        selected=whole.get('selected_candidate_id')
        if selected is not None and selected not in ids:fail('selected_not_in_saved_scored_pool')
        details=[]
        for c in candidates:
            errors,detail=candidate_checks(c,shape,domain,True);details.append({'candidate_id':c['candidate_id'],**detail,'errors':errors})
            answer['errors']+=['candidate:'+str(c['candidate_id'])+':'+e for e in errors]
        g0=whole.get('g0_candidate')
        if g0 is not None:
            errors,detail=candidate_checks(g0,shape,domain,False);answer['g0_path_check']={**detail,'errors':errors};answer['errors']+=['G0:'+e for e in errors]
        errors,detail=g0_record_check(whole);answer['g0_pool_check']=detail;answer['errors']+=errors
        if rec.get('status')!=whole.get('status'):fail('record_whole_status_mismatch')
        if whole.get('diagnostics',{}).get('same_material_identity_confirmed') is not False:fail('unexpected_material_identity_claim')
        if whole.get('diagnostics',{}).get('n_minus_one_used') is not False:fail('unexpected_n_minus_one_used')
        nonfinite=finite_paths(saved)
        if nonfinite:fail('nonfinite_record:'+','.join(nonfinite[:6]))
        answer['candidate_checks']=details
    except Exception as exc:fail(str(exc));answer['traceback']=traceback.format_exc()
    answer['ok']=not answer['errors'];return answer

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    if Path(args.run).name!=args.run or args.run in ('.','..'):raise ValueError('Direct-child run required')
    run=io.HERE/args.run;reportpath=run/'independent_structure_audit.json'
    if reportpath.exists():raise ValueError('Preserve existing audit; do not overwrite failed receipts')
    begun=time.time();cv2.setNumThreads(1);errors=[];start=load(run/'run_start.json');summary=load(run/'summary.json');protocol=load(run/'source_snapshot/protocol.json')
    allrows=io.selection('all');allowed={r['output_file']:r for r in allrows};ids=start['input_ids'];mode=summary['mode'];expected=[r['output_file'] for r in io.selection(mode)]
    if len(allrows)!=400 or Counter(r['camera_role'] for r in allrows)!={'left':200,'right':200}:errors.append('dataset_denominator_not_400_left200_right200')
    if ids!=expected or len(ids)!=len(set(ids)):errors.append('run_inputs_not_fixed_ordered_selection')
    if any(k not in allowed for k in ids):raise ValueError('Input outside explicit open400 manifest')
    rows=[allowed[k] for k in ids]
    if summary.get('planned')!=len(rows) or summary.get('completed')!=len(rows) or summary.get('failures') or summary.get('exit_code')!=0:errors.append('run_incomplete_or_failed')
    recorded=[v['output_file'] for v in summary['records']]
    if len(recorded)!=len(set(recorded)) or set(recorded)!=set(ids):errors.append('summary_records_not_exact_denominator')
    sources,baseline,errs=source_checks(run,start,protocol);errors+=errs
    order=g0_source_order(run)
    if not order['passed']:errors.append('G0_frozen_source_order_or_RGB_dependency_check_failed')
    inventory,parentchecks=parent_inventory(protocol);production=load(run/'output_manifest.json');manifest={Path(v['path']).as_posix():v for v in production['files']}
    expect=set()
    for r in rows:
        rel=Path(r['output_file']);role=rel.parent.name;expect.update((Path(k)/role/rel.name).as_posix() for k in FOLDERS);expect.add((Path('记录')/role/(rel.stem+'.json')).as_posix())
    if len(manifest)!=len(production['files']) or set(manifest)!=expect or production['case_count']!=len(rows):errors.append('output_manifest_membership_count_mismatch')
    with ThreadPoolExecutor(max_workers=max(1,min(args.workers,4))) as pool:cases=list(pool.map(lambda r:audit_case(r,run,inventory,manifest),rows))
    statuses=dict(Counter(c.get('status','AUDIT_FILE_FAILURE') for c in cases))
    if statuses!=summary['statuses']:errors.append('summary_status_count_mismatch')
    passed=not errors and all(c['ok'] for c in cases)
    report={'audit_version':'A5-independent-1','run':args.run,'scope':'Saved files, frozen inputs/sources, image arrays, candidate bookkeeping and conditional G0 ledger; no rerun or accuracy claim.','planned_count':len(rows),'checked_count':len(cases),'passed_count':sum(c['ok'] for c in cases),'camera_counts':dict(Counter(r['camera_role'] for r in rows)),'global_errors':errors,'source_checks':sources,'baseline_source_checks':baseline,'parent_manifest_checks':parentchecks,'G0_source_check':order,'statuses':statuses,'nonempty_mask_cases':sum(c.get('whole_pixels',0)>0 for c in cases),'empty_mask_cases':sum(c.get('whole_pixels')==0 for c in cases),'cases':cases,'passed':passed,'accuracy_verified':False,'seconds':time.time()-begun,'audit_source_sha256':io.sha(__file__)}
    io.dump(reportpath,report);io.dump(run/'independent_output_manifest.json',{'case_count':len(cases),'files':[v for c in cases for v in c['files']],'audit_source_sha256':report['audit_source_sha256']})
    snap=run/'audit_tool_snapshot';snap.mkdir();shutil.copy2(__file__,snap/Path(__file__).name);shutil.copy2(io.HERE/'whole_io.py',snap/'whole_io.py')
    io.dump(run/'audit_command_receipt.json',{'command':sys.argv,'interpreter':sys.executable,'exit_code':0 if passed else 1,'audit_sha256':io.sha(reportpath),'note':'Exit code computed by actual audit process; shell outcome separately observed by caller.'})
    print(json.dumps({k:report[k] for k in ('run','passed','checked_count','passed_count','global_errors','statuses','nonempty_mask_cases','empty_mask_cases','seconds')},ensure_ascii=False))
    return 0 if passed else 1
if __name__=='__main__':sys.exit(main())
