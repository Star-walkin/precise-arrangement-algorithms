"""Independent A6 saved-output audit: no detector/scorer imports or reruns.

Checks bytes, masks, candidate bookkeeping and image-first/late-F3 evidence.
Passing does not establish material identity, true contact or visual accuracy.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,math,shutil,sys,time,traceback
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import cv2,numpy as np
import whole_io as io

CORE=('whole_io.py','free_rope.py','photometric_ribbon.py','attached_groove.py','appearance_template.py','free_seed_recovery.py','axial_reacquisition.py','image_beam.py','pipeline.py','run_experiment.py','protocol.json')
FOLDERS=('01_可信前缀掩膜','02_原图叠加','03_待确认候选并集','04_最佳外观候选掩膜')
BRANCH_FOLDER='05_多候选分支'
POINT_FIELDS=('search_supported','search_direct_supported','template_score','groove_score','neighbor_advantage','supported','direct_template_supported','inside_search','angle_from_free_deg','original_groove_score')

def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def norm(path):return str(Path(path).resolve())
def finite_paths(value,path='$'):
    if isinstance(value,float) and not math.isfinite(value):return [path]
    if isinstance(value,dict):return [q for k,v in value.items() for q in finite_paths(v,path+'.'+k)]
    if isinstance(value,list):return [q for i,v in enumerate(value) for q in finite_paths(v,path+'['+str(i)+']')]
    return []

def independent_domain(surface):
    binary=np.where(surface>0,255,0).astype(np.uint8);cs,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);n=int(np.count_nonzero(binary));cs=[c for c in cs if cv2.contourArea(c)>=max(50.,.0001*n)];out=np.zeros_like(binary)
    if not cs:return out
    cv2.fillConvexPoly(out,cv2.convexHull(np.concatenate(cs)),255)
    if np.count_nonzero(out)>1.8*max(1,n):out[:]=0;cv2.drawContours(out,cs,-1,255,cv2.FILLED)
    return out

def independent_paint(shape,xy,widths):
    out=np.zeros(shape,np.uint8);xy=np.asarray(xy,float);widths=np.asarray(widths,float)
    for i in range(1,len(xy)):
        diff=xy[i]-xy[i-1];normal=np.array([-diff[1],diff[0]])/max(float(np.linalg.norm(diff)),1e-8)
        corners=[xy[i-1]+normal*widths[i-1]/2,xy[i]+normal*widths[i]/2,xy[i]-normal*widths[i]/2,xy[i-1]-normal*widths[i-1]/2]
        cv2.fillConvexPoly(out,np.rint(corners).astype(np.int32),255)
    return out

def parent_inventory(protocol):
    inventory={};checks=[]
    for name,expected in protocol['parent_manifest_sha256'].items():
        p=Path(name);actual=io.sha(p);checks.append({'path':name,'expected':expected,'actual':actual,'ok':actual==expected})
        if actual!=expected:raise ValueError('Parent manifest changed: '+name)
        j=load(p);entries=[f for c in j['cases'] for f in c['files']] if 'cases' in j else j.get('files',j.get('file_manifest',[]))
        for entry in entries:
            q=entry.get('absolute_path')
            if q is None:
                q=Path(entry.get('path',entry.get('relative_path')))
                if not q.is_absolute():q=p.parent/q
            inventory[norm(q)]=entry['sha256']
    return inventory,checks

def expected_dependencies(row):
    rel=Path(row['output_file']);role=rel.parent.name;stem=rel.stem
    return [io.C4/'roi/surface_masks'/role/rel.name,io.A1/'records'/role/(stem+'.json'),io.A2/'records'/role/(stem+'.json'),io.C3/'boundaries/records'/role/(stem+'.json'),io.F3/'记录/cases'/role/(stem+'.json')]+[io.F3/'记录/masks'/key/role/rel.name for key in ('merged','inferred','exclusion')]

def source_checks(run,start,protocol):
    errors=[];checks=[];frozen=start['sources_sha256']
    for name in CORE:
        if str(io.HERE/name) not in frozen:errors.append('missing_core_source_hash:'+name)
    external=io.ASSETS/'camera_condition_incoming_rope_20261026_v2/free_refinement.py'
    if str(external) not in frozen:errors.append('missing_external_A2_source')
    if len({Path(k).name for k in frozen})!=len(frozen):errors.append('flat_source_snapshot_collision')
    for name,expected in frozen.items():
        source=Path(name);snapshot=run/'source_snapshot'/source.name;a=io.sha(source) if source.is_file() else None;b=io.sha(snapshot) if snapshot.is_file() else None
        checks.append({'source':name,'expected':expected,'current':a,'snapshot':b,'ok':a==b==expected})
        if a!=expected:errors.append('current_source_changed:'+name)
        if b!=expected:errors.append('snapshot_source_changed:'+name)
    if (run/'freeze_snapshot.json').is_file() and load(run/'freeze_snapshot.json').get('sources_sha256')!=frozen:errors.append('freeze_vs_executed_sources_mismatch')
    baseline=dict(protocol['source_parent_sha256']);f3path=io.F3.parent/'freeze_all.json'
    if io.sha(f3path)!=protocol['F3_sources_freeze_sha256']:errors.append('F3_freeze_changed')
    baseline.update(load(f3path)['source_sha256']);baseline[str(Path(protocol['A5_output_root'])/'output_manifest.json')]=protocol['A5_output_manifest_sha256'];old=[]
    for name,expected in baseline.items():
        actual=io.sha(name) if Path(name).is_file() else None;old.append({'path':name,'expected':expected,'actual':actual,'ok':actual==expected})
        if actual!=expected:errors.append('old_baseline_changed:'+name)
    return checks,old,errors

def image_first_source_check(run):
    tree=ast.parse((run/'source_snapshot/image_beam.py').read_text(encoding='utf-8'));fn=next((v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name=='generate_image_candidates'),None)
    if fn is None:return {'passed':False,'reason':'image_generator_not_found'}
    helper=ast.parse((run/'source_snapshot/axial_reacquisition.py').read_text(encoding='utf-8'));probe=next((v for v in helper.body if isinstance(v,ast.FunctionDef) and v.name=='probe_axis_bridge'),None)
    arguments=[v.arg for v in fn.args.args];helper_arguments=[v.arg for v in probe.args.args] if probe else [];forbidden=[]
    for filename,module in (('image_beam.py',tree),('axial_reacquisition.py',helper)):
        for node in ast.walk(module):
            if isinstance(node,ast.Subscript) and isinstance(node.slice,ast.Constant) and node.slice.value in ('merged','curve_refs','f3_record','c3_record','inferred','exclusion'):forbidden.append({'source':filename,'line':node.lineno,'key':node.slice.value})
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='get' and node.args and isinstance(node.args[0],ast.Constant) and node.args[0].value in ('merged','curve_refs','f3_record','c3_record','inferred','exclusion'):forbidden.append({'source':filename,'line':node.lineno,'key':node.args[0].value})
    pipeline=ast.parse((run/'source_snapshot/pipeline.py').read_text(encoding='utf-8'));calls={}
    for node in ast.walk(pipeline):
        if isinstance(node,ast.Call):
            name=node.func.id if isinstance(node.func,ast.Name) else (node.func.attr if isinstance(node.func,ast.Attribute) else '')
            calls.setdefault(name,[]).append(node.lineno)
    full_pool_contract=any(isinstance(v,ast.Constant) and v.value=='candidate_pool' for v in ast.walk(pipeline))
    return {'passed':arguments==['original','surface','pitch','free','template'] and helper_arguments==['fields','domain','free','template'] and not forbidden,'image_generator_arguments':arguments,'axial_helper_arguments':helper_arguments,'AST_sources_checked':['image_beam.py','axial_reacquisition.py'],'full_pool_contract_declared':full_pool_contract,'forbidden_F3_input_reads':forbidden,'pipeline_generation_call_lines':calls.get('generate_image_candidates',[]),'pipeline_late_check_call_lines':{k:v for k,v in calls.items() if 'late' in k.lower()},'scope':'Static allowed-input boundary plus saved per-candidate generation flags; no physical-identity claim.'}

def candidate_checks(c,shape,domain,result,entries,ledger,full_pool=False):
    errors=[];xy=np.asarray(c.get('xy',[]),float);n=len(xy);h,w=shape
    if xy.ndim!=2 or xy.shape[1]!=2 or n<2:return ['invalid_xy_shape'],{}
    if not np.isfinite(xy).all() or np.any(xy[:,0]<0) or np.any(xy[:,0]>=w) or np.any(xy[:,1]<0) or np.any(xy[:,1]>=h):errors.append('path_nonfinite_or_outside_original')
    ww=np.asarray(c.get('widths',[]),float)
    if ww.shape!=(n,) or not np.isfinite(ww).all() or np.any(ww<=0):errors.append('invalid_widths')
    search_value=c.get('xy',[]) if full_pool else c.get('search_xy',[])
    search=np.asarray(search_value,float);sw=np.asarray(c.get('widths',[]) if full_pool else c.get('search_widths',[]),float);offset=0 if full_pool else c.get('search_prefix_offset');sn=len(search)
    if search.ndim!=2 or search.shape[1]!=2 or sn<2 or sw.shape!=(sn,) or not isinstance(offset,int) or offset<0 or offset+sn!=n:errors.append('search_geometry_or_prefix_shape')
    elif not np.array_equal(xy[offset:],search) or not np.array_equal(ww[offset:],sw):errors.append('search_geometry_not_exact_suffix')
    for key in (('supported','direct_template_supported') if full_pool else POINT_FIELDS):
        if not isinstance(c.get(key),list) or len(c[key])!=sn:errors.append('search_point_field_length:'+key)
    for key,value in c.get('evidence',{}).items():
        if not isinstance(value,list) or len(value)!=sn:errors.append('evidence_field_length:'+key)
    prefix=None if full_pool else c.get('trusted_prefix_count')
    if not full_pool:
        if not isinstance(prefix,int) or not 0<=prefix<=n:errors.append('invalid_trusted_prefix_count')
        elif c['candidate_id']!=result.get('selected_candidate_id') and prefix!=0:errors.append('nonselected_candidate_claims_trusted_prefix')
    if c.get('F3_used_for_generation') is not False:errors.append('missing_or_true_F3_generation_flag')
    if c.get('same_material_identity') not in (None,False) or (not full_pool and c.get('same_material_identity_confirmed') is not False) or (full_pool and c.get('same_material_identity_confirmed') not in (None,False)):errors.append('unexpected_identity_claim')
    entered=cv2.remap((domain>0).astype(np.float32),xy[:,0].astype(np.float32)[None,:],xy[:,1].astype(np.float32)[None,:],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).ravel()>.3
    if not np.any(entered):errors.append('candidate_never_entered_significant_surface')
    item=ledger.get(c.get('candidate_id'));entry=entries.get(c.get('entry_id'));late=c.get('late_F3',{})
    if item is None:errors.append('candidate_not_in_saved_image_first_pool')
    else:
        digest=hashlib.sha256(json.dumps(search_value,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if item.get('point_count')!=sn or item.get('xy_sha256')!=digest or item.get('entry_id')!=c.get('entry_id'):errors.append('late_check_changed_pre_F3_geometry')
    if full_pool:
        if entry is None:errors.append('pool_entry_not_saved')
        else:
            d=np.asarray(entry['free']['geometry']['direction_xy'],float)
            if np.any(np.diff(search@d)<=0):errors.append('pool_path_not_monotone_along_free_axis')
        return errors,{'points':n,'search_points':sn,'significant_surface_samples':int(entered.sum()),'full_pool_record':True,'late_check_scope':'Saved scalar scores and support arrays; detailed per-point late_F3 fields audited on output branches.'}
    for key in ('checked','edge_distance_px','alignment','body_crossing_suspect'):
        if not isinstance(late.get(key),list) or len(late[key])!=sn:errors.append('late_F3_field_length:'+key)
    checked_count=0
    if not errors and entry is not None:
        params=result['diagnostics']['parameters'];width0=float(entry['free']['geometry']['width_px']);angle=np.asarray(c['angle_from_free_deg']);inside=np.asarray(c['inside_search'],bool);evidence=np.asarray(c['supported'],bool)
        continuity=np.r_[True,abs(np.diff(sw))<=.10*np.maximum(sw[:-1],1.)];window=max(3,int(params['late_evidence_window_widths']*width0/8.));window=max(1,min(window,sn));fraction=np.convolve(evidence.astype(float),np.ones(window)/window,mode='same')
        expected=(angle>params['late_minimum_bend_deg'])&inside&continuity&(fraction>=params['late_minimum_image_fraction']);checked=np.asarray(late['checked'],bool);distance=np.asarray(late['edge_distance_px'],float);alignment=np.asarray(late['alignment'],float);cross=np.asarray(late['body_crossing_suspect'],bool)
        if not np.array_equal(expected,checked):errors.append('late_F3_checked_before_sustained_bend_evidence')
        if distance.shape!=(sn,2):errors.append('late_F3_edge_distance_shape')
        elif np.any(distance[~checked]!=-1) or np.any(distance[checked]<0):errors.append('late_F3_distance_sentinel_or_validity')
        if np.any(alignment[~checked]!=0) or np.any(cross&~checked):errors.append('late_F3_effect_outside_eligible_gate')
        checked_count=int(checked.sum())
        direct=np.asarray(c['direct_template_supported'],bool);groove=np.asarray(c['original_groove_score'],float)
        if np.any(cross&((alignment>=.2)|(groove>=.15)|direct)):errors.append('late_body_crossing_without_original_image_conditions')
    elif entry is None:errors.append('candidate_entry_not_saved')
    return errors,{'points':n,'search_points':sn,'significant_surface_samples':int(entered.sum()),'late_F3_checked_points':checked_count,'trusted_prefix_count':prefix,'full_pool_record':full_pool}

def audit_case(row,run,inventory,manifest,required_pool=False):
    rel=Path(row['output_file']);role=rel.parent.name;stem=rel.stem;answer={'input':row['output_file'],'camera_role':row['camera_role'],'errors':[],'files':[],'input_checks':[]};fail=answer['errors'].append
    try:
        recordpath=run/'记录'/role/(stem+'.json');saved=load(recordpath);record=saved['record'];result=saved['result'];answer['status']=record.get('status');mapping=record['candidate_file_map'];candidates=result.get('candidates',[])
        if len(candidates)>6 or len(mapping)!=len(candidates):fail('branch_count_or_file_mapping_mismatch')
        paths=[run/k/role/rel.name for k in FOLDERS]+[recordpath]
        for i,entry in enumerate(mapping):
            expected=(Path(BRANCH_FOLDER)/role/stem/('candidate_'+str(i).zfill(2)+'.png'))
            if Path(entry['path'])!=expected or entry['branch_index']!=i or entry['candidate_id']!=candidates[i].get('candidate_id'):fail('unsafe_or_wrong_branch_file_mapping')
            paths.append(run/expected)
        for path in paths:
            key=path.relative_to(run).as_posix();digest=io.sha(path);size=path.stat().st_size;answer['files'].append({'path':key,'size':size,'sha256':digest});entry=manifest.get(key)
            if entry is None or entry.get('sha256')!=digest or entry.get('size')!=size:fail('output_manifest_mismatch:'+key)
        for key in ('output_file','camera_role','source_group','source_time_s','dataset_use_status'):
            if str(record.get(key))!=str(row.get(key)):fail('input_metadata_mismatch:'+key)
        expected={norm(p):inventory.get(norm(p)) for p in expected_dependencies(row)};original_path=io.DATA/rel;expected[norm(original_path)]=row['output_sha256'];saveddeps={norm(k):v for k,v in record['input_dependency_sha256'].items()}
        if set(saveddeps)!=set(expected):fail('input_dependency_membership_mismatch')
        for name,digest in expected.items():
            actual=io.sha(name);ok=digest is not None and actual==digest and saveddeps.get(name)==digest;answer['input_checks'].append({'path':name,'expected':digest,'actual':actual,'ok':ok})
            if not ok:fail('input_dependency_changed:'+name)
        original=io.read_image(original_path);shape=original.shape[:2];trusted,review,best=[io.read_image(paths[i],cv2.IMREAD_UNCHANGED) for i in (0,2,3)];overlay=io.read_image(paths[1],cv2.IMREAD_UNCHANGED);branchmasks=[io.read_image(p,cv2.IMREAD_UNCHANGED) for p in paths[5:]]
        masks=[trusted,review,best]+branchmasks
        for name,mask in zip(['trusted','review','best']+['branch_'+str(i) for i in range(len(branchmasks))],masks):
            if mask.dtype!=np.uint8 or mask.shape!=shape or np.any((mask!=0)&(mask!=255)):fail('mask_not_original_uint8_binary:'+name)
        if not all(v.shape==shape for v in masks):raise ValueError('Mask shape mismatch prevents union comparison')
        union=np.zeros(shape,np.uint8)
        for mask in branchmasks:union|=mask
        if np.any((trusted>0)&(best==0)):fail('trusted_not_subset_best')
        if not np.array_equal(review,np.where(trusted>0,0,union).astype(np.uint8)):fail('review_not_branch_union_minus_trusted')
        if np.any((best>0)&((union|trusted)==0)):fail('best_outside_branch_union_and_trusted')
        answer.update(trusted_pixels=int(np.count_nonzero(trusted)),review_pixels=int(np.count_nonzero(review)),best_pixels=int(np.count_nonzero(best)),branch_union_pixels=int(np.count_nonzero(union)),branch_count=len(branchmasks))
        allowed=(trusted>0).astype(np.uint8);cs,_=cv2.findContours(trusted,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(allowed,cs,-1,1,1)
        for c,mask in zip(candidates,branchmasks):
            cs,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(allowed,cs,-1,1,1);cv2.polylines(allowed,[np.rint(c['xy']).astype(np.int32)],False,1,1)
        cv2.rectangle(allowed,(12,12),(1810,112),1,-1)
        if overlay.shape!=original.shape or overlay.dtype!=np.uint8:fail('overlay_shape_dtype')
        else:
            outside=int(np.count_nonzero(np.any(overlay!=original,axis=2)&(allowed==0)));answer['overlay_background_changed_pixels']=outside
            if outside:fail('original_background_changed_outside_declared_rendering')
        surface=io.read_image(io.C4/'roi/surface_masks'/role/rel.name,0);domain=independent_domain(surface);ids=[c.get('candidate_id') for c in candidates]
        if len(ids)!=len(set(ids)):fail('duplicate_candidate_id')
        entries={e['entry_id']:e for e in result.get('entries',[])};ledger={v['candidate_id']:v for v in result.get('image_pool_ledger',[])}
        if len(entries)!=len(result.get('entries',[])):fail('duplicate_entry_id')
        if len(ledger)!=len(result.get('image_pool_ledger',[])):fail('duplicate_pre_F3_pool_id')
        for entry in entries.values():
            if entry.get('search',{}).get('F3_used_during_generation') is not False:fail('entry_missing_or_true_generation_F3_flag')
        if result.get('diagnostics',{}).get('F3_used_during_candidate_generation') is not False:fail('result_missing_or_true_generation_F3_flag')
        if result.get('diagnostics',{}).get('same_material_identity_confirmed') is not False:fail('unexpected_result_identity_claim')
        full=result.get('candidate_pool')
        if required_pool and candidates and full is None:fail('frozen_full_pool_contract_missing_candidate_pool')
        if full is not None:
            fullmap={c['candidate_id']:c for c in full}
            if len(fullmap)!=len(full) or set(fullmap)!=set(ledger):fail('complete_late_pool_does_not_match_pre_F3_ledger')
            fullchecks=[]
            for c in full:
                errors,detail=candidate_checks(c,shape,domain,result,entries,ledger,full_pool=True);fullchecks.append({'candidate_id':c['candidate_id'],**detail,'errors':errors});answer['errors']+=['full_pool:'+c['candidate_id']+':'+e for e in errors]
            answer['full_pool_candidate_checks']=fullchecks;answer['full_pool_candidate_count']=len(full)
            for c in candidates:
                prior=fullmap.get(c['candidate_id'])
                if prior is None:fail('output_branch_absent_from_complete_pool');continue
                if c['search_xy']!=prior['xy'] or c['search_widths']!=prior['widths']:fail('output_branch_changed_complete_pool_geometry')
                for key in ('supported','direct_template_supported','entry_id','image_only_score','late_combined_score','length_px','turn_anchor_bin','F3_used_for_generation'):
                    if c.get(key)!=prior.get(key):fail('output_branch_changed_complete_pool_evidence:'+key)
            if candidates:
                selected=result.get('selected_candidate_id');main=fullmap.get(selected)
                if main is not None:
                    width0=entries[main['entry_id']]['free']['geometry']['width_px'];minimum=result['diagnostics']['parameters']['minimum_branch_span_widths']*width0
                    expected_competitors=[c['candidate_id'] for c in full if c['entry_id']==main['entry_id'] and c['length_px']>=minimum]
                    if result['diagnostics'].get('prefix_competitor_ids')!=expected_competitors:fail('prefix_competitors_not_entire_eligible_same_entry_pool')
        elif result.get('diagnostics',{}).get('prefix_competition_scope') is not None:fail('all_pool_prefix_claim_missing_candidate_pool_evidence')
        checks=[]
        for c,mask in zip(candidates,branchmasks):
            errors,detail=candidate_checks(c,shape,domain,result,entries,ledger)
            if not errors and not np.array_equal(mask,independent_paint(shape,c['xy'],c['widths'])):errors.append('branch_mask_not_saved_path_ribbon')
            checks.append({'candidate_id':c.get('candidate_id'),**detail,'errors':errors});answer['errors']+=['candidate:'+str(c.get('candidate_id'))+':'+e for e in errors]
        answer['candidate_checks']=checks
        if candidates:
            selected=result.get('selected_candidate_id')
            if selected not in ids:fail('selected_not_in_saved_branches')
            elif not np.array_equal(best,branchmasks[ids.index(selected)]):fail('best_mask_not_selected_branch')
            chosen=candidates[ids.index(selected)] if selected in ids else None
            if chosen is not None:
                expected_prefix=chosen['search_prefix_offset']+result['diagnostics']['trusted_search_point_count'] if result['diagnostics']['trusted_search_point_count'] else 0
                if chosen['trusted_prefix_count']!=expected_prefix:fail('trusted_prefix_count_vs_search_count_mismatch')
                if chosen['trusted_prefix_count']==0 and np.any(trusted):fail('zero_trusted_prefix_has_nonempty_trusted_mask')
        if result.get('status')!=record.get('status'):fail('record_result_status_mismatch')
        nonfinite=finite_paths(saved)
        if nonfinite:fail('nonfinite_record:'+','.join(nonfinite[:6]))
    except Exception as exc:fail(str(exc));answer['traceback']=traceback.format_exc()
    answer['ok']=not answer['errors'];return answer

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--report-suffix',default='');args=p.parse_args()
    if Path(args.run).name!=args.run or args.run in ('.','..') or any(c not in '_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' for c in args.report_suffix):raise ValueError('Unsafe run or report suffix')
    run=io.HERE/args.run;suffix=args.report_suffix;reportpath=run/('independent_structure_audit'+suffix+'.json')
    if reportpath.exists():raise ValueError('Preserve old audit; use a new suffix for a justified repair')
    started=time.time();cv2.setNumThreads(1);errors=[];start=load(run/'run_start.json');summary=load(run/'summary.json');protocol=load(run/'source_snapshot/protocol.json');allrows=io.selection('all');allowed={r['output_file']:r for r in allrows};ids=start['input_ids'];mode=summary['mode'];expected=[r['output_file'] for r in io.selection(mode)]
    if len(allrows)!=400 or Counter(r['camera_role'] for r in allrows)!={'left':200,'right':200}:errors.append('denominator_not_400_left200_right200')
    if ids!=expected or len(ids)!=len(set(ids)):errors.append('run_not_fixed_ordered_selection')
    if any(i not in allowed for i in ids):raise ValueError('Input outside explicit open400')
    rows=[allowed[i] for i in ids]
    if summary.get('planned')!=len(rows) or summary.get('completed')!=len(rows) or summary.get('failures') or summary.get('exit_code')!=0:errors.append('run_incomplete_or_failed')
    recs=[v['output_file'] for v in summary['records']]
    if len(recs)!=len(set(recs)) or set(recs)!=set(ids):errors.append('summary_records_not_exact_denominator')
    sources,baseline,e=source_checks(run,start,protocol);errors+=e;generation=image_first_source_check(run)
    if not generation['passed']:errors.append('image_first_source_contract_failed')
    inventory,parentchecks=parent_inventory(protocol);production=load(run/'output_manifest.json');manifest={Path(v['path']).as_posix():v for v in production['files']}
    if len(manifest)!=len(production['files']) or production['case_count']!=len(rows):errors.append('output_manifest_count_or_duplicates')
    cases=[]
    with ThreadPoolExecutor(max_workers=max(1,min(4,args.workers))) as pool:
        for case in pool.map(lambda row:audit_case(row,run,inventory,manifest,generation.get('full_pool_contract_declared',False)),rows):
            cases.append(case)
            if len(cases)%25==0:print('Audit '+str(len(cases))+'/'+str(len(rows)),flush=True)
    actual={v['path'] for c in cases for v in c['files']}
    if actual!=set(manifest):errors.append('output_manifest_not_exact_generated_files')
    statuses=dict(Counter(c.get('status','AUDIT_FILE_FAILURE') for c in cases))
    if statuses!=summary['statuses']:errors.append('summary_status_count_mismatch')
    passed=not errors and all(c['ok'] for c in cases);report={'audit_version':'A6-independent-1','run':args.run,'scope':'Saved files, fixed open inputs, frozen source and image-first evidence bookkeeping only; not visual or physical identity accuracy.','checked_count':len(cases),'passed_count':sum(c['ok'] for c in cases),'camera_counts':dict(Counter(r['camera_role'] for r in rows)),'global_errors':errors,'source_checks':sources,'baseline_source_checks':baseline,'parent_manifest_checks':parentchecks,'image_first_source_check':generation,'statuses':statuses,'nonempty_trusted_cases':sum(c.get('trusted_pixels',0)>0 for c in cases),'nonempty_best_cases':sum(c.get('best_pixels',0)>0 for c in cases),'candidate_count':sum(c.get('branch_count',0) for c in cases),'cases':cases,'passed':passed,'accuracy_verified':False,'seconds':time.time()-started,'audit_source_sha256':io.sha(__file__)}
    io.dump(reportpath,report);io.dump(run/('independent_output_manifest'+suffix+'.json'),{'case_count':len(cases),'files':[v for c in cases for v in c['files']],'audit_source_sha256':report['audit_source_sha256']});snap=run/('audit_tool_snapshot'+suffix);snap.mkdir();shutil.copy2(__file__,snap/Path(__file__).name);shutil.copy2(io.HERE/'whole_io.py',snap/'whole_io.py')
    io.dump(run/('audit_command_receipt'+suffix+'.json'),{'command':sys.argv,'interpreter':sys.executable,'exit_code':0 if passed else 1,'audit_sha256':io.sha(reportpath),'note':'Computed process exit; caller records the observed shell exit separately.'});print(json.dumps({k:report[k] for k in ('run','passed','checked_count','passed_count','global_errors','statuses','nonempty_trusted_cases','nonempty_best_cases','candidate_count','seconds')},ensure_ascii=False));return 0 if passed else 1
if __name__=='__main__':sys.exit(main())
