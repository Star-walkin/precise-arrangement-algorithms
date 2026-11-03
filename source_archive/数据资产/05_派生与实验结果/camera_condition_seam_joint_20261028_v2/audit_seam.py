"""独立只读F2审计：不调用检测/拟合；从冻结F1/C3/D4读取参考。

逐像素验证过渡区外保持、增删账、接续栅格及保存浮点证据场。
结构通过不是边界准确率或同根物理身份；无接续图仍留在分母。
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from datetime import datetime,timezone
import argparse,ast,json,math
import cv2
import numpy as np
from seam_io import HERE,F1,DATA,selection,inputs,sha,read_json,read_image

MASKS=('transition','merged','inferred','added','removed','replacement','baseline','new_join','simple_bridge','exclusion')


def issue(errors,code,detail=None):errors.append({'code':code,'detail':detail})


def parameters(snapshot):
    tree=ast.parse(snapshot.read_text(encoding='utf8'))
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PARAMETERS' for t in n.targets):return ast.literal_eval(n.value)
    raise ValueError('no literal PARAMETERS')


def parts(row,run):
    rel=Path(row['output_file']);role,name,stem=rel.parent.name,rel.name,rel.stem
    result=[Path('图像结果')/role/name,*[Path('记录/masks')/key/role/name for key in MASKS],
            Path('记录/fields')/role/(stem+'.npz'),Path('记录/cases')/role/(stem+'.json')]
    extra=Path('重叠区增强')/role/name
    if (run/extra).exists():result.append(extra)
    return result


def audit_one(task):
    row,run,op,jp=task;cv2.setNumThreads(1);errors=[]
    result={'output_file':row['output_file'],'errors':errors}
    try:
        rel=Path(row['output_file']);role,name,stem=rel.parent.name,rel.name,rel.stem
        # inputs仅纯IO，重读并逐项核验冻结参考；不接受case自报基线。
        data,deps=inputs(row);shape=data['surface'].shape;surface=data['surface']>0
        reference_case=F1/'all_001/记录/cases'/role/(stem+'.json')
        deps[str(reference_case)]=sha(reference_case)
        case=read_json(run/'记录/cases'/role/(stem+'.json'));rec=case['record']
        if rec.get('status')!='completed':raise ValueError('case not completed')
        if rec.get('input_dependency_sha256')!=deps:issue(errors,'frozen_dependency_hash_or_key_mismatch')
        if rec.get('original_sha256')!=row['output_sha256'] or sha(DATA/rel)!=row['output_sha256']:issue(errors,'original_source_hash')
        masks={key:read_image(run/'记录/masks'/key/role/name,0) for key in MASKS}
        for key,m in masks.items():
            if m.shape!=shape or m.dtype!=np.uint8 or np.any((m!=0)&(m!=255)):raise ValueError('mask shape/binary '+key)
            if np.any(m[~surface]):issue(errors,'mask_outside_surface',key)
        t=masks['transition']>0;bright=surface&(data['bright']>0);dark=surface&(data['dark']>0)
        expected_t=np.zeros(shape,bool)
        if np.any(bright) and np.any(dark):
            db=cv2.distanceTransform((~bright).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            dd=cv2.distanceTransform((~dark).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            span=op['overlap_distance_pitch']*data['pitch'];expected_t=surface&(db<=span)&(dd<=span)
        if not np.array_equal(t,expected_t):issue(errors,'transition_not_frozen_distance_intersection')
        baseline=data['f1_merged']>0;old_inf=data['f1_inferred']>0
        if not np.array_equal(masks['baseline']>0,baseline):issue(errors,'baseline_not_frozen_F1')
        merged=masks['merged']>0;inf=masks['inferred']>0;replace=masks['replacement']>0;new=masks['new_join']>0
        for key,reference in [('merged',data['f1_merged']),('inferred',data['f1_inferred'])]:
            if not np.array_equal(masks[key][~t],reference[~t]):issue(errors,'outside_transition_changed',key)
        image=read_image(run/'图像结果'/role/name)
        if image.shape!=data['f1_image'].shape or image.dtype!=np.uint8:raise ValueError('display shape/dtype')
        if not np.array_equal(image[~t],data['f1_image'][~t]):issue(errors,'display_outside_transition_changed')
        if np.any(image[~surface]):issue(errors,'display_outside_surface')
        for key in ('added','removed','replacement','new_join','simple_bridge','exclusion'):
            if np.any(masks[key][~t]):issue(errors,'changes_not_subset_transition',key)
        if not np.array_equal(masks['added']>0,merged&~baseline):issue(errors,'added_ledger')
        if not np.array_equal(masks['removed']>0,baseline&~merged):issue(errors,'removed_ledger')
        if np.any(replace&~baseline):issue(errors,'replacement_not_subset_baseline')
        if not np.array_equal(merged,(baseline&~replace)|new):issue(errors,'replacement_merge_equation')
        if np.any(inf&~merged):issue(errors,'inferred_not_subset_merged')
        if np.any((data['c3_exclusion']>0)&t&(masks['exclusion']==0)):issue(errors,'lost_frozen_source_exclusion')
        fp=run/'记录/fields'/role/(stem+'.npz')
        with np.load(fp,allow_pickle=False) as z:fields={k:z[k].copy() for k in z.files}
        x0,y0=int(fields['x0']),int(fields['y0']);body=fields['body_barrier'];broad=fields['broad_score'];unknown=fields['unknown_mask']
        if body.dtype!=np.float32 or body.ndim!=2 or not np.all(np.isfinite(body)) or np.any((body<0)|(body>1)):raise ValueError('invalid float body field')
        hh,ww=body.shape
        if x0<0 or y0<0 or x0+ww>shape[1] or y0+hh>shape[0]:raise ValueError('field bounds')
        if any(v.shape!=body.shape for k,v in fields.items() if k not in ('x0','y0')):raise ValueError('field dimensions differ')
        if not np.all(np.isfinite(broad)) or np.any((broad<0)|(broad>1)):raise ValueError('invalid broad field')
        if np.any((unknown!=0)&(unknown!=255)):raise ValueError('nonbinary unknown')
        saved_exclusion=np.zeros(shape,np.uint8);saved_exclusion[y0:y0+hh,x0:x0+ww]=unknown;saved_exclusion[~t]=0
        if not np.array_equal(saved_exclusion,masks['exclusion']):issue(errors,'exclusion_not_saved_guard_unknown')
        yy,xx=np.nonzero(new);lx,ly=xx-x0,yy-y0
        within=(lx>=0)&(lx<ww)&(ly>=0)&(ly<hh)
        if not np.all(within):issue(errors,'new_join_outside_float_field')
        if np.any(masks['exclusion'][yy,xx]):issue(errors,'new_join_crosses_exclusion')
        if np.all(within):
            if np.any(unknown[ly,lx]):issue(errors,'new_join_crosses_guard_unknown')
            if np.any(body[ly,lx]>.65):issue(errors,'new_join_crosses_body_above_065')
        joins=case.get('joins',[]);groups=case.get('new_groups',[]);rejected=case.get('rejections',[])
        gm={g['candidate_id']:g for g in groups}
        if len(gm)!=len(groups):issue(errors,'duplicate_candidate_id')
        if {g['candidate_id'] for g in groups if g['status']=='accepted'}!={g['candidate_id'] for g in joins}:issue(errors,'accepted_group_ledger')
        if {g['candidate_id'] for g in groups if g['status']!='accepted'}!={g['candidate_id'] for g in rejected}:issue(errors,'rejection_group_ledger')
        obs={};raw_counts={}
        for key,source in [('c3','C3'),('d4','D4')]:
            ps=case['observations'][key]['points'];raw_counts[source]=len(ps)
            for p in ps:
                pid=p['id']
                if pid in obs or not pid.startswith(source+'_new_'):issue(errors,'not_unique_new_observation_id',pid)
                obs[pid]=p
                x,y=int(round(p['x'])),int(round(p['y']))
                if not(0<=x<shape[1] and 0<=y<shape[0] and t[y,x]):issue(errors,'new_observation_outside_transition',pid)
        used_endpoints=[];used_obs=[];rebuild=np.zeros(shape,np.uint8);arrays=[]
        for c in joins:
            cid=c['candidate_id']
            if gm.get(cid)!=c or c.get('status')!='accepted':issue(errors,'join_not_exact_accepted_candidate',cid)
            if c.get('same_physical_rope_identity') is not None or c.get('visual_correspondence_only') is not True:issue(errors,'unsupported_identity_claim',cid)
            xy=np.asarray(c['sampled_points_xy'],float)
            if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<2 or not np.all(np.isfinite(xy)):raise ValueError('invalid accepted join xy')
            if np.any(np.abs(np.diff(xy[:,0])-1)>1e-6):issue(errors,'join_x_sampling',cid)
            rr=np.rint(xy).astype(np.int32)
            if np.any((rr[:,0]<0)|(rr[:,0]>=shape[1])|(rr[:,1]<0)|(rr[:,1]>=shape[0])):issue(errors,'join_coordinates_outside_image',cid)
            cv2.polylines(rebuild,[rr],False,255,1,cv2.LINE_8)
            used_endpoints.extend([c['bright_endpoint'],c['dark_endpoint']]);used_obs.extend(c['observation_ids'])
            if any(pid not in obs for pid in c['observation_ids']):issue(errors,'join_unknown_new_observation',cid)
            arrays.append((cid,xy))
        if len(set(used_endpoints))!=len(used_endpoints):issue(errors,'endpoint_shared_between_joins')
        if len(set(used_obs))!=len(used_obs):issue(errors,'new_observation_shared_between_joins')
        accepted=case.get('accepted_points',[])
        if Counter(p['id'] for p in accepted)!=Counter(used_obs):issue(errors,'accepted_point_ledger')
        for p in accepted:
            q=obs.get(p['id'])
            if q is None or (float(p['x']),float(p['y']),p['source'])!=(float(q['x']),float(q['y']),q['source']):issue(errors,'accepted_point_coordinate_changed',p['id'])
        if not np.array_equal(rebuild>0,new):issue(errors,'new_join_not_exact_accepted_raster')
        new_inf=np.zeros(shape,bool)
        if np.all(within):new_inf[yy,xx]=broad[ly,lx]<jp['broad_support_threshold']
        if not np.array_equal(inf,((old_inf&~replace)|new_inf)&merged):issue(errors,'inferred_not_saved_image_support')
        for i,(cid,xy) in enumerate(arrays):
            for qid,q in arrays[i+1:]:
                lo=max(xy[0,0],q[0,0]);hi=min(xy[-1,0],q[-1,0])
                if hi<lo:continue
                xs=np.arange(math.ceil(lo),math.floor(hi)+1);delta=np.interp(xs,xy[:,0],xy[:,1])-np.interp(xs,q[:,0],q[:,1])
                if len(delta) and (np.min(np.abs(delta))<jp['minimum_parallel_separation_pitch']*data['pitch']-1e-6 or np.min(delta)<0<np.max(delta)):issue(errors,'join_order_or_spacing',[cid,qid])
        if not joins and (not np.array_equal(image,data['f1_image']) or np.any(masks['added']) or np.any(masks['removed'])):issue(errors,'no_join_case_not_preserved')
        if rec['join_diagnostics']['accepted_joins']!=len(joins) or rec['new_observation_counts']!=raw_counts:issue(errors,'record_counts')
        for key,value in rec['pixels'].items():
            if value!=int(np.count_nonzero(masks[key])):issue(errors,'reported_pixel_count',key)
        outputs=parts(row,run)
        for detector in ('c3','d4'):
            for key,value in case['observations'][detector]['fit'].items():
                if not isinstance(value,dict) or 'png' not in value:continue
                expected=Path('记录/detector_masks')/detector/key/role/name
                if value['png']!=expected.as_posix():issue(errors,'detector_mask_reference_path',[detector,key]);continue
                path=run/expected
                if sha(path)!=value.get('sha256'):issue(errors,'detector_mask_reference_hash',[detector,key])
                dm=read_image(path,0)
                if dm.shape!=shape or dm.dtype!=np.uint8 or np.any((dm!=0)&(dm!=255)):issue(errors,'detector_mask_shape_binary',[detector,key])
                elif np.any(dm[~t]):issue(errors,'detector_mask_outside_transition',[detector,key])
                outputs.append(expected)
        files=[{'path':p.as_posix(),'size':(run/p).stat().st_size,'sha256':sha(run/p)} for p in outputs]
        result.update({'joins':len(joins),'rejections':len(rejected),'new_observations':raw_counts,'added_pixels':int(np.count_nonzero(masks['added'])),
                       'removed_pixels':int(np.count_nonzero(masks['removed'])),'dependency_count':len(deps),'files':files})
    except Exception as exc:issue(errors,'audit_exception',repr(exc))
    result['status']='failed' if errors else 'passed';return result


def audit_run(run,workers):
    errors=[];context=read_json(run/'run_context.json');summary=read_json(run/'summary.json');mode=context['mode'];rows=selection(mode)
    if mode not in ('pilot','all') or len(rows)!=(12 if mode=='pilot' else 400):issue(errors,'fixed_denominator')
    if context['selection']!=[r['output_file'] for r in rows]:issue(errors,'fixed_selection')
    if any(summary.get(k)!=v for k,v in {'planned':len(rows),'completed':len(rows),'failed':0,'exit_code':0,'source_unchanged':True}.items()):issue(errors,'unsuccessful_run_summary')
    snapshots=context['source_snapshot_paths'];source=context['source_sha256']
    if set(snapshots)!=set(source):issue(errors,'snapshot_keys')
    for original,h in source.items():
        part=Path(snapshots[original])
        if part.is_absolute() or '..' in part.parts:raise ValueError('unsafe snapshot')
        if sha(run/part)!=h:issue(errors,'snapshot_hash',original)
    protocol=read_json(run/snapshots[str(HERE/'protocol.json')])
    for p,h in protocol['baseline_source_sha256'].items():
        if sha(p)!=h or source.get(p)!=h:issue(errors,'frozen_baseline_hash',p)
    if sha(DATA/'manifest.csv')!=protocol['input_manifest_sha256']:issue(errors,'original_manifest_hash')
    if sha(F1/'all_001/output_manifest.json')!=protocol['f1_output_manifest_sha256']:issue(errors,'F1_manifest_hash')
    if mode=='all' and source!=read_json(HERE/'freeze_all.json')['source_sha256']:issue(errors,'full_run_freeze_mismatch')
    op=parameters(run/snapshots[str(HERE/'seam_observation.py')]);jp=parameters(run/snapshots[str(HERE/'seam_join.py')])
    logs=[json.loads(s) for s in (run/'run_log.jsonl').read_text(encoding='utf8').splitlines()]
    if len(logs)!=len(rows) or any(a.get('output_file')!=b['output_file'] or a.get('status')!='completed' for a,b in zip(logs,rows)):issue(errors,'run_log_denominator_or_order')
    cases=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,c in enumerate(pool.map(audit_one,[(r,run,op,jp) for r in rows]),1):
            cases.append(c)
            if c['errors']:print(json.dumps({'audit_failure':c['output_file'],'errors':c['errors']},ensure_ascii=False),flush=True)
            if i%25==0 or i==len(rows):print(json.dumps({'audited':i,'planned':len(rows),'failed':sum(c['status']=='failed' for c in cases)}),flush=True)
    return {'name':'F2 independent frozen-reference pixel/ledger audit','mode':mode,'run':str(run),'workers':workers,
        'finished_utc':datetime.now(timezone.utc).isoformat(),'auditor_sha256':sha(Path(__file__)),
        'planned':len(rows),'passed':sum(c['status']=='passed' for c in cases),'failed':sum(c['status']=='failed' for c in cases),'global_errors':errors,
        'accepted_joins':sum(c.get('joins',0) for c in cases),'rejections':sum(c.get('rejections',0) for c in cases),
        'images_without_joins':sum(c.get('joins')==0 for c in cases),'cases':cases,'accuracy':None,
        'meaning':'Independent structural/pixel consistency with frozen references; saved field evidence does not certify true grooves or same physical rope.',
        'no_detection_or_fit_rerun':True,'equipment_access':False,'reserved_access':False}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--workers',type=int,default=3);ap.add_argument('--output',type=Path);args=ap.parse_args()
    run=args.run.absolute() if args.run.is_absolute() else (HERE/args.run).absolute();out=args.output.absolute() if args.output else run/'audit.json'
    if run.parent!=HERE or out.parent!=run or out.exists():raise ValueError('existing directchild run, new directchild audit required')
    report=audit_run(run,args.workers)
    if not report['failed'] and not report['global_errors']:
        manifest={'version':'RB-F2.0.0','run':str(run),'case_count':len(report['cases']),'file_count':sum(len(c['files']) for c in report['cases']),
            'cases':[{'output_file':c['output_file'],'joins':c['joins'],'files':c['files']} for c in report['cases']],
            'scope':'Immutable prediction masks/image/float fields/record and saved pilot normalization; excludes later QA and audit.'}
        p=run/'output_manifest.json'
        if p.exists():
            if read_json(p)!=manifest:issue(report['global_errors'],'existing_manifest_differs')
        else:
            with p.open('x',encoding='utf8') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
        report['output_manifest_sha256']=sha(p)
    with out.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ['planned','passed','failed','global_errors','accepted_joins','images_without_joins']},ensure_ascii=False),flush=True)
    return int(bool(report['failed'] or report['global_errors']))


if __name__=='__main__':raise SystemExit(main())
