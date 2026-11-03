"""Independent saved-output audit; never imports/reruns detection or fitting.

Checks source provenance, full-surface observation mapping, F2 preservation,
new raster evidence and accepted-point membership. This is not a truth score.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from datetime import datetime, timezone
import argparse, ast, json, math
import cv2
import numpy as np
from surface_io import HERE, F2, DATA, selection, inputs, sha, read_json, read_image

MASKS=('merged','inferred','new','removed','transition','full_d4','full_d4_inferred','full_d4_observed','exclusion')


def issue(errors, code, detail=None):
    if len(errors)<100: errors.append({'code':code,'detail':detail})


def params(path):
    for node in ast.parse(path.read_text(encoding='utf8')).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PARAMETERS' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('No literal PARAMETERS in '+str(path))


def runs(mask):
    z=np.r_[False,np.asarray(mask,bool),False].astype(np.int8)
    return list(zip(np.flatnonzero(np.diff(z)==1),np.flatnonzero(np.diff(z)==-1)))


def sample(a,xy,x0=0,y0=0,outside=0):
    q=np.rint(xy).astype(np.int32);x=q[:,0]-x0;y=q[:,1]-y0
    inside=(x>=0)&(x<a.shape[1])&(y>=0)&(y<a.shape[0])
    out=np.full(len(q),outside,dtype=a.dtype);out[inside]=a[y[inside],x[inside]]
    return out


def field(f,key,xy,outside=0):return sample(f[key],xy,int(f['x0']),int(f['y0']),outside)


def raster(xy,shape):
    rr=np.rint(xy).astype(np.int32)
    if len(rr)<2 or np.any(rr[:,0]<0) or np.any(rr[:,0]>=shape[1]) or np.any(rr[:,1]<0) or np.any(rr[:,1]>=shape[0]):
        raise ValueError('Invalid curve raster coordinates')
    low=rr.min(axis=0);high=rr.max(axis=0)
    patch=np.zeros((high[1]-low[1]+1,high[0]-low[0]+1),np.uint8)
    cv2.polylines(patch,[rr-low],False,255,1,cv2.LINE_8)
    y,x=np.nonzero(patch);return np.c_[x+low[0],y+low[1]]


def path_array(value):
    xy=np.asarray(value,np.float64)
    if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<2 or not np.isfinite(xy).all() or np.any(np.diff(xy[:,0])<0):
        raise ValueError('Invalid/nonmonotone sampled path')
    return xy


def independent_neighbors(base,xy,pitch,p):
    samples=xy[np.linspace(0,len(xy)-1,min(9,len(xy))).astype(int)];good=0
    for x,y in samples:
        col=int(round(x));ys=np.flatnonzero(base[:,max(0,col-1):min(base.shape[1],col+2)].any(axis=1))
        above=ys[ys<y];below=ys[ys>y]
        if not len(above) or not len(below):continue
        a=y-above[-1];b=below[0]-y
        good+=int(min(a,b)>=p['independent_neighbor_minimum_pitch']*pitch and max(a,b)<=p['independent_neighbor_maximum_pitch']*pitch)
    return good/max(1,len(samples))


def audit_one(task):
    row,run,p,sp,tp,cp=task;cv2.setNumThreads(1);errors=[]
    result={'output_file':row['output_file'],'errors':errors}
    try:
        rel=Path(row['output_file']);role,name,stem=rel.parent.name,rel.name,rel.stem
        data,deps=inputs(row);shape=data['surface'].shape;s=data['surface']>0;pitch=float(data['pitch'])
        case=read_json(run/'记录/cases'/role/(stem+'.json'));record=case['record'];obs=case['observations']
        if record.get('status')!='completed':raise ValueError('Incomplete case')
        if record['input_dependency_sha256']!=deps:issue(errors,'input_dependency_hash_or_keys')
        if record['original_sha256']!=row['output_sha256'] or sha(DATA/rel)!=row['output_sha256']:issue(errors,'original_hash')
        masks={k:read_image(run/'记录/masks'/k/role/name,0) for k in MASKS}
        for k,a in masks.items():
            if a.shape!=shape or a.dtype!=np.uint8 or np.any((a!=0)&(a!=255)):raise ValueError('Nonbinary/wrong size '+k)
            if k!='exclusion' and np.any(a[~s]):issue(errors,'outside_surface',k)
        t=masks['transition']>0;base=data['f2_merged']>0;oldinf=data['f2_inferred']>0
        new=masks['new']>0;merged=masks['merged']>0;inferred=masks['inferred']>0
        if not np.array_equal(masks['transition'],data['f2_transition']):issue(errors,'transition_not_frozen_F2')
        if np.any(t&~s) or np.any(new&~t):issue(errors,'new_or_transition_outside_authorized_region')
        if np.any(base&~merged) or np.any(masks['removed']):issue(errors,'old_F2_geometry_removed')
        if not np.array_equal(new,merged&~base) or not np.array_equal(merged,base|new):issue(errors,'new_pixel_ledger')
        if not np.array_equal(merged[~t],base[~t]) or not np.array_equal(inferred[~t],oldinf[~t]):issue(errors,'outside_transition_mask_changed')
        if np.any(inferred&~merged) or np.any(oldinf&~inferred):issue(errors,'inferred_subset_or_preservation')
        display=read_image(run/'图像结果/02_F3融合边界'/role/name);candidate=read_image(run/'图像结果/01_D4全卷面候选'/role/name)
        for key,image in [('formal',display),('candidate',candidate)]:
            if image.shape!=data['original'].shape or image.dtype!=np.uint8:raise ValueError('Wrong image '+key)
            if np.any(image[~s]):issue(errors,'display_outside_surface',key)
        if not np.array_equal(display[~t],data['f2_image'][~t]):issue(errors,'display_outside_transition_changed')
        if not np.any(new) and not case['accepted_points'] and not np.array_equal(display,data['f2_image']):issue(errors,'empty_addition_display_changed')
        with np.load(run/'记录/fields'/role/(stem+'.npz'),allow_pickle=False) as z:f={k:z[k].copy() for k in z.files}
        x0,y0=int(f['x0']),int(f['y0']);body=f['body_barrier'];hh,ww=body.shape
        if x0<0 or y0<0 or x0+ww>shape[1] or y0+hh>shape[0]:raise ValueError('Field bounds')
        if body.dtype!=np.float32:raise ValueError('Body field must be original float32')
        for k,a in f.items():
            if k in ('x0','y0'):continue
            if a.shape!=body.shape or not np.isfinite(a).all():raise ValueError('Field shape/finite '+k)
            if k in ('body_barrier','broad_score') and np.any((a<0)|(a>1)):raise ValueError('Field range '+k)
            if k in ('unknown_mask','valid_support','cleaned_shadow') and np.any((a!=0)&(a!=255)):raise ValueError('Field binary '+k)
        sy,sx=np.nonzero(s)
        if len(sx):
            margin=int(math.ceil(sp['margin_pitch']*pitch))
            expected=[max(0,int(sx.min())-margin),max(0,int(sy.min())-margin),min(shape[1],int(sx.max())+margin+1),min(shape[0],int(sy.max())+margin+1)]
            if [x0,y0,x0+ww,y0+hh]!=expected:issue(errors,'field_crop_not_full_surface',expected)
        od=record['observation_diagnostics']
        if od['observation_scope']!='entire_existing_surface_S_not_F2_transition_band' or od['surface_pixels']!=int(s.sum()) or od['shadow'].get('roi_pixels',0)!=int(s.sum()):issue(errors,'full_surface_scope_record')
        source_ex=data['c3_exclusion'].copy();source_ex[y0:y0+hh,x0:x0+ww]|=f['unknown_mask']
        if not np.array_equal(source_ex,masks['exclusion']):issue(errors,'exclusion_union_mismatch')
        pts=obs['points'];by_id={int(q['point_id']):q for q in pts};by_name={q['id']:q for q in pts}
        if len(by_id)!=len(pts) or len(by_name)!=len(pts) or set(by_id)!=set(range(len(pts))):issue(errors,'point_ids_unique_sequential')
        rebuilt_obs=np.zeros(shape,np.uint8);outside_t_points=0
        for q in pts:
            pid=int(q['point_id']);xy=np.asarray([[q['x'],q['y']]],float)
            if q['id']!=f'D4_full_new_{pid:06d}' or q.get('source')!='D4_full':issue(errors,'point_source_identity',pid)
            if not sample(s,xy)[0] or not field(f,'cleaned_shadow',xy)[0] or not field(f,'valid_support',xy)[0]:issue(errors,'point_not_in_surface_clean_shadow_support',pid)
            if field(f,'unknown_mask',xy)[0] or field(f,'body_barrier',xy,1)[0]>tp['body_limit']:issue(errors,'point_unknown_body',pid)
            score=float(field(f,'broad_score',xy)[0]);width=float(field(f,'width_px',xy)[0]);slope=float(field(f,'tangent_slope',xy)[0])
            if abs(score-q['score'])>1e-6 or abs(width-q['width'])>1e-6 or abs(slope-q['slope'])>1e-6:issue(errors,'point_not_saved_field_value',pid)
            if score<tp['minimum_score']:issue(errors,'point_score_gate',pid)
            normal=float(q['pitch'])/math.sqrt(1+slope*slope)
            if not tp['minimum_width_pitch']*normal<=width<=tp['maximum_width_pitch']*normal:issue(errors,'point_width_gate',pid)
            x,y=np.rint(xy[0]).astype(int);rebuilt_obs[y,x]=255;outside_t_points+=int(not t[y,x])
        if not np.array_equal(rebuilt_obs,masks['full_d4_observed']):issue(errors,'observed_mask_point_mapping')
        tracks={str(q['track_id']):q for q in obs['tracks']};members=[]
        if len(tracks)!=len(obs['tracks']):issue(errors,'duplicate_track_id')
        for tid,track in tracks.items():
            for q in track['points']:
                members.append(q['point_id'])
                if by_id.get(q['point_id'])!=q:issue(errors,'track_point_not_actual_global_point',tid)
        if Counter(members)!=Counter(by_id.keys()):issue(errors,'each_point_in_exactly_one_track')
        curves={q['curve_id']:q for q in obs['fit']['curves']};source_members={};full=np.zeros(shape,np.uint8);fullinf=np.zeros(shape,np.uint8)
        if len(curves)!=len(obs['fit']['curves']):issue(errors,'duplicate_curve_id')
        for cid,c in curves.items():
            xy=path_array(c['sampled_points_xy']);local=tracks.get(str(c['track_id']),{}).get('points',[]);indices=c['source_point_indices']
            if len(set(indices))!=len(indices) or any(not 0<=int(i)<len(local) for i in indices):raise ValueError('Source index not local track index '+cid)
            source_members[cid]={local[int(i)]['point_id'] for i in indices}
            if len(indices)<cp['minimum_points']:issue(errors,'D4_curve_insufficient_points',cid)
            states=np.asarray(c['point_states'])
            if len(states)!=len(xy) or np.any(~np.isin(states,['observed','inferred'])):raise ValueError('Invalid source point_states')
            rr=raster(xy,shape)
            if np.any(sample(s,rr)==0) or np.any(field(f,'unknown_mask',rr,255)>0):issue(errors,'full_D4_raster_outside_surface_or_unknown',cid)
            ij=np.rint(xy).astype(np.int32)
            cv2.polylines(full,[ij],False,255,1,cv2.LINE_8)
            for j in np.flatnonzero((states[:-1]=='inferred')|(states[1:]=='inferred')):
                cv2.line(fullinf,tuple(ij[j]),tuple(ij[j+1]),255,1,cv2.LINE_8)
        if not np.array_equal(full,masks['full_d4']) or not np.array_equal(fullinf,masks['full_d4_inferred']):issue(errors,'full_D4_masks_not_source_curve_raster')
        for key,maskkey in [('smooth_mask','full_d4'),('inferred_mask','full_d4_inferred')]:
            expected=Path('记录/masks')/maskkey/role/name;ref=obs['fit'][key]
            if ref!={'png':expected.as_posix(),'sha256':sha(run/expected)}:issue(errors,'fit_mask_reference',key)
        groups=case['groups'];gids={g['group_id'] for g in groups};accepted=case['accepted_points'];apmap={q['id']:q for q in accepted}
        if len(gids)!=len(groups) or len(apmap)!=len(accepted):issue(errors,'duplicate_accepted_group_or_point')
        rebuilt_new=np.zeros(shape,np.uint8);rebuilt_inf=np.zeros(shape,np.uint8);used_names=[];paths=[]
        for g in groups:
            gid=g['group_id'];cid=g['source_curve_id'];xy=path_array(g['sampled_points_xy']);rr=raster(xy,shape)
            if g['status']!='accepted' or cid not in curves or g['track_id']!=curves[cid]['track_id']:issue(errors,'group_not_actual_source_curve',gid);continue
            if any(int(i) not in source_members[cid] for i in g['source_point_ids']):issue(errors,'group_source_point_local_mapping',gid)
            for key in ('transition','exclusion'):
                vals=sample(masks[key],rr)
                if (key=='transition' and np.any(vals==0)) or (key=='exclusion' and np.any(vals>0)):issue(errors,'group_raster_region_'+key,gid)
            if np.any(field(f,'unknown_mask',rr,255)) or np.any(field(f,'body_barrier',rr,1)>p['body_hard_threshold']):issue(errors,'group_unknown_or_body',gid)
            broad=field(f,'broad_score',rr);fraction=float(np.mean(broad>=p['broad_threshold']))
            unsupported=field(f,'broad_score',xy)<p['broad_threshold']
            longest=max((float(np.linalg.norm(np.diff(xy[a:z],axis=0),axis=1).sum())+1 for a,z in runs(unsupported)),default=0.)
            if fraction<p['minimum_broad_fraction'] or longest>p['maximum_unsupported_run_pitch']*pitch:issue(errors,'group_image_support_gate',gid)
            if abs(fraction-g['statistics']['broad_fraction'])>1e-6 or abs(longest-g['statistics']['longest_unsupported_px'])>1e-5:issue(errors,'group_support_statistics',gid)
            points=[by_id[int(i)] for i in g['source_point_ids']];lo=float(xy[:,0].min());hi=float(xy[:,0].max());unique={tuple(np.rint([q['x'],q['y']]).astype(int)) for q in points}
            span=(max(q['x'] for q in points)-min(q['x'] for q in points))/max(1,hi-lo) if points else 0
            if len(unique)<p['minimum_fragment_points'] or span<p['minimum_observation_span_fraction']:issue(errors,'group_observation_span_or_count',gid)
            for q in points:
                qxy=np.asarray([[q['x'],q['y']]])
                if not lo<=q['x']<=hi or abs(q['y']-np.interp(q['x'],xy[:,0],xy[:,1]))>.30*pitch or not sample(t,qxy)[0]:issue(errors,'group_source_point_outside_path',gid)
            if g['mode']=='independent_supported_fragment':
                neigh=independent_neighbors(base,xy,pitch,p)
                if len(unique)<p['independent_minimum_points'] or hi-lo<p['independent_minimum_span_pitch']*pitch or fraction<p['independent_broad_fraction'] or neigh<.70:issue(errors,'independent_neighbor_or_support_gate',gid)
                if abs(neigh-g['statistics']['neighbor_valid_fraction'])>1e-6:issue(errors,'independent_neighbor_statistics',gid)
            elif g['mode']!='anchored_visual_row' or not g['reference_matches']:issue(errors,'unsupported_group_mode',gid)
            for pid in g['accepted_point_ids']:
                used_names.append(pid);ap=apmap.get(pid);raw=by_name.get(pid)
                if ap is None or raw is None or ap!={**raw,'group_id':gid}:issue(errors,'accepted_point_not_exact_actual_observation',pid)
                elif raw['point_id'] not in g['source_point_ids']:issue(errors,'accepted_point_not_in_group_source_members',pid)
            fresh=~base[rr[:,1],rr[:,0]]
            if int(fresh.sum())!=g['new_pixels']:issue(errors,'group_new_pixels',gid)
            q=rr[fresh];rebuilt_new[q[:,1],q[:,0]]=255
            gap=(broad<p['broad_threshold'])|sample(masks['full_d4_inferred'],rr).astype(bool)
            q=rr[fresh&gap];rebuilt_inf[q[:,1],q[:,0]]=255;paths.append((gid,xy))
        if Counter(used_names)!=Counter(apmap.keys()):issue(errors,'accepted_points_group_ledger')
        if not np.array_equal(rebuilt_new,masks['new']) or not np.array_equal(inferred,oldinf|(rebuilt_inf>0)):issue(errors,'new_or_inferred_raster_reconstruction')
        for i,(gid,xy) in enumerate(paths):
            for qid,q in paths[i+1:]:
                lo=max(xy[:,0].min(),q[:,0].min());hi=min(xy[:,0].max(),q[:,0].max())
                if hi<=lo:continue
                xs=np.arange(math.ceil(lo),math.floor(hi)+1);d=np.interp(xs,xy[:,0],xy[:,1])-np.interp(xs,q[:,0],q[:,1])
                if len(d) and (np.min(np.abs(d))<p['new_fragment_separation_pitch']*pitch or np.min(d)<0<np.max(d)):issue(errors,'new_groups_order_or_spacing',[gid,qid])
        for k,v in record['pixels'].items():
            if v!=int(np.count_nonzero(masks[k])):issue(errors,'record_pixel_count',k)
        if record['new_full_surface_points']!=len(pts) or record['full_surface_curves']!=len(curves) or record['accepted_point_count']!=len(accepted):issue(errors,'record_object_counts')
        outputs=[Path('图像结果')/folder/role/name for folder in ('01_D4全卷面候选','02_F3融合边界')]
        outputs += [Path('记录/masks')/k/role/name for k in MASKS]+[Path('记录/fields')/role/(stem+'.npz'),Path('记录/cases')/role/(stem+'.json')]
        result.update({'points':len(pts),'points_outside_T':outside_t_points,'full_curves':len(curves),'accepted_groups':len(groups),'accepted_points':len(accepted),'added_pixels':int(new.sum()),'dependency_count':len(deps),'files':[{'path':q.as_posix(),'size':(run/q).stat().st_size,'sha256':sha(run/q)} for q in outputs]})
    except Exception as exc:issue(errors,'audit_exception',repr(exc))
    result['status']='failed' if errors else 'passed';return result


def audit_run(run,workers):
    errors=[];context=read_json(run/'run_context.json');summary=read_json(run/'summary.json');mode=context['mode'];rows=selection(mode)
    if mode not in ('pilot','all') or len(rows)!=(12 if mode=='pilot' else 400) or context['selection']!=[q['output_file'] for q in rows]:issue(errors,'fixed_selection_or_denominator')
    if any(summary.get(k)!=v for k,v in {'planned':len(rows),'completed':len(rows),'failed':0,'exit_code':0,'source_unchanged':True}.items()):issue(errors,'unsuccessful_summary')
    snapshots=context['source_snapshot_paths'];sources=context['source_sha256'];provenance=[]
    if set(snapshots)!=set(sources):issue(errors,'snapshot_keys')
    for original,expected in sources.items():
        q=Path(snapshots[original])
        if q.is_absolute() or '..' in q.parts:raise ValueError('Unsafe source snapshot')
        actual=sha(run/q)
        if actual!=expected:issue(errors,'source_snapshot_hash',original)
        provenance.append({'original_source':original,'snapshot':q.as_posix(),'size':(run/q).stat().st_size,'sha256':actual})
    protocol=read_json(run/snapshots[str(HERE/'protocol.json')])
    for original,expected in protocol['baseline_source_sha256'].items():
        if sha(original)!=expected or sources.get(original)!=expected:issue(errors,'frozen_baseline_hash',original)
    if sha(DATA/'manifest.csv')!=protocol['input_manifest_sha256'] or sha(F2/'all_001/output_manifest.json')!=protocol['f2_output_manifest_sha256']:issue(errors,'frozen_reference_manifest')
    if mode=='all' and sources!=read_json(HERE/'freeze_all.json')['source_sha256']:issue(errors,'full_run_freeze')
    p=params(run/snapshots[str(HERE/'surface_fusion.py')]);dp={}
    for filename in ('shadow_width.py','shadow_tracks.py','ribbon_curves.py'):
        original=next(k for k in sources if Path(k).name==filename and 'dark_width_20261027_v4' in k);dp[filename]=params(run/snapshots[original])
    logs=[json.loads(x) for x in (run/'run_log.jsonl').read_text(encoding='utf8').splitlines()]
    if len(logs)!=len(rows) or any(a.get('output_file')!=b['output_file'] or a.get('status')!='completed' for a,b in zip(logs,rows)):issue(errors,'run_log_denominator_or_order')
    cases=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks=[(q,run,p,dp['shadow_width.py'],dp['shadow_tracks.py'],dp['ribbon_curves.py']) for q in rows]
        for i,c in enumerate(pool.map(audit_one,tasks),1):
            cases.append(c)
            if c['errors']:print(json.dumps({'failure':c['output_file'],'errors':c['errors']},ensure_ascii=False),flush=True)
            if i%25==0 or i==len(rows):print(json.dumps({'audited':i,'planned':len(rows),'failed':sum(q['status']=='failed' for q in cases)}),flush=True)
    return {'name':'F3 independent full-surface-source and additive-fusion audit','run':str(run),'mode':mode,'workers':workers,'auditor_sha256':sha(Path(__file__)),'finished_utc':datetime.now(timezone.utc).isoformat(),'planned':len(rows),'passed':sum(q['status']=='passed' for q in cases),'failed':sum(q['status']=='failed' for q in cases),'global_errors':errors,'new_points':sum(q.get('points',0) for q in cases),'accepted_points':sum(q.get('accepted_points',0) for q in cases),'added_pixels':sum(q.get('added_pixels',0) for q in cases),'images_without_additions':sum(q.get('added_pixels')==0 for q in cases),'cases':cases,'checked_source_snapshots':provenance,'accuracy':None,'meaning':'Independent structural/provenance and saved image-evidence checks; not true boundary accuracy or physical rope identity.','no_detector_or_fit_rerun':True,'equipment_access':False,'reserved_access':False}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True,type=Path);ap.add_argument('--workers',type=int,default=3);ap.add_argument('--output',type=Path);args=ap.parse_args()
    run=args.run.absolute() if args.run.is_absolute() else HERE/args.run;out=args.output.absolute() if args.output else run/'audit.json'
    if run.parent!=HERE or out.parent!=run or out.exists():raise ValueError('New audit in existing direct-child run required')
    report=audit_run(run,args.workers)
    if not report['failed'] and not report['global_errors']:
        manifest={'version':'RB-F3.0.0','run':str(run),'case_count':len(report['cases']),'file_count':sum(len(q['files']) for q in report['cases']),'cases':[{'output_file':q['output_file'],'files':q['files']} for q in report['cases']],'scope':'Immutable prediction images/masks/float fields/case JSON; excludes QA and independent audit.'}
        target=run/'output_manifest.json'
        if target.exists():
            if read_json(target)!=manifest:issue(report['global_errors'],'existing_manifest_differs')
        else:
            with target.open('x',encoding='utf8') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
        report['output_manifest_sha256']=sha(target)
    with out.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    archive=run/'记录/audit_tools'/('audit_surface_'+report['auditor_sha256'][:12]+'.py');archive.parent.mkdir(parents=True,exist_ok=True)
    if not archive.exists():archive.write_bytes(Path(__file__).read_bytes())
    elif sha(archive)!=report['auditor_sha256']:raise ValueError('Auditor archive differs')
    prov=run/(out.stem+'_provenance.json')
    payload={'audit_path':out.name,'audit_sha256':sha(out),'auditor_snapshot':archive.relative_to(run).as_posix(),'auditor_sha256':sha(archive),'checked_source_snapshots':report['checked_source_snapshots'],'run_context_sha256':sha(run/'run_context.json'),'summary_sha256':sha(run/'summary.json'),'output_manifest_sha256':report.get('output_manifest_sha256')}
    with prov.open('x',encoding='utf8') as f:json.dump(payload,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:report[k] for k in ('planned','passed','failed','global_errors','accepted_points','added_pixels','images_without_additions')},ensure_ascii=False),flush=True)
    return int(bool(report['failed'] or report['global_errors']))


if __name__=='__main__':raise SystemExit(main())
