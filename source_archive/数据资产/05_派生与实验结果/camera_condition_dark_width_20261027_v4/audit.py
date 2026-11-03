"""D4独立审计：只读已经保存的预测，不调用检测、归类或拟合算法。

检查像素、成员、曲线、哈希和固定分母；通过不代表视觉真值准确。
用法：python audit.py --run pilot_002（默认另存该目录/audit.json）。
PNG量化场只能提供保守界；body_float NPZ存在时优先用原float32场。
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import struct
import cv2
import numpy as np
from data_io import HERE, D1, D2, C3, DATA, old, selection, read_json, read_image


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def issue(errors, code, detail=None):
    errors.append({'code': code, 'detail': detail})


def sample(a, xy, x0=0, y0=0):
    xy = np.asarray(xy, np.float32)
    return cv2.remap(a.astype(np.float32), (xy[:, 0]-x0)[None],
                     (xy[:, 1]-y0)[None], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)[0]


def intervals(mask):
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))


def longest_arc(xy, mask):
    # 与模块公开门槛一致：每一采样点还代表一个前向路径步长。
    ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return max((float(ds[a:max(a, z-1)].sum()+ds[min(a, len(ds)-1)])
                for a, z in intervals(mask)), default=0.) if len(ds) else float(np.any(mask))


def body_bad(xy, hit, pitch):
    return (longest_arc(xy, hit) >= max(1, math.ceil(.18*pitch)) or
            float(np.mean(hit)) > .12)


def paths_for(row):
    rel = Path(row['output_file']); role, name, stem = rel.parent.name, rel.name, rel.stem
    return [*(Path('图像结果')/folder/role/name for folder in
              ('01_阴影二值化', '02_去细股阴影', '03_新黄点', '04_边界结果')),
            *(Path('记录/masks')/key/role/name for key in ('observed', 'smooth', 'inferred')),
            *(Path('记录/fields')/key/role/name for key in ('body_barrier', 'unknown', 'broad_score')),
            Path('记录/cases')/role/(stem+'.json')]


def deps_for(row):
    rel=Path(row['output_file']); role,name,stem=rel.parent.name,rel.name,rel.stem
    return [old.C4/'roi/surface_masks'/role/name,
            C3/'boundaries/records'/role/(stem+'.json'),
            old.C4/'boundaries/records'/role/(stem+'.json'),
            D1/'all_001/记录/masks/dark_roi'/role/name,
            D1/'all_001/图像结果/02_对比度增强ROI'/role/name,
            D1/'all_001/记录/cases'/role/(stem+'.json')]


def audit_one(row, run, d1_inventory):
    errors=[]; limited=[]; parts=paths_for(row)
    result={'output_file':row['output_file'], 'errors':errors, 'limitations':limited}
    try:
        rel=Path(row['output_file']); role,name,stem=rel.parent.name,rel.name,rel.stem
        if rel.is_absolute() or '..' in rel.parts or len(rel.parts)!=3 or rel.parts[0]!='images':
            raise ValueError('input path outside fixed image namespace')
        source=DATA/rel
        if digest(source)!=row['output_sha256']: issue(errors,'original_image_hash')
        with source.open('rb') as f: header=f.read(24)
        if header[:8]!=b'\x89PNG\r\n\x1a\n' or header[12:16]!=b'IHDR': raise ValueError('not PNG')
        width,height=struct.unpack('>II',header[16:24]); shape=(height,width)
        case=read_json(run/parts[-1]); record=case['record']
        if record.get('status')!='completed': raise ValueError('case did not complete')
        if record.get('source_png_sha256')!=row['output_sha256']: issue(errors,'record_source_hash')
        for key in ('output_file','camera_role','source_group','source_file','source_time_s','dataset_use_status'):
            if record.get(key)!=row.get(key): issue(errors,'source_metadata_mismatch',key)
        deps=record.get('input_dependency_sha256',{}); expected=deps_for(row)
        if set(deps)!=set(map(str,expected)): issue(errors,'six_dependency_keys')
        for p in expected:
            h=digest(p)
            if deps.get(str(p))!=h: issue(errors,'dependency_hash',str(p))
            if p.is_relative_to(D1/'all_001') and d1_inventory.get(p.relative_to(D1/'all_001').as_posix())!=h:
                issue(errors,'frozen_d1_pixel_or_record_hash',str(p))
        roi=read_image(expected[3],0)
        if roi.shape!=shape: raise ValueError('ROI shape')
        outside=roi==0
        images=[read_image(run/p, cv2.IMREAD_UNCHANGED) for p in parts[:-1]]
        for p,img in zip(parts[:-1],images):
            if img is None or img.shape[:2]!=shape or img.dtype!=np.uint8: raise ValueError('output dimensions/dtype '+str(p))
            if np.any(img[outside]): issue(errors,'outside_roi_not_zero',str(p))
        raw,clean,_,_,observed,smooth,inferred,body_png,unknown,broad_png=images
        for key,img in [('raw',raw),('clean',clean),('observed',observed),('smooth',smooth),('inferred',inferred),('unknown',unknown)]:
            if img.ndim!=2 or np.any((img!=0)&(img!=255)): issue(errors,'not_binary',key)
        if np.any((clean>0)&(raw==0)): issue(errors,'opened_not_subset_raw')
        if np.any((inferred>0)&(smooth==0)): issue(errors,'inferred_not_subset_smooth')
        pitch=float(record['pitch_px'])
        if not math.isfinite(pitch) or not 8<=pitch<=70: raise ValueError('invalid pitch')
        body_path=Path('记录/fields/body_float')/role/(stem+'.npz')
        if (run/body_path).exists():
            with np.load(run/body_path,allow_pickle=False) as z:
                body=z['body'].copy(); x0=int(z['x0']); y0=int(z['y0'])
            if body.ndim!=2 or body.dtype!=np.float32 or not np.all(np.isfinite(body)) or np.any((body<0)|(body>1)):
                raise ValueError('invalid body float field')
            hh,ww=body.shape
            if x0<0 or y0<0 or x0+ww>width or y0+hh>height: raise ValueError('body crop outside image')
            rounded=np.zeros(shape,np.uint8); rounded[y0:y0+hh,x0:x0+ww]=np.rint(body*255).astype(np.uint8); rounded[outside]=0
            if not np.array_equal(rounded,body_png): issue(errors,'body_float_png_disagreement')
            body_lo=body_hi=body; result['body_precision']='float32_npz'; parts.append(body_path)
        else:
            # 不能将166/255当作原浮点值；166包含跨越0.65的量化区间。
            body_lo=np.maximum(0,(body_png.astype(np.float32)-.5)/255)
            body_hi=np.minimum(1,(body_png.astype(np.float32)+.5)/255)
            x0=y0=0; result['body_precision']='png_interval_limited'
            limited.append('body float32 unavailable: only definite threshold violations checked; upper/lower rounding bounds retained')
        points=case.get('new_points',[]); tracks=case.get('new_tracks',[]); curves=case.get('fit',{}).get('curves',[])
        ids=[p['point_id'] for p in points]
        if len(set(ids))!=len(ids) or ids!=list(range(len(ids))): issue(errors,'new_point_ids_not_unique_fresh_local_sequence')
        lookup={p['point_id']:p for p in points}; used=[]; trackmap={}; observed_expected=np.zeros(shape,np.uint8)
        for p in points:
            xx=int(p['x']); yy=int(round(p['y']))
            if not (0<=xx<width and 0<=yy<height and roi[yy,xx] and clean[yy,xx]):
                issue(errors,'new_point_not_in_roi_and_cleaned_shadow',p['point_id']); continue
            if unknown[yy,xx]: issue(errors,'new_point_in_unknown',p['point_id'])
            cv2.circle(observed_expected,(xx,yy),1,255,-1)
        observed_expected[outside]=0
        if not np.array_equal(observed_expected,observed): issue(errors,'observed_mask_not_new_points')
        for t in tracks:
            tid=t['track_id']
            if not re.fullmatch(r'wide_track_\d{5,}',tid) or tid in trackmap: issue(errors,'not_unique_new_track_id',tid)
            trackmap[tid]=t
            for p in t['points']:
                used.append(p['point_id'])
                if lookup.get(p['point_id'])!=p: issue(errors,'track_point_differs_from_new_detection',p.get('point_id'))
        if Counter(used)!=Counter(ids): issue(errors,'each_point_must_belong_to_exactly_one_track')
        if record.get('tracking_diagnostics',{}).get('old_observations_used') is not False: issue(errors,'old_id_usage_not_explicitly_false')
        if (len(points),len(tracks),len(curves))!=(record.get('new_yellow_count'),record.get('new_track_count'),record.get('curve_count')):
            issue(errors,'record_counts')
        rebuilt=np.zeros(shape,np.uint8); rebuilt_inf=np.zeros(shape,np.uint8); curve_arrays=[]; curve_ids=set(); uncertain_body=0
        for c in curves:
            cid=c['curve_id']; tid=c['track_id']; xy=np.asarray(c['sampled_points_xy'],np.float32); states=np.asarray(c['point_states'])
            if cid in curve_ids: issue(errors,'duplicate_curve_id',cid)
            curve_ids.add(cid)
            if tid not in trackmap or c.get('original_group_ids')!=[tid]: issue(errors,'curve_old_or_unknown_group',cid); continue
            indices=c.get('source_point_indices',[])
            if len(set(indices))!=len(indices) or any(not isinstance(i,int) or i<0 or i>=len(trackmap[tid]['points']) for i in indices):
                issue(errors,'curve_source_indices',cid)
            if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<2 or not np.all(np.isfinite(xy)): issue(errors,'invalid_curve_coordinates',cid); continue
            if np.any(np.abs(np.diff(xy[:,0])-1)>1e-5): issue(errors,'curve_x_must_sample_every_pixel',cid)
            if len(states)!=len(xy) or not np.all(np.isin(states,['observed','inferred'])): issue(errors,'invalid_point_states',cid); continue
            if np.any(sample(roi,xy)<=127): issue(errors,'curve_outside_roi',cid)
            if np.any(sample(unknown,xy)>0): issue(errors,'curve_crosses_unknown_float_path',cid)
            low=sample(body_lo,xy,x0,y0)>.65; high=sample(body_hi,xy,x0,y0)>.65
            if body_bad(xy,low,pitch): issue(errors,'curve_body_gate_violation',cid)
            elif body_bad(xy,high,pitch): uncertain_body+=1
            inf=states=='inferred'
            if np.any(inf&low): issue(errors,'inference_crosses_supported_body',cid)
            if longest_arc(xy,inf)>1.5*pitch+1e-5: issue(errors,'inferred_gap_exceeds_1_5_pitch',cid)
            clean_v=sample(clean,xy); broad_value=sample(broad_png,xy)/255
            # 线性插值权重非负；±半量化级是保守界，不需为每条曲线复制全图。
            broad_low=np.maximum(0,broad_value-.5/255)
            broad_high=np.minimum(1,broad_value+.5/255)
            if np.any((~inf)&((clean_v<127)|(broad_high<.10))): issue(errors,'observed_state_lacks_saved_image_support',cid)
            if np.any(inf&(clean_v>=127)&(broad_low>=.10)): issue(errors,'inferred_state_despite_definite_saved_support',cid)
            rr=np.rint(xy).astype(int)
            for j in range(len(rr)-1):
                cv2.line(rebuilt,tuple(rr[j]),tuple(rr[j+1]),255,1,cv2.LINE_8)
                if inf[j] or inf[j+1]: cv2.line(rebuilt_inf,tuple(rr[j]),tuple(rr[j+1]),255,1,cv2.LINE_8)
            curve_arrays.append((cid,xy))
        if uncertain_body: limited.append({'body_decisions_ambiguous_after_png_quantization':uncertain_body})
        if np.any((rebuilt>0)&(unknown>0)): issue(errors,'drawn_curve_pixel_crosses_unknown')
        if np.any(rebuilt[outside]): issue(errors,'drawn_curve_pixel_outside_roi_before_clipping')
        rebuilt[outside]=0; rebuilt_inf[outside]=0
        if not np.array_equal(rebuilt,smooth): issue(errors,'smooth_mask_not_exactly_accepted_curves')
        if not np.array_equal(rebuilt_inf,inferred): issue(errors,'inferred_mask_not_exactly_accepted_states')
        for i,(cid,xy) in enumerate(curve_arrays):
            for qid,q in curve_arrays[i+1:]:
                lo=max(xy[0,0],q[0,0]); hi=min(xy[-1,0],q[-1,0])
                if hi<lo: continue
                xs=np.arange(math.ceil(float(lo)),math.floor(float(hi))+1)
                if not len(xs): continue
                d=np.interp(xs,xy[:,0],xy[:,1])-np.interp(xs,q[:,0],q[:,1])
                if np.min(np.abs(d))<.45*pitch-1e-5 or np.min(d)<0<np.max(d):
                    issue(errors,'accepted_curves_too_close_or_crossing',[cid,qid,float(np.min(np.abs(d)))])
        rejected_ids={r.get('curve_id') for r in case.get('fit',{}).get('rejections',[]) if r.get('curve_id')}
        if curve_ids&rejected_ids: issue(errors,'rejected_curve_in_accepted_output',sorted(curve_ids&rejected_ids))
        result.update({'new_points':len(points),'new_tracks':len(tracks),'accepted_curves':len(curves),
                       'source_sha256':row['output_sha256'],'dependency_count':len(expected),
                       'immutable_files':[{'path':p.as_posix(),'size':(run/p).stat().st_size,'sha256':digest(run/p)} for p in parts]})
    except Exception as e:
        issue(errors,'audit_exception',repr(e))
    result['status']='failed' if errors else 'passed'
    return result


def audit_run(run):
    cv2.setNumThreads(1); errors=[]; context=read_json(run/'run_context.json'); summary=read_json(run/'summary.json')
    mode=context.get('mode'); rows,_=selection(mode)
    if mode not in ('pilot','all'): issue(errors,'invalid_mode',mode)
    if context.get('selection')!=[r['output_file'] for r in rows]: issue(errors,'fixed_selection_mismatch')
    if len(rows)!=(400 if mode=='all' else 12): issue(errors,'fixed_denominator')
    if any(summary.get(k)!=v for k,v in {'planned':len(rows),'completed':len(rows),'failed':0,'source_unchanged':True,'exit_code':0}.items()): issue(errors,'unsuccessful_run_summary',summary)
    protocol=read_json(HERE/'protocol.json'); baseline=protocol['frozen_baseline_sha256']
    if len(baseline)!=20: issue(errors,'expected_20_baseline_sources',len(baseline))
    for p,h in baseline.items():
        if digest(p)!=h: issue(errors,'baseline_source_changed',p)
    source=context['source_sha256']; snaps=context['source_snapshot_paths']
    if set(source)!=set(snaps): issue(errors,'snapshot_keys')
    for p,h in source.items():
        part=Path(snaps.get(p,''))
        if part.is_absolute() or '..' in part.parts or not part.parts or part.parts[0]!='source_snapshot': issue(errors,'unsafe_snapshot_path',p); continue
        if digest(run/part)!=h: issue(errors,'snapshot_hash',p)
    if any(source.get(p)!=h for p,h in baseline.items()): issue(errors,'snapshot_baseline_mismatch')
    if mode=='all' and source!=read_json(HERE/'freeze_all.json')['source_sha256']: issue(errors,'full_source_freeze_mismatch')
    if digest(DATA/'manifest.csv')!=read_json(D1/'protocol.json')['input_manifest_sha256']: issue(errors,'original_manifest_hash')
    d1_manifest=D1/'all_001/output_manifest.json'
    if digest(d1_manifest)!=read_json(D2/'protocol.json')['d1_output_manifest_sha256']: issue(errors,'d1_manifest_hash')
    d1_inventory={f['path']:f['sha256'] for f in read_json(d1_manifest)['files']}
    log=[json.loads(line) for line in (run/'run_log.jsonl').read_text(encoding='utf8').splitlines()]
    if len(log)!=len(rows) or any(a.get('output_file')!=b['output_file'] or a.get('status')!='completed' for a,b in zip(log,rows)): issue(errors,'log_denominator_or_order')
    cases=[]
    for i,row in enumerate(rows,1):
        cases.append(audit_one(row,run,d1_inventory))
        if i%25==0 or i==len(rows): print(json.dumps({'audited':i,'planned':len(rows),'failed':sum(c['status']=='failed' for c in cases)}),flush=True)
    if mode=='all' and any(c.get('body_precision')!='float32_npz' for c in cases):
        issue(errors,'final_400_requires_exact_body_float_field')
    return {'audit_name':'D4 independent saved-prediction structural audit','run':str(run),'mode':mode,
            'finished_utc':datetime.now(timezone.utc).isoformat(),'auditor_sha256':digest(Path(__file__)),
            'planned':len(rows),'passed':sum(c['status']=='passed' for c in cases),'failed':sum(c['status']=='failed' for c in cases),
            'limited_body_cases':sum(c.get('body_precision')!='float32_npz' for c in cases),
            'images_without_accepted_curves':sum(c.get('accepted_curves')==0 for c in cases),
            'global_errors':errors,'baseline_source_count':len(baseline),'run_snapshot_count':len(source),'cases':cases,
            'ground_truth_accuracy':None,'equipment_access':False,'reserved_access':False,
            'meaning':'Saved membership, image support, geometry, pixels and hashes only; no visual accuracy, material identity or contact truth. No-curve cases remain in denominator.',
            'field_precision_note':'Body NPZ is exact saved float32. Broad-score PNG has rounding intervals; ambiguous support thresholds cannot be certified exactly.'}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',type=Path,required=True); ap.add_argument('--output',type=Path); args=ap.parse_args()
    run=args.run.absolute() if args.run.is_absolute() else (HERE/args.run).absolute()
    if run.parent!=HERE or not run.is_dir(): raise ValueError('run must be existing direct child')
    out=args.output.absolute() if args.output else run/'audit.json'
    if out.parent!=run or out.exists(): raise ValueError('new audit file directly inside run required')
    report=audit_run(run)
    if not report['failed'] and not report['global_errors']:
        cases=[{'output_file':c['output_file'],'new_points':c['new_points'],'accepted_curves':c['accepted_curves'],'files':c['immutable_files']} for c in report['cases']]
        manifest={'version':read_json(HERE/'protocol.json')['version'],'run':str(run),'planned':report['planned'],
                  'case_count':len(cases),'file_count':sum(len(c['files']) for c in cases),'cases':cases,'incomplete_cases':[],
                  'scope':'Only immutable per-image predictions, saved fields and case records. Excludes audit, QA, diagnostics, reports and later generated files.'}
        path=run/'output_manifest.json'
        if path.exists():
            if read_json(path)!=manifest: issue(report['global_errors'],'existing_output_manifest_differs')
        else:
            with path.open('x',encoding='utf8') as f: json.dump(manifest,f,ensure_ascii=False,indent=2,allow_nan=False)
        report['output_manifest']={'path':str(path),'sha256':digest(path),'case_count':len(cases),'file_count':manifest['file_count']}
    with out.open('x',encoding='utf8') as f: json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ('planned','passed','failed','limited_body_cases','global_errors')},ensure_ascii=False),flush=True)
    return int(bool(report['failed'] or report['global_errors']))


if __name__=='__main__': raise SystemExit(main())
