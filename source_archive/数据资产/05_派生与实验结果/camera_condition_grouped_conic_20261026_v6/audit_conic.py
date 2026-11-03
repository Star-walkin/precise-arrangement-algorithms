"""独立核验C3两阶段契约：逐黄点归属、整组单曲线、光栅与B1不变。

不调用拟合器。以保存的点引用还原原B1输入，以系数重新计算每个点的残差。
本审计证明算法输出结构和像素约定，不证明分组就是物理上同一根钢丝绳。
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

BASE=Path(__file__).resolve().parent
ASSETS=BASE.parent
B1=ASSETS/'camera_condition_bright_rope_boundaries_20261025_v3/all_001'
DATA=ASSETS/'camera_condition_images_20261009_v2'
QA=ASSETS/'camera_condition_bright_rope_boundaries_20261009_v1/qa_plan.csv'
PRESERVED=['roi/reference_masks','roi/effective_masks','roi/effective_masked',
           'boundaries/observed','boundaries/incoming_rope_occlusion','boundaries/unknown_exclusion']

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def read(p,flag=0):
    a=cv2.imdecode(np.fromfile(str(p),np.uint8),flag)
    if a is None:raise ValueError('decode failed '+str(p))
    return a

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();run=args.run;out=args.output
    if out.exists():raise FileExistsError(out)
    summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    context=json.loads((run/'run_context.json').read_text(encoding='utf-8'))
    manifest=DATA/'manifest.csv'
    protocol=json.loads((BASE/'protocol.json').read_text(encoding='utf-8'))
    assert sha(manifest)==protocol['input_manifest_sha256']
    assert sha(QA)==protocol['qa_plan_sha256']
    rows=list(csv.DictReader(manifest.open(encoding='utf-8-sig')))
    qa={r['output_file'] for r in csv.DictReader(QA.open(encoding='utf-8-sig'))}
    if summary['mode']=='pilot':rows=[r for r in rows if r['output_file'] in qa]
    log=[json.loads(x) for x in (run/'run_log.jsonl').read_text(encoding='utf-8').splitlines()]
    global_errors=[]
    for source,expected_hash in context['source_sha256'].items():
        if sha(run/'source_snapshot'/Path(source).name)!=expected_hash:
            global_errors.append('source snapshot mismatch '+source)
    if Counter(r['output_file'] for r in rows)!=Counter(r['output_file'] for r in log):global_errors.append('input accounting mismatch')
    if summary['planned']!=len(rows):global_errors.append('summary denominator mismatch')
    b1audit=json.loads((B1/'integrity_audit.json').read_text(encoding='utf-8'))
    assert b1audit['passed']==400 and b1audit['failed']==0
    oldhash={r['path']:r['sha256'] for r in b1audit['file_manifest'] if r['status']=='present'}
    findings=[];file_manifest=[]
    for i,row in enumerate(rows,1):
        errors=[];close=[];cross=[]
        rel=Path(row['output_file']);role=rel.parent.name;name=rel.name
        try:
            if sha(DATA/rel)!=row['output_sha256']:errors.append('original source hash changed')
            paths=[]
            for directory in PRESERVED:
                old=B1/directory/role/name;new=run/directory/role/name;paths.append(new)
                if sha(new)!=oldhash[str(old)] or sha(old)!=oldhash[str(old)]:errors.append('B1 byte change '+directory)
            recordpath=run/'boundaries/records'/role/name.replace('.png','.json');paths.append(recordpath)
            z=json.loads(recordpath.read_text(encoding='utf-8'));record=z['record'];curves=z['fit']['curves'];pitch=record['pitch_px']
            oldpath=B1/'boundaries/records'/role/name.replace('.png','.json')
            old=json.loads(oldpath.read_text(encoding='utf-8'))
            inputs={(line['line_id'],j):p for line in old['lines'] for j,p in enumerate(line['observed_points_xy'])}
            groups=z['fit'].get('groups',[]);seen=[];groupids=[];accepted_ids=[]
            curve_by_id={c['curve_id']:c for c in curves}
            if len(curve_by_id)!=len(curves):errors.append('duplicate curve IDs')
            for group in groups:
                groupids.append(group['group_id'])
                refs=[(p['fragment_id'],p['point_index']) for p in group['point_refs']]
                seen.extend(refs)
                points=np.asarray(group['observed_points_xy'],float).reshape(-1,2)
                if len(refs)!=len(points):errors.append('point reference/coordinate count mismatch');continue
                if any(ref not in inputs for ref in refs):errors.append('unknown point reference');continue
                if not np.array_equal(points,np.asarray([inputs[ref] for ref in refs],float).reshape(-1,2)):
                    errors.append('group point changed from B1')
                if set(group['source_fragment_ids'])!={ref[0] for ref in refs}:errors.append('source fragment IDs mismatch')
                if not group.get('curve_id'):continue
                accepted_ids.append(group['curve_id'])
                c=curve_by_id.get(group['curve_id'])
                if c is None or c['group_id']!=group['group_id']:errors.append('group/curve association mismatch');continue
                weights=np.asarray(group['fit_weights'],float);residuals=np.asarray(group['residuals_px'],float)
                if len(weights)!=len(points) or np.any(~np.isfinite(weights)) or np.any(weights<=0):
                    errors.append('not all group points have positive fit weight')
                m=c['model'];u=(points[:,0]-m['u_center_x_px'])/m['u_scale_x_px']
                predicted=m['normalized_k']+m['normalized_b']*u+m['normalized_c']*u*u
                if residuals.shape!=(len(points),) or not np.allclose(residuals,points[:,1]-predicted,rtol=1e-7,atol=1e-6):
                    errors.append('all-point residual mismatch')
                # 独立重放最终目标函数：所有点的实际权重必须能重新求得保存系数。
                reg=group['fit_regularization'];design=np.column_stack([np.ones(len(points)),u,u*u])
                normal=design.T@(weights[:,None]*design)
                target=design.T@(weights*points[:,1])
                normal[1,1]+=reg['lambda_b'];normal[2,2]+=reg['lambda_c']
                target[1]+=reg['lambda_b']*reg['prior_b'];target[2]+=reg['lambda_c']*reg['prior_c']
                solution=np.linalg.solve(normal,target)
                if solution[2]<0:
                    solution=np.r_[np.linalg.solve(normal[:2,:2],target[:2]),0.]
                saved=np.asarray([m['normalized_k'],m['normalized_b'],m['normalized_c']])
                if not np.allclose(solution,saved,rtol=1e-7,atol=1e-5):errors.append('not the saved ALL-point weighted optimum')
                span=[float(points[:,0].min()),float(points[:,0].max())]
                if not np.allclose(c['observed_span_x_px'],span):errors.append('curve not using whole group span')
            if len(set(groupids))!=len(groupids):errors.append('duplicate group IDs')
            if Counter(seen)!=Counter(inputs.keys()):errors.append('input points not partitioned exactly once')
            if Counter(accepted_ids)!=Counter(curve_by_id.keys()):errors.append('not exactly one curve per accepted group')
            if record['grouped_point_count']!=len(inputs):errors.append('group point accounting mismatch')
            masks={k:read(run/'boundaries'/k/role/name) for k in ['smooth_boundaries','fit_near_observations','geometric_gap']}
            paths.extend(run/'boundaries'/k/role/name for k in masks)
            roi=read(run/'roi/reference_masks'/role/name);effective=read(run/'roi/effective_masks'/role/name)
            for key,a in masks.items():
                if a.shape!=(3000,4096) or a.dtype!=np.uint8 or np.any((a!=0)&(a!=255)):errors.append('binary violation '+key)
                if np.any((a>0)&(effective==0)):errors.append('out of effective ROI '+key)
            smooth=masks['smooth_boundaries'];near=masks['fit_near_observations'];gap=masks['geometric_gap']
            if np.any((near>0)&(gap>0)) or not np.array_equal(np.maximum(near,gap),smooth):errors.append('fit/gap partition mismatch')
            expected=np.zeros_like(roi);arrays=[]
            for c in curves:
                points=np.asarray(c['sampled_points_xy'],np.int32).reshape(-1,2);arrays.append(points)
                if not len(points):errors.append('empty curve');continue
                x,y=points[:,0],points[:,1]
                if np.any(np.diff(x)<=0):errors.append('curve x not strictly increasing')
                m=c['model'];u=(x-m['u_center_x_px'])/m['u_scale_x_px']
                yp=np.rint(m['normalized_k']+m['normalized_b']*u+m['normalized_c']*u*u).astype(int)
                if not np.array_equal(y,yp):errors.append('curve not on saved conic')
                if m['normalized_c']<0:errors.append('wrong opening direction')
                if x.min()<c['observed_span_x_px'][0] or x.max()>c['observed_span_x_px'][1]:errors.append('extrapolated outside observed span')
                # 接受曲线须完整绘制自身观测跨度，不得裁掉冲突像素来隐藏交叉。
                full_x=np.arange(int(np.ceil(c['observed_span_x_px'][0])),int(np.floor(c['observed_span_x_px'][1]))+1)
                fu=(full_x-m['u_center_x_px'])/m['u_scale_x_px']
                full_y=np.rint(m['normalized_k']+m['normalized_b']*fu+m['normalized_c']*fu*fu).astype(int)
                inside=(full_y>=0)&(full_y<effective.shape[0])&(full_x>=0)&(full_x<effective.shape[1])
                full_x,full_y=full_x[inside],full_y[inside]
                legal=effective[full_y,full_x]>0
                full_expected=np.column_stack([full_x[legal],full_y[legal]])
                if not np.array_equal(points,full_expected):errors.append('accepted whole curve has unexplained clipping')
                expected[y,x]=255
            if not np.array_equal(expected,smooth):errors.append('curve/raster mismatch')
            for aidx,a in enumerate(arrays):
                for bidx in range(aidx+1,len(arrays)):
                    b=arrays[bidx];xx,ia,ib=np.intersect1d(a[:,0],b[:,0],return_indices=True)
                    if len(xx)<10:continue
                    dy=a[ia,1]-b[ib,1]
                    if np.min(dy)<=0<=np.max(dy):cross.append([curves[aidx]['curve_id'],curves[bidx]['curve_id']])
                    elif np.min(np.abs(dy))<.45*pitch:close.append([curves[aidx]['curve_id'],curves[bidx]['curve_id'],int(np.min(np.abs(dy)))])
            # 近距对是几何诊断；交叉是硬错误。近距是否有合法片段原因需逐图查看。
            if cross:errors.append('crossing modeled courses')
            for directory in ['overlays','smooth_overlays','group_overlays']:
                paths.append(run/directory/role/name.replace('.png','.jpg'))
            if row['output_file'] in qa:paths.append(run/'qa_compare'/role/name.replace('.png','.jpg'))
            for p in paths:file_manifest.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'output_file':row['output_file']})
            for field,key in [('smooth_pixels','smooth_boundaries'),('gap_pixels','geometric_gap'),('near_observation_fit_pixels','fit_near_observations')]:
                if record[field]!=np.count_nonzero(masks[key]):errors.append('count mismatch '+field)
        except Exception as exc:errors.append(repr(exc))
        findings.append({'output_file':row['output_file'],'status':'fail' if errors else 'pass','errors':errors,
                         'crossings':cross,'close_pairs':close})
        if i%40==0:print('audited',i,'/',len(rows),flush=True)
    passed=sum(r['status']=='pass' for r in findings)
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'run':str(run),'audit_script_sha256':sha(Path(__file__)),
            'planned':len(rows),'passed':passed,'failed':len(rows)-passed,'global_errors':global_errors,
            'findings':findings,'file_manifest':file_manifest,'file_count':len(file_manifest),
            'images_with_close_pairs':sum(bool(r['close_pairs']) for r in findings),
            'meaning':'all-point partition, one-conic-per-group and pixel consistency only; not true boundary identity or overexposure accuracy',
            'exit_code':int(passed!=len(rows) or bool(global_errors))}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['planned','passed','failed','images_with_close_pairs','file_count','exit_code']}),flush=True)
    raise SystemExit(result['exit_code'])

if __name__=='__main__':main()
