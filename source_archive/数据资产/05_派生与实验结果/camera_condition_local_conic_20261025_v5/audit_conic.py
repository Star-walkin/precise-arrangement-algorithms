"""只读核验 C2 文件、B1不变性、圆锥曲线光栅与像素约定；不调用拟合器。"""
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
            for directory in ['overlays','smooth_overlays']:
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
            'meaning':'file and conic consistency only; not real overexposed ground truth or material identity',
            'exit_code':int(passed!=len(rows) or bool(global_errors))}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['planned','passed','failed','images_with_close_pairs','file_count','exit_code']}),flush=True)
    raise SystemExit(result['exit_code'])

if __name__=='__main__':main()
