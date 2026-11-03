"""在冻结 B1 黄点上拟合圆锥曲线；新输出另存，不重新检测黄点。"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
ASSETS=HERE.parent
B1=ASSETS/'camera_condition_bright_rope_boundaries_20261025_v3/all_001'
DATA=ASSETS/'camera_condition_images_20261009_v2'
MANIFEST=DATA/'manifest.csv'
QA=ASSETS/'camera_condition_bright_rope_boundaries_20261009_v1/qa_plan.csv'
PRESERVED=['roi/reference_masks','roi/effective_masks','roi/effective_masked',
           'boundaries/observed','boundaries/incoming_rope_occlusion','boundaries/unknown_exclusion']
_HASHES=None

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path,flag=cv2.IMREAD_COLOR):
    image=cv2.imdecode(np.fromfile(str(path),np.uint8),flag)
    if image is None:raise ValueError('Cannot decode '+str(path))
    return image

def write(path,image):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    ok,data=cv2.imencode(path.suffix,image,[cv2.IMWRITE_JPEG_QUALITY,94] if path.suffix=='.jpg' else [])
    if not ok:raise ValueError('Cannot encode '+str(path))
    data.tofile(str(path))

def dump(path,value):
    def convert(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,Path):return str(v)
        raise TypeError(type(v).__name__)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=convert,allow_nan=False)+'\n',encoding='utf-8')

def verified(path):
    global _HASHES
    if _HASHES is None:
        audit=json.loads((B1/'integrity_audit.json').read_text(encoding='utf-8'))
        _HASHES={r['path']:r['sha256'] for r in audit['file_manifest'] if r['status']=='present'}
    actual=sha(path)
    if actual!=_HASHES.get(str(path)):
        raise ValueError('B1 frozen input changed: '+str(path))
    return actual

def selection(mode):
    protocol=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
    assert sha(MANIFEST)==protocol['input_manifest_sha256']
    assert sha(QA)==protocol['qa_plan_sha256']
    rows=list(csv.DictReader(MANIFEST.open(encoding='utf-8-sig',newline='')))
    qa={r['output_file'] for r in csv.DictReader(QA.open(encoding='utf-8-sig',newline=''))}
    assert len(rows)==400 and len(qa)==38
    assert Counter(r['camera_role'] for r in rows)=={'left':200,'right':200}
    for row in rows:
        if '20260925' in row['source_file'] or any(t in row['dataset_use_status'].lower() for t in ['reserved','heldout']):
            raise ValueError('Reserved input rejected')
    return ([r for r in rows if r['output_file'] in qa] if mode=='pilot' else rows),qa

def display_border(roi):
    # 仅供显示：用每列上下限构造外包络，不改变精确 ROI 掩码。
    envelope=np.zeros_like(roi);ys,xs=np.nonzero(roi)
    if not len(xs):return envelope
    cols=[];tops=[];bottoms=[]
    for x in range(int(xs.min()),int(xs.max())+1):
        yy=np.flatnonzero(roi[:,x])
        if len(yy):cols.append(x);tops.append(yy[0]);bottoms.append(yy[-1])
    allcols=np.arange(cols[0],cols[-1]+1)
    top=np.interp(allcols,cols,tops).astype(np.float32)
    bottom=np.interp(allcols,cols,bottoms).astype(np.float32)
    top=cv2.GaussianBlur(top[None],(0,0),sigmaX=11)[0]
    bottom=cv2.GaussianBlur(bottom[None],(0,0),sigmaX=11)[0]
    for x,y0,y1 in zip(allcols,np.rint(top).astype(int),np.rint(bottom).astype(int)):
        envelope[max(0,y0):min(roi.shape[0],y1+1),x]=255
    contours,_=cv2.findContours(envelope,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    out=np.zeros_like(roi);cv2.drawContours(out,contours,-1,255,2)
    return out

def render(image,roi,obs,smooth,gap,rope,unknown,title,mode='evidence',maxwidth=1100):
    ys,xs=np.nonzero(roi)
    if not len(xs):return np.zeros((200,400,3),np.uint8)
    x0=max(0,int(xs.min())-40);x1=min(roi.shape[1],int(xs.max())+41)
    y0=max(0,int(ys.min())-40);y1=min(roi.shape[0],int(ys.max())+41)
    out=image[y0:y1,x0:x1].copy()
    if mode!='original':
        for mask,color in [(rope,(0,100,255)),(unknown,(255,100,0))]:
            hit=mask[y0:y1,x0:x1]>0;out[hit]=(out[hit]*.72+np.asarray(color)*.28).astype(np.uint8)
        # 线条可视化时画粗一像素；正式 mask 始终使用逐列单像素曲线。
        sm=cv2.dilate(smooth,np.ones((2,2),np.uint8)) if mode!='baseline' else smooth
        gm=cv2.dilate(gap,np.ones((2,2),np.uint8)) if mode!='baseline' else gap
        out[sm[y0:y1,x0:x1]>0]=(255,230,0)
        out[gm[y0:y1,x0:x1]>0]=(255,0,255)
        if mode in ('evidence','baseline'):out[obs[y0:y1,x0:x1]>0]=(0,255,255)
    out[display_border(roi)[y0:y1,x0:x1]>0]=(0,255,0)
    if out.shape[1]>maxwidth:
        out=cv2.resize(out,(maxwidth,round(out.shape[0]*maxwidth/out.shape[1])),interpolation=cv2.INTER_AREA)
    out=cv2.copyMakeBorder(out,32,0,0,0,cv2.BORDER_CONSTANT)
    cv2.putText(out,title,(7,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(240,240,240),1,cv2.LINE_AA)
    return out

def process(task):
    row,out,qa=task;out=Path(out);started=time.perf_counter();cv2.setNumThreads(1)
    try:
        from conic_family import fit_conic_family
        rel=Path(row['output_file']);role=rel.parent.name;name=rel.name
        imagepath=DATA/rel
        if sha(imagepath)!=row['output_sha256']:raise ValueError('Source image changed')
        image=read(imagepath)
        oldpath=B1/'boundaries/records'/role/name.replace('.png','.json')
        inputs={str(oldpath):verified(oldpath)}
        old=json.loads(oldpath.read_text(encoding='utf-8'))
        for directory in PRESERVED:
            src=B1/directory/role/name;inputs[str(src)]=verified(src)
        roi=read(B1/'roi/reference_masks'/role/name,0)
        effective=read(B1/'roi/effective_masks'/role/name,0)
        obs=read(B1/'boundaries/observed'/role/name,0)
        rope=read(B1/'boundaries/incoming_rope_occlusion'/role/name,0)
        unknown=read(B1/'boundaries/unknown_exclusion'/role/name,0)
        pitch=float(old['record']['pitch_px']);exclusion=np.maximum(rope,unknown)
        # 仅向拟合器传入观测点，避免读取旧紫线、family_id 或其他已拟合答案。
        lines=[{'line_id':line['line_id'],'observed_points_xy':line['observed_points_xy']}
               for line in old['lines']]
        fit=fit_conic_family(lines,pitch,roi,exclusion)
        smooth=np.zeros_like(roi)
        for curve in fit.get('curves',[]):
            for x,y in curve['sampled_points_xy']:
                if not (isinstance(x,int) and isinstance(y,int) and 0<=x<roi.shape[1] and 0<=y<roi.shape[0]):
                    raise ValueError('Invalid curve point')
                if effective[y,x]==0:raise ValueError('Curve outside effective ROI')
                smooth[y,x]=255
        if np.any(obs):
            distances=cv2.distanceTransform((obs==0).astype(np.uint8),cv2.DIST_L2,5)
            gap=np.where((smooth>0)&(distances>max(3.,.22*pitch)),255,0).astype(np.uint8)
        else:gap=smooth.copy()
        supported=np.where((smooth>0)&(gap==0),255,0).astype(np.uint8)
        # 三种正式结果分别保存：完整拟合线、靠近观测的拟合线、缺口几何补线。
        for directory in PRESERVED:
            dst=out/directory/role/name;dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(B1/directory/role/name,dst)
        for directory,mask in [('smooth_boundaries',smooth),('fit_near_observations',supported),('geometric_gap',gap)]:
            write(out/'boundaries'/directory/role/name,mask)
        fit={k:v for k,v in fit.items() if not isinstance(v,np.ndarray)}
        original_points=np.asarray([p for line in lines for p in line['observed_points_xy']],np.int32).reshape(-1,2)
        if np.any(smooth) and len(original_points):
            dist=cv2.distanceTransform((smooth==0).astype(np.uint8),cv2.DIST_L2,5)
            errors=dist[original_points[:,1],original_points[:,0]]
            matched=float(np.mean(errors<=max(3.,.20*pitch)))
            median=float(np.median(errors));p95=float(np.percentile(errors,95))
        else:matched=0.;median=None;p95=None
        record={'output_file':row['output_file'],'camera_role':row['camera_role'],
                'source_group':row['source_group'],'source_file':row['source_file'],'source_time_s':row['source_time_s'],
                'source_sha256':row['output_sha256'],'dataset_use_status':row['dataset_use_status'],
                'baseline':'RB-B1.0.0 / v3 all_001','input_dependency_sha256':inputs,'pitch_px':pitch,
                'baseline_yellow_points':len(original_points),'curve_count':len(fit.get('curves',[])),
                'smooth_pixels':int(np.count_nonzero(smooth)),'gap_pixels':int(np.count_nonzero(gap)),
                'near_observation_fit_pixels':int(np.count_nonzero(supported)),
                'yellow_near_curve_fraction':matched,'yellow_to_curve_median_px':median,'yellow_to_curve_p95_px':p95,
                'fit_status':fit.get('diagnostics',{}).get('status',fit.get('status')),
                'qa_selected':row['output_file'] in qa,
                'meaning':'Every smooth pixel is modeled; gap pixels are inferred, never direct evidence or material identity'}
        dump(out/'boundaries/records'/role/name.replace('.png','.json'),{'record':record,'fit':fit})
        zero=np.zeros_like(roi)
        write(out/'overlays'/role/name.replace('.png','.jpg'),render(image,roi,obs,smooth,gap,rope,unknown,'C2 | yellow=B1 points cyan=fit magenta=gap'))
        write(out/'smooth_overlays'/role/name.replace('.png','.jpg'),render(image,roi,obs,smooth,gap,rope,unknown,'C2 | smooth conic curves; magenta=inferred gap','smooth'))
        if row['output_file'] in qa:
            panels=[render(image,roi,obs,zero,zero,rope,unknown,'ORIGINAL','original',800),
                    render(image,roi,obs,zero,zero,rope,unknown,'B1 | frozen yellow observations','baseline',800),
                    render(image,roi,obs,smooth,gap,rope,unknown,'C2 | smooth fit and gap inference','smooth',800)]
            height=max(p.shape[0] for p in panels)
            write(out/'qa_compare'/role/name.replace('.png','.jpg'),np.concatenate([
                cv2.copyMakeBorder(p,0,height-p.shape[0],0,0,cv2.BORDER_CONSTANT) for p in panels],axis=1))
        record['status']='completed'
    except Exception as exc:
        import traceback
        record={'output_file':row['output_file'],'camera_role':row['camera_role'],'source_group':row['source_group'],
                'status':'error','error':repr(exc),'traceback':traceback.format_exc()}
    record['elapsed_seconds']=round(time.perf_counter()-started,3)
    return record

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['pilot','all'],required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,choices=[1,2],default=2)
    args=ap.parse_args();out=args.output
    if out.exists():raise FileExistsError('Refusing overwrite '+str(out))
    rows,qa=selection(args.mode)
    source=[HERE/'run_conic.py',HERE/'conic_family.py',HERE/'base_conic.py',HERE/'protocol.json']
    hashes={str(p):sha(p) for p in source}
    out.mkdir(parents=True);snapshot=out/'source_snapshot';snapshot.mkdir()
    for p in source:shutil.copyfile(p,snapshot/p.name)
    dump(out/'run_context.json',{'started_utc':datetime.now(timezone.utc).isoformat(),'command':sys.argv,
         'python':platform.python_version(),'executable':sys.executable,'numpy':np.__version__,'opencv':cv2.__version__,
         'source_sha256':hashes,'baseline_audit_sha256':sha(B1/'integrity_audit.json'),
         'manifest_sha256':sha(MANIFEST),'qa_sha256':sha(QA),'mode':args.mode,'workers':args.workers,
         'equipment_access':False,'reserved_inputs':False})
    tasks=[(r,str(out),qa) for r in rows];pool=ProcessPoolExecutor(max_workers=args.workers) if args.workers>1 else None
    results=pool.map(process,tasks) if pool else map(process,tasks);records=[]
    for i,r in enumerate(results,1):
        records.append(r)
        with (out/'run_log.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        print(json.dumps({k:r.get(k) for k in ['output_file','status','curve_count','gap_pixels','error']}|{'i':i,'total':len(rows)},ensure_ascii=False),flush=True)
    if pool:pool.shutdown()
    if hashes!={str(p):sha(p) for p in source}:raise RuntimeError('Source changed during run')
    good=[r for r in records if r['status']=='completed']
    summary={'finished_utc':datetime.now(timezone.utc).isoformat(),'planned':len(rows),'completed':len(good),
         'failed':len(rows)-len(good),'mode':args.mode,'roles':dict(Counter(r['camera_role'] for r in good)),
         'nonempty_curve_images':sum(r['curve_count']>0 for r in good),'images_with_gap':sum(r['gap_pixels']>0 for r in good),
         'curve_count':sum(r['curve_count'] for r in good),'smooth_pixels':sum(r['smooth_pixels'] for r in good),
         'gap_pixels':sum(r['gap_pixels'] for r in good),'errors':[r for r in records if r['status']!='completed'],
         'exit_code':int(len(good)!=len(rows)),'meaning':'modeled geometry and descriptive coverage, not hidden ground truth'}
    dump(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
    raise SystemExit(summary['exit_code'])

if __name__=='__main__':main()
