"""固定开放图片集的v3离线运行器。旧ROI只读；各算法输出另存。"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent
ROOT = ASSETS.parents[1]
DATA = ASSETS / 'camera_condition_images_20261009_v2'
V1 = ASSETS / 'camera_condition_bright_rope_boundaries_20261009_v1'
V2 = ASSETS / 'camera_condition_bright_rope_boundaries_20261023_v2'
MANIFEST = DATA / 'manifest.csv'
QA_PLAN = V1 / 'qa_plan.csv'
MODULES = ['run_v3.py', 'full_roi_observation.py', 'free_rope_v3.py', 'curve_family_v3.py']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()

def read(path, flag=cv2.IMREAD_COLOR):
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), flag)
    if image is None:
        raise ValueError(f'Image decode failed: {path}')
    return image

def write(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix, image,
        [cv2.IMWRITE_JPEG_QUALITY, 92] if path.suffix == '.jpg' else [])
    if not ok:
        raise ValueError(path)
    data.tofile(str(path))

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def plan(mode):
    protocol = json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
    if sha(MANIFEST) != protocol['input_manifest_sha256'] or sha(QA_PLAN) != protocol['fixed_qa_plan_sha256']:
        raise ValueError('Frozen input or QA plan changed')
    with MANIFEST.open(encoding='utf-8-sig', newline='') as f:
        rows=list(csv.DictReader(f))
    with QA_PLAN.open(encoding='utf-8-sig', newline='') as f:
        qa={r['output_file'] for r in csv.DictReader(f)}
    if len(rows)!=400 or Counter(r['camera_role'] for r in rows)!={'left':200,'right':200} or len(qa)!=38:
        raise ValueError('Unexpected selection')
    for r in rows:
        if ('20260925' in r['source_file']+r['output_file'] or
                any(s in r['dataset_use_status'].lower() for s in ['reserved','heldout'])):
            raise ValueError('Reserved input in plan')
    return ([r for r in rows if r['output_file'] in qa] if mode=='pilot' else rows), qa

def binary(mask, shape, name):
    if mask.shape!=shape or mask.dtype!=np.uint8 or np.any((mask!=0)&(mask!=255)):
        raise ValueError(f'{name}: wrong shape/dtype/values')

def points_of(lines):
    points=[p for line in lines for p in line.get('observed_points_xy', [])]
    return np.asarray(points, np.int32).reshape(-1,2)

def agreement(points, target, tolerance):
    if not len(points):
        return None
    if not np.any(target):
        return 0.0
    distance=cv2.distanceTransform((target==0).astype(np.uint8), cv2.DIST_L2, 3)
    return float(np.mean(distance[points[:,1],points[:,0]]<=tolerance))

def crop_box(roi, pad=55):
    y,x=np.nonzero(roi)
    return (max(0,int(x.min())-pad),max(0,int(y.min())-pad),
            min(roi.shape[1],int(x.max())+pad+1),min(roi.shape[0],int(y.max())+pad+1))

def display_contour(roi, pitch):
    # 只画外包络；不拿显示用填洞结果参与任何边界识别。
    y,x=np.nonzero(roi); envelope=np.zeros_like(roi)
    if not len(x): return envelope
    cols=np.arange(x.min(),x.max()+1)
    top=[];bottom=[];valid=[]
    for col in cols:
        ys=np.flatnonzero(roi[:,col])
        if len(ys)>=5:
            valid.append(col);top.append(ys[0]);bottom.append(ys[-1])
    if len(valid)<4:return envelope
    t=np.interp(cols,valid,top).astype(np.float32)
    d=np.interp(cols,valid,bottom).astype(np.float32)
    for values in (t,d):
        values[:]=cv2.GaussianBlur(values[None],(0,0),sigmaX=max(2,min(20,pitch*.5)))[0]
    for col,lo,hi in zip(cols,np.rint(t).astype(int),np.rint(d).astype(int)):
        envelope[max(0,lo):min(roi.shape[0],hi+1),col]=255
    return cv2.morphologyEx(envelope,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))

def render(image, roi, observed, inferred, rope, unknown, title, max_width=1250):
    box=crop_box(roi); x0,y0,x1,y1=box
    out=image[y0:y1,x0:x1].copy()
    for mask,color in [(rope,(0,100,255)),(unknown,(255,100,0))]:
        selected=mask[y0:y1,x0:x1]>0
        out[selected]=(out[selected]*.65+np.array(color)*.35).astype(np.uint8)
    out[inferred[y0:y1,x0:x1]>0]=(255,0,255)
    out[observed[y0:y1,x0:x1]>0]=(0,255,255)
    contour=display_contour(roi,22.)
    out[contour[y0:y1,x0:x1]>0]=(0,255,0)
    if out.shape[1]>max_width:
        out=cv2.resize(out,(max_width,round(out.shape[0]*max_width/out.shape[1])),interpolation=cv2.INTER_AREA)
    out=cv2.copyMakeBorder(out,32,0,0,0,cv2.BORDER_CONSTANT)
    cv2.putText(out,title,(7,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(240,240,240),1,cv2.LINE_AA)
    return out

def compare(image,roi,old_obs,old_inf,old_rope,old_unknown,obs,inf,rope,unknown):
    zero=np.zeros_like(roi)
    panels=[render(image,roi,zero,zero,zero,zero,'ORIGINAL | same crop',850),
            render(image,roi,old_obs,old_inf,old_rope,old_unknown,'v2 | yellow=observed purple=inferred',850),
            render(image,roi,obs,inf,rope,unknown,'v3 | yellow=observed purple=inferred',850)]
    h=max(p.shape[0] for p in panels)
    return np.concatenate([cv2.copyMakeBorder(p,0,h-p.shape[0],0,0,cv2.BORDER_CONSTANT) for p in panels],axis=1)

def process(row,out,qa,detector,occluder,family):
    rel=Path(row['output_file']); role=rel.parent.name; name=rel.name
    if sha(DATA/rel)!=row['output_sha256']:
        raise ValueError('Source image hash mismatch')
    image=read(DATA/rel)
    roi_path=V1/'all_001/roi/masks'/role/name
    roi=read(roi_path,0)
    if image.shape!=(3000,4096,3):raise ValueError('Unexpected image dimensions')
    binary(roi,image.shape[:2],'reference ROI')
    old_json=V2/'all_002/boundaries/records'/role/name.replace('.png','.json')
    old=json.loads(old_json.read_text(encoding='utf-8'))
    # 仅复用旧的图像尺度估计。旧黄点/线号不会传给v3三个算法模块。
    pitch=old['record'].get('pitch_px')
    if pitch is None or not math.isfinite(pitch) or not 8<=pitch<=70:
        raise ValueError('No usable image pitch; explicit failure, no silent replacement')
    rope,unknown,rope_diag=occluder(image,roi,row['camera_role'],pitch)
    for key,m in [('rope',rope),('unknown',unknown)]:binary(m,roi.shape,key)
    if np.any((rope>0)&(unknown>0)) or np.any(((rope>0)|(unknown>0))&(roi==0)):
        raise ValueError('Occlusion contract violation')
    exclusion=np.maximum(rope,unknown)
    obs,lines,obs_diag=detector(image,roi,exclusion,pitch)
    inf,lines,family_diag=family(image,roi,obs,lines,rope,unknown,pitch)
    for key,m in [('observed',obs),('inferred',inf)]:binary(m,roi.shape,key)
    if np.any((obs>0)&((roi==0)|(exclusion>0))) or np.any((inf>0)&((roi==0)|(unknown>0)|(obs>0))):
        raise ValueError('Observation/inference contract violation')
    for line in lines:
        for field,mask in [('observed_points_xy',obs),('inferred_points_xy',inf)]:
            for x,y in line.get(field,[]):
                if not (0<=x<4096 and 0<=y<3000 and mask[y,x]==255):
                    raise ValueError(f'Point/raster mismatch: {field}, {x}, {y}')
    effective=np.where((roi>0)&(exclusion==0),255,0).astype(np.uint8)
    ref_out=out/'roi/reference_masks'/role/name;ref_out.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(roi_path,ref_out)
    images={'roi/effective_masks':effective,'roi/effective_masked':cv2.bitwise_and(image,image,mask=effective),
            'boundaries/observed':obs,'boundaries/inferred':inf,'boundaries/incoming_rope_occlusion':rope,
            'boundaries/unknown_exclusion':unknown,
            'boundaries/inferred_under_occlusion':np.where((inf>0)&(rope>0),255,0).astype(np.uint8)}
    for key,value in images.items():write(out/key/role/name,value)
    old_obs=read(V2/'all_002/boundaries/observed'/role/name,0)
    old_inf=read(V2/'all_002/boundaries/inferred'/role/name,0)
    old_rope=read(V2/'all_002/boundaries/incoming_rope_occlusion'/role/name,0)
    old_unknown=read(V2/'all_002/boundaries/unknown_exclusion'/role/name,0)
    new_points=points_of(lines);old_points=points_of(old['lines'])
    tolerance=max(3.,.18*pitch)
    record={
        'output_file':row['output_file'],'camera_role':row['camera_role'],
        'source_group':row['source_group'],'source_file':row['source_file'],'source_time_s':row['source_time_s'],
        'source_sha256':row['output_sha256'],'reference_roi_sha256':sha(roi_path),
        'dataset_use_status':row['dataset_use_status'],'pitch_px':pitch,'pitch_source':'v2 optical scale only; no line coordinates',
        'v2_observed_points':len(old_points),'v3_observed_points':len(new_points),
        'v2_points_near_v3_fraction':agreement(old_points,obs,tolerance),
        'v3_points_near_v2_fraction':agreement(new_points,old_obs,tolerance),
        'comparison_tolerance_px':tolerance,
        'observed_pixels':int(np.count_nonzero(obs)), 'inferred_pixels':int(np.count_nonzero(inf)),
        'inferred_under_occlusion_pixels':int(np.count_nonzero((inf>0)&(rope>0))),
        'rope_pixels':int(np.count_nonzero(rope)),'unknown_pixels':int(np.count_nonzero(unknown)),
        'fragment_or_family_count':len(lines),'occlusion_diagnostics':rope_diag,
        'observed_diagnostics':obs_diag,'family_diagnostics':family_diag,
        'qa_selected':row['output_file'] in qa,'interpretation':'visual candidates, point agreement is not accuracy',
    }
    dump(out/'boundaries/records'/role/name.replace('.png','.json'),{'record':record,'lines':lines})
    write(out/'overlays'/role/name.replace('.png','.jpg'),render(image,roi,obs,inf,rope,unknown,
          'v3 | yellow=observed purple=inferred orange=rope blue=unknown'))
    if row['output_file'] in qa:
        write(out/'qa_compare'/role/name.replace('.png','.jpg'),
              compare(image,roi,old_obs,old_inf,old_rope,old_unknown,obs,inf,rope,unknown))
    return record

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['pilot','all'],required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output
    if out.exists():raise FileExistsError('Refusing to overwrite '+str(out))
    rows,qa=plan(args.mode)
    from full_roi_observation import detect_full_roi
    from free_rope_v3 import estimate_occlusion
    from curve_family_v3 import infer_family
    cv2.setNumThreads(1)
    out.mkdir(parents=True)
    source_paths=[HERE/name for name in MODULES]+[
        ROOT/'vision_geometry_lab/rope_detector.py',
        ROOT/'vision_prototype/semantic_locator.py',V2/'free_rope_occlusion.py']
    model_paths=[ROOT/'vision_training/task_left_v1/models/left_camera_thick_v1/left_camera_thick_v1.onnx',
                 ROOT/'vision_training/task_v1/models/right_camera_poc_v1/right_camera_poc_v1.onnx']
    metadata_paths=[p.parent/'onnx_metadata.json' for p in model_paths]
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for path in source_paths:
        shutil.copyfile(path,snapshot/path.name)
    dump(out/'run_context.json',{
        'started_utc':datetime.now(timezone.utc).isoformat(),'command':sys.argv,
        'python':platform.python_version(),'executable':sys.executable,'opencv':cv2.__version__,'numpy':np.__version__,
        'mode':args.mode,'input_manifest_sha256':sha(MANIFEST),'qa_plan_sha256':sha(QA_PLAN),
        'protocol_sha256':sha(HERE/'protocol.json'),'source_sha256':{str(p):sha(p) for p in source_paths},
        'model_sha256':{str(p):sha(p) for p in model_paths},
        'model_metadata_sha256':{str(p):sha(p) for p in metadata_paths},
        'equipment_access':False,'reserved_input_access':False})
    records=[]
    for i,row in enumerate(rows,1):
        started=time.perf_counter()
        try:
            result=process(row,out,qa,detect_full_roi,estimate_occlusion,infer_family)
            result['status']='completed'
        except Exception as exc:
            import traceback
            result={'output_file':row['output_file'],'camera_role':row['camera_role'],
                    'source_group':row['source_group'],'status':'error',
                    'error':repr(exc),'traceback':traceback.format_exc()}
        result['elapsed_seconds']=round(time.perf_counter()-started,3)
        records.append(result)
        with (out/'run_log.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
        print(json.dumps({'i':i,'planned':len(rows),'image':row['output_file'],'status':result['status'],
                          'points':result.get('v3_observed_points'),'purple':result.get('inferred_pixels'),
                          'seconds':result['elapsed_seconds'],'error':result.get('error')},ensure_ascii=False),flush=True)
    good=[r for r in records if r['status']=='completed']
    groups=[]
    for role,group in sorted({(r['camera_role'],r['source_group']) for r in records}):
        rr=[r for r in records if r['camera_role']==role and r['source_group']==group]
        gg=[r for r in rr if r['status']=='completed']
        groups.append({'camera_role':role,'source_group':group,'planned':len(rr),'completed':len(gg),
                       'v2_points':sum(r['v2_observed_points'] for r in gg),'v3_points':sum(r['v3_observed_points'] for r in gg)})
    summary={'finished_utc':datetime.now(timezone.utc).isoformat(),'mode':args.mode,'planned':len(rows),
             'completed':len(good),'failed':len(rows)-len(good),'roles':dict(Counter(r['camera_role'] for r in good)),
             'groups':groups,'v2_observed_points':sum(r['v2_observed_points'] for r in good),
             'v3_observed_points':sum(r['v3_observed_points'] for r in good),
             'images_with_more_observed_points':sum(r['v3_observed_points']>r['v2_observed_points'] for r in good),
             'inferred_pixels':sum(r['inferred_pixels'] for r in good),
             'inferred_under_occlusion_pixels':sum(r['inferred_under_occlusion_pixels'] for r in good),
             'errors':[r for r in records if r['status']=='error'],
             'meaning':'descriptive coverage; not accuracy or rope identity; all previous failures retained',
             'exit_code':int(len(good)!=len(rows))}
    dump(out/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['planned','completed','failed','v2_observed_points','v3_observed_points']},ensure_ascii=False),flush=True)
    raise SystemExit(summary['exit_code'])

if __name__=='__main__':main()
