"""C4完整卷面离线批处理。SAM原色ROI与边界叠图分开保存，不改原图或旧基准。"""
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
ROOT=ASSETS.parents[1]
DATA=ASSETS/'camera_condition_images_20261009_v2'
V3=ASSETS/'camera_condition_bright_rope_boundaries_20261025_v3'
B1=V3/'all_001'
C3=ASSETS/'camera_condition_grouped_conic_20261026_v6/all_001'
SAM=ASSETS/'camera_condition_sam2_20261009_v1/all_001'
SAM_FINAL=SAM/'body_branches_002'
MANIFEST=DATA/'manifest.csv'
QA=ASSETS/'camera_condition_bright_rope_boundaries_20261009_v1/qa_plan.csv'
SAM_MANIFEST=SAM/'independent_final_audit_001/output_manifest.csv'
_INDEX=None
SAM_INDEX=None
sys.path.insert(0,str(V3))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def read(path,flag=cv2.IMREAD_COLOR):
    x=cv2.imdecode(np.fromfile(str(path),np.uint8),flag)
    if x is None:raise ValueError('Decode failed '+str(path))
    return x

def write(path,array):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    ok,data=cv2.imencode(path.suffix,array,[cv2.IMWRITE_JPEG_QUALITY,94] if path.suffix=='.jpg' else [])
    if not ok:raise ValueError('Encode failed '+str(path))
    data.tofile(str(path))

def dump(path,data):
    def cast(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,Path):return str(v)
        raise TypeError(type(v).__name__)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=cast,allow_nan=False)+'\n',encoding='utf-8')

def verified(path):
    global _INDEX
    if _INDEX is None:
        _INDEX={}
        for root in (B1,C3):
            d=json.loads((root/'integrity_audit.json').read_text(encoding='utf-8'))
            _INDEX.update({r['path']:r['sha256'] for r in d['file_manifest']})
    actual=sha(path)
    if actual!=_INDEX.get(str(path)):raise ValueError('Frozen source changed '+str(path))
    return actual

def sam_index():
    global SAM_INDEX
    if SAM_INDEX is None:
        SAM_INDEX={'images/'+r['output_file']:r for r in csv.DictReader(SAM_MANIFEST.open(encoding='utf-8-sig',newline=''))}
    return SAM_INDEX

def selection(mode):
    p=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
    for f,k in [(MANIFEST,'input_manifest_sha256'),(QA,'qa_plan_sha256'),(SAM_MANIFEST,'sam_output_manifest_sha256')]:
        if sha(f)!=p[k]:raise ValueError('Manifest changed '+str(f))
    rows=list(csv.DictReader(MANIFEST.open(encoding='utf-8-sig',newline='')))
    qa={r['output_file'] for r in csv.DictReader(QA.open(encoding='utf-8-sig',newline=''))}
    assert len(rows)==400 and len(qa)==38
    assert Counter(r['camera_role'] for r in rows)=={'left':200,'right':200}
    for r in rows:
        if '20260925' in r['source_file']+r['output_file'] or any(s in r['dataset_use_status'].lower() for s in ['reserved','heldout']):
            raise ValueError('Reserved input rejected')
    return ([r for r in rows if r['output_file'] in qa] if mode=='pilot' else rows),qa

def binary(m,shape):
    if m.shape!=shape or m.dtype!=np.uint8 or np.any((m!=0)&(m!=255)):
        raise ValueError('Expected same-size uint8 0/255 mask')

def contour(mask):
    c,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    out=np.zeros_like(mask);cv2.drawContours(out,c,-1,255,2)
    return out

def overlay(image,surface,bright,obs,dark,modeled,inferred,rope,unknown,show_points=True):
    # 全分辨率输出，黑底保留SAM内原色；可视化线条加粗，正式mask仍逐像素。
    out=cv2.bitwise_and(image,image,mask=surface)
    for m,color in [(rope,(0,100,255)),(unknown,(255,100,0))]:
        hit=m>0;out[hit]=(out[hit]*.68+np.array(color)*.32).astype(np.uint8)
    out[cv2.dilate(modeled,np.ones((2,2),np.uint8))>0]=(255,225,0)
    out[cv2.dilate(inferred,np.ones((2,2),np.uint8))>0]=(255,0,255)
    if show_points:
        out[obs>0]=(0,255,255)
        out[cv2.dilate(dark,np.ones((2,2),np.uint8))>0]=(0,180,255)
    out[contour(surface)>0]=(0,255,0)
    out[(contour(bright)>0)&(surface>0)]=(190,190,190)
    return out

def preview(image,surface,title,width=1350):
    yy,xx=np.nonzero(surface)
    if not len(xx):return np.zeros((200,500,3),np.uint8)
    box=(max(0,int(xx.min())-25),max(0,int(yy.min())-25),
         min(image.shape[1],int(xx.max())+26),min(image.shape[0],int(yy.max())+26))
    x0,y0,x1,y1=box;out=image[y0:y1,x0:x1].copy()
    if out.shape[1]>width:out=cv2.resize(out,(width,round(out.shape[0]*width/out.shape[1])),interpolation=cv2.INTER_AREA)
    out=cv2.copyMakeBorder(out,34,0,0,0,cv2.BORDER_CONSTANT)
    cv2.putText(out,title,(8,23),cv2.FONT_HERSHEY_SIMPLEX,.52,(240,240,240),1,cv2.LINE_AA)
    return out

def group_image(image,surface,fit):
    out=cv2.bitwise_and(image,image,mask=surface)
    for i,g in enumerate(fit.get('groups',[])):
        pts=np.asarray(g.get('observed_points_xy',[]),float).reshape(-1,2)
        if not len(pts):continue
        color=tuple(int(v) for v in cv2.cvtColor(np.uint8([[[i*67%180,230,255]]]),cv2.COLOR_HSV2BGR)[0,0])
        if g.get('status')=='rejected':color=(145,145,145)
        for x,y in np.rint(pts).astype(int):cv2.circle(out,(x,y),3,color,-1)
        label=str(g.get('group_id',i))
        for j in {int(np.argmin(pts[:,0])),int(np.argmax(pts[:,0]))}:
            x,y=np.rint(pts[j]).astype(int)
            cv2.putText(out,label,(max(0,x-10),max(14,y-6)),cv2.FONT_HERSHEY_SIMPLEX,.40,(0,0,0),3)
            cv2.putText(out,label,(max(0,x-10),max(14,y-6)),cv2.FONT_HERSHEY_SIMPLEX,.40,color,1)
    out[contour(surface)>0]=(0,255,0)
    return preview(out,surface,'C4 | grouped original + screened dark observations')

def process(task):
    row,out,qa=task;out=Path(out);start=time.perf_counter();cv2.setNumThreads(1)
    try:
        from free_rope_v3 import estimate_occlusion
        from full_surface_core import fit_full_surface
        rel=Path(row['output_file']);role=rel.parent.name;name=rel.name;stem=rel.stem
        if sha(DATA/rel)!=row['output_sha256']:raise ValueError('Original image hash mismatch')
        image=read(DATA/rel);assert image.shape==(3000,4096,3)
        source=sam_index()[row['output_file']]
        deps={}
        for k in ['mask','masked']:
            p=Path(source[k+'_path']);h=sha(p)
            if h!=source[k+'_sha256']:raise ValueError('SAM output changed '+str(p))
            deps[str(p)]=h
        sam_surface=read(source['mask_path'],0)
        c3path=C3/'boundaries/records'/role/(stem+'.json')
        deps[str(c3path)]=verified(c3path)
        seed=json.loads(c3path.read_text(encoding='utf-8'))
        pitch=float(seed['record']['pitch_px'])
        preserved={}
        for key,d in [('bright','roi/reference_masks'),('old_obs','boundaries/observed'),
                      ('old_rope','boundaries/incoming_rope_occlusion'),('old_unknown','boundaries/unknown_exclusion')]:
            p=B1/d/role/name;deps[str(p)]=verified(p);preserved[key]=read(p,0)
        bright=preserved['bright'];original_obs=preserved['old_obs']
        # 第二轮修正：扩大工作区不能裁掉用户已接受的亮区。原SAM另存，
        # 所有图片采用同一并集规则；不按评分或人工逐图勾选改变ROI。
        surface=np.maximum(sam_surface,bright)
        binary(surface,image.shape[:2]);binary(bright,surface.shape)
        # 旧掩码被亮ROI裁过，必须在新SAM区域重跑，不能把暗区原零值当无绳。
        new_rope,new_unknown,occlusion=estimate_occlusion(image,surface,row['camera_role'],pitch)
        outside=bright==0
        rope=np.where(outside,new_rope,preserved['old_rope']).astype(np.uint8)
        unknown=np.where(outside,new_unknown,preserved['old_unknown']).astype(np.uint8)
        rope[surface==0]=0;unknown[(surface==0)|(rope>0)]=0
        exclusion=np.maximum(rope,unknown)
        effective=np.where((surface>0)&(exclusion==0),255,0).astype(np.uint8)
        fit=fit_full_surface(image,surface,bright,seed['fit'],exclusion,pitch)
        dark=fit['dark_observed_mask'];modeled=fit['modeled_mask']
        inferred=fit['inferred_mask'];simple=fit['simple_c3_extension_mask']
        for m in [rope,unknown,effective,dark,modeled,inferred,simple]:
            binary(m,surface.shape)
        for m in [dark,modeled,inferred,simple]:
            if np.any((m>0)&(effective==0)):raise ValueError('Output outside effective full surface')
        if np.any((inferred>0)&(modeled==0)):raise ValueError('Inference outside modeled mask')
        copy_inputs=[(Path(source['mask_path']),'roi/sam_reference_masks'),
                     (Path(source['masked_path']),'roi/sam_reference_masked'),
                     (B1/'roi/reference_masks'/role/name,'roi/bright_reference_masks'),
                     (B1/'boundaries/observed'/role/name,'boundaries/original_observed')]
        for src,directory in copy_inputs:
            dst=out/directory/role/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        write(out/'roi/surface_masks'/role/name,surface)
        write(out/'roi/surface_masked'/role/name,cv2.bitwise_and(image,image,mask=surface))
        masks={'roi/effective_masks':effective,'boundaries/dark_observed':dark,'boundaries/modeled':modeled,
               'boundaries/inferred':inferred,'boundaries/simple_c3_extension':simple,
               'boundaries/incoming_rope_occlusion':rope,'boundaries/unknown_exclusion':unknown}
        for d,m in masks.items():write(out/d/role/name,m)
        clean_fit={k:v for k,v in fit.items() if not isinstance(v,np.ndarray)}
        points=np.asarray([p for g in seed['fit'].get('groups',[]) for p in g.get('observed_points_xy',[])],int).reshape(-1,2)
        if len(points) and np.any(modeled):
            distance=cv2.distanceTransform((modeled==0).astype(np.uint8),cv2.DIST_L2,5)
            err=distance[points[:,1],points[:,0]]
            bright_hit=float(np.mean(err<=.25*pitch));bright_p95=float(np.percentile(err,95))
        else:bright_hit=0.;bright_p95=None
        record={k:row[k] for k in ['output_file','camera_role','source_group','source_file','source_time_s','dataset_use_status']}
        record.update({'version':'RB-C4.0.0','baseline':'RB-C3.0.0','source_sha256':row['output_sha256'],
            'input_dependency_sha256':deps,'pitch_px':pitch,'qa_selected':row['output_file'] in qa,
            'roi_policy':'SAM2_v2_union_B1_bright',
            'sam_raw_pixels':int(np.count_nonzero(sam_surface)),
            'union_added_bright_pixels':int(np.count_nonzero((bright>0)&(sam_surface==0))),
            'surface_pixels':int(np.count_nonzero(surface)),'bright_pixels_inside_surface':int(np.count_nonzero((surface>0)&(bright>0))),
            'expanded_pixels':int(np.count_nonzero((surface>0)&outside)),
            'dark_observed_pixels':int(np.count_nonzero(dark)),'modeled_pixels':int(np.count_nonzero(modeled)),
            'modeled_outside_bright_pixels':int(np.count_nonzero((modeled>0)&outside)),
            'inferred_pixels':int(np.count_nonzero(inferred)),
            'simple_c3_extension_pixels':int(np.count_nonzero(simple)),
            'rope_pixels':int(np.count_nonzero(rope)),'unknown_pixels':int(np.count_nonzero(unknown)),
            'outside_bright_rope_pixels':int(np.count_nonzero((rope>0)&outside)),
            'outside_bright_unknown_pixels':int(np.count_nonzero((unknown>0)&outside)),
            'bright_yellow_points':len(points),'bright_yellow_near_curve_fraction':bright_hit,'bright_yellow_to_curve_p95_px':bright_p95,
            'group_count':len(clean_fit.get('groups',[])),
            'fit_diagnostics':clean_fit.get('diagnostics',{}),'tangency_coverage':'not_evaluable_no_contact_labels',
            'complete_free_rope_claim':False})
        dump(out/'boundaries/records'/role/(stem+'.json'),{'record':record,'fit':clean_fit,'occlusion_diagnostics':occlusion})
        full=overlay(image,surface,bright,original_obs,dark,modeled,inferred,rope,unknown,True)
        write(out/'overlays'/role/name,full)
        write(out/'previews'/role/(stem+'.jpg'),preview(full,surface,'C4 | full SAM surface; yellow=original, amber=screened dark, magenta=inferred'))
        smooth=overlay(image,surface,bright,original_obs,dark,modeled,inferred,rope,unknown,False)
        write(out/'smooth_previews'/role/(stem+'.jpg'),preview(smooth,surface,'C4 | one conic per group; cyan=near evidence, magenta=inferred'))
        write(out/'group_overlays'/role/(stem+'.jpg'),group_image(image,surface,clean_fit))
        if row['output_file'] in qa:
            old_model=read(C3/'boundaries/smooth_boundaries'/role/name,0)
            zero=np.zeros_like(surface)
            base=overlay(image,surface,bright,zero,zero,old_model,zero,rope,unknown,False)
            sm=overlay(image,surface,bright,zero,zero,simple,simple,rope,unknown,False)
            panels=[preview(cv2.bitwise_and(image,image,mask=surface),surface,'Full surface = SAM2 + preserved bright ROI',850),
                    preview(base,surface,'C3 bright ROI baseline',850),
                    preview(sm,surface,'Simple C3 coefficient extension',850),
                    preview(full,surface,'C4 screened dark + whole conic',850)]
            height=max(p.shape[0] for p in panels)
            write(out/'qa_compare'/role/(stem+'.jpg'),np.concatenate([cv2.copyMakeBorder(p,0,height-p.shape[0],0,0,cv2.BORDER_CONSTANT) for p in panels],axis=1))
        record['status']='completed'
    except Exception as e:
        import traceback
        record={'output_file':row['output_file'],'camera_role':row['camera_role'],'status':'error','error':repr(e),'traceback':traceback.format_exc()}
    record['elapsed_seconds']=round(time.perf_counter()-start,3)
    return record

def source_paths():
    return [HERE/'run_full_surface.py',HERE/'full_surface_core.py',HERE/'protocol.json',HERE/'protocol_revision_002.json',HERE/'evaluation_protocol.json',
            V3/'free_rope_v3.py',ASSETS/'camera_condition_bright_rope_boundaries_20261023_v2/free_rope_occlusion.py',
            ROOT/'vision_geometry_lab/rope_detector.py',ROOT/'vision_prototype/semantic_locator.py',
            ASSETS/'camera_condition_grouped_conic_20261026_v6/conic_family.py']

def model_paths():
    dirs=[ROOT/'vision_training/task_left_v1/models/left_camera_thick_v1',ROOT/'vision_training/task_v1/models/right_camera_poc_v1']
    return [p for d in dirs for p in [d/(d.name+'.onnx'),d/'onnx_metadata.json']]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['pilot','all'],required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,choices=[1,2],default=2)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    rows,qa=selection(a.mode);sources=source_paths();hashes={str(p):sha(p) for p in sources}
    a.output.mkdir(parents=True);snap=a.output/'source_snapshot';snap.mkdir()
    for p in sources:shutil.copyfile(p,snap/p.name)
    context={'started_utc':datetime.now(timezone.utc).isoformat(),'command':sys.argv,'working_directory':str(Path.cwd()),
             'python':platform.python_version(),'executable':sys.executable,'numpy':np.__version__,'opencv':cv2.__version__,
             'source_sha256':hashes,'model_sha256':{str(p):sha(p) for p in model_paths()},
             'manifest_sha256':sha(MANIFEST),'sam_manifest_sha256':sha(SAM_MANIFEST),'qa_sha256':sha(QA),
             'mode':a.mode,'workers':a.workers,'roi_policy':'SAM2_v2_union_B1_bright','equipment_access':False,'reserved_inputs':False}
    dump(a.output/'run_context.json',context)
    tasks=[(r,str(a.output),qa) for r in rows];records=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,r in enumerate(pool.map(process,tasks),1):
            records.append(r)
            with (a.output/'run_log.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
            print(json.dumps({'i':i,'n':len(rows),'output_file':r['output_file'],'status':r['status'],
                              'dark_observed_pixels':r.get('dark_observed_pixels'),'modeled_pixels':r.get('modeled_pixels'),
                              'error':r.get('error')},ensure_ascii=False),flush=True)
    unchanged=hashes=={str(p):sha(p) for p in sources}
    good=[r for r in records if r['status']=='completed']
    summary={'finished_utc':datetime.now(timezone.utc).isoformat(),'planned':len(rows),'completed':len(good),
             'failed':len(rows)-len(good),'mode':a.mode,'source_unchanged':unchanged,
             'roles':dict(Counter(r['camera_role'] for r in good)),
             'nonempty_curve_images':sum(r['modeled_pixels']>0 for r in good),
             'images_with_dark_observations':sum(r['dark_observed_pixels']>0 for r in good),
             'images_with_extended_models':sum(r['modeled_outside_bright_pixels']>0 for r in good)}
    for k in ['sam_raw_pixels','union_added_bright_pixels','surface_pixels','expanded_pixels','dark_observed_pixels','modeled_pixels','modeled_outside_bright_pixels',
              'inferred_pixels','rope_pixels','unknown_pixels','outside_bright_rope_pixels','outside_bright_unknown_pixels']:
        summary[k]=sum(r[k] for r in good)
    summary.update({'errors':[r for r in records if r['status']!='completed'],
                    'exit_code':int(len(good)!=len(rows) or not unchanged),
                    'meaning':'output quantity and geometric consistency, not full-surface truth or physical tangency coverage'})
    dump(a.output/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
    raise SystemExit(summary['exit_code'])

if __name__=='__main__':main()
