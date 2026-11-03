"""固定400图的活动绳离线批处理；先pilot、再冻结后all，旧产物不覆盖。"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import json
import platform
import shutil
import sys
import time
import traceback
import cv2
import numpy as np
from experiment_io import (HERE, MANIFEST, QA, BASELINE, select, inputs, dump,
                           source_inventory, sha, write_image, read_json)
from render_results import render, build_review, points

def json_safe(value):
    # 像素mask单独存PNG；禁止把整幅numpy mask混进JSON当作人工标签。
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items() if not isinstance(v, np.ndarray)}
    if isinstance(value, list): return [json_safe(v) for v in value]
    if isinstance(value, np.generic): return value.item()
    return value

def draw_free_support(shape, free):
    mask = np.zeros(shape, np.uint8)
    left = points(free.get('boundary_left_xy'))
    right = points(free.get('boundary_right_xy'))
    # 仅相邻有剖面支持的截面间画绳带。大缺口不被当作已检测像素。
    if len(left) == len(right) and len(left) >= 2:
        width = float(free.get('width_px') or 0)
        max_gap = max(35., min(100., 6*width))
        for i in range(len(left)-1):
            gap = np.linalg.norm((left[i+1]+right[i+1]-left[i]-right[i])*.5)
            if gap <= max_gap:
                poly = np.rint([left[i], right[i], right[i+1], left[i+1]]).astype(np.int32)
                cv2.fillConvexPoly(mask, poly, 255)
    return mask

def straight_baseline(free, surface):
    """公平保存最简单的直段延长对照；没有像素证据，也不输出接触点。"""
    direction=free.get('toward_surface_direction_xy')
    p=points(free.get('observed_points_xy'))
    width=free.get('width_px')
    if direction is None or width is None or len(p)<2:
        return {'status':'unavailable','path_xy':[],'contact_point_xy':None}
    d=np.asarray(direction,float);d/=max(np.linalg.norm(d),1e-9)
    entry=p[int(np.argmax(p@d))]
    t=np.arange(0.,min(90*float(width),1800.)+1,4.)
    path=entry+t[:,None]*d
    h,w=surface.shape
    valid=(path[:,0]>=0)&(path[:,0]<w)&(path[:,1]>=0)&(path[:,1]<h)
    path=path[valid]
    # 光线进入SAM掩码的投影位置只作几何诊断，绝不是物理接触标签。
    ij=np.floor(path).astype(int)
    on=surface[ij[:,1],ij[:,0]]>0 if len(path) else np.zeros(0,bool)
    return {'status':'geometry_only','path_xy':path.round(2).tolist(),
            'on_surface':on.tolist(),'contact_point_xy':None,
            'meaning':'Finite straight projection only; no original-image observation or contact evidence'}

def process(task):
    row, out, qa = task; out = Path(out); start=time.perf_counter()
    cv2.setNumThreads(1)
    rel=Path(row['output_file']);role=rel.parent.name;stem=rel.stem
    base = {k: row[k] for k in ['output_file','camera_role','source_group','source_file',
                               'source_time_s','dataset_use_status']}
    base.update({'qa_selected':row['output_file'] in qa,'version':'IR-A1.0.0',
                 'source_png_sha256':row['output_sha256']})
    try:
        from free_segment import detect_free_segment
        from connection_core import find_connections
        image,surface,c3,c4,deps = inputs(row)
        free = detect_free_segment(image,surface,row['camera_role'],
                  prior={'occlusion_diagnostics':c4['occlusion_diagnostics'],
                         'pitch_px':float(c3['record']['pitch_px'])})
        baseline=straight_baseline(free,surface)
        connection = find_connections(image,surface,free,c3['fit'])
        candidates = connection.get('candidates',[])
        if len(candidates)>6: raise ValueError('Candidate budget exceeded')
        selected = connection.get('selected_candidate_id')
        if selected is not None and selected not in [c['candidate_id'] for c in candidates]:
            raise ValueError('Selected candidate missing from pool')
        selected_obs=np.zeros(surface.shape,np.uint8);selected_inf=selected_obs.copy()
        for c in candidates:
            p = points(c.get('path_xy')); states=c.get('point_states',[])
            if len(p)!=len(states): raise ValueError('Path and evidence states disagree')
            for kind in ['observed_mask','inferred_mask']:
                m=c.get(kind)
                if m is None: continue
                if m.shape!=surface.shape or m.dtype!=np.uint8 or np.any((m!=0)&(m!=255)):
                    raise ValueError('Candidate mask must be original-size uint8 0/255')
                path=out/'candidate_masks'/role/stem/(c['candidate_id']+'_'+kind+'.png')
                write_image(path,m)
                c[kind+'_path']=str(path.relative_to(out)).replace('\\','/')
                if c['candidate_id']==selected:
                    if kind=='observed_mask':selected_obs=m
                    else:selected_inf=m
        free_mask=draw_free_support(surface.shape,free)
        for name,mask in [('free_observed',free_mask),('selected_observed',selected_obs),('selected_inferred',selected_inf)]:
            write_image(out/'masks'/name/role/rel.name,mask)
        overlay,preview,crop=render(image,surface,c3['fit'],free,connection)
        write_image(out/'overlays'/role/rel.name,overlay)
        write_image(out/'previews'/role/(stem+'.jpg'),preview)
        _,baseline_preview,_=render(image,surface,c3['fit'],free,
            {'status':'STRAIGHT_GEOMETRY_ONLY','selected_candidate_id':None,
             'candidates':[{'candidate_id':'LINE0','path_xy':baseline['path_xy'],
                            'point_states':['inferred']*len(baseline['path_xy'])}]},
            title='IR-A1 straight baseline')
        write_image(out/'baseline_previews'/role/(stem+'.jpg'),baseline_preview)
        status=free.get('status','unknown')
        if status not in {'observed','direction_only','unknown'}:raise ValueError('Unknown free status')
        base.update({'status':'completed','free_status':status,
                     'connection_status':connection.get('status','UNKNOWN'),
                     'width_px':free.get('width_px'),'angle_deg':free.get('angle_deg'),
                     'support_length_px':free.get('support_length_px'),
                     'candidate_count':len(candidates),'selected_candidate_id':selected,
                     'old_c4_history_valid':c4['occlusion_diagnostics'].get('history_valid',False),
                     'free_observed_pixels':int(np.count_nonzero(free_mask)),
                     'selected_observed_pixels':int(np.count_nonzero(selected_obs)),
                     'selected_inferred_pixels':int(np.count_nonzero(selected_inf)),
                     'physical_contact_verified':False,
                     'image_presence_ground_truth':None,
                     'input_dependency_sha256':deps,'preview_crop_xyxy':crop})
        dump(out/'records'/role/(stem+'.json'),{'record':base,'free_segment':json_safe(free),
                  'connection':json_safe(connection),
                  'comparison':{'old_c4_occlusion_diagnostics':c4['occlusion_diagnostics'],
                                'straight_extension':baseline}})
    except Exception as error:
        base.update({'status':'failed','free_status':'error','connection_status':'ERROR',
                     'error':repr(error),'traceback':traceback.format_exc()})
        dump(out/'records'/role/(stem+'.json'),{'record':base})
    # 耗时只进运行日志，不进入预测内容，便于pilot/full逐字节比对。
    return {**base,'elapsed_s':round(time.perf_counter()-start,3)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['pilot','all'],required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();out=args.output.absolute()
    if out.parent!=HERE or out.exists():raise ValueError('Use a NEW direct child run directory')
    rows,qa=select(args.mode)
    sources,models=source_inventory()
    if args.mode=='all':
        frozen=read_json(HERE/'freeze_all.json')
        if sources!=frozen['source_sha256'] or models!=frozen['model_sha256']:
            raise ValueError('Final source/model differs from frozen pilot selection')
    out.mkdir()
    for path in sources:
        target=out/'source_snapshot'/Path(path).name;target.parent.mkdir(exist_ok=True)
        shutil.copyfile(path,target)
    context={'version':'IR-A1.0.0','mode':args.mode,'started_utc':datetime.now(timezone.utc).isoformat(),
             'command':sys.argv,'python':sys.version,'python_executable':sys.executable,
             'platform':platform.platform(),'opencv':cv2.__version__,'numpy':np.__version__,
             'source_sha256':sources,'model_sha256':models,'manifest_sha256':sha(MANIFEST),
             'qa_sha256':sha(QA),'baseline_sha256':sha(BASELINE),'workers':args.workers,
             'selection':[r['output_file'] for r in rows],'equipment_access':False,'reserved_inputs':False}
    dump(out/'run_context.json',context)
    records=[]
    with (out/'run_log.jsonl').open('w',encoding='utf-8') as log:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for i,result in enumerate(executor.map(process,[(r,str(out),qa) for r in rows]),1):
                records.append(result);log.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');log.flush()
                print(json.dumps({'i':i,'n':len(rows),'image':result['output_file'],
                                 'status':result['status'],'free':result.get('free_status'),
                                 'connection':result.get('connection_status'),'error':result.get('error')},ensure_ascii=False),flush=True)
    src_end,models_end=source_inventory();same=sources==src_end and models==models_end
    counts=Counter(r['status'] for r in records)
    summary={'version':'IR-A1.0.0','planned':len(rows),'completed':counts['completed'],'failed':counts['failed'],
             'free_status':dict(Counter(r['free_status'] for r in records)),
             'connection_status':dict(Counter(r['connection_status'] for r in records)),
             'by_camera':{side:{'images':sum(r['camera_role']==side for r in records),
                 'free':dict(Counter(r['free_status'] for r in records if r['camera_role']==side)),
                 'connection':dict(Counter(r['connection_status'] for r in records if r['camera_role']==side))} for side in ['left','right']},
             'old_c4_history_valid_count':sum(bool(r.get('old_c4_history_valid')) for r in records),
             'source_unchanged':same,'exit_code':0 if same and not counts['failed'] else 1,
             'finished_utc':datetime.now(timezone.utc).isoformat(),
             'physical_contact_accuracy':None,'presence_accuracy':None,
             'meaning':'Output counts only; not independently labeled presence/contact accuracy'}
    dump(out/'summary.json',summary)
    columns=['output_file','camera_role','source_group','qa_selected','status','free_status','connection_status',
             'width_px','angle_deg','support_length_px','candidate_count','selected_candidate_id','old_c4_history_valid']
    with (out/'selection.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(records)
    build_review(out)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    raise SystemExit(summary['exit_code'])

if __name__=='__main__':main()
