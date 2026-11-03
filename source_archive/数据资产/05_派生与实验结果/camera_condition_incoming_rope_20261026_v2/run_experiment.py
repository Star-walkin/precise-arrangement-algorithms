"""A2：同一输入比较三种照明处理；冻结后跑全400，失败也保留。"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from datetime import datetime,timezone
import argparse,csv,json,platform,shutil,sys,time,traceback
import cv2
import numpy as np
from experiment_io import HERE,A1,selection,inputs,sources,read_json,dump,write_image,sha
from free_refinement import refine_free_segment
from illumination_c3 import normalize_surface,extract_c3_boundaries
from paired_connection import find_connections
from render_results import render,build_review

def clean(value):
    if isinstance(value,dict):
        return {k:clean(v) for k,v in value.items() if not (isinstance(v,np.ndarray) and v.shape[:2]==(3000,4096))}
    if isinstance(value,list):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value

def ribbon(shape,left,right,states=None,kind='observed'):
    mask=np.zeros(shape,np.uint8);left=np.asarray(left,float);right=np.asarray(right,float)
    if left.ndim!=2 or right.shape!=left.shape:return mask
    for i in range(len(left)-1):
        if states is not None and not (states[i]==kind and states[i+1]==kind):continue
        if np.linalg.norm(left[i+1]-left[i])>45 or np.linalg.norm(right[i+1]-right[i])>45:continue
        q=np.rint([left[i],right[i],right[i+1],left[i+1]]).astype(np.int32)
        cv2.fillConvexPoly(mask,q,255)
    return mask

def process(task):
    row,out,variant,qa,pilot=task;out=Path(out);begin=time.perf_counter();cv2.setNumThreads(1)
    rel=Path(row['output_file']);role=rel.parent.name;stem=rel.stem
    record={k:row[k] for k in ['output_file','camera_role','source_group','source_file','source_time_s','dataset_use_status']}
    record.update({'qa_selected':row['output_file'] in qa,'pilot_selected':row['output_file'] in pilot,
                   'source_png_sha256':row['output_sha256'],'version':'IR-A2.0.0','variant':variant})
    try:
        image,surface,oldc3,oldc4,a1,deps=inputs(row)
        free=refine_free_segment(image,surface,a1['free_segment'])
        pitch=float(oldc3['record']['pitch_px'])
        norm=normalize_surface(image,surface,pitch,mode=variant)
        c3=extract_c3_boundaries(image,norm['enhanced_bgr'],surface,oldc3['fit'],pitch,free,normalization=norm)
        connection=find_connections(image,norm['enhanced_bgr'],surface,free,c3)
        chosen=next((c for c in connection['candidates'] if c['candidate_id']==connection['selected_candidate_id']),None)
        masks={'c3_observed':c3['observed_mask'],'c3_smooth':c3['fit']['smooth_mask'],
            'free_observed':ribbon(surface.shape,free.get('boundary_left_xy',[]),free.get('boundary_right_xy',[])),
            'connection_observed':np.zeros(surface.shape,np.uint8),'connection_inferred':np.zeros(surface.shape,np.uint8)}
        if chosen:
            for kind in ['observed','inferred']:
                masks['connection_'+kind]=ribbon(surface.shape,chosen['boundary_left_xy'],chosen['boundary_right_xy'],chosen['point_states'],kind)
                masks['connection_'+kind][surface==0]=0
            masks['connection_inferred'][masks['connection_observed']>0]=0
        for key,mask in masks.items():
            if mask.shape!=surface.shape or mask.dtype!=np.uint8 or np.any((mask!=0)&(mask!=255)):raise ValueError('Invalid binary mask '+key)
            write_image(out/'masks'/key/role/rel.name,mask)
        roi=norm['enhanced_bgr'].copy();roi[surface==0]=0
        write_image(out/'enhanced_roi'/role/rel.name,roi)
        overlay,enhanced_overlay,preview,enhanced_preview,crop=render(image,norm['enhanced_bgr'],surface,free,c3,connection)
        for folder,data in [('overlays',overlay),('enhanced_overlays',enhanced_overlay)]:write_image(out/folder/role/rel.name,data)
        for folder,data in [('previews',preview),('enhanced_previews',enhanced_preview)]:write_image(out/folder/role/(stem+'.jpg'),data)
        refine=free.get('refinement',{})
        record.update({'status':'completed','free_status':free['status'],'refinement_status':refine.get('status','unavailable'),
            'connection_status':connection['status'],'selected_candidate_id':connection['selected_candidate_id'],
            'c3_point_count':c3['metrics']['raw_supported_points'],'c3_curve_count':c3['metrics']['new_c3_accepted_curves'],
            'width_px':free.get('width_px'),'angle_deg':free.get('angle_deg'),
            'a1_connection_status':a1['connection']['status'],'input_dependency_sha256':deps,'preview_crop_xyxy':crop,
            'normalization':norm['diagnostics'],'mask_pixels':{k:int(np.count_nonzero(v)) for k,v in masks.items()}})
        dump(out/'records'/role/(stem+'.json'),{'record':record,'free_segment':clean(free),'c3':clean(c3),
            'connection':clean(connection),'comparison':{'a1_free_segment':a1['free_segment'],
            'a1_selected_candidate_id':a1['connection']['selected_candidate_id'],'old_c3_curve_count':len(oldc3['fit']['curves'])}})
    except Exception as e:
        record.update({'status':'failed','free_status':'error','connection_status':'ERROR','error':repr(e),'traceback':traceback.format_exc()})
        dump(out/'records'/role/(stem+'.json'),{'record':record})
    return {**record,'elapsed_s':round(time.perf_counter()-begin,3)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['pilot','all'],required=True)
    ap.add_argument('--variant',choices=['raw','global_gamma','illumination'],required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=2);args=ap.parse_args()
    out=args.output.absolute()
    if out.parent!=HERE or out.exists():raise ValueError('Use a new direct child run directory')
    rows,qa,pilot=selection(args.mode);source=sources()
    if args.mode=='all':
        frozen=read_json(HERE/'freeze_all.json')
        if source!=frozen['source_sha256'] or args.variant!=frozen['variant']:raise ValueError('Frozen source/variant mismatch')
    out.mkdir();snapshots={}
    for i,p in enumerate(source):
        target=out/'source_snapshot'/(str(i).zfill(2)+'_'+Path(p).name);target.parent.mkdir(exist_ok=True)
        shutil.copyfile(p,target);snapshots[p]=target.relative_to(out).as_posix()
    dump(out/'run_context.json',{'version':'IR-A2.0.0','mode':args.mode,'variant':args.variant,'command':sys.argv,
        'started_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':source,'source_snapshot_paths':snapshots,
        'python':sys.version,'python_executable':sys.executable,'opencv':cv2.__version__,'numpy':np.__version__,
        'platform':platform.platform(),'workers':args.workers,'selection':[r['output_file'] for r in rows],
        'equipment_access':False,'reserved_inputs':False})
    records=[]
    with (out/'run_log.jsonl').open('w',encoding='utf-8') as log:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i,r in enumerate(ex.map(process,[(r,str(out),args.variant,qa,pilot) for r in rows]),1):
                records.append(r);log.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');log.flush()
                print(json.dumps({'i':i,'n':len(rows),'image':r['output_file'],'status':r['status'],'free':r['free_status'],
                    'connection':r['connection_status'],'points':r.get('c3_point_count'), 'error':r.get('error')},ensure_ascii=False),flush=True)
    same=source==sources();fail=sum(r['status']=='failed' for r in records)
    summary={'version':'IR-A2.0.0','mode':args.mode,'variant':args.variant,'planned':len(rows),'completed':len(rows)-fail,
        'failed':fail,'source_unchanged':same,'exit_code':int(bool(fail) or not same),
        'free_status':dict(Counter(r['free_status'] for r in records)),
        'refinement_status':dict(Counter(r.get('refinement_status','error') for r in records)),
        'connection_status':dict(Counter(r['connection_status'] for r in records)),
        'new_c3_points':sum(r.get('c3_point_count',0) for r in records),'new_c3_curves':sum(r.get('c3_curve_count',0) for r in records),
        'by_camera':{side:{'images':sum(r['camera_role']==side for r in records),
            'free':dict(Counter(r['free_status'] for r in records if r['camera_role']==side)),
            'connection':dict(Counter(r['connection_status'] for r in records if r['camera_role']==side))} for side in ['left','right']},
        'finished_utc':datetime.now(timezone.utc).isoformat(),'accuracy':None,'meaning':'Output counts, not ground-truth accuracy'}
    dump(out/'summary.json',summary)
    cols=['output_file','camera_role','source_group','status','free_status','refinement_status','width_px','angle_deg',
        'connection_status','a1_connection_status','c3_point_count','c3_curve_count']
    with (out/'selection.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    build_review(out)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    raise SystemExit(summary['exit_code'])

if __name__=='__main__':main()
