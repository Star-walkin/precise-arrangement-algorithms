"""D3固定图像/黄点的离线实验；新运行另存，不生成HTML。"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import argparse,json,shutil,sys,time,traceback
import cv2
import numpy as np
from data_io import HERE,D2,selection,inputs,sources,sha,read_json,dump,write_image

def small(v):
    if isinstance(v,dict):return {k:small(x) for k,x in v.items() if not(isinstance(x,np.ndarray) and x.size>200000)}
    if isinstance(v,list):return [small(x) for x in v]
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    return v

def draw(enh,roi,curves,lines):
    canvas=enh.copy()
    for c in curves:
        xy=np.asarray(c.get('sampled_points_xy',[]),int)
        states=c.get('point_states',['supported_fit']*len(xy))
        uncertain=str(c.get('quality','')).startswith(('ambiguous','unreliable'))
        for i in range(len(xy)-1):
            distance=np.linalg.norm(xy[i+1]-xy[i])
            if distance>20:continue
            q=np.rint(np.linspace(xy[i],xy[i+1],max(2,int(distance)+1))).astype(int)
            if np.any(roi[q[:,1],q[:,0]]==0):continue
            infer=states[i]=='inferred_short_gap'
            if (infer or uncertain) and (i//5)%2:continue
            color=(210,65,210) if infer else ((40,115,255) if uncertain else (230,190,35))
            cv2.line(canvas,tuple(xy[i]),tuple(xy[i+1]),color,2,cv2.LINE_AA)
    for line in lines:
        for p in line['observed_points_xy'][::2]:
            cv2.drawMarker(canvas,tuple(map(int,p)),(0,230,255),cv2.MARKER_CROSS,5,1,cv2.LINE_AA)
    cv2.drawContours(canvas,cv2.findContours(roi,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(0,135,0),1)
    canvas[roi==0]=0
    return canvas

def process(task):
    from groove_field import build_groove_field
    from ordered_paths import fit_groove_paths
    row,out,variant,is_pilot=task;out=Path(out);cv2.setNumThreads(1);start=time.perf_counter()
    rel=Path(row['output_file']);role=rel.parent.name
    record={k:row[k] for k in ['output_file','camera_role','source_group','source_file','source_time_s','dataset_use_status']}
    record.update({'version':read_json(HERE/'protocol.json')['version'],'variant':variant,'source_png_sha256':row['output_sha256']})
    try:
        im,roi,enh,pitch,previous,c3,deps=inputs(row)
        shape=((c3.get('fit') or {}).get('model') or {}).get('shared_shape_for_grouping_and_regularization') or {}
        if not shape or not previous.get('observed_lines'):
            # 固定400中的合法无亮区锚案例：保留ROI，不能因空model崩溃或造线。
            field={'x0':0,'y0':0,'score':np.zeros((1,1),np.float32),
                   'support':np.zeros((1,1),np.uint8),'diagnostics':{'status':'no_trusted_reference_or_observations'}}
        else:
            field=build_groove_field(im,enh,roi,pitch,shape)
        if is_pilot:
            for key,value in [('field_score',np.rint(field['score']*255).astype(np.uint8)),('field_support',field['support'])]:
                layer=np.zeros_like(roi);y0,x0=int(field['y0']),int(field['x0']);hh,ww=value.shape
                layer[y0:y0+hh,x0:x0+ww]=value
                write_image(out/'记录'/key/role/rel.name,layer)
        result=fit_groove_paths(previous,c3,roi,pitch,field,variant=variant)
        lines=previous.get('observed_lines',[])
        observed=np.zeros_like(roi)
        for line in lines:
            for x,y in line['observed_points_xy']:cv2.circle(observed,(int(x),int(y)),1,255,-1)
        observed[roi==0]=0
        masks={'observed':observed,'smooth':result['fit']['smooth_mask'],'inferred':result['inferred_mask']}
        for key,m in masks.items():
            if m.dtype!=np.uint8 or m.shape!=roi.shape or np.any((m!=0)&(m!=255)) or np.any(m[roi==0]):raise ValueError('mask contract '+key)
            write_image(out/'记录/masks'/key/role/rel.name,m)
        curves=result['fit'].get('curves',[])
        write_image(out/'图像结果'/role/rel.name,draw(enh,roi,curves,lines))
        record.update({'status':'completed','pitch_px':pitch,'input_dependency_sha256':deps,
            'dark_roi_pixels':int(np.count_nonzero(roi)),'observed_point_count':sum(len(l['observed_points_xy']) for l in lines),
            'curve_count':len(curves),'observed_pixels':int(np.count_nonzero(observed)),
            'smooth_pixels':int(np.count_nonzero(masks['smooth'])),'inferred_pixels':int(np.count_nonzero(masks['inferred'])),
            'field_diagnostics':field.get('diagnostics',{}),'path_diagnostics':result.get('diagnostics',{}),
            'old_d2_curve_count':previous['record'].get('curve_count',0)})
        dump(out/'记录/cases'/role/(rel.stem+'.json'),{'record':record,'observed_lines':lines,
            'fit':small(result['fit']),'new_groups':small(result.get('new_groups',[])),
            'point_assignments':small(result.get('point_assignments',[]))})
    except Exception as e:
        record.update({'status':'failed','error':repr(e),'traceback':traceback.format_exc()})
        dump(out/'记录/cases'/role/(rel.stem+'.json'),{'record':record})
    return {**record,'elapsed_s':round(time.perf_counter()-start,3)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['pilot','all'],required=True);ap.add_argument('--variant',choices=['image','geometry'],default='image');ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=2);args=ap.parse_args()
    out=args.output.absolute()
    if out.parent!=HERE or out.exists():raise ValueError('new direct child output required')
    rows,_=selection(args.mode);source=sources();protocol=read_json(HERE/'protocol.json')
    if any(sha(p)!=h for p,h in protocol['frozen_d2_source_sha256'].items()):raise ValueError('D2 baseline source changed')
    if args.mode=='all':
        frozen=read_json(HERE/'freeze_all.json')
        if source!=frozen['source_sha256'] or args.variant!='image':raise ValueError('Final freeze mismatch')
    out.mkdir();snapshots={}
    for i,p in enumerate(source):
        target=out/'source_snapshot'/(f'{i:02}_'+Path(p).name);target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target);snapshots[p]=target.relative_to(out).as_posix()
    dump(out/'run_context.json',{'source_sha256':source,'source_snapshot_paths':snapshots,'selection':[r['output_file'] for r in rows],
       'started_utc':datetime.now(timezone.utc).isoformat(),'command':sys.argv,'python_executable':sys.executable,'python':sys.version,
       'opencv':cv2.__version__,'numpy':np.__version__,'workers':args.workers,'mode':args.mode,'variant':args.variant,'version':protocol['version']})
    records=[]
    with (out/'run_log.jsonl').open('w',encoding='utf-8') as log:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i,r in enumerate(pool.map(process,[(r,str(out),args.variant,args.mode=='pilot') for r in rows]),1):
                records.append(r);log.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');log.flush()
                print(json.dumps({'i':i,'n':len(rows),'image':r['output_file'],'status':r['status'],'curves':r.get('curve_count'),'error':r.get('error')},ensure_ascii=False),flush=True)
    failed=sum(r['status']!='completed' for r in records);same=source==sources()
    summary={'version':protocol['version'],'variant':args.variant,'planned':len(rows),'completed':len(rows)-failed,'failed':failed,'source_unchanged':same,
       'exit_code':int(bool(failed) or not same),'by_camera':{c:sum(r['camera_role']==c for r in records) for c in ['left','right']},
       'total_observed_points':sum(r.get('observed_point_count',0) for r in records),'total_curves':sum(r.get('curve_count',0) for r in records),
       'images_with_curves':sum(r.get('curve_count',0)>0 for r in records),'finished_utc':datetime.now(timezone.utc).isoformat(),
       'accuracy':None,'html_generated':False,'equipment_access':False,'reserved_access':False}
    dump(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True);raise SystemExit(summary['exit_code'])

if __name__=='__main__':main()
