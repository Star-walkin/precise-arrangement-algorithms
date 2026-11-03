"""D4离线分阶段批处理：重检黄点、重分组、绳体穿越拒绝；输出不含HTML。"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import argparse,json,sys,shutil,time,traceback
import cv2
import numpy as np
from data_io import HERE,selection,inputs,sources,sha,read_json,dump,write_image

def small(v):
    if isinstance(v,dict):return {k:small(x) for k,x in v.items() if not(isinstance(x,np.ndarray) and x.size>100000)}
    if isinstance(v,(list,tuple)):return [small(x) for x in v]
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    return v

def overlay(enh,roi,points,curves=()):
    canvas=enh.copy()
    for c in curves:
        pts=c['sampled_points_xy'];states=c['point_states']
        for i in range(len(pts)-1):
            a,b=pts[i],pts[i+1]
            if abs(a[0]-b[0])>1 or abs(a[1]-b[1])>5:continue
            inferred=states[i]=='inferred' or states[i+1]=='inferred'
            if inferred and (i//5)%2:continue
            cv2.line(canvas,tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)),(210,65,210) if inferred else (230,190,35),1,cv2.LINE_8)
    for p in points[::2]:cv2.drawMarker(canvas,(int(p['x']),int(round(p['y']))),(0,230,255),cv2.MARKER_CROSS,5,1,cv2.LINE_AA)
    cv2.drawContours(canvas,cv2.findContours(roi,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(0,135,0),1)
    canvas[roi==0]=0
    return canvas

def process(task):
    from shadow_width import build_shadow_field
    from period_guard import build_period_guard
    from shadow_tracks import detect_and_group
    from ribbon_curves import fit_validated_curves
    row,out,is_pilot=task;out=Path(out);cv2.setNumThreads(1);start=time.perf_counter()
    rel=Path(row['output_file']);role=rel.parent.name
    record={k:row[k] for k in ['output_file','camera_role','source_group','source_file','source_time_s','dataset_use_status']}
    record.update({'version':read_json(HERE/'protocol.json')['version'],'source_png_sha256':row['output_sha256']})
    try:
        im,roi,enh,pitch,c3,deps=inputs(row)
        times={'load_s':time.perf_counter()-start};tick=time.perf_counter()
        shape=((c3.get('fit') or {}).get('model') or {}).get('shared_shape_for_grouping_and_regularization')
        field=build_shadow_field(im,enh,roi,pitch,shape)
        times['shadow_s']=time.perf_counter()-tick;tick=time.perf_counter()
        guard=build_period_guard(im,enh,roi,pitch,shape,field)
        times['body_s']=time.perf_counter()-tick;tick=time.perf_counter()
        detected=detect_and_group(field,guard,roi,pitch)
        times['group_s']=time.perf_counter()-tick;tick=time.perf_counter()
        result=fit_validated_curves(detected['tracks'],roi,pitch,field,guard)
        times['curve_s']=time.perf_counter()-tick;tick=time.perf_counter()
        x0,y0=field['x0'],field['y0'];hh,ww=field['binary_shadow'].shape
        for key,folder in [('binary_shadow','01_阴影二值化'),('cleaned_shadow','02_去细股阴影')]:
            layer=np.zeros_like(roi);layer[y0:y0+hh,x0:x0+ww]=field[key];layer[roi==0]=0
            write_image(out/'图像结果'/folder/role/rel.name,layer)
        write_image(out/'图像结果/03_新黄点'/role/rel.name,overlay(enh,roi,detected['points']))
        curves=result['curves'];tracks_by_id={t['track_id']:t for t in detected['tracks']}
        # 只显示真正参与通过检查的曲线子段的点，不能把整条母轨迹的坏点重新画回来。
        used_points={tracks_by_id[c['track_id']]['points'][j]['point_id']
                     for c in curves for j in c['source_point_indices']}
        accepted_points=[p for p in detected['points'] if p['point_id'] in used_points]
        write_image(out/'图像结果/04_边界结果'/role/rel.name,overlay(enh,roi,accepted_points,curves))
        for key,value in [('observed',detected['observed_mask']),('smooth',result['smooth_mask']),('inferred',result['inferred_mask'])]:
            if value.shape!=roi.shape or value.dtype!=np.uint8 or np.any((value!=0)&(value!=255)) or np.any(value[roi==0]):raise ValueError('mask contract '+key)
            write_image(out/'记录/masks'/key/role/rel.name,value)
        # 绳体门与unknown逐图保留，让最终拒绝规则可独立重验。
        for key,value in [('body_barrier',np.rint(guard['body_barrier']*255).astype(np.uint8)),
                          ('unknown',guard['unknown_mask']),('broad_score',np.rint(field['broad_score']*255).astype(np.uint8))]:
            layer=np.zeros_like(roi);layer[y0:y0+hh,x0:x0+ww]=value;layer[roi==0]=0
            write_image(out/'记录/fields'/key/role/rel.name,layer)
        # PNG量化会丢失0.65附近的阈值细节；独立审计使用未经量化的原始浮点场。
        body_path=out/'记录/fields/body_float'/role/(rel.stem+'.npz')
        body_path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(body_path,x0=np.int32(x0),y0=np.int32(y0),body=guard['body_barrier'].astype(np.float32))
        times['output_s']=time.perf_counter()-tick
        record.update({'status':'completed','pitch_px':pitch,'input_dependency_sha256':deps,
          'roi_pixels':int(np.count_nonzero(roi)),'new_yellow_count':len(detected['points']),
          'curve_count':len(curves),'new_track_count':len(detected['tracks']),
          'rejected_point_count':len(detected['rejected_points']),'rejected_curve_count':len(result.get('rejections',[])),
          'shadow_diagnostics':field['diagnostics'],'period_diagnostics':guard['diagnostics'],
          'tracking_diagnostics':detected['diagnostics'],'fitting_diagnostics':result.get('diagnostics',{}),
          'stage_seconds':{k:round(v,4) for k,v in times.items()},'displayed_yellow_count':len(accepted_points)})
        dump(out/'记录/cases'/role/(rel.stem+'.json'),{'record':record,'new_points':detected['points'],
          'new_tracks':detected['tracks'],'rejected_points':detected['rejected_points'],
          'rejected_links':detected['rejected_links'],'fit':small(result)})
    except Exception as e:
        record.update({'status':'failed','error':repr(e),'traceback':traceback.format_exc()})
        dump(out/'记录/cases'/role/(rel.stem+'.json'),{'record':record})
    record['elapsed_s']=round(time.perf_counter()-start,3)
    return record

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['pilot','all'],required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=2);args=ap.parse_args()
    out=args.output.absolute()
    if out.parent!=HERE or out.exists():raise ValueError('new direct child required')
    rows,_=selection(args.mode);source=sources();protocol=read_json(HERE/'protocol.json')
    if any(sha(p)!=h for p,h in protocol['frozen_baseline_sha256'].items()):raise ValueError('baseline changed')
    if args.mode=='all' and source!=read_json(HERE/'freeze_all.json')['source_sha256']:raise ValueError('final freeze mismatch')
    out.mkdir();snapshots={}
    for i,p in enumerate(source):
        target=out/'source_snapshot'/(f'{i:02}_'+Path(p).name);target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target);snapshots[p]=target.relative_to(out).as_posix()
    dump(out/'run_context.json',{'source_sha256':source,'source_snapshot_paths':snapshots,'selection':[r['output_file'] for r in rows],
       'started_utc':datetime.now(timezone.utc).isoformat(),'command':sys.argv,'python_executable':sys.executable,'python':sys.version,
       'opencv':cv2.__version__,'numpy':np.__version__,'workers':args.workers,'mode':args.mode,'version':protocol['version']})
    records=[]
    with (out/'run_log.jsonl').open('w',encoding='utf8') as log:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i,r in enumerate(pool.map(process,[(r,str(out),args.mode=='pilot') for r in rows]),1):
                records.append(r);log.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');log.flush()
                print(json.dumps({'i':i,'n':len(rows),'image':r['output_file'],'status':r['status'],'new_points':r.get('new_yellow_count'),'curves':r.get('curve_count'),'error':r.get('error')},ensure_ascii=False),flush=True)
    failed=sum(r['status']!='completed' for r in records);same=source==sources()
    summary={'version':protocol['version'],'planned':len(rows),'completed':len(rows)-failed,'failed':failed,'source_unchanged':same,'exit_code':int(bool(failed) or not same),
      'by_camera':{c:sum(r['camera_role']==c for r in records) for c in ['left','right']},'total_new_points':sum(r.get('new_yellow_count',0) for r in records),
      'total_curves':sum(r.get('curve_count',0) for r in records),'images_with_curves':sum(r.get('curve_count',0)>0 for r in records),
      'finished_utc':datetime.now(timezone.utc).isoformat(),'accuracy':None,'html_generated':False,'equipment_access':False,'reserved_access':False}
    dump(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True);raise SystemExit(summary['exit_code'])
if __name__=='__main__':main()
