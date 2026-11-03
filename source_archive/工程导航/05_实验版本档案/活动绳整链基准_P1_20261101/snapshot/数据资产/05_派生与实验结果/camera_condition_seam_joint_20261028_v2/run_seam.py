"""固定开放图离线运行；C3、D4重新检测重叠带，保存候选及拒绝证据。"""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import argparse, json, shutil, sys, time, traceback
import cv2
import numpy as np
from seam_io import HERE, F1, selection, inputs, sources, sha, read_json, dump, write_image, baseline_unchanged

MASK_KEYS = ('transition','merged','inferred','added','removed','replacement','baseline','new_join','simple_bridge')

def plain(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [plain(v) for v in value]
    return value

def render(data, result, old_points):
    """T外连显示图也原样保留。T内清除旧叠加后绘制正式边界，避免叠两条线。"""
    if not result['joins']: return data['f1_image'].copy()
    canvas=data['original'].copy(); surface=data['surface']>0
    dark=surface&(data['dark']>0)&(data['bright']==0)
    canvas[dark]=data['enhanced'][dark]; canvas[~surface]=0
    merged=result['merged_mask']>0; inferred=result['inferred_mask']>0
    canvas[merged&~inferred]=(230,190,35)
    yy,xx=np.nonzero(inferred); take=(xx//6)%2==0
    canvas[yy[take],xx[take]]=(210,65,210)
    # 旧黄点只在仍保留的边界附近显示，新黄点来自本次两路实际检测。
    retained=(result['baseline_mask']>0)&(result['replacement_mask']==0)
    near=cv2.dilate(retained.astype(np.uint8),np.ones((5,5),np.uint8))>0
    h,w=near.shape
    for p in old_points[::2]:
        x,y=int(round(p['x'])),int(round(p['y']))
        if 0<=x<w and 0<=y<h and near[y,x]:
            cv2.drawMarker(canvas,(x,y),(0,255,255),cv2.MARKER_CROSS,5,1,cv2.LINE_8)
    for p in result['accepted_points'][::2]:
        cv2.drawMarker(canvas,(int(round(p['x'])),int(round(p['y']))),(0,255,255),cv2.MARKER_CROSS,5,1,cv2.LINE_8)
    out=data['f1_image'].copy(); t=result['transition_mask']>0
    out[t]=canvas[t]; return out

def process(task):
    from seam_observation import detect_overlap
    from seam_join import join_seams
    row,root=task; out=Path(root); cv2.setNumThreads(1); start=time.perf_counter()
    rel=Path(row['output_file']); role=rel.parent.name
    rec={k:row[k] for k in ('output_file','camera_role','source_group','source_file','source_time_s','dataset_use_status')}
    rec.update({'version':'RB-F2.0.0','original_sha256':row['output_sha256']})
    try:
        data,deps=inputs(row)
        detected=detect_overlap(data['original'],data['surface'],data['bright'],data['dark'],data['pitch'],data['c3_record'],data['c3_exclusion'])
        result=join_seams(data,detected)
        old_points=data['f1_case']['display_point_centers']
        image=render(data,result,old_points)
        t=result['transition_mask']>0
        assert np.array_equal(result['merged_mask'][~t],data['f1_merged'][~t])
        assert np.array_equal(result['inferred_mask'][~t],data['f1_inferred'][~t])
        assert np.array_equal(image[~t],data['f1_image'][~t])
        assert not np.any(result['merged_mask'][data['surface']==0])
        write_image(out/'图像结果'/role/rel.name,image)
        pixels={}
        for key in MASK_KEYS:
            mask=result[key+'_mask']; assert not np.any((mask!=0)&(mask!=255))
            write_image(out/'记录/masks'/key/role/rel.name,mask); pixels[key]=int(np.count_nonzero(mask))
        write_image(out/'记录/masks/exclusion'/role/rel.name,detected['exclusion_mask'])
        fieldpath=out/'记录/fields'/role/(rel.stem+'.npz'); fieldpath.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(fieldpath,**detected['fields'])
        if out.name.startswith('pilot'):
            normal=detected['normalized_bgr'].copy(); normal[~t]=0
            write_image(out/'重叠区增强'/role/rel.name,normal)
        rec.update({'status':'completed','input_dependency_sha256':deps,'pixels':pixels,
          'new_observation_counts':{'C3':len(detected['c3']['points']),'D4':len(detected['d4']['points'])},
          'join_diagnostics':result['diagnostics'],'observation_diagnostics':detected['diagnostics']})
        payload={k:result[k] for k in ('joins','rejections','new_groups','accepted_points','endpoints')}
        # 全尺寸二值图单独保存PNG，JSON只写曲线/点/诊断；禁止把1200万像素展开为文本。
        payload['observations']={}
        for k in ('c3','d4'):
            fit={}
            for name,value in detected[k]['fit'].items():
                if isinstance(value,np.ndarray) and value.shape==data['surface'].shape:
                    destination=out/'记录/detector_masks'/k/name/role/rel.name
                    write_image(destination,value)
                    fit[name]={'png':destination.relative_to(out).as_posix(),'sha256':sha(destination)}
                else: fit[name]=value
            payload['observations'][k]={'points':detected[k]['points'],'fit':fit,'diagnostics':detected[k]['diagnostics']}
        rec['elapsed_s']=round(time.perf_counter()-start,3)
        dump(out/'记录/cases'/role/(rel.stem+'.json'),plain({'record':rec,**payload}))
    except Exception as exc:
        rec.update({'status':'failed','error':repr(exc),'traceback':traceback.format_exc(),'elapsed_s':round(time.perf_counter()-start,3)})
        dump(out/'记录/cases'/role/(rel.stem+'.json'),rec)
    return rec

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=('pilot','all'),required=True)
    ap.add_argument('--output',type=Path,required=True); ap.add_argument('--workers',type=int,default=3); args=ap.parse_args()
    out=args.output.absolute()
    if out.parent!=HERE or out.exists(): raise ValueError('new direct child required')
    assert baseline_unchanged(); rows=selection(args.mode); frozen=sources()
    if args.mode=='all': assert frozen==read_json(HERE/'freeze_all.json')['source_sha256']
    out.mkdir(); snapshots={}
    for i,p in enumerate(frozen):
        target=out/'source_snapshot'/f'{i:03d}_{Path(p).name}'; target.parent.mkdir(exist_ok=True)
        shutil.copyfile(p,target); snapshots[p]=target.relative_to(out).as_posix()
    dump(out/'run_context.json',{'mode':args.mode,'version':'RB-F2.0.0','started_utc':datetime.now(timezone.utc).isoformat(),
      'selection':[r['output_file'] for r in rows],'source_sha256':frozen,'source_snapshot_paths':snapshots,
      'command':sys.argv,'python_executable':sys.executable,'python':sys.version,'opencv':cv2.__version__,'numpy':np.__version__,'workers':args.workers})
    records=[]
    with (out/'run_log.jsonl').open('w',encoding='utf8') as log:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i,r in enumerate(pool.map(process,[(r,str(out)) for r in rows]),1):
                records.append(r); log.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n'); log.flush()
                if i%10==0 or args.mode=='pilot' or r['status']=='failed':
                    print(json.dumps({'i':i,'n':len(rows),'status':r['status'],'joins':r.get('join_diagnostics',{}).get('accepted_joins'),'seconds':r['elapsed_s'],'error':r.get('error')},ensure_ascii=False),flush=True)
    failed=sum(r['status']!='completed' for r in records); unchanged=frozen==sources() and baseline_unchanged()
    summary={'version':'RB-F2.0.0','planned':len(rows),'completed':len(rows)-failed,'failed':failed,
      'source_unchanged':unchanged,'exit_code':int(failed>0 or not unchanged),
      'by_camera':{c:sum(r['camera_role']==c for r in records) for c in ('left','right')},
      'accepted_joins':sum(r.get('join_diagnostics',{}).get('accepted_joins',0) for r in records),
      'images_with_joins':sum(r.get('join_diagnostics',{}).get('accepted_joins',0)>0 for r in records),
      'added_pixels':sum(r.get('pixels',{}).get('added',0) for r in records),
      'new_observations':{s:sum(r.get('new_observation_counts',{}).get(s,0) for r in records) for s in ('C3','D4')},
      'accuracy':None,'equipment_access':False,'reserved_access':False,'html':False,'finished_utc':datetime.now(timezone.utc).isoformat()}
    dump(out/'summary.json',summary); print(json.dumps(summary,ensure_ascii=False),flush=True); return summary['exit_code']
if __name__=='__main__': raise SystemExit(main())
