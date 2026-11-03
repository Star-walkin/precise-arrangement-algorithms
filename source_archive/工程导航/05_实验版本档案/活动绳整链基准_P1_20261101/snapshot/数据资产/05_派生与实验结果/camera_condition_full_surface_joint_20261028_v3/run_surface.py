"""F3离线运行：全卷面D4候选和C3/F2协同结果分别保存。"""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import argparse,json,shutil,sys,time,traceback
import cv2,numpy as np
from surface_io import HERE,selection,inputs,sources,sha,read_json,dump,write_image,baseline_unchanged

def plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    return v

def render(data,detected,result):
    """第一组显示D4全卷面候选；第二组只把接受的新边界/真观测加入F2。"""
    candidate=detected['normalized_bgr'].copy()
    smooth=detected['fit']['smooth_mask']>0;inf=detected['fit']['inferred_mask']>0
    candidate[smooth&~inf]=(230,190,35)
    yy,xx=np.nonzero(inf);keep=(xx//6)%2==0;candidate[yy[keep],xx[keep]]=(210,65,210)
    for p in detected['points'][::2]:
        cv2.drawMarker(candidate,(int(round(p['x'])),int(round(p['y']))),(0,255,255),cv2.MARKER_CROSS,5,1,cv2.LINE_8)
    candidate[data['surface']==0]=0
    out=data['f2_image'].copy();layer=out.copy();new=result['new_mask']>0;inferred=result['inferred_mask']>0
    layer[new&~inferred]=(230,190,35)
    yy,xx=np.nonzero(new&inferred);keep=(xx//6)%2==0;layer[yy[keep],xx[keep]]=(210,65,210)
    for p in result['accepted_points'][::2]:
        cv2.drawMarker(layer,(int(round(p['x'])),int(round(p['y']))),(0,255,255),cv2.MARKER_CROSS,5,1,cv2.LINE_8)
    t=result['transition_mask']>0;out[t]=layer[t]
    return candidate,out

def process(task):
    from surface_observation import detect_surface
    from surface_fusion import fuse_surface
    row,root=task;out=Path(root);cv2.setNumThreads(1);start=time.perf_counter()
    part=Path(row['output_file']);role=part.parent.name;name=part.name
    rec={k:row[k] for k in ('output_file','camera_role','source_group','source_file','source_time_s','dataset_use_status')}
    rec.update({'version':'RB-F3.0.0','original_sha256':row['output_sha256']})
    try:
        data,deps=inputs(row);detected=detect_surface(data);result=fuse_surface(data,detected)
        candidate,image=render(data,detected,result)
        t=result['transition_mask']>0
        assert np.array_equal(t,data['f2_transition']>0)
        assert np.array_equal(image[~t],data['f2_image'][~t])
        assert np.array_equal(result['merged_mask'][~t],data['f2_merged'][~t])
        assert np.array_equal(result['inferred_mask'][~t],data['f2_inferred'][~t])
        assert not np.any((data['f2_merged']>0)&(result['merged_mask']==0))
        assert not np.any(result['merged_mask'][data['surface']==0])
        write_image(out/'图像结果/01_D4全卷面候选'/role/name,candidate)
        write_image(out/'图像结果/02_F3融合边界'/role/name,image)
        masks={k:result[k+'_mask'] for k in ('merged','inferred','new','removed','transition')}
        masks.update({'full_d4':detected['fit']['smooth_mask'],'full_d4_inferred':detected['fit']['inferred_mask'],
            'exclusion':detected['exclusion_mask']})
        obs=np.zeros(data['surface'].shape,np.uint8)
        for p in detected['points']:obs[int(round(p['y'])),int(round(p['x']))]=255
        masks['full_d4_observed']=obs
        pixels={}
        for k,m in masks.items():
            assert m.dtype==np.uint8 and m.shape==data['surface'].shape and not np.any((m!=0)&(m!=255))
            write_image(out/'记录/masks'/k/role/name,m);pixels[k]=int(np.count_nonzero(m))
        dst=out/'记录/fields'/role/(part.stem+'.npz');dst.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dst,**detected['fields'])
        fit={}
        for k,v in detected['fit'].items():
            if isinstance(v,np.ndarray) and v.shape==data['surface'].shape:
                key={'smooth_mask':'full_d4','inferred_mask':'full_d4_inferred'}[k]
                p=Path('记录/masks')/key/role/name
                fit[k]={'png':p.as_posix(),'sha256':sha(out/p)}
            else:fit[k]=v
        rec.update({'status':'completed','input_dependency_sha256':deps,'pixels':pixels,
          'new_full_surface_points':len(detected['points']),'full_surface_curves':len(detected['fit']['curves']),
          'accepted_point_count':len(result['accepted_points']),'fusion_diagnostics':result['diagnostics'],
          'observation_diagnostics':detected['diagnostics'],'elapsed_s':round(time.perf_counter()-start,3)})
        payload={'record':rec,'observations':{'points':detected['points'],'tracks':detected['tracks'],'fit':fit},
          'accepted_points':result['accepted_points'],'groups':result['groups'],'rejections':result['rejections']}
        p=out/'记录/cases'/role/(part.stem+'.json');p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(plain(payload),ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf8')
    except Exception as exc:
        rec.update({'status':'failed','error':repr(exc),'traceback':traceback.format_exc(),'elapsed_s':round(time.perf_counter()-start,3)})
        dump(out/'记录/cases'/role/(part.stem+'.json'),rec)
    return rec

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=('pilot','all'),required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
    out=args.output.absolute()
    if out.parent!=HERE or out.exists():raise ValueError('new direct child required')
    assert baseline_unchanged();rows=selection(args.mode);frozen=sources()
    if args.mode=='all':assert frozen==read_json(HERE/'freeze_all.json')['source_sha256']
    out.mkdir();snap={}
    for i,p in enumerate(frozen):
        target=out/'source_snapshot'/f'{i:03d}_{Path(p).name}';target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target);snap[p]=target.relative_to(out).as_posix()
    dump(out/'run_context.json',{'version':'RB-F3.0.0','mode':args.mode,'started_utc':datetime.now(timezone.utc).isoformat(),
       'selection':[r['output_file'] for r in rows],'source_sha256':frozen,'source_snapshot_paths':snap,
       'command':sys.argv,'python_executable':sys.executable,'python':sys.version,'opencv':cv2.__version__,'numpy':np.__version__,'workers':args.workers})
    records=[]
    with (out/'run_log.jsonl').open('w',encoding='utf8') as log:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i,r in enumerate(pool.map(process,[(r,str(out)) for r in rows]),1):
                records.append(r);log.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');log.flush()
                if args.mode=='pilot' or i%10==0 or r['status']=='failed':
                    print(json.dumps({'i':i,'n':len(rows),'status':r['status'],'added':r.get('pixels',{}).get('new'),
                       'accepted_points':r.get('accepted_point_count'),'seconds':r['elapsed_s'],'error':r.get('error')},ensure_ascii=False),flush=True)
    failed=sum(r['status']!='completed' for r in records);unchanged=frozen==sources() and baseline_unchanged()
    summary={'version':'RB-F3.0.0','planned':len(rows),'completed':len(rows)-failed,'failed':failed,'source_unchanged':unchanged,
      'exit_code':int(failed>0 or not unchanged),'by_camera':{c:sum(r['camera_role']==c for r in records) for c in ('left','right')},
      'new_full_surface_points':sum(r.get('new_full_surface_points',0) for r in records),
      'full_surface_curves':sum(r.get('full_surface_curves',0) for r in records),
      'new_accepted_points':sum(r.get('accepted_point_count',0) for r in records),
      'added_pixels':sum(r.get('pixels',{}).get('new',0) for r in records),
      'images_with_additions':sum(r.get('pixels',{}).get('new',0)>0 for r in records),
      'accuracy':None,'equipment_access':False,'reserved_access':False,'html':False,'finished_utc':datetime.now(timezone.utc).isoformat()}
    dump(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True);return summary['exit_code']
if __name__=='__main__':raise SystemExit(main())
