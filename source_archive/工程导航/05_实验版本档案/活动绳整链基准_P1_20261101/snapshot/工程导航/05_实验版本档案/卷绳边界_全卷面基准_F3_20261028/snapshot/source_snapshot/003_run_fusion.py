"""把冻结C3与D4结果在原图坐标合并；不重新运行模型或检测器。"""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import argparse,json,shutil,sys,time,traceback
import cv2,numpy as np
from fusion_io import HERE,MASK_KEYS,selection,inputs,sources,sha,read_json,dump,write_image,baseline_unchanged

def process(task):
    from fusion import combine,render
    row,root=task;out=Path(root);cv2.setNumThreads(1);start=time.perf_counter()
    rel=Path(row['output_file']);role=rel.parent.name
    rec={k:row[k] for k in ('output_file','camera_role','source_group','source_file','source_time_s','dataset_use_status')}
    rec.update({'version':'RB-F1.0.0','original_sha256':row['output_sha256']})
    try:
        data,deps=inputs(row);masks,points,metadata=combine(data);image=render(data,masks,points)
        write_image(out/'图像结果'/role/rel.name,image)
        for key in MASK_KEYS:write_image(out/'记录/masks'/key/role/rel.name,masks[key])
        rec.update({'status':'completed','input_dependency_sha256':deps,**metadata})
        dump(out/'记录/cases'/role/(rel.stem+'.json'),{'record':rec,'display_point_centers':points})
    except Exception as exc:
        rec.update({'status':'failed','error':repr(exc),'traceback':traceback.format_exc()})
        dump(out/'记录/cases'/role/(rel.stem+'.json'),{'record':rec})
    rec['elapsed_s']=round(time.perf_counter()-start,3)
    return rec

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=('pilot','all'),required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    out=args.output.absolute()
    if out.parent!=HERE or out.exists():raise ValueError('new direct child required')
    assert baseline_unchanged()
    rows=selection(args.mode);frozen=sources()
    if args.mode=='all':assert frozen==read_json(HERE/'freeze_all.json')['source_sha256']
    out.mkdir();snap={}
    for p in frozen:
        target=out/'source_snapshot'/Path(p).name;target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target);snap[p]=target.relative_to(out).as_posix()
    dump(out/'run_context.json',{'mode':args.mode,'version':'RB-F1.0.0','started_utc':datetime.now(timezone.utc).isoformat(),
      'selection':[r['output_file'] for r in rows],'source_sha256':frozen,'source_snapshot_paths':snap,
      'command':sys.argv,'python_executable':sys.executable,'python':sys.version,'opencv':cv2.__version__,'numpy':np.__version__,'workers':args.workers})
    records=[]
    with (out/'run_log.jsonl').open('w',encoding='utf8') as log:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i,r in enumerate(pool.map(process,[(r,str(out)) for r in rows]),1):
                records.append(r);log.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');log.flush()
                if i%20==0 or args.mode=='pilot' or r['status']=='failed':print(json.dumps({'i':i,'n':len(rows),'status':r['status'],'error':r.get('error')},ensure_ascii=False),flush=True)
    failed=sum(r['status']!='completed' for r in records);unchanged=frozen==sources() and baseline_unchanged()
    summary={'version':'RB-F1.0.0','planned':len(rows),'completed':len(rows)-failed,'failed':failed,
      'source_unchanged':unchanged,'exit_code':int(failed>0 or not unchanged),
      'by_camera':{c:sum(r['camera_role']==c for r in records) for c in ('left','right')},
      'pixels_total':{k:sum(r.get('pixels',{}).get(k,0) for r in records) for k in MASK_KEYS},
      'images_with_boundaries':sum(r.get('pixels',{}).get('merged',0)>0 for r in records),
      'new_boundary_pixels_generated':0,'fitted_again':False,'accuracy':None,
      'equipment_access':False,'reserved_access':False,'html':False,'finished_utc':datetime.now(timezone.utc).isoformat()}
    dump(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True);return summary['exit_code']
if __name__=='__main__':raise SystemExit(main())
