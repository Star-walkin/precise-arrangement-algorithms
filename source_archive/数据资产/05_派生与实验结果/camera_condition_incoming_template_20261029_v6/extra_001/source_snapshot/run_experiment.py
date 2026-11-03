"""A6 fixed open-image runner. New run directories, frozen source/input bytes."""
from __future__ import annotations
import argparse,sys,time,platform,traceback,shutil
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import cv2,numpy as np
import whole_io as io

CORE=['whole_io.py','free_rope.py','photometric_ribbon.py','attached_groove.py','appearance_template.py','free_seed_recovery.py','axial_reacquisition.py','image_beam.py','pipeline.py','run_experiment.py','protocol.json']
FOLDERS=('01_可信前缀掩膜','02_原图叠加','03_待确认候选并集','04_最佳外观候选掩膜')
BRANCH_FOLDER='05_多候选分支'
COLORS=[(0,170,255),(255,170,50),(220,80,220),(220,220,40),(50,120,240),(180,100,240)]

def dependencies():return [str(io.ASSETS/'camera_condition_incoming_rope_20261026_v2/free_refinement.py')]

def compact(value):
    if isinstance(value,np.ndarray):
        return {'array_shape':list(value.shape),'dtype':str(value.dtype)} if value.ndim>=2 else value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:compact(v) for k,v in value.items() if k!='fields'}
    if isinstance(value,(list,tuple)):return [compact(v) for v in value]
    return value

def validate_mask(mask,shape,name):
    if not isinstance(mask,np.ndarray) or mask.dtype!=np.uint8 or mask.shape!=shape or np.any((mask!=0)&(mask!=255)):raise ValueError(name+' must be original-size uint8 0/255')

def render(original,result,branches):
    out=original.copy();trusted=result['trusted_mask']>0
    out[trusted]=(out[trusted]*.78+np.array([65,230,80])*.22).astype(np.uint8)
    contours,_=cv2.findContours(result['trusted_mask'],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(out,contours,-1,(65,230,80),1)
    for i,(candidate,mask) in enumerate(zip(result['candidates'],branches)):
        color=COLORS[i%len(COLORS)];contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(out,contours,-1,color,1)
        # Thin uncertain outlines keep the original strand texture readable.
        points=np.rint(candidate['xy']).astype(np.int32)
        if len(points)>1:cv2.polylines(out,[points],False,color,1)
    cv2.rectangle(out,(12,12),(1810,112),(20,20,20),-1)
    cv2.putText(out,io.VERSION+'  '+result['status'],(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(out,'Green: trusted prefix | Color outlines: competing appearance routes | ID NOT VERIFIED',(30,92),cv2.FONT_HERSHEY_SIMPLEX,.72,(230,230,230),1)
    return out

def _crop_bounds(data,result):
    domain=(data['surface']>0)|(result['trusted_mask']>0)|(result['review_mask']>0)|(result['best_mask']>0);yy,xx=np.where(domain)
    h,w=domain.shape
    return (max(0,int(xx.min())-90),max(0,int(yy.min())-90),min(w,int(xx.max())+91),min(h,int(yy.max())+91)) if len(xx) else (0,0,w,h)

def save_qa(data,result,overlay,branches,run,role,stem,rel):
    protocol=io.read_json(io.HERE/'protocol.json');oldroot=Path(protocol['A5_output_root']);manifest=oldroot/'output_manifest.json'
    if io.sha(manifest)!=protocol['A5_output_manifest_sha256']:raise ValueError('A5 output manifest changed')
    old=oldroot/'02_原图叠加'/role/rel.name;wanted=str(old.relative_to(oldroot))
    entry=next(v for v in io.read_json(manifest)['files'] if Path(v['path'])==Path(wanted))
    if io.sha(old)!=entry['sha256']:raise ValueError('A5 comparison bytes changed')
    previous=io.read_image(old);x0,y0,x1,y1=_crop_bounds(data,result);panels=[]
    for title,im in [('Original',data['original']),('A5 saved output',previous),(io.VERSION+' trusted prefix and branches',overlay)]:
        q=im[y0:y1,x0:x1].copy();scale=min(1.,1800/q.shape[1]);q=cv2.resize(q,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        cv2.rectangle(q,(0,0),(min(q.shape[1]-1,1500),36),(15,15,15),-1);cv2.putText(q,title,(10,27),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1);panels.append(q)
    io.write_image(run/'qa'/role/(stem+'.jpg'),np.vstack(panels))
    # Branch close-up uses all original-image branch extents, including entry.
    debug=data['original'].copy()
    for i,(c,mask) in enumerate(zip(result['candidates'],branches)):
        q=np.rint(c['xy']).astype(np.int32);color=COLORS[i%len(COLORS)];cv2.polylines(debug,[q],False,color,2)
        if len(q):cv2.putText(debug,str(i)+' '+str(c.get('candidate_id','')),(int(q[-1,0]),int(q[-1,1])),cv2.FONT_HERSHEY_SIMPLEX,.55,color,1)
    crop=debug[y0:y1,x0:x1];scale=min(1.,2000/crop.shape[1]);io.write_image(run/'qa_candidates'/role/(stem+'.jpg'),cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA))

def process(row,run_name,mode):
    from pipeline import detect
    from image_beam import paint
    cv2.setNumThreads(1);start=time.time();run=io.HERE/run_name;rel=Path(row['output_file']);role=rel.parent.name;stem=rel.stem
    try:
        data=io.inputs(row);result=detect(data);shape=data['surface'].shape
        for key in ('trusted_mask','review_mask','best_mask'):validate_mask(result[key],shape,key)
        candidates=result.get('candidates',[])
        if len(candidates)>6:raise ValueError('Pipeline returned more than fixed maximum6 branches')
        branches=[];branch_map=[];paths=[]
        for i,c in enumerate(candidates):
            q=np.asarray(c['xy'],float);widths=np.asarray(c['widths'],float)
            if q.ndim!=2 or q.shape[1]!=2 or len(q)<2 or widths.shape!=(len(q),) or not np.isfinite(q).all() or not np.isfinite(widths).all() or np.any(widths<=0):raise ValueError('Invalid branch geometry')
            mask=paint(shape,q,widths);validate_mask(mask,shape,'branch');branches.append(mask)
            p=run/BRANCH_FOLDER/role/stem/('candidate_'+str(i).zfill(2)+'.png');io.write_image(p,mask);paths.append(p)
            branch_map.append({'candidate_id':c.get('candidate_id'),'path':str(p.relative_to(run)),'branch_index':i})
        overlay=render(data['original'],result,branches)
        for folder,array in zip(FOLDERS,[result['trusted_mask'],overlay,result['review_mask'],result['best_mask']]):
            p=run/folder/role/rel.name;io.write_image(p,array);paths.append(p)
        record={'output_file':row['output_file'],'camera_role':row['camera_role'],'source_group':row['source_group'],'source_time_s':row['source_time_s'],'dataset_use_status':row['dataset_use_status'],'status':result['status'],'elapsed_seconds':time.time()-start,'input_dependency_sha256':data['input_dependency_sha256'],'candidate_file_map':branch_map}
        path=run/'记录'/role/(stem+'.json');io.dump(path,{'record':record,'result':compact(result)});paths.append(path)
        if mode!='all':save_qa(data,result,overlay,branches,run,role,stem,rel)
        files=[{'path':str(p.relative_to(run)),'size':p.stat().st_size,'sha256':io.sha(p)} for p in paths]
        return {'record':record,'files':files}
    except Exception as exc:return {'failure':{'output_file':row['output_file'],'error':str(exc),'traceback':traceback.format_exc()}}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['pilot','extra','all'],default='pilot');parser.add_argument('--run',required=True);parser.add_argument('--workers',type=int,default=4);parser.add_argument('--freeze',default='');args=parser.parse_args()
    if Path(args.run).name!=args.run or args.run in ('.','..'):raise ValueError('Direct child run name required')
    run=io.HERE/args.run
    if run.exists():raise ValueError('New run directory required; preserve all old failures')
    run.mkdir();started=time.time();sources={str(io.HERE/n):io.sha(io.HERE/n) for n in CORE};sources.update({n:io.sha(n) for n in dependencies()})
    if args.freeze:
        freeze=io.HERE/args.freeze;frozen=io.read_json(freeze)
        if frozen['sources_sha256']!=sources:raise ValueError('Frozen sources differ from current implementation')
        shutil.copy2(freeze,run/'freeze_snapshot.json')
    for name,digest in io.read_json(io.HERE/'protocol.json')['source_parent_sha256'].items():
        if io.sha(name)!=digest:raise ValueError('Historical baseline changed: '+name)
    snapshot=run/'source_snapshot';snapshot.mkdir()
    for name in sources:shutil.copy2(name,snapshot/Path(name).name)
    rows=io.selection(args.mode);io.dump(run/'run_start.json',{'version':io.VERSION,'command':sys.argv,'python_executable':sys.executable,'sources_sha256':sources,'input_ids':[r['output_file'] for r in rows],'input_count':len(rows),'mode':args.mode,'python':sys.version,'opencv':cv2.__version__,'numpy':np.__version__,'platform':platform.platform(),'workers':args.workers})
    records=[];files=[];failures=[]
    with (run/'run.log').open('w',encoding='utf-8') as log,ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(process,row,args.run,args.mode):row['output_file'] for row in rows}
        for future in as_completed(jobs):
            try:answer=future.result()
            except Exception as exc:answer={'failure':{'output_file':jobs[future],'error':str(exc),'traceback':traceback.format_exc()}}
            if 'failure' in answer:failures.append(answer['failure']);line=answer['failure']['traceback']
            else:
                rec=answer['record'];records.append(rec);files+=answer['files'];line=str(len(records))+'/'+str(len(rows))+' '+rec['output_file']+' '+rec['status']+' %.1fs'%rec['elapsed_seconds']
            log.write(line+'\n');log.flush();print(line,flush=True);io.dump(run/'progress.json',{'completed':len(records),'failed':len(failures),'total':len(rows)})
    unchanged=all(io.sha(name)==digest for name,digest in sources.items());exit_code=0 if not failures and len(records)==len(rows) and unchanged else 1
    io.dump(run/'output_manifest.json',{'version':io.VERSION,'case_count':len(records),'files':files})
    io.dump(run/'summary.json',{'version':io.VERSION,'mode':args.mode,'completed':len(records),'planned':len(rows),'failures':failures,'statuses':dict(Counter(r['status'] for r in records)),'records':records,'sources_unchanged':unchanged,'elapsed_seconds':time.time()-started,'exit_code':exit_code})
    return exit_code
if __name__=='__main__':sys.exit(main())
