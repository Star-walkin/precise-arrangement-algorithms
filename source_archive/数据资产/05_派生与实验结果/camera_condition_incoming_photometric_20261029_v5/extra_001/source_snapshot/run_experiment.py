"""固定开放图片离线实验：新目录、预测先保存、原图和旧版本对照分开。"""
import argparse,sys,time,platform,traceback,shutil
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter
import cv2,numpy as np
import whole_io as io
from free_rope import refine_and_extend,dependencies
from photometric_trace import trace_photometric,paint

CORE=['whole_io.py','free_rope.py','photometric_ribbon.py','attached_groove.py','lane_candidates.py','photometric_trace.py','run_experiment.py','protocol.json']
def compact(x):
 if isinstance(x,np.ndarray) and x.ndim>=2:return {'array_shape':list(x.shape),'dtype':str(x.dtype)}
 if isinstance(x,dict):return {k:compact(v) for k,v in x.items() if k!='fields'}
 if isinstance(x,list):return [compact(v) for v in x]
 return x
def render(image,result):
 out=image.copy();whole=result['whole_mask']>0;support=(result['supported_mask']>0)&(result['review_mask']==0)
 color=np.zeros_like(out);color[:]=(0,160,255);color[support]=(60,220,70)
 out[whole]=(out[whole]*.60+color[whole]*.40).astype(np.uint8)
 contours,_=cv2.findContours(result['whole_mask'],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(out,contours,-1,(50,230,80),1)
 cv2.rectangle(out,(12,12),(1700,112),(20,20,20),-1)
 cv2.putText(out,io.VERSION+'  '+result['status'],(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
 cv2.putText(out,'Green: image support | Orange: inferred / competing continuation | ID NOT VERIFIED',(30,92),cv2.FONT_HERSHEY_SIMPLEX,.75,(230,230,230),1)
 return out
def process(row,run_name,mode):
 cv2.setNumThreads(1);start=time.time();run=io.HERE/run_name;rel=io.Path(row['output_file']);role=rel.parent.name;stem=rel.stem
 try:
  data=io.inputs(row);free=refine_and_extend(data);result=trace_photometric(data,free);overlay=render(data['original'],result)
  paths=[run/k/role/rel.name for k in ['01_整根候选掩膜','02_原图叠加','03_图像支持','04_接续待确认']]
  for p,a in zip(paths,[result['whole_mask'],overlay,result['supported_mask'],result['review_mask']]):io.write_image(p,a)
  record={'output_file':row['output_file'],'camera_role':row['camera_role'],'source_group':row['source_group'],'source_time_s':row['source_time_s'],'dataset_use_status':row['dataset_use_status'],'status':result['status'],'elapsed_seconds':time.time()-start,'input_dependency_sha256':data['input_dependency_sha256']}
  paths.append(run/'记录'/role/(stem+'.json'));io.dump(paths[-1],{'record':record,'free':compact(free),'whole':compact(result)})
  if mode!='all':
   old=io.ASSETS/'camera_condition_whole_incoming_20261028_v4/all_002/02_原图叠加'/role/rel.name
   p=io.read_json(io.HERE/'protocol.json');manifest=io.Path(p['A4_output_root'])/'output_manifest.json';assert io.sha(manifest)==p['A4_output_manifest_sha256']
   ent=next(f for f in io.read_json(manifest)['files'] if f['path']==str(old.relative_to(manifest.parent)))
   assert io.sha(old)==ent['sha256']
   older=io.read_image(old)
   ys,xs=np.where((data['surface']>0)|(result['whole_mask']>0));x0=max(0,int(xs.min())-70) if len(xs) else 0;y0=max(0,int(ys.min())-70) if len(ys) else 0;x1=min(4096,int(xs.max())+71) if len(xs) else 4096;y1=min(3000,int(ys.max())+71) if len(ys) else 3000
   panels=[]
   for name,im in [('Original',data['original']),('A4 saved baseline',older),(io.VERSION+' selected',overlay)]:
    crop=im[y0:y1,x0:x1].copy();cv2.rectangle(crop,(0,0),(min(1200,crop.shape[1]),35),(0,0,0),-1);cv2.putText(crop,name,(10,26),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1);panels.append(crop)
   io.write_image(run/'qa'/role/(stem+'.jpg'),np.vstack(panels))
   debug=data['original'].copy();colors=[(40,200,255),(255,100,60),(100,255,80),(220,70,220)]
   ranked=sorted(result['candidates'],key=lambda c:(c['candidate_id']!=result.get('selected_candidate_id'),-c['score']))
   for c,color in zip(ranked[:4],colors):
    q=np.rint(c['xy']).astype(np.int32);cv2.polylines(debug,[q],False,color,3)
    if len(q):cv2.putText(debug,c['candidate_id']+' %.3f'%c['score'],tuple(q[-1]),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
   io.write_image(run/'qa_candidates'/role/(stem+'.jpg'),debug[y0:y1,x0:x1])
   g0=result.get('g0_candidate')
   if g0:
    geo=render(data['original'],{'whole_mask':paint(data['surface'].shape,g0['xy'],g0['widths']),'supported_mask':np.zeros(data['surface'].shape,np.uint8),'review_mask':paint(data['surface'].shape,g0['xy'],g0['widths']),'status':'G0 GEOMETRY ONLY; NOT IMAGE SELECTED'})
    io.write_image(run/'qa_g0'/role/(stem+'.jpg'),geo[y0:y1,x0:x1])
  files=[{'path':str(p.relative_to(run)),'size':p.stat().st_size,'sha256':io.sha(p)} for p in paths]
  return {'record':record,'files':files}
 except Exception as e:return {'failure':{'output_file':row['output_file'],'error':str(e),'traceback':traceback.format_exc()}}

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['pilot','extra','all'],default='pilot');p.add_argument('--run',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--freeze',default='');a=p.parse_args();run=io.HERE/a.run
 assert not run.exists(),'New experiment directory required';run.mkdir()
 sources={str(io.HERE/n):io.sha(io.HERE/n) for n in CORE};sources.update({n:io.sha(n) for n in dependencies()})
 if a.freeze:
  frozen=io.read_json(io.HERE/a.freeze);assert frozen['sources_sha256']==sources
  shutil.copy2(io.HERE/a.freeze,run/'freeze_snapshot.json')
 for n,h in io.read_json(io.HERE/'protocol.json')['source_parent_sha256'].items():assert io.sha(n)==h
 snapshot=run/'source_snapshot';snapshot.mkdir()
 for n in sources:shutil.copy2(n,snapshot/io.Path(n).name)
 rows=io.selection(a.mode);io.dump(run/'run_start.json',{'version':io.VERSION,'command':sys.argv,'python_executable':sys.executable,'sources_sha256':sources,'input_ids':[r['output_file'] for r in rows],'input_count':len(rows),'python':sys.version,'opencv':cv2.__version__,'numpy':np.__version__,'platform':platform.platform(),'workers':a.workers})
 start=time.time();records=[];files=[];failures=[]
 with (run/'run.log').open('w',encoding='utf-8') as log,ProcessPoolExecutor(max_workers=a.workers) as executor:
  jobs=[executor.submit(process,r,a.run,a.mode) for r in rows]
  for future in as_completed(jobs):
   answer=future.result()
   if 'failure' in answer:failures.append(answer['failure']);line=answer['failure']['traceback']
   else:
    rec=answer['record'];records.append(rec);files+=answer['files'];line=f"{len(records)}/{len(rows)} {rec['output_file']} {rec['status']} {rec['elapsed_seconds']:.1f}s"
   log.write(line+'\n');log.flush();print(line,flush=True)
   io.dump(run/'progress.json',{'completed':len(records),'failed':len(failures),'total':len(rows)})
 unchanged=all(io.sha(n)==h for n,h in sources.items());exit_code=0 if not failures and len(records)==len(rows) and unchanged else 1
 io.dump(run/'output_manifest.json',{'version':io.VERSION,'case_count':len(records),'files':files})
 io.dump(run/'summary.json',{'version':io.VERSION,'mode':a.mode,'completed':len(records),'planned':len(rows),'failures':failures,'statuses':dict(Counter(r['status'] for r in records)),'records':records,'sources_unchanged':unchanged,'elapsed_seconds':time.time()-start,'exit_code':exit_code})
 return exit_code
if __name__=='__main__':sys.exit(main())
