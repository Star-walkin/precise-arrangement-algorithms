"""离线可复现运行；先保存预测，再进行独立图像检查。"""
import argparse,sys,time,traceback,platform,shutil
from collections import Counter
import cv2,numpy as np
import whole_io as io
from free_rope import refine_and_extend,dependencies
from boundary_occlusion import analyze_boundary_occlusion
from whole_path import trace_whole, build_search_surface

def compact(x):
 if isinstance(x,np.ndarray) and x.ndim>=2:return {'array_shape':list(x.shape),'dtype':str(x.dtype)}
 if isinstance(x,dict):return {k:compact(v) for k,v in x.items() if k!='fields'}
 if isinstance(x,list):return [compact(v) for v in x]
 return x

def render(image,whole,supported,result):
 out=image.copy();mask=whole>0;support=supported>0
 support&=result.get('review_mask',np.zeros_like(whole))==0
 color=np.zeros_like(out);color[:]=(0,160,255);color[support]=(60,220,70)
 out[mask]=(out[mask]*.60+color[mask]*.40).astype(np.uint8)
 contours,_=cv2.findContours(whole,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(out,contours,-1,(50,230,80),1)
 cv2.rectangle(out,(12,12),(1540,112),(20,20,20),-1)
 cv2.putText(out,io.VERSION+'  '+result['status'],(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
 cv2.putText(out,'Green: image support | Orange: continuation needs review | ID NOT VERIFIED',(30,92),cv2.FONT_HERSHEY_SIMPLEX,.75,(230,230,230),1)
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['pilot','all'],default='pilot');p.add_argument('--run',required=True);p.add_argument('--only',default='');p.add_argument('--shard',type=int,default=-1);p.add_argument('--shards',type=int,default=1);args=p.parse_args()
 run=io.HERE/args.run
 if args.shard<0:
  if run.exists():raise ValueError('Never overwrite an experiment run')
  run.mkdir();context=run
 else:
  assert args.mode=='all' and 0<=args.shard<args.shards
  run.mkdir(exist_ok=True);context=run/'分片记录'/f'part_{args.shard:02d}'
  if context.exists():raise ValueError('Never overwrite a shard')
  context.mkdir(parents=True)
 rows=io.selection(args.mode)
 if args.shard>=0:rows=[r for i,r in enumerate(rows) if i%args.shards==args.shard]
 if args.only:rows=[r for r in rows if args.only in r['output_file']]
 sources={str(io.HERE/f):io.sha(io.HERE/f) for f in ['whole_io.py','free_rope.py','boundary_occlusion.py','whole_path.py','run_whole.py','protocol.json','protocol_addendum_001.json','protocol_addendum_002.json']}
 sources.update({str(f):io.sha(f) for f in dependencies()})
 (context/'source_snapshot').mkdir()
 for source in sources:shutil.copy2(source,context/'source_snapshot'/io.Path(source).name)
 io.dump(context/'run_start.json',{'version':io.VERSION,'command':sys.argv,'sources_sha256':sources,'inputs':[r['output_file'] for r in rows],'python':sys.version,'opencv':cv2.__version__,'numpy':np.__version__,'platform':platform.platform()})
 results=[];failures=[];files=[];start=time.time();cv2.setNumThreads(1)
 for i,r in enumerate(rows):
  t=time.time();rel=io.Path(r['output_file']);role=rel.parent.name;stem=rel.stem
  try:
   data=io.inputs(r);free=refine_and_extend(data)
   search_surface,search_info=build_search_surface(data['surface'])
   occ=analyze_boundary_occlusion(data['original'],data['surface'],data['merged'],data['inferred'],data['curve_refs'],free['geometry'],data['exclusion'],data['pitch'],search_surface=search_surface)
   occ['diagnostics']['search_domain_construction']=search_info
   result=trace_whole(data,free,occ)
   overlay=render(data['original'],result['whole_mask'],result['supported_mask'],result)
   paths=[run/'01_整根候选掩膜'/role/rel.name,run/'02_原图叠加'/role/rel.name,run/'03_实测支持掩膜'/role/rel.name,run/'04_接续待确认'/role/rel.name,run/'记录'/role/(stem+'.json')]
   for path,image in zip(paths,[result['whole_mask'],overlay,result['supported_mask'],result['review_mask']]):io.write_image(path,image)
   record={'output_file':r['output_file'],'camera_role':r['camera_role'],'source_group':r['source_group'],'source_time_s':r['source_time_s'],'dataset_use_status':r['dataset_use_status'],'status':result['status'],'elapsed_seconds':time.time()-t,'input_dependency_sha256':data['input_dependency_sha256']}
   io.dump(paths[-1],{'record':record,'free':compact(free),'occlusion':compact(occ),'whole':compact(result)})
   if args.mode=='pilot':
    if free['geometry'].get('status')=='observed':
     # 公平简单对照：同一自动自由入口，仅作有限直线延长，不用新搜索。
     geo=free['geometry'];d=np.asarray(geo['direction_xy']);origin=np.asarray(geo['origin_xy']);width=float(geo['width_px']);base=data['original'].copy();seen=False;old=None
     for dist in np.arange(0,2500,4):
      q=origin+dist*d;x,y=np.rint(q).astype(int)
      if not(0<=x<base.shape[1] and 0<=y<base.shape[0]):break
      inside=data['surface'][y,x]>0
      if seen and not inside:break
      seen|=inside
      if old is not None:cv2.line(base,tuple(old),tuple([x,y]),(240,100,240),max(1,int(width)))
      old=np.array([x,y])
     io.write_image(run/'geometry_baseline'/role/rel.name,base)
    points=np.argwhere((result['whole_mask']>0)|(data['surface']>0));y0=max(0,int(points[:,0].min())-100) if len(points) else 1400;x0=max(0,int(points[:,1].min())-100) if len(points) else 500;y1=min(3000,int(points[:,0].max())+100) if len(points) else 3000;x1=min(4096,int(points[:,1].max())+100) if len(points) else 3500
    # 无绘线原图与叠加并列，保留原像素供人工比较。
    a=data['original'][y0:y1,x0:x1];b=overlay[y0:y1,x0:x1];view=np.vstack([a,b]);io.write_image(run/'qa'/role/(stem+'.jpg'),view)
   results.append(record)
   files.extend([{'path':str(q.relative_to(run)),'size':q.stat().st_size,'sha256':io.sha(q)} for q in paths])
   print(i+1,len(rows),r['output_file'],result['status'],round(time.time()-t,2),flush=True)
  except Exception as e:
   failure={'output_file':r['output_file'],'error':str(e),'traceback':traceback.format_exc()};failures.append(failure);print(failure['traceback'],flush=True)
  io.dump(context/'progress.json',{'completed':len(results),'failed':len(failures),'total':len(rows)})
 io.dump(context/'output_manifest.json',{'version':io.VERSION,'files':files,'case_count':len(results)})
 io.dump(context/'summary.json',{'version':io.VERSION,'completed':len(results),'failures':failures,'statuses':dict(Counter(r['status'] for r in results)),'records':results,'elapsed_seconds':time.time()-start,'exit_code':1 if failures else 0})
 return 1 if failures else 0
if __name__=='__main__':sys.exit(main())
