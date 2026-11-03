"""Read-only F3 all400 saved raster/endpoint audit; no detector imports."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from datetime import datetime,timezone
import json,hashlib,time
import cv2,numpy as np
R=Path(__file__).resolve().parents[1];ROOT=R.parent;F2=ROOT.parent/'camera_condition_seam_joint_20261028_v2'/'all_001'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def im(p):return cv2.imdecode(np.frombuffer(p.read_bytes(),np.uint8),0)
def raster(xy):
 q=np.rint(xy).astype(np.int32);lo=q.min(axis=0);hi=q.max(axis=0);m=np.zeros((hi[1]-lo[1]+1,hi[0]-lo[0]+1),np.uint8);cv2.polylines(m,[q-lo],False,255,1,cv2.LINE_8);yy,xx=np.nonzero(m);return np.c_[xx+lo[0],yy+lo[1]]
def angle(xy,side):
 q=xy if side=='left' else xy[::-1];ss=np.r_[0,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))];v=np.arange(0,min(9.,ss[-1])+.01,3.)
 if len(v)<3:return 0.
 a=np.c_[np.interp(v,ss,q[:,0]),np.interp(v,ss,q[:,1])];d=np.diff(a,axis=0);d/=np.maximum(np.linalg.norm(d,axis=1),1e-9)[:,None];return float(np.degrees(np.arccos(np.clip(np.sum(d[1:]*d[:-1],axis=1),-1,1))).max())
def process(rel):
 cv2.setNumThreads(1);part=Path(rel);role=part.parent.name;cp=R/'记录/cases'/role/(part.stem+'.json');j=read(cp)
 base=im(F2/'记录/masks/merged'/role/part.name);merged=im(R/'记录/masks/merged'/role/part.name);t=im(R/'记录/masks/transition'/role/part.name)
 out={'output_file':rel,'case_sha256':sha(cp),'accepted_groups':len(j['groups']),'accepted_points':len(j['accepted_points']),'new_pixels':int(np.count_nonzero((merged>0)&(base==0))),'old_pixels_removed':int(np.count_nonzero((base>0)&(merged==0))),'outside_T_changes':int(np.count_nonzero((base!=merged)&(t==0))),'same_x_vertical_steps_gt1px':0,'non_monotonic_groups':0,'missing_new_path_pixels':0,'endpoint_count':0,'unanchored_fragment_ends':0,'connection_failures':0,'endpoint_turns_gt25':0,'maximum_endpoint_angle_3px':0.,'endpoints':[],'flagged_groups':[]}
 for g in j['groups']:
  xy=np.asarray(g['sampled_points_xy']);q=raster(xy);missing=int(np.count_nonzero(merged[q[:,1],q[:,0]]==0));v=np.diff(xy,axis=0);vertical=int(np.count_nonzero((abs(v[:,0])<1e-8)&(abs(v[:,1])>1.)));monotone=bool(np.all(v[:,0]>0))
  out['missing_new_path_pixels']+=missing;out['same_x_vertical_steps_gt1px']+=vertical;out['non_monotonic_groups']+=int(not monotone)
  if missing or vertical or not monotone:out['flagged_groups'].append({'group_id':g['group_id'],'missing_pixels':missing,'vertical_steps_gt1px':vertical,'strict_increasing_x':monotone})
  for side in ('left','right'):
   a=g['anchors'][side]
   if a is None:out['unanchored_fragment_ends']+=1;continue
   x,y=np.rint(a).astype(int);distance=int(np.max(abs(q-np.array([x,y])),axis=1).min());connected=bool(base[y,x] and merged[y,x] and distance==0);turn=angle(xy,side)
   out['endpoint_count']+=1;out['connection_failures']+=int(not connected);out['endpoint_turns_gt25']+=int(turn>25);out['maximum_endpoint_angle_3px']=max(out['maximum_endpoint_angle_3px'],turn)
   out['endpoints'].append({'group_id':g['group_id'],'side':side,'anchor':a,'connected_to_old_F2_pixel':connected,'distance_to_new_path_chebyshev_px':distance,'angle_3px_deg':turn})
 return out

def main():
 start=time.perf_counter();summary=read(R/'summary.json');assert summary['completed']==400 and summary['failed']==0
 context=read(R/'run_context.json');rows=context['selection'];assert len(rows)==400
 dest=R/'记录/endpoint_shape_review_all.json';assert not dest.exists();results=[]
 with ProcessPoolExecutor(max_workers=4) as pool:
  for i,c in enumerate(pool.map(process,rows),1):
   results.append(c)
   if i%50==0:print(json.dumps({'completed':i,'planned':400,'elapsed_s':round(time.perf_counter()-start,2)}),flush=True)
 keys=['accepted_groups','accepted_points','new_pixels','old_pixels_removed','outside_T_changes','same_x_vertical_steps_gt1px','non_monotonic_groups','missing_new_path_pixels','endpoint_count','unanchored_fragment_ends','connection_failures','endpoint_turns_gt25'];total={k:sum(c[k] for c in results) for k in keys};total['maximum_endpoint_angle_3px']=max(c['maximum_endpoint_angle_3px'] for c in results)
 failure_keys=['old_pixels_removed','outside_T_changes','same_x_vertical_steps_gt1px','non_monotonic_groups','missing_new_path_pixels','connection_failures','endpoint_turns_gt25']
 report={'version':'RB-F3.0.0','run':'all_001','planned':400,'completed':len(results),'passed':all(total[k]==0 for k in failure_keys),'meaning':'Saved prediction raster/shape checks only; not visual truth or physical material identity. Unanchored fragment ends are explicitly allowed. Turn measured in 3px arc windows near attached ends.','totals':total,'cases':results,'script_sha256':sha(Path(__file__)),'run_context_sha256':sha(R/'run_context.json'),'freeze_sha256':sha(ROOT/'freeze_all.json'),'finished_utc':datetime.now(timezone.utc).isoformat(),'elapsed_s':round(time.perf_counter()-start,2),'algorithm_modified':False,'detector_rerun':False,'accuracy':None}
 dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in report.items() if k!='cases'},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
