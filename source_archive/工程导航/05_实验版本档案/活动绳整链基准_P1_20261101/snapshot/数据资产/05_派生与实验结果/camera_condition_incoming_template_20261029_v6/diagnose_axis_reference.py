"""只读机制诊断：精确自由轴是否因束搜索剪枝而丢失。无人工路径/阈值调参。"""
import json,sys
from pathlib import Path
import numpy as np
import whole_io as io
from pipeline import prepare_entries
from appearance_template import build_fields,evaluate_template
from image_beam import domain_from_surface,sample

def probe(data):
 entries,_,_=prepare_entries(data);reports=[]
 for entry in entries:
  g=entry['free']['geometry'];o=np.array(g['origin_xy']);d=np.array(g['direction_xy']);w=float(g['width_px']);ss=np.arange(0.,2641,8.);xy=o+ss[:,None]*d
  fields=build_fields(data['original'],w,data['pitch']);domain=domain_from_surface(data['surface'])
  for ref in [w,float(entry['template'].get('reference_width_px',w))]:
   fade=np.clip(ss/(4*w),0,1);fade=fade*fade*(3-2*fade);ww=w+(ref-w)*fade
   m=evaluate_template(fields,xy,d,ww,entry['template']);inside=sample(domain,xy)>76
   weak=0;stop=len(ss)
   for i in range(1,len(ss)):
    weak=0 if m['supported'][i] else weak+8
    if weak>6*w:stop=i;break
   good=np.flatnonzero(m['supported'][:stop]);end=int(good[-1]) if len(good) else 0
   reports.append({'entry':entry['entry_id'],'width':w,'target_width':ref,'stop_s_px':float(ss[stop-1]),'last_supported_before_stop':float(ss[end]),'surface_entry_s_px':float(ss[np.flatnonzero(inside)[0]]) if inside.any() else None,'xy':xy.tolist(),'supported':m['supported'].tolist(),'score':m['score'].tolist(),'profile_match':m['profile_match'].tolist(),'raw_pair':m['raw_pair'].tolist()})
 return reports

if __name__=='__main__':
 target=io.HERE/sys.argv[1]
 if target.exists():raise ValueError('preserve old probe')
 ids={'images/left_camera/L_run_20260923_01__f000015__left.png','images/right_camera/dual_20260923_101643__pair0315__right.png'}
 result=[]
 for row in io.selection('pilot'):
  if row['output_file'] not in ids:continue
  result.append({'output_file':row['output_file'],'probe':probe(io.inputs(row))})
 io.dump(target,{'purpose':'candidate-loss mechanism only; no labels, no full pilot','cases':result})
 print(json.dumps([{'id':v['output_file'],'variants':[{k:r[k] for k in ('width','target_width','stop_s_px','last_supported_before_stop','surface_entry_s_px')} for r in v['probe']]} for v in result]))
