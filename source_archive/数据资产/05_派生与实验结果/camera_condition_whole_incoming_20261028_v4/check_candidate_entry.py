"""Synthetic trace-level regression for candidates that never reach the surface.

Only profile measurements and tangent proposals are stubbed. The actual beam,
state propagation, terminal reconstruction, support trimming and mask output run.
No original images, labels, models or devices are accessed.
"""
from pathlib import Path
import hashlib,importlib.util,json,sys
import cv2,numpy as np

HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def run_trace(module,case):
 shape=(320,640);surface=np.zeros(shape,np.uint8)
 if case=='never_enters':surface[180:260,150:500]=255
 elif case in ('enters','weak_only_entry'):surface[40:100,150:500]=255
 elif case=='starts_inside':surface[40:100,50:500]=255
 elif case!='empty_search':raise ValueError(case)
 image=np.zeros((*shape,3),np.uint8);free_mask=np.zeros(shape,np.uint8);straight=np.zeros(shape,np.uint8)
 free_mask[55:66,20:61]=255;straight[55:66,61:81]=255
 free={'status':'observed','geometry':{'status':'observed','origin_xy':[60.,60.],'direction_xy':[1.,0.],'width_px':10.,'visible_end_xy':[80.,60.]},'free_segment':{'observed_segment_xy':[[20.,60.],[60.,60.]]},'free_mask':free_mask,'straight_observed_mask':straight,'texture_descriptor':{'status':'unknown'}}
 data={'original':image,'surface':surface,'pitch':20.,'curve_refs':[]};before=surface.copy()
 originals={name:getattr(module,name) for name in ('_fields','_profiles','_tangent_zones')}
 def profiles(fields,centers,slopes,width,xsign,texture=None):
  n=len(centers);support=np.ones(n,bool)
  if case=='weak_only_entry':support=centers[:,0]<150
  return np.ones(n),support,np.full(n,width),{'raw_weak':np.full(n,20.),'texture_score':np.zeros(n),'single_edge_support':np.zeros(n,bool)}
 try:
  module._fields=lambda *args:None
  module._profiles=profiles
  module._tangent_zones=lambda *args:[]
  result=module.trace_whole(data,free,{'candidates':[]})
 finally:
  for name,value in originals.items():setattr(module,name,value)
 assert np.array_equal(surface,before),'trace modified the source surface'
 assert set(np.unique(result['whole_mask'])).issubset({0,255})
 c=next((p for p in result.get('candidates',[]) if p.get('candidate_id')==result.get('selected_candidate_id')),None)
 search,_=module.build_search_surface(surface)
 inside=0
 if c:
  q=np.rint(c['xy']).astype(int);valid=(q[:,0]>=0)&(q[:,0]<shape[1])&(q[:,1]>=0)&(q[:,1]<shape[0]);inside=int(np.count_nonzero(search[q[valid,1],q[valid,0]]))
 detail={'status':result['status'],'reason':result['diagnostics'].get('reason'),'candidate_count':len(result.get('candidates',[])),'selected_inside_point_count':inside,'mask_equals_measured_fallback':bool(np.array_equal(result['whole_mask'],free_mask|straight)),'whole_pixels':int(np.count_nonzero(result['whole_mask'])),'original_surface_unchanged':True,'diagnostics':result['diagnostics']}
 return result,detail

def main():
 source=HERE/'whole_path.py';before_hash=sha(source);new=load('candidate_entry_new',source)
 old_path=HERE/'pilot_002/source_snapshot/whole_path.py';old=load('candidate_entry_frozen_before_fix',old_path)
 old_results={}
 for case in ('never_enters','weak_only_entry'):
  _,d=run_trace(old,case);assert d['candidate_count']>0 and d['selected_inside_point_count']==0
  old_results[case]=d
 results={}
 for case in ('never_enters','empty_search','weak_only_entry'):
  r,d=run_trace(new,case)
  assert d['candidate_count']==0,(case,d)
  assert d['mask_equals_measured_fallback'],(case,d)
  expected_reason='no_significant_surface_for_bounded_search' if case=='empty_search' else 'no_candidate_reaches_search_surface'
  assert d['reason']==expected_reason,(case,d)
  assert r['status'].startswith('FREE_ONLY'),(case,d)
  results[case]=d
 for case in ('enters','starts_inside'):
  _,d=run_trace(new,case);assert d['candidate_count']>0 and d['selected_inside_point_count']>0,(case,d)
  assert not d['mask_equals_measured_fallback'],(case,d)
  results[case]=d
 assert sha(source)==before_hash,'algorithm changed during synthetic check'
 receipt={'scope':'Actual trace state/terminal/trim/mask path with synthetic arrays and constant profile stubs. Structural regression only, not real-image accuracy.','passed':True,'frozen_before_fix':old_results,'after_fix':results,'all_original_surfaces_unmodified':True,'source_sha256':{str(source):before_hash,str(old_path):sha(old_path),str(Path(__file__)):sha(Path(__file__))},'python':sys.version,'opencv':cv2.__version__,'numpy':np.__version__,'images_or_labels_read':False,'equipment_connected':False,'exit_code':0}
 path=HERE/'candidate_entry_selfcheck_001.json'
 if path.exists():raise ValueError('Preserve previous check receipt')
 path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 print(json.dumps({'passed':True,'receipt':str(path),'source_sha256':before_hash,'cases':{k:{q:v for q,v in d.items() if q!='diagnostics'} for k,d in results.items()}},ensure_ascii=False))

if __name__=='__main__':main()
