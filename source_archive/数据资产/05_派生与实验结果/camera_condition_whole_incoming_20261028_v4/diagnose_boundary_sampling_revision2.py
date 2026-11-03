"""Fixed16 read-only geometry diagnosis; no image-truth scoring or tuning."""
from pathlib import Path
import importlib.util,json,hashlib,sys,time
from collections import Counter
import cv2,numpy as np
import boundary_occlusion_revision2 as new
from whole_path import build_search_surface

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'camera_condition_whole_incoming_20261028_v3'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
old=load('boundary_sampling_old_a3',OLD/'boundary_occlusion.py')
io=load('boundary_sampling_old_io',OLD/'whole_io.py')

def main():
 cv2.setNumThreads(1);cases=[];started=time.time()
 for r in io.selection('pilot'):
  rel=Path(r['output_file']);saved=OLD/'all_001/记录'/rel.parent.name/(rel.stem+'.json')
  record=io.read_json(saved);geo=record['free']['geometry']
  if geo.get('status')!='observed':
   cases.append({'output_file':r['output_file'],'status':'not_applicable','free_status':geo.get('status')});continue
  data=io.inputs(r);S=data['surface'];search,searchdiag=build_search_surface(S);pitch=data['pitch']
  origin,d,n,width,end=new._geometry(geo,pitch)
  og,od=old._seed_rows(data['curve_refs'],origin,d,n,end,width,pitch,S)
  ng,nd=new._seed_rows(data['curve_refs'],origin,d,n,end,width,pitch,S)
  hg,hd=new._seed_rows(data['curve_refs'],origin,d,n,end,width,pitch,search)
  common={'output_file':r['output_file'],'status':'diagnosed','saved_geometry_source':str(saved),'saved_geometry_sha256':io.sha(saved),'input_dependencies':data['input_dependency_sha256'],'pitch_px':pitch,'width_px':width,'origin_xy':origin.tolist(),'visible_end_xy':end.tolist(),'visible_end_s_px':float(np.dot(end-origin,d)),'old_probe_count':len(og),'new_saved_S_probe_count':len(ng),'new_search_domain_probe_count':len(hg),'old_probe_s_range_px':[float(min(g[0]['free_s'] for g in og)),float(max(g[0]['free_s'] for g in og))] if og else None,'new_search_probe_s_range_px':[float(min(g[0]['free_s'] for g in hg)),float(max(g[0]['free_s'] for g in hg))] if hg else None,'new_saved_S_diagnostics':nd,'new_search_diagnostics':hd,'search_domain':searchdiag}
  cases.append(common);print(rel.name,len(og),len(ng),len(hg),hd['retained_before_old_lower'],flush=True)
  del data
 valid=[c for c in cases if c['status']=='diagnosed']
 restored=[c for c in valid if c['new_search_diagnostics']['retained_before_old_lower']>0]
 report={'scope':'Fixed16 previous all_001 automatic geometry, same F3 references, new entry/probe geometry only. Original raw image loaded by verified old I/O but not scored. No new full-path predictions or truth evaluation.','count':len(cases),'applicable':len(valid),'old_probe_count':sum(c['old_probe_count'] for c in valid),'new_saved_S_probe_count':sum(c['new_saved_S_probe_count'] for c in valid),'new_search_domain_probe_count':sum(c['new_search_domain_probe_count'] for c in valid),'cases_with_restored_pre_endpoint_probes':len(restored),'restored_pre_endpoint_probes':sum(c['new_search_diagnostics']['retained_before_old_lower'] for c in valid),'cases':cases,'elapsed_seconds':time.time()-started,'sources_sha256':{str(p):io.sha(p) for p in [HERE/'boundary_occlusion_revision2.py',HERE/'whole_path.py',OLD/'boundary_occlusion.py']}}
 path=HERE/'boundary_sampling_diagnosis_revision2.json'
 if path.exists():raise ValueError('Never overwrite diagnosis receipt')
 io.dump(path,report);print(json.dumps({k:v for k,v in report.items() if k not in ('cases','sources_sha256')},ensure_ascii=False))

if __name__=='__main__':main()
