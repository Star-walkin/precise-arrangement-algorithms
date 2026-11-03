"""Small frozen offline experiment entry point. Does not import production code."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import paired_audit as pa
import blind_review as br
import strip_motion as sm
import source_sampling as ss

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
V26=ROOT/'vision_geometry_lab/outputs/v26_causal_material/v26_20261005_001'
OUT=ROOT/'vision_geometry_lab/outputs/v27_material_observability/v27_20261006_002'
SOURCES={
 'left':ROOT/'现场视频/训练采集/双相机同步/dual_20260924_111500/left_DA9627065.avi',
 'right':ROOT/'现场视频/训练采集/双相机同步/dual_20260924_111143/right_DA9626975.avi'}
CONFIG={'run_id':'v27_20261006_002','maximum_external_models':1,
 'maximum_new_baseline_configs':2,'paired_wall_budget_s':600,'strip_wall_budget_s':900,
 'blind_wall_budget_s':300,'holdouts_excluded':True,'hardware_access':False,
 'CoTracker3':'conditional_not_started_license_unconfirmed',
 'candidate_separability':'not_started_independent_material_labels_absent',
 'label_files_read_by_inference':[],'online_deployment':False}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def write(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()

def freeze():
 if OUT.exists():raise FileExistsError('Existing run must never be overwritten')
 old=read(V26/'protocol.json')
 inputs=pa.input_paths(V26)
 for p in SOURCES.values():
  inputs[rel(p)]=p; q=p.with_name(p.name.split('_')[0]+'_frames.csv');inputs[rel(q)]=q
 for side in SOURCES:
  p=V26/f'pixel_hashes/{side}.json';inputs[rel(p)]=p
 for p in (V26/'posthoc/review.json',ROOT/'vision_prototype/dual_camera_recorder.py',ROOT/'vision_prototype/live_camera.py'):
  inputs[rel(p)]=p
 # Frozen historical source files remain unchanged too; paths come from V26 only.
 for key in old.get('source_sha256',{}):inputs[key]=ROOT/key
 hashes={k:sha(v) for k,v in inputs.items()}
 for k,v in old.get('source_sha256',{}).items():
  if hashes[k]!=v:raise ValueError('V26 source changed: '+k)
 code={rel(p):sha(p) for p in HERE.glob('*.py')}
 launcher=ROOT/'vision_geometry_lab/复核V27材料与换层.cmd';code[rel(launcher)]=sha(launcher)
 import cv2
 protocol={'frozen_utc':datetime.now(timezone.utc).isoformat(),'config':CONFIG,
   'paired_config':pa.CONFIG,'strip_config':sm.CONFIG,'blind_config':br.CONFIG,'sampling_config':ss.CONFIG,
   'source_sha256':code,'input_sha256':hashes,'v26_protocol_sha256':sha(V26/'protocol.json'),
   'python':sys.version,'numpy':np.__version__,'opencv':cv2.__version__,
   'source_videos':{s:str(p) for s,p in SOURCES.items()},
   'truth_status':{'material':0,'independent_t1':0,'angle':0},
   'old_gates_unchanged':old.get('gates'),
   'license_preflight':{'model':'CoTracker3 online','commit':'82e02e8029753ad4ef13cf06be7f4fc5facdda4d',
     'license':'CC-BY-NC-4.0 majority code','enterprise_use_authorization':'unconfirmed',
     'model_inference_executed':False,'weight_sha256':None,'conditional_status':'not_executed'},
   'causality':'diagnostic replay; no future geometry selection; no claims of measured live latency',
   'previous_attempt':{'run_id':'v27_20261006_001','status':'paired_and_strip_stopped_at_manifest_schema_guard_before_measurement',
     'repair':'use V26 feature_sha256 for features, not cohort input_hashes; no threshold or algorithm configuration changes',
     'source_snapshot':'../v27_20261006_001/source_snapshot','earlier_outputs_preserved':True}}
 OUT.mkdir(parents=True,exist_ok=False);write(OUT/'protocol.json',protocol)
 print(json.dumps({'protocol_sha256':sha(OUT/'protocol.json'),'inputs':len(hashes),'sources':len(code)}),flush=True)

def verify():
 p=read(OUT/'protocol.json')
 for collection in ('source_sha256','input_sha256'):
  for key,value in p[collection].items():
   if sha(ROOT/key)!=value:raise ValueError('Frozen hash mismatch: '+key)
 return p

def run_strip(p,digest):
 folder=OUT/'strips'
 if folder.exists():raise FileExistsError('Refuse existing strip outputs')
 folder.mkdir()
 geometry={s:pa.load_geometry_records(V26,s,expected_input_sha256=p['input_sha256']) for s in SOURCES}
 reports=[];started=time.perf_counter()
 for side in pa.CONFIG['sides']:
  for start in pa.CONFIG['starts']:
   if time.perf_counter()-started>CONFIG['strip_wall_budget_s']:raise TimeoutError('Frozen strip budget exhausted')
   group=f'{side}_{start:06d}';d=V26/'cohorts'/group;meta=read(d/'meta.json')
   with np.load(d/'data.npz',allow_pickle=False) as a:
    data={k:a[k] for k in ('gray_native','times','source_frames','pixel_hashes')}
   report,arrays=sm.process_cohort(meta,data,geometry[side])
   report['protocol_sha256']=digest
   sm.write_artifacts(folder/group,report,arrays);reports.append(report)
   print(group, report['status'], round(time.perf_counter()-started,2),flush=True)
 summary=sm.summarize_reports(reports)
 summary.update(protocol_sha256=digest,elapsed_s=time.perf_counter()-started)
 write(folder/'summary.json',summary)
 return {'groups':len(reports),'elapsed_s':summary['elapsed_s']}

def join_periods(digest):
 """Same-cohort/ID/frame differences vs all texture recurrences, no branch verdict."""
 outputs=[];cache={}
 with (OUT/'paired/paired_points.jsonl').open(encoding='utf-8') as f:
  for line in f:
   point=json.loads(line)
   if point['state']!='both':continue
   group=point['group_id']
   if group not in cache:
    cache[group]={(r['configuration'],tuple(r['frame_pair'])):r for r in read(OUT/'strips'/group/'report.json').get('pairs',[])}
   for config,field in [('raw_fixed','initial_tangent_px'),('geometry_compensated','compensated_tangent_px')]:
    strip=cache[group].get((config,tuple(point['frame_pair'])))
    if strip is None:continue
    delta=point['paired_difference'][field]
    outputs.append({'session':point['session'],'camera':point['camera'],'group_id':group,
     'point_id':point['point_id'],'frame_pair':point['frame_pair'],'dt_s':point['dt_s'],
     'configuration':config,'paired_difference_px':delta,
     'bands':[{ 'band':b['band'], 'period_diagnostic':b['period_diagnostic'],
       'integer_branch_proximity':sm.period_multiple_diagnostic(delta,b['period_diagnostic']['period_candidates_px']),
       'candidate_displacements_px':b['candidate_displacements_px'],
       'correlation_scores':b['correlation_scores'], 'phase_peaks':b['phase_correlation_peaks']}
       for b in strip['bands']],
     'material_identity_confirmed':False,'correct_branch':None,'metric_velocity_m_s':None})
 folder=OUT/'period_join';folder.mkdir(exist_ok=False)
 with (folder/'all_common_point_periods.jsonl').open('x',encoding='utf-8') as f:
  for r in outputs:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
 target=[r for r in outputs if r['group_id']=='left_000240' and r['frame_pair']==[241,242]]
 write(folder/'left240_frame241_242.json',{'protocol_sha256':digest,'rows':target,'interpretation':'arithmetic_proximity_not_material_truth'})
 return {'common_point_config_records':len(outputs),'left241_242_records':len(target)}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','verify','paired','blind','strip','sampling','join','synthetic'])
 args=parser.parse_args()
 if args.stage=='freeze':freeze();return
 start=time.perf_counter();p=verify();digest=sha(OUT/'protocol.json')
 receipt={'argv':sys.argv,'stage':args.stage,'protocol_sha256':digest,'started_utc':datetime.now(timezone.utc).isoformat()}
 try:
  if args.stage=='paired':
   result=pa.run(V26,OUT/'paired',digest,p['input_sha256']);detail={'cohorts':result['fixed_cohort_denominator']}
  elif args.stage=='blind':
   br.export_package(V26,OUT/'blind_review',SOURCES,digest)
   detail=br.validate_package(OUT/'blind_review/manifest.json');write(OUT/'blind_review/validation.json',detail)
  elif args.stage=='strip':detail=run_strip(p,digest)
  elif args.stage=='join':detail=join_periods(digest)
  elif args.stage=='synthetic':detail=sm.run_synthetic_suite(OUT/'synthetic')
  elif args.stage=='sampling':
   detail=ss.run(SOURCES);detail['protocol_sha256']=digest;write(OUT/'source_sampling_diagnosis.json',detail)
  else:detail={'all_frozen_hashes_unchanged':True}
  verify();receipt.update(exit_code=0,details=detail)
 except Exception as exc:
  receipt.update(exit_code=1,error=repr(exc));raise
 finally:
  receipt['elapsed_s']=time.perf_counter()-start
  write(OUT/(args.stage+'_execution_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'.json'),receipt)
 print(json.dumps({'stage':args.stage,'exit_code':0,'elapsed_s':receipt['elapsed_s']}),flush=True)

if __name__=='__main__':main()
