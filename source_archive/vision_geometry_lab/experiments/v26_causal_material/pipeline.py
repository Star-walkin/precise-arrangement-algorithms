"""Pre-registered offline replay: freeze -> prepare -> A/B. No hardware imports."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from .common import HERE,LAB,ROOT,BASE,SESSIONS,CONFIG,sha,read,write,save_npz,timestamps,crop_box,cache_path
from .point_tracks import seed_points
from . import a_runner,b_runner

RUN_ID='v26_20261005_001'
GATES={
 'kind':'research_screen_only_not_industrial_acceptance',
 'A_min_independent_events':10,'A_min_verified_negative_minutes':10,
 'A_min_recall':.80,'A_max_false_candidates_per_verified_negative_minute':.1,
 'A_max_estimate_interval_s':.5,'A_max_first_report_delay_s':.5,
 'B_inherited_bend_coverage':.80,'B_min_independent_verified_point_groups':10,
 'B_max_verified_wrong_identity_fraction':.05,'B_max_seed_failure_fraction':.20,
 'B_compare_mislinks_at_matched_valid_fraction':True,
 'lack_independent_labels':'not_evaluable_not_pass',
 'no_identity_gain':'stop_candidate_classifier_no_training',
 'no_physical_t1_or_directed_angle':'t2_unavailable',
 'no_parameter_retry_after_results':True}

def array_hash(frame):
 desc=json.dumps(dict(shape=list(frame.shape),dtype=str(frame.dtype)),sort_keys=True,separators=(',',':')).encode('ascii')
 h=hashlib.sha256(desc);h.update(np.ascontiguousarray(frame).tobytes());return h.hexdigest()

def sources():
 paths=list(HERE.glob('*.py'))+[HERE/'audit_manifest.json',HERE/'audit_notes.md',HERE/'tapnext_manifest.json']
 paths += [ROOT/p for p in a_runner.SOURCE_FILES]
 paths += [LAB/p for p in ('v16_video_measurement.py','v11_path_contract.py','rope_detector.py','arc_detector.py','geometry.py','wound_rope_tracker.py')]
 return {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(set(paths))}

def freeze(base):
 base=Path(base)
 if base.exists():raise FileExistsError('Use a new output, never overwrite a frozen run')
 data={};videos={}
 for side,path in SESSIONS.items():
  csv_path=path.parent/f'{side}_frames.csv';ts=timestamps(csv_path)
  if len(ts)!=601:raise ValueError('Unexpected source length')
  videos[side]=dict(path=str(path),sha256=sha(path),timestamps_path=str(csv_path),timestamps_sha256=sha(csv_path),frames=len(ts))
  for i in range(0,601,CONFIG['feature_stride']):
   p=cache_path(side,i);data[str(p.relative_to(ROOT)).replace('\\','/')]=sha(p)
  p=LAB/f'outputs/v16_event_feasibility/measure/{side}/center_only.json'
  data[str(p.relative_to(ROOT)).replace('\\','/')]=sha(p)
 protocol=dict(schema='v26_causal_material_1',run_id=RUN_ID,frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
  config=CONFIG,gates=GATES,A=a_runner.CONFIG,B=b_runner.CONFIG,
  source_sha256=sources(),data_sha256=data,videos=videos,
  pretrained_tracker=dict(model='official TAPNext++ 256',status='blocked_weight_download',
   manifest_sha256=sha(HERE/'tapnext_manifest.json'),weight_sha256=None,real_inference_permitted=False),
  budget=dict(A_configs_per_method=1,LK_configs=['lk256','lk_native'],pretrained_configs=1,
   maximum_cohorts=60,frames_per_cohort=21,maximum_prepare_s=1800,maximum_A_s=300,maximum_B_s=600,
   training_steps=0,download_budget_s=900,download_budget_exhausted=True),
  coordinate_contract='unmirrored source BGR xy; fixed 1024 crop; half-pixel resize to 256; all tracks map back to source',
  timestamp_contract='host_time_s SDK delivery receive time, not exposure; original session offsets retained',
  cache_contract='V16 automatic current-frame outputs, rehashed source and redecoded pixels; no cache recomputation or label-derived initialization',
  signal_sampling='every second saved video frame, 301 rows/stream; LK every saved frame, 10fps nominal',
  signal_joint_validity='angle and x_ref jointly gated; scanline parallelism rejection does not prove angle unobservable',
  comparison_contract='CUSUM vs BOCPD same angle/x_ref inputs; legacy gap baseline uses different historical measurement and is context only',
  inference_label_files=[],reserved_20260925_read=False,hardware_access=False,deployment_permitted=False,
  git='not_a_repository',environment=dict(python=sys.version,executable=sys.executable,cv2=cv2.__version__,numpy=np.__version__,platform=platform.platform()))
 write(base/'protocol.json',protocol)
 return dict(stage='frozen',run_id=RUN_ID,protocol_sha256=sha(base/'protocol.json'))

def verify(base,full=False):
 p=read(Path(base)/'protocol.json')
 if p['config']!=CONFIG or p['gates']!=GATES:raise ValueError('Config changed after freeze')
 for name,expected in p['source_sha256'].items():
  if sha(ROOT/name)!=expected:raise ValueError('Source changed: '+name)
 if full:
  for name,expected in p['data_sha256'].items():
   if sha(ROOT/name)!=expected:raise ValueError('Cache changed: '+name)
  for v in p['videos'].values():
   if sha(v['path'])!=v['sha256'] or sha(v['timestamps_path'])!=v['timestamps_sha256']:raise ValueError('Source video/timestamps changed')
 return p

def optical_quality(frame,rope):
 """Only the currently observed free span; no contrast normalization over time."""
 seg=np.asarray(rope.get('observed_segment') or [],float);width=rope.get('width_px')
 if seg.shape!=(2,2) or not np.isfinite(seg).all() or width is None or not np.isfinite(width) or width<=0:return None,None
 d=seg[1]-seg[0];length=np.linalg.norm(d)
 if length<1e-9:return None,None
 n=np.array([-d[1],d[0]])/length
 xy=seg[0]+np.linspace(.05,.95,128)[:,None]*(seg[1]-seg[0])
 xy=xy[:,None,:]+np.linspace(-.25*width,.25*width,9)[None,:,None]*n
 if not ((xy>=0).all() and (xy[:,:,0]<=frame.shape[1]-1).all() and (xy[:,:,1]<=frame.shape[0]-1).all()):return None,None
 gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
 values=cv2.remap(gray,xy[:,:,0].astype(np.float32),xy[:,:,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
 return float(np.mean(values>=250)),float(np.percentile(values,95)-np.percentile(values,5))

def feature_row(side,index,ts,meta,old,frame,pixel_hash,previous_hash=None):
 start=time.perf_counter();r=meta['rope'];reason=r.get('reason','missing_free_fit')
 sat,contrast=optical_quality(frame,r);normal=r.get('normal');angle=None;xref=None
 valid=bool(r.get('valid'))
 if valid:
  try:
   nx,ny=normal
   if abs(nx)<CONFIG['line_intersection_min_abs_nx']:valid=False;reason='near_parallel_reference_scanline'
   else:
    xref=(r['rho_px']-ny*CONFIG['reference_y_fraction']*frame.shape[0])/nx
    angle=float(np.deg2rad(r['angle_deg'])%np.pi)
    valid=bool(np.isfinite([xref,angle]).all())
    if not valid:reason='nonfinite_line'
  except (TypeError,KeyError,ValueError):valid=False;reason='malformed_line'
 if sat is None:valid=False;reason='missing_current_quality'
 elif sat>CONFIG['signal_max_saturation']:valid=False;reason='free_span_saturated'
 if pixel_hash==previous_hash:valid=False;reason='duplicate_decoded_pixels'
 if valid:reason='ok'
 seg=r.get('observed_segment');length=None
 if seg is not None:
  arr=np.asarray(seg,float)
  if arr.shape==(2,2) and np.isfinite(arr).all():length=float(np.linalg.norm(arr[1]-arr[0]))
 finite=lambda value:float(value) if value is not None and np.isfinite(value) else None
 gap=old.get('features',{});diagnostics=r.get('diagnostics',{})
 return dict(stream_id=SESSIONS[side].parent.name+':'+side,source_frame=index,sdk_source_frame=ts['source_frame'],
  time_s=ts['host_time_s'],max_source_timestamp=ts['host_time_s'],available_at_s=ts['host_time_s'],
  nominal_time_s=index/CONFIG['nominal_fps'],available=valid,reason=reason,
  angle_proxy_rad=angle if valid else None,x_ref_px=float(xref) if valid else None,
  width_px=finite(r.get('width_px')),rope_width_px=finite(r.get('width_px')),free_span_length_px=length,fit_residual_px=finite(r.get('residual_px')),
  inlier_ratio=None,inlier_ratio_reason='fit_reports_support_count_not_total_trials',
  observed_support=r.get('observed_support'),saturation=sat,contrast=contrast,
  fit_support_coverage=finite(diagnostics.get('support_coverage')),
  motion_status='unknown',motion_source='unavailable',timestamp_basis='host_monotonic_sdk_receive_not_exposure',
  measurement_source='sha_verified_V16_current_frame_free_fit_plus_current_raw_pixel_quality',quality_reason=reason,
  old_gap_usable=bool(gap.get('usable')) and pixel_hash!=previous_hash,
  old_gap_proxy=gap.get('image_gap_proxy'),old_gap_reason=gap.get('reason'),
  old_gap_unit='image_optical_width_not_axial_rope_count',pixel_sha256=pixel_hash,
  cached_extraction_timing_ms=meta.get('timing_ms'),quality_and_feature_ms=1000*(time.perf_counter()-start),
  emit_clock='causal_simulation_not_live_completion',physical_t1_confirmed=False)

def prepare(base):
 base=Path(base);p=verify(base,full=True);protocol_sha=sha(base/'protocol.json');started=time.perf_counter()
 if (base/'features').exists() or (base/'cohorts').exists():raise FileExistsError('No partial overwrite/restart')
 cv2.setNumThreads(2);input_hashes={};feature_hashes={};audits={}
 for side,path in SESSIONS.items():
  ts=timestamps(path.parent/f'{side}_frames.csv')
  olds={r['frame_index']:r for r in read(LAB/f'outputs/v16_event_feasibility/measure/{side}/center_only.json')['records']}
  cap=cv2.VideoCapture(str(path));features=[];groups={};digests=[];decode_ms=[]
  if not cap.isOpened():raise ValueError('Cannot open source video')
  try:
   for i in range(len(ts)):
    if time.perf_counter()-started>p['budget']['maximum_prepare_s']:raise TimeoutError('Preparation budget exceeded')
    before=time.perf_counter();ok,frame=cap.read();decode_ms.append((time.perf_counter()-before)*1000)
    if not ok or list(frame.shape[:2])!=CONFIG['source_shape_hw']:raise ValueError('Unexpected decode/shape')
    ph=array_hash(frame);digests.append(ph);meta=None
    if i%CONFIG['feature_stride']==0:
     meta=read(cache_path(side,i))
     if ph!=meta['frame_sha256']:raise ValueError('Raw pixels differ from frozen automatic fit')
     features.append(feature_row(side,i,ts[i],meta,olds[i],frame,ph,digests[-2] if i else None))
    if i in CONFIG['cohort_start_indices']:
     seed=meta.get('free_seed');width=meta['rope'].get('width_px');reason=meta.get('seed_reason');points=[];sd={}
     if seed and width and width>0:
      box=crop_box(frame.shape,seed);points,sd=seed_points(frame,seed,width,box);reason=sd['reason']
     else:box=[(frame.shape[1]-1024)//2,(frame.shape[0]-1024)//2,1024,1024]
     group_id=f'{side}_{i:06d}'
     groups[i]=dict(meta=dict(group_id=group_id,side=side,start_index=i,available=len(points)>=CONFIG['min_points'],
      seeds_native=points,box=box,reason=reason,seed_diagnostics=sd,free_seed=seed,optical_width_px=width,
      automatic_fit_sha256=sha(cache_path(side,i)),protocol_sha256=protocol_sha,material_identity_confirmed=False,
      identity_source='current_frame_free_span_texture_candidates_not_material_ground_truth',
      disabled_fallback_crop=not bool(seed),rotation_or_mirror=False),frames256=[],gray_native=[],times=[],source_frames=[],sdk_frames=[],pixel_hashes=[])
    for start,g in list(groups.items()):
     if start<=i<start+CONFIG['cohort_frames']:
      x,y,s,_=g['meta']['box'];crop=frame[y:y+s,x:x+s]
      g['frames256'].append(cv2.resize(crop,(256,256),interpolation=cv2.INTER_AREA))
      g['gray_native'].append(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY));g['times'].append(ts[i]['host_time_s'])
      g['source_frames'].append(i);g['sdk_frames'].append(ts[i]['source_frame']);g['pixel_hashes'].append(ph)
     if i==start+CONFIG['cohort_frames']-1:
      folder=base/'cohorts'/g['meta']['group_id'];write(folder/'meta.json',g['meta'])
      save_npz(folder/'data.npz',**{k:np.asarray(v) for k,v in g.items() if k!='meta'})
      for name in ('meta.json','data.npz'):input_hashes[str((folder/name).relative_to(base)).replace('\\','/')]=sha(folder/name)
      del groups[start]
    if i%100==0:print(json.dumps(dict(stage='prepare',side=side,frame=i,elapsed_s=round(time.perf_counter()-started,1))),flush=True)
   if cap.read()[0] or groups:raise ValueError('Unfinished cohorts or extra source frames')
  finally:cap.release()
  path_out=base/'features'/f'{side}.json';write(path_out,dict(protocol_sha256=protocol_sha,rows=features));feature_hashes[side]=sha(path_out)
  write(base/'pixel_hashes'/f'{side}.json',dict(pixel_hashes=digests,hash_contract='V11 array JSON descriptor plus contiguous BGR bytes'))
  audits[side]=dict(decoded_frames=len(digests),unique_pixel_hashes=len(set(digests)),duplicate_adjacent=sum(a==b for a,b in zip(digests,digests[1:])),
   features=len(features),available_features=sum(r['available'] for r in features),
   decode_ms=a_runner.stats(decode_ms),quality_feature_ms=a_runner.stats([r['quality_and_feature_ms'] for r in features]))
 verify(base,full=True)
 receipt=dict(protocol_sha256=protocol_sha,input_hashes=input_hashes,feature_sha256=feature_hashes,audits=audits,
  preparation_wall_s=time.perf_counter()-started,label_files_read=[],hardware_access=False)
 write(base/'prepare_manifest.json',receipt);return receipt

def run_A(base):
 p=verify(base);r=read(Path(base)/'prepare_manifest.json')
 return a_runner.run(base,Path(base)/'A',sha(Path(base)/'protocol.json'),r['feature_sha256'],{k:p['source_sha256'][k] for k in a_runner.SOURCE_FILES})

def run_B(base):
 verify(base);r=read(Path(base)/'prepare_manifest.json')
 return b_runner.run(base,Path(base)/'B',sha(Path(base)/'protocol.json'),expected_input_sha256=r['input_hashes'])

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('freeze','prepare','A','B','verify'))
 parser.add_argument('--base',type=Path,default=BASE);a=parser.parse_args()
 result={'freeze':freeze,'prepare':prepare,'A':run_A,'B':run_B,'verify':verify}[a.stage](a.base)
 print(json.dumps(dict(stage=a.stage,output=str(a.base),completed=True,keys=list(result)),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
