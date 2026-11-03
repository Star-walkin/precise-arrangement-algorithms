"""Offline file/coordinate contracts. Never imports camera or actuator modules."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
LAB=HERE.parents[1]
ROOT=LAB.parent
BASE=LAB/'outputs/v26_causal_material/v26_20261005_001'
SESSIONS={
 'left':ROOT/'现场视频/训练采集/双相机同步/dual_20260924_111500/left_DA9627065.avi',
 'right':ROOT/'现场视频/训练采集/双相机同步/dual_20260924_111143/right_DA9626975.avi'}
CONFIG=dict(seed=20261005, source_shape_hw=[3000,4096], nominal_fps=10.,
 feature_stride=2, cohort_start_indices=list(range(0,600,20)), cohort_frames=21,
 crop_size=1024, crop_forward_px=256., model_size=256, max_age_s=2.0,
 max_frame_gap_s=.25, max_points=12, min_points=4,
 seed_min_distance_px=12., seed_quality=.02, seed_half_width_fraction=.25,
 seed_last_widths=12., seed_end_margin_widths=1., seed_max_saturation=.25,
 seed_min_patch_contrast=8., seed_patch_half=4,
 lk_win_model=21, lk_win_native=41, lk_levels=4, lk_iterations=30,
 lk_epsilon=.01, fb_max_native_px=4., lk_max_gray_error=30.,
 reference_y_fraction=.70, line_intersection_min_abs_nx=.10,
 signal_max_saturation=.80, inference_status='candidate_never_confirmed',
 cohort_reidentification=False, motion_status='unknown', hardware_access=False)

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def save_npz(path,**arrays):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as f:np.savez_compressed(f,**arrays)

def timestamps(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 result=[]
 for i,row in enumerate(rows):
  r={k:(float(v) if k=='host_time_s' else int(v)) for k,v in row.items()}
  if r['video_frame']!=i or not np.isfinite(r['host_time_s']):raise ValueError('Invalid frame/time metadata')
  if i and (r['host_time_s']<=result[-1]['host_time_s'] or r['source_frame']<=result[-1]['source_frame']):
   raise ValueError('Non-increasing timestamp or repeated source frame')
  result.append(r)
 return result

def crop_box(shape,seed):
 p=np.asarray(seed,float)
 if p.ndim!=2 or p.shape[1]!=2 or len(p)<2 or not np.isfinite(p).all():raise ValueError('Invalid crop seed')
 direction=p[-1]-p[0];length=np.linalg.norm(direction)
 if length<1e-9:raise ValueError('Degenerate crop seed')
 direction/=length
 center=p[-1]+CONFIG['crop_forward_px']*direction;s=CONFIG['crop_size'];h,w=shape[:2]
 if min(h,w)<s:raise ValueError('Source too small for frozen crop')
 origin=np.clip(np.floor(center-s/2),[0,0],[w-s,h-s]).astype(int)
 return [int(origin[0]),int(origin[1]),s,s]

def map_points(points,box,size,to_native=False):
 p=np.asarray(points,float);origin=np.asarray(box[:2]);scale=size/box[2]
 return (p+.5)/scale-.5+origin if to_native else (p-origin+.5)*scale-.5

def cache_path(side,index):
 return LAB/f'outputs/v16_event_feasibility/extract/{side}/frames/f{index:06d}.json'

def stream_guard(previous,time_s,source_frame,pixel_hash):
 if not np.isfinite(time_s):return 'nonfinite_timestamp'
 if previous is None:return None
 if time_s<=previous['time_s'] or source_frame<=previous['source_frame']:return 'non_increasing_input'
 if pixel_hash==previous['pixel_hash']:return 'duplicate_decoded_pixels'
 if time_s-previous['time_s']>CONFIG['max_frame_gap_s']:return 'frame_gap_exceeded'
 return None
