"""Same automatic texture seeds; native/256 LK, no material identity claims."""
from __future__ import annotations
import time
import cv2
import numpy as np
from .common import CONFIG,map_points,stream_guard

def seed_points(frame,seed,width,box):
 gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);x,y,s,_=box
 crop=gray[y:y+s,x:x+s];p=np.asarray(seed,float)-[x,y]
 d=p[-1]-p[0];length=np.linalg.norm(d);d/=length;n=np.array([-d[1],d[0]])
 yy,xx=np.mgrid[:s,:s];u=(xx-p[-1,0])*d[0]+(yy-p[-1,1])*d[1]
 v=(xx-p[-1,0])*n[0]+(yy-p[-1,1])*n[1]
 mask=((u<=-CONFIG['seed_end_margin_widths']*width)&
       (u>=-min(length-CONFIG['seed_end_margin_widths']*width,CONFIG['seed_last_widths']*width))&
       (np.abs(v)<=CONFIG['seed_half_width_fraction']*width)&(crop>15)&(crop<245)).astype(np.uint8)*255
 found=cv2.goodFeaturesToTrack(crop,96,CONFIG['seed_quality'],CONFIG['seed_min_distance_px'],mask=mask,blockSize=5)
 accepted=[];diagnostics=[];half=CONFIG['seed_patch_half']
 for point in ([] if found is None else found.reshape(-1,2)):
  px,py=np.round(point).astype(int)
  if not(half<=px<s-half and half<=py<s-half):continue
  patch=crop[py-half:py+half+1,px-half:px+half+1]
  sat=float(np.mean(patch>=250));contrast=float(np.percentile(patch,95)-np.percentile(patch,5))
  valid=sat<=CONFIG['seed_max_saturation'] and contrast>=CONFIG['seed_min_patch_contrast']
  diagnostics.append(dict(xy=(point+[x,y]).tolist(),saturation=sat,contrast=contrast,accepted=bool(valid)))
  if valid:accepted.append((point+[x,y]).tolist())
  if len(accepted)==CONFIG['max_points']:break
 return accepted,dict(reason='ok' if len(accepted)>=CONFIG['min_points'] else 'insufficient_unsaturated_texture',
  raw_corner_count=0 if found is None else len(found),candidates=diagnostics,
  material_identity_confirmed=False,method='shi_tomasi_inside_current_free_span_not_centerline_points')

def track_lk(frames,times,source_frames,pixel_hashes,seeds_native,box,mode):
 """All outputs depend only on arrived adjacent frames. Failure is absorbing."""
 size=256 if mode=='lk256' else CONFIG['crop_size']
 if mode not in ('lk256','lk_native'):raise ValueError(mode)
 if not len(frames) or len({len(frames),len(times),len(source_frames),len(pixel_hashes)})!=1:
  raise ValueError('Empty or unequal input sequence lengths')
 if not np.isfinite(np.asarray(times,float)).all():raise ValueError('Nonfinite timestamp')
 if not np.isfinite(np.asarray(seeds_native,float)).all():raise ValueError('Nonfinite seed')
 seeds=map_points(seeds_native,box,size).astype(np.float32).reshape(-1,1,2)
 if not len(seeds) or not ((seeds>=0)&(seeds<=size-1)).all():raise ValueError('Empty or out-of-crop seed')
 active=np.ones(len(seeds),bool);points=seeds.copy();rows=[];prev=None;previous_meta=None
 for k,(frame,t,source,ph) in enumerate(zip(frames,times,source_frames,pixel_hashes)):
  started=time.perf_counter();reason=stream_guard(previous_meta,float(t),int(source),str(ph))
  previous_meta=dict(time_s=float(t),source_frame=int(source),pixel_hash=str(ph))
  age=float(t-times[0]);fb=np.full(len(seeds),np.nan);err=np.full(len(seeds),np.nan)
  if reason or age>CONFIG['max_age_s']+1e-9:
   active[:]=False;reason=reason or 'cohort_age_limit'
  if frame.shape[:2]!=(size,size):raise ValueError('Unmatched LK input shape')
  gray=frame if frame.ndim==2 else cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  if k and np.any(active):
   ids=np.flatnonzero(active);p=points[ids]
   kw=dict(winSize=(CONFIG['lk_win_model'] if size==256 else CONFIG['lk_win_native'],)*2,
     maxLevel=CONFIG['lk_levels'],criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,CONFIG['lk_iterations'],CONFIG['lk_epsilon']))
   q,st,e=cv2.calcOpticalFlowPyrLK(prev,gray,p,None,**kw)
   if q is None:active[ids]=False;reason='flow_unavailable'
   else:
    back,bst,_=cv2.calcOpticalFlowPyrLK(gray,prev,q,None,**kw)
    if back is None:active[ids]=False;reason='backward_flow_unavailable'
    else:
     f=np.linalg.norm(back.reshape(-1,2)-p.reshape(-1,2),axis=1)*CONFIG['crop_size']/size
     e=e.reshape(-1);candidate=q.reshape(-1,2)
     inside=(candidate>=0).all(1)&(candidate<=size-1).all(1)
     good=(st.reshape(-1).astype(bool)&bst.reshape(-1).astype(bool)&np.isfinite(candidate).all(1)&inside&
          (f<=CONFIG['fb_max_native_px'])&(e<=CONFIG['lk_max_gray_error']))
     fb[ids]=f;err[ids]=e;points[ids]=q;active[ids]=good
     if not np.any(active):reason='all_points_rejected'
  native=map_points(points.reshape(-1,2),box,size,to_native=True)
  # No connecting historical point positions into a current-frame rope curve.
  rows.append(dict(time_s=float(t),t_emit=float(t),max_source_timestamp=float(t),source_frame=int(source),
   age_s=age,points_xy_native=[[float(v) if np.isfinite(v) else None for v in p] for p in native],valid_candidate=active.tolist(),
   fb_error_native_px=[float(v) if np.isfinite(v) else None for v in fb],
   lk_gray_error=[float(v) if np.isfinite(v) else None for v in err],
   rejection_reason=reason,visibility='unknown_not_confirmed',status='candidate' if np.any(active) else 'invalid',
   material_identity_confirmed=False,current_new_rope_membership='unknown',
   processing_ms=(time.perf_counter()-started)*1000,waiting_future_frames=0))
  prev=gray
 return rows
