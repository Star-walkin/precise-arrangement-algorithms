"""Causal original-pixel S1 tracker; an E1 condition is never a measured line.

Current ABC always comes from a fresh frozen S1Adapter call. Reacquisition
callback(image,t,frame_index) may return a pixel-bound current P1/A7 seed.
No d, future observation, target path or neural model is loaded here.
"""
from __future__ import annotations
import hashlib
import math
import time
import types
from pathlib import Path
import cv2
import numpy as np


SOURCE='causal current-pixel validation of previous measured S1 free support; fresh frozen S1 each frame'


def pixel_sha256(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def load_frozen_s1_adapter():
    path=Path(__file__).resolve().parent.parent/'LC1_locked_column_20261105_v1/lc1_s1_adapter.py'
    module=types.ModuleType('_read_only_original_S1_adapter_for_tracker')
    module.__file__=str(path)
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    return module.S1Adapter()


def _fit_direction(points,reference):
    a=np.asarray(points,np.float32).reshape(-1,2)
    vx,vy,x,y=map(float,cv2.fitLine(a,cv2.DIST_HUBER,0,.01,.01).reshape(-1))
    direction=np.array([vx,vy])
    if direction@reference<0:direction=-direction
    return np.array([x,y]),direction/np.linalg.norm(direction)


class LiveS1Tracker:
    def __init__(self,surface_mask_original,*,s1_adapter=None,initial_seed_xy=None,
                 initial_seed_source=None,initial_seed_frame_index=0,
                 initial_seed_pixel_sha256=None,reacquire_callback=None,
                 minimum_S1_span_px=200.,minimum_S1_points=24,
                 seed_length_px=44.,max_tracking_dt_s=.5,image_validator=None):
        self.mask=np.asarray(surface_mask_original)>0
        if self.mask.ndim!=2:raise ValueError('mask must be original-coordinate 2D')
        self.outside_distance=cv2.distanceTransform((~self.mask).astype(np.uint8),cv2.DIST_L2,3)
        self.s1=s1_adapter if s1_adapter is not None else load_frozen_s1_adapter()
        self.initial_seed=None if initial_seed_xy is None else np.asarray(initial_seed_xy,float).reshape(2,2)
        if self.initial_seed is not None and not initial_seed_source:raise ValueError('explicit initial seed source required')
        self.initial_source=initial_seed_source
        self.initial_frame=int(initial_seed_frame_index)
        self.initial_hash=initial_seed_pixel_sha256
        self.reacquire_callback=reacquire_callback
        self.minimum_span=float(minimum_S1_span_px);self.minimum_points=int(minimum_S1_points)
        self.seed_length=float(seed_length_px)
        if self.seed_length<40.:raise ValueError('frozen S1 requires E1 length >=40px')
        self.max_dt=float(max_tracking_dt_s);self.prev=None
        self.validator=image_validator
        self.total_S1_calls=0;self.total_callback_calls=0;self.total_reported_model_forwards=0

    def reset(self):
        self.prev=None

    def _seed_witness(self,image,seed):
        """Re-run exact frozen S1 seed predicate locally, with a real halo.

        Returned centers are actual bilateral measurements inside the seed,
        not the two input E1 endpoints. This prevents 40px/frame seed walking.
        They are used for next-frame seeding only, never to fit output ABC.
        """
        if self.validator is not None:return self.validator(image,seed)
        core=self.s1.core;c=core._config(self.s1.config)
        seed=np.asarray(seed,np.float32).reshape(2,2)
        vector=seed[1]-seed[0];length=float(np.linalg.norm(vector))
        if length<40:return dict(valid=False,reason='E1_too_short')
        direction=vector/length;normal=np.array([-direction[1],direction[0]],np.float32)
        margin=int(math.ceil(c['profile_radius_px']+max(abs(v)for v in c['profile_tangent_offsets_px'])+24))
        h,w=image.shape[:2]
        x0=max(0,int(np.floor(seed[:,0].min()))-margin);x1=min(w,int(np.ceil(seed[:,0].max()))+margin+1)
        y0=max(0,int(np.floor(seed[:,1].min()))-margin);y1=min(h,int(np.ceil(seed[:,1].max()))+margin+1)
        if x1-x0<20 or y1-y0<20:return dict(valid=False,reason='seed_outside_image')
        crop=np.ascontiguousarray(image[y0:y1,x0:x1])
        blurred=cv2.GaussianBlur(crop,(0,0),sigmaX=c['blur_sigma_px'],sigmaY=c['blur_sigma_px'],borderType=cv2.BORDER_REFLECT_101)
        gray=cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)
        gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3,scale=.125)
        gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3,scale=.125)
        gradient=gx*float(normal[0])+gy*float(normal[1])
        positions=np.linspace(.12,.88,int(c['seed_profile_count']),dtype=np.float32)*length
        centers=seed[0]+positions[:,None]*direction
        local=centers-np.array([x0,y0],np.float32)
        offsets,intensity,derivative,valid=core._profile_samples(gray,gradient,local,direction,normal,c)
        measured=core._measure_seed(offsets,intensity,derivative,valid,c)
        if measured is None:return dict(valid=False,reason='current_frozen_S1_seed_bilateral_not_confirmed')
        keep=measured['seed_support'];points=centers[keep]+measured['seed_measured_centers'][keep,None]*normal
        return dict(valid=True,points_xy=points.astype(float).tolist(),
            support_count=int(measured['seed_support_count']),width_px=float(measured['median_measured_width']),
            polarity=int(measured['seed_polarity']),score=float(measured['score']),
            edge_gradient=float(measured['seed_edge_gradient']),contrast=float(measured['seed_body_contrast']),
            source='exact_frozen_S1_seed_profile_bilateral_measurements_not_E1_endpoints')

    def _pool(self,state):
        points=np.asarray(state['tracking_support_xy'],float).reshape(-1,2)
        h,w=self.mask.shape;ix=np.rint(points[:,0]).astype(int);iy=np.rint(points[:,1]).astype(int)
        valid=(ix>=80)&(ix<w-80)&(iy>=80)&(iy<h-80)
        points=points[valid];ix=ix[valid];iy=iy[valid]
        if len(points)<12:return None,dict(reason='too_few_previous_measured_free_points')
        axis=state['direction'];projection=points@axis
        outside=self.outside_distance[iy,ix]>=10.
        preferred=points[outside]
        outside_used=bool(len(preferred)>=12 and np.ptp(preferred@axis)>=160.)
        if outside_used:points=preferred;projection=points@axis
        order=np.argsort(projection);points=points[order];projection=projection[order]
        points=points[projection<=projection.min()+900.]
        if len(points)<12 or np.ptp(points@axis)<160.:
            return None,dict(reason='previous_measured_free_region_too_short',outside_mask_preference_used=outside_used)
        points=points[np.unique(np.linspace(0,len(points)-1,min(32,len(points))).astype(int))]
        return points,dict(outside_mask_preference_used=outside_used,previous_support_points=len(points),
                           previous_support_span_px=float(np.ptp(points@axis)))

    def _track_seed(self,image,gray,t,frame_index):
        state=self.prev
        if state is None:return None,None,dict(reason='no_previous_valid_current_S1')
        if frame_index!=state['frame_index']+1 or not 0<t-state['t']<=self.max_dt:
            return None,None,dict(reason='nonconsecutive_or_stale_previous_frame')
        points,diagnostics=self._pool(state)
        if points is None:return None,None,diagnostics
        h,w=gray.shape;pad=100
        x0=max(0,int(points[:,0].min())-pad);x1=min(w,int(points[:,0].max())+pad+1)
        y0=max(0,int(points[:,1].min())-pad);y1=min(h,int(points[:,1].max())+pad+1)
        p=(points-np.array([x0,y0])).astype(np.float32).reshape(-1,1,2)
        before=np.ascontiguousarray(state['gray'][y0:y1,x0:x1]);now=np.ascontiguousarray(gray[y0:y1,x0:x1])
        flow_points=None
        try:
            q,status,_=cv2.calcOpticalFlowPyrLK(before,now,p,None,winSize=(21,21),maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,25,.01))
            back,status_back,_=cv2.calcOpticalFlowPyrLK(now,before,q,None,winSize=(21,21),maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,25,.01))
            fb=np.linalg.norm(back[:,0]-p[:,0],axis=1)
            movement=np.linalg.norm(q[:,0]-p[:,0],axis=1)
            keep=(status[:,0]>0)&(status_back[:,0]>0)&np.isfinite(fb)&(fb<=2.)&(movement<=80.)
            if keep.sum()>=6:
                proposed=q[keep,0]+np.array([x0,y0])
                if np.ptp(proposed@state['direction'])>=120.:flow_points=proposed
            diagnostics.update(LK_current_matches=int(keep.sum()),LK_FB_median_px=float(np.median(fb[keep])) if keep.any() else None)
        except (cv2.error,TypeError,ValueError):diagnostics['LK_failed']=True
        base_points=points if flow_points is None else flow_points
        origin,direction=_fit_direction(base_points,state['direction'])
        far=float(np.min(base_points@direction));normal=np.array([-direction[1],direction[0]])
        # E1 coordinates are derived conditions inside previous measured support,
        # then verified on THIS image; none is used as an output line sample.
        origin=origin+(far-origin@direction)*direction
        offsets=[0.,-8.,8.,-16.,16.] if flow_points is None else [0.,-8.,8.]
        best=None
        for shift in offsets:
            start=origin+shift*normal
            seed=np.stack((start,start+self.seed_length*direction))
            witness=self._seed_witness(image,seed)
            if witness.get('valid') and (best is None or witness.get('score',0)>best[1].get('score',0)):
                best=(seed,witness,shift)
            if shift==0 and witness.get('valid'):break
        if best is None:return None,None,dict(diagnostics,reason='current_image_seed_bilateral_validation_failed')
        diagnostics.update(current_normal_search_shift_px=float(best[2]),
            LK_used=flow_points is not None,seed_is_measurement=False,
            seed_provenance='previous_frame_actual_S1_support_plus_current_frozen_seed_validation')
        return best[0],best[1],diagnostics

    def _accept(self,result,seed):
        if result.get('valid') is not True:return False,'current_S1_UNKNOWN'
        pts=np.asarray(result.get('line_points_xy',[]),float).reshape(-1,2)
        if len(pts)<self.minimum_points:return False,'current_S1_support_points_insufficient'
        direction=(seed[1]-seed[0]);direction=direction/np.linalg.norm(direction)
        if np.ptp(pts@direction)<self.minimum_span:return False,'current_S1_support_span_insufficient'
        return True,'current_frozen_S1_measured_support_accepted'

    def step(self,bgr_original,t,frame_index):
        started=time.perf_counter();image=np.asarray(bgr_original)
        if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3 or image.shape[:2]!=self.mask.shape:
            raise ValueError('expected original uint8 BGR pixels with original mask shape')
        t=float(t);frame_index=int(frame_index)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        diagnostics=dict(coordinate_system='original_unmirrored_image_xy',seed_is_input_not_detection=True,
            seed_endpoints_used_to_fit_output_line=False,history_ABC_held_on_failure=False,
            image_validator_is_test_double=self.validator is not None)
        witness=None;seed=None;seed_source=None;calls=0;model_forwards=0;callback_called=False
        if self.initial_seed is not None and frame_index==self.initial_frame:
            if self.initial_hash is None or self.initial_hash==pixel_sha256(image):
                seed=self.initial_seed.copy();seed_source=self.initial_source
                witness=self._seed_witness(image,seed)
            else:diagnostics['initial_seed_pixel_hash_mismatch']=True
            self.initial_seed=None
        if seed is None:
            seed,witness,tracking=self._track_seed(image,gray,t,frame_index)
            diagnostics['tracking']=tracking
            if seed is not None:seed_source='previous_measured_S1_free_support_tracked_and_current_bilateral_verified'
        result=dict(valid=False,line_abc=None,line_points_xy=[],supported_segments_xy=[],s1_status='UNKNOWN',
                    diagnostics=dict(s1_core_executed=False,stop_reason='no_current_verified_E1'))
        accepted=False;accept_reason='no_current_verified_E1'
        s1_seconds=0.
        for attempt in range(2):
            if seed is not None and witness is not None and witness.get('valid'):
                before=time.perf_counter();result=self.s1.step(image,seed.tolist(),seed_source=seed_source)
                s1_seconds+=time.perf_counter()-before;calls+=1
                accepted,accept_reason=self._accept(result,seed)
                if accepted:break
            if attempt==1 or self.reacquire_callback is None:break
            callback_called=True
            callback=self.reacquire_callback(image,t,frame_index)
            if callback is None:break
            model_forwards+=int(callback.get('model_forward_count',0))
            candidate=callback.get('seed_xy')
            diagnostics['reacquire']=callback.get('diagnostics',{})
            if candidate is None:break
            supplied_hash=callback.get('source_pixel_sha256',callback.get('pixel_sha256'))
            if supplied_hash!=pixel_sha256(image):
                diagnostics['reacquire_seed_pixel_hash_mismatch_or_missing']=True
                break
            seed=np.asarray(candidate,float).reshape(2,2);seed_source=str(callback.get('seed_source','current_P1_A7_reacquisition'))
            witness=self._seed_witness(image,seed)
        if calls==0:
            # Preserve the faithful adapter's original UNKNOWN/provenance even
            # when no current E1 exists; this does NOT execute its image core.
            before=time.perf_counter()
            result=self.s1.step(image,None,seed_source='no_current_verified_E1_after_validation')
            s1_seconds+=time.perf_counter()-before
        self.total_S1_calls+=calls;self.total_callback_calls+=int(callback_called)
        self.total_reported_model_forwards+=model_forwards
        diagnostics.update(accept_reason=accept_reason,S1_calls=calls,S1_seconds=s1_seconds,
            callback_called=callback_called,reported_current_model_forwards=model_forwards,
            seed_validation=witness)
        abc=None;confidence=0.
        if accepted:
            pts=np.asarray(result['line_points_xy'],float).reshape(-1,2)
            extra=[];witness_agrees=False
            expected=result.get('diagnostics',{}).get('s1_prediction',{}).get('seed_measurement',{})
            if witness and witness.get('valid') and expected:
                witness_agrees=bool(witness.get('support_count')==expected.get('supported_profiles')
                    and abs(witness.get('width_px',0)-expected.get('width_px',-100))<.01
                    and witness.get('polarity')==expected.get('selected_polarity'))
                if witness_agrees:extra=witness['points_xy']
            diagnostics['seed_witness_matches_primary_S1_seed_measurement']=witness_agrees
            support=np.vstack((pts,np.asarray(extra,float).reshape(-1,2))) if extra else pts
            direction=(seed[1]-seed[0]);direction=direction/np.linalg.norm(direction)
            self.prev=dict(gray=gray,tracking_support_xy=support,direction=direction,t=t,frame_index=frame_index)
            abc=result['line_abc'];confidence=float(min(1.,len(pts)/100.)*min(1.,np.ptp(pts@direction)/500.))
        else:self.prev=None
        diagnostics['total_seconds']=time.perf_counter()-started
        return dict(valid=bool(accepted),current_line_abc=abc,line_abc=abc,
            line_points_xy=result.get('line_points_xy',[]) if accepted else [],
            confidence=confidence,confidence_is_uncalibrated_heuristic=True,
            source=SOURCE,seed_xy_original=None if seed is None else seed.tolist(),seed_source=seed_source,
            S1=result,diagnostics=diagnostics,model_forward_count=model_forwards,
            same_rope_identity_confirmed=False,whole_path_candidate_supported=False)
