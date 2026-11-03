"""V2: fresh gray centers on EVERY source frame; original B and H6 scoring."""
import importlib.util,sys,copy,hashlib,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
V1=HERE.parent/'active_rope_video_20261101_v1'
spec=importlib.util.spec_from_file_location('_video_v1_readonly_engine',V1/'frame_engine/engine.py')
base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
from fresh_surface import FreshSurface
cv2=base.cv2

class FrameEngine(base.FrameEngine):
    def __init__(self,job,anchor_time,threads=2):
        super().__init__(job,anchor_time,threads)
        reference=self.priors['f3_vector_records']['C3']['fit']['model'].get('shared_shape_for_grouping_and_regularization',{})
        self.observer=FreshSurface(self.priors['surface_half'],self.pitch,reference)
        self.anchor_tracks=[];self.tracks=[];self.anchor_half=None
        self.last_entry_attempt=-float('inf');self.fresh_diag={}
        self.receipt.update(version='IR-V2.0.0',scene_initialization='offline fixed ROI/direction calibration from existing allowed scene; no saved line coordinates used',
            per_frame_current_pixel_detection=True,saved_gray_center_coordinates_used=False,
            scene_roi_is_static_calibration_not_dynamic_SAM2=True)
    def _scene(self,gray,t):
        self.tracks,self.fresh_diag=self.observer.detect(gray)
        self.scene_ok=bool(self.tracks)
        return dict(self.fresh_diag,age_s=0.,new_gray_center_measurement=True,new_full_f3_fusion=False,
            scene_geometry_supported=self.scene_ok)
    def _capture_entries(self,bgr,gray,t):
        # Entry discovery has the same effective retry cadence as the old2Hz
        # version; gray-center observation and conditional inference do not skip.
        if t-self.last_entry_attempt<.5:
            return [],dict(status='entry_retry_cooldown',source='no stale target path used',valid_entries=0)
        self.last_entry_attempt=t
        return super()._capture_entries(bgr,gray,t)
    def _propagate_entries(self,gray,t):
        kept,diagnostics=super()._propagate_entries(gray,t)
        ids={s['entry_id'] for s in kept}
        # LK is underconstrained along a thin straight rope. If it fails, a
        # bounded NORMAL profile search must independently remeasure the short
        # entrance in current pixels; the old path is never output as a result.
        for seed in self.seeds:
            if seed['entry_id'] in ids:continue
            xy=np.asarray(seed['xy'],float);d=xy[-1]-xy[0];d/=max(np.linalg.norm(d),1e-8);n=np.array([-d[1],d[0]])
            candidates=[];g=gray.astype(np.float32)/255
            for offset in np.linspace(-4.,4.,9):
                q=xy+offset*n
                if np.any(q<0) or np.any(q[:,0]>=4096) or np.any(q[:,1]>=3000):continue
                support=self._raw_seed_check(gray,q,seed['width_px'])
                if support<.5:continue
                points=np.linspace(q[0],q[-1],24)*.5
                v=base.sample(g,points[:,None,:]+n*np.asarray([-1.,-.6,0.,.6,1.])[None,:,None]*seed['width_px']*.5)
                bright=np.minimum(v[:,2]-v[:,:2].mean(1),v[:,2]-v[:,3:].mean(1))
                dark=np.minimum(v[:,:2].mean(1)-v[:,2],v[:,3:].mean(1)-v[:,2])
                score=float(np.maximum(bright,dark).mean())-.0005*abs(offset)
                candidates.append((score,support,offset,q))
            if candidates:
                score,support,offset,q=max(candidates,key=lambda x:x[0]);s=copy.deepcopy(seed)
                s.update(xy=q.tolist(),source='current_raw_normal_profile_refined_prior_automatic_entry',
                    last_current_image_check_time_s=float(t),entry_age_s=float(t-seed['detected_time_s']),current_raw_support_fraction=support)
                kept.append(s);diagnostics.append(dict(entry_id=seed['entry_id'],accepted=True,
                    method='current_raw_profile_after_LK_aperture_failure',normal_offset_original_px=float(offset),current_raw_support_fraction=support))
        return kept,diagnostics
    def _measured_tracks(self,gray,heat):
        tracks=super()._measured_tracks(gray,heat)
        for tr in tracks:
            tr['video_source']='fresh_current_frame_boundaries_then_current_B_profiles'
            tr['historical_F3_pair_flags_are_not_fresh_boundary_detection']=False
            tr['boundary_flags_source']='current_frame_detected_boundaries'
        return tracks
    def step(self,bgr,t,refresh_entry=False):
        result=super().step(bgr,t,refresh_entry)
        for tr in result['surface_tracks']:
            tr['source']='fresh_current_frame_boundaries_and_adjacent_midpoints'
            tr['source_age_s']=0.
        coords=np.asarray([p for tr in result['surface_tracks'] for p in tr['points_xy']],np.float32).reshape(-1,2)
        result['diagnostics']['gray_center_sha256']=hashlib.sha256(coords.tobytes()).hexdigest()
        result['diagnostics']['fresh_gray_detection_calls']=self.observer.calls
        result['diagnostics']['fresh_surface']=self.fresh_diag
        result['diagnostics']['offline_static_scene_calibration']=True
        result['diagnostics']['future_target_paths_used']=False
        return result

anchor_job=base.anchor_job
