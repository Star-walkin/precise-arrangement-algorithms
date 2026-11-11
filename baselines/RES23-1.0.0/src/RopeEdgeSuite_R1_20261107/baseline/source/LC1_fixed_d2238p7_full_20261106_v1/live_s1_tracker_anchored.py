"""Fixed spatial far-anchor variant; original frozen tracker stays unchanged.

LK/material motion estimates a current line only. Its tangential displacement
must never advance the E1 starting x. A verified current reacquisition is the
only event that can establish a different spatial anchor.
"""
from __future__ import annotations
import hashlib
import types
from pathlib import Path
import numpy as np


BASE_SHA256='4a404bd403378b9f0bfcfc2524bf34244386639435ca490fb5bd1c6b95d5f0f5'
_path=Path(__file__).with_name('live_s1_tracker.py')
_bytes=_path.read_bytes()
if hashlib.sha256(_bytes).hexdigest()!=BASE_SHA256:
    raise ValueError('anchored tracker requires the unchanged frozen base')
_base=types.ModuleType('_read_only_frozen_S1_tracker_base')
_base.__file__=str(_path)
exec(compile(_bytes,str(_path),'exec'),_base.__dict__)
pixel_sha256=_base.pixel_sha256


class LiveS1Tracker(_base.LiveS1Tracker):
    def __init__(self,surface_mask_original,*,minimum_S1_span_px=400.,seed_length_px=120.,
                 reacquire_callback=None,**kwargs):
        self.anchor_x_original=None
        self.anchor_source=None
        self.anchor_reason=None
        self.anchor_frame=None
        self.anchor_reset_count=0
        self._external_reacquire=reacquire_callback
        super().__init__(surface_mask_original,minimum_S1_span_px=minimum_S1_span_px,
                         seed_length_px=seed_length_px,
                         reacquire_callback=self._current_reacquire if reacquire_callback else None,
                         **kwargs)

    def _set_anchor(self,seed,source,frame_index,reason):
        p=np.asarray(seed,float).reshape(2,2)
        self.anchor_x_original=float(p[0,0])
        self.anchor_source=source
        self.anchor_frame=int(frame_index)
        self.anchor_reason=reason
        self.anchor_reset_count+=1

    def _current_reacquire(self,image,t,frame_index):
        result=self._external_reacquire(image,t,frame_index)
        if result is None:return None
        supplied=result.get('source_pixel_sha256',result.get('pixel_sha256'))
        seed=result.get('seed_xy')
        if seed is not None and supplied==pixel_sha256(image):
            p=np.asarray(seed,float).reshape(2,2)
            if np.isfinite(p).all() and np.linalg.norm(p[1]-p[0])>=40.:
                self._set_anchor(p,result.get('seed_source'),frame_index,
                                 'explicit_pixel_bound_current_reacquisition')
        return result

    def _track_seed(self,image,gray,t,frame_index):
        proposal,_,diagnostics=super()._track_seed(image,gray,t,frame_index)
        if proposal is None:return None,None,diagnostics
        if self.anchor_x_original is None:
            return None,None,dict(diagnostics,reason='no_verified_spatial_far_anchor')
        direction=proposal[1]-proposal[0]
        direction=direction/np.linalg.norm(direction)
        if abs(direction[0])<.2:
            return None,None,dict(diagnostics,reason='near_vertical_fixed_x_anchor_degenerate')
        # Current LK/profile line evaluated at a FIXED spatial x. Neither its
        # material endpoint nor an accumulated optical-flow displacement sets x.
        y=float(proposal[0,1]+(self.anchor_x_original-proposal[0,0])*direction[1]/direction[0])
        best=None
        for normal_shift in (0.,-8.,8.,-16.,16.):
            # A normal correction is represented as a y shift at fixed x.
            # Adding shift*normal directly would also move x and violate anchor.
            start=np.array([self.anchor_x_original,y+normal_shift/direction[0]])
            seed=np.stack((start,start+self.seed_length*direction))
            witness=self._seed_witness(image,seed)
            if witness.get('valid') and (best is None or witness.get('score',0)>best[1].get('score',0)):
                best=(seed,witness,normal_shift)
            if normal_shift==0. and witness.get('valid'):break
        if best is None:
            return None,None,dict(diagnostics,reason='fixed_spatial_anchor_current_bilateral_validation_failed')
        diagnostics.update(seed_provenance='fixed_spatial_far_anchor_current_line_and_current_bilateral_validation',
            anchor_x_original=self.anchor_x_original,
            material_minimum_x_used_as_final_E1=False,
            fixed_anchor_normal_shift_px=float(best[2]))
        return best[0],best[1],diagnostics

    def step(self,bgr_original,t,frame_index):
        if self.initial_seed is not None and int(frame_index)==self.initial_frame:
            if self.initial_hash is None or self.initial_hash==pixel_sha256(bgr_original):
                self._set_anchor(self.initial_seed,self.initial_source,frame_index,
                                 'verified_current_initial_far_E1')
        result=super().step(bgr_original,t,frame_index)
        seed=result.get('seed_xy_original')
        delta=None if seed is None or self.anchor_x_original is None else float(seed[0][0]-self.anchor_x_original)
        result['source']='fixed spatial far-anchor E1, current original-pixel frozen S1; no material marching'
        result['diagnostics']['spatial_anchor']=dict(x_original=self.anchor_x_original,
            source=self.anchor_source,reason=self.anchor_reason,established_frame=self.anchor_frame,
            reset_count=self.anchor_reset_count,seed_start_x_drift_px=delta,
            seed_length_px=self.seed_length,minimum_S1_support_span_px=self.minimum_span,
            resets_only_on_explicit_current_reacquisition=True)
        return result


AnchoredLiveS1Tracker=LiveS1Tracker
