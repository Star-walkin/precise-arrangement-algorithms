"""RES-1.5 numeric stages; per-video cold state and explicit mirroring."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,time,traceback
import cv2,numpy as np
from common import *
for p in [R11,R12,R14,BASE/'baseline/source/LC1_boundary_BG4_local_temporal_20261107_v1']:
    sys.path.insert(0,str(p))
from tracker_sg3 import Tracker
import region_bg5_refined as bg5
from boundary_bg8 import Boundary
from camera_cg2 import CameraContinuity

CORE_FILES=[ROOT/'tracker_sg3.py',R12/'tracker_sg2.py',R11/'region_bg5_refined.py',ROOT/'boundary_bg8.py',
    ROOT/'camera_cg2.py',ROOT/'engine.py',ROOT/'common.py']
def transform_image(image,camera):return cv2.flip(image,1) if camera=='right' else image
def original_abc(abc,width,camera):
    if abc is None:return None
    A,B,C=abc
    return [-A,B,A*(width-1)+C] if camera=='right' else [A,B,C]
def empty_track(reason):
    return dict(valid=False,current_line_abc=None,line_abc=None,line_points_xy=[],supported_segments_xy=[],
        model_forward_count=0,diagnostics=dict(accept_reason=reason,attempts=[],reacquire=None))
class Engine:
    def __init__(self):
        cv2.setNumThreads(1);self.tracker=Tracker(np.zeros((3000,4096),np.uint8))
        self.boundary=Boundary();self.camera=CameraContinuity()
    def step(self,image,t,i):
        errors=[];start=time.perf_counter()
        try:region=bg5.segment(image)
        except Exception:
            errors.append(dict(stage='BG5',traceback=traceback.format_exc()))
            region=dict(record=dict(valid=False,reasons=['BG5_exception'],top_curve_xy=[],top_observed_xy=[]),surface_mask=np.zeros(image.shape[:2],np.uint8))
        self.tracker.set_region(region)
        try:track=self.tracker.step(image,t,i)
        except Exception:
            errors.append(dict(stage='SG3',traceback=traceback.format_exc()));track=empty_track('SG3_exception')
            self.tracker=Tracker(np.zeros(image.shape[:2],np.uint8))
        try:boundary=self.boundary.step(image,region['record'],track,i)
        except Exception:
            errors.append(dict(stage='BG8',traceback=traceback.format_exc()))
            boundary=dict(fit_valid=False,count_boundary_valid=False,reasons=['BG8_exception'],top_curve_xy=[],edge_y=None)
            self.boundary=Boundary()
        try:camera=self.camera.step(image)
        except Exception:
            errors.append(dict(stage='camera',traceback=traceback.format_exc()));camera=dict(confirmed=False,camera_changed=False)
        good=bool(camera.get('confirmed') and not camera.get('camera_changed'))
        if not good:
            boundary['fit_valid']=False;boundary['count_boundary_valid']=False;boundary['reasons'].append('camera_motion_invalid')
            if camera.get('camera_changed'):self.boundary=Boundary()
        return plain(dict(track=track,proposal=region['record'],boundary=boundary,camera=camera,
            algorithm_errors=errors,elapsed_s=time.perf_counter()-start,
            stages=dict(BG5='fresh_current_frame',SG3='fresh_current_frame',BG8='fresh_causal_fit',
                F3='not_run_in_RES15_boundary_tracking_scope',counter='not_run',physical_contact='not_certified')))
