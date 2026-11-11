"""Direct current-frame implementation of RES-1.8.1, without stage caches."""
import os,sys,time
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import cv2,numpy as np
from common import *
for p in [R11,R12,PARENT/'RopeEdgeSuite_R15_20261109']:
    sys.path.append(str(p))
from tracker_sg2 import Tracker as Strict,DynamicReacquire
from tracker_sg3 import Tracker as Relaxed
from tracker_sg5 import Fusion
from boundary_bg11 import Boundary
from camera_cg2 import CameraContinuity
import region_bg5_refined as bg5

class Engine:
    def __init__(self):
        cv2.setNumThreads(1)
        self.strict=Strict(np.zeros((3000,4096),np.uint8),DynamicReacquire(PITCH))
        self.relaxed=Relaxed(np.zeros((3000,4096),np.uint8))
        self.fusion=Fusion();self.boundary=Boundary();self.camera=CameraContinuity()
    def step(self,image,t,i):
        tick=time.perf_counter();region=bg5.segment(image)
        self.strict.set_region(region);self.relaxed.set_region(region)
        a=self.strict.step(image,t,i);b=self.relaxed.step(image,t,i)
        track=self.fusion.step(image,a,b,t,i);boundary=self.boundary.step(image,region['record'],track,i)
        camera=self.camera.step(image)
        if not camera['confirmed'] or camera['camera_changed']:
            boundary['fit_valid']=False;boundary['count_boundary_valid']=False
            boundary['reasons'].append('camera_motion_invalid')
            if camera['camera_changed']:self.boundary=Boundary()
        return plain(dict(track=track,proposal=region['record'],boundary=boundary,camera=camera,
            algorithm_errors=[],elapsed_s=time.perf_counter()-tick,
            stages=dict(BG5='fresh_current_frame',SG2='fresh_current_frame',SG3='fresh_current_frame',
                SG5='fresh_current_anchor_and_fusion',BG11='fresh_causal_fit',CG2='fresh_current_background',
                F3='not_run',counter='not_run',physical_contact='not_certified')))
