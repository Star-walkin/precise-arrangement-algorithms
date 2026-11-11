"""Check whether supplied S2 contributes current pixels to rejected bend gaps.
This diagnostic never promotes S2's conditioned axis to a calibrated d.
"""
import sys
import cv2,numpy as np
from common import *
sys.path.insert(0,str(BASE/'baseline/source/LC1_locked_column_20261105_v1'))
from lc1_s2_adapter import S2Adapter

def main():
    out=ROOT/'d_calibration/s2_gap_probe';out.mkdir(exist_ok=True)
    cap=cv2.VideoCapture(read(RUN12/'binding.json')['source']);adapter=S2Adapter(retain_step_records=True);rows=[]
    for i in [20,120]:
        r=read(ROOT/'d_calibration'/f'frame_{i:06d}.json');track=r['S1'];abc=track['current_line_abc'];seedx=np.array([700.,820.])
        seed=np.c_[seedx,-(abc[0]*seedx+abc[2])/abc[1]]
        cap.set(1,i);ok,im=cap.read();assert ok and pixelsha(im)==r['source_pixel_sha256']
        s2=adapter.step(im,seed,seed_source='same_current_SG2_axis_for_gap_audit_only',width_px=track['diagnostics']['seed_validation']['width_px'])
        p1=r['P1_H6'];xy=np.asarray(p1['path_xy']);supported=np.array(p1['diagnostics']['path_point_support']);st=np.array(p1['diagnostics']['path_stage'])
        p=np.asarray(s2['measured_center_points_xy'],float).reshape(-1,2)
        free=xy[st=='free'];target=free[(free[:,0]>np.asarray(track['line_points_xy'])[:,0].max())]
        coverage=[]
        for q in target:coverage.append(bool(len(p) and np.min(np.linalg.norm(p-q,axis=1))<8.))
        item=dict(frame=i,source_pixel_sha256=pixelsha(im),S2=s2,
            free_gap_target_samples=len(target),S2_nearby_supported_samples=sum(coverage),
            S2_nearby_fraction=float(np.mean(coverage)) if coverage else 0.,
            same_rope_identity_confirmed=False,calibration_promoted=False)
        dump(out/f'frame_{i:06d}.json',item);rows.append({k:v for k,v in item.items() if k!='S2'})
        print('S2 gap',rows[-1],flush=True)
    cap.release();dump(out/'receipt.json',dict(status='complete',rows=rows,calibration_promoted=False,
        limitation='S2 cross sections alone cannot prove same-rope identity or observe a curved path'))
if __name__=='__main__':main()
