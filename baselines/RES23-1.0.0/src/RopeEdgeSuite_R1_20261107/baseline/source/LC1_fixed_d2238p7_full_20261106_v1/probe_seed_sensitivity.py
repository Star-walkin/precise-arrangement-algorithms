"""Current-frame frozen S1 seed sensitivity; no neural inference or tracking truth."""
import json
import math
from pathlib import Path
import cv2
import numpy as np
from live_s1_tracker import load_frozen_s1_adapter,pixel_sha256


def main():
    root=Path(__file__).resolve().parent
    obs=root.parent/'LC1_joint_d_20261105_v1/observations'
    cap=cv2.VideoCapture(r'E:\study\program\济宁\双相机同步\dual_20260924_111500\left_DA9627065.avi')
    adapter=load_frozen_s1_adapter();rows=[]
    variants=[('original_current_A7',None,0.,0.),('length44',44.,0.,0.),
        ('length44_normal_minus4',44.,-4.,0.),('length44_normal_plus4',44.,4.,0.),
        ('length44_angle_minus0p5deg',44.,0.,-.5),('length44_angle_plus0p5deg',44.,0.,.5),
        ('length44_normal_plus12',44.,12.,0.)]
    for frame in [0,15,30]:
        raw=json.loads((obs/f'frame_{frame:06d}.json').read_text(encoding='utf-8'))
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read()
        if not ok or pixel_sha256(image)!=raw['source_pixel_sha256']:
            raise RuntimeError('current frame pixel binding failed')
        seed=np.asarray(raw['P1_H6']['seed_xy'],float);direction=seed[1]-seed[0]
        length0=float(np.linalg.norm(direction));direction/=length0
        normal=np.array([-direction[1],direction[0]])
        for name,length,shift,angle in variants:
            theta=math.radians(angle)
            changed_direction=math.cos(theta)*direction+math.sin(theta)*normal
            start=seed[0]+shift*normal
            changed=np.stack((start,start+(length0 if length is None else length)*changed_direction))
            result=adapter.step(image,changed.tolist(),seed_source='current_source_bound_A7_seed_sensitivity_experiment')
            abc=result.get('line_abc')
            y=None if abc is None else float(-(abc[0]*2238.7+abc[2])/abc[1])
            rows.append(dict(frame=frame,variant=name,seed_xy=changed.tolist(),valid=result['valid'],
                             active_y_at_d2238p7=y,S1_status=result['s1_status'],
                             stop_reason=result['diagnostics']['stop_reason'],
                             supported_points=len(result['line_points_xy'])))
    cap.release()
    for row in rows:
        baseline=next(x for x in rows if x['frame']==row['frame'] and x['variant']=='original_current_A7')
        row['delta_y_from_same_frame_A7_conditioned_S1_px']=None if row['active_y_at_d2238p7'] is None or baseline['active_y_at_d2238p7'] is None else row['active_y_at_d2238p7']-baseline['active_y_at_d2238p7']
    out=root/'tracker_probe';out.mkdir(exist_ok=True)
    (out/'seed_sensitivity.json').write_text(json.dumps(dict(records=rows,
        reference_is_same_S1_with_current_A7_condition_not_physical_truth=True,
        frozen_configuration_changed=False,new_neural_inference=False),indent=2),encoding='utf-8')
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
