"""Allowed source-frame0..39 spatial-anchor probe; no new NN inference."""
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from live_s1_tracker_anchored import LiveS1Tracker,pixel_sha256,BASE_SHA256


def main():
    root=Path(__file__).resolve().parent;base=root.parent
    obs=base/'LC1_joint_d_20261105_v1/observations'
    video=Path(r'E:\study\program\济宁\双相机同步\dual_20260924_111500\left_DA9627065.avi')
    mask_path=base/'camera_condition_sam2_20261009_v1/all_001/body_branches_002/masks/left_camera/dual_20260924_111500__pair0015__left.png'
    mask=cv2.imdecode(np.fromfile(mask_path,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
    first=json.loads((obs/'frame_000000.json').read_text(encoding='utf-8'));callbacks=[]
    def reacquire(image,t,index):
        raw=json.loads((obs/f'frame_{index:06d}.json').read_text(encoding='utf-8'));callbacks.append(index)
        return dict(seed_xy=raw['P1_H6'].get('seed_xy'),seed_source='same_current_pixel_bound_actual_A7_cache',
                    source_pixel_sha256=raw['source_pixel_sha256'],model_forward_count=0)
    tracker=LiveS1Tracker(mask,initial_seed_xy=first['P1_H6']['seed_xy'],
        initial_seed_source='frame0_actual_P1_A7',initial_seed_pixel_sha256=first['source_pixel_sha256'],
        reacquire_callback=reacquire)
    cap=cv2.VideoCapture(str(video));records=[];started=time.perf_counter()
    for i in range(40):
        ok,image=cap.read()
        if not ok:raise RuntimeError('allowed video decode failed')
        raw=json.loads((obs/f'frame_{i:06d}.json').read_text(encoding='utf-8'))
        if pixel_sha256(image)!=raw['source_pixel_sha256']:raise RuntimeError('source pixel binding mismatch')
        result=tracker.step(image,i/10.,i);result['frame_index']=i
        result['reference_is_same_frame_A7_conditioned_S1_not_truth']=True
        ref=raw['S1'];difference=None
        if result['valid'] and ref['valid']:
            line=result['line_abc'];abc=ref['line_abc']
            difference=-(line[0]*2238.7+line[2])/line[1]+(abc[0]*2238.7+abc[2])/abc[1]
        result['same_frame_A7_S1_y_difference_px']=difference
        records.append(result)
        if i<3 or i%10==9:
            print(i,result['valid'],round(result['diagnostics']['total_seconds'],3),
                  None if difference is None else round(difference,3),
                  result['diagnostics']['spatial_anchor'],flush=True)
    cap.release()
    diffs=[abs(r['same_frame_A7_S1_y_difference_px'])for r in records if r['same_frame_A7_S1_y_difference_px'] is not None]
    summary=dict(frames=40,valid=sum(r['valid']for r in records),callback_frames=callbacks,new_NN_forwards=0,
        actual_S1_calls=sum(r['diagnostics']['S1_calls']for r in records),
        zero_S1_call_frames=[r['frame_index']for r in records if not r['diagnostics']['S1_calls']],
        median_tracker_seconds=float(np.median([r['diagnostics']['total_seconds']for r in records])),
        median_S1_seconds=float(np.median([r['diagnostics']['S1_seconds']for r in records])),
        median_abs_y_difference_from_current_A7_condition_px=float(np.median(diffs)),
        max_abs_y_difference_from_current_A7_condition_px=float(max(diffs)),
        maximum_seed_start_drift_from_its_current_spatial_anchor_px=max(abs(r['diagnostics']['spatial_anchor']['seed_start_x_drift_px']or 0.)for r in records),
        first_seed_x=records[0]['seed_xy_original'][0][0],last_seed_x=records[-1]['seed_xy_original'][0][0],
        wall_seconds=time.perf_counter()-started,base_sha256=BASE_SHA256,
        anchored_source_sha256=hashlib.sha256((root/'live_s1_tracker_anchored.py').read_bytes()).hexdigest())
    out=root/'tracker_probe';out.mkdir(exist_ok=True)
    (out/'anchored_left_000_039.json').write_text(json.dumps(dict(summary=summary,records=records),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
