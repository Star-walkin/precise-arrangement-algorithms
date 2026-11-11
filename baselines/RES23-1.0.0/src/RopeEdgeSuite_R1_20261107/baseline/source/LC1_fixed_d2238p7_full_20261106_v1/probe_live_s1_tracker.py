"""Small allowed-video causal tracker probe; cached CURRENT A7 seeds only."""
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from live_s1_tracker import LiveS1Tracker,pixel_sha256


def main():
    root=Path(__file__).resolve().parent
    base=root.parent
    observation_dir=base/'LC1_joint_d_20261105_v1/observations'
    video=Path(r'E:\study\program\济宁\双相机同步\dual_20260924_111500\left_DA9627065.avi')
    mask_path=base/'camera_condition_sam2_20261009_v1/all_001/body_branches_002/masks/left_camera/dual_20260924_111500__pair0015__left.png'
    mask=cv2.imdecode(np.fromfile(mask_path,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
    first=json.loads((observation_dir/'frame_000000.json').read_text(encoding='utf-8'))
    callbacks=[]
    def reacquire(image,t,frame_index):
        path=observation_dir/f'frame_{frame_index:06d}.json'
        if not path.exists():return None
        raw=json.loads(path.read_text(encoding='utf-8'))
        callbacks.append(frame_index)
        return dict(seed_xy=raw['P1_H6'].get('seed_xy'),
                    seed_source='same_current_frame_pixel_bound_cached_actual_P1_A7_seed_only',
                    source_pixel_sha256=raw['source_pixel_sha256'],model_forward_count=0,
                    diagnostics=dict(observation_file=str(path),target_line_or_path_reused=False))
    tracker=LiveS1Tracker(mask,initial_seed_xy=first['P1_H6']['seed_xy'],
        initial_seed_source='frame0_current_actual_P1_A7_seed',
        initial_seed_pixel_sha256=first['source_pixel_sha256'],reacquire_callback=reacquire)
    cap=cv2.VideoCapture(str(video));records=[];started=time.perf_counter()
    for index in range(40):
        ok,image=cap.read()
        if not ok:raise RuntimeError(f'cannot decode allowed frame {index}')
        result=tracker.step(image,index/10.,index)
        result['frame_index']=index;result['source_pixel_sha256']=pixel_sha256(image)
        records.append(result)
        if index<3 or index%10==9:
            print(index,result['valid'],round(result['diagnostics']['total_seconds'],3),
                  result['diagnostics'].get('seed_witness_matches_primary_S1_seed_measurement'),
                  result['diagnostics']['callback_called'],flush=True)
    cap.release()
    elapsed=time.perf_counter()-started
    summary=dict(frames=40,valid=sum(r['valid']for r in records),
        callback_frames=callbacks,actual_new_model_forwards=0,
        all_S1_current_frames_rerun=all(r['diagnostics']['S1_calls']>0 for r in records),
        S1_core_call_count=sum(r['diagnostics']['S1_calls']for r in records),
        frames_without_S1_core_call=[r['frame_index']for r in records if r['diagnostics']['S1_calls']==0],
        median_tracker_seconds=float(np.median([r['diagnostics']['total_seconds']for r in records])),
        median_S1_seconds=float(np.median([r['diagnostics']['S1_seconds']for r in records])),
        wall_seconds=elapsed,tracker_source_sha256=hashlib.sha256((root/'live_s1_tracker.py').read_bytes()).hexdigest())
    out=root/'tracker_probe';out.mkdir(exist_ok=True)
    (out/'left_000_039.json').write_text(json.dumps(dict(summary=summary,records=records),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
