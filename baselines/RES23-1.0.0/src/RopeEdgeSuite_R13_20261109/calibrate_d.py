"""Actual new P1/B750/H6 calls on declared early clear far-edge frames."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,time
import cv2,numpy as np
from common import *
sys.path.insert(0,str(BASE/'baseline/source/LC1_locked_column_20261105_v1'))
from lc1_p1_adapter import P1H6Adapter
sys.path.insert(0,str(ROOT.parent/'LC1_joint_d_20261105_v1'))
from collect_joint_window import observed_geometry
sys.path.insert(0,str(ROOT.parent/'LC1_screened_d_20261106_v1'))
from frame_quality_gate import FrameQualityGate,enrich_current_connector

TRAIN=[20,60,100];HOLDOUT=[40,80,120]
def main():
    cv2.setNumThreads(2);out=ROOT/'d_calibration';out.mkdir(exist_ok=True)
    bind=read(RUN12/'binding.json');weight=BASE/'baseline/source/LC1_locked_column_20261105_v1/supplied_assets/step_0750.pt'
    dump(out/'plan.json',dict(train=TRAIN,holdout=HOLDOUT,selected_before_new_model_calls=True,
        minimum_far_gap_pitch=5,whole_path_required=True,current_curve_support_required=True,
        fit_promotion_requires_three_training_and_two_holdout_frames=True,source_sha256=bind['source_sha256']))
    cap=cv2.VideoCapture(bind['source']);adapter=None;summary=[];tick=time.perf_counter()
    gate=FrameQualityGate(PITCH)
    for i in sorted(TRAIN+HOLDOUT):
        cap.set(1,i);ok,im=cap.read();assert ok
        old=record(i);assert pixelsha(im)==old['source_pixel_sha256']
        mask=cv2.imdecode(np.fromfile(FRONT12/'regions'/f'{i:06d}.png',np.uint8),0)
        if adapter is None:adapter=P1H6Adapter(mask,PITCH,weight,threads=2,camera='left',surface_mode='entry_band')
        else:adapter.surface=mask
        path=out/f'frame_{i:06d}.json'
        if path.exists():r=read(path);assert r['source_pixel_sha256']==pixelsha(im)
        else:
            p=adapter.step(im)
            r=dict(frame_index=i,time_s=i/10,source_pixel_sha256=pixelsha(im),P1_H6=p,
                S1=old['S1_tracker_full'],observation_file=str(path),source_sha256=bind['source_sha256'],
                provenance=dict(mode='fresh_P1_H6_current_mask_and_pixel_bound_same_frame_SG2_S1'),model_sha256=sha(weight))
            dump(path,r)
        geom=enrich_current_connector(observed_geometry(r,PITCH));quality=gate.step(geom)
        gap=old['intersection_y_px']-old['region']['edge_y'] if old['intersection_y_px'] is not None and old['region']['edge_y'] is not None else None
        # A conservative screening margin is not a distance truth annotation.
        far=gap is not None and gap-2*PITCH>=5*PITCH
        eligible=bool(far and old['region']['valid'] and r['P1_H6']['whole_path_candidate_supported'] and quality['calibration_eligible'])
        item=dict(frame=i,split='train' if i in TRAIN else 'holdout',far_screen_pass=far,
            raw_gap_pitch=None if gap is None else gap/PITCH,whole_supported=r['P1_H6']['whole_path_candidate_supported'],
            eligible=eligible,P1_reasons=r['P1_H6']['reasons'],quality=quality,geometry=geom,
            observation_sha256=sha(path),model_forwards=r['P1_H6']['model_forward_count'])
        summary.append(item);dump(out/f'quality_{i:06d}.json',item)
        print('calibration',i,'whole',item['whole_supported'],'eligible',eligible,flush=True)
    cap.release()
    eligible=[r for r in summary if r['eligible']]
    # No estimate is promoted without independent curve and holdout evidence.
    receipt=dict(status='candidate_collection_complete',train=TRAIN,holdout=HOLDOUT,eligible_frames=[x['frame'] for x in eligible],
        calibration_valid=False,d_original_px=None,d_sigma_px=None,fallback_reference_d=D,
        reason='insufficient_current_whole_bend_and_holdout_evidence',
        actual_model_forwards=sum(x['model_forwards'] for x in summary),elapsed_s=time.perf_counter()-tick,
        source_sha256=bind['source_sha256'],policy_sha256=sha(__file__),observations=summary)
    if sum(x['eligible'] for x in summary if x['frame'] in TRAIN)>=3 and sum(x['eligible'] for x in summary if x['frame'] in HOLDOUT)>=2:
        receipt['reason']='sufficient_candidates_require_joint_fit_and_holdout_before_promotion'
    dump(out/'receipt.json',receipt);print('CALIBRATION_COLLECTION_COMPLETE',receipt['reason'],flush=True)
if __name__=='__main__':main()
