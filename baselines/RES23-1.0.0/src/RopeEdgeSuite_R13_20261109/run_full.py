"""Controlled BG6/GV1 experiment on every source frame.

Unchanged SG2 current-frame measurements are source-bound reused. F3 reuse is
allowed only for byte-identical current ROI and identical current SG2 output;
an edited ROI triggers fresh F3+curve extraction. Invalid ROI never calls F3.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import sys,time,gzip,json,copy,argparse
from collections import deque,Counter as Tally
from concurrent.futures import ProcessPoolExecutor
import cv2,numpy as np
from common import *
from region_bg6 import validate
from geometry_gv1 import assess
sys.path.insert(0,str(R11));from run_bg5 import e

def fresh_spatial(im,mask,track,region,old,out):
    import importlib.util
    spec=importlib.util.spec_from_file_location('_res12_fresh_extract',PREV/'run_full.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    e.extract_current=module.extract_current
    e.segment=lambda image:dict(record=region,surface_mask=mask)
    return e.spatial_frame(im,track,old['camera_motion'],old['frame_index'],old['nominal_time_s'],
        read(RUN12/'binding.json')['source_sha256'],out)

def worker(im,i,out):
    cv2.setNumThreads(1);tick=time.perf_counter();old=record(i)
    assert pixelsha(im)==old['source_pixel_sha256']
    region=validate(im,old['region_BG5']);track=old['S1_tracker_full'];regsmall=region
    mask=cv2.imdecode(np.fromfile(FRONT12/'regions'/f'{i:06d}.png',np.uint8),0)
    assert pixelsha(mask)==old['current_mask_sha256']
    originalmask=pixelsha(mask)
    for lo,hi in region['identity_guard'].get('rejected_column_ranges_xy',[]):
        mask[:,max(0,int(lo)):min(mask.shape[1],int(hi)+1)]=0
    currentmask=pixelsha(mask);changed=currentmask!=originalmask
    if region['valid'] and changed:
        r=fresh_spatial(im,mask,track,regsmall,old,out);origin='fresh_current_F3_and_curves_after_foreground_ROI_exclusion'
    elif region['valid']:
        r=old;origin='exact_same_frame_F3_reuse_identical_pixels_ROI_and_SG2'
    else:
        r=old;r['BG4']=None;r['observation']=None;r['fresh_F3']=dict(called=False,point_count=0,raw_points=[],BG3_curve_count=0)
        origin='not_called_boundary_rejected'
    r['region']=regsmall;r['region_BG6']=region;r['region_gate_passed']=region['valid']
    r['geometry_GV1']=assess(track,region,read(ROOT/'d_calibration/receipt.json'))
    r['version']='RES-1.3/BG6-GV1';r['S1_tracker_full']=track
    r['baseline_RES12']=dict(s1_valid=old['s1']['valid'],region_valid=old['region_BG5']['valid'],
        old_raw_projection_gap=None if old['intersection_y_px'] is None or old['region_BG5']['edge_y'] is None else old['intersection_y_px']-old['region_BG5']['edge_y'],
        count=old['temporal'])
    r['provenance_R13']=dict(source_pixel_sha256=old['source_pixel_sha256'],
        same_frame_RES12_record_sha256=sha(RUN12/'frames'/f'{i:06d}.json.gz'),
        original_mask_pixel_sha256=originalmask,current_mask_pixel_sha256=currentmask,mask_changed=changed,
        SG2_mode='unchanged_source_bound_same_frame_reuse',F3_mode=origin,
        boundary_code_sha256=sha(ROOT/'region_bg6.py'),geometry_code_sha256=sha(ROOT/'geometry_gv1.py'))
    r['reasons']=list(dict.fromkeys(region['reasons']+r.get('reasons',[])))
    r['host_time_s']=old['host_time_s'];r['nominal_time_s']=i/10
    r['R13_spatial_seconds']=time.perf_counter()-tick
    return r

def temporal_step(counter,r):
    g=r['geometry_GV1'];original=copy.deepcopy(r.get('observation'));r['pre_GV1_observation']=original
    if not g['conditional_projection_valid']:
        counter.clear();r['observation']=None
    elif original:
        obs=copy.deepcopy(original);pitch=obs['pitch_px']
        # Explicit projection sensitivity participates in observation weighting.
        # d sensitivity stays a separate scenario because d has no calibration.
        obs['sigma']=max(obs['sigma'],g['conditional_gap_sensitivity_px']/pitch)
        obs['conditional_reference_d_only']=not g['d_calibrated'];r['observation']=obs
    t=counter.step(r)
    if not g['conditional_projection_valid']:
        assert t['value'] is None
        t['events']=list(dict.fromkeys(g['reasons']+t['events']))
    t['certified_distance_valid']=g['certified_distance_valid'] and t['value'] is not None
    t['value_semantics']='conditional_estimate_at_unvalidated_reference_d' if not g['d_calibrated'] else 'calibrated_d_estimate'
    return t

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);ap.add_argument('--limit',type=int,default=600);a=ap.parse_args()
    cv2.setNumThreads(1);out=ROOT/'runs'/('dual_20260923_113745' if a.limit==600 else f'pilot{a.limit}')
    out.mkdir(parents=True,exist_ok=False);(out/'frames').mkdir()
    binding=read(RUN12/'binding.json');baseline=e.verify_baseline();oldmanifest=sha(PREV/'delivery_manifest.json')
    binding.update(algorithm='RES-1.3/BG6-GV1',processed_frames_planned=a.limit,
        experiment='current-image foreground guard plus conditional projection validation; unchanged SG2 controlled ablation',
        d_calibration_receipt_sha256=sha(ROOT/'d_calibration/receipt.json'),d_calibration_valid=False,
        prior_delivery_manifest_sha256=oldmanifest,runner_sha256=sha(__file__),
        boundary_code_sha256=sha(ROOT/'region_bg6.py'),geometry_code_sha256=sha(ROOT/'geometry_gv1.py'))
    dump(out/'binding.json',binding);cap=cv2.VideoCapture(binding['source']);counter=e.Counter();pending=deque();rows=[];start=time.perf_counter()
    def consume():
        future=pending.popleft();r=future.result();i=r['frame_index'];r['temporal']=temporal_step(counter,r)
        # The legacy proximity detector cannot promote a rejected interface.
        r['ECW_decision_R13']=dict(valid=False,physical_contact_confirmed=False,
            reason='contact_not_promoted_without_current_validated_bend_geometry')
        path=out/'frames'/f'{i:06d}.json.gz';tmp=path.with_suffix('.part')
        with gzip.open(tmp,'wt',encoding='utf-8',compresslevel=3) as f:json.dump(plain(r),f,ensure_ascii=False,allow_nan=False)
        tmp.replace(path)
        row=dict(frame=i,time_s=i/10,tracking_valid=r['s1']['valid'],region_valid=r['region']['valid'],
            foreground_rejected='broad_foreground_occludes_count_interface' in r['region']['reasons'],
            geometry_valid=r['geometry_GV1']['conditional_projection_valid'],geometry_reasons=r['geometry_GV1']['reasons'],
            count_state=r['temporal']['state'],value=r['temporal']['value'],display_N=r['temporal']['display_N'],
            F3_mode=r['provenance_R13']['F3_mode'],points=r['fresh_F3']['point_count'],
            old_region_valid=r['baseline_RES12']['region_valid'],old_value=r['baseline_RES12']['count']['value'],
            raw_gap=r['geometry_GV1'].get('raw_gap_px'))
        rows.append(row)
        if len(rows)%20==0 or len(rows)==a.limit:
            progress=dict(processed=len(rows),planned=a.limit,elapsed_s=round(time.perf_counter()-start,1),
                region_valid=sum(q['region_valid'] for q in rows),foreground_rejected=sum(q['foreground_rejected'] for q in rows),
                projection_pass=sum(q['geometry_valid'] for q in rows),numeric=sum(q['value'] is not None for q in rows),
                F3_modes=dict(Tally(q['F3_mode'] for q in rows)))
            dump(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i in range(a.limit):
            ok,im=cap.read();assert ok
            pending.append(pool.submit(worker,im,i,str(out)))
            if len(pending)>=a.workers+1:consume()
        while pending:consume()
    cap.release();assert e.verify_baseline()==baseline and sha(PREV/'delivery_manifest.json')==oldmanifest
    assert sha(binding['source'])==binding['source_sha256']
    dump(out/'compact.json',rows)
    dump(out/'receipt.json',dict(status='complete',frames=len(rows),elapsed_s=time.perf_counter()-start,
        baseline_unchanged=True,previous_delivery_manifest_unchanged=True,source_unchanged=True,
        d_calibration_valid=False,absolute_count_accuracy_validated=False))
if __name__=='__main__':main()
