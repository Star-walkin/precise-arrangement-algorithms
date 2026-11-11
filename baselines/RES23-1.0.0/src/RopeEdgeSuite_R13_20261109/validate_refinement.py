import sys,time
import cv2,numpy as np
from common import *
from test_gates import scene
from local_foreground_v2 import local_count_guard
from refine_boundary import refine

def main():
    cv2.setNumThreads(1)
    im,reg=scene();cv2.rectangle(im,(220,255),(650,345),(245,245,245),-1)
    v=local_count_guard(im,reg,450,20)
    assert not v['valid'],'Broad tool contaminates the lower reference and must still be detected'
    assert max(x['threshold'] for x in v['columns'])<=200
    im,reg=scene();assert local_count_guard(im,reg,450,20)['valid']
    # Same10 visually labeled samples are now regression checks; not a new holdout.
    labels=read(ROOT/'visual_holdout_labels.json');cap=cv2.VideoCapture(read(RUN12/'binding.json')['source']);regression=[]
    for i in sorted(labels['clear']+labels['foreground_in_first_three_pitch']):
        cap.set(1,i);ok,im=cap.read();assert ok;r=record(i)
        g=local_count_guard(im,r['region_BG5']);expected=i in labels['clear']
        regression.append(dict(frame=i,expected=expected,passed=g['valid'],match=g['valid']==expected))
    additional=[]
    for i in [368,388,392,400]:
        cap.set(1,i);ok,im=cap.read();assert ok;r=record(i);g=local_count_guard(im,r['region_BG5'])
        additional.append(dict(frame=i,local_guard_rejected=not g['valid'],broad_fraction=g.get('broad_fraction')))
    cap.release();assert all(r['match'] for r in regression);assert all(r['local_guard_rejected'] for r in additional)
    dump(ROOT/'refinement_test_receipt.json',dict(passed=True,synthetic_checks=2,
        previous_holdout_now_regression=regression,known_missed_case_regression=additional,
        code_sha256=sha(ROOT/'local_foreground_v2.py'),not_independent_new_video_accuracy=True))
    run=ROOT/'runs/dual_20260923_113745_refined';prior=ROOT/'runs/dual_20260923_113745';rows=[];fail=[]
    bind=read(run/'binding.json')
    for i in range(600):
        deadline=time.monotonic()+600
        while not (run/'frames'/f'{i:06d}.json.gz').exists():
            assert time.monotonic()<deadline;time.sleep(.2)
        r=record(i,run);p=record(i,prior);prov=r['refinement_provenance'];g=r['geometry_GV1'];reg=r['region']
        checks=dict(source=r['source_pixel_sha256']==p['source_pixel_sha256'],frame=r['frame_index']==i,
            first_pass_binding=prov['first_pass_record_sha256']==sha(prior/'frames'/f'{i:06d}.json.gz'),
            code=prov['local_guard_code_sha256']==bind['local_foreground_code_sha256'],
            SG2_unchanged=r['S1_tracker_full']==p['S1_tracker_full'],
            only_additional_rejections=not reg['valid'] or p['region']['valid'],
            no_false_calibration=not g['d_calibrated'] and not r['temporal']['certified_distance_valid'],
            numeric_gate=r['temporal']['value'] is None or reg['valid'] and g['conditional_projection_valid'],
            no_contact_promotion=not r['ECW_decision_R13']['physical_contact_confirmed'])
        if reg['valid']:
            checks['spatial_reuse_exact']=r['fresh_F3']==p['fresh_F3'] and r['BG4']==p['BG4']
            checks['local_guard_pass']=reg['local_count_guard_v2']['valid']
        else:checks['rejected_no_F3_no_edge']=r['BG4'] is None and not r['fresh_F3']['called'] and reg['edge_y'] is None and not reg['top_curve_supported_segments']
        bad=[k for k,v in checks.items() if not v]
        if bad:fail.append(dict(frame=i,failed=bad))
        rows.append(dict(frame=i,passed=not bad,checks=checks))
        if i%100==0:print('refined audit',i,'failures',len(fail),flush=True)
    dump(run/'audit_rows.json',rows);dump(run/'audit_summary.json',dict(complete=True,frames=600,passed=not fail,failures=fail,accuracy_is_not_tested=True))
    print('REFINED_AUDIT_COMPLETE',len(fail),flush=True);assert not fail
if __name__=='__main__':main()
