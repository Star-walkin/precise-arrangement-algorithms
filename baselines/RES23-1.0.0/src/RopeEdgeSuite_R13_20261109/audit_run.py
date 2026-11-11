"""Independent artifact/eligibility audit; no truth-accuracy claims."""
import time,json
from collections import Counter
import cv2,numpy as np
from common import *

def main():
    cv2.setNumThreads(1);run=ROOT/'runs/dual_20260923_113745';bind=read(run/'binding.json');rows=[];failures=[]
    for i in range(bind['processed_frames_planned']):
        deadline=time.monotonic()+600
        while not (run/'frames'/f'{i:06d}.json.gz').exists():
            assert time.monotonic()<deadline;time.sleep(.2)
        r=record(i,run);old=record(i);p=r['provenance_R13'];reg=r['region'];g=r['geometry_GV1'];t=r['temporal'];checks={}
        checks['frame_binding']=r['frame_index']==old['frame_index']==i and r['source_pixel_sha256']==old['source_pixel_sha256']
        checks['old_record_hash']=p['same_frame_RES12_record_sha256']==sha(RUN12/'frames'/f'{i:06d}.json.gz')
        checks['code_binding']=p['boundary_code_sha256']==bind['boundary_code_sha256'] and p['geometry_code_sha256']==bind['geometry_code_sha256']
        checks['tracking_unchanged']=r['S1_tracker_full']==old['S1_tracker_full']
        checks['fixed_reference_only']=g['d_original_px']==D and not g['d_calibrated'] and not t['certified_distance_valid']
        checks['numeric_requires_all_current_gates']=t['value'] is None or reg['valid'] and g['conditional_projection_valid']
        checks['display_five_range']=t['display_N'] is None or 0<=t['display_N']<=5
        checks['rejected_roi_no_F3']=reg['valid'] or not r['fresh_F3']['called'] and r['fresh_F3']['point_count']==0 and r['BG4'] is None and reg['edge_y'] is None and not reg['top_curve_supported_segments']
        if r['s1']['valid']:
            a,b,c=r['s1']['current_line_abc'];checks['raw_projection_formula']=abs(g['y_at_d']-(-(a*D+c)/b))<1e-6
        mask=cv2.imdecode(np.fromfile(FRONT12/'regions'/f'{i:06d}.png',np.uint8),0)
        checks['input_mask_binding']=pixelsha(mask)==p['original_mask_pixel_sha256']
        ranges=reg['identity_guard'].get('rejected_column_ranges_xy',[])
        for lo,hi in ranges:mask[:,max(0,int(lo)):min(mask.shape[1],int(hi)+1)]=0
        checks['ROI_exclusion_hash']=pixelsha(mask)==p['current_mask_pixel_sha256']
        mode=p['F3_mode']
        if mode.startswith('exact_same_frame'):
            checks['reuse_inputs_equal']=not p['mask_changed'] and r['fresh_F3']==old['fresh_F3'] and r['BG4']==old['BG4']
        elif mode.startswith('fresh_current'):
            checks['fresh_F3_required_by_ROI_change']=p['mask_changed'] and reg['valid']
            pts=np.asarray(r['fresh_F3']['raw_points'],float).reshape(-1,8)
            if len(pts):
                xx=pts[:,1]*2
                checks['new_points_outside_rejected_columns']=not any(np.any((xx>=max(0,int(lo)))&(xx<min(mask.shape[1],int(hi)+1))) for lo,hi in ranges)
        checks['no_contact_promotion']=r['ECW_decision_R13']['physical_contact_confirmed'] is False
        bad=[k for k,v in checks.items() if not v]
        if bad:failures.append(dict(frame=i,failed=bad))
        rows.append(dict(frame=i,passed=not bad,checks=checks))
        if i%100==0:print('audited',i,'failures',len(failures),flush=True)
    dump(run/'audit_rows.json',rows);dump(run/'audit_summary.json',dict(frames=len(rows),complete=True,passed=not failures,failures=failures,
        check_counts=dict(Counter(k for r in rows for k in r['checks'])),accuracy_is_not_tested=True))
    print('AUDIT_COMPLETE',len(rows),'FAILURES',failures[:10],flush=True)
    assert not failures
if __name__=='__main__':main()
