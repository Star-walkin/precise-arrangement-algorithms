"""Actual wrong-winding negatives plus the faithfully reproduced first loss."""
from pathlib import Path
import sys,json,gzip
import cv2,numpy as np
from tracker_sg2 import ROOT,RES,Tracker,SegmentedS1,PARAMETERS
sys.path.insert(0,str(ROOT.parent/'RopeEdgeSuite_R11_20261108'))
from run_bg5 import e,record,dump
import region_bg5_refined as bg5

def main():
    cv2.setNumThreads(1);prior=RES/'transfer_runs/dual_20260923_113745'
    binding=e.read(prior/'binding.json');cap=cv2.VideoCapture(binding['source']);rows=[]
    for n in (336,526):
        cap.set(1,n);ok,im=cap.read();old=record(prior,n);assert ok and e.pixelsha(im)==old['source_pixel_sha256']
        reg=bg5.segment(im);a=Tracker(reg['surface_mask']);a.set_region(reg)
        s=old['S1_tracker_full'];seed=np.asarray(s['seed_xy_original'])
        a.entry_seed=seed;a.entry_witness=a._seed_witness(im,seed)
        good,why,metrics=a.identity(s['S1'],seed,s['diagnostics']['seed_validation'],n/10)
        rows.append(dict(frame=n,case='original_wrong_winding_candidate',accepted=good,reasons=why,metrics=metrics))
        assert not good and 'cold_start_free_support_insufficient' in why
    cap.set(1,316);ok,im=cap.read();assert ok and e.pixelsha(im)==record(prior,316)['source_pixel_sha256']
    diag=e.read(ROOT.parent/'S1_loss_113745_20261109/replay_316.json');seed=diag['seed_xy_original']
    adapter=SegmentedS1();result=adapter.step(im,seed,seed_source='same_current_seed_regression_case')
    g=result['diagnostics']['SG2_segments'];assert result['valid'] and g['support_span_px']>=400
    rows.append(dict(frame=316,case='reproduced_first_loss_current_evidence_recovery',accepted=True,
        observed_points=len(result['line_points_xy']),support_span_px=g['support_span_px'],bridges=g['bridges']))
    cap.release();dump(ROOT/'actual_case_test_receipt.json',dict(passed=True,cases=rows))
    print(json.dumps(rows,ensure_ascii=False))

if __name__=='__main__':main()
