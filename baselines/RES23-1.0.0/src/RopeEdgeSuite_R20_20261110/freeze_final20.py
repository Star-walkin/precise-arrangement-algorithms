"""Freeze the delivered implementation and link its complete receipts."""
import shutil,datetime
from common20 import *

def main():
    audit=read(ROOT/'final_audit.json');assert audit['passed']
    for name in ('f3_v2_receipt.json','group_v2_receipt.json','geometry_v5_receipt.json','count_receipt.json','delivery_receipt.json'):
        assert read(ROOT/name)['status']=='complete',name
    assert read(ROOT/'d_calibration.json')['partial'] is False
    source=ROOT/'source_snapshot';source.mkdir(exist_ok=True)
    production=['common20.py','roi20.py','roi21.py','roi22.py','probe_roi22.py','correct_roi22_batch.py','extract_roi21.py','extract_parallel22.py','groups20.py','group_roi21.py','tangent_column.py',
        'coil_edge_axis.py','coil_edge_axis_v3.py','coil_edge_axis_v4.py','geometry20.py','geometry22.py','measure_geometry_v5.py',
        'calibrate_d.py','calibrate_original.py','calibration20.py','calibration21_dense.py','calibrate_guard.py','refine_sparse_d.py','publish_dense_calibration.py','phase20.py','phase21.py',
        'count20.py','count_all.py','recover_points20.py','temporal_contact20.py','contact_fallback20.py','render21.py','finish_v2.py','render22.py','finish_v3.py','finish_v4.py',
        'audit_results20.py','test_measurement_contracts.py','build_report20.py','audit_axis_stability.py','freeze_final20.py',
        'region_fragments23.py','current_boundary23.py','probe_fragments23.py','correct_fragments23.py',
        'region_fragments24.py','probe_fragments24.py','correct_fragments24.py','create_fragment24_trial.py','gate_inactive_f3.py',
        'promote_current_identity24.py','review_fragment_candidates.py','final_statistics20.py','update_visual_review20.py']
    copies=[]
    for name in production:
        p=ROOT/name;dest=source/name;shutil.copy2(p,dest);assert sha(p)==sha(dest);copies.append(dict(path=str(p),snapshot=str(dest),sha256=sha(p)))
    for name in ('motion_phases_v2.json','d_calibration.json','iteration_log.md','measurement_contract_tests.json','visual_review.json','axis_stability.json'):
        shutil.copy2(ROOT/name,source/name)
    receipts={name:sha(ROOT/name) for name in ('f3_v2_receipt.json','group_v2_receipt.json','geometry_v5_receipt.json','d_calibration.json','count_receipt.json','delivery_receipt.json','final_audit.json','measurement_contract_tests.json')}
    receipts['roi22_correction_receipt.json']=sha(ROOT/'roi22_correction_receipt.json')
    receipts['fragments23_correction_receipt.json']=sha(ROOT/'fragments23_correction_receipt.json')
    receipts['fragments24_correction_receipt.json']=sha(ROOT/'fragments24_correction_receipt.json')
    for name in ('final_statistics.json','visual_review.json','report.html'):
        receipts[name]=sha(ROOT/name)
    manifest=dict(name='RopeEdgeSuite R20 / F3-ROI21 with ROI22 count-band correction + CE4-TC1 + per-phase d + local contact evidence',
        date=datetime.datetime.now().isoformat(),status='full_video_revalidation_delivered',sources=plan(),
        production_sources=copies,receipts=receipts,front_end='accepted_R19_delivery_manifest_selected_runs',
        front_end_manifest_sha256=sha(R19/'delivery_manifest.json'),front_end_corrections='Explicit per-frame current-image flange fragment repairs linked by count record hashes; accepted rope tracking preserved',
        caveats=['conditional_d_is_not_direct_physical_bend_ground_truth','offline_d_calibration_can_use_later_phase_frames',
            'single_camera_visual_contact_candidate_not_material_contact_confirmation','coverage_is_not_accuracy',
            'three_short_sources_have_insufficient_temporal_calibration_information'])
    dump(ROOT/'release_manifest.json',manifest);print('Frozen',len(copies),'production sources',flush=True)

if __name__=='__main__':main()
