"""在第一次新预测之前保存输入、预算、比较范围；拒绝覆盖协议。"""
from datetime import datetime, timezone
from experiment_io import HERE, MANIFEST, QA, BASELINE, C3, C4, sha, dump

if __name__ == '__main__':
    target = HERE / 'protocol.json'
    if target.exists(): raise FileExistsError(target)
    dump(target, {
        'version': 'IR-A1.0.0', 'name': '活动绳直段与转弯候选 A1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'input_manifest_sha256': sha(MANIFEST), 'qa_plan_sha256': sha(QA),
        'baseline_manifest_sha256': sha(BASELINE),
        'c3_integrity_sha256': sha(C3 / 'integrity_audit.json'),
        'c4_output_manifest_sha256': sha(C4 / 'output_manifest.json'),
        'input_count': 400, 'camera_counts': {'left': 200, 'right': 200},
        'pilot_count': 38, 'selection': 'Unchanged pre-existing fixed QA38; all400 remains denominator',
        'allowed_inputs': 'Only manifest400 original PNG plus frozen C3/C4 automatic predictions',
        'max_core_development_rounds': 3,
        'revision_policy': 'Each round in new directory with code snapshot; never overwrite failed predictions. Stop branch when no independently observed net gain; no ranking network training.',
        'final_policy': 'Select and freeze one source set after fixed38 inspection, then apply once to all400. Rendering/integrity fixes documented separately.',
        'comparison': ['existing_C4_free_span_diagnostics', 'straight_geometric_extension', 'same_pool_G0_geometry', 'same_pool_G2_original_image_support'],
        'candidate_limit': 6,
        'acceptance': {
            'execution': '38/38 before full400, finite coordinates and same original frame; original/source hashes unchanged',
            'image_evidence': 'Inspect fixed QA including failures and near ties, compare with unmodified original; report observed wrong paths even when formal identity is unscorable',
            'contact': 'No physical contact acceptance without independent evidence; projected straight-to-curve change only image transition candidate',
            'selection': 'No automatic not_detected treated as verified absence; selected/uncertain/not_detected all retained',
            'no_success_by_abstention': 'Report output coverage alongside rejection; all unknown is not a successful contact detector',
            'stop': 'Maximum3 core rounds or no concrete improvement, whichever comes first; retain unresolved failures in final results'
        },
        'units': {'width': 'original_image_pixels_optical_not_physical_diameter', 'angle': 'degrees_image_x_right_y_down', 'contact': 'image_coordinates_not_3D'},
        'truth_isolation': 'Any matching open reference labels only in independent evaluator after saved predictions, never in algorithm inputs',
        'known_roi_risk': 'C4 SAM union masks can include gloves/tools or miss dark winding surface; automatic semantic proposals are not truth',
        'equipment_access': False, 'reserved_inputs': False, 'online_replacement': False,
    })
    print(str(target))
