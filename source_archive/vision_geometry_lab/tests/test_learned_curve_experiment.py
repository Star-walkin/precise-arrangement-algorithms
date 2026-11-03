from pathlib import Path
import hashlib
import json
import tempfile
import unittest

import cv2
import numpy as np
from vision_geometry_lab.evaluate_learned_curve import (
    _file_sha256,_json_sha256,anchored_component,build_negative_probe_protocol,
    crop_roi,intersect_rois,mask_diagnostics,point_mask_distances,
    prompt_compliance,validate_worker_bundle)


def test_blank_mask_never_reports_zero_distance():
    mask=np.zeros((20,30),bool)
    assert point_mask_distances(mask,[[2,2],[5,5]])==[None,None]


def test_disconnected_mask_cannot_count_as_connected_target():
    mask=np.zeros((30,50),bool);mask[5:10,5:10]=True;mask[20:25,35:40]=True
    assert point_mask_distances(mask,[[37,22]])==[0.0]
    component=anchored_component(mask,[7,7])
    assert point_mask_distances(component,[[37,22]])[0]>20
    assert anchored_component(mask,[15,15]).sum()==0


def test_crop_is_clamped_and_reference_not_used():
    assert crop_roi((300,400,3),[10,10],[100,80])==[0,0,100,80]
    assert crop_roi((300,400,3),[380,280],[100,80])==[330,240,400,300]


def test_subpixel_border_coordinates_do_not_index_outside_image():
    mask=np.ones((20,30),bool)
    assert point_mask_distances(mask,[[29.9,19.9],[30.,19.]])==[0.,None]


def _audit(frame_index=246):
    return dict(reference=dict(frame_index=frame_index,points_original_px=[]),
                optical_width_px=4.,free_segment=[[5.,5.],[45.,5.]],
                common_roi=[0,0,60,20],fixed_sameframe_common_roi=[0,0,60,20],
                audit_regions={})


def test_free_span_reports_raw_and_anchor_connected_support():
    mask=np.zeros((20,60),bool)
    mask[4:7,0:25]=True
    mask[4:7,26:60]=True
    case=dict(name='split',roi=[0,0,60,20],points=[[5.,5.]],labels=[1],frame_index=246)
    _,_,diagnostics=mask_diagnostics(mask,(20,60,3),case,_audit())
    assert diagnostics['raw_free_span_support_within_half_width_case_roi']==1.0
    assert .4<diagnostics['connected_free_span_support_within_half_width_case_roi']<.7
    assert diagnostics['raw_free_span_support_within_half_width_fixed_sameframe_common_roi']==1.0
    assert diagnostics['connected_free_span_support_within_half_width_fixed_sameframe_common_roi']<.7


def test_same_frame_comparison_roi_is_one_fixed_intersection():
    rois=[[0,0,100,100],[10,10,90,90],[20,5,80,70]]
    fixed=intersect_rois(*rois)
    assert fixed==[20,10,80,70]
    audit=dict(reference=dict(frame_index=246,points_original_px=[]),optical_width_px=4.,
               free_segment=None,common_roi=rois[1],fixed_sameframe_common_roi=fixed,
               audit_regions={})
    for index,roi in enumerate(rois):
        crop=np.ones((roi[3]-roi[1],roi[2]-roi[0]),bool)
        case=dict(name=f'case_{index}',roi=roi,points=[[30.,20.]],labels=[1],frame_index=246)
        _,_,diagnostics=mask_diagnostics(crop,(100,100,3),case,audit)
        assert diagnostics['fixed_sameframe_common_roi']==fixed
        assert diagnostics['fixed_sameframe_common_roi_pixels']==3600
        assert diagnostics['foreground_fraction_in_fixed_sameframe_common_roi']==1.0


def test_prompt_compliance_distinguishes_connectivity_and_negative_failure():
    disconnected=np.zeros((30,50),bool)
    disconnected[4:8,4:8]=True;disconnected[4:8,29:33]=True
    component=anchored_component(disconnected,[5,5])
    audit=prompt_compliance(disconnected,component,[[5,5],[30,5],[20,20]],[1,1,0])
    assert audit['positive_prompt_inside']==[True,True]
    assert audit['negative_prompt_outside']==[True]
    assert audit['prompt_constraints_satisfied'] is True
    assert audit['positive_prompt_in_anchor_component']==[True,False]
    assert audit['all_positive_prompts_anchor_connected'] is False

    broad=np.zeros((30,50),bool);broad[4:22,4:33]=True
    audit=prompt_compliance(broad,anchored_component(broad,[5,5]),
                            [[5,5],[30,5],[20,20]],[1,1,0])
    assert audit['all_positive_prompts_anchor_connected'] is True
    assert audit['negative_prompt_outside']==[False]
    assert audit['prompt_constraints_satisfied'] is False


def test_empty_regions_and_samples_are_none_and_json_safe():
    case=dict(name='empty',roi=[0,0,10,10],points=[[2,2]],labels=[1],frame_index=246)
    audit=dict(reference=dict(frame_index=246,points_original_px=[]),optical_width_px=2.,
               free_segment=[[20,20],[30,20]],common_roi=[20,20,30,30],
               fixed_sameframe_common_roi=[20,20,30,30],
               audit_regions={'outside':[20,20,30,30]})
    _,_,diagnostics=mask_diagnostics(np.zeros((10,10),bool),(40,40,3),case,audit)
    assert diagnostics['case_visible_audit_roi'] is None
    assert diagnostics['foreground_fraction_in_case_visible_audit_roi'] is None
    assert diagnostics['fixed_sameframe_common_roi'] is None
    assert diagnostics['case_visible_free_span_samples']==0
    assert diagnostics['raw_free_span_support_within_half_width_case_roi'] is None
    assert diagnostics['connected_free_span_support_within_half_width_case_roi'] is None
    assert diagnostics['coarse_audit_regions']['outside']['case_roi_overlap_fraction']==0.0
    assert diagnostics['coarse_audit_regions']['outside']['foreground_fraction_in_case_overlap'] is None
    assert diagnostics['negative_prompt_satisfaction_fraction'] is None
    json.dumps(diagnostics,allow_nan=False)


def _write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),'utf-8')


def _worker_bundle(tmp_path):
    output=tmp_path/'output';output.mkdir()
    image=tmp_path/'clean.png';pixels=np.arange(75,dtype=np.uint8).reshape(5,5,3)
    assert cv2.imwrite(str(image),pixels)
    checkpoint=tmp_path/'checkpoint.pt';checkpoint.write_bytes(b'checkpoint')
    case=dict(name='case',image=str(image),roi=[0,0,5,5],points=[[2.,2.]],labels=[1],
              manual_input_indices=[],frame_index=246,kind='main',roi_kind='full')
    protocol=dict(schema_version=1,output=str(output),cases=[case],
                  mask_selection='highest SAM score, frozen before diagnostics',
                  branch_type='primary_pre_registered_cases',training=False,hardware_access=False)
    protocol_path=output/'protocol.json';_write_json(protocol_path,protocol)
    protocol_hash=_file_sha256(protocol_path);checkpoint_hash=_file_sha256(checkpoint)
    folder=output/'case';folder.mkdir()
    masks=[]
    for index in range(3):
        path=folder/f'mask_{index}.png'
        candidate=np.zeros((5,5),np.uint8);candidate[index:index+2,index:index+2]=255
        assert cv2.imwrite(str(path),candidate);masks.append(_file_sha256(path))
    record=dict(case,selected_index=1,scores=[.1,.9,.2],
                selection='argmax_model_score; no reference metric used',
                protocol_sha256=protocol_hash,case_sha256=_json_sha256(case),
                image_sha256=_file_sha256(image),mask_sha256=masks,
                checkpoint_sha256=checkpoint_hash,checkpoint=str(checkpoint))
    _write_json(folder/'inference.json',record)
    worker=dict(protocol_sha256=protocol_hash,checkpoint_sha256=checkpoint_hash,
                branch_type=protocol['branch_type'],training=False,hardware_access=False,cases=[record])
    _write_json(output/'worker_results.json',worker)
    audit=dict(reference=dict(frame_index=246,points_original_px=[]),
               clean_sha256=hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest())
    return protocol,audit,output,image,checkpoint


def test_reuse_accepts_untampered_manifest():
    with tempfile.TemporaryDirectory() as folder:
        protocol,audit,_,_,_=_worker_bundle(Path(folder))
        integrity=validate_worker_bundle(protocol,audit)
        assert integrity['protocol_sha256']==_file_sha256(Path(protocol['output'])/'protocol.json')
        assert integrity['checkpoint_sha256']


def _assert_reuse_rejects(tamper,message):
    with tempfile.TemporaryDirectory() as folder:
        protocol,audit,output,image,checkpoint=_worker_bundle(Path(folder))
        inference_path=output/'case/inference.json';worker_path=output/'worker_results.json'
        if tamper=='protocol':
            with (output/'protocol.json').open('a',encoding='utf-8') as stream: stream.write('\n')
        elif tamper=='case':
            record=json.loads(inference_path.read_text('utf-8'));record['case_sha256']='0'*64
            _write_json(inference_path,record)
            worker=json.loads(worker_path.read_text('utf-8'));worker['cases']=[record];_write_json(worker_path,worker)
        elif tamper=='image':
            pixels=cv2.imread(str(image));pixels[0,0]=255-pixels[0,0];assert cv2.imwrite(str(image),pixels)
        elif tamper=='checkpoint':
            with checkpoint.open('ab') as stream: stream.write(b'changed')
        else:
            path=output/'case/mask_0.png';mask=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            mask[4,4]=255-mask[4,4];assert cv2.imwrite(str(path),mask)
        with unittest.TestCase().assertRaisesRegex(ValueError,message):
            validate_worker_bundle(protocol,audit)


def test_reuse_rejects_tampered_protocol():
    _assert_reuse_rejects('protocol','protocol_sha256')


def test_reuse_rejects_tampered_case():
    _assert_reuse_rejects('case','case_sha256')


def test_reuse_rejects_tampered_image():
    _assert_reuse_rejects('image','image_sha256')


def test_reuse_rejects_tampered_checkpoint():
    _assert_reuse_rejects('checkpoint','checkpoint_sha256')


def test_reuse_rejects_tampered_mask():
    _assert_reuse_rejects('mask','mask_sha256')


def test_negative_probe_is_explicit_posthoc_child_protocol():
    with tempfile.TemporaryDirectory() as folder:
        tmp_path=Path(folder)
        points=[[10.,10.],[15.,12.],[20.,15.],[25.,19.],[30.,24.],[35.,30.]]
        cases=[]
        for roi_kind in ('focus','local'):
            cases.append(dict(name=roi_kind+'_manual_all6',image='clean.png',roi=[0,0,100,100],
                              points=[[5.,5.],*points],labels=[1]*7,manual_input_indices=list(range(6)),
                              frame_index=246,kind='main',roi_kind=roi_kind))
        original=dict(output=str(tmp_path/'primary'),cases=cases,mask_selection='argmax',
                      training=False,hardware_access=False)
        audit=dict(reference=dict(points_original_px=points),optical_width_px=4.)
        protocol=build_negative_probe_protocol(original,audit,'parent-hash')
        assert protocol['branch_type']=='exploratory_negative_prompt_probe'
        assert protocol['post_hoc'] is True and protocol['blind'] is False
        assert protocol['primary_pre_registered'] is False
        assert protocol['exploratory_after_primary'] is True
        assert protocol['parent_protocol_sha256']=='parent-hash'
        assert protocol['negative_semantics']
        assert protocol['training'] is False and protocol['hardware_access'] is False
        assert {case['source_case_name'] for case in protocol['cases']}=={
            'focus_manual_all6','local_manual_all6'}
        assert all(case['kind']=='exploratory_negative_probe' for case in protocol['cases'])
