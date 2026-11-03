import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import vision_geometry_lab.evaluate_centerline_multiscale_v9 as evaluator
from vision_geometry_lab.centerline_multiscale_contract_v9 import rebuild_input_contract,PIXEL_CENTER_MAPPING
from vision_geometry_lab.centerline_quality_cutoff_v8 import build_quality_cutoff_payload
from vision_geometry_lab.centerline_quality_v5 import load_label_quality


def _label(identifier='fixture'):
    return dict(schema_version=1,sample_id=identifier,image_sha256='a'*64,
        coordinate_system='full_image_xy',status='partial',reviewed=True,notes='',
        polylines=dict(centerline=[[10,50],[140,50]],edge_flange=[[10,45],[140,45]],
                       edge_surface=[[10,55],[140,55]],free_seed=[[10,50],[20,50]]),ignore_polygons=[])


def _quality():
    return dict(quality_sha256=None,precision_exclude_polygons=[],regions=[])


def _empty(reason='missing'):
    return dict(segments=[],widths=[],reason=reason,identity_confirmed=False)


def _raises(call,fragment=''):
    try:
        call()
    except ValueError as exc:
        assert fragment in str(exc),(fragment,str(exc))
    else:
        raise AssertionError('Expected explicit rejection')


def _fixture(folder):
    folder=Path(folder)
    rgb=np.zeros((100,160,3),np.uint8);rgb[:,:,0]=np.arange(160,dtype=np.uint8)[None]
    image=folder/'image.png';Image.fromarray(rgb).save(image)
    label=_label();label['image_sha256']=evaluator.file_hash(image)
    annotation=folder/'label.json';annotation.write_text(json.dumps(label),'utf-8')
    sample=dict(sample_id='fixture',group_id='train_session',split='train',image_path=str(image),
        image_sha256=label['image_sha256'],image_shape_hw=[100,160],roi_xyxy=[0,0,160,100],
        annotation_path=str(annotation),source_set='old_pilot_12',
        source=dict(video_path='fixture.avi',frame_index=0,time_s=0.,camera='fixture'))
    record=dict(sample=sample,label=label,quality=_quality(),annotation_sha256=evaluator.file_hash(annotation),cohort='old_pilot')
    manifest=folder/'manifest.json';manifest.write_text(json.dumps(dict(schema_version=1,samples=[sample])),'utf-8')
    manifest_hash=evaluator.file_hash(manifest)
    snapshot=dict(train_group_ids=['train_session'],train_annotation_sha256_by_sample={'fixture':record['annotation_sha256']},
                  train_quality_sha256_by_sample={'fixture':None})
    contract=rebuild_input_contract(rgb,label['polylines']['free_seed'],'native_detail')
    architecture=dict(name='global_descriptor_local_unet',version=1,global_descriptor_channels=8,base_channels=8)
    checkpoint=folder/'checkpoint.pt';checkpoint.write_bytes(b'offline-fixture-not-a-model')
    model=folder/'model.py';model.write_text('# fixture source','utf-8')
    report=folder/'report.json';report.write_text(json.dumps(dict(mode='native_detail',manifest_sha256=manifest_hash,
        architecture=architecture,artifacts=dict(checkpoint=str(checkpoint),checkpoint_sha256=evaluator.file_hash(checkpoint)),**snapshot)),'utf-8')
    output=folder/'fixture.npz'
    body=np.zeros((50,80),np.float32);center=body.copy();coverage=contract['expected_coverage_mask']
    np.savez_compressed(output,body_probability=body,center_probability=center,coverage_mask=coverage)
    meta=dict(schema_version=1,format=evaluator.FORMAT,experiment_version=9,sample_id='fixture',group_id='train_session',split='train',
        image_path=str(image),image_sha256=sample['image_sha256'],annotation_path=str(annotation),
        annotation_sha256=record['annotation_sha256'],quality_sha256=None,mode='native_detail',
        full_image_hw=[100,160],native_roi_hw=[100,160],probability_map_hw=[50,80],roi_xyxy=[0,0,160,100],
        scale_xy=[.5,.5],pixel_center_mapping=PIXEL_CENTER_MAPPING,human_free_seed_used=True,
        ground_truth_center_or_edges_used_as_input=False,quality_used_for_input=False,
        body_head_supervised=True,center_head_supervised=True,real_annotation_trained=True,
        coverage_restricts_reference_denominator=False,
        manifest_sha256=manifest_hash,training_manifest_sha256=manifest_hash,**snapshot,
        checkpoint=str(checkpoint),checkpoint_sha256=evaluator.file_hash(checkpoint),
        output_npz=str(output),output_sha256=evaluator.file_hash(output),
        contract_source=str(evaluator.CONTRACT_SOURCE),contract_source_sha256=evaluator.file_hash(evaluator.CONTRACT_SOURCE),
        model_source=str(model),model_source_sha256=evaluator.file_hash(model),
        training_report=str(report),training_report_sha256=evaluator.file_hash(report),architecture=architecture,
        input_contract=contract['input_contract'],tile_layout=contract['tile_layout'],
        tile_selection_inputs=contract['tile_layout']['selection_inputs'])
    for key in ('input_contract_sha256','global_input_sha256','local_inputs_sha256','free_seed_sha256'):
        meta[key]=contract['input_contract'][key]
    path=folder/'fixture.json';path.write_text(json.dumps(meta),'utf-8')
    return dict(folder=folder,record=record,rgb=rgb,meta=meta,meta_path=path,model=model,manifest=manifest,
                snapshot=snapshot,manifest_hash=manifest_hash,contract=contract)


def _load(fixture):
    with patch.object(evaluator,'MODEL_SOURCE',fixture['model'].resolve()):
        return evaluator.load_prediction(fixture['folder'],fixture['record'],fixture['rgb'],
            fixture['snapshot'],fixture['manifest_hash'],'native_detail')


def test_native_half_resolution_contract_loads_and_keeps_full_coordinates():
    with tempfile.TemporaryDirectory() as folder:
        fixture=_fixture(folder)
        meta,body,center,mask=_load(fixture)
        assert body.shape==center.shape==mask.shape==(50,80)
        assert meta['roi_xyxy']==[0,0,160,100] and np.all(mask==1)


def test_prediction_rejects_source_seed_quality_mode_group_and_target_contamination():
    with tempfile.TemporaryDirectory() as folder:
        f=_fixture(folder);original=copy.deepcopy(f['meta'])
        changes=[('image_sha256','b'*64),('annotation_sha256','b'*64),('quality_sha256','b'*64),
            ('free_seed_sha256','b'*64),('mode','lowpass_control'),('group_id','other_session'),
            ('ground_truth_center_or_edges_used_as_input',True),('quality_used_for_input',True),
            ('train_group_ids',['train_session','extra']),('pixel_center_mapping','naive xy/2')]
        for key,value in changes:
            changed=copy.deepcopy(original);changed[key]=value;f['meta_path'].write_text(json.dumps(changed),'utf-8')
            _raises(lambda:_load(f))


def test_prediction_reconstructs_inputs_and_tiles_not_just_claimed_hashes():
    with tempfile.TemporaryDirectory() as folder:
        f=_fixture(folder);original=copy.deepcopy(f['meta'])
        for field in ('input_contract','tile_layout'):
            changed=copy.deepcopy(original)
            if field=='input_contract': changed[field]['global_input_sha256']='b'*64
            else: changed[field]['tiles'][0]['native_xyxy'][0]+=2
            f['meta_path'].write_text(json.dumps(changed),'utf-8')
            _raises(lambda:_load(f))


def test_coverage_tampering_and_nonzero_unobserved_probabilities_are_rejected():
    with tempfile.TemporaryDirectory() as folder:
        f=_fixture(folder);m=f['meta'];coverage=f['contract']['expected_coverage_mask'].copy();coverage[:,60:]=0
        body=np.zeros((50,80),np.float32);center=body.copy()
        np.savez_compressed(m['output_npz'],body_probability=body,center_probability=center,coverage_mask=coverage)
        m['output_sha256']=evaluator.file_hash(m['output_npz']);f['meta_path'].write_text(json.dumps(m),'utf-8')
        _raises(lambda:_load(f),'Coverage differs')
        expected=dict(f['contract'],expected_coverage_mask=coverage)
        body[:,65]=.1
        np.savez_compressed(m['output_npz'],body_probability=body,center_probability=center,coverage_mask=coverage)
        m['output_sha256']=evaluator.file_hash(m['output_npz']);f['meta_path'].write_text(json.dumps(m),'utf-8')
        with patch.object(evaluator,'rebuild_input_contract',return_value=expected):
            _raises(lambda:_load(f),'outside model coverage')


def test_window_coverage_never_removes_outside_reference_from_denominator():
    label=_label();label['polylines']['centerline']=[[20,50],[140,50]]
    mask=np.zeros((50,80),np.uint8);mask[:,:40]=1
    result=evaluator.score_decoded(label,_quality(),dict(segments=[[[20,50],[60,50]]]),mask,[100,160])
    score=result['metrics']['downstream'];support=result['coverage_support']['downstream']
    assert score['reference_length_px']==120
    assert .30<score['covered_fraction']<.4
    assert .45<support['fraction']<.55 and support['reference_length_px']==120


def test_quality_clips_both_paths_without_inventing_gap_connections():
    label=_label();label['polylines']['centerline']=[[20,50],[140,50]]
    quality=_quality();quality['precision_exclude_polygons']=[[[60,0],[100,0],[100,99],[60,99]]]
    result=evaluator.score_decoded(label,quality,dict(segments=[[[20,50],[140,50]]]),np.ones((50,80),np.uint8),[100,160])
    score=result['metrics']['downstream']
    assert np.isclose(score['reference_length_px'],80) and np.isclose(score['predicted_length_px'],80)
    assert score['reference_segment_count']==score['predicted_segment_count']==2
    assert np.isclose(score['covered_fraction'],1.)


def test_confirmed_cutoff_loader_supplies_only_trusted_center_reference():
    with tempfile.TemporaryDirectory() as folder:
        f=_fixture(folder);label=f['record']['label']
        label['polylines']['centerline']=[[10+10*i,50] for i in range(8)]
        raw=json.dumps(label).encode();Path(f['record']['sample']['annotation_path']).write_bytes(raw)
        payload=build_quality_cutoff_payload(f['record']['sample'],raw)['quality_payload']
        Path(f['record']['sample']['annotation_path']).with_suffix('.quality.json').write_text(json.dumps(payload),'utf-8')
        quality=load_label_quality(f['record']['sample'])
        reference,source=evaluator.scorable_reference(label,quality)
        assert reference==label['polylines']['centerline'][:7] and len(reference)==7
        result=evaluator.score_decoded(label,quality,dict(segments=[[[20,50],[140,50]]]),np.ones((50,80),np.uint8),[100,160])
        assert np.isclose(result['metrics']['downstream']['reference_length_px'],50)
        assert result['reference_source']=='hash_bound_confirmed_trusted_centerline'


def test_no_prediction_scores_zero_but_no_reference_is_null():
    result=evaluator.score_decoded(_label(),_quality(),_empty(),None,[100,160])
    assert result['metrics']['downstream']['covered_fraction']==0
    assert result['metrics']['downstream']['prediction_precision_at_5px'] is None
    label=_label();label['ignore_polygons']=[[[0,0],[159,0],[159,99],[0,99]]]
    result=evaluator.score_decoded(label,_quality(),_empty(),None,[100,160])
    assert result['metrics']['downstream']['covered_fraction'] is None
    assert result['coverage_support']['downstream']['fraction'] is None


def test_missing_width_and_missing_prediction_rows_are_retained():
    with tempfile.TemporaryDirectory() as folder:
        f=_fixture(folder)
        with patch.object(evaluator,'independent_width',return_value=dict(width_px=None,reason='insufficient_rgb_evidence')):
            with patch.object(evaluator,'load_prediction',return_value=None):
                result=evaluator.evaluate_manifest(f['manifest'],folder,'native_detail')
        assert len(result['rows'])==1 and result['no_rows_dropped']
        row=result['rows'][0]
        assert row['metrics']['downstream']['covered_fraction']==0
        assert row['independent_width']['width_px'] is None and row['prediction_status']=='missing_prediction'
        assert result['split_summaries'][1]['reference_length_px']>0


def test_loaded_prediction_with_missing_width_is_empty_and_still_in_denominator():
    with tempfile.TemporaryDirectory() as folder:
        f=_fixture(folder)
        loaded=(f['meta'],np.ones((50,80),np.float32),np.ones((50,80),np.float32),np.ones((50,80),np.uint8))
        with (patch.object(evaluator,'independent_width',return_value=dict(width_px=None,reason='ambiguous_rgb')),
                patch.object(evaluator,'load_prediction',return_value=loaded)):
            result=evaluator.evaluate_manifest(f['manifest'],folder,'native_detail')
        row=result['rows'][0]
        assert row['prediction_status']=='loaded' and row['decoded']['reason']=='independent_free_width_unavailable'
        assert row['metrics']['downstream']['covered_fraction']==0 and not row['decoded']['segments']


def test_unscored_prediction_never_inflates_aggregate_precision():
    rows=[]
    for unscored in (False,True):
        label=_label()
        if unscored: label['status']='unidentifiable'
        decoded=dict(segments=[[[20,50],[140,50]]] if unscored else [])
        score=evaluator.score_decoded(label,_quality(),decoded,np.ones((50,80),np.uint8),[100,160])
        rows.append(dict(split='holdout',cohort='old_pilot',prediction_status='loaded',
                         independent_width=dict(width_px=10),decoded=decoded,**score))
    result=next(s for s in evaluator.aggregate(rows) if s['split']=='holdout' and s['cohort']=='all' and s['scope']=='downstream')
    assert result['covered_fraction']==0 and result['prediction_precision_at_5px'] is None
    assert result['unscored_reference_samples']==1 and result['unscored_predicted_length_px']==120


def test_width_and_decoder_do_not_receive_reference_quality_or_roi():
    with tempfile.TemporaryDirectory() as folder:
        f=_fixture(folder);calls=[];m=f['meta']
        def width(image,seed):
            assert image.shape==(100,160,3) and seed==f['record']['label']['polylines']['free_seed']
            calls.append(('width',copy.deepcopy(seed)));return dict(width_px=10.,reason='fixture')
        def decoder(body,center,roi,seed,prior):
            assert prior==10 and roi==[0,0,160,100]
            calls.append(('decode',copy.deepcopy(seed)));return _empty('fixture')
        loaded=(m,np.zeros((50,80),np.float32),np.zeros((50,80),np.float32),np.ones((50,80),np.uint8))
        with patch.object(evaluator,'independent_width',side_effect=width),patch.object(evaluator,'decode_heatmaps',side_effect=decoder),patch.object(evaluator,'load_prediction',return_value=loaded):
            result=evaluator.evaluate_manifest(f['manifest'],folder,'native_detail')
        assert [c[0] for c in calls]==['width','decode']
        assert not result['quality_used_for_detector_input']


def test_whole_session_leakage_and_blind_source_rejected_before_image_reads():
    with tempfile.TemporaryDirectory() as folder:
        f=_fixture(folder);sample=f['record']['sample'];other=dict(sample,sample_id='second',split='holdout',annotation_path=str(Path(folder)/'other.json'))
        f['manifest'].write_text(json.dumps(dict(schema_version=1,samples=[sample,other])),'utf-8')
        _raises(lambda:evaluator._dataset(f['manifest']))
        sample['source']['video_path']='dual_20260925_unseen.avi'
        f['manifest'].write_text(json.dumps(dict(schema_version=1,samples=[sample])),'utf-8')
        with patch.object(evaluator,'file_hash',side_effect=AssertionError('Must not read referenced files')):
            _raises(lambda:evaluator._dataset(f['manifest']),'9/25')


def test_old_and_supplement_holdout_are_not_hidden_in_pooled_aggregate():
    rows=[]
    for cohort,segments in (('old_pilot',[]),('supplement',[[[20,50],[140,50]]])):
        result=evaluator.score_decoded(_label(),_quality(),dict(segments=segments),np.ones((50,80),np.uint8),[100,160])
        rows.append(dict(split='holdout',cohort=cohort,prediction_status='loaded',independent_width=dict(width_px=10),
                         decoded=dict(segments=segments),**result))
    summaries=evaluator.aggregate(rows)
    scores={r['cohort']:r for r in summaries if r['split']=='holdout' and r['scope']=='downstream'}
    assert scores['old_pilot']['covered_fraction']==0
    assert scores['supplement']['covered_fraction']==1
    assert scores['all']['covered_fraction']==.5


def test_missing_width_calls_same_frozen_decoder_and_cannot_fabricate_path():
    body=np.ones((50,80),np.float32);center=body.copy()
    result=evaluator.decode_heatmaps(body,center,[0,0,160,100],[[10,50],[20,50]],None)
    assert result['reason']=='independent_free_width_unavailable' and not result['segments']


def test_independent_width_passes_only_rgb_and_free_seed_to_existing_estimator():
    image=np.zeros((100,160,3),np.uint8);seed=[[10,50],[20,50]]
    def detector(actual,**kwargs):
        assert actual is image and kwargs==dict(roi=None,rope_mask=None,seed_points=seed)
        return dict(valid=True,width_px=None,reason='one_edge_only',width_interval_px=None)
    with patch.object(evaluator,'detect_rope',side_effect=detector):
        width=evaluator.independent_width(image,seed)
    assert width['width_px'] is None and not width['target_geometry_used']


def test_protocol_artifact_never_overwrites_prior_file():
    with tempfile.TemporaryDirectory() as folder:
        report=evaluator.write_protocol(folder)
        assert report['protocol']['path_protocol_sha256']==evaluator.protocol_sha256()
        assert set(report['unchanged_dependencies'])==set(evaluator.DEPENDENCY_NAMES)
        assert 'centerline_multiscale_v9.py' in report['unchanged_dependencies']
        assert 'prepare_centerline_v5.py' in report['unchanged_dependencies']
        before=(Path(folder)/'protocol.json').read_bytes()
        try:evaluator.write_protocol(folder)
        except FileExistsError:pass
        else:raise AssertionError('Prior protocol overwritten')
        assert (Path(folder)/'protocol.json').read_bytes()==before


def test_frozen_protocol_rejects_post_registration_source_or_rule_change():
    with tempfile.TemporaryDirectory() as folder:
        evaluator.write_protocol(folder);path=Path(folder)/'protocol.json'
        evaluator.validate_frozen_protocol(path)
        report=json.loads(path.read_text('utf-8'));report['evaluator_source_sha256']='b'*64
        path.write_text(json.dumps(report),'utf-8')
        _raises(lambda:evaluator.validate_frozen_protocol(path),'changed after')


def test_frozen_protocol_rejects_removed_dependency_or_changed_dependency_hash():
    with tempfile.TemporaryDirectory() as folder:
        original=evaluator.write_protocol(folder);path=Path(folder)/'protocol.json'
        report=copy.deepcopy(original);del report['unchanged_dependencies']['centerline_multiscale_v9.py']
        path.write_text(json.dumps(report),'utf-8')
        _raises(lambda:evaluator.validate_frozen_protocol(path),'dependency list')
        report=copy.deepcopy(original);report['unchanged_dependencies']['prepare_centerline_v5.py']='b'*64
        path.write_text(json.dumps(report),'utf-8')
        _raises(lambda:evaluator.validate_frozen_protocol(path),'dependency changed')
