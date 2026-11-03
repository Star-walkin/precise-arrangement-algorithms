import json
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import vision_geometry_lab.centerline_benchmark_v5 as benchmark
from vision_geometry_lab.centerline_benchmark_v5 import (
    decode_heatmaps, nearest_on_segments, refinement_segments, score_centerline,
    load_prediction)
from vision_geometry_lab.prepare_centerline_v5 import check_groups, file_hash


def test_group_split_cannot_share_one_recording():
    samples=[dict(sample_id='a',group_id='g',split='train'),
             dict(sample_id='b',group_id='g',split='holdout')]
    try: check_groups(samples)
    except ValueError: pass
    else: raise AssertionError('Recording leakage allowed')


def test_missing_center_prediction_is_not_zero_error():
    score=score_centerline([[0,0],[100,0]],[])
    assert score['covered_fraction']==0
    assert score['distance_p95_px'] is None
    assert score['normal_error_p95_on_matched_px'] is None
    json.dumps(score,allow_nan=False)


def test_parallel_normal_error_and_partial_coverage_report_together():
    score=score_centerline([[0,0],[100,0]],[[[0,3],[50,3]]])
    assert .45<score['covered_fraction']<.6
    assert abs(score['normal_error_p95_on_matched_px']-3)<1e-6
    assert score['distance_p95_px']>40
    assert score['identity_jump_rate'] is None


def test_ignored_reference_does_not_count_as_success():
    s=score_centerline([[5,5],[15,5]],[],ignore_polygons=[[[0,0],[20,0],[20,10],[0,10]]])
    assert s['reference_samples']==0 and s['covered_fraction'] is None
    assert s['reference_samples_before_exclusion']==s['excluded_reference_samples']
    assert not s['precision_score_available']
    assert not s['physical_precision_validated']


def test_precision_exclusion_removes_uncertain_tail_from_coverage_denominator():
    reference=[[0,10],[100,10]]
    prediction=[[[0,10],[40,10]]]
    original=score_centerline(reference,prediction)
    excluded=score_centerline(reference,prediction,
        ignore_polygons=[[[48,5],[110,5],[110,15],[48,15]]])
    assert original['reference_samples']==26 and original['covered_fraction']<.5
    assert excluded['reference_samples_before_exclusion']==26
    assert excluded['reference_samples']==12 and excluded['excluded_reference_samples']==14
    assert excluded['covered_fraction']==1.
    assert excluded['precision_score_available']
    assert not excluded['physical_precision_validated']


def test_nearest_geometry_does_not_bridge_distinct_segments():
    d,_=nearest_on_segments(np.array([[50.,0.]]),[[[0,0],[20,0]],[[80,0],[100,0]]])
    assert d[0]==30
    r=dict(records=[dict(valid=True,refined_xy=[0,0]),dict(valid=True,refined_xy=[10,0]),
                    dict(valid=False,refined_xy=None),dict(valid=True,refined_xy=[30,0]),
                    dict(valid=True,refined_xy=[40,0])])
    assert len(refinement_segments(r))==2


def test_decoder_seed_anchored_candidate_and_blank_abstention():
    yy,xx=np.mgrid[:120,:200]
    center=np.exp(-.5*((yy-60)/1.7)**2).astype(np.float32)
    body=(np.abs(yy-60)<=7).astype(np.float32)
    result=decode_heatmaps(body,center,[0,0,200,120],[[10,60],[30,60]],14.)
    assert len(result['segments'])==1 and len(result['segments'][0])>10
    assert max(abs(p[1]-60) for p in result['segments'][0])<1e-6
    assert not result['identity_confirmed']
    blank=np.zeros_like(body)
    assert not decode_heatmaps(blank,blank,[0,0,200,120],[[10,60],[30,60]],14.)['segments']
    assert not decode_heatmaps(body,center,[0,0,200,120],[[10,60],[30,60]],None)['segments']


def test_decoder_rejects_broad_surface_mask_even_with_center_ridge():
    yy,xx=np.mgrid[:120,:200]
    center=np.exp(-.5*((yy-60)/1.7)**2).astype(np.float32)
    assert not decode_heatmaps(np.ones_like(center),center,[0,0,200,120],[[10,60],[30,60]],14.)['segments']


def test_decoder_roi_border_is_not_a_rope_body_edge():
    yy,xx=np.mgrid[:20,:200]
    center=np.exp(-.5*((yy-10)/1.7)**2).astype(np.float32)
    assert not decode_heatmaps(np.ones_like(center),center,[0,0,200,20],[[10,10],[30,10]],14.)['segments']


def test_decoder_nonzero_roi_translation_preserves_native_pixel_centers():
    yy,xx=np.mgrid[:120,:200]
    center=np.exp(-.5*((yy-60)/1.7)**2).astype(np.float32)
    body=(np.abs(yy-60)<=7).astype(np.float32)
    result=decode_heatmaps(body,center,[100,200,300,320],[[110,260],[130,260]],14.)
    assert result['segments'] and max(abs(p[1]-260) for p in result['segments'][0])<1e-6


def test_field_benchmark_rejects_synthetic_and_holdout_leakage():
    with tempfile.TemporaryDirectory() as directory:
        sample=dict(sample_id='sample',image_sha256='a'*64,split='holdout',group_id='test')
        meta=dict(sample_id='sample',image_sha256='a'*64,
                  ground_truth_center_or_edges_used_as_input=False,
                  train_group_ids=[],real_annotation_trained=False)
        path=Path(directory)/'sample.json'
        path.write_text(json.dumps(meta),encoding='utf-8')
        try: load_prediction(directory,sample)
        except ValueError as error: assert 'Synthetic' in str(error)
        else: raise AssertionError('Synthetic checkpoint promoted to field model')
        meta.update(real_annotation_trained=True,train_group_ids=['test'])
        path.write_text(json.dumps(meta),encoding='utf-8')
        try: load_prediction(directory,sample)
        except ValueError as error: assert 'Holdout' in str(error)
        else: raise AssertionError('Holdout recording leakage allowed')


def test_load_batch_prediction_metadata_and_reject_stale_labels():
    with tempfile.TemporaryDirectory() as directory:
        folder=Path(directory)
        annotation=folder/'label.json'
        label=dict(polylines=dict(free_seed=[[1.,2.],[3.,2.]]))
        annotation.write_text(json.dumps(label),encoding='utf-8')
        checkpoint=folder/'fixture.pt';checkpoint.write_bytes(b'not-a-real-model-unit-fixture')
        inputs=folder/'input.npz';np.savez(inputs,rgb=np.zeros((20,30,3)),seed_heatmap=np.zeros((20,30)))
        output=folder/'sample.npz';np.savez(output,body_probability=np.zeros((10,15)),center_probability=np.zeros((10,15)))
        sample=dict(sample_id='sample',image_sha256='a'*64,group_id='g2',split='holdout',
                    annotation_path=str(annotation),roi_xyxy=[0,0,30,20])
        payload=json.dumps(label['polylines']['free_seed'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')
        meta=dict(sample_id='sample',image_sha256='a'*64,train_group_ids=['g1'],
                  raster_encoding_version=2,
                  ground_truth_center_or_edges_used_as_input=False,real_annotation_trained=True,
                  body_head_supervised=True,center_head_supervised=True,human_free_seed_used=True,
                  roi_xyxy=[0,0,30,20],annotation_sha256=file_hash(annotation),
                  free_seed_sha256=hashlib.sha256(payload).hexdigest(),
                  output_npz=str(output),output_sha256=file_hash(output),
                  input_npz=str(inputs),input_sha256=file_hash(inputs),
                  checkpoint=str(checkpoint),checkpoint_sha256=file_hash(checkpoint),
                  probability_map_hw=[10,15],native_roi_hw=[20,30])
        (folder/'sample.json').write_text(json.dumps(meta),encoding='utf-8')
        loaded=load_prediction(folder,sample,label)
        assert loaded[1].shape==(10,15)
        for old_version in (None,1,'2'):
            incompatible=dict(meta)
            if old_version is None:
                incompatible.pop('raster_encoding_version')
            else:
                incompatible['raster_encoding_version']=old_version
            (folder/'sample.json').write_text(json.dumps(incompatible),encoding='utf-8')
            try: load_prediction(folder,sample,label)
            except ValueError as error: assert 'raster_encoding_version' in str(error)
            else: raise AssertionError('Incompatible probability encoding accepted')
        (folder/'sample.json').write_text(json.dumps(meta),encoding='utf-8')
        annotation.write_text(json.dumps(dict(label,notes='changed')),encoding='utf-8')
        try: load_prediction(folder,sample,label)
        except ValueError as error: assert 'Annotation changed' in str(error)
        else: raise AssertionError('Stale annotations not detected')


def _quality_evaluation_fixture(folder):
    """Tiny independent PNG/label/model-output fixture; never field files."""
    image=folder/'clean.png'
    Image.fromarray(np.full((20,120,3),80,np.uint8)).save(image)
    annotation=folder/'label.json'
    label=dict(schema_version=1,sample_id='quality_case',image_sha256=file_hash(image),
        coordinate_system='full_image_xy',status='partial',reviewed=True,
        polylines=dict(centerline=[[0,10],[100,10]],edge_flange=[[0,7],[100,7]],
                       edge_surface=[[0,13],[100,13]],free_seed=[[1.,10.],[9.,10.]]),
        ignore_polygons=[],notes='synthetic quality exclusion fixture')
    annotation.write_text(json.dumps(label),encoding='utf-8')
    sample=dict(sample_id='quality_case',group_id='fixture_holdout',split='holdout',
        image_path=str(image.resolve()),image_sha256=file_hash(image),image_shape_hw=[20,120],
        annotation_path=str(annotation.resolve()),roi_xyxy=[0,0,120,20],
        source=dict(video_path='fixture.avi',frame_index=0,time_s=0.,camera='fixture'))
    manifest=folder/'manifest.json'
    manifest.write_text(json.dumps(dict(schema_version=1,samples=[sample])),encoding='utf-8')
    checkpoint=folder/'fixture.pt';checkpoint.write_bytes(b'not-a-real-model-unit-fixture')
    inputs=folder/'input.npz'
    np.savez(inputs,rgb=np.full((20,120,3),80/255),seed_heatmap=np.zeros((20,120)))
    output=folder/'output.npz'
    np.savez(output,body_probability=np.zeros((10,60)),center_probability=np.zeros((10,60)))
    seed_payload=json.dumps(label['polylines']['free_seed'],sort_keys=True,separators=(',',':'),
                            ensure_ascii=False).encode('utf-8')
    meta=dict(sample_id=sample['sample_id'],image_sha256=sample['image_sha256'],
        train_group_ids=['fixture_training'],raster_encoding_version=2,
        ground_truth_center_or_edges_used_as_input=False,real_annotation_trained=True,
        body_head_supervised=True,center_head_supervised=True,human_free_seed_used=True,
        roi_xyxy=sample['roi_xyxy'],annotation_sha256=file_hash(annotation),
        free_seed_sha256=hashlib.sha256(seed_payload).hexdigest(),
        output_npz=str(output),output_sha256=file_hash(output),
        input_npz=str(inputs),input_sha256=file_hash(inputs),
        checkpoint=str(checkpoint),checkpoint_sha256=file_hash(checkpoint),
        probability_map_hw=[10,60],native_roi_hw=[20,120],inference_ms=1.)
    metadata=folder/'quality_case.json'
    metadata.write_text(json.dumps(meta),encoding='utf-8')
    return manifest,sample,label,meta,metadata


def _write_fixture_quality(sample,polygon):
    sidecar=Path(sample['annotation_path']).with_suffix('.quality.json')
    content=dict(schema_version=1,sample_id=sample['sample_id'],
        image_sha256=sample['image_sha256'],annotation_sha256=file_hash(sample['annotation_path']),
        coordinate_system='full_image_xy',regions=[dict(region_id='uncertain_tail',
            kind='coarse_direction_only',exclude_from_precision=True,polygon_xy=polygon,
            reason='synthetic quality exclusion fixture')])
    sidecar.write_text(json.dumps(content),encoding='utf-8')
    return sidecar,benchmark.load_label_quality(sample)


def test_prediction_rejects_missing_stale_and_removed_quality_revision():
    with tempfile.TemporaryDirectory() as directory:
        folder=Path(directory)
        _,sample,label,meta,metadata=_quality_evaluation_fixture(folder)
        sidecar,quality=_write_fixture_quality(sample,[[48,5],[110,5],[110,15],[48,15]])
        for revision in (None,'c'*64):
            stale=dict(meta,quality_sha256=revision)
            metadata.write_text(json.dumps(stale),encoding='utf-8')
            try: load_prediction(folder,sample,label)
            except ValueError as error: assert 'quality changed' in str(error)
            else: raise AssertionError('Stale quality revision accepted')
        current=dict(meta,quality_sha256=quality['quality_sha256'])
        metadata.write_text(json.dumps(current),encoding='utf-8')
        assert load_prediction(folder,sample,label)[1].shape==(10,60)
        # No sidecar now exists: a prediction bound to a removed quality file
        # must also fail, rather than silently restore previously excluded truth.
        sidecar.unlink()
        try: load_prediction(folder,sample,label)
        except ValueError as error: assert 'quality changed' in str(error)
        else: raise AssertionError('Removed quality revision not detected')


def test_evaluation_quality_is_scoring_only_not_cv_decoder_or_refinement_input():
    with tempfile.TemporaryDirectory() as directory:
        folder=Path(directory)
        manifest,sample,label,meta,metadata=_quality_evaluation_fixture(folder)
        polygon=[[48,5],[110,5],[110,15],[48,15]]
        _,quality=_write_fixture_quality(sample,polygon)
        metadata.write_text(json.dumps(dict(meta,quality_sha256=quality['quality_sha256'])),
                            encoding='utf-8')
        candidate=[[0.,10.],[40.,10.]]
        calls=[]
        def analyze(frame,*,seed_points):
            calls.append('cv')
            assert frame.shape==(20,120,3) and np.all(frame==80)
            assert seed_points==label['polylines']['free_seed']
            return dict(wound_curve=dict(candidates=[dict(observed_segments=[candidate])]),
                        rope=dict(width_px=6.))
        def decode(body,center,roi,free_seed,width_prior):
            calls.append('decode')
            assert np.array_equal(body,np.zeros((10,60))) and np.array_equal(center,body)
            assert roi==sample['roi_xyxy'] and free_seed==label['polylines']['free_seed']
            assert width_prior==6.
            return dict(segments=[candidate],widths=[6.,6.],reason='fixture')
        def refine(frame,seed_points_xy,*,expected_width_px):
            calls.append('refine')
            assert frame.shape==(20,120,3) and np.all(frame==80)
            assert seed_points_xy==candidate and seed_points_xy!=label['polylines']['centerline']
            assert expected_width_px==[6.,6.]
            return dict(records=[dict(valid=True,refined_xy=p) for p in seed_points_xy])
        with patch.object(benchmark,'analyze_frame',side_effect=analyze), \
             patch.object(benchmark,'decode_heatmaps',side_effect=decode), \
             patch.object(benchmark,'refine_centerline',side_effect=refine):
            result=benchmark.evaluate(manifest,folder)
        assert calls==['cv','decode','refine']
        row=result['rows'][0]
        assert row['quality_sha256']==quality['quality_sha256']
        assert row['precision_exclusion']['quality_exclude_polygons']==[polygon]
        assert row['precision_exclusion']['used_only_for_scoring']
        assert row['precision_exclusion']['no_score_is_not_success']
        for method in ('cv_only','model_only','model_plus_cv'):
            metrics=row['methods'][method]['metrics']
            assert metrics['reference_samples']==12 and metrics['excluded_reference_samples']==14
            assert metrics['covered_fraction']==1.
            assert not metrics['physical_precision_validated']
        assert not result['field_accuracy_established']
        json.dumps(result,allow_nan=False)


def test_evaluation_wholly_uncertain_reference_has_no_precision_success():
    with tempfile.TemporaryDirectory() as directory:
        folder=Path(directory)
        manifest,sample,label,_,_=_quality_evaluation_fixture(folder)
        _write_fixture_quality(sample,[[0,0],[110,0],[110,19],[0,19]])
        candidate=label['polylines']['centerline']
        prediction=dict(wound_curve=dict(candidates=[dict(observed_segments=[candidate])]),
                        rope=dict(width_px=6.))
        with patch.object(benchmark,'analyze_frame',return_value=prediction):
            result=benchmark.evaluate(manifest)
        metrics=result['rows'][0]['methods']['cv_only']['metrics']
        assert metrics['status']=='all_reference_ignored'
        assert metrics['reference_samples']==0 and metrics['excluded_reference_samples']==26
        assert metrics['covered_fraction'] is None and metrics['distance_p95_px'] is None
        assert metrics['normal_error_p95_on_matched_px'] is None
        assert not metrics['precision_score_available'] and not result['field_accuracy_established']
        json.dumps(result,allow_nan=False)
