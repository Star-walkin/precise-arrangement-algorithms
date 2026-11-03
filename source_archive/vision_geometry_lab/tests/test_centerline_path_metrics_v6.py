import copy
import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from vision_geometry_lab.centerline_path_metrics_v6 import (
    PROTOCOL, clip_scoring_segments, evaluate_paths, evaluate_manifest,
    load_export_prediction, load_v6_prediction, protocol_sha256, write_protocol,
)


def test_extra_prediction_is_not_hidden_by_perfect_reference_coverage():
    result = evaluate_paths([[0,0],[100,0]], [[[0,0],[100,0]], [[0,30],[100,30]]], [[-10,0],[0,0]])
    for scope in ('full_path','downstream'):
        metrics = result[scope]
        assert metrics['covered_fraction'] == 1.
        assert metrics['prediction_precision_at_5px'] == .5
        assert metrics['extra_predicted_fraction'] == .5
        assert metrics['extra_predicted_length_px'] == 100.
        assert metrics['longest_supported_prediction_run_px'] == 100.
        assert not metrics['identity_confirmed'] and not metrics['physical_precision_validated']
    assert result['seed_connection']['connected']
    assert result['seed_connection']['connected_prediction_length_fraction'] == .5
    assert result['seed_connection']['disconnected_prediction_length_px'] == 100.


def test_complete_reverse_mirror_preserves_all_scalar_metrics():
    reference = [[-10,0],[20,0],[60,20]]
    prediction = [[[-10,2],[20,2],[60,22]], [[30,50],[50,50]]]
    seed = [[-30,0],[0,0]]
    ignored = [[[28,0],[36,0],[36,35],[28,35]]]
    mirror = lambda points: [[-x,-y] for x,y in points]
    first = evaluate_paths(reference,prediction,seed,ignore_polygons=ignored)
    second = evaluate_paths(mirror(reference),[mirror(p) for p in prediction],mirror(seed),
                            ignore_polygons=[mirror(p) for p in ignored])
    for scope in ('full_path','downstream','seed_connection'):
        for key,value in first[scope].items():
            if value is None or isinstance(value,(bool,str)):
                assert value == second[scope][key]
            elif isinstance(value,(float,int)):
                assert np.isclose(value,second[scope][key],atol=1e-9), (scope,key)


def test_halfplane_crossing_is_interpolated_not_dropped_or_quantized():
    reference = [[-10,-2],[10,2]]
    result = evaluate_paths(reference,[reference],[[-5,0],[0,0]])
    retained = result['score_support']['downstream_reference_segments']
    assert np.allclose(retained,[[[0,0],[10,2]]])
    assert np.isclose(result['downstream']['reference_length_px'],np.hypot(10,2))
    assert result['downstream']['covered_fraction'] == 1.
    assert np.isclose(result['full_path']['reference_length_px'],2*np.hypot(10,2))


def test_empty_prediction_is_zero_coverage_not_zero_error_or_perfect_precision():
    result = evaluate_paths([[0,0],[100,0]],[],[[-10,0],[0,0]])
    metrics = result['downstream']
    assert metrics['covered_fraction'] == 0.
    assert metrics['prediction_precision_at_5px'] is None and metrics['extra_predicted_fraction'] is None
    assert metrics['normal_error_p95_on_matched_px'] is None and metrics['distance_p95_px'] is None
    assert metrics['longest_uncovered_reference_run_px'] == 100.
    assert not result['seed_connection']['connected']
    json.dumps(result,allow_nan=False)


def test_all_ignored_reference_is_unscored_not_success():
    reference = [[0,0],[100,0]]
    result = evaluate_paths(reference,[reference],[[-10,0],[0,0]],
                            ignore_polygons=[[[-1,-10],[101,-10],[101,10],[-1,10]]])
    for scope in ('full_path','downstream'):
        metrics = result[scope]
        assert metrics['status'] == 'no_scored_reference'
        assert not metrics['precision_score_available']
        assert metrics['covered_fraction'] is None and metrics['prediction_precision_at_5px'] is None
        assert metrics['extra_predicted_fraction'] is None
    # Exclusion is only a score mask: it cannot alter the detector's raw path connectivity.
    assert result['seed_connection']['connected']
    json.dumps(result,allow_nan=False)


def test_long_and_short_seed_with_same_endpoint_direction_have_identical_scope():
    reference = [[-20,0],[100,0]]
    long_seed = evaluate_paths(reference,[reference],[[-200,0],[-70,0],[0,0]])
    short_seed = evaluate_paths(reference,[reference],[[-.1,0],[0,0]])
    assert long_seed['downstream'] == short_seed['downstream']
    assert long_seed['score_support'] == short_seed['score_support']
    assert long_seed['downstream_plane'] == short_seed['downstream_plane']


def test_plane_boundary_is_included_and_upstream_only_reference_unscored():
    seed = [[-1,0],[0,0]]
    on_plane = [[0,-10],[0,10]]
    result = evaluate_paths(on_plane,[on_plane],seed)
    assert result['downstream']['reference_length_px'] == 20.
    assert result['downstream']['covered_fraction'] == 1.
    upstream = evaluate_paths([[-20,0],[-1,0]],[[[-20,0],[-1,0]]],seed)
    assert upstream['full_path']['covered_fraction'] == 1.
    assert upstream['downstream']['covered_fraction'] is None


def test_input_gaps_are_not_bridged_and_longest_run_is_not_total_coverage():
    result = evaluate_paths([[0,0],[100,0]],[[[0,0],[20,0]],[[80,0],[100,0]]],[[-10,0],[0,0]])
    metrics = result['downstream']
    assert .45 < metrics['covered_fraction'] < .6
    assert metrics['prediction_precision_at_5px'] == 1.
    assert 44. < metrics['longest_uncovered_reference_run_px'] < 56.
    assert metrics['longest_matched_reference_run_px'] < metrics['covered_reference_length_px']
    assert metrics['longest_supported_prediction_run_px'] == 20.
    assert len(metrics['reference_coverage_gaps']) == 1
    assert result['seed_connection']['connected_prediction_length_fraction'] == .5


def test_ignore_gap_removed_from_denominator_without_joining_runs():
    reference = [[0,0],[100,0]]
    result = evaluate_paths(reference,[reference],[[-10,0],[0,0]],
                            ignore_polygons=[[[40,-5],[60,-5],[60,5],[40,5]]])
    metrics = result['downstream']
    assert metrics['reference_length_px'] == 80. and metrics['predicted_length_px'] == 80.
    assert metrics['reference_segment_count'] == 2 and metrics['predicted_segment_count'] == 2
    assert metrics['covered_fraction'] == 1. and metrics['prediction_precision_at_5px'] == 1.
    assert metrics['longest_matched_reference_run_px'] == 40.
    assert metrics['longest_uncovered_reference_run_px'] == 0.
    assert not metrics['reference_coverage_gaps']


def test_parallel_normal_error_is_conditional_and_exact():
    result = evaluate_paths([[0,0],[100,0]],[[[0,3],[100,3]]],[[-10,0],[0,0]])
    metrics = result['downstream']
    assert metrics['covered_fraction'] == 1.
    assert metrics['normal_error_p95_on_matched_px'] == 3.
    assert metrics['distance_p95_px'] == 3.


def test_turnback_halfplane_splits_support_instead_of_guessing_curve_order():
    path = [[-10,0],[10,0],[-10,20],[10,40]]
    clipped = clip_scoring_segments([path],free_seed=[[-5,0],[0,0]],downstream=True)
    assert len(clipped) == 2
    assert np.allclose(clipped[0],[[0,0],[10,0],[0,10]])
    assert np.allclose(clipped[1],[[0,30],[10,40]])
    assert 'turn-back' in PROTOCOL['applicability']


def test_collinear_ignore_boundary_and_point_contacts_have_zero_scored_length():
    polygon = [[[0,0],[10,0],[10,10],[0,10]]]
    assert clip_scoring_segments([[[0,0],[10,0]]],ignore_polygons=polygon) == []
    clipped = clip_scoring_segments([[[-5,0],[15,0]]],ignore_polygons=polygon)
    assert np.allclose(clipped,[[[-5,0],[0,0]],[[10,0],[15,0]]])
    assert clip_scoring_segments([[[-1,0],[0,0]]],free_seed=[[-2,0],[0,0]],downstream=True) == []


def test_empty_reference_and_invalid_seed_are_explicit():
    result = evaluate_paths([],[[[0,0],[100,0]]],[[-1,0],[0,0]])
    assert result['downstream']['covered_fraction'] is None
    assert result['downstream']['prediction_precision_at_5px'] is None
    for seed in ([],[[0,0]],[[0,0],[0,0]],[[0,0],[1,0],[0,0]]):
        try: evaluate_paths([[0,0],[1,0]],[],seed)
        except ValueError: pass
        else: raise AssertionError('Invalid seed accepted')


def test_quality_scoring_does_not_modify_any_source_coordinates():
    reference = [[-10,0],[100,0]]
    prediction = [[[-10,0],[100,0]]]
    seed = [[-10,0],[0,0]]
    polygons = [[[40,-5],[60,-5],[60,5],[40,5]]]
    before = copy.deepcopy((reference,prediction,seed,polygons))
    result = evaluate_paths(reference,prediction,seed,ignore_polygons=polygons)
    assert (reference,prediction,seed,polygons) == before
    assert result['quality_used_for_prediction_input'] is False
    assert result['protocol_sha256'] == protocol_sha256()


def test_protocol_artifact_is_fixed_and_cannot_overwrite_previous_run():
    with tempfile.TemporaryDirectory() as directory:
        path = write_protocol(Path(directory)/'new')
        content = json.loads(path.read_text('utf-8'))
        assert content['protocol'] == PROTOCOL
        assert content['protocol_sha256'] == protocol_sha256()
        try: write_protocol(path.parent)
        except FileExistsError: pass
        else: raise AssertionError('Previous protocol artifact overwritten')


def _export_fixture(folder):
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    image = folder/'clean.png'
    Image.fromarray(np.full((20,120,3),80,np.uint8)).save(image)
    label = dict(schema_version=1,sample_id='sample',image_sha256=digest(image),
        coordinate_system='full_image_xy',status='partial',reviewed=True,
        polylines=dict(centerline=[[0.,10.],[100.,10.]],edge_flange=[[0.,7.],[100.,7.]],
                       edge_surface=[[0.,13.],[100.,13.]],free_seed=[[1.,10.],[20.,10.]]),
        ignore_polygons=[],notes='not real data')
    annotation = folder/'label.json'
    annotation.write_text(json.dumps(label),encoding='utf-8')
    sample = dict(sample_id='sample',group_id='fixture_holdout',split='holdout',
        image_path=str(image.resolve()),image_sha256=digest(image),image_shape_hw=[20,120],
        annotation_path=str(annotation.resolve()),roi_xyxy=[0,0,120,20],
        source=dict(video_path='fixture.avi',camera='fixture',frame_index=0,time_s=0.))
    quality = dict(schema_version=1,sample_id='sample',image_sha256=sample['image_sha256'],
        annotation_sha256=digest(annotation),coordinate_system='full_image_xy',
        regions=[dict(region_id='uncertain',kind='coarse_direction_only',exclude_from_precision=True,
                      polygon_xy=[[60,5],[80,5],[80,15],[60,15]])])
    sidecar = annotation.with_suffix('.quality.json')
    sidecar.write_text(json.dumps(quality),encoding='utf-8')
    manifest = folder/'manifest.json'
    manifest.write_text(json.dumps(dict(schema_version=1,samples=[sample])),encoding='utf-8')
    inputs = folder/'input.npz'
    np.savez(inputs,rgb=np.full((20,120,3),80/255),seed_heatmap=np.zeros((20,120)))
    output = folder/'output.npz'
    np.savez(output,body_probability=np.zeros((11,64)),center_probability=np.zeros((11,64)))
    checkpoint = folder/'model.pt'; checkpoint.write_bytes(b'unit fixture only')
    geometry = dict(native_hw=[20,120],model_input_hw=[32,64],content_hw=[11,64],
                    content_xyxy=[0,10,64,21],padding_ltrb=[0,10,0,11],scale_xy=[64/120,11/20])
    seed_payload = json.dumps(label['polylines']['free_seed'],sort_keys=True,separators=(',',':')).encode('utf-8')
    meta = dict(sample_id='sample',image_sha256=sample['image_sha256'],train_group_ids=['training'],
        real_annotation_trained=True,ground_truth_center_or_edges_used_as_input=False,
        raster_encoding_version=2,preprocess_version=1,body_head_supervised=True,center_head_supervised=True,
        human_free_seed_used=True,roi_xyxy=sample['roi_xyxy'],annotation_sha256=digest(annotation),
        free_seed_sha256=hashlib.sha256(seed_payload).hexdigest(),quality_sha256=digest(sidecar),
        input_npz=str(inputs),input_sha256=digest(inputs),output_npz=str(output),output_sha256=digest(output),
        checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),native_roi_hw=[20,120],
        probability_map_hw=[11,64],model_input_hw=[32,64],scale_xy=geometry['scale_xy'],letterbox=geometry,
        architecture='fixture',inference_ms=1.)
    metadata = folder/'sample.json'
    metadata.write_text(json.dumps(meta),encoding='utf-8')
    return manifest,sample,label,meta,metadata


def test_v6_export_mapping_accepts_unpadded_content_and_rejects_bad_geometry():
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        _,sample,label,meta,path = _export_fixture(folder)
        assert load_v6_prediction(folder,sample,label)[1].shape == (11,64)
        for alteration in ('padding','scale','version','content'):
            broken = copy.deepcopy(meta)
            if alteration == 'padding': broken['letterbox']['padding_ltrb'][1] = 9
            if alteration == 'scale': broken['scale_xy'] = [1.,1.]
            if alteration == 'version': broken['preprocess_version'] = 0
            if alteration == 'content': broken['letterbox']['content_hw'] = [32,64]
            path.write_text(json.dumps(broken),encoding='utf-8')
            try: load_v6_prediction(folder,sample,label)
            except ValueError: pass
            else: raise AssertionError('Wrong letterbox metadata accepted')


def test_explicit_v5_dispatch_needs_no_fake_letterbox_fields():
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        _,sample,label,meta,path = _export_fixture(folder)
        for key in ('preprocess_version','letterbox','architecture'):
            meta.pop(key)
        output = Path(meta['output_npz'])
        np.savez(output,body_probability=np.zeros((32,64)),center_probability=np.zeros((32,64)))
        meta.update(probability_map_hw=[32,64],scale_xy=[64/120,32/20],
                    output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    probability_arrays={key:'model-resolution HxW float32 spanning native ROI'
                                        for key in ('body_probability','center_probability')})
        path.write_text(json.dumps(meta),encoding='utf-8')
        loaded = load_export_prediction(folder,sample,label)
        assert loaded[0]['validated_prediction_format'] == 'v5_direct_roi_resize'
        assert 'letterbox' not in loaded[0] and 'preprocess_version' not in loaded[0]
        meta['scale_xy'] = [1.,1.]
        path.write_text(json.dumps(meta),encoding='utf-8')
        try: load_export_prediction(folder,sample,label)
        except ValueError as error: assert 'scale mismatch' in str(error)
        else: raise AssertionError('Incorrect V5 direct resize scale accepted')


def test_v6_cannot_downgrade_by_omitting_its_letterbox_fields():
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        _,sample,label,meta,path = _export_fixture(folder)
        meta.pop('preprocess_version');meta.pop('letterbox')
        path.write_text(json.dumps(meta),encoding='utf-8')
        try: load_export_prediction(folder,sample,label)
        except ValueError as error: assert 'metadata may be missing' in str(error)
        else: raise AssertionError('Corrupted V6 export silently treated as V5')


def test_manifest_runner_keeps_quality_and_reference_out_of_detector_inputs():
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        manifest,sample,label,_,_ = _export_fixture(folder)
        candidate = [[10.,10.],[90.,10.]]
        calls = []
        def analyze(frame,*,seed_points):
            calls.append('analyze')
            assert frame.shape == (20,120,3) and np.all(frame == 80)
            assert seed_points == label['polylines']['free_seed']
            return dict(wound_curve=dict(candidates=[dict(observed_segments=[candidate])]),rope=dict(width_px=6.))
        def decode(body,center,roi,seed,width):
            calls.append('decode')
            assert body.shape == (11,64) and np.all(body == 0) and np.all(center == 0)
            assert roi == sample['roi_xyxy'] and seed == label['polylines']['free_seed'] and width == 6.
            return dict(segments=[candidate],widths=[6.,6.],reason='fixture')
        def refine(frame,points,*,expected_width_px):
            calls.append('refine')
            assert np.all(frame == 80) and points == candidate
            assert points != label['polylines']['centerline'] and expected_width_px == [6.,6.]
            return dict(records=[dict(valid=True,refined_xy=point) for point in points])
        with patch('vision_geometry_lab.geometry.analyze_frame',side_effect=analyze), \
             patch('vision_geometry_lab.centerline_benchmark_v5.decode_heatmaps',side_effect=decode), \
             patch('vision_geometry_lab.centerline_refine_v5.refine_centerline',side_effect=refine):
            report = evaluate_manifest(manifest,folder)
        assert calls == ['analyze','decode','refine']
        row = report['rows'][0]
        for method in ('cv_only','model_only','model_plus_cv'):
            metrics = row['methods'][method]['metrics']
            assert metrics['full_path']['reference_length_px'] == 80.
            assert metrics['downstream']['reference_length_px'] == 60.
            assert metrics['v5_full_path_control']['status'] == 'scored_manual_reference'
            assert metrics['v5_full_path_control']['reference_samples'] > 0
        assert not row['quality_used_for_prediction_input']
        assert not report['field_accuracy_established']
        json.dumps(report,allow_nan=False)


def test_manifest_rejected_prediction_is_not_silent_empty_success():
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        manifest,_,_,meta,path = _export_fixture(folder)
        meta['quality_sha256'] = 'f'*64
        path.write_text(json.dumps(meta),encoding='utf-8')
        fake = dict(wound_curve=dict(candidates=[]),rope=dict(width_px=6.))
        with patch('vision_geometry_lab.geometry.analyze_frame',return_value=fake):
            report = evaluate_manifest(manifest,folder)
        methods = report['rows'][0]['methods']
        assert methods['model_only']['status'] == 'rejected_prediction'
        assert 'metrics' not in methods['model_only']
        summary = next(s for s in report['split_summaries'] if s['split']=='holdout'
                       and s['method']=='model_only' and s['scope']=='downstream')
        assert summary['unavailable_method_samples'] == 1
        assert summary['covered_fraction'] is None and summary['prediction_precision_at_5px'] is None
