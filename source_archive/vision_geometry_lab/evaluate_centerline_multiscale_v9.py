"""V9 seed-conditioned geometry evaluation; no hardware or blind-set access.

Reuse the unchanged V5 decoder and V6 full/downstream 5px path protocol.
Tile coverage is diagnostic only and NEVER removes reference denominator.
Missing predictions/width retain their sample with an empty predicted path.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import cv2
import numpy as np

from .centerline_benchmark_v5 import decode_heatmaps
from .centerline_path_metrics_v6 import evaluate_paths,protocol_sha256,_sampling
from .centerline_quality_v5 import load_label_quality
from .centerline_review_v5 import load_manifest,validate_annotation,read_clean_image
from .prepare_centerline_v5 import file_hash,check_groups
from .rope_detector import detect_rope
from .centerline_multiscale_contract_v9 import rebuild_input_contract,PIXEL_CENTER_MAPPING


ROOT=Path(__file__).resolve().parents[1]
CONTRACT_SOURCE=Path(__file__).with_name('centerline_multiscale_contract_v9.py').resolve()
MODEL_SOURCE=Path(__file__).with_name('centerline_multiscale_v9.py').resolve()
FORMAT='centerline_multiscale_v9_seed_conditioned_pair'
MODES=('native_detail','lowpass_control')
DEPENDENCY_NAMES=('centerline_benchmark_v5.py','centerline_path_metrics_v6.py',
    'rope_detector.py','centerline_quality_v5.py','centerline_quality_cutoff_v8.py',
    'centerline_review_v5.py','prepare_centerline_v5.py','centerline_model_v5.py',
    'centerline_multiscale_contract_v9.py','centerline_multiscale_v9.py')
V9_PROTOCOL=dict(name='v9_multiscale_fixed_path_evaluation',version=1,
    path_protocol_sha256=protocol_sha256(),decoder='unchanged V5 decode_heatmaps',
    independent_width='original RGB plus human free_seed; detect_rope with no ROI/mask/GT',
    probability_mapping='map_xy=(full_xy+0.5)*scale_xy-0.5',
    coverage_policy='binary supported footprint; zero probabilities outside; never crop reference denominator',
    coverage_diagnostic='V6 <=4px arc-length Voronoi samples; conservative full bilinear support',
    trusted_reference='validated quality.supervision_polylines.centerline when present, else original centerline',
    quality_policy='score reference and prediction only; never width, decoder, input, ROI, or tiles',
    missing_prediction='empty path, zero coverage when a scored reference exists',
    missing_width='empty path, retained row and reference denominator',
    empty_reference='null scores, never a precision success',
    strata=['all_train','all_holdout','old_pilot_train','old_pilot_holdout','supplement_train','supplement_holdout'],
    identity_confirmed=False,physical_precision_validated=False,
    applicability='seed-downstream half-plane is not physical contact; partial references are not exhaustive negatives')


def canonical_hash(value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def independent_width(image_bgr,free_seed):
    """Same free-rope estimator called first by geometry.analyze_frame."""
    result=detect_rope(image_bgr,roi=None,rope_mask=None,seed_points=free_seed)
    width=result.get('width_px')
    valid=width is not None and np.isfinite(width) and width>0
    return dict(width_px=float(width) if valid else None,rope_valid=bool(result.get('valid')),
        reason=result.get('reason','unknown'),width_interval_px=result.get('width_interval_px'),
        source='original RGB + unchanged human free_seed only',target_geometry_used=False)


def scorable_reference(label,quality):
    if label['status'] not in ('clear','partial'):
        return [],'no_precision_reference_for_status'
    trusted=quality.get('supervision_polylines')
    if trusted is not None:
        return trusted['centerline'],'hash_bound_confirmed_trusted_centerline'
    return label['polylines']['centerline'],'original_partial_centerline'


def coverage_support(segments,coverage_mask,image_shape_hw):
    """Length-weighted support audit only; never returns a clipped reference."""
    pieces=_sampling(segments)
    total=sum(piece['length'] for piece in pieces)
    if not total:
        return dict(reference_length_px=0.,supported_reference_length_px=0.,fraction=None)
    supported=0.
    if coverage_mask is not None:
        mask=np.asarray(coverage_mask,np.float32)
        height,width=image_shape_hw
        for piece in pieces:
            grid=(piece['points']+.5)*np.array([mask.shape[1]/width,mask.shape[0]/height])-.5
            inside=(grid[:,0]>=0)&(grid[:,0]<mask.shape[1]-1)&(grid[:,1]>=0)&(grid[:,1]<mask.shape[0]-1)
            values=cv2.remap(mask,grid[:,0].astype(np.float32)[None],grid[:,1].astype(np.float32)[None],
                             cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)[0]
            supported+=float(piece['weights'][inside&(values>=1.-1e-6)].sum())
    return dict(reference_length_px=float(total),supported_reference_length_px=supported,
                fraction=supported/total,does_not_change_score_denominator=True)


def score_decoded(label,quality,decoded,coverage_mask,image_shape_hw):
    reference,source=scorable_reference(label,quality)
    ignored=label['ignore_polygons']+quality['precision_exclude_polygons']
    metrics=evaluate_paths(reference,decoded['segments'],label['polylines']['free_seed'],ignore_polygons=ignored)
    support=metrics['score_support']
    return dict(metrics=metrics,reference_source=source,coverage_support={
        scope:coverage_support(support[scope+'_reference_segments'],coverage_mask,image_shape_hw)
        for scope in ('full','downstream')},quality_used_for_decode=False)


def _dataset(manifest_path):
    manifest=load_manifest(manifest_path)
    # Reject before opening any image/path in a forbidden future source.
    if '20260925' in json.dumps(manifest,ensure_ascii=False):
        raise ValueError('9/25 blind sources must not be read')
    check_groups(manifest['samples'])
    cohorts={}
    inventory_path=manifest.get('inventory_path')
    if inventory_path:
        if file_hash(inventory_path)!=manifest.get('inventory_sha256'):
            raise ValueError('Inventory source hash mismatch')
        inventory=json.loads(Path(inventory_path).read_text('utf-8-sig'))
        cohorts={r['sample_id']:r['source_set'] for r in inventory['samples']}
    records=[]
    for sample in manifest['samples']:
        if sample['sample_id'].startswith(('C03__','C06__')) or sample.get('geometry_role')=='diagnostic_only':
            raise ValueError('Diagnostic-only C03/C06 cannot enter the geometry evaluation')
        height,width=sample['image_shape_hw']
        if sample['roi_xyxy']!=[0,0,width,height] or height%2 or width%2:
            raise ValueError('V9 requires fixed full-image even-sized input, not target-selected ROI')
        if file_hash(sample['image_path'])!=sample['image_sha256']:
            raise ValueError('Source image hash mismatch')
        annotation_hash=file_hash(sample['annotation_path'])
        label=validate_annotation(json.loads(Path(sample['annotation_path']).read_text('utf-8-sig')),sample)
        quality=load_label_quality(sample)
        for key,actual in (('annotation_sha256',annotation_hash),('quality_sha256',quality['quality_sha256'])):
            if key in sample and sample[key]!=actual:
                raise ValueError('Dataset snapshot changed: '+key)
        cohort=cohorts.get(sample['sample_id'],sample.get('source_set'))
        if cohort not in ('old_pilot_12','supplement_8'):
            raise ValueError('Explicit source_set required for old/new cohort accounting')
        records.append(dict(sample=sample,label=label,quality=quality,annotation_sha256=annotation_hash,
                            cohort='old_pilot' if cohort=='old_pilot_12' else 'supplement'))
    train=[r for r in records if r['sample']['split']=='train']
    snapshot=dict(train_group_ids=sorted({r['sample']['group_id'] for r in train}),
        train_annotation_sha256_by_sample={r['sample']['sample_id']:r['annotation_sha256'] for r in train},
        train_quality_sha256_by_sample={r['sample']['sample_id']:r['quality']['quality_sha256'] for r in train})
    return manifest,records,snapshot


def load_prediction(folder,record,image_rgb,training_snapshot,manifest_sha256,expected_mode):
    """Validate provenance and reconstruct RGB+seed inputs without torch/GT."""
    sample=record['sample'];label=record['label'];quality=record['quality']
    path=Path(folder)/(sample['sample_id']+'.json')
    if not path.exists():
        return None
    meta=json.loads(path.read_text('utf-8-sig'))
    required=dict(schema_version=1,format=FORMAT,experiment_version=9,sample_id=sample['sample_id'],
        group_id=sample['group_id'],split=sample['split'],image_sha256=sample['image_sha256'],
        annotation_sha256=record['annotation_sha256'],quality_sha256=quality['quality_sha256'],
        mode=expected_mode,full_image_hw=sample['image_shape_hw'],native_roi_hw=sample['image_shape_hw'],
        roi_xyxy=sample['roi_xyxy'],scale_xy=[.5,.5],pixel_center_mapping=PIXEL_CENTER_MAPPING,
        human_free_seed_used=True,ground_truth_center_or_edges_used_as_input=False,
        quality_used_for_input=False,manifest_sha256=manifest_sha256,
        training_manifest_sha256=manifest_sha256,body_head_supervised=True,
        center_head_supervised=True,real_annotation_trained=True,coverage_restricts_reference_denominator=False)
    for key,value in required.items():
        if key not in meta or meta[key]!=value or (type(value) is bool and type(meta[key]) is not bool):
            raise ValueError('V9 prediction metadata mismatch: '+key)
    if expected_mode not in MODES:
        raise ValueError('Unknown V9 paired mode')
    for key,value in training_snapshot.items():
        if meta.get(key)!=value:
            raise ValueError('Training snapshot mismatch: '+key)
    if sample['split']=='holdout' and sample['group_id'] in meta['train_group_ids']:
        raise ValueError('Holdout session leaked into training')
    for key in ('image_path','annotation_path'):
        if Path(meta.get(key,'')).resolve()!=Path(sample[key]).resolve():
            raise ValueError('Source path mismatch: '+key)
    for name in ('checkpoint','output_npz','contract_source','model_source','training_report'):
        hash_key='output_sha256' if name=='output_npz' else name+'_sha256'
        if not meta.get(name) or not Path(meta[name]).is_absolute() or file_hash(meta[name])!=meta.get(hash_key):
            raise ValueError('Missing or changed source artifact: '+name)
    if Path(meta['contract_source']).resolve()!=CONTRACT_SOURCE or Path(meta['model_source']).resolve()!=MODEL_SOURCE:
        raise ValueError('Unexpected V9 implementation source')
    architecture=meta.get('architecture')
    if (not isinstance(architecture,dict) or architecture.get('name')!='global_descriptor_local_unet'
            or architecture.get('version')!=1 or architecture.get('global_descriptor_channels')!=8):
        raise ValueError('Unknown V9 architecture')
    training=json.loads(Path(meta['training_report']).read_text('utf-8-sig'))
    if (training.get('mode')!=expected_mode or training.get('manifest_sha256')!=manifest_sha256
            or training.get('architecture')!=architecture
            or training.get('artifacts',{}).get('checkpoint_sha256')!=meta['checkpoint_sha256']):
        raise ValueError('Training report/checkpoint/config mismatch')
    if (Path(meta['training_report']).resolve()!=Path(meta['checkpoint']).with_name('report.json').resolve()
            or Path(training['artifacts'].get('checkpoint','')).resolve()!=Path(meta['checkpoint']).resolve()):
        raise ValueError('Training report/checkpoint path aliases disagree')
    for key,value in training_snapshot.items():
        if training.get(key)!=value:
            raise ValueError('Training report snapshot mismatch: '+key)
    contract=rebuild_input_contract(image_rgb,label['polylines']['free_seed'],expected_mode)
    expected=contract['input_contract']
    if meta.get('input_contract')!=expected:
        raise ValueError('RGB/seed reconstructed input contract differs from export')
    for key in ('input_contract_sha256','global_input_sha256','local_inputs_sha256','free_seed_sha256'):
        if meta.get(key)!=expected[key]:
            raise ValueError('Top-level input binding mismatch: '+key)
    if meta.get('tile_layout')!=contract['tile_layout']:
        raise ValueError('Tile selection differs from seed-only contract')
    if meta.get('tile_selection_inputs')!=contract['tile_layout']['selection_inputs']:
        raise ValueError('Unknown tile selection inputs')
    height,width=sample['image_shape_hw'];shape=[height//2,width//2]
    with np.load(meta['output_npz'],allow_pickle=False) as arrays:
        if set(arrays.files)!={'body_probability','center_probability','coverage_mask'}:
            raise ValueError('Unexpected V9 prediction arrays')
        body=arrays['body_probability'].copy();center=arrays['center_probability'].copy()
        coverage=arrays['coverage_mask'].copy()
    if meta.get('probability_map_hw')!=shape or list(body.shape)!=shape or center.shape!=body.shape or coverage.shape!=body.shape:
        raise ValueError('Expected native/2 full-image probability maps')
    if body.dtype!=np.float32 or center.dtype!=np.float32 or coverage.dtype not in (np.dtype('uint8'),np.dtype('bool')):
        raise ValueError('V9 arrays have unexpected dtypes')
    if (not np.isfinite(body).all() or not np.isfinite(center).all()
            or np.any((body<0)|(body>1)) or np.any((center<0)|(center>1))
            or np.any((coverage!=0)&(coverage!=1))):
        raise ValueError('Invalid V9 probabilities/coverage')
    if not np.array_equal(coverage,contract['expected_coverage_mask']):
        raise ValueError('Coverage differs from seed-only fixed windows')
    if np.any(body[coverage==0]!=0) or np.any(center[coverage==0]!=0):
        raise ValueError('Probabilities outside model coverage must be exactly zero')
    return meta,body,center,coverage


def evaluate_manifest(manifest_path,predictions_dir,expected_mode):
    if expected_mode not in MODES:
        raise ValueError('Choose one preregistered V9 mode')
    manifest,records,snapshot=_dataset(manifest_path)
    manifest_hash=file_hash(manifest_path)
    rows=[];loaded_identities=set()
    for record in records:
        sample=record['sample'];label=record['label'];quality=record['quality']
        image_rgb=np.asarray(read_clean_image(sample))
        seed=label['polylines']['free_seed']
        started=time.perf_counter()
        width=independent_width(np.ascontiguousarray(image_rgb[:,:,::-1]),seed)
        width_ms=(time.perf_counter()-started)*1000
        loaded=None;error=None;status='missing_prediction'
        try:
            loaded=load_prediction(predictions_dir,record,image_rgb,snapshot,manifest_hash,expected_mode)
            if loaded is not None:
                status='loaded'
        except (ValueError,KeyError,OSError,TypeError) as exc:
            status='rejected_prediction';error=str(exc)
        decoded=dict(segments=[],widths=[],reason=status,identity_confirmed=False)
        coverage=None;source=None
        if loaded is not None:
            meta,body,center,coverage=loaded
            loaded_identities.add((meta['checkpoint_sha256'],canonical_hash(meta['architecture']),meta['model_source_sha256']))
            decoded=decode_heatmaps(body,center,sample['roi_xyxy'],seed,width['width_px'])
            source={key:meta[key] for key in ('checkpoint_sha256','model_source_sha256','contract_source_sha256',
                'input_contract_sha256','output_sha256','training_report_sha256','mode','architecture')}
        score=score_decoded(label,quality,decoded,coverage,sample['image_shape_hw'])
        rows.append(dict(sample_id=sample['sample_id'],split=sample['split'],group_id=sample['group_id'],
            cohort=record['cohort'],prediction_status=status,prediction_error=error,prediction_source=source,
            image_sha256=sample['image_sha256'],annotation_sha256=record['annotation_sha256'],
            quality_sha256=quality['quality_sha256'],independent_width=width,width_elapsed_ms=width_ms,
            decoded=decoded,**score))
    if len(loaded_identities)>1:
        raise ValueError('Mixed V9 checkpoints/architectures/sources in one evaluation')
    return dict(schema_version=1,experiment_version=9,mode=expected_mode,manifest_sha256=manifest_hash,
        protocol=V9_PROTOCOL,protocol_sha256=canonical_hash(V9_PROTOCOL),path_protocol_sha256=protocol_sha256(),
        rows=rows,split_summaries=aggregate(rows),training_snapshot=snapshot,
        provenance_complete=all(r['prediction_status']=='loaded' for r in rows),
        invalid_prediction_count=sum(r['prediction_status']=='rejected_prediction' for r in rows),
        no_rows_dropped=True,quality_used_for_detector_input=False,coverage_excluded_from_denominator=False,
        identity_confirmed=False,physical_precision_validated=False,hardware_access=False,
        warning='Incomplete/rejected provenance is not a valid model comparison; empty prediction is not success. '
                'Downstream is a seed half-plane, not physical rope contact; partial references are not exhaustive negatives.')


def aggregate(rows):
    groups=[]
    for split in ('train','holdout'):
        for cohort in ('all','old_pilot','supplement'):
            selected=[r for r in rows if r['split']==split and (cohort=='all' or r['cohort']==cohort)]
            for scope in ('full_path','downstream'):
                scores=[r['metrics'][scope] for r in selected]
                scored=[s for s in scores if s['precision_score_available']]
                ref=sum(s['reference_length_px'] for s in scored)
                pred=sum(s['predicted_length_px'] for s in scored)
                covered=sum(s['covered_reference_length_px'] or 0 for s in scored)
                extra=sum(s['extra_predicted_length_px'] or 0 for s in scored)
                normal=[s['normal_error_p95_on_matched_px'] for s in scored if s['normal_error_p95_on_matched_px'] is not None]
                gaps=[s['longest_uncovered_reference_run_px'] for s in scored if s['longest_uncovered_reference_run_px'] is not None]
                cs=[r['coverage_support']['full' if scope=='full_path' else scope] for r in selected]
                support=sum(c['supported_reference_length_px'] for c in cs)
                groups.append(dict(split=split,cohort=cohort,scope=scope,samples=len(selected),
                    scored_reference_samples=sum(s['precision_score_available'] for s in scores),
                    unscored_reference_samples=len(scores)-len(scored),
                    unscored_predicted_length_px=sum(s['predicted_length_px'] for s in scores if not s['precision_score_available']),
                    unavailable_prediction_samples=sum(r['prediction_status']!='loaded' for r in selected),
                    unavailable_width_samples=sum(r['independent_width']['width_px'] is None for r in selected),
                    reference_length_px=ref,predicted_length_px=pred,covered_reference_length_px=covered,
                    covered_fraction=covered/ref if ref else None,
                    prediction_precision_at_5px=1-extra/pred if pred and ref else None,
                    extra_predicted_fraction=extra/pred if pred and ref else None,
                    unsupported_predicted_length_px=extra if ref else None,
                    window_reference_support_fraction=support/ref if ref else None,
                    seed_connected_count=sum(r['metrics']['seed_connection']['connected'] for r in selected),
                    zero_coverage_count=sum(s['covered_fraction']==0 for s in scores),
                    empty_prediction_count=sum(not r['decoded']['segments'] for r in selected),
                    median_sample_matched_normal_p95_px=float(np.median(normal)) if normal else None,
                    samples_with_matched_normal_error=len(normal),
                    median_longest_uncovered_reference_run_px=float(np.median(gaps)) if gaps else None,
                    identity_confirmed=False))
    return groups


def write_protocol(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    report=dict(protocol=V9_PROTOCOL,protocol_sha256=canonical_hash(V9_PROTOCOL),
        evaluator_source_sha256=file_hash(__file__),unchanged_dependencies={
            name:file_hash(Path(__file__).parent/name) for name in DEPENDENCY_NAMES})
    with (output/'protocol.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,ensure_ascii=False,allow_nan=False)
    return report


def validate_frozen_protocol(path):
    report=json.loads(Path(path).read_text('utf-8'))
    if (report.get('protocol')!=V9_PROTOCOL or report.get('protocol_sha256')!=canonical_hash(V9_PROTOCOL)
            or report.get('evaluator_source_sha256')!=file_hash(__file__)):
        raise ValueError('Evaluator/protocol changed after preregistration')
    if set(report.get('unchanged_dependencies',{}))!=set(DEPENDENCY_NAMES):
        raise ValueError('Frozen evaluation dependency list changed or incomplete')
    for name,expected in report['unchanged_dependencies'].items():
        if file_hash(Path(__file__).parent/name)!=expected:
            raise ValueError('Frozen evaluation dependency changed: '+name)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'vision_geometry_lab/outputs/centerline_v9/dataset/manifest.json')
    parser.add_argument('--predictions-dir',type=Path)
    parser.add_argument('--mode',choices=MODES)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--protocol-only',action='store_true')
    parser.add_argument('--frozen-protocol',type=Path,default=ROOT/'vision_geometry_lab/outputs/centerline_v9/path_protocol_v1/protocol.json')
    args=parser.parse_args();cv2.setNumThreads(2)
    if args.output.exists():
        raise FileExistsError('Use a new output directory; prior protocols/results cannot be overwritten')
    if args.protocol_only:
        print(json.dumps(write_protocol(args.output)));return
    if args.predictions_dir is None or args.mode is None:
        parser.error('--predictions-dir and --mode required for evaluation')
    validate_frozen_protocol(args.frozen_protocol)
    report=evaluate_manifest(args.manifest,args.predictions_dir,args.mode)
    report['source_snapshot']=write_protocol(args.output)
    with (args.output/'summary.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps(dict(report=str(args.output/'summary.json'),samples=len(report['rows']),
                         provenance_complete=report['provenance_complete'])))


if __name__=='__main__':
    main()
