"""Provenance preflight for V7, followed by the unchanged V6 path evaluation.

No torch dependency. This checks export/cached-feature provenance, not the
correctness or identity of a predicted rope. Human target curves remain
score-only in the underlying frozen evaluator.
"""
from pathlib import Path
import argparse
import hashlib
import json

import cv2
import numpy as np
from PIL import Image

from .centerline_path_metrics_v6 import (
    evaluate_manifest, load_export_prediction, write_protocol,
)
from .centerline_review_v5 import load_manifest, validate_annotation
from .prepare_centerline_v5 import file_hash


ROOT = Path(__file__).resolve().parents[1]
FACTORY_SHA256 = '7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69'


def validate_transfer_provenance(meta, sample):
    if meta.get('experiment_version') != 7:
        raise ValueError('Expected a V7 transfer experiment export')
    mode = meta.get('feature_mode')
    if mode not in ('sam2_frozen', 'zero_control'):
        raise ValueError('Unknown V7 feature mode')
    provenance = meta.get('feature_provenance', {})
    if provenance.get('backbone_trainable') is not False:
        raise ValueError('Frozen-backbone provenance required')
    if provenance.get('input_semantics') != 'rgb_only_no_seed_no_target':
        raise ValueError('Feature input must be RGB only, not seed/target annotations')
    if (meta.get('trained_checkpoint') != meta.get('checkpoint') or
            meta.get('trained_checkpoint_sha256') != meta.get('checkpoint_sha256')):
        raise ValueError('V7 trained-head checkpoint aliases disagree')
    if mode == 'zero_control':
        if provenance.get('strict_zero_features') is not True:
            raise ValueError('Control requires exact zero features')
        for key in ('cache_npz', 'cache_sha256', 'source_checkpoint', 'source_checkpoint_sha256'):
            if provenance.get(key) is not None:
                raise ValueError('Zero control must not load an image feature cache or SAM weights')
        return dict(sample_id=sample['sample_id'], mode=mode, strict_zero_features=True)
    if provenance.get('strict_zero_features') is not False:
        raise ValueError('Pretrained run must use the frozen nonzero-feature mode')
    if provenance.get('source_checkpoint_sha256') != FACTORY_SHA256:
        raise ValueError('Only the audited factory SAM2 checkpoint is allowed')
    if file_hash(provenance['source_checkpoint']) != FACTORY_SHA256:
        raise ValueError('Factory SAM2 checkpoint file changed')
    cache_path = Path(provenance['cache_npz'])
    if file_hash(cache_path) != provenance['cache_sha256']:
        raise ValueError('Feature cache hash changed')
    with np.load(cache_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {'metadata_json', 'feature_0', 'feature_1', 'feature_2'}:
            raise ValueError('Unexpected feature cache payload')
        cached = json.loads(str(arrays['metadata_json'].item()))
        for key, expected in (
                ('sample_id', sample['sample_id']), ('image_sha256', sample['image_sha256']),
                ('roi_xyxy', sample['roi_xyxy']), ('model_input_hw', meta['model_input_hw']),
                ('cache_key', provenance['cache_key']),
                ('source_checkpoint_sha256', FACTORY_SHA256),
                ('extractor_code_sha256', provenance['extractor_code_sha256']),
                ('input_semantics', 'rgb_only_no_seed_no_target'),
                ('ground_truth_used', False), ('free_seed_used', False)):
            if cached.get(key) != expected:
                raise ValueError(f'Feature cache identity mismatch: {key}')
        height, width = meta['model_input_hw']
        for index, (channels, stride) in enumerate(((32, 4), (64, 8), (256, 16))):
            feature = arrays[f'feature_{index}']
            if feature.shape != (channels, height // stride, width // stride):
                raise ValueError('Feature cache does not span the expected canvas grid')
            if not np.isfinite(feature).all():
                raise ValueError('Non-finite feature cache')
        if cached.get('sam_embedding') != meta.get('sam_embedding'):
            raise ValueError('SAM square embedding/cropping provenance mismatch')
    return dict(sample_id=sample['sample_id'], mode=mode,
                factory_checkpoint_sha256=FACTORY_SHA256,
                cache_sha256=provenance['cache_sha256'], cache_key=provenance['cache_key'])


def preflight(manifest_path, predictions_dir):
    manifest = load_manifest(manifest_path)
    records = []
    for sample in manifest['samples']:
        label = validate_annotation(json.loads(Path(sample['annotation_path']).read_text('utf-8-sig')), sample)
        loaded = load_export_prediction(predictions_dir, sample, label)
        if loaded is None:
            raise ValueError(f"Missing V7 export: {sample['sample_id']}")
        meta = loaded[0]
        record = validate_transfer_provenance(meta, sample)
        report_path = Path(meta['checkpoint']).with_name('report.json')
        training = json.loads(report_path.read_text('utf-8'))
        if (training.get('feature_mode') != meta['feature_mode'] or
                training.get('manifest_sha256') != file_hash(manifest_path) or
                training['artifacts']['checkpoint_sha256'] != meta['checkpoint_sha256']):
            raise ValueError('Training report and inference provenance disagree')
        expected_provenance = training['feature_provenance_by_sample'][sample['sample_id']]
        for key in ('cache_sha256', 'cache_key', 'extractor_code_sha256', 'source_checkpoint_sha256'):
            if expected_provenance.get(key) != meta['feature_provenance'].get(key):
                raise ValueError(f'Training/inference feature provenance differs: {key}')
        if meta['feature_mode'] == 'sam2_frozen':
            with np.load(meta['feature_provenance']['cache_npz'], allow_pickle=False) as arrays:
                cached = json.loads(str(arrays['metadata_json'].item()))
            for key in ('vendor_code_sha256', 'sam_config_sha256'):
                training_key = 'config_sha256' if key == 'sam_config_sha256' else key
                if cached.get(key) != training['sam2'].get(training_key):
                    raise ValueError(f'Cached feature/backbone report mismatch: {key}')
            if cached.get('rgb_canvas_sha256') != exported_rgb_canvas_sha256(meta):
                raise ValueError('Feature cache was not extracted from the exported RGB canvas')
        record['training_report_sha256'] = file_hash(report_path)
        record['initial_head_state_sha256'] = training['head']['initial_head_state_sha256']
        records.append(record)
    if len({record['mode'] for record in records}) != 1:
        raise ValueError('A comparison run must not mix feature modes or be empty')
    return records


def exported_rgb_canvas_sha256(meta):
    """Recreate V6 RGB-only fit/pad from exported native input without torch."""
    with np.load(meta['input_npz'], allow_pickle=False) as arrays:
        rgb = arrays['rgb']
    geometry = meta['letterbox']
    height, width = geometry['model_input_hw']
    left, top, right, bottom = geometry['content_xyxy']
    pixels = np.rint(rgb * 255).astype(np.uint8)
    resized = np.asarray(Image.fromarray(pixels).resize(
        (right-left, bottom-top), resample=Image.Resampling.BILINEAR), dtype=np.float32)/255.
    canvas = np.zeros((height, width, 3), np.float32)
    canvas[top:bottom, left:right] = resized
    return hashlib.sha256(np.ascontiguousarray(canvas.transpose(2, 0, 1)).tobytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'vision_geometry_lab/outputs/centerline_v5/pilot/manifest.json')
    parser.add_argument('--predictions-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Use a new output directory; previous reports are preserved')
    cv2.setNumThreads(2)
    records = preflight(args.manifest, args.predictions_dir)
    report = dict(schema_version=1, experiment_version=7, preflight=records,
                  provenance_preflight_passed=True, wrapper_sha256=file_hash(__file__),
                  manifest_sha256=file_hash(args.manifest), hardware_access=False)
    if not args.preflight_only:
        report['path_evaluation'] = evaluate_manifest(args.manifest, args.predictions_dir)
        report['path_evaluation']['model_family'] = 'v7_frozen_feature_transfer_or_zero_control'
        report['path_evaluation']['geometry_contract'] = 'same native fit-pad and unpadded probability maps as V6'
    report['timing_limit'] = 'Export inference_ms is cached-feature access plus head, not live end-to-end inference.'
    write_protocol(args.output)
    with (args.output/'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(dict(report=str(args.output/'summary.json'), samples=len(records),
                          feature_mode=records[0]['mode'], preflight_only=args.preflight_only)))


if __name__ == '__main__':
    main()
