"""Single-train GT-target oracle diagnostic, deliberately NOT model prediction.

This isolates target rasterization/fixed decoder compatibility. Unknown target
pixels are zero evidence, not relabeled negative training examples. No source,
loss, training protocol, threshold, or evaluation rule is changed.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'vision_geometry_lab/outputs/v10_single_fit'


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def build(output_dir):
    import torch
    from .v10_single_fit import verify_protocol
    from .centerline_multiscale_v9 import prepare_record, _stitch_tiles, _write_prediction
    output_dir = Path(output_dir).resolve()
    oracle_dir = output_dir / 'target_oracle'
    if oracle_dir.exists():
        raise FileExistsError('oracle artifacts may not be overwritten')
    oracle_dir.mkdir()
    protocol, record = verify_protocol(output_dir / 'protocol.json')
    torch.set_num_threads(2)
    prepared = prepare_record(record, 'native_detail', include_supervision=True)
    predictions, tile_statistics = [], []
    for index, tile in enumerate(prepared['supervision']):
        body = (tile['body_target'] * tile['body_valid_mask'])[0].numpy()
        center = (tile['center_target'] * tile['center_valid_mask'])[0].numpy()
        predictions.append((body, center))
        valid = tile['center_valid_mask'][0].numpy() > 0
        known = center[valid]
        tile_statistics.append(dict(tile=index, valid_pixels=int(valid.sum()),
            known_center_max=float(known.max()) if len(known) else None,
            known_center_pixels_ge_half=int(np.sum(known >= .5)),
            soft_positive_mass=float(known.sum())))
    body, center, coverage = _stitch_tiles(predictions, prepared['tile_layout'], record['image_shape_hw'])
    path = _write_prediction(oracle_dir / 'gt_target_oracle.npz', body, center, coverage)
    write_json(oracle_dir / 'metadata.json', dict(schema_version=1,
        diagnostic='GROUND TRUTH target oracle, NOT A MODEL PREDICTION',
        sample_id=record['sample_id'], train_only=True, mode='native_detail',
        gt_target_used_as_decoder_input=True, no_model_weights_used=True,
        unknown_policy='per-head invalid pixels are zero evidence; scoring keeps exclusions',
        protocol_sha256=file_hash(output_dir / 'protocol.json'),
        source_sha256=file_hash(__file__),
        prediction=str(path), prediction_sha256=file_hash(path),
        tile_statistics=tile_statistics, stitched_center_max=float(center.max()),
        sigma_native_px=2, model_scale=.5, threshold_changed=False,
        hardware_access=False, generalization_claim=False))
    print(json.dumps(dict(event='gt_target_oracle_built', center_max=float(center.max()),
                         metadata=str(oracle_dir / 'metadata.json'))), flush=True)


def score(output_dir):
    import cv2
    from .centerline_benchmark_v5 import decode_heatmaps
    from .centerline_quality_v5 import load_label_quality
    from .centerline_review_v5 import read_clean_image, validate_annotation
    from .evaluate_centerline_multiscale_v9 import independent_width, score_decoded
    output_dir = Path(output_dir).resolve()
    oracle_dir = output_dir / 'target_oracle'
    protocol = json.loads((output_dir / 'protocol.json').read_text('utf-8'))
    meta = json.loads((oracle_dir / 'metadata.json').read_text('utf-8'))
    if meta['source_sha256'] != file_hash(__file__) or meta['protocol_sha256'] != file_hash(output_dir / 'protocol.json'):
        raise ValueError('oracle or train protocol changed')
    for path, expected in protocol['dependencies_sha256'].items():
        if file_hash(path) != expected:
            raise ValueError('frozen dependency changed')
    manifest = json.loads(Path(protocol['manifest']).read_text('utf-8'))
    if file_hash(protocol['manifest']) != protocol['manifest_sha256'] or len(manifest['samples']) != 1:
        raise ValueError('single train manifest mismatch')
    sample = manifest['samples'][0]
    if sample['sample_id'] != 'dual_20260924_111500__left__f00246' or sample['split'] != 'train':
        raise ValueError('oracle only permits the fixed train sample')
    if file_hash(sample['annotation_path']) != protocol['annotation_sha256']:
        raise ValueError('label changed')
    label = validate_annotation(json.loads(Path(sample['annotation_path']).read_text('utf-8-sig')), sample)
    quality = load_label_quality(sample)
    if quality['quality_sha256'] != protocol['quality_sha256']:
        raise ValueError('quality changed')
    image = cv2.cvtColor(np.asarray(read_clean_image(sample)), cv2.COLOR_RGB2BGR)
    width = independent_width(image, label['polylines']['free_seed'])
    if file_hash(meta['prediction']) != meta['prediction_sha256']:
        raise ValueError('oracle target changed')
    with np.load(meta['prediction'], allow_pickle=False) as data:
        body, center, coverage = (data[key] for key in ('body_probability', 'center_probability', 'coverage_mask'))
    decoded = decode_heatmaps(body, center, sample['roi_xyxy'], label['polylines']['free_seed'], width['width_px'])
    scored = score_decoded(label, quality, decoded, coverage, sample['image_shape_hw'])
    result = dict(diagnostic=meta['diagnostic'], gt_target_used_as_decoder_input=True,
        width_from_rgb_plus_seed_only=width, decoded=decoded, score=scored,
        model_capability_established=False, physical_event_accuracy_established=False,
        center_target_max=meta['stitched_center_max'], metadata_sha256=file_hash(oracle_dir / 'metadata.json'))
    write_json(oracle_dir / 'evaluation.json', result)
    print(json.dumps(dict(event='gt_target_oracle_scored',
        covered_fraction=scored['metrics']['downstream']['covered_fraction'],
        precision=scored['metrics']['downstream']['prediction_precision_at_5px'],
        predicted_length=scored['metrics']['downstream']['predicted_length_px'],
        reason=decoded['reason'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'score'))
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    (build if args.command == 'build' else score)(args.output_dir)
