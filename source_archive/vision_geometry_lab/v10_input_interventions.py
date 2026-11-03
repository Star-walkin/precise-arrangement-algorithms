"""Fixed-window, single-training-image input interventions; not a new model.

The original human seed STILL selects all windows and anchors the unchanged
decoder. Zeroing network seed channels therefore does not remove seed from
the complete system and cannot establish seed-free inference or identity.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIT_DIR = ROOT / 'vision_geometry_lab/outputs/v10_single_fit'
DEFAULT_OUTPUT = FIT_DIR / 'input_interventions'
CONDITIONS = ('baseline', 'global_descriptor_zero', 'seed_channels_zero')
ARMS = ('masked_bce', 'balanced_bce_dice')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def intervene_inputs(global_input, local_inputs, condition):
    """Copy arrays without moving windows or touching RGB; descriptor flag separate."""
    global_input, local_inputs = np.asarray(global_input), np.asarray(local_inputs)
    if (global_input.ndim != 3 or global_input.shape[0] != 4
            or local_inputs.ndim != 4 or local_inputs.shape[:2] != (9, 4)):
        raise ValueError('expected global 4xHxW and local 9x4xHxW')
    if condition not in CONDITIONS:
        raise ValueError('unknown fixed intervention')
    changed_global, changed_local = global_input.copy(), local_inputs.copy()
    if condition == 'seed_channels_zero':
        changed_global[3] = 0
        changed_local[:, 3] = 0
    if not np.array_equal(changed_global[:3], global_input[:3]) or not np.array_equal(changed_local[:, :3], local_inputs[:, :3]):
        raise AssertionError('RGB must never change')
    return changed_global, changed_local, condition == 'global_descriptor_zero'


def difference_summary(values, baseline, coverage):
    values, baseline, coverage = np.asarray(values), np.asarray(baseline), np.asarray(coverage)
    if values.shape != baseline.shape or values.shape != coverage.shape:
        raise ValueError('comparison grid mismatch')
    difference = np.abs(values.astype(np.float64)-baseline.astype(np.float64))
    known = difference[coverage > 0]
    return dict(full_grid_mean_abs=float(difference.mean()), full_grid_max_abs=float(difference.max()),
                covered_grid_mean_abs=float(known.mean()) if known.size else None,
                covered_grid_max_abs=float(known.max()) if known.size else None,
                full_grid_includes_zero_uncovered_area=True)


def freeze(output_dir):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError('intervention output must be new')
    fit_protocol_path = FIT_DIR / 'protocol.json'
    fit_protocol = json.loads(fit_protocol_path.read_text('utf-8'))
    for path, expected in fit_protocol['dependencies_sha256'].items():
        if file_hash(path) != expected:
            raise ValueError('frozen fit source changed')
    arms = {}
    for arm in ARMS:
        report_path = FIT_DIR / arm / 'report.json'
        report = json.loads(report_path.read_text('utf-8'))
        if report['steps_completed'] != 1200 or report['protocol_sha256'] != file_hash(fit_protocol_path):
            raise ValueError('only fixed final step 1200 is allowed')
        arms[arm] = dict(checkpoint=report['final_checkpoint'], checkpoint_sha256=file_hash(report['final_checkpoint']),
            training_report=str(report_path), training_report_sha256=file_hash(report_path),
            original_prediction=report['final_prediction'], original_prediction_sha256=report['final_prediction_sha256'])
    protocol = dict(version=1, diagnostic='V10 single-train fixed-window input interventions',
        source=str(Path(__file__).resolve()), source_sha256=file_hash(__file__),
        fit_protocol=str(fit_protocol_path), fit_protocol_sha256=file_hash(fit_protocol_path),
        fit_manifest=fit_protocol['manifest'], fit_manifest_sha256=fit_protocol['manifest_sha256'],
        sample_id='dual_20260924_111500__left__f00246', arms=arms,
        conditions=list(CONDITIONS), threads=2, tile_batch_size=2,
        baseline='unchanged original RGB + seed inputs and actual global descriptor',
        global_descriptor_zero='same global/local inputs; replace encoded 8D descriptor with zero',
        seed_channels_zero='zero channel 3 in BOTH global/local inputs; recompute descriptor; RGB unchanged',
        fixed_variables=['original RGB', 'original nine tile rectangles and coordinates', 'input sizes',
                         'original decoder free_seed', 'independent RGB width', 'final checkpoint',
                         'decoder and 5px path scoring'],
        gt_use='quality/reference for scoring only, never input or window selection',
        inference_count=6, supervised_rasterization=False, no_training=True,
        baseline_reproduction='compare to saved fixed final prediction; max abs <=1e-6 required',
        conclusion_limit='network-channel dependence on one memorized train image only; whole system still seed-conditioned',
        holdout_content_read=False, hardware_access=False)
    output_dir.mkdir(parents=True)
    write_new_json(output_dir / 'protocol.json', protocol)
    print(json.dumps(dict(event='input_intervention_protocol_frozen',
        protocol_sha256=file_hash(output_dir / 'protocol.json'))), flush=True)


def verify(output_dir):
    output_dir = Path(output_dir).resolve()
    protocol_path = output_dir / 'protocol.json'
    protocol = json.loads(protocol_path.read_text('utf-8'))
    if file_hash(protocol['source']) != protocol['source_sha256']:
        raise ValueError('intervention source changed after freeze')
    for key in ('fit_protocol', 'fit_manifest'):
        if file_hash(protocol[key]) != protocol[key + '_sha256']:
            raise ValueError('fit source changed: ' + key)
    fit = json.loads(Path(protocol['fit_protocol']).read_text('utf-8'))
    for path, expected in fit['dependencies_sha256'].items():
        if file_hash(path) != expected:
            raise ValueError('original frozen source changed')
    for arm in protocol['arms'].values():
        for key in ('checkpoint', 'training_report', 'original_prediction'):
            if file_hash(arm[key]) != arm[key + '_sha256']:
                raise ValueError('final fit artifact changed: ' + key)
    return protocol


def infer(output_dir):
    import torch
    from .v10_single_fit import verify_protocol
    from .centerline_multiscale_v9 import build_model, prepare_record, _stitch_tiles, _write_prediction
    output_dir = Path(output_dir).resolve()
    protocol = verify(output_dir)
    if (output_dir / 'inference.json').exists() or any(output_dir.glob('*.npz')):
        raise FileExistsError('intervention inference must not overwrite previous results')
    fit, record = verify_protocol(protocol['fit_protocol'])
    torch.set_num_threads(protocol['threads'])
    prepared = prepare_record(record, 'native_detail', include_supervision=False)
    original_global = prepared['global_input'].numpy()
    original_local = prepared['local_inputs'].numpy()
    rows = []
    for arm, artifacts in protocol['arms'].items():
        checkpoint = torch.load(artifacts['checkpoint'], map_location='cpu', weights_only=True)
        if checkpoint['step'] != 1200 or checkpoint['sample_id'] != protocol['sample_id']:
            raise ValueError('unexpected checkpoint')
        model = build_model(8)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.eval()
        baseline = None
        for condition in protocol['conditions']:
            global_input, local_inputs, zero_descriptor = intervene_inputs(original_global, original_local, condition)
            started = time.perf_counter()
            with torch.inference_mode():
                descriptor = model.encode_global(torch.from_numpy(global_input)[None])
                if zero_descriptor:
                    descriptor = torch.zeros_like(descriptor)
                predictions = []
                for index in range(0, 9, protocol['tile_batch_size']):
                    local = torch.from_numpy(local_inputs[index:index+protocol['tile_batch_size']])
                    output = model.decode_local(local, descriptor.expand(len(local), -1))
                    body, center = (torch.sigmoid(output[head + '_logits'])[:, 0].numpy() for head in ('body', 'center'))
                    predictions.extend((body[i], center[i]) for i in range(len(body)))
            body, center, coverage = _stitch_tiles(predictions, prepared['tile_layout'], record['image_shape_hw'])
            if condition == 'baseline':
                baseline = (body, center, coverage)
                with np.load(artifacts['original_prediction'], allow_pickle=False) as old:
                    reproduction = {key: difference_summary(value, old[key], coverage)
                        for key, value in (('body_probability', body), ('center_probability', center))}
                    if not np.array_equal(coverage, old['coverage_mask']) or any(
                            value['full_grid_max_abs'] > 1e-6 for value in reproduction.values()):
                        raise ValueError('baseline no longer reproduces fixed training export')
            elif not np.array_equal(coverage, baseline[2]):
                raise AssertionError('intervention moved coverage')
            path = _write_prediction(output_dir / f'{arm}__{condition}.npz', body, center, coverage)
            rows.append(dict(arm=arm, condition=condition, prediction=str(path), prediction_sha256=file_hash(path),
                differences_from_same_checkpoint_baseline={
                    'body': difference_summary(body, baseline[0], coverage),
                    'center': difference_summary(center, baseline[1], coverage)},
                original_export_reproduction=reproduction if condition == 'baseline' else None,
                descriptor_l2=float(torch.linalg.vector_norm(descriptor)),
                global_seed_sum=float(global_input[3].sum()), local_seed_sum=float(local_inputs[:, 3].sum()),
                global_rgb_unchanged=bool(np.array_equal(global_input[:3], original_global[:3])),
                local_rgb_unchanged=bool(np.array_equal(local_inputs[:, :3], original_local[:, :3])),
                original_tile_layout=prepared['tile_layout'], original_decoder_seed_retained=True,
                elapsed_seconds=time.perf_counter()-started, no_target_raster_used=True))
            print(json.dumps(dict(event='intervention_inferred', arm=arm, condition=condition,
                center_mean_abs=rows[-1]['differences_from_same_checkpoint_baseline']['center']['full_grid_mean_abs'])), flush=True)
    verify(output_dir)
    write_new_json(output_dir / 'inference.json', dict(protocol_sha256=file_hash(output_dir / 'protocol.json'), rows=rows,
        input_contract=prepared['input_contract'], no_training=True, no_gt_target_input=True,
        seed_still_used_for_windows_and_decoder=True))


def score(output_dir):
    import cv2
    from .centerline_benchmark_v5 import decode_heatmaps
    from .centerline_quality_v5 import load_label_quality
    from .centerline_review_v5 import read_clean_image, validate_annotation
    from .evaluate_centerline_multiscale_v9 import independent_width, score_decoded
    output_dir = Path(output_dir).resolve()
    protocol = verify(output_dir)
    inference_path = output_dir / 'inference.json'
    inference = json.loads(inference_path.read_text('utf-8'))
    if inference['protocol_sha256'] != file_hash(output_dir / 'protocol.json'):
        raise ValueError('intervention protocol mismatch')
    manifest = json.loads(Path(protocol['fit_manifest']).read_text('utf-8'))
    if len(manifest['samples']) != 1 or manifest['samples'][0]['sample_id'] != protocol['sample_id']:
        raise ValueError('single training scope required')
    sample = manifest['samples'][0]
    fit = json.loads(Path(protocol['fit_protocol']).read_text('utf-8'))
    if sample['split'] != 'train' or file_hash(sample['annotation_path']) != fit['annotation_sha256']:
        raise ValueError('train label changed')
    label = validate_annotation(json.loads(Path(sample['annotation_path']).read_text('utf-8-sig')), sample)
    quality = load_label_quality(sample)
    if quality['quality_sha256'] != fit['quality_sha256']:
        raise ValueError('quality changed')
    image = cv2.cvtColor(np.asarray(read_clean_image(sample)), cv2.COLOR_RGB2BGR)
    width = independent_width(image, label['polylines']['free_seed'])
    rows = []
    for row in inference['rows']:
        if file_hash(row['prediction']) != row['prediction_sha256']:
            raise ValueError('probability map changed')
        with np.load(row['prediction'], allow_pickle=False) as data:
            body, center, coverage = (data[key] for key in ('body_probability', 'center_probability', 'coverage_mask'))
        decoded = decode_heatmaps(body, center, sample['roi_xyxy'], label['polylines']['free_seed'], width['width_px'])
        scored = score_decoded(label, quality, decoded, coverage, sample['image_shape_hw'])
        rows.append(dict(arm=row['arm'], condition=row['condition'], decoded=decoded, score=scored,
                         differences=row['differences_from_same_checkpoint_baseline']))
        metric = scored['metrics']['downstream']
        print(json.dumps(dict(event='intervention_scored', arm=row['arm'], condition=row['condition'],
            coverage=metric['covered_fraction'], precision=metric['prediction_precision_at_5px'],
            predicted_length_px=metric['predicted_length_px'])), flush=True)
    write_new_json(output_dir / 'evaluation.json', dict(protocol_sha256=file_hash(output_dir / 'protocol.json'),
        inference_sha256=file_hash(inference_path), independent_width=width, rows=rows,
        interpretation='one training image only, no generalization or rope identity claim; seed retained in windows/decoder',
        no_training=True, hardware_access=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'infer', 'score'))
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    {'freeze': freeze, 'infer': infer, 'score': score}[args.command](args.output_dir)
