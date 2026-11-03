"""One-training-image optimization sanity check, not generalization evidence.

Reuse frozen V9 architecture/inputs/targets. Change only the training loss;
never use held-out image/label contents or select the best checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from .centerline_model_v5 import inspect_manifest, masked_two_head_loss
from .centerline_multiscale_v9 import (
    PreparedRecordCache, _batch_from_sequence, _pixel_diagnostic,
    _state_dict_sha256, _write_prediction, build_model,
    build_training_sequence, predict_record,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / 'vision_geometry_lab/outputs/centerline_v9/dataset/manifest.json'
DEFAULT_OUTPUT = ROOT / 'vision_geometry_lab/outputs/v10_single_fit'
SAMPLE_ID = 'dual_20260924_111500__left__f00246'
ARMS = ('masked_bce', 'balanced_bce_dice')
DEPENDENCIES = (
    'v10_single_fit.py', 'v10_single_fit_evaluate.py',
    'centerline_model_v5.py', 'centerline_multiscale_v9.py',
    'centerline_multiscale_contract_v9.py', 'centerline_review_v5.py',
    'centerline_quality_v5.py', 'centerline_quality_cutoff_v8.py',
    'centerline_benchmark_v5.py', 'centerline_path_metrics_v6.py',
    'evaluate_centerline_multiscale_v9.py', 'rope_detector.py',
)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _balanced_head(logits, target, valid):
    """Equal positive/negative SOFT target mass; unknown pixels never negative."""
    valid = valid.to(logits.dtype)
    safe_target = torch.where(valid > 0, target, torch.zeros_like(target)).to(logits.dtype)
    safe_logits = torch.where(valid > 0, logits, torch.zeros_like(logits))
    positive = valid * safe_target
    negative = valid * (1.0 - safe_target)
    positive_mass, negative_mass = positive.sum(), negative.sum()
    positive_present = (positive_mass > 0).to(logits.dtype)
    negative_present = (negative_mass > 0).to(logits.dtype)
    positive_term = (positive * F.softplus(-safe_logits)).sum() / positive_mass.clamp_min(1e-6)
    negative_term = (negative * F.softplus(safe_logits)).sum() / negative_mass.clamp_min(1e-6)
    balanced_bce = (positive_term * positive_present + negative_term * negative_present) / (
        positive_present + negative_present).clamp_min(1.0)
    probabilities = torch.sigmoid(safe_logits)
    epsilon = 1e-6
    dice = (1.0 - (2 * (positive * probabilities).sum() + epsilon) / (
        (valid * probabilities).sum() + positive_mass + epsilon)) * positive_present
    # No-positive known-negative tiles use BCE alone; no supervision returns zero.
    loss = torch.where(positive_mass > 0, .5 * balanced_bce + .5 * dice, balanced_bce)
    return loss, balanced_bce, dice


def paired_loss(outputs, batch, arm):
    if arm not in ARMS:
        raise ValueError('unknown fixed loss arm')
    # Shared preconditions are inherited from the unchanged baseline loss.
    baseline = masked_two_head_loss(outputs, batch['body_target'], batch['center_target'],
        body_valid_mask=batch['body_valid_mask'], center_valid_mask=batch['center_valid_mask'])
    if arm == 'masked_bce':
        return baseline
    result = {}
    for head in ('body', 'center'):
        loss, bce, dice = _balanced_head(outputs[head + '_logits'], batch[head + '_target'],
                                         batch[head + '_valid_mask'])
        result[head + '_loss'] = loss
        result[head + '_balanced_bce'] = bce
        result[head + '_dice_loss'] = dice
        result[head + '_valid_weight'] = baseline[head + '_valid_weight']
    result['loss'] = result['body_loss'] + result['center_loss']
    return result


def select_train_sample(manifest):
    selected = [item for item in manifest.get('samples', []) if item.get('sample_id') == SAMPLE_ID]
    if len(selected) != 1 or selected[0].get('split') != 'train':
        raise ValueError('the single allowed training record must exist exactly once')
    if '20260925' in json.dumps(selected[0]):
        raise ValueError('future blind source prohibited')
    return selected[0]


def freeze(manifest_path, output_dir):
    output_dir = Path(output_dir).resolve()
    if (output_dir / 'protocol.json').exists() or (output_dir / 'manifest.json').exists():
        raise FileExistsError('existing freeze must not be overwritten')
    manifest_path = Path(manifest_path).resolve()
    source = json.loads(manifest_path.read_text('utf-8-sig'))
    sample = select_train_sample(source)
    minimal = dict(schema_version=1, task='seed_conditioned_new_winding_centerline',
        samples=[sample], use='training-only single-image fit sanity; no holdout reads',
        source_manifest=str(manifest_path), source_manifest_sha256=file_hash(manifest_path))
    # Only this one sample is passed to the loader; it never receives the full dataset.
    output_dir.mkdir(parents=True, exist_ok=True)
    write_new_json(output_dir / 'manifest.json', minimal)
    inspected = inspect_manifest(output_dir / 'manifest.json')
    if len(inspected['usable']['train']) != 1 or inspected['usable']['holdout'] or inspected['skipped']:
        raise ValueError('single training record failed strict validation')
    record = inspected['usable']['train'][0]
    dependencies = {str(Path(__file__).with_name(name).resolve()):
                    file_hash(Path(__file__).with_name(name)) for name in DEPENDENCIES}
    sequence = build_training_sequence([record], 1200, 2, 20260927)
    protocol = dict(version=1, experiment='V10 single-image paired-loss optimization sanity',
        manifest=str(output_dir / 'manifest.json'), manifest_sha256=file_hash(output_dir / 'manifest.json'),
        sample_id=SAMPLE_ID, source_manifest=str(manifest_path), source_manifest_sha256=file_hash(manifest_path),
        image_sha256=record['image_sha256'], annotation_sha256=record['annotation_sha256'],
        quality_sha256=record['quality_sha256'], dependencies_sha256=dependencies,
        mode='native_detail', architecture='unchanged V9 global descriptor/local UNet',
        base_channels=8, seed=20260927, threads=2, batch_size=2, steps=1200,
        optimizer='Adam', learning_rate=.001, arms=list(ARMS),
        sampler_sequence_sha256=value_hash(sequence),
        checkpoints=[300, 600, 900, 1200], checkpoint_selection='last 1200 only; intermediates diagnostic',
        changed_variable_only='loss function', input='RGB plus original human free_seed, unchanged nine windows',
        target='unchanged V9 native sigma2, masks/quality; validated C7 trusted targets when present',
        this_sample_quality='historical f246 quality polygon, not a newly invented C7 cutoff',
        balanced_loss=dict(per_head='0.5 balanced soft-target BCE + 0.5 masked softDice when positive mass exists',
            positive='sum(mask*target*softplus(-logit))/sum(mask*target)',
            negative='sum(mask*(1-target)*softplus(logit))/sum(mask*(1-target))',
            balance='arithmetic mean of nonempty positive/negative terms over current batch',
            dice='1-(2*sum(mask*p*target)+1e-6)/(sum(mask*p)+sum(mask*target)+1e-6)',
            no_positive='known-negative BCE only; dice disabled',
            no_valid='zero loss and zero output gradient; Adam may still apply prior momentum',
            probabilistic_calibration_claim=False),
        evaluation='unchanged V5 decoder threshold0.5 and V6 5 native-pixel path metrics; train-only',
        width='independent original RGB + free_seed, no target width',
        holdout_content_read=False, future_blind_content_read=False,
        hardware_access=False, production_deployment=False)
    write_new_json(output_dir / 'protocol.json', protocol)
    print(json.dumps(dict(event='protocol_frozen', output=str(output_dir),
        protocol_sha256=file_hash(output_dir / 'protocol.json'))), flush=True)
    return protocol


def verify_protocol(protocol_path):
    protocol_path = Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text('utf-8-sig'))
    for path, expected in protocol['dependencies_sha256'].items():
        if file_hash(path) != expected:
            raise ValueError('frozen implementation changed: ' + path)
    for key in ('manifest', 'source_manifest'):
        if file_hash(protocol[key]) != protocol[key + '_sha256']:
            raise ValueError('frozen manifest changed')
    inspected = inspect_manifest(protocol['manifest'])
    if len(inspected['usable']['train']) != 1 or inspected['usable']['holdout'] or inspected['skipped']:
        raise ValueError('training-only scope changed')
    record = inspected['usable']['train'][0]
    for key in ('sample_id', 'image_sha256', 'annotation_sha256', 'quality_sha256'):
        if record[key] != protocol[key]:
            raise ValueError('sample snapshot changed: ' + key)
    return protocol, record


def center_response(model, prepared):
    values, positive_values = [], []
    model.eval()
    with torch.inference_mode():
        descriptor = model.encode_global(prepared['global_input'][None])
        for index, supervision in enumerate(prepared['supervision']):
            output = model.decode_local(prepared['local_inputs'][index:index+1], descriptor)
            cp = torch.sigmoid(output['center_logits'])[0]
            valid = supervision['center_valid_mask'] > 0
            if valid.any():
                values.append(cp[valid].numpy())
            positive = valid & (supervision['center_target'] >= .5)
            if positive.any():
                positive_values.append(cp[positive].numpy())
    def describe(arrays):
        if not arrays:
            return None
        data = np.concatenate(arrays)
        return dict(count=int(data.size), min=float(data.min()), max=float(data.max()),
            mean=float(data.mean()), median=float(np.median(data)),
            fraction_ge_half=float(np.mean(data >= .5)))
    return dict(known_corridor=describe(values), reference_target_ge_half=describe(positive_values),
                response_not_calibrated_probability=True)


def train(protocol_path, arm):
    protocol, record = verify_protocol(protocol_path)
    if arm not in ARMS:
        raise ValueError('unknown arm')
    output = Path(protocol_path).resolve().parent / arm
    if output.exists():
        raise FileExistsError('run directory already exists; no overwrite or resume')
    output.mkdir()
    torch.set_num_threads(protocol['threads'])
    sequence = build_training_sequence([record], protocol['steps'], protocol['batch_size'], protocol['seed'])
    if value_hash(sequence) != protocol['sampler_sequence_sha256']:
        raise ValueError('sampler reconstruction differs')
    started = time.perf_counter()
    cache = PreparedRecordCache([record], protocol['mode'], cache_samples=True)
    prepared = cache.get(0)  # Native masks generated once, not every optimization step.
    torch.manual_seed(protocol['seed'])
    model = build_model(protocol['base_channels'])
    initial_hash = _state_dict_sha256(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol['learning_rate'])
    history, diagnostics = [], []
    loop_started = time.perf_counter()
    model.train()
    for step, items in enumerate(sequence, 1):
        batch, zero_items = _batch_from_sequence(cache, items)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch['global_input'], batch['local_input'])
        losses = paired_loss(outputs, batch, arm)
        losses['loss'].backward()
        optimizer.step()
        history.append(dict(step=step, loss=float(losses['loss'].detach()),
            body_loss=float(losses['body_loss'].detach()), center_loss=float(losses['center_loss'].detach()),
            zero_supervision_items=zero_items, tiles=[x['tile_index'] for x in items]))
        if step == 1 or step % 100 == 0:
            print(json.dumps(dict(event='v10_fit_progress', arm=arm, step=step,
                loss=history[-1]['loss'], loop_seconds=time.perf_counter()-loop_started)), flush=True)
        if step in protocol['checkpoints']:
            checkpoint = dict(state_dict=model.state_dict(), step=step, arm=arm,
                initial_state_sha256=initial_hash, final_state_sha256=_state_dict_sha256(model.state_dict()),
                protocol_sha256=file_hash(protocol_path), sample_id=record['sample_id'],
                mode='native_detail', train_only=True)
            torch.save(checkpoint, output / f'checkpoint_{step:04d}.pt')
            diagnostics.append(dict(step=step,
                common_masked_bce=_pixel_diagnostic(model, cache, 1.0, 'train-only; not checkpoint selection'),
                center_response=center_response(model, prepared)))
            model.train()
    loop_seconds = time.perf_counter()-loop_started
    model.eval()
    body, center, coverage, contract, timings = predict_record(model, {'mode': 'native_detail'}, record,
                                                              tile_batch_size=2)
    prediction = _write_prediction(output / 'final_prediction.npz', body, center, coverage)
    verify_protocol(protocol_path)
    report = dict(version=1, status='completed_single_training_image_only', arm=arm,
        protocol=str(Path(protocol_path).resolve()), protocol_sha256=file_hash(protocol_path),
        sample_id=record['sample_id'], steps_completed=1200,
        initial_state_sha256=initial_hash, final_state_sha256=_state_dict_sha256(model.state_dict()),
        sampler_sequence_sha256=value_hash(sequence), input_contract=prepared['input_contract'],
        supervision_metadata=prepared['supervision_metadata'], history=history,
        diagnostics=diagnostics, final_prediction=str(prediction), final_prediction_sha256=file_hash(prediction),
        final_checkpoint=str(output / 'checkpoint_1200.pt'), checkpoint_selection='fixed last only',
        timing=dict(total_seconds=time.perf_counter()-started, loop_seconds=loop_seconds,
                    preparation_seconds=prepared['preparation_seconds'], uncached_inference_ms=timings),
        no_holdout_evaluated=True, frozen_sources_verified=True, hardware_access=False,
        generalization_established=False, physical_event_accuracy_established=False)
    write_new_json(output / 'report.json', report)
    print(json.dumps(dict(event='v10_fit_complete', arm=arm, report=str(output / 'report.json'))), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    item = sub.add_parser('freeze')
    item.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    item.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    item = sub.add_parser('train')
    item.add_argument('--protocol', type=Path, default=DEFAULT_OUTPUT / 'protocol.json')
    item.add_argument('--arm', choices=ARMS, required=True)
    args = parser.parse_args(argv)
    if args.command == 'freeze':
        freeze(args.manifest, args.output_dir)
    else:
        train(args.protocol, args.arm)


if __name__ == '__main__':
    main()
