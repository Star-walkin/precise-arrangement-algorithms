"""Bounded posthoc final-checkpoint gradient comparison on V14 training ten.

No optimizer, parameter update, holdout loading or threshold selection. The
shared-trunk gradients are derivatives of two losses on the SAME forward
pass. Their instantaneous cosine cannot prove what caused whole-training
centerline performance to change.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .v14_joint_experiment import BASE, TARGET_KEYS, load_inputs
from .v14_joint_boundary_model import build_model, joint_path_loss


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _measure(primary, auxiliary):
    primary_square = edge_square = dot = 0.
    primary_unused = edge_unused = 0
    for first, second in zip(primary, auxiliary):
        if first is None:
            primary_unused += 1
        if second is None:
            edge_unused += 1
        a = first.detach().reshape(-1).double() if first is not None else None
        b = second.detach().reshape(-1).double() if second is not None else None
        if a is not None:
            if not torch.isfinite(a).all():
                raise ValueError('nonfinite primary gradient')
            primary_square += float(a @ a)
        if b is not None:
            if not torch.isfinite(b).all():
                raise ValueError('nonfinite auxiliary gradient')
            edge_square += float(b @ b)
        if a is not None and b is not None:
            dot += float(a @ b)
    primary_norm, edge_norm = math.sqrt(primary_square), math.sqrt(edge_square)
    cosine = dot/(primary_norm*edge_norm) if primary_norm and edge_norm else None
    return dict(primary_gradient_l2=primary_norm, raw_edge_ce_gradient_l2=edge_norm,
        shared_gradient_dot_product=dot,
        cosine=max(-1., min(1., cosine)) if cosine is not None else None,
        weighted_edge_to_primary_norm_ratio=.5*edge_norm/primary_norm if primary_norm else None,
        primary_zero_norm=primary_norm == 0., edge_zero_norm=edge_norm == 0.,
        primary_unused_parameter_tensors=primary_unused, edge_unused_parameter_tensors=edge_unused)


def run(experiment=BASE, destination=None):
    started = time.perf_counter()
    experiment = Path(experiment).resolve()
    protocol_path = experiment/'protocol.json'
    checkpoint_path = experiment/'development/joint/checkpoint.pt'
    report_path = experiment/'development/joint/report.json'
    manifest_path = experiment/'train_manifest.json'
    protocol, report, manifest = (_read(path) for path in (protocol_path, report_path, manifest_path))
    protocol_hash, checkpoint_hash = _sha(protocol_path), _sha(checkpoint_path)
    if (report['protocol_sha256'] != protocol_hash or report['checkpoint_sha256'] != checkpoint_hash
            or report['phase'] != 'train' or report['arm'] != 'joint'
            or report['steps'] != protocol['development_steps']):
        raise ValueError('not the frozen final joint training checkpoint')
    for path, expected in protocol['source_snapshot'].items():
        if _sha(path) != expected:
            raise ValueError('frozen source changed: '+path)
    selected = manifest['samples']
    if len(selected) != 10 or any(sample['split'] != 'train' for sample in selected):
        raise ValueError('exactly ten training samples required')
    if '20260925' in json.dumps(selected):
        raise ValueError('reserved data forbidden')
    torch.set_num_threads(protocol['cpu_threads'])
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if (checkpoint['protocol_sha256'] != protocol_hash or checkpoint['phase'] != 'train'
            or checkpoint['arm'] != 'joint' or checkpoint['steps'] != protocol['development_steps']):
        raise ValueError('checkpoint internal provenance mismatch')
    model = build_model(protocol['seed'], protocol['base_channels'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    state_before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    shared = [(name, parameter) for name, parameter in model.named_parameters()
              if name.startswith(('encoders.', 'context.', 'decoders.'))]
    parameters = tuple(parameter for _, parameter in shared)
    if not parameters:
        raise ValueError('no shared trunk parameters')
    loaded = load_inputs(experiment, 'train', True)
    if {entry['sample_id'] for entry, _ in loaded} != {s['sample_id'] for s in selected}:
        raise ValueError('loaded training scope changed')
    rows = []
    for entry, arrays in loaded:
        sample_started = time.perf_counter()
        # Exactly the real-data duplicated minibatch used by the fixed run.
        inputs = torch.from_numpy(np.repeat(arrays['input'][None], 2, axis=0)).float()
        batch = {key: torch.from_numpy(np.repeat(arrays[key][None], 2, axis=0)).float()
                 for key in (*TARGET_KEYS, 'edge_target_distribution', 'edge_row_valid', 'edge_column_valid')}
        outputs = model(inputs)
        losses = joint_path_loss(outputs, batch, edge_weight=.5)
        primary = torch.autograd.grad(losses['primary_path_loss'], parameters,
                                      retain_graph=True, allow_unused=True)
        auxiliary = torch.autograd.grad(losses['edge_ce_loss'], parameters,
                                        retain_graph=False, allow_unused=True)
        row = dict(sample_id=entry['sample_id'], split='train',
            inference_input_sha256=entry['inference_input_sha256'],
            cache_sha256=entry['cache_sha256'],
            primary_path_loss=float(losses['primary_path_loss'].detach()),
            primary_center_ce=float(losses['center_loss'].detach()),
            primary_slope_loss=float(losses['slope_loss'].detach()),
            raw_auxiliary_edge_ce=float(losses['edge_ce_loss'].detach()),
            valid_center_columns=int(losses['valid_columns']),
            valid_edge_columns=int(losses['valid_edge_columns']),
            **_measure(primary, auxiliary), elapsed_ms=(time.perf_counter()-sample_started)*1000)
        rows.append(row)
        print(json.dumps({key: row[key] for key in ('sample_id', 'cosine', 'primary_gradient_l2',
            'raw_edge_ce_gradient_l2', 'weighted_edge_to_primary_norm_ratio')}, ensure_ascii=False), flush=True)
        del primary, auxiliary, losses, outputs
    unchanged = all(torch.equal(value, model.state_dict()[key]) for key, value in state_before.items())
    if not unchanged or _sha(checkpoint_path) != checkpoint_hash:
        raise RuntimeError('diagnostic unexpectedly changed weights/checkpoint')
    cosines = [row['cosine'] for row in rows if row['cosine'] is not None]
    ratios = [row['weighted_edge_to_primary_norm_ratio'] for row in rows
              if row['weighted_edge_to_primary_norm_ratio'] is not None]
    output = dict(diagnostic='V14 final-checkpoint shared-trunk primary-versus-raw-edge gradients',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_hash,
        protocol=str(protocol_path), protocol_sha256=protocol_hash,
        training_report_sha256=_sha(report_path), train_manifest_sha256=_sha(manifest_path),
        diagnostic_source_sha256=_sha(Path(__file__)),
        frozen_source_snapshot=protocol['source_snapshot'],
        scope='ten training images only; no holdout arrays or evaluation scores loaded',
        primary_objective='original center CE + 0.1 slope loss',
        auxiliary_objective='raw unweighted auxiliary two-edge CE', auxiliary_weight_for_ratio=.5,
        real_batch_size=2, same_forward_for_both_gradients=True,
        parameter_names=[name for name, _ in shared],
        shared_parameter_count=sum(parameter.numel() for parameter in parameters),
        excluded_prefixes=['center_head.', 'dense_slope_head.', 'edge_head.'],
        dtype='float32 model and forward; float64 gradient dot/norm accumulation',
        checkpoint_mode='eval; this architecture has no dropout or batch-normalization state',
        optimizer_created=False, parameter_updates=0, gradients_clipped=False,
        final_checkpoint_weights_unchanged=unchanged, gradient_accumulation_into_parameters=False,
        torch_version=str(torch.__version__), elapsed_seconds=time.perf_counter()-started,
        limitations=[
            'Cosine describes one local final-checkpoint loss landscape, not the whole training trajectory.',
            'A negative dot means infinitesimal pure auxiliary descent locally opposes primary descent; it is not proof of training causality.',
            'The actual fixed training used Adam, gradient clipping, synthetic alternation and changing parameters; none is simulated here.',
            'Gradient norm ratios do not directly predict centerline pixel error or event timing error.',
            'These are training-image posthoc diagnostics, not generalization or industrial acceptance results.'],
        rows=rows, summary=dict(samples=len(rows), defined_cosines=len(cosines),
            negative_cosines=sum(value < 0 for value in cosines),
            cosine_mean=float(np.mean(cosines)) if cosines else None,
            cosine_median=float(np.median(cosines)) if cosines else None,
            cosine_min=min(cosines) if cosines else None, cosine_max=max(cosines) if cosines else None,
            weighted_edge_norm_ratio_median=float(np.median(ratios)) if ratios else None))
    if destination is None:
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        destination = experiment/'development/diagnostics'/('gradient_final_'+stamp+'.json')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(destination), **output['summary']), ensure_ascii=False), flush=True)
    return output, destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=BASE)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run(args.experiment, args.output)


if __name__ == '__main__':
    main()
