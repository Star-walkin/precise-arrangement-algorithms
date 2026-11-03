"""Finite offline V11 seed-conditioned cross-section path experiment.

CV preparation/scoring and torch training run in the existing separate venvs.
No hardware, no change to live software, no automatic event/force inference.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np

LAB = Path(__file__).resolve().parent
BASE = LAB/'outputs/v11_path_identity'
MANIFEST = LAB/'outputs/centerline_v9/dataset/manifest.json'
SAMPLE = 'dual_20260924_111500__left__f00246'
ARMS = ('single_real', 'single_identity')
SOURCES = ('v11_path_experiment.py', 'v11_path_contract.py', 'v11_path_decode.py',
           'v11_path_model.py', 'v11_identity_probe.py')
TARGET_KEYS = ('target_distribution', 'column_valid', 'row_valid', 'slope_target', 'slope_valid')


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def freeze(output):
    if output.exists():
        raise FileExistsError('Use a new output directory; do not replace experiments')
    source = read(MANIFEST)
    if '20260925' in json.dumps(source):
        raise ValueError('Blind source prohibited')
    records = source['samples']
    assert sum(s['split']=='train' for s in records) == 10
    assert sum(s['split']=='holdout' for s in records) == 8
    from .run_centerline_experiment_v9 import PROTOCOL, verify_snapshot
    old = read(PROTOCOL)
    verify_snapshot(old)
    protocol = dict(version=11, frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        task='seed-conditioned spatial cross-section distribution and path connection',
        single_image=SAMPLE, arms=list(ARMS), steps=1200, final_checkpoint_only=True,
        seed=20260928, cpu_threads=2, base_channels=8, learning_rate=.001, batch_size=2,
        arm_difference='single_real: same real image twice; single_identity: alternate real and generated same-RGB two-prompt pairs',
        synthetic_train_seed_base=100000, synthetic_test_seeds=list(range(900000,900016)),
        probe_tolerance_strip_rows=2., identity_probe_pair_pass=.9,
        conditional_development_gate=dict(real_downstream_coverage=.9, real_downstream_precision=.9, synthetic_pair_pass_fraction=.9),
        development_steps=2400, development_samples=10, development_holdout=8,
        development_training='alternate round-robin real image and generated two-prompt pair, final checkpoint only',
        full_data_run_requires_single_identity_gate=True, manual_seed_still_required=True,
        inference_inputs='RGB + reviewed free seed + independently measured RGB optical width',
        quality_or_target_affects_inference=False, unknown_tail_is_background=False,
        gate_warning='Mechanism checks only, not physical accuracy or deployment acceptance',
        evaluation='unchanged V6 5-native-pixel full/downstream arc-length scores; no clipping denominator to strip',
        decision='No threshold/source/step changes after freeze; report failure and defer development holdout if gate fails',
        source_snapshot={str(LAB/name):sha(LAB/name) for name in SOURCES},
        old_source_snapshot=old['source_snapshot'], dataset_snapshot=old['dataset_snapshot'],
        manifest_sha256=sha(MANIFEST), hardware_access=False, force_command=None)
    write_json(output/'protocol.json', protocol)
    print(json.dumps(dict(protocol=str(output/'protocol.json'), sha256=sha(output/'protocol.json'))))


def verify(output):
    protocol = read(output/'protocol.json')
    for category in ('source_snapshot','old_source_snapshot','dataset_snapshot'):
        for path, expected in protocol[category].items():
            if expected is None:
                if Path(path).exists(): raise ValueError('Previously absent source appeared: '+path)
            elif sha(path) != expected:
                raise ValueError('Frozen source or data changed: '+path)
    if sha(MANIFEST) != protocol['manifest_sha256']:
        raise ValueError('Manifest changed')
    return protocol


def subset_manifest(output, split):
    source = read(MANIFEST)
    selected = [s for s in source['samples'] if s['sample_id']==SAMPLE] if split=='single' else source['samples']
    if split == 'single' and (len(selected)!=1 or selected[0]['split']!='train'):
        raise ValueError('Single training sample mismatch')
    source['samples'] = selected
    path = output/(split+'_manifest.json')
    write_json(path, source)
    return path


def require_gate(output):
    gate = read(output/'gate.json')
    if not gate['allow_development_run']:
        raise ValueError('Single-image/identity mechanism gate failed: no development run authorized by protocol')
    if gate['protocol_sha256'] != sha(output/'protocol.json'):
        raise ValueError('Gate from different protocol')
    if gate.get('evaluation_sha256') != sha(output/'single_identity/evaluation.json'):
        raise ValueError('Gate evaluation changed')
    if gate.get('checkpoint_sha256') != sha(output/'single_identity/checkpoint.pt'):
        raise ValueError('Gate model changed')


def prepare(output, development=False):
    verify(output)
    if development: require_gate(output)
    folder = output/('development_cache' if development else 'single_cache')
    if folder.exists(): raise FileExistsError(folder)
    from .evaluate_centerline_multiscale_v9 import _dataset, independent_width, scorable_reference
    from .centerline_review_v5 import read_clean_image
    from .v11_path_contract import build_input, build_targets
    manifest = subset_manifest(output, 'development' if development else 'single')
    _, records, _ = _dataset(manifest)
    entries = []
    for record in records:
        sample, label, quality = record['sample'], record['label'], record['quality']
        start = time.perf_counter()
        rgb = np.asarray(read_clean_image(sample))
        width = independent_width(np.ascontiguousarray(rgb[:,:,::-1]), label['polylines']['free_seed'])
        entry = dict(sample_id=sample['sample_id'], split=sample['split'], group_id=sample['group_id'],
            width=width, image_sha256=sample['image_sha256'], annotation_sha256=record['annotation_sha256'],
            quality_sha256=quality['quality_sha256'], cohort=record['cohort'])
        if width['width_px'] is None:
            entry.update(available=False, reason='independent_width_unavailable')
        else:
            built = build_input(rgb, label['polylines']['free_seed'], width['width_px'])
            # Training cache only contains supervision for actual train rows.
            arrays = dict(input=built['input'], valid_mask=built['valid_mask'])
            if sample['split']=='train':
                reference, _ = scorable_reference(label, quality)
                arrays.update(build_targets(built['geometry'], reference,
                    label['ignore_polygons']+quality['precision_exclude_polygons'], built['valid_mask']))
            path = folder/(sample['sample_id']+'.npz')
            write_npz(path, **{k:arrays[k] for k in ('input','valid_mask',*TARGET_KEYS) if k in arrays})
            entry.update(available=True, cache=str(path), cache_sha256=sha(path), geometry=built['geometry'],
                inference_input_sha256=hashlib.sha256(built['input'].tobytes()).hexdigest(),
                supervised_columns=int(np.count_nonzero(arrays.get('column_valid',[]))))
        entry['preparation_ms'] = (time.perf_counter()-start)*1000
        entries.append(entry)
        print(json.dumps(dict(prepared=entry['sample_id'], available=entry['available'],
            columns=entry.get('supervised_columns')),ensure_ascii=False),flush=True)
    write_json(folder/'index.json', dict(entries=entries, manifest=str(manifest),
        manifest_sha256=sha(manifest), protocol_sha256=sha(output/'protocol.json')))
    verify(output)


def load_cache(output, development=False):
    folder = output/('development_cache' if development else 'single_cache')
    index = read(folder/'index.json')
    if index['protocol_sha256'] != sha(output/'protocol.json') or index['manifest_sha256'] != sha(index['manifest']):
        raise ValueError('Cache provenance mismatch')
    result = []
    for entry in index['entries']:
        if not entry['available']:
            result.append((entry,None)); continue
        if sha(entry['cache']) != entry['cache_sha256']: raise ValueError('Cache changed')
        with np.load(entry['cache'], allow_pickle=False) as archive:
            arrays = {key:archive[key].copy() for key in archive.files}
        result.append((entry,arrays))
    return index, result


def train(output, arm):
    protocol = verify(output)
    development = arm=='development'
    if development: require_gate(output)
    run = output/arm
    if run.exists(): raise FileExistsError('Do not replace or silently resume a prior run')
    import torch
    from .v11_path_model import build_model, path_loss
    from .v11_identity_probe import make_pair
    from .centerline_multiscale_v9 import _state_dict_sha256
    torch.set_num_threads(protocol['cpu_threads'])
    index, cached = load_cache(output, development)
    real = [(e,a) for e,a in cached if e['split']=='train' and a is not None]
    if not real: raise ValueError('No usable training input')
    if not development and (len(real)!=1 or real[0][0]['sample_id']!=SAMPLE): raise ValueError('Single-image scope')
    model = build_model(protocol['seed'], base_channels=protocol['base_channels'])
    initial_sha = _state_dict_sha256(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol['learning_rate'])
    steps = protocol['development_steps'] if development else protocol['steps']
    log, sequence = [], []
    start = time.perf_counter()
    model.train()
    for step in range(steps):
        synthetic = arm!='single_real' and step%2==1
        if synthetic:
            pair = make_pair(protocol['synthetic_train_seed_base']+step//2)
            x = pair['inputs']
            batch = {key:torch.from_numpy(np.stack([t[key] for t in pair['targets']])).float() for key in TARGET_KEYS}
            identity = 'synthetic:'+str(protocol['synthetic_train_seed_base']+step//2)
        else:
            entry, arrays = real[(step//2 if arm!='single_real' else step)%len(real)]
            x = np.repeat(arrays['input'][None],2,axis=0)
            batch = {key:torch.from_numpy(np.repeat(arrays[key][None],2,axis=0)).float() for key in TARGET_KEYS}
            identity = entry['sample_id']
        sequence.append(identity)
        optimizer.zero_grad(set_to_none=True)
        losses = path_loss(model(torch.from_numpy(x).float()), batch)
        loss = losses['loss']
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step==0 or (step+1)%100==0:
            row = dict(step=step+1, loss=float(loss.detach()), synthetic=synthetic,
                       elapsed_seconds=time.perf_counter()-start)
            log.append(row); print(json.dumps(dict(arm=arm,**row)),flush=True)
    run.mkdir(parents=True)
    checkpoint = run/'checkpoint.pt'
    torch.save(dict(state_dict=model.state_dict(), seed=protocol['seed'], base_channels=protocol['base_channels'],
        arm=arm, steps=steps, protocol_sha256=sha(output/'protocol.json')), checkpoint)
    write_json(run/'report.json', dict(arm=arm, steps=steps, initial_state_sha256=initial_sha,
        final_state_sha256=_state_dict_sha256(model.state_dict()), train_ids=[e['sample_id'] for e,a in real],
        train_groups=sorted({e['group_id'] for e,a in real}),
        sequence_sha256=hashlib.sha256(json.dumps(sequence).encode()).hexdigest(),
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), loss_log=log,
        elapsed_seconds=time.perf_counter()-start, parameters=sum(p.numel() for p in model.parameters()),
        protocol_sha256=sha(output/'protocol.json'), cache_index_sha256=sha(output/('development_cache' if development else 'single_cache')/'index.json'),
        field_generalization_validated=False, hardware_access=False))
    verify(output)


def probabilities(outputs, valid):
    import torch
    logits = outputs['center_logits']
    mask = valid > .5
    safe = torch.where(mask, logits, torch.full_like(logits,-1.e4))
    probability = torch.softmax(safe, dim=1)*mask.to(logits.dtype)
    return probability/probability.sum(dim=1,keepdim=True).clamp_min(1.e-8)


def export(output, arm):
    protocol = verify(output)
    import torch
    from .v11_path_model import build_model
    from .v11_identity_probe import make_pair, assess_pair
    torch.set_num_threads(protocol['cpu_threads'])
    run = output/arm
    folder = run/'predictions'
    if folder.exists(): raise FileExistsError(folder)
    checkpoint = torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=True)
    if checkpoint['protocol_sha256'] != sha(output/'protocol.json') or checkpoint['arm'] != arm:
        raise ValueError('Checkpoint protocol mismatch')
    model = build_model(checkpoint['seed'],base_channels=checkpoint['base_channels'])
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    report = read(run/'report.json')
    if report['arm']!=arm or report['protocol_sha256']!=sha(output/'protocol.json'):
        raise ValueError('Training report belongs to different arm/protocol')
    if sha(run/'checkpoint.pt') != report['checkpoint_sha256']: raise ValueError('Checkpoint changed')
    cache_index=output/('development_cache' if arm=='development' else 'single_cache')/'index.json'
    if report['cache_index_sha256']!=sha(cache_index): raise ValueError('Training cache index changed')
    _, cached = load_cache(output, arm=='development')
    records=[]
    with torch.inference_mode():
        for entry, arrays in cached:
            if arrays is None:
                records.append(dict(sample_id=entry['sample_id'], available=False));continue
            start=time.perf_counter()
            x=torch.from_numpy(arrays['input'][None]).float()
            out=model(x)
            p=probabilities(out,torch.from_numpy(arrays['valid_mask'][None]))[0].numpy()
            slope=out['slope'][0].numpy()
            elapsed=(time.perf_counter()-start)*1000
            path=folder/(entry['sample_id']+'.npz')
            write_npz(path,probability=p,slope=slope)
            records.append(dict(sample_id=entry['sample_id'],available=True,path=str(path),sha256=sha(path),
                forward_ms=elapsed,input_sha256=entry['inference_input_sha256'],geometry=entry['geometry']))
        probes=[]
        for seed in protocol['synthetic_test_seeds']:
            pair=make_pair(seed,test=True)
            out=model(torch.from_numpy(pair['inputs']))
            p=probabilities(out,torch.from_numpy(pair['inputs'][:,7])).numpy()
            probes.append(dict(seed=seed,**assess_pair(p,pair)))
        # A zero-prompt output has no unique true target: sensitivity only.
        first=next(a for e,a in cached if a is not None and e['split']=='train')
        base=torch.from_numpy(first['input'][None]).float()
        valid=torch.from_numpy(first['valid_mask'][None])
        baseline=probabilities(model(base),valid).numpy()
        altered=base.clone();altered[:,4]=0
        no_seed=probabilities(model(altered),valid).numpy()
        sensitivity=dict(mean_probability_change=float(np.abs(baseline-no_seed).mean()),
            median_argmax_shift_rows=float(np.median(np.abs(baseline.argmax(1)-no_seed.argmax(1)))),
            crop_rgb_coordinates_unchanged=True,no_correct_path_claim_for_missing_prompt=True)
    write_json(run/'exports.json',dict(records=records,synthetic_probes=probes,
        synthetic_pair_pass_fraction=float(np.mean([p['both_queries_pass'] for p in probes])),
        zero_prompt_diagnostic=sensitivity, checkpoint_sha256=sha(run/'checkpoint.pt'),
        arm=arm,training_report_sha256=sha(run/'report.json'),cache_index_sha256=sha(cache_index),
        protocol_sha256=sha(output/'protocol.json'),identity_confirmed=False,hardware_access=False))
    verify(output)


def evaluate(output, arm):
    verify(output)
    from .evaluate_centerline_multiscale_v9 import _dataset, score_decoded, aggregate
    from .v11_path_decode import decode_path
    from .v11_path_contract import image_to_strip
    import cv2
    run=output/arm;exports=read(run/'exports.json')
    if exports['protocol_sha256']!=sha(output/'protocol.json') or exports['checkpoint_sha256']!=sha(run/'checkpoint.pt'):
        raise ValueError('Export provenance mismatch')
    if exports['arm']!=arm or exports['training_report_sha256']!=sha(run/'report.json'):
        raise ValueError('Export training report mismatch')
    index,cached=load_cache(output,arm=='development')
    if exports['cache_index_sha256']!=sha(output/('development_cache' if arm=='development' else 'single_cache')/'index.json'):
        raise ValueError('Export input cache mismatch')
    _, records,_=_dataset(index['manifest'])
    predictions={r['sample_id']:r for r in exports['records']}
    entries={e['sample_id']:(e,a) for e,a in cached}
    rows=[]
    for record in records:
        sample,label,quality=record['sample'],record['label'],record['quality']
        entry,arrays=entries[sample['sample_id']];pred=predictions[sample['sample_id']]
        coverage=None
        if pred['available']:
            if sha(pred['path'])!=pred['sha256'] or pred['input_sha256']!=entry['inference_input_sha256']:
                raise ValueError('Prediction/input changed')
            with np.load(pred['path'],allow_pickle=False) as ar:
                p,slope=ar['probability'],ar['slope']
            start=time.perf_counter()
            decoded=decode_path(p,slope,entry['geometry'],arrays['valid_mask'])
            decode_ms=(time.perf_counter()-start)*1000
            height,width=sample['image_shape_hw']
            gy,gx=np.indices((height//2,width//2),dtype=np.float32)
            xy=np.stack([(gx+.5)*2-.5,(gy+.5)*2-.5],axis=-1)
            uv=image_to_strip(xy,entry['geometry'])
            support=cv2.remap(arrays['valid_mask'].astype(np.float32),uv[...,0].astype(np.float32),
                uv[...,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            coverage=(support>=1-1e-6).astype(np.uint8)
        else:
            decoded=dict(segments=[],reason='width_unavailable',identity_confirmed=False);decode_ms=0.
        score=score_decoded(label,quality,decoded,coverage,sample['image_shape_hw'])
        rows.append(dict(sample_id=sample['sample_id'],split=sample['split'],group_id=sample['group_id'],
            cohort=record['cohort'],prediction_status='loaded' if pred['available'] else 'width_unavailable',
            decoded=decoded,independent_width=entry['width'],**score,
            timing=dict(prepare_ms=entry['preparation_ms'],forward_ms=pred.get('forward_ms'),decode_ms=decode_ms)))
    result=dict(arm=arm,rows=rows,summaries=aggregate(rows),synthetic_pair_pass_fraction=exports['synthetic_pair_pass_fraction'],
        protocol_sha256=sha(output/'protocol.json'),checkpoint_sha256=sha(run/'checkpoint.pt'),
        exports_sha256=sha(run/'exports.json'),
        identity_confirmed=False,event_timing_validated=False,force_command=None)
    write_json(run/'evaluation.json',result)
    for row in rows:
        print(json.dumps(dict(arm=arm,sample=row['sample_id'],coverage=row['metrics']['downstream']['covered_fraction'],
            precision=row['metrics']['downstream']['prediction_precision_at_5px'],reason=row['decoded']['reason'])),flush=True)


def gate(output):
    protocol=verify(output)
    result=read(output/'single_identity/evaluation.json')
    if (result['arm']!='single_identity' or result['protocol_sha256']!=sha(output/'protocol.json')
            or result['checkpoint_sha256']!=sha(output/'single_identity/checkpoint.pt')
            or result['exports_sha256']!=sha(output/'single_identity/exports.json')
            or len(result['rows'])!=1 or result['rows'][0]['sample_id']!=SAMPLE):
        raise ValueError('Gate requires current single_identity evaluation on the one training image')
    real=result['rows'][0]['metrics']['downstream']['covered_fraction']
    precision=result['rows'][0]['metrics']['downstream']['prediction_precision_at_5px']
    synthetic=result['synthetic_pair_pass_fraction']
    passed=bool(real is not None and real>=.9 and precision is not None and precision>=.9 and synthetic>=.9)
    write_json(output/'gate.json',dict(real_downstream_coverage=real,real_downstream_precision=precision,synthetic_pair_pass_fraction=synthetic,
        allow_development_run=passed,thresholds=protocol['conditional_development_gate'],
        protocol_sha256=sha(output/'protocol.json'),
        evaluation_sha256=sha(output/'single_identity/evaluation.json'),
        checkpoint_sha256=sha(output/'single_identity/checkpoint.pt'),
        warning='Not a deployment gate: development holdout may be evaluated only after mechanism checks',
        action='continue source-isolated development evaluation' if passed else 'stop expansion and report mechanism failure'))
    print(json.dumps(dict(allow_development_run=passed,real_coverage=real,synthetic_pair_pass_fraction=synthetic)))


def oracle(output):
    """GT distribution input, clearly separate from model performance."""
    verify(output)
    from .evaluate_centerline_multiscale_v9 import _dataset,score_decoded
    from .v11_path_decode import decode_path
    index,cached=load_cache(output)
    _,records,_=_dataset(index['manifest'])
    entry,arrays=cached[0];record=records[0]
    decoded=decode_path(arrays['target_distribution'],arrays['slope_target'],entry['geometry'],arrays['valid_mask'])
    score=score_decoded(record['label'],record['quality'],decoded,None,record['sample']['image_shape_hw'])
    write_json(output/'oracle.json',dict(ground_truth_is_input=True,model_performance=False,
        decoded=decoded,**score,protocol_sha256=sha(output/'protocol.json')))
    print(json.dumps(dict(oracle_downstream=score['metrics']['downstream'],model_performance=False)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','prepare','prepare-development','train','export','evaluate','gate','verify','oracle'))
    parser.add_argument('--output',type=Path,default=BASE)
    parser.add_argument('--arm',choices=(*ARMS,'development'))
    args=parser.parse_args()
    if args.stage in ('train','export','evaluate') and args.arm is None: parser.error('--arm required')
    if args.stage=='freeze': freeze(args.output)
    elif args.stage.startswith('prepare'): prepare(args.output,args.stage=='prepare-development')
    elif args.stage=='train': train(args.output,args.arm)
    elif args.stage=='export': export(args.output,args.arm)
    elif args.stage=='evaluate': evaluate(args.output,args.arm)
    elif args.stage=='gate': gate(args.output)
    elif args.stage=='oracle': oracle(args.output)
    else: verify(args.output);print('V11 frozen sources/data verified')


if __name__=='__main__': main()
