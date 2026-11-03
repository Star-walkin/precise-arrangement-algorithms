"""One new 0.1 auxiliary-weight arm; V14 0/0.5 remain explicit references.

No old checkpoint or protocol is rewritten. Fresh single/development budgets,
inputs, ordering, optimizer, model, decoding and gates match frozen V14. A
failed training gate prohibits loading holdout arrays for every arm.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import time

import numpy as np

from .v11_path_experiment import read, sha, write_json, write_npz, TARGET_KEYS
from . import v14_joint_experiment as v14


LAB = Path(__file__).resolve().parent
BASE = LAB/'outputs/v15_aux_weight'
NEW_ARM = 'joint_low'
REFERENCE_ARMS = ('center_only', 'joint')
ARMS = ('center_only', 'joint', NEW_ARM)
HOLDOUT_ARMS = ('center_only', NEW_ARM)
AUDIT_SAMPLES = dict(single=v14.SAMPLE, train='dual_20260924_111143__right__f00190')
AUDIT_RTOL, AUDIT_ATOL = 1e-6, 1e-7
MATCH_FIELDS = ('initial_state_sha256', 'sequence_sha256', 'input_sha256', 'cache_sha256',
                'steps', 'optimizer', 'learning_rate', 'torch_version', 'numpy_version')


def _new_root(output):
    output = Path(output).resolve()
    old_root = v14.BASE.resolve()
    if output == old_root or old_root in output.parents:
        raise ValueError('V15 output must not be inside the frozen V14 tree')
    return output


def location(output, phase, arm):
    if phase not in ('single', 'train', 'holdout') or arm not in ARMS:
        raise ValueError('unknown phase or arm')
    return Path(output)/('single' if phase == 'single' else 'development')/arm


def assert_matched_reports(reports):
    """Equality excludes arm/auxiliary weight, the only planned intervention."""
    if set(reports) != set(ARMS):
        raise ValueError('must compare the new arm and both explicit references')
    baseline = reports['center_only']
    for arm in ('joint', NEW_ARM):
        for key in MATCH_FIELDS:
            if reports[arm][key] != baseline[key]:
                raise ValueError('unmatched '+arm+' report: '+key)


def freeze(output):
    output = _new_root(output)
    if output.exists():
        raise FileExistsError('use a new output directory; never replace a trial')
    old = v14.verify(v14.BASE)
    if (old['single_steps'], old['development_steps']) != (1200, 2400):
        raise ValueError('unexpected reference fixed budgets')
    references = {}
    artifacts = dict(old['artifact_snapshot'])
    artifacts[str(v14.BASE/'protocol.json')] = sha(v14.BASE/'protocol.json')
    for phase in ('single', 'train', 'holdout'):
        artifacts[str(v14.BASE/(phase+'_manifest.json'))] = sha(v14.BASE/(phase+'_manifest.json'))
    for phase in ('single', 'train'):
        references[phase] = {}
        old_reports = {}
        for arm in REFERENCE_ARMS:
            folder = v14.location(v14.BASE, phase, arm)
            old_reports[arm] = v14.checked_report(v14.BASE, phase, arm)
            files = {name: folder/name for name in ('report.json', 'checkpoint.pt',
                     'exports_'+phase+'.json', 'evaluation_'+phase+'.json')}
            for path in files.values():
                artifacts[str(path)] = sha(path)
            if arm == 'center_only':
                exported = read(files['exports_'+phase+'.json'])
                prediction = next(r for r in exported['records'] if r['sample_id'] == AUDIT_SAMPLES[phase])
                if sha(prediction['path']) != prediction['sha256']:
                    raise ValueError('reuse-audit reference prediction changed')
                artifacts[prediction['path']] = prediction['sha256']
            evaluation = read(files['evaluation_'+phase+'.json'])
            if (evaluation['phase'] != phase or evaluation['arm'] != arm
                    or evaluation['protocol_sha256'] != sha(v14.BASE/'protocol.json')
                    or evaluation['checkpoint_sha256'] != sha(files['checkpoint.pt'])
                    or evaluation['exports_sha256'] != sha(files['exports_'+phase+'.json'])):
                raise ValueError('reference evaluation provenance mismatch')
            references[phase][arm] = dict(origin='reused_frozen_V14_not_new_training',
                auxiliary_weight=old['edge_weight'][arm],
                files={name: dict(path=str(path), sha256=sha(path)) for name, path in files.items()})
        for key in MATCH_FIELDS:
            if old_reports['center_only'][key] != old_reports['joint'][key]:
                raise ValueError('old reference arms not matched: '+key)
    manifest = read(v14.BASE/'train_manifest.json')
    if '20260925' in json.dumps(manifest):
        raise ValueError('reserved blind sources prohibited')
    sources = dict(old['source_snapshot'])
    sources[str(Path(__file__).resolve())] = sha(__file__)
    # Copy every training/gate constant, then change only the planned arm and
    # provenance declarations. Reference protocols remain separate artifacts.
    protocol = dict(old)
    protocol.update(version=15, frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        arms=list(ARMS), newly_trained_arms=[NEW_ARM], reference_arms=list(REFERENCE_ARMS),
        edge_weight=dict(center_only=0., joint=.5, joint_low=.1),
        objective='unchanged V11 center CE + 0.1 slope; new arm adds 0.1*two-edge column CE',
        intervention='only auxiliary edge loss weight: 0.1; no threshold/data/model/decoder changes',
        initialization='new joint_low fresh from same V14 seed separately for single and development; no warm start',
        reference_root=str(v14.BASE.resolve()),
        reference_protocol_sha256=sha(v14.BASE/'protocol.json'), references=references,
        reference_results_are_new_training=False,
        holdout_arms=list(HOLDOUT_ARMS),
        holdout_rule='after V15 train gate: only joint_low and reused center_only; joint 0.5 remains historical training context',
        reuse_audit=dict(required_before_gate=True, reference_arm='center_only', samples=AUDIT_SAMPLES.copy(),
            probability_and_slope_rtol=AUDIT_RTOL, probability_and_slope_atol=AUDIT_ATOL,
            purpose='two exact-reference repeat forwards validate computational reuse; not new accuracy scores'),
        artifact_snapshot=artifacts, source_snapshot=sources)
    write_json(output/'protocol.json', protocol)
    for phase in ('single', 'train', 'holdout'):
        write_json(output/(phase+'_manifest.json'), read(v14.BASE/(phase+'_manifest.json')))
    print(json.dumps(dict(protocol=str(output/'protocol.json'), sha256=sha(output/'protocol.json'))))


def verify(output):
    output = _new_root(output)
    old = v14.verify(v14.BASE)
    p = read(output/'protocol.json')
    if (p['version'] != 15 or p['newly_trained_arms'] != [NEW_ARM]
            or p['reference_protocol_sha256'] != sha(v14.BASE/'protocol.json')
            or Path(p['reference_root']).resolve() != v14.BASE.resolve()
            or p['edge_weight'] != dict(center_only=0., joint=.5, joint_low=.1)
            or p['holdout_arms'] != list(HOLDOUT_ARMS)
            or p['reuse_audit']['samples'] != AUDIT_SAMPLES
            or p['reuse_audit']['probability_and_slope_rtol'] != AUDIT_RTOL
            or p['reuse_audit']['probability_and_slope_atol'] != AUDIT_ATOL):
        raise ValueError('wrong V15 intervention/provenance')
    for category in ('artifact_snapshot', 'source_snapshot'):
        for path, expected in p[category].items():
            if sha(path) != expected:
                raise ValueError('frozen V15 source/reference/data changed: '+path)
    for key in ('seed', 'base_channels', 'cpu_threads', 'single_steps', 'development_steps',
                'learning_rate', 'batch_size', 'optimizer', 'synthetic_train_seed_base',
                'synthetic_test_seeds', 'single_sample', 'gate'):
        if p[key] != old[key]:
            raise ValueError('V15 must retain V14 '+key)
    for phase in ('single', 'train', 'holdout'):
        if read(output/(phase+'_manifest.json')) != read(v14.BASE/(phase+'_manifest.json')):
            raise ValueError('V15 selected data differs from frozen V14')
    return p


def checked_report(output, phase, arm):
    p = read(Path(output)/'protocol.json')
    if arm in REFERENCE_ARMS:
        return v14.checked_report(Path(p['reference_root']), phase, arm)
    if arm != NEW_ARM:
        raise ValueError('unknown arm')
    folder = location(output, phase, arm)
    r = read(folder/'report.json')
    expected_phase = 'single' if phase == 'single' else 'train'
    if (r['phase'] != expected_phase or r['arm'] != arm
            or r['protocol_sha256'] != sha(Path(output)/'protocol.json')
            or r['checkpoint_sha256'] != sha(folder/'checkpoint.pt')
            or r['steps'] != p['single_steps' if phase == 'single' else 'development_steps']
            or r['edge_weight'] != .1):
        raise ValueError('wrong new-arm final training report')
    return r


def checkpoint_source(output, phase, arm):
    """Checkpoint keeps its ORIGINAL protocol; V15 only binds its provenance."""
    if phase == 'holdout' and arm not in HOLDOUT_ARMS:
        raise ValueError('only center_only and joint_low may enter V15 holdout')
    p = read(Path(output)/'protocol.json')
    r = checked_report(output, phase, arm)
    old = arm in REFERENCE_ARMS
    root = Path(p['reference_root']) if old else Path(output)
    folder = location(root, phase, arm)
    return dict(origin='reused_frozen_V14_reference' if old else 'new_V15_joint_low_training',
        checkpoint=str(folder/'checkpoint.pt'), checkpoint_sha256=r['checkpoint_sha256'],
        checkpoint_protocol=str(root/'protocol.json'),
        checkpoint_protocol_sha256=sha(root/'protocol.json'),
        training_report=str(folder/'report.json'), training_report_sha256=sha(folder/'report.json'),
        training_phase='single' if phase == 'single' else 'train', arm=arm,
        steps=r['steps'], auxiliary_weight=p['edge_weight'][arm])


def evaluation_source(output, phase, arm):
    if phase == 'holdout' and arm not in HOLDOUT_ARMS:
        raise ValueError('joint 0.5 has historical training context only, no V15 holdout')
    p = read(Path(output)/'protocol.json')
    reused = phase != 'holdout' and arm in REFERENCE_ARMS
    root = Path(p['reference_root']) if reused else Path(output)
    path = location(root, phase, arm)/('evaluation_'+phase+'.json')
    return dict(origin='reused_V14_evaluation_not_rerun' if reused else 'new_V15_evaluation',
                path=str(path), sha256=sha(path))


def load_inputs(output, phase, with_targets=False):
    if phase == 'holdout':
        require_gate(output, 'train')  # Before the frozen loader opens any arrays.
    if with_targets and phase not in ('single', 'train'):
        raise ValueError('holdout can never supply training targets')
    return v14.load_inputs(Path(output), phase, with_targets)


def compare_reused_array(actual, saved):
    """Fixed tight tolerance; mismatches are reported, never tuned away."""
    actual, saved = np.asarray(actual), np.asarray(saved)
    same_shape = actual.shape == saved.shape
    finite = bool(same_shape and actual.size and np.isfinite(actual).all() and np.isfinite(saved).all())
    return dict(actual_shape=list(actual.shape), saved_shape=list(saved.shape),
        finite_matching_shapes=finite,
        bit_equal=bool(finite and np.array_equal(actual, saved)),
        passed=bool(finite and np.allclose(actual, saved, rtol=AUDIT_RTOL, atol=AUDIT_ATOL, equal_nan=False)),
        maximum_absolute_difference=float(np.max(abs(actual.astype(float)-saved.astype(float)))) if finite else None,
        rtol=AUDIT_RTOL, atol=AUDIT_ATOL)


def reuse_audit(output):
    """Re-forward two TRAINING-scope saved references; never load holdout."""
    output = _new_root(output); p = verify(output)
    if (output/'reuse_audit.json').exists():
        raise FileExistsError('cannot overwrite a reuse audit')
    import torch
    from .v14_joint_boundary_model import build_model
    from .v11_path_model import mask_normalized_probabilities
    torch.set_num_threads(p['cpu_threads'])
    index = {entry['sample_id']:entry for entry in read(v14.V11/'development_cache/index.json')['entries']}
    rows = []
    for phase, sample_id in AUDIT_SAMPLES.items():
        source = checkpoint_source(output, phase, 'center_only')
        report = checked_report(output, phase, 'center_only')
        samples = {s['sample_id']:s for s in read(output/(phase+'_manifest.json'))['samples']}
        sample, entry = samples[sample_id], index[sample_id]
        if sample['split'] != 'train' or entry['split'] != 'train' or not entry['available']:
            raise ValueError('reuse audit is restricted to available training samples')
        for key in ('image_sha256', 'annotation_sha256', 'quality_sha256', 'split', 'group_id'):
            if entry[key] != sample[key]:
                raise ValueError('reuse audit cache source mismatch: '+key)
        if sha(entry['cache']) != entry['cache_sha256']:
            raise ValueError('reuse audit cached input changed')
        with np.load(entry['cache'], allow_pickle=False) as arrays:
            # Do not read GT/annotation masks for reference inference.
            image_input, image_valid = arrays['input'].copy(), arrays['valid_mask'].copy()
        if hashlib.sha256(image_input.tobytes()).hexdigest() != entry['inference_input_sha256']:
            raise ValueError('reuse audit input tensor mismatch')
        reference_folder = Path(source['checkpoint']).parent
        exports = read(reference_folder/('exports_'+phase+'.json'))
        if (exports['protocol_sha256'] != source['checkpoint_protocol_sha256']
                or exports['checkpoint_sha256'] != source['checkpoint_sha256']
                or exports['training_report_sha256'] != source['training_report_sha256']):
            raise ValueError('reference export provenance mismatch')
        saved_record = next(r for r in exports['records'] if r['sample_id'] == sample_id)
        if (sha(saved_record['path']) != saved_record['sha256']
                or saved_record['input_sha256'] != entry['inference_input_sha256']):
            raise ValueError('saved reference prediction/input changed')
        checkpoint = torch.load(source['checkpoint'], map_location='cpu', weights_only=True)
        if (checkpoint['protocol_sha256'] != source['checkpoint_protocol_sha256']
                or checkpoint['phase'] != source['training_phase'] or checkpoint['arm'] != 'center_only'
                or checkpoint['steps'] != source['steps']):
            raise ValueError('reuse checkpoint original provenance mismatch')
        model = build_model(p['seed'], p['base_channels']); model.load_state_dict(checkpoint['state_dict']); model.eval()
        before = {key:value.clone() for key,value in model.state_dict().items()}
        started = time.perf_counter()
        with torch.inference_mode():
            out = model(torch.from_numpy(image_input[None]).float())
            actual_probability = mask_normalized_probabilities(out, torch.from_numpy(image_valid[None]))[0].numpy()
            actual_slope = out['slope'][0].numpy()
        with np.load(saved_record['path'], allow_pickle=False) as saved:
            probability = compare_reused_array(actual_probability, saved['probability'])
            slope = compare_reused_array(actual_slope, saved['slope'])
        unchanged = all(torch.equal(value, model.state_dict()[key]) for key,value in before.items())
        unchanged = unchanged and sha(source['checkpoint']) == source['checkpoint_sha256']
        versions_match = report['torch_version'] == str(torch.__version__) and report['numpy_version'] == str(np.__version__)
        row = dict(phase=phase, sample_id=sample_id, split='train', checkpoint_source=source,
            cache=entry['cache'], cache_sha256=entry['cache_sha256'], input_sha256=entry['inference_input_sha256'],
            saved_prediction=saved_record['path'], saved_prediction_sha256=saved_record['sha256'],
            probability=probability, slope=slope, checkpoint_weights_unchanged=unchanged,
            runtime_versions_match=versions_match,
            passed=bool(probability['passed'] and slope['passed'] and unchanged and versions_match),
            forward_and_comparison_ms=(time.perf_counter()-started)*1000)
        rows.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
    record = dict(kind='V15 reference-reuse repeat-forward audit, not new accuracy evaluation',
        protocol_sha256=sha(output/'protocol.json'), rows=rows, passed=all(row['passed'] for row in rows),
        cpu_threads=p['cpu_threads'], torch_version=str(torch.__version__), numpy_version=str(np.__version__),
        no_training=True, no_optimizer=True, no_holdout=True, gt_used_for_inference=False,
        mismatch_policy='stop gate on unexplained mismatch; do not retune tolerance or relabel references')
    write_json(output/'reuse_audit.json', record); verify(output)
    print(json.dumps(dict(reuse_audit_passed=record['passed'], path=str(output/'reuse_audit.json'))))


def require_reuse_audit(output):
    output = Path(output); record = read(output/'reuse_audit.json')
    p = read(output/'protocol.json')
    if record['protocol_sha256'] != sha(output/'protocol.json') or record['cpu_threads'] != p['cpu_threads']:
        raise ValueError('foreign reference reuse audit')
    if len(record['rows']) != 2 or {r['phase']:r['sample_id'] for r in record['rows']} != AUDIT_SAMPLES:
        raise ValueError('wrong reference reuse audit sample scope')
    passed = True
    for row in record['rows']:
        if (row['split'] != 'train'
                or row['checkpoint_source'] != checkpoint_source(output, row['phase'], 'center_only')
                or sha(row['cache']) != row['cache_sha256']
                or sha(row['saved_prediction']) != row['saved_prediction_sha256']):
            raise ValueError('reference reuse audit source changed')
        expected = row['probability']['passed'] and row['slope']['passed'] and row['checkpoint_weights_unchanged'] and row['runtime_versions_match']
        if row['passed'] != expected:
            raise ValueError('inconsistent reference audit decision')
        passed = passed and expected
    if record['passed'] != passed or not passed:
        raise ValueError('reference reuse audit failed; no mechanism gate allowed')


def train(output, phase, arm=NEW_ARM):
    if arm != NEW_ARM or phase not in ('single', 'train'):
        raise ValueError('only joint_low single/train may be trained in V15')
    output = _new_root(output)
    p = verify(output)
    if phase == 'train':
        require_gate(output, 'single')
    folder = location(output, phase, arm)
    if folder.exists():
        raise FileExistsError('never overwrite, resume or warm-start training')
    import torch
    from .v14_joint_boundary_model import build_model, joint_path_loss
    from .v11_identity_probe import make_pair
    from .centerline_multiscale_v9 import _state_dict_sha256
    torch.set_num_threads(p['cpu_threads'])
    real = load_inputs(output, phase, True)
    model = build_model(p['seed'], p['base_channels'])
    initial = _state_dict_sha256(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=p['learning_rate'], betas=tuple(p['optimizer']['betas']),
        eps=p['optimizer']['eps'], weight_decay=0., amsgrad=False)
    steps = p['single_steps'] if phase == 'single' else p['development_steps']
    sequence = []; log = []; last_real = None; started = time.perf_counter(); model.train()
    for step in range(steps):
        synthetic = step % 2 == 1
        if synthetic:
            pair = make_pair(p['synthetic_train_seed_base']+step//2)
            x = pair['inputs']
            batch = {k: torch.from_numpy(np.stack([t[k] for t in pair['targets']])).float() for k in TARGET_KEYS}
            n, _, h, w = x.shape
            batch.update(edge_target_distribution=torch.zeros((n, 2, h, w)),
                edge_row_valid=torch.zeros((n, 2, h, w)), edge_column_valid=torch.zeros((n, 2, w)))
            sid = 'synthetic:'+str(p['synthetic_train_seed_base']+step//2)
        else:
            e, a = real[(step//2) % len(real)]; x = np.repeat(a['input'][None], 2, axis=0)
            batch = {k: torch.from_numpy(np.repeat(a[k][None], 2, axis=0)).float() for k in
                (*TARGET_KEYS, 'edge_target_distribution', 'edge_row_valid', 'edge_column_valid')}
            sid = e['sample_id']
        sequence.append(sid); opt.zero_grad(set_to_none=True)
        losses = joint_path_loss(model(torch.from_numpy(x).float()), batch, edge_weight=p['edge_weight'][arm])
        if not torch.isfinite(losses['loss']):
            raise RuntimeError('nonfinite loss')
        losses['loss'].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), p['optimizer']['clip_norm']); opt.step()
        if not synthetic:
            last_real = dict(step=step+1, primary=float(losses['primary_path_loss'].detach()),
                edge=float(losses['edge_ce_loss'].detach()))
        if step == 0 or (step+1) % 100 == 0:
            row = dict(step=step+1, loss=float(losses['loss'].detach()), primary=float(losses['primary_path_loss'].detach()),
                edge=float(losses['edge_ce_loss'].detach()), synthetic=synthetic, last_real=last_real,
                seconds=time.perf_counter()-started)
            log.append(row); print(json.dumps(dict(phase=phase, arm=arm, **row)), flush=True)
    folder.mkdir(parents=True)
    checkpoint = folder/'checkpoint.pt'
    with checkpoint.open('xb') as stream:
        torch.save(dict(state_dict=model.state_dict(), phase=phase, arm=arm, steps=steps,
            protocol_sha256=sha(output/'protocol.json')), stream)
    write_json(folder/'report.json', dict(phase=phase, arm=arm, steps=steps, edge_weight=.1,
        origin='new_V15_training_not_a_relabelled_reference',
        initial_state_sha256=initial, final_state_sha256=_state_dict_sha256(model.state_dict()),
        sequence_sha256=hashlib.sha256(json.dumps(sequence).encode()).hexdigest(),
        input_sha256={e['sample_id']: e['inference_input_sha256'] for e, a in real},
        cache_sha256={e['sample_id']: e['cache_sha256'] for e, a in real},
        train_ids=[e['sample_id'] for e, a in real], optimizer=p['optimizer'], learning_rate=p['learning_rate'],
        loss_log=log, torch_version=str(torch.__version__), numpy_version=str(np.__version__),
        parameters=sum(v.numel() for v in model.parameters()), elapsed_seconds=time.perf_counter()-started,
        checkpoint_sha256=sha(checkpoint), protocol_sha256=sha(output/'protocol.json'), hardware_access=False))
    verify(output)


def _check_new_evaluation_scope(phase, arm):
    if phase not in ('single', 'train', 'holdout') or arm not in ARMS:
        raise ValueError('unknown export/evaluation phase or arm')
    if phase != 'holdout' and arm != NEW_ARM:
        raise ValueError('V14 single/train reference evaluations are reused, not rerun or overwritten')
    if phase == 'holdout' and arm not in HOLDOUT_ARMS:
        raise ValueError('joint 0.5 is not authorized for V15 holdout')


def export(output, phase, arm=NEW_ARM):
    _check_new_evaluation_scope(phase, arm)
    output = _new_root(output); p = verify(output)
    if phase == 'holdout':
        require_gate(output, 'train')
    if phase == 'train':
        require_gate(output, 'single')
    source = checkpoint_source(output, phase, arm)
    folder = location(output, phase, arm); destination = folder/('predictions_'+phase)
    if destination.exists():
        raise FileExistsError('cannot overwrite inference')
    import torch
    from .v14_joint_boundary_model import build_model, image_masked_edge_probabilities
    from .v11_path_model import mask_normalized_probabilities
    from .v11_identity_probe import make_pair, assess_pair
    torch.set_num_threads(p['cpu_threads'])
    checkpoint = torch.load(source['checkpoint'], map_location='cpu', weights_only=True)
    if (checkpoint['protocol_sha256'] != source['checkpoint_protocol_sha256']
            or checkpoint['arm'] != arm or checkpoint['phase'] != source['training_phase']
            or checkpoint['steps'] != source['steps']):
        raise ValueError('checkpoint original provenance mismatch')
    model = build_model(p['seed'], p['base_channels']); model.load_state_dict(checkpoint['state_dict']); model.eval()
    rows = []; probes = []
    with torch.inference_mode():
        for e, a in load_inputs(output, phase):
            started = time.perf_counter(); out = model(torch.from_numpy(a['input'][None]).float())
            valid = torch.from_numpy(a['valid_mask'][None])
            center = mask_normalized_probabilities(out, valid)[0].numpy()
            edges = image_masked_edge_probabilities(out, valid)[0].numpy()
            elapsed = (time.perf_counter()-started)*1000
            path = destination/(e['sample_id']+'.npz')
            write_npz(path, probability=center, slope=out['slope'][0].numpy(), edge_probability=edges)
            rows.append(dict(sample_id=e['sample_id'], path=str(path), sha256=sha(path),
                input_sha256=e['inference_input_sha256'], forward_ms=elapsed))
        for seed in p['synthetic_test_seeds']:
            pair = make_pair(seed, test=True); out = model(torch.from_numpy(pair['inputs']))
            probability = mask_normalized_probabilities(out, torch.from_numpy(pair['inputs'][:, 7])).numpy()
            probes.append(dict(seed=seed, **assess_pair(probability, pair)))
    write_json(folder/('exports_'+phase+'.json'), dict(phase=phase, arm=arm, records=rows,
        synthetic_probes=probes, synthetic_pair_pass_fraction=float(np.mean([r['both_queries_pass'] for r in probes])),
        protocol_sha256=sha(output/'protocol.json'), checkpoint_source=source,
        checkpoint_sha256=source['checkpoint_sha256'],
        training_report_sha256=source['training_report_sha256'], gt_used_for_inference=False))
    verify(output)


def evaluate(output, phase, arm=NEW_ARM):
    _check_new_evaluation_scope(phase, arm)
    output = _new_root(output); verify(output)
    if phase == 'holdout':
        require_gate(output, 'train')
    if phase == 'train':
        require_gate(output, 'single')
    from .evaluate_centerline_multiscale_v9 import _dataset, score_decoded, aggregate
    from .v11_path_decode import decode_path
    from .v13_boundary_targets import targets_for_annotation
    from .v14_boundary_metrics import score_boundaries, aggregate_boundary_scores
    from .v14_reference_support import boundary_reference_support
    folder = location(output, phase, arm); source = checkpoint_source(output, phase, arm)
    exports = read(folder/('exports_'+phase+'.json'))
    if (exports['phase'] != phase or exports['arm'] != arm
            or exports['protocol_sha256'] != sha(output/'protocol.json')
            or exports['checkpoint_source'] != source
            or exports['checkpoint_sha256'] != source['checkpoint_sha256']
            or exports['training_report_sha256'] != source['training_report_sha256']):
        raise ValueError('wrong inference source/stage')
    cache = {e['sample_id']: (e, a) for e, a in load_inputs(output, phase)}
    predictions = {r['sample_id']: r for r in exports['records']}
    if len(predictions) != len(exports['records']) or set(predictions) != set(cache):
        raise ValueError('predictions must exactly cover selected stage without duplicates')
    _, records, _ = _dataset(output/(phase+'_manifest.json'))
    old_support = {r['sample_id']: r['coverage_support'] for r in read(v14.V11/'development/evaluation.json')['rows']}
    rows = []
    for record in records:
        s, l, q = record['sample'], record['label'], record['quality']; name = s['sample_id']
        e, a = cache[name]; pred = predictions[name]
        if sha(pred['path']) != pred['sha256'] or pred['input_sha256'] != e['inference_input_sha256']:
            raise ValueError('prediction/input mismatch')
        with np.load(pred['path'], allow_pickle=False) as ar:
            decoded = decode_path(ar['probability'], ar['slope'], e['geometry'], a['valid_mask'])
            targets = targets_for_annotation(e['geometry'], l, q, a['valid_mask'])
            boundary = score_boundaries(ar['edge_probability'], a['valid_mask'],
                v14.seed_downstream_targets(targets, e['geometry']), e['geometry'])
        support = boundary_reference_support(l, q, e['geometry'], a['valid_mask'], targets)
        score = score_decoded(l, q, decoded, None, s['image_shape_hw'])
        score['coverage_support'] = old_support[name]
        rows.append(dict(sample_id=name, split=s['split'], group_id=s['group_id'], cohort=record['cohort'],
            prediction_status='loaded', independent_width=e['width'], decoded=decoded, **score,
            boundary=boundary, boundary_reference_support=support, target_diagnostics=targets['diagnostics']))
    result = dict(phase=phase, arm=arm, rows=rows, summaries=aggregate(rows),
        boundary_summary=aggregate_boundary_scores([r['boundary'] for r in rows]),
        synthetic_pair_pass_fraction=exports['synthetic_pair_pass_fraction'],
        protocol_sha256=sha(output/'protocol.json'), checkpoint_source=source,
        exports_sha256=sha(folder/('exports_'+phase+'.json')), checkpoint_sha256=source['checkpoint_sha256'],
        event_timing_validated=False, force_command=None)
    write_json(folder/('evaluation_'+phase+'.json'), result); verify(output)
    print(json.dumps(dict(phase=phase, arm=arm,
        center=[r for r in result['summaries'] if r['cohort'] == 'all' and r['scope'] == 'downstream' and r['samples']],
        edge_hits={k: v['hit_5px_fraction'] for k, v in result['boundary_summary']['edges'].items()},
        width_relative=result['boundary_summary']['width']['mean_absolute_relative_error']), ensure_ascii=False))


def matched_gate_decision(values, reports, rules):
    """Exact V14 rules on baseline/new arm; 0.5 reference is context only."""
    assert_matched_reports(reports)
    if set(values) != set(ARMS):
        raise ValueError('three arm evaluations required for transparent context')
    def center(arm):
        return next(s for s in values[arm]['summaries']
                    if s['split'] == 'train' and s['cohort'] == 'all' and s['scope'] == 'downstream')
    checks = {}
    for metric in ('covered_fraction', 'prediction_precision_at_5px'):
        b, j = center('center_only')[metric], center(NEW_ARM)[metric]
        checks['center_'+metric] = b is not None and j is not None and min(b, j) >= rules['center_min'] and j >= b-rules['center_max_drop']
    checks['synthetic'] = all(values[a]['synthetic_pair_pass_fraction'] >= rules['synthetic_min']
                              for a in ('center_only', NEW_ARM))
    joint = values[NEW_ARM]['boundary_summary']
    checks['each_edge_hit'] = all(e['hit_5px_fraction'] is not None and e['hit_5px_fraction'] >= rules['each_edge_hit_min']
                                  for e in joint['edges'].values())
    width = joint['width']; pair = joint['joint']
    checks['width'] = width['mean_absolute_relative_error'] is not None and width['mean_absolute_relative_error'] <= rules['width_mean_relative_max']
    checks['width_prediction_support'] = width['trusted_columns'] > 0 and width['predicted_trusted_columns']/width['trusted_columns'] >= rules['width_prediction_fraction_min']
    checks['crossing'] = pair['crossing_fraction_among_available'] is not None and pair['crossing_fraction_among_available'] <= rules['crossing_max']
    checks['bracket'] = pair['strict_center_bracket_fraction'] is not None and pair['strict_center_bracket_fraction'] >= rules['strict_human_center_bracket_min']
    checks['reference_support'] = v14.reference_support_pass(values[NEW_ARM]['rows'], rules['each_edge_reference_support_min'])
    historical = values['joint']['boundary_summary']
    historical_context = dict(
        edge_hits={key:value['hit_5px_fraction'] for key,value in historical['edges'].items()},
        width_relative=historical['width']['mean_absolute_relative_error'],
        crossing=historical['joint']['crossing_fraction_among_available'],
        bracket=historical['joint']['strict_center_bracket_fraction'])
    return dict(passed=all(checks.values()), checks=checks, center={a: center(a) for a in ARMS},
        edge_hits={k: v['hit_5px_fraction'] for k, v in joint['edges'].items()},
        width_relative=width['mean_absolute_relative_error'], crossing=pair['crossing_fraction_among_available'],
        bracket=pair['strict_center_bracket_fraction'],
        synthetic={a: values[a]['synthetic_pair_pass_fraction'] for a in ARMS},
        comparison='new joint_low 0.1 versus reused center_only 0; reused joint 0.5 is context only',
        context_joint_05_boundary_summary=historical_context, deployment_allowed=False)


def gate_decision(output, phase):
    if phase not in ('single', 'train'):
        raise ValueError('no holdout-tuned gate')
    output = _new_root(output); p = verify(output)
    require_reuse_audit(output)
    values = {}; reports = {}
    for arm in ARMS:
        source = evaluation_source(output, phase, arm)
        value = read(source['path']); cp = checkpoint_source(output, phase, arm)
        own = arm == NEW_ARM
        expected_protocol = sha(output/'protocol.json') if own else p['reference_protocol_sha256']
        eval_folder = Path(source['path']).parent
        if (value['phase'] != phase or value['arm'] != arm
                or value['protocol_sha256'] != expected_protocol
                or value['checkpoint_sha256'] != cp['checkpoint_sha256']
                or value['exports_sha256'] != sha(eval_folder/('exports_'+phase+'.json'))
                or (own and value['checkpoint_source'] != cp)):
            raise ValueError('stale/foreign gate evaluation')
        values[arm] = value; reports[arm] = checked_report(output, phase, arm)
    return matched_gate_decision(values, reports, p['gate'])


def require_gate(output, phase):
    output = Path(output)
    record = read(output/('gate_'+phase+'.json'))
    if record['protocol_sha256'] != sha(output/'protocol.json'):
        raise ValueError('foreign V15 gate')
    if record['reuse_audit_sha256'] != sha(output/'reuse_audit.json'):
        raise ValueError('reuse audit changed after gate')
    for arm in ARMS:
        if record['evaluation_sources'][arm] != evaluation_source(output, phase, arm):
            raise ValueError('stale V15 gate source')
    calculated = gate_decision(output, phase)
    if record['decision'] != calculated or not calculated['passed']:
        raise ValueError('V15 mechanism gate did not pass; downstream data stays locked')


def gate(output, phase):
    output = _new_root(output)
    decision = gate_decision(output, phase)
    result = dict(protocol_sha256=sha(output/'protocol.json'), decision=decision,
        reuse_audit_sha256=sha(output/'reuse_audit.json'),
        evaluation_sources={a: evaluation_source(output, phase, a) for a in ARMS},
        report_sources={a: checkpoint_source(output, phase, a) for a in ARMS})
    write_json(output/('gate_'+phase+'.json'), result)
    print(json.dumps(result, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'verify', 'reuse-audit', 'train', 'export', 'evaluate', 'gate'))
    parser.add_argument('--phase', choices=('single', 'train', 'holdout'), default='single')
    parser.add_argument('--arm', choices=ARMS, default=NEW_ARM)
    parser.add_argument('--output', type=Path, default=BASE)
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze(args.output)
    elif args.stage == 'verify':
        verify(args.output); print('V15 frozen sources, matched references and data verified')
    elif args.stage == 'reuse-audit':
        reuse_audit(args.output)
    elif args.stage == 'gate':
        gate(args.output, args.phase)
    else:
        globals()[args.stage](args.output, args.phase, args.arm)


if __name__ == '__main__':
    main()
