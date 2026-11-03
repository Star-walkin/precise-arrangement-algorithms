"""Same-crop query-channel intervention, not a rope-identity evaluation.

Use only the eighteen cached manual inputs after all three exports complete.
Neither inference nor diagnostics loads annotation/quality geometry. Parent
integrity verification may hash those files without interpreting labels. Rules are
bound before any intervention. No training, new crop, new width, or hardware.
"""
from pathlib import Path
import argparse
import json
import math
import time

import numpy as np

from .v24_anchor_trial import BASE as TRIAL, verify as verify_trial
from .v23_auto_seed_chain import read, sha, write, save
from .v11_path_contract import array_sha256

ARMS = ('baseline', 'control', 'anchor_mix')
VARIANTS = ('zero_query', 'query_first_half', 'query_last_half')
RULES = dict(query_channel=4, valid_channel=7, effective_query_threshold=.05,
    effective_columns='column maximum >=0.05 over valid pixels only',
    split='first floor(N/2) effective columns versus remaining columns in increasing strip-column order',
    retention='zero channel4 outside the chosen half-plane; keep original values inside, no new query pixels',
    minimum_effective_columns_for_halves=2,
    insufficient_query='unknown intervention, not empty-image/identity success',
    geometry='unchanged original manual geometry and decoder',
    other_channels='all seven non-query channels bitwise unchanged',
    interpretation='query sensitivity/invariance only; neither is rope identity or visibility accuracy',
    zero_query='out-of-distribution missing-query sensitivity, not a valid target query',
    half_query='subset of same visible free span, not a new rope identity',
    labels_used=False, hardware_access=False, threshold_search=False)


def interventions(input_array):
    x = np.asarray(input_array)
    if x.shape != (8, 256, 512) or not np.issubdtype(x.dtype, np.floating) or not np.isfinite(x).all():
        raise ValueError('Expected finite original 8x256x512 input')
    query = x[4]; valid = x[7] > .5
    if np.any((query < 0) | (query > 1)):
        raise ValueError('Query channel outside expected 0..1 range')
    columns = np.flatnonzero(np.max(np.where(valid, query, 0.), axis=0) >= RULES['effective_query_threshold'])
    results = {}
    for variant in VARIANTS:
        metadata = dict(variant=variant, effective_columns=columns.tolist(), available=False,
            reason='insufficient_effective_query', retained_effective_columns=[], changed_channels=[4])
        if not len(columns) or (variant != 'zero_query' and len(columns) < 2):
            results[variant] = (metadata, None)
            continue
        changed = x.copy()
        if variant == 'zero_query':
            changed[4] = 0
        else:
            middle = len(columns)//2
            cut = int(columns[middle-1])
            keep = np.arange(x.shape[2]) <= cut
            if variant == 'query_last_half': keep = ~keep
            changed[4, :, ~keep] = 0
            metadata['retained_effective_columns'] = columns[keep[columns]].tolist()
            metadata['cut_after_column'] = cut
            if not metadata['retained_effective_columns']:
                results[variant] = (metadata, None)
                continue
        other = [0, 1, 2, 3, 5, 6, 7]
        if not np.array_equal(changed[other], x[other]): raise AssertionError('Non-query channel changed')
        if variant != 'zero_query' and np.any(changed[4] > query): raise AssertionError('Query expanded')
        metadata.update(available=True, reason='same_crop_query_intervention',
            input_array_sha256=array_sha256(changed), nonquery_channels_bitwise_equal=True,
            query_sum=float(changed[4].sum()), original_query_sum=float(query.sum()))
        results[variant] = (metadata, changed)
    return results


def distribution_change(original, intervention, valid_mask, native_step):
    a, b = np.asarray(original, float), np.asarray(intervention, float)
    mask = np.asarray(valid_mask, bool)
    if a.shape != b.shape or a.shape != mask.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Probability/mask mismatch')
    if not math.isfinite(native_step) or native_step <= 0: raise ValueError('Invalid original geometric step')
    columns = mask.any(axis=0)
    if not np.any(columns): return dict(available=False, reason='no_valid_image_columns')
    delta = np.where(mask, np.abs(a-b), 0.)
    pa = np.argmax(np.where(mask, a, -np.inf), axis=0)
    pb = np.argmax(np.where(mask, b, -np.inf), axis=0)
    shift = np.abs(pa[columns]-pb[columns])*native_step
    return dict(available=True, valid_pixels=int(mask.sum()), valid_columns=int(columns.sum()),
        mean_abs_probability_delta_over_valid_pixels=float(delta[mask].mean()),
        mean_total_variation_per_valid_column=float((.5*delta.sum(axis=0))[columns].mean()),
        maximum_abs_probability_delta=float(delta[mask].max()),
        peak_row_shift_mean_native_px=float(shift.mean()),
        peak_row_shift_p95_native_px=float(np.quantile(shift, .95)),
        target_presence_or_identity_assessed=False,
        note='All valid columns, including unlabelled/no-target columns; movement is not accuracy.')


def path_length(decoded):
    total = 0.
    for segment in decoded.get('segments', []):
        p = np.asarray(segment, float)
        if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
            raise ValueError('Invalid saved decoded geometry')
        if len(p) > 1: total += float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    return total


def freeze(trial=TRIAL, output=None):
    trial = Path(trial); output = Path(output) if output else trial/'query_probe'
    if output.exists(): raise FileExistsError('Use a fresh intervention output directory')
    parent = verify_trial(trial)
    index_path = trial/'cache/input_index.json'; index = read(index_path)
    if index['protocol_sha256'] != sha(trial/'protocol.json'):
        raise ValueError('Input protocol mismatch')
    entries = [index['entries'][key] for key in index['evaluation_keys'] if index['entries'][key]['variant'] == 'manual']
    if len(entries) != 18 or len({e['sample_id'] for e in entries}) != 18 or not all(e['available'] for e in entries):
        raise ValueError('Expected eighteen available manual inputs')
    paths = {Path(__file__), trial/'protocol.json', index_path}
    lab = Path(__file__).resolve().parent
    paths.update(lab/name for name in ('v24_anchor_trial.py', 'v11_path_model.py', 'v11_path_decode.py', 'v11_path_contract.py'))
    exports = {}; models = {}
    for arm in ARMS:
        export_path = trial/'exports'/arm/'index.json'
        challenge = trial/'exports'/arm/'challenge.json'
        if not export_path.is_file() or not challenge.is_file():
            raise ValueError('Wait for all three complete original exports before freezing')
        data = read(export_path)
        if data['protocol_sha256'] != sha(trial/'protocol.json') or data['input_index_sha256'] != sha(index_path):
            raise ValueError('Original export binding mismatch')
        if data['model_sha256'] != sha(data['model_path']): raise ValueError('Checkpoint changed')
        paths.update([export_path, challenge, Path(data['model_path'])])
        models[arm] = dict(path=data['model_path'], sha256=data['model_sha256'])
        rows = {r['key']: r for r in data['rows']}
        exports[arm] = {}
        for entry in entries:
            row = rows[entry['key']]
            if row['input_sha256'] != entry['input_sha256'] or not row['available']:
                raise ValueError('Original manual inference input mismatch')
            if sha(row['probability_path']) != row['probability_sha256']:
                raise ValueError('Original probability changed')
            paths.add(Path(row['probability_path']))
            exports[arm][entry['key']] = row
    for entry in entries:
        if sha(entry['input_path']) != entry['input_sha256']: raise ValueError('Manual cache changed')
        paths.add(Path(entry['input_path']))
    if any('20260925' in str(p) for p in paths): raise ValueError('Forbidden blind source')
    protocol = dict(schema='v24_same_crop_query_probe_1', rules=RULES,
        parent_protocol_sha256=sha(trial/'protocol.json'), parent_trial=str(trial),
        cpu_threads=parent['cpu_threads'], models=models, entries=entries, original_exports=exports,
        source_hashes={str(p.resolve()):sha(p) for p in sorted(paths, key=str)},
        future_label_scoring_planned=False, annotation_geometry_used_for_inference=False,
        label_files_may_be_hashed_by_parent_verification=True,
        variants_defined_before_intervention=True, inference_inputs_have_manual_free_query=True)
    write(output/'protocol.json', protocol)
    return protocol


def verify(output):
    p = read(Path(output)/'protocol.json')
    if p['rules'] != RULES: raise ValueError('Query intervention rules changed')
    for path, expected in p['source_hashes'].items():
        if sha(path) != expected: raise ValueError('Frozen source changed: '+path)
    return p


def infer(output):
    output = Path(output); p = verify(output)
    if (output/'predictions.json').exists() or (output/'probabilities').exists():
        raise FileExistsError('Do not replace intervention predictions')
    import torch
    from .v11_path_model import build_model, mask_normalized_probabilities
    from .v11_path_decode import decode_path
    torch.set_num_threads(p['cpu_threads'])
    rows = []
    for arm in ARMS:
        checkpoint = torch.load(p['models'][arm]['path'], map_location='cpu', weights_only=True)
        model = build_model(checkpoint['seed'], base_channels=checkpoint['base_channels'])
        model.load_state_dict(checkpoint['state_dict']); model.eval()
        with torch.inference_mode():
            for number, entry in enumerate(p['entries']):
                with np.load(entry['input_path'], allow_pickle=False) as a:
                    x, valid = a['input'].copy(), a['valid_mask'].copy()
                original = p['original_exports'][arm][entry['key']]
                with np.load(original['probability_path'], allow_pickle=False) as a:
                    original_probability = a['probability'].copy()
                original_length = path_length(original['decoded'])
                for variant, (metadata, changed) in interventions(x).items():
                    row = dict(arm=arm, sample_id=entry['sample_id'], split=entry['split'], variant=variant,
                        intervention=metadata, geometry=entry['geometry'],
                        original_path_length_px=original_length, original_decoded=original['decoded'],
                        probability_change=None, probability_path=None, probability_sha256=None,
                        decoded=None, model_ms=None, decode_ms=None,
                        identity_confirmed=False, target_visibility_confirmed=False)
                    if changed is not None:
                        t0 = time.perf_counter()
                        result = model(torch.from_numpy(changed[None]).float())
                        prob = mask_normalized_probabilities(result, torch.from_numpy(valid[None] > .5))[0].numpy()
                        slope = result['slope'][0].numpy(); t1 = time.perf_counter()
                        decoded = decode_path(prob, slope, entry['geometry'], valid); t2 = time.perf_counter()
                        path = output/'probabilities'/arm/f'{number:02d}_{variant}.npz'
                        save(path, probability=prob, slope=slope)
                        row.update(probability_path=str(path), probability_sha256=sha(path), decoded=decoded,
                            probability_change=distribution_change(original_probability, prob, valid, entry['geometry']['native_px_per_row']),
                            intervention_path_length_px=path_length(decoded),
                            path_length_delta_px=path_length(decoded)-original_length,
                            model_ms=(t1-t0)*1000, decode_ms=(t2-t1)*1000)
                    rows.append(row)
                print(json.dumps(dict(arm=arm, completed=number+1, total=len(p['entries']))), flush=True)
    summaries = []
    for arm in ARMS:
        for split in ('train', 'holdout'):
            for variant in VARIANTS:
                selected = [r for r in rows if r['arm'] == arm and r['split'] == split and r['variant'] == variant]
                available = [r for r in selected if r['intervention']['available']]
                def median(key):
                    return float(np.median([r['probability_change'][key] for r in available])) if available else None
                summaries.append(dict(arm=arm, split=split, variant=variant, samples=len(selected),
                    available=len(available), median_total_variation=median('mean_total_variation_per_valid_column'),
                    median_peak_shift_native_px=median('peak_row_shift_mean_native_px'),
                    median_absolute_path_length_change_px=float(np.median([abs(r['path_length_delta_px']) for r in available])) if available else None,
                    identity_accuracy=None))
    verify(output)
    write(output/'predictions.json', dict(status='complete', protocol_sha256=sha(output/'protocol.json'),
        rows=rows, summaries=summaries, annotation_geometry_used_for_inference=False,
        probabilities_are_not_visibility=True, hardware_access=False))
    return summaries


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=('freeze', 'verify', 'infer'))
    ap.add_argument('--trial', type=Path, default=TRIAL)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args(); output = args.output or args.trial/'query_probe'
    if args.stage == 'freeze': freeze(args.trial, output)
    elif args.stage == 'verify': verify(output)
    else: print(json.dumps(infer(output), indent=2))


if __name__ == '__main__': main()
