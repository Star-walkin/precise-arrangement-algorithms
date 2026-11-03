"""Frozen V11: replace manual free-span input with V16 automatic seed.

This is a paired development audit, not a newly trained model. All automatic
predictions are persisted before the scoring function opens geometry labels.
It measures the ENTIRE changed input contract (seed, width, crop), not a causal
attribution to endpoint error alone. No hardware or event inference.
"""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
import time
import numpy as np

LAB = Path(__file__).resolve().parent
BASE = LAB/'outputs/v23_auto_seed_chain'
OLD = LAB/'outputs/v11_path_identity'
FREE = LAB/'outputs/v23_free_rope_audit'
INVENTORY = LAB/'outputs/centerline_v9/dataset/inventory.json'


def read(path): return json.loads(Path(path).read_text('utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
def save(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f: np.savez_compressed(f, **arrays)


def freeze(base):
    if base.exists(): raise FileExistsError(base)
    samples = read(INVENTORY)['samples']
    if len(samples) != 20 or '20260925' in json.dumps(samples):
        raise ValueError('Only fixed 20 development images allowed')
    paths = [INVENTORY, FREE/'predictions.json', FREE/'protocol.json',
        OLD/'development/checkpoint.pt', OLD/'development/report.json',
        OLD/'development/evaluation.json', OLD/'development_manifest.json']
    paths += [LAB/name for name in ('v23_auto_seed_chain.py', 'rope_detector.py',
        'arc_detector.py', 'geometry.py', 'wound_rope_tracker.py',
        'v16_video_measurement.py', 'v11_path_contract.py', 'v11_path_model.py',
        'v11_path_decode.py', 'evaluate_centerline_multiscale_v9.py',
        'centerline_path_metrics_v6.py', 'centerline_quality_v5.py',
        'centerline_quality_cutoff_v8.py')]
    paths += [Path(s['image_path']) for s in samples]
    for sample in samples:
        if sha(sample['image_path']) != sample['image_sha256']:
            raise ValueError('Image changed')
    p = dict(schema='v23_auto_seed_chain_1',
        experiment='Frozen V11 manual input versus full automatic V16 seed/width/crop input',
        mode='one fixed inference, no thresholds or training changed',
        auto_seed='observed_segment directed by current observed arc proximity; NOT physical contact',
        free_predictions_sha256=sha(FREE/'predictions.json'),
        model_sha256=sha(OLD/'development/checkpoint.pt'),
        label_access='automatic prepare/infer do not open annotations; evaluate opens after export exists',
        score='unchanged V6/V9 5-native-pixel full and seed-downstream arc-length metrics',
        empty_predictions='retained, zero coverage if trusted reference exists',
        scope='18 geometry plus 2 negative/partial diagnostics; source-separated development, not blind',
        identity_confirmed=False, event_timing_validated=False, hardware_access=False,
        source_hashes={str(p.resolve()): sha(p) for p in paths})
    write(base/'protocol.json', p)


def verify(base):
    p = read(base/'protocol.json')
    for path, expected in p['source_hashes'].items():
        if sha(path) != expected: raise ValueError('Frozen file changed: '+path)
    return p


def prepare(base):
    verify(base)
    import cv2
    from .arc_detector import detect_arc
    from .geometry import rope_exclusion_band
    from .v16_video_measurement import automatic_seed
    from .v11_path_contract import build_input
    cv2.setNumThreads(2)
    source = read(FREE/'predictions.json')
    if source['status'] != 'predictions_complete_before_label_loading':
        raise ValueError('Free predictions incomplete')
    predictions = {r['sample_id']: r['prediction'] for r in source['rows']}
    entries = []
    for sample in read(INVENTORY)['samples']:
        rope = predictions[sample['sample_id']]
        row = dict(sample_id=sample['sample_id'], available=False,
                   split=sample['split'], group_id=sample['group_id'])
        start = time.perf_counter()
        if rope.get('valid'):
            bgr = cv2.imdecode(np.fromfile(sample['image_path'], np.uint8), cv2.IMREAD_COLOR)
            if bgr is None: raise ValueError('Image decode failed')
            mask, _ = rope_exclusion_band(bgr.shape, rope)
            arc = detect_arc(bgr, exclusion_mask=mask)
            seed, reason = automatic_seed(rope, arc)
        else:
            seed, reason = None, 'free_rope_unavailable'
        row.update(reason=reason, free_seed=seed, width_px=rope.get('width_px'))
        if seed is not None:
            built = build_input(np.ascontiguousarray(bgr[:,:,::-1]), seed, rope['width_px'])
            path = base/'inputs'/(sample['sample_id']+'.npz')
            save(path, input=built['input'], valid_mask=built['valid_mask'])
            row.update(available=True, input_path=str(path), input_sha256=sha(path), geometry=built['geometry'])
        row['prepare_ms_excluding_free_detector'] = (time.perf_counter()-start)*1000
        entries.append(row)
    write(base/'inputs.json', dict(entries=entries, protocol_sha256=sha(base/'protocol.json')))
    verify(base)


def infer(base):
    verify(base)
    import torch
    from .v11_path_model import build_model, mask_normalized_probabilities
    from .v11_path_decode import decode_path
    torch.set_num_threads(2)
    index = read(base/'inputs.json')
    if index['protocol_sha256'] != sha(base/'protocol.json'): raise ValueError('Input protocol')
    checkpoint = torch.load(OLD/'development/checkpoint.pt', map_location='cpu', weights_only=True)
    if sha(OLD/'development/checkpoint.pt') != read(OLD/'development/report.json')['checkpoint_sha256']:
        raise ValueError('Checkpoint mismatch')
    model = build_model(checkpoint['seed'], base_channels=checkpoint['base_channels'])
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    rows = []
    with torch.inference_mode():
        for entry in index['entries']:
            row = dict(entry)
            start = time.perf_counter()
            decoded = dict(segments=[], reason=entry['reason'], identity_confirmed=False)
            if entry['available']:
                if sha(entry['input_path']) != entry['input_sha256']: raise ValueError('Input changed')
                with np.load(entry['input_path'], allow_pickle=False) as a:
                    x, valid = a['input'].copy(), a['valid_mask'].copy()
                out = model(torch.from_numpy(x[None]).float())
                prob = mask_normalized_probabilities(out, torch.from_numpy(valid[None] > .5))[0].numpy()
                decoded = decode_path(prob, out['slope'][0].numpy(), entry['geometry'], valid)
            row.update(decoded=decoded, forward_decode_ms=(time.perf_counter()-start)*1000)
            rows.append(row)
    write(base/'predictions.json', dict(rows=rows, protocol_sha256=sha(base/'protocol.json'),
        inputs_sha256=sha(base/'inputs.json'), all_predictions_saved_before_geometry_label_scoring=True))
    verify(base)


def evaluate(base):
    verify(base)
    from .evaluate_centerline_multiscale_v9 import _dataset, score_decoded, aggregate
    predictions = read(base/'predictions.json')
    if (predictions['protocol_sha256'] != sha(base/'protocol.json') or
            predictions['inputs_sha256'] != sha(base/'inputs.json') or
            not predictions['all_predictions_saved_before_geometry_label_scoring']):
        raise ValueError('Prediction provenance')
    rows_by_id = {r['sample_id']: r for r in predictions['rows']}
    _, records, _ = _dataset(OLD/'development_manifest.json')
    scores = []
    for record in records:
        s = record['sample']; pred = rows_by_id[s['sample_id']]
        result = score_decoded(record['label'], record['quality'], pred['decoded'], None, s['image_shape_hw'])
        scores.append(dict(sample_id=s['sample_id'], split=s['split'], group_id=s['group_id'],
            cohort=record['cohort'], decoded=pred['decoded'],
            independent_width=dict(width_px=pred['width_px']),
            prediction_status='loaded' if pred['available'] else 'unavailable', **result))
    report = dict(auto_summaries=aggregate(scores), manual_summaries=read(OLD/'development/evaluation.json')['summaries'],
        rows=scores, diagnostic_only=[r for r in predictions['rows'] if r['sample_id'] not in {s['sample_id'] for s in scores}],
        protocol_sha256=sha(base/'protocol.json'), predictions_sha256=sha(base/'predictions.json'),
        coverage_footprint_not_scored=True, reference_not_cropped_to_input=True,
        comparison='paired model, combined seed+width+crop change; not endpoint-only causal attribution',
        development_only=True, identity_confirmed=False, hardware_access=False, force_command=None)
    write(base/'report.json', report)
    print(json.dumps(dict(auto=[r for r in report['auto_summaries'] if r['cohort']=='all' and r['scope']=='downstream'],
                          manual=[r for r in report['manual_summaries'] if r['cohort']=='all' and r['scope']=='downstream'])))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=('freeze','prepare','infer','evaluate','verify'))
    ap.add_argument('--output', type=Path, default=BASE)
    args = ap.parse_args()
    globals()[args.stage](args.output)


if __name__ == '__main__': main()
