"""Bounded numeric checks and fixed H1 pilot smoke; no labels/model calls."""
from pathlib import Path
import hashlib
import json
import sys

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
H1 = HERE.parent / 'postprocess_b_20261030_v1'
sys.path.insert(0, str(H1))
from priors import load_priors
from free_candidates import describe_free, suggest_a7_entries


def synthetic():
    h, w = 250, 400
    yy, xx = np.mgrid[:h, :w]
    gray = np.full((h, w), .5, np.float32)
    gray[abs(yy-100) < 5] = 0.  # true dark-rope pixels are valid observations
    heat = (.8*np.exp(-.5*((yy-100)/3.)**2)).astype(np.float32)
    image = np.full((775, w, 3), 127, np.uint8)
    image[525:] = np.repeat(np.rint(gray*255).astype(np.uint8)[..., None], 3, axis=2)
    center = np.c_[np.arange(0, w*2, 4.), np.full(w//2, 1250.)]
    c = dict(candidate_id='synthetic_a7', accepted=True, plausible=True,
        center_xy=center.tolist(), visible=[True]*len(center), raw_supported=[True]*len(center),
        outside=[True]*len(center), width_px=20., origin_xy=[400.,1250.], direction_xy=[1.,0.],
        anchor_segment_xy=[[80.,1250.],[600.,1250.]], anchor_span_px=520., score=.8)
    p = dict(original_half=image, original_shape=[1550, 800], a7_candidates=[c], a6_entries=[], pitch_px=30.)
    seed = dict(xy=[[40.,1250.],[160.,1250.]], entry_id='synthetic', source='synthetic')
    r = describe_free(p, seed, heat)
    assert r['available'] and sum(r['supported']) > 40
    assert r['diagnostics']['real_zero_gray_pixels'] > 0
    assert len(r['segment_status']) == len(r['points_xy'])-1
    assert all(len(v) == len(r['points_xy']) for v in r['evidence'].values())
    with_nan = heat.copy(); with_nan[:,100:120] = np.nan
    q = describe_free(p, seed, with_nan)
    coords = np.asarray(q['points_xy']); support = np.asarray(q['supported'])
    assert not support[(coords[:,0]>=100)&(coords[:,0]<120)].any()
    assert q['diagnostics']['input_nan_pixels'] == h*20
    assert not suggest_a7_entries(p, [seed])
    proposals = suggest_a7_entries(p, [])
    assert len(proposals) == 2 and all(s['direction_ambiguous'] for s in proposals)
    assert all(abs(np.linalg.norm(np.diff(np.asarray(s['xy']),axis=0))-120.) < 1.e-6 for s in proposals)
    conflict = describe_free(p, dict(xy=[[40.,1190.],[40.,1310.]],source='synthetic'), heat)
    assert conflict['entry_conflict'] and not any(conflict['supported'])
    return dict(checks=['true_zero_gray_is_valid', 'NaN_has_no_support',
        'original_to_crop_mapping', 'segment_and_evidence_alignment',
        'A7_duplicate_suppression', 'unknown_direction_two_hypotheses',
        'seed_first_120_original_px', 'competing_A7_entry_conflict'], passed=True)


def main():
    out = HERE/'free_diagnostics'; out.mkdir(exist_ok=True)
    report = dict(synthetic=synthetic(), source_sha256=hashlib.sha256((HERE/'free_candidates.py').read_bytes()).hexdigest(),
        labels_read=False, model_calls=0, devices_connected=False, real_examples=[])
    manifest = json.loads((H1/'protocol_inputs.json').read_text(encoding='utf-8-sig'))
    ids = ['dual_20260923_101517__pair0075__right', 'R_run_20260923_01__f000055__right']
    for identifier in ids:
        job = next(j for j in manifest['jobs'] if j['id'] == identifier)
        priors = load_priors(job)
        new_seeds = suggest_a7_entries(priors, job['seeds'])
        for k, seed in enumerate(job['seeds']):
            with np.load(H1/'responses'/identifier/f'entry_{k:02d}.npz', allow_pickle=False) as data:
                heat = data['response'].copy()
            result = describe_free(priors, seed, heat)
            (out/f'{identifier}__entry{k+1:02d}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            summary = {key: result.get(key) for key in ('available','reason','entry_conflict','width_half_px','a7_candidate_id')}
            summary.update(id=identifier, entry_index=k+1, points=len(result['points_xy']),
                supported_points=sum(result['supported']), continuation_starts=len(result['continuation_starts']),
                disconnected_pieces=len(result['disconnected_pieces']), elapsed_seconds=result['diagnostics']['elapsed_seconds'],
                new_entries_count=len(new_seeds))
            report['real_examples'].append(summary)
    (out/'check_receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__': main()
