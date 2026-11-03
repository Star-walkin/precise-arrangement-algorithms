"""Frozen V16 raw-measurement event trial, with prediction/scoring separated.

freeze/predict/verify do not open annotation files. score requires all four
frozen-source predictions before it opens the approximate manual references.
No hardware, model training, raw-camera access or threshold search occurs here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .v16_event_candidates import EventCandidateTracker, PROTOCOL, score_candidates

LAB = Path(__file__).resolve().parent
ROOT = LAB.parent
MEASURE_ROOT = LAB / 'outputs/v16_event_feasibility'
BASE = MEASURE_ROOT / 'events'
SESSIONS = {'left': 'dual_20260924_111500', 'right': 'dual_20260924_111143'}
ARMS = ('center_only', 'joint')
SOURCE_FILES = (
    LAB / 'v16_event_trial.py', LAB / 'v16_event_candidates.py',
    LAB / 'v16_video_measurement.py',
    ROOT / 'vision_prototype/centerline_event_fusion.py',
    ROOT / 'vision_prototype/visual_turn_phase.py',
)
LIMITATIONS = [
    'Two repeatedly used development clips; not blind or population accuracy validation.',
    'Automatic free-rope seeds and path identity are unconfirmed.',
    'gap_proxy is image/optical-width separation, not an axial rope count or millimetres.',
    'Direct links are near-distance censored (usually gap_proxy < about 2.5); rearm > 3 may never be observable. This is first-cycle research only, not repeat-cycle validation.',
    'Contact and a gap valley are proxies, not proof of last-turn or radial new-layer onset.',
    'No reliable independent full-turn phase supplied; t2 is unavailable by design, not a tested model achievement.',
    'No independent scene-stationary input in these clips; constant gap is never treated as stop.',
    'Acquisition timestamps use source AVI frame/fps only, not dual-camera host synchronization.',
    'Summed offline extraction/inference/measurement timing includes disk/compression work and possible resource contention; not live end-to-end latency.',
    'Thresholds are unchanged; missing geometry is never filled with zero or extrapolated contact.',
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def write_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(clean(payload), f, ensure_ascii=False, indent=2, allow_nan=False)


def freeze(output=BASE, measurement_root=MEASURE_ROOT):
    from .v16_video_measurement import verify as verify_measurement
    output, measurement_root = Path(output), Path(measurement_root).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh event trial root; never overwrite an experiment.')
    mp = verify_measurement(measurement_root, verify_videos=False)
    protocol_path = measurement_root / 'protocol.json'
    p = dict(schema='v16_event_trial_1',
             frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             measurement_root=str(measurement_root),
             measurement_protocol=dict(path=str(protocol_path), sha256=sha(protocol_path)),
             source_snapshot={str(path): sha(path) for path in SOURCE_FILES},
             event_rules=copy.deepcopy(PROTOCOL), sessions=SESSIONS, arms=list(ARMS),
             selected_indices={s: mp['videos'][s]['selected_indices'] for s in SESSIONS},
             period_input='none', stationary_input='none',
             timebase='source_video_frame_index_over_fps',
             labels_used_to_freeze=False, prediction_reads_truth=False,
             score_requires_all_four_predictions=True, limitations=LIMITATIONS,
             ground_truth_used=False, hardware_access=False, deployment_allowed=False)
    write_new(output / 'protocol.json', p)
    print(json.dumps(dict(stage='event_freeze', path=str(output / 'protocol.json'),
                          sha256=sha(output / 'protocol.json')), ensure_ascii=False))
    return p


def verify(output=BASE):
    from .v16_video_measurement import verify as verify_measurement
    output = Path(output)
    p = read(output / 'protocol.json')
    if (p['schema'] != 'v16_event_trial_1' or p['event_rules'] != PROTOCOL or
            p['sessions'] != SESSIONS or p['arms'] != list(ARMS) or p['period_input'] != 'none'):
        raise ValueError('Event protocol mismatch')
    expected_sources = {str(path) for path in SOURCE_FILES}
    if set(p['source_snapshot']) != expected_sources:
        raise ValueError('Event source manifest incomplete')
    for path, expected in p['source_snapshot'].items():
        if sha(path) != expected:
            raise ValueError('Frozen event source changed: ' + path)
    reference = p['measurement_protocol']
    if sha(reference['path']) != reference['sha256']:
        raise ValueError('Measurement protocol changed')
    mp = verify_measurement(p['measurement_root'], verify_videos=False)
    if p['selected_indices'] != {s: mp['videos'][s]['selected_indices'] for s in SESSIONS}:
        raise ValueError('Sampling schedule changed')
    return p, mp


def validate_measurement_payload(payload, session, arm, mp, protocol_hash):
    if (payload['session'] != session or payload['arm'] != arm or
            payload['protocol_sha256'] != protocol_hash or
            payload.get('ground_truth_used') is not False or
            payload.get('hardware_access') is not False or
            payload.get('deployment_allowed') is not False):
        raise ValueError('Foreign or unsafe measurement payload')
    rows = payload['records']
    if [r['frame_index'] for r in rows] != mp['videos'][session]['selected_indices']:
        raise ValueError('Measurement frames missing, duplicate or out of order')
    fps = mp['videos'][session]['fps']
    for r in rows:
        if not math.isfinite(r['time_s']) or abs(r['time_s'] - r['frame_index'] / fps) > 1e-8:
            raise ValueError('Timestamp is not source frame/fps')
        gap = r['features'].get('image_gap_proxy')
        if r['features']['usable'] and (gap is None or not math.isfinite(float(gap))):
            raise ValueError('Usable measurement without a finite gap')
        ms = r['approximate_serial_processing_ms']
        if not math.isfinite(ms) or ms < 0:
            raise ValueError('Invalid processing time')
    return rows


def checked_measurement(p, mp, session, arm):
    root = Path(p['measurement_root'])
    path = root / 'measure' / session / (arm + '.json')
    payload = read(path)
    rows = validate_measurement_payload(payload, session, arm, mp,
                                        p['measurement_protocol']['sha256'])
    extraction = root / 'extract' / session / 'index.json'
    inference = root / 'infer' / session / arm / 'index.json'
    if (payload['extraction_index_sha256'] != sha(extraction) or
            payload['inference_index_sha256'] != sha(inference)):
        raise ValueError('Measurement upstream index changed')
    ex, inf = read(extraction), read(inference)
    if (ex['protocol_sha256'] != p['measurement_protocol']['sha256'] or
            ex['video_sha256'] != mp['videos'][session]['sha256'] or
            inf['protocol_sha256'] != p['measurement_protocol']['sha256'] or
            inf['checkpoint_sha256'] != mp['models'][arm]['sha256'] or
            inf['extraction_index_sha256'] != sha(extraction)):
        raise ValueError('Measurement upstream provenance mismatch')
    return path, rows


def numeric_stats(values):
    values = np.asarray([v for v in values if v is not None and math.isfinite(float(v))], float)
    if not len(values):
        return dict(count=0, minimum=None, p50=None, p95=None, maximum=None)
    q = np.percentile(values, [0, 50, 95, 100])
    return dict(count=len(values), minimum=float(q[0]), p50=float(q[1]),
                p95=float(q[2]), maximum=float(q[3]))


def replay_measurements(rows):
    """No labels, no session-dependent thresholds, one prefix at a time."""
    tracker = EventCandidateTracker()
    timeline, candidates = [], []
    reason_counts, state_counts = Counter(), Counter()
    for r in rows:
        f = r['features']
        record = dict(source_frame=r['frame_index'], time_s=r['time_s'],
                      usable=f['usable'], gap_proxy=f.get('image_gap_proxy'),
                      processing_ms=r['approximate_serial_processing_ms'], stationary=False)
        start = time.perf_counter()
        new = tracker.update(record, period=None)
        event_ms = 1000. * (time.perf_counter() - start)
        lower_bound = record['time_s'] + (record['processing_ms'] + event_ms) / 1000.
        for event in new:
            event['first_available_processing_lower_bound_s'] = lower_bound
        candidates.extend(copy.deepcopy(new))
        state = dict(latest_reason=tracker.latest_reason, cycle_id=tracker.cycle_id,
                     approach_armed=tracker.approach_armed, cycle_expired=tracker.cycle_expired,
                     t2_reason=tracker.period_rejection_reason)
        reason_counts[f['reason']] += 1
        state_counts[state['latest_reason']] += 1
        timeline.append(dict(source_frame=record['source_frame'], time_s=record['time_s'],
            input=record, measurement_reason=f['reason'], tracker_state=state,
            new_candidates=new, cumulative_candidate_count=len(candidates),
            event_processing_ms=event_ms, processing_lower_bound_available_at_s=lower_bound))
    usable = [r for r in timeline if r['input']['usable']]
    stats = dict(frames=len(rows), usable_frames=len(usable),
        usable_fraction=len(usable) / len(rows) if rows else None,
        measurement_reason_counts=dict(reason_counts), tracker_reason_counts=dict(state_counts),
        usable_gap_proxy=numeric_stats([r['input']['gap_proxy'] for r in usable]),
        approximate_serial_processing_ms=numeric_stats([r['input']['processing_ms'] for r in timeline]),
        event_processing_ms=numeric_stats([r['event_processing_ms'] for r in timeline]),
        forward_ms=numeric_stats([r.get('forward_ms') for r in rows]),
        measurement_ms=numeric_stats([r.get('measurement_ms') for r in rows]),
        usable_clearance_above_rearm_count=sum(r['input']['gap_proxy'] > 3. for r in usable),
        candidate_counts=dict(Counter(c['event'] for c in candidates)),
        timing_is_live_end_to_end_measurement=False)
    return dict(prediction_rows=timeline, candidates=candidates, summary=stats,
                final_tracker=tracker.finalize(),
                t2_status='not_supplied_independent_full_turn_phase_not_evaluated',
                hardware_access=False, deployment_allowed=False)


def predict(output, session, arm):
    output = Path(output)
    target = output / 'predictions' / session / (arm + '.json')
    if target.exists():
        raise FileExistsError('Frozen prediction already exists')
    p, mp = verify(output)
    measurement_path, rows = checked_measurement(p, mp, session, arm)
    measurement_hash = sha(measurement_path)
    result = replay_measurements(rows)
    if any(c['event'] == 't2' for c in result['candidates']):
        raise AssertionError('No independent phase was supplied; t2 is forbidden')
    result.update(schema='v16_event_prediction_1', session=session, arm=arm,
        ground_truth_used=False, timebase=p['timebase'], video=mp['videos'][session],
        source_time_range_s=[rows[0]['time_s'], rows[-1]['time_s']],
        protocol_sha256=sha(output / 'protocol.json'),
        measurement_path=str(measurement_path), measurement_sha256=measurement_hash,
        measurement_protocol_sha256=p['measurement_protocol']['sha256'],
        model_sha256=mp['models'][arm]['sha256'], limitations=LIMITATIONS)
    verify(output)
    if sha(measurement_path) != measurement_hash:
        raise ValueError('Measurement changed during replay')
    write_new(target, result)
    print(json.dumps(dict(stage='event_predict', session=session, arm=arm,
                         summary=result['summary'], candidates=result['candidates']), ensure_ascii=False))
    return result


def load_all_predictions(output, p, mp):
    """Verify all four complete results BEFORE the scoring command reads truth."""
    output = Path(output)
    results = []
    for session in SESSIONS:
        for arm in ARMS:
            path = output / 'predictions' / session / (arm + '.json')
            result = read(path)
            if (result['schema'] != 'v16_event_prediction_1' or result['session'] != session or
                    result['arm'] != arm or result['protocol_sha256'] != sha(output / 'protocol.json') or
                    result.get('ground_truth_used') is not False or
                    result['measurement_protocol_sha256'] != p['measurement_protocol']['sha256'] or
                    result['model_sha256'] != mp['models'][arm]['sha256']):
                raise ValueError('Foreign prediction')
            measurement_path, rows = checked_measurement(p, mp, session, arm)
            if result['measurement_sha256'] != sha(measurement_path):
                raise ValueError('Prediction measurement changed')
            if [r['source_frame'] for r in result['prediction_rows']] != p['selected_indices'][session]:
                raise ValueError('Prediction is not full chronological source sequence')
            if any(c['event'] == 't2' for c in result['candidates']):
                raise ValueError('Illegal t2 without independent full-turn phase')
            results.append((session, arm, path, result))
    return results


def score(output=BASE, truth_path=None):
    output = Path(output)
    if (output / 'score.json').exists():
        raise FileExistsError('Score exists; do not overwrite trial results')
    p, mp = verify(output)
    all_results = load_all_predictions(output, p, mp)
    # Deliberately the first and only annotation I/O in this module.
    truth_path = (Path(truth_path) if truth_path is not None else
                  ROOT / 'vision_prototype/validation_ground_truth/dual_reversal_events.json')
    truth = read(truth_path)['sessions']
    cases = []
    for session, arm, prediction_path, prediction in all_results:
        scored = score_candidates(prediction['candidates'], truth[SESSIONS[session]]['manual_review'])
        scored['events']['t2']['evaluation_status'] = 'not_evaluated_no_independent_full_turn_phase'
        scored['events']['t2']['model_timing_achievement'] = False
        cases.append(dict(session=session, arm=arm, video_session=SESSIONS[session],
            prediction_sha256=sha(prediction_path), summary=prediction['summary'],
            candidates=prediction['candidates'], event_scores=scored))
    report = dict(schema='v16_event_score_1', protocol_sha256=sha(output / 'protocol.json'),
        truth_sha256=sha(truth_path), truth_read_after_all_four_predictions=True,
        cases=cases, research_tolerance_s=.2, limitations=LIMITATIONS,
        population_accuracy_estimated=False, deploy_decision='not_authorized_by_this_development_trial',
        hardware_access=False, deployment_allowed=False)
    verify(output)
    write_new(output / 'score.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'verify', 'predict', 'score'))
    parser.add_argument('--output', type=Path, default=BASE)
    parser.add_argument('--measurement-root', type=Path, default=MEASURE_ROOT)
    parser.add_argument('--session', choices=tuple(SESSIONS), default='left')
    parser.add_argument('--arm', choices=ARMS, default='center_only')
    parser.add_argument('--truth', type=Path, default=None,
                        help='Used only by score after all four predictions exist')
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze(args.output, args.measurement_root)
    elif args.stage == 'verify':
        verify(args.output)
        print('V16 event and measurement source hashes verified')
    elif args.stage == 'predict':
        predict(args.output, args.session, args.arm)
    else:
        score(args.output, args.truth)


if __name__ == '__main__':
    main()
