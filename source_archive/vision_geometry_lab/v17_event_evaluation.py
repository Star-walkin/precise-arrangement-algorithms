"""V17 event research: source-aware prefix replay, then separate scoring.

This does not implement the boundary cache, train a model, open raw video, or
change V16 event thresholds. Metadata checks cannot establish rope identity or
prove a hidden boundary unchanged. Predicted support is never called measured.
"""
from __future__ import annotations

from collections import Counter
import argparse
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .v16_event_candidates import EventCandidateTracker, PROTOCOL, score_candidates


MODES = ('direct_grid', 'temporal_guarded')
SESSIONS = {'left': 'dual_20260924_111500', 'right': 'dual_20260924_111143'}
BOUNDARY_STATES = ('direct', 'predicted', 'unknown')
TIMEBASE = 'source_video_frame_index_over_fps'
LAB = Path(__file__).resolve().parent
ROOT = LAB.parent
BASE = LAB / 'outputs/v17_temporal_boundary'
EVENT_SOURCES = (
    LAB / 'v17_event_evaluation.py', LAB / 'v16_event_candidates.py',
    ROOT / 'vision_prototype/centerline_event_fusion.py',
    ROOT / 'vision_prototype/visual_turn_phase.py',
)
RULES = dict(maximum_boundary_age_s=2., unchanged_event_rules=copy.deepcopy(PROTOCOL),
             past_only=True, full_turn_period_supplied=False,
             unknown_boundary_allowed=False, explicit_visibility_rejection_allowed=True,
             predicted_support_is_measured=False, require_all_four_before_truth=True)
LIMITATIONS = [
    'Fixed V16 center_only path; path_identity_confirmed=false and path_visibility_assessed=false.',
    'Current and past boundary support are distinct; predicted geometry is not an observed layer event.',
    'Background registration plus overlap elsewhere does not prove an occluded boundary stayed unchanged during new-layer growth.',
    'An age cap of 2 seconds is a validity limit, not a claim of 0.2-second event accuracy.',
    'Oldest-age consistency can be checked here; per-cell no-self-refresh must be tested in the boundary tracker itself.',
    'Near-distance-censored proxy can remain below rearm clearance of 3 widths; repeat-cycle operation is not validated.',
    'gap_proxy is optical-width-normalized image separation, not an axial rope count.',
    'Manual event labels are approximate and the two clips are repeatedly used development data, not blind testing.',
    'T2 is not evaluated: no independent unambiguous full-turn phase is provided.',
    'Offline processing lower bounds are not measured camera-to-actuator latency.',
]


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def digest(payload):
    return hashlib.sha256(json.dumps(clean(payload), sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    values = np.asarray([float(v) for v in values if finite(v)], float)
    if not len(values):
        return dict(count=0, minimum=None, p50=None, p95=None, maximum=None)
    q = np.percentile(values, [0, 50, 95, 100])
    return dict(count=int(len(values)), minimum=float(q[0]), p50=float(q[1]),
                p95=float(q[2]), maximum=float(q[3]))


def validate_schedule(records, expected_indices, fps):
    if not finite(fps) or fps <= 0 or not expected_indices:
        raise ValueError('Finite positive fps and complete nonempty schedule required')
    if (any(not isinstance(i, int) or i < 0 for i in expected_indices) or
            any(b <= a for a, b in zip(expected_indices, expected_indices[1:]))):
        raise ValueError('Schedule must be strictly increasing frame indices')
    if [r['frame_index'] for r in records] != list(expected_indices):
        raise ValueError('Records missing, duplicated, reordered, or beyond frozen schedule')
    for row in records:
        if not finite(row['time_s']) or abs(row['time_s'] - row['frame_index'] / fps) > 1e-8:
            raise ValueError('Timebase must be source frame/fps, not host or relabelled time')


def gate_input(row, mode):
    """Fail closed for unknown/unsupported metadata, without widening gates."""
    f, now = row['features'], float(row['time_s'])
    state = f.get('boundary_state', 'unknown')
    if state not in BOUNDARY_STATES:
        raise ValueError('Unrecognized boundary state')
    reason = str(f.get('reason', 'unspecified'))
    usable = f.get('usable') is True
    oldest = f.get('oldest_observed_at_s')
    age = f.get('boundary_max_age_s')
    sigma = f.get('boundary_sigma_px')
    if state == 'unknown':
        usable, reason = False, 'boundary_unknown'
    elif f.get('visibility_rejected') is True:
        usable, reason = False, 'explicit_visibility_rejection'
    elif not usable:
        reason = 'upstream_unusable:' + reason
    elif not finite(f.get('image_gap_proxy')):
        usable, reason = False, 'nonfinite_gap_proxy'
    elif not finite(sigma) or float(sigma) < 0:
        usable, reason = False, 'boundary_uncertainty_missing_or_invalid'
    elif not finite(oldest) or not finite(age) or float(oldest) > now + 1e-8:
        usable, reason = False, 'boundary_observation_time_unknown_or_future'
    elif float(age) < 0 or abs((now - float(oldest)) - float(age)) > 1e-6:
        usable, reason = False, 'boundary_age_inconsistent_with_actual_observation'
    elif state == 'direct' and float(age) > 1e-6:
        usable, reason = False, 'direct_boundary_is_not_current'
    elif state == 'predicted':
        if mode == 'direct_grid':
            usable, reason = False, 'history_forbidden_in_direct_grid_control'
        elif not (f.get('registration_valid') is True and f.get('overlap_valid') is True):
            usable, reason = False, 'predicted_boundary_missing_current_validation'
        elif not 0 < float(age) <= RULES['maximum_boundary_age_s'] + 1e-8:
            usable, reason = False, 'predicted_boundary_expired_or_self_refreshed'
    observed = f.get('boundary_observed_at_s')
    if observed is not None and any(finite(t) and float(t) > now + 1e-8 for t in observed):
        usable, reason = False, 'future_boundary_point'
    processing = row.get('approximate_serial_processing_ms', row.get('processing_ms', 0.))
    if not finite(processing) or float(processing) < 0:
        raise ValueError('Invalid processing duration')
    record = dict(source_frame=int(row['frame_index']), time_s=now, usable=bool(usable),
                  gap_proxy=f.get('image_gap_proxy'), processing_ms=float(processing),
                  stationary=bool(row.get('stationary', False)))
    info = dict(boundary_state=state, upstream_usable=f.get('usable') is True,
                accepted_for_candidate=bool(usable), gate_reason=reason,
                oldest_observed_at_s=oldest, boundary_max_age_s=age,
                boundary_sigma_px=sigma, registration_valid=f.get('registration_valid'),
                overlap_valid=f.get('overlap_valid'),
                boundary_source_counts=copy.deepcopy(f.get('boundary_source_counts', {})),
                path_identity_confirmed=False, path_visibility_assessed=False,
                physical_event_confirmed=False)
    return record, info


def build_prediction(records, *, session, mode, expected_indices, fps, provenance=None):
    if session not in SESSIONS or mode not in MODES:
        raise ValueError('Unknown cohort session or mode')
    validate_schedule(records, expected_indices, fps)
    tracker = EventCandidateTracker()
    timeline, candidates = [], []
    source_counts, accepted_counts, rejections = Counter(), Counter(), Counter()
    for raw in records:
        record, support = gate_input(raw, mode)
        before = time.perf_counter()
        emitted = tracker.update(record, period=None)
        elapsed = (time.perf_counter() - before) * 1000.
        source_counts[support['boundary_state']] += 1
        if record['usable']:
            accepted_counts[support['boundary_state']] += 1
        else:
            rejections[support['gate_reason']] += 1
        entry = dict(frame_index=raw['frame_index'], time_s=raw['time_s'], input=record,
                     input_boundary=support, new_candidates=[],
                     tracker_state=dict(latest_reason=tracker.latest_reason,
                                        approach_armed=tracker.approach_armed,
                                        cycle_expired=tracker.cycle_expired,
                                        cycle_id=tracker.cycle_id,
                                        t2_reason=tracker.period_rejection_reason),
                     event_processing_ms=elapsed)
        timeline.append(entry)
        for candidate in emitted:
            if candidate['event'] == 't2':
                raise AssertionError('No independent period supplied, t2 forbidden')
            # Diagnostic support interval only, not a claim of exhaustive
            # model ancestry. The model may have already chosen the wrong rope.
            begin = candidate['event_time_s'] - PROTOCOL['reversal_window_s']
            window = [r for r in timeline if begin <= r['time_s'] <= record['time_s']]
            supported = Counter(r['input_boundary']['boundary_state'] for r in window if r['input']['usable'])
            candidate.update(boundary_support_states=dict(supported),
                support_diagnostic_interval_s=[begin, record['time_s']],
                confirmation_boundary_state=support['boundary_state'],
                contains_predicted_boundary_support=supported['predicted'] > 0,
                unknown_frames_in_diagnostic_interval=sum(
                    r['input_boundary']['boundary_state'] == 'unknown' for r in window),
                measured_physical_node=False, path_identity_confirmed=False,
                path_visibility_assessed=False,
                first_available_processing_lower_bound_s=record['time_s'] +
                    (record['processing_ms'] + elapsed) / 1000.)
            candidates.append(copy.deepcopy(candidate))
            entry['new_candidates'].append(copy.deepcopy(candidate))
    summary = dict(frame_count=len(records), accepted_frames=sum(accepted_counts.values()),
        accepted_fraction=sum(accepted_counts.values()) / len(records),
        frame_boundary_state_counts={s: source_counts[s] for s in BOUNDARY_STATES},
        accepted_boundary_state_counts={s: accepted_counts[s] for s in BOUNDARY_STATES},
        rejection_reason_counts=dict(rejections), candidate_counts=dict(Counter(c['event'] for c in candidates)),
        candidate_support_counts=dict(Counter('predicted_involved' if c['contains_predicted_boundary_support']
                                             else 'direct_only_diagnostic_window' for c in candidates)),
        boundary_age_s=stats([r['input_boundary']['boundary_max_age_s'] for r in timeline]),
        boundary_sigma_px=stats([r['input_boundary']['boundary_sigma_px'] for r in timeline]),
        event_processing_ms=stats([r['event_processing_ms'] for r in timeline]),
        upstream_processing_ms=stats([r['input']['processing_ms'] for r in timeline]),
        max_consecutive_accepted_frames=_longest_run([r['input']['usable'] for r in timeline]),
        max_consecutive_unknown_frames=_longest_run([
            r['input_boundary']['boundary_state'] == 'unknown' for r in timeline]))
    return clean(dict(schema='v17_event_prediction_1', session=session, mode=mode,
        fps=fps, expected_indices=list(expected_indices), timebase=TIMEBASE,
        time_range_s=[records[0]['time_s'], records[-1]['time_s']],
        measurement_records_sha256=digest(records), provenance=copy.deepcopy(provenance or {}),
        event_rules=RULES, prediction_rows=timeline, candidates=candidates, summary=summary,
        final_tracker=tracker.finalize(), ground_truth_used=False,
        t2_status='not_evaluated_no_independent_full_turn_phase',
        path_identity_confirmed=False, path_visibility_assessed=False,
        measured_physical_nodes=0, limitations=LIMITATIONS,
        hardware_access=False, deployment_allowed=False))


def _longest_run(flags):
    best = length = 0
    for flag in flags:
        length = length + 1 if flag else 0
        best = max(best, length)
    return best


def validate_prediction(prediction):
    if (prediction.get('schema') != 'v17_event_prediction_1' or
            prediction.get('ground_truth_used') is not False or
            prediction.get('event_rules') != RULES or
            prediction.get('timebase') != TIMEBASE or
            prediction.get('measured_physical_nodes') != 0 or
            prediction.get('deployment_allowed') is not False):
        raise ValueError('Foreign or unsafe prediction')
    timeline = prediction['prediction_rows']
    validate_schedule(timeline, prediction['expected_indices'], prediction['fps'])
    flattened = [c for row in timeline for c in row['new_candidates']]
    if flattened != prediction['candidates']:
        raise ValueError('First-emission timeline and candidate list disagree')
    for row in timeline:
        for c in row['new_candidates']:
            if (c['event'] not in ('t0', 't1') or c['event_time_s'] > row['time_s'] or
                    c['first_available_at_s'] != row['time_s'] or
                    c.get('measured_physical_node') is not False):
                raise ValueError('Future/backdated, illegal t2 or measured-node claim')


def write_prediction(path, payload):
    validate_prediction(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(clean(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)


def score_prediction_files(paths, truth_path, output=None):
    if len(paths) != 4:
        raise ValueError('All four predictions required before truth is opened')
    predictions = {}
    for path in paths:
        path = Path(path)
        prediction = json.loads(path.read_text(encoding='utf-8'))
        validate_prediction(prediction)
        key = prediction['session'], prediction['mode']
        if key in predictions:
            raise ValueError('Duplicate session/mode prediction')
        predictions[key] = (prediction, file_sha256(path))
    if set(predictions) != {(s, m) for s in SESSIONS for m in MODES}:
        raise ValueError('Cohort missing one or more session/mode combinations')
    for session in SESSIONS:
        control, candidate = [predictions[(session, mode)][0] for mode in MODES]
        if control['fps'] != candidate['fps'] or control['expected_indices'] != candidate['expected_indices']:
            raise ValueError('Modes must use the same source schedule')
    # The only annotation read: every prediction and its prefix schema is now validated.
    truth_path = Path(truth_path)
    truth = json.loads(truth_path.read_text(encoding='utf-8'))['sessions']
    cases = []
    for (session, mode), (prediction, prediction_hash) in predictions.items():
        event_score = score_candidates(prediction['candidates'], truth[SESSIONS[session]]['manual_review'])
        event_score['events']['t2']['evaluation_status'] = 'not_evaluated_missing_independent_period'
        event_score['measured_physical_node_count'] = 0
        cases.append(dict(session=session, mode=mode, prediction_sha256=prediction_hash,
                          summary=prediction['summary'], candidates=prediction['candidates'],
                          event_scores=event_score))
    report = dict(schema='v17_event_score_1', truth_read_after_four_predictions=True,
                  truth_sha256=file_sha256(truth_path), cases=cases,
                  matched_first_not_closest=True, population_accuracy_estimated=False,
                  research_tolerance_s=.2, limitations=LIMITATIONS,
                  measured_physical_nodes=0, deployment_allowed=False)
    if output is not None:
        output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as f:
            json.dump(clean(report), f, ensure_ascii=False, indent=2, allow_nan=False)
    return report


def _write_new(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(clean(payload), f, ensure_ascii=False, indent=2, allow_nan=False)


def freeze_event_protocol(base=BASE):
    from .v17_boundary_trial import verify as verify_boundary
    base = Path(base).resolve()
    destination = base / 'events'
    if destination.exists():
        raise FileExistsError('Use a fresh event trial, do not overwrite or silently resume')
    core = verify_boundary(base)
    payload = dict(schema='v17_event_protocol_1',
        frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        base=str(base), measurement_protocol_sha256=file_sha256(base / 'protocol.json'),
        source_snapshot={str(f):file_sha256(f) for f in EVENT_SOURCES},
        rules=RULES, sessions=SESSIONS, modes=list(MODES),
        expected_indices={s:core['videos'][s]['selected_indices'] for s in SESSIONS},
        truth_read_for_prediction=False, limitations=LIMITATIONS,
        deployment_allowed=False, hardware_access=False)
    _write_new(destination / 'protocol.json', payload)
    print('Frozen V17 event protocol', file_sha256(destination / 'protocol.json'))
    return payload


def verify_event_protocol(base=BASE):
    from .v17_boundary_trial import verify as verify_boundary
    base = Path(base).resolve()
    path = base / 'events/protocol.json'
    p = json.loads(path.read_text(encoding='utf-8'))
    if (p.get('schema') != 'v17_event_protocol_1' or p.get('rules') != RULES or
            p.get('sessions') != SESSIONS or p.get('modes') != list(MODES) or
            Path(p['base']) != base):
        raise ValueError('Event protocol mismatch')
    if set(p['source_snapshot']) != {str(f) for f in EVENT_SOURCES}:
        raise ValueError('Incomplete event source manifest')
    for source, expected in p['source_snapshot'].items():
        if file_sha256(source) != expected:
            raise ValueError('Frozen event dependency changed: ' + source)
    if file_sha256(base / 'protocol.json') != p['measurement_protocol_sha256']:
        raise ValueError('Boundary measurement protocol changed')
    core = verify_boundary(base)
    if p['expected_indices'] != {s:core['videos'][s]['selected_indices'] for s in SESSIONS}:
        raise ValueError('Frame schedules changed')
    return p, core


def _load_measurement(base, p, core, session, mode):
    path = Path(base) / 'measure' / session / (mode + '.json')
    value = json.loads(path.read_text(encoding='utf-8'))
    if (value.get('schema') != 'v17_boundary_measurements_1' or
            value.get('protocol_sha256') != p['measurement_protocol_sha256'] or
            value.get('session') != session or value.get('mode') != mode or
            value.get('video') != core['videos'][session] or
            value.get('ground_truth_used') is not False or
            value.get('hardware_access') is not False or
            value.get('deployment_allowed') is not False):
        raise ValueError('Foreign measurement or source-timebase mismatch')
    validate_schedule(value['records'], p['expected_indices'][session], core['videos'][session]['fps'])
    return path, value


def predict_from_base(base=BASE, session='left', mode='direct_grid'):
    if session not in SESSIONS or mode not in MODES:
        raise ValueError('Unknown session/mode')
    base = Path(base).resolve()
    output = base / 'events/predictions' / session / (mode + '.json')
    if output.exists():
        raise FileExistsError('Prediction exists; do not silently replace it')
    p, core = verify_event_protocol(base)
    path, measurement = _load_measurement(base, p, core, session, mode)
    measurement_hash = file_sha256(path)
    provenance = dict(measurement_path=str(path), measurement_sha256=measurement_hash,
        protocol_sha256=file_sha256(base / 'events/protocol.json'),
        measurement_protocol_sha256=p['measurement_protocol_sha256'],
        video=core['videos'][session], model=core['model'])
    prediction = build_prediction(measurement['records'], session=session, mode=mode,
        expected_indices=p['expected_indices'][session], fps=core['videos'][session]['fps'],
        provenance=provenance)
    verify_event_protocol(base)
    if file_sha256(path) != measurement_hash:
        raise ValueError('Measurement changed during causal replay')
    write_prediction(output, prediction)
    print(json.dumps(dict(session=session, mode=mode, summary=prediction['summary'],
                          candidates=prediction['candidates']), ensure_ascii=False))
    return prediction


def score_from_base(base=BASE, truth_path=None):
    base = Path(base).resolve()
    target = base / 'events/score.json'
    if target.exists():
        raise FileExistsError('Score exists; use the frozen existing report')
    p, core = verify_event_protocol(base)
    predictions = []
    for session in SESSIONS:
        for mode in MODES:
            pred_path = base / 'events/predictions' / session / (mode + '.json')
            prediction = json.loads(pred_path.read_text(encoding='utf-8'))
            validate_prediction(prediction)
            provenance = prediction['provenance']
            measurement_path, measurement = _load_measurement(base, p, core, session, mode)
            if (prediction['session'] != session or prediction['mode'] != mode or
                    provenance.get('protocol_sha256') != file_sha256(base / 'events/protocol.json') or
                    provenance.get('measurement_protocol_sha256') != p['measurement_protocol_sha256'] or
                    provenance.get('measurement_sha256') != file_sha256(measurement_path) or
                    prediction.get('measurement_records_sha256') != digest(measurement['records']) or
                    prediction['expected_indices'] != p['expected_indices'][session]):
                raise ValueError('Prediction provenance mismatch')
            predictions.append(pred_path)
    # No annotation is opened until all four predictions and upstream bindings pass.
    truth_path = (Path(truth_path) if truth_path is not None else
        ROOT / 'vision_prototype/validation_ground_truth/dual_reversal_events.json')
    result = score_prediction_files(predictions, truth_path, target)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze','verify','predict','score'))
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--session', choices=tuple(SESSIONS), default='left')
    parser.add_argument('--mode', choices=MODES, default='direct_grid')
    parser.add_argument('--truth', type=Path, default=None,
                        help='Read only by score, after all four predictions exist')
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze_event_protocol(args.base)
    elif args.stage == 'verify':
        verify_event_protocol(args.base)
        print('V17 boundary and event source protocols verified')
    elif args.stage == 'predict':
        predict_from_base(args.base, args.session, args.mode)
    else:
        score_from_base(args.base, args.truth)


if __name__ == '__main__':
    main()
