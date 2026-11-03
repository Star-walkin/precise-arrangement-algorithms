"""Posthoc read-only explanation of frozen V19 predictions, not new inference.

No model, source video, device, threshold search or truth file is opened. The two
explicit reference windows are retrospective annotations supplied for this audit
only. Every gate is reconstructed from already archived row states and frozen
protocol values; neither predictions nor the frozen state machine are modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


BASE = Path(__file__).resolve().parent/'outputs/v19_front_candidates'
REFERENCE_T1_S = {'left': 27.3, 'right': 14.4}
EPS = 1e-9


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def runs(mask):
    ids = np.flatnonzero(mask)
    return [part for part in np.split(ids, np.flatnonzero(np.diff(ids)>1)+1) if len(part)]


def longest(mask):
    return max((len(part) for part in runs(mask)), default=0)


def quantiles(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return None
    return dict(zip(('min','p25','p50','p75','p90','max'), map(float, np.quantile(a,[0,.25,.5,.75,.9,1.]))))


def compact_example(record, indices, reason):
    return dict(frame_index=record['frame_index'], time_s=record['time_s'], reason=reason,
        rows=[{**record['tracker']['row_states'][j], 'quality_reason':record['quality_reason'][j]}
              for j in indices])


def ending_reason(previous, current, now, q, displacement, high, rules, frame_reason):
    if 'reset' in frame_reason or frame_reason == 'explicit_stop_no_event_clock':
        return 'frame_reset'
    origin = previous.get('first_high_s')
    if origin is not None and now-origin > rules['maximum_chain_history_s']+EPS:
        return 'chain_history_expired'
    direct = previous.get('last_direct_s')
    if direct is not None and now-direct > rules['maximum_observation_gap_s']+EPS:
        return 'direct_observation_gap_expired'
    if not q:
        return 'current_quality_unknown'
    if np.isfinite(displacement) and np.isfinite(high):
        if displacement <= high*rules['low_threshold_fraction']:
            return 'returned_to_low'
        if displacement < high:
            return 'middle_band_interrupted_high_run'
    return 'other_transition_inspect_example'


def analyze_session(payload, rules, reference):
    all_records = payload['records']
    ready = payload['startup_model']['ready_s']
    model = payload['startup_model']
    model_support = np.asarray(model['supported'], bool)
    threshold = np.asarray(model['threshold_px'], float)
    summaries = []
    transitions = []
    examples = {}
    row_stats = [Counter() for _ in model_support]
    previous = None
    size = rules['minimum_adjacent_rows']
    for record in all_records:
        now = float(record['time_s'])
        if now <= ready:
            previous = record
            continue
        states = record['tracker']['row_states']
        if len(states) != len(model_support):
            raise ValueError('row inventory changed')
        if [state['index'] for state in states] != list(range(len(states))):
            raise ValueError('rows are not the original contiguous index grid')
        q = np.asarray(record['quality_mask'], bool)
        state_q = np.array([s['quality'] for s in states], bool)
        if not np.array_equal(q, state_q):
            raise ValueError('quality mask / tracker evidence mismatch')
        d = np.asarray(record['profile_residual'], float)
        high = q & np.isfinite(d) & (d >= threshold)
        low = q & np.isfinite(d) & (d <= threshold*rules['low_threshold_fraction'])
        confirmed = np.array([s['phase']=='confirmed' for s in states])
        pending = np.array([s['phase']=='pending' for s in states])
        armed = np.array([bool(s['armed']) for s in states])
        active_confirmed = confirmed & q & (d >= threshold*rules['low_threshold_fraction'])
        masks = dict(quality=q, quality_high=high, quality_low=low,
                     confirmed_history=confirmed, confirmed_current=active_confirmed,
                     pending=pending, armed=armed)
        summary = dict(frame_index=record['frame_index'], time_s=now,
            counts={key:int(mask.sum()) for key,mask in masks.items()},
            maximum_adjacent={key:longest(mask) for key,mask in masks.items()},
            high_without_armed_low_count=int(np.sum(high & ~armed)),
            positive_but_below_high_count=int(np.sum(q & (d>0.) & (d<threshold))),
            quality_reason_counts=dict(Counter(record['quality_reason'])),
            gates=dict(three_quality=0, three_quality_high=0, three_confirmed_current=0,
                       ordered_onsets=0, retained_minimum_age=0, within_history=0,
                       recent_last_confirmation=0),
            confirmed_triple_rejections={})
        for start in range(len(states)-size+1):
            ids = list(range(start,start+size))
            if q[ids].all(): summary['gates']['three_quality'] += 1
            if high[ids].all():
                summary['gates']['three_quality_high'] += 1
                examples.setdefault('first_three_quality_high', compact_example(record,ids,'instant high != confirmed front'))
            if not active_confirmed[ids].all():
                continue
            summary['gates']['three_confirmed_current'] += 1
            onsets = [states[j]['first_high_s'] for j in ids]
            why = None
            if not all(b-a>EPS for a,b in zip(onsets,onsets[1:])):
                why = 'onsets_not_strictly_increasing'
            else:
                summary['gates']['ordered_onsets'] += 1
                age = now-onsets[0]
                if age < rules['minimum_first_onset_age_s']-EPS:
                    why = 'first_onset_retention_too_short'
                else:
                    summary['gates']['retained_minimum_age'] += 1
                    if age > rules['maximum_chain_history_s']+EPS:
                        why = 'chain_history_expired'
                    else:
                        summary['gates']['within_history'] += 1
                        confirmation_age = now-states[ids[-1]]['confirmed_at_s']
                        if not 0. <= confirmation_age <= rules['recent_confirmation_s']+EPS:
                            why = 'last_confirmation_not_recent'
                        else:
                            summary['gates']['recent_last_confirmation'] += 1
            if why:
                summary['confirmed_triple_rejections'][why] = summary['confirmed_triple_rejections'].get(why,0)+1
                examples.setdefault('first_'+why, compact_example(record,ids,why))
            else:
                examples.setdefault('first_all_image_gates_pass', compact_example(record,ids,'all image gates pass'))
        if previous is not None and previous['time_s'] > ready:
            old = previous['tracker']['row_states']
            for j,(before,after) in enumerate(zip(old,states)):
                same_onset = before['first_high_s'] is not None and before['first_high_s']==after['first_high_s']
                if after['phase']=='pending' and (before['phase']!='pending' or not same_onset):
                    transitions.append(dict(time_s=now,row_index=j,kind='pending_started'))
                    row_stats[j]['pending_started'] += 1
                if before['phase']=='pending':
                    if after['phase']=='confirmed' and same_onset:
                        kind = 'pending_confirmed'
                    elif after['phase']=='pending' and same_onset:
                        kind = None
                    else:
                        kind = 'pending_cancelled__'+ending_reason(before,after,now,q[j],d[j],threshold[j],rules,record['tracker']['reason'])
                    if kind:
                        transitions.append(dict(time_s=now,row_index=j,kind=kind,
                            first_high_s=before['first_high_s'], elapsed_s=now-before['first_high_s']))
                        row_stats[j][kind] += 1
                        examples.setdefault('first_'+kind, dict(previous=compact_example(previous,[j],'previous pending'),
                                                               current=compact_example(record,[j],kind)))
                if before['phase']=='confirmed' and after['phase']!='confirmed':
                    kind = 'confirmed_ended__'+ending_reason(before,after,now,q[j],d[j],threshold[j],rules,record['tracker']['reason'])
                    transitions.append(dict(time_s=now,row_index=j,kind=kind,
                        first_high_s=before['first_high_s'], confirmed_at_s=before['confirmed_at_s']))
                    row_stats[j][kind] += 1
        for j in range(len(states)):
            row_stats[j].update({key+'_frames':int(mask[j]) for key,mask in masks.items()})
        summaries.append(summary)
        previous = record

    def aggregate(selected, lo=None, hi=None):
        if not selected:
            return dict(frame_count=0)
        totals = Counter()
        rejected = Counter()
        reasons = Counter()
        for s in selected:
            totals.update(s['counts']);rejected.update(s['confirmed_triple_rejections'])
            reasons.update(s['quality_reason_counts'])
        trans = [t for t in transitions if (lo is None or t['time_s']>=lo-EPS)
                 and (hi is None or t['time_s']<=hi+EPS)]
        return dict(frame_count=len(selected), row_grid_size=len(model_support),
            first_time_s=selected[0]['time_s'],last_time_s=selected[-1]['time_s'],
            cell_counts=dict(totals), quality_reason_cell_counts=dict(reasons),
            maximum_adjacent={key:max(s['maximum_adjacent'][key] for s in selected) for key in masks},
            frames_with_at_least_three_adjacent={key:sum(s['maximum_adjacent'][key]>=3 for s in selected) for key in masks},
            adjacency_histograms={key:dict(Counter(str(s['maximum_adjacent'][key]) for s in selected)) for key in masks},
            gate_triple_frame_counts={key:sum(s['gates'][key] for s in selected) for key in selected[0]['gates']},
            gate_frame_counts={key:sum(s['gates'][key]>0 for s in selected) for key in selected[0]['gates']},
            confirmed_triple_rejection_counts=dict(rejected),
            transition_counts=dict(Counter(t['kind'] for t in trans)),
            pending_end_duration_s={kind:quantiles([t['elapsed_s'] for t in trans if t['kind']==kind and 'elapsed_s' in t])
                                   for kind in sorted({t['kind'] for t in trans if 'elapsed_s' in t})},
            high_without_armed_low_count=sum(s['high_without_armed_low_count'] for s in selected),
            positive_but_below_high_count=sum(s['positive_but_below_high_count'] for s in selected))

    lo,hi = reference-3.,reference+3.
    window = [s for s in summaries if lo-EPS<=s['time_s']<=hi+EPS]
    result = dict(total_archived_frames=len(all_records), warmup_frames=sum(r['time_s']<=ready for r in all_records),
        candidate_count=len(payload['candidates']),
        startup=dict(supported_rows=int(model_support.sum()), threshold_px=quantiles(threshold[model_support]),
            supported_threshold_floor_3_count=int(np.sum(model_support & (threshold==3.))),
            supported_threshold_above_10_count=int(np.sum(model_support & (threshold>10.))),
            supported_threshold_above_20_count=int(np.sum(model_support & (threshold>20.)))),
        full_post_startup=aggregate(summaries),
        posthoc_reference_window=dict(reference_time_s=reference,half_width_s=3.,
            inference_use=False,summary=aggregate(window,lo,hi)),
        per_frame=summaries, row_statistics=[dict(index=j,row_y=all_records[0]['tracker']['row_states'][j]['row_y'],
            threshold_px=None if not np.isfinite(threshold[j]) else float(threshold[j]),
            startup_supported=bool(model_support[j]),counts=dict(row_stats[j])) for j in range(len(row_stats))],
        transition_log=transitions,first_examples=examples)
    for key in ('quality','quality_high','confirmed_current'):
        best = max(summaries,key=lambda s:s['maximum_adjacent'][key])
        result.setdefault('maximum_support_examples',{})[key] = dict(frame_index=best['frame_index'],time_s=best['time_s'],
            maximum_adjacent=best['maximum_adjacent'][key],gates=best['gates'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=BASE)
    args=parser.parse_args()
    destination=args.base/'rejection_audit.json'
    if destination.exists():
        raise FileExistsError('do not overwrite a prior audit')
    protocol_path=args.base/'protocol.json'
    protocol=read(protocol_path)
    input_hashes={str(protocol_path):sha(protocol_path)}
    for path,digest in protocol['source_hashes'].items():
        if sha(path)!=digest:raise ValueError('frozen source changed '+path)
    cases={}
    for side in ('left','right'):
        path=args.base/side/'predictions.json'
        input_hashes[str(path)]=sha(path)
        payload=read(path)
        if payload['protocol_sha256']!=input_hashes[str(protocol_path)]:
            raise ValueError('prediction protocol mismatch')
        cases[side]=analyze_session(payload,protocol['rules']['front'],REFERENCE_T1_S[side])
    report=dict(schema='v19_posthoc_rejection_audit_1',
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        script_sha256=sha(__file__),input_hashes=input_hashes,
        reference_windows='retrospective requested 27.3/14.4 +/-3s; never inference inputs',
        units='counts of image grid/frame candidates, not physical accuracy',
        thresholds_changed=False,inference_rerun=False,truth_file_read=False,
        hardware_access=False,cases=cases)
    for path,digest in input_hashes.items():
        if sha(path)!=digest:raise ValueError('input changed during audit '+path)
    with destination.open('x',encoding='utf-8') as handle:
        json.dump(report,handle,ensure_ascii=False,indent=2,allow_nan=False)
    for side,r in cases.items():
        print(side,json.dumps({key:r[key] for key in ('startup','full_post_startup','posthoc_reference_window','maximum_support_examples')},ensure_ascii=False))
    print(destination)


if __name__=='__main__':
    main()
