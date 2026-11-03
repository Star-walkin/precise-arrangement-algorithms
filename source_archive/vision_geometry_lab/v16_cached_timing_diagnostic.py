"""Reproduce a TIMING-LAYER-ONLY diagnostic on historical centreline caches.

Not V16 image/model inference. The historical upstream probe used nearest-row
auxiliary joins (including potentially future rows). A prefix replay cannot
remove that inherited causality limitation. No hardware or neural model loads.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from vision_prototype.centerline_event_fusion import detect_gap_reversal
from vision_geometry_lab.v16_event_candidates import score_candidates


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'vision_geometry_lab/outputs/v16_cached_event_timing'
SESSIONS = ('dual_20260924_111143', 'dual_20260924_111500')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sanitize(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    return value


def evaluate_cache(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    history, first, trace = [], None, []
    previous = -math.inf
    for row in rows:
        now = float(row['time_s'])
        if now <= previous:
            raise ValueError('Cache order is not strictly increasing; do not silently sort.')
        previous = now
        gap = float(row.get('fused_count') or 'nan')
        history.append(dict(time_s=now,
                            fused_valid=str(row.get('fused_valid')).lower() in ('true', '1'),
                            fused_count=gap))
        # Only evidence at or before now; no cached period/truth is supplied.
        decision = detect_gap_reversal(history)
        if not decision.get('valid'):
            continue
        emitted = dict(event='t1', event_time_s=decision['new_layer_start_candidate_s'],
                       first_available_at_s=now,
                       candidate_only=True, physical_layer_confirmed=False,
                       device_advisory=False,
                       evidence=dict(minimum_gap_count=decision['minimum_gap_count'],
                                     approach_slope=decision['approach_slope_count_per_s'],
                                     departure_slope=decision['departure_slope_count_per_s']))
        trace.append(emitted)
        if first is None:
            first = emitted
    return dict(source=str(path), source_sha256=sha256(path), frame_count=len(rows),
                source_time_range_s=[history[0]['time_s'], history[-1]['time_s']],
                first_candidate=first, repeated_prefix_results=trace,
                note='Repeated prefixes are not independent events or extra detections.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report_path = args.output / 'report.json'
    if report_path.exists():
        raise FileExistsError(f'Refusing to overwrite previous audit: {report_path}')
    results = {}
    for session in SESSIONS:
        source = ROOT / 'vision_prototype/output/centerline_sequence_probe' / session / 'centerline_measurements.csv'
        results[session] = evaluate_cache(source)
    # Open validation annotations only after both independent inferences end.
    truth_path = ROOT / 'vision_prototype/validation_ground_truth/dual_reversal_events.json'
    truth = json.loads(truth_path.read_text(encoding='utf-8'))
    for session, result in results.items():
        first = result['first_candidate']
        scored = score_candidates([first] if first else [], truth['sessions'][session]['manual_review'])
        result['post_inference_t1_score'] = scored['events']['t1']
    report = dict(
        status='historical_cache_timing_layer_only_NOT_v16_model_validation',
        protocol='default_gap_reversal_on_each_prefix_period_not_required_first_output_frozen',
        source_sha256=sha256(Path(__file__)),
        detector_sha256=sha256(ROOT / 'vision_prototype/centerline_event_fusion.py'),
        validation_truth_sha256=sha256(truth_path),
        truth_loaded_only_after_all_inference=True,
        limitations=[
            'Historical upstream geometric caches may use nearest-future auxiliary rows.',
            'These are repeatedly seen development videos, not blind event validation.',
            'Gap reversal is a motion proxy, not direct proof of a new radial layer.',
            'No current V14/V16 image model or raw-video inference is tested here.',
            'Timestamp accuracy and first availability are distinct; processing/transport delay is not measured.',
            'No full-turn period supplied, no t0 or t2 prediction evaluated.',
        ],
        sessions=results, hardware_accessed=False, deployment_permitted=False)
    args.output.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(sanitize(report), ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding='utf-8')
    print(json.dumps({session: result['post_inference_t1_score'] for session, result in results.items()},
                     ensure_ascii=False, indent=2))
    print(report_path)


if __name__ == '__main__':
    main()
