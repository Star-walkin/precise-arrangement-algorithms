"""Independent read-only invariants for the full fixed-d delivery; no inference."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json
import math


ROOT = Path(__file__).resolve().parent
EXPECTED_SOURCE_SHA = 'ead815ac73239dec6fcbd141ffb2873fb2ac575dfb79347a036ee877b65255f8'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(a, b):
    return finite(a) and finite(b) and abs(a-b) <= 1.e-7


def audit(data, require_video=False):
    receipt = read(data/'receipt.json')
    binding = read(data/'binding.json')
    issues = []
    def check(condition, message):
        if not condition:
            issues.append(message)
    check(receipt['status'] == 'completed', 'analysis not completed')
    check(receipt['source_sha256'] == binding['source_sha256'] == EXPECTED_SOURCE_SHA, 'source SHA binding')
    check(sha(Path(receipt['source_video'])) == EXPECTED_SOURCE_SHA, 'source bytes changed')
    check(receipt['source_frames'] == receipt['processed_frames'] == binding['frame_stop_exclusive'] == 601, 'not all 601 frames')
    check(receipt['frame_stride'] == 1 and receipt['source_frame_range_exclusive'] == [0,601], 'source frame range/stride')
    check(receipt['binding'] == binding, 'receipt binding differs')
    check(binding['surface_coordinates'] == 'original_unmirrored_xy', 'coordinate system')
    check(binding['fixed_d_original_px'] == receipt['fixed_d_original_px'] == 2238.7, 'user d changed')
    check(binding['entry_threshold_pitch'] == 5., 'threshold changed')
    check(binding['seed_length_px'] == 120. and binding['minimum_S1_support_span_px'] == 400., 'anchored policy changed')
    sources = [
        ('tracker_sha256', 'live_s1_tracker_anchored.py', 'tracker_source_snapshot.py'),
        ('tracker_base_sha256', 'live_s1_tracker.py', 'tracker_base_source_snapshot.py'),
        ('script_sha256', 'run_full_fixed_d_anchored.py', 'runner_source_snapshot.py'),
        ('gate_sha256', 'five_pitch_gate.py', 'gate_source_snapshot.py'),
        ('reacquisition_module_sha256', 'run_full_fixed_d.py', 'reacquisition_source_snapshot.py'),
    ]
    for key, current, snapshot in sources:
        check(sha(ROOT/current) == sha(data/snapshot) == binding[key], 'source/snapshot mismatch: '+current)
    baseline = ROOT.parent/'LC1_locked_column_20261105_v1'
    baseline_unchanged = True
    for name, digest in binding['detector_hashes'].items():
        matched = sha(baseline/name) == digest
        baseline_unchanged &= matched
        check(matched, 'frozen baseline changed: '+name)
    scene = receipt['static_scene_calibration']
    check(sha(Path(scene['mask_path'])) == binding['mask_sha256'], 'static mask bytes changed')
    paths = sorted((data/'frames').glob('frame_*.json'))
    check([p.name for p in paths] == [f'frame_{i:06d}.json' for i in range(601)], 'missing/extra/noncontiguous frame files')
    with (data/'measurements.csv').open(encoding='utf-8-sig', newline='') as stream:
        table = list(csv.DictReader(stream))
    check(len(table) == 601, 'CSV does not have 601 rows')
    check(sha(data/'measurements.csv') == receipt['measurements_sha256'], 'CSV hash mismatch')
    counts = Counter()
    regions = Counter()
    first_inside = None
    last_t = None
    last_anchor = None
    max_drift = 0.
    anchor_events = []
    unknown_gate_reasons = Counter()
    unknown_surface_reasons = Counter()
    for expected_i, path in enumerate(paths):
        r = read(path)
        i = r['frame_index']; t = r['time_s']
        prefix = f'f{i}: '
        tr = r['S1_tracker']; sf = r['surface']; pr = r['prediction']
        valid = tr.get('valid') is True
        abc = tr.get('current_line_abc'); y = r.get('intersection_y_px')
        check(i == expected_i, prefix+'wrong index')
        check(finite(t) and (last_t is None or t > last_t), prefix+'nonincreasing/nonfinite source time')
        last_t = t
        check(close(r['source_nominal_time_s'], i/receipt['source_fps']), prefix+'wrong nominal time')
        check(r['fixed_d_original_px'] == 2238.7, prefix+'wrong d')
        check(pr['frame_index'] == i and pr['time_s'] == t, prefix+'gate chronology binding')
        check(sf.get('measurement_x') == 2238.7, prefix+'F3 column binding')
        check(sf.get('coordinate_system') == 'input_full_image_xy', prefix+'F3 coordinates')
        check(sf['diagnostics'].get('frame_resize') is False and sf['diagnostics'].get('frame_mirror') is False,
              prefix+'F3 resized/mirrored input')
        calls = tr['diagnostics']['S1_calls']
        check(type(calls) is int and calls >= 0, prefix+'bad S1 call counter')
        counts['actual_S1_core_calls'] += calls
        counts['frames_with_S1_core_call'] += calls > 0
        counts['frames_without_verified_seed_or_S1_call'] += calls == 0
        counts['S1_valid_frames'] += valid
        counts['current_F3_calls'] += 1
        counts['F3_valid_count_frames'] += sf.get('count_valid') is True
        counts['actual_P1_reacquisition_forwards'] += tr.get('model_forward_count',0)
        counts['reacquisition_callback_calls'] += tr['diagnostics'].get('callback_called') is True
        reacquire = tr['diagnostics'].get('reacquire', {})
        counts['reused_current_frame_P1_seed_count'] += 'cached_observation' in reacquire
        check(valid or abc is None, prefix+'stale current ABC on failure')
        check(valid or tr.get('line_points_xy') == [], prefix+'stale current measured points on failure')
        if valid:
            check(calls > 0 and tr['S1']['diagnostics'].get('s1_core_executed') is True, prefix+'valid without actual current S1 core')
            check(abc == tr['S1'].get('line_abc'), prefix+'current ABC is not current S1 ABC')
            check(len(abc) == 3 and all(finite(v) for v in abc), prefix+'nonfinite current ABC')
            measured = tr.get('line_points_xy', [])
            seed_for_span = tr.get('seed_xy_original')
            check(len(measured) >= 24 and seed_for_span is not None, prefix+'insufficient current measured support')
            if seed_for_span is not None and measured:
                dx = seed_for_span[1][0]-seed_for_span[0][0]
                dy = seed_for_span[1][1]-seed_for_span[0][1]
                length = math.hypot(dx,dy)
                projections = [(p[0]*dx+p[1]*dy)/length for p in measured]
                check(all(finite(v) for v in projections) and max(projections)-min(projections) >= 400.-1.e-6,
                      prefix+'current measured support span is below declared 400 px')
            if abs(abc[1]) > .05:
                check(close(y, -(abc[0]*2238.7+abc[2])/abc[1]), prefix+'current line/column intersection')
        anchor = tr['diagnostics'].get('spatial_anchor', {})
        ax = anchor.get('x_original'); established = anchor.get('established_frame')
        seed = tr.get('seed_xy_original'); drift = anchor.get('seed_start_x_drift_px')
        if ax is not None:
            check(finite(ax) and type(established) is int and established <= i, prefix+'future/unbound anchor')
            previous_reset_count = 0 if last_anchor is None else last_anchor[0]
            if anchor['reset_count'] > previous_reset_count:
                anchor_events.append(dict(frame_index=i,time_s=t,source_nominal_time_s=i/receipt['source_fps'],
                    anchor_x_original=ax,reset_count=anchor['reset_count'],
                    count_increment=anchor['reset_count']-previous_reset_count,
                    source=anchor.get('source'),reason=anchor.get('reason'),
                    initial_establishment=last_anchor is None))
            if last_anchor is not None:
                old_count, old_x = last_anchor
                reset_count = anchor['reset_count']
                check(reset_count >= old_count, prefix+'anchor reset counter decreased')
                if reset_count == old_count:
                    check(close(ax, old_x), prefix+'anchor changed without reacquisition')
                else:
                    check(established == i and tr['diagnostics'].get('callback_called') is True,
                          prefix+'anchor reset without current reacquisition')
            last_anchor = (anchor['reset_count'], ax)
            if seed is not None:
                check(close(drift, seed[0][0]-ax), prefix+'reported seed drift does not match coordinates')
                if finite(drift):
                    max_drift = max(max_drift, abs(drift))
                    check(abs(drift) <= 1.e-7, prefix+'seed marched from anchor')
        check(not valid or ax is not None, prefix+'valid S1 lacks anchored E1')
        number = sf.get('continuous_pitch_distance')
        usable = (valid and finite(y) and sf.get('count_valid') is True
                  and sf.get('edge_current_frame_confirmed') is True
                  and finite(number) and number >= 0)
        if usable:
            endpoints = sf.get('diagnostics',{}).get('count',{}).get('interval_y', [])
            check(len(endpoints) == 2 and close(endpoints[1], y), prefix+'F3 active endpoint binding')
        expected_region = ('inside' if number <= 5. else 'outside') if usable else 'unknown'
        check(pr['current_region'] == expected_region, prefix+'gate used unavailable/stale/candidate distance')
        check(pr['emit_distance_estimate'] is (expected_region == 'inside'), prefix+'emit policy')
        check(pr['current_N'] == (number if usable else None), prefix+'current N does not match current valid measurement')
        check(pr['future_frames_used'] is False and pr['physical_contact_confirmed'] is False,
              prefix+'future/physical-contact claim')
        check(pr['integer_remaining_rope_count'] is None, prefix+'uncalibrated integer count')
        regions[expected_region] += 1
        if expected_region == 'unknown':
            unknown_gate_reasons[pr['reason']] += 1
            unknown_surface_reasons.update(sf.get('reasons', []))
        counts['prediction_frames'] += expected_region == 'inside'
        if expected_region == 'inside' and first_inside is None:
            first_inside = i
        check(pr['first_inside_confirmed_this_frame'] is (i == first_inside), prefix+'first-inside flag')
        if expected_i < len(table):
            row = table[expected_i]
            check(int(row['frame_index']) == i, prefix+'CSV index')
            check(int(row['S1_calls']) == calls, prefix+'CSV S1 counter')
            check(row['current_region'] == expected_region, prefix+'CSV region')
            display = row['display_distance_N']
            check((display == '') if expected_region != 'inside' else close(float(display), number),
                  prefix+'CSV displayed number outside/unknown or wrong inside number')
    for key, value in counts.items():
        check(value == receipt.get(key), 'receipt counter mismatch: '+key)
    check(dict(regions) == receipt['state_counts'], 'receipt region counts mismatch')
    check(first_inside == receipt['first_inside_confirmation_frame'], 'receipt first inside mismatch')
    check(receipt['gate_consumption'] == 'FIFO_source_frame_order', 'FIFO consumption claim')
    video_info = None
    video_dir = ROOT/('video' if data.name == 'analysis' else 'video_'+data.name)
    if (video_dir/'receipt.json').exists():
        video = read(video_dir/'receipt.json')
        check(video['status'] == 'completed' and video['decoded_frames'] == 601, 'video frame count/completion')
        check(video['frame_stride'] == 1 and video['output_fps'] == receipt['source_fps'], 'video rate/stride')
        check(video['source_pixel_hashes_matched_for_every_frame'] is True, 'video/source pixel validation absent')
        check(video['outside_and_unknown_display_distance'] is False, 'video stale/outside display claim')
        check(video['analysis_receipt_sha256'] == sha(data/'receipt.json'), 'video linked wrong analysis')
        check(video['video_sha256'] == sha(Path(video['video'])), 'video SHA mismatch')
        check(video['renderer_sha256'] == sha(ROOT/'render_full_fixed_d.py'), 'renderer changed')
        video_info = dict(path=video['video'],decoded_frames=video['decoded_frames'],sha256=video['video_sha256'])
    elif require_video:
        check(False, 'video receipt missing')
    return dict(status='passed' if not issues else 'failed',issues=issues,
        analysis_directory=str(data),source_frame_files=len(paths),csv_rows=len(table),
        recomputed_counts=dict(counts),recomputed_state_counts=dict(regions),
        unknown_gate_reason_frequencies=dict(unknown_gate_reasons),
        unknown_F3_reason_frequencies=dict(unknown_surface_reasons),
        anchor_initial_establishment_and_reacquisition_events=anchor_events,
        anchor_establishment_total_counter=0 if last_anchor is None else last_anchor[0],
        anchor_reacquisition_events_after_first=max(0,len(anchor_events)-1),
        first_single_frame_inside_confirmation=first_inside,max_abs_seed_start_x_drift_px=max_drift,
        all_601_source_indices_checked=True,all_coordinate_and_current_gate_bindings_checked=True,
        source_snapshot_hashes_checked=True,old_baseline_sources_unchanged=baseline_unchanged,
        video=video_info,video_required=require_video,
        visual_accuracy_and_material_rope_identity_validated=False,
        scope='Read-only provenance/current-measurement/display invariant audit; no NN or S1/F3 inference rerun.',
        script_sha256=sha(Path(__file__)),analysis_receipt_sha256=sha(data/'receipt.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-dir', default='analysis_anchored')
    parser.add_argument('--require-video', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'audit_full_fixed_d_anchored.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve completed audit; choose a new output path')
    result = audit(ROOT/args.analysis_dir, args.require_video)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
    if result['issues']:
        raise RuntimeError('Audit failed; inspect issues')


if __name__ == '__main__':
    main()
