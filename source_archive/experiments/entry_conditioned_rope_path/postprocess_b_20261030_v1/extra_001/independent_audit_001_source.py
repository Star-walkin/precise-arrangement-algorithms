"""Independent saved-output audit for H1; never import its detector or runner.

Checks provenance, per-entry separation, saved geometry, independently rasterized
masks and display dimensions. Passing is a software/asset contract result, not a
same-rope identity score or proof that a visible bend was classified correctly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / '数据资产/05_派生与实验结果/camera_condition_images_20261009_v2'
STAGES = ['01_原图', '02_B数值热图', '03_局部候选', '04_几何提示', '05_路径候选', '06_最终叠加']
MASK_NAMES = ['候选绳带', '局部支持绳带']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def decode(path, flags=cv2.IMREAD_COLOR):
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), flags)
    if image is None:
        raise ValueError('Unreadable image: ' + str(path))
    return image


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def finite_tree(item):
    if isinstance(item, float):
        require(np.isfinite(item), 'Nonfinite JSON number')
    elif isinstance(item, list):
        for value in item:
            finite_tree(value)
    elif isinstance(item, dict):
        for value in item.values():
            finite_tree(value)


def safe_child(directory, relative):
    p = Path(relative)
    require(not p.is_absolute() and '..' not in p.parts, 'Unsafe output relative path')
    resolved = (directory / p).resolve()
    require(resolved.is_relative_to(directory.resolve()), 'Output escapes this run')
    return resolved


def check_path(candidate, seed, expected_shape):
    """Validate only persisted geometry; no detector/search function is called."""
    p = np.asarray(candidate['points_xy'], dtype=np.float64)
    require(p.ndim == 2 and p.shape[1] == 2 and len(p) >= 2, 'Invalid point list')
    n = len(p)
    w = np.asarray(candidate['width_px'], dtype=np.float64)
    supported = np.asarray(candidate['supported'])
    condition = np.asarray(candidate['entry_condition'])
    require(w.shape == (n,) and supported.shape == (n,) and condition.shape == (n,), 'Unequal per-point arrays')
    require(supported.dtype == bool and condition.dtype == bool, 'Support/condition must be boolean')
    require(len(candidate['evidence']) == n, 'Evidence array length differs')
    require(np.isfinite(p).all() and np.isfinite(w).all() and (w > 0).all(), 'Invalid coordinates/widths')
    h, image_w = expected_shape
    require(((p[:, 0] >= -.002) & (p[:, 0] <= image_w + .002) &
             (p[:, 1] >= -.002) & (p[:, 1] <= h + .002)).all(), 'Path lies outside source image')
    nc = int(condition.sum())
    conditional_indices = np.flatnonzero(condition)
    require(nc >= 2 and np.all(np.diff(conditional_indices) == 1), 'Entry condition is not one contiguous unchanged segment')
    given = np.asarray(seed['xy'], dtype=np.float64)
    expected = np.linspace(given[0], given[-1], nc)
    seed_error = float(np.max(np.abs(p[condition] - expected)))
    require(seed_error <= .002, 'Initial UNet entry coordinates changed')
    require(not supported[condition].any(), 'Conditional seed falsely counted as fresh image support')
    for i, evidence in enumerate(candidate['evidence']):
        if condition[i]:
            require(evidence is None, 'Expected explicit conditional seed evidence')
        else:
            require(isinstance(evidence, dict) and type(evidence.get('supported')) is bool, 'Missing per-point evidence')
            require(evidence['supported'] == bool(supported[i]), 'Saved support differs from evidence')
    measured = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    require(abs(measured - float(candidate['length_px'])) < .01, 'Saved path length is inconsistent')
    require(isinstance(candidate['termination'], str) and candidate['termination'], 'Missing termination reason')
    require(candidate['physical_contact_confirmed'] is False and candidate['same_root_identity_confirmed'] is False,
            'Unexpected physical/identity confirmation')
    return {'point_count': n, 'conditional_point_count': nc, 'supported_point_count': int(supported.sum()),
            'seed_max_abs_error_px': seed_error, 'termination': candidate['termination']}


def reconstruct_masks(candidates, shape):
    """Rasterize the selected recorded optical ribbon without importing H1."""
    whole = np.zeros(shape, np.uint8)
    local = np.zeros(shape, np.uint8)
    if candidates:
        selected = candidates[0]
        pixels = np.rint(np.asarray(selected['points_xy'], np.float64)).astype(np.int32)
        widths = np.asarray(selected['width_px'], np.float64)
        support = np.asarray(selected['supported'], bool)
        for i in range(1, len(pixels)):
            thickness = max(1, int(round(float(widths[i]))))
            a, b = tuple(pixels[i - 1]), tuple(pixels[i])
            cv2.line(whole, a, b, 255, thickness, cv2.LINE_8)
            if support[i - 1] and support[i]:
                cv2.line(local, a, b, 255, thickness, cv2.LINE_8)
    return whole, local


def check_tail_review(entry):
    """F3 tails remain separate review candidates, not accepted mask geometry."""
    review = entry.get('f3_tail_review')
    if review is None:
        return []
    require(entry.get('f3_tail_coordinate_system') == 'half_size_model_crop_xy; original_xy=2*xy+[0,1050]',
            'Missing explicit local-coordinate declaration for F3 tail')
    tails = review.get('candidates', [])
    require(len(tails) <= 3, 'Too many F3 review lane candidates')
    require(not tails or entry['candidates'], 'F3 tail has no selected image path to attach to')
    results = []
    for tail in tails:
        p = np.asarray(tail['points_xy'], np.float64)
        n = len(p)
        require(p.shape == (n, 2) and n >= 2 and np.isfinite(p).all(), 'Invalid tail points')
        require(((p[:,0] >= 0) & (p[:,0] < 2048) & (p[:,1] >= 0) & (p[:,1] < 975)).all(),
                'F3 tail outside local model ROI')
        for key in ('width_px','f3_pair_supported','image_supported','raw_ribbon_score',
                    'heat_shape_score','heat_level','f3_inferred_at_center'):
            array = np.asarray(tail[key])
            require(array.shape == (n,) and np.isfinite(array).all(), 'Unequal/invalid tail array: ' + key)
        require((np.asarray(tail['width_px']) > 0).all(), 'Invalid tail width')
        require(tail['same_identity_confirmed'] is False and tail['confidence'].startswith('LOW_REVIEW_ONLY'),
                'F3 inferred lane incorrectly confirmed as identity')
        expected = np.asarray(entry['candidates'][0]['points_xy'][-1])
        error = float(np.max(abs(2*p[0]+[0,1050]-expected)))
        require(error <= .01, 'F3 tail starts away from main path endpoint')
        count = int(tail['connector_point_count'])
        require(0 <= count < n, 'Invalid unsupported connector length')
        require(not np.asarray(tail['f3_pair_supported'], bool)[:count].any(), 'Connector incorrectly marked F3 measured')
        require(not np.asarray(tail['image_supported'], bool)[:count].any(), 'Connector incorrectly marked supported')
        require(len(tail['track_upper_lower_y']) == n-count, 'Tail boundary-track length differs')
        delta = np.diff(p, axis=0)
        require((delta[:,0] > 0).all() or (delta[:,0] < 0).all(), 'F3 tail is not monotonic in x')
        observed = np.asarray(tail['f3_pair_supported'], bool)
        require(observed[-1], 'Trailing unsupported F3 extension was not trimmed')
        require(abs(float(np.linalg.norm(delta,axis=1).sum())-tail['length_px']) < .01, 'Tail length inconsistent')
        if 'maximum_polyline_turn_deg_per_optical_width' in tail:
            lengths = np.linalg.norm(delta,axis=1)
            angles = np.unwrap(np.arctan2(delta[:,1],delta[:,0]))
            turns = np.abs(np.diff(angles))/np.maximum((lengths[:-1]+lengths[1:])*.5,1e-6)*tail['optical_reference_width_px']
            actual = float(np.degrees(turns).max()) if len(turns) else 0.
            require(abs(actual-tail['maximum_polyline_turn_deg_per_optical_width']) < 1e-5, 'Tail turn diagnostic differs')
            require(actual <= review['diagnostics']['parameters']['maximum_turn_deg_per_width']+1e-6,
                    'Tail exceeds declared delivered-polyline curvature limit')
        results.append({'point_count': n, 'connector_point_count': count, 'start_error_original_px': error,
                        'F3_pair_supported_points': int(observed.sum()), 'identity_confirmed': False})
    return results


def audit_case(run, job, summary_row, hash_cache):
    def checked(path, expected):
        key = str(Path(path).resolve())
        if key not in hash_cache:
            hash_cache[key] = digest(path)
        require(hash_cache[key] == expected, 'Input hash changed: ' + key)
    record_path = run / '记录' / (job['id'] + '.json')
    record = read(record_path)
    finite_tree(record)
    require(record['id'] == job['id'] and record['entry_count'] == len(job['seeds']), 'Entry/image denominator changed')
    require(len(record['entries']) == len(job['seeds']) == summary_row['entries'], 'Entry count disagrees')
    require(record['original_shape'] == job['shape'] == [3000, 4096], 'Original dimensions changed')
    require(record['display_scale'] == .5, 'Unexpected display scale')
    require(record['source_image_sha256'] == job['image_sha256'], 'Record original hash differs')
    require(Path(record['source_image']).resolve() == Path(job['image_path']).resolve(), 'Original path changed')
    require(record['labels_read'] is False and record['device_connected'] is False, 'Unexpected scope flag')
    require(record['complete_rope_identity_verified'] is False, 'Result overclaims full identity')
    checked(job['image_path'], job['image_sha256'])
    for path, expected in record['input_dependency_sha256'].items():
        checked(path, expected)
    inference = read(HERE / 'responses' / job['id'] / 'inference.json')
    require(inference['id'] == job['id'] and inference['image_sha256'] == job['image_sha256'], 'Inference provenance changed')
    require(inference['entry_conditions_merged'] is False, 'Inference merged separate entries')
    require(len(inference['entry_conditions']) == len(job['seeds']), 'Inference entry denominator differs')
    require(inference['target_labels_read'] is False, 'Unexpected label flag')
    require(inference['original_response_not_minmax_normalized'] is True, 'Numeric response was rescaled per image')
    protocol = read(HERE / 'protocol_inputs.json')
    require(inference['protocol_sha256'] == digest(HERE / 'protocol_inputs.json'), 'Numeric inference protocol changed')
    require(inference['model_sha256'] == protocol['model_B']['sha256'], 'Numeric response is not from frozen model B')
    original = decode(job['image_path'])
    require(original.shape == (3000, 4096, 3), 'Original file has unexpected shape')
    original_half = cv2.resize(original, (2048, 1500), interpolation=cv2.INTER_AREA)
    entries = []
    expected_files = []
    for k, (entry, seed) in enumerate(zip(record['entries'], job['seeds'])):
        require(entry['entry_index'] == k and entry['seed'] == seed, 'Seed or entry order changed')
        g = entry['geometry']
        require(g['seed_xy'] == seed['xy'] and g['seed_entry_unchanged'] is True, 'Prior adapter changed seed')
        require(g['actual_seed_entry_id'] == seed['entry_id'], 'Prior adapter changed seed ID')
        require(Path(g['actual_seed_source']).resolve() == Path(seed['source']).resolve(), 'Prior adapter changed seed source')
        require(abs(float(g['width_px']) - 2 * float(g['width_half_px'])) < 1e-7, 'Width scale inconsistent')
        response_path = HERE / 'responses' / job['id'] / f'entry_{k:02d}.npz'
        inf = inference['entry_conditions'][k]
        require(inf['entry_index'] == k and inf['seed'] == seed, 'Response entry does not match original condition')
        require(Path(inf['response_path']).resolve() == response_path.resolve(), 'Numeric inference response path differs')
        require(inf['response_sha256'] == entry['response_sha256'], 'Record numeric response hash differs')
        checked(response_path, inf['response_sha256'])
        with np.load(response_path, allow_pickle=False) as archive:
            response = archive['response']
            require(response.shape == (975, 2048) and response.dtype == np.float32, 'Response array shape/dtype differs')
            require(np.isfinite(response).all() and response.min() >= 0 and response.max() <= 1, 'Invalid response values')
        candidates = entry['candidates']
        require(entry['selected_candidate_index'] == (0 if candidates else None), 'Unexpected selected candidate')
        require(entry['full_identity_confirmed'] is False, 'Entry overclaims identity')
        candidate_checks = [check_path(c, seed, (3000, 4096)) for c in candidates]
        baseline_checks = [check_path(c, seed, (3000, 4096)) for c in entry['heat_only_baseline']]
        tail_checks = check_tail_review(entry)
        require(summary_row['candidate_counts'][k] == len(candidates), 'Summary candidate count differs')
        expected_length = candidates[0]['length_px'] if candidates else 0
        require(abs(summary_row['top_lengths_px'][k] - expected_length) < 1e-6, 'Summary selected length differs')
        name = job['id'] + f'__entry{k+1:02d}'
        errors = None
        for stage in STAGES:
            path = run / '分步图像' / stage / (name + '.jpg')
            image = decode(path)
            require(image.shape == (1500, 2048, 3), 'Step display dimensions differ: ' + str(path))
            expected_files.append(path.relative_to(run).as_posix())
            if stage == STAGES[0]:
                delta = np.abs(image.astype(np.int16) - original_half.astype(np.int16))
                errors = {'mean_abs_channel_error': float(delta.mean()), 'p99_abs_channel_error': float(np.percentile(delta, 99))}
                require(errors['mean_abs_channel_error'] <= 5 and errors['p99_abs_channel_error'] <= 40,
                        'Original display differs beyond declared JPEG tolerance')
        reconstructed = reconstruct_masks(candidates, (3000, 4096))
        mask_checks = []
        for kind, expected in zip(MASK_NAMES, reconstructed):
            path = run / '掩膜' / kind / (name + '.png')
            actual = decode(path, cv2.IMREAD_UNCHANGED)
            require(actual.shape == (3000, 4096) and actual.dtype == np.uint8, 'Mask shape/dtype differs')
            require(np.logical_or(actual == 0, actual == 255).all(), 'Mask is not binary 0/255')
            require(np.array_equal(actual, expected), 'Mask differs from independent JSON ribbon reconstruction')
            expected_files.append(path.relative_to(run).as_posix())
            mask_checks.append({'kind': kind, 'nonzero_pixels': int(np.count_nonzero(actual)), 'matches_json_raster': True})
        require(not np.any((reconstructed[1] > 0) & (reconstructed[0] == 0)), 'Local support is outside candidate ribbon')
        entries.append({'entry_index': k, 'entry_id': seed['entry_id'], 'candidate_checks': candidate_checks,
                        'heat_only_checks': baseline_checks, 'F3_tail_review_checks': tail_checks,
                        'masks': mask_checks, 'jpeg_original_difference': errors})
    strip = run / '长条对照图' / (job['id'] + '.jpg')
    strip_image = decode(strip)
    require(strip_image.shape == (1600 * len(job['seeds']), 12288, 3), 'Strip dimensions or separate-entry row count differ')
    expected_files.append(strip.relative_to(run).as_posix())
    listed = [Path(v).as_posix() for v in record['files']]
    require(len(listed) == len(set(listed)) and set(listed) == set(expected_files), 'Saved file list incomplete or duplicated')
    expected_files.append(record_path.relative_to(run).as_posix())
    output_files = [{'path': relative, 'bytes': safe_child(run, relative).stat().st_size,
                     'sha256': digest(safe_child(run, relative))} for relative in expected_files]
    return {'id': job['id'], 'entries': entries, 'strip_shape': list(strip_image.shape),
            'input_hashes_checked': len(record['input_dependency_sha256']), 'files': output_files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', default='independent_audit.json')
    parser.add_argument('--require-current-sources', action='store_true')
    args = parser.parse_args()
    run = (HERE / args.run).resolve()
    require(run.is_relative_to(HERE.resolve()) and run != HERE.resolve(), 'Audit target must be a child run')
    output = safe_child(run, args.output)
    require(not output.exists(), 'Do not overwrite an existing audit result')
    started = time.time(); cases = []; failures = []; hash_cache = {}; global_checks = {}
    try:
        start = read(run / 'run_start.json'); summary = read(run / 'summary.json')
        protocol_path = HERE / 'protocol_inputs.json'; protocol = read(protocol_path)
        require(digest(protocol_path) == start['protocol_sha256'], 'Input protocol hash changed')
        jobs = {j['id']: j for j in protocol['jobs']}
        require(len(jobs) == 292 and sum(len(j['seeds']) for j in jobs.values()) == 299, 'Expected292 images/299 conditions')
        require(protocol['source_count'] == 400 and protocol['upstream_skipped_count'] == len(protocol['upstream_skipped']) == 108,
                'Dropped/changed skipped upstream denominator')
        with (DATA / 'manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        allowed = {Path(r['output_file']).stem: r for r in rows}
        require(len(allowed) == 400, 'Original400 manifest denominator differs')
        for job in jobs.values():
            row = allowed[job['id']]
            require('20260925' not in row['source_file'] + row['output_file'], 'Reserved source in job manifest')
            require(not any(s in row['dataset_use_status'].lower() for s in ('reserved', 'heldout')), 'Reserved status in job manifest')
            require(row['output_sha256'] == job['image_sha256'], 'Job original hash differs from open400 manifest')
        ids = start['input_ids']
        require(len(ids) == len(set(ids)) == start['planned'] and set(ids) <= set(jobs), 'Invalid planned input list')
        command = start['command']; subset = command[command.index('--subset') + 1] if '--subset' in command else 'pilot'
        if subset == 'all':
            require(set(ids) == set(jobs), 'All run omits input images')
        elif subset == 'pilot':
            require(set(ids) == set(protocol['pilot_ids']), 'Pilot does not use frozen list')
        elif subset == 'extra':
            other = [x for x in jobs if x not in protocol['pilot_ids']]
            expected = sorted(other, key=lambda s: hashlib.sha256(('H1extra:' + s).encode()).hexdigest())[:8]
            require(set(ids) == set(expected), 'Extra run is not deterministic preselected list')
        else:
            raise AssertionError('Unknown declared subset')
        completed = {x['id']: x for x in summary['rows']}
        failed = {x['id']: x for x in summary['failures']}
        require(not set(completed) & set(failed) and set(completed) | set(failed) == set(ids), 'Failures not retained in planned denominator')
        require(summary['planned'] == len(ids) and summary['completed'] == len(completed), 'Summary counts differ')
        require(summary['entries'] == sum(len(jobs[x]['seeds']) for x in completed), 'Summary entry count differs')
        for case_id, failure in failed.items():
            saved = read(run / (case_id + '__failure.json'))
            require(saved == failure, 'Failure details were altered/dropped')
        require(summary['exit_code'] == (1 if failed else 0), 'Exit code disagrees with failures')
        if summary['source_unchanged'] is not True:
            failures.append({'scope': 'execution_source_integrity', 'reason': 'Algorithm source changed during execution'})
        source_checks = []
        for path, expected in start['sources'].items():
            snapshot = run / 'source_snapshot' / Path(path).name
            require(digest(snapshot) == expected, 'Execution source snapshot differs from startup hash')
            current_matches = digest(path) == expected
            source_checks.append({'path': path, 'snapshot_matches': True, 'current_matches': current_matches})
            if args.require_current_sources:
                require(current_matches, 'Runtime source has changed since run')
        frozen = []
        for item in protocol['frozen_sources']:
            require(digest(item['path']) == item['sha256'], 'Frozen upstream source changed: ' + item['path'])
            frozen.append({'path': item['path'], 'matches': True})
        global_checks = {'subset': subset, 'source400': 400, 'eligible_images': 292, 'eligible_entries': 299,
            'upstream_skipped': 108, 'planned': len(ids), 'completed': len(completed), 'failed': len(failed),
            'source_checks': source_checks, 'frozen_upstream_checks': frozen, 'run_exit_code': summary['exit_code'],
            'same_rope_accuracy_evaluated': False, 'current_source_equality_required': args.require_current_sources}
        for index, case_id in enumerate(ids):
            if case_id not in completed:
                continue
            try:
                cases.append(audit_case(run, jobs[case_id], completed[case_id], hash_cache))
            except Exception:
                failures.append({'id': case_id, 'error': traceback.format_exc()})
            if (index + 1) % 20 == 0 or index + 1 == len(ids):
                print(json.dumps({'audited': len(cases), 'audit_failures': len(failures), 'planned': len(ids)}), flush=True)
        global_checks['input_unique_hashes_checked'] = len(hash_cache)
        # An execution failure remains an incomplete delivery even when its
        # failure record itself is correctly preserved and audit-readable.
        if failed:
            failures.append({'scope': 'execution', 'reason': 'run_has_preserved_processing_failures', 'ids': list(failed)})
    except Exception:
        failures.append({'scope': 'global', 'error': traceback.format_exc()})
    all_files = [f for case in cases for f in case['files']]
    counts = {'cases': len(cases), 'entries': sum(len(c['entries']) for c in cases),
              'output_files': len(all_files), 'candidates': sum(len(e['candidate_checks']) for c in cases for e in c['entries'])}
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'command': [sys.executable, *sys.argv],
              'run': str(run), 'audit_source_sha256': digest(__file__), 'exit_code': 1 if failures else 0,
              'passed': not failures, 'counts': counts, 'global_checks': global_checks,
              'jpeg_tolerance': {'mean_abs_channel_error_max': 5, 'p99_abs_channel_error_max': 40,
                                 'pixel_exactness_claim': False},
              'limitations': 'Structural/provenance/display audit only; no accuracy, true endpoint or same-rope identity evaluated.',
              'cases': cases, 'failures': failures, 'elapsed_seconds': time.time() - started}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    snapshot = output.with_name(output.stem + '_source.py')
    require(not snapshot.exists(), 'Do not overwrite previous audit source')
    shutil.copy2(__file__, snapshot)
    print(json.dumps({'passed': report['passed'], 'exit_code': report['exit_code'], 'counts': counts,
                      'audit_failures': len(failures), 'seconds': round(report['elapsed_seconds'], 2)}), flush=True)
    return report['exit_code']


if __name__ == '__main__':
    sys.exit(main())
