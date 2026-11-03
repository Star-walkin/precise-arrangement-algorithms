"""A7独立输出审计；不导入/运行detector，不评价真实同根识别准确率。"""
import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import cv2
import numpy as np
import a7_io as io

CORE = ['a7_io.py', 'free_rope.py', 'free_seed_recovery.py', 'strand_signature.py',
        'straight_detector.py', 'run_experiment.py', 'protocol.json']
GROUPS = {'DETECTED': '01_已检出', 'AMBIGUOUS': '02_待确认', 'UNKNOWN': '03_未检出'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''): h.update(block)
    return h.hexdigest()


def load(path): return json.loads(Path(path).read_text('utf-8-sig'))


def image(path):
    value = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    if value is None: raise ValueError('Cannot decode ' + str(path))
    return value


def finite(value):
    if isinstance(value, float): return np.isfinite(value)
    if isinstance(value, dict): return all(finite(v) for v in value.values())
    if isinstance(value, list): return all(finite(v) for v in value)
    return True


def runs(flag):
    changes = np.diff(np.r_[False, flag, False].astype(np.int8))
    return [[int(a), int(b)] for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))]


def audit_sources(run):
    errors, checks, historical, manifests = [], [], [], []
    start = load(run / 'run_start.json')
    protocol = load(run / 'source_snapshot/protocol.json')
    expected_names = {str(io.HERE / f) for f in CORE}
    expected_names.add(str(io.ASSETS / 'camera_condition_incoming_rope_20261026_v2/free_refinement.py'))
    if set(start['sources_sha256']) != expected_names: errors.append('Unexpected runtime source list')
    for filename, expected in start['sources_sha256'].items():
        current = digest(filename)
        frozen = digest(run / 'source_snapshot' / Path(filename).name)
        ok = expected == current == frozen
        checks.append({'source': filename, 'expected': expected, 'current': current, 'snapshot': frozen, 'ok': ok})
        if not ok: errors.append('Source mismatch: ' + filename)
    if (run / 'freeze_snapshot.json').exists():
        if load(run / 'freeze_snapshot.json')['sources_sha256'] != start['sources_sha256']:
            errors.append('Freeze does not match executed snapshot')
    for field, target in [('source_parent_sha256', historical), ('parent_manifest_sha256', manifests)]:
        for filename, expected in protocol[field].items():
            actual = digest(filename)
            target.append({'path': filename, 'expected': expected, 'actual': actual, 'ok': actual == expected})
            if actual != expected: errors.append('Historical bytes changed: ' + filename)
    tree = ast.parse((run / 'source_snapshot/straight_detector.py').read_text('utf-8-sig'))
    forbidden = {'merged', 'f3_record', 'curve_refs', 'inferred', 'exclusion', 'candidate_pool'}
    found = sorted({node.slice.value for node in ast.walk(tree) if isinstance(node, ast.Subscript)
                    and isinstance(node.slice, ast.Constant) and node.slice.value in forbidden})
    if found: errors.append('Forbidden upstream prediction keys in detector')
    return errors, checks, historical, manifests, {'forbidden_prediction_keys': found,
        'scope': 'Frozen source literal-key check plus five exact input dependencies; not a general static taint proof.'}


def candidate_geometry(candidate, shape, parameters):
    errors = []
    visible = np.asarray(candidate['visible'])
    if visible.dtype != np.bool_ or visible.ndim != 1: errors.append('Non-Boolean visibility')
    n = len(visible)
    geometry = {}
    for key in ('center_xy', 'left_xy', 'right_xy'):
        points = candidate[key]
        if len(points) != n: errors.append('Mismatched ' + key); continue
        converted = []
        for point, shown in zip(points, visible):
            if point is None and not shown: converted.append([np.nan, np.nan]); continue
            value = np.asarray(point, float)
            if value.shape != (2,) or not np.isfinite(value).all(): errors.append('Invalid ' + key); break
            if shown and not (0 <= value[0] < shape[1] and 0 <= value[1] < shape[0]):
                errors.append('Visible point out of bounds')
            converted.append(value)
        geometry[key] = np.asarray(converted)
    width = float(candidate['width_px'])
    if not np.isfinite(width) or width <= 0: errors.append('Invalid pixel width')
    if not np.isfinite(candidate['score']): errors.append('Invalid score')
    for key in ('accepted', 'plausible'):
        if key in candidate and not isinstance(candidate[key], bool): errors.append('Non-Boolean ' + key)
    for key in ('raw_supported', 'outside', 'mixed_background_supported', 'local_repeat_supported'):
        if key in candidate and (len(candidate[key]) != n or any(not isinstance(v, bool) for v in candidate[key])):
            errors.append('Point field length/type mismatch: ' + key)
    if not errors and 'mixed_background_supported' in candidate:
        added = np.asarray(candidate['mixed_background_supported'], bool)
        repeat = np.asarray(candidate.get('local_repeat_supported', []), bool)
        supported = np.asarray(candidate.get('raw_supported', []), bool)
        if len(repeat) != n or len(supported) != n:
            errors.append('Missing support needed to audit mixed-background additions')
        elif np.any(added & (~repeat | ~supported)):
            errors.append('Mixed-background additions lack local repeat/raw support')
    if 'pieces' in candidate and candidate['pieces'] != runs(visible): errors.append('Visible pieces mismatch')
    maximum_filled_gap_samples = 0
    if not errors and 'raw_supported' in candidate:
        raw = np.asarray(candidate['raw_supported'], bool)
        # Short measured gaps are intentionally allowed. This checks saved
        # support bookkeeping, not whether a dark object was physically correct.
        small_limit = max(1, int(parameters['maximum_measured_gap_widths'] * width / parameters['step_px']))
        maximum_filled_gap_samples = max([b - a for a, b in runs(visible & ~raw)] or [0])
        if maximum_filled_gap_samples > small_limit:
            errors.append('Visible segment fills more than the permitted short unsupported gap')
    for piece in candidate.get('reacquired_pieces', []):
        a, b = piece['start_index'], piece['end_index']
        if piece.get('gap_filled') is not False:
            errors.append('Reacquired piece declares gap filling')
        if not isinstance(a, int) or not isinstance(b, int) or not 0 <= a < b <= n:
            errors.append('Reacquired piece index outside saved candidate'); continue
        if not np.all(visible[a:b]): errors.append('Reacquired piece is not visible')
        repeat = np.asarray(candidate.get('local_repeat_supported', []), bool)
        if len(repeat) != n:
            errors.append('Reacquired piece lacks local repeat flags'); continue
        fraction = float(repeat[a:b].mean())
        if not np.isclose(fraction, piece['local_repeat_fraction'], atol=1e-9, rtol=0):
            errors.append('Reacquired local repeat fraction differs from saved point flags')
        if fraction < parameters['reacquisition_minimum_repeat_fraction']:
            errors.append('Reacquired piece lacks the required local repeat support')
    deviation = None
    if not errors and n and 'direction_xy' in candidate and 'origin_xy' in candidate:
        direction = np.asarray(candidate['direction_xy'], float)
        origin = np.asarray(candidate['origin_xy'], float)
        if direction.shape != (2,) or origin.shape != (2,) or not np.isclose(np.linalg.norm(direction), 1., atol=1e-6, rtol=0):
            errors.append('Invalid measured axis')
        else:
            normal = np.array([-direction[1], direction[0]])
            center = geometry['center_xy']; valid = np.isfinite(center).all(axis=1)
            deviation = float(np.max(np.abs((center[valid] - origin) @ normal))) if valid.any() else 0.
            if deviation > 1e-4: errors.append('Candidate center is not a straight axis')
            for key, sign in [('left_xy', -1), ('right_xy', 1)]:
                available = valid & np.isfinite(geometry[key]).all(axis=1)
                if available.any() and not np.allclose(geometry[key][available], center[available] + sign * width * normal / 2, atol=1e-4, rtol=0):
                    errors.append('Sides do not follow reported straight axis/width')
    return errors, geometry, visible, {'point_count': n, 'visible_count': int(visible.sum()),
                                      'mixed_background_supported_count': int(sum(candidate.get('mixed_background_supported', []))),
                                      'local_repeat_supported_count': int(sum(candidate.get('local_repeat_supported', []))),
                                      'reacquired_piece_count': len(candidate.get('reacquired_pieces', [])),
                                      'maximum_filled_short_gap_samples': maximum_filled_gap_samples,
                                      'straight_axis_max_error_px': deviation}


def expected_mask(shape, candidates):
    result = np.zeros(shape, np.uint8)
    for candidate in candidates:
        if not candidate['accepted']: continue
        visible = candidate['visible']
        for i in range(1, len(visible)):
            if visible[i - 1] and visible[i]:
                polygon = [candidate['left_xy'][i - 1], candidate['right_xy'][i - 1],
                           candidate['right_xy'][i], candidate['left_xy'][i]]
                cv2.fillConvexPoly(result, np.rint(polygon).astype(np.int32), 255)
    return result


def expected_overlay(original, status, mask, candidates, version):
    result = original.copy(); active = mask > 0; green = (65, 230, 80)
    result[active] = (result[active] * .82 + np.asarray(green) * .18).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, green, 1)
    displayed = []
    for candidate in candidates:
        shown = (status == 'DETECTED' and candidate['accepted']) or (status == 'AMBIGUOUS' and candidate.get('plausible', False))
        if not shown: continue
        displayed.append(candidate['candidate_id']); color = green if status == 'DETECTED' else (0, 170, 255)
        for key in ('left_xy', 'right_xy'):
            points = candidate[key]
            for i in range(1, len(points)):
                if candidate['visible'][i - 1] and candidate['visible'][i]:
                    cv2.line(result, tuple(np.rint(points[i - 1]).astype(int)), tuple(np.rint(points[i]).astype(int)), color, 2, cv2.LINE_8)
    cv2.rectangle(result, (12, 12), (min(2200, original.shape[1] - 1), 112), (20, 20, 20), -1)
    cv2.putText(result, version + '  ' + status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1., (255, 255, 255), 2)
    cv2.putText(result, 'Green: detected straight body | Orange: unconfirmed candidate | Gaps are not bridged',
                (30, 92), cv2.FONT_HERSHEY_SIMPLEX, .75, (230, 230, 230), 1)
    return result, displayed


def audit_case(row, run, manifest, version):
    answer = {'input': row['output_file'], 'camera_role': row['camera_role'], 'errors': [], 'files': []}
    try:
        relative = Path(row['output_file']); role, stem = relative.parent.name, relative.stem
        record_path = run / '记录' / role / (stem + '.json')
        doc = load(record_path); record, result = doc['record'], doc['result']; status = result['status']
        if not finite(doc): answer['errors'].append('Nonfinite JSON')
        if status not in GROUPS: raise ValueError('Invalid status')
        answer['status'] = status
        if record['output_file'] != row['output_file'] or record['camera_role'] != row['camera_role'] or record['status'] != status:
            answer['errors'].append('Wrong case identity/status')
        for key in ('source_group', 'source_time_s', 'dataset_use_status'):
            if record.get(key) != row[key]: answer['errors'].append('Case metadata changed: ' + key)
        paths = io.input_paths(row); input_checks = []
        if set(record['input_dependency_sha256']) != {str(path) for path in paths.values()}:
            answer['errors'].append('Input set differs from the five allowed dependencies')
        for key, path in paths.items():
            actual = digest(path)
            expected = row['output_sha256'] if key == 'original' else io.checked(path)
            ok = actual == expected == record['input_dependency_sha256'].get(str(path))
            input_checks.append({'path': str(path), 'sha256': actual, 'ok': ok})
            if not ok: answer['errors'].append('Input hash mismatch: ' + key)
        answer['input_checks'] = input_checks
        original = image(paths['original']); shape = original.shape[:2]
        mask_path = run / 'masks' / role / relative.name
        comparison_path = run / '图像结果' / GROUPS[status] / role / relative.name
        for key, path in [('record_path', record_path), ('mask_path', mask_path), ('image_path', comparison_path)]:
            name = path.relative_to(run).as_posix()
            if record.get(key) != name: answer['errors'].append('Wrong saved path: ' + key)
            actual = {'path': name, 'size': path.stat().st_size, 'sha256': digest(path)}
            answer['files'].append(actual)
            if actual != manifest.get(name): answer['errors'].append('Output manifest mismatch: ' + name)
        mask, comparison = image(mask_path), image(comparison_path)
        if original.shape != (3000, 4096, 3): answer['errors'].append('Original dimensions changed')
        if mask.shape != shape or mask.dtype != np.uint8 or np.any((mask != 0) & (mask != 255)):
            raise ValueError('Mask not original-size uint8 0/255')
        if comparison.shape != (3000, 8192, 3) or comparison.dtype != np.uint8:
            raise ValueError('Comparison PNG dimensions/type wrong')
        original_delta = int(np.count_nonzero(np.any(comparison[:, :4096] != original, axis=2)))
        if original_delta: answer['errors'].append('Original left half changed')
        candidates = result.get('candidates', []); ids = [c['candidate_id'] for c in candidates]
        if len(ids) != len(set(ids)): answer['errors'].append('Duplicate candidate IDs')
        checks = []
        for candidate in candidates:
            errors, geometry, visible, detail = candidate_geometry(candidate, shape, result['diagnostics']['parameters'])
            checks.append({'candidate_id': candidate['candidate_id'], **detail, 'errors': errors})
            answer['errors'].extend(candidate['candidate_id'] + ': ' + error for error in errors)
        accepted = [c for c in candidates if c['accepted']]
        if status == 'DETECTED':
            if len(accepted) != 1 or not accepted[0].get('plausible', False):
                answer['errors'].append('DETECTED must have one accepted plausible candidate')
            if not np.any(mask): answer['errors'].append('DETECTED has an empty primary mask')
        elif accepted or np.any(mask): answer['errors'].append('Unconfirmed case has accepted mask/branch')
        if not answer['errors']:
            rebuilt = expected_mask(shape, candidates)
            mask_delta = int(np.count_nonzero(rebuilt != mask))
            if mask_delta: answer['errors'].append('Mask bridges an invisible gap or differs from accepted geometry')
            overlay, displayed = expected_overlay(original, status, mask, candidates, version)
            overlay_delta = int(np.count_nonzero(np.any(overlay != comparison[:, 4096:], axis=2)))
            if overlay_delta: answer['errors'].append('Overlay differs from status/visibility drawing contract')
            answer.update(mask_difference_pixels=mask_delta, overlay_difference_pixels=overlay_delta,
                          displayed_candidate_ids=displayed)
        for key in ('F3_used', 'bends_reconstructed', 'contact_point_claim', 'labels_used'):
            if result['diagnostics'].get(key) is not False: answer['errors'].append('Forbidden/unknown claim: ' + key)
        answer.update(original_half_changed_pixels=original_delta, mask_pixels=int(np.count_nonzero(mask)),
                      candidate_checks=checks, candidate_count=len(candidates))
    except Exception as error:
        answer['errors'].append(str(error)); answer['traceback'] = traceback.format_exc()
    answer['ok'] = not answer['errors']
    return answer


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', required=True)
    parser.add_argument('--workers', type=int, default=4); parser.add_argument('--suffix', default='')
    args = parser.parse_args(); started = time.time()
    if Path(args.run).name != args.run or args.run in ('.', '..'): raise ValueError('Invalid run name')
    if args.suffix and (Path(args.suffix).name != args.suffix or args.suffix in ('.', '..')): raise ValueError('Invalid suffix')
    run = io.HERE / args.run; suffix = '_' + args.suffix if args.suffix else ''
    target = run / ('independent_structure_audit' + suffix + '.json')
    if target.exists(): raise FileExistsError('Preserve previous audit: ' + str(target))
    start, summary = load(run / 'run_start.json'), load(run / 'summary.json')
    errors, sources, baselines, parents, source_boundary = audit_sources(run)
    rows = io.selection(start['mode'])
    if [r['output_file'] for r in rows] != start['input_ids']: errors.append('Fixed input list/order changed')
    if summary['planned'] != len(rows) or summary['completed'] != len(rows) or summary['failures']:
        errors.append('Prediction failures remain in fixed denominator')
    manifest_doc = load(run / 'output_manifest.json'); manifest = {f['path']: f for f in manifest_doc['files']}
    if len(manifest) != len(manifest_doc['files']): errors.append('Duplicate output manifest path')
    cv2.setNumThreads(1)
    # Populate the read-only parent hash inventory before launching threads.
    # Prediction workers each own a process; audit threads share this cache.
    if rows:
        for key, path in io.input_paths(rows[0]).items():
            if key != 'original': io.checked(path)
    with ThreadPoolExecutor(max_workers=max(1, min(4, args.workers))) as pool:
        cases = list(pool.map(lambda row: audit_case(row, run, manifest, start['version']), rows))
    actual = {item['path'] for case in cases for item in case['files']}
    if actual != set(manifest): errors.append('Output manifest not exactly three files per input')
    disk_files = set()
    for folder, pattern in [('图像结果', '*.png'), ('masks', '*.png'), ('记录', '*.json')]:
        root = run / folder
        if root.is_symlink() or root.is_junction(): raise ValueError('Unexpected linked output folder')
        disk_files.update(p.relative_to(run).as_posix() for p in root.rglob(pattern))
    if disk_files != set(manifest): errors.append('Unmanifested or missing formal output files')
    comparison_paths = {p.relative_to(run).as_posix() for p in (run / '图像结果').rglob('*.png')}
    expected_comparisons = {record['image_path'] for record in summary['records']}
    if comparison_paths != expected_comparisons: errors.append('Duplicated/missing classified comparison PNG')
    statuses = dict(Counter(c.get('status', 'AUDIT_FAILURE') for c in cases))
    if statuses != summary['statuses']: errors.append('Summary status counts differ')
    passed = not errors and all(c['ok'] for c in cases)
    report = {'run': args.run, 'passed': passed, 'checked_count': len(cases),
              'passed_count': sum(c['ok'] for c in cases), 'global_errors': errors,
              'camera_counts': dict(Counter(r['camera_role'] for r in rows)), 'statuses': statuses,
              'source_checks': sources, 'baseline_source_checks': baselines,
              'parent_manifest_checks': parents, 'source_boundary_check': source_boundary,
              'candidate_count': sum(c.get('candidate_count', 0) for c in cases),
              'nonempty_mask_count': sum(c.get('mask_pixels', 0) > 0 for c in cases), 'cases': cases,
              'scope': 'Fixed files, hashes, straight geometry and rendering contract only; not physical rope identity or accuracy.',
              'audit_source_sha256': digest(__file__), 'seconds': time.time() - started}
    io.dump(target, report)
    io.dump(run / ('independent_output_manifest' + suffix + '.json'),
            {'case_count': len(cases), 'files': [f for c in cases for f in c['files']],
             'audit_source_sha256': report['audit_source_sha256']})
    snap = run / ('audit_tool_snapshot' + suffix); snap.mkdir(exist_ok=True)
    shutil.copy2(__file__, snap / Path(__file__).name); shutil.copy2(io.HERE / 'a7_io.py', snap / 'a7_io.py')
    io.dump(run / ('audit_command_receipt' + suffix + '.json'), {'command': sys.argv,
            'python_executable': sys.executable, 'exit_code': 0 if passed else 1,
            'audit_sha256': digest(target), 'note': 'Caller records actual shell exit separately.'})
    print(json.dumps({k: report[k] for k in ['run', 'passed', 'checked_count', 'passed_count',
                'global_errors', 'statuses', 'candidate_count', 'nonempty_mask_count', 'seconds']}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == '__main__': sys.exit(main())
