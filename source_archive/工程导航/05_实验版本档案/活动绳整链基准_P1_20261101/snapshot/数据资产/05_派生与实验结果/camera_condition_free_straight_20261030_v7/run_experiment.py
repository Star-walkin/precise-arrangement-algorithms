"""A7离线运行工具：一图一份原图/叠加并排PNG，固定分母，保留失败。"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
import cv2
import numpy as np
import a7_io as io

# Declare dependencies here; files are checked only when a run actually starts.
CORE = ['a7_io.py', 'free_rope.py', 'free_seed_recovery.py', 'strand_signature.py',
        'straight_detector.py', 'run_experiment.py', 'protocol.json']
GROUPS = {'DETECTED': '01_已检出', 'AMBIGUOUS': '02_待确认', 'UNKNOWN': '03_未检出'}
GREEN = (65, 230, 80)
ORANGE = (0, 170, 255)


def dependencies():
    return [str(io.ASSETS / 'camera_condition_incoming_rope_20261026_v2/free_refinement.py')]


def source_hashes():
    paths = [io.HERE / name for name in CORE] + [Path(name) for name in dependencies()]
    return {str(path): io.sha(path) for path in paths}


def serializable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [serializable(item) for item in value]
    return value


def validate_result(result, shape):
    status = result['status']
    if status not in GROUPS: raise ValueError('Unknown detector status')
    mask = result['mask']
    if (not isinstance(mask, np.ndarray) or mask.dtype != np.uint8 or mask.shape != shape
            or np.any((mask != 0) & (mask != 255))):
        raise ValueError('mask must be original-size uint8 0/255')
    if (status == 'DETECTED') != bool(np.any(mask)):
        raise ValueError('Only DETECTED may have a nonempty primary mask')
    ids = set()
    for candidate in result.get('candidates', []):
        identifier = candidate['candidate_id']
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError('Candidate IDs must be unique nonempty strings')
        ids.add(identifier)
        visible = candidate['visible']
        if any(not isinstance(v, (bool, np.bool_)) for v in visible):
            raise ValueError('Candidate visibility must be Boolean')
        if not isinstance(candidate['accepted'], (bool, np.bool_)):
            raise ValueError('Candidate accepted flag must be Boolean')
        if 'plausible' in candidate and not isinstance(candidate['plausible'], (bool, np.bool_)):
            raise ValueError('Candidate plausible flag must be Boolean')
        if not np.isfinite(candidate['width_px']) or candidate['width_px'] <= 0:
            raise ValueError('Candidate pixel width must be finite and positive')
        if not np.isfinite(candidate['score']): raise ValueError('Nonfinite score')
        for name in ('left_xy', 'right_xy', 'center_xy'):
            points = candidate[name]
            if len(points) != len(visible): raise ValueError('Point/visibility lengths differ')
            for point, shown in zip(points, visible):
                # A hidden point can be null, but must not introduce NaN into JSON.
                if point is None and not shown: continue
                q = np.asarray(point, dtype=float)
                if q.shape != (2,) or not np.isfinite(q).all():
                    raise ValueError('Invalid candidate coordinates')
                if shown and not (0 <= q[0] < shape[1] and 0 <= q[1] < shape[0]):
                    raise ValueError('Visible candidate coordinate outside original image')
    saved = {key: serializable(value) for key, value in result.items() if key != 'mask'}
    # Also reject nonfinite numbers hidden in diagnostic fields.
    json.dumps(saved, allow_nan=False)
    return saved


def _visible_lines(image, points, visible, color, thickness=2):
    """Do not connect across an invisible or missing sample."""
    for i in range(1, len(points)):
        if visible[i - 1] and visible[i]:
            first = tuple(np.rint(points[i - 1]).astype(int))
            second = tuple(np.rint(points[i]).astype(int))
            cv2.line(image, first, second, color, thickness, cv2.LINE_8)


def render(original, result):
    overlay = original.copy()
    active = result['mask'] > 0
    overlay[active] = (overlay[active] * .82 + np.asarray(GREEN) * .18).astype(np.uint8)
    contours, _ = cv2.findContours(result['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, GREEN, 1)
    for candidate in result.get('candidates', []):
        # Rejected tool/background proposals stay in JSON, never clutter the
        # accepted rope image. Unknown cases retain the unaltered scene + label.
        if result['status'] == 'DETECTED':
            if not candidate['accepted']: continue
            color = GREEN
        elif result['status'] == 'AMBIGUOUS':
            if not candidate.get('plausible', False): continue
            color = ORANGE
        else:
            continue
        for side in ('left_xy', 'right_xy'):
            _visible_lines(overlay, candidate[side], candidate['visible'], color)
    # All lettering belongs to the right image; the complete left half stays exact.
    cv2.rectangle(overlay, (12, 12), (min(2200, original.shape[1] - 1), 112), (20, 20, 20), -1)
    cv2.putText(overlay, io.VERSION + '  ' + result['status'], (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1., (255, 255, 255), 2)
    cv2.putText(overlay, 'Green: detected straight body | Orange: unconfirmed candidate | Gaps are not bridged',
                (30, 92), cv2.FONT_HERSHEY_SIMPLEX, .75, (230, 230, 230), 1)
    return np.concatenate([original, overlay], axis=1)


def process(row, run_name):
    cv2.setNumThreads(1)
    start = time.time()
    run = io.HERE / run_name
    relative = Path(row['output_file'])
    role, stem = relative.parent.name, relative.stem
    try:
        # Imported only in prediction workers, never by an independent auditor.
        from straight_detector import detect
        data = io.inputs(row)
        if data['original'].shape != (3000, 4096, 3):
            raise ValueError('Fixed400 original image must be 4096x3000')
        result = detect(data)
        saved = validate_result(result, data['original'].shape[:2])
        image = render(data['original'], result)
        if not np.array_equal(image[:, :4096], data['original']):
            raise ValueError('Original half was modified')
        image_path = run / '图像结果' / GROUPS[result['status']] / role / relative.name
        mask_path = run / 'masks' / role / relative.name
        record_path = run / '记录' / role / (stem + '.json')
        io.write_image(image_path, image)
        io.write_image(mask_path, result['mask'])
        record = {'output_file': row['output_file'], 'camera_role': row['camera_role'],
                  'source_group': row['source_group'], 'source_time_s': row['source_time_s'],
                  'dataset_use_status': row['dataset_use_status'], 'status': result['status'],
                  'elapsed_seconds': time.time() - start,
                  'input_dependency_sha256': data['input_dependency_sha256'],
                  'image_path': image_path.relative_to(run).as_posix(),
                  'mask_path': mask_path.relative_to(run).as_posix(),
                  'record_path': record_path.relative_to(run).as_posix()}
        saved['mask'] = {'array_shape': list(result['mask'].shape), 'dtype': 'uint8',
                         'nonzero_pixels': int(np.count_nonzero(result['mask']))}
        io.dump(record_path, {'record': record, 'result': saved})
        files = [{'path': path.relative_to(run).as_posix(), 'size': path.stat().st_size,
                  'sha256': io.sha(path)} for path in (image_path, mask_path, record_path)]
        return {'record': record, 'files': files}
    except Exception as error:
        return {'failure': {'output_file': row['output_file'], 'error': str(error),
                            'traceback': traceback.format_exc()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['pilot', 'extra', 'all'], default='pilot')
    parser.add_argument('--run', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--freeze', default='')
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in ('.', '..'):
        raise ValueError('Direct child run name required')
    if not 1 <= args.workers <= 16: raise ValueError('Worker count outside 1..16')
    run = io.HERE / args.run
    if run.exists(): raise ValueError('New run directory required; old failures are retained')
    run.mkdir()
    started = time.time()
    try:
        sources = source_hashes()
        protocol = io.read_json(io.HERE / 'protocol.json')
        if args.freeze:
            freeze_path = Path(args.freeze)
            if not freeze_path.is_absolute(): freeze_path = io.HERE / freeze_path
            if io.read_json(freeze_path)['sources_sha256'] != sources:
                raise ValueError('Frozen sources differ from current implementation')
            shutil.copy2(freeze_path, run / 'freeze_snapshot.json')
        for filename, expected in protocol['source_parent_sha256'].items():
            if io.sha(filename) != expected: raise ValueError('Historical source changed: ' + filename)
        snapshot = run / 'source_snapshot'
        snapshot.mkdir()
        for filename in sources: shutil.copy2(filename, snapshot / Path(filename).name)
        rows = io.selection(args.mode)
        io.dump(run / 'run_start.json', {'version': io.VERSION, 'mode': args.mode,
            'command': sys.argv, 'python_executable': sys.executable,
            'sources_sha256': sources, 'input_ids': [row['output_file'] for row in rows],
            'input_count': len(rows), 'python': sys.version, 'opencv': cv2.__version__,
            'numpy': np.__version__, 'platform': platform.platform(), 'workers': args.workers})
    except Exception as error:
        io.dump(run / 'startup_failure.json', {'command': sys.argv, 'error': str(error),
                 'traceback': traceback.format_exc(), 'exit_code': 1})
        raise
    records, files, failures = [], [], []
    with (run / 'run.log').open('w', encoding='utf-8') as log, ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(process, row, args.run): row['output_file'] for row in rows}
        for future in as_completed(jobs):
            try: answer = future.result()
            except Exception as error:
                answer = {'failure': {'output_file': jobs[future], 'error': str(error), 'traceback': traceback.format_exc()}}
            if 'failure' in answer:
                failures.append(answer['failure'])
                line = answer['failure']['traceback']
            else:
                record = answer['record']
                records.append(record)
                files.extend(answer['files'])
                line = '%d/%d %s %s %.1fs' % (len(records), len(rows), record['output_file'], record['status'], record['elapsed_seconds'])
            log.write(line + '\n')
            log.flush()
            print(line, flush=True)
            io.dump(run / 'progress.json', {'completed': len(records), 'failed': len(failures), 'planned': len(rows)})
    records.sort(key=lambda row: row['output_file'])
    files.sort(key=lambda item: item['path'])
    sources_unchanged = all(io.sha(filename) == expected for filename, expected in sources.items())
    exit_code = 0 if len(records) == len(rows) and not failures and sources_unchanged else 1
    io.dump(run / 'output_manifest.json', {'version': io.VERSION, 'case_count': len(records), 'files': files})
    io.dump(run / 'summary.json', {'version': io.VERSION, 'mode': args.mode, 'planned': len(rows),
        'completed': len(records), 'failures': failures, 'statuses': dict(Counter(r['status'] for r in records)),
        'camera_counts': dict(Counter(r['camera_role'] for r in records)), 'records': records,
        'sources_unchanged': sources_unchanged, 'elapsed_seconds': time.time() - started, 'exit_code': exit_code})
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
