"""F1保存结果的独立集合/像素审计，不导入fusion或重新运行任何检测器。

只复用文件入口fusion_io。所有归属、合并、黄点筛选和显示像素均从
旧版冻结输入独立重算；结构一致不等于原边界正确或同根身份成立。
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import traceback

import cv2
import numpy as np

from fusion_io import HERE, inputs, selection, read_json, read_image, sha, dump


MASK_NAMES = ('surface', 'bright_owner', 'dark_owner', 'uncovered',
              'c3_boundaries', 'd4_boundaries', 'merged', 'inferred',
              'accepted_yellow')
SOURCE_NAMES = ('fusion_io.py', 'fusion.py', 'run_fusion.py', 'protocol.json')


def issue(errors, name, **detail):
    errors.append({'check': name, **detail})


def count(mask):
    return int(np.count_nonzero(mask))


def file_record(path, run):
    return {'path': path.relative_to(run).as_posix(), 'size': path.stat().st_size,
            'sha256': sha(path)}


def accepted_centers(data, bright_owner, dark_owner):
    """独立按来源遍历原JSON；先保留接受点，再做整数坐标去重及归属筛选。"""
    result = []
    height, width = bright_owner.shape
    source_lists = {'C3': [], 'D4': []}
    for group in data['c3_record']['fit']['groups']:
        if group.get('curve_id'):
            source_lists['C3'].extend(group['observed_points_xy'])

    track_lookup = {track['track_id']: track['points']
                    for track in data['d4_record']['new_tracks']}
    point_ids = set()
    for curve in data['d4_record']['fit']['curves']:
        track = track_lookup[curve['track_id']]
        for index in curve['source_point_indices']:
            if not isinstance(index, int) or not 0 <= index < len(track):
                raise ValueError('invalid D4 accepted source point index')
            point = track[index]
            if point['point_id'] in point_ids:
                continue
            point_ids.add(point['point_id'])
            source_lists['D4'].append((point['x'], point['y']))

    yellow = np.zeros(bright_owner.shape, np.uint8)
    for origin, owner in [('C3', bright_owner), ('D4', dark_owner)]:
        raster = np.zeros(bright_owner.shape, np.uint8)
        rounded_seen = set()
        for coordinates in source_lists[origin]:
            x, y = map(float, coordinates)
            if not np.isfinite(x) or not np.isfinite(y):
                raise ValueError('nonfinite accepted point')
            column, row = int(round(x)), int(round(y))
            if not (0 <= column < width and 0 <= row < height):
                continue
            if not owner[row, column] or (column, row) in rounded_seen:
                continue
            rounded_seen.add((column, row))
            result.append({'source': origin, 'x': column, 'y': row})
            cv2.circle(raster, (column, row), 1, 255, -1, cv2.LINE_8)
        raster[~owner] = 0
        yellow |= raster
    return result, yellow > 0


def audit_case(task):
    row, run_name = task
    run = Path(run_name)
    cv2.setNumThreads(1)
    errors, files = [], []
    rel = Path(row['output_file'])
    role, name = rel.parent.name, rel.name
    report = {'output_file': row['output_file'], 'errors': errors, 'files': files}
    try:
        # inputs核验旧输出清单hash，只负责IO，不参与任何融合判断。
        data, dependencies = inputs(row)
        surface = data['surface'] != 0
        bright = np.logical_and(surface, data['bright'] != 0)
        dark = np.logical_and.reduce((surface, data['dark'] != 0, ~bright))
        c3 = np.logical_and(bright, data['c3_smooth'] != 0)
        d4 = np.logical_and(dark, data['d4_smooth'] != 0)
        union = np.logical_or(c3, d4)
        inferred = np.logical_or(np.logical_and(bright, data['c3_inferred'] != 0),
                                 np.logical_and(dark, data['d4_inferred'] != 0))
        if np.any(inferred & ~union):
            issue(errors, 'parent_inference_outside_parent_modeled')
        centers, yellow = accepted_centers(data, bright, dark)
        expected = {'surface': surface, 'bright_owner': bright, 'dark_owner': dark,
                    'uncovered': surface & ~(bright | dark), 'c3_boundaries': c3,
                    'd4_boundaries': d4, 'merged': union, 'inferred': inferred,
                    'accepted_yellow': yellow}
        pixel_counts = {key: count(value) for key, value in expected.items()}
        for key in MASK_NAMES:
            path = run / '记录/masks' / key / role / name
            actual = read_image(path, cv2.IMREAD_UNCHANGED)
            if actual.shape != surface.shape or actual.dtype != np.uint8:
                issue(errors, 'mask_shape_or_dtype', mask=key, shape=list(actual.shape))
            elif np.any((actual != 0) & (actual != 255)):
                issue(errors, 'nonbinary_output_mask', mask=key)
            elif not np.array_equal(actual != 0, expected[key]):
                issue(errors, 'mask_set_mismatch', mask=key,
                      different_pixels=count((actual != 0) ^ expected[key]),
                      added_pixels=count((actual != 0) & ~expected[key]),
                      lost_pixels=count((actual == 0) & expected[key]))
            files.append(file_record(path, run))
        # 上述精确等式比“落在ROI内”更强：不能新增接线，也不能偷偷删原线。
        record_path = run / '记录/cases' / role / (rel.stem + '.json')
        saved = read_json(record_path)
        record = saved['record']
        if record.get('status') != 'completed':
            issue(errors, 'case_status', value=record.get('status'))
        if saved.get('display_point_centers') != centers:
            issue(errors, 'accepted_centers_or_order_mismatch')
        if record.get('input_dependency_sha256') != dependencies:
            issue(errors, 'input_dependency_hash_map_mismatch')
        if record.get('original_sha256') != row['output_sha256']:
            issue(errors, 'original_hash_mismatch')
        for key in ('output_file', 'camera_role', 'source_group', 'source_file',
                    'source_time_s', 'dataset_use_status'):
            if record.get(key) != row[key]:
                issue(errors, 'case_source_metadata_mismatch', field=key)
        if record.get('pixels') != pixel_counts:
            issue(errors, 'reported_pixel_counts_mismatch')
        scalar_expectations = {
            'input_roi_overlap_pixels': count(surface & (data['bright'] > 0) & (data['dark'] > 0)),
            'c3_boundary_clipped_pixels': count((data['c3_smooth'] > 0) & ~bright),
            'd4_boundary_clipped_pixels': count((data['d4_smooth'] > 0) & ~dark),
            'accepted_point_centers': len(centers), 'new_boundary_pixels_generated': 0,
            'same_rope_identity_between_sources': None}
        for key, value in scalar_expectations.items():
            if key not in record or record[key] != value:
                issue(errors, 'reported_metadata_mismatch', field=key)
        source_ids = {origin: [origin + ':' + str(c['curve_id'])
                              for c in data[key]['fit']['curves']]
                      for origin, key in [('C3', 'c3_record'), ('D4', 'd4_record')]}
        if record.get('source_curve_ids') != source_ids:
            issue(errors, 'source_curve_id_namespaces_mismatch')
        files.append(file_record(record_path, run))

        image_path = run / '图像结果' / role / name
        actual_image = read_image(image_path, cv2.IMREAD_UNCHANGED)
        if actual_image.shape != (3000, 4096, 3) or actual_image.dtype != np.uint8:
            issue(errors, 'overlay_shape_or_dtype')
        else:
            rendered = data['original'].copy()
            rendered[dark] = data['enhanced'][dark]
            rendered[~surface] = 0
            supported = union & ~inferred
            dashed = inferred.copy()
            dashed[:, ((np.arange(surface.shape[1]) // 6) % 2) != 0] = False
            markers = np.zeros(surface.shape, np.uint8)
            for point in centers[::2]:
                cv2.drawMarker(markers, (point['x'], point['y']), 255,
                               cv2.MARKER_CROSS, 5, 1, cv2.LINE_8)
            paint = (supported | dashed | (markers > 0)) & surface
            background_mismatch = count(np.any(actual_image != rendered, axis=2) & ~paint)
            if background_mismatch:
                issue(errors, 'nondrawing_background_mismatch', pixels=background_mismatch)
            if np.any(actual_image[~surface]):
                issue(errors, 'overlay_outside_surface_nonzero')
            rendered[supported] = (230, 190, 35)
            rendered[dashed] = (210, 65, 210)
            rendered[markers > 0] = (0, 230, 255)
            rendered[~surface] = 0
            different = count(np.any(actual_image != rendered, axis=2))
            if different:
                issue(errors, 'exact_overlay_pixels_mismatch', pixels=different)
        files.append(file_record(image_path, run))
        report.update({'pixels': pixel_counts, 'accepted_point_centers': len(centers),
                       'new_boundary_pixels': 0, 'exact_recomputation': True})
    except Exception as exc:
        issue(errors, 'audit_exception', exception=repr(exc), traceback=traceback.format_exc())
    report['status'] = 'failed' if errors else 'passed'
    return report


def audit_run(run, workers=4):
    context = read_json(run / 'run_context.json')
    protocol = read_json(HERE / 'protocol.json')
    mode = context.get('mode')
    if mode not in ('pilot', 'all'):
        raise ValueError('unknown run mode')
    rows = selection(mode)
    expected_count = 12 if mode == 'pilot' else 400
    expected_names = [row['output_file'] for row in rows]
    global_errors = []
    if len(rows) != expected_count or len(set(expected_names)) != expected_count:
        issue(global_errors, 'selection_denominator', actual=len(rows), expected=expected_count)
    if context.get('selection') != expected_names:
        issue(global_errors, 'run_selection_or_order_mismatch')
    actual_sources = {str(HERE / name): sha(HERE / name) for name in SOURCE_NAMES}
    if context.get('source_sha256') != actual_sources:
        issue(global_errors, 'runtime_source_hash_mismatch')
    snapshots = context.get('source_snapshot_paths', {})
    if set(snapshots) != set(actual_sources):
        issue(global_errors, 'source_snapshot_key_mismatch')
    for source, digest in actual_sources.items():
        if source not in snapshots:
            continue
        path = (run / snapshots[source]).resolve()
        if not path.is_relative_to(run / 'source_snapshot') or not path.is_file() or sha(path) != digest:
            issue(global_errors, 'source_snapshot_changed', source=source)
    for group in ('baseline_source_sha256', 'parent_manifest_sha256'):
        for path, digest in protocol[group].items():
            if sha(path) != digest:
                issue(global_errors, 'frozen_upstream_hash_changed', source=path)
    if mode == 'all' and read_json(HERE / 'freeze_all.json').get('source_sha256') != actual_sources:
        issue(global_errors, 'final_freeze_mismatch')

    summary = read_json(run / 'summary.json')
    for key, value in {'planned': expected_count, 'completed': expected_count,
                       'failed': 0, 'exit_code': 0, 'source_unchanged': True}.items():
        if summary.get(key) != value:
            issue(global_errors, 'summary_not_completed', field=key, value=summary.get(key))
    cameras = {camera: sum(row['camera_role'] == camera for row in rows) for camera in ('left', 'right')}
    if summary.get('by_camera') != cameras:
        issue(global_errors, 'camera_denominator_mismatch')
    # 文件名集合检查不能把缺失、重复或额外输出藏在总数量里。
    expected_png = {str(Path(row['output_file']).relative_to('images')).replace('\\', '/') for row in rows}
    for directory in [run / '图像结果'] + [run / '记录/masks' / key for key in MASK_NAMES]:
        observed = {p.relative_to(directory).as_posix() for p in directory.glob('*/*.png')}
        if observed != expected_png:
            issue(global_errors, 'output_file_set_mismatch', directory=str(directory),
                  missing=sorted(expected_png-observed), extra=sorted(observed-expected_png))
    cases_dir = run / '记录/cases'
    expected_json = {str(Path(p).with_suffix('.json')).replace('\\', '/') for p in expected_png}
    observed_json = {p.relative_to(cases_dir).as_posix() for p in cases_dir.glob('*/*.json')}
    if expected_json != observed_json:
        issue(global_errors, 'case_file_set_mismatch', missing=sorted(expected_json-observed_json),
              extra=sorted(observed_json-expected_json))

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(audit_case, [(r, str(run)) for r in rows]), 1):
            results.append(result)
            if index % 25 == 0 or index == len(rows):
                print(f'audited {index}/{len(rows)}; failed={sum(r["status"]=="failed" for r in results)}', flush=True)
    totals = {key: sum(r.get('pixels', {}).get(key, 0) for r in results) for key in MASK_NAMES}
    if summary.get('pixels_total') != totals:
        issue(global_errors, 'summary_pixel_total_mismatch')
    if actual_sources != {str(HERE / name): sha(HERE / name) for name in SOURCE_NAMES}:
        issue(global_errors, 'source_changed_during_audit')
    failed = sum(r['status'] == 'failed' for r in results)
    report = {'audit_name': 'F1 independent saved-result pixel audit', 'version': 'RB-F1.0.0',
              'run': str(run), 'mode': mode, 'planned': expected_count,
              'passed': len(results)-failed, 'failed': failed, 'global_errors': global_errors,
              'finished_utc': datetime.now(timezone.utc).isoformat(),
              'auditor_sha256': sha(Path(__file__)), 'source_sha256': actual_sources,
              'by_camera': cameras, 'cases': results,
              'exit_code': int(bool(failed or global_errors)),
              'accuracy': None, 'meaning': 'Exact source-mask/point/render equivalence only; not boundary accuracy, material identity, or newly filled coverage.'}
    inventory = {'version': 'RB-F1.0.0', 'run': str(run), 'planned': expected_count,
                 'case_count': len(results), 'file_count': sum(len(r['files']) for r in results),
                 'cases': [{'output_file': r['output_file'], 'status': r['status'], 'files': r['files']} for r in results],
                 'scope': 'Per-image saved fusion predictions and case records; excludes this inventory, audit, QA, and later documentation.',
                 'run_files': [file_record(run / name, run) for name in ('run_context.json', 'summary.json', 'run_log.jsonl')],
                 'auditor_sha256': report['auditor_sha256']}
    return report, inventory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3, 4), default=4)
    args = parser.parse_args()
    run = (args.run if args.run.is_absolute() else HERE / args.run).resolve()
    if run.parent != HERE.resolve() or not run.is_dir():
        raise ValueError('run must be an existing direct child of F1')
    if (run / 'audit.json').exists() or (run / 'output_manifest.json').exists():
        raise FileExistsError('audit or output inventory already exists; never overwrite evidence')
    report, inventory = audit_run(run, args.workers)
    dump(run / 'audit.json', report)
    dump(run / 'output_manifest.json', inventory)
    print({key: report[key] for key in ('planned', 'passed', 'failed', 'global_errors', 'exit_code')}, flush=True)
    return report['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
