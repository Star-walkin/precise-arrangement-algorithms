"""Independent A4 file/array audit, with no detector or hardware imports.

Only whole_io's fixed development selection and pure file helpers are reused.
This verifies reproducibility and image invariants, not rope identity/accuracy.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sys
import time
import traceback
import cv2
import numpy as np
import whole_io as io

REQUIRED_CORE = ('whole_io.py', 'run_whole.py', 'free_rope.py',
                 'boundary_occlusion.py', 'whole_path.py', 'protocol.json', 'protocol_addendum_001.json')

def baseline_source_audit(protocol):
    """Only source bytes and frozen manifests, never replay old experiments."""
    errors=[]; checks=[]
    frozen=dict(protocol['predecessor_source_sha256'])
    f3=load(io.F3.parent/'freeze_all.json')
    frozen.update(f3['source_sha256'])
    old_manifest=io.ASSETS/'camera_condition_whole_incoming_20261028_v3/all_001/output_manifest.json'
    frozen[str(old_manifest)]=protocol['predecessor_output_manifest_sha256']
    for name,expected in frozen.items():
        actual=io.sha(name) if Path(name).is_file() else None
        checks.append({'path':name,'expected':expected,'actual':actual,'ok':actual==expected})
        if actual!=expected:errors.append('baseline_source_changed: '+name)
    return checks,errors

def finite_record(value,path='$'):
    """JSON can parse NaN despite not being valid strict JSON; reject it explicitly."""
    if isinstance(value,float) and not math.isfinite(value):return [path]
    if isinstance(value,dict):return [x for k,v in value.items() for x in finite_record(v,path+'.'+k)]
    if isinstance(value,list):return [x for i,v in enumerate(value) for x in finite_record(v,path+'['+str(i)+']')]
    return []

def candidate_audit(saved,shape):
    errors=[];h,w=shape;whole=saved['whole'];ids=[]
    for c in whole.get('candidates',[]):
        ids.append(c.get('candidate_id'))
        q=np.asarray(c.get('xy',[]),float)
        if q.ndim!=2 or q.shape[1]!=2 or len(q)<2:
            errors.append('invalid_candidate_xy_shape');continue
        if not np.isfinite(q).all() or np.any(q[:,0]<0) or np.any(q[:,0]>=w) or np.any(q[:,1]<0) or np.any(q[:,1]>=h):errors.append('candidate_outside_original_frame_or_nonfinite')
        widths=np.asarray(c.get('widths',[]),float)
        if len(widths)!=len(q) or not np.isfinite(widths).all() or np.any(widths<=0):errors.append('invalid_candidate_widths')
        states=c.get('supported',[])
        if len(states)!=len(q) or any(type(s) is not bool for s in states):errors.append('invalid_candidate_support_states')
        if c.get('length_px',0)<=0:errors.append('nonpositive_candidate_length')
    if len(ids)!=len(set(ids)):errors.append('duplicate_candidate_id')
    selected=whole.get('selected_candidate_id')
    if selected is not None and selected not in ids:errors.append('selected_candidate_missing_from_pool')
    if saved['record'].get('status')!=whole.get('status'):errors.append('record_vs_whole_status_mismatch')
    if whole.get('diagnostics',{}).get('same_material_identity_confirmed') is not False:errors.append('unexpected_identity_claim')
    bad=finite_record(saved)
    if bad:errors.append('nonfinite_saved_json: '+','.join(bad[:8]))
    return errors

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def normalized(path):
    return str(Path(path).resolve())

def expected_dependencies(row):
    rel = Path(row['output_file']); role = rel.parent.name; stem = rel.stem
    paths = [io.C4/'roi/surface_masks'/role/rel.name,
             io.A1/'records'/role/(stem+'.json'),
             io.A2/'records'/role/(stem+'.json'),
             io.C3/'boundaries/records'/role/(stem+'.json'),
             io.F3/'记录/cases'/role/(stem+'.json')]
    paths += [io.F3/'记录/masks'/kind/role/rel.name
              for kind in ('merged', 'inferred', 'exclusion')]
    return paths

def parent_inventory(protocol):
    inventory = {}; checks = []
    for name, expected in protocol['parent_manifest_sha256'].items():
        path = Path(name); actual = io.sha(path)
        checks.append({'path': str(path), 'expected': expected, 'actual': actual, 'ok': actual == expected})
        if actual != expected:
            raise ValueError('Frozen parent manifest changed: '+str(path))
        record = load(path)
        if 'cases' in record:
            for case in record['cases']:
                for entry in case['files']:
                    target = entry.get('absolute_path')
                    if target is None:
                        target = path.parent/entry.get('path', entry.get('relative_path'))
                    inventory[normalized(target)] = entry['sha256']
        else:
            for entry in record.get('files', record.get('file_manifest', [])):
                target = Path(entry['path'])
                if not target.is_absolute(): target = path.parent/target
                inventory[normalized(target)] = entry['sha256']
    return inventory, checks

def source_audit(run, start):
    errors = []; checks = []; frozen = start.get('sources_sha256', {})
    for name in REQUIRED_CORE:
        if str(io.HERE/name) not in frozen:
            errors.append('missing_core_source_hash: '+name)
    external = io.ASSETS/'camera_condition_incoming_rope_20261026_v2/free_refinement.py'
    if str(external) not in frozen:
        errors.append('missing_executed_external_dependency: '+str(external))
    names = [Path(name).name for name in frozen]
    if len(names) != len(set(names)):
        errors.append('flat_snapshot_basename_collision')
    for name, expected in frozen.items():
        original = Path(name); snapshot = run/'source_snapshot'/original.name
        current = io.sha(original) if original.is_file() else None
        copied = io.sha(snapshot) if snapshot.is_file() else None
        item = {'source': name, 'snapshot': str(snapshot), 'expected': expected,
                'current_sha256': current, 'snapshot_sha256': copied,
                'current_matches': current == expected, 'snapshot_matches': copied == expected}
        checks.append(item)
        if current != expected: errors.append('source_changed_or_missing: '+name)
        if copied != expected: errors.append('snapshot_changed_or_missing: '+str(snapshot))
    return checks, errors

def batch_audit(run, start, expected_names, merged_manifest):
    """If this is a sharded run, independently check the merge and scheduler freeze."""
    if not (run/'batch_start.json').is_file():
        return {'applicable': False}, []
    errors = []; detail = {'applicable': True, 'shards': [], 'freeze_sources': []}
    try:
        batch = load(run/'batch_start.json')
        freeze_path = run/'freeze_all_snapshot.json'
        if io.sha(freeze_path) != batch['freeze_sha256']:
            errors.append('batch_freeze_snapshot_sha_mismatch')
        freeze = load(freeze_path); frozen = freeze['sources_sha256']
        for name, expected in frozen.items():
            actual = io.sha(name) if Path(name).is_file() else None
            detail['freeze_sources'].append({'source': name, 'expected': expected, 'actual': actual, 'ok': actual == expected})
            if actual != expected: errors.append('batch_frozen_source_changed: '+name)
        batch_source = str(io.HERE/'run_batch.py')
        if batch_source not in frozen or io.sha(run/'run_batch_snapshot.py') != frozen.get(batch_source):
            errors.append('batch_scheduler_snapshot_not_frozen_or_mismatched')
        for name, expected in start['sources_sha256'].items():
            if frozen.get(name) != expected:
                errors.append('batch_vs_worker_source_freeze_mismatch: '+name)
        workers = int(batch['workers'])
        if workers < 1 or batch.get('input_count') != len(expected_names):
            raise ValueError('Invalid batch worker/count metadata')
        all_names = []; all_files = {}
        for shard in range(workers):
            context = run/'分片记录'/f'part_{shard:02d}'
            child_start = load(context/'run_start.json')
            child_summary = load(context/'summary.json')
            child_manifest = load(context/'output_manifest.json')
            target = expected_names[shard::workers]
            child_errors = []
            if child_start.get('inputs') != target: child_errors.append('not_exact_modulo_shard')
            if child_start.get('sources_sha256') != start['sources_sha256']:
                child_errors.append('source_freeze_differs_from_root')
            if child_summary.get('completed') != len(target) or child_summary.get('failures') or child_summary.get('exit_code') != 0:
                child_errors.append('shard_summary_incomplete_or_failed')
            names = [v['output_file'] for v in child_summary.get('records', [])]
            if names != target: child_errors.append('shard_record_order_or_membership_mismatch')
            all_names.extend(names)
            if child_manifest.get('case_count') != len(target): child_errors.append('shard_manifest_case_count_mismatch')
            _, snapshot_errors = source_audit(context, child_start)
            child_errors.extend(snapshot_errors)
            for entry in child_manifest['files']:
                key = Path(entry['path']).as_posix()
                if key in all_files: child_errors.append('duplicate_file_across_shards: '+key)
                all_files[key] = entry
            detail['shards'].append({'index': shard, 'input_count': len(target), 'errors': child_errors, 'ok': not child_errors})
            errors.extend('shard_'+str(shard)+': '+value for value in child_errors)
        if len(all_names) != len(set(all_names)) or set(all_names) != set(expected_names):
            errors.append('shard_inputs_not_disjoint_exhaustive_cover')
        if set(all_files) != set(merged_manifest): errors.append('merged_manifest_not_exact_shard_union')
        for key in set(all_files) & set(merged_manifest):
            if any(all_files[key].get(field) != merged_manifest[key].get(field) for field in ('sha256', 'size')):
                errors.append('merged_entry_differs_from_shard: '+key)
    except Exception as exc:
        errors.append(str(exc)); detail['traceback'] = traceback.format_exc()
    detail['ok'] = not errors
    return detail, errors

def audit_case(row, run, inventory, manifest, frozen):
    rel = Path(row['output_file']); role = rel.parent.name; stem = rel.stem
    files = [run/'01_整根候选掩膜'/role/rel.name,
             run/'02_原图叠加'/role/rel.name,
             run/'03_实测支持掩膜'/role/rel.name,
             run/'04_接续待确认'/role/rel.name,
             run/'记录'/role/(stem+'.json')]
    result = {'output_file': row['output_file'], 'camera_role': row['camera_role'],
              'errors': [], 'files': [], 'input_checks': []}
    fail = result['errors'].append
    try:
        for path in files:
            relative = path.relative_to(run).as_posix()
            if not path.is_file():
                raise ValueError('Missing result file: '+str(path))
            digest = io.sha(path); entry = manifest.get(relative)
            result['files'].append({'path': relative, 'size': path.stat().st_size, 'sha256': digest})
            if entry is None: fail('not_in_output_manifest: '+relative)
            elif entry.get('sha256') != digest or entry.get('size') != path.stat().st_size:
                fail('output_manifest_mismatch: '+relative)
        saved = load(files[4]); rec = saved['record']
        if rec.get('output_file') != row['output_file']: fail('case_input_id_mismatch')
        if rec.get('camera_role') != row['camera_role']: fail('case_camera_role_mismatch')
        for key in ('source_group', 'source_time_s', 'dataset_use_status'):
            if str(rec.get(key)) != str(row.get(key)): fail('case_metadata_mismatch: '+key)
        original_path = io.DATA/rel
        actual_original_hash = io.sha(original_path)
        if actual_original_hash != row['output_sha256']: fail('original_dataset_sha_mismatch')
        expected = {normalized(path): inventory.get(normalized(path)) for path in expected_dependencies(row)}
        expected[normalized(original_path)] = row['output_sha256']
        saved_dependencies = {normalized(k): v for k, v in rec.get('input_dependency_sha256', {}).items()}
        if set(saved_dependencies) != set(expected): fail('dependency_path_set_mismatch')
        for path, target in expected.items():
            actual = actual_original_hash if path == normalized(original_path) else io.sha(path)
            okay = target is not None and actual == target and saved_dependencies.get(path) == target
            result['input_checks'].append({'path': path, 'expected_sha256': target, 'actual_sha256': actual, 'ok': okay})
            if not okay: fail('input_dependency_sha_mismatch: '+path)
        for path in saved.get('free', {}).get('source_dependencies', []):
            if str(Path(path)) not in frozen: fail('case_external_source_not_frozen: '+path)
        original = io.read_image(original_path, cv2.IMREAD_COLOR)
        whole = io.read_image(files[0], cv2.IMREAD_UNCHANGED)
        support = io.read_image(files[2], cv2.IMREAD_UNCHANGED)
        review = io.read_image(files[3], cv2.IMREAD_UNCHANGED)
        overlay = io.read_image(files[1], cv2.IMREAD_UNCHANGED)
        shape = original.shape[:2]
        result['errors'].extend(candidate_audit(saved,shape))
        result['candidate_count']=len(saved['whole'].get('candidates',[]))
        result['shape'] = list(shape)
        for name, mask in [('whole', whole), ('support', support), ('review', review)]:
            if mask.shape != shape or mask.dtype != np.uint8:
                fail(name+'_shape_or_dtype_mismatch')
            elif np.any((mask != 0) & (mask != 255)):
                fail(name+'_not_binary_0_255')
        if whole.shape == shape and support.shape == shape:
            outside_support = int(np.count_nonzero((support != 0) & (whole == 0)))
            result['whole_pixels'] = int(np.count_nonzero(whole))
            result['supported_pixels'] = int(np.count_nonzero(support))
            result['supported_outside_whole_pixels'] = outside_support
            if outside_support: fail('supported_mask_not_subset_of_whole')
        if whole.shape == shape and review.shape == shape:
            outside_review = int(np.count_nonzero((review != 0) & (whole == 0)))
            result['review_pixels'] = int(np.count_nonzero(review))
            result['review_outside_whole_pixels'] = outside_review
            if outside_review: fail('review_mask_not_subset_of_whole')
            # Review describes route uncertainty; it may overlap local image support.
            if support.shape == shape:
                result['review_with_local_support_pixels'] = int(np.count_nonzero((review != 0) & (support != 0)))
        if overlay.shape != original.shape or overlay.dtype != np.uint8:
            fail('overlay_shape_or_dtype_mismatch')
        elif whole.shape == shape:
            # Exactly the declared drawing region; no large generic border allowance.
            allowed = (whole != 0).astype(np.uint8)
            contours, _ = cv2.findContours(whole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(allowed, contours, -1, 1, 1)
            cv2.rectangle(allowed, (12, 12), (1540, 112), 1, -1)
            changed = np.any(overlay != original, axis=2)
            outside = int(np.count_nonzero(changed & (allowed == 0)))
            result['overlay_changed_outside_declared_drawing_pixels'] = outside
            if outside: fail('overlay_modified_original_outside_drawing_region')
        result['status'] = rec.get('status')
    except Exception as exc:
        fail(str(exc)); result['traceback'] = traceback.format_exc()
    result['ok'] = not result['errors']
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--mode', choices=('all', 'pilot'), default='all')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in ('.', '..'):
        raise ValueError('Use a run directory name within this experiment')
    run = io.HERE/args.run
    if (run/'structure_audit.json').exists() or (run/'independent_output_manifest.json').exists():
        raise ValueError('Audit receipt exists; preserve old audit, use a new filename/version for repairs')
    started = time.time(); cv2.setNumThreads(1)
    errors = []; rows = io.selection(args.mode)
    all_rows = io.selection('all')
    if len(all_rows) != 400 or Counter(r['camera_role'] for r in all_rows) != {'left': 200, 'right': 200}:
        errors.append('fixed_dataset_not_400_or_not_200_each')
    start = load(run/'run_start.json'); summary = load(run/'summary.json')
    protocol = load(run/'source_snapshot/protocol.json')
    expected_names = [r['output_file'] for r in rows]
    if start.get('inputs') != expected_names: errors.append('run_inputs_do_not_equal_fixed_ordered_selection')
    if args.mode == 'pilot' and set(expected_names) != set(protocol['pilot_inputs']):
        errors.append('pilot_inputs_differ_from_frozen_protocol')
    if summary.get('completed') != len(rows) or summary.get('failures') or summary.get('exit_code') != 0:
        errors.append('run_summary_incomplete_or_failed')
    source_checks, source_errors = source_audit(run, start); errors.extend(source_errors)
    baseline_checks,baseline_errors=baseline_source_audit(protocol);errors.extend(baseline_errors)
    inventory, parent_checks = parent_inventory(protocol)
    production_manifest = load(run/'output_manifest.json')
    entries = production_manifest['files']
    manifest = {Path(v['path']).as_posix(): v for v in entries}
    if len(manifest) != len(entries): errors.append('duplicate_output_manifest_paths')
    if production_manifest.get('case_count') != len(rows): errors.append('manifest_case_count_mismatch')
    expected_outputs = set()
    for row in rows:
        rel = Path(row['output_file']); role = rel.parent.name
        expected_outputs.update((Path(folder)/role/rel.name).as_posix()
                                for folder in ('01_整根候选掩膜', '02_原图叠加', '03_实测支持掩膜', '04_接续待确认'))
        expected_outputs.add((Path('记录')/role/(rel.stem+'.json')).as_posix())
    if set(manifest) != expected_outputs: errors.append('output_manifest_path_set_mismatch')
    batch_checks, batch_errors = batch_audit(run, start, expected_names, manifest)
    errors.extend(batch_errors)
    frozen = start['sources_sha256']
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 4))) as pool:
        results = list(pool.map(lambda row: audit_case(row, run, inventory, manifest, frozen), rows))
    passed = sum(item['ok'] for item in results)
    actual_statuses=dict(Counter(item.get('status') for item in results))
    if actual_statuses!=summary.get('statuses'):errors.append('summary_status_count_differs_from_saved_cases')
    saved_names=[r['output_file'] for r in summary.get('records',[])]
    if len(saved_names)!=len(set(saved_names)) or set(saved_names)!=set(expected_names):errors.append('summary_records_not_exact_expected_denominator')
    report = {'audit_version': 'A4-structure-1', 'run': args.run,
              'scope': 'Independent file, frozen-source and image-array audit; no detector execution or visual accuracy scoring.',
              'fixed_dataset_count': len(all_rows), 'fixed_dataset_camera_counts': dict(Counter(r['camera_role'] for r in all_rows)),
              'expected_case_count': len(rows), 'checked_case_count': len(results), 'passed_case_count': passed,
              'run_camera_counts': dict(Counter(r['camera_role'] for r in rows)),
              'global_errors': errors, 'source_checks': source_checks, 'parent_manifest_checks': parent_checks,
              'baseline_source_checks':baseline_checks,'statuses':actual_statuses,
              'nonempty_mask_cases':sum(item.get('whole_pixels',0)>0 for item in results),
              'empty_mask_cases':sum(item.get('whole_pixels',0)==0 for item in results),
              'batch_checks': batch_checks,
              'cases': results, 'elapsed_seconds': time.time()-started,
              'audit_script_sha256': io.sha(Path(__file__)),
              'passed': passed == len(rows) and not errors,
              'accuracy_contact_or_same_material_identity_verified': False}
    io.dump(run/'structure_audit.json', report)
    io.dump(run/'independent_output_manifest.json', {
        'audit_version': report['audit_version'], 'case_count': len(results),
        'files': [entry for case in results for entry in case['files']],
        'audit_script_sha256': report['audit_script_sha256']})
    print(json.dumps({k: report[k] for k in ('run', 'passed', 'passed_case_count', 'expected_case_count', 'global_errors', 'elapsed_seconds')}, ensure_ascii=False))
    return 0 if report['passed'] else 1

if __name__ == '__main__':
    sys.exit(main())
