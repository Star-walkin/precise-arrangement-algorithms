"""Byte-only independent verification of the A5-declared historical source chain.

No algorithm imports, directory recursion, image decode, label reads or hardware.
Only exact paths declared by A5 protocol and its hash-locked F3 source freeze.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import stat


HERE = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def reparse_component(path):
    # Exact allow-list paths are checked component by component before reading.
    for component in [path, *path.parents]:
        if component.exists():
            s = component.lstat()
            if getattr(s, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                return str(component)
    return None


def family(path):
    parts = set(path.parts)
    names = {
        'camera_condition_whole_incoming_20261028_v4': 'A4',
        'camera_condition_full_surface_joint_20261028_v3': 'F3',
        'camera_condition_grouped_conic_20261026_v6': 'C3',
        'camera_condition_dark_width_20261027_v4': 'D4',
    }
    return next((v for k, v in names.items() if k in parts), 'upstream_lineage')


def check(path, expected, authority):
    path = Path(path)
    r = {'path': str(path), 'family': family(path), 'expected_sha256': expected,
         'authority': authority, 'status': 'not_checked'}
    if not path.is_absolute() or path.suffix not in ('.py', '.json'):
        r['status'] = 'refused_non_source_or_nonabsolute_path'
        return r
    try:
        link = reparse_component(path)
        if link:
            r.update(status='refused_reparse_path', reparse_component=link)
        elif not path.is_file():
            r['status'] = 'missing'
        else:
            actual = sha(path)
            r.update(actual_sha256=actual, byte_count=path.stat().st_size,
                     status='match' if actual == expected else 'mismatch')
    except OSError as error:
        r.update(status='read_error', error=str(error))
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', default='parent_sources_independent_check_001.json')
    args = parser.parse_args()
    out = HERE / args.receipt
    if out.parent.resolve() != HERE or out.exists():
        raise ValueError('New receipt filename in this experiment only')
    protocol_path = HERE / 'protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    freeze = HERE.parent / 'camera_condition_full_surface_joint_20261028_v3/freeze_all.json'
    anchor = check(freeze, protocol['F3_sources_freeze_sha256'], 'A5 protocol F3_sources_freeze_sha256')
    rows = [anchor]
    allowed = {}
    if anchor['status'] == 'match':
        f3 = json.loads(freeze.read_text(encoding='utf-8'))
        for path, expected in f3['source_sha256'].items():
            allowed[path] = (expected, 'hash-locked F3 freeze_all.json source_sha256')
    for path, expected in protocol['source_parent_sha256'].items():
        if path in allowed and allowed[path][0] != expected:
            raise ValueError('Conflicting frozen authorities: ' + path)
        allowed[path] = (expected, 'A5 protocol source_parent_sha256')
    for path, (expected, authority) in allowed.items():
        rows.append(check(path, expected, authority))
    families = {}
    for row in rows:
        bucket = families.setdefault(row['family'], {'files': 0, 'matching': 0})
        bucket['files'] += 1
        bucket['matching'] += int(row['status'] == 'match')
    passed = all(row['status'] == 'match' for row in rows)
    result = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Explicit historical source and protocol byte verification; includes a frozen upstream JSON output manifest as declared in the F3 source list. No prediction image hashes are recomputed.',
        'command': 'vision_prototype/.venv/Scripts/python.exe -B check_parent_sources.py --receipt ' + args.receipt,
        'script_sha256': sha(Path(__file__)),
        'A5_protocol_sha256_at_check': sha(protocol_path),
        'unique_explicit_file_count': len(rows), 'families': families,
        'all_match': passed, 'exit_code': 0 if passed else 1,
        'algorithm_imports': False, 'image_decoding': False, 'labels_read': False,
        'equipment_connected': False, 'reserved_data_read': False,
        'historical_sources_modified': False, 'reparse_paths_followed': False,
        'interpretation': 'Matching bytes establish preservation of declared historical sources, not visual accuracy or whole-pipeline validity.',
        'files': rows,
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'receipt': str(out), 'all_match': passed, 'files': len(rows), 'families': families}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
