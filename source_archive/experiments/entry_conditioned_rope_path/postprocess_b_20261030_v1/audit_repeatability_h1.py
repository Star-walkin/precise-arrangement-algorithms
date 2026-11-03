"""Compare final fixed-pilot/extra H1 images to the all run, byte for byte.

Only persisted inputs and outputs are read. The detector/runner are not imported
or rerun. Identical outputs demonstrate repeatability, not correct rope identity.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import shutil
import sys
import traceback

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='all_001')
    ap.add_argument('--output', default='independent_repeatability_audit.json')
    args = ap.parse_args()
    run = (HERE / args.run).resolve()
    require(run.is_relative_to(HERE.resolve()), 'Run must remain under this experiment')
    dest = (run / args.output).resolve()
    require(dest.is_relative_to(run) and not dest.exists(), 'Do not escape run or overwrite an audit')
    comparisons = []; failures = []; cases = []; sources = []
    try:
        freeze = read(HERE / 'freeze_final.json')
        start = read(run / 'run_start.json'); summary = read(run / 'summary.json')
        require(start['sources'] == freeze['sources'], 'All run was not executed from final frozen source hashes')
        require(summary['completed'] == summary['planned'] == 292 and summary['entries'] == 299,
                'All run denominator is not292 images/299 conditions')
        require(summary['exit_code'] == 0 and not summary['failures'] and summary['source_unchanged'],
                'All run did not complete cleanly')
        for path, expected in freeze['sources'].items():
            actual = sha(path); require(actual == expected, 'Final algorithm source changed')
            sources.append({'path': path, 'sha256': actual})
        ids = set()
        for name, expected_count in ((freeze['pilot'], 12), ('extra_001', 8)):
            old = HERE / name
            old_start = read(old / 'run_start.json'); old_summary = read(old / 'summary.json')
            require(old_start['sources'] == freeze['sources'], 'Comparison run used different algorithm source')
            require(old_summary['completed'] == old_summary['planned'] == expected_count and old_summary['exit_code'] == 0,
                    'Comparison run did not complete its expected denominator')
            old_audit = read(old / 'independent_audit_001.json')
            require(old_audit['passed'] is True and old_audit['exit_code'] == 0, 'Comparison run has not passed independent structure audit')
            for row in old_summary['rows']:
                ident = row['id']; require(ident not in ids, 'Duplicate pilot/extra image')
                ids.add(ident)
                a = read(old / '记录' / (ident + '.json'))
                b = read(run / '记录' / (ident + '.json'))
                old_files = {Path(v).as_posix() for v in a['files']}
                new_files = {Path(v).as_posix() for v in b['files']}
                require(old_files == new_files, 'Formal image file lists differ')
                require(a['entry_count'] == b['entry_count'], 'Entry denominator changed between subset and all run')
                case_files = []
                for relative in sorted(old_files):
                    rel = Path(relative)
                    require(not rel.is_absolute() and '..' not in rel.parts and rel.suffix.lower() in ('.jpg', '.png'),
                            'Unexpected formal output path')
                    first, second = sha(old / rel), sha(run / rel)
                    record = {'id': ident, 'from_run': name, 'path': relative,
                              'subset_sha256': first, 'all_sha256': second, 'equal': first == second}
                    comparisons.append(record); case_files.append(record)
                    if first != second:
                        failures.append({'id': ident, 'path': relative, 'reason': 'formal_image_bytes_differ'})
                cases.append({'id': ident, 'from_run': name, 'entries': a['entry_count'],
                              'formal_images': len(case_files), 'all_equal': all(x['equal'] for x in case_files)})
        require(len(ids) == 20, 'Expected20 distinct fixed-pilot plus extra images')
    except Exception:
        failures.append({'scope': 'global', 'error': traceback.format_exc()})
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'command': [sys.executable, *sys.argv],
        'run': str(run), 'source_sha256': sha(__file__), 'freeze_sha256': sha(HERE / 'freeze_final.json'),
        'exit_code': 1 if failures else 0, 'passed': not failures, 'cases': cases,
        'case_count': len(cases), 'entry_count': sum(c['entries'] for c in cases),
        'file_count': len(comparisons), 'equal_count': sum(x['equal'] for x in comparisons),
        'comparisons': comparisons, 'frozen_algorithm_sources': sources, 'failures': failures,
        'limitations': 'Byte-identical JPG/PNG only establishes fixed-input repeatability, not visual or same-rope correctness.'}
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    snapshot = dest.with_name(dest.stem + '_source.py')
    require(not snapshot.exists(), 'Do not overwrite previous audit source')
    shutil.copy2(__file__, snapshot)
    print(json.dumps({k: report[k] for k in ('passed','exit_code','case_count','entry_count','file_count','equal_count')}), flush=True)
    return report['exit_code']


if __name__ == '__main__':
    sys.exit(main())
