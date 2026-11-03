"""Archive the exact independent auditor and its checked source snapshots.

This adds provenance only. It does not rerun detection, alter any threshold,
modify a prediction, or revise an existing audit/manifest.
"""
from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    args = ap.parse_args()
    run = HERE / args.run
    if run.parent != HERE or not run.is_dir():
        raise ValueError('Existing direct-child run required')
    target = run / 'audit_provenance.json'
    if target.exists():
        raise FileExistsError(target)
    audit = read(run / 'audit.json')
    context = read(run / 'run_context.json')
    auditor = HERE / 'audit_seam.py'
    if digest(auditor) != audit['auditor_sha256']:
        raise ValueError('Current auditor differs from executed auditor')
    archive = run / '记录/audit_tools/audit_seam.py'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        if digest(archive) != audit['auditor_sha256']:
            raise ValueError('Existing archived auditor differs')
    else:
        with archive.open('xb') as f:
            f.write(auditor.read_bytes())
    snapshots = []
    for original, expected in context['source_sha256'].items():
        rel = Path(context['source_snapshot_paths'][original])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe snapshot path')
        p = run / rel
        actual = digest(p)
        if actual != expected:
            raise ValueError('Snapshot digest differs: ' + str(p))
        snapshots.append({'original_source': original, 'snapshot': rel.as_posix(),
                          'size': p.stat().st_size, 'sha256': actual})
    references = []
    for name in ('audit.json', 'output_manifest.json', 'run_context.json', 'summary.json'):
        p = run / name
        references.append({'path': name, 'size': p.stat().st_size, 'sha256': digest(p)})
    if context['mode'] == 'all':
        references.append({'path': str(HERE / 'freeze_all.json'),
                           'sha256': digest(HERE / 'freeze_all.json')})
    payload = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'run': str(run),
        'scope': 'Companion provenance for the exact executed independent audit; immutable prediction manifest remains unchanged.',
        'auditor_snapshot': archive.relative_to(run).as_posix(),
        'auditor_sha256': digest(archive),
        'checked_source_snapshots': snapshots,
        'reference_files': references,
        'provenance_writer_sha256': digest(__file__),
        'detection_or_fit_rerun': False,
        'prediction_or_audit_modified': False,
    }
    with target.open('x', encoding='utf8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps({'path': str(target), 'snapshots': len(snapshots),
                      'sha256': digest(target)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
