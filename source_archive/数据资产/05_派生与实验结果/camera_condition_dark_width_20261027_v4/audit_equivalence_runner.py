"""固定单图独立核对串行审计与4worker缓存审计的完整返回值。"""
import json
from concurrent.futures import ProcessPoolExecutor
import audit
import audit_cached


def main():
    row=audit.selection('pilot')[0][0]
    inv={f['path']:f['sha256'] for f in audit.read_json(audit.D1/'all_001/output_manifest.json')['files']}
    reference=audit.audit_one(row,audit.HERE/'pilot_003',inv)
    with ProcessPoolExecutor(max_workers=4,initializer=audit_cached._audit_worker_init,initargs=(inv,)) as pool:
        result=list(pool.map(audit_cached._audit_worker,[(row,audit.HERE/'pilot_003')]))[0]
    report={'method':'Original serial audit_one vs cached4-worker submitted task; exact complete dict comparison on fixed pilot003 first image.',
            'output_file':row['output_file'],'identical':reference==result,'case_status':result['status'],'errors':result['errors'],'workers':4,
            'original_sha256':audit.digest(audit.HERE/'audit.py'),
            'cached_parallel_sha256':audit.digest(audit.HERE/'audit_cached.py'),
            'cached_serial_sha256':audit.digest(audit.HERE/'audit_cached_serial.py'),
            'earlier_harness_attempt':'Windows spawn cannot import <stdin>; moved identical check into guarded saved runner; no prediction changes.'}
    with (audit.HERE/'audit_parallel_equivalence.json').open('x',encoding='utf8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False));assert report['identical']


if __name__=='__main__':main()
