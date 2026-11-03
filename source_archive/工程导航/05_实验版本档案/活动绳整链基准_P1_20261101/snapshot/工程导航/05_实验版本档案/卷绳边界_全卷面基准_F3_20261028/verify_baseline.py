"""Explicit-list byte verification only; never follow a reparse point or decode media."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import argparse, hashlib, json, os, stat

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


@lru_cache(maxsize=None)
def safe_path(value):
    p=Path(value).absolute()
    if not p.is_relative_to(ROOT):raise ValueError('Path outside project: '+str(p))
    for q in [p,*p.parents]:
        if getattr(q.lstat(),'st_file_attributes',0)&stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError('Reparse point refused: '+str(q))
        if q==ROOT:break
    return p


def digest(path):
    p=safe_path(str(Path(path).absolute()));h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def check(item):
    try:
        p=safe_path(item['path'])
        if not p.is_file():return {'path':str(p),'error':'missing_file'}
        if 'bytes' in item and p.stat().st_size!=item['bytes']:return {'path':str(p),'error':'size_mismatch'}
        h=digest(p)
        if h!=item['sha256']:return {'path':str(p),'error':'sha256_mismatch','actual_sha256':h}
    except Exception as exc:return {'path':item.get('path'),'error':repr(exc)}
    return None


def relative(value):
    part=Path(value)
    if part.is_absolute() or '..' in part.parts:raise ValueError('Unsafe relative path')
    return safe_path(str(HERE/part))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--snapshots-only',action='store_true');ap.add_argument('--workers',type=int,default=4);ap.add_argument('--receipt',type=Path);a=ap.parse_args()
    m=json.loads((HERE/'baseline_manifest.json').read_text(encoding='utf8'));tasks=[]
    for item in m['snapshot_files']:
        tasks.append({'path':str(relative(item['snapshot_relpath'])),'sha256':item['sha256'],'bytes':item['bytes']})
        if not a.snapshots_only:
            tasks.append({'path':item['origin_path'],'sha256':item['sha256'],'bytes':item['bytes']})
            if 'original_source_path' in item:tasks.append({'path':item['original_source_path'],'sha256':item['sha256'],'bytes':item['bytes']})
    for item in m['local_files']:tasks.append({'path':str(relative(item['path'])),'sha256':item['sha256'],'bytes':item['bytes']})
    dep=HERE/'dependency_manifest.json';tasks.append({'path':str(dep),'sha256':m['dependency_manifest_sha256']})
    if not a.snapshots_only:tasks.extend(json.loads(dep.read_text(encoding='utf8'))['external_files'])
    unique={}
    for item in tasks:
        key=os.path.normcase(str(safe_path(item['path'])))
        if key in unique and unique[key]['sha256']!=item['sha256']:raise ValueError('Conflicting expected hash')
        unique[key]=item
    with ThreadPoolExecutor(max_workers=max(1,min(a.workers,8))) as pool:errors=[r for r in pool.map(check,unique.values()) if r is not None]
    report={'version':m['id'],'completed_utc':datetime.now(timezone.utc).isoformat(),'mode':'snapshots_only' if a.snapshots_only else 'all_references','checked_unique_files':len(unique),'failed':len(errors),'errors':errors,'manifest_sha256':digest(HERE/'baseline_manifest.json'),'no_reparse_points_followed':True,'media_decoded':False,'algorithms_run':False}
    if a.receipt:
        target=a.receipt.absolute() if a.receipt.is_absolute() else HERE/a.receipt
        if target.parent!=HERE:raise ValueError('Receipt must be inside archive')
        with target.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False,indent=2));return int(bool(errors))


if __name__=='__main__':raise SystemExit(main())
