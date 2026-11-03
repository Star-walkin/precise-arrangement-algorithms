"""Verify saved source/model snapshot and external frozen response files."""
import hashlib,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    m=json.loads((HERE/'baseline_manifest.json').read_text(encoding='utf8'))
    failures=[]
    for x in m['files']:
        p=HERE/x['snapshot']
        if not p.is_file() or sha(p)!=x['sha256']:failures.append(str(p))
    receipt=dict(snapshot_files=len(m['files']),failures=failures,passed=not failures,
        manifest_sha256=sha(HERE/'baseline_manifest.json'),scope='snapshot source/config/models; external response manifest hash')
    receipt['external_manifest_matches']=sha(HERE/'external_response_manifest.json')==m['external_response_manifest_sha256']
    receipt['passed'] &= receipt['external_manifest_matches']
    (HERE/'verification_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(receipt,ensure_ascii=True));return 0 if receipt['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
