"""只创建本次新版本协议；旧预测与源码均不改。"""
from pathlib import Path
import json, hashlib
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'camera_condition_whole_incoming_20261028_v3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not (HERE/'protocol.json').exists()
p=json.loads((OLD/'protocol.json').read_text(encoding='utf-8'))
p.update(version='IR-A4.0.0',name='断界寻绳 A4',created='2026-10-28',
 predecessor='IR-A3.0.1',budget='Two fixed16 pilots: search-domain repair alone, then boundary sampling geometry repair. At most one further implementation-bug repair, no threshold sweep. Freeze accepted change and process same400.',
 evaluation='Compare original/A3/A4 for all fixed16. Report restored visible support and wrong-tail/neighbor risks, no identity accuracy without ground truth. Structural tests are separate.',
 non_goals=['No training, device, reserved-data access or invented invisible tail.', 'Old cached automatic free seeds retained; missing seed remains unknown, not absence.', 'No new truth labels or per-image parameter edits.'])
p['planned_changes']=['Use significant-component search envelope without OR-ing isolated SAM fragments back in; do not mutate original S.', 'Check original boundary crossings from first stable surface entry, separate multiple probes of same row and tangencies, enforce search-domain availability. Existing raw contrast gates unchanged.']
p['predecessor_source_sha256']={str(OLD/n):sha(OLD/n) for n in ['whole_io.py','free_rope.py','boundary_occlusion.py','whole_path.py','run_whole.py','protocol.json','protocol_addendum_001.json','freeze_all.json']}
p['predecessor_output_manifest_sha256']=sha(OLD/'all_001/output_manifest.json')
dump(HERE/'protocol.json',p)
dump(HERE/'protocol_addendum_001.json',{'purpose':'A4 version-local protocol compatibility file for existing run serializer; not a new unbounded revision budget.','version':'IR-A4.0.0','budget_reference':'protocol.json','old_A3_addendum_preserved':str(OLD/'protocol_addendum_001.json')})
f=HERE/'whole_io.py'
s=f.read_text(encoding='utf-8').replace("VERSION='IR-A3.0.1'","VERSION='IR-A4.0.0'")
f.write_text(s,encoding='utf-8')
print('A4 protocol saved; fixed16 and explicit open400 manifest retained.')
