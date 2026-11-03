"""Record the final runtime freeze and actually observed shell completions."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
names=['h1_detector.py','straight_support.py','f3_tail.py','priors.py','run_h1.py']
pilot=json.loads((HERE/'pilot_003_repaired_003/run_start.json').read_text(encoding='utf-8'))
sources={str(HERE/n):sha(HERE/n) for n in names}
assert sources==pilot['sources']
audit=json.loads((HERE/'pilot_003_repaired_003/independent_audit_001.json').read_text(encoding='utf-8'))
assert audit['exit_code']==0
if not (HERE/'freeze_final.json').exists():
    dump(HERE/'freeze_final.json',dict(version='IR-H1.0.0',utc=datetime.now(timezone.utc).isoformat(),
       sources=sources,pilot='pilot_003_repaired_003',pilot_audit_sha256=sha(HERE/'pilot_003_repaired_003/independent_audit_001.json'),
       protocol_sha256=sha(HERE/'protocol_inputs.json'),numeric_audit_sha256=sha(HERE/'numeric_response_audit.json'),
       command=sys.argv,extra_rule='first8 non-pilot SHA256(H1extra:+id)',remaining_tuning=False))
print(json.dumps({'freeze_current':True,'source_count':len(sources),'final_pilot_audit_passed':True}))
