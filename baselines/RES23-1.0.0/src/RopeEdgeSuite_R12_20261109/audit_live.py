from pathlib import Path
import time,json
import audit_sg2 as audit
ROOT=Path(__file__).resolve().parent
run=ROOT/'runs/dual_20260923_113745';n=600;rows=[]
for i in range(n):
    path=run/'frames'/f'{i:06d}.json.gz';deadline=time.monotonic()+600
    while not path.exists():
        assert time.monotonic()<deadline,'analysis stalled';time.sleep(.5)
    row=audit.analyze(run,i);rows.append(row)
    assert not row['failures'],(i,row['failures'])
    if i%50==0:print('audited',i,flush=True)
audit.dump(run/'audit_rows.json',rows)
print('ALL 600 FRAME AUDITS PASSED',flush=True)
