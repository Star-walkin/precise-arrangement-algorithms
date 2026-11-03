"""Sequential postflight for this active run, with captured real exit codes."""
import json,time,sys,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;start=time.monotonic();rows=[]
while not (HERE/'all_001/summary.json').is_file():
    if time.monotonic()-start>6*3600:raise TimeoutError('Fixed run postflight wait exceeded6hours')
    p=HERE/'all_001/measurement_progress.json'
    if p.is_file():
        try:
            values=json.loads(p.read_text(encoding='utf8'))
            if any(v.get('status')=='failed' for v in values):raise RuntimeError('Measurement failed; no delivery declared')
        except json.JSONDecodeError:pass
    time.sleep(10)
for script,args in [('audit.py',['--run','all_001']),('finalize.py',[])]:
    command=[sys.executable,'-B',str(HERE/script),*args]
    with (HERE/(script.replace('.py','')+'_all_001.log')).open('w',encoding='utf8') as output:
        result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,cwd=str(HERE.parents[1]))
    rows.append(dict(command=command,exit_code=result.returncode))
    (HERE/'postflight_execution_receipt.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
    if result.returncode:raise SystemExit(result.returncode)
print('Audited and delivered16 per-frame videos.',flush=True)
