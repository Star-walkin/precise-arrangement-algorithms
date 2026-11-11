"""Pipeline the boundary revision behind verified per-frame baseline stages."""
import os,sys,subprocess,time
from common import *

def main():
    items=read(ROOT/'plan.json')['videos'];priority=['dual_20260924_111143__right','dual_20260923_113745__left',
        'dual_20260923_113305__right','dual_20260923_113029__left','dual_20260924_111500__left']
    pending=sorted(items,key=lambda x:(priority.index(x['id']) if x['id'] in priority else 9,
        not x['visual_role'].startswith('active'),x['id']))
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    logs=ROOT/'logs_bg9';logs.mkdir(exist_ok=True);running={};finished=[];start=time.time()
    while pending or running:
        base_complete=sum((ROOT/'runs'/x['id']/'receipt.json').exists() for x in items)
        workers=8 if base_complete else 4
        for item in list(pending):
            if len(running)>=workers:break
            if not (ROOT/'runs'/item['id']/'binding.json').exists():continue
            id=item['id'];out=ROOT/'runs_bg9'/id
            if (out/'receipt.json').exists():finished.append(dict(id=id,exit_code=0));pending.remove(item);continue
            if out.exists():raise RuntimeError('Diagnose prior partial BG9 output: '+str(out))
            log=(logs/(id+'.log')).open('wb')
            proc=subprocess.Popen([sys.executable,str(ROOT/'replay_bg9_one.py'),'--id',id,'--follow'],stdout=log,stderr=subprocess.STDOUT,env=env)
            running[id]=(proc,log);pending.remove(item);print('START BG9 '+id,flush=True)
        for id,(proc,log) in list(running.items()):
            rc=proc.poll()
            if rc is not None:
                log.close();del running[id];finished.append(dict(id=id,exit_code=rc));print('FINISHED BG9',id,rc,flush=True)
        progress=[]
        for item in items:
            p=ROOT/'runs_bg9'/item['id']/'progress.json'
            if p.exists():progress.append(read(p))
        state=dict(elapsed_s=round(time.time()-start,1),finished=finished,running=list(running),
            pending=[x['id'] for x in pending],base_complete=base_complete,progress=progress,
            processed_frames=sum(x['processed'] for x in progress),planned_frames=sum(x['planned_frames'] for x in items))
        dump(ROOT/'bg9_progress.json',state)
        if pending or running:time.sleep(10)
    dump(ROOT/'bg9_execution_receipt.json',state)
    assert all(x['exit_code']==0 for x in finished)
if __name__=='__main__':main()
