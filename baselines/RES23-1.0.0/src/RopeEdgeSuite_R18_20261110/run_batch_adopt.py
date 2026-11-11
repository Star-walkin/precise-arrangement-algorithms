"""Adopt the six existing workers and fill two unused video processing slots.

Only scheduling changes. All frozen per-frame code and per-video receipts are
unchanged. Existing workers are identified by the prior scheduler's manifest.
"""
import os,sys,time,subprocess
from common import *
def main():
    items=read(ROOT/'plan.json')['videos'];previous=read(ROOT/'batch_progress.json')
    active=set(previous['running']);done=[];started={};logs={};tick=time.time();last={}
    pending=[x for x in items if x['id'] not in active and not (ROOT/'runs'/x['id']/'receipt.json').exists()]
    dump(ROOT/'scheduler_binding.json',dict(adopted=list(active),workers=8,scheduler_sha256=sha(__file__),
        reason='Scheduling only; frozen detection and render files unchanged'))
    while active or pending:
        for item in list(pending):
            if len(active)>=8:break
            id=item['id'];out=ROOT/'runs'/id
            assert not out.exists(),str(out)
            log=(ROOT/'logs'/(id+'.log')).open('wb');logs[id]=log
            env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            started[id]=subprocess.Popen([sys.executable,str(ROOT/'run_one.py'),'--id',id,'--follow'],stdout=log,stderr=subprocess.STDOUT,env=env)
            active.add(id);pending.remove(item);print('START',id,flush=True)
        for id in list(active):
            out=ROOT/'runs'/id
            if (out/'failure.json').exists():raise RuntimeError('Worker failed '+id)
            if id in started and started[id].poll() not in [None,0]:raise RuntimeError('Worker exit '+id)
            if (out/'receipt.json').exists():
                receipt=read(out/'receipt.json');assert receipt['status']=='complete'
                active.remove(id);done.append(dict(id=id,exit_code=0));print('COMPLETE',id,flush=True)
                if id in logs:logs.pop(id).close()
        progress=[]
        for item in items:
            path=ROOT/'runs'/item['id']/'progress.json'
            if path.exists():
                p=read(path);progress.append(p);id=item['id'];n=p['processed']
                if id not in last or n!=last[id][0]:last[id]=(n,time.time())
                if id in active and time.time()-last[id][1]>1800:raise RuntimeError('Worker stalled '+id)
        state=dict(elapsed_s=round(previous['elapsed_s']+time.time()-tick,1),finished=done,running=sorted(active),
            pending=[x['id'] for x in pending],progress=progress,processed_frames=sum(x['processed'] for x in progress),planned_frames=4819,
            scheduler='adopted workers plus eight-slot scheduling')
        dump(ROOT/'batch_progress.json',state)
        if active or pending:time.sleep(10)
    assert len(done)==len(items)
    dump(ROOT/'execution_receipt.json',state)
if __name__=='__main__':main()
