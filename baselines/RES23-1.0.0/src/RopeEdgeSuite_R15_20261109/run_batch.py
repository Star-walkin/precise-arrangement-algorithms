"""Full fresh per-frame regression. Independent state for every source video."""
import os,sys,time,subprocess,argparse
from common import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);ap.add_argument('--ids',nargs='*');a=ap.parse_args()
    plan=read(ROOT/'plan.json');selected=[x for x in plan['videos'] if not a.ids or x['id'] in a.ids]
    priority=['dual_20260923_113029__left','dual_20260924_111500__left','dual_20260923_113305__right']
    pending=sorted(selected,key=lambda x:(priority.index(x['id']) if x['id'] in priority else 3,
        not x['visual_role'].startswith('active'),x['planned_frames']<100,x['id']))
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    (ROOT/'logs').mkdir(exist_ok=True);running={};finished=[];start=time.time()
    while pending or running:
        while pending and len(running)<a.workers:
            item=pending.pop(0);out=ROOT/'runs'/item['id']
            if (out/'receipt.json').exists():finished.append(dict(id=item['id'],exit_code=0,already_complete=True));continue
            if out.exists():raise RuntimeError('Diagnose partial output before resuming: '+str(out))
            log=(ROOT/'logs'/(item['id']+'.log')).open('wb')
            proc=subprocess.Popen([sys.executable,str(ROOT/'run_one.py'),'--id',item['id']],stdout=log,stderr=subprocess.STDOUT,env=env)
            running[item['id']]=(proc,log);print('START '+item['id'],flush=True)
        for key,(proc,log) in list(running.items()):
            rc=proc.poll()
            if rc is not None:
                log.close();finished.append(dict(id=key,exit_code=rc));del running[key];print(f'FINISHED {key} {rc}',flush=True)
        progress=[]
        for item in selected:
            p=ROOT/'runs'/item['id']/'progress.json'
            if p.exists():progress.append(read(p))
        state=dict(elapsed_s=round(time.time()-start,1),total_videos=len(selected),finished=finished,running=list(running),
            pending=[x['id'] for x in pending],processed_frames=sum(x['processed'] for x in progress),
            planned_frames=sum(x['planned_frames'] for x in selected),progress=progress)
        dump(ROOT/'batch_progress.json',state)
        if pending or running:time.sleep(10)
    dump(ROOT/'batch_execution_receipt.json',state)
    if any(x['exit_code'] for x in finished):raise RuntimeError('Video failure; see logs')
if __name__=='__main__':main()
