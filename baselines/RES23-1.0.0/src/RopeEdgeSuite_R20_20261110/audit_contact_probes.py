import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import subprocess,sys,concurrent.futures as cf
from common20 import *

def one(task):
    id,i,d=task;out=ROOT/'contact_probe_logs';out.mkdir(exist_ok=True);log=out/f'{id}_{i:06d}.log'
    with log.open('w',encoding='utf8') as f:p=subprocess.run([sys.executable,str(ROOT/'probe_counts.py'),'--id',id,'--frame',str(i),'--d',str(d),'--half','100'],stdout=f,stderr=subprocess.STDOUT)
    print(task,p.returncode,flush=True);return dict(id=id,frame=i,d_provisional=d,returncode=p.returncode,log=str(log))

if __name__=='__main__':
    jobs=[('dual_20260923_113745__left',i,2100) for i in [310,370,397,400]]
    jobs += [('dual_20260924_111500__left',i,2100) for i in [200,260,281,300]]
    jobs += [('dual_20260924_111143__right',i,2520) for i in [140,152,180]]
    with cf.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(one,jobs))
    dump(ROOT/'contact_probe_receipt.json',dict(rows=rows,provisional_d_not_final_calibration=True))
