import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import subprocess,sys,concurrent.futures as cf
from common20 import *

def one(task):
    id,i,d=task;out=ROOT/'count_probe_logs';out.mkdir(exist_ok=True);log=out/f'{id}_{i:06d}.log'
    with log.open('w',encoding='utf8') as f:p=subprocess.run([sys.executable,str(ROOT/'probe_counts.py'),'--id',id,'--frame',str(i),'--d',str(d),'--half','80'],stdout=f,stderr=subprocess.STDOUT)
    print(task,p.returncode,flush=True);return dict(id=id,frame=i,d_provisional=d,returncode=p.returncode,log=str(log))

if __name__=='__main__':
    jobs=[]
    for id,dd,frames in [
        ('dual_20260923_101517__right',2550,[10,70,140]),('dual_20260923_101643__right',2580,[10,70,140]),
        ('dual_20260923_113029__left',2120,[50,90,120]),('dual_20260923_113305__right',2600,[50,90,120]),
        ('dual_20260923_113745__left',2100,[50,100,140]),('dual_20260924_111143__right',2520,[50,90,120]),
        ('dual_20260924_111500__left',2100,[15,60,120])]:jobs.extend((id,i,dd) for i in frames)
    with cf.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(one,jobs))
    dump(ROOT/'count_probe_receipt.json',dict(rows=rows,provisional_d_not_final_calibration=True))
