import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import subprocess,sys,concurrent.futures as cf
from common20 import *

def select(item):
    n=item['planned_frames'];id=item['id'];candidates=[]
    grid=list(range(10,n,10)) if n>20 else list(range(n))
    for i in grid:
        r=record(id,i);t=r['track'];b=r['boundary']
        if not t['valid'] or not b['fit_valid']:continue
        a,bb,c=t['line_abc'];proxy=(-(a*2238.7+c)/bb-b['edge_y'])/PITCH_PRIOR
        if proxy>=5:candidates.append(i)
    if not candidates:
        candidates=[i for i in grid if record(id,i)['track']['valid']]
        role='diagnostic_only_no_far_proxy_frames'
    else:role='candidate_far_by_legacy_column_proxy_not_final_distance'
    chosen=[candidates[i] for i in np.unique(np.rint(np.linspace(0,len(candidates)-1,min(8,len(candidates)))).astype(int))] if candidates else []
    return dict(id=id,frames=chosen,selection=role)

def worker(job):
    if not job['frames']:return dict(**job,status='no_candidates')
    log=ROOT/'path_logs'/f"{job['id']}.log";log.parent.mkdir(exist_ok=True)
    with log.open('w',encoding='utf8') as f:
        p=subprocess.run([sys.executable,str(ROOT/'probe20.py'),'--id',job['id'],'--frames',','.join(map(str,job['frames'])),'--mode','p1'],stdout=f,stderr=subprocess.STDOUT)
    return dict(**job,status='complete' if p.returncode==0 else 'failed',returncode=p.returncode,log=str(log))

def main():
    jobs=[select(i) for i in plan()];dump(ROOT/'path_sampling_plan.json',dict(jobs=jobs,selection_before_reading_P1_outputs=True,
        note='Legacy d and pitch used solely as permissive far-frame screening proxy; final calibration rechecks distances and identities. Full-video offline calibration is not a causal startup claim.'))
    results=[]
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        for out in pool.map(worker,jobs):
            results.append(out);dump(ROOT/'path_collection_progress.json',results);print(out['id'],out['status'],out['frames'],flush=True)
    assert all(r['status']!='failed' for r in results)
if __name__=='__main__':main()
