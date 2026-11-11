import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import argparse,time,concurrent.futures as cf,traceback
from common20 import *

def process(task):
    from groups20 import group
    from tangent_column import roots_for_curve,estimate
    id,start,end=task;out=ROOT/'groups_v1'/id;out.mkdir(parents=True,exist_ok=True);(out/'frames').mkdir(exist_ok=True)
    done=[]
    for i in range(start,end):
        p=out/'frames'/f'{i:06d}.json.gz'
        if p.exists():continue
        r=record(id,i);fr=read(ROOT/'f3_v1'/id/'records'/f'{i:06d}.json');tick=time.time()
        result=dict(frame=i,time_s=fr['time_s'],source_pixel_sha256=fr['source_pixel_sha256'],analysis_pixel_sha256=fr['analysis_pixel_sha256'],
            raw_fields_sha256=fr['fields_sha256'],current_F3_called=fr['called'],groups=None,tangent=None,algorithm_errors=[])
        if fr['called']:
            with np.load(ROOT/'f3_v1'/id/'fields'/f'{i:06d}.npz') as z:data={k:z[k] for k in z.files}
            assert str(data['source_pixel_sha256'])==r['analysis_pixel_sha256']
            curve=r['boundary']['top_curve_xy'];abc=r['track'].get('line_abc')
            initial=roots_for_curve(curve,abc,PITCH_PRIOR) if abc is not None else None
            d=float(initial['d']) if initial else float(np.median(np.asarray(curve)[:,0]))
            try:
                result['groups']=group(data,r['boundary'],d)
                result['tangent']=estimate(result['groups']['curves'],abc)
            except ValueError as err:
                if str(err)=='insufficient_shared_shape_points':result['reason']=str(err)
                else:raise
        result['elapsed_s']=time.time()-tick
        tmp=p.with_suffix('.part')
        with gzip.open(tmp,'wt',encoding='utf8',compresslevel=2) as f:json.dump(plain(result),f,ensure_ascii=False,allow_nan=False)
        tmp.replace(p);done.append(i)
    return dict(id=id,start=start,end=end,new_frames=len(done))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    files=[ROOT/'groups20.py',ROOT/'group_all.py',ROOT/'tangent_column.py'];hashes={str(p):sha(p) for p in files}
    jobs=[(item['id'],i,min(i+10,item['planned_frames'])) for item in plan() for i in range(0,item['planned_frames'],10)]
    # Short ready blocks across all sources; avoid one video monopolizing work.
    jobs.sort(key=lambda q:q[1]);pending={};completed=[];started=time.time()
    with cf.ProcessPoolExecutor(max_workers=a.workers) as pool:
        while jobs or pending:
            for task in list(jobs):
                if len(pending)>=a.workers*2:break
                id,lo,hi=task
                if all((ROOT/'f3_v1'/id/'records'/f'{i:06d}.json').exists() for i in range(lo,hi)):
                    jobs.remove(task);pending[pool.submit(process,task)]=task
            if not pending:time.sleep(3);continue
            done,_=cf.wait(pending,timeout=5,return_when=cf.FIRST_COMPLETED)
            for f in done:
                task=pending.pop(f);result=f.result();completed.append(result)
                progress=dict(completed_chunks=len(completed),remaining_chunks=len(jobs)+len(pending),
                    frames_completed=sum(x['end']-x['start'] for x in completed),last=result,elapsed_s=round(time.time()-started,1))
                dump(ROOT/'group_progress.json',progress);print(progress,flush=True)
    assert all(sha(p)==h for p,h in hashes.items())
    dump(ROOT/'group_collection_receipt.json',dict(status='complete',frames=sum(x['end']-x['start'] for x in completed),
        chunks=completed,algorithm_hashes=hashes,algorithm_unchanged=True))
if __name__=='__main__':main()
