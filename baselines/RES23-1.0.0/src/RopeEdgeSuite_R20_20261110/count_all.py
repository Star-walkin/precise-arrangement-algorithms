import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,time,concurrent.futures as cf
from common20 import *

def process(task):
    from count20 import measure
    item,lo,hi=task;id=item['id'];out=ROOT/'counts_v1'/id/'frames';out.mkdir(parents=True,exist_ok=True)
    cp=ROOT/'calibration'/f'{id}.json';cal=read(cp);ch=sha(cp);new=0
    for i in range(lo,hi):
        dest=out/f'{i:06d}.json.gz'
        if dest.exists():continue
        phase=next(x for x in cal['epochs'] if x['start']<=i<x['end'])
        r=record(id,i);fp=ROOT/'f3_v2'/id/'records'/f'{i:06d}.json';fr=read(fp)
        gp=ROOT/'groups_v2'/id/'frames'/f'{i:06d}.json.gz';ap=ROOT/'geometry_v5'/id/'frames'/f'{i:06d}.json.gz'
        with gzip.open(gp,'rt',encoding='utf8') as f:g=json.load(f)
        with gzip.open(ap,'rt',encoding='utf8') as f:geo=json.load(f)
        data=None
        if fr['called'] and phase['d'] is not None:
            assert sha(fr['fields_path'])==fr['fields_sha256']
            with np.load(fr['fields_path']) as z:data={k:z[k] for k in z.files}
            assert str(data['source_pixel_sha256'])==r['analysis_pixel_sha256']
        tick=time.time();result=measure(g['groups'],data,r['boundary'],geo['axis'],phase)
        row=dict(frame=i,time_s=i/item['fps'],epoch=phase['epoch'],phase_direction=phase['direction'],source_pixel_sha256=r['source_pixel_sha256'],
            analysis_pixel_sha256=r['analysis_pixel_sha256'],calibration_file_sha256=ch,current_F3_called=fr['called'],F3_record_sha256=sha(fp),
            F3_fields_sha256=fr['fields_sha256'],groups_file_sha256=sha(gp),geometry_file_sha256=sha(ap),
            measurement=result,axis=geo['axis'],algorithm_errors=[],elapsed_s=time.time()-tick)
        tmp=dest.with_suffix('.part')
        with gzip.open(tmp,'wt',encoding='utf8',compresslevel=2) as f:json.dump(plain(row),f,ensure_ascii=False,allow_nan=False)
        tmp.replace(dest);new+=1
    return dict(id=id,lo=lo,hi=hi,new_frames=new)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    hashes={str(ROOT/f):sha(ROOT/f) for f in ('count20.py','count_all.py','recover_points20.py','geometry20.py')}
    jobs=[(item,i,min(i+10,item['planned_frames'])) for item in plan() for i in range(0,item['planned_frames'],10)];jobs.sort(key=lambda x:x[1])
    pending={};completed=[];start=time.time()
    with cf.ProcessPoolExecutor(max_workers=a.workers) as pool:
        while jobs or pending:
            for task in list(jobs):
                if len(pending)>=a.workers*2:break
                if (ROOT/'calibration'/f"{task[0]['id']}.json").exists():jobs.remove(task);pending[pool.submit(process,task)]=task
            if not pending:time.sleep(5);continue
            done,_=cf.wait(pending,timeout=5,return_when=cf.FIRST_COMPLETED)
            for f in done:
                pending.pop(f);completed.append(f.result());progress=dict(chunks=len(completed),frames=sum(x['hi']-x['lo'] for x in completed),remaining_chunks=len(jobs)+len(pending),elapsed_s=time.time()-start,last=completed[-1])
                dump(ROOT/'count_progress.json',progress);print(progress,flush=True)
    assert all(sha(p)==h for p,h in hashes.items())
    dump(ROOT/'count_receipt.json',dict(status='complete',frames=sum(x['hi']-x['lo'] for x in completed),producer_hashes=hashes,chunks=completed))
