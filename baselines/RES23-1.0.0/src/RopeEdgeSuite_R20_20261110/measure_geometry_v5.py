import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,time,concurrent.futures as cf,cv2
from common20 import *

def process(task):
    from geometry22 import measure
    cv2.setNumThreads(1)
    item,lo,hi=task;id=item['id'];out=ROOT/'geometry_v5'/id/'frames';out.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(item['source']);cap.set(1,lo);done=0
    for i in range(lo,hi):
        ok,raw=cap.read();assert ok
        dest=out/f'{i:06d}.json.gz'
        if dest.exists():continue
        r=record(id,i);assert pixelsha(raw)==r['source_pixel_sha256'];im=cv2.flip(raw,1) if item['camera']=='right' else raw
        gp=ROOT/'groups_v2'/id/'frames'/f'{i:06d}.json.gz'
        with gzip.open(gp,'rt',encoding='utf8') as f:g=json.load(f)
        assert g['analysis_pixel_sha256']==pixelsha(im)
        tick=time.time();result=measure(im,r,g['groups']);result.update(frame=i,time_s=i/item['fps'],source_pixel_sha256=r['source_pixel_sha256'],
            analysis_pixel_sha256=pixelsha(im),groups_file_sha256=sha(gp),elapsed_s=time.time()-tick,
            geometry_code_sha256=sha(ROOT/'geometry22.py'),axis_code_sha256=sha(ROOT/'coil_edge_axis_v4.py'))
        tmp=dest.with_suffix('.part')
        with gzip.open(tmp,'wt',encoding='utf8',compresslevel=2) as f:json.dump(plain(result),f,ensure_ascii=False,allow_nan=False)
        tmp.replace(dest);done+=1
    cap.release();return dict(id=id,lo=lo,hi=hi,new_frames=done)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);a=ap.parse_args()
    hashes={str(ROOT/f):sha(ROOT/f) for f in ('geometry22.py','coil_edge_axis.py','coil_edge_axis_v3.py','coil_edge_axis_v4.py','measure_geometry_v5.py','roi20.py','common20.py')}
    jobs=[(item,i,min(i+10,item['planned_frames'])) for item in plan() for i in range(0,item['planned_frames'],10)];jobs.sort(key=lambda x:x[1])
    pending={};completed=[];start=time.time()
    with cf.ProcessPoolExecutor(max_workers=a.workers) as pool:
        while jobs or pending:
            for task in list(jobs):
                if len(pending)>=2*a.workers:break
                item,lo,hi=task
                if all((ROOT/'groups_v2'/item['id']/'frames'/f'{i:06d}.json.gz').exists() for i in range(lo,hi)):
                    jobs.remove(task);pending[pool.submit(process,task)]=task
            if not pending:time.sleep(3);continue
            done,_=cf.wait(pending,timeout=5,return_when=cf.FIRST_COMPLETED)
            for f in done:
                pending.pop(f);completed.append(f.result());progress=dict(chunks=len(completed),frames=sum(x['hi']-x['lo'] for x in completed),
                    remaining_chunks=len(jobs)+len(pending),last=completed[-1],elapsed_s=round(time.time()-start,1))
                dump(ROOT/'geometry_v5_progress.json',progress);print(progress,flush=True)
    assert all(sha(p)==h for p,h in hashes.items())
    dump(ROOT/'geometry_v5_receipt.json',dict(status='complete',frames=sum(x['hi']-x['lo'] for x in completed),code_hashes=hashes,chunks=completed))
if __name__=='__main__':main()
