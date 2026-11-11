import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import argparse,time,concurrent.futures as cf
from common20 import *

def process(task):
    from groups20 import group
    from tangent_column import roots_for_curve,estimate
    id,start,end=task;out=ROOT/'groups_v2'/id/'frames';out.mkdir(parents=True,exist_ok=True);done=[]
    for i in range(start,end):
        dest=out/f'{i:06d}.json.gz'
        if dest.exists():continue
        r=record(id,i);frpath=ROOT/'f3_v2'/id/'records'/f'{i:06d}.json';fr=read(frpath);tick=time.time()
        result=dict(frame=i,time_s=fr['time_s'],source_pixel_sha256=fr['source_pixel_sha256'],analysis_pixel_sha256=fr['analysis_pixel_sha256'],
            raw_fields_sha256=fr['fields_sha256'],extraction_record_sha256=sha(frpath),current_F3_called=fr['called'],groups=None,tangent=None,algorithm_errors=[],exact_input_group_reuse=None)
        old=ROOT/'groups_v1'/id/'frames'/f'{i:06d}.json.gz';reused=False
        if fr['called'] and fr['reused_exact_mask_result'] and old.exists():
            with gzip.open(old,'rt',encoding='utf8') as f:prev=json.load(f)
            if prev['raw_fields_sha256']==fr['fields_sha256'] and prev['analysis_pixel_sha256']==r['analysis_pixel_sha256']:
                result.update(groups=prev['groups'],tangent=prev['tangent'],exact_input_group_reuse=dict(path=str(old),sha256=sha(old)));reused=True
        if fr['called'] and not reused:
            assert sha(fr['fields_path'])==fr['fields_sha256']
            with np.load(fr['fields_path']) as z:data={k:z[k] for k in z.files}
            assert str(data['source_pixel_sha256'])==r['analysis_pixel_sha256'];b=dict(r['boundary'])
            if 'analysis_top_curve_xy' in data:b['top_curve_xy']=data['analysis_top_curve_xy']
            abc=r['track'].get('line_abc');initial=roots_for_curve(b['top_curve_xy'],abc,PITCH_PRIOR) if abc is not None else None
            d=float(initial['d']) if initial else float(np.median(np.asarray(b['top_curve_xy'])[:,0]))
            try:result['groups']=group(data,b,d);result['tangent']=estimate(result['groups']['curves'],abc)
            except ValueError as err:
                if str(err)=='insufficient_shared_shape_points':result['reason']=str(err)
                else:raise
        result['elapsed_s']=time.time()-tick;tmp=dest.with_suffix('.part')
        with gzip.open(tmp,'wt',encoding='utf8',compresslevel=2) as f:json.dump(plain(result),f,ensure_ascii=False,allow_nan=False)
        tmp.replace(dest);done.append(i)
    return dict(id=id,start=start,end=end,new_frames=len(done))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    files=[ROOT/f for f in ('groups20.py','group_roi21.py','tangent_column.py')];hashes={str(p):sha(p) for p in files}
    jobs=[(x['id'],i,min(i+10,x['planned_frames'])) for x in plan() for i in range(0,x['planned_frames'],10)];jobs.sort(key=lambda x:x[1])
    pending={};done_rows=[];start=time.time()
    with cf.ProcessPoolExecutor(max_workers=a.workers) as pool:
        while jobs or pending:
            for task in list(jobs):
                if len(pending)>=a.workers*2:break
                id,lo,hi=task
                if all((ROOT/'f3_v2'/id/'records'/f'{i:06d}.json').exists() for i in range(lo,hi)):
                    jobs.remove(task);pending[pool.submit(process,task)]=task
            if not pending:time.sleep(3);continue
            finished,_=cf.wait(pending,timeout=5,return_when=cf.FIRST_COMPLETED)
            for f in finished:
                pending.pop(f);done_rows.append(f.result());progress=dict(chunks=len(done_rows),frames=sum(x['end']-x['start'] for x in done_rows),
                    remaining_chunks=len(jobs)+len(pending),last=done_rows[-1],elapsed_s=round(time.time()-start,1))
                dump(ROOT/'group_v2_progress.json',progress);print(progress,flush=True)
    assert all(sha(p)==h for p,h in hashes.items())
    dump(ROOT/'group_v2_receipt.json',dict(status='complete',frames=sum(x['end']-x['start'] for x in done_rows),chunks=done_rows,producer_hashes=hashes))
