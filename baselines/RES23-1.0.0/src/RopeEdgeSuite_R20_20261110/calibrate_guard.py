"""Finish each video's calibration only after a current-F3 five-rope audit."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,time,concurrent.futures as cf
from common20 import *

def observe(id,i,cal):
    from count20 import measure
    fr=read(ROOT/'f3_v2'/id/'records'/f'{i:06d}.json');data=None
    if fr['called']:
        with np.load(fr['fields_path']) as z:data={k:z[k] for k in z.files}
    with gzip.open(ROOT/'groups_v2'/id/'frames'/f'{i:06d}.json.gz','rt',encoding='utf8') as f:g=json.load(f)
    with gzip.open(ROOT/'geometry_v5'/id/'frames'/f'{i:06d}.json.gz','rt',encoding='utf8') as f:geo=json.load(f)
    r=record(id,i);q=measure(g['groups'],data,r['boundary'],geo['axis'],cal)
    return {k:q.get(k) for k in ('available','N','N_interval','status','d','d_interval','reasons','complete_F3_phase')}

def process(item):
    from calibration20 import collect,calibrate
    id=item['id'];path=ROOT/'calibration'/f'{id}.json';path.parent.mkdir(exist_ok=True)
    phases=next(v for v in read(ROOT/'motion_phases_v2.json')['videos'] if v['id']==id)['epochs'];out=[]
    for phase in phases:
        candidates,_=collect(item,phase);allowed=set();audit=[]
        for row in candidates:
            q=observe(id,row['frame'],dict(d=row['d'],interval=row['interval'],status='conditional_geometry_estimate'))
            passed=bool(q['available'] and q['N_interval'][0]>=5);audit.append(dict(frame=row['frame'],phase='dynamic_column_screen',passed=passed,count=q))
            if passed:allowed.add(row['frame'])
        final=calibrate(item,phase,allowed)
        # Re-check at the final fixed-column sensitivity interval. Removal can
        # change the robust column, so repeat until stable or unidentifiable.
        for iteration in range(3):
            if final['d'] is None:break
            rejected=[]
            for i in sorted(allowed):
                q=observe(id,i,final);passed=bool(q['available'] and q['N_interval'][0]>=5)
                audit.append(dict(frame=i,phase=f'fixed_column_audit_{iteration}',passed=passed,count=q))
                if not passed:rejected.append(i)
            if not rejected:break
            allowed.difference_update(rejected);final=calibrate(item,phase,allowed)
        else:
            final.update(status='unidentified',d=None,interval=None,reasons=final['reasons']+['far_frame_screen_not_converged'])
        final['current_F3_five_rope_audit']=audit;final['far_guard_accepted_frames']=sorted(allowed)
        d=final['d'];iv=final['interval'];mirror=item['camera']=='right'
        final['d_original_image']=None if d is None else 4095-d if mirror else d
        final['d_interval_original_image']=None if iv is None else [4095-iv[1],4095-iv[0]] if mirror else iv
        out.append(final)
    result=dict(id=id,camera=item['camera'],epochs=out,coordinates='left_entering_normalized',
        original_frame_width=4096,all_reference_frames_rechecked_with_current_F3=True,
        physical_d_confirmed=False,code_sha256=sha(__file__),calibrator_sha256=sha(ROOT/'calibration20.py'),counter_sha256=sha(ROOT/'count20.py'))
    dump(path,result);print(id,[(x['status'],x['d'],len(x['far_guard_accepted_frames'])) for x in out],flush=True);return dict(id=id,path=str(path),sha256=sha(path))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=3);a=ap.parse_args()
    hashes={str(ROOT/f):sha(ROOT/f) for f in ('calibrate_guard.py','calibration20.py','count20.py','recover_points20.py','geometry20.py')}
    jobs=list(plan());pending={};completed=[];start=time.time()
    with cf.ProcessPoolExecutor(max_workers=a.workers) as pool:
        while jobs or pending:
            for item in list(jobs):
                if len(pending)>=a.workers:break
                if all((ROOT/'geometry_v5'/item['id']/'frames'/f'{i:06d}.json.gz').exists() for i in range(item['planned_frames'])):
                    jobs.remove(item);pending[pool.submit(process,item)]=item['id']
            if not pending:time.sleep(5);continue
            done,_=cf.wait(pending,timeout=5,return_when=cf.FIRST_COMPLETED)
            for f in done:pending.pop(f);completed.append(f.result());dump(ROOT/'calibration_progress.json',dict(completed=completed,remaining=len(jobs)+len(pending),elapsed_s=time.time()-start))
    assert all(sha(p)==h for p,h in hashes.items())
    dump(ROOT/'d_calibration.json',dict(videos=[read(x['path']) for x in completed],partial=False,producer_hashes=hashes))
