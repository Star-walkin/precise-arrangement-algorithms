"""P1/H6 on each camera's original orientation, then explicit geometry transform."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import sys,argparse,time,subprocess,concurrent.futures as cf,copy,cv2
from common20 import *
from roi20 import masks
sys.path.insert(0,str(LC1))
from lc1_p1_adapter import P1H6Adapter,_original

class AuditedP1(P1H6Adapter):
    def _support(self,entry,candidate,all_ranked):
        self.candidates=[dict(rank_score=s,entry=e,index=j,kind=c['kind'],continuation_eligible=c.get('continuation_eligible'),
            points_xy=_original(c['points_xy']),stage=c['stage'],supported=c['supported']) for s,e,j,c in all_ranked[:8]]
        return super()._support(entry,candidate,all_ranked)

def normalized(result,mirror):
    out=copy.deepcopy(result)
    if not mirror:return out
    def flip(p):
        q=np.asarray(p,float).reshape(-1,2).copy();q[:,0]=4095-q[:,0];return q.tolist()
    for k in ('path_xy','line_points_xy','seed_xy'):
        if out.get(k) is not None:out[k]=flip(out[k])
    for c in out.get('candidate_paths_audit',[]):c['points_xy']=flip(c['points_xy'])
    return out

def one(job):
    item=next(x for x in plan() if x['id']==job['id']);out=ROOT/'paths_original'/item['id'];out.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(item['source']);p1=None
    for i in job['frames']:
        target=out/f'p1_{i:06d}.json'
        if target.exists():continue
        cap.set(1,i);ok,raw=cap.read();assert ok;r=record(item['id'],i);assert pixelsha(raw)==r['source_pixel_sha256']
        im=cv2.flip(raw,1) if item['camera']=='right' else raw
        surface,_,meta=masks(im,r);surface_original=cv2.flip(surface,1) if item['camera']=='right' else surface
        if p1 is None:p1=AuditedP1(surface_original,PITCH_PRIOR,LC1/'supplied_assets/step_0750.pt',threads=2,camera=item['camera'],surface_mode='entry_band')
        else:p1.surface=surface_original
        p1.candidates=[];tick=time.time();result=p1.step(raw);result['candidate_paths_audit']=getattr(p1,'candidates',[])
        dump(target,dict(id=item['id'],frame=i,time_s=i/item['fps'],source_pixel_sha256=pixelsha(raw),analysis_pixel_sha256=pixelsha(im),
            stage_record_sha256=sha(selected_run(item['id'])/'frames'/f'{i:06d}.json.gz'),
            model_input_orientation='original_unmirrored',result_orientation='left_entering_normalized',
            raw_result=result,result=normalized(result,item['camera']=='right'),elapsed_s=time.time()-tick,
            code_sha256=sha(__file__),p1_adapter_sha256=sha(LC1/'lc1_p1_adapter.py'),roi=meta))
        if p1.last_heatmaps:np.savez_compressed(out/f'heat_{i:06d}.npz',heat=p1.last_heatmaps[0],orientation='original_crop')
        print(item['id'],i,result['status'],round(time.time()-tick,1),flush=True)
    cap.release()

def worker(job):
    log=ROOT/'paths_original_logs'/f"{job['id']}.log";log.parent.mkdir(exist_ok=True)
    with log.open('w',encoding='utf8') as f:p=subprocess.run([sys.executable,__file__,'--id',job['id']],stdout=f,stderr=subprocess.STDOUT)
    print(job['id'],p.returncode,flush=True);return dict(id=job['id'],returncode=p.returncode,log=str(log))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--id');a=ap.parse_args();jobs=read(ROOT/'path_sampling_plan.json')['jobs']
    if a.id:one(next(j for j in jobs if j['id']==a.id))
    else:
        with cf.ThreadPoolExecutor(max_workers=2) as pool:r=list(pool.map(worker,jobs))
        dump(ROOT/'paths_original_receipt.json',dict(jobs=r));assert all(x['returncode']==0 for x in r)
