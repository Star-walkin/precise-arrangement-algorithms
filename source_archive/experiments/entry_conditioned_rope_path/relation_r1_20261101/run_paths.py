"""Reference-free real-entry R0/R1 full-path inference."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'1')
from common import *
import numpy as np,cv2,time,gzip,sys,concurrent.futures,traceback,argparse
from spatial import Context,P
from search_r1 import PortableRanker,run
RANK=None
def init():
    global RANK
    cv2.setNumThreads(1);RANK=PortableRanker()
def gray(path):return cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.
def edge_gallery(job,seed):
    role=job['camera_role']+'_camera';p=ROOT/'数据资产/05_派生与实验结果/camera_condition_sam2_20261009_v1/all_001/body_branches_002/masks'/role/(job['id']+'.png')
    mask=cv2.imdecode(np.fromfile(p,np.uint8),cv2.IMREAD_GRAYSCALE)>0;d=np.asarray(seed[-1])-seed[0];points=[]
    for y in range(1052,2996,6):
        xs=np.flatnonzero(mask[y])
        if len(xs):points.append([int(xs[0] if d[0]<0 else xs[-1]),y])
    return points,dict(source=str(p),sha256=sha(p),meaning='far-side visible spool silhouette candidate; no target endpoint identity truth')
def one(job,branch,outname):
    tick=time.time();key=job['key'];out=HERE/outname/branch/(key+'.json.gz')
    if out.exists():return load_gz(out)['receipt']
    cache=Path(job['cache_path']);meta=load(cache.with_suffix('.json'));assert meta['job']==job
    mode=job['mode'];edge=[];tracks=[];scene={};entry={};surface_meta=None
    if mode=='GALLERY_AUTO':
        j=next(j for j in load(HERE/'protocol.json')['jobs'] if j['id']==job['id'])
        rec=load_gz(H6/'all_001/记录'/(job['id']+'.json.gz'));tracks=rec['surface_tracks']['tracks']
        # Geometry only; old B selected target/free paths are never input.
        edge,surface_meta=edge_gallery(j,job['seed_xy']);entry=j['seeds'][job['entry']]
    else:
        ix=load(PAIR/'data/index.json');item=next(i for i in ix['items'] if i['id']==job['id'])
        if mode=='AUTO':
            entry=load(item['auto_entry_path']);scene=load(entry['scene_path']) if entry.get('available') else {}
        else:scene=load(item['scene_path']);entry=dict(mode='CONDITIONAL_KNOWN_ENTRY',free_seed=job['seed_xy'],available=True)
        edge=scene.get('target_edge_points_xy',[])
    if meta['available']:
        assert sha(cache)==meta['sha256']
        with np.load(cache) as z:response=z['response'].copy();features=z['features'].copy()
        ctx=Context(gray(job['image_path']),response,features,job['seed_xy'],tracks)
        detailed=mode!='GALLERY_AUTO' or job['id'] in load(HERE/'protocol.json')['controlled_h6_ids']
        result=run(ctx,RANK,relation=branch=='R1',edge=edge,trace=detailed)
    else:result=dict(candidates=[],selected_candidate_id=None,status='AUTOMATIC_ENTRY_UNAVAILABLE',ambiguous=False,seconds=0,search={})
    if scene:
        import scene as scene_module
        for c in result['candidates']:c['stage']=scene_module.stage_at_points(scene,np.asarray(c['points_xy'])).tolist()
    elif result['candidates']:
        for c in result['candidates']:c['stage']=[1 if s=='entry_condition' else 3 for s in c['point_status']]
    receipt=dict(key=key,id=job['id'],branch=branch,mode=mode,entry_index=job['entry'],available=meta['available'],status=result['status'],
        candidate_count=len(result['candidates']),selected=result['selected_candidate_id'],ambiguous=result['ambiguous'],
        top_length=result['candidates'][0]['length_px'] if result['candidates'] else 0,seconds=time.time()-tick,
        path=str(out),response=str(cache),response_sha256=meta.get('sha256'),ranking_before_pruning=True)
    record=dict(job=job,receipt=receipt,result=result,edge_target=scene,entry=entry,surface_edge=surface_meta,
        response_cache=meta,parameters=P,relation_model_sha256=sha(HERE/'model/final_portable.npz'),
        target_labels_read=False,old_selected_paths_as_input=False,history='own predicted nodes after initial short seed only')
    out.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(out,'wt',encoding='utf8') as f:json.dump(record,f,ensure_ascii=False,allow_nan=False)
    return receipt
def load_gz(p):
    with gzip.open(p,'rt',encoding='utf8') as f:return json.load(f)
def main():
    p=argparse.ArgumentParser();p.add_argument('--subset',choices=['train','fixed','gallery','pilot'],required=True);p.add_argument('--branches',nargs='+',default=['R1']);p.add_argument('--out',default='paths_001');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    sys.stdout.reconfigure(encoding='utf8');jobs=load(HERE/'cache_jobs.json');protocol=load(HERE/'protocol.json')
    selected=[j for j in jobs if (a.subset=='gallery' and j['mode']=='GALLERY_AUTO') or
        (a.subset=='train' and j.get('split')=='train') or (a.subset=='fixed' and j['mode']!='GALLERY_AUTO') or
        (a.subset=='pilot' and j['mode']=='GALLERY_AUTO' and j['id'] in protocol['controlled_h6_ids'][:4])]
    out=HERE/a.out;out.mkdir(exist_ok=True);tick=time.time();rows=[];errors=[]
    dump(out/f'run_start_{a.subset}.json',dict(command=sys.argv,keys=[j['key'] for j in selected],branches=a.branches,
        protocol_sha256=sha(HERE/'protocol.json'),source_hashes={str(HERE/n):sha(HERE/n) for n in ['spatial.py','search_r1.py','run_paths.py']},parameters=P))
    with concurrent.futures.ProcessPoolExecutor(a.workers,initializer=init) as ex:
        pending={ex.submit(one,j,b,a.out):(j['key'],b) for j in selected for b in a.branches}
        for f in concurrent.futures.as_completed(pending):
            try:rows.append(f.result())
            except Exception:errors.append(dict(key=pending[f],error=traceback.format_exc()))
            if (len(rows)+len(errors))%5==0 or len(rows)+len(errors)==len(pending):print(dict(done=len(rows),errors=len(errors),total=len(pending),seconds=round(time.time()-tick,1)),flush=True)
    dump(out/f'summary_{a.subset}.json',dict(rows=rows,errors=errors,planned=len(pending),exit_code=int(bool(errors)),seconds=time.time()-tick))
    if errors:print(errors[:1]);sys.exit(bool(errors))
if __name__=='__main__':main()
