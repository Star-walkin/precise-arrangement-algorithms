"""Read-only open400 input freeze and optional fresh B automatic-A7 inference."""
from pathlib import Path
import argparse,json,hashlib,sys,time
import numpy as np
HERE=Path(__file__).resolve().parent
H1=HERE.parent/'postprocess_b_20261030_v1'
ROOT=HERE.parents[2]
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare():
    import cv2
    from free_candidates import suggest_a7_entries
    old=load(H1/'protocol_inputs.json');jobs=[];added=[]
    for source in old['jobs']:
        j=dict(source);j['original_seeds']=source['seeds'];j['seeds']=[dict(s,old_entry_index=k) for k,s in enumerate(source['seeds'])]
        p=ROOT/'数据资产/05_派生与实验结果/camera_condition_free_straight_20261030_v7/all_001/记录'/(j['camera_role']+'_camera')/(j['id']+'.json')
        a7=load(p)['result'];priors=dict(a7_candidates=a7['candidates'],original_shape=j['shape'],input_dependency_sha256={str(p):sha(p)})
        new=suggest_a7_entries(priors,j['seeds'])
        for seed in new:j['seeds'].append(seed)
        j['response_files']=[str(H1/'responses'/j['id']/f'entry_{k:02d}.npz') for k in range(len(source['seeds']))]
        if new:
            d=HERE/'new_entry_cache';d.mkdir(exist_ok=True)
            sys.path.insert(0,str(H1));from prepare_inference import seed_response
            assert sha(j['image_path'])==j['image_sha256']
            im=cv2.imdecode(np.fromfile(j['image_path'],np.uint8),cv2.IMREAD_COLOR)
            gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
            image=cv2.resize(gray[1050:],(2048,975),interpolation=cv2.INTER_AREA)
            seeds=np.stack([seed_response(cv2,s['xy'],image.shape) for s in new])
            cp=d/(j['id']+'.npz');np.savez_compressed(cp,image=image,seeds=seeds)
            for k in range(len(new)):j['response_files'].append(str(HERE/'responses'/j['id']/f'a7_{k:02d}.npz'))
            added.append(dict(id=j['id'],cache=str(cp),cache_sha256=sha(cp),seeds=new,output_files=j['response_files'][-len(new):]))
        jobs.append(j)
    pilot=old['pilot_ids'].copy();pilot[pilot.index('dual_20260924_111143__pair0075__right')]='R_run_20260923_01__f001274__right'
    out=dict(version='IR-H2.0.0',source_count=400,jobs=jobs,old_entry_count=299,new_entries=added,
      entry_count=sum(len(j['seeds']) for j in jobs),eligible_count=len(jobs),unresolved=old['upstream_skipped'],
      pilot_ids=pilot,model_B=old['model_B'],input_protocol_sha256=sha(H1/'protocol_inputs.json'),
      seed_policy='all previous automatic entries retained; incompatible accepted A7 axis adds separately inferred conditions',
      no_training=True,labels_read=False,reserved_data_accessed=False,device_connected=False,
      unresolved_is_not_absent_rope=True,source_sha256=sha(__file__))
    assert not (HERE/'inputs.json').exists();dump(HERE/'inputs.json',out)
    print(json.dumps(dict(images=len(jobs),old_entries=299,new_entries=sum(len(j['seeds']) for j in added),new_images=[j['id'] for j in added])),flush=True)

def infer():
    import torch
    sys.path.insert(0,str(HERE.parent));from train import load_model,tensor_sample
    cfg=load(HERE/'inputs.json');cp=cfg['model_B'];assert sha(cp['path'])==cp['sha256']
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    checkpoint=torch.load(cp['path'],map_location='cpu',weights_only=False);net=load_model(cp['path']);rows=[];t=time.time()
    with torch.inference_mode():
        for job in cfg['new_entries']:
            assert sha(job['cache'])==job['cache_sha256']
            z=np.load(job['cache']);im=z['image']
            for k,seed in enumerate(z['seeds']):
                zeros=np.zeros_like(im);arrays=dict(image=im,seed=seed,target=zeros,valid=zeros.astype(bool),stage=zeros.astype(np.uint8),roi=np.ones_like(im,bool))
                x,_,_,_,norm=tensor_sample(arrays,checkpoint['config'],augment=False)
                response=net(x)[0,0,:975,:2048].numpy().copy()
                assert np.isfinite(response).all() and 0<=response.min()<=response.max()<=1
                dest=Path(job['output_files'][k]);dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
                np.savez_compressed(dest,response=response)
                rows.append(dict(id=job['id'],seed=job['seeds'][k],response_file=str(dest),response_sha256=sha(dest),normalization=norm,minimum=float(response.min()),maximum=float(response.max())))
    dump(HERE/'new_inference_receipt.json',dict(exit_code=0,model_sha256=sha(cp['path']),device='cpu',torch=torch.__version__,forward_calls=len(rows),seconds=time.time()-t,rows=rows,labels_read=False))
    print(json.dumps(dict(forward_calls=len(rows),seconds=time.time()-t)),flush=True)
if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','infer']);a=p.parse_args();(prepare if a.mode=='prepare' else infer)()
