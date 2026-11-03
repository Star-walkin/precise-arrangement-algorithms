"""Frozen model inference, with input-only caches and separate seed interventions."""
import argparse,sys,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,torch
from common import HERE,B_MODEL,load,dump,sha
from network import load_network
from preprocessing import prepare_input

def config(net,variant):return {**net.config,'variant':'D' if variant=='D' else 'C'}
def frozen_inputs():
    choice=load(HERE/'selection.json');manifest=load(HERE/'inputs/manifest.json')
    assert all(sha(p)==s for p,s in choice['source_hashes'].items()),'source changed after training-only choice'
    assert manifest['protocol_sha256']==sha(HERE/'protocol.json')
    return choice,manifest
def response(net,x):
    r=net(x)[0,0,:975,:2048].numpy().astype(np.float32)
    assert np.isfinite(r).all() and 0<=r.min()<=r.max()<=1
    return r
def save_response(dest,r,meta):
    dest.mkdir(parents=True,exist_ok=False);p=dest/'response.npz';np.savez_compressed(p,response=r)
    meta.update(response_file=str(p),response_sha256=sha(p),shape=[975,2048],dtype='float32',bbox=[0,1050,4096,3000],scale=.5,
        mapping='model_xy=(original_xy-[0,1050])*.5; evaluate the saved half-resolution field directly',
        distribution=None if not np.isfinite(r).any() else dict(mean=float(r.mean()),std=float(r.std()),quantiles=np.quantile(r,[0,.01,.5,.99,1]).tolist(),fraction_gt_09=float((r>.9).mean())))
    dump(dest/'inference.json',meta)
    return [{'path':str(q),'sha256':sha(q)} for q in [p,dest/'inference.json']]

@torch.inference_mode()
def verify():
    out=HERE/'baseline_replay';out.mkdir(exist_ok=False);net=load_network(B_MODEL);cfg=config(net,'B')
    manifest=load(HERE/'inputs/manifest.json');ids=load(HERE/'protocol.json')['user_cases'];rows=[]
    for j in manifest['batch']:
        if j['id'] not in ids:continue
        with np.load(j['input_cache'],allow_pickle=False) as z:gray=z['image'];seed=z['seeds'][0]
        x,meta=prepare_input(gray,seed,cfg);r=response(net,x)
        with np.load(j['response_files'][0],allow_pickle=False) as z:old=z['response']
        delta=np.abs(r-old);rows.append(dict(id=j['id'],max_abs=float(delta.max()),mean_abs=float(delta.mean()),old_response_sha256=sha(j['response_files'][0]),input_cache_sha256=sha(j['input_cache']),input=meta))
    good=len(rows)==2 and max(r['max_abs'] for r in rows)<1e-5
    dump(out/'receipt.json',dict(command=sys.argv,rows=rows,pass_tolerance=1e-5,status='passed' if good else 'failed',model_sha256=sha(B_MODEL),exit_code=int(not good)))
    print(rows,flush=True);assert good,'New cache/zero auxiliary must reproduce frozen B before new inference'

@torch.inference_mode()
def evaluation():
    choice,manifest=frozen_inputs();root=HERE/'evaluation';root.mkdir(exist_ok=True)
    assert not (root/'freeze.json').exists();files=[];tick=time.time();rows=[]
    for variant in ['B','C','D']:
        model=choice['models'][variant];assert sha(model['path'])==model['sha256'];net=load_network(model['path']);cfg=config(net,variant)
        for item in manifest['evaluation']:
            assert sha(item['input_cache'])==item['input_cache_sha256']
            dest=root/'predictions'/variant/item['mode']/item['id']
            meta={**item,'variant':variant,'model_path':model['path'],'model_sha256':model['sha256'],'input_only':True,'labels_read':False,'network_forward_called':bool(item['available'])}
            if item['available']:
                with np.load(item['input_cache'],allow_pickle=False) as z:gray=z['image'];seed=z['seed']
                x,stats=prepare_input(gray,seed,cfg);r=response(net,x);meta['input_statistics']=stats
            else:r=np.full((975,2048),np.nan,np.float32)
            files.extend(save_response(dest,r,meta));rows.append(dict(id=item['id'],variant=variant,mode=item['mode'],available=item['available']))
            if item['mode']=='CONDITIONAL':
                assert item['available'];p=np.argwhere(seed>.1);cx=float(p[:,1].mean());dx=max(8,int(seed.shape[1]*.12))*(-1 if cx>seed.shape[1]/2 else 1)
                for kind in ['zero_seed','moved_seed']:
                    xx=x.clone();s=np.zeros_like(seed) if kind=='zero_seed' else np.roll(seed,dx,axis=1)
                    if kind=='moved_seed':
                        if dx>0:s[:,:dx]=0
                        else:s[:,dx:]=0
                    xx[0,1,:975,:2048]=torch.from_numpy(s);aa=response(net,xx)
                    am={**meta,'intervention':kind,'seed_dx_model_px':dx if kind=='moved_seed' else None,
                        'seed_xy':[] if kind=='zero_seed' else (np.asarray(item['seed_xy'])+[2*dx,0]).tolist(),
                        'intervention_seed_nonempty':bool(s.sum()>0),'original_seed_xy':item['seed_xy'],
                        'scope':'Condition sensitivity only; shifted seed is not asserted to be a correct other rope.'}
                    files.extend(save_response(dest/'ablations'/kind,aa,am))
        print('evaluation predictions frozen model',variant,flush=True)
    dump(root/'prediction_receipt.json',dict(command=sys.argv,rows=rows,seconds=time.time()-tick,exit_code=0,labels_loaded=False))
    for p in [HERE/'evaluate_heatmaps.py',HERE/'evaluation_protocol_final.json']:
        assert p.exists();files.append(dict(path=str(p),sha256=sha(p)))
    dump(root/'freeze.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(HERE/'protocol.json'),selection_sha256=sha(HERE/'selection.json'),files=files,
        fixed_choice_before_evaluation=True,scored_before_freeze=False,models=choice['models'],source='Only model inference on image/seed inputs. No evaluation references used.'))

@torch.inference_mode()
def batch():
    choice,manifest=frozen_inputs();variant=choice['selected_variant'];model=choice['models'][variant]
    assert sha(model['path'])==model['sha256'];net=load_network(model['path']);cfg=config(net,variant)
    out=HERE/'batch_inference';out.mkdir(exist_ok=False);rows=[];tick=time.time()
    sources={str(HERE/n):sha(HERE/n) for n in ['infer_heatmaps.py','network.py','preprocessing.py','common.py']}
    dump(out/'run_start.json',dict(command=sys.argv,model=model,variant=variant,selection_sha256=sha(HERE/'selection.json'),sources=sources,input_manifest_sha256=sha(HERE/'inputs/manifest.json'),planned=292,conditions=301,device='cpu',threads=torch.get_num_threads()))
    with (out/'progress.jsonl').open('x',encoding='utf8') as log:
        for n,j in enumerate(manifest['batch'],1):
            assert sha(j['input_cache'])==j['input_cache_sha256']
            with np.load(j['input_cache'],allow_pickle=False) as z:gray=z['image'];seeds=z['seeds']
            entries=[]
            for k,s in enumerate(seeds):
                x,stats=prepare_input(gray,s,cfg);r=response(net,x);dest=out/'responses'/j['id']/f'entry_{k:02d}'
                meta=dict(id=j['id'],entry_index=k,available=True,seed_xy=j['seeds'][k]['xy'],mode='AUTO_SAVED_ENTRY',variant=variant,
                    model_path=model['path'],model_sha256=model['sha256'],image_sha256=j['image_sha256'],input_statistics=stats,
                    input_cache_sha256=j['input_cache_sha256'],labels_read=False)
                save_response(dest,r,meta);entries.append(dict(entry_index=k,seed_xy=j['seeds'][k]['xy'],response_file=meta['response_file'],response_sha256=meta['response_sha256']))
            row=dict(id=j['id'],entries=entries);rows.append(row);log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
            if n%12==0 or n==292:print(json.dumps(dict(done=n,total=292,seconds=round(time.time()-tick,1))),flush=True)
    assert len(rows)==292 and sum(len(r['entries']) for r in rows)==301
    dump(out/'manifest.json',dict(rows=rows,variant=variant,model=model,command=sys.argv,completed=292,entry_conditions=301,seconds=time.time()-tick,
        exit_code=0,sources_unchanged=all(sha(p)==v for p,v in sources.items()),model_unchanged=sha(model['path'])==model['sha256']))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['verify','evaluation','batch']);ap.add_argument('--threads',type=int,default=8);a=ap.parse_args()
    sys.stdout.reconfigure(encoding='utf8');torch.set_num_threads(a.threads);torch.use_deterministic_algorithms(True)
    {'verify':verify,'evaluation':evaluation,'batch':batch}[a.mode]()
