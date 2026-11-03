"""D0/D1 all-parameter U-Net continuation and genuine new inference, CPU only.

Only this version directory is written. Training never opens evaluation labels.
Inference reads images and frozen entry hints, never target masks/references.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key,'8')
import argparse,hashlib,json,random,sys,time,platform,copy,traceback
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent
BASE=HERE.parent;ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from train import tensor_sample,read_arrays,load_model,save_model,seed_all,state_sha,DEFAULTS
from model import stage_balanced_mse,balanced_mse

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def tsha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def configure(threads=8):
    torch.set_num_threads(threads);torch.use_deterministic_algorithms(True)
    if hasattr(torch,'set_num_interop_threads'):
        try:torch.set_num_interop_threads(1)
        except RuntimeError:pass
def protocol():return load(HERE/'protocol.json')
def training_index():
    p=HERE/'dataset/index.json';ix=load(p)
    assert ix.get('frozen') is True,'Freeze automatic labels and review before training'
    items=[i for i in ix['items'] if i['split']=='train']
    old=[i for i in items if i.get('origin')=='old'];new=[i for i in items if i.get('origin')=='new']
    assert len(old)==10 and new,'D1 requires actual approved new training images'
    assert len({i['id'] for i in items})==len(items)
    assert not {i['source_group'] for i in items}&set(protocol()['eval_groups'])
    return ix,old,new
def make_config():
    p=HERE/'training_config.json';pr=protocol();cp=torch.load(pr['start_model']['path'],map_location='cpu',weights_only=False)
    assert sha(pr['start_model']['path'])==pr['start_model']['sha256']
    cfg={**cp['config'],'seed':20261102,'threads':8,'batch_size':1,'learning_rate':.0002,
         'weight_decay':.0001,'train_steps':600,'train_max_seconds':2400,'checkpoint_interval':100,
         'smallcheck_steps':100,'smallcheck_max_seconds':300,
         'loss':'old stage_balanced_mse only; no independent B neighbor/background terms',
         'optimization_scope':'all135897 EntryUNet parameters trainable; no relation model',
         'sampling_D0':'uniform old10 images,600 updates',
         'sampling_D1':'even updates exact same old image/augmentation as D0; odd updates group-uniform new weak images',
         'weak_label_weight':1.,'checkpoint_selection':'largest common saved step, never evaluation-selected',
         'source_checkpoint':pr['start_model'],'source_checkpoint_update':750,
         'budget_rationale':'600 updates gives300 new-image updates under50/50mix; 2400s arm permits existingCPU fullROI training; fixed before evaluation',
         'normalization_and_augmentation':'unchanged train.tensor_sample; gamma on nonnegative gray before normalization'}
    if p.exists():assert load(p)==cfg,'Frozen training config changed'
    else:dump(p,cfg)
    return cfg
def arrays_for(items):
    out={}
    for i in items:
        assert '20260925' not in i['image_path']
        if i.get('cache_sha256'):assert sha(i['cache'])==i['cache_sha256']
        a=read_arrays(i);assert a['image'].shape==(975,2048)
        assert np.any(a['valid']&(a['target']>=.5))
        out[i['id']]=a
    return out
def new_net(cfg):
    seed_all(cfg['seed']);net=load_model(cfg['source_checkpoint']['path'])
    for p in net.parameters():p.requires_grad_(True)
    assert all(p.requires_grad for p in net.parameters())
    return net
@torch.inference_mode()
def fit_stats(net,items,arrays,cfg):
    net.eval();rows=[]
    for it in items:
        a=arrays[it['id']];x,t,v,st,meta=tensor_sample(a,cfg,False)
        pred=net(x);loss,detail=stage_balanced_mse(pred,t,v,st,.5)
        high=v&(t>=.5);low=v&(t<.1)
        vals=pred[v]
        stages={}
        for s in range(1,5):
            hh=high&(st==s);vv=v&(st==s)
            stages[str(s)]={'valid_pixels':int(vv.sum()),'high_pixels':int(hh.sum()),
                'high_response':float(pred[hh].mean()) if hh.any() else None}
        rows.append({'id':it['id'],'origin':it['origin'],'source_group':it['source_group'],'loss':float(loss),
            'high_response':float(pred[high].mean()),'low_response':float(pred[low].mean()) if low.any() else None,
            'prediction_min':float(vals.min()),'prediction_max':float(vals.max()),'stages':stages})
    return {'macro_loss':float(np.mean([r['loss'] for r in rows])),'rows':rows,
        'scope':'fit on old/new training supervision only; not independent accuracy'}
def changed_parameters(net,initial):
    rows=[]
    for name,v in net.state_dict().items():
        delta=(v-initial[name]).abs();rows.append({'name':name,'changed_elements':int((delta>0).sum()),
            'numel':v.numel(),'mean_abs_delta':float(delta.mean()),'max_abs_delta':float(delta.max())})
    return rows
def smallcheck():
    cfg=make_config();configure(cfg['threads']);ix,old,new=training_index()
    dest=HERE/'smallcheck';dest.mkdir(exist_ok=False)
    # Choose old and new with later visual-stage support; selection uses TRAIN labels only.
    arrays=arrays_for(old+new)
    def priority(it):
        a=arrays[it['id']]
        return -int(np.sum(a['valid']&(a['target']>=.5)&(a['stage']>=2))),it['id']
    chosen=sorted(old,key=priority)[:2]+sorted(new,key=priority)[:2]
    net=new_net(cfg);initial={k:v.clone() for k,v in net.state_dict().items()}
    before=fit_stats(net,chosen,arrays,cfg);optim=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    tick=time.perf_counter();logs=[];nonzero=set()
    with (dest/'log.jsonl').open('x',encoding='utf8') as f:
        for step in range(1,cfg['smallcheck_steps']+1):
            it=chosen[(step-1)%len(chosen)];x,t,v,st,meta=tensor_sample(arrays[it['id']],cfg,False)
            net.train();optim.zero_grad(set_to_none=True);pred=net(x);loss,detail=stage_balanced_mse(pred,t,v,st)
            assert torch.isfinite(loss);loss.backward()
            for n,p in net.named_parameters():
                if p.grad is not None and p.grad.abs().sum()>0:nonzero.add(n)
            optim.step();r={'step':step,'id':it['id'],'origin':it['origin'],'loss':float(loss.detach()),'seconds':time.perf_counter()-tick}
            logs.append(r);f.write(json.dumps(r)+'\n');f.flush()
            if step==1 or step%20==0:print(r,flush=True)
            if r['seconds']>=cfg['smallcheck_max_seconds']:break
    after=fit_stats(net,chosen,arrays,cfg);changed=changed_parameters(net,initial)
    passed=bool(after['macro_loss']<before['macro_loss']*.95 or after['macro_loss']<.02)
    receipt={'before':before,'after':after,'steps':step,'seconds':time.perf_counter()-tick,'passed':passed,
        'learning_chain_gate':'finite gradients and lower valid-target loss on old plus new training images; not identity accuracy',
        'nonzero_gradient_parameter_names':sorted(nonzero),'changed_parameters':changed,
        'independent_copy_discarded':True,'formal_training_must_reload_A750':True,
        'dataset_index_sha256':sha(HERE/'dataset/index.json'),'exit_code':0 if passed else 1}
    dump(dest/'receipt.json',receipt)
    assert passed,'Smallcheck did not fit; inspect input/labels/valid domains before formal training'
def plan(cfg,old,new):
    p=HERE/'sampling_plan.json';rng=np.random.default_rng(cfg['seed']);groups=sorted({i['source_group'] for i in new})
    by={g:[i for i in new if i['source_group']==g] for g in groups};rows=[]
    for step in range(1,cfg['train_steps']+1):
        oi=old[int(rng.integers(len(old)))];g=groups[int(rng.integers(len(groups)))];ni=by[g][int(rng.integers(len(by[g])))];d1=oi if step%2==0 else ni
        rows.append({'step':step,'D0':oi['id'],'D1':d1['id'],'augmentation_seed':cfg['seed']+13*step})
    if p.exists():assert load(p)==rows
    else:dump(p,rows)
    return rows
def train(branch):
    cfg=make_config();configure(cfg['threads']);ix,old,new=training_index()
    assert load(HERE/'smallcheck/receipt.json')['passed']
    assert load(HERE/'smallcheck/receipt.json')['dataset_index_sha256']==sha(HERE/'dataset/index.json')
    schedule=plan(cfg,old,new);items=old if branch=='D0' else old+new;lookup={i['id']:i for i in items};arrays=arrays_for(items)
    dest=HERE/'branches'/branch;dest.mkdir(parents=True,exist_ok=False);(dest/'checkpoints').mkdir()
    net=new_net(cfg);initial={k:v.clone() for k,v in net.state_dict().items()};start_hash=state_sha(net)
    save_model(dest/'initial.pt',net,cfg,0,{'status':'initial_A750_copy','branch':branch})
    dump(dest/'config.json',cfg);dump(dest/'environment.json',{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,
        'device':'cpu','cuda_available':torch.cuda.is_available(),'platform':platform.platform(),'command':sys.argv,
        'source_sha256':sha(__file__),'model_source_sha256':sha(BASE/'model.py'),'tensor_source_sha256':sha(BASE/'train.py'),
        'dataset_sha256':sha(HERE/'dataset/index.json'),'initial_state_sha256':start_hash,'all_parameters_trainable':True})
    diagnostic=[old[0]]+[sorted(new,key=lambda x:x['id'])[0]] if branch=='D1' else [old[0]]
    before=fit_stats(net,diagnostic,arrays,cfg);optim=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    seen=Counter();origins=Counter();groups=Counter();nonzero=set();tick=time.perf_counter();saved=[];logs=[]
    with (dest/'train_log.jsonl').open('x',encoding='utf8') as f:
        for rec in schedule:
            step=rec['step'];it=lookup[rec[branch]];torch.manual_seed(rec['augmentation_seed'])
            x,t,v,st,meta=tensor_sample(arrays[it['id']],cfg,True,random.Random(rec['augmentation_seed']))
            net.train();optim.zero_grad(set_to_none=True);pred=net(x);loss,detail=stage_balanced_mse(pred,t,v,st,cfg['high_threshold'])
            assert torch.isfinite(loss);loss.backward()
            grad={}
            for n,p in net.named_parameters():
                if p.grad is not None:
                    gn=float(p.grad.abs().sum());
                    if gn>0:nonzero.add(n)
                    if step==1 or it['origin']=='new' and origins['new']==0:grad[n]=gn
            optim.step();seen[it['id']]+=1;origins[it['origin']]+=1;groups[it['source_group']]+=1
            r={'step':step,'id':it['id'],'origin':it['origin'],'source_group':it['source_group'],
                'loss':float(loss.detach()),'high_mse':detail['high_mse'],'low_mse':detail['low_mse'],
                'active_stages':detail['active_positive_stages'],'seconds':time.perf_counter()-tick,
                'augmentation_seed':rec['augmentation_seed'],'input_sha256':tsha(x),'normalization':meta,
                'gradient_abs_sum_selected':grad,'valid_pixels':int(v.sum())}
            f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();logs.append(r)
            if step%cfg['checkpoint_interval']==0:
                target=dest/'checkpoints'/f'step_{step:04d}.pt';save_model(target,net,cfg,step,{'status':'completed','branch':branch});saved.append(step)
            if step==1 or step%50==0:print({k:r[k] for k in ['step','id','origin','loss','seconds']},flush=True)
            if r['seconds']>=cfg['train_max_seconds']:break
    train_seconds=time.perf_counter()-tick
    save_model(dest/'final.pt',net,cfg,step,{'status':'completed','branch':branch})
    after=fit_stats(net,diagnostic,arrays,cfg);changes=changed_parameters(net,initial)
    assert any(r['changed_elements'] for r in changes)
    if branch=='D1':assert origins['new']>0
    receipt={'exit_code':0,'branch':branch,'steps':step,'training_seconds':train_seconds,'checkpoint_steps':saved,
        'selected_by_evaluation':False,'initial_state_sha256':start_hash,'final_state_sha256':state_sha(net),
        'all_parameters_trainable':True,'parameter_count':sum(p.numel() for p in net.parameters()),
        'nonzero_gradient_parameter_names':sorted(nonzero),'changed_parameters':changes,
        'sampling_by_id':dict(seen),'sampling_by_origin':dict(origins),'sampling_by_source_group':dict(groups),
        'before_diagnostic_fit':before,'after_diagnostic_fit':after,'final_path':str(dest/'final.pt'),'final_sha256':sha(dest/'final.pt'),
        'dataset_index_sha256':sha(HERE/'dataset/index.json'),'config_sha256':sha(HERE/'training_config.json'),
        'sampling_plan_sha256':sha(HERE/'sampling_plan.json'),'formal_weights_reset_to_common_A750':True}
    dump(dest/'training_receipt.json',receipt)
def select():
    rr={b:load(HERE/'branches'/b/'training_receipt.json') for b in ['D0','D1']}
    assert rr['D0']['initial_state_sha256']==rr['D1']['initial_state_sha256']
    common=sorted(set(rr['D0']['checkpoint_steps'])&set(rr['D1']['checkpoint_steps']));assert common
    step=common[-1];paths={b:HERE/'branches'/b/'checkpoints'/f'step_{step:04d}.pt' for b in rr}
    # Actual shared old updates must have bit-identical augmented input tensors.
    logs={b:{r['step']:r for r in map(json.loads,(HERE/'branches'/b/'train_log.jsonl').read_text(encoding='utf8').splitlines())} for b in rr}
    shared=[]
    for s in range(2,step+1,2):
        a=logs['D0'][s];b=logs['D1'][s];assert a['id']==b['id'] and a['input_sha256']==b['input_sha256'];shared.append(s)
    p=HERE/'selected_checkpoints.json';assert not p.exists()
    dump(p,{'common_step':step,'paths':{b:{'path':str(p),'sha256':sha(p)} for b,p in paths.items()},
        'selection_rule':'largest common saved update; no evaluation targets or gallery heat inspected for selection',
        'shared_old_update_inputs_verified':len(shared),'D0_steps':rr['D0']['steps'],'D1_steps':rr['D1']['steps']})
def cv2_module():
    sys.path.append(str(ROOT/'vision_prototype/.venv/Lib/site-packages'))
    import cv2
    cv2.setNumThreads(1);return cv2
def input_from_raw(image_path,seed):
    cv2=cv2_module();bgr=cv2.imdecode(np.fromfile(image_path,np.uint8),cv2.IMREAD_COLOR)
    assert bgr.shape==(3000,4096,3)
    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
    image=cv2.resize(gray[1050:],(2048,975),interpolation=cv2.INTER_AREA)
    mask=np.zeros_like(image,np.uint8);q=np.rint((np.asarray(seed)-[0,1050])*.5).astype(np.int32)
    cv2.polylines(mask,[q],False,1,1,cv2.LINE_8);assert mask.any()
    dist=cv2.distanceTransform(1-mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    se=np.exp(-.5*(dist/2.)**2).astype(np.float32)
    return image,se
def input_tensor(image,se,cfg):
    zeros=np.zeros_like(image);arrays={'image':image,'seed':se,'target':zeros,'valid':zeros.astype(bool),'stage':zeros.astype(np.uint8),'roi':np.ones_like(image,bool)}
    x,_,_,_,meta=tensor_sample(arrays,cfg,False);return x,meta
def fixed_jobs():
    ix=load(BASE/'runs/run_20261030_002/data/index.json');jobs=[]
    for it in ix['items']:
        jobs.append({'id':it['id'],'mode':'CONDITIONAL','image_path':it['image_path'],'image_sha256':it['image_sha256'],
            'source_group':it['source_group'],'split':it['split'],'seed_xy':it['known_seed_xy'],'source_item':it})
        if it['split']=='eval':
            auto=load(it['auto_entry_path']);jobs.append({'id':it['id'],'mode':'AUTO','image_path':it['image_path'],'image_sha256':it['image_sha256'],
                'source_group':it['source_group'],'split':'eval','seed_xy':auto.get('free_seed') if auto['available'] else [],'source_item':it,
                'auto_entry_path':it['auto_entry_path']})
    ix,old,new=training_index()
    for it in new:jobs.append({'id':it['id'],'mode':'CONDITIONAL','image_path':it['image_path'],'image_sha256':it['image_sha256'],
        'source_group':it['source_group'],'split':'new_train','seed_xy':it['known_seed_xy'],'source_item':it})
    return jobs
def infer(which,branches):
    configure(8);selection=load(HERE/'selected_checkpoints.json');models={};cfg=make_config();start=time.perf_counter()
    for branch in branches:
        cp=selection['paths'][branch];assert sha(cp['path'])==cp['sha256'];models[branch]=load_model(cp['path'])
    if which=='gallery':
        jobs=[]
        for j in protocol()['gallery_jobs']:
            for k,s in enumerate(j['seeds']):jobs.append({'id':j['id'],'mode':'GALLERY_AUTO','entry_index':k,'image_path':j['image_path'],
                'image_sha256':j['image_sha256'],'source_group':j['source_group'],'seed_xy':s['xy'],'split':'display_gallery'})
    else:jobs=fixed_jobs()
    out=HERE/'inference'/which;out.mkdir(parents=True,exist_ok=True);rows=[];forwards=0
    with torch.inference_mode():
        for ji,job in enumerate(jobs,1):
            tick=time.perf_counter();assert sha(job['image_path'])==job['image_sha256']
            if job['seed_xy']:image,se=input_from_raw(job['image_path'],job['seed_xy']);x,meta=input_tensor(image,se,cfg)
            else:x=None;meta=None
            for branch,net in models.items():
                if which=='gallery':dest=out/branch/job['id']/f'entry_{job["entry_index"]:02d}.npz'
                else:dest=out/branch/job['mode']/job['id']/'response.npz'
                dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists(),'Preserve previous inference'
                if x is None:response=np.full((975,2048),np.nan,np.float32);status='entry_unavailable'
                else:
                    response=net(x)[0,0,:975,:2048].numpy().copy();forwards+=1;status='predicted'
                    assert response.dtype==np.float32 and np.isfinite(response).all() and 0<=response.min()<=response.max()<=1
                np.savez_compressed(dest,response=response)
                r={'id':job['id'],'mode':job['mode'],'split':job['split'],'source_group':job['source_group'],
                    'image_path':job['image_path'],'image_sha256':job['image_sha256'],'seed_xy':job['seed_xy'],
                    'entry_index':job.get('entry_index',0),'response_path':str(dest),'response_sha256':sha(dest),'status':status,
                    'model_path':selection['paths'][branch]['path'],'model_sha256':selection['paths'][branch]['sha256'],'model_step':selection['common_step'],
                    'branch':branch,'normalization':meta,'shape_hw':[975,2048],'bbox_original_xyxy':[0,1050,4096,3000],
                    'outside_search_domain':'unknown, not predicted','response_semantics':'original sigmoid center strength0..1; not identity probability',
                    'new_model_forward':x is not None,'target_labels_read':False,'seconds':time.perf_counter()-tick}
                if job.get('auto_entry_path'):r['entry_path']=job['auto_entry_path']
                dump(dest.with_suffix('.json'),r);rows.append(r)
            if ji==1 or ji%20==0 or ji==len(jobs):print({'inferred':ji,'total':len(jobs),'forwards':forwards,'seconds':round(time.perf_counter()-start,1)},flush=True)
    receipt=out/'receipt.json';assert not receipt.exists()
    dump(receipt,{'exit_code':0,'command':sys.argv,'rows':rows,'new_forward_count':forwards,'seconds':time.perf_counter()-start,
        'selected_checkpoints_sha256':sha(HERE/'selected_checkpoints.json'),'prediction_saved_before_evaluation':True,
        'evaluation_references_read':False,'source_sha256':sha(__file__)})
def main():
    sys.stdout.reconfigure(encoding='utf8');p=argparse.ArgumentParser();p.add_argument('mode',choices=['smallcheck','train','select','infer-fixed','infer-gallery'])
    p.add_argument('--branch',choices=['D0','D1']);p.add_argument('--branches',nargs='+',choices=['D0','D1'],default=['D0','D1']);a=p.parse_args()
    if a.mode=='smallcheck':smallcheck()
    elif a.mode=='train':assert a.branch;train(a.branch)
    elif a.mode=='select':select()
    elif a.mode=='infer-fixed':infer('fixed',a.branches)
    else:infer('gallery',[a.branch or 'D1'])
if __name__=='__main__':main()
