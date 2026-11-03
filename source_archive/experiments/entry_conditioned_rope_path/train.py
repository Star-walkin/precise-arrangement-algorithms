"""Bounded full-search-ROI training; reads only training caches, never eval labels."""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, json, math, platform, random, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from model import EntryUNet,balanced_mse,stage_balanced_mse,checks

DEFAULTS={'seed':20261029,'base_channels':8,'threads':8,'batch_size':1,
 'overfit_steps':120,'overfit_max_seconds':180,'overfit_max_final_mse':.025,'overfit_min_reduction':.65,
 'train_steps':1000,'train_max_seconds':1200,'learning_rate':.001,'weight_decay':.0001,
 'high_threshold':.5,'normalization_std_floor':.08,'pad_multiple':32,
 'brightness':[.9,1.1],'contrast':[.9,1.1],'gamma':[.9,1.1],'noise_std':.004,
 'horizontal_flip_probability':.5,'rotation_degrees':5.,'translation_fraction':.035,
 'seed_source':'initial free entry only, never post-contact target points',
 'model_input':'gray plus initial seed; seed-conditioned global descriptor and relative coordinates inside model',
 'normalization':'per-ROI mean/std floor .08 after nonnegative photometry; neutral exterior zero standardized',
 'loss':'single valid-only weighted MSE: equal stages with positive targets; each phase equal high/low mass; overall MSE diagnostic only',
 'sampling':'uniform original training image; all full search ROIs include seed',
 'budget_policy':'one overfit gate, reset to saved initial weights, one fixed baseline; no eval-driven model selection'}

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
def state_sha(net):
    h=hashlib.sha256()
    for key,v in sorted(net.state_dict().items()):h.update(key.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def save_model(path,net,cfg,step,extra=None):
    torch.save({'model_state':net.state_dict(),'base_channels':cfg['base_channels'],'config':cfg,'step':step,'extra':extra},path)
def load_model(path):
    cp=torch.load(path,map_location='cpu',weights_only=False);net=EntryUNet(cp['base_channels']);net.load_state_dict(cp['model_state']);net.eval();return net

def read_arrays(item):
    with np.load(item['cache'],allow_pickle=False) as z:a={k:z[k] for k in z.files}
    for key in ['image','seed','target','valid','stage','roi']:assert key in a,(item['id'],key)
    image,seed,target=[np.asarray(a[k],np.float32) for k in ['image','seed','target']]
    valid,roi=np.asarray(a['valid'],bool),np.asarray(a['roi'],bool);stage=np.asarray(a['stage'],np.uint8)
    assert image.ndim==2 and all(v.shape==image.shape for v in [seed,target,valid,roi,stage])
    assert np.isfinite(image).all() and 0<=image.min() and image.max()<=1
    assert np.isfinite(seed).all() and 0<=seed.min() and seed.max()<=1 and seed.sum()>0
    assert np.isfinite(target).all() and 0<=target.min() and target.max()<=1
    assert roi.any() and not np.any(valid&~roi) and stage.max()<=4
    assert not np.any(valid&(stage==0)),(item['id'],'all valid high/low pixels must have a phase1..4')
    return dict(image=image,seed=seed,target=target,valid=valid,roi=roi,stage=stage)

def tensor_sample(a,cfg,augment=False,rng=None):
    rng=rng or random
    im=torch.from_numpy(a['image'].copy());se=torch.from_numpy(a['seed'].copy());target=torch.from_numpy(a['target'].copy())
    valid=torch.from_numpy(a['valid']);roi=torch.from_numpy(a['roi']);stage=torch.from_numpy(a['stage'].astype(np.float32))
    geometry_fallback=False
    if augment:
        theta=math.radians(rng.uniform(-cfg['rotation_degrees'],cfg['rotation_degrees']))
        flip=-1 if rng.random()<cfg['horizontal_flip_probability'] else 1
        h,w=im.shape;c,s=math.cos(theta),math.sin(theta)
        # Rotation in pixel coordinates; normalized grid compensates aspect ratio.
        affine=torch.tensor([[flip*c,-s*h/w,rng.uniform(-1,1)*2*cfg['translation_fraction']],
            [flip*s*w/h,c,rng.uniform(-1,1)*2*cfg['translation_fraction']]],dtype=torch.float32)[None]
        grid=F.affine_grid(affine,(1,1,h,w),align_corners=False)
        smooth=F.grid_sample(torch.stack([im,se,target])[None],grid,mode='bilinear',padding_mode='zeros',align_corners=False)[0]
        masks=F.grid_sample(torch.stack([valid.float(),roi.float(),stage])[None],grid,mode='nearest',padding_mode='zeros',align_corners=False)[0]
        if smooth[1].sum()>1 and masks[0].sum()>10:
            im,se,target=smooth;valid=masks[0]>.5;roi=masks[1]>.5;stage=masks[2]
        else:geometry_fallback=True
        mean=im[roi].mean();im=((im-mean)*rng.uniform(*cfg['contrast'])+mean).clamp(0,1)
        im=(im.pow(rng.uniform(*cfg['gamma']))*rng.uniform(*cfg['brightness'])).clamp(0,1)
        im=(im+torch.randn_like(im)*cfg['noise_std']).clamp(0,1)
    mean=im[roi].mean();std=im[roi].std(unbiased=False).clamp_min(cfg['normalization_std_floor'])
    im=(im-mean)/std;im[~roi]=0
    valid=valid&roi;target=target.clamp(0,1);se=se.clamp(0,1)
    h,w=im.shape;m=cfg['pad_multiple'];pad=(0,(-w)%m,0,(-h)%m)
    x=F.pad(torch.stack([im,se]),pad)[None]
    t=F.pad(target,pad)[None,None];v=F.pad(valid,pad)[None,None]
    st=F.pad(stage,pad)[None,None].to(torch.uint8)
    return x,t,v,st,{'unpadded_shape':[h,w],'mean':float(mean),'std':float(std),'geometry_fallback':geometry_fallback,
        'seed_pixels_gt01':int((se>.1).sum())}

def transform_checks(cfg):
    image=np.full((141,219),.3,np.float32);mark=np.zeros_like(image);mark[45:60,55:90]=1
    a={'image':image+.5*mark,'seed':mark,'target':mark,'valid':np.ones_like(mark,bool),
       'roi':np.ones_like(mark,bool),'stage':np.ones_like(mark,np.uint8)*2}
    for j in range(8):
        x,t,v,st,meta=tensor_sample(a,cfg,True,random.Random(j))
        assert torch.equal(x[:,1:2],t),'seed/target geometry differs'
        assert x[:,1].sum()>1 and int((v&(st!=2)).sum())==0
        assert torch.isfinite(x).all() and not v[...,-1,-1].any()
    return {'synchronized_geometry_seed_target':True,'eight_nonempty_augmented_seeds':True,
        'nearest_stage_and_valid_alignment':True,'padding_unsupervised':True,
        'photometric_gamma_applied_to_nonnegative_gray_only':True}

@torch.inference_mode()
def evaluate(net,items,arrays,cfg):
    net.eval();rows=[]
    for it in items:
        x,t,v,st,meta=tensor_sample(arrays[it['id']],cfg)
        pred=net(x);loss,detail=stage_balanced_mse(pred,t,v,st,cfg['high_threshold'])
        stages={}
        for s in range(1,5):
            vv=v&(st==s);ls,dd=balanced_mse(pred,t,vv,cfg['high_threshold'])
            stages[str(s)]={'valid_pixels':int(vv.sum()),'balanced_mse':float(ls) if vv.any() else None,**dd}
        vals=pred[v]
        rows.append({'id':it['id'],'stage_balanced_mse':float(loss),'balanced_mse':detail['overall_balanced_mse'],**detail,'stages':stages,
            'prediction_quantiles':np.quantile(vals.numpy(),[0,.01,.5,.99,1]).tolist() if vals.numel() else None,
            'prediction_std':float(vals.std(unbiased=False)) if vals.numel() else None})
    return {'macro_image_balanced_mse':float(np.mean([r['balanced_mse'] for r in rows])),
      'macro_image_stage_balanced_mse':float(np.mean([r['stage_balanced_mse'] for r in rows])),'items':rows,
      'scope':'fit to supplied conservative labels only; no independent full-arc identity accuracy'}

@torch.inference_mode()
def condition_diagnostic(net,it,a,cfg,dest):
    x,t,v,st,meta=tensor_sample(a,cfg);h,w=a['image'].shape
    positions=torch.nonzero(x[0,1]>.1);cy=float(positions[:,0].float().mean());cx=float(positions[:,1].float().mean())
    # Fixed image-relative translation, not derived from target or scored alternatives.
    shift=max(8,int(w*.12))*(-1 if cx>w/2 else 1)
    moved=x.clone();moved[:,1]=torch.roll(moved[:,1],shift,-1)
    if shift>0:moved[:,1,:,:shift]=0
    else:moved[:,1,:,shift:]=0
    zero=x.clone();zero[:,1]=0
    outputs={'normal':net(x)[0,0,:h,:w].numpy(),'zero_seed':net(zero)[0,0,:h,:w].numpy(),
        'moved_seed':net(moved)[0,0,:h,:w].numpy()}
    yy,xx=np.indices((h,w));far=((xx-cx)**2+(yy-cy)**2)>128**2
    far&=a['roi'];result={'id':it['id'],'moved_seed_dx_model_px':shift,'moved_seed_nonempty':bool(moved[:,1].sum()>0),
        'far_definition':'more than128 model pixels from initial seed centroid, within search ROI',
        'scope':'input-condition sensitivity only; moved seed is not asserted to identify another rope',
        'differences':{}}
    for key,arr in outputs.items():np.save(dest/(it['id']+'__'+key+'.npy'),arr.astype(np.float32))
    for key in ['zero_seed','moved_seed']:
        delta=np.abs(outputs[key]-outputs['normal'])
        result['differences'][key]={'mean_ROI':float(delta[a['roi']].mean()),'mean_far':float(delta[far].mean()) if far.any() else None,
            'max_far':float(delta[far].max()) if far.any() else None}
    return result

def run(index_path,out,cfg):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(cfg['threads']);seed_all(cfg['seed'])
    index=load(index_path);allitems=index['items']
    train=[dict(i) for i in allitems if i['split']=='train'];assert train
    tg={i['source_group'] for i in train};eg={i['source_group'] for i in allitems if i['split']!='train'}
    assert not tg&eg,'source group split leakage'
    for i in train:
        p=Path(i['cache']);i['cache']=str(p if p.is_absolute() else Path(index_path).resolve().parent/p)
    dump(out/'config.json',cfg);dump(out/'checks.json',{**checks(cfg['base_channels']),**transform_checks(cfg)})
    arrays={i['id']:read_arrays(i) for i in train}
    alignment=[]
    for i in train:
        a=arrays[i['id']];alignment.append({'id':i['id'],'shape':list(a['image'].shape),'cache_sha256':sha(i['cache']),
            'seed_sum':float(a['seed'].sum()),'roi_pixels':int(a['roi'].sum()),
            'stage_counts':{str(s):{'valid':int((a['valid']&(a['stage']==s)).sum()),
                'high':int((a['valid']&(a['stage']==s)&(a['target']>=.5)).sum()),
                'low':int((a['valid']&(a['stage']==s)&(a['target']<.5)).sum())} for s in range(1,5)}})
    dump(out/'alignment.json',alignment)
    seed_all(cfg['seed']);net=EntryUNet(cfg['base_channels']);initial={k:v.clone() for k,v in net.state_dict().items()}
    initial_state_hash=state_sha(net)
    save_model(out/'initial.pt',net,cfg,0)
    dump(out/'environment.json',{'python':sys.version,'executable':sys.executable,'torch':torch.__version__,
        'numpy':np.__version__,'platform':platform.platform(),'device':'cpu','cuda_available':torch.cuda.is_available(),
        'threads':cfg['threads'],'command':sys.argv,'index_sha256':sha(index_path),'model_source_sha256':sha(Path(__file__).with_name('model.py')),
        'train_source_sha256':sha(__file__),'train_source_groups':sorted(tg),'nontrain_source_groups_metadata_only':sorted(eg),
        'eval_cache_or_labels_read':False,'initial_state_sha256':initial_state_hash})
    # 4 fixed training ROIs, prioritize available transition + attached support.
    ranked=sorted(train,key=lambda i:(-int(np.any(arrays[i['id']]['valid']&(arrays[i['id']]['stage']==3)&(arrays[i['id']]['target']>=.5))),i['id']))
    overfit=ranked[:min(4,len(ranked))]
    stages=sorted({s for i in overfit for s in range(1,5) if np.any(arrays[i['id']]['valid']&(arrays[i['id']]['stage']==s)&(arrays[i['id']]['target']>=.5))})
    dump(out/'overfit_images.json',{'ids':[i['id'] for i in overfit],'stages':stages,
        'has_transition_and_attached':2 in stages and 3 in stages,'seed_input':'entry only; no target prefix'})
    initial_of=evaluate(net,overfit,arrays,cfg)
    optimizer=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    # One real training step is also a bounded performance probe; reset it.
    x,t,v,st,meta=tensor_sample(arrays[overfit[0]['id']],cfg);tick=time.perf_counter()
    optimizer.zero_grad(set_to_none=True);pr=net(x);loss,_=stage_balanced_mse(pr,t,v,st);loss.backward();optimizer.step()
    duration=time.perf_counter()-tick
    dump(out/'performance_probe.json',{'seconds_one_actual_train_step':duration,'shape':list(x.shape),
        'reset_before_overfit':True,'note':'timing diagnostic only; architecture/config not changed'})
    seed_all(cfg['seed']);net.load_state_dict(initial)
    optimizer=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    tick=time.perf_counter();actual=0
    with (out/'overfit_log.jsonl').open('w',encoding='utf-8') as f:
        for step in range(1,cfg['overfit_steps']+1):
            it=overfit[(step-1)%len(overfit)];x,t,v,st,meta=tensor_sample(arrays[it['id']],cfg)
            net.train();optimizer.zero_grad(set_to_none=True);pred=net(x);loss,detail=stage_balanced_mse(pred,t,v,st,cfg['high_threshold'])
            assert torch.isfinite(loss);loss.backward();optimizer.step();actual=step
            row={'step':step,'id':it['id'],'loss':float(loss.detach()),'elapsed_seconds':time.perf_counter()-tick,**detail}
            f.write(json.dumps(row)+'\n');f.flush()
            if step==1 or step%20==0:print('overfit',step,row['loss'],round(row['elapsed_seconds'],2),flush=True)
            if time.perf_counter()-tick>=cfg['overfit_max_seconds']:break
    ofduration=time.perf_counter()-tick;final_of=evaluate(net,overfit,arrays,cfg)
    reduction=1-final_of['macro_image_stage_balanced_mse']/max(1e-12,initial_of['macro_image_stage_balanced_mse'])
    fit_gate=final_of['macro_image_stage_balanced_mse']<=cfg['overfit_max_final_mse'] and reduction>=cfg['overfit_min_reduction']
    has_stages=2 in stages and 3 in stages;gate=fit_gate and has_stages
    of={'initial':initial_of,'final':final_of,'reduction':reduction,'passed':gate,'steps':actual,
        'training_seconds':ofduration,'has_transition_and_attached':has_stages,'numerical_fit_gate':fit_gate,
        'gate_metric':'macro_image_stage_balanced_mse'}
    dump(out/'overfit_result.json',of);save_model(out/'overfit.pt',net,cfg,actual,of)
    if not gate:
        dump(out/'result.json',{'status':'overfit_missing_transition_or_attached_baseline_skipped' if not has_stages else 'overfit_failed_baseline_skipped',
            'exit_code':2,'overfit':of,'scope':'stage-specific feasibility only; full baseline gate not passed'});return 2
    seed_all(cfg['seed']);net.load_state_dict(initial)
    optimizer=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    initial_metric=evaluate(net,train,arrays,cfg)
    rng=random.Random(cfg['seed']);schedule=[rng.randrange(len(train)) for _ in range(cfg['train_steps'])]
    dump(out/'schedule.json',{'seed':cfg['seed'],'train_ids':[i['id'] for i in train],'indices':schedule,
        'sampling':cfg['sampling'],'not_used_for_model_selection':True})
    tick=time.perf_counter();actual=0;fallback=0
    with (out/'train_log.jsonl').open('w',encoding='utf-8') as f:
        for step,j in enumerate(schedule,1):
            it=train[j];x,t,v,st,meta=tensor_sample(arrays[it['id']],cfg,True,rng)
            fallback+=int(meta['geometry_fallback'])
            net.train();optimizer.zero_grad(set_to_none=True);pred=net(x);loss,detail=stage_balanced_mse(pred,t,v,st,cfg['high_threshold'])
            assert torch.isfinite(loss);loss.backward();optimizer.step();actual=step
            row={'step':step,'id':it['id'],'loss':float(loss.detach()),'elapsed_seconds':time.perf_counter()-tick,
                'geometry_fallback':meta['geometry_fallback'],'seed_pixels_gt01':meta['seed_pixels_gt01'],
                'stage_valid_pixels':{str(s):int((v&(st==s)).sum()) for s in range(1,5)},**detail}
            f.write(json.dumps(row)+'\n');f.flush()
            if step==1 or step%50==0:print('train',step,row['loss'],round(row['elapsed_seconds'],2),flush=True)
            if time.perf_counter()-tick>=cfg['train_max_seconds']:break
    duration=time.perf_counter()-tick;final_metric=evaluate(net,train,arrays,cfg)
    result={'status':'completed','exit_code':0,'steps':actual,'training_seconds':duration,
        'stop_reason':'step_budget' if actual==cfg['train_steps'] else 'time_budget','overfit':of,
        'initial_train':initial_metric,'final_train':final_metric,'train_images':len(train),'source_groups':len(tg),
        'parameters':sum(p.numel() for p in net.parameters()),'initial_state_sha256':initial_state_hash,
        'initial_weights_file_sha256':sha(out/'initial.pt'),
        'geometry_fallback_count':fallback,'eval_used':False,'scope':'partial conservative labels; full target identity requires independent evidence'}
    save_model(out/'final.pt',net,cfg,actual,result);dump(out/'result.json',result)
    dest=out/'condition_diagnostics';dest.mkdir()
    diagnostics=[condition_diagnostic(net,i,arrays[i['id']],cfg,dest) for i in overfit]
    dump(dest/'summary.json',{'items':diagnostics,'condition_source':'initial entry only','ground_truth_used_to_move_seed':False})
    print('complete',json.dumps({k:v for k,v in result.items() if k not in ['overfit','initial_train','final_train']}),flush=True)
    return 0

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    c=sub.add_parser('check');c.add_argument('--out',type=Path)
    r=sub.add_parser('train');r.add_argument('--index',type=Path,required=True);r.add_argument('--out',type=Path,required=True);r.add_argument('--config',type=Path)
    args=p.parse_args()
    if args.mode=='check':
        torch.set_num_threads(8);result={**checks(),**transform_checks(DEFAULTS)};print(json.dumps(result))
        if args.out:assert not args.out.exists();dump(args.out,result)
    else:
        cfg={**DEFAULTS,**(load(args.config) if args.config else {})}
        assert cfg['train_steps']<=1000 and cfg['train_max_seconds']<=1200 and cfg['batch_size']==1
        raise SystemExit(run(args.index,args.out,cfg))
