"""Bounded A/B continuation with synchronized inputs and grouped negative MSE.

Old model.py/train.py are imported without modification. Evaluation caches and
references are never loaded. Main comparison uses the largest shared saved step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import stage_balanced_mse
from train import (DEFAULTS as OLD_DEFAULTS, dump, load, load_model, read_arrays,
                   save_model, seed_all, sha, state_sha, tensor_sample as old_tensor_sample)


DEFAULTS = {**{k:v for k,v in OLD_DEFAULTS.items() if not k.startswith('overfit_')}, 'train_steps': 800, 'train_max_seconds': 1200,
            'smallcheck_steps': 60, 'smallcheck_max_seconds': 120,
            'checkpoint_interval': 50, 'lambda_neighbor': .5, 'lambda_background': .1,
            'optimizer_policy': 'identical fresh AdamW in A/B; source checkpoint has no optimizer',
            'loss': 'old stage-balanced center MSE + .5 mean neighbor p^2 + .1 mean background p^2',
            'budget_policy': 'independent smallcheck discarded; A then B, at most800 steps/1200s each; largest common saved step',
            'paired_augmentation': 'old gray/seed/target use identical RNG and geometry; new masks never alter RNG/fallback',
            'continuation_source': 'run_20261029_001/model/final.pt'}


def tensor_sha(tensor):
    a = tensor.detach().cpu().contiguous().numpy()
    h = hashlib.sha256(); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def pair_arrays(item):
    a = read_arrays(item)
    with np.load(item['supervision_cache'], allow_pickle=False) as z:
        a['center_target'] = np.asarray(z['center_target'], np.float32)
        a['center_valid'] = np.asarray(z['center_valid'], bool)
        a['center_stage'] = np.asarray(z['center_stage'], np.uint8) if 'center_stage' in z else a['stage'].copy()
        a['neighbor_mask'] = np.asarray(z['neighbor_mask'], bool)
        a['background_mask'] = np.asarray(z['background_mask'], bool)
    shape = a['image'].shape
    for k in ('center_target', 'center_valid', 'center_stage', 'neighbor_mask', 'background_mask'):
        assert a[k].shape == shape, (item['id'], k, 'shape mismatch')
    assert np.isfinite(a['center_target']).all() and 0 <= a['center_target'].min() <= a['center_target'].max() <= 1
    assert not np.any(a['center_valid'] & (~a['roi'] | (a['center_stage'] == 0)))
    assert not np.any(a['neighbor_mask'] & a['background_mask']), 'negative groups must be disjoint'
    for k in ('neighbor_mask', 'background_mask'):
        assert not np.any(a[k] & ~a['roi']), (item['id'], k, 'outside computed domain')
        assert not np.any(a[k] & a['center_valid'] & (a['center_target'] >= .5)), (item['id'], k, 'positive conflict')
    return a


def pair_tensor_sample(a, cfg, augment=False, rng=None):
    """Exact old augmentation order; every new field shares the same affine grid.

    Geometry acceptance depends only on OLD valid/entry; adding annotations can
    never change the sampled input or the random sequence of either branch.
    """
    rng = rng or random
    im = torch.from_numpy(a['image'].copy()); se = torch.from_numpy(a['seed'].copy())
    ot = torch.from_numpy(a['target'].copy()); ct = torch.from_numpy(a['center_target'].copy())
    ov = torch.from_numpy(a['valid']); roi = torch.from_numpy(a['roi'])
    os = torch.from_numpy(a['stage'].astype(np.float32))
    cv = torch.from_numpy(a['center_valid']); cs = torch.from_numpy(a['center_stage'].astype(np.float32))
    nm = torch.from_numpy(a['neighbor_mask']); bm = torch.from_numpy(a['background_mask'])
    fallback = False; affine = None
    if augment:
        theta = math.radians(rng.uniform(-cfg['rotation_degrees'], cfg['rotation_degrees']))
        flip = -1 if rng.random() < cfg['horizontal_flip_probability'] else 1
        h, w = im.shape; c, s = math.cos(theta), math.sin(theta)
        affine = torch.tensor([[flip*c, -s*h/w, rng.uniform(-1, 1)*2*cfg['translation_fraction']],
                               [flip*s*w/h, c, rng.uniform(-1, 1)*2*cfg['translation_fraction']]], dtype=torch.float32)[None]
        grid = F.affine_grid(affine, (1, 1, h, w), align_corners=False)
        smooth = F.grid_sample(torch.stack([im, se, ot, ct])[None], grid, mode='bilinear', padding_mode='zeros', align_corners=False)[0]
        masks = F.grid_sample(torch.stack([ov.float(), roi.float(), os, cv.float(), cs, nm.float(), bm.float()])[None],
                              grid, mode='nearest', padding_mode='zeros', align_corners=False)[0]
        if smooth[1].sum() > 1 and masks[0].sum() > 10:
            im, se, ot, ct = smooth
            ov, roi, os, cv, cs, nm, bm = masks
            ov=ov>.5; roi=roi>.5; cv=cv>.5; nm=nm>.5; bm=bm>.5
        else:
            fallback = True
        mean = im[roi].mean()
        im = ((im-mean)*rng.uniform(*cfg['contrast'])+mean).clamp(0, 1)
        im = (im.pow(rng.uniform(*cfg['gamma']))*rng.uniform(*cfg['brightness'])).clamp(0, 1)
        im = (im + torch.randn_like(im)*cfg['noise_std']).clamp(0, 1)
    mean = im[roi].mean(); std = im[roi].std(unbiased=False).clamp_min(cfg['normalization_std_floor'])
    im = (im-mean)/std; im[~roi] = 0
    ov=ov&roi; cv=cv&roi; nm=nm&roi; bm=bm&roi
    ot=ot.clamp(0, 1); ct=ct.clamp(0, 1); se=se.clamp(0, 1)
    h,w=im.shape; m=cfg['pad_multiple']; pad=(0,(-w)%m,0,(-h)%m)
    result={'x':F.pad(torch.stack([im,se]),pad)[None]}
    for key, value in [('old_target',ot),('old_valid',ov),('old_stage',os),
                       ('center_target',ct),('center_valid',cv),('center_stage',cs),
                       ('neighbor_mask',nm),('background_mask',bm),('roi',roi)]:
        result[key]=F.pad(value,pad)[None,None]
    result['old_stage']=result['old_stage'].to(torch.uint8)
    result['center_stage']=result['center_stage'].to(torch.uint8)
    result['meta']={'unpadded_shape':[h,w], 'mean':float(mean), 'std':float(std), 'geometry_fallback':fallback,
                    'seed_pixels_gt01':int((se>.1).sum()), 'affine':None if affine is None else affine.tolist()}
    return result


def zero_mse(pred, mask):
    """Group mean, not image-area weighting; empty groups have exact zero gradient."""
    mask=mask.bool(); count=mask.sum()
    return (pred.square()*mask).sum()/count.clamp_min(1)


def group_loss(pred, batch, variant, cfg):
    assert variant in ('A','B')
    if variant=='A':
        target, valid, stage = batch['old_target'], batch['old_valid'], batch['old_stage']
        neighbor=torch.zeros_like(batch['neighbor_mask']); background=torch.zeros_like(batch['background_mask'])
    else:
        target, valid, stage = batch['center_target'], batch['center_valid'], batch['center_stage']
        neighbor,background=batch['neighbor_mask'],batch['background_mask']
    # Padding/uncomputed input remains excluded even if a malformed caller sets a mask there.
    valid=valid&batch['roi']; neighbor=neighbor&batch['roi']; background=background&batch['roi']
    center, detail=stage_balanced_mse(pred,target,valid,stage,cfg['high_threshold'])
    ln=zero_mse(pred,neighbor); lb=zero_mse(pred,background)
    total=center+cfg['lambda_neighbor']*ln+cfg['lambda_background']*lb
    return total, {'center_loss':float(center.detach()), 'neighbor_loss':float(ln.detach()),
                   'background_loss':float(lb.detach()), 'neighbor_pixels':int(neighbor.sum()),
                   'background_pixels':int(background.sum()), 'center_pixels':int(valid.sum()),
                   'total_loss':float(total.detach()), **detail}


@torch.inference_mode()
def response_stats(pred, batch, quantiles=False):
    out={}
    masks={'center_high':batch['center_valid']&(batch['center_target']>=.5),
           'neighbor':batch['neighbor_mask'], 'background':batch['background_mask']}
    for s in range(1,5):masks[f'center_stage_{s}']=masks['center_high']&(batch['center_stage']==s)
    for key,mask in masks.items():
        mask=mask&batch['roi']; values=pred[mask]; n=values.numel()
        out[key]={'pixels':n,'mean':float(values.mean()) if n else None,
                  'fraction_ge_05':float((values>=.5).float().mean()) if n else None}
        if quantiles:out[key]['p95']=float(torch.quantile(values,.95)) if n else None
    return out


def shared_hashes(batch):
    return {k:tensor_sha(batch[k]) for k in ('x','old_target','old_valid','old_stage','roi')}


def numerical_checks(cfg):
    torch.set_num_threads(cfg['threads']); seed_all(cfg['seed'])
    p=torch.tensor([[[[.2,.8,.6,.9,.4,.7,.3,.5]]]],requires_grad=True)
    t=torch.tensor([[[[0.,1.,0.,1.,0.,0.,0.,0.]]]])
    v=torch.tensor([[[[1,1,1,1,0,0,0,0]]]],dtype=torch.bool)
    st=torch.tensor([[[[1,1,3,3,0,0,0,0]]]],dtype=torch.uint8)
    nm=torch.tensor([[[[0,0,0,0,1,0,0,0]]]],dtype=torch.bool)
    bm=torch.tensor([[[[0,0,0,0,0,1,0,0]]]],dtype=torch.bool)
    roi=torch.tensor([[[[1,1,1,1,1,1,1,0]]]],dtype=torch.bool)
    b={'old_target':t,'old_valid':v,'old_stage':st,'center_target':t,'center_valid':v,
       'center_stage':st,'neighbor_mask':nm,'background_mask':bm,'roi':roi}
    old,_=stage_balanced_mse(p,t,v,st); la,da=group_loss(p,b,'A',cfg)
    assert torch.equal(old,la), 'A must be exactly equivalent to old objective'
    oldg=torch.autograd.grad(old,p,retain_graph=True)[0]; ag=torch.autograd.grad(la,p,retain_graph=True)[0]
    assert torch.equal(oldg,ag)
    lb,db=group_loss(p,b,'B',cfg);lb.backward()
    expected=float(old.detach())+cfg['lambda_neighbor']*.16+cfg['lambda_background']*.49
    assert abs(float(lb.detach())-expected)<1e-7
    assert p.grad[...,6].item()==0 and p.grad[...,7].item()==0
    empty={**b,'neighbor_mask':torch.zeros_like(nm),'background_mask':torch.zeros_like(bm)}
    eb,_=group_loss(p.detach(),empty,'B',cfg);assert torch.equal(eb,old.detach())
    wide=torch.full((1,1,3,500),.7);assert abs(float(zero_mse(wide,torch.ones_like(wide,dtype=torch.bool)))-.49)<1e-6
    leaked={**b,'neighbor_mask':nm|~roi,'background_mask':bm|~roi}
    ll,_=group_loss(p.detach(),leaked,'B',cfg);assert torch.equal(ll,lb.detach())
    h,w=141,219;mark=np.zeros((h,w),np.float32);mark[45:60,55:90]=1
    a={'image':np.full((h,w),.3,np.float32)+mark*.5,'seed':mark,'target':mark,
       'valid':np.ones((h,w),bool),'roi':np.ones((h,w),bool),'stage':np.ones((h,w),np.uint8)*2,
       'center_target':mark.copy(),'center_valid':np.ones((h,w),bool),'center_stage':np.ones((h,w),np.uint8)*2,
       'neighbor_mask':mark.astype(bool),'background_mask':~mark.astype(bool)}
    for j in range(8):
        torch.manual_seed(j); oldsample=old_tensor_sample(a,cfg,True,random.Random(j))
        torch.manual_seed(j); ns=pair_tensor_sample(a,cfg,True,random.Random(j))
        for key,val in zip(('x','old_target','old_valid','old_stage'),oldsample[:4]):
            assert torch.equal(ns[key],val),(j,key,'old transform equivalence failed')
        assert torch.equal(ns['old_target'],ns['center_target']) and torch.equal(ns['x'][:,1:2],ns['center_target'])
        assert torch.equal(ns['old_valid'],ns['center_valid'])
        assert not torch.any(ns['neighbor_mask']&ns['background_mask'])
        assert torch.equal(ns['neighbor_mask']|ns['background_mask'],ns['roi'])
        assert not ns['roi'][...,-1,-1].any()
        changed={**a,'neighbor_mask':np.zeros_like(mark,bool),'background_mask':np.zeros_like(mark,bool)}
        torch.manual_seed(j); changed_sample=pair_tensor_sample(changed,cfg,True,random.Random(j))
        assert shared_hashes(ns)==shared_hashes(changed_sample)
    return {'A_loss_exact_old_equivalence':True,'A_gradient_exact_old_equivalence':True,
            'empty_negative_groups_exact_zero':True,'group_means_not_pixel_count_weighted':True,
            'ignore_gradient_zero':True,'padding_and_uncomputed_masks_excluded':True,
            'eight_transforms_old_input_target_exact_equivalence':True,'geometry_all_masks_synchronized':True,
            'new_mask_changes_do_not_change_input_or_old_target_hash':True,'photometry_nonnegative_before_gamma':True}


def read_training(index_path):
    idx=load(index_path);train=[dict(i) for i in idx['items'] if i['split']=='train']; assert len(train)==10
    tg={i['source_group'] for i in train};eg={i['source_group'] for i in idx['items'] if i['split']!='train'}
    assert len(tg)==4 and len(eg)==3 and not tg&eg
    for it in train:
        for key in ('cache','supervision_cache'):
            p=Path(it[key]);it[key]=str(p if p.is_absolute() else Path(index_path).resolve().parent/p)
    arrays={it['id']:pair_arrays(it) for it in train}
    return train,arrays,{'train_groups':sorted(tg),'eval_groups_metadata_only':sorted(eg),'eval_cache_or_labels_read':False}


@torch.inference_mode()
def evaluate_training(net,items,arrays,cfg):
    net.eval(); rows=[]
    for it in items:
        batch=pair_tensor_sample(arrays[it['id']],cfg);pred=net(batch['x'])
        _,da=group_loss(pred,batch,'A',cfg);_,db=group_loss(pred,batch,'B',cfg)
        rows.append({'id':it['id'],'A_objective':da,'B_objective':db,'response':response_stats(pred,batch,True)})
    return {'items':rows,'macro_A_center_loss':float(np.mean([r['A_objective']['center_loss'] for r in rows])),
            'macro_B_total_loss':float(np.mean([r['B_objective']['total_loss'] for r in rows])),
            'scope':'training annotation fit; engineered additional negatives are not independent identity reference'}


def prepare_run(args,cfg):
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(cfg['threads']);seed_all(cfg['seed'])
    items,arrays,split=read_training(args.index)
    net=load_model(args.checkpoint);assert net.base==cfg['base_channels']
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert 'optimizer' not in cp and 'optimizer_state' not in cp, 'record optimizer recovery policy if source differs'
    dump(out/'config.json',cfg)
    env={'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'numpy':np.__version__,
         'platform':platform.platform(),'device':'cpu','threads':cfg['threads'],'command':sys.argv,
         'source_checkpoint':str(Path(args.checkpoint).resolve()),'source_checkpoint_sha256':sha(args.checkpoint),
         'source_checkpoint_step':cp.get('step'),'initial_model_state_sha256':state_sha(net),
         'source_optimizer_absent':True,'optimizer':'fresh AdamW for all branches',
         'response_logging_scope':'same fixed B training annotation masks in both A/B; additional masks are diagnostics only and carry no gradient in A',
         'index_sha256':sha(args.index),'paired_train_source_sha256':sha(__file__),
         'old_model_source_sha256':sha(Path(__file__).with_name('model.py')),
         'old_train_source_sha256':sha(Path(__file__).with_name('train.py')),**split}
    dump(out/'environment.json',env)
    counts=[]
    for it in items:
        a=arrays[it['id']]
        counts.append({'id':it['id'],'old_cache_sha256':sha(it['cache']),'supervision_cache_sha256':sha(it['supervision_cache']),
                       'center_target_exact_old':np.array_equal(a['target'],a['center_target']),
                       'center_valid_exact_old':np.array_equal(a['valid'],a['center_valid']),
                       'center_stage_exact_old':np.array_equal(a['stage'],a['center_stage']),
                       'neighbor_pixels':int(a['neighbor_mask'].sum()),'background_pixels':int(a['background_mask'].sum()),
                       'old_center_pixels':int(a['valid'].sum()),'B_center_pixels':int(a['center_valid'].sum())})
    dump(out/'input_supervision_audit.json',counts)
    return out,items,arrays,net,env


def do_smallcheck(args,cfg):
    out,items,arrays,net,env=prepare_run(args,cfg)
    selected=sorted(items,key=lambda it:(-int(arrays[it['id']]['neighbor_mask'].any()),
                                         -int(arrays[it['id']]['background_mask'].any()),it['id']))[:4]
    dump(out/'selected_images.json',{'ids':[i['id'] for i in selected],'selection':'training-only additional-supervision availability, then id',
                                   'not_used_to_initialize_formal_training':True})
    initial=evaluate_training(net,selected,arrays,cfg)
    seed_all(cfg['seed']);optimizer=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    tick=time.perf_counter();actual=0
    with (out/'log.jsonl').open('w',encoding='utf-8') as f:
        for step in range(1,cfg['smallcheck_steps']+1):
            it=selected[(step-1)%len(selected)];batch=pair_tensor_sample(arrays[it['id']],cfg)
            net.train();optimizer.zero_grad(set_to_none=True);pred=net(batch['x']);loss,detail=group_loss(pred,batch,'B',cfg)
            assert torch.isfinite(loss);loss.backward();optimizer.step();actual=step
            row={'step':step,'id':it['id'],'elapsed_seconds':time.perf_counter()-tick,**detail,'response':response_stats(pred,batch)}
            f.write(json.dumps(row)+'\n');f.flush()
            if step==1 or step%10==0:print('smallcheck',step,detail['total_loss'],round(row['elapsed_seconds'],2),flush=True)
            if time.perf_counter()-tick>=cfg['smallcheck_max_seconds']:break
    duration=time.perf_counter()-tick;final=evaluate_training(net,selected,arrays,cfg)
    save_model(out/'smallcheck_only.pt',net,cfg,actual,{'not_formal_initialization':True})
    present={k:sum(arrays[i['id']][k].sum() for i in selected)>0 for k in ('neighbor_mask','background_mask')}
    result={'status':'completed','exit_code':0,'steps':actual,'training_seconds':duration,'present_negative_groups':{k:bool(v) for k,v in present.items()},
            'stop_reason':'step_budget' if actual==cfg['smallcheck_steps'] else 'time_budget','initial':initial,'final':final,
            'formal_initialization':'must reload original old final.pt, not smallcheck_only.pt','smallcheck_is_not_identity_validation':True}
    dump(out/'result.json',result);print('smallcheck_complete',json.dumps({'steps':actual,'training_seconds':duration}),flush=True)
    return 0


def do_train(args,cfg):
    out,items,arrays,net,env=prepare_run(args,cfg);variant=args.variant
    assert 'smallcheck' not in str(args.checkpoint).lower(), 'formal branches must start from original checkpoint'
    ckdir=out/'checkpoints';ckdir.mkdir()
    save_model(out/'branch_start.pt',net,cfg,0,{'status':'origin_checkpoint','variant':variant,'origin':env['source_checkpoint_sha256']})
    initial=evaluate_training(net,items,arrays,cfg)
    seed_all(cfg['seed']);rng=random.Random(cfg['seed'])
    schedule=[rng.randrange(len(items)) for _ in range(cfg['train_steps'])]
    dump(out/'schedule.json',{'seed':cfg['seed'],'ids':[i['id'] for i in items],'indices':schedule,
                              'augmentation_rng':'continues same local RNG after schedule; noise uses identically reset torch RNG'})
    optimizer=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    tick=time.perf_counter();actual=0;saved=[];fallback=0
    with (out/'train_log.jsonl').open('w',encoding='utf-8') as f:
        for step,j in enumerate(schedule,1):
            it=items[j];batch=pair_tensor_sample(arrays[it['id']],cfg,True,rng)
            hashes=shared_hashes(batch);fallback+=int(batch['meta']['geometry_fallback'])
            net.train();optimizer.zero_grad(set_to_none=True);pred=net(batch['x']);loss,detail=group_loss(pred,batch,variant,cfg)
            assert torch.isfinite(loss);loss.backward();optimizer.step();actual=step
            row={'step':step,'id':it['id'],'variant':variant,'elapsed_seconds':time.perf_counter()-tick,
                 'shared_hashes':hashes,'transform':batch['meta'],**detail,'response':response_stats(pred,batch)}
            f.write(json.dumps(row)+'\n');f.flush()
            if step==1 or step%50==0:print(variant,step,detail['total_loss'],detail['center_loss'],detail['neighbor_loss'],detail['background_loss'],round(row['elapsed_seconds'],2),flush=True)
            if step%cfg['checkpoint_interval']==0:
                save_model(ckdir/f'step_{step:04d}.pt',net,cfg,step,{'status':'completed','variant':variant});saved.append(step)
            if time.perf_counter()-tick>=cfg['train_max_seconds']:break
    duration=time.perf_counter()-tick
    if actual not in saved:
        save_model(ckdir/f'step_{actual:04d}.pt',net,cfg,actual,{'status':'completed','variant':variant,'budget_end':True});saved.append(actual)
    save_model(out/'final.pt',net,cfg,actual,{'status':'completed','variant':variant,'supplement_only_if_not_common_step':True})
    final=evaluate_training(net,items,arrays,cfg)
    result={'status':'completed','exit_code':0,'variant':variant,'steps':actual,'training_seconds':duration,
            'stop_reason':'step_budget' if actual==cfg['train_steps'] else 'time_budget','saved_steps':saved,
            'initial_model_state_sha256':env['initial_model_state_sha256'],'initial_checkpoint_sha256':env['source_checkpoint_sha256'],
            'initial_train':initial,'final_train':final,'geometry_fallback_count':fallback,'eval_used':False,
            'main_comparison':'largest exact saved step shared by A and B; each final is supplemental if different'}
    dump(out/'result.json',result);print('complete',json.dumps({k:v for k,v in result.items() if k not in ('initial_train','final_train')}),flush=True)
    return 0


def compare(args):
    dirs={'A':Path(args.a).resolve(),'B':Path(args.b).resolve()}
    results={k:load(p/'result.json') for k,p in dirs.items()};env={k:load(p/'environment.json') for k,p in dirs.items()}
    common=sorted(set(results['A']['saved_steps'])&set(results['B']['saved_steps']));assert common,'No common saved update step'
    step=max(common)
    assert env['A']['initial_model_state_sha256']==env['B']['initial_model_state_sha256']
    assert env['A']['source_checkpoint_sha256']==env['B']['source_checkpoint_sha256']
    for key in ('index_sha256','paired_train_source_sha256','old_model_source_sha256','old_train_source_sha256'):
        assert env['A'][key]==env['B'][key],key
    assert load(dirs['A']/'input_supervision_audit.json')==load(dirs['B']/'input_supervision_audit.json')
    assert load(dirs['A']/'config.json')==load(dirs['B']/'config.json')
    assert load(dirs['A']/'schedule.json')==load(dirs['B']/'schedule.json')
    logs={k:[json.loads(line) for line in (p/'train_log.jsonl').read_text().splitlines()] for k,p in dirs.items()}
    checked=min(len(logs['A']),len(logs['B']))
    for a,b in zip(logs['A'],logs['B']):
        assert a['step']==b['step'] and a['id']==b['id'] and a['shared_hashes']==b['shared_hashes'],a['step']
    output={'status':'paired_comparison_frozen','exit_code':0,'main_comparison_step':step,'common_saved_steps':common,
            'all_overlapping_input_augmentation_hashes_equal':True,'verified_shared_steps':checked,
            'same_origin_checkpoint':env['A']['source_checkpoint_sha256'],'same_initial_state':env['A']['initial_model_state_sha256'],
            'same_config_schedule_optimizer_policy':True,'formal_budget':{'A':results['A']['steps'],'B':results['B']['steps']},
            'main_checkpoints':{k:{'path':str(p/'checkpoints'/f'step_{step:04d}.pt'),'sha256':sha(p/'checkpoints'/f'step_{step:04d}.pt')} for k,p in dirs.items()},
            'selection_policy':'maximum common saved update count only; no response, score, or evaluation label used'}
    dump(args.out,output);print(json.dumps(output,ensure_ascii=False));return 0


def main():
    parser=argparse.ArgumentParser();subs=parser.add_subparsers(dest='mode',required=True)
    check=subs.add_parser('check');check.add_argument('--out',type=Path,required=True)
    for name in ('smallcheck','train'):
        p=subs.add_parser(name);p.add_argument('--index',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
        p.add_argument('--out',type=Path,required=True);p.add_argument('--config',type=Path)
        if name=='train':p.add_argument('--variant',choices=['A','B'],required=True)
    p=subs.add_parser('compare');p.add_argument('--a',type=Path,required=True);p.add_argument('--b',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();cfg={**DEFAULTS,**(load(args.config) if hasattr(args,'config') and args.config else {})}
    assert cfg['train_steps']<=800 and cfg['train_max_seconds']<=1200 and cfg['smallcheck_steps']<=60 and cfg['smallcheck_max_seconds']<=120
    assert cfg['batch_size']==1 and cfg['threads']==8 and cfg['checkpoint_interval']==50
    if args.mode=='check':
        assert not args.out.exists();result=numerical_checks(cfg);args.out.parent.mkdir(exist_ok=True,parents=True)
        dump(args.out,{'exit_code':0,'command':sys.argv,'source_sha256':sha(__file__),'config':cfg,**result});print(json.dumps(result));return 0
    if args.mode=='compare':return compare(args)
    if args.mode=='smallcheck':return do_smallcheck(args,cfg)
    return do_train(args,cfg)


if __name__=='__main__':raise SystemExit(main())
