"""H7 bounded matched C/D continuations; only the ten frozen training caches.

No H6 paths, evaluation pixels, new pseudo-labels, or acquisition interfaces.
The existing labels/ignore and equal-stage equal-high/low MSE are unchanged.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.append(str(ROOT))
from train import DEFAULTS as OLD_DEFAULTS, dump, load, seed_all, sha, state_sha
from paired_train import read_training, group_loss, response_stats, tensor_sha, numerical_checks as legacy_checks
from network import load_network, save_network, legacy
from preprocessing import DEFAULTS as INPUT_DEFAULTS, input_from_tensors, prepare_input, weighted_lcn

DEFAULTS={**OLD_DEFAULTS,**INPUT_DEFAULTS,'learning_rate':.0002,
          'train_steps':400,'train_max_seconds':720,'checkpoint_interval':50,
          'smallcheck_steps':60,'smallcheck_max_seconds':120,
          'lambda_neighbor':.10,'lambda_background':.05,
          'model_input':'raw_z, entry seed at index1, auxiliary LCN (D) or exact zero (C)',
          'loss':'unchanged equal-stage high/low1:1 MSE + .10 neighbor mean p2 + .05 explicit background mean p2',
          'optimizer_policy':'same B750 weights, fresh AdamW per arm; no optimizer exists in source',
          'smallcheck_gate':'finite; total objective decreases; high-center mean >= initial-.025 and >.4; transition/attached positives; both negative groups present',
          'selection_policy':'largest shared saved step; training-only same total objective and protocol guards',
          'budget_policy':'smallcheck60/120s discarded, each C/D400 updates or720s; no extension',
          'sampling':'same uniform original-image schedule; identical RNG/augmentation; all full-search ROIs contain entry'}


def sample(a,cfg,augment=False,rng=None):
    """Keep legacy geometry/photometry/RNG; LCN is deterministic after photometry."""
    rng=rng or random
    im=torch.from_numpy(a['image'].copy());se=torch.from_numpy(a['seed'].copy())
    ot=torch.from_numpy(a['target'].copy());ct=torch.from_numpy(a['center_target'].copy())
    ov=torch.from_numpy(a['valid']);roi=torch.from_numpy(a['roi'])
    os=torch.from_numpy(a['stage'].astype(np.float32));cv=torch.from_numpy(a['center_valid'])
    cs=torch.from_numpy(a['center_stage'].astype(np.float32))
    nm=torch.from_numpy(a['neighbor_mask']);bm=torch.from_numpy(a['background_mask'])
    fallback=False;affine=None
    if augment:
        theta=math.radians(rng.uniform(-cfg['rotation_degrees'],cfg['rotation_degrees']))
        flip=-1 if rng.random()<cfg['horizontal_flip_probability'] else 1
        h,w=im.shape;c,s=math.cos(theta),math.sin(theta)
        affine=torch.tensor([[flip*c,-s*h/w,rng.uniform(-1,1)*2*cfg['translation_fraction']],
                             [flip*s*w/h,c,rng.uniform(-1,1)*2*cfg['translation_fraction']]],dtype=torch.float32)[None]
        grid=F.affine_grid(affine,(1,1,h,w),align_corners=False)
        smooth=F.grid_sample(torch.stack([im,se,ot,ct])[None],grid,mode='bilinear',padding_mode='zeros',align_corners=False)[0]
        masks=F.grid_sample(torch.stack([ov.float(),roi.float(),os,cv.float(),cs,nm.float(),bm.float()])[None],
                            grid,mode='nearest',padding_mode='zeros',align_corners=False)[0]
        if smooth[1].sum()>1 and masks[0].sum()>10:
            im,se,ot,ct=smooth;ov,roi,os,cv,cs,nm,bm=masks
            ov=ov>.5;roi=roi>.5;cv=cv>.5;nm=nm>.5;bm=bm>.5
        else:fallback=True
        mean=im[roi].mean()
        im=((im-mean)*rng.uniform(*cfg['contrast'])+mean).clamp(0,1)
        im=(im.pow(rng.uniform(*cfg['gamma']))*rng.uniform(*cfg['brightness'])).clamp(0,1)
        im=(im+torch.randn_like(im)*cfg['noise_std']).clamp(0,1)
    x,meta=input_from_tensors(im,se,roi,cfg)
    ov=ov&roi;cv=cv&roi;nm=nm&roi;bm=bm&roi
    pad=tuple(meta['padding']);result={'x':x}
    for key,value in [('old_target',ot.clamp(0,1)),('old_valid',ov),('old_stage',os),
                      ('center_target',ct.clamp(0,1)),('center_valid',cv),('center_stage',cs),
                      ('neighbor_mask',nm),('background_mask',bm),('roi',roi)]:
        result[key]=F.pad(value,pad)[None,None]
    result['old_stage']=result['old_stage'].to(torch.uint8);result['center_stage']=result['center_stage'].to(torch.uint8)
    result['meta']={**meta,'geometry_fallback':fallback,'seed_pixels_gt01':int((se>.1).sum()),
                    'affine':None if affine is None else affine.tolist()}
    return result


def shared_hashes(b):
    out={k:tensor_sha(b[k]) for k in ('center_target','center_valid','center_stage','neighbor_mask','background_mask','roi')}
    out['raw_and_seed']=tensor_sha(b['x'][:,:2])
    return out


@torch.inference_mode()
def evaluate_training(net,items,arrays,cfg):
    net.eval();rows=[]
    for it in items:
        b=sample(arrays[it['id']],cfg);p=net(b['x']);_,detail=group_loss(p,b,'B',cfg)
        rows.append({'id':it['id'],'objective':detail,'response':response_stats(p,b,True),'input':b['meta']})
    result={'items':rows,'macro_total_loss':float(np.mean([r['objective']['total_loss'] for r in rows])),
            'macro_center_loss':float(np.mean([r['objective']['center_loss'] for r in rows])),
            'scope':'fixed training labels; not independent identity accuracy'}
    responses={}
    for k in rows[0]['response']:
        values=[r['response'][k]['mean'] for r in rows if r['response'][k]['pixels']]
        responses[k]={'macro_mean':float(np.mean(values)) if values else None,
                      'images':len(values),'pixels':sum(r['response'][k]['pixels'] for r in rows)}
    result['response_summary']=responses
    return result


def cfg_for(variant):return {**DEFAULTS,'variant':variant}


def prepare_run(out,variant):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    cfg=cfg_for(variant);torch.set_num_threads(cfg['threads']);seed_all(cfg['seed'])
    protocol=load(HERE/'protocol.json')
    index=protocol['legacy_index']['path'];checkpoint=protocol['source_B_model']['path']
    assert sha(index)==protocol['legacy_index']['sha256']
    assert sha(checkpoint)==protocol['source_B_model']['sha256']
    items,arrays,split=read_training(index)
    assert [i['id'] for i in items]==protocol['train_ids']
    expected={i['id']:i for i in protocol['supervision_sources']}
    audited=[]
    for it in items:
        e=expected[it['id']]
        # Resolve fixed frozen cache hashes before use; no eval arrays are read.
        assert sha(it['cache'])==e['cache_sha256']
        assert sha(it['supervision_cache'])==e['supervision_cache_sha256']
        a=arrays[it['id']]
        assert np.array_equal(a['target'],a['center_target'])
        assert np.array_equal(a['valid'],a['center_valid'])
        assert np.array_equal(a['stage'],a['center_stage'])
        audited.append({'id':it['id'],'cache_sha256':sha(it['cache']),
                        'supervision_sha256':sha(it['supervision_cache']),
                        'center_target_valid_stage_exact_legacy':True})
    net=load_network(checkpoint)
    dump(out/'config.json',cfg);dump(out/'input_audit.json',audited)
    env={'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'numpy':np.__version__,
         'platform':platform.platform(),'device':'cpu','threads':cfg['threads'],'command':sys.argv,
         'source_checkpoint':checkpoint,'source_checkpoint_sha256':sha(checkpoint),
         'source_checkpoint_step':net.checkpoint_step,'initial_state_sha256':state_sha(net),
         'protocol_sha256':sha(HERE/'protocol.json'),'optimizer':'fresh AdamW, identical lr and schedule',
         'source_sha256':{p:sha(HERE/p) for p in ['network.py','preprocessing.py','train_heatmap.py']},**split}
    dump(out/'environment.json',env)
    return out,cfg,items,arrays,net,env


def run_smallcheck(out):
    out,cfg,items,arrays,net,env=prepare_run(out,'D')
    chosen=sorted(items,key=lambda i:(-int(arrays[i['id']]['neighbor_mask'].any()),i['id']))[:4]
    dump(out/'selected_training_images.json',{'ids':[i['id'] for i in chosen],
         'rule':'training-only neighbor-mask availability, then lexical id; fixed input, no augmentation',
         'smallcheck_weights_discarded':True})
    initial=evaluate_training(net,chosen,arrays,cfg)
    seed_all(cfg['seed']);opt=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    tick=time.perf_counter();step=0
    with (out/'log.jsonl').open('w',encoding='utf8') as f:
        for step in range(1,cfg['smallcheck_steps']+1):
            it=chosen[(step-1)%len(chosen)];b=sample(arrays[it['id']],cfg)
            net.train();opt.zero_grad(set_to_none=True);p=net(b['x']);loss,detail=group_loss(p,b,'B',cfg)
            assert torch.isfinite(loss);loss.backward();opt.step()
            row={'step':step,'id':it['id'],'elapsed_seconds':time.perf_counter()-tick,**detail,'response':response_stats(p,b)}
            f.write(json.dumps(row)+'\n');f.flush()
            if step==1 or step%10==0:print('smallcheck',step,detail['total_loss'],round(row['elapsed_seconds'],2),flush=True)
            if time.perf_counter()-tick>=cfg['smallcheck_max_seconds']:break
    elapsed=time.perf_counter()-tick;final=evaluate_training(net,chosen,arrays,cfg)
    old=initial['response_summary'];new=final['response_summary']
    gates={'loss_decreased':final['macro_total_loss']<initial['macro_total_loss'],
           'high_center_not_cleared':new['center_high']['macro_mean']>.4 and new['center_high']['macro_mean']>=old['center_high']['macro_mean']-.025,
           'transition_and_attached_supervised':new['center_stage_2']['pixels']>0 and new['center_stage_3']['pixels']>0,
           'negative_groups_present':new['neighbor']['pixels']>0 and new['background']['pixels']>0,
           'finite':bool(np.isfinite(final['macro_total_loss']))}
    passed=all(gates.values())
    save_network(out/'smallcheck_only.pt',net,cfg,step,{'status':'completed','not_formal_initialization':True})
    result={'status':'passed' if passed else 'failed','exit_code':0 if passed else 2,
            'steps':step,'training_seconds':elapsed,'gate':gates,'initial_train':initial,'final_train':final,
            'stop_reason':'step_budget' if step==cfg['smallcheck_steps'] else 'time_budget',
            'scope':'optimization chain only, no identity validation; formal arms reload B750'}
    dump(out/'result.json',result)
    print('smallcheck_done',json.dumps({k:v for k,v in result.items() if k not in ('initial_train','final_train')}),flush=True)
    return result['exit_code']


def run_train(out,variant):
    assert load(HERE/'smallcheck'/'result.json')['status']=='passed','smallcheck gate not passed'
    assert load(HERE/'numerical_checks.json')['status']=='passed','numerical checks not passed'
    out,cfg,items,arrays,net,env=prepare_run(out,variant)
    assert env['source_sha256']==load(HERE/'numerical_checks.json')['source_sha256'],'source changed after checks'
    ck=out/'checkpoints';ck.mkdir()
    save_network(out/'branch_start.pt',net,cfg,0,{'status':'origin_checkpoint','origin':env['source_checkpoint_sha256']})
    initial=evaluate_training(net,items,arrays,cfg)
    seed_all(cfg['seed']);rng=random.Random(cfg['seed'])
    schedule=[rng.randrange(len(items)) for _ in range(cfg['train_steps'])]
    dump(out/'schedule.json',{'indices':schedule,'ids':[i['id'] for i in items],'seed':cfg['seed']})
    opt=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    tick=time.perf_counter();step=0;saved=[]
    with (out/'train_log.jsonl').open('w',encoding='utf8') as f:
        for step,j in enumerate(schedule,1):
            it=items[j];b=sample(arrays[it['id']],cfg,True,rng);hashes=shared_hashes(b)
            net.train();opt.zero_grad(set_to_none=True);p=net(b['x']);loss,detail=group_loss(p,b,'B',cfg)
            assert torch.isfinite(loss);loss.backward();opt.step()
            row={'step':step,'id':it['id'],'variant':variant,'elapsed_seconds':time.perf_counter()-tick,
                 'shared_hashes':hashes,'input':b['meta'],**detail,'response':response_stats(p,b)}
            f.write(json.dumps(row)+'\n');f.flush()
            if step==1 or step%50==0:print(variant,step,detail['total_loss'],detail['center_loss'],round(row['elapsed_seconds'],2),flush=True)
            if step%cfg['checkpoint_interval']==0:
                save_network(ck/f'step_{step:04d}.pt',net,cfg,step,{'status':'completed','variant':variant});saved.append(step)
            if time.perf_counter()-tick>=cfg['train_max_seconds']:break
    elapsed=time.perf_counter()-tick
    if step not in saved:
        save_network(ck/f'step_{step:04d}.pt',net,cfg,step,{'status':'completed','variant':variant,'budget_end':True});saved.append(step)
    save_network(out/'final.pt',net,cfg,step,{'status':'completed','variant':variant,'supplement_if_not_common_step':True})
    final=evaluate_training(net,items,arrays,cfg)
    result={'status':'completed','exit_code':0,'variant':variant,'steps':step,'training_seconds':elapsed,
            'saved_steps':saved,'stop_reason':'step_budget' if step==cfg['train_steps'] else 'time_budget',
            'initial_train':initial,'final_train':final,'eval_pixels_or_labels_used':False,
            'budget_includes':'augmentation, LCN, tensor hashes, optimization, logs/checkpoints; initial/final diagnostic forward passes separate',
            'atomic_step_time_overrun_possible':True}
    dump(out/'result.json',result)
    print('train_done',json.dumps({k:v for k,v in result.items() if k not in ('initial_train','final_train')}),flush=True)
    return 0


def compare_and_select():
    dirs={v:HERE/'branches'/v/'model' for v in ('C','D')}
    results={v:load(d/'result.json') for v,d in dirs.items()}
    env={v:load(d/'environment.json') for v,d in dirs.items()}
    step=max(set(results['C']['saved_steps'])&set(results['D']['saved_steps']))
    for key in ('initial_state_sha256','source_checkpoint_sha256','source_sha256','protocol_sha256'):
        assert env['C'][key]==env['D'][key],key
    assert load(dirs['C']/'schedule.json')==load(dirs['D']/'schedule.json')
    logs={v:[json.loads(x) for x in (d/'train_log.jsonl').read_text().splitlines()] for v,d in dirs.items()}
    for c,d in zip(logs['C'],logs['D']):
        assert c['step']==d['step'] and c['id']==d['id'] and c['shared_hashes']==d['shared_hashes'],c['step']
    p=load(HERE/'protocol.json');items,arrays,_=read_training(p['legacy_index']['path'])
    torch.set_num_threads(DEFAULTS['threads'])
    baseline=evaluate_training(load_network(p['source_B_model']['path']),items,arrays,cfg_for('C'))
    fits={};checkpoints={};guards={}
    for v,path in dirs.items():
        cp=path/'checkpoints'/f'step_{step:04d}.pt'
        fits[v]=evaluate_training(load_network(cp),items,arrays,cfg_for(v))
        checkpoints[v]={'path':str(cp),'sha256':sha(cp),'variant':v,'step':step}
        before=baseline['response_summary'];after=fits[v]['response_summary']
        stages={str(s):after[f'center_stage_{s}']['macro_mean']>=before[f'center_stage_{s}']['macro_mean']-.025
                for s in range(1,5) if before[f'center_stage_{s}']['images']}
        guards[v]={'positive_stage_mean_preserved':stages,
                   'background':after['background']['macro_mean']<=max(.04,before['background']['macro_mean']+.02),
                   'neighbor':after['neighbor']['macro_mean']<=max(.05,before['neighbor']['macro_mean']+.02)}
        guards[v]['passed']=all(stages.values()) and guards[v]['background'] and guards[v]['neighbor']
    eligible=[v for v in dirs if guards[v]['passed']]
    selected=min(eligible or list(dirs),key=lambda v:fits[v]['macro_total_loss'])
    result={'status':'frozen_training_only_selection','exit_code':0,'main_comparison_step':step,
            'shared_steps_checked':min(map(len,logs.values())),'shared_input_target_masks_hashes_exact':True,
            'same_origin_and_initial_state':True,'main_checkpoints':checkpoints,'guards':guards,
            'selected_variant':selected,'selected_checkpoint':checkpoints[selected],
            'accepted_by_training_guards':bool(eligible),'replace_established_B':False,
            'scores':{v:fits[v]['macro_total_loss'] for v in fits},
            'selection_scope':'training10 only; not independent quality/identity acceptance',
            'evaluation_or_gallery_used_for_choice':False}
    dump(HERE/'common_training_fit.json',{'step':step,'B':baseline,**fits})
    dump(HERE/'common_checkpoints.json',{'step':step,'models':{
         'B':{**p['source_B_model'],'variant':'B','step':750},**checkpoints},
         'training_fit_path':str(HERE/'common_training_fit.json'),
         'training_selection_diagnostics_path':str(HERE/'paired_comparison.json')})
    dump(HERE/'paired_comparison.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['smallcheck','train','compare'])
    parser.add_argument('--variant',choices=['C','D']);args=parser.parse_args()
    if args.mode=='smallcheck':return run_smallcheck(HERE/'smallcheck')
    if args.mode=='train':
        assert args.variant is not None
        return run_train(HERE/'branches'/args.variant/'model',args.variant)
    return compare_and_select()


if __name__=='__main__':raise SystemExit(main())
