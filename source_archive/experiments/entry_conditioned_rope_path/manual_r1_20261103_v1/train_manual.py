"""Bounded all-parameter U-Net continuation on new manual target/neighbor lines.

Uses the existing two-channel EntryUNet and stage-balanced MSE unchanged.
No evaluation reference is loaded by training or inference. Spatial R1 features
are captured from the NEW U-Net in the same forward as each saved response.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key, '8')
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
from train import tensor_sample, read_arrays, load_model, save_model, seed_all, state_sha, DEFAULTS
from model import stage_balanced_mse

BRANCH = 'U1'

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            value.update(block)
    return value.hexdigest()

def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def configure(threads=8):
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

def protocol():
    return load(HERE/'protocol.json')

def training_index():
    index = load(HERE/'dataset/index.json')
    assert index.get('frozen') is True, 'Freeze manual source files and caches first'
    items = [i for i in index['items'] if i['split'] == 'train']
    old = [i for i in items if i.get('origin') == 'old']
    manual = [i for i in items if i.get('origin') == 'manual']
    assert len(old) == 10 and manual
    assert len({i['id'] for i in items}) == len(items)
    assert not {i['source_group'] for i in items} & set(protocol()['eval_groups'])
    return index, old, manual

def make_config():
    p = protocol()
    start = p['start_model']
    assert sha(start['path']) == start['sha256']
    checkpoint = torch.load(start['path'], map_location='cpu', weights_only=False)
    original = checkpoint['config']
    # Inherit only active input/architecture settings, not historical budget or
    # B lambda fields that the current single-MSE implementation never calls.
    keys = ('base_channels', 'high_threshold', 'normalization_std_floor', 'pad_multiple',
            'brightness', 'contrast', 'gamma', 'noise_std', 'horizontal_flip_probability',
            'rotation_degrees', 'translation_fraction', 'seed_source', 'model_input', 'normalization')
    cfg = {key: original.get(key, DEFAULTS[key]) for key in keys}
    cfg.update(seed=20261103, threads=8, batch_size=1, learning_rate=.0002,
               weight_decay=.0001, train_steps=800, train_max_seconds=2000,
               checkpoint_interval=100, smallcheck_steps=100, smallcheck_max_seconds=300,
               loss='unchanged stage_balanced_mse, threshold0.5; no separate negative or confidence loss',
               optimization_scope='all EntryUNet parameters; R1 is separate',
               sampling='alternating manual/old 50:50; old uniform shuffled image cycles; manual uniform shuffled group cycles and shuffled image cycles within group',
               checkpoint_selection='last actual saved formal update, independent of evaluation/gallery appearance',
               source_checkpoint=start, source_checkpoint_update=int(checkpoint.get('step',750)),
               center_sigma_original_px=3., input_crop_original_xyxy=[0,1050,4096,3000],
               model_grid_hw=[975,2048], normalization_and_augmentation='unchanged tensor_sample; nonnegative photometry before per-ROI standardization',
               manual_LCN='display-only annotation transform; not used as network input',
               budget_rationale='800 updates incl400 manual exposures with a fixed2000s formal cap; one100-step independent smallcheck then A750 reset; no evaluation-driven extension')
    dest = HERE/'training_config.json'
    if dest.exists():
        assert load(dest) == cfg, 'Frozen effective config changed'
    else:
        dump(dest,cfg)
    return cfg

def arrays_for(items):
    arrays = {}
    for item in items:
        assert '20260925' not in item['image_path']
        assert sha(item['image_path']) == item['image_sha256']
        if item.get('cache_sha256'):
            assert sha(item['cache']) == item['cache_sha256']
        a = read_arrays(item)
        assert a['image'].shape == (975,2048)
        assert np.any(a['valid'] & (a['target'] >= .5)), item['id']
        if item['origin'] == 'manual':
            # Optional explicit red-line map is only a diagnostic. No new loss.
            with np.load(item['cache'], allow_pickle=False) as z:
                key = next((k for k in ('negative','manual_negative','neighbor_negative') if k in z.files),None)
                red = np.asarray(z[key],bool) if key else (a['valid'] & (a['target'] == 0))
            assert red.shape == a['image'].shape
            assert not np.any(red & ~a['valid'])
            assert not np.any(red & (a['target'] > 1e-7))
            a['negative'] = red
        arrays[item['id']] = a
    return arrays

def new_net(cfg):
    seed_all(cfg['seed'])
    net = load_model(cfg['source_checkpoint']['path'])
    for parameter in net.parameters():
        parameter.requires_grad_(True)
    assert all(p.requires_grad for p in net.parameters())
    return net

def response_metrics(pred,target,valid,red=None):
    high = valid & (target >= .5)
    low = valid & (target < .1)
    values = pred[valid]
    row = dict(valid_pixels=int(valid.sum()),high_pixels=int(high.sum()),low_pixels=int(low.sum()),
               high_response=float(pred[high].mean()) if high.any() else None,
               low_response=float(pred[low].mean()) if low.any() else None,
               prediction_min=float(values.min()) if values.numel() else None,
               prediction_max=float(values.max()) if values.numel() else None,
               prediction_std=float(values.std(unbiased=False)) if values.numel() else None)
    if red is not None:
        row.update(manual_red_pixels=int(red.sum()),
                   manual_red_response=float(pred[red].mean()) if red.any() else None,
                   manual_red_mse=float(pred[red].square().mean()) if red.any() else None)
        row['manual_target_minus_red_response']=(row['high_response']-row['manual_red_response']
                                                if high.any() and red.any() else None)
    return row

@torch.inference_mode()
def fit_stats(net,items,arrays,cfg):
    net.eval()
    rows = []
    for item in items:
        a = arrays[item['id']]
        x,target,valid,stage,meta = tensor_sample(a,cfg,False)
        pred = net(x)
        loss,detail = stage_balanced_mse(pred,target,valid,stage,cfg['high_threshold'])
        red = None
        if item['origin']=='manual':
            red = torch.from_numpy(a['negative'])[None,None]
            red = F.pad(red,(0,pred.shape[-1]-red.shape[-1],0,pred.shape[-2]-red.shape[-2]))
            red &= valid
        metrics = response_metrics(pred,target,valid,red)
        stages = {}
        for s in range(1,5):
            vv=valid & (stage==s)
            stages[str(s)] = response_metrics(pred,target,vv,red & vv if red is not None else None)
        rows.append(dict(id=item['id'],origin=item['origin'],source_group=item['source_group'],
                         loss=float(loss),loss_detail=detail,stages=stages,**metrics))
    by_origin={}
    for origin in sorted({r['origin'] for r in rows}):
        group=[r for r in rows if r['origin']==origin]
        by_origin[origin]=dict(images=len(group),macro_loss=float(np.mean([r['loss'] for r in group])))
    return dict(macro_loss=float(np.mean([r['loss'] for r in rows])),by_origin=by_origin,rows=rows,
                scope='fit on supplied old/manual training supervision; not independent target identity accuracy')

def changed_parameters(net,initial):
    rows=[]
    for name,value in net.state_dict().items():
        delta=(value-initial[name]).abs()
        rows.append(dict(name=name,numel=value.numel(),changed_elements=int((delta>0).sum()),
                         mean_abs_delta=float(delta.mean()),max_abs_delta=float(delta.max())))
    return rows

def choose_smallcheck(old,manual,arrays):
    # Choice is from training labels only. Include both views, high red coverage,
    # and existing later visual support without pretending manual stage1=contact.
    old_rank=sorted(old,key=lambda i:(-int(np.sum(arrays[i['id']]['valid']&(arrays[i['id']]['stage']>=2))),i['id']))
    chosen=old_rank[:2]
    for side in ('left','right'):
        pool=[i for i in manual if i['id'].endswith('__'+side)]
        if not pool:pool=[i for i in manual if i not in chosen]
        chosen.append(max(pool,key=lambda i:(int(arrays[i['id']]['negative'].sum()),i['id'])))
    assert len({i['id'] for i in chosen})==4
    return chosen

def smallcheck():
    cfg=make_config();configure(cfg['threads'])
    index,old,manual=training_index();arrays=arrays_for(old+manual)
    dest=HERE/'smallcheck';dest.mkdir(exist_ok=False)
    chosen=choose_smallcheck(old,manual,arrays)
    dump(dest/'selected_images.json',dict(ids=[i['id'] for i in chosen],selection='training-only old later-stage support and left/right manual red coverage'))
    net=new_net(cfg);initial={k:v.clone() for k,v in net.state_dict().items()}
    initial_hash=state_sha(net);before=fit_stats(net,chosen,arrays,cfg)
    optimizer=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    tick=time.perf_counter();nonzero=set()
    with (dest/'log.jsonl').open('x',encoding='utf-8') as f:
        for step in range(1,cfg['smallcheck_steps']+1):
            item=chosen[(step-1)%len(chosen)]
            x,target,valid,stage,meta=tensor_sample(arrays[item['id']],cfg,False)
            net.train();optimizer.zero_grad(set_to_none=True)
            pred=net(x);loss,detail=stage_balanced_mse(pred,target,valid,stage,cfg['high_threshold'])
            assert torch.isfinite(loss);loss.backward()
            for name,p in net.named_parameters():
                if p.grad is not None and p.grad.abs().sum()>0:nonzero.add(name)
            optimizer.step()
            red=valid & (target<=1e-8) if item['origin']=='manual' else None
            row=dict(step=step,id=item['id'],origin=item['origin'],loss=float(loss.detach()),
                     seconds=time.perf_counter()-tick,**response_metrics(pred.detach(),target,valid,red))
            f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
            if step==1 or step%20==0:print({k:row[k] for k in ('step','id','loss','seconds')},flush=True)
            if row['seconds']>=cfg['smallcheck_max_seconds']:break
    after=fit_stats(net,chosen,arrays,cfg)
    passed=bool(after['macro_loss']<before['macro_loss']*.95 or after['macro_loss']<.02)
    dump(dest/'receipt.json',dict(exit_code=0 if passed else 1,passed=passed,steps=step,
         seconds=time.perf_counter()-tick,before=before,after=after,
         initial_state_sha256=initial_hash,changed_parameters=changed_parameters(net,initial),
         nonzero_gradient_parameter_names=sorted(nonzero),independent_copy_discarded=True,
         formal_training_must_reload_A750=True,dataset_index_sha256=sha(HERE/'dataset/index.json'),
         training_config_sha256=sha(HERE/'training_config.json'),source_sha256=sha(__file__),
         gate='finite gradients and lower stage-balanced valid-target loss on old plus both-view manual center/neighbor samples; not full identity accuracy'))
    assert passed,'Smallcheck failed: inspect labels/inputs/ignore before formal training'

def make_plan(cfg,old,manual):
    rng=random.Random(cfg['seed'])
    groups=sorted({i['source_group'] for i in manual})
    by_group={g:sorted([i['id'] for i in manual if i['source_group']==g]) for g in groups}
    decks={};rows=[]
    def take(key,values):
        if not decks.get(key):
            decks[key]=list(values);rng.shuffle(decks[key])
        return decks[key].pop()
    for step in range(1,cfg['train_steps']+1):
        if step%2:
            group=take('group',groups);ident=take('manual/'+group,by_group[group]);origin='manual'
        else:
            ident=take('old',[i['id'] for i in old]);origin='old'
        rows.append(dict(step=step,id=ident,origin=origin,augmentation_seed=cfg['seed']+13*step))
    dest=HERE/'sampling_plan.json'
    if dest.exists():assert load(dest)==rows
    else:dump(dest,rows)
    return rows

def train():
    cfg=make_config();configure(cfg['threads'])
    index,old,manual=training_index()
    gate=load(HERE/'smallcheck/receipt.json')
    assert gate['passed'] and gate['dataset_index_sha256']==sha(HERE/'dataset/index.json')
    assert gate['training_config_sha256']==sha(HERE/'training_config.json')
    items=old+manual;arrays=arrays_for(items);lookup={i['id']:i for i in items}
    schedule=make_plan(cfg,old,manual)
    dest=HERE/'model';dest.mkdir(exist_ok=False);(dest/'checkpoints').mkdir()
    net=new_net(cfg);initial={k:v.clone() for k,v in net.state_dict().items()};initial_hash=state_sha(net)
    assert initial_hash==gate['initial_state_sha256'],'Formal run must reset after smallcheck'
    save_model(dest/'initial.pt',net,cfg,0,{'status':'fresh_A750_formal_start','branch':BRANCH})
    dump(dest/'config.json',cfg)
    dump(dest/'environment.json',dict(python=sys.version,executable=sys.executable,torch=torch.__version__,
        numpy=np.__version__,platform=platform.platform(),device='cpu',cuda_available=torch.cuda.is_available(),
        threads=cfg['threads'],command=sys.argv,source_sha256=sha(__file__),
        model_source_sha256=sha(BASE/'model.py'),tensor_source_sha256=sha(BASE/'train.py'),
        dataset_index_sha256=sha(HERE/'dataset/index.json'),initial_state_sha256=initial_hash,
        all_parameters_trainable=True,evaluation_references_read=False))
    before=fit_stats(net,items,arrays,cfg)
    dump(dest/'fit_before.json',before)
    optimizer=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    counts=Counter();origins=Counter();groups=Counter();nonzero=set();saved=[];tick=time.perf_counter()
    first_manual=True
    with (dest/'train_log.jsonl').open('x',encoding='utf-8') as f:
        for record in schedule:
            step=record['step'];item=lookup[record['id']];torch.manual_seed(record['augmentation_seed'])
            x,target,valid,stage,meta=tensor_sample(arrays[item['id']],cfg,True,random.Random(record['augmentation_seed']))
            net.train();optimizer.zero_grad(set_to_none=True);pred=net(x)
            loss,detail=stage_balanced_mse(pred,target,valid,stage,cfg['high_threshold'])
            assert torch.isfinite(loss);loss.backward()
            gradient={};total_gradient=0.
            capture_gradient=step==1 or item['origin']=='manual' and first_manual
            for name,p in net.named_parameters():
                if p.grad is None:continue
                norm=float(p.grad.abs().sum());total_gradient+=norm
                if norm>0:nonzero.add(name)
                if capture_gradient:gradient[name]=norm
            assert total_gradient>0
            optimizer.step()
            counts[item['id']]+=1;origins[item['origin']]+=1;groups[item['source_group']]+=1
            red=valid & (target<=1e-8) if item['origin']=='manual' else None
            metrics=response_metrics(pred.detach(),target,valid,red)
            row=dict(step=step,id=item['id'],origin=item['origin'],source_group=item['source_group'],
                     loss=float(loss.detach()),high_mse=detail['high_mse'],low_mse=detail['low_mse'],
                     active_stages=detail['active_positive_stages'],seconds=time.perf_counter()-tick,
                     augmentation_seed=record['augmentation_seed'],input_sha256=tensor_sha(x),normalization=meta,
                     gradient_abs_sum=total_gradient,gradient_abs_sum_selected=gradient,
                     red_diagnostic='manual valid target<=1e-8 after synchronized geometry; diagnostic only',**metrics)
            f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
            if item['origin']=='manual':first_manual=False
            if step%cfg['checkpoint_interval']==0:
                save_model(dest/'checkpoints'/f'step_{step:04d}.pt',net,cfg,step,{'status':'formal_completed_update','branch':BRANCH})
                saved.append(step)
            if step==1 or step%50==0:print({k:row[k] for k in ('step','id','origin','loss','seconds')},flush=True)
            if row['seconds']>=cfg['train_max_seconds']:break
    seconds=time.perf_counter()-tick
    if step not in saved:
        save_model(dest/'checkpoints'/f'step_{step:04d}.pt',net,cfg,step,{'status':'formal_budget_final','branch':BRANCH})
        saved.append(step)
    save_model(dest/'final.pt',net,cfg,step,{'status':'completed','branch':BRANCH})
    after=fit_stats(net,items,arrays,cfg);dump(dest/'fit_after.json',after)
    changes=changed_parameters(net,initial)
    assert all(r['changed_elements']>0 for r in changes),'Some network parameter tensors never changed'
    assert origins['manual']>0 and origins['old']>0
    dump(dest/'training_receipt.json',dict(exit_code=0,branch=BRANCH,steps=step,training_seconds=seconds,
         stop_reason='step_budget' if step==cfg['train_steps'] else 'time_budget',checkpoint_steps=saved,
         initial_state_sha256=initial_hash,final_state_sha256=state_sha(net),all_parameters_trainable=True,
         parameter_count=sum(p.numel() for p in net.parameters()),changed_parameters=changes,
         nonzero_gradient_parameter_names=sorted(nonzero),sampling_by_id=dict(counts),
         sampling_by_origin=dict(origins),sampling_by_source_group=dict(groups),
         all_manual_images_sampled=all(counts[i['id']]>0 for i in manual),
         before_diagnostic_fit=before,after_diagnostic_fit=after,
         final_path=str(dest/'final.pt'),final_sha256=sha(dest/'final.pt'),
         dataset_index_sha256=sha(HERE/'dataset/index.json'),config_sha256=sha(HERE/'training_config.json'),
         sampling_plan_sha256=sha(HERE/'sampling_plan.json'),source_sha256=sha(__file__),
         formal_weights_reset_to_A750=True,selected_by_evaluation=False,
         manual_red_is_finite_known_negative_not_unmarked_background=True))

def select():
    receipt=load(HERE/'model/training_receipt.json')
    assert receipt['exit_code']==0
    step=receipt['checkpoint_steps'][-1]
    path=HERE/'model/checkpoints'/f'step_{step:04d}.pt'
    dest=HERE/'selected_checkpoints.json';assert not dest.exists()
    dump(dest,dict(common_step=step,paths={BRANCH:dict(path=str(path),sha256=sha(path))},
                   selection_rule='last actual saved formal update; no evaluation or gallery-based model choice',
                   actual_steps=receipt['steps']))

def cv2_module():
    sys.path.append(str(ROOT/'vision_prototype/.venv/Lib/site-packages'))
    import cv2
    cv2.setNumThreads(2)
    return cv2

def input_from_raw(image_path,seed):
    cv2=cv2_module();bgr=cv2.imdecode(np.fromfile(image_path,np.uint8),cv2.IMREAD_COLOR)
    assert bgr.shape==(3000,4096,3)
    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
    image=cv2.resize(gray[1050:],(2048,975),interpolation=cv2.INTER_AREA)
    mark=np.zeros_like(image,np.uint8);q=np.rint((np.asarray(seed)-[0,1050])*.5).astype(np.int32)
    cv2.polylines(mark,[q],False,1,1,cv2.LINE_8);assert mark.any()
    distance=cv2.distanceTransform(1-mark,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    return image,np.exp(-.5*(distance/2.)**2).astype(np.float32)

def input_tensor(image,seed,cfg):
    zero=np.zeros_like(image)
    arrays=dict(image=image,seed=seed,target=zero,valid=zero.astype(bool),stage=zero.astype(np.uint8),roi=np.ones_like(image,bool))
    x,_,_,_,meta=tensor_sample(arrays,cfg,False)
    return x,meta

def fixed_jobs():
    index=load(BASE/'runs/run_20261030_002/data/index.json');jobs=[]
    for item in index['items']:
        jobs.append(dict(id=item['id'],mode='CONDITIONAL',image_path=item['image_path'],image_sha256=item['image_sha256'],
                         source_group=item['source_group'],split=item['split'],seed_xy=item['known_seed_xy']))
        if item['split']=='eval':
            auto=load(item['auto_entry_path'])
            jobs.append(dict(id=item['id'],mode='AUTO',image_path=item['image_path'],image_sha256=item['image_sha256'],
                             source_group=item['source_group'],split='eval',seed_xy=auto.get('free_seed') if auto['available'] else [],
                             auto_entry_path=item['auto_entry_path']))
    # Read only training manifest metadata; no supervision cache opens here.
    _,_,manual=training_index()
    for item in manual:
        jobs.append(dict(id=item['id'],mode='CONDITIONAL',image_path=item['image_path'],image_sha256=item['image_sha256'],
                         source_group=item['source_group'],split='manual_train',seed_xy=item['known_seed_xy']))
    return jobs

def infer(which):
    cfg=make_config();configure(cfg['threads']);selection=load(HERE/'selected_checkpoints.json')
    checkpoint=selection['paths'][BRANCH];assert sha(checkpoint['path'])==checkpoint['sha256']
    net=load_model(checkpoint['path']);net.eval();tick=time.perf_counter()
    if which=='gallery':
        jobs=[]
        for job in protocol()['gallery_jobs']:
            for k,seed in enumerate(job['seeds']):
                jobs.append(dict(id=job['id'],mode='GALLERY_AUTO',entry_index=k,image_path=job['image_path'],
                                 image_sha256=job['image_sha256'],source_group=job['source_group'],seed_xy=seed['xy'],split='display_gallery'))
    else:jobs=fixed_jobs()
    out=HERE/'inference'/which;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'receipt.json').exists(),'Preserve completed inference records'
    capture={}
    def hook(_module,_args,value):
        capture['features']=F.avg_pool2d(value,4,4).detach().to(torch.float16).cpu().numpy()[0].copy()
    handle=net.dec1.register_forward_hook(hook)
    rows=[];forwards=0
    try:
        with torch.inference_mode():
            for j,job in enumerate(jobs,1):
                started=time.perf_counter();assert sha(job['image_path'])==job['image_sha256']
                dest=(out/BRANCH/job['id']/f'entry_{job["entry_index"]:02d}.npz' if which=='gallery'
                      else out/BRANCH/job['mode']/job['id']/'response.npz')
                dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists(),'Do not replace earlier predictions'
                if not job['seed_xy']:
                    response=np.full((975,2048),np.nan,np.float32)
                    features=np.full((8,248,512),np.nan,np.float16)
                    status='entry_unavailable';meta=None
                else:
                    image,seed=input_from_raw(job['image_path'],job['seed_xy']);x,meta=input_tensor(image,seed,cfg)
                    response=net(x)[0,0,:975,:2048].numpy().copy();features=capture.pop('features')
                    assert response.dtype==np.float32 and response.shape==(975,2048)
                    assert features.dtype==np.float16 and features.shape==(8,248,512)
                    assert np.isfinite(response).all() and np.isfinite(features).all() and 0<=response.min()<=response.max()<=1
                    status='predicted';forwards+=1
                np.savez_compressed(dest,response=response,features=features)
                row=dict(id=job['id'],mode=job['mode'],split=job['split'],source_group=job['source_group'],
                         image_path=job['image_path'],image_sha256=job['image_sha256'],seed_xy=job['seed_xy'],
                         entry_index=job.get('entry_index',0),response_path=str(dest),response_sha256=sha(dest),
                         status=status,model_path=checkpoint['path'],model_sha256=checkpoint['sha256'],
                         model_step=selection['common_step'],branch=BRANCH,normalization=meta,
                         shape_hw=[975,2048],bbox_original_xyxy=[0,1050,4096,3000],outside_search_domain='unknown, not predicted',
                         response_semantics='original sigmoid center strength0..1, not calibrated target identity probability',
                         feature_shape=[8,248,512],feature_source='new trained U1 dec1 avg_pool4 in SAME new forward; no old A feature cache',
                         feature_mapping='model_pixel=(original_xy-[0,1050])/2; pooled_pixel=(model_pixel+0.5)/4-0.5',
                         new_model_forward=status=='predicted',target_labels_read=False,seconds=time.perf_counter()-started)
                if job.get('auto_entry_path'):row['entry_path']=job['auto_entry_path']
                dump(dest.with_suffix('.json'),row);rows.append(row)
                if j==1 or j%20==0 or j==len(jobs):
                    print(dict(inferred=j,total=len(jobs),forwards=forwards,seconds=round(time.perf_counter()-tick,1)),flush=True)
    finally:handle.remove()
    dump(out/'receipt.json',dict(exit_code=0,command=sys.argv,rows=rows,new_forward_count=forwards,
         seconds=time.perf_counter()-tick,selected_checkpoints_sha256=sha(HERE/'selected_checkpoints.json'),
         source_sha256=sha(__file__),evaluation_references_read=False,prediction_saved_before_evaluation=True,
         response_and_R1_spatial_features_from_same_new_model=True))

def fit_all():
    cfg=make_config();configure(cfg['threads']);selection=load(HERE/'selected_checkpoints.json')
    _,old,manual=training_index();items=old+manual;arrays=arrays_for(items)
    model=selection['paths'][BRANCH];assert sha(model['path'])==model['sha256']
    net=load_model(model['path'])
    dest=HERE/'training_fit.json';assert not dest.exists()
    dump(dest,dict(model=model,selected_step=selection['common_step'],fit=fit_stats(net,items,arrays,cfg)))

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['smallcheck','train','select','infer-fixed','infer-gallery','fit-stats'])
    args=parser.parse_args()
    if args.mode=='smallcheck':smallcheck()
    elif args.mode=='train':train()
    elif args.mode=='select':select()
    elif args.mode=='fit-stats':fit_all()
    else:infer('fixed' if args.mode=='infer-fixed' else 'gallery')

if __name__=='__main__':main()
