"""Frozen, offline paired short-horizon experiment. No hardware or event inputs.

Conditional patches explicitly contain a trusted GT past prefix. Automatic
rollout instead reads only an annotation-free whitelist and its own predictions.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time
import numpy as np

LAB=Path(__file__).resolve().parent
BASE=LAB/'outputs/v25_local_continuation/trial_001'
OLD=LAB/'outputs/v11_path_identity'
ARMS=('high','low')

def read(path):return json.loads(Path(path).read_text('utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def save(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:np.savez_compressed(f,**arrays)

def commit_segments(segments,width):
    """First four widths of predicted arclength, without any GT endpoint."""
    from .v25_local_data import slice_arc
    result=[];left=4*width
    for segment in segments:
        p=np.asarray(segment,float)
        if len(p)<2:continue
        length=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
        if length<1e-9:continue
        result.append(slice_arc(p,0.,min(left,length)))
        left-=min(left,length)
        if left<=1e-8:break
    return result

def training_plan(index,steps=600,batch=2):
    ids=index['train_sample_ids']
    if len(ids)!=10 or len(set(ids))!=10:raise ValueError('Fixed ten training images required')
    plan=[]
    for step in range(steps):
        chosen=[]
        for slot in range(batch):
            exposure=step*batch+slot;sid=ids[exposure%10];epoch=exposure//10
            keys=index['train_keys'][sid]
            if not keys:raise ValueError('Training sample lacks patches')
            key=keys[epoch%len(keys)]
            if index['entries'][key]['split']!='train':raise ValueError('Held-out target in training')
            chosen.append(key)
        plan.append(chosen)
    return plan

def freeze(base):
    from .v25_local_data import collect_sources,PATCH_RULES
    from .v25_local_contract import DECODE_PARAMETERS
    from .v25_local_evaluate import collect_sources as evaluation_sources
    if base.exists():raise FileExistsError(base)
    paths=set(collect_sources())
    paths.update(evaluation_sources())
    paths.update(LAB/n for n in ('v25_local_experiment.py','v25_local_contract.py',
        'v25_local_data.py','v25_local_evaluate.py','v11_path_model.py','v24_anchor_evaluate.py',
        'v23_path_prefix_audit.py','evaluate_centerline_multiscale_v9.py','centerline_path_metrics_v6.py',
        'centerline_review_v5.py','v18_front_diagnostics.py'))
    paths.update([OLD/'development/checkpoint.pt',OLD/'development/evaluation.json',OLD/'development_cache/index.json'])
    p=dict(schema='v25_local_detail_paired_trial_1',frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        seed=20261004,steps=600,batch_size=2,cpu_threads=2,learning_rate=.0001,
        initial_checkpoint=str(OLD/'development/checkpoint.pt'),initial_checkpoint_sha256=sha(OLD/'development/checkpoint.pt'),
        arms=list(ARMS),input_shape=[8,96,192],physical_window_widths=dict(u=[-4,8],v=[-3,3]),
        treatment='same field and tensor: native bilinear detail vs 3x3 area lowpass and bilinear reconstruction; recompute gray',
        representative_width_input_pixels=dict(high=16.,low_effective=16/3),
        training='same old V11 start; all parameters; fresh Adam; gradient clip5; final checkpoint only; each image120slots',
        patch_rules=PATCH_RULES,decoder=DECODE_PARAMETERS,
        conditional='GT past two widths disclosed; score only next four widths of trusted reference; never automatic accuracy',
        automatic='fixed V23 seed and width; own predicted last2width history; 24steps max; no GT corrections, stop/branch/orientation',
        rollout_commit_widths=4.,rollout_minimum_progress_widths=2.,rollout_steps=24,
        evaluation='conditional two models x two input details; auto matching details only; fixed original18 denominators',
        stopping='single paired trial; no threshold sweep or checkpoint selection; no deployment without auto bend and real occlusion verification',
        visibility_head_trained=False,unknown_tail_is_background=False,identity_confirmed=False,
        physical_events_validated=False,hardware_access=False,
        limitations='Short FOV/decoder differ from V11. Only within V25 high/low is a controlled image-detail contrast. High is not super-resolution.',
        source_hashes={str(p.resolve()):sha(p) for p in sorted(paths,key=str)})
    write(base/'protocol.json',p)
    print(json.dumps(dict(stage='frozen',protocol_sha256=sha(base/'protocol.json'))))

def verify(base):
    p=read(base/'protocol.json')
    for source,digest in p['source_hashes'].items():
        if sha(source)!=digest:raise ValueError('Frozen source changed: '+source)
    return p

def prepare(base):
    verify(base)
    from .v25_local_data import prepare as prepare_data
    index=prepare_data(base,sha(base/'protocol.json'))
    write(base/'training_plan.json',dict(batches=training_plan(index),input_index_sha256=sha(base/'cache/index.json')))
    verify(base)
    print(json.dumps(dict(stage='prepared',patches=len(index['entries']))))

def train(base,arm):
    p=verify(base)
    import torch
    from .v11_path_model import build_model,path_loss
    from .v25_local_data import load_input,load_train_target
    from .v25_local_contract import TARGET_KEYS
    destination=base/'models'/arm
    if destination.exists():raise FileExistsError(destination)
    index=read(base/'cache/index.json');plan=read(base/'training_plan.json')
    if plan['input_index_sha256']!=sha(base/'cache/index.json') or plan['batches']!=training_plan(index):
        raise ValueError('Training schedule modified')
    torch.set_num_threads(p['cpu_threads']);torch.manual_seed(p['seed'])
    ck=torch.load(p['initial_checkpoint'],map_location='cpu',weights_only=True)
    model=build_model(p['seed'],base_channels=ck['base_channels']);model.load_state_dict(ck['state_dict']);model.train()
    optimizer=torch.optim.Adam(model.parameters(),lr=p['learning_rate'])
    cache={}
    for key in sorted({k for batch in plan['batches'] for k in batch}):
        entry,arr=load_input(base,key);_,target=load_train_target(base,key)
        if entry['split']!='train':raise ValueError('No held-out labels permitted')
        cache[key]=(arr[arm],target)
    log=[];started=time.perf_counter()
    for step,batch in enumerate(plan['batches']):
        x=torch.from_numpy(np.stack([cache[k][0] for k in batch])).float()
        target={name:torch.from_numpy(np.stack([cache[k][1][name] for k in batch])).float() for name in TARGET_KEYS}
        optimizer.zero_grad(set_to_none=True)
        loss=path_loss(model(x),target)['loss']
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss; fixed experiment aborted')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step==0 or (step+1)%100==0:
            row=dict(step=step+1,loss=float(loss.detach()),elapsed_s=time.perf_counter()-started)
            log.append(row);print(json.dumps(dict(arm=arm,**row)),flush=True)
    destination.mkdir(parents=True,exist_ok=False)
    torch.save(dict(state_dict=model.state_dict(),base_channels=ck['base_channels'],arm=arm,
        protocol_sha256=sha(base/'protocol.json'),steps=600),destination/'checkpoint.pt')
    write(destination/'report.json',dict(arm=arm,loss_log=log,steps=600,
        checkpoint_sha256=sha(destination/'checkpoint.pt'),protocol_sha256=sha(base/'protocol.json'),
        input_index_sha256=sha(base/'cache/index.json'),training_plan_sha256=sha(base/'training_plan.json'),
        holdout_targets_used=False,elapsed_s=time.perf_counter()-started))
    verify(base)

def model_predictor(base,arm):
    import torch
    from .v11_path_model import build_model,mask_normalized_probabilities
    from .v25_local_contract import decode_local
    p=verify(base);torch.set_num_threads(p['cpu_threads'])
    path=base/'models'/arm/'checkpoint.pt';report=read(path.with_name('report.json'))
    if sha(path)!=report['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck['protocol_sha256']!=sha(base/'protocol.json'):raise ValueError('Checkpoint protocol mismatch')
    model=build_model(p['seed'],base_channels=ck['base_channels']);model.load_state_dict(ck['state_dict']);model.eval()
    def predict(tensor,valid,geometry):
        start=time.perf_counter()
        with torch.inference_mode():
            output=model(torch.from_numpy(tensor[None]).float())
            probability=mask_normalized_probabilities(output,torch.from_numpy(valid[None].astype(bool)))[0].numpy()
            slope=output['slope'][0].numpy()
        middle=time.perf_counter();decoded=decode_local(probability,slope,geometry,valid)
        decoded['segments']=commit_segments(decoded['segments'],geometry['width_px'])
        decoded['observed_segments']=decoded['segments'];decoded['commit']='first4widths_predicted_arclength_only'
        return decoded,probability,slope,(middle-start)*1000,(time.perf_counter()-middle)*1000
    return predict,path

def export_conditional(base,arm):
    from .v25_local_data import load_input
    predict,path=model_predictor(base,arm)
    destination=base/'exports'/arm
    if (destination/'conditional.json').exists() or (destination/'conditional_probabilities').exists():raise FileExistsError(destination)
    index=read(base/'cache/index.json');rows=[]
    for counter,key in enumerate(index['evaluation_keys']):
        entry,arrays=load_input(base,key)
        for mode in ARMS:
            decoded,prob,slope,fw,dm=predict(arrays[mode],arrays['valid_mask'],entry['geometry'])
            out=destination/'conditional_probabilities'/f'{counter:04d}_{mode}.npz';save(out,probability=prob,slope=slope)
            rows.append(dict(key=key,sample_id=entry['sample_id'],split=entry['split'],input_mode=mode,
                decoded=decoded,forward_ms=fw,decode_ms=dm,input_sha256=entry['input_sha256'],
                probability_path=str(out),probability_sha256=sha(out)))
        if (counter+1)%50==0:print(json.dumps(dict(arm=arm,conditional_patches=counter+1)),flush=True)
    write(destination/'conditional.json',dict(arm=arm,model_path=str(path),model_sha256=sha(path),
        protocol_sha256=sha(base/'protocol.json'),input_index_sha256=sha(base/'cache/index.json'),
        rows=rows,GT_prefix_conditioned=True,GT_future_used_for_inference=False,identity_confirmed=False))
    verify(base)

def rollout(rgb,seed,width,predict,mode,maximum_steps=24):
    from .v25_local_contract import build_input_pair
    from .v25_local_data import slice_arc
    history=np.asarray(seed,float)
    if history.ndim!=2 or len(history)<2 or not np.isfinite(history).all():raise ValueError('Invalid automatic seed')
    length=float(np.linalg.norm(np.diff(history,axis=0),axis=1).sum())
    if length<2*width:return dict(segments=[],steps=[],reason='automatic_free_span_shorter_than_two_widths')
    history=slice_arc(history,length-2*width,length);steps=[];segments=[];stop='fixed_maximum_steps'
    for step in range(maximum_steps):
        before=time.perf_counter()
        try:pair=build_input_pair(rgb,history,width)
        except ValueError:
            stop='predicted_prefix_outside_image_or_invalid';break
        patch_ms=(time.perf_counter()-before)*1000
        decoded,prob,slope,fw,dm=predict(pair[mode],pair['valid_mask'],pair['geometry'])
        candidate=decoded['segments']
        new=candidate[0] if len(candidate)==1 else []
        gap=float(np.linalg.norm(np.asarray(new[0])-np.asarray(history[-1]))) if new else None
        length=float(np.linalg.norm(np.diff(new,axis=0),axis=1).sum()) if new else 0.
        connected=bool(new and gap<=.375*width)
        reason=decoded['reason'] if connected else 'no_new_segment_connected_to_own_prefix'
        steps.append(dict(step=step,geometry=pair['geometry'],input_sha256=hashlib.sha256(pair[mode].tobytes()).hexdigest(),
            segments=new and [new] or [],accepted=connected,reason=reason,gap_px=gap,
            predicted_length_px=length,forward_ms=fw,decode_ms=dm,patch_ms=patch_ms,
            ambiguous=decoded['diagnostics'].get('ambiguous'),identity_confirmed=False))
        if not connected:stop=reason;break
        segments.append(new)
        if length<2*width or decoded['diagnostics'].get('stopped_on_unreliable_connection',True):
            stop='short_or_unreliable_continuation';break
        history=slice_arc(new,length-2*width,length)
    return dict(segments=segments,steps=steps,reason=stop,identity_confirmed=False,
        GT_used=False,visibility_confirmed=False)

def export_auto(base,arm):
    from PIL import Image
    from .v25_local_data import AUTO_ROW_KEYS
    predict,path=model_predictor(base,arm);source_path=base/'cache/auto_sources.json';sources=read(source_path)
    destination=base/'exports'/arm/'auto.json'
    if destination.exists():raise FileExistsError(destination)
    rows=[]
    for entry in sources['rows']:
        if set(entry)!=set(AUTO_ROW_KEYS):raise ValueError('Auto source must be annotation-free strict whitelist')
        row=dict(sample_id=entry['sample_id'],split=entry['split'],available=entry['available'],reason=entry['reason'],
            auto_input_sha256=sha(source_path),source_image_sha256=entry['source_image_sha256'],segments=[],steps=[])
        if entry['available']:
            if sha(entry['source_image_path'])!=entry['source_image_sha256']:raise ValueError('Image changed')
            with Image.open(entry['source_image_path']) as image:rgb=np.asarray(image.convert('RGB'))
            row.update(rollout(rgb,entry['past_seed_xy'],entry['width_px'],predict,arm))
        rows.append(row);print(json.dumps(dict(arm=arm,sample=entry['sample_id'],steps=len(row['steps']))),flush=True)
    write(destination,dict(arm=arm,model_path=str(path),model_sha256=sha(path),protocol_sha256=sha(base/'protocol.json'),
        auto_input_sha256=sha(source_path),rows=rows,annotation_access_in_rollout=False,identity_confirmed=False))
    verify(base)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','verify','prepare','train','conditional','auto'])
    parser.add_argument('--base',type=Path,default=BASE);parser.add_argument('--arm',choices=ARMS,default='high');args=parser.parse_args()
    if args.stage in ('freeze','verify','prepare'):globals()[args.stage](args.base)
    elif args.stage=='train':train(args.base,args.arm)
    elif args.stage=='conditional':export_conditional(args.base,args.arm)
    else:export_auto(args.base,args.arm)

if __name__=='__main__':main()
