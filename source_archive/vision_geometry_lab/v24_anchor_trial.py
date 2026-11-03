"""One equal-budget anchor-input finetuning trial, offline and non-deployable.

Two arms start at the same frozen V11 checkpoint. The sole treatment is the
training input-anchor schedule. Geometry, loss, step budget and decoder stay
fixed. No holdout labels or event times enter training or inference.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time
import numpy as np

LAB = Path(__file__).resolve().parent
BASE = LAB/'outputs/v24_anchor_consistency/trial_001'
OLD = LAB/'outputs/v11_path_identity'
GLOVE = LAB/'outputs/v16_event_feasibility/extract/right/frames/f000168.json'
TARGET_KEYS = ('target_distribution','column_valid','row_valid','slope_target','slope_valid')
ARMS = ('control','anchor_mix')


def read(path): return json.loads(Path(path).read_text('utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def save(path, **arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f: np.savez_compressed(f,**arrays)


def training_schedule(index, arm, steps, batch_size):
    if arm not in ARMS or steps<=0 or batch_size<=0: raise ValueError('Invalid training plan')
    ids=index['train_sample_ids']
    if len(ids)!=10 or len(set(ids))!=10: raise ValueError('Expected fixed ten train images')
    batches=[]
    for step in range(steps):
        batch=[]
        for slot in range(batch_size):
            exposure=step*batch_size+slot
            sid=ids[exposure%len(ids)]
            epoch=exposure//len(ids)
            variant_slot=0 if arm=='control' else epoch%5
            selected=index['train_keys'][sid][variant_slot]
            key=selected['resolved_key']; entry=index['entries'][key]
            if entry['split']!='train' or entry['sample_id']!=sid or not entry['available']:
                raise ValueError('Invalid training input source')
            if entry['variant'].startswith('probe'): raise ValueError('Unseen probe entered training')
            batch.append(dict(sample_id=sid,key=key,scheduled_variant=selected['scheduled_variant'],
                fallback_reason=selected['fallback_reason']))
        batches.append(batch)
    return batches


def freeze(base):
    if base.exists(): raise FileExistsError(base)
    from .v24_anchor_data import collect_sources, TRAIN_FRACTIONS, PROBE_FRACTIONS
    paths=set(collect_sources())
    paths.update([OLD/'development/checkpoint.pt',OLD/'development/report.json',GLOVE,Path(read(GLOVE)['input_path'])])
    paths.update(LAB/name for name in ('v24_anchor_trial.py','v24_anchor_data.py','v24_anchor_evaluate.py',
        'v24_candidate_support_audit.py','v11_path_model.py','v11_path_decode.py','v11_path_contract.py',
        'v23_path_prefix_audit.py','centerline_path_metrics_v6.py','v18_front_diagnostics.py'))
    p=dict(schema='v24_equal_budget_anchor_consistency_1',
        frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        seed=20261003,steps=600,batch_size=2,cpu_threads=2,learning_rate=.0001,
        optimizer='fresh Adam for each arm, all parameters trainable, grad norm clip5',
        initial_checkpoint=str(OLD/'development/checkpoint.pt'),initial_checkpoint_sha256=sha(OLD/'development/checkpoint.pt'),
        arms=list(ARMS),final_checkpoint_only=True,
        schedule='same round-robin base image exposures, 120 per image; treatment rotates five variant slots by epoch',
        control='manual input only, same 600 additional updates',
        anchor_mix='manual/subset0/subset1/subset2/auto in five-epoch cycles; missing train auto explicitly falls back to manual',
        train_fractions=TRAIN_FRACTIONS,probe_fractions=PROBE_FRACTIONS,
        width_policy='subsets retain frozen manual optical width; auto retains frozen V23 width; no new fitting',
        unchanged='V11 architecture, conditional row loss, slope weight, decode thresholds',
        target_policy='train only; quality trusted centerline and ignore masks mapped to new input; unknown tails not negative',
        evaluation='18 samples fixed original V6 reference denominator; manual/auto/probe0/probe1; auto failures retained',
        probes='different legal subsets of same manual free span, not physical contact labels',
        critical_bend='GT-only first half-optical-width departure from fixed manual seed line, +/-2 widths arc; not physical contact',
        research_gate=dict(auto_development_bend_coverage_min=.8,real_occlusion_must_not_be_false_confirmation=True),
        stopping='one paired trial, no best-checkpoint or threshold sweep; failed gate means no deployment and inspect frozen candidate support',
        challenge='right16.8s glove input frozen before trial, qualitative unless trusted occlusion mask exists',
        timings='CPU model+decode separate from cache read; not camera-to-decision latency',
        field_generalization_validated=False,identity_confirmed=False,event_time_inference=False,hardware_access=False,
        source_hashes={str(path.resolve()):sha(path) for path in sorted(paths,key=str)})
    write(base/'protocol.json',p)
    print(json.dumps(dict(stage='frozen',path=str(base/'protocol.json'),sha256=sha(base/'protocol.json'))))


def verify(base):
    p=read(base/'protocol.json')
    for path,expected in p['source_hashes'].items():
        if sha(path)!=expected: raise ValueError('Frozen source changed: '+path)
    return p


def prepare(base):
    verify(base)
    from .v24_anchor_data import prepare as prepare_data
    index=prepare_data(base,sha(base/'protocol.json'))
    for arm in ARMS:
        plan=training_schedule(index,arm,600,2)
        write(base/'plans'/(arm+'.json'),dict(arm=arm,batches=plan,
            protocol_sha256=sha(base/'protocol.json'),input_index_sha256=sha(base/'cache/input_index.json')))
    verify(base)
    print(json.dumps(dict(stage='prepared',inputs=len(index['entries']),evaluation=len(index['evaluation_keys']))))


def train(base,arm):
    p=verify(base)
    if arm not in ARMS: raise ValueError('Unknown training arm')
    run=base/'models'/arm
    if run.exists(): raise FileExistsError('Do not silently resume or replace training')
    from .v24_anchor_data import load_input, load_train_target
    import torch
    from .v11_path_model import build_model, path_loss
    torch.set_num_threads(p['cpu_threads']);torch.manual_seed(p['seed'])
    index=read(base/'cache/input_index.json');plan=read(base/'plans'/(arm+'.json'))
    if (plan['protocol_sha256']!=sha(base/'protocol.json') or
            plan['input_index_sha256']!=sha(base/'cache/input_index.json') or
            plan['batches']!=training_schedule(index,arm,p['steps'],p['batch_size'])):
        raise ValueError('Training schedule changed')
    cached={}
    for key in sorted({s['key'] for b in plan['batches'] for s in b}):
        entry,arr=load_input(base,key);target_entry,target=load_train_target(base,key)
        if target_entry['input_sha256']!=entry['input_sha256']: raise ValueError('Target not bound to input')
        cached[key]=(arr,target)
    checkpoint=torch.load(p['initial_checkpoint'],map_location='cpu',weights_only=True)
    model=build_model(p['seed'],base_channels=checkpoint['base_channels'])
    model.load_state_dict(checkpoint['state_dict']);model.train()
    optimizer=torch.optim.Adam(model.parameters(),lr=p['learning_rate'])
    log=[];start=time.perf_counter()
    for step,batch in enumerate(plan['batches']):
        selected=[cached[item['key']] for item in batch]
        x=torch.from_numpy(np.stack([a['input'] for a,t in selected])).float()
        target={key:torch.from_numpy(np.stack([t[key] for a,t in selected])).float() for key in TARGET_KEYS}
        optimizer.zero_grad(set_to_none=True)
        losses=path_loss(model(x),target);loss=losses['loss']
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss; abort fixed trial')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step==0 or (step+1)%100==0:
            row=dict(step=step+1,loss=float(loss.detach()),elapsed_s=time.perf_counter()-start)
            log.append(row);print(json.dumps(dict(arm=arm,**row)),flush=True)
    run.mkdir(parents=True,exist_ok=False)
    out=run/'checkpoint.pt'
    torch.save(dict(state_dict=model.state_dict(),base_channels=checkpoint['base_channels'],seed=p['seed'],
        arm=arm,steps=p['steps'],protocol_sha256=sha(base/'protocol.json')) ,out)
    write(run/'report.json',dict(arm=arm,steps=p['steps'],loss_log=log,elapsed_s=time.perf_counter()-start,
        initial_checkpoint_sha256=p['initial_checkpoint_sha256'],final_checkpoint_sha256=sha(out),
        protocol_sha256=sha(base/'protocol.json'),plan_sha256=sha(base/'plans'/(arm+'.json')),
        input_index_sha256=sha(base/'cache/input_index.json'),target_index_sha256=sha(base/'cache/train_target_index.json'),
        unique_training_inputs=len(cached),no_holdout_or_probe_targets_loaded=True,hardware_access=False))
    verify(base)


def export(base,arm):
    p=verify(base)
    if arm not in ('baseline',*ARMS): raise ValueError('Unknown export arm')
    folder=base/'exports'/arm
    if folder.exists(): raise FileExistsError('Do not replace exported predictions')
    import torch
    from .v24_anchor_data import load_input
    from .v11_path_model import build_model,mask_normalized_probabilities
    from .v11_path_decode import decode_path
    torch.set_num_threads(p['cpu_threads'])
    path=Path(p['initial_checkpoint']) if arm=='baseline' else base/'models'/arm/'checkpoint.pt'
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if arm!='baseline':
        report=read(path.with_name('report.json'))
        if report['final_checkpoint_sha256']!=sha(path) or checkpoint['protocol_sha256']!=sha(base/'protocol.json'):
            raise ValueError('Model provenance mismatch')
    model=build_model(p['seed'],base_channels=checkpoint['base_channels'])
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    index=read(base/'cache/input_index.json')
    if index['protocol_sha256']!=sha(base/'protocol.json'): raise ValueError('Input cache protocol mismatch')
    rows=[]
    with torch.inference_mode():
        for counter,key in enumerate(index['evaluation_keys']):
            entry,arr=load_input(base,key)
            row={k:entry[k] for k in ('sample_id','split','variant','available','reason','input_sha256')}
            row.update(key=key,probability_path=None,probability_sha256=None,forward_ms=0.,decode_ms=0.,
                decoded=dict(segments=[],observed_segments=[],reason=entry['reason'],identity_confirmed=False))
            if arr is not None:
                before=time.perf_counter()
                outputs=model(torch.from_numpy(arr['input'][None]).float())
                probability=mask_normalized_probabilities(outputs,torch.from_numpy(arr['valid_mask'][None]>.5))[0].numpy()
                slope=outputs['slope'][0].numpy();inferred=time.perf_counter()
                decoded=decode_path(probability,slope,entry['geometry'],arr['valid_mask']);ended=time.perf_counter()
                npz=folder/'probabilities'/f'{counter:03d}.npz';save(npz,probability=probability,slope=slope)
                row.update(probability_path=str(npz),probability_sha256=sha(npz),decoded=decoded,
                    forward_ms=(inferred-before)*1000,decode_ms=(ended-inferred)*1000)
            rows.append(row)
    write(folder/'index.json',dict(arm=arm,rows=rows,model_sha256=sha(path),model_path=str(path),
        protocol_sha256=sha(base/'protocol.json'),input_index_sha256=sha(base/'cache/input_index.json'),
        labels_used_for_inference=False,identity_confirmed=False,hardware_access=False))
    # A previously observed, fixed failure challenge. Never train on it here.
    meta=read(GLOVE)
    if sha(meta['input_path'])!=meta['input_sha256']: raise ValueError('Glove input changed')
    with np.load(meta['input_path'],allow_pickle=False) as a:
        x=a['input'].copy();valid=a['valid_mask'].copy()
    with torch.inference_mode():
        out=model(torch.from_numpy(x[None]).float())
        prob=mask_normalized_probabilities(out,torch.from_numpy(valid[None]>.5))[0].numpy()
        slope=out['slope'][0].numpy()
    decoded=decode_path(prob,slope,meta['geometry'],valid)
    save(folder/'glove_probability.npz',probability=prob,slope=slope)
    write(folder/'challenge.json',dict(name='right16.8s_glove',arm=arm,decoded=decoded,
        metadata_path=str(GLOVE),metadata_sha256=sha(GLOVE),geometry=meta['geometry'],
        probability_path=str(folder/'glove_probability.npz'),probability_sha256=sha(folder/'glove_probability.npz'),
        protocol_sha256=sha(base/'protocol.json'),model_sha256=sha(path),
        trusted_occlusion_mask_available=False,identity_confirmed=False,hardware_access=False))
    verify(base);print(json.dumps(dict(stage='exported',arm=arm,rows=len(rows))),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=('freeze','verify','prepare','train','export'))
    ap.add_argument('--base',type=Path,default=BASE)
    ap.add_argument('--arm',choices=('baseline',*ARMS))
    args=ap.parse_args()
    if args.stage in ('train','export'):
        if args.arm is None: ap.error('--arm required')
        globals()[args.stage](args.base,args.arm)
    else: globals()[args.stage](args.base)


if __name__=='__main__': main()
