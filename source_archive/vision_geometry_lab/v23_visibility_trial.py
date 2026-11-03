"""One frozen V23 null-head mechanism trial; no devices or 9/25 sources.

prepare uses vision_prototype venv; train/evaluate use vision_training venv.
Every command preserves old artifacts and refuses implicit overwrite/resume.
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
OLD = LAB/'outputs/v11_path_identity'
BASE = LAB/'outputs/v23_visibility/mechanism_001'
INVENTORY = LAB/'outputs/centerline_v9/dataset/inventory.json'
CHALLENGE = LAB/'outputs/v16_event_feasibility/extract/right/frames/f000168.json'


def read(path): return json.loads(Path(path).read_text('utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
def save(path, **arrays):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream: np.savez_compressed(stream,**arrays)
def archive(path):
    with np.load(path,allow_pickle=False) as value: return {k:value[k].copy() for k in value.files}


def freeze(base):
    if base.exists(): raise FileExistsError(base)
    index=read(OLD/'development_cache/index.json'); inventory=read(INVENTORY)
    manifest=read(OLD/'development_manifest.json')
    # Check actual sample/path records, not audit field names such as
    # blind_20260925_read=False in the historical inventory header.
    if '20260925' in json.dumps([index['entries'],inventory['samples'],manifest['samples']],ensure_ascii=False):
        raise ValueError('9/25 prohibited')
    paths=[OLD/'development_cache/index.json', OLD/'development_manifest.json',
        OLD/'development/checkpoint.pt', OLD/'protocol.json', INVENTORY, CHALLENGE]
    paths += [LAB/name for name in ('v23_visibility_model.py','v23_visibility_trial.py',
        'v11_path_model.py','v11_path_contract.py','v11_path_decode.py',
        'evaluate_centerline_multiscale_v9.py','centerline_quality_v5.py',
        'centerline_quality_cutoff_v8.py','centerline_review_v5.py','rope_detector.py')]
    for entry in index['entries']:
        if entry['available']:
            if sha(entry['cache'])!=entry['cache_sha256']: raise ValueError('Old cache changed')
            paths.append(Path(entry['cache']))
    for sample in manifest['samples']:
        paths += [Path(sample['image_path']),Path(sample['annotation_path'])]
        qp=Path(sample['annotation_path']).with_suffix('.quality.json')
        if qp.exists(): paths.append(qp)
    for sample in inventory['samples']:
        if sample['candidate_id'] in ('C03','C06'):
            paths += [Path(sample['image_path']),Path(sample['annotation_path'])]
    meta=read(CHALLENGE); paths.append(Path(meta['input_path']))
    p=dict(schema='v23_visibility_mechanism_1',
        frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        steps=600, seed=20261003, learning_rate=.001, batch_size=8, cpu_threads=2,
        threshold=.5, final_checkpoint_only=True, frozen_backbone=str(OLD/'development/checkpoint.pt'),
        training='10 train original/zero-query/three synthetic strip variants, trusted columns only',
        synthetic_train_seeds=[23000,24000,25000],synthetic_test_seed=83000,
        synthetic_masks='three target-independent full-height RGB strips; 4-column boundary exclusion',
        synthetic_test_difference='unseen positions/seed and sinusoidal texture, not real occlusion truth',
        feature_summary='8 weighted + 8 mean + 8 max frozen features; entropy,peak,gray mean/std,normalized abs,query max',
        head='30->24 k5 SiLU ->16 k5 SiLU ->1 k1, only head trainable',
        geometry_logits_frozen=True, gate='P(row|observable)*P(observable); threshold0.5; never renormalize',
        query_zero='same RGB,crop,coords,validity; only channel4 zero; null means no specified query',
        mechanism_gate=dict(train_original_retention_at_least=.90,train_synthetic_null_rejection_at_least=.90,
            train_zero_query_rejection_at_least=.99),
        evaluation='equally weighted trustworthy strip-column evidence, NOT arc-length or instance accuracy',
        error_definition='argmax row minus trusted Gaussian mean times native px/row; 5px tolerance, seed-normal direction',
        stop='One final model. Failed mechanism gate stops holdout/challenge inference; no threshold or budget tuning.',
        holdout='8 reused source-separated development images, not blind validation',
        unknown_tails='excluded, not negative',
        challenges='C06 free span only, no wound GT score; right16.8s glove archive qualitative, no mask truth',
        c03='confirmed absence and no real seed; bypass/no-query input contract only, no fabricated seed',
        scope='local observability candidate, neither physical absence nor same-rope identity confirmation',
        source_hashes={str(path.resolve()):sha(path) for path in paths},
        hardware_access=False,blind_20260925_read=False,force_command=None)
    write(base/'protocol.json',p)
    print(json.dumps(dict(stage='frozen',protocol=str(base/'protocol.json'),sha256=sha(base/'protocol.json'))))


def verify(base):
    p=read(base/'protocol.json')
    for path,expected in p['source_hashes'].items():
        if sha(path)!=expected: raise ValueError('Frozen source changed: '+path)
    return p


def prepare(base):
    verify(base)
    if (base/'prepared/index.json').exists(): raise FileExistsError('Already prepared')
    from .evaluate_centerline_multiscale_v9 import _dataset,scorable_reference,independent_width
    from .v11_path_contract import build_targets,build_input
    from .centerline_review_v5 import read_clean_image
    _,records,_=_dataset(OLD/'development_manifest.json')
    record_by_id={r['sample']['sample_id']:r for r in records}
    entries=[]
    for entry in read(OLD/'development_cache/index.json')['entries']:
        if not entry['available']: raise ValueError('All18 expected available')
        a=archive(entry['cache']); record=record_by_id[entry['sample_id']]
        label,quality=record['label'],record['quality']
        reference,_=scorable_reference(label,quality)
        target=build_targets(entry['geometry'],reference,
            label['ignore_polygons']+quality['precision_exclude_polygons'],a['valid_mask'])
        folder='train' if entry['split']=='train' else 'evaluation_only'
        path=base/'prepared'/folder/(entry['sample_id']+'.npz')
        mean=(target['target_distribution']*np.arange(256)[:,None]).sum(axis=0)
        save(path,input=a['input'],valid_mask=a['valid_mask'],
            trusted_columns=target['column_valid'],target_mean_row=mean)
        entries.append(dict(sample_id=entry['sample_id'],split=entry['split'],group_id=entry['group_id'],
            path=str(path),sha256=sha(path),geometry=entry['geometry'],
            trusted_columns=int(target['column_valid'].sum())))
    # C06 explicitly states wound part out of view. Its old polylines are not wound truth.
    inv=read(INVENTORY); c06=next(s for s in inv['samples'] if s['candidate_id']=='C06')
    original=read(c06['source_manifest_path'])
    sample=next(s for s in original['samples'] if s['sample_id']==c06['sample_id'])
    label=read(sample['annotation_path']); rgb=np.asarray(read_clean_image(sample))
    width=independent_width(np.ascontiguousarray(rgb[:,:,::-1]),label['polylines']['free_seed'])
    challenges=[]
    if width['width_px'] is not None:
        built=build_input(rgb,label['polylines']['free_seed'],width['width_px'])
        path=base/'prepared/challenges/c06.npz';save(path,input=built['input'],valid_mask=built['valid_mask'])
        challenges.append(dict(name='c06_free_span_only',path=str(path),sha256=sha(path),
            geometry=built['geometry'],interpretation=c06['annotation_notes'],ground_truth_geometry=False))
    meta=read(CHALLENGE);path=Path(meta['input_path'])
    challenges.append(dict(name='right_16_8s_glove',path=str(path),sha256=sha(path),
        geometry=meta['geometry'],source_frame=168,source_time_s=16.8,
        interpretation='Previously observed glove misconnection; no verified glove mask',ground_truth_geometry=False))
    write(base/'prepared/index.json',dict(protocol_sha256=sha(base/'protocol.json'),entries=entries,challenges=challenges,
        C03=dict(no_query=True,prediction_bypassed=True,physical_absence_detection_tested=False)))
    verify(base);print(json.dumps(dict(stage='prepared',entries=len(entries),challenges=len(challenges))))


def prepared(base):
    index=read(base/'prepared/index.json')
    if index['protocol_sha256']!=sha(base/'protocol.json'): raise ValueError('Prepared provenance')
    return index


def load_input(entry):
    if sha(entry['path'])!=entry['sha256']:raise ValueError('Prepared file changed')
    return archive(entry['path'])


def backbone(p):
    import torch
    from .v11_path_model import build_model
    torch.set_num_threads(p['cpu_threads'])
    ck=torch.load(p['frozen_backbone'],map_location='cpu',weights_only=True)
    model=build_model(ck['seed'],base_channels=ck['base_channels'])
    model.load_state_dict(ck['state_dict']);model.eval();model.requires_grad_(False)
    return model


def features(model,inputs):
    import torch
    from .v11_path_model import mask_normalized_probabilities
    from .v23_visibility_model import local_features
    dense=[]
    hook=model.center_head.register_forward_pre_hook(lambda module,args:dense.append(args[0].detach()))
    try:
        with torch.no_grad():
            x=torch.from_numpy(inputs[None]).float(); output=model(x)
            probability=mask_normalized_probabilities(output,x[:,7]>.5)
            f=local_features(dense[0],probability,x)
    finally:hook.remove()
    return f[0].numpy(),probability[0].numpy(),output['slope'][0].numpy()


def ratio(n,d):return float(n/d) if d else None
def counts(observable,target,mask,threshold):
    kept=observable>=threshold; pos=mask&(target>.5);neg=mask&~(target>.5)
    return dict(positive=int(pos.sum()),retained_positive=int((pos&kept).sum()),
        negative=int(neg.sum()),rejected_negative=int((neg&~kept).sum()),
        positive_retention=ratio((pos&kept).sum(),pos.sum()),
        negative_rejection=ratio((neg&~kept).sum(),neg.sum()))
def merge_counts(rows):
    values={k:sum(r[k] for r in rows) for k in ('positive','retained_positive','negative','rejected_negative')}
    values.update(positive_retention=ratio(values['retained_positive'],values['positive']),
        negative_rejection=ratio(values['rejected_negative'],values['negative']))
    return values


def train(base):
    p=verify(base);index=prepared(base)
    if (base/'train').exists():raise FileExistsError('Training cannot resume/overwrite')
    import torch
    from .v23_visibility_model import VisibilityHead,intervention,supervision,masked_loss
    torch.manual_seed(p['seed']);model=backbone(p);head=VisibilityHead()
    optimizer=torch.optim.Adam(head.parameters(),lr=p['learning_rate'])
    cached=[];started=time.perf_counter()
    entries=[e for e in index['entries'] if e['split']=='train']
    if len(entries)!=10:raise ValueError('Exactly10train required')
    for number,entry in enumerate(entries):
        a=load_input(entry)
        configs=[('original',0),('zero_query',0)]+[('synthetic_occlusion',s+number) for s in p['synthetic_train_seeds']]
        for kind,seed in configs:
            v=intervention(a['input'],kind,seed);y,m=supervision(a['trusted_columns'],v)
            f,prob,slope=features(model,v['input'])
            path=base/'train/features'/f'{number:02d}_{kind}_{seed}.npz'
            save(path,features=f,target=y,supervised=m)
            cached.append(dict(path=str(path),sha256=sha(path),sample_id=entry['sample_id'],kind=kind,
                seed=seed,features=f,target=y,mask=m))
        print(json.dumps(dict(stage='train_feature_cache',sample=number+1,total=10)),flush=True)
    xs=torch.from_numpy(np.stack([c['features'] for c in cached]));ys=torch.from_numpy(np.stack([c['target'] for c in cached]))
    masks=torch.from_numpy(np.stack([c['mask'] for c in cached]))
    log=[]
    for step in range(p['steps']):
        selected=torch.randperm(len(cached))[:p['batch_size']]
        optimizer.zero_grad(set_to_none=True)
        loss=masked_loss(head(xs[selected]),ys[selected],masks[selected])
        loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.);optimizer.step()
        if step==0 or (step+1)%100==0:
            row=dict(step=step+1,loss=float(loss.detach()),elapsed_s=time.perf_counter()-started)
            log.append(row);print(json.dumps(row),flush=True)
    head.eval()
    with torch.no_grad():obs=torch.sigmoid(head(xs)).numpy()
    metrics={}
    for kind in ('original','zero_query','synthetic_occlusion'):
        metrics[kind]=merge_counts([counts(obs[i],c['target'],c['mask'],p['threshold'])
            for i,c in enumerate(cached) if c['kind']==kind])
    gate=p['mechanism_gate']
    passed=(metrics['original']['positive_retention']>=gate['train_original_retention_at_least']
        and metrics['zero_query']['negative_rejection']>=gate['train_zero_query_rejection_at_least']
        and metrics['synthetic_occlusion']['negative_rejection']>=gate['train_synthetic_null_rejection_at_least'])
    if any(v.requires_grad for v in model.parameters()) or any(v.grad is not None for v in model.parameters()):
        raise AssertionError('Backbone was not frozen')
    ck=base/'train/checkpoint.pt'
    torch.save(dict(state_dict=head.state_dict(),protocol_sha256=sha(base/'protocol.json'),steps=p['steps']),ck)
    write(base/'train/report.json',dict(protocol_sha256=sha(base/'protocol.json'),checkpoint_sha256=sha(ck),
        prepared_index_sha256=sha(base/'prepared/index.json'),steps=p['steps'],metrics=metrics,
        mechanism_passed=bool(passed),train_ids=[e['sample_id'] for e in entries],
        cached_features=[{k:c[k] for k in ('path','sha256','sample_id','kind','seed')} for c in cached],
        loss_log=log,elapsed_s=time.perf_counter()-started,head_parameters=sum(v.numel() for v in head.parameters()),
        frozen_parameter_gradients_absent=True,identity_confirmed=False,real_occlusion_validated=False))
    verify(base);print(json.dumps(dict(stage='train_done',mechanism_passed=bool(passed),metrics=metrics)))


def evaluation(base):
    p=verify(base);index=prepared(base);train_report=read(base/'train/report.json')
    if (base/'evaluation').exists():raise FileExistsError('Evaluation cannot overwrite')
    if not train_report['mechanism_passed']:
        write(base/'evaluation/report.json',dict(status='not_run_mechanism_failed',
            train_report_sha256=sha(base/'train/report.json'),protocol_sha256=sha(base/'protocol.json'),
            reason='Precommitted stop; no holdout or qualitative challenge inference',identity_confirmed=False))
        return
    import torch
    from .v23_visibility_model import VisibilityHead,intervention,supervision,gate_probabilities
    from .v11_path_decode import decode_path
    model=backbone(p);head=VisibilityHead()
    ck_path=base/'train/checkpoint.pt'
    if sha(ck_path)!=train_report['checkpoint_sha256']:raise ValueError('Head changed')
    ck=torch.load(ck_path,map_location='cpu',weights_only=True)
    if ck['protocol_sha256']!=sha(base/'protocol.json'):raise ValueError('Foreign head')
    head.load_state_dict(ck['state_dict']);head.eval()
    rows=[];timings=[]
    holdout=[e for e in index['entries'] if e['split']=='holdout']
    if len(holdout)!=8:raise ValueError('Exactly8development images')
    for number,entry in enumerate(holdout):
        a=load_input(entry);row=dict(sample_id=entry['sample_id'],variants={})
        for kind in ('original','zero_query','synthetic_occlusion'):
            v=intervention(a['input'],kind,p['synthetic_test_seed']+number,heldout=True)
            y,m=supervision(a['trusted_columns'],v);start=time.perf_counter()
            f,prob,slope=features(model,v['input'])
            with torch.no_grad():obs=torch.sigmoid(head(torch.from_numpy(f[None])))[0].numpy()
            timings.append((time.perf_counter()-start)*1000)
            item=counts(obs,y,m,p['threshold'])
            if kind=='original':
                error=np.abs(prob.argmax(axis=0)-a['target_mean_row'])*entry['geometry']['native_px_per_row']
                correct=error<=5.;kept=obs>=p['threshold'];trusted=a['trusted_columns']
                item.update(correct_baseline=int((trusted&correct).sum()),
                    wrong_baseline=int((trusted&~correct).sum()),
                    correct_retained=int((trusted&correct&kept).sum()),
                    wrong_rejected=int((trusted&~correct&~kept).sum()),
                    baseline_within5_fraction=ratio((trusted&correct).sum(),trusted.sum()),
                    accepted_within5_fraction=ratio((trusted&correct&kept).sum(),(trusted&kept).sum()),
                    accepted_correct_fraction_of_all_trusted=ratio((trusted&correct&kept).sum(),trusted.sum()),
                    correct_retention=ratio((trusted&correct&kept).sum(),(trusted&correct).sum()),
                    wrong_rejection=ratio((trusted&~correct&~kept).sum(),(trusted&~correct).sum()))
            gated=gate_probabilities(torch.from_numpy(prob[None]),torch.from_numpy(obs[None]),p['threshold'])[0].numpy()
            decoded=decode_path(gated,slope,entry['geometry'],a['valid_mask'])
            path=base/'evaluation/predictions'/f'{number:02d}_{kind}.npz'
            save(path,input_rgb=v['input'][:3],probability=prob,observable=obs,gated_probability=gated,
                valid_mask=a['valid_mask'],trusted_columns=a['trusted_columns'],target_mean_row=a['target_mean_row'])
            item.update(prediction_path=str(path),prediction_sha256=sha(path),segments=decoded['segments'],
                observed_columns=decoded['candidates'][0]['observed_columns'] if decoded['candidates'] else 0,
                geometry=entry['geometry'],identity_confirmed=False)
            row['variants'][kind]=item
        rows.append(row);print(json.dumps(dict(stage='eval',image=number+1,total=8)),flush=True)
    aggregate={kind:merge_counts([r['variants'][kind] for r in rows]) for kind in ('original','zero_query','synthetic_occlusion')}
    correct=sum(r['variants']['original']['correct_baseline'] for r in rows)
    wrong=sum(r['variants']['original']['wrong_baseline'] for r in rows)
    retained=sum(r['variants']['original']['correct_retained'] for r in rows)
    rejected=sum(r['variants']['original']['wrong_rejected'] for r in rows)
    aggregate['original'].update(correct_baseline=correct,wrong_baseline=wrong,correct_retained=retained,wrong_rejected=rejected,
        baseline_within5_fraction=ratio(correct,correct+wrong),
        accepted_within5_fraction=ratio(retained,aggregate['original']['retained_positive']),
        accepted_correct_fraction_of_all_trusted=ratio(retained,correct+wrong),
        correct_retention=ratio(retained,correct),wrong_rejection=ratio(rejected,wrong))
    challenges=[]
    for entry in index['challenges']:
        a=load_input(entry);f,prob,slope=features(model,a['input'])
        with torch.no_grad():obs=torch.sigmoid(head(torch.from_numpy(f[None])))[0].numpy()
        gated=gate_probabilities(torch.from_numpy(prob[None]),torch.from_numpy(obs[None]),p['threshold'])[0].numpy()
        before=decode_path(prob,slope,entry['geometry'],a['valid_mask'])
        after=decode_path(gated,slope,entry['geometry'],a['valid_mask'])
        path=base/'evaluation/challenges'/(entry['name']+'.npz')
        save(path,input_rgb=a['input'][:3],probability=prob,observable=obs,gated_probability=gated,valid_mask=a['valid_mask'])
        challenges.append(dict(**entry,prediction_path=str(path),prediction_sha256=sha(path),
            before_segments=before['segments'],after_segments=after['segments'],
            post_seed_column_retention=ratio((obs[43:]>=p['threshold']).sum(),len(obs[43:])),
            post_seed_not_physical_contact=True,identity_confirmed=False,accuracy_score=None))
    report=dict(schema='v23_visibility_evaluation_1',status='completed',protocol_sha256=sha(base/'protocol.json'),
        train_report_sha256=sha(base/'train/report.json'),checkpoint_sha256=sha(ck_path),
        aggregate=aggregate,rows=rows,challenges=challenges,
        inference_ms=dict(median=float(np.median(timings)),maximum=float(np.max(timings)),
            scope='cached strips + frozen CPU backbone + head; excludes camera, automatic seed, strip preparation, decode'),
        metrics_warning='Trusted strip-column tests, not arc-length accuracy or confirmed same-rope identity',
        development_not_blind=True,real_occlusion_validated=False,physical_events_evaluated=False,
        next_action='No parameter search; report gaps and keep offline',hardware_access=False)
    write(base/'evaluation/report.json',report);verify(base)
    print(json.dumps(dict(stage='evaluated',aggregate=aggregate,challenges=[dict(name=c['name'],retention=c['post_seed_column_retention']) for c in challenges])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','verify','prepare','train','evaluate'))
    parser.add_argument('--base',type=Path,default=BASE)
    args=parser.parse_args()
    {'freeze':freeze,'verify':verify,'prepare':prepare,'train':train,'evaluate':evaluation}[args.command](args.base.resolve())
