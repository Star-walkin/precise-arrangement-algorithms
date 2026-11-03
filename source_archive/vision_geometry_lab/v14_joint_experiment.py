"""Fresh matched V14 center-only versus auxiliary-boundary training trial.

Torch and CV stages use separate existing runtimes. No physical control.
Both arms share all initial parameters and training sequence. Only auxiliary
edge loss weight differs; synthetic examples retain the old center+slope task.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import time
import numpy as np

from .v11_path_experiment import (BASE as V11, MANIFEST, SAMPLE, TARGET_KEYS,
    read, sha, write_json, write_npz, verify as verify_v11)

LAB=Path(__file__).resolve().parent
BASE=LAB/'outputs/v14_joint_supervision'
TARGETS=LAB/'outputs/v13_boundary_targets_train'
ARMS=('center_only','joint')
SOURCES=('v14_joint_experiment.py','v14_joint_boundary_model.py','v14_boundary_metrics.py',
    'v14_reference_support.py','v13_boundary_targets.py','v11_path_model.py','v11_path_decode.py',
    'v11_path_contract.py','v11_identity_probe.py','v11_path_experiment.py',
    'evaluate_centerline_multiscale_v9.py','centerline_review_v5.py','centerline_quality_v5.py',
    'centerline_path_metrics_v6.py','centerline_quality_cutoff_v8.py','centerline_multiscale_v9.py')


def location(output,phase,arm):
    return output/('single' if phase=='single' else 'development')/arm


def freeze(output):
    if output.exists():raise FileExistsError('Never overwrite existing trial')
    old=verify_v11(V11)
    tprotocol=read(TARGETS/'protocol.json');tsummary=read(TARGETS/'summary.json')
    if tsummary['protocol_sha256']!=sha(TARGETS/'protocol.json'):raise ValueError('Target provenance mismatch')
    for category in ('source_snapshot','artifact_snapshot'):
        for path,digest in tprotocol[category].items():
            if sha(path)!=digest:raise ValueError('Changed V13 target source')
    artifacts={str(p):sha(p) for p in (MANIFEST,V11/'protocol.json',V11/'development_cache/index.json',V11/'development/evaluation.json',
        TARGETS/'protocol.json',TARGETS/'summary.json')}
    for row in tsummary['rows']:
        if sha(row['target_npz'])!=row['target_npz_sha256']:raise ValueError('Targets changed')
        artifacts[row['target_npz']]=row['target_npz_sha256']
    manifest=read(MANIFEST)
    if '20260925' in json.dumps(manifest):raise ValueError('Reserved blind set prohibited')
    protocol=dict(version=14,frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        arms=list(ARMS),seed=old['seed'],base_channels=old['base_channels'],cpu_threads=2,
        single_steps=1200,development_steps=2400,learning_rate=.001,batch_size=2,
        optimizer=dict(name='Adam',betas=[.9,.999],eps=1e-8,weight_decay=0.,amsgrad=False,clip_norm=5.),
        edge_weight=dict(center_only=0.,joint=.5),
        objective='unchanged V11 center CE + 0.1 slope; auxiliary .5*CE averaged across all valid side/column entries',
        no_width_loss=True,no_texture_features=True,unchanged_center_decoder=True,
        initialization='both arms fresh from same seed for each stage, including new edge heads; never warm start development',
        sample_sequence='alternate round-robin real duplicated batch and generated same-RGB two-prompt pair',
        synthetic_train_seed_base=old['synthetic_train_seed_base'],synthetic_test_seeds=old['synthetic_test_seeds'],
        synthetic_edges_unknown=True,synthetic_primary_loss_includes_slope=True,
        single_sample=SAMPLE,gate=dict(center_min=.90,center_max_drop=.02,synthetic_min=.90,
            each_edge_hit_min=.85,width_mean_relative_max=.25,crossing_max=.05,
            strict_human_center_bracket_min=.85,width_prediction_fraction_min=.99,
            each_edge_reference_support_min=.90),
        gate_scope='single then train-only mechanism prerequisite, NOT industrial acceptance',
        edge_gate_scope='trusted seed-downstream columns; full trusted edge arc support separately required',
        final_checkpoint_only=True,heldout_loaded_after_train_gate=True,
        heldout_is_development_not_blind=True,manual_seed_still_required=True,
        artifact_snapshot=artifacts,source_snapshot={str(LAB/p):sha(LAB/p) for p in SOURCES},
        hardware_access=False,force_command=None)
    write_json(output/'protocol.json',protocol)
    for phase in ('single','train','holdout'):
        chosen=[s for s in manifest['samples'] if s['sample_id']==SAMPLE] if phase=='single' else [s for s in manifest['samples'] if s['split']==phase]
        write_json(output/(phase+'_manifest.json'),dict(manifest,samples=chosen))
    print(json.dumps(dict(protocol=str(output/'protocol.json'),sha256=sha(output/'protocol.json'))))


def verify(output):
    verify_v11(V11)
    p=read(output/'protocol.json')
    for category in ('artifact_snapshot','source_snapshot'):
        for path,digest in p[category].items():
            if sha(path)!=digest:raise ValueError('Frozen source/artifact changed: '+path)
    original=read(MANIFEST)
    for phase in ('single','train','holdout'):
        expected=[s for s in original['samples'] if s['sample_id']==SAMPLE] if phase=='single' else [s for s in original['samples'] if s['split']==phase]
        if read(output/(phase+'_manifest.json'))!=dict(original,samples=expected):raise ValueError('Scope changed')
    return p


def load_inputs(output,phase,with_targets=False):
    """Select IDs BEFORE opening cache arrays, so early stages never load heldout."""
    selected=read(output/(phase+'_manifest.json'))['samples']
    index={e['sample_id']:e for e in read(V11/'development_cache/index.json')['entries']}
    targets={r['sample_id']:r for r in read(TARGETS/'summary.json')['rows']}
    loaded=[]
    for sample in selected:
        e=index[sample['sample_id']]
        if not e['available']:raise ValueError('Expected baseline input unavailable; do not drop sample')
        for key in ('image_sha256','annotation_sha256','quality_sha256','split','group_id'):
            if e[key]!=sample[key]:raise ValueError('Input source mismatch: '+key)
        if sha(e['cache'])!=e['cache_sha256']:raise ValueError('Changed cached input')
        with np.load(e['cache'],allow_pickle=False) as ar: a={k:ar[k].copy() for k in ar.files}
        if hashlib.sha256(a['input'].tobytes()).hexdigest()!=e['inference_input_sha256']:raise ValueError('Input tensor mismatch')
        if with_targets:
            if sample['split']!='train':raise ValueError('Holdout may not supply training targets')
            t=targets[sample['sample_id']]
            if sha(t['target_npz'])!=t['target_npz_sha256']:raise ValueError('Changed target cache')
            with np.load(t['target_npz'],allow_pickle=False) as ar: ta={k:ar[k].copy() for k in ar.files}
            for key in TARGET_KEYS:
                if not np.array_equal(a[key],ta[key]):raise ValueError('Center target changed')
            a.update(edge_target_distribution=ta['edge_distribution'],edge_column_valid=ta['edge_valid'],edge_row_valid=ta['edge_row_valid'])
        loaded.append((e,a))
    if len(loaded)!=dict(single=1,train=10,holdout=8)[phase]:raise ValueError('Wrong stage sample count')
    return loaded


def require_gate(output,phase):
    record=read(output/('gate_'+phase+'.json'))
    if record['protocol_sha256']!=sha(output/'protocol.json'):raise ValueError('Foreign gate')
    for arm in ARMS:
        path=location(output,phase,arm)/('evaluation_'+phase+'.json')
        if record['evaluation_sha256'][arm]!=sha(path):raise ValueError('Stale gate')
    calculated=gate_decision(output,phase)
    if record['decision']!=calculated or not calculated['passed']:raise ValueError('Mechanism gate did not pass')


def train(output,phase,arm):
    p=verify(output)
    if phase not in ('single','train'):raise ValueError('Cannot train on holdout')
    if phase=='train':require_gate(output,'single')
    folder=location(output,phase,arm)
    if folder.exists():raise FileExistsError('Never overwrite or silently resume training')
    import torch
    from .v14_joint_boundary_model import build_model,joint_path_loss
    from .v11_identity_probe import make_pair
    from .centerline_multiscale_v9 import _state_dict_sha256
    torch.set_num_threads(p['cpu_threads'])
    real=load_inputs(output,phase,True)
    model=build_model(p['seed'],p['base_channels'])
    initial=_state_dict_sha256(model.state_dict())
    opt=torch.optim.Adam(model.parameters(),lr=p['learning_rate'],betas=tuple(p['optimizer']['betas']),
        eps=p['optimizer']['eps'],weight_decay=0.,amsgrad=False)
    steps=p['single_steps'] if phase=='single' else p['development_steps']
    sequence=[];log=[];last_real=None;started=time.perf_counter();model.train()
    for step in range(steps):
        synthetic=step%2==1
        if synthetic:
            pair=make_pair(p['synthetic_train_seed_base']+step//2)
            x=pair['inputs']
            batch={k:torch.from_numpy(np.stack([t[k] for t in pair['targets']])).float() for k in TARGET_KEYS}
            n,_,h,w=x.shape
            batch.update(edge_target_distribution=torch.zeros((n,2,h,w)),
                edge_row_valid=torch.zeros((n,2,h,w)),edge_column_valid=torch.zeros((n,2,w)))
            sid='synthetic:'+str(p['synthetic_train_seed_base']+step//2)
        else:
            e,a=real[(step//2)%len(real)];x=np.repeat(a['input'][None],2,axis=0)
            batch={k:torch.from_numpy(np.repeat(a[k][None],2,axis=0)).float() for k in
                (*TARGET_KEYS,'edge_target_distribution','edge_row_valid','edge_column_valid')}
            sid=e['sample_id']
        sequence.append(sid);opt.zero_grad(set_to_none=True)
        losses=joint_path_loss(model(torch.from_numpy(x).float()),batch,edge_weight=p['edge_weight'][arm])
        if not torch.isfinite(losses['loss']):raise RuntimeError('Nonfinite loss')
        losses['loss'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),p['optimizer']['clip_norm']);opt.step()
        if not synthetic:
            last_real=dict(step=step+1,primary=float(losses['primary_path_loss'].detach()),
                edge=float(losses['edge_ce_loss'].detach()))
        if step==0 or (step+1)%100==0:
            row=dict(step=step+1,loss=float(losses['loss'].detach()),primary=float(losses['primary_path_loss'].detach()),
                edge=float(losses['edge_ce_loss'].detach()),synthetic=synthetic,last_real=last_real,
                seconds=time.perf_counter()-started)
            log.append(row);print(json.dumps(dict(phase=phase,arm=arm,**row)),flush=True)
    folder.mkdir(parents=True)
    checkpoint=folder/'checkpoint.pt'
    torch.save(dict(state_dict=model.state_dict(),phase=phase,arm=arm,steps=steps,protocol_sha256=sha(output/'protocol.json')),checkpoint)
    write_json(folder/'report.json',dict(phase=phase,arm=arm,steps=steps,
        initial_state_sha256=initial,final_state_sha256=_state_dict_sha256(model.state_dict()),
        sequence_sha256=hashlib.sha256(json.dumps(sequence).encode()).hexdigest(),
        input_sha256={e['sample_id']:e['inference_input_sha256'] for e,a in real},
        cache_sha256={e['sample_id']:e['cache_sha256'] for e,a in real},
        train_ids=[e['sample_id'] for e,a in real],optimizer=p['optimizer'],learning_rate=p['learning_rate'],
        loss_log=log,torch_version=str(torch.__version__),numpy_version=str(np.__version__),
        parameters=sum(v.numel() for v in model.parameters()),elapsed_seconds=time.perf_counter()-started,
        checkpoint_sha256=sha(checkpoint),protocol_sha256=sha(output/'protocol.json'),hardware_access=False))
    verify(output)


def checked_report(output,phase,arm):
    folder=location(output,phase,arm);r=read(folder/'report.json')
    protocol=read(output/'protocol.json')
    expected_phase='single' if phase=='single' else 'train'
    if r['phase']!=expected_phase or r['arm']!=arm or r['protocol_sha256']!=sha(output/'protocol.json'):
        raise ValueError('Wrong training provenance')
    if r['checkpoint_sha256']!=sha(folder/'checkpoint.pt'):raise ValueError('Checkpoint changed')
    if r['steps']!=protocol['single_steps' if phase=='single' else 'development_steps']:
        raise ValueError('Wrong fixed training budget')
    return r


def export(output,phase,arm):
    p=verify(output)
    if phase=='holdout':require_gate(output,'train')
    if phase=='train':require_gate(output,'single')
    folder=location(output,phase,arm);r=checked_report(output,phase,arm)
    destination=folder/('predictions_'+phase)
    if destination.exists():raise FileExistsError('Cannot overwrite inference')
    import torch
    from .v14_joint_boundary_model import build_model,image_masked_edge_probabilities
    from .v11_path_model import mask_normalized_probabilities
    from .v11_identity_probe import make_pair,assess_pair
    torch.set_num_threads(p['cpu_threads'])
    checkpoint=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
    if checkpoint['protocol_sha256']!=sha(output/'protocol.json') or checkpoint['arm']!=arm:
        raise ValueError('Wrong model')
    model=build_model(p['seed'],p['base_channels']);model.load_state_dict(checkpoint['state_dict']);model.eval()
    rows=[];probes=[]
    with torch.inference_mode():
        for e,a in load_inputs(output,phase):
            started=time.perf_counter();out=model(torch.from_numpy(a['input'][None]).float())
            valid=torch.from_numpy(a['valid_mask'][None])
            center=mask_normalized_probabilities(out,valid)[0].numpy()
            edges=image_masked_edge_probabilities(out,valid)[0].numpy()
            elapsed=(time.perf_counter()-started)*1000
            path=destination/(e['sample_id']+'.npz')
            write_npz(path,probability=center,slope=out['slope'][0].numpy(),edge_probability=edges)
            rows.append(dict(sample_id=e['sample_id'],path=str(path),sha256=sha(path),
                input_sha256=e['inference_input_sha256'],forward_ms=elapsed))
        for seed in p['synthetic_test_seeds']:
            pair=make_pair(seed,test=True);out=model(torch.from_numpy(pair['inputs']))
            probability=mask_normalized_probabilities(out,torch.from_numpy(pair['inputs'][:,7])).numpy()
            probes.append(dict(seed=seed,**assess_pair(probability,pair)))
    write_json(folder/('exports_'+phase+'.json'),dict(phase=phase,arm=arm,records=rows,synthetic_probes=probes,
        synthetic_pair_pass_fraction=float(np.mean([r['both_queries_pass'] for r in probes])),
        protocol_sha256=sha(output/'protocol.json'),checkpoint_sha256=r['checkpoint_sha256'],
        training_report_sha256=sha(folder/'report.json'),gt_used_for_inference=False))
    verify(output)


def seed_downstream_targets(targets,geometry):
    result={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in targets.items()}
    cols=geometry['u_first_widths']+np.arange(geometry['shape_hw'][1])*geometry['step_widths']>=0
    for key in ('center_valid','width_valid'):result[key] &= cols
    result['edge_valid'] &= cols[None]
    return result


def evaluate(output,phase,arm):
    verify(output)
    if phase=='holdout':require_gate(output,'train')
    from .evaluate_centerline_multiscale_v9 import _dataset,score_decoded,aggregate
    from .v11_path_decode import decode_path
    from .v13_boundary_targets import targets_for_annotation
    from .v14_boundary_metrics import score_boundaries,aggregate_boundary_scores
    from .v14_reference_support import boundary_reference_support
    folder=location(output,phase,arm);checked_report(output,phase,arm)
    exports=read(folder/('exports_'+phase+'.json'))
    if exports['phase']!=phase or exports['arm']!=arm or exports['protocol_sha256']!=sha(output/'protocol.json'):
        raise ValueError('Wrong inference stage')
    if exports['checkpoint_sha256']!=sha(folder/'checkpoint.pt') or exports['training_report_sha256']!=sha(folder/'report.json'):
        raise ValueError('Inference source changed')
    cache={e['sample_id']:(e,a) for e,a in load_inputs(output,phase)}
    predictions={r['sample_id']:r for r in exports['records']}
    if len(predictions)!=len(exports['records']) or set(predictions)!=set(cache):
        raise ValueError('Prediction records must cover exactly the selected stage without duplicates')
    _,records,_=_dataset(output/(phase+'_manifest.json'))
    # Prior support is a source-image/input-domain property, not model output.
    old_support={r['sample_id']:r['coverage_support'] for r in read(V11/'development/evaluation.json')['rows']}
    rows=[]
    for record in records:
        s,l,q=record['sample'],record['label'],record['quality'];name=s['sample_id']
        e,a=cache[name];pred=predictions[name]
        if sha(pred['path'])!=pred['sha256'] or pred['input_sha256']!=e['inference_input_sha256']:
            raise ValueError('Prediction/input mismatch')
        with np.load(pred['path'],allow_pickle=False) as ar:
            decoded=decode_path(ar['probability'],ar['slope'],e['geometry'],a['valid_mask'])
            targets=targets_for_annotation(e['geometry'],l,q,a['valid_mask'])
            boundary=score_boundaries(ar['edge_probability'],a['valid_mask'],seed_downstream_targets(targets,e['geometry']),e['geometry'])
        support=boundary_reference_support(l,q,e['geometry'],a['valid_mask'],targets)
        score=score_decoded(l,q,decoded,None,s['image_shape_hw'])
        score['coverage_support']=old_support[name]
        rows.append(dict(sample_id=name,split=s['split'],group_id=s['group_id'],cohort=record['cohort'],
            prediction_status='loaded',independent_width=e['width'],decoded=decoded,**score,
            boundary=boundary,boundary_reference_support=support,target_diagnostics=targets['diagnostics']))
    result=dict(phase=phase,arm=arm,rows=rows,summaries=aggregate(rows),
        boundary_summary=aggregate_boundary_scores([r['boundary'] for r in rows]),
        synthetic_pair_pass_fraction=exports['synthetic_pair_pass_fraction'],
        protocol_sha256=sha(output/'protocol.json'),exports_sha256=sha(folder/('exports_'+phase+'.json')),
        checkpoint_sha256=sha(folder/'checkpoint.pt'),event_timing_validated=False,force_command=None)
    write_json(folder/('evaluation_'+phase+'.json'),result)
    verify(output)
    print(json.dumps(dict(phase=phase,arm=arm,center=[r for r in result['summaries'] if r['cohort']=='all' and r['scope']=='downstream' and r['samples']],
        edge_hits={k:v['hit_5px_fraction'] for k,v in result['boundary_summary']['edges'].items()},
        width_relative=result['boundary_summary']['width']['mean_absolute_relative_error']),ensure_ascii=False))


def gate_decision(output,phase):
    p=verify(output);rules=p['gate']
    values={arm:read(location(output,phase,arm)/('evaluation_'+phase+'.json')) for arm in ARMS}
    reports={arm:checked_report(output,phase,arm) for arm in ARMS}
    for key in ('initial_state_sha256','sequence_sha256','input_sha256','cache_sha256','steps','optimizer','learning_rate','torch_version','numpy_version'):
        if reports['joint'][key]!=reports['center_only'][key]:raise ValueError('Unmatched arms: '+key)
    for arm,value in values.items():
        folder=location(output,phase,arm)
        if value['protocol_sha256']!=sha(output/'protocol.json') or value['checkpoint_sha256']!=sha(folder/'checkpoint.pt') or value['exports_sha256']!=sha(folder/('exports_'+phase+'.json')):
            raise ValueError('Stale evaluation')
    def center(arm):
        return next(s for s in values[arm]['summaries'] if s['split']=='train' and s['cohort']=='all' and s['scope']=='downstream')
    checks={}
    for metric in ('covered_fraction','prediction_precision_at_5px'):
        b,j=center('center_only')[metric],center('joint')[metric]
        checks['center_'+metric]=b is not None and j is not None and min(b,j)>=rules['center_min'] and j>=b-rules['center_max_drop']
    checks['synthetic']=all(v['synthetic_pair_pass_fraction']>=rules['synthetic_min'] for v in values.values())
    joint=values['joint']['boundary_summary']
    checks['each_edge_hit']=all(e['hit_5px_fraction'] is not None and e['hit_5px_fraction']>=rules['each_edge_hit_min'] for e in joint['edges'].values())
    width=joint['width'];pair=joint['joint']
    checks['width']=width['mean_absolute_relative_error'] is not None and width['mean_absolute_relative_error']<=rules['width_mean_relative_max']
    checks['width_prediction_support']=width['trusted_columns']>0 and width['predicted_trusted_columns']/width['trusted_columns']>=rules['width_prediction_fraction_min']
    checks['crossing']=pair['crossing_fraction_among_available'] is not None and pair['crossing_fraction_among_available']<=rules['crossing_max']
    checks['bracket']=pair['strict_center_bracket_fraction'] is not None and pair['strict_center_bracket_fraction']>=rules['strict_human_center_bracket_min']
    checks['reference_support']=reference_support_pass(values['joint']['rows'],rules['each_edge_reference_support_min'])
    return dict(passed=all(checks.values()),checks=checks,center={a:center(a) for a in ARMS},
        edge_hits={k:v['hit_5px_fraction'] for k,v in joint['edges'].items()},width_relative=width['mean_absolute_relative_error'],
        crossing=pair['crossing_fraction_among_available'],bracket=pair['strict_center_bracket_fraction'],
        synthetic={a:values[a]['synthetic_pair_pass_fraction'] for a in ARMS},deployment_allowed=False)


def reference_support_pass(rows,threshold):
    if not rows:return False
    if any(set(row['boundary_reference_support']['edges'])!={'edge_flange','edge_surface'} for row in rows):
        return False
    return all(edge['scorable_fraction_of_trusted'] is not None and
        edge['scorable_fraction_of_trusted']>=threshold for row in rows
        for edge in row['boundary_reference_support']['edges'].values())


def gate(output,phase):
    if phase not in ('single','train'):raise ValueError('No holdout-tuned gate')
    decision=gate_decision(output,phase)
    result=dict(protocol_sha256=sha(output/'protocol.json'),decision=decision,
        evaluation_sha256={a:sha(location(output,phase,a)/('evaluation_'+phase+'.json')) for a in ARMS})
    write_json(output/('gate_'+phase+'.json'),result)
    print(json.dumps(result,ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','verify','train','export','evaluate','gate'))
    parser.add_argument('--phase',choices=('single','train','holdout'),default='single')
    parser.add_argument('--arm',choices=ARMS,default='joint')
    parser.add_argument('--output',type=Path,default=BASE)
    a=parser.parse_args()
    if a.stage=='freeze':freeze(a.output)
    elif a.stage=='verify':verify(a.output);print('V14 frozen sources and artifacts verified')
    elif a.stage=='gate':gate(a.output,a.phase)
    else:globals()[a.stage](a.output,a.phase,a.arm)


if __name__=='__main__':main()
