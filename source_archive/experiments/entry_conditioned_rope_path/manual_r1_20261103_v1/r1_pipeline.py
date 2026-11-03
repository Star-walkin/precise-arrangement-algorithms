"""Matched R1 on new manual centers/red neighbors and NEW U1 spatial features.

The original R1 descriptor/search are imported read-only with frozen parameters.
Only training-side reviewed manual labels construct pairs. Inference consumes
raw images, independent entry/scene/track geometry and saved U1 outputs only.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
from pathlib import Path
from collections import Counter
import argparse
import concurrent.futures
import gzip
import hashlib
import importlib.util
import json
import random
import sys
import time
import traceback

import numpy as np
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=BASE.parents[1]
LEGACY=BASE/'relation_r1_20261101'
H6=BASE/'postprocess_b_20261031_v6'
sys.path.append(str(ROOT/'vision_prototype/.venv/Lib/site-packages'))
import cv2

def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value
def load_gz(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)

_IMPORTED=None
def original_modules():
    global _IMPORTED
    if _IMPORTED is None:
        # Original modules resolve their pure profile helper through 'common'.
        # Each command/worker is an isolated process, not a training-thread import.
        common=module('common',LEGACY/'common.py')
        spatial=module('spatial',LEGACY/'spatial.py')
        search=module('search_r1',LEGACY/'search_r1.py')
        _IMPORTED=(common,spatial,search)
    return _IMPORTED

def source_freeze():
    _,spatial,_=original_modules()
    selection=load(HERE/'selected_checkpoints.json')['paths']['U1']
    value=dict(parameters=spatial.P,source_hashes={str(p):sha(p) for p in
        [LEGACY/'common.py',LEGACY/'spatial.py',LEGACY/'search_r1.py',BASE/'connect.py']},
        source_relation_model=dict(path=str(LEGACY/'model/final_portable.npz'),sha256=sha(LEGACY/'model/final_portable.npz')),
        U1_model=selection,dataset_index_sha256=sha(HERE/'dataset/index.json'),
        dimensions=197,feature_blocks=dict(dec1_spatial=48,heat=6,normalized_profiles=99,raw_statistics=15,geometry=21,F3=8),
        relation_training=dict(seed=20261103,layers=[197,64,32,1],formal_steps=600,formal_seconds=120,
            smallcheck_steps=100,smallcheck_seconds=60,optimizer='AdamW lr0.002 weight_decay0.0001',
            pair_loss='softplus(0.25-score_correct+score_wrong)',
            correct_proposal_distance_max_px=4,red_neighbor_distance_max_px=6,target_exclusion_distance_min_px=8,
            min_pairs=24,all_manual_training_sources=True,old10_relation_pairs_included=False,
            F3_dropout=False,unknown_is_never_negative=True,
            selection='last completed fixed budget update, never evaluation/gallery selected'))
    dest=HERE/'r1_freeze.json'
    if dest.exists():assert load(dest)==value,'R1 frozen inputs/parameters changed'
    else:dump(dest,value)
    return value

def cache_row(path):
    path=Path(path);row=load(path.with_suffix('.json'))
    selected=load(HERE/'selected_checkpoints.json')['paths']['U1']
    assert row['model_sha256']==selected['sha256'] and row['branch']=='U1'
    assert row['response_sha256']==sha(path),'U1 cache changed'
    assert 'new trained U1 dec1' in row['feature_source'] and row['target_labels_read'] is False
    if row['status']=='predicted':
        assert row['new_model_forward'] is True
        with np.load(path,allow_pickle=False) as z:heat=z['response'].copy();features=z['features'].copy()
        assert heat.shape==(975,2048) and heat.dtype==np.float32 and np.isfinite(heat).all()
        assert features.shape==(8,248,512) and features.dtype==np.float16 and np.isfinite(features).all()
        assert 0<=heat.min()<=heat.max()<=1
        return row,heat,features
    assert row['status']=='entry_unavailable'
    return row,None,None

def gray(path):
    image=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_GRAYSCALE)
    assert image.shape==(3000,4096)
    return image.astype(np.float32)/255.

def independent_tracks(sample_id):
    path=H6/'all_001/记录'/(sample_id+'.json.gz')
    if not path.exists():return [],dict(status='no_exact_saved_tracks',path=str(path))
    record=load_gz(path)
    tracks=record['surface_tracks']['tracks']
    return tracks,dict(status='exact_source_geometry',path=str(path),sha256=sha(path),track_count=len(tracks),
                        scope='surface tracks only; legacy selected/free/target paths not used')

def resample(points,step=4.):
    p=np.asarray(points,np.float64)
    cumulative=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    keep=np.r_[True,np.diff(cumulative)>1e-8];p=p[keep];cumulative=cumulative[keep]
    if len(p)<2:return p
    ss=np.arange(0,cumulative[-1]+1e-7,step)
    return np.c_[np.interp(ss,cumulative,p[:,0]),np.interp(ss,cumulative,p[:,1])]

def distance_to_segments(points,segments):
    q=np.asarray(points,np.float64).reshape(-1,2)
    best=np.full(len(q),np.inf)
    for points_xy in segments:
        p=np.asarray(points_xy,np.float64)
        for a,b in zip(p[:-1],p[1:]):
            ab=b-a;t=np.clip((q-a)@ab/max(float(ab@ab),1e-10),0,1)
            best=np.minimum(best,np.linalg.norm(q-(a+t[:,None]*ab),axis=1))
    return best

def ordered_centers(segments,seed):
    remaining=[np.asarray(s['points_xy'],np.float64) for s in segments if s['kind']=='target_center']
    previous=np.asarray(seed[-1],np.float64);output=[]
    while remaining:
        choices=[(float(np.linalg.norm(p[e]-previous)),i,e) for i,p in enumerate(remaining) for e in (0,-1)]
        _,i,e=min(choices);p=remaining.pop(i)
        if e==-1:p=p[::-1]
        output.append(p);previous=p[-1]
    return output

def annotation_for(item):
    paths=[item.get('annotation_snapshot_path'),item.get('annotation_path'),
           str(HERE/'label_audit/annotation_snapshot'/(item['id']+'.json'))]
    chosen=next((Path(p) for p in paths if p and Path(p).exists()),None)
    assert chosen is not None,'Missing frozen manual annotation '+item['id']
    assert 'annotation_snapshot' in str(chosen),'Use frozen manual snapshot, not live user-edited labels'
    a=load(chosen)
    assert a['sample_id']==item['id'] and a['image_sha256']==item['image_sha256']
    assert a['reviewed'] and a['label_source']=='manual_user'
    if item.get('annotation_sha256'):assert sha(chosen)==item['annotation_sha256']
    return chosen,a

def prepare_pairs():
    frozen=source_freeze();cv2.setNumThreads(1)
    _,spatial,_=original_modules();dest=HERE/'relation';dest.mkdir(exist_ok=False)
    items=[i for i in load(HERE/'dataset/index.json')['items'] if i.get('origin')=='manual']
    assert items and not {i['source_group'] for i in items}&set(load(HERE/'protocol.json')['eval_groups'])
    positive=[];negative=[];rows=[];diagnostics=[];missing=[];inputs={};started=time.perf_counter()
    for item in items:
        annotation_path,annotation=annotation_for(item);inputs[str(annotation_path)]=sha(annotation_path)
        path=HERE/'inference/fixed/U1/CONDITIONAL'/item['id']/'response.npz'
        meta,heat,features=cache_row(path);assert meta['status']=='predicted'
        assert meta['image_sha256']==item['image_sha256']==sha(item['image_path'])
        assert np.array_equal(np.asarray(meta['seed_xy']),np.asarray(item['known_seed_xy']))
        if item.get('cache_sha256'):assert sha(item['cache'])==item['cache_sha256']
        with np.load(item['cache'],allow_pickle=False) as z:valid=z['valid'];target=z['target']
        assert valid.shape==target.shape==(975,2048)
        tracks,track_meta=independent_tracks(item['id'])
        ctx=spatial.Context(gray(item['image_path']),heat,features,item['known_seed_xy'],tracks)
        centers=ordered_centers(annotation['segments'],item['known_seed_xy'])
        negatives=[s['points_xy'] for s in annotation['segments'] if s['kind']=='neighbor_negative']
        unknowns=[s['points_xy'] for s in annotation['segments'] if s['kind']=='uncertain']
        counts=Counter()
        for line_index,line in enumerate(centers):
            p=resample(line)
            # All four history points and future are within ONE reviewed fragment.
            for idx in range(18,len(p)-6,6):
                history=p[[idx-18,idx-12,idx-6,idx]];future=p[idx+6]
                proposals=ctx.propose(history);counts['states']+=1
                if not proposals:counts['empty_pool']+=1;continue
                q=np.asarray([v for v,_ in proposals]);grid=(q-[0,1050])*.5
                domain=spatial.sample(valid.astype(np.float32),grid)>.99
                explicit_unknown=distance_to_segments(q,unknowns)<=6 if unknowns else np.zeros(len(q),bool)
                known=domain&~explicit_unknown
                future_distance=np.linalg.norm(q-future,axis=1)
                values=spatial.sample(target,grid)
                correct=np.flatnonzero((future_distance<=4)&known&(values>.1))
                if not len(correct):
                    counts['missing_correct_proposal']+=1
                    missing.append(dict(id=item['id'],line=line_index,current_xy=history[-1].tolist(),future_xy=future.tolist(),
                                        candidate_xy=q.tolist(),reason='no valid generated proposal within4px; no oracle insertion'))
                    continue
                counts['correct_generated']+=1
                red_distance=distance_to_segments(q,negatives)
                target_distance=distance_to_segments(q,[x.tolist() for x in centers])
                low=values<.1
                wrong=np.flatnonzero(known&low&(red_distance<=6)&(target_distance>=8))
                if not len(wrong):counts['no_explicit_red_negative_in_pool']+=1;continue
                # Rank feature is computed over the COMPLETE proposal pool.
                descriptor,evidence=ctx.descriptor(history,q)
                assert descriptor.shape==(len(q),197)
                pi=int(correct[np.argmin(future_distance[correct])])
                # At most2 hard red neighbors, chosen by U1 heat, not by R1 score.
                for ni in sorted(wrong,key=lambda k:float(evidence['heat'][k]),reverse=True)[:2]:
                    positive.append(descriptor[pi]);negative.append(descriptor[ni])
                    rows.append(dict(id=item['id'],group=item['source_group'],kind='manual_target_vs_manual_red_neighbor',
                        current_xy=history[-1].tolist(),positive_xy=q[pi].tolist(),negative_xy=q[ni].tolist(),
                        positive_generated=True,negative_generated=True,full_proposal_pool_size=len(q),line=line_index,
                        positive_distance_to_future=float(future_distance[pi]),negative_distance_to_red=float(red_distance[ni]),
                        negative_distance_to_target=float(target_distance[ni]),
                        phase=int(next(s['stage'] for s in annotation['segments'] if s['kind']=='target_center'))))
                    counts['pairs']+=1
        diagnostics.append(dict(id=item['id'],group=item['source_group'],counts=dict(counts),tracks=track_meta,
                                annotation_path=str(annotation_path),annotation_sha256=sha(annotation_path),
                                U1_cache_path=str(path),U1_cache_sha256=sha(path),optical_width_px=ctx.width))
        print(dict(id=item['id'],counts=dict(counts)),flush=True)
    summary=dict(count=len(rows),images_with_pairs=len({r['id'] for r in rows}),groups_with_pairs=sorted({r['group'] for r in rows}),
        candidate_states=sum(d['counts'].get('states',0) for d in diagnostics),
        correct_generated_states=sum(d['counts'].get('correct_generated',0) for d in diagnostics),
        missing_correct_states=sum(d['counts'].get('missing_correct_proposal',0) for d in diagnostics),
        old10_relation_pairs_included=False,source_count=len(items),source_groups=sorted({i['source_group'] for i in items}))
    dump(dest/'pair_manifest.json',dict(rows=rows,diagnostics=diagnostics,input_hashes=inputs,summary=summary,
        dimensions=197,seconds=time.perf_counter()-started,evaluation_references_read=False,unknown_as_negative=False,
        complete_identity_accuracy=None,descriptor_uses_full_proposal_pool=True,freeze_sha256=sha(HERE/'r1_freeze.json')))
    dump(dest/'missing_candidates.json',dict(training_only=True,cases=missing,
        interpretation='Teacher-forced training candidate coverage, not automatic path coverage or identity accuracy'))
    assert len(rows)>=frozen['relation_training']['min_pairs'] and len(summary['groups_with_pairs'])>=2,'Insufficient legal manual pairs; inspect diagnosis without fabricating negatives'
    np.savez_compressed(dest/'pairs.npz',x=np.stack([positive,negative],axis=1).astype(np.float32),weight=np.ones(len(rows),np.float32))
    print('Prepared',len(rows),'manual pairs from',summary['images_with_pairs'],'images',flush=True)

def train():
    import torch
    from torch import nn
    frozen=source_freeze();cfg=frozen['relation_training'];torch.set_num_threads(2)
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    dest=HERE/'relation/model';dest.mkdir(exist_ok=False)
    pair_manifest=load(HERE/'relation/pair_manifest.json');rows=pair_manifest['rows']
    with np.load(HERE/'relation/pairs.npz') as z:raw=z['x'];weight=torch.from_numpy(z['weight'].copy())
    assert raw.shape==(len(rows),2,197) and np.isfinite(raw).all()
    mean=raw.reshape(-1,197).mean(0);spread=raw.reshape(-1,197).std(0)
    active=(spread>1e-5).astype(np.float32);std=np.maximum(spread,.05)
    x=torch.from_numpy(np.clip((raw-mean)/std*active,-12,12).astype(np.float32))
    class Ranker(nn.Module):
        def __init__(self):
            super().__init__();self.layers=nn.Sequential(nn.Linear(197,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
        def forward(self,v):return self.layers(v).squeeze(-1)
    net=Ranker();initial={k:v.clone() for k,v in net.state_dict().items()}
    def portable(path):
        arrays={k.replace('.','_'):v.detach().numpy() for k,v in net.state_dict().items()}
        np.savez_compressed(path,**arrays,mean=mean,std=std,active=active)
    def loss(ids):
        v=net(x[ids]);return (torch.nn.functional.softplus(.25-v[:,0]+v[:,1])*weight[ids]).sum()/weight[ids].sum()
    @torch.no_grad()
    def metrics(ids):
        scores=net(x[ids]);difference=scores[:,0]-scores[:,1]
        return dict(count=len(ids),loss=float(loss(ids)),pair_order_fraction=float((difference>0).float().mean()),mean_margin=float(difference.mean()))
    groups=sorted({r['group'] for r in rows});bygroup={g:np.flatnonzero([r['group']==g for r in rows]) for g in groups}
    small_ids=[]
    for g in groups:small_ids.extend(bygroup[g][::max(1,len(bygroup[g])//6)][:6].tolist())
    small_ids=np.asarray(small_ids,dtype=int);assert len(small_ids)>=8
    portable(dest/'initial_portable.npz');small_before=metrics(small_ids)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.0001);started=time.perf_counter();small_log=[]
    for step in range(1,cfg['smallcheck_steps']+1):
        optimizer.zero_grad();value=loss(small_ids);value.backward();optimizer.step()
        small_log.append(dict(step=step,loss=float(value.detach()),seconds=time.perf_counter()-started))
        if time.perf_counter()-started>=cfg['smallcheck_seconds']:break
    small_after=metrics(small_ids);passed=small_after['pair_order_fraction']>=.85 and small_after['loss']<small_before['loss']*.6
    dump(dest/'smallcheck.json',dict(before=small_before,after=small_after,passed=passed,steps=step,seconds=time.perf_counter()-started,
        indices=small_ids.tolist(),independent_copy_discarded=True,formal_reset_to_initial=True,log=small_log))
    assert passed,'R1 manual-pair smallcheck failed; do not expand budget/model to hide failure'
    net.load_state_dict(initial);optimizer=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.0001)
    rng=np.random.default_rng(cfg['seed']);all_ids=np.arange(len(rows));before=metrics(all_ids)
    started=time.perf_counter();seen=Counter();gradient_nonzero=set()
    with (dest/'train_log.jsonl').open('x',encoding='utf-8') as f:
        for step in range(1,cfg['formal_steps']+1):
            ids=np.asarray([rng.choice(bygroup[rng.choice(groups)]) for _ in range(64)],dtype=int)
            optimizer.zero_grad();value=loss(ids);value.backward()
            for name,p in net.named_parameters():
                if p.grad is not None and float(p.grad.abs().sum())>0:gradient_nonzero.add(name)
            optimizer.step()
            for i in ids:seen[rows[int(i)]['group']]+=1
            record=dict(step=step,loss=float(value.detach()),seconds=time.perf_counter()-started)
            f.write(json.dumps(record)+'\n');f.flush()
            if step==1 or step%100==0:print(record,flush=True)
            if record['seconds']>=cfg['formal_seconds']:break
    seconds=time.perf_counter()-started;after=metrics(all_ids)
    torch.save(dict(state=net.state_dict(),step=step,config=cfg,mean=mean,std=std,active=active),dest/'final.pt')
    portable(dest/'final_portable.npz')
    _,_,search=original_modules();portable_model=search.PortableRanker(dest/'final_portable.npz')
    with torch.inference_mode():expected=net(x[:16].reshape(-1,197)).numpy()
    actual=portable_model(raw[:16].reshape(-1,197))
    difference=float(np.max(np.abs(expected-actual)));assert difference<2e-5
    assert sha(frozen['U1_model']['path'])==frozen['U1_model']['sha256'],'U1 model changed during relation-only training'
    dump(dest/'training_receipt.json',dict(exit_code=0,steps=step,seconds=seconds,command=sys.argv,device='cpu',torch=torch.__version__,
        model_parameters=sum(p.numel() for p in net.parameters()),before=before,after=after,
        per_group={g:metrics(bygroup[g]) for g in groups},sampled_pairs_per_group=dict(seen),
        smallcheck_passed=passed,formal_reset_to_initial=True,gradient_nonzero_parameters=sorted(gradient_nonzero),
        active_feature_indices=np.flatnonzero(active).tolist(),inactive_feature_indices=np.flatnonzero(active==0).tolist(),
        portable_max_abs_difference=difference,pair_npz_sha256=sha(HERE/'relation/pairs.npz'),
        pair_manifest_sha256=sha(HERE/'relation/pair_manifest.json'),model_sha256=sha(dest/'final.pt'),portable_sha256=sha(dest/'final_portable.npz'),
        U1_frozen_for_relation_training=True,U1_sha256=frozen['U1_model']['sha256'],old10_relation_pairs_included=False,
        checkpoint_policy=cfg['selection'],training_pair_metrics_are_not_independent_accuracy=True))

_RANKERS=None
_CACHE_PRIOR_HASH=None
def worker_init():
    global _RANKERS,_CACHE_PRIOR_HASH
    cv2.setNumThreads(1);_,_,search=original_modules()
    _RANKERS=dict(R1_new=search.PortableRanker(HERE/'relation/model/final_portable.npz'),
                  R1_old=search.PortableRanker(LEGACY/'model/final_portable.npz'))
    repair=HERE/'repairs/gallery_role_001/cache_acceptance_001.json'
    if repair.exists():
        record=load(repair);assert record['repaired_pipeline_sha256']==sha(__file__)
        _CACHE_PRIOR_HASH=record['prior_pipeline_sha256']

def far_edge(job):
    # Ruler-photo IDs lack the video-frame '__left/right' suffix. Use the
    # pre-inference frozen source role, verified against the actual parent dir.
    source=next(j for j in load(HERE/'protocol.json')['gallery_jobs'] if j['id']==job['id'])
    assert source['camera_role'] in ('left','right')
    role=source['camera_role']+'_camera'
    assert Path(job['image_path']).parent.name==role
    path=ROOT/'数据资产/05_派生与实验结果/camera_condition_sam2_20261009_v1/all_001/body_branches_002/masks'/role/(job['id']+'.png')
    mask=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_GRAYSCALE)>0
    direction=np.asarray(job['seed_xy'][-1])-job['seed_xy'][0];points=[]
    for y in range(1052,2996,6):
        xx=np.flatnonzero(mask[y])
        if len(xx):points.append([int(xx[0] if direction[0]<0 else xx[-1]),y])
    return points,dict(path=str(path),sha256=sha(path),meaning='far-side visible independent surface silhouette; endpoint identity unconfirmed')

def flat_candidate(candidate):
    result={k:candidate[k] for k in ('id','points_xy','point_status','rank_score','termination','length_px','status_length_px')}
    result['supported']=[s=='visible_supported' for s in candidate['point_status']]
    result['inferred']=[s=='inferred' for s in candidate['point_status']]
    result['entry_condition']=[s=='entry_condition' for s in candidate['point_status']]
    if 'stage' in candidate:result['stage']=candidate['stage']
    result['complete_identity_verified']=False
    return result

def path_one(meta,arm,kind):
    tick=time.perf_counter();_,spatial,search=original_modules()
    key=(f'gallery/{meta["id"]}/entry_{meta["entry_index"]:02d}' if kind=='gallery'
         else f'fixed/{arm}/{meta["mode"]}/{meta["id"]}')
    dest=HERE/'paths'/(key+'.json')
    relation_path=(HERE/'relation/model/final_portable.npz' if arm=='R1_new' else LEGACY/'model/final_portable.npz')
    relation_sha=sha(relation_path) if arm!='R0' else None
    if dest.exists():
        existing=load(dest)
        assert existing['metadata']['response_sha256']==meta['response_sha256']
        assert existing['metadata']['relation_model_sha256']==relation_sha
        assert existing['metadata']['pipeline_source_sha256'] in {sha(__file__),_CACHE_PRIOR_HASH}
        assert existing['parameters']==spatial.P
        receipt=dict(existing['receipt']);receipt['sha256']=sha(dest)
        receipt['reused_existing_prediction']=True
        receipt['prediction_pipeline_source_sha256']=existing['metadata']['pipeline_source_sha256']
        return receipt
    assert sha(meta['image_path'])==meta['image_sha256']
    saved,heat,features=cache_row(meta['response_path']);assert saved==meta
    tracks=[];track_meta=dict(status='not_available_for_historical_fixed_source');edge=[];edge_meta=None;scene={}
    if kind=='gallery':
        tracks,track_meta=independent_tracks(meta['id']);edge,edge_meta=far_edge(meta)
    elif meta['status']=='predicted':
        index=load(BASE/'runs/run_20261030_002/data/index.json')
        item=next(x for x in index['items'] if x['id']==meta['id'])
        if meta['mode']=='AUTO':
            entry=load(item['auto_entry_path']);scene=load(entry['scene_path']) if entry.get('available') else {}
        else:scene=load(item['scene_path'])
        edge=scene.get('target_edge_points_xy',[])
        edge_meta=dict(scene_path=item['scene_path'],meaning='same frozen historical scene geometry as original R1; no center reference read')
    if meta['status']=='predicted':
        ctx=spatial.Context(gray(meta['image_path']),heat,features,meta['seed_xy'],tracks)
        result=search.run(ctx,_RANKERS['R1_new' if arm=='R0' else arm],relation=arm!='R0',edge=edge,trace=kind=='fixed')
        if scene:
            scene_module=module('_manual_r1_visual_scene',BASE/'scene.py')
            for c in result['candidates']:c['stage']=scene_module.stage_at_points(scene,np.asarray(c['points_xy'])).tolist()
    else:
        result=dict(candidates=[],selected_candidate_id=None,status='AUTOMATIC_ENTRY_UNAVAILABLE',ambiguous=False,
                    search={},seconds=0,relation_enabled=arm!='R0')
    candidates=[flat_candidate(c) for c in result['candidates']]
    selected=next((c for c in candidates if c['id']==result['selected_candidate_id']),None)
    alternatives=[c for c in candidates if selected is None or c['id']!=selected['id']]
    receipt=dict(id=meta['id'],mode=meta['mode'],entry_index=meta['entry_index'],arm=arm,status=result['status'],
        path=str(dest),available=meta['status']=='predicted',selected_candidate_id=result['selected_candidate_id'],
        candidate_count=len(candidates),seconds=time.perf_counter()-tick)
    record=dict(id=meta['id'],mode=meta['mode'],entry_index=meta['entry_index'],arm=arm,status=result['status'],
        selected=selected,alternatives=alternatives,best_candidate=candidates[0] if candidates else None,
        ambiguous=result['ambiguous'],result=result,receipt=receipt,scene=scene,
        metadata=dict(response_path=meta['response_path'],response_sha256=meta['response_sha256'],
            image_path=meta['image_path'],image_sha256=meta['image_sha256'],U1_model_path=meta['model_path'],U1_model_sha256=meta['model_sha256'],
            relation_model_path=str(relation_path) if arm!='R0' else None,relation_model_sha256=relation_sha,
            entry_seed_xy=meta['seed_xy'],entry_is_given_short_condition=True,
            source_group=meta['source_group'],source_split=meta['split'],surface_tracks=track_meta,edge_geometry=edge_meta,
            pipeline_source_sha256=sha(__file__),frozen_R1_sha256=sha(HERE/'r1_freeze.json'),
            selected_entry_not_truth_or_heat_ranked=True,unknown_endpoint_not_truth=True,
            target_labels_read=False,evaluation_references_read=False,old_A_features_used=False,
            stage_semantics='historical visual scene classification only; no physical contact truth' if scene else 'no gallery physical phase asserted',
            geometry_is_not_identity=True),parameters=spatial.P)
    dump(dest,record);receipt['sha256']=sha(dest)
    return receipt

def infer(kind,workers,role_map_repair=False):
    source_freeze();input_receipt=load(HERE/'inference'/kind/'receipt.json');assert input_receipt['exit_code']==0
    rows=input_receipt['rows']
    if kind=='fixed':
        rows=[r for r in rows if r['split']=='eval'];assert len(rows)==16
        arms=['R1_new','R1_old','R0']
    else:
        assert len(rows)==301 and len({r['id'] for r in rows})==292;arms=['R1_new']
    assert all(r['target_labels_read'] is False for r in rows)
    out=HERE/'paths';out.mkdir(exist_ok=True);started=time.perf_counter();results=[];errors=[]
    start_path=out/f'{kind}_run_start.json'
    if start_path.exists():
        previous=load(start_path)
        if previous['pipeline_source_sha256']!=sha(__file__):
            acceptance=load(HERE/'repairs/gallery_role_001/cache_acceptance_001.json')
            assert kind=='gallery' and role_map_repair
            assert acceptance['prior_pipeline_sha256']==previous['pipeline_source_sha256']
            assert acceptance['repaired_pipeline_sha256']==sha(__file__)
            dump(out/'gallery_run_start_repair_001.json',dict(command=sys.argv,expected=len(rows),
                pipeline_source_sha256=sha(__file__),acceptance_sha256=sha(HERE/'repairs/gallery_role_001/cache_acceptance_001.json'),
                original_run_start_preserved=True,reuse_successes_without_rewriting=True))
    else:dump(start_path,dict(command=sys.argv,expected=len(rows)*len(arms),arms=arms,keys=[(r['id'],r['mode'],r['entry_index']) for r in rows],
        pipeline_source_sha256=sha(__file__),freeze_sha256=sha(HERE/'r1_freeze.json'),labels_not_read=True))
    with concurrent.futures.ProcessPoolExecutor(workers,initializer=worker_init) as executor:
        pending={executor.submit(path_one,row,arm,kind):(row['id'],arm,row['mode'],row['entry_index']) for row in rows for arm in arms}
        for future in concurrent.futures.as_completed(pending):
            try:results.append(future.result())
            except Exception:errors.append(dict(job=pending[future],traceback=traceback.format_exc()))
            if (len(results)+len(errors))%10==0 or len(results)+len(errors)==len(pending):
                print(dict(done=len(results),errors=len(errors),total=len(pending),seconds=round(time.perf_counter()-started,1)),flush=True)
    receipt=dict(exit_code=int(bool(errors)),command=sys.argv,rows=results,errors=errors,planned=len(rows)*len(arms),
                 seconds=time.perf_counter()-started,pipeline_source_sha256=sha(__file__),evaluation_references_read=False)
    if role_map_repair:
        before=load(HERE/'repairs/gallery_role_001/cached_successes_before.json')
        assert all(sha(row['path'])==row['sha256'] for row in before['rows'])
        receipt['role_map_repair']=dict(prior_successful_predictions=len(before['rows']),
            predictions_reused=sum(bool(x.get('reused_existing_prediction')) for x in results),
            newly_completed_predictions=sum(not x.get('reused_existing_prediction',False) for x in results),
            prior_successful_prediction_bytes_unchanged=True,prior_failure_preserved=True,
            acceptance_path=str(HERE/'repairs/gallery_role_001/cache_acceptance_001.json'),
            acceptance_sha256=sha(HERE/'repairs/gallery_role_001/cache_acceptance_001.json'))
    dump(out/f'{kind}_receipt.json',receipt)
    if errors:print(errors[:1],flush=True);raise RuntimeError('R1 inference failures saved; do not omit cases')
    for row in results:
        assert sha(row['path'])==row['sha256']
        record=load(row['path']);assert len(record['result']['candidates'])<=3
        for c in record['result']['candidates']:
            q=np.asarray(c['points_xy']);assert np.isfinite(q).all() and len(q)==len(c['point_status'])
            assert (q[:,0]>=0).all() and (q[:,0]<4096).all() and (q[:,1]>=1050).all() and (q[:,1]<3000).all()
    if kind=='fixed':dump(out/'prediction_save_gate.json',dict(passed=True,case_count=48,rows=results,labels_not_yet_opened=True,
        fixed_receipt_sha256=sha(out/'fixed_receipt.json'),pipeline_source_sha256=sha(__file__),
        meaning='All three-arm fixed predictions saved and hashed before independent evaluation reference reads'))

def main():
    sys.stdout.reconfigure(encoding='utf-8');parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare-pairs','train','infer-fixed','infer-gallery'])
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--repair-gallery-role-map',action='store_true');args=parser.parse_args()
    if args.command=='prepare-pairs':prepare_pairs()
    elif args.command=='train':train()
    else:infer('fixed' if args.command=='infer-fixed' else 'gallery',args.workers,args.repair_gallery_role_map)

if __name__=='__main__':main()
