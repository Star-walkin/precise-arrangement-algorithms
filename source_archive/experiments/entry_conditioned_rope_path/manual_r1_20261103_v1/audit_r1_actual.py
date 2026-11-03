"""Read-only actual R1 pair/path audit; never opens evaluation references."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
from pathlib import Path
from collections import Counter
import json
import time
import numpy as np
import r1_pipeline as r

HERE=Path(__file__).resolve().parent
def pair_audit():
    started=time.perf_counter();r.cv2.setNumThreads(1)
    _,spatial,_=r.original_modules();manifest=r.load(HERE/'relation/pair_manifest.json')
    index=r.load(HERE/'dataset/index.json');items={i['id']:i for i in index['items'] if i.get('origin')=='manual'}
    with np.load(HERE/'relation/pairs.npz') as z:stored=z['x'].copy();weights=z['weight'].copy()
    assert stored.shape==(len(manifest['rows']),2,197) and np.all(weights==1)
    compared=0;max_difference=0.;reports=[]
    for sample_id,item in items.items():
        records=[(i,row) for i,row in enumerate(manifest['rows']) if row['id']==sample_id]
        if not records:reports.append(dict(id=sample_id,pairs=0));continue
        path,annotation=r.annotation_for(item)
        response_path=HERE/'inference/fixed/U1/CONDITIONAL'/sample_id/'response.npz'
        meta,heat,features=r.cache_row(response_path)
        tracks,track_meta=r.independent_tracks(sample_id)
        ctx=spatial.Context(r.gray(item['image_path']),heat,features,item['known_seed_xy'],tracks)
        centers=r.ordered_centers(annotation['segments'],item['known_seed_xy']);sampled=[r.resample(p) for p in centers]
        red=[s['points_xy'] for s in annotation['segments'] if s['kind']=='neighbor_negative']
        with np.load(item['cache']) as z:valid=z['valid'];target=z['target']
        local_max=0.;rebuilt={}
        for array_index,row in records:
            p=sampled[row['line']];position=np.asarray(row['current_xy'])
            k=int(np.argmin(np.linalg.norm(p-position,axis=1)))
            assert np.linalg.norm(p[k]-position)<1e-7 and k>=18 and k+6<len(p)
            key=(row['line'],k)
            if key not in rebuilt:
                history=p[[k-18,k-12,k-6,k]]
                proposals=ctx.propose(history);q=np.asarray([p for p,_ in proposals])
                descriptor,_=ctx.descriptor(history,q)
                rebuilt[key]=(q,descriptor)
            q,descriptor=rebuilt[key]
            assert len(q)==row['full_proposal_pool_size']
            pi=int(np.argmin(np.linalg.norm(q-np.asarray(row['positive_xy']),axis=1)))
            ni=int(np.argmin(np.linalg.norm(q-np.asarray(row['negative_xy']),axis=1)))
            assert np.linalg.norm(q[pi]-row['positive_xy'])<1e-7 and np.linalg.norm(q[ni]-row['negative_xy'])<1e-7
            assert np.linalg.norm(q[pi]-p[k+6])<=4
            assert r.distance_to_segments([q[ni]],red)[0]<=6
            assert r.distance_to_segments([q[ni]],centers)[0]>=8
            grid=(q[[pi,ni]]-[0,1050])*.5
            assert np.all(spatial.sample(valid.astype(np.float32),grid)>.99)
            values=spatial.sample(target,grid)
            assert values[0]>.1 and values[1]<.1
            computed=np.stack([descriptor[pi],descriptor[ni]])
            difference=float(np.max(np.abs(computed-stored[array_index])))
            assert np.array_equal(computed,stored[array_index]),(sample_id,array_index,difference)
            local_max=max(local_max,difference);compared+=1
        max_difference=max(max_difference,local_max)
        reports.append(dict(id=sample_id,pairs=len(records),descriptor_max_abs_difference=local_max,
                            full_pool_descriptor_reconstructed=True,U1_sha256=meta['model_sha256']))
    assert compared==len(manifest['rows'])
    return dict(passed=True,pairs=compared,per_image=reports,maximum_descriptor_difference=max_difference,
                positive_generated_and_valid_target=True,negative_generated_and_manual_red=True,
                unknown_excluded=True,old_A_features_used=False,seconds=time.perf_counter()-started,
                only_training_annotations_read=True)

def paths_audit(kind):
    path=HERE/'paths'/(kind+'_receipt.json')
    if not path.exists():return dict(status='pending')
    receipt=r.load(path);assert receipt['exit_code']==0 and not receipt['errors']
    counts=Counter();ranks=[];selected_scores=[];points=0;inferred=0;supported=0;rejected=0
    by_arm={}
    for row in receipt['rows']:
        assert r.sha(row['path'])==row['sha256'];value=r.load(row['path'])
        assert value['metadata']['target_labels_read'] is False
        assert value['metadata']['evaluation_references_read'] is False
        assert value['metadata']['old_A_features_used'] is False
        arm=value['arm'];by_arm.setdefault(arm,Counter());by_arm[arm][value['status']]+=1
        counts[value['status']]+=1
        assert len(value['result']['candidates'])<=3
        selected=value['selected'];candidates=([selected] if selected else [])+value['alternatives']
        assert len(candidates)==len(value['result']['candidates'])
        if selected is None:rejected+=1
        for c in candidates:
            n=len(c['points_xy']);assert n==len(c['supported'])==len(c['point_status'])==len(c['inferred'])==len(c['entry_condition'])
            assert c['supported']==[s=='visible_supported' for s in c['point_status']]
            assert c['inferred']==[s=='inferred' for s in c['point_status']]
            assert all(not sup for sup,entry in zip(c['supported'],c['entry_condition']) if entry)
            q=np.asarray(c['points_xy']);assert np.isfinite(q).all()
            points+=n;inferred+=sum(c['inferred']);supported+=sum(c['supported']);ranks.append(c['rank_score'])
        if selected:selected_scores.append(selected['rank_score'])
        if value['metadata']['response_sha256']:
            assert r.sha(value['metadata']['response_path'])==value['metadata']['response_sha256']
        if arm=='R0':
            assert value['metadata']['relation_model_sha256'] is None
            assert not value['result']['relation_enabled']
        else:
            assert value['result']['relation_enabled']
        stat=value['result']['search']
        if stat:assert stat['generated']==stat['relation_scored_before_pruning']
    if kind=='fixed':
        gate=r.load(HERE/'paths/prediction_save_gate.json');assert gate['passed'] and gate['case_count']==48
        assert len(receipt['rows'])==48
    else:assert len(receipt['rows'])==301 and len({r['id'] for r in receipt['rows']})==292
    def distribution(values):
        if not values:return None
        return dict(minimum=float(np.min(values)),median=float(np.median(values)),maximum=float(np.max(values)),
                    p10=float(np.percentile(values,10)),p90=float(np.percentile(values,90)))
    return dict(passed=True,paths=len(receipt['rows']),status_counts=dict(counts),by_arm={k:dict(v) for k,v in by_arm.items()},
                rejected_with_prefix_or_missing_entry=rejected,candidate_point_total=points,
                visible_supported_point_proxies=supported,inferred_point_proxies=inferred,
                candidate_rank_distribution=distribution(ranks),selected_rank_distribution=distribution(selected_scores),
                rank_is_not_identity_probability=True,evaluation_references_read=False)

def main():
    target=HERE/'r1_actual_audit.json'
    pair=pair_audit()
    result=dict(pair=pair,fixed=paths_audit('fixed'),gallery=paths_audit('gallery'),
                audit_source_sha256=r.sha(__file__),pipeline_source_sha256=r.sha(HERE/'r1_pipeline.py'),
                note='Geometry/descriptor/schema checks only; not independent target identity accuracy')
    if target.exists():
        old=r.load(target)
        r.dump(HERE/'r1_actual_audit_previous.json',old)
    r.dump(target,result)
    print(json.dumps({k:({x:v for x,v in value.items() if x!='per_image'} if isinstance(value,dict) else value) for k,value in result.items()},ensure_ascii=False))
if __name__=='__main__':main()
