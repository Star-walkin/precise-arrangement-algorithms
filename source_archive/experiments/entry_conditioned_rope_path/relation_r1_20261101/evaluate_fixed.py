"""Frozen-path gate FIRST, then historical8 partial-reference scoring.

No predictions, thresholds, candidate generation, entries or training are changed.
"""
from common import *
import sys,numpy as np,time
sys.path.insert(0,str(ROOT));sys.path.append(str(BASE))
from run_paths import load_gz
E=mod('_r1_fixed_partial_evaluator',BASE/'evaluate.py')
import scene
def main():
    out=HERE/'evaluation';out.mkdir(exist_ok=False);ix=load(PAIR/'data/index.json');ev=[i for i in ix['items'] if i['split']=='eval'];cases=[]
    assert len(ev)==8 and set(i['id'] for i in ev)==set(E.data_eval.SAMPLE_IDS)
    selection=load(HERE/'selection_freeze.json');assert sha(HERE/'model/final_portable.npz')==selection['relation_sha256']
    # No target reference opens until ALL32 saved mode/branch records are hashed.
    for branch in ['R0','R1']:
        for mode in ['CONDITIONAL','AUTO']:
            for item in ev:
                p=HERE/'fixed_001'/branch/'fixed'/mode/(item['id']+'.json.gz');r=load_gz(p)
                assert r['job']['id']==item['id'] and r['job']['mode']==mode and r['target_labels_read'] is False
                res=r['result'];assert len(res['candidates'])<=3
                for c in res['candidates']:
                    q=np.asarray(c['points_xy']);assert len(q)==len(c['point_status'])==len(c['stage']) and np.isfinite(q).all()
                cache=Path(r['job']['cache_path']);dist=None
                if r['response_cache']['available']:
                    assert sha(cache)==r['response_cache']['sha256']
                    with np.load(cache) as z:a=z['response']
                    dist=dict(mean=float(a.mean()),std=float(a.std()),minimum=float(a.min()),maximum=float(a.max()),semantics='A750 center response, not identity probability')
                cases.append(dict(branch=branch,sample_id=item['id'],mode=mode,prediction=res,
                    scene=r['edge_target'],scoring_scene=load(item['scene_path']),actual_input_seed_xy=r['job']['seed_xy'],
                    response_distribution=dist,prediction_files=dict(paths=str(p),paths_sha256=sha(p),response=str(cache),response_sha256=r['response_cache'].get('sha256'))))
    dump(out/'prediction_save_gate.json',dict(passed=True,case_count=32,selection_sha256=sha(HERE/'selection_freeze.json'),
        files=[dict(branch=c['branch'],mode=c['mode'],id=c['sample_id'],**c['prediction_files']) for c in cases],labels_not_yet_opened=True))
    refs={};rows=[];tick=time.time()
    for c in cases:
        sid=c['sample_id']
        if sid not in refs:refs[sid]=E.data_eval.load_evaluation_reference(sid)
        score=E.evaluate_case(c,refs[sid],scene.stage_at_points);score['branch']=c['branch']
        selected=score['selected'];score['first_sustained_reference_gap_ge60px']=next((g for g in selected['all']['uncovered_runs'] if g['length_px']>=60),None)
        # Reachability audit with all candidate-parent links, vectorized distance.
        raw=load_gz(c['prediction_files']['paths'])['result'];events=raw.get('events',[])
        lines=score['source_reference']['trusted_clipped_centerline'];seed=np.asarray(c['actual_input_seed_xy']).reshape(-1,2)
        reachable={0:bool(len(seed) and E._nearest(seed[-1:],lines)[0][0]<=5)};drop={};generated=kept=0
        if events:
            a=np.asarray([e['parent_xy'] for e in events]);b=np.asarray([e['xy'] for e in events]);t=np.linspace(0,1,5)
            pp=a[:,None]+(b-a)[:,None]*t[None,:,None];near=(E._nearest(pp.reshape(-1,2),lines)[0].reshape(-1,5).max(1)<=5)
            for e,n in zip(events,near):
                ok=reachable.get(e['parent'],False) and bool(n);reachable[e['node']]=ok
                if ok:
                    generated+=1;kept+=e['drop_reason']=='kept_beam';drop[e['drop_reason']]=drop.get(e['drop_reason'],0)+1
        score['continuous_reference_near_chain_diagnostic']=dict(generated_nodes=generated,kept_nodes=kept,drop_reasons=drop,
            meaning='single parent-linked chain within5px partial reference; not union of disconnected local edges or physical identity truth')
        rows.append(score);dump(out/c['branch']/c['mode']/(sid+'.json'),score)
    summary={b:E._aggregate([r for r in rows if r['branch']==b]) for b in ['R0','R1']}
    dump(out/'summary.json',dict(exit_code=0,summary=summary,seconds=time.time()-tick,rows=rows,
        scope='same8 historical open evaluation images,3 groups; no full endpoint truth',eval_used_for_training_or_parameter_selection=False,
        same_candidates_and_rules_except_learned_relation_contribution=True,full_arc_identity_success=None))
    print(json.dumps({b:{m:{'coverage_all':s['selected_reference_coverage']['all']['coverage_5px'],
        'attached':s['selected_reference_coverage']['attached']['coverage_5px'],'edge_near':s['selected_reference_coverage']['edge_near']['coverage_5px'],
        'rejected':s['rejected_images']} for m,s in v.items()} for b,v in summary.items()},indent=2),flush=True)
if __name__=='__main__':main()
