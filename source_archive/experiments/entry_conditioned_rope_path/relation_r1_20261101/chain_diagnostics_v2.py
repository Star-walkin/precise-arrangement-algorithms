"""Repair POSTHOC reachability semantics; never modify predictions or coverage.

Recognition excludes the input capsule. A reachability ROOT must instead test
the quality-clipped reference BEFORE that capsule is removed. A root outside
the partial reference is unassessable, not zero correct-candidate generation.
"""
from common import *
import sys,numpy as np
sys.path.insert(0,str(ROOT));sys.path.append(str(BASE))
from run_paths import load_gz
E=mod('_r1_chain_metrics_v2',BASE/'evaluate.py')
def main():
    old=load(HERE/'evaluation/summary.json');refs={};rows=[]
    for case in old['rows']:
        ident=case['sample_id']
        if ident not in refs:refs[ident]=E.data_eval.load_evaluation_reference(ident)
        ref=refs[ident];lines=E.clip_scoring_segments([ref['centerline']],ignore_polygons=ref['ignore_polygons'])
        raw=load_gz(case['prediction_files']['paths'])['result'];events=raw.get('events',[]);seed=np.asarray(case['source_reference']['actual_input_seed_xy']).reshape(-1,2)
        root_dist=float(E._nearest(seed[-1:],lines)[0][0]) if len(seed) else None
        root_ok=bool(root_dist is not None and root_dist<=5)
        root_status='reference_near_root' if root_ok else 'no_automatic_entry' if not len(seed) else 'outside_partial_reference_or_root_mismatch_unassessable'
        reachable={0:root_ok};connected=kept=local=0;drop={}
        if events:
            a=np.asarray([e['parent_xy'] for e in events]);b=np.asarray([e['xy'] for e in events]);t=np.linspace(0,1,5)
            pp=a[:,None]+(b-a)[:,None]*t[None,:,None];near=(E._nearest(pp.reshape(-1,2),lines)[0].reshape(-1,5).max(1)<=5)
            for e,n in zip(events,near):
                if n:local+=1;drop[e['drop_reason']]=drop.get(e['drop_reason'],0)+1
                ok=reachable.get(e['parent'],False) and bool(n);reachable[e['node']]=ok
                if ok:connected+=1;kept+=e['drop_reason']=='kept_beam'
        rows.append(dict(branch=case['branch'],mode=case['mode'],id=ident,root_status=root_status,root_reference_distance_px=root_dist,
            connected_reference_near_generated_nodes=connected if root_ok else None,connected_reference_near_kept_nodes=kept if root_ok else None,
            local_reference_near_edges_generated=local,local_drop_reasons=drop,
            local_union_is_not_connected_identity_path=True,prediction_sha256=case['prediction_files']['paths_sha256']))
    dump(HERE/'evaluation/chain_diagnostics_v2.json',dict(rows=rows,
        supersedes='only continuous_reference_near_chain_diagnostic fields in evaluation/summary.json; NOT its path coverage metrics',
        reason='v1 root test used input-excluded reference and produced misleading zero counts; absent reference at a root is unknown',
        predictions_and_scores_unchanged=True,recognition_capsule_exclusion_unchanged=True))
    print('Fixed posthoc chain root semantics. Assessable roots',sum(r['root_status']=='reference_near_root' for r in rows),'of32; coverage and outputs unchanged')
if __name__=='__main__':main()
