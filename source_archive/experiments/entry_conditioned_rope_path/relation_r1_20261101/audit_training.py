"""Training-side posthoc diagnostics only, prior to eight-image evaluation."""
from common import *
import sys,numpy as np,gzip
sys.path.insert(0,str(ROOT));sys.path.append(str(BASE))
from run_paths import load_gz
E=mod('_r1_reused_partial_metrics',BASE/'evaluate.py')
import scene
def main():
    ix=load(PAIR/'data/index.json');rows=[]
    for it in ix['items']:
        if it['split']!='train':continue
        s=load(PAIR.parent/'run_20261029_001/data/label_sources'/(it['id']+'.json'))
        sc=load(it['scene_path']);inputpolys=E._entry_polygons(it['known_seed_xy'])
        lines=E.clip_scoring_segments(s['trusted_center_segments_xy'],ignore_polygons=inputpolys)
        pieces=E._reference_samples(lines,sc,scene.stage_at_points)
        for branch in ['R0','R1']:
            p=HERE/'development_001'/branch/'fixed/CONDITIONAL'/(it['id']+'.json.gz');rec=load_gz(p);result=rec['result']
            selected=next((c for c in result['candidates'] if c['id']==result['selected_candidate_id']),None)
            scored=E._score_candidate(selected,pieces,dict(ignore_polygons=[]),sc,inputpolys)
            pool=[E._score_candidate(c,pieces,dict(ignore_polygons=[]),sc,inputpolys) for c in result['candidates']]
            # Continuous reachable reference-near chain, not a union-of-edges claim.
            rf=np.concatenate([q['points'] for q in pieces]);root=np.asarray(result['entry_seed_xy'])[-1:]
            rootnear=bool(E._nearest(root,lines)[0][0]<=5)
            reachable={0:rootnear};reach=[];drops={};parent_near_generated=0
            for ev in result['events']:
                a=np.asarray(ev['parent_xy']);b=np.asarray(ev['xy']);segment=np.linspace(a,b,5)
                near=bool(E._nearest(segment,lines)[0].max()<=5)
                ok=reachable.get(ev['parent'],False) and near
                reachable[ev['node']]=ok
                if ok:
                    parent_near_generated+=1;reach.append(b);drops[ev['drop_reason']]=drops.get(ev['drop_reason'],0)+1
            first=next((x for x in scored['all']['uncovered_runs'] if x['length_px']>=60),None)
            rows.append(dict(id=it['id'],branch=branch,selected=scored,pool=pool,
                first_sustained_reference_gap_ge60px=first,continuous_reference_near_generated_nodes=parent_near_generated,
                continuous_reference_near_chain_drop_reasons=drops,
                full_correct_identity_candidate_asserted=False,path_sha256=sha(p)))
    aggregate={}
    for b in ['R0','R1']:
        rr=[r for r in rows if r['branch']==b];aggregate[b]={}
        for stage in ['all','free','transition','attached','edge_near']:
            vals=[r['selected']['all'] if stage=='all' else r['selected']['stages'][stage] for r in rr]
            length=sum(v['reference_length_px'] for v in vals);covered=sum(v['covered_reference_length_px'] or 0 for v in vals)
            aggregate[b][stage]=dict(reference_length_px=length,covered_length_px=covered,coverage=covered/length if length else None,
                longest_missed=max((v['longest_uncovered_run_px'] or 0 for v in vals),default=0))
    changes={b:aggregate[b]['attached']['coverage'] for b in aggregate}
    result=dict(rows=rows,aggregate=aggregate,training_only=True,scope='given current training partial reference; not independent identity accuracy',
        rule_decision='Do not change frozen H6 rules from mere path length. R1 is a separate finite candidate search; report R0 ablation and original H6 control separately.',
        candidate_upgrade='No next-point CNN: at all658 given-training-history local states a <=4px reference-next proposal exists. This does not imply reachability from actual entry.',
        relation_adoption='experimental, no baseline replacement; fixed600 model retained whether results improve or regress')
    dump(HERE/'training_diagnostics.json',result);print(json.dumps(aggregate,indent=2))
if __name__=='__main__':main()
