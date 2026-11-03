"""Finite entry-to-edge search, learned relation applied before EVERY prune.

There are no target annotations, oracle endpoints or truth-guided candidates in
this module. The history in child states consists exclusively of predictions.
"""
import numpy as np,math,time
from spatial import Context,P,unit,angles,wrap
from common import HERE,load
class PortableRanker:
    def __init__(self,path=None):
        self.z=dict(np.load(path or HERE/'model/final_portable.npz'))
    def __call__(self,x):
        z=self.z;a=np.clip((x-z['mean'])/z['std']*z['active'],-12,12)
        for k in [0,2]:
            a=a@z[f'layers_{k}_weight'].T+z[f'layers_{k}_bias'];a=a/(1+np.exp(-np.clip(a,-60,60)))
        return (a@z['layers_4_weight'].T+z['layers_4_bias']).ravel()
def quality(n):
    if not n['scores']:return -1.
    return .65*float(np.mean(n['scores'][-8:]))+.35*float(np.mean(n['scores']))+.08*min(n['support_length']/2500,1.)-.06*n['weak']/24
def run(ctx,ranker,relation=True,edge=None,trace=False):
    tick=time.time();seed=ctx.seed;count=max(2,int(np.ceil(np.linalg.norm(seed[-1]-seed[0])/24))+1)
    points=np.linspace(seed[0],seed[-1],count).tolist();initial_count=len(points)
    root=dict(node=0,parent=None,points=points,node_ids=[0]*count,status=['entry_condition']*count,scores=[],weak=0.,support_length=0.,bridge_length=0.,why='initial')
    beam=[root];finished=[];events=[];serial=1;stats=dict(generated=0,invalid_profile=0,weak_terminal=0,merged=0,beam_pruned=0,relation_scored_before_pruning=0)
    edge=np.asarray(edge if edge is not None else [],float).reshape(-1,2)
    for iteration in range(P['max_steps']):
        pool=[]
        for parent in beam:
            proposals=ctx.propose(parent['points'])
            if not proposals:
                n=dict(parent,why='no_valid_image_domain_proposal');finished.append(n);continue
            q=np.asarray([p for p,s in proposals]);x,ev=ctx.descriptor(parent['points'],q);rel=ranker(x)
            rel_center=rel-np.median(rel);relative=np.tanh(rel_center/3.)
            stats['generated']+=len(q);stats['relation_scored_before_pruning']+=len(q)
            for i,(pos,source) in enumerate(proposals):
                nid=serial;serial+=1;length=float(np.linalg.norm(pos-np.asarray(parent['points'][-1])))
                supported=bool(ev['supported'][i] and ev['coverage'][i]>=.99)
                weak=0. if supported else parent['weak']+length
                score=float(ev['quality'][i]+(P['relation_weight']*relative[i] if relation else 0.))
                status='visible_supported' if supported else 'inferred'
                child=dict(node=nid,parent=parent['node'],points=parent['points']+[pos.tolist()],node_ids=parent['node_ids']+[nid],status=parent['status']+[status],
                    scores=parent['scores']+[score],weak=weak,support_length=parent['support_length']+(length if supported else 0.),
                    bridge_length=parent['bridge_length']+(length if not supported else 0.),why='continue')
                reason=None
                if ev['coverage'][i]<.99:reason='profile_leaves_original_valid_image';stats['invalid_profile']+=1
                elif weak>P['max_weak_original_px']:
                    reason='unsupported_prefix_stopped';stats['weak_terminal']+=1
                    # Keep the real failure tail in trace, trim it from output.
                    k=len(child['points'])-1
                    while k>=initial_count and child['status'][k]=='inferred':k-=1
                    child['points']=child['points'][:k+1];child['status']=child['status'][:k+1]
                    child['node_ids']=child['node_ids'][:k+1];child['node']=child['node_ids'][-1]
                    child['scores']=child['scores'][:max(0,k+1-initial_count)];child['weak']=0.
                    child['why']=reason;finished.append(child)
                elif len(edge) and child['support_length']>240 and np.linalg.norm(edge-pos,axis=1).min()<=18:
                    reason='surface_edge_candidate_identity_unconfirmed';child['why']=reason;finished.append(child)
                else:pool.append(child)
                if trace:
                    events.append(dict(iteration=iteration,node=nid,parent=parent['node'],parent_xy=parent['points'][-1],xy=pos.tolist(),
                        source=source,base_score=float(ev['quality'][i]),relation_raw=float(rel[i]),relation_relative=float(relative[i]),
                        relation_contribution=float(P['relation_weight']*relative[i] if relation else 0),heat=float(ev['heat'][i]),bilateral=float(ev['bilateral'][i]),
                        template=float(ev['template'][i]),profile_valid=float(ev['coverage'][i]),supported=supported,score=score,drop_reason=reason))
        pool.sort(key=quality,reverse=True);beam=[];seen=set();kept=set();merged=set()
        for n in pool:
            p=np.asarray(n['points']);a=float(angles(p[-1]-p[-2]));key=(round(p[-1,0]/8),round(p[-1,1]/8),round(np.degrees(a)/12))
            if key in seen:stats['merged']+=1;merged.add(n['node']);continue
            seen.add(key)
            if len(beam)<P['beam']:beam.append(n);kept.add(n['node'])
            else:stats['beam_pruned']+=1
        if trace:
            for evn in events:
                if evn['iteration']==iteration and evn['drop_reason'] is None:
                    evn['drop_reason']='kept_beam' if evn['node'] in kept else 'merged_endpoint' if evn['node'] in merged else 'beam_budget'
        if len(finished)>80:finished=sorted(finished,key=quality,reverse=True)[:80]
        if not beam:break
    for n in beam:n['why']='fixed_search_budget_unresolved';finished.append(n)
    finished.sort(key=quality,reverse=True);chosen=[]
    for n in finished:
        if len(n['points'])<=initial_count:continue
        p=np.asarray(n['points']);distinct=True
        for o in chosen:
            q=np.asarray(o['points']);nn=min(len(p),len(q));tail=max(initial_count,nn-30)
            if nn>tail and np.median(np.linalg.norm(p[tail:nn]-q[tail:nn],axis=1))<max(12.,ctx.width*.5):distinct=False;break
        if distinct:chosen.append(n)
        if len(chosen)>=3:break
    candidates=[]
    for j,n in enumerate(chosen):
        pp=np.asarray(n['points']);ds=np.linalg.norm(np.diff(pp,axis=0),axis=1);statuses=np.asarray(n['status'])
        lengths={s:float(ds[(statuses[:-1]==s)&(statuses[1:]==s)].sum()) for s in ['entry_condition','visible_supported','inferred']}
        candidates.append(dict(id='c'+str(j),points_xy=n['points'],point_status=n['status'],rank_score=float(quality(n)),
            termination=n['why'],length_px=float(ds.sum()),status_length_px=lengths,last_node=n['node'],complete_identity_verified=False))
    margin=candidates[0]['rank_score']-candidates[1]['rank_score'] if len(candidates)>1 else None
    selected=bool(candidates and candidates[0]['status_length_px']['visible_supported']>=100)
    ambiguous=bool(margin is not None and margin<P['selected_margin'])
    return dict(candidates=candidates,selected_candidate_id='c0' if selected else None,entry_seed_xy=ctx.seed.tolist(),entry_counted_as_recognition=False,
        ambiguous=ambiguous,status='AMBIGUOUS' if selected and ambiguous else 'PREFIX_ONLY_OR_EDGE_CANDIDATE' if selected else 'REJECTED_WITH_PREFIX',
        relation_enabled=relation,relation_insertion='all proposed edges before pool merge and beam top8',search=stats,events=events,
        iterations=iteration+1,seconds=time.time()-tick,width_px=ctx.width,width_is_not_calibrated_diameter=True,
        full_target_identity_accuracy=None,unknown_endpoint_is_not_training_stop_truth=True)
