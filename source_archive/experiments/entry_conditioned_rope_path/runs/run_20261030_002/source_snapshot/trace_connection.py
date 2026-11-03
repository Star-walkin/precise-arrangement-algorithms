"""Posthoc passive tracing of the unchanged frozen decoder, never a new decoder.

Observe Python line events, save every scored proposal and its pool/beam fate.
Assert returned paths equal previously saved paths before reading reference for
stage-reachability diagnostics. These proximity diagnostics do not prove identity.
"""
from __future__ import annotations
import argparse,inspect,json,sys,time,hashlib
from pathlib import Path
import numpy as np
import connect

CODE=Path(__file__).resolve().parent;OLD=CODE/'runs/run_20261029_001'
CASES=['dual_20260923_113745__left__f00450','dual_20260923_113305__right__f00500']

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def observe(im,response,seed,scene,bbox,cfg):
    source,first=inspect.getsourcelines(connect.search)
    marker=lambda prefix:next(first+j for j,s in enumerate(source) if s.strip().startswith(prefix))
    node_line=marker('length=(len(n');pool_line=marker('pool.sort(');beam_line=marker('if not beam:break')
    proposal_line=marker('if not proposals:break')
    nodes=[];ids={};eligible=set();retained=set();events=[];term={};root=None
    def trace(frame,event,arg):
        nonlocal root
        if frame.f_code is not connect.search.__code__:return None
        if event!='line':return trace
        at=frame.f_lineno;v=frame.f_locals
        if at==proposal_line and root is None:
            root=v['root'];ids[id(root)]=0
        elif at==node_line:
            n=v['n'];parent=v['parent'];uid=len(nodes)+1;ids[id(n)]=uid;ev=n['evidence'][-1]
            assert id(parent) in ids,'Tracing lost an ancestor'
            nodes.append([uid,ids[id(parent)],v['iteration'],*n['points'][-1],n['angle'],n['scores'][-1],
              ev['response'],ev['bilateral'],ev['entry_profile_similarity'],n['weak'],(len(n['points'])-1)*cfg['step_px']])
        elif at==pool_line:
            eligible.update(ids[id(n)] for n in v['pool'])
            for n in v['finished']:
                if id(n) in ids:term[str(ids[id(n)])]=n['reason']
        elif at==beam_line:
            kept=[ids[id(n)] for n in v['beam']];retained.update(kept)
            events.append({'iteration':v['iteration'],'scored_proposals':len(v['proposals']),
                          'eligible_before_merge_topk':len(v['pool']),'beam_node_ids':kept})
        return trace
    assert sys.gettrace() is None,'Do not override an existing tracer'
    sys.settrace(trace)
    try:result=connect.search(im,response,seed,scene,bbox,cfg)
    finally:sys.settrace(None)
    arr=np.asarray(nodes,dtype=np.float64).reshape(-1,12)
    return result,arr,np.array([i+1 in eligible for i in range(len(nodes))]),np.array([i+1 in retained for i in range(len(nodes))]),events,term

def proposal_edge_distance(query,starts,ends,block=64):
    out=np.full(len(query),np.inf)
    if not len(starts):return out
    for j in range(0,len(query),block):
        p=query[j:j+block];best=np.full(len(p),np.inf)
        for k in range(0,len(starts),5000):
            a=starts[k:k+5000];delta=ends[k:k+5000]-a
            rel=p[:,None]-a[None];t=np.clip((rel*delta[None]).sum(-1)/np.maximum((delta*delta).sum(-1),1e-12),0,1)
            distance=((rel-t[...,None]*delta[None])**2).sum(-1)
            best=np.minimum(best,distance.min(-1))
        out[j:j+block]=np.sqrt(best)
    return out

def diagnose(scope,sid,mode,paths,arr,eligible,retained,response):
    # All network predictions and original decoder outputs already exist;
    # references enter only this posthoc diagnostic, never the trace callback.
    from evaluate import _entry_polygons,_without_input_edges
    from vision_geometry_lab.centerline_path_metrics_v6 import _sampling,_nearest,clip_scoring_segments
    row=next(r for r in scope['items'] if r['sample_id']==sid)
    seed=paths['entry'].get('free_seed',[])
    seg=clip_scoring_segments(row['quality_clipped_center_segments_xy'],ignore_polygons=_entry_polygons(seed))
    pieces=_sampling(seg);points=np.concatenate([p['points'] for p in pieces]);weights=np.concatenate([p['weights'] for p in pieces])
    centers=arr[:,3:5]
    starts=np.vstack([np.asarray(seed)[-1],centers])[arr[:,1].astype(int)]
    dist={'scored_proposal':proposal_edge_distance(points,starts,centers),
          'eligible_pool':proposal_edge_distance(points,starts[eligible],centers[eligible]),
          'retained_beam':proposal_edge_distance(points,starts[retained],centers[retained])}
    candidates=paths['candidates'];selected=next((c for c in candidates if c['id']==paths['selected_candidate_id']),None)
    dist['final_candidates']=_nearest(points,[s for c in candidates for s in _without_input_edges(c)])[0]
    dist['selected']=_nearest(points,_without_input_edges(selected))[0]
    reference_response=connect.sample(response,points)
    coverage={name:float(weights[d<=5].sum()/weights.sum()) for name,d in dist.items()}
    gaps=[];offset=0
    for piece in pieces:
        n=len(piece['points']);bad=dist['selected'][offset:offset+n]>5
        starts=np.flatnonzero(bad&~np.r_[False,bad[:-1]]);ends=np.flatnonzero(bad&~np.r_[bad[1:],False])+1
        for lo,hi in zip(starts,ends):
            if piece['weights'][lo:hi].sum()<60:continue
            counts={};ids=np.arange(offset+lo,offset+hi)
            for q in ids:
                if dist['final_candidates'][q]<=5:reason='candidate_present_but_rejected_or_not_selected'
                elif dist['retained_beam'][q]<=5:reason='near_reference_node_was_in_beam_then_lost_later_or_final_ranking'
                elif dist['eligible_pool'][q]<=5:reason='near_reference_proposal_lost_in_merge_or_beam_top18'
                elif dist['scored_proposal'][q]<=5:reason='near_reference_proposal_terminated_by_support_or_edge_gate'
                else:reason='no_scored_proposal_near_reference_generation_or_prior_path_loss'
                counts[reason]=counts.get(reason,0)+float(weights[q])
            gaps.append({'from_xy':piece['points'][lo].tolist(),'to_xy':piece['points'][hi-1].tolist(),
                         'length_px':float(piece['weights'][lo:hi].sum()),'stage_loss_proxy_length_px':counts,
                         'mean_reference_response':float(np.average(reference_response[ids],weights=weights[ids])),
                         'reference_fraction_response_ge_05':float(np.average(reference_response[ids]>=.5,weights=weights[ids]))})
        offset+=n
    # Every proposal is assessed as its real parent-to-endpoint 12px edge,
    # avoiding an artificial gap from discrete proposal-point sampling.
    return {'case':sid,'mode':mode,'reference_length_px':float(weights.sum()),'proximity_coverage_by_stage':coverage,
      'first_sustained_selected_gap':gaps[0] if gaps else None,'all_sustained_selected_gaps':gaps,
      'trace_nodes':len(arr),'reference_use':'posthoc only, never search parameters or accepted output',
      'limitations':['proposal/beam proximity uses parent-to-endpoint12px edges; final candidates use continuous line distance',
                    'near-reference proposal edges need not have a correct target prefix or physical identity',
                    'merge vs top18 and subsequent pruning vs final ranking cannot be causally separated by aggregate proximity alone',
                    'no new decoder, no reference-based reranking; exact old outputs required']},points,weights,dist

def run(run):
    run=Path(run).resolve();dest=run/'connection_trace';dest.mkdir(exist_ok=False)
    scope_path=run/'evaluation_scope/scope.json';records=[]
    for variant,base in [('old',OLD),('A',run/'branches/A'),('B',run/'branches/B')]:
        idx=read(base/'data/index.json');cfg=read(base/'path_config.json')
        for sid in CASES:
            item=next(i for i in idx['items'] if i['id']==sid);mode='CONDITIONAL';folder=base/'predictions'/mode/sid
            previous=read(folder/'paths.json');seed=previous['entry']['free_seed'];scene=previous['edge_target']
            tick=time.perf_counter();im=connect.imread(item['image_path']);response=np.load(folder/'response.npy',mmap_mode='r')
            result,arr,eligible,retained,events,term=observe(im,response,seed,scene,item['bbox'],cfg)
            rows,selected,meta=result
            assert rows==previous['candidates'] and selected==previous['selected_candidate_id'] and meta==previous['search'], 'Tracing altered decoder output'
            target=dest/f'{variant}__{sid}';target.mkdir()
            np.savez_compressed(target/'blind_trace.npz',nodes=arr,eligible_pool=eligible,retained_beam=retained)
            write(target/'blind_trace.json',{'columns':['node_id','parent_id','iteration','x','y','angle','step_score','response','bilateral','template','weak_px','path_length_px'],
                'events':events,'terminal_reasons':term,'frozen_decoder_output_exactly_reproduced':True,
                'search_code_sha256':sha(Path(connect.__file__)),'paths_sha256':sha(folder/'paths.json'),
                'response_sha256':sha(folder/'response.npy'),'reference_used_during_tracing':False})
            diag,points,weights,dist=diagnose(read(scope_path),sid,mode,previous,arr,eligible,retained,response)
            np.savez_compressed(target/'posthoc_proximity.npz',reference_points=points,weights=weights,**dist)
            diag['elapsed_seconds']=time.perf_counter()-tick;diag['variant']=variant;write(target/'diagnosis.json',diag);records.append(diag)
            print(json.dumps({'variant':variant,'id':sid,'nodes':len(arr),'seconds':diag['elapsed_seconds'],'exact_replay':True}),flush=True)
    write(dest/'summary.json',{'exit_code':0,'command':sys.argv,'items':records,'source_sha256':sha(__file__),'selection_scope':'two pre-frozen target-neighbor proxy cases, CONDITIONAL; old/A/B same decoder',
                             'no_decoder_tuning':True,'diagnostic_does_not_confirm_material_identity':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);run(p.parse_args().run)
