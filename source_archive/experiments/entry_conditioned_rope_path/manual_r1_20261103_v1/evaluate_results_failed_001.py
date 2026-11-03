"""Score frozen 48 predictions only after their complete save/hash gate."""
from pathlib import Path
import json,hashlib,sys,time,csv
import numpy as np
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE));sys.path.insert(0,str(ROOT))
import evaluate as E
import compare_evaluate as C
import scene

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def up(a):
    # align_corners=False original mapping, independent of renderer.
    h,w=a.shape;x=(np.arange(2*w)+.5)/2-.5;y=(np.arange(2*h)+.5)/2-.5
    x0=np.floor(x).astype(int);y0=np.floor(y).astype(int);fx=x-x0;fy=y-y0
    b=a[:,x0.clip(0,w-1)]*(1-fx)+a[:,(x0+1).clip(0,w-1)]*fx
    return (b[y0.clip(0,h-1)]*(1-fy[:,None])+b[(y0+1).clip(0,h-1)]*fy[:,None]).astype(np.float32)
def mapped(path):
    path=Path(path)
    if path.suffix=='.npy':a=np.load(path,allow_pickle=False)
    else:
        with np.load(path,allow_pickle=False) as z:a=z['response'].copy()
    assert a.shape==(975,2048)
    result=np.full((3000,4096),np.nan,np.float32)
    if np.isfinite(a).any():
        assert np.isfinite(a).all();result[1050:]=up(a)
    return result
def mean(values):
    v=[float(x) for x in values if x is not None];return float(np.mean(v)) if v else None

def main():
    gate=load(HERE/'paths/prediction_save_gate.json');assert gate['passed'] and gate['case_count']==48
    assert len(gate['rows'])==48
    for row in gate['rows']:
        assert sha(row['path'])==row['sha256']
        r=load(row['path']);m=r['metadata']
        assert not m['target_labels_read'] and sha(m['response_path'])==m['response_sha256']
    out=HERE/'evaluation';out.mkdir(exist_ok=False);started=time.perf_counter()
    dump(out/'prediction_save_gate.json',dict(passed=True,case_count=48,
         gate_sha256=sha(HERE/'paths/prediction_save_gate.json'),labels_not_yet_opened=True,
         source_sha256=sha(__file__)))
    # Only after every prediction above has been saved/hashed may references open.
    old=load(BASE/'runs/run_20261030_002/data/index.json')
    ev={i['id']:i for i in old['items'] if i['split']=='eval'};assert len(ev)==8
    scope=load(BASE/'runs/run_20261030_002/evaluation_scope/scope.json')
    scopes={i['sample_id']:i for i in scope['items']};refs={};rows=[];heat_rows=[]
    for sid in ev:refs[sid]=E.data_eval.load_evaluation_reference(sid)
    for row in gate['rows']:
        r=load(row['path']);m=r['metadata'];item=ev[row['id']]
        with np.load(m['response_path'],allow_pickle=False) as z:a=z['response']
        finite=a[np.isfinite(a)]
        dist=None if not len(finite) else dict(mean=float(finite.mean()),std=float(finite.std()),minimum=float(finite.min()),maximum=float(finite.max()))
        case=dict(sample_id=row['id'],mode=row['mode'],prediction=r['result'],scene=r['scene'],
                  scoring_scene=load(item['scene_path']),actual_input_seed_xy=m['entry_seed_xy'],response_distribution=dist,
                  prediction_files=dict(paths=row['path'],paths_sha256=row['sha256'],response=m['response_path'],response_sha256=m['response_sha256']))
        score=E.evaluate_case(case,refs[row['id']],scene.stage_at_points)
        score['branch']=row['arm'];score['source_group']=item['source_group']
        score['first_sustained_reference_gap_ge60px']=C.first_deviation(score['selected'],refs[row['id']])
        rows.append(score);dump(out/row['arm']/row['mode']/(row['id']+'.json'),score)
    for model in ('A750','D1','U1'):
        for mode in ('AUTO','CONDITIONAL'):
            for sid,item in ev.items():
                if model=='A750':p=BASE/'runs/run_20261030_002/branches/A/predictions'/mode/sid/'response.npy'
                elif model=='D1':p=BASE/'data_expansion_d1_20261102_v1/inference/fixed/D1'/mode/sid/'response.npz'
                else:p=HERE/'inference/fixed/U1'/mode/sid/'response.npz'
                assert p.exists(),p
                full=mapped(p);sc=load(item['scene_path'])
                lines=next(r['source_reference']['trusted_clipped_centerline'] for r in rows if r['sample_id']==sid)
                target=C.line_response(full,lines,sc);pairs=[]
                for pair in scopes[sid]['target_neighbor_pairs']:
                    t=C.line_response(full,[pair['target_points_xy']])['all'];n=C.line_response(full,[pair['neighbor_points_xy']])['all']
                    pairs.append(dict(id=pair['id'],target=t,neighbor=n,delta=t['mean']-n['mean'] if t['mean'] is not None and n['mean'] is not None else None))
                heat_rows.append(dict(model=model,mode=mode,id=sid,target=target,neighbor_pairs=pairs,response_path=str(p),response_sha256=sha(p)))
    summary={arm:E._aggregate([r for r in rows if r['branch']==arm]) for arm in ('R0','R1_old','R1_new')}
    hs={}
    for model in ('A750','D1','U1'):
        hs[model]={}
        for mode in ('AUTO','CONDITIONAL'):
            mm=[r for r in heat_rows if r['model']==model and r['mode']==mode]
            hs[model][mode]=dict(center_response={s:mean([r['target']['stages'][s]['mean'] for r in mm]) for s in E.STAGES.values()},
                                target_neighbor_delta=mean([p['delta'] for r in mm for p in r['neighbor_pairs']]),
                                neighbor_pair_count=sum(p['delta'] is not None for r in mm for p in r['neighbor_pairs']))
    dump(out/'heatmap_metrics.json',dict(summary=hs,rows=heat_rows,scope='same8 historical open partial references,2 engineering neighbor proxy pairs; no complete identity accuracy'))
    dump(out/'summary.json',dict(exit_code=0,summary=summary,heatmap_summary=hs,rows=rows,seconds=time.perf_counter()-started,
         all_predictions_frozen_before_reference_reads=True,checkpoint_selected_without_evaluation=True,
         auto_missing_entries_kept_in_eight_denominator=True,full_target_identity_accuracy=None,
         scope='same8 historical open images/3source groups,5originalpx partial-reference arclength coverage; not blind'))
    with (out/'per_case_metrics.csv').open('x',encoding='utf-8-sig',newline='') as f:
        fields=['branch','mode','id','selected','all_coverage','attached_coverage','edge_coverage','longest_gap_px']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow(dict(branch=r['branch'],mode=r['mode'],id=r['sample_id'],selected=r['selected_candidate_id'],
            all_coverage=r['selected']['all']['covered_fraction'],attached_coverage=r['selected']['stages']['attached']['covered_fraction'],
            edge_coverage=r['selected']['stages']['edge_near']['covered_fraction'],longest_gap_px=r['selected']['all']['longest_uncovered_run_px']))
    print(json.dumps({b:{m:dict(attached=s['selected_reference_coverage']['attached']['coverage_5px'],edge=s['selected_reference_coverage']['edge_near']['coverage_5px'],rejected=s['rejected_images']) for m,s in v.items()} for b,v in summary.items()},indent=2))

if __name__=='__main__':main()
