"""Post-freeze OLD/A/B evaluation; never changes predictions or decoder rules."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
from datetime import datetime,timezone
import cv2,numpy as np

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
import evaluate as legacy
from scene import stage_at_points
from vision_geometry_lab.experiments.v28_local_spool_connection import data_eval
from vision_geometry_lab.centerline_path_metrics_v6 import _sampling,_nearest

PROTOCOL={'version':1,'scope':'eight open evaluation images, three source groups, unchanged old5px path evaluation',
 'response_threshold':.5,'pool_diagnostic_coverage_threshold':.8,
 'first_sustained_deviation_minimum_reference_gap_px':60.,
 'first_sustained_deviation_meaning':'geometric partial-reference proxy only, not first physical identity error',
 'background':'fixed engineering visual polygons, per-region distributions; macro statistics explicitly labelled',
 'target':'reviewed partial centers, input excluded, fixed common visual stages, arc-length weighted response',
 'neighbor_delta':'two engineering visual pairs, not independent identity accuracy',
 'missing_auto_response':'undefined response statistics, never substituted zero; rejection remains in path population',
 'pool_loss_location':'final pool only; unavailable generation/filter/beam traces cannot identify an elimination stage',
 'formal_comparison':'A versus B at same saved update; OLD is starting-point context only'}

def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def stats(values,weights=None):
    values=np.asarray(values,float);finite=np.isfinite(values)
    w=np.ones_like(values) if weights is None else np.asarray(weights,float)
    n=int(finite.sum());total=float(w.sum());kept=float(w[finite].sum())
    if not n or kept<=0:return {'status':'unavailable','finite_samples':n,'finite_weight':kept,'total_weight':total,'mean':None,'p90':None,'p95':None,'fraction_gt0_5':None}
    v,w=values[finite],w[finite];order=np.argsort(v);sw=np.cumsum(w[order])
    return {'status':'available','finite_samples':n,'finite_weight':kept,'total_weight':total,
      'mean':float(np.sum(v*w)/kept),'p90':float(np.interp(.9*kept,sw,v[order])),
      'p95':float(np.interp(.95*kept,sw,v[order])),'fraction_gt0_5':float(w[v>.5].sum()/kept)}
def sample_response(array,points):
    p=np.asarray(points,np.float32)
    if not len(p):return np.zeros(0,np.float32)
    return cv2.remap(array,p[:,0].reshape(1,-1),p[:,1].reshape(1,-1),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=float('nan'))[0]
def polygon_response(array,polygon):
    p=np.asarray(polygon,np.int32);x0,y0=p.min(0);x1,y1=p.max(0);x0=max(0,x0);y0=max(0,y0)
    x1=min(array.shape[1]-1,x1);y1=min(array.shape[0]-1,y1)
    mask=np.zeros((y1-y0+1,x1-x0+1),np.uint8);cv2.fillPoly(mask,[p-[x0,y0]],1)
    return stats(array[y0:y1+1,x0:x1+1][mask.astype(bool)])
def line_response(array,segments,scene=None):
    vals=[];weights=[];stages=[]
    for piece in _sampling(segments):
        vals.extend(sample_response(array,piece['points']));weights.extend(piece['weights'])
        stages.extend(stage_at_points(scene,piece['points']) if scene is not None else np.ones(len(piece['points']),np.uint8))
    a,w,s=np.asarray(vals),np.asarray(weights),np.asarray(stages)
    out={'all':stats(a,w)}
    if scene is not None:out['stages']={name:stats(a[s==k],w[s==k]) for k,name in legacy.STAGES.items()}
    return out
def boundary_rows(array,y0=1050,y1=3000,band=64):
    """Inspect actual float rows, including interior peaks near padding."""
    def band_stats(start,end):
        a=np.asarray(array[start:end]);finite=np.isfinite(a);count=finite.sum(1)
        means=np.divide(np.nansum(a,axis=1),count,out=np.full(len(a),np.nan),where=count>0)
        good=np.isfinite(means)
        if not good.any():return {'status':'unavailable','row_range_y':[start,end],'maximum_row_mean':None,'maximum_row_y':None,'row_means':[None]*len(a)}
        k=int(np.nanargmax(means))
        return {'status':'available','row_range_y':[start,end],'maximum_row_mean':float(means[k]),'maximum_row_y':start+k,'row_means':[float(v) if np.isfinite(v) else None for v in means]}
    def region_mean(start,end):
        a=np.asarray(array[start:end]);v=a[np.isfinite(a)];return float(v.mean()) if len(v) else None
    return {'search_rows_y':[y0,y1],'top64':band_stats(y0,y0+band),'bottom64':band_stats(y1-band,y1),
      'top_first2_mean':region_mean(y0,y0+2),'top_inner16to48_mean':region_mean(y0+16,y0+48),
      'bottom_last2_mean':region_mean(y1-2,y1),'bottom_inner16to48_mean':region_mean(y1-48,y1-16),
      'interpretation':'raw float response; distinguish exact crop boundary from an interior bright row; preprocessing unchanged'}
def first_deviation(score,reference):
    if score['candidate_id'] is None:return {'status':'no_selected_path','first_sustained_deviation':None}
    gaps=[g for g in score['all']['uncovered_runs'] if g['length_px']>=60]
    if not gaps:return {'status':'no_sustained_gap_in_partial_reference','first_sustained_deviation':None}
    rows=[]
    for gap in gaps:
        arc=float(data_eval._project_arc(np.asarray([gap['from_xy']]),reference['centerline'])[0])
        rows.append({**gap,'from_trusted_reference_start_arclength_px':arc})
    return {'status':'geometric_departure_proxy','first_sustained_deviation':min(rows,key=lambda x:x['from_trusted_reference_start_arclength_px']),
      'meaning':'earliest >=60px trusted-reference gap at5px tolerance after input exclusion; not physical identity first-error truth'}
def pool_diagnostic(geom,target_response):
    out={}
    for name in legacy.STAGES.values():
        scores=[s['stages'][name]['covered_fraction'] for s in geom['candidate_scores']]
        scores=[s for s in scores if s is not None];best=max(scores,default=None)
        selected=geom['selected']['stages'][name]['covered_fraction']
        response=target_response['stages'][name]['fraction_gt0_5']
        if best is None:label='no_reference_or_no_saved_candidate'
        elif best>=.8 and (selected is None or selected<.8):label='matching_partial_geometry_in_pool_but_not_selected_or_rejected'
        elif best<.8 and response is not None and response>=.8:label='strong_reviewed_center_response_but_no_matching_final_pool_path'
        elif best<.8:label='insufficient_final_pool_geometry_and_or_target_response'
        else:label='selected_path_covers_at_least80percent_partial_stage_reference'
        out[name]={'pool_best_partial_coverage':best,'selected_partial_coverage':selected,
          'reviewed_center_fraction_response_gt0_5':response,'diagnosis':label,'generation_filter_beam_elimination_stage':'unknown_no_ground_truth_matched_intermediate_trace'}
    return out
def evaluate_extra(case,scope,geom):
    response=np.load(case['prediction_files']['response'],mmap_mode='r',allow_pickle=False)
    scene=read(scope['common_scene']);segments=geom['source_reference']['trusted_clipped_centerline']
    target=line_response(response,segments,scene)
    background=[{'id':r['id'],'source':r['source'],'statistics':polygon_response(response,r['polygon_xy'])} for r in scope['background_regions']]
    pairs=[]
    for p in scope['target_neighbor_pairs']:
        t=line_response(response,[p['target_points_xy']])['all'];n=line_response(response,[p['neighbor_points_xy']])['all']
        pairs.append({'id':p['id'],'target':t,'neighbor':n,'Delta':t['mean']-n['mean'] if t['mean'] is not None and n['mean'] is not None else None,'interpretation':'engineering proxy contrast only'})
    return {'sample_id':case['sample_id'],'mode':case['mode'],'group_id':scope['group_id'],
      'background':background,'target_response':target,'target_neighbor':pairs,
      'raw_float_boundary_rows':boundary_rows(response),
      'selected_path_deviation':first_deviation(geom['selected'],scope['reviewed_reference']),
      'first_candidate_deviation_diagnostic':first_deviation(geom['first_candidate_diagnostic'],scope['reviewed_reference']) if geom['first_candidate_diagnostic'] else None,
      'candidate_pool_diagnostic':pool_diagnostic(geom,target),'legacy_geometry':geom,
      'prediction_files':case['prediction_files']}
def mean_available(v):
    v=[float(x) for x in v if x is not None];return float(np.mean(v)) if v else None
def summarize(rows):
    result={}
    for version in ['OLD','A','B']:
        result[version]={}
        for mode in ['CONDITIONAL','AUTO']:
            rs=[r for r in rows if r['version']==version and r['mode']==mode];assert len(rs)==8
            bg=[x['statistics'] for r in rs for x in r['background']]
            target={st:{k:mean_available([r['target_response']['stages'][st][k] for r in rs]) for k in ['mean','p90','p95','fraction_gt0_5']} for st in legacy.STAGES.values()}
            pairs=[p['Delta'] for r in rs for p in r['target_neighbor']]
            groups={g:{'background_mean':mean_available([b['statistics']['mean'] for r in rs if r['group_id']==g for b in r['background']]),
                       'target_attached_mean':mean_available([r['target_response']['stages']['attached']['mean'] for r in rs if r['group_id']==g])} for g in sorted({r['group_id'] for r in rs})}
            result[version][mode]={'images':8,'groups':3,'valid_background_regions':sum(b['mean'] is not None for b in bg),
              'background_macro_region':{k:mean_available([b[k] for b in bg]) for k in ['mean','p90','p95','fraction_gt0_5']},
              'target_macro_image':target,'proxy_Delta_macro_pair':mean_available(pairs),'valid_proxy_pairs':sum(v is not None for v in pairs),
              'source_groups':groups,'path_geometry':legacy._aggregate([r['legacy_geometry'] for r in rs])[mode]}
    return result
def run(run_dir):
    run_dir=Path(run_dir).resolve();folder=run_dir/'evaluation_scope';scope=read(folder/'scope.json');receipt=read(folder/'freeze_receipt.json')
    assert sha(folder/'scope.json')==receipt['scope_sha256']
    output=run_dir/'comparison';assert not output.exists(),'Preserve prior comparison'
    scopes={r['sample_id']:r for r in scope['items']};roots={'OLD':Path(scope['old_run']),'A':run_dir/'branches/A','B':run_dir/'branches/B'}
    paired=read(run_dir/'paired_comparison.json')
    assert paired['all_overlapping_input_augmentation_hashes_equal'] and paired['same_config_schedule_optimizer_policy']
    cases={}
    # Validate all48 prediction/response pairs before any numerical comparison.
    for version,root in roots.items():
        for mode in ['CONDITIONAL','AUTO']:
            for sid in data_eval.SAMPLE_IDS:
                cases[(version,mode,sid)]=legacy._validate_case(root,mode,sid,scopes[sid]['image_shape_hw'])
    for mode in ['CONDITIONAL','AUTO']:
        for sid in data_eval.SAMPLE_IDS:
            a,b=cases[('A',mode,sid)],cases[('B',mode,sid)]
            assert a['actual_input_seed_xy']==b['actual_input_seed_xy']
            assert a['prediction']['parameters']==b['prediction']['parameters']==cases[('OLD',mode,sid)]['prediction']['parameters']
            assert a['prediction']['image_sha256']==b['prediction']['image_sha256']==scopes[sid]['image_sha256']
            for version in ['A','B']:
                metadata=read(roots[version]/'predictions'/mode/sid/'inference.json')['phases']['final']
                assert metadata['model_step']==paired['main_comparison_step'],'Inference must use largest common saved update'
                assert metadata['model_sha256']==paired['main_checkpoints'][version]['sha256']
    output.mkdir()
    for version in ['A','B']:
        if not (roots[version]/'evaluation/summary.json').exists():
            command=[sys.executable,'-B',str(BASE/'evaluate.py'),'--run',str(roots[version])]
            completed=subprocess.run(command,capture_output=True,text=True,encoding='utf-8')
            (output/f'legacy_evaluate_{version}.log').write_text(completed.stdout+'\n'+completed.stderr,encoding='utf-8')
            if completed.returncode:raise RuntimeError(f'Unchanged evaluator {version} failed; see log')
    rows=[]
    for (version,mode,sid),case in cases.items():
        geom=read(roots[version]/'evaluation'/mode/(sid+'.json'))
        assert geom['prediction_files']['paths_sha256']==case['prediction_files']['paths_sha256']
        assert geom['prediction_files']['response_sha256']==case['prediction_files']['response_sha256']
        row=evaluate_extra(case,scopes[sid],geom);row['version']=version;rows.append(row)
        save(output/version/mode/(sid+'.json'),row)
    result=summarize(rows)
    payload={'status':'completed','exit_code':0,'created_utc':datetime.now(timezone.utc).isoformat(),
      'protocol':PROTOCOL,'scope_sha256':receipt['scope_sha256'],'comparison_code_sha256':sha(__file__),
      'legacy_evaluator_sha256':sha(BASE/'evaluate.py'),'command':sys.argv,'summary':result,
      'paired_comparison_sha256':sha(run_dir/'paired_comparison.json'),'common_update_step':paired['main_comparison_step'],
      'independent_complete_endpoint_identity_available':False}
    save(output/'summary.json',payload)
    save(output/'evaluation.json',payload)
    print(json.dumps({'status':'completed','cases':48,'output':str(output),'summary':result}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run(a.run)
