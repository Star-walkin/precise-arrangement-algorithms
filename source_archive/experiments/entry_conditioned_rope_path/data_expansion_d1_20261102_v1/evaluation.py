"""Unchanged frozen connector; save all predictions BEFORE partial references.

This script never trains, adjusts thresholds, changes entry selection or draws
over heatmaps. Frozen A750 is an optional start-point reference, not an arm.
"""
from __future__ import annotations
import argparse,json,hashlib,sys,time,csv,importlib.util
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE));sys.path.insert(0,str(ROOT))
sys.path.append(str(ROOT/'vision_prototype/.venv/Lib/site-packages'))
import cv2
import connect as frozen_connect
import evaluate as E
import compare_evaluate as C
import scene
from render_export import up2
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def items():
    ix=load(BASE/'runs/run_20261030_002/data/index.json');ev=[i for i in ix['items'] if i['split']=='eval']
    assert len(ev)==8 and {i['id'] for i in ev}==set(E.data_eval.SAMPLE_IDS)
    return ev
def mapped(p):
    with np.load(p,allow_pickle=False) as z:a=z['response'].copy()
    assert a.shape==(975,2048) and a.dtype==np.float32
    full=np.full((3000,4096),np.nan,np.float32)
    if np.isfinite(a).any():
        assert np.isfinite(a).all() and 0<=a.min()<=a.max()<=1
        full[1050:]=up2(a)
    return full
def config():
    p=BASE/'runs/run_20261030_002/branches/A/path_config.json'
    cfg=load(p);dump(HERE/'evaluation_path_config.json',{'source':str(p),'source_sha256':sha(p),'parameters':cfg,
        'connector_source':str(BASE/'connect.py'),'connector_sha256':sha(BASE/'connect.py'),
        'no_parameter_changes':True})
    return cfg
def available_scene(item,mode):
    if mode=='CONDITIONAL':return load(item['scene_path']),item['known_seed_xy'],{'available':True,'mode':'CONDITIONAL_KNOWN_ENTRY','source':'unchanged known initial120px entry only'}
    ent=load(item['auto_entry_path'])
    if ent['available']:return load(ent['scene_path']),ent['free_seed'],ent
    return {'contact_proxy':{'status':'unknown','xy':None,'t_from_entry_px':None},'target_edge_points_xy':[],
        'entry_short_xy':[],'entry_origin_xy':[0.,0.],'entry_direction_xy':[1.,0.],
        'parameters':{'contact_band_px':96.,'edge_band_px':96.},
        'actual_image_boundary_xyxy':[0,0,4095,2999],
        'target_edge_reason':'automatic entry unavailable; known-entry target geometry not supplied',
        'target_identity_at_edge_confirmed':False,'physical_contact_confirmed':False,
        'placeholder_coordinates_not_an_entry':True},[],ent
def generate():
    cv2.setNumThreads(1);ev=items();selection=load(HERE/'selected_checkpoints.json');cfg=config()
    root=HERE/'evaluation_paths';root.mkdir(exist_ok=False);rows=[];tick=time.perf_counter()
    for branch in ['D0','D1']:
        for mode in ['CONDITIONAL','AUTO']:
            for item in ev:
                p=HERE/'inference/fixed'/branch/mode/item['id']/'response.npz';meta=load(p.with_suffix('.json'))
                assert sha(p)==meta['response_sha256'] and meta['model_sha256']==selection['paths'][branch]['sha256']
                sc,seed,ent=available_scene(item,mode);assert np.array_equal(np.asarray(meta['seed_xy']),np.asarray(seed))
                if meta['status']=='predicted':
                    full=mapped(p);im=frozen_connect.imread(item['image_path'])
                    candidates,selected,diag=frozen_connect.search(im,full,seed,sc,item['bbox'],cfg)
                    del full,im
                else:candidates=[];selected=None;diag={'reason':'automatic_entry_unavailable'}
                result={'id':item['id'],'mode':mode,'selected_candidate_id':selected,'candidates':candidates,
                    'edge_target':sc,'entry':ent,'search':diag,'parameters':cfg,'coordinate_system':'original image pixels',
                    'supervision_used_at_inference':False,'response_sha256':meta['response_sha256'],
                    'image_sha256':item['image_sha256'],'source_sha256':sha(BASE/'connect.py')}
                out=root/branch/mode/(item['id']+'.json');dump(out,result)
                rows.append({'branch':branch,'mode':mode,'id':item['id'],'paths':str(out),'paths_sha256':sha(out),
                    'response':str(p),'response_sha256':sha(p),'seed_xy':seed,'scene_path':item['scene_path'],
                    'image_sha256':item['image_sha256'],'available':meta['status']=='predicted',
                    'selected':selected,'model_sha256':meta['model_sha256']})
                print({'paths_saved':len(rows),'total':32,'branch':branch,'mode':mode,'id':item['id'],'selected':selected},flush=True)
    # Ensure every mode/image pair is present, and freeze all files before references.
    assert len(rows)==32
    for r in rows:
        assert sha(r['paths'])==r['paths_sha256'] and sha(r['response'])==r['response_sha256']
        v=load(r['paths']);assert len(v['candidates'])<=3
        for c in v['candidates']:
            q=np.asarray(c['points_xy']);assert len(q)==len(c['stage'])==len(c['point_status']) and np.isfinite(q).all()
            assert np.all(q[:,0]>=0)&np.all(q[:,0]<4096)&np.all(q[:,1]>=1050)&np.all(q[:,1]<3000)
    dump(root/'prediction_save_gate.json',{'passed':True,'case_count':32,'rows':rows,'labels_not_yet_opened':True,
        'selected_checkpoints_sha256':sha(HERE/'selected_checkpoints.json'),'frozen_connector_sha256':sha(BASE/'connect.py'),
        'config_sha256':sha(HERE/'evaluation_path_config.json'),'seconds':time.perf_counter()-tick})
def mean(values):
    v=[float(x) for x in values if x is not None];return float(np.mean(v)) if v else None
def score():
    gate=load(HERE/'evaluation_paths/prediction_save_gate.json');assert gate['passed'] and len(gate['rows'])==32
    assert sha(BASE/'connect.py')==gate['frozen_connector_sha256']
    # Revalidate complete save gate immediately before ANY reference data read.
    for r in gate['rows']:assert sha(r['paths'])==r['paths_sha256'] and sha(r['response'])==r['response_sha256']
    out=HERE/'evaluation';out.mkdir(exist_ok=False);tick=time.perf_counter();refs={};rows=[]
    scope_path=BASE/'runs/run_20261030_002/evaluation_scope/scope.json'
    scope=load(scope_path);scopes={r['sample_id']:r for r in scope['items']}
    for record in gate['rows']:
        sid=record['id'];pred=load(record['paths']);full=mapped(record['response']);meta=load(Path(record['response']).with_suffix('.json'))
        if sid not in refs:refs[sid]=E.data_eval.load_evaluation_reference(sid)
        values=full[np.isfinite(full)];dist=None if not len(values) else {'mean':float(values.mean()),'std':float(values.std()),
            'minimum':float(values.min()),'maximum':float(values.max()),'semantics':'new center response0..1, not identity probability'}
        sc=load(record['scene_path']);case={'sample_id':sid,'mode':record['mode'],'prediction':pred,'scene':pred['edge_target'],
            'scoring_scene':sc,'actual_input_seed_xy':record['seed_xy'],'response_distribution':dist,
            'prediction_files':{k:record[k] for k in ['paths','paths_sha256','response','response_sha256']}}
        result=E.evaluate_case(case,refs[sid],scene.stage_at_points);result['branch']=record['branch']
        result['group_id']=scopes[sid]['group_id'];result['model_sha256']=record['model_sha256']
        target=C.line_response(full,result['source_reference']['trusted_clipped_centerline'],sc)
        result['target_center_response']=target
        pairs=[]
        for p in scopes[sid]['target_neighbor_pairs']:
            t=C.line_response(full,[p['target_points_xy']])['all'];n=C.line_response(full,[p['neighbor_points_xy']])['all']
            pairs.append({'id':p['id'],'target':t,'neighbor':n,'Delta':t['mean']-n['mean'] if t['mean'] is not None and n['mean'] is not None else None,
                'scope':'two frozen engineering proxy pairs, not independent complete identity truth'})
        result['target_neighbor_proxies']=pairs
        result['first_sustained_reference_gap_ge60px']=C.first_deviation(result['selected'],refs[sid])
        result['full_arc_identity_accuracy']=None;result['true_target_endpoint_success']=None
        rows.append(result);dump(out/record['branch']/record['mode']/(sid+'.json'),result)
        del full,values
    summary={}
    for branch in ['D0','D1']:
        rr=[r for r in rows if r['branch']==branch];geometry=E._aggregate(rr);ss={}
        for mode in ['AUTO','CONDITIONAL']:
            mr=[r for r in rr if r['mode']==mode]
            ss[mode]={'path_geometry':geometry[mode],
                'target_center_macro_image':{s:{k:mean([r['target_center_response']['stages'][s][k] for r in mr]) for k in ['mean','p95','fraction_gt0_5']} for s in E.STAGES.values()},
                'target_neighbor_delta_macro_pair':mean([p['Delta'] for r in mr for p in r['target_neighbor_proxies']]),
                'valid_proxy_pair_count':sum(p['Delta'] is not None for r in mr for p in r['target_neighbor_proxies']),
                'longest_partial_reference_gap_px':max(r['selected']['all']['longest_uncovered_run_px'] or 0 for r in mr),
                'full_identity_success':None,'endpoint_identity_success':None}
        summary[branch]=ss
    with (out/'per_case_metrics.csv').open('x',newline='',encoding='utf-8-sig') as f:
        fields=['id','branch','mode','selected','all_coverage','attached_coverage','edge_near_coverage','mean_distance_px','longest_gap_px','attached_response','edge_response']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:
            writer.writerow({'id':r['sample_id'],'branch':r['branch'],'mode':r['mode'],'selected':r['selected_candidate_id'],
                'all_coverage':r['selected']['all']['covered_fraction'],'attached_coverage':r['selected']['stages']['attached']['covered_fraction'],
                'edge_near_coverage':r['selected']['stages']['edge_near']['covered_fraction'],'mean_distance_px':r['selected']['all']['reference_to_prediction_mean_px'],
                'longest_gap_px':r['selected']['all']['longest_uncovered_run_px'],'attached_response':r['target_center_response']['stages']['attached']['mean'],
                'edge_response':r['target_center_response']['stages']['edge_near']['mean']})
    result={'exit_code':0,'summary':summary,'rows':rows,'seconds':time.perf_counter()-tick,
        'protocol':'same8 previously open development images,3 groups; AUTO failures stay8 denominator;5originalpx partial-reference geometry',
        'prediction_save_gate_sha256':sha(HERE/'evaluation_paths/prediction_save_gate.json'),'evaluation_source_sha256':sha(__file__),
        'connector_parameters_changed':False,'evaluation_used_for_training_or_checkpoint_selection':False,
        'no_full_arc_or_endpoint_identity_reference':True}
    dump(out/'summary.json',result);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
def training_fit():
    # Existing training targets may be read for TRAIN fit only; not independent eval.
    sys.path.insert(0,str(HERE));import train_infer as T
    T.configure(8);cfg=T.make_config();ix,old,new=T.training_index();arrays=T.arrays_for(old+new);selection=load(HERE/'selected_checkpoints.json');result={}
    for branch in ['D0','D1']:
        net=T.load_model(selection['paths'][branch]['path']);result[branch]={'old':T.fit_stats(net,old,arrays,cfg),'new':T.fit_stats(net,new,arrays,cfg)}
    dump(HERE/'training_fit_comparison.json',{'models':result,'scope':'fit on TRAIN labels; new weak labels are not independent truth',
        'dataset_index_sha256':sha(HERE/'dataset/index.json'),'selected_checkpoints_sha256':sha(HERE/'selected_checkpoints.json')})
def main():
    sys.stdout.reconfigure(encoding='utf8');p=argparse.ArgumentParser();p.add_argument('mode',choices=['predict','score','run','train-fit']);a=p.parse_args()
    if a.mode in ['predict','run']:generate()
    if a.mode in ['score','run']:score()
    if a.mode=='train-fit':training_fit()
if __name__=='__main__':main()
