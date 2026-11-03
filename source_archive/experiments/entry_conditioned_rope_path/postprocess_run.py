"""Run frozen P0 and P1 on original saved floats, never evaluation labels."""
from __future__ import annotations
from pathlib import Path
import argparse,copy,gzip,hashlib,json,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import cv2
import connect
import p1_segments as p1

CODE=Path(__file__).resolve().parent;OLD=CODE/'runs/run_20261030_002'
def load(p):
    p=Path(p)
    if p.suffix=='.gz':
        with gzip.open(p,'rt',encoding='utf-8') as f:return json.load(f)
    return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.suffix=='.gz':
        with gzip.open(p,'wt',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,separators=(',',':'))
    else:p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def frozen_config(run):
    cfg=load(run/'p1_config.json');freeze=load(run/'p1_freeze.json')
    assert p1.config_hash({**p1.DEFAULTS,**cfg})==freeze['config_sha256']
    assert sha(CODE/'p1_segments.py')==freeze['source_sha256']
    return cfg

def run_case(run,variant,mode,sid,gray,response_path,scene,seed,bbox,image_path,image_sha,out_dirs,cfg,reference_p0=None,entry_extra=None):
    available=len(seed)>=2;response=np.load(response_path,mmap_mode='r',allow_pickle=False)
    response_hash=sha(response_path);valid=np.isfinite(response)
    assert response.dtype==np.float32 and response.shape==gray.shape
    if available:assert valid.any()
    cfg0=load(OLD/f'branches/{variant}/path_config.json')
    common={'id':sid,'mode':mode,'variant':variant,'response_path':str(response_path),'response_sha256':response_hash,
      'image_path':str(image_path),'image_sha256':image_sha,'entry_available':available,'actual_seed_xy':seed,
      'entry':{'available':available,'free_seed':seed,**(entry_extra or {})},'edge_target':scene,'bbox_original_xyxy':list(bbox),
      'coordinate_system':'original image pixels','response_unmodified':True,'evaluation_labels_read':False,
      'scope':'center path and evidence, not true rope-body mask or calibrated identity confidence'}
    outputs={}
    for post in ['P0','P1']:
        dest=Path(out_dirs[post]);assert not (dest/'paths.json').exists(),dest
        start=time.perf_counter();extra={}
        if post=='P0':
            if available:rows,selected,meta=connect.search(gray,response,seed,scene,bbox,cfg0)
            else:rows=[];selected=None;meta=reference_p0['search'] if reference_p0 else {'reason':'entry_unavailable'}
            if reference_p0:
                assert rows==reference_p0['candidates'] and selected==reference_p0['selected_candidate_id']
                assert meta==reference_p0['search']
                extra['frozen_old_output_exactly_reproduced']=True
            parameters=cfg0;extra['source_sha256']=sha(CODE/'connect.py');extraction_seconds=0.
        elif available:
            key=hashlib.sha256((response_hash+image_sha+p1.config_hash({**p1.DEFAULTS,**cfg})+sha(CODE/'p1_segments.py')).encode()).hexdigest()
            cache=run/'p1_cache'/key;bundlefile=cache/'segments.json.gz'
            hit=bundlefile.exists()
            if hit:bundle=load(bundlefile)
            else:
                bundle,arrays=p1.extract(gray,response,valid,bbox,cfg)
                save(bundlefile,bundle);np.savez_compressed(cache/'derived.npz',**arrays)
                save(cache/'source.json',{'response_path':str(response_path),'response_sha256':response_hash,'image_sha256':image_sha,
                    'config_sha256':p1.config_hash({**p1.DEFAULTS,**cfg}),'source_sha256':sha(CODE/'p1_segments.py'),
                    'derived_data_only':True,'labels_used':False})
                del arrays
            extraction_seconds=bundle['diagnostics']['elapsed_seconds']
            connect_start=time.perf_counter();rows,selected,meta=p1.connect_segments(bundle,gray,response,valid,seed,scene,bbox,cfg)
            connection_seconds=time.perf_counter()-connect_start
            edges=meta.pop('graph_edges',[]);save(dest/'graph_edges.json.gz',edges)
            meta['graph_edges_path']=str(dest/'graph_edges.json.gz');meta['graph_edges_sha256']=sha(dest/'graph_edges.json.gz')
            extra.update(segments_path=str(bundlefile),segments_sha256=sha(bundlefile),derived_arrays_path=str(cache/'derived.npz'),
              extraction_diagnostics=bundle['diagnostics'],extraction_cache_hit=hit,connection_seconds=connection_seconds,
              source_sha256=sha(CODE/'p1_segments.py'))
            parameters=cfg
        else:
            rows=[];selected=None;meta={'reason':'entry_unavailable','selection_reason':'entry_unavailable','entry_available':False}
            extraction_seconds=0.;parameters=cfg;extra['source_sha256']=sha(CODE/'p1_segments.py')
        elapsed=time.perf_counter()-start
        result={**common,'postprocessor':post,'selected_candidate_id':selected,'candidates':rows,'search':meta,'parameters':parameters,
          'elapsed_seconds':elapsed,'extraction_seconds':extraction_seconds,'timing_note':'wall time includes derivative cache write on first P1 use; search representations differ; same beam does not imply equal compute',**extra}
        save(dest/'paths.json',result);save(dest/'input.json',{k:result[k] for k in common})
        outputs[post]={'selected':selected,'candidates':len(rows),'elapsed_seconds':elapsed,'extraction_seconds':extraction_seconds,
                      'paths_path':str(dest/'paths.json'),'paths_sha256':sha(dest/'paths.json'),
                      'budget_events':meta.get('search_budget_event'),'segments':extra.get('extraction_diagnostics',{}).get('segments',0)}
    return {'id':sid,'variant':variant,'mode':mode,'entry_available':available,'response_sha256':response_hash,'postprocessors':outputs}

def formal(run):
    cv2.setNumThreads(2);cfg=frozen_config(run);records=[];tick=time.perf_counter()
    index=load(OLD/'data/index.json');evals=[i for i in index['items'] if i['split']=='eval'];assert len(evals)==8
    for variant in ['A','B']:
        for item in evals:
            gray=connect.imread(item['image_path'])
            for mode in ['CONDITIONAL','AUTO']:
                original=OLD/'branches'/variant/'predictions'/mode/item['id'];old=load(original/'paths.json');entry=load(original/'entry.json')
                seed=entry.get('free_seed',[]) if entry.get('available',True) else []
                out={p:run/'formal'/variant/p/'predictions'/mode/item['id'] for p in ['P0','P1']}
                result=run_case(run,variant,mode,item['id'],gray,original/'response.npy',old['edge_target'],seed,item['bbox'],
                    item['image_path'],item['image_sha256'],out,cfg,reference_p0=old,entry_extra={'source':str(original/'entry.json')})
                result['original_p0_paths_sha256']=sha(original/'paths.json');records.append(result)
                print(json.dumps({'formal':len(records),'total':32,'id':item['id'],'variant':variant,'mode':mode,'results':result['postprocessors']}),flush=True)
    save(run/'formal/receipt.json',{'exit_code':0,'cases':64,'image_count':8,'source_groups':3,'seconds':time.perf_counter()-tick,
      'command':sys.argv,'records':records,'P0_config_unchanged':True,'P0_outputs_exact_old':True,'no_label_access':True,
      'p1_freeze_sha256':sha(run/'p1_freeze.json'),'source_sha256':sha(__file__)})

def batch_job(args):
    run,job,cfg=args;run=Path(run);cv2.setNumThreads(1)
    gray=connect.imread(job['image_path']);records=[]
    for condition in job['conditions']:
        scene=load(condition['scene_path'])
        for variant in ['A','B']:
            response=Path(condition['response_paths'][variant]);out={p:response.parent/p for p in ['P0','P1']}
            records.append(run_case(run,variant,'BATCH_AUTO',condition['condition_id'],gray,response,scene,condition['entry_xy'],job['bbox'],
                job['image_path'],job['image_sha256'],out,cfg,entry_extra={'source':condition['entry']['source'],
                    'entry_id':condition['entry']['entry_id'],'ambiguous':condition['entry']['ambiguous'],'entry_number':condition['entry_number']}))
    return {'id':job['id'],'records':records}

def batch(run,workers):
    cfg=frozen_config(run);manifest=load(run/'batch/manifest.json');assert (run/'batch/inference_receipt.json').exists()
    tick=time.perf_counter();records=[]
    with ProcessPoolExecutor(max_workers=workers) as executor,(run/'batch/postprocess_progress.jsonl').open('x',encoding='utf-8') as log:
        futures=[executor.submit(batch_job,(str(run),j,cfg)) for j in manifest['jobs']]
        for future in as_completed(futures):
            row=future.result();records.append(row);log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
            if len(records)==1 or len(records)%20==0:print(json.dumps({'postprocessed_images':len(records),'total':len(futures),'seconds':time.perf_counter()-tick}),flush=True)
    save(run/'batch/postprocess_receipt.json',{'exit_code':0,'images':len(records),'conditions':manifest['conditions'],'workers':workers,
      'seconds':time.perf_counter()-tick,'command':sys.argv,'source_sha256':sha(__file__),'records':records,
      'timing_note':'batch concurrent throughput, not serial formal latency comparison','P0_frozen':True,'P1_frozen':True})

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('mode',choices=['formal','batch']);p.add_argument('--run',required=True,type=Path)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args();(formal(a.run.resolve()) if a.mode=='formal' else batch(a.run.resolve(),a.workers))
