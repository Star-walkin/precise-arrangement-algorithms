"""Full-search-ROI inference, mapped to original coordinates; no eval labels.

Run with the existing torch environment (OpenCV is intentionally unnecessary).
The raw float output is NaN only outside the fixed search ROI, or everywhere
when an automatic entry is unavailable. Processed outputs replace NaN by zero.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from train import load_model,tensor_sample,read_arrays

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def entry_for(item,mode):
    if mode=='AUTO':return load(item['auto_entry_path'])
    return {'available':True,'mode':'CONDITIONAL_KNOWN_ENTRY','free_seed':item['known_seed_xy'],
      'width_px':None,'reason':'existing reviewed free-entry hint, restricted to initial120 original px',
      'scene_path':item['scene_path'],'image_sha256':item['image_sha256'],
      'hint_path':item['entry_hint'],'hint_sha256':sha(item['entry_hint']),
      'entry_from_target_postcontact_truth':False}

@torch.inference_mode()
def run(run_dir,model_path,phase,threads):
    start=time.time();run_dir=Path(run_dir).resolve();model_path=Path(model_path).resolve()
    index_path=run_dir/'data/index.json';index=load(index_path)
    torch.set_num_threads(threads);torch.use_deterministic_algorithms(True)
    checkpoint=torch.load(model_path,map_location='cpu',weights_only=False)
    cfg=checkpoint['config'];net=load_model(model_path);model_hash=sha(model_path)
    stage_name={'initial':'untrained_baseline','final':'bounded_baseline_training','overfit':'overfit_stage_check_only'}[phase]
    if phase=='initial':assert checkpoint['step']==0,'Untrained response must use initial weights'
    if phase=='final':assert checkpoint.get('extra',{}).get('status')=='completed','Final phase requires a completed baseline'
    prefix='initial_' if phase=='initial' else ''
    jobs=[(item,'CONDITIONAL') for item in index['items']]+[(item,'AUTO') for item in index['items'] if item['split']=='eval']
    records=[]
    for item,mode in jobs:
        tick=time.time();dest=run_dir/'predictions'/mode/item['id'];dest.mkdir(parents=True,exist_ok=True)
        output=dest/(prefix+'response.npy');processed=dest/(prefix+'processed_response.npy')
        if output.exists():raise FileExistsError('Preserve existing predictions: '+str(output))
        entry=entry_for(item,mode);entry_path=dest/'entry.json'
        if entry_path.exists():assert load(entry_path)==entry,'Entry changed between initial/final inference'
        else:dump(entry_path,entry)
        full=np.full(item['shape'],np.nan,np.float32);input_stats=None;quantiles=None
        status='entry_unavailable' if not entry['available'] else 'predicted'
        if entry['available']:
            a=read_arrays(item)
            if mode=='AUTO':
                seed_path=Path(item['auto_entry_path']).with_suffix('.seed.npy')
                a['seed']=np.load(seed_path,allow_pickle=False).astype(np.float32)
                assert a['seed'].shape==a['image'].shape and a['seed'].sum()>0
            x,_,_,_,input_stats=tensor_sample(a,cfg,augment=False)
            h,w=a['image'].shape;response=net(x)[0,0,:h,:w]
            assert torch.isfinite(response).all() and response.min()>=0 and response.max()<=1
            x0,y0,x1,y1=item['bbox']
            raw=F.interpolate(response[None,None],size=(y1-y0,x1-x0),mode='bilinear',align_corners=False)[0,0].numpy()
            full[y0:y1,x0:x1]=raw
            quantiles={str(q):float(v) for q,v in zip([0,.01,.1,.5,.9,.99,1],np.quantile(raw,[0,.01,.1,.5,.9,.99,1]))}
            input_stats.update({'image_mean':float(a['image'][a['roi']].mean()),'image_std':float(a['image'][a['roi']].std()),
              'image_min':float(a['image'].min()),'image_max':float(a['image'].max()),'model_input_shape':list(x.shape),
              'seed_source':mode,'seed_sum':float(a['seed'].sum()),'preprocessing':'original grayscale; per-search-ROI mean/std floor; no LCN',
              'preprocessing_source_sha256':sha(Path(__file__).with_name('train.py'))})
            distribution={'quantiles':quantiles,'mean':float(raw.mean()),'std':float(raw.std()),
              'fraction_lt_001':float((raw<.01).mean()),'fraction_gt_09':float((raw>.9).mean())}
            del a,x,response,raw
        else:distribution=None
        np.save(output,full);np.save(processed,np.nan_to_num(full,nan=0.).astype(np.float32))
        record={'phase':phase,'model_stage':stage_name,'status':status,'model_path':str(model_path),'model_sha256':model_hash,
          'model_step':checkpoint['step'],'response_path':str(output),'response_sha256':sha(output),
          'processed_response_path':str(processed),'processed_response_sha256':sha(processed),
          'input_statistics':input_stats,'response_distribution':distribution,'elapsed_seconds':time.time()-tick,
          'all_nan_due_to_missing_entry':not entry['available'],'unavailable_reason':entry.get('reason') if not entry['available'] else None}
        info_path=dest/'inference.json'
        info=load(info_path) if info_path.exists() else {'schema_version':1,'id':item['id'],'split':item['split'],'source_group':item['source_group'],
          'mode':mode,'image_path':item['image_path'],'image_sha256':item['image_sha256'],'shape_original_hw':item['shape'],
          'bbox_original_xyxy':item['bbox'],'scale':item['scale'],'coordinate_system':'full_image_xy',
          'entry_path':str(entry_path),'scene_path':entry.get('scene_path'),'phases':{},
          'mapping':'torch bilinear align_corners=False from 2x-area-downsampled full search ROI',
          'output_meaning':'center response strength 0..1; not calibrated probability or material-identity confidence',
          'outside_search_roi':'NaN in raw response; zero in processed response',
          'mask_prediction_available':False,'mask_limitation':'no validated full-body supervision; no claimed target body mask',
          'evaluation_target_labels_read':False}
        info['phases'][phase]=record;dump(info_path,info)
        records.append({'id':item['id'],'mode':mode,**record})
        print(json.dumps({'id':item['id'],'mode':mode,'phase':phase,'status':status,'seconds':round(record['elapsed_seconds'],2)}),flush=True)
    summary={'phase':phase,'model_stage':stage_name,'command':sys.argv,'exit_code':0,'python':sys.version,
      'torch':torch.__version__,'device':'cpu','threads':threads,'source_sha256':sha(__file__),
      'index_sha256':sha(index_path),'model_path':str(model_path),'model_sha256':model_hash,
      'elapsed_seconds':time.time()-start,'predicted_count':sum(r['status']=='predicted' for r in records),
      'unavailable_count':sum(r['status']=='entry_unavailable' for r in records),'items':records,'evaluation_labels_read':False}
    dump(run_dir/('inference_'+phase+'_summary.json'),summary)
    print(json.dumps({'completed':True,'phase':phase,'predicted':summary['predicted_count'],'unavailable':summary['unavailable_count'],'elapsed_seconds':summary['elapsed_seconds']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--model',required=True)
    p.add_argument('--phase',choices=['initial','final','overfit'],required=True);p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();run(a.run,a.model,a.phase,a.threads)
