"""Frozen open400 entries and exact A/B750 missing-float inference only.

No training or image-based entry revision; JPEG previews are never read.
"""
from pathlib import Path
import argparse,json,csv,hashlib,sys,time
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
CODE=Path(__file__).resolve().parent;ROOT=CODE.parents[1]
OLD_JOBS=CODE/'batch_ab_20261030_001/jobs.json'
SOURCE=ROOT/'数据资产/05_派生与实验结果/camera_condition_images_20261009_v2'
PAIR=CODE/'runs/run_20261030_002/paired_comparison.json'
EXAMPLE='dual_20260923_101517__pair0075__right'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def prepare(run):
    import cv2
    from prepare import response_from_lines
    from scene import build_scene
    from vision_geometry_lab.arc_detector import detect_arc
    cv2.setNumThreads(2)
    out=run/'batch';out.mkdir(exist_ok=False);cache=out/'input_cache';cache.mkdir()
    prev=load(OLD_JOBS);pair=load(PAIR);rows=list(csv.DictReader((SOURCE/'manifest.csv').open(encoding='utf-8-sig')))
    assert len(rows)==400;lookup={Path(r['output_file']).stem:r for r in rows}
    jobs=sorted(prev['jobs'],key=lambda j:(j['id']!=EXAMPLE,j['id']));prepared=[];tick=time.perf_counter()
    for n,j in enumerate(jobs,1):
        r=lookup[j['id']];assert '20260925' not in r['source_file']+r['output_file']
        assert not any(s in r['dataset_use_status'].lower() for s in ['reserved','heldout'])
        path=SOURCE/r['output_file'];assert sha(path)==j['image_sha256']==r['output_sha256']
        im=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR);assert im.shape[:2]==(3000,4096)
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
        model_image=cv2.resize(gray[1050:],(2048,975),interpolation=cv2.INTER_AREA)
        arc=detect_arc(im);conditions=[];seeds=[]
        for k,entry in enumerate(j['seeds']):
            seed,_=response_from_lines([entry['xy']],model_image.shape,j['bbox'],.5,4.);seeds.append(seed)
            folder=out/'conditions'/j['id']/f'entry_{k+1:02d}'
            scene=build_scene(im,entry['xy'],arc=arc);save(folder/'scene.json',scene)
            rec={'id':j['id'],'condition_id':f"{j['id']}__entry_{k+1:02d}",'entry_number':k+1,'entry_count':len(j['seeds']),
              'entry':entry,'entry_xy':entry['xy'],'scene_path':str(folder/'scene.json'),'response_paths':{v:str(folder/v/'response.npy') for v in ['A','B']}}
            save(folder/'entry.json',rec);conditions.append(rec)
        cp=cache/(j['id']+'.npz');np.savez_compressed(cp,image=model_image,seeds=np.stack(seeds))
        prepared.append({'id':j['id'],'image_path':str(path),'image_sha256':j['image_sha256'],'source_metadata':r,
            'cache':str(cp),'bbox':j['bbox'],'shape':j['shape'],'conditions':conditions})
        if n==1 or n%40==0:print(json.dumps({'prepared':n,'total':len(jobs),'seconds':time.perf_counter()-tick}),flush=True)
    save(out/'manifest.json',{'jobs':prepared,'skipped':prev['skipped'],'source_count':400,'models':pair['main_checkpoints'],'step':750,
      'source_manifest_sha256':sha(SOURCE/'manifest.csv'),'old_jobs_sha256':sha(OLD_JOBS),'old_jobs_source':str(OLD_JOBS),
      'image_count':len(prepared),'conditions':sum(len(j['conditions']) for j in prepared),'example_id':EXAMPLE,
      'mode':'saved_A6_AUTO_entries_with_three_observed_free_A2_fallbacks; distinct from formal8 V28 AUTO',
      'missing_float_reason':'prior batch saved JPEG previews only; no original float retained',
      'inference_allowed':'single forward per missing frozen-checkpoint/entry pair, identical previous input; no retraining or photometry change',
      'command':sys.argv,'exit_code':0,'source_sha256':sha(__file__)})

def infer(run):
    import torch
    import torch.nn.functional as F
    from train import load_model,tensor_sample
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    out=run/'batch';manifest=load(out/'manifest.json');nets={};cfg=None
    for key,value in manifest['models'].items():
        assert sha(value['path'])==value['sha256']
        cp=torch.load(value['path'],map_location='cpu',weights_only=False);assert cp['step']==750
        if cfg is None:cfg=cp['config']
        else:assert cfg==cp['config']
        nets[key]=load_model(value['path'])
    results=[];tick=time.perf_counter()
    with (out/'inference_progress.jsonl').open('x',encoding='utf-8') as log,torch.inference_mode():
        for n,job in enumerate(manifest['jobs'],1):
            with np.load(job['cache'],allow_pickle=False) as z:im=z['image'];seeds=z['seeds']
            zero=np.zeros_like(im);h,w=job['shape'];hh,ww=im.shape
            for k,condition in enumerate(job['conditions']):
                a={'image':im,'seed':seeds[k],'target':zero,'valid':zero.astype(bool),'stage':zero.astype(np.uint8),'roi':np.ones_like(im,bool)}
                x,_,_,_,stats=tensor_sample(a,cfg,augment=False)
                input_hash=hashlib.sha256(x.numpy().tobytes()).hexdigest()
                for variant,net in nets.items():
                    dest=Path(condition['response_paths'][variant]);dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
                    start=time.perf_counter();raw=net(x)[0,0,:hh,:ww]
                    raw=F.interpolate(raw[None,None],size=(h-1050,w),mode='bilinear',align_corners=False)[0,0].numpy()
                    assert np.isfinite(raw).all() and 0<=raw.min()<=raw.max()<=1
                    full=np.full((h,w),np.nan,np.float32);full[1050:]=raw;np.save(dest,full)
                    rec={'id':job['id'],'condition_id':condition['condition_id'],'variant':variant,'response_path':str(dest),
                      'response_sha256':sha(dest),'checkpoint':manifest['models'][variant],'step':750,'entry':condition['entry'],
                      'entry_xy':condition['entry_xy'],'image_sha256':job['image_sha256'],'image_path':job['image_path'],'input_tensor_sha256':input_hash,
                      'bbox_original_xyxy':job['bbox'],'original_shape_hw':job['shape'],'input_statistics':stats,
                      'scale':.5,'mapping':'gray INTER_AREA 4096x1950 to2048x975; standardized neutral pad to2048x992; remove padding then bilinear align_corners=False; paste original y1050:3000',
                      'outside_compute_domain':'NaN, not grayscale-zero background','float_source':'new single frozen forward because prior JPEG batch retained no float',
                      'forward_seconds':time.perf_counter()-start,'no_training':True,'no_label_access':True}
                    save(dest.parent/'inference.json',rec);results.append(rec);log.write(json.dumps(rec,ensure_ascii=False)+'\n');log.flush()
                    del full,raw
            if n==1 or n%30==0:print(json.dumps({'inferred_images':n,'total':len(manifest['jobs']),'seconds':time.perf_counter()-tick}),flush=True)
            cp=Path(job['cache']).resolve();assert cp.parent==(out/'input_cache').resolve();cp.unlink()
    save(out/'inference_receipt.json',{'exit_code':0,'images':len(manifest['jobs']),'forward_calls':len(results),'seconds':time.perf_counter()-tick,
      'command':sys.argv,'models':manifest['models'],'records':results,'no_training':True})
    print(json.dumps({'complete':True,'forward_calls':len(results),'seconds':time.perf_counter()-tick}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','infer']);p.add_argument('--run',required=True,type=Path)
    a=p.parse_args();(prepare if a.mode=='prepare' else infer)(a.run.resolve())
