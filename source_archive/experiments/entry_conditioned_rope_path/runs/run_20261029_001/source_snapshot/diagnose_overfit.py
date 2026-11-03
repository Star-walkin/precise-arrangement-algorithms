"""Separate four-training-image overfit diagnosis; never a baseline/eval result."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def infer(run):
    import torch
    import torch.nn.functional as F
    from train import read_arrays,tensor_sample,load_model
    run=Path(run).resolve();out=run/'overfit_predictions';out.mkdir(exist_ok=True)
    selected=load(run/'model/overfit_images.json')['ids'];items=[i for i in load(run/'data/index.json')['items'] if i['id'] in selected]
    assert len(items)==len(selected)==4 and all(i['split']=='train' for i in items)
    model=run/'model/overfit.pt';checkpoint=torch.load(model,map_location='cpu',weights_only=False)
    cfg=checkpoint['config'];net=load_model(model);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    rows=[];tick=time.time()
    with torch.inference_mode():
        for item in items:
            dest=out/item['id'];dest.mkdir(exist_ok=False);a=read_arrays(item)
            x,_,_,_,input_meta=tensor_sample(a,cfg,False);h,w=a['image'].shape
            response=net(x)[0,0,:h,:w].numpy();np.save(dest/'response_model.npy',response.astype(np.float32))
            x0,y0,x1,y1=item['bbox'];raw=F.interpolate(torch.from_numpy(response)[None,None],size=(y1-y0,x1-x0),mode='bilinear',align_corners=False)[0,0].numpy()
            full=np.full(item['shape'],np.nan,np.float32);full[y0:y1,x0:x1]=raw;np.save(dest/'response.npy',full)
            valid=a['valid'];high=valid&(a['target']>=.5);low=valid&(a['target']<.5);unknown=a['roi']&~valid
            mse_high=float(((response[high]-a['target'][high])**2).mean());mse_low=float(((response[low]-a['target'][low])**2).mean())
            row={'id':item['id'],'cache':item['cache'],'response':str(dest/'response.npy'),'response_sha256':sha(dest/'response.npy'),
              'model_stage':'overfit_stage_check_only_not_baseline','model_path':str(model),'model_sha256':sha(model),
              'input_statistics':input_meta,'supervised_pixels':int(valid.sum()),'unknown_pixels':int(unknown.sum()),
              'supervised_high_mse':mse_high,'supervised_low_mse':mse_low,'supervised_balanced_mse':.5*(mse_high+mse_low),
              'unknown_response_mean':float(response[unknown].mean()),'unknown_response_std':float(response[unknown].std()),
              'unknown_fraction_response_gt05':float((response[unknown]>.5).mean()),
              'unknown_fraction_response_gt09':float((response[unknown]>.9).mean()),
              'unknown_high_response_pixel_count':int((unknown&(response>.5)).sum()),
              'supervised_high_response_pixel_count':int((valid&(response>.5)).sum()),
              'interpretation':'unknown strong responses are not counted as false positives without reference; low supervised MSE alone does not show global target selectivity',
              'stage_statistics':{}}
            for st in range(1,5):
                v=valid&(a['stage']==st);hh=v&(a['target']>=.5);ll=v&(a['target']<.5)
                row['stage_statistics'][str(st)]={'valid_pixels':int(v.sum()),'high_pixels':int(hh.sum()),
                  'high_mse':float(((response[hh]-a['target'][hh])**2).mean()) if hh.any() else None,
                  'low_mse':float(((response[ll]-a['target'][ll])**2).mean()) if ll.any() else None}
            dump(dest/'diagnostic.json',row);rows.append(row);print(json.dumps({'overfit_diagnosed':item['id'],'balanced_mse':row['supervised_balanced_mse'],'unknown_gt05':row['unknown_fraction_response_gt05']}),flush=True)
    dump(out/'diagnostics.json',{'items':rows,'command_infer':sys.argv,'exit_code':0,'elapsed_seconds':time.time()-tick,
      'source_sha256':sha(__file__),'model_stage':'four training images overfit only; no independent evaluation','eval_labels_read':False})

def render(run):
    import cv2
    run=Path(run).resolve();out=run/'overfit_predictions';summary=load(out/'diagnostics.json')
    for row in summary['items']:
        dest=out/row['id'];a=np.load(row['cache']);response=np.load(dest/'response_model.npy')
        gray=np.uint8(a['image']*255);original=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        def color(v):return cv2.applyColorMap(np.uint8(np.clip(v,0,1)*255),cv2.COLORMAP_TURBO)
        seed=original.copy();seed[a['seed']>.1]=(0,255,255)
        target=color(a['target']);target[~a['valid']]=(80,80,80)
        heat=color(response);overlay=cv2.addWeighted(original,.55,heat,.45,0)
        supervision=np.full_like(original,55);supervision[a['valid']&(a['target']<.5)]=(255,100,0);supervision[a['valid']&(a['target']>=.5)]=(0,255,0)
        panels=[('original gray, fixed0..1',original),('entry hint: initial120px only',seed),('target response; gray=ignore',target),
          ('OVERFIT ONLY response, fixed0..1',heat),('response overlay; unknown may be high',overlay),('valid high green / low blue / ignore gray',supervision)]
        pw=1024;ph=488;header=44;tiles=[]
        for title,im in panels:
            tile=np.full((ph+header,pw,3),20,np.uint8);tile[header:]=cv2.resize(im,(pw,ph),interpolation=cv2.INTER_AREA)
            cv2.putText(tile,title,(12,29),cv2.FONT_HERSHEY_SIMPLEX,.64,(240,240,240),1,cv2.LINE_AA);tiles.append(tile)
        canvas=np.concatenate([np.concatenate(tiles[:3],1),np.concatenate(tiles[3:],1)],0)
        footer=np.full((72,canvas.shape[1],3),20,np.uint8)
        bar=color(np.linspace(0,1,800,dtype=np.float32)[None,:]);footer[17:42,120:920]=cv2.resize(bar,(800,25))
        cv2.putText(footer,'0',(85,38),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1)
        cv2.putText(footer,'1',(936,38),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1)
        txt=f"supervised balanced MSE={row['supervised_balanced_mse']:.5f}; ignore-area response>0.5: {100*row['unknown_fraction_response_gt05']:.2f}% (not an accuracy/FP metric)"
        cv2.putText(footer,txt,(1040,38),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),1,cv2.LINE_AA)
        canvas=np.concatenate([canvas,footer],0);cv2.imencode('.png',canvas)[1].tofile(str(dest/'comparison.png'))
        row['visualization']=str(dest/'comparison.png')
    summary['command_render']=sys.argv;summary['display_rule']='original gray and all response maps fixed0..1, Turbo colorbar; ignore gray80; no per-image stretch'
    dump(out/'diagnostics.json',summary);print('Four fixed-scale overfit diagnostics rendered')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['infer','render']);p.add_argument('--run',required=True);a=p.parse_args()
    (infer if a.mode=='infer' else render)(a.run)
