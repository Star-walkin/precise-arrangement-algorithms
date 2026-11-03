"""Frozen A750 response and spatial feature cache. Never opens target labels."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'2')
import sys,time,concurrent.futures,traceback
from common import *
import numpy as np
sys.path.append(str(ROOT/'vision_prototype/.venv/Lib/site-packages'))
import cv2
import torch
import torch.nn.functional as F

def seed_map(xy,shape):
    m=np.zeros(shape,np.uint8);q=np.rint((np.asarray(xy)-[0,1050])*.5).astype(np.int32)
    cv2.polylines(m,[q],False,1,1,cv2.LINE_8)
    assert m.any(),'entry outside model crop'
    d=cv2.distanceTransform(1-m,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    return np.exp(-.5*(d/2.)**2).astype(np.float32)
NET=None
def init():
    global NET
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.use_deterministic_algorithms(True)
    p=load(HERE/'protocol.json')['models']['A'];assert sha(p['path'])==p['sha256']
    cp=torch.load(p['path'],map_location='cpu',weights_only=False)
    m=mod('_r1_original_model',BASE/'model.py');NET=m.EntryUNet(cp['base_channels'])
    NET.load_state_dict(cp['model_state']);NET.eval()
    for p in NET.parameters():p.requires_grad_(False)
def jobs():
    protocol=load(HERE/'protocol.json');ix=load(protocol['index_path']);rows=[]
    for j in protocol['jobs']:
        for k,s in enumerate(j['seeds']):rows.append(dict(key='gallery/'+j['id']+f'/entry_{k:02d}',id=j['id'],mode='GALLERY_AUTO',entry=k,seed_xy=s['xy'],image_path=j['image_path'],image_sha256=j['image_sha256'],group=j['source_group']))
    for i in ix['items']:
        rows.append(dict(key='fixed/CONDITIONAL/'+i['id'],id=i['id'],mode='CONDITIONAL',entry=0,seed_xy=i['known_seed_xy'],image_path=i['image_path'],image_sha256=i['image_sha256'],group=i['source_group'],split=i['split']))
        if i['split']=='eval':
            auto=load(i['auto_entry_path'])
            rows.append(dict(key='fixed/AUTO/'+i['id'],id=i['id'],mode='AUTO',entry=0,seed_xy=auto.get('free_seed',[]) if auto['available'] else [],image_path=i['image_path'],image_sha256=i['image_sha256'],group=i['source_group'],split=i['split']))
    for j in rows:
        assert '20260925' not in j['image_path']
        j['cache_path']=str(HERE/'cache'/Path(j['key']+'.npz'))
    return rows
def one(j):
    dest=Path(j['cache_path']);meta=dest.with_suffix('.json')
    if meta.exists():
        r=load(meta);assert r['job']==j and (not r['available'] or sha(dest)==r['sha256']);return r
    tick=time.time();assert sha(j['image_path'])==j['image_sha256']
    # One historical fixed case needs original8-thread numerics for bitwise
    # replay. Its inputs were independently found bit-identical;2vs8 difference
    # was2.115965e-6. Preserve the failed numerical gate and recompute, do not
    # loosen the gate or alter response values.
    if j['key']=='fixed/CONDITIONAL/dual_20260923_113745__left__f00580':
        torch.set_num_threads(8)
        if dest.exists():
            import shutil
            old=dest.with_name(dest.stem+'.two_thread_failed_gate.npz')
            if not old.exists():shutil.copyfile(dest,old)
    if not j['seed_xy']:
        r=dict(job=j,available=False,status='automatic_entry_unavailable',seconds=time.time()-tick)
        dump(meta,r);return r
    bgr=cv2.imdecode(np.fromfile(j['image_path'],np.uint8),cv2.IMREAD_COLOR);assert bgr.shape==(3000,4096,3)
    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    im=cv2.resize(gray[1050:],(2048,975),interpolation=cv2.INTER_AREA);se=seed_map(j['seed_xy'],im.shape)
    g=torch.from_numpy(im);mean=g.mean();std=g.std(unbiased=False).clamp_min(.08)
    x=F.pad(torch.stack([(g-mean)/std,torch.from_numpy(se)]),(0,0,0,17))[None]
    capture={}
    def hook(m,args,out):capture['features']=F.avg_pool2d(out,4,4).detach().to(torch.float16).cpu().numpy()[0]
    h=NET.dec1.register_forward_hook(hook)
    with torch.inference_mode():response=NET(x)[0,0,:975,:2048].numpy().copy()
    h.remove();feat=capture['features'];assert feat.shape==(8,248,512)
    dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,response=response,features=feat)
    # Verify agreement with existing A750 fixed-condition response if available.
    replay=None;old=PAIR/'branches/A/predictions'/j['mode']/j['id']/'response.npy'
    if old.exists():
        full=np.load(old,mmap_mode='r');up=F.interpolate(torch.from_numpy(response)[None,None],size=(1950,4096),mode='bilinear',align_corners=False)[0,0].numpy()
        replay=float(np.max(abs(full[1050:]-up)));assert replay<2e-6
    r=dict(job=j,available=True,sha256=sha(dest),model=load(HERE/'protocol.json')['models']['A'],
        shape=[975,2048],feature_shape=list(feat.shape),feature_source='frozen dec1 spatial average_pool4; no image-global averaging',
        feature_mapping='model_pixel=(original_xy-[0,1050])/2; pooled_pixel=(model_pixel+0.5)/4-0.5',
        mean=float(mean),std=float(std),range=[float(response.min()),float(response.max())],old_A_replay_max_abs=replay,seconds=time.time()-tick)
    dump(meta,r);return r
def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--subset',choices=['fixed','gallery','all'],default='all');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    sys.stdout.reconfigure(encoding='utf8');alljobs=jobs();selected=[j for j in alljobs if a.subset=='all' or j['key'].split('/')[0]==a.subset]
    dump(HERE/'cache_jobs.json',alljobs);tick=time.time();rows=[];errors=[]
    with concurrent.futures.ProcessPoolExecutor(a.workers,initializer=init) as ex:
        pending={ex.submit(one,j):j['key'] for j in selected}
        for fut in concurrent.futures.as_completed(pending):
            try:rows.append(fut.result())
            except Exception:errors.append(dict(key=pending[fut],traceback=traceback.format_exc()))
            if (len(rows)+len(errors))%20==0 or len(rows)+len(errors)==len(selected):print(dict(done=len(rows),errors=len(errors),total=len(selected),seconds=round(time.time()-tick,1)),flush=True)
    dump(HERE/f'cache_{a.subset}_receipt.json',dict(command=sys.argv,rows=rows,errors=errors,seconds=time.time()-tick,exit_code=int(bool(errors)),new_model_forward_count=sum(r['available'] for r in rows),labels_read=False,devices=False))
    if errors:print(errors[:1]);sys.exit(bool(errors))
if __name__=='__main__':main()
