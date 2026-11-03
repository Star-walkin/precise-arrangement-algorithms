"""Prepare gray and short-entry inputs only; never load evaluation targets."""
import sys,time
import cv2,numpy as np
from common import HERE,OLD_INDEX,load,dump,sha
def seed_map(xy):
    mask=np.zeros((975,2048),np.uint8)
    points=np.rint((np.asarray(xy,float)-[0,1050])*.5).astype(np.int32)
    cv2.polylines(mask,[points],False,1,1,cv2.LINE_8)
    assert mask.any()
    d=cv2.distanceTransform(1-mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    return np.exp(-.5*(d/2.)**2).astype(np.float32)
def main():
    t=time.time();cv2.setNumThreads(2);p=load(HERE/'protocol.json');out=HERE/'inputs';out.mkdir(exist_ok=False)
    (out/'batch').mkdir();(out/'evaluation').mkdir();jobs=[];evaluation=[]
    for j in p['jobs']:
        assert sha(j['image_path'])==j['image_sha256']
        bgr=cv2.imdecode(np.fromfile(j['image_path'],np.uint8),cv2.IMREAD_COLOR);assert bgr.shape==(3000,4096,3)
        gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
        gray=cv2.resize(gray[1050:],(2048,975),interpolation=cv2.INTER_AREA)
        seeds=np.stack([seed_map(s['xy']) for s in j['seeds']]);dest=out/'batch'/(j['id']+'.npz')
        np.savez_compressed(dest,image=gray,seeds=seeds)
        jobs.append(dict(j,input_cache=str(dest),input_cache_sha256=sha(dest)))
        if len(jobs)%50==0:print('prepared',len(jobs),flush=True)
    for item in load(OLD_INDEX)['items']:
        if item['split']!='eval':continue
        # npz central directory is read, but target/valid/stage arrays are not.
        with np.load(item['cache'],allow_pickle=False) as z:gray=z['image'].copy();known=z['seed'].copy()
        assert gray.shape==(975,2048)
        for mode in ['CONDITIONAL','AUTO']:
            if mode=='CONDITIONAL':seed=known;available=True;xy=item['known_seed_xy'];entry_source=item['entry_hint']
            else:
                entry=load(item['auto_entry_path']);available=entry['available'];xy=entry.get('free_seed',[]) if available else [];entry_source=item['auto_entry_path']
                seed=np.load(str(item['auto_entry_path']).replace('.json','.seed.npy'),allow_pickle=False).astype(np.float32) if available else np.zeros_like(gray)
            dest=out/'evaluation'/(item['id']+'__'+mode+'.npz');np.savez_compressed(dest,image=gray,seed=seed)
            evaluation.append(dict(id=item['id'],mode=mode,available=available,seed_xy=xy,entry_source=entry_source,
                input_cache=str(dest),input_cache_sha256=sha(dest),source_cache=item['cache'],source_cache_sha256=sha(item['cache']),
                image_path=item['image_path'],image_sha256=item['image_sha256'],bbox=item['bbox'],scale=item['scale'],source_group=item['source_group']))
    dump(out/'manifest.json',dict(command=sys.argv,protocol_sha256=sha(HERE/'protocol.json'),batch=jobs,evaluation=evaluation,
        image_count=len(jobs),entry_conditions=sum(len(j['seeds']) for j in jobs),eval_conditions=len(evaluation),
        labels_loaded=False,source_groups_unchanged=True,seconds=time.time()-t,exit_code=0))
    print('prepared292/301 and16 evaluation input conditions',flush=True)
if __name__=='__main__':main()
