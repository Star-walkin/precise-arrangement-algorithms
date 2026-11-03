"""Assemble frozen reviewed automatic weak labels; all unknown pixels ignored."""
from pathlib import Path
import json,hashlib,sys,collections
import cv2,numpy as np
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from prepare import response_from_lines
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def length(p):return float(np.linalg.norm(np.diff(np.asarray(p,float),axis=0),axis=1).sum())
def main():
    cv2.setNumThreads(2);pr=load(HERE/'protocol.json');old=load(BASE/'runs/run_20261030_002/data/index.json')
    dest=HERE/'dataset';dest.mkdir(exist_ok=False);(dest/'cache').mkdir()
    items=[dict(i,origin='old') for i in old['items'] if i['split']=='train'];assert len(items)==10
    reports=[];status=collections.Counter();phase=collections.Counter();candidates={j['id']:j for j in pr['candidate_jobs']};imageset=set()
    annotations=[]
    for folder in ['labels_left','labels_right']:
        for p in (HERE/folder).glob('*.json'):
            a=load(p)
            if not isinstance(a,dict) or a.get('id') not in candidates or 'segments' not in a:continue
            assert a['id'] not in imageset,'duplicate annotation';imageset.add(a['id']);annotations.append((p,a))
    assert len(annotations)==24,'All24 candidates must have recorded quality outcome'
    for path,a in annotations:
        j=candidates[a['id']];assert a['image_sha256']==j['image_sha256']==sha(j['image_path'])
        assert a['source_group'] in pr['train_groups'] and a['source_group'] not in pr['eval_groups']
        rev=a['review'];assert rev['original_viewed'] and rev['native_overlay_viewed'],a['id']
        assert np.allclose(a['seed_xy'],j['seeds'][0]['xy'])
        target=np.zeros((975,2048),np.float32);valid=np.zeros_like(target,bool);stage=np.zeros_like(target,np.uint8)
        added=[];uncertain=[]
        for seg in a['segments']:
            status[seg['status']]+=1
            if seg['status']!='accepted_weak':uncertain.append(seg);continue
            line=np.asarray(seg['points_xy'],float);assert len(line)>=2 and np.isfinite(line).all()
            assert (line[:,0]>=0).all() and (line[:,0]<4096).all() and (line[:,1]>=1050).all() and (line[:,1]<3000).all()
            s=int(seg['stage']);assert s in (1,2,3,4)
            hm,d=response_from_lines([line],target.shape,[0,1050,4096,3000],.5,3.)
            radius=float(seg['valid_radius_px']);assert 3.6<=radius<=9.,(a['id'],radius)
            vv=d<=radius*.5
            # End caps are omitted: end of annotated fragment isn't a target endpoint.
            yy,xx=np.indices(target.shape)
            for end in [line[0],line[-1]]:
                q=(end-[0,1050])*.5;vv&=((xx-q[0])**2+(yy-q[1])**2)>3.**2
            target=np.maximum(target,hm*vv);stage[vv]=s;valid|=vv
            ll=length(line);phase[str(s)]+=ll;added.append(dict(length_original_px=ll,stage=s,valid_pixels=int(vv.sum()),radius=radius))
        if not added:
            reports.append(dict(id=a['id'],accepted=False,reason='No adequately supported center after review',unknown=uncertain));continue
        assert np.any(valid&(target>=.5)) and np.any(valid&(target<.5))
        bgr=cv2.imdecode(np.fromfile(j['image_path'],np.uint8),cv2.IMREAD_COLOR)
        gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
        image=cv2.resize(gray[1050:],(2048,975),interpolation=cv2.INTER_AREA)
        seed,_=response_from_lines([j['seeds'][0]['xy']],image.shape,[0,1050,4096,3000],.5,4.)
        roi=np.ones(image.shape,bool);target[~valid]=0
        cache=dest/'cache'/(a['id']+'.npz');np.savez_compressed(cache,image=image,seed=seed,target=target,valid=valid,stage=stage,roi=roi)
        items.append(dict(id=a['id'],split='train',origin='new',source_group=j['source_group'],image_path=j['image_path'],image_sha256=j['image_sha256'],
            cache=str(cache),cache_sha256=sha(cache),known_seed_xy=j['seeds'][0]['xy'],bbox=[0,1050,4096,3000],scale=.5,shape=[3000,4096],
            annotation_path=str(path),annotation_sha256=sha(path),annotation_status='accepted_automatic_weak_not_independent_human_truth'))
        reports.append(dict(id=a['id'],accepted=True,source_group=j['source_group'],segments=added,uncertain=uncertain,
            target_length_original_px=sum(r['length_original_px'] for r in added),valid_pixels=int(valid.sum()),ignore_fraction=float((~valid).mean()),
            coordinates='original_xy to crop_half=(xy-[0,1050])*0.5; native profile refinement; sigma3original unchanged'))
    new=[i for i in items if i['origin']=='new'];assert new
    stats=dict(candidate_images=24,new_unique_images=len(new),old_images=10,new_groups=sorted({i['source_group'] for i in new}-set(pr['old_train_groups'])),
        all_training_groups=sorted({i['source_group'] for i in items}),eval_groups=pr['eval_groups'],
        source_group_intersection_with_eval=list({i['source_group'] for i in items}&set(pr['eval_groups'])),
        new_center_length_original_px=sum(r.get('target_length_original_px',0) for r in reports),new_center_length_by_visual_phase=dict(phase),
        segment_status_counts=dict(status),new_independent_human_labels=0,non_center_pixels_unknown=True,
        confidence_loss_used=False,scope='weak raw-image/Codex checked partial centers; not complete attached identity/endpoint truth',items=reports)
    dump(dest/'statistics.json',stats)
    dump(dest/'index.json',dict(frozen=True,items=items,source_protocol_sha256=sha(HERE/'protocol.json'),labels_frozen_before_training=True,
        label_hashes={str(p):sha(p) for p,a in annotations},statistics=str(dest/'statistics.json'),weak_weight=1.,stage_is_visual_proxy=True))
    print(json.dumps({k:v for k,v in stats.items() if k!='items'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
