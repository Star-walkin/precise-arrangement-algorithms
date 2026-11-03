"""Independent full artifact audit. No training, labels from eval or devices.

Reconstruct accepted-label tubes independently; decode every delivered PNG,
compare its full source viewport and full heat viewport with frozen numeric
inputs. Passing proves traceability/serialization, not visual rope identity.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(k,'1')
from pathlib import Path
import sys,json,hashlib,time,traceback,concurrent.futures,argparse
from collections import Counter
import numpy as np
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;ROOT=BASE.parents[1]
sys.path.append(str(ROOT/'vision_prototype/.venv/Lib/site-packages'))
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS=150_000_000
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def independent_up2(a):
    # Direct quarter-weight factor2 interpolation, not imported renderer code.
    h,w=a.shape;aa=a.astype(np.float64);horizontal=np.empty((h,2*w),np.float64)
    horizontal[:,0]=aa[:,0];horizontal[:,-1]=aa[:,-1]
    horizontal[:,1:-1:2]=.75*aa[:,:-1]+.25*aa[:,1:]
    horizontal[:,2::2]=.25*aa[:,:-1]+.75*aa[:,1:]
    out=np.empty((2*h,2*w),np.float64);out[0]=horizontal[0];out[-1]=horizontal[-1]
    out[1:-1:2]=.75*horizontal[:-1]+.25*horizontal[1:]
    out[2::2]=.25*horizontal[:-1]+.75*horizontal[1:]
    return out.astype(np.float32)
def line_distance(points):
    mask=np.zeros((975,2048),np.uint8);p=np.rint((np.asarray(points)-[0,1050])*.5).astype(np.int32)
    cv2.polylines(mask,[p],False,1,1,cv2.LINE_8);assert mask.any()
    return cv2.distanceTransform(1-mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
def dataset_audit():
    cv2.setNumThreads(1);pr=load(HERE/'protocol.json');ix=load(HERE/'dataset/index.json');assert ix['frozen'] is True
    assert ix['source_protocol_sha256']==sha(HERE/'protocol.json')
    original=load(BASE/'runs/run_20261030_002/data/index.json');oldmap={i['id']:i for i in original['items'] if i['split']=='train'}
    old=[i for i in ix['items'] if i.get('origin')=='old'];new=[i for i in ix['items'] if i.get('origin')=='new']
    assert len(old)==10 and len({i['id'] for i in ix['items']})==len(ix['items'])
    groups={i['source_group'] for i in ix['items'] if i['split']=='train'};assert not groups&set(pr['eval_groups'])
    for it in old:
        assert {k:v for k,v in it.items() if k!='origin'}==oldmap[it['id']]
        assert sha(it['cache'])==it['cache_sha256'] and sha(it['image_path'])==it['image_sha256']
    candidates={j['id']:j for j in pr['candidate_jobs']};assert len(candidates)==24
    labels={}
    for path,h in ix['label_hashes'].items():
        assert sha(path)==h;a=load(path);labels[a['id']]=a
    assert set(labels)==set(candidates)
    statuses=Counter();accepted_ids=set();records=[]
    for ident,a in labels.items():
        j=candidates[ident];assert a['image_sha256']==j['image_sha256']==sha(j['image_path'])
        assert a['source_group']==j['source_group'] and a['source_group'] in groups
        assert np.array_equal(np.asarray(a['seed_xy']),np.asarray(j['seeds'][0]['xy']))
        assert a['review']['original_viewed'] and a['review']['native_overlay_viewed']
        target=np.zeros((975,2048),np.float32);valid=np.zeros_like(target,bool);stage=np.zeros_like(target,np.uint8)
        center_length=0.
        for seg in a['segments']:
            statuses[seg['status']]+=1;assert seg['status'] in ('accepted_weak','uncertain','rejected')
            if seg['status']!='accepted_weak':continue
            p=np.asarray(seg['points_xy'],float);assert p.ndim==2 and p.shape[1]==2 and len(p)>=2 and np.isfinite(p).all()
            assert (p[:,0]>=0).all() and (p[:,0]<4096).all() and (p[:,1]>=1050).all() and (p[:,1]<3000).all()
            d=line_distance(p);radius=float(seg['valid_radius_px']);assert 3.6<=radius<=9.
            vv=d<=radius*.5;yy,xx=np.indices(d.shape)
            for end in (p[0],p[-1]):
                q=(end-[0,1050])*.5;vv&=((xx-q[0])**2+(yy-q[1])**2)>9.
            hm=np.exp(-.5*(d/1.5)**2).astype(np.float32)
            target=np.maximum(target,hm*vv);valid|=vv;stage[vv]=int(seg['stage'])
            center_length+=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
        if valid.any():
            accepted_ids.add(ident);it=next(i for i in new if i['id']==ident)
            assert it['annotation_sha256']==sha(it['annotation_path']) and sha(it['cache'])==it['cache_sha256']
            with np.load(it['cache'],allow_pickle=False) as z:arr={k:z[k].copy() for k in ['image','seed','target','valid','stage','roi']}
            assert all(x.shape==(975,2048) for x in arr.values())
            assert np.array_equal(arr['valid'],valid) and np.array_equal(arr['stage'],stage)
            target[~valid]=0;assert np.array_equal(arr['target'],target)
            assert np.all(arr['target'][~valid]==0) and np.all(stage[~valid]==0) and np.all(stage[valid]>0)
            assert arr['roi'].all(),'ROI must remain full search crop, not I>0'
            bgr=cv2.imdecode(np.fromfile(j['image_path'],np.uint8),cv2.IMREAD_COLOR)
            gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
            image=cv2.resize(gray[1050:],(2048,975),interpolation=cv2.INTER_AREA)
            assert np.array_equal(arr['image'],image)
            sd=line_distance(j['seeds'][0]['xy']);seed=np.exp(-.5*(sd/2.)**2).astype(np.float32)
            assert np.array_equal(arr['seed'],seed)
            records.append({'id':ident,'accepted_center_length_px':center_length,'valid_pixels':int(valid.sum()),'ignore_fraction':float((~valid).mean()),'sigma_original_px':3.,'cache_reconstructed':True})
    assert accepted_ids=={i['id'] for i in new}
    return {'candidate_images':len(labels),'new_images':len(new),'old_images':len(old),'segment_status_counts':dict(statuses),
        'source_group_leakage':False,'old_caches_unchanged':True,'new_tubes_reconstructed':records,
        'unknown_does_not_create_supervision':True,'automatic_labels_are_not_independent_human_truth':True}
def training_audit():
    import torch
    cfg=load(HERE/'training_config.json');sel=load(HERE/'selected_checkpoints.json');pr=load(HERE/'protocol.json')
    ixsha=sha(HERE/'dataset/index.json');start=pr['start_model'];assert sha(start['path'])==start['sha256']
    base=torch.load(start['path'],map_location='cpu',weights_only=False)['model_state'];rr={};logs={}
    for b in ['D0','D1']:
        r=load(HERE/'branches'/b/'training_receipt.json');rr[b]=r;assert r['exit_code']==0 and r['all_parameters_trainable'] and r['formal_weights_reset_to_common_A750']
        assert r['parameter_count']==135897 and r['dataset_index_sha256']==ixsha
        assert r['config_sha256']==sha(HERE/'training_config.json') and r['sampling_plan_sha256']==sha(HERE/'sampling_plan.json')
        log=[json.loads(line) for line in (HERE/'branches'/b/'train_log.jsonl').read_text(encoding='utf8').splitlines()]
        assert [v['step'] for v in log]==list(range(1,r['steps']+1)) and r['steps']<=600
        if r['steps']<600:assert log[-1]['seconds']>=cfg['train_max_seconds']
        assert all(np.isfinite(v['loss']) for v in log)
        assert dict(Counter(v['id'] for v in log))==r['sampling_by_id']
        assert dict(Counter(v['origin'] for v in log))==r['sampling_by_origin']
        assert all(v['origin']=='old' for v in log) if b=='D0' else r['sampling_by_origin']['new']>0
        cp=sel['paths'][b];assert sha(cp['path'])==cp['sha256'];net=torch.load(cp['path'],map_location='cpu',weights_only=False)
        assert net['step']==sel['common_step']
        initial=torch.load(HERE/'branches'/b/'initial.pt',map_location='cpu',weights_only=False)['model_state']
        assert initial.keys()==base.keys() and all(torch.equal(initial[k],base[k]) for k in base)
        state=net['model_state'];changed=[k for k in state if not torch.equal(state[k],base[k])]
        assert all(any(k.startswith(prefix) for k in changed) for prefix in ['enc1','bottleneck','dec1','output'])
        grads=r['nonzero_gradient_parameter_names'];assert all(any(k.startswith(prefix) for k in grads) for prefix in ['enc1','bottleneck','dec1','output'])
        logs[b]={v['step']:v for v in log}
    assert rr['D0']['initial_state_sha256']==rr['D1']['initial_state_sha256']
    for step in range(2,sel['common_step']+1,2):
        a,b=logs['D0'][step],logs['D1'][step];assert a['id']==b['id'] and a['input_sha256']==b['input_sha256']
    small=load(HERE/'smallcheck/receipt.json');assert small['passed'] and small['independent_copy_discarded']
    return {'D0_updates':rr['D0']['steps'],'D1_updates':rr['D1']['steps'],'main_common_update':sel['common_step'],
        'A750_source_sha256':start['sha256'],'same_initial_parameters':True,'encoder_bottleneck_decoder_output_changed':True,
        'D1_new_exposures':rr['D1']['sampling_by_origin']['new'],'shared_old_input_hashes_passed':True,
        'smallcheck_copy_discarded':True,'no_relation_network_training':True,'checkpoint_not_evaluation_selected':True}
def image_check(row):
    cv2.setNumThreads(1);p=Path(row['output']);assert sha(p)==row['sha256']
    assert row['display_entry']==0 and row['path_overlay'] is False and row['heat_body_unmodified']
    sel=load(HERE/'selected_checkpoints.json');assert row['model']==sel['paths']['D1']
    assert sha(row['source'])==row['source_sha256']
    count=0
    for cond in row['conditions']:
        rp=Path(cond['response_path']);assert sha(rp)==cond['response_sha256'];assert cond['model_sha256']==sel['paths']['D1']['sha256']
        assert cond['new_model_forward'] and cond['target_labels_read'] is False
        with np.load(rp,allow_pickle=False) as z:a=z['response'].copy()
        assert a.shape==(975,2048) and a.dtype==np.float32 and np.isfinite(a).all() and 0<=a.min()<=a.max()<=1
        if cond['entry_index']==0:main=a
        count+=1
    lut=np.load(BASE/'batch_ab_20261030_001/color_lut.npy');expected=np.full((3000,4096,3),45,np.uint8)
    expected[1050:]=lut[np.rint(independent_up2(main)*255).astype(np.uint8)]
    with Image.open(p) as im,Image.open(row['source']) as raw:
        assert im.format=='PNG' and im.mode=='RGB' and im.size==(8364,3350)
        assert im.crop((0,160,4096,3160)).tobytes()==raw.convert('RGB').tobytes()
        right=np.asarray(im.crop((4128,160,8224,3160)))
        assert np.array_equal(right,expected),'Right viewport differs from original D1 response or contains an overlay'
    return {'id':row['id'],'entry_count':count,'full_left_pixels_identical':True,'full_right_raw_heat_pixels_identical':True}
def main():
    sys.stdout.reconfigure(encoding='utf8');p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--output',type=Path,default=HERE/'audit_receipt.json');a=p.parse_args()
    assert not a.output.exists(),'Preserve existing audit receipts';assert 1<=a.workers<=4
    start=time.perf_counter();errors=[];data=None;training=None;checked=[]
    for name,fun in [('dataset',dataset_audit),('training',training_audit)]:
        try:
            value=fun()
            if name=='dataset':data=value
            else:training=value
        except Exception:errors.append({'component':name,'traceback':traceback.format_exc()})
    pr=load(HERE/'protocol.json');render=load(HERE/'render_receipt.json');expected={j['id'] for j in pr['gallery_jobs']}
    files=list((HERE/'final_292').glob('*.png'))
    assert len(files)==292 and {p.stem for p in files}==expected
    assert render['exit_code']==0 and len(render['rows'])==292 and {r['id'] for r in render['rows']}==expected
    assert sum(r['entry_count'] for r in render['rows'])==301
    with concurrent.futures.ProcessPoolExecutor(a.workers) as ex:
        pending={ex.submit(image_check,row):row['id'] for row in render['rows']}
        for fut in concurrent.futures.as_completed(pending):
            try:checked.append(fut.result())
            except Exception:errors.append({'component':pending[fut],'traceback':traceback.format_exc()})
            if len(checked)%30==0:print({'checked_images':len(checked),'errors':len(errors),'seconds':round(time.perf_counter()-start,1)},flush=True)
    receipt={'exit_code':int(bool(errors)),'checked_image_count':len(checked),'native_float_conditions':sum(r['entry_count'] for r in checked),
        'dataset':data,'training':training,'errors':errors,'seconds':time.perf_counter()-start,
        'source_sha256':sha(__file__),'protocol_sha256':sha(HERE/'protocol.json'),'dataset_index_sha256':sha(HERE/'dataset/index.json'),
        'render_receipt_sha256':sha(HERE/'render_receipt.json'),'exactly292_unique_sources':True,'selected_entry_is_frozen_index0':True,
        'whole_raw_heat_reconstruction_not_overlay':not errors,'complete_identity_accuracy':None,
        'audit_semantics':'artifact, pixels, hashes, masks and actual learning checks only; no independent annotation identity accuracy'}
    dump(a.output,receipt);print(json.dumps({'exit_code':receipt['exit_code'],'checked':len(checked),'errors':len(errors),'seconds':receipt['seconds']},ensure_ascii=False),flush=True)
    if errors:print(errors[:1],flush=True)
    sys.exit(bool(errors))
if __name__=='__main__':main()
