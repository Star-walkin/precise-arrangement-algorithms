"""Prepare the frozen V11 10/8 split without loading evaluation target geometry.

Only dedicated free_seed JSON values are extracted for evaluation conditions.
Training quality exclusions clip geometry BEFORE any target rasterization.
"""
from __future__ import annotations
import argparse, hashlib, json, platform, re, sys, time
from pathlib import Path
import cv2
import numpy as np
from scene import ROOT, build_scene, short_seed, stage_at_points
from vision_geometry_lab.centerline_quality_v5 import load_label_quality
from vision_geometry_lab.centerline_path_metrics_v6 import clip_scoring_segments

MANIFEST=ROOT/'vision_geometry_lab/outputs/v11_path_identity/development_manifest.json'
V28=ROOT/'vision_geometry_lab/outputs/v28_local_spool_connection/v28_20261007_002'
PARAMETERS={'scale':.5,'crop_top_fraction':.35,'center_sigma_original_px':3.,
  'center_support_sigmas':3.,'outside_negative_width_original_px':4.,
  'quality_guard_original_px':9.,'seed_length_original_px':120.,
  'seed_sigma_original_px':4.,'roi_definition':'entire original-image search rectangle; true zero gray retained',
  'input':'original grayscale / 255; no LCN, no surface-only/effective mask',
  'target':'center-structure response, not calibrated probability or true body mask',
  'negative_definition':'low center response inside reviewed two-sided rope strip and <=4px outside reviewed sides; uncertain ends excluded',
  'unknown_definition':'all other pixels, unreviewed/inferred/coarse regions and missing continuation ignored'}

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def extract_seed_only(path):
    """Extract the dedicated hint field without parsing other label fields."""
    raw=Path(path).read_text(encoding='utf-8-sig')
    match=re.search(r'"free_seed"\s*:\s*\[',raw)
    if not match:raise ValueError('No dedicated free_seed hint')
    begin=match.end()-1;depth=0
    for end in range(begin,len(raw)):
        if raw[end]=='[':depth+=1
        elif raw[end]==']':
            depth-=1
            if depth==0:return json.loads(raw[begin:end+1])
    raise ValueError('Unclosed seed hint')

def transform(points,bbox,scale):
    return (np.asarray(points,float)-np.asarray(bbox[:2]))*scale

def line_mask(segments,shape,bbox,scale):
    out=np.zeros(shape,np.uint8)
    for s in segments:
        p=transform(s,bbox,scale)
        if len(p)>=2:cv2.polylines(out,[np.rint(p).astype(np.int32)],False,1,1,cv2.LINE_8)
    return out

def polygon_mask(polygons,shape,bbox,scale):
    out=np.zeros(shape,np.uint8)
    for p in polygons:
        if len(p)>=3:cv2.fillPoly(out,[np.rint(transform(p,bbox,scale)).astype(np.int32)],1)
    return out

def response_from_lines(segments,shape,bbox,scale,sigma):
    mask=line_mask(segments,shape,bbox,scale)
    if not mask.any():return np.zeros(shape,np.float32),np.full(shape,np.inf,np.float32)
    distance=cv2.distanceTransform(1-mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    return np.exp(-.5*(distance/(sigma*scale))**2).astype(np.float32),distance

def training_target(sample,shape,bbox,scale):
    label=load(sample['annotation_path'])
    assert label.get('reviewed') is True and label['coordinate_system']=='full_image_xy'
    assert sha(sample['annotation_path'])==sample['annotation_sha256']
    quality=load_label_quality(sample)
    assert quality['quality_sha256']==sample.get('quality_sha256')
    polylines=quality.get('supervision_polylines',label['polylines'])
    exclusions=label.get('ignore_polygons',[])+quality['precision_exclude_polygons']
    clipped={k:clip_scoring_segments([polylines.get(k,[])],ignore_polygons=exclusions)
             for k in ['centerline','edge_flange','edge_surface']}
    target,distance=response_from_lines(clipped['centerline'],shape,bbox,scale,PARAMETERS['center_sigma_original_px'])
    # Both reviewed sides constrain the lateral supervision region. This region
    # supplies regression validity only; it is not exported as a body truth.
    a=polylines.get('edge_flange',[]);b=polylines.get('edge_surface',[])
    strip=polygon_mask([a+list(reversed(b))] if a and b else [],shape,bbox,scale)
    sides=line_mask(clipped['edge_flange']+clipped['edge_surface'],shape,bbox,scale)
    side_distance=cv2.distanceTransform(1-sides,cv2.DIST_L2,cv2.DIST_MASK_PRECISE) if sides.any() else np.full(shape,np.inf)
    tube=distance<=PARAMETERS['center_support_sigmas']*PARAMETERS['center_sigma_original_px']*scale
    inside=strip.astype(bool)&tube
    outside=(~strip.astype(bool))&(side_distance<=PARAMETERS['outside_negative_width_original_px']*scale)&(distance>=PARAMETERS['center_sigma_original_px']*3*scale)
    valid=inside|outside
    # Never create endpoint negatives or blur uncertain continuation into target.
    endpoints=np.zeros(shape,np.uint8)
    for s in clipped['centerline']:
        for p in [s[0],s[-1]]:
            cv2.circle(endpoints,tuple(np.rint(transform([p],bbox,scale)[0]).astype(int)),int(np.ceil(12*scale)),1,-1)
    outside &= ~endpoints.astype(bool);valid=inside|outside
    ignore=polygon_mask(exclusions,shape,bbox,scale)
    guard=int(np.ceil(PARAMETERS['quality_guard_original_px']*scale))
    if ignore.any():ignore=cv2.dilate(ignore,np.ones((2*guard+1,2*guard+1),np.uint8))
    valid &= ~ignore.astype(bool)
    target[~valid]=0
    return target,valid,{'annotation_sha256':sample['annotation_sha256'],
        'quality_sha256':quality['quality_sha256'],'reviewed':True,'source_status':label.get('status'),
        'trusted_center_segments_xy':clipped['centerline'],
        'trusted_rope_sides_xy':{k:clipped[k] for k in ['edge_flange','edge_surface']},
        'ignored_region_count':len(exclusions),'quality_clip_before_rasterization':True,
        'negative_pixels_outside_reviewed_sides':int((outside&valid).sum()),
        'positive_is_center_heatmap_not_body_truth':True}

def prepare(run):
    started=time.time();run=Path(run).resolve();data=run/'data'
    for sub in ['cache','scenes','label_sources','checks','entry_hints','auto_entries','auto_scenes']:(data/sub).mkdir(parents=True,exist_ok=True)
    manifest=load(MANIFEST);samples=manifest['samples']
    assert len(samples)==18 and sum(s['split']=='train' for s in samples)==10
    assert all('20260925' not in json.dumps(s) for s in samples)
    groups={k:{s['group_id'] for s in samples if s['split']==v} for k,v in [('train','train'),('eval','holdout')]}
    assert not groups['train']&groups['eval']
    old_auto_inputs={s['sample_id']:s for s in load(V28/'input_manifest.json')}
    items=[];counts=[]
    for sample in samples:
        sid=sample['sample_id'];split='train' if sample['split']=='train' else 'eval'
        assert sha(sample['image_path'])==sample['image_sha256']
        image=cv2.imdecode(np.fromfile(sample['image_path'],np.uint8),cv2.IMREAD_COLOR)
        h,w=image.shape[:2];assert [h,w]==sample['image_shape_hw']
        # The only evaluation annotation content parsed is this dedicated hint.
        seed_full=extract_seed_only(sample['annotation_path']);seed=short_seed(seed_full,120.)
        hint_path=data/'entry_hints'/f'{sid}.json'
        dump(hint_path,{'id':sid,'free_seed_entry_only_xy':seed.tolist(),
            'mode':'CONDITIONAL_KNOWN_ENTRY','source_annotation_sha256':sample['annotation_sha256'],
            'evaluation_target_geometry_parsed':False if split=='eval' else None,
            'tail_of_original_seed_discarded':True})
        top=int(np.floor(.35*h));top-=top%2;bbox=[0,top,w,h];scale=.5
        assert seed[:,1].min()>=top,'Fixed search ROI would delete entry'
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
        cropped=cv2.resize(gray[top:h,:],(w//2,(h-top)//2),interpolation=cv2.INTER_AREA)
        shape=cropped.shape;roi=np.ones(shape,bool)
        seed_map,_=response_from_lines([seed],shape,bbox,scale,PARAMETERS['seed_sigma_original_px'])
        scene=build_scene(image,seed);scene_path=data/'scenes'/f'{sid}.json';dump(scene_path,scene)
        auto_path=None
        if split=='eval':
            assert old_auto_inputs[sid]['image_sha256']==sample['image_sha256']
            old_path=V28/'AUTO'/sid/'entry.json';old=load(old_path)
            auto=dict(available=bool(old['available']),free_seed=[],width_px=old.get('width_px'),
              reason=old.get('reason'),mode='AUTO_FROZEN_V28_ENTRY',source_path=str(old_path),
              source_sha256=sha(old_path),image_sha256=sample['image_sha256'],
              original_input_manifest_sha256=sha(V28/'input_manifest.json'),scene_path=None)
            if auto['available']:
                auto_seed=short_seed(old['free_seed'],120.)
                auto['free_seed']=auto_seed.tolist()
                auto_scene=build_scene(image,auto_seed,arc=old.get('arc'))
                ap=data/'auto_scenes'/f'{sid}.json';dump(ap,auto_scene)
                auto['scene_path']=str(ap)
            auto_path=data/'auto_entries'/f'{sid}.json';dump(auto_path,auto)
        ys,xs=np.indices(shape);xy=np.stack([xs/scale+bbox[0],ys/scale+bbox[1]],-1)
        stage=stage_at_points(scene,xy)
        if split=='train':target,valid,source=training_target(sample,shape,bbox,scale)
        else:target=np.zeros(shape,np.float32);valid=np.zeros(shape,bool);source={'evaluation_target_geometry_read':False,'target_and_valid_are_all_zero':True}
        dump(data/'label_sources'/f'{sid}.json',source)
        cache=data/'cache'/f'{sid}.npz'
        np.savez_compressed(cache,image=cropped.astype(np.float32),seed=seed_map,roi=roi,target=target,valid=valid,stage=stage)
        item=dict(id=sid,split=split,source_group=sample['group_id'],cache=str(cache),bbox=bbox,shape=[h,w],scale=scale,
          image_path=sample['image_path'],image_sha256=sample['image_sha256'],scene=str(scene_path),
          scene_path=str(scene_path),auto_entry_path=str(auto_path) if auto_path else None,
          entry_hint=str(hint_path),entry_mode='CONDITIONAL_KNOWN_ENTRY',seed_xy=seed.tolist(),
          known_seed_xy=seed.tolist(),
          source_metadata=sample['source'],cache_sha256=sha(cache))
        items.append(item)
        per_stage={str(k):{'valid_pixels':int((valid&(stage==k)).sum()),'high_pixels':int((valid&(stage==k)&(target>=.5)).sum()),
                         'low_pixels':int((valid&(stage==k)&(target<.5)).sum())} for k in range(1,5)}
        counts.append({'id':sid,'split':split,'shape':list(shape),'stages':per_stage,'valid_pixels':int(valid.sum()),
          'input_minmax':[float(cropped.min()),float(cropped.max())],'input_mean_std':[float(cropped.mean()),float(cropped.std())],
          'entry_preserved':True,'contact_proxy_status':scene['contact_proxy']['status'],'observed_distal_edge_points':len(scene['target_edge_points_xy'])})
        # Alignment checks use the full search ROI and a fixed 0..1 response scale.
        canvas=cv2.cvtColor(np.uint8(np.clip(cropped,0,1)*255),cv2.COLOR_GRAY2BGR)
        canvas[seed_map>.2]=(0,255,255)
        canvas[valid&(target>.5)]=(0,255,0)
        canvas[valid&(target<.05)]=(255,100,0)
        cv2.imencode('.png',canvas)[1].tofile(str(data/'checks'/f'{sid}.png'))
        print(json.dumps({'prepared':sid,'split':split,'stages':per_stage,'edge_points':len(scene['target_edge_points_xy'])}),flush=True)
    output={'schema_version':1,'parameters':PARAMETERS,'manifest':str(MANIFEST),'manifest_sha256':sha(MANIFEST),
      'items':items,'source_groups':{k:sorted(v) for k,v in groups.items()},
      'split_semantics':'V11 historical-open group-separated development evaluation; not protected blind test',
      'evaluation_label_policy':'free_seed field only; center/edge/quality geometry not parsed before prediction',
      'stage_semantics':'image/entry visual proxies, not physical phase truth','created_at_local':time.strftime('%Y-%m-%d %H:%M:%S')}
    dump(data/'index.json',output);dump(data/'supervision_statistics.json',{'items':counts,'elapsed_seconds':time.time()-started,
      'environment':{'python':sys.version,'opencv':cv2.__version__,'numpy':np.__version__,'platform':platform.platform()},
      'source_hashes':{p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('scene.py')]},'command':sys.argv,'exit_code':0})
    print(json.dumps({'completed':True,'items':len(items),'elapsed_seconds':time.time()-started}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    prepare(parser.parse_args().run)
