"""Same-frame, pixel-bound inputs from supplied 01; no old fitted curves used."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parent
BG1=ROOT.parent/'LC1_boundary_BG1_20261106_v1'
BASE=BG1/'full_002'
sys.path.insert(0,str(BG1))
from gated_rope import core,SurfaceObserver,PITCH
from inspect_regions import SOURCE
SCALE=.5
LEGACY_CACHE=ROOT.parent/'LC1_boundary_BG3_20261107_v1'/'point_cache'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x
def dump(p,x):Path(p).write_text(json.dumps(plain(x),ensure_ascii=False,allow_nan=False),encoding='utf8')

def extract(image,old):
    assert hashlib.sha256(image.tobytes()).hexdigest()==old['source_pixel_sha256']
    if not old['region']['valid']:return None
    mp=BASE/old['masks']['surface']['path'];assert sha(mp)==old['masks']['surface']['sha256']
    cached=LEGACY_CACHE/f"{old['frame_index']:06d}.npz"
    if cached.exists():
        # Reuse raw extractor fields only after source/mask/dependency checks.
        # Neither the old BG3 memberships nor fitted models are reused.
        with np.load(cached,allow_pickle=False) as f:
            assert str(f['pixel_sha'])==old['source_pixel_sha256']
            assert str(f['mask_sha'])==sha(mp)
            dep=hashlib.sha256(''.join(sha(p) for p in core.source_dependencies()).encode()).hexdigest()
            assert str(f['dependency_sha'])==dep
            pp=f['points'];members=f['members']
            assert np.array_equal(pp[:,:2]*2,np.asarray(old['F3']['raw_observed_points_xy']))
            points=np.c_[np.arange(len(pp)),pp[:,:6],members]
            return dict(points=points,surface=f['surface'],exclusion=f['exclusion'],
                broad=f['broad_score'],clean=f['cleaned_shadow'],body=f['body_barrier'],
                x0=f['x0'],y0=f['y0'],pitch=np.asarray(PITCH*SCALE),
                source_pixel_sha256=np.asarray(old['source_pixel_sha256']),
                old_record_sha256=np.asarray(sha(BASE/'frames'/f"{old['frame_index']:06d}.json")),
                extraction_code_sha256=np.asarray(sha(__file__)),
                raw_evidence_cache_origin=np.asarray(str(cached)),
                raw_evidence_cache_sha256=np.asarray(sha(cached)))
    mask=cv2.imdecode(np.fromfile(mp,np.uint8),0)
    exclusion=np.zeros_like(mask);exclusion[mask==0]=255
    pts=old['s1'].get('line_points_xy',[])
    if len(pts)>1:cv2.polylines(exclusion,[np.rint(pts).astype(np.int32)],False,255,round(.8*PITCH))
    im=cv2.resize(image,None,fx=SCALE,fy=SCALE,interpolation=cv2.INTER_AREA)
    mm=((cv2.resize(mask,(im.shape[1],im.shape[0]),interpolation=cv2.INTER_AREA)==255)*255).astype(np.uint8)
    ee=((cv2.resize(exclusion,(im.shape[1],im.shape[0]),interpolation=cv2.INTER_AREA)>0)*255).astype(np.uint8)
    p=PITCH*SCALE
    norm,_=core._module('normalization')._normalize_surface(im,mm,p)
    shadow=core._module('shadow').build_shadow_field(im,norm,mm,p,None);shadow['excluded_mask']=ee
    guard=core._module('period').build_period_guard(im,norm,mm,p,None,shadow)
    observed=core._module('tracks').detect_and_group(shadow,guard,mm,p)
    # Local tracks only estimate shared shape. Their IDs are NOT final classes,
    # and neither old D4 nor BG2 fitted curve fragments enter point grouping.
    track_by_point={pt['point_id']:k for k,t in enumerate(observed['tracks']) for pt in t['points']}
    points=np.asarray([[pt['point_id'],pt['x'],pt['y'],pt['score'],pt['width'],pt['slope'],
                        pt['pitch'],track_by_point[pt['point_id']]] for pt in observed['points']],np.float64).reshape(-1,8)
    assert np.array_equal(points[:,1:3]*2,np.asarray(old['F3']['raw_observed_points_xy']))
    x0,y0=shadow['x0'],shadow['y0'];hh,ww=shadow['body_score'].shape
    ee[y0:y0+hh,x0:x0+ww]|=guard['unknown_mask']
    return dict(points=points,surface=mm,exclusion=ee,broad=shadow['broad_score'],
        clean=shadow['cleaned_shadow'],body=guard['body_barrier'],x0=np.asarray(x0),y0=np.asarray(y0),
        pitch=np.asarray(p),source_pixel_sha256=np.asarray(old['source_pixel_sha256']),
        old_record_sha256=np.asarray(sha(BASE/'frames'/f"{old['frame_index']:06d}.json")),
        extraction_code_sha256=np.asarray(sha(__file__)),raw_evidence_cache_origin=np.asarray('fresh_extraction'))

def load_or_extract(i,image,old,cache):
    path=Path(cache)/f'{i:06d}.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as f:data={k:f[k] for k in f.files}
        assert str(data['source_pixel_sha256'])==hashlib.sha256(image.tobytes()).hexdigest()
        assert str(data['old_record_sha256'])==sha(BASE/'frames'/f'{i:06d}.json')
        if str(data['extraction_code_sha256'])==sha(__file__):return data
        # A previous local cache is kept as evidence of the earlier extractor.
        prior=path.with_name(path.stem+'.prior_'+str(data['extraction_code_sha256'])[:12]+'.npz')
        if prior.exists():assert sha(prior)==sha(path)
        else:path.rename(prior)
    data=extract(image,old)
    if data is not None:
        path.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(path,**data)
    return data
