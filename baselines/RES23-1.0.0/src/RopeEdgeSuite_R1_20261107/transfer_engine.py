"""New-video I/O adapter around sealed RES-1.0 algorithm sources.

No algorithm thresholds are changed. Existing file-bound BG4 spatial logic is
compiled as a pure tail with explicitly supplied same-frame arrays. The exact
source and the removed I/O wrapper are audited by parity_tests.py.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,hashlib,types,inspect,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent;FROZEN=ROOT/'baseline/source'
LC1=FROZEN/'LC1_locked_column_20261105_v1';BG1=FROZEN/'LC1_boundary_BG1_20261106_v1'
BG3=FROZEN/'LC1_boundary_BG3_point_global_20261107_v2';BG4=FROZEN/'LC1_boundary_BG4_local_temporal_20261107_v1'
ECW=FROZEN/'LC1_ECW_20261107_v1';TRACK=FROZEN/'LC1_fixed_d2238p7_full_20261106_v1'
for p in [LC1,TRACK,BG1,BG3,BG4,ECW]:sys.path.insert(0,str(p))
from region_gate import segment
from gated_rope import core
from global_points import group_and_fit,intersections,model_envelope
import pipeline
import local_observation as local
from temporal_robust import Counter
from camera_reference import CameraContinuity
from live_s1_tracker_anchored import LiveS1Tracker
from run_full_fixed_d import Reacquire
from ecw_v3 import measure_v3
from window_detector import WindowDetector,evidence
sys.path.insert(0,str(ROOT))
D=2238.7;PITCH=23.29412648284868;SCALE=.5
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pixelsha(im):return hashlib.sha256(im.tobytes()).hexdigest()
def plain(a):
    if isinstance(a,np.ndarray):return a.tolist()
    if isinstance(a,np.generic):return a.item()
    if isinstance(a,dict):return {k:plain(v) for k,v in a.items()}
    if isinstance(a,(tuple,list)):return [plain(v) for v in a]
    return a
def dump(p,x):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(plain(x),ensure_ascii=False,allow_nan=False),encoding='utf8');tmp.replace(p)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def verify_baseline():
    manifest=read(ROOT/'baseline/manifest.json')
    for r in manifest['files']:assert sha(ROOT/'baseline'/r['path'])==r['sha256'],r['path']
    return sha(ROOT/'baseline/manifest.json')

def extract_current(image,mask,track):
    """Same computations and coordinate grid as frozen BG3 inputs.extract."""
    exclusion=np.zeros_like(mask);exclusion[mask==0]=255
    pts=track.get('line_points_xy',[])
    if len(pts)>1:cv2.polylines(exclusion,[np.rint(pts).astype(np.int32)],False,255,round(.8*PITCH))
    im=cv2.resize(image,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
    mm=((cv2.resize(mask,(im.shape[1],im.shape[0]),interpolation=cv2.INTER_AREA)==255)*255).astype(np.uint8)
    ee=((cv2.resize(exclusion,(im.shape[1],im.shape[0]),interpolation=cv2.INTER_AREA)>0)*255).astype(np.uint8)
    p=PITCH*SCALE;norm,_=core._module('normalization')._normalize_surface(im,mm,p)
    shadow=core._module('shadow').build_shadow_field(im,norm,mm,p,None);shadow['excluded_mask']=ee
    guard=core._module('period').build_period_guard(im,norm,mm,p,None,shadow)
    observed=core._module('tracks').detect_and_group(shadow,guard,mm,p)
    track_by_point={pt['point_id']:k for k,t in enumerate(observed['tracks']) for pt in t['points']}
    points=np.asarray([[pt['point_id'],pt['x'],pt['y'],pt['score'],pt['width'],pt['slope'],pt['pitch'],track_by_point[pt['point_id']]]
                       for pt in observed['points']],np.float64).reshape(-1,8)
    x0,y0=shadow['x0'],shadow['y0'];hh,ww=shadow['body_score'].shape;ee[y0:y0+hh,x0:x0+ww]|=guard['unknown_mask']
    return dict(points=points,surface=mm,exclusion=ee,broad=shadow['broad_score'],clean=shadow['cleaned_shadow'],body=guard['body_barrier'],
                x0=np.asarray(x0),y0=np.asarray(y0),pitch=np.asarray(p),source_pixel_sha256=np.asarray(pixelsha(image)),
                raw_evidence_cache_origin=np.asarray('fresh_current_frame_no_historical_cache'))

# The frozen numeric tail runs verbatim, including rejection rules and scaling.
_original=inspect.getsource(local.analyze)
_start=_original.index("    data['model_envelope']=")
_tail=_original[_start:]
_namespace=dict(local.__dict__)
exec('def spatial_tail(old,data,r):\n    reg=old["region"];s1=old["s1"];edge=reg.get("edge_y");active=old["intersection_y_px"]\n'+_tail,_namespace)
_spatial_tail=_namespace['spatial_tail']

def spatial_bg4(old,data,motion,record_hash=None):
    reg=old['region'];s1=old['s1']
    r=dict(version='RES-1.0/BG4-frozen-spatial',frame_index=old['frame_index'],nominal_time_s=old['nominal_time_s'],
        source_pixel_sha256=old['source_pixel_sha256'],baseline_record_sha256=record_hash,fixed_d_original_px=D,
        intersection_y_px=old['intersection_y_px'],region_gate_passed=old['region_gate_passed'],
        region={k:reg.get(k) for k in ('valid','reasons','edge_y','edge_uncertainty_px','top_curve_xy','top_observed_xy')},
        s1={k:s1.get(k) for k in ('valid','current_line_abc','line_points_xy','confidence')},
        masks=old['masks'],camera_motion=motion,reasons=list(old['reasons']),BG4=None,observation=None,
        baseline_count_valid=old['count_valid'],baseline_display_N=old['display_N'])
    if not old['region_gate_passed'] or old['BG3'] is None:return r
    assert str(data['source_pixel_sha256'])==old['source_pixel_sha256']
    return _spatial_tail(old,data,r)

def spatial_frame(image,track,motion,frame_index,nominal_time,source_sha,out):
    cv2.setNumThreads(1);tick=time.perf_counter();out=Path(out)
    regions=segment(image);reg=regions['record'];abc=track.get('current_line_abc') if track.get('valid') else None
    y=None if abc is None else float(-(abc[0]*D+abc[2])/abc[1])
    old=dict(region=reg,s1=track,intersection_y_px=y,fixed_d_original_px=D,frame_index=frame_index,nominal_time_s=nominal_time,
             source_pixel_sha256=pixelsha(image),source_video_sha256=source_sha,masks={},F3_called=False,
             count_valid=False,display_N=None,reasons=list(reg['reasons']))
    data=None
    if reg['valid']:
        data=extract_current(image,regions['surface_mask'],track);old['F3_called']=True
    # Frozen BG3 only needs an input record binding, not persisted masks or caches.
    bg1folder=out/'bg1';bg1folder.mkdir(exist_ok=True);(bg1folder/'frames').mkdir(exist_ok=True)
    oldpath=bg1folder/'frames'/f'{frame_index:06d}.json';dump(oldpath,old)
    pipeline.BASE=bg1folder
    b3=pipeline.analyze(image,old,data)
    r=spatial_bg4(b3,data,motion)
    # Record exact point data for displayed curves and later audits, with no
    # wholesale image-field caches needed for a finished transfer run.
    r['fresh_F3']=dict(called=old['F3_called'],point_count=0 if data is None else len(data['points']),
                        raw_points=[] if data is None else data['points'],BG3_curve_count=0 if b3['BG3'] is None else len(b3['BG3']['curves']))
    r['S1_tracker_full']=track
    r['ECW']=measure_v3(image,frame_index,record=r)
    r['spatial_elapsed_s']=time.perf_counter()-tick
    return plain(r)

class Frontend:
    def __init__(self):
        mask=cv2.imdecode(np.fromfile(ROOT/'baseline/calibration/left_search_mask.png',np.uint8),0)
        self.reacquire=Reacquire(mask,PITCH)
        assert not self.reacquire.paths,'Transfer must not read reference-video seed caches'
        self.tracker=LiveS1Tracker(mask,reacquire_callback=self.reacquire)
        self.camera=CameraContinuity()
    def step(self,image,t,i):return self.tracker.step(image,t,i),self.camera.step(image)
