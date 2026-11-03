"""Frozen raw-video spatial observability probe. No event or actuator output.

The startup reference is fixed from <=5 s. Spatial measurements after 5 s never
read future images. Semantic side-edge candidates and independent fixed-normal
profiles are saved separately, without imputing missing/occluded measurements.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import datetime
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from vision_prototype.semantic_locator import semantic_locator_for_camera
from vision_geometry_lab.v18_reference_pose import FixedBackgroundPose, DEFAULT_PARAMETERS
from vision_geometry_lab.v18_front_observability import build_fixed_baseline, sample_normal_edges

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / 'vision_geometry_lab'
BASE = LAB / 'outputs/v18_front_observability'
PREVIOUS = LAB / 'outputs/v16_event_feasibility/protocol.json'
RULES = dict(startup_ready_s=5., end_s=60., stride=1,
    rows_start_height_fraction=1900/3000, rows_stop_height_fraction=2920/3000,
    rows_step_height_fraction=8/3000, seed_search_half_width_px=65,
    seed_weight_sigma_px=28., minimum_abs_gradient=8., maximum_regrid_row_gap_px=16.,
    peak_ambiguity_diagnostic_ratio=1.2,
    baseline_policy='fixed_startup_not_a_claim_of_old_layer',
    coordinates='full_resolution_canonical_reference_pixels_not_mm',
    experiment='observability_only_not_event_detection',
    hardware_access=False, ground_truth_input=False, deployment_allowed=False)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def pixels_sha(frame):
    h=hashlib.sha256(str((frame.shape,str(frame.dtype))).encode('ascii'))
    h.update(np.ascontiguousarray(frame).tobytes()); return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def validate_video(meta):
    for key in ('fps','frame_count','width','height'):
        value=meta[key]
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError('invalid video '+key)
    if any(int(meta[k])!=meta[k] for k in ('frame_count','width','height')):
        raise ValueError('noninteger image/video shape')


def canonical_frame(raw,camera):
    if camera not in ('left','right'):raise ValueError('unknown camera')
    return cv2.flip(raw,1) if camera=='left' else raw


def semantic_candidates(gray,regions,rows):
    """Local side-edge candidates with selected-peak check; NOT material identity."""
    mask=regions.mask('wound_rope_surface').astype(np.uint8)
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    if count>1:mask=(labels==1+np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    else:mask=np.zeros_like(mask,dtype=bool)
    x1,y1,x2,y2=regions.crop_xyxy;lh,lw=mask.shape
    fields={k:np.full(len(rows),np.nan) for k in ('x','seed','gradient','contrast')}
    for i,y in enumerate(rows):
        low_y=int(round((y-y1)*lh/(y2-y1)))
        if not 0<=low_y<lh:continue
        occupied=np.flatnonzero(mask[low_y])
        if not len(occupied):continue
        seed=x1+float(occupied[0])*(x2-x1)/lw;fields['seed'][i]=seed
        xa=max(2,int(round(seed-65)));xb=min(gray.shape[1]-2,int(round(seed+66)))
        if xb-xa<8:continue
        profile=gray[max(0,int(y)-3):min(gray.shape[0],int(y)+4),xa-1:xb+1].mean(axis=0)
        gradient=np.gradient(profile)[1:-1]
        if len(gradient)<3 or np.max(abs(gradient))<8.:continue
        xs=np.arange(xa,xb);score=abs(gradient)*np.exp(-.5*((xs-seed)/28.)**2)
        k=int(np.argmax(score));x=xa+k
        if abs(gradient[k])<8.:continue
        fields['x'][i]=x;fields['gradient'][i]=gradient[k]
        fields['contrast'][i]=profile[min(len(profile)-1,k+6)]-profile[max(0,k-4)]
    return fields


def align_x_to_reference(xs,ys,matrix,grid,max_gap=16.):
    """Current canonical -> reference, interpolation only adjacent valid rows."""
    out=np.full(len(grid),np.nan)
    matrix=np.asarray(matrix,float)
    if matrix.shape!=(2,3) or not np.isfinite(matrix).all():return out
    inv=cv2.invertAffineTransform(matrix)
    pts=np.column_stack([xs,ys])@inv[:,:2].T+inv[:,2]
    for a,b in zip(pts[:-1],pts[1:]):
        if not np.isfinite([a,b]).all() or not 0<b[1]-a[1]<=max_gap:continue
        use=(grid>=a[1])&(grid<=b[1])
        out[use]=a[0]+(grid[use]-a[1])*(b[0]-a[0])/(b[1]-a[1])
    return out


def supported_median(matrix,minimum=20):
    counts=np.isfinite(matrix).sum(axis=0);out=np.full(matrix.shape[1],np.nan)
    good=counts>=minimum
    if good.any():out[good]=np.nanmedian(matrix[:,good],axis=0)
    return out


def freeze(base=BASE):
    base=Path(base)
    if base.exists():raise FileExistsError('fresh trial directory required')
    old=read(PREVIOUS);videos={};models={}
    for side in ('left','right'):
        video=dict(old['videos'][side]);video['camera']=side
        validate_video(video)
        if sha(video['path'])!=video['sha256']:raise ValueError('source video changed')
        locator=semantic_locator_for_camera(side)
        models[side]={str(p):sha(p) for p in (locator.model_path,locator.metadata_path)}
        video['indices']=[i for i in range(video['frame_count']) if i/video['fps']<=RULES['end_s']]
        video.pop('selected_indices',None);videos[side]=video
    sources=[Path(__file__),LAB/'v18_reference_pose.py',LAB/'v18_front_observability.py',
        ROOT/'vision_prototype/semantic_locator.py']
    write(base/'protocol.json',dict(schema='v18_observability_protocol_1',
        frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),rules=RULES,
        scene_parameters=asdict(DEFAULT_PARAMETERS),videos=videos,models=models,
        source_hashes={str(p):sha(p) for p in sources},
        previous_protocol=dict(path=str(PREVIOUS),sha256=sha(PREVIOUS)),
        stop_rule='No event implementation if no raw-image-specific persistent advancing front; do not tune to manual event times'))


def verify(base=BASE,video_hashes=True):
    p=read(Path(base)/'protocol.json')
    if p['rules']!=RULES:raise ValueError('rules changed')
    for path,digest in p['source_hashes'].items():
        if sha(path)!=digest:raise ValueError('frozen source changed '+path)
    for files in p['models'].values():
        for path,digest in files.items():
            if sha(path)!=digest:raise ValueError('model changed')
    if sha(p['previous_protocol']['path'])!=p['previous_protocol']['sha256']:raise ValueError('input protocol changed')
    for v in p['videos'].values():
        validate_video(v)
        if video_hashes and sha(v['path'])!=v['sha256']:raise ValueError('video changed')
    return p


def run(base=BASE,session='left'):
    base=Path(base);p=verify(base);video=p['videos'][session];dest=base/session
    if dest.exists():raise FileExistsError('run exists, do not overwrite')
    dest.mkdir();write(dest/'started.json',{'status':'started_not_proof_of_completion'})
    locator=semantic_locator_for_camera(session);cv2.setNumThreads(2)
    h,w=video['height'],video['width'];fps=video['fps'];indices=video['indices']
    rows=np.arange(round(h*RULES['rows_start_height_fraction']),
        round(h*RULES['rows_stop_height_fraction'])+1,max(1,round(h*RULES['rows_step_height_fraction'])))
    shape=(len(indices),len(rows))
    aligned=np.full(shape,np.nan);transforms=np.full((len(indices),2,3),np.nan)
    semantic={k:np.full(shape,np.nan) for k in ('x','seed','gradient','contrast')}
    records=[];cap=cv2.VideoCapture(video['path']);pose=None;started=time.perf_counter()
    if not cap.isOpened():raise ValueError('cannot decode video')
    try:
        for i in indices:
            before=time.perf_counter();ok,raw=cap.read()
            if not ok or int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))-1!=i:raise ValueError('decode order mismatch')
            if raw.shape[:2]!=(h,w):raise ValueError('decoded shape mismatch')
            frame=canonical_frame(raw,session)
            if pose is None:pose=FixedBackgroundPose(frame)
            measured=pose.measure(frame)
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            values=semantic_candidates(gray,locator.predict(frame),rows)
            for k in semantic:semantic[k][i]=values[k]
            if measured['valid']:
                transforms[i]=measured['reference_to_current']
                aligned[i]=align_x_to_reference(values['x'],rows,transforms[i],rows)
            records.append(dict(frame_index=i,time_s=i/fps,frame_sha256=pixels_sha(raw),pose=measured,
                first_pass_ms=(time.perf_counter()-before)*1000))
            if i%100==0:print(session,'side edge',i,'elapsed',round(time.perf_counter()-started,1),flush=True)
    finally:cap.release()
    times=np.asarray(indices,float)/fps
    baseline=build_fixed_baseline(times,rows,aligned,ready_s=RULES['startup_ready_s'])
    normal={k:np.full(shape,np.nan) for k in ('offset','gradient_signed','peak_ratio','contrast','clipped_fraction')}
    profiles=np.full((*shape,131),np.nan,dtype=np.float32)
    cap=cv2.VideoCapture(video['path'])
    try:
        for i in indices:
            before=time.perf_counter();ok,raw=cap.read()
            if not ok or int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))-1!=i:raise ValueError('second pass decode mismatch')
            if pixels_sha(raw)!=records[i]['frame_sha256']:raise ValueError('frame pixels changed')
            if records[i]['pose']['valid']:
                gray=cv2.cvtColor(canonical_frame(raw,session),cv2.COLOR_BGR2GRAY)
                result=sample_normal_edges(gray,transforms[i],rows,baseline)
                for k in normal:normal[k][i]=result[k]
                profiles[i]=result['profile']
            records[i]['second_pass_ms']=(time.perf_counter()-before)*1000
            if i%100==0:print(session,'normal profiles',i,'elapsed',round(time.perf_counter()-started,1),flush=True)
    finally:cap.release()
    warm=times<=RULES['startup_ready_s']
    base_offset=supported_median(normal['offset'][warm])
    residual=normal['offset']-base_offset
    residual[warm]=np.nan
    verify(base)
    np.savez_compressed(dest/'measurement.npz',times_s=times,rows_y=rows,
        aligned_semantic_x=aligned,semantic_x_current=semantic['x'],semantic_seed_x_current=semantic['seed'],
        semantic_gradient=semantic['gradient'],semantic_contrast=semantic['contrast'],
        pose_valid=np.asarray([r['pose']['valid'] for r in records]),transforms=transforms,
        baseline_x=baseline['x_ref'],baseline_nx=baseline['nx'],baseline_ny=baseline['ny'],
        baseline_mad_px=baseline['baseline_mad_px'],
        normal_offset=normal['offset'],normal_gradient=normal['gradient_signed'],
        normal_peak_ratio=normal['peak_ratio'],normal_contrast=normal['contrast'],
        normal_clipped_fraction=normal['clipped_fraction'],normal_profiles=profiles,
        baseline_normal_offset=base_offset,normal_residual=residual)
    write(dest/'index.json',dict(schema='v18_spatial_measurements_1',video=video,records=records,
        protocol_sha256=sha(base/'protocol.json'),measurement_sha256=sha(dest/'measurement.npz'),
        rules=RULES,baseline_support_rows=int(np.isfinite(base_offset).sum()),
        frame_digest_format='sha256(repr((shape,str(dtype))) + contiguous_Bgr_bytes)',
        known_occlusion_not_classified=True,material_identity_verified=False,
        output_status='complete_observations_only_no_events',runtime_s=time.perf_counter()-started))
    # Keep the startup marker as audit history; only complete index authorizes use.
    print('COMPLETE',session,'rows',len(rows),'baseline supported',np.isfinite(base_offset).sum(),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','verify','run'))
    parser.add_argument('--base',type=Path,default=BASE)
    parser.add_argument('--session',choices=('left','right'),default='left')
    args=parser.parse_args()
    if args.stage=='freeze':freeze(args.base)
    elif args.stage=='verify':verify(args.base);print('V18 frozen inputs and sources verified')
    else:run(args.base,args.session)


if __name__=='__main__':main()
