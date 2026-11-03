"""Twenty fixed development images: automatic free-line predictions then GT scoring.

No human seed, ROI, SAM mask, target curve or event time enters detect_rope.
The five-pixel/one-degree result concerns the annotated free segment only, not
identity of the same rope after winding or a calibrated correctness probability.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from vision_geometry_lab.rope_detector import detect_rope

LAB=Path(__file__).resolve().parent
BASE=LAB/'outputs/v23_free_rope_audit'
INVENTORY=LAB/'outputs/centerline_v9/dataset/inventory.json'
RULES=dict(angular_tolerance_deg=1.,normal_error_tolerance_original_px=5.,
    normal_error_statistic='95th percentile over 1-original-pixel arclength sampled human free_seed',
    GT_direction='orthogonal least-squares principal direction of sampled free_seed',
    angle='undirected angle in 0..90 degrees',
    line_fit_pass='detector valid AND angle<=1deg AND GT-to-infinite-line P95<=5px',
    overlap='independent intersection of projected observed segment and GT free_seed extent along GT principal axis; not a pass condition',
    prediction_inputs='full image only; no ROI/seed/SAM/arc/current annotation',
    labels_opened_by_driver='only after all predictions persisted',
    physical_rope_identity_confirmed=False, wound_path_tested=False,
    development_data=True, threshold_search=False, hardware_access=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def clean(value):
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,np.generic): return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(clean(value),f,ensure_ascii=False,indent=2,allow_nan=False)


def sample_polyline(points,step=1.):
    p=np.asarray(points,float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<2 or not np.isfinite(p).all() or step<=0:
        raise ValueError('finite nonempty free-seed polyline required')
    lengths=np.linalg.norm(np.diff(p,axis=0),axis=1)
    if lengths.sum()<=1e-9: raise ValueError('zero-length free seed')
    cumulative=np.r_[0.,np.cumsum(lengths)]
    keep=np.r_[True,np.diff(cumulative)>1e-9]
    distances=np.arange(0.,cumulative[-1],step)
    distances=np.r_[distances,cumulative[-1]]
    samples=np.column_stack([np.interp(distances,cumulative[keep],p[keep,k]) for k in range(2)])
    return samples,float(cumulative[-1])


def score(prediction,free_seed,status):
    available=bool(prediction.get('valid'))
    if status=='absent':
        return dict(reference_kind='absent',detector_available=available,
            false_positive_candidate=available,line_fit_pass=None,
            note='One absent development example, not a calibrated absent-class false-positive rate.')
    try:
        points,length=sample_polyline(free_seed)
    except ValueError as exc:
        return dict(reference_kind='unscorable_seed',detector_available=available,line_fit_pass=None,reason=str(exc))
    center=points.mean(axis=0)
    _,_,vectors=np.linalg.svd(points-center,full_matrices=False)
    direction=vectors[0];ref_normal=np.array([-direction[1],direction[0]])
    output=dict(reference_kind='visible_human_free_seed',detector_available=available,
        line_fit_pass=False,reference_length_original_px=length,
        reference_sample_count=len(points),reference_TLS_normal_rms_px=float(np.sqrt(np.mean(((points-center)@ref_normal)**2))),
        angular_error_deg=None,normal_error_px=None,observed_projection_overlap_fraction=None,
        reference_direction_xy=direction.tolist(),physical_rope_identity_confirmed=False)
    if not available:
        return {**output,'reason':prediction.get('reason','detector_unavailable')}
    n=np.asarray(prediction.get('normal'),float)
    segment=np.asarray(prediction.get('observed_segment'),float)
    rho=prediction.get('rho_px')
    if n.shape!=(2,) or not np.isfinite(n).all() or np.linalg.norm(n)<1e-9 or rho is None or not np.isfinite(rho) or segment.shape!=(2,2) or not np.isfinite(segment).all():
        return {**output,'reason':'invalid_prediction_geometry'}
    norm=np.linalg.norm(n);n=n/norm;rho=float(rho)/norm
    predicted_direction=np.array([-n[1],n[0]])
    angle=float(np.degrees(np.arccos(np.clip(abs(direction@predicted_direction),0.,1.))))
    normal_error=np.abs(points@n-rho)
    low,high=np.min(points@direction),np.max(points@direction)
    plow,phigh=sorted(segment@direction)
    overlap=max(0.,min(high,phigh)-max(low,plow))
    errors=dict(p50=float(np.median(normal_error)),p95=float(np.quantile(normal_error,.95)),
        maximum=float(np.max(normal_error)),rms=float(np.sqrt(np.mean(normal_error**2))))
    passed=angle<=RULES['angular_tolerance_deg'] and errors['p95']<=RULES['normal_error_tolerance_original_px']
    return {**output,'reason':'scored_against_human_free_segment_only','angular_error_deg':angle,
        'normal_error_px':errors,'line_fit_pass':bool(passed),
        'observed_projection_overlap_fraction':float(overlap/(high-low)),
        'observed_projection_overlap_original_px':float(overlap),
        'GT_projected_extent_original_px':float(high-low),
        'observed_projected_extent_original_px':float(phigh-plow),
        'observed_segment_overlaps_reference':bool(overlap>0.),
        'line_fit_pass_without_observed_overlap':bool(passed and overlap<=0.)}


def read_image(path):
    image=cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_COLOR)
    if image is None: raise ValueError('image cannot be decoded: '+str(path))
    return image


def overlay(image,prediction,seed,label,metrics,path):
    # Original image unchanged; this downscaled copy is explicitly an evaluation overlay.
    scale=min(1.,1200./image.shape[1]);view=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    def poly(points,color):
        p=np.rint(np.asarray(points)*scale).astype(np.int32)
        if len(p)>=2: cv2.polylines(view,[p],False,color,2,cv2.LINE_AA)
    if len(seed)>=2: poly(seed,(255,255,0))
    if prediction.get('valid') and prediction.get('observed_segment') is not None: poly(prediction['observed_segment'],(0,140,255))
    canvas=np.zeros((view.shape[0]+66,view.shape[1],3),np.uint8);canvas[66:]=view
    cv2.putText(canvas,label,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
    text='Cyan=human free seed (SCORING ONLY); Orange=automatic OBSERVED support'
    cv2.putText(canvas,text,(8,41),cv2.FONT_HERSHEY_SIMPLEX,.46,(220,220,220),1,cv2.LINE_AA)
    text=f"valid={prediction.get('valid')} free-line-fit={metrics.get('line_fit_pass')} (NOT wound-rope identity)"
    cv2.putText(canvas,text,(8,60),cv2.FONT_HERSHEY_SIMPLEX,.46,(220,220,220),1,cv2.LINE_AA)
    ok,data=cv2.imencode('.jpg',canvas,[cv2.IMWRITE_JPEG_QUALITY,90])
    if not ok: raise ValueError('overlay encoding failed')
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('xb') as f: f.write(data.tobytes())


def summarize(rows):
    positive=[r for r in rows if r['metrics']['reference_kind']=='visible_human_free_seed']
    detected=[r for r in positive if r['metrics']['detector_available']]
    measured=[r for r in positive if r['metrics']['angular_error_deg'] is not None]
    def q(values):
        return dict(p50=float(np.median(values)),p90=float(np.quantile(values,.9)),maximum=float(np.max(values))) if values else None
    return dict(images=len(rows),positive_reference_images=len(positive),
        available_positive_candidates=len(detected),free_line_fit_pass_images=sum(r['metrics']['line_fit_pass'] for r in positive),
        denominator_for_pass=len(positive),unavailable_positive_images=len(positive)-len(detected),
        absent_images=sum(r['metrics']['reference_kind']=='absent' for r in rows),
        absent_false_positive_candidates=sum(r['metrics'].get('false_positive_candidate',False) for r in rows),
        angle_error_deg_on_scored_candidates=q([r['metrics']['angular_error_deg'] for r in measured]),
        normal_P95_px_on_scored_candidates=q([r['metrics']['normal_error_px']['p95'] for r in measured]),
        observed_projection_overlap_fraction_on_scored_candidates=q([r['metrics']['observed_projection_overlap_fraction'] for r in measured]),
        detector_ms=q([r['prediction_ms'] for r in rows]),
        predictions_are_same_rope_identity_confirmed=False)


def run(output=BASE):
    output=Path(output)
    if output.exists(): raise FileExistsError('new V23 output directory required')
    inventory=read(INVENTORY)
    samples=inventory['samples']
    if len(samples)!=20 or len({s['sample_id'] for s in samples})!=20:
        raise ValueError('expected unchanged 20-image inventory')
    source_paths=[Path(__file__),LAB/'rope_detector.py']
    protocol=dict(version=23,task='automatic_free_rope_independent_development_audit',rules=RULES,
        written_before_predictions=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        inventory_path=str(INVENTORY),inventory_sha256=sha(INVENTORY),
        source_sha256={str(p):sha(p) for p in source_paths},
        data=[{k:s[k] for k in ['sample_id','image_path','image_sha256','annotation_path','annotation_sha256','split','group_id','intended_use']} for s in samples],
        original_metadata_only_used_for_reporting=True,source_ROI_not_used=True,
        thresholds_chosen_before_inference=True,frames_are_existing_development_data=True,
        python_libraries=dict(opencv=cv2.__version__,numpy=np.__version__),cpu_threads=2)
    write(output/'protocol.json',protocol)
    cv2.setNumThreads(2)
    predictions=[]
    # All twenty detector calls complete before any annotation file is opened.
    for sample in samples:
        if sha(sample['image_path'])!=sample['image_sha256']: raise ValueError('image hash changed')
        image=read_image(sample['image_path'])
        started=time.perf_counter();prediction=detect_rope(image)
        elapsed=1000*(time.perf_counter()-started)
        predictions.append(dict(sample_id=sample['sample_id'],prediction=prediction,prediction_ms=elapsed))
        print(len(predictions),'/',len(samples),sample['sample_id'],prediction.get('valid'),flush=True)
    write(output/'predictions.json',dict(status='predictions_complete_before_label_loading',
        protocol_sha256=sha(output/'protocol.json'),rows=predictions))
    rows=[]
    for i,(sample,predicted) in enumerate(zip(samples,predictions)):
        if sha(sample['annotation_path'])!=sample['annotation_sha256']: raise ValueError('annotation hash changed')
        annotation=read(sample['annotation_path'])
        if annotation['image_sha256']!=sample['image_sha256'] or annotation['sample_id']!=sample['sample_id']:
            raise ValueError('annotation identity mismatch')
        seed=annotation['polylines']['free_seed']
        metrics=score(predicted['prediction'],seed,annotation['status'])
        overlay_path=output/'overlays'/f'{i+1:02d}_{sample["sample_id"]}.jpg'
        overlay(read_image(sample['image_path']),predicted['prediction'],seed,sample['sample_id'],metrics,overlay_path)
        rows.append({**predicted,'candidate_id':sample.get('candidate_id'),'split':sample['split'],
            'group_id':sample['group_id'],'intended_use':sample['intended_use'],
            'metrics':metrics,'overlay_path':str(overlay_path)})
    report=dict(status='complete',protocol_sha256=sha(output/'protocol.json'),
        predictions_sha256=sha(output/'predictions.json'),rows=rows,summary=summarize(rows),
        geometry_train=summarize([r for r in rows if r['intended_use']=='geometry_train']),
        geometry_holdout=summarize([r for r in rows if r['intended_use']=='geometry_holdout']),
        special_cases=[r['sample_id'] for r in rows if r['candidate_id'] in ('C03','C06')],
        interpretation='Fit to human free-line segment only; not full free-rope extent, winding trajectory identity, deployment accuracy or statistical generalization.',
        wound_path_tested=False,hardware_access=False)
    for path,digest in protocol['source_sha256'].items():
        if sha(path)!=digest: raise ValueError('source changed during experiment')
    write(output/'report.json',report)
    return report['summary']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=BASE)
    print(json.dumps(clean(run(p.parse_args().output)),ensure_ascii=False,indent=2,allow_nan=False))
