"""Offline paired comparison. Human target labels are ONLY used in scoring.

Missing labels/predictions are explicit pending results, not zeros or successes.
The small heatmap decoder is a fixed local monotone baseline, not a general
instance tracker; it abstains on ambiguity and does not bridge lost evidence.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import cv2
import numpy as np

from .geometry import read_image, dump_json, analyze_frame, write_image
from .prepare_centerline_v5 import OUTPUT, check_groups, file_hash
from .centerline_refine_v5 import refine_centerline
from .centerline_quality_v5 import load_label_quality


def resample(points, spacing=4.):
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all():
        return np.empty((0,2))
    d = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    keep = np.r_[True, np.diff(d) > 1e-6]
    p, d = p[keep], d[keep]
    if len(p) < 2:
        return np.empty((0,2))
    t = np.linspace(0,d[-1],max(2,int(np.ceil(d[-1]/spacing))+1))
    return np.column_stack([np.interp(t,d,p[:,k]) for k in range(2)])


def outside_ignored(points, polygons):
    keep = np.ones(len(points), bool)
    for polygon in polygons:
        poly = np.asarray(polygon,np.float32)
        if len(poly) >= 3:
            keep &= np.asarray([cv2.pointPolygonTest(poly,tuple(map(float,p)),False)<0 for p in points])
    return keep


def nearest_on_segments(points, segments):
    best = np.full(len(points), np.inf)
    nearest = np.full((len(points),2),np.nan)
    for segment in segments:
        p = np.asarray(segment,float)
        if p.ndim != 2 or len(p)<2 or p.shape[1]!=2 or not np.isfinite(p).all():
            continue
        for a,b in zip(p[:-1],p[1:]):
            v=b-a; norm=float(v@v)
            if norm<=1e-8:
                continue
            t=np.clip((points-a)@v/norm,0,1)
            q=a+t[:,None]*v
            distance=np.linalg.norm(points-q,axis=1)
            take=distance<best
            best[take]=distance[take];nearest[take]=q[take]
    return best,nearest


def score_centerline(reference, predicted_segments, *, ignore_polygons=(), tolerance_px=5.):
    points=resample(reference)
    original_count=len(points)
    if not len(points):
        return dict(status='no_center_reference', reference_samples=0, distance_p95_px=None,
                    reference_samples_before_exclusion=0,excluded_reference_samples=0,
                    precision_score_available=False,physical_precision_validated=False,
                    covered_fraction=None, normal_error_p95_on_matched_px=None)
    tangent=np.gradient(points,axis=0)
    tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-6)
    keep=outside_ignored(points,ignore_polygons)
    points,tangent=points[keep],tangent[keep]
    if not len(points):
        return dict(status='all_reference_ignored',reference_samples=0,distance_p95_px=None,
                    reference_samples_before_exclusion=original_count,
                    excluded_reference_samples=original_count,
                    precision_score_available=False,physical_precision_validated=False,
                    covered_fraction=None,normal_error_p95_on_matched_px=None)
    distance,q=nearest_on_segments(points,predicted_segments)
    matched=distance<=tolerance_px
    normals=np.column_stack([-tangent[:,1],tangent[:,0]])
    normal_error=np.abs(np.sum((q[matched]-points[matched])*normals[matched],axis=1))
    return dict(status='scored_manual_reference',reference_samples=len(points),
                reference_samples_before_exclusion=original_count,
                excluded_reference_samples=original_count-len(points),
                precision_score_available=True,
                distance_p95_px=float(np.percentile(distance,95)) if np.isfinite(distance).all() else None,
                covered_fraction=float(matched.mean()), matching_tolerance_px=float(tolerance_px),
                normal_error_p95_on_matched_px=float(np.percentile(normal_error,95)) if len(normal_error) else None,
                identity_jump_rate=None, identity_jump_rate_reason='requires independent instance review; distance is not identity',
                physical_precision_validated=False,
                warning='normal error is conditional on matched coverage; always read with covered_fraction')


def decode_heatmaps(body, center, roi, free_seed, width_prior):
    """Conservative seed-anchored local path; never uses target GT or flange."""
    body=np.asarray(body,np.float32);center=np.asarray(center,np.float32)
    empty=lambda reason:dict(segments=[],widths=[],reason=reason,identity_confirmed=False)
    if body.ndim!=2 or body.shape!=center.shape or not np.isfinite(body).all() or not np.isfinite(center).all():
        raise ValueError('Invalid probability arrays')
    if np.any((body<0)|(body>1)|(center<0)|(center>1)):
        raise ValueError('Probability outside [0,1]')
    seed=np.asarray(free_seed,float)
    if seed.ndim!=2 or len(seed)<2 or seed.shape[1]!=2:
        return empty('free_seed_required')
    if width_prior is None or not np.isfinite(width_prior) or width_prior<=0:
        return empty('independent_free_width_unavailable')
    d=seed[-1]-seed[0];length=np.linalg.norm(d)
    if length<5:
        return empty('free_seed_too_short')
    d=d/length;n=np.array([-d[1],d[0]])
    x0,y0,x1,y1=roi
    scale=np.array([body.shape[1]/(x1-x0),body.shape[0]/(y1-y0)])
    width=float(width_prior);step=max(2.,width*.35)
    origin=seed[-1];last_v=0.;slope=0.;path=[];widths=[]
    for u in np.arange(0.,np.hypot(x1-x0,y1-y0),step):
        predicted=last_v+slope*step if path else 0.
        v=np.arange(predicted-width*2,predicted+width*2+1,1.)
        xy=origin+u*d+v[:,None]*n
        # Pixel-center mapping consistent with PIL/OpenCV resize. At native
        # resolution this MUST be the identity, not a half-pixel shift.
        grid=(xy-[x0,y0]+.5)*scale-.5
        inside=(grid[:,0]>=0)&(grid[:,0]<body.shape[1]-1)&(grid[:,1]>=0)&(grid[:,1]<body.shape[0]-1)
        if inside.sum()<5:
            break
        def sample(array):
            return cv2.remap(array,grid[:,0].astype(np.float32)[None],grid[:,1].astype(np.float32)[None],
                             cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)[0]
        bp,cp=sample(body),sample(center)
        peaks=np.flatnonzero((cp[1:-1]>=cp[:-2])&(cp[1:-1]>cp[2:])&(cp[1:-1]>=.5))+1
        candidates=[]
        for k in peaks:
            if not inside[k] or bp[k]<.5 or abs(v[k]-predicted)>width*.75:
                continue
            a=b=k
            while a>0 and inside[a-1] and bp[a-1]>=.5: a-=1
            while b<len(v)-1 and inside[b+1] and bp[b+1]>=.5: b+=1
            candidate_width=(b-a)/np.sqrt(1+slope*slope)
            if (a==0 or b==len(v)-1 or not inside[a-1] or not inside[b+1]
                    or bp[a-1]>=.5 or bp[b+1]>=.5
                    or not .45*width_prior<=candidate_width<=2.0*width_prior):
                continue
            quality=float(cp[k])-.10*abs(v[k]-predicted)/width
            candidates.append((quality,k,candidate_width))
        candidates.sort(reverse=True)
        if not candidates:
            break
        if len(candidates)>1 and candidates[0][0]-candidates[1][0]<.05 and abs(v[candidates[0][1]]-v[candidates[1][1]])>width*.25:
            return dict(segments=[path] if len(path)>=2 else [],widths=widths,
                        reason='competing_ridge_stop',identity_confirmed=False)
        _,k,new_width=candidates[0];new_v=float(v[k])
        if path: slope=.65*slope+.35*(new_v-last_v)/step
        path.append(xy[k].tolist());widths.append(float(new_width))
        last_v=new_v;width=.75*width+.25*new_width
    return dict(segments=[path] if len(path)>=2 else [],widths=widths,
                reason='fixed_local_monotone_candidate_stopped_at_missing_evidence',identity_confirmed=False)


def refinement_segments(result):
    segments=[];run=[]
    for record in result['records']:
        if record['valid']:
            run.append(record['refined_xy'])
        else:
            if len(run)>=2: segments.append(run)
            run=[]
    if len(run)>=2: segments.append(run)
    return segments


def load_prediction(folder,sample,label=None):
    path=Path(folder)/(sample['sample_id']+'.json')
    if not path.exists():
        return None
    meta=json.loads(path.read_text('utf-8'))
    if meta['sample_id']!=sample['sample_id'] or meta['image_sha256']!=sample['image_sha256']:
        raise ValueError('Model prediction source mismatch')
    if meta.get('ground_truth_center_or_edges_used_as_input') is not False:
        raise ValueError('Prediction has unknown/contaminated inputs')
    if sample['split']=='holdout' and sample['group_id'] in meta.get('train_group_ids',[]):
        raise ValueError('Holdout recording leaked into training')
    if not meta.get('real_annotation_trained'):
        raise ValueError('Synthetic checkpoint cannot be scored as a field model')
    # Keep this lightweight reader independent of torch. Version 2 is the
    # peak-normalized center/free-seed protocol in centerline_model_v5.
    if meta.get('raster_encoding_version')!=2:
        raise ValueError('Incompatible raster_encoding_version; regenerate model/predictions with version 2')
    if not meta.get('body_head_supervised') or not meta.get('center_head_supervised'):
        raise ValueError('Both model heads need positive field supervision for this comparison')
    if meta.get('roi_xyxy')!=sample['roi_xyxy']:
        raise ValueError('Prediction ROI mismatch')
    if meta.get('human_free_seed_used') is not True:
        raise ValueError('All compared methods must use the same human free-seed protocol')
    if label is not None and meta.get('annotation_sha256')!=file_hash(sample['annotation_path']):
        raise ValueError('Annotation changed since prediction; rerun inference')
    if label is not None:
        seed_payload=json.dumps(label['polylines']['free_seed'],sort_keys=True,separators=(',',':'),
                                ensure_ascii=False,allow_nan=False).encode('utf-8')
        if meta.get('free_seed_sha256')!=hashlib.sha256(seed_payload).hexdigest():
            raise ValueError('Prediction used a different free seed')
    quality=load_label_quality(sample)
    if meta.get('quality_sha256')!=quality['quality_sha256']:
        raise ValueError('Label quality changed since prediction; rerun inference')
    prediction_path=meta.get('output_npz',meta.get('output'))
    input_path=meta.get('input_npz',meta.get('input'))
    if not prediction_path or not input_path:
        raise ValueError('Prediction/input paths missing')
    if (file_hash(prediction_path)!=meta['output_sha256']
            or file_hash(input_path)!=meta['input_sha256']
            or file_hash(meta['checkpoint'])!=meta['checkpoint_sha256']):
        raise ValueError('Prediction/checkpoint hash mismatch')
    with np.load(prediction_path,allow_pickle=False) as arrays:
        body,center=arrays['body_probability'].copy(),arrays['center_probability'].copy()
    if list(body.shape)!=meta.get('probability_map_hw') or center.shape!=body.shape:
        raise ValueError('Probability dimensions mismatch')
    x0,y0,x1,y1=sample['roi_xyxy']
    if meta.get('native_roi_hw')!=[y1-y0,x1-x0]:
        raise ValueError('Native ROI dimensions mismatch')
    return meta,body,center


def evaluate(manifest_path,predictions=None):
    from .centerline_review_v5 import validate_annotation
    manifest=json.loads(Path(manifest_path).read_text('utf-8'))
    check_groups(manifest['samples']);rows=[]
    for sample in manifest['samples']:
        row=dict(sample_id=sample['sample_id'],split=sample['split'],group_id=sample['group_id'])
        path=Path(sample['annotation_path'])
        if not path.exists():
            rows.append(dict(row,status='pending_human_annotation',methods=None));continue
        label=json.loads(path.read_text('utf-8'))
        label=validate_annotation(label,sample)
        if file_hash(sample['image_path'])!=sample['image_sha256']:
            raise ValueError('Clean image changed')
        quality=load_label_quality(sample)
        manual_ignored=label.get('ignore_polygons',[])
        precision_ignored=quality['precision_exclude_polygons']
        # Quality annotations affect only the score support, never the image,
        # seed, candidate selection, decoder, or original-pixel refinement.
        ignored=manual_ignored+precision_ignored
        row.update(quality_sha256=quality['quality_sha256'],precision_exclusion={
            'manual_ignore_polygons':manual_ignored,
            'quality_exclude_polygons':precision_ignored,
            'quality_regions':quality['regions'],
            'used_only_for_scoring':True,
            'no_score_is_not_success':True})
        if label['status'] in ('unidentifiable','absent'):
            rows.append(dict(row,status=label['status'],methods=None,
                             reason='no center truth; not counted as a precision pass'));continue
        frame=read_image(sample['image_path']);seed=label['polylines']['free_seed']
        started=time.perf_counter()
        result=analyze_frame(frame,seed_points=seed)
        candidates=result['wound_curve'].get('candidates',[])
        cv_segments=candidates[0]['observed_segments'] if candidates else []
        cv_ms=(time.perf_counter()-started)*1000
        reference=label['polylines']['centerline']
        score=lambda segments:score_centerline(reference,segments,ignore_polygons=ignored)
        methods={'cv_only':dict(metrics=score(cv_segments),elapsed_ms=cv_ms,
                    selection='first ranked automatic candidate; only free_seed is supplied',
                    identity_confirmed=False),
                 'model_only':dict(status='pending_field_trained_prediction'),
                 'model_plus_cv':dict(status='pending_field_trained_prediction')}
        loaded=load_prediction(predictions,sample,label) if predictions else None
        if loaded:
            meta,body,center=loaded
            decoded=decode_heatmaps(body,center,meta['roi_xyxy'],seed,result['rope'].get('width_px'))
            segments=decoded['segments']
            methods['model_only']=dict(metrics=score(segments),decoder_reason=decoded['reason'],
                                       inference_ms=meta['inference_ms'],identity_confirmed=False)
            refined=[];refine_start=time.perf_counter()
            for segment in segments:
                r=refine_centerline(frame,segment,expected_width_px=decoded['widths'][:len(segment)])
                refined.extend(refinement_segments(r))
            methods['model_plus_cv']=dict(metrics=score(refined),refine_ms=(time.perf_counter()-refine_start)*1000,
                                         identity_confirmed=False)
        rows.append(dict(row,status='reviewed_reference',methods=methods,
                         label_sha256=file_hash(path),human_free_seed_used=True,
                         note='seed-assisted experiment, not fully automatic detection'))
    return dict(schema_version=1,manifest_sha256=file_hash(manifest_path),rows=rows,
                status='waiting_for_human_labels' if all(r['status']=='pending_human_annotation' for r in rows)
                       else 'partial_offline_comparison',
                acceptance_thresholds_established=False,physical_control=False,
                field_accuracy_established=False,force_command=None)


def manual_diagnostic(output):
    """User-guided 247-frame probe, never mixed with blind field scores."""
    base=Path(__file__).resolve().parent/'outputs/wound_curve_v3/manual_247'
    ref=json.loads((base/'reference.json').read_text('utf-8'))
    prior=json.loads((base/'automatic.json').read_text('utf-8'))
    cap=cv2.VideoCapture(ref['clean_source']);cap.set(cv2.CAP_PROP_POS_FRAMES,ref['frame_index'])
    ok,frame=cap.read();cap.release()
    if not ok: raise ValueError('Cannot read reference frame')
    if hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()!=ref['original_frame_sha256']:
        raise ValueError('Reference pixels changed')
    result=refine_centerline(frame,ref['points_original_px'],expected_width_px=prior['rope']['width_px'])
    result.update(manual_input=True,independent_accuracy_test=False,
                  user_semantics='approximately rope-body centers',
                  width_source='current-frame free-span optical width, not measured wound-body width',
                  precision_ground_truth_available=False)
    dump_json(output/'manual_247_diagnostic.json',result)
    canvas=frame.copy()
    for i,record in enumerate(result['records']):
        p=tuple(np.rint(record['seed_xy']).astype(int));cv2.circle(canvas,p,9,(0,0,255),2)
        cv2.putText(canvas,str(i+1),(p[0]+12,p[1]-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        if record['valid']:
            q=tuple(np.rint(record['refined_xy']).astype(int));cv2.circle(canvas,q,7,(0,255,255),2)
            a,b=(tuple(np.rint(record[k]).astype(int)) for k in ('edge_a_xy','edge_b_xy'))
            cv2.line(canvas,a,b,(0,255,255),2)
    x0,y0=ref['crop_origin'];canvas=canvas[y0:y0+924,x0:x0+2433]
    cv2.putText(canvas,'RED: manual approximate center | YELLOW: paired-edge CANDIDATE, NOT truth',
                (12,36),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
    write_image(output/'manual_247_diagnostic.jpg',canvas)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,default=OUTPUT/'manifest.json')
    parser.add_argument('--output',type=Path,default=OUTPUT.parent/'benchmark')
    parser.add_argument('--predictions-dir',type=Path)
    parser.add_argument('--manual-diagnostic',action='store_true')
    args=parser.parse_args();cv2.setNumThreads(2)
    summary=evaluate(args.manifest,args.predictions_dir)
    dump_json(args.output/'summary.json',summary)
    if args.manual_diagnostic: manual_diagnostic(args.output)
    print(json.dumps(dict(status=summary['status'],samples=len(summary['rows']),
                         report=str(args.output/'summary.json')),ensure_ascii=False))


if __name__=='__main__': main()
