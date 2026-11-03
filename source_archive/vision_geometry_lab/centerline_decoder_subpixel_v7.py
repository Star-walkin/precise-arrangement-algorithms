"""Preregistered width-only decoder ablation; not a production replacement.

Integer mode is an exact V5 control. Subpixel mode only replaces sampled body
span by linearly interpolated 0.5 crossings, retaining all rejection gates.
The resulting width follows the original state update; no gap is bridged.
"""
from pathlib import Path
import argparse
import json

import cv2
import numpy as np


def interpolated_body_width(probability,a,b,slope=0.):
    """Width at 0.5 crossings, on a 1-native-pixel-spaced profile.

Return None when either measured below-threshold flank is unavailable. A flat
top is permitted, but a flat threshold/broad body reaching the boundary is not
invented into a bounded interval. This is a probability contour, not a physical
edge calibration.
"""
    bp=np.asarray(probability)
    if (a<=0 or b>=len(bp)-1 or a>b or bp[a-1]>=.5 or bp[b+1]>=.5
            or bp[a]<.5 or bp[b]<.5):
        return None
    left=float(a-1)+float((.5-float(bp[a-1]))/(float(bp[a])-float(bp[a-1])))
    right=float(b)+float((float(bp[b])-.5)/(float(bp[b])-float(bp[b+1])))
    return float((right-left)/np.sqrt(1+slope*slope))


def decode_heatmaps_subpixel(body,center,roi,free_seed,width_prior,*,width_mode='subpixel'):
    """Original output schema; no target reference or score exclusions accepted."""
    if width_mode not in ('integer','subpixel'):
        raise ValueError('width_mode must be integer or subpixel')
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
                    or bp[a-1]>=.5 or bp[b+1]>=.5):
                continue
            if width_mode=='subpixel':
                candidate_width=interpolated_body_width(bp,a,b,slope)
            if candidate_width is None or not .45*width_prior<=candidate_width<=2.0*width_prior:
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


BASE=Path(__file__).resolve().parent
PREREG=BASE/'outputs/centerline_v7/subpixel_width_preregistration.json'
DEFAULT_OUTPUT=BASE/'outputs/centerline_v7/subpixel_width'
CONTEXT_RUN=BASE/'outputs/centerline_v6/runs/context_letterbox_seed20260926_steps300'


def freeze_source(output):
    """Write source identity before any new real-data decoding in this module."""
    from .prepare_centerline_v5 import file_hash
    from .centerline_path_metrics_v6 import protocol_sha256
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    prereg=json.loads(PREREG.read_text('utf-8'))
    if protocol_sha256()!=prereg['comparison']['protocol_sha256']:
        raise ValueError('Path protocol differs from preregistration')
    report=dict(stage='source_frozen_before_real_comparison',
        source_sha256=file_hash(__file__),tests_sha256=file_hash(BASE/'tests/test_centerline_decoder_subpixel_v7.py'),
        preregistration_sha256=file_hash(PREREG),protocol_sha256=protocol_sha256(),
        frozen_dependencies={name:file_hash(BASE/name) for name in
            ('centerline_benchmark_v5.py','centerline_path_metrics_v6.py','centerline_decoder_trace_v7.py')})
    with (output/'source_freeze.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
    return report


def aggregate(rows):
    groups=[]
    for split in ('train','holdout'):
        selected=[row for row in rows if row['split']==split]
        for mode in ('integer','subpixel'):
            for scope in ('full_path','downstream'):
                scores=[row['modes'][mode]['metrics'][scope] for row in selected]
                reference=sum(s['reference_length_px'] for s in scores)
                predicted=sum(s['predicted_length_px'] for s in scores)
                covered=sum(s['covered_reference_length_px'] or 0 for s in scores)
                extra=sum(s['extra_predicted_length_px'] or 0 for s in scores)
                normal=[s['normal_error_p95_on_matched_px'] for s in scores if s['normal_error_p95_on_matched_px'] is not None]
                groups.append(dict(split=split,width_mode=mode,scope=scope,samples=len(selected),
                    reference_length_px=reference,predicted_length_px=predicted,
                    covered_reference_length_px=covered,covered_fraction=covered/reference if reference else None,
                    prediction_precision_at_5px=1-extra/predicted if predicted else None,
                    unsupported_predicted_length_px=extra,extra_predicted_fraction=extra/predicted if predicted else None,
                    supported_predicted_length_px=predicted-extra,
                    seed_connected_count=sum(row['modes'][mode]['metrics']['seed_connection']['connected'] for row in selected),
                    empty_prediction_count=sum(not row['modes'][mode]['decoded']['segments'] for row in selected),
                    zero_coverage_count=sum(s['covered_fraction']==0 for s in scores),
                    median_sample_matched_normal_p95_px=float(np.median(normal)) if normal else None,
                    samples_with_matched_normal_error=len(normal),
                    median_longest_uncovered_reference_run_px=float(np.median([s['longest_uncovered_reference_run_px'] for s in scores if s['longest_uncovered_reference_run_px'] is not None])),
                    identity_confirmed=False))
    return groups


def evaluate(output,*,predictions_dir=None,secondary=False):
    from .centerline_benchmark_v5 import decode_heatmaps
    from .centerline_path_metrics_v6 import evaluate_paths,load_export_prediction,protocol_sha256
    from .centerline_quality_v5 import load_label_quality
    from .centerline_review_v5 import load_manifest,validate_annotation
    from .prepare_centerline_v5 import file_hash
    output=Path(output)
    if (output/'summary.json').exists():
        raise FileExistsError('Previous comparisons cannot be overwritten')
    freeze=json.loads((output/'source_freeze.json').read_text('utf-8'))
    if (freeze['source_sha256']!=file_hash(__file__) or freeze['tests_sha256']!=file_hash(BASE/'tests/test_centerline_decoder_subpixel_v7.py')
            or freeze['preregistration_sha256']!=file_hash(PREREG)):
        raise ValueError('Source/tests/preregistration changed after freeze')
    for filename,expected in freeze['frozen_dependencies'].items():
        if file_hash(BASE/filename)!=expected:
            raise ValueError('Frozen dependency changed: '+filename)
    prereg=json.loads(PREREG.read_text('utf-8'))
    if protocol_sha256()!=freeze['protocol_sha256']:
        raise ValueError('Scoring protocol changed')
    manifest_path=BASE/'outputs/centerline_v5/pilot/manifest.json'
    if file_hash(manifest_path)!=prereg['manifest_sha256']:
        raise ValueError('Manifest differs from preregistration')
    manifest=load_manifest(manifest_path)
    old_report_path=CONTEXT_RUN/'path_metrics/summary.json'
    old_report=json.loads(old_report_path.read_text('utf-8'))
    old_rows={r['sample_id']:r for r in old_report['rows']}
    folder=Path(predictions_dir) if predictions_dir else CONTEXT_RUN/'all_export/predictions'
    rows=[]
    for sample in manifest['samples']:
        old=old_rows[sample['sample_id']]
        label=validate_annotation(json.loads(Path(sample['annotation_path']).read_text('utf-8-sig')),sample)
        quality=load_label_quality(sample)
        if (file_hash(sample['image_path'])!=sample['image_sha256'] or
                file_hash(sample['annotation_path'])!=old['annotation_sha256'] or
                quality['quality_sha256']!=old['quality_sha256']):
            raise ValueError('Image/annotation/quality changed from frozen input')
        meta,body,center=load_export_prediction(folder,sample,label)
        if not secondary and (meta['checkpoint_sha256']!=prereg['checkpoint_sha256'] or
                              meta['output_sha256']!=old['prediction_source']['output_sha256']):
            raise ValueError('Primary ablation must use preregistered context300 exports')
        seed=label['polylines']['free_seed'];width=old['independent_free_width_px']
        args=(body,center,meta['roi_xyxy'],seed,width)
        original=decode_heatmaps(*args)
        integer=decode_heatmaps_subpixel(*args,width_mode='integer')
        if integer!=original:
            raise AssertionError('Integer control differs from original V5 decoder')
        subpixel=decode_heatmaps_subpixel(*args,width_mode='subpixel')
        # Only after both paths are decoded may the scoring reference be read.
        reference=label['polylines']['centerline']
        ignored=label['ignore_polygons']+quality['precision_exclude_polygons']
        modes={mode:dict(decoded=decoded,metrics=evaluate_paths(reference,decoded['segments'],seed,ignore_polygons=ignored))
               for mode,decoded in (('integer',integer),('subpixel',subpixel))}
        if not secondary and modes['integer']['metrics']['downstream']!=old['methods']['model_only']['metrics']['downstream']:
            raise AssertionError('Integer control scoring differs from frozen V6 run')
        rows.append(dict(sample_id=sample['sample_id'],split=sample['split'],group_id=sample['group_id'],
            image_sha256=sample['image_sha256'],annotation_sha256=old['annotation_sha256'],
            quality_sha256=quality['quality_sha256'],prediction_sha256=meta['output_sha256'],
            checkpoint_sha256=meta['checkpoint_sha256'],independent_free_width_px=width,
            integer_exact_equality=True,human_free_seed_used=True,target_reference_used_for_decode=False,
            quality_used_for_decode=False,modes=modes))
    report=dict(schema_version=1,experiment='preregistered subpixel body width single-factor ablation',
        secondary_diagnostic=bool(secondary),independent_blind_test=False,main_fixed_benchmark_replaced=False,
        identity_confirmed=False,physical_precision_validated=False,physical_control=False,
        source_freeze=freeze,preregistration_sha256=file_hash(PREREG),manifest_sha256=file_hash(manifest_path),
        frozen_width_source_report_sha256=file_hash(old_report_path),predictions_directory=str(folder.resolve()),
        rows=rows,split_summaries=aggregate(rows),
        limitations=['Width-state feedback follows the original update and can change subsequent search samples.',
            'A probability 0.5 contour is not a calibrated physical rope edge.',
            'Unsupported prediction means unmatched to the partial scored reference, not proven wrong-rope identity.',
            'Train scores measure fit; held-out groups are development holdout, not a fresh blind test.',
            'No fine RGB refinement, new path bridging, threshold tuning, or hardware operation.'])
    with (output/'summary.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps(dict(report=str(output/'summary.json'),split_summaries=report['split_summaries']),ensure_ascii=False))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--freeze-only',action='store_true')
    parser.add_argument('--predictions-dir',type=Path)
    parser.add_argument('--secondary-diagnostic',action='store_true')
    args=parser.parse_args();cv2.setNumThreads(2)
    if args.freeze_only:
        print(json.dumps(freeze_source(args.output)))
    else:
        evaluate(args.output,predictions_dir=args.predictions_dir,secondary=args.secondary_diagnostic)


if __name__=='__main__':
    main()
