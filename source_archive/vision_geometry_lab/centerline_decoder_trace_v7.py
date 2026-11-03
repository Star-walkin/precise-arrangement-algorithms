"""Posthoc instrumentation of the frozen V5 decoder, with identical outputs.

No target center/edge, ignore polygon, or quality region is a tracing input.
This diagnostic changes no thresholds or paths and makes no identity claim.
"""
from collections import Counter
from pathlib import Path
import argparse
import json

import cv2
import numpy as np


def trace_decode_heatmaps(body, center, roi, free_seed, width_prior):
    """Return original-compatible ``decoded`` and explanatory ``trace``.

Numerical path operations and comparisons mirror decode_heatmaps verbatim.
First rejection is recorded in the original short-circuit order for each peak;
several different first-rejection reasons can coexist in a stopped step.
"""
    body=np.asarray(body,np.float32);center=np.asarray(center,np.float32)
    trace=dict(steps=[],first_stop=None,identity_confirmed=False,
               target_reference_used=False,quality_used=False,
               independent_width_prior_px=float(width_prior) if width_prior is not None and np.isfinite(width_prior) else None)
    def finish(decoded,reason,step=None):
        trace['first_stop']=dict(reason=reason,step_index=step['step_index'] if step else None,
            scan_distance_px=step['scan_distance_px'] if step else None,
            predicted_xy=step['predicted_xy'] if step else None,
            peak_count=step.get('peak_count') if step else None,
            eligible_candidate_count=step.get('eligible_candidate_count') if step else None,
            peak_rejection_counts=step.get('peak_rejection_counts',{}) if step else {},
            propagated_width_px=step.get('propagated_width_px') if step else None)
        trace['accepted_point_count']=sum(s.get('accepted') is not None for s in trace['steps'])
        trace['returned_point_count']=sum(len(segment) for segment in decoded['segments'])
        return dict(decoded=decoded,trace=trace)
    def empty(reason):
        return finish(dict(segments=[],widths=[],reason=reason,identity_confirmed=False),reason)
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
    stop_reason='scan_extent_exhausted';last_record=None
    for u in np.arange(0.,np.hypot(x1-x0,y1-y0),step):
        predicted=last_v+slope*step if path else 0.
        v=np.arange(predicted-width*2,predicted+width*2+1,1.)
        xy=origin+u*d+v[:,None]*n
        grid=(xy-[x0,y0]+.5)*scale-.5
        inside=(grid[:,0]>=0)&(grid[:,0]<body.shape[1]-1)&(grid[:,1]>=0)&(grid[:,1]<body.shape[0]-1)
        record=dict(step_index=len(trace['steps']),scan_distance_px=float(u),
            predicted_xy=(origin+u*d+predicted*n).tolist(),predicted_normal_offset_px=float(predicted),
            propagated_width_px=float(width),fixed_width_allowed_px=[.45*float(width_prior),2.*float(width_prior)],
            slope=float(slope),search_normal_range_px=[float(v[0]),float(v[-1])],
            samples=int(len(v)),inside_sample_count=int(inside.sum()),peaks=[],peak_count=0,
            eligible_candidate_count=0,peak_rejection_counts={},accepted=None,step_stop_reason=None)
        trace['steps'].append(record);last_record=record
        if inside.sum()<5:
            stop_reason='fewer_than_5_map_supported_samples';record['step_stop_reason']=stop_reason
            break
        def sample(array):
            return cv2.remap(array,grid[:,0].astype(np.float32)[None],grid[:,1].astype(np.float32)[None],
                             cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)[0]
        bp,cp=sample(body),sample(center)
        peaks=np.flatnonzero((cp[1:-1]>=cp[:-2])&(cp[1:-1]>cp[2:])&(cp[1:-1]>=.5))+1
        local_peaks=np.flatnonzero((cp[1:-1]>=cp[:-2])&(cp[1:-1]>cp[2:]))+1
        inside_indices=np.flatnonzero(inside)
        max_index=int(inside_indices[np.argmax(cp[inside])])
        record.update(peak_count=int(len(peaks)),center_max_inside=float(cp[inside].max()),
            body_max_inside=float(bp[inside].max()),
            center_max_inside_sample_index=max_index,
            center_at_search_endpoints=[float(cp[0]),float(cp[-1])],
            interior_local_peak_max=float(cp[local_peaks].max()) if len(local_peaks) else None,
            joint_above_half_sample_count=int(np.sum(inside&(cp>=.5)&(bp>=.5))))
        candidates=[]
        for k in peaks:
            peak=dict(sample_index=int(k),xy=xy[k].tolist(),normal_offset_px=float(v[k]),
                center_probability=float(cp[k]),body_probability=float(bp[k]),
                offset_from_propagated_px=float(abs(v[k]-predicted)),first_rejection=None)
            record['peaks'].append(peak)
            if not inside[k] or bp[k]<.5 or abs(v[k]-predicted)>width*.75:
                peak['first_rejection']=('peak_outside_map_support' if not inside[k] else
                    'body_probability_below_half' if bp[k]<.5 else 'peak_offset_exceeds_0.75_propagated_width')
                continue
            a=b=k
            while a>0 and inside[a-1] and bp[a-1]>=.5: a-=1
            while b<len(v)-1 and inside[b+1] and bp[b+1]>=.5: b+=1
            candidate_width=(b-a)/np.sqrt(1+slope*slope)
            peak.update(body_interval_sample_indices=[int(a),int(b)],
                body_interval_xy=[xy[a].tolist(),xy[b].tolist()],candidate_width_px=float(candidate_width),
                left_flank_probability=float(bp[a-1]) if a>0 and inside[a-1] else None,
                right_flank_probability=float(bp[b+1]) if b<len(v)-1 and inside[b+1] else None)
            if (a==0 or b==len(v)-1 or not inside[a-1] or not inside[b+1]
                    or bp[a-1]>=.5 or bp[b+1]>=.5
                    or not .45*width_prior<=candidate_width<=2.0*width_prior):
                if a==0:
                    rejection='body_interval_reaches_negative_search_limit'
                elif b==len(v)-1:
                    rejection='body_interval_reaches_positive_search_limit'
                elif not inside[a-1]:
                    rejection='negative_body_flank_outside_map_support'
                elif not inside[b+1]:
                    rejection='positive_body_flank_outside_map_support'
                elif bp[a-1]>=.5:
                    rejection='negative_flank_not_below_half'
                elif bp[b+1]>=.5:
                    rejection='positive_flank_not_below_half'
                elif candidate_width<.45*width_prior:
                    rejection='body_width_below_0.45_prior'
                else:
                    rejection='body_width_above_2.0_prior'
                peak['first_rejection']=rejection
                continue
            quality=float(cp[k])-.10*abs(v[k]-predicted)/width
            candidates.append((quality,k,candidate_width))
            peak['candidate_quality']=float(quality)
        candidates.sort(reverse=True)
        record['eligible_candidate_count']=len(candidates)
        record['peak_rejection_counts']=dict(Counter(p['first_rejection'] for p in record['peaks'] if p['first_rejection']))
        if not candidates:
            stop_reason='no_center_local_peak_at_half' if not len(peaks) else 'all_center_peaks_rejected'
            record['step_stop_reason']=stop_reason
            break
        if len(candidates)>1:
            record['competition']=dict(top_quality_gap=float(candidates[0][0]-candidates[1][0]),
                top_peak_separation_px=float(abs(v[candidates[0][1]]-v[candidates[1][1]])),
                separation_threshold_px=float(width*.25),quality_gap_threshold=.05)
        if len(candidates)>1 and candidates[0][0]-candidates[1][0]<.05 and abs(v[candidates[0][1]]-v[candidates[1][1]])>width*.25:
            record['step_stop_reason']='competing_ridge_stop'
            return finish(dict(segments=[path] if len(path)>=2 else [],widths=widths,
                        reason='competing_ridge_stop',identity_confirmed=False),'competing_ridge_stop',record)
        _,k,new_width=candidates[0];new_v=float(v[k])
        if path: slope=.65*slope+.35*(new_v-last_v)/step
        path.append(xy[k].tolist());widths.append(float(new_width))
        record['accepted']=dict(xy=xy[k].tolist(),width_px=float(new_width),sample_index=int(k))
        last_v=new_v;width=.75*width+.25*new_width
    return finish(dict(segments=[path] if len(path)>=2 else [],widths=widths,
                reason='fixed_local_monotone_candidate_stopped_at_missing_evidence',identity_confirmed=False),
                stop_reason,last_record)


def run(output):
    from .centerline_benchmark_v5 import decode_heatmaps
    from .centerline_path_metrics_v6 import load_export_prediction
    from .centerline_quality_v5 import load_label_quality
    from .centerline_review_v5 import load_manifest, validate_annotation
    from .prepare_centerline_v5 import file_hash
    base=Path(__file__).resolve().parent
    output=Path(output)
    if output.exists():
        raise FileExistsError('Use a new diagnostic output directory; frozen results are never overwritten')
    manifest_path=base/'outputs/centerline_v5/pilot/manifest.json'
    run_dir=base/'outputs/centerline_v6/runs/context_letterbox_seed20260926_steps300'
    path_report_path=run_dir/'path_metrics/summary.json'
    manifest=load_manifest(manifest_path)
    path_report=json.loads(path_report_path.read_text('utf-8'))
    if file_hash(manifest_path)!=path_report['manifest_sha256']:
        raise ValueError('Manifest differs from frozen path run')
    for filename,expected in path_report['unchanged_v5_dependencies'].items():
        if file_hash(base/filename)!=expected:
            raise ValueError('Frozen source dependency changed: '+filename)
    previous={row['sample_id']:row for row in path_report['rows']}
    rows=[]
    for sample in manifest['samples']:
        label=validate_annotation(json.loads(Path(sample['annotation_path']).read_text('utf-8-sig')),sample)
        quality=load_label_quality(sample)
        old=previous[sample['sample_id']]
        if (file_hash(sample['image_path'])!=sample['image_sha256'] or
                file_hash(sample['annotation_path'])!=old['annotation_sha256'] or
                quality['quality_sha256']!=old['quality_sha256']):
            raise ValueError('Image/label/quality differs from frozen path run')
        meta,body,center=load_export_prediction(run_dir/'all_export/predictions',sample,label)
        if meta['output_sha256']!=old['prediction_source']['output_sha256']:
            raise ValueError('Prediction differs from frozen path run')
        # This width was derived from RGB by analyze_frame, not from target labels.
        # Reuse the exact frozen scalar to isolate decoder behavior from re-estimation.
        width=old['independent_free_width_px'];seed=label['polylines']['free_seed']
        args=(body,center,meta['roi_xyxy'],seed,width)
        original=decode_heatmaps(*args)
        traced=trace_decode_heatmaps(*args)
        if traced['decoded']!=original:
            raise AssertionError('Trace output differs from frozen decoder: '+sample['sample_id'])
        rows.append(dict(sample_id=sample['sample_id'],split=sample['split'],
            group_id=sample['group_id'],image_sha256=sample['image_sha256'],
            annotation_sha256=old['annotation_sha256'],quality_sha256=old['quality_sha256'],
            prediction_sha256=meta['output_sha256'],checkpoint_sha256=meta['checkpoint_sha256'],
            original_decoder_exact_equality=True,independent_free_width_px=width,
            width_source='cached frozen RGB analyze_frame free-span width, not target label',
            human_free_seed=seed,**traced,
            existing_downstream_metrics=old['methods']['model_only']['metrics']['downstream']))
    splits=[]
    for split in ('train','holdout'):
        selected=[r for r in rows if r['split']==split]
        splits.append(dict(split=split,samples=len(selected),
            exact_equality_count=sum(r['original_decoder_exact_equality'] for r in selected),
            first_stop_counts=dict(Counter(r['trace']['first_stop']['reason'] for r in selected)),
            terminal_peak_rejection_counts=dict(sum((Counter(r['trace']['first_stop']['peak_rejection_counts']) for r in selected),Counter()))))
    report=dict(schema_version=1,diagnostic='V7 frozen decoder exact-parity posthoc trace',
        threshold_changes=False,new_path_selection=False,target_reference_used_for_trace=False,
        quality_used_for_trace=False,identity_confirmed=False,physical_control=False,
        manifest_sha256=file_hash(manifest_path),frozen_path_report_sha256=file_hash(path_report_path),
        original_decoder_source_sha256=file_hash(base/'centerline_benchmark_v5.py'),
        trace_source_sha256=file_hash(__file__),rows=rows,split_summaries=splits,
        limitations=['A stopped step may have different first rejection gates for different peaks.',
            'Cached independent width reproduces the prior run; this does not establish that width is correct for wound rope.',
            'Existing reference scores are copied for context only, never inputs to tracing.',
            'An output path or a passing local gate is not confirmation of rope identity.'])
    output.mkdir(parents=True)
    with (output/'summary.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,ensure_ascii=False,allow_nan=False)
    lines=['# V7 frozen decoder trace — posthoc diagnosis, no decoder changes','',
        'All 12 original-versus-instrumented outputs must match exactly, including every coordinate, width and legacy reason. '
        'The input width is the cached independent RGB free-span estimate from the frozen V6 path run. '
        'Target center/edge lines and quality regions never enter the trace. Existing scores are context only.','',
        '| Split / sample | Independent width px | First stop scan u / xy | Accepted points | Center peaks / eligible | First stop / per-peak rejections |',
        '| --- | ---: | --- | ---: | --- | --- |']
    for row in rows:
        stop=row['trace']['first_stop'];xy=stop['predicted_xy']
        position=f"{stop['scan_distance_px']:.2f} / ({xy[0]:.2f}, {xy[1]:.2f})" if xy else 'n/a'
        why=stop['reason']+'; '+', '.join(f'{k}: {v}' for k,v in stop['peak_rejection_counts'].items())
        lines.append(f"| {row['split']} / {row['sample_id']} | {row['independent_free_width_px']:.3f} | {position} | {row['trace']['accepted_point_count']} | {stop['peak_count']} / {stop['eligible_candidate_count']} | {why} |")
    lines+=['','`scan u` is distance along the original free-seed direction, not distance along GT. '
        'The stop xy is the propagated search center, not a measured target point. '
        'Per-peak rejection counts use the original short-circuit order; a single step can have multiple causes. '
        'The JSON preserves each sampled peak, body bounds, width, acceptance, and stop detail.','',
        'No gate was relaxed, no gap was bridged, and no automatic identity or physical precision is established.']
    with (output/'README.md').open('x',encoding='utf-8') as stream:
        stream.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(output/'summary.json'),split_summaries=splits),ensure_ascii=False))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'outputs/centerline_v7/decoder_trace')
    args=parser.parse_args();cv2.setNumThreads(2);run(args.output)


if __name__=='__main__':
    main()
