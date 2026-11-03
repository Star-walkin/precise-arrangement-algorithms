"""GT-GUIDED POSTHOC DIAGNOSTIC, never an automatic recognition benchmark.

Fixed context300 exports only. Sample body/center at the downstream annotated
reference and within a fixed +/-5 original-pixel normal window. No model,
decoder, threshold, annotation, quality sidecar, or frozen score is modified.
"""
from pathlib import Path
import argparse
import json

import cv2
import numpy as np

from .centerline_path_metrics_v6 import clip_scoring_segments, load_export_prediction
from .centerline_quality_v5 import load_label_quality
from .centerline_review_v5 import load_manifest, validate_annotation
from .prepare_centerline_v5 import file_hash


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'vision_geometry_lab/outputs/centerline_v6'
RUN = BASE/'runs/context_letterbox_seed20260926_steps300'
CONFIG = dict(threshold=.5, max_original_spacing_px=4., normal_radius_px=5., normal_step_px=1.,
              sampling='equal arc-length cell midpoints; cells no longer than 4 original pixels',
              mapping='(full_xy - roi_origin + 0.5) * (map_wh / roi_wh) - 0.5',
              normal_window_test='same query point must have body and center >= 0.5',
              query_support='downstream, outside quality/manual ignore, inside bilinear map support')


def reference_samples(segments):
    samples, normals, weights = [], [], []
    for vertices in segments:
        points = np.asarray(vertices,float)
        lengths = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
        if lengths[-1] <= 0:
            continue
        edges = np.linspace(0.,lengths[-1],max(1,int(np.ceil(lengths[-1]/CONFIG['max_original_spacing_px'])))+1)
        positions = (edges[:-1]+edges[1:])*.5
        xy = np.column_stack([np.interp(positions,lengths,points[:,dim]) for dim in (0,1)])
        index = np.clip(np.searchsorted(lengths,positions,side='right')-1,0,len(points)-2)
        tangent = points[index+1]-points[index]
        tangent /= np.linalg.norm(tangent,axis=1,keepdims=True)
        samples.extend(xy);normals.extend(np.column_stack([-tangent[:,1],tangent[:,0]]))
        weights.extend(np.diff(edges))
    return np.asarray(samples,float).reshape(-1,2),np.asarray(normals,float).reshape(-1,2),np.asarray(weights,float)


def sample_maps(body, center, points, roi):
    points = np.asarray(points,float)
    x0,y0,x1,y1 = roi
    scale = np.array([body.shape[1]/(x1-x0),body.shape[0]/(y1-y0)])
    grid = (points-np.array([x0,y0])+.5)*scale-.5
    inside = ((grid[:,0]>=0)&(grid[:,0]<body.shape[1]-1)&
              (grid[:,1]>=0)&(grid[:,1]<body.shape[0]-1))
    def sample(array):
        return cv2.remap(array.astype(np.float32),grid[:,0].astype(np.float32)[None],
                         grid[:,1].astype(np.float32)[None],cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)[0]
    return sample(body),sample(center),inside


def query_support(points, polygons, endpoint, direction):
    points = np.asarray(points,float)
    keep = (points-endpoint)@direction >= -1e-9
    for polygon in polygons:
        poly = np.asarray(polygon,np.float32)
        keep &= np.array([cv2.pointPolygonTest(poly,tuple(map(float,p)),False)<0 for p in points])
    return keep


def diagnose(sample,label,quality,meta,body,center,path_row):
    seed = np.asarray(label['polylines']['free_seed'],float)
    endpoint = seed[-1];direction = endpoint-seed[0];direction /= np.linalg.norm(direction)
    polygons = label['ignore_polygons']+quality['precision_exclude_polygons']
    segments = clip_scoring_segments([label['polylines']['centerline']],free_seed=seed,
                                    downstream=True,ignore_polygons=polygons)
    points,normals,weights = reference_samples(segments)
    bp,cp,inside = sample_maps(body,center,points,sample['roi_xyxy'])
    valid = inside&query_support(points,polygons,endpoint,direction)
    offsets = np.arange(-CONFIG['normal_radius_px'],CONFIG['normal_radius_px']+1e-6,CONFIG['normal_step_px'])
    window = points[:,None,:]+offsets[None,:,None]*normals[:,None,:]
    wb,wc,wi = sample_maps(body,center,window.reshape(-1,2),sample['roi_xyxy'])
    window_valid = (wi&query_support(window.reshape(-1,2),polygons,endpoint,direction)).reshape(len(points),len(offsets))
    wb,wc = wb.reshape(window_valid.shape),wc.reshape(window_valid.shape)
    threshold = CONFIG['threshold']
    flags = dict(center_at_least_half=valid&(cp>=threshold),body_at_least_half=valid&(bp>=threshold),
                 both_at_least_half=valid&(cp>=threshold)&(bp>=threshold),
                 normal_window_both_at_least_half=np.any(window_valid&(wb>=threshold)&(wc>=threshold),axis=1))
    length = float(weights.sum())
    response_lengths = {key:float(weights[flag].sum()) for key,flag in flags.items()}
    sb,sc,si = sample_maps(body,center,endpoint[None],sample['roi_xyxy'])
    seed_ok = bool(si[0] and query_support(endpoint[None],polygons,endpoint,direction)[0])
    model = path_row['methods']['model_only']
    row = dict(sample_id=sample['sample_id'],split=sample['split'],group_id=sample['group_id'],
        image_sha256=sample['image_sha256'],annotation_sha256=file_hash(sample['annotation_path']),
        quality_sha256=quality['quality_sha256'],checkpoint_sha256=meta['checkpoint_sha256'],
        prediction_sha256=meta['output_sha256'],reference_length_px=length,
        valid_map_reference_length_px=float(weights[valid].sum()),reference_samples=len(points),
        response_lengths_px=response_lengths,
        response_length_fractions={key:value/length if length else None for key,value in response_lengths.items()},
        seed_endpoint=dict(xy=endpoint.tolist(),body_probability=float(sb[0]) if si[0] else None,
                           center_probability=float(sc[0]) if si[0] else None,
                           in_scoring_support=seed_ok,both_at_least_half=bool(seed_ok and sb[0]>=threshold and sc[0]>=threshold)),
        existing_decoder_reason=model['decoder_reason'],detailed_first_failed_gate_available=False,
        decoder_reason_note='Copied from frozen first/only decoder run; generic stop labels do not identify a specific width/peak gate',
        existing_downstream_path_metrics=model['metrics']['downstream'],
        existing_seed_connection=model['metrics']['seed_connection'],profiles=[])
    for i,point in enumerate(points):
        eligible = np.flatnonzero(window_valid[i])
        best = int(eligible[np.argmax(np.minimum(wb[i,eligible],wc[i,eligible]))]) if len(eligible) else None
        row['profiles'].append(dict(xy=point.tolist(),arc_cell_length_px=float(weights[i]),
            valid_map_support=bool(valid[i]),body=float(bp[i]) if valid[i] else None,
            center=float(cp[i]) if valid[i] else None,
            normal_window_has_joint_response=bool(flags['normal_window_both_at_least_half'][i]),
            gt_guided_best_normal_offset_px=float(offsets[best]) if best is not None else None,
            gt_guided_max_min_body_center=float(min(wb[i,best],wc[i,best])) if best is not None else None))
    return row


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new posthoc output directory; prior reports are never overwritten')
    manifest_path = ROOT/'vision_geometry_lab/outputs/centerline_v5/pilot/manifest.json'
    manifest = load_manifest(manifest_path)
    path_report_path = RUN/'path_metrics/summary.json'
    path_report = json.loads(path_report_path.read_text('utf-8'))
    path_rows = {row['sample_id']:row for row in path_report['rows']}
    rows = []
    for sample in manifest['samples']:
        label = validate_annotation(json.loads(Path(sample['annotation_path']).read_text('utf-8-sig')),sample)
        quality = load_label_quality(sample)
        meta,body,center = load_export_prediction(RUN/'all_export/predictions',sample,label)
        prior = path_rows[sample['sample_id']]
        if (prior['annotation_sha256']!=file_hash(sample['annotation_path']) or
                prior['quality_sha256']!=quality['quality_sha256'] or
                prior['prediction_source']['output_sha256']!=meta['output_sha256']):
            raise ValueError('Posthoc input differs from frozen path metrics')
        rows.append(diagnose(sample,label,quality,meta,body,center,prior))
    groups = []
    for split in ('train','holdout'):
        selected = [row for row in rows if row['split']==split]
        length = sum(row['reference_length_px'] for row in selected)
        totals = {key:sum(row['response_lengths_px'][key] for row in selected)
                  for key in selected[0]['response_lengths_px']}
        groups.append(dict(split=split,samples=len(selected),reference_length_px=length,
            response_length_fractions={key:value/length if length else None for key,value in totals.items()},
            seed_joint_response_count=sum(row['seed_endpoint']['both_at_least_half'] for row in selected)))
    report = dict(schema_version=1,diagnostic='GT-guided posthoc probability-response diagnostic',
        gt_guided=True,posthoc=True,automatic_recognition_metric=False,independent_accuracy_test=False,
        decoder_rerun=False,threshold_tuned=False,identity_confirmed=False,physical_precision_validated=False,
        config=CONFIG,manifest_sha256=file_hash(manifest_path),frozen_path_report_sha256=file_hash(path_report_path),
        implementation_sha256=file_hash(__file__),rows=rows,split_summaries=groups,
        limitations=[
            'GT direction/location and a GT-centred normal window select these samples; no automatic path is inferred.',
            'Probability >= 0.5 is an uncalibrated fixed response check, not verified rope identity.',
            'Local joint response does not prove a connected ridge, compatible paired body width, or decoder reachability.',
            'Stored decoder stop labels are coarse; no particular width gate is attributed without further evidence.',
            'Partial manual references retain annotation uncertainty beyond explicit quality exclusions.',
            'Normal-window queries outside downstream/quality/map support are unavailable, never successes.'])
    output.mkdir(parents=True)
    with (output/'summary.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps(dict(report=str(output/'summary.json'),split_summaries=groups),ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE/'context300_gt_guided_response_posthoc')
    args = parser.parse_args();cv2.setNumThreads(2);run(args.output)


if __name__=='__main__':
    main()
