"""Fixed seed-downstream path metrics; geometry scores are NOT rope identity.

Protocol v1 is fixed before reading V6 predictions. A downstream half-plane is
defined by the first-to-last free-seed direction and its last point. Segments
crossing this plane or an ignored polygon are clipped by interpolation; removed
parts never join scored support across a gap. This local definition suits the
current monotone image paths, not a rope that loops back behind the seed plane.

Quality/ignore regions affect scoring only. They are never detector inputs.
The full-path score uses the same metric definition as the downstream score;
it is a V6 comparison, not a rewrite of historical V5 metric values.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import cv2
import numpy as np


PROTOCOL = {
    'name': 'seed_downstream_path_metrics', 'version': 1,
    'coordinate_system': 'full_image_xy',
    'max_sampling_spacing_px': 4., 'matching_tolerance_px': 5.,
    'seed_connection_tolerance_px': 5.,
    'downstream_rule': 'dot(point - seed_last, unit(seed_last - seed_first)) >= 0',
    'boundary_policy': 'downstream plane included; ignore polygon interiors and boundaries excluded',
    'clipping': 'linear segment intersections; never bridge removed intervals or input path gaps',
    'coverage_and_precision_weights': 'polyline arc-length Voronoi weights; not raw point counts',
    'run_lengths': 'contiguous matched/unmatched sampled arc-length cells within one scored segment',
    'seed_connection': 'minimum distance from seed endpoint to each uncut prediction segment; no joining segments',
    'normal_error': 'reference segment normal, conditional on matching tolerance',
    'quality_use': 'score support only; never image, seed, decoder, or refinement input',
    'empty_prediction': 'reference coverage 0 when reference exists; precision and errors null',
    'empty_reference': 'coverage, precision, extra fraction, and errors null; never success',
    'identity_confirmed': False, 'physical_precision_validated': False,
    'applicability': 'local monotone path; half-plane is not along-curve ordering for turn-back/looping paths',
    'parameter_selection': 'fixed before reading V6 prediction scores; identical train and holdout rules',
}
EPS = 1e-9


def protocol_sha256():
    return hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def _line(points):
    array = np.asarray(points, dtype=float)
    if array.size == 0:
        return np.empty((0, 2), float)
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError('Polyline must contain finite [x,y] coordinates')
    array = array[np.r_[True, np.linalg.norm(np.diff(array, axis=0), axis=1) > EPS]]
    return array if len(array) >= 2 else np.empty((0, 2), float)


def _polygons(polygons):
    result = []
    for polygon in polygons:
        points = np.asarray(polygon, float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
            raise ValueError('Ignore polygons require at least three finite [x,y] vertices')
        result.append(points)
    return result


def _seed_geometry(free_seed):
    seed = _line(free_seed)
    if len(seed) < 2 or np.linalg.norm(seed[-1]-seed[0]) <= EPS:
        raise ValueError('Free seed needs distinct first and last points')
    direction = seed[-1]-seed[0]
    return seed[-1], direction/np.linalg.norm(direction)


def _cross(a, b):
    return float(a[0]*b[1]-a[1]*b[0])


def _cuts_at_polygon(a, b, polygon):
    vector = b-a
    norm = float(vector@vector)
    cuts = []
    for c, d in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = d-c
        denominator = _cross(vector, edge)
        if abs(denominator) > EPS:
            t, u = _cross(c-a, edge)/denominator, _cross(c-a, vector)/denominator
            if -EPS <= t <= 1+EPS and -EPS <= u <= 1+EPS:
                cuts.append(float(np.clip(t, 0, 1)))
        elif abs(_cross(c-a, vector)) <= EPS and norm > EPS:
            # Collinear boundary endpoints still partition a path on an ignore edge.
            cuts.extend(float(np.clip((point-a)@vector/norm, 0, 1)) for point in (c, d))
    return cuts


def _outside(point, polygons):
    return all(cv2.pointPolygonTest(polygon.astype(np.float32), tuple(map(float, point)), False) < 0
               for polygon in polygons)


def clip_scoring_segments(segments, *, free_seed=None, downstream=False, ignore_polygons=()):
    """Return exact clipped polylines, with no interpolation across excluded gaps.

    The retained endpoints are limiting points at a cut boundary; a boundary has
    zero arc length. A path lying along an ignore boundary is entirely removed.
    """
    polygons = _polygons(ignore_polygons)
    origin, direction = _seed_geometry(free_seed) if downstream else (None, None)
    clipped = []
    for source in segments:
        line = _line(source)
        run = []
        def finish():
            if len(run) >= 2:
                clipped.append(np.asarray(run).tolist())
            run.clear()
        for a, b in zip(line[:-1], line[1:]):
            cuts = [0., 1.]
            if downstream:
                first, last = float((a-origin)@direction), float((b-origin)@direction)
                if abs(last-first) > EPS:
                    fraction = -first/(last-first)
                    if 0 < fraction < 1:
                        cuts.append(fraction)
            for polygon in polygons:
                cuts.extend(_cuts_at_polygon(a, b, polygon))
            cuts = sorted(set(round(float(t), 14) for t in cuts))
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                if hi-lo <= EPS:
                    continue
                middle = a+(b-a)*((lo+hi)*.5)
                keep = (not downstream or float((middle-origin)@direction) >= -EPS) and _outside(middle, polygons)
                if not keep:
                    finish()
                    continue
                start, end = a+(b-a)*lo, a+(b-a)*hi
                if run and np.linalg.norm(np.asarray(run[-1])-start) > EPS:
                    finish()
                if not run:
                    run.append(start)
                if np.linalg.norm(np.asarray(run[-1])-end) > EPS:
                    run.append(end)
        finish()
    return clipped


def _sampling(segments):
    pieces = []
    for index, source in enumerate(segments):
        points = _line(source)
        if len(points) < 2:
            continue
        lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        positions = np.linspace(0., lengths[-1], max(1, int(np.ceil(lengths[-1]/PROTOCOL['max_sampling_spacing_px'])))+1)
        cell_edges = np.r_[0., (positions[:-1]+positions[1:])*.5, lengths[-1]]
        samples = np.column_stack([np.interp(positions, lengths, points[:, dim]) for dim in (0, 1)])
        indices = np.clip(np.searchsorted(lengths, positions, side='right')-1, 0, len(points)-2)
        tangents = points[indices+1]-points[indices]
        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
        pieces.append(dict(index=index, vertices=points, vertex_positions=lengths,
                           points=samples, tangents=tangents, cell_edges=cell_edges,
                           weights=np.diff(cell_edges), length=float(lengths[-1])))
    return pieces


def _nearest(points, segments):
    distance = np.full(len(points), np.inf)
    closest = np.full((len(points), 2), np.nan)
    for source in segments:
        vertices = _line(source)
        for a, b in zip(vertices[:-1], vertices[1:]):
            vector = b-a
            projection = np.clip((points-a)@vector/(vector@vector), 0, 1)
            candidate = a+projection[:, None]*vector
            current = np.linalg.norm(points-candidate, axis=1)
            better = current < distance
            distance[better], closest[better] = current[better], candidate[better]
    return distance, closest


def _weighted_percentile(values, weights, quantile=.95):
    if not len(values):
        return None
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    return float(np.interp(quantile*np.sum(weights), np.cumsum(weights), values))


def _runs(piece, mask):
    boundaries = np.r_[0, np.flatnonzero(np.diff(mask.astype(int)))+1, len(mask)]
    runs = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        lo, hi = piece['cell_edges'][start], piece['cell_edges'][end]
        endpoints = np.column_stack([np.interp([lo, hi], piece['vertex_positions'], piece['vertices'][:, dim])
                                     for dim in (0, 1)])
        runs.append(dict(matched=bool(mask[start]), segment_index=piece['index'],
                         length_px=float(hi-lo), from_xy=endpoints[0].tolist(), to_xy=endpoints[1].tolist()))
    return runs


def _score_scope(reference_segments, predicted_segments):
    reference = _sampling(reference_segments)
    prediction = _sampling(predicted_segments)
    reference_length = sum(piece['length'] for piece in reference)
    prediction_length = sum(piece['length'] for piece in prediction)
    result = dict(status='scored_reference' if reference_length else 'no_scored_reference',
        precision_score_available=bool(reference_length),
        reference_length_px=reference_length, predicted_length_px=prediction_length,
        reference_segment_count=len(reference), predicted_segment_count=len(prediction),
        reference_samples=sum(len(piece['points']) for piece in reference),
        predicted_samples=sum(len(piece['points']) for piece in prediction),
        covered_fraction=None, covered_reference_length_px=None,
        prediction_precision_at_5px=None, extra_predicted_fraction=None, extra_predicted_length_px=None,
        distance_p95_px=None, normal_error_p95_on_matched_px=None,
        longest_matched_reference_run_px=None, longest_uncovered_reference_run_px=None,
        longest_supported_prediction_run_px=None, reference_coverage_gaps=[],
        identity_confirmed=False, physical_precision_validated=False,
        warning='Conditional normal error requires coverage; geometric precision is not rope identity; no score is not success')
    if not reference_length:
        return result
    distances, normal_errors, weights, all_matched, runs = [], [], [], [], []
    for piece in reference:
        distance, nearest = _nearest(piece['points'], predicted_segments)
        matched = distance <= PROTOCOL['matching_tolerance_px']
        normal = np.column_stack([-piece['tangents'][:, 1], piece['tangents'][:, 0]])
        error = np.abs(np.sum((nearest-piece['points'])*normal, axis=1))
        distances.extend(distance); normal_errors.extend(error); weights.extend(piece['weights']); all_matched.extend(matched)
        runs.extend(_runs(piece, matched))
    distances, normal_errors, weights, matched = map(np.asarray, (distances, normal_errors, weights, all_matched))
    covered = float(weights[matched].sum())
    result.update(covered_fraction=covered/reference_length, covered_reference_length_px=covered,
        distance_p95_px=_weighted_percentile(distances, weights) if np.isfinite(distances).all() else None,
        normal_error_p95_on_matched_px=_weighted_percentile(normal_errors[matched], weights[matched]),
        longest_matched_reference_run_px=max((run['length_px'] for run in runs if run['matched']), default=0.),
        longest_uncovered_reference_run_px=max((run['length_px'] for run in runs if not run['matched']), default=0.),
        reference_coverage_gaps=[{key: value for key, value in run.items() if key != 'matched'}
                                 for run in runs if not run['matched']])
    supported_prediction = 0.
    prediction_runs = []
    for piece in prediction:
        distance, _ = _nearest(piece['points'], reference_segments)
        matched = distance <= PROTOCOL['matching_tolerance_px']
        supported_prediction += float(piece['weights'][matched].sum())
        prediction_runs.extend(_runs(piece, matched))
    if prediction_length:
        precision = supported_prediction/prediction_length
        result.update(prediction_precision_at_5px=precision, extra_predicted_fraction=max(0., 1.-precision),
            extra_predicted_length_px=max(0., prediction_length-supported_prediction),
            longest_supported_prediction_run_px=max((run['length_px'] for run in prediction_runs if run['matched']), default=0.))
    return result


def _seed_connection(predicted_segments, endpoint):
    lengths, distances = [], []
    for source in predicted_segments:
        line = _line(source)
        if len(line) < 2:
            continue
        distances.append(float(_nearest(endpoint[None], [line])[0][0]))
        lengths.append(float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum()))
    connected = [distance <= PROTOCOL['seed_connection_tolerance_px'] for distance in distances]
    length = sum(lengths)
    connected_length = sum(value for value, matched in zip(lengths, connected) if matched)
    return dict(connected=any(connected), distance_px=min(distances) if distances else None,
        connected_prediction_segment_count=sum(connected), prediction_segment_count=len(lengths),
        connected_prediction_length_fraction=connected_length/length if length else None,
        disconnected_prediction_length_px=length-connected_length,
        from_unclipped_predictions=True, segments_are_not_joined=True, identity_confirmed=False)


def evaluate_paths(reference, predicted_segments, free_seed, *, ignore_polygons=()):
    """JSON-ready paired full/downstream score. Inputs are never modified.

    Seed connection is a separate geometric diagnostic on uncut predictions, so
    a score exclusion cannot fabricate a connection or split a detector path.
    Continuous-run scores use clipped support and cannot bridge ignored areas.
    """
    endpoint, direction = _seed_geometry(free_seed)
    full_reference = clip_scoring_segments([reference], ignore_polygons=ignore_polygons)
    full_prediction = clip_scoring_segments(predicted_segments, ignore_polygons=ignore_polygons)
    downstream_reference = clip_scoring_segments([reference], free_seed=free_seed, downstream=True,
                                                ignore_polygons=ignore_polygons)
    downstream_prediction = clip_scoring_segments(predicted_segments, free_seed=free_seed, downstream=True,
                                                 ignore_polygons=ignore_polygons)
    return dict(schema_version=1, protocol=dict(PROTOCOL), protocol_sha256=protocol_sha256(),
        downstream_plane=dict(origin_xy=endpoint.tolist(), unit_direction_xy=direction.tolist()),
        full_path=_score_scope(full_reference, full_prediction),
        downstream=_score_scope(downstream_reference, downstream_prediction),
        seed_connection=_seed_connection(predicted_segments, endpoint),
        score_support=dict(full_reference_segments=full_reference, full_prediction_segments=full_prediction,
                           downstream_reference_segments=downstream_reference,
                           downstream_prediction_segments=downstream_prediction),
        exclusion_polygon_count=len(ignore_polygons), quality_used_for_prediction_input=False,
        identity_confirmed=False, physical_precision_validated=False)


def load_v6_prediction(folder, sample, label):
    """Torch-free validation of V6's unpadded ROI-spanning probability export."""
    from .centerline_benchmark_v5 import load_prediction
    loaded = load_prediction(folder, sample, label)
    if loaded is None:
        return None
    return _validate_v6_export(loaded)


def _validate_v6_export(loaded):
    meta, body, center = loaded
    if meta.get('preprocess_version') != 1:
        raise ValueError('V6 preprocess_version=1 required')
    geometry = meta.get('letterbox')
    if not isinstance(geometry, dict):
        raise ValueError('V6 letterbox metadata required; padded maps cannot be decoded as a whole ROI')
    native = meta['native_roi_hw']
    target = meta.get('model_input_hw')
    if (not isinstance(target, list) or len(target) != 2 or
            any(type(value) is not int or value <= 0 for value in target)):
        raise ValueError('Invalid V6 model_input_hw')
    height, width = native
    target_height, target_width = target
    scale = min(target_width/width, target_height/height)
    content_width = max(1, min(target_width, int(round(width*scale))))
    content_height = max(1, min(target_height, int(round(height*scale))))
    left, top = (target_width-content_width)//2, (target_height-content_height)//2
    expected = dict(native_hw=native, model_input_hw=target, content_hw=[content_height,content_width],
        content_xyxy=[left,top,left+content_width,top+content_height],
        padding_ltrb=[left,top,target_width-content_width-left,target_height-content_height-top])
    if any(geometry.get(key) != value for key,value in expected.items()):
        raise ValueError('V6 letterbox geometry mismatch')
    expected_scale = np.array([content_width/width,content_height/height])
    for scale_value in (geometry.get('scale_xy'),meta.get('scale_xy')):
        if np.shape(scale_value) != (2,) or not np.allclose(scale_value,expected_scale,rtol=1e-9,atol=1e-12):
            raise ValueError('V6 letterbox scale mismatch')
    if list(body.shape) != [content_height,content_width]:
        raise ValueError('V6 exported probabilities must exclude letterbox padding')
    if (body.ndim != 2 or center.shape != body.shape or not np.isfinite(body).all()
            or not np.isfinite(center).all() or np.any((body<0)|(body>1)|(center<0)|(center>1))):
        raise ValueError('Invalid V6 probabilities')
    return dict(meta,validated_prediction_format='v6_letterbox_unpadded_content'),body,center


def load_export_prediction(folder, sample, label):
    """Explicit V5 direct-resize / V6 letterbox dispatch, without invented fields."""
    from .centerline_benchmark_v5 import load_prediction
    loaded = load_prediction(folder,sample,label)
    if loaded is None:
        return None
    meta,body,center = loaded
    if 'preprocess_version' in meta or 'letterbox' in meta:
        return _validate_v6_export(loaded)
    # A V6 export with stripped transform metadata must not silently downgrade.
    expected_semantics = 'model-resolution HxW float32 spanning native ROI'
    semantics = meta.get('probability_arrays',{})
    if ('architecture' in meta or any(semantics.get(key) != expected_semantics
                                      for key in ('body_probability','center_probability'))):
        raise ValueError('Unrecognized direct-ROI export; V6 transform metadata may be missing')
    target = meta.get('model_input_hw')
    if (not isinstance(target,list) or len(target) != 2 or
            any(type(value) is not int or value <= 0 for value in target) or list(body.shape) != target):
        raise ValueError('V5 direct-ROI probability/model dimensions mismatch')
    height,width = meta['native_roi_hw']
    expected_scale = [target[1]/width,target[0]/height]
    if np.shape(meta.get('scale_xy')) != (2,) or not np.allclose(meta['scale_xy'],expected_scale,rtol=1e-9,atol=1e-12):
        raise ValueError('V5 direct-ROI scale mismatch')
    if (body.ndim != 2 or center.shape != body.shape or not np.isfinite(body).all()
            or not np.isfinite(center).all() or np.any((body<0)|(body>1)|(center<0)|(center>1))):
        raise ValueError('Invalid V5 probabilities')
    return dict(meta,validated_prediction_format='v5_direct_roi_resize'),body,center


def _aggregate(rows):
    summaries = []
    for split in ('train','holdout'):
        selected = [row for row in rows if row['split'] == split]
        for method in ('cv_only','model_only','model_plus_cv'):
            for scope in ('full_path','downstream'):
                available = [row['methods'][method]['metrics'][scope] for row in selected
                             if row.get('methods') and row['methods'].get(method,{}).get('metrics')]
                scored = [metric for metric in available if metric['precision_score_available']]
                ref_length = sum(metric['reference_length_px'] for metric in scored)
                pred_length = sum(metric['predicted_length_px'] for metric in scored)
                covered = sum(metric['covered_reference_length_px'] for metric in scored)
                extra = sum(metric['extra_predicted_length_px'] or 0. for metric in scored)
                normal = [metric['normal_error_p95_on_matched_px'] for metric in scored
                          if metric['normal_error_p95_on_matched_px'] is not None]
                summaries.append(dict(split=split,method=method,scope=scope,
                    total_manifest_samples=len(selected),samples_with_method_output=len(available),
                    scored_reference_samples=len(scored),unscored_reference_samples=len(available)-len(scored),
                    unavailable_method_samples=len(selected)-len(available),
                    reference_length_px=ref_length,predicted_length_px=pred_length,
                    covered_fraction=covered/ref_length if ref_length else None,
                    prediction_precision_at_5px=1.-extra/pred_length if pred_length else None,
                    extra_predicted_fraction=extra/pred_length if pred_length else None,
                    median_sample_matched_normal_p95_px=float(np.median(normal)) if normal else None,
                    samples_with_matched_normal_error=len(normal),
                    normal_error_aggregation='median of available per-sample p95; not pooled p95; conditional on matches',
                    identity_confirmed=False,physical_precision_validated=False))
    return summaries


def evaluate_manifest(manifest_path, predictions_dir=None):
    """Use unchanged V5 candidate/decoder/refiner and independently measured width.

    Target labels and quality polygons are read for scores/provenance only.
    Missing/rejected model exports remain unavailable, not fabricated successes.
    """
    from .centerline_benchmark_v5 import decode_heatmaps, refinement_segments, score_centerline
    from .centerline_quality_v5 import load_label_quality
    from .centerline_refine_v5 import refine_centerline
    from .centerline_review_v5 import load_manifest, validate_annotation
    from .geometry import analyze_frame, read_image
    from .prepare_centerline_v5 import check_groups, file_hash
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    check_groups(manifest['samples'])
    rows = []
    for sample in manifest['samples']:
        row = {key:sample[key] for key in ('sample_id','split','group_id')}
        annotation_path = Path(sample['annotation_path'])
        if not annotation_path.exists():
            rows.append(dict(row,status='pending_annotation',methods=None))
            continue
        label = validate_annotation(json.loads(annotation_path.read_text('utf-8-sig')),sample)
        quality = load_label_quality(sample)
        row.update(annotation_sha256=file_hash(annotation_path),quality_sha256=quality['quality_sha256'],
                   image_sha256=sample['image_sha256'],quality_regions=quality['regions'])
        if label['status'] in ('absent','unidentifiable'):
            rows.append(dict(row,status=label['status'],methods=None,
                             reason='No precise center reference; not counted as success'))
            continue
        if file_hash(sample['image_path']) != sample['image_sha256']:
            raise ValueError('Image source hash mismatch')
        frame = read_image(sample['image_path'])
        seed = label['polylines']['free_seed']
        reference = label['polylines']['centerline']
        ignored = label['ignore_polygons']+quality['precision_exclude_polygons']
        def score(segments):
            result = evaluate_paths(reference,segments,seed,ignore_polygons=ignored)
            # Preserve the exact historical full-path score definition as an
            # explicitly named control, alongside the new length-based metrics.
            result['v5_full_path_control'] = score_centerline(reference,segments,ignore_polygons=ignored)
            return result
        started = time.perf_counter()
        analysis = analyze_frame(frame,seed_points=seed)
        cv_ms = (time.perf_counter()-started)*1000.
        candidates = analysis['wound_curve'].get('candidates',[])
        cv_segments = candidates[0]['observed_segments'] if candidates else []
        width = analysis['rope'].get('width_px')
        methods = dict(cv_only=dict(status='evaluated',metrics=score(cv_segments),elapsed_ms=cv_ms,
                                   selection='unchanged V5 first-ranked candidate; free_seed only'),
                       model_only=dict(status='pending_prediction'),model_plus_cv=dict(status='pending_prediction'))
        rejection = None
        try:
            loaded = load_export_prediction(predictions_dir,sample,label) if predictions_dir is not None else None
        except (ValueError,KeyError,OSError) as error:
            loaded = None
            rejection = str(error)
            methods['model_only'] = dict(status='rejected_prediction',reason=rejection)
            methods['model_plus_cv'] = dict(status='rejected_prediction',reason=rejection)
        if loaded is not None:
            meta,body,center = loaded
            decoded = decode_heatmaps(body,center,meta['roi_xyxy'],seed,width)
            methods['model_only'] = dict(status='evaluated',metrics=score(decoded['segments']),
                decoder_reason=decoded['reason'],inference_ms=meta['inference_ms'])
            refined = []
            started = time.perf_counter()
            for segment in decoded['segments']:
                result = refine_centerline(frame,segment,expected_width_px=decoded['widths'][:len(segment)])
                refined.extend(refinement_segments(result))
            methods['model_plus_cv'] = dict(status='evaluated',metrics=score(refined),
                                            refinement_ms=(time.perf_counter()-started)*1000.)
            row['prediction_source'] = {key:meta.get(key) for key in
                ('checkpoint','checkpoint_sha256','input_sha256','output_sha256','architecture',
                 'raster_encoding_version','preprocess_version','letterbox','validated_prediction_format')}
        rows.append(dict(row,status='reviewed_reference',methods=methods,
            independent_free_width_px=width,precision_exclude_polygons=ignored,
            human_free_seed_used=True,quality_used_for_prediction_input=False,
            prediction_rejection=rejection))
    base = Path(__file__).parent
    return dict(schema_version=1,protocol=dict(PROTOCOL),protocol_sha256=protocol_sha256(),
        manifest_sha256=file_hash(manifest_path),implementation_sha256=file_hash(__file__),
        unchanged_v5_dependencies={name:file_hash(base/name) for name in
            ('centerline_benchmark_v5.py','centerline_refine_v5.py','geometry.py')},
        rows=rows,split_summaries=_aggregate(rows),
        annotation_guided_input='human free_seed only; target curves/quality are score-only',
        decoder_policy='unchanged V5 decoder and thresholds; independent analyze_frame free-rope width',
        no_score_is_not_success=True,identity_confirmed=False,field_accuracy_established=False,
        physical_control=False,force_command=None)


def write_protocol(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output/'protocol.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(dict(protocol=PROTOCOL, protocol_sha256=protocol_sha256()), stream,
                  indent=2, ensure_ascii=False, allow_nan=False)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol-only', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).parent/'outputs/centerline_v5/pilot/manifest.json')
    parser.add_argument('--predictions-dir', type=Path)
    args = parser.parse_args(argv)
    if args.protocol_only:
        print(json.dumps(dict(protocol=str(write_protocol(args.output)),protocol_sha256=protocol_sha256())))
        return
    if args.output.exists():
        raise FileExistsError('Use a new output directory; V5 and previous V6 reports are never overwritten')
    cv2.setNumThreads(2)
    report = evaluate_manifest(args.manifest,args.predictions_dir)
    write_protocol(args.output)
    path = args.output/'summary.json'
    with path.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps(dict(report=str(path),samples=len(report['rows']),protocol_sha256=protocol_sha256())))


if __name__ == '__main__':
    main()
