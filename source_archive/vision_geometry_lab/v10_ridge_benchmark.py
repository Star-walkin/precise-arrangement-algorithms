"""Frozen offline four-way ridge evidence / shared connector development test.

No learned weights or target paths enter inference. The manual free-span seed
is still required. Dark grooves and a visually plausible path do not establish
rope identity, contact, a new layer, or force timing.
"""
from pathlib import Path
import argparse
import datetime
import json
import time

import cv2
import numpy as np

from .run_centerline_experiment_v9 import PROTOCOL as V9_PROTOCOL_PATH, verify_snapshot, read, sha
from .evaluate_centerline_multiscale_v9 import _dataset, independent_width, score_decoded, aggregate
from .centerline_path_metrics_v6 import protocol_sha256
from .centerline_review_v5 import read_clean_image
from .v10_seed_ridge_trace import TRACE_METHODS, PARAMETERS, trace_seed_ridges


LAB = Path(__file__).resolve().parent
BASE = LAB / 'outputs/v10_ridge_comparison'
MANIFEST = LAB / 'outputs/centerline_v9/dataset/manifest.json'
NEW_SOURCES = ('v10_ridge_benchmark.py', 'v10_seed_ridge_trace.py', 'v10_ridge_candidates.py', 'wound_rope_tracker.py')
REPRESENTATIVE_IDS = (
    'dual_20260924_111500__left__f00246',
    'C07__R_run_stop_run_20260922_01__t012.033__f000368',
    'dual_20260923_113745__left__f00550',
    'C04__dual_20260923_113029__t05__left',
    'C05__dual_20260923_113029__t25__left',
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def freeze(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new directory; never overwrite a frozen experiment')
    verify_snapshot(read(V9_PROTOCOL_PATH))
    # Metadata/quality are used solely for sample accounting, not for choosing
    # strip extent, scales, thresholds or candidate paths.
    _, records, _ = _dataset(MANIFEST)
    counts = {split:sum(r['sample']['split']==split for r in records) for split in ('train','holdout')}
    if counts != dict(train=10, holdout=8):
        raise ValueError('Expected fixed 10/8 dataset')
    protocol = dict(version=1, frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        task='four fixed photometric feature pipelines with one seed connector',
        methods=list(TRACE_METHODS), parameters=PARAMETERS,
        manual_free_seed=True, target_geometry_input=False, quality_input=False,
        main_polarity='bright_only; dark_response is diagnostic, never automatically a rope center',
        new_method_peak_gate='method-specific peak_response/supported mask and bright polarity',
        legacy_control='unchanged V3 _features, but NEW shared connector; not a full old-UI replay',
        legacy_support='original V3 narrower normal range; reported, not silently equated',
        roi='fixed seed-oriented res2 strip, forward1536 native px, normal +/-768; no GT ROI',
        selection='image top1 only; no reference-selected alternative; all outputs identity_unconfirmed',
        primary='V6 5-native-pixel arc-length-weighted downstream coverage and precision',
        path_protocol_sha256=protocol_sha256(), no_reference_cropping=True,
        missing_predictions='retained as empty; never removed from denominator',
        timing='read/hash/width once per image; each method remap+features+connect measured separately',
        representative_ids=list(REPRESENTATIVE_IDS), display_crop='GT+seed bbox for illustrations ONLY; not inference',
        threshold_tuning_after_scores=False, raw_vs_normalized_ablation=False,
        warning='Configured pipeline comparison, not proof of isolated normalization benefit, dense rope identity, or subpixel physical accuracy',
        counts=counts, source_snapshot={str((LAB/name).resolve()):sha(LAB/name) for name in NEW_SOURCES},
        previous_protocol_sha256=sha(V9_PROTOCOL_PATH), manifest_sha256=sha(MANIFEST),
        hardware_access=False, force_command=None)
    write_json(output/'protocol.json',protocol)
    print(json.dumps(dict(protocol=str(output/'protocol.json'), methods=list(TRACE_METHODS))))


def verify(output):
    protocol = read(Path(output)/'protocol.json')
    verify_snapshot(read(V9_PROTOCOL_PATH))
    if protocol['previous_protocol_sha256'] != sha(V9_PROTOCOL_PATH) or protocol['manifest_sha256'] != sha(MANIFEST):
        raise ValueError('Frozen dataset/protocol changed')
    if protocol['parameters'] != PARAMETERS or protocol['methods'] != list(TRACE_METHODS):
        raise ValueError('Pipeline parameters changed after freeze')
    for path, expected in protocol['source_snapshot'].items():
        if sha(path) != expected:
            raise ValueError('Pipeline source changed after freeze: '+path)
    return protocol


def full_image_coverage(trace, image_shape):
    """Map strip observation support to native/2; diagnostic only."""
    height, width = image_shape[:2]
    gy, gx = np.indices((height//2,width//2),dtype=np.float32)
    # Inverse of V9's pixel-center convention: native=(map+.5)*2-.5.
    gx, gy = (gx+.5)*2-.5, (gy+.5)*2-.5
    geometry, maps = trace['geometry'], trace['ridge_maps']
    origin = geometry['origin_xy']; d = geometry['d_xy']; n = geometry['n_xy']
    dx, dy = gx-origin[0], gy-origin[1]
    u, v = dx*d[0]+dy*d[1], dx*n[0]+dy*n[1]
    resolution = geometry['resolution']
    mx = ((u-geometry['u_min'])/resolution).astype(np.float32)
    my = ((v-geometry['v_min'])/resolution).astype(np.float32)
    valid = cv2.remap(maps['observation_mask'].astype(np.float32),mx,my,
                     cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    return (valid>=1.-1e-6).astype(np.uint8)


def _draw_lines(image, segments, color, crop, scale, thickness):
    for segment in segments:
        points = np.asarray(segment,float)
        if len(points) >= 2:
            points = np.rint((points-np.array(crop[:2]))*scale).astype(np.int32)
            cv2.polylines(image,[points],False,color,thickness,cv2.LINE_AA)


def render_representative(output, sample, label, results, image):
    # Read saved scored references, not inferred tail labels, for display.
    reference = results[0]['metrics']['score_support']['full_reference_segments']
    seed = label['polylines']['free_seed']
    points = np.vstack([np.asarray(seed,float), *[np.asarray(s,float) for s in reference if len(s)]])
    lower = np.maximum(np.floor(points.min(axis=0)-96),0).astype(int)
    upper = np.minimum(np.ceil(points.max(axis=0)+96),[image.shape[1],image.shape[0]]).astype(int)
    x0,y0 = lower; x1,y1 = upper
    crop = [int(x0),int(y0),int(x1),int(y1)]
    scale = min(900/(x1-x0),550/(y1-y0))
    rw,rh = max(1,round((x1-x0)*scale)),max(1,round((y1-y0)*scale))
    panels = []
    for row in results:
        panel = np.full((rh+108,900,3),22,np.uint8)
        scene = cv2.resize(image[y0:y1,x0:x1],(rw,rh),interpolation=cv2.INTER_AREA)
        _draw_lines(scene,reference,(50,220,50),crop,scale,2)
        _draw_lines(scene,[seed],(255,230,0),crop,scale,2)
        _draw_lines(scene,row['decoded']['segments'],(255,70,255),crop,scale,2)
        panel[108:108+rh,:rw] = scene
        metric = row['metrics']['downstream']
        def fmt(value): return 'n/a' if value is None else f'{value*100:.1f}%'
        lines = [row['method'],f"Candidate coverage {fmt(metric['covered_fraction'])} | precision {fmt(metric['prediction_precision_at_5px'])}",
                 f"Green=partial GT; cyan=manual seed; magenta=TOP1 CANDIDATE",
                 f"Identity/contact/force UNCONFIRMED | feature+trace {row['timing']['method_ms']:.0f} ms"]
        for j,line in enumerate(lines):
            cv2.putText(panel,line,(9,21+24*j),cv2.FONT_HERSHEY_SIMPLEX,.48,(225,225,225),1,cv2.LINE_AA)
        panels.append(panel)
    montage = np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:])])
    destination = Path(output)/'representatives'/(sample['sample_id']+'.jpg')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists(): raise FileExistsError(destination)
    ok, encoded = cv2.imencode('.jpg',montage,[cv2.IMWRITE_JPEG_QUALITY,94])
    if not ok: raise RuntimeError('Cannot encode plot')
    with destination.open('xb') as stream: stream.write(encoded.tobytes())
    return dict(sample_id=sample['sample_id'],path=str(destination.resolve()),display_crop=crop,
                crop_used_for_inference=False,source_image_sha256=sample['image_sha256'])


def run(output):
    output = Path(output); protocol = verify(output)
    if (output/'summary.json').exists() or (output/'rows').exists():
        raise FileExistsError('Do not overwrite or silently resume partial field results')
    _,records,_ = _dataset(MANIFEST)
    cv2.setNumThreads(2)
    rows, illustrations = [],[]
    for record in records:
        sample,label,quality = record['sample'],record['label'],record['quality']
        started = time.perf_counter()
        # Check the exact image again just before inference.
        if sha(sample['image_path']) != sample['image_sha256']: raise ValueError('Image changed')
        image = cv2.cvtColor(np.asarray(read_clean_image(sample)),cv2.COLOR_RGB2BGR)
        read_ms = (time.perf_counter()-started)*1000
        width_started = time.perf_counter()
        width = independent_width(image,label['polylines']['free_seed'])
        width_ms = (time.perf_counter()-width_started)*1000
        sample_rows = []
        for method in TRACE_METHODS:
            coverage = None; trace = None; method_ms = 0.
            if width['width_px'] is None:
                decoded = dict(segments=[], reason='independent_width_unavailable', identity_confirmed=False)
            else:
                trace = trace_seed_ridges(image,label['polylines']['free_seed'],width['width_px'],
                                          method=method,return_maps=True)
                method_ms = trace['metadata']['elapsed_ms']
                coverage = full_image_coverage(trace,image.shape)
                trace.pop('ridge_maps')
                decoded = dict(segments=trace['observed_segments'],reason=trace['reason'],identity_confirmed=False)
            score = score_decoded(label,quality,decoded,coverage,sample['image_shape_hw'])
            row = dict(sample_id=sample['sample_id'], split=sample['split'],group_id=sample['group_id'],
                cohort=record['cohort'], method=method,
                prediction_status='loaded' if width['width_px'] is not None else 'width_unavailable',
                prediction_kind='computed_photometric_candidate_not_model_export',
                image_sha256=sample['image_sha256'],annotation_sha256=record['annotation_sha256'],
                quality_sha256=quality['quality_sha256'],independent_width=width,
                decoded=decoded, trace=trace, **score,
                timing=dict(read_hash_decode_ms=read_ms,width_ms=width_ms,method_ms=method_ms,
                            detector_pipeline_ms=read_ms+width_ms+method_ms,
                            excludes='evaluation, drawing, SDK, UI; no caches'),
                identity_confirmed=False,new_layer_confirmed=False,force_command=None)
            write_json(output/'rows'/f"{sample['sample_id']}__{method}.json",row)
            rows.append(row);sample_rows.append(row)
        if sample['sample_id'] in REPRESENTATIVE_IDS:
            illustrations.append(render_representative(output,sample,label,sample_rows,image))
        print(json.dumps(dict(sample_id=sample['sample_id'], completed_methods=len(sample_rows),
            downstream_coverage={r['method']:r['metrics']['downstream']['covered_fraction'] for r in sample_rows})),flush=True)
    verify(output)
    summaries = {method:aggregate([r for r in rows if r['method']==method]) for method in TRACE_METHODS}
    timings = {method:{key:float(np.median([r['timing'][key] for r in rows if r['method']==method]))
                       for key in ('method_ms','detector_pipeline_ms')} for method in TRACE_METHODS}
    result = dict(protocol=protocol,protocol_sha256=sha(output/'protocol.json'),
        source_and_data_unchanged=True, rows=rows, summaries=summaries, median_timing_ms=timings,
        illustrations=illustrations, no_rows_dropped=True,
        status='development_candidate_comparison_not_identity_or_field_acceptance',
        dark_grooves_not_scored_as_rope_centers=True, no_gt_candidate_selection=True,
        physically_correct_instances_labeled=False, automatic_event_timing_validated=False,
        hardware_access=False, force_command=None)
    write_json(output/'summary.json',result)
    print(json.dumps(dict(summary=str(output/'summary.json'),rows=len(rows)),ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','run','verify'))
    parser.add_argument('--output',type=Path,default=BASE)
    args = parser.parse_args()
    if args.stage == 'freeze': freeze(args.output)
    elif args.stage == 'run': run(args.output)
    else: verify(args.output);print('Frozen ridge sources/data verified')


if __name__ == '__main__': main()
