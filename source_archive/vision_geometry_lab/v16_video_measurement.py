"""Frozen, automatic-seed video stress test; no training, labels or hardware.

V14 checkpoints were evaluated with human free-rope seeds. This experiment uses
only current-frame automatic seeds and is NOT a continuation/passage of their
training gate. Image-gap proxies are neither axial rope counts nor physical
contact/new-layer confirmations. CV extraction and Torch inference run separately.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np

LAB = Path(__file__).resolve().parent
ROOT = LAB.parent
BASE = LAB / 'outputs/v16_event_feasibility'
SESSIONS = {
    'left': ROOT / '现场视频/训练采集/双相机同步/dual_20260924_111500/left_DA9627065.avi',
    'right': ROOT / '现场视频/训练采集/双相机同步/dual_20260924_111143/right_DA9626975.avi',
}
ARMS = ('center_only', 'joint')
MODELS = {a: LAB / 'outputs/v14_joint_supervision/development' / a / 'checkpoint.pt' for a in ARMS}
SOURCES = ('v16_video_measurement.py', 'rope_detector.py', 'arc_detector.py',
           'geometry.py', 'wound_rope_tracker.py', 'v11_path_contract.py',
           'v11_path_decode.py', 'v11_path_model.py', 'v14_joint_boundary_model.py')
RULES = dict(minimum_direct_points=8, minimum_contiguous_span_widths=3.,
             maximum_median_tangent_angle_deg=15., upstream_excluded=True,
             width_source='current-frame free-rope optical stripe width; not physical diameter',
             proxy='median_direct_distance_px / optical_width_px - 0.5',
             comparison_support='unchanged V3 direct links: interior foot and distance < max(40px,3 optical widths)',
             gap_bridging=False, negative_values_clipped=False)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(str(path))
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def selected_frames(frame_count, fps, stride=2, end_s=60.):
    if fps <= 0 or not np.isfinite(fps) or frame_count <= 0 or stride < 1:
        raise ValueError('invalid video sampling metadata')
    return [i for i in range(int(frame_count)) if i % stride == 0 and i / fps <= end_s + 1e-9]


def freeze(output):
    import cv2
    output = Path(output)
    if output.exists():
        raise FileExistsError('V16 needs a fresh output root; do not replace a trial')
    videos = {}
    for name, path in SESSIONS.items():
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise ValueError('cannot open ' + str(path))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            videos[name] = dict(path=str(path), sha256=sha(path), fps=fps, frame_count=count,
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                selected_indices=selected_frames(count, fps))
        finally:
            cap.release()
    protocol = dict(version=16, frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        task='automatic_seed_distribution_stress_test', videos=videos,
        models={a: dict(path=str(p), sha256=sha(p)) for a, p in MODELS.items()},
        source_snapshot={str(LAB / f): sha(LAB / f) for f in SOURCES},
        reference_protocol=dict(path=str(LAB / 'outputs/v14_joint_supervision/protocol.json'),
            sha256=sha(LAB / 'outputs/v14_joint_supervision/protocol.json')),
        sampling=dict(stride_frames=2, start_s=0., end_s=60., chronological=True),
        model=dict(seed=20260928, base_channels=8, cpu_threads=2, unchanged_decoder=True),
        measurement_rules=RULES, labels_or_reference_event_times_used=False,
        automatic_seed='current detected observed_segment, directed by observed arc proximity only',
        old_gates_overridden=False, experiment_is_original_gate_pass=False,
        hardware_access=False, force_command=None, deployment_allowed=False)
    write(output / 'protocol.json', protocol)
    print(json.dumps(dict(stage='freeze', protocol_sha256=sha(output / 'protocol.json')), ensure_ascii=False), flush=True)


def verify(output, verify_videos=True):
    output = Path(output)
    p = read(output / 'protocol.json')
    if p['version'] != 16 or p['measurement_rules'] != RULES:
        raise ValueError('wrong V16 protocol')
    for path, expected in p['source_snapshot'].items():
        if sha(path) != expected:
            raise ValueError('frozen source changed: ' + path)
    for value in list(p['models'].values()) + [p['reference_protocol']]:
        if sha(value['path']) != value['sha256']:
            raise ValueError('frozen model/reference changed: ' + value['path'])
    if verify_videos:
        for value in p['videos'].values():
            if sha(value['path']) != value['sha256']:
                raise ValueError('video changed: ' + value['path'])
    return p


def automatic_seed(rope, arc):
    from .wound_rope_tracker import _entry
    if not rope.get('valid'):
        return None, 'free_rope_unavailable'
    width = rope.get('width_px')
    if width is None or not np.isfinite(width) or width <= 0:
        return None, 'optical_width_unavailable'
    entry, reason = _entry(rope, arc, np.empty((0, 2)))
    if entry is None:
        return None, reason
    endpoint, direction = entry
    segment = np.asarray(rope['observed_segment'], np.float64)
    ordered = segment[np.argsort(segment @ direction)]
    if not np.allclose(ordered[-1], endpoint):
        raise ValueError('entry orientation inconsistent')
    return ordered.tolist(), reason


def extract(output, session):
    import cv2
    from .rope_detector import detect_rope
    from .arc_detector import detect_arc
    from .geometry import rope_exclusion_band
    from .v11_path_contract import build_input, array_sha256
    output = Path(output); p = verify(output)
    target = output / 'extract' / session
    if target.exists():
        raise FileExistsError('extraction already exists; no implicit resume')
    target.mkdir(parents=True)
    cv2.setNumThreads(p['model']['cpu_threads'])
    v = p['videos'][session]; wanted = set(v['selected_indices'])
    cap = cv2.VideoCapture(v['path']); records = []
    if not cap.isOpened():
        raise ValueError('cannot open video')
    last = max(wanted); decode_accum = 0.; started = time.perf_counter()
    try:
        for frame_index in range(last + 1):
            before = time.perf_counter(); ok, frame = cap.read()
            decode_accum += (time.perf_counter() - before) * 1000
            if not ok:
                raise ValueError('video ended at ' + str(frame_index))
            if frame_index not in wanted:
                continue
            t0 = time.perf_counter(); rope = detect_rope(frame); t1 = time.perf_counter()
            exclusion_mask, exclusion = rope_exclusion_band(frame.shape, rope); t2 = time.perf_counter()
            arc = detect_arc(frame, exclusion_mask=exclusion_mask); t3 = time.perf_counter()
            seed, seed_reason = automatic_seed(rope, arc); t4 = time.perf_counter()
            row = dict(session=session, frame_index=frame_index, time_s=frame_index / v['fps'], fps=v['fps'],
                frame_sha256=array_sha256(frame), frame_shape=list(frame.shape),
                rope=rope, arc=arc, exclusion=exclusion, free_seed=seed, seed_reason=seed_reason,
                available=False, input_path=None, input_sha256=None, geometry=None,
                identity_confirmed=False, physical_contact_confirmed=False, force_command=None)
            if seed is not None:
                built = build_input(np.ascontiguousarray(frame[:, :, ::-1]), seed, rope['width_px'])
                path = target / 'inputs' / f'f{frame_index:06d}.npz'; path.parent.mkdir(exist_ok=True)
                np.savez_compressed(path, input=built['input'], valid_mask=built['valid_mask'])
                row.update(available=True, input_path=str(path), input_sha256=sha(path),
                    input_array_sha256=array_sha256(built['input']), geometry=built['geometry'])
            t5 = time.perf_counter()
            row['timing_ms'] = dict(decode_including_skipped=decode_accum, rope=(t1-t0)*1000,
                exclusion=(t2-t1)*1000, arc=(t3-t2)*1000, seed=(t4-t3)*1000,
                build_and_compress=(t5-t4)*1000, extraction_serial=decode_accum+(t5-t0)*1000)
            decode_accum = 0.
            meta = target / 'frames' / f'f{frame_index:06d}.json'; write(meta, row)
            records.append(dict(frame_index=frame_index, time_s=row['time_s'], available=row['available'],
                metadata_path=str(meta), metadata_sha256=sha(meta), timing_ms=row['timing_ms']))
            if len(records) % 20 == 0:
                print(json.dumps(dict(stage='extract', session=session, frames=len(records), total=len(wanted),
                    available=sum(r['available'] for r in records), elapsed_s=time.perf_counter()-started)), flush=True)
    finally:
        cap.release()
    verify(output, verify_videos=False)
    write(target / 'index.json', dict(session=session, protocol_sha256=sha(output / 'protocol.json'),
        video_sha256=v['sha256'], records=records, selected_indices=v['selected_indices'],
        elapsed_s=time.perf_counter()-started, hardware_access=False))
    print(json.dumps(dict(stage='extract_complete', session=session, frames=len(records))), flush=True)


def extracted(output, session):
    output = Path(output); p = read(output / 'protocol.json')
    index_path = output / 'extract' / session / 'index.json'; index = read(index_path)
    if index['protocol_sha256'] != sha(output / 'protocol.json') or index['video_sha256'] != p['videos'][session]['sha256']:
        raise ValueError('foreign extraction')
    if [r['frame_index'] for r in index['records']] != p['videos'][session]['selected_indices']:
        raise ValueError('extraction is not the complete selected chronological sequence')
    return index_path, index


def infer(output, session, arm):
    import torch
    from .v14_joint_boundary_model import build_model
    from .v11_path_model import mask_normalized_probabilities
    from .v11_path_contract import array_sha256
    output = Path(output); p = verify(output); index_path, index = extracted(output, session)
    target = output / 'infer' / session / arm
    if target.exists():
        raise FileExistsError('inference already exists')
    target.mkdir(parents=True); torch.set_num_threads(p['model']['cpu_threads'])
    checkpoint = torch.load(p['models'][arm]['path'], map_location='cpu', weights_only=True)
    if checkpoint['arm'] != arm or checkpoint['phase'] != 'train' or checkpoint['protocol_sha256'] != p['reference_protocol']['sha256']:
        raise ValueError('checkpoint provenance mismatch')
    model = build_model(p['model']['seed'], p['model']['base_channels'])
    model.load_state_dict(checkpoint['state_dict']); model.eval(); records = []; started = time.perf_counter()
    with torch.inference_mode():
        for i, r in enumerate(index['records']):
            if sha(r['metadata_path']) != r['metadata_sha256']:
                raise ValueError('extraction metadata changed')
            meta = read(r['metadata_path']); result = dict(frame_index=r['frame_index'], time_s=r['time_s'],
                available=False, prediction_path=None, prediction_sha256=None, forward_ms=None)
            if meta['available']:
                if sha(meta['input_path']) != meta['input_sha256']:
                    raise ValueError('input changed')
                with np.load(meta['input_path'], allow_pickle=False) as data:
                    x, valid = data['input'], data['valid_mask']
                if array_sha256(x) != meta['input_array_sha256']:
                    raise ValueError('input array changed')
                t0 = time.perf_counter(); out = model(torch.from_numpy(x[None]).float())
                probability = mask_normalized_probabilities(out, torch.from_numpy(valid[None]))[0].numpy()
                slope = out['slope'][0].numpy(); forward_ms = (time.perf_counter()-t0)*1000
                path = target / f'f{r["frame_index"]:06d}.npz'
                np.savez_compressed(path, probability=probability, slope=slope)
                result.update(available=True, prediction_path=str(path), prediction_sha256=sha(path), forward_ms=forward_ms)
            records.append(result)
            if (i+1) % 20 == 0:
                print(json.dumps(dict(stage='infer', session=session, arm=arm, frames=i+1,
                    total=len(index['records']), elapsed_s=time.perf_counter()-started)), flush=True)
    verify(output, verify_videos=False)
    write(target / 'index.json', dict(session=session, arm=arm, records=records,
        protocol_sha256=sha(output / 'protocol.json'), extraction_index_sha256=sha(index_path),
        checkpoint_sha256=p['models'][arm]['sha256'], torch_version=torch.__version__,
        elapsed_s=time.perf_counter()-started, hardware_access=False))


def _observed_links(segments):
    starts, ends = [], []
    for segment in segments:
        points = np.asarray(segment, float)
        if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 2:
            continue
        length = np.linalg.norm(np.diff(points, axis=0), axis=1)
        good = (length > 1e-6) & (length < max(5., float(np.median(length))*3.))
        starts.extend(points[:-1][good]); ends.extend(points[1:][good])
    return np.asarray(starts).reshape(-1, 2), np.asarray(ends).reshape(-1, 2)


def path_gap_features(decoded, arc, geometry):
    """Current direct-boundary comparison; never fill missing support with zero."""
    from .wound_rope_tracker import _comparison
    width = float(geometry['width_px']); origin = np.asarray(geometry['origin_xy']); direction = np.asarray(geometry['d_xy'])
    boundary = arc.get('local_boundary_segments', [])
    a, b = _observed_links(boundary)
    result = dict(usable=False, reason='no_direct_comparison', image_gap_proxy=None,
        optical_width_px=width, distance_quantiles_px=None, gap_proxy_quantiles=None,
        supported_samples=0, downstream_path_samples=0, support_fraction=0., contiguous_supported_span_px=0.,
        median_tangent_angle_deg=None, minimum_center_distance_px=None, geometric_proximity_candidate=False,
        identity_confirmed=False, physical_contact_confirmed=False, new_layer_confirmed=False, force_command=None,
        comparison_support_is_distance_censored=True, axial_count_measured=False)
    if not len(a):
        return result
    distance_values, angle_values, comparisons = [], [], []
    maximum_span = 0.; total = 0
    for segment in decoded.get('observed_segments', []):
        pts = np.asarray(segment, float)
        if pts.ndim != 2 or len(pts) < 3:
            continue
        # Preserve original segment continuity; do not join separate paths.
        chosen = np.flatnonzero((pts-origin) @ direction >= 0)
        for ids in np.split(chosen, np.flatnonzero(np.diff(chosen) > 1)+1):
            if len(ids) < 3:
                continue
            points = pts[ids]; total += len(points)
            comparison = _comparison(points, boundary, width); comparisons.append(comparison)
            if not comparison['valid']:
                continue
            # Same nearest-direct-link operation as _comparison, including its
            # interior-foot gate. Tangents are never formed across path gaps.
            v = b-a
            t = np.einsum('pki,ki->pk', points[:, None, :]-a[None], v)/np.sum(v*v, axis=1)
            feet = a[None]+np.clip(t, 0, 1)[..., None]*v
            distances = np.linalg.norm(points[:, None]-feet, axis=2)
            nearest = distances.argmin(axis=1); ri = np.arange(len(points))
            dist = distances[ri, nearest]
            keep = (t[ri, nearest] > 0) & (t[ri, nearest] < 1) & (dist < max(40., 3.*width))
            tangent = np.gradient(points, axis=0); norm = np.linalg.norm(tangent, axis=1)
            cosine = np.abs(np.sum(tangent*v[nearest], axis=1))/np.maximum(norm*np.linalg.norm(v[nearest], axis=1), 1e-12)
            angles = np.degrees(np.arccos(np.clip(cosine, 0, 1)))
            keep &= norm > 1e-8
            distance_values.extend(dist[keep].tolist()); angle_values.extend(angles[keep].tolist())
            supported = np.flatnonzero(keep)
            for run in np.split(supported, np.flatnonzero(np.diff(supported) > 1)+1):
                if len(run) > 1:
                    maximum_span = max(maximum_span, float(np.linalg.norm(np.diff(points[run], axis=0), axis=1).sum()))
    result.update(downstream_path_samples=total, supported_samples=len(distance_values),
        support_fraction=len(distance_values)/max(1, total), contiguous_supported_span_px=maximum_span,
        comparisons=comparisons)
    if not distance_values:
        return result
    q = np.percentile(distance_values, [0, 10, 25, 50, 75, 90, 100])
    angle = float(np.median(angle_values))
    result.update(distance_quantiles_px=q.tolist(), gap_proxy_quantiles=(q/width-.5).tolist(),
        minimum_center_distance_px=float(q[0]), median_tangent_angle_deg=angle,
        geometric_proximity_candidate=any(c.get('geometric_proximity_candidate', False) for c in comparisons))
    checks = dict(point_support=len(distance_values) >= RULES['minimum_direct_points'],
        span_support=maximum_span >= RULES['minimum_contiguous_span_widths']*width,
        tangent_agreement=angle <= RULES['maximum_median_tangent_angle_deg'])
    result['checks'] = checks
    result['usable'] = all(checks.values())
    result['reason'] = 'image_gap_proxy_only_identity_unconfirmed' if result['usable'] else 'insufficient_'+'_'.join(k for k,v in checks.items() if not v)
    if result['usable']:
        result['image_gap_proxy'] = float(q[3]/width-.5)
    return result


def measure(output, session, arm):
    from .v11_path_decode import decode_path
    output = Path(output); p = verify(output); index_path, index = extracted(output, session)
    inf_path = output / 'infer' / session / arm / 'index.json'; inference = read(inf_path)
    if inference['protocol_sha256'] != sha(output/'protocol.json') or inference['extraction_index_sha256'] != sha(index_path) or inference['checkpoint_sha256'] != p['models'][arm]['sha256']:
        raise ValueError('foreign inference')
    if [r['frame_index'] for r in inference['records']] != index['selected_indices']:
        raise ValueError('inference sequence incomplete')
    destination = output / 'measure' / session / (arm+'.json')
    if destination.exists():
        raise FileExistsError('measurement exists')
    records = []
    for i, (r, pred) in enumerate(zip(index['records'], inference['records'])):
        if sha(r['metadata_path']) != r['metadata_sha256']:
            raise ValueError('metadata changed')
        meta = read(r['metadata_path']); started = time.perf_counter()
        decoded = dict(valid=False, reason=meta['seed_reason'], observed_segments=[], identity_confirmed=False)
        features = dict(usable=False, reason=meta['seed_reason'], image_gap_proxy=None, identity_confirmed=False)
        if pred['available']:
            if sha(pred['prediction_path']) != pred['prediction_sha256'] or sha(meta['input_path']) != meta['input_sha256']:
                raise ValueError('arrays changed')
            with np.load(meta['input_path'], allow_pickle=False) as data:
                valid = data['valid_mask']
            with np.load(pred['prediction_path'], allow_pickle=False) as data:
                decoded = decode_path(data['probability'], data['slope'], meta['geometry'], valid)
            features = path_gap_features(decoded, meta['arc'], meta['geometry'])
        elapsed = (time.perf_counter()-started)*1000
        records.append(dict(frame_index=r['frame_index'], time_s=r['time_s'], decoded=decoded,
            features=features, extraction_available=meta['available'], seed_reason=meta['seed_reason'],
            forward_ms=pred['forward_ms'], measurement_ms=elapsed,
            approximate_serial_processing_ms=r['timing_ms']['extraction_serial']+(pred['forward_ms'] or 0)+elapsed,
            identity_confirmed=False, physical_event_confirmed=False, force_command=None))
        if (i+1) % 20 == 0:
            print(json.dumps(dict(stage='measure', session=session, arm=arm, frames=i+1,
                usable=sum(x['features']['usable'] for x in records))), flush=True)
    verify(output, verify_videos=False)
    write(destination, dict(task=p['task'], session=session, arm=arm, records=records,
        protocol_sha256=sha(output/'protocol.json'), extraction_index_sha256=sha(index_path),
        inference_index_sha256=sha(inf_path), hardware_access=False, deployment_allowed=False,
        physical_event_timing_validated=False, ground_truth_used=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'verify', 'extract', 'infer', 'measure'))
    parser.add_argument('--output', type=Path, default=BASE)
    parser.add_argument('--session', choices=tuple(SESSIONS), default='left')
    parser.add_argument('--arm', choices=ARMS, default='center_only')
    args = parser.parse_args()
    if args.stage == 'freeze': freeze(args.output)
    elif args.stage == 'verify': verify(args.output); print('V16 frozen inputs verified')
    elif args.stage == 'extract': extract(args.output, args.session)
    else: globals()[args.stage](args.output, args.session, args.arm)


if __name__ == '__main__':
    main()
