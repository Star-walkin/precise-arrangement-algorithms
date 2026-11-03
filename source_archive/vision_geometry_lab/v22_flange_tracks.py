"""Development-only natural flange texture tracking, never a mechanical encoder.

All geometric thresholds are in the 2048-wide analysis image. A homography with
rotation-like eigenvalues need not have been produced by a rotating rigid plane.
No angle integration, event labels, recurrence period, or device access is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

LAB = Path(__file__).resolve().parent
BASE = LAB / 'outputs/v22_flange_tracks'
CONTROL = LAB / 'outputs/v20_material_audit/appearance_control/protocol.json'
RULES = dict(analysis_width=2048, max_corners_per_region=100,
    corner_quality=.01, corner_min_distance_px=4., lk_window=21, lk_max_level=3,
    fb_max_px=1., lk_error_max=30., maximum_step_s=.15,
    yellow_hue_min=10, yellow_hue_max=45, yellow_saturation_min=40,
    minimum_value=15, maximum_value=245, mask_erode_size=3,
    background_top_fraction=.4, background_min_tracks=12,
    background_max_median_displacement_px=1.5,
    homography_ransac_px=1.5, minimum_inliers=16,
    minimum_inliers_per_roi=4, minimum_inlier_fraction=.7,
    maximum_symmetric_error_p90_px=1.5,
    minimum_hull_roi_area_fraction=.15, minimum_vertical_span_px=6.,
    maximum_homography_condition=1e8,
    spectral_modulus_relative_tolerance=.05, spectral_imaginary_min=1e-5,
    spectral_identity_assumption_verified=False, mechanical_angle_available=False,
    phase_accumulation=False, hardware_access=False,
    first_frame_ROI_only=True, roi_updates=False, threshold_search=False,
    scope='development_planar_image_motion_observability_not_material_or_mechanical_phase')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def pixels_sha(a):
    h = hashlib.sha256(repr((a.shape, str(a.dtype))).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(clean(value), f, ensure_ascii=False, indent=2, allow_nan=False)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def spectral_diagnostic(H):
    """Projective spectrum only: even pure nonrigid shear can pass this check."""
    out = dict(available=False, reason='not_rotation_like_spectrum',
               projective_angle_magnitude_deg=None, mechanical_angle_available=False,
               signed_angle_available=False)
    if H is None or np.asarray(H).shape != (3, 3) or not np.isfinite(H).all():
        return {**out, 'reason': 'invalid_homography'}
    e = np.linalg.eigvals(H)
    out['eigenvalues_real_imag'] = [[float(x.real), float(x.imag)] for x in e]
    real = np.flatnonzero(np.abs(e.imag) < RULES['spectral_imaginary_min'])
    complex_positive = np.flatnonzero(e.imag > RULES['spectral_imaginary_min'])
    if len(real) != 1 or len(complex_positive) != 1 or abs(e[real[0]]) < 1e-12:
        return out
    z = e[complex_positive[0]]/e[real[0]]
    error = abs(abs(z)-1.)
    out['relative_modulus_error'] = float(error)
    if error > RULES['spectral_modulus_relative_tolerance']:
        return out
    return {**out, 'available': True,
        'reason': 'spectrum_only_not_rigid_plane_or_axis_confirmation',
        'projective_angle_magnitude_deg': float(abs(np.degrees(np.angle(z))))}


def _project(H, points):
    return cv2.perspectiveTransform(np.asarray(points, np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)


def _flow_pair(previous, current, points, settings):
    """OpenCV may return None/empty outputs on failure; never feed them back in."""
    n = len(points)
    try:
        forward, status1, error = cv2.calcOpticalFlowPyrLK(previous, current, points[:, None], None, **settings)
        if any(x is None for x in (forward, status1, error)):
            return None
        if np.asarray(forward).shape != (n, 1, 2) or np.asarray(status1).size != n or np.asarray(error).size != n:
            return None
        backward, status2, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None, **settings)
        if backward is None or status2 is None or np.asarray(backward).shape != (n, 1, 2) or np.asarray(status2).size != n:
            return None
        return forward, status1, error, backward, status2
    except cv2.error:
        return None


def homography_evidence(previous, current, regions, rois):
    p = np.asarray(previous, np.float64).reshape(-1, 2)
    q = np.asarray(current, np.float64).reshape(-1, 2)
    g = np.asarray(regions, int)
    out = dict(available=False, first_rejection=None, failed_reasons=[],
        tracked_points=len(p), inlier_count=0, inlier_fraction=None,
        inliers_per_roi=[0, 0], symmetric_error_p90_px=None,
        hull_roi_area_fractions=[None, None], vertical_span_px=None,
        homography_condition=None, H_previous_to_current=None,
        spectral=spectral_diagnostic(None), mechanical_angle_available=False)
    if p.shape != q.shape or g.shape != (len(p),):
        raise ValueError('correspondence shapes do not match')
    if len(p) < 4 or not (np.isfinite(p).all() and np.isfinite(q).all()):
        out['failed_reasons'] = ['insufficient_finite_correspondences']
        out['first_rejection'] = out['failed_reasons'][0]
        return out, np.zeros(len(p), bool)
    cv2.setRNGSeed(2201)
    H, mask = cv2.findHomography(p, q, cv2.RANSAC, RULES['homography_ransac_px'],
                                maxIters=2000, confidence=.995)
    if H is None or mask is None or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        out['failed_reasons'] = ['homography_not_invertible']
        out['first_rejection'] = out['failed_reasons'][0]
        return out, np.zeros(len(p), bool)
    mask = mask.ravel().astype(bool)
    count = int(mask.sum())
    condition = float(np.linalg.cond(H))
    errors = np.maximum(np.linalg.norm(_project(H, p)-q, axis=1),
                        np.linalg.norm(_project(np.linalg.inv(H), q)-p, axis=1))
    p90 = float(np.quantile(errors[mask], .9)) if count else np.inf
    coverage = []
    counts = []
    for r, box in enumerate(rois):
        use = mask & (g == r)
        counts.append(int(use.sum()))
        area = (box[2]-box[0])*(box[3]-box[1])
        hull_area = min(cv2.contourArea(cv2.convexHull(p[use].astype(np.float32))),
                        cv2.contourArea(cv2.convexHull(q[use].astype(np.float32)))) if use.sum() >= 3 else 0.
        coverage.append(float(hull_area/area))
    span = float(np.ptp(p[mask, 1])) if count else 0.
    failed = []
    if count < RULES['minimum_inliers']: failed.append('too_few_inliers')
    if min(counts) < RULES['minimum_inliers_per_roi']: failed.append('insufficient_each_roi_support')
    if count/max(1, len(p)) < RULES['minimum_inlier_fraction']: failed.append('low_inlier_fraction')
    if p90 > RULES['maximum_symmetric_error_p90_px']: failed.append('large_symmetric_error')
    if min(coverage) < RULES['minimum_hull_roi_area_fraction']: failed.append('narrow_spatial_support')
    if span < RULES['minimum_vertical_span_px']: failed.append('degenerate_vertical_span')
    if condition > RULES['maximum_homography_condition']: failed.append('ill_conditioned_homography')
    return {**out, 'available': not failed, 'first_rejection': failed[0] if failed else None,
        'failed_reasons': failed, 'inlier_count': count, 'inlier_fraction': count/len(p),
        'inliers_per_roi': counts, 'symmetric_error_p90_px': p90,
        'hull_roi_area_fractions': coverage, 'vertical_span_px': span,
        'homography_condition': condition, 'H_previous_to_current': H.tolist(),
        'spectral': spectral_diagnostic(H)}, mask


class FlangeTracks:
    def __init__(self, rois, source_size):
        width, height = source_size
        if width <= 0 or height <= 0:
            raise ValueError('invalid source size')
        self.source_size = (width, height)
        self.scale = min(1., RULES['analysis_width']/width)
        self.size = (round(width*self.scale), round(height*self.scale))
        self.rois = np.rint(np.asarray(rois, float)*self.scale).astype(int)
        if self.rois.shape != (2, 4):
            raise ValueError('two fixed ROIs required')
        for x1, y1, x2, y2 in self.rois:
            if not (0 <= x1 < x2 <= self.size[0] and 0 <= y1 < y2 <= self.size[1]):
                raise ValueError('ROI outside analysis image')
        self.gray = None
        self.last_time = None
        self.points = np.empty((0, 2), np.float32)
        self.regions = np.empty(0, int)
        self.ids = np.empty(0, int)
        self.born = np.empty(0, float)
        self.next_id = 0
        self.chain_id = 0
        self.previous_valid = False

    def _masks(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        yellow = ((hsv[:, :, 0] >= RULES['yellow_hue_min']) &
            (hsv[:, :, 0] <= RULES['yellow_hue_max']) &
            (hsv[:, :, 1] >= RULES['yellow_saturation_min']) &
            (hsv[:, :, 2] >= RULES['minimum_value']) &
            (hsv[:, :, 2] < RULES['maximum_value'])).astype(np.uint8)*255
        yellow = cv2.erode(yellow, np.ones((RULES['mask_erode_size'],)*2, np.uint8))
        masks = []
        for x1, y1, x2, y2 in self.rois:
            mask = np.zeros(self.size[::-1], np.uint8)
            mask[y1:y2, x1:x2] = yellow[y1:y2, x1:x2]
            masks.append(mask)
        background = np.zeros_like(yellow)
        background[:int(self.size[1]*RULES['background_top_fraction'])] = 255
        masks.append(background)
        return masks

    def _seed(self, gray, masks, now):
        additions = [0, 0, 0]
        for region, mask in enumerate(masks):
            remaining = RULES['max_corners_per_region']-int(np.sum(self.regions == region))
            if remaining <= 0:
                continue
            masked = mask.copy()
            for point in self.points[self.regions == region]:
                cv2.circle(masked, tuple(np.rint(point).astype(int)),
                           int(RULES['corner_min_distance_px']), 0, -1)
            if region < 2:
                x1, y1, x2, y2 = self.rois[region]
            else:
                x1, y1, x2, y2 = 0, 0, self.size[0], int(self.size[1]*RULES['background_top_fraction'])
            points = cv2.goodFeaturesToTrack(gray[y1:y2, x1:x2], maxCorners=remaining,
                qualityLevel=RULES['corner_quality'], minDistance=RULES['corner_min_distance_px'],
                mask=masked[y1:y2, x1:x2], blockSize=3)
            if points is None:
                continue
            points = points.reshape(-1, 2)+np.array([x1, y1], np.float32)
            n = len(points)
            additions[region] = n
            self.points = np.vstack((self.points, points))
            self.regions = np.r_[self.regions, np.full(n, region, int)]
            self.ids = np.r_[self.ids, np.arange(self.next_id, self.next_id+n)]
            self.born = np.r_[self.born, np.full(n, now)]
            self.next_id += n
        return additions

    def update(self, frame, time_s):
        started = time.perf_counter()
        now = float(time_s)
        if not np.isfinite(now) or now < 0:
            raise ValueError('finite nonnegative frame time required')
        if self.last_time is not None and now <= self.last_time:
            raise ValueError('strictly increasing frame times required')
        if frame.dtype != np.uint8 or frame.shape != (self.source_size[1], self.source_size[0], 3):
            raise ValueError('source BGR uint8 dimensions changed')
        image = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        masks = self._masks(image)
        previous_count = [int(np.sum(self.regions == r)) for r in range(3)]
        geometry, inlier = homography_evidence([], [], [], self.rois)
        result = dict(time_s=now, previous_time_s=self.last_time,
            candidate_kind='planar_image_motion_candidate', candidate_available=False,
            mechanical_angle_available=False, phase_accumulated=False,
            physical_layer_confirmed=False, event_output=False, hardware_access=False,
            first_rejection=None, failed_reasons=[],
            prior_track_count_by_region=previous_count,
            fb_accepted_count_by_region=[0, 0, 0], background_median_displacement_px=None,
            fb_median_px=None, fb_p90_px=None,
            roi_mask_fraction=[float(m[y1:y2, x1:x2].mean()/255.) for m, (x1,y1,x2,y2) in zip(masks, self.rois)],
            geometry=geometry, native_track_ids=[], native_track_age_s=[],
            continued_fraction_of_prior_flange_tracks=0.)
        reset = self.gray is None or now-self.last_time > RULES['maximum_step_s']+1e-9
        if reset:
            result['failed_reasons'] = ['startup' if self.gray is None else 'frame_gap']
            self.points = np.empty((0, 2), np.float32)
            self.regions = np.empty(0, int); self.ids = np.empty(0, int); self.born = np.empty(0)
        elif not len(self.points):
            result['failed_reasons'] = ['no_previous_tracks']
        else:
            settings = dict(winSize=(RULES['lk_window'],)*2, maxLevel=RULES['lk_max_level'],
                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, .01))
            flow = _flow_pair(self.gray, gray, self.points, settings)
            if flow is None:
                result['failed_reasons'].append('lk_output_unavailable')
                n = len(self.points)
                next_points = np.full((n, 1, 2), np.nan, np.float32)
                back = np.full((n, 1, 2), np.nan, np.float32)
                status1 = status2 = np.zeros(n, np.uint8)
                errors = np.full(n, np.inf)
            else:
                next_points, status1, errors, back, status2 = flow
            q = next_points.reshape(-1, 2); back = back.reshape(-1, 2)
            fb = np.linalg.norm(back-self.points, axis=1)
            finite = np.isfinite(q).all(axis=1) & np.isfinite(fb)
            good = status1.ravel().astype(bool) & status2.ravel().astype(bool) & finite
            good &= (fb <= RULES['fb_max_px']) & (errors.ravel() <= RULES['lk_error_max'])
            pixel = np.rint(np.nan_to_num(q, nan=-1., posinf=-1., neginf=-1.)).astype(int)
            inside = (pixel[:, 0] >= 0)&(pixel[:, 0] < self.size[0])&(pixel[:, 1] >= 0)&(pixel[:, 1] < self.size[1])
            for i in np.flatnonzero(good & inside):
                if masks[self.regions[i]][pixel[i, 1], pixel[i, 0]] == 0:
                    good[i] = False
            good &= inside
            counts = [int(np.sum(good & (self.regions == r))) for r in range(3)]
            result['fb_accepted_count_by_region'] = counts
            if finite.any():
                result['fb_median_px'] = float(np.median(fb[finite]))
                result['fb_p90_px'] = float(np.quantile(fb[finite], .9))
            bg = good & (self.regions == 2)
            if bg.sum() < RULES['background_min_tracks']:
                result['failed_reasons'].append('insufficient_background_tracks')
            else:
                displacement = float(np.median(np.linalg.norm(q[bg]-self.points[bg], axis=1)))
                result['background_median_displacement_px'] = displacement
                if displacement > RULES['background_max_median_displacement_px']:
                    result['failed_reasons'].append('background_image_motion')
            flange = good & (self.regions < 2)
            geometry, inlier = homography_evidence(self.points[flange], q[flange], self.regions[flange], self.rois)
            result['geometry'] = geometry
            result['failed_reasons'] += geometry['failed_reasons']
            result['candidate_available'] = not result['failed_reasons']
            keep = bg.copy()
            if result['candidate_available']:
                keep[np.flatnonzero(flange)[inlier]] = True
            # A rejected pair cannot extend a material-track history.
            flange_keep = keep & (self.regions < 2)
            result['native_track_ids'] = self.ids[flange_keep].tolist()
            result['native_track_age_s'] = (now-self.born[flange_keep]).tolist()
            result['continued_fraction_of_prior_flange_tracks'] = float(flange_keep.sum()/max(1, sum(previous_count[:2])))
            result['tracked_previous_xy'] = self.points[flange].tolist()
            result['tracked_current_xy'] = q[flange].tolist()
            result['tracked_regions'] = self.regions[flange].tolist()
            result['homography_inlier_mask'] = inlier.tolist()
            self.points = q[keep]; self.regions = self.regions[keep]
            self.ids = self.ids[keep]; self.born = self.born[keep]
        if result['candidate_available'] and not self.previous_valid:
            self.chain_id += 1
        result['image_motion_chain_id'] = self.chain_id if result['candidate_available'] else None
        result['first_rejection'] = result['failed_reasons'][0] if result['failed_reasons'] else None
        result['new_corner_count_by_region'] = self._seed(gray, masks, now)
        result['active_count_by_region'] = [int(np.sum(self.regions == r)) for r in range(3)]
        self.gray = gray; self.last_time = now; self.previous_valid = result['candidate_available']
        result['processing_ms_excluding_decode'] = 1000*(time.perf_counter()-started)
        return result


def freeze(base=BASE, control=CONTROL):
    base, control = Path(base), Path(control)
    if base.exists():
        raise FileExistsError('fresh V22 output required')
    p = read(control)
    if p.get('schema') != 'v20_appearance_control_protocol_1':
        raise ValueError('wrong source protocol')
    videos = p['videos']
    source_indices = {}
    for side in ('left', 'right'):
        v = videos[side]
        if not np.isfinite(v['fps']) or v['fps'] != 10. or v['frame_count'] != 601:
            raise ValueError('this fixed trial requires complete 0..60s at 10fps')
        if v['indices'] != list(range(601)):
            raise ValueError('source frame index inventory changed')
        FlangeTracks(p['rois'][side], (v['width'], v['height']))
        if sha(v['path']) != v['sha256']:
            raise ValueError('video hash changed')
        index_path = Path(p['upstream_base'])/side/'index.json'
        index = read(index_path)
        if index['video'] != v or len(index['records']) != 601:
            raise ValueError('source frame digest index mismatch')
        for i, record in enumerate(index['records']):
            if record['frame_index'] != i or record['time_s'] != i/v['fps'] or len(record['frame_sha256']) != 64:
                raise ValueError('invalid source frame hash timeline')
        source_indices[side] = dict(path=str(index_path), sha256=sha(index_path))
    protocol = dict(schema='v22_flange_tracks_protocol_1', rules=RULES,
        frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        control_path=str(control), control_sha256=sha(control),
        source_sha256=sha(__file__), videos=videos, rois=p['rois'],
        source_indices=source_indices,
        synthetic_test_sha256=sha(LAB/'tests/test_v22_flange_tracks.py'),
        runtime=dict(python=sys.version, opencv=cv2.__version__, numpy=np.__version__,opencv_threads=2),
        mirroring={'left': True, 'right': False},
        stop_rule='One complete replay; no threshold or ROI adjustment. No mechanical angle, cycle or layer claim regardless of fit.',
        prior_development_images_seen='two previously selected V20 first-frame ROIs only',
        no_truth_inputs=True, analysis_coordinates='2048_width; original pixels divide by analysis/source scale')
    write(base/'protocol.json', protocol)
    return protocol


def verify(base=BASE):
    p = read(Path(base)/'protocol.json')
    if p.get('schema') != 'v22_flange_tracks_protocol_1' or p['rules'] != RULES:
        raise ValueError('frozen protocol changed')
    if sha(__file__) != p['source_sha256'] or sha(p['control_path']) != p['control_sha256']:
        raise ValueError('frozen source changed')
    for index in p['source_indices'].values():
        if sha(index['path']) != index['sha256']:
            raise ValueError('source frame digest index changed')
    return p


def summarize(records):
    reasons = Counter(reason for r in records for reason in r['failed_reasons'])
    chain_counts = Counter(r['image_motion_chain_id'] for r in records if r['candidate_available'])
    ages = [age for r in records for age in r['native_track_age_s']]
    bins = []
    for start in range(0, 60, 10):
        rows = [r for r in records if start <= r['time_s'] and (r['time_s'] < start+10 or (start == 50 and r['time_s'] == 60))]
        accepted = [r for r in rows if r['candidate_available']]
        bins.append(dict(start_s=start, end_s=start+10, frames=len(rows),
            candidate_pairs=len(accepted), fraction=len(accepted)/len(rows) if rows else None,
            median_continued_fraction=float(np.median([r['continued_fraction_of_prior_flange_tracks'] for r in rows])) if rows else None,
            max_native_age_s=max((age for r in rows for age in r['native_track_age_s']), default=0.)))
    timings = [r['processing_ms_excluding_decode'] for r in records]
    return dict(frame_count=len(records), candidate_pair_count=sum(r['candidate_available'] for r in records),
        denominator_adjacent_pairs=max(0,len(records)-1), failed_reasons=dict(reasons),
        image_motion_chain_pair_counts=dict(chain_counts), longest_chain_pairs=max(chain_counts.values(), default=0),
        maximum_native_track_age_s=max(ages, default=0.),
        native_track_observation_age_p50_s=float(np.median(ages)) if ages else None,
        native_track_observation_age_p90_s=float(np.quantile(ages,.9)) if ages else None,
        processing_ms_p50=float(np.median(timings)), processing_ms_p90=float(np.quantile(timings,.9)),
        fixed_ten_second_bins=bins, mechanical_angle_available=False, phase_accumulated=False,
        warning='Adjacent image links with reseeding are not one same-texture full-turn trajectory; spectrum can be nonrigid shear.')


def run(base=BASE, side='left'):
    base = Path(base); p = verify(base)
    out = base/side
    if out.exists(): raise FileExistsError('no replay overwrite')
    v = p['videos'][side]
    source_records = read(p['source_indices'][side]['path'])['records']
    if sha(v['path']) != v['sha256']: raise ValueError('source video changed')
    cap = cv2.VideoCapture(v['path'])
    if not cap.isOpened() or not np.isfinite(cap.get(cv2.CAP_PROP_FPS)) or cap.get(cv2.CAP_PROP_FPS) != v['fps']:
        raise ValueError('source video FPS/open failed')
    tracker = FlangeTracks(p['rois'][side], (v['width'], v['height']))
    write(out/'started.json', dict(status='started', protocol_sha256=sha(base/'protocol.json')))
    records = []
    cv2.setNumThreads(2)
    for i in range(601):
        full_start = time.perf_counter()
        ok, raw = cap.read()
        if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES))-1 != i:
            raise ValueError('sequential source read failed')
        digest = pixels_sha(raw)
        if digest != source_records[i]['frame_sha256']:
            raise ValueError('decoded frame pixels differ from archived source frame')
        canonical = cv2.flip(raw, 1) if p['mirroring'][side] else raw
        record = tracker.update(canonical, i/v['fps'])
        record.update(frame_index=i, source_frame_sha256=digest,
            serial_ms_including_decode_and_hash=1000*(time.perf_counter()-full_start))
        records.append(record)
        if i % 50 == 0:
            print(side, i, '/', 600, 'accepted', sum(r['candidate_available'] for r in records), flush=True)
    cap.release()
    result = dict(status='complete', protocol_sha256=sha(base/'protocol.json'),
        source_video=v, side=side, records=records, summary=summarize(records),
        mechanical_angle_available=False, physical_layer_confirmed=False,
        phase_accumulated=False, truth_inputs=False, hardware_access=False)
    write(out/'tracks.json', result)
    return result['summary']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze','verify','run'])
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--side', choices=['left','right'], default='left')
    args = parser.parse_args()
    output = freeze(args.base) if args.stage == 'freeze' else verify(args.base) if args.stage == 'verify' else run(args.base,args.side)
    print(json.dumps(clean(output),ensure_ascii=False,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
