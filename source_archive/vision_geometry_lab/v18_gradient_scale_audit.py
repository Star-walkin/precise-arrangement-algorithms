"""Posthoc derivative-unit audit using only left startup 0..5 seconds.

This does not rerun the frozen trial or generate events. The normalized-unit
counterfactual is a diagnostic, not a new accepted measurement or tuning result.
"""
from pathlib import Path
import json

import cv2
import numpy as np

from vision_geometry_lab.v18_front_trial import (
    BASE, canonical_frame, pixels_sha, read, sha, align_x_to_reference,
)
from vision_geometry_lab.v18_front_observability import build_fixed_baseline


DEST = BASE.parent / 'v18_gradient_scale_audit'


def quantiles(values):
    arr = np.asarray(values, float).ravel()
    arr = arr[np.isfinite(arr)]
    return None if not len(arr) else dict(zip(
        ('min', 'p10', 'median', 'p90', 'max'),
        map(float, np.quantile(arr, [0., .1, .5, .9, 1.]))))


def profile_pair(gray, y, seed):
    """Exactly V18 intensity profile, plus old local Sobel without border fill."""
    xa = max(2, int(round(seed-65)))
    xb = min(gray.shape[1]-2, int(round(seed+66)))
    ya, yb = max(0, int(y)-3), min(gray.shape[0], int(y)+4)
    intensity = gray[ya:yb, xa-1:xb+1].mean(axis=0)
    gradient = np.gradient(intensity)[1:-1]
    # Extra row/column halo makes this identical to the full-frame Sobel at
    # these interior pixels; none of the strip padding is part of the result.
    sobel = cv2.Sobel(gray[ya-1:yb+1, xa-1:xb+1], cv2.CV_32F,
                      1, 0, ksize=3)[1:-1, 1:-1]
    old = np.abs(sobel).mean(axis=0).astype(float)
    signed_mean = sobel.mean(axis=0).astype(float)/8.
    xs = np.arange(xa, xb, dtype=float)
    weights = np.exp(-.5*((xs-seed)/28.)**2)
    nk = int(np.argmax(np.abs(gradient)*weights))
    sk = int(np.argmax(old*weights))
    return dict(xs=xs, intensity=intensity[1:-1], gradient=gradient,
                sobel_abs_mean=old, sobel_signed_mean_normalized=signed_mean,
                np_selected_index=nk, sobel_selected_index=sk)


def synthetic_evidence():
    out = {}
    for name, signal in (
        ('linear_ramp_1_gray_per_pixel', np.arange(131, dtype=np.float32)),
        ('step_10_gray', (np.arange(131)>=65).astype(np.float32)*10.),
    ):
        image = np.tile(signal, (9, 1))
        sobel = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)[4, 1:-1]
        gradient = np.gradient(signal)[1:-1]
        valid = np.abs(gradient)>1e-6
        out[name] = dict(gradient_max=float(np.max(np.abs(gradient))),
                        sobel_max=float(np.max(np.abs(sobel))),
                        ratio=quantiles(sobel[valid]/gradient[valid]),
                        same_numeric_threshold_8_accepts_np=bool(np.max(abs(gradient))>=8.),
                        same_numeric_threshold_8_accepts_sobel=bool(np.max(abs(sobel))>=8.))
    return out


def main():
    if DEST.exists():
        raise FileExistsError('Do not overwrite an earlier diagnostic')
    protocol = read(BASE/'protocol.json')
    video = protocol['videos']['left']
    # Metadata is loaded for integrity, but only the 0..5 s records are used.
    index = read(BASE/'left/index.json')
    if sha(BASE/'left/measurement.npz') != index['measurement_sha256']:
        raise ValueError('frozen measurement changed')
    if sha(BASE/'protocol.json') != index['protocol_sha256']:
        raise ValueError('frozen protocol changed')
    for path, digest in protocol['source_hashes'].items():
        if sha(path) != digest:
            raise ValueError('frozen source changed')
    with np.load(BASE/'left/measurement.npz') as z:
        times_all = z['times_s']
        keep = np.flatnonzero(times_all <= 5.)
        times = times_all[keep]
        rows = z['rows_y']
        seeds = z['semantic_seed_x_current'][keep]
        poses = z['transforms'][keep]
        pose_valid = z['pose_valid'][keep]
        frozen = z['semantic_x_current'][keep]
    assert np.array_equal(keep, np.arange(51)) and video['fps'] == 10.
    nrows = len(rows)
    fields = {key: np.full((len(keep), nrows), np.nan) for key in (
        'np_max', 'np_selected', 'old_max', 'old_selected', 'old_x',
        'np_x', 'same_point_old_over_np', 'signed_mean_normalized_delta_p90')}
    alternative = {key: np.full_like(seeds, np.nan) for key in (
        'v18_gradient_8', 'gradient_1_unit_counterfactual',
        'old_sobel_8_original_gate', 'old_sobel_8_selected_gate')}
    examples = []
    frame0 = None
    hashes = []
    cap = cv2.VideoCapture(video['path'])
    cv2.setNumThreads(2)
    try:
        for i in keep:
            ok, raw = cap.read()
            if not ok or int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))-1 != i:
                raise ValueError('startup decode index mismatch')
            digest = pixels_sha(raw)
            if digest != index['records'][int(i)]['frame_sha256']:
                raise ValueError('startup pixels differ from frozen extraction')
            hashes.append(dict(frame_index=int(i), time_s=float(times[i]), sha256=digest))
            frame = canonical_frame(raw, 'left')
            if i == 0:
                frame0 = frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for j, y in enumerate(rows):
                seed = seeds[i, j]
                if not np.isfinite(seed):
                    continue
                p = profile_pair(gray, y, seed)
                g, old = np.abs(p['gradient']), p['sobel_abs_mean']
                nk, sk = p['np_selected_index'], p['sobel_selected_index']
                vals = dict(np_max=g.max(), np_selected=g[nk], old_max=old.max(),
                            old_selected=old[sk], old_x=p['xs'][sk], np_x=p['xs'][nk],
                            same_point_old_over_np=old[nk]/max(g[nk], 1e-12),
                            signed_mean_normalized_delta_p90=np.quantile(abs(
                                p['sobel_signed_mean_normalized']-p['gradient']), .9))
                for name, value in vals.items():
                    fields[name][i, j] = value
                if g.max()>=8. and g[nk]>=8.:
                    alternative['v18_gradient_8'][i, j] = p['xs'][nk]
                if g.max()>=1. and g[nk]>=1.:
                    alternative['gradient_1_unit_counterfactual'][i, j] = p['xs'][nk]
                if old.max()>=8.:
                    alternative['old_sobel_8_original_gate'][i, j] = p['xs'][sk]
                if old.max()>=8. and old[sk]>=8.:
                    alternative['old_sobel_8_selected_gate'][i, j] = p['xs'][sk]
                if i == 0 and j % 8 == 0:
                    examples.append(dict(frame_index=0, y=float(y), seed_x=float(seed),
                        **{key: float(value) for key, value in vals.items()},
                        **{key: np.asarray(p[key]).tolist() for key in (
                            'xs', 'intensity', 'gradient', 'sobel_abs_mean',
                            'sobel_signed_mean_normalized')}))
            if i % 10 == 0:
                print('startup audit frame', i, flush=True)
    finally:
        cap.release()
    np.testing.assert_allclose(alternative['v18_gradient_8'], frozen,
                               equal_nan=True, atol=0.)
    comparison = {}
    for name, values in alternative.items():
        aligned = np.full_like(values, np.nan)
        for i in keep:
            if pose_valid[i]:
                aligned[i] = align_x_to_reference(values[i], rows, poses[i], rows)
        b = build_fixed_baseline(times, rows, aligned)
        comparison[name] = dict(raw_finite_candidates=int(np.isfinite(values).sum()),
            frame0_finite_candidates=int(np.isfinite(values[0]).sum()),
            direct_rows_with_20_frames=int((np.isfinite(values).sum(axis=0)>=20).sum()),
            aligned_rows_with_20_frames=int(b['location_supported'].sum()),
            normal_supported_baseline_rows=int(b['supported'].sum()),
            candidate_location_not_material_identity=True)
    stats = {key: quantiles(values) for key, values in fields.items()}
    result = dict(schema='v18_startup_derivative_unit_audit_1',
        selection='left only; frame indices 0..50 inclusive, 0..5 s startup; no event times',
        protocol_sha256=sha(BASE/'protocol.json'), script_sha256=sha(__file__),
        input_measurement_sha256=index['measurement_sha256'], frame_hashes=hashes,
        hardware_access=False, labels_used=False, main_trial_rerun=False,
        semantic_inference_rerun=False, heldout_0925_read=False,
        synthetic=synthetic_evidence(), semantic_seed_candidates=int(np.isfinite(seeds).sum()),
        possible_row_observations=int(seeds.size), startup_pose_valid=int(pose_valid.sum()),
        gradient_summary=stats, unit_counterfactuals=comparison,
        caveats=[
            'Exactly 8x for a vertically constant profile; real Sobel also smooths vertically.',
            'Old mean(abs(Sobel)) and abs(gradient(mean(I))) differ in sign cancellation.',
            'More candidates do not establish correct material contour or event detectability.',
            'Counterfactual threshold 1 is a derivative-unit comparison, not accepted retuning.',
            'Frozen V18 outputs remain unchanged; source/core hashes verified before audit.'],
        frame0_profiles=examples)
    DEST.mkdir()
    (DEST/'startup_scale_audit.json').write_text(json.dumps(result, ensure_ascii=False,
        indent=2, allow_nan=False), encoding='utf-8')
    image = frame0.copy()
    for j, y in enumerate(rows):
        for name, color, radius in (
            ('old_sobel_8_original_gate', (0, 255, 0), 4),
            ('gradient_1_unit_counterfactual', (255, 255, 0), 3),
            ('v18_gradient_8', (0, 0, 255), 6)):
            x = alternative[name][0, j]
            if np.isfinite(x):
                cv2.circle(image, (round(x), int(y)), radius, color, -1)
    xfinite = alternative['old_sobel_8_original_gate'][0]
    xfinite = xfinite[np.isfinite(xfinite)]
    x1, x2 = max(0, int(xfinite.min())-100), min(image.shape[1], int(xfinite.max())+150)
    crop = image[max(0, int(rows[0])-60):min(image.shape[0], int(rows[-1])+60), x1:x2]
    ok, encoded = cv2.imencode('.png', crop)
    if not ok:
        raise ValueError('PNG encoding failed')
    encoded.tofile(DEST/'left_startup_0000_derivative_candidates.png')
    print(json.dumps(dict(gradient_summary=stats, unit_counterfactuals=comparison,
        output=str(DEST/'startup_scale_audit.json')), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
