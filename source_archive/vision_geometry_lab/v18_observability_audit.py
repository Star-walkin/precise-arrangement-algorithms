"""Read-only posthoc observability summary of the V18b saved measurements.

Fixed ten-second bins; no truth/event files, video frames, fitting or thresholds
are used to infer winding events. All ratios describe image candidates only.
"""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from vision_geometry_lab.v18_front_trial import read, sha


BASE = Path(__file__).resolve().parent / 'outputs/v18_gradient_units'
DEST = BASE.parent / 'v18_observability_audit'


def stats(values):
    a = np.asarray(values, float).ravel()
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(n=0, min=None, p10=None, median=None, p90=None, max=None)
    return dict(n=int(len(a)), **dict(zip(('min', 'p10', 'median', 'p90', 'max'),
        map(float, np.quantile(a, [0., .1, .5, .9, 1.])))))


def fraction(numerator, denominator):
    return None if not denominator else float(numerator/denominator)


def medians(a, axis):
    if axis == 0:
        return np.asarray([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan
                           for v in a.T])
    return medians(a.T, 0)


def contiguous_ranges(rows, supported):
    indices = np.flatnonzero(supported)
    chunks = np.split(indices, np.flatnonzero(np.diff(indices)>1)+1)
    return [dict(first_row_y=int(rows[c[0]]), last_row_y=int(rows[c[-1]]), rows=int(len(c)))
            for c in chunks if len(c)]


def summarize(side, protocol, protocol_digest):
    index_path = BASE/side/'index.json'
    measure_path = BASE/side/'measurement.npz'
    index = read(index_path)
    digest = sha(measure_path)
    if index['measurement_sha256'] != digest or index['protocol_sha256'] != protocol_digest:
        raise ValueError('measurement/protocol provenance mismatch')
    if index['output_status'] != 'complete_observations_only_no_events':
        raise ValueError('not a complete observation file')
    with np.load(measure_path) as z:
        arrays = {key: z[key].copy() for key in z.files if key != 'normal_profiles'}
    t, rows = arrays['times_s'], arrays['rows_y']
    if not np.array_equal(t, np.arange(601)/10.):
        raise ValueError('unexpected archived time coordinates')
    if any(r['frame_index']!=i or abs(r['time_s']-t[i])>1e-9
           for i, r in enumerate(index['records'])):
        raise ValueError('record coordinate mismatch')
    base_ok = np.isfinite(arrays['baseline_normal_offset'])
    pose = arrays['pose_valid']
    normal = arrays['normal_offset']
    resid = arrays['normal_residual']
    peak = arrays['normal_peak_ratio']
    grad = arrays['normal_gradient']
    clips = arrays['normal_clipped_fraction']
    warm = t <= protocol['rules']['startup_ready_s']
    baseline_polarity = np.sign(medians(grad[warm], 0))
    grad_ok = np.isfinite(grad) & (np.abs(grad)>0.)
    comparable_polarity = grad_ok & np.isfinite(baseline_polarity)[None, :] & (baseline_polarity[None, :]!=0)
    changed_polarity = comparable_polarity & (np.sign(grad)!=baseline_polarity[None, :])
    good_resid = np.isfinite(resid)
    ambiguous = np.isfinite(peak) & (peak < protocol['rules']['peak_ambiguity_diagnostic_ratio'])
    def one_window(select, label):
        count = int(select.sum())
        finite = np.isfinite(normal[select])
        supported_grid = np.broadcast_to(base_ok, (count, len(rows)))
        valid_resid = good_resid[select]
        perframe_med = medians(resid[select], 1)
        perrow_med = medians(resid[select], 0)
        relevant_pose = pose[select]
        posed_grid_n = int(relevant_pose.sum()*base_ok.sum())
        denom = int(comparable_polarity[select].sum())
        frame_positive = np.full(count, np.nan)
        for k, a in enumerate(resid[select]):
            good = np.isfinite(a)
            if good.any():
                frame_positive[k] = (a[good]>0.).mean()
        # Consecutive original frames only. NaNs retain discontinuities, and no
        # temporal gap filling or future extrapolation occurs in this summary.
        chosen = np.flatnonzero(select)
        adjacent = chosen[1:][np.diff(chosen)==1]
        deltas = resid[adjacent]-resid[adjacent-1]
        return dict(label=label, frames=count,
            pose_valid=int(relevant_pose.sum()), pose_fraction=fraction(int(relevant_pose.sum()), count),
            normal_finite_cells=int(finite.sum()),
            normal_fraction_all_rows=fraction(int(finite.sum()), count*len(rows)),
            normal_fraction_baseline_rows=fraction(int((finite&supported_grid).sum()), int(supported_grid.sum())),
            normal_fraction_baseline_rows_when_pose_valid=fraction(int((finite&supported_grid).sum()), posed_grid_n),
            residual_finite_cells=int(valid_resid.sum()),
            normal_offset_px=stats(normal[select]), residual_px=stats(resid[select]),
            perframe_residual_median_px=stats(perframe_med),
            perrow_residual_median_px=stats(perrow_med),
            perframe_positive_residual_fraction=stats(frame_positive),
            adjacent_frame_absolute_residual_change_px=stats(abs(deltas)),
            ambiguous_peak_cells=int(ambiguous[select].sum()),
            ambiguous_peak_fraction=fraction(int(ambiguous[select].sum()), int(np.isfinite(peak[select]).sum())),
            peak_ratio=stats(peak[select]), gradient_absolute=stats(abs(grad[select])),
            contrast_gray=stats(arrays['normal_contrast'][select]),
            clipped_pixel_fraction=stats(clips[select]),
            any_clipped_profile_cells=int((np.isfinite(clips[select]) & (clips[select]>0.)).sum()),
            polarity_changed_cells=int(changed_polarity[select].sum()),
            polarity_comparable_cells=denom,
            polarity_changed_fraction=fraction(int(changed_polarity[select].sum()), denom))
    windows = []
    for start in range(0, 60, 10):
        stop = start+10
        select = (t>=start) & ((t<stop) if stop<60 else (t<=stop))
        windows.append(one_window(select, f'{start}<=t<{stop}' if stop<60 else '50<=t<=60'))
    gradient_signs = Counter('positive' if p>0 else 'negative' if p<0 else 'unknown_or_zero'
                            for p in baseline_polarity[base_ok])
    pose_reasons = Counter(r['pose']['diagnostics']['reason'] for r in index['records'])
    poses = [r['pose']['diagnostics'].get('pose') or {} for r in index['records']]
    spatial = dict(supported_rows=int(base_ok.sum()), total_rows=int(len(rows)),
        row_y_all=[int(rows[0]), int(rows[-1])],
        contiguous_supported_ranges=contiguous_ranges(rows, base_ok),
        x_reference=stats(arrays['baseline_x'][base_ok]),
        y_reference=stats(rows[base_ok]),
        normal_nx=stats(arrays['baseline_nx'][base_ok]), normal_ny=stats(arrays['baseline_ny'][base_ok]),
        baseline_mad_px=stats(arrays['baseline_mad_px'][base_ok]),
        baseline_gradient_polarities=dict(gradient_signs),
        baseline_not_old_layer_identity=True)
    per_row = []
    for j in np.flatnonzero(base_ok):
        per_row.append(dict(y=int(rows[j]), x=float(arrays['baseline_x'][j]),
            startup_gradient_sign=float(baseline_polarity[j]),
            poststartup_residual=stats(resid[~warm, j]),
            fixed_10s_medians=[stats(resid[(t>=a)&((t<a+10) if a<50 else (t<=60)), j])['median']
                              for a in range(0, 60, 10)]))
    return dict(side=side, input_hashes={str(index_path):sha(index_path), str(measure_path):digest},
        background_pose=dict(valid=int(pose.sum()), total=len(t), reasons=dict(pose_reasons),
            fitted_center_shift_px=stats([p.get('center_shift_px', np.nan) for p in poses]),
            fitted_rotation_deg=stats([p.get('rotation_deg', np.nan) for p in poses]),
            fitted_scale=stats([p.get('scale', np.nan) for p in poses])),
        baseline_spatial=spatial, all_video=one_window(np.ones(len(t), bool), '0<=t<=60'),
        after_startup=one_window(~warm, '5<t<=60'), fixed_10s_windows=windows,
        supported_row_details=per_row,
        notes=['Unknown cells remain NaN, not zero; denominators distinguish all rows and baseline support.',
               'Peak ratio 1.2 is the pre-existing diagnostic reference, not a tuned detection threshold.',
               'Signed residual is normal image displacement in reference pixels, not rope count/radius.',
               'Polarity change is relative to the per-row startup median gradient, not proof of occlusion.',
               'All times are fixed bins, not manual event-centered windows.',
               'No event labels read, so no interval is classified as normal production or a true event.'])


def main():
    if DEST.exists():
        raise FileExistsError('Do not overwrite an earlier audit')
    p = read(BASE/'protocol.json')
    protocol_digest = sha(BASE/'protocol.json')
    for path, expected in dict(p['source_hashes'], **p['dependencies']).items():
        if sha(path) != expected:
            raise ValueError('frozen dependency changed '+path)
    result = dict(schema='v18b_fixed_bin_observability_audit_1',
        protocol_sha256=protocol_digest, audit_source_sha256=sha(__file__),
        data_status='posthoc_unit_corrected_development_observations_not_blind_validation',
        hardware_access=False, new_thresholds_selected=False, event_times_used=False,
        raw_videos_read=False, heldout_0925_read=False,
        sessions={side:summarize(side,p,protocol_digest) for side in ('left','right')})
    DEST.mkdir()
    (DEST/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,
                                           allow_nan=False),encoding='utf-8')
    compact={side:{key:value for key,value in item.items() if key in (
        'background_pose','baseline_spatial','after_startup','fixed_10s_windows')}
        for side,item in result['sessions'].items()}
    print(json.dumps(compact,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
