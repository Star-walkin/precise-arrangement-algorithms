"""Read-only features on an already unfolded boundary-candidate patch.

Rows follow the rope tangent and columns its normal. This module has no image
search, target, training, history or equipment access. A caller must supply
real, unpadded samples and declare whether its sampling scale came from GT.
Neither low correlation nor a dark valley proves a rope-instance boundary.
"""
from __future__ import annotations

import math
import time

import numpy as np


PARAMETERS = dict(
    minimum_rows=40, minimum_columns=15,
    candidate_normal_coordinate='(columns-1)/2; fractional centers allowed',
    side_center_exclusion_half_samples=1.,
    transition_derivative='centered finite difference over actual adjacent columns',
    transition_gradient_floor_gray_per_px=.05,
    normal_profile='median over all tangent rows',
    dark_valley_smoothing_samples=3,
    dark_valley_minimum_two_sided_prominence_gray=2.,
    texture_profile='mean over side columns, separately least-squares linear detrended',
    minimum_texture_std_gray=2.,
    autocorrelation_minimum_lag_samples=2,
    autocorrelation_maximum_lag_samples=64,
    autocorrelation_lag_upper_formula='min(64,floor((rows-1)/3)); use actual sample-center extent',
    minimum_periods_in_patch=3.,
    minimum_period_peak_correlation=.45,
    minimum_period_peak_prominence=.05,
    autocorrelation='Pearson on real overlapping detrended samples at each lag',
    cross_correlation_maximum_lag_samples=32,
    cross_correlation_maximum_lag_fraction=.25,
    cross_correlation_minimum_overlap_samples=24,
    cross_correlation_minimum_overlap_fraction=.75,
    positive_cross_lag='correlate left[t] with right[t+lag]',
    filter_boundary='no padding; normal box filter valid only; all correlations use real overlap',
    input_support='all input pixels must be real; caller must reject remap/padded patches',
    scale_source='caller supplied, not verified; may be GT-conditioned for diagnostic audit',
    combined_boundary_score=False,
)

AMBIGUITIES = [
    'The same rope may contain periodic strand texture; periodicity does not identify a rope.',
    'Adjacent ropes may have the same texture frequency and phase.',
    'Low cross-boundary correlation does not prove a boundary or different rope identities.',
    'Phase-shifted texture may have low zero-lag correlation on the same rope.',
    'High lagged correlation does not prove that two sides belong to the same rope.',
    'Averaging side columns can cancel inclined or out-of-phase strand texture.',
    'A dark valley can be a strand groove, inter-rope gap, shadow or lighting artifact.',
    'No periodic peak can mean insufficient cycles or inadequate image evidence, not absence of texture.',
    'Periods shorter than two tangent samples are unresolved or aliased by this patch sampling.',
]


def _detrend(values):
    values = np.asarray(values, float)
    position = np.linspace(-1., 1., len(values))
    centered = values-values.mean(axis=0)
    if values.ndim == 1:
        slope = float(position @ centered)/(position @ position)
        return centered-position*slope
    slope = position @ centered/(position @ position)
    return centered-position[:, None]*slope


def _pearson(left, right):
    left, right = np.asarray(left, float), np.asarray(right, float)
    a, b = left-left.mean(), right-right.mean()
    denominator = float(np.linalg.norm(a)*np.linalg.norm(b))
    if denominator <= 1e-10:
        return None
    return float(np.clip(a @ b/denominator, -1., 1.))


def _period_features(side, tangent_step):
    raw = np.mean(side, axis=1)
    residual = _detrend(raw)
    std = float(np.std(residual))
    column_std = np.std(_detrend(side), axis=0)
    column_rms = float(np.sqrt(np.mean(column_std**2)))
    cancellation = float(std/column_rms) if column_rms > 1e-10 else None
    count = len(raw)
    lag_max = min(64, int(math.floor((count-1)/3.)))
    result = dict(available=False, reasons=[], sample_count=count,
        mean_intensity_gray=float(raw.mean()), detrended_std_gray=std,
        within_side_column_texture_rms_gray=column_rms,
        side_averaging_retained_rms_ratio=cancellation,
        lag_search_samples=[2, lag_max],
        lag_search_px=[2*tangent_step, lag_max*tangent_step],
        minimum_periods=3., peak_lag_samples=None, peak_period_px=None,
        peak_correlation=None, peak_prominence=None, observed_periods=None,
        raw_peak_lag_samples=None, raw_peak_correlation=None, raw_peak_prominence=None,
        largest_tested_correlation=None, autocorrelation=[])
    if std < 2.:
        result['reasons'].append('insufficient_tangent_texture_after_side_averaging')
    if std <= 1e-10:
        return result, residual
    if lag_max < 2:
        result['reasons'].append('insufficient_samples_for_three_periods')
        return result, residual
    # One extra lag is a measured neighbor for endpoint-peak testing only;
    # it is never selected as a period with fewer than three observed cycles.
    correlations = [_pearson(residual[:-lag], residual[lag:])
                    for lag in range(1, lag_max+2)]
    candidates = []
    for lag in range(2, lag_max+1):
        value, before, after = correlations[lag-1], correlations[lag-2], correlations[lag]
        result['autocorrelation'].append(dict(lag_samples=lag,
            lag_px=lag*tangent_step, correlation=value, overlap_samples=count-lag))
        if value is None or before is None or after is None:
            continue
        if value > before and value >= after:
            # Prominence against the nearest lower shoulders toward zero
            # and the end of the permitted lag domain, without extrapolation.
            previous = [v for v in correlations[:lag-1] if v is not None]
            following = [v for v in correlations[lag:lag_max+1] if v is not None]
            prominence = value-max(min(previous), min(following))
            candidates.append((lag, value, prominence))
    finite = [entry['correlation'] for entry in result['autocorrelation']
              if entry['correlation'] is not None]
    result['largest_tested_correlation'] = max(finite) if finite else None
    if candidates:
        raw_lag, raw_value, raw_prominence = max(candidates, key=lambda item:(item[1], -item[0]))
        result.update(raw_peak_lag_samples=raw_lag, raw_peak_correlation=raw_value,
                      raw_peak_prominence=raw_prominence)
    eligible = [item for item in candidates if item[1] >= .45 and item[2] >= .05]
    if not eligible:
        result['reasons'].append('no_supported_nonzero_lag_peak_or_insufficient_cycles')
        return result, residual
    if std < 2.:
        return result, residual
    # First eligible peak, not an arbitrarily selected stronger harmonic.
    lag, value, prominence = min(eligible, key=lambda item: item[0])
    result.update(available=True, peak_lag_samples=lag, peak_period_px=lag*tangent_step,
        peak_correlation=value, peak_prominence=prominence, observed_periods=(count-1)/lag)
    return result, residual


def _cross_features(left, right, tangent_step):
    count = len(left)
    max_lag = min(32, int(math.floor(.25*count)))
    minimum_overlap = max(24, int(math.ceil(.75*count)))
    result = dict(available=False, reasons=[], zero_lag_correlation=None,
        maximum_shifted_correlation=None, maximum_shifted_absolute_correlation=None,
        best_lag_samples=None, best_lag_px=None, best_overlap_samples=None,
        lag_search_samples=[-max_lag, max_lag],
        lag_search_px=[-max_lag*tangent_step, max_lag*tangent_step],
        minimum_overlap_samples=minimum_overlap, correlations=[])
    if min(float(np.std(left)), float(np.std(right))) < 2.:
        result['reasons'].append('one_or_both_sides_have_insufficient_tangent_texture')
        return result
    for lag in range(-max_lag, max_lag+1):
        if lag < 0:
            a, b = left[-lag:], right[:count+lag]
        elif lag > 0:
            a, b = left[:count-lag], right[lag:]
        else:
            a, b = left, right
        if len(a) < minimum_overlap:
            continue
        value = _pearson(a, b)
        if value is not None:
            result['correlations'].append(dict(lag_samples=lag, lag_px=lag*tangent_step,
                correlation=value, overlap_samples=len(a)))
        if lag == 0:
            result['zero_lag_correlation'] = value
    if not result['correlations']:
        result['reasons'].append('no_nonflat_real_overlap')
        return result
    best = max(result['correlations'], key=lambda item:(item['correlation'], -abs(item['lag_samples'])))
    result.update(available=True, maximum_shifted_correlation=best['correlation'],
        maximum_shifted_absolute_correlation=max(abs(item['correlation']) for item in result['correlations']),
        best_lag_samples=best['lag_samples'], best_lag_px=best['lag_px'],
        best_overlap_samples=best['overlap_samples'])
    return result


def _transition(values, normal_step, left_columns, right_columns):
    center = (values.shape[1]-1)/2
    gradients = (values[:, 2:]-values[:, :-2])/(2*normal_step)
    coordinates = np.arange(1, values.shape[1]-1, dtype=float)
    central = np.array([np.interp(center, coordinates, row) for row in gradients])
    absolute = float(np.median(abs(central)))
    overall_mean, overall_p90 = float(np.mean(abs(gradients))), float(np.percentile(abs(gradients), 90))
    signed = float(np.median(central))
    side_difference = np.mean(values[:, right_columns], axis=1)-np.mean(values[:, left_columns], axis=1)
    reasons = []
    if overall_mean < .05:
        reasons.append('whole_patch_gradient_mean_too_small_for_relative_gradient')
    if overall_p90 < .05:
        reasons.append('whole_patch_gradient_p90_too_small_for_relative_gradient')
    return dict(available=True, reasons=reasons, signed_gradient=signed,
        absolute_gradient=absolute, absolute_signed_gradient=abs(signed),
        gradient_units='gray_per_native_normal_px',
        whole_patch_absolute_gradient_mean=overall_mean,
        whole_patch_absolute_gradient_p90=overall_p90,
        relative_to_patch_mean=absolute/overall_mean if overall_mean >= .05 else None,
        relative_to_patch_p90=absolute/overall_p90 if overall_p90 >= .05 else None,
        signed_side_intensity_change=float(np.median(side_difference)),
        absolute_side_intensity_change=float(np.median(abs(side_difference))))


def _dark_valley(values, normal_step):
    normal_profile = np.median(values, axis=0)
    # Valid convolution: every smoothed location uses exactly three real
    # columns, with no reflection/zero padding and no endpoint extremum.
    profile = np.convolve(normal_profile, np.ones(3)/3., mode='valid')
    coordinates = np.arange(1, values.shape[1]-1, dtype=float)
    center = (values.shape[1]-1)/2
    candidates = []
    for index in range(1, len(profile)-1):
        if not (profile[index] < profile[index-1] and profile[index] <= profile[index+1]):
            continue
        prominence = min(float(np.max(profile[:index])), float(np.max(profile[index+1:]))) - float(profile[index])
        if prominence < 2.:
            continue
        a, b, c = profile[index-1:index+2]
        denominator = a-2*b+c
        shift = float(np.clip(.5*(a-c)/denominator, -.5, .5)) if denominator > 1e-10 else 0.
        location = float(coordinates[index]+shift)
        candidates.append(dict(normal_column=location,
            offset_samples=location-center, offset_px=(location-center)*normal_step,
            two_sided_prominence_gray=prominence, smoothed_intensity_gray=float(profile[index])))
    closest = min(candidates, key=lambda item:abs(item['offset_samples'])) if candidates else None
    return dict(available=bool(candidates),
        reasons=[] if candidates else ['no_supported_two_sided_dark_valley'],
        offset_px=closest['offset_px'] if closest else None,
        offset_samples=closest['offset_samples'] if closest else None,
        prominence_gray=closest['two_sided_prominence_gray'] if closest else None,
        selection='nearest supported local minimum to candidate center; not a boundary verdict',
        candidate_count=len(candidates), candidates=candidates,
        profile_columns=coordinates.tolist(), smoothed_normal_profile=profile.tolist())


def analyze_patch(gray_patch_uint8, normal_step_px, tangent_step_px):
    """Return independent diagnostics; ``available`` never means boundary valid.

    Units of both steps are native image pixels per patch sample. The caller
    controls unfolding and must exclude invalid image-remap samples. Unknown
    periodicity/texture stays explicitly unavailable and must remain in an
    audit denominator. This function accepts no GT or class/instance label.
    """
    started = time.perf_counter()
    source = np.asarray(gray_patch_uint8)
    if source.ndim != 2 or source.dtype != np.uint8:
        raise ValueError('gray_patch_uint8 must be a two-dimensional uint8 array')
    normal_step, tangent_step = float(normal_step_px), float(tangent_step_px)
    if not all(math.isfinite(value) and value > 0 for value in (normal_step, tangent_step)):
        raise ValueError('normal_step_px and tangent_step_px must be finite positive values')
    rows, columns = source.shape
    result = dict(available=False, availability_meaning='diagnostic_features_computable_not_boundary_valid',
        reasons=[], parameters=PARAMETERS.copy(), ambiguities=list(AMBIGUITIES),
        geometry=dict(rows=rows, columns=columns, normal_step_px=normal_step,
            tangent_step_px=tangent_step, candidate_center_column=(columns-1)/2,
            normal_extent_px=max(0, columns-1)*normal_step,
            tangent_extent_px=max(0, rows-1)*tangent_step,
            scale_source='caller_supplied_not_verified', input_support_checked_by_module=False),
        transition=None, dark_valley=None, tangent_texture=None, cross_boundary_texture=None)
    if rows < 40 or columns < 15:
        result['reasons'].append('patch_smaller_than_40_tangent_by_15_normal_samples')
        result['elapsed_ms'] = (time.perf_counter()-started)*1000
        return result
    values = source.astype(float)
    offset = np.arange(columns)-(columns-1)/2
    left_columns, right_columns = offset < -1., offset > 1.
    left, left_residual = _period_features(values[:, left_columns], tangent_step)
    right, right_residual = _period_features(values[:, right_columns], tangent_step)
    result.update(available=True,
        transition=_transition(values, normal_step, left_columns, right_columns),
        dark_valley=_dark_valley(values, normal_step), tangent_texture=dict(left=left, right=right),
        cross_boundary_texture=_cross_features(left_residual, right_residual, tangent_step))
    result['geometry'].update(left_columns=np.flatnonzero(left_columns).tolist(),
        right_columns=np.flatnonzero(right_columns).tolist())
    for side, features in (('left', left), ('right', right)):
        result['reasons'].extend(side+': '+reason for reason in features['reasons'])
    result['reasons'].extend('cross: '+reason for reason in result['cross_boundary_texture']['reasons'])
    result['elapsed_ms'] = (time.perf_counter()-started)*1000
    return result
