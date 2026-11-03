"""Fixed-startup/current-profile consistency checks for V19 research only.

Input contract is the archived V18b fixed-reference normal profile: 131 samples
at integer offsets -65..65 reference pixels, tangent-averaged current grayscale.
No video, model inference, truth files, temporal updates or devices are accessed.
A passing row is only ``profile_consistent``: never confirmed material identity,
physical visibility, new-layer identity, revolution phase or an actuator trigger.
"""
from __future__ import annotations

import math

import numpy as np


PROFILE_SIZE = 131
PROFILE_ORIGIN = 65
MINIMUM_STARTUP_PAIRS = 20
MINIMUM_STARTUP_RELIABLE = 10
MINIMUM_PEAK_RATIO = 1.2
PLATEAU_INNER_PX = 6
PLATEAU_OUTER_PX = 12
MINIMUM_PLATEAU_CONTRAST_GRAY = 2.
MINIMUM_BASELINE_CONTRAST_FRACTION = .5
MAXIMUM_CLIPPED_FRACTION = .25
MODEL_SCHEMA = 'v19_fixed_startup_profile_model_1'


def _readonly(value, dtype=None):
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _finite_nonnegative(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name+' must be a finite nonnegative number')
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise ValueError(name+' must be a finite nonnegative number') from None
    if not math.isfinite(result) or result < 0.:
        raise ValueError(name+' must be a finite nonnegative number')
    return result


def _pose_flags(value, shape):
    """Do not let NaN, strings or arbitrary nonzero integers mean valid pose."""
    a = np.asarray(value)
    if a.shape != shape:
        raise ValueError('pose_valid shape mismatch')
    if a.dtype.kind == 'b':
        return a.astype(bool, copy=True)
    if a.dtype.kind in 'ifu':
        return np.isfinite(a) & (a == 1.)
    return np.zeros(shape, bool)


def _plateaus(profiles, offsets):
    """Right(outward)-left(inward) means, 6..12 pixels from selected peak.

    Nothing is interpolated. All 131 archived samples must be finite, and the
    peak offset must represent an actual integer sample with both plateaus in
    bounds; the parent V18 sampler already requires the full image footprint.
    """
    profiles = np.asarray(profiles, float)
    offsets = np.asarray(offsets, float)
    if profiles.shape != (*offsets.shape, PROFILE_SIZE):
        raise ValueError('profiles must end in 131 samples and match offsets')
    finite = np.isfinite(offsets) & np.isfinite(profiles).all(axis=-1)
    indices = np.zeros(offsets.shape, np.int64)
    # Round only finite, bounded offsets; malformed huge offsets never convert
    # to int and cannot overflow into an apparently supported sample.
    in_range = finite & (offsets >= -PROFILE_ORIGIN) & (offsets <= PROFILE_ORIGIN)
    indices[in_range] = np.rint(offsets[in_range]).astype(np.int64)+PROFILE_ORIGIN
    grid_sample = in_range & (np.abs(offsets-(indices-PROFILE_ORIGIN)) <= 1e-9)
    supported = grid_sample & (indices >= PLATEAU_OUTER_PX) & (
        indices+PLATEAU_OUTER_PX < PROFILE_SIZE)
    signed = np.full(offsets.shape, np.nan)
    inner = np.full(offsets.shape, np.nan)
    outer = np.full(offsets.shape, np.nan)
    for coords in zip(*np.nonzero(supported)):
        k = indices[coords]
        p = profiles[coords]
        inner[coords] = p[k-PLATEAU_OUTER_PX:k-PLATEAU_INNER_PX+1].mean()
        outer[coords] = p[k+PLATEAU_INNER_PX:k+PLATEAU_OUTER_PX+1].mean()
        signed[coords] = outer[coords]-inner[coords]
    return dict(signed=signed, inward_mean=inner, outward_mean=outer,
                supported=supported, profile_finite=finite, offset_is_sample=grid_sample)


def startup_profile_model(times, arrays, ready_s=5.):
    """Build an immutable-array model only from 0 <= time <= ready_s.

    The offset median/MAD use pose-valid finite profile/offset pairs with complete
    6..12px plateau support (>=20).
    Signs/contrast use >=10 of those pairs with peak ratio >=1.2, finite nonzero
    gradient, and full 6..12px plateau support. No post-startup observation value
    is inspected; only array shapes and supplied timestamp metadata are checked.
    Startup residuals are intentionally unused (V18 stores them as NaN).
    """
    ready_s = _finite_nonnegative(ready_s, 'ready_s')
    times = np.asarray(times, float)
    if (times.ndim != 1 or not np.isfinite(times).all()
            or (len(times)>1 and not np.all(np.diff(times)>0.))):
        raise ValueError('times must be finite, one-dimensional and strictly increasing')
    source = np.asarray(arrays['normal_offset'])
    if source.ndim != 2 or source.shape[0] != len(times):
        raise ValueError('normal_offset must have shape (time, row)')
    nrows = source.shape[1]
    shapes = dict(normal_offset=(len(times), nrows), normal_gradient=(len(times), nrows),
                  normal_peak_ratio=(len(times), nrows),
                  normal_profiles=(len(times), nrows, PROFILE_SIZE))
    select = (times >= 0.) & (times <= ready_s)
    warm = {}
    for key, shape in shapes.items():
        a = np.asarray(arrays[key])
        if a.shape != shape:
            raise ValueError(key+' shape mismatch')
        warm[key] = np.asarray(a[select], float)
    pose_all = np.asarray(arrays['pose_valid'])
    if pose_all.shape != times.shape:
        raise ValueError('pose_valid shape mismatch')
    pose = _pose_flags(pose_all[select], (int(select.sum()),))
    offset, gradient, ratio = (warm[key] for key in (
        'normal_offset', 'normal_gradient', 'normal_peak_ratio'))
    plateaus = _plateaus(warm['normal_profiles'], offset)
    pair = plateaus['supported'] & pose[:, None]
    reliable = pair & np.isfinite(ratio) & (ratio>=MINIMUM_PEAK_RATIO)
    reliable &= np.isfinite(gradient) & (gradient!=0.)
    count, reliable_count = pair.sum(axis=0), reliable.sum(axis=0)
    baseline_offset = np.full(nrows, np.nan)
    noise_sigma = np.full(nrows, np.nan)
    threshold = np.full(nrows, np.nan)
    profile_sign = np.full(nrows, np.nan)
    gradient_sign = np.full(nrows, np.nan)
    contrast = np.full(nrows, np.nan)
    supported = np.zeros(nrows, bool)
    reason = np.full(nrows, 'insufficient_startup_profile_offset_pairs', object)
    for j in range(nrows):
        if count[j] < MINIMUM_STARTUP_PAIRS:
            continue
        values = offset[pair[:, j], j]
        baseline_offset[j] = np.median(values)
        mad = np.median(abs(values-baseline_offset[j]))
        noise_sigma[j] = max(1., 1.4826*mad)
        threshold[j] = max(3., 3.*noise_sigma[j])
        if reliable_count[j] < MINIMUM_STARTUP_RELIABLE:
            reason[j] = 'insufficient_startup_unambiguous_plateau_samples'
            continue
        good = reliable[:, j]
        signs = plateaus['signed'][good, j]
        profile_sign[j] = np.sign(np.median(signs))
        gradient_sign[j] = np.sign(np.median(gradient[good, j]))
        contrast[j] = np.median(abs(signs))
        if profile_sign[j] == 0. or gradient_sign[j] == 0.:
            reason[j] = 'startup_sign_unresolved'
            continue
        if profile_sign[j] != gradient_sign[j]:
            reason[j] = 'startup_gradient_plateau_sign_disagree'
            continue
        if not np.isfinite(contrast[j]) or contrast[j] <= 0.:
            reason[j] = 'startup_contrast_unresolved'
            continue
        supported[j] = True
        reason[j] = 'fixed_startup_profile_supported_not_material_identity'
    return dict(schema=MODEL_SCHEMA, ready_s=ready_s, available_after_s=ready_s,
        baseline_offset=_readonly(baseline_offset), noise_sigma=_readonly(noise_sigma),
        threshold_px=_readonly(threshold), profile_sign=_readonly(profile_sign),
        gradient_sign=_readonly(gradient_sign), baseline_contrast=_readonly(contrast),
        supported=_readonly(supported), reason=_readonly(reason),
        profile_offset_count=_readonly(count), reliable_profile_count=_readonly(reliable_count),
        startup_times_s=_readonly(times[select]), startup_frame_count=int(select.sum()),
        units='reference_image_pixels_and_8bit_grayscale_levels',
        scale_interpretation='MAD-derived heuristic scale, not a calibrated confidence bound',
        model_updates_allowed=False, material_identity_confirmed=False,
        visibility_confirmed=False, new_layer_confirmed=False, hardware_access=False)


def profile_evidence(frame, model):
    """Check one current frame against the frozen startup profile model.

    Required frame fields: normal_offset/residual/gradient/peak_ratio (R,),
    normal_profiles (R,131), normal_clipped_fraction (R,), pose_valid scalar,
    time_s scalar. Missing/nonfinite time or time <= model.ready_s is unknown.
    ``profile_residual`` is offset minus this module's plateau-supported startup
    median. Archived ``normal_residual`` can have a different V18 startup median;
    it must be finite but equality is NOT a gate. Their difference is diagnostic
    only. No temporal state or evidence is retained.
    """
    if model.get('schema') != MODEL_SCHEMA:
        raise ValueError('unexpected startup model schema')
    supported = np.asarray(model['supported'], bool)
    if supported.ndim != 1:
        raise ValueError('model supported must be one-dimensional')
    nrows = len(supported)
    fields = {}
    for key in ('normal_offset','normal_residual','normal_gradient','normal_peak_ratio',
                'normal_clipped_fraction'):
        value = np.asarray(frame[key], float)
        if value.shape != (nrows,):
            raise ValueError(key+' must match model rows')
        fields[key] = value
    profiles = np.asarray(frame['normal_profiles'], float)
    if profiles.shape != (nrows, PROFILE_SIZE):
        raise ValueError('normal_profiles must match model rows and 131 samples')
    parameters = {}
    for key in ('baseline_offset','profile_sign','gradient_sign','baseline_contrast',
                'noise_sigma','threshold_px'):
        value = np.asarray(model[key], float)
        if value.shape != (nrows,):
            raise ValueError('model '+key+' must match rows')
        parameters[key] = value
    pose_ok = bool(_pose_flags(frame['pose_valid'], ()))
    time_provided = 'time_s' in frame
    time_ok = False
    if time_provided:
        try:
            current_time = _finite_nonnegative(frame['time_s'], 'time_s')
            time_ok = current_time > _finite_nonnegative(model['ready_s'], 'ready_s')
        except ValueError:
            time_ok = False
    offset, residual, gradient, peak, clipped = (fields[key] for key in (
        'normal_offset','normal_residual','normal_gradient','normal_peak_ratio',
        'normal_clipped_fraction'))
    p = _plateaus(profiles, offset)
    signed = p['signed']
    current_contrast = abs(signed)
    required_contrast = np.maximum(MINIMUM_PLATEAU_CONTRAST_GRAY,
        MINIMUM_BASELINE_CONTRAST_FRACTION*parameters['baseline_contrast'])
    model_finite = supported.copy()
    for value in parameters.values():
        model_finite &= np.isfinite(value)
    reference_comparable = np.isfinite(residual)&np.isfinite(offset)&np.isfinite(parameters['baseline_offset'])
    profile_residual = np.full(nrows, np.nan)
    profile_reference_finite = np.isfinite(offset)&np.isfinite(parameters['baseline_offset'])
    profile_residual[profile_reference_finite] = (
        offset[profile_reference_finite]-parameters['baseline_offset'][profile_reference_finite])
    reference_difference = np.full(nrows, np.nan)
    reference_difference[reference_comparable] = residual[reference_comparable]-(
        profile_residual[reference_comparable])
    checks = dict(
        frame_after_startup=np.full(nrows, time_ok, bool),
        startup_model_supported=model_finite,
        pose_valid=np.full(nrows, pose_ok, bool),
        finite_current_candidate=np.isfinite(offset)&np.isfinite(residual)&np.isfinite(gradient),
        complete_profile_and_plateaus=p['supported'],
        peak_unambiguous=np.isfinite(peak)&(peak>=MINIMUM_PEAK_RATIO),
        gradient_polarity_consistent=np.isfinite(gradient)&(np.sign(gradient)==parameters['gradient_sign'])&(gradient!=0.),
        plateau_polarity_consistent=np.isfinite(signed)&(np.sign(signed)==parameters['profile_sign'])&(signed!=0.),
        plateau_contrast_sufficient=np.isfinite(current_contrast)&(current_contrast>=required_contrast),
        clipped_fraction_acceptable=np.isfinite(clipped)&(clipped>=0.)&(clipped<=MAXIMUM_CLIPPED_FRACTION))
    quality = np.ones(nrows, bool)
    reason = np.full(nrows, 'profile_consistent_not_material_identity', object)
    for key, passed in checks.items():
        first_failure = quality & ~passed
        reason[first_failure] = ('startup_or_nonfinite_time' if key=='frame_after_startup'
                                 else key+'_failed')
        quality &= passed
    return dict(quality_mask=quality, profile_consistent=quality.copy(), reason=reason,
        checks=checks, signed_plateau_contrast=signed,
        profile_residual=profile_residual,
        archived_minus_profile_residual=reference_difference,
        plateau_contrast=current_contrast, required_plateau_contrast=required_contrast,
        inward_plateau_mean=p['inward_mean'], outward_plateau_mean=p['outward_mean'],
        available_after_s=float(model['ready_s']), time_checked=True, time_provided=time_provided,
        material_identity_confirmed=False, visibility_confirmed=False,
        new_layer_confirmed=False, hardware_access=False)
