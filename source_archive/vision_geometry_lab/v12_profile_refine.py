"""Bounded, image-only normal-profile refinement of saved candidate paths.

This postprocessor does not discover rope identity. A bright optical ridge need
not be the rope-body center, and a dark groove can have the same profile as a
dark rope. ``bright_ridge`` uses a 1-D Gaussian-derivative Taylor extremum, NOT
Steger's full asymmetric-profile bias correction. ``paired_edges`` estimates
the midpoint of an independently supported opposite-gradient pair.

There is no target, flange, old frame, network, training or hardware input.
Original path segments are sampled separately without extending their ends.
Every uncertain section falls back to its original *candidate* position, and
is explicitly NOT promoted to an image-measured or physically confirmed point.
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np


METHODS = ('bright_ridge', 'paired_edges')
PARAMETERS = dict(path_sample_step_px=3., profile_sample_step_px=.5,
    profile_half_range_widths=2.2,
    effective_profile_half_range='ceil(max(2.2*w,1.705*w+actual_kernel_radius_px+0.5)/0.5)*0.5',
    filtered_queries_require_full_real_sample_support=True,
    tangential_half_range_widths=.55,
    tangential_profiles=9, tangent_neighborhood_widths=1.5,
    maximum_center_shift_widths=.6, candidate_width_range=[.45, 1.3],
    bright_derivative_sigma_widths=.12, paired_derivative_sigma_widths=.055,
    derivative_sigma_floor_px=.6, minimum_contrast_gray=8.,
    minimum_gradient_gray_per_px=1.5, minimum_edge_balance=.25,
    minimum_tangential_support=.70, maximum_saturated_fraction=.15,
    saturated_channel_threshold=254, minimum_bright_scaled_curvature=3.,
    ambiguity_score_margin=.10, ambiguity_separation_widths=.20,
    profile_background='linear fit between medians of outer 10% flanks',
    fallback='original resampled candidate, not a confirmed measured point')


def _validate(image_rgb, segments, width_px, method):
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) < 3:
        raise ValueError('image_rgb must be HxWx3 uint8')
    width = float(width_px)
    if not math.isfinite(width) or not .5 <= width <= 300:
        raise ValueError('independent optical width_px must be in [0.5,300]')
    if method not in METHODS:
        raise ValueError('method must be bright_ridge or paired_edges')
    result = []
    for source in segments:
        points = np.asarray(source, float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError('each input segment needs at least two finite xy points')
        if (np.any(points < 0) or np.any(points[:, 0] > image.shape[1]-1) or
                np.any(points[:, 1] > image.shape[0]-1)):
            raise ValueError('path point outside the source image')
        points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]]
        if len(points) < 2:
            raise ValueError('degenerate path segment')
        result.append(points)
    return image, result, width


def _resample(points):
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    position = np.r_[np.arange(0., distance[-1], 3.), distance[-1]]
    # Endpoints occur once and are never continued into unobserved path ends.
    position = np.unique(position)
    return np.column_stack([np.interp(position, distance, points[:, axis]) for axis in (0, 1)]), position


def _tangent(points, arc, index, width):
    selected = np.flatnonzero(abs(arc-arc[index]) <= width*1.5)
    if len(selected) < 2:
        selected = np.arange(max(0, index-1), min(len(points), index+2))
    differences = np.diff(points[selected], axis=0)
    norm = np.linalg.norm(differences, axis=1)
    if not np.any(norm > 1e-8):
        return None
    unit = differences[norm > 1e-8]/norm[norm > 1e-8, None]
    tangent = np.median(unit, axis=0)
    norm = np.linalg.norm(tangent)
    return tangent/norm if norm > .2 else None


def _kernels(sigma, step):
    sigma_samples = sigma/step
    radius = max(2, int(math.ceil(4*sigma_samples)))
    coordinate = np.arange(-radius, radius+1, dtype=float)*step
    g = np.exp(-.5*(coordinate/sigma)**2)
    g /= g.sum()
    # cv2 correlation, hence positive first derivative moment.
    d1 = coordinate*g/sigma**2
    d1 /= np.sum(coordinate*d1)
    d2 = (coordinate**2/sigma**4-1/sigma**2)*g
    d2 -= d2.sum()*g
    d2 /= .5*np.sum(coordinate**2*d2)
    return [kernel.astype(np.float32)[None, :] for kernel in (g, d1, d2)]


def _filter(rows, kernel):
    return cv2.filter2D(np.asarray(rows, np.float32), cv2.CV_32F, kernel,
                        borderType=cv2.BORDER_REFLECT_101)


def _profile_layout(width, method):
    """Allocate real context for every filtered contrast/edge query.

    The furthest paired-edge baseline is .6*w + .65*w + .35*1.3*w
    = 1.705*w from the path. Bright baselines reach 1.7*w. The exact
    discrete Gaussian radius (including the sigma floor) and one sample
    guard are added before rounding outwards to a symmetric sample grid.
    """
    step = .5
    sigma = max(.6, width*(.12 if method == 'bright_ridge' else .055))
    kernels = _kernels(sigma, step)
    radius = (kernels[0].shape[1]//2)*step
    half = math.ceil(max(2.2*width, 1.705*width+radius+step)/step)*step
    count = int(round(half/step))
    offsets = np.arange(-count, count+1, dtype=float)*step
    return offsets, sigma, radius, kernels


def _fully_supported(profile, lo, hi):
    supported_lo, supported_hi = profile['supported_profile_interval_px']
    return lo >= supported_lo-1e-9 and hi <= supported_hi+1e-9


def _profiles(gray, clipped, point, tangent, width, method):
    step = .5
    normal = np.array([-tangent[1], tangent[0]])
    offsets, sigma, radius, kernels = _profile_layout(width, method)
    along = np.linspace(-.55*width, .55*width, 9)
    xy = point+along[:, None, None]*tangent+offsets[None, :, None]*normal
    if (xy[..., 0].min() < 1 or xy[..., 1].min() < 1 or
            xy[..., 0].max() > gray.shape[1]-2 or xy[..., 1].max() > gray.shape[0]-2):
        return None
    sampled = cv2.remap(gray, xy[..., 0].astype(np.float32), xy[..., 1].astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    saturation = cv2.remap(clipped, xy[..., 0].astype(np.float32), xy[..., 1].astype(np.float32),
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    profile = np.median(sampled, axis=0)
    tail = max(2, int(len(offsets)*.1))
    left_x, right_x = float(np.mean(offsets[:tail])), float(np.mean(offsets[-tail:]))
    left, right = float(np.median(profile[:tail])), float(np.median(profile[-tail:]))
    background = left+(right-left)*(offsets-left_x)/(right_x-left_x)
    corrected_rows = sampled-background[None]
    corrected = np.median(corrected_rows, axis=0)
    g, k1, k2 = kernels
    smooth = _filter(corrected[None], g)[0]
    first = _filter(corrected[None], k1)[0]
    second = _filter(corrected[None], k2)[0]
    row_smooth = _filter(corrected_rows, g)
    row_first = _filter(corrected_rows, k1)
    roughness = float(np.median(abs(np.diff(corrected, n=2)))) if len(corrected) > 3 else 0.
    return dict(offsets=offsets, normal=normal, profile=smooth,
                raw_profile=smooth+background, first=first, second=second,
                row_smooth=row_smooth, row_first=row_first, saturation=saturation,
                sigma_px=sigma, background_slope=float((right-left)/(right_x-left_x)),
                filter_support_radius_px=float(radius),
                effective_half_range_px=float(offsets[-1]),
                supported_profile_interval_px=[float(offsets[0]+radius), float(offsets[-1]-radius)],
                contrast_floor=max(8., roughness*2.), gradient_floor=max(1.5, roughness*.8),
                step=step)


def _interval(values, offsets, lo, hi):
    keep = (offsets >= lo) & (offsets <= hi)
    if not keep.any():
        return None
    return float(np.median(values[keep]))


def _subpixel_peak(values, index, offsets, step):
    a, b, c = values[index-1:index+2]
    denominator = float(a-2*b+c)
    shift = float(np.clip(.5*(a-c)/denominator, -.5, .5)) if abs(denominator) > 1e-10 else 0.
    return float(offsets[index]+shift*step)


def _bright_candidates(profile, width):
    offsets, first, second = (profile[key] for key in ('offsets', 'first', 'second'))
    indices = np.flatnonzero((first[:-1] >= 0) & (first[1:] < 0))
    candidates, rejected = [], {}
    for left in indices:
        index = int(left if abs(first[left]) <= abs(first[left+1]) else left+1)
        if index <= 0 or index >= len(offsets)-1 or second[index] >= -1e-8:
            continue
        correction = float(-first[index]/second[index])
        center = float(offsets[index]+correction)
        if abs(correction) > .5 or abs(center) > .6*width:
            continue
        scaled_curvature = float(-second[index]*width*width)
        if scaled_curvature < 3.:
            rejected['flat_bright_plateau'] = rejected.get('flat_bright_plateau', 0)+1
            continue
        values = profile['profile']
        if not _fully_supported(profile, center-width*1.1, center+width*1.1):
            rejected['contrast_window_lacks_filter_support'] = rejected.get('contrast_window_lacks_filter_support', 0)+1
            continue
        left_base = _interval(values, offsets, center-width*1.1, center-width*.7)
        right_base = _interval(values, offsets, center+width*.7, center+width*1.1)
        if left_base is None or right_base is None:
            continue
        baseline = max(left_base, right_base)
        peak = float(np.interp(center, offsets, values))
        prominence = peak-baseline
        if prominence < profile['contrast_floor']:
            rejected['weak_two_sided_bright_prominence'] = rejected.get('weak_two_sided_bright_prominence', 0)+1
            continue
        # Removing a broad linear background can manufacture a local maximum
        # immediately beside a SINGLE step edge. Require both sides to support
        # the bright bump in the undetrended image too; otherwise abstain.
        raw_left = _interval(profile['raw_profile'], offsets, center-width*1.1, center-width*.7)
        raw_right = _interval(profile['raw_profile'], offsets, center+width*.7, center+width*1.1)
        raw_peak = float(np.interp(center, offsets, profile['raw_profile']))
        raw_prominence = raw_peak-max(raw_left, raw_right)
        if raw_prominence < profile['contrast_floor']:
            rejected['single_edge_or_no_raw_two_sided_bump'] = rejected.get('single_edge_or_no_raw_two_sided_bump', 0)+1
            continue
        half = baseline+prominence*.5
        before = np.flatnonzero((offsets < center) & (values <= half))
        after = np.flatnonzero((offsets > center) & (values <= half))
        if not len(before) or not len(after):
            continue
        il, ir = int(before[-1]), int(after[0])
        if il+1 >= len(offsets) or ir < 1:
            continue
        # Asymmetric half-height crossings may lie farther out than the
        # nominal contrast windows; all interpolation samples must still
        # have full Gaussian support, never reflected context.
        if not _fully_supported(profile, offsets[il], offsets[ir]):
            rejected['half_height_lacks_filter_support'] = rejected.get('half_height_lacks_filter_support', 0)+1
            continue
        a = offsets[il]+profile['step']*np.clip((half-values[il])/max(float(values[il+1]-values[il]), 1e-8), 0, 1)
        b = offsets[ir-1]+profile['step']*np.clip((values[ir-1]-half)/max(float(values[ir-1]-values[ir]), 1e-8), 0, 1)
        measured = float(b-a)
        if not .45*width <= measured <= 1.3*width:
            rejected['optical_width_outside_prior_range'] = rejected.get('optical_width_outside_prior_range', 0)+1
            continue
        row_peak = profile['row_smooth'][:, index]
        row_left = np.median(profile['row_smooth'][:, (offsets >= center-width*1.1) & (offsets <= center-width*.7)], axis=1)
        row_right = np.median(profile['row_smooth'][:, (offsets >= center+width*.7) & (offsets <= center+width*1.1)], axis=1)
        support = float(np.mean(row_peak-np.maximum(row_left, row_right) >= max(4., prominence*.3)))
        core = abs(offsets-center) <= measured*.3
        saturation = float(np.mean(profile['saturation'][:, core]))
        score = (.6*min(1., prominence/50.)+.25*support+
                 .15*math.exp(-.5*(math.log(measured/width)/.4)**2))
        candidates.append(dict(offset_px=center, width_px=measured, score=score,
            polarity='bright_optical_ridge', center_definition='normal_Gaussian_derivative_extremum',
            evidence=dict(prominence_gray=prominence, scaled_curvature=scaled_curvature,
                          raw_two_sided_prominence_gray=float(raw_prominence),
                          tangential_support=support, saturated_fraction=saturation,
                          sigma_px=profile['sigma_px'], edge_pair=None)))
    return candidates, rejected


def _paired_candidates(profile, width):
    offsets, derivative = profile['offsets'], profile['first']
    magnitude = abs(derivative)
    indices = np.flatnonzero((magnitude[1:-1] > magnitude[:-2]) &
                            (magnitude[1:-1] >= magnitude[2:]) &
                            (magnitude[1:-1] >= profile['gradient_floor']))+1
    peaks = [dict(index=int(index), offset=_subpixel_peak(magnitude, int(index), offsets, profile['step']),
                  gradient=float(derivative[index])) for index in indices
             if _fully_supported(profile, offsets[index-1], offsets[index+1])]
    candidates, rejected = [], {}
    for i, first in enumerate(peaks):
        for second in peaks[i+1:]:
            if first['gradient']*second['gradient'] >= 0:
                continue
            a, b = first['offset'], second['offset']
            measured, center = b-a, (a+b)/2
            if not .45*width <= measured <= 1.3*width or abs(center) > .6*width:
                continue
            if not _fully_supported(profile, a-.35*measured, b+.35*measured):
                rejected['edge_baselines_lack_filter_support'] = rejected.get('edge_baselines_lack_filter_support', 0)+1
                continue
            balance = min(abs(first['gradient']), abs(second['gradient']))/max(abs(first['gradient']), abs(second['gradient']))
            if balance < .25:
                rejected['one_edge_dominates'] = rejected.get('one_edge_dominates', 0)+1
                continue
            polarity = 1 if first['gradient'] > 0 else -1
            inside = _interval(profile['profile'], offsets, a+.25*measured, b-.25*measured)
            outside_a = _interval(profile['profile'], offsets, a-.35*measured, a-.10*measured)
            outside_b = _interval(profile['profile'], offsets, b+.10*measured, b+.35*measured)
            if inside is None or outside_a is None or outside_b is None:
                continue
            contrast = min(polarity*(inside-outside_a), polarity*(inside-outside_b))
            if contrast < profile['contrast_floor']:
                rejected['weak_opposed_edge_body_contrast'] = rejected.get('weak_opposed_edge_body_contrast', 0)+1
                continue
            supports = []
            for edge in (first, second):
                index = edge['index']
                local = profile['row_first'][:, index-1:index+2]*np.sign(edge['gradient'])
                strength = np.max(local, axis=1)
                supports.append(float(np.mean(strength >= max(1.5, abs(edge['gradient'])*.3))))
            support = min(supports)
            core = abs(offsets-center) <= measured*.3
            saturation = float(np.mean(profile['saturation'][:, core]))
            width_score = math.exp(-.5*(math.log(measured/width)/.4)**2)
            score = .45*min(1., contrast/50.)+.20*balance+.20*support+.15*width_score
            candidates.append(dict(offset_px=float(center), width_px=float(measured), score=float(score),
                polarity='bright_body' if polarity > 0 else 'dark_body_or_groove_ambiguous',
                center_definition='midpoint_of_two_apparent_opposed_edges',
                evidence=dict(contrast_gray=float(contrast), edge_balance=float(balance),
                    tangential_support=float(support), saturated_fraction=saturation,
                    sigma_px=profile['sigma_px'], edge_pair=[float(a), float(b)])))
    return candidates, rejected


def _choose(candidates, width):
    if not candidates:
        return None, False, 'no_supported_center_candidate'
    ordered = sorted(candidates, key=lambda candidate: candidate['score'], reverse=True)
    best = ordered[0]
    evidence = best['evidence']
    if evidence['saturated_fraction'] > .15:
        return best, False, 'saturated_profile_center'
    if evidence['tangential_support'] < .70:
        return best, False, 'insufficient_tangential_support'
    for other in ordered[1:]:
        if (abs(other['offset_px']-best['offset_px']) >= .20*width and
                other['score'] >= best['score']-.10):
            return best, False, 'competing_width_compatible_centers'
    if best['polarity'] == 'dark_body_or_groove_ambiguous':
        # Width alone cannot establish whether a dark valley is rope or gap.
        # Keep it diagnostic instead of shifting a seed to an unidentified gap.
        return best, False, 'dark_body_vs_inter_rope_groove_unresolved'
    return best, True, 'accepted_photometric_refinement_identity_unconfirmed'


def refine_paths(image_rgb, segments, width_px, method):
    """Refine each saved segment separately and return every acceptance decision.

    ``segments`` in the return value retains original candidates at uncertain
    positions. Those points are NOT newly measured observations; inspect the
    per-point ``accepted`` flags and ``accepted_segments`` before interpreting
    them as photometrically refined support. Neither output confirms identity.
    ``valid`` is a compatibility alias of ``candidate_path_available`` only;
    it does not mean that CV refinement or a confirmed measurement exists.
    """
    started = time.perf_counter()
    image, sources, width = _validate(image_rgb, segments, width_px, method)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    clipped = (np.max(image, axis=2) >= 254).astype(np.float32)
    output, accepted_segments, records = [], [], []
    reasons = {}
    for segment_index, source in enumerate(sources):
        points, arc = _resample(source)
        refined, accepted_run = [], []
        for point_index, point in enumerate(points):
            tangent = _tangent(points, arc, point_index, width)
            profile = _profiles(gray, clipped, point, tangent, width, method) if tangent is not None else None
            candidates, rejected = ([], {})
            chosen, accepted = None, False
            if tangent is None:
                reason = 'local_tangent_unavailable'
            elif profile is None:
                reason = 'profile_footprint_outside_source_image'
            else:
                candidates, rejected = (_bright_candidates(profile, width) if method == 'bright_ridge'
                                        else _paired_candidates(profile, width))
                chosen, accepted, reason = _choose(candidates, width)
            proposed = point+chosen['offset_px']*profile['normal'] if chosen is not None else None
            final = proposed if accepted else point
            refined.append(final.tolist())
            if accepted:
                accepted_run.append(final.tolist())
            else:
                if len(accepted_run) >= 2:
                    accepted_segments.append(accepted_run)
                accepted_run = []
            reasons[reason] = reasons.get(reason, 0)+1
            records.append(dict(segment_index=segment_index, sample_index=point_index,
                source_arc_px=float(arc[point_index]), input_xy=point.tolist(),
                proposed_xy=proposed.tolist() if proposed is not None else None,
                final_xy=final.tolist(), accepted=bool(accepted), reason=reason,
                applied_offset_px=float(chosen['offset_px']) if accepted else 0.,
                proposed_offset_px=float(chosen['offset_px']) if chosen is not None else None,
                candidate_width_px=float(chosen['width_px']) if chosen is not None else None,
                evidence=chosen['evidence'] if chosen is not None else {},
                profile_support=(dict(filter_support_radius_px=profile['filter_support_radius_px'],
                    effective_half_range_px=profile['effective_half_range_px'],
                    supported_profile_interval_px=profile['supported_profile_interval_px'])
                    if profile is not None else None),
                candidate_count=len(candidates), rejected_candidate_reasons=rejected,
                top_candidates=sorted(candidates, key=lambda candidate:candidate['score'], reverse=True)[:3],
                fallback_is_confirmed_observation=False, identity_confirmed=False))
        if len(accepted_run) >= 2:
            accepted_segments.append(accepted_run)
        output.append(refined)
    accepted_count = sum(record['accepted'] for record in records)
    status = ('no_candidate_path' if not output else 'fallback_only' if accepted_count == 0
              else 'fully_refined_candidate' if accepted_count == len(records)
              else 'partially_refined_candidate')
    return dict(segments=output, accepted_segments=accepted_segments,
        valid=bool(output), valid_meaning='candidate_path_available_only_not_measured_validity',
        candidate_path_available=bool(output), cv_refinement_available=bool(accepted_segments),
        status=status, identity_confirmed=False, physical_center_confirmed=False,
        physical_contact_confirmed=False, new_layer_confirmed=False, force_command=None,
        diagnostics=dict(method=method, parameters=PARAMETERS.copy(), input_segments=len(sources),
            output_segments=len(output), sampled_points=len(records), accepted_points=accepted_count,
            uncertain_fallback_points=len(records)-accepted_count, reason_counts=reasons,
            records=records, independent_width_px=width, targets_used=False, previous_frames_used=False,
            original_segment_ends_not_extended=True, gaps_bridged=False,
            subpixel_profile_estimate_is_physical_accuracy=False, full_steger_bias_correction=False,
            timing_scope='image_conversion_resampling_profiles_selection_no_file_decode',
            elapsed_ms=(time.perf_counter()-started)*1000))
