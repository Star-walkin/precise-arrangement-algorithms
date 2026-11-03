"""One frozen seed-conditioned connector for four offline ridge baselines.

Only BGR pixels, a free-span seed and an independently measured optical width
enter this module. No wound target, flange boundary, previous frame or device
interface is accepted. The result is a photometric candidate, never confirmed
rope identity, contact, layer state or force timing. Missing columns break the
returned observed segments; no interpolated gap is emitted as an observation.
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from .v10_ridge_candidates import METHODS, extract_ridge_candidates


TRACE_METHODS = METHODS + ("legacy_v3",)
PARAMETERS = dict(resolution_px=2., downstream_length_px=1536.,
                  half_normal_range_px=768., free_anchor_widths=4.,
                  beam_size=24, column_step_widths=.50, peak_threshold=.12,
                  peak_nms_widths=.25, initial_corridor_widths=.75,
                  maximum_step_jump_widths=.60, maximum_absolute_slope=1.2,
                  maximum_missing_columns=2, slope_penalty=.18,
                  curvature_penalty=.45, orientation_reward=.25,
                  missing_column_penalty=1.15, free_anchor_penalty=.65,
                  free_anchor_penalty_widths=.70, slope_smoothing=.65,
                  beam_position_cell_widths=.25, beam_slope_cell=.16,
                  ambiguity_average_reward_gap=.12,
                  ambiguity_minimum_length_ratio=.75,
                  candidate_count=3, candidate_polarity="bright_only",
                  maximum_peaks_per_column=128)


def _validate(image, free_seed, width_px, method, max_length_px):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 8:
        raise ValueError("image must be BGR HxWx3, at least 8x8")
    if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
        raise ValueError("image must be finite numeric BGR")
    if image.min() < 0 or image.max() > 255:
        raise ValueError("image values must be in [0,255]")
    seed = np.asarray(free_seed, float)
    if seed.ndim != 2 or seed.shape[1] != 2 or len(seed) < 2 or not np.isfinite(seed).all():
        raise ValueError("free_seed must contain at least two finite xy points")
    if np.any(seed < 0) or np.any(seed[:, 0] > image.shape[1]-1) or np.any(seed[:, 1] > image.shape[0]-1):
        raise ValueError("free_seed must be inside the image")
    width = float(width_px)
    if not math.isfinite(width) or width <= 0 or width > 200:
        raise ValueError("independent width must be in (0,200] source pixels")
    if method not in TRACE_METHODS:
        raise ValueError("unknown ridge method")
    direction = seed[-1]-seed[0]
    length = float(np.linalg.norm(direction))
    if length < max(2., width):
        raise ValueError("free_seed direction is too short")
    downstream = PARAMETERS['downstream_length_px'] if max_length_px is None else float(max_length_px)
    if not math.isfinite(downstream) or downstream <= 0 or downstream > 1536:
        raise ValueError("max_length_px must be in (0,1536]; smaller values are for synthetic tests")
    return image.astype(np.float32, copy=False), seed, width, direction/length, downstream


def _coordinates(origin, d, n, u, v):
    x = (origin[0] + u[None, :]*d[0] + v[:, None]*n[0]).astype(np.float32)
    y = (origin[1] + u[None, :]*d[1] + v[:, None]*n[1]).astype(np.float32)
    return x, y


def _make_features(image, origin, d, n, width, downstream, method):
    resolution = PARAMETERS['resolution_px']
    length = 4*width + downstream
    if method == 'legacy_v3':
        from .wound_rope_tracker import _features
        raw = _features(image, origin, d, n, width, resolution, length, guided=False)
        u, v = raw['u'], raw['v']
        mx, my = _coordinates(origin, d, n, u, v)
        margin = max(width, resolution*max(2, int(round(width*.65/resolution))))
        valid = ((mx > margin) & (mx < image.shape[1]-margin) &
                 (my > margin) & (my < image.shape[0]-margin))
        border = max(2, int(round(width*.65/resolution)))
        valid[:border] = False
        valid[-border:] = False
        bright = np.where(valid & (raw['prominence'] >= raw['minimum_prominence']),
                          raw['response'], 0).astype(np.float32)
        tangent = np.stack([np.cos(raw['tangent']), np.sin(raw['tangent'])], axis=-1)
        return dict(u=u, v=v, bright=bright, dark=np.zeros_like(bright),
                    tangent=tangent, valid=valid,
                    detector_metadata=dict(method='legacy_v3_features_unchanged',
                        response_is_probability=False, dark_response_available=False,
                        own_half_normal_range_px=float(max(abs(v[0]), abs(v[-1])))),
                    photometric_supported=bright >= PARAMETERS['peak_threshold'])
    u = np.arange(0., length+resolution*.5, resolution)
    v = np.arange(-PARAMETERS['half_normal_range_px'],
                   PARAMETERS['half_normal_range_px']+resolution*.5, resolution)
    strip_width = width/resolution
    normalization_sigma = max(4., min(128., 2*strip_width))
    scale_ratios = (.28,) if method == 'normalized_hessian' else (.16, .28, .42)
    largest_sigma = max(min(12., max(.8, ratio*strip_width)) for ratio in scale_ratios)
    # The normalization is itself a broad convolution. Its support plus the
    # derivative support must come from real source pixels, not zero-filled
    # image padding. Remap a halo OUTSIDE the prescribed core strip first;
    # eroding only the core would destroy otherwise visible seed anchors.
    guard = int(math.ceil(4*normalization_sigma + 4*largest_sigma + 3))
    expanded_u = (np.arange(-guard, len(u)+guard, dtype=np.float32)*resolution)
    expanded_v = v[0]+np.arange(-guard, len(v)+guard, dtype=np.float32)*resolution
    mx, my = _coordinates(origin, d, n, expanded_u, expanded_v)
    source_valid = ((mx >= 1) & (mx < image.shape[1]-2) &
                    (my >= 1) & (my < image.shape[0]-2))
    expanded_strip = cv2.remap(image, mx, my, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)
    complete_support = cv2.erode(source_valid.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_RECT, (2*guard+1, 2*guard+1)),
        borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    core = np.s_[guard:guard+len(v), guard:guard+len(u)]
    valid = complete_support[core]
    core_roi = [guard, guard, guard+len(u), guard+len(v)]
    result = extract_ridge_candidates(expanded_strip, method=method,
                                      width_px=strip_width, polarity='bright', roi=core_roi)
    valid &= result['observation_mask']
    # For all three new methods, a dense Hessian/vesselness response alone is
    # insufficient. Require that method's own localized ridge-peak support.
    # In the Taylor arm this is precisely the normal-offset-within-pixel gate;
    # the other two use their fixed normal NMS. The connector is unchanged.
    bright_support = (result['supported_mask'] & valid &
                      (result['polarity_map'] > 0))
    bright = np.where(bright_support, result['peak_response'], 0)
    dark = np.where(valid, result['dark_response'], 0)
    return dict(u=u, v=v, bright=bright, dark=dark,
                tangent=result['tangent_xy'], valid=valid,
                photometric_supported=bright_support,
                detector_metadata=dict(result['metadata'],
                    path_response='peak_response_supported_mask_bright_polarity',
                    dense_response_alone_cannot_enter_connector=True,
                    image_remap_full_filter_guard_strip_px=guard,
                    normalization_sigma_strip_px=normalization_sigma,
                    largest_derivative_sigma_strip_px=largest_sigma,
                    remap_core_roi_xyxy=core_roi,
                    remap_expanded_shape_hw=list(expanded_strip.shape[:2]),
                    remap_padding_policy='full_normalization_plus_derivative_support_required',
                    lower_quality_padded_context_is_emitted=False))


def _peaks(features, column, width, field='bright'):
    value = features[field][:, column]
    keep = ((value[1:-1] >= value[:-2]) & (value[1:-1] > value[2:]) &
            (value[1:-1] >= PARAMETERS['peak_threshold']) & features['valid'][1:-1, column])
    indices = np.flatnonzero(keep)+1
    order = sorted(indices, key=lambda index: (-float(value[index]), int(index)))
    selected = []
    spacing = width*PARAMETERS['peak_nms_widths']
    for index in order:
        if all(abs(features['v'][index]-features['v'][other]) >= spacing for other in selected):
            selected.append(int(index))
            if len(selected) >= PARAMETERS['maximum_peaks_per_column']:
                break
    result = []
    for index in sorted(selected):
        a, b, c = value[index-1:index+2]
        denominator = float(a-2*b+c)
        shift = float(np.clip(.5*(a-c)/denominator, -.5, .5)) if abs(denominator) > 1e-8 else 0.
        tangent = features['tangent'][index, column]
        result.append(dict(column=int(column), u=float(features['u'][column]),
                           v=float(features['v'][index]+shift*PARAMETERS['resolution_px']),
                           quality=float(value[index]), tangent=[float(x) for x in tangent],
                           observed=True))
    return result


def _beam(features, width):
    stride = max(1, int(round(width*PARAMETERS['column_step_widths']/PARAMETERS['resolution_px'])))
    columns = list(range(1, len(features['u'])-1, stride))
    peak_lists = [_peaks(features, column, width) for column in columns]
    dark_peak_count = sum(len(_peaks(features, column, width, 'dark')) for column in columns)
    # Remap/derivative validity may remove the first columns. Start at the first
    # geometrically supported anchor column, never search beyond the known seed.
    start = next((i for i, column in enumerate(columns)
                  if features['u'][column] <= 4*width and
                  np.any(features['valid'][abs(features['v']) < .75*width, column])), None)
    stats = dict(column_stride_strip_px=stride, column_step_source_px=stride*2,
                 processed_columns=len(columns), bright_peak_count=sum(map(len, peak_lists)),
                 dark_peak_count=dark_peak_count, starting_column=None,
                 dark_peaks_used_as_centers=False)
    if start is None:
        return [], dict(stats, reason='free_anchor_outside_valid_strip')
    stats['starting_column'] = columns[start]
    initial = [node for node in peak_lists[start] if abs(node['v']) < width*.75]
    if not initial:
        return [], dict(stats, reason='no_bright_ridge_at_free_anchor')
    beam = [dict(path=(node,), slope=0., score=node['quality']-.2*abs(node['v'])/width,
                 gaps=0) for node in initial]
    finished = []
    for column, nodes in zip(columns[start+1:], peak_lists[start+1:]):
        u = float(features['u'][column])
        extensions = []
        for state in beam:
            last = state['path'][-1]
            du = u-last['u']
            predicted = last['v']+state['slope']*du
            accepted = 0
            for node in nodes:
                if abs(node['v']-last['v']) > .60*width:
                    continue
                slope = (node['v']-last['v'])/du
                if abs(slope) > 1.2:
                    continue
                direction = np.array([1., slope])/math.sqrt(1+slope*slope)
                alignment = abs(float(np.dot(direction, node['tangent'])))
                change = abs(slope-state['slope'])
                reward = node['quality']+.25*alignment-.18*abs(slope)-.45*change
                if u < 4*width:
                    reward -= .65*(node['v']/max(2., width*.7))**2
                extensions.append(dict(path=state['path']+(node,),
                    slope=.65*slope+.35*state['slope'], score=state['score']+reward, gaps=0))
                accepted += 1
            if state['gaps'] < 2:
                missing = dict(column=int(column), u=u, v=predicted, quality=0.,
                               tangent=[0., 0.], observed=False)
                extensions.append(dict(path=state['path']+(missing,), slope=state['slope'],
                                       score=state['score']-1.15, gaps=state['gaps']+1))
            elif not accepted:
                finished.append(state)
        if not extensions:
            beam = []
            break
        occupied, next_beam = set(), []
        for state in sorted(extensions, key=lambda item: item['score'], reverse=True):
            cell = (round(state['path'][-1]['v']/max(1., width*.25)),
                    round(state['slope']/.16), state['gaps'])
            if cell not in occupied:
                occupied.add(cell)
                next_beam.append(state)
            if len(next_beam) == PARAMETERS['beam_size']:
                break
        beam = next_beam
    finished.extend(beam)
    trimmed = []
    for state in finished:
        nodes = list(state['path'])
        while nodes and not nodes[-1]['observed']:
            nodes.pop()
        if sum(node['observed'] for node in nodes) < 2:
            continue
        # Ranking is fixed cumulative image reward/continuity, not closeness
        # to a wound target or flange. No label-selected best alternative.
        trimmed.append(dict(state, path=tuple(nodes)))
    return sorted(trimmed, key=lambda item: item['score'], reverse=True), dict(stats, reason='ok')


def _distinct(a, b, width):
    da = {node['column']: node['v'] for node in a['path'] if node['observed']}
    db = {node['column']: node['v'] for node in b['path'] if node['observed']}
    common = sorted(da.keys() & db.keys())
    if not common:
        return True
    return (max(abs(da[column]-db[column]) for column in common) >= width*.25 or
            abs(max(da)-max(db))*2 >= 2*width)


def trace_seed_ridges(image_bgr, free_seed, width_px, *, method,
                      max_length_px=1536., return_maps=False):
    """Top-one observed segments plus top-three diagnostic alternatives.

    ``max_length_px`` overrides only downstream search length for synthetic
    tests; benchmark runs must leave it unset. Free seed points are ordered
    from incoming free span toward the wound surface. No state inference is
    possible from this API. Timing excludes caller file decode, includes all
    local remapping, features, peak search, connecting and serialization.
    """
    started = time.perf_counter()
    image, seed, width, d, downstream = _validate(image_bgr, free_seed, width_px, method, max_length_px)
    n = np.array([-d[1], d[0]])
    origin = seed[-1]-4*width*d
    feature_started = time.perf_counter()
    features = _make_features(image, origin, d, n, width, downstream, method)
    feature_ms = (time.perf_counter()-feature_started)*1000
    connect_started = time.perf_counter()
    states, diagnostics = _beam(features, width)
    chosen = []
    for state in states:
        if all(_distinct(state, old, width) for old in chosen):
            chosen.append(state)
        if len(chosen) == 3:
            break
    candidates = []
    for state in chosen:
        segments, current = [], []
        for node in state['path']:
            if node['observed']:
                point = origin+node['u']*d+node['v']*n
                current.append(point.tolist())
            else:
                if len(current) >= 2:
                    segments.append(current)
                current = []
        if len(current) >= 2:
            segments.append(current)
        length = sum(float(np.linalg.norm(np.diff(np.asarray(segment), axis=0), axis=1).sum())
                     for segment in segments)
        observed_count = sum(node['observed'] for node in state['path'])
        candidates.append(dict(observed_segments=segments, score=float(state['score']),
            average_reward=float(state['score']/len(state['path'])),
            observed_samples=observed_count, missing_samples=len(state['path'])-observed_count,
            supported_length_px=length, identity_confirmed=False))
    ambiguous = False
    score_gap = None
    if len(candidates) > 1:
        score_gap = candidates[0]['average_reward']-candidates[1]['average_reward']
        length_ratio = candidates[1]['supported_length_px']/max(candidates[0]['supported_length_px'], 1e-8)
        ambiguous = bool(abs(score_gap) < .12 and length_ratio >= .75)
    connect_ms = (time.perf_counter()-connect_started)*1000
    u, v = features['u'], features['v']
    geometry = dict(origin_xy=origin.tolist(), direction_xy=d.tolist(), normal_xy=n.tolist(),
        d_xy=d.tolist(), n_xy=n.tolist(), resolution=2.,
        u_min=float(u[0]), u_max=float(u[-1]), v_min=float(v[0]), v_max=float(v[-1]),
        resolution_px=2., u_range_px=[float(u[0]), float(u[-1])],
        v_range_px=[float(v[0]), float(v[-1])], strip_shape_hw=list(features['bright'].shape),
        free_seed_end_u_px=4*width, source_image_shape_hw=list(image.shape[:2]),
        mapping='source_xy = origin_xy + strip_column*2*direction_xy + (v0+strip_row*2)*normal_xy',
        support_differs_from_new_methods=method == 'legacy_v3')
    result = dict(valid=bool(candidates and candidates[0]['observed_segments']),
        reason=diagnostics['reason'] if not candidates else 'photometric_candidate_identity_unconfirmed',
        observed_segments=candidates[0]['observed_segments'] if candidates else [],
        candidates=candidates, identity_confirmed=False, trajectory_confirmed=False,
        physical_contact_confirmed=False, last_turn_confirmed=False,
        new_layer_confirmed=False, force_command=None,
        ambiguity=dict(ambiguous=ambiguous, average_reward_gap=score_gap,
                       not_a_calibrated_probability=True),
        geometry=geometry, diagnostics=diagnostics,
        metadata=dict(method=method, parameters=dict(PARAMETERS, downstream_length_px=downstream),
            synthetic_length_override=downstream != 1536., free_seed=seed.tolist(),
            independent_width_px=width, original_targets_used=False, boundary_used=False,
            previous_frames_used=False, dark_ridges_used_as_centers=False,
            peak_gate=('legacy_prominence_and_column_extremum' if method == 'legacy_v3'
                       else 'method_peak_response_supported_mask_and_positive_polarity'),
            path_coordinate_refinement='column_quadratic_peak_in_strip_not_exported_taylor_xy',
            exported_path_uses_taylor_subpixel_coordinates=False,
            missing_samples_emitted=False, cached=False,
            detector=features['detector_metadata'], feature_ms=feature_ms,
            connection_ms=connect_ms,
            timing_scope='validation_remap_features_connect_output_no_file_decode'))
    if return_maps:
        result['ridge_maps'] = dict(bright_response=features['bright'],
            response=features['bright'], observation_mask=features['valid'], u=u, v=v,
            dark_response=features['dark'], tangent_xy=features['tangent'],
            valid_mask=features['valid'], photometric_supported=features['photometric_supported'],
            u_px=u, v_px=v)
    result['metadata']['elapsed_ms'] = (time.perf_counter()-started)*1000
    return result
