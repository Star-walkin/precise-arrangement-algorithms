"""V25 local continuation inputs, separate supervision and input-only decoder.

High and low use one physical window and one network grid. Low changes RGB
detail and its derived normalized gray only. No annotation/quality is accepted
by the input builder or decoder. Unknown supervision is never background.
"""
from __future__ import annotations

import math
import time

import numpy as np


HEIGHT, WIDTH = 96, 192
U_BOUNDS, V_BOUNDS = (-4., 8.), (-3., 3.)
STEP_WIDTHS = 1./16.
CONTRACT_VERSION = 25
NORMALIZATION_RADIUS = 16
TARGET_SIGMA_NATIVE_PX = 2.
TARGET_KEYS = ('target_distribution', 'column_valid', 'row_valid', 'slope_target', 'slope_valid')
INPUT_CHANNELS = ('red', 'green', 'blue', 'local_normalized_gray',
                  'past_seed_heatmap', 'u_div_8', 'v_div_3', 'real_pixel_valid')
DECODE_PARAMETERS = dict(anchor_u_widths=-1., anchor_half_widths=.375,
    query_consistency_half_widths=.375, minimum_window_mass=.45,
    minimum_peak_to_column_median=2., window_radius_target_sigmas=2.,
    maximum_absolute_slope=1.2, maximum_curvature_per_width=4.,
    maximum_unobserved_gap_widths=0., beam_size=16, maximum_peaks_per_column=24,
    peak_spacing_widths=.25, slope_penalty=.06, curvature_penalty=.15,
    learned_slope_penalty=.06, position_cell_widths=.0625,
    slope_cell=.10, alternative_count=3, ambiguity_average_reward_gap=.10,
    output_u_widths=(0., 4.))


def _line(points, minimum=2):
    line = np.asarray(points, dtype=np.float64)
    if line.size == 0 and minimum == 0:
        return np.empty((0, 2), np.float64)
    if line.ndim != 2 or line.shape[1] != 2 or len(line) < minimum or not np.isfinite(line).all():
        raise ValueError('Expected finite Nx2 points')
    return line


def _geometry(image_shape_hw, past_seed, width_px):
    seed = _line(past_seed)
    width = float(width_px)
    if not np.isfinite(width) or width <= 0:
        raise ValueError('Independent optical width must be finite and positive')
    h, w = image_shape_hw
    if np.any(seed < 0) or np.any(seed[:, 0] > w-1) or np.any(seed[:, 1] > h-1):
        raise ValueError('Past seed lies outside source image')
    vector = seed[-1]-seed[0]
    length = float(np.linalg.norm(vector))
    if length <= 1e-8:
        raise ValueError('Past seed endpoints must differ')
    direction = vector/length
    return dict(contract_version=CONTRACT_VERSION, shape_hw=[HEIGHT, WIDTH],
        source_shape_hw=[int(h), int(w)], origin_xy=seed[-1].tolist(),
        d_xy=direction.tolist(), n_xy=[float(-direction[1]), float(direction[0])],
        width_px=width, step_widths=STEP_WIDTHS, native_px_per_row=STEP_WIDTHS*width,
        native_px_per_column=STEP_WIDTHS*width, u_bounds_widths=list(U_BOUNDS),
        v_bounds_widths=list(V_BOUNDS), past_seed_xy=seed.tolist(), free_seed_xy=seed.tolist(),
        input_channels=list(INPUT_CHANNELS), seed_heatmap_sigma_widths=.25,
        pixel_center_mapping='xy=origin+w*((-4+(col+.5)/16)*d+(-3+(row+.5)/16)*n)',
        geometry_inputs='source shape, given past seed, independently fixed optical width only',
        target_or_quality_used=False,
        past_seed_provenance='caller must distinguish GT-conditioned prefixes from automatic/predicted history',
        past_seed_correctness_or_identity_confirmed=False,
        low_resolution=dict(downsample_factor=3, coarse_shape_hw=[32, 64],
            reduction='aligned 3x3 area average', reconstruction='bilinear pixel-center resize',
            effective_step_widths=3*STEP_WIDTHS, input_width_pixels=1/(3*STEP_WIDTHS)),
        normalization=dict(method='common_valid_weighted_box_mean_std', radius_local_px=NORMALIZATION_RADIUS,
            gray_std_floor=3/255, clip_then_divide=3.),
        valid_policy='common high real-pixel validity and fully observed coarse bilinear footprint')


def _checked_geometry(geometry):
    if geometry.get('contract_version') != CONTRACT_VERSION or geometry.get('shape_hw') != [HEIGHT, WIDTH]:
        raise ValueError('Unsupported V25 local geometry')
    width = float(geometry['width_px'])
    origin, d, n = (np.asarray(geometry[key], float) for key in ('origin_xy', 'd_xy', 'n_xy'))
    if (not np.isfinite(width) or width <= 0 or origin.shape != (2,) or d.shape != (2,) or n.shape != (2,)
            or not np.isfinite(np.r_[origin, d, n]).all()):
        raise ValueError('Invalid local geometric coefficients')
    if abs(np.linalg.norm(d)-1) > 1e-6 or not np.allclose(n, [-d[1], d[0]], atol=1e-6, rtol=0):
        raise ValueError('Local axes must be an orthonormal right-handed frame')
    if (geometry.get('step_widths') != STEP_WIDTHS or geometry.get('u_bounds_widths') != list(U_BOUNDS)
            or geometry.get('v_bounds_widths') != list(V_BOUNDS)
            or not np.isclose(geometry.get('native_px_per_row'), width*STEP_WIDTHS, atol=1e-10, rtol=0)
            or not np.isclose(geometry.get('native_px_per_column'), width*STEP_WIDTHS, atol=1e-10, rtol=0)):
        raise ValueError('Unexpected V25 local sampling contract')
    return origin, d, n, width


def local_to_image(points_colrow, geometry):
    points = np.asarray(points_colrow, float)
    if points.shape[-1:] != (2,) or not np.isfinite(points).all():
        raise ValueError('Local points must end in finite col,row coordinates')
    origin, d, n, width = _checked_geometry(geometry)
    u = U_BOUNDS[0]+(points[..., 0]+.5)*STEP_WIDTHS
    v = V_BOUNDS[0]+(points[..., 1]+.5)*STEP_WIDTHS
    return origin+width*(u[..., None]*d+v[..., None]*n)


def image_to_local(points_xy, geometry):
    points = np.asarray(points_xy, float)
    if points.shape[-1:] != (2,) or not np.isfinite(points).all():
        raise ValueError('Image points must end in finite xy coordinates')
    origin, d, n, width = _checked_geometry(geometry)
    relative = (points-origin)/width
    return np.stack([(relative@d-U_BOUNDS[0])/STEP_WIDTHS-.5,
                     (relative@n-V_BOUNDS[0])/STEP_WIDTHS-.5], axis=-1)


def _box_sum(array, radius):
    padded = np.pad(np.asarray(array, np.float64), ((radius, radius), (radius, radius)))
    integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    size = 2*radius+1
    return integral[size:, size:]-integral[:-size, size:]-integral[size:, :-size]+integral[:-size, :-size]


def _normalized_gray(rgb, valid):
    gray = rgb@np.array([.299, .587, .114])
    weight = _box_sum(valid, NORMALIZATION_RADIUS)
    mean = _box_sum(gray*valid, NORMALIZATION_RADIUS)/np.maximum(weight, 1)
    second = _box_sum(gray*gray*valid, NORMALIZATION_RADIUS)/np.maximum(weight, 1)
    deviation = np.sqrt(np.maximum(second-mean*mean, 0))
    result = np.clip((gray-mean)/np.maximum(deviation, 3/255.), -3, 3)/3
    result[~valid] = 0
    return result


def _upsample_three(array):
    """NumPy pixel-center bilinear resize with replicated coarse borders.

    Equivalent coordinates to align_corners=False/OpenCV linear resize; no CV
    runtime dependency is needed in the separate torch training environment.
    """
    value = np.asarray(array)
    h, w = value.shape[:2]
    if value.ndim not in (2, 3):
        raise ValueError('Bilinear resize expects HxW or HxWxC')
    y = np.clip((np.arange(h*3, dtype=float)+.5)/3-.5, 0, h-1)
    x = np.clip((np.arange(w*3, dtype=float)+.5)/3-.5, 0, w-1)
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    y1, x1 = np.minimum(y0+1, h-1), np.minimum(x0+1, w-1)
    fy, fx = (y-y0)[:, None], (x-x0)[None, :]
    if value.ndim == 3:
        fy, fx = fy[..., None], fx[..., None]
    result = ((1-fy)*(1-fx)*value[y0[:, None], x0[None, :]]+
              (1-fy)*fx*value[y0[:, None], x1[None, :]]+
              fy*(1-fx)*value[y1[:, None], x0[None, :]]+
              fy*fx*value[y1[:, None], x1[None, :]])
    return result.astype(np.float32)


def build_input_pair(image_rgb_uint8, past_seed, width_px):
    """Produce two float32[8,96,192] inputs with common query/coords/valid.

    Low RGB is a true aligned 3x3 area reduction of high sampled RGB followed
    by bilinear resizing. Gray normalization is independently recomputed AFTER
    degradation. Annotation/quality is intentionally absent from this API.
    """
    image = np.asarray(image_rgb_uint8)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) < 2:
        raise ValueError('Expected HxWx3 uint8 RGB image')
    geometry = _geometry(image.shape[:2], past_seed, width_px)
    rows, columns = np.indices((HEIGHT, WIDTH), dtype=np.float64)
    xy = local_to_image(np.stack([columns, rows], axis=-1), geometry)
    x, y = xy[..., 0], xy[..., 1]
    high_valid = (x >= 0) & (x <= image.shape[1]-1) & (y >= 0) & (y <= image.shape[0]-1)
    bx, by = np.clip(x, 0, image.shape[1]-1), np.clip(y, 0, image.shape[0]-1)
    x0, y0 = np.floor(bx).astype(int), np.floor(by).astype(int)
    x1, y1 = np.minimum(x0+1, image.shape[1]-1), np.minimum(y0+1, image.shape[0]-1)
    fx, fy = (bx-x0)[..., None], (by-y0)[..., None]
    rgb = ((1-fx)*(1-fy)*image[y0, x0]+fx*(1-fy)*image[y0, x1]+
           (1-fx)*fy*image[y1, x0]+fx*fy*image[y1, x1])/255.
    rgb[~high_valid] = 0
    coarse = rgb.reshape(HEIGHT//3, 3, WIDTH//3, 3, 3).mean(axis=(1, 3)).astype(np.float32)
    coarse_valid = high_valid.reshape(HEIGHT//3, 3, WIDTH//3, 3).all(axis=(1, 3))
    low_rgb = _upsample_three(coarse)
    reconstructed_valid = _upsample_three(coarse_valid.astype(np.float32))
    common_valid = high_valid & (reconstructed_valid >= 1-1e-6)
    rgb[~common_valid] = 0
    low_rgb[~common_valid] = 0
    seed = np.asarray(past_seed, float)
    distance2 = np.full((HEIGHT, WIDTH), np.inf)
    for a, b in zip(seed[:-1], seed[1:]):
        vector = b-a; norm2 = float(vector@vector)
        if norm2 <= 1e-12:
            continue
        fraction = np.clip(((xy-a)@vector)/norm2, 0, 1)
        distance2 = np.minimum(distance2, np.sum((xy-(a+fraction[..., None]*vector))**2, axis=-1))
    query = np.exp(-.5*distance2/(.25*geometry['width_px'])**2)
    query[~common_valid] = 0
    u = (U_BOUNDS[0]+(columns+.5)*STEP_WIDTHS)/8.
    v = (V_BOUNDS[0]+(rows+.5)*STEP_WIDTHS)/3.
    common = np.stack([query, u, v, common_valid]).astype(np.float32)
    result = dict(geometry=geometry, valid_mask=np.ascontiguousarray(common_valid))
    for name, color in (('high', rgb), ('low', low_rgb)):
        normalized = _normalized_gray(color, common_valid)
        result[name] = np.ascontiguousarray(np.concatenate([color.transpose(2, 0, 1), normalized[None], common]).astype(np.float32))
    if not np.array_equal(result['high'][4:], result['low'][4:]):
        raise AssertionError('Query/coordinates/valid differ between paired arms')
    return result


def _polygon_mask(polygons, geometry):
    rows, columns = np.indices((HEIGHT, WIDTH), dtype=np.float64)
    mask = np.zeros((HEIGHT, WIDTH), bool)
    guard = 1./geometry['native_px_per_row']
    for polygon in polygons:
        points = image_to_local(_line(polygon, 3), geometry)
        inside, near = np.zeros_like(mask), np.zeros_like(mask)
        for a, b in zip(points, np.roll(points, -1, axis=0)):
            crosses = (a[1] > rows) != (b[1] > rows)
            if abs(b[1]-a[1]) > 1e-12:
                cross_x = (b[0]-a[0])*(rows-a[1])/(b[1]-a[1])+a[0]
                inside ^= crosses & (columns < cross_x)
            vector = b-a; norm2 = float(vector@vector)
            if norm2 > 1e-12:
                fraction = np.clip(((columns-a[0])*vector[0]+(rows-a[1])*vector[1])/norm2, 0, 1)
                distance2 = (columns-(a[0]+fraction*vector[0]))**2+(rows-(a[1]+fraction*vector[1]))**2
                near |= distance2 <= guard*guard
        mask |= inside | near
    return mask


def _column_intersections(local_line):
    intersections = [[] for _ in range(WIDTH)]
    for a, b in zip(local_line[:-1], local_line[1:]):
        delta = b-a
        if np.linalg.norm(delta) <= 1e-9:
            continue
        if abs(delta[0]) <= 1e-9:
            column = int(round(float(a[0])))
            if 0 <= column < WIDTH and abs(a[0]-column) < 1e-8:
                intersections[column].extend([(float(a[1]), math.nan), (float(b[1]), math.nan)])
            continue
        first = max(0, int(math.ceil(min(a[0], b[0])-1e-9)))
        last = min(WIDTH-1, int(math.floor(max(a[0], b[0])+1e-9)))
        for column in range(first, last+1):
            fraction = (column-a[0])/delta[0]
            if -1e-9 <= fraction <= 1+1e-9:
                intersections[column].append((float(a[1]+fraction*delta[1]), float(delta[1]/delta[0])))
    return intersections


def _checked_valid(valid_mask):
    valid = np.asarray(valid_mask)
    if valid.shape != (HEIGHT, WIDTH) or not np.isin(valid, [0, 1]).all():
        raise ValueError('valid_mask must be finite binary 96x192')
    return valid.astype(bool)


def targets(centerline, ignorepolygons, geometry, valid):
    """V11 path_loss-compatible trusted single-valued targets on the local grid.

    Caller supplies the already quality-truncated centerline (including the
    C7 cutoff) and all original/quality exclusions. Past and future trusted
    columns may supervise; targetless tails and ambiguous columns never do.
    """
    _checked_geometry(geometry)
    valid = _checked_valid(valid)
    line = _line(centerline, 0)
    if len(line) == 1:
        raise ValueError('Centerline must be empty or have at least two points')
    forbidden = _polygon_mask(ignorepolygons, geometry)
    allowed = valid & ~forbidden
    sigma = TARGET_SIGMA_NATIVE_PX/geometry['native_px_per_row']
    vicinity = max(1, int(math.ceil(3*sigma)))
    target = np.zeros((HEIGHT, WIDTH), np.float32)
    column_valid = np.zeros(WIDTH, bool)
    slope_target, slope_valid = np.zeros(WIDTH, np.float32), np.zeros(WIDTH, bool)
    intersections = _column_intersections(image_to_local(line, geometry)) if len(line) else [[] for _ in range(WIDTH)]
    ambiguous = 0
    for column, values in enumerate(intersections):
        if not values:
            continue
        ys = np.asarray([v[0] for v in values])
        if float(np.ptp(ys)) > 1e-6:
            ambiguous += 1
            continue
        center = float(ys.mean())
        lo, hi = int(math.floor(center))-vicinity, int(math.ceil(center))+vicinity
        if lo < 0 or hi >= HEIGHT or not allowed[lo:hi+1, column].all():
            continue
        weights = np.exp(-.5*((np.arange(HEIGHT)-center)/sigma)**2)
        weights[~allowed[:, column]] = 0
        mass = float(weights.sum())
        if mass <= 1e-12:
            continue
        target[:, column] = weights/mass
        column_valid[column] = True
        slopes = np.asarray([v[1] for v in values])
        if np.isfinite(slopes).all():
            slope_target[column] = float(slopes.mean())
            slope_valid[column] = True
    return dict(target_distribution=target, column_valid=column_valid,
        row_valid=allowed & column_valid[None], slope_target=slope_target, slope_valid=slope_valid,
        diagnostics=dict(supervised_columns=int(column_valid.sum()), ambiguous_columns=ambiguous,
            unknown_columns=int((~column_valid).sum()), sigma_native_px=TARGET_SIGMA_NATIVE_PX,
            sigma_local_rows=float(sigma), target_neighborhood_guard_rows=vicinity,
            ignored_pixel_count=int(forbidden.sum()), unknown_is_background=False,
            null_class_trained=False, no_visibility_targets=True, targets_never_modify_input=True))


def _column_nodes(probability, valid, column, radius):
    values = np.where(valid[:, column], probability[:, column], 0.)
    cumulative = np.r_[0., np.cumsum(values)]
    rows = np.arange(HEIGHT)
    lo, hi = np.maximum(0, rows-radius), np.minimum(HEIGHT, rows+radius+1)
    mass = cumulative[hi]-cumulative[lo]
    counts = np.r_[0, np.cumsum(valid[:, column])]
    full_window = counts[hi]-counts[lo] == 2*radius+1
    maxima = np.flatnonzero((mass[1:-1] >= mass[:-2]) & (mass[1:-1] > mass[2:]))+1
    median = float(np.median(values[valid[:, column]])) if valid[:, column].any() else 0.
    nodes = []
    for row in sorted(maxima, key=lambda value: (-float(mass[value]), int(value))):
        if not full_window[row] or mass[row] < DECODE_PARAMETERS['minimum_window_mass']:
            continue
        if float(values[lo[row]:hi[row]].max()) < max(1e-10, median*DECODE_PARAMETERS['minimum_peak_to_column_median']):
            continue
        if any(abs(row-old['row'])*STEP_WIDTHS < DECODE_PARAMETERS['peak_spacing_widths'] for old in nodes):
            continue
        local = values[lo[row]:hi[row]]
        center = float(np.sum(np.arange(lo[row], hi[row])*local)/max(float(local.sum()), 1e-12))
        nodes.append(dict(column=column, row=center, mass=float(mass[row])))
        if len(nodes) >= DECODE_PARAMETERS['maximum_peaks_per_column']:
            break
    return nodes


def _new_segment(path, geometry):
    if len(path) < 2:
        return []
    points = np.asarray([[n['column'], n['row']] for n in path], float)
    u = U_BOUNDS[0]+(points[:, 0]+.5)*STEP_WIDTHS
    keep = u >= 0
    selected = points[keep]
    if len(selected) == 0:
        return []
    first = int(np.flatnonzero(keep)[0])
    if first > 0:
        fraction = -u[first-1]/(u[first]-u[first-1])
        boundary = points[first-1]+fraction*(points[first]-points[first-1])
        selected = np.vstack([boundary, selected])
    if len(selected) < 2:
        return []
    return [local_to_image(selected, geometry).tolist()]


def decode_local(probability, slope, geometry, valid):
    """Fixed query-connected candidates only; no labels, quality, gap guesses.

    A known-query consistency corridor is enforced before u=0. Each output
    edge joins adjacent genuinely supported columns. When a column has no
    reliable connected node, that candidate ends; no missing-column bridge
    or downstream restart is allowed. Only new u in [0,4] is exported.
    """
    started = time.perf_counter()
    _checked_geometry(geometry)
    valid = _checked_valid(valid)
    p = np.asarray(probability, float)
    if p.shape != (HEIGHT, WIDTH) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1+1e-6):
        raise ValueError('Expected finite probability[96,192] in [0,1]')
    if np.any(p.sum(axis=0) > 1+1e-4):
        raise ValueError('Probability must be a row distribution, not independent sigmoids')
    if slope is not None:
        slope = np.asarray(slope, float)
        if slope.shape != (WIDTH,) or not np.isfinite(slope).all():
            raise ValueError('Slope must be a finite 192-vector or None')
    query = _line(geometry.get('past_seed_xy', geometry.get('free_seed_xy')))
    query_local = image_to_local(query, geometry)
    query_intersections = _column_intersections(query_local)
    anchor_float = (DECODE_PARAMETERS['anchor_u_widths']-U_BOUNDS[0])/STEP_WIDTHS-.5
    anchor = int(math.floor(anchor_float+.5))
    column_u = U_BOUNDS[0]+(np.arange(WIDTH)+.5)*STEP_WIDTHS
    last = int(np.flatnonzero(column_u <= DECODE_PARAMETERS['output_u_widths'][1])[-1])
    sigma = TARGET_SIGMA_NATIVE_PX/geometry['native_px_per_row']
    radius = max(1, int(math.ceil(DECODE_PARAMETERS['window_radius_target_sigmas']*sigma)))
    diagnostics = dict(parameters=dict(DECODE_PARAMETERS), anchor_column=anchor,
        anchor_requested_u_widths=DECODE_PARAMETERS['anchor_u_widths'], anchor_actual_u_widths=float(column_u[anchor]),
        window_radius_rows=radius, target_sigma_native_px=TARGET_SIGMA_NATIVE_PX,
        output_last_observed_u_widths=None, quality_or_targets_used=False,
        probability_was_renormalized=False, interpolated_unobserved_gap_points_emitted=False,
        probability_is_visibility_confirmation=False, unreliable_connection_policy='stop_no_gap_bridge')

    def result(reason, candidates=None):
        candidates = candidates or []
        segments = candidates[0]['segments'] if candidates else []
        diagnostics['ambiguous'] = bool(len(candidates) > 1 and
            abs(candidates[0]['average_reward']-candidates[1]['average_reward']) < DECODE_PARAMETERS['ambiguity_average_reward_gap'])
        diagnostics['elapsed_ms'] = (time.perf_counter()-started)*1000
        diagnostics['stopped_on_unreliable_connection'] = not candidates or candidates[0]['stop_reason'] != 'fixed_four_width_horizon'
        if candidates:
            diagnostics['output_last_observed_u_widths'] = candidates[0]['last_u_widths']
        return dict(segments=segments, observed_segments=segments, candidates=candidates,
            valid=bool(segments), reason=reason, identity_confirmed=False,
            physical_contact_confirmed=False, new_layer_confirmed=False,
            force_command=None, diagnostics=diagnostics)

    values = query_intersections[anchor]
    if not values or np.ptp([value[0] for value in values]) > 1e-6:
        return result('query_does_not_uniquely_support_anchor')
    anchor_row = float(np.mean([value[0] for value in values]))
    slopes = np.asarray([value[1] for value in values])
    if not np.isfinite(slopes).all():
        return result('query_anchor_direction_unknown')
    anchor_slope = float(slopes.mean())
    if abs(anchor_slope) > DECODE_PARAMETERS['maximum_absolute_slope']:
        return result('query_anchor_outside_local_slope_contract')
    diagnostics['anchor_query_row'] = anchor_row
    node_lists = {col: _column_nodes(p, valid, col, radius) for col in range(anchor, last+1)}
    initial = [n for n in node_lists[anchor] if abs(n['row']-anchor_row)*STEP_WIDTHS <= DECODE_PARAMETERS['anchor_half_widths']]
    diagnostics['candidate_columns'] = sum(bool(nodes) for nodes in node_lists.values())
    diagnostics['query_anchored_initial_candidates'] = len(initial)
    beam = [dict(path=(node,), score=node['mass'], slope=anchor_slope) for node in initial]
    if not beam:
        return result('no_concentrated_query_anchored_distribution')
    finished = []
    for column in range(anchor+1, last+1):
        nodes = node_lists[column]
        if column_u[column] < 0:
            known = query_intersections[column]
            if not known or np.ptp([v[0] for v in known]) > 1e-6:
                nodes = []
            else:
                expected = float(np.mean([v[0] for v in known]))
                nodes = [n for n in nodes if abs(n['row']-expected)*STEP_WIDTHS <= DECODE_PARAMETERS['query_consistency_half_widths']]
        extensions = []
        for state in beam:
            accepted = False
            for node in nodes:
                measured = node['row']-state['path'][-1]['row']
                curvature = abs(measured-state['slope'])/STEP_WIDTHS
                if abs(measured) > DECODE_PARAMETERS['maximum_absolute_slope'] or curvature > DECODE_PARAMETERS['maximum_curvature_per_width']+1e-9:
                    continue
                reward = node['mass']-DECODE_PARAMETERS['slope_penalty']*abs(measured)
                reward -= DECODE_PARAMETERS['curvature_penalty']*abs(measured-state['slope'])
                if slope is not None:
                    reward -= DECODE_PARAMETERS['learned_slope_penalty']*abs(measured-float(slope[column]))
                extensions.append(dict(path=state['path']+(node,), score=state['score']+reward, slope=measured))
                accepted = True
            if not accepted:
                finished.append(dict(state, stop_reason='no_reliable_connected_next_column'))
        if not extensions:
            beam = []
            break
        occupied, beam = set(), []
        for state in sorted(extensions, key=lambda s: s['score'], reverse=True):
            cell = (round(state['path'][-1]['row']*STEP_WIDTHS/DECODE_PARAMETERS['position_cell_widths']),
                    round(state['slope']/DECODE_PARAMETERS['slope_cell']))
            if cell in occupied:
                continue
            occupied.add(cell); beam.append(state)
            if len(beam) == DECODE_PARAMETERS['beam_size']:
                break
    finished.extend(dict(state, stop_reason='fixed_four_width_horizon') for state in beam)
    ranked = []
    for state in sorted(finished, key=lambda s: s['score'], reverse=True):
        segments = _new_segment(state['path'], geometry)
        if not segments:
            continue
        end = np.asarray(segments[-1][-1])
        if any(np.linalg.norm(end-np.asarray(old['segments'][-1][-1])) < geometry['width_px']*.25 for old in ranked):
            continue
        final_col = state['path'][-1]['column']
        ranked.append(dict(segments=segments, score=float(state['score']),
            average_reward=float(state['score']/len(state['path'])),
            observed_columns=len(state['path']), unobserved_columns=0,
            last_u_widths=float(column_u[final_col]), stop_reason=state['stop_reason'],
            identity_confirmed=False))
        if len(ranked) == DECODE_PARAMETERS['alternative_count']:
            break
    return result('local_continuation_candidate_identity_unconfirmed' if ranked else 'no_reliable_new_segment', ranked)
