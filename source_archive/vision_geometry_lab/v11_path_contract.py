"""NumPy-only, seed/width-normalized V11 input and separate path supervision.

Input geometry never consumes wound labels or quality polygons. The target API
is separate: absent annotation, ambiguous same-column paths, padding and ignored
regions are UNKNOWN, never negative visibility examples. This representation
requires the trusted target to be locally single-valued in the seed direction.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np


HEIGHT, WIDTH = 256, 512
U_BOUNDS = (-8., 88.)
V_BOUNDS = (-24., 24.)
STEP_WIDTHS = .1875
INPUT_CHANNELS = ('red', 'green', 'blue', 'local_normalized_gray',
                  'free_seed_heatmap', 'u_div_88', 'v_div_24', 'real_pixel_valid')
NORMALIZATION_RADIUS = 16
CONTRACT_VERSION = 1


def array_sha256(array):
    array = np.ascontiguousarray(array)
    description = json.dumps(dict(shape=list(array.shape), dtype=str(array.dtype)),
                             sort_keys=True, separators=(',', ':')).encode('ascii')
    digest = hashlib.sha256(description)
    digest.update(array.tobytes())
    return digest.hexdigest()


def _line(points, minimum=2):
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0 and minimum == 0:
        return np.empty((0, 2), np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) < minimum or not np.isfinite(array).all():
        raise ValueError('points must be a finite Nx2 array')
    return array


def _geometry(image_shape, seed, width):
    if not math.isfinite(width) or width <= 0:
        raise ValueError('width_px must be positive, finite and independently measured')
    seed = _line(seed)
    height, image_width = image_shape
    if np.any(seed < 0) or np.any(seed[:, 0] > image_width-1) or np.any(seed[:, 1] > height-1):
        raise ValueError('free seed lies outside the source image')
    direction = seed[-1]-seed[0]
    magnitude = float(np.linalg.norm(direction))
    if magnitude <= 1e-8:
        raise ValueError('free seed endpoints must differ')
    d = direction/magnitude
    n = np.array([-d[1], d[0]])
    return dict(contract_version=CONTRACT_VERSION, shape_hw=[HEIGHT, WIDTH],
        source_shape_hw=[int(height), int(image_width)], origin_xy=seed[-1].tolist(),
        d_xy=d.tolist(), n_xy=n.tolist(), width_px=width,
        u_bounds_widths=list(U_BOUNDS), v_bounds_widths=list(V_BOUNDS),
        step_widths=STEP_WIDTHS, native_px_per_row=STEP_WIDTHS*width,
        native_px_per_column=STEP_WIDTHS*width, free_seed_xy=seed.tolist(),
        u_first_widths=U_BOUNDS[0]+STEP_WIDTHS/2,
        v_first_widths=V_BOUNDS[0]+STEP_WIDTHS/2,
        pixel_center_mapping='xy=origin+w*((-8+(col+.5)*.1875)*d+(-24+(row+.5)*.1875)*n)',
        geometry_inputs='source image shape, free seed and independent optical width only',
        target_or_quality_used=False, input_channels=list(INPUT_CHANNELS),
        normalization=dict(method='valid_weighted_box_mean_std', radius_strip_px=16,
                           gray_std_floor=3/255, clip_then_divide=3.),
        seed_heatmap_sigma_widths=.25)


def _checked_geometry(geometry):
    if geometry.get('contract_version') != CONTRACT_VERSION or geometry.get('shape_hw') != [HEIGHT, WIDTH]:
        raise ValueError('unsupported V11 geometry')
    width = float(geometry['width_px'])
    origin = np.asarray(geometry['origin_xy'], np.float64)
    d, n = (np.asarray(geometry[key], np.float64) for key in ('d_xy', 'n_xy'))
    if (not math.isfinite(width) or width <= 0 or origin.shape != (2,) or d.shape != (2,)
            or n.shape != (2,) or not np.isfinite(np.r_[origin, d, n]).all()):
        raise ValueError('invalid geometric coefficients')
    if abs(np.linalg.norm(d)-1) > 1e-6 or not np.allclose(n, [-d[1], d[0]], atol=1e-6):
        raise ValueError('geometry axes must be an orthonormal right-handed frame')
    if geometry.get('step_widths') != STEP_WIDTHS:
        raise ValueError('unexpected V11 sampling step')
    return origin, d, n, width


def strip_to_image(points_xy, geometry):
    """Floating (column,row) pixel-center indices to full-image xy."""
    points = np.asarray(points_xy, np.float64)
    if points.shape[-1:] != (2,) or not np.isfinite(points).all():
        raise ValueError('strip points must end in finite xy coordinates')
    origin, d, n, width = _checked_geometry(geometry)
    u = U_BOUNDS[0]+(points[..., 0]+.5)*STEP_WIDTHS
    v = V_BOUNDS[0]+(points[..., 1]+.5)*STEP_WIDTHS
    return origin+width*(u[..., None]*d+v[..., None]*n)


def image_to_strip(points_xy, geometry):
    """Full-image xy to floating (column,row) pixel-center indices."""
    points = np.asarray(points_xy, np.float64)
    if points.shape[-1:] != (2,) or not np.isfinite(points).all():
        raise ValueError('image points must end in finite xy coordinates')
    origin, d, n, width = _checked_geometry(geometry)
    relative = (points-origin)/width
    return np.stack([(relative@d-U_BOUNDS[0])/STEP_WIDTHS-.5,
                     (relative@n-V_BOUNDS[0])/STEP_WIDTHS-.5], axis=-1)


def _box_sum(array, radius):
    padded = np.pad(np.asarray(array, np.float64), ((radius, radius), (radius, radius)))
    integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    size = 2*radius+1
    return integral[size:, size:]-integral[:-size, size:]-integral[size:, :-size]+integral[:-size, :-size]


def build_input(image_rgb_uint8, free_seed, width_px):
    """Return input[8,256,512], geometry and boolean valid_mask.

    Bilinear sampling uses full-image integer pixel centers. RGB/normalized
    gray/seed channels are zero outside the image. Coordinates remain geometric
    metadata channels there, with valid=0. Mask-weighted normalization cannot
    turn zero padding next to a constant image into a bright/dark ridge.
    """
    image = np.asarray(image_rgb_uint8)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) < 2:
        raise ValueError('image_rgb_uint8 must be HxWx3 uint8, dimensions >=2')
    geometry = _geometry(image.shape[:2], free_seed, float(width_px))
    rows, columns = np.indices((HEIGHT, WIDTH), dtype=np.float64)
    xy = strip_to_image(np.stack([columns, rows], axis=-1), geometry)
    x, y = xy[..., 0], xy[..., 1]
    valid = (x >= 0) & (x <= image.shape[1]-1) & (y >= 0) & (y <= image.shape[0]-1)
    # Clip integer indexing only; clipped samples remain zero/invalid outside.
    bounded_x = np.clip(x, 0, image.shape[1]-1)
    bounded_y = np.clip(y, 0, image.shape[0]-1)
    x0, y0 = np.floor(bounded_x).astype(int), np.floor(bounded_y).astype(int)
    x1, y1 = np.minimum(x0+1, image.shape[1]-1), np.minimum(y0+1, image.shape[0]-1)
    fx, fy = (bounded_x-x0)[..., None], (bounded_y-y0)[..., None]
    rgb = ((1-fx)*(1-fy)*image[y0, x0] + fx*(1-fy)*image[y0, x1] +
           (1-fx)*fy*image[y1, x0] + fx*fy*image[y1, x1])/255.
    rgb[~valid] = 0
    gray = rgb@np.array([.299, .587, .114])
    weight = _box_sum(valid, NORMALIZATION_RADIUS)
    mean = _box_sum(gray, NORMALIZATION_RADIUS)/np.maximum(weight, 1)
    second = _box_sum(gray*gray, NORMALIZATION_RADIUS)/np.maximum(weight, 1)
    deviation = np.sqrt(np.maximum(second-mean*mean, 0))
    normalized = np.clip((gray-mean)/np.maximum(deviation, 3/255.), -3, 3)/3
    normalized[~valid] = 0
    seed = np.asarray(geometry['free_seed_xy'])
    distance2 = np.full((HEIGHT, WIDTH), np.inf)
    for a, b in zip(seed[:-1], seed[1:]):
        vector = b-a
        norm2 = float(vector@vector)
        if norm2 <= 1e-12:
            continue
        fraction = np.clip(((xy-a)@vector)/norm2, 0, 1)
        closest = a+fraction[..., None]*vector
        distance2 = np.minimum(distance2, np.sum((xy-closest)**2, axis=-1))
    heatmap = np.exp(-.5*distance2/(.25*geometry['width_px'])**2)
    heatmap[~valid] = 0
    u = (U_BOUNDS[0]+(columns+.5)*STEP_WIDTHS)/88.
    v = (V_BOUNDS[0]+(rows+.5)*STEP_WIDTHS)/24.
    tensor = np.concatenate([rgb.transpose(2, 0, 1), normalized[None], heatmap[None],
                             u[None], v[None], valid[None]], axis=0).astype(np.float32)
    return dict(input=np.ascontiguousarray(tensor), geometry=geometry,
                valid_mask=np.ascontiguousarray(valid))


def _polygon_mask(polygons, geometry):
    rows, columns = np.indices((HEIGHT, WIDTH), dtype=np.float64)
    mask = np.zeros((HEIGHT, WIDTH), bool)
    for polygon in polygons:
        source = _line(polygon, 3)
        points = image_to_strip(source, geometry)
        inside = np.zeros((HEIGHT, WIDTH), bool)
        near_edge = np.zeros_like(inside)
        for a, b in zip(points, np.roll(points, -1, axis=0)):
            crosses = ((a[1] > rows) != (b[1] > rows))
            if abs(b[1]-a[1]) > 1e-12:
                cross_x = (b[0]-a[0])*(rows-a[1])/(b[1]-a[1])+a[0]
                inside ^= crosses & (columns < cross_x)
            dx, dy = b-a
            norm2 = float(dx*dx+dy*dy)
            if norm2 > 1e-12:
                along = np.clip(((columns-a[0])*dx+(rows-a[1])*dy)/norm2, 0, 1)
                distance2 = (columns-(a[0]+along*dx))**2+(rows-(a[1]+along*dy))**2
                # One strip-pixel conservative boundary guard; uncertain
                # precision support may not leak through resampling cells.
                near_edge |= distance2 <= 1.
        mask |= inside | near_edge
    return mask


def build_targets(geometry, centerline, ignore_polygons, valid_mask):
    """Unique trusted intersection supervision; unknown columns stay all-zero.

    Caller must pass the quality-confirmed truncated centerline (e.g. C07/C08
    through C7) and all original/quality ignore polygons. The input builder is
    not called or changed here. Distribution sigma is max(1,2/native-row-step).
    Targets within 3 sigma of an ignored/padded row, non-single-valued columns
    and targetless tail columns receive NO center, slope or background loss.
    """
    _checked_geometry(geometry)
    valid = np.asarray(valid_mask, bool)
    if valid.shape != (HEIGHT, WIDTH):
        raise ValueError('valid_mask must be 256x512')
    line = _line(centerline, 0)
    if len(line) == 1:
        raise ValueError('centerline needs at least two points or must be empty')
    forbidden = _polygon_mask(ignore_polygons, geometry)
    allowed = valid & ~forbidden
    sigma = max(1., 2./float(geometry['native_px_per_row']))
    vicinity = int(math.ceil(3*sigma))
    target = np.zeros((HEIGHT, WIDTH), np.float32)
    column_valid = np.zeros(WIDTH, bool)
    slope_target = np.zeros(WIDTH, np.float32)
    slope_valid = np.zeros(WIDTH, bool)
    center_rows = np.full(WIDTH, np.nan, np.float64)
    ambiguous_columns = 0
    intersections = [[] for _ in range(WIDTH)]
    if len(line) >= 2:
        points = image_to_strip(line, geometry)
        for a, b in zip(points[:-1], points[1:]):
            dx, dy = b-a
            if np.linalg.norm(b-a) <= 1e-9:
                continue
            if abs(dx) <= 1e-9:
                nearest = int(round(float(a[0])))
                if 0 <= nearest < WIDTH and abs(a[0]-nearest) < 1e-8:
                    intersections[nearest].extend([(float(a[1]), math.nan), (float(b[1]), math.nan)])
                continue
            lo = max(0, int(math.ceil(min(a[0], b[0])-1e-9)))
            hi = min(WIDTH-1, int(math.floor(max(a[0], b[0])+1e-9)))
            for column in range(lo, hi+1):
                fraction = (column-a[0])/dx
                if -1e-9 <= fraction <= 1+1e-9:
                    intersections[column].append((float(a[1]+fraction*dy), float(dy/dx)))
    for column, values in enumerate(intersections):
        if not values:
            continue
        ys = np.asarray([value[0] for value in values])
        if float(np.ptp(ys)) > 1e-6:
            ambiguous_columns += 1
            continue
        center = float(np.mean(ys))
        low, high = int(math.floor(center))-vicinity, int(math.ceil(center))+vicinity
        if low < 0 or high >= HEIGHT or not allowed[low:high+1, column].all():
            continue
        weights = np.exp(-.5*((np.arange(HEIGHT)-center)/sigma)**2)
        weights[~allowed[:, column]] = 0
        total = float(weights.sum())
        if total <= 1e-12:
            continue
        target[:, column] = weights/total
        column_valid[column] = True
        center_rows[column] = center
        slopes = np.asarray([value[1] for value in values])
        if np.isfinite(slopes).all():
            slope_target[column] = float(np.mean(slopes))
            slope_valid[column] = True
    row_valid = allowed & column_valid[None]
    return dict(target_distribution=target, column_valid=column_valid, row_valid=row_valid,
        slope_target=slope_target, slope_valid=slope_valid,
        diagnostics=dict(supervised_columns=int(column_valid.sum()),
            ambiguous_columns=ambiguous_columns, unknown_columns=int((~column_valid).sum()),
            sigma_strip_rows=sigma, target_neighborhood_guard_rows=vicinity,
            ignored_pixel_count=int(forbidden.sum()), unknown_is_background=False,
            no_visibility_targets=True, targets_never_modify_input=True))
