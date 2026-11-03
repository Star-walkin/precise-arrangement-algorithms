"""Offline photometric ridge *candidates*, not rope-instance measurements.

The three fixed baselines use only the current image, an externally measured
free-span optical width, and an optional input-only ROI. No target polylines,
labels, model weights, previous frames, flange attraction or hardware are used.

``steger_taylor`` uses Gaussian derivatives, the largest-absolute Hessian
eigenvalue's normal and a normal Taylor extremum. It does NOT implement Steger's
asymmetric line-profile bias correction, line linking, or width correction.
``frangi_inspired`` is a two-dimensional multiscale eigenvalue-ratio response,
not a promise of parity with scikit-image. Bright ridges and dark grooves stay
separate: either may be a strand highlight, rope-body center, or inter-rope gap.
Subpixel numerical localization is not subpixel physical accuracy.

Reference: Steger, https://doi.org/10.1109/34.659930 ;
https://scikit-image.org/docs/0.25.x/api/skimage.filters.html#skimage.filters.frangi
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np


METHODS = ("normalized_hessian", "steger_taylor", "frangi_inspired")
DEFAULT_WIDTH_RATIOS = (.16, .28, .42)
RESPONSE_THRESHOLD = .12


def _gray(image):
    array = np.asarray(image)
    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] != 3):
        raise ValueError("image must be HxW grayscale or HxWx3 BGR")
    if min(array.shape[:2]) < 8 or not np.issubdtype(array.dtype, np.number):
        raise ValueError("image must be numeric and at least 8x8")
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 255:
        raise ValueError("image must contain finite values in [0,255]")
    array = array.astype(np.float32)
    return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY) if array.ndim == 3 else array


def _roi(roi, shape):
    if roi is None:
        return (0, 0, shape[1], shape[0])
    values = np.asarray(roi, float)
    if values.shape != (4,) or not np.isfinite(values).all() or np.any(values != np.round(values)):
        raise ValueError("roi must be integer [x0,y0,x1,y1], half-open")
    x0, y0, x1, y1 = map(int, values)
    if not (0 <= x0 < x1 <= shape[1] and 0 <= y0 < y1 <= shape[0]):
        raise ValueError("roi must be nonempty and inside image")
    return x0, y0, x1, y1


def _scales(width_px, sigmas_px):
    if not np.isfinite(width_px) or width_px <= 0:
        raise ValueError("width_px must be a positive independent image-space width")
    values = (np.clip(np.asarray(DEFAULT_WIDTH_RATIOS) * width_px, .8, 12.)
              if sigmas_px is None else np.asarray(sigmas_px, float))
    if values.ndim != 1 or not 1 <= len(values) <= 4 or not np.isfinite(values).all():
        raise ValueError("provide one to four finite sigmas")
    if np.any(values < .5) or np.any(values > 32):
        raise ValueError("sigmas must be in [0.5,32] image pixels")
    return tuple(float(value) for value in np.unique(values))


def _derivative_kernels(sigma):
    radius = int(math.ceil(4 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    gaussian = np.exp(-.5 * (x / sigma) ** 2)
    gaussian /= gaussian.sum()
    # OpenCV applies correlation. Positive first moment therefore has the
    # correct derivative sign for an image increasing in x/y.
    first = x * gaussian / sigma ** 2
    first /= np.sum(x * first)
    second = (x*x / sigma**4 - 1 / sigma**2) * gaussian
    second -= second.sum() * gaussian
    second /= .5 * np.sum(x*x * second)
    return tuple(value.astype(np.float32) for value in (gaussian, first, second))


def _derivatives(image, sigma, exact):
    if exact:
        g, first, second = _derivative_kernels(sigma)
        options = dict(ddepth=cv2.CV_32F, borderType=cv2.BORDER_REFLECT_101)
        gx = cv2.sepFilter2D(image, kernelX=first, kernelY=g, **options)
        gy = cv2.sepFilter2D(image, kernelX=g, kernelY=first, **options)
        xx = cv2.sepFilter2D(image, kernelX=second, kernelY=g, **options)
        yy = cv2.sepFilter2D(image, kernelX=g, kernelY=second, **options)
        xy = cv2.sepFilter2D(image, kernelX=first, kernelY=first, **options)
    else:
        smooth = cv2.GaussianBlur(image, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
        gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3, scale=.125)
        gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3, scale=.125)
        xx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=3, scale=.25)
        yy = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=3, scale=.25)
        xy = cv2.Sobel(smooth, cv2.CV_32F, 1, 1, ksize=3, scale=.25)
    return gx, gy, xx, yy, xy


def _hessian_normal(xx, yy, xy):
    delta = np.sqrt(np.maximum((xx-yy)**2 + 4*xy*xy, 0))
    plus = .5 * (xx+yy+delta)
    minus = .5 * (xx+yy-delta)
    plus_normal = np.abs(plus) >= np.abs(minus)
    dominant = np.where(plus_normal, plus, minus)
    secondary = np.where(plus_normal, minus, plus)
    angle = .5 * np.arctan2(2*xy, xx-yy)
    angle = np.where(plus_normal, angle, angle + np.pi/2)
    normal_x, normal_y = np.cos(angle), np.sin(angle)
    tangent_x, tangent_y = -normal_y, normal_x
    flip = (tangent_x < -1e-6) | ((np.abs(tangent_x) <= 1e-6) & (tangent_y < 0))
    tangent_x = np.where(flip, -tangent_x, tangent_x)
    tangent_y = np.where(flip, -tangent_y, tangent_y)
    return dominant, secondary, normal_x, normal_y, tangent_x, tangent_y


def extract_ridge_candidates(image, *, method, width_px, roi=None,
                             sigmas_px=None, polarity="both"):
    """Return native-pixel ROI maps and unlinked subpixel ridge candidates.

    Dense maps have shape ``(roi_height, roi_width)``; ``tangent_xy`` ends in
    (2,) and is unoriented (canonical sign, NOT motion direction). ``response``
    is the selected photometric ridge response, not a probability. Bright and
    dark maps are always exported separately. ``supported_mask`` is a thinned
    ridge-localization mask, not a semantic visibility mask. The subpixel
    coordinates are FULL-IMAGE pixel-center coordinates, one per mask pixel in
    row-major order. They are deliberately NOT linked into a rope trajectory.

    Timing includes gray conversion, local normalization and all convolutions,
    aggregation and point construction. It excludes external file decoding.
    Nothing is cached; calls do not change OpenCV global thread settings.
    """
    started = time.perf_counter()
    if method not in METHODS or polarity not in ("bright", "dark", "both"):
        raise ValueError("unknown method or polarity")
    gray = _gray(image)
    box = _roi(roi, gray.shape)
    scales = _scales(float(width_px), sigmas_px)
    used_scales = (scales[len(scales)//2],) if method == "normalized_hessian" else scales
    x0, y0, x1, y1 = box
    normalization_sigma = max(4., min(128., 2 * float(width_px)))
    # Include real image context outside ROI; do not synthesize an ROI edge.
    halo = int(math.ceil(4*normalization_sigma + 4*max(used_scales) + 3))
    hx0, hy0 = max(0, x0-halo), max(0, y0-halo)
    hx1, hy1 = min(gray.shape[1], x1+halo), min(gray.shape[0], y1+halo)
    crop = gray[hy0:hy1, hx0:hx1]
    mean = cv2.GaussianBlur(crop, (0, 0), normalization_sigma)
    variance = cv2.GaussianBlur(crop*crop, (0, 0), normalization_sigma) - mean*mean
    std = np.sqrt(np.maximum(variance, 0))
    normalized = (crop-mean) / np.maximum(std, 3.)
    target = np.s_[y0-hy0:y1-hy0, x0-hx0:x1-hx0]
    shape = (y1-y0, x1-x0)
    grid_y, grid_x = np.indices(shape, dtype=np.float32)
    raw = crop[target]
    original_y, original_x = grid_y+y0, grid_x+x0
    valid = ((std[target] >= 2.) & (raw > 1.) & (raw < 254.) &
             (original_x >= 3) & (original_x < gray.shape[1]-3) &
             (original_y >= 3) & (original_y < gray.shape[0]-3))
    # Rejected saturated/flat pixels never become observations merely because
    # normalization or broad Gaussian support has high response nearby.
    bright = np.zeros(shape, np.float32)
    dark = np.zeros(shape, np.float32)
    best = np.zeros(shape, np.float32)
    peak_best = np.zeros(shape, np.float32)
    tangent = np.zeros(shape+(2,), np.float32)
    peak_tangent = np.zeros(shape+(2,), np.float32)
    offset = np.zeros(shape+(2,), np.float32)
    chosen_scale = np.zeros(shape, np.float32)
    peak_scale = np.zeros(shape, np.float32)
    selected_polarity = np.zeros(shape, np.int8)
    peak_polarity = np.zeros(shape, np.int8)
    for sigma in used_scales:
        fields = _derivatives(normalized, sigma, method != "normalized_hessian")
        gx, gy, xx, yy, xy = (field[target] for field in fields)
        lam, other, nx, ny, tx, ty = _hessian_normal(xx, yy, xy)
        strength = np.abs(lam) * sigma*sigma
        ratio = np.abs(other) / np.maximum(np.abs(lam), 1e-8)
        if method == "frangi_inspired":
            structure = (lam*lam + other*other) * sigma**4
            score = np.exp(-ratio*ratio/(2*.5**2)) * (1-np.exp(-structure/(2*.5**2)))
        else:
            anisotropy = np.clip(1-ratio, 0, 1)
            score = strength / (strength+.15) * anisotropy
        score = np.where(valid & (strength >= .015), score, 0).astype(np.float32)
        bright_at_scale = np.where(lam < 0, score, 0)
        dark_at_scale = np.where(lam > 0, score, 0)
        np.maximum(bright, bright_at_scale, out=bright)
        np.maximum(dark, dark_at_scale, out=dark)
        selected = (bright_at_scale if polarity == "bright" else dark_at_scale
                    if polarity == "dark" else score)
        sign = np.where(lam < 0, 1, -1).astype(np.int8)
        better = selected > best
        best[better] = selected[better]
        tangent[better, 0], tangent[better, 1] = tx[better], ty[better]
        chosen_scale[better] = sigma
        selected_polarity[better] = sign[better]
        safe_lam = np.where(np.abs(lam) > 1e-8, lam, 1.)
        step = -(gx*nx + gy*ny) / safe_lam
        dx, dy = step*nx, step*ny
        if method == "steger_taylor":
            peak = (np.abs(dx) <= .5) & (np.abs(dy) <= .5) & (np.abs(lam) > 1e-8)
        else:
            # Photometric NMS only. Do not close masks or connect gaps.
            a = cv2.remap(selected, grid_x+nx, grid_y+ny, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)
            b = cv2.remap(selected, grid_x-nx, grid_y-ny, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)
            peak = (selected >= a) & (selected >= b) & ((selected > a) | (selected > b))
            dx, dy = np.zeros_like(dx), np.zeros_like(dy)
        peak &= valid & (selected >= RESPONSE_THRESHOLD)
        peak_better = peak & (selected > peak_best)
        peak_best[peak_better] = selected[peak_better]
        peak_tangent[peak_better, 0], peak_tangent[peak_better, 1] = tx[peak_better], ty[peak_better]
        offset[peak_better, 0], offset[peak_better, 1] = dx[peak_better], dy[peak_better]
        peak_scale[peak_better] = sigma
        peak_polarity[peak_better] = sign[peak_better]
    supported = peak_best > 0
    tangent[supported] = peak_tangent[supported]
    chosen_scale[supported] = peak_scale[supported]
    selected_polarity[supported] = peak_polarity[supported]
    ys, xs = np.nonzero(supported)
    subpixel = np.column_stack((xs+x0, ys+y0)).astype(np.float32) + offset[supported]
    return dict(response=best, bright_response=bright, dark_response=dark,
                tangent_xy=tangent, supported_mask=supported,
                observation_mask=valid, subpixel_xy=subpixel,
                subpixel_polarity=selected_polarity[supported],
                polarity_map=selected_polarity, sigma_map=chosen_scale,
                peak_response=peak_best,
                metadata=dict(method=method, roi_xyxy=list(box),
                              output_shape_hw=list(shape), input_width_px=float(width_px),
                              requested_sigmas_px=list(scales), used_sigmas_px=list(used_scales),
                              normalization_sigma_px=normalization_sigma,
                              polarity=polarity, response_is_probability=False,
                              identity_confirmed=False, connected_trajectory=False,
                              physical_contact_confirmed=False, subpixel_bias_corrected=False,
                              roi_uses_target_labels=False, cached=False,
                              subpixel_coordinates="full_image_pixel_centers_xy",
                              timing_scope="grayscale_normalization_filters_aggregation_points_no_file_decode",
                              elapsed_ms=(time.perf_counter()-started)*1000,
                              candidate_pixel_count=int(supported.sum())))
