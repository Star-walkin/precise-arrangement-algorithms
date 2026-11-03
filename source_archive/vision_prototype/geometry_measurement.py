"""Image geometry helpers; no video names, human annotations or device I/O."""
from __future__ import annotations

from collections import deque
from dataclasses import replace
import math

import cv2
import numpy as np


def periodic_texture_score(gray, pitch_px, roi):
    """Same Sobel/lag/box response as full-frame analysis, within an exact ROI.

    Include both lag and filter support as a halo. Never resize the local rope
    texture: pitch resolution and the mask threshold are unchanged.
    """
    x1, y1, x2, y2 = roi
    lag = max(4, int(round(pitch_px)))
    window = max(9, int(round(2.0 * pitch_px)) | 1)
    half = window // 2
    h, w = gray.shape
    gx1, gx2 = max(0, x1 - half - 2), min(w, x2 + half + 2)
    gy1, gy2 = max(0, y1 - lag - half - 2), min(h, y2 + lag + half + 2)
    gradient = np.abs(cv2.Sobel(gray[gy1:gy2, gx1:gx2], cv2.CV_32F, 0, 1, ksize=3))
    score = np.zeros_like(gradient)
    if len(score) > 2 * lag:
        score[lag:-lag] = np.minimum(gradient[lag:-lag],
                                     (gradient[:-2*lag] + gradient[2*lag:]) / 2.)
    score = cv2.boxFilter(score, cv2.CV_32F, (window, window), normalize=True)
    return score[y1-gy1:y2-gy1, x1-gx1:x2-gx1]


class GeometryReferenceFilter:
    """Filter nuisance geometry, not the moving rope position.

    A one-second causal median rejects flange-fit/pitch flicker while retaining
    the current frame's rope tangency y. Missing evidence for more than 0.6 s
    resets the reference regardless of window length. No extrapolated values
    enter this buffer.
    """
    def __init__(self, window_s=1.0):
        self.window_s = window_s
        self.history = deque()

    def update(self, time_s, measurement):
        if not measurement.valid:
            return measurement
        p = measurement.local_pitch_px
        if not math.isfinite(p) or p <= 0:
            return measurement
        if self.history and (time_s <= self.history[-1][0]
                             or time_s - self.history[-1][0] > .6
                             or abs(measurement.tangent_x - self.history[-1][1]) > 2*p):
            self.history.clear()
        self.history.append((time_s, measurement.tangent_x, measurement.edge_y, p))
        while self.history and time_s - self.history[0][0] > self.window_s:
            self.history.popleft()
        edge = float(np.median([v[2] for v in self.history]))
        pitch = float(np.median([v[3] for v in self.history]))
        distance = max(0., measurement.tangent_y - edge)
        scatter = float(np.median(np.abs(np.array([v[2] for v in self.history])-edge))) / pitch
        return replace(measurement, edge_y=edge, local_pitch_px=pitch,
                       axis_distance_px=distance, distance_rope=distance/pitch,
                       uncertainty_rope=max(measurement.uncertainty_rope, .6 + 1.5*scatter))


def local_pitch(gray, x, start_y, fallback):
    """Estimate the first strong autocorrelation peak in several narrow strips.

    Narrow strips avoid averaging away slanted rope grooves. Return confidence
    zero when texture is missing; never invent a local pitch in that case.
    """
    h, w = gray.shape
    y0 = max(0, min(h - 1, int(start_y)))
    y1 = min(h, y0 + int(24 * fallback))
    minimum = max(4, int(0.65 * fallback))
    maximum = max(minimum + 3, int(1.45 * fallback))
    pitches, scores = [], []
    for dx in (-fallback, 0, fallback, 2 * fallback):
        col = max(2, min(w - 3, int(x + dx)))
        signal = gray[y0:y1, col - 2:col + 3].mean(axis=1).astype(float)
        if len(signal) < maximum * 4 or np.std(signal) < 2:
            continue
        win = max(9, int(3 * fallback) | 1)
        smooth = np.convolve(np.pad(signal, win // 2, mode='edge'),
                             np.ones(win) / win, mode='valid')
        signal -= smooth
        ac = []
        for lag in range(minimum - 1, maximum + 2):
            a, b = signal[:-lag], signal[lag:]
            ac.append(float(np.dot(a, b) / max(1e-8, np.linalg.norm(a) * np.linalg.norm(b))))
        candidates = [i for i in range(1, len(ac) - 1)
                      if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > .35]
        if not candidates:
            continue
        i = max(candidates, key=lambda j: ac[j])
        denominator = ac[i - 1] - 2 * ac[i] + ac[i + 1]
        delta = .5 * (ac[i - 1] - ac[i + 1]) / denominator if abs(denominator) > 1e-8 else 0
        pitches.append(minimum - 1 + i + float(np.clip(delta, -.5, .5)))
        scores.append(ac[i])
    if len(pitches) < 2 or np.std(pitches) > .15 * fallback:
        return fallback, 0.0
    return float(np.median(pitches)), float(np.median(scores))


def _short_window_pitch(gray, x, center_y, fallback):
    """Estimate the shortest well-supported period in a local axial window."""
    h, w = gray.shape
    half = int(round(4.5 * fallback))
    y0, y1 = max(0, int(center_y) - half), min(h, int(center_y) + half)
    minimum = max(4, int(.60 * fallback))
    maximum = max(minimum + 3, int(1.30 * fallback))
    candidates = []
    for dx in (-2 * fallback, -fallback, 0, fallback, 2 * fallback):
        column = max(2, min(w - 3, int(x + dx)))
        signal = gray[y0:y1, column - 2:column + 3].mean(axis=1).astype(float)
        if len(signal) < 4 * maximum or np.std(signal) < 2:
            continue
        trend = cv2.GaussianBlur(signal.reshape(-1, 1), (1, 0),
                                 sigmaX=0, sigmaY=max(8, 1.5 * fallback))[:, 0]
        signal -= trend
        signal -= signal.mean()
        curve = []
        for lag in range(minimum, maximum + 1):
            a, b = signal[:-lag], signal[lag:]
            curve.append(float(np.dot(a, b) /
                               max(1e-8, np.linalg.norm(a) * np.linalg.norm(b))))
        for index in range(1, len(curve) - 1):
            if (curve[index] > curve[index - 1]
                    and curve[index] >= curve[index + 1]
                    and curve[index] >= .25):
                candidates.append((minimum + index, curve[index]))
    candidates.sort()
    for pitch, _score in candidates:
        support = [score for other, score in candidates if abs(other - pitch) <= 1]
        if len(support) >= 2:
            return float(pitch), float(np.median(support)), len(support)
    return math.nan, 0.0, 0


def adaptive_pitch_profile(gray, x, edge_y, tangent_y, fallback):
    """Estimate a per-frame effective row pitch at three axial positions.

    Perspective makes one global pixels-per-row value invalid over a long gap.
    The effective pitch is therefore the harmonic mean of local periods, which
    is equivalent to integrating ``dy / pitch(y)`` over equal path segments.
    No rope diameter, spool width, camera pose, or labelled distance is used.
    """
    if (not all(math.isfinite(value) for value in
                (x, edge_y, tangent_y, fallback)) or fallback <= 0
            or tangent_y <= edge_y):
        return math.nan, 0.0
    estimates = []
    for fraction in (.2, .5, .8):
        center_y = edge_y + fraction * (tangent_y - edge_y)
        pitch, score, support = _short_window_pitch(
            gray, x, center_y, fallback)
        if math.isfinite(pitch):
            estimates.append((pitch, score, support))
    if len(estimates) < 2:
        return math.nan, 0.0
    pitches = np.asarray([item[0] for item in estimates], dtype=float)
    effective = float(len(pitches) / np.sum(1.0 / pitches))
    score = float(np.median([item[1] for item in estimates]))
    support = float(np.median([item[2] for item in estimates]))
    confidence = float(np.clip(score * min(1.0, support / 4.0), 0.0, 1.0))
    return effective, confidence


def coordinate_percentiles(counts, percentiles):
    """Linear percentiles of integer coordinates, without expanding a mask.

    ``counts[i]`` is the number of occurrences of coordinate ``i``. The result
    matches ``np.percentile(np.repeat(np.arange(len(counts)), counts), q)``:
    including repeated coordinates, arbitrary q shapes, endpoint ranks and
    NumPy's two-sided linear interpolation rounding. Empty input follows
    NumPy's empty-percentile behaviour, rather than inventing a coordinate.
    """
    counts = np.asarray(counts)
    if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
        raise ValueError("coordinate counts must be a one-dimensional nonnegative integer array")
    quantiles = np.true_divide(np.asarray(percentiles), 100)
    if not np.all((quantiles >= 0) & (quantiles <= 1)):
        raise ValueError("Percentiles must be in the range [0, 100]")
    total = int(counts.sum())
    if not total or not quantiles.size:
        return np.percentile(np.empty(0, dtype=np.intp), percentiles)
    cumulative = np.cumsum(counts)
    ranks = (total - 1) * quantiles
    previous = np.floor(ranks).astype(np.intp)
    following = np.minimum(previous + 1, total - 1)
    low = np.searchsorted(cumulative, previous, side="right")
    high = np.searchsorted(cumulative, following, side="right")
    fraction = ranks - previous
    delta = high - low
    result = np.asarray(low + delta * fraction)
    np.subtract(high, delta * (1 - fraction), out=result, where=fraction >= .5)
    return result[()] if result.ndim == 0 else result


def flange_curve(gray, bgr, cylinder_mask, pitch, tangent_x, *, axis_slope=0., tangent_y=None,
                 sampled_curve=None):
    """Fit the flange/cylinder interface across columns independently of the
    live-rope intersection. Colour mode requires a sustained gold-to-grey
    transition, not just the first bright rope pixel. A robust quadratic rejects
    narrow crossing rope/tool edges. Grayscale mode is for synthetic tests.
    """
    h, w = gray.shape
    # Coordinate histograms are exactly the same distribution as nonzero(),
    # but do not allocate hundreds of thousands of repeated x/y coordinates.
    column_counts = np.count_nonzero(cylinder_mask, axis=0)
    if column_counts.sum() < 100:
        return None
    row_counts = np.count_nonzero(cylinder_mask, axis=1)
    left, right = coordinate_percentiles(column_counts, [3, 97])
    upper = float(coordinate_percentiles(row_counts, 3))
    x_samples = np.arange(max(3, int(left - 2 * pitch)),
                          min(w - 3, int(right)), max(3, int(pitch / 2)))
    if not len(x_samples):
        return None
    points = []
    has_rows = column_counts[x_samples] > 0
    top_rows = cylinder_mask[:, x_samples].argmax(axis=0)
    approximate = np.where(has_rows, np.minimum(top_rows, upper + 7 * pitch), upper)
    lows = np.maximum(0, np.minimum(approximate, upper) - 5 * pitch).astype(int)
    highs = np.minimum(h, np.maximum(approximate, upper) + 5 * pitch).astype(int)
    saturation = None
    if bgr is not None:
        # BGR->HSV is pixel-local: cropping changes neither saturation nor any
        # fitted point. Keep the original five-column strips and row bounds.
        crop_x = int(x_samples[0]) - 2
        crop_y = int(lows.min())
        crop_right = int(x_samples[-1]) + 3
        crop_bottom = int(highs.max())
        saturation = cv2.cvtColor(
            bgr[crop_y:crop_bottom, crop_x:crop_right], cv2.COLOR_BGR2HSV
        )[:, :, 1]
    band = max(4, int(pitch))
    for index, x in enumerate(x_samples):
        if saturation is None:
            if has_rows[index] and top_rows[index] < upper + 4 * pitch:
                points.append((float(x), float(top_rows[index])))
            continue
        lo, hi = int(lows[index]), int(highs[index])
        signal = saturation[lo - crop_y:hi - crop_y,
                            x - 2 - crop_x:x + 3 - crop_x].mean(axis=1)
        if len(signal) < 4 * band:
            continue
        smooth = np.convolve(signal, np.ones(band) / band, mode='valid')
        drop = smooth[:-band] - smooth[band:]
        i = int(np.argmax(drop))
        if drop[i] >= 25 and smooth[i] >= 65:
            points.append((float(x), float(lo + i + band)))
    if len(points) < 12:
        return None
    points = np.asarray(points)
    center = (left + right) / 2
    scale = max(1., right - left)
    xx = (points[:, 0] - center) / scale
    yy = points[:, 1]
    matrix = np.column_stack((xx * xx, xx, np.ones(len(xx))))
    # Deterministic robust fitting; no fitting to a particular video's labels.
    rng = np.random.default_rng(19)
    best = np.zeros(len(xx), dtype=bool)
    tolerance = max(3., .65 * pitch)
    for _ in range(80):
        idx = rng.choice(len(xx), 3, replace=False)
        if np.ptp(xx[idx]) < .35:
            continue
        coeff = np.linalg.lstsq(matrix[idx], yy[idx], rcond=None)[0]
        if coeff[0] < -pitch:  # cylinder end projects as a convex upper arc
            continue
        inliers = np.abs(matrix @ coeff - yy) <= tolerance
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < max(12, .55 * len(points)) or np.ptp(xx[best]) < .55:
        return None
    coeff = np.linalg.lstsq(matrix[best], yy[best], rcond=None)[0]
    residual = float(np.sqrt(np.mean((matrix[best] @ coeff - yy[best]) ** 2)))
    z = (tangent_x - center) / scale
    if tangent_x < points[best, 0].min() - 3 * pitch or tangent_x > points[best, 0].max() + 3 * pitch:
        return None
    edge_y = float(np.array([z*z, z, 1]) @ coeff)
    # Intersect the SAME fitted curve with the projected axis. Previously the
    # entire HSV conversion and RANSAC fit were repeated for a nearby column.
    if tangent_y is not None:
        for _ in range(3):
            axis_x = tangent_x + axis_slope * (edge_y - tangent_y)
            z = (axis_x - center) / scale
            edge_y = float(np.array([z*z, z, 1]) @ coeff)
        if axis_x < points[best, 0].min() - 3 * pitch or axis_x > points[best, 0].max() + 3 * pitch:
            return None
    confidence = float(best.mean() * np.exp(-residual / max(1., pitch)))
    if sampled_curve is not None:
        for px in np.linspace(points[best, 0].min(), points[best, 0].max(), 40):
            zz = (px-center)/scale
            sampled_curve.append((float(px), float(np.array([zz*zz, zz, 1]) @ coeff)))
    return edge_y, confidence, residual
