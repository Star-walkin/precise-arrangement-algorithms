"""Annotation-guided texture feasibility diagnostic, NOT an automatic detector.

Manual boundaries, directions and widths define both positive and internal
negative bands. Fixed parameters are not fitted on train OR holdout. Strand
texture can break phase inside one rope, so a discontinuity is not rope identity.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

from .centerline_quality_v5 import load_label_quality
from .centerline_review_v5 import (DEFAULT_MANIFEST, file_sha256, load_manifest,
                                   read_clean_image, validate_annotation)
from .geometry import dump_json
from .prepare_centerline_v5 import check_groups


# Protocol constants: fixed before reading field scores. No optimization loop.
PARAMETERS = dict(spacing_px=24., along_half_length_px=24, normal_rows=5,
                  band_center_width_fraction=.12, band_half_width_fraction=.04,
                  interior_offset_width_fraction=.40, min_width_px=8., max_width_px=120.,
                  black_pixel_threshold=8., saturated_pixel_threshold=247.,
                  max_black_fraction=.25, max_saturated_fraction=.10,
                  min_median_gray=12., max_median_gray=243., min_highpass_rms=2.,
                  highpass_gaussian_sigma_px=2.5, min_orientation_coherence=.15)
FEATURES = ('phase_decorrelation', 'orientation_jump', 'energy_jump',
            'texture_combined', 'brightness_gradient')
BRIGHTNESS_BINS = (('very_dark', 0., 32.), ('dim', 32., 96.),
                   ('mid', 96., 192.), ('bright', 192., 240.),
                   ('near_clipping', 240., 256.))


def brightness_bin(value):
    if value is None:
        return 'unavailable'
    return next(name for name, lo, hi in BRIGHTNESS_BINS if lo <= value < hi)


def _polygon_mask(shape, polygons):
    mask = np.zeros(shape, np.uint8)
    for polygon in polygons:
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 255)
    return mask


def _band_grid(point, tangent, normal, width):
    along = np.arange(-PARAMETERS['along_half_length_px'],
                      PARAMETERS['along_half_length_px'] + 1, dtype=float)
    distance = float(np.clip(width * PARAMETERS['band_center_width_fraction'], 1.5, 6.))
    half = float(np.clip(width * PARAMETERS['band_half_width_fraction'], .5, 2.))
    offsets = np.linspace(-half, half, PARAMETERS['normal_rows'])
    grids = [point + along[None, :, None] * tangent +
             (sign * distance + offsets[:, None, None]) * normal
             for sign in (-1., 1.)]
    return grids, distance, float(offsets[1] - offsets[0])


def _sample(image, grid):
    return cv2.remap(image, grid[..., 0].astype(np.float32),
                     grid[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


def _inside_image(grid, shape):
    # One-pixel interpolation footprint must lie within the original image.
    return bool(((grid[..., 0] >= 1) & (grid[..., 0] < shape[1] - 2) &
                 (grid[..., 1] >= 1) & (grid[..., 1] < shape[0] - 2)).all())


def _exposure_reason(strip):
    if (np.mean(strip <= PARAMETERS['black_pixel_threshold']) >= PARAMETERS['max_black_fraction']
            or np.median(strip) < PARAMETERS['min_median_gray']):
        return 'black_or_too_dark'
    if (np.mean(strip >= PARAMETERS['saturated_pixel_threshold']) >= PARAMETERS['max_saturated_fraction']
            or np.median(strip) > PARAMETERS['max_median_gray']):
        return 'saturated_or_overexposed'
    return None


def _highpass(strip):
    kernel = cv2.getGaussianKernel(15, PARAMETERS['highpass_gaussian_sigma_px'])
    smooth = cv2.sepFilter2D(strip, -1, kernel, np.ones(1), borderType=cv2.BORDER_REFLECT101)
    return (strip - smooth)[:, 7:-7]


def _orientation(strip, normal_step):
    gy, gx = np.gradient(strip, normal_step, 1.)
    xx, yy, xy = float(np.mean(gx * gx)), float(np.mean(gy * gy)), float(np.mean(gx * gy))
    coherence = float(np.hypot(xx - yy, 2 * xy) / max(xx + yy, 1e-12))
    return .5 * float(np.arctan2(2 * xy, xx - yy)), coherence


def strip_features(left, right, distance, normal_step):
    """Zero-lag correlation is a phase proxy, not a unique material boundary.

    The corresponding negative uses identical separation and direction. An
    oblique continuous texture can itself decorrelate across this separation;
    this deliberately remains a failure mode measured by the internal control.
    """
    values = {name: None for name in FEATURES}
    values['brightness_gradient'] = float(abs(np.mean(left) - np.mean(right)) / (2 * distance))
    a, b = _highpass(left), _highpass(right)
    rms_a, rms_b = float(np.sqrt(np.mean(a * a))), float(np.sqrt(np.mean(b * b)))
    if min(rms_a, rms_b) < PARAMETERS['min_highpass_rms']:
        return values, 'low_texture_amplitude'
    corr = float(np.sum(a * b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    values['phase_decorrelation'] = float((1 - np.clip(corr, -1., 1.)) / 2)
    values['energy_jump'] = abs(rms_a - rms_b) / (rms_a + rms_b)
    theta_a, coherence_a = _orientation(a, normal_step)
    theta_b, coherence_b = _orientation(b, normal_step)
    if min(coherence_a, coherence_b) >= PARAMETERS['min_orientation_coherence']:
        values['orientation_jump'] = float((1 - np.cos(2 * (theta_a - theta_b))) / 2)
        values['texture_combined'] = float(np.mean([values[name] for name in FEATURES[:3]]))
    return values, None if values['texture_combined'] is not None else 'low_orientation_coherence'


def probe_pair(gray, boundary_xy, tangent, inward_normal, width, *,
               exclusion_mask=None, body_mask=None):
    """Compare an annotated boundary with its 0.4-width inward parallel line."""
    point, tangent, normal = map(lambda x: np.asarray(x, float),
                                  (boundary_xy, tangent, inward_normal))
    result = dict(valid_pair=False, reason=None, brightness=None, brightness_bin='unavailable',
                  boundary_scores=None, interior_scores=None, texture_reasons=None)
    if (not PARAMETERS['min_width_px'] <= width <= PARAMETERS['max_width_px']
            or not np.isfinite(np.r_[point, tangent, normal, width]).all()
            or abs(np.linalg.norm(tangent) - 1) > 1e-5
            or abs(np.linalg.norm(normal) - 1) > 1e-5 or abs(tangent @ normal) > 1e-5):
        return dict(result, reason='invalid_manual_geometry')
    negative = point + normal * width * PARAMETERS['interior_offset_width_fraction']
    positive_grids, distance, step = _band_grid(point, tangent, normal, width)
    negative_grids, _, _ = _band_grid(negative, tangent, normal, width)
    grids = positive_grids + negative_grids
    if not all(_inside_image(grid, gray.shape) for grid in grids):
        return dict(result, reason='outside_image_support')
    if exclusion_mask is not None and any(np.any(_sample(exclusion_mask, grid) > 0) for grid in grids):
        return dict(result, reason='quality_or_manual_exclusion')
    if body_mask is not None and any(np.any(_sample(body_mask, grid) < 254) for grid in negative_grids):
        return dict(result, reason='internal_control_not_inside_manual_body')
    strips = [_sample(gray, grid) for grid in grids]
    brightness = float(np.mean(strips[:2]))
    result.update(brightness=brightness, brightness_bin=brightness_bin(brightness))
    exposure = [_exposure_reason(strip) for strip in strips]
    if any(exposure):
        return dict(result, reason=next(reason for reason in exposure if reason))
    positive, positive_reason = strip_features(*strips[:2], distance, step)
    negative_scores, negative_reason = strip_features(*strips[2:], distance, step)
    return dict(result, valid_pair=True, reason='exposure_valid', boundary_scores=positive,
                interior_scores=negative_scores, texture_reasons=[positive_reason, negative_reason])


def _resample_edge(points):
    points = np.asarray(points, float)
    length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(length) > 1e-6]
    points, length = points[keep], length[keep]
    if len(points) < 2:
        return np.empty((0, 2)), np.empty((0, 2))
    locations = np.arange(0., length[-1] + 1e-6, PARAMETERS['spacing_px'])
    sampled = np.column_stack([np.interp(locations, length, points[:, dim]) for dim in (0, 1)])
    # Segment directions near each location are fixed by the manual line.
    indices = np.clip(np.searchsorted(length, locations, side='right') - 1, 0, len(points) - 2)
    tangent = points[indices + 1] - points[indices]
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    return sampled, tangent


def _opposite_point(point, opposite):
    a, vectors = opposite[:-1], np.diff(opposite, axis=0)
    squared = np.sum(vectors * vectors, axis=1)
    valid = squared > 1e-10
    if not valid.any():
        return None
    a, vectors, squared = a[valid], vectors[valid], squared[valid]
    fraction = np.clip(np.sum((point - a) * vectors, axis=1) / squared, 0., 1.)
    nearest = a + fraction[:, None] * vectors
    index = int(np.argmin(np.linalg.norm(nearest - point, axis=1)))
    return nearest[index]


def probe_annotation(gray, label, polygons=()):
    lines = label['polylines']
    if any(len(lines[name]) < 2 for name in ('edge_flange', 'edge_surface')):
        return []
    edges = [np.asarray(lines[name], float) for name in ('edge_flange', 'edge_surface')]
    # A filled manual body is only a diagnostic validity guard for negative bands.
    body = _polygon_mask(gray.shape, [np.vstack([edges[0], edges[1][::-1]])])
    body = cv2.erode(body, np.ones((3, 3), np.uint8))
    excluded = _polygon_mask(gray.shape, polygons)
    excluded = cv2.dilate(excluded, np.ones((3, 3), np.uint8))
    records = []
    for edge_index, edge in enumerate(edges):
        points, tangents = _resample_edge(edge)
        for point, tangent in zip(points, tangents):
            nearest = _opposite_point(point, edges[1 - edge_index])
            geometry = dict(boundary_xy=point.tolist(), edge=('edge_flange', 'edge_surface')[edge_index])
            vector = nearest - point if nearest is not None else np.zeros(2)
            width = float(np.linalg.norm(vector))
            if nearest is None or width < 1e-6 or abs(float(vector @ tangent)) > .35 * width:
                records.append(dict(geometry, valid_pair=False, reason='opposite_edge_not_normal',
                                    brightness_bin='unavailable', boundary_scores=None, interior_scores=None))
                continue
            normal = np.array([-tangent[1], tangent[0]])
            if normal @ vector < 0:
                normal = -normal
            width = float(vector @ normal)
            result = probe_pair(gray, point, tangent, normal, width,
                                exclusion_mask=excluded, body_mask=body)
            records.append(dict(geometry, width_px=width, **result))
    return records


def paired_metrics(records, feature):
    scores = [(r['boundary_scores'][feature], r['interior_scores'][feature]) for r in records
              if r.get('boundary_scores') is not None
              and r['boundary_scores'][feature] is not None
              and r['interior_scores'][feature] is not None]
    result = dict(candidate_pairs=len(records), valid_pairs=len(scores),
                  coverage=len(scores) / len(records) if records else None,
                  paired_win_rate=None, auc=None, boundary_median=None, interior_median=None)
    if scores:
        positive, negative = np.asarray(scores).T
        ties = np.isclose(positive, negative, atol=1e-7, rtol=1e-6)
        wins = np.where(ties, .5, (positive > negative).astype(float))
        comparisons = positive[:, None] - negative[None, :]
        auc_ties = np.isclose(positive[:, None], negative[None, :], atol=1e-7, rtol=1e-6)
        auc = np.mean(np.where(auc_ties, .5, (comparisons > 0).astype(float)))
        result.update(paired_win_rate=float(np.mean(wins)), auc=float(auc),
                      boundary_median=float(np.median(positive)), interior_median=float(np.median(negative)))
    return result


def summarize(records):
    return dict(candidate_pairs=len(records), exposure_valid_pairs=sum(r.get('valid_pair', False) for r in records),
                rejection_reasons=dict(Counter(r['reason'] for r in records if not r.get('valid_pair'))),
                texture_missing_pair_reasons=dict(Counter(reason for r in records
                    for reason in set(r.get('texture_reasons') or []) if reason is not None)),
                metrics={feature: paired_metrics(records, feature) for feature in FEATURES})


def evaluate(manifest_path):
    manifest = load_manifest(manifest_path)
    check_groups(manifest['samples'])
    rows, all_records = [], []
    for sample in manifest['samples']:
        row = {name: sample[name] for name in ('sample_id', 'group_id', 'split')}
        path = Path(sample['annotation_path'])
        if not path.exists():
            rows.append(dict(row, status='missing_annotation', records=[], summary=summarize([])))
            continue
        raw = json.loads(path.read_text('utf-8-sig'))
        if raw.get('reviewed') is not True:
            rows.append(dict(row, status='pending_annotation', records=[], summary=summarize([])))
            continue
        label = validate_annotation(raw, sample)
        quality = load_label_quality(sample)  # Fail closed on stale sidecar SHA.
        image = np.asarray(read_clean_image(sample))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        records = [] if label['status'] in ('absent', 'unidentifiable') else probe_annotation(
            gray, label, label['ignore_polygons'] + quality['precision_exclude_polygons'])
        records = [dict(record, **row) for record in records]
        all_records.extend(records)
        rows.append(dict(row, status=label['status'], image_sha256=sample['image_sha256'],
                         annotation_sha256=file_sha256(path), quality_sha256=quality['quality_sha256'],
                         quality_regions=quality['regions'], records=records, summary=summarize(records)))
    stratified = []
    for split in ('train', 'holdout'):
        for stratum in ('all', *(name for name, _, _ in BRIGHTNESS_BINS), 'unavailable'):
            subset = [r for r in all_records if r['split'] == split and
                      (stratum == 'all' or r['brightness_bin'] == stratum)]
            stratified.append(dict(split=split, brightness_bin=stratum, **summarize(subset)))
    groups = [dict(group_id=group, split=next(s['split'] for s in manifest['samples'] if s['group_id'] == group),
                   **summarize([r for r in all_records if r['group_id'] == group]))
              for group in sorted({s['group_id'] for s in manifest['samples']})]
    report = dict(schema_version=1, annotation_guided_diagnostic=True, automatic_detector=False,
                independent_accuracy_test=False, physical_precision_validated=False,
                manual_direction_and_width_used=True, model_training=False, hardware_access=False,
                manifest_sha256=file_sha256(manifest_path), implementation_sha256=file_sha256(__file__),
                parameters=PARAMETERS,
                brightness_bins=[dict(name=name, lower_inclusive=lo, upper_exclusive=hi)
                                 for name, lo, hi in BRIGHTNESS_BINS],
                parameter_selection='fixed before field scores; no train or holdout fitting',
                texture_method='along-rope highpass zero-lag normalized correlation, structure tensor, RMS contrast',
                higher_score_means='stronger local discontinuity; not a predicted boundary or identity',
                limitations=[
                    'Manual boundaries/directions/widths select all evidence; this is not an automatic detector.',
                    'Internal strands and oblique continuous texture can break phase without a rope boundary.',
                    'AUC and paired wins describe selected manual bands, not localization accuracy.',
                    'Black, saturated, excluded and low-amplitude evidence is missing, never success.',
                    'Brightness uses exposure-valid support; texture comparisons also report the same common support.',
                    'Nearby points/frames are correlated; pooled metrics are descriptive, not independent trials.',
                    'Manually bounded internal controls may still contain strand seams; identity is not inferred.'
                ], rows=rows, stratified=stratified, groups=groups)
    report['split_macro_descriptive'] = []
    for split in ('train', 'holdout'):
        macro = dict(split=split, metrics={})
        for feature in FEATURES:
            sample_scores = [r['summary']['metrics'][feature]['paired_win_rate']
                             for r in rows if r['split'] == split]
            group_scores = [r['metrics'][feature]['paired_win_rate']
                            for r in groups if r['split'] == split]
            available_samples = [score for score in sample_scores if score is not None]
            available_groups = [score for score in group_scores if score is not None]
            macro['metrics'][feature] = dict(
                total_samples=len(sample_scores), samples_with_evidence=len(available_samples),
                mean_sample_paired_win_rate=float(np.mean(available_samples)) if available_samples else None,
                total_groups=len(group_scores), groups_with_evidence=len(available_groups),
                mean_group_paired_win_rate=float(np.mean(available_groups)) if available_groups else None)
        report['split_macro_descriptive'].append(macro)
    return _add_common_support(report)


def _add_common_support(report):
    # Fair brightness comparator on exactly the paired texture-valid subset.
    for row in [*report['stratified'], *report['groups']]:
        records = [r for sample in report['rows'] for r in sample['records']
                   if r['split'] == row['split'] and
                   ('group_id' not in row or r['group_id'] == row['group_id']) and
                   ('brightness_bin' not in row or row['brightness_bin'] == 'all'
                    or r['brightness_bin'] == row['brightness_bin'])]
        common = [r for r in records if r.get('boundary_scores') is not None and
                  all(r[key]['texture_combined'] is not None for key in ('boundary_scores', 'interior_scores'))]
        row['brightness_on_texture_common_support'] = paired_metrics(common, 'brightness_gradient')
        row['brightness_on_texture_common_support']['coverage_of_all_candidates'] = len(common) / len(records) if records else None
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output', type=Path, default=DEFAULT_MANIFEST.parent.parent / 'review_round3_texture/texture_probe')
    args = parser.parse_args(argv)
    cv2.setNumThreads(2)
    report = evaluate(args.manifest)
    dump_json(args.output / 'summary.json', report)
    print(json.dumps(dict(report=str(args.output / 'summary.json'), samples=len(report['rows']),
                          split_results=[row for row in report['stratified'] if row['brightness_bin'] == 'all']),
                     ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
