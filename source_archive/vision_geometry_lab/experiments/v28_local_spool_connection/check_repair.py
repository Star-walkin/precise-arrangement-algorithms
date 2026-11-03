"""Small post-save V28 candidate diagnostic; never ranks or edits candidates.

The original 5 px / 80% trusted-reference coverage policy is unchanged. A
geometric completion covering a reference is reported separately from coverage
by consecutive, strictly image-supported body samples. Unknown is not a
background label or a negative reference.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from vision_geometry_lab.experiments.v28_local_spool_connection import data_eval


def _flags(candidate, key, count, required=True):
    if key not in candidate:
        if required:
            raise ValueError(f"Candidate {candidate.get('id')} is missing {key}")
        return None
    values = np.asarray(candidate[key])
    if values.shape != (count,) or values.dtype.kind != 'b':
        raise ValueError(f"Candidate {candidate.get('id')} {key} must be an equal-length boolean list")
    return values


def _validate_candidate(candidate):
    points = np.asarray(candidate['points_xy'], float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('Each candidate needs at least two finite original xy points')
    count = len(points)
    supported = _flags(candidate, 'supported', count)
    inferred = _flags(candidate, 'inferred', count)
    unavailable = _flags(candidate, 'unavailable', count)
    if not np.all(supported.astype(int) + inferred.astype(int) + unavailable.astype(int) == 1):
        raise ValueError('supported/inferred/unavailable must be a disjoint complete partition')
    strict = _flags(candidate, 'strict_body_supported', count, required=False)
    if strict is not None and np.any(strict & (inferred | unavailable)):
        raise ValueError('Strict body support cannot include inferred or unavailable samples')
    raw = _flags(candidate, 'raw_pair_supported', count, required=False)
    if raw is None:
        evidence = candidate.get('evidence', {})
        raw = _flags(evidence, 'supported', count, required=False)
    delta = np.linalg.norm(np.diff(points, axis=0), axis=1)
    weights = .5 * (np.r_[delta, 0.] + np.r_[0., delta])
    if weights.sum() <= 0:
        raise ValueError('Zero-length candidate is not evaluable')
    return points, supported, inferred, unavailable, strict, raw, weights


def strict_supported_segments(points, support):
    """Consecutive true samples only; never connect across a missing sample."""
    points = np.asarray(points, float)
    support = np.asarray(support, bool)
    if support.shape != (len(points),):
        raise ValueError('Strict support length mismatch')
    ids = np.flatnonzero(support)
    if not len(ids):
        return []
    runs = np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1)
    return [points[run].tolist() for run in runs if len(run) >= 2]


def _region_masks(regions, shape):
    result = {}
    for key in ('explicit_background', 'outside_search_mask', 'search_mask', 'unknown_mask'):
        if regions is None or key not in regions:
            continue
        mask = np.asarray(regions[key])
        if mask.shape != tuple(shape) or mask.dtype.kind != 'b':
            raise ValueError(f'{key} must be a boolean mask in original image coordinates')
        result[key] = mask
    if 'outside_search_mask' not in result and 'search_mask' in result:
        result['outside_search_mask'] = ~result['search_mask']
    if 'explicit_background' in result and 'unknown_mask' in result:
        # Never silently promote an unknown region into a negative observation.
        result['explicit_background'] = result['explicit_background'] & ~result['unknown_mask']
    return result


def _fraction(mask, weights):
    return float(np.sum(weights[mask]) / np.sum(weights))


def evaluate_saved_pool(sample_id, pool_path, regions):
    """Inspect an existing frozen pool, then load reference only for evaluation.

    The pool is read and validated first. data_eval.evaluate_case reads it again
    and requires JSON equality before it loads the reference. Every dummy arm
    is UNKNOWN with no selected path; this function introduces no new scorer.
    """
    path = Path(pool_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    raw_bytes = path.read_bytes()
    pool = json.loads(raw_bytes.decode('utf-8-sig'))
    candidates = pool['candidates']
    validated = [_validate_candidate(candidate) for candidate in candidates]
    shape = pool.get('image_shape_hw')
    if shape is None or len(shape) != 2 or min(shape) <= 0:
        raise ValueError('Saved pool must declare original image_shape_hw')
    masks = _region_masks(regions, shape)
    origin = np.asarray(pool['origin_xy'], float)
    direction = np.asarray(pool['direction_xy'], float)
    norm = np.linalg.norm(direction)
    if origin.shape != (2,) or direction.shape != (2,) or not np.isfinite(origin).all() or not np.isfinite(norm) or norm < 1e-8:
        raise ValueError('Saved pool has invalid geometry')
    direction = direction / norm
    normal = np.array([-direction[1], direction[0]])
    rankings = {group: dict(best_id=None, selected_id=None, status='UNKNOWN',
                           reason='candidate_generation_diagnostic_no_ranking')
                for group in ('G0', 'G1', 'G2')}
    evaluation = data_eval.evaluate_case(sample_id, pool, rankings, pool.get('input_mode', 'AUTO'),
                                         persisted_pool_path=path)
    by_id = {row['id']: row for row in evaluation['candidate_scores']}
    reference = evaluation['evaluation_only_reference']
    bend = evaluation['trusted_bend_segments']
    rows = []
    for candidate, values in zip(candidates, validated):
        points, supported, inferred, unavailable, strict, raw, weights = values
        u = (points - origin) @ direction
        v = (points - origin) @ normal
        inside = ((points[:, 0] >= 0) & (points[:, 0] <= shape[1]-1) &
                  (points[:, 1] >= 0) & (points[:, 1] <= shape[0]-1))
        pixels = np.rint(points).astype(int)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, shape[1]-1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, shape[0]-1)
        fractions = {}
        for key, output_name in (
                ('explicit_background', 'explicit_background_fraction'),
                ('outside_search_mask', 'outside_search_fraction'),
                ('unknown_mask', 'unknown_region_fraction')):
            fractions[output_name] = (_fraction(masks[key][pixels[:, 1], pixels[:, 0]] & inside, weights)
                                      if key in masks else None)
        strict_coverage = None
        if strict is not None:
            pieces = strict_supported_segments(points, strict)
            pieces = data_eval.clip_scoring_segments(pieces, ignore_polygons=reference['ignore_polygons'])
            strict_coverage = data_eval._bend_coverage(bend, pieces)
        geometric = by_id[candidate['id']]['bend_geometry_5px']
        rows.append(dict(id=candidate['id'], kind=candidate.get('kind'),
            bend5px_coverage=geometric['covered_fraction'],
            geometric_bend80=geometric['covered_fraction'] is not None and geometric['covered_fraction'] >= .8,
            supported_only_bend5px_coverage=strict_coverage['covered_fraction'] if strict_coverage else None,
            strict_support_available=strict is not None,
            strict_supported_bend80=(strict_coverage['covered_fraction'] is not None and strict_coverage['covered_fraction'] >= .8) if strict_coverage else None,
            supported_fraction=_fraction(supported, weights),
            strict_body_supported_fraction=_fraction(strict, weights) if strict is not None else None,
            raw_pair_supported_fraction=_fraction(raw, weights) if raw is not None else None,
            inferred_fraction=_fraction(inferred, weights), unavailable_fraction=_fraction(unavailable, weights),
            arc_length_px=float(weights.sum()), forward_u_range_px=[float(u.min()), float(u.max())],
            lateral_v_range_px=[float(v.min()), float(v.max())], endpoint_xy=points[-1].tolist(),
            outside_image_fraction=_fraction(~inside, weights), **fractions,
            geometry_completion_is_observation=False, physical_identity_confirmed=False))
    full = [r['bend5px_coverage'] for r in rows if r['bend5px_coverage'] is not None]
    strict_values = [r['supported_only_bend5px_coverage'] for r in rows if r['supported_only_bend5px_coverage'] is not None]
    return dict(sample_id=sample_id, mode=pool.get('input_mode', 'AUTO'),
        saved_pool_path=str(path.resolve()), saved_pool_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        candidate_count=len(rows), bend_reference_length_px=evaluation['bend_reference_length_px'],
        best_bend5px_coverage=max(full) if full else None,
        best_supported_only_bend5px_coverage=max(strict_values) if strict_values else None,
        candidate_pool_contains_trusted_bend_geometry=bool(full and max(full) >= .8),
        candidate_pool_contains_strict_supported_bend=bool(strict_values and max(strict_values) >= .8) if strict_values else None,
        forward_reach_px=max((r['forward_u_range_px'][1] for r in rows), default=None),
        candidates=rows, tolerance_px=5., coverage_threshold=.8,
        unknown_is_background=False, outside_search_is_confirmed_background=False,
        ratios_are_arc_length_weighted=True, labels_opened_only_after_saved_pool_equality=True,
        ranking_performed=False, physical_identity_confirmed=False)
