"""Bounded regression for the outward-normal side assignment fix only."""
from pathlib import Path
import hashlib
import json
import time
import types

import cv2
import numpy as np

import free_rope as current
from free_rope_selfcheck import scene


def polygon_checks(result):
    free = result['free_segment']
    left = np.asarray(free['boundary_left_xy'])
    right = np.asarray(free['boundary_right_xy'])
    normal = np.asarray(free['normal_xy']); direction = np.asarray(free['direction_xy'])
    widths = (right - left) @ normal
    advances = np.diff(((left + right) / 2) @ direction)
    nonconvex = 0; tested = 0
    for i, advance in enumerate(advances):
        # Duplicate endpoint profiles have zero area, not a crossed ribbon.
        if advance <= .25: continue
        quad = np.asarray([left[i], right[i], right[i + 1], left[i + 1]], np.float32)
        nonconvex += not bool(cv2.isContourConvex(quad)); tested += 1
    components, _, stats, _ = cv2.connectedComponentsWithStats(result['free_mask'], 8)
    return {'negative_side_widths': int(np.count_nonzero(widths <= 0)),
            'minimum_signed_edge_separation_px': float(widths.min()),
            'tested_positive_advance_polygons': tested, 'nonconvex_splice_polygons': int(nonconvex),
            'free_mask_components': int(components - 1),
            'component_areas': stats[1:, cv2.CC_STAT_AREA].tolist()}


def mirrored(data):
    image = data['original']; width = image.shape[1]
    seed = dict(data['a1_record']['free_segment'])
    segment = np.asarray(seed['observed_segment_xy']); segment[:, 0] = width - 1 - segment[:, 0]
    seed['observed_segment_xy'] = segment.tolist()
    return {'original': cv2.flip(image, 1), 'surface': cv2.flip(data['surface'], 1), 'pitch': data['pitch'],
            'a1_record': {'free_segment': seed}}


def main():
    begin = time.perf_counter(); here = Path(__file__).resolve().parent
    old_path = here / 'free_probe_002/source_snapshot/free_rope.py'
    old = types.ModuleType('_free_before_normal_fix')
    # Preserve the original module's asset-relative dependency lookup while
    # executing the preserved old bytes, without modifying either file.
    old.__file__ = str(here / 'free_rope.py')
    exec(compile(old_path.read_text(encoding='utf-8'), str(old_path), 'exec'), old.__dict__)
    assert old.PARAMETERS == current.PARAMETERS
    cases = []
    for name, data in [('toward_right_outward_left', scene()), ('toward_left_outward_right', mirrored(scene()))]:
        before = old.refine_and_extend(data); after = current.refine_and_extend(data)
        before_check = polygon_checks(before); after_check = polygon_checks(after)
        assert after_check['negative_side_widths'] == 0
        assert after_check['nonconvex_splice_polygons'] == 0
        assert after_check['free_mask_components'] == 1
        assert before['geometry'] == after['geometry']
        assert before['free_segment']['width_px'] == after['free_segment']['width_px']
        assert before['free_segment']['observed_segment_xy'] == after['free_segment']['observed_segment_xy']
        assert before['free_segment']['observed_points_xy'] == after['free_segment']['observed_points_xy']
        assert np.array_equal(before['straight_observed_mask'], after['straight_observed_mask'])
        assert np.array_equal(before['straight_inferred_mask'], after['straight_inferred_mask'])
        cases.append({'name': name, 'before': before_check, 'after': after_check,
                      'free_mask_changed_pixels': int(np.count_nonzero(before['free_mask'] != after['free_mask'])),
                      'width_centerline_endpoints_and_inward_masks_unchanged': True})
    no_seed = scene()
    no_seed['a1_record'] = {'free_segment': {'status': 'unknown'}}
    no_seed['a2_record'] = {'free_segment': {'status': 'unknown'},
                            'connection': {'selected_candidate_id': 'old_curve', 'points_xy': [[10, 10], [700, 500]]}}
    unknown = current.refine_and_extend(no_seed)
    assert unknown['status'] == 'unknown'
    assert unknown['diagnostics']['reason'] == 'no_usable_automatic_free_seed'
    assert unknown['straight_candidates'] == [] and unknown['outward_extension'] is None
    assert sum(np.count_nonzero(unknown[k]) for k in ['free_mask', 'straight_observed_mask', 'straight_inferred_mask']) == 0
    receipt = {'version': 'IR-A3.0.0', 'revision': 'free outward-normal side assignment only', 'passed': True,
               'checks': cases, 'parameters_unchanged': True,
               'missing_auto_seed_stays_unknown': {'status': unknown['status'], 'reason': unknown['diagnostics']['reason'], 'mask_pixels': 0},
               'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [old_path, here / 'free_rope.py', Path(__file__)]},
               'elapsed_s': time.perf_counter() - begin, 'real_image_experiments': 0,
               'accuracy': None, 'equipment_access': False, 'reserved_inputs': False}
    path = here / 'free_rope_synthetic_003.json'
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__': main()
