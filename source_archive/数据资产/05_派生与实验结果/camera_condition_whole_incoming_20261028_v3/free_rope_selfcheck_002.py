"""Reproducible synthetic mechanism checks, not field/identity validation."""
from pathlib import Path
import copy
import hashlib
import json
import time

import cv2
import numpy as np

from free_rope import refine_and_extend, dependencies


def scene(dark_overlap=False, endpoint=630):
    h, w = 600, 800
    yy, xx = np.mgrid[:h, :w]
    surface = np.zeros((h, w), np.uint8); surface[100:520, 400:780] = 255
    base = np.where(surface != 0, 170. if dark_overlap else 65., 25.)
    cross = np.clip((8.5 - np.abs(yy - 300.)) / 1.5, 0, 1)
    span = ((xx >= 45) & (xx <= endpoint)).astype(float)
    value = np.where((surface != 0) & dark_overlap, 40., 210.)
    gray = base * (1 - cross * span) + value * cross * span
    image = np.repeat(np.clip(gray, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
    free = {'status': 'observed', 'coordinate_system': 'original_unmirrored_image_xy_pixels',
            'observed_segment_xy': [[90., 300.], [350., 300.]], 'width_px': 15.5,
            'direction_xy': [1., 0.], 'angle_deg': 0., 'diagnostics': {}}
    return {'original': image, 'surface': surface, 'pitch': 24., 'a1_record': {'free_segment': free}}


def main():
    started = time.perf_counter(); checks = {}
    data = scene(); original = data['original'].copy(); surface = data['surface'].copy()
    bright = refine_and_extend(data)
    assert bright['status'] == 'observed'
    f = bright['free_segment']; c = bright['straight_candidates'][0]
    assert abs(f['width_px'] - 15.5) < 2.
    assert abs(f['angle_deg']) < .2
    assert 610 < c['visible_end_xy'][0] < 640, c['visible_end_xy']
    assert c['stop_reason'] == 'bilateral_raw_profile_unsupported'
    assert not np.any(bright['free_mask'][surface != 0])
    assert np.array_equal(original, data['original']) and np.array_equal(surface, data['surface'])
    checks['bright_straight_stops_at_missing_stripe'] = {'width_px': f['width_px'], 'angle_deg': f['angle_deg'],
        'visible_end_xy': c['visible_end_xy'], 'stop_reason': c['stop_reason'], 'observed_profiles': c['observed_profile_count'],
        'free_mask_surface_intersection_pixels': int(np.count_nonzero(bright['free_mask'][surface != 0]))}
    dark = refine_and_extend(scene(dark_overlap=True))
    assert dark['status'] == 'observed'
    dc = dark['straight_candidates'][0]
    assert dc['visible_end_xy'][0] < 420, dc['visible_end_xy']
    assert np.count_nonzero(dark['straight_inferred_mask']) > 0
    polarities = {r['measured']['polarity'] for r in dc['support_samples'] if r['measured']}
    assert polarities == {'bright'}, polarities
    checks['dark_pair_does_not_relocate_or_remeasure_rope'] = {'visible_end_xy': dc['visible_end_xy'], 'polarities': sorted(polarities)}
    absent = scene(); absent['a1_record'] = {}; absent['a2_record'] = {'connection': {'selected_candidate_id': 'fabricated'}}
    no_seed = refine_and_extend(absent)
    assert no_seed['status'] == 'unknown' and np.count_nonzero(no_seed['free_mask']) == 0
    checks['missing_auto_seed_returns_unknown'] = no_seed['diagnostics']['reason']
    uniform = scene(); uniform['original'][:] = 35
    weak = refine_and_extend(uniform)
    assert weak['status'] == 'unknown' and np.count_nonzero(weak['straight_observed_mask']) == 0
    checks['uniform_raw_image_rejects_saved_seed'] = weak['diagnostics']['reason']
    poison = scene(); poison['a1_record']['connection'] = {'selected_candidate_id': 'wrong_curve', 'points_xy': [[0, 0], [799, 599]]}
    recheck = refine_and_extend(poison)
    assert np.array_equal(recheck['free_mask'], bright['free_mask'])
    assert np.array_equal(recheck['straight_observed_mask'], bright['straight_observed_mask'])
    checks['old_selected_connection_inert'] = True
    assert np.all((bright['free_mask'] == 0) | (bright['free_mask'] == 255))
    assert np.all((bright['straight_observed_mask'] == 0) | (bright['straight_observed_mask'] == 255))
    checks['binary_masks_and_input_arrays_unchanged'] = True
    assert bright['outward_extension'] is not None
    assert min(x[0] for x in bright['free_segment']['observed_segment_xy']) < 55
    checks['outward_real_edge_extension'] = bright['outward_extension']['visible_end_xy']
    narrow = scene(); narrow['a1_record']['free_segment']['width_px'] = 5.
    widened = refine_and_extend(narrow)
    assert widened['status'] == 'observed' and widened['free_segment']['width_px'] > 12.
    checks['narrow_seed_expands_only_from_measured_outer_pairs'] = {'width_px': widened['free_segment']['width_px'], 'diagnostic': widened['diagnostics']['wide_remeasurement']['geometry_action']}
    assert not np.any(bright['straight_inferred_mask'] & bright['straight_observed_mask'])
    checks['observed_inferred_disjoint'] = True
    here = Path(__file__).resolve().parent
    files = [here / 'free_rope.py', Path(__file__), *map(Path, dependencies())]
    receipt = {'version': 'IR-A3.0.0', 'scope': 'synthetic free/straight mechanism checks only', 'passed': True,
               'elapsed_s': time.perf_counter() - started, 'checks': checks,
               'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
               'real_images_run': 0, 'equipment_access': False, 'reserved_inputs': False, 'accuracy': None}
    output = here / 'free_rope_synthetic_002.json'
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
