"""Strict raw-image constructed checks for the finite axial re-acquisition helper."""
from pathlib import Path
import hashlib
import json
import numpy as np
from check_appearance_template import scene
from appearance_template import build_fields, build_template
from axial_reacquisition import probe_axis_bridge


def constructed(kind):
    data, free, _, _ = scene()
    yy, xx = np.indices(data['surface'].shape)
    background = np.clip(48.+.012*(xx-400)+.015*(yy-300), 0, 255).astype(np.uint8)
    replace = np.zeros_like(xx, bool)
    if kind == 'stable_reappearance': replace = (xx >= 320) & (xx < 470)
    elif kind == 'no_reappearance': replace = xx >= 320
    elif kind == 'isolated_short_patch': replace = (xx >= 320) & ~((xx >= 470) & (xx < 486))
    elif kind == 'reappearance_beyond_budget': replace = (xx >= 320) & (xx < 610)
    elif kind == 'short_gap_only': replace = (xx >= 320) & (xx < 330)
    data['original'][replace] = np.repeat(background[:, :, None], 3, axis=2)[replace]
    data['surface'][:, 550:] = 255
    points = np.c_[np.linspace(80., 240., 17), np.full(17, 300.)]
    seg = free['free_segment']; seg.update(observed_points_xy=points.tolist(),
        boundary_left_xy=(points-np.array([0., 10.])).tolist(),
        boundary_right_xy=(points+np.array([0., 10.])).tolist())
    free['geometry'] = {'status': 'observed', 'origin_xy': [240., 300.], 'direction_xy': [1., 0.], 'width_px': 20.}
    return data, free


def main():
    root = Path(__file__).resolve().parent
    out = root/'axial_reacquisition_synthetic_001.json'
    if out.exists(): raise ValueError('No overwrite of mechanism receipts')
    cases = {}; errors = []
    for name in ['stable_reappearance', 'no_reappearance', 'isolated_short_patch',
                 'reappearance_beyond_budget', 'short_gap_only', 'no_gap']:
        data, free = constructed(name)
        before = hashlib.sha256(data['original'].tobytes()).hexdigest()
        template = build_template(data, free)
        result = probe_axis_bridge(build_fields(data['original'], 20., data['pitch']), data['surface'], free, template)
        cases[name] = result
        expected = name == 'stable_reappearance'
        if result['accepted'] != expected: errors.append(name+'_acceptance')
        if result['accepted']:
            q = np.array(result['xy']); supported = np.array(result['supported'])
            if not np.allclose(q[:, 1], 300.): errors.append(name+'_must_stay_on_measured_axis')
            if supported.all() or not supported[-1]: errors.append(name+'_explicit_unknown_connector')
            if result['diagnostics']['stable_span_px'] < 40.: errors.append(name+'_center_span_below2w')
            if q[-1, 0] > 550+40: errors.append(name+'_surface_probe_bound')
        if before != hashlib.sha256(data['original'].tobytes()).hexdigest(): errors.append(name+'_original_mutated')
    # No signal anywhere cannot seed a valid template or a new bridge.
    data, free = constructed('no_gap'); template = build_template(data, free)
    field = build_fields(np.zeros_like(data['original']), 20., data['pitch'])
    result = probe_axis_bridge(field, data['surface'], free, template)
    cases['all_black_no_signal'] = result
    if result['accepted']: errors.append('all_black_no_signal')
    result = probe_axis_bridge(build_fields(data['original'], 20., data['pitch']), np.zeros_like(data['surface']), free, template)
    cases['no_surface_entry'] = result
    if result['accepted']: errors.append('no_surface_entry')
    receipt = {'scope': 'Constructed original pixels only; no real image or identity accuracy.',
               'source_sha256': hashlib.sha256((root/'axial_reacquisition.py').read_bytes()).hexdigest(),
               'test_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'cases': cases, 'passed': not errors, 'errors': errors,
               'support_thresholds_changed': False}
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'passed': not errors, 'errors': errors, 'states': {k: v['accepted'] for k, v in cases.items()}}))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
