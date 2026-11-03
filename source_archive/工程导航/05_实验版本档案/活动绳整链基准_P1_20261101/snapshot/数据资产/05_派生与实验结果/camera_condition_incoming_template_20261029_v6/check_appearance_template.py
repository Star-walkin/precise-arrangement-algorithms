"""Independent constructed mechanisms, not image-accuracy evaluation."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import appearance_template as module


HERE = Path(__file__).resolve().parent


def scene(angle=0., phase=0., opposite=False, inverted=False, gain=1., offset=0.,
          stripe_width=20., saturated=False, repeated=False):
    height, width = 600, 800
    yy, xx = np.mgrid[:height, :width]
    direction = np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([-direction[1], direction[0]])
    center = np.array([400., 300.]); points = np.c_[xx.ravel(), yy.ravel()]-center
    along = (points@direction).reshape(height, width)
    across = (points@normal).reshape(height, width)
    v = (across+22.) % 44.-22. if repeated else across
    background = 48.+.012*along+.015*across
    texture = 12.*np.sin(2*np.pi*(.8*along/20.+(-.8 if opposite else .8)*v/20.)+phase)
    body = 118.+texture+.012*along
    gray = np.where(abs(v) <= stripe_width/2, 255. if saturated else body, background)
    if inverted:
        gray = 210.-gray
    gray = np.clip(gray*gain+offset, 0., 255.).astype(np.uint8)
    image = np.repeat(gray[:, :, None], 3, axis=2)
    anchors = center+np.linspace(-200., 200., 21)[:, None]*direction
    free = {'status': 'observed', 'free_segment': {
        'status': 'observed', 'observed_points_xy': anchors.tolist(),
        'boundary_left_xy': (anchors-10*normal).tolist(),
        'boundary_right_xy': (anchors+10*normal).tolist(),
        'direction_xy': direction.tolist(), 'width_px': 20.}}
    return {'original': image, 'surface': np.zeros((height, width), np.uint8), 'pitch': 40.}, free, anchors, direction


def pack(value):
    keys = ['score', 'valid', 'supported', 'profile_match', 'texture_match', 'texture_valid',
            'width_match', 'raw_pair', 'delta_gray', 'saturated_outline_supported',
            'neighbor_advantage', 'neighbor_comparison_valid']
    return {k: np.asarray(value[k]).tolist() for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--receipt', default='appearance_template_synthetic_001.json')
    args = ap.parse_args(); out = HERE/args.receipt
    if out.exists():
        raise ValueError('Never overwrite a mechanism receipt')
    results = {}; errors = []
    def require(name, condition):
        if not bool(condition):
            errors.append(name)
    data, free, anchors, direction = scene()
    image_before = hashlib.sha256(data['original'].tobytes()).hexdigest()
    template = module.build_template(data, free)
    require('measured_template_ready', template['status'] == 'ready')
    require('measured_template_texture_available', template.get('texture_available'))
    require('template_outside_surface_only', max(template.get('accepted_surface_overlap_fraction', [1.])) == 0.)
    points = anchors[6:15:2]
    for name, arguments in [
        ('same_free_appearance', {}), ('gain_and_offset', {'gain': .65, 'offset': 35.}),
        ('polarity_reversal', {'inverted': True}), ('strand_phase_translation', {'phase': 1.7}),
        ('wrong_strand_slant', {'opposite': True}), ('narrow_strand', {'stripe_width': 6.}),
        ('rotated_rope', {'angle': .52}), ('white_clipped_body', {'saturated': True}),
        ('periodically_identical_neighbors', {'repeated': True}),
    ]:
        sample, _, locations, tangent = scene(**arguments)
        locations = locations[6:15:2]
        value = module.evaluate_template(module.build_fields(sample['original'], 20., 40.), locations, tangent, 20., template)
        results[name] = pack(value)
        if name in ['same_free_appearance', 'gain_and_offset', 'polarity_reversal', 'strand_phase_translation', 'rotated_rope']:
            require(name+'_retains_measured_ribbon', np.all(value['supported']))
            require(name+'_profile_match', np.median(value['profile_match']) > .75)
        if name == 'narrow_strand':
            require('narrow_strand_not_whole_rope', not np.any(value['supported']))
        if name == 'white_clipped_body':
            require('clipped_outline_independent_channel', np.all(value['saturated_outline_supported']) and not np.any(value['raw_reliable']))
            require('clipped_means_no_background_advantage', not np.any(value['neighbor_comparison_valid']))
        if name == 'periodically_identical_neighbors':
            require('identical_neighbors_comparison_available', np.all(value['neighbor_comparison_valid']))
            require('identical_neighbors_have_no_invented_advantage', abs(np.median(value['neighbor_advantage'])) < .03)
            require('identical_neighbors_no_identity_claim', value['identity_verified'] is False)
    require('template_texture_discriminates_slant', np.median(results['same_free_appearance']['texture_match']) > np.median(results['wrong_strand_slant']['texture_match'])+.20)
    require('template_texture_changes_score', np.median(results['same_free_appearance']['score']) > np.median(results['wrong_strand_slant']['score'])+.02)
    fields = module.build_fields(data['original'], 20., 40.)
    forward = module.evaluate_template(fields, points, direction, 20., template)
    backward = module.evaluate_template(fields, points, -direction, 20., template)
    require('traversal_direction_invariant', np.allclose(forward['score'], backward['score'], atol=1e-6))
    # A valid-looking geometric source located on the winding surface cannot
    # teach the self template. The whole measurement window is checked.
    contaminated = dict(data); contaminated['surface'] = np.full(data['surface'].shape, 255, np.uint8)
    invalid_template = module.build_template(contaminated, free)
    require('contaminated_template_rejected', invalid_template['status'] == 'unknown')
    unknown = module.evaluate_template(fields, points, direction, 20., invalid_template)
    require('unknown_template_cannot_produce_support', not np.any(unknown['supported']) and not np.any(unknown['valid']))
    results['contaminated_template'] = invalid_template
    for name, raw in [('all_white', 255), ('black', 0), ('flat_gray', 75)]:
        field = module.build_fields(np.full_like(data['original'], raw), 20., 40.)
        value = module.evaluate_template(field, points, direction, 20., template)
        require(name+'_unknown_not_support', not np.any(value['valid']) and not np.any(value['supported']))
        results[name] = pack(value)
    require('source_array_not_mutated', image_before == hashlib.sha256(data['original'].tobytes()).hexdigest())
    timings = {}
    for count in [80, 240]:
        query = np.tile(points, (int(np.ceil(count/len(points))), 1))[:count]
        start = time.perf_counter()
        batch = module.evaluate_template(fields, query, direction, 20., template)
        timings[str(count)] = time.perf_counter()-start
        require('batch_'+str(count)+'_matches_single', np.allclose(batch['score'], np.resize(forward['score'], count), atol=1e-6))
        require('batch_'+str(count)+'_finite', all(np.isfinite(v).all() for v in batch.values() if isinstance(v, np.ndarray)))
    receipt = {
        'scope': 'Constructed raw-image mechanism checks only; no real image/label/identity accuracy evaluation.',
        'source_sha256': hashlib.sha256((HERE/'appearance_template.py').read_bytes()).hexdigest(),
        'test_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'A5_readonly_dependency_sha256': hashlib.sha256(Path(module.dependencies()[0]).read_bytes()).hexdigest(),
        'template': template, 'cases': results, 'batch_seconds': timings,
        'passed': not errors, 'errors': errors, 'exit_code': int(bool(errors)),
    }
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'passed': not errors, 'errors': errors, 'batch_seconds': timings}, ensure_ascii=False))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
