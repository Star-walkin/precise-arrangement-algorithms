"""Constructed clipped-free to nonclipped-ribbon mechanism, no real images."""
from pathlib import Path
import hashlib
import json
import numpy as np
import appearance_template as module
from check_appearance_template import scene


def main():
    root = Path(__file__).resolve().parent
    output = root/'appearance_outline_transition_synthetic_001.json'
    if output.exists():
        raise ValueError('Never overwrite mechanism evidence')
    source, free, _, _ = scene(saturated=True)
    template = module.build_template(source, free)
    errors = []; cases = {}
    def require(name, condition):
        if not bool(condition):
            errors.append(name)
    require('source_only_outline_known', template['status'] == 'outline_only' and not template['profile_available'] and not template['texture_available'])
    require('outline_learned_from_measured_edges', template.get('outline_geometry', {}).get('source_samples', 0) >= 4)
    specifications = [
        ('dark_ribbon', {'inverted': True}, True),
        ('dark_lower_exposure', {'inverted': True, 'gain': .22, 'offset': 10.}, True),
        ('nonclipped_bright_ribbon', {}, True),
        ('narrow_dark_strand', {'inverted': True, 'stripe_width': 6.}, False),
        ('still_white_clipped', {'saturated': True}, True),
        ('identical_old_neighbor_ribbons', {'inverted': True, 'repeated': True}, True),
    ]
    for name, arguments, expected_support in specifications:
        data, _, points, tangent = scene(**arguments)
        value = module.evaluate_template(module.build_fields(data['original'], 20., 40.), points[6:15:2], tangent, 20., template)
        require(name+'_support_state', bool(value['supported'].all()) if expected_support else not value['supported'].any())
        require(name+'_no_invented_self_profile', not value['profile_valid'].any())
        require(name+'_no_invented_self_texture', not value['texture_valid'].any())
        require(name+'_no_clipped_mean_advantage', not value['neighbor_comparison_valid'].any() and np.all(value['neighbor_advantage'] == 0))
        if name in ('dark_ribbon', 'dark_lower_exposure', 'nonclipped_bright_ribbon'):
            require(name+'_new_independent_outline_channel', value['outline_transition_supported'].all())
        require(name+'_no_identity_claim', value['identity_verified'] is False)
        keys = ['score', 'supported', 'valid', 'raw_supported', 'profile_valid', 'texture_valid',
                'outline_template_valid', 'outline_template_match', 'outline_geometry_error',
                'outline_transition_supported', 'saturated_outline_supported', 'raw_pair', 'measured_width_px']
        cases[name] = {k: value[k].tolist() for k in keys}
    data, _, points, tangent = scene()
    for name, gray in [('all_white', np.full(data['surface'].shape, 255, np.uint8)),
                       ('all_dark', np.zeros(data['surface'].shape, np.uint8)),
                       ('single_step', np.where(np.indices(data['surface'].shape)[0] < 300, 70, 145).astype(np.uint8))]:
        image = np.repeat(gray[:, :, None], 3, axis=2)
        value = module.evaluate_template(module.build_fields(image, 20., 40.), points[6:15:2], tangent, 20., template)
        require(name+'_no_outline_support', not value['supported'].any() and not value['outline_transition_supported'].any())
        cases[name] = {'score': value['score'].tolist(), 'supported': value['supported'].tolist(), 'raw_pair': value['raw_pair'].tolist()}
    receipt = {
        'scope': 'Pure constructed exposure-transition mechanism, not field accuracy.',
        'prior_failure': 'appearance_outline_transition_failure_001.json',
        'prior_source_snapshot': 'template_mechanism_002_before/appearance_template.py',
        'source_sha256': hashlib.sha256((root/'appearance_template.py').read_bytes()).hexdigest(),
        'test_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'template': template, 'cases': cases, 'passed': not errors, 'errors': errors,
        'existing_A5_contrast_width_support_thresholds_changed': False,
        'unknown_self_profile_or_texture_restored': False,
    }
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'passed': not errors, 'errors': errors, 'source_sha256': receipt['source_sha256']}))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
