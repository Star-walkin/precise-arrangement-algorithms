"""Read-only core review: synthetic arrays only; never decodes project imagery."""
from pathlib import Path
import hashlib
import json
import copy
import numpy as np
import cv2
import whole_path as wp
import free_rope
import run_whole

HERE = Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def fixture(shape=(220, 400), origin=(100., 110.), end=(140., 110.), width=20.):
    image = np.full((*shape, 3), 90, np.uint8)
    image[101:120] = 190
    surface = np.full(shape, 255, np.uint8)
    free_mask = np.zeros(shape, np.uint8)
    cv2.rectangle(free_mask, (20, 101), (100, 119), 255, -1)
    empty = np.zeros(shape, np.uint8)
    free = {'status': 'observed', 'geometry': {
        'status': 'observed', 'direction_xy': [1., 0.], 'width_px': width,
        'origin_xy': list(origin), 'visible_end_xy': list(end)},
        'free_segment': {'observed_segment_xy': [[20., 110.], list(origin)]},
        'free_mask': free_mask, 'straight_observed_mask': empty,
        'straight_inferred_mask': empty, 'texture_descriptor': {'status': 'unknown'}}
    data = {'surface': surface, 'original': image, 'curve_refs': [], 'pitch': 24.}
    return data, free

def main():
    before = {p.name: digest(p) for p in [HERE/'whole_path.py', HERE/'run_whole.py', HERE/'free_rope.py']}
    checks = {}
    for name in ['whole_path.py', 'run_whole.py', 'free_rope.py', 'whole_io.py', 'boundary_occlusion.py']:
        compile((HERE/name).read_text(encoding='utf-8'), str(HERE/name), 'exec')
    checks['compilation'] = {'passed': True}
    layers = [[{'tag': 'seed', 'parent': 0}],
              [{'tag': 'A', 'parent': 0}, {'tag': 'B', 'parent': 0}],
              [{'tag': 'B2', 'parent': 1}], [{'tag': 'B3', 'parent': 0}]]
    dead = [v['tag'] for v in wp._reconstruct(layers[:2], 0)]
    live = [v['tag'] for v in wp._reconstruct(layers, 0)]
    assert dead == ['seed', 'A'] and live == ['seed', 'B', 'B2', 'B3']
    checks['dead_terminal_depth_and_index'] = {'passed': True, 'dead_path': dead, 'live_path': live}
    polarity = []
    for inverted in [False, True]:
        image = np.full((220, 220, 3), 190 if inverted else 90, np.uint8)
        image[101:120] = 90 if inverted else 190
        score, supported, width, detail = wp._profiles(wp._fields(image, 20.), np.array([[110., 110.]]), np.array([0.]), 20., 1., None)
        polarity.append({'inverted': inverted, 'score': float(score[0]), 'supported': bool(supported[0]), 'raw_weak': float(detail['raw_weak'][0])})
    assert all(v['supported'] for v in polarity)
    checks['dual_polarity'] = {'passed': True, 'cases': polarity}
    data, free = fixture()
    data['surface'][:, 178:186] = 0
    result = wp.trace_whole(data, free, {'candidates': []})
    endpoint = result['candidates'][0]['xy'][-1]
    checks['short_sam_gap'] = {'runtime_passed': True, 'status': result['status'],
        'stop_reason': result['diagnostics']['stop_reason'], 'endpoint_xy': endpoint,
        'gap_x_half_open': [178, 186], 'configured_exit_budget_px': 16.,
        'early_stop_despite_reentry_next_step': bool(endpoint[0] < 186),
        'expected': 'Continue until per-branch exit budget expires; the next x sample re-enters the surface.'}
    original_profiles, original_zones = wp._profiles, wp._tangent_zones
    original_parameters = copy.deepcopy(wp.PARAMETERS)
    try:
        def profiles(fields, centers, slopes, width, xsign, texture=None):
            support = (centers[:, 0] <= 180.) | (slopes > .03)
            score = np.where(support, .35, 0.)
            score += np.where((slopes <= .03) & support, .65, 0.)
            return score, support, np.full(len(centers), width), {'raw_weak': np.where(support, 100., 0.)}
        wp._profiles = profiles
        wp._tangent_zones = lambda *args: [[110., 110.]]
        # More beams isolate reconstruction from deliberate search pruning.
        wp.PARAMETERS['beam'] = 500
        wp.PARAMETERS['turn_radians'] = .06
        data, free = fixture((600, 650))
        result = wp.trace_whole(data, free, {'candidates': []})
        candidates = [{'end_reason': c['end_reason'], 'points': len(c['xy']), 'end': c['xy'][-1],
                       'length_px': c['length_px']} for c in result['candidates']]
        has_dead = any(c['end_reason'] == 'original_support_end' for c in candidates)
        has_longer = any(c['end'][0] > 300. for c in candidates)
        assert has_dead and has_longer
        checks['dead_branch_trace_execution'] = {'passed': True, 'candidate_count': len(candidates), 'earlier_terminal_retained': has_dead, 'different_survivor_continues': has_longer, 'candidates': candidates}
    finally:
        wp._profiles, wp._tangent_zones = original_profiles, original_zones
        wp.PARAMETERS.clear(); wp.PARAMETERS.update(original_parameters)
    local = {str(p.resolve()) for p in HERE.glob('*.py')}
    external = [str(Path(p).resolve()) for p in free_rope.dependencies()]
    runner_sources = local | {str(Path(p).resolve()) for p in run_whole.dependencies()}
    basenames = [Path(p).name for p in runner_sources]
    checks['external_source_freeze'] = {'runtime_dependencies': {p: digest(Path(p)) for p in external},
        'not_covered_by_current_directory_glob': [p for p in external if p not in local],
        'missing_from_updated_runner_union': [p for p in external if p not in runner_sources],
        'snapshot_flat_basename_collision': len(basenames) != len(set(basenames)),
        'source_review': 'run_whole now explicitly unions dependencies() before hashing and copying.'}
    assert not checks['external_source_freeze']['missing_from_updated_runner_union']
    assert not checks['external_source_freeze']['snapshot_flat_basename_collision']
    after = {p.name: digest(p) for p in [HERE/'whole_path.py', HERE/'run_whole.py', HERE/'free_rope.py']}
    report = {'review_type': 'synthetic_and_read_only_code_review', 'source_sha256_before': before,
              'source_sha256_after': after, 'core_unchanged_during_checks': before == after, 'checks': checks,
              'remaining_nonblocking_limits': [
                  'outside_distance counts x-step rather than path arc distance, so a steep path receives a longer geometric exit allowance.',
                  'Only evidence-invalid terminals are explicitly saved. Beam and bin pruning remain approximate and can discard a branch before it exhausts the weak-support budget.',
                  'The support fraction and highest appearance score do not establish same-material identity.'
              ],
              'not_claimed': ['real-image visual correctness', 'material identity', 'contact or layer event']}
    (HERE/'code_review_002_followup.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))

if __name__ == '__main__':
    main()
