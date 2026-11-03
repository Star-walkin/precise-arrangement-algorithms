"""Read-only saved-record comparison for the two explicitly requested pilot003 cases."""
from pathlib import Path
import hashlib
import json
import numpy as np

BASE = Path(__file__).resolve().parent
CASES = [('left_camera', 'L_run_20260923_01__f000015__left'),
         ('right_camera', 'dual_20260923_101643__pair0315__right')]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

report = {
    'scope': 'Two registered fixed development cases; saved evidence and actual native-pixel visual inspection. No algorithm rerun, no extra cases, no labels or reserved inputs.',
    'cases': [],
    'source_sha256': {},
    'native_crop_manifest': str(BASE/'pilot_003/qa_code_audit_native/manifest.json'),
    'native_crop_manifest_sha256': sha(BASE/'pilot_003/qa_code_audit_native/manifest.json'),
}
for run in ['pilot_002', 'pilot_003']:
    report['source_sha256'][run] = {name: sha(BASE/run/'source_snapshot'/name) for name in ['image_beam.py', 'pipeline.py', 'appearance_template.py']}
for camera, name in CASES:
    row = {'camera': camera, 'name': name, 'runs': {}}
    for run in ['pilot_002', 'pilot_003']:
        path = BASE/run/'记录'/camera/(name+'.json')
        result = json.loads(path.read_text(encoding='utf-8'))['result']
        c = next(c for c in result['candidates'] if c['candidate_id'] == result['selected_candidate_id'])
        entry = next(e for e in result['entries'] if e['entry_id'] == c['entry_id'])
        p = np.array(c['search_xy']); count = int(result['diagnostics']['trusted_search_point_count'])
        g = entry['free']['geometry']; d = np.array(g['direction_xy']); n = np.array([-d[1], d[0]])
        offset = np.abs((p-np.array(g['origin_xy']))@n)
        pool = result.get('candidate_pool', [])
        axis = next((a for a in pool if a['candidate_id'].endswith('axis_reference')), None)
        row['runs'][run] = {
            'record_sha256': sha(path), 'status': result['status'],
            'selected_candidate_id': c['candidate_id'],
            'selected_length_px': c['length_px'],
            'selected_combined_score': c['late_combined_score'],
            'selected_search_xy_sha256': hashlib.sha256(json.dumps(c['search_xy'],separators=(',',':')).encode()).hexdigest(),
            'selected_endpoint_xy': p[-1].tolist(),
            'trusted_search_point_count': count,
            'trusted_endpoint_xy': p[count-1].tolist() if count else None,
            'trusted_maximum_offset_widths': float(offset[:count].max()/g['width_px']) if count else None,
            'prefix_reason': result['diagnostics']['trusted_prefix_stop_reason'],
            'prefix_competition_scope': result['diagnostics'].get('prefix_competition_scope', 'displayed candidates within global tail score margin'),
            'generated_pool_count': result['diagnostics']['image_pool_count'],
            'retained_display_count': len(result['candidates']),
            'generated_pool_geometry_saved': bool(pool),
            'axis_reference': None if axis is None else {
                'length_px': axis['length_px'], 'endpoint_xy': axis['xy'][-1],
                'combined_score': axis['late_combined_score'],
                'in_displayed_candidates': any(a['candidate_id']==axis['candidate_id'] for a in result['candidates']),
                'direct_supported_points': int(sum(axis['direct_template_supported'])),
                'point_count': len(axis['xy']),
            },
        }
    report['cases'].append(row)

report['visual_observations'] = [
    {'case': CASES[0][1], 'finding': 'The pilot003 overlay contains the same long early-flattening best candidate and the same conspicuous upward/downward alternatives as pilot002. The green prefix now extends farther along the visible straight free rope. The newly recorded exact-axis reference ends near the bright-band entrance and is absent from displayed candidates after overlap dedup. It does not recover the whole visible diagonal continuation or a verified bend.'},
    {'case': CASES[1][1], 'finding': 'A short exact-axis reference is now ranked first. Its green segment remains on the visible straight incoming rope and stops earlier than the previous selected long path. Long diagonal and wrong lower-turning hypotheses are still visible as review candidates. This is a changed partial-path choice; no complete dark-side same-root path is confirmed.'},
]
report['implemented_improvements_confirmed'] = [
    'Prefix competitors are collected before display pruning and no longer discarded using the unrelated terminal-region score difference.',
    'The candidate pool now stores coordinates, widths, local support flags and scores, enabling independent geometric comparison of unrendered alternatives.',
    'A strict self-template-supported exact-axis reference is added; f15 contains a 464px axis reference and right315 a 528px reference. Unsupported end tails are not claimed as trusted.',
    'f15 best geometry is byte-identical between pilot002 and pilot003, while trusted count changes from 6 to 44. That isolates the improvement to evidence/prefix handling rather than a newly correct bend.',
]
report['remaining_limits'] = [
    'A high local self-profile or groove score still admits an old-coil path. F3 is post-generation consistency only, not proof of identity.',
    'Ranking scores still average different terminal windows for short and long candidates. In right315 the short 528px reference wins against the 1288px continuation; this comparison does not establish the long candidate is wrong.',
    'The exact-axis reference may be present only in the full recorded pool and omitted from the rendered candidate set when its overlap is treated as redundant, as in f15.',
    'More conservative or longer green prefixes do not alone measure accuracy. This review has no per-pixel ground-truth masks and does not verify material identity, true contact or the three physical process nodes.',
    'No further threshold or algorithm edits were made by this reviewer. The third-pilot budget should not be presented as an iterative success claim.',
]
report['pilot002_f15_causal_clarification'] = {
    'short_reference_length_px': 56,
    'all_eight_short_reference_points_direct_supported': True,
    'split_indices_in_pilot002_selected_path': [6, 7],
    'consequence': 'The previous stop was not caused by an unsupported tail. It arose from a small center-location difference in a narrow free ribbon and the overlap-coordinate/split rule. A diagnostic omission of that short candidate yielded 44 points, but no length-based exclusion was proposed as an algorithm fix.',
}
out = BASE/'pilot_003/independent_fixed_two_visual_review.json'
assert not out.exists(), out
out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(out)
print('source stable appearance:', report['source_sha256']['pilot_002']['appearance_template.py']==report['source_sha256']['pilot_003']['appearance_template.py'])
