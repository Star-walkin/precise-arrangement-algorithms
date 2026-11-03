"""Read saved fixed-case records; no algorithm import, image prediction or mutation."""
import json
import hashlib
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
CASES = [
    ('left_camera', 'L_run_20260923_01__f000015__left'),
    ('right_camera', 'dual_20260923_101643__pair0315__right'),
    ('right_camera', 'dual_20260923_113305__pair0015__right'),
    ('right_camera', 'dual_20260923_113305__pair0195__right'),
]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def difference(a, b):
    pa = np.array(a['search_xy']); pb = np.array(b['search_xy'])
    pb = pb[np.argsort(pb[:, 0])]
    valid = (pa[:, 0] >= pb[0, 0]) & (pa[:, 0] <= pb[-1, 0])
    delta = np.abs(pa[:, 1] - np.interp(pa[:, 0], pb[:, 0], pb[:, 1]))
    return delta, valid

def candidate_summary(c, g):
    p = np.array(c['search_xy']); w = np.array(c['search_widths'])
    direction = np.array(g['direction_xy']); normal = np.array([-direction[1], direction[0]])
    rel = p - np.array(g['origin_xy']); off = np.abs(rel @ normal)
    s = rel @ direction; angle = np.array(c['angle_from_free_deg'])
    bent = angle >= 3
    ev = c['evidence']; direct = np.array(c['direct_template_supported'])
    trusted = max(0, int(c.get('trusted_prefix_count', 0)) - int(c['search_prefix_offset']))
    half = np.flatnonzero(off > .5 * g['width_px'])
    def median(name):
        a = np.array(ev[name]); return float(np.median(a[bent])) if bent.any() else None
    return {
        'candidate_id': c['candidate_id'], 'length_px': c['length_px'],
        'turn_anchor_bin': c.get('turn_anchor_bin'),
        'image_score': c['image_only_score'], 'combined_score': c['late_combined_score'],
        'first_half_width_departure_s_px': float(s[half[0]]) if len(half) else None,
        'endpoint_xy': p[-1].tolist(), 'trusted_points': trusted,
        'trusted_endpoint_xy': p[trusted - 1].tolist() if trusted else None,
        'trusted_maximum_offset_widths': float(np.max(off[:trusted]) / g['width_px']) if trusted else None,
        'trusted_bent_direct_points': int(np.sum(bent[:trusted] & direct[:trusted])),
        'bent_point_count': int(bent.sum()),
        'bent_direct_supported_fraction': float(np.mean(direct[bent])) if bent.any() else None,
        'bent_median_profile_match': median('profile_match'),
        'bent_median_texture_match': median('texture_match'),
        'bent_median_groove_score': float(np.median(np.array(c['original_groove_score'])[bent])) if bent.any() else None,
        'bent_saturated_outline_points': int(np.sum(np.array(ev['saturated_outline_supported'])[bent])),
    }

def replay_prefix(chosen, competitors, entry):
    p = np.array(chosen['search_xy']); widths = np.array(chosen['search_widths'])
    w0 = float(entry['free']['geometry']['width_px']); ev = chosen['evidence']
    direct = np.array(chosen['direct_template_supported']); angle = np.array(chosen['angle_from_free_deg'])
    unique = np.array(ev['neighbor_comparison_valid']) & (np.array(ev['neighbor_advantage']) > .08)
    local = direct & ((angle < 3) | unique)
    cross = np.array(chosen['late_F3']['body_crossing_suspect'])
    divergence = np.zeros(len(p), bool); per = []
    for other in competitors:
        if other['candidate_id'] == chosen['candidate_id']: continue
        delta, valid = difference(chosen, other)
        split = valid & (delta > .4 * widths)
        divergence |= split
        inds = np.flatnonzero(split)
        other_supported = np.array(other['direct_template_supported'])
        per.append({'id': other['candidate_id'], 'length_px': other['length_px'],
                    'combined_score': other['late_combined_score'],
                    'overlap_chosen_points': int(valid.sum()),
                    'divergent_chosen_indices': inds.tolist(),
                    'first_divergence_xy': p[inds[0]].tolist() if len(inds) else None,
                    'other_direct_supported_indices': np.flatnonzero(other_supported).tolist()})
    weak = 0.; split_run = 0.; last = 0; reason = 'image_support_endpoint'; stop = len(p) - 1
    for i in range(1, len(p)):
        step = float(np.linalg.norm(p[i] - p[i - 1]))
        weak = 0. if local[i] and not cross[i] else weak + step
        split_run = split_run + step if divergence[i] else 0.
        if weak > w0:
            reason = 'self_template_or_edges_no_longer_distinct'; stop = i; break
        if split_run > w0:
            reason = 'retained_branches_diverge'; stop = i; break
        if local[i] and not divergence[i]: last = i
    return {'count': max(1, last + 1), 'reason': reason, 'stop_index': stop,
            'first20_direct': direct[:20].tolist(), 'first20_local': local[:20].tolist(),
            'first20_divergent': divergence[:20].tolist(), 'competitor_diagnostics': per}

report = {'scope': 'Read-only saved evidence plus visual inspection of four registered fixed cases. No ground-truth accuracy, no additional image selection, no prediction rerun.',
          'cases': [], 'snapshot_source_sha256': {},
          'native_crop_manifest': str(BASE / 'pilot_002/qa_code_audit_native/manifest.json')}
for run in ['pilot_001', 'pilot_002']:
    report['snapshot_source_sha256'][run] = {name: sha(BASE/run/'source_snapshot'/name) for name in ['image_beam.py', 'pipeline.py', 'appearance_template.py']}
for camera, name in CASES:
    item = {'camera': camera, 'name': name, 'runs': {}}
    for run in ['pilot_001', 'pilot_002']:
        path = BASE/run/'记录'/camera/(name+'.json')
        saved = json.loads(path.read_text(encoding='utf-8')); result = saved['result']
        selected = next(c for c in result['candidates'] if c['candidate_id'] == result['selected_candidate_id'])
        entry = next(e for e in result['entries'] if e['entry_id'] == selected['entry_id'])
        g = entry['free']['geometry']
        competitors = [c for c in result['candidates'] if c['entry_id'] == selected['entry_id'] and selected['late_combined_score'] - c['late_combined_score'] < .12]
        rr = {'record_sha256': sha(path), 'status': result['status'],
              'pool_count': result['diagnostics']['image_pool_count'],
              'retained_count': len(result['candidates']),
              'selected': candidate_summary(selected, g),
              'all_retained': [candidate_summary(c, g) for c in result['candidates']],
              'template_status': entry['template']['status'],
              'template_reference_width_px': entry['template']['reference_width_px'],
              'geometry_width_px': g['width_px'],
              'saved_prefix_reason': result['diagnostics']['trusted_prefix_stop_reason']}
        if run == 'pilot_002':
            rr['saved_logic_prefix_replay'] = replay_prefix(selected, competitors, entry)
            rr['omit_short_reference_replay_diagnostic_only'] = replay_prefix(selected, [c for c in competitors if c['length_px'] >= 8*g['width_px']], entry)
            assert rr['saved_logic_prefix_replay']['count'] == result['diagnostics']['trusted_search_point_count']
        item['runs'][run] = rr
    report['cases'].append(item)

report['visual_findings'] = {
    'f15': 'The selected path still flattens into an old coil at the bright band. The new later-anchor path dives downward through several visible coil lanes; the upward branch is also not the visible diagonal continuation. No retained candidate can be claimed to cover the full visible straight rope through the bright band. Green prefix is shorter, not newly following an early wrong bend.',
    'right315': 'A long diagonal candidate now overlaps the visible incoming dark rope much better than pilot001 early flattened candidates. Several lower hypotheses still cross coil bodies. The top candidate truncates before complete dark-side continuation. The visible green prefix does not extend into the obvious disputed bend.',
    'right113305_pair0015': 'The new pool now includes a long diagonal crossing the bright band, absent as an unbent alternative in pilot001. The highest score still turns/levels early onto a neighboring old coil. No trusted green bend is asserted: only three search points survive.',
    'right113305_pair0195': 'The new pool preserves a much later diagonal/lower continuation. The top-scoring path is still an early flat old-coil route. Whether the very dark terminal part remains same rope is not established. Only six search points are green; the visibly wrong early bends remain review hypotheses.',
}
report['interpretation'] = [
    'Candidate diversity genuinely improved on three of these four fixed cases; it is not a demonstrated whole-rope accuracy increase.',
    'All four selected templates are ready, not outline_only. Saturated-outline-only matching is not the common cause of these four failures.',
    'Both wrong old-coil routes with strong self-profile similarity and generic groove-supported routes survive. A real groove, high self texture score or F3 agreement cannot certify same material.',
    'Trusted prefix uses only displayed candidates and global tail-score margin in pilot002. It is not the common prefix of the entire generated pool.',
    'Full generated geometry is absent from the pilot002 image_pool_ledger (hash only), so the discarded pool cannot be independently reconstructed from this run. Review conclusions about missing target apply to saved/retained candidates, not hidden intermediate states.',
]
report['minimal_logic_recommendations'] = [
    'Store every image-generated pool candidate with coordinates and per-point support before display filtering. Derive prefix competition from that pool, not six rendered paths or global terminal-score margin.',
    'Use local overlapping, image-supported windows to test a fork. A short partial-axis reference that ends before the disputed bend must not assert continuation or impose a permanent split based on unsupported end drift. Preserve it as a clearly named local reference, without claiming same-root identity.',
    'Keep turn-anchor history in the per-step state dedup key before round-robin selection; otherwise later-family protection can be erased before the family allocator runs. Restrict global endpoint dedup to the same family, or verify complete overlap, so distinct approach histories survive.',
    'Before rewarding a turn to a generic background groove, retain and score the directly observed axial alternative in the same local window. A high late score on an old coil must not defeat an available image-supported straight continuation based on unrelated terminal segments.',
    'Do not relax raw support thresholds or fabricate dark-side identity. For f15 first verify optical whole-rope width and whether a continuing axial hypothesis actually reaches the pool; turn-anchor diversity alone has not recovered it.',
]
report['limitations'] = ['Only the four requested registered development cases were visually inspected for this report.', 'No labels, reserved set, devices, baseline source edits or prediction reruns.', 'Visual observations are qualitative diagnostics, not per-pixel annotations or material identity truth.']
out = BASE/'pilot_002'/'independent_template_beam_visual_review.json'
assert not out.exists(), out
out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
print(str(out))
for item in report['cases']:
    p = item['runs']['pilot_002']; chosen = p['selected']
    print(item['name'], 'selected', chosen['candidate_id'], 'trusted', chosen['trusted_points'], 'omit-short', p['omit_short_reference_replay_diagnostic_only']['count'], 'outline', chosen['bent_saturated_outline_points'], 'profile', chosen['bent_median_profile_match'], 'groove', chosen['bent_median_groove_score'])
    if item['name'].startswith('L_run'):
        print(json.dumps(p['saved_logic_prefix_replay'], ensure_ascii=False))
