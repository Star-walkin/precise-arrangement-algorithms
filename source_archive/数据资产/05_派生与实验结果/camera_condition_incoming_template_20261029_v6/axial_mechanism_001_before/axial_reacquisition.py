"""Bounded original-image reacquisition before surface entry, not a trusted bridge.

Only the measured free axis is probed. A long missing interval remains unsupported
even if later real ribbon evidence recurs. No F3, lane snapping, image names,
devices, IO, scoring-threshold changes or fabricated positive pixels.
"""
import numpy as np
import cv2
from appearance_template import evaluate_template


PARAMETERS = {
    'sample_step_px': 4.,
    'maximum_probe_widths': 35.,
    'maximum_probe_px': 800.,
    'beyond_first_surface_widths': 1.,
    'minimum_missing_gap_widths': 6.,
    'minimum_reacquired_span_widths': 2.,
    'minimum_reacquired_samples': 4,
    'width_transition_span_widths': 4.,
    'minimum_template_width_ratio': .60,
    'maximum_template_width_ratio': 1.70,
}


def _sample(mask, xy):
    q = np.asarray(xy, np.float32)
    return cv2.remap(mask, q[:, 0][None], q[:, 1][None], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0).ravel()


def _runs(flag):
    z = np.r_[False, np.asarray(flag, bool), False].astype(np.int8)
    return list(zip(np.flatnonzero(np.diff(z) == 1), np.flatnonzero(np.diff(z) == -1)))


def probe_axis_bridge(fields, domain, free, template):
    """Return accepted, axis xy/widths/supported/score and explicit diagnostics.

    Returned positive support flags always come from evaluate_template. Missing
    connector flags remain false. The caller must NOT merge this geometric bridge
    into a trusted prefix or claim material identity across it.
    """
    out = {'accepted': False, 'xy': [], 'widths': [], 'supported': [], 'score': [],
           'reason': 'not_applicable', 'diagnostics': {
               'parameters': dict(PARAMETERS), 'F3_used': False,
               'same_material_identity_verified': False, 'unknown_connector_is_trusted': False}}
    g = free.get('geometry', free)
    if g.get('status') != 'observed' or template.get('status') not in ('ready', 'outline_only'):
        out['reason'] = 'no_measured_axis_or_template'; return out
    o = np.asarray(g['origin_xy'], float); d = np.asarray(g['direction_xy'], float)
    w0 = float(g['width_px']); norm = float(np.linalg.norm(d))
    if o.shape != (2,) or d.shape != (2,) or not np.isfinite(np.r_[o, d, w0]).all() or norm <= 0 or w0 <= 1:
        raise ValueError('Finite original-coordinate free axis and width > 1 required')
    if tuple(domain.shape) != tuple(fields['image_shape']):
        raise ValueError('Search domain and original field dimensions differ')
    d = d/norm
    if _sample(domain, [o])[0] > 76:
        out['reason'] = 'origin_already_in_surface'; return out
    hard_limit = min(PARAMETERS['maximum_probe_px'], PARAMETERS['maximum_probe_widths']*w0)
    step = PARAMETERS['sample_step_px']
    scan_s = np.r_[np.arange(0., hard_limit, step), hard_limit]
    scan_xy = o+scan_s[:, None]*d
    first_ids = np.flatnonzero(_sample(domain, scan_xy) > 76)
    if not len(first_ids):
        out['reason'] = 'no_surface_entry_within_axial_budget'
        out['diagnostics']['hard_probe_limit_px'] = float(hard_limit); return out
    first = int(first_ids[0]); first_s = float(scan_s[first])
    limit = min(hard_limit, first_s+PARAMETERS['beyond_first_surface_widths']*w0)
    ss = np.r_[np.arange(0., limit, step), limit]
    xy = o+ss[:, None]*d
    h, w = domain.shape
    inside_frame = (xy[:, 0] >= 1) & (xy[:, 0] < w-2) & (xy[:, 1] >= 1) & (xy[:, 1] < h-2)
    if not np.all(inside_frame):
        stop = int(np.flatnonzero(~inside_frame)[0]); ss = ss[:stop]; xy = xy[:stop]
    if len(xy) < PARAMETERS['minimum_reacquired_samples']:
        out['reason'] = 'insufficient_in_frame_probe'; return out
    width_hypotheses = [('measured_geometry_width', np.full(len(ss), w0))]
    reference = float(template.get('reference_width_px', w0))
    if PARAMETERS['minimum_template_width_ratio'] <= reference/w0 <= PARAMETERS['maximum_template_width_ratio'] and abs(reference-w0) > .01*w0:
        t = np.clip(ss/(PARAMETERS['width_transition_span_widths']*w0), 0., 1.)
        fade = t*t*(3.-2.*t)
        width_hypotheses.append(('smooth_to_measured_template_width', w0+(reference-w0)*fade))
    queries = np.vstack([xy for _ in width_hypotheses])
    ww = np.concatenate([v for _, v in width_hypotheses])
    measured = evaluate_template(fields, queries, d, ww, template, compare_neighbors=False)
    reports = []; accepted = []
    for variant, (name, widths) in enumerate(width_hypotheses):
        sl = slice(variant*len(ss), (variant+1)*len(ss))
        supported = measured['supported'][sl].copy()
        scores = measured['score'][sl].copy()
        missing = _runs(~supported)
        gaps = []
        for a, b in missing:
            start = float(ss[max(0, a-1)])
            last = float(ss[b-1])
            if start < first_s and last-start >= PARAMETERS['minimum_missing_gap_widths']*w0:
                gaps.append({'start_index': int(a), 'end_index_exclusive': int(b),
                             'start_s_px': start, 'last_unsupported_s_px': last,
                             'unsupported_sample_count': int(b-a)})
        options = []
        for a, b in _runs(supported):
            preceding = [gap for gap in gaps if gap['end_index_exclusive'] <= a]
            if not preceding:
                continue
            # Multiple overlapping windows do not multiply physical coverage:
            # the actual supported CENTER span itself must cover at least 2w.
            stable_span = float(ss[b-1]-ss[a])
            if b-a < PARAMETERS['minimum_reacquired_samples'] or stable_span < PARAMETERS['minimum_reacquired_span_widths']*w0:
                continue
            stable_ends = np.flatnonzero(ss[a:b]-ss[a] >= PARAMETERS['minimum_reacquired_span_widths']*w0)
            resume = a+int(stable_ends[0])
            options.append({'resume_index': resume, 'start_s_px': float(ss[a]),
                            'resume_s_px': float(ss[resume]), 'stable_span_px': float(ss[resume]-ss[a]),
                            'stable_mean_score': float(scores[a:resume+1].mean()), 'gap': preceding[-1]})
        report = {'width_hypothesis': name, 'start_width_px': float(widths[0]), 'end_width_px': float(widths[-1]),
                  'long_missing_gaps': gaps, 'supported_runs': [
                      {'start_s_px': float(ss[a]), 'end_s_px': float(ss[b-1]), 'center_span_px': float(ss[b-1]-ss[a])}
                      for a, b in _runs(supported)], 'reacquisition_options': options,
                  'sample_count': len(ss), 'supported_samples': int(supported.sum())}
        reports.append(report)
        if options:
            option = min(options, key=lambda v: v['resume_s_px'])
            accepted.append((option['resume_s_px'], -option['stable_mean_score'], variant, option))
    out['diagnostics'].update(first_surface_s_px=first_s, hard_probe_limit_px=float(hard_limit),
                              final_probe_limit_px=float(ss[-1]), missing_gap_requirement_px=6*w0,
                              stable_center_span_requirement_px=2*w0, width_hypotheses=reports)
    if not accepted:
        any_gap = any(r['long_missing_gaps'] for r in reports)
        out['reason'] = 'no_stable_reacquisition_within_budget' if any_gap else 'no_long_pre_entry_missing_gap'
        return out
    _, _, variant, option = min(accepted)
    last = option['resume_index']+1; sl = slice(variant*len(ss), variant*len(ss)+last)
    out.update(accepted=True, xy=xy[:last].tolist(), widths=width_hypotheses[variant][1][:last].tolist(),
               supported=measured['supported'][sl].astype(bool).tolist(), score=measured['score'][sl].tolist(),
               reason='bounded_axis_reacquired_after_explicit_unknown_gap')
    out['diagnostics'].update(selected_width_hypothesis=width_hypotheses[variant][0],
                              resume_s_px=option['resume_s_px'], stable_start_s_px=option['start_s_px'],
                              stable_span_px=option['stable_span_px'], bridged_unknown_gap=option['gap'],
                              known_material_identity_across_gap=False)
    return out
