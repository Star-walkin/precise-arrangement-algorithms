"""Post-hoc input/denominator audit of frozen V23 automatic-seed predictions.

No model inference, image detection, training, threshold changes or hardware.
The trusted references are used only to inspect already saved input geometry.
Their denominator is never cropped to the available model strip.
"""
from pathlib import Path
import json
import math

import cv2
import numpy as np

from .v23_auto_seed_chain import BASE, OLD, LAB, read, sha, verify, write
from .v11_path_contract import image_to_strip
from .centerline_path_metrics_v6 import _sampling


OUT = BASE/'input_diagnostics'


def strip_support(reference_segments, geometry, valid_mask):
    """Length-weighted diagnostic only; preserve all trusted reference length."""
    mask = np.asarray(valid_mask, np.float32)
    if mask.shape != tuple(geometry['shape_hw']) or not np.isfinite(mask).all():
        raise ValueError('Invalid saved strip mask')
    if np.any((mask != 0) & (mask != 1)):
        raise ValueError('Expected binary saved strip mask')
    pieces = _sampling(reference_segments)
    total = float(sum(p['length'] for p in pieces))
    supported = 0.
    for piece in pieces:
        q = image_to_strip(piece['points'], geometry)
        inside = ((q[:, 0] >= 0) & (q[:, 0] < mask.shape[1]-1) &
                  (q[:, 1] >= 0) & (q[:, 1] < mask.shape[0]-1))
        values = cv2.remap(mask, q[:, 0].astype(np.float32)[None],
                          q[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)[0]
        supported += float(piece['weights'][inside & (values >= 1.-1e-6)].sum())
    return dict(reference_length_px=total, supported_reference_length_px=supported,
                fraction=supported/total if total else None,
                uses_future_or_labels=True, posthoc_diagnostic_only=True,
                restricts_score_denominator=False)


def geometry_difference(manual, automatic):
    delta = np.asarray(automatic['origin_xy'])-np.asarray(manual['origin_xy'])
    d, n = np.asarray(manual['d_xy']), np.asarray(manual['n_xy'])
    cosine = float(np.asarray(automatic['d_xy'])@d)
    return dict(manual_origin_xy=manual['origin_xy'], automatic_origin_xy=automatic['origin_xy'],
        origin_shift_px=float(np.linalg.norm(delta)),
        shift_along_manual_direction_px=float(delta@d),
        shift_normal_manual_direction_px=float(delta@n),
        directed_angle_difference_deg=math.degrees(math.acos(float(np.clip(cosine, -1, 1)))),
        manual_width_px=manual['width_px'], automatic_width_px=automatic['width_px'],
        width_ratio=automatic['width_px']/manual['width_px'],
        automatic_native_px_per_row=automatic['native_px_per_row'],
        manual_seed_points=manual['free_seed_xy'], automatic_seed_points=automatic['free_seed_xy'],
        original_image_pixels=True, physical_contact_position_not_known=True)


def run(output=OUT):
    output = Path(output)
    if output.exists(): raise FileExistsError('Do not replace completed audit')
    verify(BASE)
    source_paths = [Path(__file__), BASE/'protocol.json', BASE/'report.json',
        BASE/'inputs.json', BASE/'predictions.json', OLD/'development/evaluation.json',
        OLD/'development/exports.json', LAB/'v11_path_contract.py',
        LAB/'centerline_path_metrics_v6.py']
    source_hashes = {str(p.resolve()): sha(p) for p in source_paths}
    report = read(BASE/'report.json')
    old = {r['sample_id']: r for r in read(OLD/'development/evaluation.json')['rows']}
    manual = {r['sample_id']: r for r in read(OLD/'development/exports.json')['records']}
    inputs = {r['sample_id']: r for r in read(BASE/'inputs.json')['entries']}
    if len(report['rows']) != 18 or len(old) != 18 or len(inputs) != 20:
        raise ValueError('Expected unchanged 18 scored / 20 input records')
    if report['predictions_sha256'] != sha(BASE/'predictions.json'):
        raise ValueError('Predictions changed after scoring')
    rows = []
    for row in report['rows']:
        sid = row['sample_id']; previous = old[sid]; entry = inputs[sid]
        m = manual[sid]['geometry']
        denominator = {}
        for scope, key in (('full', 'full_path'), ('downstream', 'downstream')):
            before = previous['metrics'][key]['reference_length_px']
            after = row['metrics'][key]['reference_length_px']
            denominator[scope] = dict(manual_reference_length_px=before,
                automatic_reference_length_px=after, exactly_equal=before == after)
            if before != after: raise ValueError('Changed reference denominator: '+sid)
        before = previous['metrics']['downstream']; after = row['metrics']['downstream']
        item = dict(sample_id=sid, split=row['split'], available=entry['available'],
            input_reason=entry['reason'], denominator_comparison=denominator,
            manual_downstream=before, automatic_downstream=after,
            geometry_difference=None, actual_strip_reference_support=None,
            saved_width_px=entry['width_px'], manual_width_px=m['width_px'],
            decoded_reason=row['decoded'].get('reason'),
            decoded_ambiguous=row['decoded'].get('diagnostics', {}).get('ambiguous'))
        if entry['available']:
            if sha(entry['input_path']) != entry['input_sha256']:
                raise ValueError('Input file changed')
            source_hashes[str(Path(entry['input_path']).resolve())] = entry['input_sha256']
            with np.load(entry['input_path'], allow_pickle=False) as arrays:
                mask = arrays['valid_mask'].copy()
            item['geometry_difference'] = geometry_difference(m, entry['geometry'])
            item['actual_strip_reference_support'] = {}
            for scope in ('full', 'downstream'):
                segments = row['metrics']['score_support'][scope+'_reference_segments']
                support = strip_support(segments, entry['geometry'], mask)
                expected = denominator[scope]['automatic_reference_length_px']
                if not math.isclose(support['reference_length_px'], expected, abs_tol=1e-6):
                    raise ValueError('Diagnostic reference denominator mismatch')
                item['actual_strip_reference_support'][scope] = support
            segments = row['decoded']['segments']
            item['decoded_segment_extents_along_manual_seed_px'] = [
                ((np.asarray(s)[[0, -1]]-m['origin_xy'])@np.asarray(m['d_xy'])).tolist()
                for s in segments if len(s)]
        rows.append(item)
    held = [r for r in rows if r['split'] == 'holdout']
    decomposition = []
    for available in (True, False):
        subset = [r for r in held if r['available'] == available]
        reference = sum(r['automatic_downstream']['reference_length_px'] for r in subset)
        old_hit = sum(r['manual_downstream']['covered_reference_length_px'] for r in subset)
        new_hit = sum(r['automatic_downstream']['covered_reference_length_px'] for r in subset)
        decomposition.append(dict(automatic_input_available=available, samples=len(subset),
            reference_length_px=reference, manual_covered_length_px=old_hit,
            automatic_covered_length_px=new_hit,
            manual_conditional_coverage=old_hit/reference if reference else None,
            automatic_conditional_coverage=new_hit/reference if reference else None))
    result = dict(status='complete', kind='posthoc_input_distribution_and_denominator_diagnostic',
        sources_sha256=source_hashes, scored_samples=len(rows),
        all_18_full_and_downstream_denominators_exactly_equal=True,
        geometry_rows=rows, holdout_input_availability_decomposition=decomposition,
        holdout_unavailable_reasons={reason:sum(r['input_reason'] == reason for r in held)
            for reason in sorted({r['input_reason'] for r in held if not r['available']})},
        protocol_was_verified=True, predictions_rerun=False, thresholds_changed=False,
        human_labels_used_only_after_saved_predictions=True, hardware_access=False,
        identity_confirmed=False,
        limitations=[
            'Combined seed, width, crop origin, prompt heatmap and coordinate/context change; not endpoint-only causal attribution.',
            'Downstream plane uses fixed manual seed for both arms; not physical contact.',
            'Actual strip support is post-hoc and does not alter any scoring denominator.',
            'The archived aggregate window_reference_support_fraction=0 is an unmeasured placeholder from coverage_mask=None; do not quote it.',
            'This diagnostic leaves all frozen code and reports unchanged.'
        ])
    for path, expected in source_hashes.items():
        if sha(path) != expected: raise ValueError('Source changed during diagnostic: '+path)
    write(output/'diagnostics.json', result)
    print(json.dumps(dict(status='complete', path=str(output/'diagnostics.json'),
        sha256=sha(output/'diagnostics.json'), holdout_input_availability_decomposition=decomposition), indent=2))


if __name__ == '__main__': run()
