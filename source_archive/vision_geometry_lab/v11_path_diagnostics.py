"""POST-HOC V11 holdout failure diagnosis; GT is diagnostic input ONLY.

No model/source/threshold is changed, no optimization or inference is repeated.
Saved real predictions remain untouched. The separate GT-distribution oracle
is an upper-bound mechanism check, emphatically NOT model performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .v11_path_contract import build_targets, image_to_strip, strip_to_image, WIDTH, HEIGHT, STEP_WIDTHS
from .v11_path_decode import decode_path, _column_nodes
from .v11_path_experiment import BASE, read, sha, verify, load_cache, write_json
from .evaluate_centerline_multiscale_v9 import _dataset, scorable_reference, score_decoded, aggregate


REPRESENTATIVES = ('dual_20260923_113745__left__f00550', 'dual_20260923_113305__right__f00500')


def _summary(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return dict(n=int(len(values)), mean=float(values.mean()) if len(values) else None,
                median=float(np.median(values)) if len(values) else None,
                p95=float(np.percentile(values, 95)) if len(values) else None)


def _runs(points, keep):
    boundaries = np.r_[0, np.flatnonzero(np.diff(np.asarray(keep, np.int8)))+1, len(keep)]
    return [np.asarray(points)[a:b] for a, b in zip(boundaries[:-1], boundaries[1:]) if keep[a] and b-a >= 2]


def raw_diagnostics(probability, targets, geometry, valid_mask, saved_decode):
    columns = np.arange(WIDTH)
    u = -8+(columns+.5)*STEP_WIDTHS
    column_valid = targets['column_valid'] & (u >= 0)
    target_row = np.arange(HEIGHT)@targets['target_distribution']
    predicted_row = probability.argmax(axis=0)
    native_step = float(geometry['native_px_per_row'])
    signed = (predicted_row-target_row)*native_step
    seed_normal_error = abs(signed)
    local_normal_error = seed_normal_error/np.sqrt(1+targets['slope_target']**2)
    peak = probability[predicted_row, columns]
    mask5 = abs(np.arange(HEIGHT)[:, None]-target_row[None])*native_step <= 5.
    mass5 = np.sum(np.where(mask5, probability, 0), axis=0)
    errors = column_valid & (seed_normal_error > 5.)
    first_failure = None
    for start in range(WIDTH-4):
        if errors[start:start+5].all():
            first_failure = start
            break
    diagnostics = saved_decode.get('diagnostics', {})
    anchor = diagnostics.get('anchor_column', 32)
    radius = diagnostics.get('window_radius_rows', int(np.ceil(2*max(1., 2/native_step))))
    nodes = _column_nodes(probability, valid_mask, anchor, radius)
    anchored = [node for node in nodes if abs(node['row']-127.5) <= .75/STEP_WIDTHS]
    decoded_strip = [image_to_strip(segment, geometry) for segment in saved_decode['segments']]
    decoded_max = max((float(segment[:, 0].max()) for segment in decoded_strip), default=None)
    result = dict(trusted_downstream_columns=int(column_valid.sum()),
        raw_argmax_seed_normal_error_px=_summary(seed_normal_error[column_valid]),
        raw_argmax_local_tangent_normal_error_px=_summary(local_normal_error[column_valid]),
        raw_argmax_within_5_seed_normal_px_fraction=float(np.mean(seed_normal_error[column_valid] <= 5)) if column_valid.any() else None,
        raw_argmax_within_5_local_normal_px_fraction=float(np.mean(local_normal_error[column_valid] <= 5)) if column_valid.any() else None,
        model_probability_mass_within_target_5_seed_normal_px=_summary(mass5[column_valid]),
        raw_peak_probability=_summary(peak[column_valid]),
        wrong_argmax_peak_probability=_summary(peak[errors]),
        wrong_argmax_columns=int(errors.sum()),
        first_5_consecutive_wrong_columns=None if first_failure is None else dict(
            column=int(first_failure), u_widths=float(u[first_failure]),
            target_xy=strip_to_image([first_failure, target_row[first_failure]], geometry).tolist(),
            raw_argmax_xy=strip_to_image([first_failure, predicted_row[first_failure]], geometry).tolist()),
        anchor=dict(column=int(anchor), oracle_target_supervised=bool(targets['column_valid'][anchor]),
            model_candidate_count=len(anchored), model_anchor_available=bool(anchored),
            model_raw_argmax_row=int(predicted_row[anchor]), model_raw_peak=float(peak[anchor]),
            expected_row=127.5, raw_argmax_seed_normal_error_px=float(abs(predicted_row[anchor]-127.5)*native_step)),
        saved_decoder=dict(last_column=decoded_max, reason=saved_decode['reason'],
            stopped_before_last_trusted_column=bool(decoded_max is None or
                (column_valid.any() and decoded_max < np.flatnonzero(column_valid)[-1]))),
        interpretation='Raw-column fractions are diagnostic and not the arc-length weighted V6 path score. '
                       'Seed-normal is transverse to strip; local-normal uses GT tangent, not nearest-path matching.',
        per_column=[dict(column=int(column), u_widths=float(u[column]), target_row=float(target_row[column]),
            raw_argmax_row=int(predicted_row[column]), seed_normal_error_px=float(seed_normal_error[column]),
            local_normal_error_px=float(local_normal_error[column]), target_5px_mass=float(mass5[column]),
            peak_probability=float(peak[column])) for column in np.flatnonzero(column_valid)])
    return result, dict(target_row=target_row, predicted_row=predicted_row,
                        trusted_columns=column_valid, decoded_strip=decoded_strip)


def _font(size):
    path = Path('C:/Windows/Fonts/arial.ttf')
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def render_representatives(rows, figures, output):
    pw, ph, caption = 768, 384, 82
    selected = [row for name in REPRESENTATIVES for row in rows if row['sample_id'] == name]
    canvas = Image.new('RGB', (pw*2, 64+len(selected)*(ph+caption)+54), '#131c2b')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), 'V11 POST-HOC | 2 selected failure cases, NOT all 8 holdouts', fill='white', font=_font(23))
    draw.text((12, 37), 'GT for diagnosis/scoring only. No prediction or threshold was changed.', fill='#ccd5e6', font=_font(17))
    def lines(points, color, ox, oy, width):
        for segment in points:
            if len(segment) >= 2:
                draw.line([(ox+x*1.5, oy+y*1.5) for x, y in segment], fill=color, width=width)
    for index, row in enumerate(selected):
        panel = figures[row['sample_id']]
        pixels = np.uint8(np.clip(panel['input'][:3].transpose(1, 2, 0), 0, 1)*255)
        trusted = panel['raw']['trusted_columns']
        gt = _runs(np.column_stack([np.arange(WIDTH), panel['raw']['target_row']]), trusted)
        argmax = _runs(np.column_stack([np.arange(WIDTH), panel['raw']['predicted_row']]), trusted)
        for column in range(2):
            x, top = column*pw, 64+index*(ph+caption)
            y = top+caption
            canvas.paste(Image.fromarray(pixels).resize((pw, ph), Image.Resampling.BILINEAR), (x, y))
            draw.text((x+10, top+5), row['sample_id'], fill='white', font=_font(16))
            title = ('Green=trusted GT; orange=raw per-column argmax' if column == 0
                     else 'Green=trusted GT; magenta=SAVED decoder top1')
            draw.text((x+10, top+28), title, fill='#cbd5e1', font=_font(17))
            cov = row['saved_model_metrics']['covered_fraction']
            ocov = row['oracle_metrics']['covered_fraction']
            draw.text((x+10, top+52), f'Model path coverage {cov:.1%}; GT oracle {ocov:.1%}; native tolerance 5 px',
                      fill='#cbd5e1', font=_font(16))
            lines(gt, '#44ff83', x, y, 3)
            lines(argmax if column == 0 else panel['raw']['decoded_strip'],
                  '#ffaa35' if column == 0 else '#ff40df', x, y, 2)
    draw.text((12, canvas.height-38), 'Matching another contour is a hypothesis, not proven semantic identity. Unknown reference gaps remain unconnected.',
              fill='white', font=_font(16))
    with output.open('xb') as stream:
        canvas.save(stream, format='PNG')


def run(output=BASE):
    output = Path(output)
    destination = output/'development/diagnostics_posthoc.json'
    montage = output/'development/diagnostics_posthoc_representatives.png'
    if destination.exists() or montage.exists():
        raise FileExistsError('Do not replace a prior diagnostic')
    verify(output)
    index, cached = load_cache(output, development=True)
    _, records, _ = _dataset(index['manifest'])
    if len(records) != 18 or sum(record['sample']['split']=='holdout' for record in records) != 8:
        raise ValueError('Only the existing 18-record development set is allowed')
    run_dir = output/'development'
    evaluation, exports = read(run_dir/'evaluation.json'), read(run_dir/'exports.json')
    if evaluation['exports_sha256'] != sha(run_dir/'exports.json'):
        raise ValueError('Saved evaluation/export mismatch')
    if exports['protocol_sha256'] != sha(output/'protocol.json'):
        raise ValueError('Wrong frozen protocol')
    entries = {entry['sample_id']:(entry, arrays) for entry, arrays in cached}
    predictions = {record['sample_id']:record for record in exports['records']}
    saved = {row['sample_id']:row for row in evaluation['rows']}
    snapshots = {str(path):sha(path) for path in (run_dir/'evaluation.json', run_dir/'exports.json',
                 output/'development_cache/index.json', run_dir/'checkpoint.pt')}
    rows, oracle_rows, figures = [], [], {}
    for record in records:
        sample, label, quality = record['sample'], record['label'], record['quality']
        if sample['split'] != 'holdout':
            continue
        name = sample['sample_id']
        entry, arrays = entries[name]
        pred, scored = predictions[name], saved[name]
        if not entry['available'] or not pred['available']:
            raise ValueError('An unavailable record must be explicitly handled, not dropped')
        if pred['sha256'] != sha(pred['path']) or pred['input_sha256'] != entry['inference_input_sha256']:
            raise ValueError('Prediction or its input changed')
        snapshots[pred['path']] = pred['sha256']
        snapshots[entry['cache']] = entry['cache_sha256']
        with np.load(pred['path'], allow_pickle=False) as archive:
            probability = archive['probability'].copy()
        reference, _ = scorable_reference(label, quality)
        targets = build_targets(entry['geometry'], reference,
            label['ignore_polygons']+quality['precision_exclude_polygons'], arrays['valid_mask'])
        oracle = decode_path(targets['target_distribution'], targets['slope_target'],
                             entry['geometry'], arrays['valid_mask'])
        oracle_score = score_decoded(label, quality, oracle, None, sample['image_shape_hw'])
        oracle_rows.append(dict(sample_id=name, split='holdout', group_id=sample['group_id'],
            cohort=record['cohort'], prediction_status='loaded', independent_width=entry['width'],
            decoded=oracle, **oracle_score))
        raw, plot = raw_diagnostics(probability, targets, entry['geometry'], arrays['valid_mask'], scored['decoded'])
        row = dict(sample_id=name, split='holdout', group_id=sample['group_id'],
            cohort=record['cohort'], oracle_metrics=oracle_score['metrics']['downstream'],
            oracle_reason=oracle['reason'], saved_model_metrics=scored['metrics']['downstream'],
            saved_window_support=scored['coverage_support']['downstream'],
            target_construction=targets['diagnostics'], raw_distribution=raw,
            oracle_ground_truth_is_input=True, oracle_is_model_performance=False)
        rows.append(row)
        if name in REPRESENTATIVES:
            figures[name] = dict(input=arrays['input'], raw=plot)
        print(json.dumps(dict(sample=name, oracle_cov=row['oracle_metrics']['covered_fraction'],
            model_cov=row['saved_model_metrics']['covered_fraction'],
            raw_5px=raw['raw_argmax_within_5_seed_normal_px_fraction'],
            raw_median_error=raw['raw_argmax_seed_normal_error_px']['median'],
            peak_wrong=raw['wrong_argmax_peak_probability']['median'],
            anchor=raw['anchor']['model_anchor_available']), ensure_ascii=False), flush=True)
    verify(output)
    if any(sha(path) != digest for path, digest in snapshots.items()):
        raise ValueError('An input artifact changed during diagnostics')
    render_representatives(rows, figures, montage)
    write_json(destination, dict(kind='posthoc_failure_diagnosis', source=str(Path(__file__).resolve()),
        source_sha256=sha(__file__), holdout_count=8, rows=rows, oracle_summaries=aggregate(oracle_rows),
        saved_model_summaries=[summary for summary in evaluation['summaries'] if summary['split']=='holdout'],
        unchanged_artifact_sha256=snapshots, montage=str(montage), montage_sha256=sha(montage),
        montage_samples=list(REPRESENTATIVES), montage_is_all_holdouts=False,
        ground_truth_used_only_for_posthoc_diagnostics=True, predictions_changed=False,
        thresholds_changed=False, model_retrained=False, blind_20260925_read=False,
        identity_confirmed=False, force_command=None,
        interpretation='Oracle tests strip/target/decode representation, not model recognition. Raw argmax and target mass '
                       'diagnose model mistakes without altering real decoder. Other-contour identity is not proven by these scores.'))
    print(json.dumps(dict(diagnostics=str(destination), montage=str(montage)), ensure_ascii=False))
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE)
    run(parser.parse_args().output)
