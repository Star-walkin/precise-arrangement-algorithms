"""Frozen V24 comparison, unchanged V6 scoring and GT-only V23 bend audit.

No torch, hardware, blind data, threshold search, or target-assisted decoding.
Missing input is an empty prediction, never a removed reference denominator.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .evaluate_centerline_multiscale_v9 import _dataset, aggregate, score_decoded, scorable_reference
from .centerline_path_metrics_v6 import _sampling, protocol_sha256
from .centerline_review_v5 import read_clean_image
from .v11_path_contract import HEIGHT, WIDTH, _checked_geometry, image_to_strip
from .v11_path_decode import decode_path
from .v11_path_experiment import BASE as V11_BASE, read, sha, verify, write_json
from .v18_front_diagnostics import font
from .v23_path_prefix_audit import departure_diagnostic, seed_plane_origin


LAB = Path(__file__).resolve().parent
MANIFEST = V11_BASE/'development_manifest.json'
ARMS = ('baseline', 'control', 'anchor_mix')
VARIANTS = ('manual', 'auto', 'probe0', 'probe1')
DEPENDENCY_NAMES = (
    'v24_anchor_evaluate.py', 'evaluate_centerline_multiscale_v9.py',
    'centerline_path_metrics_v6.py', 'centerline_review_v5.py',
    'centerline_quality_v5.py', 'centerline_quality_cutoff_v8.py',
    'prepare_centerline_v5.py', 'centerline_benchmark_v5.py', 'rope_detector.py',
    'centerline_multiscale_contract_v9.py', 'centerline_model_v5.py',
    'v11_path_contract.py', 'v11_path_decode.py', 'v11_path_experiment.py',
    'v18_front_diagnostics.py', 'v23_path_prefix_audit.py',
)


def _safe_path(path):
    value = Path(path)
    if '20260925' in str(value):
        raise ValueError('9/25 blind sources must not be read')
    return value


def _check_json(value):
    # Flag names / protocol exclusion prose may legitimately name the blind
    # date. Reject actual referenced paths and sample IDs, never open them.
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and ('path' in key or key in ('sample_id', 'manifest')):
                _safe_path(item)
            if isinstance(key, str) and ('/' in key or '\\' in key):
                _safe_path(key)
            _check_json(item)
    elif isinstance(value, list):
        for item in value:
            _check_json(item)


def _read(path):
    result = read(_safe_path(path))
    _check_json(result)
    return result


def _bound_file(path, expected, hashes):
    path = _safe_path(path)
    actual = sha(path)
    if actual != expected:
        raise ValueError('Artifact hash changed: '+str(path))
    hashes[str(path.resolve())] = actual


def validate_inventory(index, records):
    """Require all 18 x 4 rows, including auto failures; training extras are OK."""
    samples = {r['sample']['sample_id']: r['sample'] for r in records}
    if len(samples) != 18 or sum(s['split'] == 'train' for s in samples.values()) != 10:
        raise ValueError('Fixed 10 train + 8 development holdout samples required')
    if sum(s['split'] == 'holdout' for s in samples.values()) != 8:
        raise ValueError('Fixed development holdout split changed')
    keys = index['evaluation_keys']
    if len(keys) != 72 or len(set(keys)) != 72:
        raise ValueError('All 72 evaluation keys must be unique and retained')
    seen = set()
    for key in keys:
        entry = index['entries'][key]
        name, variant = entry['sample_id'], entry['variant']
        if name not in samples or variant not in VARIANTS or entry['split'] != samples[name]['split']:
            raise ValueError('Unexpected evaluation sample/variant/split')
        if type(entry['available']) is not bool or (name, variant) in seen:
            raise ValueError('Duplicate evaluation identity or invalid availability')
        seen.add((name, variant))
    if seen != {(name, variant) for name in samples for variant in VARIANTS}:
        raise ValueError('Evaluation inventory dropped or replaced a sample')


def load_inference_input(entry, hashes):
    """Strict two-array boundary: supervision cannot accompany model inputs."""
    if not entry['available']:
        if entry.get('input_path') or entry.get('input_sha256'):
            raise ValueError('Unavailable input must not hide an input artifact')
        return None
    _bound_file(entry['input_path'], entry['input_sha256'], hashes)
    with np.load(entry['input_path'], allow_pickle=False) as archive:
        if set(archive.files) != {'input', 'valid_mask'}:
            raise ValueError('Inference archive must contain input and valid_mask only; no GT')
        arrays = {key: archive[key].copy() for key in archive.files}
    x, valid = arrays['input'], arrays['valid_mask']
    if x.shape != (8, HEIGHT, WIDTH) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError('Invalid inference input tensor')
    if valid.shape != (HEIGHT, WIDTH) or not np.isin(valid, [0, 1]).all():
        raise ValueError('Invalid inference valid mask')
    if not np.array_equal(x[7], valid):
        raise ValueError('Input real-pixel channel differs from valid mask')
    _checked_geometry(entry['geometry'])
    if entry['geometry'].get('target_or_quality_used') is not False:
        raise ValueError('Inference geometry must explicitly exclude targets and quality')
    return arrays


def decode_export(entry, prediction, arrays, hashes):
    """Replay only the fixed input-only decoder. This API accepts no GT/quality."""
    for key in ('sample_id', 'split', 'variant', 'available'):
        if prediction[key] != entry[key]:
            raise ValueError('Prediction identity/availability mismatch: '+key)
    if prediction.get('input_sha256') != entry.get('input_sha256'):
        raise ValueError('Prediction uses a different inference input')
    decoded = prediction['decoded']
    if decoded.get('identity_confirmed', False):
        raise ValueError('Candidate prediction cannot confirm rope identity')
    if not prediction['available']:
        if decoded.get('segments') or prediction.get('probability_path') or prediction.get('probability_sha256'):
            raise ValueError('Unavailable prediction must be an explicit empty path')
        return decoded
    if arrays is None:
        raise ValueError('Available prediction without input')
    _bound_file(prediction['probability_path'], prediction['probability_sha256'], hashes)
    with np.load(prediction['probability_path'], allow_pickle=False) as archive:
        if set(archive.files) != {'probability', 'slope'}:
            raise ValueError('Prediction archive must contain probability and slope only; no GT')
        probability, slope = archive['probability'].copy(), archive['slope'].copy()
    replayed = decode_path(probability, slope, entry['geometry'], arrays['valid_mask'])
    # JSON equality is exact for serialized double coordinates. The elapsed time
    # is nondeterministic and deliberately excluded; path/reason must reproduce.
    if replayed['segments'] != decoded.get('segments') or replayed['reason'] != decoded.get('reason'):
        raise ValueError('Saved path differs from unchanged input-only decoder')
    return decoded


def input_footprint(segments, geometry, valid_mask):
    """Native GT arc support sampled directly in actual inference valid pixels.

    This diagnostic never clips the reference or supplies a model prediction.
    An absent input footprint is unknown, not measured zero support.
    """
    pieces = _sampling(segments)
    total = float(sum(p['length'] for p in pieces))
    known = geometry is not None and valid_mask is not None
    result = dict(status='measured' if known else 'unknown_unavailable_input',
        reference_length_px=total, supported_reference_length_px=None,
        fraction=None, does_not_change_score_denominator=True,
        measurement='V6 arc cells; conservative bilinear support in actual input valid_mask')
    if not known:
        return result
    supported = 0.
    for piece in pieces:
        uv = image_to_strip(piece['points'], geometry)
        inside = ((uv[:, 0] >= 0) & (uv[:, 0] < valid_mask.shape[1]-1) &
                  (uv[:, 1] >= 0) & (uv[:, 1] < valid_mask.shape[0]-1))
        values = cv2.remap(valid_mask.astype(np.float32), uv[:, 0].astype(np.float32)[None],
            uv[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[0]
        supported += float(piece['weights'][inside & (values >= 1-1e-6)].sum())
    result.update(supported_reference_length_px=supported, fraction=supported/total if total else None)
    return result


def score_case(record, entry, decoded, original_width, arrays):
    """Quality enters scoring only. Seed plane and width stay original/manual."""
    sample, label, quality = record['sample'], record['label'], record['quality']
    score = score_decoded(label, quality, decoded, None, sample['image_shape_hw'])
    support = score['metrics']['score_support']
    footprint = {scope: input_footprint(support[scope+'_reference_segments'],
        entry.get('geometry') if arrays is not None else None,
        arrays['valid_mask'] if arrays is not None else None) for scope in ('full', 'downstream')}
    # Do not expose the legacy None-mask placeholder as a measured zero.
    score['coverage_support'] = footprint
    reference, _ = scorable_reference(label, quality)
    seed = label['polylines']['free_seed']
    origin, origin_reason = seed_plane_origin(reference, seed)
    bend = departure_diagnostic(reference, seed, original_width['width_px'],
        support['downstream_reference_segments'], support['downstream_prediction_segments'], origin)
    bend.update(origin_arc_px=origin, origin_reason=origin_reason,
        width_px=original_width['width_px'], width_source='fixed original V11 manual input optical width',
        is_contact=False, localization_is_GT_posthoc_only=True, used_to_crop_inference=False)
    return dict(sample_id=sample['sample_id'], split=sample['split'], group_id=sample['group_id'],
        cohort=record['cohort'], variant=entry['variant'],
        prediction_status='loaded' if entry['available'] else 'input_unavailable',
        prediction_unavailable_reason=entry.get('reason') if not entry['available'] else None,
        independent_width=original_width, input_geometry_width_px=(entry.get('geometry') or {}).get('width_px'),
        annotation_sha256=record.get('annotation_sha256'), quality_sha256=quality.get('quality_sha256'),
        decoded=decoded, GT_departure=bend, actual_input_footprint=footprint, **score)


def aggregate_fixed(rows):
    """Unchanged aggregate for metrics, with explicit unknown footprint policy."""
    scratch = copy.deepcopy(rows)
    for row in scratch:
        for cell in row['coverage_support'].values():
            # Legacy aggregate has no unknown-support representation. This
            # throwaway argument is overwritten below and never serialized.
            if cell['supported_reference_length_px'] is None:
                cell['supported_reference_length_px'] = 0.
    summaries = aggregate(scratch)
    for summary in summaries:
        selected = [r for r in rows if r['split'] == summary['split'] and
            (summary['cohort'] == 'all' or r['cohort'] == summary['cohort'])]
        scope = 'full' if summary['scope'] == 'full_path' else 'downstream'
        cells = [r['actual_input_footprint'][scope] for r in selected]
        unknown = sum(c['status'] != 'measured' for c in cells)
        length = sum(c['reference_length_px'] for c in cells)
        known_length = sum(c['reference_length_px'] for c in cells if c['status'] == 'measured')
        supported = sum(c['supported_reference_length_px'] for c in cells if c['status'] == 'measured')
        summary['window_reference_support_fraction'] = supported/length if length and not unknown else None
        bends = [r['GT_departure'] for r in selected if 'prediction_hits_5px' in r['GT_departure']]
        bend_length = sum(b['neighborhood_reference_length_px'] for b in bends)
        bend_covered = sum(b['neighborhood_reference_length_px']*(b['neighborhood_coverage_5px'] or 0) for b in bends)
        summary.update(actual_input_footprint=dict(status='unknown_partial' if unknown else 'measured',
            measured_samples=len(cells)-unknown, unknown_samples=unknown,
            measured_reference_length_px=known_length, measured_supported_reference_length_px=supported,
            measured_subset_support_fraction=supported/known_length if known_length else None,
            all_sample_support_fraction=summary['window_reference_support_fraction']),
            GT_departure_supported_samples=len(bends),
            GT_departure_prediction_hits=sum(b['prediction_hits_5px'] for b in bends),
            GT_departure_neighborhood_reference_length_px=bend_length,
            GT_departure_neighborhood_coverage_5px=bend_covered/bend_length if bend_length else None,
            physical_contact_identified=False)
    return summaries


def _pct(value):
    return '未知' if value is None else f'{value*100:.2f}%'


def render_comparison(summaries):
    image = Image.new('RGB', (1940, 1780), '#141d29')
    draw = ImageDraw.Draw(image)
    draw.text((30, 20), 'V24 锚点扰动训练 | 固定原始下游 GT 分母', font=font(34), fill='white')
    draw.text((30, 72), 'baseline=冻结 V11；control / anchor_mix=同预算微调；所有结果是路径候选，不证明绳身份。', font=font(23), fill='#cbd8e7')
    draw.text((30, 108), '开发留出已用于历史迭代，并非盲测。5px 阈值和 quality.C7 可信截断保持不变。', font=font(23), fill='#ffd58a')
    headers = [(30, '输入 / 模型'), (410, '覆盖率'), (615, '候选精度'), (835, 'GT长度(px)'),
               (1080, '弯曲命中'), (1280, '弯曲邻域覆盖'), (1530, '输入支持 / 未知张数')]
    y = 160
    for split, label in (('train', '训练10张（拟合诊断）'), ('holdout', '开发留出8张（非盲测）')):
        draw.text((30, y), label, font=font(28), fill='#75d3ff')
        y += 50
        for x, text in headers:
            draw.text((x, y), text, font=font(22), fill='white')
        y += 39
        for variant in VARIANTS:
            for arm in ARMS:
                row = next(s for s in summaries if (s['arm'], s['variant'], s['split'], s['cohort'], s['scope']) ==
                           (arm, variant, split, 'all', 'downstream'))
                if arm == 'anchor_mix':
                    draw.rectangle((20, y-2, 1920, y+39), fill='#213447')
                cells = [f'{variant} / {arm}', _pct(row['covered_fraction']), _pct(row['prediction_precision_at_5px']),
                    f"{row['reference_length_px']:.1f}",
                    f"{row['GT_departure_prediction_hits']}/{row['GT_departure_supported_samples']}",
                    _pct(row['GT_departure_neighborhood_coverage_5px']),
                    f"{_pct(row['window_reference_support_fraction'])} / {row['actual_input_footprint']['unknown_samples']}"]
                for (x, _), cell in zip(headers, cells):
                    draw.text((x, y+4), cell, font=font(21), fill='white')
                y += 43
            y += 8
        y += 28
    draw.text((30, 1600), '弯曲处 = 可信 GT 首次偏离人工自由段直线 0.5 倍旧光学宽度；邻域 = 前后 2 倍宽度弧长。', font=font(22), fill='#cbd8e7')
    draw.text((30, 1640), '它是 GT 事后几何诊断，不是真实接触点；不用来裁输入、选阈值或改变预测。缺失输入足迹明确为未知。', font=font(22), fill='#cbd8e7')
    draw.text((30, 1680), '覆盖/候选精度是旧 V6 长度加权分数。输入足迹仅报告实际 valid_mask 的几何支持，不是活动绳身份。', font=font(22), fill='#cbd8e7')
    return image


def render_failures(rows, records):
    """Posthoc worst two auto holdout cases, original source image, no crop."""
    candidates = [r for r in rows if r['arm'] == 'anchor_mix' and r['variant'] == 'auto' and r['split'] == 'holdout']
    candidates.sort(key=lambda r: (r['metrics']['downstream']['covered_fraction'] or 0, r['sample_id']))
    selected = candidates[:2]
    sources = {r['sample']['sample_id']: r for r in records}
    canvas = Image.new('RGB', (1920, 1450), '#141d29')
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 15), 'V24 自动输入失败例 | 原始图像全幅等比显示', font=font(30), fill='white')
    draw.text((24, 61), '后验选择 anchor_mix 开发留出覆盖率最低2张（含输入失败）；绿=可信GT；品红=模型候选；不标真实接触。', font=font(21), fill='#d4e2ed')
    for i, selected_row in enumerate(selected):
        name = selected_row['sample_id']
        source = read_clean_image(sources[name]['sample']).convert('RGB')
        top = 110+i*650
        for j, arm in enumerate(ARMS):
            row = next(r for r in rows if r['arm'] == arm and r['variant'] == 'auto' and r['sample_id'] == name)
            preview = ImageOps.contain(source, (620, 470))
            scale = preview.width/source.width
            paint = ImageDraw.Draw(preview)
            support = row['metrics']['score_support']
            for key, color in (('full_reference_segments', '#19ff72'), ('full_prediction_segments', '#ff36de')):
                for segment in support[key]:
                    if len(segment) >= 2:
                        paint.line([(float(p[0])*scale, float(p[1])*scale) for p in segment], fill=color, width=3)
            left = j*640+10
            draw.text((left, top), arm+' / auto', font=font(25), fill='white')
            canvas.paste(preview, (left, top+43))
            draw.text((left, top+520), name, font=font(16), fill='white')
            metrics = row['metrics']['downstream']
            draw.text((left, top+550), f"覆盖 {_pct(metrics['covered_fraction'])} | 候选精度 {_pct(metrics['prediction_precision_at_5px'])}", font=font(20), fill='#ffd58a')
            draw.text((left, top+584), '输入不可用，空预测（分母保留）' if row['prediction_status'] != 'loaded' else '绿线只画原始评分支持；排除区域不连线', font=font(18), fill='#d4e2ed')
    return canvas, [r['sample_id'] for r in selected]


def evaluate(base):
    base = _safe_path(base).resolve()
    output = base/'evaluation'
    if output.exists():
        raise FileExistsError('Evaluation outputs are immutable; choose a new experiment')
    cv2.setNumThreads(2)
    verify(V11_BASE)
    hashes = {}
    protocol_path, index_path = base/'protocol.json', base/'cache/input_index.json'
    protocol, index = _read(protocol_path), _read(index_path)
    protocol_hash = sha(protocol_path)
    if index['protocol_sha256'] != protocol_hash:
        raise ValueError('Input index belongs to another protocol')
    for category in ('source_snapshot', 'old_source_snapshot', 'dataset_snapshot'):
        for path, expected in protocol.get(category, {}).items():
            if expected is None:
                if _safe_path(path).exists():
                    raise ValueError('Previously absent protocol source appeared')
            else:
                _bound_file(path, expected, hashes)
    _, records, _ = _dataset(MANIFEST)
    validate_inventory(index, records)
    old_index = _read(V11_BASE/'development_cache/index.json')
    old_eval = _read(V11_BASE/'development/evaluation.json')
    old_exports = _read(V11_BASE/'development/exports.json')
    for path, expected in (
        (V11_BASE/'development/exports.json', old_eval['exports_sha256']),
        (V11_BASE/'development/checkpoint.pt', old_eval['checkpoint_sha256']),
        (V11_BASE/'development/checkpoint.pt', old_exports['checkpoint_sha256']),
        (V11_BASE/'development/report.json', old_exports['training_report_sha256']),
        (V11_BASE/'development_cache/index.json', old_exports['cache_index_sha256']),
    ):
        _bound_file(path, expected, hashes)
    if old_eval['protocol_sha256'] != sha(V11_BASE/'protocol.json') or old_exports['protocol_sha256'] != sha(V11_BASE/'protocol.json'):
        raise ValueError('Original V11 evaluation/model provenance changed')
    if old_index['manifest_sha256'] != sha(MANIFEST) or old_index['protocol_sha256'] != sha(V11_BASE/'protocol.json'):
        raise ValueError('Original manual width cache provenance changed')
    originals = {e['sample_id']: e for e in old_index['entries']}
    original_scores = {r['sample_id']: r for r in old_eval['rows']}
    names = {r['sample']['sample_id'] for r in records}
    if names != set(originals) or names != set(original_scores):
        raise ValueError('Original fixed sample inventory changed')
    for path in (protocol_path, index_path, MANIFEST, V11_BASE/'development_cache/index.json',
                 V11_BASE/'development/evaluation.json', V11_BASE/'protocol.json'):
        hashes[str(path.resolve())] = sha(path)
    hashes.update({str(LAB/name): sha(LAB/name) for name in DEPENDENCY_NAMES})
    by_name = {r['sample']['sample_id']: r for r in records}
    for record in records:
        sample = record['sample']; old = originals[sample['sample_id']]
        if old['annotation_sha256'] != record['annotation_sha256'] or old['quality_sha256'] != record['quality']['quality_sha256']:
            raise ValueError('Original annotation/quality cutoff changed')
        if old['width'] != original_scores[sample['sample_id']]['independent_width']:
            raise ValueError('Original fixed manual width differs from old evaluation')
        for key in ('image_path', 'annotation_path'):
            hashes[str(Path(sample[key]).resolve())] = sha(sample[key])
        quality_path = Path(sample['annotation_path']).with_suffix('.quality.json')
        if quality_path.exists():
            hashes[str(quality_path.resolve())] = sha(quality_path)
    # No training-target archive is opened by this evaluator.
    arrays = {key: load_inference_input(index['entries'][key], hashes) for key in index['evaluation_keys']}
    rows, summaries, model_hashes = [], [], {}
    for arm in ARMS:
        export_path = base/'exports'/arm/'index.json'
        exports = _read(export_path)
        hashes[str(export_path)] = sha(export_path)
        if exports['protocol_sha256'] != protocol_hash or exports['input_index_sha256'] != sha(index_path):
            raise ValueError('Export protocol/input-index binding differs')
        if not isinstance(exports['model_sha256'], str) or len(exports['model_sha256']) != 64:
            raise ValueError('Export model digest missing')
        model_hashes[arm] = exports['model_sha256']
        predictions = {r['key']: r for r in exports['rows']}
        if len(exports['rows']) != 72 or set(predictions) != set(index['evaluation_keys']):
            raise ValueError('Export must retain all fixed evaluation rows')
        for key in index['evaluation_keys']:
            entry = index['entries'][key]
            prediction = predictions[key]
            decoded = decode_export(entry, prediction, arrays[key], hashes)
            record = by_name[entry['sample_id']]
            row = score_case(record, entry, decoded, originals[entry['sample_id']]['width'], arrays[key])
            row.update(key=key, arm=arm, forward_ms=prediction.get('forward_ms'),
                input_sha256=entry.get('input_sha256'), probability_sha256=prediction.get('probability_sha256'))
            old = original_scores[entry['sample_id']]
            for scope in ('full_path', 'downstream'):
                if not np.isclose(row['metrics'][scope]['reference_length_px'], old['metrics'][scope]['reference_length_px'], atol=1e-6, rtol=0):
                    raise ValueError('Original reference denominator changed')
                if arm == 'baseline' and entry['variant'] == 'manual':
                    for metric in ('covered_reference_length_px', 'predicted_length_px', 'extra_predicted_length_px'):
                        a, b = row['metrics'][scope][metric], old['metrics'][scope][metric]
                        if (a is None) != (b is None) or (a is not None and not np.isclose(a, b, atol=1e-5, rtol=0)):
                            raise ValueError('Frozen baseline/manual does not reproduce original V11: '+metric)
            rows.append(row)
        for variant in VARIANTS:
            selected = [r for r in rows if r['arm'] == arm and r['variant'] == variant]
            summaries.extend(dict(arm=arm, variant=variant, **s) for s in aggregate_fixed(selected))
        print(json.dumps(dict(evaluated=arm, retained_rows=72), ensure_ascii=False), flush=True)
    verify(V11_BASE)
    if any(sha(path) != digest for path, digest in hashes.items()):
        raise ValueError('Evaluation input changed during run')
    output.mkdir(parents=True, exist_ok=False)
    comparison_path, failures_path = output/'comparison.png', output/'failure_overlays.png'
    render_comparison(summaries).save(comparison_path)
    failures, selected_failures = render_failures(rows, records)
    failures.save(failures_path)
    report = dict(schema='v24_anchor_fixed_denominator_evaluation_1', rows=rows, summaries=summaries,
        protocol_sha256=protocol_hash, path_protocol_sha256=protocol_sha256(), model_sha256=model_hashes,
        input_sha256=hashes, script_sha256=sha(__file__), no_rows_dropped=True,
        denominator_changed=False, thresholds_changed=False, baseline_manual_reproduced=True,
        quality_used_for_input=False, quality_used_for_decode=False,
        quality_policy='Unchanged score_decoded / quality.C7 trusted cutoff; excluded gaps never joined.',
        fixed_width_source=str(V11_BASE/'development_cache/index.json'),
        GT_departure_definition='V23 exact functions: first trusted GT departure >=0.5 original manual optical width; +/-2 widths arclength.',
        physical_contact_identified=False, identity_confirmed=False, event_timing_validated=False,
        hardware_access=False, blind_20260925_read=False, deployment_authorized=False,
        inference_repeated=False, decoder_replayed_from_input_only=True,
        selected_failure_samples=selected_failures,
        failure_selection='Posthoc two lowest anchor_mix/auto development-holdout coverage cases; input failures retained.',
        figures={p.stem: dict(path=str(p), sha256=sha(p)) for p in (comparison_path, failures_path)},
        limitations='Development holdout has been used historically. Candidate overlap is not rope identity, physical contact, exhaustive precision, or deployment validation.')
    write_json(output/'results.json', report)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(dict(evaluation=str(evaluate(args.base))), ensure_ascii=False))


if __name__ == '__main__':
    main()
