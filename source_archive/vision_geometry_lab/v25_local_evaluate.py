"""V25 posthoc evaluation: oracle-prefix local tasks and true auto rollout.

No fitting, threshold search or target-assisted inference. The two task types
have different reference populations and must never share an aggregate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .centerline_path_metrics_v6 import evaluate_paths, protocol_sha256, _sampling
from .centerline_review_v5 import read_clean_image
from .evaluate_centerline_multiscale_v9 import _dataset
from .v11_path_experiment import BASE as V11_BASE, sha, verify, write_json
from .v18_front_diagnostics import font
from .v24_anchor_evaluate import score_case, aggregate_fixed
from .v25_local_contract import HEIGHT, WIDTH, decode_local, image_to_local
from .v25_local_data import MANIFEST, _read, _safe_path, slice_arc, INPUT_KEYS, AUTO_ROW_KEYS


ARMS = ('high', 'low')
MODES = ('high', 'low')
LAB = Path(__file__).resolve().parent
DEPENDENCY_NAMES = ('v25_local_evaluate.py', 'v25_local_data.py', 'v25_local_contract.py',
    'v24_anchor_evaluate.py', 'v23_path_prefix_audit.py', 'evaluate_centerline_multiscale_v9.py',
    'centerline_path_metrics_v6.py', 'centerline_review_v5.py', 'centerline_quality_v5.py',
    'centerline_quality_cutoff_v8.py', 'prepare_centerline_v5.py', 'centerline_benchmark_v5.py',
    'rope_detector.py', 'centerline_multiscale_contract_v9.py', 'centerline_model_v5.py',
    'v11_path_contract.py', 'v11_path_decode.py', 'v11_path_experiment.py', 'v18_front_diagnostics.py')


def collect_sources():
    return [LAB/name for name in DEPENDENCY_NAMES]+[MANIFEST,
        V11_BASE/'development_cache/index.json', V11_BASE/'development/evaluation.json', V11_BASE/'protocol.json']


def _bound(path, expected, hashes):
    path = _safe_path(path)
    digest = sha(path)
    if digest != expected:
        raise ValueError('Artifact hash changed: '+str(path))
    hashes[str(path.resolve())] = digest


def _snapshot(values, hashes):
    for path, digest in values.items():
        if digest is None:
            if _safe_path(path).exists():
                raise ValueError('Previously absent source appeared: '+str(path))
        else:
            _bound(path, digest, hashes)


def _segments(decoded):
    if not isinstance(decoded, dict) or 'segments' not in decoded:
        raise ValueError('Decoded candidate must have explicit segments')
    if decoded.get('identity_confirmed', False):
        raise ValueError('A geometric candidate cannot confirm rope identity')
    result = decoded['segments']
    if not isinstance(result, list):
        raise ValueError('Segments must be a list')
    for segment in result:
        points = np.asarray(segment, float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError('Invalid prediction segment')
    return result


def _length(segments):
    return float(sum(np.linalg.norm(np.diff(np.asarray(s, float), axis=0), axis=1).sum() for s in segments))


def limit_candidate_arc(segments, maximum_px):
    """Input-only cap; never use GT to find an endpoint or join segments."""
    remaining, result = float(maximum_px), []
    for segment in segments:
        length = _length([segment])
        if length <= 1e-10:
            continue
        if remaining <= 1e-10:
            break
        result.append(segment if length <= remaining else slice_arc(segment, 0., remaining))
        remaining -= length
    return result


def _same_segments(first, second):
    return len(first) == len(second) and all(np.shape(a) == np.shape(b) and
        np.allclose(a, b, atol=1e-6, rtol=0) for a, b in zip(first, second))


def validate_inventory(index, records):
    samples = {r['sample']['sample_id']: r['sample'] for r in records}
    if len(samples) != 18 or len(records) != 18 or sum(s['split'] == 'train' for s in samples.values()) != 10 or sum(s['split'] == 'holdout' for s in samples.values()) != 8:
        raise ValueError('Fixed 10 train + 8 development holdout samples required')
    if {s['group_id'] for s in samples.values() if s['split'] == 'train'} & {s['group_id'] for s in samples.values() if s['split'] == 'holdout'}:
        raise ValueError('Train/holdout groups overlap')
    keys = index['evaluation_keys']
    if len(keys) != len(set(keys)) or set(keys) != set(index['entries']):
        raise ValueError('Conditional patch inventory changed')
    if index.get('ground_truth_prefix_conditioned') is not True or index.get('automatic_evaluation') is not False or index.get('holdout_targets_built') is not False:
        raise ValueError('Conditional cache must explicitly separate oracle tasks and holdout targets')
    covered = set()
    for key in keys:
        entry = index['entries'][key]; sid = entry['sample_id']
        if sid not in samples or entry['split'] != samples[sid]['split'] or entry['group_id'] != samples[sid]['group_id']:
            raise ValueError('Conditional patch sample/group/split changed')
        if entry['split'] == 'holdout' and ('target_path' in entry or 'target_sha256' in entry):
            raise ValueError('Holdout training-target artifact forbidden')
        covered.add(sid)
    if covered != set(samples):
        raise ValueError('At least one fixed sample lost all conditional patches')
    return samples


def load_input(entry, hashes):
    _bound(entry['input_path'], entry['input_sha256'], hashes)
    with np.load(entry['input_path'], allow_pickle=False) as archive:
        if set(archive.files) != set(INPUT_KEYS):
            raise ValueError('Input archive must contain high/low/valid_mask only; no GT')
        arrays = {name: archive[name].copy() for name in INPUT_KEYS}
    valid = arrays['valid_mask']
    if valid.shape != (HEIGHT, WIDTH) or not np.isin(valid, [0, 1]).all():
        raise ValueError('Invalid common valid mask')
    for mode in MODES:
        x = arrays[mode]
        if x.shape != (8, HEIGHT, WIDTH) or x.dtype != np.float32 or not np.isfinite(x).all() or not np.array_equal(x[7], valid):
            raise ValueError('Invalid paired inference tensor')
    if not np.array_equal(arrays['high'][4:], arrays['low'][4:]):
        raise ValueError('Resolution arms changed query/coordinates/valid channels')
    return arrays


def decode_conditional(entry, prediction, arrays, hashes):
    """Replay purely from saved probability, local geometry and actual valid."""
    if any(prediction[k] != entry[k] for k in ('key', 'sample_id', 'split')) or prediction['input_sha256'] != entry['input_sha256']:
        raise ValueError('Conditional inference input binding changed')
    if prediction['input_mode'] not in MODES:
        raise ValueError('Unknown conditional input resolution')
    segments = _segments(prediction['decoded'])
    _bound(prediction['probability_path'], prediction['probability_sha256'], hashes)
    with np.load(prediction['probability_path'], allow_pickle=False) as archive:
        if set(archive.files) != {'probability', 'slope'}:
            raise ValueError('Probability archive must contain probability/slope only; no GT')
        replayed = decode_local(archive['probability'], archive['slope'], entry['geometry'], arrays['valid_mask'])
    capped = limit_candidate_arc(replayed['segments'], 4*entry['geometry']['width_px'])
    if not _same_segments(capped, segments):
        raise ValueError('Conditional prediction differs from input-only decoder and own-arc cap')
    return prediction['decoded']


def local_footprint(reference_segments, geometry, valid):
    total = supported = 0.
    for piece in _sampling(reference_segments):
        uv = image_to_local(piece['points'], geometry)
        inside = ((uv[:, 0] >= 0) & (uv[:, 0] < WIDTH-1) & (uv[:, 1] >= 0) & (uv[:, 1] < HEIGHT-1))
        values = cv2.remap(valid.astype(np.float32), uv[:, 0].astype(np.float32)[None], uv[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[0]
        total += piece['length']; supported += float(piece['weights'][inside & (values >= 1-1e-6)].sum())
    return dict(status='measured', reference_length_px=float(total), supported_reference_length_px=float(supported),
        fraction=supported/total if total else None, does_not_change_score_denominator=True)


def conditional_case(entry, reference, decoded, valid):
    """Score only the trusted NEW 4w arc, never the given correct prefix."""
    for field in ('key', 'sample_id', 'split', 'group_id'):
        if reference[field] != entry[field]:
            raise ValueError('Conditional reference identity changed')
    geometry = entry['geometry']; future = reference['future_reference']
    width = geometry['width_px']
    if not np.isclose(_length([future]), 4*width, atol=1e-6, rtol=0) or not np.allclose(future[0], geometry['past_seed_xy'][-1], atol=1e-6, rtol=0):
        raise ValueError('Reference must be exactly the new 4w arc at the given prefix endpoint')
    segments = _segments(decoded)
    if _length(segments) > 4*width+1e-6:
        raise ValueError('Conditional candidate exceeds the fixed own-arc horizon')
    for segment in segments:
        u = (np.asarray(segment)-geometry['origin_xy'])@np.asarray(geometry['d_xy'])/width
        if np.min(u) < -1e-6 or np.max(u) > 4+1e-6:
            raise ValueError('Conditional output contains prefix or out-of-horizon predictions')
    metrics = evaluate_paths(future, segments, geometry['past_seed_xy'], ignore_polygons=reference['ignore_polygons'])
    return dict(key=entry['key'], sample_id=entry['sample_id'], split=entry['split'], group_id=entry['group_id'],
        cohort=entry['cohort'], metrics=metrics, decoded=decoded,
        primary_scope='full_path', reference_length_before_exclusion_px=4*width,
        actual_input_footprint=local_footprint(metrics['score_support']['full_reference_segments'], geometry, valid),
        ground_truth_prefix_conditioned=True, automatic_evaluation=False,
        given_prefix_in_reference=False, future_GT_used_for_prediction=False,
        quality_used_for_decode=False, identity_confirmed=False)


def _conditional_summary(rows, **identity):
    scores = [r['metrics']['full_path'] for r in rows]
    ref = sum(s['reference_length_px'] for s in scores)
    covered = sum(s['covered_reference_length_px'] or 0 for s in scores)
    pred = sum(s['predicted_length_px'] for s in scores)
    extra = sum(s['extra_predicted_length_px'] or 0 for s in scores)
    footprint = sum(r['actual_input_footprint']['supported_reference_length_px'] for r in rows)
    return dict(**identity, scope='full_path', task='oracle_correct_prefix_new_4w',
        samples=len({r['sample_id'] for r in rows}), patches=len(rows),
        reference_length_px=ref, covered_reference_length_px=covered, predicted_length_px=pred,
        extra_predicted_length_px=extra, covered_fraction=covered/ref if ref else None,
        prediction_precision_at_5px=1-extra/pred if pred and ref else None,
        input_supported_reference_fraction=footprint/ref if ref else None,
        empty_prediction_patches=sum(not r['decoded']['segments'] for r in rows),
        reference_policy='sum of fixed conditional 4w tasks; repeated/overlapping task support is not deduplicated into an old full path',
        identity_confirmed=False)


def aggregate_conditional(rows):
    summaries, samples = [], []
    for arm in ARMS:
        for mode in MODES:
            for split in ('train', 'holdout'):
                selected = [r for r in rows if (r['arm'], r['input_mode'], r['split']) == (arm, mode, split)]
                for cohort in ('all', 'old_pilot', 'supplement'):
                    group = [r for r in selected if cohort == 'all' or r['cohort'] == cohort]
                    summaries.append(_conditional_summary(group, arm=arm, input_mode=mode, split=split, cohort=cohort))
                for sid in sorted({r['sample_id'] for r in selected}):
                    group = [r for r in selected if r['sample_id'] == sid]
                    samples.append(_conditional_summary(group, arm=arm, input_mode=mode, split=split,
                        sample_id=sid, group_id=group[0]['group_id'], cohort=group[0]['cohort']))
    return summaries, samples


def validate_auto_export(export, sources, records, protocol_hash, model_hash):
    if export.get('protocol_sha256') != protocol_hash or export.get('model_sha256') != model_hash:
        raise ValueError('Automatic export protocol/model binding changed')
    by_sample = {r['sample']['sample_id']: r for r in records}
    source_rows = {r['sample_id']: r for r in sources['rows']}
    predictions = {r['sample_id']: r for r in export['rows']}
    if len(source_rows) != 18 or len(sources['rows']) != 18 or len(predictions) != 18 or len(export['rows']) != 18 or set(predictions) != set(by_sample) or set(source_rows) != set(by_sample):
        raise ValueError('Automatic evaluation must retain all fixed 18 samples')
    for sid, source in source_rows.items():
        row = predictions[sid]; sample = by_sample[sid]['sample']
        if set(source) != set(AUTO_ROW_KEYS):
            raise ValueError('Automatic source whitelist violation: potential GT metadata')
        if source['split'] != sample['split'] or source['group_id'] != sample['group_id'] or source['source_image_sha256'] != sample['image_sha256']:
            raise ValueError('Automatic source sample/group binding changed')
        if row['split'] != sample['split'] or row['available'] != source['available'] or type(row['available']) is not bool or row['source_image_sha256'] != sample['image_sha256']:
            raise ValueError('Automatic export availability/source binding changed')
        _segments(dict(segments=row['segments'], identity_confirmed=row.get('identity_confirmed', False)))
        if not row['available'] and (row['segments'] or row.get('steps')):
            raise ValueError('Unavailable automatic source must remain empty, not disappear')
    return predictions, source_rows


def _pct(value):
    return '未知' if value is None else f'{100*value:.2f}%'


def render_comparison(conditional, automatic):
    image = Image.new('RGB', (1920, 1460), '#141d29'); draw = ImageDraw.Draw(image)
    draw.text((30, 20), 'V25 高分辨率短段延拓 | 条件任务与自动延拓严格分开', font=font(32), fill='white')
    draw.text((30, 75), '同视场、同网络网格；high 与 low 只改变图像细节。开发留出不是新的盲测。', font=font(24), fill='#ffd58a')
    y = 140
    draw.text((30, y), 'A. 给定正确 GT 前段：只评分新的 4 倍绳宽弧长（full_path，不做半平面裁切）', font=font(27), fill='#75d3ff'); y += 58
    headers = [(30, '组别 / 模型→输入'), (565, '图数 / 补丁数'), (810, '覆盖率'), (1050, '候选精度'), (1300, 'GT长度(px)'), (1550, '输入支持')]
    for x, label in headers: draw.text((x, y), label, font=font(22), fill='white')
    y += 46
    for split in ('train', 'holdout'):
        for arm in ARMS:
            for mode in MODES:
                row = next(r for r in conditional if (r['arm'], r['input_mode'], r['split'], r['cohort']) == (arm, mode, split, 'all'))
                values = [f'{split} / {arm} → {mode}', f"{row['samples']} / {row['patches']}", _pct(row['covered_fraction']), _pct(row['prediction_precision_at_5px']), f"{row['reference_length_px']:.1f}", _pct(row['input_supported_reference_fraction'])]
                for (x, _), value in zip(headers, values): draw.text((x, y), value, font=font(23), fill='white')
                y += 45
        y += 12
    draw.text((30, y+10), '补丁分母是固定条件任务之和；不包含给定前段，不能与旧全路径分数直接比较。', font=font(23), fill='#cbd8e7'); y += 88
    draw.text((30, y), 'B. 真正 auto rollout：仅匹配模型→输入，固定原 18 图与原 C7 可信分母', font=font(27), fill='#75d3ff'); y += 58
    heads = [(30, '组别 / 模型→输入'), (565, '可用 / 总图数'), (810, '覆盖率'), (1050, '候选精度'), (1300, '弯段命中'), (1550, '弯段覆盖')]
    for x, label in heads: draw.text((x, y), label, font=font(22), fill='white')
    y += 46
    for split in ('train', 'holdout'):
        for arm in ARMS:
            row = next(r for r in automatic if (r['arm'], r['split'], r['cohort'], r['scope']) == (arm, split, 'all', 'downstream'))
            values = [f'{split} / {arm} → {arm}', f"{row['samples']-row['unavailable_prediction_samples']} / {row['samples']}", _pct(row['covered_fraction']), _pct(row['prediction_precision_at_5px']), f"{row['GT_departure_prediction_hits']}/{row['GT_departure_supported_samples']}", _pct(row['GT_departure_neighborhood_coverage_5px'])]
            for (x, _), value in zip(heads, values): draw.text((x, y), value, font=font(23), fill='white')
            y += 48
    footers = ['auto 输入不可用者保留为空预测；逐步输入足迹未汇总，明确未知，不伪报零。',
        '弯段 = V23 原口径：GT 首次偏离人工自由段直线 0.5 倍原 manual 光学宽；前后 2 倍宽弧长。',
        '这是事后 GT 几何诊断，不是真实接触点；GT 未来不参与解码、自动输入或停止条件。',
        '覆盖和候选精度不证明绳身份；本结果不授权部署、硬件动作或事件/接触判断。']
    for i, line in enumerate(footers): draw.text((30, 1230+i*45), line, font=font(22), fill='#cbd8e7')
    return image


def render_auto_examples(rows, records):
    high = [r for r in rows if r['arm'] == 'high' and r['split'] == 'holdout' and r['prediction_status'] == 'loaded']
    low = {r['sample_id']: r for r in rows if r['arm'] == 'low'}
    if not high:
        high = [r for r in rows if r['arm'] == 'high' and r['split'] == 'holdout']
    score = lambda r: r['metrics']['downstream']['covered_fraction'] or 0.
    worst = min(high, key=lambda r: (score(r), r['sample_id']))
    gain = max(high, key=lambda r: (score(r)-score(low[r['sample_id']]), r['sample_id']))
    selected = [('high 自动覆盖最低例（可用输入优先）', worst), ('high−low 覆盖差最大例（可能仍未改善）', gain)]
    by_sample = {r['sample']['sample_id']: r for r in records}
    canvas = Image.new('RGB', (1920, 1460), '#141d29'); draw = ImageDraw.Draw(canvas)
    draw.text((25, 20), 'V25 auto 事后选例 | 绿=可信 GT；品红=候选；原图全幅，不标真实接触', font=font(29), fill='white')
    selection = []
    for i, (rule, chosen) in enumerate(selected):
        sid = chosen['sample_id']; source = read_clean_image(by_sample[sid]['sample']).convert('RGB')
        top = 85+i*670
        draw.text((25, top), rule+' | '+sid, font=font(22), fill='#75d3ff')
        delta = score(chosen)-score(low[sid]); selection.append(dict(sample_id=sid, rule=rule, high_minus_low_coverage=delta))
        for j, arm in enumerate(ARMS):
            row = next(r for r in rows if r['arm'] == arm and r['sample_id'] == sid)
            preview = ImageOps.contain(source, (930, 485)); paint = ImageDraw.Draw(preview)
            scale = preview.width/source.width
            for key, color in (('full_reference_segments', '#19ff72'), ('full_prediction_segments', '#ff36de')):
                for segment in row['metrics']['score_support'][key]:
                    if len(segment) >= 2: paint.line([(float(x)*scale, float(y)*scale) for x, y in segment], fill=color, width=3)
            left = 15+j*960; canvas.paste(preview, (left, top+45))
            metric = row['metrics']['downstream']
            draw.text((left, top+540), f"{arm} → {arm}  覆盖 {_pct(metric['covered_fraction'])}  候选精度 {_pct(metric['prediction_precision_at_5px'])}", font=font(24), fill='white')
            draw.text((left, top+583), f"输入 {row['prediction_status']} | 候选不确认身份 | 弯段命中 {row['GT_departure'].get('prediction_hits_5px', '未知')}", font=font(20), fill='#ffd58a')
    return canvas, selection


def evaluate(base):
    base = _safe_path(base).resolve(); output = base/'evaluation'
    if output.exists():
        raise FileExistsError('Evaluation output exists; never overwrite frozen results')
    cv2.setNumThreads(2)
    protocol_path, index_path = base/'protocol.json', base/'cache/index.json'
    protocol, index = _read(protocol_path), _read(index_path)
    protocol_hash = sha(protocol_path); hashes = {}
    if index['protocol_sha256'] != protocol_hash:
        raise ValueError('Conditional cache protocol mismatch')
    frozen = {}
    for category in ('source_hashes', 'source_snapshot', 'old_source_snapshot', 'dataset_snapshot'):
        frozen.update(protocol.get(category, {}))
    if str(Path(__file__).resolve()) not in {str(Path(p).resolve()) for p in frozen}:
        raise ValueError('Evaluator source must be frozen before evaluation')
    _snapshot(frozen, hashes); _snapshot(index['source_sha256'], hashes)
    for path in (protocol_path, index_path, *collect_sources()): hashes[str(path.resolve())] = sha(path)
    verify(V11_BASE)
    _, records, _ = _dataset(MANIFEST)
    samples = validate_inventory(index, records); records_by_id = {r['sample']['sample_id']: r for r in records}
    old_cache = _read(V11_BASE/'development_cache/index.json'); old_eval = _read(V11_BASE/'development/evaluation.json')
    if old_cache['manifest_sha256'] != sha(MANIFEST) or old_cache['protocol_sha256'] != sha(V11_BASE/'protocol.json'):
        raise ValueError('Original manual-width provenance changed')
    originals = {e['sample_id']: e for e in old_cache['entries']}; old_rows = {r['sample_id']: r for r in old_eval['rows']}
    if set(originals) != set(samples) or set(old_rows) != set(samples):
        raise ValueError('Original fixed denominator inventory changed')
    for sid, record in records_by_id.items():
        if originals[sid]['annotation_sha256'] != record['annotation_sha256'] or originals[sid]['quality_sha256'] != record['quality']['quality_sha256'] or originals[sid]['width'] != old_rows[sid]['independent_width']:
            raise ValueError('Original trusted quality or manual width changed')
        _bound(record['sample']['image_path'], record['sample']['image_sha256'], hashes)
    auto_path = Path(index['auto_sources_path']); _bound(auto_path, index['auto_sources_sha256'], hashes)
    auto_sources = _read(auto_path)
    if auto_sources['protocol_sha256'] != protocol_hash:
        raise ValueError('Automatic source protocol mismatch')
    arrays, references = {}, {}
    for key in index['evaluation_keys']:
        entry = index['entries'][key]; arrays[key] = load_input(entry, hashes)
        _bound(entry['reference_path'], entry['reference_sha256'], hashes)
        references[key] = _read(entry['reference_path'])
        reference = references[key]; record = records_by_id[entry['sample_id']]
        if reference['protocol_sha256'] != protocol_hash or reference['annotation_sha256'] != record['annotation_sha256'] or reference['quality_sha256'] != record['quality']['quality_sha256']:
            raise ValueError('Conditional reference provenance changed')
    conditional, automatic, models = [], [], {}
    for arm in ARMS:
        export_path = base/'exports'/arm/'conditional.json'; export = _read(export_path)
        hashes[str(export_path.resolve())] = sha(export_path)
        if export['arm'] != arm or export['protocol_sha256'] != protocol_hash:
            raise ValueError('Conditional export arm/protocol mismatch')
        if export.get('input_index_sha256') != sha(index_path):
            raise ValueError('Conditional export input-index hash mismatch')
        digest = export['model_sha256']
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError('Missing model digest')
        models[arm] = digest
        model_path = export.get('model_path', str(base/'models'/arm/'checkpoint.pt'))
        _bound(model_path, digest, hashes)
        predictions = {(r['key'], r['input_mode']): r for r in export['rows']}
        expected = {(key, mode) for key in index['evaluation_keys'] for mode in MODES}
        if len(predictions) != len(export['rows']) or set(predictions) != expected:
            raise ValueError('Conditional export must retain every patch x both input modes')
        for key, mode in sorted(expected):
            entry = index['entries'][key]; prediction = predictions[key, mode]
            decoded = decode_conditional(entry, prediction, arrays[key], hashes)
            row = conditional_case(entry, references[key], decoded, arrays[key]['valid_mask'])
            row.update(arm=arm, input_mode=mode, input_sha256=entry['input_sha256'], probability_sha256=prediction['probability_sha256'], forward_ms=prediction.get('forward_ms'), decode_ms=prediction.get('decode_ms'))
            conditional.append(row)
        auto_export_path = base/'exports'/arm/'auto.json'; auto_export = _read(auto_export_path)
        hashes[str(auto_export_path.resolve())] = sha(auto_export_path)
        if auto_export['arm'] != arm:
            raise ValueError('Automatic arm changed')
        predictions, sources = validate_auto_export(auto_export, auto_sources, records, protocol_hash, digest)
        _bound(auto_export['model_path'], digest, hashes)
        for sid, prediction in predictions.items():
            source = sources[sid]
            if prediction['auto_input_sha256'] != index['auto_sources_sha256']:
                raise ValueError('Automatic initial source file hash mismatch')
            entry = dict(variant='auto', available=source['available'], reason=source['reason'], geometry=None)
            decoded = dict(segments=prediction['segments'], reason=prediction['reason'], identity_confirmed=False)
            row = score_case(records_by_id[sid], entry, decoded, originals[sid]['width'], None)
            for scope in ('full_path', 'downstream'):
                if not np.isclose(row['metrics'][scope]['reference_length_px'], old_rows[sid]['metrics'][scope]['reference_length_px'], atol=1e-6, rtol=0):
                    raise ValueError('Automatic evaluation changed the original fixed reference denominator')
            row.update(arm=arm, input_mode=arm, task='automatic_rollout', ground_truth_prefix_conditioned=False,
                steps=prediction.get('steps', []), auto_input_sha256=prediction['auto_input_sha256'], source_image_sha256=prediction['source_image_sha256'])
            automatic.append(row)
    conditional_summaries, conditional_samples = aggregate_conditional(conditional)
    auto_summaries = [dict(arm=arm, input_mode=arm, task='automatic_rollout', **summary)
        for arm in ARMS for summary in aggregate_fixed([r for r in automatic if r['arm'] == arm])]
    verify(V11_BASE)
    if any(sha(path) != digest for path, digest in hashes.items()):
        raise ValueError('Evaluation artifact changed during scoring')
    output.mkdir(parents=True, exist_ok=False)
    comparison_path, examples_path = output/'comparison.png', output/'auto_examples.png'
    render_comparison(conditional_summaries, auto_summaries).save(comparison_path)
    examples, selection = render_auto_examples(automatic, records); examples.save(examples_path)
    report = dict(schema='v25_local_separate_conditional_and_auto_evaluation_1', protocol_sha256=protocol_hash,
        path_protocol_sha256=protocol_sha256(), model_sha256=models, source_sha256=hashes,
        conditional=dict(rows=conditional, summaries=conditional_summaries, sample_summaries=conditional_samples,
            primary_scope='full_path', given_prefix_excluded=True, reference='trusted next 4w arc only',
            decoder_replayed_without_future_GT=True, distinct_from_old_full_path_denominator=True),
        automatic=dict(rows=automatic, summaries=auto_summaries, all_fixed_18_retained=True,
            original_full_and_downstream_denominators_unchanged=True, rollout_footprint_status='unknown_not_exported',
            GT_departure='exact V23/V24 first 0.5 original manual optical width departure and +/-2w GT arc'),
        no_training_targets_opened=True, thresholds_changed=False, identity_confirmed=False,
        physical_contact_identified=False, deployment_authorized=False, hardware_access=False,
        blind_20260925_read=False, selected_auto_examples=selection,
        figures={p.stem:dict(path=str(p), sha256=sha(p)) for p in (comparison_path, examples_path)},
        limitations='Conditional GT-prefix patches are oracle tasks, not automatic identity recovery. Repeated tasks are length-weighted without deduplication. Development holdout was used historically and is not blind. Candidate overlap is not identity or physical precision.')
    write_json(output/'report.json', report)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(dict(evaluation=str(evaluate(args.base))), ensure_ascii=False))


if __name__ == '__main__':
    main()
