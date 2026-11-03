"""Frozen posthoc query-only straight-line baseline for V25 oracle tasks.

This program never opens RGB images, input NPZs, training targets or models.
All straight predictions are published before future references are parsed.
It is an explanatory conditional-task audit, NOT an automatic rope tracker.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np

from .v11_path_experiment import sha, write_json
from .v25_local_contract import _checked_geometry


LAB = Path(__file__).resolve().parent
V25_BASE = LAB/'outputs/v25_local_continuation/trial_001'
BASE = LAB/'outputs/v25_geometry_baseline_audit/trial_001'
PARAMETERS = dict(horizon_widths=4., direction='normalized last-minus-first chord of supplied past_seed_xy',
    origin='last supplied past_seed_xy point; must equal frozen geometry.origin_xy',
    direction_must_equal_frozen_geometry=True, matching_tolerance_native_px=5., primary_scope='full_path',
    no_RGB=True, no_valid_mask=True, no_model=True, no_future_GT_for_prediction=True,
    forced_fixed_horizon=True, no_visibility_or_confidence_stop=True, no_image_boundary_clipping=True,
    fixed_train_samples=10, fixed_holdout_samples=8, fixed_train_patches=113, fixed_holdout_patches=86,
    correct_GT_past_prefix_conditioned=True, automatic=False, parameter_search=False)
DEPENDENCIES = ('v25_geometry_baseline_audit.py', 'tests/test_v25_geometry_baseline_audit.py',
    'v25_local_contract.py', 'v11_path_experiment.py', 'centerline_path_metrics_v6.py',
    'centerline_review_v5.py', 'v18_front_diagnostics.py')


def _safe(path):
    path = Path(path)
    if '20260925' in str(path):
        raise ValueError('Reserved blind source forbidden')
    return path


def _read(path):
    return json.loads(_safe(path).read_text('utf-8-sig'))


def straight_query_baseline(geometry):
    """Pure supplied-query geometry; the API accepts neither RGB nor future."""
    origin, direction, _, width = _checked_geometry(geometry)
    seed = np.asarray(geometry['past_seed_xy'], float)
    if seed.ndim != 2 or seed.shape[1] != 2 or len(seed) < 2 or not np.isfinite(seed).all():
        raise ValueError('Invalid supplied past query')
    chord = seed[-1]-seed[0]
    length = np.linalg.norm(chord)
    if length <= 1e-10:
        raise ValueError('Past query has no chord direction')
    chord /= length
    if not np.allclose(origin, seed[-1], atol=1e-8, rtol=0) or not np.allclose(direction, chord, atol=1e-8, rtol=0):
        raise ValueError('Frozen geometry differs from actual root past-query chord')
    return dict(segments=[[origin.tolist(), (origin+4*width*direction).tolist()]],
        reason='fixed_four_width_straight_extrapolation_of_given_correct_past',
        visibility_confirmed=False, identity_confirmed=False, GT_future_used=False,
        correct_GT_past_prefix_conditioned=True, automatic=False)


def _check_inventory(index):
    if index.get('ground_truth_prefix_conditioned') is not True or index.get('automatic_evaluation') is not False:
        raise ValueError('Baseline is only for explicitly oracle-prefix conditional patches')
    keys = index['evaluation_keys']; entries = index['entries']
    if len(keys) != len(set(keys)) or set(keys) != set(entries):
        raise ValueError('Conditional patch inventory changed')
    result = {}
    for split, expected_samples, expected_patches in (('train', 10, 113), ('holdout', 8, 86)):
        selected = [entries[key] for key in keys if entries[key]['split'] == split]
        samples = {e['sample_id'] for e in selected}
        if len(selected) != expected_patches or len(samples) != expected_samples:
            raise ValueError('Fixed V25 image/patch count changed: '+split)
        result[split] = dict(samples=len(samples), patches=len(selected))
    if sum(v['patches'] for v in result.values()) != len(keys):
        raise ValueError('Unexpected patch split')
    train = [e for e in entries.values() if e['split'] == 'train']
    holdout = [e for e in entries.values() if e['split'] == 'holdout']
    if ({e['sample_id'] for e in train} & {e['sample_id'] for e in holdout} or
            {e['group_id'] for e in train} & {e['group_id'] for e in holdout}):
        raise ValueError('Train/holdout sample or source group overlap')
    # Validate the exact input direction for ALL patches before freezing.
    for key in keys:
        straight_query_baseline(entries[key]['geometry'])
    return result


def freeze(base, source_base=V25_BASE):
    base, source_base = _safe(base).resolve(), _safe(source_base).resolve()
    if base.exists():
        raise FileExistsError('Do not overwrite a geometry audit')
    index_path, parent_protocol = source_base/'cache/index.json', source_base/'protocol.json'
    index = _read(index_path)
    if index['protocol_sha256'] != sha(parent_protocol):
        raise ValueError('V25 cache protocol mismatch')
    counts = _check_inventory(index)
    # Reference digests are copied from the hash-bound index, not parsed here.
    references = {e['reference_path']: e['reference_sha256'] for e in index['entries'].values()}
    if len(references) != len(index['entries']):
        raise ValueError('Each fixed patch must retain its own reference artifact')
    source_report = source_base/'evaluation/report.json'
    sources = [LAB/name for name in DEPENDENCIES]+[index_path, parent_protocol, source_report]
    protocol = dict(schema='v25_query_geometry_baseline_posthoc_audit_1',
        frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), parameters=PARAMETERS,
        parent_base=str(source_base), input_index_path=str(index_path), input_index_sha256=sha(index_path),
        parent_protocol_sha256=sha(parent_protocol), source_report_path=str(source_report),
        source_report_sha256=sha(source_report), source_hashes={str(p.resolve()):sha(p) for p in sources},
        expected_reference_sha256=references, inventory=counts,
        prediction_boundary='parse only geometry from input index; publish all predictions before parsing future reference JSONs',
        report_bytes_hashed_for_provenance_only_before_prediction=True,
        horizon_caveat='baseline endpoint u=4.0; existing neural decoder last observed column u=3.96875 before own-arc clipping (difference 0.03125w); no model retuning',
        RGB_or_input_archives_opened=False, training=False, posthoc_audit=True,
        historical_V25_results_unchanged=True, automatic_evaluation=False, identity_confirmed=False)
    write_json(base/'protocol.json', protocol)
    write_json(base/'freeze_receipt.json', dict(protocol_sha256=sha(base/'protocol.json')))
    return protocol


def verify(base):
    base = _safe(base).resolve()
    receipt = _read(base/'freeze_receipt.json')
    if sha(base/'protocol.json') != receipt['protocol_sha256']:
        raise ValueError('Frozen baseline protocol changed')
    protocol = _read(base/'protocol.json')
    if protocol['parameters'] != PARAMETERS:
        raise ValueError('Frozen baseline parameters changed')
    for path, digest in protocol['source_hashes'].items():
        if sha(_safe(path)) != digest:
            raise ValueError('Frozen baseline source changed: '+path)
    return protocol


def predict(base):
    base = _safe(base).resolve(); protocol = verify(base)
    destination = base/'predictions.json'
    if destination.exists():
        raise FileExistsError('Never overwrite baseline predictions')
    index = _read(protocol['input_index_path']); _check_inventory(index)
    rows = []
    for key in index['evaluation_keys']:
        entry = index['entries'][key]
        rows.append(dict(key=key, sample_id=entry['sample_id'], split=entry['split'],
            group_id=entry['group_id'], cohort=entry['cohort'], decoded=straight_query_baseline(entry['geometry'])))
    verify(base)
    write_json(destination, dict(protocol_sha256=sha(base/'protocol.json'),
        input_index_sha256=protocol['input_index_sha256'], rows=rows,
        no_RGB_read=True, no_input_npz_read=True, no_future_reference_parsed=True,
        automatic=False, correct_GT_past_prefix_conditioned=True))
    return destination


def score_patch(entry, reference, decoded):
    from .centerline_path_metrics_v6 import evaluate_paths
    for field in ('key', 'sample_id', 'split', 'group_id'):
        if reference[field] != entry[field]:
            raise ValueError('Future reference identity changed')
    geometry = entry['geometry']; future = np.asarray(reference['future_reference'], float)
    if future.ndim != 2 or future.shape[1] != 2 or len(future) < 2 or not np.isfinite(future).all():
        raise ValueError('Invalid future reference')
    length = float(np.linalg.norm(np.diff(future, axis=0), axis=1).sum())
    if not np.isclose(length, 4*geometry['width_px'], atol=1e-6, rtol=0) or not np.allclose(future[0], geometry['origin_xy'], atol=1e-6, rtol=0):
        raise ValueError('Fixed next-4w reference changed')
    return evaluate_paths(future, decoded['segments'], geometry['past_seed_xy'], ignore_polygons=reference['ignore_polygons'])


def _summary(rows, **identity):
    scores = [r['metrics']['full_path'] for r in rows]
    ref = sum(s['reference_length_px'] for s in scores)
    pred = sum(s['predicted_length_px'] for s in scores)
    covered = sum(s['covered_reference_length_px'] or 0 for s in scores)
    extra = sum(s['extra_predicted_length_px'] or 0 for s in scores)
    return dict(**identity, scope='full_path', samples=len({r['sample_id'] for r in rows}), patches=len(rows),
        reference_length_px=ref, predicted_length_px=pred, covered_reference_length_px=covered,
        extra_predicted_length_px=extra, covered_fraction=covered/ref if ref else None,
        prediction_precision_at_5px=1-extra/pred if ref and pred else None,
        given_prefix_in_denominator=False, task='oracle_prefix_next_4w_not_automatic', identity_confirmed=False)


def aggregate(rows):
    summaries, per_sample = [], []
    for method in ('query_straight_4w', 'high_high', 'low_low'):
        for split in ('train', 'holdout'):
            selected = [r for r in rows if r['method'] == method and r['split'] == split]
            for cohort in ('all', 'old_pilot', 'supplement'):
                group = [r for r in selected if cohort == 'all' or r['cohort'] == cohort]
                summaries.append(_summary(group, method=method, split=split, cohort=cohort))
            for sid in sorted({r['sample_id'] for r in selected}):
                group = [r for r in selected if r['sample_id'] == sid]
                per_sample.append(_summary(group, method=method, split=split, sample_id=sid,
                    cohort=group[0]['cohort'], group_id=group[0]['group_id']))
    return summaries, per_sample


def render(summaries, per_sample):
    from PIL import Image, ImageDraw
    from .v18_front_diagnostics import font
    image = Image.new('RGB', (2000, 1840), '#141d29'); draw = ImageDraw.Draw(image)
    pct = lambda value: '未知' if value is None else f'{100*value:.2f}%'
    draw.text((25, 20), 'V25 事后几何先验审计 | 已知正确前段，直线外推 4w，不读取 RGB', font=font(31), fill='white')
    draw.text((25, 74), '方向严格复用输入 geometry：前段首尾弦方向；预测全部保存后才读未来 GT 评分。', font=font(23), fill='#cbd8e7')
    draw.text((25, 112), '同原 113 训练 / 86 开发补丁，5px、仅未来 4w 完整参考；不是 auto，不代表身份识别。', font=font(23), fill='#ffd58a')
    y = 180
    for split in ('train', 'holdout'):
        draw.text((25, y), split+' | 方法                         覆盖率              候选精度              参考弧长(px)', font=font(25), fill='#75d3ff'); y += 43
        for method in ('query_straight_4w', 'high_high', 'low_low'):
            row = next(r for r in summaries if (r['method'], r['split'], r['cohort']) == (method, split, 'all'))
            for x, value in ((25, method), (640, pct(row['covered_fraction'])), (985, pct(row['prediction_precision_at_5px'])), (1340, f"{row['reference_length_px']:.3f}")):
                draw.text((x, y), value, font=font(25), fill='white')
            y += 42
        y += 14
    draw.text((25, y+10), '逐图对比：各格 = 覆盖 / 候选精度（%）；补丁重复任务不可当独立测试图或旧全路径）', font=font(23), fill='#ffd58a'); y += 61
    for x, label in ((25, '图 / 补丁数'), (860, '直线几何'), (1230, 'high→high'), (1600, 'low→low')):
        draw.text((x, y), label, font=font(24), fill='white')
    y += 43
    for split in ('train', 'holdout'):
        draw.text((25, y), split, font=font(24), fill='#75d3ff'); y += 35
        selected = [r for r in per_sample if r['method'] == 'query_straight_4w' and r['split'] == split]
        for baseline in selected:
            sid = baseline['sample_id']
            draw.text((25, y), sid+' / '+str(baseline['patches']), font=font(19), fill='white')
            for x, method in ((860, 'query_straight_4w'), (1230, 'high_high'), (1600, 'low_low')):
                row = next(r for r in per_sample if r['sample_id'] == sid and r['method'] == method)
                draw.text((x, y), pct(row['covered_fraction'])+' / '+pct(row['prediction_precision_at_5px']), font=font(21), fill='white')
            y += 43
    for i, text in enumerate(('这是看过 V25 成绩后追加的单一固定诊断，不能当作预注册模型竞赛或新的盲测。',
        '直线基线总是外推满 4w，不具备模型的置信停止机制；高覆盖可能仅反映短段平滑与 5px 容差。',
        '正确前段本身由 GT 提供；不证明能从自动入口识别入卷绳，也不授权部署或物理接触判断。')):
        draw.text((25, 1660+43*i), text, font=font(22), fill='#cbd8e7')
    return image


def score(base):
    base = _safe(base).resolve(); protocol = verify(base)
    if (base/'report.json').exists() or (base/'comparison.png').exists():
        raise FileExistsError('Do not overwrite audit scores or figures')
    predictions_path = base/'predictions.json'
    if not predictions_path.is_file():
        raise ValueError('Publish all baseline predictions before any future-reference scoring')
    prediction_hash = sha(predictions_path); predictions = _read(predictions_path)
    if predictions['protocol_sha256'] != sha(base/'protocol.json') or predictions['input_index_sha256'] != protocol['input_index_sha256']:
        raise ValueError('Baseline predictions are not frozen-input bound')
    index = _read(protocol['input_index_path']); expected = set(index['evaluation_keys'])
    decoded = {row['key']:row for row in predictions['rows']}
    if len(decoded) != len(predictions['rows']) or set(decoded) != expected:
        raise ValueError('Predictions dropped or duplicated a fixed patch')
    original = _read(protocol['source_report_path'])
    if original['protocol_sha256'] != protocol['parent_protocol_sha256']:
        raise ValueError('Original model scores use another parent protocol')
    from .centerline_path_metrics_v6 import protocol_sha256
    path_protocol = protocol_sha256()
    if original.get('path_protocol_sha256') != path_protocol:
        raise ValueError('Original model scores use another path-scoring protocol')
    rows = []; baseline_by_key = {}
    for key in index['evaluation_keys']:
        entry = index['entries'][key]; prediction = decoded[key]
        if any(prediction[field] != entry[field] for field in ('key', 'sample_id', 'split', 'group_id', 'cohort')):
            raise ValueError('Saved baseline identity differs from frozen input')
        if prediction['decoded'] != straight_query_baseline(entry['geometry']):
            raise ValueError('Saved baseline differs from frozen query-only rule')
        path = entry['reference_path']
        if sha(_safe(path)) != protocol['expected_reference_sha256'][path]:
            raise ValueError('Future reference hash changed')
        reference = _read(path)  # First semantic access to future GT, after export.
        if reference['protocol_sha256'] != protocol['parent_protocol_sha256']:
            raise ValueError('Future reference parent protocol changed')
        metrics = score_patch(entry, reference, prediction['decoded'])
        row = dict(prediction, method='query_straight_4w', metrics=metrics)
        rows.append(row); baseline_by_key[key] = row
    for arm in ('high', 'low'):
        chosen = [r for r in original['conditional']['rows'] if r['arm'] == arm and r['input_mode'] == arm]
        if len(chosen) != len(expected) or {r['key'] for r in chosen} != expected:
            raise ValueError('Model comparison must use the same fixed patches')
        for row in chosen:
            baseline = baseline_by_key[row['key']]
            if row['metrics'].get('protocol_sha256') != path_protocol or baseline['metrics']['protocol_sha256'] != path_protocol:
                raise ValueError('Per-patch path-scoring protocol changed')
            for field in ('sample_id', 'split', 'group_id', 'cohort'):
                if row[field] != baseline[field]:
                    raise ValueError('Model conditional identity changed')
            if not np.isclose(row['metrics']['full_path']['reference_length_px'], baseline['metrics']['full_path']['reference_length_px'], atol=1e-6, rtol=0):
                raise ValueError('Baseline/model reference denominator differs')
            rows.append(dict(row, method=arm+'_'+arm))
    summaries, per_sample = aggregate(rows)
    paired = []
    for row in rows:
        if row['method'] != 'query_straight_4w':
            baseline = baseline_by_key[row['key']]['metrics']['full_path']; model = row['metrics']['full_path']
            paired.append(dict(key=row['key'], sample_id=row['sample_id'], split=row['split'], method=row['method'],
                model_minus_baseline_covered_length_px=model['covered_reference_length_px']-baseline['covered_reference_length_px'],
                model_minus_baseline_coverage=model['covered_fraction']-baseline['covered_fraction']))
    verify(base)
    if sha(predictions_path) != prediction_hash:
        raise ValueError('Prediction export changed during scoring')
    figure = base/'comparison.png'
    with figure.open('xb') as stream:
        render(summaries, per_sample).save(stream, format='PNG')
    report = dict(schema='v25_query_geometry_baseline_results_1', protocol_sha256=sha(base/'protocol.json'),
        path_protocol_sha256=path_protocol,
        parent_protocol_sha256=protocol['parent_protocol_sha256'], prediction_sha256=prediction_hash,
        original_model_report_sha256=protocol['source_report_sha256'], rows=rows, summaries=summaries,
        per_sample=per_sample, paired_patch_differences=paired, parameters=PARAMETERS,
        fixed_denominators_match_original=True, all_199_predictions_precede_future_reference_parsing=True,
        RGB_or_input_npz_read=False, training=False, automatic=False, identity_confirmed=False,
        figure=dict(path=str(figure), sha256=sha(figure)),
        limitations='Posthoc single frozen diagnostic after seeing V25. Correct GT past prefix is given. Forced full-horizon extrapolation has no visibility stop, whereas the neural decoder may abstain/stop; coverage differences therefore include output-length/stop-policy differences, not only visual evidence. The baseline reaches u=4.0; the neural grid ends at observed u=3.96875 (0.03125w earlier) before own-arc clipping. Short smooth arcs and 5px tolerance may dominate. These 199 tasks come from only 18 reused images; not independent images, blind data, automatic identity recovery, or physical contact validation.')
    write_json(base/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'predict', 'score', 'verify', 'run'))
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--source-base', type=Path, default=V25_BASE)
    args = parser.parse_args()
    if args.stage in ('freeze', 'run'):
        freeze(args.base, args.source_base)
    if args.stage in ('predict', 'run'):
        predict(args.base)
    if args.stage in ('score', 'run'):
        result = score(args.base)
        print(json.dumps([r for r in result['summaries'] if r['cohort'] == 'all'], ensure_ascii=True))
    if args.stage == 'verify':
        verify(args.base)


if __name__ == '__main__':
    main()
