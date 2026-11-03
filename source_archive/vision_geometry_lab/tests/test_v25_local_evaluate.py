import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vision_geometry_lab import v25_local_evaluate as ev
from vision_geometry_lab.v25_local_contract import build_input_pair, decode_local
from vision_geometry_lab.evaluate_centerline_multiscale_v9 import score_decoded
from vision_geometry_lab.v11_path_experiment import sha, write_json, write_npz


def local_case():
    pair = build_input_pair(np.full((240, 320, 3), 128, np.uint8), [[100, 120], [120, 120]], 10.)
    entry = dict(key='sample::p0000', sample_id='sample', split='holdout', group_id='g', cohort='old_pilot', geometry=pair['geometry'])
    reference = dict(key=entry['key'], sample_id='sample', split='holdout', group_id='g',
        future_reference=[[120, 120], [160, 120]], ignore_polygons=[])
    return entry, reference, pair


def test_conditional_denominator_is_new_arc_only_and_empty_prediction_is_not_success():
    entry, ref, pair = local_case()
    row = ev.conditional_case(entry, ref, dict(segments=[]), pair['valid_mask'])
    assert row['metrics']['full_path']['reference_length_px'] == pytest.approx(40.)
    assert row['metrics']['full_path']['covered_fraction'] == 0
    assert row['metrics']['full_path']['prediction_precision_at_5px'] is None
    assert row['given_prefix_in_reference'] is False


def test_curving_behind_plane_keeps_full_new_arc_denominator():
    entry, ref, pair = local_case()
    ref['future_reference'] = [[120, 120], [110, 120], [110, 150]]
    row = ev.conditional_case(entry, ref, dict(segments=[]), pair['valid_mask'])
    assert row['primary_scope'] == 'full_path'
    assert row['metrics']['full_path']['reference_length_px'] == pytest.approx(40.)
    assert row['metrics']['downstream']['reference_length_px'] == 0


def test_conditional_prefix_and_long_candidates_are_rejected():
    entry, ref, pair = local_case()
    with pytest.raises(ValueError, match='prefix'):
        ev.conditional_case(entry, ref, dict(segments=[[[110, 120], [140, 120]]]), pair['valid_mask'])
    with pytest.raises(ValueError, match='horizon'):
        ev.conditional_case(entry, ref, dict(segments=[[[120, 120], [170, 120]]]), pair['valid_mask'])


def test_conditional_aggregate_counts_fixed_tasks_not_deduplicated_old_path():
    entry, ref, pair = local_case()
    row = ev.conditional_case(entry, ref, dict(segments=[ref['future_reference']]), pair['valid_mask'])
    row.update(arm='high', input_mode='high')
    second = copy.deepcopy(row); second['key'] = 'sample::p0001'
    summaries, samples = ev.aggregate_conditional([row, second])
    score = next(s for s in summaries if (s['arm'], s['input_mode'], s['split'], s['cohort']) == ('high', 'high', 'holdout', 'all'))
    assert score['samples'] == 1 and score['patches'] == 2
    assert score['reference_length_px'] == 80 and score['covered_fraction'] == 1
    assert samples[0]['reference_length_px'] == 80


def test_own_arc_cap_preserves_bend_and_does_not_connect_gaps():
    result = ev.limit_candidate_arc([[[0, 0], [10, 0], [10, 10]], [[50, 0], [80, 0]]], 25)
    assert result == [[[0, 0], [10, 0], [10, 10]], [[50., 0.], [55., 0.]]]
    assert ev._length(result) == 25


def test_input_archive_rejects_GT_and_resolution_channel_mismatch(tmp_path):
    entry, _, pair = local_case()
    path = tmp_path/'input.npz'
    write_npz(path, high=pair['high'], low=pair['low'], valid_mask=pair['valid_mask'], future_GT=np.zeros(1))
    entry.update(input_path=str(path), input_sha256=sha(path))
    with pytest.raises(ValueError, match='no GT'):
        ev.load_input(entry, {})
    path2 = tmp_path/'different_query.npz'; wrong = pair['low'].copy(); wrong[4] = 0
    write_npz(path2, high=pair['high'], low=wrong, valid_mask=pair['valid_mask'])
    entry.update(input_path=str(path2), input_sha256=sha(path2))
    with pytest.raises(ValueError, match='query'):
        ev.load_input(entry, {})


def test_replay_api_excludes_reference_and_catches_prediction_tampering(tmp_path):
    entry, _, pair = local_case()
    path = tmp_path/'probability.npz'
    p = np.zeros((96, 192), np.float32); slope = np.zeros(192, np.float32)
    write_npz(path, probability=p, slope=slope)
    entry['input_sha256'] = 'bound-input'
    prediction = dict(key=entry['key'], sample_id=entry['sample_id'], split=entry['split'], input_sha256='bound-input',
        input_mode='high', decoded=decode_local(p, slope, entry['geometry'], pair['valid_mask']),
        probability_path=str(path), probability_sha256=sha(path))
    assert not ev.decode_conditional(entry, prediction, pair, {})['segments']
    prediction['decoded']['segments'] = [[[120, 120], [150, 120]]]
    with pytest.raises(ValueError, match='input-only'):
        ev.decode_conditional(entry, prediction, pair, {})
    assert tuple(inspect.signature(ev.decode_conditional).parameters) == ('entry', 'prediction', 'arrays', 'hashes')


def test_curved_probability_replay_uses_own_arclength_not_GT(tmp_path):
    entry, _, pair = local_case()
    u = -4+(np.arange(192)+.5)/16
    center = 47.5+16*.08*np.maximum(u, 0)**2
    probability = np.exp(-.5*((np.arange(96)[:, None]-center[None])/2)**2)
    probability = (probability/probability.sum(axis=0)).astype(np.float32)
    slope = (.16*np.maximum(u, 0)).astype(np.float32)
    decoded = decode_local(probability, slope, entry['geometry'], pair['valid_mask'])
    assert ev._length(decoded['segments']) > 40
    decoded['segments'] = ev.limit_candidate_arc(decoded['segments'], 40)
    path = tmp_path/'curved.npz'; write_npz(path, probability=probability, slope=slope)
    entry['input_sha256'] = 'input'
    prediction = dict(key=entry['key'], sample_id=entry['sample_id'], split=entry['split'], input_sha256='input', input_mode='low',
        decoded=decoded, probability_path=str(path), probability_sha256=sha(path))
    replayed = ev.decode_conditional(entry, prediction, pair, {})
    assert ev._length(replayed['segments']) == pytest.approx(40)


def integration_fixture(tmp_path, monkeypatch):
    base = tmp_path/'trial'; base.mkdir()
    old = tmp_path/'old'; old.mkdir()
    manifest = tmp_path/'manifest.json'
    write_json(manifest, dict(samples=[])); write_json(old/'protocol.json', {})
    protocol_path = base/'protocol.json'
    write_json(protocol_path, dict(source_snapshot={str(Path(ev.__file__).resolve()): sha(ev.__file__)}))
    ph = sha(protocol_path)
    _, _, pair = local_case()
    input_path = base/'shared_input.npz'; write_npz(input_path, **{name: pair[name] for name in ('high', 'low', 'valid_mask')})
    probability_path = base/'shared_probability.npz'
    probability = np.zeros((96, 192), np.float32); slope = np.zeros(192, np.float32)
    write_npz(probability_path, probability=probability, slope=slope)
    decoded = decode_local(probability, slope, pair['geometry'], pair['valid_mask'])
    image_path = base/'source.png'; Image.new('RGB', (320, 240), '#999999').save(image_path)
    image_hash = sha(image_path)
    records, entries, old_entries, old_rows, auto_rows = [], {}, [], [], []
    for i in range(18):
        sid = f'sample_{i:02d}'; key = sid+'::p0000'; split = 'train' if i < 10 else 'holdout'; group = split+str(i)
        sample = dict(sample_id=sid, split=split, group_id=group, image_path=str(image_path), image_sha256=image_hash, image_shape_hw=[240, 320])
        label = dict(status='partial', polylines=dict(free_seed=[[100, 120], [120, 120]], centerline=[[80, 120], [180, 120]]), ignore_polygons=[])
        quality = dict(precision_exclude_polygons=[], quality_sha256=None)
        records.append(dict(sample=sample, label=label, quality=quality, annotation_sha256='annotation', cohort='old_pilot'))
        reference = dict(key=key, sample_id=sid, split=split, group_id=group, future_reference=[[120, 120], [160, 120]],
            ignore_polygons=[], protocol_sha256=ph, annotation_sha256='annotation', quality_sha256=None)
        ref_path = base/'refs'/(sid+'.json'); write_json(ref_path, reference)
        entries[key] = dict(key=key, sample_id=sid, split=split, group_id=group, cohort='old_pilot',
            geometry=pair['geometry'], input_path=str(input_path), input_sha256=sha(input_path), reference_path=str(ref_path), reference_sha256=sha(ref_path))
        width = dict(width_px=10.)
        old_entries.append(dict(sample_id=sid, annotation_sha256='annotation', quality_sha256=None, width=width))
        old_rows.append(dict(sample_id=sid, independent_width=width, **score_decoded(label, quality, dict(segments=[]), None, [240, 320])))
        available = i not in (0, 11, 12, 13)
        auto_rows.append(dict(sample_id=sid, split=split, group_id=group, source_image_path=str(image_path), source_image_sha256=image_hash,
            width_px=10. if available else None, past_seed_xy=label['polylines']['free_seed'] if available else None,
            available=available, reason='ok' if available else 'missing'))
    auto_path = base/'cache/auto_sources.json'; write_json(auto_path, dict(protocol_sha256=ph, rows=auto_rows))
    index = dict(protocol_sha256=ph, source_sha256={}, entries=entries, evaluation_keys=list(entries),
        ground_truth_prefix_conditioned=True, automatic_evaluation=False, holdout_targets_built=False,
        auto_sources_path=str(auto_path), auto_sources_sha256=sha(auto_path))
    write_json(base/'cache/index.json', index)
    write_json(old/'development_cache/index.json', dict(entries=old_entries, manifest_sha256=sha(manifest), protocol_sha256=sha(old/'protocol.json')))
    write_json(old/'development/evaluation.json', dict(rows=old_rows))
    for arm in ev.ARMS:
        model_path = base/'models'/arm/'checkpoint.pt'; model_path.parent.mkdir(parents=True); model_path.write_bytes(arm.encode())
        predictions = [dict(key=key, sample_id=entry['sample_id'], split=entry['split'], input_mode=mode, decoded=decoded,
            input_sha256=sha(input_path), probability_path=str(probability_path), probability_sha256=sha(probability_path), forward_ms=1., decode_ms=1.)
            for key, entry in entries.items() for mode in ev.MODES]
        write_json(base/'exports'/arm/'conditional.json', dict(arm=arm, protocol_sha256=ph, model_path=str(model_path), model_sha256=sha(model_path), input_index_sha256=sha(base/'cache/index.json'), rows=predictions))
        automatic = [dict(sample_id=row['sample_id'], split=row['split'], available=row['available'], reason=row['reason'], segments=[], steps=[],
            auto_input_sha256=sha(auto_path), source_image_sha256=image_hash) for row in auto_rows]
        write_json(base/'exports'/arm/'auto.json', dict(arm=arm, protocol_sha256=ph, model_path=str(model_path), model_sha256=sha(model_path), rows=automatic))
    monkeypatch.setattr(ev, 'MANIFEST', manifest); monkeypatch.setattr(ev, 'V11_BASE', old)
    monkeypatch.setattr(ev, 'verify', lambda path: None); monkeypatch.setattr(ev, '_dataset', lambda path: ({}, records, {}))
    monkeypatch.setattr(ev, 'collect_sources', lambda: [Path(ev.__file__)])
    monkeypatch.setattr(ev, 'render_comparison', lambda *args: Image.new('RGB', (20, 20)))
    monkeypatch.setattr(ev, 'render_auto_examples', lambda *args: (Image.new('RGB', (20, 20)), []))
    return base, records, index, auto_rows


def test_all_18_and_2x2_conditional_fixed_denominators_with_empty_auto(tmp_path, monkeypatch):
    base, _, _, _ = integration_fixture(tmp_path, monkeypatch)
    output = ev.evaluate(base)
    report = json.loads((output/'report.json').read_text('utf-8'))
    assert len(report['conditional']['rows']) == 72
    assert len(report['automatic']['rows']) == 36
    assert report['no_training_targets_opened']
    for row in report['conditional']['summaries']:
        if row['cohort'] == 'all':
            assert row['reference_length_px'] == (400 if row['split'] == 'train' else 320)
            assert row['covered_fraction'] == 0
    for row in report['automatic']['summaries']:
        if row['cohort'] == 'all' and row['scope'] == 'downstream':
            assert row['reference_length_px'] == (600 if row['split'] == 'train' else 480)
            assert row['covered_fraction'] == 0 and row['prediction_precision_at_5px'] is None
            assert row['unavailable_prediction_samples'] == (1 if row['split'] == 'train' else 3)
            assert row['actual_input_footprint']['status'] == 'unknown_partial'
            assert row['window_reference_support_fraction'] is None
    with pytest.raises(FileExistsError, match='never overwrite'):
        ev.evaluate(base)


def test_fixed_auto_rows_cannot_drop_failures_or_gain_GT_metadata(tmp_path, monkeypatch):
    base, records, _, sources = integration_fixture(tmp_path, monkeypatch)
    export = ev._read(base/'exports/high/auto.json'); original = copy.deepcopy(export)
    export['rows'].pop()
    with pytest.raises(ValueError, match='fixed 18'):
        ev.validate_auto_export(export, dict(rows=sources), records, export['protocol_sha256'], export['model_sha256'])
    sources[0]['future_reference'] = [[1, 2], [3, 4]]
    with pytest.raises(ValueError, match='whitelist'):
        ev.validate_auto_export(original, dict(rows=sources), records, original['protocol_sha256'], original['model_sha256'])


def test_changed_probability_or_source_hash_is_fatal(tmp_path, monkeypatch):
    base, _, _, _ = integration_fixture(tmp_path, monkeypatch)
    (base/'shared_probability.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Artifact hash changed'):
        ev.evaluate(base)
    assert not (base/'evaluation').exists()


@pytest.mark.parametrize('binding', [None, 'wrong_digest'])
def test_conditional_export_requires_exact_input_index_hash(tmp_path, monkeypatch, binding):
    base, _, _, _ = integration_fixture(tmp_path, monkeypatch)
    path = base/'exports/high/conditional.json'
    export = ev._read(path)
    if binding is None:
        export.pop('input_index_sha256')
    else:
        export['input_index_sha256'] = binding
    path.write_text(json.dumps(export), encoding='utf-8')
    with pytest.raises(ValueError, match='input-index hash mismatch'):
        ev.evaluate(base)
    assert not (base/'evaluation').exists()


def test_quality_C7_trusted_reference_excludes_unknown_tail_and_preserves_empty_penalty():
    label = dict(status='partial', polylines=dict(free_seed=[[0, 0], [20, 0]], centerline=[[0, 0], [200, 0]]), ignore_polygons=[])
    quality = dict(supervision_polylines=dict(centerline=[[0, 0], [100, 0]]), precision_exclude_polygons=[])
    score = score_decoded(label, quality, dict(segments=[]), None, [240, 320])
    assert score['metrics']['full_path']['reference_length_px'] == 100
    assert score['metrics']['downstream']['reference_length_px'] == 80
    assert score['metrics']['downstream']['covered_fraction'] == 0
