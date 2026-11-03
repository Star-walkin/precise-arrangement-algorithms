import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vision_geometry_lab import v25_local_data as data
from vision_geometry_lab.v11_path_experiment import sha
from vision_geometry_lab.v25_local_contract import build_input_pair, TARGET_KEYS


def length(line):
    return np.linalg.norm(np.diff(np.asarray(line), axis=0), axis=1).sum()


def test_exact_arc_slice_keeps_bend_vertices():
    line = [[0, 0], [30, 0], [30, 40]]
    result = data.slice_arc(line, 20, 50)
    np.testing.assert_allclose(result, [[20, 0], [30, 0], [30, 20]])
    assert length(result) == pytest.approx(30)
    with pytest.raises(ValueError, match='outside'):
        data.slice_arc(line, 20, 80)


def test_fixed_anchor_grid_and_manual_seed_plane_do_not_use_future_fit():
    plan = data.patch_plan([[0, 0], [200, 0]], [], [[40, 0], [50, 0]], 10)
    assert [p['anchor_arc_in_piece_px'] for p in plan] == [60, 100, 140]
    for patch in plan:
        assert length(patch['past_seed_xy']) == pytest.approx(20)
        assert length(patch['future_reference']) == pytest.approx(40)
        np.testing.assert_allclose(patch['past_seed_xy'][-1], patch['future_reference'][0])
        assert patch['anchor_downstream_distance_px'] >= 0


def test_quality_gap_never_bridged_and_grid_restarts_within_each_piece():
    hole = [[[90, -10], [110, -10], [110, 10], [90, 10]]]
    plan = data.patch_plan([[0, 0], [200, 0]], hole, [[-20, 0], [-10, 0]], 10)
    assert len(plan) == 2
    assert [p['piece_index'] for p in plan] == [0, 1]
    assert [p['anchor_arc_in_piece_px'] for p in plan] == [20, 20]
    np.testing.assert_allclose(plan[0]['future_reference'], [[20, 0], [60, 0]])
    np.testing.assert_allclose(plan[1]['past_seed_xy'], [[110, 0], [130, 0]])
    np.testing.assert_allclose(plan[1]['future_reference'], [[130, 0], [170, 0]])


def test_C7_truncated_tail_cannot_create_full_future_patch():
    plan = data.patch_plan([[0, 0], [59, 0]], [], [[-20, 0], [-10, 0]], 10)
    assert plan == []
    plan = data.patch_plan([[0, 0], [60, 0]], [], [[-20, 0], [-10, 0]], 10)
    assert len(plan) == 1
    assert plan[0]['future_reference'][-1] == [60., 0.]


def test_changed_future_cannot_change_selected_past_or_input_geometry():
    first = [[80, 100], [100, 100], [180, 100]]
    second = [[80, 100], [100, 100], [160, 150]]
    seed = [[40, 100], [60, 100]]
    a = data.patch_plan(first, [], seed, 10)[0]
    b = data.patch_plan(second, [], seed, 10)[0]
    assert a['past_seed_xy'] == b['past_seed_xy']
    rgb = np.full((240, 320, 3), 128, np.uint8)
    pa = build_input_pair(rgb, a['past_seed_xy'], 10)
    pb = build_input_pair(rgb, b['past_seed_xy'], 10)
    assert pa['geometry'] == pb['geometry']
    np.testing.assert_array_equal(pa['high'], pb['high'])
    np.testing.assert_array_equal(pa['low'], pb['low'])


def fixture(tmp_path, monkeypatch, no_patch=False):
    sources = tmp_path/'sources'; sources.mkdir()
    base = tmp_path/'v25'; base.mkdir()
    protocol = base/'protocol.json'; protocol.write_text('{}', encoding='utf-8')
    v24_protocol = sources/'v24_protocol.json'; v24_protocol.write_text('{}', encoding='utf-8')
    source_npz = sources/'original_input.npz'
    np.savez_compressed(source_npz, input=np.zeros(1), valid_mask=np.ones(1))
    records, entries = [], {}
    all_paths = [protocol, v24_protocol, source_npz]
    for i in range(18):
        sid = f'fixture_{i:02d}'
        split = 'train' if i < 10 else 'holdout'
        image = sources/(sid+'.png')
        Image.fromarray(np.full((240, 320, 3), 128, np.uint8)).save(image)
        label = dict(status='partial', ignore_polygons=[], polylines=dict(
            free_seed=[[60, 100], [92, 100]], centerline=[[60, 100], [100 if no_patch and i == 17 else 220, 100]]))
        annotation = sources/(sid+'.json'); annotation.write_text(json.dumps(label), encoding='utf-8')
        sample = dict(sample_id=sid, split=split, group_id=split+'_session_'+str(i),
            image_path=str(image), image_sha256=sha(image), image_shape_hw=[240, 320],
            annotation_path=str(annotation))
        records.append(dict(sample=sample, label=label, quality=dict(precision_exclude_polygons=[], quality_sha256=None),
            annotation_sha256=sha(annotation), cohort='old_pilot'))
        geometry = dict(source_shape_hw=[240, 320], width_px=8., free_seed_xy=label['polylines']['free_seed'])
        for variant in ('manual', 'auto'):
            available = not (variant == 'auto' and i in (0, 11, 12, 13))
            key = sid+'::'+variant
            entries[key] = dict(sample_id=sid, split=split, group_id=sample['group_id'], variant=variant,
                available=available, reason='ok' if available else 'no_auto_seed',
                geometry=geometry if available else None, original_image_sha256=sample['image_sha256'],
                input_path=str(source_npz) if available else None,
                input_sha256=sha(source_npz) if available else None)
        all_paths.extend([image, annotation])
    source_index = sources/'input_index.json'
    source_index.write_text(json.dumps(dict(protocol_sha256=sha(v24_protocol), entries=entries, evaluation_keys=list(entries))), encoding='utf-8')
    manifest = sources/'manifest.json'
    manifest.write_text(json.dumps(dict(samples=[r['sample'] for r in records])), encoding='utf-8')
    all_paths.extend([source_index, manifest])
    monkeypatch.setattr(data, 'MANIFEST', manifest)
    monkeypatch.setattr(data, 'V24_INDEX', source_index)
    monkeypatch.setattr(data, 'V24_PROTOCOL', v24_protocol)
    monkeypatch.setattr(data, 'collect_sources', lambda: all_paths)
    monkeypatch.setattr(data, '_load_records', lambda: records)
    monkeypatch.setattr(data, '_read_image', lambda sample: np.asarray(Image.open(sample['image_path']).convert('RGB')))
    return base, records, json.loads(source_index.read_text('utf-8'))


def test_prepare_all_fixed_rows_separate_inputs_train_targets_holdout_refs_and_auto(tmp_path, monkeypatch):
    base, records, _ = fixture(tmp_path, monkeypatch)
    original_target = data.targets
    calls = []
    def counted_target(*args):
        calls.append(1)
        return original_target(*args)
    monkeypatch.setattr(data, 'targets', counted_target)
    index = data.prepare(base, sha(base/'protocol.json'))
    assert index['ground_truth_prefix_conditioned'] and not index['automatic_evaluation']
    assert len(index['train_sample_ids']) == 10 and len(index['holdout_sample_ids']) == 8
    assert all(index['train_keys'].values()) and all(index['holdout_keys'].values())
    train_keys = [key for keys in index['train_keys'].values() for key in keys]
    holdout_keys = [key for keys in index['holdout_keys'].values() for key in keys]
    assert len(calls) == len(train_keys)
    assert len(index['evaluation_keys']) == len(train_keys)+len(holdout_keys)
    for key in index['evaluation_keys']:
        entry, arrays = data.load_input(base, key)
        assert set(arrays) == {'high', 'low', 'valid_mask'}
        np.testing.assert_array_equal(arrays['high'][4:], arrays['low'][4:])
        reference = json.loads(Path(entry['reference_path']).read_text('utf-8'))
        assert reference['future_reference_length_px'] == 32.
        assert length(reference['future_reference']) == pytest.approx(32.)
        if entry['split'] == 'train':
            _, target = data.load_train_target(base, key)
            assert set(target) == set(TARGET_KEYS)
        else:
            assert 'target_path' not in entry and 'target_sha256' not in entry
            with pytest.raises(ValueError, match='Holdout'):
                data.load_train_target(base, key)
    assert not (base/'cache/holdout/targets').exists()
    automatic = json.loads((base/'cache/auto_sources.json').read_text('utf-8'))
    assert len(automatic['rows']) == 18
    assert sum(r['available'] for r in automatic['rows']) == 14
    assert all(set(row) == set(data.AUTO_ROW_KEYS) for row in automatic['rows'])
    assert 'annotation' not in json.dumps(automatic) and 'future_reference' not in json.dumps(automatic)
    unavailable = [row for row in automatic['rows'] if not row['available']]
    assert all(row['width_px'] is None and row['past_seed_xy'] is None for row in unavailable)
    with pytest.raises(FileExistsError):
        data.prepare(base, sha(base/'protocol.json'))


def test_a_fixed_image_with_no_patch_is_error_before_cache_creation(tmp_path, monkeypatch):
    base, _, _ = fixture(tmp_path, monkeypatch, no_patch=True)
    with pytest.raises(ValueError, match='no complete oracle patch: fixture_17'):
        data.prepare(base, sha(base/'protocol.json'))
    assert not (base/'cache').exists()


def test_manifest_group_and_manual_availability_cannot_be_silently_changed(tmp_path, monkeypatch):
    _, records, index = fixture(tmp_path, monkeypatch)
    index['entries']['fixture_00::manual']['available'] = False
    with pytest.raises(ValueError, match='original manual'):
        data.source_entries(index, records)
    index['entries']['fixture_00::manual']['available'] = True
    records[10]['sample']['group_id'] = records[0]['sample']['group_id']
    with pytest.raises(ValueError, match='overlap'):
        data.source_entries(index, records)


def test_protocol_and_blind_path_gates_are_checked_before_preparation(tmp_path):
    with pytest.raises(ValueError, match='Freeze'):
        data.prepare(tmp_path, 'not_frozen')
    (tmp_path/'protocol.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='hash mismatch'):
        data.prepare(tmp_path, 'wrong_digest')
    with pytest.raises(ValueError, match='9/25'):
        data._safe_path('D:/dataset/20260925/image.png')
