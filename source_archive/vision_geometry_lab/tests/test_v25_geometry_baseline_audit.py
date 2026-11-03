import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vision_geometry_lab import v25_geometry_baseline_audit as audit
from vision_geometry_lab.v25_local_contract import _geometry
from vision_geometry_lab.v11_path_experiment import sha, write_json


def geometry(seed=None):
    return _geometry((240, 320), seed or [[100, 120], [120, 120]], 10.)


def test_baseline_uses_exact_root_chord_not_last_edge_tangent():
    g = geometry([[100, 100], [110, 100], [110, 110]])
    result = audit.straight_query_baseline(g)
    expected = np.array([110, 110])+40*np.array([1, 1])/np.sqrt(2)
    np.testing.assert_allclose(result['segments'][0], [[110, 110], expected])
    assert tuple(inspect.signature(audit.straight_query_baseline).parameters) == ('geometry',)
    assert not result['GT_future_used'] and not result['automatic']


def test_direction_and_origin_must_match_actual_past():
    g = geometry(); g['past_seed_xy'][-1][0] += 1
    with pytest.raises(ValueError, match='past-query chord'):
        audit.straight_query_baseline(g)
    g = geometry(); g['d_xy'] = [0., 1.]; g['n_xy'] = [-1., 0.]
    with pytest.raises(ValueError, match='past-query chord'):
        audit.straight_query_baseline(g)


def test_future_change_cannot_change_prediction_and_new_reference_excludes_prefix():
    g = geometry()
    entry = dict(key='k', sample_id='s', split='holdout', group_id='group', geometry=g)
    ref = dict(key='k', sample_id='s', split='holdout', group_id='group', future_reference=[[120, 120], [160, 120]], ignore_polygons=[])
    prediction = audit.straight_query_baseline(g)
    perfect = audit.score_patch(entry, ref, prediction)
    ref['future_reference'] = [[120, 120], [120, 160]]
    turning = audit.score_patch(entry, ref, prediction)
    assert perfect['full_path']['reference_length_px'] == turning['full_path']['reference_length_px'] == 40
    assert perfect['full_path']['covered_fraction'] == 1
    assert turning['full_path']['covered_fraction'] < .2
    assert prediction == audit.straight_query_baseline(g)


def test_returning_reference_is_not_halfplane_clipped():
    g = geometry(); entry = dict(key='k', sample_id='s', split='holdout', group_id='g', geometry=g)
    ref = dict(key='k', sample_id='s', split='holdout', group_id='g', future_reference=[[120, 120], [110, 120], [110, 150]], ignore_polygons=[])
    score = audit.score_patch(entry, ref, audit.straight_query_baseline(g))
    assert score['full_path']['reference_length_px'] == 40
    assert score['downstream']['reference_length_px'] == 0


def test_baseline_is_forced_horizon_even_outside_image_not_visibility_claim():
    g = _geometry((240, 320), [[285, 120], [305, 120]], 10.)
    out = audit.straight_query_baseline(g)
    assert out['segments'][0][-1] == [345., 120.]
    assert out['visibility_confirmed'] is False


def fixture(tmp_path, monkeypatch):
    source = tmp_path/'v25'; source.mkdir()
    write_json(source/'protocol.json', {})
    ph = sha(source/'protocol.json'); entries, models = {}, []
    for i in range(18):
        split = 'train' if i < 10 else 'holdout'; count = (12 if i < 3 else 11) if i < 10 else (11 if i < 16 else 10)
        sid = f'{split}_{i:02d}'
        for p in range(count):
            key = sid+'::p'+str(p); ref_path = source/'references'/(key.replace(':', '_')+'.json')
            entry = dict(key=key, sample_id=sid, split=split, group_id=split+str(i), cohort='old_pilot', geometry=geometry())
            reference = dict(key=key, sample_id=sid, split=split, group_id=entry['group_id'],
                future_reference=[[120, 120], [160, 120]], ignore_polygons=[], protocol_sha256=ph)
            write_json(ref_path, reference)
            entry.update(reference_path=str(ref_path), reference_sha256=sha(ref_path),
                input_path=str(source/'MUST_NOT_OPEN_RGB_INPUT.npz'), target_path=str(source/'MUST_NOT_OPEN_TARGET.npz'))
            entries[key] = entry
            for arm in ('high', 'low'):
                decoded = dict(segments=[])
                metrics = audit.score_patch(entry, reference, decoded)
                models.append(dict(key=key, sample_id=sid, split=split, group_id=entry['group_id'], cohort='old_pilot',
                    arm=arm, input_mode=arm, decoded=decoded, metrics=metrics))
    index = dict(protocol_sha256=ph, entries=entries, evaluation_keys=list(entries),
        ground_truth_prefix_conditioned=True, automatic_evaluation=False)
    write_json(source/'cache/index.json', index)
    write_json(source/'evaluation/report.json', dict(protocol_sha256=ph,
        path_protocol_sha256=models[0]['metrics']['protocol_sha256'], conditional=dict(rows=models)))
    monkeypatch.setattr(audit, 'render', lambda *args: Image.new('RGB', (20, 20)))
    return tmp_path/'audit', source


def test_freeze_predict_then_score_all_fixed_tasks_without_images_or_targets(tmp_path, monkeypatch):
    base, source = fixture(tmp_path, monkeypatch)
    audit.freeze(base, source)
    original_read = audit._read; references = []
    def checked_read(path):
        if 'references' in Path(path).parts:
            assert (base/'predictions.json').is_file()
            references.append(str(path))
        assert Path(path).suffix not in ('.npz', '.png')
        return original_read(path)
    monkeypatch.setattr(audit, '_read', checked_read)
    audit.predict(base)
    assert references == []
    report = audit.score(base)
    assert len(references) == 199 and report['fixed_denominators_match_original']
    assert len(report['per_sample']) == 54
    for split, length in (('train', 4520), ('holdout', 3440)):
        row = next(r for r in report['summaries'] if (r['method'], r['split'], r['cohort']) == ('query_straight_4w', split, 'all'))
        assert row['covered_fraction'] == row['prediction_precision_at_5px'] == 1
        assert row['reference_length_px'] == length
    with pytest.raises(FileExistsError): audit.predict(base)
    with pytest.raises(FileExistsError): audit.score(base)
    with pytest.raises(FileExistsError): audit.freeze(base, source)


def test_scoring_before_prediction_is_forbidden(tmp_path, monkeypatch):
    base, source = fixture(tmp_path, monkeypatch)
    audit.freeze(base, source)
    with pytest.raises(ValueError, match='Publish all'):
        audit.score(base)


def test_future_reference_change_is_detected_only_after_predictions(tmp_path, monkeypatch):
    base, source = fixture(tmp_path, monkeypatch)
    protocol = audit.freeze(base, source)
    path = Path(next(iter(protocol['expected_reference_sha256'])))
    path.write_text('{}', encoding='utf-8')
    audit.predict(base)
    with pytest.raises(ValueError, match='Future reference hash'):
        audit.score(base)
    assert (base/'predictions.json').exists() and not (base/'report.json').exists()


def test_changed_source_protocol_and_inventory_are_rejected(tmp_path, monkeypatch):
    base, source = fixture(tmp_path, monkeypatch)
    index = audit._read(source/'cache/index.json'); index['evaluation_keys'].pop()
    with pytest.raises(ValueError, match='inventory'):
        audit._check_inventory(index)
    audit.freeze(base, source)
    (source/'protocol.json').write_text('{"changed":true}', encoding='utf-8')
    with pytest.raises(ValueError, match='source changed'):
        audit.predict(base)


def test_original_and_per_patch_path_protocols_must_match(tmp_path, monkeypatch):
    base, source = fixture(tmp_path, monkeypatch)
    report_path = source/'evaluation/report.json'; report = audit._read(report_path)
    report['conditional']['rows'][0]['metrics']['protocol_sha256'] = 'different_protocol'
    report_path.write_text(json.dumps(report), encoding='utf-8')
    audit.freeze(base, source); audit.predict(base)
    with pytest.raises(ValueError, match='Per-patch path-scoring protocol'):
        audit.score(base)
