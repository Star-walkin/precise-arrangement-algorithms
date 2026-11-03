"""Synthetic-only contract checks; no development image/label reads."""
import json

import numpy as np
import pytest

from vision_geometry_lab import v24_anchor_data as m
from vision_geometry_lab.v11_path_contract import build_input, strip_to_image


def test_arclength_not_vertex_fraction_and_retains_corner():
    # Length 40: desired interval 4..28 traverses the corner at length 10.
    got = m.subpolyline([[0, 0], [10, 0], [10, 30]], .1, .7)
    np.testing.assert_allclose(got, [[4, 0], [10, 0], [10, 18]])


def test_reverse_order_preserved():
    got = m.subpolyline([[10, 30], [10, 0], [0, 0]], .25, 1)
    np.testing.assert_allclose(got, [[10, 20], [10, 0], [0, 0]])


def test_duplicate_vertices_do_not_extend_subset():
    got = m.subpolyline([[0, 0], [0, 0], [10, 0], [10, 0], [20, 0]], .25, .75)
    np.testing.assert_allclose(got, [[5, 0], [10, 0], [15, 0]])


@pytest.mark.parametrize('low,high', [(-.1, .4), (.5, .5), (.8, .2), (.1, 1.1), (float('nan'), .5)])
def test_illegal_fractions_rejected(low, high):
    with pytest.raises(ValueError): m.subpolyline([[0, 0], [10, 0]], low, high)


def test_degenerate_or_nonfinite_seed_rejected():
    for seed in ([[0, 0], [0, 0]], [[0, 0], [float('inf'), 1]], [[0, 1]]):
        with pytest.raises(ValueError): m.subpolyline(seed, 0, .5)


def test_holdout_has_no_training_subsets():
    assert m.input_variants('holdout') == ('manual', 'auto', 'probe0', 'probe1')
    assert not set(m.TRAIN_FRACTIONS) & set(m.PROBE_FRACTIONS)
    with pytest.raises(ValueError): m.input_variants('diagnostic_only')


def target_fixture():
    built = build_input(np.zeros((256, 800, 3), np.uint8), [[150, 128], [250, 128]], 4.)
    label = dict(polylines={'centerline': [[150, 128], [700, 128]]}, ignore_polygons=[])
    quality = dict(supervision_polylines={'centerline': [[150, 128], [350, 128]]}, precision_exclude_polygons=[])
    return built, label, quality


def test_trusted_cutoff_remapped_not_original_tail():
    built, label, quality = target_fixture()
    before = built['input'].copy()
    arrays, diagnostic = m.train_target_arrays('train', 'subset0', built['geometry'], built['valid_mask'], label, quality)
    assert set(arrays) == set(m.TARGET_KEYS)
    columns = np.flatnonzero(arrays['column_valid'])
    xy = strip_to_image(np.column_stack([columns, np.full(len(columns), 127.5)]), built['geometry'])
    assert xy[:, 0].max() <= 350.00001
    assert diagnostic['unknown_is_background'] is False
    np.testing.assert_array_equal(before, built['input'])


@pytest.mark.parametrize('split,variant', [('holdout', 'manual'), ('train', 'probe0'), ('train', 'probe1')])
def test_targets_prohibited_for_holdout_and_probes(split, variant):
    built, label, quality = target_fixture()
    with pytest.raises(ValueError):
        m.train_target_arrays(split, variant, built['geometry'], built['valid_mask'], label, quality)


def test_unusable_training_variant_fails_instead_of_dropping():
    built, label, quality = target_fixture()
    quality['supervision_polylines']['centerline'] = []
    with pytest.raises(ValueError, match='no trustworthy'):
        m.train_target_arrays('train', 'subset2', built['geometry'], built['valid_mask'], label, quality)


def test_auto_missing_fallback_is_explicit_but_eval_remains_unavailable():
    entries = {m.entry_key('s', variant): dict(available=variant != 'auto', reason='ambiguous' if variant == 'auto' else 'ok') for variant in m.TRAIN_VARIANTS}
    slots = m._schedule(['s'], entries)['s']
    assert len(slots) == 5
    assert slots[-1]['resolved_key'] == 's::manual'
    assert slots[-1]['scheduled_key'] == 's::auto'
    assert slots[-1]['fallback_reason'] == 'auto_unavailable: ambiguous'
    assert entries['s::auto']['available'] is False


def test_missing_nonautomatic_training_variant_not_silently_replaced():
    entries = {m.entry_key('s', variant): dict(available=variant != 'subset0', reason='bad') for variant in m.TRAIN_VARIANTS}
    with pytest.raises(ValueError): m._schedule(['s'], entries)


def test_requires_parent_freeze_before_any_source_access(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'verify_v11', lambda _: pytest.fail('Should not inspect sources'))
    with pytest.raises(ValueError, match='Freeze parent'): m.prepare(tmp_path)
    (tmp_path/'protocol.json').write_text('{}')
    with pytest.raises(ValueError, match='hash mismatch'): m.prepare(tmp_path, 'wrong')


def test_existing_cache_refused_before_source_reads(tmp_path, monkeypatch):
    (tmp_path/'protocol.json').write_text('{}')
    (tmp_path/'cache').mkdir()
    monkeypatch.setattr(m, 'verify_v11', lambda _: pytest.fail('Should not inspect sources'))
    with pytest.raises(FileExistsError): m.prepare(tmp_path)


def test_load_input_returns_only_inference_arrays_and_checks_hash(tmp_path):
    (tmp_path/'protocol.json').write_text('{}')
    path = tmp_path/'cache/inputs/s/manual.npz'
    arr = np.zeros((8, 256, 512), np.float32)
    m.save(path, input=arr, valid_mask=np.ones((256, 512), bool))
    entry = dict(available=True, input_path=str(path), input_sha256=m.sha(path), input_array_sha256=m.array_sha256(arr))
    m.write(tmp_path/'cache/input_index.json', dict(protocol_sha256=m.sha(tmp_path/'protocol.json'), entries={'s::manual': entry}))
    _, got = m.load_input(tmp_path, 's::manual')
    assert set(got) == set(m.INPUT_ARRAY_KEYS)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'): m.load_input(tmp_path, 's::manual')
