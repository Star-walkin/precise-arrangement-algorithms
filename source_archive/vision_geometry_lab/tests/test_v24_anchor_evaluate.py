import copy
import inspect

import numpy as np
import pytest

from vision_geometry_lab import v24_anchor_evaluate as evaluation
from vision_geometry_lab.v11_path_contract import build_input
from vision_geometry_lab.v11_path_experiment import sha


def fixture():
    label = dict(status='partial', polylines=dict(centerline=[[10, 50], [140, 50]],
        free_seed=[[10, 50], [20, 50]]), ignore_polygons=[])
    quality = dict(precision_exclude_polygons=[], quality_sha256=None)
    record = dict(sample=dict(sample_id='fixture', split='holdout', group_id='development',
        image_shape_hw=[100, 160]), label=label, quality=quality, cohort='old_pilot')
    entry = dict(sample_id='fixture', split='holdout', variant='auto', available=False,
        reason='no_anchor', input_path=None, input_sha256=None, geometry=None)
    decoded = dict(segments=[], reason='no_anchor', identity_confirmed=False)
    width = dict(width_px=4., source='original manual')
    return record, entry, decoded, width


def test_unavailable_empty_prediction_keeps_fixed_denominator_and_unknown_footprint():
    record, entry, decoded, width = fixture()
    row = evaluation.score_case(record, entry, decoded, width, None)
    assert row['metrics']['downstream']['reference_length_px'] == pytest.approx(120)
    assert row['metrics']['downstream']['covered_fraction'] == 0
    assert row['metrics']['downstream']['prediction_precision_at_5px'] is None
    assert row['actual_input_footprint']['downstream']['supported_reference_length_px'] is None
    summary = next(s for s in evaluation.aggregate_fixed([row])
                   if (s['split'], s['cohort'], s['scope']) == ('holdout', 'all', 'downstream'))
    assert summary['reference_length_px'] == pytest.approx(120)
    assert summary['covered_fraction'] == 0
    assert summary['window_reference_support_fraction'] is None
    assert summary['actual_input_footprint']['unknown_samples'] == 1


def test_footprint_zero_is_measured_only_when_valid_mask_was_observed():
    image = np.zeros((100, 160, 3), np.uint8)
    built = build_input(image, [[10, 50], [20, 50]], 4.)
    zero = np.zeros_like(built['valid_mask'])
    support = evaluation.input_footprint([[[20, 50], [140, 50]]], built['geometry'], zero)
    assert support['status'] == 'measured'
    assert support['supported_reference_length_px'] == 0
    assert support['fraction'] == 0


def test_input_support_does_not_crop_denominator():
    record, entry, decoded, width = fixture()
    built = build_input(np.zeros((100, 160, 3), np.uint8), [[10, 50], [20, 50]], 4.)
    entry.update(available=True, geometry=built['geometry'])
    built['valid_mask'][:] = False
    row = evaluation.score_case(record, entry, decoded, width, built)
    assert row['metrics']['downstream']['reference_length_px'] == pytest.approx(120)
    assert row['actual_input_footprint']['downstream']['fraction'] == 0


def test_quality_trusted_cutoff_preserved_without_mutating_prediction():
    record, entry, decoded, width = fixture()
    decoded['segments'] = [[[20, 50], [140, 50]]]
    before = copy.deepcopy(decoded)
    record['quality']['supervision_polylines'] = {'centerline': [[10, 50], [80, 50]]}
    record['quality']['precision_exclude_polygons'] = [[[45, 40], [55, 40], [55, 60], [45, 60]]]
    row = evaluation.score_case(record, entry, decoded, width, None)
    assert row['metrics']['downstream']['reference_length_px'] == pytest.approx(50)
    assert row['reference_source'] == 'hash_bound_confirmed_trusted_centerline'
    assert decoded == before
    assert row['quality_used_for_decode'] is False
    assert len(row['metrics']['score_support']['downstream_reference_segments']) == 2


def test_GT_departure_uses_original_manual_width_not_variant_geometry():
    record, entry, decoded, width = fixture()
    record['label']['polylines']['centerline'] = [[10, 50], [30, 50], [60, 80]]
    entry['geometry'] = {'width_px': 100.}
    row = evaluation.score_case(record, entry, decoded, width, None)
    diagnostic = row['GT_departure']
    np.testing.assert_allclose(diagnostic['point_xy'], [32, 52])
    assert diagnostic['width_px'] == 4
    assert row['input_geometry_width_px'] == 100
    assert not diagnostic['is_contact'] and not diagnostic['used_to_crop_inference']
    assert diagnostic['neighborhood_half_widths'] == 2.


def test_decoder_API_has_no_GT_or_quality_argument():
    assert tuple(inspect.signature(evaluation.decode_export).parameters) == ('entry', 'prediction', 'arrays', 'hashes')


def test_prediction_archive_rejects_GT_even_with_valid_hash(tmp_path):
    _, entry, decoded, _ = fixture()
    built = build_input(np.zeros((100, 160, 3), np.uint8), [[10, 50], [20, 50]], 4.)
    entry.update(available=True, geometry=built['geometry'])
    path = tmp_path/'contaminated.npz'
    np.savez_compressed(path, probability=np.zeros((256, 512), np.float32),
        slope=np.zeros(512, np.float32), target_distribution=np.zeros((256, 512), np.float32))
    prediction = dict(entry, decoded=decoded, probability_path=str(path), probability_sha256=sha(path))
    with pytest.raises(ValueError, match='probability and slope only; no GT'):
        evaluation.decode_export(entry, prediction, built, {})


def test_input_archive_rejects_supervision_arrays(tmp_path):
    _, entry, _, _ = fixture()
    path = tmp_path/'input.npz'
    np.savez_compressed(path, input=np.zeros((8, 256, 512), np.float32),
        valid_mask=np.zeros((256, 512), bool), target_distribution=np.zeros((256, 512)))
    entry.update(available=True, input_path=str(path), input_sha256=sha(path))
    with pytest.raises(ValueError, match='input and valid_mask only; no GT'):
        evaluation.load_inference_input(entry, {})


def test_unavailable_prediction_cannot_contain_a_path():
    _, entry, decoded, _ = fixture()
    prediction = dict(entry, decoded=decoded)
    assert evaluation.decode_export(entry, prediction, None, {}) == decoded
    prediction['decoded']['segments'] = [[[0, 0], [10, 0]]]
    with pytest.raises(ValueError, match='explicit empty path'):
        evaluation.decode_export(entry, prediction, None, {})


def test_inventory_requires_unavailable_auto_and_all_fixed_rows():
    records = [dict(sample=dict(sample_id=str(i), split='train' if i < 10 else 'holdout')) for i in range(18)]
    entries = {f'{i}:{v}': dict(sample_id=str(i), split='train' if i < 10 else 'holdout',
        variant=v, available=v != 'auto') for i in range(18) for v in evaluation.VARIANTS}
    index = dict(entries=entries, evaluation_keys=list(entries))
    evaluation.validate_inventory(index, records)
    index['evaluation_keys'].remove('17:auto')
    with pytest.raises(ValueError, match='72'):
        evaluation.validate_inventory(index, records)


def test_output_refuses_overwrite_before_reading_sources(tmp_path):
    (tmp_path/'evaluation').mkdir()
    with pytest.raises(FileExistsError):
        evaluation.evaluate(tmp_path)


def test_blind_paths_rejected_but_prohibition_flags_are_allowed():
    evaluation._check_json({'blind_20260925_read': False, 'policy': '20260925 excluded'})
    with pytest.raises(ValueError, match='9/25'):
        evaluation._check_json({'sample': {'image_path': 'D:/source/20260925/frame.png'}})
