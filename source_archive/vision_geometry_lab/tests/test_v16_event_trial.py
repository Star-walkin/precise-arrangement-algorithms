import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from vision_geometry_lab import v16_event_trial as trial


def rows():
    gaps = [1.2, 1., .8, .6, .4, .2, .15, .1, .1, .2, .3, .4, .5, .6, .7, .8]
    return [dict(frame_index=2 * i, time_s=i / 5.,
        features=dict(usable=True, image_gap_proxy=g, reason='synthetic_test_only'),
        approximate_serial_processing_ms=25., forward_ms=10., measurement_ms=5.)
        for i, g in enumerate(gaps)]


def mp():
    return dict(videos={s: dict(fps=10., selected_indices=[r['frame_index'] for r in rows()],
                               path='synthetic_' + s, frame_count=40, sha256='video')
                       for s in trial.SESSIONS},
                models={a: dict(sha256='model_' + a) for a in trial.ARMS})


def payload():
    return dict(session='left', arm='center_only', protocol_sha256='proto',
        ground_truth_used=False, hardware_access=False, deployment_allowed=False, records=rows())


def test_trial_keeps_every_input_and_separates_confirmation_and_processing_time():
    original = rows()
    saved = copy.deepcopy(original)
    report = trial.replay_measurements(original)
    assert original == saved
    assert len(report['prediction_rows']) == len(original)
    assert report['summary']['usable_fraction'] == 1.
    assert set(report['summary']['candidate_counts']) == {'t0', 't1'}
    assert not report['deployment_allowed']
    t0 = report['candidates'][0]
    assert t0['event_time_s'] == 1.
    assert t0['first_available_at_s'] == 1.4
    assert t0['first_available_processing_lower_bound_s'] >= 1.425
    assert report['t2_status'] == 'not_supplied_independent_full_turn_phase_not_evaluated'


def test_trial_prefix_is_causal_and_later_suffix_does_not_revise_first_event():
    shorter = trial.replay_measurements(rows()[:9])
    longer = trial.replay_measurements(rows())
    assert not [c for c in shorter['candidates'] if c['event'] == 't1']
    for field in ('event_time_s', 'first_available_at_s', 'source_frame', 'confirmation_frame'):
        assert shorter['candidates'][0][field] == longer['candidates'][0][field]


def test_measurement_validation_rejects_frame_drops_duplicates_and_false_timebase():
    valid = payload()
    assert trial.validate_measurement_payload(valid, 'left', 'center_only', mp(), 'proto')
    variants = []
    dropped = copy.deepcopy(valid); dropped['records'].pop(4); variants.append(dropped)
    duplicate = copy.deepcopy(valid); duplicate['records'][4] = duplicate['records'][3]; variants.append(duplicate)
    timebase = copy.deepcopy(valid); timebase['records'][3]['time_s'] += .01; variants.append(timebase)
    for bad in variants:
        try:
            trial.validate_measurement_payload(bad, 'left', 'center_only', mp(), 'proto')
            assert False, 'invalid input was accepted'
        except ValueError:
            pass


def test_measurement_validation_rejects_foreign_provenance_or_nonfinite_usable_gap():
    for key, value in (('arm', 'joint'), ('protocol_sha256', 'different'),
                       ('ground_truth_used', True), ('hardware_access', True)):
        bad = dict(payload(), **{key: value})
        try:
            trial.validate_measurement_payload(bad, 'left', 'center_only', mp(), 'proto')
            assert False
        except ValueError:
            pass
    bad = payload(); bad['records'][2]['features']['image_gap_proxy'] = None
    try:
        trial.validate_measurement_payload(bad, 'left', 'center_only', mp(), 'proto')
        assert False
    except ValueError:
        pass


def test_missing_measurements_are_unavailable_not_zero_gap():
    data = rows()
    for r in data:
        r['features'] = dict(usable=False, image_gap_proxy=None, reason='not_visible')
    report = trial.replay_measurements(data)
    assert report['summary']['usable_frames'] == 0
    assert report['summary']['measurement_reason_counts'] == {'not_visible': len(data)}
    assert not report['candidates']
    assert all(r['input']['gap_proxy'] is None for r in report['prediction_rows'])


def test_missing_four_predictions_prevents_truth_read():
    with tempfile.TemporaryDirectory() as temp:
        with patch.object(trial, 'verify', return_value=({}, {})), \
             patch.object(trial, 'load_all_predictions', side_effect=FileNotFoundError('fourth result missing')), \
             patch.object(trial, 'read') as read:
            try:
                trial.score(Path(temp), Path(temp) / 'forbidden_truth.json')
                assert False
            except FileNotFoundError:
                pass
            read.assert_not_called()


def test_freeze_binds_measurement_protocol_and_event_dependency_sources():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp); measure = root / 'measure'; measure.mkdir()
        (measure / 'protocol.json').write_text('{}', encoding='utf-8')
        destination = root / 'events'
        with patch('vision_geometry_lab.v16_video_measurement.verify', return_value=mp()):
            p = trial.freeze(destination, measure)
            assert p['measurement_protocol']['sha256'] == trial.sha(measure / 'protocol.json')
            assert set(p['source_snapshot']) == {str(f) for f in trial.SOURCE_FILES}
            trial.verify(destination)
            (measure / 'protocol.json').write_text('{"changed": true}', encoding='utf-8')
            try:
                trial.verify(destination)
                assert False
            except ValueError:
                pass


def test_no_overwrite_and_no_nonfinite_json_leak():
    with tempfile.TemporaryDirectory() as temp:
        p = Path(temp) / 'result.json'
        trial.write_new(p, {'unknown': float('nan')})
        assert json.loads(p.read_text()) == {'unknown': None}
        try:
            trial.write_new(p, {'unknown': 0})
            assert False
        except FileExistsError:
            pass


def test_protocol_preserves_first_cycle_censoring_warning_without_threshold_tuning():
    assert trial.PROTOCOL['rearm_clear_gap_widths'] == 3.
    assert any('first-cycle' in note and 'censored' in note for note in trial.LIMITATIONS)
    assert any('not blind' in note for note in trial.LIMITATIONS)
    assert any('t2 is unavailable' in note for note in trial.LIMITATIONS)
