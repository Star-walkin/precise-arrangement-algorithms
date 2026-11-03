import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from vision_geometry_lab import v17_event_evaluation as ev


def rows():
    gaps = [1.2, 1., .8, .6, .4, .2, .15, .1, .1, .2, .3, .4, .5, .6, .7, .8]
    return [dict(frame_index=i * 2, time_s=i / 5., processing_ms=20.,
        features=dict(usable=True, image_gap_proxy=g, reason='synthetic',
            boundary_state='direct', oldest_observed_at_s=i / 5.,
            boundary_max_age_s=0., boundary_sigma_px=.5,
            registration_valid=False, overlap_valid=False,
            boundary_source_counts=dict(direct=20, predicted=0, unknown=0)))
        for i, g in enumerate(gaps)]


def prediction(data=None, session='left', mode='direct_grid'):
    data = rows() if data is None else data
    return ev.build_prediction(data, session=session, mode=mode,
                               expected_indices=[r['frame_index'] for r in data], fps=10.)


def predicted_row(time_s=1., age=.2):
    r = rows()[5]
    r['time_s'] = time_s
    r['features'].update(boundary_state='predicted', oldest_observed_at_s=time_s-age,
                         boundary_max_age_s=age, registration_valid=True, overlap_valid=True,
                         boundary_source_counts=dict(direct=5, predicted=15, unknown=0))
    return r


def test_direct_first_frame_does_not_require_history_validation():
    record, meta = ev.gate_input(rows()[0], 'direct_grid')
    assert record['usable'] and meta['boundary_state'] == 'direct'
    assert not meta['path_visibility_assessed']


def test_predicted_requires_current_registration_and_overlap():
    r = predicted_row()
    assert ev.gate_input(r, 'temporal_guarded')[0]['usable']
    for key in ('registration_valid', 'overlap_valid'):
        bad = copy.deepcopy(r); bad['features'][key] = False
        assert not ev.gate_input(bad, 'temporal_guarded')[0]['usable']
    assert not ev.gate_input(r, 'direct_grid')[0]['usable']


def test_boundary_age_future_and_self_refresh_are_rejected():
    for r in (predicted_row(4., 2.001), predicted_row(4., 0.), predicted_row(4., -.1)):
        assert not ev.gate_input(r, 'temporal_guarded')[0]['usable']
    r = predicted_row(4., .5)
    r['features']['oldest_observed_at_s'] = 2.
    assert not ev.gate_input(r, 'temporal_guarded')[0]['usable']
    r = predicted_row(4., .5)
    r['features']['boundary_observed_at_s'] = [3.5, 4.1]
    assert not ev.gate_input(r, 'temporal_guarded')[0]['usable']


def test_unknown_visibility_rejection_or_sigma_missing_cannot_be_usable():
    for update in ({'boundary_state':'unknown'}, {'visibility_rejected':True},
                   {'boundary_sigma_px':None}, {'boundary_sigma_px':-1.}):
        r = rows()[5]
        r['features'].update(update)
        assert not ev.gate_input(r, 'temporal_guarded')[0]['usable']


def test_no_future_suffix_changes_first_event_and_period_none_gives_no_t2():
    partial, full = prediction(rows()[:9]), prediction()
    assert not [c for c in partial['candidates'] if c['event'] == 't1']
    assert partial['candidates'][0]['event_time_s'] == full['candidates'][0]['event_time_s']
    assert partial['candidates'][0]['first_available_at_s'] == full['candidates'][0]['first_available_at_s']
    assert not [c for c in full['candidates'] if c['event'] == 't2']
    assert full['measured_physical_nodes'] == 0


def test_history_involved_event_is_labelled_predicted_not_measured():
    data = rows()
    for r in data[1:]:
        r['features'].update(boundary_state='predicted', oldest_observed_at_s=r['time_s']-.2,
                             boundary_max_age_s=.2, registration_valid=True, overlap_valid=True)
    result = prediction(data, mode='temporal_guarded')
    assert result['candidates']
    assert all(c['contains_predicted_boundary_support'] for c in result['candidates'])
    assert all(not c['measured_physical_node'] for c in result['candidates'])
    assert result['summary']['accepted_boundary_state_counts']['predicted'] == len(data)-1


def test_unknown_frames_do_not_count_as_contact_or_filled_zero():
    data = rows()
    for r in data[5:10]:
        r['features'].update(boundary_state='unknown', usable=True, image_gap_proxy=0.)
    result = prediction(data)
    assert not result['candidates']
    assert result['summary']['accepted_boundary_state_counts']['unknown'] == 0
    assert result['summary']['max_consecutive_unknown_frames'] == 5


def test_inputs_unchanged_and_frame_schedule_complete():
    data = rows(); saved = copy.deepcopy(data)
    prediction(data)
    assert data == saved
    for bad in (data[:-1], list(reversed(data)), data[:4]+data[5:]):
        try:
            ev.build_prediction(bad, session='left', mode='direct_grid',
                expected_indices=[r['frame_index'] for r in data], fps=10.)
            assert False
        except ValueError:
            pass


def test_scoring_requires_four_unique_modes_before_truth_read():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        paths = []
        for s in ev.SESSIONS:
            for mode in ev.MODES:
                path = root / (s+'_'+mode+'.json')
                ev.write_prediction(path, prediction(session=s, mode=mode))
                paths.append(path)
        truth_path = root / 'truth.json'
        original = Path.read_text
        opened = []
        def spy(path, *args, **kwargs):
            opened.append(path)
            return original(path, *args, **kwargs)
        with patch.object(Path, 'read_text', new=spy):
            for bad in (paths[:3], paths[:3] + paths[:1]):
                try:
                    ev.score_prediction_files(bad, truth_path)
                    assert False
                except ValueError:
                    pass
        assert truth_path not in opened


def test_post_score_preserves_t2_not_evaluated_and_all_are_research_candidates():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp); paths = []
        for session in ev.SESSIONS:
            for mode in ev.MODES:
                p = root / (session+'_'+mode+'.json')
                ev.write_prediction(p, prediction(session=session, mode=mode))
                paths.append(p)
        truth = root/'truth.json'
        manual = dict(previous_layer_last_revolution_start_s=1., new_layer_start_s=1.4,
                      new_layer_first_revolution_end_s=3.4)
        truth.write_text(json.dumps({'sessions':{s:{'manual_review':manual} for s in ev.SESSIONS.values()}}))
        result = ev.score_prediction_files(paths, truth)
        assert result['truth_read_after_four_predictions']
        assert result['measured_physical_nodes'] == 0 and not result['deployment_allowed']
        assert all(c['event_scores']['events']['t2']['evaluation_status'].startswith('not_evaluated')
                   for c in result['cases'])


def test_prediction_validation_rejects_backdating_and_physical_claims():
    result = prediction()
    for mutate in ('time', 'physical'):
        bad = copy.deepcopy(result)
        target = next(r for r in bad['prediction_rows'] if r['new_candidates'])
        if mutate == 'time':
            target['new_candidates'][0]['first_available_at_s'] -= .2
            bad['candidates'][0]['first_available_at_s'] -= .2
        else:
            bad['measured_physical_nodes'] = 1
        try:
            ev.validate_prediction(bad)
            assert False
        except ValueError:
            pass


def test_event_protocol_binds_core_hash_and_event_dependencies_without_results():
    with tempfile.TemporaryDirectory() as temp:
        base = Path(temp)
        (base/'protocol.json').write_text('{}', encoding='utf-8')
        core = dict(videos={s:dict(selected_indices=[0,2], fps=10.) for s in ev.SESSIONS})
        with patch('vision_geometry_lab.v17_boundary_trial.verify', return_value=core):
            p = ev.freeze_event_protocol(base)
            assert set(p['source_snapshot']) == {str(f) for f in ev.EVENT_SOURCES}
            assert p['measurement_protocol_sha256'] == ev.file_sha256(base/'protocol.json')
            ev.verify_event_protocol(base)
            (base/'protocol.json').write_text('{"changed":1}', encoding='utf-8')
            try:
                ev.verify_event_protocol(base)
                assert False
            except ValueError:
                pass


def test_readonly_generation_does_not_assume_path_probability_proves_visibility():
    data = rows()
    for r in data:
        r['features']['path_identity_confirmed'] = True
        r['features']['path_visibility_assessed'] = True
    result = prediction(data)
    assert not result['path_identity_confirmed']
    assert not result['path_visibility_assessed']
    assert all(not c['path_identity_confirmed'] and not c['path_visibility_assessed']
               for c in result['candidates'])
