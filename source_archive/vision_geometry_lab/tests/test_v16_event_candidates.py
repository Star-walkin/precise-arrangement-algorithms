import copy
import math

from vision_geometry_lab.v16_event_candidates import EventCandidateTracker, score_candidates


def frame(time_s, gap, usable=True, **extra):
    return dict(time_s=round(time_s, 6), source_frame=round(time_s * 10),
                usable=usable, gap_proxy=gap, processing_ms=25., **extra)


def trajectory():
    # Descent to contact, followed by one monotonic departure.
    return [frame(i * .2, 1.2 - i * .2) for i in range(6)] + [
        frame(1.2, .15), frame(1.4, .1), frame(1.6, .1),
        frame(1.8, .2), frame(2., .3), frame(2.2, .4), frame(2.4, .5),
        frame(2.6, .6), frame(2.8, .7), frame(3., .8)]


def run(rows, period=None):
    tracker = EventCandidateTracker()
    emitted = []
    for row in rows:
        p = period(row) if period else None
        emitted.extend(tracker.update(row, p))
    return tracker, emitted


def test_contact_requires_previous_approach_and_three_consecutive_frames():
    _, static = run([frame(i * .2, .1) for i in range(20)])
    assert not static
    tracker, events = run(trajectory())
    t0 = next(e for e in events if e['event'] == 't0')
    assert t0['event_time_s'] == 1.
    assert t0['first_available_at_s'] == 1.4
    assert t0['candidate_only'] and not t0['physical_layer_confirmed']
    assert not tracker.finalize()['deployment_permitted']


def test_missing_frame_breaks_consecutive_contact():
    rows = trajectory()[:6] + [frame(1.2, .1, False), frame(1.4, .1), frame(1.6, .1)]
    _, events = run(rows)
    assert not [e for e in events if e['event'] == 't0']


def test_reversal_is_period_independent_but_not_new_layer_confirmation():
    tracker, events = run(trajectory())
    t1 = next(e for e in events if e['event'] == 't1')
    assert 1.4 <= t1['event_time_s'] <= 1.6
    assert t1['first_available_at_s'] >= t1['event_time_s'] + .6
    assert t1['evidence']['departure_span_s'] >= .6 - 1e-9
    assert not t1['physical_layer_confirmed']
    assert not [e for e in events if e['event'] == 't2']
    assert tracker.finalize()['t2_reason'] != 'ok'


def test_reversal_does_not_require_contact_event():
    _, events = run([dict(r, gap_proxy=r['gap_proxy'] + .7) for r in trajectory()])
    assert not [e for e in events if e['event'] == 't0']
    assert len([e for e in events if e['event'] == 't1']) == 1


def test_no_t1_is_invented_from_contact_and_elapsed_time():
    rows = trajectory()[:8] + [frame(1.6 + i * .2, .1) for i in range(40)]
    _, events = run(rows)
    assert len([e for e in events if e['event'] == 't0']) == 1
    assert not [e for e in events if e['event'] in ('t1', 't2')]


def test_future_suffix_cannot_change_frozen_first_reports():
    tracker, events = run(trajectory())
    frozen = copy.deepcopy(events)
    for i in range(20):
        tracker.update(frame(3.2 + i * .2, 1. + i * .05))
    assert tracker.finalize()['candidates'] == frozen
    prefix, prefix_events = run(trajectory()[:9])
    assert not [e for e in prefix_events if e['event'] == 't1']


def test_large_frame_gap_resets_approach_evidence():
    tracker, _ = run(trajectory()[:5])
    for i in range(3):
        assert not tracker.update(frame(2. + .2 * i, .1))
    assert any(r['reason'] == 'frame_gap_exceeded' for r in tracker.resets)


def test_geometry_missing_over_limit_resets_without_crossing_gap():
    rows = trajectory()[:5] + [frame(1. + .2 * i, .1, False) for i in range(4)]
    rows += [frame(1.8 + .2 * i, .1) for i in range(4)]
    tracker, events = run(rows)
    assert not events
    assert any(r['reason'] == 'usable_geometry_gap_exceeded' for r in tracker.resets)


def test_stationary_and_reverse_time_reset_without_false_contact():
    for special in (frame(1., .1, stationary=True), frame(.6, .1)):
        tracker, _ = run(trajectory()[:5])
        assert not tracker.update(special)
        for i in range(4):
            assert not tracker.update(frame(1.2 + .2 * i, .1))
        assert tracker.resets


def test_invalid_gap_or_timestamp_is_not_a_zero_contact():
    tracker, _ = run(trajectory()[:5])
    assert not tracker.update(frame(1., math.nan))
    assert not tracker.update(dict(time_s=math.nan, source_frame=11, usable=True, gap_proxy=.1))
    assert not tracker.update(frame(1.2, .1))


def test_events_need_clear_then_new_approach_to_rearm():
    tracker, events = run(trajectory())
    first = copy.deepcopy(events)
    # A second valley at the same near-edge position is not a second cycle.
    for r in trajectory():
        tracker.update(dict(r, time_s=r['time_s'] + 3.2, source_frame=r['source_frame'] + 32))
    assert tracker.finalize()['candidates'] == first
    for i, g in enumerate([1., 2., 3.2, 3.5, 3.3, 3., 2.5, 2., 1.5, 1., .2, .1, .1]):
        tracker.update(frame(6.4 + .2 * i, g))
    contacts = [e for e in tracker.candidates if e['event'] == 't0']
    assert len(contacts) == 2 and contacts[-1]['cycle_id'] == 1


def full_period(row, **extra):
    return dict(time_s=row['time_s'], available_at_s=row['time_s'], period_s=2.,
                valid=True, full_turn_confirmed=True, ambiguous=False,
                source='independent_test_reference', **extra)


def test_t2_requires_independent_unambiguous_full_turn_period():
    rows = trajectory() + [frame(3.2 + .2 * i, .9 + .1 * i) for i in range(5)]
    tracker, events = run(rows, full_period)
    t1 = next(e for e in events if e['event'] == 't1')
    t2 = next(e for e in events if e['event'] == 't2')
    assert abs(t2['event_time_s'] - t1['event_time_s'] - 2.) < 1e-8
    assert t2['first_available_at_s'] >= t2['event_time_s']
    assert not t2['device_advisory']
    for key, value in (('ambiguous', True), ('full_turn_confirmed', False),
                       ('available_at_s', 100.)):
        _, blocked = run(rows, lambda r: dict(full_period(r), **{key: value}))
        assert not [e for e in blocked if e['event'] == 't2']


def test_missing_period_is_not_filled_with_historical_constant():
    rows = trajectory() + [frame(3.2 + .2 * i, .9 + .1 * i) for i in range(5)]
    _, events = run(rows, lambda r: None if r['time_s'] == 2.8 else full_period(r))
    assert not [e for e in events if e['event'] == 't2']


def test_manual_or_occlusion_fields_are_not_an_inference_input():
    _, reference = run(trajectory())
    modified = [dict(r, manual_hammer_strike_s=r['time_s'], occlusion=True,
                     new_layer_start_s=-100.) for r in trajectory()]
    _, result = run(modified)
    assert reference == result


def test_scoring_uses_first_candidate_not_best_matching_later_candidate():
    truth = dict(previous_layer_last_revolution_start_s=1., new_layer_start_s=2.,
                 new_layer_first_revolution_end_s=4.)
    rows = [dict(event='t1', event_time_s=1., first_available_at_s=1.8),
            dict(event='t1', event_time_s=2., first_available_at_s=2.8)]
    result = score_candidates(rows, truth)
    assert result['events']['t1']['signed_error_s'] == -1.
    assert result['events']['t1']['extra_candidate_count'] == 1
    assert not result['events']['t1']['within_research_tolerance']
    assert result['events']['t2']['missed']
    assert not result['deployment_permitted']


def test_expired_contact_does_not_generate_late_t1_or_t2():
    rows = trajectory()[:8] + [frame(1.6 + i * .2, .1) for i in range(70)]
    rows += [frame(15.6 + i * .2, .2 + i * .1) for i in range(12)]
    tracker, events = run(rows, full_period)
    assert tracker.cycle_expired
    assert len([e for e in events if e['event'] == 't0']) == 1
    assert not [e for e in events if e['event'] in ('t1', 't2')]


def test_inputs_and_returned_copies_cannot_rewrite_tracker_evidence():
    rows = trajectory()
    saved = copy.deepcopy(rows)
    tracker, events = run(rows)
    assert rows == saved
    events[0]['event_time_s'] = -100.
    events[0]['evidence']['gap_proxy'] = -100.
    snapshot = tracker.snapshot()
    assert snapshot['candidates'][0]['event_time_s'] >= 0.
    snapshot['candidates'].clear()
    snapshot['protocol']['contact_maximum_gap_widths'] = 100.
    assert tracker.snapshot()['candidates']
    assert tracker.snapshot()['protocol']['contact_maximum_gap_widths'] == .25
