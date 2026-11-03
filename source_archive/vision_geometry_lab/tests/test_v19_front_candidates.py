import numpy as np
import pytest

from vision_geometry_lab.v19_front_candidates import FrontTracker


def drive(function, stop=4., rows=3, tracker=None, quality=None, running=None):
    tracker = tracker or FrontTracker(np.arange(rows)*8., np.full(rows, 3.))
    results = []
    for tick in range(int(round(stop*10))+1):
        t = tick/10.
        results.append(tracker.update(t, function(t),
            np.ones(rows, bool) if quality is None else quality(t),
            running=None if running is None else running(t)))
    return tracker, results


def growth(t, onsets=(1., 1.3, 1.6)):
    return np.array([4. if t >= onset else 0. for onset in onsets])


def events(results):
    return [item for row in results for item in row['new_candidates']]


@pytest.mark.parametrize('level', [0., 2., 4.])
def test_static_never_arms_or_emits_from_elapsed_time_alone(level):
    _, r = drive(lambda t: np.full(3, level), stop=10.)
    assert events(r) == []


def test_ordered_growth_emits_once_with_distinct_confirmation_time():
    _, r = drive(growth)
    e = events(r)
    assert len(e) == 1
    assert e[0]['estimated_onset'] == 1.
    assert e[0]['first_available'] == 1.9
    assert e[0]['delay'] == pytest.approx(.9)
    assert e[0]['lastlow_firsthigh_interval'] == [.9, 1.]
    assert [row['index'] for row in e[0]['rows']] == [0, 1, 2]
    assert not e[0]['new_layer_confirmed'] and e[0]['actuator_command'] is None


def test_spikes_and_global_step_are_not_ordered_fronts():
    _, spike = drive(lambda t: np.array([4. if abs(t-1.-j*.3)<.01 else 0. for j in range(3)]))
    _, step = drive(lambda t: np.full(3, 4. if t >= 1. else 0.))
    assert events(spike) == [] and events(step) == []


def test_reverse_spatial_order_is_not_accepted():
    _, r = drive(lambda t: growth(t, (1.6, 1.3, 1.)))
    assert events(r) == []


def test_pending_middle_band_cannot_age_into_confirmed_high():
    def signal(t):
        return np.array([0. if t < onset else 4. if t < onset+.1 else 2.
                         for onset in (1., 1.3, 1.6)])
    _, r = drive(signal)
    assert events(r) == []
    assert all(row['phase'] != 'confirmed' for row in r[-1]['row_states'])


def test_confirmed_middle_band_can_retain_existing_change():
    def signal(t):
        return np.array([0. if t < onset else 4. if t <= onset+.3+1e-9 else 2.
                         for onset in (1., 1.3, 1.6)])
    _, r = drive(signal)
    assert len(events(r)) == 1


def test_pending_unknown_erases_arming_and_cannot_recover_already_high():
    def q(t):
        return np.array([True, not 1.4 <= t <= 1.5, True])
    _, r = drive(growth, quality=q)
    assert events(r) == []


def test_no_event_uses_current_unknown_row_even_if_previously_confirmed():
    def q(t):
        return np.array([not 1.9 <= t <= 2.1, True, True])
    _, r = drive(growth, quality=q)
    assert events(r) == []


def test_frame_gap_resets_and_high_recovery_is_not_new_onset():
    tracker = FrontTracker(np.arange(3)*8., 3.)
    out = []
    for tick in list(range(16))+list(range(21, 41)):
        t = tick/10.
        out.append(tracker.update(t, growth(t), np.ones(3, bool)))
    assert events(out) == []
    assert out[16]['reason'] == 'frame_gap_reset'


def test_maximum_gap_boundary_and_minimum_point_counts():
    tracker = FrontTracker([0., 8., 16.], 3.)
    tracker.update(0., [0., 0., 0.], [1, 1, 1])
    tracker.update(.2, [0., 0., 0.], [1, 1, 1])
    r = tracker.update(.4, [0., 0., 0.], [1, 1, 1])
    assert all(row['armed'] for row in r['row_states'])
    r = tracker.update(.7, [4., 0., 0.], [1, 1, 1])
    assert r['reason'] != 'frame_gap_reset'
    assert r['row_states'][0]['phase'] == 'pending'


def test_explicit_stop_discards_pending_and_cannot_mature_clock():
    _, r = drive(growth, running=lambda t: False if 1.7 <= t <= 2.1 else None)
    assert events(r) == []
    assert r[17]['state'] == 'stopped_reset'
    assert r[-1]['running_state'] == 'unknown'


def test_rewind_and_nonfinite_timestamp_reset():
    tracker, _ = drive(growth, stop=1.8)
    r = tracker.update(1.2, [4., 4., 0.], [1, 1, 1])
    assert r['reason'] == 'nonincreasing_time_reset' and not r['new_candidates']
    r = tracker.update(np.nan, [4., 4., 4.], [1, 1, 1])
    assert r['time_s'] is None and r['reason'] == 'nonfinite_time'


def test_unsupported_original_row_cannot_be_skipped():
    tracker = FrontTracker(np.arange(4)*8., [3., np.nan, 3., 3.])
    _, r = drive(lambda t: growth(t, (1., 1.1, 1.3, 1.6)), rows=4, tracker=tracker)
    assert events(r) == []


def test_chain_expiry_does_not_reinvent_static_old_high():
    _, r = drive(lambda t: growth(t, (1., 6.9, 7.2)), stop=10.)
    assert events(r) == []


def test_too_fast_chain_cannot_later_trigger_only_by_waiting():
    _, r = drive(lambda t: growth(t, (1., 1.1, 1.2)), stop=4.)
    # Last confirmation at 1.5 expires before earliest onset is .8s old.
    assert events(r) == []


def test_episode_lock_prevents_multiple_overlapping_neighbourhood_reports():
    _, r = drive(lambda t: growth(t, (1., 1.3, 1.6, 1.9, 2.2, 2.5)), rows=6, stop=6.)
    assert len(events(r)) == 1


def test_far_stable_low_rows_do_not_release_reported_front_episode():
    _, r = drive(lambda t: np.r_[growth(t), [0., 0., 0.]], rows=6, stop=5.)
    assert len(events(r)) == 1
    assert r[-1]['episode_locked']


def test_fresh_stable_low_of_reported_rows_releases_episode():
    def signal(t):
        if 2.5 <= t < 3.2:
            return np.zeros(3)
        return growth(t, (3.5, 3.8, 4.1)) if t >= 3.2 else growth(t)
    _, r = drive(signal, stop=5.)
    e = events(r)
    assert len(e) == 2
    assert e[0]['first_available'] < e[1]['estimated_onset']
    assert any(row['state'] == 'episode_rearmed' for row in r)


def test_prefix_invariance_for_first_available_candidate():
    _, full = drive(growth, stop=4.)
    for stop in (1.2, 1.8, 1.9, 2., 3.5):
        _, prefix = drive(growth, stop=stop)
        assert prefix == full[:len(prefix)]


def test_startup_scaled_periodic_wave_negative_example():
    # The scale is computed from the same 0..5s startup samples, not tuned to an
    # event. A bounded repeating wave already represented there stays below 3σ.
    times = np.arange(51)/10.
    startup = np.sin(times[:, None]*2*np.pi/2.+np.arange(3)[None, :]*.3)*4.
    ref = np.median(startup, axis=0)
    mad = np.median(abs(startup-ref), axis=0)
    threshold = np.maximum(3., 3.*1.4826*mad)
    tracker = FrontTracker(np.arange(3)*8., threshold)
    out = []
    for tick in range(51, 201):
        t = tick/10.
        d = np.sin(t*2*np.pi/2.+np.arange(3)*.3)*4.-ref
        out.append(tracker.update(t, d, np.ones(3, bool)))
    assert events(out) == []


def test_long_period_false_front_challenge_is_explicitly_not_resolved():
    # A travelling positive pulse can be identical to real retained growth up
    # to the decision time, then return much later. This intentional candidate
    # exposes finite-prefix ambiguity; do not tune it out using future samples.
    def long_wave(t):
        return growth(t) if t < 4. else np.zeros(3)
    _, growth_result = drive(growth, stop=3.)
    _, wave_result = drive(long_wave, stop=5.)
    assert wave_result[:31] == growth_result
    e = events(wave_result)
    assert len(e) == 1
    assert 'long_period_wave' in e[0]['ambiguity']
    assert not e[0]['new_layer_confirmed']


def test_nonfinite_quality_and_displacement_are_unknown():
    tracker = FrontTracker([0., 8., 16.], 3.)
    r = tracker.update(0., [0., np.nan, 0.], [1., 1., np.nan])
    assert [row['quality'] for row in r['row_states']] == [True, False, False]


def test_input_validation_and_threshold_floor():
    with pytest.raises(ValueError):
        FrontTracker([0., 0.], 3.)
    with pytest.raises(ValueError):
        FrontTracker([0., 8.], [3., np.inf])
    tracker = FrontTracker([0., 8., 16.], 2.)
    np.testing.assert_array_equal(tracker.high, [3., 3., 3.])
    with pytest.raises(ValueError):
        tracker.update(0., [0., 0., 0.], [1, 1, 1], running='unknown')
