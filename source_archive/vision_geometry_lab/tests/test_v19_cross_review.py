"""Independent synthetic profile-to-event contract review; no real data/truth."""
import numpy as np

from vision_geometry_lab.v19_front_candidates import FrontTracker
from vision_geometry_lab.v19_profile_evidence import startup_profile_model, profile_evidence


ROWS = np.array([2200., 2208., 2216.])


def profile(offset=0., contrast=40.):
    return 100.+contrast*(np.arange(-65, 66) >= offset)


def model():
    t = np.arange(26)/5.
    off = np.zeros((26, 3))
    source = dict(normal_offset=off,
        normal_profiles=np.broadcast_to(profile(), (26, 3, 131)).copy(),
        normal_gradient=np.full((26, 3), 20.), normal_peak_ratio=np.full((26, 3), 2.),
        pose_valid=np.ones(26, bool))
    return startup_profile_model(t, source)


def frame(t, offset, peak_ratio=None, contrast=40.):
    offset = np.asarray(offset, float)
    return dict(time_s=t, normal_offset=offset.copy(), normal_residual=offset.copy(),
        normal_gradient=np.full(3, contrast/2.),
        normal_peak_ratio=np.full(3, 2.) if peak_ratio is None else np.asarray(peak_ratio, float),
        normal_profiles=np.array([profile(o, contrast) for o in offset]),
        normal_clipped_fraction=np.zeros(3), pose_valid=True)


def growth(t, onset=(6.1, 6.4, 6.7)):
    return np.array([4. if t >= s else 0. for s in onset])


def run(signal=growth, quality=None, end=9.):
    m = model()
    tracker = FrontTracker(ROWS, m['threshold_px'])
    output = []
    for tick in range(51, int(round(end*10))+1):
        t = tick/10.
        f = frame(t, signal(t), None if quality is None else quality(t))
        evidence = profile_evidence(f, m)
        output.append(tracker.update(t, evidence['profile_residual'], evidence['quality_mask']))
    return output


def candidates(output):
    return [e for r in output for e in r['new_candidates']]


def test_complete_profile_growth_retains_truthful_threshold_crossing_interval():
    e = candidates(run())
    assert len(e) == 1
    assert e[0]['estimated_onset'] == 6.1
    assert e[0]['first_available'] == 7.
    assert e[0]['lastlow_firsthigh_interval'] == [6., 6.1]


def test_simultaneous_step_plus_staggered_quality_recovery_is_not_a_front():
    def ratio(t):
        # Physically simultaneous image step: artificial ordered visibility must
        # not be converted into spatially ordered event-onset timestamps.
        return [2., .9 if 6.1 <= t <= 6.3 else 2., .9 if 6.1 <= t <= 6.6 else 2.]
    result = run(lambda t: np.full(3, 4. if t >= 6.1 else 0.), ratio)
    assert candidates(result) == []


def test_missing_preceding_low_sample_cannot_make_a_false_narrow_interval():
    def ratio(t):
        return np.full(3, .9 if t == 6. else 2.)
    result = run(growth, ratio)
    # Row 0 recovers already high and must observe a new direct stable low before
    # being allowed to initiate a front. No fabricated [6.0, 6.1] interval.
    assert candidates(result) == []


def test_profile_rejection_during_pending_high_discards_its_armed_history():
    def ratio(t):
        return [2., .9 if 6.5 <= t <= 6.6 else 2., 2.]
    assert candidates(run(growth, ratio)) == []


def test_unknown_confirmed_row_cannot_supply_current_chain_support():
    def ratio(t):
        return [.9 if 6.9 <= t <= 7.2 else 2., 2., 2.]
    assert candidates(run(growth, ratio)) == []


def test_plateau_means_do_not_prove_an_edge_or_material_identity():
    m = model()
    f = frame(6., [0., 0., 0.])
    # A lighting ramp plus central detail has no constant side plateaus. Derive
    # the gradient/peak ratio from the same profile rather than inventing a ratio
    # that the frozen V18 sampler could not have produced for a pure linear ramp.
    x = np.arange(-65, 66)
    p = 130.+x+8.*np.tanh(x/2.)
    g = np.gradient(p)
    score = abs(g)*np.exp(-.5*(x/28.)**2)
    assert np.argmax(score) == 65
    f['normal_profiles'] = np.tile(p, (3, 1))
    f['normal_gradient'][:] = g[65]
    f['normal_peak_ratio'][:] = score[65]/score[abs(x)>8].max()
    r = profile_evidence(f, m)
    assert r['quality_mask'].all()
    assert not r['material_identity_confirmed'] and not r['visibility_confirmed']


def test_repeated_profile_edge_can_pass_both_branches_without_becoming_new_layer():
    permanent = run(end=8.)
    pulse = run(lambda t: growth(t) if t < 8.1 else np.zeros(3), end=9.)
    assert pulse[:len(permanent)] == permanent
    e = candidates(pulse)
    assert len(e) == 1
    assert not e[0]['new_layer_confirmed']
    assert not e[0]['mechanical_revolution_confirmed']
    assert e[0]['actuator_command'] is None


def test_valid_profile_does_not_override_explicit_stop_in_tracker():
    m = model()
    tracker = FrontTracker(ROWS, m['threshold_px'])
    output = []
    for tick in range(51, 91):
        t = tick/10.
        r = profile_evidence(frame(t, growth(t)), m)
        output.append(tracker.update(t, r['profile_residual'], r['quality_mask'],
                                    running=False if 6.8 <= t <= 7.1 else None))
    assert candidates(output) == []
