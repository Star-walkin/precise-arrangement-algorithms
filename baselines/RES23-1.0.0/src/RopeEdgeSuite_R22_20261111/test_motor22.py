import unittest,numpy as np
from rotation22 import estimate_period
from motor22 import target_time

class MotorTests(unittest.TestCase):
    def test_live_prefix_does_not_confirm_before_search_window_finishes(self):
        from motor22 import analyze
        rng=np.random.default_rng(42);features=np.tile(rng.normal(size=(20,4,96)).astype('float32'),(8,1,1))
        rows=[dict(time_s=i/10,N=8-.05*i if i<100 else .6,available=True,boundary_occluded=False,
            axis_slope=.2 if i<105 else .22,current_coil_side_axis=True) for i in range(160)]
        motion=dict(frames=[dict(evidence_mode='current',motion='approaching',state='near' if 100<=i<=140 else 'approaching') for i in range(160)])
        overlap=dict(candidates=[dict(start_s=10.,confirmed_s=10.4)])
        full=analyze(rows,overlap,motion,features,10)
        self.assertTrue(full['events'][0]['recurrence_supported'])
        short=analyze(rows[:124],overlap,dict(frames=motion['frames'][:124]),features[:124],10)
        self.assertEqual(short['frames'],full['frames'][:124])

    def test_new_layer_cue_is_required_before_displaying_motor_target(self):
        from motor22 import analyze
        rng=np.random.default_rng(42);features=np.tile(rng.normal(size=(20,4,96)).astype('float32'),(8,1,1))
        rows=[dict(time_s=i/10,N=8-.05*i if i<100 else .6,available=True,boundary_occluded=False,
            axis_slope=.2,current_coil_side_axis=True) for i in range(160)]
        motion=dict(frames=[dict(evidence_mode='current',motion='approaching',state='near' if i>=100 else 'approaching') for i in range(160)])
        out=analyze(rows,dict(candidates=[dict(start_s=10.,confirmed_s=10.4)]),motion,features,10)
        self.assertFalse(out['events'][0]['recurrence_supported'])
        self.assertTrue(all(x.get('target') is None for x in out['frames']))

    def test_repeated_unique_texture_returns_fundamental_not_second_harmonic(self):
        rng=np.random.default_rng(42);pattern=rng.normal(size=(20,4,96)).astype('float32')
        a=np.tile(pattern,(6,1,1))
        p=estimate_period(a,10)
        self.assertTrue(p['available'])
        self.assertAlmostEqual(p['period_s'],2.,delta=.11)

    def test_stationary_texture_is_not_rotation(self):
        rng=np.random.default_rng(42)
        a=np.repeat(rng.normal(size=(1,4,96)),120,axis=0)
        self.assertFalse(estimate_period(a,10)['available'])

    def test_nonperiodic_noise_does_not_supply_a_turn(self):
        self.assertFalse(estimate_period(np.random.default_rng(6).normal(size=(120,4,96)),10)['available'])

    def test_short_clip_cannot_prove_period(self):
        self.assertFalse(estimate_period(np.zeros((9,4,96)),10)['available'])

    def test_missing_visible_regions_cannot_prove_period(self):
        self.assertFalse(estimate_period(np.zeros((120,0,96)),10)['available'])

    def test_motor_time_is_one_turn_after_start_not_departure(self):
        p=dict(available=True,period_s=2.,period_interval_s=[1.9,2.1])
        out=target_time(dict(start_s=10.,confirmed_s=10.4),p)
        self.assertIsNotNone(out)
        self.assertAlmostEqual(out['time_s'],12.)
        self.assertEqual(out['interval_s'],[11.9,12.5])
        self.assertFalse(out['physical_layer_change_confirmed'])

    def test_missing_start_or_period_no_motor_target(self):
        self.assertIsNone(target_time(None,dict(available=True,period_s=2.)))
        self.assertIsNone(target_time(dict(start_s=10.,confirmed_s=10.4),dict(available=False)))

if __name__=='__main__':unittest.main()
