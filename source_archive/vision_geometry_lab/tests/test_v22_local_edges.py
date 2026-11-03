import tempfile
import unittest
from pathlib import Path

import numpy as np

from vision_geometry_lab import v22_local_edges as v22


def stripe(left=-10.,right=10.,kind='bright'):
    x=np.arange(-65,66,dtype=float)
    inside=(x>=left)&(x<=right)
    p=np.full(131,50. if kind=='bright' else 140.)
    p[inside]=140. if kind=='bright' else 50.
    return p


def observation(profile):
    return v22.observe_profile(profile,True,0.)


class ProfileTests(unittest.TestCase):
    def test_constant_black_white_no_edges(self):
        for value in (0.,80.,255.):
            result=observation(np.full(131,value))
            self.assertTrue(result['valid']);self.assertEqual(result['edges'],[])

    def test_linear_ramp_does_not_manufacture_pair(self):
        r=observation(np.linspace(0.,255.,131))
        self.assertEqual(len(r['edges']),0);self.assertEqual(r['pairs'],[])

    def test_single_step_preserves_one_edge_not_band(self):
        p=np.where(np.arange(131)<65,40.,140.)
        r=observation(p)
        self.assertEqual(len(r['edges']),1);self.assertEqual(r['edges'][0]['polarity'],1)
        self.assertAlmostEqual(r['edges'][0]['offset_px'],-.5,places=5)
        self.assertEqual(r['pairs'],[])

    def test_bright_and_dark_pair_polarities_and_raw_contrast(self):
        for kind,signs in (('bright',[1,-1]),('dark',[-1,1])):
            r=observation(stripe(kind=kind))
            self.assertEqual([e['polarity'] for e in r['edges']],signs)
            self.assertEqual(len(r['pairs']),1)
            pair=r['pairs'][0]
            self.assertTrue(pair['band_candidate']);self.assertEqual(pair['polarity_kind'],kind)
            self.assertAlmostEqual(pair['width_px'],21.,places=5)
            self.assertAlmostEqual(pair['raw_two_sided_contrast_gray'],90.)

    def test_multiple_peaks_retained_not_one_winner(self):
        p=stripe(-20,-10);p[(np.arange(-65,66)>=10)&(np.arange(-65,66)<=20)]=140.
        r=observation(p)
        self.assertEqual(len(r['edges']),4)
        self.assertEqual([p['polarity_kind'] for p in r['pairs']],['bright','dark','bright'])

    def test_out_of_range_pair_does_not_remove_edge_evidence(self):
        r=observation(stripe(-26,26))
        self.assertEqual(len(r['edges']),2);self.assertEqual(r['pairs'],[])
        self.assertEqual(r['opposite_pairs_outside_analysis_range'],1)

    def test_brightness_offset_does_not_change_edges_or_pair_width(self):
        a=observation(stripe());b=observation(stripe()+30.)
        np.testing.assert_allclose([e['offset_px'] for e in a['edges']],[e['offset_px'] for e in b['edges']],atol=1e-10)
        self.assertAlmostEqual(a['pairs'][0]['width_px'],b['pairs'][0]['width_px'])

    def test_missing_pose_nan_and_saturation_fail_closed(self):
        p=stripe();p[20]=np.nan
        cases=[v22.observe_profile(stripe(),False,0.),v22.observe_profile(p,True,0.),
               v22.observe_profile(stripe(),True,.3),v22.observe_profile(stripe(),True,np.nan)]
        for r in cases:self.assertFalse(r['valid']);self.assertEqual(r['edges'],[])

    def test_profile_size_and_gray_range_guard(self):
        with self.assertRaises(ValueError):observation(np.ones(130))
        self.assertFalse(observation(np.full(131,256.))['valid'])

    def test_outermost_filter_artifact_not_an_edge(self):
        p=np.full(131,40.);p[:2]=200.
        self.assertEqual(observation(p)['edges'],[])


class TrackerTests(unittest.TestCase):
    def test_static_stripe_persists_but_is_not_physical_birth(self):
        tracker=v22.RowTracker();out=None
        for i in range(10):out=tracker.update(i/10.,observation(stripe()))
        self.assertTrue(out['pairs'][0]['persistent'])
        self.assertEqual(out['pairs'][0]['first_observed_s'],0.)
        self.assertEqual(out['pairs'][0]['first_available_s'],.4)
        self.assertEqual(out['pairs'][0]['relative_separation_change_px'],0.)
        self.assertFalse(out['pairs'][0]['first_observed_is_physical_birth'])

    def test_static_wait_without_enough_observations_does_not_confirm(self):
        tracker=v22.RowTracker();r=observation(stripe())
        tracker.update(0.,r);out=tracker.update(.2,r);out=tracker.update(.4,r)
        self.assertFalse(out['pairs'][0]['persistent']);self.assertIsNone(out['pairs'][0]['first_available_s'])

    def test_first_available_is_actual_fifth_observation_not_theoretical_point4(self):
        tracker=v22.RowTracker();r=observation(stripe())
        for t in (0.,.2,.4,.6,.8):out=tracker.update(t,r)
        self.assertTrue(out['pairs'][0]['persistent']);self.assertEqual(out['pairs'][0]['first_available_s'],.8)

    def test_flash_does_not_persist(self):
        tracker=v22.RowTracker()
        for i in range(3):out=tracker.update(i/10.,observation(stripe()))
        self.assertFalse(out['pairs'][0]['persistent'])
        out=tracker.update(.3,observation(np.full(131,50.)))
        self.assertEqual(out['pairs'],[])

    def test_global_translation_keeps_relative_width(self):
        tracker=v22.RowTracker()
        for i in range(6):out=tracker.update(i/10.,observation(stripe(-10+i,10+i)))
        pair=out['pairs'][0]
        self.assertTrue(pair['persistent']);self.assertAlmostEqual(pair['relative_separation_change_px'],0.)
        self.assertAlmostEqual(pair['center_px'],5.)

    def test_local_separation_growth_recorded_not_event(self):
        tracker=v22.RowTracker()
        for i in range(6):out=tracker.update(i/10.,observation(stripe(-10,10+i)))
        pair=out['pairs'][0]
        self.assertTrue(pair['persistent']);self.assertAlmostEqual(pair['relative_separation_change_px'],5.)
        self.assertFalse(pair['first_observed_is_physical_birth'])

    def test_missing_profile_breaks_track_even_one_frame(self):
        tracker=v22.RowTracker()
        for i in range(6):out=tracker.update(i/10.,observation(stripe()))
        old=out['pairs'][0]['track_id']
        tracker.update(.6,v22.observe_profile(np.full(131,np.nan)))
        out=tracker.update(.7,observation(stripe()))
        self.assertNotEqual(out['pairs'][0]['track_id'],old)
        self.assertEqual(out['pairs'][0]['observations'],1);self.assertFalse(out['pairs'][0]['persistent'])

    def test_gap_and_rewind_reset(self):
        tracker=v22.RowTracker();r=observation(stripe())
        tracker.update(1.,r);out=tracker.update(1.4,r)
        self.assertEqual(out['pairs'][0]['observations'],1)
        out=tracker.update(1.3,r)
        self.assertFalse(out['valid']);self.assertEqual(out['pairs'],[])

    def test_displacement_jump_breaks_association(self):
        tracker=v22.RowTracker()
        a=tracker.update(0.,observation(stripe()));b=tracker.update(.1,observation(stripe(0,20)))
        self.assertNotEqual(a['edges'][0]['track_id'],b['edges'][0]['track_id'])
        self.assertEqual(b['pairs'][0]['observations'],1)

    def test_equal_distance_ambiguity_never_arbitrarily_continues(self):
        def fake(offsets):
            return dict(valid=True,reason='synthetic',edges=[dict(offset_px=float(x),polarity=1,abs_gradient=2.) for x in offsets],pairs=[])
        tracker=v22.RowTracker();a=tracker.update(0.,fake([-1,1]));b=tracker.update(.1,fake([0]))
        self.assertNotIn(b['edges'][0]['track_id'],[x['track_id'] for x in a['edges']])
        self.assertEqual(b['edges'][0]['observations'],1)

    def test_periodic_texture_can_persist_without_proving_new_layer(self):
        x=np.arange(131);p=100.+35.*np.cos(2*np.pi*x/20.)
        tracker=v22.RowTracker()
        for i in range(7):out=tracker.update(i/10.,observation(p))
        self.assertGreater(sum(pair['persistent'] for pair in out['pairs']),1)
        self.assertTrue(all(not pair['first_observed_is_physical_birth'] for pair in out['pairs']))


class ReplayTests(unittest.TestCase):
    def data(self):
        times=np.arange(71)/10.;profiles=np.tile(stripe(),(len(times),1,1))
        return times,profiles,np.ones(len(times),bool),np.zeros((len(times),1)),np.array([66])

    def test_startup_no_candidates_and_exact_current_confirmation(self):
        result=v22.replay(*self.data())
        self.assertTrue(all(r['edge_count']==0 for r in result['records'][:51]))
        confirm=[r for r in result['records'] if r['new_persistence_confirmations']]
        self.assertEqual(len(confirm),1);self.assertEqual(confirm[0]['time_s'],5.5)
        self.assertEqual(confirm[0]['new_persistence_confirmations'][0]['first_observed_s'],5.1)
        self.assertEqual(confirm[0]['new_persistence_confirmations'][0]['first_available_s'],5.5)

    def test_prefix_full_replay_equal(self):
        args=self.data();full=v22.replay(*args)
        for cut in (20,51,55,60):
            prefix=v22.replay(*[a[:cut] for a in args[:4]],args[4])
            self.assertEqual(full['records'][:cut],prefix['records'])
            np.testing.assert_array_equal(full['arrays']['edge_counts'][:cut],prefix['arrays']['edge_counts'])

    def test_summary_twelve_buckets_and_longest_track(self):
        result=v22.replay(*self.data());s=v22.summary(result['records'],result['arrays'])
        self.assertEqual(len(s['fixed_five_second_buckets']),12)
        self.assertEqual(s['complete']['row_frame_cells'],71)
        self.assertEqual(s['opposite_pair_tracks']['earliest_first_available_s'],5.5)
        self.assertAlmostEqual(s['opposite_pair_tracks']['longest_direct_chain']['duration_s'],1.9)
        self.assertIn('Static bands',s['negative_control_warning'])

    def test_no_edges_empty_tables_valid_summary(self):
        args=list(self.data());args[1][:]=50.
        r=v22.replay(*args);s=v22.summary(r['records'],r['arrays'])
        self.assertEqual(r['arrays']['edge_table'].shape,(0,11));self.assertEqual(r['arrays']['pair_table'].shape,(0,18))
        self.assertEqual(s['opposite_pair_tracks']['persistent_row_pair_tracks'],0)

    def test_noncausal_or_nonbool_source_rejected(self):
        args=list(self.data());args[0][3]=args[0][2]
        with self.assertRaises(ValueError):v22.replay(*args)
        args=list(self.data());args[2]=np.ones(71,float)
        with self.assertRaises(ValueError):v22.replay(*args)

    def test_fresh_output_required_before_dependency_access(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileExistsError):v22.freeze(Path(td))


if __name__=='__main__':unittest.main()
