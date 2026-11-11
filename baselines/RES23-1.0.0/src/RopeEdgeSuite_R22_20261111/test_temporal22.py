import os,sys,unittest,importlib
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent/'RopeEdgeSuite_R21_20261110'))
analyze=importlib.import_module(os.getenv('RES_STATE_MODULE','events22')).analyze

def row(i,n,available=True):
    return dict(frame=i,time_s=i/10,N=n,N_interval=[n-.3,n+.3],available=available,
        current_coil_side_axis=True,boundary_occluded=False,boundary_anchor_hull=True,
        pitch=20,boundary_uncertainty_px=3,epoch=0)

class DirectionTests(unittest.TestCase):
    def test_far_distance_can_be_approaching(self):
        out=analyze([row(i,12-.1*i) for i in range(41)],10)
        self.assertEqual(out['frames'][-1]['state'],'approaching')

    def test_one_outlier_does_not_reverse_direction(self):
        rows=[row(i,12-.06*i) for i in range(70)]
        rows[50]=row(50,18)
        out=analyze(rows,10)['frames']
        self.assertTrue(all(x['state']=='approaching' for x in out[40:]))

    def test_actual_reversal_recovers(self):
        rows=[row(i,8-.1*i if i<40 else 4+.1*(i-40)) for i in range(80)]
        out=analyze(rows,10)['frames']
        self.assertEqual(out[35]['state'],'approaching')
        self.assertEqual(out[-1]['state'],'departing')

    def test_missing_frames_have_bounded_explicit_history(self):
        rows=[row(i,12-.06*i,i<40) for i in range(60)]
        out=analyze(rows,10)['frames']
        self.assertEqual(out[41]['state'],'approaching')
        self.assertEqual(out[41].get('evidence_mode'),'history_hold')
        self.assertEqual(out[59]['state'],'unknown')

    def test_no_measurements_no_motion_or_event(self):
        out=analyze([row(i,.2,False) for i in range(50)],10)
        self.assertEqual(out['events'],[])
        self.assertTrue(all(x['state']=='unknown' for x in out['frames']))

    def test_display_hold_cannot_create_near_event(self):
        rows=[row(i,8-.1*i) for i in range(50)]
        rows += [row(i,.1,False) for i in range(50,60)]
        self.assertEqual(analyze(rows,10)['events'],[])

    def test_pitch_changes_cannot_reverse_geometric_approach(self):
        rows=[dict(row(i,5+.05*i),motion_distance=12-.06*i) for i in range(60)]
        self.assertEqual(analyze(rows,10)['frames'][-1]['state'],'approaching')

    def test_winding_plateau_keeps_supported_longer_trend(self):
        rows=[row(i,12-.08*i if i<50 else 8.) for i in range(79)]
        out=analyze(rows,10)['frames']
        self.assertEqual(out[-1]['state'],'approaching')

if __name__=='__main__':unittest.main()
