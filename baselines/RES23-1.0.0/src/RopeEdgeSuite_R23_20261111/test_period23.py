import sys,unittest
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent/'RopeEdgeSuite_R22_20261111'))
from period23 import estimate_count_period,combine_periods,timeline

def row(i,n):
    return dict(frame=i,time_s=i/10,N=n,N_interval=[n-.4,n+.4],available=True,current_coil_side_axis=True,
        boundary_occluded=False,epoch=0,state22=dict(evidence_mode='current'))

class PeriodTests(unittest.TestCase):
    def test_one_pitch_every_two_and_half_seconds(self):
        p=estimate_count_period([row(i,20-.04*i) for i in range(201)],20.)
        self.assertTrue(p['available']);self.assertAlmostEqual(p['period_s'],2.5,delta=.1)
    def test_sparse_outliers_do_not_change_period(self):
        rows=[row(i,20-.04*i) for i in range(201)]
        for i in [35,70,101,170]:rows[i]['N']+=15
        p=estimate_count_period(rows,20.)
        self.assertTrue(p['available']);self.assertAlmostEqual(p['period_s'],2.5,delta=.15)
    def test_short_or_stationary_samples_are_not_period(self):
        for rows in [[row(i,8.) for i in range(201)],[row(i,8-.04*i) for i in range(20)]]:
            self.assertFalse(estimate_count_period(rows,rows[-1]['time_s'])['available'])
    def test_reversal_is_not_constant_speed_winding(self):
        rows=[row(i,12-.04*i if i<100 else 8+.04*(i-100)) for i in range(201)]
        self.assertFalse(estimate_count_period(rows,20.)['available'])
    def test_disagreeing_periods_are_not_silently_averaged(self):
        a=dict(available=True,period_s=2.,period_interval_s=[1.9,2.1])
        b=dict(available=True,period_s=4.,period_interval_s=[3.9,4.1])
        p=combine_periods(a,b);self.assertFalse(p['available'])
    def test_two_cycles_from_near_event_not_one_cycle(self):
        e=dict(start_s=9.7,entry_confirmed_s=10.)
        p=dict(available=True,period_s=2.5,period_interval_s=[2.4,2.6])
        t=timeline(e,p)
        self.assertEqual(t['t1'],12.5);self.assertEqual(t['t2'],15.)
    def test_marked_acceleration_is_not_one_constant_period(self):
        rows=[row(i,30-.02*i if i<150 else 27-.08*(i-150)) for i in range(201)]
        self.assertFalse(estimate_count_period(rows,20.)['available'])

if __name__=='__main__':unittest.main()
