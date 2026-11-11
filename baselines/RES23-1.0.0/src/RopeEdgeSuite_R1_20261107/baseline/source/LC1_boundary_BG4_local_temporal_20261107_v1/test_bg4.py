import unittest,copy
import numpy as np
from temporal_count import Counter
from local_observation import local_crossings

def frame(i,n=None,anchor=True):
    return dict(nominal_time_s=i/10,region=dict(valid=True,edge_y=100),s1=dict(valid=True,current_line_abc=[0,1,-140]),
        intersection_y_px=140,camera_motion={},observation=None if n is None else dict(value=n,kind='measured' if anchor else 'estimated',
            sigma=.18,phase_anchor=anchor,geometry_value=n-.08,pitch_px=20))

class TemporalTests(unittest.TestCase):
    def locked(self):
        c=Counter()
        for i in range(15):r=c.step(frame(i,4-.05*i))
        self.assertNotEqual(r['state'],'unknown');return c,r
    def test_short_gap_and_timeout(self):
        c,last=self.locked()
        for i in range(15,18):
            r=c.step(frame(i));self.assertEqual(r['state'],'predicted');self.assertLess(r['value'],last['value'])
        self.assertEqual(c.step(frame(18))['state'],'unknown')
    def test_no_prediction_without_anchor(self):
        c=Counter()
        for i in range(20):self.assertEqual(c.step(frame(i,3,False))['state'],'unknown')
    def test_geometry_cannot_extend_indefinitely(self):
        c,_=self.locked();out=[]
        for i in range(15,30):out.append(c.step(frame(i,4-.05*i,False)))
        self.assertTrue(all(r['state']=='unknown' for r in out[10:]))
    def test_camera_shift_invalidates_d_until_relock(self):
        c,_=self.locked();f=frame(15,3);f['camera_motion']={'camera_changed':True}
        self.assertEqual(c.step(f)['state'],'unknown')
        for i in range(16,25):self.assertEqual(c.step(frame(i,3))['state'],'unknown')
    def test_outlier_is_not_measurement(self):
        c,last=self.locked();r=c.step(frame(15,12))
        self.assertFalse(r['observation_accepted']);self.assertIn('measurement_innovation_rejected',r['events'])
        self.assertLess(abs(r['value']-last['value']),.2)
    def test_direction_only_after_evidence(self):
        c=Counter();r=c.step(frame(0,4));self.assertEqual(r['direction'],'unconfirmed')
        c,r=self.locked();self.assertEqual(r['direction'],'approaching')
    def test_reversal_releases_approach_constraint(self):
        c,last=self.locked();states=[]
        for i in range(15,23):states.append(c.step(frame(i,3.3+.12*(i-14))))
        self.assertEqual(states[-1]['direction'],'departing')
        self.assertGreater(states[-1]['value'],last['value'])
    def test_prefix_invariance(self):
        frames=[frame(i,4-.02*i) for i in range(30)]
        a=Counter();b=Counter()
        x=[a.step(copy.deepcopy(f)) for f in frames]
        y=[b.step(copy.deepcopy(f)) for f in frames[:17]]
        self.assertEqual(x[:17],y)
    def test_region_failure_no_measurement_can_be_short_prediction(self):
        c,_=self.locked();f=frame(15);f['region']['valid']=False
        self.assertEqual(c.step(f)['state'],'predicted')
    def test_edge_passage_resets(self):
        c,_=self.locked();f=frame(15);f['intersection_y_px']=99
        self.assertEqual(c.step(f)['state'],'unknown')

class LocalTests(unittest.TestCase):
    def case(self):
        xy=np.c_[np.arange(20,81),np.full(61,60.)]
        xx=np.arange(20,81,5);points=np.c_[np.arange(len(xx)),xx,np.full(len(xx),60.),np.ones((len(xx),5))]
        data=dict(pitch=np.array(10.),surface=np.full((100,100),255,np.uint8),exclusion=np.zeros((100,100),np.uint8),
            broad=np.ones((100,100)),clean=np.full((100,100),255,np.uint8),body=np.zeros((100,100)),x0=0,y0=0)
        curve=dict(boundary_id='a',xy=xy,point_ids=list(range(len(xx))),support_fraction=1.,recovered_local=False)
        return curve,points,data
    def test_same_polyline_single_crossing(self):
        c,p,d=self.case();cross,rej=local_crossings([c],p,d,50.3)
        self.assertFalse(rej);self.assertEqual(len(cross),1);self.assertEqual(cross[0]['y'],60.)
    def test_actual_column_exclusion_remains_hard_failure(self):
        c,p,d=self.case();d['exclusion'][59:62,49:53]=255
        cross,rej=local_crossings([c],p,d,50.3)
        self.assertFalse(cross);self.assertIn('count_intersection_unknown_or_excluded',rej[0]['reasons'])
    def test_long_yellow_point_gap_not_filled_for_measurement(self):
        c,p,d=self.case();d['pitch']=np.array(3.);c['point_ids']=[0,1,2,10,11,12]
        cross,rej=local_crossings([c],p,d,50.3);self.assertFalse(cross);self.assertTrue(rej)

if __name__=='__main__':unittest.main(verbosity=2)
