import unittest,copy
from common21 import R20
from measurement21 import DynamicGate
from measurement21b import DynamicGate as DynamicGateB
from events21b import analyze
from row_overlap21 import observe

def sample(i,n,ok=True):
    return dict(frame=i,time_s=i/10,available=ok,N=n,N_interval=[n-.5,n+.5],current_coil_side_axis=True,
        boundary_occluded=False,boundary_anchor_hull=True)

class Contracts(unittest.TestCase):
    def test_departure_does_not_wait_for_absolute_threshold(self):
        ns=[3-i*.1 for i in range(20)]+[.7]*20+[.7-i*.05 for i in range(14)]+[i*.06 for i in range(20)]
        rows=[sample(i,n) for i,n in enumerate(ns)];r=analyze(rows,10)
        self.assertEqual(len(r['events']),1);e=r['events'][0];self.assertIsNotNone(e['departure_s']);self.assertLess(e['departure_center_pitch'],1.25)
        self.assertFalse(e['physical_contact_confirmed'])
    def test_no_events_from_predictions_or_occlusion(self):
        for field,value in [('available',False),('boundary_occluded',True),('current_coil_side_axis',False)]:
            rows=[dict(sample(i,.3),**{field:value}) for i in range(20)]
            self.assertEqual(analyze(rows,10)['events'],[])
    def test_long_loss_is_not_declared_departure(self):
        rows=[sample(i,.6,i<10) for i in range(25)];r=analyze(rows,10)
        self.assertEqual(r['events'][0]['termination'],'evidence_lost_not_departure');self.assertIsNone(r['events'][0]['departure_s'])
    def test_timestamp_shift_equivariance(self):
        rows=[sample(i,3-i*.1 if i<25 else .5+(i-40)*.1 if i>40 else .5) for i in range(60)]
        a=analyze(rows,10)['events'];b=analyze([dict(r,time_s=r['time_s']+100) for r in rows],10)['events']
        self.assertEqual(len(a),len(b))
        for x,y in zip(a,b):self.assertAlmostEqual(y['start_s']-x['start_s'],100)
    def test_dynamic_gate_requires_real_temporal_support(self):
        g=DynamicGate();q=dict(available=True,d=2500.,N=3.,N_interval=[2.,4.],central=dict(boundary=dict(occluded=False,inside_current_anchor_hull=True)))
        axis=dict(current_coil_side_measured=True);tc=dict(valid=True,interval=[2470,2530])
        for i in range(3):self.assertFalse(g.step(i/10,0,q,axis,tc)[0])
        self.assertTrue(g.step(.3,0,q,axis,tc)[0]);self.assertFalse(g.step(.4,1,q,axis,tc)[0])
        self.assertFalse(g.step(.5,1,q,dict(current_coil_side_measured=False),tc)[0])
    def test_uncertainty_gate_tolerates_noise_but_rejects_fast_track(self):
        axis=dict(current_coil_side_measured=True);tc=dict(valid=True,interval=[2470,2530])
        for ns,expected in [([3.25,2.75,3.25,2.75],True),([0,1,2,3],False)]:
            gate=DynamicGateB()
            for i,n in enumerate(ns):
                q=dict(available=True,d=2500.,N=n,N_interval=[n-.75,n+.75],central=dict(boundary=dict(occluded=False,inside_current_anchor_hull=True)))
                passed,_=gate.step(i/10,0,q,axis,tc)
            self.assertEqual(passed,expected)
    def test_missing_top_groove_cannot_certify_first_row(self):
        m=dict(available=True,N=.5,central=dict(local_pitch_px=20,active_y=110,boundary=dict(y=100,uncertainty_px=5,
            inside_current_anchor_hull=True,occluded=False),crossings=[dict(y=y) for y in [120,140,160,180,200]]))
        a=dict(current_coil_side_measured=True)
        self.assertTrue(observe(m,a)['overlap'])
        m['central']['crossings']=m['central']['crossings'][1:]
        self.assertFalse(observe(m,a)['eligible'])
    def test_bounded_shape_extension_is_distinct_from_hidden_edge(self):
        rows=[dict(sample(i,.3),boundary_anchor_hull=False,pitch=20.,boundary_uncertainty_px=10.) for i in range(10)]
        self.assertEqual(len(analyze(rows,10)['events']),1)
        self.assertEqual(analyze([dict(r,boundary_uncertainty_px=25.) for r in rows],10)['events'],[])
        self.assertEqual(analyze([dict(r,boundary_occluded=True) for r in rows],10)['events'],[])

if __name__=='__main__':unittest.main()
