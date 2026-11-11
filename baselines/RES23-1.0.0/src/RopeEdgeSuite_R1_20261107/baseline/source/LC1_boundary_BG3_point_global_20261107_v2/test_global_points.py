import unittest
import numpy as np
from global_points import group_and_fit,intersections,model_envelope

def fixture(missing=False,absent=False):
    rng=np.random.default_rng(42);pts=[];truth=[];pid=0
    for row in range(12):
        if absent and row==5:continue
        for x in np.arange(20,561,3):
            if missing and row==5 and 220<x<320:continue
            scale=.95+.00015*x;base=55+.0006*(x-300)**2
            y=base+(row*12)*scale+rng.normal(0,.18)
            tid=row*3+(0 if x<220 else 1 if x<320 else 2)
            pts.append([pid,x,y,.9,3.,.0012*(x-300)+row*12*.00015,12.,tid]);truth.append(row);pid+=1
    surface=np.full((300,600),255,np.uint8)
    data=dict(points=np.asarray(pts),surface=surface,exclusion=np.zeros_like(surface),
        broad=np.ones_like(surface,np.float32),clean=surface.copy(),body=np.zeros_like(surface,np.float32),
        pitch=np.asarray(12.),x0=np.asarray(0),y0=np.asarray(0))
    data['model_envelope']=surface.copy()
    return data,np.asarray(truth)

class GlobalPointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data,cls.truth=fixture(missing=True)
        cls.curves,cls.diag,cls.points=group_and_fit(cls.data,270.)
    def test_one_class_one_continuous_curve(self):
        self.assertEqual(len(self.curves),12)
        for c in self.curves:self.assertTrue(np.all(np.diff(c['xy'][:,0])==1))
    def test_point_identity_unique_and_correct(self):
        allids=[]
        for c in self.curves:
            ids=c['point_ids'];allids+=ids
            labels,counts=np.unique(self.truth[ids],return_counts=True)
            self.assertGreater(counts.max()/len(ids),.98)
        self.assertEqual(len(allids),len(set(allids)))
    def test_long_gap_keeps_curve_but_blocks_count(self):
        c=next(c for c in self.curves if np.median(self.truth[c['point_ids']])==5)
        self.assertLess(c['xy'][0,0],220);self.assertGreater(c['xy'][-1,0],320)
        crossing,rejected=intersections(self.curves,self.points,self.data,270.)
        self.assertNotIn(c['boundary_id'],[p['boundary_id'] for p in crossing])
        self.assertIn(c['boundary_id'],[p['boundary_id'] for p in rejected])
    def test_gap_fit_has_known_synthetic_accuracy(self):
        c=next(c for c in self.curves if np.median(self.truth[c['point_ids']])==5)
        y=np.interp(270,c['xy'][:,0],c['xy'][:,1]);expected=55+.0006*(270-300)**2+60*(.95+.00015*270)
        self.assertLess(abs(y-expected),.7)
    def test_no_curve_crossings(self):
        for a,b in zip(self.curves,self.curves[1:]):
            xs=np.arange(max(a['xy'][0,0],b['xy'][0,0]),min(a['xy'][-1,0],b['xy'][-1,0]))
            self.assertTrue(np.all(np.interp(xs,b['xy'][:,0],b['xy'][:,1])>np.interp(xs,a['xy'][:,0],a['xy'][:,1])))
    def test_missing_entire_rope_is_not_invented(self):
        data,truth=fixture(absent=True);curves,diag,p=group_and_fit(data,270.)
        self.assertEqual(len(curves),11)
    def test_display_envelope_does_not_change_evidence(self):
        surface=np.full((100,100),255,np.uint8);surface[30:60,40:60]=0
        prior=surface.copy();env=model_envelope(surface)
        self.assertTrue(np.array_equal(surface,prior));self.assertEqual(env[45,50],255);self.assertEqual(surface[45,50],0)
    def test_occlusion_retains_model_and_rejects_intersection(self):
        data,_=fixture();data['exclusion'][:,250:290]=255
        curves,diag,p=group_and_fit(data,270.);crossing,rejected=intersections(curves,p,data,270.)
        self.assertEqual(len(curves),12);self.assertEqual(len(crossing),0)
        for c in curves:
            k=np.searchsorted(c['xy'][:,0],270);self.assertEqual(c['point_states'][k],'inferred')

if __name__=='__main__':unittest.main()
