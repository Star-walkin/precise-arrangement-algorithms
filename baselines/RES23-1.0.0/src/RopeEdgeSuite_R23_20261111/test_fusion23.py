import unittest,numpy as np
from test_period23 import row
from fusion23 import analyze

def fixture():
    rows=[]
    for i in range(340):
        r=row(i,20-.04*i if i<200 else (.7 if i<242 else .7+.04*(i-242)))
        r['state22'].update(state='approaching' if i<200 else ('near' if i<242 else 'departing'),
            motion='approaching' if i<242 else 'departing',current_evidence=True)
        rows.append(r)
    near=[dict(start_s=19.7,entry_confirmed_s=20.,start_frame=197)]
    pattern=np.random.default_rng(8).normal(size=(25,4,96)).astype('float32')
    features=np.tile(pattern,(14,1,1))[:340]
    return rows,near,features

class FusionTests(unittest.TestCase):
    def test_two_cycles_plus_current_evidence_gives_candidate(self):
        rows,near,f=fixture();out=analyze(rows,near,f,10)
        self.assertEqual(len(out['events']),1)
        self.assertAlmostEqual(out['events'][0]['timeline']['t2'],25.,delta=.15)
        self.assertTrue(out['events'][0]['image_supported'])
        self.assertAlmostEqual(out['events'][0]['candidate_s'],25.,delta=.15)
    def test_timer_and_texture_cannot_confirm_without_current_rope(self):
        rows,near,f=fixture()
        for r in rows[210:]:
            r['available']=False;r['current_coil_side_axis']=False;r['state22']['evidence_mode']='unavailable'
        out=analyze(rows,near,f,10)
        self.assertFalse(any(e.get('image_supported') for e in out['events']))
    def test_prefix_has_no_future_confirmation(self):
        rows,near,f=fixture();out=analyze(rows,near,f,10)
        for n in [199,201,225,251,255,258,310]:
            short=analyze(rows[:n],near,f[:n],10)
            self.assertEqual(short['frames'],out['frames'][:n])
    def test_no_near_anchor_no_event(self):
        rows,near,f=fixture();self.assertEqual(analyze(rows,[],f,10)['events'],[])

    def test_time_origin_does_not_change_the_detected_phase(self):
        rows,near,f=fixture();base=analyze(rows,near,f,10)
        shifted=[dict(r,time_s=r['time_s']+100) for r in rows]
        anchors=[dict(e,start_s=e['start_s']+100,entry_confirmed_s=e['entry_confirmed_s']+100) for e in near]
        out=analyze(shifted,anchors,f,10)
        self.assertAlmostEqual(out['events'][0]['candidate_s']-base['events'][0]['candidate_s'],100)

    def test_near_stay_alone_is_not_motor_completion(self):
        rows,near,f=fixture()
        for r in rows[200:]:r['state22'].update(state='near',motion='approaching')
        self.assertFalse(analyze(rows,near,f,10)['events'][0]['image_supported'])

if __name__=='__main__':unittest.main()
