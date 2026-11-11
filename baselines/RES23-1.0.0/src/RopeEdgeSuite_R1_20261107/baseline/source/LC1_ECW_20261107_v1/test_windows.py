import unittest
from window_detector import WindowDetector,evidence
class Tests(unittest.TestCase):
    def test_missing_F3_is_not_zero(self):
        b=dict(region=dict(valid=False),s1=dict(valid=True),observation=None)
        e=evidence(dict(valid=False),b);self.assertIsNone(e['current_pitch_px']);self.assertFalse(e['near'])
    def test_single_frame_near_does_not_create_window(self):
        d=WindowDetector()
        for f in range(30):d.step(f,f*.1,near=f in [8,15,22],far=f not in [8,15,22])
        d.finish();self.assertFalse(d.events)
    def test_current_missing_does_not_confirm(self):
        d=WindowDetector()
        for f in range(6):d.step(f,f*.1,near=True)
        r=d.step(6,.6);self.assertEqual(r['state'],'candidate_pending_evidence');self.assertEqual(r['current_evidence'],'unresolved')
        for f in range(7,12):r=d.step(f,f*.1)
        self.assertFalse(r['event_active']);self.assertEqual(len(d.events),1)
        self.assertEqual(d.events[0]['last_near']['frame'],5)
    def test_hysteresis_exit(self):
        d=WindowDetector()
        for f in range(6):d.step(f,f*.1,near=True)
        for f in range(6,9):r=d.step(f,f*.1,far=True)
        self.assertEqual(r['state'],'separated');self.assertEqual(d.events[0]['end_reason'],'sustained_separation')
        self.assertEqual(d.events[0]['first_departure_evidence']['frame'],6)
    def test_cadence_not_frame_count(self):
        d=WindowDetector()
        for f in range(5):r=d.step(f,f*.01,near=True)
        self.assertFalse(r['event_active'])
    def test_camera_break(self):
        d=WindowDetector()
        for f in range(6):d.step(f,f*.1,near=True)
        r=d.step(6,.6,near=True,camera_valid=False)
        self.assertEqual(r['state'],'unknown');self.assertEqual(d.events[0]['end_reason'],'camera_unconfirmed_or_changed')
    def test_no_retroactive_online_output(self):
        d=WindowDetector();states=[]
        for f in range(6):states.append(d.step(f,f*.1,near=True)['state'])
        self.assertEqual(states[:3],['near_pending']*3);self.assertEqual(d.active['detection']['frame'],3)
    def test_negative_huge_not_contact(self):
        r=dict(valid=True,center_gap_proxy_px=-80,sensitivity_bound_px=3,rope=dict(optical_width_vertical_px=20))
        b=dict(region=dict(valid=False),s1=dict(valid=True),observation={})
        self.assertFalse(evidence(r,b)['near'])
    def test_high_uncertainty_reject(self):
        r=dict(valid=True,center_gap_proxy_px=0,sensitivity_bound_px=18,rope=dict(optical_width_vertical_px=20))
        b=dict(region=dict(valid=False),s1=dict(valid=True),observation={})
        self.assertFalse(evidence(r,b)['near'])
    def test_intermediate_measurement_cannot_fake_physical_contact(self):
        d=WindowDetector()
        for f in range(10):r=d.step(f,.1*f,near=True)
        self.assertFalse(r['physical_contact_confirmed'])
if __name__=='__main__':unittest.main()
