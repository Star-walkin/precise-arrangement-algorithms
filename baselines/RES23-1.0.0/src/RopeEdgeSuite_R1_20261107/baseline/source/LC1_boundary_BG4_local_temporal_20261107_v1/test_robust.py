import unittest,copy
import test_bg4,test_reference
from temporal_robust import Counter

class RobustDirectionTests(unittest.TestCase):
    def test_small_three_frame_rise_does_not_reacquire(self):
        c=Counter()
        for i in range(15):c.step(test_bg4.frame(i,4-.05*i))
        events=[]
        for i,n in enumerate([3.39,3.46,3.53],15):events+=c.step(test_bg4.frame(i,n))['events']
        self.assertNotIn('reversal_reacquired_from_current_observations',events)
    def test_direction_expires_without_recent_trend_evidence(self):
        c=Counter()
        for i in range(15):c.step(test_bg4.frame(i,4-.05*i))
        c.direction_time=0
        c.history.clear()
        r=c.step(test_bg4.frame(15,3.25,False))
        self.assertEqual(r['direction'],'unconfirmed')

def suite():
    test_bg4.Counter=Counter
    s=unittest.TestSuite()
    for module in (test_bg4,test_reference):s.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    s.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(RobustDirectionTests));return s

if __name__=='__main__':
    from bg4_common import ROOT,dump
    result=unittest.TextTestRunner(verbosity=2).run(suite())
    dump(ROOT/'robust_test_receipt.json',dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful()))
    raise SystemExit(0 if result.wasSuccessful() else 1)
