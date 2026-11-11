import unittest
import cv2,numpy as np
from common import *
from geometry_gv1 import assess
from region_bg6 import profile_guard,validate

def scene(tool=False):
    im=np.full((600,900,3),(40,180,210),np.uint8)
    for y in range(200,600):
        val=int(55+35*np.sin((y-200)*np.pi/10));im[y,:,:]=val
    if tool:cv2.rectangle(im,(220,205),(650,250),(245,245,245),-1)
    reg=dict(valid=True,reasons=[],edge_y=200.,edge_uncertainty_px=2.,
        top_curve_xy=[[x,200.] for x in range(30,871,8)],top_observed_xy=[[x,200.] for x in range(30,871,8)],
        top_curve_supported_segments=[[[x,200.] for x in range(30,871,8)]])
    return im,reg

def track(lo=1300,hi=2200,dy=80):
    x=np.arange(lo,hi,4.);y=np.full_like(x,2000.+dy)
    return dict(valid=True,line_points_xy=np.c_[x,y].tolist(),current_line_abc=[0.,1.,-(2000.+dy)])

def reg():return dict(valid=True,edge_y=2000.,edge_uncertainty_px=2.,top_curve_xy=[[x,2000.] for x in range(800,3000,8)])

class Checks(unittest.TestCase):
    def test_clear_coil_passes(self):
        im,r=scene();self.assertTrue(profile_guard(im,r,450,20)['valid'])
    def test_broad_tool_cannot_be_boundary(self):
        im,r=scene(True);v=validate(im,r,450,20);self.assertFalse(v['valid']);self.assertIsNone(v['edge_y']);self.assertFalse(v['top_curve_supported_segments'])
    def test_perfect_but_far_line_is_not_local_measurement(self):
        g=assess(track(800,1300),reg());self.assertIn('unsupported_extrapolation_too_long',g['reasons'])
    def test_short_supported_projection_is_only_conditional_without_d(self):
        g=assess(track(),reg());self.assertTrue(g['conditional_projection_valid']);self.assertFalse(g['certified_distance_valid'])
    def test_negative_gap_is_not_zero_count(self):
        g=assess(track(dy=-40),reg());self.assertIn('projection_boundary_incompatible',g['reasons'])
    def test_failed_s1_produces_no_projection(self):
        g=assess(dict(valid=False),reg());self.assertIsNone(g['y_at_d']);self.assertFalse(g['conditional_projection_valid'])
    def test_current_boundary_required(self):
        r=reg();r['valid']=False;g=assess(track(),r);self.assertFalse(g['conditional_projection_valid'])
    def test_weak_curvature_changes_remote_projection(self):
        t=track();p=np.asarray(t['line_points_xy']);p[:,1]+=.00012*(p[:,0]-p[:,0].mean())**2;t['line_points_xy']=p.tolist()
        self.assertIn('projected_position_sensitive_to_support_subset',assess(t,reg())['reasons'])

def main():
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks);res=unittest.TextTestRunner(verbosity=2).run(suite)
    dump(ROOT/'test_receipt.json',dict(tests=res.testsRun,passed=res.wasSuccessful(),failures=len(res.failures),errors=len(res.errors),
        geometry_code_sha256=sha(ROOT/'geometry_gv1.py'),boundary_code_sha256=sha(ROOT/'region_bg6.py')))
    assert res.wasSuccessful()
    # Raw-image visual labels made before running this fixed policy on holdout.
    clear=[75,175,225,275,325,355,475,575];blocked=[375,395];ambiguous=[415]
    dump(ROOT/'visual_holdout_labels.json',dict(clear=clear,foreground_in_first_three_pitch=blocked,ambiguous=ambiguous,
        labeler='assistant raw-image visual review; not human pixel/count ground truth',source='holdout_raw_contact_sheet.jpg'))
    cap=cv2.VideoCapture(read(RUN12/'binding.json')['source']);rows=[]
    for i in sorted(clear+blocked+ambiguous):
        cap.set(1,i);ok,im=cap.read();assert ok;r=record(i);assert pixelsha(im)==r['source_pixel_sha256']
        rr=validate(im,r['region_BG5']);passed=rr['identity_guard']['valid'];expected=None if i in ambiguous else i in clear
        rows.append(dict(frame=i,expected_guard_pass=expected,guard_pass=passed,combined_region_pass=rr['valid'],
            broad_fraction=rr['identity_guard'].get('broad_fraction'),reasons=rr['reasons'],match=None if expected is None else passed==expected))
    cap.release();scored=[r for r in rows if r['expected_guard_pass'] is not None]
    dump(ROOT/'holdout_receipt.json',dict(rows=rows,scored=len(scored),matched=sum(r['match'] for r in scored),
        boundary_code_sha256=sha(ROOT/'region_bg6.py'),no_parameter_changes_from_holdout=True))
    print('HOLDOUT',[(r['frame'],r['match'],r['reasons']) for r in rows])
if __name__=='__main__':main()
