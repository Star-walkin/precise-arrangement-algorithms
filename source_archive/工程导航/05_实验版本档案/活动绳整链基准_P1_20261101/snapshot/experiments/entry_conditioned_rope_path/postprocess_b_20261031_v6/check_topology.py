"""Finite synthetic contracts; no rope identity accuracy claims."""
import numpy as np
from common import HERE,dump,sha
from topology import crossings,prepare_task_free,check_attachment

def lane(name,y):return dict(track_id=name,boundary_ids=[name+'a',name+'b'],points_xy=[[0.,y],[200.,y]])
tracks=[lane('a',30.),lane('b',50.),lane('c',70.)]
p=np.c_[np.arange(0.,201.,3.),np.arange(0.,201.,3.)*.5+10.]
n=len(p)
f=dict(points_xy=p.tolist(),supported=[False]*n,protect_support=[True]*n,point_status=['heatmap_supported']*n,
    observed_straight_support=[True]*n,raw_a7_supported=[False]*n,width_half_px=5.,segment_status=['heatmap_supported']*(n-1))
events=crossings(p,tracks);assert len(events)==3
out,receipt=prepare_task_free(f,tracks);assert receipt['trimmed'] and not receipt['retained_prefix_events']
# A short hotspot on one side of the crossing is not a covering48px run.
g=dict(f,supported=[i<10 for i in range(n)])
_,r=prepare_task_free(g,tracks);assert r['trimmed']
g=dict(f,supported=[True]*n,point_status=['image_supported']*n)
_,r=prepare_task_free(g,tracks);assert not r['trimmed'] and len(r['events'])==3
bridge=np.array([[10.,10.],[50.,50.],[80.,70.]])
tail=np.array([[80.,70.],[180.,70.]])
assert not check_attachment(bridge,tail,tracks,'c')['allowed']
safe=np.array([[10.,10.],[25.,20.],[40.,30.]])
assert check_attachment(safe,np.array([[40.,30.],[180.,30.]]),tracks,'a')['allowed']
early=np.array([[10.,10.],[30.,40.],[50.,20.],[80.,30.]])
assert not check_attachment(early,np.array([[80.,30.],[180.,30.]]),tracks,'a')['allowed']
touch=np.array([[10.,10.],[50.,30.],[80.,10.]])
assert crossings(touch,[tracks[0]])[0]['kind']=='touch'
assert crossings(np.array([[10.,30.],[80.,30.]]),[tracks[0]])[0]['kind']=='overlap'
# Near-parallel crossings remain crossings, even at arbitrarily small angle.
near=np.array([[10.,29.999],[190.,30.001]])
assert len(crossings(near,[tracks[0]]))==1
assert len(crossings(p[::-1],tracks))==3
scale=lambda z:[dict(t,points_xy=(np.asarray(t['points_xy'])*2).tolist()) for t in z]
assert len(crossings(p*2,scale(tracks)))==3
assert not check_attachment(bridge,tail,tracks,'c')['allowed'] # no stage labels are an input
dump(HERE/'topology_contract_check.json',dict(status='passed',checks=[
    'unlicensed_straight_crossing_tail_retained_as_unresolved','support_must_cover_intersection',
    'sustained_original_supported_free_projection_is_not_attachment','bridge_crossing_other_lane_rejected',
    'single_endpoint_target_capture_allowed','early_target_cross_return_rejected','non_target_touch_recorded',
    'collinear_overlap_recorded','near_parallel_crossing_not_skipped','left_right_symmetric',
    'coordinate_scale_invariant','stage_label_cannot_bypass_geometry'],
    source_sha256=sha(HERE/'topology.py'),identity_accuracy_measured=False))
print('12 topology contracts passed')
