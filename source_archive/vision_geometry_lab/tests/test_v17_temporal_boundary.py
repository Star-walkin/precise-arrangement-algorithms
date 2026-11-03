import numpy as np

from vision_geometry_lab.v17_temporal_boundary import TemporalBoundary, raster_direct


def segment(start=20, end=200, shift=0):
    return [[float(x), 90.+.0005*(x-100)**2+shift] for x in range(start,end+1,2)]


def stable(dx=0,dy=0):
    return dict(pose_stable=True, history_action='continue', previous_to_current=[[1,0,dx],[0,1,dy]])


def run(tracker,t,segments=None,scene=None,temporal=True):
    return tracker.update(t,(300,300,3),[segment()] if segments is None else segments,10.,stable() if scene is None else scene,temporal=temporal)


def test_current_only_cannot_fill_gap():
    r=run(TemporalBoundary(),0,[segment(20,80),segment(140,200)])
    assert not any(80<x<140 for x,y in r['points'])
    assert r['boundary_state']=='direct'


def test_temporal_gap_labelled_and_timestamp_not_refreshed():
    t=TemporalBoundary();run(t,0)
    r=run(t,.2,[segment(20,80),segment(140,200)])
    assert r['boundary_state']=='predicted'
    for point,state,stamp in zip(r['points'],r['states'],r['observed_at_s']):
        if 80<point[0]<140:
            assert state=='predicted' and stamp==0
    r=run(t,.4,[segment(20,80),segment(140,200)])
    assert r['boundary_max_age_s']==.4


def test_prediction_expires_without_becoming_its_own_observation():
    t=TemporalBoundary();run(t,0)
    for now in np.arange(.2,2.41,.2):
        r=run(t,float(now),[segment(20,80),segment(140,200)])
    assert r['predicted_point_count']==0


def test_bad_scene_drops_history_but_preserves_direct():
    t=TemporalBoundary();run(t,0)
    r=run(t,.2,[segment(20,80)],dict(pose_stable=False,history_action='reset'))
    assert r['predicted_point_count']==0 and r['boundary_state']=='direct'


def test_absent_direct_cannot_propagate_entire_curve():
    t=TemporalBoundary();run(t,0)
    r=run(t,.2,[])
    assert r['boundary_state']=='unknown' and not r['valid']


def test_large_current_boundary_change_resets_even_with_stable_camera():
    t=TemporalBoundary();run(t,0)
    r=run(t,.2,[segment(20,80,10),segment(140,200,10)])
    assert not r['overlap_valid'] and r['predicted_point_count']==0


def test_supported_pose_transform_moves_old_points_without_time_refresh():
    t=TemporalBoundary();run(t,0)
    r=run(t,.2,[segment(20,80,2),segment(140,200,2)],stable(dy=2))
    assert r['overlap_valid'] and r['predicted_point_count']>0
    assert r['oldest_observed_at_s']==0


def test_nonchronological_frame_does_not_use_future_history():
    t=TemporalBoundary();run(t,1)
    r=run(t,.2,[segment(20,80)])
    assert r['predicted_point_count']==0
    assert all(v==.2 for v in r['observed_at_s'])


def test_direct_control_disables_prediction():
    t=TemporalBoundary();run(t,0,temporal=False)
    r=run(t,.2,[segment(20,80),segment(140,200)],temporal=False)
    assert r['predicted_point_count']==0


def test_long_segment_link_and_branch_conflict_not_interpolated():
    grid=np.arange(0.,300.,4)
    values=raster_direct([[[20,90],[200,90]]],grid)
    assert not np.isfinite(values).any()
    values=raster_direct([segment(),segment(20,200,20)],grid)
    assert not np.isfinite(values).any()


def test_uncertainty_grows_while_direct_samples_stay_current():
    t=TemporalBoundary();run(t,0)
    a=run(t,.2,[segment(20,80),segment(140,200)])
    b=run(t,.4,[segment(20,80),segment(140,200)])
    assert b['boundary_sigma_px']>a['boundary_sigma_px']


def test_missing_width_or_resolution_change_cannot_hold_old_shape():
    t=TemporalBoundary();run(t,0)
    r=t.update(.2,(300,500,3),[segment(20,80)],10,stable())
    assert r['predicted_point_count']==0
    r=t.update(.4,(300,500,3),[segment()],float('nan'),stable())
    assert r['boundary_state']=='unknown'
