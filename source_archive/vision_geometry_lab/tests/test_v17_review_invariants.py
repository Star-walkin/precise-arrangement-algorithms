"""Independent synthetic review checks; no videos/labels/training are loaded."""
import numpy as np
import pytest
import copy

from vision_geometry_lab.v17_temporal_boundary import TemporalBoundary, boundary_path_features
from vision_geometry_lab.v17_boundary_trial import validate_inputs, validate_metadata


def segment(lo=20,hi=240):
    return [[float(x),90.] for x in range(lo,hi+1,2)]


def stable(dx=0):
    return dict(pose_stable=True,history_action='continue',previous_to_current=[[1.,0.,dx],[0.,1.,0.]])


def update(tracker,now,segments=None,scene=None):
    return tracker.update(now,(300,300,3),[segment()] if segments is None else segments,10.,stable() if scene is None else scene)


@pytest.mark.parametrize('bad_width',[None,'unavailable',float('nan'),float('inf'),0.,-1.])
def test_invalid_optical_width_clears_old_history(bad_width):
    tracker=TemporalBoundary();update(tracker,0.)
    result=tracker.update(.2,(300,300,3),[segment()],bad_width,stable())
    assert result['boundary_state']=='unknown'
    result=update(tracker,.4,[segment(20,80),segment(180,240)])
    assert result['predicted_point_count']==0


@pytest.mark.parametrize('bad_time',[None,'unknown',float('nan'),float('inf')])
def test_invalid_time_clears_history(bad_time):
    tracker=TemporalBoundary();update(tracker,0.)
    result=tracker.update(bad_time,(300,300,3),[segment()],10.,stable())
    assert result['boundary_state']=='unknown'
    assert update(tracker,.4,[segment(20,80),segment(180,240)])['predicted_point_count']==0


def test_predicted_gap_never_claims_current_observation_even_with_good_sigma():
    tracker=TemporalBoundary();update(tracker,0.)
    boundary=update(tracker,.2,[segment(20,80),segment(180,240)])
    path=[[[float(x),92.] for x in range(101,160,2)]]
    features=boundary_path_features(dict(observed_segments=path),boundary,
        dict(width_px=10.,origin_xy=[0.,0.],d_xy=[1.,0.]))
    assert features['usable']
    assert features['boundary_state']=='predicted'
    assert features['oldest_observed_at_s']==0.
    assert features['boundary_max_age_s']==.2
    assert features['interval_is_calibrated_confidence'] is False
    assert features['path_visibility_assessed'] is False
    assert features['physical_event_confirmed'] is False


def test_missing_current_overlap_prevents_hidden_curve_continuation():
    tracker=TemporalBoundary();update(tracker,0.)
    result=update(tracker,.2,[segment(20,40)])
    assert not result['overlap_valid']
    assert result['predicted_point_count']==0


def test_real_new_direct_observation_refreshes_only_its_own_columns():
    tracker=TemporalBoundary();update(tracker,0.)
    result=update(tracker,.2,[segment(20,80),segment(180,240)])
    for point,state,stamp in zip(result['points'],result['states'],result['observed_at_s']):
        if 80<point[0]<180:
            assert state=='predicted' and stamp==0.
        else:
            assert state=='direct' and stamp==.2


def test_subgrid_registration_resampling_preserves_oldest_hidden_age():
    tracker=TemporalBoundary();update(tracker,0.)
    update(tracker,.2,[segment(22,82),segment(182,242)],stable(2))
    result=update(tracker,.4,[segment(24,84),segment(184,244)],stable(2))
    assert result['predicted_point_count']>0
    assert result['boundary_max_age_s']==.4
    assert result['oldest_observed_at_s']==0.


def payloads():
    extraction=dict(protocol_sha256='protocol',records=[dict(frame_index=i,time_s=i/10.) for i in (0,2,4)])
    measurement=copy.deepcopy(extraction);measurement['extraction_index_sha256']='extract'
    return extraction,measurement


@pytest.mark.parametrize('fault',['duplicate','reorder','time','hash'])
def test_driver_rejects_ambiguous_or_wrong_input_sequence(fault):
    extraction,measurement=payloads()
    if fault=='duplicate':measurement['records'].append(dict(measurement['records'][-1]))
    elif fault=='reorder':measurement['records'].reverse()
    elif fault=='time':measurement['records'][1]['time_s']=.4
    else:measurement['extraction_index_sha256']='wrong'
    with pytest.raises(ValueError):
        validate_inputs(extraction,measurement,[0,2,4],10.,'protocol','extract')


def test_driver_accepts_matched_provenance_and_metadata():
    extraction,measurement=payloads()
    validate_inputs(extraction,measurement,[0,2,4],10.,'protocol','extract')
    validate_metadata(dict(frame_index=2,time_s=.2,fps=10.),2,10.)


@pytest.mark.parametrize('field,value',[('frame_index',4),('time_s',.4),('fps',20.),
    pytest.param('fps',float('nan'),marks=pytest.mark.xfail(strict=True,
        reason='Known frozen V17 helper defense gap: NaN fps bypasses abs comparison; hashed actual inputs are finite 10.0 fps. Fix in next protocol, not frozen core.'))])
def test_driver_rejects_wrong_metadata(field,value):
    meta=dict(frame_index=2,time_s=.2,fps=10.);meta[field]=value
    with pytest.raises(ValueError):validate_metadata(meta,2,10.)


def test_registration_for_different_previous_frame_cannot_move_history():
    tracker=TemporalBoundary();update(tracker,0.)
    scene=stable();scene['previous_time_s']=.1
    result=update(tracker,.2,[segment(20,80),segment(180,240)],scene)
    assert result['predicted_point_count']==0
