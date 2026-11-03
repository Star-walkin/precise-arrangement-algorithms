"""Synthetic-only contracts; no reference event times or development arrays."""
import json

import cv2
import numpy as np
import pytest

from vision_geometry_lab import v22_flange_tracks as m


ROIS = [[80, 240, 240, 330], [380, 240, 540, 330]]


def points():
    p = []
    g = []
    for k, (x1,y1,x2,y2) in enumerate(ROIS):
        xx, yy = np.meshgrid(np.linspace(x1+10,x2-10,8),np.linspace(y1+10,y2-10,5))
        p.extend(np.column_stack((xx.ravel(),yy.ravel())))
        g.extend([k]*xx.size)
    return np.asarray(p), np.asarray(g)


def image():
    rng = np.random.default_rng(221)
    a = np.zeros((480,640,3),np.uint8)
    bg = cv2.GaussianBlur(rng.integers(30,200,(190,640),np.uint8),(3,3),0)
    a[:190] = bg[:,:,None]
    for x1,y1,x2,y2 in ROIS:
        hsv = np.zeros((y2-y1,x2-x1,3),np.uint8)
        hsv[:,:,0] = 25; hsv[:,:,1] = 150
        hsv[:,:,2] = cv2.GaussianBlur(rng.integers(50,225,(y2-y1,x2-x1),np.uint8),(3,3),0)
        a[y1:y2,x1:x2] = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return a


def test_known_translation_is_image_motion_not_angle():
    p,g=points()
    out,mask=m.homography_evidence(p,p+[2.,-1.],g,ROIS)
    assert out['available'] and mask.all()
    assert out['symmetric_error_p90_px'] < 1e-4
    assert out['mechanical_angle_available'] is False
    assert not out['spectral']['available']


def test_projective_conjugate_recovers_unsigned_spectrum_only():
    a=np.deg2rad(12.)
    R=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    P=np.array([[1.1,.15,300],[.05,.8,240],[.0002,.0001,1.]])
    H=P@R@np.linalg.inv(P)
    diagnostic=m.spectral_diagnostic(H)
    assert diagnostic['available']
    assert diagnostic['projective_angle_magnitude_deg']==pytest.approx(12.)
    assert not diagnostic['mechanical_angle_available']
    assert not diagnostic['signed_angle_available']


def test_nonrigid_shear_has_rotation_like_spectrum_counterexample():
    # This exact planar mapping passes the image model but is NOT a rotation.
    k=.2
    H=np.array([[1-k*k,k,0],[-k,1,0],[0,0,1.]])
    p,g=points()
    out,_=m.homography_evidence(p,m._project(H,p),g,ROIS)
    assert out['available']
    assert out['spectral']['available']
    assert out['spectral']['projective_angle_magnitude_deg']==pytest.approx(np.degrees(np.arccos(1-k*k/2)),abs=.001)
    assert not out['mechanical_angle_available']


def test_collinear_points_are_not_sufficient():
    p=np.column_stack((np.arange(40)*10+80,np.full(40,270)))
    out,_=m.homography_evidence(p,p+[2,0],np.arange(40)%2,ROIS)
    assert not out['available']
    assert out['failed_reasons']


def test_random_nonrigid_correspondences_rejected():
    p,g=points()
    q=p+np.random.default_rng(13).normal(0,12,p.shape)
    out,_=m.homography_evidence(p,q,g,ROIS)
    assert not out['available']
    assert 'low_inlier_fraction' in out['failed_reasons']


def test_only_one_patch_not_sufficient():
    p,g=points(); keep=g==0
    out,_=m.homography_evidence(p[keep],p[keep]+[1,0],g[keep],ROIS)
    assert not out['available']
    assert 'insufficient_each_roi_support' in out['failed_reasons']
    assert 'narrow_spatial_support' in out['failed_reasons']


def test_static_duplicate_tracks_but_does_not_claim_mechanical_stop():
    a=image(); tracker=m.FlangeTracks(ROIS,(640,480))
    first=tracker.update(a,0.)
    second=tracker.update(a,.1)
    assert first['first_rejection']=='startup'
    assert second['candidate_available']
    assert second['background_median_displacement_px'] < .01
    assert max(second['native_track_age_s'])==pytest.approx(.1)
    assert not second['phase_accumulated']
    assert not second['mechanical_angle_available']


def test_flange_translation_with_stationary_background():
    a=image()
    shifted=cv2.warpAffine(a,np.array([[1.,0.,2.],[0.,1.,0.]]),(640,480))
    shifted[:190]=a[:190]
    tracker=m.FlangeTracks(ROIS,(640,480))
    tracker.update(a,0.)
    r=tracker.update(shifted,.1)
    assert r['candidate_available']
    assert not r['mechanical_angle_available']


def test_background_motion_is_explicit_rejection():
    a=image(); b=cv2.warpAffine(a,np.array([[1.,0.,3.],[0.,1.,0.]]),(640,480))
    tracker=m.FlangeTracks(ROIS,(640,480)); tracker.update(a,0.)
    r=tracker.update(b,.1)
    assert not r['candidate_available']
    assert 'background_image_motion' in r['failed_reasons']
    assert r['native_track_ids']==[]


def test_camera_rotation_is_not_promoted_to_flange_rotation():
    a=image()
    A=cv2.getRotationMatrix2D((320.,240.),1.2,1.)
    b=cv2.warpAffine(a,A,(640,480))
    tracker=m.FlangeTracks(ROIS,(640,480)); tracker.update(a,0.)
    r=tracker.update(b,.1)
    assert 'background_image_motion' in r['failed_reasons']
    assert not r['candidate_available']
    assert not r['mechanical_angle_available']


def test_frame_gap_breaks_ids_and_motion_chain():
    a=image(); tracker=m.FlangeTracks(ROIS,(640,480))
    tracker.update(a,0.); before=tracker.update(a,.1)
    gap=tracker.update(a,.5)
    after=tracker.update(a,.6)
    assert gap['first_rejection']=='frame_gap'
    assert not gap['candidate_available']
    assert set(before['native_track_ids']).isdisjoint(after['native_track_ids'])
    assert max(after['native_track_age_s'])==pytest.approx(.1)
    assert after['image_motion_chain_id'] != before['image_motion_chain_id']


def test_highlight_rejects_flange_but_not_zero_motion_output():
    a=image(); b=a.copy()
    for x1,y1,x2,y2 in ROIS: b[y1:y2,x1:x2]=255
    tracker=m.FlangeTracks(ROIS,(640,480)); tracker.update(a,0.)
    r=tracker.update(b,.1)
    assert not r['candidate_available']
    assert r['roi_mask_fraction']==[0.,0.]
    assert r['native_track_ids']==[]


def test_mild_brightness_shift_can_pass_not_photometric_identity_guarantee():
    a=image(); b=a.copy()
    for x1,y1,x2,y2 in ROIS:
        patch=b[y1:y2,x1:x2].astype(np.int16)+5
        b[y1:y2,x1:x2]=np.clip(patch,0,255).astype(np.uint8)
    tracker=m.FlangeTracks(ROIS,(640,480)); tracker.update(a,0.)
    r=tracker.update(b,.1)
    assert r['candidate_available']  # This is deliberately exposed, not tuned out.
    assert not r['mechanical_angle_available']


def test_blank_and_nan_times():
    tracker=m.FlangeTracks(ROIS,(640,480)); a=np.zeros((480,640,3),np.uint8)
    tracker.update(a,0.)
    r=tracker.update(a,.1)
    assert not r['candidate_available']
    assert r['first_rejection']=='no_previous_tracks'
    with pytest.raises(ValueError): tracker.update(a,float('nan'))
    with pytest.raises(ValueError): tracker.update(a,.1)


@pytest.mark.parametrize('output',[(None,None,None),(np.empty((0,1,2),np.float32),np.empty(0),np.empty(0))])
def test_missing_lk_output_is_unknown_and_never_second_call(monkeypatch,output):
    a=image(); tracker=m.FlangeTracks(ROIS,(640,480)); tracker.update(a,0.)
    calls=[]
    def fake(*args,**kwargs):
        calls.append(1)
        return output
    monkeypatch.setattr(cv2,'calcOpticalFlowPyrLK',fake)
    r=tracker.update(a,.1)
    assert len(calls)==1
    assert not r['candidate_available']
    assert r['first_rejection']=='lk_output_unavailable'
    assert r['native_track_ids']==[]


def test_backward_lk_none_is_unknown(monkeypatch):
    a=image(); tracker=m.FlangeTracks(ROIS,(640,480)); tracker.update(a,0.)
    real=cv2.calcOpticalFlowPyrLK
    calls=[]
    def fake(*args,**kwargs):
        calls.append(1)
        return real(*args,**kwargs) if len(calls)==1 else (None,None,None)
    monkeypatch.setattr(cv2,'calcOpticalFlowPyrLK',fake)
    r=tracker.update(a,.1)
    assert len(calls)==2
    assert r['first_rejection']=='lk_output_unavailable'
    assert not r['candidate_available']


def test_freeze_rejects_nonfinite_fps_before_file_access(tmp_path):
    control=tmp_path/'control.json'
    control.write_text(json.dumps(dict(schema='v20_appearance_control_protocol_1',videos={'left':{'fps':float('nan'),'frame_count':601}})))
    with pytest.raises(ValueError,match='complete'):
        m.freeze(tmp_path/'trial',control)


def test_json_strict_nonfinite():
    assert m.clean({'a':np.array([1.,np.nan]),'b':float('inf')})=={'a':[1.,None],'b':None}


def test_signed_spectrum_and_real_only_not_promoted():
    H=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    for candidate in [H,-3*H,np.linalg.inv(H)]:
        d=m.spectral_diagnostic(candidate)
        assert d['projective_angle_magnitude_deg']==pytest.approx(90.)
        assert not d['signed_angle_available']
    assert not m.spectral_diagnostic(np.eye(3))['available']
