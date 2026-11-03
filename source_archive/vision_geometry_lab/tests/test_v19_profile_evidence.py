"""Synthetic V18b field-contract checks; no videos or truth labels are read."""
from copy import deepcopy

import numpy as np
import pytest

from vision_geometry_lab.v19_profile_evidence import (
    profile_evidence, startup_profile_model,
)


def profile(offset=0, sign=1, contrast=40.):
    x = np.arange(-65, 66)
    return 100.+sign*contrast*(x>=offset)


def startup(n=26, nrows=2, offsets=None, sign=1, contrast=40.):
    times = np.arange(n, dtype=float)/5.
    off = np.zeros((n, nrows)) if offsets is None else np.asarray(offsets, float)
    profiles = np.asarray([[profile(o, sign, contrast) for o in row] for row in off])
    arrays = dict(normal_offset=off.copy(), normal_profiles=profiles,
        normal_gradient=np.full(off.shape, sign*contrast/2.),
        normal_peak_ratio=np.full(off.shape, 2.),
        normal_clipped_fraction=np.zeros(off.shape),
        normal_residual=np.full(off.shape, np.nan), pose_valid=np.ones(n, bool))
    return times, arrays


def current(model, offset=0, sign=1, contrast=40., time_s=6.):
    nrows = len(model['supported'])
    offsets = np.broadcast_to(np.asarray(offset, float), (nrows,)).copy()
    return dict(normal_offset=offsets, normal_residual=offsets-model['baseline_offset'],
        normal_gradient=np.full(nrows, sign*contrast/2.),
        normal_peak_ratio=np.full(nrows, 2.),
        normal_clipped_fraction=np.zeros(nrows),
        normal_profiles=np.asarray([profile(o,sign,contrast) for o in offsets]),
        pose_valid=True, time_s=time_s)


def model():
    return startup_profile_model(*startup())


def test_supported_startup_and_positive_current_do_not_claim_material_identity():
    m = model()
    np.testing.assert_array_equal(m['supported'], True)
    np.testing.assert_array_equal(m['baseline_offset'], 0.)
    np.testing.assert_array_equal(m['noise_sigma'], 1.)
    np.testing.assert_array_equal(m['threshold_px'], 3.)
    np.testing.assert_array_equal(m['profile_sign'], 1.)
    np.testing.assert_array_equal(m['baseline_contrast'], 40.)
    r = profile_evidence(current(m,offset=7.),m)
    assert r['quality_mask'].all() and r['time_checked']
    assert np.all(r['reason']=='profile_consistent_not_material_identity')
    for obj in (m,r):
        assert not obj['material_identity_confirmed']
        assert not obj['visibility_confirmed']
        assert not obj['new_layer_confirmed']
        assert not obj['hardware_access']


def test_startup_reads_no_future_observation_values():
    times,a=startup(41)
    first=startup_profile_model(times,a)
    for k in ('normal_offset','normal_gradient','normal_peak_ratio','normal_profiles'):
        a[k][times>5.]=np.nan
    a['pose_valid'][times>5.]=False
    second=startup_profile_model(times,a)
    for k in ('baseline_offset','noise_sigma','threshold_px','profile_sign',
              'gradient_sign','baseline_contrast','supported','profile_offset_count',
              'reliable_profile_count'):
        np.testing.assert_array_equal(first[k],second[k])
    assert second['startup_frame_count']==26
    assert second['startup_times_s'].max()==5.


def test_model_arrays_are_detached_and_readonly():
    times,a=startup()
    m=startup_profile_model(times,a)
    a['normal_offset'][:]=20.
    times[:]=-10.
    assert m['baseline_offset'][0]==0.
    assert m['startup_times_s'][0]==0.
    with pytest.raises(ValueError):
        m['baseline_offset'][0]=123.


def test_mad_noise_and_threshold_are_heuristic_scales():
    off=np.tile(np.tile([-2.,0.,2.],8)[:,None],(1,2))
    times,a=startup(24,offsets=off)
    m=startup_profile_model(times,a)
    np.testing.assert_allclose(m['noise_sigma'],1.4826*2.)
    np.testing.assert_allclose(m['threshold_px'],3.*1.4826*2.)
    assert 'not a calibrated confidence bound' in m['scale_interpretation']


def test_minimum_twenty_startup_pairs():
    m=startup_profile_model(*startup(19))
    assert not m['supported'].any()
    assert np.isnan(m['baseline_offset']).all()
    assert np.all(m['reason']=='insufficient_startup_profile_offset_pairs')


def test_noise_does_not_use_offsets_without_complete_plateau_support():
    offsets=np.zeros((26,2))
    offsets[:7,0]=54.
    times,a=startup(offsets=offsets)
    m=startup_profile_model(times,a)
    np.testing.assert_array_equal(m['profile_offset_count'],[19,26])
    assert not m['supported'][0]
    assert np.isnan(m['noise_sigma'][0])


def test_minimum_ten_reliable_startup_samples_at_inclusive_peak_threshold():
    times,a=startup()
    a['normal_peak_ratio'][:]=1.19
    a['normal_peak_ratio'][:9,0]=1.2
    a['normal_peak_ratio'][:10,1]=1.2
    m=startup_profile_model(times,a)
    np.testing.assert_array_equal(m['supported'],[False,True])
    np.testing.assert_array_equal(m['reliable_profile_count'],[9,10])
    assert np.isfinite(m['noise_sigma']).all()


def test_invalid_pose_and_profile_samples_do_not_count_toward_startup():
    times,a=startup()
    a['pose_valid'][:7]=False
    m=startup_profile_model(times,a)
    assert (m['profile_offset_count']==19).all()
    assert not m['supported'].any()
    times,a=startup()
    a['normal_profiles'][:7,0,20]=np.nan
    m=startup_profile_model(times,a)
    np.testing.assert_array_equal(m['supported'],[False,True])


def test_disagreeing_startup_gradient_and_plateau_polarity_is_unsupported():
    times,a=startup()
    a['normal_gradient'][:,0]*=-1.
    m=startup_profile_model(times,a)
    assert m['reason'][0]=='startup_gradient_plateau_sign_disagree'
    np.testing.assert_array_equal(m['supported'],[False,True])


def test_equal_positive_negative_startup_signs_are_unresolved():
    times,a=startup(20)
    a['normal_profiles'][:10,0]=profile(sign=-1)
    a['normal_gradient'][:10,0]=-20.
    m=startup_profile_model(times,a)
    assert m['reason'][0]=='startup_sign_unresolved'
    assert not m['supported'][0]


@pytest.mark.parametrize('sign',[-1,1])
def test_both_consistent_edge_polarities_are_accepted(sign):
    m=startup_profile_model(*startup(sign=sign))
    r=profile_evidence(current(m,sign=sign),m)
    assert r['quality_mask'].all()
    np.testing.assert_array_equal(r['signed_plateau_contrast'],sign*40.)


@pytest.mark.parametrize('offset,expected',[(-53,True),(53,True),(-54,False),(54,False)])
def test_full_six_to_twelve_pixel_plateaus_required(offset,expected):
    m=model()
    r=profile_evidence(current(m,offset=offset),m)
    assert r['quality_mask'].all()==expected
    assert r['checks']['complete_profile_and_plateaus'].all()==expected


@pytest.mark.parametrize('offset',[0.25,np.nan,np.inf,1e300])
def test_non_grid_nonfinite_and_huge_offsets_stay_unknown(offset):
    m=model()
    f=current(m)
    f['normal_offset'][:]=offset
    f['normal_residual'][:]=offset
    r=profile_evidence(f,m)
    assert not r['quality_mask'].any()
    assert not r['checks']['complete_profile_and_plateaus'].any()


def test_missing_one_profile_sample_cannot_be_interpolated():
    m=model()
    f=current(m)
    f['normal_profiles'][0,10]=np.nan
    r=profile_evidence(f,m)
    np.testing.assert_array_equal(r['quality_mask'],[False,True])
    assert np.isnan(r['signed_plateau_contrast'][0])


def test_same_gradient_with_opposite_plateaus_is_rejected():
    m=model()
    f=current(m,sign=-1)
    f['normal_gradient'][:]=20.
    r=profile_evidence(f,m)
    assert r['checks']['gradient_polarity_consistent'].all()
    assert not r['checks']['plateau_polarity_consistent'].any()
    assert not r['quality_mask'].any()


def test_opposite_gradient_with_same_plateaus_is_rejected():
    m=model()
    f=current(m)
    f['normal_gradient'][:]*=-1.
    r=profile_evidence(f,m)
    assert not r['checks']['gradient_polarity_consistent'].any()
    assert r['checks']['plateau_polarity_consistent'].all()
    assert not r['quality_mask'].any()


def test_contrast_fraction_inclusive_boundary_and_absolute_floor():
    m=model()
    assert profile_evidence(current(m,contrast=20.),m)['quality_mask'].all()
    assert not profile_evidence(current(m,contrast=19.),m)['quality_mask'].any()
    m=startup_profile_model(*startup(contrast=3.))
    assert profile_evidence(current(m,contrast=2.),m)['quality_mask'].all()
    assert not profile_evidence(current(m,contrast=1.9),m)['quality_mask'].any()


@pytest.mark.parametrize('clip,expected',[(.25,True),(.25001,False),(-.01,False),(np.nan,False),(np.inf,False)])
def test_clipped_fraction_boundary_and_invalid_values(clip,expected):
    m=model()
    f=current(m)
    f['normal_clipped_fraction'][:]=clip
    r=profile_evidence(f,m)
    assert r['quality_mask'].all()==expected


@pytest.mark.parametrize('pose',[False,np.nan,2,'true'])
def test_invalid_pose_cannot_pass_by_truthiness(pose):
    m=model()
    f=current(m)
    f['pose_valid']=pose
    assert not profile_evidence(f,m)['quality_mask'].any()


@pytest.mark.parametrize('time_s',[5.,4.9,np.nan,np.inf,-1.])
def test_model_cannot_apply_to_startup_or_invalid_time_when_time_supplied(time_s):
    m=model()
    r=profile_evidence(current(m,time_s=time_s),m)
    assert r['time_checked']
    assert not r['quality_mask'].any()
    assert (r['reason']=='startup_or_nonfinite_time').all()


def test_absent_time_is_unknown_and_does_not_claim_causal_event():
    m=model()
    f=current(m)
    del f['time_s']
    r=profile_evidence(f,m)
    assert r['time_checked'] and not r['time_provided']
    assert not r['quality_mask'].any()
    assert (r['reason']=='startup_or_nonfinite_time').all()
    assert r['available_after_s']==5.
    assert not r['new_layer_confirmed']


def test_peak_ratio_boundary_and_distinct_reference_diagnostic():
    m=model()
    f=current(m)
    f['normal_peak_ratio'][:]=[1.2,1.19999]
    np.testing.assert_array_equal(profile_evidence(f,m)['quality_mask'],[True,False])
    f=current(m)
    f['normal_residual'][0]+=1.
    r=profile_evidence(f,m)
    assert r['quality_mask'].all()
    np.testing.assert_array_equal(r['profile_residual'],0.)
    np.testing.assert_array_equal(r['archived_minus_profile_residual'],[1.,0.])


def test_new_reference_uses_complete_startup_plateaus_not_archived_median():
    offsets=np.tile(np.r_[np.full(5,54.),np.arange(-10.,11.)][:,None],(1,2))
    times,a=startup(offsets=offsets)
    m=startup_profile_model(times,a)
    np.testing.assert_array_equal(m['baseline_offset'],0.)
    archived=np.median(offsets,axis=0)
    assert (archived!=m['baseline_offset']).all()
    f=current(m,offset=8.)
    f['normal_residual']=f['normal_offset']-archived
    r=profile_evidence(f,m)
    assert r['quality_mask'].all()
    np.testing.assert_array_equal(r['profile_residual'],8.)
    np.testing.assert_array_equal(r['archived_minus_profile_residual'],-archived)


def test_archive_residual_must_still_be_finite():
    m=model()
    f=current(m,offset=8.)
    f['normal_residual'][0]=np.nan
    r=profile_evidence(f,m)
    np.testing.assert_array_equal(r['quality_mask'],[False,True])
    np.testing.assert_array_equal(r['profile_residual'],8.)
    assert np.isnan(r['archived_minus_profile_residual'][0])


def test_startup_and_frame_malformed_shapes_raise_instead_of_broadcast():
    times,a=startup()
    a['normal_profiles']=a['normal_profiles'][...,1:]
    with pytest.raises(ValueError,match='shape mismatch'):
        startup_profile_model(times,a)
    m=model()
    f=current(m)
    f['normal_offset']=np.array([0.])
    with pytest.raises(ValueError,match='match model rows'):
        profile_evidence(f,m)


@pytest.mark.parametrize('times',[[0.,np.nan],[0.,np.inf],[1.,0.],[0.,0.]])
def test_invalid_timestamp_metadata_rejected(times):
    _,a=startup(2)
    with pytest.raises(ValueError,match='times must be finite'):
        startup_profile_model(times,a)


def test_frame_and_model_inputs_are_not_modified_or_updated():
    m=model()
    f=current(m)
    before_f=deepcopy(f)
    before_m=deepcopy(m)
    profile_evidence(f,m)
    for k,v in f.items():
        np.testing.assert_equal(v,before_f[k])
    for k,v in m.items():
        np.testing.assert_equal(v,before_m[k])
