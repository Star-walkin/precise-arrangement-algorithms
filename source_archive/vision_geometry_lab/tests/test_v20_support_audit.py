"""Synthetic denominator controls; no archived production data or truth."""
from copy import deepcopy
import json

import numpy as np
import pytest

from vision_geometry_lab import v20_support_audit as audit


def prediction(residual=None,quality=None,supported=None):
    t=np.arange(601)/10.;rows=1900+np.arange(128)*8
    base=np.ones(128,bool) if supported is None else np.asarray(supported,bool)
    q=np.broadcast_to(base,(601,128)).copy() if quality is None else np.asarray(quality,bool).copy()
    q[t<=5.]=False
    d=np.zeros((601,128)) if residual is None else np.asarray(residual,float).copy()
    states=[{'row_y':int(y)} for y in rows]
    return dict(status='complete',session='left',video={'fps':10.},
        startup_model=dict(supported=base.tolist(),threshold_px=[3.]*128,ready_s=5.,
            profile_offset_count=[51]*128,reliable_profile_count=[51]*128),
        records=[dict(time_s=float(time),frame_index=i,quality_mask=q[i].tolist(),
            profile_residual=d[i].tolist(),tracker={'row_states':deepcopy(states)})
            for i,time in enumerate(t)],
        material_identity_confirmed=False,physical_new_layer_confirmed=False,hardware_access=False)


def scope(result,bucket,name='startup_supported_rows'):
    return result['fixed_5s_buckets'][bucket]['scopes'][name]


def test_denominator_loss_can_raise_high_valid_without_more_observed_high():
    d=np.zeros((601,128));d[:,:64]=4.
    q=np.ones((601,128),bool);q[300:,64:]=False
    r=audit.audit_prediction(prediction(d,q))
    before,after=scope(r,2),scope(r,6)
    assert before['high_given_direct_quality']==.5
    assert after['high_given_direct_quality']==1.
    assert before['observed_high_per_all_slots']==after['observed_high_per_all_slots']==.5
    assert before['observed_high_slots']==after['observed_high_slots']==3200
    assert r['posthoc_common_support']['poststartup_11_buckets_count']==64
    assert scope(r,2,'common_5_to_60_bins_posthoc')['high_given_direct_quality']==1.
    assert scope(r,6,'common_5_to_60_bins_posthoc')['high_given_direct_quality']==1.


def test_persistent_shift_survives_fixed_and_common_denominators():
    d=np.zeros((601,128));d[300:]=4.
    r=audit.audit_prediction(prediction(d))
    assert scope(r,2)['observed_high_per_all_slots']==0.
    for k in range(6,12):
        s=scope(r,k)
        assert s['high_given_direct_quality']==s['observed_high_per_all_slots']==1.
        assert s['median_direct_residual_px']==4.
    assert r['posthoc_common_support']['poststartup_11_buckets_count']==128


def test_startup_bucket_is_unknown_and_forces_literal_all_bucket_intersection_empty():
    r=audit.audit_prediction(prediction())
    first=scope(r,0)
    assert first['direct_quality_slots']==0
    assert first['high_given_direct_quality'] is None
    assert first['median_direct_residual_px'] is None
    assert first['observed_high_per_all_slots']==0.
    assert r['posthoc_common_support']['all_12_buckets_count']==0
    assert r['posthoc_common_support']['uses_future']
    assert not r['posthoc_common_support']['online_usable']


def test_fixed_bins_partition_all_601_frames_once():
    times=np.arange(601)/10.
    bins=audit.fixed_bins(times)
    assert [int(mask.sum()) for _,_,mask in bins]==[50]*11+[51]
    np.testing.assert_array_equal(np.asarray([m for _,_,m in bins]).sum(axis=0),1)
    assert not bins[0][2][50] and bins[1][2][50]
    assert bins[-1][2][-1]


def test_startup_support_keeps_the_same_original_denominator():
    base=np.ones(128,bool);base[::2]=False
    r=audit.audit_prediction(prediction(supported=base))
    assert r['startup_support']['supported_count']==64
    for b in r['fixed_5s_buckets']:
        assert b['scopes']['startup_supported_rows']['rows']==64
        assert b['scopes']['startup_supported_rows']['all_slots']==b['frames']*64
        assert len(b['per_original_row'])==128
    assert r['startup_support']['supported_original_indices']==list(range(1,128,2))


def test_spatial_partitions_are_fixed_original_index_blocks():
    d=np.zeros((601,128));d[:,31]=4.;d[:,32]=4.
    r=audit.audit_prediction(prediction(d))
    parts=r['fixed_5s_buckets'][2]['spatial_partitions']
    assert [p['original_index_range'] for p in parts]==[[0,32],[32,64],[64,96],[96,128]]
    assert [p['all_original']['observed_high_slots'] for p in parts]==[50,50,0,0]


def test_invalid_and_missing_quality_is_not_a_direct_low_or_high():
    d=np.full((601,128),50.)
    q=np.zeros((601,128),bool)
    r=audit.audit_prediction(prediction(d,q))
    for b in r['fixed_5s_buckets']:
        s=b['scopes']['startup_supported_rows']
        assert s['observed_high_slots']==0 and s['high_given_direct_quality'] is None
        assert s['median_direct_residual_px'] is None
        assert s['unknown_slots']==s['all_slots']


def test_intersection_requires_each_poststartup_bin_not_global_mean():
    q=np.ones((601,128),bool)
    q[300:326,127]=False  # 24/50 in one bin; otherwise almost always valid.
    r=audit.audit_prediction(prediction(quality=q))
    common=r['posthoc_common_support']['poststartup_11_buckets_original_indices']
    assert 127 not in common and len(common)==127


def test_posthoc_rows_can_change_when_future_coverage_changes_not_claimed_causal():
    original=audit.audit_prediction(prediction())
    q=np.ones((601,128),bool);q[550:,64:]=False
    changed=audit.audit_prediction(prediction(quality=q))
    assert original['posthoc_common_support']['poststartup_11_buckets_count']==128
    assert changed['posthoc_common_support']['poststartup_11_buckets_count']==64
    for k in range(11):
        assert scope(original,k)==scope(changed,k)


def test_bad_complete_state_coordinates_support_and_quality_fail_closed():
    p=prediction();cases=[]
    c=deepcopy(p);c['status']='started';cases.append(c)
    c=deepcopy(p);c['video']['fps']=np.nan;cases.append(c)
    c=deepcopy(p);c['records'][80]['time_s']+=.01;cases.append(c)
    c=deepcopy(p);c['records'][80]['profile_residual'][0]=None;cases.append(c)
    c=deepcopy(p);c['records'][80]['quality_mask'][0]=None;cases.append(c)
    c=deepcopy(p);c['startup_model']['supported'][0]=False;cases.append(c)
    c=deepcopy(p);c['records'][0]['quality_mask'][0]=True;cases.append(c)
    c=deepcopy(p);c['records'][80]['tracker']['row_states'][0]['row_y']+=1;cases.append(c)
    for bad in cases:
        with pytest.raises(ValueError):audit.audit_prediction(bad)


def test_archive_inputs_are_unchanged_and_result_is_strict_json():
    p=prediction();before=deepcopy(p)
    r=audit.audit_prediction(p)
    assert p==before
    encoded=json.dumps(r,allow_nan=False)
    assert 'NaN' not in encoded and 'Infinity' not in encoded


def test_synthetic_protocol_binds_sources_and_rejects_overwrite(tmp_path):
    source=tmp_path/'source';source.mkdir()
    (source/'protocol.json').write_text('{}',encoding='utf-8')
    for side in ('left','right'):
        folder=source/side;folder.mkdir()
        p=prediction();p['session']=side;p['protocol_sha256']=audit.sha(source/'protocol.json')
        (folder/'predictions.json').write_text(json.dumps(p),encoding='utf-8')
    target=tmp_path/'audit'
    audit.freeze(target,source)
    audit.verify(target)
    result=audit.run(target)
    assert set(result)=={'left','right'}
    with pytest.raises(FileExistsError):audit.run(target)
    with pytest.raises(FileExistsError):audit.freeze(target,source)
    with (source/'left/predictions.json').open('a') as f:f.write(' ')
    with pytest.raises(ValueError,match='archived prediction changed'):audit.verify(target)
