"""Synthetic-only root-driver tests: no production files, labels or hardware."""
from copy import deepcopy
import json

import numpy as np
import pytest

from vision_geometry_lab import v19_front_trial as trial


def synthetic(stop=10., growth=True):
    times=np.arange(int(round(stop*10))+1)/10.
    rows=np.array([100.,108.,116.])
    offsets=np.zeros((len(times),len(rows)))
    if growth:
        for j,onset in enumerate([6.,6.3,6.6]):
            offsets[times>=onset,j]=4.
    profile=100.+40.*(np.arange(-65,66)[None,None,:]>=offsets[...,None])
    residual=offsets.copy();residual[times<=5.]=np.nan
    return dict(times_s=times,rows_y=rows,normal_offset=offsets,
        normal_residual=residual,normal_gradient=np.full_like(offsets,20.),
        normal_peak_ratio=np.full_like(offsets,2.),normal_profiles=profile,
        normal_clipped_fraction=np.zeros_like(offsets),pose_valid=np.ones(len(times),bool))


def source_index(arrays):
    n=len(arrays['times_s'])
    return dict(output_status='complete_observations_only_no_events',
        video=dict(path='synthetic_do_not_open.avi',fps=10.,frame_count=n,
                   width=400,height=300,indices=list(range(n))),
        records=[dict(time_s=float(t),frame_index=i,frame_sha256='a'*64)
                 for i,t in enumerate(arrays['times_s'])])


def prefix(arrays,n):
    return {k:(v.copy() if k=='rows_y' else v[:n].copy()) for k,v in arrays.items()}


def fake_upstream(tmp_path,monkeypatch):
    up=tmp_path/'upstream';up.mkdir()
    (up/'protocol.json').write_text('{"synthetic":true}',encoding='utf-8')
    for side in ('left','right'):
        d=up/side;d.mkdir()
        a=synthetic()
        np.savez_compressed(d/'measurement.npz',**a)
        idx=source_index(a)
        idx['measurement_sha256']=trial.sha(d/'measurement.npz')
        idx['protocol_sha256']=trial.sha(up/'protocol.json')
        (d/'index.json').write_text(json.dumps(idx),encoding='utf-8')
    calls=[]
    def verify_mock(base):
        assert str(base)==str(up)
        calls.append(str(base))
        return {'synthetic_only':True}
    monkeypatch.setattr(trial.upstream,'verify',verify_mock)
    return up,calls


def test_ordered_candidate_preserves_fields_and_adds_scoring_aliases():
    result=trial.replay_arrays(synthetic())
    assert len(result['candidates'])==1
    c=result['candidates'][0]
    assert c['estimated_onset_s']==c['estimated_onset']==6.
    assert c['first_available_s']==c['first_available']==6.9
    assert c['confirmation_delay_s']==c['delay']==pytest.approx(.9)
    assert c['rows_y']==[100.,108.,116.]
    assert c['last_low_first_high_interval_s']==c['lastlow_firsthigh_interval']==[5.9,6.]
    assert c['first_available_frame_index']==69
    assert result['records'][69]['tracker']['new_candidates']==[c]
    assert c['actuator_command'] is None
    assert result['running_state']=='unknown'
    assert not result['physical_new_layer_confirmed']
    assert not result['mechanical_revolution_confirmed']
    assert not result['hardware_access'] and not result['deployment_allowed']


def test_warmup_never_updates_tracker_or_exposes_derived_threshold(monkeypatch):
    original=trial.FrontTracker
    observed=[]
    class Spy(original):
        def update(self,time_s,*args,**kwargs):
            observed.append(time_s)
            return super().update(time_s,*args,**kwargs)
    monkeypatch.setattr(trial,'FrontTracker',Spy)
    result=trial.replay_arrays(synthetic())
    assert min(observed)==5.1
    assert len(observed)==50
    for r in result['records'][:51]:
        assert r['tracker']['state']=='warmup'
        assert not any(r['quality_mask'])
        assert r['profile_residual']==[None,None,None]
        assert all(s['high_threshold_px'] is None for s in r['tracker']['row_states'])


@pytest.mark.parametrize('n',[1,20,51,52,64,69,70,81])
def test_replay_prefix_records_are_identical(n):
    a=synthetic()
    full=trial.replay_arrays(a)
    selected=trial.replay_arrays(a,frame_indices=np.arange(n))
    truncated=trial.replay_arrays(prefix(a,n))
    assert selected['records']==full['records'][:n]
    assert truncated['records']==full['records'][:n]
    expected=[c for c in full['candidates'] if c['first_available_frame_index']<n]
    assert selected['candidates']==truncated['candidates']==expected


def test_future_samples_cannot_change_model_or_earlier_records():
    a=synthetic()
    full=trial.replay_arrays(a)
    changed=deepcopy(a)
    for field in trial.FRAME_FIELDS:
        if field=='pose_valid':
            changed[field][81:]=False
        else:
            changed[field][81:]=np.nan
    replay=trial.replay_arrays(changed)
    assert replay['startup_model']==full['startup_model']
    assert replay['records'][:81]==full['records'][:81]


def test_selected_frame_gaps_are_not_interpolated_or_hidden():
    a=synthetic()
    ids=np.r_[np.arange(65),np.arange(70,101)]
    replay=trial.replay_arrays(a,frame_indices=ids)
    assert not replay['candidates']
    resumed=next(r for r in replay['records'] if r['frame_index']==70)
    assert resumed['tracker']['reason']=='frame_gap_reset'


def test_empty_requested_prefix_has_no_records_or_events():
    r=trial.replay_arrays(synthetic(),frame_indices=[])
    assert r['records']==[] and r['candidates']==[] and r['timings_ms']==[]


@pytest.mark.parametrize('ids',[[1,1],[2,1],[-1],[.5],[200],np.array([3,1],np.uint64)])
def test_bad_requested_indices_rejected(ids):
    with pytest.raises(ValueError):
        trial.replay_arrays(synthetic(),frame_indices=ids)


@pytest.mark.parametrize('fps',[np.nan,np.inf,0.,-1.,True])
def test_bad_video_fps_explicitly_rejected(fps):
    a=synthetic();idx=source_index(a);idx['video']['fps']=fps
    with pytest.raises(ValueError,match='invalid video fps'):
        trial.validate_source(idx,a)


def test_source_validates_timestamps_frame_indices_and_complete_state():
    a=synthetic();idx=source_index(a)
    trial.validate_source(idx,a)
    cases=[]
    bad=deepcopy(idx);bad['output_status']='started';cases.append(bad)
    bad=deepcopy(idx);bad['records'][30]['time_s']+=.01;cases.append(bad)
    bad=deepcopy(idx);bad['records'][30]['frame_index']=29;cases.append(bad)
    bad=deepcopy(idx);bad['video']['fps']=20.;cases.append(bad)
    bad=deepcopy(idx);bad['video']['indices'][30]=29;cases.append(bad)
    bad=deepcopy(idx);bad['records'][30]['frame_sha256']='not-a-sha';cases.append(bad)
    bad=deepcopy(idx);bad['records'].pop();cases.append(bad)
    for bad in cases:
        with pytest.raises(ValueError):
            trial.validate_source(bad,a)


@pytest.mark.parametrize('field',list(trial.FRAME_FIELDS))
def test_array_shapes_cannot_silently_broadcast(field):
    a=synthetic();a[field]=a[field][:-1]
    with pytest.raises(ValueError,match='invalid shape'):
        trial.replay_arrays(a)


def test_nonfinite_nonincreasing_time_or_rows_rejected():
    for key,value in [('times_s',np.nan),('times_s',0.),('rows_y',np.inf),('rows_y',100.)]:
        a=synthetic();a[key][1]=value
        with pytest.raises(ValueError):
            trial.replay_arrays(a)


def test_quality_unknown_cannot_bridge_an_ordered_chain():
    a=synthetic();a['normal_peak_ratio'][64:66,1]=1.
    r=trial.replay_arrays(a)
    assert not r['candidates']
    assert not r['records'][64]['quality_mask'][1]
    assert r['records'][64]['quality_reason'][1]=='peak_unambiguous_failed'


def test_strict_json_replaces_nonfinite_recursively_and_preserves_false():
    value=trial.strict_json(dict(x=np.array([np.nan,np.inf,-np.inf,2.]), b=np.bool_(False)))
    assert value=={'x':[None,None,None,2.],'b':False}
    encoded=json.dumps(trial.replay_arrays(synthetic()),allow_nan=False)
    assert 'NaN' not in encoded and 'Infinity' not in encoded


def test_replay_does_not_modify_input_arrays():
    a=synthetic();before=deepcopy(a)
    trial.replay_arrays(a)
    for k in a:
        np.testing.assert_equal(a[k],before[k])


def test_synthetic_freeze_verify_run_provenance_and_no_overwrite(tmp_path,monkeypatch):
    up,calls=fake_upstream(tmp_path,monkeypatch)
    dest=tmp_path/'trial'
    p=trial.freeze(dest,up)
    assert len(p['source_hashes'])==3 and len(p['dependencies'])==5
    trial.verify(dest)
    output=trial.run(dest,'left')
    assert output['status']=='complete'
    assert output['source_measurement_sha256']==trial.sha(up/'left/measurement.npz')
    assert output['source_index_sha256']==trial.sha(up/'left/index.json')
    assert output['protocol_sha256']==trial.sha(dest/'protocol.json')
    saved=json.loads((dest/'left/predictions.json').read_text(encoding='utf-8'))
    assert saved==output
    assert calls
    with pytest.raises(FileExistsError):trial.freeze(dest,up)
    with pytest.raises(FileExistsError):trial.run(dest,'left')


def test_verify_rejects_modified_frozen_source_digest_or_dependency(tmp_path,monkeypatch):
    up,_=fake_upstream(tmp_path,monkeypatch)
    dest=tmp_path/'trial';trial.freeze(dest,up)
    ip=up/'left/index.json'
    original=ip.read_text(encoding='utf-8')
    ip.write_text(original+' ',encoding='utf-8')
    with pytest.raises(ValueError,match='frozen input or source changed'):
        trial.verify(dest)
    ip.write_text(original,encoding='utf-8')
    p=json.loads((dest/'protocol.json').read_text(encoding='utf-8'))
    first=next(iter(p['source_hashes']));p['source_hashes'][first]='0'*64
    (dest/'protocol.json').write_text(json.dumps(p),encoding='utf-8')
    with pytest.raises(ValueError,match='frozen input or source changed'):
        trial.verify(dest)


def test_failed_replay_leaves_no_complete_output(tmp_path,monkeypatch):
    up,_=fake_upstream(tmp_path,monkeypatch)
    dest=tmp_path/'trial';trial.freeze(dest,up)
    def fail(_):raise ValueError('synthetic replay failure')
    monkeypatch.setattr(trial,'replay_arrays',fail)
    with pytest.raises(ValueError,match='synthetic replay failure'):
        trial.run(dest,'right')
    assert (dest/'right/started.json').exists()
    assert not (dest/'right/predictions.json').exists()
