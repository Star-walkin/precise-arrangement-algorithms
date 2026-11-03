import json

import numpy as np

from vision_geometry_lab.centerline_benchmark_v5 import decode_heatmaps
from vision_geometry_lab.centerline_decoder_trace_v7 import trace_decode_heatmaps


def _maps(body_half=6,offset=0):
    yy,xx=np.mgrid[:120,:180]
    body=np.where(abs(yy-60-offset)<=body_half,.9,.1).astype(np.float32)
    center=(.1+.8*np.exp(-.5*((yy-60-offset)/1.5)**2)).astype(np.float32)
    return body,center


def _check(body,center,roi=(0,0,180,120),seed=((10,60),(30,60)),width=12):
    result=trace_decode_heatmaps(body,center,roi,seed,width)
    assert result['decoded']==decode_heatmaps(body,center,roi,seed,width)
    json.dumps(result,allow_nan=False)
    return result['trace']


def test_trace_clear_band_keeps_all_original_points_and_map_stop():
    trace=_check(*_maps())
    assert trace['accepted_point_count']>20
    assert trace['first_stop']['reason']=='fewer_than_5_map_supported_samples'


def test_trace_no_peak_is_distinct_from_no_body():
    body,center=_maps()
    trace=_check(body,np.full_like(center,.1))
    assert trace['first_stop']['reason']=='no_center_local_peak_at_half'
    trace=_check(np.full_like(body,.1),center)
    assert trace['first_stop']['peak_rejection_counts']=={'body_probability_below_half':1}


def test_trace_flat_above_threshold_map_still_requires_a_local_peak():
    body,center=_maps()
    trace=_check(body,np.full_like(center,.9))
    assert trace['first_stop']['reason']=='no_center_local_peak_at_half'
    assert trace['steps'][0]['joint_above_half_sample_count']>0


def test_trace_search_endpoint_response_is_not_an_interior_peak():
    body,center=_maps()
    center[:]=np.linspace(.9,.5,120,dtype=np.float32)[:,None]
    trace=_check(body,center)
    assert trace['first_stop']['reason']=='no_center_local_peak_at_half'
    assert trace['steps'][0]['center_max_inside_sample_index']==0
    assert trace['steps'][0]['center_at_search_endpoints'][0]>.5


def test_trace_unbounded_body_is_not_a_width_measurement():
    body,center=_maps()
    trace=_check(np.full_like(body,.9),center)
    assert trace['first_stop']['peak_rejection_counts']=={'body_interval_reaches_negative_search_limit':1}


def test_trace_narrow_and_wide_body_gates():
    narrow=_check(*_maps(body_half=1))
    wide=_check(*_maps(body_half=16))
    assert narrow['first_stop']['peak_rejection_counts']=={'body_width_below_0.45_prior':1}
    assert wide['first_stop']['peak_rejection_counts']=={'body_width_above_2.0_prior':1}


def test_trace_offset_gate_comes_before_interval_gate():
    trace=_check(*_maps(body_half=1,offset=14))
    assert trace['first_stop']['peak_rejection_counts']=={'peak_offset_exceeds_0.75_propagated_width':1}


def test_trace_competing_peaks_preserves_legacy_competition_exit():
    body,_=_maps(body_half=8)
    yy,xx=np.mgrid[:120,:180]
    center=(.1+.8*np.maximum(np.exp(-.5*((yy-56)/1.)**2),np.exp(-.5*((yy-64)/1.)**2))).astype(np.float32)
    trace=_check(body,center)
    assert trace['first_stop']['reason']=='competing_ridge_stop'
    assert trace['first_stop']['eligible_candidate_count']==2


def test_trace_exact_parity_resized_nonzero_roi_and_reverse_seed():
    body,center=_maps()
    _check(body,center,roi=(100,200,460,440),seed=((120,320),(160,320)),width=24)
    _check(body,center,seed=((160,60),(140,60)))


def test_trace_exact_parity_guard_exits_and_invalid_arrays():
    body,center=_maps()
    assert _check(body,center,width=None)['first_stop']['reason']=='independent_free_width_unavailable'
    assert _check(body,center,width=float('nan'))['first_stop']['reason']=='independent_free_width_unavailable'
    assert _check(body,center,seed=((20,60),(21,60)))['first_stop']['reason']=='free_seed_too_short'
    assert _check(body,center,seed=[])['first_stop']['reason']=='free_seed_required'
    for decoder in (decode_heatmaps,trace_decode_heatmaps):
        try:
            decoder(body,center+1,(0,0,180,120),((10,60),(30,60)),12)
        except ValueError:
            pass
        else:
            raise AssertionError('Out-of-range array was not rejected')


def test_trace_randomized_exact_parity_and_no_input_mutation():
    rng=np.random.default_rng(20260927)
    for index in range(80):
        body=rng.random((30,50),dtype=np.float32)
        center=rng.random((30,50),dtype=np.float32)
        prior_body=body.copy();prior_center=center.copy()
        angle=rng.uniform(-1,1);d=np.array([np.cos(angle),np.sin(angle)])
        endpoint=np.array([rng.uniform(8,40),rng.uniform(8,22)])
        _check(body,center,roi=(0,0,50,30),seed=[endpoint-6*d,endpoint],width=rng.uniform(2,12))
        assert np.array_equal(body,prior_body) and np.array_equal(center,prior_center)


def test_trace_has_no_target_reference_or_quality_inputs():
    import inspect
    assert list(inspect.signature(trace_decode_heatmaps).parameters)==['body','center','roi','free_seed','width_prior']
    trace=_check(*_maps())
    assert not trace['target_reference_used'] and not trace['quality_used']
