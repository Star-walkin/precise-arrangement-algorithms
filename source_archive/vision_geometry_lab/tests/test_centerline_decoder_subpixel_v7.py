import inspect
import json

import numpy as np

from vision_geometry_lab.centerline_benchmark_v5 import decode_heatmaps
from vision_geometry_lab.centerline_decoder_subpixel_v7 import decode_heatmaps_subpixel,interpolated_body_width


def _maps(halfwidth=6):
    yy,xx=np.mgrid[:120,:180]
    body=np.where(abs(yy-60)<=halfwidth,.9,.1).astype(np.float32)
    center=(.1+.8*np.exp(-.5*((yy-60)/1.5)**2)).astype(np.float32)
    return body,center


def _decode(body,center,mode='subpixel',roi=(0,0,180,120),seed=((10,60),(30,60)),width=12):
    result=decode_heatmaps_subpixel(body,center,roi,seed,width,width_mode=mode)
    json.dumps(result,allow_nan=False)
    if mode=='integer':
        assert result==decode_heatmaps(body,center,roi,seed,width)
    return result


def test_soft_edges_recover_known_crossing_width_and_slope_correction():
    x=np.arange(30,dtype=float)
    p=np.clip(.5+np.minimum(x-5.25,19.65-x)/3,0,1)
    support=np.flatnonzero(p>=.5)
    assert np.isclose(interpolated_body_width(p,support[0],support[-1]),14.4,atol=1e-12)
    assert np.isclose(interpolated_body_width(p,support[0],support[-1],.75),14.4/1.25,atol=1e-12)


def test_flat_top_and_exact_half_endpoint_are_well_defined():
    p=np.array([.1,.5,.9,.9,.5,.1])
    assert interpolated_body_width(p,1,4)==3.
    p=np.array([.1,.9,.9,.9,.1])
    assert np.isclose(interpolated_body_width(p,1,3),3.)


def test_missing_flanks_and_broad_plateaus_remain_rejected():
    for p,a,b in (([.9,.9,.1],0,1),([.1,.9,.9],1,2),([.9,.9,.9,.9],1,2)):
        assert interpolated_body_width(p,a,b) is None
    body,center=_maps()
    assert not _decode(np.ones_like(body)*.9,center)['segments']
    assert not _decode(np.ones_like(body)*.9,center,'integer')['segments']


def test_decoder_first_soft_width_is_known_at_native_resolution():
    body,center=_maps();yy=np.arange(120,dtype=float)[:,None]
    body[:]=np.clip(.5+np.minimum(yy-53.25,66.65-yy)/3,0,1)
    result=_decode(body,center,width=12)
    assert np.isclose(result['widths'][0],13.4,atol=2e-6)
    control=_decode(body,center,'integer',width=12)
    assert control['widths'][0]==12.


def test_blank_flat_center_and_under_body_return_no_candidate():
    body,center=_maps()
    for mode in ('integer','subpixel'):
        assert not _decode(body,center*.1,mode)['segments']
        assert not _decode(body,np.full_like(center,.9),mode)['segments']
        assert not _decode(body*.1,center,mode)['segments']


def test_width_gates_not_removed_in_subpixel_mode():
    for mode in ('integer','subpixel'):
        assert not _decode(*_maps(halfwidth=1),mode)['segments']
        assert not _decode(*_maps(halfwidth=16),mode)['segments']


def test_competing_peaks_still_stop_without_fabricating_path():
    body,_=_maps(halfwidth=8);yy,xx=np.mgrid[:120,:180]
    center=(.1+.8*np.maximum(np.exp(-.5*(yy-56)**2),np.exp(-.5*(yy-64)**2))).astype(np.float32)
    for mode in ('integer','subpixel'):
        result=_decode(body,center,mode)
        assert result['reason']=='competing_ridge_stop' and not result['segments']


def test_roi_edge_cannot_be_used_as_missing_body_flank():
    body,center=_maps();body[:]=.1;body[:8]=.9
    yy=np.arange(120,dtype=float)[:,None]
    center[:]=.1+.8*np.exp(-.5*((yy-3)/1.5)**2)
    for mode in ('integer','subpixel'):
        assert not _decode(body,center,mode,seed=((10,3),(30,3)))['segments']


def test_integer_parity_scaled_roi_reverse_seed_and_guards():
    body,center=_maps()
    _decode(body,center,'integer',roi=(100,200,460,440),seed=((120,320),(160,320)),width=24)
    _decode(body,center,'integer',seed=((160,60),(140,60)))
    for width in (None,float('nan'),0,-1):
        assert _decode(body,center,'integer',width=width)['reason']=='independent_free_width_unavailable'
    assert _decode(body,center,'integer',seed=[])['reason']=='free_seed_required'


def test_integer_mode_exact_randomized_parity_and_inputs_unchanged():
    rng=np.random.default_rng(20260928)
    for _ in range(100):
        body=rng.random((30,50),dtype=np.float32);center=rng.random((30,50),dtype=np.float32)
        before=(body.copy(),center.copy())
        angle=rng.uniform(-1,1);d=np.array([np.cos(angle),np.sin(angle)])
        endpoint=np.array([rng.uniform(8,40),rng.uniform(8,22)])
        args=dict(roi=(0,0,50,30),seed=[endpoint-6*d,endpoint],width=rng.uniform(2,12))
        _decode(body,center,'integer',**args)
        _decode(body,center,'subpixel',**args)
        assert np.array_equal(body,before[0]) and np.array_equal(center,before[1])


def test_no_reference_or_quality_inputs_and_invalid_mode_rejected():
    assert list(inspect.signature(decode_heatmaps_subpixel).parameters)==['body','center','roi','free_seed','width_prior','width_mode']
    body,center=_maps()
    try:
        _decode(body,center,'tuned')
    except ValueError:
        pass
    else:
        raise AssertionError('Unregistered mode accepted')


def test_bad_probability_arrays_rejected_before_path_selection():
    body,center=_maps()
    for mode in ('integer','subpixel'):
        for bad in (center+1,center*np.nan):
            try:
                _decode(body,bad,mode)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid probabilities accepted')
