import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_geometry_lab import v23_free_rope_audit as a


def prediction(normal=(0,1),rho=0,segment=((0,0),(100,0))):
    return dict(valid=True,normal=list(normal),rho_px=rho,observed_segment=list(segment))


def test_perfect_line_and_reversed_direction():
    for normal in ((0,1),(0,-1)):
        r=a.score(prediction(normal),[[100,0],[0,0]],'partial')
        assert r['line_fit_pass']
        assert r['angular_error_deg']==pytest.approx(0)
        assert r['observed_projection_overlap_fraction']==pytest.approx(1)


def test_five_pixel_boundary_and_error_rejection():
    assert a.score(prediction(rho=5),[[0,0],[100,0]],'partial')['line_fit_pass']
    r=a.score(prediction(rho=5.1),[[0,0],[100,0]],'partial')
    assert not r['line_fit_pass']
    assert r['normal_error_px']['p95']==pytest.approx(5.1)


def test_angle_is_independent_from_short_line_position_fit():
    theta=np.deg2rad(2.)
    r=a.score(prediction(normal=(-np.sin(theta),np.cos(theta))),[[0,0],[10,0]],'partial')
    assert r['normal_error_px']['p95']<5
    assert r['angular_error_deg']==pytest.approx(2.)
    assert not r['line_fit_pass']


def test_infinite_line_fit_does_not_imply_observed_support_overlap():
    r=a.score(prediction(segment=((200,0),(300,0))),[[0,0],[100,0]],'partial')
    assert r['line_fit_pass'] and r['line_fit_pass_without_observed_overlap']
    assert r['observed_projection_overlap_fraction']==0
    r=a.score(prediction(segment=((50,0),(150,0))),[[0,0],[100,0]],'partial')
    assert r['observed_projection_overlap_fraction']==pytest.approx(.5)


def test_absent_and_missing_prediction_are_not_zero_error_success():
    assert a.score(prediction(),[],'absent')['false_positive_candidate']
    assert not a.score({'valid':False},[],'absent')['false_positive_candidate']
    r=a.score({'valid':False,'reason':'no_support'},[[0,0],[100,0]],'partial')
    assert not r['line_fit_pass'] and r['angular_error_deg'] is None
    assert r['reason']=='no_support'


def test_arclength_sampling_handles_duplicate_vertices():
    sampled,length=a.sample_polyline([[0,0],[0,0],[10,0],[10,10]])
    assert length==20 and len(sampled)==21
    assert np.allclose(sampled[10],[10,0])
    with pytest.raises(ValueError): a.sample_polyline([[1,1],[1,1]])


def test_all_predictions_finish_before_annotation_loading(tmp_path,monkeypatch):
    image=np.zeros((40,60,3),np.uint8)
    ok,data=cv2.imencode('.png',image);assert ok
    image_path=tmp_path/'image.png';image_path.write_bytes(data.tobytes())
    samples=[]
    for i in range(20):
        label=tmp_path/f'label_{i}.json'
        label.write_text(json.dumps(dict(image_sha256=a.sha(image_path),sample_id=str(i),
            status='partial',polylines=dict(free_seed=[[10,20],[50,20]]))),encoding='utf-8')
        samples.append(dict(sample_id=str(i),image_path=str(image_path),image_sha256=a.sha(image_path),
            annotation_path=str(label),annotation_sha256=a.sha(label),split='train',group_id='synthetic',
            intended_use='geometry_train'))
    inventory=tmp_path/'inventory.json';inventory.write_text(json.dumps(dict(samples=samples)),encoding='utf-8')
    monkeypatch.setattr(a,'INVENTORY',inventory)
    calls=[];original_read=a.read
    def detector(frame):
        assert frame.shape==image.shape
        calls.append(1)
        return prediction(rho=20,segment=((10,20),(50,20)))
    def guarded_read(path):
        if Path(path).name.startswith('label_'): assert len(calls)==20
        return original_read(path)
    monkeypatch.setattr(a,'detect_rope',detector);monkeypatch.setattr(a,'read',guarded_read)
    r=a.run(tmp_path/'output')
    assert r['free_line_fit_pass_images']==20 and r['denominator_for_pass']==20
    assert (tmp_path/'output'/'predictions.json').exists()
