import numpy as np

from vision_geometry_lab.v10_ridge_benchmark import full_image_coverage
from vision_geometry_lab.evaluate_centerline_multiscale_v9 import score_decoded


def test_strip_coverage_half_pixel_mapping():
    trace = dict(geometry=dict(origin_xy=[0,0],d_xy=[1,0],n_xy=[0,1],
                 resolution=2,u_min=0,v_min=0),
                 ridge_maps=dict(observation_mask=np.ones((8,8),bool)))
    mask = full_image_coverage(trace,(12,12,3))
    assert mask.shape == (6,6) and mask.all()
    trace['ridge_maps']['observation_mask'][:3,:] = False
    mask = full_image_coverage(trace,(12,12,3))
    assert not mask[:2].any() and mask[-1].all()


def test_rotated_strip_coverage_is_not_score_crop():
    trace = dict(geometry=dict(origin_xy=[10,0],d_xy=[0,1],n_xy=[-1,0],
                 resolution=2,u_min=0,v_min=0),
                 ridge_maps=dict(observation_mask=np.zeros((6,6),bool)))
    mask = full_image_coverage(trace,(12,12,3))
    assert mask.sum() == 0
    label = dict(status='partial',polylines=dict(centerline=[[2,2],[9,2]],free_seed=[[0,2],[2,2]]),ignore_polygons=[])
    quality = dict(precision_exclude_polygons=[])
    decoded = dict(segments=[],identity_confirmed=False)
    score = score_decoded(label,quality,decoded,mask,[12,12])
    assert score['metrics']['downstream']['reference_length_px'] > 0
    assert score['metrics']['downstream']['covered_fraction'] == 0
    assert score['coverage_support']['downstream']['fraction'] == 0
