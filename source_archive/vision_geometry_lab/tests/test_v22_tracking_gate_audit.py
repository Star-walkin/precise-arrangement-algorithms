import cv2
import numpy as np
import pytest

from vision_geometry_lab import v22_flange_tracks as frozen
from vision_geometry_lab import v22_tracking_gate_audit as audit


def test_waterfall_distinguishes_each_stage_without_double_counting():
    p=np.column_stack((np.arange(9)+10,np.full(9,10))).astype(float)
    q=p.copy();back=p.copy();s1=np.ones(9);s2=np.ones(9);errors=np.zeros(9)
    s1[0]=0;s2[1]=0;back[2]=np.nan;back[3]+=2;errors[4]=40
    q[5]=[200,200];q[6]=[40,10];q[7]=[19,10]
    color=np.full((100,100),255,np.uint8);color[10,19]=0
    result,kept=audit.breakdown(p,np.zeros(9,int),q,s1,back,s2,errors,[[0,0,30,30],[60,60,90,90]],color)
    assert [x['rejected_at_this_step'] for x in result['waterfall']]==[1]*8
    assert [x['retained'] for x in result['waterfall']]==list(range(8,0,-1))
    assert kept.sum()==1 and kept[-1]
    assert result['individual_failed_nonexclusive']['forward_backward_error']==2
    assert result['flow_supported_before_spatial_color']==4
    assert result['flow_supported_outside_ROI']==2
    assert result['flow_supported_inside_ROI_color_failed']==1


def test_split_color_matches_frozen_combined_ROI_masks():
    rng=np.random.default_rng(25)
    image=rng.integers(0,256,(120,160,3),np.uint8)
    rois=[[10,50,60,100],[90,50,140,100]]
    tracker=frozen.FlangeTracks(rois,(160,120))
    original=tracker._masks(image)
    color=audit.yellow_mask(image)
    for k,(x1,y1,x2,y2) in enumerate(rois):
        expected=np.zeros((120,160),np.uint8)
        expected[y1:y2,x1:x2]=color[y1:y2,x1:x2]
        assert np.array_equal(original[k],expected)


def test_pair_static_synthetic_mask_matches_original():
    rng=np.random.default_rng(83)
    hsv=np.zeros((240,320,3),np.uint8)
    hsv[:,:,0]=25;hsv[:,:,1]=140;hsv[:,:,2]=rng.integers(40,225,(240,320),np.uint8)
    image=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    r=audit.inspect_pair(image,image,[[20,130,140,210],[180,130,300,210]],0.)
    assert r['combined_mask_matches_frozen_implementation']
    assert r['initial_point_count']==200
    assert r['retained_count']==200
    assert r['flow_supported_outside_ROI']==0


def test_invalid_shapes_and_background_groups_are_rejected():
    p=np.array([[10.,10.]])
    with pytest.raises(ValueError):
        audit.breakdown(p,[2],p,[1],p,[1],[0],[[0,0,20,20]]*2,np.ones((30,30),np.uint8))
    with pytest.raises(ValueError):
        audit.breakdown(p,[0],p,[1,1],p,[1],[0],[[0,0,20,20]]*2,np.ones((30,30),np.uint8))


def test_nonfinite_metrics_are_not_reported_as_valid_displacement():
    p=np.array([[10.,10.]])
    result,kept=audit.breakdown(p,[0],[[np.nan,np.nan]],[0],[[np.nan,np.nan]],[0],[np.nan],[[0,0,20,20]]*2,np.ones((30,30),np.uint8))
    assert not kept.any()
    assert result['LK_candidate_displacement_px']['count']==0
    assert result['bidirectionally_supported_displacement_px']['count']==0
