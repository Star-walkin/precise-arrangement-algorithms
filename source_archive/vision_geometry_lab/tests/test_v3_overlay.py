import numpy as np
from vision_geometry_lab.geometry import draw_overlay


def test_local_boundary_survives_failed_global_ellipse_and_does_not_bridge():
    frame=np.zeros((100,180,3),np.uint8)
    arc=dict(valid=True,points=[],local_boundary_valid=True,
             local_boundary_segments=[[[10,50],[50,50]],[[100,50],[160,50]]])
    result=dict(arc=arc,rope=dict(valid=False),conic=dict(valid=False),relation=dict(valid=False))
    image=draw_overlay(frame,result)
    assert image[50,30,1]>200 and image[50,130,1]>200
    assert not image[:,60:90].any()


def test_optional_ellipse_is_not_shown_as_measured_local_edge():
    frame=np.zeros((100,180,3),np.uint8)
    result=dict(arc=dict(valid=True,points=[],local_boundary_segments=[]),rope=dict(valid=False),
                conic=dict(valid=True,model_points=[[10,50],[40,50],[70,50],[100,50]],model_supported=[True]*4),
                relation=dict(valid=False))
    assert not draw_overlay(frame,result).any()
    image=draw_overlay(frame,result,show_conic=True)
    assert image.any() and image[50,20,0]>image[50,20,1]  # light blue, not measured green


def test_wound_candidates_do_not_draw_unsupported_bridges():
    frame=np.zeros((100,180,3),np.uint8)
    result=dict(arc=dict(valid=False),rope=dict(valid=False),relation=dict(valid=False),
                wound_curve=dict(identity_confirmed=False,candidates=[dict(points=[[10,50],[160,50]],
                  observed_segments=[[[10,50],[40,50]],[[100,50],[160,50]]])]))
    shown=draw_overlay(frame,result)
    assert shown[50,20].any() and shown[50,130].any()
    assert not shown[:,50:90].any()
    assert not draw_overlay(frame,result,show_wound=False).any()
