import numpy as np
from vision_geometry_lab.geometry import rope_exclusion_band,draw_overlay


def line(width=10):
    return dict(valid=True,normal=[0,1],rho_px=50,width_px=width,residual_px=0,
                observed_segment=None,diagnostics={})


def test_parallel_corridor_uses_current_frame_width_and_margin():
    mask,info=rope_exclusion_band((100,160,3),line(),3)
    assert info['valid'] and abs(info['half_width_px']-8)<1e-8
    assert mask[42].all() and mask[58].all() and not mask[41].any()


def test_invalid_line_never_creates_stale_exclusion():
    mask,info=rope_exclusion_band((100,160,3),dict(valid=False))
    assert mask is None and not info['valid']


def test_missing_width_proxy_is_for_exclusion_not_measurement():
    r=line(None)
    r['diagnostics']={'candidate_optical_width_px':8,'width_mad_px':2}
    mask,info=rope_exclusion_band((200,200,3),r,3)
    assert info['valid'] and info['half_width_px']==9
    assert not info['physical_diameter_measured'] and r['width_px'] is None


def test_unknown_width_does_not_overpaint_orange_centerline_yellow():
    image=np.zeros((100,160,3),np.uint8)
    result={'arc':{'valid':False},'rope':line(None),'relation':{'valid':False}}
    shown=draw_overlay(image,result)
    # Orange has green channel160; old bug painted a yellow230 line at zero width.
    assert shown[50,80,1]<190 and shown[50,80,2]>180,shown[50,80]


def test_uncertain_interval_draws_dashes_not_solid_claim():
    image=np.zeros((100,160,3),np.uint8)
    r=line(None); r['width_interval_px']=[8,12]
    shown=draw_overlay(image,{'arc':{'valid':False},'rope':r,'relation':{'valid':False}})
    # Two estimated boundaries exist but gaps in each show uncertain support.
    row=shown[45,:,2]
    assert np.any(row>120) and np.any(row==0)


def test_measured_width_is_dashed_outside_observed_span():
    r=line(10); r['observed_segment']=[[50,50],[100,50]]
    shown=draw_overlay(np.zeros((100,160,3),np.uint8),{'arc':{'valid':False},'rope':r,'relation':{'valid':False}})
    assert np.all(shown[45,52:98,2]>120)
    assert np.any(shown[45,:45,2]==0)
