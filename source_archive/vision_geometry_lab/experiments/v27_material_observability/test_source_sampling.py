import importlib.util
from pathlib import Path
import math
import pytest

spec=importlib.util.spec_from_file_location('v27_sampling',Path(__file__).with_name('source_sampling.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_no_physical_t2():
    x=m.conditional_period(.3,.5,.004)
    assert x['t2_available'] is False
    assert x['T_s']==pytest.approx(math.hypot(2*math.pi*.3,.004)/.5)

def test_sensitivity():
    a=m.conditional_period(.3,.5,.004)
    b=m.conditional_period(.300001,.5,.004)
    assert (b['T_s']-a['T_s'])/1e-6==pytest.approx(a['dT_dR'],rel=1e-5)

@pytest.mark.parametrize('r,v',[(0,1),(1,0),(-1,1),(1,float('nan'))])
def test_bad_inputs(r,v):
    with pytest.raises(ValueError):m.conditional_period(r,v)

def test_receive_clock_rejects_duplicate(tmp_path):
    p=tmp_path/'sample.csv'
    p.write_text('host_time_s,source_frame,device_ticks,width,height\n1,2,3,4096,3000\n1,3,4,4096,3000\n')
    with pytest.raises(ValueError):m.audit_csv(p)

def test_camera_fps_not_from_saved_fps(tmp_path):
    p=tmp_path/'sample.csv'
    p.write_text('host_time_s,source_frame,device_ticks,width,height\n1,2,3,4096,3000\n1.1,5,4,4096,3000\n1.2,8,5,4096,3000\n')
    result=m.audit_csv(p)
    assert result['saved_effective_fps']==pytest.approx(10)
    assert result['actual_camera_acquisition_fps'] is None
    assert result['metric_velocity_m_s'] is None
