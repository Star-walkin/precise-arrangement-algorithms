import numpy as np
from vision_geometry_lab.render_centerline_model_v5 import heat_overlay,panel


def test_probability_display_is_fixed_scale_and_preserves_output_shape():
    image=np.full((90,200,3),50,np.uint8)
    zero=heat_overlay(image,np.zeros((24,32),np.float32))
    one=heat_overlay(image,np.ones((24,32),np.float32))
    assert zero.shape==one.shape==image.shape
    assert not np.array_equal(zero,one)
    assert panel(image,'test').shape==(436,720,3)


def test_probability_display_rejects_bad_values():
    image=np.zeros((20,20,3),np.uint8)
    for value in (float('nan'),-.1,1.1):
        try:heat_overlay(image,np.full((12,12),value,np.float32))
        except ValueError:pass
        else:raise AssertionError('Invalid probability accepted')
