from common20 import *
import numpy as np
from roi21 import masks
old=load('_roi21_slow_check',ROOT/'roi21_slow_snapshot.py');blank=np.zeros((3000,4096,3),np.uint8);checks=[]
for id,i in [('dual_20260923_113745__left',370),('dual_20260924_111500__left',281),('dual_20260924_111143__right',152)]:
    r=record(id,i);a=old.masks(blank,r);b=masks(blank,r)
    assert np.array_equal(a[1],b[1]),(id,i)
    assert np.allclose(a[3],b[3],atol=1e-8),(id,i)
    assert b[2]['expansion_pixels']==0
    checks.append(dict(id=id,frame=i,mask_exactly_equal=True,top_max_difference=float(np.max(abs(a[3]-b[3]))),extension_pixels=b[2]['extension_pixels_within_current_surface']))
dump(ROOT/'roi21_vector_parity.json',dict(passed=True,checks=checks));print(checks)
