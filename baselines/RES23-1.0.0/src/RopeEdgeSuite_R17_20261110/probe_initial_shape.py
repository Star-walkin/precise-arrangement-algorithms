import numpy as np
from common import *
from boundary_bg9 import design,robust_fit
rows=[]
for i in [0,3,5,12,20,23,24]:
    b=record(ROOT/'runs'/'dual_20260923_113305__right',i)['boundary']
    excluded={tuple(p) for p in b['excluded_candidates_xy']}
    pts=np.array([p for p in b['raw_candidates_xy'] if tuple(p) not in excluded])
    X=design(pts[:,0],4096);coef,keep=robust_fit(X,pts[:,1]);hull=pts[keep,0]
    u=design(np.linspace(hull.min(),hull.max(),100),4096)[:,1]
    curvature=2*coef[2]+6*coef[3]*u+12*coef[4]*u*u
    q=np.linalg.lstsq(X[keep,:3],pts[keep,1],rcond=None)[0]
    row=dict(frame=i,current_min_curvature=b['shape_observability']['min_curvature'],
        accepted_hull_min_curvature=float(curvature.min()),accepted_hull=[float(hull.min()),float(hull.max())],
        quadratic_residual_q90=float(np.quantile(abs(pts[keep,1]-X[keep,:3]@q),.9)),quadratic_curvature=2*float(q[2]))
    rows.append(row);print(row)
dump(ROOT/'initial_shape_hypothesis.json',rows)
