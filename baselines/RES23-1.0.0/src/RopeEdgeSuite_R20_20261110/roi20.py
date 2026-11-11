import cv2,numpy as np
from common20 import PITCH_PRIOR

def masks(image,r):
    surface=np.zeros(image.shape[:2],np.uint8)
    for p in r['proposal'].get('surface_contours_xy',[]):
        cv2.fillPoly(surface,[np.rint(p).astype(np.int32)],255)
    f3=surface.copy();boundary=r['boundary'];meta=dict(valid=False,reason='boundary_unavailable')
    if not boundary['fit_valid']:f3[:]=0;return surface,f3,meta
    curve=np.asarray(boundary['top_curve_xy'],float)
    if len(curve)<3:f3[:]=0;return surface,f3,meta
    h,w=f3.shape;xx=np.arange(w);yy=np.arange(h)[:,None]
    inset=max(2.,float(boundary.get('fit_residual_MAD_px') or 3.))
    top=np.interp(xx,curve[:,0],curve[:,1])+inset
    keep=(xx>=curve[0,0])&(xx<=curve[-1,0])
    f3[(yy<top[None,:])|(~keep[None,:])]=0
    for p in boundary.get('foreground_polygons',[]):cv2.fillPoly(f3,[np.rint(p).astype(np.int32)],0)
    meta=dict(valid=bool(np.count_nonzero(f3)>.025*h*w),reason='current_surface_intersection_current_fitted_top_and_foreground',
        inset_px=inset,pixels=int(np.count_nonzero(f3)),expansion_pixels=int(np.count_nonzero((f3>0)&(surface==0))))
    return surface,f3,meta
