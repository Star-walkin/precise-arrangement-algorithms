"""ROI21: visible outline span is not the whole visible coil's x domain.

Preserve ROI20 exactly in its existing domain. Outside it, use the current
proposal silhouette and conservative current-anchored shape estimate; never
include pixels above the uncertainty guard or inside foreground polygons.
"""
import cv2,numpy as np
from common20 import PITCH_PRIOR
from roi20 import masks as old_masks
from geometry20 import boundary_at

def masks(image,r):
    surface,old,meta=old_masks(image,r);b=r['boundary']
    if not b.get('fit_valid') or len(b.get('top_curve_xy',[]))<3:return surface,old,dict(meta,interface_version='ROI21'),np.asarray(b.get('top_curve_xy',[]))
    curve=np.asarray(b['top_curve_xy'],float);h,w=old.shape;xs=np.arange(w);domain=np.flatnonzero(np.any(surface>0,axis=0))
    tops=np.full(w,np.nan);guards=np.full(w,np.nan);allowed=[]
    for x in domain:
        if curve[0,0]<=x<=curve[-1,0]:
            tops[x]=np.interp(x,curve[:,0],curve[:,1]);guards[x]=meta['inset_px'];continue
        q=boundary_at(b,float(x))
        if q['valid'] and q['uncertainty_px']<=2*PITCH_PRIOR:
            tops[x]=q['y'];guards[x]=max(meta['inset_px'],q['uncertainty_px']);allowed.append(x)
    new=old.copy();yy=np.arange(h)
    for x in allowed:new[:,x]=np.where((surface[:,x]>0)&(yy>=tops[x]+guards[x]),255,0).astype(np.uint8)
    for poly in b.get('foreground_polygons',[]):cv2.fillPoly(new,[np.rint(poly).astype(np.int32)],0)
    added=int(np.count_nonzero((new>0)&(old==0)));inside=np.flatnonzero(np.isfinite(tops))
    # Include holes in the vector domain only as model geometry. The surface
    # and exclusion masks still retain all unknown/occluded pixels.
    top=curve if added==0 else np.c_[inside,tops[inside]+guards[inside]]
    result=dict(meta,interface_version='ROI21',valid=bool(np.count_nonzero(new)>.025*h*w),pixels=int(np.count_nonzero(new)),
        extension_pixels_within_current_surface=added,old_visible_outline_span=curve[[0,-1],0].tolist(),
        conservative_inferred_columns=len(allowed),model_outline_is_not_current_local_edge_evidence=True,
        expansion_pixels=int(np.count_nonzero((new>0)&(surface==0))))
    assert result['expansion_pixels']==0 and not np.any((old>0)&(new==0))
    return surface,new,result,top
