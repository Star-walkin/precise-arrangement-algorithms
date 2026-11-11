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
    if not len(domain) or (curve[-1,0]-curve[0,0])>=.75*np.ptp(domain):
        return surface,old,dict(meta,interface_version='ROI21',extension_pixels_within_current_surface=0,
            extension_reason='visible_outline_already_covers_three_quarters_of_current_surface'),curve
    pts=np.asarray(b.get('top_observed_xy',[]),float).reshape(-1,2)
    if len(pts)<8:return surface,old,dict(meta,interface_version='ROI21',extension_pixels_within_current_surface=0),curve
    current=np.sort(pts[:,0]);idx=np.searchsorted(current,xs);left=current[np.clip(idx-1,0,len(current)-1)];right=current[np.clip(idx,0,len(current)-1)]
    nearest=np.minimum(abs(xs-left),abs(xs-right));n=b['normalized_x'];u=(xs-n['center'])/n['scale']
    poly=np.polynomial.polynomial.polyval(u,b['fit_coefficients_ascending'])
    ensemble=np.array([np.polynomial.polynomial.polyval(u,z['coef']) for z in b.get('shape_state',[])])
    spread=np.percentile(abs(ensemble-np.median(ensemble,axis=0)),90,axis=0) if len(ensemble)>=3 else np.full(w,6.)
    unc=np.maximum(max(3.,2*float(b.get('fit_residual_MAD_px',3.))),spread)+.025*np.maximum(0.,nearest-2*PITCH_PRIOR)
    hull=(xs>=current[0])&(xs<=current[-1]);unc[~hull]+=.01*np.minimum(abs(xs[~hull]-current[0]),abs(xs[~hull]-current[-1]))
    occluded=np.zeros(w,bool)
    for lo,hi in b.get('foreground_ranges',[]):occluded|=(xs>=lo)&(xs<=hi)
    local=(nearest<=2*PITCH_PRIOR)&~occluded;unc[~local]=np.maximum(unc[~local],6.)
    in_old=(xs>=curve[0,0])&(xs<=curve[-1,0]);in_surface=np.any(surface>0,axis=0)
    allowed=np.flatnonzero(in_surface&~in_old&(unc<=2*PITCH_PRIOR))
    tops=np.full(w,np.nan);guards=np.full(w,np.nan)
    oldcols=np.flatnonzero(in_surface&in_old);tops[oldcols]=np.interp(oldcols,curve[:,0],curve[:,1]);guards[oldcols]=meta['inset_px']
    tops[allowed]=poly[allowed];guards[allowed]=np.maximum(meta['inset_px'],unc[allowed])
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
