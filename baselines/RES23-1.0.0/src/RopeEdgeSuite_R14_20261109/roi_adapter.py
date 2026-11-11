"""Pass only current conservative coil pixels to the existing 01/F3 stage."""
import cv2,numpy as np
from common import D,PITCH,plain
def build_roi(current_mask,boundary):
    mask=current_mask.copy();h,w=mask.shape
    meta=dict(spatial_roi_valid=False,count_ready=False,expanded_pixels=0,
        boundary_type=boundary.get('d_edge_semantics'),root_count_recomputed=False)
    if not boundary['fit_valid']:
        mask[:]=0;meta['reason']='boundary_fit_unavailable';return mask,meta
    curve=np.asarray(boundary['top_curve_xy']);xs=np.arange(w)
    top=np.interp(xs,curve[:,0],curve[:,1]);inside=(xs>=curve[0,0])&(xs<=curve[-1,0])
    # Model-only gaps retain an explicit conservative inset; no invented pixels.
    inset=max(2.,boundary['edge_uncertainty_px'])
    top=np.ceil(top+inset).astype(int)
    mask[:,~inside]=0
    for x in np.flatnonzero(inside):mask[:max(0,min(h,top[x])),x]=0
    for poly in boundary.get('foreground_polygons',[]):
        cv2.fillPoly(mask,[np.rint(poly).astype(np.int32)],0)
    meta['spatial_roi_valid']=bool(np.count_nonzero(mask)>.025*h*w)
    xa,xb=max(0,int(D-PITCH)),min(w,int(D+PITCH)+1)
    ya,yb=max(0,int(boundary['edge_y']+PITCH)),min(h,int(boundary['edge_y']+6*PITCH))
    coverage=float(np.mean(mask[ya:yb,xa:xb]>0)) if ya<yb else 0.
    meta.update(count_neighborhood_coverage=coverage,
        count_ready=bool(boundary['count_boundary_valid'] and coverage>=.95),
        reason='current_mask_intersected_with_fitted_boundary_and_foreground_exclusion',
        conservative_inset_px=inset,usable_pixels=int(np.count_nonzero(mask)))
    assert np.all(mask<=current_mask)
    return mask,plain(meta)
