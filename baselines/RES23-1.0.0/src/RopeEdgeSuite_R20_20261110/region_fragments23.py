"""BG5.1: BG5 plus observed yellow fragments split by specular reflections.

Scene search rectangle is a prior, never an accepted mask. No saved SAM mask,
rope-edge detector, count, future frame or previous accepted mask is used here.
Image-bottom truncation and unobserved lateral margins remain explicit.
"""
from __future__ import annotations
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, median_filter
from scipy.interpolate import PchipInterpolator

PARAMETERS = dict(scale=.25, yellow_chroma=.45, yellow_min_light=18.,
                  minimum_yellow_area_fraction=.018, minimum_surface_area_fraction=.025,
                  minimum_top_support_fraction=.65, max_top_gap_pitch=3.,
                  min_local_support_fraction=.70, maximum_local_residual_pitch=.20,
                  search_rect_fraction=[0.,.50,.90,1.], texture_std_floor=1.1,
                  minimum_flange_interface_pairs=20,minimum_flange_pair_span_pitch=12.,
                  minimum_flange_adjacency_fraction=.60,
                  minimum_local_flange_pairs=6)

def yellow_fragments(mask,pitch,scale):
    """Associate sizeable current components in the main flange's height band.

    No bridge pixels are painted. This is only a proposal: all existing
    texture, interface, d-bracketing and residual gates still have to pass.
    """
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8)
    diag=dict(method='current_component_height_band_union_without_painting_gaps',used=False,selected=[])
    if n<=1:return np.zeros_like(mask),diag
    main=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));area_min=.002*mask.size
    ytol=max(.75*stats[main,cv2.CC_STAT_HEIGHT],6*pitch*scale)
    # Side fragments may be separated by a white highlight across the center;
    # the lower hand/tool region is outside the primary component height band.
    selected=[k for k in range(1,n) if stats[k,cv2.CC_STAT_AREA]>=area_min and abs(centers[k,1]-centers[main,1])<=ytol]
    out=(np.isin(labels,selected).astype(np.uint8)*255)
    diag.update(used=len(selected)>1,selected=[dict(component=k,area_processing_px=int(stats[k,cv2.CC_STAT_AREA]),
                center_processing_xy=centers[k].tolist(),bbox_processing=stats[k,:4].tolist()) for k in selected],
                area_min_processing_px=area_min,vertical_center_tolerance_processing_px=float(ytol),
                unsupported_white_pixels_added=0)
    return out,diag

def flange_presence(outer,accepted,pitch,yellow_small,scale,d):
    """Validate connected yellow material directly above the observed coil edge.

    This does NOT upgrade a missing outer edge at d to a measured edge. The
    outside rim is a separate diagnostic, and cannot validate a coil region.
    """
    r=dict(valid=False,reason='global_flange_structure_insufficient',
           outer_local_valid=bool(outer.get('valid')),outer_local_reason=outer.get('reason'),
           semantics='current_yellow_material_above_current_coil_interface_not_outer_edge_at_d')
    if len(accepted)<20:return r
    hs,ws=yellow_small.shape;keep=[]
    for x,y in accepted:
        xa=max(0,int((x-.3*pitch)*scale));xb=min(ws,int((x+.3*pitch)*scale)+1)
        ya=max(0,int((y-2*pitch)*scale));yb=min(hs,int((y-.35*pitch)*scale)+1)
        patch=yellow_small[ya:yb,xa:xb]
        keep.append(bool(patch.size and np.mean(patch>0)>=.35))
    q=accepted[np.asarray(keep)];local=q[abs(q[:,0]-d)<=2.5*pitch]
    fraction=float(np.mean(keep));pairspan=float(np.ptp(q[:,0])) if len(q) else 0.
    local_bracket=bool(len(local)>=PARAMETERS['minimum_local_flange_pairs'] and
                       local[:,0].min()<d-pitch and local[:,0].max()>d+pitch)
    r.update(paired_points=len(q),pair_span_px=pairspan,adjacency_fraction=fraction,
             local_paired_points=len(local),local_bracket=local_bracket)
    r['valid']=bool(len(q)>=PARAMETERS['minimum_flange_interface_pairs'] and
                    fraction>=PARAMETERS['minimum_flange_adjacency_fraction'] and
                    pairspan>=PARAMETERS['minimum_flange_pair_span_pitch']*pitch and local_bracket)
    if r['valid']:r['reason']=None
    return r

def largest(mask):
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    if n<=1:return np.zeros_like(mask,np.uint8)
    k=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    return (labels==k).astype(np.uint8)*255

def runs(a):
    e=np.diff(np.r_[False,np.asarray(a,bool),False].astype(np.int8))
    return list(zip(np.flatnonzero(e==1),np.flatnonzero(e==-1)))

def contours(mask):
    cc,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return [c[:,0,:].tolist() for c in sorted(cc,key=cv2.contourArea,reverse=True)[:2]]

def outer_rim(image,yellow_small,scale,pitch,d):
    """Fit a proposal to yellow top, then require fresh luminance-edge evidence.

    Specular white rim pixels are accepted by a strong current dark-to-bright
    interface, not painted in solely because a polynomial crosses them.
    """
    h,w=image.shape[:2];ys,xs=np.nonzero(yellow_small)
    lo,hi=int(xs.min()),int(xs.max());ww=hi-lo
    cols=np.arange(int(lo+.16*ww),int(lo+.88*ww),2)
    raw=np.asarray([[x/scale,np.flatnonzero(yellow_small[:,x])[0]/scale] for x in cols if np.any(yellow_small[:,x])],float)
    if len(raw)<20:return dict(valid=False,points_xy=[],reason='outer_rim_seed_insufficient')
    center=float(raw[:,0].mean());span=float(np.ptp(raw[:,0])/2);u=(raw[:,0]-center)/span
    rng=np.random.default_rng(810);best=None
    for _ in range(100):
        ii=rng.choice(len(raw),4,replace=False);p=np.polyfit(u[ii],raw[ii,1],2)
        if p[0]<0:continue
        res=raw[:,1]-np.polyval(p,u);keep=abs(res)<.65*pitch
        if best is None or keep.sum()>best[0]:best=(int(keep.sum()),p,keep)
    if best is None:return dict(valid=False,points_xy=[],reason='outer_rim_shape_invalid')
    p=np.polyfit(u[best[2]],raw[best[2],1],2)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(float);points=[]
    for x in range(int(raw[:,0].min()),int(raw[:,0].max())+1,8):
        yp=float(np.polyval(p,(x-center)/span));prof=np.median(gray[:,x-4:x+5],axis=1)
        pp=np.r_[0.,np.cumsum(prof)];y=np.arange(max(25,int(yp-2*pitch)),min(h-25,int(yp+2*pitch)))
        z=max(8,int(.6*pitch));a=3
        above=(pp[y-a]-pp[y-z])/(z-a);below=(pp[y+z]-pp[y+a])/(z-a)
        score=below-above;valid=(score>8)&(below>above*1.18)
        if not np.any(valid):continue
        k=int(np.argmax(np.where(valid,score,-100)));points.append([x,int(y[k])])
    pts=np.asarray(points,float).reshape(-1,2)
    if len(pts)<20:return dict(valid=False,points_xy=points,reason='outer_rim_current_edge_insufficient')
    keep=np.ones(len(pts),bool)
    for _ in range(3):
        c=np.polyfit((pts[keep,0]-center)/span,pts[keep,1],2)
        res=pts[:,1]-np.polyval(c,(pts[:,0]-center)/span)
        keep=abs(res)<.75*pitch
    pts=pts[keep];local=pts[abs(pts[:,0]-d)<3*pitch]
    good=len(pts)>len(raw)*.6 and len(local)>=6 and local[:,0].min()<d-pitch and local[:,0].max()>d+pitch
    return dict(valid=bool(good),points_xy=pts.tolist(),reason=None if good else 'outer_rim_local_evidence_incomplete',
                semantics='observed_upper_outer_rim_not_count_origin')

def segment(image,d=2238.7,pitch=23.29412648284868):
    if image.dtype!=np.uint8 or image.ndim!=3:raise ValueError('BGR uint8 required')
    h,w=image.shape[:2];s=PARAMETERS['scale'];small=cv2.resize(image,None,fx=s,fy=s,interpolation=cv2.INTER_AREA)
    hs,ws=small.shape[:2];b,g,r=cv2.split(small.astype(np.float32))
    light=(r+g+b)/3;chroma=((r+g)/2-b)/np.maximum(light,12)
    roi=np.zeros((hs,ws),np.uint8);roi[int(.50*hs):,:int(.90*ws)]=255
    yellow=((chroma>.45)&(g>.72*r)&(light>18)&(roi>0)).astype(np.uint8)*255
    yellow=cv2.morphologyEx(yellow,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    all_yellow_components=yellow.copy()
    yellow,selection=yellow_fragments(all_yellow_components,pitch,s)
    # Fill only small holes. A large surface notch must remain outside yellow.
    filled=binary_fill_holes(yellow>0);holes=(filled&(yellow==0)).astype(np.uint8)
    n,lab,stats,_=cv2.connectedComponentsWithStats(holes,8)
    for k in range(1,n):
        if stats[k,cv2.CC_STAT_AREA]<350:yellow[lab==k]=255
    gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean=cv2.boxFilter(gray,-1,(9,13));var=np.maximum(0,cv2.boxFilter(gray*gray,-1,(9,13))-mean*mean)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3,scale=.125);gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3,scale=.125)
    xx=cv2.boxFilter(gx*gx,-1,(11,15));yy=cv2.boxFilter(gy*gy,-1,(11,15));xy=cv2.boxFilter(gx*gy,-1,(11,15))
    anis=np.sqrt((xx-yy)**2+4*xy*xy)/(xx+yy+1e-5)
    horizontal=(yy-xx)/(xx+yy+1e-5)
    texture=(np.sqrt(var)>1.1)&(anis>.23)&(horizontal>-.10)&(gray>6)
    rec=dict(version='BG5_current_fragments_R20_23',parameters=PARAMETERS.copy(),valid=False,
             reasons=[],edge_y=None,edge_uncertainty_px=None,top_observed_xy=[],top_curve_xy=[],
             bottom_is_image_crop=True,physical_boundary_accuracy_validated=False,
             no_static_surface_mask=True,no_temporal_mask_hold=True)
    rec['yellow_fragment_association']=selection
    out=dict(record=rec,yellow_mask=np.zeros((h,w),np.uint8),surface_mask=np.zeros((h,w),np.uint8))
    ys,xs=np.nonzero(yellow)
    if len(xs)<.018*hs*ws:
        yellow,selection=yellow_fragments(all_yellow_components,pitch,s)
        rec['yellow_fragment_association']=selection
        ys,xs=np.nonzero(yellow)
    if len(xs)<.018*hs*ws:
        rec['reasons']=['insufficient_yellow_flange'];return out
    fx0,fx1=int(xs.min()),int(xs.max());fy0,fy1=int(ys.min()),int(ys.max())
    candidate=(texture&(chroma<.50)&(roi>0)).astype(np.uint8)*255
    candidate[:int(fy0+.35*(fy1-fy0))]=0
    candidate[:,:fx0+8]=0;candidate[:,fx1-8:]=0
    candidate[yellow>0]=0
    candidate=cv2.morphologyEx(candidate,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    candidate=cv2.morphologyEx(candidate,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    candidate=largest(candidate)
    candidate=(binary_fill_holes(candidate>0).astype(np.uint8)*255)
    # No filled region may include strong yellow pixels.
    candidate[yellow>0]=0
    sy,sx=np.nonzero(candidate)
    out['yellow_mask']=cv2.resize(yellow,(w,h),interpolation=cv2.INTER_NEAREST)
    rec['yellow_contours_xy']=contours(out['yellow_mask'])
    rec['yellow_area_px']=int(np.count_nonzero(out['yellow_mask']))
    outer=outer_rim(image,yellow,s,pitch,d);rec['outer_rim']=outer
    # Do not paint specular white into the yellow mask from an outer-rim
    # proposal: a bright incoming rope can fool that independent proposal.
    if len(sx)<.025*hs*ws:
        rec['reasons']=['insufficient_connected_surface_texture'];return out
    # Test the top profile only where the coarse surface and yellow meet.
    x0,x1=int(np.quantile(sx,.01)/s),int(np.quantile(sx,.99)/s)
    fimage=image.astype(np.float32);bb,gg,rr=cv2.split(fimage)
    ll=(rr+gg+bb)/3;cc=((rr+gg)/2-bb)/np.maximum(ll,12)
    # Horizontal median suppresses cable strands; no line is synthesized here.
    obs=[];tested=[];strength=[]
    for x in range(x0,x1+1,8):
        j=min(ws-1,int(x*s));col=np.flatnonzero(candidate[:,j])
        if not len(col):continue
        yprior=float(col.min()/s)
        yl=max(2,int(yprior-2.3*pitch));yh=min(h-2,int(yprior+1.2*pitch))
        if yh-yl<5:continue
        crop=cc[:,max(0,x-5):min(w,x+6)];profile=np.median(crop,axis=1)
        profile=median_filter(profile,size=5)
        pp=np.r_[0.,np.cumsum(profile)]
        ys0=np.arange(yl,yh)
        a=max(4,int(.22*pitch));z=max(a+2,int(.72*pitch))
        good=(ys0>z)&(ys0+z<h);ys0=ys0[good]
        above=(pp[ys0-a]-pp[ys0-z])/(z-a)
        below=(pp[ys0+z]-pp[ys0+a])/(z-a)
        tex=np.asarray([np.mean(texture[min(hs-1,int((y+.8*pitch)*s)):min(hs,int((y+2.0*pitch)*s)),max(0,j-2):j+3]) for y in ys0])
        score=above-below
        valid=(above>.45)&(below<.50)&(score>.18)&(tex>.40)
        tested.append(x)
        if not np.any(valid):continue
        k=int(np.argmax(np.where(valid,score,-100)))
        y=int(ys0[k]);obs.append([x,y]);strength.append(float(score[k]))
    obs=np.asarray(obs,float).reshape(-1,2)
    if len(obs)<20:
        rec['reasons']=['insufficient_yellow_surface_interface_points'];return out
    # Robust global curve rejects folds, active-rope edges and detached patches.
    center=float(np.median(obs[:,0]));span=float(np.ptp(obs[:,0])/2)
    u=(obs[:,0]-center)/max(span,1);keep=np.ones(len(obs),bool)
    for _ in range(4):
        coef=np.polyfit(u[keep],obs[keep,1],4);res=obs[:,1]-np.polyval(coef,u)
        med=np.median(res[keep]);mad=1.4826*np.median(abs(res[keep]-med))
        keep=abs(res-med)<max(.40*pitch,3*mad)
    accepted=obs[keep];fraction=len(accepted)/max(1,len(tested))
    # Model supplies a search neighborhood, local median residual supplies edge.
    local=accepted[abs(accepted[:,0]-d)<=2.5*pitch]
    rawlocal=obs[abs(obs[:,0]-d)<=2.5*pitch]
    predicted=float(np.polyval(coef,(d-center)/span))
    lres=local[:,1]-np.polyval(coef,(local[:,0]-center)/span) if len(local) else np.array([])
    ly=predicted+float(np.median(lres)) if len(local) else None
    lmad=1.4826*float(np.median(abs(lres-np.median(lres)))) if len(lres) else None
    local_test=sum(abs(x-d)<=2.5*pitch for x in tested)
    gap=max(np.diff(accepted[:,0]),default=1.)
    localbracket=bool(len(local) and local[:,0].min()<d-pitch and local[:,0].max()>d+pitch)
    rec.update(top_observed_xy=accepted.tolist(),rejected_top_observations_xy=obs[~keep].tolist(),
               top_support_fraction=float(fraction),top_largest_observation_gap_px=float(gap),
               local_support_fraction=len(local)/max(1,local_test),local_residual_MAD_px=lmad,
               edge_y=ly,edge_uncertainty_px=None if lmad is None else max(2.,2*lmad),
               surface_texture_fraction=float(np.mean(texture[candidate>0])),
               surface_candidate_area_px=int(len(sx)/s**2))
    reasons=[]
    presence=flange_presence(outer,accepted,pitch,yellow,s,d);rec['flange_presence']=presence
    if not presence['valid']:reasons.append('global_flange_structure_insufficient')
    if fraction<.65:reasons.append('incomplete_top_interface_support')
    # A remote occlusion is removed from the usable ROI instead of being filled
    # by the polynomial. The count column has its own stricter evidence gate.
    rec['remote_top_gaps_excluded']=bool(gap>3*pitch)
    if not localbracket or len(local)/max(1,local_test)<.70:reasons.append('d_not_bracketed_by_current_interface_evidence')
    if lmad is None or lmad>.20*pitch:reasons.append('local_edge_residual_too_large')
    # Keep only the observed horizontal extent; do not extrapolate side surfaces.
    full=cv2.resize(candidate,(w,h),interpolation=cv2.INTER_NEAREST)
    full=cv2.erode(full,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    start=int(accepted[:,0].min());end=int(accepted[:,0].max())
    full[:,:start]=0;full[:,end+1:]=0
    curve_x=np.arange(start,end+1)
    # A smooth, shape-preserving correction uses observed residuals. Drawn
    # curve, ROI cut and count origin now share exactly the same geometry.
    residual=accepted[:,1]-np.polyval(coef,(accepted[:,0]-center)/span)
    correction=PchipInterpolator(accepted[:,0],median_filter(residual,size=5,mode='nearest'),extrapolate=False)
    curve_y=np.polyval(coef,(curve_x-center)/span)+correction(curve_x)
    if localbracket:
        ly=float(np.polyval(coef,(d-center)/span)+correction(d))
        rec['edge_y']=ly
    supported_x=np.zeros(len(curve_x),bool)
    for lo,hi in zip(accepted[:-1,0],accepted[1:,0]):
        if hi-lo<=3*pitch:supported_x|=(curve_x>=lo)&(curve_x<=hi)
    rec['top_curve_supported_segments']=[]
    for lo,hi in runs(supported_x):
        xx=curve_x[lo:hi];yy=curve_y[lo:hi]
        sample=np.unique(np.r_[np.arange(0,len(xx),8),len(xx)-1])
        draw_x=np.unique(np.r_[xx[sample],d] if xx[0]<=d<=xx[-1] else xx[sample])
        draw_y=np.polyval(coef,(draw_x-center)/span)+correction(draw_x)
        rec['top_curve_supported_segments'].append(np.c_[draw_x,draw_y].tolist())
    rec['top_curve_semantics']='current_point_bracketed_smooth_curve; unsupported_gaps_excluded_from_ROI'
    # Reinsert a narrow top margin only where a current observation exists nearby.
    for idx,(x,y) in enumerate(zip(curve_x,curve_y)):
        if not supported_x[idx]:
            full[:min(h,int(y+2*pitch)),x]=0
            continue
        first=max(0,min(h-1,int(np.ceil(y+2))))
        col=np.flatnonzero(full[:,x])
        full[:first,x]=0
        if len(col) and col[0]>first and col[0]-first<4*pitch:
            tx=min(ws-1,int(x*s))
            patch=texture[int(first*s):max(int(first*s)+1,int(col[0]*s)),max(0,tx-2):tx+3]
            if np.mean(patch)>.70:full[first:int(col[0]),x]=255
    # Small internal holes caused by yellow reflections may be filled only if
    # the underlying current image still contains coherent rope-surface texture.
    holes=binary_fill_holes(full>0)&(full==0)
    tex_full=cv2.resize(texture.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
    full[holes&tex_full]=255
    draw_x=np.unique(np.r_[curve_x[::8],end,d] if start<=d<=end else np.r_[curve_x[::8],end])
    draw_y=np.polyval(coef,(draw_x-center)/span)+correction(draw_x)
    rec['top_curve_xy']=np.c_[draw_x,draw_y].tolist()
    rec['surface_contours_xy']=contours(full)
    rec['surface_area_px']=int(np.count_nonzero(full))
    # A single connected usable surface and a clear neighborhood of x=d.
    full=largest(full)
    out['yellow_mask'][full>0]=0
    rec['yellow_contours_xy']=contours(out['yellow_mask'])
    if ly is not None:
        xa,xb=max(0,int(d-1.2*pitch)),min(w,int(d+1.2*pitch)+1)
        ya,yb=max(0,int(ly+pitch)),min(h,int(ly+6*pitch))
        coverage=float(np.mean(full[ya:yb,xa:xb]>0)) if ya<yb else 0.
        rec['count_neighborhood_surface_coverage']=coverage
        if coverage<.95:reasons.append('count_neighborhood_surface_incomplete')
    rec['lateral_boundary_semantics']='conservative_texture_supported_interior_not_unseen_physical_edge'
    rec['reasons']=reasons;rec['valid']=not reasons
    rec['region_status']='usable_current_coil' if rec['valid'] else 'insufficient_current_coil_evidence'
    rec['outer_rim_status']='localized_at_d' if outer['valid'] else 'not_localized_at_d'
    out['surface_mask']=full
    return out
