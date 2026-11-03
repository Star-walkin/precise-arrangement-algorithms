"""D4 阴影宽度筛选：轻滤波、动态暗前景、开运算，再查原图亮肩。

本模块只接收数组；不读文件、旧黄点、标签或设备。白色前景代表阴影。
二值开运算先腐蚀后膨胀，去掉不能容纳指定宽度结构元素的细缝。
宽阴影仍不自动是绳间沟槽：还需原图双侧亮肩及切向持续支持。
返回值全为原图像素尺度的裁块，左上角为(x0,y0)，ROI外统一0。
"""
from __future__ import annotations

import math
import cv2
import numpy as np

PARAMETERS = {
    "margin_pitch": 2.0,
    "light_blur_sigma_pitch": .055,
    "coarse_gradient_sigma_pitch": .16,
    "tensor_integration_sigma_pitch": .65,
    "tensor_direction_max_weight": .90,
    "minimum_tensor_coherence": .28,
    "prefilter_half_span_pitch": .75,
    "prefilter_sigma_pitch": .40,
    "prefilter_samples": 9,
    "adaptive_background_sigma_pitch": .55,
    "adaptive_shadow_std_factor": .18,
    "adaptive_shadow_floor": 1.0,
    "opening_radius_normal_pitch": .080,
    "opening_radius_min_px": 1,
    "opening_radius_max_px": 4,
    "direction_offsets": (-.30, -.15, 0., .15, .30),
    "max_abs_slope": 2.5,
    "tangent_half_span_pitch": .80,
    "tangent_samples": 5,
    "flank_normal_pitch": .40,
    "raw_weak_floor": .20,
    "raw_strong_floor": .60,
    "raw_std_threshold_factor": .12,
    "minimum_valid_fraction": .80,
    "minimum_shadow_continuity": .42,
    "body_fill_radius_pitch": .075,
    "body_smooth_sigma_pitch": .16,
}


def _blur(im, sigma):
    return cv2.GaussianBlur(im, (0, 0), max(.5, float(sigma)))


def _weighted_blur(im, valid, sigma):
    den = _blur(valid, sigma)
    return _blur(im*valid, sigma)/np.maximum(den, 1e-5)


def _remap(im, xx, yy):
    return cv2.remap(im, np.asarray(xx, np.float32), np.asarray(yy, np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _disk(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))


def _reference_slope(reference, x):
    """C3只给大致方向；缺失参考也可做阴影/绳体提取。"""
    required = ('center_x_px','scale_x_px','b_px_per_u','c_px_per_u2')
    if not isinstance(reference, dict) or any(k not in reference for k in required):
        return np.zeros_like(x, np.float32), False
    v=[float(reference[k]) for k in required]
    if not all(math.isfinite(a) for a in v) or v[1]<=0:
        return np.zeros_like(x, np.float32), False
    slope=(v[2]+2*v[3]*(x-v[0])/v[1])/v[1]
    return np.clip(slope,-PARAMETERS['max_abs_slope'],PARAMETERS['max_abs_slope']).astype(np.float32),True


def _average_tangent(im, valid, xx, yy, slope, pitch):
    span=PARAMETERS['tangent_half_span_pitch']*pitch
    shifts=np.linspace(-span,span,PARAMETERS['tangent_samples'])
    weights=np.asarray([1.,2.,3.,2.,1.],np.float32);weights/=weights.sum()
    num=np.zeros_like(im);den=np.zeros_like(im)
    for dx,w in zip(shifts,weights):
        vx=_remap(valid,xx+dx,yy+slope*dx)
        num+=w*_remap(im*valid,xx+dx,yy+slope*dx)
        den+=w*vx
    return num/np.maximum(den,.01),den


def _coarse_direction(im,valid,xx,pitch,reference):
    """先抹平细股尺度，再用梯度张量找宽纹理切向；C3只给弱方向参考。

    梯度最强方向大致垂直于宽沟。小窗内平均梯度乘积，可避免
    单根细股的一个亮斑直接决定方向。低一致性处仍须后续拒识。
    """
    coarse=_weighted_blur(im,valid,PARAMETERS['coarse_gradient_sigma_pitch']*pitch)
    gx=cv2.Sobel(coarse,cv2.CV_32F,1,0,ksize=3,scale=.125)
    gy=cv2.Sobel(coarse,cv2.CV_32F,0,1,ksize=3,scale=.125)
    sigma=PARAMETERS['tensor_integration_sigma_pitch']*pitch
    jxx=_weighted_blur(gx*gx,valid,sigma)
    jyy=_weighted_blur(gy*gy,valid,sigma)
    jxy=_weighted_blur(gx*gy,valid,sigma)
    anis=np.sqrt(np.maximum((jxx-jyy)**2+4*jxy*jxy,0))
    coherence=np.clip(anis/np.maximum(jxx+jyy,1e-5),0,1)
    normal_angle=.5*np.arctan2(2*jxy,jxx-jyy)
    tx=-np.sin(normal_angle);ty=np.cos(normal_angle)
    sign=np.where(tx<0,-1.,1.);tx*=sign;ty*=sign
    direction_ok=(coherence>=PARAMETERS['minimum_tensor_coherence'])&(np.abs(ty)<=PARAMETERS['max_abs_slope']*np.abs(tx))
    image_slope=np.clip(ty/np.maximum(tx,.05),-PARAMETERS['max_abs_slope'],PARAMETERS['max_abs_slope'])
    guide,used=_reference_slope(reference,xx)
    # 用角度混合，侧缘较大的图像斜率不会被C3斜率上限当成错误剔除。
    weight=PARAMETERS['tensor_direction_max_weight']*coherence
    angle=(1-weight)*np.arctan(guide)+weight*np.arctan(image_slope)
    slope=np.clip(np.tan(angle),-PARAMETERS['max_abs_slope'],PARAMETERS['max_abs_slope'])
    return slope.astype(np.float32),coherence.astype(np.float32),direction_ok,used


def _prefilter_tangent(im,valid,xx,yy,slope,pitch):
    """灰度阶段沿宽纹理方向短平均，先消除珠链细纹，再做二值化。"""
    span=PARAMETERS['prefilter_half_span_pitch']*pitch
    shifts=np.linspace(-span,span,PARAMETERS['prefilter_samples'])
    ww=np.exp(-.5*(shifts/(PARAMETERS['prefilter_sigma_pitch']*pitch))**2);ww/=ww.sum()
    num=np.zeros_like(im);den=np.zeros_like(im)
    # dx按弧长转为水平分量，高斜率处不能无意跨越更长的纹理。
    norm=np.sqrt(1+slope*slope)
    for step,weight in zip(shifts,ww):
        dx=step/norm
        vv=_remap(valid,xx+dx,yy+slope*dx)
        num+=weight*_remap(im*valid,xx+dx,yy+slope*dx);den+=weight*vv
    return num/np.maximum(den,.01),den


def build_shadow_field(original_bgr, enhanced_roi, roi, pitch, reference_shape=None):
    """二值阶段 uint8 0/255；分数 float32 0..1；width_px为二值暗区局部直径。"""
    if (not isinstance(original_bgr,np.ndarray) or original_bgr.dtype!=np.uint8
        or original_bgr.ndim!=3 or original_bgr.shape[2]!=3):
        raise ValueError('original_bgr must be uint8 HxWx3')
    if not isinstance(enhanced_roi,np.ndarray) or enhanced_roi.shape!=original_bgr.shape or enhanced_roi.dtype!=np.uint8:
        raise ValueError('enhanced_roi must match original BGR array')
    if not isinstance(roi,np.ndarray) or roi.shape!=original_bgr.shape[:2] or roi.dtype!=np.uint8 or np.any((roi!=0)&(roi!=255)):
        raise ValueError('roi must be original-size binary uint8')
    pitch=float(pitch)
    if not math.isfinite(pitch) or not 8<=pitch<=70:raise ValueError('pitch must be in8..70 original pixels')
    mask=roi>0
    if not np.any(mask):
        z=np.zeros((0,0),np.uint8);f=z.astype(np.float32)
        return {'x0':0,'y0':0,'binary_shadow':z.copy(),'cleaned_shadow':z.copy(),
                'broad_score':f.copy(),'body_score':f.copy(),'valid_support':z.copy(),
                'width_px':f.copy(),'tangent_slope':f.copy(),
                'coarse_tangent_slope':f.copy(),'prefiltered_gray':f.copy(),
                'diagnostics':{'status':'empty_roi','parameters':PARAMETERS.copy()}}
    yy0,xx0=np.nonzero(mask);h,w=roi.shape
    margin=int(math.ceil(PARAMETERS['margin_pitch']*pitch))
    x0=max(0,int(xx0.min())-margin);x1=min(w,int(xx0.max())+margin+1)
    y0=max(0,int(yy0.min())-margin);y1=min(h,int(yy0.max())+margin+1)
    valid=mask[y0:y1,x0:x1].astype(np.float32)
    raw=cv2.cvtColor(original_bgr[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY).astype(np.float32)
    enh=cv2.cvtColor(enhanced_roi[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY).astype(np.float32)
    # 轻滤波不能先把细缝扩宽；后续必须在二值阴影上显式测宽。
    sigma=PARAMETERS['light_blur_sigma_pitch']*pitch
    raw=_weighted_blur(raw,valid,sigma);enh=_weighted_blur(enh,valid,sigma)
    ph,pw=valid.shape
    xx,yy=np.meshgrid(np.arange(pw,dtype=np.float32),np.arange(ph,dtype=np.float32))
    base_slope,coherence,direction_ok,has_reference=_coarse_direction(enh,valid,xx+x0,pitch,reference_shape)
    # 上一版先二值再做方向支持，细股交叉亮暗斑已变成宽岛，无法补救。
    # 现在先沿粗纹理切向抑制周期细纹，再决定哪些像素属于动态暗前景。
    enh,prefilter_coverage=_prefilter_tangent(enh,valid,xx,yy,base_slope,pitch)
    raw,_=_prefilter_tangent(raw,valid,xx,yy,base_slope,pitch)
    bg_sigma=PARAMETERS['adaptive_background_sigma_pitch']*pitch
    bg=_weighted_blur(enh,valid,bg_sigma)
    var=np.maximum(_weighted_blur(enh*enh,valid,bg_sigma)-bg*bg,0)
    std=np.sqrt(var)
    threshold=np.maximum(PARAMETERS['adaptive_shadow_floor'],PARAMETERS['adaptive_shadow_std_factor']*std)
    shadow=((enh<bg-threshold)&(valid>0)).astype(np.uint8)*255
    raw_bg=_weighted_blur(raw,valid,bg_sigma)
    raw_std=np.sqrt(np.maximum(_weighted_blur(raw*raw,valid,bg_sigma)-raw_bg*raw_bg,0))
    normal_pitch=pitch/np.sqrt(1+base_slope*base_slope)
    # 圆盘无需假定局部方向恰好正确；远侧投影绳距变窄时，结构元素同步变小。
    radii=np.clip(np.rint(PARAMETERS['opening_radius_normal_pitch']*normal_pitch),
                  PARAMETERS['opening_radius_min_px'],PARAMETERS['opening_radius_max_px']).astype(np.int16)
    cleaned=np.zeros_like(shadow)
    for radius in np.unique(radii):
        kernel=_disk(int(radius))
        eroded=cv2.erode(shadow,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
        opened=cv2.dilate(eroded,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
        sel=radii==radius;cleaned[sel]=opened[sel]
    cleaned[(shadow==0)|(valid==0)]=0
    distance=cv2.distanceTransform(shadow,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    width=(2*distance).astype(np.float32)
    best=np.zeros_like(raw);support=np.zeros_like(shadow);best_slope=base_slope.copy()
    raw_weak_limit=np.maximum(PARAMETERS['raw_weak_floor'],PARAMETERS['raw_std_threshold_factor']*raw_std)
    raw_strong_limit=np.maximum(PARAMETERS['raw_strong_floor'],.24*raw_std)
    candidate=cleaned.astype(np.float32)/255
    per_direction=[]
    for offset in PARAMETERS['direction_offsets']:
        slope=np.clip(base_slope+offset,-PARAMETERS['max_abs_slope'],PARAMETERS['max_abs_slope'])
        den=np.sqrt(1+slope*slope);nx,ny=-slope/den,1/den
        center,coverage=_average_tangent(raw,valid,xx,yy,slope,pitch)
        continuous,_=_average_tangent(candidate,valid,xx,yy,slope,pitch)
        flank=PARAMETERS['flank_normal_pitch']*pitch/den
        a=_remap(center,xx+flank*nx,yy+flank*ny)
        b=_remap(center,xx-flank*nx,yy-flank*ny)
        va=_remap(coverage,xx+flank*nx,yy+flank*ny)
        vb=_remap(coverage,xx-flank*nx,yy-flank*ny)
        weak=np.minimum(a-center,b-center);strong=np.maximum(a-center,b-center)
        coverage_ok=np.minimum(np.minimum(coverage,va),vb)>=PARAMETERS['minimum_valid_fraction']
        ok=((cleaned>0)&direction_ok&coverage_ok&(weak>=raw_weak_limit)&(strong>=raw_strong_limit)
            &(continuous>=PARAMETERS['minimum_shadow_continuity']))
        valley_strength=np.clip(weak/np.maximum(.65*raw_std,.5),0,1)
        width_strength=np.clip(distance/np.maximum(.20*normal_pitch,1),0,1)
        score=(.45*valley_strength+.30*continuous+.25*width_strength)*ok
        update=score>best;best[update]=score[update];best_slope[update]=slope[update]
        support[ok]=255;per_direction.append(int(np.count_nonzero(ok)))
    # 亮绳体阻力另行计算：灰度闭运算填内部窄暗缝，不填真正宽沟。
    # 只在局部显著亮于低频背景时高分，不能把所有“非暗沟”都当绳体。
    fill_radius=max(1,int(round(PARAMETERS['body_fill_radius_pitch']*pitch)))
    filled=cv2.morphologyEx(enh,cv2.MORPH_CLOSE,_disk(fill_radius))
    body_signal=_weighted_blur(filled,valid,PARAMETERS['body_smooth_sigma_pitch']*pitch)
    body_bg=_weighted_blur(body_signal,valid,bg_sigma)
    body=np.clip((body_signal-body_bg)/np.maximum(.65*std,1.5),0,1)
    body[cleaned>0]*=.15
    for a in [best,body,width,best_slope,base_slope,enh]:a[valid==0]=0
    support[valid==0]=0
    result={'x0':x0,'y0':y0,'binary_shadow':shadow,'cleaned_shadow':cleaned,
            'broad_score':np.clip(best,0,1).astype(np.float32),
            'body_score':np.clip(body,0,1).astype(np.float32),'valid_support':support,
            'width_px':width,'tangent_slope':best_slope.astype(np.float32),
            'coarse_tangent_slope':base_slope.astype(np.float32),
            'prefiltered_gray':enh.astype(np.float32)}
    result['diagnostics']={'status':'completed','crop_xyxy':[x0,y0,x1,y1],
      'pitch_px':pitch,'reference_used_for_direction':has_reference,
      'roi_pixels':int(np.count_nonzero(valid)),'binary_shadow_pixels':int(np.count_nonzero(shadow)),
      'cleaned_shadow_pixels':int(np.count_nonzero(cleaned)),
      'raw_supported_shadow_pixels':int(np.count_nonzero(support)),
      'strong_body_pixels':int(np.count_nonzero(body>.60)),
      'opening_radii_px':[int(a) for a in np.unique(radii[valid>0])],
      'coarse_tensor_coherence_median':float(np.median(coherence[valid>0])),
      'tensor_direction_eligible_fraction':float(np.mean(direction_ok[valid>0])),
      'prefilter_coverage_median':float(np.median(prefilter_coverage[valid>0])),
      'per_direction_supported_pixels':per_direction,'parameters':PARAMETERS.copy(),
      'meaning':'Width-filtered image evidence only; neither boundary identity nor material identity ground truth.'}
    return result


__all__=['PARAMETERS','build_shadow_field']
