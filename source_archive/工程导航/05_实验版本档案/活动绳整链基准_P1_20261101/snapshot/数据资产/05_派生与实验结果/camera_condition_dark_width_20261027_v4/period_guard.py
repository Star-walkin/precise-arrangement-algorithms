"""D4 原图纹理周期诊断与亮绳体阻力；数组输入，无文件/设备副作用。

这里测到的是图像竖向剖面的重复间隔，不是实体绳径、股数或螺距。
周期只调整证据可信度，不以等距中点凭空生成绳体。强亮粗长条带视为
未知遮挡，调用者必须禁止跨越unknown_mask，不能把它当卷面绳体绕行。
"""
from __future__ import annotations

import math
import cv2
import numpy as np

PARAMETERS = {
    'period_column_step_pitch': 2.0,
    'coarse_period_range_pitch': (.65, 1.40),
    'fine_period_range_pitch': (.12, .55),
    'minimum_profile_length_pitch': 4.0,
    'minimum_period_correlation': .25,
    'minimum_peak_margin': .035,
    'raw_body_fill_radius_pitch': .075,
    'raw_body_blur_sigma_pitch': .16,
    'raw_body_background_sigma_pitch': .55,
    'raw_body_contrast_floor': .75,
    'body_tangent_half_span_pitch': 1.10,
    'body_tangent_sigma_pitch': .50,
    'body_tangent_samples': 21,
    'body_normal_half_span_pitch': .24,
    'body_normal_sigma_pitch': .11,
    'body_normal_samples': 7,
    'body_flank_distance_pitch': .43,
    'body_minimum_sample_coverage': .80,
    'unknown_bright_floor': 180.,
    'unknown_min_mean_gray': 185.,
    'unknown_max_inner_std': 32.,
    'unknown_min_width_pitch': .18,
    'unknown_min_length_pitch': 2.5,
    'unknown_min_area_pitch_squared': .25,
    'unknown_margin_pitch': .12,
}


def _blur(im, sigma):
    return cv2.GaussianBlur(im, (0, 0), max(.5, float(sigma)))


def _weighted_blur(im, valid, sigma):
    return _blur(im*valid, sigma)/np.maximum(_blur(valid, sigma), 1e-5)


def _disk(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                     (2*radius+1, 2*radius+1))


def _remap(im, xx, yy):
    return cv2.remap(im, np.asarray(xx,np.float32), np.asarray(yy,np.float32),
                     cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)


def _directional_average(im, valid, xx, yy, slope, pitch, tangent=True):
    """沿绳走向平均掉斜向细股纹，再沿法向轻平滑；不跨黑色ROI填值。

    pitch 是同一列中的竖向间隔。法向绳距因此是 pitch/sqrt(1+slope²)。
    这里使用图像给出的粗方向，不把 C3 曲线几何本身当成亮绳体证据。
    """
    norm=np.sqrt(1+slope*slope)
    if tangent:
        ux,uy=1/norm,slope/norm
        scale=pitch
        span=PARAMETERS['body_tangent_half_span_pitch']
        sigma=PARAMETERS['body_tangent_sigma_pitch']
        count=PARAMETERS['body_tangent_samples']
    else:
        ux,uy=-slope/norm,1/norm
        scale=pitch/norm
        span=PARAMETERS['body_normal_half_span_pitch']
        sigma=PARAMETERS['body_normal_sigma_pitch']
        count=PARAMETERS['body_normal_samples']
    offsets=np.linspace(-span,span,count)
    weights=np.exp(-.5*(offsets/sigma)**2);weights/=weights.sum()
    mult=valid if im.ndim==2 else valid[:,:,None]
    numerator=np.zeros_like(im);coverage=np.zeros_like(valid)
    weighted=im*mult
    for amount,weight in zip(offsets,weights):
        mx=xx+amount*scale*ux;my=yy+amount*scale*uy
        numerator+=weight*_remap(weighted,mx,my)
        coverage+=weight*_remap(valid,mx,my)
    denominator=coverage if im.ndim==2 else coverage[:,:,None]
    return numerator/np.maximum(denominator,1e-5),coverage


def _acf_peaks(profile, valid, pitch, limits):
    """只接受非零局部极大值；从0滞后平滑下降不是周期峰。"""
    lo=max(2, int(math.ceil(limits[0]*pitch)))
    hi=min(len(profile)-2, int(math.floor(limits[1]*pitch)))
    if hi<=lo or np.count_nonzero(valid)<PARAMETERS['minimum_profile_length_pitch']*pitch:
        return {'period_px':None,'correlation':0.,'status':'insufficient_profile'}
    values=np.asarray(profile, np.float64)
    values=values-float(np.median(values[valid]))
    acf=np.full(hi+2, np.nan, np.float64)
    for lag in range(max(1,lo-1),hi+2):
        paired=valid[:-lag]&valid[lag:]
        if np.count_nonzero(paired)<max(16,2*pitch):continue
        a,b=values[:-lag][paired],values[lag:][paired]
        denominator=math.sqrt(float(a@a)*float(b@b))
        if denominator>1e-8:acf[lag]=float(a@b)/denominator
    peaks=[k for k in range(lo,hi+1) if np.isfinite(acf[k-1:k+2]).all()
           and acf[k]>acf[k-1] and acf[k]>=acf[k+1] and acf[k]>0]
    if not peaks:return {'period_px':None,'correlation':0.,'status':'no_nonzero_local_peak'}
    peaks.sort(key=lambda k:-acf[k]);best=peaks[0]
    second=max([float(acf[k]) for k in peaks[1:] if abs(k-best)>=.15*pitch] or [-1.])
    margin=float(acf[best])-second
    accepted=(acf[best]>=PARAMETERS['minimum_period_correlation'] and
              margin>=PARAMETERS['minimum_peak_margin'])
    return {'period_px':float(best) if accepted else None,
            'strongest_candidate_px':int(best),'correlation':float(acf[best]),
            'competing_peak_margin':margin,'candidate_peaks_px':peaks,
            'status':'accepted_projected_texture_period' if accepted else 'weak_or_ambiguous'}


def _unknown_bright(raw, valid, pitch):
    """只把原图强亮、粗长且较均匀的条带排除；不声称是活动绳分割。"""
    bg=_weighted_blur(raw,valid,max(3.,pitch))
    anomalous=((raw>=PARAMETERS['unknown_bright_floor']) &
               (raw>=bg+25.) & (valid>0)).astype(np.uint8)*255
    anomalous=cv2.morphologyEx(anomalous,cv2.MORPH_CLOSE,_disk(1))
    n,labels,stats,_=cv2.connectedComponentsWithStats(anomalous)
    unknown=np.zeros_like(anomalous)
    for label in range(1,n):
        x,y,w,h,area=map(int,stats[label])
        if area<PARAMETERS['unknown_min_area_pitch_squared']*pitch*pitch:continue
        region=labels[y:y+h,x:x+w]==label
        if max(w,h)<PARAMETERS['unknown_min_length_pitch']*pitch:continue
        # 面积/长轴给粗度近似；小亮高光细线通常不能通过此门。
        if area/max(w,h)<PARAMETERS['unknown_min_width_pitch']*pitch:continue
        samples=raw[y:y+h,x:x+w][region]
        if float(samples.mean())<PARAMETERS['unknown_min_mean_gray']:continue
        if float(samples.std())>PARAMETERS['unknown_max_inner_std']:continue
        unknown[y:y+h,x:x+w][region]=255
    margin=max(1,int(round(PARAMETERS['unknown_margin_pitch']*pitch)))
    unknown=cv2.dilate(unknown,_disk(margin))
    unknown[valid==0]=0
    return unknown


def build_period_guard(original_bgr, enhanced_roi, roi, pitch,
                       reference_shape, shadow_field):
    """返回同shadow裁块的body_barrier、unknown_mask和逐列纹理周期。

    body_barrier高分必须同时得到增强图和原图的宽亮脊及两侧暗谷支持。
    pilot_001中各向同性平滑仍留下珠链细纹；本版先按图像粗方向平均，
    不再用未平滑body_score最大值把细纹写回，也不把cleaned_shadow当真值。
    local_pitch_by_column为原图竖向像素间距；估计失败时回退输入pitch，
    同时period_confidence=0。这个回退本身不会制造body_barrier。
    """
    if (not isinstance(original_bgr,np.ndarray) or original_bgr.dtype!=np.uint8
        or original_bgr.ndim!=3 or original_bgr.shape[2]!=3):raise ValueError('original must be uint8 BGR')
    if enhanced_roi.shape!=original_bgr.shape or roi.shape!=original_bgr.shape[:2]:
        raise ValueError('input shapes must agree')
    pitch=float(pitch)
    if not math.isfinite(pitch) or not 8<=pitch<=70:raise ValueError('pitch outside8..70px')
    x0,y0=int(shadow_field['x0']),int(shadow_field['y0'])
    body=np.asarray(shadow_field['body_score'],np.float32)
    if body.ndim!=2 or not np.isfinite(body).all():raise ValueError('invalid body_score')
    ph,pw=body.shape
    if x0<0 or y0<0 or x0+pw>roi.shape[1] or y0+ph>roi.shape[0]:raise ValueError('invalid field crop')
    if not body.size:
        z=np.zeros(body.shape,np.uint8)
        return {'x0':x0,'y0':y0,'body_barrier':body.copy(),'unknown_mask':z,
                'excluded_mask':z.copy(),'local_pitch_by_column':np.zeros(pw,np.float32),
                'period_confidence_by_column':np.zeros(pw,np.float32),
                'diagnostics':{'status':'empty_crop','parameters':PARAMETERS.copy()}}
    valid=(roi[y0:y0+ph,x0:x0+pw]>0).astype(np.float32)
    raw=cv2.cvtColor(original_bgr[y0:y0+ph,x0:x0+pw],cv2.COLOR_BGR2GRAY).astype(np.float32)
    unknown=_unknown_bright(raw,valid,pitch)
    supplied=shadow_field.get('unknown_mask',shadow_field.get('excluded_mask'))
    if supplied is not None:
        supplied=np.asarray(supplied)
        if supplied.shape==roi.shape:supplied=supplied[y0:y0+ph,x0:x0+pw]
        if supplied.shape!=body.shape:raise ValueError('excluded mask crop mismatch')
        unknown[supplied>0]=255
    evidence_valid=valid*(unknown==0)
    radius=max(1,int(round(PARAMETERS['raw_body_fill_radius_pitch']*pitch)))
    enh=cv2.cvtColor(enhanced_roi[y0:y0+ph,x0:x0+pw],cv2.COLOR_BGR2GRAY).astype(np.float32)
    slope_source=('coarse_tangent_slope' if 'coarse_tangent_slope' in shadow_field
                  else 'tangent_slope')
    slope=np.asarray(shadow_field.get(slope_source,np.zeros_like(raw)),np.float32)
    if slope.shape!=raw.shape or not np.isfinite(slope).all():raise ValueError('invalid tangent slope')
    slope=np.clip(slope,-2.5,2.5)
    # 最佳沟槽方向在逐像素候选间可能跳变；只对方向做宽尺度平滑。
    # double-angle平均避免方向在角度环绕处失真。
    angle=2*np.arctan(slope)
    c=_weighted_blur(np.cos(angle),evidence_valid,.35*pitch)
    s=_weighted_blur(np.sin(angle),evidence_valid,.35*pitch)
    slope=np.clip(np.tan(.5*np.arctan2(s,c)),-2.5,2.5).astype(np.float32)
    xx,yy=np.meshgrid(np.arange(pw,dtype=np.float32),np.arange(ph,dtype=np.float32))
    # 闭运算只填窄细缝；沿绳体方向的长平均才负责抑制斜向股纹珠链。
    filled=cv2.morphologyEx(raw,cv2.MORPH_CLOSE,_disk(radius))
    enh_filled=cv2.morphologyEx(enh,cv2.MORPH_CLOSE,_disk(radius))
    pair=np.stack((filled,enh_filled),axis=2)
    tangent,coverage=_directional_average(pair,evidence_valid,xx,yy,slope,pitch,True)
    coarse,normal_coverage=_directional_average(tangent,evidence_valid,xx,yy,slope,pitch,False)
    norm=np.sqrt(1+slope*slope);nx,ny=-slope/norm,1/norm
    flank=PARAMETERS['body_flank_distance_pitch']*pitch/norm
    left=_remap(coarse,xx+flank*nx,yy+flank*ny)
    right=_remap(coarse,xx-flank*nx,yy-flank*ny)
    effective_coverage=np.minimum(coverage,normal_coverage)*evidence_valid
    coverage_ok=np.minimum.reduce((effective_coverage,
        _remap(effective_coverage,xx+flank*nx,yy+flank*ny),
        _remap(effective_coverage,xx-flank*nx,yy-flank*ny)))>=PARAMETERS['body_minimum_sample_coverage']
    # 真正宽亮绳体应高于左右两侧的粗尺度暗谷；只高于局部均值不够。
    # 暗沟本身不会因cleaned_shadow标签而自动免检，而由原图亮度否证。
    peaks=[]
    for channel,floor in [(0,PARAMETERS['raw_body_contrast_floor']),(1,1.5)]:
        broad=coarse[:,:,channel]
        bg=_weighted_blur(broad,evidence_valid,PARAMETERS['raw_body_background_sigma_pitch']*pitch)
        std=np.sqrt(np.maximum(_weighted_blur(broad*broad,evidence_valid,.55*pitch)-bg*bg,0))
        prominence=np.minimum(broad-left[:,:,channel],broad-right[:,:,channel])
        peak=np.clip(prominence/np.maximum(.75*std,floor),0,1)
        peak[~coverage_ok]=0.
        peaks.append(peak)
    raw_peak,independent_peak=peaks
    step=max(8,int(round(PARAMETERS['period_column_step_pitch']*pitch)))
    nodes=list(range(0,pw,step))
    if nodes[-1]!=pw-1:nodes.append(pw-1)
    estimates=[];periods=[];confidences=[]
    for x in nodes:
        use=evidence_valid[:,x]>0
        profile=raw[:,x].copy()
        # 小尺度残差用于诊断细纹周期；粗尺度曲线用于一绳距主重复。
        low=cv2.GaussianBlur(profile[:,None],(1,0),max(.6,.12*pitch)).ravel()
        trend=cv2.GaussianBlur(low[:,None],(1,0),max(2.,1.2*pitch)).ravel()
        fine=_acf_peaks(profile-low,use,pitch,PARAMETERS['fine_period_range_pitch'])
        coarse=_acf_peaks(low-trend,use,pitch,PARAMETERS['coarse_period_range_pitch'])
        period=coarse['period_px']
        confidence=(float(np.clip((coarse['correlation']-.20)/.60,0,1))
                    if period is not None else 0.)
        periods.append(float(period) if period is not None else pitch)
        confidences.append(confidence)
        estimates.append({'original_x':x+x0,'valid_samples':int(np.count_nonzero(use)),
                          'fine_texture':fine,'coarse_texture':coarse})
    local_pitch=np.interp(np.arange(pw),nodes,periods).astype(np.float32)
    confidence=np.interp(np.arange(pw),nodes,confidences).astype(np.float32)
    # 不用未平滑的旧body_score作max合并，避免把已去除的珠链细纹写回。
    barrier=np.sqrt(independent_peak*raw_peak)*(.70+.30*confidence[None,:])
    # 原图的宽亮脊是必要条件；细亮点/无对比块不可靠。
    core=((barrier>.45)&(raw_peak>.35)&(evidence_valid>0)).astype(np.uint8)*255
    core=cv2.morphologyEx(core,cv2.MORPH_OPEN,_disk(1))
    barrier[core==0]=0.
    barrier[(valid==0)|(unknown>0)]=0.
    cleaned=np.asarray(shadow_field.get('cleaned_shadow',np.zeros_like(core)))
    if cleaned.shape!=body.shape:raise ValueError('cleaned shadow crop mismatch')
    diagnostics={'status':'completed','parameters':PARAMETERS.copy(),'pitch_px':pitch,
      'period_nodes':estimates,'accepted_coarse_nodes':sum(e['coarse_texture']['period_px'] is not None for e in estimates),
      'accepted_fine_nodes':sum(e['fine_texture']['period_px'] is not None for e in estimates),
      'unknown_pixels':int(np.count_nonzero(unknown)),
      'body_direction_source':slope_source,
      'body_evidence':'Tangent-smoothed raw and enhanced broad bright ridges with darker bilateral normal flanks; no shadow-label immunity.',
      'strong_body_pixels':int(np.count_nonzero(barrier>.65)),
      'shadow_body_conflict_pixels':int(np.count_nonzero((barrier>.65)&(cleaned>0))),
      'meaning':'Image projected texture repetition and raw-supported bright ridge only; not physical rope/strand pitch, material identity or guaranteed active-rope segmentation.',
      'caller_requirement':'Never bridge unknown_mask; zero body barrier there means unknown, not a safe valley.'}
    return {'x0':x0,'y0':y0,'body_barrier':np.clip(barrier,0,1).astype(np.float32),
            'unknown_mask':unknown.astype(np.uint8),'excluded_mask':unknown.astype(np.uint8).copy(),
            'local_pitch_by_column':local_pitch,'period_confidence_by_column':confidence,
            'diagnostics':diagnostics}


__all__=['PARAMETERS','build_period_guard']
