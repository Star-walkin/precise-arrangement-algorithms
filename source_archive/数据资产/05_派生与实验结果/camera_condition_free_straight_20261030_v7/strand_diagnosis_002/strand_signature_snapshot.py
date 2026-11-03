"""A7 自由直绳内带股纹重复性：纯数组诊断，无候选选择或身份判断。

外轮廓再直也可能是工具。这里排除两侧外边缘，把候选中心内带拉直，
去掉缓慢光照变化，再检查沿轴阴影是否重复，以及几个局部窗口是否
有相似周期。没有可见纹理时返回 unknown；它不是“确认没有绳”。
输入的直段必须由自动几何模块提出；不接受文件名或人工正例提示。
"""
import math
import cv2
import numpy as np

PARAMETERS = {
    'axis_step_px': 1., 'normal_step_px': .5, 'interior_half_width': .30,
    'maximum_axis_samples': 4096, 'maximum_windows': 12,
    # 最大周期为3.5w，要求至少3次重复；窗口至少12w才真正覆盖该范围。
    'window_widths': 12., 'window_min_px': 128., 'window_max_px': 420.,
    'period_min_widths': .35, 'period_max_widths': 3.5,
    'minimum_cycles_per_window': 3., 'minimum_highpass_std': 1.5,
    'minimum_peak_correlation': .28, 'minimum_peak_prominence': .08,
    'minimum_local_gradient_coherence': .18,
    'minimum_oblique_angle_deg': 10., 'maximum_oblique_angle_deg': 80.,
    'relative_period_tolerance': .22, 'minimum_periodic_window_fraction': .5,
    'minimum_consistent_windows': 2,
}


def _one_window(patch, width, axis_step, normal_step):
    """只用内部像素：缓变照明、中心均值、外侧轮廓均不提供周期证据。"""
    p=np.asarray(patch,np.float32)
    # y沿候选轴，x跨候选轴。只去沿轴缓变分量，保留斜股纹亮暗交替。
    sigma=max(2.,1.4*width/axis_step)
    high=p-cv2.GaussianBlur(p,(1,0),sigmaX=0,sigmaY=sigma)
    std=float(np.std(high));saturation=float(np.mean(p>=248));mean=float(np.mean(p))
    base={'highpass_std':std,'mean_gray':mean,'saturated_fraction':saturation,
          'sample_count':int(len(p)),'period_px':None,'peak_correlation':0.,
          'peak_prominence':0.,'periodic':False,'oblique':False}
    if std<PARAMETERS['minimum_highpass_std']:
        base['reason']='saturated_texture_unknown' if saturation>.5 else 'weak_texture_unknown'
        return base
    # 剪掉个别坏点，防止单条划痕/孤立亮点支配相关量；没有二值化纹理。
    high=np.clip(high,-4*std,4*std)
    gt=cv2.Sobel(high,cv2.CV_32F,0,1,ksize=3)/(8*axis_step)
    gn=cv2.Sobel(high,cv2.CV_32F,1,0,ksize=3)/(8*normal_step)
    gt=gt[2:-2,2:-2];gn=gn[2:-2,2:-2]
    jtt=float(np.mean(gt*gt));jnn=float(np.mean(gn*gn));jtn=float(np.mean(gt*gn))
    energy=jtt+jnn;coherence=float(math.hypot(jtt-jnn,2*jtn)/max(energy,1e-9))
    gradient_angle=.5*math.atan2(2*jtn,jtt-jnn)
    orientation=(math.degrees(gradient_angle)+90.)%180.
    acute=min(orientation,180.-orientation)
    oblique=coherence>=PARAMETERS['minimum_local_gradient_coherence'] and PARAMETERS['minimum_oblique_angle_deg']<=acute<=PARAMETERS['maximum_oblique_angle_deg']
    base.update(gradient_energy=energy,gradient_coherence=coherence,
                texture_orientation_from_axis_deg=orientation,oblique=bool(oblique))
    low=max(3,int(round(PARAMETERS['period_min_widths']*width/axis_step)))
    upper=min(int(PARAMETERS['period_max_widths']*width/axis_step),int(len(p)/PARAMETERS['minimum_cycles_per_window']))
    if upper<=low+2:
        base['reason']='window_too_short_for_period';return base
    # 对每个横向采样列保留相位，自相关后再汇总。直接对横截面求均值会
    # 把相位不同的斜纹抵消；相关用两段各自能量归一化，不使用总体曝光。
    lags=np.arange(1,upper+2);corr=[]
    for lag in lags:
        a=high[:-lag];b=high[lag:]
        corr.append(float(np.sum(a*b)/max(math.sqrt(float(np.sum(a*a))*float(np.sum(b*b))),1e-9)))
    corr=np.asarray(corr)
    peaks=[i for i in range(low-1,upper) if 0<i<len(corr)-1 and corr[i]>=corr[i-1] and corr[i]>=corr[i+1]]
    if not peaks:
        base['reason']='no_repeated_peak';return base
    candidates=[]
    for i in peaks:
        lag=int(lags[i]);lo=max(0,int(.55*lag)-1);hi=min(len(corr),int(1.45*lag)+1)
        prominence=float(corr[i]-.5*(float(np.min(corr[lo:i+1]))+float(np.min(corr[i:hi]))))
        candidates.append((float(corr[i]),prominence,lag))
    # 先取最高相关峰；近似等强时取短周期，避免把整数倍周期当新的节距。
    best=max(candidates,key=lambda q:(round(q[0]/.03),-q[2]))
    peak,prominence,lag=best;period=float(lag*axis_step)
    periodic=peak>=PARAMETERS['minimum_peak_correlation'] and prominence>=PARAMETERS['minimum_peak_prominence']
    base.update(period_px=period,peak_correlation=peak,peak_prominence=prominence,
                periodic=bool(periodic),reason='periodic_internal_texture' if periodic else 'weak_or_irregular_repetition',
                autocorrelation_peaks=[{'period_px':float(l*axis_step),'correlation':float(c),'prominence':float(pr)} for c,pr,l in sorted(candidates,reverse=True)[:4]])
    return base


def measure_strand_signature(gray,segment_xy,width_px):
    """Measure original-pixel inner-band repetition, return only JSON-safe values.

    gray: HxW grayscale original pixels; segment_xy: two AUTO endpoints in (x,y);
    width_px: measured optical width, not physical rope diameter. This function
    does not inspect surface masks, old selected paths, IDs, filenames or labels.
    The status is image evidence, never a material-identity probability.
    """
    im=np.asarray(gray);seg=np.asarray(segment_xy,float);width=float(width_px)
    if im.ndim!=2 or seg.shape!=(2,2) or not np.isfinite(seg).all() or not math.isfinite(width) or width<=2:
        raise ValueError('Need finite grayscale image,2x2 segment and measured width>2px')
    delta=seg[1]-seg[0];length=float(np.linalg.norm(delta))
    out={'status':'unknown','period_px':None,'period_width_ratio':None,'periodic_window_fraction':0.,
         'consistent_window_fraction':0.,'oblique_window_fraction':0.,'window_count':0,
         'measured_length_px':length,'optical_width_px':width,'parameters':dict(PARAMETERS),
         'same_material_identity_confirmed':False,'probability':None,'windows':[]}
    if length<max(32.,3*width):out['reason']='insufficient_visible_center_span';return out
    direction=delta/length
    # 固定轴正向，使交换两个端点时的方向描述不变。
    if direction[0]<0 or (abs(direction[0])<1e-8 and direction[1]<0):seg=seg[::-1];direction=-direction
    normal=np.array([-direction[1],direction[0]])
    count=min(PARAMETERS['maximum_axis_samples'],int(length/PARAMETERS['axis_step_px'])+1)
    ss=np.linspace(0.,length,count);step=float(ss[1]-ss[0]);radius=max(1.5,PARAMETERS['interior_half_width']*width)
    normals=np.arange(-radius,radius+PARAMETERS['normal_step_px']/2,PARAMETERS['normal_step_px'])
    xy=seg[0]+ss[:,None,None]*direction+normals[None,:,None]*normal
    h,w=im.shape;valid=(xy[...,0]>=1)&(xy[...,0]<w-2)&(xy[...,1]>=1)&(xy[...,1]<h-2)
    rows=valid.all(axis=1);out['in_frame_fraction']=float(rows.mean())
    if rows.mean()<.85:out['reason']='insufficient_in_frame_interior';return out
    patch=cv2.remap(im.astype(np.float32),xy[...,0].astype(np.float32),xy[...,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    window_span=float(np.clip(PARAMETERS['window_widths']*width,PARAMETERS['window_min_px'],PARAMETERS['window_max_px']))
    size=min(len(patch),max(32,int(window_span/step)))
    starts=np.unique(np.rint(np.linspace(0,len(patch)-size,min(PARAMETERS['maximum_windows'],max(1,1+int((len(patch)-size)/(size*.5)))))).astype(int))
    reports=[]
    for start in starts:
        if not rows[start:start+size].all():continue
        r=_one_window(patch[start:start+size],width,step,PARAMETERS['normal_step_px'])
        r.update(start_s_px=float(ss[start]),end_s_px=float(ss[start+size-1]));reports.append(r)
    out['windows']=reports;out['window_count']=len(reports)
    if not reports:out['reason']='no_complete_inner_window';return out
    good=[r for r in reports if r['periodic']]
    out['periodic_window_fraction']=len(good)/len(reports)
    out['oblique_window_fraction']=sum(r['oblique'] for r in reports)/len(reports)
    out['median_highpass_std']=float(np.median([r['highpass_std'] for r in reports]))
    out['median_peak_correlation']=float(np.median([r['peak_correlation'] for r in reports]))
    out['saturated_fraction']=float(np.mean(patch[rows]>=248))
    if not good:
        out['status']='unknown' if out['median_highpass_std']<PARAMETERS['minimum_highpass_std'] else 'irregular_texture'
        out['reason']='no_local_repeated_internal_texture';return out
    periods=np.asarray([r['period_px'] for r in good]);median=float(np.median(periods))
    consistent=np.abs(periods-median)/median<=PARAMETERS['relative_period_tolerance']
    stable_count=int(consistent.sum());out['period_px']=median;out['period_width_ratio']=median/width
    out['period_relative_mad']=float(np.median(abs(periods-median))/median)
    out['consistent_window_fraction']=stable_count/len(reports)
    if stable_count>=PARAMETERS['minimum_consistent_windows'] and out['consistent_window_fraction']>=PARAMETERS['minimum_periodic_window_fraction']:
        out['status']='periodic_strand_evidence' if out['oblique_window_fraction']>=.5 else 'periodic_texture_orientation_uncertain'
        out['reason']='repeated_inner_pattern_consistent_across_windows'
    else:
        out['status']='inconsistent_or_short_repetition';out['reason']='local_peaks_do_not_establish_stable_visible_period'
    return out
