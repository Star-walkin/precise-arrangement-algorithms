"""A5 原图绳带/两侧背景差分。纯 NumPy/OpenCV，无文件或设备访问。

坐标为原图 xy；本模块只测给定候选处的图像证据，不生成候选像素。
build_fields 仅从原图生成灰度/轻滤波副本。sample_ribbon 将候选内带与
其左右一根绳宽附近的背景比较：旧卷圈的重复亮峰通常在旁侧也存在，
不能像旧 max(亮峰, 暗峰) 那样靠一条股沟便取得高分。

输出支持只是二维局部外观，不能证明同根材料、接触或物理圈数。
"""
from __future__ import annotations
import math
import cv2
import numpy as np

PARAMETERS={
    'tangent_half_widths':1.25,'tangent_samples':11,
    'normal_half_widths':1.30,'normal_samples':33,
    'trim_each_end_samples':2,
    'minimum_raw_body_contrast':2.0,
    'minimum_raw_paired_edge_contrast':1.0,
    'minimum_body_coverage':.55,
    'minimum_along_consistency':.65,
    'minimum_valid_fraction':.97,
    'maximum_body_saturation_fraction':.55,
    'maximum_patch_saturation_fraction':.80,
    'minimum_patch_dynamic_range':4.0,
    'minimum_signal_maximum':5.0,
    'minimum_measured_width_ratio':.60,'maximum_measured_width_ratio':1.50,
    'minimum_supported_score':.30,
    'outline_minimum_raw_edge_contrast':8.,
    'outline_maximum_background_saturation_fraction':.30,
    'outline_minimum_along_consistency':.80,
    'outline_minimum_body_coverage':.75,
}


def build_fields(image,width,pitch):
    """从未绘线原图构建只读量测副本，不做曝光合成或候选自评分。

    width 和 pitch 都是像素尺度。width是自由绳光学宽度，pitch是卷面
    绳距；两者不强行等同。返回约 5 字节/像素的两个数组。
    """
    image=np.asarray(image)
    if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:
        raise ValueError('image must be original uint8 BGR HxWx3')
    width=float(width);pitch=float(pitch)
    if not (np.isfinite(width) and np.isfinite(pitch) and width>1 and pitch>1):
        raise ValueError('finite optical width and surface pitch >1 px required')
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sigma=float(np.clip(min(width,pitch)*.035,.6,1.5))
    smooth=cv2.GaussianBlur(gray.astype(np.float32),(0,0),sigma)
    return {'gray':gray,'smooth':smooth,'image_shape':gray.shape,
            'width_px':width,'pitch_px':pitch,'filter_sigma_px':sigma,
            'measurement_source':'original_image_only'}


def _sample(a,q):
    shape=q.shape[:-1];q=np.asarray(q,np.float32)
    # OpenCV remap的单维坐标有32767限制；保持短法向列而非拉成超长行。
    mx=q[...,0].reshape(-1,shape[-1]);my=q[...,1].reshape(-1,shape[-1]);out=np.empty(mx.shape,np.float32)
    for start in range(0,len(mx),32000):
        out[start:start+32000]=cv2.remap(a,mx[start:start+32000],my[start:start+32000],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return out.reshape(shape)


def _tensor(patch,along,across,width):
    """局部切向/法向坐标中的细纹方向；排除绳体外缘后再调用。"""
    n=len(patch)
    if patch.shape[1]<3 or patch.shape[2]<3:
        return np.zeros(n),np.zeros(n),np.zeros(n)
    dt=max(1e-5,float(along[1]-along[0]));dn=max(1e-5,float(across[1]-across[0]))
    gt=np.diff(patch,axis=1)[:,:,:-1]/(dt*width[:,None,None])
    gn=np.diff(patch,axis=2)[:,:-1,:]/(dn*width[:,None,None])
    tt=np.mean(gt*gt,axis=(1,2));nn=np.mean(gn*gn,axis=(1,2));tn=np.mean(gt*gn,axis=(1,2))
    energy=tt+nn;coherence=np.sqrt((tt-nn)**2+4*tn*tn)/np.maximum(energy,1e-6)
    angle=.5*np.arctan2(2*tn,tt-nn)+math.pi/2
    coherence=np.where(energy>=.15,coherence,0.)
    return angle,coherence,energy


def sample_ribbon(fields,xy,tangent,width,free_texture=None):
    """量测 N 个有方向/宽度的绳带剖面，所有数组均按 N 返回。

    xy/tangent: Nx2（单个xy或tangent也可）；width: 标量或N个像素宽度。
    tangent不必单位化。每个剖面沿切向覆盖2.5w、法向覆盖2.6w。
    free_texture可传A4的descriptor并补free_axis_angle_rad；它只提供弱
    辅助分，不改变原灰度支持条件。score范围0..1，supported是真正的
    双边差分支持，single_edge_support供调用者有限直线推断，二者分开。
    """
    p=np.asarray(xy,np.float32).reshape(-1,2);n=len(p)
    if n==0:raise ValueError('at least one candidate sample required')
    t=np.asarray(tangent,np.float32)
    if t.shape==(2,):t=np.repeat(t[None,:],n,axis=0)
    t=t.reshape(-1,2)
    ww=np.asarray(width,np.float32)
    if ww.ndim==0:ww=np.full(n,float(ww),np.float32)
    ww=ww.reshape(-1)
    if len(t)!=n or len(ww)!=n or not np.isfinite(np.r_[p.ravel(),t.ravel(),ww]).all() or np.any(ww<=1):
        raise ValueError('finite matching xy/tangent/width arrays with width>1 required')
    norm=np.linalg.norm(t,axis=1)
    if np.any(norm<1e-6):raise ValueError('nonzero tangent required')
    t=t/norm[:,None];normal=np.c_[-t[:,1],t[:,0]]
    a=np.linspace(-PARAMETERS['tangent_half_widths'],PARAMETERS['tangent_half_widths'],PARAMETERS['tangent_samples'],dtype=np.float32)
    o=np.linspace(-PARAMETERS['normal_half_widths'],PARAMETERS['normal_half_widths'],PARAMETERS['normal_samples'],dtype=np.float32)
    q=p[:,None,None,:]+ww[:,None,None,None]*(a[None,:,None,None]*t[:,None,None,:]+o[None,None,:,None]*normal[:,None,None,:])
    h,w=fields['image_shape'];valid=(q[...,0]>=1)&(q[...,0]<w-2)&(q[...,1]>=1)&(q[...,1]<h-2)
    raw=_sample(fields['gray'],q);filtered=_sample(fields['smooth'],q)
    # 去掉沿切向最亮/最暗各2个值：细股的局部亮斑、暗沟不决定整带均值。
    trim=PARAMETERS['trim_each_end_samples']
    prof=np.sort(filtered,axis=1)[:,trim:len(a)-trim,:].mean(axis=1)
    rawprof=np.sort(raw,axis=1)[:,trim:len(a)-trim,:].mean(axis=1)
    body=abs(o)<=.30;left=(o>=-1.20)&(o<=-.85);right=(o>=.85)&(o<=1.20)
    inner_mean=rawprof[:,body].mean(axis=1);left_mean=rawprof[:,left].mean(axis=1);right_mean=rawprof[:,right].mean(axis=1)
    # 线性插值背景抵消照明斜坡。由内带均值差的符号决定亮/暗极性，
    # 不分别在多处寻找最大暗谷/亮峰再挑一个最有利答案。
    lx=float(np.mean(o[left]));rx=float(np.mean(o[right]));mix=(o-lx)/(rx-lx)
    bg=left_mean[:,None]*(1-mix)[None,:]+right_mean[:,None]*mix[None,:]
    residual=prof-bg;body_difference=inner_mean-(left_mean+right_mean)/2
    polarity=np.where(body_difference>=0,1.,-1.).astype(np.float32)
    contrast=abs(body_difference)
    body_residual=polarity[:,None]*residual[:,body]
    body_coverage=np.mean(body_residual>=np.maximum(1.,.35*contrast)[:,None],axis=1)
    # 整绳两外缘应在约 ±半宽处。窄股沟的两条边都挤在中心，不能通过。
    # 两侧真实外缘必须来自原灰度。背景插值只用于均值差；若在残差上
    # 求边，邻带一侧含真绳造成的背景斜坡会伪造另一条根本不存在的边。
    gradient=np.diff(rawprof,axis=1)/(o[1]-o[0])
    mid=(o[:-1]+o[1:])/2;li=np.flatnonzero((mid>=-.75)&(mid<=-.30));ri=np.flatnonzero((mid>=.30)&(mid<=.75))
    lidx=li[np.argmax(gradient[:,li]*polarity[:,None],axis=1)]
    ridx=ri[np.argmax(-gradient[:,ri]*polarity[:,None],axis=1)]
    measured_width=(mid[ridx]-mid[lidx])*ww
    il=(o>=-.40)&(o<=-.20);ol=(o>=-.75)&(o<=-.58)
    ir=(o>=.20)&(o<=.40);orr=(o>=.58)&(o<=.75)
    left_edge=polarity*(rawprof[:,il].mean(axis=1)-rawprof[:,ol].mean(axis=1))
    right_edge=polarity*(rawprof[:,ir].mean(axis=1)-rawprof[:,orr].mean(axis=1))
    paired=np.minimum(left_edge,right_edge);strong=np.maximum(left_edge,right_edge)
    width_ratio=measured_width/ww
    # 每个切向小截面都与它自身两旁背景相减；抑制斜着穿过窗口的旧纹。
    local_body=raw[:,:,body].mean(axis=2)
    local_bg=.5*(raw[:,:,left].mean(axis=2)+raw[:,:,right].mean(axis=2))
    local_difference=local_body-local_bg
    along_consistency=np.mean(polarity[:,None]*local_difference>=np.maximum(1.,.20*contrast)[:,None],axis=1)
    dynamic=np.percentile(raw,95,axis=(1,2))-np.percentile(raw,5,axis=(1,2))
    background_noise=np.median(abs(local_bg-np.median(local_bg,axis=1)[:,None]),axis=1)*1.4826
    # 原始灰度噪声尺度有下限，暗图不能仅靠归一化把1灰度噪声变成强信号。
    scale=np.maximum(4.,np.maximum(.18*dynamic,.65*background_noise))
    normalized_contrast=contrast/scale;normalized_pair=paired/scale
    saturate=(raw<=1)|(raw>=253)
    body_sat=saturate[:,:,body].mean(axis=(1,2));patch_sat=saturate.mean(axis=(1,2))
    valid_fraction=valid.mean(axis=(1,2));signal_max=np.percentile(raw,95,axis=(1,2))
    reliable=(valid_fraction>=PARAMETERS['minimum_valid_fraction'])&(body_sat<=PARAMETERS['maximum_body_saturation_fraction'])&(patch_sat<=PARAMETERS['maximum_patch_saturation_fraction'])&(dynamic>=PARAMETERS['minimum_patch_dynamic_range'])&(signal_max>=PARAMETERS['minimum_signal_maximum'])
    # 阴影宽度单独量测：窄股阴影不会凭这一项直接获得整绳支持。
    shadow_threshold=np.maximum(2.,.25*scale)
    sl=(o>=-.85)&(o<=-.30);sr=(o>=.30)&(o<=.85)
    left_shadow=np.sum(residual[:,sl]<-shadow_threshold[:,None],axis=1)*(o[1]-o[0])*ww
    right_shadow=np.sum(residual[:,sr]<-shadow_threshold[:,None],axis=1)*(o[1]-o[0])*ww
    left_shadow_depth=np.maximum(0.,-np.min(residual[:,sl],axis=1));right_shadow_depth=np.maximum(0.,-np.min(residual[:,sr],axis=1))
    # 内带与左右背景的股纹方向差，只在有梯度能量时才有效。
    angle,coherence,energy=_tensor(raw[:,:,body],a,o[body],ww)
    al,cl,_=_tensor(raw[:,:,left],a,o[left],ww);ar,cr,_=_tensor(raw[:,:,right],a,o[right],ww)
    texture_distinction=.5*(cl*(1-np.cos(2*(angle-al)))+cr*(1-np.cos(2*(angle-ar))))*coherence/2
    free_advantage=np.zeros(n,np.float32)
    if free_texture and free_texture.get('status')=='supported' and free_texture.get('free_axis_angle_rad') is not None and free_texture.get('angle_deg') is not None:
        target=math.radians(float(free_texture['angle_deg']))-float(free_texture['free_axis_angle_rad'])
        self_match=coherence*np.cos(2*(angle-target));other_match=.5*(cl*np.cos(2*(al-target))+cr*np.cos(2*(ar-target)))
        free_advantage=np.maximum(0.,self_match-other_match)
    contrast_score=np.clip(normalized_contrast/1.2,0,1)
    edge_score=np.clip(normalized_pair/.8,0,1)
    raw_score=(.50*contrast_score+.30*edge_score+.15*along_consistency+.05*texture_distinction)
    raw_score*=np.clip(body_coverage/.75,0,1)*np.clip(along_consistency/.80,0,1)
    raw_score+=.025*np.clip(free_advantage,0,1)*contrast_score
    score=np.where(reliable,np.clip(raw_score,0,1),0.)
    width_valid=(width_ratio>=PARAMETERS['minimum_measured_width_ratio'])&(width_ratio<=PARAMETERS['maximum_measured_width_ratio'])
    supported=reliable&(contrast>=PARAMETERS['minimum_raw_body_contrast'])&(paired>=PARAMETERS['minimum_raw_paired_edge_contrast'])&(normalized_contrast>=.20)&(body_coverage>=PARAMETERS['minimum_body_coverage'])&(along_consistency>=PARAMETERS['minimum_along_consistency'])&width_valid&(score>=PARAMETERS['minimum_supported_score'])
    single=reliable&(strong>=2.)&(contrast>=2.)&(body_coverage>=.45)&(along_consistency>=.65)&~supported
    # 白色绳体可能曝光截断，但两条外轮廓仍可量测。它不是可用的均值
    # 差分，保持reliable/supported/score的原门不动，另报轮廓证据。
    # 两旁也饱和、全白、仅单边或窄亮丝均不能靠此通道被称为整绳。
    high_body_sat=np.mean(raw[:,:,body]>=253,axis=(1,2))
    bg_sat=np.c_[saturate[:,:,left].mean(axis=(1,2)),saturate[:,:,right].mean(axis=(1,2))]
    outline=(high_body_sat>PARAMETERS['maximum_body_saturation_fraction'])&(polarity>0)&(valid_fraction>=PARAMETERS['minimum_valid_fraction'])&(np.max(bg_sat,axis=1)<=PARAMETERS['outline_maximum_background_saturation_fraction'])&(paired>=PARAMETERS['outline_minimum_raw_edge_contrast'])&width_valid&(body_coverage>=PARAMETERS['outline_minimum_body_coverage'])&(along_consistency>=PARAMETERS['outline_minimum_along_consistency'])
    outline_quality=.80*np.clip(paired/np.maximum(12.,.25*dynamic),0,1)+.20*along_consistency
    outline_score=np.where(outline,np.clip(outline_quality,0,1),0.).astype(np.float32)
    # 未知原因是位标志，避免beam内重复构造长字符串。调用者可据此汇总。
    reasons=np.zeros(n,np.uint16)
    for bit,condition in [(1,valid_fraction<PARAMETERS['minimum_valid_fraction']),(2,body_sat>PARAMETERS['maximum_body_saturation_fraction']),(4,patch_sat>PARAMETERS['maximum_patch_saturation_fraction']),(8,(dynamic<PARAMETERS['minimum_patch_dynamic_range'])|(signal_max<PARAMETERS['minimum_signal_maximum'])),(16,contrast<PARAMETERS['minimum_raw_body_contrast']),(32,paired<PARAMETERS['minimum_raw_paired_edge_contrast']),(64,body_coverage<PARAMETERS['minimum_body_coverage']),(128,along_consistency<PARAMETERS['minimum_along_consistency']),(256,~width_valid)]:reasons|=np.where(condition,bit,0).astype(np.uint16)
    return {'score':score.astype(np.float32),'supported':supported,'reliable':reliable,'single_edge_support':single,
            'saturated_outline_supported':outline,'saturated_outline_score':outline_score,
            'high_body_saturation_fraction':high_body_sat,'background_saturation_fraction':bg_sat,
            'inner_mean_raw':inner_mean,'left_background_mean_raw':left_mean,'right_background_mean_raw':right_mean,
            'signed_body_background_difference_raw':body_difference,'polarity':polarity,
            'left_edge_contrast_raw':left_edge,'right_edge_contrast_raw':right_edge,'paired_edge_contrast_raw':paired,
            'normalized_body_contrast':normalized_contrast,'normalized_paired_contrast':normalized_pair,
            'body_coverage':body_coverage,'along_consistency':along_consistency,'background_asymmetry_raw':right_mean-left_mean,
            'background_noise_raw':background_noise,'normalization_scale_raw':scale,
            'left_shadow_width_px':left_shadow,'right_shadow_width_px':right_shadow,
            'left_shadow_depth_raw':left_shadow_depth,'right_shadow_depth_raw':right_shadow_depth,
            'measured_edge_width_px':measured_width,'measured_width_ratio':width_ratio,
            'body_saturation_fraction':body_sat,'patch_saturation_fraction':patch_sat,
            'raw_dynamic_range':dynamic,'valid_fraction':valid_fraction,
            'texture_distinction':texture_distinction,'free_texture_advantage':free_advantage,'interior_texture_coherence':coherence,
            'unknown_reason_bits':reasons,
            'unknown_reason_legend':{1:'image_border',2:'body_saturation',4:'patch_saturation',8:'low_raw_signal',16:'weak_body_background_difference',32:'no_paired_outer_edge',64:'too_narrow_or_inconsistent_body',128:'inconsistent_along_candidate',256:'outer_edge_width_mismatch'},
            'measurement_source':'original_image_only','identity_verified':False}


def compare_ribbon_neighbors(fields,xy,tangent,width,pitch=None,free_texture=None):
    """同条件比较中心与法向 ±1 绳距的平行邻带，不偷看候选结果图。

    返回中心sample_ribbon的全部字段，并增加comparison_score及左右邻带
    分数。差分为中心分−较强邻带分（-1..1）；三处原图均可量测时才有效。
    不可用邻带不作零分竞争者，避免凭缺信息给中心制造优势。此比较描述
    局部区分力，不能独自证明是活动绳；支持仍须通过中心原灰度/双边门。
    """
    p=np.asarray(xy,np.float32).reshape(-1,2);n=len(p)
    t=np.asarray(tangent,np.float32)
    if t.shape==(2,):t=np.repeat(t[None,:],n,axis=0)
    t=t.reshape(-1,2);length=np.linalg.norm(t,axis=1)
    if len(t)!=n or np.any(length<1e-6):raise ValueError('matching nonzero tangents required')
    t=t/length[:,None];normal=np.c_[-t[:,1],t[:,0]]
    ww=np.broadcast_to(np.asarray(width,np.float32), (n,)).copy()
    pp=np.broadcast_to(np.asarray(fields['pitch_px'] if pitch is None else pitch,np.float32),(n,)).copy()
    if not np.isfinite(pp).all() or np.any(pp<=1):raise ValueError('finite neighbor pitch >1 px required')
    q=np.vstack([p,p-pp[:,None]*normal,p+pp[:,None]*normal])
    measured=sample_ribbon(fields,q,np.tile(t,(3,1)),np.tile(ww,3),free_texture)
    center={k:(v[:n].copy() if isinstance(v,np.ndarray) else v) for k,v in measured.items()}
    left=measured['score'][n:2*n];right=measured['score'][2*n:]
    available=measured['reliable'][:n]&measured['reliable'][n:2*n]&measured['reliable'][2*n:]
    advantage=measured['score'][:n]-np.maximum(left,right)
    center.update({'neighbor_left_score':left.copy(),'neighbor_right_score':right.copy(),
                   'neighbor_left_reliable':measured['reliable'][n:2*n].copy(),'neighbor_right_reliable':measured['reliable'][2*n:].copy(),
                   'neighbor_left_delta_gray':measured['signed_body_background_difference_raw'][n:2*n].copy(),
                   'neighbor_right_delta_gray':measured['signed_body_background_difference_raw'][2*n:].copy(),
                   'comparison_score':np.where(available,advantage,0.).astype(np.float32),
                   'comparison_reliable':available,'comparison_offset_pitch_px':pp,
                   'comparison_meaning':'center score minus stronger same-condition parallel neighbor score; unavailable comparison stays unknown'})
    return center
