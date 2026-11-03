"""A5 贴轨阶段：在F3正式边界附近直接测原图的两条宽暗沟。

自由悬绳有“绳体到背景”的灰度台阶；贴合后相邻绳体可能同样亮，边界
表现为两边高、中间低的沟谷，不能复用自由段的台阶差来否决它。
本模块只量测给定边界处的原像素，不生成轨道，不修改F3，不确认同根。
"""
from __future__ import annotations
import cv2
import numpy as np

PARAMETERS={
    'normal_half_pitch':.60,'normal_samples':37,
    'tangent_half_pitch':.60,'tangent_samples':7,'tangent_trim':1,
    'valley_search_pitch':.18,
    'shoulder_inner_pitch':.30,'shoulder_outer_pitch':.58,
    'minimum_groove_width_pitch':.12,'maximum_groove_width_pitch':.65,
    'minimum_raw_weak_contrast':1.,'minimum_raw_strong_contrast':2.,
    'minimum_raw_range':3.,'maximum_saturation_fraction':.65,
    'minimum_valid_fraction':.97,'minimum_along_consistency':.55,
}


def _sample(a,q):
    shape=q.shape[:-1];q=np.asarray(q,np.float32)
    mx=q[...,0].reshape(-1,shape[-1]);my=q[...,1].reshape(-1,shape[-1]);out=np.empty(mx.shape,np.float32)
    for start in range(0,len(mx),32000):
        out[start:start+32000]=cv2.remap(a,mx[start:start+32000],my[start:start+32000],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return out.reshape(shape)


def sample_attached_grooves(fields,boundary_xy,tangent,widths,pitch=None):
    """返回N个轨道截面的原图双沟证据。纯函数，无文件/设备访问。

    fields: photometric_ribbon.build_fields返回对象（使用其中原gray）；
    boundary_xy: N×2×2，两条F3边界的原图xy；tangent: N×2或一个2向量；
    widths: 轨道法向宽度N数组或标量。默认它也是局部绳距，适应边侧投影。
    pitch可显式提供N数组/标量覆盖量测尺度，但不能混入毫米。
    score/supported/available为N；edge_*及宽度/反差/偏移为N×2。
    """
    b=np.asarray(boundary_xy,np.float32)
    if b.ndim!=3 or b.shape[1:]!=(2,2) or not len(b):raise ValueError('boundary_xy must be nonempty Nx2x2')
    n=len(b);t=np.asarray(tangent,np.float32)
    if t.shape==(2,):t=np.repeat(t[None,:],n,axis=0)
    if t.shape!=(n,2):raise ValueError('tangent must be Nx2 or 2')
    length=np.linalg.norm(t,axis=1)
    ww=np.broadcast_to(np.asarray(widths,np.float32),(n,)).copy()
    pp=np.broadcast_to(np.asarray(ww if pitch is None else pitch,np.float32),(n,)).copy()
    if not np.isfinite(np.r_[b.ravel(),t.ravel(),ww,pp]).all() or np.any(length<1e-6) or np.any(ww<=1) or np.any(pp<=1):raise ValueError('finite geometry and pixel scales >1 required')
    t=t/length[:,None];normal=np.c_[-t[:,1],t[:,0]]
    a=np.linspace(-PARAMETERS['tangent_half_pitch'],PARAMETERS['tangent_half_pitch'],PARAMETERS['tangent_samples'],dtype=np.float32)
    o=np.linspace(-PARAMETERS['normal_half_pitch'],PARAMETERS['normal_half_pitch'],PARAMETERS['normal_samples'],dtype=np.float32)
    q=b[:,:,None,None,:]+pp[:,None,None,None,None]*(a[None,None,:,None,None]*t[:,None,None,None,:]+o[None,None,None,:,None]*normal[:,None,None,None,:])
    h,w=fields['image_shape'];valid=(q[...,0]>=1)&(q[...,0]<w-2)&(q[...,1]>=1)&(q[...,1]<h-2)
    raw=_sample(fields['gray'],q)
    # 宽度直接在沿切向聚合后的原灰度上测，不用各向同性模糊放宽细股沟。
    trim=PARAMETERS['tangent_trim'];prof=np.sort(raw,axis=2)[:,:,trim:len(a)-trim,:].mean(axis=2)
    profile=prof.reshape(-1,len(o));count=len(profile)
    center=np.flatnonzero(abs(o)<=PARAMETERS['valley_search_pitch'])
    idx=center[np.argmin(profile[:,center],axis=1)]
    left=(o>=-PARAMETERS['shoulder_outer_pitch'])&(o<=-PARAMETERS['shoulder_inner_pitch'])
    right=(o>=PARAMETERS['shoulder_inner_pitch'])&(o<=PARAMETERS['shoulder_outer_pitch'])
    shoulders=np.c_[np.median(profile[:,left],axis=1),np.median(profile[:,right],axis=1)]
    value=profile[np.arange(count),idx];contrast=shoulders-value[:,None]
    weak=np.min(contrast,axis=1);strong=np.max(contrast,axis=1)
    half=value+.5*np.maximum(weak,0.)
    lo=idx.copy();hi=idx.copy()
    for i in range(count):
        while lo[i]>0 and profile[i,lo[i]-1]<half[i]:lo[i]-=1
        while hi[i]<len(o)-1 and profile[i,hi[i]+1]<half[i]:hi[i]+=1
    local_pitch=np.repeat(pp,2);step=float(o[1]-o[0])
    groove_width=(o[hi]-o[lo]+step)*local_pitch
    ratio=groove_width/local_pitch
    broad=(ratio>=PARAMETERS['minimum_groove_width_pitch'])&(ratio<=PARAMETERS['maximum_groove_width_pitch'])
    p95=np.percentile(raw,95,axis=(2,3)).ravel();p05=np.percentile(raw,5,axis=(2,3)).ravel();dynamic=p95-p05
    saturation=np.mean((raw<=1)|(raw>=253),axis=(2,3)).ravel();valid_fraction=valid.mean(axis=(2,3)).ravel()
    available=(valid_fraction>=PARAMETERS['minimum_valid_fraction'])&(dynamic>=PARAMETERS['minimum_raw_range'])&(saturation<=PARAMETERS['maximum_saturation_fraction'])
    per_along=raw.reshape(count,len(a),len(o))
    local_shoulders=np.minimum(np.median(per_along[:,:,left],axis=2),np.median(per_along[:,:,right],axis=2))
    valley_along=np.take_along_axis(per_along,idx[:,None,None].repeat(len(a),axis=1),axis=2)[:,:,0]
    consistency=np.mean(local_shoulders-valley_along>=PARAMETERS['minimum_raw_weak_contrast'],axis=1)
    support=available&broad&(weak>=PARAMETERS['minimum_raw_weak_contrast'])&(strong>=PARAMETERS['minimum_raw_strong_contrast'])&(consistency>=PARAMETERS['minimum_along_consistency'])
    scale=np.maximum(3.,.25*dynamic)
    edge_score=np.clip(weak/scale,0,1)*(.80+.20*consistency)
    # 沟太窄、太宽、方向不持续或原图不可用都不能靠归一化获得高分。
    edge_score=np.where(available&broad,edge_score,0.)
    shift=o[idx]*local_pitch
    measured=b.reshape(-1,2)+shift[:,None]*np.repeat(normal,2,axis=0)
    bits=np.zeros(count,np.uint16)
    for bit,fail in [(1,valid_fraction<PARAMETERS['minimum_valid_fraction']),(2,dynamic<PARAMETERS['minimum_raw_range']),(4,saturation>PARAMETERS['maximum_saturation_fraction']),(8,ratio<PARAMETERS['minimum_groove_width_pitch']),(16,ratio>PARAMETERS['maximum_groove_width_pitch']),(32,weak<PARAMETERS['minimum_raw_weak_contrast']),(64,strong<PARAMETERS['minimum_raw_strong_contrast']),(128,consistency<PARAMETERS['minimum_along_consistency'])]:bits|=np.where(fail,bit,0).astype(np.uint16)
    scores=edge_score.reshape(n,2);supports=support.reshape(n,2);usable=available.reshape(n,2)
    return {'score':np.min(scores,axis=1).astype(np.float32),'supported':supports.all(axis=1),'available':usable.all(axis=1),'reliable':usable.all(axis=1),
            'edge_scores':scores,'edge_support':supports,'edge_available':usable,
            'groove_width_px':groove_width.reshape(n,2),'groove_width_ratio':ratio.reshape(n,2),
            'raw_depth_weak':weak.reshape(n,2),'raw_depth_strong':strong.reshape(n,2),
            'raw_valley_gray':value.reshape(n,2),'raw_shoulders_gray':shoulders.reshape(n,2,2),
            'valley_shift_px':shift.reshape(n,2),'measured_valley_xy':measured.reshape(n,2,2),
            'along_consistency':consistency.reshape(n,2),'saturation_fraction':saturation.reshape(n,2),
            'raw_dynamic_range':dynamic.reshape(n,2),'valid_fraction':valid_fraction.reshape(n,2),
            'unknown_reason_bits':bits.reshape(n,2),'unknown_reason_legend':{1:'image_border',2:'low_raw_range',4:'saturation',8:'narrow_strand_groove',16:'overwide_shadow',32:'weak_one_shoulder',64:'weak_both_shoulders',128:'not_persistent_along_boundary'},
            'measurement_source':'original_gray_at_formal_boundary_proposals','identity_verified':False}
