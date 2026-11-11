"""F2重叠区重新观测：同一连续增强、同一T，实际运行冻结B1/C3和D4。

本模块只接收当前图像数组与方向参考，不读数据集或旧黄点作新候选。
旧C3/D4源码仅按只读纯算法路径导入；不导入采集、设备或UI入口。
输出的新黄点、分组、各算法曲线不是最终接缝接受结果，也不证明同根材料。
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path
import sys
from threading import Lock
import time

import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
ASSETS=HERE.parent
D4=ASSETS/'camera_condition_dark_width_20261027_v4'
DEPENDENCIES={
    'b1':ASSETS/'camera_condition_bright_rope_boundaries_20261025_v3/full_roi_observation.py',
    'c3':ASSETS/'camera_condition_grouped_conic_20261026_v6/conic_family.py',
    'raw_recheck':ASSETS/'camera_condition_dark_roi_20261026_v1/dark_c3.py',
    'shadow':D4/'shadow_width.py','period':D4/'period_guard.py',
    'tracks':D4/'shadow_tracks.py','curves':D4/'ribbon_curves.py',
}
SOURCE_DEPENDENCIES=list(DEPENDENCIES.values())
PARAMETERS={'overlap_distance_pitch':6.,'normalization_sigma_pitch':4.,
            'normalization_std_floor':2.5,'normalization_center':128.,
            'normalization_contrast':42.,'normalization_margin_pitch':12.}
_MODULES={};_LOCK=Lock()


def source_dependencies():
    return list(SOURCE_DEPENDENCIES)


def _module(key):
    with _LOCK:
        if key not in _MODULES:
            path=DEPENDENCIES[key]
            spec=importlib.util.spec_from_file_location('f2_frozen_'+key,path)
            if spec is None or spec.loader is None:raise RuntimeError('missing frozen dependency: '+str(path))
            obj=importlib.util.module_from_spec(spec)
            sys.modules[spec.name]=obj  # B1 dataclass需在加载时可查到模块。
            spec.loader.exec_module(obj);_MODULES[key]=obj
        return _MODULES[key]


def _binary(mask,shape,name):
    if not isinstance(mask,np.ndarray) or mask.shape!=shape or mask.dtype!=np.uint8:
        raise ValueError(name+' must be same-size uint8 mask')
    if np.any((mask!=0)&(mask!=255)):raise ValueError(name+' must be0/255')
    return mask>0


def _weighted_blur(value,valid,sigma):
    den=cv2.GaussianBlur(valid,(0,0),sigma)
    return cv2.GaussianBlur(value*valid,(0,0),sigma)/np.maximum(den,1e-5)


def _normalize_surface(original,surface,pitch):
    """在整个已有卷面上连续估计照明/反差，不按B/D分块，也不用D1硬接缝。

    局部均值和标准差仅增强显示/提议；保留点仍需原灰度弱谷复核。
    低纹理区的标准差设下限，避免把近乎全黑的微小噪声无限放大。
    """
    out=np.zeros_like(original)
    yy,xx=np.nonzero(surface)
    if not len(xx):return out,{'status':'empty_surface'}
    h,w=surface.shape;margin=int(math.ceil(PARAMETERS['normalization_margin_pitch']*pitch))
    x0=max(0,int(xx.min())-margin);x1=min(w,int(xx.max())+margin+1)
    y0=max(0,int(yy.min())-margin);y1=min(h,int(yy.max())+margin+1)
    valid=(surface[y0:y1,x0:x1]>0).astype(np.float32)
    raw=cv2.cvtColor(original[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY).astype(np.float32)
    sigma=PARAMETERS['normalization_sigma_pitch']*pitch
    mean=_weighted_blur(raw,valid,sigma)
    variance=np.maximum(_weighted_blur(raw*raw,valid,sigma)-mean*mean,0)
    std=np.sqrt(variance)
    z=(raw-mean)/np.maximum(std,PARAMETERS['normalization_std_floor'])
    gray=np.clip(PARAMETERS['normalization_center']+PARAMETERS['normalization_contrast']*z,0,255)
    gray=np.rint(gray).astype(np.uint8);gray[valid==0]=0
    out[y0:y1,x0:x1]=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    return out,{'status':'completed','bbox_xyxy':[x0,y0,x1,y1],
        'sigma_px':sigma,'std_floor':PARAMETERS['normalization_std_floor'],
        'whole_surface_single_rule':True,'owner_masks_used_in_normalization':False,
        'median_local_std':float(np.median(std[valid>0]))}


def _overlap(surface,bright,dark,pitch):
    s=surface>0;b=s&(bright>0);d=s&(dark>0)
    if not np.any(b) or not np.any(d):
        return np.zeros(surface.shape,np.uint8),{'status':'missing_bright_or_dark_region'}
    to_b=cv2.distanceTransform((~b).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    to_d=cv2.distanceTransform((~d).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    span=PARAMETERS['overlap_distance_pitch']*pitch
    t=s&(to_b<=span)&(to_d<=span)
    return t.astype(np.uint8)*255,{'status':'completed' if np.any(t) else 'no_overlap_within_budget',
        'distance_limit_px':span,'pixels':int(t.sum()),
        'bright_side_pixels':int(np.count_nonzero(t&b)),
        'dark_side_pixels':int(np.count_nonzero(t&d)),
        'previously_unowned_gap_pixels':int(np.count_nonzero(t&~b&~d))}


def _c3_points(lines,fit):
    assignment=fit.get('diagnostics',{}).get('fragment_assignment',{})
    points=[]
    for line in lines:
        evidence={tuple(e['xy']):e for e in line.get('original_point_evidence',[])}
        for x,y in line.get('observed_points_xy',[]):
            e=evidence.get((x,y),{});weak=float(e.get('original_weak_contrast',0.));strong=float(e.get('original_strong_contrast',0.))
            score=.5*weak/(1+max(0.,weak))+.5*strong/(2+max(0.,strong))
            i=len(points)
            points.append({'id':f'C3_new_{i:06d}','point_id':i,'source':'C3','x':int(x),'y':float(y),
                'score':float(np.clip(score,0,1)),'line_id':line['line_id'],
                'group_id':assignment.get(line['line_id']),
                'original_weak_contrast':weak,'original_strong_contrast':strong,
                'evidence':'new_B1_candidate_rechecked_in_original_gray_not_a_truth_label'})
    return points


def detect_overlap(original_bgr,surface,bright_roi,dark_roi,pitch,c3_record,c3_exclusion):
    """所有mask为原图H×W 0/255；场的x0/y0为原图坐标，内部数组为同裁块。

    返回 c3/d4 各自新points和原算法fit/curves；最终接缝配对必须另查
    原图支持、body/unknown及行序。C3 sampled_points_xy为整数原图坐标，
    D4为浮点原图坐标。任何源无候选时保留空列表，不回填旧黄点。
    """
    if not isinstance(original_bgr,np.ndarray) or original_bgr.dtype!=np.uint8 or original_bgr.ndim!=3 or original_bgr.shape[2]!=3:
        raise ValueError('original_bgr must be uint8 HxWx3')
    shape=original_bgr.shape[:2]
    for name,mask in [('surface',surface),('bright_roi',bright_roi),('dark_roi',dark_roi)]:_binary(mask,shape,name)
    if c3_exclusion is None:c3_exclusion=np.zeros(shape,np.uint8)
    _binary(c3_exclusion,shape,'c3_exclusion')
    pitch=float(pitch)
    if not math.isfinite(pitch) or not 8<=pitch<=70:raise ValueError('pitch outside8..70 pixels')
    times={};start=time.perf_counter()
    t,td=_overlap(surface,bright_roi,dark_roi,pitch)
    normalized,nd=_normalize_surface(original_bgr,surface,pitch)
    times['overlap_and_normalization_s']=time.perf_counter()-start
    reference=(((c3_record or {}).get('fit') or {}).get('model') or {}).get('shared_shape_for_grouping_and_regularization')
    tick=time.perf_counter()
    shadow=_module('shadow').build_shadow_field(original_bgr,normalized,t,pitch,reference)
    # 同一强遮挡并集给双方。unknown处body为0也绝不能当安全接续通道。
    shadow['excluded_mask']=np.where((t>0)&(c3_exclusion>0),255,0).astype(np.uint8)
    guard=_module('period').build_period_guard(original_bgr,normalized,t,pitch,reference,shadow)
    x0,y0=shadow['x0'],shadow['y0'];hh,ww=shadow['body_score'].shape
    exclusion=np.zeros(shape,np.uint8)
    if hh and ww:exclusion[y0:y0+hh,x0:x0+ww]=guard['unknown_mask']
    exclusion[t==0]=0
    effective=t.copy();effective[exclusion>0]=0
    times['shadow_and_guard_s']=time.perf_counter()-tick;tick=time.perf_counter()
    # 原B1在T重新找点，原图复核后再调用原C3；不把旧曲线坐标当观察。
    _,proposals,b1_diag=_module('b1').detect_full_roi(normalized,t,exclusion,pitch)
    lines,recheck=_module('raw_recheck')._recheck_original(original_bgr,effective,proposals,pitch)
    fit_c3=_module('c3').fit_conic_family(lines,pitch,t,exclusion)
    c3_points=_c3_points(lines,fit_c3)
    times['new_b1_c3_s']=time.perf_counter()-tick;tick=time.perf_counter()
    if hh and ww:
        detected=_module('tracks').detect_and_group(shadow,guard,t,pitch)
    else:
        detected={'points':[],'tracks':[],'rejected_points':[],'rejected_links':[],
                  'observed_mask':np.zeros(shape,np.uint8),'diagnostics':{'status':'empty_overlap'}}
    fit_d4=_module('curves').fit_validated_curves(detected['tracks'],t,pitch,shadow,guard)
    d4_points=[{**p,'source':'D4','id':f"D4_new_{p['point_id']:06d}"} for p in detected['points']]
    times['new_d4_group_fit_s']=time.perf_counter()-tick
    fields={'x0':x0,'y0':y0,'broad_score':shadow['broad_score'],'width_px':shadow['width_px'],
            'tangent_slope':shadow['tangent_slope'],'body_barrier':guard['body_barrier'],
            'unknown_mask':guard['unknown_mask'],'valid_support':shadow['valid_support']}
    return {'overlap_mask':t,'effective_overlap_mask':effective,'normalized_bgr':normalized,
        'c3':{'points':c3_points,'observed_lines':lines,'fit':fit_c3,'curves':fit_c3['curves'],
              'diagnostics':{'b1':b1_diag,'original_recheck':recheck,
                'proposed_points':sum(len(p.get('observed_points_xy',[])) for p in proposals),
                'rechecked_points':len(c3_points),'old_observed_points_used':False}},
        'd4':{'points':d4_points,'tracks':detected['tracks'],'fit':fit_d4,'curves':fit_d4['curves'],
              'rejected_points':detected['rejected_points'],'rejected_links':detected['rejected_links'],
              'diagnostics':detected['diagnostics']},
        'shadow':shadow,'guard':guard,'fields':fields,'exclusion_mask':exclusion,
        'source_dependencies':[str(p) for p in SOURCE_DEPENDENCIES],
        'diagnostics':{'status':'completed' if np.any(t) else 'empty_overlap','overlap':td,
            'normalization':nd,'stage_seconds':times,'parameters':PARAMETERS.copy(),
            'new_c3_point_count':len(c3_points),'new_d4_point_count':len(d4_points),
            'c3_curve_count':len(fit_c3['curves']),'d4_curve_count':len(fit_d4['curves']),
            'unknown_pixels':int(np.count_nonzero(exclusion)),
            'same_overlap_mask_for_both_algorithms':True,'original_weak_valley_required':True,
            'old_c3_usage':'shared shape direction reference only; no old yellow coordinates',
            'normalization_owner_split':False,'physical_rope_identity_confirmed':False,
            'limitation':'accepted source curves remain proposals for seam matching; no cross-source identity implied'}}


__all__=['detect_overlap','SOURCE_DEPENDENCIES','source_dependencies','PARAMETERS']
