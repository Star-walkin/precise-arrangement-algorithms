"""F3完整卷面重新观测：冻结连续归一化 + 冻结D4整链。

与F2只在过渡带观测不同，本模块把完整已有surface S交给D4，让真实
图像轨迹利用更长上下文。不修改D4任何阈值，也不拿旧黄点当新检测。
本模块不读取数据集、标签、设备或UI；只按固定路径加载纯算法源码。
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from threading import Lock
import time

import numpy as np

HERE=Path(__file__).resolve().parent
ASSETS=HERE.parent
D4=ASSETS/'camera_condition_dark_width_20261027_v4'
DEPENDENCIES={
    'normalization':ASSETS/'camera_condition_seam_joint_20261028_v2/seam_observation.py',
    'shadow':D4/'shadow_width.py','period':D4/'period_guard.py',
    'tracks':D4/'shadow_tracks.py','curves':D4/'ribbon_curves.py',
}
SOURCE_DEPENDENCIES=list(DEPENDENCIES.values())
_MODULES={};_LOCK=Lock()


def source_dependencies():
    return list(SOURCE_DEPENDENCIES)


def _module(name):
    with _LOCK:
        if name not in _MODULES:
            path=DEPENDENCIES[name]
            spec=importlib.util.spec_from_file_location('f3_surface_frozen_'+name,path)
            if spec is None or spec.loader is None:raise RuntimeError('missing pure dependency: '+str(path))
            module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
            spec.loader.exec_module(module);_MODULES[name]=module
        return _MODULES[name]


def _validate(data):
    for name in ('original','surface','pitch','c3_record','c3_exclusion'):
        if name not in data:raise ValueError('missing required input '+name)
    original=data['original']
    if (not isinstance(original,np.ndarray) or original.dtype!=np.uint8
        or original.ndim!=3 or original.shape[2]!=3):
        raise ValueError('original must be uint8 HxWx3 BGR')
    for name in ('surface','c3_exclusion'):
        mask=data[name]
        if not isinstance(mask,np.ndarray) or mask.dtype!=np.uint8 or mask.shape!=original.shape[:2]:
            raise ValueError(name+' must be same-size uint8 mask')
        if np.any((mask!=0)&(mask!=255)):raise ValueError(name+' must be0/255')
    pitch=float(data['pitch'])
    if not math.isfinite(pitch) or not 8<=pitch<=70:raise ValueError('pitch must be8..70 original pixels')
    return original,data['surface'],pitch,data['c3_exclusion']


def detect_surface(data):
    """返回原图尺度的新观测、轨迹、原D4拟合和局部图像证据场。

    points[i].point_id是D4本轮全局点号。fit曲线的source_point_indices
    则是该curve.track_id对应轨迹points的局部索引，不能当全局点索引。
    tracks中的点保留point_id/id，可追溯到points。所有坐标均为原图xy。
    exclusion_mask为旧排除与新unknown的全图并集，旧排除的任何像素均保留。
    """
    original,surface,pitch,old_exclusion=_validate(data)
    times={};tick=time.perf_counter()
    normalized,norm_diag=_module('normalization')._normalize_surface(original,surface,pitch)
    times['normalization_s']=time.perf_counter()-tick;tick=time.perf_counter()
    reference=(((data['c3_record'] or {}).get('fit') or {}).get('model') or {}).get('shared_shape_for_grouping_and_regularization')
    shadow=_module('shadow').build_shadow_field(original,normalized,surface,pitch,reference)
    # 完整旧排除直接交给原guard；guard会按其crop裁切，不可默默丢弃旧遮挡。
    shadow['excluded_mask']=old_exclusion
    times['shadow_s']=time.perf_counter()-tick;tick=time.perf_counter()
    guard=_module('period').build_period_guard(original,normalized,surface,pitch,reference,shadow)
    times['guard_s']=time.perf_counter()-tick;tick=time.perf_counter()
    x0,y0=int(shadow['x0']),int(shadow['y0']);hh,ww=shadow['body_score'].shape
    exclusion=old_exclusion.copy()
    if hh and ww:exclusion[y0:y0+hh,x0:x0+ww]|=guard['unknown_mask']
    if hh and ww:
        detected=_module('tracks').detect_and_group(shadow,guard,surface,pitch)
    else:
        detected={'points':[],'tracks':[],'rejected_points':[],'rejected_links':[],
                  'diagnostics':{'status':'empty_surface','new_point_count':0,'track_count':0}}
    # 加来源信息只新增字典字段，不挪动任何点，也不改变轨迹成员和顺序。
    points=[{**p,'source':'D4_full','id':f"D4_full_new_{p['point_id']:06d}"} for p in detected['points']]
    by_id={p['point_id']:p for p in points}
    tracks=[{**t,'points':[by_id[p['point_id']] for p in t['points']]} for t in detected['tracks']]
    times['group_s']=time.perf_counter()-tick;tick=time.perf_counter()
    fit=_module('curves').fit_validated_curves(tracks,surface,pitch,shadow,guard)
    times['fit_s']=time.perf_counter()-tick
    fields={'x0':x0,'y0':y0,'broad_score':shadow['broad_score'],
            'width_px':shadow['width_px'],'tangent_slope':shadow['tangent_slope'],
            'body_barrier':guard['body_barrier'],'unknown_mask':guard['unknown_mask'],
            'valid_support':shadow['valid_support'],'cleaned_shadow':shadow['cleaned_shadow']}
    return {'normalized_bgr':normalized,'points':points,'tracks':tracks,'fit':fit,
        'fields':fields,'exclusion_mask':exclusion,
        'source_dependencies':[str(p) for p in SOURCE_DEPENDENCIES],
        'diagnostics':{'status':'completed' if np.any(surface) else 'empty_surface',
            'observation_scope':'entire_existing_surface_S_not_F2_transition_band',
            'surface_pixels':int(np.count_nonzero(surface)),
            'old_exclusion_pixels':int(np.count_nonzero(old_exclusion)),
            'combined_exclusion_pixels':int(np.count_nonzero(exclusion)),
            'old_exclusion_preserved':bool(np.all(exclusion[old_exclusion>0]==255)),
            'new_point_count':len(points),'new_track_count':len(tracks),
            'accepted_curve_count':len(fit['curves']),
            'rejected_point_count':len(detected['rejected_points']),
            'rejected_link_count':len(detected['rejected_links']),
            'normalization':norm_diag,'shadow':shadow['diagnostics'],'period':guard['diagnostics'],
            'tracking':detected['diagnostics'],'fitting':fit['diagnostics'],'stage_seconds':times,
            'baseline_parameters_changed':False,'old_observations_used':False,
            'old_C3_usage':'shared shape direction reference only',
            'source_index_mapping':'curve.source_point_indices indexes tracks_by_id[curve.track_id].points; point_id maps into global points',
            'physical_rope_identity_confirmed':False,
            'limitation':'full existing surface still inherits original SAM omissions; image-supported curves are not truth labels'}}


__all__=['detect_surface','SOURCE_DEPENDENCIES','source_dependencies']
