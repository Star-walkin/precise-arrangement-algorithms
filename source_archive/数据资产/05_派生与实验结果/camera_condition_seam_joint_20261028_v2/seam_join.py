"""F2纯数组接续：多候选图像对应、端切向Hermite桥、歧义拒绝。

旧F1在T外逐像素保留；T内只替换参与接受接续的旧曲线段。
这里的“同一边界组”是单帧图像对应，绝不表示同根物理材料。
"""
from __future__ import annotations

import math
import cv2
import numpy as np

PARAMETERS = {
    'minimum_anchor_samples': 4,
    'tangent_window_pitch': 1.25,
    'minimum_join_span_pitch': .45,
    'maximum_join_span_pitch': 14.,
    'endpoint_integral_error_pitch': 1.25,
    'maximum_absolute_slope': 2.7,
    'maximum_derivative_reversal': .12,
    'observation_corridor_pitch': .30,
    'maximum_bump_correction_pitch': .20,
    'minimum_new_observations': 6,
    'minimum_observed_span_fraction': .50,
    'maximum_observation_residual_p90_pitch': .27,
    'broad_support_threshold': .10,
    'minimum_broad_support_fraction': .55,
    'maximum_unsupported_arc_pitch': 1.25,
    'body_hard_threshold': .65,
    'minimum_pairing_cost_margin': .12,
    'minimum_parallel_separation_pitch': .40,
}


def _runs(mask):
    a=np.r_[False,np.asarray(mask,bool),False].astype(np.int8)
    return list(zip(np.flatnonzero(np.diff(a)==1),np.flatnonzero(np.diff(a)==-1)))


def _at(mask,xy,outside=0):
    ij=np.rint(xy).astype(int);h,w=mask.shape
    okay=(ij[:,0]>=0)&(ij[:,0]<w)&(ij[:,1]>=0)&(ij[:,1]<h)
    values=np.full(len(ij),outside,dtype=mask.dtype)
    values[okay]=mask[ij[okay,1],ij[okay,0]]
    return values


def _field(field,name,xy,outside=0):
    local=np.asarray(xy,np.float32)-np.array([field['x0'],field['y0']],np.float32)
    return _at(field[name],local,outside)


def _paint_coordinates(xy,shape,polyline=True):
    """只开曲线局部矩形，不为每条候选分配完整float图。"""
    rr=np.rint(xy).astype(np.int32)
    if not len(rr):return np.empty((0,2),np.int32)
    h,w=shape;x0=max(0,int(rr[:,0].min()));x1=min(w,int(rr[:,0].max())+1)
    y0=max(0,int(rr[:,1].min()));y1=min(h,int(rr[:,1].max())+1)
    if x1<=x0 or y1<=y0:return np.empty((0,2),np.int32)
    if not polyline:
        take=(rr[:,0]>=0)&(rr[:,0]<w)&(rr[:,1]>=0)&(rr[:,1]<h)
        return np.unique(rr[take],axis=0)
    patch=np.zeros((y1-y0,x1-x0),np.uint8)
    cv2.polylines(patch,[rr-np.array([x0,y0])],False,255,1,cv2.LINE_8)
    yy,xx=np.nonzero(patch)
    return np.c_[xx+x0,yy+y0].astype(np.int32)


def _tangent(curve,xy,anchor,pitch,source):
    if source=='C3':
        m=curve.get('model',{})
        if m.get('type')=='vertical_axis_parabola' and m.get('u_scale_x_px',0)>0:
            scale=float(m['u_scale_x_px']);u=(anchor[0]-float(m['u_center_x_px']))/scale
            return float((float(m['normalized_b'])+2*float(m['normalized_c'])*u)/scale)
    take=np.abs(xy[:,0]-anchor[0])<=PARAMETERS['tangent_window_pitch']*pitch
    if np.count_nonzero(take)<3:
        take=np.zeros(len(xy),bool);take[np.argsort(np.abs(xy[:,0]-anchor[0]))[:min(8,len(xy))]]=True
    q=xy[take]
    if len(q)<2 or np.ptp(q[:,0])<1:return 0.
    return float(np.polyfit(q[:,0]-anchor[0],q[:,1],1)[0])


def _endpoints(data,t,owners,pitch):
    """每个源曲线在左右过渡带各取面向源内部的一端作锚点。

    左右由亮ROI的横向中心划分，仅用于本项目横向单值卷面曲线；
    没有亮ROI或没有足够旧样本时拒识，不猜入口。
    """
    yy,xx=np.nonzero(owners['C3'])
    if not len(xx):return [],[]
    center=float(np.median(xx));result=[];old=[]
    for source,key in [('C3','c3_record'),('D4','d4_record')]:
        for curve in data[key]['fit'].get('curves',[]):
            xy=np.asarray(curve.get('sampled_points_xy',[]),np.float64).reshape(-1,2)
            if len(xy)<PARAMETERS['minimum_anchor_samples'] or not np.isfinite(xy).all():continue
            xy=xy[np.argsort(xy[:,0],kind='stable')]
            identifier=source+':'+str(curve['curve_id'])
            old.append({'source':source,'id':identifier,'xy':xy})
            eligible=(_at(owners[source],xy)>0)&(_at(t,xy)>0)
            for side in (-1,1):
                side_ok=xy[:,0]<center if side<0 else xy[:,0]>=center
                for run_index,(a,z) in enumerate(_runs(eligible&side_ok)):
                    q=xy[a:z]
                    if len(q)<PARAMETERS['minimum_anchor_samples']:continue
                    # 右接缝：C3左端、D4右端；左接缝方向反过来。
                    use_first=(source=='C3' and side>0) or (source=='D4' and side<0)
                    anchor=q[0] if use_first else q[-1]
                    face=q[-1] if use_first else q[0]
                    tangent=_tangent(curve,xy,anchor,pitch,source)
                    if not math.isfinite(tangent) or abs(tangent)>PARAMETERS['maximum_absolute_slope']:continue
                    raster=_paint_coordinates(xy,t.shape,source=='D4')
                    if len(raster):
                        inside=(_at(t,raster)>0)&(_at(owners[source],raster)>0)
                        inside&=(raster[:,0]>=math.floor(q[0,0]))&(raster[:,0]<=math.ceil(q[-1,0]))
                        raster=raster[inside]
                    result.append({'endpoint_id':identifier+f':side{side}:part{run_index}',
                        'curve_id':identifier,'source':source,'side':side,
                        'anchor':anchor,'face':face,'tangent':tangent,'remove_xy':raster,
                        'source_span_x':[float(q[0,0]),float(q[-1,0])]})
    return result,old


def _hermite(left,right,correction=0.):
    x0,y0=left['anchor'];x1,y1=right['anchor'];span=x1-x0
    x=np.arange(math.ceil(x0),math.floor(x1)+1,dtype=np.float64)
    u=(x-x0)/span;m0,m1=left['tangent'],right['tangent']
    y=(2*u**3-3*u*u+1)*y0+(u**3-2*u*u+u)*span*m0+(-2*u**3+3*u*u)*y1+(u**3-u*u)*span*m1
    bump=16*u*u*(1-u)**2
    y+=correction*bump
    derivative=((6*u*u-6*u)*y0+(3*u*u-4*u+1)*span*m0+(-6*u*u+6*u)*y1+(3*u*u-2*u)*span*m1)/span
    derivative+=correction*(32*u*(1-u)*(1-2*u))/span
    return np.c_[x,y],derivative,bump


def _observations(detected):
    points=[]
    for source,key in [('C3','c3'),('D4','d4')]:
        for index,p in enumerate(detected[key].get('points',[])):
            x,y=float(p['x']),float(p['y'])
            if not math.isfinite(x) or not math.isfinite(y):continue
            points.append({**p,'source':source,'id':str(p.get('id',f'{source}_new_{index:06d}')),
                           'x':x,'y':y,'score':float(np.clip(p.get('score',1.),.05,1.))})
    coords=np.asarray([[p['x'],p['y']] for p in points],np.float64).reshape(-1,2)
    return points,coords


def _candidate(bright,dark,points,coords,detected,t,pitch,identifier):
    left,right=(dark,bright) if bright['side']<0 else (bright,dark)
    span=float(right['anchor'][0]-left['anchor'][0])
    reasons=[]
    stats={'span_px':span}
    base={'candidate_id':identifier,'bright_endpoint':bright['endpoint_id'],
          'dark_endpoint':dark['endpoint_id'],'source_curve_ids':[bright['curve_id'],dark['curve_id']],
          'side':bright['side'],'status':'rejected','observation_ids':[],'statistics':stats}
    if not PARAMETERS['minimum_join_span_pitch']*pitch<=span<=PARAMETERS['maximum_join_span_pitch']*pitch:
        base['reasons']=['endpoint_span'];return base,None
    integral_error=abs((right['anchor'][1]-left['anchor'][1])-.5*(left['tangent']+right['tangent'])*span)
    stats['endpoint_integral_error_px']=float(integral_error)
    if integral_error>PARAMETERS['endpoint_integral_error_pitch']*pitch:
        base['reasons']=['endpoint_position_tangent_incompatible'];return base,None
    xy,derivative,bump=_hermite(left,right)
    geometric=xy.copy()
    inspan=(coords[:,0]>=xy[0,0])&(coords[:,0]<=xy[-1,0]) if len(coords) else np.zeros(0,bool)
    residual=np.abs(coords[:,1]-np.interp(coords[:,0],xy[:,0],xy[:,1])) if len(coords) else np.zeros(0)
    selected=np.flatnonzero(inspan&(residual<=PARAMETERS['observation_corridor_pitch']*pitch))
    correction=0.
    if len(selected)>=3:
        pxy=coords[selected];basis=np.interp(pxy[:,0],xy[:,0],bump)
        delta=pxy[:,1]-np.interp(pxy[:,0],xy[:,0],xy[:,1])
        weights=np.asarray([points[i]['score'] for i in selected])
        correction=float(np.clip(np.sum(weights*basis*delta)/max(1e-5,np.sum(weights*basis*basis)),
                     -PARAMETERS['maximum_bump_correction_pitch']*pitch,PARAMETERS['maximum_bump_correction_pitch']*pitch))
        xy,derivative,_=_hermite(left,right,correction)
        residual=np.abs(coords[:,1]-np.interp(coords[:,0],xy[:,0],xy[:,1]))
        selected=np.flatnonzero(inspan&(residual<=PARAMETERS['observation_corridor_pitch']*pitch))
    stats['correction_px']=correction
    # 两路检测命中同一像素只算一个位置，不能把重复检测当独立覆盖。
    unique={tuple(np.rint(coords[i]).astype(int)) for i in selected}
    obs_span=(float(np.ptp(coords[selected,0]))/span if len(selected) else 0.)
    q90=float(np.percentile(residual[selected],90)) if len(selected) else None
    stats.update({'new_observation_count':len(selected),'unique_observation_positions':len(unique),
                  'observation_span_fraction':obs_span,'observation_residual_q90_px':q90,
                  'new_sources':sorted({points[i]['source'] for i in selected})})
    if len(unique)<PARAMETERS['minimum_new_observations']:reasons.append('insufficient_new_yellow_points')
    if obs_span<PARAMETERS['minimum_observed_span_fraction']:reasons.append('new_points_do_not_cover_join')
    if q90 is not None and q90>PARAMETERS['maximum_observation_residual_p90_pitch']*pitch:reasons.append('new_group_residual')
    reversal=float(np.sum(np.maximum(0.,-np.diff(derivative))))
    stats['derivative_reversal']=reversal
    if np.max(np.abs(derivative))>PARAMETERS['maximum_absolute_slope'] or reversal>PARAMETERS['maximum_derivative_reversal']:
        reasons.append('tangent_or_curvature_reversal')
    pixels=_paint_coordinates(xy,t.shape)
    fields=detected['fields']
    body=_field(fields,'body_barrier',pixels,outside=1.)
    unknown=_field(fields,'unknown_mask',pixels,outside=255)
    broad=_field(fields,'broad_score',pixels,outside=0.)
    # 栅格化后的每个像素都受硬门约束，禁止靠裁切掩盖越界桥。
    if len(pixels)==0 or np.any(_at(t,pixels)==0):reasons.append('bridge_outside_transition')
    if np.any(unknown>0) or np.any(_at(detected['exclusion_mask'],pixels)>0):reasons.append('bridge_unknown_occlusion')
    if np.any(body>PARAMETERS['body_hard_threshold']):reasons.append('bridge_crosses_strong_body')
    supported=broad>=PARAMETERS['broad_support_threshold']
    support_fraction=float(np.mean(supported)) if len(supported) else 0.
    # 沿有序中心线量缺口长度；raster坐标按行排序，不能拿它量连续弧长。
    center_supported=_field(fields,'broad_score',xy)>=PARAMETERS['broad_support_threshold']
    lengths=np.linalg.norm(np.diff(xy,axis=0),axis=1)
    longest=max((float(np.sum(lengths[a:min(z,len(lengths))])) for a,z in _runs(~center_supported)),default=0.)
    stats.update({'broad_support_fraction':support_fraction,'longest_unsupported_arc_px':longest,
                  'strong_body_pixels':int(np.count_nonzero(body>PARAMETERS['body_hard_threshold'])),
                  'unknown_pixels':int(np.count_nonzero(unknown))})
    if support_fraction<PARAMETERS['minimum_broad_support_fraction']:reasons.append('weak_broad_groove_support')
    if longest>PARAMETERS['maximum_unsupported_arc_pitch']*pitch:reasons.append('long_image_support_gap')
    cost=.25*integral_error/pitch+.45*(float(np.median(residual[selected]))/pitch if len(selected) else 1.)+.60*(1-support_fraction)
    base.update({'status':'viable' if not reasons else 'rejected','reasons':reasons,
                 'observation_ids':[points[i]['id'] for i in selected],
                 'cost':float(cost),'sampled_points_xy':xy.tolist(),
                 'anchor_points_xy':[left['anchor'].tolist(),right['anchor'].tolist()],
                 'anchor_tangents':[float(left['tangent']),float(right['tangent'])],
                 'continuous_endpoint_tangents_preserved':True})
    return base,{'xy':xy,'pixels':pixels,'inferred_pixels':pixels[~supported],
                 'geometric':geometric,'selected':selected,'bright':bright,'dark':dark}


def join_seams(data,detected):
    """返回全尺寸正式mask、T内增删像素、新组/接受接续/全部拒绝理由。"""
    pitch=float(data.get('pitch',data['c3_record']['record']['pitch_px']))
    surface=data['surface']>0
    t=(detected['overlap_mask']>0)&surface
    bright=surface&(data['bright']>0);dark=surface&(data['dark']>0)&~bright
    baseline=((data['c3_smooth']>0)&bright)|((data['d4_smooth']>0)&dark)
    baseline_inferred=((data['c3_inferred']>0)&bright)|((data['d4_inferred']>0)&dark)
    if 'f1_merged' in data and not np.array_equal(baseline,data['f1_merged']>0):raise ValueError('F1 base mismatch')
    if 'f1_inferred' in data and not np.array_equal(baseline_inferred,data['f1_inferred']>0):raise ValueError('F1 inference mismatch')
    endpoints,old_paths=_endpoints(data,t,{'C3':bright,'D4':dark},pitch)
    points,coords=_observations(detected)
    candidates=[];internal={};simple=np.zeros(surface.shape,np.uint8)
    simple_best={}
    for a in [e for e in endpoints if e['source']=='C3']:
        for b in [e for e in endpoints if e['source']=='D4' and e['side']==a['side']]:
            # 远离预计同行的候选不用做整条采样，但保留由端切向定义的邻绳竞争。
            left,right=(b,a) if a['side']<0 else (a,b)
            span=right['anchor'][0]-left['anchor'][0]
            if span<=0 or span>PARAMETERS['maximum_join_span_pitch']*pitch:continue
            err=abs((right['anchor'][1]-left['anchor'][1])-.5*(left['tangent']+right['tangent'])*span)
            if err>PARAMETERS['endpoint_integral_error_pitch']*pitch:continue
            identifier=f'new_visual_group_{len(candidates)+1:06d}'
            proposal,cache=_candidate(a,b,points,coords,detected,t,pitch,identifier)
            candidates.append(proposal)
            if cache is not None:
                internal[identifier]=cache
                if a['endpoint_id'] not in simple_best or err<simple_best[a['endpoint_id']][0]:
                    simple_best[a['endpoint_id']]=(float(err),cache['geometric'])
    for _,xy in simple_best.values():
        p=_paint_coordinates(xy,t.shape);p=p[_at(t,p)>0]
        if len(p):simple[p[:,1],p[:,0]]=255

    viable=[c for c in candidates if c['status']=='viable']
    choices={}
    for c in viable:
        for key in ('bright_endpoint','dark_endpoint'):
            choices.setdefault(c[key],[]).append(c)
    for options in choices.values():options.sort(key=lambda c:c['cost'])
    accepted=[];used=set();used_observations=set();new=np.zeros_like(simple);remove=np.zeros_like(simple);new_inferred=np.zeros_like(simple)
    for c in sorted(viable,key=lambda q:q['cost']):
        reasons=[]
        for key in ('bright_endpoint','dark_endpoint'):
            options=choices[c[key]]
            if options[0]['candidate_id']!=c['candidate_id']:reasons.append('not_mutual_best_pair')
            if len(options)>1 and options[1]['cost']-options[0]['cost']<PARAMETERS['minimum_pairing_cost_margin']:
                reasons.append('ambiguous_neighbor_pairing')
            if c[key] in used:reasons.append('endpoint_already_used')
        cache=internal[c['candidate_id']];xy=cache['xy']
        for prior in accepted:
            q=np.asarray(prior['sampled_points_xy']);lo=max(xy[0,0],q[0,0]);hi=min(xy[-1,0],q[-1,0])
            if hi<lo:continue
            xx=np.arange(math.ceil(lo),math.floor(hi)+1)
            delta=np.interp(xx,xy[:,0],xy[:,1])-np.interp(xx,q[:,0],q[:,1])
            if len(delta) and (np.min(np.abs(delta))<PARAMETERS['minimum_parallel_separation_pitch']*pitch or np.min(delta)<0<np.max(delta)):
                reasons.append('accepted_pair_order_or_spacing_conflict');break
        if any(pid in used_observations for pid in c['observation_ids']):reasons.append('new_observation_shared_between_joins')
        # 不参与本次接续的旧曲线不能被桥穿过；仅替换指定两源片段。
        removal=np.vstack([cache['bright']['remove_xy'],cache['dark']['remove_xy']])
        removed_keys=set((removal[:,1]*surface.shape[1]+removal[:,0]).tolist()) if len(removal) else set()
        p=cache['pixels'];hits=p[_at(baseline,p)>0]
        if any(int(y)*surface.shape[1]+int(x) not in removed_keys for x,y in hits):
            reasons.append('bridge_hits_unmatched_existing_curve')
        if reasons:
            c['status']='rejected';c['reasons']=sorted(set(reasons));continue
        c['status']='accepted';c['visual_correspondence_only']=True;c['same_physical_rope_identity']=None
        c['removed_pixel_count']=len(removed_keys)
        accepted.append(c);used.update([c['bright_endpoint'],c['dark_endpoint']]);used_observations.update(c['observation_ids'])
        if len(removal):remove[removal[:,1],removal[:,0]]=255
        if len(p):new[p[:,1],p[:,0]]=255
        ip=cache['inferred_pixels']
        if len(ip):new_inferred[ip[:,1],ip[:,0]]=255
    remove[(~t)|(~baseline)]=0;new[~t]=0;new_inferred[~t]=0
    final=(baseline&(remove==0))|(new>0)
    inferred=(baseline_inferred&(remove==0))|(new_inferred>0)
    inferred&=final
    added=final&~baseline;removed=baseline&~final
    if np.any((added|removed)&~t) or not np.array_equal(inferred[~t],baseline_inferred[~t]):raise AssertionError('outside T changed')
    accepted_points=[p for p in points if p['id'] in used_observations]
    # 同名“组”保留所有失败候选及其新观测，不能只导出成功例。
    endpoint_records=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in e.items() if k!='remove_xy'} for e in endpoints]
    return {'merged_mask':final.astype(np.uint8)*255,'inferred_mask':inferred.astype(np.uint8)*255,
        'baseline_mask':baseline.astype(np.uint8)*255,'transition_mask':t.astype(np.uint8)*255,
        'added_mask':added.astype(np.uint8)*255,'removed_mask':removed.astype(np.uint8)*255,
        'replacement_mask':remove,'new_join_mask':new,'simple_bridge_mask':simple,
        'new_groups':candidates,'joins':accepted,'accepted_points':accepted_points,
        'rejections':[c for c in candidates if c['status']!='accepted'],
        'endpoints':endpoint_records,
        'diagnostics':{'parameters':PARAMETERS.copy(),'old_endpoint_count':len(endpoints),
          'candidate_count':len(candidates),'viable_before_pairing':len(viable),'accepted_joins':len(accepted),
          'accepted_new_point_count':len(accepted_points),'added_pixels':int(np.count_nonzero(added)),
          'removed_pixels':int(np.count_nonzero(removed)),'outside_transition_unchanged':True,
          'simple_bridge_meaning':'Endpoint-only geometric comparator; not accepted prediction or truth.',
          'meaning':'New image-supported ordered visual correspondence; no cross-source physical identity claim.'}}


__all__=['join_seams','PARAMETERS']
