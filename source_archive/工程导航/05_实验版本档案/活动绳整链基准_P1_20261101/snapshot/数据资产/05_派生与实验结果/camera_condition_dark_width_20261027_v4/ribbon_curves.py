"""D4：新黄点组的局部平滑与逐像素绳体检查（纯数组函数）。

这里不重新找点，也不挪动黄点；输入点必须来自本轮清洁阴影。
曲线仅表示图像中的沟槽走向，不能证明同根材料或三维接触。
宽度/亮体/未知遮挡仍可能判断错误，因此保留所有拆组和拒绝记录。
"""
from __future__ import annotations

import math
import cv2
import numpy as np

PARAMETERS = {
    "minimum_points": 8,
    "minimum_span_pitch": 3.0,
    "maximum_gap_pitch": 1.5,
    "fit_full_window_pitch": 5.0,
    "fit_knot_step_pitch": .35,
    "robust_iterations": 3,
    "huber_floor_pitch": .05,
    "jump_residual_pitch": .65,
    "body_threshold": .65,
    "body_bad_run_pitch": .18,
    "body_bad_fraction": .12,
    "split_padding_pitch": .10,
    "maximum_refit_depth": 2,
    "broad_support_threshold": .10,
    "residual_q90_limit_pitch": .35,
    "minimum_curve_separation_pitch": .45,
    "shape_prior": "nondecreasing_image_y_derivative",
    "raster_line_width_px": 1,
}


def _runs(mask):
    """返回连续True的半开区间，便于量出穿过亮绳体的连续长度。"""
    z=np.r_[False,np.asarray(mask,bool),False].astype(np.int8)
    return list(zip(np.flatnonzero(np.diff(z)==1),np.flatnonzero(np.diff(z)==-1)))


def _sample(array, xy, x0, y0, outside=0):
    if len(xy)==0:return np.zeros(0,np.float32)
    h,w=array.shape
    if not h or not w:return np.full(len(xy),outside,np.float32)
    xx=np.asarray(xy[:,0]-x0,np.float32)[None,:]
    yy=np.asarray(xy[:,1]-y0,np.float32)[None,:]
    return cv2.remap(array.astype(np.float32,copy=False),xx,yy,cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT,borderValue=float(outside))[0]


def _line(a,b):
    n=max(2,int(math.ceil(np.linalg.norm(b-a)))+1)
    return np.linspace(a,b,n,dtype=np.float32)


def _arc_run_length(xy,a,z):
    if z<=a:return 0.
    if len(xy)<2:return float(z-a)
    ds=np.linalg.norm(np.diff(xy,axis=0),axis=1)
    # 每次沿x采样1px，但斜线实际走得更长；门槛使用路径像素长度。
    return float(np.sum(ds[a:max(a,z-1)])+ds[min(a,len(ds)-1)])


def _isotonic_increasing(values,weights):
    """相邻斜率若反向变小，就按覆盖宽度合并：加权PAVA，无外部依赖。"""
    blocks=[]
    for i,(v,w) in enumerate(zip(values,weights)):
        blocks.append([i,i+1,float(w),float(v*w)])
        while len(blocks)>1 and blocks[-2][3]/blocks[-2][2]>blocks[-1][3]/blocks[-1][2]:
            right=blocks.pop();left=blocks.pop()
            blocks.append([left[0],right[1],left[2]+right[2],left[3]+right[3]])
    out=np.empty(len(values),np.float64)
    for a,b,w,total in blocks:out[a:b]=total/w
    return out


def _weighted_median(values,weights):
    order=np.argsort(values);v=np.asarray(values)[order];w=np.asarray(weights)[order]
    return float(v[min(len(v)-1,int(np.searchsorted(np.cumsum(w),.5*float(w.sum()))))])


def _convex_integral(knots,ky,xs):
    """允许曲率随位置变化，但图像纵坐标斜率只能逐渐增大。

    先平顺小窗的平均斜率，再连续线性插值导数并积分。得到C1连续的
    分段二次曲线，不强迫所有区间共用一条抛物线或固定圆/椭圆半径。
    本开发集的卷面弧顶在图像上方，左右侧均有该凸性；不是通用物理定律。
    """
    dx=np.diff(knots);raw=np.diff(ky)/dx;sec=_isotonic_increasing(raw,dx)
    mids=(knots[:-1]+knots[1:])/2
    if len(sec)>1:
        left_rate=max(0.,float((sec[1]-sec[0])/(mids[1]-mids[0])))
        right_rate=max(0.,float((sec[-1]-sec[-2])/(mids[-1]-mids[-2])))
    else:left_rate=right_rate=0.
    edges=np.r_[knots[0],mids,knots[-1]]
    derivative=np.r_[sec[0]-left_rate*(mids[0]-knots[0]),sec,
                     sec[-1]+right_rate*(knots[-1]-mids[-1])]
    lengths=np.diff(edges)
    cumulative=np.r_[0.,np.cumsum(.5*(derivative[:-1]+derivative[1:])*lengths)]
    jj=np.clip(np.searchsorted(edges,xs,side='right')-1,0,len(edges)-2)
    t=xs-edges[jj]
    values=cumulative[jj]+derivative[jj]*t+.5*(derivative[jj+1]-derivative[jj])/lengths[jj]*t*t
    return values,{'convexity_prior_applied':True,
        'secant_adjustment_rms':float(np.sqrt(np.mean((sec-raw)**2))),
        'derivative_decrease_after_projection':float(min(0.,np.min(np.diff(derivative))))}


def _raster_invalid_x(xy,roi,unknown,x0,y0):
    """按实际1px LINE_8画法逐像素查ROI/unknown，不能只查浮点中心采样。"""
    if len(xy)==0:return np.empty(0,np.int32)
    rr=np.rint(xy).astype(np.int32);h,w=roi.shape
    outside=(rr[:,0]<0)|(rr[:,0]>=w)|(rr[:,1]<0)|(rr[:,1]>=h)
    bad=list(rr[outside,0])
    bx=max(0,int(rr[:,0].min())-1);ex=min(w,int(rr[:,0].max())+2)
    by=max(0,int(rr[:,1].min())-1);ey=min(h,int(rr[:,1].max())+2)
    if ex<=bx or ey<=by:return np.unique(bad).astype(np.int32)
    painted=np.zeros((ey-by,ex-bx),np.uint8)
    cv2.polylines(painted,[rr-np.asarray([bx,by])],False,255,
                  PARAMETERS['raster_line_width_px'],cv2.LINE_8)
    invalid=roi[by:ey,bx:ex]==0
    uh,uw=unknown.shape
    ax=max(bx,x0);az=min(ex,x0+uw);ay=max(by,y0);ae=min(ey,y0+uh)
    if az>ax and ae>ay:
        invalid[ay-by:ae-by,ax-bx:az-bx]|=unknown[ay-y0:ae-y0,ax-x0:az-x0]>0
    py,px=np.nonzero((painted>0)&invalid)
    if len(px):bad.extend((px+bx).tolist())
    return np.unique(bad).astype(np.int32)


def _smooth_xy(points,pitch):
    """每个结点只用约5倍绳距内的点拟合二次小窗，再施加卷面弧凸性。

    不要求所有点落在一条抛物线上；不同小窗可有不同曲率。
    Huber权重仅降低离群点的牵引，原黄点坐标和成员记录不改变。
    """
    pts=np.asarray([[p['x'],p['y']] for p in points],np.float64)
    x,y=pts.T
    base=np.asarray([max(.05,min(1.,float(p.get('score',1.)))) for p in points])
    half=PARAMETERS['fit_full_window_pitch']*pitch/2
    step=max(2.,PARAMETERS['fit_knot_step_pitch']*pitch)
    knots=np.linspace(x[0],x[-1],max(2,int(math.ceil((x[-1]-x[0])/step))+1))
    ky=[]
    for at in knots:
        dx=(x-at)/half
        take=np.abs(dx)<1.0
        if take.sum()<4:
            take=np.zeros(len(x),bool);take[np.argsort(np.abs(dx))[:min(8,len(x))]]=True
        u=(x[take]-at)/pitch
        A=np.c_[np.ones(len(u)),u,u*u]
        # 三次核在小窗边缘趋于0，避免入窗/出窗造成曲线折角。
        spatial=np.maximum(0.,1-np.minimum(np.abs(dx[take]),.98)**3)**3
        ww=base[take]*spatial
        coef=np.zeros(3)
        for _ in range(PARAMETERS['robust_iterations']):
            root=np.sqrt(np.maximum(ww,1e-6))
            coef=np.linalg.lstsq(A*root[:,None],y[take]*root,rcond=None)[0]
            residual=y[take]-A@coef
            scale=max(PARAMETERS['huber_floor_pitch']*pitch,
                      1.4826*float(np.median(np.abs(residual-np.median(residual)))))
            ww=base[take]*spatial*np.minimum(1.,1.5*scale/np.maximum(np.abs(residual),1e-6))
        ky.append(float(coef[0]))
    ky=np.asarray(ky)
    xs=np.arange(math.ceil(x[0]),math.floor(x[-1])+1,dtype=np.float64)
    if not len(xs):return np.empty((0,2),np.float32),{}
    ys,convex_stats=_convex_integral(knots,ky,xs)
    # 纵向位置仍由本组全部黄点确定；不能拿别组或旧C3的高度代替。
    vertical_shift=_weighted_median(y-np.interp(x,xs,ys),base)
    ys+=vertical_shift
    residual=np.abs(y-np.interp(x,xs,ys))
    return np.c_[xs,ys].astype(np.float32),{
        'residual_median_px':float(np.median(residual)),
        'residual_q90_px':float(np.percentile(residual,90)),
        'window_px':float(2*half),'knot_count':len(knots),
        'vertical_shift_px':vertical_shift,**convex_stats}


def fit_validated_curves(tracks,roi,pitch,shadow,guard):
    """输出原图坐标曲线、原图尺寸0/255掩码及可追溯拒绝原因。

    point_states='observed'表示拟合处有本轮清洁阴影响应支持，
    'inferred'表示短缺口补线；两者都是拟合结果，不是逐像素真值。
    强亮体/未知区不允许补线。超过限制的穿绳组不会带颜色混入结果。
    """
    if not isinstance(roi,np.ndarray) or roi.ndim!=2 or roi.dtype!=np.uint8:
        raise ValueError('roi must be original-size uint8 mask')
    pitch=float(pitch)
    if not math.isfinite(pitch) or not 8<=pitch<=70:raise ValueError('pitch must be8..70 pixels')
    shape=roi.shape;x0=int(shadow.get('x0',0));y0=int(shadow.get('y0',0))
    broad=np.asarray(shadow['broad_score'],np.float32)
    cleaned=np.asarray(shadow['cleaned_shadow'],np.uint8)
    body=np.asarray(guard.get('body_barrier',shadow['body_score']),np.float32)
    unknown=np.asarray(guard.get('unknown_mask',guard.get('excluded_mask',np.zeros_like(cleaned))),np.uint8)
    if any(v.shape!=broad.shape for v in (cleaned,body,unknown)):
        raise ValueError('shadow and guard fields must share the same crop')
    if ('x0' in guard and int(guard['x0'])!=x0) or ('y0' in guard and int(guard['y0'])!=y0):
        raise ValueError('guard crop origin differs from shadow')
    # 每张图只转换一次；逐点采样时不能反复复制整张原图/裁块。
    roi_float=roi.astype(np.float32)
    unknown_float=unknown.astype(np.float32)
    cleaned_float=cleaned.astype(np.float32)
    smooth=np.zeros(shape,np.uint8);inferred=np.zeros(shape,np.uint8)
    curves=[];rejections=[];counts={'input_tracks':len(tracks),'input_points':0,
        'pre_split_groups':0,'refit_attempts':0,'accepted_fragments':0}
    limit_gap=PARAMETERS['maximum_gap_pitch']*pitch
    body_t=PARAMETERS['body_threshold']

    def record(tid,indices,reason,**kwargs):
        item={'track_id':str(tid),'original_group_ids':[str(tid)],
              'source_point_indices':[int(i) for i in indices],'reason':reason}
        item.update(kwargs);rejections.append(item)

    def fields(xy):
        return (_sample(body,xy,x0,y0),_sample(unknown_float,xy,x0,y0)>0,
                _sample(roi_float,xy,0,0)>127)

    def split_and_refit(tid,points,indices,bad_x,depth,reason):
        # 坏带两侧重新归为独立子组，不能把它们再用同一条曲线连回去。
        counts['refit_attempts']+=1
        bad_x=np.asarray(bad_x)
        intervals=[]
        if len(bad_x):
            cuts=np.flatnonzero(np.diff(bad_x)>1.5)+1
            for zz in np.split(bad_x,cuts):
                intervals.append((float(zz[0]-.10*pitch),float(zz[-1]+.10*pitch)))
        groups=[];current=[];current_ids=[];previous_region=None
        rejected_ids=[]
        for p,idx in zip(points,indices):
            px=float(p['x']);inside=any(a<=px<=b for a,b in intervals)
            region=sum(px>b for _,b in intervals)
            if inside:
                if current:groups.append((current,current_ids));current=[];current_ids=[]
                rejected_ids.append(idx);previous_region=None;continue
            if current and region!=previous_region:
                groups.append((current,current_ids));current=[];current_ids=[]
            current.append(p);current_ids.append(idx);previous_region=region
        if current:groups.append((current,current_ids))
        if rejected_ids:record(tid,rejected_ids,'points_inside_rejected_band',parent_reason=reason,depth=depth)
        for part,ids in groups:
            if len(part)==len(points):
                record(tid,ids,'refit_cannot_split_same_group',parent_reason=reason,depth=depth)
            else:fit_part(tid,part,ids,depth+1)

    def fit_part(tid,points,indices,depth):
        if len(points)<PARAMETERS['minimum_points'] or float(points[-1]['x'])-float(points[0]['x'])<PARAMETERS['minimum_span_pitch']*pitch:
            record(tid,indices,'insufficient_points_or_span',depth=depth,point_count=len(points));return
        xy,stats=_smooth_xy(points,pitch)
        b,u,inside=fields(xy)
        hit=b>body_t
        longest=max((_arc_run_length(xy,i,j) for i,j in _runs(hit)),default=0.)
        fraction=float(hit.mean()) if len(hit) else 0.
        unsupported=(_sample(broad,xy,x0,y0)<PARAMETERS['broad_support_threshold'])|(_sample(cleaned_float,xy,x0,y0)<127)
        long_missing=np.zeros(len(xy),bool)
        for a,z in _runs(unsupported):
            if _arc_run_length(xy,a,z)>limit_gap:long_missing[a:z]=True
        # 补线即使只碰到1个强亮体像素也不可保留；有支持处按整组门槛判断。
        forbidden_infer=unsupported&hit
        body_bad=longest>=max(1,math.ceil(PARAMETERS['body_bad_run_pitch']*pitch)) or fraction>PARAMETERS['body_bad_fraction']
        bad=u|(~inside)|long_missing|forbidden_infer
        reasons=[]
        if body_bad:bad|=hit;reasons.append('crosses_bright_rope_body')
        if np.any(u):reasons.append('unknown_occlusion')
        if np.any(~inside):reasons.append('outside_roi')
        if np.any(long_missing):reasons.append('long_image_evidence_gap')
        if np.any(forbidden_infer):reasons.append('inference_crosses_body')
        raster_bad_x=_raster_invalid_x(xy,roi,unknown,x0,y0)
        if len(raster_bad_x):
            reasons.append('raster_pixels_outside_roi_or_in_unknown')
            bad|=np.isin(np.rint(xy[:,0]).astype(np.int32),raster_bad_x)
        if stats['residual_q90_px']>PARAMETERS['residual_q90_limit_pitch']*pitch:
            reasons.append('yellow_group_inconsistent_with_smooth_path')
            px=np.asarray([p['x'] for p in points]);py=np.asarray([p['y'] for p in points])
            bad_obs=np.abs(py-np.interp(px,xy[:,0],xy[:,1]))>PARAMETERS['residual_q90_limit_pitch']*pitch
            for a in px[bad_obs]:bad|=np.abs(xy[:,0]-a)<.2*pitch
        if reasons:
            record(tid,indices,'+'.join(reasons),depth=depth,body_fraction=fraction,
                   body_longest_run_px=float(longest),**stats)
            if depth<PARAMETERS['maximum_refit_depth'] and np.any(bad):
                split_and_refit(tid,points,indices,xy[bad,0],depth,'+'.join(reasons))
            return
        states=np.where(unsupported,'inferred','observed').tolist()
        curve_id=f'{tid}__s{len(curves):05d}'
        curves.append({'curve_id':curve_id,'track_id':str(tid),'original_group_ids':[str(tid)],
            'source_point_indices':[int(i) for i in indices],
            'sampled_points_xy':xy.tolist(),'point_states':states,
            'fit_statistics':stats,'refit_depth':depth,
            'validation_support_score':float(np.mean([float(p.get('score',1.)) for p in points])*(.5+.5*float(np.mean(~unsupported))))})
        rr=np.rint(xy).astype(int)
        good=(rr[:,0]>=0)&(rr[:,0]<shape[1])&(rr[:,1]>=0)&(rr[:,1]<shape[0])
        rr=rr[good];ss=np.asarray(unsupported)[good]
        # 1px中心线：把相邻采样点连接，最后再次用ROI裁切。
        for j in range(len(rr)-1):
            target=inferred if ss[j] or ss[j+1] else smooth
            cv2.line(target,tuple(rr[j]),tuple(rr[j+1]),255,1,cv2.LINE_8)
        counts['accepted_fragments']+=1

    for n,track in enumerate(tracks):
        tid=str(track.get('track_id',n));raw=track.get('points',[]);counts['input_points']+=len(raw)
        indexed=[]
        for i,p in enumerate(raw):
            if not all(math.isfinite(float(p.get(k,float('nan')))) for k in ('x','y')):
                record(tid,[i],'non_finite_observation');continue
            indexed.append((i,p))
        indexed.sort(key=lambda pair:(float(pair[1]['x']),-float(pair[1].get('score',1.))))
        unique=[]
        for idx,p in indexed:
            if unique and float(p['x'])==float(unique[-1][1]['x']):
                record(tid,[idx],'duplicate_x_requires_separate_track');continue
            unique.append((idx,p))
        groups=[];part=[];ids=[]
        for idx,p in unique:
            why=[]
            if part:
                prev=part[-1];dx=float(p['x'])-float(prev['x']);dy=float(p['y'])-float(prev['y'])
                if dx>limit_gap:why.append('long_yellow_gap')
                slopes=[float(v.get('slope',0.)) for v in (prev,p)]
                # 两端方向都解释不了的跨行跳跃，不能仅靠平滑拟合掩盖。
                if min(abs(dy-s*dx) for s in slopes)>PARAMETERS['jump_residual_pitch']*pitch:
                    why.append('incompatible_point_jump')
                seg=_line(np.asarray([prev['x'],prev['y']],np.float32),np.asarray([p['x'],p['y']],np.float32))
                bb,uu,valid=fields(seg)
                if np.any(uu):why.append('unknown_between_points')
                if np.any(~valid):why.append('roi_gap_between_points')
                bh=bb>body_t
                if (max((z-a for a,z in _runs(bh)),default=0)>=max(1,math.ceil(PARAMETERS['body_bad_run_pitch']*pitch))
                    or (len(bh) and float(bh.mean())>PARAMETERS['body_bad_fraction'])):
                    why.append('body_between_points')
            if why:
                record(tid,[ids[-1],idx],'pre_split:'+'+'.join(why))
                groups.append((part,ids));part=[];ids=[]
            part.append(p);ids.append(idx)
        if part:groups.append((part,ids))
        counts['pre_split_groups']+=len(groups)
        for part,ids in groups:fit_part(tid,part,ids,0)
    # 不借助旧ID排序：不同新组若在同x靠得过近或交叉，只保留支持较强段。
    # 不能平移弱段来凑绳距；整段拒绝与原黄点索引都写入记录。
    kept=[]
    for curve in sorted(curves,key=lambda c:c['validation_support_score'],reverse=True):
        xy=np.asarray(curve['sampled_points_xy'],np.float32);conflict=None
        for prior in kept:
            q=np.asarray(prior['sampled_points_xy'],np.float32)
            lo=max(xy[0,0],q[0,0]);hi=min(xy[-1,0],q[-1,0])
            if hi<lo:continue
            xs=np.arange(math.ceil(float(lo)),math.floor(float(hi))+1,dtype=np.float32)
            if not len(xs):continue
            delta=np.interp(xs,xy[:,0],xy[:,1])-np.interp(xs,q[:,0],q[:,1])
            nearest=float(np.min(np.abs(delta)))
            crossing=bool(np.min(delta)<0<np.max(delta))
            if crossing or nearest<PARAMETERS['minimum_curve_separation_pitch']*pitch:
                conflict=(prior['curve_id'],nearest,crossing);break
        if conflict is None:kept.append(curve)
        else:
            record(curve['track_id'],curve['source_point_indices'],'curve_conflict_weaker_segment_rejected',
                   curve_id=curve['curve_id'],stronger_curve_id=conflict[0],
                   minimum_distance_px=conflict[1],crossing=conflict[2],
                   validation_support_score=curve['validation_support_score'])
    curves=kept;smooth.fill(0);inferred.fill(0)
    for curve in curves:
        rr=np.rint(np.asarray(curve['sampled_points_xy'])).astype(int)
        ss=np.asarray(curve['point_states'])=='inferred'
        for j in range(len(rr)-1):
            cv2.line(smooth,tuple(rr[j]),tuple(rr[j+1]),255,1,cv2.LINE_8)
            if ss[j] or ss[j+1]:cv2.line(inferred,tuple(rr[j]),tuple(rr[j+1]),255,1,cv2.LINE_8)
    counts['accepted_fragments_before_collision']=counts['accepted_fragments']
    counts['accepted_fragments']=len(curves)
    smooth[roi==0]=0;inferred[roi==0]=0
    # smooth_mask包含所有已接受曲线；inferred_mask只是其中短缺口子集。
    smooth|=inferred
    return {'curves':curves,'smooth_mask':smooth,'inferred_mask':inferred,
            'rejections':rejections,'diagnostics':{**counts,'rejection_records':len(rejections),
                'parameters':PARAMETERS.copy(),
                'state_meaning':{'observed':'fitted path with current image support',
                                 'inferred':'short image-evidence gap, not a measured boundary'},
                'limitations':'image-supported geometry is not physical rope identity or contact truth'}}
