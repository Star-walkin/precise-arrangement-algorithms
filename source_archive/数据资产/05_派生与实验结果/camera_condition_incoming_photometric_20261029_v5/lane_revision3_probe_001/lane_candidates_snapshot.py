"""F3 正式边界约束的有限绳体轨道候选（纯几何，不选择活动绳身份）。

这个文件不读取磁盘、不导入设备，也不读取原图灰度。curve_refs 只是寻找
正式 merged 像素的提案，不能把尚未融合的 D4 曲线当成已接受边界。
每个保留点均记录双边到正式掩膜的距离、来源及是否经过插值/外推。
同一条候选先沿自由绳直线，再用 C1 连续过渡接到两条相邻边界之间，
最后沿该轨道延伸。候选排序只用于有限预算，不是图像评分或同根结论。
"""
from __future__ import annotations
import math
from collections import Counter
import cv2
import numpy as np

PARAMETERS = {
    'maximum_candidates': 12,
    'maximum_reference_fragments': 256,
    'sample_step_px': 4.,
    'maximum_snap_pitch': .25,
    'maximum_internal_gap_pitch': 1.5,
    'maximum_endpoint_extension_pitch': 2.,
    'minimum_boundary_span_pitch': 2.,
    'minimum_lane_span_pitch': 3.,
    'minimum_normal_separation_pitch': .30,
    'maximum_normal_separation_pitch': 1.80,
    'free_width_mismatch_cost_weight': .15,
    'maximum_free_width_mismatch_cost': 1.5,
    'free_width_role': 'soft_cost_not_lane_existence_gate',
    'maximum_intermediate_formal_fraction': .25,
    'transition_lengths_width': [4., 8.],
    'maximum_join_offset_width': 2.5,
    'maximum_join_angle_deg': 35.,
    'maximum_transition_turn_per_width': .50,
    'tangent_fit_half_window_pitch': 2.,
    'geometry_pixel_tolerance_min_px': 1.25,
    'geometry_pixel_tolerance_width_fraction': .075,
    'reverse_angle_pixel_conversion': 6.,
}

def _runs(flag):
    z=np.r_[False,np.asarray(flag,bool),False].astype(np.int8)
    return list(zip(np.flatnonzero(np.diff(z)==1),np.flatnonzero(np.diff(z)==-1)))

def _sample(a,xy):
    p=np.asarray(xy,np.float32).reshape(-1,2)
    if not len(p):return np.zeros(0,np.float32)
    return cv2.remap(a,p[:,0][None,:],p[:,1][None,:],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=1e5).ravel()

def _clean_curve(xy):
    q=np.asarray(xy,float)
    if q.ndim!=2 or q.shape[1]!=2 or len(q)<3 or not np.isfinite(q).all():return None
    q=q[np.argsort(q[:,0],kind='stable')]
    q=q[np.r_[True,np.diff(q[:,0])>.1]]
    return q if len(q)>=3 else None

def _proposals(data):
    out=[]
    for i,ref in enumerate(data.get('curve_refs',[])):
        q=_clean_curve(ref.get('sampled_points_xy',[]))
        if q is not None:out.append((str(ref.get('curve_id','ref_'+str(i))),str(ref.get('source','reference')),q))
    # F3 的新增接受片段也是提案，仍逐点经过正式 merged 门控。
    for g in data.get('f3_record',{}).get('groups',[]):
        if g.get('status')!='accepted':continue
        q=_clean_curve(g.get('sampled_points_xy',[]))
        if q is not None:out.append(('F3_group:'+str(g.get('group_id')),'F3_accepted_group',q))
    return out

def _fill_small_gaps(flag,maximum_count):
    out=flag.copy()
    for a,b in _runs(~flag):
        if a>0 and b<len(flag) and b-a<=maximum_count:out[a:b]=True
    return out

def _column_peaks(mask,x,cache):
    x=int(x)
    if x not in cache:
        ys=np.flatnonzero(mask[:,x])
        if len(ys):
            edges=np.r_[0,np.flatnonzero(np.diff(ys)>1)+1,len(ys)]
            # 选真实存在的中心像素，而不是两条不同线之间的平均位置。
            cache[x]=np.array([ys[(a+b-1)//2] for a,b in zip(edges[:-1],edges[1:])],float)
        else:cache[x]=np.zeros(0,float)
    return cache[x]

def _snap_refs(data,origin,d,width):
    mask=data['merged']>0;h,w=mask.shape;pitch=float(data['pitch']);step=PARAMETERS['sample_step_px']
    distance=cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    normal=np.array([-d[1],d[0]]);cache={};tracks=[];counts=Counter();covered=np.zeros(mask.shape,np.uint8)
    for ref_id,source,q in _proposals(data):
        counts['proposals']+=1
        xs=np.arange(max(0,math.ceil(q[0,0]/step)*step),min(w-1,q[-1,0])+1,step)
        if len(xs)<3:continue
        guess=np.interp(xs,q[:,0],q[:,1]);slopes=np.gradient(guess,xs)
        snapped=np.full(len(xs),np.nan);snap_distance=np.full(len(xs),np.nan)
        for i,(x,y,m) in enumerate(zip(xs,guess,slopes)):
            peaks=_column_peaks(mask,x,cache)
            if not len(peaks):continue
            distances=abs(peaks-y);order=np.argsort(distances);j=order[0]
            normal_error=float(distances[j]/math.sqrt(1+m*m))
            if normal_error>PARAMETERS['maximum_snap_pitch']*pitch:continue
            # 提案几乎位于两条正式线之间时不决定归属，避免切换到邻行。
            if len(order)>1 and distances[order[1]]-distances[j]<max(1.,.08*pitch):continue
            snapped[i]=peaks[j];snap_distance[i]=normal_error
        valid=np.isfinite(snapped);counts['sampled_proposal_points']+=len(xs);counts['formally_snapped_points']+=int(valid.sum())
        connected=_fill_small_gaps(valid,int(PARAMETERS['maximum_internal_gap_pitch']*pitch/step))
        for a,b in _runs(connected):
            if xs[b-1]-xs[a]<PARAMETERS['minimum_boundary_span_pitch']*pitch or np.count_nonzero(valid[a:b])<4:continue
            xx=xs[a:b];good=valid[a:b];yy=np.interp(xx,xx[good],snapped[a:b][good])
            # 正常情况下吸附残差连续；突然跳到另一根绳的片段必须拆开。
            residual=yy-guess[a:b]
            split=np.r_[0,np.flatnonzero(abs(np.diff(residual))>.35*pitch)+1,len(xx)]
            for lo,hi in zip(split[:-1],split[1:]):
                if hi-lo<4 or xx[hi-1]-xx[lo]<PARAMETERS['minimum_boundary_span_pitch']*pitch:continue
                x=xx[lo:hi];y=yy[lo:hi];observed=good[lo:hi];p=np.c_[x,y]
                # 统计覆盖只计正式支持的连续采样段，不计填补缺口。
                for k0,k1 in _runs(observed):
                    if k1-k0>=2:cv2.polylines(covered,[np.rint(p[k0:k1]).astype(np.int32)],False,255,1)
                forward=(p-origin)@d;off=abs((p-origin)@normal)
                if not np.any((forward>-4*pitch)&(off<max(6*pitch,6*width))):
                    counts['far_from_free_strip_fragments']+=1;continue
                tracks.append({'id':ref_id+':'+str(a+lo),'refs':[ref_id],'source':source,'x':x,'y':y,
                  'inferred':~observed,'snap':np.where(observed,abs(y-guess[a:b][lo:hi])/np.sqrt(1+slopes[a:b][lo:hi]**2),0.),
                  'source_point':[ref_id if v else ref_id+':short_gap' for v in observed]})
    near=cv2.dilate(covered,np.ones((3,3),np.uint8))>0
    counts['formal_mask_pixels']=int(mask.sum());counts['formal_mask_pixels_covered_by_reference_proposals']=int((mask&near).sum())
    counts['formal_mask_pixels_uncovered_by_reference_proposals']=int((mask&~near).sum())
    counts['retained_fragments_before_budget']=len(tracks)
    # 有限预算按与自由直线接近程度排序；超过预算显式记录，不冒充全集。
    tracks.sort(key=lambda t:float(np.min(abs((np.c_[t['x'],t['y']]-origin)@normal))))
    counts['reference_fragments_budget_omitted']=max(0,len(tracks)-PARAMETERS['maximum_reference_fragments'])
    return tracks[:PARAMETERS['maximum_reference_fragments']],distance,cache,dict(counts)

def _same_row(a,b,pitch):
    lo=max(a['x'][0],b['x'][0]);hi=min(a['x'][-1],b['x'][-1])
    if hi-lo>=pitch:
        x=np.arange(lo,hi+1,PARAMETERS['sample_step_px']);ya=np.interp(x,a['x'],a['y']);yb=np.interp(x,b['x'],b['y'])
        m=.5*(np.gradient(ya,x)+np.gradient(yb,x));err=abs(ya-yb)/np.sqrt(1+m*m)
        return bool(np.median(err)<.20*pitch and np.percentile(err,90)<.30*pitch and np.median(abs(np.arctan(np.gradient(ya,x))-np.arctan(np.gradient(yb,x))))<.22)
    first,second=(a,b) if a['x'][0]<b['x'][0] else (b,a)
    gap=second['x'][0]-first['x'][-1]
    if not 0<gap<=PARAMETERS['maximum_internal_gap_pitch']*pitch:return False
    n=min(6,len(first['x']),len(second['x']))
    m1=float(np.polyfit(first['x'][-n:]-first['x'][-1],first['y'][-n:],1)[0]);m2=float(np.polyfit(second['x'][:n]-second['x'][0],second['y'][:n],1)[0])
    error=abs(first['y'][-1]+gap*(m1+m2)/2-second['y'][0])/math.sqrt(1+((m1+m2)/2)**2)
    return error<.22*pitch and abs(math.atan(m1)-math.atan(m2))<.22

def _merge_rows(tracks,pitch):
    # Complete-link 合并：新片段必须与组中有重叠的所有片段一致，避免
    # A接B、B接C却让明显不同的A/C通过传递关系混成一条行。
    groups=[]
    for t in sorted(tracks,key=lambda r:-(r['x'][-1]-r['x'][0])):
        best=None
        for group in groups:
            touching=[q for q in group if min(t['x'][-1],q['x'][-1])-max(t['x'][0],q['x'][0])>=-PARAMETERS['maximum_internal_gap_pitch']*pitch]
            if touching and all(_same_row(t,q,pitch) for q in touching):best=group;break
        if best is None:groups.append([t])
        else:best.append(t)
    rows=[];step=PARAMETERS['sample_step_px']
    for gi,g in enumerate(groups):
        values={}
        for t in g:
            for i,x in enumerate(t['x']):values.setdefault(int(x),[]).append((t['y'][i],bool(t['inferred'][i]),float(t['snap'][i]),t['source_point'][i]))
        xs=np.arange(min(values),max(values)+step,step);ys=np.full(len(xs),np.nan);infer=np.ones(len(xs),bool);snap=np.zeros(len(xs));src=['inter_fragment_short_gap']*len(xs)
        for i,x in enumerate(xs):
            v=values.get(int(x))
            if not v:continue
            v.sort(key=lambda z:(z[1],z[2]));chosen=v[0]
            ys[i],infer[i],snap[i],src[i]=chosen
        good=np.isfinite(ys);keep=_fill_small_gaps(good,int(PARAMETERS['maximum_internal_gap_pitch']*pitch/step))
        for a,b in _runs(keep):
            if xs[b-1]-xs[a]<PARAMETERS['minimum_boundary_span_pitch']*pitch:continue
            y=np.interp(xs[a:b],xs[good],ys[good])
            rows.append({'id':'formal_row_'+str(len(rows)).zfill(3),'x':xs[a:b],'y':y,'inferred':infer[a:b],
                         'snap':snap[a:b],'source_point':src[a:b],'refs':sorted(set(q for t in g for q in t['refs']))})
    return rows

def _lane_pairs(rows,data,cache,width):
    pitch=float(data['pitch']);mask=data['merged']>0;step=PARAMETERS['sample_step_px'];lanes=[];counts=Counter()
    for ai,a in enumerate(rows):
        for b in rows[ai+1:]:
            lo=max(a['x'][0],b['x'][0]);hi=min(a['x'][-1],b['x'][-1])
            if hi-lo<PARAMETERS['minimum_lane_span_pitch']*pitch:continue
            xs=np.arange(lo,hi+1,step);ya=np.interp(xs,a['x'],a['y']);yb=np.interp(xs,b['x'],b['y'])
            sign=np.sign(np.median(yb-ya))
            if sign==0 or np.any((yb-ya)*sign<=1.):counts['crossing_or_duplicate_pairs']+=1;continue
            lower,upper=(a,b) if sign>0 else (b,a);yl,yu=(ya,yb) if sign>0 else (yb,ya)
            yy=(yl+yu)/2;m=np.gradient(yy,xs);ww=(yu-yl)/np.sqrt(1+m*m)
            # 自由段的光学亮条宽度可能只覆盖一部分绳体。它不能推翻由
            # 正式F3双边和绳距已经支持的相邻轨道。比例只参与有限软代价。
            valid=(ww>=PARAMETERS['minimum_normal_separation_pitch']*pitch)&(ww<=PARAMETERS['maximum_normal_separation_pitch']*pitch)
            for start,end in _runs(valid):
                if xs[end-1]-xs[start]<PARAMETERS['minimum_lane_span_pitch']*pitch:continue
                xx=xs[start:end];y0=yl[start:end];y1=yu[start:end]
                checks=np.unique(np.linspace(0,len(xx)-1,min(30,len(xx))).astype(int));between=[]
                margin=max(2.,.15*pitch)
                for k in checks:
                    peaks=_column_peaks(mask,xx[k],cache);between.append(bool(np.any((peaks>y0[k]+margin)&(peaks<y1[k]-margin))))
                fraction=float(np.mean(between)) if between else 1.
                if fraction>PARAMETERS['maximum_intermediate_formal_fraction']:counts['non_adjacent_pairs']+=1;continue
                ia=np.rint((xx-lower['x'][0])/step).astype(int);ib=np.rint((xx-upper['x'][0])/step).astype(int)
                lane={'lane_id':'lane_'+str(len(lanes)).zfill(3),'x':xx,'y':yy[start:end],'widths':ww[start:end],
                      'boundary_xy':np.stack([np.c_[xx,y0],np.c_[xx,y1]],axis=1),
                      'inferred':lower['inferred'][ia]|upper['inferred'][ib],
                      'source_boundary_ids':[lower['id'],upper['id']],
                      'source_ref_ids':[lower['refs'],upper['refs']],
                      'point_boundary_sources':[[lower['source_point'][i],upper['source_point'][j]] for i,j in zip(ia,ib)],
                      'snap_distance_px':np.c_[lower['snap'][ia],upper['snap'][ib]],
                      'intermediate_formal_fraction':fraction}
                lanes.append(lane)
    return lanes,dict(counts)

def _local_lane_geometry(lane,pitch):
    """Fit both formal rails in a local window; 4 px raster steps are not tangents."""
    x=lane['x'];rails=lane['boundary_xy'][:,:,1];out=[]
    for xi in x:
        use=abs(x-xi)<=PARAMETERS['tangent_fit_half_window_pitch']*pitch
        if np.count_nonzero(use)<5:
            use=np.zeros(len(x),bool);use[np.argsort(abs(x-xi))[:min(5,len(x))]]=True
        u=(x[use]-xi)/pitch;design=np.c_[u*u,u,np.ones(len(u))]
        coef=np.linalg.lstsq(design,rails[use],rcond=None)[0]
        residual=float(np.sqrt(np.mean((design@coef-rails[use])**2)))
        span=max(float(np.ptp(x[use])),pitch)
        out.append([float(np.mean(coef[1])/pitch),float(2*coef[0,0]/pitch**2),float(2*coef[0,1]/pitch**2),span,residual])
    return np.asarray(out)

def _curvature_context(lane,index,lanes,pitch,pixel_tolerance):
    x=float(lane['x'][index]);y=float(lane['y'][index]);fit=lane['_local_geometry'][index]
    # A small coordinate error over the fit span can change the sign of an
    # almost-straight rail. Such a sign is unknown, never a forced rejection.
    tolerance=2*max(pixel_tolerance,fit[4])/fit[3]**2
    pair=fit[1:3];signs=np.where(abs(pair)>tolerance,np.sign(pair),0)
    if signs[0]*signs[1]<0:return None,'formal_rail_curvature_conflict'
    votes=[];nearby=[]
    for other in lanes:
        if not other['x'][0]<=x<=other['x'][-1]:continue
        if abs(np.interp(x,other['x'],other['y'])-y)>3*pitch:continue
        i=int(np.argmin(abs(other['x']-x)));f=other['_local_geometry'][i];value=float(np.mean(f[1:3]));eps=2*max(pixel_tolerance,f[4])/f[3]**2
        nearby.append(value)
        if abs(value)>eps:votes.append(int(np.sign(value)))
    trend=0
    if len(votes)>=2 and abs(sum(votes))/len(votes)>=.34:trend=int(np.sign(sum(votes)))
    elif signs[0]==signs[1] and signs[0]!=0:trend=int(signs[0])
    if trend and float(np.mean(pair))*trend < -tolerance:return None,'lane_curvature_opposes_neighbors'
    return {'formal_curvature_sign':trend,'formal_rail_second_derivatives':pair.tolist(),'formal_neighbor_second_derivatives':nearby,'formal_curvature_tolerance':float(tolerance)},None

def _join_options(lane,origin,d,width,pitch,lanes=None,rejections=None):
    rejections=Counter() if rejections is None else rejections
    lanes=[lane] if lanes is None else lanes
    if '_local_geometry' not in lane:lane['_local_geometry']=_local_lane_geometry(lane,pitch)
    x=lane['x'];y=lane['y'];m=lane['_local_geometry'][:,0];normal=np.array([-d[1],d[0]]);p=np.c_[x,y]
    forward=(p-origin)@d;off=(p-origin)@normal;direction=np.c_[np.full(len(x),np.sign(d[0])),np.sign(d[0])*m];direction/=np.linalg.norm(direction,axis=1)[:,None]
    angle=np.arccos(np.clip(direction@d,-1,1))
    remaining=(x[-1]-x) if d[0]>0 else (x-x[0])
    scale=np.maximum(width,lane['widths'])
    ratio=lane['widths']/width
    width_cost=PARAMETERS['free_width_mismatch_cost_weight']*np.minimum(PARAMETERS['maximum_free_width_mismatch_cost'],abs(np.log(np.maximum(ratio,1e-6))))
    valid=(forward>=0)&(abs(off)<=PARAMETERS['maximum_join_offset_width']*scale)&(angle<=math.radians(PARAMETERS['maximum_join_angle_deg']))&(remaining>=2*scale*abs(d[0]))
    ids=np.flatnonzero(valid)
    if not len(ids):return []
    geometric=abs(off)/scale+angle/.35+width_cost
    # First test all existing valid locations. Rejecting an early S-curve must
    # leave the later, genuinely compatible tangent meeting eligible.
    options=[]
    for i in ids:
        pixel_tolerance=max(PARAMETERS['geometry_pixel_tolerance_min_px'],PARAMETERS['geometry_pixel_tolerance_width_fraction']*scale[i])
        context,reason=_curvature_context(lane,i,lanes,pitch,pixel_tolerance)
        if reason:rejections[reason]+=1;continue
        local_options=[]
        for length_factor in PARAMETERS['transition_lengths_width']:
            s0=forward[i]-length_factor*scale[i]
            if s0 < -2*scale[i]:continue
            start=origin+s0*d;x0=float(start[0]);x1=float(x[i]);dx=x1-x0
            if dx*d[0]<=scale[i]*abs(d[0]):continue
            tx=np.linspace(x0,x1,max(4,int(abs(dx)/PARAMETERS['sample_step_px'])+1));t=(tx-x0)/dx
            m0=d[1]/d[0];m1=m[i]
            ty=(2*t**3-3*t**2+1)*start[1]+(t**3-2*t**2+t)*dx*m0+(-2*t**3+3*t**2)*y[i]+(t**3-t**2)*dx*m1
            derivative=(6*t*t-6*t)*start[1]/dx+(3*t*t-4*t+1)*m0+(-6*t*t+6*t)*y[i]/dx+(3*t*t-2*t)*m1
            tangent=np.arctan(derivative);turn=np.abs(np.diff(tangent));arc=np.hypot(np.diff(tx),np.diff(ty))
            rate=float(np.max(turn/np.maximum(arc,1e-6))*scale[i])
            if rate>PARAMETERS['maximum_transition_turn_per_width']:continue
            # 过渡段不能反向折返；C1 连续由 Hermite 两端导数直接保证。
            if np.any(abs(derivative)>4.):continue
            # Sort the tangent by increasing image x, so both cameras use the
            # same curvature sign. A cylinder lane may bend strongly, but a
            # join should not first bend against neighboring rails then return.
            theta=tangent if dx>0 else tangent[::-1]
            down=float(np.max(np.maximum.accumulate(theta)-theta));up=float(np.max(theta-np.minimum.accumulate(theta)))
            trend=context['formal_curvature_sign'];reverse=down if trend>0 else (up if trend<0 else min(down,up))
            allowed=math.atan(PARAMETERS['reverse_angle_pixel_conversion']*pixel_tolerance/max(float(arc.sum()),1.))
            if reverse>allowed:rejections['transition_reverse_curvature']+=1;continue
            local_options.append({'index':int(i),'transition':np.c_[tx,ty], 'transition_start_s':float(s0),
                            'geometry_cost':float(geometric[i]+.04*length_factor+.15*rate),
                            'join_offset_px':float(off[i]),'join_angle_deg':float(math.degrees(angle[i])),
                            'transition_turn_per_width':rate,'transition_length_width':float(length_factor),
                            'transition_scale_px':float(scale[i]),'lane_free_width_ratio':float(ratio[i]),
                            'join_forward_s':float(forward[i]),'join_tangent_estimator':'local_two_rail_quadratic_halfwindow_2pitch',
                            'transition_reverse_turn_deg':float(math.degrees(reverse)),'allowed_reverse_turn_deg':float(math.degrees(allowed)),
                            'geometry_pixel_tolerance_px':float(pixel_tolerance),**context})
        if local_options:options.append(min(local_options,key=lambda q:q['geometry_cost']))
    rejections['valid_locations_examined']+=len(ids);rejections['legal_join_locations']+=len(options)
    if not options:return []
    best=min(options,key=lambda q:q['geometry_cost']);best['join_option_role']='best_geometry'
    later=[v for v in options if v['join_forward_s']>=best['join_forward_s']+2*best['transition_scale_px'] and v['geometry_cost']<=best['geometry_cost']+1.]
    if later:
        late=max(later,key=lambda q:q['join_forward_s']);late['join_option_role']='later_distinct_join';return [best,late]
    return [best]

def _make_candidate(lane,opt,origin,d,width,pitch,outer,distance,inferred_mask,shape):
    h,w=shape;sign=int(np.sign(d[0]));join=int(opt['index']);indices=np.arange(join,len(lane['x'])) if sign>0 else np.arange(join,-1,-1)
    tail=np.c_[lane['x'][indices],lane['y'][indices]];tw=lane['widths'][indices];bxy=lane['boundary_xy'][indices];ti=lane['inferred'][indices].copy()
    sources=[lane['point_boundary_sources'][int(i)] for i in indices];snaps=lane['snap_distance_px'][indices]
    ti|=np.any(_sample((inferred_mask>0).astype(np.float32),bxy.reshape(-1,2)).reshape(-1,2)>.3,axis=1)
    states=['lane']*len(tail)
    # 只允许至多2p的端推断。它不叫观察到的侧面终点，且每点显式标出。
    end_extra=[];end_width=[];extra_b=[]
    if len(tail)>=6:
        n=min(len(tail),max(6,int(2*pitch/PARAMETERS['sample_step_px'])))
        x0=float(tail[-1,0]);coef=np.polyfit((tail[-n:,0]-x0)/pitch,tail[-n:,1],2)
        slope0=float(np.polyval(np.polyder(coef),0)/pitch)
        limit=PARAMETERS['maximum_endpoint_extension_pitch']*pitch
        for dx in np.arange(PARAMETERS['sample_step_px'],limit+1,PARAMETERS['sample_step_px'])*sign:
            x=x0+dx;y=float(np.polyval(coef,dx/pitch));m=float(np.polyval(np.polyder(coef),dx/pitch)/pitch)
            if not(1<=x<w-1 and 1<=y<h-1) or abs(math.atan(m)-math.atan(slope0))>.45:break
            normal=np.array([-m,1.])/math.sqrt(1+m*m)
            end_extra.append([x,y]);end_width.append(float(tw[-1]));extra_b.append([np.array([x,y])-normal*tw[-1]/2,np.array([x,y])+normal*tw[-1]/2])
    if end_extra:
        tail=np.vstack([tail,end_extra]);tw=np.r_[tw,end_width];bxy=np.concatenate([bxy,np.asarray(extra_b)]);ti=np.r_[ti,np.ones(len(end_extra),bool)];states+=['lane_extrapolation']*len(end_extra);sources+=[['finite_endpoint_extrapolation','finite_endpoint_extrapolation'] for _ in end_extra];snaps=np.vstack([snaps,np.zeros((len(end_extra),2))])
    transition=opt['transition'];start=transition[0];prefix=np.linspace(outer,start,max(2,int(np.linalg.norm(start-outer)/PARAMETERS['sample_step_px'])+1))
    # 共享端点仅保存一次：join_index对应真正的轨道开始，前段用phase切开评分。
    xy=np.vstack([prefix[:-1],transition[:-1],tail]);start_index=len(prefix)-1;join_index=len(prefix)-1+len(transition)-1
    trans_width=np.linspace(width,float(tw[0]),len(transition))
    widths=np.r_[np.full(len(prefix)-1,width),trans_width[:-1],tw]
    phases=['free_straight']*(len(prefix)-1)+['transition']*(len(transition)-1)+states
    inferred=np.r_[np.ones(len(prefix)-1,bool),np.ones(len(transition)-1,bool),ti]
    # free_straight是几何轴向前缀；是否实测由外部自由段检测单独决定。
    tang=np.gradient(xy,axis=0);norm=np.maximum(np.linalg.norm(tang,axis=1),1e-6);normal=np.c_[-tang[:,1],tang[:,0]]/norm[:,None]
    geometry_edges=np.stack([xy-normal*widths[:,None]/2,xy+normal*widths[:,None]/2],axis=1)
    swap=geometry_edges[:,0,1]>geometry_edges[:,1,1]
    geometry_edges[swap]=geometry_edges[swap,::-1]
    geometry_edges[join_index:]=bxy
    distances=_sample(distance,geometry_edges.reshape(-1,2)).reshape(-1,2)
    inherited=_sample((inferred_mask>0).astype(np.float32),geometry_edges.reshape(-1,2)).reshape(-1,2)>.3
    point_source=[['free_axis_geometry','free_axis_geometry'] for _ in prefix[:-1]]+[['C1_transition','C1_transition'] for _ in transition[:-1]]+sources
    if np.any(xy[:,0]<0) or np.any(xy[:,0]>=w) or np.any(xy[:,1]<0) or np.any(xy[:,1]>=h):return None
    return {'candidate_id':None,'lane_id':lane['lane_id'],'xy':xy.tolist(),'widths':widths.tolist(),'phase':phases,
            'point_inferred':inferred.tolist(),'formal_boundary_xy':geometry_edges.tolist(),
            'formal_boundary_distance_px':distances.tolist(),'formal_boundary_sources':point_source,
            'formal_boundary_inferred_mask':inherited.tolist(),'boundary_snap_distance_px':np.vstack([np.zeros((join_index,2)),snaps]).tolist(),
            'source_boundary_ids':lane['source_boundary_ids'],'source_ref_ids':lane['source_ref_ids'],
            'join_x':float(tail[0,0]),'transition_start_x':float(start[0]),'join_index':join_index,'transition_start_index':start_index,
            'geometry_cost':opt['geometry_cost'],'join_offset_px':opt['join_offset_px'],'join_angle_deg':opt['join_angle_deg'],
            'transition_length_width':opt['transition_length_width'],'transition_turn_per_width':opt['transition_turn_per_width'],
            'transition_scale_px':opt['transition_scale_px'],'lane_free_width_ratio':opt['lane_free_width_ratio'],
            'join_forward_s':opt['join_forward_s'],'join_option_role':opt['join_option_role'],
            **{k:opt[k] for k in ('join_tangent_estimator','formal_curvature_sign','formal_rail_second_derivatives','formal_neighbor_second_derivatives','formal_curvature_tolerance','transition_reverse_turn_deg','allowed_reverse_turn_deg','geometry_pixel_tolerance_px')},
            'finite_endpoint_extension_px':float(np.linalg.norm(tail[-1]-tail[-1-len(end_extra)])) if end_extra else 0.,
            'maximum_endpoint_extension_x_px':PARAMETERS['maximum_endpoint_extension_pitch']*pitch,
            'terminal_reason':'formal_lane_end_with_finite_extrapolation' if end_extra else 'formal_lane_support_end',
            'visible_side_reached':None,'same_material_identity':None,'physical_contact':None,
            'interpretation':'F3-constrained geometric lane candidate, not independently observed active rope'}

def generate_lane_candidates(data,free):
    """Required data: merged, inferred, pitch, curve_refs; optional f3_record.

    free may be the full free-rope result or its geometry dict. All coordinates
    are original-image pixels, unmirrored. No candidate is declared the winner.
    """
    result={'status':'UNKNOWN','candidates':[],'lanes':[],'diagnostics':{'parameters':dict(PARAMETERS),'image_scoring_performed':False,'identity_claim':False}}
    geo=free.get('geometry',free)
    if geo.get('status')!='observed' or geo.get('direction_xy') is None:
        result['diagnostics']['reason']='no_unambiguous_free_geometry';return result
    mask=np.asarray(data['merged']);pitch=float(data['pitch']);width=float(geo['width_px']);origin=np.asarray(geo['origin_xy'],float);d=np.asarray(geo['direction_xy'],float)
    if mask.ndim!=2 or not 2<=pitch<=100 or width<=0 or not np.isfinite(np.r_[origin,d,width,pitch]).all():raise ValueError('Invalid original-size lane inputs')
    d=d/max(1e-9,np.linalg.norm(d))
    if abs(d[0])<.25:result['diagnostics']['reason']='near_vertical_free_axis_not_supported_by_x_monotonic_lanes';return result
    if not np.any(mask):result['diagnostics']['reason']='empty_formal_F3_mask';return result
    ends=np.asarray(geo.get('observed_segment_xy',free.get('free_segment',{}).get('observed_segment_xy',[origin,origin])),float).reshape(-1,2)
    outer=ends[int(np.argmin(ends@d))]
    tracks,distance,cache,counts=_snap_refs(data,origin,d,width);rows=_merge_rows(tracks,pitch)
    lanes,pair_counts=_lane_pairs(rows,data,cache,width);options=[];join_rejections=Counter()
    for lane in lanes:lane['_local_geometry']=_local_lane_geometry(lane,pitch)
    for lane in lanes:
        joins=_join_options(lane,origin,d,width,pitch,lanes,join_rejections)
        for rank,opt in enumerate(joins):options.append((rank,opt['geometry_cost'],lane,opt))
    # 每个不同轨道先获得一个机会，再考虑同轨道第二种接法。
    options.sort(key=lambda q:(q[0],q[1] if q[0]==0 else -q[3]['join_forward_s']));selected=[];seen=[];omitted=0
    inferred_mask=np.asarray(data.get('inferred',np.zeros_like(mask)))
    for rank,cost,lane,opt in options:
        candidate=_make_candidate(lane,opt,origin,d,width,pitch,outer,distance,inferred_mask,mask.shape)
        if candidate is None:continue
        # 几何上几乎重合的双提案不占用邻绳竞争预算。
        q=np.asarray(candidate['xy']);tail=q[candidate['join_index']:];duplicate=False
        for other in seen:
            oo=np.asarray(other['xy'])[other['join_index']:];a=max(min(tail[:,0]),min(oo[:,0]));b=min(max(tail[:,0]),max(oo[:,0]))
            if b-a<3*pitch:continue
            x=np.linspace(a,b,30);ta=tail[np.argsort(tail[:,0])];tb=oo[np.argsort(oo[:,0])]
            if np.median(abs(np.interp(x,ta[:,0],ta[:,1])-np.interp(x,tb[:,0],tb[:,1])))<.25*width and abs(candidate['join_x']-other['join_x'])<2*width:
                duplicate=True;break
        if duplicate:continue
        if len(selected)>=PARAMETERS['maximum_candidates']:omitted+=1;continue
        candidate['candidate_id']='lane_path_'+str(len(selected)).zfill(2);selected.append(candidate);seen.append(candidate)
    kept={c['lane_id'] for c in selected}
    for lane in lanes:
        if lane['lane_id'] not in kept:continue
        boundary_dist=_sample(distance,lane['boundary_xy'].reshape(-1,2)).reshape(-1,2)
        result['lanes'].append({'lane_id':lane['lane_id'],'xy':np.c_[lane['x'],lane['y']].tolist(),'widths':lane['widths'].tolist(),
                              'boundary_xy':lane['boundary_xy'].tolist(),'point_inferred':lane['inferred'].tolist(),
                              'formal_boundary_distance_px':boundary_dist.tolist(),'formal_boundary_sources':lane['point_boundary_sources'],
                              'boundary_snap_distance_px':lane['snap_distance_px'].tolist(),'source_boundary_ids':lane['source_boundary_ids'],
                              'source_ref_ids':lane['source_ref_ids'],'intermediate_formal_fraction':lane['intermediate_formal_fraction']})
    result['candidates']=selected;result['status']='MULTI_CANDIDATE' if len(selected)>1 else ('CANDIDATE_ONLY' if selected else 'UNKNOWN')
    result['diagnostics'].update({'reason':'finite_formal_F3_lane_geometry' if selected else 'no_supported_adjacent_lane_with_bounded_join',
       'reference_extraction':counts,'boundary_row_count':len(rows),'lane_pair_count':len(lanes),'pair_rejections':pair_counts,
       'join_option_count':len(options),'candidate_budget_omitted':omitted,'selected_candidate_count':len(selected),
       'join_rejections':dict(join_rejections),
       'formal_authority':'merged only; references are proposals snapped to formal pixels',
       'coverage_limit':'F2-only formal pieces may have no corresponding C3/fullD4/F3-group proposal; uncovered formal pixels are reported.',
       'tail_limit':'Formal lane end plus at most 2 pitch in x; not a verified physical disappearance endpoint.',
       'free_prefix_limit':'The geometric straight prefix is not automatically measured support; caller must use independent free-rope mask.'})
    return result
