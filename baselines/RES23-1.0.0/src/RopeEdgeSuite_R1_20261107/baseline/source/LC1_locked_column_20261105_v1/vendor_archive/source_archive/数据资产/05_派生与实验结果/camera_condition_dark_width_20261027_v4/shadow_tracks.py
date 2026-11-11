"""从新宽沟槽提取黄点并沿图像沟槽归类；不使用D2/D3旧点或旧组号。"""
import math
import cv2
import numpy as np

PARAMETERS={'column_step_pitch':.22,'minimum_score':.32,'minimum_width_pitch':.12,
 'maximum_width_pitch':.72,'column_nms_pitch':.58,'body_limit':.65,
 'maximum_gap_pitch':1.25,'link_search_pitch':.38,'max_abs_slope':2.5,
 'link_body_run_pitch':.18,'link_body_fraction':.12,'minimum_track_points':8}

def _values(layer,x,y,field):
    xx=np.rint(x).astype(int)-field['x0'];yy=np.rint(y).astype(int)-field['y0']
    valid=(xx>=0)&(xx<layer.shape[1])&(yy>=0)&(yy<layer.shape[0])
    result=np.zeros(np.shape(xx),float)
    result[valid]=layer[yy[valid],xx[valid]]
    return result

def _link_clear(a,b,shadow,guard,pitch):
    count=max(2,int(math.ceil(math.hypot(b['x']-a['x'],b['y']-a['y'])))+1)
    xx=np.linspace(a['x'],b['x'],count);yy=np.linspace(a['y'],b['y'],count)
    if np.any(_values(guard['unknown_mask'],xx,yy,shadow)>0):return False,'unknown_occlusion'
    strong=_values(guard['body_barrier'],xx,yy,shadow)>PARAMETERS['body_limit']
    run=longest=0
    for v in strong:
        run=run+1 if v else 0;longest=max(longest,run)
    if longest>=max(3,int(math.ceil(PARAMETERS['link_body_run_pitch']*pitch))) or strong.mean()>PARAMETERS['link_body_fraction']:
        return False,'crosses_supported_rope_body'
    return True,'clear'

def detect_and_group(shadow,guard,roi,pitch):
    """黄点先过宽度/原图/绳体门，再按局部预测、顺序和路径像素归组。"""
    x0,y0=int(shadow['x0']),int(shadow['y0'])
    mask=(shadow['valid_support']>0)&(shadow['cleaned_shadow']>0)
    score=shadow['broad_score'];width=shadow['width_px'];slope=shadow['tangent_slope']
    _,labels=cv2.connectedComponents(shadow['cleaned_shadow'])
    step=max(3,int(round(PARAMETERS['column_step_pitch']*pitch)))
    points=[];rejected=[];columns=[]
    for ix in range(0,mask.shape[1],step):
        local_pitch=float(np.clip(guard['local_pitch_by_column'][ix],.70*pitch,1.40*pitch))
        rows=np.flatnonzero(mask[:,ix])
        if not len(rows):continue
        groups=np.split(rows,np.flatnonzero(np.diff(rows)>1)+1)
        candidates=[]
        for group in groups:
            iy=int(group[np.argmax(score[group,ix]+.08*np.minimum(width[group,ix]/pitch,1.))])
            p={'x':ix+x0,'y':float(iy+y0),'score':float(score[iy,ix]),
               'width':float(width[iy,ix]),'slope':float(slope[iy,ix]),'component':int(labels[iy,ix]),'pitch':local_pitch}
            reasons=[]
            normal_pitch=local_pitch/math.sqrt(1+p['slope']**2)
            if p['score']<PARAMETERS['minimum_score']:reasons.append('weak_original_groove_support')
            if not PARAMETERS['minimum_width_pitch']*normal_pitch<=p['width']<=PARAMETERS['maximum_width_pitch']*normal_pitch:
                reasons.append('shadow_width_outside_rope_gap_scale')
            if guard['body_barrier'][iy,ix]>PARAMETERS['body_limit']:reasons.append('inside_supported_rope_body')
            if guard['unknown_mask'][iy,ix]:reasons.append('unknown_occlusion')
            if reasons:rejected.append({**p,'reasons':reasons})
            else:candidates.append(p)
        kept=[]
        for p in sorted(candidates,key=lambda p:-p['score']):
            if any(abs(p['y']-q['y'])<PARAMETERS['column_nms_pitch']*local_pitch for q in kept):
                rejected.append({**p,'reasons':['same_column_competing_shadow']})
            else:kept.append(p)
        kept.sort(key=lambda p:p['y'])
        for p in kept:p['point_id']=len(points);points.append(p)
        columns.append((ix+x0,kept))
    tracks=[];active=[];link_rejections=[]
    for x,candidates in columns:
        active=[i for i in active if x-tracks[i]['points'][-1]['x']<=PARAMETERS['maximum_gap_pitch']*pitch]
        edges=[]
        for ti in active:
            t=tracks[ti]['points'];last=t[-1];dx=x-last['x']
            if dx<=0:continue
            recent=t[-6:]
            if len(recent)>=3:
                xx=np.asarray([p['x'] for p in recent],float);yy=np.asarray([p['y'] for p in recent],float)
                direction=float(np.polyfit(xx-xx.mean(),yy,1)[0])
                direction=.8*direction+.2*last['slope']
            else:direction=last['slope']
            direction=float(np.clip(direction,-PARAMETERS['max_abs_slope'],PARAMETERS['max_abs_slope']))
            predicted=last['y']+direction*dx
            for pi,p in enumerate(candidates):
                error=abs(p['y']-predicted)
                if error>PARAMETERS['link_search_pitch']*pitch+.10*dx:continue
                if abs((p['y']-last['y'])/dx)>PARAMETERS['max_abs_slope']:continue
                okay,reason=_link_clear(last,p,shadow,guard,pitch)
                if not okay:
                    link_rejections.append({'from_point':last['point_id'],'to_point':p['point_id'],'reason':reason});continue
                cost=error/pitch+.10*abs(p['slope']-direction)+.08*abs(math.log(max(p['width'],1)/max(last['width'],1)))
                edges.append((cost,ti,pi))
        used_t=set();used_p=set();matches=[]
        for cost,ti,pi in sorted(edges):
            if ti in used_t or pi in used_p:continue
            previous=tracks[ti]['points'][-1]
            # 同一列的一对连接不得反转上下顺序；不能用交叉线解释周期多解。
            if any((previous['y']-tracks[tj]['points'][-1]['y'])*(candidates[pi]['y']-candidates[pj]['y'])<0 for tj,pj in matches):
                continue
            matches.append((ti,pi));used_t.add(ti);used_p.add(pi)
        for ti,pi in matches:tracks[ti]['points'].append(candidates[pi])
        for pi,p in enumerate(candidates):
            if pi not in used_p:
                active.append(len(tracks));tracks.append({'track_id':f'wide_track_{len(tracks)+1:05d}','points':[p]})
    observed=np.zeros_like(roi)
    for p in points:cv2.circle(observed,(p['x'],int(round(p['y']))),1,255,-1)
    observed[roi==0]=0
    return {'points':points,'tracks':tracks,'rejected_points':rejected,'rejected_links':link_rejections,
       'observed_mask':observed,'diagnostics':{'new_point_count':len(points),'rejected_point_count':len(rejected),
         'track_count':len(tracks),'link_veto_count':len(link_rejections),'parameters':PARAMETERS.copy(),
         'old_observations_used':False,'interpretation':'fresh image-supported candidates, not truth labels'}}
