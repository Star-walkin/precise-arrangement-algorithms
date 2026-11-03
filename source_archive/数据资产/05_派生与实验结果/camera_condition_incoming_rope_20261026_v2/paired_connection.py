"""A2 双边接续：直行必须参与竞争，增强只用于找候选，原图仍要支持。

这是单帧图像假设，不输出物理接触。C3暗沟是卷面排列先验，不能自动
变成活动绳身份。本模块不读文件、不访问设备，不接收任何目标标注。
"""
import math
import cv2
import numpy as np

PARAMETERS={'step_px':6.,'max_forward_px':2800.,'max_candidates':7,
            'width_range':(.60,1.55),'max_join_distance_widths':1.15,
            'max_turn_angle_deg':35.,'transition_length_widths':6.,
            'minimum_bend_radius_widths':3.,'min_branch_support':.45,
            'min_transition_support':.65,'max_transition_gap_widths':2.,
            'min_branch_samples':12,'min_curve_advantage':.14,
            'max_supported_straight_fraction':.30,'min_ambiguity_margin':.08}

def sample(a,xy):
    xy=np.asarray(xy,np.float32)
    return cv2.remap(a,xy[...,0],xy[...,1],cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT,borderValue=0)

def model_y(m,x):
    u=(x-m['u_center_x_px'])/m['u_scale_x_px']
    return m['normalized_k']+m['normalized_b']*u+m['normalized_c']*u*u

def model_slope(m,x):
    return (m['normalized_b']+2*m['normalized_c']*(x-m['u_center_x_px'])/m['u_scale_x_px'])/m['u_scale_x_px']

def edge_evidence(raw,enh,path,widths,surface):
    """每个法向剖面独立验证两边。这里的 supported 是图像支持，非同根证明。"""
    path=np.asarray(path,float);widths=np.asarray(widths,float)
    if len(path)<3:return None
    tangent=np.gradient(path,axis=0);tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-6)
    normals=np.column_stack((-tangent[:,1],tangent[:,0]))
    offsets=np.linspace(-1.25,1.25,61)
    xy=path[:,None,:]+normals[:,None,:]*widths[:,None,None]*offsets[None,:,None]
    r=sample(raw,xy);e=sample(enh,xy)
    rs=cv2.GaussianBlur(r,(5,1),.8);es=cv2.GaussianBlur(e,(5,1),.8)
    rg=np.gradient(rs,axis=1);eg=np.gradient(es,axis=1)
    left=np.where((offsets>=-.70)&(offsets<=-.30))[0]
    right=np.where((offsets>=.30)&(offsets<=.70))[0]
    li=left[np.argmax(np.abs(eg[:,left]),axis=1)]
    ri=right[np.argmax(np.abs(eg[:,right]),axis=1)]
    ix=np.arange(len(path))
    # 两侧应为相反极性的边缘；强度用相同距离上的原图复核。
    el,er=eg[ix,li],eg[ix,ri];rl,rr=rg[ix,li],rg[ix,ri]
    opposite=(el*er<0)&(rl*rr<0)
    raw_edge=np.minimum(np.abs(rl),np.abs(rr))
    enh_edge=np.minimum(np.abs(el),np.abs(er))
    noise=np.median(np.abs(np.diff(r,n=2,axis=1)),axis=1)*.65+.25
    band=np.ptp(rs,axis=1)
    rawlevel=np.median(r,axis=1)
    satur=np.mean(r>=250,axis=1)
    # 深暗区域即使增强后有纹理，也不能因此成为已观测点。
    valid=(rawlevel>=5)&(satur<.45)&(band>=3.)
    support=valid&opposite&(raw_edge>=np.maximum(.65,noise*.65))&(enh_edge>=.9)
    h,w=surface.shape
    inside=(path[:,0]>=1)&(path[:,0]<w-1)&(path[:,1]>=1)&(path[:,1]<h-1)
    support &= inside
    score=np.clip(raw_edge/(noise+1.),0,4.)*.16+np.clip(enh_edge/4.,0,1.)*.25+opposite*.11
    score=np.where(valid&inside,score,0.)
    # 双边定位在图像实际梯度峰上；只有supported的点可作为实测绘出。
    edge_l=path+normals*widths[:,None]*offsets[li,None]
    edge_r=path+normals*widths[:,None]*offsets[ri,None]
    refined_center=(edge_l+edge_r)*.5
    on_surface=sample((surface>0).astype(np.float32),path[:,None,:]).ravel()>.5
    return {'support':support,'score':score,'raw_edge':raw_edge,'noise':noise,
            'left':edge_l,'right':edge_r,'refined_center':refined_center,
            'valid':valid&inside,'on_surface':on_surface}

def _longest_run(mask):
    padded=np.r_[False,mask,False].astype(np.int8);d=np.diff(padded)
    return int(max((b-a for a,b in zip(np.flatnonzero(d==1),np.flatnonzero(d==-1))),default=0))

def _candidate(cid,kind,path,widths,raw,enh,surface,join=None,groups=None):
    ev=edge_evidence(raw,enh,path,widths,surface)
    if ev is None:return None
    on=ev['on_surface'];observed=ev['support']&on
    states=np.where(observed,'observed',np.where(on&ev['valid'],'inferred','unavailable'))
    return {'candidate_id':cid,'kind':kind,'path_xy':np.round(path,3).tolist(),
        'widths_px':np.round(widths,3).tolist(),'boundary_left_xy':np.round(ev['left'],3).tolist(),
        'boundary_right_xy':np.round(ev['right'],3).tolist(),'point_states':states.tolist(),
        'join_index':join,'c3_group_ids':groups or [],'transition_center_xy':None,
        'observed_count':int(observed.sum()),'surface_sample_count':int(on.sum()),
        'score_mean':float(np.mean(ev['score'][on])) if on.any() else 0.,
        'reject_reasons':[],'_ev':ev}

def find_connections(image_bgr,enhanced_bgr,surface_mask,free,c3_result):
    result={'status':'UNKNOWN','selected_candidate_id':None,'candidates':[],
            'physical_contact_confirmed':False,'contact_point_xy':None,
            'diagnostics':{'parameters':PARAMETERS,'straight_is_competitor':True}}
    if free.get('status')!='observed' or free.get('toward_surface_direction_xy') is None:
        result['reason']='free_span_or_direction_unavailable';return result
    p=np.asarray(free.get('observed_points_xy',[]),float);width=float(free['width_px'])
    if len(p)<3 or width<2:return result
    d=np.asarray(free['toward_surface_direction_xy'],float);d/=np.linalg.norm(d)
    n=np.array([-d[1],d[0]])
    # 用整个自由段的联合拟合直线定义起点。单个末端观测点带有股纹噪声，
    # 直接拿它外推，会把这一个点的误差平移到整个卷面。
    line_center=np.asarray(free['center_xy'],float)
    start=line_center+float(np.max((p-line_center)@d))*d
    raw=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)
    enh=cv2.cvtColor(enhanced_bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)
    # 范围取实际surface在方向上的跨度；薄绳也能抵达暗侧，而非固定90绳宽。
    yy,xx=np.where(surface_mask[::4,::4]>0)
    if not len(xx):return result
    surface_points=np.column_stack((xx*4,yy*4))
    length=min(PARAMETERS['max_forward_px'],float(np.max((surface_points-start)@d))+2*width)
    t=np.arange(0,max(0,length),PARAMETERS['step_px'])
    path=start+t[:,None]*d;h,w=surface_mask.shape
    ok=(path[:,0]>=1)&(path[:,0]<w-1)&(path[:,1]>=1)&(path[:,1]<h-1)
    path=path[ok];t=t[ok]
    if len(path)<15:return result
    widths=np.full(len(path),width)
    straight=_candidate('S0','straight_continuation',path,widths,raw,enh,surface_mask)
    candidates=[straight]
    curves=c3_result.get('fit',{}).get('curves',[])
    # 接续只取新C3观测拟合的暗沟，不拿A1的错接轨迹作输入。
    usable=[c for c in curves if c.get('model') and len(c.get('sampled_points_xy',[]))>=4]
    xref=float(np.median(path[:,0]))
    usable.sort(key=lambda c:float(model_y(c['model'],xref)))
    proposals=[]
    if abs(d[0])>.1:
        free_slope=d[1]/d[0]
        for i,a in enumerate(usable):
            for b in usable[i+1:i+4]:
                y1=model_y(a['model'],path[:,0]);y2=model_y(b['model'],path[:,0])
                center=(y1+y2)*.5
                slopes=(model_slope(a['model'],path[:,0])+model_slope(b['model'],path[:,0]))*.5
                band_width=np.abs(y2-y1)/np.sqrt(1+slopes**2)
                offset=(center-path[:,1])/math.sqrt(1+free_slope**2)
                a_pts=np.asarray(a['sampled_points_xy'],float);b_pts=np.asarray(b['sampled_points_xy'],float)
                # 候选只能在有新C3曲线范围的邻近处相接，不无限外推隐藏暗面。
                xlo=max(a_pts[:,0].min(),b_pts[:,0].min());xhi=min(a_pts[:,0].max(),b_pts[:,0].max())
                eligible=(t>2*width)&(t<t[-1]-5*width)&(path[:,0]>=xlo)&(path[:,0]<=xhi)
                eligible&=(band_width>=width*.6)&(band_width<=width*1.55)
                # 弯段末端可以有不同切向；起点由连续过渡匹配直线，不能
                # 提前要求目标旧圈在起点就与直线平行。仍限制总转角和曲率。
                turn=np.abs(np.arctan(slopes)-math.atan(free_slope))
                transition=PARAMETERS['transition_length_widths']*width
                end_x=path[:,0]+transition*d[0]
                eligible&=(np.abs(offset)<=width*1.15)&(turn<=math.radians(PARAMETERS['max_turn_angle_deg']))
                eligible&=(end_x>=xlo)&(end_x<=xhi)&(t+transition<t[-1])
                if not eligible.any():continue
                geometry=np.abs(offset)/width+turn*2
                geometry[~eligible]=np.inf;j=int(np.argmin(geometry))
                proposals.append((float(geometry[j]),j,center,band_width,a,b,xlo,xhi))
        proposals.sort(key=lambda q:q[0])
    for geom,j,center,band_width,a,b,xlo,xhi in proposals:
        # C1连续的有限过渡：接点附近的变弯仍要在原图上战胜直行假设。
        blend=np.clip((t-t[j])/(PARAMETERS['transition_length_widths']*width),0,1);blend=blend*blend*(3-2*blend)
        curved=path.copy();curved[:,1]=(1-blend)*path[:,1]+blend*center
        model_width=(1-blend)*width+blend*band_width
        velocity=np.gradient(curved,axis=0);acceleration=np.gradient(velocity,axis=0)
        curvature=np.abs(velocity[:,0]*acceleration[:,1]-velocity[:,1]*acceleration[:,0])/np.maximum(np.linalg.norm(velocity,axis=1)**3,1e-6)
        changing=(t>=t[j])&(t<=t[j]+PARAMETERS['transition_length_widths']*width)
        if np.max(curvature[changing],initial=0.)>1./(PARAMETERS['minimum_bend_radius_widths']*width):continue
        if any(np.mean(np.linalg.norm(curved-np.asarray(c['path_xy']),axis=1))<width*.12 for c in candidates[1:]):continue
        c=_candidate('C'+str(len(candidates)),'c3_paired_groove_connection',curved,model_width,raw,enh,surface_mask,j,
                     [a.get('group_id'),b.get('group_id')])
        c['geometry_cost']=geom
        ce=c['_ev'];se=straight['_ev']
        # 有限新C3采样范围以外只保存候选数学坐标，不作为可显示/可选路径。
        domain=(t<t[j])|((path[:,0]>=xlo)&(path[:,0]<=xhi))
        ce['support']&=domain;ce['on_surface']&=domain
        c['point_states']=[s if domain[k] else 'unavailable' for k,s in enumerate(c['point_states'])]
        c['observed_count']=sum(s=='observed' for s in c['point_states'])
        c['surface_sample_count']=int(ce['on_surface'].sum())
        separation=np.linalg.norm(curved-path,axis=1)
        branch=(separation>=width*.6)&ce['on_surface']&(t>=t[j])&(path[:,0]>=xlo)&(path[:,0]<=xhi)
        count=int(branch.sum());support=float(np.mean(ce['support'][branch])) if count else 0.
        straight_support=float(np.mean(se['support'][branch])) if count else 0.
        advantage=float(np.mean(ce['score'][branch]-se['score'][branch])) if count else 0.
        # 可见直绳若仍持续，不允许仅因旁边旧圈很清晰就提前弯走。
        straight_run=_longest_run(se['support']&branch)
        c['branch_comparison']={'samples':count,'curve_support':support,'straight_support':straight_support,
            'enhanced_and_raw_score_advantage':advantage,'straight_supported_run_samples':straight_run}
        # 远端旧圈纹路很清楚，并不能证明直绳能穿过中间暗缺口接上它。
        # 将整个有限过渡段作为独立验收范围，要求连续的原图双边证据。
        transition_ids=np.flatnonzero(changing)
        transition_support=float(np.mean(ce['support'][transition_ids]&ce['on_surface'][transition_ids])) if len(transition_ids) else 0.
        transition_gap=_longest_run(~(ce['support'][transition_ids]&ce['on_surface'][transition_ids]))*PARAMETERS['step_px']
        c['transition_evidence']={'sample_count':len(transition_ids),'paired_support_fraction':transition_support,
            'longest_unsupported_gap_px':transition_gap,'required_max_gap_px':PARAMETERS['max_transition_gap_widths']*width}
        if transition_support<PARAMETERS['min_transition_support'] or transition_gap>PARAMETERS['max_transition_gap_widths']*width:
            c['reject_reasons'].append('transition_itself_not_continuously_observed')
        if count<12:c['reject_reasons'].append('insufficient_distinct_branch_samples')
        if support<.45:c['reject_reasons'].append('paired_curve_edges_not_sustained')
        if advantage<.14:c['reject_reasons'].append('curve_does_not_beat_straight')
        if straight_support>.30 or straight_run>=8:c['reject_reasons'].append('visible_straight_continuation_contradicts_bend')
        c['selection_score']=advantage+.2*support
        if not c['reject_reasons']:c['transition_center_xy']=curved[j].round(3).tolist()
        candidates.append(c)
        if len(candidates)>=PARAMETERS['max_candidates']:break
    viable=sorted([c for c in candidates[1:] if not c['reject_reasons']],key=lambda c:-c['selection_score'])
    surface_support=straight['observed_count']/max(1,straight['surface_sample_count'])
    if viable and (len(viable)==1 or viable[0]['selection_score']-viable[1]['selection_score']>=.08):
        result['status']='CURVE_CANDIDATE';result['selected_candidate_id']=viable[0]['candidate_id']
        result['reason']='paired_c3_projection_locally_exceeds_straight_not_physical_contact'
    elif viable:
        result['status']='AMBIGUOUS';result['reason']='multiple_c3_connections_remain'
    elif surface_support>=.25 and straight['observed_count']>=15:
        result['status']='STRAIGHT_SUPPORTED';result['selected_candidate_id']='S0'
        result['reason']='visible_straight_portions_only_no_confirmed_turn'
    else:result['reason']='insufficient_paired_edge_evidence'
    result['diagnostics'].update({'new_c3_curve_count':len(usable),'candidate_proposal_count':len(proposals),
        'forward_length_px':float(t[-1]),'straight_support_fraction':surface_support})
    for c in candidates:c.pop('_ev',None)
    result['candidates']=candidates
    return result
