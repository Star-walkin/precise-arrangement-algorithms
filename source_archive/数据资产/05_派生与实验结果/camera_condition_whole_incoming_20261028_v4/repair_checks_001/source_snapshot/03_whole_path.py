"""连续带状路径搜索。每一步同时量测两侧和内部，不把沟边当绳中心。

这是二维可见外观候选，不是已经验证的材料身份。父节点保持独立路径，
不能把不同旧圈的好点合并成一根绳。候选评分相近时明确保留歧义。
"""
import math
import cv2
import numpy as np
PARAMETERS={'step_x':7.,'beam':72,'turn_radians':.014,'maximum_tangent':3.2,
 'maximum_weak_run_width':4.,'minimum_supported_fraction':.35,
 'maximum_search_widths':110.,'width_scales':[.8,1.,1.2],
 'curve_weight':.12,'texture_weight':.07,'minimum_support_score':.27}
PARAMETERS['straight_single_edge_budget_width']=18.

def build_search_surface(surface):
 """仅构造搜索范围，不改变 SAM 原掩膜，也不把该范围当绳体实测。

 A3 先过滤小轮廓生成凸包，却又把原掩膜 OR 回凸包，导致被过滤的
 碎片重新出现。自由绳碰到几像素碎片后就会被认为已经进入卷面，
 随后的正常自由空间反而触发“已离开卷面”。这里保留原来的面积
 阈值和凸包限制，只让显著组件决定进入/离开搜索域的状态。

 返回 (搜索掩膜, 诊断信息)。此函数只依赖输入数组，供其他纯离线
 模块复用；无文件访问、设备初始化或输入数组的原地修改。
 """
 source=np.asarray(surface)
 if source.ndim!=2:raise ValueError('surface must be a two-dimensional mask')
 binary=np.where(source>0,255,0).astype(np.uint8)
 original_pixels=int(np.count_nonzero(binary));minimum_area=max(50.,.0001*original_pixels)
 contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 selected=[c for c in contours if cv2.contourArea(c)>=minimum_area]
 significant=np.zeros_like(binary)
 if selected:cv2.drawContours(significant,selected,-1,255,cv2.FILLED)
 discarded=int(np.count_nonzero((binary>0)&(significant==0)))
 candidate_ratio=None
 if selected:
  envelope=np.zeros_like(binary)
  cv2.fillConvexPoly(envelope,cv2.convexHull(np.vstack(selected)),255)
  candidate_ratio=float(np.count_nonzero(envelope)/max(1,original_pixels))
  if candidate_ratio<=1.8:
   search=envelope;method='significant_component_convex_hull'
  else:
   search=significant;method='significant_components_hull_area_limit'
 else:
  search=significant;method='no_significant_component'
 return search,{'search_surface_method':method,'minimum_contour_area_px':float(minimum_area),
  'original_surface_pixels':original_pixels,'selected_component_count':len(selected),
  'significant_component_pixels':int(np.count_nonzero(significant)),
  'discarded_fragment_pixels':discarded,'convex_hull_candidate_area_ratio':candidate_ratio,
  'visibility_envelope_area_ratio':float(np.count_nonzero(search)/max(1,original_pixels)),
  'visibility_envelope_interpretation':'bounded significant-component search envelope only; never observed rope or verified physical boundary'}

def _sample(a,xy):
 q=np.asarray(xy,np.float32);shape=q.shape[:-1]
 return cv2.remap(a,q[...,0].reshape(1,-1),q[...,1].reshape(1,-1),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101).reshape(shape)

def _fields(image,width):
 g=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)
 sigma=max(3.,width*1.8);m=cv2.GaussianBlur(g,(0,0),sigma)
 std=np.sqrt(np.maximum(cv2.GaussianBlur((g-m)**2,(0,0),sigma),4.))
 z=(g-m)/std
 # 原灰度梯度的结构张量描述局部股纹方向；归一化仅用于暗部增强。
 b=cv2.GaussianBlur(g,(0,0),1.2);gx=cv2.Sobel(b,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(b,cv2.CV_32F,0,1,ksize=3)
 s=max(2.,width*.3);jxx=cv2.GaussianBlur(gx*gx,(0,0),s);jyy=cv2.GaussianBlur(gy*gy,(0,0),s);jxy=cv2.GaussianBlur(gx*gy,(0,0),s)
 return g,z,std,jxx,jyy,jxy

def _profiles(fields,centers,slopes,width,xsign,texture=None):
 g,z,std,jxx,jyy,jxy=fields;n=len(centers)
 tang=np.c_[np.full(n,xsign),slopes*xsign];tang/=np.linalg.norm(tang,axis=1)[:,None]
 normal=np.c_[-tang[:,1],tang[:,0]]
 offsets=np.array([-1.,-.75,-.55,-.4,-.2,0,.2,.4,.55,.75,1.])
 scores=[];supports=[];contrasts=[];strongs=[];rawstrongs=[]
 for ws in PARAMETERS['width_scales']:
  ww=width*ws
  # 沿路径方向平均，压制窄而斜的股间纹理；跨绳方向保留宽沟/外缘。
  coords=centers[:,None,None,:]+offsets[None,:,None,None]*ww*normal[:,None,None,:]+np.array([-.45,0,.45])[None,None,:,None]*ww*tang[:,None,None,:]
  prof=_sample(z,coords).mean(axis=2);raw=_sample(g,coords).mean(axis=2)
  body=prof[:,4:7].mean(axis=1);l=prof[:,1:3].min(axis=1);r=prof[:,8:10].min(axis=1)
  rl=raw[:,1:3].min(axis=1);rr=raw[:,8:10].min(axis=1);rb=raw[:,4:7].mean(axis=1)
  weak=np.minimum(body-l,body-r);strong=np.maximum(body-l,body-r)
  rawweak=np.minimum(rb-rl,rb-rr)
  # 直绳在高亮卷面上可能成为暗条。两种极性使用同一宽度约束，
  # 不能把暗部的单个细股缝直接当一根绳：要有相隔约光学宽度的双边。
  dl=prof[:,1:3].max(axis=1)-body;dr=prof[:,8:10].max(axis=1)-body
  darkweak=np.minimum(dl,dr);darkstrong=np.maximum(dl,dr)
  darkraw=np.minimum(raw[:,1:3].max(axis=1)-rb,raw[:,8:10].max(axis=1)-rb)
  bright_raw_strong=np.maximum(rb-rl,rb-rr);dark_raw_strong=np.maximum(raw[:,1:3].max(axis=1)-rb,raw[:,8:10].max(axis=1)-rb)
  brightscore=.70*np.clip(weak/.7,0,1)+.30*np.clip(strong/1.2,0,1)
  darkscore=.70*np.clip(darkweak/.7,0,1)+.30*np.clip(darkstrong/1.2,0,1)
  use_dark=darkscore>brightscore
  weak=np.where(use_dark,darkweak,weak);rawweak=np.where(use_dark,darkraw,rawweak)
  score=np.maximum(brightscore,darkscore)-.07*abs(ws-1.)/.2
  # 两侧都有原图差异才称实测支持；归一化不能凭空制造暗处证据。
  support=(weak>.16)&(rawweak>1.0)&(score>PARAMETERS['minimum_support_score'])
  scores.append(score);supports.append(support);contrasts.append(rawweak)
  strongs.append(np.where(use_dark,darkstrong,strong));rawstrongs.append(np.where(use_dark,dark_raw_strong,bright_raw_strong))
 scores=np.asarray(scores);best=np.argmax(scores,axis=0);ii=np.arange(n)
 result=scores[best,ii];support=np.asarray(supports)[best,ii]
 width_out=width*np.asarray(PARAMETERS['width_scales'])[best]
 texscore=np.zeros(n)
 if texture and texture.get('status')=='supported':
  xx=_sample(jxx,centers);yy=_sample(jyy,centers);xy=_sample(jxy,centers)
  angle=.5*np.arctan2(2*xy,xx-yy)+math.pi/2
  coherence=np.sqrt((xx-yy)**2+4*xy**2)/(xx+yy+1e-6)
  target=math.radians(float(texture.get('angle_deg',0)))+np.arctan(slopes)-float(texture.get('free_axis_angle_rad',0))
  texscore=coherence*np.cos(2*(angle-target))
  result+=PARAMETERS['texture_weight']*texscore
 xx=_sample(jxx,centers);yy=_sample(jyy,centers);xy=_sample(jxy,centers)
 alignment=(normal[:,0]**2*xx+2*normal[:,0]*normal[:,1]*xy+normal[:,1]**2*yy)/(xx+yy+1e-6)
 one=(np.asarray(strongs)[best,ii]>.35)&(np.asarray(rawstrongs)[best,ii]>3.)&(alignment>.60)
 return result,support,width_out,{'raw_weak':np.asarray(contrasts)[best,ii],'texture_score':texscore,'single_edge_support':one}

def _reconstruct(layers,index):
 out=[]
 for layer in reversed(layers):
  s=layer[index];out.append(s);index=s['parent']
 return list(reversed(out))

def _candidate_entered_surface(states):
 """最终实际保存的路径须触及显著搜索域；探测后又裁掉的点不算。

 全程在域外的纹理响应，只能是自由绳/背景的候选，不能证明卷面接续。
 原SAM漏分可能让真实路径也被拒绝，所以拒绝原因是搜索域不足，
 绝不能解释成原图没有活动绳或已经到了物理消失点。
 """
 return bool(states) and (bool(states[0].get('entered_surface',False)) or any(bool(s.get('inside_surface',False)) for s in states))

def _paint(shape,xy,widths,states=None):
 mask=np.zeros(shape,np.uint8);p=np.asarray(xy,float)
 if len(p)<2:return mask
 for i in range(1,len(p)):
  if states is not None and not(states[i-1] and states[i]):continue
  d=p[i]-p[i-1];length=np.linalg.norm(d)
  if length<.01:continue
  n=np.array([-d[1],d[0]])/length
  poly=np.array([p[i-1]+n*widths[i-1]/2,p[i]+n*widths[i]/2,p[i]-n*widths[i]/2,p[i-1]-n*widths[i-1]/2])
  cv2.fillConvexPoly(mask,np.rint(poly).astype(np.int32),255)
 return mask

def _row_bonus(points,occlusion,width,origin,d):
 if not occlusion.get('candidates'):return 0.,None
 p=np.asarray(points);best=(0.,None);normal=np.array([-d[1],d[0]])
 take=abs((p-origin)@normal)>.6*width;p=p[take]
 if len(p)<6:return best
 order=np.argsort(p[:,0]);p=p[order];pm=np.gradient(p[:,1],p[:,0])
 for c in occlusion['candidates']:
  q=np.asarray(c['sampled_points_xy'],float);q=q[np.argsort(q[:,0])]
  valid=(p[:,0]>=q[0,0])&(p[:,0]<=q[-1,0])
  if np.count_nonzero(valid)<6:continue
  pp=p[valid];yy=np.interp(pp[:,0],q[:,0],q[:,1]);mm=np.interp(pp[:,0],q[:,0],np.gradient(q[:,1],q[:,0]))
  # 沟边的±半宽处才是候选绳体；用法向距离和切向同时确认贴行。
  err=np.minimum(abs((pp[:,1]-yy)/np.sqrt(1+mm*mm)-width/2),abs((pp[:,1]-yy)/np.sqrt(1+mm*mm)+width/2))
  match=(err<width*.35)&(abs(np.arctan(pm[valid])-np.arctan(mm))<.16)
  score=float(c['transition_score'])*float(np.mean(match))*min(1.,np.count_nonzero(match)*7/(4*width))
  if score>best[0]:best=(float(score),c['candidate_id'])
 return best

def _tangent_zones(refs,origin,d,width,pitch):
 """只用实际采样范围寻找与直绳近切的卷面边，禁止无限抛物线外推。
 双边距离约束是几何候选条件，不是接触真值。"""
 n=np.array([-d[1],d[0]]);zones=[]
 for ref in refs:
  q=np.asarray(ref['sampled_points_xy'],float)
  if len(q)<12 or np.ptp(q[:,0])<3*width:continue
  q=q[np.argsort(q[:,0])];q=q[np.r_[True,np.diff(q[:,0])>0]]
  x=np.arange(q[0,0]+width,q[-1,0]-width,7.)
  if not len(x):continue
  y=np.interp(x,q[:,0],q[:,1]);m=(np.interp(x+width,q[:,0],q[:,1])-np.interp(x-width,q[:,0],q[:,1]))/(2*width)
  tangent=np.c_[np.ones(len(m)),m];tangent/=np.linalg.norm(tangent,axis=1)[:,None]
  alignment=np.abs(tangent@d);signed=(np.c_[x,y]-origin)@n
  valid=(alignment>math.cos(math.radians(8)))&(abs(signed)<max(width,1.0*pitch))&(((np.c_[x,y]-origin)@d)>0)
  for a,b in zip(x[valid][::3],y[valid][::3]):zones.append([float(a),float(b)])
 return zones

def trace_whole(data,free,occlusion):
 shape=data['surface'].shape;empty=np.zeros(shape,np.uint8)
 result={'status':'UNKNOWN','whole_mask':empty.copy(),'supported_mask':empty.copy(),'review_mask':empty.copy(),'candidates':[],
 'diagnostics':{'physical_contact':None,'same_material_identity_confirmed':False,'parameters':PARAMETERS.copy()}}
 if free['status']!='observed' or free['geometry']['status']!='observed':
  if free['status']=='free_width_ambiguous':result['status']='FREE_WIDTH_AMBIGUOUS'
  if free['status']=='observed':result.update({'status':'FREE_ONLY_DIRECTION_AMBIGUOUS','whole_mask':free['free_mask'].copy(),'supported_mask':free['free_mask'].copy()})
  result['diagnostics']['reason']='no_revalidated_unambiguous_free_seed';return result
 geo=free['geometry'];d=np.asarray(geo['direction_xy']);width=float(geo['width_px'])
 if abs(d[0])<.25:
  result['diagnostics']['reason']='near_vertical_free_rope_outside_x_monotonic_search';return result
 origin=np.asarray(geo['origin_xy']);end=np.asarray(geo['visible_end_xy']);sign=float(np.sign(d[0]));slope=float(d[1]/d[0]);step=PARAMETERS['step_x']
 zones=_tangent_zones(data['curve_refs'],origin,d,width,float(data['pitch']))
 zone_x=np.array([q[0] for q in zones]);zone_margin=2*max(width,float(data['pitch']))
 fields=_fields(data['original'],width)
 texture=dict(free['texture_descriptor']);texture['free_axis_angle_rad']=math.atan(slope)
 # 仅用显著组件决定是否已进入/离开卷面；不把原S中的碎片重新加入。
 # SAM原掩膜仍保留在data['surface']中，搜索域不会当成测量结果输出。
 surface,search_diagnostics=build_search_surface(data['surface'])
 result['diagnostics'].update(search_diagnostics)
 if not np.any(surface):
  result.update({'status':'FREE_ONLY','whole_mask':free['free_mask']|free['straight_observed_mask'],'supported_mask':free['free_mask']|free['straight_observed_mask']})
  result['diagnostics']['reason']='no_significant_surface_for_bounded_search';return result
 # 从直段末端稍前开始保留转弯可能，避免一次失测把起点推到旧圈上。
 start=end-d*min(width*1.5,max(0.,np.dot(end-origin,d)))
 entered=bool(surface[int(np.clip(start[1],0,shape[0]-1)),int(np.clip(start[0],0,shape[1]-1))])
 layers=[[{'x':float(start[0]),'y':float(start[1]),'slope':slope,'score':0.,'parent':0,'weak':0.,'supported':True,'width':width,'local_score':0.,'entered_surface':entered,'outside_distance':0.,'curved':False}]]
 initial_dist=float(np.dot(start-origin,d));maxsteps=int(min(2500.,width*PARAMETERS['maximum_search_widths'])/step)
 end_reason='bounded_search';h,w=shape;terminals=[]
 for k in range(1,maxsteps+1):
  old=layers[-1];xx=start[0]+sign*step*k
  if xx<width or xx>=w-width:end_reason='image_side';break
  parents=[];ys=[];ms=[];oldscore=[];weak=[];curved=[]
  for i,state in enumerate(old):
   angle=math.atan(state['slope'])
   can_bend=state['curved'] or (len(zone_x)>0 and np.min(abs(zone_x-xx))<zone_margin)
   # 同一可见圆柱侧的卷面投影为单向弯曲，禁止弯后反向去追邻圈亮纹。
   deltas=[0,sign,2*sign] if can_bend else [0]
   for delta in deltas:
    m=math.tan(np.clip(angle+delta*PARAMETERS['turn_radians'],-1.27,1.27))
    if abs(m)>PARAMETERS['maximum_tangent']:continue
    y=state['y']+sign*step*(m+state['slope'])/2
    if y<width or y>=h-width:continue
    parents.append(i);ys.append(y);ms.append(m);oldscore.append(state['score']-.025*delta*delta);weak.append(state['weak']);curved.append(state['curved'] or delta!=0)
  if not ys:end_reason='image_vertical_side';break
  centers=np.c_[np.full(len(ys),xx),ys];ms=np.asarray(ms);parents=np.asarray(parents)
  scores,support,widths,diag=_profiles(fields,centers,ms,width,sign,texture)
  inside=_sample((surface>0).astype(np.float32),centers)>.3
  # 在已进入卷面后，只允许短暂越出SAM边界，禁止延长到法兰/工具。
  entered=np.array([old[i]['entered_surface'] for i in parents])|inside
  distances=np.hypot(step,step*ms)
  outside=np.where(inside,0.,np.array([old[i]['outside_distance'] for i in parents])+distances)
  scores-=.45*((~inside)&entered)
  newweak=np.where(support,0.,np.asarray(weak)+distances)
  single=(~support)&diag['single_edge_support']&(abs(np.arctan(ms)-math.atan(slope))<.018)
  # 亮带常仅能看见斜绳的一侧阴影。只有保持原自由直线时才允许
  # 更长的单边推断；不重定位绳心、不改变宽度，也不涂成双边实测。
  budget=np.where(single,width*PARAMETERS['straight_single_edge_budget_width'],max(35.,width*PARAMETERS['maximum_weak_run_width']))
  valid=(newweak<budget)&(~entered|inside|(outside<width*.8))
  widths=np.where(single,width,widths)
  # 保存因失测或轮廓退出而终止的父路径，不能让另一条旧圈走得更长
  # 就把已经到达正确可见端点的路径从候选池中抹掉。
  alive=set(parents[valid].tolist())
  for parent_index,state in enumerate(old):
   if parent_index not in alive and len(layers)>5:
    reason='surface_visibility_end' if state['entered_surface'] and state.get('outside_distance',0)>0 else 'original_support_end'
    terminals.append((len(layers),parent_index,reason))
  cumulative=np.asarray(oldscore)+scores-.25
  # 以中心/斜率分桶保留不同走向；不能让同一条局部极值占满beam。
  order=np.argsort(-cumulative);seen=set();layer=[]
  for j in order:
   if not valid[j]:continue
   bucket=(int(round(ys[j]/max(2.,width*.18))),int(round(math.atan(ms[j])/.075)))
   if bucket in seen:continue
   seen.add(bucket);layer.append({'x':float(xx),'y':float(ys[j]),'slope':float(ms[j]),'score':float(cumulative[j]),'parent':int(parents[j]),'weak':float(newweak[j]),'supported':bool(support[j]),'width':float(widths[j]),'local_score':float(scores[j]),'inside_surface':bool(inside[j]),'entered_surface':bool(entered[j]),'outside_distance':float(outside[j]),'curved':curved[j]})
   if len(layer)>=PARAMETERS['beam']:break
  if not layer:end_reason='no_two_edge_image_support';break
  layers.append(layer)
 paths=[];last=layers[-1];unentered_rejections=[]
 terminals.extend((len(layers),i,end_reason) for i in range(len(last)))
 # 同一排序口径用于候选生成和最终选择；短而亮的上游段不冒充整根。
 terminals.sort(key=lambda v:-(layers[v[0]-1][v[1]]['score']/max(1,v[0])+.12*min(1,v[0]*step/(12*width))))
 for depth,i,terminal_reason in terminals[:400]:
  states=_reconstruct(layers[:depth],int(i));xy=np.array([[s['x'],s['y']] for s in states]);flags=[s['supported'] for s in states]
  # 末尾失去双边支持的探测点不绘为可见绳。
  good=np.flatnonzero(flags)
  if not len(good):continue
  n=int(good[-1])+1;xy=xy[:n];states=states[:n];flags=flags[:n]
  if len(xy)<3:continue
  if not _candidate_entered_surface(states):
   unentered_rejections.append({'endpoint_xy':xy[-1].tolist(),'saved_point_count':len(xy),'reason':'saved_candidate_never_entered_significant_search_surface'})
   continue
  # 在共同可见段的末1/3比较，不让共同的长自由直段吞并分叉邻圈。
  duplicate=False
  for p in paths:
   ncommon=min(len(xy),len(p['xy']));lo=max(0,int(ncommon*.66))
   if abs(len(xy)-len(p['xy']))*step<2*width and np.mean(np.linalg.norm(xy[lo:ncommon]-np.asarray(p['xy'])[lo:ncommon],axis=1))<width*.55:duplicate=True;break
  if duplicate:continue
  bonus,row=_row_bonus(xy,occlusion,width,origin,d)
  score=float(np.mean([s['local_score'] for s in states]))+PARAMETERS['curve_weight']*bonus
  score+=.12*min(1.,len(xy)*step/(12*width))
  paths.append({'xy':xy.tolist(),'widths':[s['width'] for s in states],'supported':flags,'score':score,'image_score':float(np.mean([s['local_score'] for s in states])),'row_bonus':bonus,'row_candidate':row,'observed_fraction':float(np.mean(flags)),'length_px':float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),'end_reason':terminal_reason})
  if len(paths)>=12:break
 paths.sort(key=lambda p:-p['score'])
 result['diagnostics']['rejected_no_surface_entry_count']=len(unentered_rejections)
 result['diagnostics']['rejected_no_surface_entry_paths']=unentered_rejections
 for i,p in enumerate(paths):p['candidate_id']=f'path_{i:02d}'
 if not paths:
  result.update({'status':'FREE_ONLY','whole_mask':free['free_mask']|free['straight_observed_mask'],'supported_mask':free['free_mask']|free['straight_observed_mask']})
  result['diagnostics']['reason']='no_candidate_reaches_search_surface' if unentered_rejections else 'no_curved_candidate';return result
 chosen=paths[0];ambiguity=len(paths)>1 and chosen['score']-paths[1]['score']<.07
 if chosen['observed_fraction']<PARAMETERS['minimum_supported_fraction']:
  result.update({'status':'FREE_ONLY_CURVE_LOW_SUPPORT','whole_mask':free['free_mask']|free['straight_observed_mask'],'supported_mask':free['free_mask']|free['straight_observed_mask'],'candidates':paths})
  result['diagnostics']['reason']='curved_candidate_below_frozen_observed_fraction';return result
 # 输出单条最高外观分候选，绝不把多圈候选并成“整根”掩膜。
 pxy=np.asarray(chosen['xy']);ends=np.asarray(free['free_segment']['observed_segment_xy']);outer=ends[np.argmin(ends@d)]
 widths=np.asarray(chosen['widths'],np.float32)
 widths=np.array([np.median(widths[max(0,i-3):i+4]) for i in range(len(widths))],np.float32)
 chosen['widths']=cv2.GaussianBlur(widths.reshape(-1,1),(1,0),1.2).ravel().tolist()
 prefix=np.linspace(outer,start,max(2,int(np.linalg.norm(outer-start)/step)))
 xy=np.vstack([prefix,pxy]);ww=[width]*len(prefix)+chosen['widths']
 whole=_paint(shape,xy,ww)
 observed=free['free_mask']|free['straight_observed_mask']|_paint(shape,pxy,chosen['widths'],chosen['supported'])
 observed&=whole
 # 原图双边支持只是“这里像绳”，还不能确认暗尾接的是同根绳。
 # 未获独立断界证据的弯段、接近分数的竞争尾段均另标橙色待复核。
 normal=np.array([-d[1],d[0]])
 bent=abs((pxy-origin)@normal)>.5*width
 row=next((c for c in occlusion.get('candidates',[]) if c['candidate_id']==chosen.get('row_candidate')),None)
 # 一条沟槽连续，甚至一组“中断后恢复”，都不能独立证明这是同一根绳。
 # 本版将弯段始终另标为待复核；原图双边支持仍保存在supported_mask中。
 raw_transition_chain=bool(row and row.get('interruption_then_continuity_chain',False))
 result['diagnostics']['selected_raw_transition_chain']=raw_transition_chain
 review_flags=bent.copy()
 for alternative in paths[1:]:
  if chosen['score']-alternative['score']>=.07:continue
  q=np.asarray(alternative['xy']);q=q[np.argsort(q[:,0])]
  valid=(pxy[:,0]>=q[0,0])&(pxy[:,0]<=q[-1,0]);delta=abs(pxy[:,1]-np.interp(pxy[:,0],q[:,0],q[:,1]))
  review_flags|=valid&(delta>.4*width)
 review=_paint(shape,pxy,chosen['widths'],review_flags)|(whole&~observed)
 review&=whole
 end_reason=chosen['end_reason']
 reached_side=end_reason in ('left_saved_surface_visibility','image_side','image_vertical_side','surface_visibility_end')
 normal=np.array([-d[1],d[0]])
 bend_present=(float(np.max(abs((pxy-origin)@normal)))>width and chosen['length_px']>5*width)
 reached_side=reached_side and bend_present
 status=('AMBIGUOUS_' if ambiguity else '')+('WHOLE_APPEARANCE_CANDIDATE' if reached_side else 'PARTIAL_APPEARANCE_CANDIDATE')
 result.update({'status':status,'whole_mask':whole,'supported_mask':observed,'review_mask':review,'candidates':paths,'selected_candidate_id':chosen['candidate_id']})
 result['diagnostics'].update({'reason':'finite_original_image_ribbon_search','beam_layers':len(layers),'start_xy':start.tolist(),'free_straight_distance_px':initial_dist,'stop_reason':end_reason,'candidate_count':len(paths),'tangency_zones':zones,'curvature_sign':'convex_image_y_of_x','bend_present':bool(bend_present),'visible_endpoint_basis':'saved_surface_or_frame_limit_not_verified_physical_disappearance','score_margin':float(chosen['score']-paths[1]['score']) if len(paths)>1 else None,'whole_pixels':int(np.count_nonzero(whole)),'supported_pixels':int(np.count_nonzero(observed)),'interpretation':'selected 2D appearance route for review; competing old coils may remain; no invisible tail'})
 return result
