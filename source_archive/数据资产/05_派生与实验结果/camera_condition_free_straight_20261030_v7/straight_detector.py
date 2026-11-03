"""A7：只测未卷入的直线活动绳；不调用弯段、F3轨道或物理切点判断。

 第一步用卷面外的真实双边确定一条无向直轴；第二步沿这条轴检查整幅
 原图。图像在卷面上重叠不等于已卷入，遮挡也不等于绳在此终止。
 因此允许同轴上多个可见段，但只画有双边证据的像素，不填遮挡。
 """
import math
import cv2
import numpy as np
import free_rope as legacy
from free_seed_recovery import recover_free_seeds
from strand_signature import measure_strand_signature,_one_window

PARAMETERS={'step_px':4.,'profile_half_length_widths':1.6,'profile_normal_radius_widths':1.5,
 'minimum_contrast':3.5,'minimum_edge_gradient':.8,'maximum_center_shift_widths':.32,
 'width_min_ratio':.65,'width_max_ratio':1.45,'minimum_anchor_profiles':14,
 'minimum_anchor_span_px':100.,'maximum_fit_angle_change_deg':2.5,
 'minimum_visible_run_widths':2.5,'maximum_measured_gap_widths':.65,
 'maximum_extension_without_support_widths':10.,'minimum_unique_score_margin':.13,
 'maximum_candidates':8,'minimum_axis_span_widths':8.,
 'minimum_signature_repeat':.22,'minimum_nonperiodic_span_widths':30.,
 'near_border_widths':3.,'minimum_saturated_fraction':.60,
 'fragment_join_angle_deg':.8,'fragment_join_offset_widths':.9,'fragment_join_period_ratio':1.65,
 'local_repeat_minimum_correlation':.40,'local_period_ratio_min':.70,'local_period_ratio_max':1.40,
 'local_texture_angle_tolerance_deg':18.,'reacquisition_minimum_repeat_fraction':.60}

def _sample(a,xy):
 p=np.asarray(xy,np.float32);shape=p.shape[:-1]
 return cv2.remap(a,p[...,0].reshape(-1,p.shape[-2]),p[...,1].reshape(-1,p.shape[-2]),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).reshape(shape)

def _runs(flag):
 z=np.diff(np.r_[False,flag,False].astype(np.int8))
 return list(zip(np.flatnonzero(z==1),np.flatnonzero(z==-1)))

def _fields(image):
 b,g,r=cv2.split(image.astype(np.float32));lo=np.minimum(np.minimum(b,g),r);hi=np.maximum(np.maximum(b,g),r)
 ach=lo*255/np.maximum(hi,1);ach[hi<15]=0
 return {'brightness':lo,'achromatic':ach,'yellow':((r>b*1.35+8)&(g>b*1.25+8)&(hi>40)).astype(np.float32),'gray':cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)}

def _axis_points(origin,direction,shape,width):
 h,w=shape;o=np.asarray(origin);d=np.asarray(direction);margin=max(3.,2.25*width)
 lo,hi=-1.e6,1.e6
 for value,slope,limit in zip(o,d,[w,h]):
  if abs(slope)<1e-8:
   if not margin<=value<=limit-1-margin:return np.empty((0,2)),np.empty(0)
  else:
   a,b=sorted([(margin-value)/slope,(limit-1-margin-value)/slope]);lo=max(lo,a);hi=min(hi,b)
 if hi<=lo:return np.empty((0,2)),np.empty(0)
 s=np.arange(lo,hi+.1,PARAMETERS['step_px']);return o+s[:,None]*d,s

def _measure(fields,points,direction,width):
 """每个截面同时寻找相反方向的两条边，而非找一根最亮的股纹。

 背景是黄色时使用无彩色比例，其余使用原图亮度。亮绳、暗绳分别测量，
 极性可以变化；两边的间距和中点必须保持与已确定直轴一致。
 """
 d=np.asarray(direction);n=np.array([-d[1],d[0]]);ns=np.linspace(-1.5*width,1.5*width,121);ts=np.linspace(-PARAMETERS['profile_half_length_widths']*width,PARAMETERS['profile_half_length_widths']*width,17)
 xy=points[:,None,None,:]+ts[None,:,None,None]*d+ns[None,None,:,None]*n
 ys=_sample(fields['yellow'],xy[:,:,np.abs(ns)>width]);yellow=ys.mean(axis=(1,2))>.55
 a=np.median(_sample(fields['brightness'],xy),axis=1);b=np.median(_sample(fields['achromatic'],xy),axis=1)
 profile=np.where(yellow[:,None],b,a).astype(np.float32);profile=cv2.GaussianBlur(profile,(5,1),.8)
 delta=ns[1]-ns[0];grad=np.gradient(profile,delta,axis=1)
 li=np.flatnonzero((ns>=-.85*width)&(ns<=-.22*width));ri=np.flatnonzero((ns>=.22*width)&(ns<=.85*width))
 body=np.median(profile[:,np.abs(ns)<.22*width],axis=1)
 lb=np.median(profile[:,ns<-1.05*width],axis=1);rb=np.median(profile[:,ns>1.05*width],axis=1)
 noise=1.4826*np.median(np.abs(grad[:,np.abs(ns)>width]),axis=1)
 options=[];row=np.arange(len(points))
 for polarity in [1.,-1.]:
  l=li[np.argmax(polarity*grad[:,li],axis=1)];r=ri[np.argmax(-polarity*grad[:,ri],axis=1)]
  offset=(ns[l]+ns[r])/2;ww=ns[r]-ns[l];strength=np.minimum(polarity*grad[row,l],-polarity*grad[row,r]);contrast=np.minimum(polarity*(body-lb),polarity*(body-rb))
  valid=(ww>=PARAMETERS['width_min_ratio']*width)&(ww<=PARAMETERS['width_max_ratio']*width)&(np.abs(offset)<=PARAMETERS['maximum_center_shift_widths']*width)
  valid&=(strength>=np.maximum(PARAMETERS['minimum_edge_gradient'],.6*noise))&(contrast>=np.maximum(PARAMETERS['minimum_contrast'],1.3*noise))
  quality=np.minimum(strength/8,1)*.35+np.minimum(contrast/25,1)*.45+.20*np.maximum(0,1-abs(ww/width-1))
  quality=np.where(valid,quality,-1.)
  options.append((quality,valid,ns[l],ns[r],strength,contrast))
 select=options[1][0]>options[0][0]
 outputs=[np.where(select,options[1][i],options[0][i]) for i in range(6)]
 # 绳的一侧可以是亮卷面，另一侧可以是暗阴影；这时两个外沿的梯度未必
 # 相反。这里只提出混合背景双边，调用方还必须验证本绳的局部重复股纹。
 l=li[np.argmax(abs(grad[:,li]),axis=1)];r=ri[np.argmax(abs(grad[:,ri]),axis=1)]
 mixed_width=ns[r]-ns[l];mixed_offset=(ns[r]+ns[l])/2
 strength=np.minimum(abs(grad[row,l]),abs(grad[row,r]));contrast=np.maximum(abs(body-lb),abs(body-rb))
 mixed=(mixed_width>=PARAMETERS['width_min_ratio']*width)&(mixed_width<=PARAMETERS['width_max_ratio']*width)&(abs(mixed_offset)<=.27*width)
 mixed&=(strength>=np.maximum(1.2,.8*noise))&(contrast>=np.maximum(5.,1.5*noise))
 return dict(zip(['quality','supported','left_offset','right_offset','edge_strength','contrast'],outputs))|{'polarity':np.where(select,-1,1),'yellow_channel':yellow,'mixed_supported':mixed,'mixed_left':ns[l],'mixed_right':ns[r]}

def _local_repeat_support(gray,points,d,width,signature):
 """直轴上局部原图股纹核验；不是把直线坐标本身当观测。

 每个窗口独立测量周期和纹路方向，仅给窗口中部点背书，避免拿远处
 可见纹理替遮挡处背书。没有本绳参考周期就不启用这种补充证据。
 """
 support=np.zeros(len(points),bool);period=signature.get('period_px');windows=signature.get('windows',[])
 angles=[w['texture_orientation_from_axis_deg'] for w in windows if w.get('periodic') and w.get('oblique')]
 if period is None or not angles:return support,[]
 vv=np.mean(np.exp(2j*np.radians(angles)));reference=float(np.degrees(np.angle(vv))/2%180)
 n=np.array([-d[1],d[0]]);ns=np.arange(-.30*width,.30*width+.1,.5)
 patch=_sample(gray,points[:,None,:]+ns[None,:,None]*n);step=PARAMETERS['step_px'];length=min(len(points),max(32,int(12*width/step)));stride=max(1,int(4*width/step));reports=[]
 for a in range(0,max(1,len(points)-length+1),stride):
  b=min(len(points),a+length);r=_one_window(patch[a:b],width,step,.5);p=r.get('period_px');angle=r.get('texture_orientation_from_axis_deg')
  mismatch=abs((angle-reference+90)%180-90) if angle is not None else 90.
  accepted=bool(r['periodic'] and r['oblique'] and r['peak_correlation']>=PARAMETERS['local_repeat_minimum_correlation'] and p is not None and PARAMETERS['local_period_ratio_min']<=p/period<=PARAMETERS['local_period_ratio_max'] and mismatch<=PARAMETERS['local_texture_angle_tolerance_deg'])
  if accepted:support[a+length//4:b-length//4]=True
  reports.append({'start_index':a,'end_index':b,'period_px':p,'correlation':r['peak_correlation'],'angle_difference_deg':mismatch,'accepted':accepted})
 return support,reports

def _outside(surface,p,d,m):
 n=np.array([-d[1],d[0]]);left=p+m['left_offset'][:,None]*n;right=p+m['right_offset'][:,None]*n
 return (_sample(surface,p[:,None,:]).ravel()==0)&(_sample(surface,left[:,None,:]).ravel()==0)&(_sample(surface,right[:,None,:]).ravel()==0)

def _robust_axis(points,s,m,anchor,width,o,d):
 ids=np.flatnonzero(anchor);offset=(m['left_offset']+m['right_offset'])/2
 if len(ids)<PARAMETERS['minimum_anchor_profiles'] or np.ptp(s[ids])<PARAMETERS['minimum_anchor_span_px']:return None
 keep=np.ones(len(ids),bool);coef=np.zeros(2)
 for _ in range(4):
  coef=np.polyfit(s[ids][keep],offset[ids][keep],1);res=offset[ids]-np.polyval(coef,s[ids]);scale=max(1.,1.4826*np.median(np.abs(res[keep]-np.median(res[keep]))))
  keep=abs(res)<=max(1.5,.18*width,2.5*scale)
  if keep.sum()<PARAMETERS['minimum_anchor_profiles']:return None
 if abs(math.degrees(math.atan(coef[0])))>PARAMETERS['maximum_fit_angle_change_deg']:return None
 ww=m['right_offset'][ids[keep]]-m['left_offset'][ids[keep]];w=float(np.median(ww));n=np.array([-d[1],d[0]])
 newd=d+coef[0]*n;newd/=np.linalg.norm(newd);newo=o+coef[1]*n
 return newo,newd,w,{'anchor_count':int(keep.sum()),'span_px':float(np.ptp(s[ids[keep]])),'residual_p90_px':float(np.percentile(abs(res[keep]),90)),'width_p10_p90':np.percentile(ww,[10,90]).tolist()}

def _close_small_gaps(flag,maxgap):
 out=flag.copy()
 for a,b in _runs(~flag):
  if a>0 and b<len(flag) and b-a<=maxgap:out[a:b]=True
 return out

def _paint(shape,left,right,visible):
 mask=np.zeros(shape,np.uint8)
 for i in range(1,len(left)):
  if visible[i-1] and visible[i]:cv2.fillConvexPoly(mask,np.rint([left[i-1],right[i-1],right[i],left[i]]).astype(np.int32),255)
 return mask

def _candidate(data,fields,seed,source):
 segment=np.asarray(seed['observed_segment_xy'],float);d=segment[1]-segment[0];d/=max(np.linalg.norm(d),1e-8)
 if d[0]<0:d=-d
 o=segment.mean(axis=0);width=float(seed['width_px']);pitch=float(data['pitch']);detail={}
 if source.startswith('saved'):
  measured,diag=legacy._remeasure(data['original'],data['surface'],seed)
  measured,wide=legacy._wide_remeasure(data['original'],data['surface'],seed,measured,pitch)
  detail={'old_seed_raw_remeasurement':diag,'wide_envelope':wide}
  if measured is not None:
   segment=np.asarray(measured['observed_segment_xy']);o=segment.mean(axis=0);d=np.asarray(measured['direction_xy']);width=float(measured['width_px'])
 if not 3<=width<=2*pitch:return None,{'reason':'width_outside_search_scale','width':width}
 points,s=_axis_points(o,d,data['surface'].shape,width)
 if len(points)<20:return None,{'reason':'short_axis_in_image'}
 m=_measure(fields,points,d,width);outside=_outside(data['surface'],points,d,m)
 seed_s=(segment-o)@d;anchor=m['supported']&outside&(s>=seed_s.min()-2*width)&(s<=seed_s.max()+2*width)
 fit=_robust_axis(points,s,m,anchor,width,o,d)
 if fit is None:return None,{'reason':'outside_axis_not_supported','supported_profiles':int(anchor.sum()),**detail}
 o,d,width,fitdiag=fit;points,s=_axis_points(o,d,data['surface'].shape,width);m=_measure(fields,points,d,width);outside=_outside(data['surface'],points,d,m)
 # 截面只参与是否可见和宽度核验。显示用共同直轴和稳健宽度，消除股纹
 # 锯齿导致的每几像素抖动，不让单点把整条绳拉向背景。
 raw=m['supported'].copy();offset=(m['left_offset']+m['right_offset'])/2
 raw&=abs(offset)<=max(1.5,.27*width)
 seed_s=(segment-o)@d;anchor=raw&outside&(s>=seed_s.min()-2*width)&(s<=seed_s.max()+2*width)
 if anchor.sum()<14:return None,{'reason':'refit_lost_external_edges',**detail}
 ids=np.flatnonzero(anchor);anchor_seg=np.array([points[ids[0]],points[ids[-1]]]);signature=measure_strand_signature(fields['gray'],anchor_seg,width)
 local_repeat,local_reports=_local_repeat_support(fields['gray'],points,d,width,signature)
 added=m['mixed_supported']&local_repeat&~raw
 raw|=added
 # 只连接不足一根绳宽的截面采样缺口，不跨黑杆或长暗区填实。
 short=_close_small_gaps(raw,max(1,int(.65*width/PARAMETERS['step_px'])))
 visible=np.zeros(len(points),bool);pieces=[]
 for a,b in _runs(short):
  if s[b-1]-s[a]<PARAMETERS['minimum_visible_run_widths']*width:continue
  visible[a:b]=True;pieces.append([int(a),int(b)])
 # 从实测锚段向两边检查，长失证段后的其它直边不自动归为同一根绳。
 # 允许有界遮挡前后共线的可见段；失证像素本身仍然不涂实。
 anchor_ids=np.flatnonzero(anchor);lo=int(anchor_ids[0]);hi=int(anchor_ids[-1])
 measured_visible=visible.copy()
 for direction,start in [(-1,lo),(1,hi)]:
  gap=0.
  for k in range(start, len(points) if direction>0 else -1,direction):
   gap=0. if visible[k] else gap+PARAMETERS['step_px']
   if gap>PARAMETERS['maximum_extension_without_support_widths']*width:
    if direction>0:visible[k:]=False
    else:visible[:k+1]=False
    break
 reacquired=[]
 for a,b in _runs(measured_visible&~visible):
  span=s[b-1]-s[a];fraction=float(np.mean(local_repeat[a:b]))
  if span>=max(100.,4*width) and fraction>=PARAMETERS['reacquisition_minimum_repeat_fraction']:
   visible[a:b]=True;reacquired.append({'start_index':int(a),'end_index':int(b),'local_repeat_fraction':fraction,'span_px':float(span),'gap_filled':False})
 pieces=[[int(a),int(b)] for a,b in _runs(visible)]
 # 股纹签名仅在原图卷面外的实测锚段上建立，不从拟合路径生成标签。
 anchor_length=float(s[ids[-1]]-s[ids[0]])
 n=np.array([-d[1],d[0]]);left=points-.5*width*n;right=points+.5*width*n
 # 分支选择暂用可解释的外部长度/双边质量。纹理判据由独立测量字段加入，
 # 缺少纹理明确保留unknown，不把它当不存在。
 repeat=float(signature.get('consistent_window_fraction',0.)*max(0.,signature.get('median_peak_correlation',0.)))
 texture_status=signature.get('status','unknown')
 border_distance=min(float(np.min([anchor_seg[:,0].min(),anchor_seg[:,1].min(),data['surface'].shape[1]-1-anchor_seg[:,0].max(),data['surface'].shape[0]-1-anchor_seg[:,1].max()])),1.e6)
 quality=float(np.mean(m['quality'][anchor]));score=.40*min(1,anchor_length/(30*width))+.30*quality+.30*repeat
 observed_length=float(np.sum(visible)*PARAMETERS['step_px'])
 periodic=texture_status in ('periodic_strand_evidence','periodic_texture_orientation_uncertain') and repeat>=PARAMETERS['minimum_signature_repeat']
 long_envelope=anchor_length>=PARAMETERS['minimum_nonperiodic_span_widths']*width
 clipped_border=border_distance<PARAMETERS['near_border_widths']*width and signature.get('saturated_fraction',0)>=PARAMETERS['minimum_saturated_fraction']
 plausible=anchor_length>=max(100.,8*width) and visible.any() and (periodic or long_envelope or clipped_border)
 c={'candidate_id':source,'accepted':False,'center_xy':points.tolist(),'left_xy':left.tolist(),'right_xy':right.tolist(),'visible':visible.tolist(),'raw_supported':raw.tolist(),'mixed_background_supported':added.tolist(),'local_repeat_supported':local_repeat.tolist(),'local_repeat_windows':local_reports,'reacquired_pieces':reacquired,'outside':outside.tolist(),'width_px':width,'direction_xy':d.tolist(),'origin_xy':o.tolist(),'anchor_segment_xy':anchor_seg.tolist(),'anchor_measurements_xy':(points[anchor]+offset[anchor,None]*n).tolist(),'anchor_measured_widths':(m['right_offset'][anchor]-m['left_offset'][anchor]).tolist(),'anchor_span_px':anchor_length,'observed_straight_length_px':observed_length,'score':score,'plausible':bool(plausible),'reason':'straight_axis_measured_no_contact_claim' if plausible else 'straight_object_without_sufficient_rope_evidence','acceptance_evidence':{'periodic_interior':periodic,'long_measured_envelope':long_envelope,'saturated_near_frame_border':clipped_border},'pieces':pieces,'signature':signature,'outside_edge_quality':quality,'anchor_border_distance_px':border_distance,'fit':fitdiag,'diagnostics':detail,'surface_overlap_not_contact':True,'physical_contact':None,'seed_source':source}
 return c,None

def _same_axis(a,b):
 d=np.array(a['direction_xy']);db=np.array(b['direction_xy']);n=np.array([-d[1],d[0]])
 if abs(d@db)<math.cos(math.radians(.6)):return False
 ends=np.asarray(a['anchor_segment_xy']);q=np.asarray(b['origin_xy']);nb=np.array([-db[1],db[0]])
 return max(abs((ends-q)@nb))<.45*max(a['width_px'],b['width_px'])

def _join_fragments(data,fields,candidates):
 """遮挡两侧的同轴观测共同重拟合，而非误作两条独立绳。

 只合并方向、外部轴位置和可用股纹周期均兼容的片段，合并后重新测
 原图双边；不直接平均两条旧线，也不填中间的遮挡。
 """
 groups=[]
 for c in sorted(candidates,key=lambda v:-v['anchor_span_px']):
  for group in groups:
   reference=group[0];d=np.array(reference['direction_xy']);n=np.array([-d[1],d[0]])
   angle=abs(d@np.array(c['direction_xy']))
   points=np.asarray(c['anchor_segment_xy']);distance=max(abs((points-np.array(reference['origin_xy']))@n))
   p,q=c['signature'].get('period_px'),reference['signature'].get('period_px')
   period_ok=p is None or q is None or max(p,q)/min(p,q)<=PARAMETERS['fragment_join_period_ratio']
   if angle>=math.cos(math.radians(PARAMETERS['fragment_join_angle_deg'])) and distance<=PARAMETERS['fragment_join_offset_widths']*max(c['width_px'],reference['width_px']) and period_ok:
    group.append(c);break
  else:groups.append([c])
 joined=[];ledger=[]
 for group in groups:
  if len(group)==1:joined+=group;continue
  points=np.vstack([c['anchor_measurements_xy'] for c in group]);widths=np.concatenate([c['anchor_measured_widths'] for c in group]);reference=group[0]
  o=points.mean(axis=0);d=np.asarray(reference['direction_xy']);n=np.array([-d[1],d[0]]);s=(points-o)@d;y=(points-o)@n;keep=np.ones(len(points),bool)
  for _ in range(4):
   coef=np.polyfit(s[keep],y[keep],1);res=y-np.polyval(coef,s);scale=max(1.,1.4826*np.median(abs(res[keep]-np.median(res[keep]))));keep=abs(res)<=max(2.,2.5*scale)
  axis=d+coef[0]*n;axis/=np.linalg.norm(axis);center=o+coef[1]*n
  ends=center+np.array([s[keep].min(),s[keep].max()])[:,None]*axis
  seed={'status':'observed','observed_segment_xy':ends.tolist(),'width_px':float(np.median(widths[keep])),'coordinate_system':'original_unmirrored_image_xy_pixels'}
  identifier='joint_'+'_'.join(c['candidate_id'] for c in group)
  new,failure=_candidate(data,fields,seed,identifier)
  ledger.append({'sources':[c['candidate_id'] for c in group],'raw_anchor_count':len(points),'joint_fit_inliers':int(keep.sum()),'failure':failure})
  joined.extend([new] if new is not None else group)
 return joined,ledger

def detect(data):
 fields=_fields(data['original']);seeds=[];attempts=[]
 for key in ['a2_record','a1_record']:
  s=(data.get(key) or {}).get('free_segment',{})
  if s.get('status')=='observed' and s.get('coordinate_system')=='original_unmirrored_image_xy_pixels':
   seeds.append(('saved_'+key,s));break
 recovered=recover_free_seeds(data)
 for c in recovered.get('candidates',[]):seeds.append((c['candidate_id'],c['seed']))
 candidates=[]
 for source,seed in seeds[:PARAMETERS['maximum_candidates']]:
  c,failure=_candidate(data,fields,seed,source)
  attempts.append({'source':source,'failure':failure,'accepted_for_review':c is not None})
  if c is not None:candidates.append(c)
 candidates,join_ledger=_join_fragments(data,fields,candidates)
 candidates.sort(key=lambda c:-c['score']);unique=[]
 for c in candidates:
  if any(_same_axis(c,old) for old in unique):continue
  unique.append(c)
 candidates=unique;eligible=[c for c in candidates if c['plausible']];mask=np.zeros(data['surface'].shape,np.uint8)
 status='AMBIGUOUS' if candidates else 'UNKNOWN'
 if eligible:
  best=eligible[0];close=[c for c in eligible if best['score']-c['score']<PARAMETERS['minimum_unique_score_margin']]
  if len(close)==1:
   best['accepted']=True;status='DETECTED';mask=_paint(mask.shape,np.asarray(best['left_xy']),np.asarray(best['right_xy']),best['visible'])
  else:status='AMBIGUOUS'
 return {'status':status,'mask':mask,'candidates':candidates,'diagnostics':{'version':'IR-A7.0.0','parameters':PARAMETERS,'attempts':attempts,'fragment_join':join_ledger,'raw_recovery':recovered['diagnostics'],'free_line_does_not_require_winding_direction':True,'F3_used':False,'bends_reconstructed':False,'contact_point_claim':False,'labels_used':False}}
