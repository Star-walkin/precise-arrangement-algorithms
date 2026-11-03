"""A5：先比较原图中的整绳条带，再在有限F3轨道中选择接续。

未贴卷面的钢丝绳保持原自由直线。贴合段的平均灰度可能与邻圈相近，
因此分阶段评分，不能要求整条路径永远比背景更亮或更暗。
"""
import cv2,json,hashlib
import numpy as np
from photometric_ribbon import build_fields,compare_ribbon_neighbors
from lane_candidates import generate_lane_candidates
from attached_groove import sample_attached_grooves

PARAMETERS={'ray_step_px':6.,'maximum_ray_px':2800.,'weak_run_widths':4.,
 'minimum_path_span_widths':3.,'minimum_path_image_fraction':.25,
 'unique_straight_advantage':.08,'strong_straight_score':.50,
 'premature_turn_offset_width':.60,'ambiguity_margin':.065,
 'minimum_candidate_score':.30,'max_unsupported_lane_widths':5.,
 'minimum_attached_support_fraction':.15,'maximum_premature_turn_fraction':.15}

def search_surface(surface):
 """显著组件包络仅限制搜索，绝不当成新的绳体或卷面真值。"""
 contours,_=cv2.findContours(surface,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 size=np.count_nonzero(surface);cs=[c for c in contours if cv2.contourArea(c)>=max(50.,.0001*size)]
 significant=np.zeros_like(surface)
 if not cs:return significant
 cv2.drawContours(significant,cs,-1,255,-1)
 envelope=np.zeros_like(surface);cv2.fillConvexPoly(envelope,cv2.convexHull(np.vstack(cs)),255)
 return envelope if np.count_nonzero(envelope)<=1.8*max(1,size) else significant

def sample_image(a,xy):
 p=np.asarray(xy,np.float32)
 return cv2.remap(a,p[:,0][None,:],p[:,1][None,:],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0).ravel()

def paint(shape,xy,widths,flags=None):
 out=np.zeros(shape,np.uint8);p=np.asarray(xy,float);ws=np.broadcast_to(widths,(len(p),))
 for i in range(1,len(p)):
  if flags is not None and not(flags[i-1] and flags[i]):continue
  d=p[i]-p[i-1];norm=np.linalg.norm(d)
  if norm<1e-6:continue
  n=np.array([-d[1],d[0]])/norm
  poly=np.array([p[i-1]+n*ws[i-1]/2,p[i]+n*ws[i]/2,p[i]-n*ws[i]/2,p[i-1]-n*ws[i-1]/2])
  cv2.fillConvexPoly(out,np.rint(poly).astype(np.int32),255)
 return out

def _tangents(xy):
 t=np.gradient(xy,axis=0);return t/np.maximum(np.linalg.norm(t,axis=1)[:,None],1e-6)

def _mean(v,take,default=0.):
 return float(np.mean(v[take])) if np.any(take) else float(default)

def _measure(fields,xy,ww,texture):
 return compare_ribbon_neighbors(fields,xy,_tangents(xy),ww,pitch=fields['pitch_px']*1.1,free_texture=texture)

def _trim(candidate,n):
 """裁短时所有逐点台账同步裁短，避免用已删除的弱探测点作证据。"""
 old=len(candidate['xy'])
 for k,v in list(candidate.items()):
  if isinstance(v,list) and len(v)==old and k not in ('source_boundary_ids','source_ref_ids'):candidate[k]=v[:n]
 candidate['join_index']=min(candidate.get('join_index',n),n)
 candidate['transition_start_index']=min(candidate.get('transition_start_index',n),n)
 return candidate

def _straight_candidate(fields,domain,free,texture):
 g=free['geometry'];o=np.asarray(g['origin_xy']);d=np.asarray(g['direction_xy']);width=float(g['width_px'])
 ends=np.asarray(free['free_segment']['observed_segment_xy']);outer=ends[np.argmin(ends@d)]
 h,w=domain.shape;s=np.arange(0,PARAMETERS['maximum_ray_px'],PARAMETERS['ray_step_px']);p=o+s[:,None]*d
 valid=(p[:,0]>width)&(p[:,0]<w-width)&(p[:,1]>width)&(p[:,1]<h-width)
 p=p[:np.flatnonzero(~valid)[0]] if np.any(~valid) else p
 if len(p)<3:return None,{'reason':'ray_outside_frame'}
 inside=sample_image(domain,p)>.3*255;ids=np.flatnonzero(inside)
 if not len(ids):return None,{'reason':'straight_ray_never_reaches_search_surface'}
 first=int(ids[0]);end=len(p);outside_run=0.
 for i in range(first,len(p)):
  outside_run=0. if inside[i] else outside_run+PARAMETERS['ray_step_px']
  if outside_run>width*.8:end=i;break
 p=p[:end];m=_measure(fields,p,np.full(len(p),width),texture)
 # 强均值差和真实双边来自原像素；允许极性变换，不累计亮/暗模板投票。
 usable=m['supported']|m['saturated_outline_supported']|(m['single_edge_support']&(m['normalized_body_contrast']>.5)&(m['along_consistency']>.8))
 weak=0.;last=-1;stop='search_surface_limit'
 for i in range(len(p)):
  if usable[i]:weak=0.;last=i
  else:weak+=PARAMETERS['ray_step_px']
  if i>=first and weak>width*PARAMETERS['weak_run_widths']:stop='photometric_support_end';break
 if last<first or last*PARAMETERS['ray_step_px']<2*width:return None,{'reason':'no_supported_straight_overlap','first_entry_sample':first}
 p=p[:last+1];prefix=np.linspace(outer,o,max(2,int(np.linalg.norm(o-outer)/6)+1))[:-1];xy=np.vstack([prefix,p]);n=len(xy)
 return {'candidate_id':'straight_photo','lane_id':None,'xy':xy.tolist(),'widths':[width]*n,'phase':['free_straight']*n,'point_inferred':[False]*n,'join_index':n,'transition_start_index':n,'geometry_cost':0.,'terminal_reason':stop,'join_x':None,'transition_start_x':None}, {'reason':'original_axis_differential_trace','sample_count':len(p),'first_entry_sample':first,'stop':stop}

def _score(candidate,fields,domain,free,texture):
 c=dict(candidate);xy=np.asarray(c['xy'],float);ww=np.asarray(c['widths'],float)
 g=free['geometry'];o=np.asarray(g['origin_xy']);d=np.asarray(g['direction_xy']);width=float(g['width_px']);normal=np.array([-d[1],d[0]])
 m=_measure(fields,xy,ww,texture);phase=np.array(c['phase']);inside=sample_image(domain,xy)>255*.3
 progress=(xy-o)@d;forward=progress>=0;lane=np.isin(phase,['lane','lane_extrapolation'])
 # 正常贴合绳的均值可能等于背景。这里用原灰度中的双边幅值，
 # 而不是强迫其继续维持未贴合段的负/正均值差。
 left=m['left_edge_contrast_raw']*m['polarity'];right=m['right_edge_contrast_raw']*m['polarity']
 raw_pair=np.minimum(abs(left),abs(right));pair_ok=(left*right>0)&(raw_pair>=1.)&m['reliable']
 pair_score=np.clip(raw_pair/np.maximum(3.,m['normalization_scale_raw']),0,1)*(left*right>0)*m['reliable']
 formal=np.zeros(len(xy),bool);groove_score=np.zeros(len(xy));groove_supported=np.zeros(len(xy),bool);groove_available=np.zeros(len(xy),bool)
 if c.get('formal_boundary_distance_px') is not None:
  distance=np.asarray(c['formal_boundary_distance_px']);formal=np.max(distance,axis=1)<=3.
  gm=sample_attached_grooves(fields,np.asarray(c['formal_boundary_xy']),_tangents(xy),ww)
  groove_score=gm['score'];groove_supported=gm['supported'];groove_available=gm['available']
 # 贴合卷面两根绳之间是凹谷，不是自由绳外缘的灰度台阶。必须切换测量模型。
 supported=m['supported']|m['saturated_outline_supported']|(lane&groove_supported&formal)
 # 饱和内灰度不用于背景差分；但两条真实外轮廓仍可提供独立形状支持。
 appearance=np.maximum(m['score'],.7*m['saturated_outline_score'])
 # 几何轨道不能穿过无限长低信息尾。保留不足5w的有界推断，长缺失后截断。
 if np.any(lane):
  first_lane=int(np.flatnonzero(lane)[0]);weak=0.;cut=len(xy)
  for i in range(first_lane,len(xy)):
   step=float(np.linalg.norm(xy[i]-xy[i-1])) if i else 0.
   weak=0. if (supported[i] and inside[i]) else weak+step
   if weak>PARAMETERS['max_unsupported_lane_widths']*width:cut=i;break
  if cut<len(xy):
   c=_trim(c,cut);c['terminal_reason']='long_unsupported_lane_truncated';xy=xy[:cut];ww=ww[:cut]
   m={k:(v[:cut] if isinstance(v,np.ndarray) else v) for k,v in m.items()};phase=phase[:cut];inside=inside[:cut];progress=progress[:cut];forward=forward[:cut];lane=lane[:cut];formal=formal[:cut];supported=supported[:cut];pair_score=pair_score[:cut];groove_score=groove_score[:cut];groove_supported=groove_supported[:cut];groove_available=groove_available[:cut];appearance=appearance[:cut]
 if len(xy)<3 or not np.any(inside&forward):return None
 span=float(np.linalg.norm(np.diff(xy[forward],axis=0),axis=1).sum()) if np.count_nonzero(forward)>1 else 0.
 support_fraction=_mean(supported,forward&inside)
 if span<width*PARAMETERS['minimum_path_span_widths'] or support_fraction<PARAMETERS['minimum_path_image_fraction']:return None
 if c['lane_id'] is None:
  terminal=forward&(progress>=max(0,float(progress[-1])-8*width));incoming=terminal;junction=terminal
  premature=0.;novel=_mean(np.maximum(0,m['comparison_score']),terminal&m['comparison_reliable'])
  score=.55*_mean(appearance,terminal)+.25*_mean(pair_score,terminal)+.20*min(1.,novel/.25)
 else:
  ji=min(c['join_index'],len(xy)-1);join_s=float(progress[ji]);ts=min(c['transition_start_index'],len(xy)-1);start_s=float(progress[ts])
  incoming=forward&(progress>=max(0,start_s-6*width))&(progress<=join_s)
  junction=forward&(progress>=join_s-2*width)&(progress<=join_s+3*width)
  # 对比候选已经偏离的位置与原自由直线。直线上若仍有宽带独特证据，
  # 就不应因为邻圈较亮而提前转过去。只在三条带均可用时计比较优势。
  check=forward&(progress>=start_s)&(progress<=join_s+5*width)
  si=np.flatnonzero(check);premature=0.
  if len(si):
   ray=o+progress[si,None]*d;rm=compare_ribbon_neighbors(fields,ray,d,width,pitch=fields['pitch_px']*1.1,free_texture=texture)
   strong=rm['supported']&(rm['score']>PARAMETERS['strong_straight_score'])&rm['comparison_reliable']&(rm['comparison_score']>PARAMETERS['unique_straight_advantage'])
   offset=abs((xy[si]-o)@normal)>width*PARAMETERS['premature_turn_offset_width']
   premature=float(np.mean(strong&offset))
  novel=_mean(np.maximum(0,m['comparison_score']),incoming&m['comparison_reliable'])
  # 转弯窗口已开始贴卷：自由绳条带或双沟可支持其外观，不把谷误判成无边缘。
  junction_score=np.where(lane,np.maximum(appearance,groove_score),appearance)
  score=.43*_mean(appearance,incoming)+.25*_mean(junction_score,junction)+.17*_mean(groove_score,lane&inside)+.10*min(1.,novel/.25)+.05*_mean(formal,lane)-.70*premature
  score-=.05*min(3.,float(c.get('geometry_cost',0)))
 c.update({'score':float(score),'combined_score':float(score),'score_kind':'image evidence plus explicit geometry/formal prior and contradiction penalty','support_fraction':support_fraction,'stage_support_fraction':{s:_mean(supported,forward&inside&(phase==s)) for s in ['free_straight','transition','lane','lane_extrapolation']},'stage_point_counts':{s:int(np.sum(forward&inside&(phase==s))) for s in ['free_straight','transition','lane','lane_extrapolation']},'length_px':span,'premature_turn_penalty_fraction':premature,'incoming_mean_score':_mean(m['score'],incoming),'junction_mean_score':_mean(m['score'],junction),'lane_paired_score':_mean(pair_score,lane&inside),'lane_groove_score':_mean(groove_score,lane&inside),'lane_groove_support_fraction':_mean(groove_supported,lane&inside),'groove_score':groove_score.tolist(),'groove_supported':groove_supported.tolist(),'groove_available':groove_available.tolist(),'neighbor_advantage':novel,'supported':supported.tolist(),'inside_search':inside.tolist(),
  'photometric':{k:m[k].tolist() for k in ['score','reliable','saturated_outline_supported','saturated_outline_score','signed_body_background_difference_raw','inner_mean_raw','left_background_mean_raw','right_background_mean_raw','paired_edge_contrast_raw','normalized_body_contrast','body_coverage','along_consistency','comparison_score','comparison_reliable','unknown_reason_bits']},
  'interpretation':'appearance route, not verified same-rope or physical contact'})
 return c

def trace_photometric(data,free):
 shape=data['surface'].shape;empty=np.zeros(shape,np.uint8)
 result={'status':'UNKNOWN','whole_mask':empty.copy(),'supported_mask':empty.copy(),'review_mask':empty.copy(),'candidates':[],'lanes':[],'diagnostics':{'same_material_identity_confirmed':False,'n_minus_one_used':False,'parameters':dict(PARAMETERS)}}
 if free['status']!='observed' or free['geometry']['status']!='observed':
  if free['status']=='observed':result.update(status='FREE_ONLY_DIRECTION_AMBIGUOUS',whole_mask=free['free_mask'].copy(),supported_mask=free['free_mask'].copy())
  result['diagnostics']['reason']='no_unambiguous_revalidated_free_entry';return result
 g=free['geometry'];width=float(g['width_px']);d=np.asarray(g['direction_xy']);o=np.asarray(g['origin_xy']);domain=search_surface(data['surface'])
 pool=generate_lane_candidates(data,free)
 # G0在任何RGB测量、拒绝或裁短之前固定，保证纯几何对照不被图像筛选污染。
 g0=min(pool['candidates'],key=lambda c:c['geometry_cost']) if pool['candidates'] else None
 result['g0_candidate']=g0
 def point_hash(v):return hashlib.sha256(json.dumps(v,separators=(',',':'),allow_nan=False).encode()).hexdigest()
 result['geometry_pool_ledger']=[{'candidate_id':c['candidate_id'],'lane_id':c['lane_id'],'geometry_cost':c['geometry_cost'],'point_count':len(c['xy']),'xy_sha256':point_hash(c['xy']),'widths_sha256':point_hash(c['widths'])} for c in pool['candidates']]
 fields=build_fields(data['original'],width,data['pitch']);texture=dict(free['texture_descriptor']);texture['free_axis_angle_rad']=float(np.arctan2(d[1],d[0]))
 straight,sdiag=_straight_candidate(fields,domain,free,texture)
 proposed=pool['candidates']+([straight] if straight is not None else [])
 scored=[];rejected=[]
 for c in proposed:
  value=_score(c,fields,domain,free,texture)
  if value is None:rejected.append(c['candidate_id'])
  else:scored.append(value)
 scored.sort(key=lambda c:-c['score']);result.update(candidates=scored,lanes=pool['lanes'])
 result['diagnostics'].update(lane_generation=pool['diagnostics'],straight_trace=sdiag,rejected_candidate_ids=rejected,proposed_count=len(proposed))
 if not scored or scored[0]['score']<PARAMETERS['minimum_candidate_score']:
  measured=free['free_mask']|free['straight_observed_mask'];result.update(status='FREE_ONLY',whole_mask=measured,supported_mask=measured.copy())
  result['diagnostics']['reason']='no_supported_photometric_route';return result
 # 直段回答“目前能跟到哪里”；接续回答“后面哪条轨道”。两者不是相同终点
 # 的竞争任务，不能因为短直段均分高就永远排除有原图双沟支持的转弯。
 continuations=[c for c in scored if c['lane_id'] is not None and c['score']>=PARAMETERS['minimum_candidate_score'] and c['premature_turn_penalty_fraction']<=PARAMETERS['maximum_premature_turn_fraction'] and c['lane_groove_support_fraction']>=PARAMETERS['minimum_attached_support_fraction']]
 alternatives=continuations if continuations else [c for c in scored if c['lane_id'] is None]
 if not alternatives:
  measured=free['free_mask']|free['straight_observed_mask'];result.update(status='FREE_ONLY',whole_mask=measured,supported_mask=measured.copy())
  result['diagnostics']['reason']='no_admissible_continuation_or_partial_straight';return result
 chosen=alternatives[0];xy=np.asarray(chosen['xy']);ww=np.asarray(chosen['widths']);support=np.array(chosen['supported']);phase=np.array(chosen['phase'])
 whole=paint(shape,xy,ww)|free['free_mask'];observed=(paint(shape,xy,ww,support)|free['free_mask'])&whole
 margin=chosen['score']-alternatives[1]['score'] if len(alternatives)>1 else None;ambiguous=margin is not None and margin<PARAMETERS['ambiguity_margin']
 review_flags=(phase!='free_straight')|~support
 if ambiguous:
  for alternative in alternatives[1:]:
   if chosen['score']-alternative['score']>=PARAMETERS['ambiguity_margin']:continue
   alt=np.asarray(alternative['xy']);order=np.argsort(alt[:,0]);alt=alt[order]
   valid=(xy[:,0]>=alt[0,0])&(xy[:,0]<=alt[-1,0]);delta=abs(xy[:,1]-np.interp(xy[:,0],alt[:,0],alt[:,1]));review_flags|=valid&(delta>.35*width)
 review=paint(shape,xy,ww,review_flags)|(whole&~observed);review&=whole
 status=('AMBIGUOUS_' if ambiguous else '')+('LANE_APPEARANCE_CANDIDATE' if chosen['lane_id'] is not None else 'STRAIGHT_OVERLAP_CANDIDATE')
 result.update(status=status,whole_mask=whole,supported_mask=observed,review_mask=review,selected_candidate_id=chosen['candidate_id'],g0_selected_candidate_id=g0['candidate_id'] if g0 else None)
 result['diagnostics']['selection_policy']='image-supported lane continuation when admissible; otherwise partial straight; G0 from identical pre-image lane pool'
 result['diagnostics'].update(reason='relative_background_then_formal_lane',score_margin=margin,selected_lane=chosen['lane_id'],whole_pixels=int(np.count_nonzero(whole)),supported_pixels=int(np.count_nonzero(observed)),physical_contact=None,visible_side_identity_verified=False)
 return result
