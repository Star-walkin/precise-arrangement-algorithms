"""A6：自身模板→原图多路径→延后检查F3→输出共同可信前缀。

可信是本图可见外观的连续支持，不是三维接触或材料身份的认证。
没有可靠后段时，主掩膜停在证据终点，竞争分支单独保存。
"""
import copy,hashlib,json,math
import cv2
import numpy as np
from free_rope import refine_and_extend
from free_seed_recovery import recover_free_seeds
from appearance_template import build_template,evaluate_template
from attached_groove import sample_attached_grooves
from image_beam import generate_image_candidates,sample,tangents,paint

PARAMETERS={'maximum_revalidated_entries':2,'maximum_recovery_probes':6,'maximum_output_candidates':6,
 'late_minimum_bend_deg':3.,'late_evidence_window_widths':3.,'late_minimum_image_fraction':.5,
 'F3_maximum_score_weight':.08,'divergence_width_ratio':.40,
 'trusted_maximum_gap_widths':1.,'minimum_branch_span_widths':3.,'trusted_unique_advantage':.08}

def _serial_geometry(free):
 return {'status':free['status'],'geometry':free['geometry'],'outside_measurement':free.get('diagnostics',{}),'free_segment':free.get('free_segment',{}),'texture_descriptor':free.get('texture_descriptor',{})}

def _same_entry(a,b):
 ga,gb=a['free']['geometry'],b['free']['geometry'];da=np.asarray(ga['direction_xy']);db=np.asarray(gb['direction_xy'])
 if da@db<math.cos(math.radians(2.5)):return False
 p=np.asarray(ga['origin_xy']);q=np.asarray(gb['origin_xy']);normal=np.array([-da[1],da[0]])
 return abs((q-p)@normal)<.45*max(ga['width_px'],gb['width_px'])

def prepare_entries(data):
 """入口失败可从原图重新找；没有用已知失败文件名或人工点接管。"""
 original=refine_and_extend(data);entries=[];attempts=[];recovery=None
 def accept(free,source,seed=None):
  if free['status']!='observed' or free['geometry']['status']!='observed':
   attempts.append({'source':source,'status':free['status'],'geometry_status':free['geometry']['status'],'reason':free.get('diagnostics',{}).get('reason')});return
  template=build_template(data,free);attempts.append({'source':source,'status':free['status'],'template_status':template['status'],'template_reason':template.get('reason')})
  if template['status'] not in ('ready','outline_only'):return
  entry={'entry_id':source,'free':free,'template':template,'recovered':seed is not None,'seed_evidence':seed}
  for i,old in enumerate(entries):
   if _same_entry(entry,old):
    if template['measured_span_px']>old['template']['measured_span_px']:entries[i]=entry
    return
  entries.append(entry)
 accept(original,'saved_seed_revalidated')
 if not entries:
  recovery=recover_free_seeds(data)
  for i,candidate in enumerate(recovery.get('candidates',[])[:PARAMETERS['maximum_recovery_probes']]):
   modified=dict(data);modified['a2_record']={'free_segment':candidate['seed']};modified['a1_record']={}
   accept(refine_and_extend(modified),'recovered_'+str(i).zfill(2),candidate)
 entries.sort(key=lambda x:-x['template']['measured_span_px'])
 omitted=[e['entry_id'] for e in entries[PARAMETERS['maximum_revalidated_entries']:]]
 return entries[:PARAMETERS['maximum_revalidated_entries']],{'attempts':attempts,'recovery':recovery,'omitted_entry_ids_by_budget':omitted,'original_seed_status':original['status']},original

def _rolling(v,n):
 n=max(1,min(int(n),len(v)));return np.convolve(np.asarray(v,float),np.ones(n)/n,mode='same')

def _late_check(candidate,fields,data,entry,domain):
 """只有原图已提出弯段且持续有证据，才查询F3，不改变原候选坐标。"""
 c=copy.deepcopy(candidate);p=np.asarray(c['xy']);ww=np.asarray(c['widths']);t=tangents(p);n=np.c_[-t[:,1],t[:,0]]
 free=entry['free'];g=free['geometry'];d=np.asarray(g['direction_xy']);w0=float(g['width_px']);o=np.asarray(g['origin_xy'])
 m=evaluate_template(fields,p,t,ww,entry['template']);edge=np.stack([p-n*ww[:,None]/2,p+n*ww[:,None]/2],axis=1)
 groove=sample_attached_grooves(fields,edge,t,ww);inside=sample(domain,p)>76
 angle=np.arccos(np.clip(t@d,-1,1));bent=angle>math.radians(PARAMETERS['late_minimum_bend_deg'])
 evidence=m['supported']|(groove['supported']&bent&inside)
 continuity=np.r_[True,abs(np.diff(ww))<=.10*np.maximum(ww[:-1],1.)]
 win=max(3,int(PARAMETERS['late_evidence_window_widths']*w0/8.))
 eligible=bent&inside&continuity&(_rolling(evidence,win)>=PARAMETERS['late_minimum_image_fraction'])
 alignment=np.zeros(len(p));distance=np.full((len(p),2),-1.);crossing=np.zeros(len(p),bool)
 if np.any(eligible):
  formal_distance=cv2.distanceTransform((data['merged']==0).astype(np.uint8),cv2.DIST_L2,3)
  distance[eligible]=sample(formal_distance,edge[eligible].reshape(-1,2)).reshape(-1,2)
  alignment[eligible]=np.clip(1.-np.mean(distance[eligible],axis=1)/np.maximum(.60*ww[eligible],1.),0,1)
  # F3不重合本身不判错；还须原图同时缺双边/双沟，才记穿越绳体的嫌疑。
  crossing=eligible&(alignment<.2)&(groove['score']<.15)&~m['supported']
 tail=np.arange(len(p))>=max(1,len(p)-max(5,int(12*w0/8)))
 local=np.where(m['supported'],m['score'],np.where(bent&groove['supported'],.63*groove['score'],0.))
 image_score=float(np.mean(local[tail])) if tail.any() else 0.
 original_only=float(np.mean(np.where(m['raw_supported'],np.minimum(1,m['raw_pair']/25.),0.)[tail]))
 f3score=float(np.mean(alignment[eligible])) if eligible.any() else 0.
 combined=image_score+PARAMETERS['F3_maximum_score_weight']*f3score-.25*float(np.mean(crossing[tail]))
 c.update(entry_id=entry['entry_id'],image_only_score=image_score,raw_edge_strength_proxy=original_only,late_combined_score=combined,
  supported=evidence.tolist(),direct_template_supported=m['supported'].tolist(),inside_search=inside.tolist(),angle_from_free_deg=np.degrees(angle).tolist(),
  late_F3={'checked':eligible.tolist(),'edge_distance_px':distance.tolist(),'alignment':alignment.tolist(),'body_crossing_suspect':crossing.tolist(),'physical_contact':None,'interpretation':'post-generation consistency only; not material identity'},
  evidence={k:m[k].tolist() for k in ['score','valid','profile_match','profile_valid','texture_match','texture_valid','width_match','measured_width_px','raw_pair','delta_gray','body_gray_raw','side_gray_raw','neighbor_advantage','neighbor_comparison_valid','unknown_bits','saturated_outline_supported']},
  original_groove_score=groove['score'].tolist(),F3_used_for_generation=False)
 return c

def _difference(a,b,entry):
 """在自由绳方向上比较同一前进位置，不依赖图像横坐标是否单调。

 搜索限制相对自由段转角小于65度，故沿自由轴的投影严格递增。
 不同候选的长短、转弯后的高分尾段，都不能代替共同区域的比较。
 """
 d=np.asarray(entry['free']['geometry']['direction_xy']);n=np.array([-d[1],d[0]])
 pa=np.asarray(a['xy']);pb=np.asarray(b['xy']);sa=pa@d;sb=pb@d;order=np.argsort(sb);sb=sb[order]
 valid=(sa>=sb[0])&(sa<=sb[-1]);delta=np.abs(pa@n-np.interp(sa,sb,(pb@n)[order]))
 return delta,valid

def _trusted_prefix(chosen,competitors,entry):
 p=np.asarray(chosen['xy']);width=np.asarray(chosen['widths']);m=chosen['evidence'];w0=float(entry['free']['geometry']['width_px'])
 direct=np.array(chosen['direct_template_supported']);angle=np.array(chosen['angle_from_free_deg']);unique=np.array(m['neighbor_comparison_valid'])&(np.array(m['neighbor_advantage'])>PARAMETERS['trusted_unique_advantage'])
 local=direct&((angle<3.)|unique);cross=np.array(chosen['late_F3']['body_crossing_suspect'])
 divergence=np.zeros(len(p),bool)
 for other in competitors:
  if other['candidate_id']==chosen['candidate_id']:continue
  delta,valid=_difference(chosen,other,entry)
  d=np.asarray(entry['free']['geometry']['direction_xy']);s=p@d;other_s=np.asarray(other['xy'])@d
  # 对方必须在当前相同位置有原图支持；不能凭一个无证据的几何尾巴截断。
  # 但不以各自远处尾段的平均分排除它：那是在比较不同图像区域，不能
  # 证明当前分叉已经消歧。检查整个生成池，先做判断，再裁剪展示数量。
  j=np.searchsorted(other_s,s,side='left');j=np.clip(j,0,len(other_s)-1);previous=np.maximum(0,j-1)
  previous=np.where(np.isclose(s,other_s[j],atol=1e-6,rtol=0),j,previous)
  flags=np.asarray(other['supported'],bool);supported=flags[j]&flags[previous]
  divergence|=valid&supported&(delta>PARAMETERS['divergence_width_ratio']*width)
 weak=0.;split=0.;last=0;reason='image_support_endpoint'
 for i in range(1,len(p)):
  step=float(np.linalg.norm(p[i]-p[i-1]));weak=0. if local[i] and not cross[i] else weak+step;split=split+step if divergence[i] else 0.
  if weak>PARAMETERS['trusted_maximum_gap_widths']*w0:reason='self_template_or_edges_no_longer_distinct';break
  if split>w0:reason='retained_branches_diverge';break
  if local[i] and not divergence[i]:last=i
 return max(1,last+1),reason

def detect(data):
 shape=data['surface'].shape;zero=np.zeros(shape,np.uint8)
 out={'status':'UNKNOWN','trusted_mask':zero.copy(),'review_mask':zero.copy(),'best_mask':zero.copy(),'candidates':[],'entries':[],
  'diagnostics':{'version':'IR-A6.0.0','parameters':dict(PARAMETERS),'same_material_identity_confirmed':False,'physical_contact':None,'F3_used_during_candidate_generation':False}}
 entries,diag,original=prepare_entries(data);out['diagnostics']['entry_search']=diag
 if not entries:
  # 旧复测尚有不完整外部观测时保留它，但不把缺模板当作整根不存在。
  if original['status']=='observed':out.update(status='FREE_ONLY_TEMPLATE_OR_DIRECTION_UNKNOWN',trusted_mask=original['free_mask'].copy(),best_mask=original['free_mask'].copy())
  return out
 scored=[];entry_map={e['entry_id']:e for e in entries};ledgers=[]
 for entry in entries:
  free=entry['free'];pool,fields,domain,search=generate_image_candidates(data['original'],data['surface'],data['pitch'],free,entry['template'])
  # 原图候选先记账，F3检查不能重新编造或移到邻轨。
  for c in pool:
   c['candidate_id']=entry['entry_id']+'_'+c['candidate_id'];ledger={'candidate_id':c['candidate_id'],'entry_id':entry['entry_id'],'point_count':len(c['xy']),'xy_sha256':hashlib.sha256(json.dumps(c['xy'],separators=(',',':'),allow_nan=False).encode()).hexdigest(),'image_search_score':c['image_search_score'],'geometry_cost':c['geometry_cost']};ledgers.append(ledger)
   scored.append(_late_check(c,fields,data,entry,domain))
  out['entries'].append({'entry_id':entry['entry_id'],'recovered':entry['recovered'],'free':_serial_geometry(free),'template':entry['template'],'search':search})
 out['image_pool_ledger']=ledgers
 if not scored:
  if len(entries)==1:
   mask=entries[0]['free']['free_mask'].copy();out.update(status='FREE_ONLY_NO_SUPPORTED_BEND',trusted_mask=mask,best_mask=mask.copy())
  else:out['status']='MULTIPLE_FREE_ENTRIES_NO_CONTINUATION'
  return out
 scored.sort(key=lambda c:-c['late_combined_score'])
 # 不同弯法保留，几乎完全重合的短前缀不占据全部输出名额。
 kept=[]
 for c in scored:
  if c['length_px']<PARAMETERS['minimum_branch_span_widths']*entry_map[c['entry_id']]['free']['geometry']['width_px']:continue
  duplicate=False
  for k in kept:
   if k['entry_id']!=c['entry_id']:continue
   delta,valid=_difference(c,k,entry_map[c['entry_id']])
   if valid.sum()>3 and np.median(delta[valid])<.18*entry_map[c['entry_id']]['free']['geometry']['width_px'] and np.max(delta[valid])<.4*entry_map[c['entry_id']]['free']['geometry']['width_px']:
    # 只在重叠部分相同，后面仍有新弯段时不能当重复。短前缀可以被更长解释
    # 包含，但不能反过来删掉长候选；也不把包含关系当作身份正确的证明。
    candidate_new_tail=c['length_px']>k['length_px']+entry_map[c['entry_id']]['free']['geometry']['width_px']
    if not candidate_new_tail:duplicate=True;break
  if not duplicate:kept.append(c)
 if not kept:kept=scored[:1]
 if len(kept)>PARAMETERS['maximum_output_candidates']:
  # 最佳外观候选之外，保留早/晚转弯族和未转弯的局部参考，不只取分数前六。
  grouped={}
  for c in kept:grouped.setdefault((c['entry_id'],c.get('turn_anchor_bin',-1)),[]).append(c)
  selected=[kept[0]];keys=sorted(grouped)
  if len(keys)>PARAMETERS['maximum_output_candidates']-1:
   keys=[keys[i] for i in np.unique(np.rint(np.linspace(0,len(keys)-1,PARAMETERS['maximum_output_candidates']-1)).astype(int))]
  for key in keys:
   for c in grouped[key]:
    if all(c is not old for old in selected):selected.append(c);break
   if len(selected)>=PARAMETERS['maximum_output_candidates']:break
  for c in kept:
   if len(selected)>=PARAMETERS['maximum_output_candidates']:break
   if all(c is not old for old in selected):selected.append(c)
  kept=selected
 chosen=kept[0];entry=entry_map[chosen['entry_id']]
 competitors=[c for c in scored if c['entry_id']==chosen['entry_id'] and c['length_px']>=PARAMETERS['minimum_branch_span_widths']*entry['free']['geometry']['width_px']]
 # 完整候选池是可复核证据；显示最多六条不等于仅存在六条解释。
 # 此处在添加外部自由段之前复制坐标，与生成时ledger哈希直接对应。
 out['candidate_pool']=[{key:copy.deepcopy(c[key]) for key in ('candidate_id','entry_id','xy','widths','supported','direct_template_supported','image_only_score','late_combined_score','length_px','turn_anchor_bin','F3_used_for_generation')} for c in scored]
 count,reason=_trusted_prefix(chosen,competitors,entry)
 trusted=paint(shape,chosen['xy'][:count],chosen['widths'][:count])|entry['free']['free_mask']
 if len(entries)>1:
  # 两条独立入口都能解释图像时不把其中一条宣称可信同根。
  trusted[:]=0;count=0;reason='multiple_independent_free_entries'
 union=zero.copy();best=zero.copy()
 for ci,c in enumerate(kept):
  e=entry_map[c['entry_id']];g=e['free']['geometry'];d=np.asarray(g['direction_xy']);o=np.asarray(g['origin_xy']);ends=np.asarray(e['free']['free_segment']['observed_segment_xy']);outer=ends[np.argmin(ends@d)]
  prefix=np.linspace(outer,o,max(2,int(np.linalg.norm(outer-o)/8)+1))[:-1]
  c['search_xy']=c['xy'];c['search_widths']=c['widths'];c['search_prefix_offset']=len(prefix)
  c['xy']=np.vstack([prefix,np.asarray(c['search_xy'])]).tolist();c['widths']=np.r_[np.full(len(prefix),g['width_px']),c['search_widths']].tolist()
  c['trusted_prefix_count']=len(prefix)+count if ci==0 and count else 0
  c['same_material_identity_confirmed']=False;c['candidate_role']='separate_appearance_hypothesis'
  mask=paint(shape,c['xy'],c['widths']);union|=mask
  if ci==0:best=mask
 trusted&=best;review=union.copy();review[trusted>0]=0
 status='MULTIPLE_PATHS_PREFIX_ONLY' if len(competitors)>1 or len(entries)>1 else 'PARTIAL_TRUSTED_WITH_UNVERIFIED_TAIL'
 out.update(status=status,trusted_mask=trusted,review_mask=review,best_mask=best,candidates=kept,selected_candidate_id=chosen['candidate_id'])
 out['diagnostics'].update(trusted_prefix_stop_reason=reason,trusted_search_point_count=count,retained_branch_count=len(kept),image_pool_count=len(scored),entry_count=len(entries),
  prefix_competitor_ids=[c['candidate_id'] for c in competitors],prefix_competition_scope='all_same_entry_generated_paths_with_local_image_support_before_display_pruning',
  image_only_selected_candidate_id=max(scored,key=lambda c:c['image_only_score'])['candidate_id'],geometry_ranked_in_image_pool=min(scored,key=lambda c:c['geometry_cost'])['candidate_id'],
  baseline_scope='rankings share an image-generated pool; not independent whole-pipeline ablations',branch_scores_are_identity_probabilities=False)
 return out
