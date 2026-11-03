"""先看原图的整绳条带，再保留若干平滑走向。这里不接收F3边界。

beam（束搜索）好比同时让几位观察者沿绳向前走：每一步都保留不同位置、
不同倾角的合理接法，而不是只让当前最亮的一条活下来。所有坐标为原图像素。
"""
import math
import cv2
import numpy as np
from appearance_template import build_fields,evaluate_template
from attached_groove import sample_attached_grooves
from axial_reacquisition import probe_axis_bridge

PARAMETERS={'step_px':8.,'beam_width':18,'maximum_steps':330,
 'turn_increments_deg':[-2.5,-1.25,0.,1.25,2.5],
 'maximum_axis_turn_deg':65.,'maximum_weak_widths':6.,
 'maximum_candidates':8,'minimum_candidate_span_widths':3.,
 'maximum_outside_after_entry_widths':1.5,'width_min_ratio':.60,'width_max_ratio':1.70,
 'width_update_interval':4,'maximum_normal_offset_widths':16.,'end_diversity_width':.55,
 'protected_axis_states':3,'turn_anchor_bin_widths':4.}

def sample(a,xy):
 p=np.asarray(xy,np.float32).reshape(-1,2)
 return cv2.remap(a,p[:,0][None],p[:,1][None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).ravel()

def domain_from_surface(surface):
 """仅限制搜索的显著卷面包络；绝不修改或宣称修复原SAM。"""
 cs,_=cv2.findContours(surface,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);n=np.count_nonzero(surface)
 cs=[c for c in cs if cv2.contourArea(c)>=max(50.,.0001*n)];out=np.zeros_like(surface)
 if not cs:return out
 hull=cv2.convexHull(np.vstack(cs));cv2.fillConvexPoly(out,hull,255)
 if np.count_nonzero(out)>1.8*max(n,1):out[:]=0;cv2.drawContours(out,cs,-1,255,-1)
 return out

def tangents(p):
 t=np.gradient(np.asarray(p,float),axis=0)
 return t/np.maximum(np.linalg.norm(t,axis=1)[:,None],1e-8)

def paint(shape,xy,widths,flags=None):
 a=np.zeros(shape,np.uint8);p=np.asarray(xy,float);w=np.broadcast_to(widths,(len(p),))
 for i in range(1,len(p)):
  if flags is not None and not(flags[i-1] and flags[i]):continue
  d=p[i]-p[i-1];n=np.array([-d[1],d[0]])/max(np.linalg.norm(d),1e-8)
  q=[p[i-1]+n*w[i-1]/2,p[i]+n*w[i]/2,p[i]-n*w[i]/2,p[i-1]-n*w[i-1]/2]
  cv2.fillConvexPoly(a,np.rint(q).astype(np.int32),255)
 return a

def _path(nodes,index):
 indices=[]
 while index>=0:indices.append(index);index=nodes[index]['parent']
 indices.reverse();return [nodes[i] for i in indices]

def _straight_reference(fields,domain,free,template,step):
 """保留一条严格由实测自由轴提出的局部参考，避免束搜索把直行逐步剪掉。

 它仍逐点接受同一个自身模板检查，使用相同6倍绳宽的弱证据预算。
 不使用双沟把背景旧圈当作直绳支持，不越过长缺口，不声明直线即真值。
 它只是候选池中的简单几何基线；到证据终点截断，不强行寻找接触点。
 """
 g=free['geometry'];o=np.asarray(g['origin_xy']);d=np.asarray(g['direction_xy']);w0=float(g['width_px'])
 ss=np.arange(0.,PARAMETERS['maximum_steps']*step+step/2,step);p=o+ss[:,None]*d
 fade=np.clip(ss/(4*w0),0,1);fade=fade*fade*(3-2*fade)
 reference=float(template.get('reference_width_px',w0));reference=float(np.clip(reference,PARAMETERS['width_min_ratio']*w0,PARAMETERS['width_max_ratio']*w0))
 ww=w0+(reference-w0)*fade;m=evaluate_template(fields,p,d,ww,template);inside=sample(domain,p)>76
 h,w=domain.shape;weak=0.;outside=0.;entered=bool(inside[0]);end=0
 for i in range(1,len(p)):
  if not(1.5*ww[i]<p[i,0]<w-1.5*ww[i] and 1.5*ww[i]<p[i,1]<h-1.5*ww[i]):break
  old_entered=entered;entered=entered or bool(inside[i]);outside=0. if inside[i] else outside+step
  weak=0. if m['supported'][i] else weak+step
  if weak>PARAMETERS['maximum_weak_widths']*w0 or (old_entered and outside>PARAMETERS['maximum_outside_after_entry_widths']*w0):break
  if m['supported'][i]:end=i
 if ss[end]<PARAMETERS['minimum_candidate_span_widths']*w0 or not inside[:end+1].any():return None
 p=p[:end+1];ww=ww[:end+1];tail=max(0,end+1-max(3,int(10*w0/step)))
 return {'candidate_id':'axis_reference','xy':p.tolist(),'widths':ww.tolist(),'search_supported':m['supported'][:end+1].tolist(),'search_direct_supported':m['supported'][:end+1].tolist(),'template_score':m['score'][:end+1].tolist(),'groove_score':[0.]*len(p),'neighbor_advantage':m['neighbor_advantage'][:end+1].tolist(),'image_search_score':float(np.mean(np.where(m['supported'][tail:end+1],m['score'][tail:end+1],0.))),'geometry_cost':0.,'length_px':float(ss[end]),'turn_anchor_bin':-1,'turn_anchor_bin_widths':PARAMETERS['turn_anchor_bin_widths'],'F3_used_for_generation':False,'terminal_reason':'measured_axis_local_reference_stopped_at_image_evidence','physical_contact':None,'same_material_identity':None}

def generate_image_candidates(original,surface,pitch,free,template):
 """只允许原图、搜索包络和自身模板进入候选生成。没有F3参数或外部标签。"""
 g=free['geometry'];o=np.asarray(g['origin_xy'],float);d=np.asarray(g['direction_xy'],float)
 w0=float(g['width_px']);theta0=float(math.atan2(d[1],d[0]));normal=np.array([-d[1],d[0]])
 fields=build_fields(original,w0,pitch);domain=domain_from_surface(surface);h,w=surface.shape
 step=PARAMETERS['step_px'];outside_distance=cv2.distanceTransform((domain==0).astype(np.uint8),cv2.DIST_L2,3)
 nodes=[{'parent':-1,'xy':o.tolist(),'theta':theta0,'width':w0,'value':0.,'total':0.,'support':True,'image_support':True,'weak':0.,'outside':0.,'entered':bool(sample(domain,[o])[0]>76),'depth':0,'template_score':0.,'groove_score':0.,'neighbor_advantage':0.,'curvature':0.,'turn_anchor':-1}]
 nodes[0].update(arc_px=0.,departure_px=0.)
 active=[0];terminals=[];stats={'evaluated_nodes':0,'retained_nodes':1,'pruned_diversity':0,'F3_used_during_generation':False,'parameters':dict(PARAMETERS)}
 bridge=probe_axis_bridge(fields,domain,free,template);stats['axial_reacquisition']=bridge
 if bridge.get('accepted'):
  bp=np.asarray(bridge['xy']);bw=np.asarray(bridge['widths']);bs=np.asarray(bridge['supported']);scores=np.asarray(bridge['score'])
  for i in range(1,len(bp)):
   old=nodes[-1];inside=bool(sample(domain,[bp[i]])[0]>76);dist=float(np.linalg.norm(bp[i]-bp[i-1]));reward=float(scores[i]) if bs[i] else -.28
   total=old['total']+reward*dist
   nodes.append({'parent':len(nodes)-1,'xy':bp[i].tolist(),'theta':theta0,'width':float(bw[i]),'value':float(total/i),'total':float(total),'support':bool(bs[i]),'image_support':bool(bs[i]),'weak':0. if bs[i] else old['weak']+dist,'outside':0. if inside else old['outside']+dist,'entered':old['entered'] or inside,'depth':i,'template_score':float(scores[i]),'groove_score':0.,'neighbor_advantage':0.,'curvature':0.,'reacquisition_axis':True,'turn_anchor':-1})
   nodes[-1].update(arc_px=old['arc_px']+dist,departure_px=0.,value=total/max(old['arc_px']+dist,1.))
  active=[len(nodes)-1];stats['retained_nodes']=len(nodes)
 last_depth=nodes[-1]['depth']+int(max(0,PARAMETERS['maximum_steps']*step-nodes[-1]['arc_px'])/step)
 for depth in range(nodes[-1]['depth']+1,last_depth+1):
  proposed=[]
  for ni in active:
   node=nodes[ni];p=np.asarray(node['xy']);theta=node['theta']
   # 离卷面较远时只允许小角度校正，避免被远处机架纹路带走。
   near=node['entered'] or sample(outside_distance,[p])[0]<3*w0
   changes=PARAMETERS['turn_increments_deg'] if near else [-.5,0.,.5]
   ratios=[.96,1.,1.04] if depth%PARAMETERS['width_update_interval']==0 else [1.]
   for degrees in changes:
    turn=math.radians(degrees);angle=theta+turn
    if abs(angle-theta0)>math.radians(PARAMETERS['maximum_axis_turn_deg']):continue
    direction=np.array([math.cos(angle),math.sin(angle)]);q=p+step*direction
    if (q-o)@d<0 or abs((q-o)@normal)>PARAMETERS['maximum_normal_offset_widths']*w0:continue
    for ratio in ratios:
     width=node['width']*ratio
     if not PARAMETERS['width_min_ratio']*w0<=width<=PARAMETERS['width_max_ratio']*w0:continue
     if not(width*1.5<q[0]<w-width*1.5 and width*1.5<q[1]<h-width*1.5):continue
     proposed.append((ni,q,angle,width,turn,ratio))
  if not proposed:terminals+=active;break
  q=np.asarray([p[1] for p in proposed]);ang=np.array([p[2] for p in proposed]);ww=np.array([p[3] for p in proposed]);tt=np.c_[np.cos(ang),np.sin(ang)]
  measured=evaluate_template(fields,q,tt,ww,template)
  normals=np.c_[-tt[:,1],tt[:,0]];edges=np.stack([q-.5*ww[:,None]*normals,q+.5*ww[:,None]*normals],axis=1)
  # 图像中真的存在双沟，也能提出贴卷后的外观分支。这里仍不查F3。
  groove=sample_attached_grooves(fields,edges,tt,ww)
  inside=sample(domain,q)>76;values=[];stats['evaluated_nodes']+=len(proposed)
  for i,(parent,point,angle,width,turn,ratio) in enumerate(proposed):
   old=nodes[parent];entered=old['entered'] or bool(inside[i]);bending=abs(angle-theta0)>math.radians(3.)
   direct=bool(measured['supported'][i]);paired=bool(groove['supported'][i] and bending and inside[i])
   support=direct or paired
   evidence=max(float(measured['score'][i]) if direct else .35*float(measured['score'][i]),.63*float(groove['score'][i]) if paired else 0.)
   advantage=float(measured['neighbor_advantage'][i]) if measured['neighbor_comparison_valid'][i] else 0.
   reward=evidence+.12*np.clip(advantage,-.5,.5)-.030*abs(turn)/math.radians(2.5)-.035*abs(turn-old['curvature'])/math.radians(2.5)-.01*abs(math.log(ratio))/.04
   if not support:reward-=.28
   weak=0. if support else old['weak']+step;outside=0. if inside[i] else old['outside']+step
   arc=old['arc_px']+step;total=old['total']+reward*step;value=total/max(arc,1.)
   anchor=old['turn_anchor']
   departing=abs(angle-theta0)>math.radians(4.) and abs((point-o)@normal)>.55*w0
   departure=old['departure_px']+step if departing else 0.
   if anchor<0 and departure>=w0:
    anchor=int(max(0,(point-o)@d-departure)/(PARAMETERS['turn_anchor_bin_widths']*w0))
   child={'parent':parent,'xy':point.tolist(),'theta':float(angle),'width':float(width),'value':float(value),'total':float(total),'support':support,'image_support':direct,'weak':float(weak),'outside':float(outside),'entered':entered,'depth':depth,'template_score':float(measured['score'][i]),'groove_score':float(groove['score'][i]),'neighbor_advantage':advantage,'curvature':float(turn),'turn_anchor':anchor}
   child.update(arc_px=arc,departure_px=departure)
   if weak>PARAMETERS['maximum_weak_widths']*w0 or (old['entered'] and outside>PARAMETERS['maximum_outside_after_entry_widths']*w0):continue
   values.append(child)
  # 一个试探方向失败，不代表这个父节点已经到终点；只登记无活子节点的父节点。
  alive_parents={v['parent'] for v in values}
  terminals += [i for i in active if i not in alive_parents and nodes[i]['entered']]
  if not values:break
  values.sort(key=lambda v:-v['value']);kept=[];bins=set();unique=[]
  for child in values:
   pp=np.asarray(child['xy']);key=(round(float((pp-o)@normal)/(w0*.30)),round((child['theta']-theta0)/math.radians(4.)),round(math.log(child['width']/w0)/.15))
   if key in bins:stats['pruned_diversity']+=1;continue
   bins.add(key);unique.append(child)
  # 留出直行名额：尚未贴卷的活动绳物理上仍接近直线。它必须通过同样的
  # 图像支持与弱段预算；这里只保护候选，不把几何推断改成可信像素。
  axis=[v for v in unique if v['turn_anchor']<0 and abs(v['theta']-theta0)<math.radians(3.) and abs((np.asarray(v['xy'])-o)@normal)<.6*w0]
  exact=sorted(axis,key=lambda v:abs(v['theta']-theta0)+abs((np.asarray(v['xy'])-o)@normal)/w0)
  selected=exact[:1]
  for v in axis:
   if all(v is not old for old in selected):selected.append(v)
   if len(selected)>=PARAMETERS['protected_axis_states']:break
  # 按首次明显偏离直线的位置轮流分配名额，避免早弯的旧圈占满束宽。
  groups={}
  for v in unique:
   if any(v is old for old in selected):continue
   key=(v['turn_anchor'],int(np.sign(v['theta']-theta0)))
   groups.setdefault(key,[]).append(v)
  keys=sorted(groups,key=lambda k:-groups[k][0]['value'])
  while len(selected)<PARAMETERS['beam_width'] and any(groups[k] for k in keys):
   for key in keys:
    if groups[key]:selected.append(groups[key].pop(0))
    if len(selected)>=PARAMETERS['beam_width']:break
  for child in selected:nodes.append(child);kept.append(len(nodes)-1)
  active=kept;stats['retained_nodes']+=len(kept)
  if not active:break
 terminals+=active
 # 同一末端分叉先去重，不把同一条路径的小抖动冒充多个竞争解释。
 terminals=sorted(set(terminals),key=lambda i:-(nodes[i]['value']+.03*min(nodes[i]['arc_px']/(30*w0),1.)))
 axis_reference=_straight_reference(fields,domain,free,template,step)
 candidates=[] if axis_reference is None else [axis_reference]
 ends=[] if axis_reference is None else [np.asarray(axis_reference['xy'][-1])]
 stats['exact_axis_reference']={'present':axis_reference is not None,'length_px':axis_reference['length_px'] if axis_reference is not None else 0.,'not_an_identity_claim':True}
 families={}
 for index in terminals:families.setdefault(nodes[index]['turn_anchor'],[]).append(index)
 # 末端同样保留不同转弯起点。轴向候选有真实图像支持才会通过下方检查。
 ordered=[]
 keys=sorted(families)
 if len(keys)>PARAMETERS['maximum_candidates']:
  keys=[keys[i] for i in np.unique(np.rint(np.linspace(0,len(keys)-1,PARAMETERS['maximum_candidates'])).astype(int))]
  families={k:families[k] for k in keys}
 while any(families.values()):
  for key in keys:
   if families[key]:ordered.append(families[key].pop(0))
 for index in ordered:
  path=_path(nodes,index)
  if len(path)<4 or not any(n['entered'] for n in path):continue
  # 搜索可以越过短弱区；没有证据的末尾不涂成绳体。
  good=[i for i,n in enumerate(path) if n['support']]
  if not good:continue
  path=path[:good[-1]+1]
  if path[-1]['arc_px']<PARAMETERS['minimum_candidate_span_widths']*w0:continue
  if not any(n['entered'] for n in path):continue
  end=np.asarray(path[-1]['xy'])
  if any(np.linalg.norm(end-x)<PARAMETERS['end_diversity_width']*w0 for x in ends):continue
  q=np.asarray([n['xy'] for n in path]);widths=np.asarray([n['width'] for n in path]);tail=path[-max(3,int(10*w0/step)):]
  if not np.any(sample(domain,q)>76):continue
  candidates.append({'candidate_id':'image_path_'+str(len(candidates)).zfill(2),'xy':q.tolist(),'widths':widths.tolist(),'search_supported':[n['support'] for n in path],'search_direct_supported':[n['image_support'] for n in path],'template_score':[n['template_score'] for n in path],'groove_score':[n['groove_score'] for n in path],'neighbor_advantage':[n['neighbor_advantage'] for n in path],'image_search_score':float(np.mean([n['template_score'] if n['image_support'] else .63*n['groove_score'] for n in tail])),'geometry_cost':float(np.sum(abs(np.diff([n['theta'] for n in path])))),'length_px':float(np.linalg.norm(np.diff(q,axis=0),axis=1).sum()),'turn_anchor_bin':int(path[-1]['turn_anchor']),'turn_anchor_bin_widths':PARAMETERS['turn_anchor_bin_widths'],'F3_used_for_generation':False,'terminal_reason':'bounded_original_image_support','physical_contact':None,'same_material_identity':None})
  ends.append(end)
  if len(candidates)>=PARAMETERS['maximum_candidates']:break
 stats['candidate_count']=len(candidates);return candidates,fields,domain,stats
