"""Bounded image+response beam connection. No target labels are imported/read."""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, json, math, sys, time
import cv2
import numpy as np
from scene import stage_at_points

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()
def imread(p):return cv2.imdecode(np.fromfile(p,np.uint8),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.
def sample(im,xy):
    p=np.asarray(xy,np.float32);shape=p.shape[:-1]
    flat=p.reshape(-1,2);out=np.empty(len(flat),np.float32)
    for lo in range(0,len(flat),16000):
        q=flat[lo:lo+16000]
        out[lo:lo+len(q)]=cv2.remap(im,q[:,0].reshape(1,-1),q[:,1].reshape(1,-1),cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101).ravel()
    return out.reshape(shape)
def profiles(im,xy,angles,widths,cfg):
    xy=np.asarray(xy,float).reshape(-1,2);a=np.asarray(angles,float).reshape(-1)
    widths=np.broadcast_to(widths,(len(xy),));normal=np.stack([-np.sin(a),np.cos(a)],1)
    tangent=np.stack([np.cos(a),np.sin(a)],1)
    u=np.linspace(-1.25,1.25,cfg['profile_positions'])
    offsets=np.asarray(cfg['profile_tangent_offsets_px'])
    points=xy[:,None,None,:]+normal[:,None,None,:]*widths[:,None,None,None]*u[None,None,:,None]+tangent[:,None,None,:]*offsets[None,:,None,None]
    h,w=im.shape
    valid=(points[...,0]>=0)&(points[...,0]<=w-1)&(points[...,1]>=0)&(points[...,1]<=h-1)
    return sample(im,points).mean(1),valid.mean((1,2))
def evidence(prof,cfg,template=None):
    u=np.linspace(-1.25,1.25,prof.shape[-1]);std=np.maximum(prof.std(-1),cfg['profile_std_floor'])
    li=prof[:,(u>=-.35)&(u<=-.15)].mean(-1);lo=prof[:,(u>=-.9)&(u<=-.65)].mean(-1)
    ri=prof[:,(u>=.15)&(u<=.35)].mean(-1);ro=prof[:,(u>=.65)&(u<=.9)].mean(-1)
    dl=(li-lo)/std;dr=(ro-ri)/std
    two=np.clip(np.minimum(np.abs(dl),np.abs(dr))/2,0,1)*(dl*dr<0)
    z=(prof-prof.mean(-1,keepdims=True))/std[:,None]
    corr=np.zeros(len(prof)) if template is None else np.clip((z*template).mean(-1),0,1)
    return two,corr,z
def free_template(im,seed,cfg):
    p=np.asarray(seed,float);d=p[-1]-p[0];a=math.atan2(d[1],d[0]);t=np.linspace(.1,.9,7)
    centers=p[0]+t[:,None]*d
    options=[]
    for width in cfg['width_candidates_px']:
        prof,coverage=profiles(im,centers,np.full(len(t),a),float(width),cfg)
        two,_,z=evidence(prof,cfg)
        two[coverage<1]=0
        options.append((float(np.median(two)),width,z.mean(0),prof.mean(0)))
    best=max(options,key=lambda x:x[0]);template=best[2];template/=max(float(template.std()),1e-6)
    return float(best[1]),template,dict(width_px=float(best[1]),source='two-sided original-gray profiles within initial120px entry only',
        reliable_bilateral_proxy=best[0]>=cfg['support_bilateral_min'],pixel_scale_not_physical_diameter=True,
        candidates=[{'width_px':float(v[1]),'median_bilateral':v[0]} for v in options],
        profile_mean=best[3].tolist(),profile_normalized_positions=np.linspace(-1.25,1.25,cfg['profile_positions']).tolist())
def wrap(a):return (a+np.pi)%(2*np.pi)-np.pi
def node_quality(n,cfg):
    scores=n['scores'];tail=scores[-cfg['score_mean_tail_points']:]
    return .7*float(np.mean(scores))+.3*float(np.mean(tail)) if scores else 0.
def path_dist(a,b):
    # Compare corresponding arc positions, with endpoint extension for short paths.
    a=np.asarray(a);b=np.asarray(b);n=max(len(a),len(b));idx=np.arange(max(0,n-80),n)
    return float(np.median(np.linalg.norm(a[np.minimum(idx,len(a)-1)]-b[np.minimum(idx,len(b)-1)],axis=1)))

def search(im,response,seed,scene,bbox,cfg):
    width,template,widthmeta=free_template(im,seed,cfg)
    p=np.asarray(seed,float);d=p[-1]-p[0];d/=np.linalg.norm(d);a0=math.atan2(d[1],d[0])
    # Densified entry is a condition, not an observed prediction; retain this fact.
    count=max(2,int(np.ceil(np.linalg.norm(p[-1]-p[0])/cfg['step_px']))+1)
    initial=np.linspace(p[0],p[-1],count).tolist()
    root=dict(points=initial,angle=a0,scores=[0.]*count,evidence=[None]*count,weak=0.,reason=None)
    beam=[root];finished=[];edge=np.asarray(scene['target_edge_points_xy'],float).reshape(-1,2)
    offsets=np.deg2rad(cfg['angles_degrees']);widths=width*np.array(cfg['width_factors'])
    x0,y0,x1,y1=bbox;step=cfg['step_px'];h,w=im.shape;iterations=0
    for iteration in range(cfg['max_steps']):
        proposals=[]
        for parent in beam:
            for da in offsets:
                angle=parent['angle']+da
                if math.cos(angle-a0)<cfg['initial_forward_projection_min']:continue
                q=np.array(parent['points'][-1])+step*np.array([math.cos(angle),math.sin(angle)])
                if q[0]<x0+2 or q[0]>=x1-2 or q[1]<y0+2 or q[1]>=y1-2:
                    n=dict(parent);n['reason']='out_of_view' if q[0]<2 or q[0]>=w-2 or q[1]<2 or q[1]>=h-2 else 'search_crop_truncated_unknown'
                    finished.append(n);continue
                proposals.append((parent,q,angle,da))
        if not proposals:break
        q=np.array([r[1] for r in proposals]);angles=np.array([r[2] for r in proposals])
        pq=np.repeat(q,len(widths),axis=0);pa=np.repeat(angles,len(widths));pw=np.tile(widths,len(q))
        prof,coverage=profiles(im,pq,pa,pw,cfg);two,corr,_=evidence(prof,cfg,template)
        # Reflected values are only a sampling implementation detail; they
        # cannot count as observed flanks/texture outside the original frame.
        two[coverage<1]=0;corr[coverage<1]=0
        coverage=coverage.reshape(len(q),-1)
        two=two.reshape(len(q),-1);corr=corr.reshape(len(q),-1)
        chosen=np.argmax(cfg['bilateral_weight']*two+cfg['template_weight']*corr,axis=1)
        two=two[np.arange(len(q)),chosen];corr=corr[np.arange(len(q)),chosen]
        response_value=np.nan_to_num(sample(response,q),nan=0.)
        scores=cfg['response_weight']*response_value+cfg['bilateral_weight']*two+cfg['template_weight']*corr
        pool=[]
        for i,(parent,pos,angle,da) in enumerate(proposals):
            supported=bool(response_value[i]>=cfg['support_response_min'] and
              (two[i]>=cfg['support_bilateral_min'] or corr[i]>=cfg['support_template_min']))
            weak=0. if supported else parent['weak']+step
            score=float(scores[i]-cfg['curvature_penalty']*abs(da)/max(abs(offsets)))
            ev=dict(response=float(response_value[i]),bilateral=float(two[i]),entry_profile_similarity=float(corr[i]),
                optical_width_px=float(widths[chosen[i]]),image_supported_proxy=supported,
                original_image_profile_valid_fraction=float(coverage[i,chosen[i]]))
            n=dict(points=parent['points']+[pos.tolist()],angle=angle,scores=parent['scores']+[score],
                   evidence=parent['evidence']+[ev],weak=weak,reason=None)
            length=(len(n['points'])-1)*step
            edge_distance=float(np.sqrt(((edge-pos)**2).sum(1)).min()) if len(edge) else math.inf
            if length>240 and edge_distance<=cfg['edge_arrival_distance_px']:
                n['reason']='observed_edge_candidate_reached_identity_unconfirmed';finished.append(n)
            elif weak>=cfg['max_unsupported_distance_px']:
                n['reason']='prolonged_weak_support_ambiguous';finished.append(n)
            else:pool.append(n)
        iterations=iteration+1
        pool.sort(key=lambda n:float(np.mean(n['scores'][-24:]))+.25*node_quality(n,cfg),reverse=True)
        beam=[];keys=set()
        for n in pool:
            xy=n['points'][-1];key=(round(xy[0]/cfg['merge_endpoint_px']),round(xy[1]/cfg['merge_endpoint_px']),
                                  round(math.degrees(wrap(n['angle']))/cfg['merge_angle_degrees']))
            if key in keys:continue
            keys.add(key);beam.append(n)
            if len(beam)>=cfg['beam_width']:break
        # Bound saved terminal paths too; no thousands of almost-identical routes.
        if len(finished)>120:
            finished=sorted(finished,key=lambda n:node_quality(n,cfg),reverse=True)[:120]
        if not beam:break
    for n in beam:n['reason']='arc_length_budget_unresolved';finished.append(n)
    if not finished:return [],None,dict(width=widthmeta,iterations=iterations,reason='no_viable_path')
    # A small, bounded length reward avoids preferring an early dead end solely
    # because its prefix was easy; evidence still dominates total score.
    rank=lambda n:node_quality(n,cfg)+.03*min((len(n['points'])-count)*step/1600,1.)
    finished.sort(key=rank,reverse=True);chosen=[]
    for n in finished:
        if all(path_dist(n['points'],other['points'])>=cfg['distinct_median_distance_px'] for other in chosen):chosen.append(n)
        if len(chosen)>=cfg['final_candidate_count']:break
    margin=rank(chosen[0])-rank(chosen[1]) if len(chosen)>1 else None
    rows=[]
    for j,n in enumerate(chosen):
        points=np.asarray(n['points']);stages=stage_at_points(scene,points).tolist();statuses=[];first_amb=None
        competitor=chosen[1 if j==0 else 0] if len(chosen)>1 else None
        for k,ev in enumerate(n['evidence']):
            status='ambiguous' if ev is None or not ev['image_supported_proxy'] else 'visible_supported'
            if k<count:status='entry_condition'
            if competitor is not None and margin is not None and margin<cfg['selection_score_margin']:
                other=competitor['points'][min(k,len(competitor['points'])-1)]
                if np.linalg.norm(points[k]-other)>max(width*.5,8):
                    if first_amb is None:first_amb=k
            if first_amb is not None and k>=first_amb:status='ambiguous'
            statuses.append(status)
        evs=[v for v,s in zip(n['evidence'],stages) if s>=3 and v is not None]
        fraction=float(np.mean([v['image_supported_proxy'] for v in evs])) if evs else 0.
        segment=np.linalg.norm(np.diff(points,axis=0),axis=1)
        lengths={s:float(segment[np.array(statuses[1:])==s].sum()) for s in ['entry_condition','visible_supported','ambiguous','inferred']}
        rows.append(dict(id=f'c{j}',points_xy=points.tolist(),point_status=statuses,stage=stages,
            evidence=n['evidence'],rank_score=float(rank(n)),mean_evidence_score=node_quality(n,cfg),
            termination=n['reason'],length_px=float(segment.sum()),status_length_px=lengths,
            attached_side_image_support_fraction=fraction,first_competitor_divergence_index=first_amb,
            reached_edge_candidate=n['reason']=='observed_edge_candidate_reached_identity_unconfirmed',
            stage_length_px={str(s):float(segment[np.array(stages[1:])==s].sum()) for s in range(1,5)},
            full_target_identity_confirmed=False))
    top=rows[0]
    selectable=(margin is None or margin>=cfg['selection_score_margin']) and top['attached_side_image_support_fraction']>=cfg['selection_attached_support_fraction']
    # Selection is only a provisional algorithm decision, never an acceptance
    # that the geometric endpoint is the correct target termination.
    selected='c0' if selectable else None
    return rows,selected,dict(width=widthmeta,iterations=iterations,score_margin=margin,
        selection_reason='provisional best image+response path' if selected else 'competing score margin or attached-side evidence insufficient',
        visible_supported_definition='local image and response evidence only, NOT independently verified target identity',
        inferred_occlusion_length_px=0.,occlusion_policy='dark/weak is ambiguous; no unsupported assertion of real occlusion')

def connect_one(item,mode,out,cfg):
    inf=load(out/'inference.json');entry=load(out/'entry.json')
    if mode=='AUTO':
        auto=load(item['auto_entry_path'])
        scene=load(auto['scene_path']) if auto.get('available') else dict(
            contact_proxy={'status':'unknown','xy':None,'t_from_entry_px':None},
            target_edge_points_xy=[],entry_short_xy=[],entry_origin_xy=[0.,0.],entry_direction_xy=[1.,0.],
            parameters={'contact_band_px':96.,'edge_band_px':96.},
            actual_image_boundary_xyxy=[0,0,item['shape'][1]-1,item['shape'][0]-1],
            target_edge_reason='automatic entry unavailable; no known-entry scene used',
            target_identity_at_edge_confirmed=False,physical_contact_confirmed=False,
            placeholder_coordinates_not_an_entry=True)
        seed=auto.get('free_seed',[]);available=auto.get('available',False)
    else:scene=load(item['scene_path']);seed=item['known_seed_xy'];available=True
    if available:
        im=imread(item['image_path']);response=np.load(out/'response.npy',mmap_mode='r');rows,selected,meta=search(im,response,seed,scene,item['bbox'],cfg)
    else:rows=[];selected=None;meta={'reason':'automatic_entry_unavailable','details':entry}
    result=dict(id=item['id'],mode=mode,selected_candidate_id=selected,candidates=rows,edge_target=scene,entry=entry,
        search=meta,parameters=cfg,coordinate_system='original image pixels',
        supervision_used_at_inference=False,scope='candidate arc and local support; no complete endpoint identity ground truth',
        response_sha256=sha(out/'response.npy'),image_sha256=item['image_sha256'],source_sha256=sha(__file__))
    dump(out/'paths.json',result)
    mask=np.zeros(item['shape'],np.uint8)
    # Thin schematic centerlines have no body width semantics. Only selected
    # candidate is rendered; competitors are saved separately, never unioned.
    if selected:
        pts=np.rint(rows[0]['points_xy']).astype(np.int32);cv2.polylines(mask,[pts],False,255,1)
    cv2.imencode('.png',mask)[1].tofile(str(out/'selected_centerline_schematic.png'))
    return {'id':item['id'],'mode':mode,'selected':selected,'candidate_count':len(rows),
            'top_length_px':rows[0]['length_px'] if rows else 0,'top_termination':rows[0]['termination'] if rows else 'entry_unavailable'}

def run(run_dir):
    tick=time.perf_counter();run_dir=Path(run_dir).resolve();cfg=load(run_dir/'path_config.json');index=load(run_dir/'data/index.json')
    outputs=[]
    for item in index['items']:
        for mode in ['CONDITIONAL']+(['AUTO'] if item['split']=='eval' else []):
            out=run_dir/'predictions'/mode/item['id'];assert (out/'response.npy').exists(),out
            assert not (out/'paths.json').exists(),'refuse to replace frozen predictions'
            row=connect_one(item,mode,out,cfg);outputs.append(row);print(json.dumps(row),flush=True)
    dump(run_dir/'connection_receipt.json',{'items':outputs,'elapsed_seconds':time.perf_counter()-tick,'command':sys.argv,
        'exit_code':0,'config_sha256':sha(run_dir/'path_config.json'),'source_sha256':sha(__file__),
        'labels_read':False,'geometry_boundary_is_not_identity_acceptance':True})

def checks():
    cfg=dict(step_px=12.,max_steps=80,beam_width=9,angles_degrees=[-16,0,16],width_candidates_px=[12,16,20,24],width_factors=[.8,1,1.25],
        profile_positions=33,profile_tangent_offsets_px=[-6,0,6],profile_std_floor=.015,response_weight=.7,bilateral_weight=.2,template_weight=.1,
        curvature_penalty=.025,initial_forward_projection_min=-.15,merge_endpoint_px=6,merge_angle_degrees=16,max_unsupported_distance_px=120,
        support_response_min=.25,support_bilateral_min=.18,support_template_min=.35,edge_arrival_distance_px=12,final_candidate_count=3,
        distinct_median_distance_px=12,selection_score_margin=.025,selection_attached_support_fraction=.65,score_mean_tail_points=80)
    h,w=320,700;im=np.full((h,w),.7,np.float32);im[150:170]=.2
    yy=np.arange(h)[:,None];response=np.broadcast_to(np.exp(-.5*((yy-160)/3)**2),(h,w)).copy().astype(np.float32)
    scene={'entry_origin_xy':[30,160],'entry_direction_xy':[1,0],'contact_proxy':{'t_from_entry_px':200},
        'target_edge_points_xy':[[650,160]],'parameters':{'contact_band_px':50,'edge_band_px':30}}
    rows,selected,meta=search(im,response,[[30,160],[150,160]],scene,[0,0,w,h],cfg)
    assert rows and rows[0]['reached_edge_candidate'] and max(abs(np.array(rows[0]['points_xy'])[:,1]-160))<8
    flat=np.full_like(im,.5);zero=np.zeros_like(response)
    empty,_,_=search(flat,zero,[[30,160],[150,160]],scene,[0,0,w,h],cfg)
    assert all(r['termination']=='prolonged_weak_support_ambiguous' for r in empty)
    assert len(rows)<=3 and all(len(r['points_xy'])==len(r['stage'])==len(r['point_status']) for r in rows)
    return {'straight_supported_center_followed':True,'flat_no_response_stops_unknown':True,'maximum_three_competitors':True,
            'original_coordinates_status_alignment':True,'no_identity_or_occlusion_assertion':True}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.check:print(json.dumps(checks()))
    else:run(a.run)
