"""P1: frozen-response ridges -> ordered fragments -> bounded directed gap graph.

Only NumPy/OpenCV and the existing pure visual-stage helper are used. Coordinates
and every geometric parameter are original-image pixels. Evidence states never
assert target identity, material continuity, physical contact, or real occlusion.
"""
from __future__ import annotations
import hashlib,json,math,time
from collections import Counter,defaultdict
import cv2
import numpy as np
from scene import stage_at_points


DEFAULTS=dict(work_scale=.5, gaussian_sigma_px=2., gaussian_truncate=3.,
    high_response=.45,low_response=.20,nms_offset_px=3.,peak_contrast_offset_px=6.,
    minimum_peak_contrast=.020,minimum_negative_curvature=.0005,valid_border_guard_px=6.,
    direction_link_degrees=42.,pixel_link_tangent_degrees=65.,thinning_iterations=12,
    minimum_segment_length_px=24.,resample_step_px=4.,maximum_ridge_pixels=80000,
    maximum_raw_chains=6000,maximum_segments=1500,maximum_chain_nodes=3000,
    maximum_extract_seconds=15.,image_width_candidates_px=[16.,24.,36.],
    image_profile_std_floor=.015,image_bilateral_min=.18,minimum_segment_supported_fraction=.65,
    maximum_gap_px=40.,maximum_total_bridge_px=120.,maximum_bridge_count=6,
    direct_endpoint_distance_px=1.,maximum_turn_degrees=48.,maximum_gap_tangent_degrees=55.,
    maximum_lateral_offset_px=10.,maximum_bandwidth_ratio=3.,maximum_optical_width_ratio=2.5,
    minimum_gap_response=.12,minimum_gap_image_support=.18,
    maximum_neighbors_per_endpoint=6,maximum_near_endpoint_tests=12,maximum_graph_edge_tests=12000,
    maximum_graph_seconds=8.,maximum_search_seconds=5.,maximum_expansions=5000,
    maximum_path_segments=40,maximum_path_length_px=4500.,beam_width=18,
    maximum_start_candidates=8,entry_attach_distance_px=40.,entry_tangent_degrees=60.,entry_direct_projection_px=4.,
    final_candidate_count=3,distinct_path_distance_px=14.,selection_score_margin=.025,
    minimum_new_supported_length_px=120.,maximum_selected_bridge_fraction=.15,
    edge_arrival_distance_px=24.,edge_minimum_new_length_px=240.,
    response_weight=.75,image_weight=.25,gap_penalty=.16,lateral_penalty=.08,
    curvature_penalty=.04,length_reward=.025,
    threshold_policy='one absolute high/low response rule shared by A and B; no per-image normalization',
    evidence_policy='heatmap_supported or weak heatmap plus original-gray image_supported; bridges are inferred geometry, not detected rope or occlusion recovery')


def config_hash(cfg):return hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def unit(v):
    a=np.asarray(v,float);return a/max(float(np.linalg.norm(a)),1e-9)
def angle(a,b):return math.degrees(math.acos(float(np.clip(np.dot(unit(a),unit(b)),-1,1))))
def sample(im,xy,interpolation=cv2.INTER_LINEAR):
    q=np.asarray(xy,np.float32);shape=q.shape[:-1];q=q.reshape(-1,2);out=np.empty(len(q),np.float32)
    for i in range(0,len(q),30000):
        v=q[i:i+30000]
        out[i:i+len(v)]=cv2.remap(im,v[:,0][None],v[:,1][None],interpolation,borderMode=cv2.BORDER_CONSTANT,borderValue=0).ravel()
    return out.reshape(shape)
def resample(points,step):
    p=np.asarray(points,float);d=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    if d[-1]<=1e-8:return p[:1]
    t=np.r_[np.arange(0,d[-1],step),d[-1]]
    return np.stack([np.interp(t,d,p[:,0]),np.interp(t,d,p[:,1])],1)
def tangents(p):
    p=np.asarray(p,float);v=np.gradient(p,axis=0);return v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-8)
def image_evidence(gray,points,directions,cfg):
    p=np.asarray(points,float).reshape(-1,2);d=np.asarray(directions,float).reshape(-1,2)
    if not len(p):return np.zeros(0),np.zeros(0)
    n=np.stack([-d[:,1],d[:,0]],1);u=np.linspace(-.9,.9,13);widths=np.asarray(cfg['image_width_candidates_px'])
    q=p[:,None,None,:]+n[:,None,None,:]*widths[None,:,None,None]*u[None,None,:,None]
    v=sample(gray,q);std=np.maximum(v.std(-1),cfg['image_profile_std_floor'])
    left=v[:,:,4]-v[:,:,1];right=v[:,:,11]-v[:,:,8]
    strength=np.clip(np.minimum(abs(left),abs(right))/std/2,0,1)*(left*right<0)
    h,w=gray.shape;inside=((q[...,0]>=0)&(q[...,0]<w-1)&(q[...,1]>=0)&(q[...,1]<h-1)).all(-1)
    strength*=inside;chosen=np.argmax(strength,axis=1)
    return strength[np.arange(len(p)),chosen],widths[chosen]


def thin(mask,iterations):
    z=mask.astype(np.uint8)
    for _ in range(iterations):
        changed=0
        for phase in (0,1):
            p=np.pad(z,1);a=[p[:-2,1:-1],p[:-2,2:],p[1:-1,2:],p[2:,2:],
                p[2:,1:-1],p[2:,:-2],p[1:-1,:-2],p[:-2,:-2]]
            count=sum(a);trans=sum(((a[j]==0)&(a[(j+1)%8]>0)).astype(np.uint8) for j in range(8))
            if phase==0:tri=(a[0]*a[2]*a[4]==0)&(a[2]*a[4]*a[6]==0)
            else:tri=(a[0]*a[2]*a[6]==0)&(a[0]*a[4]*a[6]==0)
            remove=(z>0)&(count>=2)&(count<=6)&(trans==1)&tri
            changed+=int(remove.sum());z[remove]=0
        if not changed:break
    return z.astype(bool)


def _empty_extract(cfg,bbox,shape,tick,reason,diag=None,arrays=None):
    return ({'schema_version':1,'segments':[],'bbox':list(bbox),'original_shape':list(shape),
             'config_sha256':config_hash(cfg),'coordinate_system':'original image pixels',
             'diagnostics':{'reason':reason,'elapsed_seconds':time.perf_counter()-tick,'segments':0,'endpoints':0,**(diag or {})}}, arrays or {})


def extract(gray,response,valid,bbox,cfg=None):
    """Return (JSON-friendly fragment bundle, sparse NumPy derivative arrays).

    The caller may save arrays with np.savez_compressed. Raw response is never
    modified. Work-grid extraction only re-samples the model's original grid.
    """
    cfg={**DEFAULTS,**(cfg or {})};tick=time.perf_counter()
    gray=np.asarray(gray,np.float32);raw=np.asarray(response);valid=np.asarray(valid,bool)&np.isfinite(raw)
    assert gray.shape==raw.shape==valid.shape and gray.ndim==2
    assert np.isfinite(gray).all() and gray.min()>=0 and gray.max()<=1
    x0,y0,x1,y1=map(int,bbox);assert 0<=x0<x1<=gray.shape[1] and 0<=y0<y1<=gray.shape[0]
    scale=cfg['work_scale'];w=round((x1-x0)*scale);h=round((y1-y0)*scale)
    # Area mapping is fixed; nominal half-scale is exact for the established bbox.
    assert abs(w/(x1-x0)-scale)<1e-6 and abs(h/(y1-y0)-scale)<1e-6
    rc=np.nan_to_num(raw[y0:y1,x0:x1],nan=0.).astype(np.float32)
    vc=valid[y0:y1,x0:x1].astype(np.float32)
    r=cv2.resize(rc,(w,h),interpolation=cv2.INTER_AREA);v=cv2.resize(vc,(w,h),interpolation=cv2.INTER_AREA)>=.999
    sigma=cfg['gaussian_sigma_px']*scale;k=2*int(math.ceil(cfg['gaussian_truncate']*sigma))+1
    support=cv2.GaussianBlur(v.astype(np.float32),(k,k),sigma)
    sm=cv2.GaussianBlur(r*v,(k,k),sigma)/np.maximum(support,1e-6)
    distance=cv2.distanceTransform(np.pad(v.astype(np.uint8),1),cv2.DIST_L2,3)[1:-1,1:-1]/scale
    # Explicit crop edges also lack contextual response support, even if all ROI pixels are finite.
    yy,xx=np.indices((h,w),dtype=np.float32)
    guard=np.minimum.reduce([distance,(xx+.5)/scale,(w-.5-xx)/scale,(yy+.5)/scale,(h-.5-yy)/scale])
    eligible=v&(support>.999)&(guard>=cfg['valid_border_guard_px'])
    dxx=cv2.Sobel(sm,cv2.CV_32F,2,0,ksize=3,scale=.25)*scale**2
    dyy=cv2.Sobel(sm,cv2.CV_32F,0,2,ksize=3,scale=.25)*scale**2
    dxy=cv2.Sobel(sm,cv2.CV_32F,1,1,ksize=3,scale=.25)*scale**2
    disc=np.sqrt((dxx-dyy)**2+4*dxy**2);lam=.5*(dxx+dyy-disc)
    normal_angle=.5*np.arctan2(2*dxy,dxx-dyy)+np.pi/2
    nx=np.cos(normal_angle);ny=np.sin(normal_angle)
    def along(offset):
        return (cv2.remap(sm,xx+nx*offset*scale,yy+ny*offset*scale,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT),
                cv2.remap(sm,xx-nx*offset*scale,yy-ny*offset*scale,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT))
    a,b=along(cfg['nms_offset_px']);c,d=along(cfg['peak_contrast_offset_px'])
    contrast=sm-.5*(c+d)
    ridge=eligible&(sm>=cfg['low_response'])&(lam<=-cfg['minimum_negative_curvature'])&(sm>=a)&(sm>=b)&(contrast>=cfg['minimum_peak_contrast'])
    n_before=int(ridge.sum());trunc=[]
    if n_before>cfg['maximum_ridge_pixels']:
        ids=np.flatnonzero(ridge);keep=ids[np.argpartition(sm.ravel()[ids],-cfg['maximum_ridge_pixels'])[-cfg['maximum_ridge_pixels']:]]
        ridge[:]=False;ridge.ravel()[keep]=True;trunc.append('ridge_pixel_budget_response_rank')
    sy,sx=np.nonzero(ridge);points=np.stack([(sx+.5)/scale-.5+x0,(sy+.5)/scale-.5+y0],1)
    direction=np.stack([-ny[sy,sx],nx[sy,sx]],1)
    image_score,_=image_evidence(gray,points,direction,cfg)
    weak_good=(sm[sy,sx]>=cfg['high_response'])|(image_score>=cfg['image_bilateral_min'])
    ridge[sy[~weak_good],sx[~weak_good]]=False
    # Weak pixels survive only in a component anchored by a genuine high peak.
    _,labels=cv2.connectedComponents(ridge.astype(np.uint8),connectivity=8)
    anchor=np.unique(labels[ridge&(sm>=cfg['high_response'])]);anchor=anchor[anchor>0]
    ridge&=np.isin(labels,anchor)
    if not ridge.any():return _empty_extract(cfg,bbox,gray.shape,tick,'no_high_anchored_ridge',{'ridge_before_filter':n_before},
        {'working_shape':np.array([h,w],np.int32),'filtered_response':np.where(v,sm,np.nan).astype(np.float32),
         'candidate_mask':ridge,'valid_working':v,'ridge_yx_working':np.empty((0,2),np.int16),
         'ridge_strength':np.empty(0,np.float32),'ridge_response':np.empty(0,np.float16),'ridge_normal_xy':np.empty((0,2),np.float16)})
    if time.perf_counter()-tick>cfg['maximum_extract_seconds']:
        return _empty_extract(cfg,bbox,gray.shape,tick,'extract_time_budget_before_chains')
    skeleton=thin(ridge,cfg['thinning_iterations']);ys,xs=np.nonzero(skeleton);n=len(xs)
    idsmap=np.full((h,w),-1,np.int32);idsmap[ys,xs]=np.arange(n)
    directions=np.stack([-ny[ys,xs],nx[ys,xs]],1)
    neighbors=np.full((n,8),-1,np.int32)
    offsets=[(-1,-1),(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0)]
    dotmin=math.cos(math.radians(cfg['pixel_link_tangent_degrees']));pairmin=math.cos(math.radians(cfg['direction_link_degrees']))
    for j,(dx,dy) in enumerate(offsets):
        qx=xs+dx;qy=ys+dy;inside=(qx>=0)&(qx<w)&(qy>=0)&(qy<h);ix=np.flatnonzero(inside)
        nb=idsmap[qy[ix],qx[ix]];keep=nb>=0;ix=ix[keep];nb=nb[keep]
        u=np.array([dx,dy])/math.hypot(dx,dy)
        ok=(abs(directions[ix]@u)>=dotmin)&(abs(directions[nb]@u)>=dotmin)&(abs((directions[ix]*directions[nb]).sum(1))>=pairmin)
        neighbors[ix[ok],j]=nb[ok]
    degree=(neighbors>=0).sum(1);visited=set();chains=[];loops=0
    starts=np.r_[np.flatnonzero(degree!=2),np.flatnonzero(degree==2)]
    for start in starts:
        if len(chains)>=cfg['maximum_raw_chains'] or time.perf_counter()-tick>cfg['maximum_extract_seconds']:
            trunc.append('raw_chain_or_extract_time_budget');break
        for first in neighbors[start]:
            if len(chains)>=cfg['maximum_raw_chains']:
                trunc.append('raw_chain_budget');break
            if first<0:continue
            edge=min(int(start),int(first))*n+max(int(start),int(first))
            if edge in visited:continue
            visited.add(edge);chain=[int(start),int(first)];previous=int(start);current=int(first)
            while degree[current]==2 and len(chain)<cfg['maximum_chain_nodes']:
                options=neighbors[current];nxt=int(options[(options>=0)&(options!=previous)][0])
                e=min(current,nxt)*n+max(current,nxt)
                if e in visited:break
                visited.add(e);chain.append(nxt);previous,current=current,nxt
            if len(chain)==cfg['maximum_chain_nodes']:trunc.append('individual_chain_node_budget')
            q=np.stack([(xs[chain]+.5)/scale-.5+x0,(ys[chain]+.5)/scale-.5+y0],1)
            length=float(np.linalg.norm(np.diff(q,axis=0),axis=1).sum())
            if length>=cfg['minimum_segment_length_px']:
                chains.append((q,length,float(sm[ys[chain],xs[chain]].mean())))
    chains.sort(key=lambda row:row[2]*math.sqrt(row[1]),reverse=True)
    if len(chains)>cfg['maximum_segments']:trunc.append('segment_budget_response_length_rank')
    chains=chains[:cfg['maximum_segments']]
    segments=[];raw_finite=np.nan_to_num(raw,nan=0.).astype(np.float32)
    # Resampled points are subpixel original coordinates; no broadening of the response.
    for ci,(q,length,_) in enumerate(chains):
        if time.perf_counter()-tick>cfg['maximum_extract_seconds']:
            trunc.append('segment_evidence_extract_time_budget');break
        p=resample(q,cfg['resample_step_px']);td=tangents(p);ims,width=image_evidence(gray,p,td,cfg)
        rv=sample(raw_finite,p)
        work=(p-np.array([x0,y0])+.5)*scale-.5
        cv=sample(contrast,work);lv=sample(lam,work);band=np.sqrt(np.maximum(rv,.01)/np.maximum(-lv,1e-6))
        states=np.where(rv>=cfg['high_response'],'heatmap_supported',np.where((rv>=cfg['low_response'])&(ims>=cfg['image_bilateral_min']),'image_supported','ambiguous'))
        if np.mean(states!='ambiguous')<cfg['minimum_segment_supported_fraction']:continue
        turns=np.degrees(np.arccos(np.clip((td[1:]*td[:-1]).sum(1),-1,1)))
        segments.append({'id':f's{len(segments)}','points_xy':np.round(p,3).tolist(),'point_evidence_state':states.tolist(),
            'response':np.round(rv,5).tolist(),'image_support':np.round(ims,5).tolist(),
            'length_px':length,'endpoint_tangent_xy':[unit(p[min(3,len(p)-1)]-p[0]).tolist(),unit(p[-1]-p[max(0,len(p)-4)]).tolist()],
            'center_response_mean':float(rv.mean()),'center_response_p10':float(np.quantile(rv,.1)),
            'normal_peak_contrast_mean':float(cv.mean()),'image_supported_fraction':float(np.mean(ims>=cfg['image_bilateral_min'])),
            'response_sigma_proxy_px':float(np.median(np.clip(band,1,30))),
            'optical_profile_width_proxy_px':float(np.median(width)),
            'curvature_turn_p95_degrees':float(np.quantile(turns,.95)) if len(turns) else 0.,
            'near_calculation_edge':bool(np.min(sample(guard,work))<cfg['valid_border_guard_px']*2),
            'widths_are_not_physical_rope_diameter':True})
    arrays={'working_shape':np.array([h,w],np.int32),'filtered_response':np.where(v,sm,np.nan).astype(np.float32),
            'candidate_mask':ridge,'valid_working':v,'ridge_yx_working':np.stack([ys,xs],1).astype(np.int16),
            'ridge_strength':contrast[ys,xs].astype(np.float32),'ridge_response':sm[ys,xs].astype(np.float16),
            'ridge_normal_xy':np.stack([nx[ys,xs],ny[ys,xs]],1).astype(np.float16)}
    bundle={'schema_version':1,'segments':segments,'bbox':list(bbox),'original_shape':list(gray.shape),
        'working_shape':[h,w],'work_scale':scale,'work_to_original':'(working_xy + 0.5) / scale - 0.5 + bbox_origin',
        'config_sha256':config_hash(cfg),'coordinate_system':'original image pixels',
        'diagnostics':{'elapsed_seconds':time.perf_counter()-tick,'ridge_before_filter':n_before,'ridge_after_thinning':n,
            'high_anchored_components':len(anchor),'raw_accepted_chains':len(chains),'segments':len(segments),'endpoints':2*len(segments),
            'budget_events':sorted(set(trunc)),'gaussian_sigma_original_px':cfg['gaussian_sigma_px'],'gaussian_sigma_work_px':sigma,
            'gaussian_kernel_work_px':k,'all_thresholds_shared_AB':True,'true_zero_gray_retained':True,
            'labels_or_seed_used_in_extraction':False,'response_unchanged':True}}
    return bundle,arrays


def build_graph(bundle,gray,response,valid,cfg):
    tick=time.perf_counter();segments=bundle['segments'];endpoints=[]
    raw=np.nan_to_num(response,nan=0.).astype(np.float32);vf=(np.asarray(valid,bool)&np.isfinite(response)).astype(np.float32)
    for si,s in enumerate(segments):
        p=np.asarray(s['points_xy']);ts=np.asarray(s['endpoint_tangent_xy'])
        for side in (0,1):endpoints.append({'segment':si,'side':side,'xy':p[0 if side==0 else -1],
            'into':ts[0] if side==0 else -ts[1]})
    cell=cfg['maximum_gap_px'];buckets=defaultdict(list)
    for i,e in enumerate(endpoints):buckets[tuple(np.floor(e['xy']/cell).astype(int))].append(i)
    records=[];provisional=[];budget=[];tests=0
    for i,a in enumerate(endpoints):
        if tests>=cfg['maximum_graph_edge_tests'] or time.perf_counter()-tick>cfg['maximum_graph_seconds']:
            budget.append('graph_tests_or_time_budget');break
        key=np.floor(a['xy']/cell).astype(int);near=[]
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for j in buckets.get(tuple(key+[dx,dy]),[]):
                    b=endpoints[j]
                    if b['segment']==a['segment']:continue
                    dist=float(np.linalg.norm(b['xy']-a['xy']))
                    if dist<=cell:near.append((dist,j))
        near.sort()
        for dist,j in near[:cfg['maximum_near_endpoint_tests']]:
            if tests>=cfg['maximum_graph_edge_tests']:
                budget.append('graph_edge_test_budget');break
            tests+=1;b=endpoints[j];out=-a['into'];incoming=b['into'];delta=b['xy']-a['xy'];u=unit(delta)
            turn=angle(out,incoming);alignment=max(angle(out,u),angle(incoming,u)) if dist>1 else turn
            lateral=max(abs(out[0]*delta[1]-out[1]*delta[0]),abs(incoming[0]*delta[1]-incoming[1]*delta[0]))
            sa,sb=segments[a['segment']],segments[b['segment']]
            widthratio=max(sa['response_sigma_proxy_px'],sb['response_sigma_proxy_px'])/max(min(sa['response_sigma_proxy_px'],sb['response_sigma_proxy_px']),1.)
            optical=max(sa['optical_profile_width_proxy_px'],sb['optical_profile_width_proxy_px'])/max(min(sa['optical_profile_width_proxy_px'],sb['optical_profile_width_proxy_px']),1.)
            reason=None
            if turn>cfg['maximum_turn_degrees']:reason='endpoint_tangent_turn'
            elif alignment>cfg['maximum_gap_tangent_degrees']:reason='gap_not_forward_aligned'
            elif lateral>cfg['maximum_lateral_offset_px']:reason='lateral_jump'
            elif widthratio>cfg['maximum_bandwidth_ratio'] or optical>cfg['maximum_optical_width_ratio']:reason='local_width_proxy_incompatible'
            rec={'id':f'e{len(records)}','from_endpoint':i,'to_endpoint':j,'from_segment':sa['id'],'to_segment':sb['id'],
                 'endpoints_xy':[a['xy'].tolist(),b['xy'].tolist()],'length_px':dist,'direction_difference_degrees':turn,
                 'gap_alignment_degrees':alignment,'lateral_offset_px':float(lateral),'response_bandwidth_ratio':widthratio,
                 'optical_width_ratio':optical,'accepted':False,'reason':reason,'bridged':dist>cfg['direct_endpoint_distance_px']}
            records.append(rec)
            if reason is None:
                p=resample([a['xy'],b['xy']],cfg['resample_step_px']) if dist>1e-8 else np.array([a['xy']])
                provisional.append((len(records)-1,p,np.tile(unit(out+incoming),(len(p),1))))
    if provisional and time.perf_counter()-tick>cfg['maximum_graph_seconds']:
        budget.append('graph_time_budget_before_evidence')
        for ri,_,_ in provisional:records[ri]['reason']='graph_time_budget_unscored'
        provisional=[]
    if provisional:
        sizes=[len(p) for _,p,_ in provisional];pts=np.concatenate([p for _,p,_ in provisional]);dirs=np.concatenate([d for _,_,d in provisional])
        rr=sample(raw,pts);vv=sample(vf,pts);ii,_=image_evidence(gray,pts,dirs,cfg);offset=0
        if time.perf_counter()-tick>cfg['maximum_graph_seconds']:budget.append('graph_evidence_atomic_time_budget')
        for pi,((ri,p,_),count) in enumerate(zip(provisional,sizes)):
            if pi%128==0 and time.perf_counter()-tick>cfg['maximum_graph_seconds']:
                budget.append('graph_time_budget_during_edge_decisions')
                for remaining,_,_ in provisional[pi:]:records[remaining]['reason']='graph_time_budget_unscored'
                break
            rec=records[ri];rs=rr[offset:offset+count];vs=vv[offset:offset+count];ims=ii[offset:offset+count];offset+=count
            rec.update(heatmap_mean=float(rs.mean()),heatmap_min=float(rs.min()),image_support_mean=float(ims.mean()),
                       valid_fraction=float((vs>.999).mean()),points_xy=np.round(p,3).tolist())
            if not np.all(vs>.999):rec['reason']='outside_computed_response_domain'
            elif rs.mean()<cfg['minimum_gap_response'] and ims.mean()<cfg['minimum_gap_image_support']:rec['reason']='gap_has_no_heatmap_or_image_support'
            else:
                rec['accepted']=True;rec['reason']='short_direction_width_and_local_evidence_compatible'
                rec['score']=float(cfg['response_weight']*rs.mean()+cfg['image_weight']*ims.mean()-cfg['gap_penalty']*rec['length_px']/cfg['maximum_gap_px']-
                    cfg['lateral_penalty']*rec['lateral_offset_px']/cfg['maximum_lateral_offset_px']-cfg['curvature_penalty']*rec['direction_difference_degrees']/cfg['maximum_turn_degrees'])
    adjacency=defaultdict(list)
    for i,rec in enumerate(records):
        if rec['accepted']:adjacency[rec['from_endpoint']].append(i)
    for key,values in adjacency.items():
        values.sort(key=lambda i:records[i]['score'],reverse=True)
        for i in values[cfg['maximum_neighbors_per_endpoint']:]:records[i].update(accepted=False,reason='endpoint_neighbor_budget_lower_score')
        adjacency[key]=values[:cfg['maximum_neighbors_per_endpoint']]
    return endpoints,records,adjacency,{'elapsed_seconds':time.perf_counter()-tick,'edge_tests':tests,
        'accepted_edges':sum(r['accepted'] for r in records),'reason_counts':dict(Counter(r['reason'] for r in records)),
        'budget_events':budget,'atomic_budget_overrun_seconds':max(0.,time.perf_counter()-tick-cfg['maximum_graph_seconds']),
        'budget_check_policy':'checked between bounded vector operations and every128 edge decisions','labels_or_seed_used':False}


def _path_rank(s,cfg):
    observed=max(s['observed_length'],1.)
    return s['evidence_sum']/observed-cfg['gap_penalty']*s['bridge_length']/max(s['new_length'],1.)+cfg['length_reward']*min(s['new_length']/1600.,1.)
def _path_distance(a,b):
    a=np.asarray(a);b=np.asarray(b);da=np.r_[0,np.cumsum(np.linalg.norm(np.diff(a,axis=0),axis=1))];db=np.r_[0,np.cumsum(np.linalg.norm(np.diff(b,axis=0),axis=1))]
    m=min(da[-1],db[-1]);t=np.linspace(max(0,m-1200),m,30)
    aa=np.stack([np.interp(t,da,a[:,0]),np.interp(t,da,a[:,1])],1);bb=np.stack([np.interp(t,db,b[:,0]),np.interp(t,db,b[:,1])],1)
    return float(np.median(np.linalg.norm(aa-bb,axis=1)))


def connect_segments(bundle,gray,response,valid,seed,scene,bbox,cfg=None):
    """Return (old-path-compatible candidates, selected id, metadata).

    segment_status (N-1) is authoritative. point_status maps bridge to inferred
    ONLY for old plotting compatibility; no real occlusion inference is claimed.
    """
    cfg={**DEFAULTS,**(cfg or {})};tick=time.perf_counter();segments=bundle['segments'];pseed=np.asarray(seed,float)
    if len(pseed)<2 or np.linalg.norm(pseed[-1]-pseed[0])<1:return [],None,{'reason':'entry_unavailable','entry_available':False}
    endpoints,edges,adj,graphdiag=build_graph(bundle,gray,response,valid,cfg)
    starts=[];start_records=[];direction=unit(pseed[-1]-pseed[0]);tip=pseed[-1];raw=np.nan_to_num(response,nan=0.).astype(np.float32)
    for si,s in enumerate(segments):
        p=np.asarray(s['points_xy']);dist=np.linalg.norm(p-tip,axis=1);k=int(np.argmin(dist))
        if dist[k]>cfg['entry_attach_distance_px']:continue
        for orientation in (1,-1):
            q=p[k:] if orientation==1 else p[:k+1][::-1]
            if len(q)<2:continue
            td=unit(q[min(3,len(q)-1)]-q[0]);turn=angle(direction,td);delta=q[0]-tip
            align=max(angle(direction,delta),angle(td,delta)) if dist[k]>cfg['entry_direct_projection_px'] else 0.
            lateral=max(abs(direction[0]*delta[1]-direction[1]*delta[0]),abs(td[0]*delta[1]-td[1]*delta[0]))
            reason='entry_position_direction_gap_compatible'
            if turn>cfg['entry_tangent_degrees']:reason='entry_fragment_tangent_incompatible'
            elif dist[k]>cfg['entry_direct_projection_px'] and align>cfg['maximum_gap_tangent_degrees']:reason='entry_gap_not_forward_aligned'
            elif lateral>cfg['maximum_lateral_offset_px']:reason='entry_lateral_jump'
            accepted=reason=='entry_position_direction_gap_compatible'
            start_records.append({'segment_id':s['id'],'orientation':orientation,'gap_px':float(dist[k]),
                'entry_fragment_turn_degrees':turn,'gap_alignment_degrees':align,'lateral_offset_px':float(lateral),
                'accepted_geometry':accepted,'reason':reason})
            if accepted:starts.append((float(dist[k])-.1*s['center_response_mean'],si,k,orientation))
    starts.sort();starts=starts[:cfg['maximum_start_candidates']]
    seedpoints=resample(pseed,cfg['resample_step_px']);edgepoints=np.asarray(scene.get('target_edge_points_xy',[]),float).reshape(-1,2)
    def append_segment(state,si,orientation,start_index=None,bridge=None):
        s=segments[si];pp=np.asarray(s['points_xy']);ss=s['point_evidence_state'];rr=np.array(s['response']);ii=np.array(s['image_support'])
        indices=np.arange(len(pp)) if orientation==1 else np.arange(len(pp)-1,-1,-1)
        if start_index is not None:indices=indices[np.flatnonzero(indices==start_index)[0]:]
        pp=pp[indices];states=[ss[j] for j in indices];rr=rr[indices];ii=ii[indices]
        ns={**state,'points':list(state['points']),'statuses':list(state['statuses']),'point_evidence':list(state['point_evidence']),
            'segments':list(state['segments']),'edge_ids':list(state['edge_ids']),'bridges':list(state['bridges'])}
        if bridge is not None:
            bp=np.asarray(bridge['points_xy']);bl=float(bridge['length_px']);isbridge=bl>cfg['direct_endpoint_distance_px']
            if isbridge:
                ns['bridge_length']+=bl;ns['bridge_count']+=1;ns['bridges'].append(bridge)
            for q in bp[1:]:
                ns['points'].append(q.tolist());ns['statuses'].append('bridge' if isbridge else 'heatmap_supported')
                ns['point_evidence'].append({'evidence_state':'bridged' if isbridge else 'heatmap_supported','bridge_edge_id':bridge['id'],
                    'response':bridge.get('heatmap_mean'),'image_support':bridge.get('image_support_mean')})
            ns['new_length']+=bl;ns['edge_ids'].append(bridge['id'])
        if np.linalg.norm(np.asarray(ns['points'][-1])-pp[0])>1e-4:
            return None
        for j in range(1,len(pp)):
            length=float(np.linalg.norm(pp[j]-pp[j-1]));status=states[j]
            ns['points'].append(pp[j].tolist());ns['statuses'].append(status)
            ns['point_evidence'].append({'evidence_state':status,'response':float(rr[j]),'image_support':float(ii[j]),'segment_id':s['id']})
            ns['new_length']+=length;ns['observed_length']+=length
            ns['evidence_sum']+=length*(cfg['response_weight']*rr[j]+cfg['image_weight']*ii[j])
            if status in ('heatmap_supported','image_supported'):ns['supported_length']+=length
            if ns['new_length']>=cfg['maximum_path_length_px']:
                ns['reason']='path_length_budget';break
            if len(edgepoints) and ns['new_length']>=cfg['edge_minimum_new_length_px'] and np.min(np.linalg.norm(edgepoints-pp[j],axis=1))<=cfg['edge_arrival_distance_px']:
                ns['reason']='observed_edge_candidate_reached_identity_unconfirmed';break
        ns['segments'].append(si);ns['endpoint']=2*si+(1 if orientation==1 else 0)
        return ns
    beam=[];finished=[]
    for _,si,k,orientation in starts:
        p=np.asarray(segments[si]['points_xy']);gap=resample([tip,p[k]],cfg['resample_step_px'])
        rv=sample(raw,gap);validgap=sample(np.asarray(valid,np.float32),gap)
        ims,_=image_evidence(gray,gap,np.tile(direction,(len(gap),1)),cfg)
        if not np.all(validgap>.999) or (rv.mean()<cfg['minimum_gap_response'] and ims.mean()<cfg['minimum_gap_image_support']):continue
        bridge={'id':f'entry_to_s{si}_{orientation}','points_xy':gap.tolist(),'length_px':float(np.linalg.norm(tip-p[k])),
                'reason':'entry_local_position_direction_and_evidence','heatmap_mean':float(rv.mean()),'image_support_mean':float(ims.mean())}
        state={'points':seedpoints.tolist(),'statuses':['entry_condition']*(len(seedpoints)-1),
            'point_evidence':[{'evidence_state':'entry_condition'}]*len(seedpoints),'segments':[],'edge_ids':[],
            'bridge_length':0.,'bridge_count':0,'bridges':[],'new_length':0.,'observed_length':0.,'supported_length':0.,'evidence_sum':0.,'reason':None}
        n=append_segment(state,si,orientation,k,bridge)
        if n is not None:
            (finished if n['reason'] else beam).append(n)
    expansions=0;searchstart=time.perf_counter();search_budget=None
    for depth in range(cfg['maximum_path_segments']):
        pool=[]
        for state in beam:
            if len(state['segments'])>=cfg['maximum_path_segments']:
                finished.append({**state,'reason':'path_segment_budget'});continue
            any_child=False
            for ei in adj.get(state['endpoint'],[]):
                edge=edges[ei];dest=endpoints[edge['to_endpoint']];si=dest['segment']
                if si in state['segments']:continue
                bl=edge['length_px'] if edge['bridged'] else 0.
                if state['bridge_length']+bl>cfg['maximum_total_bridge_px'] or state['bridge_count']+int(edge['bridged'])>cfg['maximum_bridge_count']:continue
                if expansions>=cfg['maximum_expansions'] or time.perf_counter()-searchstart>cfg['maximum_search_seconds']:
                    search_budget='expansion_or_search_time_budget';break
                expansions+=1
                n=append_segment(state,si,1 if dest['side']==0 else -1,bridge=edge)
                if n is not None:
                    any_child=True;(finished if n['reason'] else pool).append(n)
            if not any_child:
                n={**state,'reason':state['reason'] or 'fragment_graph_end_no_admissible_short_gap'};finished.append(n)
            if search_budget:break
        if search_budget:break
        pool.sort(key=lambda s:_path_rank(s,cfg),reverse=True);beam=pool[:cfg['beam_width']]
        if len(finished)>120:finished=sorted(finished,key=lambda s:_path_rank(s,cfg),reverse=True)[:120]
        if not beam:break
    for state in beam:
        finished.append({**state,'reason':search_budget or 'path_segment_budget'})
    finished.sort(key=lambda s:_path_rank(s,cfg),reverse=True);chosen=[]
    for state in finished:
        if all(_path_distance(state['points'],o['points'])>=cfg['distinct_path_distance_px'] for o in chosen):chosen.append(state)
        if len(chosen)>=cfg['final_candidate_count']:break
    margin=_path_rank(chosen[0],cfg)-_path_rank(chosen[1],cfg) if len(chosen)>1 else None
    rows=[]
    for j,state in enumerate(chosen):
        p=np.asarray(state['points']);statuses=list(state['statuses']);stages=stage_at_points(scene,p);dist=np.linalg.norm(np.diff(p,axis=0),axis=1)
        # Keep local evidence labels; unresolved path competition is a separate flag,
        # not a retroactive claim that all its pixels lacked local observations.
        precise=['entry_condition']+[('bridged' if s=='bridge' else s) for s in statuses]
        compatible=[('visible_supported' if s in ('heatmap_supported','image_supported') else 'inferred' if s=='bridged' else s) for s in precise]
        length_by={s:float(dist[np.array(statuses)==s].sum()) for s in ('entry_condition','heatmap_supported','image_supported','bridge','ambiguous')}
        attached=(stages[1:]>=3)&np.isin(statuses,['heatmap_supported','image_supported'])
        attached_image=attached&np.array([float(e.get('image_support') or 0)>=cfg['image_bilateral_min'] for e in state['point_evidence'][1:]])
        rows.append({'id':f'c{j}','points_xy':p.tolist(),'point_status':compatible,'point_evidence_state':precise,
            'segment_status':statuses,'stage':stages.tolist(),'evidence':state['point_evidence'],
            'segment_ids':[segments[i]['id'] for i in state['segments']],'graph_edge_ids':state['edge_ids'],'bridges':state['bridges'],
            'rank_score':_path_rank(state,cfg),'mean_evidence_score':state['evidence_sum']/max(state['observed_length'],1.),
            'termination':state['reason'],'length_px':float(dist.sum()),'evidence_length_px':length_by,
            'status_length_px':{'entry_condition':length_by['entry_condition'],'visible_supported':length_by['heatmap_supported']+length_by['image_supported'],
                'ambiguous':length_by['ambiguous'],'inferred':length_by['bridge']},
            'bridge_length_px':length_by['bridge'],'bridge_fraction_excluding_entry':length_by['bridge']/max(state['new_length'],1.),
            'supported_new_length_px':state['supported_length'],
            'stage_length_px':{str(s):float(dist[stages[1:]==s].sum()) for s in range(1,5)},
            'attached_side_image_support_fraction':float(dist[attached_image].sum()/max(dist[stages[1:]>=3].sum(),1.)),
            'attached_side_supported_fraction':float(dist[attached].sum()/max(dist[stages[1:]>=3].sum(),1.)),
            'reached_edge_candidate':state['reason']=='observed_edge_candidate_reached_identity_unconfirmed',
            'first_competitor_divergence_index':None,'competition_ambiguous':margin is not None and margin<cfg['selection_score_margin'],
            'full_target_identity_confirmed':False,'bridge_is_not_occlusion_recovery':True})
    selected=None;why='no_entry_compatible_fragment'
    if rows:
        top=rows[0]
        if top['supported_new_length_px']<cfg['minimum_new_supported_length_px']:why='insufficient_new_supported_length'
        elif top['bridge_fraction_excluding_entry']>cfg['maximum_selected_bridge_fraction']:why='too_much_bridge'
        elif margin is not None and margin<cfg['selection_score_margin']:why='competing_fragment_routes_close_score'
        else:selected='c0';why='provisional_local_evidence_route_not_identity_confirmation'
    meta={'entry_available':True,'elapsed_seconds':time.perf_counter()-tick,'graph':graphdiag,'graph_edges':edges,
        'start_candidates':len(starts),'entry_candidate_checks':start_records,'search_expansions':expansions,'search_budget_event':search_budget,
        'score_margin':margin,'selection_reason':why,'rejection_reason':None if selected else why,
        'config_sha256':config_hash(cfg),'coordinate_system':'original image pixels','labels_used':False,
        'compatibility_mapping':'heatmap/image_supported -> visible_supported; bridge -> inferred only for old renderer, NOT real occlusion',
        'authoritative_status':'segment_status is N-1; do not join support across bridge/ambiguous',
        'bounds':{k:cfg[k] for k in ('maximum_segments','maximum_neighbors_per_endpoint','maximum_graph_edge_tests','maximum_expansions',
            'maximum_gap_px','maximum_total_bridge_px','maximum_bridge_count','maximum_extract_seconds','maximum_graph_seconds','maximum_search_seconds')}}
    return rows,selected,meta


def checks():
    cfg={**DEFAULTS,'minimum_segment_length_px':20.};h,w=220,700;yy,xx=np.indices((h,w))
    response=np.exp(-.5*((yy-110)/3.)**2).astype(np.float32);response[:,330:346]=0
    gray=np.full((h,w),.65,np.float32);gray[100:121]=.2;valid=np.ones((h,w),bool)
    scene={'entry_origin_xy':[20,110],'entry_direction_xy':[1,0],'contact_proxy':{'t_from_entry_px':220},
        'target_edge_points_xy':[[650,110]],'parameters':{'contact_band_px':50,'edge_band_px':30}}
    bundle,sparse=extract(gray,response,valid,[0,0,w,h],cfg)
    rows,selected,meta=connect_segments(bundle,gray,response,valid,[[20,110],[140,110]],scene,[0,0,w,h],cfg)
    assert bundle['segments'] and rows
    assert any(r['bridge_length_px']>0 for r in rows)
    assert all(len(r['segment_status'])==len(r['points_xy'])-1 for r in rows)
    assert all(len(r['point_status'])==len(r['points_xy'])==len(r['point_evidence_state']) for r in rows)
    assert len(rows)<=3 and all(r['bridge_length_px']<=cfg['maximum_total_bridge_px']+1e-3 for r in rows)
    zero_gray=np.zeros_like(gray);zero_valid=valid.copy()
    b0,_=extract(zero_gray,response,zero_valid,[0,0,w,h],cfg);assert b0['segments'],'true gray zero cannot invalidate heatmap-supported ridge'
    nan_response=response.copy();nan_response[:]=np.nan
    be,_=extract(gray,nan_response,valid,[0,0,w,h],cfg);assert not be['segments']
    return {'status':'passed','straight_fragments_extracted':len(bundle['segments']),'short_gap_bridge_recorded':True,
        'candidate_count':len(rows),'selected':selected,'authoritative_segment_status_aligned':True,
        'true_zero_gray_not_invalid':True,'nan_response_not_observation':True,'maximum_three_candidates':True,
        'bridge_total_bounded':True,'no_label_access':True}


if __name__=='__main__':print(json.dumps(checks(),ensure_ascii=False))
