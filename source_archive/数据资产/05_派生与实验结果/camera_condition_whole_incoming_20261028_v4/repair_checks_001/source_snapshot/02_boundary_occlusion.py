"""IR-A4 sampling revision 2: measure original boundary evidence from entry onward.

The saved F3 mask can contain deliberate exclusions. Its holes NEVER count as
physical occlusion evidence. Curve references propose sampling locations only;
all continuity/interruption scores are remeasured from unpainted original RGB.
Image-row candidates are NOT verified material identity, contact, t1 or t2.
Pure NumPy/OpenCV; no file access, detector import, hardware or global state.
"""
from __future__ import annotations
import math
from collections import Counter
import cv2
import numpy as np

PARAMETERS={
    'maximum_candidates':12,'maximum_rows':64,'maximum_extrapolation_pitch':6.,
    'local_fit_span_pitch':4.,'normal_profile_half_pitch':.60,
    'valley_search_pitch':.18,'minimum_groove_width_pitch':.12,
    'maximum_groove_width_pitch':.65,'minimum_raw_range':3.,
    'minimum_raw_weak_contrast':1.0,'minimum_raw_strong_contrast':2.0,
    'minimum_normalized_contrast':.12,'minimum_orientation_ratio':.50,
    'maximum_saturation_fraction':.65,'minimum_region_samples':3,
    'supported_fraction':.55,'interrupted_fraction':.25,
    'minimum_available_fraction':.65,'normalization_sigma_pitch':2.0,
}


def _at(a,xy,x0=0,y0=0,outside=0):
    q=np.rint(xy).astype(int);x=q[:,0]-x0;y=q[:,1]-y0
    valid=(x>=0)&(y>=0)&(x<a.shape[1])&(y<a.shape[0]);out=np.full(len(q),outside,dtype=a.dtype);out[valid]=a[y[valid],x[valid]];return out


def _sample(a,xy,x0=0,y0=0):
    q=np.asarray(xy,np.float32);shape=q.shape[:-1]
    xx=(q[...,0]-x0).reshape(1,-1);yy=(q[...,1]-y0).reshape(1,-1)
    return cv2.remap(a,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101).reshape(shape)


def _geometry(free,pitch):
    # Canonical names are origin_xy/direction_xy/width_px/visible_end_xy.
    # Aliases allow existing free-span adapters without changing old code.
    origin=free.get('origin_xy',free.get('center_xy'))
    direction=free.get('direction_xy',free.get('toward_surface_direction_xy',free.get('direction')))
    width=free.get('width_px',free.get('optical_width_px'))
    if direction is not None and origin is None and free.get('observed_segment_xy') is not None:
        segment=np.asarray(free['observed_segment_xy'],float).reshape(-1,2);dv=np.asarray(direction,float).reshape(2)
        origin=segment[int(np.argmax(segment@dv))]
    if origin is None or direction is None or width is None:raise ValueError('free geometry requires direction and width plus origin or observed_segment_xy')
    origin=np.asarray(origin,float).reshape(2);direction=np.asarray(direction,float).reshape(2);width=float(width)
    if not np.isfinite(np.r_[origin,direction,width]).all() or np.linalg.norm(direction)<1e-6 or width<=0:raise ValueError('invalid free geometry')
    direction/=np.linalg.norm(direction);normal=np.array([-direction[1],direction[0]])
    end=np.asarray(free.get('visible_end_xy',origin),float).reshape(2)
    return origin,direction,normal,width,end


def _endpoint_poly(xy,x,pitch):
    take=abs(xy[:,0]-x)<=PARAMETERS['local_fit_span_pitch']*pitch;q=xy[take]
    if len(q)<4:q=xy[np.argsort(abs(xy[:,0]-x))[:min(12,len(xy))]]
    if len(q)<3 or np.ptp(q[:,0])<2:return None
    coef=np.polyfit((q[:,0]-x)/pitch,q[:,1],2)
    # The extension must start on the actual stored endpoint, not a regression
    # intercept a few pixels away. Interpolation inside the span remains exact.
    coef[2]=float(np.interp(x,xy[:,0],xy[:,1]));return coef


def _curve_model(xy,x,pitch):
    local=_endpoint_poly(xy,x,pitch)
    left=_endpoint_poly(xy,float(xy[0,0]),pitch);right=_endpoint_poly(xy,float(xy[-1,0]),pitch)
    if local is None or left is None or right is None:return None
    return {'x0':float(x),'scale':float(pitch),'coef':local,'xy':xy,'left_coef':left,'right_coef':right}


def _eval(model,x):
    x=np.asarray(x,float);q=model['xy'];pitch=model['scale'];v=np.interp(x,q[:,0],q[:,1])
    left=x<q[0,0];right=x>q[-1,0]
    v[left]=np.polyval(model['left_coef'],(x[left]-q[0,0])/pitch)
    v[right]=np.polyval(model['right_coef'],(x[right]-q[-1,0])/pitch)
    limit=PARAMETERS['maximum_extrapolation_pitch']*pitch
    v[(x<q[0,0]-limit)|(x>q[-1,0]+limit)]=np.nan
    return v


def _same_observed_row(a,b,pitch):
    if a['reference_id']==b['reference_id']:return True
    qa=a['model']['xy'];qb=b['model']['xy'];lo=max(qa[0,0],qb[0,0]);hi=min(qa[-1,0],qb[-1,0])
    if hi-lo<pitch:return False
    xs=np.linspace(lo,hi,min(80,max(6,int((hi-lo)/3))))
    ya=np.interp(xs,qa[:,0],qa[:,1]);yb=np.interp(xs,qb[:,0],qb[:,1]);delta=ya-yb
    return bool(np.median(abs(delta))<.25*pitch and np.percentile(abs(delta),90)<.40*pitch and np.median(abs(np.gradient(ya,xs)-np.gradient(yb,xs)))<.25)


def _entry_interval(surface,origin,direction,width,pitch,end_s):
    """Locate a sustained entry into the largest search component, not contact.

    The previously extended straight endpoint can lie hundreds of pixels past
    the first winding-surface crossing. It is a measurement endpoint, never a
    valid lower bound for the interruption study. A three-ray strip suppresses
    isolated mask islands; the main component is only a bounded search domain.
    """
    binary=(surface>0).astype(np.uint8)
    count,labels,stats,_=cv2.connectedComponentsWithStats(binary,8)
    if count<=1:
        return end_s-2*pitch,{'status':'no_search_component','entry_s_px':None,'old_lower_s_px':end_s-2*pitch,'lower_s_px':end_s-2*pitch}
    main=(labels==1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))).astype(np.uint8)
    step=max(2.,min(5.,pitch/6.));ss=np.arange(-2*pitch,float(np.hypot(*surface.shape)),step)
    normal=np.array([-direction[1],direction[0]])
    q=origin+ss[:,None,None]*direction+np.array([-.3,0.,.3])[None,:,None]*width*normal
    h,w=surface.shape
    inside_image=(q[...,0]>=0)&(q[...,0]<w)&(q[...,1]>=0)&(q[...,1]<h)
    values=_at(main,q.reshape(-1,2)).reshape(-1,3)>0
    good=(values.mean(axis=1)>=2/3)&inside_image.all(axis=1)
    sustained=max(3,int(math.ceil(max(pitch,width)/step)))
    runs=np.convolve(good.astype(np.int32),np.ones(sustained,np.int32),'valid')
    take=np.flatnonzero(runs==sustained)
    if not len(take):
        return end_s-2*pitch,{'status':'no_sustained_entry','entry_s_px':None,'old_lower_s_px':end_s-2*pitch,'lower_s_px':end_s-2*pitch,'ray_step_px':step}
    entry=float(ss[take[0]]);lower=entry-2*pitch
    return lower,{'status':'search_component_entry','entry_s_px':entry,'entry_xy':(origin+entry*direction).tolist(),'old_lower_s_px':end_s-2*pitch,'lower_s_px':lower,'visible_end_s_px':end_s,'ray_step_px':step,'sustained_length_px':sustained*step,'main_component_pixels':int(np.count_nonzero(main)),'meaning':'search-domain entry, not measured physical contact'}


def _probe_family(kind):
    return 'approach' if kind=='tangential_strip_approach' else ('extrapolated_crossing' if kind=='finite_reference_extrapolation' else 'crossing')


def _fair_probe_budget(groups,budget,end_s):
    """Round-robin independent row/family queues before secondary probes."""
    queues={}
    for group in groups:
        key=(group[0]['visual_row_id'],_probe_family(group[0]['geometry_kind']))
        queues.setdefault(key,[]).append(group)
    for queue in queues.values():queue.sort(key=lambda g:(abs(g[0]['free_s']-end_s),g[0]['free_s']))
    keys=sorted(queues,key=lambda k:(abs(queues[k][0][0]['free_s']-end_s),k))
    families={family:[] for family in ('crossing','approach','extrapolated_crossing')}
    while any(queues[k] for k in keys):
        for key in keys:
            if queues[key]:families[key[1]].append(queues[key].pop(0))
    kept=[]
    while len(kept)<budget and any(families.values()):
        for family in families:
            if families[family] and len(kept)<budget:kept.append(families[family].pop(0))
    omitted=[g for family in families for g in families[family]]
    return kept,omitted


def _candidate_pool(rows,end_s):
    """Reserve real continuity/transition rows before score-only unknown rows."""
    budget=PARAMETERS['maximum_candidates'];pool=[];reasons={}
    def add(i,reason):
        if 0<=i<len(rows) and i not in pool and len(pool)<budget:pool.append(i);reasons[i]=reason
    rank=lambda i:(-bool(rows[i].get('previous_interrupted_row_ids')),-rows[i]['transition_score'],abs(rows[i]['free_axis_s_px']-end_s))
    continuous=sorted([i for i,r in enumerate(rows) if r['original_evidence_state']=='raw_continuous'],key=rank)
    for i in continuous:add(i,'reserved_original_continuity')
    index={r['row_id']:i for i,r in enumerate(rows)}
    for i in continuous:
        for neighbor in rows[i]['neighbor_row_ids']:add(index[neighbor],'neighbor_of_reserved_continuity')
    interrupted=sorted([i for i,r in enumerate(rows) if r['original_evidence_state']=='raw_interruption_supported'],key=rank)
    for i in interrupted:
        add(i,'reserved_original_interruption')
        for neighbor in rows[i]['neighbor_row_ids']:add(index[neighbor],'neighbor_of_interruption')
    queues={}
    for i in sorted(range(len(rows)),key=rank):
        if i not in pool:queues.setdefault((rows[i]['visual_row_id'],rows[i]['probe_family']),[]).append(i)
    families={k:[] for k in ('crossing','approach','extrapolated_crossing')}
    keys=sorted(queues,key=lambda key:rank(queues[key][0]))
    while any(queues.values()):
        for key in keys:
            if queues[key]:families[key[1]].append(queues[key].pop(0))
    while len(pool)<budget and any(families.values()):
        for family in families:
            if families[family]:add(families[family].pop(0),'fair_remaining_'+family)
    return pool,reasons,{'raw_continuous_rows':[rows[i]['row_id'] for i in continuous],'raw_continuous_rows_retained':[rows[i]['row_id'] for i in continuous if i in pool],'raw_continuous_rows_budget_omitted':[rows[i]['row_id'] for i in continuous if i not in pool],'maximum_candidates':budget,'omitted_probe_ids':[r['row_id'] for i,r in enumerate(rows) if i not in pool],'retained_geometry_families':dict(Counter(rows[i]['probe_family'] for i in pool))}


def _seed_rows(refs,origin,direction,normal,end,width,pitch,surface):
    """Seed groove rows at center/half-width lines AND near-tangent approaches.

    A free-rope center should continue into rope body, not necessarily a groove.
    Band-edge intersections retain both signs as alternatives. They are not
    measured contact points and do not turn the n-th row into a selected rope.
    """
    seeds=[];end_s=float(np.dot(end-origin,direction));h,w=surface.shape
    lower_s,entry_diag=_entry_interval(surface,origin,direction,width,pitch,end_s)
    offsets=(0.,-.5*width,.5*width)
    for i,ref in enumerate(refs):
        xy=np.asarray(ref.get('sampled_points_xy',[]),float).reshape(-1,2)
        if len(xy)<4 or not np.isfinite(xy).all():continue
        xy=xy[np.argsort(xy[:,0],kind='stable')];xy=xy[np.r_[True,np.diff(xy[:,0])>0]]
        if len(xy)<4:continue
        signed=(xy-origin)@normal;centers=[]
        for band_offset in offsets:
            distance=signed-band_offset;found=[]
            for k in np.flatnonzero(distance[:-1]*distance[1:]<=0):
                den=distance[k+1]-distance[k]
                if abs(den)<1e-8:continue
                q=xy[k]-distance[k]/den*(xy[k+1]-xy[k])
                kind='within_reference_span' if band_offset==0 else 'free_band_edge_intersection'
                found.append((q,kind,0.,float(band_offset)))
            if not found:
                # No zero crossing in the actual sampled curve: an interior
                # polynomial root would be fictitious. Only extend from a real
                # endpoint, on its outward side, within the declared 6p limit.
                for edge_index,outward in [(0,-1),(-1,1)]:
                    anchor=xy[edge_index]
                    if abs(np.dot(anchor-origin,normal)-band_offset)>max(6*pitch,4*width):continue
                    model=_curve_model(xy,float(anchor[0]),pitch)
                    if model is None:continue
                    aa,bb,cc=model['left_coef'] if outward<0 else model['right_coef']
                    coeff=np.array([normal[1]*aa,normal[0]*pitch+normal[1]*bb,normal[0]*anchor[0]+normal[1]*cc-np.dot(normal,origin)-band_offset]);coeff[np.abs(coeff)<1e-8]=0.;coeff=np.trim_zeros(coeff,'f')
                    roots=np.roots(coeff) if len(coeff)>1 else []
                    for u in roots:
                        if abs(np.imag(u))>1e-6:continue
                        x=float(anchor[0]+pitch*np.real(u));extra=(x-anchor[0])*outward
                        if not .01<extra<PARAMETERS['maximum_extrapolation_pitch']*pitch-1.:continue
                        q=np.array([x,float(_eval(model,np.array([x]))[0])])
                        if abs(np.dot(q-origin,normal)-band_offset)>1e-4:continue
                        found.append((q,'finite_reference_extrapolation',float(extra),float(band_offset)))
            centers.extend(found)
        # Exact intersection is not necessary. A groove bordering the same rope
        # can approach a parallel free centerline without crossing it at all.
        forward_s=(xy-origin)@direction
        eligible=np.flatnonzero(forward_s>=lower_s)
        if len(eligible):
            cost=abs(signed[eligible])+.001*abs(forward_s[eligible]-end_s)
            k=int(eligible[np.argmin(cost)])
        else:k=int(np.argmin(abs(signed)))
        closest=xy[k]
        tangent=xy[min(len(xy)-1,k+3)]-xy[max(0,k-3)];tangent/=max(1e-9,np.linalg.norm(tangent))
        if abs(signed[k])<=max(.60*width,.75*pitch) and abs(np.dot(tangent,normal))<.25:
            nearest_band=float(min(offsets,key=lambda v:abs(signed[k]-v)))
            centers.append((closest.copy(),'tangential_strip_approach',0.,nearest_band))
        for center,kind,extra,band_offset in centers:
            ss=float(np.dot(center-origin,direction))
            if ss<lower_s or not (-pitch<=center[0]<w+pitch and -pitch<=center[1]<h+pitch):continue
            model=_curve_model(xy,float(center[0]),pitch)
            if model is None:continue
            actual_y=float(_eval(model,np.array([center[0]]))[0])
            if not np.isfinite(actual_y) or abs(actual_y-center[1])>max(1.,.05*pitch):continue
            near=xy[np.argsort(np.linalg.norm(xy-center,axis=1))[:min(8,len(xy))]]
            if not np.any(_at(surface,near)>0):continue
            seeds.append({'reference_id':str(ref.get('curve_id',ref.get('id',i))),'source':str(ref.get('source','unspecified')),'center':center,'free_center_projection':origin+ss*direction,'free_band_target_offset_px':band_offset,'signed_distance_to_free_center_px':float(np.dot(center-origin,normal)),'free_s':ss,'model':model,'geometry_kind':kind,'extrapolated_x_px':extra,'reference_span':float(np.ptp(xy[:,0]))})
    # Merge only identical reference IDs or consistent ACTUAL overlapping
    # samples. Never extrapolate distant short arcs to a global sorting x.
    preference={'within_reference_span':0,'free_band_edge_intersection':1,'tangential_strip_approach':2,'finite_reference_extrapolation':3}
    ordered=sorted(seeds,key=lambda r:(preference[r['geometry_kind']],r['extrapolated_x_px'],-r['reference_span']));visual_groups=[]
    for seed in ordered:
        seed['order_y']=float(seed['center'][1]);match=None
        for group in visual_groups:
            if _same_observed_row(seed,group[0],pitch):match=group;break
        if match is None:visual_groups.append([seed])
        else:match.append(seed)
    # A visual row and its intersection probes are different objects. A curved
    # row can cross this strip twice; a near-tangent probe also tests a different
    # hypothesis from a true crossing. Neither may replace the other silently.
    groups=[]
    for vi,visual in enumerate(visual_groups):
        probes=[]
        for seed in visual:
            seed['visual_row_id']=f'visual_row_{vi:03d}';match=None
            for group in probes:
                ss=[q['free_s'] for q in group]+[seed['free_s']]
                if _probe_family(seed['geometry_kind'])==_probe_family(group[0]['geometry_kind']) and max(ss)-min(ss)<=2*width:match=group;break
            if match is None:probes.append([seed])
            else:match.append(seed)
        groups.extend(probes)
    before=len(groups);groups,omitted=_fair_probe_budget(groups,PARAMETERS['maximum_rows'],end_s)
    groups.sort(key=lambda g:(g[0]['free_s'],g[0]['visual_row_id'],preference[g[0]['geometry_kind']]))
    return groups,{'reference_count':len(refs),'geometric_intersections':len(seeds),'tangential_strip_approaches':sum(g['geometry_kind']=='tangential_strip_approach' for g in seeds),'free_band_offsets_probed_px':list(offsets),'deduplicated_rows':len(groups),'visual_row_count':len(visual_groups),'probe_count_before_budget':before,'truncated_rows':len(omitted),'omitted_probes':[{'visual_row_id':g[0]['visual_row_id'],'kind':g[0]['geometry_kind'],'free_s_px':g[0]['free_s']} for g in omitted],'retained_before_old_lower':sum(g[0]['free_s']<end_s-2*pitch for g in groups),'entry_sampling_domain':entry_diag,'order_probe_x':None,'row_order_meaning':'probe order along automatic free-axis toward surface; visual rows separately identified','row_merge_rule':'actual overlapping visual rows; separate probes for geometry family or along-axis span >2w'}


def _fields(original,groups,pitch,width):
    h,w=original.shape[:2];points=[]
    for g in groups:
        c=g[0]['center'];span=12*pitch;xs=np.linspace(c[0]-span,c[0]+span,25);ys=_eval(g[0]['model'],xs);points.extend(np.c_[xs,ys].tolist())
    q=np.asarray(points);q=q[np.isfinite(q).all(axis=1)];pad=int(math.ceil(2*pitch));x0=max(0,int(q[:,0].min())-pad);x1=min(w,int(q[:,0].max())+pad+1);y0=max(0,int(q[:,1].min())-pad);y1=min(h,int(q[:,1].max())+pad+1)
    gray=cv2.cvtColor(original[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray=cv2.GaussianBlur(gray,(0,0),max(.6,.035*pitch));sigma=PARAMETERS['normalization_sigma_pitch']*pitch
    mean=cv2.GaussianBlur(gray,(0,0),sigma);variance=np.maximum(0,cv2.GaussianBlur(gray*gray,(0,0),sigma)-mean*mean);std=np.sqrt(variance+1.)
    normalized=np.clip((gray-mean)/np.maximum(std,3.),-4,4).astype(np.float32)
    coarse=cv2.GaussianBlur(gray,(0,0),max(.8,.10*pitch));gx=cv2.Sobel(coarse,cv2.CV_32F,1,0,ksize=3)/8;gy=cv2.Sobel(coarse,cv2.CV_32F,0,1,ksize=3)/8
    return {'x0':x0,'y0':y0,'original_gray':gray,'normalized_gray':normalized,'local_std':std,'gradient_x':gx,'gradient_y':gy}


def _profile(fields,point,tangent,free_direction,pitch,surface,width,search_surface=None):
    normal=np.array([-tangent[1],tangent[0]]);offsets=np.arange(-.60*pitch,.60*pitch+.5,1.)
    tangent_half=min(.25*pitch,.20*width);along=np.array([-tangent_half,0,tangent_half]);locations=point+along[:,None,None]*tangent+offsets[None,:,None]*normal
    raw=_sample(fields['original_gray'],locations,fields['x0'],fields['y0']);norm=_sample(fields['normalized_gray'],locations,fields['x0'],fields['y0']);profile=raw.mean(axis=0);nprofile=norm.mean(axis=0)
    middle=np.flatnonzero(abs(offsets)<=PARAMETERS['valley_search_pitch']*pitch);idx=int(middle[np.argmin(nprofile[middle])]);val=float(profile[idx]);nval=float(nprofile[idx])
    left=offsets<-.30*pitch;right=offsets>.30*pitch
    shoulders=np.array([np.median(profile[left]),np.median(profile[right])]);nshoulders=np.array([np.median(nprofile[left]),np.median(nprofile[right])]);contrast=shoulders-val;nc=float(min(nshoulders)-nval)
    half=val+.5*max(0,float(min(contrast)));lo=hi=idx
    while lo>0 and profile[lo-1]<half:lo-=1
    while hi<len(profile)-1 and profile[hi+1]<half:hi+=1
    width=float(offsets[hi]-offsets[lo]+1);raw_range=float(np.percentile(raw,95)-np.percentile(raw,5));saturated=float(np.mean((raw>=253)|(raw<=1)))
    gx=_sample(fields['gradient_x'],locations,fields['x0'],fields['y0']);gy=_sample(fields['gradient_y'],locations,fields['x0'],fields['y0'])
    gn=float(np.mean(abs(gx*normal[0]+gy*normal[1])));gt=float(np.mean(abs(gx*tangent[0]+gy*tangent[1])));align=gn/max(1e-6,gn+gt)
    fn=np.array([-free_direction[1],free_direction[0]]);fg=float(np.mean(abs(gx*fn[0]+gy*fn[1])));ft=float(np.mean(abs(gx*free_direction[0]+gy*free_direction[1])));falign=fg/max(1e-6,fg+ft)
    domain=surface if search_surface is None else search_surface
    footprint=_at(domain,locations.reshape(-1,2)).reshape(locations.shape[:-1])>0
    domain_fraction=float(footprint.mean());domain_center=bool(_at(domain,point[None,:])[0])
    raw_available=raw_range>=PARAMETERS['minimum_raw_range'] and saturated<=PARAMETERS['maximum_saturation_fraction']
    domain_available=domain_center and domain_fraction>=PARAMETERS['minimum_available_fraction']
    available=raw_available and domain_available
    broad=PARAMETERS['minimum_groove_width_pitch']*pitch<=width<=PARAMETERS['maximum_groove_width_pitch']*pitch
    support=available and broad and min(contrast)>=PARAMETERS['minimum_raw_weak_contrast'] and max(contrast)>=PARAMETERS['minimum_raw_strong_contrast'] and nc>=PARAMETERS['minimum_normalized_contrast'] and align>=PARAMETERS['minimum_orientation_ratio']
    score=float(np.clip(nc/.7,0,1)*np.clip(min(contrast)/5,0,1)*np.clip(align,0,1)) if available and broad else 0.
    measured=point+offsets[idx]*normal
    return {'sample_xy':point.tolist(),'measured_valley_xy':measured.tolist(),'available':bool(available),'raw_available':bool(raw_available),'domain_available':bool(domain_available),'search_domain_fraction':domain_fraction,'search_domain_at_sample':domain_center,'original_groove_support':bool(support),'support_score':score,'groove_width_px':width,'strand_like_narrow_shadow':bool(width<PARAMETERS['minimum_groove_width_pitch']*pitch and max(contrast)>2),'raw_weak_contrast':float(min(contrast)),'raw_strong_contrast':float(max(contrast)),'normalized_contrast':nc,'raw_range':raw_range,'saturation_fraction':saturated,'wound_direction_alignment':align,'free_direction_alignment':falign,'surface_at_sample':bool(_at(surface,point[None,:])[0])}


def _region(samples,ids):
    s=[samples[i] for i in ids];available=[q for q in s if q['available']]
    return {'samples':len(s),'available_fraction':len(available)/max(1,len(s)),'domain_available_fraction':sum(q.get('domain_available',True) for q in s)/max(1,len(s)),'raw_available_fraction':sum(q.get('raw_available',q['available']) for q in s)/max(1,len(s)),'support_fraction_available':sum(q['original_groove_support'] for q in available)/max(1,len(available)),'support_score_mean':float(np.mean([q['support_score'] for q in available])) if available else 0.,'free_direction_alignment_mean':float(np.mean([q['free_direction_alignment'] for q in available])) if available else None,'wound_direction_alignment_mean':float(np.mean([q['wound_direction_alignment'] for q in available])) if available else None,'sufficient':len(s)>=PARAMETERS['minimum_region_samples'] and len(available)/max(1,len(s))>=PARAMETERS['minimum_available_fraction']}


def _unknown_reasons(regions,near_parallel):
    reasons=[]
    if near_parallel:reasons.append('near_tangent_not_a_two_sided_crossing')
    for name in ('left_far','right_far','center'):
        r=regions[name]
        if r['samples']<PARAMETERS['minimum_region_samples']:reasons.append(name+':insufficient_samples')
        elif not r['sufficient']:
            if r['domain_available_fraction']<PARAMETERS['minimum_available_fraction']:reasons.append(name+':outside_search_domain')
            if r['raw_available_fraction']<PARAMETERS['minimum_available_fraction']:reasons.append(name+':raw_contrast_or_saturation_unavailable')
        if name!='center' and r['sufficient'] and r['support_fraction_available']<PARAMETERS['supported_fraction']:reasons.append(name+':insufficient_wide_groove_support')
    cc=regions['center']
    if cc['sufficient'] and PARAMETERS['interrupted_fraction']<cc['support_fraction_available']<PARAMETERS['supported_fraction']:reasons.append('center:intermediate_support_not_broken_or_continuous')
    return reasons


def _assign_crossing_evidence(rows):
    """Order actual crossings, never count two probes as two winding rows.

    Nearby support and n-1-like transitions are only image evidence. A run is
    broken by an unknown/continuous competing row, rather than skipping it to
    assemble a favorable chain. Near-tangent approaches do not enter this list.
    """
    ordered=sorted([r for r in rows if not r['near_parallel_overlap'] and r['geometry_kind']!='tangential_strip_approach'],key=lambda r:r['free_axis_s_px'])
    for r in rows:r.update({'crossing_order_index':None,'previous_interrupted_row_ids':[],'previous_interrupted_visual_rows':[],'interruption_then_continuity_chain':False})
    for i,row in enumerate(ordered):
        row['crossing_order_index']=i;seen={row['visual_row_id']};previous=[]
        for other in reversed(ordered[:i]):
            if other['visual_row_id'] in seen:continue
            seen.add(other['visual_row_id'])
            if other['original_evidence_state']!='raw_interruption_supported':break
            previous.append(other)
            if len(previous)>=2:break
        row['previous_interrupted_row_ids']=[p['row_id'] for p in reversed(previous)]
        row['previous_interrupted_visual_rows']=[p['visual_row_id'] for p in reversed(previous)]
        row['interruption_then_continuity_chain']=row['original_evidence_state']=='raw_continuous' and len(previous)>=2
    for row in rows:
        candidates=[q for q in ordered if q['visual_row_id']!=row['visual_row_id']]
        before=[q for q in candidates if q['free_axis_s_px']<row['free_axis_s_px']];after=[q for q in candidates if q['free_axis_s_px']>row['free_axis_s_px']]
        row['neighbor_row_ids']=[q['row_id'] for q in ((before[-1:] if before else [])+(after[:1] if after else []))]
    return {'crossing_probe_count':len(ordered),'independent_visual_rows_in_crossing_order':len(set(r['visual_row_id'] for r in ordered)),'chain_count':sum(r['interruption_then_continuity_chain'] for r in rows),'chain_requirement':'two consecutive independent visual rows interrupted before continuous crossing; unknown rows break chain; near tangencies excluded'}


def analyze_boundary_occlusion(original_bgr,surface,merged,inferred,curve_refs,free_geometry,exclusion_mask=None,pitch=None,search_surface=None):
    """Return competing image-row hypotheses and original-image evidence.

    refs: list of {curve_id,source,sampled_points_xy}; free geometry:
    origin_xy, direction_xy (toward surface), width_px, visible_end_xy.
    F3 masks are diagnostics only and never enter raw continuity ranking.
    """
    pitch=float(pitch if pitch is not None else free_geometry.get('pitch_px',20.))
    if not math.isfinite(pitch) or not 8<=pitch<=70:raise ValueError('pitch must be 8..70 image pixels')
    if original_bgr.ndim!=3 or surface.shape!=original_bgr.shape[:2] or merged.shape!=surface.shape or inferred.shape!=surface.shape:raise ValueError('original-size inputs required')
    refs=curve_refs.get('curves',[]) if isinstance(curve_refs,dict) else list(curve_refs)
    if free_geometry.get('status','observed').lower() in ('unknown','rejected','unavailable'):
        return {'status':'UNKNOWN','rows':[],'candidates':[],'fields':{},'diagnostics':{'reason':'free_geometry_unavailable','physical_contact':None}}
    if free_geometry.get('direction_xy',free_geometry.get('toward_surface_direction_xy',free_geometry.get('direction'))) is None:
        return {'status':'UNKNOWN','rows':[],'candidates':[],'fields':{},'diagnostics':{'reason':'ambiguous_free_direction_call_each_direction_separately','direction_candidates':free_geometry.get('directions',free_geometry.get('direction_candidates_xy',[])),'physical_contact':None}}
    domain=surface if search_surface is None else np.asarray(search_surface)
    if domain.shape!=surface.shape:raise ValueError('search_surface must have original image shape')
    origin,direction,normal,width,end=_geometry(free_geometry,pitch);groups,diag=_seed_rows(refs,origin,direction,normal,end,width,pitch,domain)
    if not groups:return {'status':'UNKNOWN','rows':[],'candidates':[],'fields':{},'diagnostics':{**diag,'reason':'no_referenced_row_reaches_free_strip','physical_contact':None}}
    fields=_fields(original_bgr,groups,pitch,width);old_near=cv2.dilate((merged>0).astype(np.uint8),np.ones((5,5),np.uint8));excluded=np.zeros_like(surface) if exclusion_mask is None else exclusion_mask
    rows=[]
    for i,group in enumerate(groups):
        seed=group[0];model=seed['model'];center=seed['center'];near_y=_eval(model,np.array([center[0]-1,center[0]+1]));slope=float((near_y[1]-near_y[0])/2);tangent=np.array([1.,slope]);tangent/=np.linalg.norm(tangent)
        crossing=abs(float(np.dot(tangent,normal)));span=min(12*pitch,max(4*pitch,2.2*width/max(.15,crossing)));xs=np.arange(center[0]-span,center[0]+span+1,max(2.,pitch/6));ys=_eval(model,xs);xy=np.c_[xs,ys];h,w=surface.shape;valid=(xs>=pitch)&(xs<w-pitch)&(ys>=pitch)&(ys<h-pitch)&np.isfinite(ys);xy=xy[valid]
        if len(xy)<9:continue
        slopes=np.gradient(xy[:,1],xy[:,0]);samples=[]
        for p,m in zip(xy,slopes):
            tb=np.array([1.,m]);tb/=np.linalg.norm(tb);s=_profile(fields,p,tb,direction,pitch,surface,width,domain);s.update({'signed_free_distance_px':float(np.dot(p-origin,normal)),'formal_F3_nearby':bool(_at(old_near,p[None,:])[0]),'formal_inferred_here':bool(_at(inferred,p[None,:])[0]),'algorithm_excluded_here':bool(_at(excluded,p[None,:])[0])});samples.append(s)
        tvals=np.array([s['signed_free_distance_px'] for s in samples]);dx=xy[:,0]-center[0];inside=abs(tvals)<=.55*width;outside=abs(tvals)>=.85*width;near=abs(tvals)<1.5*width
        regions={'center':_region(samples,np.flatnonzero(inside)),'left_near':_region(samples,np.flatnonzero((dx<0)&outside&near)),'right_near':_region(samples,np.flatnonzero((dx>0)&outside&near)),'left_far':_region(samples,np.flatnonzero((dx<0)&outside)),'right_far':_region(samples,np.flatnonzero((dx>0)&outside))}
        lf,rf,cc=regions['left_far'],regions['right_far'],regions['center'];both=lf['sufficient'] and rf['sufficient'];side=min(lf['support_fraction_available'],rf['support_fraction_available']);continuous=both and cc['sufficient'] and side>=.55 and cc['support_fraction_available']>=.55;broken=both and cc['sufficient'] and side>=.55 and cc['support_fraction_available']<=.25
        near_parallel=crossing<.15 or seed['geometry_kind']=='tangential_strip_approach'
        state='raw_continuous' if continuous and not near_parallel else ('raw_interruption_supported' if broken and not near_parallel else 'unknown_original_evidence')
        center_samples=[samples[k] for k in np.flatnonzero(inside)];mask_gap=float(np.mean([not s['formal_F3_nearby'] for s in center_samples])) if center_samples else None;exclusion_fraction=float(np.mean([s['algorithm_excluded_here'] for s in center_samples])) if center_samples else None
        mask_relation='not_enough_center_samples'
        if mask_gap is not None:
            if mask_gap>=.5 and continuous:mask_relation='algorithm_gap_but_original_continuous'
            elif mask_gap>=.5 and broken:mask_relation='original_interruption_measured_independently_of_mask'
            elif mask_gap>=.5:mask_relation='mask_gap_cannot_be_used_as_occlusion_evidence'
            else:mask_relation='formal_mask_presence_is_not_original_truth'
        row={'row_id':f'row_{len(rows):03d}','order_y_at_probe':seed['order_y'],'intersection_xy':center.tolist(),'free_axis_s_px':seed['free_s'],'geometry_kind':seed['geometry_kind'],'extrapolated_x_px':seed['extrapolated_x_px'],'reference_ids':[g['reference_id'] for g in group],'reference_sources':[g['source'] for g in group],'free_center_projection_xy':seed['free_center_projection'].tolist(),'free_band_target_offset_px':seed['free_band_target_offset_px'],'free_band_offsets_present_px':sorted(set(g['free_band_target_offset_px'] for g in group)),'geometry_variants':[{'reference_id':g['reference_id'],'intersection_xy':g['center'].tolist(),'free_center_projection_xy':g['free_center_projection'].tolist(),'free_band_target_offset_px':g['free_band_target_offset_px'],'signed_distance_to_free_center_px':g['signed_distance_to_free_center_px'],'geometry_kind':g['geometry_kind']} for g in group[:8]],'sampled_points_xy':xy.tolist(),'regions':regions,'samples':samples,'original_evidence_state':state,'crossing_sine':crossing,'near_parallel_overlap':bool(crossing<.15),'formal_center_gap_fraction':mask_gap,'algorithm_exclusion_center_fraction':exclusion_fraction,'mask_relation':mask_relation,'surface_center':bool(_at(surface,center[None,:])[0]),'physical_occlusion':None,'physical_contact':None}
        row.update({'visual_row_id':seed['visual_row_id'],'probe_family':_probe_family(seed['geometry_kind']),'near_parallel_overlap':near_parallel,'near_tangent_state':('local_groove_supported' if cc['sufficient'] and cc['support_fraction_available']>=PARAMETERS['supported_fraction'] else 'unknown') if near_parallel else None,'unknown_reasons':_unknown_reasons(regions,near_parallel) if state=='unknown_original_evidence' else [],'measurement_x_range_px':[float(xy[0,0]),float(xy[-1,0])],'measurement_free_axis_range_px':[float(np.min((xy-origin)@direction)),float(np.max((xy-origin)@direction))],'reference_span_x_px':[float(model['xy'][0,0]),float(model['xy'][-1,0])],'search_domain_center':bool(_at(domain,center[None,:])[0])})
        rows.append(row)
    # Rank a bounded pool, retaining neighboring competitors. No n-1 winner.
    crossing_diag=_assign_crossing_evidence(rows)
    for i,row in enumerate(rows):
        broken_prev=row['previous_interrupted_row_ids'];regions=row['regions'];side=min(regions['left_far']['support_fraction_available'],regions['right_far']['support_fraction_available']);continuity=regions['center']['support_fraction_available'] if regions['center']['sufficient'] else 0.
        raw_score=.45*continuity+.30*bool(broken_prev)+.20*side
        geometry_score=.05*float(row['geometry_kind']=='within_reference_span')
        # A high local response with no complete cross-strip evidence cannot
        # earn a material-connection bonus merely by being geometrically close.
        row['appearance_proxy_score']=float(raw_score+geometry_score)
        row['geometry_prior_score']=float(geometry_score)
        row['raw_evidence_score']=float(raw_score) if row['original_evidence_state']!='unknown_original_evidence' else 0.
        row['transition_score']=row['raw_evidence_score']
    pool,pool_reasons,pool_diagnostics=_candidate_pool(rows,float(np.dot(end-origin,direction)))
    candidates=[]
    for rank,i in enumerate(pool):
        r=rows[i];candidates.append({'candidate_id':f'boundary_row_{rank:02d}','pool_inclusion_reason':pool_reasons[i],'row_id':r['row_id'],'intersection_xy':r['intersection_xy'],'sampled_points_xy':r['sampled_points_xy'],'reference_ids':r['reference_ids'],'free_center_projection_xy':r['free_center_projection_xy'],'free_band_target_offset_px':r['free_band_target_offset_px'],'free_band_offsets_present_px':r['free_band_offsets_present_px'],'geometry_variants':r['geometry_variants'],'transition_score':r['transition_score'],'original_evidence_state':r['original_evidence_state'],'previous_interrupted_row_ids':r['previous_interrupted_row_ids'],'competing_neighbor_row_ids':r['neighbor_row_ids'],'geometry_kind':r['geometry_kind'],'selection_status':'candidate_only','physical_contact':None,'same_material_identity':None})
    for candidate in candidates:
        row=next(r for r in rows if r['row_id']==candidate['row_id'])
        candidate.update({k:row[k] for k in ('visual_row_id','probe_family','crossing_order_index','near_parallel_overlap','near_tangent_state','previous_interrupted_visual_rows','interruption_then_continuity_chain','unknown_reasons','measurement_x_range_px','appearance_proxy_score','geometry_prior_score','raw_evidence_score')})
    diag.update({'candidate_pool':pool_diagnostics,'crossing_evidence':crossing_diag,'parameters':PARAMETERS.copy(),'rows_evaluated':len(rows),'candidate_count':len(candidates),'original_states':dict(Counter(r['original_evidence_state'] for r in rows)),'unknown_reason_counts':dict(Counter(reason for r in rows for reason in r['unknown_reasons'])),'sample_domain':'explicit_search_surface' if search_surface is not None else 'saved_surface','original_surface_unchanged':True,'algorithm_gap_but_raw_continuous':sum(r['mask_relation']=='algorithm_gap_but_original_continuous' for r in rows),'formal_mask_used_in_continuity_score':False,'old_exclusion_used_as_occlusion_truth':False,'selected_candidate_id':None,'physical_contact':None,'limitations':['Original-image contrast can still come from a neighboring rope or strand.','Reference extrapolation is bounded geometry only.','Missing/low-contrast/saturated evidence remains unknown.','Crossing order and visual rows are not physical winding or material identity.','Near-tangent appearance does not demonstrate two-sided interruption.']})
    return {'status':'MULTI_CANDIDATE' if len(candidates)>1 else ('CANDIDATE_ONLY' if candidates else 'UNKNOWN'),'rows':rows,'candidates':candidates,'fields':fields,'diagnostics':diag}


# Explicit short alias for callers; one implementation only.
analyze_occlusion=analyze_boundary_occlusion
