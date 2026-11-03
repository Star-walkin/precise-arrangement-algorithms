"""Recover AUTO free-rope seed proposals from the original image only.

This module never reads old selected routes, F3 identities, labels or filenames.
Long paired outside edges are only proposals. Call the preserved A5
refine_and_extend on each seed and retain competing revalidated entries.
"""
from __future__ import annotations
import math
from collections import Counter
import cv2
import numpy as np

PARAMETERS={
    'detector_scale':.5,'detector_blur_original_pitch_fraction':.15,'minimum_segment_pitch':6.,'minimum_segment_px':110.,
    'maximum_segments_per_channel':200,'maximum_pair_proposals':80,
    'maximum_parallel_angle_deg':2.,'minimum_width_pitch':.24,'maximum_width_pitch':1.8,
    'minimum_overlap_pitch':5.,'minimum_overlap_px':100.,
    'maximum_ray_px':1800.,'maximum_ray_image_fraction':.45,
    'maximum_recovered_seeds':6,'minimum_measured_profiles':12,
    'minimum_good_profile_fraction':.48,'minimum_bilateral_contrast':10.,
    'minimum_support_span_px':100.,'minimum_support_span_pitch':5.,
    'minimum_oblique_texture_correlation':.25,
    'minimum_longitudinal_modulation_px':3.,
    'saturated_outline_fraction':.60,
}

def _sample(a,xy):
    q=np.asarray(xy,np.float32)
    return cv2.remap(a,q[...,0],q[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)

def _surface_domain(surface):
    """Drop isolated SAM islands for directional geometry; never add identity."""
    binary=np.where(surface>0,255,0).astype(np.uint8);cs,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    keep=[c for c in cs if cv2.contourArea(c)>=max(50.,.0001*np.count_nonzero(binary))]
    domain=np.zeros_like(binary)
    if keep:cv2.drawContours(domain,keep,-1,255,cv2.FILLED)
    return domain

def _ray_hits(segment,d,domain):
    length=min(PARAMETERS['maximum_ray_px'],PARAMETERS['maximum_ray_image_fraction']*max(domain.shape));dist=np.arange(0,length+4,4.)
    hits=[];h,w=domain.shape
    for end,ray in ((0,-d),(1,d)):
        q=segment[end]+dist[:,None]*ray;ix=np.rint(q[:,0]).astype(int);iy=np.rint(q[:,1]).astype(int);valid=(ix>=0)&(ix<w)&(iy>=0)&(iy<h)
        pos=np.flatnonzero(valid);inside=pos[domain[iy[pos],ix[pos]]>0]
        hits.append({'end_index':end,'intersects_surface':bool(len(inside)),'first_hit_distance_px':float(dist[inside[0]]) if len(inside) else None,'first_hit_xy':q[inside[0]].tolist() if len(inside) else None})
    return hits

def _segments(channels,domain,pitch):
    scale=PARAMETERS['detector_scale'];detector=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD);items=[];counts={}
    minimum=max(PARAMETERS['minimum_segment_px'],PARAMETERS['minimum_segment_pitch']*pitch)
    for name,channel in channels.items():
        small=cv2.resize(np.clip(channel,0,255).astype(np.uint8),None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        # Strand highlights make the true envelope a chain of short jagged
        # edges. A fraction-of-pitch blur restores its long outer envelope;
        # all actual width/contrast measurements still use original pixels.
        small=cv2.GaussianBlur(small,(0,0),float(np.clip(PARAMETERS['detector_blur_original_pitch_fraction']*pitch*scale,.8,2.2)))
        detected=detector.detect(small)[0];raw=[]
        for line in ([] if detected is None else detected.reshape(-1,4)):
            q=line.reshape(2,2).astype(float)/scale;v=q[1]-q[0];length=float(np.linalg.norm(v))
            if length<minimum:continue
            d=v/length
            if d[0]<0 or (abs(d[0])<1e-6 and d[1]<0):q=q[::-1];d=-d
            samples=q[0]+np.linspace(0,1,max(25,int(length/4)+1))[:,None]*(q[1]-q[0])
            outside=_sample(domain,samples[None,:,:]).ravel()<127
            # LSD may follow the same free rope across the cylinder. Rejecting
            # that complete line for a low outside ratio would lose its useful
            # outside half. Split first; only actual outside subsegments enter.
            change=np.diff(np.r_[False,outside,False].astype(np.int8))
            for lo,hi in zip(np.flatnonzero(change==1),np.flatnonzero(change==-1)):
                piece=np.array([samples[lo],samples[hi-1]]);piece_length=float(np.linalg.norm(piece[1]-piece[0]))
                if piece_length<minimum:continue
                raw.append({'xy':piece,'d':d,'length':piece_length,'channel':name,'outside_fraction':1.})
        raw.sort(key=lambda v:-v['length']);counts[name]={'long_outside_segments':len(raw),'kept_segments':min(len(raw),PARAMETERS['maximum_segments_per_channel'])}
        items+=raw[:PARAMETERS['maximum_segments_per_channel']]
    return items,counts

def _pair_proposals(lines,domain,pitch):
    proposed=[];counts=Counter();minspan=max(PARAMETERS['minimum_overlap_px'],PARAMETERS['minimum_overlap_pitch']*pitch)
    for i,a in enumerate(lines):
        d=a['d'];normal=np.array([-d[1],d[0]]);ac=a['xy'].mean(axis=0)
        for j in range(i+1,len(lines)):
            b=lines[j];dot=float(np.clip(d@b['d'],-1,1))
            if dot<math.cos(math.radians(PARAMETERS['maximum_parallel_angle_deg'])):continue
            av=a['xy']@d;bv=b['xy']@d;lo=max(min(av),min(bv));hi=min(max(av),max(bv))
            if hi-lo<minspan:continue
            bn=b['xy']@normal;an=a['xy']@normal;width=abs(float(np.mean(bn)-np.mean(an)))
            if not max(3.,PARAMETERS['minimum_width_pitch']*pitch)<=width<=PARAMETERS['maximum_width_pitch']*pitch:continue
            # End-to-end separation must stay parallel as well as angle-close.
            if abs(bn[1]-bn[0])>max(3.,.35*width):continue
            offset=(float(np.mean(bn))+float(np.mean(an)))/2;seg=np.array([lo*d+offset*normal,hi*d+offset*normal]);hits=_ray_hits(seg,d,domain)
            if not any(v['intersects_surface'] for v in hits):counts['paired_lines_do_not_approach_surface']+=1;continue
            proposed.append({'segment':seg,'direction':d,'width':width,'overlap':hi-lo,'source_segments':[i,j],'channels':[a['channel'],b['channel']],'ray_hits':hits})
    proposed.sort(key=lambda v:-v['overlap']);unique=[]
    for v in proposed:
        middle=v['segment'].mean(axis=0);d=v['direction'];n=np.array([-d[1],d[0]])
        duplicate=False
        for old in unique:
            if abs(d@old['direction'])<math.cos(math.radians(1.)):continue
            if abs((middle-old['segment'].mean(axis=0))@n)>max(2.,min(v['width'],old['width'])*.3):continue
            av=v['segment']@d;bv=old['segment']@d
            if min(max(av),max(bv))-max(min(av),min(bv))>.5*min(v['overlap'],old['overlap']):duplicate=True;break
        if duplicate:counts['same_axis_pair_duplicates']+=1;continue
        unique.append(v)
    counts['raw_paired_proposals']=len(proposed);counts['unique_paired_proposals']=len(unique);counts['proposal_budget_omitted']=max(0,len(unique)-PARAMETERS['maximum_pair_proposals'])
    return unique[:PARAMETERS['maximum_pair_proposals']],dict(counts)

def _remeasure(data,p,channels,domain):
    image=data['original'];surface=data['surface'];pitch=float(data['pitch']);h,w=surface.shape;d=p['direction'];n=np.array([-d[1],d[0]]);center=p['segment'].mean(axis=0);span=p['overlap'];hint=p['width']
    t=np.linspace(-span/2,span/2,min(90,max(18,int(span/8))));ns=np.arange(-max(2*pitch,2*hint),max(2*pitch,2*hint)+.25,.5);along=np.linspace(-.5*pitch,.5*pitch,9)
    rows=[];reject=Counter();yellow=[]
    for ti in t:
        point=center+ti*d;xy=point+along[:,None,None]*d+ns[None,:,None]*n
        if np.any(xy[...,0]<1) or np.any(xy[...,0]>=w-2) or np.any(xy[...,1]<1) or np.any(xy[...,1]>=h-2):reject['profile_border']+=1;continue
        if _sample(surface,point[None,None,:])[0,0]>0:reject['inside_surface']+=1;continue
        bgr=_sample(image,xy).astype(float);flank=(abs(ns)>=max(hint*.9,.8*pitch))&(abs(ns)<=max(hint*1.5,1.4*pitch));colors=bgr[:,flank,:];b,g,r=colors[...,0],colors[...,1],colors[...,2]
        yf=float(np.mean((r>b*1.35+8)&(g>b*1.25+8)&(np.maximum(r,g)>40)));yellow.append(yf);channel=channels['achromatic'] if yf>=.55 else channels['minimum_brightness']
        patch=_sample(channel,xy);prof=np.quantile(patch,.65,axis=0).astype(np.float32);prof=cv2.GaussianBlur(prof[None,:],(5,1),.8)[0]
        near=np.flatnonzero(abs(ns)<=max(2.,hint*.40));k=int(near[np.argmax(prof[near])]);peak=float(prof[k]);lb=float(np.percentile(prof[ns<-.75*max(2*pitch,2*hint)],35));rb=float(np.percentile(prof[ns>.75*max(2*pitch,2*hint)],35));contrast=min(peak-lb,peak-rb)
        if contrast<PARAMETERS['minimum_bilateral_contrast']:reject['weak_two_flank_contrast']+=1;continue
        threshold=np.where(ns<=ns[k],lb+.45*(peak-lb),rb+.45*(peak-rb));fg=(prof>=threshold).astype(np.uint8)[None,:]
        close=max(3,int(.15*hint/.5)|1);fg=cv2.morphologyEx(fg,cv2.MORPH_CLOSE,np.ones((1,close),np.uint8))[0];lo=hi=k
        while lo>0 and fg[lo-1]:lo-=1
        while hi+1<len(ns) and fg[hi+1]:hi+=1
        if lo<2 or hi>=len(ns)-2:reject['no_outer_flanks']+=1;continue
        measured=float(ns[hi]-ns[lo]);offset=float((ns[lo]+ns[hi])/2)
        if not max(3.,.2*pitch)<=measured<=1.8*pitch or abs(offset)>max(.6*pitch,.65*hint):reject['width_or_axis_disagreement']+=1;continue
        gradient=np.gradient(prof,.5);lwin=np.flatnonzero(abs(ns-ns[lo])<=max(1.5,.12*measured));rwin=np.flatnonzero(abs(ns-ns[hi])<=max(1.5,.12*measured));ls=float(np.max(gradient[lwin]));rs=float(np.max(-gradient[rwin]))
        if min(ls,rs)<.9:reject['missing_opposite_edge_gradients']+=1;continue
        edge=point+np.array([ns[lo],ns[hi]])[:,None]*n
        if np.any(_sample(surface,edge[None,:,:])>0):reject['edges_inside_surface']+=1;continue
        rows.append([ti,offset,measured,contrast,min(ls,rs),yf])
    diag={'tested_profiles':len(t),'paired_profiles':len(rows),'reject_counts':dict(reject),'yellow_flank_fraction':float(np.mean(yellow)) if yellow else 0.}
    if len(rows)<PARAMETERS['minimum_measured_profiles'] or len(rows)/len(t)<PARAMETERS['minimum_good_profile_fraction']:return None,{**diag,'reason':'insufficient_outside_paired_profiles'}
    a=np.asarray(rows);coef=np.polyfit(a[:,0],a[:,1],1);res=a[:,1]-np.polyval(coef,a[:,0]);med=float(np.median(a[:,2]));keep=(abs(res)<=max(2.,.2*med))&(abs(a[:,2]-med)<=max(3.,.3*med))
    if np.count_nonzero(keep)<PARAMETERS['minimum_measured_profiles'] or np.ptp(a[keep,0])<max(PARAMETERS['minimum_support_span_px'],PARAMETERS['minimum_support_span_pitch']*pitch):return None,{**diag,'reason':'short_or_unstable_measured_span'}
    a=a[keep];coef=np.polyfit(a[:,0],a[:,1],1);res=a[:,1]-np.polyval(coef,a[:,0]);nd=d+coef[0]*n;nd/=np.linalg.norm(nd);ends_t=np.array([a[:,0].min(),a[:,0].max()]);segment=center+ends_t[:,None]*d+np.polyval(coef,ends_t)[:,None]*n
    if abs(math.degrees(math.atan(coef[0])))>2.5 or np.percentile(abs(res),90)>max(2.,.18*med):return None,{**diag,'reason':'axis_not_straight_or_seed_angle_changed'}
    # Texture is evidence strength, never a requirement that saturated white
    # steel contain visible strands. Featureless outlines remain explicit.
    length=float(np.linalg.norm(segment[1]-segment[0]));tangent=np.arange(0,length,1.)[:,None,None];normals=np.arange(-.32*med,.32*med+.1,.5)[None,:,None];xy=segment[0]+tangent*nd+normals*np.array([-nd[1],nd[0]])
    patch=_sample(channels['gray'],xy);high=patch-cv2.GaussianBlur(patch,(0,0),1.5);texture=float(np.std(high));saturation=float(np.mean(patch>248));uniform=float(np.std(np.mean(patch,axis=1)))
    gn=cv2.Sobel(patch,cv2.CV_32F,1,0)/4.;gt=cv2.Sobel(patch,cv2.CV_32F,0,1)/8.
    if patch.shape[1]>6:patch=patch[:,2:-2];gn=gn[:,2:-2];gt=gt[:,2:-2]
    jnn=float(np.mean(gn*gn));jtt=float(np.mean(gt*gt));jnt=float(np.mean(gn*gt));correlation=abs(jnt)/max(math.sqrt(jnn*jtt),1e-6)
    modulation=float(np.mean(np.std(patch,axis=0)));saturation=float(np.mean(patch>248));body_mean=float(np.mean(patch))
    diagonal=correlation>=PARAMETERS['minimum_oblique_texture_correlation'] and modulation>=PARAMETERS['minimum_longitudinal_modulation_px'] and jnn>=1. and jtt>=1.
    saturated=saturation>=PARAMETERS['saturated_outline_fraction'] and body_mean>=200
    texture_evidence={'tangent_gradient_energy':jtt,'normal_gradient_energy':jnn,'oblique_gradient_correlation':float(correlation),'longitudinal_modulation':modulation,'body_mean_raw':body_mean,'raw_saturation_fraction':saturation,'texture_status':'oblique_strand_evidence' if diagonal else ('saturated_outline_only' if saturated else 'no_oblique_strand_evidence')}
    # Two long bright edges alone also describe machine-frame highlights.
    # The rope's diagonal strand pattern couples tangent and normal gradients;
    # a plain longitudinal metal stripe has almost no cross-gradient coupling.
    if not diagonal and not saturated:return None,{**diag,**texture_evidence,'reason':'plain_parallel_edges_without_strand_or_saturated_body_evidence'}
    hits=_ray_hits(segment,nd,domain)
    if not any(v['intersects_surface'] for v in hits):return None,{**diag,'reason':'remeasured_axis_no_surface_approach'}
    seed={'status':'observed','coordinate_system':'original_unmirrored_image_xy_pixels','observed_segment_xy':segment.tolist(),'center_xy':segment.mean(axis=0).tolist(),'direction_xy':nd.tolist(),'normal_xy':[-float(nd[1]),float(nd[0])],'width_px':float(np.median(a[:,2])),'width_kind':'AUTO_outside_paired_optical_seed_not_physical_diameter','angle_deg':float(math.degrees(math.atan2(nd[1],nd[0]))%180.),'diagnostics':{'selected_profile_refinement':{'channel':'bilateral_yellow_free_span' if np.mean(a[:,5])>=.55 else 'minimum_brightness'},'selected_candidate':{'approach_evidence':{'bilateral_yellow_background_fraction':float(np.mean(a[:,5]))}},'automatic_source':'A6_LSD_parallel_outer_edges_original_profile_remeasurement'},'contact_or_tangency_claim':False}
    quality=float(.35*min(1.,np.ptp(a[:,0])/(12*pitch))+.25*min(1.,np.median(a[:,3])/45)+.20*len(a)/len(t)+.10*min(1.,texture/5)+.10*min(1.,np.median(a[:,4])/5))
    diag.update(reason='paired_outside_axis_proposal',measured_profiles=len(a),measured_span_px=float(np.ptp(a[:,0])),width_mad_px=float(np.median(abs(a[:,2]-np.median(a[:,2])))),fit_residual_p90_px=float(np.percentile(abs(res),90)),median_bilateral_contrast=float(np.median(a[:,3])),raw_texture_std=texture,raw_longitudinal_variation=uniform,surface_ray_hits=hits,score=quality,**texture_evidence)
    return seed,diag

def recover_free_seeds(data):
    """Return <=6 AUTO proposals; downstream must independently revalidate each.

    data requires only original uint8 BGR, surface same-size mask, pitch (px).
    No proposal is declared the physical active rope. Candidates with comparable
    score on distinct axes are explicitly conflicting, not silently discarded.
    """
    image=np.asarray(data['original']);surface=np.asarray(data['surface']);pitch=float(data['pitch'])
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3 or surface.shape!=image.shape[:2] or not 2<=pitch<=100:raise ValueError('Invalid original/surface/pitch')
    bgr=image.astype(np.float32);low=bgr.min(axis=2);high=bgr.max(axis=2);achromatic=low*255./np.maximum(high,1.);achromatic[high<25]=0
    channels={'minimum_brightness':low,'achromatic':achromatic};domain=_surface_domain(surface);lines,line_diag=_segments(channels,domain,pitch);proposals,pair_diag=_pair_proposals(lines,domain,pitch);channels['gray']=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32);kept=[];rejected=[]
    for i,p in enumerate(proposals):
        seed,evidence=_remeasure(data,p,channels,domain)
        if seed is None:rejected.append({'proposal_index':i,'evidence':evidence});continue
        d=np.array(seed['direction_xy']);n=np.array([-d[1],d[0]]);q=np.array(seed['observed_segment_xy']);duplicate=False
        for c in kept:
            other=c['seed'];od=np.array(other['direction_xy']);oq=np.array(other['observed_segment_xy'])
            if abs(d@od)>.9994 and abs((q.mean(axis=0)-oq.mean(axis=0))@n)<.6*min(seed['width_px'],other['width_px']):
                gap=max(min(q@d),min(oq@d))-min(max(q@d),max(oq@d))
                if gap<=12*pitch:
                    c.setdefault('same_axis_fragments',[]).append({'seed':seed,'evidence':evidence,'proposal_index':i});duplicate=True;break
        if duplicate:rejected.append({'proposal_index':i,'evidence':{'reason':'duplicate_remeasured_axis'}});continue
        kept.append({'candidate_id':None,'seed':seed,'score':evidence['score'],'evidence':evidence,'proposal_index':i,'source_detector_channels':p['channels'],'conflicts':[]})
    kept.sort(key=lambda c:-c['score']);omitted=max(0,len(kept)-PARAMETERS['maximum_recovered_seeds']);kept=kept[:PARAMETERS['maximum_recovered_seeds']]
    for i,c in enumerate(kept):c['candidate_id']='recovered_seed_'+str(i).zfill(2)
    for c in kept:c['conflicts']=[q['candidate_id'] for q in kept if q is not c and abs(q['score']-c['score'])<.15]
    return {'status':'AMBIGUOUS_SEED_PROPOSALS' if len(kept)>1 else ('SEED_PROPOSAL' if kept else 'UNKNOWN'),'candidates':kept,'diagnostics':{'parameters':dict(PARAMETERS),'line_detection':line_diag,'pair_generation':pair_diag,'rejected':rejected,'candidate_budget_omitted':omitted,'same_material_identity_confirmed':False,'old_seed_or_selected_route_used':False,'interpretation':'Original outside paired-edge proposals; each requires unchanged legacy free remeasurement before tracing.'}}
