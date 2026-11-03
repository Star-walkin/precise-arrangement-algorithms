"""V28 repair: whole-envelope section pairs and spatially diverse band tracks.

This module has no target/label IO. Every source section is sampled in the
unchanged image. A ridge is never called a whole rope: two edges, a body
interval, and continuity over several source sections are needed. Dark pairs
remain groove/rope ambiguity. Missing sections are hypotheses, not observations.
"""
from __future__ import annotations

from copy import deepcopy
import math

import cv2
import numpy as np

from .structure import strip_to_original, original_to_strip, _runs


DEFAULT_CONFIG = dict(
    maximum_forward_px=2400., search_backward_widths=2.,
    search_half_widths=28., section_step_widths=.40, section_step_min_px=6.,
    normal_step_px=1., along_profiles=9, along_half_widths=.48,
    derivative_sigma_widths=.065, broad_sigma_widths=.18,
    minimum_width_ratio=.65, maximum_width_ratio=1.70,
    minimum_gradient=1.0, minimum_contrast=5., minimum_edge_balance=.25,
    minimum_tangential_support=.65, maximum_saturated_fraction=.20,
    maximum_edge_saturated_fraction=.45, maximum_baseline_saturated_fraction=.25,
    nms_center_widths=.48, maximum_section_nodes=70,
    maximum_gap_widths=3., minimum_track_nodes=8,
    minimum_track_length_widths=6., minimum_track_body_fraction=.55,
    maximum_structures=32, diversity_u_bins=8, diversity_v_widths=2.,
    adaptive_section_tangents=True, tangent_sigma_width_ratios=[.22,.35],
    tangent_minimum_scale_agreement=.90,
)


def _mask(regions, key, shape, fallback=None):
    if regions is None:
        return np.ones(shape, bool) if fallback is None else fallback
    value = regions.get(key)
    if value is None:
        return np.ones(shape, bool) if fallback is None else fallback
    value = np.asarray(value, bool)
    if value.shape != shape:
        raise ValueError(key+' must be an original-image mask')
    return value


def _sample_scalar(image, xy, interpolation=cv2.INTER_LINEAR):
    return cv2.remap(np.asarray(image, np.float32),
                     xy[..., 0].astype(np.float32), xy[..., 1].astype(np.float32),
                     interpolation, borderMode=cv2.BORDER_CONSTANT)


def _tangent_field(gray,width,d,cfg):
    """Coarse two-scale source orientation proposes sampling directions only.

    Diagonal strand detail is filtered at the envelope scale. Agreement of two
    scales is needed; this orientation is never rope identity or edge support.
    The median profile and its two source boundaries still supply evidence.
    """
    scale=.5
    work=cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    directions=[];qualities=[]
    for ratio in cfg['tangent_sigma_width_ratios']:
        smooth=cv2.GaussianBlur(work,(0,0),max(1.,width*ratio*scale))
        aa=cv2.Sobel(smooth,cv2.CV_32F,2,0,ksize=3)
        bb=cv2.Sobel(smooth,cv2.CV_32F,1,1,ksize=3)
        cc=cv2.Sobel(smooth,cv2.CV_32F,0,2,ksize=3)
        delta=np.sqrt((aa-cc)**2+4*bb**2)
        first=.5*(aa+cc+delta);second=.5*(aa+cc-delta)
        lam=np.where(abs(first)>=abs(second),first,second)
        other=np.where(abs(first)>=abs(second),second,first)
        nx=bb;ny=lam-aa
        norm=np.sqrt(nx*nx+ny*ny)
        # Stable diagonal matrix fallback when the off-diagonal entry is zero.
        nx=np.where(norm>1e-6,nx,np.where(abs(aa)>=abs(cc),1.,0.))
        ny=np.where(norm>1e-6,ny,np.where(abs(aa)>=abs(cc),0.,1.))
        norm=np.maximum(np.sqrt(nx*nx+ny*ny),1e-6)
        tx=-ny/norm;ty=nx/norm
        sign=np.where(tx*d[0]+ty*d[1]>=0,1.,-1.)
        directions.append(np.stack((tx*sign,ty*sign),axis=-1))
        qualities.append((abs(lam)>.15)&(abs(lam)>1.8*abs(other)))
    agreement=np.sum(directions[0]*directions[1],axis=-1)
    good=qualities[0]&qualities[1]&(agreement>=cfg['tangent_minimum_scale_agreement'])
    tangent=directions[0]+directions[1]
    tangent/=np.maximum(np.linalg.norm(tangent,axis=-1,keepdims=True),1e-6)
    # Adjacent rows share a sampling orientation on the visible old layer.
    # Averaging cannot manufacture edge support and does not straighten paths.
    nx=np.array([-d[1],d[0]])
    angle=np.arctan2(tangent@nx,tangent@d)
    good&=(abs(angle)<=math.radians(65.))
    weight=cv2.GaussianBlur(good.astype(np.float32),(0,0),1.5)
    angle=cv2.GaussianBlur((angle*good).astype(np.float32),(0,0),1.5)/np.maximum(weight,1e-6)
    angle=np.where(weight>.5,angle,0.)
    tx=d[0]*np.cos(angle)-d[1]*np.sin(angle)
    ty=d[1]*np.cos(angle)+d[0]*np.sin(angle)
    return dict(tx=tx.astype(np.float32),ty=ty.astype(np.float32),
                confidence=weight.astype(np.float32),scale=scale)


def _extrema(values, floor):
    absolute = abs(values)
    return np.flatnonzero((absolute[1:-1] > absolute[:-2]) &
                         (absolute[1:-1] >= absolute[2:]) &
                         (absolute[1:-1] >= floor))+1


def _section_period(signal, old, valid, clipped, width, step):
    """Report macro optical period; never use a period as an identity label.

    Tangential aggregation suppresses diagonal strand marks. It is only a
    scale discriminator for visible old layers, and must agree across source
    sections before a half-scale bright interior can be excluded as texture.
    """
    usable=(np.mean(old&valid,axis=0)>=.8)&(np.mean(clipped,axis=0)<.25)
    periods=[]
    for lo,hi in _runs(usable):
        if hi-lo<8*width/step:continue
        values=signal[lo:hi].astype(np.float32)
        trend=cv2.GaussianBlur(values[None],(0,1),sigmaX=3.*width/step)[0]
        values=values-trend
        minimum=max(6,int(math.floor(.8*width/step)))
        maximum=min(int(math.ceil(2.3*width/step)),len(values)//5)
        if maximum-minimum<4:continue
        correlation=[]
        for lag in range(minimum,maximum+1):
            aa=values[:-lag];bb=values[lag:]
            correlation.append(float(aa@bb/max(np.linalg.norm(aa)*np.linalg.norm(bb),1e-8)))
        curve=np.asarray(correlation)
        ids=np.flatnonzero((curve[1:-1]>curve[:-2])&(curve[1:-1]>=curve[2:]))+1
        if not len(ids):continue
        maximum_value=max(curve[ids]);plausible=[i for i in ids if curve[i]>=max(.35,.82*maximum_value)]
        if not plausible:continue
        index=min(plausible)
        periods.append(dict(period_px=float((minimum+index)*step),correlation=float(curve[index]),
                            normal_interval_px=[float(lo*step),float(hi*step)]))
    return periods


def _section(image, gray, saturation, u, v, entry, d, width, search,
             spool, cfg, tangent_field=None):
    n = np.array([-d[1], d[0]])
    along = np.linspace(-cfg['along_half_widths']*width,
                        cfg['along_half_widths']*width, cfg['along_profiles'])
    centers=entry+u*d+v[:,None]*n
    if tangent_field is not None:
        workxy=centers[None]*tangent_field['scale']
        tx=_sample_scalar(tangent_field['tx'],workxy)[0]
        ty=_sample_scalar(tangent_field['ty'],workxy)[0]
        tangents=np.column_stack((tx,ty))
        norms=np.linalg.norm(tangents,axis=1)
        tangents[norms<.2]=d
        tangents/=np.maximum(np.linalg.norm(tangents,axis=1,keepdims=True),1e-8)
    else:
        tangents=np.broadcast_to(d,(len(v),2))
    xy=centers[None]+along[:,None,None]*tangents[None]
    raw = _sample_scalar(gray, xy)
    clipped = _sample_scalar(saturation, xy)
    allowed = _sample_scalar(search, xy, cv2.INTER_NEAREST) > .5
    old = _sample_scalar(spool, xy, cv2.INTER_NEAREST) > .5
    inside = ((xy[..., 0] >= 1) & (xy[..., 0] < image.shape[1]-2) &
              (xy[..., 1] >= 1) & (xy[..., 1] < image.shape[0]-2))
    valid = allowed & inside
    step = cfg['normal_step_px']
    sigma = max(.7, width*cfg['derivative_sigma_widths'])
    smooth = cv2.GaussianBlur(raw.astype(np.float32), (0, 1), sigmaX=sigma/step)
    broad = cv2.GaussianBlur(raw.astype(np.float32), (0, 1),
                             sigmaX=max(1., width*cfg['broad_sigma_widths'])/step)
    signal = np.median(smooth, axis=0)
    broad_signal = np.median(broad, axis=0)
    derivative_rows = np.gradient(smooth, step, axis=1)
    derivative = np.median(derivative_rows, axis=0)
    period_proposals=_section_period(broad_signal,old,valid,clipped,width,step)
    single_periods=[p['period_px'] for p in period_proposals if p['correlation']>=.40]
    single_macro=float(np.median(single_periods)) if single_periods else None
    peaks = _extrema(derivative, cfg['minimum_gradient'])
    pairs = []
    narrow_rejected = 0
    raw_pair_count = 0
    for ia, a in enumerate(peaks):
        for b in peaks[ia+1:]:
            measured = float(v[b]-v[a])
            if measured > cfg['maximum_width_ratio']*width:
                break
            if measured < cfg['minimum_width_ratio']*width:
                narrow_rejected += int(derivative[a]*derivative[b] < 0)
                continue
            if derivative[a]*derivative[b] >= 0:
                continue
            balance = min(abs(derivative[a]), abs(derivative[b])) / max(abs(derivative[a]),abs(derivative[b]))
            if balance < cfg['minimum_edge_balance']:
                continue
            radius = max(2, int(round(.20*measured/step)))
            lo, hi = a-radius, b+radius
            if lo < 0 or hi >= len(v) or not valid[:, lo:hi+1].all():
                continue
            polarity = 1 if derivative[a] > 0 else -1
            midlo, midhi = int(round(a+.20*(b-a))), int(round(b-.20*(b-a)))
            inside_body = float(np.median(signal[midlo:midhi+1]))
            outside_a = float(np.median(signal[lo:a]))
            outside_b = float(np.median(signal[b+1:hi+1]))
            contrast = min(polarity*(inside_body-outside_a),polarity*(inside_body-outside_b))
            if contrast < cfg['minimum_contrast']:
                continue
            broad_body = float(np.median(broad_signal[midlo:midhi+1]))
            broad_contrast = min(polarity*(broad_body-float(np.median(broad_signal[lo:a]))),
                                 polarity*(broad_body-float(np.median(broad_signal[b+1:hi+1]))))
            # A wide pair around several strand highlights can have opposed
            # gradients, but its broad body collapses across an internal groove.
            interior = polarity*(signal[a+1:b]-max(outside_a,outside_b) if polarity > 0
                                 else signal[a+1:b]-min(outside_a,outside_b))
            interior_low = float(np.percentile(interior, 20))
            strengths=[]
            for edge in (a,b):
                local=derivative_rows[:,max(0,edge-1):edge+2]*np.sign(derivative[edge])
                strengths.append(float(np.mean(np.max(local,axis=1)>=max(.7,abs(float(derivative[edge]))*.3))))
            tangent_support=min(strengths)
            saturation_fraction=float(np.mean(clipped[:,a:b+1]))
            edge_saturation=max(float(np.mean(clipped[:,max(0,a-2):a+3])),
                                float(np.mean(clipped[:,max(0,b-2):b+3])))
            baseline_saturation=max(float(np.mean(clipped[:,lo:a])),
                                    float(np.mean(clipped[:,b+1:hi+1])))
            # A clipped bright interior does not erase two visible outer
            # boundaries. Their own footprints and both outer baselines must
            # still be observable. Clipped texture remains explicitly unknown.
            raw_pair=(tangent_support>=cfg['minimum_tangential_support'] and
                      edge_saturation<=cfg['maximum_edge_saturated_fraction'] and
                      baseline_saturation<=cfg['maximum_baseline_saturated_fraction'])
            raw_pair_count += int(raw_pair)
            if not raw_pair:
                continue
            broad_ok=(broad_contrast>=max(2., contrast*.18) and interior_low>=-contrast*.30)
            old_fraction=float(np.mean(old[:,a:b+1]))
            bright_body=bool(polarity>0 and broad_ok)
            kind='body' if bright_body else 'groove' if polarity<0 else 'ambiguous'
            scale_reason=None
            if bright_body and single_macro is not None and not .62*single_macro<=measured<=1.45*single_macro:
                bright_body=False;kind='ambiguous'
                scale_reason=('half_scale_bright_interior_or_strand' if measured<.62*single_macro
                              else 'wide_pair_may_span_neighbouring_ropes')
            center_v=float((v[a]+v[b])*.5)
            center=entry+u*d+center_v*n
            local_tangent=np.median(tangents[a:b+1],axis=0)
            local_tangent/=max(float(np.linalg.norm(local_tangent)),1e-8)
            quality=float(.30*min(1.,contrast/35.)+.25*tangent_support+.20*balance+
                          .15*min(1.,max(0.,broad_contrast)/15.)+
                          .10*math.exp(-.5*(math.log(measured/width)/.45)**2))
            pairs.append(dict(center_xy=center.tolist(),v_px=center_v,
                width_px=measured,left_xy=(entry+u*d+v[a]*n).tolist(),
                right_xy=(entry+u*d+v[b]*n).tolist(),
                supported=False,strict_body=bright_body,raw_pair_supported=True,
                quality=quality,kind=kind,direction_xy=local_tangent.tolist(),
                sampling_tangent_source='two_coarse_original_Hessian_scales' if tangent_field is not None else 'free_direction',
                old_layer_fraction=old_fraction,
                tangent_support=tangent_support,contrast_gray=float(contrast),
                broad_body_contrast_gray=float(broad_contrast),
                saturation_fraction=saturation_fraction,edge_balance=float(balance),
                edge_saturation_fraction=edge_saturation,
                baseline_saturation_fraction=baseline_saturation,
                body_texture_observable=bool(saturation_fraction<=cfg['maximum_saturated_fraction']),
                identity_confirmed=False,source='original_multisample_opposed_edges'))
            if scale_reason is not None:
                pairs[-1]['ambiguity_reason']=scale_reason
                pairs[-1]['pre_nms_scale_rejected']=True
    # Keep explanations spatially; bright and dark overlap remain alternatives.
    chosen=[]
    for kind in ('body','groove','ambiguous'):
        for node in sorted((x for x in pairs if x['kind']==kind),
                           key=lambda x:(-x['quality'],x['v_px'])):
            if any(x['kind']==kind and abs(x['v_px']-node['v_px'])<cfg['nms_center_widths']*width
                   for x in chosen):
                continue
            chosen.append(node)
    # Bin round robin prevents one bright central patch consuming this budget.
    bins={}
    for node in chosen:
        key=int(math.floor(node['v_px']/(2*width)))
        bins.setdefault(key,[]).append(node)
    ordered=[]
    for rank in range(max([len(x) for x in bins.values()] or [0])):
        for key in sorted(bins):
            if rank<len(bins[key]):ordered.append(bins[key][rank])
    selected=ordered[:cfg['maximum_section_nodes']]
    selected.sort(key=lambda x:(x['v_px'],x['kind']))
    return dict(u_px=float(u), nodes=selected,
                optical_period_proposals=period_proposals,
                pre_nms_scale_ambiguous_pairs=sum(bool(n.get('pre_nms_scale_rejected')) for n in pairs),
                checked_original_samples=int(valid.sum()),
                checked_old_layer_samples=int((valid&old).sum()),
                raw_pair_count=raw_pair_count,narrow_texture_pairs_rejected=narrow_rejected,
                proposed_pairs=len(pairs),spatially_retained_nodes=len(selected))


def _macro_scale_gate(sections,width):
    """Keep the scale ambiguity explicit; macro period alone is insufficient."""
    proposals=[(i,p) for i,s in enumerate(sections) for p in s['optical_period_proposals']]
    downgraded=0;wide_downgraded=0;estimates=[]
    for i,s in enumerate(sections):
        local=[p for j,p in proposals if abs(j-i)<=12]
        if len(local)<5:continue
        period=float(np.median([p['period_px'] for p in local]))
        spread=float(np.median(abs(np.asarray([p['period_px'] for p in local])-period)))
        confidence=float(np.median([p['correlation'] for p in local]))
        if spread>.15*period or confidence<.40:continue
        s['macro_scale_px']=period;s['macro_scale_evidence_sections']=len(local)
        estimates.append(period)
        for node in s['nodes']:
            node['macro_scale_px']=period
            # A narrow high spot inside an envelope may have two optical
            # edges. Preserve its hypothesis but never draw it as whole rope.
            node['scale_consistent_with_whole_envelope']=bool(.62*period<=node['width_px']<=1.45*period)
            if node['strict_body'] and not node['scale_consistent_with_whole_envelope']:
                node['strict_body']=False;node['kind']='ambiguous'
                if node['width_px']<.62*period:
                    node['ambiguity_reason']='half_scale_bright_interior_or_strand';downgraded+=1
                else:
                    node['ambiguity_reason']='wide_pair_may_span_neighbouring_ropes';wide_downgraded+=1
    return dict(sections_with_consistent_macro_scale=len(estimates),
                median_macro_scale_px=float(np.median(estimates)) if estimates else None,
                half_scale_bodies_preserved_as_ambiguous=downgraded,
                overwide_pairs_preserved_as_ambiguous=wide_downgraded,
                period_confirms_rope_identity=False)


def _tracks(sections,width,cfg):
    active=[]; finished=[]
    maxgap=cfg['maximum_gap_widths']*width
    for si,section in enumerate(sections):
        u=section['u_px']
        # Dark valleys never become an old rope by graph continuity alone.
        nodes=[(j,node) for j,node in enumerate(section['nodes'])
               if node['strict_body'] and node['old_layer_fraction']>=.70]
        possibilities=[]
        for ti,track in enumerate(active):
            last=track['items'][-1][2]; lastu=track['items'][-1][1]
            du=u-lastu
            if du>maxgap:continue
            predicted=last['v_px']+track['slope']*du
            for ni,node in nodes:
                distance=abs(node['v_px']-predicted)
                radius=max(.62*width,.42*du)
                if distance>radius:continue
                width_change=abs(math.log(node['width_px']/last['width_px']))
                if width_change>.50:continue
                slope=(node['v_px']-last['v_px'])/du
                if abs(slope)>3.:continue
                cost=distance/radius+.35*width_change+.20*abs(slope-track['slope'])-.18*node['quality']
                possibilities.append((cost,ti,ni,slope))
        used_tracks=set();used_nodes=set()
        for cost,ti,ni,slope in sorted(possibilities):
            if ti in used_tracks or ni in used_nodes:continue
            node=section['nodes'][ni]
            active[ti]['items'].append((si,u,node))
            active[ti]['slope']=.65*active[ti]['slope']+.35*slope
            used_tracks.add(ti);used_nodes.add(ni)
        remaining=[]
        for ti,track in enumerate(active):
            if u-track['items'][-1][1]>maxgap:finished.append(track)
            else:remaining.append(track)
        active=remaining
        for ni,node in nodes:
            if ni not in used_nodes:active.append(dict(items=[(si,u,node)],slope=0.))
    finished+=active
    kept=[]
    for track in finished:
        items=track['items'];span=items[-1][1]-items[0][1]
        if len(items)<cfg['minimum_track_nodes'] or span<cfg['minimum_track_length_widths']*width:continue
        track['quality']=float(np.median([x[2]['quality'] for x in items]))
        track['span']=float(span)
        # At least three observed adjacent sections establish body continuity.
        for k,item in enumerate(items):
            neighborhood=items[max(0,k-1):min(len(items),k+2)]
            item[2]['supported']=bool(len(neighborhood)>=2 and
                all(b[1]-a[1]<=max(.8*width,12.) for a,b in zip(neighborhood[:-1],neighborhood[1:])))
        kept.append(track)
    return kept, len(finished)


def _select_diverse(tracks,width,length,cfg):
    # First representative for every actual u/v tile, then remaining quality.
    ordered=sorted(tracks,key=lambda t:(-t['span'],-t['quality'],t['items'][0][1]))
    selected=[]; covered=set()
    while ordered and len(selected)<cfg['maximum_structures']:
        choices=[]
        for i,t in enumerate(ordered):
            cells={(min(cfg['diversity_u_bins']-1,int(x[1]/length*cfg['diversity_u_bins'])),
                    int(math.floor(x[2]['v_px']/(cfg['diversity_v_widths']*width)))) for x in t['items']}
            novel=len(cells-covered)
            choices.append((novel,t['span']/length,t['quality'],-i,cells))
        winner=max(choices,key=lambda x:x[:4]);i=-winner[3]
        t=ordered.pop(i);selected.append(t);covered|=winner[4]
    return selected,covered


def _structure(track,index,entry,d,width,section_step):
    items=track['items'];observed_u=np.asarray([x[1] for x in items])
    # Include missing section locations as explicit hypotheses.
    query=np.unique(np.r_[np.arange(observed_u[0],observed_u[-1]+section_step*.25,section_step),observed_u[-1]])
    measured_v=np.asarray([x[2]['v_px'] for x in items])
    vv=np.interp(query,observed_u,measured_v)
    xy=strip_to_original(np.column_stack((query,vv)),entry,d)
    widths=np.interp(query,observed_u,[x[2]['width_px'] for x in items])
    supported=[];pair=[];left=[];right=[];records=[];texture_observable=[]
    for q,p in zip(query,xy):
        j=int(np.argmin(abs(observed_u-q)));node=items[j][2]
        observed=bool(abs(observed_u[j]-q)<section_step*.2)
        ok=bool(observed and node['supported'])
        supported.append(ok);pair.append(bool(observed and node['raw_pair_supported']))
        texture_observable.append(bool(observed and node['body_texture_observable']))
        left.append(node['left_xy'] if observed else None)
        right.append(node['right_xy'] if observed else None)
        records.append(dict(reason='continuous_whole_envelope_pair' if ok else
                            'isolated_pair_ambiguous' if observed else 'missing_section_inferred',
                            accepted=ok,pair_supported=bool(observed),
                            contrast_gray=node['contrast_gray'] if observed else None,
                            tangential_support=node['tangent_support'] if observed else None,
                            body_texture_observable=bool(observed and node['body_texture_observable']),
                            edge_saturation_fraction=node['edge_saturation_fraction'] if observed else None,
                            baseline_saturation_fraction=node['baseline_saturation_fraction'] if observed else None,
                            polarity='bright_body' if observed else None))
    sup=np.asarray(supported);inf=~sup
    return dict(id=f'B{index+1:02d}',points_xy=xy.tolist(),
        widths_px=[float(w) if p else None for w,p in zip(widths,pair)],
        measured_or_local_width_px=widths.tolist(),nominal_width_px=float(np.median(widths)),
        left_xy=left,right_xy=right,supported=supported,strict_body_supported=supported,
        candidate_supported=pair,raw_pair_supported=pair,
        body_texture_observable=texture_observable,
        inferred=inf.tolist(),unavailable=[False]*len(xy),ridge_supported=[False]*len(xy),
        supported_ranges=_runs(sup),inferred_ranges=_runs(inf),unavailable_ranges=[],
        polarity='bright_body_envelope',source='many_original_sections_then_width_and_edge_continuity',
        identity_confirmed=False,physical_center_confirmed=False,
        image_pair_does_not_resolve_rope_vs_gap=True,
        extraction_quality=track['quality'],paired_refinement_fraction=float(sup.mean()),
        paired_candidate_fraction=float(np.mean(pair)),ridge_support_fraction=0.,
        seed_uv=[float(query[len(query)//2]),float(vv[len(vv)//2])],
        observed_section_count=len(items),span_u_px=float(track['span']),
        profile_records=records)


def extract_local_bands(image_bgr,entry_xy,direction_xy,width_px,regions=None,
                        roi_xyxy=None,config=None,arc_points_xy=None):
    cfg=deepcopy(DEFAULT_CONFIG)
    if config:
        unknown=set(config)-set(cfg)
        if unknown:raise ValueError('unknown bands config: '+str(sorted(unknown)))
        cfg.update(config)
    image=np.asarray(image_bgr);entry=np.asarray(entry_xy,float);d=np.asarray(direction_xy,float)
    d=d/max(float(np.linalg.norm(d)),1e-9);width=float(width_px)
    shape=image.shape[:2]
    spool=_mask(regions,'spool_mask',shape)
    search=_mask(regions,'search_mask',shape,spool)
    if regions is not None and 'explicit_background' in regions:
        search=search&~_mask(regions,'explicit_background',shape)
    if roi_xyxy is not None:
        roi=np.zeros(shape,bool);x0,y0,x1,y1=map(int,roi_xyxy);roi[y0:y1,x0:x1]=True
        search&=roi;spool&=roi
    step=max(cfg['section_step_min_px'],width*cfg['section_step_widths'])
    u=np.arange(-cfg['search_backward_widths']*width,cfg['maximum_forward_px']+step*.25,step)
    v=np.arange(-cfg['search_half_widths']*width,cfg['search_half_widths']*width+cfg['normal_step_px']*.25,
                cfg['normal_step_px'])
    gray=np.min(image,axis=2).astype(np.float32)
    saturation=(np.max(image,axis=2)>=254).astype(np.float32)
    tangent_field=_tangent_field(gray,width,d,cfg) if cfg['adaptive_section_tangents'] else None
    sections=[_section(image,gray,saturation,float(q),v,entry,d,width,search,spool,cfg,tangent_field) for q in u]
    macro_scale=_macro_scale_gate(sections,width)
    tracks,total_tracks=_tracks(sections,width,cfg)
    selected,cells=_select_diverse(tracks,width,cfg['maximum_forward_px'],cfg)
    structures=[_structure(t,i,entry,d,width,step) for i,t in enumerate(selected)]
    corners=strip_to_original(np.array([[u[0],v[0]],[u[-1],v[0]],[u[-1],v[-1]],[u[0],v[-1]]]),entry,d)
    box=[max(0,int(corners[:,0].min())),max(0,int(corners[:,1].min())),
         min(shape[1],int(math.ceil(corners[:,0].max()))+1),min(shape[0],int(math.ceil(corners[:,1].max()))+1)]
    bins=[]
    for a,b in zip(np.linspace(u[0],u[-1],9)[:-1],np.linspace(u[0],u[-1],9)[1:]):
        local=[s for s in sections if a<=s['u_px']<=b]
        bins.append(dict(u_range_px=[float(a),float(b)],sections_checked=len(local),
            actual_original_samples=sum(s['checked_original_samples'] for s in local),
            actual_old_samples=sum(s['checked_old_layer_samples'] for s in local),
            body_nodes=sum(n['kind']=='body' for s in local for n in s['nodes']),
            groove_nodes=sum(n['kind']=='groove' for s in local for n in s['nodes'])))
    return dict(roi_xyxy=box,structures=structures,sections=sections,
        sampling=dict(u_px=u.tolist(),v_px=v.tolist(),entry_xy=entry.tolist(),direction_xy=d.tolist(),
            normal_xy=[float(-d[1]),float(d[0])],section_step_px=float(step),normal_step_px=cfg['normal_step_px']),
        config=cfg,coordinate_system='original_image_pixel_centers_xy',
        status='local_band_hypotheses' if structures else 'NO_CONTINUOUS_WHOLE_ENVELOPE_BAND',
        diagnostics=dict(entry_xy=entry.tolist(),direction_xy=d.tolist(),independent_width_px=width,
            strip_step_px=cfg['normal_step_px'],strip_u_range_px=[float(u[0]),float(u[-1])],
            strip_v_range_px=[float(v[0]),float(v[-1])],seed_u_px=u.tolist(),
            full_search_actual_sections=len(sections),search_u_bins=bins,linked_paths=total_tracks,
            unique_paths=len(tracks),retained_paths=len(selected),truncated_structures=max(0,len(tracks)-len(selected)),
            spatial_diversity_cells=len(cells),narrow_texture_pairs_rejected=sum(s['narrow_texture_pairs_rejected'] for s in sections),
            macro_scale_diagnostic=macro_scale,
            pre_nms_scale_ambiguous_pairs=sum(s['pre_nms_scale_ambiguous_pairs'] for s in sections),
            adaptive_tangents_used=cfg['adaptive_section_tangents'],
            original_image_used_for_final_profiles=True,labels_used=False,hardware_access=False,
            regularity_applies_to='visible old-layer body only; transition nodes exempt',
            groove_identity='unresolved, not promoted to a whole-rope observation',
            inference_is_image_evidence=False))


extract_local_structure=extract_local_bands
