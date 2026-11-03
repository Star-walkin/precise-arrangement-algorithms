"""V28 repair: finite paths through image sections, with explicit missing spans.

No annotation IO. Old-layer membership is a description, never a mandatory
destination. Unknown transitions can produce hypotheses but no image support.
"""
from __future__ import annotations
from copy import deepcopy
import math
import cv2
import numpy as np
from . import pipeline

DEFAULT_CONFIG = dict(beam_size=96, maximum_candidates=8, maximum_gap_widths=26.,
    maximum_observed_jump_widths=.65, maximum_slope=2.2,
    history_checkpoint_px=350., endpoint_bin_widths=.40, slope_bin=.12,
    terminal_every_px=220., minimum_forward_widths=8.,
    minimum_pair_fraction=.18, maximum_short_gap_widths=4.,
    diversity_distance_widths=.40, final_extent_bin_px=400.,
    unknown_slope_deviation=.45, continuation_minimum_extent_fraction=.60,
    maximum_partial_candidates=1, ending_phases_widths=[0.,-1.,1.,-2.,2.])


def _inside(regions, p, key='search_mask'):
    mask=regions[key]; x,y=np.rint(p).astype(int)
    return bool(0<=x<mask.shape[1] and 0<=y<mask.shape[0] and mask[y,x])


def _unavailable(image,p,width,d):
    n=np.array([-d[1],d[0]])
    xy=np.asarray(p)+np.linspace(-.55,.55,13)[:,None]*width*n
    raw=pipeline.sample(image,xy[None])[0]
    gray=raw.min(axis=-1)
    return bool((raw.max(axis=-1)>=254).mean()>.20 or np.percentile(gray,90)<12
                or np.percentile(gray,90)-np.percentile(gray,10)<8)


def _distance(a,b,width):
    pa=np.asarray(a['points_xy']);pb=np.asarray(b['points_xy'])
    # Compare only the shared forward range; avoid treating a longer tail as
    # evidence of a different connection.
    ua=pa[:,0];ub=pb[:,0]
    low=max(ua[0],ub[0]);high=min(ua[-1],ub[-1])
    if high<=low:return float('inf')
    low=max(low,4*width)
    q=np.linspace(low,high,60)
    distance=abs(np.interp(q,ua,pa[:,1])-np.interp(q,ub,pb[:,1]))
    # A short divergent bend must not be swallowed by a shared long free span.
    window=max(3,min(15,int(math.ceil(3*width/max((high-low)/59.,1.)))))
    return float(max(np.median(distance[j:j+window]) for j in range(len(distance)-window+1))/width)


def _candidate(image, path, entry, d, width, regions, index, bands, cfg):
    uv=np.asarray([[x['u'],x['v']] for x in path['items']])
    measured_uv=uv.copy()
    if len(uv)>=7:
        padded=np.pad(uv[:,1],(2,2),mode='edge')
        median=np.array([np.median(padded[j:j+5]) for j in range(len(uv))])
        uv[:,1]=cv2.GaussianBlur(median.reshape(1,-1).astype(np.float32),(7,1),1.5)[0]
        uv[0]=measured_uv[0]
        for j,q in enumerate(uv):
            p=entry+q[0]*d+q[1]*np.array([-d[1],d[0]])
            if not _inside(regions,p):uv[j]=measured_uv[j]
    xy=entry+uv[:,0,None]*d+uv[:,1,None]*np.array([-d[1],d[0]])
    nominal=np.asarray([x['width'] for x in path['items']])
    pair=np.asarray([x['node'] is not None and x['node']['raw_pair_supported'] for x in path['items']])
    body=np.asarray([x['node'] is not None and (x['node']['strict_body'] or
                     x['node'].get('anchored_free_envelope_supported',False)) for x in path['items']])
    strict=np.zeros(len(uv),bool)
    for j in range(1,len(uv)-1):
        strict[j]=bool(body[j-1:j+2].all() and abs(uv[j,1]-measured_uv[j,1])<=.15*nominal[j] and
                       np.max(nominal[j-1:j+2])/np.min(nominal[j-1:j+2])<1.45 and
                       abs((uv[j+1,1]-uv[j,1])-(uv[j,1]-uv[j-1,1]))<.45*width)
    # Do not densify measured markers into unobserved centerline segments.
    left=[x['node']['left_xy'] if x['node'] is not None else None for x in path['items']]
    right=[x['node']['right_xy'] if x['node'] is not None else None for x in path['items']]
    unavailable=np.asarray([x['unavailable'] and not strict[j] for j,x in enumerate(path['items'])])
    inferred=~strict&~unavailable
    old_match=[]
    compatible_old_bands=[]
    for band in bands['structures']:
        bp=np.asarray(band['points_xy'])
        if len(bp):
            dist=np.sqrt(((xy[:,None]-bp[None])**2).sum(axis=2)).min(axis=1)
            matches=(dist<.4*width)&strict
            if matches.any():compatible_old_bands.append(band['id'])
            # A crossing can touch the same old-band envelope at scattered
            # sections. Those touches do not constitute an old-row join.
            ids=np.flatnonzero(matches)
            runs=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1) if len(ids) else []
            longest=max([float(np.linalg.norm(np.diff(xy[run],axis=0),axis=1).sum()) for run in runs if len(run)>1] or [0.])
            if longest>=4*width:old_match.append((longest,band['id']))
    structure_id=max(old_match)[1] if old_match and not path.get('force_new_kind') else None
    join_indices=np.flatnonzero(pair & (uv[:,0]>=3*width))
    join_index=int(join_indices[0]) if len(join_indices) else None
    if structure_id:
        band=next(b for b in bands['structures'] if b['id']==structure_id)
        bp=np.asarray(band['points_xy'])
        matches=(np.sqrt(((xy[:,None]-bp[None])**2).sum(axis=2)).min(axis=1)<.4*width)&strict
        if matches.any():join_index=int(np.flatnonzero(matches)[0])
    evidence=pipeline.profile_evidence(image,xy,nominal)
    # Raw legacy profiles remain separate, especially opposite dark groove pairs.
    old_points=[_inside(regions,p,'spool_mask') for p in xy]
    return dict(id=f'R{index+1}',kind='old_row_connection' if structure_id else 'new_segment_may_cross_old_rows',
        structure_id=structure_id,structure_phase='visible_old_layer' if structure_id else 'cross_layer_unknown',
        compatible_old_band_ids=compatible_old_bands,
        possible_connection_types=['old_row_connection','new_segment_may_cross_old_rows'] if structure_id else ['new_segment_may_cross_old_rows','UNRESOLVED'],
        join_xy=xy[join_index].tolist() if join_index is not None else None,
        join_source='observed_old_band' if structure_id else 'first_post_free_original_pair',
        join_is_identity_confirmation=False,
        inferred_prefix_px=float(uv[join_index,0]) if join_index is not None else float(uv[-1,0]),
        generator='incoming_bright_dark_envelope_hypothesis' if path.get('force_new_kind') else 'diverse_section_graph_v28_repair',
        points_xy=xy.tolist(),widths_px=nominal.tolist(),supported=strict.tolist(),
        strict_body_supported=strict.tolist(),raw_pair_supported=pair.tolist(),
        inferred=inferred.tolist(),unavailable=unavailable.tolist(),left_xy=left,right_xy=right,
        evidence=evidence,original_section_u_px=uv[:,0].tolist(),
        original_measured_section_center_xy=(entry+measured_uv[:,0,None]*d+measured_uv[:,1,None]*np.array([-d[1],d[0]])).tolist(),
        old_layer_membership=old_points,proposal_cost=float(path['cost']/max(1,len(path['items'])-1)),
        terminal_reason=path.get('stop','search_extent_or_snapshot'),
        strict_body_note='bright whole envelope at three original adjacent sections; no identity assertion',
        physical_identity_confirmed=False,depth_order='UNKNOWN',
        image_pair_does_not_resolve_rope_vs_gap=True)


def generate_pool(image,free_seed,width_px,bands,regions,config=None):
    cfg=deepcopy(DEFAULT_CONFIG)
    if config:cfg.update(config)
    width=float(width_px);entry=np.asarray(free_seed[-1],float)
    d=pipeline.unit(np.asarray(free_seed[-1])-free_seed[0]);n=np.array([-d[1],d[0]])
    sections=[s for s in bands['sections'] if s['u_px']>.1]
    active=[dict(items=[dict(u=0.,v=0.,width=width,node=None,unavailable=False,step_cost=0.)],
                 slope=0.,gap=0.,short_gap=0.,cost=0.,history=(),observed_slope=0.)]
    terminals=[];logs=[];next_snapshot=cfg['terminal_every_px']
    for section in sections:
        u=section['u_px']; expanded=[]
        for path in active:
            last=path['items'][-1];du=u-last['u']
            if du<=.01:continue
            pred=last['v']+path['slope']*du
            radius=max(cfg['maximum_observed_jump_widths']*width,.50*du)
            nodes=[node for node in section['nodes'] if abs(node['v_px']-pred)<=radius]
            for node in nodes:
                slope=(node['v_px']-last['v'])/du
                if abs(slope)>cfg['maximum_slope']:continue
                width_delta=abs(math.log(node['width_px']/last['width']))
                if width_delta>.60:continue
                acceleration=abs(slope-path['slope'])
                if acceleration>1.2 and path['gap']<width:continue
                # Grooves remain competing hypotheses; they cannot earn body support.
                stepcost=(.45*(1-node['quality'])+.16*(node['kind']!='body')+
                          .28*abs(node['v_px']-pred)/width+.14*width_delta+
                          .22*acceleration)
                item=dict(u=u,v=node['v_px'],width=node['width_px'],node=node,unavailable=False,step_cost=stepcost)
                history=path['history']
                if int(u/cfg['history_checkpoint_px'])>int(last['u']/cfg['history_checkpoint_px']):
                    history=(*history,int(round(item['v']/(.7*width))))[-4:]
                expanded.append(dict(items=path['items']+[item],slope=.45*path['slope']+.55*slope,
                    gap=0.,short_gap=0.,cost=path['cost']+stepcost,history=history,
                    observed_slope=.45*path['slope']+.55*slope))
            # A missing source section is kept as a dotted hypothesis. Glare or
            # dark regions allow a finite longer bridge, never a measured band.
            for ds in (0.,-.035,.035) if path['gap']>=3*width else (0.,):
                slope=path['slope']+ds
                if abs(slope)>cfg['maximum_slope']:continue
                if abs(slope-path['observed_slope'])>cfg['unknown_slope_deviation']:continue
                v=last['v']+slope*du;p=entry+u*d+v*n
                if not _inside(regions,p) or _inside(regions,p,'explicit_background'):continue
                unavailable=_unavailable(image,p,last['width'],d)
                gap=path['gap']+du;shortgap=path['short_gap']+(0. if unavailable else du)
                if gap>cfg['maximum_gap_widths']*width or shortgap>cfg['maximum_short_gap_widths']*width:continue
                history=path['history']
                if int(u/cfg['history_checkpoint_px'])>int(last['u']/cfg['history_checkpoint_px']):
                    history=(*history,int(round(v/(.7*width))))[-4:]
                stepcost=.70+.06*abs(ds)/.035
                item=dict(u=u,v=v,width=last['width'],node=None,unavailable=unavailable,step_cost=stepcost)
                expanded.append(dict(items=path['items']+[item],slope=slope,gap=gap,short_gap=shortgap,
                    cost=path['cost']+stepcost,history=history,observed_slope=path['observed_slope']))
            if not any(x['items'][-2] is last for x in expanded):
                terminal=dict(path,stop='region_or_source_support_gap_limit')
                terminals.append(terminal)
        # Endpoint AND historical branches survive; equal endpoints do not erase
        # an earlier crossing. Spatial round robin precedes the finite beam cap.
        unique={}
        for path in sorted(expanded,key=lambda x:x['cost']):
            last=path['items'][-1]
            key=(int(round(last['v']/(cfg['endpoint_bin_widths']*width))),
                 int(round(path['slope']/cfg['slope_bin'])),path['history'])
            if key not in unique:unique[key]=path
        cells={}
        for path in unique.values():
            last=path['items'][-1];key=(int(round(last['v']/width)),int(round(path['slope']/.3)))
            cells.setdefault(key,[]).append(path)
        active=[]
        for rank in range(max([len(v) for v in cells.values()] or [0])):
            for key in sorted(cells,key=lambda k:cells[k][0]['cost']):
                if rank<len(cells[key]):active.append(cells[key][rank])
            if len(active)>=cfg['beam_size']:break
        active=active[:cfg['beam_size']]
        logs.append(dict(u_px=float(u),source_nodes=len(section['nodes']),expanded=len(expanded),
            endpoint_history_states=len(unique),retained=len(active),terminal_count=len(terminals)))
        if u>=next_snapshot:
            terminals.extend(dict(p,stop='intermediate_extent_snapshot') for p in active)
            next_snapshot+=cfg['terminal_every_px']
        if not active:break
    terminals.extend(dict(p,stop='search_extent') for p in active)
    eligible=[]
    for p in terminals:
        observed=[j for j,x in enumerate(p['items']) if x['node'] is not None]
        if not observed:continue
        # A dark-region extrapolation tail must not win a final slot by length.
        last_observed=observed[-1]
        p=dict(p,items=p['items'][:last_observed+1])
        p['cost']=sum(x['step_cost'] for x in p['items'][1:])
        pair=sum(x['node'] is not None for x in p['items'])/len(p['items'])
        if p['items'][-1]['u']>=cfg['minimum_forward_widths']*width and pair>=cfg['minimum_pair_fraction']:
            eligible.append(p)
    # At most eight final routes, with ending range and path diversity. Rich
    # intermediate structures have already been inspected across the full ROI.
    eligible.sort(key=lambda p:p['cost']/len(p['items']))
    bins={}
    for p in eligible:
        last=p['items'][-1]
        key=(int(last['u']/cfg['final_extent_bin_px']),int(round(last['v']/(2*width))))
        bins.setdefault(key,[]).append(p)
    ordered=[]
    # Visit source-backed ending tiles before any tile gets a second path.
    # Central continuation and crossing ends therefore survive longer tails.
    extents={}
    for key,paths in bins.items():extents.setdefault(key[0],[]).append((key,paths))
    extent_orders={}
    for extent,tiles in extents.items():
        local=[]
        tiles.sort(key=lambda pair:pair[1][0]['cost']/len(pair[1][0]['items']))
        for rank in range(max(len(v) for _,v in tiles)):
            local.extend(v[rank] for _,v in tiles if rank<len(v))
        extent_orders[extent]=local
    for rank in range(max([len(v) for v in extent_orders.values()] or [0])):
        for extent in sorted(extent_orders,reverse=True):
            if rank<len(extent_orders[extent]):ordered.append(extent_orders[extent][rank])
    # Reserve adjacent continuation hypotheses around the incoming ray before
    # far old-layer tiles consume the entire cap. Each is still traced through
    # the source sections; this is neither a forced row nor a synthetic ray.
    farthest=max([p['items'][-1]['u'] for p in eligible] or [0.])
    # A dark silhouette and a bright old row are different image explanations.
    # Keeping only the brighter family silently forces the continuation onto
    # the visible old layer. Retain finite alternatives from the SAME graph.
    families={'dark':[],'bright':[],'ambiguous':[]}
    for p in eligible:
        if p['items'][-1]['u']<cfg['continuation_minimum_extent_fraction']*farthest:continue
        nodes=[x['node'] for x in p['items'] if x['node'] is not None]
        bright=sum(n['kind']=='body' for n in nodes)/max(1,len(nodes))
        dark=sum(n['kind']=='groove' for n in nodes)/max(1,len(nodes))
        family='dark' if dark>=.60 else 'bright' if bright>=.60 else 'ambiguous'
        families[family].append(p)
    for family in families:
        families[family].sort(key=lambda p:p['cost']/len(p['items']))
    # Alternate source-polarity families, then spatial tiles. This is candidate
    # diversity, not evidence that a groove has become a rope.
    family_order=[]
    for rank in range(max([len(v) for v in families.values()] or [0])):
        for family in ('dark','bright','ambiguous'):
            if rank<len(families[family]):family_order.append(families[family][rank])
        if len(family_order)>=30:break
    # Five adjacent ending phases around the measured incoming geometry. This
    # keeps near-ray new/crossing explanations alongside bright old-layer rows,
    # rather than spending all eight slots on one high-response phase family.
    # Endpoints must be real source pairs; no extrapolation is an anchor.
    complete=[p for p in eligible if p['items'][-1]['u']>=cfg['continuation_minimum_extent_fraction']*farthest]
    neighboring=[]
    for phase in cfg['ending_phases_widths']:
        if not complete:break
        def endpoint_cost(p):
            end=p['items'][-1]
            return abs(end['v']/width-phase)+.25*p['cost']/len(p['items'])
        neighboring.append(min(complete,key=endpoint_cost))
    partials=[p for p in eligible if p['items'][-1]['u']<cfg['continuation_minimum_extent_fraction']*farthest
              and p.get('stop')=='region_or_source_support_gap_limit']
    partials.sort(key=lambda p:p['cost']/len(p['items']))
    # Keep one source-backed stopping explanation. Longer rows must not force
    # the active rope past an unresolved gap simply by occupying every slot.
    ordered=partials[:cfg['maximum_partial_candidates']]+neighboring+family_order+ordered
    selected=[];accepted_uv=[];partial_count=0
    for path in ordered:
        uv=dict(points_xy=[[x['u'],x['v']] for x in path['items']])
        partial=uv['points_xy'][-1][0]<cfg['continuation_minimum_extent_fraction']*farthest
        if partial and partial_count>=cfg['maximum_partial_candidates']:continue
        if any(_distance(uv,other,width)<cfg['diversity_distance_widths'] and
               abs(uv['points_xy'][-1][0]-other['points_xy'][-1][0])<cfg['terminal_every_px']*1.5
               for other in accepted_uv):continue
        candidate=_candidate(image,path,entry,d,width,regions,len(selected),bands,cfg)
        candidate['partial_stop']=bool(partial)
        selected.append(candidate);accepted_uv.append(uv)
        partial_count+=int(partial)
        if len(selected)>=cfg['maximum_candidates']:break
    return dict(schema='v28_repair_pool_1',coordinate_system='full_original_image_xy',
        image_shape_hw=list(image.shape[:2]),origin_xy=entry.tolist(),direction_xy=d.tolist(),
        normal_xy=n.tolist(),free_seed=np.asarray(free_seed).tolist(),width_px=width,
        candidates=selected,candidate_cap=cfg['maximum_candidates'],
        local_arrangement=bands['structures'],config=cfg,truncated=len(eligible)>len(selected),
        diagnostics=dict(section_search_log=logs,all_structure_sections_checked=len(bands['sections']),
            connection_sections_reached=len(logs),terminal_hypotheses=len(terminals),
            eligible_hypotheses=len(eligible),polarity_families={key:len(value) for key,value in families.items()},
            ending_phase_hypotheses=len(neighboring),no_forced_old_row=True),
        input_uses_labels=False,geometry_completion_is_evidence=False,
        automatic_entry_success=True,input_mode='AUTO',connection_decision='UNKNOWN')
