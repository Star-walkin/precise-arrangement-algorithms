"""F3: retain F2 and add supported full-surface D4 fragments only inside old T.

All identifiers describe image-row correspondence, not physical rope identity.
This pure-array module never opens an image or imports a device/detector.
"""
from __future__ import annotations
import math
from collections import Counter
import cv2
import numpy as np

PARAMETERS = {
    'reference_residual_pitch': .27,
    'reference_minimum_overlap_pitch': 1.0,
    'reference_tangent_error': .40,
    'minimum_fragment_span_pitch': .85,
    'minimum_fragment_points': 4,
    'minimum_observation_span_fraction': .45,
    'minimum_broad_fraction': .70,
    'broad_threshold': .10,
    'maximum_unsupported_run_pitch': .60,
    'body_hard_threshold': .65,
    'existing_duplicate_radius_pitch': .40,
    'independent_neighbor_minimum_pitch': .55,
    'independent_neighbor_maximum_pitch': 1.65,
    'independent_minimum_span_pitch': 1.50,
    'independent_minimum_points': 7,
    'independent_broad_fraction': .88,
    'maximum_endpoint_extension_pitch': .70,
    'maximum_endpoint_snap_pitch': .55,
    'new_fragment_separation_pitch': .55,
}


def _runs(mask):
    z=np.r_[False,np.asarray(mask,bool),False].astype(np.int8)
    return list(zip(np.flatnonzero(np.diff(z)==1),np.flatnonzero(np.diff(z)==-1)))


def _sample(layer,xy,x0=0,y0=0,outside=0):
    q=np.rint(xy).astype(np.int32);x=q[:,0]-x0;y=q[:,1]-y0
    good=(x>=0)&(y>=0)&(x<layer.shape[1])&(y<layer.shape[0])
    out=np.full(len(q),outside,dtype=layer.dtype)
    out[good]=layer[y[good],x[good]]
    return out


def _field(fields,name,xy,outside=0):
    return _sample(fields[name],xy,int(fields['x0']),int(fields['y0']),outside)


def _raster(xy):
    rr=np.rint(xy).astype(np.int32)
    if not len(rr):return rr.reshape(-1,2)
    low=rr.min(axis=0);high=rr.max(axis=0)
    patch=np.zeros((int(high[1]-low[1]+1),int(high[0]-low[0]+1)),np.uint8)
    cv2.polylines(patch,[rr-low],False,255,1,cv2.LINE_8)
    yy,xx=np.nonzero(patch)
    return np.c_[xx+low[0],yy+low[1]].astype(np.int32)


def _longest(xy,mask):
    return max((float(np.linalg.norm(np.diff(xy[a:z],axis=0),axis=1).sum())+1 for a,z in _runs(mask)),default=0.)


def _references(data):
    # F2 already records accepted visual correspondence. Union its source IDs so
    # C3 and D4 pieces of a saved bridge are not mistaken for two rival rows.
    parent={}
    def find(k):
        parent.setdefault(k,k)
        if parent[k]!=k:parent[k]=find(parent[k])
        return parent[k]
    def union(a,b):parent[find(b)]=find(a)
    for j in data.get('f2_case',{}).get('joins',[]):
        ids=j.get('source_curve_ids',[])
        if len(ids)==2:union(ids[0],ids[1])
    refs=[];base_near=cv2.dilate(data['f2_merged'],np.ones((3,3),np.uint8))
    for source,key in [('C3','c3_record'),('D4','d4_record')]:
        for c in data.get(key,{}).get('fit',{}).get('curves',[]):
            xy=np.asarray(c.get('sampled_points_xy',[]),float).reshape(-1,2)
            if len(xy)<3:continue
            cid=source+':'+str(c['curve_id']);find(cid)
            for a,z in _runs(_sample(base_near,xy)>0):
                if z-a>=3:refs.append({'id':cid,'xy':xy[a:z],'group':cid})
    for j in data.get('f2_case',{}).get('joins',[]):
        xy=np.asarray(j.get('sampled_points_xy',[]),float).reshape(-1,2)
        ids=j.get('source_curve_ids',[])
        if len(xy)>2 and ids:refs.append({'id':'F2:'+j['candidate_id'],'xy':xy,'group':ids[0]})
    for r in refs:r['group']=find(r['group'])
    return refs


def _match_reference(xy,refs,pitch):
    # Match full trajectories, including bright-area support outside T. Several
    # references belonging to the same saved F2 group count as one image row.
    matches=[];matched_references=[]
    for r in refs:
        q=r['xy'];lo=max(xy[0,0],q[0,0]);hi=min(xy[-1,0],q[-1,0])
        if hi-lo<PARAMETERS['reference_minimum_overlap_pitch']*pitch:continue
        xs=np.arange(math.ceil(lo),math.floor(hi)+1,max(1,int(pitch/5)))
        yy=np.interp(xs,xy[:,0],xy[:,1]);ry=np.interp(xs,q[:,0],q[:,1])
        res=np.abs(yy-ry);near=res<=PARAMETERS['reference_residual_pitch']*pitch
        spans=[(a,z) for a,z in _runs(near) if xs[z-1]-xs[a]>=PARAMETERS['reference_minimum_overlap_pitch']*pitch]
        if not spans:continue
        a,z=max(spans,key=lambda s:s[1]-s[0]);err=float(np.median(res[a:z]))
        tangent=float(np.median(np.abs(np.gradient(yy[a:z],xs[a:z])-np.gradient(ry[a:z],xs[a:z]))))
        if tangent>PARAMETERS['reference_tangent_error']:continue
        matches.append({'reference_id':r['id'],'visual_group':r['group'],'median_residual_px':err,
                        'support_span_x':[float(xs[a]),float(xs[z-1])],'tangent_error':tangent})
        matched_references.append(r)
    # Previously unconnected bright/dark pieces have different old IDs. A real
    # long new trajectory may support both at disjoint x intervals. Reject only
    # competing separated rows in the SAME x interval; different IDs alone do
    # not establish either a conflict or physical material identity.
    ambiguous=False
    for i,(ma,ra) in enumerate(zip(matches,matched_references)):
        for mb,rb in zip(matches[i+1:],matched_references[i+1:]):
            if ma['visual_group']==mb['visual_group']:continue
            lo=max(ma['support_span_x'][0],mb['support_span_x'][0])
            hi=min(ma['support_span_x'][1],mb['support_span_x'][1])
            if hi-lo<.25*pitch:continue
            xs=np.arange(math.ceil(lo),math.floor(hi)+1)
            dy=np.abs(np.interp(xs,ra['xy'][:,0],ra['xy'][:,1])-np.interp(xs,rb['xy'][:,0],rb['xy'][:,1]))
            if len(dy) and np.median(dy)>.15*pitch:ambiguous=True
    return matches,ambiguous


def _baseline_lookup(base,t,pitch):
    yy,xx=np.nonzero(t)
    if not len(xx):return None
    margin=int(math.ceil(2*pitch));h,w=base.shape
    x0=max(0,int(xx.min())-margin);x1=min(w,int(xx.max())+margin+1)
    y0=max(0,int(yy.min())-margin);y1=min(h,int(yy.max())+margin+1)
    crop=base[y0:y1,x0:x1]>0
    if not np.any(crop):
        return {'x0':x0,'y0':y0,'base':crop,'distance':np.full(crop.shape,1e6,np.float32),'labels':np.zeros(crop.shape,np.int32),'zero_xy':np.empty((0,2),int)}
    distance,labels=cv2.distanceTransformWithLabels((~crop).astype(np.uint8),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
    zy,zx=np.nonzero(crop)
    return {'x0':x0,'y0':y0,'base':crop,'distance':distance,'labels':labels,'zero_xy':np.c_[zx+x0,zy+y0]}


def _distance(lookup,xy):return _sample(lookup['distance'],xy,lookup['x0'],lookup['y0'],1e6)


def _nearest(lookup,point):
    lab=int(_sample(lookup['labels'],np.asarray(point)[None,:],lookup['x0'],lookup['y0'],0)[0])
    return lookup['zero_xy'][lab-1].astype(float) if lab>0 else None


def _neighbors(lookup,xy,pitch):
    # Independent fragments need the expected empty row between two existing
    # rows. This is only a spacing constraint, not a source of yellow points.
    samples=xy[np.linspace(0,len(xy)-1,min(9,len(xy))).astype(int)];good=0;details=[]
    for x,y in samples:
        col=int(round(x))-lookup['x0'];row=y-lookup['y0']
        if col<0 or col>=lookup['base'].shape[1]:continue
        ys=np.flatnonzero(lookup['base'][:,max(0,col-1):min(lookup['base'].shape[1],col+2)].any(axis=1))
        top=ys[ys<row];bottom=ys[ys>row]
        if not len(top) or not len(bottom):continue
        a=float(row-top[-1]);b=float(bottom[0]-row)
        okay=min(a,b)>=PARAMETERS['independent_neighbor_minimum_pitch']*pitch and max(a,b)<=PARAMETERS['independent_neighbor_maximum_pitch']*pitch
        good+=int(okay);details.append([float(x),a,b,bool(okay)])
    return good/max(1,len(samples)),details


def _old_endpoint_tangent(lookup,target,direction,pitch,reference_slope):
    """Estimate old-side slope from existing raster only, in a narrow corridor.

    direction points outward from the new fragment. Neighboring rows are not
    allowed to provide the old endpoint slope when the old row is too short.
    """
    values=[]
    for step in range(0,int(math.ceil(1.1*pitch))+1):
        x=int(round(target[0]+direction*step));col=x-lookup['x0']
        if not 0<=col<lookup['base'].shape[1]:continue
        expected=target[1]+reference_slope*(x-target[0])-lookup['y0']
        lo=max(0,int(math.floor(expected-.25*pitch)));hi=min(lookup['base'].shape[0],int(math.ceil(expected+.25*pitch))+1)
        ys=np.flatnonzero(lookup['base'][lo:hi,col])+lo
        if len(ys):values.append([float(x),float(ys[np.argmin(abs(ys-expected))]+lookup['y0'])])
    if len(values)<4:return None
    q=np.asarray(values);dx=q[:,0]-target[0]
    if np.ptp(dx)<.40*pitch:return None
    coef=np.polyfit(dx,q[:,1]-target[1],1)
    residual=np.abs(q[:,1]-target[1]-np.polyval(coef,dx))
    if np.percentile(residual,90)>.12*pitch:return None
    return float(coef[0])


def _endpoint_window_angle(xy,direction):
    """Measure visible 3px-scale turns, ignoring harmless subpixel raster steps."""
    q=xy if direction<0 else xy[::-1]
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))]
    n=min(9.,float(arc[-1]))
    if n<6:return 0.
    ss=np.arange(0,n+.01,3.)
    pts=np.c_[np.interp(ss,arc,q[:,0]),np.interp(ss,arc,q[:,1])]
    v=np.diff(pts,axis=0);v/=np.maximum(1e-9,np.linalg.norm(v,axis=1))[:,None]
    return float(np.degrees(np.arccos(np.clip(np.sum(v[1:]*v[:-1],axis=1),-1,1))).max()) if len(v)>1 else 0.


def _extend_endpoint(q,full,a,z,which,lookup,t,fields,pitch):
    """Bounded cubic end correction instead of a same-x vertical snap.

    Old F2 pixels never move. The correction matches the actual old endpoint
    and its measured tangent, fades to zero with zero derivative inside the
    new D4 path, and is validated again on every raster pixel. Failure returns
    the untouched measured fragment, not an unsupported forced connector.
    """
    end=q[0] if which<0 else q[-1];idx=a if which<0 else z-1
    limit=int(math.ceil(PARAMETERS['maximum_endpoint_extension_pitch']*pitch))
    direction=-1 if which<0 else 1;choices=[]
    for step in range(0,limit+1):
        k=idx+direction*step
        if not 0<=k<len(full):break
        p=full[k]
        if not _sample(t,p[None,:])[0]:break
        target=_nearest(lookup,p)
        if target is None:continue
        delta=float(np.linalg.norm(target-p))
        if delta>PARAMETERS['maximum_endpoint_snap_pitch']*pitch:continue
        if (target[0]-end[0])*direction<0 or (target[0]-p[0])*direction<0:continue
        if not full[0,0]<=target[0]<=full[-1,0]:continue
        source_slope=float((full[min(len(full)-1,k+3),1]-full[max(0,k-3),1])/max(1.,full[min(len(full)-1,k+3),0]-full[max(0,k-3),0]))
        source_at_target=float(np.interp(target[0],full[:,0],full[:,1]))
        vertical_delta=float(target[1]-source_at_target)
        if abs(vertical_delta)>.22*pitch:continue
        old_slope=_old_endpoint_tangent(lookup,target,direction,pitch,source_slope)
        if old_slope is None or abs(old_slope-source_slope)>.35:continue
        lo=target[0] if which<0 else q[0,0];hi=q[-1,0] if which<0 else target[0]
        window=min(1.25*pitch,.40*(hi-lo))
        if window<max(6.,.35*pitch):continue
        xs=np.arange(math.ceil(lo),math.floor(hi)+1,dtype=float)
        ys=np.interp(xs,full[:,0],full[:,1])
        inside=(xs>=q[0,0])&(xs<=q[-1,0])
        ys[inside]=np.interp(xs[inside],q[:,0],q[:,1])
        inward=-direction;distance=(xs-target[0])*inward;u=np.clip(distance/window,0,1)
        near=(distance>=0)&(distance<=window)
        # h00 sets endpoint offset; h10 sets derivative offset. Both values and
        # derivatives vanish at the inner end, giving a C1 correction there.
        offset=vertical_delta*(2*u**3-3*u*u+1)+window*inward*(old_slope-source_slope)*(u**3-2*u*u+u)
        offset[~near]=0
        if np.max(abs(offset))>.25*pitch:continue
        ys+=offset;path=np.c_[xs,ys]
        if _endpoint_window_angle(path,direction)>25.:continue
        rr=_raster(path)
        if np.any(_sample(t,rr)==0) or np.any(_field(fields,'unknown_mask',rr)>0) or np.any(_field(fields,'body_barrier',rr,1)>PARAMETERS['body_hard_threshold']):continue
        changed=rr[(rr[:,0]-target[0])*inward<=window]
        if len(changed) and np.mean(_field(fields,'broad_score',changed)>=PARAMETERS['broad_threshold'])<.70:continue
        choices.append((delta+.05*step+.10*abs(old_slope-source_slope),path,target))
    if not choices:return q,None
    _,path,target=min(choices,key=lambda x:x[0]);return path,target.tolist()


def fuse_surface(data,detected):
    """Add accepted full-surface observations in old F2 T; never replace F2.

    source_point_indices are LOCAL to the original D4 track, not global IDs.
    Accepted yellow points are original new measurements near added geometry.
    """
    pitch=float(data['pitch']);base=np.asarray(data['f2_merged'],np.uint8)
    if not math.isfinite(pitch) or not 8<=pitch<=70:raise ValueError('pitch must be 8..70 pixels')
    t=((np.asarray(data['f2_transition'])>0)&(data['surface']>0)).astype(np.uint8)*255
    fields=detected['fields'];lookup=_baseline_lookup(base,t,pitch)
    new=np.zeros_like(base);new_inferred=np.zeros_like(base);removed=np.zeros_like(base)
    groups=[];reject=[];accepted_points=[];point_ids=set();refs=_references(data)
    tracks={str(v['track_id']):v for v in detected.get('tracks',[])}
    diagnostics={'parameters':PARAMETERS.copy(),'old_pixels_removed':0,'source':'actual_full_surface_D4_detection','physical_material_identity':False}
    if lookup is None:
        return {'merged_mask':base.copy(),'inferred_mask':data['f2_inferred'].copy(),'new_mask':new,'removed_mask':removed,'transition_mask':t,'accepted_points':[],'groups':[],'rejections':[],'diagnostics':{**diagnostics,'accepted_groups':0,'reason':'empty_old_transition'}}
    proposals=[]
    for c in detected.get('fit',{}).get('curves',[]):
        xy=np.asarray(c.get('sampled_points_xy',[]),float).reshape(-1,2)
        if len(xy)<3:continue
        xy=xy[np.argsort(xy[:,0],kind='stable')]
        if np.any(np.diff(xy[:,0])<=0) or not np.isfinite(xy).all():
            reject.append({'curve_id':c.get('curve_id'),'reason':'invalid_or_multivalued_path'});continue
        local=tracks.get(str(c.get('track_id')),{}).get('points',[])
        points=[local[int(i)] for i in c.get('source_point_indices',[]) if 0<=int(i)<len(local)]
        if not points:
            reject.append({'curve_id':c['curve_id'],'reason':'missing_source_point_mapping'});continue
        matches,ambiguous=_match_reference(xy,refs,pitch)
        if ambiguous:
            reject.append({'curve_id':c['curve_id'],'reason':'full_track_matches_different_saved_rows','reference_matches':matches});continue
        in_t=_sample(t,xy)>0
        # Intersect hard gates before cutting a missing fragment.
        clear=in_t&(_field(fields,'unknown_mask',xy)==0)&(_field(fields,'body_barrier',xy,1)<=PARAMETERS['body_hard_threshold'])
        missing=_distance(lookup,xy)>PARAMETERS['existing_duplicate_radius_pitch']*pitch
        for a,z in _runs(clear&missing):
            q=xy[a:z]
            gid=f"F3_{c['curve_id']}__part{a:05d}"
            rec={'group_id':gid,'source_curve_id':c['curve_id'],'track_id':str(c['track_id']),'reference_matches':matches,'status':'rejected'}
            if len(q)<2 or q[-1,0]-q[0,0]<PARAMETERS['minimum_fragment_span_pitch']*pitch:
                reject.append({**rec,'reason':'duplicate_or_too_short_missing_fragment'});continue
            q,la=_extend_endpoint(q,xy,a,z,-1,lookup,t,fields,pitch)
            q,ra=_extend_endpoint(q,xy,a,z,1,lookup,t,fields,pitch)
            rr=_raster(q);broad=_field(fields,'broad_score',rr);supported=broad>=PARAMETERS['broad_threshold']
            reasons=[]
            if np.any(_sample(t,rr)==0):reasons.append('outside_original_transition')
            if np.any(_field(fields,'unknown_mask',rr)>0):reasons.append('unknown_crossing')
            if np.any(_field(fields,'body_barrier',rr,1)>PARAMETERS['body_hard_threshold']):reasons.append('supported_body_crossing')
            support_fraction=float(np.mean(supported));long_gap=_longest(q,_field(fields,'broad_score',q)<PARAMETERS['broad_threshold'])
            if support_fraction<PARAMETERS['minimum_broad_fraction']:reasons.append('insufficient_broad_groove_fraction')
            if long_gap>PARAMETERS['maximum_unsupported_run_pitch']*pitch:reasons.append('long_unsupported_path')
            lo=float(np.min(q[:,0]));hi=float(np.max(q[:,0]));ordered=q[np.argsort(q[:,0],kind='stable')]
            pts=[]
            for p in points:
                x,y=float(p['x']),float(p['y'])
                if lo<=x<=hi and abs(y-np.interp(x,ordered[:,0],ordered[:,1]))<=.30*pitch:
                    at=np.asarray([[x,y]])
                    if _sample(t,at)[0] and not _field(fields,'unknown_mask',at)[0] and _field(fields,'body_barrier',at,1)[0]<=PARAMETERS['body_hard_threshold']:
                        pts.append(p)
            unique={tuple(np.rint([p['x'],p['y']]).astype(int)) for p in pts}
            obs_span=(max(p['x'] for p in pts)-min(p['x'] for p in pts))/max(1,hi-lo) if pts else 0.
            if len(unique)<PARAMETERS['minimum_fragment_points'] or obs_span<PARAMETERS['minimum_observation_span_fraction']:reasons.append('insufficient_new_observations_on_fragment')
            mode='anchored_visual_row' if matches else 'independent_supported_fragment'
            neighbor_fraction,neighbor_detail=_neighbors(lookup,q,pitch)
            if not matches:
                if len(unique)<PARAMETERS['independent_minimum_points'] or hi-lo<PARAMETERS['independent_minimum_span_pitch']*pitch:reasons.append('independent_fragment_too_short')
                if support_fraction<PARAMETERS['independent_broad_fraction']:reasons.append('independent_fragment_weak_support')
                if neighbor_fraction<.70:reasons.append('independent_fragment_spacing_not_supported')
            stats={'span_px':hi-lo,'new_unique_points':len(unique),'observation_span_fraction':float(obs_span),'broad_fraction':support_fraction,'longest_unsupported_px':long_gap,'neighbor_valid_fraction':neighbor_fraction}
            rec.update({'mode':mode,'anchors':{'left':la,'right':ra},'statistics':stats,'neighbor_checks':neighbor_detail})
            if reasons:reject.append({**rec,'reason':'+'.join(reasons)});continue
            score=support_fraction+.20*min(1,len(unique)/12)+.10*bool(matches)
            proposals.append((score,rec,q,rr,pts))
    kept=[]
    for score,rec,q,rr,pts in sorted(proposals,key=lambda x:-x[0]):
        conflict=None
        for prior in kept:
            p=prior['xy'];lo=max(q[:,0].min(),p[:,0].min());hi=min(q[:,0].max(),p[:,0].max())
            if hi<=lo:continue
            xs=np.arange(math.ceil(lo),math.floor(hi)+1)
            delta=np.interp(xs,q[:,0],q[:,1])-np.interp(xs,p[:,0],p[:,1])
            if len(delta) and (np.min(np.abs(delta))<PARAMETERS['new_fragment_separation_pitch']*pitch or np.min(delta)<0<np.max(delta)):
                conflict=prior['id'];break
        if conflict:
            reject.append({**rec,'reason':'new_fragment_order_or_spacing_conflict','stronger_group_id':conflict});continue
        fresh=base[rr[:,1],rr[:,0]]==0
        if not np.any(fresh):reject.append({**rec,'reason':'no_new_geometry_after_deduplication'});continue
        new[rr[fresh,1],rr[fresh,0]]=255
        inferred=(_field(fields,'broad_score',rr)<PARAMETERS['broad_threshold'])
        source_inferred=detected.get('fit',{}).get('inferred_mask')
        if isinstance(source_inferred,np.ndarray):
            if source_inferred.shape!=base.shape:raise ValueError('source inferred mask must have original shape')
            inferred|=_sample(source_inferred,rr)>0
        new_inferred[rr[fresh&inferred,1],rr[fresh&inferred,0]]=255
        rec.update({'status':'accepted','sampled_points_xy':q.tolist(),'new_pixels':int(np.count_nonzero(fresh)),'source_point_ids':[int(p.get('point_id',-1)) for p in pts],'accepted_point_ids':[]})
        # Keep only actual newly measured points associated with the added part.
        for p in pts:
            pid=str(p.get('id',f"D4_full_new_{int(p.get('point_id',-1)):06d}"))
            if pid in point_ids:continue
            point_ids.add(pid);ap={**p,'id':pid,'source':'D4_full','group_id':rec['group_id']}
            accepted_points.append(ap);rec['accepted_point_ids'].append(pid)
        groups.append(rec);kept.append({'id':rec['group_id'],'xy':q})
    merged=base|new;inferred=np.asarray(data['f2_inferred'],np.uint8)|new_inferred
    assert np.array_equal(merged[t==0],base[t==0])
    diagnostics.update({'input_curves':len(detected.get('fit',{}).get('curves',[])),'accepted_groups':len(groups),'rejected_fragments':len(reject),'new_pixels':int(np.count_nonzero(new)),'accepted_new_points':len(accepted_points),'accepted_modes':dict(Counter(g['mode'] for g in groups)),'rejection_counts':dict(Counter(r['reason'] for r in reject)),'existing_geometry_unchanged':True,'transition_outside_unchanged':True,'note':'Image-supported fragment acceptance does not establish same physical rope identity or accuracy.'})
    return {'merged_mask':merged,'inferred_mask':inferred,'new_mask':new,'removed_mask':removed,'transition_mask':t,'accepted_points':accepted_points,'groups':groups,'rejections':reject,'diagnostics':diagnostics}


