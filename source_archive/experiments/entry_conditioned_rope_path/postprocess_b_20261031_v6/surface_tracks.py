"""H6: reconstruct final F3 visual rows from saved vector provenance.

No F3 detector, labels, model, or device is invoked. C3/D4 accepted fits,
F2 accepted joins and F3 accepted additions supply geometry. F3's full-D4
candidate collection is NOT adopted as final boundaries. Coordinates returned
are half-size H1 crop xy; original_xy=2*xy+[0,1050]. Row IDs are visual
correspondence, not physical rope identities.
"""
from collections import defaultdict,Counter
from pathlib import Path
import json,hashlib,time

import cv2
import numpy as np

from common import ROOT,H2,module

OLD=module('h6_readonly_surface_measurements',H2/'surface_tracks.py')
A=ROOT/'数据资产/05_派生与实验结果'
F3=A/'camera_condition_full_surface_joint_20261028_v3/all_001'
F2=A/'camera_condition_seam_joint_20261028_v2/all_001'
C3=A/'camera_condition_grouped_conic_20261026_v6/all_001'
D4=A/'camera_condition_dark_width_20261027_v4/all_001'
PARAMETERS=dict(original_grid_step_px=2.,output_step_original_px=6.,
    final_raster_tolerance_original_px=2.5,duplicate_distance_original_px=3.,
    same_lineage_conflict_original_px=5.,minimum_pair_gap_original_px=6.,
    maximum_pair_gap_original_px=120.,minimum_track_span_original_px=30.,
    local_spacing_ratio_min=.45,local_spacing_ratio_max=1.8,spacing_neighborhood_pairs=3,
    maximum_source_gap_original_px=320.,maximum_source_gap_pitch=12.,
    gap_direction_change_degrees=20.,minimum_gap_exclusion_fraction=.90,
    no_b_refinement=True,no_source_extrapolation=True,maximum_boundary_rows=512)
COORDINATES='half_size_model_crop_xy; original_xy=2*xy+[0,1050]'


def _json(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _runs(mask):
    d=np.diff(np.r_[False,np.asarray(mask,bool),False].astype(np.int8))
    return list(zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)))
def _sample(a,p):return OLD._sample(a,p)


def load_surface_priors(job):
    priors=OLD.load_surface_priors(job)
    role=priors['camera_role']+'_camera';name=job['id'];files={
        'C3':C3/'boundaries/records'/role/(name+'.json'),
        'D4':D4/'记录/cases'/role/(name+'.json'),
        'F2':F2/'记录/cases'/role/(name+'.json'),
        'F3':F3/'记录/cases'/role/(name+'.json')}
    records={k:_json(p) for k,p in files.items()}
    hashes={str(p):_sha(p) for p in files.values()}
    # F3's own recorded input receipts bind C3/D4/F2 to the actual frozen run.
    dependencies=records['F3']['record']['input_dependency_sha256']
    for key in ('C3','D4','F2'):
        path=files[key]
        expected=dependencies.get(str(path))
        if expected is None:
            expected=next((v for k,v in dependencies.items() if Path(k).resolve()==path.resolve()),None)
        if expected!=hashes[str(path)]:raise ValueError('F3 source receipt mismatch: '+str(path))
    p=F3/'记录/masks/exclusion'/role/(name+'.png')
    mask=cv2.imdecode(np.fromfile(p,np.uint8),cv2.IMREAD_GRAYSCALE)
    if mask is None or tuple(mask.shape)!=tuple(job['shape']):raise ValueError('F3 exclusion shape mismatch')
    h,w=mask.shape
    priors['f3_exclusion_half']=mask.reshape(h//2,2,w//2,2).max((1,3))
    hashes[str(p)]=_sha(p);priors['input_dependency_sha256'].update(hashes)
    priors['f3_vector_records']=records
    priors['f3_vector_source_paths']={k:str(p) for k,p in files.items()}
    priors['surface_policy']='F3_original_C4_surface_for_F3_rows; independent_SAM2_retained_separately'
    return priors


def _saved_pieces(records):
    parent={};pieces=[];links=[]
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    def union(a,b,why):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
        links.append(dict(source_a=a,source_b=b,basis=why))
    def add(cid,xy,priority,source,status='accepted'):
        p=np.asarray(xy,float).reshape(-1,2)
        if len(p)<2 or not np.isfinite(p).all() or np.any(np.diff(p[:,0])<=0):return
        find(cid);pieces.append(dict(id=cid,xy=p,priority=priority,source=source,status=status))
    for kind in ('C3','D4'):
        for c in records[kind].get('fit',{}).get('curves',[]):
            # C3's accepted fit list uses a geometric model status, not the
            # literal word "accepted"; rejected fits live in fit.rejected.
            if c.get('status','accepted') not in ('accepted','completed','single_whole_group_conic'):continue
            add(kind+':'+str(c['curve_id']),c.get('sampled_points_xy',[]),1 if kind=='C3' else 0,kind)
    for j in records['F2'].get('joins',[]):
        if j.get('status')!='accepted':continue
        cid='F2:'+j['candidate_id'];add(cid,j.get('sampled_points_xy',[]),3,'F2')
        for old in j.get('source_curve_ids',[]):union(cid,str(old),'frozen_F2_accepted_source_curve_correspondence')
    for g in records['F3'].get('groups',[]):
        if g.get('status')!='accepted':continue
        cid='F3:'+g['group_id'];add(cid,g.get('sampled_points_xy',[]),4,'F3')
        # Each accepted addition belongs to its saved full-source curve;
        # rejected/unaccepted portions of that curve are never inserted.
        full='F3_accepted_source:'+str(g['source_curve_id'])
        union(cid,full,'same_frozen_F3_source_curve_accepted_parts_only')
        for r in g.get('reference_matches',[]):
            union(cid,str(r['reference_id']),'frozen_F3_accepted_reference_correspondence')
    for p in pieces:p['lineage']=find(p['id'])
    return pieces,links


def _source_piece(p,xs,distance,inferred,exclusion,domain,pitch):
    q=p['xy'];inside=(xs>=q[0,0])&(xs<=q[-1,0]);x=xs[inside]
    if not len(x):return None
    y=np.interp(x,q[:,0],q[:,1]);xy=np.c_[x*.5,y*.5]
    known=_sample(distance,xy)*2<=PARAMETERS['final_raster_tolerance_original_px']
    inf=_sample(inferred,xy)>.1;excluded=_sample(exclusion,xy)>.5
    domain_ok=_sample(domain,xy)>.5
    # Interpolation is confined to this exact saved vector's finite extent.
    # A long missing raster section may be shown only in its known exclusion
    # corridor, with observed endpoints and bounded direction/length.
    state=np.where(known,np.where(inf,2,1),0).astype(np.uint8)
    gap_receipts=[]
    for a,b in _runs(~known):
        if a==0 or b==len(x):continue
        length=float(x[b]-x[a-1]);max_gap=min(PARAMETERS['maximum_source_gap_original_px'],PARAMETERS['maximum_source_gap_pitch']*pitch)
        if length>max_gap or not domain_ok[a-1:b+1].all():continue
        exclusion_fraction=float(excluded[a:b].mean())
        if exclusion_fraction<PARAMETERS['minimum_gap_exclusion_fraction']:continue
        left=max(0,a-5);right=min(len(x)-1,b+4)
        dl=np.array([x[a-1]-x[left],y[a-1]-y[left]]);dr=np.array([x[right]-x[b],y[right]-y[b]])
        if np.linalg.norm(dl)<2 or np.linalg.norm(dr)<2:continue
        angle=float(np.degrees(np.arccos(np.clip(dl@dr/(np.linalg.norm(dl)*np.linalg.norm(dr)),-1,1))))
        if angle>PARAMETERS['gap_direction_change_degrees']:continue
        state[a:b]=3
        gap_receipts.append(dict(source_id=p['id'],start_original_xy=[float(x[a]),float(y[a])],end_original_xy=[float(x[b-1]),float(y[b-1])],
            length_original_px=length,exclusion_fraction=exclusion_fraction,endpoint_direction_change_degrees=angle,
            state='inferred_gap',same_material_identity_confirmed=False))
    state[~domain_ok]=0
    return np.flatnonzero(inside),y,state,gap_receipts


def _decimate(indices,state):
    if not len(indices):return indices
    stride=max(1,int(round(PARAMETERS['output_step_original_px']/PARAMETERS['original_grid_step_px'])))
    picks=set(range(0,len(indices),stride));picks.add(len(indices)-1)
    for a in np.flatnonzero(np.diff(state.astype(int))):picks.add(int(a));picks.add(int(a+1))
    return indices[sorted(picks)]


def _center_track(a,b,indices,xs,rows,gray,heat,serial):
    ya,yb=rows[a]['y'],rows[b]['y'];sa,sb=rows[a]['state'],rows[b]['state']
    status_all=np.maximum(sa[indices],sb[indices]);indices=_decimate(indices,status_all)
    x=xs[indices]*.5;y0=ya[indices]*.5-525.;y1=yb[indices]*.5-525.
    points=np.c_[x,(y0+y1)*.5];upper=np.c_[x,y0];lower=np.c_[x,y1]
    tangent=np.gradient(points,axis=0);tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-8)
    normal=np.c_[-tangent[:,1],tangent[:,0]];width=(y1-y0)*abs(normal[:,1]);width=np.maximum(width,1.)
    raw,hs,hl,rd,hd=OLD._appearance(gray,heat,points,normal,width)
    statuses=np.maximum(sa[indices],sb[indices]);observed=statuses==1;inferred=statuses>=2
    ro=raw>=OLD.PARAMETERS['raw_support_threshold'];ho=(hs>=OLD.PARAMETERS['heat_support_threshold'])&(hl>=OLD.PARAMETERS['heat_level_support_threshold'])
    names={1:'observed',2:'inferred',3:'inferred_gap'};states=[names[int(v)] for v in statuses]
    pair=[rows[a]['id'],rows[b]['id']];tid='lane_'+hashlib.sha256('|'.join(pair).encode()).hexdigest()[:16]+'_'+str(serial)
    return dict(track_id=tid,boundary_ids=pair,points_xy=points.tolist(),midpoint_xy=points.tolist(),
        upper_boundary_xy=upper.tolist(),lower_boundary_xy=lower.tolist(),spacing_px=width.tolist(),vertical_spacing_px=(y1-y0).tolist(),width_px=width.tolist(),
        width_is_calibrated_rope_diameter=False,refine_offset_px=[0.]*len(points),f3_pair_supported=[True]*len(points),
        f3_pair_observed=observed.tolist(),f3_pair_inferred=inferred.tolist(),
        upper_boundary_inferred=(sa[indices]>=2).tolist(),lower_boundary_inferred=(sb[indices]>=2).tolist(),
        point_status=states,segment_status=[names[int(max(statuses[k-1],statuses[k]))] for k in range(1,len(points))],
        raw_score=raw.tolist(),heat_score=hs.tolist(),heat_level=hl.tolist(),raw_signed_center_minus_sides=rd.tolist(),heat_signed_center_minus_sides=hd.tolist(),
        image_supported=ro.tolist(),heat_supported=ho.tolist(),supported=(ro&ho&observed).tolist(),
        score=float(np.mean(.45*raw+.35*hs+.2*hl)),length_px=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),
        support_summary=dict(sample_count=len(points),observed_pair_fraction=float(observed.mean()),inferred_gap_fraction=float((statuses==3).mean()),
            raw_supported_fraction=float(ro.mean()),heat_supported_fraction=float(ho.mean()),joint_supported_fraction=float((ro&ho&observed).mean()),
            spacing_median_half_px=float(np.median(width)),spacing_range_half_px=[float(width.min()),float(width.max())],maximum_refine_offset_half_px=0.),
        source_boundary_lineage_ids=pair,adjacency_checked_at_original_step_px=PARAMETERS['original_grid_step_px'],
        stop_reason='fixed_source_boundary_pair_end_missing_geometry_or_neighbor_order_change; no_neighbor_switch',
        identity='unknown',same_material_identity_confirmed=False,coordinate_system=COORDINATES)


def build_tracks(priors,heat,gray=None):
    tick=time.perf_counter();heat=np.asarray(heat,np.float32)
    if gray is None:gray=cv2.cvtColor(priors['original_half'][525:],cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
    if heat.shape!=gray.shape or not np.isfinite(heat).all():raise ValueError('Finite frozen B crop required')
    merged=np.asarray(priors['f3_merged_half'])>0;inf=cv2.dilate((np.asarray(priors['f3_inferred_half'])>0).astype(np.uint8),np.ones((3,3),np.uint8)).astype(np.float32)
    excluded=(np.asarray(priors['f3_exclusion_half'])>0).astype(np.float32)
    domain=(np.asarray(priors['surface_half'])>0).astype(np.float32)
    distance=cv2.distanceTransform((~merged).astype(np.uint8),cv2.DIST_L2,3)
    pieces,links=_saved_pieces(priors['f3_vector_records']);xs=np.arange(0,priors['original_shape'][1],PARAMETERS['original_grid_step_px'])
    grouped=defaultdict(list)
    for p in pieces:grouped[p['lineage']].append(p)
    if len(grouped)>PARAMETERS['maximum_boundary_rows']:raise ValueError('Saved lineage row budget exceeded')
    rows=[];gap_receipts=[];rejected=Counter()
    for lineage,pp in sorted(grouped.items()):
        y=np.full(len(xs),np.nan);state=np.zeros(len(xs),np.uint8);priority=np.full(len(xs),-1,int)
        conflicts=np.zeros(len(xs),bool);sources=[]
        for p in sorted(pp,key=lambda p:(p['priority'],p['id'])):
            result=_source_piece(p,xs,distance,inf,excluded,domain,priors['pitch_px'])
            if result is None:continue
            ix,py,ps,gaps=result;gap_receipts.extend(gaps);sources.append(p['id'])
            overlap=(state[ix]>0)&(ps>0)
            conflicts[ix[overlap & (abs(y[ix]-py)>PARAMETERS['same_lineage_conflict_original_px'])]]=True
            replace=(ps>0)&((priority[ix]<p['priority'])|(state[ix]==0))
            k=ix[replace];y[k]=py[replace];state[k]=ps[replace];priority[k]=p['priority']
        state[conflicts]=0;rejected['same_lineage_multiple_y']=int(rejected['same_lineage_multiple_y']+conflicts.sum())
        if (state>0).sum()<3:continue
        rows.append(dict(id='f3row_'+hashlib.sha256(lineage.encode()).hexdigest()[:16],lineage=lineage,y=y,state=state,priority=priority,source_ids=sources))
    pairs=defaultdict(list);duplicate_count=0;columns=[];existing_gaps=[]
    for k,x in enumerate(xs):
        present=[i for i,r in enumerate(rows) if r['state'][k]>0 and r['y'][k]>=1050.]
        present.sort(key=lambda i:rows[i]['y'][k]);unique=[]
        for i in present:
            if unique and rows[i]['y'][k]-rows[unique[-1]]['y'][k]<PARAMETERS['duplicate_distance_original_px']:
                # Retain vector provenance rather than fusing neighboring IDs.
                old=unique[-1]
                if rows[i]['priority'][k]>rows[old]['priority'][k]:unique[-1]=i
                duplicate_count+=1;continue
            unique.append(i)
        gaps=[float(rows[b]['y'][k]-rows[a]['y'][k]) for a,b in zip(unique[:-1],unique[1:])]
        columns.append((k,unique,gaps))
        existing_gaps.extend(g for g in gaps if PARAMETERS['minimum_pair_gap_original_px']<=g<=PARAMETERS['maximum_pair_gap_original_px'])
    image_gap_median=float(np.median(existing_gaps)) if existing_gaps else None
    for k,unique,gaps in columns:
        x=xs[k]
        for gi,(a,b) in enumerate(zip(unique[:-1],unique[1:])):
            gap=gaps[gi]
            if not PARAMETERS['minimum_pair_gap_original_px']<=gap<=PARAMETERS['maximum_pair_gap_original_px']:rejected['spacing_pixel_sanity']+=1;continue
            n=PARAMETERS['spacing_neighborhood_pairs']
            local=[v for v in gaps[max(0,gi-n):gi+n+1] if v>=PARAMETERS['minimum_pair_gap_original_px']]
            median=float(np.median(local)) if len(local)>=3 else image_gap_median
            if median is None or not PARAMETERS['local_spacing_ratio_min']<=gap/median<=PARAMETERS['local_spacing_ratio_max']:
                rejected['local_spacing_ambiguity']+=1;continue
            # Actual adjacency at every original 2px column. The pair IDs are
            # immutable; a third row appearing inside stops this pair.
            yy=np.linspace(rows[a]['y'][k],rows[b]['y'][k],5)
            if not (_sample(domain,np.c_[np.full(5,x*.5),yy*.5])>.5).all():rejected['outside_F3_C4_surface']+=1;continue
            pairs[(a,b)].append(k)
    tracks=[]
    for (a,b),ix in sorted(pairs.items()):
        ix=np.asarray(ix,int);breaks=np.flatnonzero(np.diff(ix)>1)+1
        for serial,run in enumerate(np.split(ix,breaks)):
            if len(run)<3 or xs[run[-1]]-xs[run[0]]<PARAMETERS['minimum_track_span_original_px']:rejected['short_fixed_pair']+=1;continue
            tracks.append(_center_track(a,b,run,xs,rows,np.asarray(gray,np.float32),heat,serial))
    boundaries=[]
    for row in rows:
        pieces_out=[]
        for a,b in _runs(row['state']>0):
            ix=_decimate(np.arange(a,b),row['state'][a:b]);p=np.c_[xs[ix]*.5,row['y'][ix]*.5-525.]
            states=[{1:'observed',2:'inferred',3:'inferred_gap'}[int(v)] for v in row['state'][ix]]
            pieces_out.append(dict(points_xy=p.tolist(),point_status=states,segment_status=[states[k] if states[k]==states[k-1] else 'inferred_gap' if 'inferred_gap' in (states[k-1],states[k]) else 'inferred' for k in range(1,len(states))]))
        boundaries.append(dict(boundary_id=row['id'],source_lineage_id=row['lineage'],source_ids=row['source_ids'],pieces=pieces_out,
            points_xy=[p for q in pieces_out for p in q['points_xy']],
            point_status=[p for q in pieces_out for p in q['point_status']],
            gap_between_pieces_is_not_a_segment=True,id_definition='frozen_F3_visual_source_correspondence_not_physical_rope'))
    diagnostics=dict(parameters=PARAMETERS,source_piece_count=len(pieces),boundary_lineage_count=len(rows),
        track_fragment_count=len(tracks),paired_sample_count=sum(len(t['points_xy']) for t in tracks),
        total_track_length_half_px=sum(t['length_px'] for t in tracks),rejected=dict(rejected),duplicate_row_samples=duplicate_count,
        saved_source_gap_count=len(gap_receipts),saved_source_gap_receipts=gap_receipts,source_correspondences=links,
        observed_boundary_gap_median_original_px=image_gap_median,spacing_used_only_to_reject_not_fill=True,
        source_vector_paths=priors['f3_vector_source_paths'],surface_policy='F3_original_C4_surface; SAM2_retained_separately',
        candidate_full_D4_curves_adopted=False,center_refinement_uses_B=False,fixed_spacing_lines_generated=0,extrapolated_points=0,
        sparse_raster_CC_used_as_identity=False,attached_center_is_exact_saved_boundary_midpoint=True,
        labels_read=False,same_material_identity_confirmed=False,elapsed_seconds=time.perf_counter()-tick)
    return dict(tracks=tracks,boundaries=boundaries,coordinate_system=COORDINATES,diagnostics=diagnostics,
        topology=dict(boundary_key='boundaries[].pieces[]; do_not_connect_separate_pieces',center_key='tracks[]',
            center_boundary_ids_immutable=True,adjacency_checked_at_original_step_px=PARAMETERS['original_grid_step_px'],
            boundary_and_center_points_coordinate_system=COORDINATES))


def draw_tracks(crop_bgr,result,thickness=1):return OLD.draw_tracks(crop_bgr,result,thickness)
