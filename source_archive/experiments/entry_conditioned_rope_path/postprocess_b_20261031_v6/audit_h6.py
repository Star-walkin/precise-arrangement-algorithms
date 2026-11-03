"""Independent saved-artifact audit. No detector/model/label/hardware imports.

Only the prior independent H5 arithmetic checker is reused. Intersections,
projection permission, trimming and vector-source checks are independently
computed below, without importing H6 production modules.
"""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import argparse,gzip,hashlib,importlib.util,json,sys,traceback
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
H5=HERE.parent/'postprocess_b_20261031_v5'
spec=importlib.util.spec_from_file_location('prior_independent_h5_audit',H5/'audit_h5.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
load=prior.load;sha=prior.sha
EPS=1.e-6

def pts(value):return np.asarray(value,float).reshape(-1,2)
def arclength(p):return np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]

def events(path,tracks):
    """Piecewise-linear event reconstruction at every knot of both curves."""
    p=pts(path)
    if len(p)<2:return []
    s=arclength(p);dx=np.diff(p[:,0]);direction=dx[abs(dx)>EPS]
    if len(direction)==0 or (np.any(direction<0) and np.any(direction>0)):
        return [dict(kind='nonmonotone_path_unknown',track_id=None,progress_half_px=0.,xy=p[0].tolist())]
    if direction[0]<0:p=p[::-1];s=s[::-1]
    keep=np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>EPS];p=p[keep];s=s[keep]
    if np.any(np.diff(p[:,0])<=EPS):
        return [dict(kind='vertical_path_unknown',track_id=None,progress_half_px=0.,xy=pts(path)[0].tolist())]
    answer=[]
    for tr in tracks:
        q=pts(tr['points_xy'])
        if len(q)<2 or p[:,1].min()>q[:,1].max()+EPS or p[:,1].max()<q[:,1].min()-EPS:continue
        if q[0,0]>q[-1,0]:q=q[::-1]
        if np.any(np.diff(q[:,0])<=EPS):
            answer.append(dict(kind='nonmonotone_track_unknown',track_id=tr['track_id'],progress_half_px=0.,xy=pts(path)[0].tolist()));continue
        lo=max(p[0,0],q[0,0]);hi=min(p[-1,0],q[-1,0])
        if lo>hi:continue
        knots=np.unique(np.r_[lo,hi,p[(p[:,0]>lo)&(p[:,0]<hi),0],q[(q[:,0]>lo)&(q[:,0]<hi),0]])
        diff=np.interp(knots,p[:,0],p[:,1])-np.interp(knots,q[:,0],q[:,1]);zero=abs(diff)<=EPS
        local=[];i=0
        while i<len(knots):
            if zero[i]:
                j=i
                while j+1<len(knots) and zero[j+1]:j+=1
                local.append((knots[i],knots[j],'overlap' if knots[j]-knots[i]>EPS else 'touch'));i=j+1
            else:i+=1
        for i in range(len(knots)-1):
            if not(zero[i] or zero[i+1]) and diff[i]*diff[i+1]<0:
                x=knots[i]+(knots[i+1]-knots[i])*(-diff[i])/(diff[i+1]-diff[i]);local.append((x,x,'cross'))
        for xa,xb,kind in local:
            a=float(np.interp(xa,p[:,0],s));b=float(np.interp(xb,p[:,0],s));x=xa if a<=b else xb
            answer.append(dict(kind=kind,track_id=tr['track_id'],progress_half_px=min(a,b),end_progress_half_px=max(a,b),xy=[float(x),float(np.interp(x,q[:,0],q[:,1]))]))
    return sorted(answer,key=lambda x:(x['progress_half_px'],str(x['track_id']),x['kind']))

def license_at(f,event,params):
    p=pts(f['points_xy']);s=2*arclength(p);t=2*event['progress_half_px']
    strong=(np.asarray(f['supported'],bool)|np.asarray(f.get('raw_a7_supported',[False]*len(p)),bool))&(np.asarray(f['point_status'])!='entry_condition')
    k=int(np.clip(np.searchsorted(s,t,side='right')-1,0,len(p)-2));span=0.;supported=bool(strong[k] and strong[k+1])
    if supported:
        a=k;b=k+1
        while a>0 and strong[a-1]:a-=1
        while b<len(p)-1 and strong[b+1]:b+=1
        window=params['free_crossing_window_original_px']/2
        span=max(0.,min(s[b],t+window)-max(s[a],t-window))
    return supported and span>=params['free_crossing_continuous_original_px']-1.e-6 and event['kind'] in ('cross','touch'),supported,span

def verify_projection(entry,tracks,params):
    full=entry['full_free_observation'];task=entry['free'];receipt=entry['free_projection'];p=pts(full['points_xy'])
    if len(p)<2:
        assert task==full and not receipt['events'];return dict(trimmed=False,retained_events=0,full_events=0,unresolved_tail_original_px=0.)
    actual=events(p,tracks);saved=receipt['events'];assert len(actual)==len(saved),('projection_event_count',len(actual),len(saved))
    bad=[]
    for a,b in zip(actual,saved):
        assert a['track_id']==b['track_id'] and a['kind']==b['kind']
        assert np.allclose(a['xy'],b['xy'],atol=1.e-5) and np.isclose(a['progress_half_px'],b['progress_half_px'],atol=1.e-5)
        okay,at,span=license_at(full,a,params)
        assert okay==b['permitted_projected_free_crossing'] and at==b['original_image_supported_interval_contains_crossing']
        assert np.isclose(span,b['continuous_support_original_px'],atol=1.e-5)
        assert not b['physical_free_state_or_contact_verified']
        if not okay:bad.append(a)
    s=arclength(p);limit=s[-1] if not bad else max(0.,bad[0]['progress_half_px']-params['stop_before_unlicensed_crossing_widths']*max(3.,full['width_half_px']))
    last=int(np.clip(np.searchsorted(s,limit,side='right')-1,0,len(p)-1));n=last+1
    assert receipt['task_prefix_last_index']==last and receipt['trimmed']==(n<len(p))
    assert np.array_equal(pts(task['points_xy']),p[:n])
    for key in ('supported','protect_support','point_status','raw_a7_supported','observed_straight_support','B_conflict'):
        assert task[key]==full[key][:n],('trimmed_array',key)
    assert task['segment_status']==full['segment_status'][:max(0,n-1)]
    for key,value in full.get('evidence',{}).items():assert task['evidence'][key]==value[:n]
    if receipt['trimmed'] and 'scalar_length_scope' in task:
        post=np.asarray(task['point_status'])!='entry_condition';ds=2*np.diff(s[:n]);valid=post[:-1]&post[1:]
        def measure(q):
            q=np.asarray(q,bool);return float(ds[valid&q[:-1]&q[1:]].sum())
        heat=np.asarray(task['protect_support'],bool);raw=np.asarray(task['raw_a7_supported'],bool)
        hl=measure(heat);jl=measure(task['supported']);ol=measure(heat|raw)
        expected=dict(visible_straight_length_original_px=hl,joint_straight_length_original_px=jl,heat_only_straight_length_original_px=hl-jl,observed_straight_length_original_px=ol,raw_a7_straight_length_original_px=measure(raw),raw_a7_only_length_original_px=measure(raw&~heat),observed_added_length_original_px=ol-hl)
        for key,value in expected.items():assert np.isclose(task[key],value,atol=1.e-6),('task_scalar',key)
        assert task['diagnostics']['scope']=='full_free_observation_detector_diagnostics_not_task_prefix'
        for field in ('visible_runs','gap_hypotheses'):
            for fragment in task[field]:
                a=fragment['start_index'];b=fragment['end_index'];assert 0<=a<=b<n
                assert np.isclose(fragment['span_original_px'],2*(s[b]-s[a]),atol=1.e-6)
                source=full[field][fragment['source_full_observation_index']]
                assert fragment['source_full_observation_field']==field and source['start_index']==a and min(source['end_index'],last)==b
                if field=='gap_hypotheses':assert fragment['length_original_px']==fragment['span_original_px'] and not fragment['supported']
    assert np.isclose(receipt['unresolved_tail_original_px'],2*(s[-1]-s[last]),atol=1.e-5)
    assert not receipt['physical_contact_inferred_from_mask']
    assert all(e['permitted_projected_free_crossing'] for e in receipt['retained_prefix_events'])
    return dict(trimmed=receipt['trimmed'],retained_events=len(receipt['retained_prefix_events']),full_events=len(actual),unresolved_tail_original_px=receipt['unresolved_tail_original_px'])

def verify_attachment(candidate,trackmap):
    p=pts(candidate['points_xy']);stage=np.asarray(candidate['stage']);free=np.flatnonzero(stage=='free');transition=np.flatnonzero(stage=='transition');surface=np.flatnonzero(stage=='surface')
    assert len(free) and len(transition) and len(surface)
    # A contiguous geometrical suffix is reconstructed; production status flags
    # never decide which part of this suffix is exempt from intersection checks.
    assert np.array_equal(free,np.arange(len(free))) and np.array_equal(transition,np.arange(len(free),len(free)+len(transition)))
    bridge=p[len(free)-1:len(free)+len(transition)];tail=p[len(free)+len(transition)-1:]
    assert np.allclose(bridge[0],candidate['join_start_xy']) and np.allclose(bridge[-1],candidate['join_end_xy'])
    target=trackmap[candidate['track_id']];source=pts(target['points_xy']);ix=np.asarray(candidate['surface_source_indices'],int)
    assert np.array_equal(p[surface],source[ix]) and len(ix)>0
    assert np.all(np.diff(ix)==1) or np.all(np.diff(ix)==-1)
    step=1 if (ix[-1]>ix[0] or len(ix)==1 and source[-1,0]>bridge[-1,0]) else -1
    landing=ix[0]-step;assert 0<=landing<len(source) and np.array_equal(bridge[-1],source[landing])
    other=events(np.vstack([bridge,tail[1:]]),[v for k,v in trackmap.items() if k!=candidate['track_id']]);assert not other,('forbidden_other_track',other[:3])
    own=events(bridge,[target]);assert own,('missing_target_capture',candidate['track_id'])
    total=arclength(bridge)[-1];t=source if source[0,0]<source[-1,0] else source[::-1]
    b=bridge if bridge[0,0]<bridge[-1,0] else bridge[::-1]
    for e in own:
        if e['progress_half_px']>=total-EPS*8:continue
        x=e['xy'][0];end=bridge[-1,0];lo,hi=sorted((x,end))
        assert lo>=t[0,0]-EPS and hi<=t[-1,0]+EPS
        knots=np.unique(np.r_[lo,hi,b[(b[:,0]>lo)&(b[:,0]<hi),0],t[(t[:,0]>lo)&(t[:,0]<hi),0]])
        deviation=abs(np.interp(knots,b[:,0],b[:,1])-np.interp(knots,t[:,0],t[:,1]))
        assert np.all(deviation<=EPS),('premature_target_leave_and_return',e['xy'],float(deviation.max()))
    assert candidate['topology']['allowed'] and not candidate['topology']['stage_labels_used_to_skip_geometry']
    assert not candidate['topology']['other_track_events'] and not candidate['topology']['premature_target_events']
    return len(surface)

def source_vectors(records):
    pieces={};parent={}
    def find(a):
        parent.setdefault(a,a)
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    def join(a,b):
        a,b=find(a),find(b);parent[max(a,b)]=min(a,b)
    def add(cid,xy):
        p=pts(xy)
        if len(p)<2 or not np.isfinite(p).all() or np.any(np.diff(p[:,0])<=0):return
        find(cid);pieces[cid]=p
    for kind in ('C3','D4'):
        for c in records[kind].get('fit',{}).get('curves',[]):
            if c.get('status','accepted') in ('accepted','completed','single_whole_group_conic'):add(kind+':'+str(c['curve_id']),c.get('sampled_points_xy',[]))
    for c in records['F2'].get('joins',[]):
        if c.get('status')!='accepted':continue
        cid='F2:'+c['candidate_id'];add(cid,c.get('sampled_points_xy',[]))
        for q in c.get('source_curve_ids',[]):join(cid,str(q))
    for c in records['F3'].get('groups',[]):
        if c.get('status')!='accepted':continue
        cid='F3:'+c['group_id'];add(cid,c.get('sampled_points_xy',[]));join(cid,'F3_accepted_source:'+str(c['source_curve_id']))
        for q in c.get('reference_matches',[]):join(cid,str(q['reference_id']))
    return pieces,{k:find(k) for k in pieces}

def verify_tracks(record,params,hashes):
    result=record['surface_tracks'];paths=result['diagnostics']['source_vector_paths'];records={}
    for kind,path in paths.items():
        if path not in hashes:hashes[path]=sha(path)
        assert hashes[path]==record['input_dependency_sha256'][path];records[kind]=load(path)
    receipt=records['F3']['record']['input_dependency_sha256']
    for k in ('C3','D4','F2'):
        expected=next(v for path,v in receipt.items() if Path(path).resolve()==Path(paths[k]).resolve());assert expected==hashes[paths[k]]
    pieces,lineages=source_vectors(records);boundaries={b['boundary_id']:b for b in result['boundaries']}
    for bid,b in boundaries.items():
        lineage=b['source_lineage_id'];assert bid=='f3row_'+hashlib.sha256(lineage.encode()).hexdigest()[:16]
        assert b['gap_between_pieces_is_not_a_segment']
        for source in b['source_ids']:assert source in pieces and lineages[source]==lineage,('unaccepted_or_false_lineage',source)
        for fragment in b['pieces']:
            p=pts(fragment['points_xy'])*2+[0,1050];assert np.isfinite(p).all() and np.all(np.diff(p[:,0])>0)
            matches=np.zeros(len(p),bool)
            for source in b['source_ids']:
                q=pieces[source];within=(p[:,0]>=q[0,0])&(p[:,0]<=q[-1,0]);matches|=within&np.isclose(p[:,1],np.interp(p[:,0],q[:,0],q[:,1]),atol=1.e-7,rtol=0)
            assert matches.all(),('boundary_without_finite_accepted_vector',bid)
            assert len(fragment['point_status'])==len(p) and len(fragment['segment_status'])==len(p)-1
    trackmap={t['track_id']:t for t in result['tracks']};assert len(trackmap)==len(result['tracks'])
    for tid,t in trackmap.items():
        p=prior.aligned_points(t['points_xy']);up=pts(t['upper_boundary_xy']);down=pts(t['lower_boundary_xy']);n=len(p)
        assert np.array_equal(p,(up+down)*.5) and np.all(down[:,1]>up[:,1])
        assert np.all(np.diff(p[:,0])>0) and np.max(np.diff(p[:,0]))<=params['output_step_original_px']*.5+1.e-6
        pair=t['boundary_ids'];assert pair==t['source_boundary_lineage_ids'] and all(b in boundaries for b in pair)
        assert tid.startswith('lane_'+hashlib.sha256('|'.join(pair).encode()).hexdigest()[:16]+'_')
        for q,bid in zip((up,down),pair):
            covered=np.zeros(n,bool)
            for fragment in boundaries[bid]['pieces']:
                bp=pts(fragment['points_xy']);inside=(q[:,0]>=bp[0,0])&(q[:,0]<=bp[-1,0]);covered|=inside&np.isclose(q[:,1],np.interp(q[:,0],bp[:,0],bp[:,1]),atol=1.e-5,rtol=0)
            # Independent decimation can introduce subpixel interpolation error;
            # fall back to exact source vectors at the queried coordinate.
            if not covered.all():
                original=q*2+[0,1050]
                for source in boundaries[bid]['source_ids']:
                    bp=pieces[source];inside=(original[:,0]>=bp[0,0])&(original[:,0]<=bp[-1,0]);covered|=inside&np.isclose(original[:,1],np.interp(original[:,0],bp[:,0],bp[:,1]),atol=1.e-7,rtol=0)
            assert covered.all(),('center_boundary_not_source',tid,bid)
        assert not np.asarray(t['refine_offset_px']).any()
        state=np.asarray(t['point_status']);observed=state=='observed'
        assert np.array_equal(t['f3_pair_observed'],observed) and np.array_equal(t['f3_pair_inferred'],~observed)
        assert not (np.asarray(t['supported'])&~observed).any()
    d=result['diagnostics'];assert not d['center_refinement_uses_B'] and not d['candidate_full_D4_curves_adopted']
    assert not d['sparse_raster_CC_used_as_identity'] and d['extrapolated_points']==0 and d['fixed_spacing_lines_generated']==0
    return trackmap

def audit(run_name,out):
    run=HERE/run_name;start=load(run/'run_start.json');summary=load(run/'summary.json');config=load(HERE/'inputs.json')
    jobs={j['id']:j for j in config['jobs']};errors=[];rows=[];counts=Counter();hashes={};surface_points=0
    try:
        assert len(jobs)==292 and sum(len(j['seeds']) for j in jobs.values())==301
        assert summary['exit_code']==0 and summary['completed']==summary['planned']==len(start['input_ids'])
        assert summary['source_unchanged'] and sha(HERE/'inputs.json')==start['input_protocol_sha256']
        for path,digest in (start['sources']|start['reused_sources']).items():
            checked=run/'source_snapshot'/Path(path).name if path in start['sources'] else Path(path)
            assert sha(checked)==digest,('source_hash',path)
    except Exception as e:errors.append(dict(scope='run',error=str(e),traceback=traceback.format_exc()))
    for sid in start['input_ids']:
        try:
            r=load(run/'记录'/(sid+'.json.gz'));old=load(H5/'all_001/记录'/(sid+'.json.gz'));job=jobs[sid]
            tracks=verify_tracks(r,start['surface_parameters'],hashes);counts['center_tracks']+=len(tracks)
            assert r['source_image_sha256']==old['source_image_sha256']==job['image_sha256'];assert len(r['entries'])==len(old['entries'])==len(job['seeds'])
            local=[]
            for ei,(e,before) in enumerate(zip(r['entries'],old['entries'])):
                counts['entry_conditions']+=1;assert e['seed']==before['seed']==job['seeds'][ei]
                rp=e['response_file'];assert rp==before['response_file']==job['response_files'][ei]
                if rp not in hashes:hashes[rp]=sha(rp)
                assert hashes[rp]==e['response_sha256']==before['response_sha256'];counts['response_conditions_verified']+=1
                full=e['full_free_observation'];f=e['free']
                for key in ('points_xy','supported','protect_support','point_status','segment_status','raw_a7_supported','observed_straight_support','B_conflict'):
                    assert full[key]==before['free'][key],('full_H5_observation_changed',key)
                if full['points_xy']:assert full['evidence']==before['free']['evidence']
                projection=verify_projection(e,list(tracks.values()),start['topology_parameters']);counts['trimmed_conditions']+=projection['trimmed'];counts['licensed_projection_events_retained']+=projection['retained_events']
                cs=e['connection']['candidates'];assert len(cs)<=3
                if len(f['points_xy'])<3:assert not cs
                for c in cs:
                    counts['candidates']+=1;cp=prior.aligned_points(c['points_xy']);m=len(cp)
                    assert all(len(c[k])==m for k in ('supported','point_status','stage','width_half_px')) and not c['identity_verified']
                    prior.validate_score(c,e['connection']['parameters'],f)
                    stage=np.asarray(c['stage']);nf=int((stage=='free').sum());assert np.array_equal(cp[:nf],pts(f['points_xy'])[:nf])
                    assert c['point_status'][:nf]==f['point_status'][:nf] and c['supported'][:nf]==f['supported'][:nf]
                    if c['kind']=='free_fragment_no_resolved_connection':
                        assert np.array_equal(cp,pts(f['points_xy']))
                        if projection['trimmed']:assert c['termination']=='unresolved_surface_projection'
                    else:
                        assert c['kind']=='straight_free_to_F3_center_track';counts['surface_candidates']+=1;surface_points+=verify_attachment(c,tracks)
                        assert np.array_equal(e['candidate_track_geometry'][c['track_id']],tracks[c['track_id']]['points_xy'])
                local.append(dict(entry=ei,**projection,candidate_count=len(cs)))
            counts['task_direction_excluded_conditions']+=prior.validate_direction(r,start['parameters'])
            si=r['selected_entry'];ci=r['selected_candidate'];kind=None if ci is None else r['entries'][si]['connection']['candidates'][ci]['kind']
            if ci is not None:assert r['entries'][si]['task_direction']['selection_allowed']
            counts['selected_'+str(kind)]+=1
            with Image.open(run/'四栏检测结果'/(sid+'.jpg')) as im:assert im.size==(8192,1655);im.verify()
            rows.append(dict(id=sid,selected_entry=si,selected_kind=kind,entries=local))
        except Exception as e:errors.append(dict(id=sid,error=str(e),traceback=traceback.format_exc()))
    report=dict(status='passed' if not errors else 'failed',exit_code=int(bool(errors)),created_utc=datetime.now(timezone.utc).isoformat(),command=sys.argv,run=str(run),audit_sha256=sha(__file__),reused_independent_h5_audit_sha256=sha(H5/'audit_h5.py'),checked_images=len(rows),planned_images=len(start['input_ids']),counts=dict(counts),surface_points_exact_source_indices=surface_points,errors=errors,records=rows,
        scope='Saved predictions only. Frozen inputs; accepted F3 vector lineage and finite midpoint source; complete geometric crossing and support-based task trimming. No labels or devices.',
        limits='This is an artifact/topology audit, not target identity accuracy. Source visual correspondences may be wrong. Gray inferred gaps remain inferred. Full292 audit does not imply full292 visual identity review.')
    with (run/out).open('x',encoding='utf8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k not in ('errors','records')},ensure_ascii=False))
    if errors:print(json.dumps(errors,ensure_ascii=False));raise SystemExit(1)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf8');ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--out',default='independent_geometry_audit.json');args=ap.parse_args();audit(args.run,args.out)
