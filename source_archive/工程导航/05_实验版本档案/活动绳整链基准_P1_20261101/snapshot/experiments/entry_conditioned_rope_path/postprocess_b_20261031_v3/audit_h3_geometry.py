"""Independent H3 geometry/product audit; no model, detector, label or device imports."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import argparse,gzip,json,hashlib,sys,time
import cv2,numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;H2=HERE.parent/'postprocess_b_20261030_v2'
def load(p):
    with (gzip.open if Path(p).suffix=='.gz' else open)(p,'rt',encoding='utf-8-sig') as f:return json.load(f)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def arc(p):return np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
def chosen(record,entry=None):
    entry=record['selected_entry'] if entry is None else entry
    cs=record['entries'][entry]['connection'].get('candidates',[])
    return cs[0] if cs else None
def tail_from_seed(c,seed):
    if c is None:return np.empty((0,2))
    p=np.asarray(c['points_xy'],float)*2+[0,1050]
    j=int(np.argmin(np.linalg.norm(p-np.asarray(seed['xy'][-1]),axis=1)))
    return p[j:]
def shape_metrics(p,horizon=None):
    if len(p)<3:return None
    s=arc(p);length=float(s[-1]);limit=min(length,horizon) if horizon is not None else length
    if limit<48:return {'length_original_px':length,'measured_length_px':limit,'status':'too_short_for48px_direction_chord'}
    t=np.linspace(0,limit,max(3,int(limit/8)+1));q=np.stack([np.interp(t,s,p[:,j]) for j in range(2)],1)
    lo=np.maximum(0,t-24);hi=np.minimum(limit,t+24)
    a=np.stack([np.interp(lo,s,p[:,j]) for j in range(2)],1);b=np.stack([np.interp(hi,s,p[:,j]) for j in range(2)],1)
    angles=np.unwrap(np.arctan2((b-a)[:,1],(b-a)[:,0]));turn=np.abs(np.diff(angles));ds=np.diff(t)
    return {'status':'measured','length_original_px':length,'measured_length_px':limit,
        'tangent_amplitude_degrees':float(np.degrees(np.ptp(angles))),
        'total_tangent_variation_degrees':float(np.degrees(turn.sum())),
        'p95_tangent_increment_degrees_at8px':float(np.degrees(np.quantile(turn,.95))),
        'rms_curvature_radians_per_original_px':float(np.sqrt(np.sum(turn**2/ds)/np.sum(ds))),
        'measurement_only_chord_original_px':48.,'output_coordinates_changed':False}
def nearest(points,tracks):
    segments=np.array([[a,b] for tr in tracks for a,b in zip(tr['points_xy'][:-1],tr['points_xy'][1:])],float).reshape(-1,2,2)
    segments=segments[np.linalg.norm(segments[:,1]-segments[:,0],axis=1)>1e-8] if len(segments) else segments
    result=np.full(len(points),np.inf)
    for lo in range(0,len(segments),256):
        a,b=segments[lo:lo+256,0],segments[lo:lo+256,1];d=b-a;r=points[:,None,:]-a[None,:,:]
        t=np.clip((r*d[None,:,:]).sum(2)/(d*d).sum(1)[None,:],0,1)
        result=np.minimum(result,np.sqrt(((r-t[:,:,None]*d[None,:,:])**2).sum(2)).min(1))
    return result

def baseline():
    summary=load(H2/'all_001/summary.json');rows=[];tick=time.perf_counter()
    for row in summary['rows']:
        if row['kind']!='direct_H1_response_path':continue
        r=load(H2/'all_001/记录'/(row['id']+'.json.gz'));c=chosen(r);p=np.asarray(c['points_xy'],float)
        tail=np.asarray(c['stage'])=='response_continuation'
        maskfile=next(p for p in r['input_dependency_sha256'] if 'camera_condition_sam2_' in p and '\\masks\\' in p)
        mask=cv2.imdecode(np.fromfile(maskfile,np.uint8),cv2.IMREAD_GRAYSCALE)
        orig=np.rint(p*2+[0,1050]).astype(int);valid=(orig[:,0]>=0)&(orig[:,0]<4096)&(orig[:,1]>=0)&(orig[:,1]<3000)
        inside=np.zeros(len(p),bool);inside[valid]=mask[orig[valid,1],orig[valid,0]]>0;keep=tail&inside
        d=nearest(p[keep],r['surface_tracks']['tracks'])*2 if keep.any() else np.zeros(0)
        rows.append({'id':row['id'],'entry':r['selected_entry'],'sampled_continuation_points_inside_SAM2':int(keep.sum()),
            'nearest_any_F3_center_mean_original_px':float(d.mean()) if len(d) and np.isfinite(d).all() else None,
            'nearest_any_F3_center_p95_original_px':float(np.quantile(d,.95)) if len(d) and np.isfinite(d).all() else None,
            'fraction_within5_original_px':float(np.mean(d<=5)) if len(d) else None,
            'fraction_exceeding10_original_px':float(np.mean(d>10)) if len(d) else None,
            'candidate_is_exact_F3_track':False,'path_record_sha256':sha(H2/'all_001/记录'/(row['id']+'.json.gz'))})
    payload={'status':'completed','source':'H2 all001 frozen selected paths','source_images':292,'direct_H1_images':118,
        'metric':'point-sampled saved response_continuation inside independent original SAM2 ROI; exact nearest line-segment distance to ANY saved F3 center; original px. Not target accuracy or fixed-track identity.',
        'direct_cases_with_reference_points':sum(x['sampled_continuation_points_inside_SAM2']>0 for x in rows),
        'cases_p95_gt5px':sum((x['nearest_any_F3_center_p95_original_px'] or 0)>5 for x in rows),
        'cases_p95_gt10px':sum((x['nearest_any_F3_center_p95_original_px'] or 0)>10 for x in rows),
        'cases_more_than_half_gt5px':sum(x['fraction_within5_original_px'] is not None and x['fraction_within5_original_px']<.5 for x in rows),
        'rows':rows,'seconds':time.perf_counter()-tick,'labels_read':False}
    save(HERE/'h2_direct_lane_alignment_audit.json',payload);print(json.dumps({k:v for k,v in payload.items() if k!='rows'}))

def audit(name,outname):
    run=HERE/name;summary=load(run/'summary.json');start=load(run/'run_start.json');inputs=load(H2/'inputs.json')
    jobs={j['id']:j for j in inputs['jobs']};errors=[];records=[];checked_hashes={};candidate_count=surface_points=0
    sources={**start['sources'],**start['reused_sources']}
    for p,digest in sources.items():
        if sha(p)!=digest:errors.append({'code':'source_changed','path':p})
    if not summary['source_unchanged']:errors.append({'code':'runner_source_changed'})
    assert len(jobs)==292 and sum(len(j['seeds']) for j in jobs.values())==301
    assert summary['completed']==summary['planned']==len(start['input_ids'])
    for sid in start['input_ids']:
        r=load(run/'记录'/(sid+'.json.gz'));old=load(H2/'all_001/记录'/(sid+'.json.gz'));job=jobs[sid]
        assert len(r['entries'])==len(old['entries'])==len(job['seeds'])
        comparisons=[];npoints=0;max_residual=0.;surface_candidates=0
        for ei,(entry,before) in enumerate(zip(r['entries'],old['entries'])):
            assert entry['seed']==before['seed']==job['seeds'][ei]
            assert entry['response_file']==before['response_file']==job['response_files'][ei]
            rp=entry['response_file']
            if rp not in checked_hashes:checked_hashes[rp]=sha(rp)
            assert checked_hashes[rp]==entry['response_sha256']==before['response_sha256']
            cs=entry['connection'].get('candidates',[]);assert len(cs)<=3
            for c in cs:
                candidate_count+=1;p=np.asarray(c['points_xy']);n=len(p)
                assert p.shape==(n,2) and n>=2 and np.isfinite(p).all()
                assert (p[:,0]>=0).all() and (p[:,0]<2048).all() and (p[:,1]>=0).all() and (p[:,1]<975).all()
                assert all(len(c[k])==n for k in ['width_half_px','supported','stage','point_status'])
                assert c['kind']!='direct_H1_response_path'
                free=p[np.array(c['stage'])=='free']
                if len(free)>=3:
                    center=free.mean(0);_,_,vh=np.linalg.svd(free-center,full_matrices=False)
                    residual=float(np.max(abs((free-center)@vh[-1])));max_residual=max(max_residual,residual)
                    assert residual<1.e-7,('free_not_straight',sid,ei,residual)
                surf=p[np.array(c['stage'])=='surface']
                if len(surf):
                    surface_candidates+=1;npoints+=len(surf);surface_points+=len(surf)
                    tr=entry['candidate_track_geometry'][str(c['track_id'])]
                    track_points=tr['points_xy'] if isinstance(tr,dict) else tr
                    indices=np.asarray(c['surface_source_indices'],int)
                    assert len(indices)==len(surf) and (indices>=0).all() and (indices<len(track_points)).all()
                    assert np.array_equal(surf,np.asarray(track_points)[indices]),('surface_not_exact_track',sid,ei)
                    assert np.all(np.diff(indices)==1) or np.all(np.diff(indices)==-1)
                    if ei==old['selected_entry']:
                        oldtrack=next(t for t in old['surface_tracks']['tracks'] if t['track_id']==c['track_id'])
                        assert track_points==oldtrack['points_xy'],('saved_F3_track_changed',sid,ei)
                    dm=c['metrics']['direction'];assert 0<dm['lookahead_coverage']<=1
            a=chosen(old,ei);b=cs[0] if cs else None
            pa=tail_from_seed(a,entry['seed']);pb=tail_from_seed(b,entry['seed'])
            la=float(arc(pa)[-1]) if len(pa)>1 else 0.;lb=float(arc(pb)[-1]) if len(pb)>1 else 0.;common=min(la,lb)
            comparisons.append({'entry':ei,'H2_kind':a['kind'] if a else None,'H3_kind':b['kind'] if b else None,
                'H2_post_seed_length_original_px':la,'H3_post_seed_length_original_px':lb,
                'length_ratio_H3_over_H2':lb/la if la else None,'common_post_seed_arclength_px':common,
                'H2_common_geometry':shape_metrics(pa,common),'H3_common_geometry':shape_metrics(pb,common),
                'interpretation':'same input entry and matched prefix arclength from nearest seed-tip point; shortened horizon separately disclosed. Geometry not physical identity or a success score.'})
        img=run/'四栏检测结果'/(sid+'.jpg')
        with Image.open(img) as im:assert im.size==(8192,1655);im.verify()
        records.append({'id':sid,'entry_count':len(r['entries']),'surface_candidate_count':surface_candidates,
            'surface_points_exact_source_indices':npoints,'maximum_free_line_residual_half_px':max_residual,
            'selected_entry_H2':old['selected_entry'],'selected_entry_H3':r['selected_entry'],
            'selected_entry_changed':old['selected_entry']!=r['selected_entry'],'geometry_comparisons':comparisons})
    payload={'status':'passed' if not errors else 'failed','exit_code':int(bool(errors)),
        'created_utc':datetime.now(timezone.utc).isoformat(),'command':sys.argv,'audit_sha256':sha(__file__),
        'run':str(run),'checked_images':len(records),'checked_entry_conditions':sum(r['entry_count'] for r in records),
        'candidate_count':candidate_count,'surface_points_exact_source_indices':surface_points,'errors':errors,
        'new4_and_reused4_sources_unchanged':not errors,'frozen_total_input_images':292,'frozen_total_input_conditions':301,
        'response_files_rehashed':len(checked_hashes),'original_images_rehashed':False,
        'records':records,'scope':'artifact and constrained geometry validation; no identity annotation, S1 labels, reserved data, training or devices',
        'scoring_caution':'common-horizon smoothness cannot compensate for lost forward extent; inspect length ratios and free-only failures separately; no full-identity success rate'}
    save(run/outname,payload);print(json.dumps({k:v for k,v in payload.items() if k!='records'}))
    if errors:raise SystemExit(1)
if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('--run');p.add_argument('--out',default='independent_geometry_audit.json');p.add_argument('--baseline-all',action='store_true');a=p.parse_args()
    if a.baseline_all:baseline()
    else:audit(a.run,a.out)
