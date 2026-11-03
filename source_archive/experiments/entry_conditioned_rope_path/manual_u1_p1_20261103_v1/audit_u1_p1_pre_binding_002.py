"""Independent saved-artifact audit: new U1 floats + unchanged P1/H6 arithmetic.

Does not import the adapter, renderer, detector, neural model or user labels.
Uses the prior independent H6/H5 arithmetic audit only, never their old outputs.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse,concurrent.futures,gzip,hashlib,importlib.util,json,sys,time,traceback
import cv2
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent;BASE=HERE.parent
BULK=Path('C:/precise_arrangement_results/manual_u1_p1_20261103_v1')
U1=BASE/'manual_r1_20261103_v1'
H6=BASE/'postprocess_b_20261031_v6'
EXPECTED_MODEL='a35494f57df6596f0369011a5a1b322c1d84288370a74da888528bdd2565216e'

def load(p):
    with (gzip.open if Path(p).suffix=='.gz' else open)(p,'rt',encoding='utf-8-sig') as f:return json.load(f)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def pure_audit():
    global _AUDIT
    if '_AUDIT' not in globals():
        spec=importlib.util.spec_from_file_location('_independent_prior_h6_arithmetic',H6/'audit_h6.py')
        _AUDIT=importlib.util.module_from_spec(spec);spec.loader.exec_module(_AUDIT)
    return _AUDIT
def sample(field,xy):
    p=np.asarray(xy,np.float32);shape=p.shape[:-1];q=p.reshape(-1,2);out=np.empty(len(q),np.float32)
    for k in range(0,len(q),24000):
        z=q[k:k+24000]
        out[k:k+len(z)]=cv2.remap(field,z[:,0][None],z[:,1][None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).ravel()
    return out.reshape(shape)
def profile_evidence(gray,heat,points,direction,width):
    """Numerically reconstruct the actual new-array source of saved free evidence."""
    p=np.asarray(points,float);d=np.asarray(direction,float);normal=np.array([-d[1],d[0]])
    u=np.linspace(-1.2,1.2,41);v=np.array([-1.,-.5,0.,.5,1.])
    q=p[:,None,None,:]+width*(v[None,:,None,None]*d+u[None,None,:,None]*normal)
    raw=sample(gray,q).mean(1);hp=sample(heat,q).mean(1)
    good=sample(np.ones(heat.shape,np.float32),q).mean((1,2))>=.98
    center=hp[:,abs(u)<=.18].mean(1);li=np.flatnonzero((u<=-.45)&(u>=-1.05));ri=np.flatnonzero((u>=.45)&(u<=1.05))
    leftidx=li[np.argmin(hp[:,li],axis=1)];rightidx=ri[np.argmin(hp[:,ri],axis=1)];row=np.arange(len(p))
    left=hp[row,leftidx];right=hp[row,rightidx];drop=center-np.maximum(left,right);asym=abs(abs(u[leftidx])-abs(u[rightidx]))
    lb=raw[:,u<=-.9].mean(1);rb=raw[:,u>=.9].mean(1)
    residual=raw-(lb[:,None]+(rb-lb)[:,None]*(u[None]+1.05)/2.1)
    body=residual[:,abs(u)<=.24].mean(1);polarity=np.where(body>=0,1.,-1.);scale=np.maximum(residual.std(1),.014)
    gradient=np.diff(raw,axis=1);mid=(u[:-1]+u[1:])/2
    le=gradient[:,(mid>-.8)&(mid<-.22)].sum(1)*polarity
    re=-gradient[:,(mid>.22)&(mid<.8)].sum(1)*polarity
    rawscore=(.65*np.clip(np.minimum(le,re)/(1.8*scale),0,1)+.35*np.clip(abs(body)/(2*scale),0,1))*np.clip(abs(body)/(.014*3),0,1)
    return dict(center=center,left_low=left,right_low=right,side_drop=drop,
                side_distance_asymmetry_widths=asym,raw_score=rawscore,raw_contrast=abs(body),valid=good)
def track_evidence(gray,heat,track):
    p=np.asarray(track['points_xy'],float);d=np.gradient(p,axis=0);d/=np.maximum(np.linalg.norm(d,axis=1)[:,None],1e-8)
    n=np.c_[-d[:,1],d[:,0]]
    width=np.maximum((np.asarray(track['lower_boundary_xy'])[:,1]-np.asarray(track['upper_boundary_xy'])[:,1])*abs(n[:,1]),1.)
    assert np.allclose(track['spacing_px'],width,rtol=0,atol=1e-9)
    offsets=np.array([-.5,-.4,-.2,0,.2,.4,.5],np.float32)
    q=p[:,None]+n[:,None]*width[:,None,None]*offsets[None,:,None]
    raw=sample(gray,q);hh=sample(heat,q);rc=raw[:,2:5].mean(1);hc=hh[:,2:5].mean(1)
    rd=np.minimum(rc-raw[:,:2].mean(1),rc-raw[:,-2:].mean(1));hd=np.minimum(hc-hh[:,:2].mean(1),hc-hh[:,-2:].mean(1))
    return dict(raw_score=np.clip(rd/np.maximum(raw.std(1),.02),0,1),heat_score=np.clip(hd/.15,0,1),
                heat_level=hc,raw_signed_center_minus_sides=rd,heat_signed_center_minus_sides=hd)
def compare_fields(actual,expected,tolerance=1e-7):
    largest=0.
    for key,value in expected.items():
        saved=np.asarray(actual[key]);value=np.asarray(value);assert saved.shape==value.shape,(key,saved.shape,value.shape)
        if value.dtype==bool:assert np.array_equal(saved,value),key
        else:
            diff=float(np.max(np.abs(saved-value))) if value.size else 0.;largest=max(largest,diff)
            assert np.allclose(saved,value,rtol=0,atol=tolerance),(key,diff)
    return largest
def original_heat_view(a):
    # Four neighboring source samples, no renderer import or image resize API.
    h,w=a.shape;u=(np.arange(2*w,dtype=np.float32)+.5)*.5-.5;v=(np.arange(2*h,dtype=np.float32)+.5)*.5-.5
    x=np.floor(u).astype(int);y=np.floor(v).astype(int);dx=u-x;dy=v-y
    x0=x.clip(0,w-1);x1=(x+1).clip(0,w-1);y0=y.clip(0,h-1);y1=(y+1).clip(0,h-1)
    out=np.empty((2*h,2*w),np.float32)
    for k in range(0,2*h,96):
        stop=min(k+96,2*h);wy=dy[k:stop,None]
        out[k:stop]=(a[y0[k:stop,None],x0]*(1-dx)*(1-wy)+a[y0[k:stop,None],x1]*dx*(1-wy)+
                     a[y1[k:stop,None],x0]*(1-dx)*wy+a[y1[k:stop,None],x1]*dx*wy)
    lut=cv2.applyColorMap(np.arange(256,dtype=np.uint8)[:,None],cv2.COLORMAP_TURBO)[:,0,::-1]
    display=np.full((3000,4096,3),45,np.uint8);display[1050:]=lut[np.rint(out*255).astype(np.uint8)]
    return display
def audit_one(args):
    sid,run,render_row=args;cv2.setNumThreads(1);api=pure_audit();freeze=load(HERE/'adapter_freeze.json')
    record_path=Path(run)/'记录'/(sid+'.json.gz');r=load(record_path)
    original_job=next(j for j in load(H6/'inputs.json')['jobs'] if j['id']==sid)
    task_job=next(j for j in load(U1/'protocol.json')['gallery_jobs'] if j['id']==sid)
    assert r['id']==sid and r['source_image_sha256']==task_job['image_sha256']==original_job['image_sha256']
    assert sha(r['source_image'])==r['source_image_sha256'];assert r['U1_model']['sha256']==EXPECTED_MODEL
    assert not r['old_B_heat_or_paths_loaded'] and r['old_H1_paths_diagnostic_count']==0 and r['all_heat_dependent_evidence_recomputed']
    assert not r['labels_read'] and not r['device_connected'] and not r['response_entry_conditions_merged']
    assert r['source_freeze_sha256']==sha(HERE/'adapter_freeze.json')
    assert len(r['entries'])==len(original_job['seeds'])==len(task_job['seeds'])
    # Pure independent source-vector midpoint reconstruction and no-extrapolation.
    hashes={};tracks=api.verify_tracks(r,freeze['surface_parameters'],hashes)
    bgr=cv2.imdecode(np.fromfile(r['source_image'],np.uint8),cv2.IMREAD_COLOR)
    half=cv2.resize(bgr,(2048,1500),interpolation=cv2.INTER_AREA)
    raw=cv2.cvtColor(half[525:],cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
    smoothed=cv2.GaussianBlur(raw,(0,0),.7)
    counters=Counter();float_differences=[];selected_heat=None
    for ei,e in enumerate(r['entries']):
        expected=U1/'inference/gallery/U1'/sid/f'entry_{ei:02d}.npz'
        assert Path(e['response_file']).resolve()==expected.resolve()
        metadata=load(expected.with_suffix('.json'));assert e['response_metadata']==metadata
        assert metadata['model_sha256']==EXPECTED_MODEL and metadata['model_step']==800 and metadata['branch']=='U1'
        assert metadata['target_labels_read'] is False and metadata['new_model_forward'] is True
        assert metadata['image_sha256']==r['source_image_sha256'] and metadata['entry_index']==ei
        assert np.array_equal(e['seed']['xy'],task_job['seeds'][ei]['xy']) and e['seed']==original_job['seeds'][ei]
        assert np.array_equal(e['seed']['xy'],metadata['seed_xy'])
        assert sha(expected)==e['response_sha256']==metadata['response_sha256']
        with np.load(expected,allow_pickle=False) as z:a=z['response'].copy()
        assert a.shape==(975,2048) and a.dtype==np.float32 and np.isfinite(a).all() and 0<=a.min()<=a.max()<=1
        if ei==r['selected_entry']:selected_heat=a
        heat_blur=cv2.GaussianBlur(a,(0,0),.7)
        full=e['full_free_observation'];f=e['free'];p=api.pts(full['points_xy']);n=len(p)
        if n:
            direction=full['diagnostics']['selected_direction_xy'];width=full['width_half_px']
            fresh=profile_evidence(smoothed,heat_blur,p,direction,width)
            float_differences.append(compare_fields(full['evidence'],fresh))
            hp=full['diagnostics']['parameters'];ev={k:np.asarray(v) for k,v in full['evidence'].items()}
            heat_ok=ev['valid']&(ev['center']>=hp['minimum_heat_center'])&(ev['side_drop']>=hp['minimum_side_drop'])&(ev['side_distance_asymmetry_widths']<=hp['maximum_side_distance_asymmetry_widths'])
            image_ok=heat_ok&(ev['raw_score']>=hp['minimum_raw_score'])&(ev['raw_contrast']>=hp['minimum_image_contrast'])
            assert np.array_equal(ev['heat_profile_supported'],heat_ok) and np.array_equal(ev['image_heat_supported'],image_ok)
            assert api.prior.line_residual(p)<1e-7
        api.prior.validate_raw_a7(full,r);api.prior.validate_r4_pool(full)
        projection=api.verify_projection(e,list(tracks.values()),freeze['topology_parameters'])
        counters['trimmed_conditions']+=int(projection['trimmed']);counters['licensed_free_projection_events']+=projection['retained_events']
        candidates=e['connection']['candidates'];assert len(candidates)<=3
        assert e['connection'].get('direct_H1_paths_diagnostic_only',0)==0 and e['connection'].get('off_track_H1_candidates_selected',0)==0
        for c in candidates:
            cp=api.prior.aligned_points(c['points_xy']);nn=len(cp)
            assert all(len(c[key])==nn for key in ('supported','point_status','stage','width_half_px'))
            assert not c['identity_verified'];api.prior.validate_score(c,e['connection']['parameters'],f)
            nf=int(np.sum(np.asarray(c['stage'])=='free'));assert np.array_equal(cp[:nf],api.pts(f['points_xy'])[:nf])
            assert c['point_status'][:nf]==f['point_status'][:nf] and c['supported'][:nf]==f['supported'][:nf]
            if c['kind']=='straight_free_to_F3_center_track':
                counters['attached_candidates']+=1;counters['surface_exact_source_points']+=api.verify_attachment(c,tracks)
                assert np.array_equal(e['candidate_track_geometry'][c['track_id']],tracks[c['track_id']]['points_xy'])
            else:assert c['kind']=='free_fragment_no_resolved_connection' and np.array_equal(cp,api.pts(f['points_xy']))
            counters['candidates']+=1
        counters['entry_conditions']+=1
        # Geometry independent of heat, evidence NOT independent of heat. Check
        # every entry's track measurements against exactly this new response.
        if ei==r['selected_entry']:
            for tr in tracks.values():float_differences.append(compare_fields(tr,track_evidence(raw,a,tr)))
    counters['task_direction_excluded']+=api.prior.validate_direction(r,freeze['parameters'])
    ranked=[]
    for ei,e in enumerate(r['entries']):
        for ci,c in enumerate(e['connection']['candidates']):ranked.append((c['rank_score'],ei,ci,c))
    ranked.sort(key=lambda x:(r['entries'][x[1]]['task_direction']['selection_allowed'],x[0],x[3]['metrics'].get('free',{}).get('quality',0)),reverse=True)
    best=ranked[0] if ranked else None
    assert r['selected_entry']==(best[1] if best else 0) and r['selected_candidate']==(best[2] if best else None)
    assert [(q['entry'],q['candidate'],q['score']) for q in r['displayed_candidates']]==[(b[1],b[2],b[0]) for b in ranked[:3]]
    if render_row is not None:
        # Assigned after renderer receipt schema is observed, never inferred
        # from image appearance. These are fixed native-coordinate viewports.
        path=Path(render_row['output']);assert sha(path)==render_row['sha256']
        boxes=render_row['image_viewports_xyxy'];assert len(boxes)==4
        with Image.open(path) as im:
            assert im.format=='PNG' and im.size==(16620,3400)
            assert all((b[2]-b[0],b[3]-b[1])==(4096,3000) for b in boxes)
            assert np.array_equal(np.asarray(im.crop(boxes[0])),bgr[:,:,::-1])
            assert np.array_equal(np.asarray(im.crop(boxes[1])),original_heat_view(selected_heat)), 'selected U1 raw-heat raster mismatch'
        counters['native_png_left_and_heat_pixels_checked']+=1
    return dict(id=sid,entries=len(r['entries']),selected_entry=r['selected_entry'],selected_candidate=r['selected_candidate'],
                record_path=str(record_path),record_sha256=sha(record_path),counts=dict(counters),
                max_new_U1_evidence_difference=max(float_differences,default=0.),
                complete_identity_accuracy=None)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',default='all_001')
    parser.add_argument('--records-only',action='store_true');parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--out',default='audit_u1_p1_receipt.json');args=parser.parse_args()
    run=BULK/args.run;summary=load(run/'summary.json');start=load(run/'run_start.json')
    assert summary['exit_code']==0 and summary['completed']==summary['planned']==len(start['input_ids'])
    source=load(HERE/'p1_source_integrity_independent.json');assert source['all_current_match'] and source['all_snapshot_match']
    for row in source['rows']:assert sha(row['origin'])==row['expected_sha256']
    frozen=load(HERE/'adapter_freeze.json');assert not frozen['new_training'] and not frozen['new_model_forward'] and not frozen['R1_used']
    assert sha(frozen['model']['path'])==EXPECTED_MODEL==frozen['model']['sha256']
    assert start['frozen_adapter_sha256']==frozen['source_sha256']==sha(HERE/'run_p1_u1.py')
    render=None
    if not args.records_only:
        receipt=load(HERE/'render_receipt.json');assert receipt['exit_code']==0 and len(receipt['rows'])==292
        render={r['id']:r for r in receipt['rows']};assert set(render)==set(start['input_ids'])
        assert len(list((BULK/'四栏检测结果').glob('*.png')))==292
    started=time.perf_counter();rows=[];errors=[]
    with concurrent.futures.ProcessPoolExecutor(args.workers) as executor:
        futures={executor.submit(audit_one,(sid,run,None if render is None else render[sid])):sid for sid in start['input_ids']}
        for f in concurrent.futures.as_completed(futures):
            try:rows.append(f.result())
            except Exception:errors.append(dict(id=futures[f],traceback=traceback.format_exc()))
            if (len(rows)+len(errors))%12==0 or len(rows)+len(errors)==len(futures):
                print(dict(done=len(rows),errors=len(errors),total=len(futures),seconds=round(time.perf_counter()-started,1)),flush=True)
    counts=Counter()
    for row in rows:counts.update(row['counts'])
    if not errors:assert counts['entry_conditions']==start['expected_entries']
    report=dict(exit_code=int(bool(errors)),command=sys.argv,seconds=time.perf_counter()-started,
        checked_images=len(rows),planned=len(start['input_ids']),records_only=args.records_only,
        source_sha256=sha(__file__),source_integrity_sha256=sha(HERE/'p1_source_integrity_independent.json'),
        counts=dict(counts),max_new_U1_evidence_difference=max((r['max_new_U1_evidence_difference'] for r in rows),default=0),
        no_adapter_or_renderer_import=True,new_U1_profile_and_track_values_reconstructed=True,
        no_old_B_response_or_path_loaded=True,source_no_source_extrapolation_and_non_crossing_arithmetic_checked=True,
        complete_identity_accuracy=None,errors=errors,rows=rows)
    target=HERE/args.out
    if target.exists():raise ValueError('Preserve prior audit output; choose a new --out')
    dump(target,report);print(json.dumps({k:v for k,v in report.items() if k not in ('rows','errors')},indent=2))
    if errors:print(json.dumps(errors[:2],indent=2));raise SystemExit(1)

if __name__=='__main__':main()
