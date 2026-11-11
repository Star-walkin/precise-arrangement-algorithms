"""Full-output invariants and paired stability metrics, not a truth-accuracy test."""
import argparse,json,time,hashlib,traceback
from collections import Counter
import numpy as np
import cv2
from scipy.interpolate import BSpline
from inputs import ROOT,BASE,sha,dump,PITCH,SCALE
from global_points import model_envelope,supported_path,sample

def read(p):return json.loads(p.read_text(encoding='utf8'))
def wait(p):
    start=time.monotonic()
    while True:
        try:return read(p)
        except (FileNotFoundError,json.JSONDecodeError):
            if time.monotonic()-start>1800:raise
            time.sleep(.3)

def summary(values):
    v=np.asarray(values,float)
    if not len(v):return None
    return dict(absolute_delta_median=float(np.median(np.abs(v))),absolute_delta_p95=float(np.percentile(np.abs(v),95)),
        maximum_absolute_delta=float(np.max(np.abs(v))),increasing_pairs=int(np.sum(v>0)))

def audit_frame(r,old,data,counts):
    i=r['frame_index'];assert r['source_pixel_sha256']==old['source_pixel_sha256']
    assert r['reused_BG1_record_sha256']==sha(BASE/'frames'/f'{i:06d}.json')
    assert r['region']==old['region'] and r['s1']==old['s1']
    assert r['fixed_d_original_px']==2238.7
    if not r['region_gate_passed']:
        assert r['BG3'] is None and not r['count_valid'] and r['display_N'] is None
        counts['blocked_frames']+=1;return
    b=r['BG3']
    if b is None:
        assert not r['count_valid'] and r['display_N'] is None
        counts['unresolved_models']+=1;return
    counts['modeled_frames']+=1
    assert str(data['source_pixel_sha256'])==r['source_pixel_sha256']
    p=float(data['pitch']);d=2238.7*SCALE
    assert np.array_equal(np.asarray(b['raw_points']),data['points'])
    points=np.asarray(b['raw_points']);assert np.array_equal(points[:,1:3]*2,np.asarray(old['F3']['raw_observed_points_xy']))
    # Validate that model-hole filling has not altered the actual observation mask.
    mp=BASE/r['masks']['surface']['path'];assert sha(mp)==r['masks']['surface']['sha256']
    original=cv2.imdecode(np.fromfile(mp,np.uint8),0)
    expected=(cv2.resize(original,(2048,1500),interpolation=cv2.INTER_AREA)==255)*255
    assert np.array_equal(expected,data['surface'])
    env=model_envelope(data['surface'],np.asarray(r['region']['top_curve_xy'])*SCALE)
    dg=b['diagnostics'];models={};allids=[];membership=np.asarray(dg['point_membership'])
    sd=dg['shared_shape'];center=BSpline(sd['knots_processing'],sd['coefficients'],3)
    stretch=BSpline(sd['knots_processing'],sd['stretch_coefficients'],3)
    def scale(x):return np.clip(1+stretch(x)-sd['stretch_anchor'],.65,1.5)
    residuals=[]
    for k,c in enumerate(b['curves']):
        xy=np.asarray(c['xy'])*SCALE;ids=np.asarray(c['point_ids'],int);raw=points[ids]
        assert len(ids)>=16 and np.all(raw[:,0]==ids)
        assert len(set(raw[:,1]))==len(raw)
        assert np.all(membership[ids]==k)
        assert len(xy)==len(c['point_states']) and np.all(np.diff(xy[:,0])==1)
        assert xy[0,0]==np.ceil(raw[:,1].min()) and xy[-1,0]==np.floor(raw[:,1].max())
        assert np.all(sample(env,xy)>254)
        _,_,support=supported_path(data,xy)
        assert np.array_equal(support,np.asarray(c['point_states'])=='observed')
        m=c['model'];spline=BSpline(m['residual_knots_processing'],m['residual_coefficients_processing'],m['degree'])
        def predict(x,spline=spline):return center(x)+scale(x)*spline(x)
        assert np.max(np.abs(predict(xy[:,0])-xy[:,1]))<1e-6
        models[c['boundary_id']]=(c,xy,predict,raw)
        residuals.extend((np.abs(predict(raw[:,1])-raw[:,2])/p).tolist())
        allids.extend(ids.tolist());counts['curves']+=1;counts['curve_samples']+=len(xy)
        counts['inferred_samples']+=int(np.sum(~support))
    assert len(allids)==len(set(allids))==dg['classified_point_count']
    assert set(allids)==set(np.flatnonzero(membership>=0))
    assert dg['class_count']==len(b['curves']) and dg['curve_splits']==0
    # Test both the serialized polyline and the analytic models between pixels.
    vals=list(models.values())
    if vals:
        gx0=min(v[1][0,0] for v in vals);gx1=max(v[1][-1,0] for v in vals)
        grid=np.arange(gx0,gx1+.001,.5);gap_grid=.45*p*scale(grid)
        dense={v[0]['boundary_id']:v[2](grid) for v in vals}
    for j,(ca,a,fa,ra) in enumerate(vals):
        for cb,bb,fb,rb in vals[j+1:]:
            lo=max(a[0,0],bb[0,0]);hi=min(a[-1,0],bb[-1,0])
            if hi<lo:continue
            ix0=round((lo-gx0)*2);ix1=round((hi-gx0)*2)+1
            diff=dense[cb['boundary_id']][ix0:ix1]-dense[ca['boundary_id']][ix0:ix1]
            # IDs are frame-local labels. A curve ending before d may have a
            # different label order without geometrically crossing its neighbor.
            assert np.min(diff)>0 or np.max(diff)<0,(i,ca['boundary_id'],cb['boundary_id'],'crossing')
            gap=gap_grid[ix0:ix1]
            assert np.min(np.abs(diff)-gap)>-1e-3,(i,ca['boundary_id'],cb['boundary_id'],'spacing')
            counts['ordered_pair_checks']+=1
    seen=[]
    for c in b['crossings']:
        seen.append(c['boundary_id']);model,xy,fn,raw=models[c['boundary_id']]
        assert xy[0,0]<=d<=xy[-1,0]
        y=float(np.interp(d,xy[:,0],xy[:,1]));assert abs(y-c['y']*SCALE)<1e-7
        inside,unknown,support=supported_path(data,np.asarray([[d,y]]))
        assert inside[0] and not unknown[0]
        left=raw[raw[:,1]<=d];right=raw[raw[:,1]>=d]
        assert len(left)>0 and len(right)>0
        span=right[:,1].min()-left[:,1].max()
        assert abs(span-c['point_bracket_gap_px'])<1e-7
        assert span<=(6 if support[0] else 2)*p
        assert c['status']==('image_supported_fitted_curve' if support[0] else 'point_bracketed_model_interpolation')
    assert len(seen)==len(set(seen));counts['unique_count_intersections']+=len(seen)
    if r['count_valid']:
        assert not r['reasons'] and not b['uncertain_count_evidence'] and r['s1']['valid']
        n=b['continuous_pitch_distance'];assert n is not None
        assert r['display_N']==(n if 0<=n<=5 else None)
    else:assert r['display_N'] is None and b['continuous_pitch_distance'] is None
    counts['raw_points']+=len(points);counts['classified_points']+=len(allids)
    return float(np.percentile(residuals,90)) if residuals else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='full_001');ap.add_argument('--stream',action='store_true');ap.add_argument('--stop-after',type=int)
    args=ap.parse_args();run=ROOT/args.run;binding=wait(run/'binding.json');counts=Counter();issues=[];q90=[]
    rows=[];old1=[];old2=[];reasons=Counter();rejected=Counter();t=time.perf_counter()
    for i in binding['frame_indices']:
        r=wait(run/'frames'/f'{i:06d}.json') if args.stream else read(run/'frames'/f'{i:06d}.json')
        old=read(BASE/'frames'/f'{i:06d}.json');data=None
        if r['region_gate_passed']:
            with np.load(ROOT/'cache'/f'{i:06d}.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
        try:
            err=audit_frame(r,old,data,counts)
            if err is not None:q90.append(err)
        except Exception as e:
            issue=dict(frame=i,error=repr(e),traceback=traceback.format_exc());issues.append(issue)
            print('AUDIT ISSUE',issue,flush=True)
        rows.append(r);old1.append(old)
        old2.append(read(ROOT.parent/'LC1_boundary_BG2_20261106_v1'/'full_001'/'frames'/f'{i:06d}.json'))
        reasons.update(r['reasons'])
        if r['BG3']:
            for c in r['BG3']['diagnostics']['rejections']:rejected.update(c['reasons'])
        if len(rows)%20==0:
            dump(run/'audit_progress.json',dict(audited=len(rows),issues=issues,elapsed_s=time.perf_counter()-t))
            print('Audited',len(rows),'issues',len(issues),flush=True)
        if args.stop_after and len(rows)>=args.stop_after:
            dump(run/'audit_subset.json',dict(audited=len(rows),issues=issues,counters=dict(counts)))
            return
    receipt=wait(run/'receipt.json');assert receipt['status']=='completed'
    for path,h in binding['code_sha256'].items():assert sha(path)==h,path
    metrics=dict(frames=len(rows),region_passed=sum(r['region_gate_passed'] for r in rows),
        region_blocked=sum(not r['region_gate_passed'] for r in rows),new_count_valid=sum(r['count_valid'] for r in rows),
        new_displayed=sum(r['display_N'] is not None for r in rows),BG2_count_valid=sum(r['count_valid'] for r in old2),
        BG2_displayed=sum(r['display_N'] is not None for r in old2),counts=dict(counts),reasons=dict(reasons),
        whole_class_rejection_reasons=dict(rejected),algorithm_runtime_s=receipt['elapsed_s'],absolute_physical_accuracy=None,
        raw_point_fit_q90_pitch_median=float(np.median(q90)),raw_point_fit_q90_pitch_max=float(np.max(q90)),
        matched_jumps={})
    for label,baseline in [('BG1',old1),('BG2',old2)]:
        pairs=[]
        for k in range(1,len(rows)):
            a,b=rows[k-1],rows[k];c,e=baseline[k-1],baseline[k]
            if not (121<=b['frame_index']<=270 and b['frame_index']-a['frame_index']==1):continue
            v=[z['display_N'] for z in (a,b,c,e)]
            if all(z is not None for z in v):pairs.append(dict(frame=b['frame_index'],old=v[3]-v[2],new=v[1]-v[0]))
        metrics['matched_jumps'][label]=dict(window=[121,270],pairs=len(pairs),old=summary([p['old'] for p in pairs]),
            new=summary([p['new'] for p in pairs]),details=pairs)
    target=ROOT/'delivery' if args.run=='full_001' else run
    dump(target/'geometry_audit.json',dict(status='passed' if not issues else 'failed',frames=len(rows),issues=issues,
        counters=dict(counts),elapsed_s=time.perf_counter()-t,code_hashes_unchanged=True,
        notes='Pixel-grid and 0.5-processing-pixel analytic ordering checks; no human ground-truth accuracy claim.'))
    dump(target/'metrics.json',metrics)
    print('AUDIT',len(issues),'issues',dict(counts),flush=True)
    if issues:raise SystemExit(1)
if __name__=='__main__':main()
