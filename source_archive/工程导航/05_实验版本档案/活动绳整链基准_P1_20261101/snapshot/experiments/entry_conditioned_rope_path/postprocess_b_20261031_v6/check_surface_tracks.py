"""Read-only H2/H6 geometry comparison on the fixed16; no new predictions."""
import argparse,json,time
from pathlib import Path
import numpy as np
import cv2
from common import HERE,inputs,sha,dump,surface
import surface_tracks as new


def numerical_checks():
    h,w=800,200;x=np.arange(20.,381.)
    curves=[dict(curve_id=f'c{i}',status='single_whole_group_conic',sampled_points_xy=np.c_[x,np.full(len(x),y)].tolist()) for i,y in enumerate([1200.,1224.,1248.,1272.,1296.])]
    merged=np.zeros((h,w),np.uint8)
    for c in curves:
        p=np.rint(np.asarray(c['sampled_points_xy'])*.5).astype(int);cv2.polylines(merged,[p],False,255,1)
    merged[:,75:100]=0
    exclusion=np.zeros_like(merged);exclusion[580:670,70:105]=255
    domain=np.zeros_like(merged);domain[550:750,5:195]=255
    p=dict(original_shape=[1600,400],original_half=np.full((h,w,3),128,np.uint8),
        f3_merged_half=merged,f3_inferred_half=np.zeros_like(merged),f3_exclusion_half=exclusion,
        surface_half=domain,pitch_px=24.,f3_vector_source_paths={'fixture':'synthetic'},
        f3_vector_records={'C3':{'fit':{'curves':curves}},'D4':{'fit':{'curves':[]}},'F2':{'joins':[]},'F3':{'groups':[],'observations':{'fit':{'curves':[{'not_adopted':True}]}}}})
    heat=np.full((275,w),.5,np.float32);a=new.build_tracks(p,heat)
    assert len(a['tracks'])==4 and a['diagnostics']['source_piece_count']==5
    assert a['diagnostics']['saved_source_gap_count']==5
    assert all('inferred_gap' in t['point_status'] for t in a['tracks'])
    assert all(not t['f3_pair_observed'][i] for t in a['tracks'] for i,s in enumerate(t['point_status']) if s=='inferred_gap')
    b=new.build_tracks(dict(p,f3_exclusion_half=np.zeros_like(exclusion)),heat)
    assert len(b['tracks'])==8 and b['diagnostics']['saved_source_gap_count']==0
    b=new.build_tracks(p,np.zeros_like(heat))
    assert [t['points_xy'] for t in a['tracks']]==[t['points_xy'] for t in b['tracks']]
    for t in a['tracks']:
        q=np.asarray(t['points_xy']);u=np.asarray(t['upper_boundary_xy']);l=np.asarray(t['lower_boundary_xy'])
        assert np.max(abs(q-(u+l)*.5))<1.e-9
        assert min(q[:,0])>=10 and max(q[:,0])<=190
        assert len(t['segment_status'])==len(q)-1 and len(t['boundary_ids'])==2
    # A missing interior boundary creates a double gap. It is rejected rather
    # than silently constructing a rope through the absent neighbor.
    cc=[c for c in curves if c['curve_id']!='c2'];rr=dict(p['f3_vector_records']);rr['C3']={'fit':{'curves':cc}}
    gap=new.build_tracks(dict(p,f3_vector_records=rr),heat)
    assert gap['diagnostics']['rejected'].get('local_spacing_ambiguity',0)>0
    assert all(max(t['vertical_spacing_px'])<20 for t in gap['tracks'])
    return dict(passed=True,checks=['accepted_C3_model_status_read','candidate_full_D4_not_adopted',
        'same_source_exclusion_gap_is_inferred','gap_never_observed','unknown_gap_remains_disconnected',
        'B_does_not_change_center_geometry','exact_adjacent_boundary_midpoint',
        'finite_source_extent_no_extrapolation','aligned_states_fixed_pair_ids','double_spacing_rejected'])


def stats(result,domain):
    lengths=np.array([t['length_px']*2 for t in result['tracks']])
    edge=cv2.distanceTransform((domain>0).astype(np.uint8),cv2.DIST_L2,3)*2
    near_length=0.;state_length={}
    for t in result['tracks']:
        q=np.asarray(t['points_xy']);arc=np.linalg.norm(np.diff(q,axis=0),axis=1)*2
        e=new._sample(edge,q);near_length+=float(arc[(e[:-1]<=40)&(e[1:]<=40)].sum())
        st=t.get('point_status',['observed' if z else 'inferred' for z in t['f3_pair_observed']])
        for k,d in enumerate(arc):
            state=st[k] if st[k]==st[k+1] else 'mixed_transition';state_length[state]=state_length.get(state,0.)+float(d)
    return dict(track_count=len(lengths),total_original_px=float(lengths.sum()),longest_original_px=float(lengths.max()) if len(lengths) else 0.,
        median_original_px=float(np.median(lengths)) if len(lengths) else 0.,
        length_within40_original_px_of_same_C4_boundary=near_length,point_state_interval_lengths_original_px=state_length)


def raster_difference(priors,result):
    merged=(priors['f3_merged_half'][525:]>0);render=np.zeros(merged.shape,np.uint8)
    observed_render=np.zeros_like(render)
    for b in result['boundaries']:
        for piece in b['pieces']:
            q=np.rint(piece['points_xy']).astype(np.int32);st=piece['point_status']
            for k in range(1,len(q)):
                if min(q[k,1],q[k-1,1])<0:continue
                cv2.line(render,tuple(q[k-1]),tuple(q[k]),255,1)
                if st[k]!='inferred_gap' and st[k-1]!='inferred_gap':cv2.line(observed_render,tuple(q[k-1]),tuple(q[k]),255,1)
    dv=cv2.distanceTransform((observed_render==0).astype(np.uint8),cv2.DIST_L2,3)
    df=cv2.distanceTransform((~merged).astype(np.uint8),cv2.DIST_L2,3)
    return dict(final_F3_raster_half_pixels=int(merged.sum()),
        final_F3_pixels_within2_half_px_of_saved_vectors=float((dv[merged]<=2).mean()),
        vector_raster_half_pixels=int((observed_render>0).sum()),
        vector_pixels_within2_half_px_of_final_F3=float((df[observed_render>0]<=2).mean()) if (observed_render>0).any() else None,
        inferred_gap_pixels_not_assumed_observed=int(((render>0)&(observed_render==0)).sum()),
        note='Raster geometric coverage only; neither annotation accuracy nor target identity.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='surface_checks_001');args=ap.parse_args()
    out=HERE/args.run;out.mkdir(exist_ok=False);tick=time.perf_counter();source=sha(HERE/'surface_tracks.py')
    report=dict(numeric=numerical_checks(),surface_sha256=source,script_sha256=sha(Path(__file__)),cases=[],
        labels_read=False,reserved_read=False,training=False,inference=False,devices=False,
        comparison_response='Each fixed image first unchanged original B entry; H6 geometry independently B-invariant.')
    cfg=inputs();allowed=set(cfg['pilot_ids'])
    for j in cfg['jobs']:
        if j['id'] not in allowed:continue
        p=new.load_surface_priors(dict(j,seeds=j['original_seeds']));heat=np.load(j['response_files'][0])['response']
        a=surface.build_tracks(p,heat);b=new.build_tracks(p,heat)
        report['cases'].append(dict(id=j['id'],old_H2=stats(a,p['surface_half'][525:]),new_H6=stats(b,p['surface_half'][525:]),
            raster_geometry=raster_difference(p,b),source_piece_count=b['diagnostics']['source_piece_count'],
            source_gap_count=b['diagnostics']['saved_source_gap_count'],rejected=b['diagnostics']['rejected'],
            surface_policy=b['diagnostics']['surface_policy'],source_receipts=p['input_dependency_sha256']))
    assert len(report['cases'])==16 and sha(HERE/'surface_tracks.py')==source
    report['elapsed_seconds']=time.perf_counter()-tick;report['exit_code']=0
    dump(out/'receipt.json',report);print(json.dumps({k:v for k,v in report.items() if k!='cases'}))


if __name__=='__main__':main()
