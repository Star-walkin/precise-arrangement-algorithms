"""Incremental, read-only audit of all committed transfer frame records."""
import argparse,gzip,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def brief_ecw(x):
    return {k:v for k,v in x.items() if not isinstance(v,(list,dict))}
def analyze(run,i):
    path=run/'frames'/f'{i:06d}.json.gz'
    with gzip.open(path,'rt',encoding='utf8') as f:r=json.load(f)
    failures=[]
    def check(ok,name):
        if not ok:failures.append(name)
    check(r['frame_index']==i,'frame_index')
    check(r['fixed_d_original_px']==2238.7,'fixed_d')
    check(r['ECW']['source_pixel_sha256']==r['source_pixel_sha256'],'ECW_pixel_binding')
    check(r['fresh_F3']['called']==r['region']['valid'],'F3_only_after_region_gate')
    check(not r['ECW']['physical_contact_confirmed'],'physical_contact_not_asserted')
    ev=r['ECW_decision']['evidence'];temp=r['temporal']
    check(not ev['near'] or ev['local_valid'],'near_requires_local_evidence')
    check(not ev['near'] or ev['geometric_consistency'],'near_geometric_agreement')
    check(ev['physical_gap_px'] is None,'physical_gap_not_invented')
    val=temp['display_N']
    check(val is None or (0<=val<=5 and val==temp['value']),'display_scope')
    if temp['state']!='unknown':
        check(temp['last_observation_age_s']<=.300001,'maximum_prediction_age')
        check(temp['last_phase_age_s']<=1.000001,'maximum_phase_age')
        check(temp['nominal_filter_sigma']<=.650001,'maximum_filter_sigma')
    raw=np.asarray(r['fresh_F3']['raw_points'],float).reshape(-1,8)
    check(len(raw)==r['fresh_F3']['point_count'],'raw_point_count')
    byid={int(p[0]):p for p in raw};owned=set();bid={};curves=0;crossings=0;recovered=0;max_outside=0.
    if r['BG4']:
        for c in r['BG4']['curves']:
            curves+=1;recovered+=int(c['recovered_local']);xy=np.asarray(c['xy'],float);ids=c['point_ids']
            check(c['boundary_id'] not in bid,'unique_boundary_id');bid[c['boundary_id']]=c
            check(len(xy)==len(c['point_states']),'point_state_length')
            check(np.all(np.isfinite(xy)) and np.all(np.diff(xy[:,0])>0),'finite_x_monotonic_curve')
            check(not owned.intersection(ids),'raw_point_unique_owner');owned.update(ids)
            check(all(j in byid for j in ids),'point_ids_in_current_frame')
            if ids and all(j in byid for j in ids):
                px=np.asarray([byid[j][1]*2 for j in ids])
                check(xy[0,0]>=px.min()-2.001 and xy[-1,0]<=px.max()+2.001,'curve_support_domain')
            max_outside=max(max_outside,c.get('outside_validated_surface_fraction',0.))
        seen=set()
        for c in r['BG4']['crossings']:
            crossings+=1;b=c['boundary_id'];check(b not in seen,'one_crossing_per_boundary');seen.add(b)
            check(b in bid,'crossing_boundary_exists')
            if b in bid:
                xy=np.asarray(bid[b]['xy']);check(xy[0,0]<=2238.7<=xy[-1,0],'crossing_inside_curve_domain')
                check(abs(c['y']-np.interp(2238.7,xy[:,0],xy[:,1]))<1e-6,'display_count_same_curve')
    old=read(run/'bg1/frames'/f'{i:06d}.json');reg=old['region']
    check(old['source_pixel_sha256']==r['source_pixel_sha256'],'BG1_pixel_binding')
    s=r['s1'];span=None;extrap=None
    if s['valid']:
        xy=np.asarray(s['line_points_xy']);span=float(np.ptp(xy[:,0]));extrap=float(max(0,2238.7-xy[:,0].max(),xy[:,0].min()-2238.7))
    return dict(frame=i,failures=sorted(set(failures)),s1_valid=s['valid'],region_valid=r['region']['valid'],
        region_reasons=r['region']['reasons'],outer_reason=reg.get('outer_rim',{}).get('reason'),
        s1_reason=r['S1_tracker_full']['diagnostics'].get('accept_reason'),
        top_reasons=r['reasons'],count_events=temp['events'],count_state=temp['state'],count_value=temp['value'],display_N=val,
        count_sigma=temp['nominal_filter_sigma'],direction=temp['direction'],
        ecw_valid=r['ECW']['valid'],ecw_near=ev['near'],ecw_far=ev['far'],ecw_state=r['ECW_decision']['temporal']['state'],
        ecw_rope_reason=r['ECW'].get('rope',{}).get('reason',r['ECW'].get('reason')),
        ecw_boundary_reason=r['ECW'].get('boundary',{}).get('reason'),
        ecw_gap=r['ECW'].get('center_gap_proxy_px'),ecw_extrapolation_px=r['ECW'].get('rope',{}).get('extrapolation_to_d_px'),
        s1_support_span=span,s1_extrapolation_px=extrap,
        camera_changed=r['camera_motion']['camera_changed'],raw_points=len(raw),curves=curves,crossings=crossings,
        recovered=recovered,max_curve_outside_fraction=max_outside,
        front_s=r['frontend_elapsed_s'],spatial_s=r['spatial_elapsed_s'])
def counter(rows,key):return dict(Counter(x for r in rows for x in (r[key] if isinstance(r[key],list) else [r[key]]) if x is not None))
def quant(rows,key):
    a=[r[key] for r in rows if r[key] is not None];return None if not a else dict(min=float(np.min(a)),median=float(np.median(a)),max=float(np.max(a)))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--session',required=True);a=ap.parse_args();run=ROOT/'transfer_runs'/a.session
    n=read(run/'progress.json')['processed'];dest=run/'audit_rows.json';rows=read(dest) if dest.exists() else []
    if rows and 'ecw_extrapolation_px' not in rows[0]:rows=[]
    for i in range(len(rows),n):rows.append(analyze(run,i))
    dump(dest,rows)
    done=(run/'receipt.json').exists();binding=read(run/'binding.json')
    if done:assert read(run/'receipt.json')['frames']==len(rows)==binding['source_frames']
    pairs=[(a,b,b['display_N']-a['display_N']) for a,b in zip(rows,rows[1:]) if a['display_N'] is not None and b['display_N'] is not None]
    rises=[x for x in pairs if x[2]>.01]
    summary=dict(session=a.session,complete=done,frames=len(rows),planned=binding['source_frames'],
        checks_passed=not any(r['failures'] for r in rows),check_failures=counter(rows,'failures'),
        S1_valid=sum(r['s1_valid'] for r in rows),region_valid=sum(r['region_valid'] for r in rows),
        displayed_count=sum(r['display_N'] is not None for r in rows),ecw_valid=sum(r['ecw_valid'] for r in rows),
        near_frames=sum(r['ecw_near'] for r in rows),far_frames=sum(r['ecw_far'] for r in rows),
        curves=sum(r['curves'] for r in rows),crossings=sum(r['crossings'] for r in rows),recovered=sum(r['recovered'] for r in rows),
        count_states=counter(rows,'count_state'),region_reasons=counter(rows,'region_reasons'),outer_reasons=counter(rows,'outer_reason'),
        S1_reasons=counter(rows,'s1_reason'),ECW_rope_reasons=counter(rows,'ecw_rope_reason'),ECW_states=counter(rows,'ecw_state'),
        count_events=counter(rows,'count_events'),S1_extrapolation_px=quant(rows,'s1_extrapolation_px'),
        ECW_extrapolation_px=quant(rows,'ecw_extrapolation_px'),
        count_value=quant(rows,'count_value'),frontend_s=quant(rows,'front_s'),spatial_s=quant(rows,'spatial_s'),
        display_step_diagnostics=dict(adjacent_display_pairs=len(pairs),rises_over_0_01=len(rises),
            largest_rise=None if not rises else float(max(x[2] for x in rises)),
            rise_direction_counts=dict(Counter(x[1]['direction'] for x in rises)),
            note='Rises are output behavior, not labeled error; physical reversals are not independently annotated.'),
        absolute_accuracy_validated=False,calibration_transfer_validated=False)
    dump(run/'audit_summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
