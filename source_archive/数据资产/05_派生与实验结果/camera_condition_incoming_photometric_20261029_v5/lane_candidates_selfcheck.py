"""Meaningful structural synthetic checks, not visual performance evidence."""
import argparse,hashlib,json,time
from pathlib import Path
import cv2,numpy as np
from lane_candidates import generate_lane_candidates,PARAMETERS
HERE=Path(__file__).resolve().parent

def make_data(shift=0):
    mask=np.zeros((340,520),np.uint8);x=np.arange(80.,481.,4.);refs=[]
    for k in range(8):
        y=60+k*22+.00045*(x-220)**2
        q=np.c_[x,y];cv2.polylines(mask,[np.rint(q).astype(np.int32)],False,255,1)
        refs.append({'curve_id':'synthetic_'+str(k),'source':'test_proposal','sampled_points_xy':(q+np.array([0,shift])).tolist()})
    return {'merged':mask,'inferred':np.zeros_like(mask),'pitch':22.,'curve_refs':refs}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',default='lane_candidates_selfcheck_001.json');args=parser.parse_args()
    if Path(args.receipt).name!=args.receipt:raise ValueError('Direct child receipt required')
    cv2.setNumThreads(1);started=time.time();free={'geometry':{'status':'observed','origin_xy':[20.,100.],'direction_xy':[.98418332,.177153],'width_px':21.,'observed_segment_xy':[[2.,96.76],[20.,100.]]}}
    data=make_data();original=data['merged'].copy();r=generate_lane_candidates(data,free)
    checks={'multiple_neighbor_candidates':len(r['candidates'])>=2,'candidate_budget':len(r['candidates'])<=12,'input_unchanged':np.array_equal(original,data['merged'])}
    for c in r['candidates']:
        xy=np.asarray(c['xy']);dist=np.asarray(c['formal_boundary_distance_px']);inf=np.asarray(c['point_inferred']);ph=np.asarray(c['phase']);j=c['join_index'];t=c['transition_start_index']
        checks[c['candidate_id']+'_finite']=bool(np.isfinite(xy).all() and np.isfinite(dist).all())
        checks[c['candidate_id']+'_aligned_arrays']=all(len(c[k])==len(xy) for k in ['widths','phase','point_inferred','formal_boundary_sources','formal_boundary_distance_px'])
        checks[c['candidate_id']+'_phase']=bool(0<t<j<len(xy) and ph[t]=='transition' and ph[j] in ('lane','lane_extrapolation'))
        take=(ph=='lane')&~inf
        checks[c['candidate_id']+'_actual_formal_support']=bool(np.any(take) and np.max(dist[take])<=1.)
        ext=np.flatnonzero(ph=='lane_extrapolation')
        checks[c['candidate_id']+'_finite_extension']=not len(ext) or abs(xy[-1,0]-xy[ext[0]-1,0])<=2*data['pitch']+1e-5
    # Reference geometry alone must never authorize a line absent from merged.
    blank=make_data();blank['merged'][:]=0
    checks['blank_formal_mask_rejects_reference_only']=not generate_lane_candidates(blank,free)['candidates']
    bad=generate_lane_candidates(data,{'geometry':{'status':'unknown'}})
    checks['unknown_free_has_no_lane_candidate']=not bad['candidates']
    # Small disagreement is snapped to real mask, not used as the final boundary.
    shifted=generate_lane_candidates(make_data(3),free)
    checks['small_ref_shift_recovers_formal_lanes']=len(shifted['candidates'])>=2
    narrow=json.loads(json.dumps(free));narrow['geometry']['width_px']=10.
    narrow_result=generate_lane_candidates(data,narrow)
    checks['narrow_optical_width_does_not_erase_formal_adjacent_lanes']=len(narrow_result['candidates'])>=2
    checks['narrow_transition_scale_not_below_free_or_lane_width']=all(c['transition_scale_px']>=10. and c['transition_scale_px']>=c['widths'][c['join_index']]-1e-6 for c in narrow_result['candidates'])
    checks={k:bool(v) for k,v in checks.items()}
    report={'scope':'Synthetic only; no real-image accuracy claim','passed':all(checks.values()),'checks':checks,'candidate_count':len(r['candidates']),'shifted_candidate_count':len(shifted['candidates']),'diagnostics':r['diagnostics'],'seconds':time.time()-started,'source_sha256':hashlib.sha256((HERE/'lane_candidates.py').read_bytes()).hexdigest()}
    out=HERE/args.receipt
    if out.exists():raise ValueError('Preserve previous selfcheck')
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':report['passed'],'candidate_count':len(r['candidates']),'failed':[k for k,v in checks.items() if not v],'seconds':report['seconds']}))
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
