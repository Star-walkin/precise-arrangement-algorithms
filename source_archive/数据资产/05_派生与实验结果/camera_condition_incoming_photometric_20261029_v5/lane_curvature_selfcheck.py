"""Independent synthetic geometry checks; prior slope-case failure is preserved."""
import json,math,time
import numpy as np
import whole_io as io
from lane_candidates import generate_lane_candidates
from lane_candidates_selfcheck import make_data

def main():
    start=time.time();free={'geometry':{'status':'observed','origin_xy':[20.,100.],'direction_xy':[1.,0.],'width_px':21.,'observed_segment_xy':[[2.,100.],[20.,100.]]}}
    data=make_data();r=generate_lane_candidates(data,free);checks={'multiple_legal_neighbor_candidates_retained':len(r['candidates'])>=2,'candidate_budget':len(r['candidates'])<=12,'reverse_options_explicitly_rejected':r['diagnostics']['join_rejections'].get('transition_reverse_curvature',0)>0}
    for c in r['candidates']:
        q=np.asarray(c['xy']);j=c['join_index'];t=c['transition_start_index'];q=q[t:j+1];mid=np.mean(q[:,0]);scale=np.ptp(q[:,0]);p=np.polyfit((q[:,0]-mid)/scale,q[:,1],3);xx=np.linspace(q[:,0].min(),q[:,0].max(),1001);theta=np.arctan(np.polyval(np.polyder(p),(xx-mid)/scale)/scale)
        down=float(np.max(np.maximum.accumulate(theta)-theta));up=float(np.max(theta-np.minimum.accumulate(theta)));sign=c['formal_curvature_sign'];reverse=down if sign>0 else (up if sign<0 else min(down,up))
        checks[c['candidate_id']+'_independent_dense_reverse_bound']=math.degrees(reverse)<=c['allowed_reverse_turn_deg']+.005
        checks[c['candidate_id']+'_formal_positive_curvature']=all(v>0 for v in c['formal_rail_second_derivatives'])
        # The known input parabola has y''=.0009. Raster fitting can vary but
        # must retain its sign and approximate magnitude in this clean scene.
        checks[c['candidate_id']+'_recovers_synthetic_second_derivative']=all(abs(v-.0009)<.0007 for v in c['formal_rail_second_derivatives'])
        checks[c['candidate_id']+'_unchanged_cost_definition']=math.isclose(c['geometry_cost'],abs(c['join_offset_px'])/c['transition_scale_px']+math.radians(c['join_angle_deg'])/.35+.15*min(1.5,abs(math.log(c['lane_free_width_ratio'])))+.04*c['transition_length_width']+.15*c['transition_turn_per_width'],abs_tol=1e-9)
    shifted=generate_lane_candidates(make_data(3),free);checks['shifted_refs_still_formal_competitors']=len(shifted['candidates'])>=2
    narrow=json.loads(json.dumps(free));narrow['geometry']['width_px']=10.;nr=generate_lane_candidates(data,narrow);checks['narrow_optical_width_not_hard_gate']=bool(nr['candidates'])
    checks={k:bool(v) for k,v in checks.items()};report={'scope':'Synthetic local tangent/curvature tests only; not real-image accuracy','prior_selfcheck_failure_preserved':'lane_candidates_selfcheck_003.json','changed_test_fixture_reason':'The original sloping ray has only one curvature-compatible meeting; this horizontal ray has several, so legal-neighbor preservation is still tested without requiring invalid S-shaped joins.','passed':all(checks.values()),'checks':checks,'candidate_count':len(r['candidates']),'shifted_count':len(shifted['candidates']),'narrow_count':len(nr['candidates']),'source_sha256':io.sha(io.HERE/'lane_candidates.py'),'seconds':time.time()-start}
    path=io.HERE/'lane_curvature_selfcheck_001.json'
    if path.exists():raise ValueError('No overwrite')
    io.dump(path,report);print(json.dumps(report,ensure_ascii=False));return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
