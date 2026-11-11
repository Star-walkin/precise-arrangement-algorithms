from collections import Counter
import numpy as np
from common import *
from boundary_bg7 import design
def main():
    run=ROOT/'boundary_final';binding=read(run/'binding.json');assert binding['boundary_code_sha256']==sha(ROOT/'boundary_bg7.py')
    rows=read(run/'compact.json');checks=[];failures=[]
    for i in range(600):
        n=record(i,run);old=record(i)
        q=dict(frame=n['frame']==i,source_pixel=n['source_pixel_sha256']==old['source_pixel_sha256'],
            source_record=sha(RUN12/'frames'/f'{i:06d}.json.gz')==n['source_record_sha256'],
            causal=all(s['frame']<=i for s in n.get('shape_state',[])),
            no_count_through_occlusion=not n['count_boundary_valid'] or not n.get('d_foreground_occluded'),
            no_physical_contact_claim=not n['physical_contact_boundary_calibrated'])
        if n.get('top_curve_xy'):
            xy=np.asarray(n['top_curve_xy']);c=np.asarray(n['fit_coefficients_ascending'])
            q['one_formula_for_all_segments']=bool(np.allclose(xy[:,1],design(xy[:,0],4096)@c,atol=1e-7))
            q['one_count_origin']=abs(n['edge_y']-design([D],4096)[0]@c)<1e-7
            accepted=set(map(tuple,n['top_observed_xy']));excluded=set(map(tuple,n['excluded_candidates_xy']))
            q['excluded_points_not_fitted']=not bool(accepted&excluded)
        if n['fit_valid']:q['current_points_required']=len(n['top_observed_xy'])>=30
        bad=[k for k,v in q.items() if not v]
        if bad:failures.append(dict(frame=i,failed=bad))
        checks.append(dict(frame=i,passed=not bad))
        if i%100==0:print('audit',i,len(failures),flush=True)
    stats=dict(frames=600,outline_fit_pass=sum(r['fit_valid'] for r in rows),
        count_boundary_eligible=sum(r['count_boundary_valid'] for r in rows),
        locally_supported_d=sum(bool(r.get('d_locally_observed')) and r['fit_valid'] for r in rows),
        model_only_d=sum(not bool(r.get('d_locally_observed')) and r['fit_valid'] for r in rows),
        foreground_at_d=sum(bool(r.get('d_foreground_occluded')) for r in rows),
        outline_after31_pass=sum(r['fit_valid'] for r in rows if r['frame']>=310),
        reasons=dict(Counter(v for r in rows for v in r['reasons'])),
        interpretation='algorithm eligibility coverage, not accuracy')
    dump(ROOT/'audit_receipt.json',dict(passed=not failures,frames=600,failures=failures,statistics=stats))
    dump(ROOT/'audit_rows.json',checks);assert not failures
    print(json.dumps(stats,ensure_ascii=False))
if __name__=='__main__':main()
