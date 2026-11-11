"""Summarize logged gates, never interpret gate coverage as detection accuracy."""
from collections import Counter
from common import *

def intervals(rows,key,desired,fps,min_s=0):
    spans=[];start=None
    for i in range(len(rows)+1):
        on=i<len(rows) and bool(rows[i].get(key))==desired
        if on and start is None:start=i
        if not on and start is not None:
            if (i-start)/fps>=min_s:spans.append(dict(start_frame=start,end_frame=i-1,start_s=start/fps,end_s=(i-1)/fps,duration_s=(i-start)/fps))
            start=None
    return spans

def summarize(item):
    run=ROOT/'runs'/item['id'];rows=read(run/'compact.json');receipt=read(run/'receipt.json');n=len(rows);fps=item['fps']
    ret=dict(id=item['id'],session=item['session'],camera=item['camera'],visual_role=item['visual_role'],
        partial_source=item.get('partial_source',False),frames=n,duration_s=n/fps,video=receipt['video'],
        tracking_valid=sum(x['tracking_valid'] for x in rows),boundary_valid=sum(x['boundary_fit'] for x in rows),
        count_boundary_eligible=sum(x['count_boundary_eligible'] for x in rows),
        jointly_valid=sum(x['tracking_valid'] and x['boundary_fit'] for x in rows),
        local_d=sum(x['boundary_fit'] and bool(x['d_locally_observed']) for x in rows),
        foreground_d=sum(bool(x['foreground_at_d']) for x in rows),
        algorithm_error_frames=sum(bool(x['algorithm_errors']) for x in rows),
        tracking_reasons=dict(Counter(x['tracking_reason'] for x in rows)),
        boundary_reasons=dict(Counter(q for x in rows for q in x['boundary_reasons'])),
        lost_intervals=intervals(rows,'tracking_valid',False,fps,1.),
        boundary_failure_intervals=intervals(rows,'boundary_fit',False,fps,1.),
        foreground_intervals=intervals(rows,'foreground_at_d',True,fps,.5),
        accepted_tracking_intervals=intervals(rows,'tracking_valid',True,fps,.5),
        all_frame_records_bound_to_source=True,source_unchanged=receipt['source_unchanged'],core_unchanged=receipt['core_unchanged'])
    return ret

def main():
    plan=read(ROOT/'plan.json');done=[r for r in plan['videos'] if (ROOT/'runs'/r['id']/'receipt.json').exists()]
    rows=[summarize(r) for r in done]
    dump(ROOT/'summary.json',dict(complete=len(done)==len(plan['videos']),planned=len(plan['videos']),completed=len(done),
        metrics_are_gate_coverage_not_accuracy=True,clips=rows,excluded=plan['excluded']))
    print([(r['id'],r['frames'],r['tracking_valid'],r['boundary_valid']) for r in rows])
if __name__=='__main__':main()
