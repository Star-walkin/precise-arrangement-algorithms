from collections import Counter
from common import *
BASELINE=PARENT/'RopeEdgeSuite_Batch_RES14_20261109';OLD=PARENT/'RopeEdgeSuite_R18_20261110'
def choose_run(id):
    audit=next(x for x in read(ROOT/'trace_invariance.json')['videos'] if x['id']==id)
    new=ROOT/'runs'/id
    if (new/'receipt.json').exists():return new
    assert audit['numerically_invariant'],('New run required',id,audit['trigger_frames'])
    return OLD/'runs'/id
def spans(flags,fps):
    found=[];start=None
    for i,v in enumerate(flags+[False]):
        if v and start is None:start=i
        if not v and start is not None:found.append([start/fps,(i-1)/fps,i-start]);start=None
    return sorted(found,key=lambda r:-r[2])
def main(*args):
    out=[]
    for item in read(ROOT/'plan.json')['videos']:
        run=choose_run(item['id']);new=read(run/'compact.json');old=read(BASELINE/'runs'/item['id']/'compact.json');assert len(new)==len(old)
        changes={}
        for name in ['tracking_valid','boundary_fit','count_boundary_eligible']:
            changes[name]=dict(old=sum(r[name] for r in old),new=sum(r[name] for r in new),
                recovered=sum(not a[name] and b[name] for a,b in zip(old,new)),lost=sum(a[name] and not b[name] for a,b in zip(old,new)),
                longest_missing_intervals_s=spans([not r[name] for r in new],item['fps'])[:8])
        out.append(dict(id=item['id'],frames=len(new),role=item['visual_role'],changes=changes,
            selected_run=str(run),reused_after_invariance_audit=run.parent.parent==OLD,
            boundary_reasons=Counter(x for r in new if not r['boundary_fit'] for x in r['boundary_reasons']),
            errors=sum(bool(r['algorithm_errors']) for r in new)))
    dump(ROOT/'comparison.json',dict(note='Current-evidence gate availability, not labelled accuracy',videos=out))
    for r in out:print(r['id'],'track',r['changes']['tracking_valid']['old'],r['changes']['tracking_valid']['new'],
        'boundary',r['changes']['boundary_fit']['old'],r['changes']['boundary_fit']['new'],flush=True)
if __name__=='__main__':main()
