"""Coverage diagnostics, explicitly not accuracy without independent ground truth."""
from collections import Counter
from common import *
OLD=PARENT/'RopeEdgeSuite_Batch_RES14_20261109'

def spans(flags,fps):
    spans=[];start=None
    for i,v in enumerate(flags+[False]):
        if v and start is None:start=i
        if not v and start is not None:spans.append([start/fps,(i-1)/fps,i-start]);start=None
    return sorted(spans,key=lambda q:-q[2])

def main(runs_name='runs',output_name='comparison.json'):
    out=[]
    for item in read(ROOT/'plan.json')['videos']:
        run=ROOT/runs_name/item['id']
        if not (run/'receipt.json').exists():continue
        new=read(run/'compact.json');old=read(OLD/'runs'/item['id']/'compact.json');assert len(new)==len(old)
        changes={}
        for name in ['tracking_valid','boundary_fit','count_boundary_eligible']:
            changes[name]=dict(old=sum(r[name] for r in old),new=sum(r[name] for r in new),
                recovered=sum(not o[name] and n[name] for o,n in zip(old,new)),
                lost=sum(o[name] and not n[name] for o,n in zip(old,new)),
                longest_missing_intervals_s=spans([not r[name] for r in new],item['fps'])[:8])
        out.append(dict(id=item['id'],frames=len(new),role=item['visual_role'],changes=changes,
            boundary_reasons=Counter(x for r in new if not r['boundary_fit'] for x in r['boundary_reasons']),
            errors=sum(bool(r['algorithm_errors']) for r in new)))
    dump(ROOT/output_name,dict(note='Internal gate availability, not independently labelled detection accuracy',runs=runs_name,videos=out))
    for x in out:
        print(x['id'],x['frames'],'track',x['changes']['tracking_valid']['old'],x['changes']['tracking_valid']['new'],
            'boundary',x['changes']['boundary_fit']['old'],x['changes']['boundary_fit']['new'],'errors',x['errors'])
if __name__=='__main__':main()
