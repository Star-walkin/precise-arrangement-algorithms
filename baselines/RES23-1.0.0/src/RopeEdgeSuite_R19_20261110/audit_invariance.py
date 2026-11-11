"""Audit whether BG11 could change any recorded BG10 transition.

When its trigger is always false, BG11 delegates and returns exactly BG10:
identical initial state implies identical subsequent states and outputs by
induction. Camera reset handling is also identical. This is a trace branch
invariance check, not a claim to have rerun every frame through the new wrapper.
"""
from common import *
from boundary_bg11 import startup_repair_eligible
OLD=PARENT/'RopeEdgeSuite_R18_20261110'
def main():
    rows=[];previous={}
    if (ROOT/'trace_invariance.json').exists():
        saved=read(ROOT/'trace_invariance.json')
        if saved['candidate_sha256']==sha(ROOT/'boundary_bg11.py'):previous={r['id']:r for r in saved['videos']}
    for item in read(ROOT/'plan.json')['videos']:
        id=item['id'];run=OLD/'runs'/id;receipt_path=run/'receipt.json'
        if not receipt_path.exists():continue
        receipt=read(receipt_path);assert receipt['status']=='complete';triggers=[]
        if id in previous and previous[id]['original_receipt_sha256']==sha(receipt_path):
            rows.append(previous[id]);print(id,'verified audit reused',flush=True);continue
        for i in range(item['planned_frames']):
            r=record(run,i);b=dict(r['boundary'])
            # The wrapper is called before the runner appends camera reasons.
            b['reasons']=[s for s in b['reasons'] if s!='camera_motion_invalid']
            if startup_repair_eligible(b):triggers.append(i)
        rows.append(dict(id=id,frames=item['planned_frames'],trigger_frames=triggers,
            numerically_invariant=not triggers,original_receipt_sha256=sha(receipt_path)))
        print(id,'trigger',triggers,flush=True)
    dump(ROOT/'trace_invariance.json',dict(method='Exhaustive recorded-transition trigger audit plus unchanged delegation; not a fresh whole-video rerun',
        candidate_sha256=sha(ROOT/'boundary_bg11.py'),videos=rows))
if __name__=='__main__':main()
