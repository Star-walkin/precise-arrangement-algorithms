from common21 import *
from events21 import analyze
from pose_transition21 import estimate
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--new',action='store_true');a=ap.parse_args()
for item in plan():
    id=item['id']
    if a.new:
        path=ROOT/'measurements'/id/'frames.json'
        if not path.exists():continue
        rows=read(path)
    else:
        rows=read(R20/'delivery'/id/'frames.json')
        for r in rows:
            c=r['contact'];r.update(current_coil_side_axis=c.get('current_coil_side_axis',False),boundary_occluded=c.get('boundary_occluded',True),boundary_anchor_hull=True)
    result=analyze(rows,item['fps'])
    if a.new:
        for event in result['events']:event['pose_transition']=estimate(rows,event)
    dump(ROOT/('events' if a.new else 'event_trial_baseline')/f'{id}.json',result)
    print(id,[{k:e[k] for k in ('start_s','entry_confirmed_s','end_s','departure_s','termination')} for e in result['events']],flush=True)
