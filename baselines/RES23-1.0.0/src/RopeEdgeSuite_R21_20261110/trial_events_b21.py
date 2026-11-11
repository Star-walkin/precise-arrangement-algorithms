from common21 import *
from events21b import analyze
from pose_transition21 import estimate
for item in plan():
    id=item['id'];rows=read(ROOT/'final_measurements'/id/'frames.json');result=analyze(rows,item['fps']);old=read(ROOT/'events'/f'{id}.json')
    for e in result['events']:e['pose_transition']=estimate(rows,e)
    changes=[i for i,(a,b) in enumerate(zip(old['frames'],result['frames'])) if a['state']!=b['state'] or a.get('active_event_start_s')!=b.get('active_event_start_s')]
    result['changed_display_frames']=changes;dump(ROOT/'events_b'/f'{id}.json',result)
    print(id,'changes',len(changes),'events',[{k:e[k] for k in ['start_s','entry_confirmed_s','end_s','departure_s','termination']} for e in result['events']],flush=True)
