from common21 import *
from pose_transition21 import estimate
id='dual_20260924_111500__left';event=read(ROOT/'event_trial_baseline'/f'{id}.json')['events'][0]
rows=[]
for r in read(R20/'delivery'/id/'frames.json'):
    if event['start_s']-2<=r['time_s']<=event['departure_s']+2:
        a=geometry(id,r['frame']).get('axis') or {};abc=a.get('line_abc');c=r['contact']
        rows.append(dict(r,axis_slope=-abc[0]/abc[1] if abc else None,current_coil_side_axis=a.get('current_coil_side_measured',False),boundary_occluded=c.get('boundary_occluded',True)))
result=estimate(rows,event);print(result);dump(ROOT/'inspection'/'111500_pose_trial.json',result)
