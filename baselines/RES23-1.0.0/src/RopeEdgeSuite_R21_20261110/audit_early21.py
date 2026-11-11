from common21 import *
import collections
id='dual_20260924_111143__right';rows=[]
for i in range(152):
    row=gzread(ROOT/'measurements'/id/'frames'/f'{i:06d}.json.gz');m=row['measurement']
    rows.append(dict(frame=i,t=i/10,available=m['available'],N=m.get('N'),interval=m.get('N_interval'),d=m.get('d'),mode=m['measurement_mode'],reasons=m['reasons']))
good=[r for r in rows if r['available']]
result=dict(frames=len(rows),available=len(good),first_time=good[0]['t'] if good else None,
    reasons=dict(collections.Counter(s for r in rows if not r['available'] for s in r['reasons'])),
    unavailable_frames=[r['frame'] for r in rows if not r['available']],samples=[r for r in rows if r['frame']%10==0])
dump(ROOT/'inspection'/'111143_early_audit.json',result);print(json.dumps(result,indent=2))
