from pathlib import Path
import sys,numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.append(str(ROOT.parent/'RopeEdgeSuite_R22_20261111'))
from common22 import read,dump,plan
from probe23 import slope
out=[]
for item in plan():
    rows=read(ROOT.parent/'RopeEdgeSuite_R22_20261111'/'delivery'/item['id']/'frames.json')
    events=read(ROOT.parent/'RopeEdgeSuite_R22_20261111'/'events'/f"{item['id']}.json")['events']
    for e in events:
        t0=e['entry_confirmed_s'];result=[]
        for lower in [2,3,4,5]:
            for w in [12,16,24,30]:
                rr=[r for r in rows if t0-w<=r['time_s']<t0-.5 and r['available'] and r['N']>lower
                    and r['current_coil_side_axis'] and not r['boundary_occluded'] and r['state22']['evidence_mode']=='current'
                    and r['N_interval'][1]-r['N_interval'][0]<=4]
                s=slope(rr)
                if s and s['span']>=6:result.append(dict(lower=lower,w=w,**s))
        out.append(dict(id=item['id'],t0=t0,fits=result));print(item['id'],[(r['lower'],r['w'],round(r['T'],3),r['support']) for r in result])
dump(ROOT/'inspection/qualified_windows.json',out)
