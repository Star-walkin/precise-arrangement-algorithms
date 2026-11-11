from common21 import *
from row_overlap21 import observe,summarize
id='dual_20260924_111500__left';rows=[]
for i in range(210,311):
    r=count20(id,i);rows.append(dict(frame=i,time_s=i/10,evidence=observe(r['measurement'],r['axis'])))
result=dict(candidates=summarize(rows,10),frames=rows);dump(ROOT/'inspection'/'111500_row_overlap_trial.json',result)
print(result['candidates']);print([dict(t=r['time_s'],**r['evidence']) for r in rows if 24<=r['time_s']<=25 and r['evidence']['eligible']])
