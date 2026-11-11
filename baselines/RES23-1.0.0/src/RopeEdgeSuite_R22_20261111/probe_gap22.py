from common22 import *
for id in ['dual_20260923_101517__right','dual_20260924_111500__left']:
    path=ROOT/'inputs'/id/'frames.json'
    if not path.exists():continue
    rows=read(path)
    idx=range(420,510,5) if '101517' in id else range(0,201,10)
    print(id,[dict(t=rows[i]['time_s'],N=rows[i]['N'],motion=rows[i].get('motion_distance'),pitch=rows[i].get('pitch')) for i in idx])
