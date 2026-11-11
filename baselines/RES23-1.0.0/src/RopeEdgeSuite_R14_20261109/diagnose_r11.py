from collections import Counter
from common import *
def main():
    a=read(RUN11/'compact.json');b=read(RUN12/'compact.json')
    stats={}
    for name,rows in [('R11',a),('SG2',b)]:
        q=[r for r in rows if r['frame']>=310]
        stats[name]=dict(after_s=31,frames=len(q),track_pass=sum(r['s1_valid'] for r in q),
            region_pass=sum(r['region_valid'] for r in q),ecw_pass=sum(r['ecw_valid'] for r in q))
    examples=[]
    for i in [310,316,336,350,370,390,412,450,599]:
        r=record(i,RUN11);n=record(i)
        t=r['S1_tracker_full'];d=t.get('diagnostics',{})
        examples.append(dict(frame=i,old_track=t['valid'],new_track=n['S1_tracker_full']['valid'],
            old_track_diagnostics=d,old_ecw={k:v for k,v in r['ECW'].items() if k not in ['boundary','rope']},
            old_ecw_components={k:{kk:vv for kk,vv in r['ECW'].get(k,{}).items() if not isinstance(vv,(list,dict))} for k in ['boundary','rope']},
            original_boundary_at_d=r['region']['edge_y']))
    dump(ROOT/'R11_diagnosis.json',dict(stats=stats,examples=examples))
    print(json.dumps(stats,ensure_ascii=False))
    for r in examples:
        print(r['frame'],r['old_track'],r['new_track'],json.dumps(r['old_ecw_components'],ensure_ascii=False))
if __name__=='__main__':main()
