import collections
from common22 import *
from events22 import analyze
def main():
    results=[]
    for item in plan():
        rows=read(ROOT/'inputs'/item['id']/'frames.json');new=analyze(rows,item['fps'])
        dump(ROOT/'events'/f"{item['id']}.json",new)
        old=[r['near_event']['state'] for r in rows];states=[r['state'] for r in new['frames']]
        summary=dict(id=item['id'],old_transitions=sum(a!=b for a,b in zip(old,old[1:])),
            new_transitions=sum(a!=b for a,b in zip(states,states[1:])),states=dict(collections.Counter(states)),
            first20=dict(collections.Counter(states[:200])),current=sum(r['current_evidence'] for r in new['frames']),
            holds=sum(r['evidence_mode']=='history_hold' for r in new['frames']),
            rejected=sum(r['rejection_reason']=='isolated_temporal_outlier' for r in new['frames']),
            transitions=[dict(t=rows[i]['time_s'],state=states[i],N=rows[i]['N'],trend=new['frames'][i].get('trend')) for i in range(1,len(states)) if states[i]!=states[i-1]])
        results.append(summary);print({k:v for k,v in summary.items() if k!='transitions'})
    dump(ROOT/'inspection/state_trial_B.json',results)
if __name__=='__main__':main()
