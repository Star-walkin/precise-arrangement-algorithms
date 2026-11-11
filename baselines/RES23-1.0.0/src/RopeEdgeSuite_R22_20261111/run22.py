import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import collections
from common22 import *
from events22 import analyze as motion_analyze
from motor22 import analyze as motor_analyze

def main():
    summaries=[]
    for item in plan():
        id=item['id'];rows=read(ROOT/'inputs'/id/'frames.json')
        motion=motion_analyze(rows,item['fps'])
        overlap=read(R21/'row_overlap'/f'{id}.json')
        rr=read(ROOT/'rotation'/id/'receipt.json')
        assert sha(rr['features_path'])==rr['features_sha256']
        features=np.load(rr['features_path'])['features']
        motor=motor_analyze(rows,overlap,motion,features,item['fps'])
        dump(ROOT/'events'/f'{id}.json',motion);dump(ROOT/'motor'/f'{id}.json',motor)
        result=[dict(r,state22=s,motor22=m) for r,s,m in zip(rows,motion['frames'],motor['frames'])]
        old=[r['near_event']['state'] for r in rows];new=[r['state'] for r in motion['frames']]
        summary=dict(id=id,frames=len(result),fps=item['fps'],source=item['source'],source_sha256=item['source_sha256'],
            video=str(ROOT/'delivery'/id/f'RES22_{id}_full.mp4'),baseline_video=str(R21/'delivery'/id/f'RES21_{id}_full.mp4'),
            old_state_changes=sum(a!=b for a,b in zip(old,old[1:])),new_state_changes=sum(a!=b for a,b in zip(new,new[1:])),
            states=dict(collections.Counter(new)),first20_states=dict(collections.Counter(new[:200])),
            history_hold_frames=sum(s['evidence_mode']=='history_hold' for s in motion['frames']),
            rejected_outliers=sum(s['rejection_reason']=='isolated_temporal_outlier' for s in motion['frames']),
            near_events=motion['events'],motor_events=motor['events'],rotation_regions=len(rr['regions']),
            available_distance_frames=sum(r['available'] for r in rows),current_F3_calls=sum(r['current_F3_called'] for r in rows),
            physical_layer_change_confirmed=False,physical_contact_confirmed=False,physical_revolution_confirmed=False)
        dump(ROOT/'delivery'/id/'frames.json',result);dump(ROOT/'delivery'/id/'summary.json',summary);summaries.append(summary)
        print(id,summary['old_state_changes'],summary['new_state_changes'],len(motor['events']),flush=True)
    dump(ROOT/'analysis_receipt.json',dict(videos=summaries,algorithm='RopeEdgeSuite R22 / MTD1 + MFT1',
        code_hashes={f:sha(ROOT/f) for f in ['run22.py','events22.py','motor22.py','rotation22.py','common22.py']}))

if __name__=='__main__':main()
