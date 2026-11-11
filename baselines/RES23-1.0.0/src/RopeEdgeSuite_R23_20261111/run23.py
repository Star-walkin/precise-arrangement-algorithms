import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from common23 import *
from fusion23 import analyze

def main():
    summary=[]
    for item in plan():
        id=item['id'];rows=read(R22/'delivery'/id/'frames.json');near=read(R22/'events'/f'{id}.json')['events']
        receipt=read(R22/'rotation'/id/'receipt.json')
        assert sha(receipt['features_path'])==receipt['features_sha256']
        features=np.load(receipt['features_path'])['features']
        result=analyze(rows,near,features,item['fps'])
        dump(ROOT/'events'/f'{id}.json',result)
        out=[dict(r,fusion23=f) for r,f in zip(rows,result['frames'])]
        dump(ROOT/'delivery'/id/'frames.json',out)
        s=dict(id=id,frames=len(rows),fps=item['fps'],source=item['source'],source_sha256=item['source_sha256'],
            baseline_video=str(R22/'delivery'/id/f'RES22_{id}_full.mp4'),
            video=str(ROOT/'delivery'/id/f'RES23_{id}_full.mp4'),events=result['events'])
        dump(ROOT/'delivery'/id/'summary.json',s);summary.append(s)
        print(id,[dict(count=e['count_period'].get('period_s'),texture=e['texture_period'].get('period_s'),
            timeline=e['timeline'],candidate=e['candidate_s'],confirmation=e['confirmation_s'],score=e.get('recurrence_score')) for e in result['events']],flush=True)
    dump(ROOT/'analysis_receipt.json',dict(videos=summary,code_hashes={f:sha(ROOT/f) for f in ['common23.py','period23.py','fusion23.py','run23.py']}))

if __name__=='__main__':main()
