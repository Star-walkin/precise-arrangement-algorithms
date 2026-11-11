from common22 import *
from rotation22 import estimate_period
from motor22 import analyze
def main():
    results=[]
    for item in plan():
        receipt=ROOT/'rotation'/item['id']/'receipt.json'
        ip=ROOT/'inputs'/item['id']/'frames.json'
        if not receipt.exists() or not ip.exists():continue
        rr=read(receipt)
        if not rr['regions']:continue
        features=np.load(rr['features_path'])['features']
        if features.ndim!=3 or features.shape[1]<2:continue
        rows=read(ip);overlap=read(R21/'row_overlap'/f"{item['id']}.json")
        motion=read(ROOT/'events'/f"{item['id']}.json")
        motor=analyze(rows,overlap,motion,features,item['fps'])
        dump(ROOT/'motor'/f"{item['id']}.json",motor)
        periods=[dict(t=t/10,period=estimate_period(features[:t+1],10)) for t in [min(200,len(rows)-1),min(250,len(rows)-1)]]
        summary=dict(id=item['id'],periods=periods,events=motor['events'])
        results.append(summary)
        print(item['id'],[(x['t'],x['period'].get('period_s'),x['period'].get('region_support')) for x in periods],
            [dict(target=e['target'],recurrence_s=e['recurrence_s'],confirmation_s=e['confirmation_s'],pose=e.get('pose_corroboration_s')) for e in motor['events']],flush=True)
    dump(ROOT/'inspection/motor_trial_A.json',results)
if __name__=='__main__':main()
